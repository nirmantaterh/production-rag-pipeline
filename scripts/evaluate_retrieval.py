"""
Retrieval-quality evaluation on BEIR SciFact.

Measures the pipeline's retrieval stages against labeled relevance judgments:

  1. dense            BGE-M3 dense vectors only
  2. sparse           BGE-M3 lexical weights only
  3. hybrid (RRF)     dense + sparse fused via Reciprocal Rank Fusion (as in app/retriever.py)
  4. hybrid + rerank  hybrid top-20 reranked with BGE-M3 multi-vector (ColBERT-style
                      late interaction, maxsim), cut to top-10

Uses an in-memory Qdrant instance so no Docker is required. Reproducible:
fixed seed, corpus = all gold documents + random fillers up to --corpus-size.

Usage:
    python scripts/evaluate_retrieval.py --corpus-size 1000 --max-queries 150

Writes eval/results_scifact.json and prints a markdown table.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, PointStruct,
    Prefetch, SparseVector, FusionQuery, Fusion,
)
from FlagEmbedding import BGEM3FlagModel

COLLECTION = "eval_documents"
DENSE_DIM = 1024
MODEL_ID = "BAAI/bge-m3"
SEED = 42
K = 10          # metrics cutoff
RERANK_POOL = 20  # hybrid candidates fed to the reranker


def load_scifact(corpus_size: int, max_queries: int):
    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries = load_dataset("BeIR/scifact", "queries", split="queries")
    qrels = load_dataset("BeIR/scifact-qrels", split="test")

    gold = {}  # query_id -> set(doc_id)
    for row in qrels:
        gold.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))

    gold_doc_ids = set().union(*gold.values())
    all_doc_ids = [str(d["_id"]) for d in corpus]
    rng = random.Random(SEED)
    fillers = [d for d in all_doc_ids if d not in gold_doc_ids]
    rng.shuffle(fillers)
    keep = gold_doc_ids | set(fillers[: max(0, corpus_size - len(gold_doc_ids))])

    docs = [
        {"doc_id": str(d["_id"]), "text": (d["title"] + " " + d["text"]).strip()}
        for d in corpus
        if str(d["_id"]) in keep
    ]

    qids = sorted(gold.keys())
    rng.shuffle(qids)
    qids = qids[:max_queries]
    qtexts = {str(q["_id"]): q["text"] for q in queries}
    eval_queries = [{"query_id": qid, "text": qtexts[qid], "gold": gold[qid]} for qid in qids]
    return docs, eval_queries


def build_index(client: QdrantClient, model, docs):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    colbert_vecs = {}
    batch = 8
    for start in range(0, len(docs), batch):
        chunk = docs[start:start + batch]
        out = model.encode(
            [d["text"] for d in chunk],
            batch_size=batch, max_length=512,
            return_dense=True, return_sparse=True, return_colbert_vecs=True,
        )
        points = []
        for i, doc in enumerate(chunk):
            lw = out["lexical_weights"][i]
            points.append(PointStruct(
                id=start + i,
                vector={
                    "dense": out["dense_vecs"][i].tolist(),
                    "sparse": {"indices": [int(k) for k in lw.keys()],
                               "values": [float(v) for v in lw.values()]},
                },
                payload={"doc_id": doc["doc_id"]},
            ))
            colbert_vecs[doc["doc_id"]] = out["colbert_vecs"][i].astype(np.float16)
        client.upsert(collection_name=COLLECTION, points=points)
        done = min(start + batch, len(docs))
        if done % 96 < batch or done == len(docs):
            print(f"  indexed {done}/{len(docs)} docs", flush=True)
    return colbert_vecs


def search(client, q_out, mode: str, limit: int):
    dense_vec = q_out["dense_vecs"][0].tolist()
    lw = q_out["lexical_weights"][0]
    sparse_q = SparseVector(indices=[int(k) for k in lw.keys()],
                            values=[float(v) for v in lw.values()])
    if mode == "dense":
        res = client.query_points(COLLECTION, query=dense_vec, using="dense", limit=limit)
    elif mode == "sparse":
        res = client.query_points(COLLECTION, query=sparse_q, using="sparse", limit=limit)
    else:  # hybrid RRF, same fusion as app/retriever.py
        res = client.query_points(
            COLLECTION,
            prefetch=[
                Prefetch(query=dense_vec, using="dense", limit=limit),
                Prefetch(query=sparse_q, using="sparse", limit=limit),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
        )
    return [p.payload["doc_id"] for p in res.points]


def maxsim(q_vecs: np.ndarray, d_vecs: np.ndarray) -> float:
    """ColBERT-style late interaction: mean over query tokens of max dot product."""
    sim = q_vecs.astype(np.float32) @ d_vecs.astype(np.float32).T
    return float(sim.max(axis=1).mean())


def metrics_at_k(ranked: list[str], gold: set[str], k: int):
    hits = [1 if d in gold else 0 for d in ranked[:k]]
    recall = sum(hits) / len(gold)
    mrr = 0.0
    for rank, h in enumerate(hits, start=1):
        if h:
            mrr = 1.0 / rank
            break
    dcg = sum(h / math.log2(r + 1) for r, h in enumerate(hits, start=1))
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gold), k) + 1))
    ndcg = dcg / ideal if ideal else 0.0
    return recall, mrr, ndcg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-size", type=int, default=1000)
    parser.add_argument("--max-queries", type=int, default=150)
    args = parser.parse_args()

    print("Loading SciFact...", flush=True)
    docs, eval_queries = load_scifact(args.corpus_size, args.max_queries)
    print(f"  corpus: {len(docs)} docs · queries: {len(eval_queries)}", flush=True)

    print("Loading BGE-M3...", flush=True)
    model = BGEM3FlagModel(MODEL_ID, use_fp16=False)

    client = QdrantClient(":memory:")
    t0 = time.time()
    print("Indexing corpus (dense + sparse + colbert vecs)...", flush=True)
    colbert_vecs = build_index(client, model, docs)
    print(f"  index built in {time.time() - t0:.0f}s", flush=True)

    configs = ["dense", "sparse", "hybrid", "hybrid+rerank"]
    totals = {c: np.zeros(3) for c in configs}

    for qi, q in enumerate(eval_queries, start=1):
        q_out = model.encode(
            [q["text"]], max_length=128,
            return_dense=True, return_sparse=True, return_colbert_vecs=True,
        )
        for mode in ["dense", "sparse", "hybrid"]:
            ranked = search(client, q_out, mode, K)
            totals[mode] += metrics_at_k(ranked, q["gold"], K)

        pool = search(client, q_out, "hybrid", RERANK_POOL)
        q_vecs = q_out["colbert_vecs"][0]
        reranked = sorted(pool, key=lambda d: maxsim(q_vecs, colbert_vecs[d]), reverse=True)
        totals["hybrid+rerank"] += metrics_at_k(reranked[:K], q["gold"], K)

        if qi % 25 == 0 or qi == len(eval_queries):
            print(f"  evaluated {qi}/{len(eval_queries)} queries", flush=True)

    n = len(eval_queries)
    results = {
        "dataset": "BeIR/scifact (test qrels)",
        "corpus_size": len(docs),
        "num_queries": n,
        "k": K,
        "rerank_pool": RERANK_POOL,
        "model": MODEL_ID,
        "seed": SEED,
        "configs": {
            c: {"recall@10": round(v[0] / n, 4), "mrr@10": round(v[1] / n, 4), "ndcg@10": round(v[2] / n, 4)}
            for c, v in totals.items()
        },
    }

    out_path = Path(__file__).resolve().parent.parent / "eval" / "results_scifact.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}\n")

    print("| Configuration | Recall@10 | MRR@10 | nDCG@10 |")
    print("|---------------|-----------|--------|---------|")
    labels = {
        "dense": "Dense only (BGE-M3)",
        "sparse": "Sparse only (BGE-M3 lexical)",
        "hybrid": "Hybrid RRF (dense + sparse)",
        "hybrid+rerank": "Hybrid + multi-vector rerank",
    }
    for c in configs:
        m = results["configs"][c]
        print(f"| {labels[c]} | {m['recall@10']:.3f} | {m['mrr@10']:.3f} | {m['ndcg@10']:.3f} |")


if __name__ == "__main__":
    main()
