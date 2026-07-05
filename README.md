# Production RAG Pipeline

Agentic RAG pipeline built with **LangGraph**, **BGE-M3 hybrid search**, **Qdrant**, and **ColBERT reranking**. Runs fully local (no OpenAI required) or with any OpenAI-compatible endpoint.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange)
![Qdrant](https://img.shields.io/badge/Qdrant-vector%20db-red)
![Docker](https://img.shields.io/badge/Docker-compose-blue?logo=docker&logoColor=white)

---

## Architecture:

```
User Query
    │
    ▼
LangGraph Orchestrator
    │
    ├──► Route: RAG / Tool / Direct
    │
    ├──► [RAG Path]
    │       ├─ Query Rewriting          (LLM-based — improves sparse + dense recall)
    │       ├─ BGE-M3 Hybrid Retrieval  (dense + sparse vectors, Qdrant RRF fusion)
    │       ├─ ColBERT Reranking        (late interaction, 10-100x faster than cross-encoder)
    │       └─ Faithfulness Gate        (retry generation if score < 0.5, max 2 retries)
    │
    └──► LLM Generator → MLflow Evaluator (faithfulness, latency, context length)
```

---

## Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Orchestration | **LangGraph** | Stateful graph with conditional retries and routing |
| Embeddings | **BGE-M3** | Dense + sparse vectors in one model pass, fused via RRF |
| Vector DB | **Qdrant** | Native hybrid search with RRF fusion built-in |
| Reranking | **ColBERT** via RAGatouille | Late interaction — faster than cross-encoders at inference |
| Query rewriting | **LLM + LCEL** | Improves retrieval recall on ambiguous or short queries |
| LLM | **GPT-4o-mini** (swappable) | Grounded generation; swap via `.env` |
| Evaluation | **MLflow** | Tracks faithfulness, latency, answer length per run |
| Serving | **FastAPI** | Async, single endpoint |

---

## Key Design Decisions

**BGE-M3 over sentence-transformers**: BGE-M3 produces dense + sparse (SPLADE-style) vectors in one forward pass. Qdrant fuses them via Reciprocal Rank Fusion — no separate BM25 index to maintain.

**ColBERT over cross-encoder reranking**: Cross-encoders rerun the full model per (query, doc) pair at inference. ColBERT precomputes document token embeddings at index time and uses late interaction (MaxSim) at query time — same reranking quality, fraction of the latency.

**LangGraph over LangChain chains**: Explicit state machine with conditional edges. Retries (faithfulness gate), fallback routing, and observability are graph edges — not hidden chain behavior.

---

## Retrieval Evaluation

Retrieval quality measured on [BEIR SciFact](https://github.com/beir-cellar/beir) (test qrels), comparing the pipeline's actual retrieval stages against each other. 1000-document corpus (all gold documents + random fillers), 150 queries, fixed seed. Full methodology in [`scripts/evaluate_retrieval.py`](scripts/evaluate_retrieval.py); raw output in [`eval/results_scifact.json`](eval/results_scifact.json).

| Configuration | Recall@10 | MRR@10 | nDCG@10 |
|---------------|-----------|--------|---------|
| Dense only (BGE-M3) | 0.877 | 0.759 | 0.787 |
| Sparse only (BGE-M3 lexical) | 0.875 | 0.766 | 0.786 |
| Hybrid RRF (dense + sparse) | **0.916** | 0.787 | 0.816 |
| Hybrid + multi-vector rerank | **0.916** | **0.801** | **0.825** |

Hybrid fusion recovers cases where dense and sparse retrieval individually miss the gold document (+4pp recall over either alone). Reranking with BGE-M3's multi-vector (ColBERT-style late interaction) output doesn't change what's retrieved but improves how it's ordered — MRR and nDCG both increase at fixed recall, meaning the correct document surfaces higher and more consistently.

Reproduce:
```bash
pip install FlagEmbedding qdrant-client datasets
python scripts/evaluate_retrieval.py --corpus-size 1000 --max-queries 150
```

## Quick Start

```bash
git clone https://github.com/nirmantaterh/production-rag-pipeline
cd production-rag-pipeline

# Full stack (app + Qdrant + MLflow)
cp .env.example .env  # add OPENAI_API_KEY
docker compose up --build

# Or run locally
pip install -r requirements.txt
docker compose up qdrant mlflow -d
python scripts/index_documents.py --source data/docs/
uvicorn app.main:app --reload
```

---

## API

```
POST /query
{"query": "What is RAG?"}
```

---

## Project Structure

```
production-rag-pipeline/
├── app/
│   ├── graph.py        # LangGraph RAG graph (routing + retry logic)
│   ├── retriever.py    # BGE-M3 + Qdrant hybrid retrieval (dense + sparse, RRF)
│   ├── reranker.py     # ColBERT reranking via RAGatouille
│   ├── rewriter.py     # LLM query rewriting (LCEL)
│   ├── evaluator.py    # MLflow experiment tracking
│   └── main.py         # FastAPI serving layer
├── scripts/
│   └── index_documents.py   # Chunk + embed + upsert to Qdrant
├── tests/
│   ├── conftest.py          # Stubs for heavy deps (runs without full install)
│   ├── test_evaluator.py
│   ├── test_rewriter.py
│   ├── test_reranker.py
│   └── test_chunking.py
├── data/docs/               # Drop .txt/.pdf files here to index
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Running Tests

```bash
pip install pytest
pytest tests/
```

All tests stub heavy dependencies (BGE-M3, Qdrant, ColBERT, LangGraph) — run without a full install.
