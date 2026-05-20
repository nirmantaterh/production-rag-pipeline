"""
BGE-M3 hybrid retrieval (dense + sparse) backed by Qdrant.
Uses dense + sparse vectors fused via RRF. BGE-M3 also supports multi-vector but that is not wired here.
"""
from __future__ import annotations
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, PointStruct
from FlagEmbedding import BGEM3FlagModel

COLLECTION = "documents"
DENSE_DIM = 1024
MODEL_ID = "BAAI/bge-m3"


class HybridRetriever:
    """Dense + sparse retrieval fused via Reciprocal Rank Fusion (RRF)."""

    def __init__(self, qdrant_url: str = "http://localhost:6333", top_k: int = 20):
        self.top_k = top_k
        self.client = QdrantClient(url=qdrant_url)
        self.model = BGEM3FlagModel(MODEL_ID, use_fp16=True)
        self._ensure_collection()

    def _ensure_collection(self):
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION not in existing:
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams()},
            )

    def index(self, documents: list[dict]):
        """Embed and upsert documents with BGE-M3 dense + sparse vectors."""
        texts = [d["text"] for d in documents]
        out = self.model.encode(texts, batch_size=12, return_dense=True, return_sparse=True)
        points = []
        for i, doc in enumerate(documents):
            lw = out["lexical_weights"][i]
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": out["dense_vecs"][i].tolist(),
                    "sparse": {"indices": list(lw.keys()), "values": list(lw.values())},
                },
                payload=doc,
            ))
        self.client.upsert(collection_name=COLLECTION, points=points)

    def retrieve(self, query: str) -> list[dict]:
        """Hybrid search with RRF fusion of dense + sparse results."""
        q_out = self.model.encode([query], return_dense=True, return_sparse=True)
        dense_vec = q_out["dense_vecs"][0].tolist()
        lw = q_out["lexical_weights"][0]
        results = self.client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                {"query": dense_vec, "using": "dense", "limit": self.top_k},
                {"query": {"indices": list(lw.keys()), "values": list(lw.values())},
                 "using": "sparse", "limit": self.top_k},
            ],
            query={"fusion": "rrf"},
            limit=self.top_k,
        )
        return [{"text": r.payload.get("text", ""), "score": r.score,
                 "source": r.payload.get("source", "")} for r in results.points]
