"""
ColBERT token-level late-interaction reranking via RAGatouille.
10-100x faster than cross-encoders at near cross-encoder accuracy.
"""
from __future__ import annotations
from ragatouille import RAGPretrainedModel


class ColBERTReranker:
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        self.model = RAGPretrainedModel.from_pretrained(model_name)

    def rerank(self, query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
        if not docs:
            return []
        texts = [d["text"] for d in docs]
        results = self.model.rerank(query=query, documents=texts, k=top_k)
        reranked = []
        for r in results:
            original = next((d for d in docs if d["text"] == r["content"]), {})
            reranked.append({**original, "colbert_score": r["score"]})
        return reranked
