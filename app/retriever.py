"""FAISS-based retriever with cross-encoder reranking."""
from __future__ import annotations
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from sentence_transformers import CrossEncoder

class FAISSRetriever:
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, index_path: str = "data/faiss_index", top_k: int = 5, rerank: bool = True):
        self.top_k = top_k
        self.rerank = rerank
        self.embeddings = HuggingFaceEmbeddings(model_name=self.EMBED_MODEL)
        self.reranker = CrossEncoder(self.RERANK_MODEL) if rerank else None
        self._load_index(index_path)

    def _load_index(self, path: str):
        try:
            self.vectorstore = FAISS.load_local(path, self.embeddings)
        except Exception:
            self.vectorstore = None

    def retrieve(self, query: str) -> list[Document]:
        if self.vectorstore is None:
            return []
        k = self.top_k * 3 if self.rerank else self.top_k
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        if not self.rerank or self.reranker is None:
            return [d for d, _ in docs_and_scores[:self.top_k]]
        pairs = [(query, doc.page_content) for doc, _ in docs_and_scores]
        rerank_scores = self.reranker.predict(pairs)
        ranked = sorted(zip(docs_and_scores, rerank_scores), key=lambda x: x[1], reverse=True)
        results = []
        for (doc, _), score in ranked[:self.top_k]:
            doc.metadata["score"] = float(score)
            results.append(doc)
        return results

    @classmethod
    def build_index(cls, documents: list[Document], save_path: str = "data/faiss_index"):
        embeddings = HuggingFaceEmbeddings(model_name=cls.EMBED_MODEL)
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(save_path)
        return vectorstore
