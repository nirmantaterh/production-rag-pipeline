from unittest.mock import MagicMock, patch
from app.reranker import ColBERTReranker


def _reranker():
    rr = ColBERTReranker.__new__(ColBERTReranker)
    rr.model = MagicMock()
    return rr


def test_empty_docs_returns_empty():
    rr = _reranker()
    assert rr.rerank("query", []) == []


def test_returns_top_k_results():
    rr = _reranker()
    docs = [{"text": f"doc {i}"} for i in range(10)]
    rr.model.rerank.return_value = [
        {"content": f"doc {i}", "score": 1.0 - i * 0.1} for i in range(5)
    ]
    results = rr.rerank("query", docs, top_k=5)
    assert len(results) == 5


def test_colbert_score_attached():
    rr = _reranker()
    docs = [{"text": "hello world", "source": "a.txt"}]
    rr.model.rerank.return_value = [{"content": "hello world", "score": 0.92}]
    results = rr.rerank("hello", docs, top_k=1)
    assert results[0]["colbert_score"] == 0.92


def test_original_fields_preserved():
    rr = _reranker()
    docs = [{"text": "hello world", "source": "a.txt", "score": 0.7}]
    rr.model.rerank.return_value = [{"content": "hello world", "score": 0.95}]
    results = rr.rerank("hello", docs)
    assert results[0]["source"] == "a.txt"
