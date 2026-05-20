from unittest.mock import MagicMock
from app.rewriter import QueryRewriter


def _rewriter():
    rw = QueryRewriter(MagicMock())
    rw.chain = MagicMock()
    return rw


def test_returns_stripped_result():
    rw = _rewriter()
    rw.chain.invoke.return_value = "  better retrieval query  "
    assert rw.rewrite("my query") == "better retrieval query"


def test_falls_back_to_original_on_exception():
    rw = _rewriter()
    rw.chain.invoke.side_effect = RuntimeError("LLM timeout")
    assert rw.rewrite("original query") == "original query"


def test_passes_query_to_chain():
    rw = _rewriter()
    rw.chain.invoke.return_value = "rewritten"
    rw.rewrite("test input")
    rw.chain.invoke.assert_called_once_with({"query": "test input"})
