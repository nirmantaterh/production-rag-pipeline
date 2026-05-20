"""
Tests run without ragas installed — score_faithfulness falls back to
_sentence_overlap, which is tested directly here.
"""
import pytest
from app.evaluator import RAGEvaluator


@pytest.fixture
def ev():
    return RAGEvaluator.__new__(RAGEvaluator)


# --- fallback path (ragas not installed) ---

def test_empty_answer_returns_one(ev):
    assert ev.score_faithfulness("", "some context") == 1.0


def test_fully_supported_sentence(ev):
    score = ev.score_faithfulness(
        "The sky is blue.",
        "The sky is blue and clear today.",
    )
    assert score == 1.0


def test_unsupported_sentence(ev):
    score = ev.score_faithfulness(
        "Penguins swim underwater.",
        "Elephants roam savanna grasslands.",
    )
    assert score == 0.0


def test_partial_support(ev):
    score = ev.score_faithfulness(
        "The sky is blue. Penguins swim underwater.",
        "The sky is blue and clear.",
    )
    assert 0.0 < score < 1.0


def test_score_is_rounded_to_4dp(ev):
    score = ev.score_faithfulness("a b. c d. e f.", "a b c d e f")
    assert score == round(score, 4)


# --- signature ---

def test_accepts_query_kwarg(ev):
    score = ev.score_faithfulness(
        "The sky is blue.",
        "The sky is blue.",
        query="What colour is the sky?",
    )
    assert isinstance(score, float)


# --- sentence overlap unit tests ---

def test_sentence_overlap_empty(ev):
    assert ev._sentence_overlap("", "ctx") == 1.0


def test_sentence_overlap_full(ev):
    assert ev._sentence_overlap("sky is blue.", "sky is blue") == 1.0
