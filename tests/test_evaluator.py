import pytest
from app.evaluator import RAGEvaluator


@pytest.fixture
def ev():
    return RAGEvaluator.__new__(RAGEvaluator)


def test_empty_answer_returns_one(ev):
    assert ev.score_faithfulness("", "some context") == 1.0


def test_fully_supported_sentence(ev):
    score = ev.score_faithfulness(
        "The sky is blue.",
        "The sky is blue and clear today.",
    )
    assert score == 1.0


def test_unsupported_sentence(ev):
    # No word in the answer appears as a substring in the context
    score = ev.score_faithfulness(
        "Penguins swim underwater.",
        "Elephants roam savanna grasslands.",
    )
    assert score == 0.0


def test_partial_support(ev):
    score = ev.score_faithfulness(
        "The sky is blue. Cats fly to the moon.",
        "The sky is blue and clear.",
    )
    assert 0.0 < score < 1.0


def test_score_is_rounded_to_4dp(ev):
    score = ev.score_faithfulness(
        "a b. c d. e f.",
        "a b c d e f",
    )
    assert score == round(score, 4)
