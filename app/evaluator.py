"""MLflow RAG evaluation with RAGAS-style faithfulness scoring."""
from __future__ import annotations
import mlflow


class RAGEvaluator:
    def __init__(self, experiment_name: str = "rag-production"):
        mlflow.set_experiment(experiment_name)

    def score_faithfulness(self, answer: str, context: str) -> float:
        """Sentence-level faithfulness: fraction of answer sentences supported by context."""
        sentences = [s.strip() for s in answer.split(".") if s.strip()]
        if not sentences:
            return 1.0
        supported = sum(1 for s in sentences if any(w in context for w in s.split()[:6]))
        return round(supported / len(sentences), 4)

    def log(self, query: str, answer: str, context: str, faithfulness: float, latency_ms: float):
        with mlflow.start_run(nested=True):
            mlflow.log_param("query_length", len(query))
            mlflow.log_metric("faithfulness", faithfulness)
            mlflow.log_metric("latency_ms", latency_ms)
            mlflow.log_metric("answer_length", len(answer))
            mlflow.log_metric("context_length", len(context))
