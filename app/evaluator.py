"""MLflow experiment tracking with RAGAS faithfulness scoring."""
from __future__ import annotations
import mlflow


class RAGEvaluator:
    def __init__(self, experiment_name: str = "rag-production"):
        mlflow.set_experiment(experiment_name)

    def score_faithfulness(self, answer: str, context: str, query: str = "") -> float:
        """RAGAS faithfulness score. Falls back to sentence overlap if ragas is not installed."""
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness
            from datasets import Dataset
            data = {
                "question": [query or answer[:80]],
                "answer": [answer],
                "contexts": [[context]],
            }
            result = evaluate(Dataset.from_dict(data), metrics=[faithfulness])
            score = result["faithfulness"]
            return round(float(score) if not hasattr(score, "mean") else float(score.mean()), 4)
        except Exception:
            return self._sentence_overlap(answer, context)

    def _sentence_overlap(self, answer: str, context: str) -> float:
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
