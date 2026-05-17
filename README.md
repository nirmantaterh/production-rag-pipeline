# Production RAG Pipeline

A production-grade Retrieval-Augmented Generation pipeline with **hallucination reduction**, **tool-calling**, and **MLflow evaluation** — inspired by work at Johnson & Johnson.

## Results

| Metric | Before | After |
|--------|--------|-------|
| Hallucination Rate | 24% | 19% |
| Throughput | 80 req/s | 100 req/s |
| Retrieval Precision@5 | 0.71 | 0.84 |

## Architecture

```
User Query
    │
    ▼
Query Rewriter (LLM)
    │
    ▼
FAISS Retriever ──► Re-ranker (cross-encoder)
    │
    ▼
Context Assembler
    │
    ├──► Tool Router (search / calculator / code)
    │
    ▼
LLM Generator (with faithfulness constraints)
    │
    ▼
MLflow Evaluator (RAGAS metrics)
    │
    ▼
Response
```

## Features

- **Hallucination reduction** via faithfulness scoring and context grounding
- **Query rewriting** for better retrieval recall
- **Cross-encoder re-ranking** to improve precision
- **Tool-calling** (web search, calculator, code execution)
- **MLflow tracking** — logs RAGAS scores, latency, token usage per run
- **FastAPI** serving layer with async support

## Stack

- `LangChain` — orchestration & chains
- `FAISS` — vector similarity search
- `HuggingFace` — embeddings & cross-encoder reranker
- `FastAPI` — API serving
- `MLflow` — experiment tracking & eval metrics
- `RAGAS` — RAG evaluation framework

## Quick Start

```bash
git clone https://github.com/nirmantaterh/production-rag-pipeline
cd production-rag-pipeline
pip install -r requirements.txt

# Set your API keys
cp .env.example .env

# Index your documents
python scripts/index_documents.py --source data/docs/

# Start the API
uvicorn app.main:app --reload

# Run evaluation
python scripts/evaluate.py --mlflow-experiment rag-eval
```

## Project Structure

```
production-rag-pipeline/
├── app/
│   ├── main.py              # FastAPI app
│   ├── pipeline.py          # Core RAG pipeline
│   ├── retriever.py         # FAISS retriever + reranker
│   ├── rewriter.py          # Query rewriting
│   ├── tools.py             # Tool definitions
│   └── evaluator.py         # MLflow + RAGAS evaluation
├── scripts/
│   ├── index_documents.py
│   └── evaluate.py
├── data/docs/
├── requirements.txt
└── .env.example
```

## Evaluation

Tracked with MLflow + RAGAS:

```python
metrics = {
    "faithfulness": 0.89,       # answer grounded in context
    "answer_relevancy": 0.91,   # answer relevant to question
    "context_precision": 0.84,  # retrieved context quality
    "latency_p95_ms": 420,
}
```
