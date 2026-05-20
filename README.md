# Production RAG Pipeline

An agentic RAG system built on the 2026 stack: **LangGraph**, **BGE-M3 hybrid search**, **Qdrant**, and **ColBERT reranking**.

## Architecture

```
User Query
    │
    ▼
LangGraph Orchestrator
    │
    ├──► Route: RAG / Tool / Direct
    │
    ├──► [RAG]
    │       ├─ Query Rewriting          (LLM-based, improves sparse + dense recall)
    │       ├─ BGE-M3 Hybrid Retrieval  (dense + sparse vectors, Qdrant RRF fusion)
    │       ├─ ColBERT Token Reranking  (late interaction, 10-100x faster than cross-encoder)
    │       └─ Faithfulness Gate        (retry generation if score < 0.5, max 2x)
    │
    └──► LLM Generator → MLflow Evaluator (faithfulness, latency, context length)
```

## Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Orchestration | **LangGraph** | Stateful agentic loops, conditional retries |
| Embeddings | **BGE-M3** | Dense + sparse + multi-vector in one model pass |
| Vector DB | **Qdrant** | Native hybrid search, RRF fusion |
| Reranking | **ColBERT** via RAGatouille | Late interaction, faster than cross-encoders |
| Query rewriting | **LLM + LCEL** | Improves retrieval recall on ambiguous queries |
| LLM | **GPT-4o-mini** | Grounded generation |
| Evaluation | **MLflow** | Experiment tracking — faithfulness, latency, answer length |
| Serving | **FastAPI** | Async, production-ready |

## Quick Start

```bash
git clone https://github.com/nirmantaterh/production-rag-pipeline
cd production-rag-pipeline

# Start the full stack (app + Qdrant + MLflow)
cp .env.example .env  # add your OPENAI_API_KEY
docker compose up --build

# Or run locally
pip install -r requirements.txt
docker compose up qdrant mlflow -d

python scripts/index_documents.py --source data/docs/
uvicorn app.main:app --reload
```

## Project Structure

```
production-rag-pipeline/
├── app/
│   ├── graph.py        # LangGraph agentic flow
│   ├── retriever.py    # BGE-M3 + Qdrant hybrid retrieval (dense + sparse, RRF)
│   ├── reranker.py     # ColBERT reranking via RAGatouille
│   ├── rewriter.py     # LLM query rewriting (LCEL)
│   ├── evaluator.py    # MLflow experiment tracking
│   └── main.py         # FastAPI serving layer
├── scripts/
│   └── index_documents.py   # Chunk + embed + upsert to Qdrant
├── tests/
│   ├── conftest.py          # Stubs for heavy deps (runs without full install)
│   ├── test_evaluator.py
│   ├── test_rewriter.py
│   ├── test_reranker.py
│   └── test_chunking.py
├── data/docs/               # Drop .txt files here to index
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Running Tests

```bash
pip install pytest
pytest tests/
```

Tests stub all heavy dependencies (BGE-M3, Qdrant, ColBERT, LangGraph) so they run without a full install.

## API

```
POST /query   {"query": "What is RAG?"}
GET  /health
```
