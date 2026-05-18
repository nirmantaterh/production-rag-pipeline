"""FastAPI serving layer — wraps LangGraph agentic RAG pipeline."""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from .retriever import HybridRetriever
from .reranker import ColBERTReranker
from .evaluator import RAGEvaluator
from .graph import build_rag_graph

app = FastAPI(title="Production RAG Pipeline", version="2.0.0")

retriever = HybridRetriever(qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"))
reranker = ColBERTReranker()
evaluator = RAGEvaluator()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
rag_graph = build_rag_graph(retriever, reranker, llm, evaluator)


class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    faithfulness_score: float
    latency_ms: float
    sources: list[dict]


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    try:
        result = rag_graph.invoke({"query": req.query})
        return QueryResponse(
            answer=result["answer"],
            faithfulness_score=result.get("faithfulness", 0.0),
            latency_ms=result.get("latency_ms", 0.0),
            sources=[{"source": d.get("source", ""), "score": d.get("score", 0)} for d in result.get("docs", [])],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
