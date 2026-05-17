"""FastAPI serving layer."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .pipeline import RAGPipeline

app = FastAPI(title="Production RAG Pipeline", version="1.0.0")
pipeline = RAGPipeline()

class QueryRequest(BaseModel):
    query: str
    user_id: str | None = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    faithfulness_score: float
    latency_ms: float

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    try:
        r = pipeline.run(req.query, user_id=req.user_id)
        return QueryResponse(answer=r.answer, sources=r.sources,
                             faithfulness_score=r.faithfulness_score, latency_ms=r.latency_ms)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
