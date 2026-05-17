"""Core RAG pipeline with hallucination reduction and tool-calling."""
from __future__ import annotations
from dataclasses import dataclass
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from .retriever import FAISSRetriever
from .rewriter import QueryRewriter
from .evaluator import RAGEvaluator

SYSTEM_PROMPT = """You are a helpful assistant. Answer ONLY using the provided context.
If the context does not contain enough information, say so.

Context:
{context}
"""

@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
    faithfulness_score: float
    latency_ms: float

class RAGPipeline:
    def __init__(self, model_name: str = "gpt-4o-mini", top_k: int = 5,
                 rerank: bool = True, mlflow_experiment: str = "rag-production"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        self.retriever = FAISSRetriever(top_k=top_k, rerank=rerank)
        self.rewriter = QueryRewriter(self.llm)
        self.evaluator = RAGEvaluator(experiment_name=mlflow_experiment)
        self.chain = LLMChain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "{question}"),
            ]),
        )

    def run(self, query: str, user_id: str | None = None) -> RAGResponse:
        import time
        start = time.perf_counter()
        rewritten = self.rewriter.rewrite(query)
        docs = self.retriever.retrieve(rewritten)
        context = "\n\n".join(d.page_content for d in docs)
        sources = [{"source": d.metadata.get("source", ""), "score": d.metadata.get("score", 0)} for d in docs]
        answer = self.chain.run(question=query, context=context)
        latency_ms = (time.perf_counter() - start) * 1000
        faithfulness = self.evaluator.score_faithfulness(answer, context)
        self.evaluator.log(query=query, answer=answer, context=context,
                           faithfulness=faithfulness, latency_ms=latency_ms)
        return RAGResponse(answer=answer, sources=sources,
                           faithfulness_score=faithfulness, latency_ms=latency_ms)
