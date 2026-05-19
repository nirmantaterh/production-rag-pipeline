"""
Agentic RAG graph built with LangGraph.
Stateful cyclic flow: route → rewrite → retrieve → rerank → generate → evaluate → (retry if needed).
"""
from __future__ import annotations
from typing import Literal, NotRequired, TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END


class RAGState(TypedDict):
    query: str
    route: Literal["rag", "tool", "direct"]
    rewritten_query: NotRequired[str]
    docs: list[dict]
    context: str
    answer: str
    faithfulness: float
    latency_ms: float
    iteration: int


def build_rag_graph(retriever, reranker, llm, evaluator, rewriter=None):
    """Compile the LangGraph agentic RAG pipeline."""

    def analyze_query(state: RAGState) -> RAGState:
        q = state["query"].lower()
        if any(kw in q for kw in ["calculate", "compute", "+", "-", "*", "/"]):
            route = "tool"
        elif any(kw in q for kw in ["hi", "hello", "thanks", "who are you"]):
            route = "direct"
        else:
            route = "rag"
        return {**state, "route": route, "iteration": 0}

    def rewrite_query(state: RAGState) -> RAGState:
        rewritten = rewriter.rewrite(state["query"]) if rewriter else state["query"]
        return {**state, "rewritten_query": rewritten}

    def retrieve(state: RAGState) -> RAGState:
        q = state.get("rewritten_query") or state["query"]
        docs = retriever.retrieve(q)
        return {**state, "docs": docs}

    def rerank_docs(state: RAGState) -> RAGState:
        reranked = reranker.rerank(state["query"], state["docs"])
        context = "\n\n".join(d["text"] for d in reranked[:5])
        return {**state, "docs": reranked, "context": context}

    def generate(state: RAGState) -> RAGState:
        import time
        start = time.perf_counter()
        prompt = f"""Answer using ONLY the context below. If the context is insufficient, say so — do not hallucinate.

Context:
{state.get("context", "")}

Question: {state["query"]}"""
        response = llm.invoke([HumanMessage(content=prompt)])
        return {**state, "answer": response.content,
                "latency_ms": (time.perf_counter() - start) * 1000,
                "iteration": state.get("iteration", 0) + 1}

    def evaluate(state: RAGState) -> RAGState:
        score = evaluator.score_faithfulness(state["answer"], state.get("context", ""))
        evaluator.log(query=state["query"], answer=state["answer"],
                      context=state.get("context", ""),
                      faithfulness=score, latency_ms=state.get("latency_ms", 0))
        return {**state, "faithfulness": score}

    def should_retry(state: RAGState) -> Literal["generate", "end"]:
        if state["faithfulness"] < 0.5 and state["iteration"] < 2:
            return "generate"
        return "end"

    def handle_tool(state: RAGState) -> RAGState:
        return {**state, "answer": f"[Tool: {state['query']}]", "faithfulness": 1.0}

    def handle_direct(state: RAGState) -> RAGState:
        r = llm.invoke([HumanMessage(content=state["query"])])
        return {**state, "answer": r.content, "faithfulness": 1.0}

    def route_query(state: RAGState) -> str:
        return {"rag": "rewrite", "tool": "tool", "direct": "direct"}[state["route"]]

    g = StateGraph(RAGState)
    g.add_node("analyze", analyze_query)
    g.add_node("rewrite", rewrite_query)
    g.add_node("retrieve", retrieve)
    g.add_node("rerank", rerank_docs)
    g.add_node("generate", generate)
    g.add_node("evaluate", evaluate)
    g.add_node("tool", handle_tool)
    g.add_node("direct", handle_direct)

    g.set_entry_point("analyze")
    g.add_conditional_edges("analyze", route_query)
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "generate")
    g.add_edge("generate", "evaluate")
    g.add_conditional_edges("evaluate", should_retry, {"generate": "generate", "end": END})
    g.add_edge("tool", END)
    g.add_edge("direct", END)

    return g.compile()
