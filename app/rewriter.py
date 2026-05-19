"""Query rewriting for better retrieval recall."""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

REWRITE_TEMPLATE = (
    "Rewrite this query to be more retrieval-friendly. "
    "Return only the rewritten query.\n\n"
    "Original: {query}\nRewritten:"
)


class QueryRewriter:
    def __init__(self, llm):
        prompt = PromptTemplate.from_template(REWRITE_TEMPLATE)
        self.chain = prompt | llm | StrOutputParser()

    def rewrite(self, query: str) -> str:
        try:
            return self.chain.invoke({"query": query}).strip()
        except Exception:
            return query
