"""Query rewriting for better retrieval recall."""
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

REWRITE_TEMPLATE = "Rewrite this query to be more retrieval-friendly. Return only the rewritten query.\n\nOriginal: {query}\nRewritten:"

class QueryRewriter:
    def __init__(self, llm):
        self.chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(REWRITE_TEMPLATE))

    def rewrite(self, query: str) -> str:
        try:
            return self.chain.run(query=query).strip()
        except Exception:
            return query
