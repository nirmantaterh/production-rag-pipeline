"""Stub all heavy external deps so tests run without installing them."""
import sys
from unittest.mock import MagicMock

for _mod in [
    "mlflow",
    "langchain_core",
    "langchain_core.messages",
    "langchain_core.prompts",
    "langchain_core.output_parsers",
    "langchain_openai",
    "langgraph",
    "langgraph.graph",
    "qdrant_client",
    "qdrant_client.models",
    "FlagEmbedding",
    "ragatouille",
    "fastapi",
    "pydantic",
]:
    sys.modules[_mod] = MagicMock()
