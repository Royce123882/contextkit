"""RAG context management for contextkit.

Provides retrieval-augmented context assembly with pluggable
retriever backends, chunk ranking, and budget-aware retrieval.
"""

from contextkit.rag.backends import Chunk, RetrieverBackend
from contextkit.rag.context import RAGContext

__all__ = ["RAGContext", "RetrieverBackend", "Chunk"]
