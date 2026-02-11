"""RAG context management for contextkit.

Provides retrieval-augmented context assembly with pluggable
retriever backends, chunk ranking, and budget-aware retrieval.

Backends (require optional dependencies)::

    pip install contextkit[qdrant]    # QdrantRetriever
    pip install contextkit[chroma]    # ChromaRetriever
    pip install contextkit[pgvector]  # PgvectorRetriever
    pip install contextkit[pinecone]  # PineconeRetriever
"""

from contextkit.rag.chunk import Chunk, RetrieverBackend
from contextkit.rag.context import RAGContext

__all__ = [
    "Chunk",
    "ChromaRetriever",
    "PgvectorRetriever",
    "PineconeRetriever",
    "QdrantRetriever",
    "RAGContext",
    "RetrieverBackend",
]


def __getattr__(name: str) -> type:
    """Lazy-import retriever backends to avoid hard dependencies."""
    if name == "QdrantRetriever":
        from contextkit.rag.qdrant_retriever import QdrantRetriever

        return QdrantRetriever
    if name == "ChromaRetriever":
        from contextkit.rag.chroma_retriever import ChromaRetriever

        return ChromaRetriever
    if name == "PgvectorRetriever":
        from contextkit.rag.pgvector_retriever import PgvectorRetriever

        return PgvectorRetriever
    if name == "PineconeRetriever":
        from contextkit.rag.pinecone_retriever import PineconeRetriever

        return PineconeRetriever
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
