"""RAG retriever backend protocol.

Defines the RetrieverBackend protocol that all retriever
implementations must follow, along with the Chunk data model.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from contextkit.utils.text_similarity import word_overlap_score

# Scoring weights for combining keyword match and relevance.
_QUERY_MATCH_WEIGHT = 0.7
_RELEVANCE_WEIGHT = 0.3


class Chunk(BaseModel):
    """A retrieved text chunk from a knowledge source.

    Attributes:
        content: The text content of the chunk.
        source: Source identifier (e.g. file path, URL, doc ID).
        relevance_score: Relevance to the query (0.0-1.0).
        metadata: Additional metadata (page number, section, etc.).
    """

    content: str
    source: str = ""
    relevance_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class RetrieverBackend(Protocol):
    """Protocol for RAG retriever backends.

    Implementations must provide retrieve() and health_check().
    The SDK can work with any vector store, search engine,
    or custom retrieval system that implements this protocol.
    """

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[Chunk]:
        """Retrieve chunks matching a query.

        Args:
            query: The search query.
            top_k: Maximum chunks to return.

        Returns:
            List of Chunks ranked by relevance.
        """
        ...

    async def health_check(self) -> bool:
        """Check if the retriever backend is healthy.

        Returns:
            True if the backend is ready, False otherwise.
        """
        ...


class InMemoryRetriever:
    """Simple in-memory retriever for development and testing.

    Stores chunks in a list and retrieves by keyword matching.
    Not suitable for production use.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []

    def add_chunk(self, chunk: Chunk) -> None:
        """Add a chunk to the store."""
        self._chunks.append(chunk)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add multiple chunks to the store."""
        self._chunks.extend(chunks)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[Chunk]:
        """Retrieve chunks matching the query by keyword overlap."""
        scored: list[tuple[float, Chunk]] = []
        for chunk in self._chunks:
            keyword_score = word_overlap_score(query, chunk.content)
            final_score = (
                keyword_score * _QUERY_MATCH_WEIGHT
                + chunk.relevance_score * _RELEVANCE_WEIGHT
            )
            scored.append((final_score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    async def health_check(self) -> bool:
        """Return True (in-memory backend is always healthy)."""
        return True

    @property
    def chunk_count(self) -> int:
        """Number of chunks in the store."""
        return len(self._chunks)
