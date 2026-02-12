"""RAG chunk data model and retriever backend protocol."""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from contextkit.constants import DEFAULT_TOP_K


class Chunk(BaseModel):
    """A retrieved text chunk from a knowledge source.

    Attributes:
        content: The text content of the chunk.
        source: Source identifier (e.g. file path, URL, doc ID).
        relevance_score: Relevance to the query (0.0-1.0).
        metadata: Additional metadata (page number, section, etc.).
    """

    content: str = Field(description="The text content of the chunk.")
    source: str = Field(default="", description="Source identifier (e.g. file path, URL, doc ID).")
    relevance_score: float = Field(default=0.0, description="Relevance to the query (0.0-1.0).")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata (page number, section, etc.).")


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
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Chunk]:
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
