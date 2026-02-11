"""Memory record data model and backend protocol."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from contextkit.constants import DEFAULT_IMPORTANCE, DEFAULT_TOP_K


class MemoryRecord(BaseModel):
    """A single record in a memory backend.

    Attributes:
        key: Unique identifier for this record.
        content: The stored content string.
        metadata: Arbitrary key-value pairs (timestamps, tags, etc.).
        tags: Categorization tags for filtering.
        stored_at: When this record was stored.
        importance: Importance score for retrieval ranking (0.0-1.0).
    """

    key: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    stored_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    importance: float = DEFAULT_IMPORTANCE


@runtime_checkable
class MemoryBackend(Protocol):
    """Protocol for memory storage backends.

    Implementations must provide 4 async methods: store, retrieve,
    delete, and list_records. The SDK ships InMemoryBackend as a
    default and SQLiteBackend as an optional persistent backend.
    """

    async def store(
        self,
        key: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
        tags: List[str] | None = None,
        importance: float = DEFAULT_IMPORTANCE,
    ) -> MemoryRecord:
        """Store a memory record.

        Args:
            key: Unique identifier.
            content: The content to store.
            metadata: Optional key-value pairs.
            tags: Optional categorization tags.
            importance: Importance score (0.0-1.0).

        Returns:
            The stored MemoryRecord.
        """
        ...

    async def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        tags: List[str] | None = None,
    ) -> List[MemoryRecord]:
        """Retrieve records matching a query.

        Args:
            query: The search query string.
            top_k: Maximum number of records to return.
            tags: Optional tag filter.

        Returns:
            List of matching MemoryRecords, ranked by relevance.
        """
        ...

    async def delete(self, key: str) -> bool:
        """Delete a record by key.

        Args:
            key: The unique identifier of the record to delete.

        Returns:
            True if the record was found and deleted, False otherwise.
        """
        ...

    async def list_records(
        self,
        tags: List[str] | None = None,
    ) -> List[MemoryRecord]:
        """List all records, optionally filtered by tags.

        Args:
            tags: Optional tag filter (records must have ALL tags).

        Returns:
            List of matching MemoryRecords.
        """
        ...
