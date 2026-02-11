"""Memory backend protocol and built-in implementations.

Defines the MemoryBackend protocol that all memory backends must
implement, along with the MemoryRecord data model and a default
InMemoryBackend for development and testing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from contextkit.constants import (
    DEFAULT_IMPORTANCE,
    DEFAULT_TOP_K,
    IMPORTANCE_WEIGHT,
    WORD_MATCH_WEIGHT,
)
from contextkit.utils.text_similarity import word_overlap_score


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


class InMemoryBackend:
    """In-memory storage backend for development and testing.

    Records are stored in a dict keyed by record key. Retrieval
    uses simple substring matching on content. Not suitable for
    production use with large datasets.
    """

    def __init__(self) -> None:
        self._records: Dict[str, MemoryRecord] = {}

    async def store(
        self,
        key: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
        tags: List[str] | None = None,
        importance: float = DEFAULT_IMPORTANCE,
    ) -> MemoryRecord:
        """Store a record in the in-memory dict."""
        record = MemoryRecord(
            key=key,
            content=content,
            metadata=metadata or {},
            tags=tags or [],
            importance=importance,
        )
        self._records[key] = record
        return record

    async def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        tags: List[str] | None = None,
    ) -> List[MemoryRecord]:
        """Retrieve records by keyword matching and importance."""
        results: List[MemoryRecord] = []

        for record in self._records.values():
            if tags and not all(tag in record.tags for tag in tags):
                continue
            results.append(record)

        def relevance_score(rec: MemoryRecord) -> float:
            score = word_overlap_score(query, rec.content)
            return score * WORD_MATCH_WEIGHT + rec.importance * IMPORTANCE_WEIGHT

        results.sort(key=relevance_score, reverse=True)
        return results[:top_k]

    async def delete(self, key: str) -> bool:
        """Delete a record by key."""
        if key in self._records:
            del self._records[key]
            return True
        return False

    async def list_records(
        self,
        tags: List[str] | None = None,
    ) -> List[MemoryRecord]:
        """List records, optionally filtered by tags."""
        if tags is None:
            return list(self._records.values())
        return [
            record
            for record in self._records.values()
            if all(tag in record.tags for tag in tags)
        ]

    @property
    def record_count(self) -> int:
        """Number of records in the backend."""
        return len(self._records)
