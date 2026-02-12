"""Memory record data model and backend protocol.

Extends the base record with temporal decay support inspired by
the Ebbinghaus forgetting curve, where memory strength decays
exponentially over time but is reinforced by repeated access.

Research basis: Ebbinghaus (1885) -- retention decays as
``R = e^(-t/S)`` where *S* (stability) increases with each
retrieval (spaced repetition effect).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from contextkit.constants import DEFAULT_IMPORTANCE, DEFAULT_TOP_K, MEMORY_HALF_LIFE_HOURS


class MemoryRecord(BaseModel):
    """A single record in a memory backend.

    Supports temporal decay via :meth:`decay_factor`, which models
    Ebbinghaus-style forgetting.  Repeated retrieval increases
    ``access_count``, which extends the effective half-life
    (spaced repetition).

    Attributes:
        key: Unique identifier for this record.
        content: The stored content string.
        metadata: Arbitrary key-value pairs (timestamps, tags, etc.).
        tags: Categorization tags for filtering.
        stored_at: When this record was stored.
        importance: Importance score for retrieval ranking (0.0-1.0).
        access_count: Number of times this record has been retrieved.
        last_accessed: Timestamp of the most recent retrieval.
    """

    key: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    stored_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    importance: float = DEFAULT_IMPORTANCE
    access_count: int = 0
    last_accessed: datetime | None = None

    def decay_factor(
        self,
        half_life_hours: float = MEMORY_HALF_LIFE_HOURS,
        now: datetime | None = None,
    ) -> float:
        """Compute a temporal decay factor in the range (0.0, 1.0].

        Uses an exponential decay model where the effective half-life
        grows with ``access_count`` (spaced repetition effect):

            effective_half_life = half_life_hours * (1 + access_count)
            hours_elapsed = (now - last_accessed).total_seconds() / 3600
            factor = 2 ** -(hours_elapsed / effective_half_life)

        A freshly-accessed record returns ~1.0; a record untouched for
        several half-lives returns a value approaching 0.0.

        Args:
            half_life_hours: Base half-life in hours before access
                reinforcement.
            now: Reference time; defaults to ``datetime.now(UTC)``.

        Returns:
            Decay factor between 0.0 (exclusive) and 1.0 (inclusive).
        """
        reference = self.last_accessed or self.stored_at
        if now is None:
            now = datetime.now(timezone.utc)

        hours_elapsed = (now - reference).total_seconds() / 3600.0
        if hours_elapsed <= 0:
            return 1.0

        effective_half_life = half_life_hours * (1 + self.access_count)
        return math.pow(2, -(hours_elapsed / effective_half_life))

    def record_access(self) -> None:
        """Record a retrieval access, updating count and timestamp."""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)


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
