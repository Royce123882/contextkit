"""In-memory storage backend for development and testing."""

from __future__ import annotations

from typing import Any, Dict, List

from contextkit.constants import (
    DEFAULT_IMPORTANCE,
    DEFAULT_TOP_K,
    IMPORTANCE_WEIGHT,
    WORD_MATCH_WEIGHT,
)
from contextkit.memory.record import MemoryRecord
from contextkit.utils.text_similarity import word_overlap_score


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
