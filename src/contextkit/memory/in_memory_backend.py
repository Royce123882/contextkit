"""In-memory storage backend for development and testing.

Incorporates temporal decay in retrieval scoring so that recently
accessed records are ranked higher, modelling the Ebbinghaus
forgetting curve with spaced-repetition reinforcement.
"""

from __future__ import annotations

from typing import Any, Dict, List

from contextkit.constants import (
    DECAY_WEIGHT,
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
    ranks by a blend of word-overlap relevance, importance, and
    temporal decay.  Retrieved records have their access metadata
    updated (spaced repetition).

    The scoring formula is::

        score = word_match * WORD_MATCH_WEIGHT
              + importance * IMPORTANCE_WEIGHT
              + decay      * DECAY_WEIGHT
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
        """Retrieve records by keyword matching, importance, and decay.

        Retrieved records have their ``access_count`` and
        ``last_accessed`` updated to reinforce future retention.
        """
        results: List[MemoryRecord] = []

        for record in self._records.values():
            if tags and not all(tag in record.tags for tag in tags):
                continue
            results.append(record)

        def relevance_score(rec: MemoryRecord) -> float:
            word_match = word_overlap_score(query, rec.content)
            decay = rec.decay_factor()
            return (
                word_match * WORD_MATCH_WEIGHT
                + rec.importance * IMPORTANCE_WEIGHT
                + decay * DECAY_WEIGHT
            )

        results.sort(key=relevance_score, reverse=True)
        top_results = results[:top_k]

        for record in top_results:
            record.record_access()

        return top_results

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
