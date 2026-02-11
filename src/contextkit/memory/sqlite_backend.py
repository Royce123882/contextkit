"""SQLite-backed persistent memory storage.

Provides a zero-config persistent MemoryBackend using aiosqlite.
Records survive process restarts and are stored in a local SQLite
database file.

Requires the ``aiosqlite`` optional dependency::

    pip install contextkit[sqlite]
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from contextkit.constants import (
    DEFAULT_IMPORTANCE,
    DEFAULT_TOP_K,
    IMPORTANCE_WEIGHT,
    WORD_MATCH_WEIGHT,
)
from contextkit.memory.backends import MemoryRecord
from contextkit.utils.text_similarity import word_overlap_score

# SQL statements used by the backend.
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memory_records (
    key TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    tags TEXT NOT NULL DEFAULT '[]',
    stored_at TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5
)
"""

_UPSERT_SQL = """
INSERT INTO memory_records (key, content, metadata, tags, stored_at, importance)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(key) DO UPDATE SET
    content = excluded.content,
    metadata = excluded.metadata,
    tags = excluded.tags,
    stored_at = excluded.stored_at,
    importance = excluded.importance
"""

_SELECT_ALL_SQL = (
    "SELECT key, content, metadata, tags, stored_at, importance" " FROM memory_records"
)

_DELETE_SQL = "DELETE FROM memory_records WHERE key = ?"

_COUNT_SQL = "SELECT COUNT(*) FROM memory_records"


class SQLiteBackend:
    """Persistent memory backend using SQLite via aiosqlite.

    Records are stored in a local SQLite database file. The schema
    is auto-created on first use. Tags and metadata are serialized
    as JSON text columns.

    Args:
        db_path: Path to the SQLite database file.
            Defaults to ``"memory.db"`` in the current directory.
    """

    def __init__(self, db_path: str = "memory.db") -> None:
        self._db_path = db_path
        self._initialized = False

    async def _ensure_table(self) -> None:
        """Create the memory_records table if it doesn't exist."""
        if self._initialized:
            return

        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(_CREATE_TABLE_SQL)
            await db.commit()
        self._initialized = True

    async def store(
        self,
        key: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
        tags: List[str] | None = None,
        importance: float = DEFAULT_IMPORTANCE,
    ) -> MemoryRecord:
        """Store a record in the SQLite database.

        Uses INSERT OR REPLACE (upsert) semantics — storing with
        an existing key updates the record.

        Args:
            key: Unique identifier for the record.
            content: The content string to store.
            metadata: Optional key-value pairs.
            tags: Optional categorization tags.
            importance: Importance score (0.0-1.0).

        Returns:
            The stored MemoryRecord.
        """
        import aiosqlite

        await self._ensure_table()

        stored_at = datetime.now(timezone.utc)
        record = MemoryRecord(
            key=key,
            content=content,
            metadata=metadata or {},
            tags=tags or [],
            stored_at=stored_at,
            importance=importance,
        )

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                _UPSERT_SQL,
                (
                    record.key,
                    record.content,
                    json.dumps(record.metadata),
                    json.dumps(record.tags),
                    record.stored_at.isoformat(),
                    record.importance,
                ),
            )
            await db.commit()

        return record

    async def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        tags: List[str] | None = None,
    ) -> List[MemoryRecord]:
        """Retrieve records matching a query, ranked by relevance.

        Uses word-overlap scoring blended with importance for ranking.
        Optionally filters by tags (records must have ALL specified tags).

        Args:
            query: The search query string.
            top_k: Maximum number of records to return.
            tags: Optional tag filter.

        Returns:
            List of matching MemoryRecords, ranked by relevance.
        """
        await self._ensure_table()

        records = await self._fetch_all_records()

        # Filter by tags
        if tags:
            records = [r for r in records if all(tag in r.tags for tag in tags)]

        # Score and rank by relevance
        def relevance_score(record: MemoryRecord) -> float:
            score = word_overlap_score(query, record.content)
            return score * WORD_MATCH_WEIGHT + record.importance * IMPORTANCE_WEIGHT

        records.sort(key=relevance_score, reverse=True)
        return records[:top_k]

    async def delete(self, key: str) -> bool:
        """Delete a record by key.

        Args:
            key: The unique identifier of the record to delete.

        Returns:
            True if the record was found and deleted, False otherwise.
        """
        import aiosqlite

        await self._ensure_table()

        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute(_DELETE_SQL, (key,))
            await db.commit()
            row_count: int = cursor.rowcount or 0
            return row_count > 0

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
        await self._ensure_table()

        records = await self._fetch_all_records()
        if tags is None:
            return records
        return [r for r in records if all(tag in r.tags for tag in tags)]

    async def _fetch_all_records(self) -> List[MemoryRecord]:
        """Load all records from the database."""
        import aiosqlite

        records: List[MemoryRecord] = []
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(_SELECT_ALL_SQL) as cursor:
                async for row in cursor:
                    record = _row_to_record(row)
                    records.append(record)
        return records

    @property
    async def record_count(self) -> int:
        """Count the number of records in the database."""
        import aiosqlite

        await self._ensure_table()

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(_COUNT_SQL) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0


def _row_to_record(row: Any) -> MemoryRecord:
    """Convert a database row to a MemoryRecord.

    Args:
        row: A tuple of (key, content, metadata_json, tags_json,
            stored_at_iso, importance).

    Returns:
        A MemoryRecord instance.
    """
    key, content, metadata_json, tags_json, stored_at_iso, importance = row
    return MemoryRecord(
        key=key,
        content=content,
        metadata=json.loads(metadata_json),
        tags=json.loads(tags_json),
        stored_at=datetime.fromisoformat(stored_at_iso),
        importance=importance,
    )
