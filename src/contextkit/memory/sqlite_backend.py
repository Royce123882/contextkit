"""SQLite-backed persistent memory storage.

Provides a zero-config persistent MemoryBackend using aiosqlite.
Records survive process restarts and are stored in a local SQLite
database file.

Supports temporal decay via ``access_count`` and ``last_accessed``
columns, and incorporates decay into retrieval scoring for
Ebbinghaus-style spaced repetition.

Requires the ``aiosqlite`` optional dependency::

    pip install contextkit[sqlite]
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List

from contextkit.constants import (
    DECAY_WEIGHT,
    DEFAULT_IMPORTANCE,
    DEFAULT_TOP_K,
    IMPORTANCE_WEIGHT,
    WORD_MATCH_WEIGHT,
)
from contextkit.memory.record import MemoryRecord
from contextkit.utils.text_similarity import memory_relevance_score

if TYPE_CHECKING:
    import aiosqlite

# SQL statements used by the backend.
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memory_records (
    key TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    tags TEXT NOT NULL DEFAULT '[]',
    stored_at TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT
)
"""

_MIGRATE_ACCESS_COLUMNS_SQL = [
    "ALTER TABLE memory_records ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE memory_records ADD COLUMN last_accessed TEXT",
]

_UPSERT_SQL = """
INSERT INTO memory_records
    (key, content, metadata, tags, stored_at, importance, access_count, last_accessed)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(key) DO UPDATE SET
    content = excluded.content,
    metadata = excluded.metadata,
    tags = excluded.tags,
    stored_at = excluded.stored_at,
    importance = excluded.importance,
    access_count = excluded.access_count,
    last_accessed = excluded.last_accessed
"""

_SELECT_ALL_SQL = (
    "SELECT key, content, metadata, tags, stored_at, importance,"
    " access_count, last_accessed FROM memory_records"
)

_DELETE_SQL = "DELETE FROM memory_records WHERE key = ?"

_COUNT_SQL = "SELECT COUNT(*) FROM memory_records"

_UPDATE_ACCESS_SQL = (
    "UPDATE memory_records SET access_count = ?, last_accessed = ? WHERE key = ?"
)


def _import_aiosqlite() -> Any:
    """Import aiosqlite at runtime, raising a clear error if missing."""
    import aiosqlite as _aiosqlite  # noqa: WPS433

    return _aiosqlite


class SQLiteBackend:
    """Persistent memory backend using SQLite via aiosqlite.

    Records are stored in a local SQLite database file. The schema
    is auto-created on first use. Tags and metadata are serialized
    as JSON text columns.

    Retrieval scoring blends word overlap, importance, and temporal
    decay.  Retrieved records have their access metadata updated
    in the database for spaced repetition.

    Args:
        db_path: Path to the SQLite database file.
            Defaults to ``"memory.db"`` in the current directory.
    """

    def __init__(self, db_path: str = "memory.db") -> None:
        self._db_path = db_path
        self._initialized = False

    def _connect(self) -> aiosqlite.Connection:
        """Return an aiosqlite connection context manager for the database."""
        return _import_aiosqlite().connect(self._db_path)

    async def _ensure_table(self) -> None:
        """Create the memory_records table if it doesn't exist.

        Also migrates existing databases by adding ``access_count``
        and ``last_accessed`` columns if they are missing.
        """
        if self._initialized:
            return

        async with self._connect() as db:
            await db.execute(_CREATE_TABLE_SQL)

            # Migrate older databases that lack the decay columns
            for sql in _MIGRATE_ACCESS_COLUMNS_SQL:
                try:
                    await db.execute(sql)
                except Exception:
                    pass  # Column already exists

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

        Uses INSERT OR REPLACE (upsert) semantics -- storing with
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

        async with self._connect() as db:
            await db.execute(
                _UPSERT_SQL,
                (
                    record.key,
                    record.content,
                    json.dumps(record.metadata),
                    json.dumps(record.tags),
                    record.stored_at.isoformat(),
                    record.importance,
                    record.access_count,
                    None,
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

        Uses word-overlap scoring blended with importance and temporal
        decay for ranking.  Retrieved records have their access
        metadata updated in the database (spaced repetition).

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

        # Score and rank by relevance with decay
        def relevance_score(record: MemoryRecord) -> float:
            return memory_relevance_score(
                query=query,
                content=record.content,
                importance=record.importance,
                decay=record.decay_factor(),
                word_match_weight=WORD_MATCH_WEIGHT,
                importance_weight=IMPORTANCE_WEIGHT,
                decay_weight=DECAY_WEIGHT,
            )

        records.sort(key=relevance_score, reverse=True)
        top_results = records[:top_k]

        # Update access metadata on retrieved records
        now = datetime.now(timezone.utc)
        async with self._connect() as db:
            for record in top_results:
                record.record_access()
                await db.execute(
                    _UPDATE_ACCESS_SQL,
                    (record.access_count, now.isoformat(), record.key),
                )
            await db.commit()

        return top_results

    async def delete(self, key: str) -> bool:
        """Delete a record by key.

        Args:
            key: The unique identifier of the record to delete.

        Returns:
            True if the record was found and deleted, False otherwise.
        """
        await self._ensure_table()

        async with self._connect() as db:
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
        records: List[MemoryRecord] = []
        async with self._connect() as db:
            async with db.execute(_SELECT_ALL_SQL) as cursor:
                async for row in cursor:
                    record = _row_to_record(row)
                    records.append(record)
        return records

    @property
    async def record_count(self) -> int:
        """Count the number of records in the database."""
        await self._ensure_table()

        async with self._connect() as db:
            async with db.execute(_COUNT_SQL) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0


def _row_to_record(row: Any) -> MemoryRecord:
    """Convert a database row to a MemoryRecord.

    Args:
        row: A tuple of (key, content, metadata_json, tags_json,
            stored_at_iso, importance, access_count, last_accessed_iso).

    Returns:
        A MemoryRecord instance.
    """
    (
        key,
        content,
        metadata_json,
        tags_json,
        stored_at_iso,
        importance,
        access_count,
        last_accessed_iso,
    ) = row

    last_accessed = (
        datetime.fromisoformat(last_accessed_iso)
        if last_accessed_iso
        else None
    )

    return MemoryRecord(
        key=key,
        content=content,
        metadata=json.loads(metadata_json),
        tags=json.loads(tags_json),
        stored_at=datetime.fromisoformat(stored_at_iso),
        importance=importance,
        access_count=access_count,
        last_accessed=last_accessed,
    )
