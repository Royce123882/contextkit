"""PostgreSQL-backed memory storage.

Provides a durable, ACID-compliant MemoryBackend using asyncpg --
the same driver used by :class:`~contextkit.rag.PgvectorRetriever`.
Users who already run pgvector for RAG can share the same Postgres
instance (and optionally the same asyncpg pool) for memory storage.

Records are stored in a ``memory_records`` table with full decay
metadata (``access_count``, ``last_accessed``).

Requires the ``asyncpg`` optional dependency (shared with pgvector)::

    pip install contextkit[pgvector]
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    pass

from contextkit.constants import (
    DECAY_WEIGHT,
    DEFAULT_IMPORTANCE,
    DEFAULT_TOP_K,
    IMPORTANCE_WEIGHT,
    WORD_MATCH_WEIGHT,
)
from contextkit.memory.record import MemoryRecord
from contextkit.utils.text_similarity import memory_relevance_score

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memory_records (
    key TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    tags JSONB NOT NULL DEFAULT '[]',
    stored_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    importance DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TIMESTAMPTZ
)
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_memory_tags ON memory_records USING GIN (tags)
"""

_UPSERT_SQL = """
INSERT INTO memory_records
    (key, content, metadata, tags, stored_at, importance, access_count, last_accessed)
VALUES ($1, $2, $3::jsonb, $4::jsonb, $5, $6, $7, $8)
ON CONFLICT (key) DO UPDATE SET
    content = EXCLUDED.content,
    metadata = EXCLUDED.metadata,
    tags = EXCLUDED.tags,
    stored_at = EXCLUDED.stored_at,
    importance = EXCLUDED.importance,
    access_count = EXCLUDED.access_count,
    last_accessed = EXCLUDED.last_accessed
"""

_SELECT_ALL_SQL = """
SELECT key, content, metadata, tags, stored_at, importance,
       access_count, last_accessed
FROM memory_records
"""

_DELETE_SQL = "DELETE FROM memory_records WHERE key = $1"

_COUNT_SQL = "SELECT COUNT(*) FROM memory_records"

_UPDATE_ACCESS_SQL = """
UPDATE memory_records
SET access_count = $2, last_accessed = $3
WHERE key = $1
"""

_RETRIEVE_LIMIT: int = 1000
"""Maximum rows fetched before in-memory scoring during retrieval."""


def _import_asyncpg() -> Any:
    """Import asyncpg at runtime, raising a clear error if missing."""
    try:
        import asyncpg as _asyncpg

        return _asyncpg
    except ImportError as exc:
        raise ImportError(
            "asyncpg is required for PostgresBackend. "
            "Install it with: pip install contextkit[pgvector]"
        ) from exc


class PostgresBackend:
    """PostgreSQL-backed memory storage with decay and access tracking.

    Records are stored in a ``memory_records`` table with JSONB columns
    for metadata and tags.  A GIN index on ``tags`` enables efficient
    tag-based filtering.

    Retrieval scoring blends word overlap, importance, and temporal
    decay.  Retrieved records have their access metadata updated
    in the database (spaced repetition).

    **Sharing a pool with PgvectorRetriever** -- if you already use
    ``PgvectorRetriever`` for RAG, pass its pool to avoid creating
    a second connection pool::

        import asyncpg

        pool = await asyncpg.create_pool(dsn)

        rag = PgvectorRetriever(dsn=dsn, embedding_function=embed)
        memory = PostgresBackend(pool=pool)  # shares the same Postgres

    Args:
        dsn: PostgreSQL connection string.  Ignored when *pool* is
            provided.
        pool: An existing ``asyncpg.Pool`` to reuse (e.g. from
            PgvectorRetriever).  When supplied, no new pool is created.
        pool_min_size: Minimum connections (only used when creating
            a new pool).
        pool_max_size: Maximum connections (only used when creating
            a new pool).
    """

    def __init__(
        self,
        dsn: str = "postgresql://localhost:5432/contextkit",
        pool: Any = None,
        pool_min_size: int = 2,
        pool_max_size: int = 10,
    ) -> None:
        self._dsn = dsn
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._pool: Any = pool
        self._initialized = False

    async def _ensure_pool(self) -> Any:
        """Lazily create the connection pool and ensure the table exists."""
        if self._pool is None:
            self._pool = await _import_asyncpg().create_pool(
                self._dsn,
                min_size=self._pool_min_size,
                max_size=self._pool_max_size,
            )

        if not self._initialized:
            async with self._pool.acquire() as conn:
                await conn.execute(_CREATE_TABLE_SQL)
                await conn.execute(_CREATE_INDEX_SQL)
            self._initialized = True

        return self._pool

    async def store(
        self,
        key: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
        tags: List[str] | None = None,
        importance: float = DEFAULT_IMPORTANCE,
    ) -> MemoryRecord:
        """Store a record in PostgreSQL with upsert semantics.

        Args:
            key: Unique identifier for the record.
            content: The content string to store.
            metadata: Optional key-value pairs (stored as JSONB).
            tags: Optional categorization tags (stored as JSONB array).
            importance: Importance score (0.0-1.0).

        Returns:
            The stored MemoryRecord.
        """
        pool = await self._ensure_pool()
        stored_at = datetime.now(timezone.utc)
        record_tags = tags or []
        record_metadata = metadata or {}

        record = MemoryRecord(
            key=key,
            content=content,
            metadata=record_metadata,
            tags=record_tags,
            stored_at=stored_at,
            importance=importance,
        )

        async with pool.acquire() as conn:
            await conn.execute(
                _UPSERT_SQL,
                key,
                content,
                json.dumps(record_metadata),
                json.dumps(record_tags),
                stored_at,
                importance,
                0,
                None,
            )

        return record

    async def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        tags: List[str] | None = None,
    ) -> List[MemoryRecord]:
        """Retrieve records ranked by relevance, importance, and decay.

        Retrieved records have their ``access_count`` and
        ``last_accessed`` updated in the database.

        Args:
            query: The search query string.
            top_k: Maximum number of records to return.
            tags: Optional tag filter.

        Returns:
            List of matching MemoryRecords, ranked by relevance.
        """
        pool = await self._ensure_pool()

        sql = _SELECT_ALL_SQL
        params: List[Any] = []
        if tags:
            conditions = []
            for i, tag in enumerate(tags, start=1):
                conditions.append(f"tags @> ${i}::jsonb")
                params.append(json.dumps([tag]))
            sql += " WHERE " + " AND ".join(conditions)
        param_index = len(params) + 1
        sql += f" LIMIT ${param_index}"
        params.append(_RETRIEVE_LIMIT)

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        records = [self._row_to_record(row) for row in rows]

        # Score and rank
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

        # Update access metadata
        now = datetime.now(timezone.utc)
        async with pool.acquire() as conn:
            for record in top_results:
                record.record_access()
                await conn.execute(
                    _UPDATE_ACCESS_SQL,
                    record.key,
                    record.access_count,
                    now,
                )

        return top_results

    async def delete(self, key: str) -> bool:
        """Delete a record by key.

        Args:
            key: The unique identifier of the record to delete.

        Returns:
            True if the record existed and was deleted.
        """
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(_DELETE_SQL, key)
            return result == "DELETE 1"

    async def list_records(
        self,
        tags: List[str] | None = None,
        offset: int = 0,
        limit: int = 0,
    ) -> List[MemoryRecord]:
        """List all records, optionally filtered by tags with pagination.

        Tag filtering is pushed into SQL using JSONB containment
        operators for efficient querying with the GIN index.

        Args:
            tags: Optional tag filter (records must have ALL tags).
            offset: Number of records to skip (for pagination).
            limit: Maximum number of records to return. 0 means no limit.

        Returns:
            List of matching MemoryRecords.
        """
        pool = await self._ensure_pool()

        sql = _SELECT_ALL_SQL
        params: List[Any] = []
        if tags:
            conditions = []
            for i, tag in enumerate(tags, start=1):
                conditions.append(f"tags @> ${i}::jsonb")
                params.append(json.dumps([tag]))
            sql += " WHERE " + " AND ".join(conditions)

        if limit:
            param_index = len(params) + 1
            sql += f" LIMIT ${param_index} OFFSET ${param_index + 1}"
            params.extend([limit, offset])
        elif offset:
            param_index = len(params) + 1
            sql += f" OFFSET ${param_index}"
            params.append(offset)

        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [self._row_to_record(row) for row in rows]

    async def health_check(self) -> bool:
        """Verify PostgreSQL connectivity.

        Returns:
            True if a simple SELECT 1 succeeds.
        """
        try:
            pool = await self._ensure_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def record_count(self) -> int:
        """Count the number of records in the database.

        Returns:
            The total number of stored records.
        """
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(_COUNT_SQL)

    @staticmethod
    def _row_to_record(row: Any) -> MemoryRecord:
        """Convert a database row to a MemoryRecord."""
        tags_raw = row["tags"]
        if isinstance(tags_raw, str):
            tags_raw = json.loads(tags_raw)

        metadata_raw = row["metadata"]
        if isinstance(metadata_raw, str):
            metadata_raw = json.loads(metadata_raw)

        return MemoryRecord(
            key=row["key"],
            content=row["content"],
            metadata=metadata_raw,
            tags=tags_raw,
            stored_at=row["stored_at"],
            importance=float(row["importance"]),
            access_count=int(row["access_count"]),
            last_accessed=row["last_accessed"],
        )
