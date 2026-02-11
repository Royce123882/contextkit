"""PostgreSQL + pgvector retriever backend.

Requires the ``pgvector`` optional dependency group::

    pip install contextkit[pgvector]
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from contextkit.constants import DEFAULT_TOP_K
from contextkit.rag.chunk import Chunk

logger = logging.getLogger("contextkit")


class PgvectorRetriever:
    """Retriever backed by PostgreSQL with the pgvector extension.

    Uses ``asyncpg`` for async connection pooling and ``<=>``
    (cosine distance) for vector similarity search.

    Args:
        dsn: PostgreSQL connection string
            (e.g. ``"postgresql://user:pass@localhost/db"``).
        embed_fn: Callable that maps a query string to a vector.
        table: Table name storing documents and embeddings.
        content_column: Column that stores the chunk text.
        source_column: Column that stores the source identifier.
        embedding_column: Column that stores the vector embedding.
        pool_min: Minimum connections in the pool.
        pool_max: Maximum connections in the pool.
    """

    def __init__(
        self,
        dsn: str,
        embed_fn: Callable[[str], List[float]],
        *,
        table: str = "documents",
        content_column: str = "content",
        source_column: str = "source",
        embedding_column: str = "embedding",
        metadata_column: str = "metadata",
        pool_min: int = 2,
        pool_max: int = 10,
    ) -> None:
        try:
            import asyncpg  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "asyncpg is required for PgvectorRetriever. "
                "Install it with: pip install contextkit[pgvector]"
            ) from exc

        self._dsn = dsn
        self._embed_fn = embed_fn
        self._table = table
        self._content_col = content_column
        self._source_col = source_column
        self._embedding_col = embedding_column
        self._metadata_col = metadata_column
        self._pool_min = pool_min
        self._pool_max = pool_max
        self._pool: Any = None

    async def _get_pool(self) -> Any:
        """Lazily create the asyncpg connection pool."""
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                self._dsn,
                min_size=self._pool_min,
                max_size=self._pool_max,
            )
        return self._pool

    async def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Chunk]:
        """Search pgvector for nearest-neighbour chunks.

        Args:
            query: The search query.
            top_k: Maximum number of chunks to return.

        Returns:
            Chunks ranked by cosine similarity.
        """
        import json

        vector = self._embed_fn(query)
        vector_str = "[" + ",".join(str(v) for v in vector) + "]"

        pool = await self._get_pool()
        sql = (
            f"SELECT {self._content_col}, {self._source_col}, "  # noqa: S608
            f"       {self._metadata_col}, "
            f"       1 - ({self._embedding_col} <=> $1::vector) AS score "
            f"FROM {self._table} "
            f"ORDER BY {self._embedding_col} <=> $1::vector "
            f"LIMIT $2"
        )

        rows = await pool.fetch(sql, vector_str, top_k)

        chunks: List[Chunk] = []
        for row in rows:
            raw_meta = row[self._metadata_col]
            metadata: Dict[str, Any] = (
                json.loads(raw_meta) if isinstance(raw_meta, str) else (raw_meta or {})
            )
            score = max(0.0, min(1.0, float(row["score"])))
            chunks.append(
                Chunk(
                    content=str(row[self._content_col]),
                    source=str(row[self._source_col] or ""),
                    relevance_score=score,
                    metadata=metadata,
                )
            )
        return chunks

    async def health_check(self) -> bool:
        """Return True if PostgreSQL is reachable and pgvector is installed."""
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchval("SELECT 1")
                return row == 1
        except Exception:  # noqa: BLE001
            logger.debug("pgvector health check failed", exc_info=True)
            return False

    async def close(self) -> None:
        """Shut down the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
