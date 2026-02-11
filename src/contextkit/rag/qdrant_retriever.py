"""Qdrant vector database retriever backend.

Requires the ``qdrant`` optional dependency group::

    pip install contextkit[qdrant]
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from contextkit.constants import DEFAULT_TOP_K
from contextkit.rag.chunk import Chunk

logger = logging.getLogger("contextkit")


class QdrantRetriever:
    """Retriever backed by a Qdrant vector database.

    Supports both remote Qdrant servers and local in-memory/on-disk
    mode for development (``location=":memory:"``).

    Args:
        collection_name: Qdrant collection to search.
        embed_fn: A callable that maps a query string to a
            list of floats (embedding vector).
        url: Qdrant server URL.  Mutually exclusive with *location*.
        api_key: API key for Qdrant Cloud.
        location: Local Qdrant path or ``":memory:"`` for in-process.
        content_field: Name of the payload field that stores chunk text.
        source_field: Name of the payload field that stores the source id.
        score_threshold: Minimum similarity score to include.
    """

    def __init__(
        self,
        collection_name: str,
        embed_fn: Callable[[str], List[float]],
        *,
        url: str | None = None,
        api_key: str | None = None,
        location: str | None = None,
        content_field: str = "content",
        source_field: str = "source",
        score_threshold: float = 0.0,
    ) -> None:
        try:
            from qdrant_client import AsyncQdrantClient
        except ImportError as exc:
            raise ImportError(
                "qdrant-client is required for QdrantRetriever. "
                "Install it with: pip install contextkit[qdrant]"
            ) from exc

        self._collection = collection_name
        self._embed_fn = embed_fn
        self._content_field = content_field
        self._source_field = source_field
        self._score_threshold = score_threshold

        if url is not None:
            self._client = AsyncQdrantClient(url=url, api_key=api_key)
        elif location is not None:
            self._client = AsyncQdrantClient(location=location)
        else:
            raise ValueError("Either 'url' or 'location' must be provided")

    async def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Chunk]:
        """Search Qdrant for chunks matching *query*.

        Args:
            query: The search query.
            top_k: Maximum number of chunks to return.

        Returns:
            Chunks ranked by vector similarity.
        """
        vector = self._embed_fn(query)
        results = await self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=top_k,
            score_threshold=self._score_threshold,
        )

        chunks: List[Chunk] = []
        for point in results:
            payload: Dict[str, Any] = point.payload or {}
            content = str(payload.get(self._content_field, ""))
            source = str(payload.get(self._source_field, ""))
            # Qdrant cosine scores are in [0, 1] for normalized vectors
            score = max(0.0, min(1.0, point.score))
            metadata = {
                k: v
                for k, v in payload.items()
                if k not in (self._content_field, self._source_field)
            }
            chunks.append(
                Chunk(
                    content=content,
                    source=source,
                    relevance_score=score,
                    metadata=metadata,
                )
            )
        return chunks

    async def health_check(self) -> bool:
        """Return True if the Qdrant collection is reachable."""
        try:
            info = await self._client.get_collection(self._collection)
            return info.status.value == "green"
        except Exception:  # noqa: BLE001
            logger.debug("Qdrant health check failed", exc_info=True)
            return False
