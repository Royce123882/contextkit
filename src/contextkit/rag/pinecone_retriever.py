"""Pinecone vector database retriever backend.

Requires the ``pinecone`` optional dependency group::

    pip install contextkit[pinecone]
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from contextkit.constants import DEFAULT_TOP_K
from contextkit.rag.chunk import Chunk

logger = logging.getLogger("contextkit")


def _import_pinecone() -> Any:
    """Import pinecone at runtime, raising a clear error if missing."""
    try:
        import pinecone as _pinecone  # noqa: WPS433

        return _pinecone
    except ImportError as exc:
        raise ImportError(
            "pinecone is required for PineconeRetriever. "
            "Install it with: pip install contextkit[pinecone]"
        ) from exc


class PineconeRetriever:
    """Retriever backed by the Pinecone managed vector database.

    Uses the official ``pinecone`` Python SDK with async support.

    Args:
        index_name: Name of the Pinecone index.
        embed_fn: Callable that maps a query string to a vector.
        api_key: Pinecone API key.
        namespace: Optional Pinecone namespace for scoping.
        content_field: Metadata key that stores chunk text.
        source_field: Metadata key that stores the source id.
    """

    def __init__(
        self,
        index_name: str,
        embed_fn: Callable[[str], List[float]],
        *,
        api_key: str,
        namespace: str = "",
        content_field: str = "content",
        source_field: str = "source",
    ) -> None:
        pinecone_mod = _import_pinecone()

        self._embed_fn = embed_fn
        self._namespace = namespace
        self._content_field = content_field
        self._source_field = source_field

        pc = pinecone_mod.Pinecone(api_key=api_key)
        self._index = pc.Index(index_name)

    async def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Chunk]:
        """Query Pinecone for matching vectors.

        Args:
            query: The search query.
            top_k: Maximum number of chunks to return.

        Returns:
            Chunks ranked by similarity score.
        """
        vector = self._embed_fn(query)
        results = self._index.query(
            vector=vector,
            top_k=top_k,
            namespace=self._namespace or None,
            include_metadata=True,
        )

        chunks: List[Chunk] = []
        for match in results.get("matches", []):
            metadata: Dict[str, Any] = dict(match.get("metadata") or {})
            content = str(metadata.pop(self._content_field, ""))
            source = str(metadata.pop(self._source_field, ""))
            score = max(0.0, min(1.0, float(match.get("score", 0.0))))
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
        """Return True if the Pinecone index is reachable."""
        try:
            stats = self._index.describe_index_stats()
            return stats is not None
        except Exception:  # noqa: BLE001
            logger.debug("Pinecone health check failed", exc_info=True)
            return False
