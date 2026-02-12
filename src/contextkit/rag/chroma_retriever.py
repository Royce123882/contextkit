"""ChromaDB retriever backend.

Requires the ``chroma`` optional dependency group::

    pip install contextkit[chroma]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

from contextkit.constants import DEFAULT_TOP_K
from contextkit.rag.chunk import Chunk

if TYPE_CHECKING:
    pass

logger = logging.getLogger("contextkit")


def _import_chromadb() -> Any:
    """Import chromadb at runtime, raising a clear error if missing."""
    try:
        import chromadb as _chromadb

        return _chromadb
    except ImportError as exc:
        raise ImportError(
            "chromadb is required for ChromaRetriever. "
            "Install it with: pip install contextkit[chroma]"
        ) from exc


class ChromaRetriever:
    """Retriever backed by a ChromaDB collection.

    Works in embedded mode (no server needed) or against a
    remote Chroma instance via ``AsyncHttpClient``.

    Args:
        collection_name: Chroma collection to search.
        client: A ``chromadb.AsyncClientAPI`` instance.
            If *None*, an ephemeral in-memory client is created.
        content_field: Metadata key that stores chunk text when
            using metadata-based storage.  By default Chroma
            stores content in the ``documents`` return field.
        source_field: Metadata key for the source identifier.
    """

    def __init__(
        self,
        collection_name: str,
        client: Any = None,
        *,
        source_field: str = "source",
    ) -> None:
        _import_chromadb()  # Fail fast if chromadb is not installed

        self._collection_name = collection_name
        self._source_field = source_field
        self._client = client
        self._collection: Any = None

    async def _get_collection(self) -> Any:
        """Lazily resolve the Chroma collection."""
        if self._collection is not None:
            return self._collection

        if self._client is None:
            self._client = await _import_chromadb().AsyncClient()

        self._collection = await self._client.get_or_create_collection(
            name=self._collection_name,
        )
        return self._collection

    async def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Chunk]:
        """Query ChromaDB for matching documents.

        Args:
            query: The search query (embedded by Chroma's model).
            top_k: Maximum number of chunks to return.

        Returns:
            Chunks ranked by distance (converted to relevance).
        """
        collection = await self._get_collection()
        results = await collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: List[Chunk] = []
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances, strict=False):
            meta = meta or {}
            source = str(meta.pop(self._source_field, ""))
            # Chroma returns L2 distances; convert to a 0-1 score
            score = 1.0 / (1.0 + dist)
            metadata: Dict[str, Any] = dict(meta)
            chunks.append(
                Chunk(
                    content=str(doc or ""),
                    source=source,
                    relevance_score=score,
                    metadata=metadata,
                )
            )
        return chunks

    async def health_check(self) -> bool:
        """Return True if the Chroma collection is accessible."""
        try:
            await self._get_collection()
            return True
        except Exception:
            logger.debug("Chroma health check failed", exc_info=True)
            return False
