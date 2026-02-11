"""Simple in-memory retriever for development and testing."""

from __future__ import annotations

from typing import List, Tuple

from contextkit.constants import DEFAULT_TOP_K, IMPORTANCE_WEIGHT, WORD_MATCH_WEIGHT
from contextkit.rag.chunk import Chunk
from contextkit.utils.text_similarity import word_overlap_score


class InMemoryRetriever:
    """Simple in-memory retriever for development and testing.

    Stores chunks in a list and retrieves by keyword matching.
    Not suitable for production use.
    """

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []

    def add_chunk(self, chunk: Chunk) -> None:
        """Add a chunk to the store."""
        self._chunks.append(chunk)

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add multiple chunks to the store."""
        self._chunks.extend(chunks)

    async def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Chunk]:
        """Retrieve chunks matching the query by keyword overlap."""
        scored: List[Tuple[float, Chunk]] = []
        for chunk in self._chunks:
            keyword_score = word_overlap_score(query, chunk.content)
            final_score = (
                keyword_score * WORD_MATCH_WEIGHT
                + chunk.relevance_score * IMPORTANCE_WEIGHT
            )
            scored.append((final_score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    async def health_check(self) -> bool:
        """Return True (in-memory backend is always healthy)."""
        return True

    @property
    def chunk_count(self) -> int:
        """Number of chunks in the store."""
        return len(self._chunks)
