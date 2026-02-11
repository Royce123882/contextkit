"""RAG context assembly.

Retrieves chunks from a backend, ranks them, deduplicates,
and converts to ContextBlocks with full provenance tracking.
Supports budget-aware retrieval (stops when token limit is reached).
"""

from __future__ import annotations

from typing import List

from contextkit.utils.token_counting import count as count_tokens
from contextkit.constants import (
    DEFAULT_ENCODING,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    PRIORITY_RAG_CHUNK,
)
from contextkit.core import BlockType, ContextBlock
from contextkit.observe.provenance import Origin
from contextkit.rag.chunk import Chunk, RetrieverBackend
from contextkit.utils.text_similarity import word_overlap_similarity


class RAGContext:
    """Retrieval-augmented context assembly.

    Wraps a RetrieverBackend to provide high-level retrieval
    with ranking, deduplication, attribution tracking, and
    budget-aware retrieval.

    Args:
        retriever: A RetrieverBackend instance.
        retriever_name: Human-readable name for the retriever
            (used in Origin tracking).
    """

    def __init__(
        self,
        retriever: RetrieverBackend,
        retriever_name: str = "default",
    ) -> None:
        self._retriever = retriever
        self._retriever_name = retriever_name

    async def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        max_tokens: int | None = None,
        min_relevance: float = 0.0,
        encoding: str = DEFAULT_ENCODING,
        priority: int = PRIORITY_RAG_CHUNK,
    ) -> List[ContextBlock]:
        """Retrieve chunks and convert to ContextBlocks.

        Chunks below min_relevance are filtered out. If max_tokens
        is set, retrieval stops when the budget is exhausted.
        Each block has Origin auto-populated with query, retriever,
        and relevance score.

        Args:
            query: The search query.
            top_k: Maximum chunks to retrieve.
            max_tokens: Token budget for all retrieved content.
            min_relevance: Minimum relevance threshold (0.0-1.0).
            encoding: Tiktoken encoding for token counting.
            priority: Block priority for created blocks.

        Returns:
            List of ContextBlocks with retrieved content.
        """
        chunks = await self._retriever.retrieve(query=query, top_k=top_k)

        # Filter by relevance
        if min_relevance > 0:
            chunks = [c for c in chunks if c.relevance_score >= min_relevance]

        # Deduplicate by content similarity
        chunks = _deduplicate_chunks(chunks)

        # Convert to blocks with budget awareness
        blocks: List[ContextBlock] = []
        total_tokens = 0

        for chunk in chunks:
            token_count = count_tokens(chunk.content, encoding)

            if max_tokens is not None:
                if total_tokens + token_count > max_tokens:
                    break

            origin = Origin(
                source="rag",
                details={
                    "query": query,
                    "retriever": self._retriever_name,
                    "relevance_score": chunk.relevance_score,
                    "source": chunk.source,
                },
            )

            block = ContextBlock(
                type=BlockType.RAG,
                content=chunk.content,
                priority=priority,
                name=f"rag_{chunk.source or len(blocks)}",
                origin=origin,
                metadata=chunk.metadata,
            )
            blocks.append(block)
            total_tokens += token_count

        return blocks

    async def health_check(self) -> bool:
        """Check if the retriever backend is healthy."""
        return await self._retriever.health_check()


def _deduplicate_chunks(
    chunks: List[Chunk],
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> List[Chunk]:
    """Remove near-duplicate chunks based on content overlap.

    Uses a simple word-overlap metric. For production use,
    consider using embedding-based similarity.

    Args:
        chunks: List of chunks to deduplicate.
        similarity_threshold: Overlap threshold for dedup (0.0-1.0).

    Returns:
        Deduplicated list of chunks.
    """
    if len(chunks) <= 1:
        return chunks

    result: List[Chunk] = []
    for chunk in chunks:
        is_duplicate = any(
            word_overlap_similarity(chunk.content, existing.content)
            >= similarity_threshold
            for existing in result
        )
        if not is_duplicate:
            result.append(chunk)

    return result
