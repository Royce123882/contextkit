"""Post-retrieval RAG compression pipeline step.

Compresses RAG blocks by extracting the most query-relevant
sentences and implements selective augmentation (dropping
irrelevant chunks entirely).

Research basis: RECOMP (Xu et al., 2023) -- extractive/abstractive
compression between retrieval and generation with selective
augmentation achieves 6% compression rate with minimal loss.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, List

from contextkit.constants import DEFAULT_RAG_MAX_SENTENCES
from contextkit.core import ContextBlock
from contextkit.core.block import BlockType
from contextkit.observe.provenance import Mutation
from contextkit.pipeline.base import PipelineStep
from contextkit.utils.text_similarity import extract_key_sentences
from contextkit.utils.token_counting import count as count_tokens


class RAGCompressStep(PipelineStep):
    """Compress RAG blocks via sentence extraction and selective drop.

    Only targets blocks with ``type=BlockType.RAG``.  Other block
    types pass through unchanged.

    For each RAG block the step:

    1. Checks the origin relevance score -- if below *drop_threshold*
       the block is removed entirely (selective augmentation).
    2. Extracts the top *max_sentences* most query-relevant sentences
       using the query stored in ``block.origin.details["query"]``.
    3. Records a :class:`Mutation` with before/after token counts.

    Args:
        compressor: Optional custom function ``(text, query) -> text``
            to replace the default extractive compressor.
        max_sentences: Maximum sentences to keep per RAG block.
        drop_threshold: Minimum relevance score to keep a RAG block.
            Blocks below this are removed entirely.
    """

    def __init__(
        self,
        compressor: Callable[[str, str], str] | None = None,
        max_sentences: int = DEFAULT_RAG_MAX_SENTENCES,
        drop_threshold: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._compressor = compressor
        self._max_sentences = max_sentences
        self._drop_threshold = drop_threshold

    @property
    def name(self) -> str:
        """Return the step name."""
        return "RAGCompressStep"

    def process(self, blocks: List[ContextBlock]) -> List[ContextBlock]:
        """Compress or drop RAG blocks; pass through everything else."""
        result: List[ContextBlock] = []

        for block in blocks:
            if block.type != BlockType.RAG:
                result.append(block)
                continue

            compressed = self._process_rag_block(block)
            if compressed is not None:
                result.append(compressed)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_origin_relevance(self, block: ContextBlock) -> float:
        """Extract the relevance score from a block's origin."""
        if block.origin is None:
            return 0.0
        return float(block.origin.details.get("relevance_score", 0.0))

    def _get_origin_query(self, block: ContextBlock) -> str:
        """Extract the retrieval query from a block's origin."""
        if block.origin is None:
            return ""
        return str(block.origin.details.get("query", ""))

    def _process_rag_block(
        self, block: ContextBlock
    ) -> ContextBlock | None:
        """Compress a single RAG block, or return None to drop it."""
        relevance = self._get_origin_relevance(block)

        # Selective augmentation: drop low-relevance blocks
        if relevance < self._drop_threshold:
            block.mutations.append(
                Mutation(
                    step=self.name,
                    action="removed",
                    detail=(
                        f"relevance {relevance:.2f} below "
                        f"drop threshold {self._drop_threshold:.2f}"
                    ),
                    tokens_before=block.token_count,
                    tokens_after=0,
                )
            )
            return None

        # Only compress string content
        if not isinstance(block.content, str):
            return block

        tokens_before = block.token_count
        query = self._get_origin_query(block)
        compressed_content = self._compress(block.content, query)

        if compressed_content == block.content:
            return block

        tokens_after = count_tokens(compressed_content)

        new_block = block.model_copy(update={"content": compressed_content})
        new_block.mutations.append(
            Mutation(
                step=self.name,
                action="compressed",
                detail=(
                    f"{tokens_before:,} -> {tokens_after:,} tokens "
                    f"(kept {self._max_sentences} sentences)"
                ),
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                before_content=block.content,
                after_content=compressed_content,
            )
        )
        return new_block

    def _compress(self, content: str, query: str) -> str:
        """Apply the compressor function or default extractive logic."""
        if self._compressor is not None:
            return self._compressor(content, query)
        return extract_key_sentences(content, query, self._max_sentences)
