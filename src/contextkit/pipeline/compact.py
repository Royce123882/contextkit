"""Compact pipeline step.

Compacts verbose blocks by summarization or truncation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import List

from contextkit.utils.token_counting import count as count_tokens
from contextkit.constants import (
    DEFAULT_COMPACT_MIN_TOKENS,
    DEFAULT_COMPACT_TARGET_RATIO,
)
from contextkit.core import ContextBlock
from contextkit.observe.provenance import Mutation
from contextkit.pipeline.base import PipelineStep


class CompactStep(PipelineStep):
    """Compact verbose blocks by summarization.

    Uses a provided compaction function to summarize long blocks.
    Records full before/after content in the mutation log.

    Args:
        compactor: A function that takes content string and returns
            a shorter summary. If None, uses simple truncation.
        target_ratio: Target compression ratio (0.0-1.0).
            0.5 means aim for 50% of original size.
        min_tokens: Only compact blocks above this token count.
    """

    def __init__(
        self,
        compactor: Callable[[str], str] | None = None,
        target_ratio: float = DEFAULT_COMPACT_TARGET_RATIO,
        min_tokens: int = DEFAULT_COMPACT_MIN_TOKENS,
    ) -> None:
        self._compactor = compactor or self._default_compactor
        self._target_ratio = target_ratio
        self._min_tokens = min_tokens

    @property
    def name(self) -> str:
        """Return the step name."""
        return "CompactStep"

    def process(self, blocks: List[ContextBlock]) -> List[ContextBlock]:
        """Compact long blocks by truncation or custom compactor."""
        return [self._compact_block(block) for block in blocks]

    def _compact_block(self, block: ContextBlock) -> ContextBlock:
        """Compact a single block if it exceeds the minimum token threshold."""
        if not isinstance(block.content, str):
            return block

        tokens_before = block.token_count
        if tokens_before < self._min_tokens:
            return block

        before_content = block.content
        compacted = self._compactor(before_content)
        tokens_after = count_tokens(compacted)

        if tokens_after >= tokens_before:
            return block

        new_block = block.model_copy(update={"content": compacted})
        new_block.mutations.append(
            Mutation(
                step=self.name,
                action="compacted",
                detail=f"{tokens_before:,} -> {tokens_after:,} tokens",
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                before_content=before_content,
                after_content=compacted,
            )
        )
        return new_block

    def _default_compactor(self, content: str) -> str:
        """Truncate content to the target ratio, breaking at sentence boundaries."""
        target_len = int(len(content) * self._target_ratio)
        if target_len >= len(content):
            return content

        truncated = content[:target_len]
        last_period = truncated.rfind(".")
        if last_period > target_len * 0.5:
            return truncated[: last_period + 1]
        return truncated
