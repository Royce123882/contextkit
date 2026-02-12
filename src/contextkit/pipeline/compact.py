"""Compact pipeline step.

Compacts verbose blocks by summarization or truncation.
Includes collapse detection to warn when compaction loses too
much information (keyword retention drops below a threshold).

Research basis: Ravaut et al. (2023) -- summarization can cause
"information collapse" where key terms vanish; monitoring
keyword retention catches this before it reaches the model.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import List

from contextkit.utils.text_similarity import word_overlap_score
from contextkit.utils.token_counting import count as count_tokens
from contextkit.constants import (
    DEFAULT_COMPACT_MIN_TOKENS,
    DEFAULT_COMPACT_TARGET_RATIO,
)
from contextkit.core import ContextBlock
from contextkit.observe.provenance import Mutation
from contextkit.pipeline.base import PipelineStep

logger = logging.getLogger("contextkit")


class CompactStep(PipelineStep):
    """Compact verbose blocks by summarization.

    Uses a provided compaction function to summarize long blocks.
    Records full before/after content in the mutation log.

    After compaction, checks keyword retention between the original
    and compacted content.  If retention drops below *max_info_loss*,
    a warning is logged so operators can tune the compactor.

    Args:
        compactor: A function that takes content string and returns
            a shorter summary. If None, uses simple truncation.
        target_ratio: Target compression ratio (0.0-1.0).
            0.5 means aim for 50% of original size.
        min_tokens: Only compact blocks above this token count.
        max_info_loss: Maximum acceptable keyword loss (0.0-1.0).
            If ``1.0 - keyword_retention`` exceeds this value, a
            warning is logged.  Defaults to 0.5 (50% loss).
    """

    def __init__(
        self,
        compactor: Callable[[str], str] | None = None,
        target_ratio: float = DEFAULT_COMPACT_TARGET_RATIO,
        min_tokens: int = DEFAULT_COMPACT_MIN_TOKENS,
        max_info_loss: float = 0.5,
    ) -> None:
        self._compactor = compactor or self._default_compactor
        self._target_ratio = target_ratio
        self._min_tokens = min_tokens
        self._max_info_loss = max_info_loss

    @property
    def name(self) -> str:
        """Return the step name."""
        return "CompactStep"

    def process(self, blocks: List[ContextBlock]) -> List[ContextBlock]:
        """Compact long blocks by truncation or custom compactor."""
        return [self._compact_block(block) for block in blocks]

    def _compact_block(self, block: ContextBlock) -> ContextBlock:
        """Compact a single block if it exceeds the minimum token threshold.

        After compaction, checks keyword retention and logs a warning
        if information loss exceeds the configured threshold.
        """
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

        keyword_retention = word_overlap_score(before_content, compacted)
        info_loss = 1.0 - keyword_retention

        collapse_warning = ""
        if info_loss > self._max_info_loss:
            collapse_warning = (
                f" [COLLAPSE WARNING: keyword retention {keyword_retention:.0%}, "
                f"loss {info_loss:.0%} exceeds threshold {self._max_info_loss:.0%}]"
            )
            logger.warning(
                "Collapse detected in block '%s': keyword retention %.0f%%, "
                "info loss %.0f%% exceeds threshold %.0f%%",
                block.display_name,
                keyword_retention * 100,
                info_loss * 100,
                self._max_info_loss * 100,
            )

        new_block = block.model_copy(update={"content": compacted})
        new_block.mutations.append(
            Mutation(
                step=self.name,
                action="compacted",
                detail=(
                    f"{tokens_before:,} -> {tokens_after:,} tokens"
                    f"{collapse_warning}"
                ),
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
