"""Trim pipeline step.

Removes low-priority blocks to fit within a token budget.
Supports optional query-aware pruning where blocks are scored
by a blend of priority and relevance to the current query.

Research basis: DYCP (Choi et al., 2025) -- dynamic context
pruning reduces first-token latency by ~3x by retaining only
query-relevant content.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import List, Tuple

from contextkit.constants import DEFAULT_RELEVANCE_WEIGHT
from contextkit.core import ContextBlock
from contextkit.observe.provenance import Mutation
from contextkit.pipeline.base import PipelineStep
from contextkit.utils.text_similarity import word_overlap_score


class TrimStep(PipelineStep):
    """Remove low-priority blocks to fit a token budget.

    Blocks below the priority threshold are removed first. If that's
    not enough, remaining blocks are removed in priority order
    (lowest first) until the budget is met.

    When *query* is provided, blocks are scored by a blend of
    normalized priority and query relevance, so a low-priority but
    highly relevant block can survive over a high-priority irrelevant one.

    Args:
        max_tokens: Maximum total tokens allowed.
        min_priority: Minimum priority threshold (blocks below are removed).
        query: Optional query string for relevance-aware pruning.
        relevance_weight: Balance between relevance and priority
            (0.0 = priority only, 1.0 = relevance only).
        score_fn: Custom scoring function ``(query, content) -> float``.
            Defaults to :func:`word_overlap_score`.
    """

    def __init__(
        self,
        max_tokens: int | None = None,
        min_priority: int = 0,
        query: str | None = None,
        relevance_weight: float = DEFAULT_RELEVANCE_WEIGHT,
        score_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        self._max_tokens = max_tokens
        self._min_priority = min_priority
        self._query = query
        self._relevance_weight = relevance_weight
        self._score_fn = score_fn or word_overlap_score

    @property
    def name(self) -> str:
        """Return the step name."""
        return "TrimStep"

    def process(self, blocks: List[ContextBlock]) -> List[ContextBlock]:
        """Remove low-priority blocks and enforce token budget."""
        result, removed = self._remove_below_priority(blocks)

        if self._max_tokens is not None:
            result = self._enforce_budget(result, removed)

        return result

    def _remove_below_priority(
        self, blocks: List[ContextBlock]
    ) -> Tuple[List[ContextBlock], List[ContextBlock]]:
        """Remove blocks whose priority is below the minimum threshold."""
        result: List[ContextBlock] = []
        removed: List[ContextBlock] = []

        for block in blocks:
            if block.priority < self._min_priority:
                block.mutations.append(
                    Mutation(
                        step=self.name,
                        action="removed",
                        detail=(
                            f"priority {block.priority} below "
                            f"threshold {self._min_priority}"
                        ),
                        tokens_before=block.token_count,
                        tokens_after=0,
                    )
                )
                removed.append(block)
            else:
                result.append(block)

        return result, removed

    def _block_sort_score(self, block: ContextBlock) -> float:
        """Compute a blended sort score for a block.

        When no query is set, returns normalized priority (0-1).
        When a query is set, blends priority and relevance.
        """
        max_priority = 100
        normalized_priority = min(block.priority / max_priority, 1.0)

        if self._query is None:
            return normalized_priority

        content = block.content if isinstance(block.content, str) else ""
        relevance = self._score_fn(self._query, content)

        return (
            (1.0 - self._relevance_weight) * normalized_priority
            + self._relevance_weight * relevance
        )

    def _enforce_budget(
        self,
        blocks: List[ContextBlock],
        removed: List[ContextBlock],
    ) -> List[ContextBlock]:
        """Remove lowest-scoring blocks until total fits the budget."""
        assert self._max_tokens is not None

        total = sum(b.token_count for b in blocks)
        if total <= self._max_tokens:
            return blocks

        # Sort ascending by score (lowest score removed first)
        blocks = sorted(blocks, key=self._block_sort_score)
        kept: List[ContextBlock] = []
        budget_used = 0

        # Keep from highest-scoring end
        for block in reversed(blocks):
            if budget_used + block.token_count <= self._max_tokens:
                kept.append(block)
                budget_used += block.token_count
            else:
                score = self._block_sort_score(block)
                detail = (
                    f"budget exceeded, removed to fit {self._max_tokens} tokens"
                )
                if self._query is not None:
                    detail += f" (blended_score={score:.2f})"
                block.mutations.append(
                    Mutation(
                        step=self.name,
                        action="removed",
                        detail=detail,
                        tokens_before=block.token_count,
                        tokens_after=0,
                    )
                )
                removed.append(block)

        return list(reversed(kept))
