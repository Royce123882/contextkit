"""Trim pipeline step.

Removes low-priority blocks to fit within a token budget.
"""

from __future__ import annotations

from contextkit.core import ContextBlock
from contextkit.observe.provenance import Mutation
from contextkit.pipeline.base import PipelineStep


class TrimStep(PipelineStep):
    """Remove low-priority blocks to fit a token budget.

    Blocks below the priority threshold are removed first. If that's
    not enough, remaining blocks are removed in priority order
    (lowest first) until the budget is met.

    Args:
        max_tokens: Maximum total tokens allowed.
        min_priority: Minimum priority threshold (blocks below are removed).
    """

    def __init__(
        self,
        max_tokens: int | None = None,
        min_priority: int = 0,
    ) -> None:
        self._max_tokens = max_tokens
        self._min_priority = min_priority

    @property
    def name(self) -> str:
        """Return the step name."""
        return "TrimStep"

    def process(self, blocks: list[ContextBlock]) -> list[ContextBlock]:
        """Remove low-priority blocks and enforce token budget."""
        result, removed = self._remove_below_priority(blocks)

        if self._max_tokens is not None:
            result = self._enforce_budget(result, removed)

        return result

    def _remove_below_priority(
        self, blocks: list[ContextBlock]
    ) -> tuple[list[ContextBlock], list[ContextBlock]]:
        """Remove blocks whose priority is below the minimum threshold."""
        result: list[ContextBlock] = []
        removed: list[ContextBlock] = []

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

    def _enforce_budget(
        self,
        blocks: list[ContextBlock],
        removed: list[ContextBlock],
    ) -> list[ContextBlock]:
        """Remove lowest-priority blocks until total fits the budget."""
        assert self._max_tokens is not None

        total = sum(b.token_count for b in blocks)
        if total <= self._max_tokens:
            return blocks

        # Sort by priority ascending (remove lowest first)
        blocks.sort(key=lambda b: b.priority)
        kept: list[ContextBlock] = []
        budget_used = 0

        # Keep from highest priority end
        for block in reversed(blocks):
            if budget_used + block.token_count <= self._max_tokens:
                kept.append(block)
                budget_used += block.token_count
            else:
                block.mutations.append(
                    Mutation(
                        step=self.name,
                        action="removed",
                        detail=(
                            f"budget exceeded, removed to fit {self._max_tokens} tokens"
                        ),
                        tokens_before=block.token_count,
                        tokens_after=0,
                    )
                )
                removed.append(block)

        return list(reversed(kept))
