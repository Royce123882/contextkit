"""Filter pipeline step.

Removes blocks below a relevance or quality threshold.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import List

from contextkit.core import ContextBlock
from contextkit.observe.provenance import Mutation
from contextkit.pipeline.base import PipelineStep


class FilterStep(PipelineStep):
    """Remove blocks below a relevance or quality threshold.

    Uses a custom filter function or removes blocks whose origin
    relevance_score is below a threshold.

    Args:
        min_relevance: Minimum relevance score (from Origin).
        filter_fn: Optional custom filter function.
    """

    def __init__(
        self,
        min_relevance: float = 0.0,
        filter_fn: Callable[[ContextBlock], bool] | None = None,
    ) -> None:
        self._min_relevance = min_relevance
        self._filter_fn = filter_fn

    @property
    def name(self) -> str:
        """Return the step name."""
        return "FilterStep"

    def process(self, blocks: List[ContextBlock]) -> List[ContextBlock]:
        """Filter blocks by relevance score or custom predicate."""
        result: List[ContextBlock] = []

        for block in blocks:
            if self._should_keep(block):
                result.append(block)
            else:
                self._record_removal(block)

        return result

    def _should_keep(self, block: ContextBlock) -> bool:
        """Determine whether a block passes the filter criteria."""
        if self._filter_fn is not None and not self._filter_fn(block):
            return False

        if (
            self._min_relevance > 0
            and block.origin is not None
            and block.origin.relevance_score is not None
            and block.origin.relevance_score < self._min_relevance
        ):
            return False

        return True

    def _record_removal(self, block: ContextBlock) -> None:
        """Record a mutation for a filtered-out block."""
        score = "N/A"
        if block.origin is not None and block.origin.relevance_score is not None:
            score = str(block.origin.relevance_score)

        block.mutations.append(
            Mutation(
                step=self.name,
                action="removed",
                detail=f"relevance {score} below threshold {self._min_relevance}",
                tokens_before=block.token_count,
                tokens_after=0,
            )
        )
