"""Reorder pipeline step.

Reorders blocks for optimal attention patterns, placing
high-priority content at the start and end of the sequence.
"""

from __future__ import annotations

from contextkit.core import ContextBlock
from contextkit.observe.provenance import Mutation
from contextkit.pipeline.base import PipelineStep


class ReorderStep(PipelineStep):
    """Reorder blocks for optimal attention patterns.

    Places high-priority content at the start and end of the
    sequence (important edges), with lower-priority content
    in the middle.

    Args:
        strategy: Reordering strategy. "important_edges" places
            highest priority at start/end.
    """

    def __init__(self, strategy: str = "important_edges") -> None:
        self._strategy = strategy

    @property
    def name(self) -> str:
        """Return the step name."""
        return "ReorderStep"

    def process(self, blocks: list[ContextBlock]) -> list[ContextBlock]:
        """Reorder blocks using the configured strategy."""
        if len(blocks) <= 2:
            return blocks

        if self._strategy == "important_edges":
            return self._reorder_edges(blocks)
        return blocks

    def _reorder_edges(self, blocks: list[ContextBlock]) -> list[ContextBlock]:
        """Place highest priority at start/end, lowest in middle."""
        sorted_by_priority = sorted(
            enumerate(blocks),
            key=lambda x: x[1].priority,
            reverse=True,
        )

        block_count = len(sorted_by_priority)
        result: list[ContextBlock | None] = [None] * block_count

        start_idx = 0
        end_idx = block_count - 1

        for rank, (orig_idx, block) in enumerate(sorted_by_priority):
            if rank % 2 == 0:
                result[start_idx] = block
                new_pos = start_idx
                start_idx += 1
            else:
                result[end_idx] = block
                new_pos = end_idx
                end_idx -= 1

            if new_pos != orig_idx:
                block.mutations.append(
                    Mutation(
                        step=self.name,
                        action="moved",
                        detail=f"position {orig_idx} -> position {new_pos}",
                        tokens_before=block.token_count,
                        tokens_after=block.token_count,
                    )
                )

        return [b for b in result if b is not None]
