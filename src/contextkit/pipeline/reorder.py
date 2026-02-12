"""Reorder pipeline step.

Reorders blocks for optimal attention patterns. Supports two strategies:

- ``"important_edges"``: Places high-priority blocks at the start and
  end of the sequence where LLM attention is strongest (based on
  "Lost in the Middle", Liu et al. 2023).
- ``"prefix_stable"``: Groups stable block types (system prompt, tool
  definitions, examples, schemas) first as a shared prefix, then
  applies important-edges within the dynamic section.  This maximises
  KV-cache / prefix-cache hits on providers that support it (Anthropic,
  Google).
"""

from __future__ import annotations

from typing import List, Set, Tuple

from contextkit.core import ContextBlock
from contextkit.core.block import BlockType
from contextkit.observe.provenance import Mutation
from contextkit.pipeline.base import PipelineStep

_STABLE_BLOCK_TYPES: Set[BlockType] = {
    BlockType.SYSTEM_PROMPT,
    BlockType.TOOL_DEFINITIONS,
    BlockType.OUTPUT_SCHEMAS,
    BlockType.EXAMPLES,
}
"""Block types whose content rarely changes between requests."""


class ReorderStep(PipelineStep):
    """Reorder blocks for optimal attention patterns.

    Args:
        strategy: ``"important_edges"`` places highest priority at
            start/end.  ``"prefix_stable"`` groups stable types first
            then applies important-edges within the dynamic section.
    """

    _VALID_STRATEGIES = {"important_edges", "prefix_stable"}

    def __init__(self, strategy: str = "important_edges") -> None:
        if strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}. "
                f"Choose from: {', '.join(sorted(self._VALID_STRATEGIES))}"
            )
        self._strategy = strategy

    @property
    def name(self) -> str:
        """Return the step name."""
        return "ReorderStep"

    def process(self, blocks: List[ContextBlock]) -> List[ContextBlock]:
        """Reorder blocks using the configured strategy."""
        if len(blocks) <= 2:
            return blocks

        if self._strategy == "important_edges":
            return self._reorder_edges(blocks)
        if self._strategy == "prefix_stable":
            return self._reorder_prefix_stable(blocks)
        return blocks

    # ------------------------------------------------------------------
    # Strategy: important_edges
    # ------------------------------------------------------------------

    def _reorder_edges(self, blocks: List[ContextBlock]) -> List[ContextBlock]:
        """Place highest priority at start/end, lowest in middle."""
        sorted_by_priority = sorted(
            enumerate(blocks),
            key=lambda x: x[1].priority,
            reverse=True,
        )

        block_count = len(sorted_by_priority)
        result: List[ContextBlock | None] = [None] * block_count

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

    # ------------------------------------------------------------------
    # Strategy: prefix_stable
    # ------------------------------------------------------------------

    def _reorder_prefix_stable(
        self, blocks: List[ContextBlock]
    ) -> List[ContextBlock]:
        """Group stable block types first, then reorder the dynamic tail."""
        indexed = list(enumerate(blocks))

        stable = [(i, b) for i, b in indexed if b.type in _STABLE_BLOCK_TYPES]
        dynamic = [(i, b) for i, b in indexed if b.type not in _STABLE_BLOCK_TYPES]

        # Stable section: sort by priority descending (system prompt first)
        stable.sort(key=lambda x: x[1].priority, reverse=True)

        # Dynamic section: apply important-edges within the sub-list
        dynamic_blocks = [b for _, b in dynamic]
        if len(dynamic_blocks) > 2:
            dynamic_blocks = self._reorder_edges(dynamic_blocks)

        # Map each dynamic block back to its original index
        dynamic_orig_idx = {id(b): i for i, b in dynamic}

        # Build ordered list with original indices preserved
        ordered_pairs: List[Tuple[int, ContextBlock]] = [
            (orig_idx, b) for orig_idx, b in stable
        ]
        ordered_pairs += [
            (dynamic_orig_idx.get(id(b), -1), b) for b in dynamic_blocks
        ]

        ordered = [b for _, b in ordered_pairs]

        # Record mutations for blocks whose position changed
        for new_pos, (orig_idx, block) in enumerate(ordered_pairs):
            if new_pos != orig_idx:
                region = "prefix" if block.type in _STABLE_BLOCK_TYPES else "dynamic"
                block.mutations.append(
                    Mutation(
                        step=self.name,
                        action="moved",
                        detail=(
                            f"position {orig_idx} -> position {new_pos} "
                            f"({region} region, {block.type.value})"
                        ),
                        tokens_before=block.token_count,
                        tokens_after=block.token_count,
                    )
                )

        return ordered
