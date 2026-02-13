"""Prune stale tool outputs from the context.

Removes tool output blocks that are older than a configured number
of turns or superseded by newer calls to the same tool. In agent
loops, old tool results accumulate rapidly and become noise.

Research basis: LOCA-bench (arXiv:2602.07962) -- removing stale
tool calls substantially improves model performance in long-running
agentic sessions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from contextkit.core import ContextBlock
from contextkit.core.block import BlockType
from contextkit.observe.provenance import Mutation
from contextkit.pipeline.base import PipelineStep


class PruneStaleStep(PipelineStep):
    """Remove stale tool outputs from the context.

    A tool output is considered stale if:

    - It is from a tool that has been called again more recently
      (superseded), or
    - Its ``turn_number`` metadata is older than ``max_age_turns``
      turns from the most recent turn.

    Args:
        max_age_turns: Maximum turn age before a tool output is pruned.
            If None, only superseded outputs are removed (those beyond
            ``keep_last_per_tool``).
        keep_last_per_tool: Always keep the N most recent outputs per
            tool name, even if they exceed max_age_turns. Default is 1.
    """

    def __init__(
        self,
        max_age_turns: int | None = None,
        keep_last_per_tool: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._max_age_turns = max_age_turns
        self._keep_last_per_tool = max(1, keep_last_per_tool)

    @property
    def name(self) -> str:
        """Return the step name."""
        return "PruneStaleStep"

    def process(self, blocks: List[ContextBlock]) -> List[ContextBlock]:
        """Remove stale tool output blocks.

        Non-tool blocks are always kept. For tool outputs, the most
        recent ``keep_last_per_tool`` per tool name are kept; older
        ones are removed if they exceed ``max_age_turns``.

        Args:
            blocks: Input blocks to process.

        Returns:
            Filtered list with stale tool outputs removed.
        """
        # Group tool-output indices by tool name, preserving order
        tool_indices_by_name: Dict[str, List[int]] = defaultdict(list)
        for block_index, block in enumerate(blocks):
            if block.type == BlockType.TOOL_OUTPUTS:
                tool_name = self._extract_tool_name(block)
                tool_indices_by_name[tool_name].append(block_index)

        if not tool_indices_by_name:
            return blocks

        # Decide which tool-output indices to remove
        latest_turn = self._find_latest_turn(blocks)
        indices_to_remove: set[int] = set()

        for tool_name, ordered_indices in tool_indices_by_name.items():
            # Keep the N most recent (last in list = most recent in context)
            protected_indices = set(ordered_indices[-self._keep_last_per_tool:])
            candidate_indices = ordered_indices[:-self._keep_last_per_tool]

            for block_index in candidate_indices:
                if block_index in protected_indices:
                    continue
                block = blocks[block_index]
                if self._exceeds_max_age(block, latest_turn):
                    block.mutations.append(
                        Mutation(
                            step=self.name,
                            action="removed",
                            detail=f"stale tool output for '{tool_name}'",
                            tokens_before=block.token_count,
                            tokens_after=0,
                        )
                    )
                    indices_to_remove.add(block_index)

        return [
            block for block_index, block in enumerate(blocks)
            if block_index not in indices_to_remove
        ]

    def _extract_tool_name(self, block: ContextBlock) -> str:
        """Extract tool name from block origin or metadata.

        Args:
            block: A TOOL_OUTPUTS block.

        Returns:
            The tool name, falling back to the block's display name.
        """
        if block.origin and block.origin.details.get("tool_name"):
            return str(block.origin.details["tool_name"])
        if block.metadata.get("tool_name"):
            return str(block.metadata["tool_name"])
        return block.display_name

    def _find_latest_turn(self, blocks: List[ContextBlock]) -> int:
        """Find the highest turn_number across all blocks.

        Args:
            blocks: All context blocks.

        Returns:
            The maximum turn number found, or 0 if none set.
        """
        latest = 0
        for block in blocks:
            turn = block.metadata.get("turn_number", 0)
            if isinstance(turn, int) and turn > latest:
                latest = turn
        return latest

    def _exceeds_max_age(self, block: ContextBlock, latest_turn: int) -> bool:
        """Check whether a block is older than the configured max age.

        Args:
            block: The tool output block to check.
            latest_turn: The highest turn number in the context.

        Returns:
            True if the block should be pruned.
        """
        if self._max_age_turns is None:
            # No age limit configured -- prune all superseded outputs
            return True

        turn = block.metadata.get("turn_number", 0)
        if not isinstance(turn, int):
            return True

        return (latest_turn - turn) > self._max_age_turns
