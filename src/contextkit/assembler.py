"""Context assembler for composing blocks into a window.

The assembler takes a list of ContextBlocks and composes them into a
ContextWindow with ordering and priority rules. It produces an
AssemblyReport that powers explain() for understanding why blocks
were included or excluded.
"""

from __future__ import annotations

from pydantic import BaseModel

from contextkit.core import ContextBlock, ContextWindow
from contextkit.observe.events import ContextEvent, EventData, emit


class BlockDecision(BaseModel):
    """Records the assembler's decision about a single block.

    Attributes:
        block_name: Display name of the block.
        block_type: The BlockType value.
        tokens: Token count for this block.
        priority: Block priority.
        origin_summary: Human-readable origin summary.
        reason: Why it was excluded (None if included).
    """

    block_name: str
    block_type: str
    tokens: int
    priority: int
    origin_summary: str
    reason: str | None = None


class AssemblyReport(BaseModel):
    """Full report of an assembly operation.

    Attributes:
        included: Blocks that were added to the window.
        excluded: Blocks that were skipped (with reasons).
        total_tokens: Total tokens in the assembled window.
        budget_remaining: Tokens remaining after assembly.
        cost_estimate: Estimated input cost in USD.
    """

    included: list[BlockDecision]
    excluded: list[BlockDecision]
    total_tokens: int
    budget_remaining: int
    cost_estimate: float


class ContextAssembler:
    """Composes blocks into a ContextWindow with priority-based ordering.

    The assembler sorts blocks by priority (highest first) and adds them
    to the window until the budget is exhausted. Blocks that don't fit
    are recorded in the exclusion list with reasons.

    Args:
        window: The target ContextWindow to assemble into.
    """

    def __init__(self, window: ContextWindow) -> None:
        self._window = window

    def assemble(self, blocks: list[ContextBlock]) -> ContextWindow:
        """Assemble blocks into the window by priority order.

        Sorts blocks by priority (highest first), adds those that fit,
        and records exclusions. Produces an AssemblyReport stored on
        the window for use by explain().

        Args:
            blocks: The candidate blocks to assemble.

        Returns:
            The ContextWindow with assembled blocks.
        """
        sorted_blocks = sorted(blocks, key=lambda b: b.priority, reverse=True)

        included: list[BlockDecision] = []
        excluded: list[BlockDecision] = []

        for block in sorted_blocks:
            block_tokens = block.token_count
            origin_summary = block.origin.summary() if block.origin else "unknown"

            if self._window.token_count + block_tokens <= self._window.max_tokens:
                self._window.add_unchecked(block)
                included.append(
                    BlockDecision(
                        block_name=block.display_name,
                        block_type=block.type.value,
                        tokens=block_tokens,
                        priority=block.priority,
                        origin_summary=origin_summary,
                    )
                )
            else:
                excluded.append(
                    BlockDecision(
                        block_name=block.display_name,
                        block_type=block.type.value,
                        tokens=block_tokens,
                        priority=block.priority,
                        origin_summary=origin_summary,
                        reason="budget_exceeded",
                    )
                )

        report = AssemblyReport(
            included=included,
            excluded=excluded,
            total_tokens=self._window.token_count,
            budget_remaining=self._window.budget_remaining,
            cost_estimate=self._window.cost_estimate,
        )

        self._window.assembly_report = report

        # Emit ASSEMBLY_COMPLETE event
        emit(
            EventData(
                event=ContextEvent.ASSEMBLY_COMPLETE,
                details={
                    "included_count": len(included),
                    "excluded_count": len(excluded),
                    "total_tokens": report.total_tokens,
                    "budget_remaining": report.budget_remaining,
                },
            )
        )

        # Check budget warnings after assembly
        self._window.check_budget_warnings()

        return self._window

    @property
    def report(self) -> AssemblyReport | None:
        """The report from the last assembly, if available."""
        report = self._window.assembly_report
        if isinstance(report, AssemblyReport):
            return report
        return None
