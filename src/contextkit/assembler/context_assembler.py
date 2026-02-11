"""Context assembler for composing blocks into a window."""

from __future__ import annotations

from typing import List, Tuple

from contextkit.assembler.report import AssemblyReport, BlockDecision
from contextkit.core import ContextBlock, ContextWindow
from contextkit.observe.events import ContextEvent, EventData, emit


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

    def assemble(self, blocks: List[ContextBlock]) -> ContextWindow:
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
        included, excluded = self._partition_blocks(sorted_blocks)

        report = self._build_report(included, excluded)
        self._window.assembly_report = report
        self._emit_assembly_complete(report, included, excluded)
        self._window.check_budget_warnings()

        return self._window

    def _partition_blocks(
        self, sorted_blocks: List[ContextBlock]
    ) -> Tuple[List[BlockDecision], List[BlockDecision]]:
        """Add fitting blocks to the window and partition into included/excluded."""
        included: List[BlockDecision] = []
        excluded: List[BlockDecision] = []

        for block in sorted_blocks:
            block_tokens = block.token_count
            fits_budget = (
                self._window.token_count + block_tokens <= self._window.max_tokens
            )

            if fits_budget:
                self._window.add_unchecked(block)
                included.append(self._create_decision(block, block_tokens))
            else:
                excluded.append(
                    self._create_decision(block, block_tokens, reason="budget_exceeded")
                )

        return included, excluded

    @staticmethod
    def _create_decision(
        block: ContextBlock,
        block_tokens: int,
        reason: str | None = None,
    ) -> BlockDecision:
        """Create a BlockDecision record for a block."""
        origin_summary = block.origin.summary() if block.origin else "unknown"
        return BlockDecision(
            block_name=block.display_name,
            block_type=block.type.value,
            tokens=block_tokens,
            priority=block.priority,
            origin_summary=origin_summary,
            reason=reason,
        )

    def _build_report(
        self,
        included: List[BlockDecision],
        excluded: List[BlockDecision],
    ) -> AssemblyReport:
        """Build the assembly report from included/excluded decisions."""
        return AssemblyReport(
            included=included,
            excluded=excluded,
            total_tokens=self._window.token_count,
            budget_remaining=self._window.budget_remaining,
            cost_estimate=self._window.cost_estimate,
        )

    @staticmethod
    def _emit_assembly_complete(
        report: AssemblyReport,
        included: List[BlockDecision],
        excluded: List[BlockDecision],
    ) -> None:
        """Emit the ASSEMBLY_COMPLETE event."""
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

    @property
    def report(self) -> AssemblyReport | None:
        """The report from the last assembly, if available."""
        report = self._window.assembly_report
        if isinstance(report, AssemblyReport):
            return report
        return None
