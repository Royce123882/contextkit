"""Context pipeline orchestrator.

Runs an ordered sequence of pipeline steps on a context window,
collecting reports and emitting events for each step.
"""

from __future__ import annotations

from typing import List, Tuple

from contextkit.core import ContextBlock, ContextWindow
from contextkit.observe.events import (
    ContextEvent,
    EventData,
    PipelineEventData,
    emit,
)
from contextkit.pipeline.base import PipelineReport, PipelineStep, StepReport


class ContextPipeline:
    """Named, ordered pipeline of transformation steps.

    Steps are executed in order, each receiving the output
    of the previous step. The pipeline records a full report
    with per-step breakdowns and emits events.

    Args:
        steps: Ordered list of PipelineStep instances.
    """

    def __init__(self, steps: List[PipelineStep]) -> None:
        self._steps = steps
        self._last_report: PipelineReport | None = None

    @property
    def steps(self) -> List[PipelineStep]:
        """The pipeline steps."""
        return list(self._steps)

    @property
    def last_report(self) -> PipelineReport | None:
        """Report from the last run, if available."""
        return self._last_report

    def run(self, window: ContextWindow) -> ContextWindow:
        """Run the pipeline on a context window.

        Executes each step in order. The window is modified in
        place (blocks are replaced). Returns the same window
        for chaining.

        Args:
            window: The ContextWindow to optimize.

        Returns:
            The same ContextWindow (modified in place).
        """
        blocks = list(window.blocks)
        total_tokens_before = sum(b.token_count for b in blocks)

        step_reports = self._execute_all_steps(blocks, window)

        total_tokens_after = sum(b.token_count for b in blocks)
        self._last_report = self._build_report(
            step_reports, total_tokens_before, total_tokens_after, window
        )

        self._emit_pipeline_complete()

        return window

    def _execute_all_steps(
        self,
        blocks: List[ContextBlock],
        window: ContextWindow,
    ) -> List[StepReport]:
        """Execute all pipeline steps and collect reports."""
        step_reports: List[StepReport] = []

        for step in self._steps:
            report, blocks[:] = self._execute_step(step, blocks)
            step_reports.append(report)

        window.replace_blocks(blocks)
        return step_reports

    def _execute_step(
        self,
        step: PipelineStep,
        blocks: List[ContextBlock],
    ) -> Tuple[StepReport, List[ContextBlock]]:
        """Execute a single pipeline step and return its report."""
        tokens_before = sum(b.token_count for b in blocks)
        blocks_before_count = len(blocks)

        processed_blocks = step.process(blocks)

        tokens_after = sum(b.token_count for b in processed_blocks)
        blocks_after_count = len(processed_blocks)

        report = StepReport(
            step_name=step.name,
            blocks_modified=abs(blocks_before_count - blocks_after_count),
            blocks_removed=blocks_before_count - blocks_after_count,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_before - tokens_after,
        )

        emit(
            PipelineEventData(
                event=ContextEvent.PIPELINE_STEP,
                step_name=step.name,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
            )
        )

        return report, processed_blocks

    def _build_report(
        self,
        step_reports: List[StepReport],
        total_tokens_before: int,
        total_tokens_after: int,
        window: ContextWindow,
    ) -> PipelineReport:
        """Build the final pipeline report with cost delta."""
        cost_before = total_tokens_before * window.input_cost_per_mtok / 1_000_000
        cost_after = total_tokens_after * window.input_cost_per_mtok / 1_000_000

        return PipelineReport(
            steps=step_reports,
            total_tokens_before=total_tokens_before,
            total_tokens_after=total_tokens_after,
            total_tokens_saved=total_tokens_before - total_tokens_after,
            cost_delta=cost_before - cost_after,
        )

    def _emit_pipeline_complete(self) -> None:
        """Emit the PIPELINE_COMPLETE event."""
        assert self._last_report is not None
        emit(
            EventData(
                event=ContextEvent.PIPELINE_COMPLETE,
                details={
                    "steps": len(self._steps),
                    "tokens_saved": self._last_report.total_tokens_saved,
                    "cost_delta": self._last_report.cost_delta,
                },
            )
        )
