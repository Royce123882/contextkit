"""Context pipeline orchestrator.

Runs an ordered sequence of pipeline steps on a context window,
collecting reports and emitting events for each step.
"""

from __future__ import annotations

import copy
import logging
from typing import List, Tuple

from contextkit.core import ContextBlock, ContextWindow
from contextkit.observe.events import (
    ContextEvent,
    EventData,
    PipelineEventData,
    emit,
)
from contextkit.pipeline.base import PipelineReport, PipelineStep, StepReport
from contextkit.pipeline.deduplicate import DeduplicateStep
from contextkit.pipeline.filter import FilterStep
from contextkit.pipeline.mask import MaskStep
from contextkit.pipeline.reorder import ReorderStep
from contextkit.pipeline.trim import TrimStep

logger = logging.getLogger("contextkit")


class ContextPipeline:
    """Named, ordered pipeline of transformation steps.

    Steps are executed in order, each receiving the output
    of the previous step. The pipeline records a full report
    with per-step breakdowns and emits events.

    Args:
        steps: Ordered list of PipelineStep instances.
    """

    def __init__(
        self,
        steps: List[PipelineStep],
        capture_snapshots: bool = False,
    ) -> None:
        self._steps = steps
        self._last_report: PipelineReport | None = None
        self._capture_snapshots = capture_snapshots
        self._snapshots: List[List[ContextBlock]] = []

    # ------------------------------------------------------------------
    # Preset factory methods
    # ------------------------------------------------------------------

    @classmethod
    def balanced(
        cls,
        max_tokens: int | None = None,
        query: str | None = None,
    ) -> "ContextPipeline":
        """Create a balanced pipeline with sensible defaults.

        Deduplicates near-duplicates, filters low-relevance content,
        trims to budget, and reorders for KV-cache efficiency.

        Args:
            max_tokens: Token budget for the trim step.
            query: Optional query for relevance-aware trimming.

        Returns:
            A ContextPipeline with balanced step configuration.
        """
        steps: List[PipelineStep] = [
            DeduplicateStep(similarity_threshold=0.85),
            FilterStep(min_relevance=0.3),
            TrimStep(max_tokens=max_tokens, query=query),
            ReorderStep(strategy="prefix_stable"),
        ]
        return cls(steps)

    @classmethod
    def aggressive(
        cls,
        max_tokens: int | None = None,
        query: str | None = None,
    ) -> "ContextPipeline":
        """Create an aggressive pipeline for maximum compression.

        Applies tight deduplication, strict filtering, budget trimming,
        observation masking, and cache-friendly reordering.

        To include LLM-based compaction, add a ``CompactStep`` with
        your LLM callable to the pipeline manually.

        Args:
            max_tokens: Token budget for the trim step.
            query: Optional query for relevance-aware trimming.

        Returns:
            A ContextPipeline with aggressive compression settings.
        """
        steps: List[PipelineStep] = [
            DeduplicateStep(similarity_threshold=0.7),
            FilterStep(min_relevance=0.5),
            TrimStep(max_tokens=max_tokens, query=query),
            MaskStep(window=3),
            ReorderStep(strategy="prefix_stable"),
        ]
        return cls(steps)

    @classmethod
    def conservative(
        cls,
        max_tokens: int | None = None,
    ) -> "ContextPipeline":
        """Create a conservative pipeline with minimal changes.

        Only deduplicates near-exact matches and trims to budget.
        Preserves most content with minimal information loss.

        Args:
            max_tokens: Token budget for the trim step.

        Returns:
            A ContextPipeline with conservative settings.
        """
        steps: List[PipelineStep] = [
            DeduplicateStep(similarity_threshold=0.95),
            TrimStep(max_tokens=max_tokens),
            ReorderStep(strategy="important_edges"),
        ]
        return cls(steps)

    @property
    def steps(self) -> List[PipelineStep]:
        """The pipeline steps."""
        return list(self._steps)

    @property
    def last_report(self) -> PipelineReport | None:
        """Report from the last run, if available."""
        return self._last_report

    @property
    def snapshots(self) -> List[List[ContextBlock]]:
        """Intermediate block states between pipeline steps.

        Only populated if ``capture_snapshots=True`` was passed to the
        constructor. Each entry is a deep copy of the block list as
        it existed before the corresponding step ran.

        Returns:
            List of block lists (one per executed step).
        """
        return list(self._snapshots)

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
        step_names = ", ".join(s.name for s in self._steps)
        logger.info("Running pipeline (%d steps: %s)", len(self._steps), step_names)

        blocks = list(window.blocks)
        total_tokens_before = sum(b.token_count for b in blocks)

        step_reports = self._execute_all_steps(blocks, window)

        total_tokens_after = sum(b.token_count for b in blocks)
        self._last_report = self._build_report(
            step_reports, total_tokens_before, total_tokens_after, window
        )

        self._emit_pipeline_complete()

        saved = total_tokens_before - total_tokens_after
        logger.info(
            "Pipeline complete: %s -> %s tokens (saved %s)",
            f"{total_tokens_before:,}",
            f"{total_tokens_after:,}",
            f"{saved:,}",
        )

        return window

    async def arun(self, window: ContextWindow) -> ContextWindow:
        """Async version of :meth:`run` with true async step execution.

        Uses each step's ``async_process()`` method, allowing steps
        that need async I/O (e.g. LLM-based summarization) to yield
        control properly.

        Args:
            window: The ContextWindow to optimize.

        Returns:
            The same ContextWindow (modified in place).
        """
        step_names = ", ".join(s.name for s in self._steps)
        logger.info(
            "Running async pipeline (%d steps: %s)", len(self._steps), step_names
        )

        blocks = list(window.blocks)
        total_tokens_before = sum(b.token_count for b in blocks)

        step_reports: List[StepReport] = []
        for step in self._steps:
            if not step.should_run(blocks):
                logger.debug("Skipping step '%s' (guard returned False)", step.name)
                continue
            report, blocks = await self._execute_step_async(step, blocks)
            step_reports.append(report)

        window.replace_blocks(blocks)
        total_tokens_after = sum(b.token_count for b in blocks)

        self._last_report = self._build_report(
            step_reports, total_tokens_before, total_tokens_after, window
        )
        self._emit_pipeline_complete()

        saved = total_tokens_before - total_tokens_after
        logger.info(
            "Async pipeline complete: %s -> %s tokens (saved %s)",
            f"{total_tokens_before:,}",
            f"{total_tokens_after:,}",
            f"{saved:,}",
        )
        return window

    def _execute_all_steps(
        self,
        blocks: List[ContextBlock],
        window: ContextWindow,
    ) -> List[StepReport]:
        """Execute all pipeline steps synchronously and collect reports."""
        self._snapshots.clear()
        step_reports: List[StepReport] = []

        for step in self._steps:
            if not step.should_run(blocks):
                logger.debug("Skipping step '%s' (guard returned False)", step.name)
                continue
            if self._capture_snapshots:
                self._snapshots.append(copy.deepcopy(blocks))
            report, blocks[:] = self._execute_step(step, blocks)
            step_reports.append(report)

        window.replace_blocks(blocks)
        return step_reports

    def _execute_step(
        self,
        step: PipelineStep,
        blocks: List[ContextBlock],
    ) -> Tuple[StepReport, List[ContextBlock]]:
        """Execute a single pipeline step synchronously and return its report."""
        tokens_before = sum(b.token_count for b in blocks)
        blocks_before_count = len(blocks)

        logger.debug("Running step '%s' on %d blocks", step.name, blocks_before_count)
        processed_blocks = step.process(blocks)

        tokens_after = sum(b.token_count for b in processed_blocks)
        blocks_after_count = len(processed_blocks)

        if tokens_before != tokens_after:
            logger.debug(
                "Step '%s': %s -> %s tokens, %d blocks -> %d blocks",
                step.name,
                f"{tokens_before:,}",
                f"{tokens_after:,}",
                blocks_before_count,
                blocks_after_count,
            )

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

    async def _execute_step_async(
        self,
        step: PipelineStep,
        blocks: List[ContextBlock],
    ) -> Tuple[StepReport, List[ContextBlock]]:
        """Execute a single pipeline step asynchronously.

        Uses the step's ``async_process()`` method for true async support.

        Args:
            step: The pipeline step to execute.
            blocks: Current list of context blocks.

        Returns:
            Tuple of (step report, processed blocks).
        """
        tokens_before = sum(b.token_count for b in blocks)
        blocks_before_count = len(blocks)

        logger.debug(
            "Running async step '%s' on %d blocks", step.name, blocks_before_count
        )
        processed_blocks = await step.async_process(blocks)

        tokens_after = sum(b.token_count for b in processed_blocks)
        blocks_after_count = len(processed_blocks)

        if tokens_before != tokens_after:
            logger.debug(
                "Step '%s': %s -> %s tokens, %d blocks -> %d blocks",
                step.name,
                f"{tokens_before:,}",
                f"{tokens_after:,}",
                blocks_before_count,
                blocks_after_count,
            )

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
