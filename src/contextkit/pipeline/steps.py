"""Pipeline steps for context optimization.

Provides a pipeline framework with named, ordered transformation
steps. Each step records mutations on the blocks it modifies so
every change is traceable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from contextkit._tokens import count as count_tokens
from contextkit.core import ContextBlock, ContextWindow
from contextkit.observe.events import (
    ContextEvent,
    EventData,
    PipelineEventData,
    emit,
)
from contextkit.observe.provenance import Mutation


class StepReport(BaseModel):
    """Report from a single pipeline step.

    Attributes:
        step_name: Name of the step.
        blocks_modified: Number of blocks modified.
        blocks_removed: Number of blocks removed.
        tokens_before: Total tokens before the step.
        tokens_after: Total tokens after the step.
        tokens_saved: Tokens saved by this step.
        details: Additional step-specific details.
    """

    step_name: str
    blocks_modified: int = 0
    blocks_removed: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    tokens_saved: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


class PipelineReport(BaseModel):
    """Report from a full pipeline run.

    Attributes:
        steps: Reports from each step.
        total_tokens_before: Total tokens before pipeline.
        total_tokens_after: Total tokens after pipeline.
        total_tokens_saved: Total tokens saved.
        cost_delta: Estimated cost savings.
    """

    steps: list[StepReport] = Field(default_factory=list)
    total_tokens_before: int = 0
    total_tokens_after: int = 0
    total_tokens_saved: int = 0
    cost_delta: float = 0.0


class PipelineStep(ABC):
    """Base class for pipeline steps.

    Subclasses must implement process() which takes a list of blocks
    and returns a modified list. Mutations should be recorded on
    the blocks themselves.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this step."""
        ...

    @abstractmethod
    def process(self, blocks: list[ContextBlock]) -> list[ContextBlock]:
        """Process blocks and return the modified list.

        Args:
            blocks: Input blocks to process.

        Returns:
            Modified list of blocks (may be shorter).
        """
        ...


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
        result: list[ContextBlock] = []
        removed: list[ContextBlock] = []

        # First pass: remove blocks below min_priority
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

        # Second pass: trim by budget if needed
        if self._max_tokens is not None:
            total = sum(b.token_count for b in result)
            if total > self._max_tokens:
                # Sort by priority ascending (remove lowest first)
                result.sort(key=lambda b: b.priority)
                kept: list[ContextBlock] = []
                budget_used = 0

                # Keep from highest priority
                for block in reversed(result):
                    if budget_used + block.token_count <= self._max_tokens:
                        kept.append(block)
                        budget_used += block.token_count
                    else:
                        block.mutations.append(
                            Mutation(
                                step=self.name,
                                action="removed",
                                detail=(
                                    "budget exceeded, removed to "
                                    f"fit {self._max_tokens} tokens"
                                ),
                                tokens_before=block.token_count,
                                tokens_after=0,
                            )
                        )
                        removed.append(block)

                result = list(reversed(kept))

        return result


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

    def process(self, blocks: list[ContextBlock]) -> list[ContextBlock]:
        """Filter blocks by relevance score or custom predicate."""
        result: list[ContextBlock] = []

        for block in blocks:
            keep = True

            # Check custom filter
            if self._filter_fn is not None:
                keep = self._filter_fn(block)

            # Check relevance score
            if (
                keep
                and self._min_relevance > 0
                and block.origin is not None
                and block.origin.relevance_score is not None
            ):
                if block.origin.relevance_score < self._min_relevance:
                    keep = False

            if keep:
                result.append(block)
            else:
                score = "N/A"
                if (
                    block.origin is not None
                    and block.origin.relevance_score is not None
                ):
                    score = str(block.origin.relevance_score)
                reason = f"relevance {score} below threshold {self._min_relevance}"
                block.mutations.append(
                    Mutation(
                        step=self.name,
                        action="removed",
                        detail=reason,
                        tokens_before=block.token_count,
                        tokens_after=0,
                    )
                )

        return result


class DeduplicateStep(PipelineStep):
    """Remove blocks with overlapping content.

    Uses word-overlap similarity to detect near-duplicates.
    When two blocks are similar, the one with lower priority
    is removed.

    Args:
        similarity_threshold: Overlap threshold (0.0-1.0).
    """

    def __init__(self, similarity_threshold: float = 0.8) -> None:
        self._threshold = similarity_threshold

    @property
    def name(self) -> str:
        """Return the step name."""
        return "DeduplicateStep"

    def process(self, blocks: list[ContextBlock]) -> list[ContextBlock]:
        """Remove duplicate blocks based on content similarity."""
        # Sort by priority descending so higher-priority kept
        sorted_blocks = sorted(blocks, key=lambda b: b.priority, reverse=True)
        result: list[ContextBlock] = []

        for block in sorted_blocks:
            if not isinstance(block.content, str):
                result.append(block)
                continue

            block_words = set(block.content.lower().split())
            is_duplicate = False

            for existing in result:
                if not isinstance(existing.content, str):
                    continue
                existing_words = set(existing.content.lower().split())
                if not block_words or not existing_words:
                    continue

                overlap = len(block_words & existing_words)
                similarity_score = overlap / min(len(block_words), len(existing_words))

                if similarity_score >= self._threshold:
                    is_duplicate = True
                    block.mutations.append(
                        Mutation(
                            step=self.name,
                            action="removed",
                            detail=(
                                f"overlaps with "
                                f'"{existing.display_name}" '
                                f"({similarity_score:.0%} "
                                f"similarity)"
                            ),
                            tokens_before=block.token_count,
                            tokens_after=0,
                        )
                    )
                    break

            if not is_duplicate:
                result.append(block)

        return result


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

        n = len(sorted_by_priority)
        result: list[ContextBlock | None] = [None] * n

        # Assign positions: alternate start and end
        start_idx = 0
        end_idx = n - 1

        for i, (orig_idx, block) in enumerate(sorted_by_priority):
            if i % 2 == 0:
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
                        detail=(f"position {orig_idx} -> position {new_pos}"),
                        tokens_before=block.token_count,
                        tokens_after=block.token_count,
                    )
                )

        return [b for b in result if b is not None]


class CompactStep(PipelineStep):
    """Compact verbose blocks by summarization.

    Uses a provided compaction function to summarize long blocks.
    Records full before/after content in the mutation log.

    Args:
        compactor: A function that takes content string and returns
            a shorter summary. If None, uses simple truncation.
        target_ratio: Target compression ratio (0.0-1.0).
            0.5 means aim for 50% of original size.
        min_tokens: Only compact blocks above this token count.
    """

    def __init__(
        self,
        compactor: Callable[[str], str] | None = None,
        target_ratio: float = 0.5,
        min_tokens: int = 100,
    ) -> None:
        self._compactor = compactor or self._default_compactor
        self._target_ratio = target_ratio
        self._min_tokens = min_tokens

    @property
    def name(self) -> str:
        """Return the step name."""
        return "CompactStep"

    def process(self, blocks: list[ContextBlock]) -> list[ContextBlock]:
        """Compact long blocks by truncation or custom compactor."""
        result: list[ContextBlock] = []

        for block in blocks:
            if not isinstance(block.content, str):
                result.append(block)
                continue

            tokens_before = block.token_count
            if tokens_before < self._min_tokens:
                result.append(block)
                continue

            before_content = block.content
            compacted = self._compactor(before_content)
            tokens_after = count_tokens(compacted)

            if tokens_after < tokens_before:
                # Create a new block with compacted content
                new_block = block.model_copy(update={"content": compacted})
                new_block.mutations.append(
                    Mutation(
                        step=self.name,
                        action="compacted",
                        detail=(f"{tokens_before:,} -> {tokens_after:,} tokens"),
                        tokens_before=tokens_before,
                        tokens_after=tokens_after,
                        before_content=before_content,
                        after_content=compacted,
                    )
                )
                result.append(new_block)
            else:
                result.append(block)

        return result

    def _default_compactor(self, content: str) -> str:
        """Default compaction: keep first N characters."""
        target_len = int(len(content) * self._target_ratio)
        if target_len >= len(content):
            return content
        # Try to break at a sentence boundary
        truncated = content[:target_len]
        last_period = truncated.rfind(".")
        if last_period > target_len * 0.5:
            return truncated[: last_period + 1]
        return truncated


class ContextPipeline:
    """Named, ordered pipeline of transformation steps.

    Steps are executed in order, each receiving the output
    of the previous step. The pipeline records a full report
    with per-step breakdowns and emits events.

    Args:
        steps: Ordered list of PipelineStep instances.
    """

    def __init__(self, steps: list[PipelineStep]) -> None:
        self._steps = steps
        self._last_report: PipelineReport | None = None

    @property
    def steps(self) -> list[PipelineStep]:
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

        step_reports: list[StepReport] = []

        for step in self._steps:
            tokens_before = sum(b.token_count for b in blocks)
            blocks_before_count = len(blocks)

            blocks = step.process(blocks)

            tokens_after = sum(b.token_count for b in blocks)
            blocks_after_count = len(blocks)

            report = StepReport(
                step_name=step.name,
                blocks_modified=abs(blocks_before_count - blocks_after_count),
                blocks_removed=(blocks_before_count - blocks_after_count),
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                tokens_saved=tokens_before - tokens_after,
            )
            step_reports.append(report)

            # Emit PIPELINE_STEP event
            emit(
                PipelineEventData(
                    event=ContextEvent.PIPELINE_STEP,
                    step_name=step.name,
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                )
            )

        total_tokens_after = sum(b.token_count for b in blocks)

        # Replace window blocks via public API
        window.replace_blocks(blocks)

        cost_before = total_tokens_before * window.input_cost_per_mtok / 1_000_000
        cost_after = total_tokens_after * window.input_cost_per_mtok / 1_000_000

        self._last_report = PipelineReport(
            steps=step_reports,
            total_tokens_before=total_tokens_before,
            total_tokens_after=total_tokens_after,
            total_tokens_saved=(total_tokens_before - total_tokens_after),
            cost_delta=cost_before - cost_after,
        )

        # Emit PIPELINE_COMPLETE event
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

        return window
