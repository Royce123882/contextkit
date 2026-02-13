"""Fluent builder for constructing context pipelines.

Provides a discoverable, IDE-autocomplete-friendly API for
assembling pipelines step by step. Access via
``ContextPipeline.builder()``.
"""

from __future__ import annotations

from typing import Any, List

from contextkit.pipeline.base import PipelineStep
from contextkit.pipeline.deduplicate import DeduplicateStep
from contextkit.pipeline.filter import FilterStep
from contextkit.pipeline.mask import MaskStep
from contextkit.pipeline.pipeline import ContextPipeline
from contextkit.pipeline.prune_stale import PruneStaleStep
from contextkit.pipeline.reorder import ReorderStep
from contextkit.pipeline.strip_thinking import StripThinkingStep
from contextkit.pipeline.trim import TrimStep


class PipelineBuilder:
    """Fluent builder for constructing ContextPipeline instances.

    Provides a discoverable, IDE-autocomplete-friendly API for
    assembling pipelines step by step.

    Example::

        pipeline = (
            ContextPipeline.builder()
            .deduplicate(threshold=0.85)
            .filter(min_relevance=0.3)
            .trim(max_tokens=100_000, query="user question")
            .reorder("prefix_stable")
            .build()
        )
    """

    def __init__(self) -> None:
        self._steps: List[PipelineStep] = []
        self._capture_snapshots = False

    def deduplicate(
        self,
        threshold: float = 0.85,
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add a deduplication step.

        Args:
            threshold: Similarity threshold for near-duplicate detection.
            **kwargs: Additional arguments passed to DeduplicateStep.

        Returns:
            This builder for chaining.
        """
        self._steps.append(DeduplicateStep(similarity_threshold=threshold, **kwargs))
        return self

    def filter(
        self,
        min_relevance: float = 0.3,
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add a filter step.

        Args:
            min_relevance: Minimum relevance score to keep a block.
            **kwargs: Additional arguments passed to FilterStep.

        Returns:
            This builder for chaining.
        """
        self._steps.append(FilterStep(min_relevance=min_relevance, **kwargs))
        return self

    def trim(
        self,
        max_tokens: int | None = None,
        query: str | None = None,
        min_priority: int = 0,
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add a trim step.

        Args:
            max_tokens: Token budget to enforce.
            query: Optional query for relevance-aware trimming.
            min_priority: Minimum priority threshold.
            **kwargs: Additional arguments passed to TrimStep.

        Returns:
            This builder for chaining.
        """
        self._steps.append(
            TrimStep(max_tokens=max_tokens, query=query, min_priority=min_priority, **kwargs)
        )
        return self

    def reorder(
        self,
        strategy: str = "important_edges",
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add a reorder step.

        Args:
            strategy: Reordering strategy ("important_edges" or "prefix_stable").
            **kwargs: Additional arguments passed to ReorderStep.

        Returns:
            This builder for chaining.
        """
        self._steps.append(ReorderStep(strategy=strategy, **kwargs))
        return self

    def mask(
        self,
        window: int = 3,
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add a mask (sliding window) step.

        Args:
            window: Number of recent observations to keep.
            **kwargs: Additional arguments passed to MaskStep.

        Returns:
            This builder for chaining.
        """
        self._steps.append(MaskStep(window=window, **kwargs))
        return self

    def prune_stale(
        self,
        max_age_turns: int | None = None,
        keep_last_per_tool: int = 1,
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add a stale tool output pruning step.

        Args:
            max_age_turns: Maximum turn age before removal. If None,
                only superseded outputs are removed.
            keep_last_per_tool: Always keep the N most recent outputs
                per tool name.
            **kwargs: Additional arguments passed to PruneStaleStep.

        Returns:
            This builder for chaining.
        """
        self._steps.append(
            PruneStaleStep(
                max_age_turns=max_age_turns,
                keep_last_per_tool=keep_last_per_tool,
                **kwargs,
            )
        )
        return self

    def strip_thinking(
        self,
        patterns: list[str] | None = None,
        strip_empty: bool = True,
        **kwargs: Any,
    ) -> "PipelineBuilder":
        """Add a reasoning trace stripping step.

        Args:
            patterns: Regex patterns to strip. Defaults to common
                wrappers (``<thinking>``, ``<scratchpad>``, etc.).
            strip_empty: Remove blocks that become empty after stripping.
            **kwargs: Additional arguments passed to StripThinkingStep.

        Returns:
            This builder for chaining.
        """
        self._steps.append(
            StripThinkingStep(patterns=patterns, strip_empty=strip_empty, **kwargs)
        )
        return self

    def step(self, custom_step: PipelineStep) -> "PipelineBuilder":
        """Add a custom pipeline step.

        Args:
            custom_step: Any PipelineStep instance.

        Returns:
            This builder for chaining.
        """
        self._steps.append(custom_step)
        return self

    def capture_snapshots(self, enabled: bool = True) -> "PipelineBuilder":
        """Enable or disable intermediate snapshot capture.

        Args:
            enabled: Whether to capture snapshots between steps.

        Returns:
            This builder for chaining.
        """
        self._capture_snapshots = enabled
        return self

    def build(self) -> "ContextPipeline":
        """Build and return the configured ContextPipeline.

        Returns:
            A ContextPipeline with the configured steps.

        Raises:
            ValueError: If no steps have been added.
        """
        if not self._steps:
            raise ValueError(
                "Cannot build an empty pipeline. Add at least one step "
                "(e.g., .deduplicate(), .trim(), .reorder())."
            )
        return ContextPipeline(
            steps=list(self._steps),
            capture_snapshots=self._capture_snapshots,
        )
