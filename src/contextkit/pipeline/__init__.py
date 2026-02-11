"""Context pipeline for contextkit.

Provides a pipeline of transformation steps (trim, deduplicate,
filter, reorder, compact) with full mutation tracking and
observability.
"""

from contextkit.pipeline.steps import (
    CompactStep,
    ContextPipeline,
    DeduplicateStep,
    FilterStep,
    PipelineReport,
    PipelineStep,
    ReorderStep,
    StepReport,
    TrimStep,
)

__all__ = [
    "ContextPipeline",
    "PipelineStep",
    "TrimStep",
    "DeduplicateStep",
    "FilterStep",
    "ReorderStep",
    "CompactStep",
    "PipelineReport",
    "StepReport",
]
