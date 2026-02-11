"""Context pipeline for contextkit.

Provides a pipeline of transformation steps (trim, deduplicate,
filter, reorder, compact) with full mutation tracking and
observability.
"""

from contextkit.pipeline.base import PipelineReport, PipelineStep, StepReport
from contextkit.pipeline.compact import CompactStep
from contextkit.pipeline.deduplicate import DeduplicateStep
from contextkit.pipeline.filter import FilterStep
from contextkit.pipeline.pipeline import ContextPipeline
from contextkit.pipeline.reorder import ReorderStep
from contextkit.pipeline.trim import TrimStep

__all__ = [
    "CompactStep",
    "ContextPipeline",
    "DeduplicateStep",
    "FilterStep",
    "PipelineReport",
    "PipelineStep",
    "ReorderStep",
    "StepReport",
    "TrimStep",
]
