"""Context pipeline for contextkit.

Provides a pipeline of transformation steps (trim, deduplicate,
filter, reorder, compact, rag_compress, mask, prune_stale,
strip_thinking) with full mutation tracking and observability.

Also provides a ``PipelineBuilder`` for fluent pipeline construction.
"""

from contextkit.pipeline.base import PipelineReport, PipelineStep, StepReport
from contextkit.pipeline.builder import PipelineBuilder
from contextkit.pipeline.compact import CompactStep
from contextkit.pipeline.deduplicate import DeduplicateStep
from contextkit.pipeline.filter import FilterStep
from contextkit.pipeline.mask import MaskStep
from contextkit.pipeline.pipeline import ContextPipeline
from contextkit.pipeline.prune_stale import PruneStaleStep
from contextkit.pipeline.rag_compress import RAGCompressStep
from contextkit.pipeline.reorder import ReorderStep
from contextkit.pipeline.strip_thinking import StripThinkingStep
from contextkit.pipeline.trim import TrimStep

__all__ = [
    "CompactStep",
    "ContextPipeline",
    "DeduplicateStep",
    "FilterStep",
    "MaskStep",
    "PipelineBuilder",
    "PipelineReport",
    "PipelineStep",
    "PruneStaleStep",
    "RAGCompressStep",
    "ReorderStep",
    "StepReport",
    "StripThinkingStep",
    "TrimStep",
]
