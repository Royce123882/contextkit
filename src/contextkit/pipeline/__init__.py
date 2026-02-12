"""Context pipeline for contextkit.

Provides a pipeline of transformation steps (trim, deduplicate,
filter, reorder, compact, compress, rag_compress, mask) with full
mutation tracking and observability.
"""

from contextkit.pipeline.base import PipelineReport, PipelineStep, StepReport
from contextkit.pipeline.compact import CompactStep
from contextkit.pipeline.compress import CompressStep
from contextkit.pipeline.deduplicate import DeduplicateStep
from contextkit.pipeline.filter import FilterStep
from contextkit.pipeline.mask import MaskStep
from contextkit.pipeline.pipeline import ContextPipeline
from contextkit.pipeline.rag_compress import RAGCompressStep
from contextkit.pipeline.reorder import ReorderStep
from contextkit.pipeline.trim import TrimStep

__all__ = [
    "CompactStep",
    "CompressStep",
    "ContextPipeline",
    "DeduplicateStep",
    "FilterStep",
    "MaskStep",
    "PipelineReport",
    "PipelineStep",
    "RAGCompressStep",
    "ReorderStep",
    "StepReport",
    "TrimStep",
]
