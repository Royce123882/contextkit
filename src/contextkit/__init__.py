"""contextkit -- Build context once. Send it anywhere. Know exactly what the model sees.

A context engineering SDK for building reliable AI agents.
Treats the context window as a first-class, engineerable artifact
with full observability, provenance tracking, and provider-agnostic
formatting.
"""

__version__ = "0.1.0"

from contextkit.adapters import (
    AnthropicAdapter,
    BedrockAdapter,
    LiteLLMAdapter,
    OllamaAdapter,
    OpenAIAdapter,
)
from contextkit.assembler import AssemblyReport, BlockDecision, ContextAssembler
from contextkit.compaction import CompactionStore, LocalCompactionStore
from contextkit.core import (
    BlockType,
    BudgetExceededError,
    ContextBlock,
    ContextWindow,
)
from contextkit.files import FileContext
from contextkit.logging import configure_logging, enable_debug, silence
from contextkit.memory import LongTermMemory, ShortTermMemory
from contextkit.models import ModelSpec, UnknownModelError
from contextkit.observe import QualityScorer, SufficiencyChecker
from contextkit.observe.provenance import Mutation, Origin
from contextkit.pipeline import (
    CompactStep,
    ContextPipeline,
    DeduplicateStep,
    FilterStep,
    MaskStep,
    PipelineStep,
    RAGCompressStep,
    ReorderStep,
    TrimStep,
)
from contextkit.prompts import PromptManager
from contextkit.rag import RAGContext
from contextkit.scope import ContextScope, HandoffPackage, Scratchpad, SharedMemory
from contextkit.tools import ToolRegistry

__all__ = [
    "AnthropicAdapter",
    "AssemblyReport",
    "BedrockAdapter",
    "BlockDecision",
    "BlockType",
    "BudgetExceededError",
    "CompactStep",
    "CompactionStore",
    "ContextAssembler",
    "ContextBlock",
    "ContextPipeline",
    "ContextScope",
    "ContextWindow",
    "DeduplicateStep",
    "FileContext",
    "FilterStep",
    "HandoffPackage",
    "LiteLLMAdapter",
    "LocalCompactionStore",
    "LongTermMemory",
    "MaskStep",
    "ModelSpec",
    "Mutation",
    "OllamaAdapter",
    "OpenAIAdapter",
    "Origin",
    "PipelineStep",
    "PromptManager",
    "QualityScorer",
    "RAGCompressStep",
    "RAGContext",
    "ReorderStep",
    "Scratchpad",
    "SharedMemory",
    "ShortTermMemory",
    "SufficiencyChecker",
    "ToolRegistry",
    "TrimStep",
    "UnknownModelError",
    "__version__",
    "configure_logging",
    "enable_debug",
    "silence",
]
