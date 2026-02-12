"""contextkit -- Build context once. Send it anywhere. Know exactly what the model sees.

A context engineering SDK for building reliable AI agents.
Treats the context window as a first-class, engineerable artifact
with full observability, provenance tracking, and provider-agnostic
formatting.
"""

__version__ = "0.1.0"

from contextkit.assembler import AssemblyReport, BlockDecision, ContextAssembler
from contextkit.core import (
    BlockType,
    BudgetExceededError,
    ContextBlock,
    ContextWindow,
)
from contextkit.logging import configure_logging, enable_debug, silence
from contextkit.models import ModelSpec, UnknownModelError
from contextkit.observe.provenance import Mutation, Origin

__all__ = [
    "AssemblyReport",
    "BlockDecision",
    "BlockType",
    "BudgetExceededError",
    "ContextAssembler",
    "ContextBlock",
    "ContextWindow",
    "ModelSpec",
    "Mutation",
    "Origin",
    "UnknownModelError",
    "__version__",
    "configure_logging",
    "enable_debug",
    "silence",
]
