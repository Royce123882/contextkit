"""Multi-agent context scoping.

Provides ContextScope for isolating context per agent, SharedMemory
for cross-agent memory, handoff protocol for context transfer,
and Scratchpad for agent working notes.
"""

from contextkit.scope.context_scope import ContextScope
from contextkit.scope.handoff_package import HandoffPackage
from contextkit.scope.scratchpad import Scratchpad
from contextkit.scope.shared_memory import SharedMemory

__all__ = [
    "ContextScope",
    "HandoffPackage",
    "Scratchpad",
    "SharedMemory",
]
