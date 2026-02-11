"""Memory management for contextkit.

Provides short-term and long-term memory with pluggable backends,
conversation trimming strategies, and sync wrappers.
"""

from contextkit.memory.backends import (
    InMemoryBackend,
    MemoryBackend,
    MemoryRecord,
)
from contextkit.memory.long_term import LongTermMemory
from contextkit.memory.short_term import ShortTermMemory, trim_conversation

__all__ = [
    "InMemoryBackend",
    "LongTermMemory",
    "MemoryBackend",
    "MemoryRecord",
    "ShortTermMemory",
    "trim_conversation",
]
