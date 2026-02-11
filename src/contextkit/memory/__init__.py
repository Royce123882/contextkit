"""Memory management for contextkit.

Provides short-term and long-term memory with pluggable backends,
conversation trimming strategies, and sync wrappers.
"""

from contextkit.memory.in_memory_backend import InMemoryBackend
from contextkit.memory.long_term import LongTermMemory
from contextkit.memory.record import (
    MemoryBackend,
    MemoryRecord,
)
from contextkit.memory.short_term import ShortTermMemory, trim_conversation
from contextkit.memory.sqlite_backend import SQLiteBackend

__all__ = [
    "InMemoryBackend",
    "LongTermMemory",
    "MemoryBackend",
    "MemoryRecord",
    "SQLiteBackend",
    "ShortTermMemory",
    "trim_conversation",
]
