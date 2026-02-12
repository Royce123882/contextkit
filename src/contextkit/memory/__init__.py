"""Memory management for contextkit.

Provides short-term and long-term memory with pluggable backends,
conversation trimming strategies, and sync wrappers.

PostgresBackend is lazily imported to avoid a hard dependency on
asyncpg.  It shares the same driver as PgvectorRetriever, so users
who already use pgvector for RAG can reuse the same Postgres instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from contextkit.memory.in_memory_backend import InMemoryBackend
from contextkit.memory.long_term import LongTermMemory
from contextkit.memory.record import (
    MemoryBackend,
    MemoryRecord,
)
from contextkit.memory.short_term import ShortTermMemory, trim_conversation
from contextkit.memory.sqlite_backend import SQLiteBackend

if TYPE_CHECKING:
    from contextkit.memory.postgres_backend import PostgresBackend

__all__ = [
    "InMemoryBackend",
    "LongTermMemory",
    "MemoryBackend",
    "MemoryRecord",
    "PostgresBackend",
    "SQLiteBackend",
    "ShortTermMemory",
    "trim_conversation",
]

_LAZY_IMPORTS = {
    "PostgresBackend": "contextkit.memory.postgres_backend",
}


def __getattr__(name: str) -> object:
    """Lazy-import production backends to avoid hard dependencies."""
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
