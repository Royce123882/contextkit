"""Compaction artifact storage for contextkit.

Provides pluggable storage backends for persisting original content
before LLM-based compaction.  The ``CompactionStore`` protocol defines
the interface; ``LocalCompactionStore`` is the zero-config default.

S3 backend (requires optional dependency)::

    pip install contextkit[s3]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from contextkit.compaction.store import CompactionStore, LocalCompactionStore

if TYPE_CHECKING:
    from contextkit.compaction.s3_store import S3CompactionStore

__all__ = [
    "CompactionStore",
    "LocalCompactionStore",
    "S3CompactionStore",
]


def __getattr__(name: str) -> type:
    """Lazy-import S3 backend to avoid hard dependency on aiobotocore."""
    if name == "S3CompactionStore":
        from contextkit.compaction.s3_store import S3CompactionStore

        return S3CompactionStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
