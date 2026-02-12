"""Synchronous wrappers for async APIs.

Provides sync versions of async memory and retrieval operations
for scripts, notebooks, and non-async codebases.

Detects whether a running event loop already exists and raises a
clear error instead of silently deadlocking.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, cast

from contextkit.constants import (
    DEFAULT_IMPORTANCE,
    DEFAULT_TOP_K,
    PRIORITY_LONG_TERM_MEMORY,
)
from contextkit.core import ContextBlock
from contextkit.exceptions import SyncInAsyncError
from contextkit.memory.long_term import (
    LongTermMemory as AsyncLongTermMemory,
)
from contextkit.memory.record import (
    MemoryBackend,
    MemoryRecord,
)


def _run_sync(coro: Any) -> Any:
    """Run an async coroutine synchronously.

    Raises SyncInAsyncError if called from within a running event
    loop, where it would deadlock with ``asyncio.run()``.

    Args:
        coro: The coroutine to run.

    Returns:
        The coroutine's return value.

    Raises:
        SyncInAsyncError: If an event loop is already running.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        raise SyncInAsyncError()

    return asyncio.run(coro)


class LongTermMemory:
    """Synchronous wrapper around the async LongTermMemory.

    Args:
        backend: A MemoryBackend instance. Defaults to InMemoryBackend.
    """

    def __init__(
        self,
        backend: MemoryBackend | None = None,
    ) -> None:
        self._async = AsyncLongTermMemory(backend=backend)

    def store(
        self,
        key: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
        tags: List[str] | None = None,
        importance: float = DEFAULT_IMPORTANCE,
    ) -> MemoryRecord:
        """Store a memory record (sync).

        Args:
            key: Unique identifier for the record.
            content: The content string to store.
            metadata: Optional key-value pairs.
            tags: Optional categorization tags.
            importance: Importance score (0.0-1.0).

        Returns:
            The stored MemoryRecord.
        """
        return cast(
            MemoryRecord,
            _run_sync(
                self._async.store(
                    key=key,
                    content=content,
                    metadata=metadata,
                    tags=tags,
                    importance=importance,
                )
            ),
        )

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        tags: List[str] | None = None,
    ) -> List[MemoryRecord]:
        """Retrieve matching records (sync).

        Args:
            query: The search query string.
            top_k: Maximum number of records to return.
            tags: Optional tag filter.

        Returns:
            List of matching MemoryRecords.
        """
        return cast(
            List[MemoryRecord],
            _run_sync(self._async.retrieve(query=query, top_k=top_k, tags=tags)),
        )

    def retrieve_as_blocks(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        tags: List[str] | None = None,
        priority: int = PRIORITY_LONG_TERM_MEMORY,
    ) -> List[ContextBlock]:
        """Retrieve as ContextBlocks (sync).

        Args:
            query: The search query string.
            top_k: Maximum number of records to return.
            tags: Optional tag filter.
            priority: Block priority for returned blocks.

        Returns:
            List of ContextBlocks with retrieved content.
        """
        return cast(
            List[ContextBlock],
            _run_sync(
                self._async.retrieve_as_blocks(
                    query=query,
                    top_k=top_k,
                    tags=tags,
                    priority=priority,
                )
            ),
        )

    def delete(self, key: str) -> bool:
        """Delete a record by key (sync).

        Args:
            key: The unique identifier of the record to delete.

        Returns:
            True if the record was found and deleted.
        """
        return cast(bool, _run_sync(self._async.delete(key)))

    def list_records(self, tags: List[str] | None = None) -> List[MemoryRecord]:
        """List all records (sync).

        Args:
            tags: Optional tag filter.

        Returns:
            List of matching MemoryRecords.
        """
        return cast(
            List[MemoryRecord],
            _run_sync(self._async.list_records(tags=tags)),
        )
