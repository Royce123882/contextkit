"""Synchronous wrappers for async APIs.

Provides sync versions of async memory and retrieval operations
for scripts, notebooks, and non-async codebases.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from contextkit.core import ContextBlock
from contextkit.memory.backends import (
    MemoryBackend,
    MemoryRecord,
)
from contextkit.memory.long_term import (
    LongTermMemory as AsyncLongTermMemory,
)


def _run_sync(coro: Any) -> Any:
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
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
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        importance: float = 0.5,
    ) -> MemoryRecord:
        """Store a memory record (sync)."""
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
        top_k: int = 5,
        tags: list[str] | None = None,
    ) -> list[MemoryRecord]:
        """Retrieve matching records (sync)."""
        return cast(
            list[MemoryRecord],
            _run_sync(self._async.retrieve(query=query, top_k=top_k, tags=tags)),
        )

    def retrieve_as_blocks(
        self,
        query: str,
        top_k: int = 5,
        tags: list[str] | None = None,
        priority: int = 60,
    ) -> list[ContextBlock]:
        """Retrieve as ContextBlocks (sync)."""
        return cast(
            list[ContextBlock],
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
        """Delete a record by key (sync)."""
        return cast(bool, _run_sync(self._async.delete(key)))

    def list_records(self, tags: list[str] | None = None) -> list[MemoryRecord]:
        """List all records (sync)."""
        return cast(
            list[MemoryRecord],
            _run_sync(self._async.list_records(tags=tags)),
        )
