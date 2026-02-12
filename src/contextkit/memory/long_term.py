"""Long-term memory with pluggable backends.

Provides persistent memory storage and retrieval with automatic
Origin population. Supports any backend implementing the
MemoryBackend protocol.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("contextkit")

from contextkit.constants import (
    DEFAULT_IMPORTANCE,
    DEFAULT_TOP_K,
    PRIORITY_LONG_TERM_MEMORY,
)
from contextkit.core import BlockType, ContextBlock
from contextkit.memory.in_memory_backend import InMemoryBackend
from contextkit.memory.record import (
    MemoryBackend,
    MemoryRecord,
)
from contextkit.observe.provenance import Origin


class LongTermMemory:
    """Persistent memory store with pluggable backends.

    Wraps a MemoryBackend to provide a high-level API for storing
    and retrieving memories. Retrieved memories are automatically
    converted to ContextBlocks with Origin populated.

    Args:
        backend: A MemoryBackend instance. Defaults to InMemoryBackend.
    """

    def __init__(
        self,
        backend: MemoryBackend | None = None,
    ) -> None:
        self._backend: MemoryBackend = backend or InMemoryBackend()

    @property
    def backend(self) -> MemoryBackend:
        """The underlying memory backend."""
        return self._backend

    async def store(
        self,
        key: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
        tags: List[str] | None = None,
        importance: float = DEFAULT_IMPORTANCE,
    ) -> MemoryRecord:
        """Store a memory record.

        Args:
            key: Unique identifier.
            content: The content to store.
            metadata: Optional key-value pairs.
            tags: Optional categorization tags.
            importance: Importance score (0.0-1.0).

        Returns:
            The stored MemoryRecord.
        """
        logger.debug("Storing memory '%s' (importance=%.2f)", key, importance)
        return await self._backend.store(
            key=key,
            content=content,
            metadata=metadata,
            tags=tags,
            importance=importance,
        )

    async def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        tags: List[str] | None = None,
    ) -> List[MemoryRecord]:
        """Retrieve memories matching a query.

        Args:
            query: The search query.
            top_k: Maximum results to return.
            tags: Optional tag filter.

        Returns:
            List of matching MemoryRecords.
        """
        logger.info("Retrieving memories: query=%r, top_k=%d", query[:50] if query else "", top_k)
        return await self._backend.retrieve(query=query, top_k=top_k, tags=tags)

    async def retrieve_as_blocks(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        tags: List[str] | None = None,
        priority: int = PRIORITY_LONG_TERM_MEMORY,
    ) -> List[ContextBlock]:
        """Retrieve memories and convert to ContextBlocks.

        Each block has Origin auto-populated with the query, tags,
        and record metadata.

        Args:
            query: The search query.
            top_k: Maximum results to return.
            tags: Optional tag filter.
            priority: Priority for the created blocks.

        Returns:
            List of ContextBlocks with populated Origin.
        """
        records = await self.retrieve(query=query, top_k=top_k, tags=tags)
        blocks: List[ContextBlock] = []
        for record in records:
            origin = Origin(
                source="memory",
                details={
                    "key": record.key,
                    "query": query,
                    "tags": record.tags,
                    "importance": record.importance,
                },
            )
            block = ContextBlock(
                type=BlockType.LONG_TERM_MEMORY,
                content=record.content,
                priority=priority,
                name=f"memory_{record.key}",
                origin=origin,
                metadata=record.metadata,
            )
            blocks.append(block)
        return blocks

    async def delete(self, key: str) -> bool:
        """Delete a memory by key.

        Args:
            key: The record key to delete.

        Returns:
            True if deleted, False if not found.
        """
        return await self._backend.delete(key)

    async def list_records(self, tags: List[str] | None = None) -> List[MemoryRecord]:
        """List all memories, optionally filtered by tags.

        Args:
            tags: Optional tag filter.

        Returns:
            List of all matching MemoryRecords.
        """
        return await self._backend.list_records(tags=tags)
