"""Shared memory for cross-agent block exchange."""

from __future__ import annotations

import threading
from typing import Dict, List

from contextkit.core import ContextBlock


class SharedMemory:
    """Memory blocks shared across multiple agents.

    A simple shared store where agents can publish and read
    named blocks. Thread-safe via internal lock.
    """

    def __init__(self) -> None:
        self._blocks: Dict[str, ContextBlock] = {}
        self._lock = threading.Lock()

    def publish(
        self,
        name: str,
        block: ContextBlock,
    ) -> None:
        """Publish a block to shared memory.

        Args:
            name: Shared block name.
            block: The block to share.
        """
        with self._lock:
            self._blocks[name] = block

    def read(self, name: str) -> ContextBlock | None:
        """Read a block from shared memory.

        Args:
            name: Shared block name.

        Returns:
            The block, or None if not found.
        """
        with self._lock:
            return self._blocks.get(name)

    def list_blocks(self) -> List[str]:
        """List all shared block names."""
        with self._lock:
            return sorted(self._blocks.keys())

    def remove(self, name: str) -> bool:
        """Remove a block from shared memory.

        Args:
            name: Block name to remove.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            if name in self._blocks:
                del self._blocks[name]
                return True
            return False

    def get_all(self) -> List[ContextBlock]:
        """Get all shared blocks."""
        with self._lock:
            return list(self._blocks.values())

    @property
    def block_count(self) -> int:
        """Number of shared blocks."""
        with self._lock:
            return len(self._blocks)
