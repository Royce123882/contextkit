"""Shared memory for cross-agent block exchange."""

from __future__ import annotations

from typing import Dict, List

from contextkit.core import ContextBlock


class SharedMemory:
    """Memory blocks shared across multiple agents.

    A simple shared store where agents can publish and read
    named blocks. Thread-safe for basic use cases.
    """

    def __init__(self) -> None:
        self._blocks: Dict[str, ContextBlock] = {}

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
        self._blocks[name] = block

    def read(self, name: str) -> ContextBlock | None:
        """Read a block from shared memory.

        Args:
            name: Shared block name.

        Returns:
            The block, or None if not found.
        """
        return self._blocks.get(name)

    def list_blocks(self) -> List[str]:
        """List all shared block names."""
        return sorted(self._blocks.keys())

    def remove(self, name: str) -> bool:
        """Remove a block from shared memory.

        Args:
            name: Block name to remove.

        Returns:
            True if removed, False if not found.
        """
        if name in self._blocks:
            del self._blocks[name]
            return True
        return False

    def get_all(self) -> List[ContextBlock]:
        """Get all shared blocks."""
        return list(self._blocks.values())

    @property
    def block_count(self) -> int:
        """Number of shared blocks."""
        return len(self._blocks)
