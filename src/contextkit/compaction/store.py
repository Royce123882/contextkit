"""Compaction store protocol.

Defines the ``CompactionStore`` protocol for persisting original
numbered content so that ``[N]`` references in compacted summaries
can be resolved later.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class CompactionStore(Protocol):
    """Protocol for compaction artifact storage.

    Stores the original numbered content as markdown so that
    ``[N]`` references in the compacted output can be resolved.
    """

    async def save(
        self,
        key: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        """Save original content.

        Args:
            key: Unique identifier for this compaction artifact.
            content: The numbered markdown content to persist.
            metadata: Optional metadata to store alongside.

        Returns:
            A reference URI (file path or URL) for the stored artifact.
        """
        ...

    async def load(self, key: str) -> str | None:
        """Load original content by key.

        Args:
            key: The artifact identifier.

        Returns:
            The stored content, or None if not found.
        """
        ...

    async def delete(self, key: str) -> bool:
        """Delete a stored artifact.

        Args:
            key: The artifact identifier.

        Returns:
            True if the artifact was deleted, False if not found.
        """
        ...

    async def list_keys(self) -> List[str]:
        """List all stored artifact keys.

        Returns:
            A list of artifact keys currently in the store.
        """
        ...
