"""Scratchpad for agent working notes."""

from __future__ import annotations

from typing import Dict, List

from contextkit.constants import PRIORITY_HANDOFF_METADATA
from contextkit.core import BlockType, ContextBlock
from contextkit.observe.provenance import Origin


class Scratchpad:
    """Agent working notes for multi-step reasoning.

    A simple key-value store that can be converted into a
    ContextBlock. Notes are preserved across turns and can
    be inspected via the observability tools.
    """

    def __init__(self) -> None:
        self._notes: Dict[str, str] = {}

    def write(self, key: str, content: str) -> None:
        """Write a note.

        Args:
            key: Note identifier.
            content: Note content.
        """
        self._notes[key] = content

    def read(self, key: str) -> str | None:
        """Read a note.

        Args:
            key: Note identifier.

        Returns:
            Note content, or None if not found.
        """
        return self._notes.get(key)

    def delete(self, key: str) -> bool:
        """Delete a note.

        Args:
            key: Note identifier.

        Returns:
            True if the note was found and deleted.
        """
        if key in self._notes:
            del self._notes[key]
            return True
        return False

    def list_keys(self) -> List[str]:
        """List all note keys."""
        return sorted(self._notes.keys())

    def to_block(self, priority: int = PRIORITY_HANDOFF_METADATA) -> ContextBlock:
        """Convert notes to a ContextBlock.

        Args:
            priority: Block priority for the created block.

        Returns:
            A ContextBlock of type SCRATCHPAD with all notes.
        """
        content = "\n".join(f"[{k}] {v}" for k, v in sorted(self._notes.items()))
        return ContextBlock(
            type=BlockType.SCRATCHPAD,
            content=content or "(empty)",
            priority=priority,
            name="scratchpad",
            origin=Origin(
                source="scratchpad",
                details={"note_count": len(self._notes)},
            ),
        )

    def clear(self) -> None:
        """Clear all notes."""
        self._notes.clear()

    @property
    def note_count(self) -> int:
        """Number of notes."""
        return len(self._notes)
