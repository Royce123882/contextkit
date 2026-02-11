"""Multi-agent context scoping.

Provides ContextScope for isolating context per agent, SharedMemory
for cross-agent memory, handoff protocol for context transfer,
and Scratchpad for agent working notes.
"""

from __future__ import annotations

from typing import Any, Dict, List

from contextkit.constants import PRIORITY_HANDOFF_METADATA
from contextkit.core import (
    BlockType,
    BudgetExceededError,
    ContextBlock,
    ContextWindow,
)
from contextkit.observe.provenance import Origin
from contextkit.observe.timeline import ContextTimeline


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
            True if deleted, False if not found.
        """
        if key in self._notes:
            del self._notes[key]
            return True
        return False

    def list_keys(self) -> List[str]:
        """List all note keys."""
        return sorted(self._notes.keys())

    def to_block(self, priority: int = PRIORITY_HANDOFF_METADATA) -> ContextBlock:
        """Convert all notes to a ContextBlock.

        Args:
            priority: Block priority.

        Returns:
            A ContextBlock of type SCRATCHPAD.
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


class HandoffPackage:
    """Structured context transfer between agents.

    Contains the blocks, scratchpad notes, and metadata that one
    agent passes to another during handoff.
    """

    def __init__(
        self,
        source_agent: str,
        target_agent: str,
        blocks: List[ContextBlock] | None = None,
        scratchpad: Scratchpad | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        self.source_agent = source_agent
        self.target_agent = target_agent
        self.blocks = blocks or []
        self.scratchpad = scratchpad
        self.metadata = metadata or {}

    @property
    def block_count(self) -> int:
        """Number of blocks in the handoff."""
        return len(self.blocks)


class ContextScope:
    """Isolated context scope for a single agent.

    Each agent gets its own ContextScope with a window, scratchpad,
    optional shared memory access, and optional timeline tracking.

    Args:
        agent_name: Name of the agent.
        window: The agent's context window.
        shared_memory: Optional shared memory to read from.
        track_history: Whether to record timeline snapshots.
    """

    def __init__(
        self,
        agent_name: str,
        window: ContextWindow,
        shared_memory: SharedMemory | None = None,
        track_history: bool = False,
    ) -> None:
        self._agent_name = agent_name
        self._window = window
        self._shared_memory = shared_memory
        self._scratchpad = Scratchpad()
        self._timeline: ContextTimeline | None = None

        if track_history:
            self._timeline = ContextTimeline(max_tokens=window.max_tokens)

    @property
    def agent_name(self) -> str:
        """The agent's name."""
        return self._agent_name

    @property
    def window(self) -> ContextWindow:
        """The agent's context window."""
        return self._window

    @property
    def scratchpad(self) -> Scratchpad:
        """The agent's scratchpad."""
        return self._scratchpad

    @property
    def shared_memory(self) -> SharedMemory | None:
        """Shared memory, if configured."""
        return self._shared_memory

    @property
    def timeline(self) -> ContextTimeline | None:
        """Timeline tracker, if enabled."""
        return self._timeline

    def record_turn(self, events: List[str] | None = None) -> None:
        """Record a timeline snapshot for the current turn.

        Args:
            events: Any events that occurred during this turn.
        """
        if self._timeline is not None:
            block_names = [b.display_name for b in self._window.blocks]
            self._timeline.record(
                token_count=self._window.token_count,
                block_names=block_names,
                events=events,
            )

    def import_shared(self, block_names: List[str] | None = None) -> int:
        """Import blocks from shared memory into the window.

        Args:
            block_names: Specific blocks to import.
                If None, imports all shared blocks.

        Returns:
            Number of blocks imported.
        """
        if self._shared_memory is None:
            return 0

        imported = 0
        blocks_to_import = block_names or self._shared_memory.list_blocks()

        for name in blocks_to_import:
            block = self._shared_memory.read(name)
            if block is not None:
                try:
                    self._window.add(block)
                    imported += 1
                except BudgetExceededError:
                    pass  # Skip blocks that exceed budget

        return imported

    def handoff(
        self,
        target_agent: str,
        block_names: List[str] | None = None,
        include_scratchpad: bool = True,
        metadata: Dict[str, Any] | None = None,
    ) -> HandoffPackage:
        """Create a handoff package for another agent.

        Args:
            target_agent: Name of the receiving agent.
            block_names: Specific blocks to hand off.
                If None, includes all blocks.
            include_scratchpad: Include scratchpad notes.
            metadata: Additional handoff metadata.

        Returns:
            A HandoffPackage ready for the target agent.
        """
        if block_names is not None:
            blocks = [b for b in self._window.blocks if b.display_name in block_names]
        else:
            blocks = list(self._window.blocks)

        return HandoffPackage(
            source_agent=self._agent_name,
            target_agent=target_agent,
            blocks=blocks,
            scratchpad=(self._scratchpad if include_scratchpad else None),
            metadata=metadata or {},
        )

    def receive_handoff(self, package: HandoffPackage) -> int:
        """Receive a handoff package from another agent.

        Adds the handed-off blocks to this agent's window.

        Args:
            package: The handoff package to receive.

        Returns:
            Number of blocks successfully added.
        """
        added = 0
        for block in package.blocks:
            try:
                self._window.add(block)
                added += 1
            except BudgetExceededError:
                pass  # Skip if budget exceeded

        # Import scratchpad notes
        if package.scratchpad is not None:
            for key in package.scratchpad.list_keys():
                content = package.scratchpad.read(key)
                if content is not None:
                    self._scratchpad.write(
                        f"from_{package.source_agent}_{key}",
                        content,
                    )

        return added
