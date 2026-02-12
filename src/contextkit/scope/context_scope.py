"""Isolated context scope for a single agent."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from contextkit.core import BudgetExceededError, ContextWindow
from contextkit.observe.context_timeline import ContextTimeline
from contextkit.scope.handoff_package import HandoffPackage
from contextkit.scope.scratchpad import Scratchpad
from contextkit.scope.shared_memory import SharedMemory

logger = logging.getLogger("contextkit")


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
            block_names = [block.display_name for block in self._window.blocks]
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
                    logger.warning(
                        "Skipped shared block '%s': exceeds budget", name
                    )

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
            blocks = [
                block for block in self._window.blocks
                if block.display_name in block_names
            ]
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
                logger.warning(
                    "Skipped handoff block '%s': exceeds budget",
                    block.display_name,
                )

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
