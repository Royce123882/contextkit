"""Handoff package data model for agent-to-agent context transfer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from contextkit.core import ContextBlock

if TYPE_CHECKING:
    from contextkit.scope.scratchpad import Scratchpad


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
