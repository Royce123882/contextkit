"""Context timeline tracking.

Records how a context window changes across turns in an agent loop.
Each turn stores a snapshot that can be inspected later.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Set

from pydantic import BaseModel, Field


class TurnSnapshot(BaseModel):
    """A snapshot of the context window at a single turn.

    Attributes:
        turn: Turn number (1-indexed).
        timestamp: When this snapshot was taken.
        token_count: Total tokens at this turn.
        max_tokens: Token budget.
        block_count: Number of blocks.
        block_names: Names of all blocks at this turn.
        blocks_added: Block names added since last turn.
        blocks_removed: Block names removed since last turn.
        blocks_mutated: Block names mutated since last turn.
        budget_percent: Budget utilization percentage.
        events: Any events that fired during this turn.
    """

    turn: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    token_count: int = 0
    max_tokens: int = 0
    block_count: int = 0
    block_names: List[str] = Field(default_factory=list)
    blocks_added: List[str] = Field(default_factory=list)
    blocks_removed: List[str] = Field(default_factory=list)
    blocks_mutated: List[str] = Field(default_factory=list)
    budget_percent: float = 0.0
    events: List[str] = Field(default_factory=list)


class ContextTimeline:
    """Tracks context window evolution across turns.

    Records snapshots at each turn so developers can see how
    context changed over time. Supports inspection at any
    past turn and full timeline export.
    """

    def __init__(self, max_tokens: int = 0) -> None:
        self._snapshots: List[TurnSnapshot] = []
        self._max_tokens = max_tokens
        self._current_turn = 0

    @property
    def snapshots(self) -> List[TurnSnapshot]:
        """All recorded snapshots."""
        return list(self._snapshots)

    @property
    def current_turn(self) -> int:
        """The current turn number."""
        return self._current_turn

    def record(
        self,
        token_count: int,
        block_names: List[str],
        events: List[str] | None = None,
    ) -> TurnSnapshot:
        """Record a snapshot for the current turn.

        Computes diffs from the previous snapshot to track
        additions, removals, and mutations.

        Args:
            token_count: Current total token count.
            block_names: Names of all blocks at this turn.
            events: Any events that occurred during this turn.

        Returns:
            The recorded TurnSnapshot.
        """
        self._current_turn += 1

        # Compute diffs from previous turn
        prev_names: Set[str] = set()
        if self._snapshots:
            prev_names = set(self._snapshots[-1].block_names)

        current_names = set(block_names)
        added = sorted(current_names - prev_names)
        removed = sorted(prev_names - current_names)

        budget_pct = (
            (token_count / self._max_tokens * 100) if self._max_tokens > 0 else 0.0
        )

        snapshot = TurnSnapshot(
            turn=self._current_turn,
            token_count=token_count,
            max_tokens=self._max_tokens,
            block_count=len(block_names),
            block_names=sorted(block_names),
            blocks_added=added,
            blocks_removed=removed,
            budget_percent=budget_pct,
            events=events or [],
        )
        self._snapshots.append(snapshot)
        return snapshot

    def snapshot_at(self, turn: int) -> TurnSnapshot | None:
        """Get the snapshot at a specific turn.

        Args:
            turn: Turn number (1-indexed).

        Returns:
            The TurnSnapshot, or None if not found.
        """
        for snapshot in self._snapshots:
            if snapshot.turn == turn:
                return snapshot
        return None

    def timeline_summary(self) -> str:
        """Render a human-readable timeline summary.

        Returns:
            A formatted string showing how context evolved.
        """
        lines: List[str] = []
        for snap in self._snapshots:
            parts: List[str] = [
                f"Turn {snap.turn:>3}:",
                f"{snap.token_count:>8,} tokens",
                f"({snap.budget_percent:.1f}%)",
            ]

            changes: List[str] = []
            if snap.blocks_added:
                for name in snap.blocks_added:
                    changes.append(f"+{name}")
            if snap.blocks_removed:
                for name in snap.blocks_removed:
                    changes.append(f"-{name}")
            if snap.events:
                for event in snap.events:
                    changes.append(event)

            if changes:
                parts.append("  " + "  ".join(changes))

            lines.append(" ".join(parts))

        return "\n".join(lines)

    def export(self, path: str) -> None:
        """Export full timeline to a JSON file.

        Args:
            path: File path to write the JSON export.
        """
        data = {
            "total_turns": self._current_turn,
            "max_tokens": self._max_tokens,
            "snapshots": [s.model_dump(mode="json") for s in self._snapshots],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def clear(self) -> None:
        """Clear all recorded snapshots."""
        self._snapshots.clear()
        self._current_turn = 0
