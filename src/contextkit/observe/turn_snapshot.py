"""Turn snapshot data model for context timeline tracking."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

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

    turn: int = Field(description="Turn number (1-indexed).")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When this snapshot was taken.")
    token_count: int = Field(default=0, description="Total tokens at this turn.")
    max_tokens: int = Field(default=0, description="Token budget.")
    block_count: int = Field(default=0, description="Number of blocks.")
    block_names: List[str] = Field(default_factory=list, description="Names of all blocks at this turn.")
    blocks_added: List[str] = Field(default_factory=list, description="Block names added since last turn.")
    blocks_removed: List[str] = Field(default_factory=list, description="Block names removed since last turn.")
    blocks_mutated: List[str] = Field(default_factory=list, description="Block names mutated since last turn.")
    budget_percent: float = Field(default=0.0, description="Budget utilization percentage.")
    events: List[str] = Field(default_factory=list, description="Any events that fired during this turn.")
