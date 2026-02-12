"""Assembly report data models."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class BlockDecision(BaseModel):
    """Records the assembler's decision about a single block.

    Attributes:
        block_name: Display name of the block.
        block_type: The BlockType value.
        tokens: Token count for this block.
        priority: Block priority.
        origin_summary: Human-readable origin summary.
        reason: Why it was excluded (None if included).
    """

    block_name: str = Field(description="Display name of the block.")
    block_type: str = Field(description="The BlockType value.")
    tokens: int = Field(description="Token count for this block.")
    priority: int = Field(description="Block priority.")
    origin_summary: str = Field(description="Human-readable origin summary.")
    reason: str | None = Field(default=None, description="Why it was excluded (None if included).")


class AssemblyReport(BaseModel):
    """Full report of an assembly operation.

    Attributes:
        included: Blocks that were added to the window.
        excluded: Blocks that were skipped (with reasons).
        total_tokens: Total tokens in the assembled window.
        budget_remaining: Tokens remaining after assembly.
        cost_estimate: Estimated input cost in USD.
    """

    included: List[BlockDecision] = Field(description="Blocks that were added to the window.")
    excluded: List[BlockDecision] = Field(description="Blocks that were skipped (with reasons).")
    total_tokens: int = Field(description="Total tokens in the assembled window.")
    budget_remaining: int = Field(description="Tokens remaining after assembly.")
    cost_estimate: float = Field(description="Estimated input cost in USD.")
