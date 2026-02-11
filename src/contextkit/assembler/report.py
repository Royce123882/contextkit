"""Assembly report data models."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel


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

    block_name: str
    block_type: str
    tokens: int
    priority: int
    origin_summary: str
    reason: str | None = None


class AssemblyReport(BaseModel):
    """Full report of an assembly operation.

    Attributes:
        included: Blocks that were added to the window.
        excluded: Blocks that were skipped (with reasons).
        total_tokens: Total tokens in the assembled window.
        budget_remaining: Tokens remaining after assembly.
        cost_estimate: Estimated input cost in USD.
    """

    included: List[BlockDecision]
    excluded: List[BlockDecision]
    total_tokens: int
    budget_remaining: int
    cost_estimate: float
