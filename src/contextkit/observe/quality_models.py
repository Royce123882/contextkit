"""Data models for context quality scoring.

Separates data models from the scoring implementation
to follow the Single Responsibility Principle.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel


class PositionScore(BaseModel):
    """Quality score for a single block based on its position.

    Attributes:
        block_name: Display name of the scored block.
        position: Zero-based position in the block sequence.
        total_blocks: Total number of blocks in the window.
        attention_weight: Estimated attention weight at this
            position (1.0 = high attention, 0.0 = low).
        priority: The block's priority value.
        risk: Risk level for information loss at this position.
            One of ``"low"``, ``"medium"``, or ``"high"``.
    """

    block_name: str
    position: int
    total_blocks: int
    attention_weight: float
    priority: int
    risk: str


class QualityReport(BaseModel):
    """Aggregate quality report for a context window.

    Attributes:
        overall_score: Weighted quality score (0.0-1.0) where 1.0
            means all important blocks are well-placed.
        positional_scores: Per-block position analysis.
        warnings: Human-readable warnings for blocks at risk
            of being missed by the model.
    """

    overall_score: float
    positional_scores: List[PositionScore]
    warnings: List[str]
