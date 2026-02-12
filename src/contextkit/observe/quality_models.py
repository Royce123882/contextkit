"""Data models for context quality scoring.

Separates data models from the scoring implementation
to follow the Single Responsibility Principle.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


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

    block_name: str = Field(description="Display name of the scored block.")
    position: int = Field(description="Zero-based position in the block sequence.")
    total_blocks: int = Field(description="Total number of blocks in the window.")
    attention_weight: float = Field(
        description="Estimated attention weight at this position (1.0 = high, 0.0 = low)."
    )
    priority: int = Field(description="The block's priority value.")
    risk: str = Field(
        description="Risk level for information loss ('low', 'medium', or 'high')."
    )


class QualityReport(BaseModel):
    """Aggregate quality report for a context window.

    Attributes:
        overall_score: Weighted quality score (0.0-1.0) where 1.0
            means all important blocks are well-placed.
        positional_scores: Per-block position analysis.
        warnings: Human-readable warnings for blocks at risk
            of being missed by the model.
    """

    overall_score: float = Field(
        description="Weighted quality score (0.0-1.0) where 1.0 means all important blocks are well-placed."
    )
    positional_scores: List[PositionScore] = Field(
        description="Per-block position analysis."
    )
    warnings: List[str] = Field(
        description="Human-readable warnings for blocks at risk of being missed."
    )
