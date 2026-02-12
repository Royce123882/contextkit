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
        signal_to_noise_ratio: Ratio of high-priority token mass
            to total token mass (0.0-1.0).
        redundancy_score: Fraction of blocks that are near-duplicates
            of another block (0.0-1.0, lower is better).
        information_density: Ratio of unique terms to total tokens
            across all blocks (higher = more diverse vocabulary).
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
    signal_to_noise_ratio: float = Field(
        default=0.0,
        description="Ratio of high-priority tokens to total tokens (0.0-1.0).",
    )
    redundancy_score: float = Field(
        default=0.0,
        description="Fraction of blocks that are near-duplicates (0.0-1.0, lower is better).",
    )
    information_density: float = Field(
        default=0.0,
        description="Unique terms per token across all blocks (higher = more diverse).",
    )


class DriftReport(BaseModel):
    """Report from drift detection across conversation turns.

    Attributes:
        current_score: Quality score for the current turn.
        drift: Magnitude of quality degradation since the beginning.
            Positive values mean quality has dropped.
        drifting: Whether the drift exceeds the configured threshold.
        turn_count: Number of turns recorded so far.
        suggestion: Actionable suggestion if drift is detected.
    """

    current_score: float = Field(
        description="Quality score for the current turn.",
    )
    drift: float = Field(
        default=0.0,
        description="Quality degradation since the beginning (positive = worse).",
    )
    drifting: bool = Field(
        default=False,
        description="Whether drift exceeds the configured threshold.",
    )
    turn_count: int = Field(
        default=0,
        description="Number of turns recorded so far.",
    )
    suggestion: str = Field(
        default="",
        description="Actionable suggestion if drift is detected.",
    )
