"""Context quality scoring based on positional attention analysis.

Estimates the "findability" of key information given its position
in the context window.  High-priority blocks buried in the middle
of the sequence are flagged as at-risk.

Research basis: "Lost in the Middle" (Liu et al., 2023) -- LLMs
show a U-shaped performance curve where information at the start
and end is utilised effectively but middle content is often missed.
"""

from __future__ import annotations

import math
from typing import List

from contextkit.constants import HIGH_PRIORITY_THRESHOLD, LOW_ATTENTION_THRESHOLD
from contextkit.core import ContextBlock
from contextkit.observe.quality_models import PositionScore, QualityReport


class QualityScorer:
    """Score a context window for positional quality.

    Computes a per-block attention weight using a U-shaped curve
    that mirrors empirical LLM attention patterns, then flags
    high-priority blocks placed in low-attention positions.

    Args:
        high_priority_threshold: Blocks at or above this priority
            are considered "important" and checked for position risk.
        low_attention_threshold: Attention weight below this value
            is considered a risky position for important blocks.
    """

    def __init__(
        self,
        high_priority_threshold: int = HIGH_PRIORITY_THRESHOLD,
        low_attention_threshold: float = LOW_ATTENTION_THRESHOLD,
    ) -> None:
        self._priority_threshold = high_priority_threshold
        self._attention_threshold = low_attention_threshold

    def score(self, blocks: List[ContextBlock]) -> QualityReport:
        """Produce a quality report for the given block sequence.

        Args:
            blocks: Ordered list of context blocks.

        Returns:
            A :class:`QualityReport` with overall score, per-block
            position scores, and human-readable warnings.
        """
        total = len(blocks)
        if total == 0:
            return QualityReport(
                overall_score=1.0, positional_scores=[], warnings=[]
            )

        positional_scores: List[PositionScore] = []
        warnings: List[str] = []
        weighted_sum = 0.0
        weight_total = 0.0

        for idx, block in enumerate(blocks):
            attention = _u_curve_weight(idx, total)
            risk = self._assess_risk(block.priority, attention)

            positional_scores.append(
                PositionScore(
                    block_name=block.display_name,
                    position=idx,
                    total_blocks=total,
                    attention_weight=round(attention, 3),
                    priority=block.priority,
                    risk=risk,
                )
            )

            # Weight the contribution by priority (important blocks count more)
            block_weight = block.priority / 100.0
            weighted_sum += attention * block_weight
            weight_total += block_weight

            if risk == "high":
                warnings.append(
                    f"Block '{block.display_name}' (priority={block.priority}) "
                    f"is at position {idx}/{total} "
                    f"(attention={attention:.2f}, risk=high)"
                )

        overall = weighted_sum / weight_total if weight_total > 0 else 1.0
        return QualityReport(
            overall_score=round(min(1.0, overall), 3),
            positional_scores=positional_scores,
            warnings=warnings,
        )

    def _assess_risk(self, priority: int, attention_weight: float) -> str:
        """Classify risk level for a block given its priority and position."""
        if priority >= self._priority_threshold:
            if attention_weight < self._attention_threshold:
                return "high"
            if attention_weight < self._attention_threshold + 0.15:
                return "medium"
        return "low"


def _u_curve_weight(position: int, total: int) -> float:
    """Compute attention weight using a U-shaped curve.

    Returns a value in approximately [0.4, 1.0] where the edges
    (position 0 and position total-1) receive weight ~1.0 and
    the centre receives weight ~0.4.

    Args:
        position: Zero-based index in the sequence.
        total: Total number of items in the sequence.

    Returns:
        Estimated attention weight for this position.
    """
    if total <= 2:
        return 1.0
    normalised = position / (total - 1)
    return 1.0 - 0.6 * math.sin(math.pi * normalised)
