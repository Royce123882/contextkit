"""Context drift detection across conversation turns.

Tracks quality scores over time and detects when context quality
degrades beyond a configured threshold. Useful for long-running
multi-turn conversations where context can silently rot.

Research basis: "Context Drift" (arXiv:2510.07777) -- 39% average
performance drop observed in multi-turn LLM interactions due to
gradual context quality degradation.
"""

from __future__ import annotations

from typing import List

from contextkit.core import ContextBlock
from contextkit.observe.quality import QualityScorer
from contextkit.observe.quality_models import DriftReport

_DEFAULT_DRIFT_THRESHOLD = 0.15
_MINIMUM_TURNS_FOR_DETECTION = 3


class DriftDetector:
    """Detect context quality degradation across conversation turns.

    Records quality scores at each turn and compares the recent
    average to the initial average. When the difference exceeds
    the threshold, drift is flagged.

    Args:
        quality_scorer: A QualityScorer instance for evaluating
            block quality. If None, a default scorer is created.
        drift_threshold: Minimum score drop to flag as drifting.
            Default is 0.15 (15 percentage points).
    """

    def __init__(
        self,
        quality_scorer: QualityScorer | None = None,
        drift_threshold: float = _DEFAULT_DRIFT_THRESHOLD,
    ) -> None:
        self._scorer = quality_scorer or QualityScorer()
        self._drift_threshold = drift_threshold
        self._score_history: List[float] = []

    @property
    def score_history(self) -> List[float]:
        """Quality scores recorded at each turn."""
        return list(self._score_history)

    @property
    def turn_count(self) -> int:
        """Number of turns recorded so far."""
        return len(self._score_history)

    def record(self, blocks: List[ContextBlock]) -> DriftReport:
        """Score the current context and compare to historical trend.

        Args:
            blocks: The current context blocks to evaluate.

        Returns:
            A DriftReport with the current score, drift magnitude,
            whether drift exceeds the threshold, and a suggestion.
        """
        report = self._scorer.score(blocks)
        current_score = report.overall_score
        self._score_history.append(current_score)

        drift = self._compute_drift()
        is_drifting = drift > self._drift_threshold

        suggestion = ""
        if is_drifting:
            suggestion = (
                "Context quality has degraded by "
                f"{drift:.0%} since the start. "
                "Consider compacting old turns, re-retrieving "
                "context, or resetting the conversation."
            )

        return DriftReport(
            current_score=round(current_score, 3),
            drift=round(drift, 3),
            drifting=is_drifting,
            turn_count=len(self._score_history),
            suggestion=suggestion,
        )

    def reset(self) -> None:
        """Clear all recorded history."""
        self._score_history.clear()

    def _compute_drift(self) -> float:
        """Compute the drift magnitude from the score history.

        Compares the average of the first N scores to the average
        of the most recent N scores. Returns a positive value when
        quality has dropped.

        Returns:
            Drift magnitude (positive = quality degraded).
        """
        if len(self._score_history) < _MINIMUM_TURNS_FOR_DETECTION:
            return 0.0

        window_size = min(_MINIMUM_TURNS_FOR_DETECTION, len(self._score_history))
        initial_scores = self._score_history[:window_size]
        recent_scores = self._score_history[-window_size:]

        initial_average = sum(initial_scores) / len(initial_scores)
        recent_average = sum(recent_scores) / len(recent_scores)

        return max(0.0, initial_average - recent_average)
