"""Context quality scoring based on positional attention analysis.

Estimates the "findability" of key information given its position
in the context window.  High-priority blocks buried in the middle
of the sequence are flagged as at-risk.

Extended with signal-to-noise ratio, redundancy detection, and
information density metrics.

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
from contextkit.utils.text_similarity import word_overlap_similarity


class QualityScorer:
    """Score a context window for positional quality.

    Computes a per-block attention weight using a U-shaped curve
    that mirrors empirical LLM attention patterns, then flags
    high-priority blocks placed in low-attention positions.

    Also computes signal-to-noise ratio, redundancy, and
    information density metrics.

    The U-curve depth can be adapted per model via ``curve_depth``.
    Newer long-context models (Claude 4.x, GPT-4.1) lose less
    information in the middle and should use a shallower curve
    (lower depth). See :class:`~contextkit.models.AttentionProfile`.

    Args:
        high_priority_threshold: Blocks at or above this priority
            are considered "important" and checked for position risk.
        low_attention_threshold: Attention weight below this value
            is considered a risky position for important blocks.
        curve_depth: Depth of the U-curve trough (0.0-1.0). Controls
            how much the scorer penalises middle positions. Default
            0.6 matches the original "Lost in the Middle" findings.
            Pass a lower value (e.g. 0.3) for strong long-context
            models. Can also be set from a model's AttentionProfile.
    """

    def __init__(
        self,
        high_priority_threshold: int = HIGH_PRIORITY_THRESHOLD,
        low_attention_threshold: float = LOW_ATTENTION_THRESHOLD,
        curve_depth: float = 0.6,
    ) -> None:
        self._priority_threshold = high_priority_threshold
        self._attention_threshold = low_attention_threshold
        self._curve_depth = max(0.0, min(1.0, curve_depth))

    @classmethod
    def for_model(cls, model_name: str) -> "QualityScorer":
        """Create a QualityScorer tuned for a specific model.

        Looks up the model's AttentionProfile and configures the
        U-curve depth accordingly. Falls back to the standard
        profile if the model is not found or has no profile.

        Args:
            model_name: A registered model name (e.g. "claude-opus-4-6").

        Returns:
            A QualityScorer with model-appropriate curve depth.
        """
        from contextkit.models import get_model, UnknownModelError

        try:
            spec = get_model(model_name)
        except UnknownModelError:
            return cls()

        if spec.attention_profile is not None:
            return cls(curve_depth=spec.attention_profile.curve_depth)
        return cls()

    def score(self, blocks: List[ContextBlock]) -> QualityReport:
        """Produce a quality report for the given block sequence.

        Args:
            blocks: Ordered list of context blocks.

        Returns:
            A QualityReport with overall score, per-block position
            scores, extended metrics, and human-readable warnings.
        """
        total = len(blocks)
        if total == 0:
            return QualityReport(
                overall_score=1.0,
                positional_scores=[],
                warnings=[],
                signal_to_noise_ratio=1.0,
                redundancy_score=0.0,
                information_density=0.0,
            )

        positional_scores = self._compute_positional_scores(blocks, total)
        warnings = self._collect_warnings(positional_scores)
        overall = self._compute_overall_score(blocks, positional_scores)
        signal_to_noise = self._compute_signal_to_noise(blocks)
        redundancy = self._compute_redundancy(blocks)
        density = self._compute_information_density(blocks)

        return QualityReport(
            overall_score=round(min(1.0, overall), 3),
            positional_scores=positional_scores,
            warnings=warnings,
            signal_to_noise_ratio=round(signal_to_noise, 3),
            redundancy_score=round(redundancy, 3),
            information_density=round(density, 3),
        )

    def _compute_positional_scores(
        self,
        blocks: List[ContextBlock],
        total: int,
    ) -> List[PositionScore]:
        """Compute per-block positional attention scores.

        Args:
            blocks: Ordered list of context blocks.
            total: Total number of blocks.

        Returns:
            List of PositionScore entries.
        """
        scores: List[PositionScore] = []
        for position, block in enumerate(blocks):
            attention = _u_curve_weight(position, total, self._curve_depth)
            risk = self._assess_risk(block.priority, attention)
            scores.append(
                PositionScore(
                    block_name=block.display_name,
                    position=position,
                    total_blocks=total,
                    attention_weight=round(attention, 3),
                    priority=block.priority,
                    risk=risk,
                )
            )
        return scores

    def _collect_warnings(
        self,
        positional_scores: List[PositionScore],
    ) -> List[str]:
        """Collect human-readable warnings for high-risk blocks.

        Args:
            positional_scores: Per-block position analysis.

        Returns:
            List of warning strings.
        """
        warnings: List[str] = []
        for score in positional_scores:
            if score.risk == "high":
                warnings.append(
                    f"Block '{score.block_name}' (priority={score.priority}) "
                    f"is at position {score.position}/{score.total_blocks} "
                    f"(attention={score.attention_weight:.2f}, risk=high)"
                )
        return warnings

    def _compute_overall_score(
        self,
        blocks: List[ContextBlock],
        positional_scores: List[PositionScore],
    ) -> float:
        """Compute weighted overall quality score.

        Args:
            blocks: The context blocks.
            positional_scores: Per-block position scores.

        Returns:
            Weighted score between 0.0 and 1.0.
        """
        weighted_sum = 0.0
        weight_total = 0.0
        for block, pos_score in zip(blocks, positional_scores):
            block_weight = block.priority / 100.0
            weighted_sum += pos_score.attention_weight * block_weight
            weight_total += block_weight
        return weighted_sum / weight_total if weight_total > 0 else 1.0

    def _compute_signal_to_noise(self, blocks: List[ContextBlock]) -> float:
        """Compute the signal-to-noise ratio of the context.

        Signal = tokens in high-priority blocks.
        Noise = tokens in low-priority blocks.

        Args:
            blocks: The context blocks.

        Returns:
            Ratio between 0.0 and 1.0 (1.0 = all signal, no noise).
        """
        signal_tokens = sum(
            block.token_count
            for block in blocks
            if block.priority >= self._priority_threshold
        )
        total_tokens = sum(block.token_count for block in blocks)
        if total_tokens == 0:
            return 1.0
        return signal_tokens / total_tokens

    def _compute_redundancy(self, blocks: List[ContextBlock]) -> float:
        """Compute the fraction of blocks that are near-duplicates.

        A block is considered redundant if its word overlap with any
        previously seen block exceeds 80%.

        Args:
            blocks: The context blocks.

        Returns:
            Fraction between 0.0 and 1.0 (lower is better).
        """
        if len(blocks) <= 1:
            return 0.0

        seen_contents: List[str] = []
        redundant_count = 0

        for block in blocks:
            if not isinstance(block.content, str):
                continue
            is_redundant = False
            for seen in seen_contents:
                if word_overlap_similarity(block.content, seen) > 0.8:
                    is_redundant = True
                    break
            if is_redundant:
                redundant_count += 1
            else:
                seen_contents.append(block.content)

        string_block_count = sum(
            1 for block in blocks if isinstance(block.content, str)
        )
        return redundant_count / max(string_block_count, 1)

    def _compute_information_density(
        self,
        blocks: List[ContextBlock],
    ) -> float:
        """Compute information density as unique terms per token.

        Higher density means more varied vocabulary (more informative).

        Args:
            blocks: The context blocks.

        Returns:
            Density score (higher is better).
        """
        all_words: set[str] = set()
        total_tokens = 0
        for block in blocks:
            if isinstance(block.content, str):
                all_words.update(block.content.lower().split())
                total_tokens += block.token_count
        if total_tokens == 0:
            return 0.0
        return len(all_words) / total_tokens

    def _assess_risk(self, priority: int, attention_weight: float) -> str:
        """Classify risk level for a block given its priority and position.

        Args:
            priority: The block's priority value.
            attention_weight: Estimated attention at this position.

        Returns:
            Risk level: "low", "medium", or "high".
        """
        if priority >= self._priority_threshold:
            if attention_weight < self._attention_threshold:
                return "high"
            if attention_weight < self._attention_threshold + 0.15:
                return "medium"
        return "low"


def _u_curve_weight(position: int, total: int, curve_depth: float = 0.6) -> float:
    """Compute attention weight using a U-shaped curve.

    The edges (position 0 and position total-1) receive weight ~1.0.
    The centre receives weight ~(1.0 - curve_depth).

    Args:
        position: Zero-based index in the sequence.
        total: Total number of items in the sequence.
        curve_depth: How much the trough dips (0.0-1.0). Higher
            means the model loses more information in the middle.
            Default 0.6 matches "Lost in the Middle" findings.

    Returns:
        Estimated attention weight for this position.
    """
    if total <= 2:
        return 1.0
    normalised = position / (total - 1)
    return 1.0 - curve_depth * math.sin(math.pi * normalised)
