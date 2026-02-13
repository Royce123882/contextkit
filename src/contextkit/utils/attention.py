"""Positional attention weight utilities.

Provides functions for estimating how much attention an LLM pays
to content at a given position in the context window, based on
the "Lost in the Middle" U-curve findings (Liu et al., 2023).
"""

from __future__ import annotations

import math


def u_curve_weight(position: int, total: int, curve_depth: float = 0.6) -> float:
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
        Estimated attention weight for this position (0.0-1.0).
    """
    if total <= 2:
        return 1.0
    normalised = position / (total - 1)
    return 1.0 - curve_depth * math.sin(math.pi * normalised)
