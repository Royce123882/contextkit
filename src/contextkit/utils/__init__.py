"""Shared utility functions for contextkit.

Provides reusable helpers for text similarity, token counting,
and other cross-cutting concerns.
"""

from contextkit.utils.text_similarity import (
    word_overlap_score,
    word_overlap_similarity,
)

__all__ = [
    "word_overlap_score",
    "word_overlap_similarity",
]
