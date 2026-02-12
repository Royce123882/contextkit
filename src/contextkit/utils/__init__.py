"""Shared utility functions for contextkit.

Provides reusable helpers for text similarity, token counting,
caching, and other cross-cutting concerns.
"""

from contextkit.utils.cache import clear_token_cache, token_count_cache
from contextkit.utils.text_similarity import (
    memory_relevance_score,
    word_overlap_score,
    word_overlap_similarity,
)
from contextkit.utils.token_counting import content_hash, count, fits_budget

__all__ = [
    "clear_token_cache",
    "content_hash",
    "count",
    "fits_budget",
    "memory_relevance_score",
    "token_count_cache",
    "word_overlap_score",
    "word_overlap_similarity",
]
