"""Shared caching utilities for contextkit.

Provides LRU cache configuration used by token counting
and other repeated computations.
"""

from __future__ import annotations

from typing import Tuple

from cachetools import LRUCache

from contextkit.constants import DEFAULT_TOKEN_CACHE_SIZE

# Global token count cache shared across all callers.
token_count_cache: LRUCache[Tuple[int, str], int] = LRUCache(
    maxsize=DEFAULT_TOKEN_CACHE_SIZE
)


def clear_token_cache() -> None:
    """Clear the global token count cache."""
    token_count_cache.clear()
