"""Shared caching utilities for contextkit.

Provides LRU cache configuration used by token counting
and other repeated computations.
"""

from __future__ import annotations

from cachetools import LRUCache

# Default cache size for token count results.
# Keyed by (content_hash, encoding_name).
DEFAULT_TOKEN_CACHE_SIZE = 4096

# Global token count cache shared across all callers.
token_count_cache: LRUCache[tuple[int, str], int] = LRUCache(
    maxsize=DEFAULT_TOKEN_CACHE_SIZE
)


def clear_token_cache() -> None:
    """Clear the global token count cache."""
    token_count_cache.clear()
