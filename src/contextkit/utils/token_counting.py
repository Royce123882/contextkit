"""Token counting with LRU caching.

Provides standalone utilities for counting tokens and checking
whether content fits within a token budget. Uses tiktoken for
BPE tokenization when available and falls back to a heuristic
estimator when tiktoken encodings cannot be loaded.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    import tiktoken

from contextkit.utils.cache import token_count_cache
from contextkit.constants import (
    CHARS_PER_TOKEN_ESTIMATE,
    DEFAULT_ENCODING,
    MESSAGE_OVERHEAD_TOKENS,
)

logger = logging.getLogger("contextkit")


def content_hash(content: str) -> int:
    """Return a stable integer hash for content string."""
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return int(digest, 16)


def _get_encoder(encoding: str) -> Any:
    """Try to get a tiktoken encoder, return None if unavailable.

    Catches broad Exception because tiktoken may raise ImportError
    (not installed), KeyError (unknown encoding), or network errors
    (ConnectionError/OSError) when downloading encoding files.
    """
    try:
        import tiktoken

        return tiktoken.get_encoding(encoding)
    except Exception:  # noqa: BLE001
        return None


def count(
    content: str | List[Dict[str, Any]],
    encoding: str = DEFAULT_ENCODING,
) -> int:
    """Count the number of tokens in content.

    Uses tiktoken when the encoding is available, otherwise falls
    back to a character-based estimation (~4 chars per token).

    Args:
        content: A string or a list of message dicts (each with
            at least a "content" key).
        encoding: The tiktoken encoding name to use.

    Returns:
        The total token count.
    """
    if isinstance(content, str):
        return _count_string(content, encoding)
    return _count_messages(content, encoding)


def _count_string(text: str, encoding: str) -> int:
    """Count tokens in a plain string, using cache."""
    cache_key = (content_hash(text), encoding)
    cached: int | None = token_count_cache.get(cache_key)
    if cached is not None:
        return cached

    enc = _get_encoder(encoding)
    if enc is not None:
        result = len(enc.encode(text))
    else:
        # Fallback: estimate based on character count
        result = max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)

    token_count_cache[cache_key] = result
    return result


def _count_messages(messages: List[Dict[str, Any]], encoding: str) -> int:
    """Count tokens in a list of message dicts.

    Each message is expected to have at least a "content" key.
    Adds per-message overhead for role/delimiters.
    """
    total = 0
    for message in messages:
        msg_content = message.get("content", "")
        if isinstance(msg_content, str):
            total += _count_string(msg_content, encoding)
        total += MESSAGE_OVERHEAD_TOKENS  # role + delimiters
    return total


def fits_budget(
    content: str | List[Dict[str, Any]],
    max_tokens: int,
    encoding: str = DEFAULT_ENCODING,
) -> bool:
    """Check whether content fits within a token budget.

    Args:
        content: A string or list of message dicts.
        max_tokens: The maximum allowed token count.
        encoding: The tiktoken encoding name.

    Returns:
        True if the content fits, False otherwise.
    """
    return count(content, encoding) <= max_tokens
