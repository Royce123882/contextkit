"""Tests for token counting and caching."""

from __future__ import annotations

from contextkit._cache import clear_token_cache, token_count_cache
from contextkit._tokens import _content_hash, count, fits_budget


class TestTokenCount:
    """Tests for the count() function."""

    def setup_method(self) -> None:
        clear_token_cache()

    def test_count_simple_string(self) -> None:
        result = count("Hello world")
        assert result > 0
        assert isinstance(result, int)

    def test_count_empty_string(self) -> None:
        result = count("")
        # Empty string should be 0 or at minimum a small count
        assert result >= 0

    def test_count_long_string(self) -> None:
        short = count("Hello")
        long = count("Hello world, this is a much longer string")
        assert long > short

    def test_count_message_list(self) -> None:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = count(messages)
        assert result > 0

    def test_count_message_list_includes_overhead(self) -> None:
        text_tokens = count("Hello")
        msg_tokens = count([{"role": "user", "content": "Hello"}])
        # Message count should be higher due to per-message overhead
        assert msg_tokens > text_tokens

    def test_count_empty_message_list(self) -> None:
        result = count([])
        assert result == 0

    def test_count_message_with_empty_content(self) -> None:
        result = count([{"role": "user", "content": ""}])
        # Should at least have message overhead
        assert result >= 4

    def test_cache_hit(self) -> None:
        clear_token_cache()
        text = "Cache test string"
        first = count(text)
        # Check cache has an entry
        cache_size_after_first = len(token_count_cache)
        second = count(text)
        # Cache size should not increase
        assert len(token_count_cache) == cache_size_after_first
        assert first == second

    def test_different_content_different_count(self) -> None:
        count1 = count("abc")
        count2 = count("This is a much longer piece of text for testing")
        assert count1 != count2


class TestFitsBudget:
    """Tests for the fits_budget() function."""

    def test_fits_when_under_budget(self) -> None:
        assert fits_budget("Hello", max_tokens=100) is True

    def test_does_not_fit_when_over_budget(self) -> None:
        long_text = "word " * 1000
        assert fits_budget(long_text, max_tokens=10) is False

    def test_fits_message_list(self) -> None:
        messages = [{"role": "user", "content": "Hi"}]
        assert fits_budget(messages, max_tokens=100) is True

    def test_exact_boundary(self) -> None:
        text = "Hello"
        token_count = count(text)
        assert fits_budget(text, max_tokens=token_count) is True
        assert fits_budget(text, max_tokens=token_count - 1) is False


class TestContentHash:
    """Tests for the _content_hash function."""

    def test_deterministic(self) -> None:
        h1 = _content_hash("test")
        h2 = _content_hash("test")
        assert h1 == h2

    def test_different_content_different_hash(self) -> None:
        h1 = _content_hash("hello")
        h2 = _content_hash("world")
        assert h1 != h2

    def test_returns_integer(self) -> None:
        result = _content_hash("test")
        assert isinstance(result, int)


class TestCacheManagement:
    """Tests for cache management."""

    def test_clear_cache(self) -> None:
        count("test string for cache")
        assert len(token_count_cache) > 0
        clear_token_cache()
        assert len(token_count_cache) == 0

    def test_cache_size_limit(self) -> None:
        clear_token_cache()
        # Add many entries
        for i in range(100):
            count(f"string number {i}")
        assert len(token_count_cache) <= 4096  # Default max size
