"""Tests for memory management: records, backends, short-term and
long-term memory, conversation trimming, and temporal decay.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from contextkit.memory.in_memory_backend import InMemoryBackend
from contextkit.memory.long_term import LongTermMemory
from contextkit.memory.record import MemoryRecord
from contextkit.memory.short_term import ShortTermMemory, trim_conversation
from contextkit.memory.sqlite_backend import SQLiteBackend


class TestMemoryRecord:
    """Tests for the MemoryRecord model."""

    def test_create_with_defaults(self) -> None:
        record = MemoryRecord(key="k1", content="test")
        assert record.key == "k1"
        assert record.content == "test"
        assert record.tags == []
        assert record.importance == 0.5

    def test_create_with_all_fields(self) -> None:
        record = MemoryRecord(
            key="k1",
            content="test",
            tags=["style", "pref"],
            importance=0.9,
            metadata={"source": "user"},
        )
        assert record.tags == ["style", "pref"]
        assert record.importance == 0.9


class TestInMemoryBackend:
    """Tests for the InMemoryBackend."""

    def test_store_and_retrieve(self) -> None:
        backend = InMemoryBackend()
        asyncio.run(backend.store("k1", "user likes Python"))
        results = asyncio.run(backend.retrieve("Python"))
        assert len(results) >= 1
        assert results[0].key == "k1"

    def test_store_with_tags(self) -> None:
        backend = InMemoryBackend()
        asyncio.run(backend.store("k1", "content", tags=["style"]))
        results = asyncio.run(backend.retrieve("content", tags=["style"]))
        assert len(results) == 1

    def test_retrieve_filters_by_tags(self) -> None:
        backend = InMemoryBackend()
        asyncio.run(backend.store("k1", "content", tags=["a"]))
        asyncio.run(backend.store("k2", "content", tags=["b"]))
        results = asyncio.run(backend.retrieve("content", tags=["a"]))
        assert len(results) == 1
        assert results[0].key == "k1"

    def test_delete(self) -> None:
        backend = InMemoryBackend()
        asyncio.run(backend.store("k1", "content"))
        assert asyncio.run(backend.delete("k1")) is True
        assert asyncio.run(backend.delete("k1")) is False

    def test_list_records(self) -> None:
        backend = InMemoryBackend()
        asyncio.run(backend.store("k1", "a"))
        asyncio.run(backend.store("k2", "b"))
        records = asyncio.run(backend.list_records())
        assert len(records) == 2

    def test_list_records_with_tags(self) -> None:
        backend = InMemoryBackend()
        asyncio.run(backend.store("k1", "a", tags=["x"]))
        asyncio.run(backend.store("k2", "b", tags=["y"]))
        records = asyncio.run(backend.list_records(tags=["x"]))
        assert len(records) == 1

    def test_record_count(self) -> None:
        backend = InMemoryBackend()
        assert backend.record_count == 0
        asyncio.run(backend.store("k1", "a"))
        assert backend.record_count == 1

    def test_retrieve_top_k(self) -> None:
        backend = InMemoryBackend()
        for i in range(10):
            asyncio.run(backend.store(f"k{i}", f"item {i}"))
        results = asyncio.run(backend.retrieve("item", top_k=3))
        assert len(results) == 3


class TestShortTermMemory:
    """Tests for ShortTermMemory."""

    def test_add_turn(self) -> None:
        stm = ShortTermMemory()
        stm.add_turn("user", "Hello")
        stm.add_turn("assistant", "Hi!")
        assert stm.turn_count == 2

    def test_sliding_window_trims(self) -> None:
        stm = ShortTermMemory(strategy="sliding_window", max_turns=3)
        for i in range(5):
            stm.add_turn("user", f"Message {i}")
        assert stm.turn_count == 3
        assert stm.messages[0]["content"] == "Message 2"

    def test_token_budget_trims(self) -> None:
        stm = ShortTermMemory(
            strategy="token_budget",
            max_turns=100,
            max_tokens=50,
        )
        for i in range(20):
            stm.add_turn("user", f"This is message number {i} " * 3)
        assert stm.token_count <= 50

    def test_to_block(self) -> None:
        stm = ShortTermMemory()
        stm.add_turn("user", "Hello")
        stm.add_turn("assistant", "Hi!")
        block = stm.to_block()
        assert block.type.value == "short_term_memory"
        assert block.origin is not None
        assert block.origin.source == "conversation"
        assert "turn_range" in block.origin.details

    def test_to_block_shows_trimming(self) -> None:
        stm = ShortTermMemory(max_turns=2)
        for i in range(5):
            stm.add_turn("user", f"Msg {i}")
        block = stm.to_block()
        assert block.origin is not None
        assert block.origin.details["trimmed_turns"] == 3

    def test_total_turns_added(self) -> None:
        stm = ShortTermMemory(max_turns=3)
        for i in range(10):
            stm.add_turn("user", f"Msg {i}")
        assert stm.total_turns_added == 10
        assert stm.turn_count == 3

    def test_add_messages_bulk(self) -> None:
        stm = ShortTermMemory()
        stm.add_messages(
            [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "B"},
            ]
        )
        assert stm.turn_count == 2

    def test_clear(self) -> None:
        stm = ShortTermMemory()
        stm.add_turn("user", "Hello")
        stm.clear()
        assert stm.turn_count == 0

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            ShortTermMemory(strategy="invalid")

    def test_messages_returns_copy(self) -> None:
        stm = ShortTermMemory()
        stm.add_turn("user", "Hello")
        msgs = stm.messages
        msgs.clear()
        assert stm.turn_count == 1

    def test_token_count(self) -> None:
        stm = ShortTermMemory()
        stm.add_turn("user", "Hello world")
        assert stm.token_count > 0

    def test_empty_token_count(self) -> None:
        stm = ShortTermMemory()
        assert stm.token_count == 0


class TestTrimConversation:
    """Tests for the standalone trim_conversation utility."""

    def test_sliding_window(self) -> None:
        messages = [{"role": "user", "content": f"Msg {i}"} for i in range(10)]
        trimmed = trim_conversation(messages, strategy="sliding_window", max_turns=5)
        assert len(trimmed) == 5
        assert trimmed[0]["content"] == "Msg 5"

    def test_token_budget(self) -> None:
        messages = [{"role": "user", "content": "word " * 20} for _ in range(10)]
        trimmed = trim_conversation(messages, strategy="token_budget", max_tokens=100)
        assert len(trimmed) < 10

    def test_no_trimming_needed(self) -> None:
        messages = [
            {"role": "user", "content": "Hi"},
        ]
        trimmed = trim_conversation(messages, max_turns=10)
        assert len(trimmed) == 1

    def test_invalid_strategy(self) -> None:
        with pytest.raises(ValueError):
            trim_conversation([], strategy="bad")

    def test_returns_copy(self) -> None:
        messages = [{"role": "user", "content": "Hi"}]
        trimmed = trim_conversation(messages, max_turns=10)
        trimmed.append({"role": "user", "content": "Extra"})
        assert len(messages) == 1


class TestLongTermMemory:
    """Tests for LongTermMemory."""

    def test_store_and_retrieve(self) -> None:
        ltm = LongTermMemory()
        asyncio.run(ltm.store("k1", "Python is great"))
        results = asyncio.run(ltm.retrieve("Python"))
        assert len(results) >= 1

    def test_retrieve_as_blocks(self) -> None:
        ltm = LongTermMemory()
        asyncio.run(ltm.store("k1", "Python tips", tags=["code"]))
        blocks = asyncio.run(ltm.retrieve_as_blocks("Python", top_k=5))
        assert len(blocks) >= 1
        assert blocks[0].type.value == "long_term_memory"
        assert blocks[0].origin is not None
        assert blocks[0].origin.source == "memory"

    def test_delete(self) -> None:
        ltm = LongTermMemory()
        asyncio.run(ltm.store("k1", "content"))
        assert asyncio.run(ltm.delete("k1")) is True
        assert asyncio.run(ltm.delete("k1")) is False

    def test_list_records(self) -> None:
        ltm = LongTermMemory()
        asyncio.run(ltm.store("a", "content_a"))
        asyncio.run(ltm.store("b", "content_b"))
        records = asyncio.run(ltm.list_records())
        assert len(records) == 2

    def test_custom_backend(self) -> None:
        backend = InMemoryBackend()
        ltm = LongTermMemory(backend=backend)
        asyncio.run(ltm.store("k1", "test"))
        assert backend.record_count == 1


aiosqlite = pytest.importorskip("aiosqlite", reason="aiosqlite not installed")


class TestSQLiteBackend:
    """Tests for the SQLiteBackend persistent memory storage."""

    def _make_backend(self, tmp_path: str) -> SQLiteBackend:
        """Create a backend with a temporary database file."""
        db_path = os.path.join(tmp_path, "test_memory.db")
        return SQLiteBackend(db_path=db_path)

    def test_store_and_retrieve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._make_backend(tmp)
            asyncio.run(backend.store("k1", "user likes Python"))
            results = asyncio.run(backend.retrieve("Python"))
            assert len(results) >= 1
            assert results[0].key == "k1"
            assert results[0].content == "user likes Python"

    def test_store_with_metadata_and_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._make_backend(tmp)
            record = asyncio.run(
                backend.store(
                    "k1",
                    "content",
                    metadata={"source": "user"},
                    tags=["style"],
                    importance=0.9,
                )
            )
            assert record.key == "k1"
            assert record.metadata == {"source": "user"}
            assert record.tags == ["style"]
            assert record.importance == 0.9

    def test_retrieve_filters_by_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._make_backend(tmp)
            asyncio.run(backend.store("k1", "content a", tags=["a"]))
            asyncio.run(backend.store("k2", "content b", tags=["b"]))
            results = asyncio.run(backend.retrieve("content", tags=["a"]))
            assert len(results) == 1
            assert results[0].key == "k1"

    def test_retrieve_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._make_backend(tmp)
            for i in range(10):
                asyncio.run(backend.store(f"k{i}", f"item number {i}"))
            results = asyncio.run(backend.retrieve("item", top_k=3))
            assert len(results) == 3

    def test_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._make_backend(tmp)
            asyncio.run(backend.store("k1", "content"))
            assert asyncio.run(backend.delete("k1")) is True
            assert asyncio.run(backend.delete("k1")) is False

    def test_list_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._make_backend(tmp)
            asyncio.run(backend.store("k1", "a"))
            asyncio.run(backend.store("k2", "b"))
            records = asyncio.run(backend.list_records())
            assert len(records) == 2

    def test_list_records_with_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._make_backend(tmp)
            asyncio.run(backend.store("k1", "a", tags=["x"]))
            asyncio.run(backend.store("k2", "b", tags=["y"]))
            records = asyncio.run(backend.list_records(tags=["x"]))
            assert len(records) == 1
            assert records[0].key == "k1"

    def test_upsert_overwrites_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._make_backend(tmp)
            asyncio.run(backend.store("k1", "original"))
            asyncio.run(backend.store("k1", "updated"))
            records = asyncio.run(backend.list_records())
            assert len(records) == 1
            assert records[0].content == "updated"

    def test_persistence_across_instances(self) -> None:
        """Verify data persists when creating a new backend instance."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "persist.db")
            # Store with first instance
            backend1 = SQLiteBackend(db_path=db_path)
            asyncio.run(backend1.store("k1", "persistent data"))
            # Retrieve with second instance
            backend2 = SQLiteBackend(db_path=db_path)
            results = asyncio.run(backend2.retrieve("persistent"))
            assert len(results) == 1
            assert results[0].content == "persistent data"

    def test_record_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._make_backend(tmp)
            count = asyncio.run(backend.record_count)
            assert count == 0
            asyncio.run(backend.store("k1", "a"))
            count = asyncio.run(backend.record_count)
            assert count == 1

    def test_with_long_term_memory(self) -> None:
        """Verify SQLiteBackend works as LongTermMemory backend."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "ltm.db")
            backend = SQLiteBackend(db_path=db_path)
            ltm = LongTermMemory(backend=backend)
            asyncio.run(ltm.store("k1", "Python tips", tags=["code"]))
            blocks = asyncio.run(ltm.retrieve_as_blocks("Python"))
            assert len(blocks) >= 1
            assert blocks[0].type.value == "long_term_memory"


# ---------------------------------------------------------------------------
# Temporal memory decay (Ebbinghaus forgetting curve)
# ---------------------------------------------------------------------------


class TestTemporalMemoryDecay:
    """MemoryRecord.decay_factor models exponential forgetting with
    spaced-repetition reinforcement on repeated access.
    """

    def test_fresh_record_has_no_decay(self) -> None:
        """A just-created record has a decay factor of 1.0."""
        now = datetime.now(timezone.utc)
        record = MemoryRecord(key="fresh", content="test", stored_at=now)
        assert record.decay_factor(now=now) == 1.0

    def test_one_half_life_yields_half_retention(self) -> None:
        """After exactly one half-life, decay factor is approximately 0.5."""
        now = datetime.now(timezone.utc)
        one_week_ago = now - timedelta(hours=168)
        record = MemoryRecord(key="week_old", content="test", stored_at=one_week_ago)
        factor = record.decay_factor(half_life_hours=168.0, now=now)
        assert 0.4 <= factor <= 0.6

    def test_repeated_access_slows_decay(self) -> None:
        """More access_count extends the effective half-life (spaced repetition)."""
        now = datetime.now(timezone.utc)
        one_week_ago = now - timedelta(hours=168)

        never_accessed = MemoryRecord(
            key="never_accessed", content="test", stored_at=one_week_ago
        )
        frequently_accessed = MemoryRecord(
            key="frequently_accessed", content="test", stored_at=one_week_ago, access_count=5
        )

        decay_without_access = never_accessed.decay_factor(half_life_hours=168.0, now=now)
        decay_with_access = frequently_accessed.decay_factor(half_life_hours=168.0, now=now)

        assert decay_with_access > decay_without_access

    def test_record_access_increments_count_and_timestamp(self) -> None:
        """record_access() updates access_count and last_accessed."""
        record = MemoryRecord(key="track_access", content="test")
        assert record.access_count == 0
        assert record.last_accessed is None

        record.record_access()
        assert record.access_count == 1
        assert record.last_accessed is not None

    def test_decay_uses_last_accessed_over_stored_at(self) -> None:
        """If last_accessed is set, it becomes the decay reference point."""
        now = datetime.now(timezone.utc)
        old_stored_at = now - timedelta(hours=1000)
        recent_access = now - timedelta(hours=1)

        record = MemoryRecord(
            key="recently_accessed",
            content="test",
            stored_at=old_stored_at,
            last_accessed=recent_access,
        )
        factor = record.decay_factor(half_life_hours=168.0, now=now)
        assert factor > 0.99

    def test_decay_factor_never_reaches_zero(self) -> None:
        """Even very old records have a positive (non-zero) decay factor."""
        now = datetime.now(timezone.utc)
        ten_years_ago = now - timedelta(days=365 * 10)
        record = MemoryRecord(key="ancient", content="test", stored_at=ten_years_ago)
        factor = record.decay_factor(half_life_hours=168.0, now=now)
        assert factor > 0.0


# ---------------------------------------------------------------------------
# In-memory backend decay integration
# ---------------------------------------------------------------------------


class TestInMemoryBackendWithDecay:
    """InMemoryBackend retrieval should integrate temporal decay into ranking."""

    def test_retrieval_updates_access_metadata(self) -> None:
        """Retrieved records have their access_count and last_accessed updated."""
        backend = InMemoryBackend()
        asyncio.run(backend.store("python_tips", "python programming"))
        results = asyncio.run(backend.retrieve("python"))
        assert results[0].access_count == 1
        assert results[0].last_accessed is not None

    def test_recent_record_ranked_above_stale_record(self) -> None:
        """A recently stored record ranks higher than an older identical one."""
        backend = InMemoryBackend()
        now = datetime.now(timezone.utc)

        asyncio.run(backend.store("stale_record", "python tips"))
        asyncio.run(backend.store("fresh_record", "python tips"))

        backend._records["stale_record"].stored_at = now - timedelta(days=30)
        backend._records["fresh_record"].stored_at = now

        results = asyncio.run(backend.retrieve("python"))
        assert results[0].key == "fresh_record"
