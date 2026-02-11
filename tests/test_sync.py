"""Tests for synchronous wrappers."""

from __future__ import annotations

from contextkit.sync import LongTermMemory


class TestSyncLongTermMemory:
    """Tests for the sync LongTermMemory wrapper."""

    def test_store_and_retrieve(self) -> None:
        ltm = LongTermMemory()
        ltm.store("k1", "Python is great")
        results = ltm.retrieve("Python")
        assert len(results) >= 1

    def test_retrieve_as_blocks(self) -> None:
        ltm = LongTermMemory()
        ltm.store("k1", "Python tips", tags=["code"])
        blocks = ltm.retrieve_as_blocks("Python")
        assert len(blocks) >= 1
        assert blocks[0].type.value == "long_term_memory"

    def test_delete(self) -> None:
        ltm = LongTermMemory()
        ltm.store("k1", "content")
        assert ltm.delete("k1") is True
        assert ltm.delete("k1") is False

    def test_list_records(self) -> None:
        ltm = LongTermMemory()
        ltm.store("a", "content_a")
        ltm.store("b", "content_b")
        records = ltm.list_records()
        assert len(records) == 2

    def test_list_records_with_tags(self) -> None:
        ltm = LongTermMemory()
        ltm.store("a", "content_a", tags=["x"])
        ltm.store("b", "content_b", tags=["y"])
        records = ltm.list_records(tags=["x"])
        assert len(records) == 1

    def test_store_with_importance(self) -> None:
        ltm = LongTermMemory()
        record = ltm.store("k1", "content", importance=0.9)
        assert record.importance == 0.9
