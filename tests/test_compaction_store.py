"""Tests for compaction store implementations."""

from __future__ import annotations

from contextkit.compaction.local_store import LocalCompactionStore
from contextkit.compaction.store import CompactionStore


class TestLocalCompactionStore:
    """Tests for the LocalCompactionStore filesystem backend."""

    async def test_save_creates_markdown_file(self, tmp_path) -> None:
        store = LocalCompactionStore(base_dir=str(tmp_path / "compacted"))
        uri = await store.save("test_key", "# Hello\n\nWorld")
        assert uri.endswith("test_key.md")
        assert (tmp_path / "compacted" / "test_key.md").exists()

    async def test_load_returns_saved_content(self, tmp_path) -> None:
        store = LocalCompactionStore(base_dir=str(tmp_path / "compacted"))
        await store.save("roundtrip", "some content here")
        loaded = await store.load("roundtrip")
        assert loaded == "some content here"

    async def test_load_returns_none_for_missing_key(self, tmp_path) -> None:
        store = LocalCompactionStore(base_dir=str(tmp_path / "compacted"))
        result = await store.load("nonexistent")
        assert result is None

    async def test_delete_removes_file(self, tmp_path) -> None:
        store = LocalCompactionStore(base_dir=str(tmp_path / "compacted"))
        await store.save("to_delete", "content")
        assert await store.delete("to_delete") is True
        assert await store.load("to_delete") is None

    async def test_delete_returns_false_for_missing_key(self, tmp_path) -> None:
        store = LocalCompactionStore(base_dir=str(tmp_path / "compacted"))
        assert await store.delete("nonexistent") is False

    async def test_list_keys_returns_all_saved(self, tmp_path) -> None:
        store = LocalCompactionStore(base_dir=str(tmp_path / "compacted"))
        await store.save("alpha", "a")
        await store.save("beta", "b")
        await store.save("gamma", "c")
        keys = await store.list_keys()
        assert set(keys) == {"alpha", "beta", "gamma"}

    async def test_list_keys_empty_when_no_files(self, tmp_path) -> None:
        store = LocalCompactionStore(base_dir=str(tmp_path / "empty"))
        keys = await store.list_keys()
        assert keys == []

    async def test_save_creates_directory_lazily(self, tmp_path) -> None:
        nested_dir = tmp_path / "deep" / "nested" / "dir"
        store = LocalCompactionStore(base_dir=str(nested_dir))
        await store.save("lazy", "content")
        assert (nested_dir / "lazy.md").exists()


class TestCompactionStoreProtocol:
    """Tests for the CompactionStore protocol."""

    def test_local_store_satisfies_protocol(self) -> None:
        store = LocalCompactionStore()
        assert isinstance(store, CompactionStore)
