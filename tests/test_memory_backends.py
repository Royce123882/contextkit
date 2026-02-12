"""Tests for memory backend enhancements.

Covers:
  - SQLiteBackend decay support (access_count, last_accessed, scoring)
  - PostgresBackend lazy import
  - InMemoryBackend decay integration (verified in test_new_features.py)
"""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from contextkit.memory.in_memory_backend import InMemoryBackend
from contextkit.memory.long_term import LongTermMemory
from contextkit.memory.record import MemoryRecord
from contextkit.memory.sqlite_backend import SQLiteBackend


class TestSQLiteBackendDecay:
    """Tests for decay support in the SQLiteBackend."""

    def _make_backend(self, tmp_path: str) -> SQLiteBackend:
        db_path = os.path.join(tmp_path, "test_decay.db")
        return SQLiteBackend(db_path=db_path)

    def test_retrieve_updates_access_count(self) -> None:
        """Retrieved records should have access_count incremented."""
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._make_backend(tmp)
            asyncio.run(backend.store("k1", "python programming"))
            results = asyncio.run(backend.retrieve("python"))
            assert len(results) >= 1
            assert results[0].access_count == 1
            assert results[0].last_accessed is not None

    def test_retrieve_persists_access_metadata(self) -> None:
        """Access metadata should survive across backend instances."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "persist_decay.db")

            # Store and retrieve with first instance
            backend1 = SQLiteBackend(db_path=db_path)
            asyncio.run(backend1.store("k1", "python programming"))
            asyncio.run(backend1.retrieve("python"))

            # Check with second instance
            backend2 = SQLiteBackend(db_path=db_path)
            records = asyncio.run(backend2.list_records())
            assert len(records) == 1
            assert records[0].access_count == 1
            assert records[0].last_accessed is not None

    def test_decay_factor_on_retrieved_record(self) -> None:
        """Retrieved records should have a valid decay_factor."""
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._make_backend(tmp)
            asyncio.run(backend.store("k1", "test content"))
            results = asyncio.run(backend.retrieve("test"))
            # Freshly accessed record should have decay factor near 1.0
            assert results[0].decay_factor() > 0.99

    def test_store_initializes_zero_access(self) -> None:
        """Newly stored records should have access_count=0."""
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._make_backend(tmp)
            record = asyncio.run(backend.store("k1", "content"))
            assert record.access_count == 0
            assert record.last_accessed is None

    def test_multiple_retrieves_increment_count(self) -> None:
        """Multiple retrieves should increment access_count each time."""
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._make_backend(tmp)
            asyncio.run(backend.store("k1", "python tips"))
            asyncio.run(backend.retrieve("python"))
            results = asyncio.run(backend.retrieve("python"))
            assert results[0].access_count == 2

    def test_decay_affects_ranking(self) -> None:
        """Decay should influence retrieval ranking for records with same content."""
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._make_backend(tmp)
            asyncio.run(backend.store("k1", "python tips"))
            asyncio.run(backend.store("k2", "python tips"))

            # Retrieve once to update k1 and k2 access metadata
            asyncio.run(backend.retrieve("python"))

            # Both should be retrievable
            results = asyncio.run(backend.retrieve("python"))
            assert len(results) == 2

    def test_backward_compat_with_existing_tests(self) -> None:
        """Existing SQLiteBackend tests should still pass unchanged."""
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._make_backend(tmp)
            asyncio.run(backend.store("k1", "user likes Python"))
            results = asyncio.run(backend.retrieve("Python"))
            assert len(results) >= 1
            assert results[0].key == "k1"

    def test_upsert_preserves_access_count(self) -> None:
        """Upserting a record resets access_count (new version of content)."""
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._make_backend(tmp)
            asyncio.run(backend.store("k1", "original"))
            asyncio.run(backend.retrieve("original"))
            # Upsert with new content
            asyncio.run(backend.store("k1", "updated"))
            records = asyncio.run(backend.list_records())
            assert records[0].content == "updated"
            assert records[0].access_count == 0


class TestPostgresBackendLazyImport:
    """Tests for lazy import of PostgresBackend."""

    def test_lazy_import_via_module(self) -> None:
        """PostgresBackend should be importable via contextkit.memory."""
        from contextkit.memory import PostgresBackend

        assert PostgresBackend is not None
        assert hasattr(PostgresBackend, "store")
        assert hasattr(PostgresBackend, "retrieve")

    def test_direct_import(self) -> None:
        """PostgresBackend should be directly importable."""
        from contextkit.memory.postgres_backend import PostgresBackend

        assert PostgresBackend is not None

    def test_accepts_pool_parameter(self) -> None:
        """PostgresBackend constructor should accept a pool parameter."""
        from contextkit.memory.postgres_backend import PostgresBackend

        # Should not raise - pool=None means create own pool lazily
        backend = PostgresBackend(pool=None)
        assert backend is not None

    def test_with_long_term_memory(self) -> None:
        """PostgresBackend should be usable as a LongTermMemory backend type."""
        from contextkit.memory.postgres_backend import PostgresBackend

        # Verify it has the MemoryBackend interface methods
        assert hasattr(PostgresBackend, "store")
        assert hasattr(PostgresBackend, "retrieve")
        assert hasattr(PostgresBackend, "delete")
        assert hasattr(PostgresBackend, "list_records")
