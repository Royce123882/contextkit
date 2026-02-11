"""Tests for retriever backend lazy imports and import errors."""

from __future__ import annotations

import pytest


class TestRetrieverLazyImports:
    """Test that retriever backends are lazily importable."""

    def test_rag_module_imports_without_backends(self) -> None:
        from contextkit.rag import Chunk, RAGContext, RetrieverBackend

        assert Chunk is not None
        assert RAGContext is not None
        assert RetrieverBackend is not None

    def test_qdrant_retriever_import_error(self) -> None:
        with pytest.raises(ImportError, match="qdrant-client"):
            from contextkit.rag.qdrant_retriever import QdrantRetriever

            QdrantRetriever(
                collection_name="test",
                embed_fn=lambda q: [0.1],
                url="http://localhost:6333",
            )

    def test_pgvector_retriever_import_error(self) -> None:
        with pytest.raises(ImportError, match="asyncpg"):
            from contextkit.rag.pgvector_retriever import PgvectorRetriever

            PgvectorRetriever(
                dsn="postgresql://test@localhost/test",
                embed_fn=lambda q: [0.1],
            )

    def test_pinecone_retriever_import_error(self) -> None:
        with pytest.raises(ImportError, match="pinecone"):
            from contextkit.rag.pinecone_retriever import PineconeRetriever

            PineconeRetriever(
                index_name="test",
                embed_fn=lambda q: [0.1],
                api_key="test-key",
            )

    def test_rag_module_getattr_invalid(self) -> None:
        with pytest.raises(AttributeError, match="nonexistent"):
            from contextkit import rag

            _ = rag.nonexistent  # type: ignore[attr-defined]

    def test_rag_module_getattr_qdrant(self) -> None:
        from contextkit import rag

        # The __getattr__ should resolve QdrantRetriever
        # (though instantiation will fail without qdrant-client)
        cls = rag.QdrantRetriever
        assert cls.__name__ == "QdrantRetriever"
