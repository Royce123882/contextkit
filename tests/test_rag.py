"""Tests for RAG context and retriever backends (Phase 3)."""

from __future__ import annotations

import asyncio

from contextkit.rag.chunk import Chunk, RetrieverBackend
from contextkit.rag.context import RAGContext, _deduplicate_chunks
from contextkit.rag.in_memory_retriever import InMemoryRetriever


class TestChunk:
    """Tests for the Chunk model."""

    def test_create_with_defaults(self) -> None:
        chunk = Chunk(content="Hello world")
        assert chunk.content == "Hello world"
        assert chunk.source == ""
        assert chunk.relevance_score == 0.0
        assert chunk.metadata == {}

    def test_create_with_all_fields(self) -> None:
        chunk = Chunk(
            content="data",
            source="doc1.pdf",
            relevance_score=0.95,
            metadata={"page": 3},
        )
        assert chunk.source == "doc1.pdf"
        assert chunk.relevance_score == 0.95
        assert chunk.metadata == {"page": 3}


class TestInMemoryRetriever:
    """Tests for the InMemoryRetriever."""

    def test_add_and_retrieve(self) -> None:
        retriever = InMemoryRetriever()
        retriever.add_chunk(Chunk(content="Python is a programming language"))
        results = asyncio.run(retriever.retrieve("Python"))
        assert len(results) >= 1

    def test_add_chunks_bulk(self) -> None:
        retriever = InMemoryRetriever()
        chunks = [
            Chunk(content="First chunk"),
            Chunk(content="Second chunk"),
        ]
        retriever.add_chunks(chunks)
        assert retriever.chunk_count == 2

    def test_retrieve_top_k(self) -> None:
        retriever = InMemoryRetriever()
        for i in range(10):
            retriever.add_chunk(Chunk(content=f"Document about topic {i}"))
        results = asyncio.run(retriever.retrieve("topic", top_k=3))
        assert len(results) == 3

    def test_retrieve_relevance_scoring(self) -> None:
        retriever = InMemoryRetriever()
        retriever.add_chunk(
            Chunk(
                content="Python programming language basics",
                relevance_score=0.9,
            )
        )
        retriever.add_chunk(
            Chunk(
                content="Java enterprise development",
                relevance_score=0.1,
            )
        )
        results = asyncio.run(retriever.retrieve("Python programming"))
        # Python chunk should rank higher
        assert "Python" in results[0].content

    def test_health_check(self) -> None:
        retriever = InMemoryRetriever()
        result = asyncio.run(retriever.health_check())
        assert result is True

    def test_chunk_count(self) -> None:
        retriever = InMemoryRetriever()
        assert retriever.chunk_count == 0
        retriever.add_chunk(Chunk(content="test"))
        assert retriever.chunk_count == 1

    def test_implements_protocol(self) -> None:
        retriever = InMemoryRetriever()
        assert isinstance(retriever, RetrieverBackend)


class TestDeduplicateChunks:
    """Tests for the _deduplicate_chunks helper."""

    def test_empty_list(self) -> None:
        result = _deduplicate_chunks([])
        assert result == []

    def test_single_chunk(self) -> None:
        chunks = [Chunk(content="Hello world")]
        result = _deduplicate_chunks(chunks)
        assert len(result) == 1

    def test_removes_duplicates(self) -> None:
        chunks = [
            Chunk(content="Python is a great language"),
            Chunk(content="Python is a great language"),
        ]
        result = _deduplicate_chunks(chunks)
        assert len(result) == 1

    def test_keeps_different_chunks(self) -> None:
        chunks = [
            Chunk(content="Python is a programming language"),
            Chunk(content="The weather is sunny today"),
        ]
        result = _deduplicate_chunks(chunks)
        assert len(result) == 2

    def test_custom_threshold(self) -> None:
        chunks = [
            Chunk(content="Python is great"),
            Chunk(content="Python is good"),
        ]
        # With low threshold, they should be considered different
        result = _deduplicate_chunks(chunks, similarity_threshold=0.99)
        assert len(result) == 2


class TestRAGContext:
    """Tests for the RAGContext."""

    def test_retrieve_basic(self) -> None:
        retriever = InMemoryRetriever()
        retriever.add_chunk(
            Chunk(
                content="Python basics",
                source="doc1",
                relevance_score=0.8,
            )
        )
        rag = RAGContext(retriever)
        blocks = asyncio.run(rag.retrieve("Python"))
        assert len(blocks) >= 1
        assert blocks[0].type.value == "rag"
        assert blocks[0].origin is not None
        assert blocks[0].origin.source == "rag"

    def test_retrieve_with_origin_details(self) -> None:
        retriever = InMemoryRetriever()
        retriever.add_chunk(Chunk(content="Data", source="src1", relevance_score=0.9))
        rag = RAGContext(retriever, retriever_name="my_retriever")
        blocks = asyncio.run(rag.retrieve("Data"))
        assert blocks[0].origin.details["retriever"] == "my_retriever"
        assert blocks[0].origin.details["query"] == "Data"

    def test_retrieve_min_relevance(self) -> None:
        retriever = InMemoryRetriever()
        retriever.add_chunk(Chunk(content="High relevance", relevance_score=0.9))
        retriever.add_chunk(Chunk(content="Low relevance", relevance_score=0.1))
        rag = RAGContext(retriever)
        blocks = asyncio.run(rag.retrieve("relevance", min_relevance=0.5))
        # Only the high relevance chunk should pass
        assert len(blocks) == 1

    def test_retrieve_max_tokens(self) -> None:
        retriever = InMemoryRetriever()
        for i in range(10):
            retriever.add_chunk(Chunk(content=f"Long content about topic {i} " * 50))
        rag = RAGContext(retriever)
        blocks = asyncio.run(rag.retrieve("topic", max_tokens=50))
        total = sum(b.token_count for b in blocks)
        assert total <= 50

    def test_retrieve_deduplicates(self) -> None:
        retriever = InMemoryRetriever()
        retriever.add_chunk(Chunk(content="Exact same content here"))
        retriever.add_chunk(Chunk(content="Exact same content here"))
        rag = RAGContext(retriever)
        blocks = asyncio.run(rag.retrieve("content"))
        assert len(blocks) == 1

    def test_health_check(self) -> None:
        retriever = InMemoryRetriever()
        rag = RAGContext(retriever)
        result = asyncio.run(rag.health_check())
        assert result is True

    def test_retrieve_top_k(self) -> None:
        retriever = InMemoryRetriever()
        for i in range(10):
            retriever.add_chunk(Chunk(content=f"Document {i}", source=f"doc{i}"))
        rag = RAGContext(retriever)
        blocks = asyncio.run(rag.retrieve("Document", top_k=3))
        assert len(blocks) <= 3

    def test_retrieve_priority(self) -> None:
        retriever = InMemoryRetriever()
        retriever.add_chunk(Chunk(content="test data"))
        rag = RAGContext(retriever)
        blocks = asyncio.run(rag.retrieve("test", priority=85))
        assert blocks[0].priority == 85
