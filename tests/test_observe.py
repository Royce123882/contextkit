"""Tests for provenance tracking (Origin and Mutation models)."""

from __future__ import annotations

from datetime import datetime

from contextkit.observe.provenance import Mutation, Origin


class TestOrigin:
    """Tests for the Origin provenance model."""

    def test_create_with_source_only(self) -> None:
        origin = Origin(source="prompt")
        assert origin.source == "prompt"
        assert isinstance(origin.created_at, datetime)
        assert origin.details == {}

    def test_create_with_all_fields(self) -> None:
        origin = Origin(
            source="rag",
            details={
                "query": "how does auth work",
                "retriever": "chroma/docs-index",
                "relevance_score": 0.87,
            },
        )
        assert origin.source == "rag"
        assert origin.query == "how does auth work"
        assert origin.retriever == "chroma/docs-index"
        assert origin.relevance_score == 0.87

    def test_prompt_origin(self) -> None:
        origin = Origin(
            source="prompt",
            details={"template": "assistant_v1", "version": "1.0"},
        )
        assert origin.template == "assistant_v1"
        assert origin.details["version"] == "1.0"

    def test_conversation_origin(self) -> None:
        origin = Origin(
            source="conversation",
            details={"turn_range": "1-12"},
        )
        assert origin.turn_range == "1-12"

    def test_file_origin(self) -> None:
        origin = Origin(
            source="file",
            details={"file_path": "/docs/auth.md", "chunk_index": 3},
        )
        assert origin.file_path == "/docs/auth.md"
        assert origin.details["chunk_index"] == 3

    def test_tool_origin(self) -> None:
        origin = Origin(
            source="tool",
            details={"tool_name": "search_docs"},
        )
        assert origin.tool_name == "search_docs"

    def test_memory_origin(self) -> None:
        origin = Origin(
            source="memory",
            details={"query": "user preferences", "key": "pref_style"},
        )
        assert origin.query == "user preferences"

    def test_convenience_properties_return_none_for_missing(self) -> None:
        origin = Origin(source="prompt")
        assert origin.template is None
        assert origin.query is None
        assert origin.retriever is None
        assert origin.relevance_score is None
        assert origin.file_path is None
        assert origin.tool_name is None
        assert origin.turn_range is None

    def test_summary_simple(self) -> None:
        origin = Origin(source="prompt")
        assert origin.summary() == "prompt"

    def test_summary_with_retriever(self) -> None:
        origin = Origin(
            source="rag",
            details={"retriever": "chroma", "relevance_score": 0.92},
        )
        summary = origin.summary()
        assert "rag" in summary
        assert "chroma" in summary
        assert "0.92" in summary

    def test_summary_with_query(self) -> None:
        origin = Origin(
            source="rag",
            details={"query": "a" * 50},
        )
        summary = origin.summary()
        assert "..." in summary  # Long query is truncated

    def test_summary_with_template(self) -> None:
        origin = Origin(
            source="prompt",
            details={"template": "assistant_v1"},
        )
        summary = origin.summary()
        assert "assistant_v1" in summary

    def test_serialization_to_dict(self) -> None:
        origin = Origin(
            source="rag",
            details={"query": "test", "relevance_score": 0.5},
        )
        data = origin.model_dump()
        assert data["source"] == "rag"
        assert data["details"]["query"] == "test"
        assert "created_at" in data

    def test_serialization_to_json(self) -> None:
        origin = Origin(source="prompt")
        json_str = origin.model_dump_json()
        assert "prompt" in json_str

    def test_created_at_is_utc(self) -> None:
        origin = Origin(source="test")
        assert origin.created_at.tzinfo is not None


class TestMutation:
    """Tests for the Mutation data model."""

    def test_create_basic_mutation(self) -> None:
        mutation = Mutation(
            step="TrimStep",
            action="removed",
            detail="priority 40 below threshold 50",
            tokens_before=1800,
            tokens_after=0,
        )
        assert mutation.step == "TrimStep"
        assert mutation.action == "removed"
        assert mutation.tokens_before == 1800
        assert mutation.tokens_after == 0
        assert mutation.before_content is None
        assert mutation.after_content is None

    def test_create_mutation_with_content(self) -> None:
        mutation = Mutation(
            step="CompactStep",
            action="compacted",
            detail="8200 -> 4100 tokens",
            tokens_before=8200,
            tokens_after=4100,
            before_content="Original very long text...",
            after_content="Compacted summary...",
        )
        assert mutation.before_content == "Original very long text..."
        assert mutation.after_content == "Compacted summary..."

    def test_create_reorder_mutation(self) -> None:
        mutation = Mutation(
            step="ReorderStep",
            action="moved",
            detail="position 3 -> position 1",
            tokens_before=500,
            tokens_after=500,
        )
        assert mutation.action == "moved"
        assert mutation.tokens_before == mutation.tokens_after

    def test_create_dedup_mutation(self) -> None:
        mutation = Mutation(
            step="DeduplicateStep",
            action="removed",
            detail='chunk "auth-overview" overlaps with "auth-detail" (82%)',
            tokens_before=1200,
            tokens_after=0,
        )
        assert "82%" in mutation.detail

    def test_timestamp_auto_set(self) -> None:
        mutation = Mutation(
            step="test",
            action="test",
            detail="test",
            tokens_before=0,
            tokens_after=0,
        )
        assert isinstance(mutation.timestamp, datetime)

    def test_serialization_to_dict(self) -> None:
        mutation = Mutation(
            step="TrimStep",
            action="removed",
            detail="test",
            tokens_before=100,
            tokens_after=0,
            before_content="old",
            after_content="",
        )
        data = mutation.model_dump()
        assert data["step"] == "TrimStep"
        assert data["before_content"] == "old"

    def test_serialization_to_json(self) -> None:
        mutation = Mutation(
            step="test",
            action="test",
            detail="test",
            tokens_before=0,
            tokens_after=0,
        )
        json_str = mutation.model_dump_json()
        assert "test" in json_str
