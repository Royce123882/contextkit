"""Tests for observability: provenance tracking (Origin, Mutation),
context quality scoring, and context sufficiency checking.
"""

from __future__ import annotations

from datetime import datetime

from contextkit.core import BlockType, ContextBlock
from contextkit.observe.events import ContextEvent, clear_handlers, register_handler
from contextkit.observe.provenance import Mutation, Origin
from contextkit.observe.quality import QualityScorer
from contextkit.observe.quality_models import PositionScore, QualityReport
from contextkit.observe.sufficiency import SufficiencyChecker
from contextkit.observe.sufficiency_models import SufficiencyResult


def _make_block(
    name: str,
    content: str = "test content",
    priority: int = 50,
    block_type: BlockType = BlockType.USER_CONTEXT,
    origin: Origin | None = None,
) -> ContextBlock:
    """Create a ContextBlock with sensible defaults for testing."""
    return ContextBlock(
        type=block_type,
        content=content,
        priority=priority,
        name=name,
        origin=origin,
    )


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


# ---------------------------------------------------------------------------
# Context quality scoring (positional attention analysis)
# ---------------------------------------------------------------------------


class TestContextQualityScorer:
    """QualityScorer evaluates how well high-priority content aligns with
    high-attention positions in the context window.
    """

    def test_empty_block_list_yields_perfect_score(self) -> None:
        """An empty block list has nothing misplaced, so score is 1.0."""
        scorer = QualityScorer()
        report = scorer.score([])
        assert report.overall_score == 1.0
        assert report.positional_scores == []
        assert report.warnings == []

    def test_single_block_receives_full_attention(self) -> None:
        """A lone block should receive attention_weight of 1.0."""
        scorer = QualityScorer()
        blocks = [_make_block("only_block", priority=80)]
        report = scorer.score(blocks)
        assert report.overall_score > 0.9
        assert len(report.positional_scores) == 1
        assert report.positional_scores[0].attention_weight == 1.0

    def test_middle_position_receives_lower_attention(self) -> None:
        """Blocks in the middle of a long sequence have lower attention weight."""
        scorer = QualityScorer()
        blocks = [_make_block(f"block_{i}", priority=50) for i in range(11)]
        report = scorer.score(blocks)
        middle_block = report.positional_scores[5]
        first_block = report.positional_scores[0]
        assert middle_block.attention_weight < first_block.attention_weight

    def test_warns_when_high_priority_block_in_low_attention_position(self) -> None:
        """A high-priority block buried in a low-attention position triggers a warning."""
        scorer = QualityScorer(
            high_priority_threshold=70,
            low_attention_threshold=0.5,
        )
        blocks = [_make_block(f"filler_{i}", priority=30) for i in range(11)]
        blocks[5] = _make_block("important_block", priority=90)
        report = scorer.score(blocks)
        assert len(report.warnings) >= 1
        assert "important_block" in report.warnings[0]

    def test_returns_pydantic_model_types(self) -> None:
        """Report and position scores are proper Pydantic models."""
        scorer = QualityScorer()
        blocks = [_make_block("single_block", priority=50)]
        report = scorer.score(blocks)
        assert isinstance(report, QualityReport)
        assert isinstance(report.positional_scores[0], PositionScore)

    def test_overall_score_stays_within_unit_range(self) -> None:
        """Overall score is always between 0.0 and 1.0."""
        scorer = QualityScorer()
        blocks = [_make_block(f"block_{i}", priority=50) for i in range(20)]
        report = scorer.score(blocks)
        assert 0.0 <= report.overall_score <= 1.0


# ---------------------------------------------------------------------------
# Context sufficiency checking
# ---------------------------------------------------------------------------


class TestContextSufficiencyChecker:
    """SufficiencyChecker determines whether the assembled context
    adequately covers the user's query.
    """

    def test_context_covering_query_is_sufficient(self) -> None:
        """Blocks that contain query keywords produce a sufficient result."""
        checker = SufficiencyChecker(min_query_coverage=0.5)
        blocks = [
            _make_block("python_tutorial", content="python programming language tutorial"),
        ]
        result = checker.check("python programming", blocks)
        assert result.sufficient is True
        assert result.query_coverage >= 0.5

    def test_context_missing_query_terms_is_insufficient(self) -> None:
        """Blocks unrelated to the query produce an insufficient result."""
        checker = SufficiencyChecker(min_query_coverage=0.8)
        blocks = [
            _make_block("weather_block", content="weather forecast"),
        ]
        result = checker.check("python programming tutorial", blocks)
        assert result.sufficient is False
        assert len(result.suggestions) >= 1

    def test_rag_relevance_scores_factor_into_check(self) -> None:
        """RAG blocks with origin relevance scores affect avg_relevance."""
        checker = SufficiencyChecker()
        blocks = [
            _make_block(
                "rag_python_examples",
                content="python code examples",
                block_type=BlockType.RAG,
                origin=Origin(
                    source="rag",
                    details={"relevance_score": 0.9},
                ),
            ),
        ]
        result = checker.check("python code", blocks)
        assert result.avg_relevance > 0.0

    def test_multiple_source_types_detected(self) -> None:
        """Blocks from different sources show up in source_types."""
        checker = SufficiencyChecker()
        blocks = [
            _make_block("system_instructions", content="system instructions", block_type=BlockType.SYSTEM_PROMPT),
            _make_block("rag_content", content="retrieved content", block_type=BlockType.RAG),
            _make_block("memory_pref", content="user preference", block_type=BlockType.LONG_TERM_MEMORY),
        ]
        result = checker.check("instructions", blocks)
        assert len(result.source_types) >= 3

    def test_emits_event_when_context_insufficient(self) -> None:
        """An insufficient result fires a CONTEXT_INSUFFICIENT event."""
        clear_handlers()
        captured_events = []
        register_handler(
            ContextEvent.CONTEXT_INSUFFICIENT,
            lambda event: captured_events.append(event),
        )

        checker = SufficiencyChecker(min_query_coverage=0.99)
        blocks = [_make_block("unrelated_block", content="unrelated")]
        checker.check("specific technical query", blocks)

        assert len(captured_events) == 1
        clear_handlers()

    def test_returns_sufficiency_result_model(self) -> None:
        """The return value is a SufficiencyResult Pydantic model."""
        checker = SufficiencyChecker()
        result = checker.check("test", [_make_block("test_block", content="test")])
        assert isinstance(result, SufficiencyResult)

    def test_confidence_stays_within_unit_range(self) -> None:
        """Confidence is always between 0.0 and 1.0."""
        checker = SufficiencyChecker()
        result = checker.check("test query", [_make_block("test_block", content="test")])
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Model-adaptive QualityScorer (curve_depth and for_model)
# ---------------------------------------------------------------------------


class TestModelAdaptiveQualityScorer:
    """Tests for model-adaptive attention curve depth in QualityScorer."""

    def test_default_curve_depth(self) -> None:
        scorer = QualityScorer()
        blocks = [_make_block(f"b{i}", priority=80) for i in range(10)]
        report = scorer.score(blocks)
        assert 0.0 <= report.overall_score <= 1.0

    def test_shallow_curve_higher_scores(self) -> None:
        """A shallower curve should give higher scores for middle blocks."""
        blocks = [_make_block(f"b{i}", priority=80) for i in range(10)]

        standard_scorer = QualityScorer(curve_depth=0.6)
        shallow_scorer = QualityScorer(curve_depth=0.3)

        standard_report = standard_scorer.score(blocks)
        shallow_report = shallow_scorer.score(blocks)

        assert shallow_report.overall_score >= standard_report.overall_score

    def test_for_model_factory(self) -> None:
        scorer = QualityScorer.for_model("claude-opus-4-6")
        assert scorer._curve_depth == 0.3

    def test_for_model_unknown_model(self) -> None:
        scorer = QualityScorer.for_model("nonexistent-model")
        assert scorer._curve_depth == 0.6

    def test_for_model_standard_profile(self) -> None:
        scorer = QualityScorer.for_model("gpt-4o")
        assert scorer._curve_depth == 0.6


# ---------------------------------------------------------------------------
# Effective window lint check
# ---------------------------------------------------------------------------


class TestEffectiveWindowLint:
    """Tests for the exceeds_effective_window lint check."""

    def test_no_warning_below_effective_limit(self) -> None:
        from contextkit.core import ContextWindow
        from contextkit.observe.linter import ContextLinter

        window = ContextWindow(model="gpt-4o")
        window.add(_make_block("small", content="hello"))
        linter = ContextLinter()
        warnings = linter.lint(window)
        effective_warnings = [w for w in warnings if w.code == "exceeds_effective_window"]
        assert len(effective_warnings) == 0

    def test_no_warning_for_manual_window(self) -> None:
        from contextkit.core import ContextWindow
        from contextkit.observe.linter import ContextLinter

        window = ContextWindow(max_tokens=100_000)
        linter = ContextLinter()
        warnings = linter.lint(window)
        effective_warnings = [w for w in warnings if w.code == "exceeds_effective_window"]
        assert len(effective_warnings) == 0
