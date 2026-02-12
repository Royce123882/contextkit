"""Tests for context quality scoring, sufficiency checking, and
temporal memory decay with spaced-repetition reinforcement.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from contextkit.core import BlockType, ContextBlock
from contextkit.memory.in_memory_backend import InMemoryBackend
from contextkit.memory.record import MemoryRecord
from contextkit.observe.events import ContextEvent, clear_handlers, register_handler
from contextkit.observe.provenance import Origin
from contextkit.observe.quality import QualityScorer
from contextkit.observe.quality_models import PositionScore, QualityReport
from contextkit.observe.sufficiency import SufficiencyChecker
from contextkit.observe.sufficiency_models import SufficiencyResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
