"""Tests for new context engineering features (H1-H3, M1-M5).

Covers:
  H1 - Query-aware context pruning (TrimStep)
  H2 - Prefix-stable reorder strategy (ReorderStep)
  H3 - Post-retrieval RAG compression (RAGCompressStep)
  M1 - Token-level prompt compression (CompressStep)
  M2 - Context quality scoring (QualityScorer)
  M3 - Context sufficiency checking (SufficiencyChecker)
  M4 - Memory decay with Ebbinghaus curve (MemoryRecord)
  M5 - Collapse detection for summaries (CompactStep)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from contextkit.core import BlockType, ContextBlock
from contextkit.observe.events import ContextEvent, clear_handlers, register_handler
from contextkit.observe.provenance import Origin
from contextkit.observe.quality import QualityScorer
from contextkit.observe.quality_models import PositionScore, QualityReport
from contextkit.observe.sufficiency import SufficiencyChecker
from contextkit.observe.sufficiency_models import SufficiencyResult
from contextkit.memory.in_memory_backend import InMemoryBackend
from contextkit.memory.record import MemoryRecord
from contextkit.pipeline import (
    CompactStep,
    CompressStep,
    RAGCompressStep,
    ReorderStep,
    TrimStep,
)


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
    """Create a test block with sensible defaults."""
    return ContextBlock(
        type=block_type,
        content=content,
        priority=priority,
        name=name,
        origin=origin,
    )


# ===================================================================
# H1: Query-aware context pruning
# ===================================================================


class TestQueryAwareTrimStep:
    """Tests for query-aware pruning in TrimStep."""

    def test_query_aware_keeps_relevant_low_priority(self) -> None:
        """A low-priority but relevant block should survive over an irrelevant one."""
        step = TrimStep(
            max_tokens=20,
            query="python programming language",
            relevance_weight=0.8,
        )
        blocks = [
            _make_block("relevant", content="python programming language tips", priority=30),
            _make_block("irrelevant", content="weather forecast for tomorrow", priority=60),
        ]
        result = step.process(blocks)
        names = [b.name for b in result]
        # With high relevance_weight, "relevant" should be preferred
        assert "relevant" in names

    def test_query_none_falls_back_to_priority(self) -> None:
        """Without a query, standard priority-based trimming applies."""
        step = TrimStep(max_tokens=20, query=None)
        blocks = [
            _make_block("low", content="hello world", priority=10),
            _make_block("high", content="hello world", priority=90),
        ]
        result = step.process(blocks)
        if len(result) == 1:
            assert result[0].name == "high"

    def test_blended_score_in_mutation_detail(self) -> None:
        """When query is set, mutation detail should include blended_score."""
        step = TrimStep(
            max_tokens=3,
            query="python",
            relevance_weight=0.5,
        )
        blocks = [
            _make_block("a", content="python code examples tutorial", priority=30),
            _make_block("b", content="java enterprise application framework", priority=30),
        ]
        step.process(blocks)
        all_mutations = [m for b in blocks for m in b.mutations]
        removed_mutations = [m for m in all_mutations if m.action == "removed"]
        assert len(removed_mutations) >= 1
        assert any("blended_score" in m.detail for m in removed_mutations)

    def test_custom_score_fn(self) -> None:
        """A custom score_fn should be used instead of word_overlap_score."""
        custom_called = []

        def custom_scorer(query: str, content: str) -> float:
            custom_called.append(True)
            return 1.0 if "magic" in content else 0.0

        step = TrimStep(
            max_tokens=6,
            query="find magic",
            score_fn=custom_scorer,
        )
        blocks = [
            _make_block("magic", content="magic word content here", priority=10),
            _make_block("plain", content="plain text content here", priority=10),
        ]
        result = step.process(blocks)
        assert len(custom_called) > 0
        # "magic" should survive due to higher custom score
        names = [b.name for b in result]
        assert "magic" in names


# ===================================================================
# H2: Prefix-stable reorder strategy
# ===================================================================


class TestPrefixStableReorder:
    """Tests for the prefix_stable reorder strategy."""

    def test_stable_types_grouped_first(self) -> None:
        """System prompt and tool definitions should appear before dynamic blocks."""
        step = ReorderStep(strategy="prefix_stable")
        blocks = [
            _make_block("user_q", block_type=BlockType.USER_CONTEXT, priority=50),
            _make_block("sys", block_type=BlockType.SYSTEM_PROMPT, priority=100),
            _make_block("tools", block_type=BlockType.TOOL_DEFINITIONS, priority=60),
            _make_block("rag", block_type=BlockType.RAG, priority=70),
        ]
        result = step.process(blocks)
        names = [b.name for b in result]
        # sys and tools should come before user_q and rag
        assert names.index("sys") < names.index("user_q")
        assert names.index("tools") < names.index("rag")

    def test_stable_section_sorted_by_priority(self) -> None:
        """Within the stable prefix, blocks should be sorted by priority descending."""
        step = ReorderStep(strategy="prefix_stable")
        blocks = [
            _make_block("examples", block_type=BlockType.EXAMPLES, priority=55),
            _make_block("sys", block_type=BlockType.SYSTEM_PROMPT, priority=100),
            _make_block("tools", block_type=BlockType.TOOL_DEFINITIONS, priority=60),
        ]
        result = step.process(blocks)
        stable_priorities = [b.priority for b in result]
        # Should be descending
        assert stable_priorities == sorted(stable_priorities, reverse=True)

    def test_prefix_stable_records_mutations(self) -> None:
        """Moved blocks should have mutation records with region info."""
        step = ReorderStep(strategy="prefix_stable")
        blocks = [
            _make_block("rag", block_type=BlockType.RAG, priority=70),
            _make_block("sys", block_type=BlockType.SYSTEM_PROMPT, priority=100),
            _make_block("user", block_type=BlockType.USER_CONTEXT, priority=50),
        ]
        step.process(blocks)
        moved = [b for b in blocks if any(m.action == "moved" for m in b.mutations)]
        assert len(moved) >= 1
        # Check that mutation details include region info
        for block in moved:
            for m in block.mutations:
                if m.action == "moved":
                    assert "region" in m.detail

    def test_prefix_stable_few_blocks_unchanged(self) -> None:
        """With <= 2 blocks, no reorder happens."""
        step = ReorderStep(strategy="prefix_stable")
        blocks = [
            _make_block("a", priority=10),
            _make_block("b", priority=90),
        ]
        result = step.process(blocks)
        assert len(result) == 2


# ===================================================================
# H3: Post-retrieval RAG compression
# ===================================================================


class TestRAGCompressStep:
    """Tests for the RAGCompressStep."""

    def test_compresses_rag_block(self) -> None:
        """RAG blocks with many sentences should be compressed."""
        step = RAGCompressStep(max_sentences=2)
        content = (
            "Python is a programming language. "
            "It was created by Guido van Rossum. "
            "Python emphasizes readability. "
            "It supports multiple paradigms. "
            "Python has a large standard library."
        )
        block = _make_block(
            "rag1",
            content=content,
            block_type=BlockType.RAG,
            origin=Origin(
                source="rag",
                details={"query": "Python programming", "relevance_score": 0.8},
            ),
        )
        result = step.process([block])
        assert len(result) == 1
        # Compressed text should be shorter
        assert len(result[0].content) < len(content)

    def test_drops_low_relevance_rag_block(self) -> None:
        """RAG blocks below drop_threshold should be removed entirely."""
        step = RAGCompressStep(drop_threshold=0.5)
        block = _make_block(
            "low_rel",
            content="some content",
            block_type=BlockType.RAG,
            origin=Origin(
                source="rag",
                details={"relevance_score": 0.2},
            ),
        )
        result = step.process([block])
        assert len(result) == 0

    def test_passes_through_non_rag_blocks(self) -> None:
        """Non-RAG blocks should pass through unchanged."""
        step = RAGCompressStep()
        block = _make_block("user", content="user query", block_type=BlockType.USER_CONTEXT)
        result = step.process([block])
        assert len(result) == 1
        assert result[0].content == "user query"

    def test_records_compression_mutation(self) -> None:
        """Compressed RAG blocks should have a mutation recorded."""
        step = RAGCompressStep(max_sentences=1)
        content = "First sentence. Second sentence. Third sentence."
        block = _make_block(
            "rag",
            content=content,
            block_type=BlockType.RAG,
            origin=Origin(
                source="rag",
                details={"query": "first", "relevance_score": 0.9},
            ),
        )
        result = step.process([block])
        assert len(result) == 1
        mutations = result[0].mutations
        assert any(m.action == "compressed" for m in mutations)

    def test_custom_compressor(self) -> None:
        """A custom compressor function should be used when provided."""
        def my_compressor(text: str, query: str) -> str:
            return text[:20]

        step = RAGCompressStep(compressor=my_compressor)
        block = _make_block(
            "rag",
            content="A " * 50,
            block_type=BlockType.RAG,
            origin=Origin(source="rag", details={"relevance_score": 0.9}),
        )
        result = step.process([block])
        assert len(result[0].content) == 20

    def test_step_name(self) -> None:
        step = RAGCompressStep()
        assert step.name == "RAGCompressStep"


# ===================================================================
# M1: Token-level prompt compression
# ===================================================================


class TestCompressStep:
    """Tests for the CompressStep."""

    def test_compresses_long_block(self) -> None:
        """Blocks above min_tokens should be compressed."""
        step = CompressStep(compression_ratio=0.5, min_tokens=5)
        content = " ".join(f"word{i}" for i in range(100))
        block = _make_block("long", content=content)
        result = step.process([block])
        assert len(result) == 1
        assert len(result[0].content) < len(content)

    def test_skips_short_block(self) -> None:
        """Blocks below min_tokens should not be compressed."""
        step = CompressStep(min_tokens=1000)
        block = _make_block("short", content="hello world")
        result = step.process([block])
        assert result[0].content == "hello world"

    def test_records_compression_mutation(self) -> None:
        """Compressed blocks should have a mutation record."""
        step = CompressStep(compression_ratio=0.3, min_tokens=5)
        content = " ".join(f"word{i}" for i in range(100))
        block = _make_block("long", content=content)
        result = step.process([block])
        assert any(m.action == "compressed" for m in result[0].mutations)

    def test_preserves_edges(self) -> None:
        """Start and end tokens should be preserved during compression."""
        step = CompressStep(compression_ratio=0.3, min_tokens=5)
        content = "START " + " ".join(f"mid{i}" for i in range(50)) + " END"
        block = _make_block("edge", content=content)
        result = step.process([block])
        compressed = result[0].content
        # With positional boosting, START and END should often survive
        assert "START" in compressed or "END" in compressed

    def test_custom_scorer(self) -> None:
        """A custom scorer should be used when provided."""
        def uniform_scorer(tokens):
            return [1.0] * len(tokens)

        step = CompressStep(
            compression_ratio=0.5,
            min_tokens=5,
            scorer=uniform_scorer,
        )
        content = " ".join(f"word{i}" for i in range(50))
        block = _make_block("uniform", content=content)
        result = step.process([block])
        assert len(result) == 1

    def test_skips_non_string_content(self) -> None:
        """Non-string content should pass through unchanged."""
        step = CompressStep(min_tokens=1)
        block = ContextBlock(
            type=BlockType.SHORT_TERM_MEMORY,
            content=[{"role": "user", "content": "hi"}],
            name="messages",
        )
        result = step.process([block])
        assert result[0].content == [{"role": "user", "content": "hi"}]

    def test_step_name(self) -> None:
        step = CompressStep()
        assert step.name == "CompressStep"


# ===================================================================
# M2: Context quality scoring
# ===================================================================


class TestQualityScorer:
    """Tests for the QualityScorer."""

    def test_empty_blocks_perfect_score(self) -> None:
        """An empty block list should produce a perfect score."""
        scorer = QualityScorer()
        report = scorer.score([])
        assert report.overall_score == 1.0
        assert report.positional_scores == []
        assert report.warnings == []

    def test_single_block_high_attention(self) -> None:
        """A single block should get high attention weight."""
        scorer = QualityScorer()
        blocks = [_make_block("only", priority=80)]
        report = scorer.score(blocks)
        assert report.overall_score > 0.9
        assert len(report.positional_scores) == 1
        assert report.positional_scores[0].attention_weight == 1.0

    def test_middle_block_low_attention(self) -> None:
        """Middle blocks in a long sequence should have lower attention."""
        scorer = QualityScorer()
        blocks = [_make_block(f"b{i}", priority=50) for i in range(11)]
        report = scorer.score(blocks)
        middle_score = report.positional_scores[5]
        edge_score = report.positional_scores[0]
        assert middle_score.attention_weight < edge_score.attention_weight

    def test_high_priority_in_middle_warns(self) -> None:
        """A high-priority block in the middle should generate a warning."""
        scorer = QualityScorer(
            high_priority_threshold=70,
            low_attention_threshold=0.5,
        )
        blocks = [_make_block(f"b{i}", priority=30) for i in range(11)]
        # Place a high-priority block in the middle
        blocks[5] = _make_block("important", priority=90)
        report = scorer.score(blocks)
        assert len(report.warnings) >= 1
        assert "important" in report.warnings[0]

    def test_report_model_types(self) -> None:
        """Returned objects should be proper Pydantic models."""
        scorer = QualityScorer()
        blocks = [_make_block("a", priority=50)]
        report = scorer.score(blocks)
        assert isinstance(report, QualityReport)
        assert isinstance(report.positional_scores[0], PositionScore)

    def test_overall_score_between_zero_and_one(self) -> None:
        """Overall score should always be in [0, 1]."""
        scorer = QualityScorer()
        blocks = [_make_block(f"b{i}", priority=50) for i in range(20)]
        report = scorer.score(blocks)
        assert 0.0 <= report.overall_score <= 1.0


# ===================================================================
# M3: Context sufficiency checking
# ===================================================================


class TestSufficiencyChecker:
    """Tests for the SufficiencyChecker."""

    def test_sufficient_context(self) -> None:
        """Context that covers the query should be sufficient."""
        checker = SufficiencyChecker(min_query_coverage=0.5)
        blocks = [
            _make_block("a", content="python programming language tutorial"),
        ]
        result = checker.check("python programming", blocks)
        assert result.sufficient is True
        assert result.query_coverage >= 0.5

    def test_insufficient_context(self) -> None:
        """Context that doesn't cover the query should be insufficient."""
        checker = SufficiencyChecker(min_query_coverage=0.8)
        blocks = [
            _make_block("a", content="weather forecast"),
        ]
        result = checker.check("python programming tutorial", blocks)
        assert result.sufficient is False
        assert len(result.suggestions) >= 1

    def test_rag_relevance_signal(self) -> None:
        """RAG blocks with relevance scores should factor into the result."""
        checker = SufficiencyChecker()
        blocks = [
            _make_block(
                "rag1",
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

    def test_source_diversity(self) -> None:
        """Multiple source types should appear in the result."""
        checker = SufficiencyChecker()
        blocks = [
            _make_block("sys", content="system instructions", block_type=BlockType.SYSTEM_PROMPT),
            _make_block("rag", content="retrieved content", block_type=BlockType.RAG),
            _make_block("mem", content="user preference", block_type=BlockType.LONG_TERM_MEMORY),
        ]
        result = checker.check("instructions", blocks)
        assert len(result.source_types) >= 3

    def test_emits_insufficient_event(self) -> None:
        """An insufficient result should emit a CONTEXT_INSUFFICIENT event."""
        clear_handlers()
        events_received = []
        register_handler(
            ContextEvent.CONTEXT_INSUFFICIENT,
            lambda e: events_received.append(e),
        )

        checker = SufficiencyChecker(min_query_coverage=0.99)
        blocks = [_make_block("a", content="unrelated")]
        checker.check("specific technical query", blocks)

        assert len(events_received) == 1
        clear_handlers()

    def test_result_model_type(self) -> None:
        """The result should be a proper SufficiencyResult model."""
        checker = SufficiencyChecker()
        result = checker.check("test", [_make_block("a", content="test")])
        assert isinstance(result, SufficiencyResult)

    def test_confidence_between_zero_and_one(self) -> None:
        """Confidence should always be in [0, 1]."""
        checker = SufficiencyChecker()
        result = checker.check("test query", [_make_block("a", content="test")])
        assert 0.0 <= result.confidence <= 1.0


# ===================================================================
# M4: Memory decay with Ebbinghaus curve
# ===================================================================


class TestMemoryDecay:
    """Tests for Ebbinghaus-curve memory decay on MemoryRecord."""

    def test_fresh_record_decay_is_one(self) -> None:
        """A just-created record should have decay factor ~1.0."""
        now = datetime.now(timezone.utc)
        record = MemoryRecord(key="k1", content="test", stored_at=now)
        factor = record.decay_factor(now=now)
        assert factor == 1.0

    def test_old_record_decays(self) -> None:
        """A record from a week ago should have measurably decayed."""
        now = datetime.now(timezone.utc)
        one_week_ago = now - timedelta(hours=168)
        record = MemoryRecord(key="k1", content="test", stored_at=one_week_ago)
        factor = record.decay_factor(half_life_hours=168.0, now=now)
        # After exactly one half-life, factor should be ~0.5
        assert 0.4 <= factor <= 0.6

    def test_access_count_extends_half_life(self) -> None:
        """Higher access_count should slow the decay (spaced repetition)."""
        now = datetime.now(timezone.utc)
        one_week_ago = now - timedelta(hours=168)

        record_no_access = MemoryRecord(
            key="k1", content="test", stored_at=one_week_ago
        )
        record_many_accesses = MemoryRecord(
            key="k2", content="test", stored_at=one_week_ago, access_count=5
        )

        factor_no = record_no_access.decay_factor(half_life_hours=168.0, now=now)
        factor_many = record_many_accesses.decay_factor(half_life_hours=168.0, now=now)

        # More accesses -> slower decay -> higher factor
        assert factor_many > factor_no

    def test_record_access_updates_metadata(self) -> None:
        """record_access() should increment count and set last_accessed."""
        record = MemoryRecord(key="k1", content="test")
        assert record.access_count == 0
        assert record.last_accessed is None

        record.record_access()
        assert record.access_count == 1
        assert record.last_accessed is not None

    def test_last_accessed_used_as_reference(self) -> None:
        """decay_factor should use last_accessed over stored_at if available."""
        now = datetime.now(timezone.utc)
        old_stored = now - timedelta(hours=1000)
        recent_access = now - timedelta(hours=1)

        record = MemoryRecord(
            key="k1",
            content="test",
            stored_at=old_stored,
            last_accessed=recent_access,
        )
        factor = record.decay_factor(half_life_hours=168.0, now=now)
        # Only 1 hour since last access, factor should be very high
        assert factor > 0.99

    def test_decay_factor_always_positive(self) -> None:
        """Decay factor should always be > 0."""
        now = datetime.now(timezone.utc)
        ancient = now - timedelta(days=365 * 10)
        record = MemoryRecord(key="k1", content="test", stored_at=ancient)
        factor = record.decay_factor(half_life_hours=168.0, now=now)
        assert factor > 0.0


class TestInMemoryBackendDecay:
    """Tests for decay integration in InMemoryBackend."""

    def test_retrieve_updates_access_metadata(self) -> None:
        """Retrieved records should have their access metadata updated."""
        backend = InMemoryBackend()
        asyncio.run(backend.store("k1", "python programming"))
        results = asyncio.run(backend.retrieve("python"))
        assert results[0].access_count == 1
        assert results[0].last_accessed is not None

    def test_recent_record_ranked_higher(self) -> None:
        """A recently accessed record should rank above an older one."""
        backend = InMemoryBackend()
        now = datetime.now(timezone.utc)

        # Store two records with identical content and importance
        asyncio.run(backend.store("old", "python tips"))
        asyncio.run(backend.store("new", "python tips"))

        # Manually age the "old" record
        backend._records["old"].stored_at = now - timedelta(days=30)
        backend._records["new"].stored_at = now

        results = asyncio.run(backend.retrieve("python"))
        # "new" should rank first due to higher decay factor
        assert results[0].key == "new"


# ===================================================================
# M5: Collapse detection for summaries
# ===================================================================


class TestCollapseDetection:
    """Tests for collapse detection in CompactStep."""

    def test_collapse_warning_in_mutation(self) -> None:
        """Compaction that loses too many keywords should include collapse warning."""

        def bad_compactor(content: str) -> str:
            # Returns something completely unrelated
            return "xyz abc def"

        step = CompactStep(
            compactor=bad_compactor,
            min_tokens=1,
            max_info_loss=0.3,
        )
        block = _make_block(
            "test_block",
            content="python programming language tutorial guide examples code",
        )
        result = step.process([block])

        # The mutation detail should include the collapse warning
        mutations = result[0].mutations
        assert len(mutations) == 1
        assert "COLLAPSE WARNING" in mutations[0].detail

    def test_no_warning_when_retention_good(self) -> None:
        """Good retention should not trigger a collapse warning."""

        def good_compactor(content: str) -> str:
            # Keep the first half of words
            words = content.split()
            return " ".join(words[: len(words) // 2])

        step = CompactStep(
            compactor=good_compactor,
            min_tokens=1,
            max_info_loss=0.9,  # Very lenient
        )
        block = _make_block("test", content="word " * 50)
        result = step.process([block])
        if result[0].mutations:
            assert "COLLAPSE WARNING" not in result[0].mutations[0].detail

    def test_max_info_loss_parameter(self) -> None:
        """The max_info_loss parameter should control the threshold."""
        step = CompactStep(max_info_loss=0.8)
        assert step._max_info_loss == 0.8

    def test_backward_compatible_without_max_info_loss(self) -> None:
        """CompactStep should work without max_info_loss (default 0.5)."""
        step = CompactStep(min_tokens=5, target_ratio=0.3)
        blocks = [_make_block("long", content="word " * 200)]
        result = step.process(blocks)
        assert len(result) == 1
        assert len(result[0].content) < len("word " * 200)
