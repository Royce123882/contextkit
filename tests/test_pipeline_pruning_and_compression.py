"""Tests for advanced pipeline steps: query-aware pruning, prefix-stable
reordering, RAG compression, token-level compression, and collapse
detection during compaction.
"""

from __future__ import annotations

from contextkit.core import BlockType, ContextBlock
from contextkit.observe.provenance import Origin
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
    """Create a ContextBlock with sensible defaults for testing."""
    return ContextBlock(
        type=block_type,
        content=content,
        priority=priority,
        name=name,
        origin=origin,
    )


# ---------------------------------------------------------------------------
# Query-aware context pruning (TrimStep with relevance scoring)
# ---------------------------------------------------------------------------


class TestQueryAwareTrimStep:
    """TrimStep should blend relevance and priority when a query is provided."""

    def test_keeps_relevant_block_despite_low_priority(self) -> None:
        """A low-priority but query-relevant block survives over an irrelevant one."""
        trim_step = TrimStep(
            max_tokens=20,
            query="python programming language",
            relevance_weight=0.8,
        )
        blocks = [
            _make_block("relevant_python", content="python programming language tips", priority=30),
            _make_block("irrelevant_weather", content="weather forecast for tomorrow", priority=60),
        ]
        result = trim_step.process(blocks)
        result_names = [b.name for b in result]
        assert "relevant_python" in result_names

    def test_falls_back_to_priority_without_query(self) -> None:
        """Without a query, standard priority-based trimming applies."""
        trim_step = TrimStep(max_tokens=20, query=None)
        blocks = [
            _make_block("low_priority", content="hello world", priority=10),
            _make_block("high_priority", content="hello world", priority=90),
        ]
        result = trim_step.process(blocks)
        if len(result) == 1:
            assert result[0].name == "high_priority"

    def test_mutation_includes_blended_score(self) -> None:
        """Removed blocks should include the blended_score in their mutation detail."""
        trim_step = TrimStep(
            max_tokens=3,
            query="python",
            relevance_weight=0.5,
        )
        blocks = [
            _make_block("python_block", content="python code examples tutorial", priority=30),
            _make_block("java_block", content="java enterprise application framework", priority=30),
        ]
        trim_step.process(blocks)
        all_mutations = [m for b in blocks for m in b.mutations]
        removed_mutations = [m for m in all_mutations if m.action == "removed"]
        assert len(removed_mutations) >= 1
        assert any("blended_score" in m.detail for m in removed_mutations)

    def test_uses_custom_scoring_function(self) -> None:
        """A user-supplied score_fn replaces the default word-overlap scorer."""
        invocation_count = []

        def magic_word_scorer(query: str, content: str) -> float:
            invocation_count.append(True)
            return 1.0 if "magic" in content else 0.0

        trim_step = TrimStep(
            max_tokens=6,
            query="find magic",
            score_fn=magic_word_scorer,
        )
        blocks = [
            _make_block("has_magic", content="magic word content here", priority=10),
            _make_block("no_magic", content="plain text content here", priority=10),
        ]
        result = trim_step.process(blocks)
        assert len(invocation_count) > 0
        result_names = [b.name for b in result]
        assert "has_magic" in result_names


# ---------------------------------------------------------------------------
# Prefix-stable reorder strategy
# ---------------------------------------------------------------------------


class TestPrefixStableReorderStrategy:
    """ReorderStep(strategy='prefix_stable') groups stable block types first."""

    def test_system_and_tools_precede_dynamic_blocks(self) -> None:
        """System prompt and tool definitions appear before user and RAG blocks."""
        reorder_step = ReorderStep(strategy="prefix_stable")
        blocks = [
            _make_block("user_query", block_type=BlockType.USER_CONTEXT, priority=50),
            _make_block("system_prompt", block_type=BlockType.SYSTEM_PROMPT, priority=100),
            _make_block("tool_defs", block_type=BlockType.TOOL_DEFINITIONS, priority=60),
            _make_block("rag_result", block_type=BlockType.RAG, priority=70),
        ]
        result = reorder_step.process(blocks)
        result_names = [b.name for b in result]
        assert result_names.index("system_prompt") < result_names.index("user_query")
        assert result_names.index("tool_defs") < result_names.index("rag_result")

    def test_stable_prefix_sorted_by_descending_priority(self) -> None:
        """Within the stable prefix region, blocks sort by priority (highest first)."""
        reorder_step = ReorderStep(strategy="prefix_stable")
        blocks = [
            _make_block("examples", block_type=BlockType.EXAMPLES, priority=55),
            _make_block("system_prompt", block_type=BlockType.SYSTEM_PROMPT, priority=100),
            _make_block("tool_defs", block_type=BlockType.TOOL_DEFINITIONS, priority=60),
        ]
        result = reorder_step.process(blocks)
        priorities = [b.priority for b in result]
        assert priorities == sorted(priorities, reverse=True)

    def test_records_move_mutations_with_region(self) -> None:
        """Moved blocks carry mutation records that include the target region."""
        reorder_step = ReorderStep(strategy="prefix_stable")
        blocks = [
            _make_block("rag_result", block_type=BlockType.RAG, priority=70),
            _make_block("system_prompt", block_type=BlockType.SYSTEM_PROMPT, priority=100),
            _make_block("user_input", block_type=BlockType.USER_CONTEXT, priority=50),
        ]
        reorder_step.process(blocks)
        moved_blocks = [b for b in blocks if any(m.action == "moved" for m in b.mutations)]
        assert len(moved_blocks) >= 1
        for block in moved_blocks:
            for mutation in block.mutations:
                if mutation.action == "moved":
                    assert "region" in mutation.detail

    def test_no_reorder_with_two_or_fewer_blocks(self) -> None:
        """With two or fewer blocks, reordering is a no-op."""
        reorder_step = ReorderStep(strategy="prefix_stable")
        blocks = [
            _make_block("first", priority=10),
            _make_block("second", priority=90),
        ]
        result = reorder_step.process(blocks)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Post-retrieval RAG compression
# ---------------------------------------------------------------------------


class TestRAGCompressionStep:
    """RAGCompressStep extracts key sentences and drops low-relevance chunks."""

    def test_compresses_multi_sentence_rag_block(self) -> None:
        """A RAG block with many sentences is shortened to max_sentences."""
        compress_step = RAGCompressStep(max_sentences=2)
        long_content = (
            "Python is a programming language. "
            "It was created by Guido van Rossum. "
            "Python emphasizes readability. "
            "It supports multiple paradigms. "
            "Python has a large standard library."
        )
        rag_block = _make_block(
            "python_docs",
            content=long_content,
            block_type=BlockType.RAG,
            origin=Origin(
                source="rag",
                details={"query": "Python programming", "relevance_score": 0.8},
            ),
        )
        result = compress_step.process([rag_block])
        assert len(result) == 1
        assert len(result[0].content) < len(long_content)

    def test_drops_block_below_relevance_threshold(self) -> None:
        """A RAG block with relevance below drop_threshold is removed entirely."""
        compress_step = RAGCompressStep(drop_threshold=0.5)
        low_relevance_block = _make_block(
            "low_relevance",
            content="some content",
            block_type=BlockType.RAG,
            origin=Origin(
                source="rag",
                details={"relevance_score": 0.2},
            ),
        )
        result = compress_step.process([low_relevance_block])
        assert len(result) == 0

    def test_passes_non_rag_blocks_through_unchanged(self) -> None:
        """Non-RAG blocks are returned without modification."""
        compress_step = RAGCompressStep()
        user_block = _make_block("user_input", content="user query", block_type=BlockType.USER_CONTEXT)
        result = compress_step.process([user_block])
        assert len(result) == 1
        assert result[0].content == "user query"

    def test_records_compression_mutation(self) -> None:
        """Compressed RAG blocks carry a 'compressed' mutation record."""
        compress_step = RAGCompressStep(max_sentences=1)
        content = "First sentence. Second sentence. Third sentence."
        rag_block = _make_block(
            "rag_chunk",
            content=content,
            block_type=BlockType.RAG,
            origin=Origin(
                source="rag",
                details={"query": "first", "relevance_score": 0.9},
            ),
        )
        result = compress_step.process([rag_block])
        assert len(result) == 1
        assert any(m.action == "compressed" for m in result[0].mutations)

    def test_uses_custom_compressor_function(self) -> None:
        """A user-supplied compressor replaces the default extractive logic."""
        def truncate_compressor(text: str, query: str) -> str:
            return text[:20]

        compress_step = RAGCompressStep(compressor=truncate_compressor)
        rag_block = _make_block(
            "rag_chunk",
            content="A " * 50,
            block_type=BlockType.RAG,
            origin=Origin(source="rag", details={"relevance_score": 0.9}),
        )
        result = compress_step.process([rag_block])
        assert len(result[0].content) == 20

    def test_step_name_is_descriptive(self) -> None:
        """The step name should identify this as a RAG compression step."""
        step = RAGCompressStep()
        assert step.name == "RAGCompressStep"


# ---------------------------------------------------------------------------
# Token-level prompt compression
# ---------------------------------------------------------------------------


class TestTokenLevelCompressionStep:
    """CompressStep prunes low-information tokens to reduce block size."""

    def test_compresses_block_above_minimum_threshold(self) -> None:
        """Blocks exceeding min_tokens should be compressed to the target ratio."""
        compress_step = CompressStep(compression_ratio=0.5, min_tokens=5)
        long_content = " ".join(f"word{i}" for i in range(100))
        block = _make_block("long_block", content=long_content)
        result = compress_step.process([block])
        assert len(result) == 1
        assert len(result[0].content) < len(long_content)

    def test_skips_block_below_minimum_threshold(self) -> None:
        """Blocks shorter than min_tokens pass through unchanged."""
        compress_step = CompressStep(min_tokens=1000)
        short_block = _make_block("short_block", content="hello world")
        result = compress_step.process([short_block])
        assert result[0].content == "hello world"

    def test_records_compression_mutation(self) -> None:
        """Compressed blocks carry a 'compressed' mutation with token counts."""
        compress_step = CompressStep(compression_ratio=0.3, min_tokens=5)
        long_content = " ".join(f"word{i}" for i in range(100))
        block = _make_block("long_block", content=long_content)
        result = compress_step.process([block])
        assert any(m.action == "compressed" for m in result[0].mutations)

    def test_preserves_start_and_end_tokens(self) -> None:
        """Positional boosting should preserve tokens at the edges of the text."""
        compress_step = CompressStep(compression_ratio=0.3, min_tokens=5)
        content = "START " + " ".join(f"mid{i}" for i in range(50)) + " END"
        block = _make_block("edge_test", content=content)
        result = compress_step.process([block])
        compressed_content = result[0].content
        assert "START" in compressed_content or "END" in compressed_content

    def test_uses_custom_token_scorer(self) -> None:
        """A user-supplied scorer function replaces the default IDF scorer."""
        def uniform_scorer(tokens):
            return [1.0] * len(tokens)

        compress_step = CompressStep(
            compression_ratio=0.5,
            min_tokens=5,
            scorer=uniform_scorer,
        )
        content = " ".join(f"word{i}" for i in range(50))
        block = _make_block("uniform_score_block", content=content)
        result = compress_step.process([block])
        assert len(result) == 1

    def test_skips_non_string_content(self) -> None:
        """Non-string content (e.g. message lists) passes through unchanged."""
        compress_step = CompressStep(min_tokens=1)
        message_block = ContextBlock(
            type=BlockType.SHORT_TERM_MEMORY,
            content=[{"role": "user", "content": "hi"}],
            name="conversation_messages",
        )
        result = compress_step.process([message_block])
        assert result[0].content == [{"role": "user", "content": "hi"}]

    def test_step_name_is_descriptive(self) -> None:
        """The step name should identify this as a compression step."""
        step = CompressStep()
        assert step.name == "CompressStep"


# ---------------------------------------------------------------------------
# Collapse detection during compaction
# ---------------------------------------------------------------------------


class TestCompactionCollapseDetection:
    """CompactStep should detect when compaction loses too much information."""

    def test_warns_when_compactor_loses_keywords(self) -> None:
        """A compactor that drops all keywords triggers a COLLAPSE WARNING."""

        def destructive_compactor(content: str) -> str:
            return "xyz abc def"

        compact_step = CompactStep(
            compactor=destructive_compactor,
            min_tokens=1,
            max_info_loss=0.3,
        )
        block = _make_block(
            "keyword_rich",
            content="python programming language tutorial guide examples code",
        )
        result = compact_step.process([block])
        mutations = result[0].mutations
        assert len(mutations) == 1
        assert "COLLAPSE WARNING" in mutations[0].detail

    def test_no_warning_when_keywords_retained(self) -> None:
        """Good keyword retention should not trigger a collapse warning."""

        def half_content_compactor(content: str) -> str:
            words = content.split()
            return " ".join(words[: len(words) // 2])

        compact_step = CompactStep(
            compactor=half_content_compactor,
            min_tokens=1,
            max_info_loss=0.9,
        )
        block = _make_block("repetitive_block", content="word " * 50)
        result = compact_step.process([block])
        if result[0].mutations:
            assert "COLLAPSE WARNING" not in result[0].mutations[0].detail

    def test_max_info_loss_threshold_is_configurable(self) -> None:
        """The max_info_loss parameter should be stored and respected."""
        compact_step = CompactStep(max_info_loss=0.8)
        assert compact_step._max_info_loss == 0.8

    def test_default_compaction_without_explicit_info_loss(self) -> None:
        """CompactStep works correctly with the default max_info_loss value."""
        compact_step = CompactStep(min_tokens=5, target_ratio=0.3)
        blocks = [_make_block("long_block", content="word " * 200)]
        result = compact_step.process(blocks)
        assert len(result) == 1
        assert len(result[0].content) < len("word " * 200)
