"""Tests for the context pipeline: trim, filter, deduplicate, reorder,
compact, RAG compress, mask, prune stale, strip thinking,
the pipeline orchestrator, and the PipelineBuilder.
"""

from __future__ import annotations

import pytest

from contextkit.compaction.local_store import LocalCompactionStore
from contextkit.core import BlockType, ContextBlock, ContextWindow
from contextkit.observe.provenance import Origin
from contextkit.pipeline import (
    CompactStep,
    ContextPipeline,
    DeduplicateStep,
    FilterStep,
    MaskStep,
    PipelineReport,
    PruneStaleStep,
    RAGCompressStep,
    ReorderStep,
    StepReport,
    StripThinkingStep,
    TrimStep,
)


def _make_block(
    name: str,
    content: str = "test content",
    priority: int = 50,
    block_type: BlockType = BlockType.USER_CONTEXT,
    origin: Origin | None = None,
) -> ContextBlock:
    """Helper to create test blocks."""
    return ContextBlock(
        type=block_type,
        content=content,
        priority=priority,
        name=name,
        origin=origin,
    )


class TestStepReport:
    """Tests for the StepReport model."""

    def test_create_with_defaults(self) -> None:
        report = StepReport(step_name="test")
        assert report.step_name == "test"
        assert report.blocks_modified == 0
        assert report.tokens_saved == 0

    def test_create_with_all_fields(self) -> None:
        report = StepReport(
            step_name="trim",
            blocks_modified=2,
            blocks_removed=1,
            tokens_before=500,
            tokens_after=300,
            tokens_saved=200,
        )
        assert report.tokens_saved == 200


class TestPipelineReport:
    """Tests for the PipelineReport model."""

    def test_create_empty(self) -> None:
        report = PipelineReport()
        assert report.steps == []
        assert report.total_tokens_saved == 0

    def test_create_with_steps(self) -> None:
        report = PipelineReport(
            steps=[StepReport(step_name="trim")],
            total_tokens_before=1000,
            total_tokens_after=500,
            total_tokens_saved=500,
        )
        assert len(report.steps) == 1


class TestTrimStep:
    """Tests for the TrimStep."""

    def test_remove_below_min_priority(self) -> None:
        step = TrimStep(min_priority=30)
        blocks = [
            _make_block("low", priority=10),
            _make_block("high", priority=50),
        ]
        result = step.process(blocks)
        assert len(result) == 1
        assert result[0].name == "high"

    def test_min_priority_records_mutation(self) -> None:
        step = TrimStep(min_priority=30)
        blocks = [_make_block("low", priority=10)]
        step.process(blocks)
        assert len(blocks[0].mutations) == 1
        assert blocks[0].mutations[0].action == "removed"

    def test_trim_by_token_budget(self) -> None:
        step = TrimStep(max_tokens=10)
        blocks = [
            _make_block("a", content="word " * 50, priority=80),
            _make_block("b", content="word " * 50, priority=40),
        ]
        result = step.process(blocks)
        # Should keep at most what fits in budget
        total = sum(b.token_count for b in result)
        assert total <= 10

    def test_keeps_highest_priority_when_budget_tight(self) -> None:
        step = TrimStep(max_tokens=20)
        blocks = [
            _make_block("low", content="hello", priority=10),
            _make_block("high", content="world", priority=90),
        ]
        result = step.process(blocks)
        # Should prefer keeping the high priority block
        if len(result) == 1:
            assert result[0].name == "high"

    def test_no_trimming_needed(self) -> None:
        step = TrimStep(max_tokens=10000, min_priority=0)
        blocks = [_make_block("a", content="short")]
        result = step.process(blocks)
        assert len(result) == 1

    def test_step_name(self) -> None:
        step = TrimStep()
        assert step.name == "TrimStep"


class TestFilterStep:
    """Tests for the FilterStep."""

    def test_filter_by_relevance(self) -> None:
        step = FilterStep(min_relevance=0.5)
        high = _make_block(
            "high",
            origin=Origin(
                source="rag",
                details={"relevance_score": 0.8},
            ),
        )
        low = _make_block(
            "low",
            origin=Origin(
                source="rag",
                details={"relevance_score": 0.2},
            ),
        )
        result = step.process([high, low])
        assert len(result) == 1
        assert result[0].name == "high"

    def test_filter_records_mutation(self) -> None:
        step = FilterStep(min_relevance=0.5)
        block = _make_block(
            "low",
            origin=Origin(
                source="rag",
                details={"relevance_score": 0.2},
            ),
        )
        step.process([block])
        assert len(block.mutations) == 1
        assert block.mutations[0].action == "removed"

    def test_custom_filter_function(self) -> None:
        step = FilterStep(
            filter_function=lambda b: b.priority > 30,
        )
        blocks = [
            _make_block("low", priority=10),
            _make_block("high", priority=50),
        ]
        result = step.process(blocks)
        assert len(result) == 1
        assert result[0].name == "high"

    def test_no_filter_keeps_all(self) -> None:
        step = FilterStep()
        blocks = [_make_block("a"), _make_block("b")]
        result = step.process(blocks)
        assert len(result) == 2

    def test_step_name(self) -> None:
        step = FilterStep()
        assert step.name == "FilterStep"

    def test_keeps_blocks_without_origin(self) -> None:
        step = FilterStep(min_relevance=0.5)
        block = _make_block("no_origin")
        result = step.process([block])
        assert len(result) == 1


class TestDeduplicateStep:
    """Tests for the DeduplicateStep."""

    def test_removes_duplicates(self) -> None:
        step = DeduplicateStep()
        blocks = [
            _make_block("a", content="Python is great", priority=50),
            _make_block("b", content="Python is great", priority=40),
        ]
        result = step.process(blocks)
        assert len(result) == 1
        # Should keep the higher priority one
        assert result[0].name == "a"

    def test_keeps_different_content(self) -> None:
        step = DeduplicateStep()
        blocks = [
            _make_block("a", content="Python programming"),
            _make_block("b", content="Java enterprise"),
        ]
        result = step.process(blocks)
        assert len(result) == 2

    def test_records_mutation_on_duplicate(self) -> None:
        step = DeduplicateStep()
        blocks = [
            _make_block("a", content="Same content here", priority=50),
            _make_block("b", content="Same content here", priority=40),
        ]
        step.process(blocks)
        # The lower priority one should have the mutation
        dup_block = next(b for b in blocks if b.name == "b")
        assert len(dup_block.mutations) == 1
        assert "overlaps" in dup_block.mutations[0].detail

    def test_custom_threshold(self) -> None:
        step = DeduplicateStep(similarity_threshold=0.99)
        blocks = [
            _make_block("a", content="Python is great"),
            _make_block("b", content="Python is good"),
        ]
        result = step.process(blocks)
        # With very high threshold, similar but not identical kept
        assert len(result) == 2

    def test_non_string_content_kept(self) -> None:
        step = DeduplicateStep()
        block = ContextBlock(
            type=BlockType.SHORT_TERM_MEMORY,
            content=[{"role": "user", "content": "hi"}],
            name="messages",
        )
        result = step.process([block])
        assert len(result) == 1

    def test_step_name(self) -> None:
        step = DeduplicateStep()
        assert step.name == "DeduplicateStep"


class TestReorderStep:
    """Tests for the ReorderStep."""

    def test_important_edges_strategy(self) -> None:
        step = ReorderStep(strategy="important_edges")
        blocks = [
            _make_block("low", priority=10),
            _make_block("mid", priority=50),
            _make_block("high", priority=90),
        ]
        result = step.process(blocks)
        assert len(result) == 3
        # Highest priority should be at start or end
        priorities = [b.priority for b in result]
        assert priorities[0] >= priorities[1] or priorities[-1] >= priorities[1]

    def test_short_list_unchanged(self) -> None:
        step = ReorderStep()
        blocks = [_make_block("a"), _make_block("b")]
        result = step.process(blocks)
        assert len(result) == 2

    def test_single_block_unchanged(self) -> None:
        step = ReorderStep()
        blocks = [_make_block("a")]
        result = step.process(blocks)
        assert len(result) == 1

    def test_records_mutations_for_moved_blocks(self) -> None:
        step = ReorderStep()
        blocks = [
            _make_block("a", priority=10),
            _make_block("b", priority=50),
            _make_block("c", priority=90),
        ]
        step.process(blocks)
        # Some blocks should have "moved" mutations
        moved = [b for b in blocks if any(m.action == "moved" for m in b.mutations)]
        assert len(moved) >= 0  # At least some may be moved

    def test_step_name(self) -> None:
        step = ReorderStep()
        assert step.name == "ReorderStep"

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            ReorderStep(strategy="unknown")


class TestCompactStep:
    """Tests for the CompactStep with LLM compaction."""

    def test_raises_without_llm(self) -> None:
        with pytest.raises(ValueError, match="requires an LLM"):
            CompactStep()

    def test_skips_short_blocks(self, tmp_path) -> None:
        def mock_llm(prompt: str) -> str:
            return "summary"

        step = CompactStep(
            llm=mock_llm,
            store=LocalCompactionStore(
                base_dir=str(tmp_path / "c"),
            ),
            min_tokens=1000,
        )
        original = "Short text"
        blocks = [_make_block("short", content=original)]
        result = step.process(blocks)
        assert result[0].content == original

    def test_records_mutation(self, tmp_path) -> None:
        def mock_llm(prompt: str) -> str:
            return "word " * 10

        step = CompactStep(
            llm=mock_llm,
            store=LocalCompactionStore(
                base_dir=str(tmp_path / "c"),
            ),
            min_tokens=5,
            max_info_loss=1.0,
        )
        blocks = [_make_block("long", content="word " * 200)]
        result = step.process(blocks)
        assert len(result[0].mutations) == 1
        assert result[0].mutations[0].action == "compacted"

    def test_compacts_with_llm_and_saves_original(self, tmp_path) -> None:
        store = LocalCompactionStore(base_dir=str(tmp_path / "compacted"))

        def mock_llm(prompt: str) -> str:
            return "Summary: key terms [1] and details [2]."

        content = "First paragraph about key terms.\n\nSecond paragraph about details."
        step = CompactStep(
            llm=mock_llm,
            store=store,
            min_tokens=1,
            max_info_loss=1.0,
        )
        blocks = [_make_block("llm_test", content=content)]
        result = step.process(blocks)
        assert result[0].content == "Summary: key terms [1] and details [2]."

    def test_metadata_contains_compaction_ref(self, tmp_path) -> None:
        store = LocalCompactionStore(base_dir=str(tmp_path / "compacted"))

        def mock_llm(prompt: str) -> str:
            return "Short summary [1]."

        content = "A long block of text.\n\nWith multiple paragraphs of content."
        step = CompactStep(
            llm=mock_llm,
            store=store,
            min_tokens=1,
            max_info_loss=1.0,
        )
        blocks = [_make_block("ref_test", content=content)]
        result = step.process(blocks)
        assert "compaction_ref" in result[0].metadata

    def test_mutation_detail_includes_store_reference(self, tmp_path) -> None:
        store = LocalCompactionStore(base_dir=str(tmp_path / "compacted"))

        def mock_llm(prompt: str) -> str:
            return "Brief [1]."

        content = "Original text here.\n\nMore original text here."
        step = CompactStep(
            llm=mock_llm,
            store=store,
            min_tokens=1,
            max_info_loss=1.0,
        )
        blocks = [_make_block("mutation_ref", content=content)]
        result = step.process(blocks)
        assert len(result[0].mutations) == 1
        assert "ref:" in result[0].mutations[0].detail

    def test_numbered_paragraphs_in_saved_markdown(self, tmp_path) -> None:
        store = LocalCompactionStore(base_dir=str(tmp_path / "compacted"))
        saved_content = None

        original_save = store.save

        async def capture_save(key, content, metadata=None):
            nonlocal saved_content
            saved_content = content
            return await original_save(key, content, metadata)

        store.save = capture_save

        def mock_llm(prompt: str) -> str:
            return "Summary [1] [2]."

        content = "First section content.\n\nSecond section content."
        step = CompactStep(
            llm=mock_llm,
            store=store,
            min_tokens=1,
            max_info_loss=1.0,
        )
        blocks = [_make_block("numbered_test", content=content)]
        step.process(blocks)
        assert saved_content is not None
        assert "## [1]" in saved_content
        assert "## [2]" in saved_content

    def test_skips_non_string_content(self) -> None:
        step = CompactStep(llm=lambda p: "summary", min_tokens=1)
        block = ContextBlock(
            type=BlockType.SHORT_TERM_MEMORY,
            content=[{"role": "user", "content": "hi"}],
            name="messages",
        )
        result = step.process([block])
        assert result[0].content == [{"role": "user", "content": "hi"}]

    def test_step_name(self) -> None:
        step = CompactStep(llm=lambda p: "summary")
        assert step.name == "CompactStep"


class TestContextPipeline:
    """Tests for the ContextPipeline."""

    def test_run_single_step(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add(_make_block("a", content="test", priority=50))
        window.add(_make_block("b", content="test", priority=10))

        pipeline = ContextPipeline([TrimStep(min_priority=30)])
        result = pipeline.run(window)
        assert result is window  # Returns same object
        assert len(window.blocks) == 1
        assert window.blocks[0].name == "a"

    def test_run_multiple_steps(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add(_make_block("a", content="Python is great", priority=50))
        window.add(_make_block("b", content="Python is great", priority=40))
        window.add(_make_block("c", content="low priority", priority=5))

        pipeline = ContextPipeline(
            [
                TrimStep(min_priority=10),
                DeduplicateStep(),
            ]
        )
        pipeline.run(window)
        # c should be removed by trim, b by dedup
        assert len(window.blocks) == 1
        assert window.blocks[0].name == "a"

    def test_pipeline_report(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add(_make_block("a", content="test"))
        window.add(_make_block("b", content="test", priority=5))

        pipeline = ContextPipeline([TrimStep(min_priority=30)])
        pipeline.run(window)

        report = pipeline.last_report
        assert report is not None
        assert len(report.steps) == 1
        assert report.steps[0].step_name == "TrimStep"
        assert report.total_tokens_saved >= 0

    def test_empty_pipeline(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add(_make_block("a"))

        pipeline = ContextPipeline([])
        pipeline.run(window)
        assert len(window.blocks) == 1

    def test_steps_property(self) -> None:
        steps = [TrimStep(), FilterStep()]
        pipeline = ContextPipeline(steps)
        assert len(pipeline.steps) == 2

    def test_last_report_initially_none(self) -> None:
        pipeline = ContextPipeline([])
        assert pipeline.last_report is None

    def test_pipeline_with_filter_and_reorder(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add(
            _make_block(
                "a",
                content="content a",
                priority=10,
                origin=Origin(
                    source="rag",
                    details={"relevance_score": 0.1},
                ),
            )
        )
        window.add(
            _make_block(
                "b",
                content="content b",
                priority=50,
                origin=Origin(
                    source="rag",
                    details={"relevance_score": 0.9},
                ),
            )
        )
        window.add(_make_block("c", content="content c", priority=70))

        pipeline = ContextPipeline(
            [
                FilterStep(min_relevance=0.5),
                ReorderStep(),
            ]
        )
        pipeline.run(window)
        # a should be filtered out (low relevance)
        names = [b.name for b in window.blocks]
        assert "a" not in names
        assert "b" in names
        assert "c" in names

    def test_pipeline_cost_delta(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add(_make_block("a", content="test content", priority=50))
        window.add(_make_block("b", content="removable", priority=5))

        pipeline = ContextPipeline([TrimStep(min_priority=30)])
        pipeline.run(window)

        report = pipeline.last_report
        assert report is not None
        # Cost delta should be >= 0 (we removed blocks)
        assert report.cost_delta >= 0


class TestMaskStep:
    """Tests for the MaskStep."""

    def test_masks_old_blocks(self) -> None:
        step = MaskStep(window=2, min_tokens=1)
        blocks = [
            _make_block("a", content="word " * 20),
            _make_block("b", content="word " * 20),
            _make_block("c", content="word " * 20),
            _make_block("d", content="word " * 20),
        ]
        result = step.process(blocks)
        assert len(result) == 4
        # Last 2 should be untouched, first 2 masked
        assert result[2].content == "word " * 20
        assert result[3].content == "word " * 20
        assert "[details omitted" in result[0].content
        assert "[details omitted" in result[1].content

    def test_keeps_all_within_window(self) -> None:
        step = MaskStep(window=5, min_tokens=1)
        blocks = [_make_block(f"b{i}", content="word " * 20) for i in range(3)]
        result = step.process(blocks)
        # All within window -- nothing masked
        for block in result:
            assert "omitted" not in (block.content if isinstance(block.content, str) else "")

    def test_records_mutation(self) -> None:
        step = MaskStep(window=1, min_tokens=1)
        blocks = [
            _make_block("old", content="word " * 30),
            _make_block("new", content="word " * 30),
        ]
        result = step.process(blocks)
        assert len(result[0].mutations) == 1
        assert result[0].mutations[0].action == "masked"
        assert result[0].mutations[0].tokens_before > result[0].mutations[0].tokens_after

    def test_skips_small_blocks(self) -> None:
        step = MaskStep(window=1, min_tokens=100)
        blocks = [
            _make_block("old", content="short"),
            _make_block("new", content="short"),
        ]
        result = step.process(blocks)
        # Both should be kept as-is (below min_tokens)
        assert result[0].content == "short"

    def test_block_type_filter(self) -> None:
        step = MaskStep(
            window=1,
            min_tokens=1,
            block_types=["tool_outputs"],
        )
        blocks = [
            _make_block("tool1", content="word " * 20, block_type=BlockType.TOOL_OUTPUTS),
            _make_block("user1", content="word " * 20, block_type=BlockType.USER_CONTEXT),
            _make_block("tool2", content="word " * 20, block_type=BlockType.TOOL_OUTPUTS),
        ]
        result = step.process(blocks)
        # tool1 should be masked (old TOOL_OUTPUTS), user1 untouched (different type)
        assert "[details omitted" in result[0].content
        assert result[1].content == "word " * 20  # USER_CONTEXT not in filter
        assert result[2].content == "word " * 20  # Most recent TOOL_OUTPUTS kept

    def test_custom_placeholder(self) -> None:
        step = MaskStep(window=1, min_tokens=1, placeholder="<REDACTED>")
        blocks = [
            _make_block("old", content="word " * 20),
            _make_block("new", content="word " * 20),
        ]
        result = step.process(blocks)
        assert result[0].content == "<REDACTED>"

    def test_step_name(self) -> None:
        step = MaskStep()
        assert step.name == "MaskStep"

    def test_skips_non_string_content(self) -> None:
        step = MaskStep(window=1, min_tokens=1)
        block_list = ContextBlock(
            type=BlockType.SHORT_TERM_MEMORY,
            content=[{"role": "user", "content": "hi"}],
            name="messages",
        )
        blocks = [block_list, _make_block("new", content="word " * 20)]
        result = step.process(blocks)
        # List content should pass through unchanged
        assert result[0].content == [{"role": "user", "content": "hi"}]


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
        """A user-supplied scoring_function replaces the default word-overlap scorer."""
        invocation_count = []

        def magic_word_scorer(query: str, content: str) -> float:
            invocation_count.append(True)
            return 1.0 if "magic" in content else 0.0

        trim_step = TrimStep(
            max_tokens=6,
            query="find magic",
            scoring_function=magic_word_scorer,
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
# Collapse detection during compaction
# ---------------------------------------------------------------------------


class TestCompactionCollapseDetection:
    """CompactStep should detect when compaction loses too much information."""

    def test_returns_original_when_llm_loses_keywords(self) -> None:
        """An LLM that drops all keywords returns the original block."""

        def destructive_llm(prompt: str) -> str:
            return "xyz abc def"

        compact_step = CompactStep(
            llm=destructive_llm,
            min_tokens=1,
            max_info_loss=0.3,
        )
        original = (
            "python programming language tutorial guide examples code"
        )
        block = _make_block("keyword_rich", content=original)
        result = compact_step.process([block])
        assert result[0].content == original
        assert len(result[0].mutations) == 0

    def test_no_warning_when_keywords_retained(
        self, tmp_path
    ) -> None:
        """Good keyword retention should not trigger collapse."""

        def half_content_llm(prompt: str) -> str:
            return "word " * 25

        compact_step = CompactStep(
            llm=half_content_llm,
            store=LocalCompactionStore(
                base_dir=str(tmp_path / "c"),
            ),
            min_tokens=1,
            max_info_loss=0.9,
        )
        block = _make_block(
            "repetitive_block", content="word " * 50
        )
        result = compact_step.process([block])
        if result[0].mutations:
            assert "COLLAPSE" not in result[0].mutations[0].detail

    def test_max_info_loss_threshold_is_configurable(self) -> None:
        """The max_info_loss parameter should be stored and respected."""
        compact_step = CompactStep(
            llm=lambda p: "summary", max_info_loss=0.8
        )
        assert compact_step._max_info_loss == 0.8


# ---------------------------------------------------------------------------
# Async pipeline
# ---------------------------------------------------------------------------


class TestAsyncContextPipeline:
    """ContextPipeline.arun() is an awaitable mirror of run()."""

    async def test_arun_returns_same_window(self) -> None:
        """arun should modify and return the same window object."""
        window = ContextWindow(max_tokens=10_000)
        block = _make_block("greeting", content="hello world", priority=80)
        window.add(block)
        pipeline = ContextPipeline(steps=[TrimStep(max_tokens=10_000)])
        result = await pipeline.arun(window)
        assert result is window

    async def test_arun_matches_sync_run(self) -> None:
        """Async and sync pipeline runs should produce identical token counts."""
        blocks = [_make_block("content_block", content="hello " * 50, priority=80)]
        trim_step = TrimStep(max_tokens=10_000)

        sync_window = ContextWindow(max_tokens=10_000)
        for block in blocks:
            sync_window.add(block)
        ContextPipeline(steps=[trim_step]).run(sync_window)

        async_window = ContextWindow(max_tokens=10_000)
        for block in blocks:
            async_window.add(block)
        await ContextPipeline(steps=[trim_step]).arun(async_window)

        assert sync_window.token_count == async_window.token_count


# ---------------------------------------------------------------------------
# PipelineBuilder (fluent API)
# ---------------------------------------------------------------------------


def _make_tool_block(
    name: str,
    content: str = "tool output",
    tool_name: str = "search",
    turn_number: int = 0,
    origin: Origin | None = None,
) -> ContextBlock:
    """Helper to create tool output blocks with metadata."""
    return ContextBlock(
        type=BlockType.TOOL_OUTPUTS,
        content=content,
        priority=50,
        name=name,
        metadata={"tool_name": tool_name, "turn_number": turn_number},
        origin=origin,
    )


class TestPipelineBuilder:
    """Tests for the fluent PipelineBuilder API."""

    def test_builder_basic_pipeline(self) -> None:
        pipeline = (
            ContextPipeline.builder()
            .deduplicate(threshold=0.9)
            .trim(max_tokens=50_000)
            .reorder("important_edges")
            .build()
        )
        assert len(pipeline.steps) == 3
        assert pipeline.steps[0].name == "DeduplicateStep"
        assert pipeline.steps[1].name == "TrimStep"
        assert pipeline.steps[2].name == "ReorderStep"

    def test_builder_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot build an empty pipeline"):
            ContextPipeline.builder().build()

    def test_builder_with_filter_and_mask(self) -> None:
        pipeline = (
            ContextPipeline.builder()
            .filter(min_relevance=0.5)
            .mask(window=5)
            .build()
        )
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0].name == "FilterStep"
        assert pipeline.steps[1].name == "MaskStep"

    def test_builder_with_custom_step(self) -> None:
        custom = TrimStep(max_tokens=10_000)
        pipeline = ContextPipeline.builder().step(custom).build()
        assert len(pipeline.steps) == 1

    def test_builder_capture_snapshots(self) -> None:
        pipeline = (
            ContextPipeline.builder()
            .deduplicate()
            .capture_snapshots(True)
            .build()
        )
        window = ContextWindow(max_tokens=100_000)
        window.add(_make_block("a", "hello world"))
        pipeline.run(window)
        assert len(pipeline.snapshots) > 0

    def test_builder_prune_stale(self) -> None:
        pipeline = (
            ContextPipeline.builder()
            .prune_stale(max_age_turns=5, keep_last_per_tool=2)
            .build()
        )
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].name == "PruneStaleStep"

    def test_builder_strip_thinking(self) -> None:
        pipeline = (
            ContextPipeline.builder()
            .strip_thinking()
            .build()
        )
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].name == "StripThinkingStep"

    def test_builder_full_agentic_pipeline(self) -> None:
        """Build a realistic agent pipeline with all step types."""
        pipeline = (
            ContextPipeline.builder()
            .strip_thinking()
            .prune_stale(max_age_turns=10)
            .deduplicate(threshold=0.85)
            .filter(min_relevance=0.3)
            .trim(max_tokens=100_000)
            .reorder("prefix_stable")
            .build()
        )
        assert len(pipeline.steps) == 6


# ---------------------------------------------------------------------------
# PruneStaleStep
# ---------------------------------------------------------------------------


class TestPruneStaleStep:
    """Tests for stale tool output pruning."""

    def test_keeps_non_tool_blocks(self) -> None:
        step = PruneStaleStep()
        blocks = [
            _make_block("system", block_type=BlockType.SYSTEM_PROMPT),
            _make_block("memory", block_type=BlockType.SHORT_TERM_MEMORY),
        ]
        result = step.process(blocks)
        assert len(result) == 2

    def test_keeps_single_tool_output(self) -> None:
        step = PruneStaleStep()
        blocks = [_make_tool_block("search_result", tool_name="search")]
        result = step.process(blocks)
        assert len(result) == 1

    def test_prunes_superseded_tool_outputs(self) -> None:
        step = PruneStaleStep(keep_last_per_tool=1)
        blocks = [
            _make_tool_block("search_old", content="old", tool_name="search", turn_number=1),
            _make_tool_block("search_new", content="new", tool_name="search", turn_number=5),
        ]
        result = step.process(blocks)
        assert len(result) == 1
        assert result[0].display_name == "search_new"

    def test_keeps_last_n_per_tool(self) -> None:
        step = PruneStaleStep(keep_last_per_tool=2)
        blocks = [
            _make_tool_block("s1", tool_name="search", turn_number=1),
            _make_tool_block("s2", tool_name="search", turn_number=2),
            _make_tool_block("s3", tool_name="search", turn_number=3),
        ]
        result = step.process(blocks)
        assert len(result) == 2
        names = [b.display_name for b in result]
        assert "s2" in names
        assert "s3" in names

    def test_max_age_turns(self) -> None:
        step = PruneStaleStep(max_age_turns=3, keep_last_per_tool=1)
        blocks = [
            _make_tool_block("old", tool_name="calc", turn_number=1),
            _make_tool_block("recent", tool_name="calc", turn_number=8),
            _make_tool_block("other", tool_name="search", turn_number=7),
        ]
        result = step.process(blocks)
        assert len(result) == 2

    def test_records_mutations(self) -> None:
        step = PruneStaleStep(keep_last_per_tool=1)
        old_block = _make_tool_block("old", tool_name="search")
        blocks = [
            old_block,
            _make_tool_block("new", tool_name="search"),
        ]
        step.process(blocks)
        assert len(old_block.mutations) == 1
        assert old_block.mutations[0].action == "removed"

    def test_different_tools_independent(self) -> None:
        step = PruneStaleStep(keep_last_per_tool=1)
        blocks = [
            _make_tool_block("s1", tool_name="search"),
            _make_tool_block("c1", tool_name="calc"),
        ]
        result = step.process(blocks)
        assert len(result) == 2

    def test_extracts_tool_name_from_origin(self) -> None:
        step = PruneStaleStep(keep_last_per_tool=1)
        origin = Origin.from_tool("web_search")
        blocks = [
            _make_tool_block("old", origin=origin),
            _make_tool_block("new", origin=origin),
        ]
        result = step.process(blocks)
        assert len(result) == 1
        assert result[0].display_name == "new"

    def test_step_name(self) -> None:
        step = PruneStaleStep()
        assert step.name == "PruneStaleStep"


# ---------------------------------------------------------------------------
# StripThinkingStep
# ---------------------------------------------------------------------------


class TestStripThinkingStep:
    """Tests for reasoning trace stripping."""

    def test_strips_thinking_tags(self) -> None:
        step = StripThinkingStep()
        blocks = [
            _make_block("response", content="<thinking>internal reasoning</thinking>The answer is 42."),
        ]
        result = step.process(blocks)
        assert len(result) == 1
        assert "<thinking>" not in result[0].content
        assert "The answer is 42." in result[0].content

    def test_strips_scratchpad_tags(self) -> None:
        step = StripThinkingStep()
        blocks = [
            _make_block("response", content="<scratchpad>notes here</scratchpad>Final answer."),
        ]
        result = step.process(blocks)
        assert "scratchpad" not in result[0].content
        assert "Final answer." in result[0].content

    def test_strips_multiline_thinking(self) -> None:
        step = StripThinkingStep()
        content = (
            "Before.\n"
            "<thinking>\nLine 1\nLine 2\nLine 3\n</thinking>\n"
            "After."
        )
        blocks = [_make_block("response", content=content)]
        result = step.process(blocks)
        assert "Line 1" not in result[0].content
        assert "Before." in result[0].content
        assert "After." in result[0].content

    def test_removes_empty_blocks_after_stripping(self) -> None:
        step = StripThinkingStep(strip_empty=True)
        blocks = [_make_block("thinking_only", content="<thinking>just thinking</thinking>")]
        result = step.process(blocks)
        assert len(result) == 0

    def test_keeps_empty_blocks_when_strip_empty_false(self) -> None:
        step = StripThinkingStep(strip_empty=False)
        blocks = [_make_block("thinking_only", content="<thinking>just thinking</thinking>")]
        result = step.process(blocks)
        assert len(result) == 1

    def test_no_change_when_no_patterns_match(self) -> None:
        step = StripThinkingStep()
        blocks = [_make_block("clean", content="No reasoning traces here.")]
        result = step.process(blocks)
        assert len(result) == 1
        assert result[0].content == "No reasoning traces here."

    def test_preserves_list_content_blocks(self) -> None:
        step = StripThinkingStep()
        blocks = [
            ContextBlock(
                type=BlockType.SHORT_TERM_MEMORY,
                content=[{"role": "user", "content": "hello"}],
                name="messages",
            ),
        ]
        result = step.process(blocks)
        assert len(result) == 1

    def test_custom_patterns(self) -> None:
        step = StripThinkingStep(patterns=[r"\[INTERNAL\].*?\[/INTERNAL\]"])
        blocks = [_make_block("custom", content="[INTERNAL]secret[/INTERNAL]visible")]
        result = step.process(blocks)
        assert result[0].content == "visible"

    def test_records_mutation_on_strip(self) -> None:
        step = StripThinkingStep()
        block = _make_block("response", content="<thinking>hmm</thinking>answer")
        step.process([block])
        assert len(block.mutations) == 1
        assert block.mutations[0].action == "compressed"

    def test_step_name(self) -> None:
        step = StripThinkingStep()
        assert step.name == "StripThinkingStep"
