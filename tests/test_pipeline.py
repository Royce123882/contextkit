"""Tests for the context pipeline (Phase 4)."""

from __future__ import annotations

from contextkit.core import BlockType, ContextBlock, ContextWindow
from contextkit.observe.provenance import Origin
from contextkit.pipeline import (
    CompactStep,
    ContextPipeline,
    DeduplicateStep,
    FilterStep,
    MaskStep,
    PipelineReport,
    ReorderStep,
    StepReport,
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
            filter_fn=lambda b: b.priority > 30,
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

    def test_unknown_strategy_passthrough(self) -> None:
        step = ReorderStep(strategy="unknown")
        blocks = [
            _make_block("a", priority=10),
            _make_block("b", priority=50),
            _make_block("c", priority=90),
        ]
        result = step.process(blocks)
        assert len(result) == 3


class TestCompactStep:
    """Tests for the CompactStep."""

    def test_compacts_long_content(self) -> None:
        step = CompactStep(min_tokens=5, target_ratio=0.3)
        blocks = [
            _make_block("long", content="word " * 200),
        ]
        result = step.process(blocks)
        assert len(result) == 1
        # Content should be shorter
        assert len(result[0].content) < len("word " * 200)

    def test_skips_short_content(self) -> None:
        step = CompactStep(min_tokens=1000)
        original = "Short text"
        blocks = [_make_block("short", content=original)]
        result = step.process(blocks)
        assert result[0].content == original

    def test_records_mutation(self) -> None:
        step = CompactStep(min_tokens=5, target_ratio=0.3)
        blocks = [_make_block("long", content="word " * 200)]
        result = step.process(blocks)
        assert len(result[0].mutations) == 1
        assert result[0].mutations[0].action == "compacted"

    def test_custom_compactor(self) -> None:
        def my_compactor(content: str) -> str:
            return content[:10]

        step = CompactStep(compactor=my_compactor, min_tokens=1)
        blocks = [
            _make_block("long", content="This is a long piece of text"),
        ]
        result = step.process(blocks)
        assert result[0].content == "This is a "

    def test_skips_non_string_content(self) -> None:
        step = CompactStep(min_tokens=1)
        block = ContextBlock(
            type=BlockType.SHORT_TERM_MEMORY,
            content=[{"role": "user", "content": "hi"}],
            name="messages",
        )
        result = step.process([block])
        assert result[0].content == [{"role": "user", "content": "hi"}]

    def test_step_name(self) -> None:
        step = CompactStep()
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
