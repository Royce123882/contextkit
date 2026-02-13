"""Tests for new features: PipelineBuilder, fluent ContextWindow API,
PruneStaleStep, StripThinkingStep, effective window profiles,
model-adaptive QualityScorer, and enhanced BudgetExceededError.
"""

from __future__ import annotations

import pytest

from contextkit.core import BlockType, BudgetExceededError, ContextBlock, ContextWindow
from contextkit.models import AttentionProfile, ModelSpec, get_model, register_model
from contextkit.observe.linter import ContextLinter
from contextkit.observe.quality import QualityScorer
from contextkit.pipeline import (
    ContextPipeline,
    PipelineBuilder,
    PruneStaleStep,
    StripThinkingStep,
)
from contextkit.observe.provenance import Origin


def _make_block(
    name: str,
    content: str = "test content here",
    priority: int = 50,
    block_type: BlockType = BlockType.USER_CONTEXT,
    metadata: dict | None = None,
    origin: Origin | None = None,
) -> ContextBlock:
    """Helper to create test blocks."""
    return ContextBlock(
        type=block_type,
        content=content,
        priority=priority,
        name=name,
        metadata=metadata or {},
        origin=origin,
    )


# ===================================================================
# PipelineBuilder
# ===================================================================

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
        from contextkit.pipeline import TrimStep

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
        # Run the pipeline to verify snapshots work
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
        """Test a realistic agent pipeline with all new steps."""
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


# ===================================================================
# Fluent ContextWindow API
# ===================================================================

class TestFluentContextWindow:
    """Tests for the with_* fluent methods on ContextWindow."""

    def test_with_system(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        result = window.with_system("You are a helpful assistant.")
        assert result is window
        assert len(window) == 1
        assert window.blocks[0].type == BlockType.SYSTEM_PROMPT

    def test_chaining_multiple_with_methods(self) -> None:
        window = (
            ContextWindow(max_tokens=100_000)
            .with_system("Be helpful.")
            .with_rag("Retrieved document content.")
            .with_memory("User said hello previously.")
        )
        assert len(window) == 3
        types = [b.type for b in window.blocks]
        assert BlockType.SYSTEM_PROMPT in types
        assert BlockType.RAG in types
        assert BlockType.SHORT_TERM_MEMORY in types

    def test_with_tools(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        window.with_tools("tool definitions JSON")
        assert window.blocks[0].type == BlockType.TOOL_DEFINITIONS

    def test_with_examples(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        window.with_examples("Q: What is 2+2? A: 4")
        assert window.blocks[0].type == BlockType.EXAMPLES

    def test_with_file(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        window.with_file("file content", file_path="/tmp/test.py", name="test_file")
        block = window.blocks[0]
        assert block.type == BlockType.FILES
        assert block.display_name == "test_file"

    def test_with_custom_priority(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        window.with_rag("doc content", priority=90, name="important_doc")
        assert window.blocks[0].priority == 90
        assert window.blocks[0].display_name == "important_doc"


# ===================================================================
# PruneStaleStep
# ===================================================================

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
        blocks = [
            _make_block(
                "search_result",
                content="result 1",
                block_type=BlockType.TOOL_OUTPUTS,
                metadata={"tool_name": "search"},
            ),
        ]
        result = step.process(blocks)
        assert len(result) == 1

    def test_prunes_superseded_tool_outputs(self) -> None:
        step = PruneStaleStep(keep_last_per_tool=1)
        blocks = [
            _make_block(
                "search_old",
                content="old result",
                block_type=BlockType.TOOL_OUTPUTS,
                metadata={"tool_name": "search", "turn_number": 1},
            ),
            _make_block(
                "search_new",
                content="new result",
                block_type=BlockType.TOOL_OUTPUTS,
                metadata={"tool_name": "search", "turn_number": 5},
            ),
        ]
        result = step.process(blocks)
        assert len(result) == 1
        assert result[0].display_name == "search_new"

    def test_keeps_last_n_per_tool(self) -> None:
        step = PruneStaleStep(keep_last_per_tool=2)
        blocks = [
            _make_block("s1", block_type=BlockType.TOOL_OUTPUTS,
                        metadata={"tool_name": "search", "turn_number": 1}),
            _make_block("s2", block_type=BlockType.TOOL_OUTPUTS,
                        metadata={"tool_name": "search", "turn_number": 2}),
            _make_block("s3", block_type=BlockType.TOOL_OUTPUTS,
                        metadata={"tool_name": "search", "turn_number": 3}),
        ]
        result = step.process(blocks)
        assert len(result) == 2
        names = [b.display_name for b in result]
        assert "s2" in names
        assert "s3" in names

    def test_max_age_turns(self) -> None:
        step = PruneStaleStep(max_age_turns=3, keep_last_per_tool=1)
        blocks = [
            _make_block("old", block_type=BlockType.TOOL_OUTPUTS,
                        metadata={"tool_name": "calc", "turn_number": 1}),
            _make_block("recent", block_type=BlockType.TOOL_OUTPUTS,
                        metadata={"tool_name": "calc", "turn_number": 8}),
            _make_block("other", block_type=BlockType.TOOL_OUTPUTS,
                        metadata={"tool_name": "search", "turn_number": 7}),
        ]
        result = step.process(blocks)
        # "old" is superseded by "recent" and exceeds max_age_turns
        # "recent" is kept (last per tool)
        # "other" is sole entry for "search", so kept
        assert len(result) == 2

    def test_records_mutations(self) -> None:
        step = PruneStaleStep(keep_last_per_tool=1)
        old_block = _make_block(
            "old", block_type=BlockType.TOOL_OUTPUTS,
            metadata={"tool_name": "search"},
        )
        blocks = [
            old_block,
            _make_block("new", block_type=BlockType.TOOL_OUTPUTS,
                        metadata={"tool_name": "search"}),
        ]
        step.process(blocks)
        assert len(old_block.mutations) == 1
        assert old_block.mutations[0].action == "removed"

    def test_different_tools_independent(self) -> None:
        step = PruneStaleStep(keep_last_per_tool=1)
        blocks = [
            _make_block("s1", block_type=BlockType.TOOL_OUTPUTS,
                        metadata={"tool_name": "search"}),
            _make_block("c1", block_type=BlockType.TOOL_OUTPUTS,
                        metadata={"tool_name": "calc"}),
        ]
        result = step.process(blocks)
        # Each is the only output for its tool -- both kept
        assert len(result) == 2

    def test_extracts_tool_name_from_origin(self) -> None:
        step = PruneStaleStep(keep_last_per_tool=1)
        origin = Origin.from_tool("web_search")
        blocks = [
            _make_block("old", block_type=BlockType.TOOL_OUTPUTS, origin=origin),
            _make_block("new", block_type=BlockType.TOOL_OUTPUTS, origin=origin),
        ]
        result = step.process(blocks)
        assert len(result) == 1
        assert result[0].display_name == "new"


# ===================================================================
# StripThinkingStep
# ===================================================================

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


# ===================================================================
# Effective Window Profiles
# ===================================================================

class TestEffectiveWindowProfiles:
    """Tests for effective_max_tokens in ModelSpec and lint warning."""

    def test_model_has_effective_max_tokens(self) -> None:
        spec = get_model("claude-opus-4-6")
        assert spec.effective_max_tokens is not None
        assert spec.effective_max_tokens < spec.max_context

    def test_model_has_attention_profile(self) -> None:
        spec = get_model("claude-opus-4-6")
        assert spec.attention_profile is not None
        assert spec.attention_profile.curve_depth == 0.3

    def test_gpt4o_has_standard_profile(self) -> None:
        spec = get_model("gpt-4o")
        assert spec.attention_profile is not None
        assert spec.attention_profile.curve_depth == 0.6

    def test_custom_model_without_effective_tokens(self) -> None:
        register_model(
            "test-custom-model",
            ModelSpec(
                max_context=50_000,
                encoding="cl100k_base",
                input_cost_per_mtok=1.0,
                output_cost_per_mtok=5.0,
            ),
        )
        spec = get_model("test-custom-model")
        assert spec.effective_max_tokens is None
        assert spec.attention_profile is None

    def test_attention_profile_standalone(self) -> None:
        profile = AttentionProfile(curve_depth=0.4, label="custom")
        assert profile.curve_depth == 0.4
        assert profile.label == "custom"


# ===================================================================
# Effective Window Lint Check
# ===================================================================

class TestEffectiveWindowLint:
    """Tests for the exceeds_effective_window lint check."""

    def test_no_warning_below_effective_limit(self) -> None:
        window = ContextWindow(model="gpt-4o")
        window.add(_make_block("small", content="hello"))
        linter = ContextLinter()
        warnings = linter.lint(window)
        effective_warnings = [w for w in warnings if w.code == "exceeds_effective_window"]
        assert len(effective_warnings) == 0

    def test_no_warning_for_manual_window(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        linter = ContextLinter()
        warnings = linter.lint(window)
        effective_warnings = [w for w in warnings if w.code == "exceeds_effective_window"]
        assert len(effective_warnings) == 0


# ===================================================================
# Model-Adaptive QualityScorer
# ===================================================================

class TestModelAdaptiveQualityScorer:
    """Tests for model-adaptive attention curve depth."""

    def test_default_curve_depth(self) -> None:
        scorer = QualityScorer()
        # Default curve_depth is 0.6
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

        # Shallow curve means less penalty in the middle, so higher overall score
        assert shallow_report.overall_score >= standard_report.overall_score

    def test_for_model_factory(self) -> None:
        scorer = QualityScorer.for_model("claude-opus-4-6")
        # Claude 4.x has strong_long_context profile (curve_depth=0.3)
        assert scorer._curve_depth == 0.3

    def test_for_model_unknown_model(self) -> None:
        scorer = QualityScorer.for_model("nonexistent-model")
        # Falls back to default
        assert scorer._curve_depth == 0.6

    def test_for_model_standard_profile(self) -> None:
        scorer = QualityScorer.for_model("gpt-4o")
        assert scorer._curve_depth == 0.6


# ===================================================================
# Enhanced BudgetExceededError
# ===================================================================

class TestEnhancedBudgetExceededError:
    """Tests for context-aware budget error messages."""

    def test_error_includes_tokens_needed(self) -> None:
        error = BudgetExceededError(
            block_name="big_block",
            block_tokens=5000,
            budget_remaining=1000,
            max_tokens=100_000,
        )
        assert "4,000 more" in str(error)

    def test_error_includes_removable_blocks(self) -> None:
        error = BudgetExceededError(
            block_name="big_block",
            block_tokens=5000,
            budget_remaining=1000,
            max_tokens=100_000,
            removable_blocks=[("low_priority_doc", 3000), ("old_memory", 2000)],
        )
        message = str(error)
        assert "low_priority_doc" in message
        assert "5,000 tokens available" in message

    def test_error_without_removable_blocks(self) -> None:
        error = BudgetExceededError(
            block_name="block",
            block_tokens=100,
            budget_remaining=50,
            max_tokens=1000,
        )
        message = str(error)
        assert "Suggestions:" in message
        assert "Remove low-priority" not in message

    def test_window_raises_with_removable_info(self) -> None:
        # Filler block is ~28 tokens. Budget is 40 so filler fits, but
        # overflow (~28 tokens) won't fit in remaining 12.
        window = ContextWindow(max_tokens=40)
        window.add(_make_block("filler", content="alpha beta gamma delta " * 5, priority=30))
        with pytest.raises(BudgetExceededError) as exc_info:
            window.add(_make_block("overflow", content="echo foxtrot golf hotel " * 5))
        # The filler block has priority 30 (<= 50 median), so it should appear
        assert len(exc_info.value.removable_blocks) > 0
        assert exc_info.value.removable_blocks[0][0] == "filler"
