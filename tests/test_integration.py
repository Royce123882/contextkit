"""End-to-end integration tests for contextkit.

Tests the full Phase 0 API as described in the PRD, verifying
that all documented import paths work and all features integrate
correctly.
"""

from __future__ import annotations

import json
import os
import tempfile

from contextkit import (
    AssemblyReport,
    BlockType,
    BudgetExceededError,
    ContextAssembler,
    ContextBlock,
    ContextWindow,
    ModelSpec,
    Mutation,
    Origin,
    __version__,
)
from contextkit.adapters import AnthropicAdapter, OpenAIAdapter
from contextkit.events import ContextEvent, clear_handlers, on
from contextkit.models import get_model, list_models, register_model
from contextkit.observe.events import BlockEventData, BudgetEventData
from contextkit.tokens import count, fits_budget


class TestImportPaths:
    """Verify all import paths from the PRD work."""

    def test_top_level_imports(self) -> None:
        assert ContextWindow is not None
        assert ContextBlock is not None
        assert BlockType is not None
        assert Origin is not None
        assert Mutation is not None
        assert ContextAssembler is not None
        assert AssemblyReport is not None
        assert BudgetExceededError is not None

    def test_tokens_imports(self) -> None:
        assert count is not None
        assert fits_budget is not None

    def test_adapters_imports(self) -> None:
        assert AnthropicAdapter is not None
        assert OpenAIAdapter is not None

    def test_models_imports(self) -> None:
        assert get_model is not None
        assert register_model is not None
        assert ModelSpec is not None
        assert list_models is not None

    def test_events_imports(self) -> None:
        assert on is not None
        assert ContextEvent is not None

    def test_version(self) -> None:
        assert __version__ == "0.1.0"


class TestFullWorkflow:
    """Tests the full Phase 0 workflow from the PRD."""

    def setup_method(self) -> None:
        clear_handlers()

    def test_prd_phase0_api_example(self) -> None:
        """Run the main API example from the PRD."""
        # Build context with model awareness
        window = ContextWindow(
            model="claude-sonnet-4-5-20250929",
            budget_warnings=[0.75, 0.90, 0.95],
        )

        system = ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="You are a helpful assistant with expertise in Python.",
            priority=100,
            name="system_prompt",
            origin=Origin(source="prompt", details={"template": "assistant_v1"}),
        )
        history = ContextBlock(
            type=BlockType.SHORT_TERM_MEMORY,
            content=[
                {"role": "user", "content": "How do I use context engineering?"},
                {"role": "assistant", "content": "Context engineering involves..."},
            ],
            priority=80,
            name="conversation_history",
            origin=Origin(source="conversation", details={"turn_range": "1-2"}),
        )

        window.add(system)
        window.add(history)

        # Inspect
        summary = window.inspect()
        assert "system_prompt" in summary
        assert "conversation_history" in summary

        # Drill down
        detail = window.inspect("conversation_history")
        assert "Messages:" in detail
        assert "user" in detail

        # Explain
        explanation = window.explain("system_prompt")
        assert "INCLUDED" in explanation

        # Cost
        assert window.cost_estimate >= 0
        assert window.budget_remaining > 0

        # Format for Anthropic
        adapter = AnthropicAdapter()
        payload = adapter.format(window)
        assert "system" in payload
        assert "messages" in payload
        assert payload["model"] == "claude-sonnet-4-5-20250929"

    def test_standalone_token_utilities(self) -> None:
        """Test standalone utilities from the PRD."""
        token_count = count("Hello world")
        assert token_count > 0

        assert fits_budget("Hello world", max_tokens=100) is True
        assert fits_budget("a" * 10000, max_tokens=10) is False

    def test_standalone_adapter_formatting(self) -> None:
        """Test adapter format_messages without ContextWindow."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        # Anthropic format
        anthropic_payload = AnthropicAdapter.format_messages(
            messages, system="Be helpful."
        )
        assert anthropic_payload["system"] == "Be helpful."
        assert anthropic_payload["messages"] == messages

        # OpenAI format
        openai_payload = OpenAIAdapter.format_messages(messages, system="Be helpful.")
        assert openai_payload["messages"][0]["role"] == "system"
        assert len(openai_payload["messages"]) == 3

    def test_event_hooks_workflow(self) -> None:
        """Test event hooks as described in the PRD."""
        events_log: list[str] = []

        @on(ContextEvent.BLOCK_ADDED)
        def log_addition(event: BlockEventData) -> None:
            events_log.append(f"Added {event.block_name}: {event.token_count} tokens")

        @on(ContextEvent.BUDGET_WARNING)
        def log_budget(event: BudgetEventData) -> None:
            events_log.append(f"Budget at {event.percent}%")

        window = ContextWindow(
            max_tokens=100,
            budget_warnings=[0.5],
        )
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="a" * 300,
                name="big_prompt",
            )
        )

        assert any("Added" in e for e in events_log)

    def test_assembler_with_explain_workflow(self) -> None:
        """Test assembler + explain integration."""
        window = ContextWindow(max_tokens=100)
        assembler = ContextAssembler(window)

        blocks = [
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="Short system prompt",
                priority=100,
                name="sys",
                origin=Origin(source="prompt"),
            ),
            ContextBlock(
                type=BlockType.RAG,
                content="a" * 2000,
                priority=30,
                name="rag_big",
                origin=Origin(
                    source="rag",
                    details={"retriever": "chroma", "relevance_score": 0.42},
                ),
            ),
        ]

        assembler.assemble(blocks)

        # Explain included
        sys_explanation = window.explain("sys")
        assert "INCLUDED" in sys_explanation
        assert "Added by: ContextAssembler" in sys_explanation

        # Explain excluded
        rag_explanation = window.explain("rag_big")
        assert "EXCLUDED" in rag_explanation
        assert "budget_exceeded" in rag_explanation

    def test_diff_workflow(self) -> None:
        """Test window diff comparison."""
        window_before = ContextWindow(max_tokens=100_000)
        window_before.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="Old prompt",
                name="sys",
            )
        )

        window_after = ContextWindow(max_tokens=100_000)
        window_after.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="New improved prompt with more detail",
                name="sys",
            )
        )
        window_after.add(
            ContextBlock(
                type=BlockType.RAG,
                content="New RAG context",
                name="rag",
            )
        )

        diff_output = window_before.diff(window_after)
        assert "Added:" in diff_output
        assert "rag" in diff_output
        assert "Changed:" in diff_output
        assert "sys" in diff_output

    def test_dump_workflow(self) -> None:
        """Test JSON dump and reload."""
        window = ContextWindow(model="gpt-4o")
        origin = Origin(
            source="rag",
            details={"query": "test query", "relevance_score": 0.85},
        )
        window.add(
            ContextBlock(
                type=BlockType.RAG,
                content="Retrieved content about auth...",
                priority=70,
                name="rag_auth",
                origin=origin,
            )
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            window.dump(path)

            with open(path) as f:
                data = json.load(f)

            assert data["model"] == "gpt-4o"
            assert data["max_tokens"] == 128_000
            assert len(data["blocks"]) == 1
            assert data["blocks"][0]["origin"]["source"] == "rag"
            assert data["blocks"][0]["origin"]["details"]["query"] == "test query"
            assert data["token_count"] > 0
            assert data["cost_estimate"] >= 0
        finally:
            os.unlink(path)

    def test_custom_model_registration(self) -> None:
        """Test registering and using a custom model."""
        register_model(
            "my-fine-tune",
            ModelSpec(
                max_context=32_000,
                encoding="cl100k_base",
                input_cost_per_mtok=0.50,
                output_cost_per_mtok=1.50,
            ),
        )

        window = ContextWindow(model="my-fine-tune")
        assert window.max_tokens == 32_000

        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="Hello",
            )
        )
        assert window.cost_estimate > 0

    def test_model_aware_context_limits(self) -> None:
        """Test that different models have different limits."""
        window_claude = ContextWindow(model="claude-sonnet-4-5-20250929")
        window_gpt = ContextWindow(model="gpt-4o")

        assert window_claude.max_tokens == 200_000
        assert window_gpt.max_tokens == 128_000
        assert window_claude.encoding != window_gpt.encoding

    def test_budget_exceeded_event_fires(self) -> None:
        """Test that BUDGET_EXCEEDED event fires."""
        events: list[str] = []

        @on(ContextEvent.BUDGET_EXCEEDED)
        def handler(event: BudgetEventData) -> None:
            events.append("exceeded")

        window = ContextWindow(max_tokens=5)
        try:
            window.add(
                ContextBlock(
                    type=BlockType.SYSTEM_PROMPT,
                    content="This content is way too long to fit",
                )
            )
        except BudgetExceededError:
            pass
        assert "exceeded" in events

    def test_all_block_types_with_origin(self) -> None:
        """Test that all block types work with Origin."""
        window = ContextWindow(max_tokens=100_000)

        block_configs = [
            (BlockType.SYSTEM_PROMPT, "prompt", "System"),
            (BlockType.SHORT_TERM_MEMORY, "conversation", "History"),
            (BlockType.LONG_TERM_MEMORY, "memory", "Memory"),
            (BlockType.FILES, "file", "File"),
            (BlockType.TOOL_DEFINITIONS, "tool", "Tools"),
            (BlockType.TOOL_OUTPUTS, "tool_output", "Output"),
            (BlockType.RAG, "rag", "RAG"),
            (BlockType.EXAMPLES, "example", "Example"),
            (BlockType.OUTPUT_SCHEMAS, "schema", "Schema"),
            (BlockType.SYSTEM_METADATA, "metadata", "Meta"),
            (BlockType.SCRATCHPAD, "scratchpad", "Notes"),
            (BlockType.USER_CONTEXT, "user", "User"),
        ]

        for block_type, source, name in block_configs:
            block = ContextBlock(
                type=block_type,
                content=f"{name} content",
                name=name.lower(),
                origin=Origin(source=source),
            )
            window.add(block)

        assert len(window) == 12

        # All should be inspectable
        summary = window.inspect()
        for _, _, name in block_configs:
            assert name.lower() in summary

    def test_assembly_complete_event(self) -> None:
        """Test ASSEMBLY_COMPLETE event fires."""
        events: list[dict[str, object]] = []

        @on(ContextEvent.ASSEMBLY_COMPLETE)
        def handler(event: object) -> None:
            events.append({"event": "complete"})

        window = ContextWindow(max_tokens=100_000)
        assembler = ContextAssembler(window)
        assembler.assemble(
            [
                ContextBlock(
                    type=BlockType.SYSTEM_PROMPT,
                    content="test",
                )
            ]
        )
        assert len(events) == 1

    def test_window_rendered_event(self) -> None:
        """Test WINDOW_RENDERED event fires on adapter format."""
        events: list[str] = []

        @on(ContextEvent.WINDOW_RENDERED)
        def handler(event: object) -> None:
            events.append("rendered")

        window = ContextWindow(model="gpt-4o")
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="test",
            )
        )

        adapter = OpenAIAdapter()
        adapter.format(window)
        assert len(events) == 1
