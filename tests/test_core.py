"""Tests for core data structures (BlockType, ContextBlock, ContextWindow)."""

from __future__ import annotations

import pytest

from contextkit.core import (
    BlockType,
    BudgetExceededError,
    ContextBlock,
    ContextWindow,
)
from contextkit.events import ContextEvent, clear_handlers
from contextkit.observe.provenance import Mutation, Origin


class TestBlockType:
    """Tests for the BlockType enum."""

    def test_all_twelve_types_exist(self) -> None:
        expected = [
            "SYSTEM_PROMPT",
            "SHORT_TERM_MEMORY",
            "LONG_TERM_MEMORY",
            "FILES",
            "TOOL_DEFINITIONS",
            "TOOL_OUTPUTS",
            "RAG",
            "EXAMPLES",
            "OUTPUT_SCHEMAS",
            "SYSTEM_METADATA",
            "SCRATCHPAD",
            "USER_CONTEXT",
        ]
        actual = [bt.name for bt in BlockType]
        assert sorted(actual) == sorted(expected)

    def test_block_type_values(self) -> None:
        assert BlockType.SYSTEM_PROMPT.value == "system_prompt"
        assert BlockType.RAG.value == "rag"
        assert BlockType.USER_CONTEXT.value == "user_context"

    def test_block_type_count(self) -> None:
        assert len(BlockType) == 12


class TestContextBlock:
    """Tests for the ContextBlock model."""

    def test_create_with_minimal_fields(self) -> None:
        block = ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="You are helpful.",
        )
        assert block.type == BlockType.SYSTEM_PROMPT
        assert block.content == "You are helpful."
        assert block.priority == 50
        assert block.metadata == {}
        assert block.name is None
        assert block.origin is None
        assert block.mutations == []

    def test_create_with_all_fields(self) -> None:
        origin = Origin(source="prompt", details={"template": "v1"})
        mutation = Mutation(
            step="test",
            action="test",
            detail="test",
            tokens_before=10,
            tokens_after=5,
        )
        block = ContextBlock(
            type=BlockType.RAG,
            content="Auth uses JWT...",
            priority=70,
            metadata={"source": "docs"},
            name="rag_auth",
            origin=origin,
            mutations=[mutation],
        )
        assert block.priority == 70
        assert block.name == "rag_auth"
        assert block.origin is not None
        assert block.origin.source == "prompt"
        assert len(block.mutations) == 1

    def test_create_with_message_list_content(self) -> None:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        block = ContextBlock(
            type=BlockType.SHORT_TERM_MEMORY,
            content=messages,
            priority=80,
        )
        assert isinstance(block.content, list)
        assert len(block.content) == 2

    def test_token_count_property(self) -> None:
        block = ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="Hello world",
        )
        assert block.token_count > 0

    def test_token_count_for_message_list(self) -> None:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        block = ContextBlock(
            type=BlockType.SHORT_TERM_MEMORY,
            content=messages,
        )
        assert block.token_count > 0

    def test_display_name_uses_name_if_set(self) -> None:
        block = ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="test",
            name="my_prompt",
        )
        assert block.display_name == "my_prompt"

    def test_display_name_falls_back_to_type(self) -> None:
        block = ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="test",
        )
        assert block.display_name == "system_prompt"

    def test_repr_includes_key_info(self) -> None:
        block = ContextBlock(
            type=BlockType.RAG,
            content="test content",
            priority=70,
            name="rag_block",
            origin=Origin(source="rag"),
        )
        repr_str = repr(block)
        assert "rag" in repr_str
        assert "rag_block" in repr_str
        assert "70" in repr_str

    def test_serialization_preserves_provenance(self) -> None:
        origin = Origin(source="rag", details={"query": "test"})
        block = ContextBlock(
            type=BlockType.RAG,
            content="test",
            origin=origin,
        )
        data = block.model_dump()
        assert data["origin"]["source"] == "rag"
        assert data["origin"]["details"]["query"] == "test"

    def test_serialization_preserves_mutations(self) -> None:
        mutation = Mutation(
            step="TrimStep",
            action="removed",
            detail="test",
            tokens_before=100,
            tokens_after=0,
        )
        block = ContextBlock(
            type=BlockType.RAG,
            content="test",
            mutations=[mutation],
        )
        data = block.model_dump()
        assert len(data["mutations"]) == 1
        assert data["mutations"][0]["step"] == "TrimStep"


class TestContextWindow:
    """Tests for the ContextWindow container."""

    def setup_method(self) -> None:
        clear_handlers()

    def test_create_with_model_name(self) -> None:
        window = ContextWindow(model="claude-sonnet-4-5-20250929")
        assert window.model_name == "claude-sonnet-4-5-20250929"
        assert window.max_tokens == 200_000

    def test_create_with_manual_max_tokens(self) -> None:
        window = ContextWindow(max_tokens=10_000)
        assert window.model_name is None
        assert window.max_tokens == 10_000

    def test_create_without_model_or_max_tokens_raises(self) -> None:
        with pytest.raises(ValueError, match="Either 'model' or 'max_tokens'"):
            ContextWindow()

    def test_add_block(self) -> None:
        window = ContextWindow(max_tokens=10_000)
        block = ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="test",
            name="sys",
        )
        window.add(block)
        assert len(window) == 1
        assert window.blocks[0].display_name == "sys"

    def test_add_block_emits_event(self) -> None:
        window = ContextWindow(max_tokens=10_000)
        events_received: list[str] = []

        from contextkit.events import on

        @on(ContextEvent.BLOCK_ADDED)
        def handler(event: object) -> None:
            events_received.append("added")

        block = ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="test",
        )
        window.add(block)
        assert len(events_received) == 1

    def test_add_block_budget_exceeded(self) -> None:
        window = ContextWindow(max_tokens=5)
        block = ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="This is a longer string that exceeds the budget",
        )
        with pytest.raises(BudgetExceededError):
            window.add(block)

    def test_budget_exceeded_emits_event(self) -> None:
        window = ContextWindow(max_tokens=5)
        events_received: list[str] = []

        from contextkit.events import on

        @on(ContextEvent.BUDGET_EXCEEDED)
        def handler(event: object) -> None:
            events_received.append("exceeded")

        block = ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="This is a longer string that exceeds the budget",
        )
        with pytest.raises(BudgetExceededError):
            window.add(block)
        assert len(events_received) == 1

    def test_remove_block(self) -> None:
        window = ContextWindow(max_tokens=10_000)
        block = ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="test",
            name="sys",
        )
        window.add(block)
        assert len(window) == 1
        window.remove("sys")
        assert len(window) == 0

    def test_remove_block_emits_event(self) -> None:
        window = ContextWindow(max_tokens=10_000)
        events_received: list[str] = []

        from contextkit.events import on

        @on(ContextEvent.BLOCK_REMOVED)
        def handler(event: object) -> None:
            events_received.append("removed")

        block = ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="test",
            name="sys",
        )
        window.add(block)
        window.remove("sys")
        assert len(events_received) == 1

    def test_remove_nonexistent_block_raises(self) -> None:
        window = ContextWindow(max_tokens=10_000)
        with pytest.raises(KeyError, match="No block with name"):
            window.remove("nonexistent")

    def test_token_count_updates_on_add(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        block1 = ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="Hello",
            name="b1",
        )
        block2 = ContextBlock(
            type=BlockType.USER_CONTEXT,
            content="World",
            name="b2",
        )
        window.add(block1)
        first_count = window.token_count
        window.add(block2)
        assert window.token_count > first_count

    def test_token_count_updates_on_remove(self) -> None:
        window = ContextWindow(max_tokens=100_000)
        block = ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="Hello",
            name="sys",
        )
        window.add(block)
        count_with = window.token_count
        window.remove("sys")
        assert window.token_count < count_with

    def test_budget_remaining(self) -> None:
        window = ContextWindow(max_tokens=1000)
        block = ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="Hello",
        )
        window.add(block)
        assert window.budget_remaining == 1000 - window.token_count

    def test_cost_estimate_with_model(self) -> None:
        window = ContextWindow(model="claude-sonnet-4-5-20250929")
        block = ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="Hello world",
        )
        window.add(block)
        assert window.cost_estimate > 0

    def test_cost_estimate_without_model(self) -> None:
        window = ContextWindow(max_tokens=1000)
        block = ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="Hello",
        )
        window.add(block)
        assert window.cost_estimate == 0.0

    def test_cost_estimate_with_response(self) -> None:
        window = ContextWindow(model="claude-sonnet-4-5-20250929")
        block = ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="Hello world",
        )
        window.add(block)
        cost_input = window.cost_estimate
        cost_total = window.cost_estimate_with_response(500)
        assert cost_total > cost_input

    def test_render_returns_messages(self) -> None:
        window = ContextWindow(max_tokens=10_000)
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="System prompt",
                priority=100,
                name="sys",
            )
        )
        window.add(
            ContextBlock(
                type=BlockType.USER_CONTEXT,
                content="User info",
                priority=80,
                name="user",
            )
        )
        messages = window.render()
        assert len(messages) == 2
        # Higher priority first
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System prompt"

    def test_render_with_message_list_content(self) -> None:
        window = ContextWindow(max_tokens=10_000)
        window.add(
            ContextBlock(
                type=BlockType.SHORT_TERM_MEMORY,
                content=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ],
                priority=80,
            )
        )
        messages = window.render()
        assert len(messages) == 2

    def test_render_emits_event(self) -> None:
        window = ContextWindow(max_tokens=10_000)
        events_received: list[str] = []

        from contextkit.events import on

        @on(ContextEvent.WINDOW_RENDERED)
        def handler(event: object) -> None:
            events_received.append("rendered")

        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="test",
            )
        )
        window.render()
        assert len(events_received) == 1

    def test_to_dict_includes_provenance(self) -> None:
        window = ContextWindow(model="gpt-4o")
        origin = Origin(source="rag", details={"query": "test"})
        mutation = Mutation(
            step="TrimStep",
            action="removed",
            detail="test",
            tokens_before=100,
            tokens_after=0,
        )
        block = ContextBlock(
            type=BlockType.RAG,
            content="test",
            name="rag_block",
            origin=origin,
            mutations=[mutation],
        )
        window.add(block)
        data = window.to_dict()
        assert data["model"] == "gpt-4o"
        assert len(data["blocks"]) == 1
        assert data["blocks"][0]["origin"]["source"] == "rag"
        assert len(data["blocks"][0]["mutations"]) == 1

    def test_get_block_found(self) -> None:
        window = ContextWindow(max_tokens=10_000)
        block = ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="test",
            name="sys",
        )
        window.add(block)
        found = window.get_block("sys")
        assert found is not None
        assert found.display_name == "sys"

    def test_get_block_not_found(self) -> None:
        window = ContextWindow(max_tokens=10_000)
        assert window.get_block("missing") is None

    def test_clear_cache(self) -> None:
        window = ContextWindow(max_tokens=10_000)
        block = ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="test",
        )
        window.add(block)
        _ = window.token_count
        window.clear_cache()
        # Should recompute without error
        assert window.token_count > 0

    def test_budget_warnings(self) -> None:
        events_received: list[float] = []

        from contextkit.events import BudgetEventData, on

        @on(ContextEvent.BUDGET_WARNING)
        def handler(event: BudgetEventData) -> None:
            events_received.append(event.percent)

        window = ContextWindow(
            max_tokens=100,
            budget_warnings=[0.5],
        )
        # Add a block that uses >50% of budget
        block = ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="a" * 300,  # Should be >50 tokens
        )
        window.add(block)
        assert len(events_received) >= 1

    def test_repr(self) -> None:
        window = ContextWindow(model="gpt-4o")
        repr_str = repr(window)
        assert "gpt-4o" in repr_str
        assert "blocks=0" in repr_str

    def test_len(self) -> None:
        window = ContextWindow(max_tokens=10_000)
        assert len(window) == 0
        window.add(
            ContextBlock(type=BlockType.SYSTEM_PROMPT, content="test")
        )
        assert len(window) == 1

    def test_blocks_returns_copy(self) -> None:
        window = ContextWindow(max_tokens=10_000)
        block = ContextBlock(
            type=BlockType.SYSTEM_PROMPT, content="test"
        )
        window.add(block)
        blocks = window.blocks
        blocks.clear()
        assert len(window) == 1  # Internal list unaffected

    def test_encoding_property(self) -> None:
        window = ContextWindow(model="gpt-4o")
        assert window.encoding == "o200k_base"

    def test_encoding_default_for_manual_tokens(self) -> None:
        window = ContextWindow(max_tokens=1000)
        assert window.encoding == "cl100k_base"
