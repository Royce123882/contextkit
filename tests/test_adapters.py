"""Tests for provider adapters (Anthropic and OpenAI)."""

from __future__ import annotations

from contextkit.adapters.anthropic_adapter import AnthropicAdapter
from contextkit.adapters.openai_adapter import OpenAIAdapter
from contextkit.core import BlockType, ContextBlock, ContextWindow
from contextkit.events import ContextEvent, clear_handlers, on
from contextkit.observe.events import EventData


class TestAnthropicAdapter:
    """Tests for the Anthropic provider adapter."""

    def setup_method(self) -> None:
        clear_handlers()

    def test_format_produces_valid_structure(self) -> None:
        window = ContextWindow(model="claude-sonnet-4-5-20250929")
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="You are helpful.",
                priority=100,
                name="sys",
            )
        )
        window.add(
            ContextBlock(
                type=BlockType.USER_CONTEXT,
                content="User likes Python.",
                priority=80,
                name="user",
            )
        )

        adapter = AnthropicAdapter()
        payload = adapter.format(window)

        assert "model" in payload
        assert "system" in payload
        assert "messages" in payload
        assert payload["model"] == "claude-sonnet-4-5-20250929"
        assert payload["system"] == "You are helpful."

    def test_system_prompt_extracted_to_top_level(self) -> None:
        window = ContextWindow(max_tokens=10_000)
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="System instruction",
                priority=100,
            )
        )
        window.add(
            ContextBlock(
                type=BlockType.USER_CONTEXT,
                content="User info",
                priority=80,
            )
        )

        adapter = AnthropicAdapter()
        payload = adapter.format(window)

        assert payload["system"] == "System instruction"
        # System prompt should NOT be in messages
        for msg in payload["messages"]:
            assert msg["content"] != "System instruction"

    def test_multiple_system_prompts_joined(self) -> None:
        window = ContextWindow(max_tokens=10_000)
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="Rule 1",
                priority=100,
                name="sys1",
            )
        )
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="Rule 2",
                priority=90,
                name="sys2",
            )
        )

        adapter = AnthropicAdapter()
        payload = adapter.format(window)

        assert "Rule 1" in payload["system"]
        assert "Rule 2" in payload["system"]

    def test_message_list_content_expanded(self) -> None:
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

        adapter = AnthropicAdapter()
        payload = adapter.format(window)

        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "user"

    def test_format_messages_standalone(self) -> None:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        payload = AnthropicAdapter.format_messages(messages, system="Be helpful.")
        assert payload["system"] == "Be helpful."
        assert payload["messages"] == messages

    def test_format_messages_without_system(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        payload = AnthropicAdapter.format_messages(messages)
        assert "system" not in payload
        assert payload["messages"] == messages

    def test_format_emits_window_rendered_event(self) -> None:
        received: list[EventData] = []

        @on(ContextEvent.WINDOW_RENDERED)
        def handler(event: EventData) -> None:
            received.append(event)

        window = ContextWindow(max_tokens=10_000)
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="test",
            )
        )

        adapter = AnthropicAdapter()
        adapter.format(window)
        assert len(received) == 1

    def test_no_system_blocks_means_no_system_key(self) -> None:
        window = ContextWindow(max_tokens=10_000)
        window.add(
            ContextBlock(
                type=BlockType.USER_CONTEXT,
                content="User info",
                priority=80,
            )
        )

        adapter = AnthropicAdapter()
        payload = adapter.format(window)
        assert "system" not in payload


class TestOpenAIAdapter:
    """Tests for the OpenAI provider adapter."""

    def setup_method(self) -> None:
        clear_handlers()

    def test_format_produces_valid_structure(self) -> None:
        window = ContextWindow(model="gpt-4o")
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="You are helpful.",
                priority=100,
                name="sys",
            )
        )
        window.add(
            ContextBlock(
                type=BlockType.USER_CONTEXT,
                content="User likes Python.",
                priority=80,
                name="user",
            )
        )

        adapter = OpenAIAdapter()
        payload = adapter.format(window)

        assert "model" in payload
        assert "messages" in payload
        assert payload["model"] == "gpt-4o"

    def test_system_prompt_in_messages_array(self) -> None:
        window = ContextWindow(max_tokens=10_000)
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="System instruction",
                priority=100,
            )
        )
        window.add(
            ContextBlock(
                type=BlockType.USER_CONTEXT,
                content="User info",
                priority=80,
            )
        )

        adapter = OpenAIAdapter()
        payload = adapter.format(window)

        # System should be in messages as role=system
        system_msgs = [m for m in payload["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "System instruction"

    def test_message_list_content_expanded(self) -> None:
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

        adapter = OpenAIAdapter()
        payload = adapter.format(window)

        assert len(payload["messages"]) == 2

    def test_format_messages_standalone(self) -> None:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        payload = OpenAIAdapter.format_messages(messages, system="Be helpful.")
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "Be helpful."
        assert len(payload["messages"]) == 3

    def test_format_messages_without_system(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        payload = OpenAIAdapter.format_messages(messages)
        assert len(payload["messages"]) == 1

    def test_format_emits_window_rendered_event(self) -> None:
        received: list[EventData] = []

        @on(ContextEvent.WINDOW_RENDERED)
        def handler(event: EventData) -> None:
            received.append(event)

        window = ContextWindow(max_tokens=10_000)
        window.add(
            ContextBlock(
                type=BlockType.SYSTEM_PROMPT,
                content="test",
            )
        )

        adapter = OpenAIAdapter()
        adapter.format(window)
        assert len(received) == 1

    def test_scratchpad_uses_assistant_role(self) -> None:
        window = ContextWindow(max_tokens=10_000)
        window.add(
            ContextBlock(
                type=BlockType.SCRATCHPAD,
                content="My working notes...",
                priority=50,
            )
        )

        adapter = OpenAIAdapter()
        payload = adapter.format(window)
        assert payload["messages"][0]["role"] == "assistant"


class TestCustomAdapter:
    """Tests that custom adapters work via the Protocol."""

    def test_custom_adapter_implements_protocol(self) -> None:
        from contextkit.adapters.base import ProviderAdapter

        class MyAdapter:
            def format(self, window: ContextWindow) -> dict:  # type: ignore[type-arg]
                return {"custom": True, "messages": []}

            @staticmethod
            def format_messages(
                messages: list[dict],  # type: ignore[type-arg]
                system: str | None = None,
            ) -> dict:  # type: ignore[type-arg]
                return {"messages": messages}

        adapter = MyAdapter()
        assert isinstance(adapter, ProviderAdapter)

        window = ContextWindow(max_tokens=1000)
        result = adapter.format(window)
        assert result["custom"] is True
