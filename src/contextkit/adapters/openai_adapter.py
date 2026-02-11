"""OpenAI provider adapter.

Formats a ContextWindow for the OpenAI Chat Completions API. System
prompts are included as messages with role "system" in the messages
array, matching OpenAI's expected format.
"""

from __future__ import annotations

from typing import Any

from contextkit.core import BlockType, ContextWindow
from contextkit.observe.events import (
    BlockEventData,
    ContextEvent,
    emit,
)


class OpenAIAdapter:
    """Adapter for the OpenAI Chat Completions API.

    OpenAI's API expects system prompts as messages with role "system"
    in the messages array, unlike Anthropic which uses a top-level param.
    """

    def format(self, window: ContextWindow) -> dict[str, Any]:
        """Format a ContextWindow for the OpenAI API.

        System prompts are added as {"role": "system", "content": ...}
        messages at the start of the messages array.

        Args:
            window: The context window to format.

        Returns:
            A dict with keys: model, messages.
        """
        messages: list[dict[str, Any]] = []

        sorted_blocks = sorted(
            window.blocks, key=lambda b: b.priority, reverse=True
        )

        for block in sorted_blocks:
            if isinstance(block.content, str):
                role = _block_type_to_role(block.type)
                messages.append(
                    {"role": role, "content": block.content}
                )
            elif isinstance(block.content, list):
                if block.type == BlockType.SYSTEM_PROMPT:
                    for msg in block.content:
                        messages.append(
                            {
                                "role": "system",
                                "content": msg.get("content", ""),
                            }
                        )
                else:
                    messages.extend(block.content)

        payload: dict[str, Any] = {"messages": messages}

        if window.model_name:
            payload["model"] = window.model_name

        emit(
            BlockEventData(
                event=ContextEvent.WINDOW_RENDERED,
                token_count=window.token_count,
                details={
                    "adapter": "openai",
                    "block_count": len(window.blocks),
                },
            )
        )

        return payload

    @staticmethod
    def format_messages(
        messages: list[dict[str, Any]],
        system: str | None = None,
    ) -> dict[str, Any]:
        """Format raw messages for the OpenAI API.

        Args:
            messages: List of message dicts.
            system: Optional system prompt string.

        Returns:
            A dict ready for openai_client.chat.completions.create().
        """
        result_messages: list[dict[str, Any]] = []
        if system:
            result_messages.append(
                {"role": "system", "content": system}
            )
        result_messages.extend(messages)
        return {"messages": result_messages}


def _block_type_to_role(block_type: BlockType) -> str:
    """Map a BlockType to an OpenAI message role."""
    if block_type == BlockType.SYSTEM_PROMPT:
        return "system"
    if block_type == BlockType.SCRATCHPAD:
        return "assistant"
    return "user"
