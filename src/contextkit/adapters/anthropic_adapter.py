"""Anthropic provider adapter.

Formats a ContextWindow for the Anthropic Messages API. Extracts
system prompts to the top-level `system` parameter and converts
remaining blocks to the Anthropic message format.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("contextkit")

from contextkit.core import BlockType, ContextWindow
from contextkit.observe.events import (
    BlockEventData,
    ContextEvent,
    emit,
)


class AnthropicAdapter:
    """Adapter for the Anthropic Messages API.

    Anthropic's API takes system prompts as a top-level parameter,
    not as a message in the messages array. This adapter handles
    that difference transparently.
    """

    def format(self, window: ContextWindow) -> Dict[str, Any]:
        """Format a ContextWindow for the Anthropic API.

        Extracts SYSTEM_PROMPT blocks to the top-level `system` param.
        All other blocks are converted to messages in Anthropic format.

        Args:
            window: The context window to format.

        Returns:
            A dict with keys: model, max_tokens, system, messages.
        """
        system_parts: List[str] = []
        messages: List[Dict[str, Any]] = []

        sorted_blocks = sorted(window.blocks, key=lambda b: b.priority, reverse=True)

        for block in sorted_blocks:
            if block.type == BlockType.SYSTEM_PROMPT:
                if isinstance(block.content, str):
                    system_parts.append(block.content)
                elif isinstance(block.content, list):
                    for msg in block.content:
                        content = msg.get("content", "")
                        if content:
                            system_parts.append(content)
            else:
                if isinstance(block.content, str):
                    role = _block_type_to_role(block.type)
                    messages.append({"role": role, "content": block.content})
                elif isinstance(block.content, list):
                    messages.extend(block.content)

        system_text = "\n\n".join(system_parts) if system_parts else ""

        logger.info(
            "Formatted for Anthropic: %d messages, %d system parts, %s tokens",
            len(messages),
            len(system_parts),
            f"{window.token_count:,}",
        )

        payload: Dict[str, Any] = {
            "messages": messages,
        }

        if window.model_name:
            payload["model"] = window.model_name

        if system_text:
            payload["system"] = system_text

        emit(
            BlockEventData(
                event=ContextEvent.WINDOW_RENDERED,
                token_count=window.token_count,
                details={
                    "adapter": "anthropic",
                    "block_count": len(window.blocks),
                },
            )
        )

        return payload

    @staticmethod
    def format_messages(
        messages: List[Dict[str, Any]],
        system: str | None = None,
    ) -> Dict[str, Any]:
        """Format raw messages for the Anthropic API.

        Args:
            messages: List of message dicts.
            system: Optional system prompt string.

        Returns:
            A dict ready for anthropic_client.messages.create().
        """
        payload: Dict[str, Any] = {"messages": messages}
        if system:
            payload["system"] = system
        return payload


def _block_type_to_role(block_type: BlockType) -> str:
    """Map a BlockType to an Anthropic message role."""
    role_map = {
        BlockType.SHORT_TERM_MEMORY: "user",
        BlockType.LONG_TERM_MEMORY: "user",
        BlockType.USER_CONTEXT: "user",
        BlockType.RAG: "user",
        BlockType.FILES: "user",
        BlockType.EXAMPLES: "user",
        BlockType.TOOL_DEFINITIONS: "user",
        BlockType.TOOL_OUTPUTS: "user",
        BlockType.OUTPUT_SCHEMAS: "user",
        BlockType.SYSTEM_METADATA: "user",
        BlockType.SCRATCHPAD: "assistant",
    }
    return role_map.get(block_type, "user")
