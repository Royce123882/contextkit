"""Ollama provider adapter.

Formats a ContextWindow for Ollama's ``/api/chat`` endpoint.
System prompts are included as the first message with role "system"
in the messages array.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from contextkit.core import BlockType, ContextWindow
from contextkit.observe.event_models import BlockEventData, ContextEvent
from contextkit.observe.events import emit

logger = logging.getLogger("contextkit")

_BLOCK_TYPE_ROLE_MAP: Dict[BlockType, str] = {
    BlockType.SYSTEM_PROMPT: "system",
    BlockType.SCRATCHPAD: "assistant",
}


class OllamaAdapter:
    """Adapter for Ollama's /api/chat endpoint.

    Ollama uses a simple message-based format similar to OpenAI.
    System prompts are sent as messages with role "system". The
    model name maps to the ``model`` field in Ollama's API.
    """

    def format(self, window: ContextWindow) -> Dict[str, Any]:
        """Format a ContextWindow for the Ollama chat API.

        System prompts are placed first as ``role: "system"``
        messages. All other blocks follow in priority order.

        Args:
            window: The context window to format.

        Returns:
            A dict with keys: model, messages, stream.
        """
        messages: List[Dict[str, str]] = []

        sorted_blocks = sorted(
            window.blocks, key=lambda block: block.priority, reverse=True
        )

        for block in sorted_blocks:
            if isinstance(block.content, str):
                role = _block_type_to_role(block.type)
                messages.append({"role": role, "content": block.content})
            elif isinstance(block.content, list):
                if block.type == BlockType.SYSTEM_PROMPT:
                    for message in block.content:
                        content = message.get("content", "")
                        if content:
                            messages.append({"role": "system", "content": content})
                else:
                    for message in block.content:
                        messages.append({
                            "role": message.get("role", "user"),
                            "content": message.get("content", ""),
                        })

        logger.info(
            "Formatted for Ollama: %d messages, %s tokens",
            len(messages),
            f"{window.token_count:,}",
        )

        payload: Dict[str, Any] = {
            "messages": messages,
            "stream": False,
        }

        if window.model_name:
            payload["model"] = window.model_name

        emit(
            BlockEventData(
                event=ContextEvent.WINDOW_RENDERED,
                token_count=window.token_count,
                details={
                    "adapter": "ollama",
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
        """Format raw messages for the Ollama chat API.

        Args:
            messages: List of message dicts.
            system: Optional system prompt string.

        Returns:
            A dict ready for Ollama's /api/chat endpoint.
        """
        result_messages: List[Dict[str, Any]] = []
        if system:
            result_messages.append({"role": "system", "content": system})
        result_messages.extend(messages)
        return {"messages": result_messages, "stream": False}


def _block_type_to_role(block_type: BlockType) -> str:
    """Map a BlockType to an Ollama message role.

    Args:
        block_type: The BlockType to map.

    Returns:
        The corresponding role string.
    """
    return _BLOCK_TYPE_ROLE_MAP.get(block_type, "user")
