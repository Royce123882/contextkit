"""LiteLLM provider adapter.

Formats a ContextWindow for the LiteLLM unified API, which uses
OpenAI-compatible message format. System prompts are included as
messages with role "system" in the messages array.

LiteLLM acts as a proxy supporting 100+ LLM providers through a
single OpenAI-compatible interface, so this adapter follows the
OpenAI format while preserving the model identifier for LiteLLM
routing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from contextkit.core import BlockType, ContextWindow
from contextkit.observe.events import BlockEventData, ContextEvent, emit

logger = logging.getLogger("contextkit")

_BLOCK_TYPE_ROLE_MAP: Dict[BlockType, str] = {
    BlockType.SYSTEM_PROMPT: "system",
    BlockType.SCRATCHPAD: "assistant",
}


class LiteLLMAdapter:
    """Adapter for the LiteLLM unified API.

    LiteLLM uses OpenAI-compatible format. The model name is passed
    through directly since LiteLLM uses it for provider routing
    (e.g. "anthropic/claude-3-sonnet", "bedrock/claude-3-haiku").
    """

    def format(self, window: ContextWindow) -> Dict[str, Any]:
        """Format a ContextWindow for the LiteLLM API.

        System prompts are added as messages with role "system".
        The model name is preserved for LiteLLM's provider routing.

        Args:
            window: The context window to format.

        Returns:
            A dict with keys: model, messages.
        """
        messages: List[Dict[str, Any]] = []

        sorted_blocks = sorted(
            window.blocks, key=lambda b: b.priority, reverse=True
        )

        for block in sorted_blocks:
            if isinstance(block.content, str):
                role = _block_type_to_role(block.type)
                messages.append({"role": role, "content": block.content})
            elif isinstance(block.content, list):
                if block.type == BlockType.SYSTEM_PROMPT:
                    for msg in block.content:
                        content = msg.get("content", "")
                        if content:
                            messages.append({"role": "system", "content": content})
                else:
                    messages.extend(block.content)

        logger.info(
            "Formatted for LiteLLM: %d messages, %s tokens",
            len(messages),
            f"{window.token_count:,}",
        )

        payload: Dict[str, Any] = {"messages": messages}

        if window.model_name:
            payload["model"] = window.model_name

        emit(
            BlockEventData(
                event=ContextEvent.WINDOW_RENDERED,
                token_count=window.token_count,
                details={
                    "adapter": "litellm",
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
        """Format raw messages for the LiteLLM API.

        Args:
            messages: List of message dicts.
            system: Optional system prompt string.

        Returns:
            A dict ready for litellm.completion().
        """
        result_messages: List[Dict[str, Any]] = []
        if system:
            result_messages.append({"role": "system", "content": system})
        result_messages.extend(messages)
        return {"messages": result_messages}


def _block_type_to_role(block_type: BlockType) -> str:
    """Map a BlockType to a LiteLLM/OpenAI message role.

    Args:
        block_type: The BlockType to map.

    Returns:
        The corresponding role string.
    """
    return _BLOCK_TYPE_ROLE_MAP.get(block_type, "user")
