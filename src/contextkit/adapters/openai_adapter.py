"""OpenAI provider adapter.

Formats a ContextWindow for the OpenAI Chat Completions API. System
prompts are included as messages with role "system" in the messages
array, matching OpenAI's expected format.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("contextkit")

from contextkit.core import BlockType, ContextWindow
from contextkit.observe.event_models import BlockEventData, ContextEvent
from contextkit.observe.events import emit


class OpenAIAdapter:
    """Adapter for the OpenAI Chat Completions API.

    OpenAI's API expects system prompts as messages with role "system"
    in the messages array, unlike Anthropic which uses a top-level param.
    """

    def format(self, window: ContextWindow) -> Dict[str, Any]:
        """Format a ContextWindow for the OpenAI API.

        System prompts are added as {"role": "system", "content": ...}
        messages at the start of the messages array.

        Args:
            window: The context window to format.

        Returns:
            A dict with keys: model, messages.
        """
        messages: List[Dict[str, Any]] = []

        sorted_blocks = sorted(window.blocks, key=lambda block: block.priority, reverse=True)

        for block in sorted_blocks:
            if isinstance(block.content, str):
                role = _block_type_to_role(block.type)
                messages.append({"role": role, "content": block.content})
            elif isinstance(block.content, list):
                if block.type == BlockType.SYSTEM_PROMPT:
                    for message in block.content:
                        messages.append(
                            {
                                "role": "system",
                                "content": message.get("content", ""),
                            }
                        )
                else:
                    messages.extend(block.content)

        logger.info(
            "Formatted for OpenAI: %d messages, %s tokens",
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
                    "adapter": "openai",
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
        """Format raw messages for the OpenAI API.

        Args:
            messages: List of message dicts.
            system: Optional system prompt string.

        Returns:
            A dict ready for openai_client.chat.completions.create().
        """
        result_messages: List[Dict[str, Any]] = []
        if system:
            result_messages.append({"role": "system", "content": system})
        result_messages.extend(messages)
        return {"messages": result_messages}


_BLOCK_TYPE_ROLE_MAP: Dict[BlockType, str] = {
    BlockType.SYSTEM_PROMPT: "system",
    BlockType.SCRATCHPAD: "assistant",
}


def _block_type_to_role(block_type: BlockType) -> str:
    """Map a BlockType to an OpenAI message role.

    Uses a dictionary lookup for easy extension. Defaults to
    ``"user"`` for types not in the mapping.

    Args:
        block_type: The BlockType to map.

    Returns:
        The corresponding OpenAI role string.
    """
    return _BLOCK_TYPE_ROLE_MAP.get(block_type, "user")
