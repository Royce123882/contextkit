"""AWS Bedrock provider adapter.

Formats a ContextWindow for the AWS Bedrock Converse API. Like the
Anthropic adapter, system prompts are extracted to a separate
top-level field rather than included in the messages array.

The output dict is suitable for passing to the boto3
``bedrock-runtime`` client's ``converse()`` method.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from contextkit.core import BlockType, ContextWindow
from contextkit.observe.events import BlockEventData, ContextEvent, emit

logger = logging.getLogger("contextkit")

_BLOCK_TYPE_ROLE_MAP: Dict[BlockType, str] = {
    BlockType.SYSTEM_PROMPT: "user",
    BlockType.SCRATCHPAD: "assistant",
}


class BedrockAdapter:
    """Adapter for the AWS Bedrock Converse API.

    AWS Bedrock's Converse API takes system prompts as a top-level
    ``system`` parameter (a list of content blocks), similar to
    Anthropic. This adapter handles that formatting transparently.
    """

    def format(self, window: ContextWindow) -> Dict[str, Any]:
        """Format a ContextWindow for the Bedrock Converse API.

        Extracts SYSTEM_PROMPT blocks into the top-level ``system``
        parameter as content blocks. All other blocks are converted
        to messages in Bedrock's format.

        Args:
            window: The context window to format.

        Returns:
            A dict with keys: modelId, system, messages.
        """
        system_blocks: List[Dict[str, Any]] = []
        messages: List[Dict[str, Any]] = []

        sorted_blocks = sorted(
            window.blocks, key=lambda b: b.priority, reverse=True
        )

        for block in sorted_blocks:
            if block.type == BlockType.SYSTEM_PROMPT:
                system_text = self._extract_text(block.content)
                if system_text:
                    system_blocks.append({"text": system_text})
            else:
                role = _block_type_to_role(block.type)
                content_text = self._extract_text(block.content)
                if content_text:
                    messages.append({
                        "role": role,
                        "content": [{"text": content_text}],
                    })
                elif isinstance(block.content, list):
                    for msg in block.content:
                        msg_content = msg.get("content", "")
                        msg_role = msg.get("role", role)
                        if msg_content:
                            messages.append({
                                "role": msg_role,
                                "content": [{"text": msg_content}],
                            })

        logger.info(
            "Formatted for Bedrock: %d messages, %d system blocks, %s tokens",
            len(messages),
            len(system_blocks),
            f"{window.token_count:,}",
        )

        payload: Dict[str, Any] = {"messages": messages}

        if window.model_name:
            payload["modelId"] = window.model_name

        if system_blocks:
            payload["system"] = system_blocks

        emit(
            BlockEventData(
                event=ContextEvent.WINDOW_RENDERED,
                token_count=window.token_count,
                details={
                    "adapter": "bedrock",
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
        """Format raw messages for the Bedrock Converse API.

        Args:
            messages: List of message dicts with "role" and "content".
            system: Optional system prompt string.

        Returns:
            A dict ready for bedrock_client.converse().
        """
        bedrock_messages: List[Dict[str, Any]] = []
        for msg in messages:
            bedrock_messages.append({
                "role": msg.get("role", "user"),
                "content": [{"text": msg.get("content", "")}],
            })

        payload: Dict[str, Any] = {"messages": bedrock_messages}
        if system:
            payload["system"] = [{"text": system}]
        return payload

    @staticmethod
    def _extract_text(content: Any) -> str:
        """Extract plain text from block content.

        Args:
            content: Block content (string, list, or other).

        Returns:
            Extracted text string, or empty string if not extractable.
        """
        if isinstance(content, str):
            return content
        return ""


def _block_type_to_role(block_type: BlockType) -> str:
    """Map a BlockType to a Bedrock Converse message role.

    Args:
        block_type: The BlockType to map.

    Returns:
        The corresponding Bedrock role string.
    """
    return _BLOCK_TYPE_ROLE_MAP.get(block_type, "user")
