"""Short-term memory management.

Provides conversation history tracking with automatic trimming
strategies (sliding window, token budget) and standalone utility
functions for trimming conversations without the full framework.
"""

from __future__ import annotations

from typing import Any

from contextkit._tokens import count as count_tokens
from contextkit.core import BlockType, ContextBlock
from contextkit.observe.provenance import Origin


class ShortTermMemory:
    """Manages conversation history with automatic trimming.

    Supports sliding window (keep last N turns) and token-budget
    trimming (keep most recent turns that fit within a token limit).
    Auto-populates Origin with turn range and trimming info.

    Args:
        strategy: Trimming strategy - "sliding_window" or "token_budget".
        max_turns: Maximum number of turns to keep (for sliding_window).
        max_tokens: Maximum token budget for history (for token_budget).
        encoding: Tiktoken encoding name for token counting.
    """

    def __init__(
        self,
        strategy: str = "sliding_window",
        max_turns: int = 20,
        max_tokens: int | None = None,
        encoding: str = "cl100k_base",
    ) -> None:
        if strategy not in ("sliding_window", "token_budget"):
            raise ValueError(
                f"Unknown strategy: {strategy}. Use 'sliding_window' or 'token_budget'."
            )
        self._strategy = strategy
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._encoding = encoding
        self._messages: list[dict[str, Any]] = []
        self._total_turns_added = 0

    @property
    def strategy(self) -> str:
        """The active trimming strategy."""
        return self._strategy

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Current messages after trimming."""
        return list(self._messages)

    @property
    def turn_count(self) -> int:
        """Number of messages currently in memory."""
        return len(self._messages)

    @property
    def total_turns_added(self) -> int:
        """Total turns ever added (including trimmed)."""
        return self._total_turns_added

    @property
    def token_count(self) -> int:
        """Total tokens in current messages."""
        if not self._messages:
            return 0
        return count_tokens(self._messages, self._encoding)

    def add_turn(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a conversation turn.

        Args:
            role: Message role ("user", "assistant", etc.).
            content: Message content.
            metadata: Optional metadata for this turn.
        """
        message: dict[str, Any] = {
            "role": role,
            "content": content,
        }
        if metadata:
            message["metadata"] = metadata
        self._messages.append(message)
        self._total_turns_added += 1
        self._apply_trimming()

    def add_messages(self, messages: list[dict[str, Any]]) -> None:
        """Add multiple messages at once.

        Args:
            messages: List of message dicts with "role" and "content".
        """
        self._messages.extend(messages)
        self._total_turns_added += len(messages)
        self._apply_trimming()

    def to_block(self, name: str = "conversation_history") -> ContextBlock:
        """Convert current history to a ContextBlock.

        Auto-populates the Origin with turn range and trimming info.

        Args:
            name: Block name for inspection.

        Returns:
            A ContextBlock of type SHORT_TERM_MEMORY.
        """
        trimmed = self._total_turns_added - len(self._messages)
        first_turn = self._total_turns_added - len(self._messages) + 1
        last_turn = self._total_turns_added

        origin_details: dict[str, Any] = {
            "turn_range": f"{first_turn}-{last_turn}",
            "strategy": self._strategy,
            "total_turns_seen": self._total_turns_added,
            "turns_kept": len(self._messages),
        }
        if trimmed > 0:
            origin_details["trimmed_turns"] = trimmed

        return ContextBlock(
            type=BlockType.SHORT_TERM_MEMORY,
            content=list(self._messages),
            priority=80,
            name=name,
            origin=Origin(
                source="conversation",
                details=origin_details,
            ),
        )

    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()

    def _apply_trimming(self) -> None:
        """Apply the configured trimming strategy."""
        if self._strategy == "sliding_window":
            self._trim_sliding_window()
        elif self._strategy == "token_budget":
            self._trim_token_budget()

    def _trim_sliding_window(self) -> None:
        """Keep only the last max_turns messages."""
        if len(self._messages) > self._max_turns:
            excess = len(self._messages) - self._max_turns
            self._messages = self._messages[excess:]

    def _trim_token_budget(self) -> None:
        """Keep most recent messages that fit within max_tokens."""
        if self._max_tokens is None:
            return

        # Also apply sliding window if set
        if len(self._messages) > self._max_turns:
            excess = len(self._messages) - self._max_turns
            self._messages = self._messages[excess:]

        # Then trim by token budget from the oldest
        while (
            self._messages
            and count_tokens(self._messages, self._encoding) > self._max_tokens
        ):
            self._messages.pop(0)


def trim_conversation(
    messages: list[dict[str, Any]],
    strategy: str = "sliding_window",
    max_turns: int = 20,
    max_tokens: int | None = None,
    encoding: str = "cl100k_base",
) -> list[dict[str, Any]]:
    """Standalone utility to trim a conversation without ShortTermMemory.

    Args:
        messages: List of message dicts with "role" and "content".
        strategy: "sliding_window" or "token_budget".
        max_turns: Maximum number of turns to keep.
        max_tokens: Maximum token budget (for token_budget strategy).
        encoding: Tiktoken encoding name.

    Returns:
        A trimmed copy of the messages list.
    """
    result = list(messages)

    if strategy == "sliding_window":
        if len(result) > max_turns:
            result = result[-max_turns:]
    elif strategy == "token_budget":
        if len(result) > max_turns:
            result = result[-max_turns:]
        if max_tokens is not None:
            while result and count_tokens(result, encoding) > max_tokens:
                result.pop(0)
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. Use 'sliding_window' or 'token_budget'."
        )

    return result
