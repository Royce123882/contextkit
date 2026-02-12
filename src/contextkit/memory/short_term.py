"""Short-term memory management.

Provides conversation history tracking with automatic trimming
strategies (sliding window, token budget) and standalone utility
functions for trimming conversations without the full framework.

Uses ``collections.deque`` internally for O(1) left-side removal,
avoiding the O(n) cost of ``list.pop(0)`` during token-budget trimming.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List

from contextkit.constants import DEFAULT_ENCODING, PRIORITY_SHORT_TERM_MEMORY
from contextkit.core import BlockType, ContextBlock
from contextkit.exceptions import InvalidBlockError
from contextkit.observe.provenance import Origin
from contextkit.utils.token_counting import count as count_tokens

logger = logging.getLogger("contextkit")

_VALID_STRATEGIES = frozenset({"sliding_window", "token_budget"})


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

    Raises:
        InvalidBlockError: If an unknown strategy is provided.
    """

    def __init__(
        self,
        strategy: str = "sliding_window",
        max_turns: int = 20,
        max_tokens: int | None = None,
        encoding: str = DEFAULT_ENCODING,
    ) -> None:
        if strategy not in _VALID_STRATEGIES:
            raise InvalidBlockError(
                f"Unknown strategy: {strategy}. "
                f"Use 'sliding_window' or 'token_budget'."
            )
        self._strategy = strategy
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._encoding = encoding
        self._messages: deque[Dict[str, Any]] = deque()
        self._total_turns_added = 0

    @property
    def strategy(self) -> str:
        """The active trimming strategy."""
        return self._strategy

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """Current messages after trimming (returned as a list copy)."""
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
        return count_tokens(list(self._messages), self._encoding)

    def add_turn(
        self,
        role: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Add a conversation turn.

        Args:
            role: Message role ("user", "assistant", etc.).
            content: Message content.
            metadata: Optional metadata for this turn.
        """
        message: Dict[str, Any] = {
            "role": role,
            "content": content,
        }
        if metadata:
            message["metadata"] = metadata
        self._messages.append(message)
        self._total_turns_added += 1
        before_count = len(self._messages)
        self._apply_trimming()
        trimmed = before_count - len(self._messages)
        if trimmed > 0:
            logger.debug(
                "Trimmed %d turns (strategy=%s, keeping %d)",
                trimmed,
                self._strategy,
                len(self._messages),
            )

    def add_messages(self, messages: List[Dict[str, Any]]) -> None:
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

        origin_details: Dict[str, Any] = {
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
            priority=PRIORITY_SHORT_TERM_MEMORY,
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
        """Keep only the last max_turns messages using O(1) popleft."""
        while len(self._messages) > self._max_turns:
            self._messages.popleft()

    def _trim_token_budget(self) -> None:
        """Keep most recent messages that fit within max_tokens.

        Uses deque.popleft() for O(1) removal from the front,
        avoiding the O(n) cost of list.pop(0).
        """
        if self._max_tokens is None:
            return

        # Also apply sliding window if set
        while len(self._messages) > self._max_turns:
            self._messages.popleft()

        # Then trim by token budget from the oldest
        while (
            self._messages
            and count_tokens(list(self._messages), self._encoding)
            > self._max_tokens
        ):
            self._messages.popleft()


def trim_conversation(
    messages: List[Dict[str, Any]],
    strategy: str = "sliding_window",
    max_turns: int = 20,
    max_tokens: int | None = None,
    encoding: str = DEFAULT_ENCODING,
) -> List[Dict[str, Any]]:
    """Standalone utility to trim a conversation without ShortTermMemory.

    Uses deque internally for O(1) front removal during token-budget trimming.

    Args:
        messages: List of message dicts with "role" and "content".
        strategy: "sliding_window" or "token_budget".
        max_turns: Maximum number of turns to keep.
        max_tokens: Maximum token budget (for token_budget strategy).
        encoding: Tiktoken encoding name.

    Returns:
        A trimmed copy of the messages list.

    Raises:
        InvalidBlockError: If an unknown strategy is provided.
    """
    result: deque[Dict[str, Any]] = deque(messages)

    if strategy == "sliding_window":
        while len(result) > max_turns:
            result.popleft()
    elif strategy == "token_budget":
        while len(result) > max_turns:
            result.popleft()
        if max_tokens is not None:
            while result and count_tokens(list(result), encoding) > max_tokens:
                result.popleft()
    else:
        raise InvalidBlockError(
            f"Unknown strategy: {strategy}. "
            f"Use 'sliding_window' or 'token_budget'."
        )

    return list(result)
