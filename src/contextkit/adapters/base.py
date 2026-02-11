"""Base protocol for provider adapters.

Defines the interface that all provider adapters must implement.
Each adapter translates a ContextWindow into the message format
expected by a specific LLM provider's API.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from contextkit.core import ContextWindow


@runtime_checkable
class ProviderAdapter(Protocol):
    """Protocol for provider adapters.

    Implementors must provide a `format` method that converts a
    ContextWindow into a provider-specific payload dict.
    """

    def format(self, window: ContextWindow) -> dict[str, Any]:
        """Format a ContextWindow into a provider-specific payload.

        Args:
            window: The context window to format.

        Returns:
            A dict ready to be passed to the provider's API.
        """
        ...

    @staticmethod
    def format_messages(
        messages: list[dict[str, Any]],
        system: str | None = None,
    ) -> dict[str, Any]:
        """Format raw messages without a ContextWindow.

        Convenience method for standalone use. Converts a list of
        message dicts into a provider-specific payload.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            system: Optional system prompt string.

        Returns:
            A dict ready to be passed to the provider's API.
        """
        ...
