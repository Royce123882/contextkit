"""Custom exception hierarchy for contextkit.

All contextkit exceptions inherit from ContextKitError, making it
easy to catch any SDK-level error with a single except clause.
"""

from __future__ import annotations

from typing import List


class ContextKitError(Exception):
    """Base exception for all contextkit errors."""


class PathTraversalError(ContextKitError):
    """Raised when a file path escapes the allowed base directory.

    Attributes:
        attempted_path: The path that was rejected.
        allowed_directory: The base directory that paths must stay within.
    """

    def __init__(self, attempted_path: str, allowed_directory: str) -> None:
        self.attempted_path = attempted_path
        self.allowed_directory = allowed_directory
        super().__init__(
            f"Path '{attempted_path}' is outside allowed directory "
            f"'{allowed_directory}'"
        )


class BackendConnectionError(ContextKitError):
    """Raised when a memory or retriever backend is unreachable."""


class BackendTimeoutError(ContextKitError):
    """Raised when a backend operation exceeds the allowed time."""


class RecordNotFoundError(ContextKitError):
    """Raised when a memory record or block is not found."""


class InvalidBlockError(ContextKitError, ValueError):
    """Raised when a ContextBlock fails validation."""


class TemplateRenderError(ContextKitError):
    """Raised when a prompt template cannot be rendered.

    Attributes:
        template_name: Name of the template that failed.
        missing_variables: List of variable names that were not provided.
    """

    def __init__(
        self,
        template_name: str,
        missing_variables: list[str],
    ) -> None:
        self.template_name = template_name
        self.missing_variables = missing_variables
        super().__init__(
            f"Template '{template_name}' has unresolved variables: "
            f"{missing_variables}"
        )


class SyncInAsyncError(ContextKitError):
    """Raised when sync wrappers are called inside a running event loop.

    Use the async API directly instead of the sync wrappers when
    running inside an existing event loop.
    """

    def __init__(self) -> None:
        super().__init__(
            "Cannot use sync wrappers inside an async context. "
            "Use the async API directly (e.g., await memory.retrieve(...))."
        )


class UnknownModelError(ContextKitError):
    """Raised when a model name is not found in the registry.

    Attributes:
        model_name: The model identifier that was not found.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        super().__init__(
            f"Unknown model: '{model_name}'. Use register_model() to add custom models."
        )


class BudgetExceededError(ContextKitError):
    """Raised when adding a block would exceed the token budget.

    Includes context-aware suggestions when removable blocks are
    available in the window.

    Attributes:
        block_name: Name of the block that couldn't fit.
        block_tokens: Token count of the rejected block.
        budget_remaining: Tokens remaining before the budget is exhausted.
        max_tokens: Total token budget of the window.
        removable_blocks: Pairs of (block_name, token_count) for
            low-priority blocks that could be removed to free space.
    """

    def __init__(
        self,
        block_name: str,
        block_tokens: int,
        budget_remaining: int,
        max_tokens: int,
        removable_blocks: List[tuple[str, int]] | None = None,
    ) -> None:
        self.block_name = block_name
        self.block_tokens = block_tokens
        self.budget_remaining = budget_remaining
        self.max_tokens = max_tokens
        self.removable_blocks = removable_blocks or []

        tokens_needed = block_tokens - budget_remaining
        lines = [
            f"Block '{block_name}' needs {block_tokens:,} tokens "
            f"but only {budget_remaining:,} remain "
            f"(max: {max_tokens:,}, need {tokens_needed:,} more).",
            "",
            "Suggestions:",
            f"  - Run a pipeline: ContextPipeline.balanced(max_tokens={max_tokens})",
            "  - Use window.will_fit(block) to check before adding",
        ]

        if self.removable_blocks:
            total_removable = sum(tokens for _, tokens in self.removable_blocks)
            removable_summary = ", ".join(
                f"'{name}' ({tokens:,}t)"
                for name, tokens in self.removable_blocks[:3]
            )
            lines.append(
                f"  - Remove low-priority blocks ({total_removable:,} tokens "
                f"available): {removable_summary}"
            )

        super().__init__("\n".join(lines))
