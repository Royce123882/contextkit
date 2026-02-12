"""Custom exception hierarchy for contextkit.

All contextkit exceptions inherit from ContextKitError, making it
easy to catch any SDK-level error with a single except clause.
"""

from __future__ import annotations


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
