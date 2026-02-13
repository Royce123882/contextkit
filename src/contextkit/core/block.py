"""Core data models: BlockType enum, ContextBlock, and BudgetExceededError."""

from __future__ import annotations

import enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from contextkit.utils.token_counting import count as count_tokens
from contextkit.constants import (
    DEFAULT_ENCODING,
    PRIORITY_DEFAULT,
    PRIORITY_SYSTEM_PROMPT,
    PRIORITY_RAG_CHUNK,
    PRIORITY_TOOL_OUTPUT,
    PRIORITY_FILE_CONTEXT,
    PRIORITY_EXAMPLE,
)
from contextkit.observe.provenance import Mutation, Origin


class BlockType(enum.Enum):
    """Enumeration of the 12 recognized context types."""

    SYSTEM_PROMPT = "system_prompt"
    SHORT_TERM_MEMORY = "short_term_memory"
    LONG_TERM_MEMORY = "long_term_memory"
    FILES = "files"
    TOOL_DEFINITIONS = "tool_definitions"
    TOOL_OUTPUTS = "tool_outputs"
    RAG = "rag"
    EXAMPLES = "examples"
    OUTPUT_SCHEMAS = "output_schemas"
    SYSTEM_METADATA = "system_metadata"
    SCRATCHPAD = "scratchpad"
    USER_CONTEXT = "user_context"


class ContextBlock(BaseModel):
    """A typed unit of context with content, priority, metadata, and origin.

    Every piece of information fed to a model -- prompts, memory, files,
    tool schemas, RAG chunks -- is a ContextBlock. Blocks carry provenance
    (where they came from) and mutation history (how they were modified).

    Attributes:
        type: The kind of context this block represents.
        content: The actual content -- a string for prompts, or a list
            of message dicts for conversation history.
        priority: Higher values mean more important to keep (default 50).
        metadata: Arbitrary key-value pairs for custom data.
        name: Optional human-readable label for inspection/explain.
        origin: Where this block came from (auto-populated by SDK managers).
        mutations: History of changes made by pipeline steps.
    """

    type: BlockType = Field(description="The kind of context this block represents.")
    content: str | List[Dict[str, Any]] = Field(
        description="The actual content -- a string for prompts, or a list of message dicts for conversation history."
    )
    priority: int = Field(
        default=PRIORITY_DEFAULT,
        description="Higher values mean more important to keep (default 50).",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value pairs for custom data.",
    )
    name: str | None = Field(
        default=None,
        description="Optional human-readable label for inspection/explain.",
    )
    origin: Origin | None = Field(
        default=None,
        description="Where this block came from (auto-populated by SDK managers).",
    )
    mutations: List[Mutation] = Field(
        default_factory=list,
        description="History of changes made by pipeline steps.",
    )

    model_config = {"arbitrary_types_allowed": True}

    @property
    def token_count(self) -> int:
        """Count tokens in this block's content using cached tokenizer."""
        encoding = self.metadata.get("encoding", DEFAULT_ENCODING)
        return count_tokens(self.content, encoding)

    @property
    def display_name(self) -> str:
        """Return the name for display, falling back to the block type."""
        return self.name or self.type.value

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        origin_str = f", origin={self.origin.source}" if self.origin else ""
        return (
            f"ContextBlock(type={self.type.value}, "
            f"name={self.display_name!r}, "
            f"tokens={self.token_count}, "
            f"priority={self.priority}"
            f"{origin_str})"
        )

    # ------------------------------------------------------------------
    # Convenience factory methods
    # ------------------------------------------------------------------

    @classmethod
    def system(
        cls,
        content: str,
        *,
        priority: int = PRIORITY_SYSTEM_PROMPT,
        name: str | None = None,
        **kwargs: Any,
    ) -> "ContextBlock":
        """Create a SYSTEM_PROMPT block.

        Args:
            content: The system prompt text.
            priority: Block priority (defaults to PRIORITY_SYSTEM_PROMPT).
            name: Optional display name.
            **kwargs: Additional fields passed to the constructor.

        Returns:
            A ContextBlock of type SYSTEM_PROMPT.
        """
        return cls(
            type=BlockType.SYSTEM_PROMPT,
            content=content,
            priority=priority,
            name=name or "system_prompt",
            **kwargs,
        )

    @classmethod
    def rag(
        cls,
        content: str,
        *,
        priority: int = PRIORITY_RAG_CHUNK,
        origin: Origin | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> "ContextBlock":
        """Create a RAG block.

        Args:
            content: The retrieved content.
            priority: Block priority (defaults to PRIORITY_RAG_CHUNK).
            origin: Optional provenance origin.
            name: Optional display name.
            **kwargs: Additional fields passed to the constructor.

        Returns:
            A ContextBlock of type RAG.
        """
        return cls(
            type=BlockType.RAG,
            content=content,
            priority=priority,
            origin=origin,
            name=name or "rag",
            **kwargs,
        )

    @classmethod
    def memory(
        cls,
        content: str,
        *,
        priority: int = PRIORITY_DEFAULT,
        name: str | None = None,
        **kwargs: Any,
    ) -> "ContextBlock":
        """Create a SHORT_TERM_MEMORY block.

        Args:
            content: The memory content.
            priority: Block priority (defaults to PRIORITY_DEFAULT).
            name: Optional display name.
            **kwargs: Additional fields passed to the constructor.

        Returns:
            A ContextBlock of type SHORT_TERM_MEMORY.
        """
        return cls(
            type=BlockType.SHORT_TERM_MEMORY,
            content=content,
            priority=priority,
            name=name or "memory",
            **kwargs,
        )

    @classmethod
    def tool_output(
        cls,
        content: str,
        *,
        tool_name: str = "",
        priority: int = PRIORITY_TOOL_OUTPUT,
        name: str | None = None,
        **kwargs: Any,
    ) -> "ContextBlock":
        """Create a TOOL_OUTPUTS block with auto-populated Origin.

        Args:
            content: The tool output content.
            tool_name: Name of the tool that produced the output.
            priority: Block priority (defaults to PRIORITY_TOOL_OUTPUT).
            name: Optional display name.
            **kwargs: Additional fields passed to the constructor.

        Returns:
            A ContextBlock of type TOOL_OUTPUTS.
        """
        origin = Origin.from_tool(tool_name) if tool_name else None
        return cls(
            type=BlockType.TOOL_OUTPUTS,
            content=content,
            priority=priority,
            origin=origin,
            name=name or "tool_output",
            **kwargs,
        )

    @classmethod
    def examples(
        cls,
        content: str | List[Dict[str, Any]],
        *,
        priority: int = PRIORITY_EXAMPLE,
        name: str | None = None,
        **kwargs: Any,
    ) -> "ContextBlock":
        """Create an EXAMPLES block.

        Args:
            content: Example content (string or list of message dicts).
            priority: Block priority (defaults to PRIORITY_EXAMPLE).
            name: Optional display name.
            **kwargs: Additional fields passed to the constructor.

        Returns:
            A ContextBlock of type EXAMPLES.
        """
        return cls(
            type=BlockType.EXAMPLES,
            content=content,
            priority=priority,
            name=name or "examples",
            **kwargs,
        )

    @classmethod
    def file(
        cls,
        content: str,
        *,
        file_path: str = "",
        priority: int = PRIORITY_FILE_CONTEXT,
        name: str | None = None,
        **kwargs: Any,
    ) -> "ContextBlock":
        """Create a FILES block with auto-populated Origin.

        Args:
            content: The file content.
            file_path: Path to the source file (used for Origin).
            priority: Block priority (defaults to PRIORITY_FILE_CONTEXT).
            name: Optional display name.
            **kwargs: Additional fields passed to the constructor.

        Returns:
            A ContextBlock of type FILES.
        """
        origin = Origin.from_file(file_path) if file_path else None
        return cls(
            type=BlockType.FILES,
            content=content,
            priority=priority,
            origin=origin,
            name=name or "file",
            **kwargs,
        )


class BudgetExceededError(Exception):
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
            block_list = ", ".join(
                f"'{name}' ({tokens:,}t)"
                for name, tokens in self.removable_blocks[:3]
            )
            lines.append(
                f"  - Remove low-priority blocks ({total_removable:,} tokens "
                f"available): {block_list}"
            )

        super().__init__("\n".join(lines))
