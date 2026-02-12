"""Core data models: BlockType enum, ContextBlock, and BudgetExceededError."""

from __future__ import annotations

import enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from contextkit.utils.token_counting import count as count_tokens
from contextkit.constants import DEFAULT_ENCODING, PRIORITY_DEFAULT
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


class BudgetExceededError(Exception):
    """Raised when adding a block would exceed the token budget."""

    def __init__(
        self,
        block_name: str,
        block_tokens: int,
        budget_remaining: int,
        max_tokens: int,
    ) -> None:
        self.block_name = block_name
        self.block_tokens = block_tokens
        self.budget_remaining = budget_remaining
        self.max_tokens = max_tokens
        super().__init__(
            f"Block '{block_name}' ({block_tokens:,} tokens) exceeds "
            f"budget remaining ({budget_remaining:,} / {max_tokens:,} tokens)"
        )
