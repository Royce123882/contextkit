"""Provenance tracking for context blocks.

Every ContextBlock carries an Origin that records where it came from,
and a list of Mutations that record how it was modified during assembly
or pipeline processing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from pydantic import BaseModel, Field


class Origin(BaseModel):
    """Records where a context block came from.

    The `source` field identifies the subsystem that produced the block
    (e.g. "prompt", "conversation", "rag", "file", "tool", "memory",
    "user", "example", "tool_output").

    The `details` dict stores source-specific metadata such as template
    names, queries, retriever names, relevance scores, file paths, etc.
    """

    source: str = Field(description="Subsystem that produced the block (e.g. 'prompt', 'rag', 'memory', 'file', 'tool').")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When this origin was created.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Source-specific metadata such as template names, queries, retriever names, etc.")

    # Convenience properties for common detail fields
    @property
    def template(self) -> str | None:
        """Template name for prompt origins."""
        return self.details.get("template")

    @property
    def query(self) -> str | None:
        """Query string for RAG or memory retrieval origins."""
        return self.details.get("query")

    @property
    def retriever(self) -> str | None:
        """Retriever name for RAG origins."""
        return self.details.get("retriever")

    @property
    def relevance_score(self) -> float | None:
        """Relevance score for retrieved content."""
        return self.details.get("relevance_score")

    @property
    def file_path(self) -> str | None:
        """File path for file-based origins."""
        return self.details.get("file_path")

    @property
    def tool_name(self) -> str | None:
        """Tool name for tool-related origins."""
        return self.details.get("tool_name")

    @property
    def turn_range(self) -> str | None:
        """Turn range for conversation origins."""
        return self.details.get("turn_range")

    def summary(self) -> str:
        """Return a concise human-readable summary of this origin."""
        parts = [self.source]
        if self.retriever:
            parts.append(self.retriever)
        if self.template:
            parts.append(self.template)
        if self.tool_name:
            parts.append(self.tool_name)
        if self.file_path:
            parts.append(self.file_path)
        if self.relevance_score is not None:
            parts.append(f"relevance={self.relevance_score:.2f}")
        if self.query:
            truncated = self.query[:30] + "..." if len(self.query) > 30 else self.query
            parts.append(f'query="{truncated}"')
        return "/".join(parts[:3]) + (
            f" ({', '.join(parts[3:])})" if len(parts) > 3 else ""
        )


class Mutation(BaseModel):
    """Records a modification made to a context block.

    Created when a pipeline step trims, compacts, deduplicates,
    reorders, or otherwise modifies a block. The full history is
    preserved so every change is traceable.

    Attributes:
        step: Name of the pipeline step that made the modification.
        action: Type of action performed (e.g. 'removed', 'compressed', 'moved').
        detail: Human-readable description of the change.
        tokens_before: Token count before the modification.
        tokens_after: Token count after the modification.
        before_content: Original content before modification (optional).
        after_content: Content after modification (optional).
        timestamp: When the mutation occurred.
    """

    step: str = Field(description="Name of the pipeline step that made the modification.")
    action: str = Field(description="Type of action performed (e.g. 'removed', 'compressed', 'moved').")
    detail: str = Field(description="Human-readable description of the change.")
    tokens_before: int = Field(description="Token count before the modification.")
    tokens_after: int = Field(description="Token count after the modification.")
    before_content: str | None = Field(default=None, description="Original content before modification.")
    after_content: str | None = Field(default=None, description="Content after modification.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the mutation occurred.")
