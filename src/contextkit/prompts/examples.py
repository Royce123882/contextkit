"""Few-shot example store.

Manages canonical input-output pairs for few-shot prompting.
Supports dynamic selection based on similarity and budget-aware
example fitting.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from contextkit._tokens import count as count_tokens
from contextkit.core import BlockType, ContextBlock
from contextkit.observe.provenance import Origin
from contextkit.utils.text_similarity import word_overlap_score


class Example(BaseModel):
    """A single few-shot example.

    Attributes:
        example_id: Unique identifier.
        input_text: The example input.
        output_text: The expected output.
        tags: Categorization tags.
        metadata: Additional metadata.
    """

    example_id: str
    input_text: str
    output_text: str
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def formatted(self) -> str:
        """Formatted example as input/output pair."""
        return f"Input: {self.input_text}\nOutput: {self.output_text}"


class ExampleStore:
    """Manages few-shot examples with selection and budget fitting.

    Register examples and select the best ones for a given input.
    Supports budget-aware fitting to maximize examples within a
    token limit.
    """

    def __init__(self) -> None:
        self._examples: dict[str, Example] = {}

    def add(
        self,
        example_id: str,
        input_text: str,
        output_text: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Example:
        """Add an example to the store.

        Args:
            example_id: Unique identifier.
            input_text: Example input.
            output_text: Expected output.
            tags: Optional categorization tags.
            metadata: Optional metadata.

        Returns:
            The created Example.
        """
        example = Example(
            example_id=example_id,
            input_text=input_text,
            output_text=output_text,
            tags=tags or [],
            metadata=metadata or {},
        )
        self._examples[example_id] = example
        return example

    def get(self, example_id: str) -> Example:
        """Get an example by ID.

        Args:
            example_id: The example identifier.

        Returns:
            The matching Example.

        Raises:
            KeyError: If not found.
        """
        if example_id not in self._examples:
            raise KeyError(f"No example with ID '{example_id}'")
        return self._examples[example_id]

    def select(
        self,
        input_text: str,
        top_k: int = 3,
        tags: list[str] | None = None,
    ) -> list[tuple[Example, float]]:
        """Select examples most similar to the given input.

        Uses word-overlap similarity for selection. Returns
        examples sorted by similarity score.

        Args:
            input_text: The current input to match against.
            top_k: Maximum examples to return.
            tags: Optional tag filter.

        Returns:
            List of (Example, similarity_score) tuples.
        """
        scored: list[tuple[float, Example]] = []

        for example in self._examples.values():
            if tags and not all(t in example.tags for t in tags):
                continue

            similarity = word_overlap_score(input_text, example.input_text)
            scored.append((similarity, example))

        scored.sort(key=lambda entry: entry[0], reverse=True)
        return [(example, score) for score, example in scored[:top_k]]

    def fit_to_budget(
        self,
        input_text: str,
        max_tokens: int,
        top_k: int = 10,
        tags: list[str] | None = None,
        encoding: str = "cl100k_base",
        priority: int = 55,
    ) -> list[ContextBlock]:
        """Select best examples that fit within a token budget.

        Selects examples by similarity, then greedily adds them
        until the token budget is exhausted.

        Args:
            input_text: Current input for similarity matching.
            max_tokens: Maximum total tokens for examples.
            top_k: Maximum candidates to consider.
            tags: Optional tag filter.
            encoding: Tiktoken encoding name.
            priority: Block priority for created blocks.

        Returns:
            List of ContextBlocks, one per example, within budget.
        """
        selected = self.select(input_text, top_k=top_k, tags=tags)

        blocks: list[ContextBlock] = []
        total_tokens = 0

        for example, similarity in selected:
            content = example.formatted
            token_count = count_tokens(content, encoding)

            if total_tokens + token_count > max_tokens:
                continue

            origin = Origin(
                source="example",
                details={
                    "example_id": example.example_id,
                    "similarity_score": round(similarity, 3),
                },
            )

            block = ContextBlock(
                type=BlockType.EXAMPLES,
                content=content,
                priority=priority,
                name=f"example_{example.example_id}",
                origin=origin,
            )
            blocks.append(block)
            total_tokens += token_count

        return blocks

    def list_examples(self) -> list[str]:
        """List all example IDs."""
        return sorted(self._examples.keys())

    @property
    def example_count(self) -> int:
        """Number of stored examples."""
        return len(self._examples)
