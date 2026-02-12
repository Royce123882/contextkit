"""Few-shot example store with selection and budget fitting."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from contextkit.utils.token_counting import count as count_tokens
from contextkit.constants import (
    DEFAULT_ENCODING,
    DEFAULT_EXAMPLE_CANDIDATE_LIMIT,
    DEFAULT_EXAMPLE_TOP_K,
    PRIORITY_EXAMPLE,
)
from contextkit.core import BlockType, ContextBlock
from contextkit.observe.provenance import Origin
from contextkit.prompts.example import Example
from contextkit.utils.text_similarity import word_overlap_score

logger = logging.getLogger("contextkit")


class ExampleStore:
    """Manages few-shot examples with selection and budget fitting.

    Register examples and select the best ones for a given input.
    Supports budget-aware fitting to maximize examples within a
    token limit.
    """

    def __init__(self) -> None:
        self._examples: Dict[str, Example] = {}

    def add(
        self,
        example_id: str,
        input_text: str,
        output_text: str,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
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
        logger.debug("Added example '%s'", example_id)
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
        top_k: int = DEFAULT_EXAMPLE_TOP_K,
        tags: List[str] | None = None,
    ) -> List[Tuple[Example, float]]:
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
        scored: List[Tuple[float, Example]] = []

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
        top_k: int = DEFAULT_EXAMPLE_CANDIDATE_LIMIT,
        tags: List[str] | None = None,
        encoding: str = DEFAULT_ENCODING,
        priority: int = PRIORITY_EXAMPLE,
    ) -> List[ContextBlock]:
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

        blocks: List[ContextBlock] = []
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

    async def aselect(
        self,
        input_text: str,
        top_k: int = DEFAULT_EXAMPLE_TOP_K,
        tags: List[str] | None = None,
    ) -> List[Tuple[Example, float]]:
        """Async version of :meth:`select`."""
        return self.select(input_text, top_k=top_k, tags=tags)

    async def afit_to_budget(
        self,
        input_text: str,
        max_tokens: int,
        top_k: int = DEFAULT_EXAMPLE_CANDIDATE_LIMIT,
        tags: List[str] | None = None,
        encoding: str = DEFAULT_ENCODING,
        priority: int = PRIORITY_EXAMPLE,
    ) -> List[ContextBlock]:
        """Async version of :meth:`fit_to_budget`."""
        return self.fit_to_budget(
            input_text,
            max_tokens,
            top_k=top_k,
            tags=tags,
            encoding=encoding,
            priority=priority,
        )

    def list_examples(self) -> List[str]:
        """List all example IDs."""
        return sorted(self._examples.keys())

    @property
    def example_count(self) -> int:
        """Number of stored examples."""
        return len(self._examples)
