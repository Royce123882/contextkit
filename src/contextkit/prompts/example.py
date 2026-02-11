"""Few-shot example data model."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


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
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def formatted(self) -> str:
        """Formatted example as input/output pair."""
        return f"Input: {self.input_text}\nOutput: {self.output_text}"
