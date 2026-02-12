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

    example_id: str = Field(description="Unique identifier.")
    input_text: str = Field(description="The example input.")
    output_text: str = Field(description="The expected output.")
    tags: List[str] = Field(default_factory=list, description="Categorization tags.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata.")

    @property
    def formatted(self) -> str:
        """Formatted example as input/output pair."""
        return f"Input: {self.input_text}\nOutput: {self.output_text}"
