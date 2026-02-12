"""Data models for context sufficiency checking.

Separates data models from the checking implementation
to follow the Single Responsibility Principle.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class SufficiencyResult(BaseModel):
    """Result of a context sufficiency check.

    Attributes:
        sufficient: Whether the context is likely sufficient
            for the given query.
        confidence: Confidence in the sufficiency assessment
            (0.0-1.0).
        query_coverage: Fraction of query terms found in the
            context blocks (0.0-1.0).
        avg_relevance: Mean relevance score across RAG blocks
            (0.0-1.0).
        source_types: Distinct block types present in the context.
        suggestions: Actionable recommendations for improving
            context coverage.
    """

    sufficient: bool = Field(
        description="Whether the context is likely sufficient for the given query."
    )
    confidence: float = Field(
        description="Confidence in the sufficiency assessment (0.0-1.0)."
    )
    query_coverage: float = Field(
        description="Fraction of query terms found in the context blocks (0.0-1.0)."
    )
    avg_relevance: float = Field(
        description="Mean relevance score across RAG blocks (0.0-1.0)."
    )
    source_types: List[str] = Field(
        description="Distinct block types present in the context."
    )
    suggestions: List[str] = Field(
        description="Actionable recommendations for improving context coverage."
    )
