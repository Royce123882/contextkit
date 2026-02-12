"""Data models for context sufficiency checking.

Separates data models from the checking implementation
to follow the Single Responsibility Principle.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel


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

    sufficient: bool
    confidence: float
    query_coverage: float
    avg_relevance: float
    source_types: List[str]
    suggestions: List[str]
