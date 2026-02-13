"""Context sufficiency checking.

Estimates whether the assembled context is sufficient to answer
a query before sending to the model.  Uses heuristic signals:
query term coverage, RAG relevance distribution, and block-type
diversity.

Research basis: Google Research, "Sufficient Context" (ICLR 2025)
-- quantifies whether context is "enough" to answer correctly and
recommends a sufficiency check before generation.
"""

from __future__ import annotations

import logging
from typing import List

from contextkit.constants import DEFAULT_MIN_AVG_RELEVANCE, DEFAULT_MIN_QUERY_COVERAGE
from contextkit.core import ContextBlock
from contextkit.core.block import BlockType
from contextkit.observe.event_models import ContextEvent, EventData
from contextkit.observe.events import emit
from contextkit.observe.sufficiency_models import SufficiencyResult
from contextkit.utils.text_similarity import word_overlap_score

logger = logging.getLogger("contextkit")


class SufficiencyChecker:
    """Estimate whether assembled context adequately covers a query.

    Combines three signals into a confidence score:

    1. **Query term coverage** -- fraction of query words that
       appear somewhere in the context.
    2. **RAG relevance** -- mean ``relevance_score`` from RAG block
       origins.
    3. **Source diversity** -- whether multiple block types contribute.

    Args:
        min_query_coverage: Minimum query-term coverage to be
            considered sufficient.
        min_avg_relevance: Minimum average RAG relevance to be
            considered sufficient.
    """

    def __init__(
        self,
        min_query_coverage: float = DEFAULT_MIN_QUERY_COVERAGE,
        min_avg_relevance: float = DEFAULT_MIN_AVG_RELEVANCE,
    ) -> None:
        self._min_coverage = min_query_coverage
        self._min_relevance = min_avg_relevance

    def check(
        self,
        query: str,
        blocks: List[ContextBlock],
    ) -> SufficiencyResult:
        """Run a sufficiency check for *query* against *blocks*.

        Args:
            query: The user query to evaluate coverage for.
            blocks: The assembled context blocks.

        Returns:
            A :class:`SufficiencyResult` with assessment and suggestions.
        """
        coverage = self._compute_query_coverage(query, blocks)
        avg_relevance = self._compute_avg_relevance(blocks)
        source_types = self._distinct_source_types(blocks)
        suggestions = self._generate_suggestions(
            coverage, avg_relevance, source_types
        )

        confidence = self._compute_confidence(
            coverage, avg_relevance, len(source_types)
        )
        # avg_relevance == 0.0 means no RAG blocks are present, so the
        # relevance gate is skipped — non-RAG contexts should not fail
        # the sufficiency check due to a missing signal.
        sufficient = (
            coverage >= self._min_coverage
            and (avg_relevance >= self._min_relevance or avg_relevance == 0.0)
        )

        result = SufficiencyResult(
            sufficient=sufficient,
            confidence=round(confidence, 3),
            query_coverage=round(coverage, 3),
            avg_relevance=round(avg_relevance, 3),
            source_types=source_types,
            suggestions=suggestions,
        )

        if not sufficient:
            self._emit_insufficient_event(query, result)

        return result

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_query_coverage(
        query: str, blocks: List[ContextBlock]
    ) -> float:
        """Fraction of query words found across all block content."""
        combined_content = " ".join(
            block.content
            for block in blocks
            if isinstance(block.content, str)
        )
        return word_overlap_score(query, combined_content)

    @staticmethod
    def _compute_avg_relevance(blocks: List[ContextBlock]) -> float:
        """Mean relevance score across RAG blocks."""
        rag_scores: List[float] = []
        for block in blocks:
            if block.type != BlockType.RAG:
                continue
            if block.origin is None:
                continue
            score = block.origin.details.get("relevance_score")
            if score is not None:
                rag_scores.append(float(score))
        if not rag_scores:
            return 0.0
        return sum(rag_scores) / len(rag_scores)

    @staticmethod
    def _distinct_source_types(blocks: List[ContextBlock]) -> List[str]:
        """Sorted list of distinct block types present."""
        return sorted({block.type.value for block in blocks})

    # ------------------------------------------------------------------
    # Confidence and suggestions
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(
        coverage: float, avg_relevance: float, type_count: int
    ) -> float:
        """Blend signals into a single confidence score (0-1)."""
        coverage_signal = min(coverage, 1.0)
        relevance_signal = min(avg_relevance, 1.0) if avg_relevance > 0 else 0.5
        diversity_signal = min(type_count / 3.0, 1.0)
        return 0.5 * coverage_signal + 0.3 * relevance_signal + 0.2 * diversity_signal

    def _generate_suggestions(
        self,
        coverage: float,
        avg_relevance: float,
        source_types: List[str],
    ) -> List[str]:
        """Generate actionable suggestions based on signal gaps."""
        suggestions: List[str] = []

        if coverage < self._min_coverage:
            suggestions.append(
                f"Query coverage is {coverage:.0%} (target: {self._min_coverage:.0%}). "
                "Consider retrieving more documents or broadening the search query."
            )
        if avg_relevance > 0 and avg_relevance < self._min_relevance:
            suggestions.append(
                f"Average RAG relevance is {avg_relevance:.2f} "
                f"(target: {self._min_relevance:.2f}). "
                "Consider re-ranking or using a better retriever."
            )
        if len(source_types) <= 1:
            suggestions.append(
                "Context has limited source diversity. "
                "Consider adding memory, examples, or tool outputs."
            )

        return suggestions

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_insufficient_event(
        query: str, result: SufficiencyResult
    ) -> None:
        """Emit a CONTEXT_INSUFFICIENT event."""
        emit(
            EventData(
                event=ContextEvent.CONTEXT_INSUFFICIENT,
                details={
                    "query": query,
                    "confidence": result.confidence,
                    "query_coverage": result.query_coverage,
                    "suggestions": result.suggestions,
                },
            )
        )
