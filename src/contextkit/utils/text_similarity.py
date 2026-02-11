"""Word-overlap text similarity utilities.

Provides reusable functions for calculating text similarity
using word-level set intersection. Used by the pipeline
deduplication step, RAG deduplication, example selection,
memory retrieval, and tool selection.
"""

from __future__ import annotations


def word_overlap_similarity(text_a: str, text_b: str) -> float:
    """Calculate symmetric word-overlap similarity between two strings.

    Computes the Jaccard-like overlap of lowercased word sets,
    normalized by the size of the smaller set. A score of 1.0
    means the smaller text is fully contained in the larger.

    Args:
        text_a: First text string.
        text_b: Second text string.

    Returns:
        A similarity score between 0.0 and 1.0.
    """
    words_a = _tokenize_to_word_set(text_a)
    words_b = _tokenize_to_word_set(text_b)

    if not words_a or not words_b:
        return 0.0

    overlap = len(words_a & words_b)
    return overlap / min(len(words_a), len(words_b))


def word_overlap_score(query: str, target: str) -> float:
    """Calculate asymmetric query-to-target relevance using word overlap.

    Computes the fraction of query words that appear in the target.
    Suitable for search/retrieval scoring where query coverage
    matters more than target coverage.

    Args:
        query: The search query string.
        target: The target content string.

    Returns:
        A relevance score between 0.0 and 1.0.
    """
    query_words = _tokenize_to_word_set(query)
    target_words = _tokenize_to_word_set(target)

    if not query_words:
        return 0.0

    overlap = len(query_words & target_words)
    return overlap / len(query_words)


def _tokenize_to_word_set(text: str) -> set[str]:
    """Split text into a set of unique lowercased words."""
    return set(text.lower().split())
