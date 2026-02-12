"""Word-overlap text similarity utilities.

Provides reusable functions for calculating text similarity
using word-level set intersection. Used by the pipeline
deduplication step, RAG deduplication, example selection,
memory retrieval, and tool selection.
"""

from __future__ import annotations

import re
from typing import Set


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


def information_density(text: str) -> float:
    """Compute information density as the ratio of unique words to total words.

    Higher density means more varied vocabulary (more informative).
    Lower density means many repeated words (less informative).

    Args:
        text: The text to analyze.

    Returns:
        A density score between 0.0 and 1.0.
    """
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def extract_key_sentences(
    text: str,
    query: str,
    max_sentences: int = 3,
) -> str:
    """Extract the most query-relevant sentences from text.

    Scores each sentence by its word overlap with the query,
    then returns the top-scoring sentences in original order.

    Args:
        text: Source text to extract from.
        query: Query to score relevance against.
        max_sentences: Maximum sentences to return.

    Returns:
        A string containing the selected sentences.
    """
    sentences = _split_sentences(text)
    if len(sentences) <= max_sentences:
        return text

    scored = [
        (idx, word_overlap_score(query, sentence), sentence)
        for idx, sentence in enumerate(sentences)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = sorted(scored[:max_sentences], key=lambda x: x[0])
    return " ".join(sentence for _, _, sentence in selected)


def memory_relevance_score(
    query: str,
    content: str,
    importance: float,
    decay: float,
    word_match_weight: float,
    importance_weight: float,
    decay_weight: float,
) -> float:
    """Compute a blended relevance score for memory retrieval.

    Combines word-overlap matching, record importance, and temporal
    decay into a single score.  Used by all memory backends to ensure
    consistent ranking behaviour.

    Args:
        query: The search query string.
        content: The record content to score against.
        importance: Record importance (0.0-1.0).
        decay: Temporal decay factor (0.0-1.0).
        word_match_weight: Weight for the word-overlap component.
        importance_weight: Weight for the importance component.
        decay_weight: Weight for the decay component.

    Returns:
        A composite relevance score.
    """
    word_match = word_overlap_score(query, content)
    return (
        word_match * word_match_weight
        + importance * importance_weight
        + decay * decay_weight
    )


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on period/question/exclamation boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def _tokenize_to_word_set(text: str) -> Set[str]:
    """Split text into a set of unique lowercased words."""
    return set(text.lower().split())
