"""Centralized constants for the contextkit SDK.

All hardcoded numeric and string constants used across the codebase
are defined here for maintainability and discoverability.
"""

# ---------------------------------------------------------------------------
# Token Estimation
# ---------------------------------------------------------------------------

DEFAULT_ENCODING: str = "cl100k_base"
"""Default tiktoken encoding used for token counting."""

MESSAGE_OVERHEAD_TOKENS: int = 4
"""Approximate token overhead per chat message (role + delimiters)."""

CHARS_PER_TOKEN_ESTIMATE: int = 4
"""Fallback character-to-token ratio when tiktoken is unavailable."""

DEFAULT_TOKEN_CACHE_SIZE: int = 4096
"""Maximum entries in the LRU token-count cache."""

# ---------------------------------------------------------------------------
# Block Priority Defaults
# ---------------------------------------------------------------------------
# Higher priority = more likely to be retained during budget trimming.

MAX_PRIORITY: int = 100
"""Maximum priority value on the 0-100 scale."""

PRIORITY_SYSTEM_PROMPT: int = 100
"""Priority for system prompt blocks (highest)."""

PRIORITY_SHORT_TERM_MEMORY: int = 80
"""Priority for short-term (conversation) memory blocks."""

PRIORITY_RAG_CHUNK: int = 70
"""Priority for RAG retrieval chunks."""

PRIORITY_TOOL_OUTPUT: int = 70
"""Priority for captured tool-call outputs."""

PRIORITY_TOOL_DEFINITION: int = 60
"""Priority for tool definition / schema blocks."""

PRIORITY_FILE_CONTEXT: int = 60
"""Priority for injected file context blocks."""

PRIORITY_LONG_TERM_MEMORY: int = 60
"""Priority for long-term memory blocks."""

PRIORITY_EXAMPLE: int = 55
"""Priority for few-shot example blocks."""

PRIORITY_DEFAULT: int = 50
"""Default priority for generic context blocks."""

PRIORITY_HANDOFF_METADATA: int = 40
"""Priority for agent handoff metadata blocks."""

# ---------------------------------------------------------------------------
# Retrieval Defaults
# ---------------------------------------------------------------------------

DEFAULT_TOP_K: int = 5
"""Default number of results to return from retrieval queries."""

DEFAULT_EXAMPLE_TOP_K: int = 3
"""Default number of examples returned by ExampleStore.select."""

DEFAULT_EXAMPLE_CANDIDATE_LIMIT: int = 10
"""Default candidate pool size for ExampleStore.fit_to_budget."""

DEFAULT_IMPORTANCE: float = 0.5
"""Default importance score for memory records (0.0-1.0)."""

# ---------------------------------------------------------------------------
# Scoring Weights
# ---------------------------------------------------------------------------
# For blended relevance scoring in retrieval operations.
# Word-match contributes 70 %, record importance contributes 30 %.

WORD_MATCH_WEIGHT: float = 0.7
"""Weight applied to word-overlap score during retrieval ranking."""

IMPORTANCE_WEIGHT: float = 0.3
"""Weight applied to record importance during retrieval ranking."""

# ---------------------------------------------------------------------------
# Similarity Thresholds
# ---------------------------------------------------------------------------

DEFAULT_SIMILARITY_THRESHOLD: float = 0.8
"""Default similarity threshold for deduplication (0.0-1.0)."""

# ---------------------------------------------------------------------------
# Compaction Defaults
# ---------------------------------------------------------------------------

DEFAULT_COMPACT_TARGET_RATIO: float = 0.5
"""Default target compression ratio for the CompactStep."""

DEFAULT_COMPACT_MIN_TOKENS: int = 100
"""Minimum token count before a block qualifies for compaction."""

# ---------------------------------------------------------------------------
# Memory Decay
# ---------------------------------------------------------------------------

MEMORY_HALF_LIFE_HOURS: float = 168.0
"""Default half-life for memory decay in hours (1 week)."""

DECAY_WEIGHT: float = 0.3
"""Weight applied to temporal decay factor during retrieval ranking."""

# ---------------------------------------------------------------------------
# Query-Aware Pruning
# ---------------------------------------------------------------------------

DEFAULT_RELEVANCE_WEIGHT: float = 0.5
"""Default weight for relevance vs. priority in query-aware pruning."""

# ---------------------------------------------------------------------------
# Compaction Storage
# ---------------------------------------------------------------------------

DEFAULT_COMPACTION_STORAGE_DIR: str = ".contextkit/compacted"
"""Default directory for local compaction artifact storage."""

# ---------------------------------------------------------------------------
# Context Quality Scoring
# ---------------------------------------------------------------------------

HIGH_PRIORITY_THRESHOLD: int = 70
"""Blocks at or above this priority are considered 'high priority'."""

LOW_ATTENTION_THRESHOLD: float = 0.5
"""Attention weight below this value is considered a risky position."""

# ---------------------------------------------------------------------------
# Context Sufficiency
# ---------------------------------------------------------------------------

DEFAULT_MIN_QUERY_COVERAGE: float = 0.6
"""Minimum fraction of query terms that must appear in context."""

DEFAULT_MIN_AVG_RELEVANCE: float = 0.3
"""Minimum average relevance score across RAG blocks."""

# ---------------------------------------------------------------------------
# RAG Compression
# ---------------------------------------------------------------------------

DEFAULT_RAG_MAX_SENTENCES: int = 3
"""Default max sentences to keep per RAG block during compression."""
