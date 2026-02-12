# Implementation Plan: High & Medium Priority Features

## High Priority (v0.8)

### H1. Query-Aware Context Pruning

**Goal**: TrimStep and ShortTermMemory score blocks by relevance to the current query, not just priority/recency.

**Files to change**:
- `src/contextkit/pipeline/trim.py` — add optional `query` + `score_fn` params
- `src/contextkit/memory/short_term.py` — add `query` param to `_trim_token_budget()`
- `src/contextkit/utils/text_similarity.py` — already has `word_overlap_score()`, reuse it

**TrimStep changes**:
```python
class TrimStep(PipelineStep):
    def __init__(
        self,
        max_tokens: int | None = None,
        min_priority: int = 0,
        query: str | None = None,                          # NEW
        relevance_weight: float = 0.5,                     # NEW
        score_fn: Callable[[str, str], float] | None = None,  # NEW
    ) -> None:
```

When `query` is set, `_enforce_budget()` sorts by a blended score:
```python
blended = (1 - relevance_weight) * normalized_priority + relevance_weight * relevance
```
where `relevance = score_fn(query, block.content)` (defaults to `word_overlap_score`). This means a low-priority block that is highly relevant to the query survives over a high-priority but irrelevant block.

Mutation detail records: `"blended_score=0.72 (priority=0.60, relevance=0.84)"`.

**ShortTermMemory changes**:
```python
def trim_to_query(self, query: str, max_tokens: int | None = None) -> None:
    """Score each turn for relevance to query, drop lowest-scoring first."""
```
Adds a third strategy `"query_aware"` alongside `sliding_window` and `token_budget`. Uses `word_overlap_score(query, turn_content)` to rank turns, then trims from the least relevant until budget fits.

**Tests**: 5 new tests in `test_pipeline.py` and `test_memory.py`.

---

### H2. Prefix-Aware Context Structuring

**Goal**: Reorder blocks so stable content (system prompt, tool defs) comes first as a prefix, maximizing provider cache hits.

**Files to change**:
- `src/contextkit/pipeline/reorder.py` — add `"prefix_stable"` strategy

**ReorderStep changes**:
```python
def _reorder_prefix_stable(self, blocks: List[ContextBlock]) -> List[ContextBlock]:
    """Place stable block types first (prefix), dynamic content after."""
    STABLE_TYPES = {
        BlockType.SYSTEM_PROMPT,
        BlockType.TOOL_DEFINITIONS,
        BlockType.OUTPUT_SCHEMAS,
        BlockType.EXAMPLES,
    }
    stable = [b for b in blocks if b.type in STABLE_TYPES]
    dynamic = [b for b in blocks if b.type not in STABLE_TYPES]
    # Sort stable by priority descending (system prompt first)
    stable.sort(key=lambda b: b.priority, reverse=True)
    # Sort dynamic by important_edges within the dynamic section
    dynamic = self._reorder_edges(dynamic) if len(dynamic) > 2 else dynamic
    result = stable + dynamic
    # Record mutations for moved blocks
    ...
    return result
```

Usage: `ReorderStep(strategy="prefix_stable")` — enables automatic prefix caching optimization for Anthropic/Google.

Mutation detail: `"moved to prefix region (stable block type: system_prompt)"`.

**Tests**: 3 new tests in `test_pipeline.py`.

---

### H3. Post-Retrieval RAG Compression

**Goal**: A pipeline step that compresses RAG chunks before they enter the context window, using extractive or abstractive compression.

**Files to create**:
- `src/contextkit/pipeline/rag_compress.py`

**Design**:
```python
class RAGCompressStep(PipelineStep):
    """Compress RAG blocks using extractive sentence selection.

    Only targets blocks with type=BlockType.RAG. Other blocks pass through.
    Implements selective augmentation: if a RAG block's relevance score
    (from Origin) is below drop_threshold, it is removed entirely.
    """
    def __init__(
        self,
        compressor: Callable[[str], str] | None = None,
        max_sentences: int = 3,
        drop_threshold: float = 0.0,
    ) -> None:
```

Default compressor: extractive — splits content into sentences, scores each by word overlap with the query (from `block.origin.details["query"]`), keeps top `max_sentences`.

Selective augmentation: blocks with `origin.details["relevance_score"] < drop_threshold` are removed entirely with mutation `"dropped: relevance 0.12 below threshold 0.30"`.

**Pipeline `__init__`** updated to export `RAGCompressStep`.

**Tests**: 5 new tests.

---

## Medium Priority (v0.9)

### M1. Token-Level Prompt Compression (CompressStep)

**Goal**: A pipeline step that prunes low-information tokens using a scoring function, distinct from CompactStep's summarization approach.

**Files to create**:
- `src/contextkit/pipeline/compress.py`

**Design**:
```python
class CompressStep(PipelineStep):
    """Prune low-information tokens from block content.

    Uses a scorer function to rank each token by information value,
    then removes the lowest-scoring tokens to hit the target ratio.
    Default scorer uses inverse document frequency (IDF) approximation
    via word frequency in the block itself.
    """
    def __init__(
        self,
        compression_ratio: float = 0.5,
        scorer: Callable[[List[str]], List[float]] | None = None,
        min_tokens: int = DEFAULT_COMPACT_MIN_TOKENS,
    ) -> None:
```

Default scorer: for each token (whitespace-split word), score = `1.0 / (1.0 + count_in_block)`. High-frequency words (the, is, a) score low and get pruned first. Tokens are pruned from the middle outward to preserve start/end structure (inspired by Lost in the Middle).

Output is the remaining tokens joined by spaces. Records mutation with before/after token counts and compression ratio achieved.

Optional integration: if `llmlingua` or `transformers` is installed, provide a `perplexity_scorer()` factory that uses a small model for genuine perplexity-based scoring.

**Tests**: 4 new tests.

---

### M2. Context Quality Scoring

**Goal**: Score the assembled context window for "findability" of key information, warning when important blocks are in low-attention positions.

**Files to create**:
- `src/contextkit/observe/quality.py`

**Design**:
```python
class QualityScorer:
    """Score a context window for positional quality.

    Based on the "Lost in the Middle" U-curve: blocks at start/end
    of the sequence have high attention, blocks in the middle are
    at risk of being missed by the model.
    """
    def score(self, blocks: List[ContextBlock]) -> QualityReport:
        """Return a quality score (0.0-1.0) and per-block position scores."""

    def suggest_reorder(self, blocks: List[ContextBlock]) -> List[int]:
        """Return suggested indices for optimal block placement."""

class QualityReport(BaseModel):
    overall_score: float          # 0.0-1.0 (1.0 = all important blocks well-placed)
    positional_scores: List[PositionScore]  # per-block
    warnings: List[str]           # e.g. "Block 'rag_doc3' (priority=90) is at position 5/10 (middle)"

class PositionScore(BaseModel):
    block_name: str
    position: int
    total_blocks: int
    attention_weight: float       # U-curve weight for this position
    priority: int
    risk: str                     # "low", "medium", "high"
```

**Attention weight formula** (U-curve):
```python
def _u_curve_weight(position: int, total: int) -> float:
    """Return attention weight [0,1] based on position. U-shaped."""
    if total <= 2:
        return 1.0
    normalized = position / (total - 1)  # 0.0 to 1.0
    # U-curve: high at 0 and 1, low at 0.5
    return 1.0 - 0.6 * math.sin(math.pi * normalized)
```

Risk = "high" if block.priority >= 70 and attention_weight < 0.5 (important block in low-attention zone).

Integrates with `ContextWindow.explain()` by appending quality warnings.

**Tests**: 5 new tests.

---

### M3. Context Sufficiency Checking

**Goal**: Estimate whether the assembled context is sufficient for the query before sending to the model.

**Files to create**:
- `src/contextkit/observe/sufficiency.py`

**Design**:
```python
class SufficiencyChecker:
    """Estimate whether context covers the query adequately.

    Uses heuristic signals:
    1. Query term coverage: fraction of query words found in context
    2. Relevance distribution: are RAG chunks actually relevant?
    3. Block diversity: are multiple source types represented?
    """
    def __init__(
        self,
        min_coverage: float = 0.6,
        min_avg_relevance: float = 0.3,
    ) -> None:

    def check(
        self, query: str, blocks: List[ContextBlock]
    ) -> SufficiencyResult:

class SufficiencyResult(BaseModel):
    sufficient: bool
    confidence: float             # 0.0-1.0
    query_coverage: float         # fraction of query terms in context
    avg_relevance: float          # mean relevance of RAG blocks
    source_types: List[str]       # distinct BlockTypes present
    suggestions: List[str]        # e.g. "Consider retrieving more documents"
```

New event: add `CONTEXT_INSUFFICIENT = "context_insufficient"` to `ContextEvent` enum. Emitted when `sufficient=False`.

**Tests**: 4 new tests.

---

### M4. Memory Decay (Ebbinghaus Forgetting Curve)

**Goal**: Memories accessed less frequently and further in the past have lower retrieval priority.

**Files to change**:
- `src/contextkit/memory/record.py` — add `access_count`, `last_accessed` fields
- `src/contextkit/memory/in_memory_backend.py` — update scoring in `retrieve()`
- `src/contextkit/memory/long_term.py` — expose decay configuration

**MemoryRecord changes**:
```python
class MemoryRecord(BaseModel):
    key: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    stored_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    importance: float = DEFAULT_IMPORTANCE
    access_count: int = 0                                           # NEW
    last_accessed: datetime | None = None                           # NEW

    def decay_factor(self, half_life_hours: float = 168.0) -> float:  # NEW
        """Compute temporal decay factor (0.0-1.0).

        Uses exponential decay: factor = 2^(-t / half_life)
        where t = hours since last_accessed (or stored_at if never accessed).
        Default half_life of 168h (1 week).
        """
        reference = self.last_accessed or self.stored_at
        elapsed = (datetime.now(timezone.utc) - reference).total_seconds() / 3600
        return 2.0 ** (-elapsed / half_life_hours)
```

**InMemoryBackend scoring change**:
```python
# Current: final_score = word_match * 0.7 + importance * 0.3
# New:     final_score = word_match * 0.5 + importance * 0.2 + decay * 0.3
```

Weight constants added to `constants.py`:
```python
DECAY_WEIGHT: float = 0.3
MEMORY_HALF_LIFE_HOURS: float = 168.0
```

**Backend auto-updates**: `retrieve()` increments `access_count` and sets `last_accessed` on returned records.

**Tests**: 5 new tests in `test_memory.py`.

---

### M5. Collapse Detection for Summaries

**Goal**: Detect when CompactStep summaries lose too much information compared to the original.

**Files to change**:
- `src/contextkit/pipeline/compact.py` — add collapse detection
- `src/contextkit/utils/text_similarity.py` — add `information_density()` utility

**New utility**:
```python
def information_density(text: str) -> float:
    """Compute information density as unique-word-to-total-word ratio."""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)
```

**CompactStep changes**:
```python
class CompactStep(PipelineStep):
    def __init__(
        self,
        compactor: Callable[[str], str] | None = None,
        target_ratio: float = DEFAULT_COMPACT_TARGET_RATIO,
        min_tokens: int = DEFAULT_COMPACT_MIN_TOKENS,
        max_info_loss: float = 0.5,                        # NEW
    ) -> None:
```

After compaction, compute:
```python
density_before = information_density(before_content)
density_after = information_density(compacted)
keyword_retention = word_overlap_score(before_content, compacted)

# Collapse detected if keyword retention is too low
if keyword_retention < (1.0 - max_info_loss):
    # Log warning, record in mutation detail
    mutation.detail += f" [COLLAPSE WARNING: {keyword_retention:.0%} keyword retention]"
    logger.warning(
        "Possible context collapse in '%s': only %.0f%% keyword retention",
        block.display_name, keyword_retention * 100,
    )
```

This doesn't block compaction (that would be surprising) but records a warning so users can detect quality degradation in their pipelines.

**Tests**: 3 new tests.

---

## Implementation Order

```
Phase 1 (v0.8 sprint):
  H1. Query-aware pruning        — extends existing TrimStep + ShortTermMemory
  H2. Prefix-stable reorder      — extends existing ReorderStep
  H3. RAG compress step          — new file, integrates with existing pipeline

Phase 2 (v0.9 sprint):
  M4. Memory decay               — extends MemoryRecord + InMemoryBackend
  M5. Collapse detection         — extends CompactStep
  M2. Context quality scoring    — new file in observe/
  M3. Sufficiency checking       — new file in observe/
  M1. Token-level compression    — new file, lower priority
```

## Dependency Graph

```
H1 (query pruning) ← depends on nothing, uses existing text_similarity
H2 (prefix reorder) ← depends on nothing, extends ReorderStep
H3 (RAG compress) ← depends on nothing, new PipelineStep

M4 (memory decay) ← depends on nothing, extends MemoryRecord
M5 (collapse detect) ← uses text_similarity.information_density (new util)
M2 (quality scorer) ← depends on nothing, standalone observer
M3 (sufficiency) ← uses text_similarity.word_overlap_score
M1 (token compress) ← depends on nothing, new PipelineStep
```

No circular dependencies. All features are independently implementable.
