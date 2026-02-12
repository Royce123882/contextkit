# contextkit — Implementation Plan v2

> Based on full source code audit (Feb 2026). Every item verified against actual code with file paths and line numbers.

---

## Current State Summary

**What exists and works:**
- 12 BlockTypes, ContextBlock, ContextWindow with token tracking and budget enforcement
- ContextAssembler with priority-based ordering and explainable reports
- 2 provider adapters (Anthropic, OpenAI)
- Full observability: `inspect()`, `explain()`, `diff()`, `dump()`, timeline tracking
- ShortTermMemory (sliding_window, token_budget), LongTermMemory with 3 backends (InMemory, SQLite, Postgres)
- PromptManager with Jinja2, FileContext with lazy loading, ToolRegistry with dynamic selection
- RAGContext with 5 retriever backends (InMemory, Chroma, Pgvector, Pinecone, Qdrant)
- 8 pipeline steps: Trim (query-aware), Compact, Compress (IDF), Deduplicate, Filter, Reorder (prefix_stable), RAGCompress, Mask
- QualityScorer (positional), SufficiencyChecker
- Multi-agent: ContextScope, SharedMemory, Scratchpad, HandoffPackage, Timeline
- Memory decay via Ebbinghaus curve on MemoryRecord
- Thread-safe events and SharedMemory
- Postgres connection pooling
- 621+ tests, 90% coverage target

**What's broken or missing:** See phases below.

---

## Phase 1: Developer Ergonomics (v0.7.0)

**Goal:** Cut boilerplate in half. Make the common case a one-liner.

### 1.1 Origin Convenience Constructors

**File:** `src/contextkit/observe/provenance.py` (lines 16–85)

Current Origin has only the raw constructor: `Origin(source="rag", details={"retriever": "bm25", "query": "..."})`.

Add class methods:

```python
# provenance.py — add after line 85

@classmethod
def from_rag(cls, query: str, retriever: str = "", relevance_score: float = 0.0, **kwargs) -> "Origin":
    details = {"query": query, "retriever": retriever, "relevance_score": relevance_score, **kwargs}
    return cls(source="rag", details=details)

@classmethod
def from_file(cls, file_path: str, chunk_index: int = 0, **kwargs) -> "Origin":
    details = {"file_path": file_path, "chunk_index": chunk_index, **kwargs}
    return cls(source="file", details=details)

@classmethod
def from_tool(cls, tool_name: str, **kwargs) -> "Origin":
    details = {"tool_name": tool_name, **kwargs}
    return cls(source="tool", details=details)

@classmethod
def from_prompt(cls, template: str, version: str = "1", **kwargs) -> "Origin":
    details = {"template": template, "version": version, **kwargs}
    return cls(source="prompt", details=details)

@classmethod
def from_memory(cls, key: str, query: str = "", **kwargs) -> "Origin":
    details = {"key": key, "query": query, **kwargs}
    return cls(source="memory", details=details)

@classmethod
def from_conversation(cls, turn_range: str = "", **kwargs) -> "Origin":
    details = {"turn_range": turn_range, **kwargs}
    return cls(source="conversation", details=details)
```

**Tests:** `tests/test_observe.py` — add cases for each factory method, verify `source` and `details` fields.

### 1.2 ContextBlock Type Shortcuts

**File:** `src/contextkit/core/block.py` (lines 32–97)

Current ContextBlock requires `type=BlockType.RAG` every time. Add class methods:

```python
# block.py — add to ContextBlock class after __repr__

@classmethod
def system(cls, content: str, *, priority: int = PRIORITY_SYSTEM, name: str | None = None, **kwargs) -> "ContextBlock":
    return cls(type=BlockType.SYSTEM_PROMPT, content=content, priority=priority, name=name or "system_prompt", **kwargs)

@classmethod
def rag(cls, content: str, *, priority: int = PRIORITY_DEFAULT, origin: Origin | None = None, **kwargs) -> "ContextBlock":
    return cls(type=BlockType.RAG, content=content, priority=priority, origin=origin, **kwargs)

@classmethod
def memory(cls, content: str, *, priority: int = PRIORITY_DEFAULT, **kwargs) -> "ContextBlock":
    return cls(type=BlockType.SHORT_TERM_MEMORY, content=content, priority=priority, **kwargs)

@classmethod
def tool_output(cls, content: str, *, tool_name: str = "", priority: int = PRIORITY_DEFAULT, **kwargs) -> "ContextBlock":
    origin = Origin.from_tool(tool_name) if tool_name else None
    return cls(type=BlockType.TOOL_OUTPUTS, content=content, priority=priority, origin=origin, **kwargs)

@classmethod
def examples(cls, content: str | list[dict], *, priority: int = PRIORITY_DEFAULT, **kwargs) -> "ContextBlock":
    return cls(type=BlockType.EXAMPLES, content=content, priority=priority, **kwargs)

@classmethod
def file(cls, content: str, *, file_path: str = "", priority: int = PRIORITY_DEFAULT, **kwargs) -> "ContextBlock":
    origin = Origin.from_file(file_path) if file_path else None
    return cls(type=BlockType.FILES, content=content, priority=priority, origin=origin, **kwargs)
```

**Tests:** `tests/test_core.py` — verify each shortcut sets correct `type`, `priority`, and auto-populated `origin`.

### 1.3 Fluent ContextWindow API

**File:** `src/contextkit/core/context_window.py`

Change `add()` (line 138) and `add_unchecked()` (line 369) to return `self`:

```python
# context_window.py line 138
def add(self, block: ContextBlock) -> "ContextWindow":
    # ... existing logic ...
    return self

# context_window.py line 369
def add_without_budget_check(self, block: ContextBlock) -> "ContextWindow":
    # ... existing logic ...
    return self
```

Also rename `add_unchecked` → `add_without_budget_check` (line 369). Keep `add_unchecked` as a deprecated alias.

Add `will_fit()` dry-run method:

```python
# context_window.py — add after get_block()
def will_fit(self, block: ContextBlock) -> bool:
    """Check if a block would fit within the token budget without adding it."""
    return block.token_count <= self.budget_remaining
```

Add block query helpers:

```python
def blocks_of_type(self, block_type: BlockType) -> list[ContextBlock]:
    """Return all blocks matching the given type."""
    return [b for b in self._blocks if b.type == block_type]

def find_blocks(self, predicate: Callable[[ContextBlock], bool]) -> list[ContextBlock]:
    """Return all blocks matching a predicate function."""
    return [b for b in self._blocks if predicate(b)]

def clone(self) -> "ContextWindow":
    """Create a deep copy of this window with the same configuration and blocks."""
    import copy
    new_window = ContextWindow(model=self._model_name, max_tokens=self._max_tokens)
    for block in self._blocks:
        new_window.add_without_budget_check(copy.deepcopy(block))
    return new_window
```

**Tests:** `tests/test_core.py` — chaining, `will_fit()`, `blocks_of_type()`, `find_blocks()`, `clone()`.

### 1.4 Pipeline Presets

**File:** `src/contextkit/pipeline/pipeline.py` (lines 24–99)

Add class methods to `ContextPipeline`:

```python
# pipeline.py — add to ContextPipeline class

@classmethod
def balanced(cls, max_tokens: int | None = None, query: str | None = None) -> "ContextPipeline":
    """Sensible defaults: deduplicate, filter low-relevance, trim to budget, reorder for cache."""
    steps = [
        DeduplicateStep(similarity_threshold=0.85),
        FilterStep(min_relevance=0.3),
        TrimStep(max_tokens=max_tokens, query=query),
        ReorderStep(strategy="prefix_stable"),
    ]
    return cls(steps)

@classmethod
def aggressive(cls, max_tokens: int | None = None, query: str | None = None) -> "ContextPipeline":
    """Maximum compression: deduplicate, filter, compress, trim, mask old, reorder."""
    steps = [
        DeduplicateStep(similarity_threshold=0.7),
        FilterStep(min_relevance=0.5),
        CompressStep(compression_ratio=0.5),
        TrimStep(max_tokens=max_tokens, query=query),
        MaskStep(window=3),
        ReorderStep(strategy="prefix_stable"),
    ]
    return cls(steps)

@classmethod
def conservative(cls, max_tokens: int | None = None) -> "ContextPipeline":
    """Minimal changes: only deduplicate exact matches and reorder."""
    steps = [
        DeduplicateStep(similarity_threshold=0.95),
        TrimStep(max_tokens=max_tokens),
        ReorderStep(strategy="important_edges"),
    ]
    return cls(steps)
```

**Tests:** `tests/test_pipeline.py` — verify each preset creates expected steps, runs without error.

### 1.5 Expand Top-Level Exports

**File:** `src/contextkit/__init__.py` (lines 1–38)

Add to imports and `__all__`:

```python
# Adapters
from contextkit.adapters import AnthropicAdapter, OpenAIAdapter

# Pipeline
from contextkit.pipeline import ContextPipeline, TrimStep, CompactStep, CompressStep
from contextkit.pipeline import FilterStep, DeduplicateStep, ReorderStep, MaskStep, RAGCompressStep

# Scope
from contextkit.scope import ContextScope, SharedMemory, Scratchpad, HandoffPackage

# Memory
from contextkit.memory import ShortTermMemory, LongTermMemory

# Managers
from contextkit.rag import RAGContext
from contextkit.tools import ToolRegistry
from contextkit.prompts import PromptManager
from contextkit.files import FileContext

# Observe
from contextkit.observe import QualityScorer, SufficiencyChecker
```

**Tests:** `tests/test_imports.py` — new file, verify every class is importable from `contextkit` directly.

### 1.6 Improved Error Messages

**File:** `src/contextkit/core/block.py` (lines 100–117)

`BudgetExceededError` currently just shows numbers. Add actionable suggestions:

```python
def __str__(self) -> str:
    msg = (
        f"Block '{self.block_name}' needs {self.block_tokens} tokens "
        f"but only {self.budget_remaining} remain (max: {self.max_tokens}).\n"
        f"Suggestions:\n"
        f"  - Run a pipeline to free space: ContextPipeline.balanced(max_tokens={self.max_tokens})\n"
        f"  - Remove low-priority blocks: window.remove('block_name')\n"
        f"  - Use window.will_fit(block) to check before adding"
    )
    return msg
```

---

## Phase 2: Bug Fixes & Production Hardening (v0.7.1)

**Goal:** Fix confirmed bugs. No new features, just correctness.

### 2.1 Fix O(n²) Message Trimming

**Files:**
- `src/contextkit/memory/short_term.py` — lines 189, 221
- Standalone `trim_conversation()` — line 221

**Problem:** `list.pop(0)` is O(n) per call. In a while loop trimming to budget, this is O(n²).

**Fix:** Replace internal `_messages: list` with `collections.deque`. Use `popleft()` (O(1)).

```python
# short_term.py — change __init__
from collections import deque

self._messages: deque[dict[str, Any]] = deque()

# line 189 — change pop(0) to popleft()
self._messages.popleft()

# For sliding_window strategy (line 168-172):
# Replace slice assignment with: while len(self._messages) > self._max_turns: self._messages.popleft()

# Standalone trim_conversation (line 221):
# Use deque internally, convert back to list on return
```

**Backward compatibility:** The `messages` property (line 58) already returns `list(self._messages)`, so a copy is made. No API change.

**Tests:** Verify trimming produces same results. Add performance test with 10,000+ messages.

### 2.2 Path Traversal Protection in FileContext

**File:** `src/contextkit/files/file_context.py` — lines 202–234

**Problem:** `load_single(path)` reads any file path with no validation.

**Fix:** Add path validation at the start of `load_single()` and `load()`:

```python
# file_context.py — add helper method
def _validate_path(self, path: str) -> Path:
    resolved = Path(path).resolve()
    allowed = self._base_path.resolve()
    if not resolved.is_relative_to(allowed):
        raise PathTraversalError(f"Path '{path}' is outside allowed directory '{allowed}'")
    return resolved
```

Call `_validate_path()` at the top of `load_single()` (line 202) and in the load loop (line 149).

Also add `followlinks=False` to `os.walk()` in `scan()` (line 50).

**New exception:** Add `PathTraversalError` to `src/contextkit/core/block.py` or a new `src/contextkit/exceptions.py`.

**Tests:** Verify `load_single("../../etc/passwd")` raises `PathTraversalError`. Verify symlink traversal is blocked.

### 2.3 SQL-Pushed Filtering in SQLite and Postgres

**Files:**
- `src/contextkit/memory/sqlite_backend.py` — line 212
- `src/contextkit/memory/postgres_backend.py` — line 229

**Problem:** Both `retrieve()` methods call `_fetch_all_records()` and score everything in Python.

**Fix for SQLite:**
```python
# sqlite_backend.py — replace _fetch_all_records() call in retrieve()
async def retrieve(self, query: str, top_k: int = 5, tags: list[str] | None = None):
    async with self._connect() as db:
        if tags:
            # Filter in SQL using LIKE on JSON-encoded tags column
            placeholders = " AND ".join(f"tags LIKE ?" for _ in tags)
            params = [f'%"{tag}"%' for tag in tags]
            cursor = await db.execute(f"SELECT * FROM memory_records WHERE {placeholders}", params)
        else:
            cursor = await db.execute("SELECT * FROM memory_records")
        rows = await cursor.fetchall()
    # Score and rank the filtered subset
    ...
```

**Fix for Postgres:**
```python
# postgres_backend.py — push tag filtering into SQL (already partially done at line 219-226)
# Add LIMIT clause to reduce memory usage:
sql += " LIMIT $N"  # e.g., LIMIT 1000 to cap in-memory scoring
```

**Also add pagination to `list_records()`:**
```python
async def list_records(self, tags: list[str] | None = None, offset: int = 0, limit: int = 100) -> list[MemoryRecord]:
```

Update the `MemoryBackend` protocol (`memory/record.py:153-165`) to include `offset` and `limit` parameters.

**Tests:** Verify tag filtering returns correct subsets. Verify pagination returns correct windows.

### 2.4 Fix Sync-to-Async Bridge

**File:** `src/contextkit/sync.py` — line 37

**Problem:** `concurrent.futures.ThreadPoolExecutor` deadlocks when called from within an existing event loop.

**Fix:** Use `anyio` or detect the situation and raise a clear error:

```python
# sync.py — replace _run_sync()
def _run_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop running — safe to use asyncio.run()
        return asyncio.run(coro)

    # Loop is running — we're inside async code calling sync wrapper
    # Option A: Use anyio (preferred)
    import anyio
    return anyio.from_thread.run(coro)

    # Option B: Raise clear error
    raise RuntimeError(
        "Cannot use sync wrappers inside an async context. "
        "Use the async API directly (e.g., await memory.retrieve(...))."
    )
```

**New optional dependency:** `anyio>=4.0` in `pyproject.toml`.

**Tests:** Test sync wrapper works from sync context. Test it either works or raises clear error from async context.

### 2.5 Custom Exception Hierarchy

**File:** New `src/contextkit/exceptions.py`

```python
class ContextKitError(Exception):
    """Base exception for all contextkit errors."""

class BudgetExceededError(ContextKitError):
    """Raised when a block would exceed the token budget."""
    # Move existing class from block.py

class PathTraversalError(ContextKitError):
    """Raised when a file path escapes the allowed directory."""

class BackendConnectionError(ContextKitError):
    """Raised when a memory/retriever backend is unreachable."""

class BackendTimeoutError(ContextKitError):
    """Raised when a backend operation times out."""

class RecordNotFoundError(ContextKitError):
    """Raised when a memory record is not found."""

class InvalidBlockError(ContextKitError):
    """Raised when a ContextBlock fails validation."""

class TemplateRenderError(ContextKitError):
    """Raised when a prompt template cannot be rendered."""
```

Migrate `BudgetExceededError` from `block.py:100-117` to `exceptions.py`. Keep re-export from `block.py` for backward compatibility.

Replace generic exceptions throughout:
- `KeyError` in `tool_registry.py:63` → `RecordNotFoundError`
- `ValueError` in `short_term.py:44` → `InvalidBlockError`
- Broad `except Exception` in `token_counting.py:37` → `except (ImportError, KeyError, ValueError)`

**Tests:** Verify each exception type is raised in appropriate scenarios.

### 2.6 PromptManager Variable Validation

**File:** `src/contextkit/prompts/prompt_manager.py` — lines 282–299

**Problem:** `_interpolate()` silently leaves unmatched `{{var}}` in output.

**Fix:** Add a `strict` mode that warns or raises on missing variables:

```python
# prompt_manager.py — modify _interpolate()
def _interpolate(template: str, variables: dict[str, str], strict: bool = False) -> str:
    pattern = r"\{\{(\w+)\}\}"
    missing = []

    def replacer(match):
        key = match.group(1)
        if key in variables:
            return str(variables[key])
        missing.append(key)
        return match.group(0)

    result = re.sub(pattern, replacer, template)

    if missing and strict:
        raise TemplateRenderError(f"Missing template variables: {missing}")
    elif missing:
        logger.warning("Unresolved template variables: %s", missing)

    return result
```

**Tests:** Verify strict mode raises, non-strict mode logs warning.

---

## Phase 3: Advanced Context Intelligence (v0.8.0)

**Goal:** Smarter context assembly informed by academic research.

### 3.1 Extended Quality Metrics

**File:** `src/contextkit/observe/quality.py` (lines 22–127)

Current `QualityScorer` only computes positional attention via U-curve. Add:

**Signal-to-Noise Ratio:**
```python
def _compute_snr(self, blocks: list[ContextBlock]) -> float:
    """Ratio of high-priority to low-priority token mass."""
    signal_tokens = sum(b.token_count for b in blocks if b.priority >= self._high_priority_threshold)
    noise_tokens = sum(b.token_count for b in blocks if b.priority < self._high_priority_threshold)
    if noise_tokens == 0:
        return 1.0
    return signal_tokens / (signal_tokens + noise_tokens)
```

**Redundancy Detection:**
```python
def _compute_redundancy(self, blocks: list[ContextBlock]) -> float:
    """Fraction of blocks that are near-duplicates of another block."""
    from contextkit.utils.text_similarity import word_overlap_score
    redundant = 0
    for i, a in enumerate(blocks):
        for b in blocks[i+1:]:
            if isinstance(a.content, str) and isinstance(b.content, str):
                if word_overlap_score(a.content, b.content) > 0.8:
                    redundant += 1
                    break
    return redundant / max(len(blocks), 1)
```

**Information Density:**
```python
def _compute_density(self, blocks: list[ContextBlock]) -> float:
    """Unique terms per token — higher means more information-dense."""
    all_words = []
    total_tokens = 0
    for b in blocks:
        if isinstance(b.content, str):
            all_words.extend(b.content.lower().split())
            total_tokens += b.token_count
    if total_tokens == 0:
        return 0.0
    return len(set(all_words)) / total_tokens
```

Extend `QualityReport` to include `snr`, `redundancy`, `density` fields alongside existing `overall_score`.

**Tests:** Verify SNR with mixed-priority blocks. Verify redundancy detects duplicate content. Verify density varies with diverse vs repetitive content.

### 3.2 Collapse Detection in CompactStep

**File:** `src/contextkit/pipeline/compact.py` (lines 31–140)

**Problem:** Compaction can lose critical information with no detection.

Add `max_info_loss` parameter (already declared at line 52 but not enforced):

```python
# compact.py — in _compact_block(), after compaction:
def _compact_block(self, block: ContextBlock) -> ContextBlock:
    # ... existing compaction ...

    # Collapse detection: compare entity/term counts
    original_terms = set(original_content.lower().split())
    compacted_terms = set(compacted_content.lower().split())
    term_retention = len(original_terms & compacted_terms) / max(len(original_terms), 1)

    if (1.0 - term_retention) > self._max_info_loss:
        logger.warning(
            "Collapse detected in block '%s': %.0f%% term loss exceeds %.0f%% threshold",
            block.display_name, (1.0 - term_retention) * 100, self._max_info_loss * 100
        )
        # Return original block unmodified
        return block

    # ... proceed with compacted version ...
```

**Tests:** Verify collapse detection prevents over-compression. Verify it allows acceptable compression.

### 3.3 Anti-Drift Detection

**File:** New `src/contextkit/observe/drift.py`

**Research basis:** "Context Drift" (arXiv:2510.07777) — 39% average performance drop in multi-turn.

```python
class DriftDetector:
    """Detect context quality degradation across turns using timeline snapshots."""

    def __init__(self, quality_scorer: QualityScorer | None = None, drift_threshold: float = 0.15):
        self._scorer = quality_scorer or QualityScorer()
        self._threshold = drift_threshold
        self._history: list[float] = []

    def record(self, blocks: list[ContextBlock]) -> DriftReport:
        """Score current context and compare to historical trend."""
        report = self._scorer.score(blocks)
        self._history.append(report.overall_score)

        drift = 0.0
        if len(self._history) >= 3:
            recent_avg = sum(self._history[-3:]) / 3
            initial_avg = sum(self._history[:3]) / min(3, len(self._history))
            drift = initial_avg - recent_avg

        return DriftReport(
            current_score=report.overall_score,
            drift=drift,
            drifting=drift > self._threshold,
            turn_count=len(self._history),
            suggestion="Consider compacting or re-retrieving context" if drift > self._threshold else "",
        )
```

Integrate with `ContextScope.record_turn()` (`scope/context_scope.py:71-83`) — optionally run drift detection on each turn.

**Tests:** Simulate degrading context over 10 turns, verify drift is detected. Verify stable context shows no drift.

### 3.4 Conditional Pipeline Steps

**File:** `src/contextkit/pipeline/base.py` (lines 57–81)

Add `run_if` guard to `PipelineStep`:

```python
# base.py — add to PipelineStep
class PipelineStep(ABC):
    def __init__(self, run_if: Callable[[list[ContextBlock]], bool] | None = None):
        self._run_if = run_if

    def should_run(self, blocks: list[ContextBlock]) -> bool:
        if self._run_if is None:
            return True
        return self._run_if(blocks)
```

Update `ContextPipeline._execute_step()` (`pipeline.py:116`) to check `step.should_run(blocks)` before processing.

Usage:
```python
pipeline = ContextPipeline([
    CompressStep(compression_ratio=0.5, run_if=lambda blocks: sum(b.token_count for b in blocks) > 5000),
    TrimStep(max_tokens=4000),
])
```

**Tests:** Verify step is skipped when condition is false. Verify step runs when condition is true.

### 3.5 Real Async Pipeline Execution

**File:** `src/contextkit/pipeline/pipeline.py` — lines 87–99

**Problem:** `arun()` is a stub that calls the synchronous `_execute_all_steps()`.

**Fix:** Add `async_process()` to `PipelineStep` base and use it in `arun()`:

```python
# base.py — add to PipelineStep
async def async_process(self, blocks: list[ContextBlock]) -> list[ContextBlock]:
    """Async version of process(). Default delegates to sync process()."""
    return self.process(blocks)

# pipeline.py — replace arun()
async def arun(self, window: ContextWindow) -> ContextWindow:
    blocks = list(window.blocks)
    step_reports = []
    for step in self._steps:
        if not step.should_run(blocks):
            continue
        tokens_before = sum(b.token_count for b in blocks)
        blocks = await step.async_process(blocks)
        tokens_after = sum(b.token_count for b in blocks)
        step_reports.append(StepReport(step_name=step.name, tokens_before=tokens_before, tokens_after=tokens_after, tokens_saved=tokens_before - tokens_after))
    window.replace_blocks(blocks)
    self._last_report = self._build_report(step_reports)
    return window
```

**Tests:** Verify `arun()` produces same results as `run()`. Add a custom async step that yields control.

---

## Phase 4: Ecosystem Integration (v0.9.0)

**Goal:** First-class integration with the broader AI ecosystem.

### 4.1 Additional Provider Adapters

**Directory:** `src/contextkit/adapters/`

Currently: `anthropic_adapter.py`, `openai_adapter.py` only.

Add:

**Bedrock Adapter** (`bedrock_adapter.py`):
- Maps to AWS Bedrock's Converse API format
- Handles `system` as separate field (like Anthropic)
- No new dependency needed (uses dict output for boto3)

**LiteLLM Adapter** (`litellm_adapter.py`):
- Formats for LiteLLM's unified API (OpenAI-compatible)
- Delegates to OpenAI format with `model` field preserved

**Ollama Adapter** (`ollama_adapter.py`):
- Formats for Ollama's `/api/chat` endpoint
- System prompt as first message with `role: "system"`

Each adapter implements `ProviderAdapter` protocol (`adapters/base.py:16`). Pattern is well-established from existing adapters — ~100 lines each.

**Tests:** Verify each adapter produces valid output shape, handles SYSTEM_PROMPT correctly, emits WINDOW_RENDERED event.

### 4.2 MCP Integration

**Directory:** New `src/contextkit/mcp/`

**Research basis:** Anthropic's Model Context Protocol — adopted by OpenAI, Google, Microsoft. CA-MCP (arXiv:2601.11595) shows shared context stores enable autonomous server coordination.

**Files:**
```
src/contextkit/mcp/
    __init__.py
    context_provider.py    # Expose ContextWindow as MCP resources
    context_consumer.py    # Read context from MCP servers
```

**Context Provider** — expose context blocks as MCP resources:
```python
class MCPContextProvider:
    """Expose a ContextWindow's blocks as MCP-compatible resources."""

    def __init__(self, window: ContextWindow):
        self._window = window

    def list_resources(self) -> list[dict]:
        """Return MCP resource descriptors for each block."""
        return [
            {"uri": f"context://{b.display_name}", "name": b.display_name,
             "mimeType": "text/plain", "description": f"{b.type.value} block ({b.token_count} tokens)"}
            for b in self._window.blocks
        ]

    def read_resource(self, uri: str) -> str:
        """Read a single block's content by URI."""
        name = uri.replace("context://", "")
        block = self._window.get_block(name)
        if block is None:
            raise RecordNotFoundError(f"No block with name '{name}'")
        return block.content if isinstance(block.content, str) else json.dumps(block.content)
```

**Context Consumer** — pull context from MCP servers:
```python
class MCPContextConsumer:
    """Pull resources from an MCP server into ContextBlocks."""

    async def fetch(self, server_url: str, resource_uris: list[str], priority: int = 50) -> list[ContextBlock]:
        """Fetch MCP resources and convert to ContextBlocks."""
        # Uses httpx or MCP SDK to connect and read resources
        ...
```

**New optional dependency:** `mcp>=1.0` in `pyproject.toml`.

**Tests:** Verify resources are listed correctly, read correctly, and round-trip through provider → consumer.

### 4.3 RAG Feedback Loop

**File:** `src/contextkit/rag/context.py` (lines 41–160)

**Problem:** RAGContext retrieves chunks but never learns from outcomes. No way to close the quality loop.

Add feedback tracking:

```python
# context.py — add to RAGContext

def __init__(self, retriever, retriever_name="", feedback_store: dict | None = None):
    ...
    self._feedback: dict[str, list[bool]] = feedback_store if feedback_store is not None else {}

def record_feedback(self, block: ContextBlock, useful: bool) -> None:
    """Record whether a retrieved block was useful for the task."""
    key = f"{block.display_name}:{block.origin.query if block.origin else ''}"
    self._feedback.setdefault(key, []).append(useful)

def feedback_summary(self) -> dict[str, float]:
    """Return usefulness ratio per query-block pair."""
    return {k: sum(v) / len(v) for k, v in self._feedback.items() if v}
```

This is a lightweight in-memory feedback loop. Phase 5+ can persist feedback and use it to re-rank retrievals.

**Tests:** Record feedback, verify summary reflects correct ratios.

### 4.4 LangChain Bridge

**File:** New `src/contextkit/bridges/langchain_bridge.py`

```python
class ContextKitRetriever:
    """LangChain-compatible retriever backed by contextkit's RAGContext."""

    def __init__(self, rag_context: RAGContext):
        self._rag = rag_context

    async def ainvoke(self, query: str, **kwargs) -> list:
        """LangChain retriever interface."""
        blocks = await self._rag.retrieve(query, **kwargs)
        # Convert ContextBlocks to LangChain Document format
        from langchain_core.documents import Document
        return [Document(page_content=b.content if isinstance(b.content, str) else str(b.content),
                         metadata={"origin": b.origin.summary() if b.origin else "", "tokens": b.token_count})
                for b in blocks]


class ContextKitMemory:
    """LangChain-compatible memory backed by contextkit's ShortTermMemory."""

    def __init__(self, stm: ShortTermMemory):
        self._stm = stm

    def load_memory_variables(self, inputs: dict) -> dict:
        return {"history": self._stm.messages}

    def save_context(self, inputs: dict, outputs: dict) -> None:
        if "input" in inputs:
            self._stm.add_turn("user", inputs["input"])
        if "output" in outputs:
            self._stm.add_turn("assistant", outputs["output"])
```

**New optional dependency:** `langchain-core>=0.3` in `pyproject.toml`.

**Tests:** Verify retriever returns LangChain Documents. Verify memory round-trips correctly.

---

## Phase 5: Observability & Developer Tools (v1.0.0)

**Goal:** Deep visibility into context behavior.

### 5.1 Structured Logging

**File:** Replace `src/contextkit/logging.py` internals

Current: Basic `logging.getLogger("contextkit")`.

Add optional `structlog` support:

```python
def configure_logging(format: str = "text", level: str = "INFO"):
    """Configure contextkit logging. format: 'text' or 'json'."""
    if format == "json":
        import structlog
        structlog.configure(
            processors=[structlog.processors.JSONRenderer()],
            logger_factory=structlog.stdlib.LoggerFactory(),
        )
    # ... existing text logging setup ...
```

No breaking changes — defaults to existing text logging. JSON mode is opt-in.

**New optional dependency:** `structlog>=24.0` in `pyproject.toml`.

### 5.2 OpenTelemetry Integration

**File:** New `src/contextkit/observe/telemetry.py`

```python
class OTelExporter:
    """Export context metrics to OpenTelemetry-compatible backends."""

    def __init__(self):
        from opentelemetry import metrics
        self._meter = metrics.get_meter("contextkit")
        self._token_counter = self._meter.create_counter("contextkit.tokens.total")
        self._block_counter = self._meter.create_counter("contextkit.blocks.added")
        self._budget_gauge = self._meter.create_up_down_counter("contextkit.budget.remaining")
        self._retrieval_histogram = self._meter.create_histogram("contextkit.retrieval.latency_ms")

    def attach(self) -> None:
        """Register event handlers to automatically export metrics."""
        from contextkit.observe.events import register_handler, ContextEvent
        register_handler(ContextEvent.BLOCK_ADDED, self._on_block_added)
        register_handler(ContextEvent.WINDOW_RENDERED, self._on_rendered)
        # ... etc
```

**New optional dependency:** `opentelemetry-api>=1.20` in `pyproject.toml`.

### 5.3 Pipeline Replay with Intermediate States

**File:** `src/contextkit/pipeline/pipeline.py` — modify `_execute_step()` (line 116)

Add optional snapshot capture:

```python
class ContextPipeline:
    def __init__(self, steps, capture_snapshots: bool = False):
        ...
        self._snapshots: list[list[ContextBlock]] = []

    def _execute_step(self, step, blocks):
        if self._capture_snapshots:
            import copy
            self._snapshots.append(copy.deepcopy(blocks))
        result = step.process(blocks)
        return result

    @property
    def snapshots(self) -> list[list[ContextBlock]]:
        """Intermediate block states between pipeline steps. Only populated if capture_snapshots=True."""
        return self._snapshots
```

**Tests:** Run pipeline with `capture_snapshots=True`, verify each snapshot matches the state before the corresponding step.

### 5.4 Context Linting

**File:** New `src/contextkit/observe/linter.py`

Static analysis to detect common anti-patterns:

```python
class ContextLinter:
    """Detect common context assembly anti-patterns."""

    def lint(self, window: ContextWindow) -> list[LintWarning]:
        warnings = []
        warnings.extend(self._check_missing_system_prompt(window))
        warnings.extend(self._check_lost_in_middle(window))
        warnings.extend(self._check_oversized_blocks(window))
        warnings.extend(self._check_redundant_blocks(window))
        warnings.extend(self._check_empty_blocks(window))
        return warnings

    def _check_missing_system_prompt(self, window):
        if not window.blocks_of_type(BlockType.SYSTEM_PROMPT):
            return [LintWarning("no_system_prompt", "No SYSTEM_PROMPT block found. Most models perform better with explicit instructions.")]
        return []

    def _check_lost_in_middle(self, window):
        """Warn if high-priority blocks are in the middle third of the context."""
        blocks = window.blocks
        n = len(blocks)
        if n < 6:
            return []
        middle_start, middle_end = n // 3, 2 * n // 3
        middle_high = [b for b in blocks[middle_start:middle_end] if b.priority >= 80]
        if middle_high:
            names = [b.display_name for b in middle_high]
            return [LintWarning("lost_in_middle", f"High-priority blocks {names} are in the middle third. Models attend less to middle positions. Consider using ReorderStep.")]
        return []

    def _check_oversized_blocks(self, window):
        """Warn if any single block uses >50% of the budget."""
        for b in window.blocks:
            if b.token_count > window.max_tokens * 0.5:
                return [LintWarning("oversized_block", f"Block '{b.display_name}' uses {b.token_count}/{window.max_tokens} tokens (>{50}% of budget). Consider splitting or compressing.")]
        return []

    def _check_redundant_blocks(self, window):
        """Warn if blocks have high content overlap."""
        from contextkit.utils.text_similarity import word_overlap_score
        seen = []
        warnings = []
        for b in window.blocks:
            if not isinstance(b.content, str):
                continue
            for s in seen:
                if word_overlap_score(b.content, s.content) > 0.8:
                    warnings.append(LintWarning("redundant_blocks", f"Blocks '{b.display_name}' and '{s.display_name}' have >80% overlap. Consider deduplication."))
                    break
            seen.append(b)
        return warnings

    def _check_empty_blocks(self, window):
        """Warn if any block has empty content."""
        return [LintWarning("empty_block", f"Block '{b.display_name}' has empty content.")
                for b in window.blocks
                if (isinstance(b.content, str) and not b.content.strip()) or (isinstance(b.content, list) and not b.content)]
```

Add `window.lint()` convenience method on ContextWindow.

**Tests:** Create windows with each anti-pattern, verify correct warnings are produced.

---

## Phase 6: Multi-Agent Hardening (v1.1.0)

**Goal:** Make multi-agent context sharing production-safe.

### 6.1 SharedMemory ACLs

**File:** `src/contextkit/scope/shared_memory.py` (lines 18–77)

Current: Any agent can read/write/delete any block. No access control.

```python
class SharedMemory:
    def __init__(self, acl: dict[str, set[str]] | None = None):
        """
        acl: Optional access control list mapping block names to sets of agent names
             that can read them. If None, all blocks are readable by all agents.
        """
        self._acl = acl
        ...

    def publish(self, name: str, block: ContextBlock, readable_by: set[str] | None = None) -> None:
        """Publish a block, optionally restricting which agents can read it."""
        with self._lock:
            self._blocks[name] = block
            if readable_by is not None:
                if self._acl is None:
                    self._acl = {}
                self._acl[name] = readable_by

    def read(self, name: str, agent_name: str = "") -> ContextBlock | None:
        """Read a block. If ACLs are set, checks agent has access."""
        with self._lock:
            if name not in self._blocks:
                return None
            if self._acl and name in self._acl and agent_name not in self._acl[name]:
                logger.warning("Agent '%s' denied access to shared block '%s'", agent_name, name)
                return None
            return self._blocks[name]
```

**Backward compatible:** ACLs are optional. Existing code with no ACLs works unchanged.

### 6.2 SharedMemory Pub/Sub

**File:** `src/contextkit/scope/shared_memory.py`

Add event notification so agents don't need to poll:

```python
class SharedMemory:
    def __init__(self, ...):
        ...
        self._subscribers: dict[str, list[Callable]] = {}  # block_name -> callbacks

    def subscribe(self, name: str, callback: Callable[[str, ContextBlock], None]) -> None:
        """Subscribe to changes on a specific block name."""
        self._subscribers.setdefault(name, []).append(callback)

    def subscribe_all(self, callback: Callable[[str, ContextBlock], None]) -> None:
        """Subscribe to all block changes."""
        self._subscribers.setdefault("*", []).append(callback)

    def publish(self, name: str, block: ContextBlock, ...) -> None:
        with self._lock:
            self._blocks[name] = block
            ...
        # Notify subscribers outside the lock
        for cb in self._subscribers.get(name, []) + self._subscribers.get("*", []):
            cb(name, block)
```

### 6.3 Handoff Validation

**File:** `src/contextkit/scope/handoff_package.py` (lines 20–37)

Add Pydantic validation:

```python
class HandoffPackage:
    def __init__(self, source_agent, target_agent, blocks=None, scratchpad=None, metadata=None):
        if not source_agent or not source_agent.strip():
            raise InvalidBlockError("source_agent must be a non-empty string")
        if not target_agent or not target_agent.strip():
            raise InvalidBlockError("target_agent must be a non-empty string")
        if source_agent == target_agent:
            raise InvalidBlockError("Cannot handoff to self")
        ...
```

### 6.4 SharedMemory Eviction Policy

**File:** `src/contextkit/scope/shared_memory.py`

Add `max_blocks` and LRU eviction:

```python
class SharedMemory:
    def __init__(self, max_blocks: int | None = None, ...):
        self._max_blocks = max_blocks
        self._access_order: list[str] = []  # LRU tracking

    def publish(self, name: str, block: ContextBlock, ...) -> None:
        with self._lock:
            self._blocks[name] = block
            # Track access order
            if name in self._access_order:
                self._access_order.remove(name)
            self._access_order.append(name)
            # Evict LRU if over limit
            while self._max_blocks and len(self._blocks) > self._max_blocks:
                evict_name = self._access_order.pop(0)
                del self._blocks[evict_name]
                logger.info("Evicted block '%s' from shared memory (LRU)", evict_name)
```

---

## Phase 7: Self-Improving Context (v2.0.0)

**Goal:** Research-frontier features. Contexts that optimize themselves.

### 7.1 Self-Route Hybrid RAG

**Research:** arXiv:2407.16833 — auto-route queries to RAG vs full-context based on complexity.

**File:** New `src/contextkit/rag/self_route.py`

A router that estimates query complexity and decides whether to use RAG retrieval or include full documents. Uses the existing `SufficiencyChecker` to gauge whether retrieval is enough.

### 7.2 Prompt Optimization Loop

**Research:** OPRO (arXiv:2309.03409) — LLMs as prompt optimizers.

**File:** New `src/contextkit/prompts/optimizer.py`

Evolutionary optimization: generate prompt variants via `PromptManager`, score against an eval function, mutate and select. Uses existing prompt versioning system.

### 7.3 Agentic Context Engineering (ACE)

**Research:** arXiv:2510.04618 — contexts as evolving playbooks.

**File:** New `src/contextkit/ace/`

Contexts accumulate successful strategies over time. After each agent run, analyze which context patterns led to success and persist them for future runs.

### 7.4 Context Window Effectiveness Tracking

**Research:** arXiv:2509.21361 — effective context is drastically different from advertised.

**File:** Extend `src/contextkit/observe/quality.py`

Measure actual model attention vs theoretical max. Track what fraction of the context window is actually "effective" for the task.

---

## Timeline Summary

| Phase | Version | Focus | Key Deliverables | Effort |
|---|---|---|---|---|
| 1 | v0.7.0 | Developer Ergonomics | Origin factories, block shortcuts, fluent API, pipeline presets, expanded exports | 1–2 weeks |
| 2 | v0.7.1 | Bug Fixes & Hardening | O(n²) fix, path traversal fix, SQL optimization, sync bridge fix, exceptions | 1–2 weeks |
| 3 | v0.8.0 | Context Intelligence | Extended quality metrics, collapse detection, anti-drift, conditional steps, async pipeline | 2–3 weeks |
| 4 | v0.9.0 | Ecosystem Integration | Bedrock/LiteLLM/Ollama adapters, MCP integration, RAG feedback, LangChain bridge | 3–4 weeks |
| 5 | v1.0.0 | Observability & Tools | Structured logging, OpenTelemetry, pipeline replay, context linter | 2–3 weeks |
| 6 | v1.1.0 | Multi-Agent Hardening | SharedMemory ACLs, pub/sub, handoff validation, eviction | 2–3 weeks |
| 7 | v2.0.0 | Self-Improving Context | Self-route, prompt optimization, ACE, effectiveness tracking | 4–6 weeks |

---

## Dependency Graph

```
Phase 1 (Ergonomics)
  |
  v
Phase 2 (Bug Fixes) -----> can start in parallel with Phase 1
  |
  v
Phase 3 (Intelligence) --> depends on Phase 2 fixes
  |
  +------+------+
  |             |
  v             v
Phase 4       Phase 5 ----> can run in parallel
(Ecosystem)   (Observability)
  |             |
  +------+------+
         |
         v
Phase 6 (Multi-Agent) ----> depends on Phase 4 MCP + Phase 5 telemetry
         |
         v
Phase 7 (Self-Improving) -> depends on all prior phases
```

---

*Plan compiled February 2026. All file paths and line numbers verified against current source.*
