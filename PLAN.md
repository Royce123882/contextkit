# contextkit -- Execution Plan

Step-by-step build order for Phase 0 (v0.1.0). Each step produces working, tested code that builds on the previous step. Later phases are outlined at the end.

---

## Phase 0: Foundation (v0.1.0)

### Step 1: Project scaffolding

Set up the repo so that `pip install -e ".[dev]"` works from the first commit.

**Tasks:**

- [ ] Create `src/contextkit/__init__.py` with version string
- [ ] Create `pyproject.toml` (see RELEASING.md for full contents)
- [ ] Create `.github/workflows/ci.yml` (pytest + ruff + mypy on push)
- [ ] Create `.github/workflows/publish.yml` (PyPI on release)
- [ ] Verify `pip install -e ".[dev]"` works
- [ ] Verify `pytest`, `ruff check src/`, `mypy src/` all pass (on empty project)

**Output:** An installable, empty package that passes CI.

---

### Step 2: Provenance data models

Build the origin and mutation types first. These are used by every other component.

**Tasks:**

- [ ] Create `src/contextkit/observe/provenance.py`:
  - `Origin` Pydantic model:
    - `source: str` (e.g. "prompt", "conversation", "rag", "file", "tool", "memory", "user", "example")
    - `created_at: datetime` (auto-set)
    - Extra fields vary by source type, stored in `details: dict`
    - Common detail fields: `template`, `query`, `retriever`, `relevance_score`, `file_path`, `chunk_index`, `tool_name`, `turn_range`
  - `Mutation` Pydantic model:
    - `step: str` (pipeline step name, e.g. "CompactStep", "TrimStep")
    - `action: str` (e.g. "compacted", "removed", "moved", "merged")
    - `detail: str` (human-readable description)
    - `tokens_before: int`
    - `tokens_after: int`
    - `before_content: str | None` (original content, stored for compaction/trim)
    - `after_content: str | None` (modified content)
    - `timestamp: datetime` (auto-set)
- [ ] Write tests in `tests/test_observe.py`:
  - Origin creation with different source types
  - Mutation creation with before/after tracking
  - Serialization to dict/JSON

**Output:** `Origin` and `Mutation` importable from `contextkit.observe`.

---

### Step 3: BlockType enum and ContextBlock

The fundamental data unit. Now includes provenance from the start.

**Tasks:**

- [ ] Define `BlockType` enum in `src/contextkit/core.py` with all 12 context types
- [ ] Define `ContextBlock` as a Pydantic model:
  - `type: BlockType`
  - `content: str | list[dict]` (string for prompts, list for message histories)
  - `priority: int` (higher = more important to keep, default 50)
  - `metadata: dict` (optional, arbitrary key-value pairs)
  - `name: str | None` (optional human-readable label)
  - `origin: Origin | None` (where this block came from)
  - `mutations: list[Mutation]` (history of changes, empty by default)
- [ ] Add `__repr__` that shows type, name, token count, and origin source
- [ ] Write tests in `tests/test_core.py`:
  - Block creation with all fields including origin
  - Block creation with defaults (origin=None, mutations=[])
  - Serialization to dict/JSON preserves provenance
  - Validation errors for bad inputs

**Output:** `ContextBlock` and `BlockType` importable from `contextkit`.

---

### Step 4: Token counting with caching

Token counting is used everywhere. Build it once, cache it, and make it available standalone.

**Tasks:**

- [ ] Create `src/contextkit/utils/token_counting.py`:
  - `count(content: str | list[dict], encoding: str = "cl100k_base") -> int`
  - `fits_budget(content: str | list[dict], max_tokens: int, encoding: str = "cl100k_base") -> bool`
  - LRU cache on `count()` keyed by content hash + encoding name
  - For `list[dict]` content (message lists), count each message's content and add per-message overhead
- [ ] Create `src/contextkit/utils/cache.py` with shared LRU cache config (max size, TTL)
- [ ] Add `token_count` property to `ContextBlock` that uses the cached counter
- [ ] Re-export `count` and `fits_budget` from `contextkit.tokens` (public API)
- [ ] Write tests in `tests/test_tokens.py`:
  - Count for simple strings
  - Count for message lists
  - Cache hit verification (same content returns same result without recomputation)
  - `fits_budget` true/false cases

**Output:** `from contextkit.tokens import count, fits_budget` works standalone.

---

### Step 5: Model registry

The registry maps model names to their properties. This powers model-aware context limits, correct tokenizer selection, and cost estimation.

**Tasks:**

- [ ] Create `src/contextkit/models.py`:
  - `ModelSpec` Pydantic model: `max_context: int`, `encoding: str`, `input_cost_per_mtok: float`, `output_cost_per_mtok: float`
  - `_REGISTRY: dict[str, ModelSpec]` with built-in entries:
    - `claude-opus-4-6` (200k, cl100k_base, pricing)
    - `claude-sonnet-4-5-20250929` (200k, cl100k_base, pricing)
    - `claude-haiku-4-5-20251001` (200k, cl100k_base, pricing)
    - `gpt-4o` (128k, o200k_base, pricing)
    - `gpt-4o-mini` (128k, o200k_base, pricing)
    - `gpt-4.1` (1M, o200k_base, pricing)
  - `get_model(name: str) -> ModelSpec` (raises `UnknownModel` if not found)
  - `register_model(name: str, spec: ModelSpec) -> None`
  - `list_models() -> list[str]`
- [ ] Write tests in `tests/test_models.py`:
  - Lookup known model
  - Register and lookup custom model
  - UnknownModel error for bad name
  - list_models returns all built-in models

**Output:** `from contextkit.models import get_model, register_model` works.

---

### Step 6: Event system

Build the event/hook system early so all subsequent components can emit events.

**Tasks:**

- [ ] Create `src/contextkit/observe/events.py`:
  - `ContextEvent` enum:
    - `BLOCK_ADDED`, `BLOCK_REMOVED`, `BLOCK_MUTATED`
    - `BUDGET_WARNING`, `BUDGET_EXCEEDED`
    - `ASSEMBLY_COMPLETE`
    - `PIPELINE_STEP`, `PIPELINE_COMPLETE`
    - `WINDOW_RENDERED`
  - `EventData` Pydantic model -- base class for event payloads:
    - `event: ContextEvent`
    - `timestamp: datetime`
    - Subclasses: `BlockEvent` (has block), `BudgetEvent` (has percent, threshold), `PipelineEvent` (has step, report)
  - `on(event: ContextEvent)` -- decorator to register a callback
  - `emit(event_data: EventData)` -- fire an event to all registered callbacks
  - `clear_handlers()` -- reset all handlers (useful in tests)
  - Global handler registry (module-level, simple list per event type)
- [ ] Write tests in `tests/test_events.py`:
  - Register a handler and verify it fires
  - Multiple handlers for same event
  - Handler receives correct event data
  - clear_handlers resets state

**Output:** `from contextkit.events import on, ContextEvent` works.

---

### Step 7: Budget warnings

Configurable threshold warnings that fire through the event system and Python logging.

**Tasks:**

- [ ] Create `src/contextkit/observe/warnings.py`:
  - `BudgetMonitor` class:
    - Constructor takes `thresholds: list[float]` (e.g. [0.75, 0.90, 0.95])
    - `check(token_count: int, max_tokens: int) -> None` -- emits BUDGET_WARNING event and logs a warning if a threshold is crossed
    - Tracks which thresholds have already fired to avoid repeat warnings
    - Log message includes: percent used, tokens used/total, largest block name and its share
  - Integrates with Python `logging` module (logger name: `contextkit`)
- [ ] Write tests in `tests/test_warnings.py`:
  - Warning fires at correct threshold
  - Warning does not re-fire for same threshold
  - Warning resets when token count drops below threshold
  - Log message contains expected details

**Output:** Budget warnings fire automatically when tokens cross thresholds.

---

### Step 8: ContextWindow

The core container. Holds blocks, tracks tokens, enforces budgets, emits events, and checks budget warnings.

**Tasks:**

- [ ] Define `ContextWindow` in `src/contextkit/core.py`:
  - Constructor: `ContextWindow(model: str | None = None, max_tokens: int | None = None, budget_warnings: list[float] | None = None)`
    - If model is given, pull max_tokens and encoding from the registry
    - If only max_tokens is given, use it directly with default encoding
    - If neither, raise an error
    - If budget_warnings is given, create a `BudgetMonitor`
  - `add(block: ContextBlock) -> None`:
    - Adds block, raises `BudgetExceeded` if it would overflow
    - Emits `BLOCK_ADDED` event
    - Runs budget monitor check
  - `remove(name: str) -> None`:
    - Removes block by name
    - Emits `BLOCK_REMOVED` event
  - `blocks: list[ContextBlock]` -- ordered list of blocks
  - `token_count: int` -- total tokens across all blocks (cached, invalidated on add/remove)
  - `budget_remaining: int` -- max_tokens minus token_count
  - `cost_estimate: float` -- input cost based on token_count and model pricing (0.0 if no model)
  - `cost_estimate_with_response(output_tokens: int) -> float` -- input + expected output cost
  - `render() -> list[dict]` -- returns assembled messages list, ordered by block priority; emits `WINDOW_RENDERED` event
  - `to_dict() -> dict` -- full serialization including provenance
  - `clear_cache() -> None`
- [ ] Define `BudgetExceeded` exception in `core.py` -- emits `BUDGET_EXCEEDED` event on raise
- [ ] Write tests in `tests/test_core.py` (extend existing file):
  - Create window with model name
  - Create window with manual max_tokens
  - Add blocks and verify token_count updates
  - BudgetExceeded when adding a block that overflows
  - BLOCK_ADDED event fires on add
  - BLOCK_REMOVED event fires on remove
  - Budget warning fires when threshold crossed
  - Remove block and verify token_count decreases
  - cost_estimate returns correct value
  - render() returns blocks ordered by priority (high to low)
  - to_dict() includes origin and mutations for each block

**Output:** `from contextkit import ContextWindow` works with events and budget monitoring.

---

### Step 9: ContextAssembler

The assembler composes blocks into a window with ordering and priority rules. Produces a full report that powers `explain()`.

**Tasks:**

- [ ] Create `src/contextkit/assembler.py`:
  - `ContextAssembler`:
    - Constructor takes a `ContextWindow`
    - `assemble(blocks: list[ContextBlock]) -> ContextWindow` -- adds blocks in priority order, skips any that would exceed budget, returns the window
    - Emits `ASSEMBLY_COMPLETE` event with the report
  - `AssemblyReport` Pydantic model:
    - `included: list[BlockDecision]` (block name, tokens, priority, origin summary)
    - `excluded: list[BlockDecision]` (block name, tokens, priority, origin summary, reason)
    - `total_tokens: int`
    - `budget_remaining: int`
    - `cost_estimate: float`
  - `BlockDecision` Pydantic model:
    - `block_name: str`
    - `block_type: BlockType`
    - `tokens: int`
    - `priority: int`
    - `origin_summary: str` (e.g. "rag/chroma, relevance 0.87")
    - `reason: str | None` (only for excluded: "budget_exceeded", "below_threshold", etc.)
  - The report is stored on the window as `window._assembly_report` for use by `explain()`
- [ ] Write tests in `tests/test_core.py` (or `tests/test_assembler.py`):
  - Assemble blocks that all fit
  - Assemble blocks where some don't fit (low priority dropped)
  - Report correctly lists included and excluded blocks with reasons
  - Report includes origin summaries
  - ASSEMBLY_COMPLETE event fires with report

**Output:** `from contextkit import ContextAssembler` works with explainable reports.

---

### Step 10: Provider adapters

The feature that makes Phase 0 worth installing. Format a ContextWindow for any provider.

**Tasks:**

- [ ] Create `src/contextkit/adapters/base.py`:
  - `ProviderAdapter` Protocol with one method: `format(window: ContextWindow) -> dict`
  - Also a static convenience: `format_messages(messages: list[dict], system: str | None = None) -> dict` for standalone use without ContextWindow
- [ ] Create `src/contextkit/adapters/anthropic.py`:
  - `AnthropicAdapter` implementing `ProviderAdapter`
  - `format()`:
    - Extracts SYSTEM_PROMPT blocks into top-level `system` param
    - Converts remaining blocks to Anthropic `messages` format
    - Returns `{"model": ..., "max_tokens": ..., "system": ..., "messages": [...]}`
    - Emits `WINDOW_RENDERED` event
  - `format_messages()` for standalone use
- [ ] Create `src/contextkit/adapters/openai.py`:
  - `OpenAIAdapter` implementing `ProviderAdapter`
  - `format()`:
    - Converts SYSTEM_PROMPT blocks into `{"role": "system", "content": ...}` messages
    - Converts remaining blocks to OpenAI messages format
    - Returns `{"model": ..., "messages": [...]}`
    - Emits `WINDOW_RENDERED` event
  - `format_messages()` for standalone use
- [ ] Re-export adapters from `contextkit.adapters.__init__`
- [ ] Write tests in `tests/test_adapters.py`:
  - Anthropic adapter produces valid payload structure
  - System prompt is extracted to top level for Anthropic
  - OpenAI adapter puts system prompt in messages array
  - format_messages works without a ContextWindow
  - Custom adapter (implementing Protocol) works
  - WINDOW_RENDERED event fires

**Output:** `from contextkit.adapters import AnthropicAdapter, OpenAIAdapter` works.

---

### Step 11: Context inspection

Multi-level inspection: summary table, per-block drill-down, and JSON dump.

**Tasks:**

- [ ] Create `src/contextkit/observe/renderers.py`:
  - `TextRenderer` -- formats tables as plain text for terminals
  - `HtmlRenderer` -- formats tables as rich HTML for Jupyter notebooks
  - Auto-detect environment (check for IPython/Jupyter) to pick default renderer
- [ ] Create `src/contextkit/observe/inspect.py`:
  - `inspect_window(window: ContextWindow, block_name: str | None = None, format: str = "auto") -> str`:
    - If `block_name` is None: summary table with columns: Block, Type, Tokens, Priority, Cost, Budget %, Origin
    - If `block_name` is given: drill into that block:
      - For SHORT_TERM_MEMORY: per-message table (index, role, tokens, content preview)
      - For RAG: per-chunk table (source, tokens, relevance, content preview)
      - For TOOL_DEFINITIONS: per-tool table (name, description, param count)
      - For other types: full content with metadata
    - Footer row with totals and budget remaining
  - `diff_windows(a: ContextWindow, b: ContextWindow, format: str = "auto") -> str`:
    - Blocks added (in b but not a)
    - Blocks removed (in a but not b)
    - Blocks changed (same name, different token count or content)
    - Token delta and cost delta
  - `dump_window(window: ContextWindow, path: str) -> None`:
    - Full JSON snapshot: all blocks with content, provenance, mutations, metadata
    - Assembly report if available
    - Model info, token counts, cost estimate
- [ ] Add convenience methods on `ContextWindow`:
  - `inspect(block_name=None, format="auto")` -- prints or displays `inspect_window(self, ...)`
  - `diff(other, format="auto")` -- prints or displays `diff_windows(self, other, ...)`
  - `dump(path)` -- calls `dump_window(self, path)`
- [ ] Write tests in `tests/test_observe.py` (extend):
  - Summary inspect output contains all block names, token counts, origin column
  - Drill-down inspect for SHORT_TERM_MEMORY shows per-message rows
  - Drill-down inspect for RAG shows per-chunk rows with relevance
  - Cost column appears when model is set, absent when no model
  - diff shows added/removed/changed blocks with token delta
  - dump creates valid JSON that can be loaded and verified

**Output:** `window.inspect()`, `window.inspect("block_name")`, `window.diff()`, `window.dump()` all work.

---

### Step 12: Explain mode

Query why any block was included or excluded. Pulls from assembly report and mutation log.

**Tasks:**

- [ ] Create `src/contextkit/observe/explain.py` (or add to inspect.py):
  - `explain_block(window: ContextWindow, block_name: str) -> str`:
    - If block is in window: report status INCLUDED, token count, priority, budget share, origin details, any mutations applied
    - If block is in assembly report's excluded list: report status EXCLUDED, reason, what it would have cost
    - If block is unknown: report "no block with this name found"
    - Returns a multi-line plain-language explanation
- [ ] Add convenience method on `ContextWindow`:
  - `explain(block_name: str)` -- prints `explain_block(self, block_name)`
- [ ] Write tests in `tests/test_observe.py` (extend):
  - explain() for included block mentions origin and token count
  - explain() for excluded block mentions reason and would-have-been cost
  - explain() for unknown block returns clear message
  - explain() after pipeline run includes mutation history

**Output:** `window.explain("block_name")` gives a clear, plain-language answer.

---

### Step 13: Public API and exports

Wire everything together into clean top-level imports.

**Tasks:**

- [ ] Update `src/contextkit/__init__.py` to export:
  - `ContextWindow`, `ContextBlock`, `BlockType`, `BudgetExceeded`
  - `ContextAssembler`, `AssemblyReport`
  - `Origin`, `Mutation`
- [ ] Create `src/contextkit/tokens.py` (public module) re-exporting from `_tokens`:
  - `count`, `fits_budget`
- [ ] Create `src/contextkit/events.py` (public module) re-exporting from `observe/events.py`:
  - `on`, `ContextEvent`
- [ ] Verify all import paths from PRD API sketches work:
  - `from contextkit import ContextWindow, ContextBlock, BlockType, Origin`
  - `from contextkit.tokens import count, fits_budget`
  - `from contextkit.adapters import AnthropicAdapter, OpenAIAdapter`
  - `from contextkit.models import get_model, register_model, ModelSpec`
  - `from contextkit.events import on, ContextEvent`
- [ ] Write an integration test that runs the full Phase 0 API example from the PRD:
  - Create window with model and budget_warnings
  - Add blocks with origins
  - inspect() produces expected output
  - inspect("block_name") drills down correctly
  - explain() returns correct explanation
  - Adapter formats valid payload
  - Events fire for all actions
  - dump() produces valid JSON with provenance

**Output:** All documented imports work. PRD examples run without modification.

---

### Step 14: README, docs, and release prep

Make the project presentable and publishable.

**Tasks:**

- [ ] Rewrite README.md with:
  - One-liner: "Know exactly what your model sees."
  - Install instructions
  - Quick start example (the Phase 0 API sketch from the PRD)
  - Observability section showing inspect(), explain(), and event hooks
  - Standalone utilities section
  - Links to PRD for full roadmap
- [ ] Update CHANGELOG.md with v0.1.0 entries
- [ ] Run full check: `pytest && ruff check src/ && mypy src/`
- [ ] Test install from built wheel: `python -m build && pip install dist/*.whl`
- [ ] Test upload to TestPyPI (see RELEASING.md)
- [ ] Tag v0.1.0 and publish

**Output:** v0.1.0 live on PyPI.

---

## Phase 1: Memory Management (v0.2.0) -- Outline

| Step | What                          | Key deliverable                              |
|------|-------------------------------|----------------------------------------------|
| 15   | MemoryBackend protocol        | `Protocol` class with 4 async methods        |
| 16   | MemoryRecord data model       | Pydantic model with content, metadata, tags  |
| 17   | InMemoryBackend               | Default backend for development and tests    |
| 18   | ShortTermMemory               | Sliding window + token-budget trimming; auto-populates Origin with turn range and trimming info |
| 19   | trim_conversation() standalone| Works without ShortTermMemory class          |
| 20   | SQLiteBackend                 | Async SQLite via aiosqlite, zero-config      |
| 21   | LongTermMemory                | Store/retrieve with pluggable backends; auto-populates Origin with query and tags |
| 22   | Sync wrappers                 | `contextkit.sync` module for non-async code  |
| 23   | Integration tests             | SQLite persistence across process restarts   |
| 24   | Custom backend example        | Documented example of writing a Redis backend|

---

## Phase 2: Prompts & Files (v0.3.0) -- Outline

| Step | What                    | Key deliverable                                    |
|------|-------------------------|----------------------------------------------------|
| 25   | PromptManager           | Jinja2 templates with register/render; Origin auto-populated with template name, version, variables |
| 26   | Prompt versioning       | Version tracking and diff between prompt versions  |
| 27   | Prompt composition      | Base + override layering                           |
| 28   | FileContext indexing     | Scan directory, store metadata without loading      |
| 29   | FileContext loading      | Lazy load with chunking by token budget; Origin auto-populated with file path, chunk index |
| 30   | Format-aware parsing    | Markdown, code, CSV, PDF text extraction           |
| 31   | ExampleStore            | Few-shot management with similarity selection; Origin auto-populated with example ID, similarity score |
| 32   | Example budget mgmt     | Fit N best examples within token limit             |

---

## Phase 3: Tools & RAG (v0.4.0) -- Outline

| Step | What                    | Key deliverable                                    |
|------|-------------------------|----------------------------------------------------|
| 33   | RetrieverBackend protocol| `Protocol` with 2 async methods                   |
| 34   | ToolRegistry            | Register tools with schemas; Origin auto-populated with tool name, selection reason |
| 35   | Dynamic tool selection  | Select relevant tools based on task description    |
| 36   | Tool output capture     | ToolOutput block type; Origin auto-populated with tool name, call ID, latency |
| 37   | MCP compatibility       | Read/write MCP-formatted tool definitions          |
| 38   | RAGContext              | Retrieval with ranking, dedup, attribution; Origin auto-populated with query, retriever, relevance |
| 39   | Budget-aware retrieval  | Stop retrieving when token budget is full          |
| 40   | Chroma retriever example| Working RetrieverBackend implementation            |

---

## Phase 4: Context Pipeline (v0.5.0) -- Outline

| Step | What                    | Key deliverable                                    |
|------|-------------------------|----------------------------------------------------|
| 41   | Pipeline base           | ContextPipeline with ordered step execution        |
| 42   | TrimStep                | Remove low-priority blocks; records Mutation with reason and before/after |
| 43   | DeduplicateStep         | Remove redundant info; records Mutation with similarity score |
| 44   | FilterStep              | Remove blocks below threshold; records Mutation with threshold and score |
| 45   | ReorderStep             | Key info at start/end; records Mutation with old/new position |
| 46   | CompactStep             | LLM summarization; records Mutation with full before/after content |
| 47   | Pipeline observability  | Run reports with per-step breakdown, tokens saved, cost delta; fires PIPELINE_STEP and PIPELINE_COMPLETE events |
| 48   | Quality scoring         | Signal-to-noise estimation                         |

---

## Phase 5: Multi-Agent & Timeline (v0.6.0) -- Outline

| Step | What                    | Key deliverable                                    |
|------|-------------------------|----------------------------------------------------|
| 49   | ContextScope            | Isolate context per agent                          |
| 50   | SharedMemory            | Memory blocks shared across agents                 |
| 51   | Handoff protocol        | Structured context transfer between agents         |
| 52   | Scratchpad              | Agent working notes for multi-step reasoning       |
| 53   | Context timeline        | `track_history=True`, `timeline()`, `snapshot_at(turn)`, `export_timeline()` |
| 54   | Rich notebook rendering | HTML renderer for inspect/diff/timeline in Jupyter |
| 55   | Context snapshots       | Save/restore for debugging and replay              |

---

## Dependencies between steps

```
Step 1 (scaffolding)
  |
  v
Step 2 (Origin + Mutation models) ---------> used by everything
  |
  v
Step 3 (BlockType + ContextBlock) ---------> depends on Step 2
  |
  v
Step 4 (token counting) -------------------> standalone: contextkit.tokens
  |
  v
Step 5 (model registry)
  |
  v
Step 6 (event system) ---------------------> used by Steps 7, 8, 9, 10
  |
  v
Step 7 (budget warnings) ------------------> depends on Step 6
  |
  v
Step 8 (ContextWindow) --------------------> depends on Steps 3, 4, 5, 6, 7
  |
  +-------+----------+---------+
  |       |          |         |
  v       v          v         v
Step 9  Step 10    Step 11   Step 12
(assembler) (adapters) (inspect)  (explain)
  |       |          |         |
  +-------+----------+---------+
  |
  v
Step 13 (public API + integration) --------> depends on all above
  |
  v
Step 14 (README + release) ----------------> depends on Step 13
```

Steps 9, 10, 11, and 12 are independent of each other and can be built in parallel once Step 8 is done.
