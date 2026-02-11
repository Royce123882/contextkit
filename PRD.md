# contextkit -- Product Requirements Document

> Build context once. Send it anywhere. Know exactly what the model sees.

---

## Problem

Every AI agent framework solves tool calling, orchestration, and routing. Almost none of them treat the **context window** as a first-class, engineerable artifact. Developers end up hand-rolling prompt assembly, memory trimming, token budgeting, and RAG injection with bespoke glue code that breaks as soon as requirements change.

Worse, every provider has a different message format. Claude puts `system` at the top level. OpenAI puts it in the messages array. Token limits differ. Pricing differs. Developers rewrite their context assembly layer every time they switch providers or add a new one.

**contextkit** treats the context window the way a build system treats compilation: typed blocks go in, an optimized and provider-ready payload comes out. And like a good build system, it tells you exactly what happened at every step -- what went in, what got cut, why, and what it cost.

---

## Principles

1. **No black boxes.** The developer must always be able to answer: "What exactly is the model seeing right now, and why?" Every block tracks where it came from, every mutation is recorded, and every decision is explainable.
2. **Context is data.** Every piece of information fed to a model -- prompts, memory, files, tool schemas, RAG chunks, examples -- is a typed block with metadata and provenance.
3. **Assembly is a pipeline.** Blocks are composed, ordered, trimmed, deduplicated, and compressed through explicit, inspectable steps. Each step explains what it did and why.
4. **Token budgets are constraints, not afterthoughts.** The SDK enforces budgets at every stage so you never hit a surprise context overflow.
5. **Provider-agnostic by default.** Build your context once. Format it for Anthropic, OpenAI, or any other provider with a single adapter call.
6. **Pluggable with concrete interfaces.** Backends are `Protocol` classes with 3-4 methods. The SDK ships sensible defaults and makes custom backends simple to write.
7. **Incrementally adoptable.** Use the full framework or cherry-pick individual utilities (token counting, conversation trimming, provider formatting). No all-or-nothing buy-in.
8. **Async-first.** All I/O-bound operations are async. Sync wrappers are provided for scripts and notebooks.

---

## Context Taxonomy

The SDK recognizes 12 context types. Each maps to a `BlockType` enum and has purpose-built management utilities.

| # | Context Type              | Description                                                    |
|---|---------------------------|----------------------------------------------------------------|
| 1 | System Prompts            | Behavioral instructions, role definitions, rules               |
| 2 | Short-Term Memory         | Current session conversation history, working state            |
| 3 | Long-Term Memory          | Persistent knowledge across sessions (KV store, vector store)  |
| 4 | Files / Documents         | PDFs, code files, CSVs -- loaded or referenced                 |
| 5 | Tool Definitions          | Schemas describing available tools the agent can call          |
| 6 | Tool Outputs              | Results returned from tool calls, fed back into context        |
| 7 | Retrieved Knowledge (RAG) | Dynamically retrieved chunks from external sources             |
| 8 | Examples / Few-Shot       | Canonical input-output pairs demonstrating desired behavior    |
| 9 | Structured Output Schemas | JSON/XML schemas constraining model output format              |
|10 | System Metadata           | Token counts, context window stats, agent state metadata       |
|11 | Scratchpad / Working Notes| Agent's self-maintained notes for multi-step reasoning         |
|12 | User / Entity Context     | User profiles, preferences, session metadata                   |

---

## Tech Stack

| Component        | Choice                       | Rationale                                           |
|------------------|------------------------------|-----------------------------------------------------|
| Language         | Python >= 3.10               | Dominant in AI/ML ecosystem                         |
| Build system     | Hatchling                    | Modern, standards-compliant, minimal config         |
| Data validation  | Pydantic >= 2.0              | Fast, typed, widely adopted for schemas             |
| Token counting   | tiktoken >= 0.7.0            | OpenAI's fast BPE tokenizer, industry standard      |
| Caching          | cachetools >= 5.3             | LRU/TTL caches for token counts and repeated ops    |
| Testing          | pytest >= 8.0                | De facto standard                                   |
| Async tests      | pytest-asyncio >= 0.23       | Needed for async memory backends                    |
| Linting          | Ruff >= 0.4.0                | Fast, replaces flake8 + isort + pyupgrade           |
| Type checking    | mypy >= 1.10 (strict)        | Catch bugs before runtime                           |
| SQLite backend   | aiosqlite >= 0.19.0          | Zero-config persistent memory (optional dep)        |
| Vector backend   | chromadb >= 0.5.0            | Popular embedding store (optional dep)              |
| Templating       | Jinja2 >= 3.1                | Prompt templates with variable interpolation        |
| CI/CD            | GitHub Actions               | Trusted Publishing to PyPI, no API tokens needed    |
| Docs             | MkDocs or Read the Docs      | Developer-friendly documentation                    |
| License          | MIT                          | Maximum adoption                                    |

---

## Project Layout

```
contextkit/
  src/
    contextkit/
      __init__.py
      core.py                # ContextWindow, ContextBlock, BlockType, Origin
      assembler.py           # ContextAssembler, AssemblyReport
      models.py              # Model registry (limits, tokenizers, pricing)
      _tokens.py             # Token counting with LRU cache
      _cache.py              # Shared caching utilities
      adapters/
        __init__.py
        base.py              # ProviderAdapter protocol
        anthropic.py         # Anthropic message formatting
        openai.py            # OpenAI message formatting
      observe/
        __init__.py
        provenance.py        # Origin, Mutation data models
        inspect.py           # inspect(), inspect(block_name), explain()
        diff.py              # diff() between two windows
        timeline.py          # Multi-turn context history tracking
        events.py            # Event system (hooks, callbacks)
        warnings.py          # Budget warning system
        renderers.py         # Text and HTML output renderers
      memory/
        __init__.py
        short_term.py
        long_term.py
        backends.py          # MemoryBackend protocol + in-memory default
      prompts/
        __init__.py
        manager.py
      files/
        __init__.py
        context.py
      tools/
        __init__.py
        registry.py
      rag/
        __init__.py
        context.py
        backends.py          # RetrieverBackend protocol
      pipeline/
        __init__.py
        steps.py
      sync.py                # Sync wrappers for async APIs
  tests/
    test_core.py
    test_adapters.py
    test_models.py
    test_tokens.py
    test_observe.py          # Provenance, inspect, explain, diff
    test_events.py           # Event hooks and callbacks
    test_timeline.py         # Multi-turn history tracking
    test_warnings.py         # Budget warning thresholds
    test_memory.py
    test_prompts.py
    test_files.py
    test_tools.py
    test_rag.py
    test_pipeline.py
  pyproject.toml
  README.md
  LICENSE
  CHANGELOG.md
  PRD.md
  PLAN.md
  RELEASING.md
  .github/
    workflows/
      publish.yml
      ci.yml
```

---

## Observability and Explainability

This is the core differentiator. Most frameworks let you assemble context. None of them tell you what the model is actually seeing, why, or how it got there. contextkit makes every piece of context traceable, every decision explainable, and every mutation visible.

### Block Provenance

Every `ContextBlock` carries an `origin` field that records where it came from. This is set automatically when blocks are created through the SDK's managers (PromptManager, ShortTermMemory, RAGContext, etc.) and can be set manually for custom blocks.

```python
block = ContextBlock(
    type=BlockType.RAG,
    content="Auth uses JWT tokens with 24h expiry...",
    priority=70,
    origin=Origin(
        source="rag",                          # which system produced this
        query="how does auth work",            # what triggered the retrieval
        retriever="chroma/docs-index",         # which backend
        relevance_score=0.87,                  # how relevant the system thinks it is
        retrieved_at="2026-02-09T10:30:00Z",   # when
    ),
)

# Or let the SDK fill it in automatically
rag = RAGContext(retriever=my_retriever)
blocks = await rag.retrieve("how does auth work", top_k=3)
# Each block's origin is already populated
print(blocks[0].origin.source)           # "rag"
print(blocks[0].origin.relevance_score)  # 0.87
```

Every block type has a relevant origin shape:

| Block type           | Origin fields                                                |
|----------------------|--------------------------------------------------------------|
| System prompt        | `source="prompt"`, template name, template version, variables used |
| Short-term memory    | `source="conversation"`, turn index range, trimming applied  |
| Long-term memory     | `source="memory"`, key, tags, stored_at, retrieval query     |
| File / document      | `source="file"`, file path, chunk index, byte range          |
| RAG chunk            | `source="rag"`, query, retriever name, relevance score       |
| Tool definition      | `source="tool"`, tool name, selection reason                 |
| Tool output          | `source="tool_output"`, tool name, call ID, latency          |
| Few-shot example     | `source="example"`, example ID, similarity score             |
| User context         | `source="user"`, user ID, fields included                    |

### Mutation Log

When a block is modified -- trimmed, compacted, deduplicated, reordered -- the SDK records a `Mutation` entry on the block. The full history is preserved so you can trace what happened to any piece of context.

```python
# After running a pipeline
optimized = pipeline.run(window)

for block in optimized.blocks:
    if block.mutations:
        print(f"{block.name}:")
        for m in block.mutations:
            print(f"  [{m.step}] {m.action}: {m.detail}")

# system_prompt:
#   (no mutations)
# short_term_memory:
#   [CompactStep] compacted: 8,200 -> 4,100 tokens (summary model: claude-haiku)
#   [ReorderStep] moved: position 3 -> position 1
# rag_chunks:
#   [DeduplicateStep] removed: chunk "auth-overview" overlaps with "auth-detail" (82% similarity)
#   [TrimStep] removed: chunk "auth-legacy" dropped (priority 40 below threshold 50)

# See the before and after for any compaction
m = optimized.blocks[1].mutations[0]
print(m.before_content[:100])  # Original text (first 100 chars)
print(m.after_content[:100])   # Compacted text
print(m.tokens_before)         # 8200
print(m.tokens_after)          # 4100
```

### Context Inspection

Multi-level inspection from summary to full detail.

```python
# High-level summary table
window.inspect()
# +------------------------+--------+----------+--------+--------+------------------+
# | Block                  | Tokens | Priority | Cost   | Budget | Origin           |
# +------------------------+--------+----------+--------+--------+------------------+
# | system_prompt          |    320 |      100 | $0.001 |   0.2% | prompt/assistant |
# | user_context           |    150 |       90 | $0.000 |   0.1% | user/profile     |
# | short_term_memory (12) |  8,200 |       80 | $0.025 |   4.1% | conversation     |
# | rag_chunks (3)         |  3,100 |       70 | $0.009 |   1.6% | rag/chroma       |
# | tool_definitions (5)   |  1,430 |       60 | $0.004 |   0.7% | tool/registry    |
# +------------------------+--------+----------+--------+--------+------------------+
# | Total                  | 13,200 |          | $0.040 |   6.6% |                  |
# | Budget remaining       |186,800 |          |        |  93.4% |                  |
# +------------------------+--------+----------+--------+--------+------------------+

# Drill into a specific block
window.inspect("short_term_memory")
# Short-term memory: 12 messages, 8,200 tokens
# +-----+------------+--------+-----------------------------------+
# | #   | Role       | Tokens | Content (preview)                 |
# +-----+------------+--------+-----------------------------------+
# | 1   | user       |    45  | Can you explain how auth works... |
# | 2   | assistant  |   820  | Sure. The authentication system...|
# | 3   | user       |    32  | What about refresh tokens?        |
# | ... | ...        |   ...  | ...                               |
# | 12  | user       |    28  | Thanks, one more question about...|
# +-----+------------+--------+-----------------------------------+
# Origin: conversation, turns 1-12
# Trimming: none applied (12/20 max turns, 8,200/8,000 max tokens)

# Drill into a RAG block
window.inspect("rag_chunks")
# RAG chunks: 3 chunks, 3,100 tokens
# +---+------------------+--------+-----------+--------------------------+
# | # | Source           | Tokens | Relevance | Content (preview)        |
# +---+------------------+--------+-----------+--------------------------+
# | 1 | docs/auth.md:4-8 |  1,200 |     0.92  | JWT tokens are issued... |
# | 2 | docs/auth.md:12  |    980 |     0.87  | Refresh tokens use a...  |
# | 3 | docs/api.md:3    |    920 |     0.71  | The /auth endpoint...    |
# +---+------------------+--------+-----------+--------------------------+
# Query: "how does auth work"
# Retriever: chroma/docs-index

# Compare two windows
window.diff(other_window)
# Shows blocks added, removed, changed, with token and cost deltas

# Full JSON export for external tools or logging
window.dump("snapshot.json")
```

### Explain Mode

Ask the SDK why a block was included or excluded. This queries the assembly report and pipeline mutation log to give a plain-language answer.

```python
window.explain("rag_chunks")
# Block "rag_chunks" is INCLUDED (3,100 tokens, priority 70)
# - Added by: ContextAssembler
# - Origin: RAGContext query "how does auth work" via chroma/docs-index
# - 3 of 5 retrieved chunks kept (2 excluded: 1 below min_relevance 0.7, 1 deduplicated)
# - No mutations applied
# - Budget impact: 1.6% of 200,000 token window

window.explain("auth-legacy")
# Block "auth-legacy" is EXCLUDED
# - Origin: RAGContext query "how does auth work" via chroma/docs-index
# - Retrieved with relevance 0.42 (below min_relevance threshold 0.7)
# - Would have used 1,800 tokens (0.9% of budget)
# - Excluded at: assembly stage (relevance filter)
```

### Budget Warnings

The SDK emits warnings at configurable thresholds so developers know when context is getting tight before it overflows.

```python
from contextkit import ContextWindow

window = ContextWindow(
    model="claude-sonnet-4-5-20250929",
    budget_warnings=[0.75, 0.90, 0.95],  # warn at 75%, 90%, 95%
)

window.add(large_history_block)
# [contextkit] WARNING: budget at 78% (156,000 / 200,000 tokens)
#   Largest block: short_term_memory (120,000 tokens, 60% of total)
#   Consider: trimming short_term_memory or compacting with pipeline

window.add(rag_block)
# [contextkit] WARNING: budget at 92% (184,000 / 200,000 tokens)
#   Only 16,000 tokens remaining
#   3 blocks use 89% of budget -- run window.inspect() for details
```

Warnings go through Python's standard `logging` module by default, so they integrate with whatever logging setup the developer already has.

### Event Hooks

Developers can register callbacks for context lifecycle events. This plugs into existing observability stacks -- logging, OpenTelemetry, Weights & Biases, or any custom system.

```python
from contextkit.events import on, ContextEvent

@on(ContextEvent.BLOCK_ADDED)
def log_addition(event):
    print(f"Added {event.block.name}: {event.block.token_count} tokens")
    print(f"  Origin: {event.block.origin}")

@on(ContextEvent.BLOCK_TRIMMED)
def log_trim(event):
    print(f"Trimmed {event.block.name}: {event.tokens_before} -> {event.tokens_after}")

@on(ContextEvent.BUDGET_WARNING)
def alert_on_budget(event):
    slack.post(f"Context budget at {event.percent}% for agent {event.window.name}")

@on(ContextEvent.WINDOW_RENDERED)
def track_cost(event):
    metrics.record("context_cost", event.window.cost_estimate)
    metrics.record("context_tokens", event.window.token_count)
```

Available events:

| Event                | Fires when                                                  |
|----------------------|-------------------------------------------------------------|
| `BLOCK_ADDED`        | A block is added to a window                                |
| `BLOCK_REMOVED`      | A block is removed from a window                            |
| `BLOCK_MUTATED`      | A pipeline step modifies a block (trim, compact, reorder)   |
| `BUDGET_WARNING`     | Token usage crosses a warning threshold                     |
| `BUDGET_EXCEEDED`    | A block is rejected for exceeding budget                    |
| `ASSEMBLY_COMPLETE`  | Assembler finishes with full report                         |
| `PIPELINE_STEP`      | A pipeline step completes with its report                   |
| `PIPELINE_COMPLETE`  | Full pipeline completes with summary report                 |
| `WINDOW_RENDERED`    | Window is formatted for a provider (final payload ready)    |

### Context Timeline

For agents that run multi-turn loops, the SDK can record how the context window changes across turns. This is opt-in since it requires storing snapshots.

```python
from contextkit import ContextWindow

window = ContextWindow(model="claude-sonnet-4-5-20250929", track_history=True)

# ... agent loop runs 20 turns ...

# See how context evolved
window.timeline()
# Turn  1: 12,400 tokens (6.2%)   +system +user_context +history(2)
# Turn  5: 34,800 tokens (17.4%)  +rag(3) +tool_output  history(10)
# Turn 10: 89,200 tokens (44.6%)  +rag(5) history(20)   compacted: -12,000
# Turn 15: 142,000 tokens (71.0%) history(30) trimmed: -18,000  WARNING: 75%
# Turn 20: 178,400 tokens (89.2%) history(38) trimmed: -24,000  WARNING: 90%

# Inspect any past turn
window.snapshot_at(turn=10).inspect()

# Export full timeline for analysis
window.export_timeline("timeline.json")
```

### Rich Output for Notebooks

In Jupyter environments, `inspect()`, `diff()`, and `timeline()` return rich HTML with color-coded budget utilization, collapsible block details, and interactive drill-down. In terminal environments, they fall back to formatted text tables.

```python
# In Jupyter -- automatically renders as rich HTML
window.inspect()  # Color-coded table with expandable blocks

# Force text output in any environment
window.inspect(format="text")
```

---

## Cross-Cutting Concerns

These capabilities apply across all phases. Each is woven into the relevant phase rather than shipped as a standalone milestone.

### 1. Provider Adapters

Build context once, format it for any provider. Adapters handle the differences between Anthropic (system as top-level param, content blocks), OpenAI (system as message role, string content), and others.

```python
from contextkit.adapters import AnthropicAdapter, OpenAIAdapter

payload = AnthropicAdapter.format(window)
response = anthropic_client.messages.create(**payload)

payload = OpenAIAdapter.format(window)
response = openai_client.chat.completions.create(**payload)
```

Writing a custom adapter means implementing one method:

```python
from contextkit.adapters.base import ProviderAdapter

class MyAdapter(ProviderAdapter):
    def format(self, window: ContextWindow) -> dict: ...
```

### 2. Model Registry

The SDK ships with a registry of known models and their properties: context window size, tokenizer encoding, and input/output pricing. Developers can register custom models.

```python
from contextkit import ContextWindow

window = ContextWindow(model="claude-sonnet-4-5-20250929")
print(window.max_tokens)       # 200_000
print(window.cost_estimate)    # $0.0024

# Custom model
from contextkit.models import register_model, ModelSpec
register_model("my-fine-tune", ModelSpec(
    max_context=32_000,
    encoding="cl100k_base",
    input_cost_per_mtok=0.50,
    output_cost_per_mtok=1.50,
))
```

### 3. Async-First Design

All I/O-bound operations (memory backends, RAG retrieval, LLM-powered compaction) are async. Sync wrappers are provided for scripts, notebooks, and codebases that are not async.

```python
# Async (default)
results = await ltm.retrieve("user preferences", top_k=5)

# Sync wrapper
from contextkit.sync import LongTermMemory as SyncLTM
results = SyncLTM(backend="sqlite", path="./db").retrieve("user preferences", top_k=5)
```

Pure computation (token counting, block assembly, rendering) stays synchronous since there is no I/O to await.

### 4. Cost Estimation

Token counts are already tracked. Combined with the model registry's pricing data, every window can report its estimated cost before the API call is made.

```python
print(window.cost_estimate)                    # $0.040
print(window.cost_estimate_with_response(500)) # $0.055 (includes expected output)
```

### 5. Caching

Token counting is CPU work. Encoding the same strings repeatedly is waste. The SDK uses LRU caches internally for:

- Token count results (keyed by content hash + encoding)
- Rendered block outputs (keyed by block content + version)
- Model registry lookups

```python
from contextkit import ContextWindow

# Second call is instant -- cached
window.token_count  # computed
window.token_count  # cached

# Cache can be cleared if needed
window.clear_cache()
```

### 6. Concrete Backend Protocols

Every pluggable interface is a Python `Protocol` with a small, documented surface area. Developers can see exactly what to implement.

**Memory backend (4 methods):**

```python
from contextkit.memory.backends import MemoryBackend, MemoryRecord

class RedisBackend(MemoryBackend):
    async def store(self, key: str, value: str, metadata: dict) -> None: ...
    async def retrieve(self, query: str, top_k: int) -> list[MemoryRecord]: ...
    async def delete(self, key: str) -> None: ...
    async def list_records(self, tags: list[str] | None = None) -> list[MemoryRecord]: ...
```

**RAG retriever backend (2 methods):**

```python
from contextkit.rag.backends import RetrieverBackend, Chunk

class PineconeRetriever(RetrieverBackend):
    async def retrieve(self, query: str, top_k: int) -> list[Chunk]: ...
    async def health_check(self) -> bool: ...
```

**Provider adapter (1 method):**

```python
from contextkit.adapters.base import ProviderAdapter

class LiteLLMAdapter(ProviderAdapter):
    def format(self, window: ContextWindow) -> dict: ...
```

### 7. Incremental Adoption

Every piece of the SDK works standalone. Developers can use one utility without buying into the full framework.

```python
# Just count tokens
from contextkit.tokens import count, fits_budget
count("Hello world", model="claude-sonnet-4-5-20250929")  # 2
fits_budget(messages, max_tokens=8000)                     # True

# Just trim a conversation
from contextkit.memory import trim_conversation
trimmed = trim_conversation(messages, strategy="sliding_window", max_turns=20)

# Just format for a provider
from contextkit.adapters import AnthropicAdapter
payload = AnthropicAdapter.format_messages(messages, system="You are helpful.")
```

### 8. Model-Aware Context Limits

Different models have different context windows, tokenizers, and message formats. The SDK handles this so developers don't have to.

```python
from contextkit import ContextWindow

# Claude Opus: 200k context, uses Anthropic message format
window_opus = ContextWindow(model="claude-opus-4-6")

# GPT-4o: 128k context, uses OpenAI message format
window_gpt = ContextWindow(model="gpt-4o")

# Window knows its own limits
window_opus.add(huge_block)  # Raises BudgetExceeded if block won't fit
```

When no model is specified, `ContextWindow(max_tokens=N)` works as a generic container with manual token limits.

---

## Phased Delivery

### Phase 0 -- Foundation (v0.1.0)

**Goal:** Make v0.1.0 worth installing on its own. A developer should be able to build context, format it for their provider, see exactly what the model will receive, know what it costs, and trace where every piece came from.

**Scope:**

- `ContextBlock` -- a typed unit of context with content, priority, metadata, and origin
- `BlockType` enum covering all 12 context types
- `Origin` data model -- provenance tracking for every block (source, query, timestamps)
- `ContextWindow` -- the core container representing an assembled context window
- `ContextAssembler` -- composes blocks with ordering and priority rules, produces `AssemblyReport`
- Token counting via tiktoken with LRU caching
- Model registry with known models (Claude, GPT-4o, GPT-4o-mini, Llama, Mistral)
- Provider adapters for Anthropic and OpenAI
- Observability (ships in v0.1.0, not bolted on later):
  - `window.inspect()` -- summary table with tokens, priority, cost, budget %, origin per block
  - `window.inspect("block_name")` -- drill into a specific block (per-message or per-chunk detail)
  - `window.explain("block_name")` -- plain-language explanation of why a block is included or excluded
  - `window.diff(other)` -- side-by-side comparison of two windows
  - `window.dump(path)` -- full JSON snapshot with provenance and metadata
  - Budget warnings at configurable thresholds via Python logging
  - Event hooks for BLOCK_ADDED, BLOCK_REMOVED, BUDGET_WARNING, BUDGET_EXCEEDED, ASSEMBLY_COMPLETE, WINDOW_RENDERED
- JSON serialization of assembled context
- Full test suite, type checking, linting

**API:**

```python
from contextkit import ContextWindow, ContextBlock, BlockType
from contextkit.adapters import AnthropicAdapter

# Build context with model awareness
window = ContextWindow(
    model="claude-sonnet-4-5-20250929",
    budget_warnings=[0.75, 0.90, 0.95],
)

system = ContextBlock(
    type=BlockType.SYSTEM_PROMPT,
    content="You are a helpful assistant...",
    priority=100,
    origin=Origin(source="prompt", template="assistant_v1"),
)
history = ContextBlock(
    type=BlockType.SHORT_TERM_MEMORY,
    content=conversation_messages,
    priority=80,
    origin=Origin(source="conversation", turns="1-12"),
)

window.add(system)
window.add(history)

# See exactly what the model will receive
window.inspect()                           # Summary table
window.inspect("short_term_memory")        # Per-message breakdown
window.explain("system_prompt")            # Why is this block here?

print(f"Cost: {window.cost_estimate}")
print(f"Remaining: {window.budget_remaining} tokens")

# Format for your provider and send
payload = AnthropicAdapter.format(window)
response = anthropic_client.messages.create(**payload)
```

**Event hooks for your observability stack:**

```python
from contextkit.events import on, ContextEvent

@on(ContextEvent.BLOCK_ADDED)
def log_block(event):
    logger.info(f"Added {event.block.name} ({event.block.token_count} tokens)")

@on(ContextEvent.BUDGET_WARNING)
def alert_budget(event):
    logger.warning(f"Budget at {event.percent}%")
```

**Standalone utilities (work without ContextWindow):**

```python
from contextkit.tokens import count, fits_budget
from contextkit.adapters import OpenAIAdapter

count("Hello world", model="gpt-4o")     # 2
fits_budget(msgs, max_tokens=8000)        # True
OpenAIAdapter.format_messages(msgs)       # Ready-to-send dict
```

**Exit criteria:** installable via `pip install contextkit`, all tests green, README examples run without modification, adapters produce valid payloads for Anthropic and OpenAI APIs, `inspect()` and `explain()` return correct output for all block types.

---

### Phase 1 -- Memory Management (v0.2.0)

**Goal:** First-class memory with concrete backend interfaces and async support.

**Scope:**

- `ShortTermMemory` -- conversation history with trimming strategies
  - Sliding window (keep last N turns)
  - Token-budget trimming (keep most recent that fit)
  - Summary-based compression (LLM-powered compaction)
- `LongTermMemory` -- persistent store with pluggable backends
  - `MemoryBackend` protocol (4 methods: store, retrieve, delete, list_records)
  - In-memory backend (default, for development)
  - SQLite backend (built-in, zero-config)
  - Vector store interface (bring your own: Chroma, Pinecone, etc.)
- Memory read/write with metadata (timestamps, tags, importance scores)
- Automatic relevance scoring on retrieval
- All backends are async-first with sync wrappers in `contextkit.sync`
- Standalone `trim_conversation()` utility for use without the full framework

**API:**

```python
from contextkit.memory import ShortTermMemory, LongTermMemory

# Short-term with automatic trimming
stm = ShortTermMemory(
    strategy="sliding_window",
    max_turns=20,
    max_tokens=8000,
)
stm.add_turn(role="user", content="Hello")
stm.add_turn(role="assistant", content="Hi there!")

# Long-term with SQLite persistence
ltm = LongTermMemory(backend="sqlite", path="./memory.db")
await ltm.store("user_preference", "Prefers concise answers", tags=["style"])
results = await ltm.retrieve("what does the user prefer?", top_k=5)

# Custom backend -- implement 4 methods
from contextkit.memory.backends import MemoryBackend, MemoryRecord

class RedisBackend(MemoryBackend):
    async def store(self, key: str, value: str, metadata: dict) -> None: ...
    async def retrieve(self, query: str, top_k: int) -> list[MemoryRecord]: ...
    async def delete(self, key: str) -> None: ...
    async def list_records(self, tags: list[str] | None = None) -> list[MemoryRecord]: ...

ltm = LongTermMemory(backend=RedisBackend(url="redis://localhost"))
```

**Exit criteria:** memory backends pass integration tests, SQLite persistence works across process restarts, custom backend protocol is documented with a working example.

---

### Phase 2 -- Prompts & Files (v0.3.0)

**Goal:** Structured management of static and semi-static context.

**Scope:**

- `PromptManager` -- versioned prompt templates
  - Jinja2-based templates with variable interpolation
  - Prompt versioning and diffing
  - Prompt composition (base + overrides)
- `FileContext` -- smart file handling for context injection
  - Lazy loading (metadata-only until needed)
  - Chunking strategies (by section, by token budget, by relevance)
  - Format-aware parsing (markdown, code, CSV, PDF text extraction)
  - Reference tracking (file paths as lightweight pointers)
- `ExampleStore` -- few-shot example management
  - Dynamic example selection based on similarity to input
  - Example budget management (fit N best examples within token limit)

**API:**

```python
from contextkit.prompts import PromptManager
from contextkit.files import FileContext

pm = PromptManager()
pm.register("assistant_v1", "You are {{role}}. Rules: {{rules}}")
prompt_block = pm.render("assistant_v1", role="analyst", rules=rules_text)

fc = FileContext("./docs/")
fc.index()
refs = fc.search("authentication flow", top_k=3)
blocks = fc.load(refs, max_tokens=4000)
```

**Exit criteria:** templates render correctly with edge cases (missing vars, nested includes), file chunking respects token budgets.

---

### Phase 3 -- Tools & RAG (v0.4.0)

**Goal:** Dynamic context with concrete retriever interfaces.

**Scope:**

- `ToolRegistry` -- manage tool definitions as context blocks
  - Register tools with schemas
  - Dynamic tool selection (only include relevant tools per request)
  - Tool output capture and context injection
  - MCP compatibility layer
- `RAGContext` -- retrieval-augmented context assembly
  - `RetrieverBackend` protocol (2 methods: retrieve, health_check)
  - Chunk ranking and deduplication
  - Source attribution tracking
  - Token-budget-aware retrieval
- `ToolOutput` block type for structured capture of tool results
- All retrieval operations are async-first

**API:**

```python
from contextkit.tools import ToolRegistry
from contextkit.rag import RAGContext

registry = ToolRegistry()
registry.register(
    name="search_docs",
    description="Search internal documentation",
    parameters={"query": {"type": "string"}},
)
tool_blocks = registry.select(task_description="Find the auth spec")

rag = RAGContext(retriever=my_chroma_retriever)
knowledge_blocks = await rag.retrieve(
    query="How does our auth system work?",
    max_tokens=3000,
    min_relevance=0.7,
)

# Custom retriever -- implement 2 methods
from contextkit.rag.backends import RetrieverBackend, Chunk

class ElasticRetriever(RetrieverBackend):
    async def retrieve(self, query: str, top_k: int) -> list[Chunk]: ...
    async def health_check(self) -> bool: ...
```

**Exit criteria:** tool schemas serialize to OpenAI/Anthropic format, RAG retriever interface has at least one working implementation, custom retriever protocol is documented with a working example.

---

### Phase 4 -- Context Pipeline & Optimization (v0.5.0)

**Goal:** Context assembly as a build pipeline where every transformation is traced, explained, and reversible.

**Scope:**

- `ContextPipeline` -- named, ordered transformation steps
  - `TrimStep` -- remove low-priority blocks to fit budget
  - `CompactStep` -- LLM-powered summarization of verbose blocks
  - `DeduplicateStep` -- remove redundant information across blocks
  - `ReorderStep` -- optimize block ordering for attention patterns (key info at start/end)
  - `FilterStep` -- remove blocks below relevance threshold
- `Mutation` log -- every step records what it changed, with before/after content and token counts
- Pipeline observability -- each step fires PIPELINE_STEP events and produces a step-level report
- Token budget optimizer -- maximize information within constraints
- Context quality scoring -- signal-to-noise estimation
- Pipeline results include cost delta (how much money the optimization saved)
- `explain()` works on pipeline decisions: "why was this block trimmed?" "what did compaction remove?"

**API:**

```python
from contextkit.pipeline import ContextPipeline, TrimStep, CompactStep

pipeline = ContextPipeline([
    DeduplicateStep(),
    CompactStep(model="claude-sonnet-4-5-20250929", target_ratio=0.5),
    TrimStep(strategy="priority", max_tokens=100_000),
    ReorderStep(strategy="important_edges"),
])

optimized_window = pipeline.run(window)

print(pipeline.last_run.report)
# > Removed 3 duplicate blocks (1,200 tokens saved)
# > Compacted short_term_memory: 8,000 -> 4,100 tokens
# > Trimmed 2 low-priority blocks (2,400 tokens)
# > Final: 45,200 / 100,000 tokens (45.2% utilization)
# > Estimated cost savings: $0.012 per call

# Inspect what happened to a specific block
for m in optimized_window.blocks[1].mutations:
    print(f"[{m.step}] {m.action}: {m.tokens_before} -> {m.tokens_after} tokens")
    # See the actual content before and after
    print(f"  Before: {m.before_content[:80]}...")
    print(f"  After:  {m.after_content[:80]}...")

# Ask why a decision was made
optimized_window.explain("low_priority_block")
# Block "low_priority_block" was EXCLUDED by TrimStep
# - Priority 30 was below cutoff 50 (set by token budget constraint)
# - Would have used 2,400 tokens (2.4% of budget)
# - Removing it freed space for 1 higher-priority block
```

**Exit criteria:** pipeline produces deterministic, reproducible reports; every mutation is logged with before/after content; trim + dedup steps have full test coverage.

---

### Phase 5 -- Multi-Agent & Scoping (v0.6.0)

**Goal:** Support for multi-agent architectures with isolated and shared context, plus full timeline observability across agent loops.

**Scope:**

- `ContextScope` -- isolate context per agent with explicit sharing rules
- `SharedMemory` -- memory blocks shared across multiple agents
- Agent handoff protocol -- structured context transfer between agents
- Scratchpad -- agent-managed working notes for multi-step reasoning
- Context timeline (`track_history=True`):
  - Records a snapshot at each turn in an agent loop
  - `window.timeline()` -- shows how context evolved across turns (tokens, blocks added/removed/mutated, warnings)
  - `window.snapshot_at(turn=N)` -- inspect the context window at any past turn
  - `window.export_timeline(path)` -- full timeline export for external analysis
- Context snapshots -- save and restore context state for debugging and replay

**Exit criteria:** two agents can share a memory backend, hand off context, and operate with isolated scopes in a single test; timeline correctly records 20+ turns with snapshots inspectable at any point.

---

## Versioning

Semantic versioning. Pre-1.0, breaking changes are expected between minor versions.

| Version | Milestone              | Developer value proposition                                                           |
|---------|------------------------|---------------------------------------------------------------------------------------|
| 0.1.0   | Foundation             | Build, inspect, explain, and send context to any provider -- full observability from day one |
| 0.2.0   | Memory management      | Conversation memory with trimming that just works, custom backends in 4 methods            |
| 0.3.0   | Prompts & files        | Manage all static context in one place with versioning and lazy loading                    |
| 0.4.0   | Tools & RAG            | Dynamic context plugs in cleanly, custom retrievers in 2 methods                          |
| 0.5.0   | Context pipeline       | Optimize context with traceable mutations and full before/after visibility                 |
| 0.6.0   | Multi-agent & scoping  | Agents share context with isolation; timeline tracks evolution across turns                |
| 1.0.0   | Stable API             | Production-ready with backwards compatibility guarantees                                  |

---

## Publishing

- Package name: `contextkit` (verify availability on PyPI before first publish)
- Build: `python -m build`
- Test upload: `twine upload --repository testpypi dist/*`
- Production upload: `twine upload dist/*` (or Trusted Publishing via GitHub Actions)
- Tag each release: `git tag v0.x.0 && git push origin v0.x.0`
- See `RELEASING.md` for the full publishing playbook.

---

## Success Metrics

- Installable from PyPI with zero errors
- 90%+ test coverage on core modules
- Type-clean under `mypy --strict`
- Lint-clean under `ruff check`
- README examples run without modification
- Adapters produce valid payloads verified against real Anthropic and OpenAI APIs
- All backend protocols have at least one working example implementation
- `inspect()` returns correct output for all block types including per-item drill-down
- `explain()` gives a clear answer for any included or excluded block
- Every block created through SDK managers has origin auto-populated
- Every pipeline mutation is logged with before/after content
- Event hooks fire for all documented events
- At least one external contributor or issue filed within 30 days of launch
