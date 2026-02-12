# Context Engineering Research & Roadmap

> A comprehensive research report covering the context engineering landscape, academic foundations, contextkit SDK analysis, and proposed roadmap.

---

## Table of Contents

1. [What Is Context Engineering?](#1-what-is-context-engineering)
2. [Industry Landscape](#2-industry-landscape)
3. [Core Techniques & Approaches](#3-core-techniques--approaches)
4. [Academic Foundations (arXiv Papers)](#4-academic-foundations-arxiv-papers)
5. [Tools & Frameworks Landscape](#5-tools--frameworks-landscape)
6. [contextkit SDK Analysis](#6-contextkit-sdk-analysis)
7. [SDK UX Improvement Opportunities](#7-sdk-ux-improvement-opportunities)
8. [Proposed Roadmap](#8-proposed-roadmap)
9. [References](#9-references)

---

## 1. What Is Context Engineering?

Context engineering is the discipline of designing and building dynamic systems that provide the right information and tools, in the right format, at the right time, to give an LLM everything it needs to accomplish a task.

**Canonical definition** (Philipp Schmid, Google DeepMind):
> "Context Engineering is the discipline of designing and building dynamic systems that provides the right information and tools, in the right format, at the right time, to give a LLM everything it needs to accomplish a task."

**Key mental model** (Andrej Karpathy): Think of an LLM as a CPU and its context window as RAM/working memory. The engineer's job is akin to an operating system -- load that working memory with just the right code and data for the task at hand.

**Critical insight**: Most agent failures are not model failures -- they are context failures. Even a weaker LLM can perform well with the right context, but no state-of-the-art model can compensate for a poor one.

### Relationship to Other Approaches

| Approach | Scope | Relationship to Context Engineering |
|----------|-------|-------------------------------------|
| **Prompt Engineering** | How you phrase instructions | A component -- handles the "instruction" portion of context |
| **RAG** | Retrieving external data at inference time | A key technique within context engineering |
| **Fine-tuning** | Modifying model weights | Complementary; context engineering operates at inference time |
| **Few-shot Learning** | Providing examples in the prompt | A context engineering technique for pattern establishment |
| **Memory Systems** | Persisting state across interactions | A context engineering component for long-running agents |
| **Tool Use** | Giving models access to functions | A component; tool descriptions and outputs are context |

**The analogy**: Prompt engineering is how you ask the question. Context engineering is making sure the model has access to the right textbook, calculator, and notes before it starts thinking.

---

## 2. Industry Landscape

### Key People

| Person | Affiliation | Contribution |
|--------|-------------|-------------|
| **Tobi Lutke** | CEO, Shopify | Catalyzed mainstream adoption of the term (June 2025) |
| **Andrej Karpathy** | ex-OpenAI/Tesla | Endorsed and popularized the CPU/RAM mental model |
| **Philipp Schmid** | Google DeepMind | Authored canonical definition and widely-shared guide |
| **Simon Willison** | Datasette | Argued the term would stick due to better "inferred definition" |

### Key Companies & Contributions

| Company | Contribution |
|---------|-------------|
| **Anthropic** | Published "Effective Context Engineering for AI Agents" (Sept 2025); built Claude Code with context engineering as core design principle; introduced MCP (Nov 2024) |
| **Google** | Released Agent Development Kit (ADK) with context engineering as first-class concern |
| **Spotify** | Published 3-part engineering blog from 1,500+ production PRs with coding agents |
| **LangChain** | Published dedicated context engineering blog; evolved LangGraph for stateful orchestration |
| **Gartner** | Declared "context engineering is in, prompt engineering is out" (July 2025) |

---

## 3. Core Techniques & Approaches

### 3.1 Compaction (Summarization & Compression)

The primary technique for managing long-running agent sessions. When a context window nears capacity, the conversation is summarized and the agent continues with compressed context.

- Claude Code implements "auto-compact" at 95% capacity
- Preserves architectural decisions and unresolved bugs while discarding redundant tool outputs
- What you *remove* from context can matter as much as what you keep
- A focused 300-token context often outperforms an unfocused 113,000-token context

### 3.2 Just-in-Time Context (Dynamic Retrieval)

Rather than pre-loading all relevant data, agents maintain lightweight identifiers and use tools to dynamically load data at runtime. Anthropic's Claude Code uses this hybrid approach: CLAUDE.md files are loaded up front, while tools like `glob` and `grep` enable just-in-time navigation.

### 3.3 Context Trimming & Filtering

Pruning context using heuristics -- removing older messages, filtering by importance, or using trained pruners. More surgical than full summarization, useful for maintaining token budgets without invoking an LLM.

### 3.4 Multi-Agent Context Isolation

Partitioning context across specialized sub-agents rather than cramming everything into one window. Anthropic's research demonstrated that many agents with isolated contexts outperformed single-agent implementations by 90% on complex information retrieval tasks.

### 3.5 State Object Isolation

An agent's runtime state is designed as a structured schema. Only one field is exposed to the LLM at each step, while others remain isolated for selective use.

### 3.6 Tool Design as Context Engineering

Tools act as the agent's interface to its environment. Well-designed tools are small, distinct, and efficient. Applying RAG principles to tool descriptions has achieved 3x improvements in tool selection accuracy.

### 3.7 Memory Systems

- **Short-term memory**: Maintains state within a session
- **Long-term memory**: Persists knowledge across sessions (vector stores or knowledge graphs)
- Spotify found that agents writing plans to persistent memory was essential for surviving context truncation

### 3.8 RAG & GraphRAG

RAG was one of the first context engineering techniques. The state of the art is moving toward GraphRAG, combining vector search with graph traversal for richer context assembly.

### Known Failure Modes

| Failure Mode | Description |
|-------------|-------------|
| **Context rot** | Accuracy decreases as token count increases, even within stated context window |
| **Context poisoning** | Hallucinated/outdated information enters context and is repeatedly referenced |
| **Brevity bias** | Summarization drops domain-specific insights for concise but shallow summaries |
| **Context collapse** | Iterative rewriting erodes detail over time |
| **Lost in the middle** | Models attend more to beginning and end of long contexts |

---

## 4. Academic Foundations (arXiv Papers)

### 4.1 Foundational Surveys

| Paper | ID | Key Finding |
|-------|-----|-------------|
| **A Survey of Context Engineering for LLMs** | arXiv:2507.13334 | Definitive survey analyzing 1,400+ papers; establishes formal taxonomy: context retrieval, processing, and management |
| **Context Engineering 2.0** | arXiv:2510.26493 | Traces roots back 20+ years; maps evolution through distinct phases |
| **Agentic Context Engineering (ACE)** | arXiv:2510.04618 | Treats contexts as evolving "playbooks"; +10.6% on agent benchmarks; open-source model matches IBM's GPT-4.1 agent |
| **Invasive Context Engineering** | arXiv:2512.03001 | Explores operator-side context manipulation for safety and control |

### 4.2 Context Window Optimization

| Paper | ID | Key Finding |
|-------|-----|-------------|
| **Maximum Effective Context Window** | arXiv:2509.21361 | Effective context is drastically different from advertised -- some models fail at 100 tokens |
| **LongRoPE2** | arXiv:2502.20082 | Near-lossless context window scaling via modified positional encoding |
| **InfiniteICL** | arXiv:2504.01707 | Reduces context by 90% while achieving 103% of full-context performance |
| **Solving Context Window Overflow** | arXiv:2511.22729 | Memory pointers instead of raw data -- shifts paradigm from fitting data to referencing it |
| **How to Train Long-Context LLMs** | arXiv:2410.02660 | Disabling cross-document attention benefits both short and long-context performance |

### 4.3 Context Compression

| Paper | ID | Key Finding |
|-------|-----|-------------|
| **Prompt Compression Survey** | arXiv:2410.12388 | Comprehensive survey: LLMLingua achieves up to 20x compression |
| **LLMLingua-2** | arXiv:2403.12968 | 3-6x faster than existing compression; robust generalization across LLMs |
| **SAC (Semantic Anchors)** | arXiv:2510.08907 | Anchor token selection eliminates training-inference mismatch |
| **COMI** | arXiv:2602.01719 | Two-stage coarse-to-fine compression, near-SOTA across QA and summarization |
| **Semantic Compression** | arXiv:2312.09571 | Information-theory-inspired, 6-8x context extension without fine-tuning |

### 4.4 RAG & Context Construction

| Paper | ID | Key Finding |
|-------|-----|-------------|
| **RAG for LLMs Survey** | arXiv:2312.10997 | Defines Naive/Advanced/Modular RAG taxonomy |
| **RAG vs Long-Context** | arXiv:2407.16833 | Long-context outperforms RAG on quality but at higher cost; Self-Route hybrid approach wins |
| **Stronger Baselines for RAG** | arXiv:2506.03989 | Simple retrieve-then-read competitive with complex iterative pipelines |
| **RAG Systems Review** | arXiv:2507.18910 | 1,200+ RAG papers on arXiv in 2024 alone (vs <100 previous year) |

### 4.5 Memory Management in AI Agents

| Paper | ID | Key Finding |
|-------|-----|-------------|
| **Agentic Memory** | arXiv:2601.01885 | Unifies LTM/STM as tool-based actions; agent autonomously manages memory |
| **A-Mem** | arXiv:2502.12110 | Dynamic memory without predefined structures; Zettelkasten-inspired |
| **Mem0** | arXiv:2504.19413 | Production-ready: 26% improvement, 91% lower p95 latency, 90%+ token cost reduction |
| **Memory in the Age of AI Agents** | arXiv:2512.13564 | Traditional LTM/STM taxonomies are insufficient for agent memory |

### 4.6 Multi-Turn Conversation

| Paper | ID | Key Finding |
|-------|-----|-------------|
| **LLMs Get Lost In Multi-Turn** | arXiv:2505.06120 | Average 39% performance drop in multi-turn vs single-turn |
| **Context Drift** | arXiv:2510.07777 | Formalizes "context drift" -- gradual degradation of conversational state |
| **MultiChallenge** | arXiv:2501.17399 | All frontier models <50% accuracy; Claude 3.5 Sonnet scored 41.4% |

### 4.7 Tool Use & Agent Context

| Paper | ID | Key Finding |
|-------|-----|-------------|
| **ARTIST** | arXiv:2505.01441 | Unified framework for thinking, tool queries, and tool outputs |
| **Dynamic Tool Retrieval** | arXiv:2512.17052 | Retrieval conditioned on query + tool-call history, not all tools at once |
| **ScaleMCP** | arXiv:2505.06416 | Dynamic tool retriever with MCP as single source of truth |
| **LOCA-bench** | arXiv:2602.07962 | Benchmarks agent performance under extreme context growth |

### 4.8 Infrastructure (KV Cache)

| Paper | ID | Key Finding |
|-------|-----|-------------|
| **KV Cache Management Survey** | arXiv:2412.19442 | Categorizes token-level, model-level, and system-level strategies |
| **LMCache** | arXiv:2510.09665 | Production-grade KV cache used in vLLM, Dynamo, llm-d, KServe |

### 4.9 MCP & Agent Protocols

| Paper | ID | Key Finding |
|-------|-----|-------------|
| **CA-MCP** | arXiv:2601.11595 | Shared Context Store enables autonomous MCP server coordination |
| **MCP-Universe** | arXiv:2508.14704 | Benchmark: even GPT-5 only 43.72% on MCP tasks |
| **Agent Interoperability Survey** | arXiv:2505.02279 | Compares MCP, ACP, A2A, ANP protocols |

---

## 5. Tools & Frameworks Landscape

### Core Frameworks

| Framework | Strength | Context Engineering Role |
|-----------|----------|------------------------|
| **LangChain / LangGraph** | Orchestration, agents, tools | Stateful cyclic graphs; memory systems |
| **LlamaIndex** | Data ingestion, indexing, RAG | Advanced indexing; context-aware agents |
| **Google ADK** | Multi-agent production | First-class context stack; session/state separation |
| **Claude Code** | Coding agents | Auto-compaction, CLAUDE.md, just-in-time retrieval |
| **CrewAI** | Multi-agent workflows | Role-based coordination with isolated contexts |
| **DSPy** | Programmatic prompt optimization | Treats prompts as optimizable programs |
| **Haystack** | Production RAG pipelines | Modular retrieval and context assembly |

### Key Protocols

- **MCP (Model Context Protocol)** -- Open standard by Anthropic (Nov 2024); adopted by OpenAI, Google, Microsoft, HuggingFace; 40-60% faster agent deployment
- **A2A (Agent-to-Agent)** -- Standardizes cross-agent communication; enables "swarm" architectures

### Emerging Production Stack (2026)

```
LlamaIndex          → Knowledge/data layer
LangChain/LangGraph → Orchestration layer
MCP                 → Context delivery protocol
Vector DBs          → Retrieval (Pinecone, Weaviate, Chroma, Qdrant)
n8n / similar       → Workflow engine
```

---

## 6. contextkit SDK Analysis

### What It Is

**contextkit** is a context engineering SDK that treats the context window as a first-class, engineerable artifact with full observability, provenance tracking, and provider-agnostic formatting.

**Core vision**: "Build context once. Send it anywhere. Know exactly what the model sees."

### Architecture: 12 Context Types

| Type | Purpose |
|------|---------|
| `SYSTEM_PROMPT` | Behavioral instructions, role definitions |
| `SHORT_TERM_MEMORY` | Current session conversation history |
| `LONG_TERM_MEMORY` | Persistent knowledge across sessions |
| `FILES` | PDFs, code files, CSVs |
| `TOOL_DEFINITIONS` | Schemas for available tools |
| `TOOL_OUTPUTS` | Results returned from tool calls |
| `RAG` | Dynamically retrieved chunks |
| `EXAMPLES` | Few-shot demonstration pairs |
| `OUTPUT_SCHEMAS` | JSON/XML schemas for output formatting |
| `SYSTEM_METADATA` | Token counts, context stats, agent state |
| `SCRATCHPAD` | Agent's self-maintained working notes |
| `USER_CONTEXT` | User profiles, preferences, session metadata |

### Core Abstractions

| Abstraction | Role |
|-------------|------|
| **ContextBlock** | Fundamental unit: typed content with priority, metadata, origin, mutations |
| **ContextWindow** | Assembled container: holds blocks, tracks tokens, enforces budgets |
| **Origin** | Provenance: records where each block came from |
| **Mutation** | Change tracking: records how each block was modified |
| **ContextAssembler** | Composition engine: priority-based ordering, budget enforcement |
| **ContextPipeline** | Optimization: trim, compact, deduplicate, reorder, filter |
| **ContextScope** | Multi-agent isolation with scratchpad, handoff, shared memory |

### What's Complete (v0.1.0 - v0.6.0)

- Core data models (BlockType, ContextBlock, Origin, Mutation)
- ContextWindow with token tracking, budget enforcement, cost estimation
- Token counting with bounded LRU caching (tiktoken-based, maxsize=4096)
- Model registry (Claude Opus/Sonnet/Haiku, GPT-4o/mini, GPT-4.1)
- Provider adapters (Anthropic, OpenAI)
- Full observability: `inspect()`, `explain()`, `diff()`, `timeline()`
- ShortTermMemory (sliding_window, token_budget strategies)
- LongTermMemory with pluggable backends (InMemory, SQLite, Postgres with connection pooling)
- PromptManager with Jinja2 templates, FileContext with lazy loading
- ToolRegistry with dynamic selection, RAGContext with 5 retriever backends
- Pipeline steps: Trim (with query-aware mode), Compact, Compress (IDF-based), Deduplicate, Filter, Reorder (with prefix-stable KV cache strategy), RAGCompress, Mask
- Quality scoring (positional attention model) and sufficiency checking
- Multi-agent: ContextScope, SharedMemory, HandoffPackage, Scratchpad, Timeline
- 22 test files, 621+ tests, 90% coverage target enforced, strict mypy, CI/CD via GitHub Actions

### Current Gaps (Verified Against Source Code)

| Category | Issue | Severity | Status |
|----------|-------|----------|--------|
| **Performance** | O(n^2) message trimming in ShortTermMemory (`pop(0)` in loop at `short_term.py:189,221`) | HIGH | Confirmed |
| **Performance** | SQLiteBackend `retrieve()` loads ALL records via `_fetch_all_records()` then filters in Python (`sqlite_backend.py:212`) | HIGH | Confirmed |
| **Security** | FileContext `load()`/`load_single()` reads arbitrary paths with no traversal validation (`file_context.py:150,234`) | CRITICAL | Confirmed |
| **Async** | sync-to-async bridge uses ThreadPoolExecutor anti-pattern with deadlock risk (`sync.py:37-39`) | HIGH | Confirmed |
| **Async** | No pagination (limit/offset) in any memory backend's `list_records()` | MEDIUM | Confirmed |
| **Docs** | Sphinx RTD has only scaffolding (`conf.py`, `index.rst`, `getting-started.rst`); no advanced guides | MEDIUM | Confirmed |

#### Already Fixed (Previously Reported as Gaps)

| Claim | Actual Status | Evidence |
|-------|---------------|----------|
| PostgresBackend lacks connection pooling | **FIXED** -- uses `asyncpg.create_pool()` with configurable min/max sizes | `postgres_backend.py:129-146` |
| Token counting cache is unbounded | **FIXED** -- uses `LRUCache(maxsize=4096)` | `utils/cache.py:16-17`, `constants.py:20` |

---

## 7. SDK UX Improvement Opportunities

### A. Developer Ergonomics

| Issue | Current State | Proposed Improvement |
|-------|--------------|---------------------|
| **Verbose origin creation** | Manual `Origin(source=..., details={...})` | Add `Origin.from_rag()`, `Origin.from_file()`, `Origin.from_tool()` convenience constructors |
| **Async/sync confusion** | Async-first with sync wrappers | Offer sync-first option for scripts/notebooks; clearer docs |
| **Pipeline config verbosity** | Each step manually instantiated | Builder pattern + named presets (`"aggressive"`, `"balanced"`, `"conservative"`) |
| **Block naming collisions** | Same-name blocks cause ambiguous `remove()` | Use internal block IDs; names are display-only |
| **No fluent/chainable API** | Each operation is a separate statement | Add `window.add(...).add(...).optimize(pipeline)` chaining |

### B. Performance & Scalability

| Issue | Fix | Status |
|-------|-----|--------|
| O(n^2) message trimming | Use `collections.deque(maxlen=N)` | Open |
| SQLite loads entire DB | Push filtering into SQL WHERE, add LIMIT/OFFSET | Open |
| ~~No connection pooling~~ | ~~Add persistent connections + pool~~ | **Already fixed** (PostgresBackend uses `asyncpg.create_pool`) |
| ~~Unbounded token cache~~ | ~~Document cache limits~~ | **Already fixed** (`LRUCache(maxsize=4096)`) |

### C. Observability Gaps

| Issue | Fix | Status |
|-------|-----|--------|
| Inconsistent Origin population | Enforce Origin creation through manager constructors only | Open |
| No intermediate pipeline states | Optional step-by-step snapshots | Open |
| Context quality metrics are positional only | Extend beyond positional scoring to include S/N ratio, redundancy, density | Partial (`observe/quality.py` has positional scoring) |
| Coarse timeline snapshots | Option for detailed block-level diffs per turn | Open |

### D. Integration Friction

| Issue | Fix |
|-------|-----|
| Minimal provider adapter interface | Add hooks for custom transformations; metadata preservation |
| Only 2 provider adapters | Add examples for LiteLLM, Ollama, Bedrock |
| Loose retriever protocol | Expand with optional health check, pagination, batch ops |
| No RAG feedback loop | Add `ContextBlock.feedback()` for post-hoc relevance tuning |

### E. Feature Status (Verified Against Source Code)

| Feature | Status | Location |
|---------|--------|----------|
| **Query-aware context pruning** | **Already implemented** | `pipeline/trim.py:49` -- TrimStep accepts `query` param for relevance-weighted blended scoring |
| **Prefix-aware structuring** | **Already implemented** | `pipeline/reorder.py:113-160` -- ReorderStep has `"prefix_stable"` strategy for KV cache optimization |
| **Post-retrieval RAG compression** | **Already implemented** | `pipeline/rag_compress.py:26-151` -- RAGCompressStep with sentence extraction and selective augmentation |
| **Token-level prompt compression** | **Already implemented** | `pipeline/compress.py:47-150` -- CompressStep with IDF-based scoring (distinct from CompactStep) |
| **Context quality scoring** | **Partially implemented** | `observe/quality.py:22-127` -- Positional quality scoring; missing S/N ratio, redundancy, density |
| **Context sufficiency checking** | **Already implemented** | `observe/sufficiency.py` -- Checks whether context is sufficient for the task |
| **Context quality dashboard** | Not yet implemented | -- |
| **Cost estimation preview** | Not yet implemented | -- |

### F. Remaining High-Value UX Features (Confirmed Missing)

| Feature | Impact | Complexity |
|---------|--------|------------|
| **Origin convenience constructors** (`from_rag()`, `from_file()`, `from_tool()`) | Reduce boilerplate | Low |
| **Pipeline presets** (`ContextPipeline.balanced()`, `.aggressive()`) | Faster onboarding | Low |
| **Fluent/chainable API** (`window.add().add()`) | Better ergonomics (`add()` returns `None` today) | Low |
| **Block IDs** (unique identifiers beyond names) | Unambiguous block operations | Medium |
| **RAG feedback loop** (`block.feedback()`) | Close retrieval quality loop | Medium |
| **Memory decay** | Time-based importance reduction | Medium |
| **Collapse detection** | Detect information loss during compression | Medium |

### F. Documentation & Discoverability

| Gap | Fix |
|-----|-----|
| Under-documented Origin detail fields | Publish Origin schema per BlockType |
| No troubleshooting guide | Add debugging patterns for common issues |
| No plugin development guide | Add "Building Custom Backends" tutorial |
| No migration guide from raw prompts | "From Prompt Engineering to Context Engineering" guide |

---

## 8. Proposed Roadmap

### Phase 1: Developer Experience (v0.7.0) -- Near Term

**Goal**: Make the SDK delightful to use for the common case.

- [ ] **Convenience constructors** for Origin (`from_rag`, `from_file`, `from_tool`, `from_prompt`)
- [ ] **Pipeline presets** (`ContextPipeline.balanced()`, `.aggressive()`, `.conservative()`)
- [ ] **Fluent API** for window operations (`window.add(...).add(...).optimize(...)`)
- [ ] **Sync-first wrappers** for notebooks/scripts (no async needed for simple cases)
- [ ] **Block IDs** for unambiguous identification (names become display-only)
- [ ] **Cost estimation preview** before API calls
- [ ] **Improved error messages** with actionable suggestions

### Phase 2: Performance & Production Hardening (v0.8.0)

**Goal**: Make the SDK production-ready for scale.

- [ ] **Fix O(n^2) trimming** in ShortTermMemory (use deque) -- `short_term.py:189,221`
- [ ] **Optimize SQLiteBackend** (SQL WHERE filters, pagination) -- `sqlite_backend.py:212`
- [x] ~~**PostgresBackend connection pooling**~~ -- Already uses `asyncpg.create_pool` at `postgres_backend.py:129-146`
- [ ] **Path traversal protection** in FileContext -- `file_context.py:150,234`
- [ ] **Robust async bridge** (replace ThreadPoolExecutor with anyio) -- `sync.py:37-39`
- [x] ~~**Bounded caches**~~ -- Already uses `LRUCache(maxsize=4096)` at `utils/cache.py:16-17`
- [ ] **Add pagination** to all memory backend `list_records()` methods
- [ ] **Benchmark suite** for performance regression testing

### Phase 3: Advanced Context Intelligence (v0.9.0)

**Goal**: Smarter context assembly informed by academic research.

- [x] ~~**Query-aware pruning**~~ -- Already implemented in TrimStep with `query` param and blended scoring (`pipeline/trim.py:49`)
- [x] ~~**Prefix-aware structuring**~~ -- Already implemented as `"prefix_stable"` strategy in ReorderStep (`pipeline/reorder.py:113-160`)
- [x] ~~**Post-retrieval compression**~~ -- Already implemented as RAGCompressStep (`pipeline/rag_compress.py:26-151`)
- [x] ~~**Context quality scoring (positional)**~~ -- Already implemented (`observe/quality.py:22-127`)
- [x] ~~**Context sufficiency checking**~~ -- Already implemented (`observe/sufficiency.py`)
- [ ] **Extended quality metrics** -- signal-to-noise ratio, redundancy detection, information density (beyond current positional scoring)
- [ ] **Memory decay** -- time-based importance reduction for long-term memory records
- [ ] **Collapse detection** -- detect information loss during compression pipelines
- [ ] **Adaptive memory** -- agent decides what/when to store/retrieve (inspired by AgeMem, arXiv:2601.01885)
- [ ] **Anti-drift mechanisms** -- detect and mitigate context drift in multi-turn (inspired by arXiv:2510.07777)

### Phase 4: Ecosystem Integration (v1.0.0)

**Goal**: First-class integration with the broader AI ecosystem.

- [ ] **MCP integration** -- read context from MCP servers, expose context as MCP resources
- [ ] **Additional provider adapters** -- LiteLLM, Ollama, Bedrock, Azure OpenAI
- [ ] **LangChain/LangGraph bridge** -- use contextkit as context layer within LangChain agents
- [ ] **LlamaIndex bridge** -- use LlamaIndex retrievers as contextkit backends
- [ ] **RAG feedback loop** -- `block.feedback(useful=True/False)` for retrieval quality improvement
- [ ] **GraphRAG support** -- knowledge graph traversal as a context source
- [ ] **A2A protocol support** -- context exchange across agent boundaries

### Phase 5: Observability & Developer Tools (v1.1.0)

**Goal**: Deep visibility into context behavior.

- [ ] **Context quality dashboard** -- web UI showing token distribution, redundancy, cost over time
- [ ] **Pipeline replay** -- step through pipeline execution with full intermediate states
- [ ] **Context diff viewer** -- visual comparison of context windows across turns
- [ ] **Budget alerts** -- Slack/webhook notifications for budget anomalies
- [ ] **OpenTelemetry integration** -- export context metrics to observability platforms
- [ ] **Context linting** -- static analysis to detect common anti-patterns

### Phase 6: Self-Improving Context (v2.0.0) -- Long Term

**Goal**: Contexts that optimize themselves, informed by frontier research.

- [ ] **Agentic Context Engineering (ACE)** -- contexts as evolving playbooks that accumulate strategies (arXiv:2510.04618)
- [ ] **Self-Route hybrid** -- automatically route queries to RAG vs full-context based on complexity (arXiv:2407.16833)
- [ ] **Automated compression selection** -- choose optimal compression strategy per block type
- [ ] **Context window effectiveness tracking** -- measure actual effective window vs theoretical max (arXiv:2509.21361)
- [ ] **Cross-session context transfer** -- learn optimal context patterns from successful agent runs
- [ ] **Benchmark integration** -- continuous evaluation against LOCA-bench (arXiv:2602.07962) and MultiChallenge (arXiv:2501.17399)

### Timeline Summary

| Phase | Version | Focus | Timeframe |
|-------|---------|-------|-----------|
| 1 | v0.7.0 | Developer Experience | Near-term |
| 2 | v0.8.0 | Performance & Production | Near-term |
| 3 | v0.9.0 | Advanced Context Intelligence | Medium-term |
| 4 | v1.0.0 | Ecosystem Integration | Medium-term |
| 5 | v1.1.0 | Observability & Dev Tools | Medium-term |
| 6 | v2.0.0 | Self-Improving Context | Long-term |

---

## 9. References

### Industry Sources

- [Anthropic -- Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Philipp Schmid -- The New Skill in AI is Not Prompting, It's Context Engineering](https://www.philschmid.de/context-engineering)
- [Simon Willison -- Context Engineering](https://simonwillison.net/2025/jun/27/context-engineering/)
- [LangChain -- Context Engineering for Agents](https://blog.langchain.com/context-engineering-for-agents/)
- [Spotify Engineering -- Background Coding Agents: Context Engineering](https://engineering.atspotify.com/2025/11/context-engineering-background-coding-agents-part-2)
- [Weaviate -- Context Engineering: LLM Memory and Retrieval](https://weaviate.io/blog/context-engineering)
- [O'Reilly -- Context Engineering: Bringing Engineering Discipline to Prompts](https://www.oreilly.com/radar/context-engineering-bringing-engineering-discipline-to-prompts-part-1/)

### arXiv Papers

| Paper | arXiv ID |
|-------|----------|
| A Survey of Context Engineering for LLMs | 2507.13334 |
| Context Engineering 2.0 | 2510.26493 |
| Agentic Context Engineering (ACE) | 2510.04618 |
| Invasive Context Engineering | 2512.03001 |
| Maximum Effective Context Window | 2509.21361 |
| LongRoPE2 | 2502.20082 |
| InfiniteICL | 2504.01707 |
| Solving Context Window Overflow | 2511.22729 |
| How to Train Long-Context LLMs | 2410.02660 |
| Beyond the Limits: Context Extension Survey | 2402.02244 |
| RAG for LLMs: A Survey | 2312.10997 |
| RAG or Long-Context LLMs? | 2407.16833 |
| Stronger Baselines for RAG | 2506.03989 |
| Prompt Compression Survey | 2410.12388 |
| LLMLingua-2 | 2403.12968 |
| SAC (Semantic Anchors) | 2510.08907 |
| COMI: Coarse-to-fine Compression | 2602.01719 |
| Semantic Compression | 2312.09571 |
| Agentic Memory (AgeMem) | 2601.01885 |
| A-Mem | 2502.12110 |
| Mem0 | 2504.19413 |
| Memory in the Age of AI Agents | 2512.13564 |
| LLMs Get Lost In Multi-Turn | 2505.06120 |
| Context Drift | 2510.07777 |
| MultiChallenge Benchmark | 2501.17399 |
| ARTIST | 2505.01441 |
| Dynamic Tool Dependency Retrieval | 2512.17052 |
| ScaleMCP | 2505.06416 |
| LOCA-bench | 2602.07962 |
| KV Cache Management Survey | 2412.19442 |
| LMCache | 2510.09665 |
| CA-MCP | 2601.11595 |
| MCP-Universe | 2508.14704 |
| Agent Interoperability Survey | 2505.02279 |
| The Prompt Report | 2406.06608 |
| Multi-turn Dialogue Systems Survey | 2402.18013 |

---

*Research compiled February 2026*
