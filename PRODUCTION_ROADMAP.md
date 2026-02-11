# contextkit -- Production Roadmap

Analysis of changes required to make contextkit production-ready. Organized by priority and effort.

---

## 1. Database: PostgreSQL + pgvector Backend

The current backends (InMemoryBackend, SQLiteBackend) are suitable for development and single-process use. Production workloads require PostgreSQL with pgvector for vector similarity search.

### What to Build

```
src/contextkit/memory/
    postgres_backend.py      # asyncpg + pgvector backend
```

### Interface Changes

The existing `MemoryBackend` protocol needs these additions:

```python
# New method: semantic retrieval via embeddings
async def retrieve_semantic(
    self,
    query_vector: list[float],
    top_k: int = 5,
    distance_metric: str = "cosine",
) -> list[MemoryRecord]:
    """Retrieve by vector similarity (pgvector <-> operator)."""

# New method: batch operations for throughput
async def store_batch(
    self,
    records: list[tuple[str, str, dict, list[str], float]],
) -> list[MemoryRecord]:
    """Batch insert with transaction support."""

# New: connection lifecycle
async def connect(self) -> None:
    """Initialize connection pool."""

async def disconnect(self) -> None:
    """Cleanup connections."""
```

### Implementation Notes

- Use `asyncpg` (not `psycopg2`) for true async I/O
- Connection pooling via `asyncpg.create_pool(min_size=5, max_size=20)`
- Store embeddings in `vector(1536)` column (pgvector extension)
- Add IVFFlat index: `CREATE INDEX ON memory_records USING ivfflat (embedding vector_cosine_ops)`
- Add schema migration support (Alembic or hand-rolled versioned SQL)
- New optional dependency group: `postgres = ["asyncpg>=0.29.0", "pgvector>=0.2.0"]`

### SQLiteBackend Scalability Fix (Immediate)

The current `retrieve()` method loads **all records into memory** then filters:

```python
# sqlite_backend.py line 161 — loads entire DB
records = await self._fetch_all_records()
```

**Fix**: Push filtering into SQL WHERE clauses. Add `LIMIT` and `OFFSET` for pagination.

---

## 2. Async Patterns

### 2a. Connection Pooling (Critical)

Every SQLiteBackend operation opens a new connection. Add pooling:

```python
class SQLiteBackend:
    async def initialize(self) -> None:
        """Create persistent connection (call once at startup)."""
        self._connection = await aiosqlite.connect(self._db_path)
        await self._connection.execute(_CREATE_TABLE_SQL)
        await self._connection.commit()

    async def close(self) -> None:
        """Close the connection (call at shutdown)."""
        if self._connection:
            await self._connection.close()
```

For PostgreSQL, use `asyncpg.create_pool()` with configurable pool size.

### 2b. O(n²) Message Trimming (Critical)

`short_term.py` line 177 uses `list.pop(0)` which is O(n) per call, making the trim loop O(n²):

```python
# Current — O(n²)
while self._messages and count_tokens(...) > self._max_tokens:
    self._messages.pop(0)  # O(n) shift

# Fix — O(1) with deque
from collections import deque
self._message_buffer: deque = deque(maxlen=self._max_turns)
# Then: self._message_buffer.popleft()  # O(1)
```

### 2c. Sync-to-Async Bridge Improvement

`sync.py` uses `concurrent.futures.ThreadPoolExecutor` which can deadlock when called from within an existing event loop. Consider using `anyio` for a robust sync/async bridge, or document the limitation.

### 2d. Missing Pagination

`list_records()` on all backends returns all records at once. Add:

```python
async def list_records(
    self,
    tags: list[str] | None = None,
    offset: int = 0,
    limit: int = 100,
) -> list[MemoryRecord]:
```

---

## 3. Security Hardening

### 3a. Path Traversal in FileContext (Critical)

`file_context.py` does not validate that user-provided paths stay within the `base_path`:

```python
# load_single() accepts any path — vulnerable to traversal
def load_single(self, path: str, ...) -> ContextBlock:
    filepath = Path(path)
    content = filepath.read_text(...)  # Could read /etc/passwd
```

**Fix**: Validate paths resolve under the allowed base directory:

```python
resolved = Path(path).resolve()
allowed = self._base_path.resolve()
if not str(resolved).startswith(str(allowed)):
    raise ValueError(f"Path {path} is outside allowed directory {allowed}")
```

Also in `scan()`, follow symlinks cautiously — add `followlinks=False` to `os.walk()`.

### 3b. Input Validation on MemoryRecord

`MemoryRecord` accepts any `importance` value — should be bounded [0.0, 1.0]:

```python
@field_validator('importance')
def validate_importance(cls, v: float) -> float:
    if not 0.0 <= v <= 1.0:
        raise ValueError('importance must be between 0.0 and 1.0')
    return v
```

### 3c. JSON Deserialization Safety

`sqlite_backend.py` line 252 calls `json.loads()` on database content without validation. Add depth/size limits or validate through Pydantic models.

---

## 4. Error Handling Improvements

### 4a. Overly Broad Exception Catching

`token_counting.py` line 37 catches bare `Exception`:

```python
# Current
except Exception:
    return None

# Fix — catch only expected failures
except (ImportError, KeyError, ValueError):
    return None
```

### 4b. Silent File I/O Failures

`file_context.py` logs skipped files at `DEBUG` level only. Users won't see failures unless they enable debug logging. Change to `WARNING` for permission errors, and return skipped file info:

```python
def load(self, refs, ...) -> tuple[list[ContextBlock], list[str]]:
    """Returns (blocks, skipped_file_paths)."""
```

### 4c. Custom Exception Hierarchy

Replace generic `KeyError`/`ValueError` with domain-specific exceptions:

```python
class ContextKitError(Exception): pass
class BackendConnectionError(ContextKitError): pass
class BackendTimeoutError(ContextKitError): pass
class RecordNotFoundError(ContextKitError): pass
class InvalidBlockError(ContextKitError): pass
class PathTraversalError(ContextKitError): pass
```

---

## 5. Retry Logic and Resilience

### 5a. Retry on Transient Failures

All database and network operations need retry with exponential backoff:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def retrieve(self, query: str, ...) -> list[MemoryRecord]:
    ...
```

New dependency: `tenacity>=8.0` (or implement a lightweight decorator).

### 5b. Graceful Degradation

Add a `ResilientBackend` wrapper that falls back to a secondary backend on failure:

```python
class ResilientBackend:
    def __init__(self, primary: MemoryBackend, fallback: MemoryBackend):
        ...

    async def retrieve(self, query, ...):
        try:
            return await asyncio.wait_for(
                self.primary.retrieve(query, ...), timeout=5.0
            )
        except (asyncio.TimeoutError, BackendConnectionError):
            return await self.fallback.retrieve(query, ...)
```

### 5c. Health Checks on All Backends

Currently only `RetrieverBackend` has `health_check()`. Add to `MemoryBackend` protocol:

```python
async def health_check(self) -> bool:
    """Return True if the backend is reachable and operational."""
```

---

## 6. Observability and Metrics

### 6a. Structured Logging

Replace basic `logging.getLogger()` with structured JSON logging:

```python
import structlog
logger = structlog.get_logger("contextkit")

logger.info("block_added", block_name=block.display_name, tokens=block.token_count)
```

### 6b. Metrics Export

Add optional Prometheus metrics:

```python
from prometheus_client import Counter, Histogram

TOKENS_COUNTED = Counter("contextkit_tokens_counted_total", "Total tokens counted")
RETRIEVAL_LATENCY = Histogram("contextkit_retrieval_seconds", "Retrieval latency")
BLOCKS_ASSEMBLED = Counter("contextkit_blocks_assembled_total", "Blocks assembled")
```

### 6c. Request/Trace ID Propagation

Use `contextvars` to propagate request IDs through async call chains:

```python
import contextvars
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id")
```

---

## 7. Configuration Management

Replace hardcoded constants with a configuration system:

```python
# contextkit.yaml
memory:
  backend: postgresql
  connection_string: ${DATABASE_URL}
  pool_size: 10

retrieval:
  default_top_k: 5
  timeout_ms: 5000

tokens:
  cache_size: 8192
  default_encoding: cl100k_base
```

Support loading from:
1. YAML config file
2. Environment variables (12-factor app)
3. Programmatic override

---

## 8. Type Safety Improvements

### 8a. Replace `Any` with Specific Types

| Location | Current | Recommended |
|----------|---------|-------------|
| `context_window.py:49` | `_assembly_report: Any` | `AssemblyReport \| None` |
| `block.py:51` | `content: str \| List[Dict[str, Any]]` | `str \| List[MessageDict]` |
| `provenance.py:29` | `details: Dict[str, Any]` | `TypedDict` per source type |

### 8b. Add MessageDict TypedDict

```python
from typing import TypedDict

class MessageDict(TypedDict):
    role: str
    content: str
```

---

## 9. Naming Improvements

| Location | Current | Suggested | Reason |
|----------|---------|-----------|--------|
| `context_window.py:357` | `add_unchecked()` | `add_without_budget_check()` | Clearer intent |
| `in_memory_backend.py:54` | `results` | `matching_records` | More descriptive |
| `short_term.py:47` | `_messages` | `_message_buffer` | Clarifies it's a buffer with trimming |
| `file_context.py:93` | `scored` | `scored_references` | Type clarity |

---

## 10. Priority Matrix

| # | Item | Severity | Effort | Phase |
|---|------|----------|--------|-------|
| 1 | Path traversal fix in FileContext | CRITICAL | Low | Immediate |
| 2 | O(n²) message trimming fix | CRITICAL | Low | Immediate |
| 3 | Connection pooling for backends | CRITICAL | Medium | Immediate |
| 4 | SQLite retrieve: push filters to SQL | HIGH | Medium | v0.7 |
| 5 | Add pagination to list operations | HIGH | Medium | v0.7 |
| 6 | Add aiosqlite to dev deps | HIGH | Low | Done |
| 7 | Retry logic for backend operations | HIGH | Medium | v0.7 |
| 8 | Custom exception hierarchy | HIGH | Low | v0.7 |
| 9 | Input validation (MemoryRecord) | HIGH | Low | v0.7 |
| 10 | Health checks on all backends | HIGH | Medium | v0.8 |
| 11 | PostgreSQL + pgvector backend | HIGH | High | v0.8 |
| 12 | Structured logging | MEDIUM | Medium | v0.8 |
| 13 | Configuration management | MEDIUM | High | v0.9 |
| 14 | Prometheus metrics | MEDIUM | High | v0.9 |
| 15 | Type safety improvements | MEDIUM | Medium | v0.9 |
| 16 | Graceful degradation patterns | MEDIUM | High | v1.0 |
| 17 | Naming improvements | LOW | Low | Ongoing |

---

## 11. Vector Database Retriever Backends (Done)

Four production retriever backends have been implemented, each
satisfying the existing `RetrieverBackend` protocol (async `retrieve` + `health_check`).

| Backend | Module | Optional Extra | Key Features |
|---------|--------|----------------|--------------|
| **Qdrant** | `rag/qdrant_retriever.py` | `pip install contextkit[qdrant]` | AsyncQdrantClient, local `:memory:` mode for dev, cosine scoring, metadata passthrough |
| **ChromaDB** | `rag/chroma_retriever.py` | `pip install contextkit[chroma]` | Embedded-first (no server), auto-embedding, lazy collection init |
| **pgvector** | `rag/pgvector_retriever.py` | `pip install contextkit[pgvector]` | asyncpg connection pool, cosine distance via `<=>`, configurable table/columns |
| **Pinecone** | `rag/pinecone_retriever.py` | `pip install contextkit[pinecone]` | Fully managed, namespace support, score normalization |

All four follow the same patterns:
- Lazy dependency import (fail-fast with helpful `ImportError`)
- Score normalization to `[0.0, 1.0]`
- Metadata extraction into `Chunk.metadata`
- `health_check()` returns `False` on any exception (logged at DEBUG)

### Future backends to consider

| Database | Priority | Rationale |
|----------|----------|-----------|
| **Milvus / Zilliz** | Medium | Best for billion-vector scale; heavy deploy (etcd + MinIO) |
| **Weaviate** | Medium | Built-in vectorization modules, GraphQL API |
| **LanceDB** | Low (wait for 1.0) | Promising embedded columnar format, still pre-1.0 |
| **Redis Vector Search** | Low | Good if Redis already deployed; `redisvl` SDK still maturing |
| **Elasticsearch / OpenSearch** | Low | Excellent hybrid search; very heavy dependency |

---

## 12. Research-Backed Context Engineering Features

Features informed by academic research on context engineering for LLMs.
Organized by priority tier.

### Tier 1: High Priority (v0.8-v0.9)

#### 12a. Observation Masking -- `MaskStep` (Done)

**Research**: JetBrains Research, "Cutting Through the Noise" (2025) -- observation masking matches LLM summarization in cost savings and task-solving.

**Implementation**: `pipeline/mask.py` -- replaces older block content with a short placeholder text while keeping the block in the sequence. Configurable recency window, type filters, and minimum-token threshold. Records `Mutation` objects with before/after token counts.

#### 12b. Query-Aware Context Pruning

**Research**: DYCP (Choi et al., 2025, arXiv:2601.07994) -- dynamically segments and retrieves relevant memory at query time, reducing first-token latency by ~3x.

**Implementation plan**: Extend `TrimStep` with an optional `query` parameter. When provided, score each block's relevance to the query (via word overlap or an embedding function) and prune by combined priority + relevance instead of priority alone. Also extend `ShortTermMemory` to support query-aware trimming (retain relevant older turns, drop irrelevant recent ones).

#### 12c. Prefix-Aware Context Structuring

**Research**: KV cache optimization survey (arXiv:2407.18003), Anthropic prompt caching, Google context caching.

**Implementation plan**: A `PrefixOptimizer` utility that analyzes a set of context blocks and reorders them so that stable, shared content (system prompts, tool definitions) appears first as a stable prefix. This maximizes cache hits on providers that support prefix caching. Would integrate with the existing `ReorderStep` as an alternative strategy.

#### 12d. Post-Retrieval Compression for RAG

**Research**: RECOMP (Xu et al., 2023, arXiv:2310.04408) -- extractive/abstractive compression between retrieval and generation, with selective augmentation (drop irrelevant chunks entirely).

**Implementation plan**: A `RAGCompressStep` that takes retrieved chunks, optionally extracts key sentences or generates summaries, and implements selective augmentation. Would compose with the existing `FilterStep` and `CompactStep` but be RAG-aware (uses relevance scores from Origin).

### Tier 2: Medium Priority (v0.9-v1.0)

#### 12e. Token-Level Prompt Compression

**Research**: LLMLingua (Jiang et al., 2023, arXiv:2310.05736) and LLMLingua-2 (2024, arXiv:2403.12968) -- coarse-to-fine token pruning using perplexity from a small proxy model. Up to 20x compression with minimal loss.

**Implementation plan**: A `CompressStep` (distinct from the existing `CompactStep`) that uses a small local model to compute per-token information scores and prune low-information tokens. Accepts a `compression_ratio` parameter. Requires an optional dependency on a small model (e.g. GPT-2 via transformers).

#### 12f. Context Quality Scoring

**Research**: "Lost in the Middle" (Liu et al., 2023, arXiv:2307.03172) -- U-shaped performance curve where models perform best when key info is at start/end. NeedleBench (arXiv:2407.11963) -- multi-needle evaluation framework.

**Implementation plan**: A `QualityScorer` in the `observe` module that estimates the "findability" of key information given its position and total context length. Would compute a signal-to-noise ratio and warn when important blocks are buried in the middle. Integrates with `ContextWindow.explain()`.

#### 12g. Context Sufficiency Checking

**Research**: Google Research, "Sufficient Context" (ICLR 2025) -- quantifies whether context is "enough" to answer correctly. Recommends sufficiency check before generation.

**Implementation plan**: A `SufficiencyChecker` that estimates whether assembled context covers the query. Uses heuristics (query term coverage, relevance score distribution) or an optional LLM call. Emits a new `CONTEXT_INSUFFICIENT` event when confidence is low.

#### 12h. Memory Decay (Ebbinghaus Forgetting Curve)

**Research**: MemoryBank (Zhong et al., 2023, arXiv:2305.10250) -- external memory with Ebbinghaus forgetting curve for memory decay. M+ (Wang et al., 2025, arXiv:2502.00592) -- co-trained retriever for long-term memory.

**Implementation plan**: Extend `MemoryRecord` with `access_count` and `last_accessed` fields. Add a `decay_score()` method that combines semantic similarity with a temporal decay factor (exponential decay based on time since last access). Update `LongTermMemory.retrieve()` scoring to incorporate decay.

#### 12i. Collapse Detection for Summaries

**Research**: "Agentic Context Engineering: Learning Compositional Abstractions" (2025) -- identifies **context collapse** where accumulated context compresses into shorter, less informative summaries.

**Implementation plan**: Extend `CompactStep` with a collapse detection check that compares information density (entity count, unique terms) of summaries vs. originals. Warn or retry when too much information is lost. Add a `max_info_loss` parameter.

### Tier 3: Lower Priority (v1.0+)

#### 12j. Prompt Optimization Loop

**Research**: OPRO (Yang et al., 2023, arXiv:2309.03409) -- LLMs as optimizers for prompts. SPRIG (2024, arXiv:2410.14826) -- edit-based genetic algorithm for system prompts.

**Implementation plan**: A `PromptOptimizer` companion to `PromptManager` that implements an evolutionary optimization loop: generate prompt variants, score them against an eval function, converge on a high-performing prompt. Uses the existing prompt versioning system.

#### 12k. Teacher-Student Cascade

**Research**: In-Context Distillation with Self-Consistency Cascades (Sarukkai et al., 2025, arXiv:2512.02543) -- training-free approach that retrieves teacher demonstrations for a cheaper student model.

**Implementation plan**: A `CascadeStep` that detects when context exceeds the target model's optimal range and switches to a cascade pattern: use a larger model's outputs for expensive context blocks as compact few-shot examples for a smaller model. Leverages `ExampleStore` for demonstration storage.

---

## 13. Priority Matrix (Updated)

| # | Item | Severity | Effort | Phase | Status |
|---|------|----------|--------|-------|--------|
| 1 | Path traversal fix in FileContext | CRITICAL | Low | Immediate | |
| 2 | O(n²) message trimming fix | CRITICAL | Low | Immediate | |
| 3 | Connection pooling for backends | CRITICAL | Medium | Immediate | |
| 4 | Thread safety in events.py | CRITICAL | Low | Immediate | **Done** |
| 5 | Thread safety in shared_memory.py | CRITICAL | Low | Immediate | **Done** |
| 6 | Handler exception safety in events.py | HIGH | Low | Immediate | **Done** |
| 7 | Silent BudgetExceededError logging | HIGH | Low | Immediate | **Done** |
| 8 | In-place sort mutation in trim.py | HIGH | Low | Immediate | **Done** |
| 9 | SQLite retrieve: push filters to SQL | HIGH | Medium | v0.7 |  |
| 10 | Add pagination to list operations | HIGH | Medium | v0.7 |  |
| 11 | Retry logic for backend operations | HIGH | Medium | v0.7 |  |
| 12 | Custom exception hierarchy | HIGH | Low | v0.7 |  |
| 13 | Input validation (MemoryRecord) | HIGH | Low | v0.7 |  |
| 14 | Qdrant retriever backend | HIGH | Medium | v0.8 | **Done** |
| 15 | ChromaDB retriever backend | HIGH | Medium | v0.8 | **Done** |
| 16 | pgvector retriever backend | HIGH | Medium | v0.8 | **Done** |
| 17 | Pinecone retriever backend | HIGH | Medium | v0.8 | **Done** |
| 18 | MaskStep pipeline step | HIGH | Low | v0.8 | **Done** |
| 19 | Query-aware context pruning | HIGH | Medium | v0.8 |  |
| 20 | Prefix-aware context structuring | HIGH | Medium | v0.8 |  |
| 21 | Post-retrieval RAG compression | HIGH | Medium | v0.8 |  |
| 22 | PostgreSQL + pgvector memory backend | HIGH | High | v0.8 |  |
| 23 | Health checks on all backends | HIGH | Medium | v0.8 |  |
| 24 | Structured logging | MEDIUM | Medium | v0.8 |  |
| 25 | Token-level prompt compression | MEDIUM | High | v0.9 |  |
| 26 | Context quality scoring | MEDIUM | Medium | v0.9 |  |
| 27 | Context sufficiency checking | MEDIUM | Medium | v0.9 |  |
| 28 | Memory decay (Ebbinghaus curve) | MEDIUM | Medium | v0.9 |  |
| 29 | Collapse detection for summaries | MEDIUM | Medium | v0.9 |  |
| 30 | Configuration management | MEDIUM | High | v0.9 |  |
| 31 | Prometheus metrics | MEDIUM | High | v0.9 |  |
| 32 | Type safety improvements | MEDIUM | Medium | v0.9 |  |
| 33 | Prompt optimization loop | LOW | High | v1.0 |  |
| 34 | Teacher-student cascade | LOW | High | v1.0 |  |
| 35 | Graceful degradation patterns | MEDIUM | High | v1.0 |  |
| 36 | Naming improvements | LOW | Low | Ongoing |  |

---

## Suggested Release Plan (Updated)

| Version | Focus | Key Deliverables |
|---------|-------|------------------|
| **v0.7.0** | Hardening | Security fixes, retry logic, pagination, custom exceptions |
| **v0.8.0** | Backends + Research | Qdrant/Chroma/pgvector/Pinecone retrievers, MaskStep, query-aware pruning, prefix optimization, RAG compression, PostgreSQL memory backend |
| **v0.9.0** | Intelligence | Token-level compression, context quality scoring, sufficiency checking, memory decay, collapse detection, config management, metrics |
| **v1.0.0** | Production GA | Prompt optimization, teacher-student cascade, graceful degradation, full docs, benchmarks |

---

## Key Research References

| Paper | Year | Key Technique | contextkit Feature |
|-------|------|---------------|-------------------|
| Lost in the Middle (Liu et al.) | 2023 | U-shaped positional bias | ReorderStep, QualityScorer |
| LLMLingua (Jiang et al.) | 2023 | Perplexity-based token pruning | CompressStep |
| LLMLingua-2 | 2024 | Token classification for compression | CompressStep |
| RECOMP (Xu et al.) | 2023 | Post-retrieval extractive/abstractive compression | RAGCompressStep |
| Gist Tokens (Mu et al.) | 2023 | Prompt compression into virtual tokens | PrefixOptimizer |
| MemoryBank (Zhong et al.) | 2023 | Ebbinghaus forgetting curve | Memory decay |
| OPRO (Yang et al.) | 2023 | LLMs as prompt optimizers | PromptOptimizer |
| DYCP (Choi et al.) | 2025 | Dynamic context pruning | Query-aware TrimStep |
| Sufficient Context (Google) | 2025 | Context sufficiency quantification | SufficiencyChecker |
| ACE (Agentic Context Eng.) | 2025 | Context collapse detection | Collapse detection |
| JetBrains Obs. Masking | 2025 | Observation masking | MaskStep |
| In-Context Distillation | 2025 | Teacher-student cascades | CascadeStep |
| NeedleBench | 2024 | Multi-needle evaluation | QualityScorer |
| Sparse RAG (Zhu et al.) | 2024 | Selective document caching | RAGCompressStep |
