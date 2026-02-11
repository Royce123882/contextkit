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

## Suggested Release Plan

| Version | Focus | Key Deliverables |
|---------|-------|------------------|
| **v0.7.0** | Hardening | Security fixes, retry logic, pagination, custom exceptions, narrower exception catching |
| **v0.8.0** | Scalability | PostgreSQL + pgvector backend, connection pooling, health checks, structured logging |
| **v0.9.0** | Operations | Configuration management, metrics/observability, type safety cleanup |
| **v1.0.0** | Production GA | Graceful degradation, full documentation, migration guide, performance benchmarks |
