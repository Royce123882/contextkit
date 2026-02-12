# Plan: LLM Compaction with Pluggable Storage (Local + S3)

## Summary

Replace the current truncation-based `CompactStep` with an LLM-based compactor that:
1. Splits content into numbered paragraphs
2. Saves the original to a pluggable `CompactionStore` (local filesystem or S3)
3. Calls a user-provided LLM to produce a summary with `[N]` reference pointers
4. Replaces block content with the compact summary

Remove `CompressStep` entirely (IDF token pruning is redundant with LLM compaction).

---

## Files Changed

### Delete
- `src/contextkit/pipeline/compress.py`

### New Files
- `src/contextkit/compaction/__init__.py` — package exports + lazy S3 import
- `src/contextkit/compaction/store.py` — `CompactionStore` protocol + `LocalCompactionStore`
- `src/contextkit/compaction/s3_store.py` — `S3CompactionStore`

### Modified Files
- `src/contextkit/pipeline/compact.py` — rewrite with LLM logic
- `src/contextkit/pipeline/__init__.py` — remove `CompressStep`
- `src/contextkit/pipeline/pipeline.py` — remove `CompressStep` import, update `aggressive()` preset
- `src/contextkit/__init__.py` — remove `CompressStep`, add `CompactionStore`, `LocalCompactionStore`
- `src/contextkit/constants.py` — remove `DEFAULT_COMPRESSION_RATIO`, add LLM compact constants
- `pyproject.toml` — add `s3 = ["aiobotocore>=2.7.0"]` optional dependency
- `tests/test_pipeline.py` — remove `TestTokenLevelCompressionStep`, rewrite `TestCompactStep` + `TestCompactionCollapseDetection`
- `tests/test_compaction_store.py` — new test file for store implementations

---

## Step 1: `CompactionStore` protocol + `LocalCompactionStore`

**File: `src/contextkit/compaction/store.py`**

```python
@runtime_checkable
class CompactionStore(Protocol):
    """Protocol for compaction artifact storage.

    Stores the original numbered content as markdown so that
    [N] references in the compacted output can be resolved.
    """

    async def save(self, key: str, content: str, metadata: Dict[str, Any] | None = None) -> str:
        """Save original content. Returns a reference URI (path or URL)."""
        ...

    async def load(self, key: str) -> str | None:
        """Load original content by key. Returns None if not found."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete a stored artifact. Returns True if deleted."""
        ...

    async def list_keys(self) -> List[str]:
        """List all stored artifact keys."""
        ...
```

```python
class LocalCompactionStore:
    """Filesystem-backed compaction store.

    Saves markdown files to a configurable directory.
    Default: `.contextkit/compacted/` relative to cwd.

    Args:
        base_dir: Directory for storing compaction artifacts.
    """

    def __init__(self, base_dir: str = ".contextkit/compacted") -> None:
        self._base_dir = Path(base_dir)

    async def save(self, key: str, content: str, metadata: Dict[str, Any] | None = None) -> str:
        # os.makedirs(self._base_dir, exist_ok=True)
        # path = self._base_dir / f"{key}.md"
        # path.write_text(content, encoding="utf-8")
        # return str(path)
        ...

    async def load(self, key: str) -> str | None:
        # path = self._base_dir / f"{key}.md"
        # return path.read_text() if path.exists() else None
        ...

    async def delete(self, key: str) -> bool:
        # path = self._base_dir / f"{key}.md"
        # path.unlink(missing_ok=True)
        ...

    async def list_keys(self) -> List[str]:
        # glob *.md, strip extension
        ...
```

Sync wrappers: `save()` is async to match the S3 path but `LocalCompactionStore`
uses synchronous file I/O internally (same pattern as `SQLiteBackend`).

---

## Step 2: `S3CompactionStore`

**File: `src/contextkit/compaction/s3_store.py`**

Follows the exact same pattern as `ChromaRetriever`, `PgvectorRetriever`, etc:
- Lazy import of `aiobotocore` with clear error message
- `TYPE_CHECKING` guard for type hints
- Listed in `pyproject.toml` as optional extra: `s3 = ["aiobotocore>=2.7.0"]`

```python
class S3CompactionStore:
    """S3-backed compaction store.

    Stores compaction artifacts as markdown objects in an S3 bucket.

    Requires: pip install contextkit[s3]

    Args:
        bucket: S3 bucket name.
        prefix: Key prefix within the bucket (default: "contextkit/compacted/").
        region: AWS region (default: from environment).
        session: Optional aiobotocore AioSession for custom credentials.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "contextkit/compacted/",
        region: str | None = None,
        session: Any = None,
    ) -> None:
        _import_aiobotocore()  # Fail fast
        self._bucket = bucket
        self._prefix = prefix
        self._region = region
        self._session = session

    async def save(self, key: str, content: str, metadata: Dict[str, Any] | None = None) -> str:
        # PutObject to s3://{bucket}/{prefix}{key}.md
        # Returns s3:// URI
        ...

    async def load(self, key: str) -> str | None:
        # GetObject, return body as string
        # Return None on NoSuchKey
        ...

    async def delete(self, key: str) -> bool:
        # DeleteObject
        ...

    async def list_keys(self) -> List[str]:
        # ListObjectsV2 with prefix, strip prefix and .md suffix
        ...
```

---

## Step 3: `compaction/__init__.py`

**File: `src/contextkit/compaction/__init__.py`**

Follows the same lazy-import pattern as `memory/__init__.py` and `rag/__init__.py`:

```python
from contextkit.compaction.store import CompactionStore, LocalCompactionStore

__all__ = [
    "CompactionStore",
    "LocalCompactionStore",
    "S3CompactionStore",
]

def __getattr__(name: str) -> type:
    """Lazy-import S3 backend to avoid hard dependency on aiobotocore."""
    if name == "S3CompactionStore":
        from contextkit.compaction.s3_store import S3CompactionStore
        return S3CompactionStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

---

## Step 4: Rewrite `CompactStep`

**File: `src/contextkit/pipeline/compact.py`**

Replace the entire implementation. The class keeps its name `CompactStep`.

### Constructor:

```python
class CompactStep(PipelineStep):
    """LLM-based compaction with reference pointers to original content.

    Splits block content into numbered paragraphs, saves the original
    to a CompactionStore, and calls an LLM to produce a concise summary
    using [N] references to cite original sections.

    Args:
        llm: Sync callable (prompt: str) -> str. User provides their
            own LLM integration (Anthropic, OpenAI, etc.).
        async_llm: Async callable for use with pipeline.arun().
            If only llm is provided, async_process wraps it.
        store: CompactionStore for saving originals. Defaults to
            LocalCompactionStore(".contextkit/compacted").
        min_tokens: Only compact blocks above this token count.
        max_info_loss: Max keyword loss (0.0-1.0). If exceeded,
            returns original block unmodified (collapse detection).
        paragraph_separator: How to split content into sections.
            Default: double newline ("\n\n").
    """

    def __init__(
        self,
        llm: Callable[[str], str] | None = None,
        async_llm: Callable[[str], Awaitable[str]] | None = None,
        store: CompactionStore | None = None,
        min_tokens: int = DEFAULT_LLM_COMPACT_MIN_TOKENS,
        max_info_loss: float = 0.5,
        paragraph_separator: str = "\n\n",
        **kwargs: Any,
    ) -> None:
```

When `llm` is None, falls back to the old truncation behavior (backwards compatible
for users who don't want LLM costs).

### Key methods:

- `process(blocks)` — sync, iterates blocks
- `async_process(blocks)` — async override, uses `async_llm` when available
- `_compact_block(block)` / `_compact_block_async(block)` — per-block orchestration:
  1. Skip if non-string or below min_tokens
  2. `_split_paragraphs(content)` → list of strings
  3. `_build_numbered_content(paragraphs)` → `"## [1]\n...\n\n## [2]\n..."`
  4. `_build_markdown(block, numbered_content)` → full markdown with header
  5. `_save_original(block, markdown)` → call `store.save()`, get reference URI
  6. `_build_prompt(numbered_content)` → wrap with compaction instructions
  7. Call `llm(prompt)` → get compacted text with `[N]` references
  8. Collapse detection via `word_overlap_score`
  9. Update block content, set `metadata["compaction_ref"]` to URI
  10. Record `Mutation` with before/after + URI in detail

### The LLM prompt:

```
Summarize the following content concisely. Use [N] references to cite
the original sections by their number. Preserve key technical terms,
names, and numbers. Do not introduce information not in the original.

## [1]
First paragraph content...

## [2]
Second paragraph content...
```

### The saved markdown format:

```markdown
# Compacted Context: {block.display_name}
# Block type: {block.type.value}
# Saved: {ISO timestamp}

## [1]
First paragraph...

## [2]
Second paragraph...
```

### Backwards compatibility:

When `llm=None` and `async_llm=None`, the step falls back to
simple truncation (same as the old default behavior). This means
existing users who do `CompactStep(target_ratio=0.3)` still work.
The `target_ratio` parameter is kept for this fallback path only.

---

## Step 5: Remove `CompressStep`

### Delete file:
- `src/contextkit/pipeline/compress.py`

### Update `src/contextkit/pipeline/__init__.py`:
- Remove `from contextkit.pipeline.compress import CompressStep`
- Remove `"CompressStep"` from `__all__`

### Update `src/contextkit/pipeline/pipeline.py`:
- Remove `from contextkit.pipeline.compress import CompressStep`
- Update `aggressive()` preset: replace `CompressStep(compression_ratio=0.5)`
  with `CompactStep()` (no LLM by default, uses truncation fallback)

### Update `src/contextkit/__init__.py`:
- Remove `CompressStep` from pipeline import block
- Remove `"CompressStep"` from `__all__`
- Add `CompactionStore` and `LocalCompactionStore` imports from `contextkit.compaction`

### Update `src/contextkit/constants.py`:
- Remove `DEFAULT_COMPRESSION_RATIO`
- Add:
  ```python
  DEFAULT_LLM_COMPACT_MIN_TOKENS = 200
  DEFAULT_COMPACTION_STORAGE_DIR = ".contextkit/compacted"
  ```

### Update `pyproject.toml`:
- Add `s3 = ["aiobotocore>=2.7.0"]` to `[project.optional-dependencies]`
- Add `"aiobotocore"` to mypy overrides `ignore_missing_imports`
- Update `all` extra to include `s3`

---

## Step 6: Tests

### `tests/test_pipeline.py` — modifications:

**Remove:**
- `TestTokenLevelCompressionStep` class entirely (7 test methods)
- `CompressStep` from imports

**Rewrite `TestCompactStep`:**
```python
class TestCompactStep:
    def test_compacts_with_llm_and_saves_original(self):
        """Mock LLM returns summary with [N] refs, original saved to store."""

    def test_fallback_truncation_when_no_llm(self):
        """Without llm= arg, uses truncation (backwards compatible)."""

    def test_skips_short_blocks(self):
        """Blocks below min_tokens pass through unchanged."""

    def test_skips_non_string_content(self):
        """List content (message history) passes through unchanged."""

    def test_records_mutation_with_store_reference(self):
        """Mutation detail includes the compaction store URI."""

    def test_metadata_contains_compaction_ref(self):
        """block.metadata["compaction_ref"] is set to the store URI."""

    def test_numbered_paragraphs_in_saved_markdown(self):
        """Saved markdown contains ## [1], ## [2], etc. headers."""

    def test_step_name(self):
        """Step name is 'CompactStep'."""
```

**Rewrite `TestCompactionCollapseDetection`:**
```python
class TestCompactionCollapseDetection:
    def test_returns_original_when_llm_loses_keywords(self):
        """LLM that drops all keywords triggers collapse detection."""

    def test_no_warning_when_keywords_retained(self):
        """Good keyword retention proceeds normally."""

    def test_max_info_loss_threshold_is_configurable(self):
        """max_info_loss parameter is respected."""
```

### `tests/test_compaction_store.py` — new file:

```python
class TestLocalCompactionStore:
    async def test_save_creates_markdown_file(self):
    async def test_load_returns_saved_content(self):
    async def test_load_returns_none_for_missing_key(self):
    async def test_delete_removes_file(self):
    async def test_list_keys_returns_all_saved(self):
    async def test_save_creates_directory_lazily(self):

class TestCompactionStoreProtocol:
    def test_local_store_satisfies_protocol(self):
        """LocalCompactionStore is runtime-checkable against CompactionStore."""
```

S3 tests would use mocking (or be marked as integration tests requiring credentials).

---

## Step 7: Wire exports

### `src/contextkit/compaction/__init__.py`:
Export `CompactionStore`, `LocalCompactionStore`, lazy-import `S3CompactionStore`.

### `src/contextkit/__init__.py`:
Add to imports and `__all__`:
```python
from contextkit.compaction import CompactionStore, LocalCompactionStore
```

---

## Architecture Diagram

```
User code
    │
    │  CompactStep(
    │      llm=my_summarizer,
    │      store=S3CompactionStore(bucket="my-bucket"),
    │  )
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  CompactStep.process(blocks)                        │
│                                                     │
│  For each block above min_tokens:                   │
│                                                     │
│  ① Split into paragraphs                            │
│  ② Build numbered content: ## [1] ... ## [2] ...    │
│  ③ store.save(key, markdown) → URI                  │
│  ④ llm(prompt_with_numbered_content) → summary      │
│  ⑤ Collapse detection (word_overlap_score)          │
│  ⑥ Replace block.content, set metadata, log Mutation│
└────────────────┬──────────────────────────────────┘
                 │
     ┌───────────┴───────────────┐
     ▼                           ▼
┌──────────────────┐   ┌───────────────────────┐
│ LocalCompaction  │   │ S3CompactionStore     │
│ Store            │   │                       │
│                  │   │ bucket: my-bucket     │
│ .contextkit/     │   │ prefix: contextkit/   │
│ compacted/       │   │         compacted/    │
│ block_abc.md     │   │ block_abc.md          │
└──────────────────┘   └───────────────────────┘
     (default)              (pip install
                             contextkit[s3])
```

---

## Dependency & Execution Order

```
Step 1: CompactionStore protocol + LocalCompactionStore
    │
    ├──► Step 2: S3CompactionStore (independent of Step 3/4)
    │
    └──► Step 3: compaction/__init__.py (needs Step 1, optionally Step 2)
            │
            ▼
         Step 4: Rewrite CompactStep (needs Step 1 + Step 3)
            │
            ▼
         Step 5: Remove CompressStep + update all imports
            │
            ▼
         Step 6: Tests (needs Steps 4 + 5)
            │
            ▼
         Step 7: Wire top-level exports + pyproject.toml
```

Steps 1 and 2 can be done in parallel.
Steps 4 and 5 can be done in parallel (no dependency between rewrite and removal).
