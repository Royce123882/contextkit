"""Core data structures for contextkit.

Contains the fundamental building blocks: BlockType enum, ContextBlock
data model, ContextWindow container, and BudgetExceeded exception.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

from contextkit._cache import clear_token_cache
from contextkit._tokens import count as count_tokens
from contextkit.constants import DEFAULT_ENCODING, PRIORITY_DEFAULT
from contextkit.models import get_model
from contextkit.observe.events import (
    BlockEventData,
    BudgetEventData,
    ContextEvent,
    emit,
)
from contextkit.observe.provenance import Mutation, Origin
from contextkit.observe.warnings import BudgetMonitor


class BlockType(enum.Enum):
    """Enumeration of the 12 recognized context types."""

    SYSTEM_PROMPT = "system_prompt"
    SHORT_TERM_MEMORY = "short_term_memory"
    LONG_TERM_MEMORY = "long_term_memory"
    FILES = "files"
    TOOL_DEFINITIONS = "tool_definitions"
    TOOL_OUTPUTS = "tool_outputs"
    RAG = "rag"
    EXAMPLES = "examples"
    OUTPUT_SCHEMAS = "output_schemas"
    SYSTEM_METADATA = "system_metadata"
    SCRATCHPAD = "scratchpad"
    USER_CONTEXT = "user_context"


class ContextBlock(BaseModel):
    """A typed unit of context with content, priority, metadata, and origin.

    Every piece of information fed to a model -- prompts, memory, files,
    tool schemas, RAG chunks -- is a ContextBlock. Blocks carry provenance
    (where they came from) and mutation history (how they were modified).

    Attributes:
        type: The kind of context this block represents.
        content: The actual content -- a string for prompts, or a list
            of message dicts for conversation history.
        priority: Higher values mean more important to keep (default 50).
        metadata: Arbitrary key-value pairs for custom data.
        name: Optional human-readable label for inspection/explain.
        origin: Where this block came from (auto-populated by SDK managers).
        mutations: History of changes made by pipeline steps.
    """

    type: BlockType
    content: str | List[Dict[str, Any]]
    priority: int = PRIORITY_DEFAULT
    metadata: Dict[str, Any] = Field(default_factory=dict)
    name: str | None = None
    origin: Origin | None = None
    mutations: List[Mutation] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def token_count(self) -> int:
        """Count tokens in this block's content using cached tokenizer."""
        encoding = self.metadata.get("encoding", DEFAULT_ENCODING)
        return count_tokens(self.content, encoding)

    @property
    def display_name(self) -> str:
        """Return the name for display, falling back to the block type."""
        return self.name or self.type.value

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        origin_str = f", origin={self.origin.source}" if self.origin else ""
        return (
            f"ContextBlock(type={self.type.value}, "
            f"name={self.display_name!r}, "
            f"tokens={self.token_count}, "
            f"priority={self.priority}"
            f"{origin_str})"
        )


class BudgetExceededError(Exception):
    """Raised when adding a block would exceed the token budget."""

    def __init__(
        self,
        block_name: str,
        block_tokens: int,
        budget_remaining: int,
        max_tokens: int,
    ) -> None:
        self.block_name = block_name
        self.block_tokens = block_tokens
        self.budget_remaining = budget_remaining
        self.max_tokens = max_tokens
        super().__init__(
            f"Block '{block_name}' ({block_tokens:,} tokens) exceeds "
            f"budget remaining ({budget_remaining:,} / {max_tokens:,} tokens)"
        )


class ContextWindow:
    """The core container representing an assembled context window.

    Holds blocks, tracks tokens, enforces budgets, emits events,
    and provides inspection/explain/diff/dump convenience methods.

    Args:
        model: Model name from the registry (provides max_tokens, encoding, pricing).
        max_tokens: Manual token limit (used if model is not specified).
        budget_warnings: Thresholds at which to warn
            (e.g. [0.75, 0.90]).
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int | None = None,
        budget_warnings: List[float] | None = None,
    ) -> None:
        self._model_name: str | None = None
        if model is not None:
            self._init_from_model(model)
        elif max_tokens is not None:
            self._init_with_manual_tokens(max_tokens)
        else:
            raise ValueError("Either 'model' or 'max_tokens' must be provided.")

        self._blocks: List[ContextBlock] = []
        self._cached_token_count: int | None = None
        self._assembly_report: Any = None  # Set by ContextAssembler

        self._budget_monitor: BudgetMonitor | None = None
        if budget_warnings:
            self._budget_monitor = BudgetMonitor(budget_warnings)

    def _init_from_model(self, model: str) -> None:
        """Configure the window from a registered model spec."""
        spec = get_model(model)
        self._model_name = model
        self._max_tokens = spec.max_context
        self._encoding = spec.encoding
        self._input_cost_per_mtok = spec.input_cost_per_mtok
        self._output_cost_per_mtok = spec.output_cost_per_mtok

    def _init_with_manual_tokens(self, max_tokens: int) -> None:
        """Configure the window with manual token limits."""
        self._model_name = None
        self._max_tokens = max_tokens
        self._encoding = DEFAULT_ENCODING
        self._input_cost_per_mtok = 0.0
        self._output_cost_per_mtok = 0.0

    @property
    def model_name(self) -> str | None:
        """The model name, or None if using manual max_tokens."""
        return self._model_name

    @property
    def max_tokens(self) -> int:
        """Maximum token budget for this window."""
        return self._max_tokens

    @property
    def encoding(self) -> str:
        """The tiktoken encoding used for token counting."""
        return self._encoding

    @property
    def blocks(self) -> List[ContextBlock]:
        """Ordered list of context blocks."""
        return list(self._blocks)

    @property
    def token_count(self) -> int:
        """Total tokens across all blocks (cached)."""
        if self._cached_token_count is None:
            self._cached_token_count = sum(b.token_count for b in self._blocks)
        return self._cached_token_count

    @property
    def budget_remaining(self) -> int:
        """Tokens remaining before budget is exhausted."""
        return max(0, self._max_tokens - self.token_count)

    @property
    def cost_estimate(self) -> float:
        """Estimated input cost in USD based on token count and model pricing."""
        if self._input_cost_per_mtok <= 0:
            return 0.0
        return self.token_count * self._input_cost_per_mtok / 1_000_000

    def cost_estimate_with_response(self, output_tokens: int) -> float:
        """Estimated total cost including expected response tokens.

        Args:
            output_tokens: Expected number of output tokens.

        Returns:
            Total estimated cost in USD.
        """
        input_cost = self.cost_estimate
        output_cost = output_tokens * self._output_cost_per_mtok / 1_000_000
        return input_cost + output_cost

    def add(self, block: ContextBlock) -> None:
        """Add a context block to the window.

        Raises BudgetExceededError if the block would overflow the budget.
        Emits BLOCK_ADDED event and runs budget monitor check.

        Args:
            block: The block to add.
        """
        block_tokens = block.token_count
        if self.token_count + block_tokens > self._max_tokens:
            self._raise_budget_exceeded(block, block_tokens)

        self._blocks.append(block)
        self._invalidate_cache()
        self._emit_block_added_event(block)
        self._check_budget_warnings()

    def remove(self, name: str) -> None:
        """Remove a block by name.

        Args:
            name: The name (or type value) of the block to remove.

        Raises:
            KeyError: If no block with the given name is found.
        """
        for i, block in enumerate(self._blocks):
            if block.display_name == name:
                removed = self._blocks.pop(i)
                self._invalidate_cache()

                emit(
                    BlockEventData(
                        event=ContextEvent.BLOCK_REMOVED,
                        block_name=removed.display_name,
                        block_type=removed.type.value,
                        token_count=removed.token_count,
                    )
                )
                return

        raise KeyError(f"No block with name '{name}' found.")

    def render(self) -> List[Dict[str, Any]]:
        """Return assembled messages list, ordered by block priority.

        System prompts are placed first, then remaining blocks sorted
        by priority (highest first). Emits WINDOW_RENDERED event.

        Returns:
            A list of message dicts ready for further formatting.
        """
        sorted_blocks = sorted(self._blocks, key=lambda b: b.priority, reverse=True)

        messages: List[Dict[str, Any]] = []
        for block in sorted_blocks:
            if isinstance(block.content, str):
                role = "system" if block.type == BlockType.SYSTEM_PROMPT else "user"
                messages.append({"role": role, "content": block.content})
            elif isinstance(block.content, list):
                messages.extend(block.content)

        emit(
            BlockEventData(
                event=ContextEvent.WINDOW_RENDERED,
                token_count=self.token_count,
                details={"block_count": len(self._blocks)},
            )
        )

        return messages

    def to_dict(self) -> Dict[str, Any]:
        """Full serialization including provenance and metadata.

        Returns:
            A dict with model info, blocks (with origin/mutations), and stats.
        """
        return {
            "model": self._model_name,
            "max_tokens": self._max_tokens,
            "encoding": self._encoding,
            "token_count": self.token_count,
            "budget_remaining": self.budget_remaining,
            "cost_estimate": self.cost_estimate,
            "blocks": [self._serialize_block(b) for b in self._blocks],
            "assembly_report": (
                self._assembly_report.model_dump(mode="json")
                if self._assembly_report
                else None
            ),
        }

    @staticmethod
    def _serialize_block(block: ContextBlock) -> Dict[str, Any]:
        """Serialize a single block to a dict for JSON output."""
        return {
            "name": block.display_name,
            "type": block.type.value,
            "priority": block.priority,
            "token_count": block.token_count,
            "content": block.content,
            "metadata": block.metadata,
            "origin": (block.origin.model_dump(mode="json") if block.origin else None),
            "mutations": [m.model_dump(mode="json") for m in block.mutations],
        }

    def inspect(
        self,
        block_name: str | None = None,
        format: str = "text",
    ) -> str:
        """Inspect the context window or a specific block.

        Args:
            block_name: If provided, drill into this specific block.
            format: Output format ("text" or "html").

        Returns:
            Formatted inspection output.
        """
        from contextkit.observe.inspect import inspect_window

        return inspect_window(self, block_name=block_name, format=format)

    def explain(self, block_name: str) -> str:
        """Explain why a block is included or excluded.

        Args:
            block_name: The name of the block to explain.

        Returns:
            A plain-language explanation.
        """
        from contextkit.observe.explain import explain_block

        return explain_block(self, block_name)

    def diff(
        self,
        other: ContextWindow,
        format: str = "text",
    ) -> str:
        """Compare this window with another.

        Args:
            other: The other ContextWindow to compare against.
            format: Output format ("text" or "html").

        Returns:
            A formatted diff showing added, removed, and changed blocks.
        """
        from contextkit.observe.diff import diff_windows

        return diff_windows(self, other, format=format)

    def dump(self, path: str) -> None:
        """Write a full JSON snapshot to a file.

        Args:
            path: The file path to write the JSON snapshot to.
        """
        from contextkit.observe.inspect import dump_window

        dump_window(self, path)

    def clear_cache(self) -> None:
        """Clear the token count cache."""
        self._invalidate_cache()
        clear_token_cache()

    def get_block(self, name: str) -> ContextBlock | None:
        """Find a block by name.

        Args:
            name: The display name to search for.

        Returns:
            The matching ContextBlock, or None if not found.
        """
        for block in self._blocks:
            if block.display_name == name:
                return block
        return None

    def _invalidate_cache(self) -> None:
        """Invalidate the cached token count."""
        self._cached_token_count = None

    def _check_budget_warnings(self) -> None:
        """Run the budget monitor if configured."""
        if self._budget_monitor is None:
            return

        largest_name, largest_tokens = self._find_largest_block()
        self._budget_monitor.check(
            token_count=self.token_count,
            max_tokens=self._max_tokens,
            largest_block_name=largest_name,
            largest_block_tokens=largest_tokens,
        )

    def _find_largest_block(self) -> Tuple[str | None, int]:
        """Find the block with the most tokens.

        Returns:
            A tuple of (block_name, token_count) for the largest block.
        """
        largest_name: str | None = None
        largest_tokens = 0
        for block in self._blocks:
            block_tokens = block.token_count
            if block_tokens > largest_tokens:
                largest_tokens = block_tokens
                largest_name = block.display_name
        return largest_name, largest_tokens

    @property
    def assembly_report(self) -> Any:
        """The assembly report, if set by a ContextAssembler."""
        return self._assembly_report

    @assembly_report.setter
    def assembly_report(self, report: Any) -> None:
        """Set the assembly report (called by ContextAssembler)."""
        self._assembly_report = report

    @property
    def input_cost_per_mtok(self) -> float:
        """Input cost per million tokens."""
        return self._input_cost_per_mtok

    def add_unchecked(self, block: ContextBlock) -> None:
        """Add a block bypassing the budget check.

        Used by ContextAssembler which manages its own budget
        logic. Emits a BLOCK_ADDED event.

        Args:
            block: The block to add.
        """
        self._blocks.append(block)
        self._invalidate_cache()
        self._emit_block_added_event(block)

    def _raise_budget_exceeded(self, block: ContextBlock, block_tokens: int) -> None:
        """Emit a BUDGET_EXCEEDED event and raise BudgetExceededError."""
        emit(
            BudgetEventData(
                event=ContextEvent.BUDGET_EXCEEDED,
                tokens_used=self.token_count,
                tokens_max=self._max_tokens,
                details={
                    "block_name": block.display_name,
                    "block_tokens": block_tokens,
                },
            )
        )
        raise BudgetExceededError(
            block_name=block.display_name,
            block_tokens=block_tokens,
            budget_remaining=self.budget_remaining,
            max_tokens=self._max_tokens,
        )

    def _emit_block_added_event(self, block: ContextBlock) -> None:
        """Emit a BLOCK_ADDED event for the given block."""
        emit(
            BlockEventData(
                event=ContextEvent.BLOCK_ADDED,
                block_name=block.display_name,
                block_type=block.type.value,
                token_count=block.token_count,
            )
        )

    def replace_blocks(self, blocks: List[ContextBlock]) -> None:
        """Replace all blocks (used by ContextPipeline).

        Args:
            blocks: The new block list.
        """
        self._blocks = list(blocks)
        self._invalidate_cache()

    def check_budget_warnings(self) -> None:
        """Run the budget monitor if configured.

        Public entry point for subsystems that modify blocks
        (e.g. ContextAssembler, ContextPipeline).
        """
        self._check_budget_warnings()

    # -- Keep private aliases for backwards compatibility --
    def _add_no_check(self, block: ContextBlock) -> None:
        """Delegate to add_unchecked (kept for compatibility)."""
        self.add_unchecked(block)

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        model_str = self._model_name or "custom"
        return (
            f"ContextWindow(model={model_str!r}, "
            f"blocks={len(self._blocks)}, "
            f"tokens={self.token_count}/{self._max_tokens})"
        )

    def __len__(self) -> int:
        """Return the number of blocks in the window."""
        return len(self._blocks)
