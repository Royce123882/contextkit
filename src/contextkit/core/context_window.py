"""ContextWindow: the core container for assembled context."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from typing import Any, Dict, List, Tuple

from contextkit.utils.cache import clear_token_cache
from contextkit.constants import (
    DEFAULT_ENCODING,
    PRIORITY_EXAMPLE,
    PRIORITY_FILE_CONTEXT,
    PRIORITY_RAG_CHUNK,
    PRIORITY_SHORT_TERM_MEMORY,
    PRIORITY_SYSTEM_PROMPT,
    PRIORITY_TOOL_DEFINITION,
)
from contextkit.core.block import BlockType, ContextBlock
from contextkit.exceptions import BudgetExceededError
from contextkit.models import get_model
from contextkit.observe.events import (
    BlockEventData,
    BudgetEventData,
    ContextEvent,
    emit,
)
from contextkit.observe.diff import diff_windows
from contextkit.observe.explain import explain_block
from contextkit.observe.inspect import dump_window, inspect_window
from contextkit.observe.warnings import BudgetMonitor

logger = logging.getLogger("contextkit")


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
        logger.info(
            "ContextWindow created for model '%s' (%s tokens)",
            model,
            f"{spec.max_context:,}",
        )

    def _init_with_manual_tokens(self, max_tokens: int) -> None:
        """Configure the window with manual token limits."""
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        self._model_name = None
        self._max_tokens = max_tokens
        self._encoding = DEFAULT_ENCODING
        self._input_cost_per_mtok = 0.0
        self._output_cost_per_mtok = 0.0
        logger.info("ContextWindow created with manual budget (%s tokens)", f"{max_tokens:,}")

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
            self._cached_token_count = sum(
                block.token_count for block in self._blocks
            )
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

    def add(self, block: ContextBlock) -> "ContextWindow":
        """Add a context block to the window.

        Raises BudgetExceededError if the block would overflow the budget.
        Emits BLOCK_ADDED event and runs budget monitor check.

        Args:
            block: The block to add.

        Returns:
            This ContextWindow instance for method chaining.
        """
        block_tokens = block.token_count
        if self.token_count + block_tokens > self._max_tokens:
            self._raise_budget_exceeded(block, block_tokens)

        self._blocks.append(block)
        self._invalidate_cache()
        self._emit_block_added_event(block)
        self._check_budget_warnings()
        logger.debug(
            "Added block '%s' (%s tokens, priority %d)",
            block.display_name,
            f"{block_tokens:,}",
            block.priority,
        )
        return self

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
        sorted_blocks = sorted(self._blocks, key=lambda block: block.priority, reverse=True)

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
            "blocks": [self._serialize_block(block) for block in self._blocks],
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
        return inspect_window(self, block_name=block_name, format=format)

    def explain(self, block_name: str) -> str:
        """Explain why a block is included or excluded.

        Args:
            block_name: The name of the block to explain.

        Returns:
            A plain-language explanation.
        """
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
        return diff_windows(self, other, format=format)

    def dump(self, path: str) -> None:
        """Write a full JSON snapshot to a file.

        Args:
            path: The file path to write the JSON snapshot to.
        """
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

    def will_fit(self, block: ContextBlock) -> bool:
        """Check whether a block would fit within the remaining token budget.

        Does not add the block; this is a dry-run check.

        Args:
            block: The block to test.

        Returns:
            True if the block's tokens fit within the remaining budget.
        """
        return block.token_count <= self.budget_remaining

    def blocks_of_type(self, block_type: BlockType) -> List[ContextBlock]:
        """Return all blocks matching the given type.

        Args:
            block_type: The BlockType to filter by.

        Returns:
            List of matching ContextBlocks (may be empty).
        """
        return [block for block in self._blocks if block.type == block_type]

    def find_blocks(self, predicate: Callable[[ContextBlock], bool]) -> List[ContextBlock]:
        """Return all blocks matching a predicate function.

        Args:
            predicate: A callable that accepts a ContextBlock and returns True to include it.

        Returns:
            List of matching ContextBlocks.
        """
        return [block for block in self._blocks if predicate(block)]

    def clone(self) -> "ContextWindow":
        """Create a deep copy of this window with the same configuration and blocks.

        Returns:
            A new ContextWindow with deeply copied blocks.
        """
        new_window = ContextWindow(max_tokens=self._max_tokens)
        new_window._model_name = self._model_name
        new_window._encoding = self._encoding
        new_window._input_cost_per_mtok = self._input_cost_per_mtok
        new_window._output_cost_per_mtok = self._output_cost_per_mtok
        for block in self._blocks:
            new_window.add_unchecked(copy.deepcopy(block))
        return new_window

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

    def add_unchecked(self, block: ContextBlock) -> "ContextWindow":
        """Add a block bypassing the budget check.

        Used by ContextAssembler which manages its own budget
        logic. Emits a BLOCK_ADDED event.

        Args:
            block: The block to add.

        Returns:
            This ContextWindow instance for method chaining.
        """
        self._blocks.append(block)
        self._invalidate_cache()
        self._emit_block_added_event(block)
        return self

    def _raise_budget_exceeded(self, block: ContextBlock, block_tokens: int) -> None:
        """Emit a BUDGET_EXCEEDED event and raise BudgetExceededError.

        Collects low-priority blocks that could be removed to make space,
        sorted by priority ascending (lowest priority first).
        """
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

        # Find removable blocks sorted by token count (largest first)
        removable_priority_threshold = 50
        removable_blocks = sorted(
            [
                (existing_block.display_name, existing_block.token_count)
                for existing_block in self._blocks
                if existing_block.priority <= removable_priority_threshold
            ],
            key=lambda pair: pair[1],
            reverse=True,
        )

        raise BudgetExceededError(
            block_name=block.display_name,
            block_tokens=block_tokens,
            budget_remaining=self.budget_remaining,
            max_tokens=self._max_tokens,
            removable_blocks=removable_blocks,
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

    def lint(self) -> list:
        """Run the context linter and return a list of warnings.

        Checks for common anti-patterns such as missing system prompts,
        high-priority blocks in the middle (lost-in-middle), oversized
        blocks, redundant content, and empty blocks.

        Returns:
            A list of LintWarning objects describing detected issues.
        """
        from contextkit.observe.linter import ContextLinter

        return ContextLinter().lint(self)

    # ------------------------------------------------------------------
    # Fluent convenience methods
    # ------------------------------------------------------------------

    def with_system(
        self,
        content: str,
        *,
        priority: int | None = None,
        name: str | None = None,
    ) -> "ContextWindow":
        """Add a system prompt block and return self for chaining.

        Args:
            content: The system prompt text.
            priority: Optional priority override.
            name: Optional display name.

        Returns:
            This ContextWindow instance.
        """
        block = ContextBlock.system(
            content, priority=priority or PRIORITY_SYSTEM_PROMPT, name=name
        )
        return self.add(block)

    def with_rag(
        self,
        content: str,
        *,
        priority: int | None = None,
        name: str | None = None,
    ) -> "ContextWindow":
        """Add a RAG block and return self for chaining.

        Args:
            content: The retrieved content.
            priority: Optional priority override.
            name: Optional display name.

        Returns:
            This ContextWindow instance.
        """
        block = ContextBlock.rag(
            content, priority=priority or PRIORITY_RAG_CHUNK, name=name
        )
        return self.add(block)

    def with_memory(
        self,
        content: str | list,
        *,
        priority: int | None = None,
        name: str | None = None,
    ) -> "ContextWindow":
        """Add a memory block and return self for chaining.

        Args:
            content: Memory content (string or message list).
            priority: Optional priority override.
            name: Optional display name.

        Returns:
            This ContextWindow instance.
        """
        block = ContextBlock(
            type=BlockType.SHORT_TERM_MEMORY,
            content=content,
            priority=priority or PRIORITY_SHORT_TERM_MEMORY,
            name=name or "memory",
        )
        return self.add(block)

    def with_tools(
        self,
        content: str,
        *,
        priority: int | None = None,
        name: str | None = None,
    ) -> "ContextWindow":
        """Add a tool definitions block and return self for chaining.

        Args:
            content: Tool definitions content.
            priority: Optional priority override.
            name: Optional display name.

        Returns:
            This ContextWindow instance.
        """
        block = ContextBlock(
            type=BlockType.TOOL_DEFINITIONS,
            content=content,
            priority=priority or PRIORITY_TOOL_DEFINITION,
            name=name or "tools",
        )
        return self.add(block)

    def with_examples(
        self,
        content: str | list,
        *,
        priority: int | None = None,
        name: str | None = None,
    ) -> "ContextWindow":
        """Add an examples block and return self for chaining.

        Args:
            content: Example content (string or message list).
            priority: Optional priority override.
            name: Optional display name.

        Returns:
            This ContextWindow instance.
        """
        block = ContextBlock.examples(
            content, priority=priority or PRIORITY_EXAMPLE, name=name
        )
        return self.add(block)

    def with_file(
        self,
        content: str,
        *,
        file_path: str = "",
        priority: int | None = None,
        name: str | None = None,
    ) -> "ContextWindow":
        """Add a file context block and return self for chaining.

        Args:
            content: The file content.
            file_path: Path to the source file.
            priority: Optional priority override.
            name: Optional display name.

        Returns:
            This ContextWindow instance.
        """
        block = ContextBlock.file(
            content,
            file_path=file_path,
            priority=priority or PRIORITY_FILE_CONTEXT,
            name=name,
        )
        return self.add(block)

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
