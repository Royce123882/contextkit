"""Pipeline base class for pipeline steps.

Provides the PipelineStep abstract base class with optional
conditional guards and async support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import List

from contextkit.core import ContextBlock
from contextkit.pipeline.base_models import PipelineReport, StepReport


class PipelineStep(ABC):
    """Base class for pipeline steps.

    Subclasses must implement ``process()`` which takes a list of blocks
    and returns a modified list. Mutations should be recorded on
    the blocks themselves.

    Supports an optional ``run_if`` guard: a callable that receives
    the current block list and returns True if the step should execute.
    When the guard returns False, the step is skipped entirely.

    Args:
        run_if: Optional callable ``(blocks) -> bool`` that determines
            whether this step should run. If None, always runs.
    """

    def __init__(
        self,
        run_if: Callable[[List[ContextBlock]], bool] | None = None,
    ) -> None:
        self._run_if = run_if

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this step."""
        ...

    @abstractmethod
    def process(self, blocks: List[ContextBlock]) -> List[ContextBlock]:
        """Process blocks and return the modified list.

        Args:
            blocks: Input blocks to process.

        Returns:
            Modified list of blocks (may be shorter).
        """
        ...

    async def async_process(self, blocks: List[ContextBlock]) -> List[ContextBlock]:
        """Async version of ``process()``.

        Default implementation delegates to the synchronous
        ``process()`` method. Override in subclasses that need
        true async I/O (e.g. calling an LLM for summarization).

        Args:
            blocks: Input blocks to process.

        Returns:
            Modified list of blocks.
        """
        return self.process(blocks)

    def should_run(self, blocks: List[ContextBlock]) -> bool:
        """Check whether this step should execute on the given blocks.

        Returns True if no ``run_if`` guard was set, or if the guard
        returns True.

        Args:
            blocks: The current block list.

        Returns:
            True if the step should run.
        """
        if self._run_if is None:
            return True
        return self._run_if(blocks)
