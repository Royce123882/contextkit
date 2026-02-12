"""Pipeline base classes and report models.

Provides the PipelineStep abstract base class and the StepReport
and PipelineReport data models used by all pipeline steps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from contextkit.core import ContextBlock


class StepReport(BaseModel):
    """Report from a single pipeline step.

    Attributes:
        step_name: Name of the step.
        blocks_modified: Number of blocks modified.
        blocks_removed: Number of blocks removed.
        tokens_before: Total tokens before the step.
        tokens_after: Total tokens after the step.
        tokens_saved: Tokens saved by this step.
        details: Additional step-specific details.
    """

    step_name: str = Field(description="Name of the step.")
    blocks_modified: int = Field(default=0, description="Number of blocks modified.")
    blocks_removed: int = Field(default=0, description="Number of blocks removed.")
    tokens_before: int = Field(default=0, description="Total tokens before the step.")
    tokens_after: int = Field(default=0, description="Total tokens after the step.")
    tokens_saved: int = Field(default=0, description="Tokens saved by this step.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional step-specific details.")


class PipelineReport(BaseModel):
    """Report from a full pipeline run.

    Attributes:
        steps: Reports from each step.
        total_tokens_before: Total tokens before pipeline.
        total_tokens_after: Total tokens after pipeline.
        total_tokens_saved: Total tokens saved.
        cost_delta: Estimated cost savings.
    """

    steps: List[StepReport] = Field(default_factory=list, description="Reports from each step.")
    total_tokens_before: int = Field(default=0, description="Total tokens before pipeline.")
    total_tokens_after: int = Field(default=0, description="Total tokens after pipeline.")
    total_tokens_saved: int = Field(default=0, description="Total tokens saved.")
    cost_delta: float = Field(default=0.0, description="Estimated cost savings.")


class PipelineStep(ABC):
    """Base class for pipeline steps.

    Subclasses must implement process() which takes a list of blocks
    and returns a modified list. Mutations should be recorded on
    the blocks themselves.
    """

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
