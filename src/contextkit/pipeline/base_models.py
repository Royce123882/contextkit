"""Data models for pipeline steps and reports."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


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
