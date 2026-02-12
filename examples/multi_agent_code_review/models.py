"""Pydantic request/response models for the multi-agent code review API."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


# ---------------------------------------------------------------
# Request models
# ---------------------------------------------------------------


class ReviewRequest(BaseModel):
    """Request to submit code for multi-agent review.

    Attributes:
        code: The code or PR diff to review.
        description: Description of the changes being reviewed.
    """

    code: str = Field(
        ..., min_length=1, max_length=50000, description="Code or PR diff to review."
    )
    description: str = Field(
        default="",
        max_length=1000,
        description="Description of the changes being reviewed.",
    )


# ---------------------------------------------------------------
# Response models
# ---------------------------------------------------------------


class AgentFinding(BaseModel):
    """A single finding from a review agent.

    Attributes:
        severity: Finding severity (critical, high, medium, low).
        description: Description of the issue found.
    """

    severity: str = Field(..., description="Severity: critical, high, medium, or low.")
    description: str = Field(..., description="Description of the issue.")


class AgentReport(BaseModel):
    """Report from a single review agent.

    Attributes:
        agent_name: Name of the reviewing agent.
        findings: List of findings from this agent.
        scratchpad_notes: Agent's working notes during review.
        block_count: Number of context blocks used.
        token_count: Total tokens in the agent's context window.
    """

    agent_name: str = Field(..., description="Reviewing agent name.")
    findings: List[AgentFinding] = Field(..., description="Agent findings.")
    scratchpad_notes: Dict[str, str] = Field(
        ..., description="Agent's working notes."
    )
    block_count: int = Field(..., description="Context blocks used.")
    token_count: int = Field(..., description="Tokens in context window.")


class TimelineEntry(BaseModel):
    """A timeline snapshot for an agent.

    Attributes:
        turn: Turn number.
        block_count: Blocks at this turn.
        token_count: Tokens at this turn.
        budget_percent: Budget usage percentage.
    """

    turn: int = Field(..., description="Turn number.")
    block_count: int = Field(..., description="Blocks at this turn.")
    token_count: int = Field(..., description="Tokens at this turn.")
    budget_percent: float = Field(..., description="Budget usage percentage.")


class ReviewResult(BaseModel):
    """Consolidated result from the multi-agent code review.

    Attributes:
        review_id: Unique identifier for this review.
        decision: Final review decision (approve, request_changes).
        summary: Human-readable summary of all findings.
        fix_priority: Prioritized list of fixes needed.
        agent_reports: Per-agent detailed reports.
        shared_memory_blocks: Names of blocks in shared memory.
        timelines: Per-agent timeline data.
        synthesizer_context: Context inspection for the final synthesizer.
    """

    review_id: str = Field(..., description="Unique review identifier.")
    decision: str = Field(
        ..., description="Final decision: approve or request_changes."
    )
    summary: str = Field(..., description="Summary of all findings.")
    fix_priority: List[str] = Field(
        ..., description="Prioritized list of fixes needed."
    )
    agent_reports: List[AgentReport] = Field(
        ..., description="Per-agent detailed reports."
    )
    shared_memory_blocks: List[str] = Field(
        ..., description="Block names in shared memory."
    )
    timelines: Dict[str, List[TimelineEntry]] = Field(
        ..., description="Per-agent timeline data."
    )
    synthesizer_context: Dict[str, Any] = Field(
        ..., description="Context inspection for the synthesizer agent."
    )


class ReviewSummaryResponse(BaseModel):
    """Brief summary of a completed review.

    Attributes:
        review_id: Unique review identifier.
        decision: Final review decision.
        finding_counts: Count of findings by severity.
        agent_names: Names of agents that participated.
    """

    review_id: str = Field(..., description="Unique review identifier.")
    decision: str = Field(..., description="Final review decision.")
    finding_counts: Dict[str, int] = Field(
        ..., description="Findings by severity (critical, high, medium, low)."
    )
    agent_names: List[str] = Field(
        ..., description="Agents that participated in the review."
    )


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status: Service status.
        knowledge_base_chunks: Number of coding standards indexed.
        registered_tools: Number of registered tools.
        completed_reviews: Number of completed reviews.
    """

    status: str = Field(..., description="Service status.")
    knowledge_base_chunks: int = Field(
        ..., description="Coding standards indexed."
    )
    registered_tools: int = Field(..., description="Registered tools.")
    completed_reviews: int = Field(..., description="Completed reviews.")
