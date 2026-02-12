"""Pydantic request/response models for the research report API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------
# Request models
# ---------------------------------------------------------------


class ResearchSessionRequest(BaseModel):
    """Request to create a new research session.

    Attributes:
        topic: The research topic to investigate.
        audience: Target audience level for the report.
        max_sources: Maximum number of RAG sources to retrieve.
    """

    topic: str = Field(
        ..., min_length=3, max_length=500, description="Research topic to investigate."
    )
    audience: str = Field(
        default="technical",
        description="Target audience level (e.g., technical, executive, general).",
    )
    max_sources: int = Field(
        default=5, ge=1, le=20, description="Maximum RAG sources to retrieve."
    )


class ResearchQueryRequest(BaseModel):
    """A follow-up research query within an existing session.

    Attributes:
        query: The research question to investigate.
    """

    query: str = Field(
        ..., min_length=1, max_length=1000, description="Research question."
    )


# ---------------------------------------------------------------
# Response models
# ---------------------------------------------------------------


class ScratchpadNote(BaseModel):
    """A single note from the agent's scratchpad.

    Attributes:
        key: Note identifier.
        content: Note content.
    """

    key: str = Field(..., description="Note identifier.")
    content: str = Field(..., description="Note content.")


class PipelineStepReport(BaseModel):
    """Report for a single pipeline optimization step.

    Attributes:
        step_name: Name of the pipeline step.
        tokens_before: Token count before this step.
        tokens_after: Token count after this step.
        tokens_saved: Tokens removed by this step.
        blocks_removed: Number of blocks removed.
    """

    step_name: str = Field(..., description="Pipeline step name.")
    tokens_before: int = Field(..., description="Tokens before the step.")
    tokens_after: int = Field(..., description="Tokens after the step.")
    tokens_saved: int = Field(..., description="Tokens removed by the step.")
    blocks_removed: int = Field(..., description="Blocks removed by the step.")


class ContextInspection(BaseModel):
    """Detailed inspection of the assembled context window.

    Attributes:
        total_blocks: Number of blocks in the context.
        total_tokens: Total token count.
        max_tokens: Token budget.
        budget_used_percent: Percentage of budget used.
        blocks: Summary of each block.
        pipeline_steps: Per-step pipeline reports.
        total_tokens_saved: Total tokens removed by the pipeline.
        sufficiency_confident: Whether the context is sufficient.
        sufficiency_score: Sufficiency confidence score.
        quality_score: Positional quality score.
    """

    total_blocks: int = Field(..., description="Blocks in the context.")
    total_tokens: int = Field(..., description="Total token count.")
    max_tokens: int = Field(..., description="Token budget.")
    budget_used_percent: float = Field(..., description="Budget percentage used.")
    blocks: List[Dict[str, Any]] = Field(..., description="Per-block summaries.")
    pipeline_steps: List[PipelineStepReport] = Field(
        ..., description="Per-step pipeline reports."
    )
    total_tokens_saved: int = Field(
        ..., description="Total tokens removed by the pipeline."
    )
    sufficiency_confident: bool = Field(
        ..., description="Whether the context is sufficient."
    )
    sufficiency_score: float = Field(..., description="Sufficiency confidence score.")
    quality_score: float = Field(..., description="Positional quality score.")


class TimelineSnapshot(BaseModel):
    """A snapshot from the agent's timeline tracker.

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


class ResearchSessionResponse(BaseModel):
    """Response after creating a research session.

    Attributes:
        session_id: Unique identifier for the session.
        topic: The research topic.
        audience: Target audience level.
        scratchpad_notes: Agent's initial working notes.
        formatted_payload: Provider-formatted payload for the LLM API.
        context_inspection: Detailed context window inspection.
    """

    session_id: str = Field(..., description="Unique session identifier.")
    topic: str = Field(..., description="Research topic.")
    audience: str = Field(..., description="Target audience.")
    scratchpad_notes: List[ScratchpadNote] = Field(
        ..., description="Agent's working notes."
    )
    formatted_payload: Dict[str, Any] = Field(
        ..., description="Provider-formatted payload for the LLM API."
    )
    context_inspection: ContextInspection = Field(
        ..., description="Detailed context window inspection."
    )


class ResearchQueryResponse(BaseModel):
    """Response after processing a follow-up research query.

    Attributes:
        session_id: Session this query belongs to.
        query: The research question asked.
        scratchpad_notes: Updated working notes.
        formatted_payload: Provider-formatted payload for the LLM API.
        context_inspection: Detailed context window inspection.
    """

    session_id: str = Field(..., description="Session identifier.")
    query: str = Field(..., description="Research question asked.")
    scratchpad_notes: List[ScratchpadNote] = Field(
        ..., description="Updated working notes."
    )
    formatted_payload: Dict[str, Any] = Field(
        ..., description="Provider-formatted payload for the LLM API."
    )
    context_inspection: ContextInspection = Field(
        ..., description="Detailed context window inspection."
    )


class ScratchpadResponse(BaseModel):
    """Current scratchpad state for a research session.

    Attributes:
        session_id: Session identifier.
        notes: All scratchpad notes.
        note_count: Total number of notes.
    """

    session_id: str = Field(..., description="Session identifier.")
    notes: List[ScratchpadNote] = Field(..., description="All scratchpad notes.")
    note_count: int = Field(..., description="Total number of notes.")


class TimelineResponse(BaseModel):
    """Timeline tracking data for a research session.

    Attributes:
        session_id: Session identifier.
        snapshots: Timeline snapshots across turns.
    """

    session_id: str = Field(..., description="Session identifier.")
    snapshots: List[TimelineSnapshot] = Field(
        ..., description="Timeline snapshots."
    )


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status: Service status.
        knowledge_base_chunks: Number of research articles indexed.
        registered_tools: Number of registered tools.
        active_sessions: Number of active research sessions.
    """

    status: str = Field(..., description="Service status.")
    knowledge_base_chunks: int = Field(
        ..., description="Research articles indexed."
    )
    registered_tools: int = Field(..., description="Registered tools.")
    active_sessions: int = Field(..., description="Active research sessions.")
