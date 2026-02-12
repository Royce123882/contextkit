"""Research Report Agent -- FastAPI application.

A research assistant that gathers academic sources, takes structured
notes via a scratchpad, optimizes context with a full pipeline, and
formats output for multiple LLM providers.

Usage:
    uvicorn examples.research_report_agent.main:app --reload --port 8001
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException

from examples.research_report_agent.agent import ResearchAgent
from examples.research_report_agent.models import (
    ContextInspection,
    HealthResponse,
    PipelineStepReport,
    ResearchQueryRequest,
    ResearchQueryResponse,
    ResearchSessionRequest,
    ResearchSessionResponse,
    ScratchpadNote,
    ScratchpadResponse,
    TimelineResponse,
    TimelineSnapshot,
)

research_agent = ResearchAgent()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Initialize the research agent on startup."""
    await research_agent.initialize()
    yield


app = FastAPI(
    title="Research Report Agent",
    description="AI-powered research assistant using contextkit for RAG "
    "retrieval, scratchpad notes, pipeline optimization, and "
    "dual-provider formatting.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------


@app.post("/research", response_model=ResearchSessionResponse)
async def create_research_session(
    request: ResearchSessionRequest,
) -> ResearchSessionResponse:
    """Create a new research session and assemble initial context.

    Retrieves relevant sources, initializes the scratchpad with a
    report outline, runs the optimization pipeline, and returns the
    formatted payload ready for the LLM API.

    Args:
        request: Research session parameters (topic, audience, max_sources).

    Returns:
        The session with formatted payload and context inspection.
    """
    result = await research_agent.create_session(
        topic=request.topic,
        audience=request.audience,
        max_sources=request.max_sources,
    )

    return ResearchSessionResponse(
        session_id=result["session_id"],
        topic=result["topic"],
        audience=result["audience"],
        scratchpad_notes=[
            ScratchpadNote(**note) for note in result["scratchpad_notes"]
        ],
        formatted_payload=result["formatted_payload"],
        context_inspection=ContextInspection(
            **{
                **result["context_inspection"],
                "pipeline_steps": [
                    PipelineStepReport(**step)
                    for step in result["context_inspection"]["pipeline_steps"]
                ],
            }
        ),
    )


@app.post(
    "/research/{session_id}/query", response_model=ResearchQueryResponse
)
async def submit_research_query(
    session_id: str, request: ResearchQueryRequest
) -> ResearchQueryResponse:
    """Submit a follow-up research query within an existing session.

    Retrieves additional sources for the new query, updates the
    scratchpad, re-optimizes the context, and formats the result.

    Args:
        session_id: Existing research session identifier.
        request: The follow-up research query.

    Returns:
        Updated context with formatted payload and inspection.

    Raises:
        HTTPException: 404 if the session does not exist.
    """
    try:
        result = await research_agent.query_session(
            session_id=session_id,
            query=request.query,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ResearchQueryResponse(
        session_id=result["session_id"],
        query=result["query"],
        scratchpad_notes=[
            ScratchpadNote(**note) for note in result["scratchpad_notes"]
        ],
        formatted_payload=result["formatted_payload"],
        context_inspection=ContextInspection(
            **{
                **result["context_inspection"],
                "pipeline_steps": [
                    PipelineStepReport(**step)
                    for step in result["context_inspection"]["pipeline_steps"]
                ],
            }
        ),
    )


@app.get(
    "/research/{session_id}/scratchpad", response_model=ScratchpadResponse
)
async def get_scratchpad(session_id: str) -> ScratchpadResponse:
    """View the agent's scratchpad notes for a research session.

    Args:
        session_id: Research session identifier.

    Returns:
        All scratchpad notes and their count.

    Raises:
        HTTPException: 404 if the session does not exist.
    """
    session = research_agent.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Session {session_id} not found"
        )

    scratchpad = session.scope.scratchpad
    notes = [
        ScratchpadNote(key=key, content=scratchpad.read(key) or "")
        for key in scratchpad.list_keys()
    ]

    return ScratchpadResponse(
        session_id=session_id,
        notes=notes,
        note_count=scratchpad.note_count,
    )


@app.get(
    "/research/{session_id}/timeline", response_model=TimelineResponse
)
async def get_timeline(session_id: str) -> TimelineResponse:
    """View the context timeline for a research session.

    Shows how token usage and block count evolved across turns.

    Args:
        session_id: Research session identifier.

    Returns:
        Timeline snapshots ordered by turn number.

    Raises:
        HTTPException: 404 if the session does not exist.
    """
    session = research_agent.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Session {session_id} not found"
        )

    timeline = session.scope.timeline
    snapshots = []
    if timeline:
        for snapshot in timeline.snapshots:
            snapshots.append(
                TimelineSnapshot(
                    turn=snapshot.turn,
                    block_count=snapshot.block_count,
                    token_count=snapshot.token_count,
                    budget_percent=round(snapshot.budget_percent, 1),
                )
            )

    return TimelineResponse(session_id=session_id, snapshots=snapshots)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check the service health and component status.

    Returns:
        Service status with knowledge base, tool, and session counts.
    """
    return HealthResponse(
        status="healthy",
        knowledge_base_chunks=research_agent.knowledge_base_size,
        registered_tools=research_agent.registered_tool_count,
        active_sessions=research_agent.active_session_count,
    )


if __name__ == "__main__":
    uvicorn.run(
        "examples.research_report_agent.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
