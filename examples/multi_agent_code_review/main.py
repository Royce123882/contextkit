"""Multi-Agent Code Review -- FastAPI application.

Three specialized agents (Architect, Security Analyst, Code Reviewer)
collaborate on code reviews using isolated scopes, shared memory, and
structured handoffs via contextkit.

Usage:
    uvicorn examples.multi_agent_code_review.main:app --reload --port 8002
"""

import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException

from contextkit.core import BlockType, ContextBlock
from contextkit.memory.in_memory_backend import InMemoryBackend
from contextkit.memory.long_term import LongTermMemory
from contextkit.prompts.prompt_manager import PromptManager
from contextkit.rag.chunk import Chunk
from contextkit.rag.context import RAGContext
from contextkit.rag.in_memory_retriever import InMemoryRetriever
from contextkit.scope import SharedMemory
from contextkit.tools.tool_registry import ToolRegistry
from examples.multi_agent_code_review.agents import (
    run_architect_agent,
    run_security_agent,
    run_synthesizer_agent,
)
from examples.multi_agent_code_review.models import (
    AgentFinding,
    AgentReport,
    HealthResponse,
    ReviewRequest,
    ReviewResult,
    ReviewSummaryResponse,
    TimelineEntry,
)

# ---------------------------------------------------------------
# Knowledge base: coding standards and security guidelines
# ---------------------------------------------------------------

CODING_STANDARDS: List[Chunk] = [
    Chunk(
        content="All API endpoints must validate input using Pydantic models. "
        "Never trust raw user input. Use Field() with constraints for "
        "numeric ranges, string lengths, and regex patterns.",
        source="standards/input-validation.md",
        relevance_score=0.95,
        metadata={"category": "security"},
    ),
    Chunk(
        content="Database queries must use parameterized statements. Never "
        "concatenate user input into SQL strings. Use ORM methods or "
        "prepared statements for all database operations.",
        source="standards/sql-injection.md",
        relevance_score=0.93,
        metadata={"category": "security"},
    ),
    Chunk(
        content="Functions should follow the Single Responsibility Principle. "
        "Maximum function length is 50 lines. Extract helper functions "
        "for complex logic. Each function should do one thing well.",
        source="standards/code-quality.md",
        relevance_score=0.85,
        metadata={"category": "architecture"},
    ),
    Chunk(
        content="All public APIs must have type annotations. Use Pydantic "
        "BaseModel for request/response schemas. Document all parameters "
        "with Google-style docstrings.",
        source="standards/type-safety.md",
        relevance_score=0.88,
        metadata={"category": "architecture"},
    ),
    Chunk(
        content="Error handling: never catch bare Exception. Use specific "
        "exception types. Log errors with context (request ID, user ID). "
        "Return structured error responses with appropriate HTTP codes.",
        source="standards/error-handling.md",
        relevance_score=0.82,
        metadata={"category": "reliability"},
    ),
    Chunk(
        content="Authentication tokens must be validated on every request. "
        "Use JWT with short expiration (15 min access, 7 day refresh). "
        "Store secrets in environment variables, never in code.",
        source="standards/auth-guidelines.md",
        relevance_score=0.91,
        metadata={"category": "security"},
    ),
]


# ---------------------------------------------------------------
# Shared application state
# ---------------------------------------------------------------

rag_retriever = InMemoryRetriever()
rag_context = RAGContext(retriever=rag_retriever, retriever_name="coding-standards")
tool_registry = ToolRegistry()
memory_backend = InMemoryBackend()
long_term_memory = LongTermMemory(backend=memory_backend)
prompt_manager = PromptManager()

completed_reviews: Dict[str, Dict[str, Any]] = {}


async def _initialize_components() -> None:
    """Load knowledge base, register tools, prompts, and seed memory.

    Called once at application startup via the lifespan handler.
    """
    rag_retriever.add_chunks(CODING_STANDARDS)

    tool_registry.register(
        name="run_static_analysis",
        description="Run static analysis (ruff, mypy, bandit) on changed files.",
        parameters={
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File paths to analyze.",
                },
            },
            "required": ["files"],
        },
        tags=["analysis", "code"],
    )
    tool_registry.register(
        name="run_tests",
        description="Run the test suite for affected modules.",
        parameters={
            "type": "object",
            "properties": {
                "modules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Modules to test.",
                },
            },
            "required": ["modules"],
        },
        tags=["testing", "code"],
    )
    tool_registry.register(
        name="check_dependencies",
        description="Check for known vulnerabilities in dependencies.",
        parameters={
            "type": "object",
            "properties": {
                "lockfile": {
                    "type": "string",
                    "description": "Path to lockfile.",
                },
            },
            "required": ["lockfile"],
        },
        tags=["security", "dependencies"],
    )

    prompt_manager.register(
        "architect_v1",
        "You are a software architect reviewing code for design quality.\n\n"
        "Focus on:\n"
        "- API design and REST conventions\n"
        "- Type safety and input validation\n"
        "- Separation of concerns\n"
        "- Error handling patterns\n\n"
        "Project: {{project}}",
        description="Architect agent system prompt",
    )
    prompt_manager.register(
        "security_analyst_v1",
        "You are a security analyst reviewing code for vulnerabilities.\n\n"
        "Focus on:\n"
        "- SQL injection and input sanitization\n"
        "- Authentication and authorization\n"
        "- Data exposure and privacy\n"
        "- OWASP Top 10 vulnerabilities\n\n"
        "Recent incidents: {{incidents}}",
        description="Security analyst agent system prompt",
    )
    prompt_manager.register(
        "code_reviewer_v1",
        "You are a senior code reviewer synthesizing findings from "
        "specialized agents.\n\n"
        "Your job:\n"
        "- Consolidate architecture and security findings\n"
        "- Prioritize issues by severity and impact\n"
        "- Provide actionable fix suggestions\n"
        "- Make a final approve/request-changes decision",
        description="Code reviewer synthesizer system prompt",
    )

    await long_term_memory.store(
        "project_config",
        "FastAPI project with SQLAlchemy ORM. Python 3.12. "
        "Code coverage target: 90%. Security review required for all "
        "endpoints handling user data.",
        tags=["project", "config"],
        importance=0.9,
    )
    await long_term_memory.store(
        "recent_incidents",
        "SQL injection vulnerability found in orders module last month. "
        "Team committed to parameterized queries in all new code.",
        tags=["security", "history"],
        importance=0.8,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Initialize all components on startup."""
    await _initialize_components()
    yield


app = FastAPI(
    title="Multi-Agent Code Review",
    description="Three specialized agents collaborate on code reviews "
    "using contextkit for isolated scopes, shared memory, "
    "handoffs, and pipeline optimization.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------
# Helper to extract timeline data from a scope
# ---------------------------------------------------------------


def _extract_timeline(scope: Any) -> List[Dict[str, Any]]:
    """Extract timeline snapshots from an agent's scope.

    Args:
        scope: The agent's ContextScope.

    Returns:
        List of timeline entry dicts.
    """
    entries: List[Dict[str, Any]] = []
    if scope.timeline:
        for snapshot in scope.timeline.snapshots:
            entries.append({
                "turn": snapshot.turn,
                "block_count": snapshot.block_count,
                "token_count": snapshot.token_count,
                "budget_percent": round(snapshot.budget_percent, 1),
            })
    return entries


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------


@app.post("/review", response_model=ReviewResult)
async def submit_review(request: ReviewRequest) -> ReviewResult:
    """Submit code for multi-agent review.

    Runs three agents in sequence:
    1. Architect reviews design and structure
    2. Security Analyst finds vulnerabilities
    3. Code Reviewer synthesizes findings and decides

    Args:
        request: Code and description to review.

    Returns:
        Consolidated review with decision, findings, and fix plan.
    """
    review_id = str(uuid.uuid4())[:8]
    shared_memory = SharedMemory()

    # Publish the code under review to shared memory
    code_block = ContextBlock(
        type=BlockType.USER_CONTEXT,
        content=request.code,
        priority=90,
        name="pr_diff",
    )
    shared_memory.publish("pr_code", code_block)

    # Agent 1: Architect
    architect_result = await run_architect_agent(
        shared_memory=shared_memory,
        rag_context=rag_context,
        prompt_manager=prompt_manager,
        long_term_memory=long_term_memory,
    )

    # Agent 2: Security Analyst
    security_result = await run_security_agent(
        shared_memory=shared_memory,
        rag_context=rag_context,
        tool_registry=tool_registry,
        prompt_manager=prompt_manager,
        long_term_memory=long_term_memory,
    )

    # Agent 3: Code Reviewer (Synthesizer)
    synthesizer_result = run_synthesizer_agent(
        shared_memory=shared_memory,
        architect_scope=architect_result["scope"],
        security_scope=security_result["scope"],
        prompt_manager=prompt_manager,
    )

    # Build agent reports
    agent_reports = []
    all_timelines: Dict[str, List[Dict[str, Any]]] = {}

    for result in [architect_result, security_result, synthesizer_result]:
        agent_name = result["agent_name"]
        scope = result["scope"]

        agent_reports.append(AgentReport(
            agent_name=agent_name,
            findings=[
                AgentFinding(**finding)
                for finding in result.get("findings", [])
            ],
            scratchpad_notes=result["scratchpad_notes"],
            block_count=result["block_count"],
            token_count=result["token_count"],
        ))

        all_timelines[agent_name] = _extract_timeline(scope)

    # Store completed review
    review_data = {
        "review_id": review_id,
        "decision": synthesizer_result["decision"],
        "summary": synthesizer_result["summary"],
        "fix_priority": synthesizer_result["fix_priority"],
        "agent_reports": agent_reports,
        "shared_memory_blocks": shared_memory.list_blocks(),
        "timelines": {
            name: [TimelineEntry(**entry) for entry in entries]
            for name, entries in all_timelines.items()
        },
        "synthesizer_context": synthesizer_result["synthesizer_context"],
    }
    completed_reviews[review_id] = review_data

    return ReviewResult(**review_data)


@app.get("/review/{review_id}", response_model=ReviewSummaryResponse)
async def get_review_summary(review_id: str) -> ReviewSummaryResponse:
    """Get a summary of a completed code review.

    Args:
        review_id: Unique review identifier.

    Returns:
        Brief summary with decision and finding counts.

    Raises:
        HTTPException: 404 if the review does not exist.
    """
    review_data = completed_reviews.get(review_id)
    if review_data is None:
        raise HTTPException(
            status_code=404, detail=f"Review {review_id} not found"
        )

    # Count findings by severity
    severity_counts: Dict[str, int] = {
        "critical": 0, "high": 0, "medium": 0, "low": 0
    }
    for agent_report in review_data["agent_reports"]:
        for finding in agent_report.findings:
            severity = finding.severity
            if severity in severity_counts:
                severity_counts[severity] += 1

    return ReviewSummaryResponse(
        review_id=review_id,
        decision=review_data["decision"],
        finding_counts=severity_counts,
        agent_names=[
            report.agent_name for report in review_data["agent_reports"]
        ],
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check the service health and component status.

    Returns:
        Service status with component counts.
    """
    return HealthResponse(
        status="healthy",
        knowledge_base_chunks=rag_retriever.chunk_count,
        registered_tools=tool_registry.tool_count,
        completed_reviews=len(completed_reviews),
    )


if __name__ == "__main__":
    uvicorn.run(
        "examples.multi_agent_code_review.main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
    )
