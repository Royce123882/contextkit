"""Multi-Agent Code Review -- end-to-end agentic use case.

Three specialized agents (Architect, Security Analyst, Code Reviewer)
collaborate on a code review using isolated scopes, shared memory for
cross-agent findings, structured handoffs, scratchpads for working
notes, and pipeline optimization.

Usage:
    python examples/multi_agent_code_review/main.py
"""

import asyncio
import json

from contextkit.adapters.anthropic_adapter import AnthropicAdapter
from contextkit.core import BlockType, ContextBlock, ContextWindow
from contextkit.memory.in_memory_backend import InMemoryBackend
from contextkit.memory.long_term import LongTermMemory
from contextkit.observe.inspect import inspect_window
from contextkit.observe.quality import QualityScorer
from contextkit.observe.sufficiency import SufficiencyChecker
from contextkit.pipeline import (
    ContextPipeline,
    DeduplicateStep,
    ReorderStep,
    TrimStep,
)
from contextkit.prompts.prompt_manager import PromptManager
from contextkit.rag.chunk import Chunk
from contextkit.rag.context import RAGContext
from contextkit.rag.in_memory_retriever import InMemoryRetriever
from contextkit.scope import ContextScope, SharedMemory
from contextkit.tools.tool_registry import ToolRegistry


# ---------------------------------------------------------------
# 1. Knowledge Base -- coding standards and security guidelines
# ---------------------------------------------------------------

CODING_STANDARDS = [
    Chunk(
        content="All API endpoints must validate input using Pydantic models. "
        "Never trust raw user input. Use Field() with constraints for "
        "numeric ranges, string lengths, and regex patterns.",
        source="standards/input-validation.md",
        relevance_score=0.95,
        metadata={"category": "security", "severity": "critical"},
    ),
    Chunk(
        content="Database queries must use parameterized statements. Never "
        "concatenate user input into SQL strings. Use ORM methods or "
        "prepared statements for all database operations.",
        source="standards/sql-injection.md",
        relevance_score=0.93,
        metadata={"category": "security", "severity": "critical"},
    ),
    Chunk(
        content="Functions should follow the Single Responsibility Principle. "
        "Maximum function length is 50 lines. Extract helper functions "
        "for complex logic. Each function should do one thing well.",
        source="standards/code-quality.md",
        relevance_score=0.85,
        metadata={"category": "architecture", "severity": "medium"},
    ),
    Chunk(
        content="All public APIs must have type annotations. Use Pydantic "
        "BaseModel for request/response schemas. Document all parameters "
        "with Google-style docstrings.",
        source="standards/type-safety.md",
        relevance_score=0.88,
        metadata={"category": "architecture", "severity": "medium"},
    ),
    Chunk(
        content="Error handling: never catch bare Exception. Use specific "
        "exception types. Log errors with context (request ID, user ID). "
        "Return structured error responses with appropriate HTTP codes.",
        source="standards/error-handling.md",
        relevance_score=0.82,
        metadata={"category": "reliability", "severity": "high"},
    ),
    Chunk(
        content="Authentication tokens must be validated on every request. "
        "Use JWT with short expiration (15 min access, 7 day refresh). "
        "Store secrets in environment variables, never in code.",
        source="standards/auth-guidelines.md",
        relevance_score=0.91,
        metadata={"category": "security", "severity": "critical"},
    ),
]

# The code under review (simulated PR diff)
CODE_UNDER_REVIEW = """
# PR #342: Add user profile update endpoint
# File: api/routes/users.py

@router.put("/users/{user_id}")
async def update_profile(user_id: int, data: dict):
    \"\"\"Update user profile.\"\"\"
    query = f"UPDATE users SET name='{data['name']}', email='{data['email']}' WHERE id={user_id}"
    await db.execute(query)

    if data.get('role'):
        await db.execute(f"UPDATE users SET role='{data['role']}' WHERE id={user_id}")

    user = await db.fetch_one(f"SELECT * FROM users WHERE id={user_id}")
    return {"status": "updated", "user": dict(user)}


@router.get("/users/{user_id}/export")
async def export_user_data(user_id: int):
    \"\"\"Export all user data as JSON.\"\"\"
    try:
        user = await db.fetch_one(f"SELECT * FROM users WHERE id={user_id}")
        orders = await db.fetch_all(f"SELECT * FROM orders WHERE user_id={user_id}")
        return {"user": dict(user), "orders": [dict(o) for o in orders]}
    except Exception:
        return {"error": "Something went wrong"}
""".strip()


# ---------------------------------------------------------------
# 2. Tool Definitions -- static analysis, test runner, lint
# ---------------------------------------------------------------


def register_tools(registry: ToolRegistry) -> None:
    """Register code review tools."""
    registry.register(
        name="run_static_analysis",
        description="Run static analysis (ruff, mypy) on changed files. "
        "Returns type errors, lint warnings, and security issues.",
        parameters={
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File paths to analyze",
                },
                "checks": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["ruff", "mypy", "bandit"]},
                    "description": "Which analyzers to run",
                },
            },
            "required": ["files"],
        },
        tags=["analysis", "code"],
    )

    registry.register(
        name="run_tests",
        description="Run the test suite for affected modules. Returns "
        "pass/fail status and coverage information.",
        parameters={
            "type": "object",
            "properties": {
                "modules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Modules to test",
                },
                "coverage": {
                    "type": "boolean",
                    "description": "Include coverage report",
                },
            },
            "required": ["modules"],
        },
        tags=["testing", "code"],
    )

    registry.register(
        name="check_dependencies",
        description="Check for known vulnerabilities in project dependencies. "
        "Returns CVE details and fix suggestions.",
        parameters={
            "type": "object",
            "properties": {
                "lockfile": {
                    "type": "string",
                    "description": "Path to lockfile (requirements.txt or pyproject.toml)",
                },
            },
            "required": ["lockfile"],
        },
        tags=["security", "dependencies"],
    )


# ---------------------------------------------------------------
# 3. Simulated tool executions
# ---------------------------------------------------------------


def simulate_static_analysis() -> str:
    """Simulate static analysis results."""
    return json.dumps({
        "files_analyzed": ["api/routes/users.py"],
        "issues": [
            {
                "rule": "S608",
                "severity": "high",
                "message": "Possible SQL injection via string formatting",
                "line": 7,
                "file": "api/routes/users.py",
            },
            {
                "rule": "S608",
                "severity": "high",
                "message": "Possible SQL injection via string formatting",
                "line": 10,
                "file": "api/routes/users.py",
            },
            {
                "rule": "E722",
                "severity": "medium",
                "message": "Bare except clause",
                "line": 19,
                "file": "api/routes/users.py",
            },
            {
                "rule": "ANN001",
                "severity": "low",
                "message": "Missing type annotation for 'data' parameter",
                "line": 5,
                "file": "api/routes/users.py",
            },
        ],
        "summary": {"high": 2, "medium": 1, "low": 1},
    })


def simulate_test_results() -> str:
    """Simulate test runner results."""
    return json.dumps({
        "total_tests": 24,
        "passed": 20,
        "failed": 4,
        "coverage": 72.5,
        "failures": [
            {
                "test": "test_update_profile_validates_input",
                "error": "No input validation on data parameter",
            },
            {
                "test": "test_update_profile_prevents_role_escalation",
                "error": "Role field accepts arbitrary values",
            },
            {
                "test": "test_export_handles_missing_user",
                "error": "Returns 500 instead of 404",
            },
            {
                "test": "test_export_requires_authentication",
                "error": "No auth check on export endpoint",
            },
        ],
    })


# ---------------------------------------------------------------
# 4. Main multi-agent loop
# ---------------------------------------------------------------


async def run_agents() -> None:
    """Run the multi-agent code review."""
    # -- Shared resources --
    shared_memory = SharedMemory()
    memory_backend = InMemoryBackend()
    long_term = LongTermMemory(backend=memory_backend)

    retriever = InMemoryRetriever()
    retriever.add_chunks(CODING_STANDARDS)
    rag = RAGContext(retriever=retriever, retriever_name="coding-standards")

    tool_registry = ToolRegistry()
    register_tools(tool_registry)

    prompt_mgr = PromptManager()

    # Store project context in long-term memory
    await long_term.store(
        "project_config",
        "FastAPI project with SQLAlchemy ORM. Python 3.12. "
        "Code coverage target: 90%. Security review required for all "
        "endpoints handling user data.",
        tags=["project", "config"],
        importance=0.9,
    )
    await long_term.store(
        "recent_incidents",
        "SQL injection vulnerability found in orders module last month. "
        "Team committed to parameterized queries in all new code.",
        tags=["security", "history"],
        importance=0.8,
    )

    # Publish the code under review to shared memory
    code_block = ContextBlock(
        type=BlockType.USER_CONTEXT,
        content=CODE_UNDER_REVIEW,
        priority=90,
        name="pr_diff",
    )
    shared_memory.publish("pr_code", code_block)

    print("=" * 60)
    print("MULTI-AGENT CODE REVIEW")
    print("=" * 60)
    print(f"Shared memory: {shared_memory.block_count} blocks")

    # =============================================================
    # AGENT 1: Architect -- reviews design and code structure
    # =============================================================
    print("\n" + "-" * 60)
    print("AGENT 1: ARCHITECT")
    print("-" * 60)

    prompt_mgr.register(
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

    arch_window = ContextWindow(max_tokens=3000)
    arch_scope = ContextScope(
        agent_name="architect",
        window=arch_window,
        shared_memory=shared_memory,
        track_history=True,
    )

    # System prompt
    project_blocks = await long_term.retrieve_as_blocks("project config", top_k=1)
    project_text = project_blocks[0].content if project_blocks else "N/A"
    arch_system = prompt_mgr.render("architect_v1", project=project_text)
    arch_scope.window.add(arch_system)

    # Import code under review from shared memory
    imported = arch_scope.import_shared(["pr_code"])
    print(f"Imported {imported} blocks from shared memory")

    # RAG: coding standards for architecture
    arch_rag = await rag.retrieve(
        query="code quality type safety API design", top_k=3, min_relevance=0.5
    )
    for block in arch_rag:
        arch_scope.window.add(block)
    print(f"RAG: {len(arch_rag)} standards retrieved")

    # Architect's analysis (scratchpad notes)
    arch_scope.scratchpad.write(
        "finding_1",
        "CRITICAL: update_profile accepts raw dict instead of Pydantic model. "
        "No input validation whatsoever.",
    )
    arch_scope.scratchpad.write(
        "finding_2",
        "MEDIUM: update_profile does two things (update profile + update role). "
        "Violates SRP. Role updates should be a separate endpoint with "
        "admin authorization.",
    )
    arch_scope.scratchpad.write(
        "finding_3",
        "MEDIUM: export_user_data uses bare except. Should catch specific "
        "exceptions and return proper HTTP error codes.",
    )
    arch_scope.scratchpad.write(
        "finding_4",
        "LOW: Missing response models. Return types should use Pydantic "
        "schemas for API documentation.",
    )
    arch_scope.window.add(arch_scope.scratchpad.to_block())

    arch_scope.record_turn(events=["architecture_review_complete"])

    # Publish findings to shared memory
    arch_findings = ContextBlock(
        type=BlockType.LONG_TERM_MEMORY,
        content="Architecture Review Findings:\n"
        "1. [CRITICAL] No input validation - raw dict parameter\n"
        "2. [MEDIUM] SRP violation - role update mixed with profile update\n"
        "3. [MEDIUM] Bare except clause in export endpoint\n"
        "4. [LOW] Missing Pydantic response models",
        priority=80,
        name="architect_findings",
    )
    arch_scope.window.add(arch_findings)
    shared_memory.publish("architect_findings", arch_findings)

    print(f"Context: {len(arch_scope.window.blocks)} blocks, "
          f"{arch_scope.window.token_count:,} tokens")
    print(f"Scratchpad: {arch_scope.scratchpad.note_count} notes")
    print(f"Shared memory: {shared_memory.block_count} blocks")

    # =============================================================
    # AGENT 2: Security Analyst -- focuses on security vulnerabilities
    # =============================================================
    print("\n" + "-" * 60)
    print("AGENT 2: SECURITY ANALYST")
    print("-" * 60)

    prompt_mgr.register(
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

    sec_window = ContextWindow(max_tokens=3000)
    sec_scope = ContextScope(
        agent_name="security_analyst",
        window=sec_window,
        shared_memory=shared_memory,
        track_history=True,
    )

    # System prompt with incident history
    incident_blocks = await long_term.retrieve_as_blocks("security incident", top_k=1)
    incident_text = incident_blocks[0].content if incident_blocks else "None"
    sec_system = prompt_mgr.render("security_analyst_v1", incidents=incident_text)
    sec_scope.window.add(sec_system)

    # Import code + architect findings from shared memory
    imported = sec_scope.import_shared(["pr_code", "architect_findings"])
    print(f"Imported {imported} blocks from shared memory")

    # RAG: security-specific standards
    sec_rag = await rag.retrieve(
        query="SQL injection authentication security", top_k=3, min_relevance=0.5
    )
    for block in sec_rag:
        sec_scope.window.add(block)
    print(f"RAG: {len(sec_rag)} standards retrieved")

    # Tool outputs: static analysis + test results
    analysis_result = simulate_static_analysis()
    analysis_output = tool_registry.capture_output(
        tool_name="run_static_analysis",
        result=analysis_result,
        latency_ms=890.3,
    )
    sec_scope.window.add(analysis_output.to_block(priority=75))
    print(f"Tool output: run_static_analysis ({analysis_output.latency_ms}ms)")

    test_result = simulate_test_results()
    test_output = tool_registry.capture_output(
        tool_name="run_tests",
        result=test_result,
        latency_ms=2340.7,
    )
    sec_scope.window.add(test_output.to_block(priority=75))
    print(f"Tool output: run_tests ({test_output.latency_ms}ms)")

    # Security analyst's findings (scratchpad)
    sec_scope.scratchpad.write(
        "vuln_1",
        "CRITICAL: SQL injection on lines 7, 10, 12, 17. Uses f-string "
        "formatting to build SQL queries. Must use parameterized queries.",
    )
    sec_scope.scratchpad.write(
        "vuln_2",
        "CRITICAL: No authentication on either endpoint. User can update "
        "any profile including role escalation (admin takeover possible).",
    )
    sec_scope.scratchpad.write(
        "vuln_3",
        "HIGH: export_user_data exposes all columns (SELECT *). May leak "
        "sensitive fields (password hash, internal IDs). Use explicit "
        "column selection.",
    )
    sec_scope.scratchpad.write(
        "vuln_4",
        "HIGH: No rate limiting on export endpoint. Could be used for "
        "bulk data scraping.",
    )
    sec_scope.window.add(sec_scope.scratchpad.to_block())

    sec_scope.record_turn(events=["security_review_complete"])

    # Publish security findings
    sec_findings = ContextBlock(
        type=BlockType.LONG_TERM_MEMORY,
        content="Security Review Findings:\n"
        "1. [CRITICAL] SQL injection (4 instances) - use parameterized queries\n"
        "2. [CRITICAL] No authentication or authorization checks\n"
        "3. [HIGH] Data exposure via SELECT * - use explicit columns\n"
        "4. [HIGH] No rate limiting on data export endpoint\n"
        "Static analysis: 2 high, 1 medium, 1 low\n"
        "Tests: 4/24 failing (input validation, auth, error handling)",
        priority=85,
        name="security_findings",
    )
    sec_scope.window.add(sec_findings)
    shared_memory.publish("security_findings", sec_findings)

    print(f"Context: {len(sec_scope.window.blocks)} blocks, "
          f"{sec_scope.window.token_count:,} tokens")
    print(f"Scratchpad: {sec_scope.scratchpad.note_count} notes")
    print(f"Shared memory: {shared_memory.block_count} blocks")

    # =============================================================
    # AGENT 3: Code Reviewer -- synthesizes all findings
    # =============================================================
    print("\n" + "-" * 60)
    print("AGENT 3: CODE REVIEWER (SYNTHESIZER)")
    print("-" * 60)

    prompt_mgr.register(
        "code_reviewer_v1",
        "You are a senior code reviewer synthesizing findings from "
        "specialized agents.\n\n"
        "Your job:\n"
        "- Consolidate architecture and security findings\n"
        "- Prioritize issues by severity and impact\n"
        "- Provide actionable fix suggestions\n"
        "- Make a final approve/request-changes decision",
        description="Code reviewer agent system prompt",
    )

    review_window = ContextWindow(max_tokens=4000)
    review_scope = ContextScope(
        agent_name="code_reviewer",
        window=review_window,
        shared_memory=shared_memory,
        track_history=True,
    )

    # System prompt
    review_system = prompt_mgr.render("code_reviewer_v1")
    review_scope.window.add(review_system)

    # Receive handoffs from both agents
    arch_handoff = arch_scope.handoff(
        target_agent="code_reviewer",
        block_names=["architect_findings"],
        metadata={"review_type": "architecture", "agent": "architect"},
    )
    sec_handoff = sec_scope.handoff(
        target_agent="code_reviewer",
        block_names=["security_findings"],
        metadata={"review_type": "security", "agent": "security_analyst"},
    )

    arch_received = review_scope.receive_handoff(arch_handoff)
    sec_received = review_scope.receive_handoff(sec_handoff)
    print(f"Received handoffs: {arch_received} blocks from architect, "
          f"{sec_received} blocks from security")

    # Import the code under review directly
    imported = review_scope.import_shared(["pr_code"])
    print(f"Imported {imported} blocks from shared memory")

    # Check what scratchpad notes were transferred
    transferred_keys = review_scope.scratchpad.list_keys()
    print(f"Scratchpad notes from handoffs: {len(transferred_keys)}")

    # Reviewer's own scratchpad analysis
    review_scope.scratchpad.write(
        "decision",
        "REQUEST CHANGES - Multiple critical security issues",
    )
    review_scope.scratchpad.write(
        "summary",
        "2 critical, 2 high, 2 medium, 1 low issues found. "
        "PR requires significant rework before merge.",
    )
    review_scope.scratchpad.write(
        "fix_priority",
        "1. Fix SQL injection (critical/immediate)\n"
        "2. Add authentication middleware (critical/immediate)\n"
        "3. Add Pydantic input validation (high/before merge)\n"
        "4. Fix data exposure with explicit columns (high/before merge)\n"
        "5. Split role update to separate endpoint (medium/next sprint)\n"
        "6. Fix error handling (medium/next sprint)\n"
        "7. Add response models (low/backlog)",
    )
    review_scope.window.add(review_scope.scratchpad.to_block())

    review_scope.record_turn(events=["synthesis_complete"])

    # -- Pipeline: optimize the reviewer's context --
    pipeline = ContextPipeline(
        steps=[
            DeduplicateStep(similarity_threshold=0.7),
            TrimStep(max_tokens=review_window.max_tokens, min_priority=20),
            ReorderStep(strategy="important_edges"),
        ]
    )
    pipeline.run(review_scope.window)

    report = pipeline.last_report
    print(f"\nPipeline: {report.total_tokens_saved:,} tokens saved")

    # -- Sufficiency check --
    review_query = "code review security architecture quality assessment"
    checker = SufficiencyChecker()
    sufficiency = checker.check(review_query, review_scope.window.blocks)
    print(
        f"Sufficiency: sufficient={sufficiency.sufficient}, "
        f"confidence={sufficiency.confidence:.2f}"
    )

    # -- Quality check --
    scorer = QualityScorer()
    quality = scorer.score(review_scope.window.blocks)
    print(f"Quality: score={quality.overall_score:.2f}")
    if quality.warnings:
        for w in quality.warnings:
            print(f"  {w}")

    # -- Inspect the final context --
    print("\n" + "=" * 60)
    print("FINAL REVIEW CONTEXT")
    print("=" * 60)
    print(inspect_window(review_scope.window))

    print(
        f"\nTotal: {len(review_scope.window.blocks)} blocks, "
        f"{review_scope.window.token_count:,}/{review_scope.window.max_tokens:,} tokens"
    )

    # -- Show all agent timelines --
    print("\n" + "=" * 60)
    print("AGENT TIMELINES")
    print("=" * 60)

    for name, scope in [
        ("Architect", arch_scope),
        ("Security Analyst", sec_scope),
        ("Code Reviewer", review_scope),
    ]:
        print(f"\n  {name}:")
        if scope.timeline:
            for snapshot in scope.timeline.snapshots:
                print(
                    f"    Turn {snapshot.turn}: "
                    f"{snapshot.block_count} blocks, "
                    f"{snapshot.token_count:,} tokens, "
                    f"budget {snapshot.budget_percent:.0f}%"
                )

    # -- Show shared memory state --
    print(f"\nShared memory: {shared_memory.list_blocks()}")

    # -- Format for Anthropic --
    payload = AnthropicAdapter().format(review_scope.window)
    print(
        f"\nAnthropic payload: system={len(payload.get('system', ''))} chars, "
        f"messages={len(payload['messages'])}"
    )


if __name__ == "__main__":
    asyncio.run(run_agents())
