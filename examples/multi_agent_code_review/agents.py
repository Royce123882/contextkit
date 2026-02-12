"""Agent definitions for the multi-agent code review system.

Contains three specialized agents (Architect, Security Analyst,
Code Reviewer) that collaborate via shared memory and handoffs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from contextkit.core import BlockType, ContextBlock, ContextWindow
from contextkit.memory.long_term import LongTermMemory
from contextkit.observe.quality import QualityScorer
from contextkit.observe.sufficiency import SufficiencyChecker
from contextkit.pipeline import (
    ContextPipeline,
    DeduplicateStep,
    ReorderStep,
    TrimStep,
)
from contextkit.prompts.prompt_manager import PromptManager
from contextkit.rag.context import RAGContext
from contextkit.scope import ContextScope, SharedMemory
from contextkit.tools.tool_registry import ToolRegistry


# ---------------------------------------------------------------
# Simulated tool outputs
# ---------------------------------------------------------------


def simulate_static_analysis(files: List[str]) -> str:
    """Simulate running static analysis on code files.

    Args:
        files: List of file paths to analyze.

    Returns:
        JSON string of analysis results.
    """
    return json.dumps({
        "files_analyzed": files,
        "issues": [
            {
                "rule": "S608",
                "severity": "high",
                "message": "Possible SQL injection via string formatting",
                "line": 7,
            },
            {
                "rule": "S608",
                "severity": "high",
                "message": "Possible SQL injection via string formatting",
                "line": 10,
            },
            {
                "rule": "E722",
                "severity": "medium",
                "message": "Bare except clause",
                "line": 19,
            },
            {
                "rule": "ANN001",
                "severity": "low",
                "message": "Missing type annotation for parameter",
                "line": 5,
            },
        ],
        "summary": {"high": 2, "medium": 1, "low": 1},
    })


def simulate_test_results(modules: List[str]) -> str:
    """Simulate running the test suite.

    Args:
        modules: List of modules to test.

    Returns:
        JSON string of test results.
    """
    return json.dumps({
        "modules": modules,
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
# Architect Agent
# ---------------------------------------------------------------


async def run_architect_agent(
    shared_memory: SharedMemory,
    rag_context: RAGContext,
    prompt_manager: PromptManager,
    long_term_memory: LongTermMemory,
) -> Dict[str, Any]:
    """Run the architect agent to review code structure and design.

    Retrieves coding standards via RAG, analyzes the code for
    design quality issues, and publishes findings to shared memory.

    Args:
        shared_memory: Shared memory for cross-agent communication.
        rag_context: RAG context with coding standards knowledge base.
        prompt_manager: Prompt manager with registered templates.
        long_term_memory: Long-term memory with project context.

    Returns:
        Agent report dict with findings, notes, and context info.
    """
    architect_window = ContextWindow(max_tokens=3000)
    architect_scope = ContextScope(
        agent_name="architect",
        window=architect_window,
        shared_memory=shared_memory,
        track_history=True,
    )

    # System prompt with project context
    project_blocks = await long_term_memory.retrieve_as_blocks(
        "project config", top_k=1
    )
    project_context = project_blocks[0].content if project_blocks else "N/A"
    system_prompt_block = prompt_manager.render(
        "architect_v1", project=project_context
    )
    architect_scope.window.add(system_prompt_block)

    # Import code under review from shared memory
    architect_scope.import_shared(["pr_code"])

    # RAG: coding standards for architecture
    architecture_standards = await rag_context.retrieve(
        query="code quality type safety API design error handling",
        top_k=3,
        min_relevance=0.5,
    )
    for block in architecture_standards:
        architect_scope.window.add(block)

    # Architect's analysis (scratchpad)
    architect_scope.scratchpad.write(
        "finding_1",
        "CRITICAL: update_profile accepts raw dict instead of Pydantic model. "
        "No input validation whatsoever.",
    )
    architect_scope.scratchpad.write(
        "finding_2",
        "MEDIUM: update_profile does two things (update profile + update role). "
        "Violates SRP. Role updates should be a separate endpoint.",
    )
    architect_scope.scratchpad.write(
        "finding_3",
        "MEDIUM: export_user_data uses bare except. Should catch specific "
        "exceptions and return proper HTTP error codes.",
    )
    architect_scope.scratchpad.write(
        "finding_4",
        "LOW: Missing response models. Return types should use Pydantic schemas.",
    )
    architect_scope.window.add(architect_scope.scratchpad.to_block())

    # Publish findings to shared memory
    architect_findings_block = ContextBlock(
        type=BlockType.LONG_TERM_MEMORY,
        content="Architecture Review Findings:\n"
        "1. [CRITICAL] No input validation - raw dict parameter\n"
        "2. [MEDIUM] SRP violation - role update mixed with profile update\n"
        "3. [MEDIUM] Bare except clause in export endpoint\n"
        "4. [LOW] Missing Pydantic response models",
        priority=80,
        name="architect_findings",
    )
    architect_scope.window.add(architect_findings_block)
    shared_memory.publish("architect_findings", architect_findings_block)

    architect_scope.record_turn(events=["architecture_review_complete"])

    findings = [
        {"severity": "critical", "description": "No input validation - raw dict parameter"},
        {"severity": "medium", "description": "SRP violation - role update mixed with profile update"},
        {"severity": "medium", "description": "Bare except clause in export endpoint"},
        {"severity": "low", "description": "Missing Pydantic response models"},
    ]

    scratchpad_notes = {
        key: architect_scope.scratchpad.read(key) or ""
        for key in architect_scope.scratchpad.list_keys()
    }

    return {
        "agent_name": "architect",
        "findings": findings,
        "scratchpad_notes": scratchpad_notes,
        "block_count": len(architect_scope.window.blocks),
        "token_count": architect_scope.window.token_count,
        "scope": architect_scope,
    }


# ---------------------------------------------------------------
# Security Analyst Agent
# ---------------------------------------------------------------


async def run_security_agent(
    shared_memory: SharedMemory,
    rag_context: RAGContext,
    tool_registry: ToolRegistry,
    prompt_manager: PromptManager,
    long_term_memory: LongTermMemory,
) -> Dict[str, Any]:
    """Run the security analyst agent to find vulnerabilities.

    Imports architect findings, runs static analysis and tests,
    retrieves security standards via RAG, and publishes security
    findings to shared memory.

    Args:
        shared_memory: Shared memory for cross-agent communication.
        rag_context: RAG context with coding standards knowledge base.
        tool_registry: Registry for analysis and testing tools.
        prompt_manager: Prompt manager with registered templates.
        long_term_memory: Long-term memory with incident history.

    Returns:
        Agent report dict with findings, notes, and context info.
    """
    security_window = ContextWindow(max_tokens=3000)
    security_scope = ContextScope(
        agent_name="security_analyst",
        window=security_window,
        shared_memory=shared_memory,
        track_history=True,
    )

    # System prompt with incident history
    incident_blocks = await long_term_memory.retrieve_as_blocks(
        "security incident", top_k=1
    )
    incident_context = incident_blocks[0].content if incident_blocks else "None"
    system_prompt_block = prompt_manager.render(
        "security_analyst_v1", incidents=incident_context
    )
    security_scope.window.add(system_prompt_block)

    # Import code + architect findings from shared memory
    security_scope.import_shared(["pr_code", "architect_findings"])

    # RAG: security-specific standards
    security_standards = await rag_context.retrieve(
        query="SQL injection authentication security data exposure",
        top_k=3,
        min_relevance=0.5,
    )
    for block in security_standards:
        security_scope.window.add(block)

    # Tool outputs: static analysis + test results
    analysis_json = simulate_static_analysis(["api/routes/users.py"])
    analysis_output = tool_registry.capture_output(
        tool_name="run_static_analysis",
        result=analysis_json,
        latency_ms=890.3,
    )
    security_scope.window.add(analysis_output.to_block(priority=75))

    test_json = simulate_test_results(["api.routes.users"])
    test_output = tool_registry.capture_output(
        tool_name="run_tests",
        result=test_json,
        latency_ms=2340.7,
    )
    security_scope.window.add(test_output.to_block(priority=75))

    # Security analyst's findings (scratchpad)
    security_scope.scratchpad.write(
        "vuln_1",
        "CRITICAL: SQL injection on lines 7, 10, 12, 17. Uses f-string "
        "formatting to build SQL queries. Must use parameterized queries.",
    )
    security_scope.scratchpad.write(
        "vuln_2",
        "CRITICAL: No authentication on either endpoint. User can update "
        "any profile including role escalation.",
    )
    security_scope.scratchpad.write(
        "vuln_3",
        "HIGH: export_user_data exposes all columns (SELECT *). May leak "
        "sensitive fields. Use explicit column selection.",
    )
    security_scope.scratchpad.write(
        "vuln_4",
        "HIGH: No rate limiting on export endpoint. Could be used for "
        "bulk data scraping.",
    )
    security_scope.window.add(security_scope.scratchpad.to_block())

    # Publish security findings
    security_findings_block = ContextBlock(
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
    security_scope.window.add(security_findings_block)
    shared_memory.publish("security_findings", security_findings_block)

    security_scope.record_turn(events=["security_review_complete"])

    findings = [
        {"severity": "critical", "description": "SQL injection (4 instances) - use parameterized queries"},
        {"severity": "critical", "description": "No authentication or authorization checks"},
        {"severity": "high", "description": "Data exposure via SELECT * - use explicit columns"},
        {"severity": "high", "description": "No rate limiting on data export endpoint"},
    ]

    scratchpad_notes = {
        key: security_scope.scratchpad.read(key) or ""
        for key in security_scope.scratchpad.list_keys()
    }

    return {
        "agent_name": "security_analyst",
        "findings": findings,
        "scratchpad_notes": scratchpad_notes,
        "block_count": len(security_scope.window.blocks),
        "token_count": security_scope.window.token_count,
        "scope": security_scope,
    }


# ---------------------------------------------------------------
# Code Reviewer (Synthesizer) Agent
# ---------------------------------------------------------------


def run_synthesizer_agent(
    shared_memory: SharedMemory,
    architect_scope: ContextScope,
    security_scope: ContextScope,
    prompt_manager: PromptManager,
) -> Dict[str, Any]:
    """Run the synthesizer agent to consolidate findings.

    Receives handoffs from both specialized agents, imports the
    original code, synthesizes all findings, and produces a final
    review decision with a prioritized fix plan.

    Args:
        shared_memory: Shared memory for cross-agent communication.
        architect_scope: The architect agent's completed scope.
        security_scope: The security analyst's completed scope.
        prompt_manager: Prompt manager with registered templates.

    Returns:
        Synthesizer report dict with decision, findings, and context info.
    """
    reviewer_window = ContextWindow(max_tokens=4000)
    reviewer_scope = ContextScope(
        agent_name="code_reviewer",
        window=reviewer_window,
        shared_memory=shared_memory,
        track_history=True,
    )

    # System prompt
    system_prompt_block = prompt_manager.render("code_reviewer_v1")
    reviewer_scope.window.add(system_prompt_block)

    # Receive handoffs from both agents
    architect_handoff = architect_scope.handoff(
        target_agent="code_reviewer",
        block_names=["architect_findings"],
        metadata={"review_type": "architecture"},
    )
    security_handoff = security_scope.handoff(
        target_agent="code_reviewer",
        block_names=["security_findings"],
        metadata={"review_type": "security"},
    )
    reviewer_scope.receive_handoff(architect_handoff)
    reviewer_scope.receive_handoff(security_handoff)

    # Import the original code
    reviewer_scope.import_shared(["pr_code"])

    # Synthesize findings in scratchpad
    reviewer_scope.scratchpad.write(
        "decision",
        "REQUEST CHANGES - Multiple critical security issues",
    )
    reviewer_scope.scratchpad.write(
        "summary",
        "2 critical, 2 high, 2 medium, 1 low issues found. "
        "PR requires significant rework before merge.",
    )
    reviewer_scope.scratchpad.write(
        "fix_priority",
        "1. Fix SQL injection (critical/immediate)\n"
        "2. Add authentication middleware (critical/immediate)\n"
        "3. Add Pydantic input validation (high/before merge)\n"
        "4. Fix data exposure with explicit columns (high/before merge)\n"
        "5. Split role update to separate endpoint (medium/next sprint)\n"
        "6. Fix error handling (medium/next sprint)\n"
        "7. Add response models (low/backlog)",
    )
    reviewer_scope.window.add(reviewer_scope.scratchpad.to_block())

    reviewer_scope.record_turn(events=["synthesis_complete"])

    # Pipeline optimization
    pipeline = ContextPipeline(
        steps=[
            DeduplicateStep(similarity_threshold=0.7),
            TrimStep(max_tokens=reviewer_window.max_tokens, min_priority=20),
            ReorderStep(strategy="important_edges"),
        ]
    )
    pipeline.run(reviewer_scope.window)
    pipeline_report = pipeline.last_report

    # Quality checks
    sufficiency_checker = SufficiencyChecker()
    sufficiency_result = sufficiency_checker.check(
        "code review security architecture", reviewer_scope.window.blocks
    )
    quality_scorer = QualityScorer()
    quality_report = quality_scorer.score(reviewer_scope.window.blocks)

    synthesizer_context = {
        "total_blocks": len(reviewer_scope.window.blocks),
        "total_tokens": reviewer_scope.window.token_count,
        "max_tokens": reviewer_scope.window.max_tokens,
        "pipeline_tokens_saved": pipeline_report.total_tokens_saved,
        "sufficiency_score": round(sufficiency_result.confidence, 2),
        "quality_score": round(quality_report.overall_score, 2),
    }

    scratchpad_notes = {
        key: reviewer_scope.scratchpad.read(key) or ""
        for key in reviewer_scope.scratchpad.list_keys()
    }

    return {
        "agent_name": "code_reviewer",
        "decision": "request_changes",
        "summary": (
            "2 critical, 2 high, 2 medium, 1 low issues found. "
            "PR requires significant rework before merge."
        ),
        "fix_priority": [
            "Fix SQL injection (critical/immediate)",
            "Add authentication middleware (critical/immediate)",
            "Add Pydantic input validation (high/before merge)",
            "Fix data exposure with explicit columns (high/before merge)",
            "Split role update to separate endpoint (medium/next sprint)",
            "Fix error handling (medium/next sprint)",
            "Add response models (low/backlog)",
        ],
        "scratchpad_notes": scratchpad_notes,
        "block_count": len(reviewer_scope.window.blocks),
        "token_count": reviewer_scope.window.token_count,
        "scope": reviewer_scope,
        "synthesizer_context": synthesizer_context,
    }
