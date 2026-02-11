"""Example 4: Multi-Agent System with Context Handoff.

Demonstrates multi-agent context management: isolated agent scopes,
shared memory between agents, scratchpad for working notes, and
structured handoff of context from one agent to another.

Usage:
    python examples/04_multi_agent_handoff.py
"""

import asyncio

from contextkit.core import BlockType, ContextBlock, ContextWindow
from contextkit.memory.in_memory_backend import InMemoryBackend
from contextkit.memory.long_term import LongTermMemory
from contextkit.observe.inspect import inspect_window
from contextkit.scope import (
    ContextScope,
    HandoffPackage,
    Scratchpad,
    SharedMemory,
)


async def main() -> None:
    """Run a multi-agent handoff example."""
    # ---------------------------------------------------------------
    # Step 1: Set up shared resources
    # ---------------------------------------------------------------
    shared_memory = SharedMemory()
    memory_backend = InMemoryBackend()
    long_term_memory = LongTermMemory(backend=memory_backend)

    # Store some long-term memories
    await long_term_memory.store(
        "user_profile",
        "User is a senior Python developer working on an e-commerce "
        "platform. Prefers type-safe code with Pydantic models.",
        tags=["user", "profile"],
        importance=0.9,
    )
    await long_term_memory.store(
        "project_context",
        "The project uses FastAPI, SQLAlchemy, and PostgreSQL. "
        "The team follows TDD and has 90% code coverage target.",
        tags=["project", "tech-stack"],
        importance=0.8,
    )

    print("Shared resources initialized")
    print(f"Long-term memories: {memory_backend.record_count}")

    # ---------------------------------------------------------------
    # Step 2: Research Agent -- gathers and analyzes information
    # ---------------------------------------------------------------
    print("\n=== RESEARCH AGENT ===")

    research_window = ContextWindow(max_tokens=4000)
    research_scope = ContextScope(
        agent_name="researcher",
        window=research_window,
        shared_memory=shared_memory,
        track_history=True,
    )

    # Research agent uses a scratchpad for working notes
    research_pad = Scratchpad()
    research_pad.write("task", "Analyze best practices for API rate limiting")
    research_pad.write(
        "finding_1",
        "Token bucket algorithm: allows bursts while maintaining "
        "average rate. Good for APIs with variable traffic.",
    )
    research_pad.write(
        "finding_2",
        "Sliding window: smoother rate limiting but more memory. "
        "Better for strict per-user quotas.",
    )
    research_pad.write(
        "recommendation",
        "Use token bucket for public endpoints, sliding window "
        "for authenticated per-user limits.",
    )

    # Add research context to the scope's window
    research_scope.window.add(
        ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="You are a research analyst. Gather information "
            "and produce structured findings.",
            priority=100,
            name="research_system_prompt",
        )
    )

    # Retrieve relevant memories
    memory_blocks = await long_term_memory.retrieve_as_blocks(
        "project tech stack",
        top_k=2,
    )
    for block in memory_blocks:
        research_scope.window.add(block)

    # Add scratchpad notes as a block
    research_scope.window.add(research_pad.to_block())

    # Record a turn snapshot
    research_scope.record_turn(events=["research_started"])

    print(
        f"Research scope: {len(research_scope.window.blocks)} blocks, "
        f"{research_scope.window.token_count:,} tokens"
    )
    print(f"Scratchpad notes: {research_pad.note_count}")

    # Publish findings to shared memory
    findings_block = ContextBlock(
        type=BlockType.LONG_TERM_MEMORY,
        content="Rate Limiting Analysis:\n"
        "1. Token bucket for public endpoints (burst-friendly)\n"
        "2. Sliding window for per-user quotas (strict limits)\n"
        "3. Recommended: hybrid approach with both algorithms",
        priority=75,
        name="rate_limiting_findings",
    )
    shared_memory.publish("research_findings", findings_block)

    # Record another turn after publishing
    research_scope.record_turn(events=["findings_published"])

    print(f"Published findings to shared memory ({shared_memory.block_count} blocks)")

    # ---------------------------------------------------------------
    # Step 3: Create handoff package from research to coding agent
    # ---------------------------------------------------------------
    handoff = HandoffPackage(
        source_agent="researcher",
        target_agent="coder",
        blocks=[findings_block, *memory_blocks],
        scratchpad=research_pad,
        metadata={
            "task": "Implement rate limiting middleware",
            "priority": "high",
            "deadline": "sprint-42",
        },
    )

    print(f"\nHandoff created: {handoff.source_agent} -> {handoff.target_agent}")
    print(f"  Blocks: {handoff.block_count}")
    print(f"  Metadata: {handoff.metadata}")

    # ---------------------------------------------------------------
    # Step 4: Coding Agent -- receives handoff and implements
    # ---------------------------------------------------------------
    print("\n=== CODING AGENT ===")

    coder_window = ContextWindow(max_tokens=4000)
    coder_scope = ContextScope(
        agent_name="coder",
        window=coder_window,
        shared_memory=shared_memory,
        track_history=True,
    )

    # Add system prompt for the coding agent
    coder_scope.window.add(
        ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="You are a Python developer. Implement code based on "
            "research findings. Follow the project's coding standards.",
            priority=100,
            name="coder_system_prompt",
        )
    )

    # Receive the handoff package (adds blocks + imports scratchpad notes)
    blocks_received = coder_scope.receive_handoff(handoff)
    print(f"Received {blocks_received} blocks from handoff")

    # Read the researcher's scratchpad notes (now in coder's scratchpad)
    recommendation = coder_scope.scratchpad.read(
        "from_researcher_recommendation",
    )
    if recommendation:
        print(f"Received recommendation: {recommendation}")

    # Import shared memory findings
    imported = coder_scope.import_shared(["research_findings"])
    print(f"Imported {imported} blocks from shared memory")

    # Record the initial turn
    coder_scope.record_turn(events=["handoff_received"])

    # Coder creates their own scratchpad notes
    coder_scope.scratchpad.write(
        "approach",
        "FastAPI middleware with token bucket",
    )
    coder_scope.scratchpad.write("status", "implementing core logic")

    # Record another turn
    coder_scope.record_turn(events=["implementation_started"])

    print(
        f"Coder scope: {len(coder_scope.window.blocks)} blocks, "
        f"{coder_scope.window.token_count:,} tokens"
    )

    # ---------------------------------------------------------------
    # Step 5: Inspect both agent contexts
    # ---------------------------------------------------------------
    print("\n--- Research Agent Context ---")
    print(inspect_window(research_scope.window))

    print("\n--- Coding Agent Context ---")
    print(inspect_window(coder_scope.window))

    # ---------------------------------------------------------------
    # Step 6: Show timeline tracking
    # ---------------------------------------------------------------
    print("\n--- Research Agent Timeline ---")
    research_timeline = research_scope.timeline
    if research_timeline:
        for snapshot in research_timeline.snapshots:
            print(
                f"  Turn {snapshot.turn}: "
                f"{snapshot.block_count} blocks, "
                f"{snapshot.token_count:,} tokens, "
                f"budget {snapshot.budget_percent:.0f}%"
            )

    print("\n--- Coding Agent Timeline ---")
    coder_timeline = coder_scope.timeline
    if coder_timeline:
        for snapshot in coder_timeline.snapshots:
            print(
                f"  Turn {snapshot.turn}: "
                f"{snapshot.block_count} blocks, "
                f"{snapshot.token_count:,} tokens, "
                f"budget {snapshot.budget_percent:.0f}%"
            )

    # ---------------------------------------------------------------
    # Step 7: Show shared memory state
    # ---------------------------------------------------------------
    print(f"\nShared memory blocks: {shared_memory.list_blocks()}")


if __name__ == "__main__":
    asyncio.run(main())
