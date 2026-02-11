"""Example 5: Budget-Aware Context Pipeline.

Demonstrates the context pipeline system: trimming low-priority blocks,
deduplicating overlapping content, compacting verbose blocks, reordering
for optimal placement, and filtering by relevance -- all within a
token budget.

Usage:
    python examples/05_budget_aware_pipeline.py
"""

from contextkit.core import BlockType, ContextBlock, ContextWindow
from contextkit.observe.inspect import inspect_window
from contextkit.pipeline import (
    CompactStep,
    ContextPipeline,
    DeduplicateStep,
    FilterStep,
    ReorderStep,
    TrimStep,
)


def main() -> None:
    """Run a budget-aware pipeline example."""
    # ---------------------------------------------------------------
    # Step 1: Create a context window with a tight budget
    # ---------------------------------------------------------------
    window = ContextWindow(max_tokens=500)
    print(f"Token budget: {window.max_tokens:,}")

    # ---------------------------------------------------------------
    # Step 2: Add many blocks that exceed the budget
    # ---------------------------------------------------------------

    # High priority: system prompt (always keep)
    window.add(
        ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="You are a concise technical assistant.",
            priority=100,
            name="system_prompt",
        )
    )

    # High priority: recent user context
    window.add(
        ContextBlock(
            type=BlockType.USER_CONTEXT,
            content="Help me debug a memory leak in my Python application.",
            priority=90,
            name="user_query",
        )
    )

    # Medium priority: RAG chunks with some overlap
    window.add(
        ContextBlock(
            type=BlockType.RAG,
            content="Memory leaks in Python often occur with circular references. "
            "The garbage collector handles most cases, but C extensions "
            "and __del__ methods can prevent collection. Use objgraph "
            "or tracemalloc to identify leaked objects.",
            priority=70,
            name="rag_memory_leaks",
        )
    )

    window.add(
        ContextBlock(
            type=BlockType.RAG,
            content="Python memory leaks happen with circular references between "
            "objects. The gc module can help detect cycles. Tools like "
            "objgraph visualize object reference graphs to find leaks.",
            priority=70,
            name="rag_memory_leaks_duplicate",
        )
    )

    window.add(
        ContextBlock(
            type=BlockType.RAG,
            content="tracemalloc is a Python standard library module for tracking "
            "memory allocations. Enable it with tracemalloc.start() at "
            "program start. Use take_snapshot() to capture allocation "
            "state and compare_to() to find growth between snapshots. "
            "This is the recommended first step for debugging memory "
            "issues in production Python applications.",
            priority=65,
            name="rag_tracemalloc",
        )
    )

    # Lower priority: general context
    window.add(
        ContextBlock(
            type=BlockType.LONG_TERM_MEMORY,
            content="User has been working on a Flask web application with "
            "SQLAlchemy ORM. Previous sessions discussed database "
            "connection pooling and query optimization.",
            priority=50,
            name="memory_project_history",
        )
    )

    # Low priority: older example
    window.add(
        ContextBlock(
            type=BlockType.EXAMPLES,
            content="Example: To find memory leaks, first add "
            "tracemalloc.start() to your main module, then "
            "periodically call tracemalloc.take_snapshot().",
            priority=40,
            name="example_tracemalloc",
        )
    )

    print(
        f"\nBefore pipeline: {len(window.blocks)} blocks, {window.token_count:,} tokens"
    )
    print(f"Over budget by: {window.token_count - window.max_tokens:,} tokens")

    # ---------------------------------------------------------------
    # Step 3: Configure and run the pipeline
    # ---------------------------------------------------------------
    pipeline = ContextPipeline(
        steps=[
            # Step 1: Remove near-duplicate content
            DeduplicateStep(similarity_threshold=0.7),
            # Step 2: Compact verbose blocks (shrink to 60% size)
            CompactStep(target_ratio=0.6, min_tokens=30),
            # Step 3: Trim low-priority blocks to fit budget
            TrimStep(
                max_tokens=window.max_tokens,
                min_priority=20,
            ),
            # Step 4: Reorder for optimal placement
            # (system prompts first, then by priority)
            ReorderStep(),
        ]
    )

    # Run the pipeline
    pipeline.run(window)
    report = pipeline.last_report

    print("\n--- Pipeline Report ---")
    print(f"Steps executed: {len(report.steps)}")

    for step_report in report.steps:
        print(f"\n  {step_report.step_name}:")
        print(
            f"    Tokens: {step_report.tokens_before:,} -> "
            f"{step_report.tokens_after:,} "
            f"(saved {step_report.tokens_saved:,})"
        )
        print(f"    Blocks removed: {step_report.blocks_removed}")
        if step_report.details:
            for key, value in step_report.details.items():
                print(f"    {key}: {value}")

    print(
        f"\nAfter pipeline: {len(window.blocks)} blocks, {window.token_count:,} tokens"
    )
    print(f"Within budget: {window.token_count <= window.max_tokens}")

    # ---------------------------------------------------------------
    # Step 4: Inspect the optimized context
    # ---------------------------------------------------------------
    print("\n--- Optimized Context ---")
    print(inspect_window(window))

    # ---------------------------------------------------------------
    # Step 5: Check mutation history on surviving blocks
    # ---------------------------------------------------------------
    print("\n--- Mutation History ---")
    for block in window.blocks:
        if block.mutations:
            print(f"\n  {block.display_name}:")
            for mutation in block.mutations:
                print(f"    [{mutation.step}] {mutation.action}: {mutation.detail}")

    # ---------------------------------------------------------------
    # Step 6: Show final statistics
    # ---------------------------------------------------------------
    print("\n--- Final Statistics ---")
    print(f"Blocks: {len(window.blocks)}")
    print(f"Tokens before pipeline: {report.total_tokens_before:,}")
    print(f"Tokens after pipeline: {report.total_tokens_after:,}")
    print(f"Total saved: {report.total_tokens_saved:,} tokens")

    # ---------------------------------------------------------------
    # Step 7: Demonstrate the FilterStep separately
    # ---------------------------------------------------------------
    print("\n--- FilterStep Demo ---")
    filter_window = ContextWindow(max_tokens=1000)
    filter_window.add(
        ContextBlock(
            type=BlockType.SYSTEM_PROMPT,
            content="System prompt",
            priority=100,
            name="system",
        )
    )
    filter_window.add(
        ContextBlock(
            type=BlockType.RAG,
            content="Relevant content about memory management",
            priority=80,
            name="relevant_rag",
        )
    )
    filter_window.add(
        ContextBlock(
            type=BlockType.RAG,
            content="Unrelated content about cooking recipes",
            priority=30,
            name="irrelevant_rag",
        )
    )

    # Filter out low-priority blocks using a custom filter function
    filter_pipeline = ContextPipeline(
        steps=[
            FilterStep(
                filter_fn=lambda block: block.priority >= 50,
            ),
        ]
    )
    filter_pipeline.run(filter_window)
    filter_report = filter_pipeline.last_report

    print(f"Blocks removed: {filter_report.steps[0].blocks_removed}")
    print(f"Remaining: {[b.name for b in filter_window.blocks]}")


if __name__ == "__main__":
    main()
