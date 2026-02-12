"""Customer Support Agent -- end-to-end agentic use case.

Combines RAG knowledge retrieval, tool usage (order lookup, refund),
conversation memory, pipeline optimization, sufficiency checking,
and provider-agnostic formatting into a complete support agent.

Usage:
    python examples/customer_support_agent/main.py
"""

import asyncio
import json

from contextkit.adapters.anthropic_adapter import AnthropicAdapter
from contextkit.core import BlockType, ContextBlock, ContextWindow
from contextkit.memory.in_memory_backend import InMemoryBackend
from contextkit.memory.long_term import LongTermMemory
from contextkit.memory.short_term import ShortTermMemory
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
from contextkit.tools.tool_registry import ToolRegistry


# ---------------------------------------------------------------
# 1. Knowledge Base -- support articles indexed for RAG
# ---------------------------------------------------------------

SUPPORT_ARTICLES = [
    Chunk(
        content="To return an item, go to Orders > Select Order > Request Return. "
        "Returns are accepted within 30 days of delivery for most items. "
        "Electronics have a 15-day return window. Items must be in original "
        "packaging with all accessories included.",
        source="help/returns-policy.md",
        relevance_score=0.95,
        metadata={"category": "returns"},
    ),
    Chunk(
        content="Refunds are processed within 5-7 business days after we receive "
        "the returned item. The refund goes to the original payment method. "
        "Shipping costs are non-refundable unless the return is due to our error.",
        source="help/refund-timeline.md",
        relevance_score=0.90,
        metadata={"category": "refunds"},
    ),
    Chunk(
        content="For order tracking, use the tracking number in your shipment "
        "confirmation email. Orders typically ship within 1-2 business days. "
        "Standard delivery takes 3-5 business days. Express delivery is 1-2 days.",
        source="help/shipping-tracking.md",
        relevance_score=0.85,
        metadata={"category": "shipping"},
    ),
    Chunk(
        content="To change your subscription plan, go to Account > Subscription > "
        "Change Plan. Upgrades take effect immediately. Downgrades apply at "
        "the next billing cycle. Cancel anytime with no penalty.",
        source="help/subscription-changes.md",
        relevance_score=0.70,
        metadata={"category": "subscriptions"},
    ),
    Chunk(
        content="If you received a damaged or incorrect item, contact support "
        "immediately. We will arrange a free return and send a replacement "
        "or full refund including shipping costs. Photo evidence may be required.",
        source="help/damaged-items.md",
        relevance_score=0.92,
        metadata={"category": "returns"},
    ),
]


# ---------------------------------------------------------------
# 2. Tool Definitions -- order lookup, refund, escalate
# ---------------------------------------------------------------


def register_tools(registry: ToolRegistry) -> None:
    """Register all support agent tools."""
    registry.register(
        name="lookup_order",
        description="Look up an order by order ID. Returns order status, "
        "items, shipping info, and delivery date.",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID (e.g., ORD-12345)",
                },
            },
            "required": ["order_id"],
        },
        tags=["orders", "lookup"],
    )

    registry.register(
        name="process_refund",
        description="Initiate a refund for an order. Requires order ID and reason.",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID to refund",
                },
                "reason": {
                    "type": "string",
                    "enum": ["damaged", "wrong_item", "not_as_described", "changed_mind"],
                },
                "amount": {
                    "type": "number",
                    "description": "Refund amount in USD (optional, defaults to full)",
                },
            },
            "required": ["order_id", "reason"],
        },
        tags=["orders", "refund"],
    )

    registry.register(
        name="escalate_to_human",
        description="Escalate the conversation to a human support agent. "
        "Use when the issue is complex or the customer is frustrated.",
        parameters={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Brief description of why escalation is needed",
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "urgent"],
                },
            },
            "required": ["reason", "priority"],
        },
        tags=["escalation"],
    )


# ---------------------------------------------------------------
# 3. Simulated tool execution
# ---------------------------------------------------------------


def simulate_order_lookup(order_id: str) -> str:
    """Simulate an order lookup API call."""
    return json.dumps({
        "order_id": order_id,
        "status": "delivered",
        "items": [
            {"name": "Wireless Headphones", "price": 79.99, "qty": 1},
            {"name": "USB-C Cable", "price": 12.99, "qty": 2},
        ],
        "total": 105.97,
        "delivered_at": "2025-01-28",
        "return_eligible": True,
        "return_deadline": "2025-02-12",
    })


# ---------------------------------------------------------------
# 4. Main agent loop
# ---------------------------------------------------------------


async def run_agent() -> None:
    """Run the customer support agent."""
    # -- Setup components --
    retriever = InMemoryRetriever()
    retriever.add_chunks(SUPPORT_ARTICLES)
    rag = RAGContext(retriever=retriever, retriever_name="support-kb")

    tool_registry = ToolRegistry()
    register_tools(tool_registry)

    memory_backend = InMemoryBackend()
    long_term = LongTermMemory(backend=memory_backend)

    conversation = ShortTermMemory(strategy="sliding_window", max_turns=10)

    prompt_mgr = PromptManager()
    prompt_mgr.register(
        "support_agent_v1",
        "You are a helpful customer support agent for ShopCo.\n\n"
        "Guidelines:\n"
        "- Be empathetic and professional\n"
        "- Use the knowledge base to answer policy questions\n"
        "- Use tools to look up orders and process refunds\n"
        "- Escalate to a human agent if the issue is complex\n\n"
        "Customer profile:\n{{profile}}",
        description="Main support agent system prompt",
    )

    # -- Store customer profile in long-term memory --
    await long_term.store(
        "customer_profile",
        "Name: Jane Smith | Tier: Gold | Member since: 2022 | "
        "Lifetime orders: 47 | Preferred contact: email",
        tags=["customer", "profile"],
        importance=0.9,
    )
    await long_term.store(
        "recent_interaction",
        "Last contact 2 weeks ago about a delayed shipment (resolved). "
        "Customer was satisfied with the resolution.",
        tags=["customer", "history"],
        importance=0.6,
    )

    # -- Simulate a multi-turn conversation --
    print("=" * 60)
    print("CUSTOMER SUPPORT AGENT")
    print("=" * 60)

    conversation.add_turn(
        "user",
        "Hi, I received my order ORD-98765 but the headphones "
        "seem to be damaged. The left ear cup has a crack.",
    )
    conversation.add_turn(
        "assistant",
        "I'm sorry to hear about the damaged headphones. Let me "
        "look up your order right away.",
    )
    conversation.add_turn("user", "Thanks. Can I get a refund or replacement?")

    # -- Build the context window for this turn --
    window = ContextWindow(max_tokens=4000)
    user_query = "Can I get a refund or replacement for damaged headphones?"

    # 1. System prompt with customer profile
    profile_blocks = await long_term.retrieve_as_blocks("customer profile", top_k=2)
    profile_text = "\n".join(
        b.content for b in profile_blocks
    ) or "No profile available"

    system_block = prompt_mgr.render("support_agent_v1", profile=profile_text)
    window.add(system_block)
    print(f"\n[System prompt] {system_block.token_count} tokens")

    # 2. Conversation history
    history_block = conversation.to_block()
    window.add(history_block)
    print(f"[Conversation]  {history_block.token_count} tokens ({conversation.turn_count} turns)")

    # 3. RAG -- retrieve relevant support articles
    rag_blocks = await rag.retrieve(query=user_query, top_k=3, min_relevance=0.5)
    for block in rag_blocks:
        window.add(block)
    print(f"[RAG]           {len(rag_blocks)} chunks retrieved")

    # 4. Tool definitions
    tool_blocks = tool_registry.select(user_query, max_tools=3)
    for block in tool_blocks:
        window.add(block)
    print(f"[Tools]         {len(tool_blocks)} tools selected")

    # 5. Simulated tool output (order lookup)
    order_result = simulate_order_lookup("ORD-98765")
    tool_output = tool_registry.capture_output(
        tool_name="lookup_order",
        result=order_result,
        latency_ms=45.2,
    )
    window.add(tool_output.to_block(priority=75))
    print(f"[Tool output]   lookup_order ({tool_output.latency_ms}ms)")

    # -- Pipeline: optimize context --
    pipeline = ContextPipeline(
        steps=[
            DeduplicateStep(similarity_threshold=0.7),
            TrimStep(max_tokens=window.max_tokens, min_priority=20),
            ReorderStep(),
        ]
    )
    pipeline.run(window)

    report = pipeline.last_report
    print(f"\n[Pipeline]      {report.total_tokens_saved} tokens saved")

    # -- Sufficiency check --
    checker = SufficiencyChecker()
    sufficiency = checker.check(user_query, window.blocks)
    print(f"[Sufficiency]   sufficient={sufficiency.sufficient}, "
          f"confidence={sufficiency.confidence:.2f}, "
          f"coverage={sufficiency.query_coverage:.0%}")
    if sufficiency.suggestions:
        for s in sufficiency.suggestions:
            print(f"                {s}")

    # -- Quality check --
    scorer = QualityScorer()
    quality = scorer.score(window.blocks)
    print(f"[Quality]       score={quality.overall_score:.2f}")
    if quality.warnings:
        for w in quality.warnings:
            print(f"                {w}")

    # -- Inspect the final context --
    print("\n" + "=" * 60)
    print("ASSEMBLED CONTEXT")
    print("=" * 60)
    print(inspect_window(window))

    print(f"\nTotal: {len(window.blocks)} blocks, "
          f"{window.token_count:,}/{window.max_tokens:,} tokens")

    # -- Format for Anthropic --
    payload = AnthropicAdapter().format(window)
    print(f"\nAnthropic payload: system={len(payload.get('system', ''))} chars, "
          f"messages={len(payload['messages'])}")


if __name__ == "__main__":
    asyncio.run(run_agent())
