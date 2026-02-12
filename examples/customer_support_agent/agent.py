"""Core agent logic for the customer support agent.

Handles context assembly, pipeline optimization, and provider
formatting using contextkit components.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from contextkit.adapters.anthropic_adapter import AnthropicAdapter
from contextkit.core import BlockType, ContextBlock, ContextWindow
from contextkit.memory.in_memory_backend import InMemoryBackend
from contextkit.memory.long_term import LongTermMemory
from contextkit.memory.short_term import ShortTermMemory
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
# Knowledge base articles for the support agent
# ---------------------------------------------------------------

SUPPORT_ARTICLES: List[Chunk] = [
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
# Simulated order database
# ---------------------------------------------------------------

MOCK_ORDERS: Dict[str, Dict[str, Any]] = {
    "ORD-98765": {
        "order_id": "ORD-98765",
        "status": "delivered",
        "items": [
            {"name": "Wireless Headphones", "price": 79.99, "qty": 1},
            {"name": "USB-C Cable", "price": 12.99, "qty": 2},
        ],
        "total": 105.97,
        "delivered_at": "2025-01-28",
        "return_eligible": True,
        "return_deadline": "2025-02-27",
    },
    "ORD-11111": {
        "order_id": "ORD-11111",
        "status": "shipped",
        "items": [
            {"name": "Mechanical Keyboard", "price": 149.99, "qty": 1},
        ],
        "total": 149.99,
        "delivered_at": "",
        "return_eligible": False,
        "return_deadline": "",
    },
}


class SupportAgent:
    """Customer support agent that assembles context using contextkit.

    Manages RAG retrieval, tool selection, conversation memory,
    pipeline optimization, and provider formatting for each
    customer interaction.
    """

    def __init__(self) -> None:
        """Initialize the support agent with all contextkit components."""
        self._rag_retriever = InMemoryRetriever()
        self._rag_context = RAGContext(
            retriever=self._rag_retriever, retriever_name="support-kb"
        )
        self._tool_registry = ToolRegistry()
        self._memory_backend = InMemoryBackend()
        self._long_term_memory = LongTermMemory(backend=self._memory_backend)
        self._prompt_manager = PromptManager()
        self._adapter = AnthropicAdapter()
        self._quality_scorer = QualityScorer()
        self._sufficiency_checker = SufficiencyChecker()

        # Per-customer conversation histories
        self._conversations: Dict[str, ShortTermMemory] = {}

    async def initialize(self) -> None:
        """Load the knowledge base, register tools, and set up prompts.

        Call once at application startup.
        """
        self._rag_retriever.add_chunks(SUPPORT_ARTICLES)
        self._register_tools()
        self._register_prompts()
        await self._seed_long_term_memory()

    def _register_tools(self) -> None:
        """Register all support agent tools with their JSON schemas."""
        self._tool_registry.register(
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
        self._tool_registry.register(
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
                        "enum": [
                            "damaged",
                            "wrong_item",
                            "not_as_described",
                            "changed_mind",
                        ],
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
        self._tool_registry.register(
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

    def _register_prompts(self) -> None:
        """Register versioned prompt templates."""
        self._prompt_manager.register(
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

    async def _seed_long_term_memory(self) -> None:
        """Seed long-term memory with default customer profile data."""
        await self._long_term_memory.store(
            "customer_profile",
            "Name: Jane Smith | Tier: Gold | Member since: 2022 | "
            "Lifetime orders: 47 | Preferred contact: email",
            tags=["customer", "profile"],
            importance=0.9,
        )
        await self._long_term_memory.store(
            "recent_interaction",
            "Last contact 2 weeks ago about a delayed shipment (resolved). "
            "Customer was satisfied with the resolution.",
            tags=["customer", "history"],
            importance=0.6,
        )

    def _get_conversation(self, customer_id: str) -> ShortTermMemory:
        """Get or create a conversation history for a customer.

        Args:
            customer_id: Unique customer identifier.

        Returns:
            The customer's conversation memory instance.
        """
        if customer_id not in self._conversations:
            self._conversations[customer_id] = ShortTermMemory(
                strategy="sliding_window", max_turns=10
            )
        return self._conversations[customer_id]

    def lookup_order(self, order_id: str) -> Dict[str, Any] | None:
        """Look up an order by ID.

        Args:
            order_id: The order identifier.

        Returns:
            Order data dict if found, None otherwise.
        """
        return MOCK_ORDERS.get(order_id)

    def get_conversation_turns(
        self, customer_id: str
    ) -> List[Dict[str, str]]:
        """Get the conversation history for a customer.

        Args:
            customer_id: Unique customer identifier.

        Returns:
            List of turn dicts with role and content keys.
        """
        conversation = self._get_conversation(customer_id)
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in conversation.messages
        ]

    async def handle_message(
        self, customer_id: str, message: str
    ) -> Dict[str, Any]:
        """Process a customer message and assemble the agent context.

        Performs the full context assembly pipeline:
        1. Render system prompt with customer profile
        2. Append conversation history
        3. Retrieve relevant knowledge base articles via RAG
        4. Select applicable tools for the query
        5. Simulate tool execution (order lookup if mentioned)
        6. Run optimization pipeline (dedup, trim, reorder)
        7. Evaluate sufficiency and quality
        8. Format for the Anthropic API

        Args:
            customer_id: Unique customer identifier.
            message: The customer's message text.

        Returns:
            Dict with keys: formatted_payload, context_info.
        """
        conversation = self._get_conversation(customer_id)
        conversation.add_turn("user", message)

        context_window = ContextWindow(max_tokens=4000)

        # 1. System prompt with customer profile from long-term memory
        profile_blocks = await self._long_term_memory.retrieve_as_blocks(
            "customer profile", top_k=2
        )
        profile_text = "\n".join(
            block.content for block in profile_blocks
        ) or "No profile available"
        system_prompt_block = self._prompt_manager.render(
            "support_agent_v1", profile=profile_text
        )
        context_window.add(system_prompt_block)

        # 2. Conversation history
        conversation_block = conversation.to_block()
        context_window.add(conversation_block)

        # 3. RAG retrieval -- relevant support articles
        rag_blocks = await self._rag_context.retrieve(
            query=message, top_k=3, min_relevance=0.5
        )
        for block in rag_blocks:
            context_window.add(block)

        # 4. Tool selection based on the query
        tool_definition_blocks = self._tool_registry.select(message, max_tools=3)
        selected_tool_names = [
            block.name.removeprefix("tool_") for block in tool_definition_blocks
        ]
        for block in tool_definition_blocks:
            context_window.add(block)

        # 5. Simulate tool execution for order lookups
        order_ids = self._extract_order_ids(message)
        for order_id in order_ids:
            order_data = self.lookup_order(order_id)
            if order_data:
                tool_output = self._tool_registry.capture_output(
                    tool_name="lookup_order",
                    result=json.dumps(order_data),
                    latency_ms=45.2,
                )
                context_window.add(tool_output.to_block(priority=75))

        # 6. Pipeline optimization
        pipeline = ContextPipeline(
            steps=[
                DeduplicateStep(similarity_threshold=0.7),
                TrimStep(
                    max_tokens=context_window.max_tokens, min_priority=20
                ),
                ReorderStep(),
            ]
        )
        pipeline.run(context_window)
        pipeline_report = pipeline.last_report

        # 7. Sufficiency and quality checks
        sufficiency_result = self._sufficiency_checker.check(
            message, context_window.blocks
        )
        quality_report = self._quality_scorer.score(context_window.blocks)

        # 8. Format for Anthropic
        formatted_payload = self._adapter.format(context_window)

        # Add a simulated assistant response to conversation
        conversation.add_turn(
            "assistant",
            "I can help you with that. Let me look into it.",
        )

        context_info = {
            "total_blocks": len(context_window.blocks),
            "total_tokens": context_window.token_count,
            "max_tokens": context_window.max_tokens,
            "budget_used_percent": round(
                context_window.token_count / context_window.max_tokens * 100, 1
            ),
            "rag_chunks_retrieved": len(rag_blocks),
            "tools_selected": selected_tool_names,
            "pipeline_tokens_saved": pipeline_report.total_tokens_saved,
            "sufficiency_score": round(sufficiency_result.confidence, 2),
            "quality_score": round(quality_report.overall_score, 2),
        }

        return {
            "formatted_payload": formatted_payload,
            "context_info": context_info,
        }

    @staticmethod
    def _extract_order_ids(text: str) -> List[str]:
        """Extract order IDs matching the pattern ORD-XXXXX from text.

        Args:
            text: Message text to scan for order IDs.

        Returns:
            List of extracted order ID strings.
        """
        import re

        return re.findall(r"ORD-\d+", text)

    @property
    def knowledge_base_size(self) -> int:
        """Number of chunks in the knowledge base."""
        return self._rag_retriever.chunk_count

    @property
    def registered_tool_count(self) -> int:
        """Number of registered tools."""
        return self._tool_registry.tool_count
