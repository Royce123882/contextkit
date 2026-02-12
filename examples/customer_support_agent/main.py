"""Customer Support Agent -- FastAPI application.

A customer support chatbot that uses contextkit for context assembly:
RAG knowledge retrieval, tool usage, conversation memory, pipeline
optimization, and provider-agnostic formatting.

Usage:
    uvicorn examples.customer_support_agent.main:app --reload
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException

from examples.customer_support_agent.agent import SupportAgent
from examples.customer_support_agent.models import (
    ChatRequest,
    ChatResponse,
    ContextInfo,
    ConversationHistoryResponse,
    ConversationTurn,
    HealthResponse,
    OrderResponse,
)

support_agent = SupportAgent()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Initialize the support agent on startup."""
    await support_agent.initialize()
    yield


app = FastAPI(
    title="Customer Support Agent",
    description="AI-powered customer support using contextkit for "
    "context assembly, RAG retrieval, and pipeline optimization.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a customer message and return the assembled agent context.

    Assembles context from the knowledge base, conversation history,
    tools, and long-term memory. Runs the optimization pipeline and
    formats the result for the Anthropic API.

    Args:
        request: The incoming chat message.

    Returns:
        The agent response with formatted payload and context metadata.
    """
    agent_result = await support_agent.handle_message(
        customer_id=request.customer_id,
        message=request.message,
    )

    return ChatResponse(
        customer_id=request.customer_id,
        formatted_payload=agent_result["formatted_payload"],
        context_info=ContextInfo(**agent_result["context_info"]),
    )


@app.get(
    "/chat/history/{customer_id}",
    response_model=ConversationHistoryResponse,
)
async def get_conversation_history(
    customer_id: str,
) -> ConversationHistoryResponse:
    """Retrieve the conversation history for a customer.

    Args:
        customer_id: Unique customer identifier.

    Returns:
        The customer's conversation turns and turn count.
    """
    turns_data = support_agent.get_conversation_turns(customer_id)
    turns = [ConversationTurn(**turn) for turn in turns_data]

    return ConversationHistoryResponse(
        customer_id=customer_id,
        turns=turns,
        turn_count=len(turns),
    )


@app.get("/orders/{order_id}", response_model=OrderResponse)
async def get_order(order_id: str) -> OrderResponse:
    """Look up an order by ID.

    Args:
        order_id: The order identifier (e.g., ORD-98765).

    Returns:
        Order details including status, items, and return eligibility.

    Raises:
        HTTPException: 404 if the order is not found.
    """
    order_data = support_agent.lookup_order(order_id)
    if order_data is None:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")

    return OrderResponse(**order_data)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check the service health and component status.

    Returns:
        Service status with knowledge base and tool counts.
    """
    return HealthResponse(
        status="healthy",
        knowledge_base_chunks=support_agent.knowledge_base_size,
        registered_tools=support_agent.registered_tool_count,
    )


if __name__ == "__main__":
    uvicorn.run(
        "examples.customer_support_agent.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
