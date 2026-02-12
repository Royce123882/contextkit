"""Pydantic request/response models for the customer support API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------
# Request models
# ---------------------------------------------------------------


class ChatRequest(BaseModel):
    """Incoming chat message from a customer.

    Attributes:
        customer_id: Unique identifier for the customer.
        message: The customer's message text.
    """

    customer_id: str = Field(
        ..., description="Unique identifier for the customer."
    )
    message: str = Field(
        ..., min_length=1, max_length=2000, description="Customer message text."
    )


class RefundRequest(BaseModel):
    """Request to process a refund for an order.

    Attributes:
        order_id: The order to refund.
        reason: Reason for the refund.
        amount: Optional partial refund amount in USD.
    """

    order_id: str = Field(..., description="The order ID to refund.")
    reason: str = Field(
        ...,
        description="Reason for the refund.",
        json_schema_extra={
            "enum": ["damaged", "wrong_item", "not_as_described", "changed_mind"],
        },
    )
    amount: Optional[float] = Field(
        None, gt=0, description="Partial refund amount in USD. Omit for full refund."
    )


# ---------------------------------------------------------------
# Response models
# ---------------------------------------------------------------


class ContextInfo(BaseModel):
    """Metadata about the assembled context window.

    Attributes:
        total_blocks: Number of context blocks assembled.
        total_tokens: Total tokens in the context window.
        max_tokens: Maximum token budget.
        budget_used_percent: Percentage of token budget consumed.
        rag_chunks_retrieved: Number of RAG knowledge chunks retrieved.
        tools_selected: Names of tools selected for the query.
        pipeline_tokens_saved: Tokens removed by the optimization pipeline.
        sufficiency_score: Confidence that the context is sufficient.
        quality_score: Positional quality score.
    """

    total_blocks: int = Field(..., description="Number of context blocks assembled.")
    total_tokens: int = Field(..., description="Total tokens in the context window.")
    max_tokens: int = Field(..., description="Maximum token budget.")
    budget_used_percent: float = Field(
        ..., description="Percentage of token budget consumed."
    )
    rag_chunks_retrieved: int = Field(
        ..., description="Number of RAG knowledge chunks retrieved."
    )
    tools_selected: List[str] = Field(
        ..., description="Names of tools selected for the query."
    )
    pipeline_tokens_saved: int = Field(
        ..., description="Tokens removed by the optimization pipeline."
    )
    sufficiency_score: float = Field(
        ..., description="Confidence that the context is sufficient."
    )
    quality_score: float = Field(..., description="Positional quality score.")


class ChatResponse(BaseModel):
    """Response from the support agent.

    Attributes:
        customer_id: The customer who sent the message.
        formatted_payload: The provider-formatted payload ready for LLM API.
        context_info: Metadata about the assembled context window.
    """

    customer_id: str = Field(
        ..., description="The customer who sent the message."
    )
    formatted_payload: Dict[str, Any] = Field(
        ..., description="Provider-formatted payload ready for the LLM API."
    )
    context_info: ContextInfo = Field(
        ..., description="Metadata about the assembled context window."
    )


class OrderResponse(BaseModel):
    """Response from an order lookup.

    Attributes:
        order_id: The looked-up order ID.
        status: Current order status.
        items: Items in the order.
        total: Order total in USD.
        delivered_at: Delivery date.
        return_eligible: Whether the order is eligible for return.
        return_deadline: Deadline for return requests.
    """

    order_id: str = Field(..., description="The looked-up order ID.")
    status: str = Field(..., description="Current order status.")
    items: List[Dict[str, Any]] = Field(..., description="Items in the order.")
    total: float = Field(..., description="Order total in USD.")
    delivered_at: str = Field(..., description="Delivery date.")
    return_eligible: bool = Field(
        ..., description="Whether the order is eligible for return."
    )
    return_deadline: str = Field(..., description="Deadline for return requests.")


class ConversationTurn(BaseModel):
    """A single turn in the conversation history.

    Attributes:
        role: The speaker role (user or assistant).
        content: The message content.
    """

    role: str = Field(..., description="Speaker role (user or assistant).")
    content: str = Field(..., description="Message content.")


class ConversationHistoryResponse(BaseModel):
    """The current conversation history for a customer.

    Attributes:
        customer_id: The customer whose history is returned.
        turns: Ordered list of conversation turns.
        turn_count: Total number of turns.
    """

    customer_id: str = Field(
        ..., description="The customer whose history is returned."
    )
    turns: List[ConversationTurn] = Field(
        ..., description="Ordered list of conversation turns."
    )
    turn_count: int = Field(..., description="Total number of turns.")


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status: Service status.
        knowledge_base_chunks: Number of articles in the knowledge base.
        registered_tools: Number of registered tools.
    """

    status: str = Field(..., description="Service status.")
    knowledge_base_chunks: int = Field(
        ..., description="Number of articles in the knowledge base."
    )
    registered_tools: int = Field(..., description="Number of registered tools.")
