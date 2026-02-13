"""Data models for the model registry."""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class AttentionProfile(BaseModel):
    """Model-specific attention curve parameters.

    Different models have different sensitivity to positional effects.
    These parameters control the U-curve used by QualityScorer.

    Research basis: "Lost in the Middle" (Liu et al., 2023) shows
    model-dependent attention patterns -- newer models handle long
    context better than older ones.

    Attributes:
        curve_depth: Depth of the U-curve trough (0.0-1.0). Lower means
            the model loses less information in the middle. Default 0.6.
        label: Human-readable description of this profile.
    """

    curve_depth: float = Field(
        default=0.6,
        description="Depth of U-curve trough (0.0-1.0). Lower = better middle retention.",
    )
    label: str = Field(
        default="standard",
        description="Human-readable label for this attention profile.",
    )


# Pre-built attention profiles for known model families
ATTENTION_PROFILES: Dict[str, AttentionProfile] = {
    "strong_long_context": AttentionProfile(
        curve_depth=0.3,
        label="Strong long-context (Claude 4.x, GPT-4.1)",
    ),
    "standard": AttentionProfile(
        curve_depth=0.6,
        label="Standard (GPT-4o, older models)",
    ),
    "weak_long_context": AttentionProfile(
        curve_depth=0.8,
        label="Weak long-context (smaller/older models)",
    ),
}


class ModelSpec(BaseModel):
    """Specification for a language model.

    Attributes:
        max_context: Maximum context window size in tokens.
        effective_max_tokens: Practical effective window size. Models often
            degrade before hitting advertised limits. Based on research
            from arXiv:2509.21361 ("Maximum Effective Context Window").
            If None, defaults to max_context.
        encoding: The tiktoken encoding name for this model.
        input_cost_per_mtok: Cost per million input tokens in USD.
        output_cost_per_mtok: Cost per million output tokens in USD.
        attention_profile: Model-specific attention curve parameters
            for quality scoring. If None, uses the "standard" profile.
    """

    max_context: int = Field(description="Maximum context window size in tokens.")
    effective_max_tokens: int | None = Field(
        default=None,
        description="Practical effective window size. If None, defaults to max_context.",
    )
    encoding: str = Field(description="The tiktoken encoding name for this model.")
    input_cost_per_mtok: float = Field(description="Cost per million input tokens in USD.")
    output_cost_per_mtok: float = Field(description="Cost per million output tokens in USD.")
    attention_profile: AttentionProfile | None = Field(
        default=None,
        description="Model-specific attention curve parameters.",
    )
