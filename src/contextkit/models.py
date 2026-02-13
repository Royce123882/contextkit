"""Model registry for known LLM models.

Maps model names to their properties: context window size, tokenizer
encoding, and input/output pricing. Developers can register custom
models for fine-tuned or self-hosted variants.
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field

from contextkit.exceptions import UnknownModelError


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


# Built-in model registry
_REGISTRY: Dict[str, ModelSpec] = {
    "claude-opus-4-6": ModelSpec(
        max_context=200_000,
        effective_max_tokens=160_000,
        encoding="cl100k_base",
        input_cost_per_mtok=15.0,
        output_cost_per_mtok=75.0,
        attention_profile=ATTENTION_PROFILES["strong_long_context"],
    ),
    "claude-sonnet-4-5-20250929": ModelSpec(
        max_context=200_000,
        effective_max_tokens=160_000,
        encoding="cl100k_base",
        input_cost_per_mtok=3.0,
        output_cost_per_mtok=15.0,
        attention_profile=ATTENTION_PROFILES["strong_long_context"],
    ),
    "claude-haiku-4-5-20251001": ModelSpec(
        max_context=200_000,
        effective_max_tokens=160_000,
        encoding="cl100k_base",
        input_cost_per_mtok=0.80,
        output_cost_per_mtok=4.0,
        attention_profile=ATTENTION_PROFILES["strong_long_context"],
    ),
    "gpt-4o": ModelSpec(
        max_context=128_000,
        effective_max_tokens=96_000,
        encoding="o200k_base",
        input_cost_per_mtok=2.50,
        output_cost_per_mtok=10.0,
        attention_profile=ATTENTION_PROFILES["standard"],
    ),
    "gpt-4o-mini": ModelSpec(
        max_context=128_000,
        effective_max_tokens=96_000,
        encoding="o200k_base",
        input_cost_per_mtok=0.15,
        output_cost_per_mtok=0.60,
        attention_profile=ATTENTION_PROFILES["standard"],
    ),
    "gpt-4.1": ModelSpec(
        max_context=1_000_000,
        effective_max_tokens=800_000,
        encoding="o200k_base",
        input_cost_per_mtok=2.0,
        output_cost_per_mtok=8.0,
        attention_profile=ATTENTION_PROFILES["strong_long_context"],
    ),
}


def get_model(name: str) -> ModelSpec:
    """Look up a model by name.

    Args:
        name: The model identifier (e.g. "claude-sonnet-4-5-20250929").

    Returns:
        The ModelSpec for the requested model.

    Raises:
        UnknownModelError: If the model name is not registered.
    """
    spec = _REGISTRY.get(name)
    if spec is None:
        raise UnknownModelError(name)
    return spec


def register_model(name: str, spec: ModelSpec) -> None:
    """Register a custom model in the global registry.

    Args:
        name: The model identifier.
        spec: The model specification.
    """
    _REGISTRY[name] = spec


def list_models() -> List[str]:
    """Return a sorted list of all registered model names."""
    return sorted(_REGISTRY.keys())
