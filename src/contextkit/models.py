"""Model registry for known LLM models.

Maps model names to their properties: context window size, tokenizer
encoding, and input/output pricing. Developers can register custom
models for fine-tuned or self-hosted variants.
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    """Specification for a language model.

    Attributes:
        max_context: Maximum context window size in tokens.
        encoding: The tiktoken encoding name for this model.
        input_cost_per_mtok: Cost per million input tokens in USD.
        output_cost_per_mtok: Cost per million output tokens in USD.
    """

    max_context: int = Field(description="Maximum context window size in tokens.")
    encoding: str = Field(description="The tiktoken encoding name for this model.")
    input_cost_per_mtok: float = Field(description="Cost per million input tokens in USD.")
    output_cost_per_mtok: float = Field(description="Cost per million output tokens in USD.")


class UnknownModelError(Exception):
    """Raised when a model name is not found in the registry."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        super().__init__(
            f"Unknown model: '{model_name}'. Use register_model() to add custom models."
        )


# Built-in model registry
_REGISTRY: Dict[str, ModelSpec] = {
    "claude-opus-4-6": ModelSpec(
        max_context=200_000,
        encoding="cl100k_base",
        input_cost_per_mtok=15.0,
        output_cost_per_mtok=75.0,
    ),
    "claude-sonnet-4-5-20250929": ModelSpec(
        max_context=200_000,
        encoding="cl100k_base",
        input_cost_per_mtok=3.0,
        output_cost_per_mtok=15.0,
    ),
    "claude-haiku-4-5-20251001": ModelSpec(
        max_context=200_000,
        encoding="cl100k_base",
        input_cost_per_mtok=0.80,
        output_cost_per_mtok=4.0,
    ),
    "gpt-4o": ModelSpec(
        max_context=128_000,
        encoding="o200k_base",
        input_cost_per_mtok=2.50,
        output_cost_per_mtok=10.0,
    ),
    "gpt-4o-mini": ModelSpec(
        max_context=128_000,
        encoding="o200k_base",
        input_cost_per_mtok=0.15,
        output_cost_per_mtok=0.60,
    ),
    "gpt-4.1": ModelSpec(
        max_context=1_000_000,
        encoding="o200k_base",
        input_cost_per_mtok=2.0,
        output_cost_per_mtok=8.0,
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
