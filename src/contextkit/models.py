"""Model registry for known LLM models.

Maps model names to their properties: context window size, tokenizer
encoding, and input/output pricing. Developers can register custom
models for fine-tuned or self-hosted variants.
"""

from __future__ import annotations

from typing import Dict, List

from contextkit.exceptions import UnknownModelError
from contextkit.model_models import ATTENTION_PROFILES, AttentionProfile, ModelSpec


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
