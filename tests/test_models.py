"""Tests for the model registry."""

from __future__ import annotations

import pytest

from contextkit.exceptions import UnknownModelError
from contextkit.model_spec import AttentionProfile, ModelSpec
from contextkit.model_registry import get_model, list_models, register_model


class TestModelSpec:
    """Tests for the ModelSpec model."""

    def test_create_model_spec(self) -> None:
        spec = ModelSpec(
            max_context=32_000,
            encoding="cl100k_base",
            input_cost_per_mtok=0.50,
            output_cost_per_mtok=1.50,
        )
        assert spec.max_context == 32_000
        assert spec.encoding == "cl100k_base"
        assert spec.input_cost_per_mtok == 0.50
        assert spec.output_cost_per_mtok == 1.50


class TestGetModel:
    """Tests for looking up models."""

    def test_lookup_claude_sonnet(self) -> None:
        spec = get_model("claude-sonnet-4-5-20250929")
        assert spec.max_context == 200_000
        assert spec.encoding == "cl100k_base"

    def test_lookup_claude_opus(self) -> None:
        spec = get_model("claude-opus-4-6")
        assert spec.max_context == 200_000

    def test_lookup_claude_haiku(self) -> None:
        spec = get_model("claude-haiku-4-5-20251001")
        assert spec.max_context == 200_000

    def test_lookup_gpt4o(self) -> None:
        spec = get_model("gpt-4o")
        assert spec.max_context == 128_000
        assert spec.encoding == "o200k_base"

    def test_lookup_gpt4o_mini(self) -> None:
        spec = get_model("gpt-4o-mini")
        assert spec.max_context == 128_000

    def test_lookup_gpt41(self) -> None:
        spec = get_model("gpt-4.1")
        assert spec.max_context == 1_000_000

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(UnknownModelError) as exc_info:
            get_model("nonexistent-model")
        assert "nonexistent-model" in str(exc_info.value)
        assert exc_info.value.model_name == "nonexistent-model"


class TestRegisterModel:
    """Tests for custom model registration."""

    def test_register_and_lookup(self) -> None:
        spec = ModelSpec(
            max_context=32_000,
            encoding="cl100k_base",
            input_cost_per_mtok=0.50,
            output_cost_per_mtok=1.50,
        )
        register_model("my-custom-model", spec)
        result = get_model("my-custom-model")
        assert result.max_context == 32_000
        assert result.input_cost_per_mtok == 0.50

    def test_register_overwrites_existing(self) -> None:
        spec1 = ModelSpec(
            max_context=10_000,
            encoding="cl100k_base",
            input_cost_per_mtok=1.0,
            output_cost_per_mtok=2.0,
        )
        spec2 = ModelSpec(
            max_context=20_000,
            encoding="cl100k_base",
            input_cost_per_mtok=1.5,
            output_cost_per_mtok=3.0,
        )
        register_model("overwrite-test", spec1)
        register_model("overwrite-test", spec2)
        result = get_model("overwrite-test")
        assert result.max_context == 20_000


class TestListModels:
    """Tests for listing registered models."""

    def test_list_models_returns_builtin(self) -> None:
        models = list_models()
        assert "claude-sonnet-4-5-20250929" in models
        assert "gpt-4o" in models

    def test_list_models_is_sorted(self) -> None:
        models = list_models()
        assert models == sorted(models)

    def test_list_models_includes_registered(self) -> None:
        register_model(
            "list-test-model",
            ModelSpec(
                max_context=1000,
                encoding="cl100k_base",
                input_cost_per_mtok=0,
                output_cost_per_mtok=0,
            ),
        )
        assert "list-test-model" in list_models()

    def test_list_models_has_at_least_six(self) -> None:
        models = list_models()
        assert len(models) >= 6


# ---------------------------------------------------------------------------
# Effective window profiles and attention profiles
# ---------------------------------------------------------------------------


class TestEffectiveWindowProfiles:
    """Tests for effective_max_tokens and AttentionProfile on ModelSpec."""

    def test_model_has_effective_max_tokens(self) -> None:
        spec = get_model("claude-opus-4-6")
        assert spec.effective_max_tokens is not None
        assert spec.effective_max_tokens < spec.max_context

    def test_model_has_attention_profile(self) -> None:
        spec = get_model("claude-opus-4-6")
        assert spec.attention_profile is not None
        assert spec.attention_profile.curve_depth == 0.3

    def test_gpt4o_has_standard_profile(self) -> None:
        spec = get_model("gpt-4o")
        assert spec.attention_profile is not None
        assert spec.attention_profile.curve_depth == 0.6

    def test_custom_model_without_effective_tokens(self) -> None:
        register_model(
            "test-custom-no-effective",
            ModelSpec(
                max_context=50_000,
                encoding="cl100k_base",
                input_cost_per_mtok=1.0,
                output_cost_per_mtok=5.0,
            ),
        )
        spec = get_model("test-custom-no-effective")
        assert spec.effective_max_tokens is None
        assert spec.attention_profile is None

    def test_attention_profile_standalone(self) -> None:
        profile = AttentionProfile(curve_depth=0.4, label="custom")
        assert profile.curve_depth == 0.4
        assert profile.label == "custom"
