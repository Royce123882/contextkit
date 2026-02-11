"""Tests for prompt templates and example store (Phase 2)."""

from __future__ import annotations

import pytest

from contextkit.prompts.examples import Example, ExampleStore
from contextkit.prompts.manager import (
    PromptManager,
    PromptVersion,
    _interpolate,
)


class TestPromptVersion:
    """Tests for the PromptVersion model."""

    def test_create_with_defaults(self) -> None:
        pv = PromptVersion(name="test", version="1", template="Hello {{name}}")
        assert pv.name == "test"
        assert pv.version == "1"
        assert pv.description == ""
        assert pv.created_at is not None

    def test_create_with_description(self) -> None:
        pv = PromptVersion(
            name="test",
            version="2",
            template="Hi",
            description="A greeting",
        )
        assert pv.description == "A greeting"


class TestInterpolate:
    """Tests for the _interpolate helper."""

    def test_basic_substitution(self) -> None:
        result = _interpolate("Hello {{name}}", {"name": "World"})
        assert result == "Hello World"

    def test_multiple_variables(self) -> None:
        result = _interpolate(
            "{{greeting}} {{name}}!",
            {"greeting": "Hi", "name": "Alice"},
        )
        assert result == "Hi Alice!"

    def test_missing_variable_preserved(self) -> None:
        result = _interpolate(
            "Hello {{name}} and {{other}}",
            {"name": "Alice"},
        )
        assert result == "Hello Alice and {{other}}"

    def test_no_variables(self) -> None:
        result = _interpolate("No placeholders here", {})
        assert result == "No placeholders here"

    def test_empty_template(self) -> None:
        result = _interpolate("", {"name": "test"})
        assert result == ""

    def test_non_string_value(self) -> None:
        result = _interpolate("Count: {{n}}", {"n": 42})
        assert result == "Count: 42"


class TestPromptManager:
    """Tests for the PromptManager."""

    def test_register_auto_version(self) -> None:
        pm = PromptManager()
        pv1 = pm.register("greet", "Hello {{name}}")
        pv2 = pm.register("greet", "Hi {{name}}")
        assert pv1.version == "1"
        assert pv2.version == "2"

    def test_register_explicit_version(self) -> None:
        pm = PromptManager()
        pv = pm.register("greet", "Hello", version="v1.0")
        assert pv.version == "v1.0"

    def test_get_template_latest(self) -> None:
        pm = PromptManager()
        pm.register("greet", "Hello v1")
        pm.register("greet", "Hello v2")
        pv = pm.get_template("greet")
        assert pv.template == "Hello v2"

    def test_get_template_specific_version(self) -> None:
        pm = PromptManager()
        pm.register("greet", "Hello v1")
        pm.register("greet", "Hello v2")
        pv = pm.get_template("greet", version="1")
        assert pv.template == "Hello v1"

    def test_get_template_not_found(self) -> None:
        pm = PromptManager()
        with pytest.raises(KeyError, match="No template registered"):
            pm.get_template("nonexistent")

    def test_get_template_version_not_found(self) -> None:
        pm = PromptManager()
        pm.register("greet", "Hello")
        with pytest.raises(KeyError, match="Version"):
            pm.get_template("greet", version="999")

    def test_render_basic(self) -> None:
        pm = PromptManager()
        pm.register("greet", "Hello {{user}}")
        block = pm.render("greet", user="World")
        assert block.content == "Hello World"
        assert block.type.value == "system_prompt"

    def test_render_with_origin(self) -> None:
        pm = PromptManager()
        pm.register("greet", "Hello {{user}}")
        block = pm.render("greet", user="World")
        assert block.origin is not None
        assert block.origin.source == "prompt"
        assert block.origin.details["template"] == "greet"
        assert "user" in block.origin.details["variables"]

    def test_render_priority(self) -> None:
        pm = PromptManager()
        pm.register("greet", "Hello")
        block = pm.render("greet", priority=80)
        assert block.priority == 80

    def test_compose_two_templates(self) -> None:
        pm = PromptManager()
        pm.register("base", "You are {{role}}.")
        pm.register("rules", "Rules: {{rules}}")
        block = pm.compose(
            "base",
            "rules",
            role="analyst",
            rules="Be concise",
        )
        assert "You are analyst." in block.content
        assert "Rules: Be concise" in block.content
        assert block.origin is not None
        assert "base+rules" in block.origin.details["template"]

    def test_compose_custom_separator(self) -> None:
        pm = PromptManager()
        pm.register("a", "Part A")
        pm.register("b", "Part B")
        block = pm.compose("a", "b", separator=" --- ")
        assert block.content == "Part A --- Part B"

    def test_diff_same_content(self) -> None:
        pm = PromptManager()
        pm.register("greet", "Hello", version="1")
        pm.register("greet", "Hello", version="2")
        diff = pm.diff("greet", "1", "2")
        assert "No changes" in diff

    def test_diff_changed_content(self) -> None:
        pm = PromptManager()
        pm.register("greet", "Hello World", version="1")
        pm.register("greet", "Hi World", version="2")
        diff = pm.diff("greet", "1", "2")
        assert "---" in diff
        assert "+++" in diff

    def test_list_templates(self) -> None:
        pm = PromptManager()
        pm.register("b_template", "B")
        pm.register("a_template", "A")
        templates = pm.list_templates()
        assert templates == ["a_template", "b_template"]

    def test_list_versions(self) -> None:
        pm = PromptManager()
        pm.register("greet", "v1", version="1.0")
        pm.register("greet", "v2", version="2.0")
        versions = pm.list_versions("greet")
        assert versions == ["1.0", "2.0"]

    def test_list_versions_not_found(self) -> None:
        pm = PromptManager()
        with pytest.raises(KeyError):
            pm.list_versions("nonexistent")


class TestExample:
    """Tests for the Example model."""

    def test_create_example(self) -> None:
        ex = Example(
            example_id="e1",
            input_text="What is 2+2?",
            output_text="4",
        )
        assert ex.example_id == "e1"
        assert ex.tags == []
        assert ex.metadata == {}

    def test_formatted_property(self) -> None:
        ex = Example(
            example_id="e1",
            input_text="What is 2+2?",
            output_text="4",
        )
        formatted = ex.formatted
        assert "Input: What is 2+2?" in formatted
        assert "Output: 4" in formatted

    def test_create_with_tags(self) -> None:
        ex = Example(
            example_id="e1",
            input_text="x",
            output_text="y",
            tags=["math"],
        )
        assert ex.tags == ["math"]


class TestExampleStore:
    """Tests for the ExampleStore."""

    def test_add_and_get(self) -> None:
        store = ExampleStore()
        ex = store.add("e1", "What is 2+2?", "4")
        assert ex.example_id == "e1"
        retrieved = store.get("e1")
        assert retrieved.input_text == "What is 2+2?"

    def test_get_not_found(self) -> None:
        store = ExampleStore()
        with pytest.raises(KeyError, match="No example"):
            store.get("nonexistent")

    def test_add_with_tags(self) -> None:
        store = ExampleStore()
        store.add("e1", "x", "y", tags=["math"])
        ex = store.get("e1")
        assert ex.tags == ["math"]

    def test_select_by_similarity(self) -> None:
        store = ExampleStore()
        store.add("e1", "What is Python?", "A programming language")
        store.add("e2", "What is Java?", "A programming language")
        store.add("e3", "How to cook pasta?", "Boil water first")
        results = store.select("Tell me about Python programming")
        assert len(results) >= 1
        # The Python example should rank higher
        assert results[0][0].example_id == "e1"

    def test_select_with_tags(self) -> None:
        store = ExampleStore()
        store.add("e1", "What is Python?", "A language", tags=["code"])
        store.add("e2", "How to cook?", "Boil water", tags=["food"])
        results = store.select("Python", tags=["code"])
        assert len(results) == 1
        assert results[0][0].example_id == "e1"

    def test_select_top_k(self) -> None:
        store = ExampleStore()
        for i in range(10):
            store.add(f"e{i}", f"Example {i} about Python", f"Out {i}")
        results = store.select("Python", top_k=3)
        assert len(results) == 3

    def test_fit_to_budget(self) -> None:
        store = ExampleStore()
        store.add("e1", "Short input", "Short output")
        store.add("e2", "Another input", "Another output")
        blocks = store.fit_to_budget("Short", max_tokens=1000)
        assert len(blocks) >= 1
        assert blocks[0].type.value == "examples"
        assert blocks[0].origin is not None
        assert blocks[0].origin.source == "example"

    def test_fit_to_budget_respects_limit(self) -> None:
        store = ExampleStore()
        for i in range(20):
            store.add(
                f"e{i}",
                f"This is example number {i} " * 10,
                f"Output for example {i} " * 10,
            )
        blocks = store.fit_to_budget("example", max_tokens=50)
        total_tokens = sum(b.token_count for b in blocks)
        assert total_tokens <= 50

    def test_list_examples(self) -> None:
        store = ExampleStore()
        store.add("b_example", "x", "y")
        store.add("a_example", "x", "y")
        assert store.list_examples() == ["a_example", "b_example"]

    def test_example_count(self) -> None:
        store = ExampleStore()
        assert store.example_count == 0
        store.add("e1", "x", "y")
        assert store.example_count == 1
