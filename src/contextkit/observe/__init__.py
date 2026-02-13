"""Observability and explainability for context windows.

Quality and sufficiency modules are lazily imported to avoid
circular imports (they depend on ``contextkit.core`` which
depends on ``contextkit.observe.provenance``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from contextkit.observe.provenance import Mutation, Origin

if TYPE_CHECKING:
    from contextkit.observe.drift import DriftDetector
    from contextkit.observe.linter import ContextLinter
    from contextkit.observe.linter_models import LintWarning
    from contextkit.observe.quality import QualityScorer
    from contextkit.observe.quality_models import DriftReport, PositionScore, QualityReport
    from contextkit.observe.sufficiency import SufficiencyChecker
    from contextkit.observe.sufficiency_models import SufficiencyResult
    from contextkit.observe.telemetry import OTelExporter

__all__ = [
    "ContextLinter",
    "DriftDetector",
    "DriftReport",
    "LintWarning",
    "Mutation",
    "OTelExporter",
    "Origin",
    "PositionScore",
    "QualityReport",
    "QualityScorer",
    "SufficiencyChecker",
    "SufficiencyResult",
]

_LAZY_IMPORTS = {
    "ContextLinter": "contextkit.observe.linter",
    "DriftDetector": "contextkit.observe.drift",
    "DriftReport": "contextkit.observe.quality_models",
    "LintWarning": "contextkit.observe.linter_models",
    "OTelExporter": "contextkit.observe.telemetry",
    "QualityScorer": "contextkit.observe.quality",
    "PositionScore": "contextkit.observe.quality_models",
    "QualityReport": "contextkit.observe.quality_models",
    "SufficiencyChecker": "contextkit.observe.sufficiency",
    "SufficiencyResult": "contextkit.observe.sufficiency_models",
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
