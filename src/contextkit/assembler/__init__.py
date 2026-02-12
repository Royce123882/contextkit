"""Context assembler for composing blocks into a window.

The assembler takes a list of ContextBlocks and composes them into a
ContextWindow with ordering and priority rules. It produces an
AssemblyReport that powers explain() for understanding why blocks
were included or excluded.

ContextAssembler is lazily imported to avoid a circular dependency:
  context_window -> observe.explain -> assembler.report
  -> assembler.__init__ (eager) -> context_assembler -> context_window
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextkit.assembler.context_assembler import ContextAssembler
    from contextkit.assembler.report import AssemblyReport, BlockDecision

__all__ = [
    "AssemblyReport",
    "BlockDecision",
    "ContextAssembler",
]

_LAZY_IMPORTS = {
    "AssemblyReport": "contextkit.assembler.report",
    "BlockDecision": "contextkit.assembler.report",
    "ContextAssembler": "contextkit.assembler.context_assembler",
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
