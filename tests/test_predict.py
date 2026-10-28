"""Tests for predict.py's artifact naming and for undefined-name regressions.

`predict.main()` needs torch/transformers, which are deliberately absent from
the local analysis venv -- so the local suite can never execute it. That gap is
exactly how `best_spans(..., max_answer_tokens)` (an unbound name) shipped in
78b2a19 and only surfaced when the cluster ran the job: both predict jobs died
with NameError after a full BERT pass. test_no_undefined_globals below closes
that gap statically.
"""

from __future__ import annotations

import builtins
import symtable
import unittest
from pathlib import Path

from qa_baseline.predict import sidecar_paths

SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "qa_baseline"
ALLOWED_IMPLICIT = {"__name__", "__file__", "__doc__", "__package__", "__spec__", "__builtins__"}


def _module_level_names(table: symtable.SymbolTable) -> set[str]:
    return {symbol.get_name() for symbol in table.get_symbols()}


def _undefined_globals(table: symtable.SymbolTable, module_names: set[str], found: list[str]) -> None:
    for symbol in table.get_symbols():
        name = symbol.get_name()
        if not symbol.is_global():
            continue  # locals, params, deferred imports, comprehension targets
        if name in module_names or name in ALLOWED_IMPLICIT or hasattr(builtins, name):
            continue
        found.append(f"{table.get_name()}: {name}")
    for child in table.get_children():
        _undefined_globals(child, module_names, found)


class TestSidecarPaths(unittest.TestCase):
    def test_dev_convention(self) -> None:
        scores, confidence = sidecar_paths(Path("runs/baseline/predictions.json"))
        self.assertEqual(scores, Path("runs/baseline/scores.npz"))
        self.assertEqual(confidence, Path("runs/baseline/confidence.json"))

    def test_val_convention(self) -> None:
        scores, confidence = sidecar_paths(Path("runs/baseline/val-predictions.json"))
        self.assertEqual(scores, Path("runs/baseline/val-scores.npz"))
        self.assertEqual(confidence, Path("runs/baseline/val-confidence.json"))

    def test_val_and_dev_never_collide(self) -> None:
        dev = set(sidecar_paths(Path("runs/baseline/predictions.json")))
        val = set(sidecar_paths(Path("runs/baseline/val-predictions.json")))
        self.assertEqual(dev & val, set(), "val and dev caches must not share a path")


class TestNoUndefinedGlobals(unittest.TestCase):
    def test_no_undefined_globals(self) -> None:
        problems: list[str] = []
        for path in sorted(SRC_DIR.glob("*.py")):
            table = symtable.symtable(path.read_text(encoding="utf-8"), str(path), "exec")
            _undefined_globals(table, _module_level_names(table), problems)
        self.assertEqual(problems, [], f"unbound global names (NameError at runtime): {problems}")


if __name__ == "__main__":
    unittest.main()
