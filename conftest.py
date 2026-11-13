"""`--rows-for`: the planted-bug rows a diff needs (NFR-2); root, so pytest always loads it."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

#: Paths whose change can matter to no planted row but those whose guards they hold (NFR-2).
_ROWS = "test_integrity_mutants.py"
_NARROWING = ("docs/", ".claude/agents/")
#: The two integrity tests every narrowed run keeps.
_CONTROLS = {
    "test_every_guard_passes_in_the_unplanted_copy",
    "test_every_threshold_use_is_planted_or_excused",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--rows-for",
        nargs="*",
        metavar="PATH",
        help="keep only the integrity rows whose guards these changed paths hold, plus controls",
    )


def _narrows(path: str) -> bool:
    """True for a docs, agent or top-level test file, else every row runs (NFR-2)."""
    p = Path(path)
    test_file = p.parent == Path("tests") and p.name.startswith("test_") and p.suffix == ".py"
    return path.startswith(_NARROWING) or (test_file and p.name != _ROWS)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """``--rows-for``: deselect each integrity row whose guards none of the paths holds (NFR-2)."""
    paths = config.getoption("rows_for")
    # An empty list is a failed or mistyped git diff more often than an empty one: run every row.
    if not paths:
        return
    paths = {os.path.normpath(p) for p in paths}
    if not all(map(_narrows, paths)):
        return
    keep, drop = [], []
    for item in items:
        if item.path.name != _ROWS:
            keep.append(item)
            continue
        callspec = getattr(item, "callspec", None)
        params = callspec.params if callspec else {}
        mutant = params.get("mutant") or next(
            (m for m in item.module.MUTANTS if m.name == params.get("name")), None
        )
        held = mutant and any(t.split("::")[0] in paths for t in mutant.tests)
        (keep if item.originalname in _CONTROLS or held else drop).append(item)
    config.hook.pytest_deselected(items=drop)
    items[:] = keep
