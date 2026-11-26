"""`--rows-for`: the planted-bug rows a diff needs (NFR-2); root, so pytest always loads it."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# One OpenMP/BLAS thread per xdist worker, set before numpy, XGBoost or torch load (#109).
if os.environ.get("PYTEST_XDIST_WORKER"):
    os.environ.setdefault("OMP_NUM_THREADS", "1")

#: Paths whose change can matter to no planted row but those whose guards they hold (NFR-2).
_ROWS = "test_integrity_mutants.py"
_NARROWING = ("docs/", ".claude/agents/")
#: The integrity test every narrowed run keeps.
_CONTROLS = {"test_every_guard_passes_in_the_unplanted_copy"}


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
        if item.path.name != _ROWS or not item.get_closest_marker("integrity"):
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
