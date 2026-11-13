"""``--rows-for`` collects the planted-bug rows a diff needs, and every row when unsure (NFR-2)."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_integrity_mutants import MUTANTS

MODULE = "tests/test_integrity_mutants.py::"
CONTROLS = {
    MODULE + "test_every_guard_passes_in_the_unplanted_copy",
    MODULE + "test_every_threshold_use_is_planted_or_excused",
}


def _collected(repo_root: Path, *args: str, where: tuple[str, ...] = (MODULE[:-2],)) -> set[str]:
    """Node ids collected; ``where=()`` gives no test path, as the auditor's command does."""
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]
        + ["-m", "integrity", *where, *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return {line for line in out.stdout.splitlines() if line.startswith(MODULE)}


@pytest.mark.parametrize("path", ["tests/test_score.py", "./tests/test_score.py"])
def test_a_guard_file_selects_its_rows_and_the_controls(repo_root: Path, path: str) -> None:
    score = {m.name for m in MUTANTS if any(t.startswith("tests/test_score.py::") for t in m.tests)}
    assert score
    rows = {MODULE + f"test_planting_the_bug_fails_its_guard[{n}]" for n in score}
    assert _collected(repo_root, "--rows-for", path) == rows | CONTROLS


def test_a_docs_only_diff_selects_only_the_controls(repo_root: Path) -> None:
    paths = ("docs/design.md", ".claude/agents/integrity-auditor.md")
    assert _collected(repo_root, "--rows-for", *paths, where=()) == CONTROLS


def test_no_guard_file_imports_another_test_file(repo_root: Path) -> None:
    """A narrowing test file would otherwise change a guard without selecting its row."""
    for guard in {t.split("::")[0] for m in MUTANTS for t in m.tests}:
        tree = ast.parse((repo_root / guard).read_text())
        names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        froms = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        names += [n.module or "" for n in froms]
        assert not [n for n in names if n.split(".")[0] == "tests"], guard
        assert not [n for n in froms if n.level], guard  # a relative import is a sibling file


@pytest.fixture(scope="module")
def every(repo_root: Path) -> set[str]:
    return _collected(repo_root)


@pytest.mark.parametrize(
    "path",
    [
        "mulegraph/pipeline.py",
        "pyproject.toml",
        "uv.lock",
        "configs/elliptic_mvp.yaml",
        "tests/conftest.py",
        "tests/test_integrity_mutants.py",
        "README.md",
        "docs/../mulegraph/pipeline.py",
    ],
)
def test_any_other_path_selects_every_row(repo_root: Path, every: set[str], path: str) -> None:
    assert len(every) > len(MUTANTS)
    assert _collected(repo_root, "--rows-for", "docs/design.md", path) == every


def test_an_empty_list_selects_every_row(repo_root: Path, every: set[str]) -> None:
    """A mistyped base makes git diff print nothing; that must not narrow the run."""
    assert _collected(repo_root, "--rows-for") == every
