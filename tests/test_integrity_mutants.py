"""Each integrity guard fails its named tests when its bug is planted (PR-E4, PR-E7)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

pytestmark = pytest.mark.integrity

PIPELINE = "tests/test_pipeline.py::"
FIT_THRESHOLD = PIPELINE + "test_the_threshold_is_the_validation_optimum_applied_to_every_test_row"
LOGGED_F1 = PIPELINE + "test_every_logged_f1_is_scored_at_its_runs_validation_optimum[tiny_config]"
ROLLING_FIT = PIPELINE + "test_each_rolling_step_is_decided_at_its_own_validation_threshold"
ROLLING_LOGGED = PIPELINE + "test_every_logged_rolling_step_threshold_is_its_own_validation_optimum"
ROLLING_TABLE = PIPELINE + "test_rolling_regime_writes_one_child_run_per_seed"
#: The threshold searched on the test rows in scope, which no guarded site may use.
ON_TEST = "choose_threshold(test.y, test.proba)[0]"


class Mutant(NamedTuple):
    name: str
    file: str
    site: str
    broken: str
    tests: tuple[str, ...]


MUTANTS = [
    Mutant(
        "threshold on test inside fit()",
        "mulegraph/pipeline.py",
        "threshold, val_f1 = choose_threshold(val.y, val.proba)",
        "threshold, val_f1 = (choose_threshold(data.y[step.test],"
        " model.predict_proba(data, feats, step.test))[0], choose_threshold(val.y, val.proba)[1])",
        (FIT_THRESHOLD,),
    ),
    Mutant(
        "per-row test threshold searched on test inside fit()",
        "mulegraph/pipeline.py",
        "np.full(test.idx.size, threshold, dtype=np.float64)",
        f"np.full(test.idx.size, {ON_TEST}, dtype=np.float64)",
        (FIT_THRESHOLD,),
    ),
    Mutant(
        "rolling step threshold searched on test inside fit()",
        "mulegraph/pipeline.py",
        '"ts_threshold": threshold,',
        f'"ts_threshold": {ON_TEST},',
        (ROLLING_FIT,),
    ),
    Mutant(
        "logged threshold searched on test",
        "mulegraph/pipeline.py",
        '"threshold": fitted.threshold,\n                "val_f1": fitted.val_f1,',
        f'"threshold": {ON_TEST},\n                "val_f1": fitted.val_f1,',
        (LOGGED_F1,),
    ),
    Mutant(
        "test F1 at a test-searched threshold",
        "mulegraph/pipeline.py",
        "compute_metrics(test.y, test.proba, fitted.test_threshold, metrics,",
        f"compute_metrics(test.y, test.proba, {ON_TEST}, metrics,",
        (LOGGED_F1,),
    ),
    Mutant(
        "per-timestep curve at a test-searched threshold",
        "mulegraph/pipeline.py",
        "curve = per_timestep(test, fitted.test_threshold)",
        f"curve = per_timestep(test, {ON_TEST})",
        (LOGGED_F1,),
    ),
    Mutant(
        "curves table at a test-searched threshold",
        "mulegraph/pipeline.py",
        "curves.append(\n                        curve.assign(",
        "curves.append(\n                        per_timestep(fitted.test, choose_threshold("
        "fitted.test.y, fitted.test.proba)[0]).assign(",
        (LOGGED_F1,),
    ),
    Mutant(
        "logged rolling ts_threshold altered",
        "mulegraph/pipeline.py",
        "mlflow.log_metrics(step_metrics, step=t)",
        "mlflow.log_metrics({**step_metrics, 'ts_threshold': choose_threshold("
        "test.y[test.time == t], test.proba[test.time == t])[0]}, step=t)",
        (ROLLING_LOGGED,),
    ),
    Mutant(
        "rank metrics pooled on a rolling run",
        "mulegraph/pipeline.py",
        'metrics = [m for m in cfg.eval.metrics if m == "f1"] if fitted.steps else',
        "metrics = cfg.eval.metrics if fitted.steps else",
        (ROLLING_TABLE,),
    ),
]


def _copy(repo_root: Path, tmp_path: Path) -> Path:
    """The package, its tests and configs, so pytest in the copy imports the copy."""
    root = tmp_path / "repo"
    for name in ("mulegraph", "tests", "configs"):
        shutil.copytree(repo_root / name, root / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(repo_root / "pyproject.toml", root)
    return root


def _plant(root: Path, mutant: Mutant) -> None:
    path = root / mutant.file
    text = path.read_text()
    if text.count(mutant.site) != 1:
        pytest.fail(f"site moved: re-check this guard ({mutant.name}, {mutant.file})")
    path.write_text(text.replace(mutant.site, mutant.broken))


def _run_pytest(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    # cwd and PYTHONPATH both put the copy ahead of the editable install's .pth entry.
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
        cwd=root,
        # A wide terminal keeps each failure's reason on its summary line.
        env={**os.environ, "PYTHONPATH": str(root), "COLUMNS": "1000"},
        capture_output=True,
        text=True,
    )


def test_every_guard_passes_in_the_unplanted_copy(repo_root: Path, tmp_path: Path) -> None:
    """Control: the copy imports itself and its guards pass, so a row's failure is its bug."""
    root = _copy(repo_root, tmp_path)
    (root / "tests" / "test_which.py").write_text(
        "import mulegraph\n\n\n"
        "def test_which():\n"
        f"    assert mulegraph.__file__ == {str(root / 'mulegraph' / '__init__.py')!r}\n"
    )
    guards = sorted({t for m in MUTANTS for t in m.tests})
    out = _run_pytest(root, "tests/test_which.py", *guards)
    assert out.returncode == 0, out.stdout + out.stderr


def test_a_missing_site_fails_with_site_moved(repo_root: Path, tmp_path: Path) -> None:
    root = _copy(repo_root, tmp_path)
    gone = MUTANTS[0]._replace(site="no such fragment")
    with pytest.raises(pytest.fail.Exception, match="site moved"):
        _plant(root, gone)


@pytest.mark.parametrize("mutant", MUTANTS, ids=[m.name for m in MUTANTS])
def test_planting_the_bug_fails_its_guard(mutant: Mutant, repo_root: Path, tmp_path: Path) -> None:
    root = _copy(repo_root, tmp_path)
    _plant(root, mutant)
    out = _run_pytest(root, *mutant.tests)
    log = out.stdout + out.stderr
    # A NameError or crash would also exit nonzero without proving the guard catches anything.
    for t in mutant.tests:
        pattern = rf"FAILED {re.escape(t)} - (assert|AssertionError)"
        assert re.search(pattern, log), log
