"""Every docstring is one line, except the allowlisted ones that predate the rule (CLAUDE.md)."""

import ast
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).parents[1]

# Multi-line docstrings that predate the test. Shorten one, then delete its entry.
ALLOWED = {
    "mulegraph/data/elliptic.py::<module>",
    "mulegraph/drift/monitor.py::score_batches",
    "mulegraph/features/aggregate.py::<module>",
    "mulegraph/features/builder.py::<module>",
    "mulegraph/features/builder.py::build_features",
    "mulegraph/features/gfp.py::<module>",
    "mulegraph/features/select.py::<module>",
    "mulegraph/models/sage.py::<module>",
    "mulegraph/models/sage_edge.py::<module>",
    "mulegraph/types.py::GraphDataset",
    "tests/conftest.py::<module>",
    "tests/test_features_aggregate.py::<module>",
    "tests/test_features_aggregate.py::test_max_is_exact_on_real_gfp_output",
    "tests/test_features_causality.py::<module>",
    "tests/test_features_causality.py::_subgraph",
    "tests/test_features_gfp_layout.py::<module>",
    "tests/test_features_gfp_layout.py::test_no_double_count_and_why_partial_fit_is_forbidden",
    "tests/test_features_gfp_layout.py::test_window_bounds_how_far_back_a_batch_can_see",
    "tests/test_features_select.py::<module>",
    "tests/test_loader_elliptic.py::<module>",
    "tests/test_loader_elliptic.py::test_elliptic_pp_extras_are_dropped",
    "tests/test_metrics.py::<module>",
    "tests/test_models_sage.py::<module>",
    "tests/test_models_sage.py::test_planted_signal_is_learnable",
    "tests/test_models_xgb.py::<module>",
    "tests/test_pipeline.py::<module>",
    "tests/test_pipeline.py::test_rerunning_a_config_does_not_pool_the_old_seeds",
    "tests/test_report_tables.py::<module>",
    "tests/test_report_tables.py::allow_file_store",
    "tests/test_score.py::<module>",
    "tests/test_score.py::test_labels_of_the_scored_batch_change_nothing",
    "tests/test_splits.py::<module>",
    "tests/test_splits.py::test_cache_is_keyed_on_the_definition_not_the_partition",
    "scripts/readme_figures.py::<module>",
}


def _docstrings() -> Iterator[tuple[str, str]]:
    """Yield (path::qualname, docstring) for every module, class and function checked."""

    def walk(node: ast.AST, prefix: str) -> Iterator[tuple[str, ast.AST]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                yield prefix + child.name, child
                yield from walk(child, prefix + child.name + ".")
            else:
                yield from walk(child, prefix)

    for d in ("mulegraph", "tests", "scripts"):
        for path in sorted((ROOT / d).rglob("*.py")):
            tree = ast.parse(path.read_text())
            for qualname, node in [("<module>", tree), *walk(tree, "")]:
                if doc := ast.get_docstring(node):
                    yield f"{path.relative_to(ROOT)}::{qualname}", doc


def test_docstrings_are_one_line() -> None:
    """A new multi-line docstring fails, and so does a stale ALLOWED entry."""
    multi = {key for key, doc in _docstrings() if "\n" in doc}
    assert not (new := sorted(multi - ALLOWED)), f"one line, as CLAUDE.md asks: {new}"
    assert not (stale := sorted(ALLOWED - multi)), f"stale ALLOWED entries, delete them: {stale}"
