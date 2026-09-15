"""Typer entry point; the only experiment option is ``--config``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from mulegraph import __version__
from mulegraph.util import setup_logging

app = typer.Typer(
    name="mulegraph",
    help="Benchmark and drift-monitoring toolkit for laundering-network detection.",
    no_args_is_help=True,
    add_completion=False,
)

log = logging.getLogger("mulegraph")

ConfigOption = Annotated[
    Path, typer.Option("--config", "-c", help="Experiment YAML under configs/.")
]


def _fail(message: str, code: int = 1) -> None:
    """Print a one-sentence reason without a traceback, then exit."""
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", help="Print the version and exit.")] = False,
) -> None:
    setup_logging()
    if version:
        typer.echo(f"mulegraph {__version__}")
        raise typer.Exit()


@app.command()
def run(config: ConfigOption) -> None:
    """Run a full benchmark: load, features, split, fit, evaluate, log to MLflow."""
    from mulegraph.config import load_config
    from mulegraph.pipeline import run_benchmark
    from mulegraph.splits.builder import RegimeNotSupportedError

    try:
        cfg = load_config(config)
    except FileNotFoundError as exc:
        _fail(str(exc))
    except ValueError as exc:
        _fail(f"Invalid config {config}: {exc}")

    try:
        table = run_benchmark(cfg, config)
    except (RegimeNotSupportedError, FileNotFoundError) as exc:
        _fail(str(exc))
    except NotImplementedError as exc:
        _fail(str(exc), code=2)

    typer.secho(f"results table: {table}", fg=typer.colors.GREEN)


@app.command()
def smoke(
    keep: Annotated[bool, typer.Option("--keep", help="Keep the temporary run directory.")] = False,
) -> None:
    """End-to-end check on a synthetic graph, under two minutes on CPU. Used in CI."""
    from mulegraph.pipeline import run_smoke

    table = run_smoke(keep=keep)
    typer.secho(f"smoke OK: {table}", fg=typer.colors.GREEN)


@app.command()
def drift(config: ConfigOption) -> None:
    """Run label-free drift detectors on a fitted model (v2)."""
    _fail(
        "drift is v2 (5 Jan - 13 Feb 2027): the detectors, batching and lead-time "
        "reporting are not built yet.",
        code=2,
    )


@app.command()
def simulate(config: ConfigOption) -> None:
    """Replay the test period under retraining policies (v2)."""
    _fail(
        "simulate is v2 (5 Jan - 13 Feb 2027): the retraining-policy simulator is not built yet.",
        code=2,
    )


@app.command()
def report(
    milestone: Annotated[str, typer.Option("--milestone", help="Milestone tag, e.g. v1a.")],
) -> None:
    """Regenerate dissertation tables and figures from MLflow (v1a)."""
    _fail(
        "report is v1a: paired gaps and per-timestep figures are not built yet, and "
        "`mulegraph run` already writes report/tables/<experiment>_results.csv.",
        code=2,
    )


if __name__ == "__main__":  # pragma: no cover
    app()
