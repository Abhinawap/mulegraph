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


def _fail(message: str) -> None:
    """Print a one-sentence reason without a traceback, then exit."""
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


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

    typer.secho(f"results table: {table}", fg=typer.colors.GREEN)


@app.command()
def smoke(
    keep: Annotated[bool, typer.Option("--keep", help="Keep the temporary run directory.")] = False,
) -> None:
    """End-to-end check on a synthetic graph: a benchmark, then one scored batch. Used in CI."""
    from mulegraph.pipeline import run_smoke

    table, alerts, health = run_smoke(keep=keep)
    typer.secho(f"smoke OK: {table}\nalerts: {alerts}\nhealth: {health}", fg=typer.colors.GREEN)


@app.command()
def drift(config: ConfigOption) -> None:
    """Fit, then flag post-training batches with label-free detectors and report lead time."""
    from mulegraph.config import DriftRunConfig, load_config
    from mulegraph.pipeline import run_drift
    from mulegraph.splits.builder import RegimeNotSupportedError

    try:
        cfg = load_config(config, DriftRunConfig)
    except FileNotFoundError as exc:
        _fail(str(exc))
    except ValueError as exc:
        _fail(f"Invalid config {config}: {exc}")

    try:
        table = run_drift(cfg, config)
    except (RegimeNotSupportedError, FileNotFoundError) as exc:
        _fail(str(exc))

    typer.secho(f"lead-time table: {table}", fg=typer.colors.GREEN)


#: ``score`` exit status when a health detector flags the batch; 1 is an error, 2 a usage error.
EXIT_DRIFT = 3


@app.command()
def score(config: ConfigOption) -> None:
    """Score one batch: a ranked alert queue and a label-free health check. Exits 3 on a flag."""
    from mulegraph.config import ScoreConfig, load_config
    from mulegraph.pipeline import run_score

    try:
        cfg = load_config(config, ScoreConfig)
    except FileNotFoundError as exc:
        _fail(str(exc))
    except ValueError as exc:
        _fail(f"Invalid config {config}: {exc}")

    try:
        alerts, health, flagged = run_score(cfg)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))

    typer.secho(f"alerts: {alerts}\nhealth: {health}", fg=typer.colors.GREEN)
    if flagged:
        typer.secho(
            f"drift flagged on batch {cfg.batch} by {', '.join(flagged)}; the scores may not "
            "be trustworthy, see the health file",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(EXIT_DRIFT)


if __name__ == "__main__":  # pragma: no cover
    app()
