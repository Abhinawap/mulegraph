"""Paths, hashing, provenance, seeding, timing, logging. Must not import torch at module level."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("mulegraph")


@dataclass(frozen=True)
class Paths:
    """Filesystem roots, from the environment (see ``.env.example``)."""

    data_dir: Path
    report_dir: Path

    @classmethod
    def from_env(cls) -> Paths:
        return cls(
            data_dir=Path(os.environ.get("MULEGRAPH_DATA_DIR", "./data")).expanduser(),
            report_dir=Path(os.environ.get("MULEGRAPH_REPORT_DIR", "./report")).expanduser(),
        )

    def raw_dir(self, dataset: str, version: str) -> Path:
        return self.data_dir / "raw" / dataset / version

    def cache_dir(self, dataset: str, version: str) -> Path:
        return self.data_dir / "cache" / dataset / version

    def tables_dir(self) -> Path:
        return self.report_dir / "tables"

    def figures_dir(self) -> Path:
        return self.report_dir / "figures"


def canonical_json(obj: Any) -> str:
    """Stable JSON for hashing: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def hash_dict(obj: Any, length: int = 16) -> str:
    """sha256 prefix over a JSON-serialisable object."""
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()[:length]


def hash_arrays(arrays: dict[str, np.ndarray], extra: Any = None, length: int = 16) -> str:
    """sha256 prefix over named arrays plus an optional JSON-serialisable tail."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        h.update(name.encode())
        h.update(np.ascontiguousarray(arrays[name]).tobytes())
    if extra is not None:
        h.update(canonical_json(extra).encode())
    return h.hexdigest()[:length]


def hash_files(paths: list[Path], length: int = 16) -> str:
    """sha256 prefix over file contents, for raw-data provenance."""
    h = hashlib.sha256()
    for path in sorted(paths):
        with open(path, "rb") as fh:
            for chunk in iter(lambda fh=fh: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:length]


def git_commit(short: bool = True) -> str:
    """Current commit, or ``"unknown"`` outside a repository (PR-O1)."""
    args = ["git", "rev-parse", *(["--short"] if short else []), "HEAD"]
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def git_dirty() -> bool:
    """True when the working tree has uncommitted changes (a run may not be regenerable)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and bool(out.stdout.strip())


def seed_all(seed: int) -> None:
    """Seed Python, NumPy and (if importable) torch + CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> str:
    """Map ``auto|cuda|cpu`` to an actual device string, falling back with a warning."""
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        available = torch.cuda.is_available()
    except ImportError:
        available = False
    if requested == "cuda" and not available:
        log.warning("device 'cuda' requested but no CUDA device is available; using cpu")
        return "cpu"
    return "cuda" if available else "cpu"


class Timer:
    """Wall-clock timer; fit durations feed the W budget arithmetic (D2)."""

    def __init__(self) -> None:
        self.seconds: float = 0.0
        self._start: float | None = None

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        assert self._start is not None
        self.seconds = time.perf_counter() - self._start


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once, honouring ``MULEGRAPH_LOG_LEVEL``."""
    level = level or os.environ.get("MULEGRAPH_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def write_json(path: Path, obj: Any) -> Path:
    """Write pretty JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    return path
