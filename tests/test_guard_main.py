"""The guard_main hook judges the branch of the checkout a commit targets (#91)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).parents[1] / ".claude" / "hooks" / "guard_main.py"


def git(cwd: Path, *args: str) -> None:
    """Run git quietly in cwd."""
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A main checkout on main plus sibling worktrees on a feature branch and on main."""
    root = tmp_path_factory.mktemp("guard")
    main = root / "main"
    main.mkdir()
    git(main, "init", "-q", "-b", "main")
    ident = ["-c", "user.name=t", "-c", "user.email=t@t"]
    git(main, *ident, "commit", "-q", "--allow-empty", "-m", "init")
    git(main, "worktree", "add", "-q", "-b", "feature/x", str(root / "wt"))
    git(main, "worktree", "add", "-q", "-f", str(root / "wt-main"), "main")
    return root


def run_hook(cwd: Path, command: str) -> int:
    """Feed the hook a Bash payload whose session cwd is `cwd`; return its exit code."""
    env = {k: v for k, v in os.environ.items() if k != "MULEGRAPH_ALLOW_MAIN_COMMIT"}
    payload = json.dumps({"cwd": str(cwd), "tool_input": {"command": command}})
    return subprocess.run(
        [sys.executable, str(HOOK)], input=payload, text=True, capture_output=True, cwd="/", env=env
    ).returncode


@pytest.mark.parametrize(
    "command",
    [
        "cd ../wt && git commit -m x",
        "git -C ../wt commit -m x",
        "cd .. && git -C wt commit -m x",
        "(cd ../wt && git commit -m x)",
    ],
)
def test_commit_in_feature_worktree_allowed(repo: Path, command: str) -> None:
    """A commit reached by cd or -C into a feature-branch worktree passes."""
    assert run_hook(repo / "main", command) == 0


@pytest.mark.parametrize(
    "cwd, command",
    [
        ("main", "git add -A && git commit -m x"),
        ("wt", "cd ../wt-main && git commit -m x"),
        ("wt", "git -C ../wt-main commit -m x"),
        ("wt", "cd ../main; git commit -m x"),
        ("wt", "git commit -m x && git -C ../wt-main commit -m y"),
        ("wt", "(cd ../wt-main && git commit -m x)"),
        ("wt", "cd ../wt-main && cd ../wt && cd - && git commit -m x"),
    ],
)
def test_commit_on_main_refused(repo: Path, cwd: str, command: str) -> None:
    """A commit on main is refused in the main checkout and in a worktree on main."""
    assert run_hook(repo / cwd, command) == 2


def test_inline_override_allows_commit_on_main(repo: Path) -> None:
    """The override the error message suggests works when written in the command."""
    assert run_hook(repo / "main", "MULEGRAPH_ALLOW_MAIN_COMMIT=1 git commit -m x") == 0


def test_non_commit_ignored(repo: Path) -> None:
    """Commands that only mention commit are not judged."""
    assert run_hook(repo / "main", "git log --grep=commit") == 0
