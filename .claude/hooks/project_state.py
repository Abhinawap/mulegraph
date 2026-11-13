#!/usr/bin/env python3
"""SessionStart — put current project state in front of Claude.

Division of labour: CLAUDE.md holds the rules (stable, hand-written),
this holds the state (changes weekly, read from the repo). That is what
lets CLAUDE.md stay short enough to load every session.

Reads docs/project_status.md rather than duplicating it, so there is one
source of truth for the milestone.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
STATUS = PROJECT_DIR / "docs" / "project_status.md"


def git(*args: str, cwd: Path = PROJECT_DIR, timeout: float = 5) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def field(text: str, label: str) -> str | None:
    """Pull `**Label:** value` out of the status file."""
    m = re.search(rf"^\*\*{re.escape(label)}:\*\*\s*(.+?)\s*$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


def build_context() -> str:
    lines: list[str] = []

    if STATUS.is_file():
        text = STATUS.read_text(encoding="utf-8", errors="replace")
        for label in ("Current milestone", "Overall state", "Last updated"):
            value = field(text, label)
            if value:
                lines.append(f"{label}: {value}")

        open_items = len(re.findall(r"^\s*- \[ \]", text, re.MULTILINE))
        done_items = len(re.findall(r"^\s*- \[[xX]\]", text, re.MULTILINE))
        if open_items or done_items:
            lines.append(f"Checklist in project_status.md: {done_items} done, {open_items} open")
    else:
        lines.append("docs/project_status.md not found — project state unknown.")

    # --abbrev-ref fails before the first commit; --show-current does not.
    branch = git("branch", "--show-current") or git("rev-parse", "--abbrev-ref", "HEAD")
    commit = git("rev-parse", "--short", "HEAD") or "no commits yet"
    lines.append(f"Git: {branch} @ {commit}")

    git("fetch", "--quiet", "origin", "main", timeout=3)  # offline: the count below is stale
    behind = git("rev-list", "--count", "main..origin/main")
    if behind and behind != "0":
        lines.append(f"main is {behind} commit(s) behind origin/main: pull before reporting status")

    # Merged = reached through a merge; a fresh branch's tip sits on main's own line.
    fmt = "--format=%(refname:short) %(objectname)"
    own_line = set((git("rev-list", "--first-parent", "origin/main") or "").split())
    merged = {
        name
        for name, sha in (
            x.split() for x in (git("branch", "--merged", "origin/main", fmt) or "").splitlines()
        )
        if sha not in own_line
    }
    blocks = (git("worktree", "list", "--porcelain") or "").split("\n\n")
    main_path = blocks[0].splitlines()[0].removeprefix("worktree ") if blocks[0] else ""
    if main_path and git("status", "--porcelain", cwd=Path(main_path)):
        lines.append(
            f"Main checkout {main_path} has uncommitted changes: another session may be "
            "using it; start your branch in a worktree."
        )
    for block in blocks[1:]:
        path = block.splitlines()[0].removeprefix("worktree ")
        heads = [x for x in block.splitlines() if x.startswith("branch refs/heads/")]
        branch = heads[0].removeprefix("branch refs/heads/") if heads else ""
        dotgit = Path(path) / ".git"  # a linked worktree's .git is a "gitdir: <path>" file
        gitdir = (
            Path(dotgit.read_text().removeprefix("gitdir: ").strip())
            if dotgit.is_file()
            else dotgit
        )
        busy = [n for n in ("MERGE_HEAD", "rebase-merge", "rebase-apply") if (gitdir / n).exists()]
        if busy or branch in merged:
            why = f"{'/'.join(busy)} in progress" if busy else "branch merged"
            lines.append(f"Stale worktree: {path} [{branch or 'detached'}] ({why})")

    dirty = git("status", "--porcelain")
    if dirty:
        lines.append(f"Uncommitted changes: {len(dirty.splitlines())} path(s)")

    lines.append(
        "Reminder: when work completes, update docs/project_status.md and "
        "docs/changelog.md alongside the code."
    )
    return "Project state (from docs/project_status.md):\n" + "\n".join(
        f"  {line}" for line in lines
    )


def main() -> int:
    try:
        context = build_context()
    except Exception as exc:  # never break session startup over a status file
        context = f"Project state unavailable: {exc}"

    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": context,
            }
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
