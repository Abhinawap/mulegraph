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


def git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_DIR), *args],
            capture_output=True,
            text=True,
            timeout=5,
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
            lines.append(
                f"Checklist in project_status.md: {done_items} done, {open_items} open"
            )
    else:
        lines.append("docs/project_status.md not found — project state unknown.")

    # --abbrev-ref fails before the first commit; --show-current does not.
    branch = git("branch", "--show-current") or git("rev-parse", "--abbrev-ref", "HEAD")
    commit = git("rev-parse", "--short", "HEAD") or "no commits yet"
    lines.append(f"Git: {branch} @ {commit}")

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
