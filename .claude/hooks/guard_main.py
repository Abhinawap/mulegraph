#!/usr/bin/env python3
"""PreToolUse(Bash) — refuse commits made directly on the default branch.

CLAUDE.md says "NEVER commit directly to main". This makes that real:
CLAUDE.md asks, a hook enforces.

Exit 0 = allow, exit 2 = block with stderr as the reason.

Escape hatches, both deliberate:
  * the repository has no commits yet (an initial commit has nowhere else to go)
  * MULEGRAPH_ALLOW_MAIN_COMMIT=1 is set in the environment
"""

import json
import os
import re
import shlex
import subprocess
import sys

DEFAULT_BRANCHES = {"main", "master"}

# git global options that consume the following token, so the subcommand
# scanner must skip two positions rather than one.
GIT_GLOBALS_WITH_VALUE = {
    "-C",
    "-c",
    "--git-dir",
    "--work-tree",
    "--namespace",
    "--exec-path",
    "--super-prefix",
}


def git(*args: str) -> str | None:
    """Run a git command, returning stripped stdout or None if it failed."""
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def is_commit(command: str) -> bool:
    """True if any segment of the command invokes `git commit`.

    Checks the git *subcommand* rather than searching for the word anywhere,
    so `git log --grep=commit` and `git show HEAD --format=%H commit` do not
    trip the guard. Handles `git -C path commit` and `git add -A && git commit`.
    """
    for segment in re.split(r"\|\||&&|[;&|\n]", command):
        segment = segment.strip()
        if not segment:
            continue
        try:
            tokens = shlex.split(segment)
        except ValueError:  # unbalanced quotes — fall back to a crude split
            tokens = segment.split()

        start = next(
            (i for i, t in enumerate(tokens) if t == "git" or t.endswith("/git")),
            None,
        )
        if start is None:
            continue

        i = start + 1
        while i < len(tokens):
            token = tokens[i]
            if token in GIT_GLOBALS_WITH_VALUE:
                i += 2
            elif token.startswith("-"):
                i += 1
            else:
                if token == "commit":
                    return True
                break  # a different subcommand; check the next segment
    return False


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # Malformed input is not this hook's problem; do not block.

    command = (payload.get("tool_input") or {}).get("command") or ""
    if not is_commit(command):
        return 0

    if os.environ.get("MULEGRAPH_ALLOW_MAIN_COMMIT") == "1":
        return 0

    # No commits yet: the initial commit must land on the default branch.
    if git("rev-parse", "--verify", "HEAD") is None:
        return 0

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch not in DEFAULT_BRANCHES:
        return 0

    print(
        f"Refusing to commit on '{branch}'. CLAUDE.md: never commit directly to "
        f"the default branch.\n"
        f"Branch first:  git checkout -b feature/<description>\n"
        f"Override once: MULEGRAPH_ALLOW_MAIN_COMMIT=1 git commit ...",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
