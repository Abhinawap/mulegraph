#!/usr/bin/env python3
"""PreToolUse(Bash) — refuse commits made directly on the default branch.

CLAUDE.md says "NEVER commit directly to main". This makes that real:
CLAUDE.md asks, a hook enforces.

Exit 0 = allow, exit 2 = block with stderr as the reason.

The branch is read in the directory the commit runs in: the session's cwd from
the payload, moved by any `cd <dir>` segment and `git -C <dir>` (worktrees, #91).

Escape hatches, both deliberate:
  * the repository has no commits yet (an initial commit has nowhere else to go)
  * MULEGRAPH_ALLOW_MAIN_COMMIT=1 is set in the environment or before `git commit`
"""

import json
import os
import re
import shlex
import subprocess
import sys

DEFAULT_BRANCHES = {"main", "master"}
OVERRIDE = "MULEGRAPH_ALLOW_MAIN_COMMIT"

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


def git(cwd: str, *args: str) -> str | None:
    """Run a git command in cwd, returning stripped stdout or None if it failed."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def commit_targets(command: str, cwd: str) -> list[tuple[str | None, bool]]:
    """Directory of every `git commit` in the command (None if unknown) and its inline override.

    Checks the git *subcommand* rather than searching for the word anywhere,
    so `git log --grep=commit` and `git show HEAD --format=%H commit` do not
    trip the guard. Handles `git add -A && git commit`. An inline
    `MULEGRAPH_ALLOW_MAIN_COMMIT=1 git commit` counts as the override, since
    the hook cannot see the command's own environment.
    """
    targets: list[tuple[str | None, bool]] = []
    here: str | None = cwd
    for segment in re.split(r"\|\||&&|[;&|\n]", command):
        segment = segment.strip().lstrip("(").rstrip(")").strip()
        if not segment:
            continue
        try:
            tokens = shlex.split(segment)
        except ValueError:  # unbalanced quotes — fall back to a crude split
            tokens = segment.split()

        if tokens[:1] == ["cd"]:
            # ponytail: subshell scope, pushd and --git-dir not tracked; follow them if they
            # start appearing. `cd -`, a bare `cd` or a flag loses the directory (refused).
            arg = tokens[1] if len(tokens) == 2 else "-"
            ok = here is not None and not arg.startswith("-")
            here = os.path.join(here, os.path.expanduser(arg)) if ok else None
            continue

        start = next(
            (i for i, t in enumerate(tokens) if t == "git" or t.endswith("/git")),
            None,
        )
        if start is None:
            continue

        where = here
        i = start + 1
        while i < len(tokens):
            token = tokens[i]
            if token == "-C" and i + 1 < len(tokens) and where is not None:
                where = os.path.join(where, os.path.expanduser(tokens[i + 1]))
            if token in GIT_GLOBALS_WITH_VALUE:
                i += 2
            elif token.startswith("-"):
                i += 1
            else:
                if token == "commit":
                    targets.append((where, f"{OVERRIDE}=1" in tokens[:start]))
                break  # a different subcommand; check the next segment
    return targets


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # Malformed input is not this hook's problem; do not block.

    command = (payload.get("tool_input") or {}).get("command") or ""
    if os.environ.get(OVERRIDE) == "1":
        return 0
    hint = f"Override once: {OVERRIDE}=1 git commit ..."

    for where, inline_override in commit_targets(command, payload.get("cwd") or os.getcwd()):
        if inline_override:
            continue
        if where is None:
            print(
                "Refusing to commit: a `cd -`, bare `cd` or `cd` flag hides which "
                f"checkout the commit runs in. Use `git -C <dir> commit`.\n{hint}",
                file=sys.stderr,
            )
            return 2
        # No commits yet: the initial commit must land on the default branch.
        if git(where, "rev-parse", "--verify", "HEAD") is None:
            continue
        branch = git(where, "rev-parse", "--abbrev-ref", "HEAD")
        if branch in DEFAULT_BRANCHES:
            print(
                f"Refusing to commit on '{branch}' in {where}. CLAUDE.md: never commit "
                f"directly to the default branch.\n"
                f"Branch first:  git checkout -b feature/<description>\n{hint}",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
