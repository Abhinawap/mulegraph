---
description: Update project_status.md and changelog.md to match the working tree, then commit. Use when committing work that changes project state.
disable-model-invocation: true
argument-hint: [optional commit message hint]
---

# Update docs, then commit

Documentation drift is a named failure mode on this project: `project_status.md`
and `changelog.md` are only useful while they are true, and they go stale exactly
when the project is busiest. This command keeps them in step with the code.

Extra context from the user, if any: `$ARGUMENTS`

## 1. Read the change

- `git status --porcelain` and `git diff HEAD` (or `git diff --cached` if
  something is staged) to see what actually changed.
- Read `docs/project_status.md` and `docs/changelog.md` before editing either.

## 2. Update `docs/changelog.md`

- Add entries under a dated section for today. Create the section if the newest
  one is not today; otherwise append to it.
- Group under `### Added`, `### Changed`, `### Fixed`, or `### Decided`.
- Describe the change and its effect, not the file list. Reference the
  requirement id (`PR-F2`, `D2`, …) where the change implements or affects one.
- Anything that alters behaviour, requirements, or reproducibility gets an
  entry. Pure refactors and formatting do not.

## 3. Update `docs/project_status.md`

Only where the change actually moves something:

- Tick checklist items that are now genuinely done — verify against the tree,
  do not take the diff's word for it.
- Move completed work into **What's been accomplished** with today's date.
- Update **What's next** if the head of the list has been consumed.
- Tick or add checklist items.
- Refresh **Last updated** and **Overall state**.

Keep the file about *now and next*. History belongs in the changelog.

## 4. Commit

- If on `main` or `master`, stop and ask the user to name a branch — a hook
  will refuse the commit anyway (`.claude/hooks/guard_main.py`).
- Stage the code changes together with the doc updates. They belong in one
  commit; a doc update that lands separately defeats the point.
- Write a message that says what changed and why, and names the requirement id
  where one applies.
- End the message with the attribution lines given in this session's
  system-reminder, if one is present. Do not invent or reuse a session URL.

## 5. Report

State what you changed in each doc and show the commit subject. If you
deliberately left a doc untouched, say why.
