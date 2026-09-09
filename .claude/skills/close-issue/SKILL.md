---
description: Close a mulegraph GitHub issue as delivered, citing the commit(s) that delivered it, and tick the matching line in project_status.md. Use when a tracked deliverable is done and verified.
disable-model-invocation: true
argument-hint: <issue number> [commit sha ...]
allowed-tools: Read, Edit, Grep, Bash, mcp__plugin_github_github__issue_read, mcp__plugin_github_github__add_issue_comment, mcp__plugin_github_github__issue_write
---

# Close issue `$ARGUMENTS` as delivered

An issue closed without a commit reference is a claim; one closed with it is
evidence. Every number in the dissertation must be regenerable from a tagged
commit (NFR-1, S5), so the closing comment names the commits and the status
document is updated in the same breath — the two must never disagree.

## Steps

1. **Resolve the repository** from `git remote get-url origin`. Never print
   `$GITHUB_PERSONAL_ACCESS_TOKEN`.

2. **Read the issue** with `issue_read`. If it is already closed, say so and
   stop. Note its milestone, labels and the requirement ids in its body.

3. **Verify every commit.** For each sha given (or, if none, ask for one — do
   not guess from `git log`):
   - `git log --oneline -1 <sha>` must succeed; refuse the close if a sha does
     not exist locally.
   - `git branch -r --contains <sha>` — if it is on no `origin/*` branch, warn
     that the link in the closing comment will not resolve until pushed, and
     ask whether to continue.

4. **Verify the deliverable is actually done.** Read the issue's "Done when"
   section and check each line against the tree (a file exists, a test passes,
   a config key is present). If any line is not met, report which and stop;
   an issue is closed on evidence, not on the commit message's say-so.

5. **Comment**, one paragraph:

   ```
   Delivered in <sha> (<subject>)[, <sha> (<subject>)]. <one sentence naming
   what verified it — the test file, the check that ran, the artefact produced>.
   ```

6. **Close** with `issue_write` (`method: update`, `state: closed`,
   `state_reason: completed`).

7. **Tick the status document.** In `docs/project_status.md`, find the
   checklist line that matches the issue — by requirement id first, then by
   title words — under *MVP definition of done*, *Week-1 gates*, or *What's
   next*, and change `- [ ]` to `- [x]` with the date. If no line matches, say
   so rather than inventing one. If this was the milestone's last open
   deliverable, say that too: the milestone table row and the changelog need
   updating, which is `/update-docs-and-commit`'s job.

Do **not** commit. Point to `/update-docs-and-commit`, which will also record
the change in `docs/changelog.md`.

## Report

- The issue number and title, the commits cited, and the status-doc line
  ticked (with its section) — or exactly which "Done when" line blocked the
  close.
