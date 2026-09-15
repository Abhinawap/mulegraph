---
description: Report where GitHub issues and docs/project_status.md disagree, which milestones are at risk, and which gate or supervisor issues are blocking. Read-only; proposes edits. Use before a supervisor meeting or a status update.
disable-model-invocation: true
argument-hint: (no arguments)
allowed-tools: Read, Grep, Bash, mcp__plugin_github_github__list_issues
---

# Reconcile GitHub issues with the status document

`docs/project_status.md` and the GitHub issue list describe the same work from
two directions, and they drift in the same way the docs drift from the code —
silently, when the project is busiest. This command reports every
disagreement. It changes nothing unless told to.

## Steps

1. **Resolve the repository** from `git remote get-url origin`. Never print
   `$GITHUB_PERSONAL_ACCESS_TOKEN`.

2. **Fetch GitHub state.**
   - `list_issues` for the repo, all states, fields `number, title, state,
     labels` (page through if more than 100).
   - Milestones with due dates, at run time:
     ```
     curl -s -H "Authorization: Bearer $GITHUB_PERSONAL_ACCESS_TOKEN" \
       "https://api.github.com/repos/<owner>/<repo>/milestones?state=all"
     ```
   - Each issue's milestone comes from the REST issue list
     (`/issues?state=all&per_page=100`), since `list_issues` omits it.

3. **Read the status document.** From `docs/project_status.md` collect every
   checklist line (`- [ ]` / `- [x]`) with its section, the milestone table
   rows with their status glyphs, and the *Open blockers*.

4. **Match** issues to lines by requirement id (`PR-*`, `D*`) first, then by
   distinctive title words. Be explicit about lines that matched nothing and
   issues that matched no line.

5. **Report, in this order:**
   - **Disagreements.** Issues closed on GitHub whose status line is unticked;
     ticked lines whose issue is still open; milestone-table rows marked done
     whose milestone still has open issues.
   - **Milestones.** For each: open / closed counts, days to `due_on` (today's
     date from `date +%F`), and whether it is past due with open issues.
   - **Blockers.** Every open issue labelled `gate` or `supervisor` — these
     block downstream work by the spec's own dependency order.
   - **Orphans.** Issues with no milestone (other than `scope`), and status
     lines with no issue.

6. **Propose** the exact edits to `docs/project_status.md` that would resolve
   the disagreements — which line, which tick, which date — and say which
   issues should be closed via `/close-issue` instead. Apply nothing unless
   the user says so.

## Report

Plain sections matching step 5. If everything agrees, say so in one line and
still give the milestone table, since "days remaining" is the number a
supervisor meeting needs.
