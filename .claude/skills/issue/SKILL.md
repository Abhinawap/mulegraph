---
description: Create a well-formed GitHub issue for a mulegraph requirement id or a free-text task, with the right milestone and labels, refusing duplicates and out-of-scope work. Use when new work needs tracking.
disable-model-invocation: true
argument-hint: <PR-id | D* | NFR-* | S* | free text title>
allowed-tools: Read, Grep, Bash, mcp__plugin_github_github__search_issues, mcp__plugin_github_github__list_issues, mcp__plugin_github_github__issue_write
---

# Create an issue for `$ARGUMENTS`

Issues mirror `docs/project_spec.md` §1.3: one per milestone deliverable, each
listing its requirement ids. This command keeps that structure intact when work
is added later — it looks the id up in the spec, puts the issue on the right
milestone, and refuses anything the spec has parked in "Later" or "Not in scope"
(NFR-5).

## Steps

1. **Resolve the repository.** `git remote get-url origin` → `owner/repo`.
   Never hard-code it and never print `$GITHUB_PERSONAL_ACCESS_TOKEN`.

2. **Classify the argument.**
   - If it matches `PR-[A-Z]\d+`, `D\d`, `NFR-\d`, `S\d` or `H1`: it is a
     requirement id. Grep `docs/project_spec.md` for the bare token and quote the
     full statement plus any qualification found elsewhere in the spec (a
     dataset-specific exception, a fallback ladder, a "reported, not gated"
     caveat). Do not answer from memory.
   - Otherwise it is a free-text task or bug. Use it as the title.

3. **Map to a milestone.** Requirement ids belong to the milestone whose §1.3
   row delivers them:

   | Milestone | Ids |
   |---|---|
   | MVP | PR-D1, PR-D4, PR-F1, PR-F2, PR-F3, PR-M1, PR-M2, PR-M3, PR-M5, PR-M7, PR-E1 (a, b), PR-E2, PR-E4, PR-E6, PR-O1, D1, D3 |
   | v1a | PR-M6, PR-E3, PR-E5, D2, NFR-2 |
   | v1b | PR-D2, PR-M4, PR-E1 (c) |
   | v2 | PR-R1–PR-R6, PR-P1–PR-P4, PR-O2, PR-O3, PR-D3, D4, S3, S4, H1 |

   For free text, read §1.3 and pick the milestone whose scope it falls under;
   if it clearly belongs to **Later** or **Not in scope** (actor graph, TGN,
   Ethereum, explanations, cost-sensitive thresholds, SynthAML, GAT, dashboards,
   LLMs, SMOTE, real bank data…), **stop and refuse**, quoting NFR-5: nothing
   there starts before the v2 hard stop. Do not create the issue.

4. **Look up the milestone number** at run time — it is never stored anywhere:

   ```
   curl -s -H "Authorization: Bearer $GITHUB_PERSONAL_ACCESS_TOKEN" \
     "https://api.github.com/repos/<owner>/<repo>/milestones?state=all"
   ```

   Match on `title`. If the milestone is missing, say so and stop.

5. **Labels.** One area label from the id prefix — `D`→`data`, `F`→`features`,
   `E1`→`splits`, other `E`→`eval`, `M6`→`search`, other `M`→`models`,
   `R`→`drift`, `P`→`policies`, `O`→`reporting` — or the best fit for free text
   (`infra`, `writing`). Add `deliverable` when it is a §1.3 bold item,
   `supervisor` when it needs a supervisor decision, `gate` for week-1 timing.

6. **Refuse duplicates.** Check the live issue list, not only search:
   `list_issues` for the repo, all states, fields `number, title, body, state`
   (page through if more than 100), and grep title and body for the id and for
   the distinctive title words. Then `search_issues` as a second net. GitHub's
   search index lags new issues by minutes, so search alone would let a
   duplicate through right after a batch was created — which is exactly when
   this command tends to be run. If an existing issue already covers it — open
   or closed — report its number and URL and stop. A deliverable has one issue.

7. **Create** with `issue_write` (`method: create`, `milestone`, `labels`) using
   this body:

   ```
   **Milestone:** <name> (due <date>) · **Spec:** docs/project_spec.md §1.3 <row>, Appendix A
   **Status in docs:** docs/project_status.md → <checklist section, if one exists>

   ## Deliverable
   > <the spec sentence(s), quoted verbatim>

   ## Requirements covered
   - [ ] <id> — <one-line paraphrase>

   ## Done when
   <the matching definition-of-done lines, or a concrete observable outcome>

   ## Constraints that apply
   <the D1–D4 / CLAUDE.md integrity rules this item can violate, one line each>
   ```

## Report

- The issue URL, milestone, and labels — or the existing issue it duplicated,
  or the NFR-5 refusal with the §1.3 row that parks the item.
- If the spec qualifies the requirement somewhere other than its register
  entry, quote that qualification; it is usually the part that matters.
