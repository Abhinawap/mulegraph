---
description: Look up a mulegraph requirement, design decision, or success criterion by id from the project spec. Use when a PR-*, D*, NFR-*, or S* id needs checking.
disable-model-invocation: true
argument-hint: <id> (e.g. PR-E4, D2, NFR-3, S3, H1)
allowed-tools: Read, Grep
---

# Look up requirement `$ARGUMENTS`

Find and report what the spec actually says about `$ARGUMENTS`. Do not answer
from memory — the spec is the source of truth and it is version 0.6.

## Steps

1. Grep `docs/project_spec.md` for the id. Ids appear in several forms, so
   search for the bare token rather than a fixed pattern:
   - `PR-*` requirements live in **Appendix A** (the requirement register)
   - `D1`–`D4` design decisions live in **§2.0**
   - `NFR-*` live in **§1.4**
   - `S1`–`S5` success criteria live in **§1.5**
   - `H1` is the pre-registered drift hypothesis (§1.2, §2.5)
2. Read enough surrounding context to get the full statement, not a fragment.
3. Grep the rest of the spec for other mentions of the same id — most
   requirements are qualified somewhere else (a dataset-specific exception, a
   fallback ladder, a "reported, not gated" caveat). Those qualifications are
   usually the part that matters.

## Report

- **The requirement**, quoted from the spec.
- **Where it lives** — section and line reference.
- **Qualifications** found elsewhere in the spec, each with its location.
- **What it constrains in code** — one or two sentences, only if the mapping is
  unambiguous. If the id has no clear code surface yet, say so rather than
  inventing one.

If the id does not exist in the spec, say that plainly and list the nearest
ids that do. Do not guess at what it might have meant.
