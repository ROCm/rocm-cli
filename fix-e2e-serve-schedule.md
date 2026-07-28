# WIP: E2E serve schedule + paths-filter + mock-lane overhead

**Stage:** 4-design
**Pipeline:** standard
**Ticket:** none yet — file when work starts.
**Branch:** not yet created (will re-branch off fresh main; child of [[fix-speed-up-e2e]])
**Pre-PR-check:** none
**Last Updated:** 2026-07-28

**Token Usage:** in=0 out=0 cache_create=0 cache_read=0 calls=0

---

## Problem

The remaining, lower-leverage chunks of the E2E speed-up effort (the container
[[fix-speed-up-e2e]] shipped the big wins as PRs #126/#128/#136). These reduce how
OFTEN the heavy real-GPU serve matrix runs, skip it for non-serve PRs, and trim
mock-lane fixed overhead — without losing coverage.

## Solution

Three independent levers:
- Gate the heavy real-GPU serve matrix to `merge_group` only, keeping ONE minimal
  real serve on `pull_request` as a pre-merge canary.
- Add a narrow `serve` paths-filter so Rust-but-not-serve PRs skip the GPU matrix.
- Reduce mock-lane per-scenario fixed overhead.

## Implementation Steps

### Todo 📋
- 📋 Task #8 — Gate the heavy real-GPU serve matrix to `merge_group` only (not per
  push), BUT keep ONE minimal real serve on `pull_request` as a pre-merge canary.
  Prereqs: fix the Strix-Windows flake first (merge-time flakes bounce good PRs);
  verify no moved job is a required check (would stall the queue). Note: `@nightly`
  + `E2E_INCLUDE_NIGHTLY=1` tiering already exists — this verifies/tunes it, not
  build from scratch. **Check first:** PR #142 (`a1a8079`, "ci: stabilize GPU E2E
  and merge queue", merged) may already address this.
- 📋 Task #9 — Add a narrow `serve` paths-filter (engines/**, apps/rocm serve code,
  crates/rocm-core, **/*.feature, e2e-cucumber + broad-dep safety nets) so
  Rust-but-not-serve PRs (dash-only, unrelated crates) skip the GPU matrix. Today
  the coarse `heavy` filter trips the whole matrix on ANY `.rs`. Err toward
  inclusion.
- 📋 Task #10 — Reduce mock lane per-scenario overhead (fixed overhead ~4.8s/
  scenario, multiply across 12). Distinct from the mock/real split
  ([[test-e2e-mock-real-split]]), which moved scenarios *off* GPU, not the mock-lane
  fixed cost.

## Next Steps

- Verify whether PR #142 already did Task #8's merge_group gating before starting.
- Re-branch off fresh main when work begins; file a ticket.

## Blockers / Open Questions

- **Strix-Windows flake** is a prereq for Task #8 (merge-time flakes bounce good
  PRs).
- **Tiering already exists** (feeds Task #8): `@nightly` + `E2E_INCLUDE_NIGHTLY=1`
  already separates heavy scenarios from per-PR runs. Task #8 verifies/tunes
  merge_group gating, not build from scratch.

## Notes

Split out of the [[fix-speed-up-e2e]] umbrella (2026-07-28) so each task/branch/
ticket is tracked in its own WIP. Sibling children: [[test-e2e-mock-real-split]]
(EAI-7484, #136), [[test-e2e-smallest-serve-model]] (#128), [[test-e2e-diagnose]]
(#127).

## Worktree Context

**Container**: will share the [[fix-speed-up-e2e]] worktree
(`/Users/fres/Developer/rocm-cli-wt/fix-speed-up-e2e`), re-branching in place.
