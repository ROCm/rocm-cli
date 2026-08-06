# WIP: EPIC: Speed up E2E test suite

**Stage:** 0-idea
**Pipeline:** lightweight
**Branch:** epic-speed-up-e2e-test-suite
**Pre-PR-check:** none
**Last Updated:** 2026-08-06

**Token Usage:** in=0 out=0 cache_create=0 cache_read=0 calls=0

---

## Problem

Coordinator container for the E2E-speedup 'Task #N' effort (rocm-cli). An EPIC proper: no worktree/code of its own; its children do the work, a coordinator sequences them.

WHY a fresh epic (not reusing WL-123): WL-123 'Speed up E2E test suite' was the ORIGINAL informal umbrella, but it degenerated into a hybrid — tagged 'task', with its OWN worktree (rocm-cli-wt/fix-speed-up-e2e) + live session doing code (the mock-real-split task, whose PR merged), and its children were never actually parented to it. An epic must be a pure container; converting WL-123 would drag a worktree + merged branch into the epic role and confuse the rollup/coordinator logic. So WL-123 is being closed separately (by fres) as the leaf it actually became, and this clean epic takes over the umbrella role.

CHILDREN (still-open speedup tasks, reparented under this epic):
- WL-175 — E2E Task #8: gate GPU serve matrix to merge_group + keep a PR canary
- WL-176 — E2E Task #9: narrow 'serve' paths-filter so non-serve Rust PRs skip GPU
- WL-89  — Task #11: rocm fix fix-2-unset-override --dry-run panics rc=101

ALREADY RESOLVED (not children, for history): WL-177 (Task #10 mock-lane overhead — closed obsolete 2026-08-01, profiling refuted the premise), WL-217 (serve schedule+paths-filter+mock-lane umbrella, resolved), WL-123 (mock-real-split, PR merged; fres closing).

COORDINATOR NOTE: WL-175 (#157) and WL-176 (#156) are open PRs that have collided before (cross-PR conflict handled earlier this session) — a prime reason for a coordinator: sequence the parallel-safe ones, hold conflicts, surface only real gates. WL-89 (#185) is APPROVED, waiting on GPU E2E CI.

## Solution

_TBD — design the approach._

## Next Steps

1. Design the solution, then implement.

## Notes

- Promoted from WL-366 (rocm-cli, +epic +wl:366).

## Worktree Context

**Worktree directory**: created on start under `~/Developer/rocm-cli-wt/epic-speed-up-e2e-test-suite`.

## Work Log

### 2026-08-06

- Promoted from WL-366 into a worktree-backed task.
