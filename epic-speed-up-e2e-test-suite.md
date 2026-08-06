# WIP: EPIC: Speed up E2E test suite

**Stage:** 1-active
**Pipeline:** lightweight
**Branch:** epic-speed-up-e2e-test-suite
**Pre-PR-check:** none
**Last Updated:** 2026-08-06

**Token Usage:** in=120 out=82581 cache_create=450428 cache_read=7697262 calls=63 → refreshed 2026-08-06

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

Coordinator sequences the three children by file-collision + dependency.

### Wave plan (as of 2026-08-06)

All three children are LIVE (adopted, not restarted) with open PRs:

| Child | PR | State | Wave | Gate |
|-------|-----|-------|------|------|
| WL-175 | #157 | OPEN, review-required, BEHIND main | 1 (foundation) | stale `windows-build-and-test` red on head 0fd0aed (predates #174) → nudged to rebase onto ec2bcb3 |
| WL-89  | #185 | OPEN, **APPROVED**, BLOCKED | 1 (disjoint) | `E2E tests (GPU)` 0s infra-fail — **both MI300X runners offline** (mi300x-0/1). REAL gate: needs runner stood up (restore-app-dev-runner) |
| WL-176 | #156 | OPEN, review-required, rebased locally | 2 (held) | Collides with #157 on ci.yml. Per fres 2026-08-05: #157 lands first, then #156 reduces to paths-filter-only delta. Reduction = source Edit → blocked by relay-gate on any relay; needs fres directly. Held behind #157. |

- **Collision:** #157 (WL-175) and #156 (WL-176) both edit `ci.yml` + e2e scenario files → serialized. #157 is foundation; #156 rebases to a delta after.
- **Disjoint:** #185 (WL-89) touches only the `rocm fix` SIGPIPE path → parallel-safe.

## Next Steps

1. WL-175/#157: await rebase result; if Windows goes green, foundation is ready → surface for merge (fres).
2. WL-89/#185: MI300X runners offline — surface to manager; the GPU E2E can't pass until a runner is online. Code is approved + otherwise green.
3. WL-176/#156: hold until #157 lands; the scope-reduction needs fres's direct go-ahead (relay-gate blocks source edits on relay).

## Notes

- Promoted from WL-366 (rocm-cli, +epic +wl:366).

## Worktree Context

**Worktree directory**: created on start under `~/Developer/rocm-cli-wt/epic-speed-up-e2e-test-suite`.

## Work Log

### 2026-08-06

- Promoted from WL-366 into a worktree-backed task.
- Coordinator session started. Reconciled epic: all 3 children (WL-175/#157, WL-176/#156, WL-89/#185) are ALREADY LIVE with open PRs → adopted, not restarted. Ownership precondition confirmed all 3 still `parent:wl-366`.
- Built wave plan (see Solution). Nudged WL-175/#157 (`--coordinator`) to rebase onto ec2bcb3 to clear the stale `windows-build-and-test` red (predates #174's Windows fix).
- Diagnosed WL-89/#185 gate: `E2E tests (GPU)` 0s instant-fail = MI300X runners offline (mi300x-0/1 both `offline`; re-runs can't clear it). Real gate → escalating.
- WL-176/#156 correctly held behind #157 per fres's 2026-08-05 cross-PR decision.
