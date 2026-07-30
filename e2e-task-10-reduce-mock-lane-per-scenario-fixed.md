# WIP: E2E Task #10: reduce mock-lane per-scenario fixed overhead

**Stage:** 0-idea
**Pipeline:** lightweight
**Branch:** e2e-task-10-reduce-mock-lane-per-scenario-fixed
**Pre-PR-check:** none
**Last Updated:** 2026-07-30

**Token Usage:** in=0 out=0 cache_create=0 cache_read=0 calls=0

---

## Problem

Parent: fix-speed-up-e2e umbrella (wlticket #47). Parent WIP: /Users/fres/Developer/rocm-cli-progress/fix-speed-up-e2e.md (Task #10). Gets its own WIP referencing the parent when work starts.

GOAL: Reduce the mock lane's fixed per-scenario overhead (~4.8s/scenario, multiplied across ~12 scenarios on the GitHub-hosted no-GPU lane).

DISTINCT FROM Tasks #5-#7 (mock/real split, EAI-7484, shipped PR #136): those MOVED scenarios off GPU onto the mock lane; this attacks the mock lane's own fixed cost per scenario (setup/teardown/fixture/process spin-up), which is now more impactful since more scenarios run there post-#136.

NEXT STEP when work starts: profile where the ~4.8s goes (binary launch, temp-dir config/data/cache setup, MockServer start, service-record planting) and cut the shared fixed cost. Lowest priority of the remaining speedup tasks (P3). SUPERSEDES the mock-lane-overhead portion of old bundled ticket #44.

## Solution

_TBD — design the approach._

## Next Steps

1. Design the solution, then implement.

## Notes

- Promoted from WL-177 (rocm-cli, +perf +task).

## Worktree Context

**Worktree directory**: created on start under `~/Developer/rocm-cli-wt/e2e-task-10-reduce-mock-lane-per-scenario-fixed`.

## Work Log

### 2026-07-30

- Promoted from WL-177 into a worktree-backed task.
