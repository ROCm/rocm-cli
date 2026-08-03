# WIP: rocm fix fix-2-unset-override --dry-run panics rc=101

**Stage:** 0-idea
**Pipeline:** lightweight
**Branch:** rocm-fix-fix-2-unset-override-dry-run-panics-rc
**Pre-PR-check:** none
**Last Updated:** 2026-08-03

**Token Usage:** in=0 out=0 cache_create=0 cache_read=0 calls=0

---

## Problem

`rocm fix fix-2-unset-override --dry-run` panics rc=101 — a dry-run should never panic. Correctness bug found while probing E2E speedups (was Task #11 of the fix-speed-up-e2e umbrella, not a speedup task). Split out to its own item 2026-07-28.
Parent umbrella: fix-speed-up-e2e (wlticket #47); parent WIP /Users/fres/Developer/rocm-cli-progress/fix-speed-up-e2e.md (Task #11).

## Solution

_TBD — design the approach._

## Next Steps

1. Design the solution, then implement.

## Notes

- Promoted from WL-89 (rocm-cli, +bug +wl:89).

## Worktree Context

**Worktree directory**: created on start under `~/Developer/rocm-cli-wt/rocm-fix-fix-2-unset-override-dry-run-panics-rc`.

## Work Log

### 2026-08-03

- Promoted from WL-89 into a worktree-backed task.
