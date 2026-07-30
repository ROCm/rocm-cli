# WIP: Fix per-scenario 8.8 GB devel-tar unpack blowing E2E 90-min CI cap

**Stage:** 0-idea
**Pipeline:** lightweight
**Branch:** fix-per-scenario-8-8-gb-devel-tar-unpack
**Pre-PR-check:** none
**Last Updated:** 2026-07-30

**Token Usage:** in=0 out=0 cache_create=0 cache_read=0 calls=0

---

## Problem

Fix the per-scenario 8.8 GB devel-tar unpack that blows the E2E 90-min CI cap. Root-caused: `rocm install sdk` pulls `rocm[libraries,devel]` and the post-install probe extracts `_devel.tar` (8.8->12 GB) into each scenario's isolated data dir; 10 of 11 GPU scenarios only serve/chat and don't need devel. Designed fix: env-gate the extras in `therock_pip_package_specs` (`apps/rocm/src/therock.rs`) via a new `ROCM_CLI_THEROCK_EXTRAS` (default `libraries,devel`); harness sets it to `libraries` for the `a managed runtime is active` precondition, `runtime-install-sdk-active` keeps full devel. Prove with a 2-scenario `@probe` run before full dispatch. Also watch: GH `timeout-minutes` didn't self-cancel promptly (~95min). (moved from test-add-e2e-robot-framework 2026-07-17)

## Solution

_TBD — design the approach._

## Next Steps

1. Design the solution, then implement.

## Notes

- Promoted from WL-88 (rocm-cli, +ci +perf).

## Worktree Context

**Worktree directory**: created on start under `~/Developer/rocm-cli-wt/fix-per-scenario-8-8-gb-devel-tar-unpack`.

## Work Log

### 2026-07-30

- Promoted from WL-88 into a worktree-backed task.
