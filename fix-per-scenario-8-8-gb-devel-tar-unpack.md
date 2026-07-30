# WIP: Fix per-scenario 8.8 GB devel-tar unpack blowing E2E 90-min CI cap

**Stage:** 2-implementer
**Pipeline:** lightweight
**Branch:** fix-per-scenario-8-8-gb-devel-tar-unpack
**Pre-PR-check:** none
**Last Updated:** 2026-07-30

**Token Usage:** in=144 out=49649 cache_create=371292 cache_read=9927916 calls=72

---

## Problem

Fix the per-scenario 8.8 GB devel-tar unpack that blows the E2E 90-min CI cap. Root-caused: `rocm install sdk` pulls `rocm[libraries,devel]` and the post-install probe extracts `_devel.tar` (8.8->12 GB) into each scenario's isolated data dir; 10 of 11 GPU scenarios only serve/chat and don't need devel. Designed fix: env-gate the extras in `therock_pip_package_specs` (`apps/rocm/src/therock.rs`) via a new `ROCM_CLI_THEROCK_EXTRAS` (default `libraries,devel`); harness sets it to `libraries` for the `a managed runtime is active` precondition, `runtime-install-sdk-active` keeps full devel. Prove with a 2-scenario `@probe` run before full dispatch. Also watch: GH `timeout-minutes` didn't self-cancel promptly (~95min). (moved from test-add-e2e-robot-framework 2026-07-17)

## Solution

✅ **Env-gate extras in therock.rs**: Added `ROCM_CLI_THEROCK_EXTRAS` (default `libraries,devel`) read by `parse_therock_extras()`/`therock_extras()`; `therock_pip_package_specs()` now constructs `rocm[...]` using the env-gated extras. Updated display/policy string to show actual extras. 5 new unit tests (default/explicit/blank parsing, library-only spec).

✅ **Wire extras=libraries into precondition**: Updated `a managed runtime is active` step in `runtime_steps.rs` to install with `ROCM_CLI_THEROCK_EXTRAS=libraries`, skipping the 8.8→12 GB devel unpack. `runtime-install-sdk-active` unchanged (plain `run_rocm` → full devel).

✅ **Verify probe fallback**: Confirmed from `ROCM_SDK_PROBE_SCRIPT` that `root_path`/`bin_path` fall back to package-derived roots when absent, and required `amdhip64`/`hipblas` resolve from `libraries` alone — devel not needed for serve/chat scenarios.

## Next Steps

1. Run 2-scenario `@probe` GPU dispatch to confirm devel tar is skipped and timing improves.
2. Full dispatch and monitor E2E runtime against 90-min cap.
3. Consider GH Actions `timeout-minutes` self-cancellation delay (~95 min vs. spec).

## Notes

- Promoted from WL-88 (rocm-cli, +ci +perf).

## Worktree Context

**Worktree directory**: created on start under `~/Developer/rocm-cli-wt/fix-per-scenario-8-8-gb-devel-tar-unpack`.

## Work Log

### 2026-07-30

- Promoted from WL-88 into a worktree-backed task.
- Implemented env-gating in `therock_pip_package_specs()` via `ROCM_CLI_THEROCK_EXTRAS` (default `libraries,devel`); added `parse_therock_extras()` pure function and 4 unit tests for parsing (default, explicit, blank fallback).
- Updated `a managed runtime is active` precondition step to install with `extras=libraries` via `run_rocm_with_env()`, skipping the 8.8 GB devel unpack in shared-tree scenarios.
- Confirmed both crates build; all 50 therock unit tests pass (5 new).
