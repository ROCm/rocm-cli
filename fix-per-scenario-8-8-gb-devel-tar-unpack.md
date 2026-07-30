# WIP: Fix per-scenario 8.8 GB devel-tar unpack blowing E2E 90-min CI cap

**Stage:** 3-test-proof — ON HOLD
**Pipeline:** lightweight
**Branch:** fix-per-scenario-8-8-gb-devel-tar-unpack
**Pre-PR-check:** none
**Last Updated:** 2026-07-30

**Token Usage:** in=1186 out=107614 cache_create=371292 cache_read=9927916 calls=88

---

## Problem

Fix the per-scenario 8.8 GB devel-tar unpack that blows the E2E 90-min CI cap. Root-caused: `rocm install sdk` pulls `rocm[libraries,devel]` and the post-install probe extracts `_devel.tar` (8.8->12 GB) into each scenario's isolated data dir; 10 of 11 GPU scenarios only serve/chat and don't need devel. Designed fix: env-gate the extras in `therock_pip_package_specs` (`apps/rocm/src/therock.rs`) via a new `ROCM_CLI_THEROCK_EXTRAS` (default `libraries,devel`); harness sets it to `libraries` for the `a managed runtime is active` precondition, `runtime-install-sdk-active` keeps full devel. Prove with a 2-scenario `@probe` run before full dispatch. Also watch: GH `timeout-minutes` didn't self-cancel promptly (~95min). (moved from test-add-e2e-robot-framework 2026-07-17)

## Solution

✅ **Env-gate extras in therock.rs**: Added `ROCM_CLI_THEROCK_EXTRAS` (default `libraries,devel`) read by `parse_therock_extras()`/`therock_extras()`; `therock_pip_package_specs()` now constructs `rocm[...]` using the env-gated extras. Updated display/policy string to show actual extras. 5 new unit tests (default/explicit/blank parsing, library-only spec).

✅ **Wire extras=libraries into precondition**: Updated `a managed runtime is active` step in `runtime_steps.rs` to install with `ROCM_CLI_THEROCK_EXTRAS=libraries`, skipping the 8.8→12 GB devel unpack. `runtime-install-sdk-active` unchanged (plain `run_rocm` → full devel).

✅ **Verify probe fallback**: Confirmed from `ROCM_SDK_PROBE_SCRIPT` that `root_path`/`bin_path` fall back to package-derived roots when absent, and required `amdhip64`/`hipblas` resolve from `libraries` alone — devel not needed for serve/chat scenarios.

## Blockers

**BLOCKED (awaiting user):** Ready to prove with 2-scenario `@probe` GPU dispatch. Implementation complete and locally verified (both crates build; 50/50 therock unit tests pass, 5 new). Awaiting go-ahead to burn shared GPU CI (~30–75 min) for real-GPU timing proof before full dispatch.

## Next Steps

1. *User to approve*: Run 2-scenario `@probe` GPU dispatch to confirm devel tar is skipped and timing improves.
2. Full dispatch and monitor E2E runtime against 90-min cap.
3. Consider GH Actions `timeout-minutes` self-cancellation delay (~95 min vs. spec).

## Notes

- Promoted from WL-88 (rocm-cli, +ci +perf).

## Worktree Context

**Worktree directory**: created on start under `~/Developer/rocm-cli-wt/fix-per-scenario-8-8-gb-devel-tar-unpack`.

## Work Log

### 2026-07-30 — Implementation Complete

- Implemented `ROCM_CLI_THEROCK_EXTRAS` env (default `libraries,devel`) in `therock.rs`: added `parse_therock_extras()` pure parser, `therock_rocm_spec_with_extras()` builder, updated `therock_pip_package_specs()` to use env-gated extras. Updated dry-run policy display to show actual extras.
- Added 5 unit tests: default/explicit/blank parsing, library-only rocm spec, all pass.
- Wired `ROCM_CLI_THEROCK_EXTRAS=libraries` into `a managed runtime is active` precondition (runtime_steps.rs) via `run_rocm_with_env()`, skipping 8.8→12 GB devel unpack in shared-tree scenarios. `runtime-install-sdk-active` unchanged (full devel).
- Both crates build; 50/50 therock tests pass; awaiting user go-ahead for 2-scenario GPU `@probe` dispatch proof.
