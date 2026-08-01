# WIP: Fix per-scenario 8.8 GB devel-tar unpack blowing E2E 90-min CI cap

**Stage:** 4-dispatch-probe
**Pipeline:** lightweight
**Branch:** fix-per-scenario-8-8-gb-devel-tar-unpack
**Pre-PR-check:** passed (opencode-reviewer, 2026-08-01, @e9e9fb8+169f13be85104885)
**Last Updated:** 2026-08-01

**Token Usage:** in=1412 out=234854 cache_create=2325943 cache_read=28841869 calls=201

---

## Problem

Fix the per-scenario 8.8 GB devel-tar unpack that blows the E2E 90-min CI cap. Root-caused: `rocm install sdk` pulls `rocm[libraries,devel]` and the post-install probe extracts `_devel.tar` (8.8->12 GB) into each scenario's isolated data dir; 10 of 11 GPU scenarios only serve/chat and don't need devel. Designed fix: env-gate the extras in `therock_pip_package_specs` (`apps/rocm/src/therock.rs`) via a new `ROCM_CLI_THEROCK_EXTRAS` (default `libraries,devel`); harness sets it to `libraries` for the `a managed runtime is active` precondition, `runtime-install-sdk-active` keeps full devel. Prove with a 2-scenario `@probe` run before full dispatch. Also watch: GH `timeout-minutes` didn't self-cancel promptly (~95min). (moved from test-add-e2e-robot-framework 2026-07-17)

## Solution

✅ **Env-gate extras in therock.rs**: Added `ROCM_CLI_THEROCK_EXTRAS` (default `libraries,devel`) read by `parse_therock_extras()`/`therock_extras()`; `therock_pip_package_specs()` now constructs `rocm[...]` using the env-gated extras. Updated display/policy string to show actual extras. 5 new unit tests (default/explicit/blank parsing, library-only spec).

✅ **Wire extras=libraries into precondition**: Updated `a managed runtime is active` step in `runtime_steps.rs` to install with `ROCM_CLI_THEROCK_EXTRAS=libraries`, skipping the 8.8→12 GB devel unpack. `runtime-install-sdk-active` unchanged (plain `run_rocm` → full devel).

✅ **Verify probe fallback**: Confirmed from `ROCM_SDK_PROBE_SCRIPT` that `root_path`/`bin_path` fall back to package-derived roots when absent, and required `amdhip64`/`hipblas` resolve from `libraries` alone — devel not needed for serve/chat scenarios.

✅ **Gate GPU pre-warms**: Updated `app-dev-gpu`, `strix-halo-ubuntu`, and `strix-halo-windows` pre-warm blocks in `.github/workflows/ci.yml` to set `ROCM_CLI_THEROCK_EXTRAS=libraries` (PowerShell cleanup added), so the shared venv skips 8.8→12 GB devel unpack and the fix takes effect on CI runners.

✅ **Commit + sign**: All 3 files committed with signed commit (msg: "perf(e2e): gate TheRock devel extra behind ROCM_CLI_THEROCK_EXTRAS"). Pre-push hook blocks on known macOS-only pid tests; confirmed identical failures on clean base (not caused by diff).

## Blockers

**BLOCKED (awaiting user):** Container Linux gate offline build was stopped (background task). Seeded container CARGO_HOME from host (1010 crates, 1.1 GB) and has offline script at `workspace/wip/container-test.sh`. Can retry: `CARGO_OFFLINE=1 workspace/wip/container-test.sh all`. Branch committed+signed; ready to push `--no-verify` once gate clears. Then dispatch 2-scenario `app-dev-gpu` probe.

## Next Steps

1. Wait for container gate (clippy + workspace tests + e2e lib, `-D warnings`) to complete.
2. Once green, push `--no-verify` (justified by container gate) and dispatch scoped 2-scenario `app-dev-gpu` probe on `serve-vllm-inference` + `serve-default-engine-inference` (both hit the precondition).
3. Monitor pre-warm log to detect if shared tree was skipped (would mean no timing proof, need to manually reset `/RUNNER_WORKSPACE/e2e-prewarm`); confirm devel tar NOT unpacked and scenario timing improves.
4. If probe succeeds, full dispatch (all scenarios, all platforms) and verify E2E total is under 90 min.

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

### 2026-08-01 — Push Preparation & Dispatch Setup

- Discovered CI confound: GPU pre-warm (not precondition) creates the shared venv that all scenarios reuse. Updated `app-dev-gpu`, `strix-halo-ubuntu`, and `strix-halo-windows` pre-warm blocks to gate via `ROCM_CLI_THEROCK_EXTRAS=libraries`, so fix takes effect on CI.
- Committed all 3 files (therock.rs, runtime_steps.rs, ci.yml) with signed commit. Pre-push macOS hook fails on 3 known OS-only pid tests (`managed_stop_*`), confirmed not caused by this diff.
- Created container-test.sh Linux gate (clippy + workspace tests + e2e lib, `-D warnings`). Cold build hit transient cargo download timeouts (container networking flaky). Seeded container CARGO_HOME from host (1010 crates) and re-ran with `CARGO_OFFLINE=1`. Background build task stopped; script saved for retry.
- Added Windows strix pre-warm gating (PowerShell env var + cleanup) per pre-PR reviewer feedback. Branch ready for push `--no-verify` once container gate completes.
