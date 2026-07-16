# WIP: Fix flaky Strix-Halo Windows E2E — share the lemonade engine across scenarios

**Stage:** 2-implementation
**Pipeline:** standard
**Branch:** fix-e2e-share-lemonade-engine (local-only, not pushed; off origin/main @ abb80fa)
**Last Updated:** 2026-07-17 (idle flush)

---

## Problem

`E2E tests (Strix Halo, Windows)` fails on nearly every CI run (15 failures observed
2026-07-16; ~14–19 min then red). This is the scarcest CI platform — one physical
Strix-Halo Windows box, cannot be scaled — so each failure (which re-queues and doubles
load) is a real merge-queue throughput drain.

## Root cause (traced, not guessed)

- Two regression scenarios: `serve-default-engine-working-endpoint`
  (model_serving.feature:68) and `serve-readiness-contract` (feature:98). Both are
  `rocm serve` on the **lemonade** engine (native Windows skips vLLM → serves route to
  lemonade).
- Each lemonade serve triggers a **cold backend install** —
  `therock-dist-windows-gfx1151-7.13.0.tar.gz` (**4591 MB**) + a llama.cpp zip (217 MB) —
  which overruns `E2E_SERVE_TIMEOUT_SECS=300`, so the serve never reaches ready and the
  scenario fails.
- Why cold every time: the harness shares heavy artifacts across scenarios, but the
  lemonade **engine dir is not among them**. `use_shared_runtimes()` only symlinks
  `<isolated_root>/data/runtimes`. The lemonade engine + its 4.6 GB backend live under
  `data/engines/lemonade` (traced: lemonade `manifest_path` → `engine_dir("lemonade")` →
  `AppPaths::engine_dir` = `data_dir/engines/lemonade`, crates/rocm-core/src/lib.rs:807) —
  a SIBLING of `data/runtimes`, not symlinked → each scenario reinstalls it into its own
  isolated temp dir.
- Linux jobs are green because their serves use vLLM (covered by shared runtime + HF
  cache); only native-Windows/lemonade hits the un-shared engine path.

## Fix (implemented, not yet committed)

Mirror the proven `E2E_SHARED_RUNTIMES_DIR` pattern for a fourth shared artifact — the
engines dir. Two coordinated pieces (both required):

1. **Harness — share `data/engines` (opt-in).** `tests/e2e-cucumber/tests/e2e.rs`:
   `shared_engines_dir()` (reads `E2E_SHARED_ENGINES_DIR`) + `E2eWorld::use_shared_engines()`
   symlinking `<data>/engines` → shared tree, byte-for-byte parity with
   `use_shared_runtimes()` incl. the Windows symlink-privilege fallback.
   `tests/e2e-cucumber/tests/e2e/runtime_steps.rs`: call `world.use_shared_engines()` in the
   `"a managed runtime is active"` step (scenarios 6/6b/7/8 opt in; clean-slate scenarios
   stay isolated).
2. **CI — pre-warm the engine once.** `.github/workflows/ci.yml`, all three E2E jobs:
   export `E2E_SHARED_ENGINES_DIR=$prewarm/data/engines` and an independently-guarded
   `rocm engines install lemonade` into `$prewarm` so the 4.6 GB backend installs once per
   runner. Windows is the one that needs it; GPU + Strix-Ubuntu get it for parity.

Verified during implementation: `engines install lemonade` → `install_response` →
`install_best_llamacpp_backend` pulls exactly the 4.6 GB backend;
`engine_manages_own_runtime("lemonade")==true` so the engine pre-warm needs no active
runtime; local `cargo build` + `cargo clippy --locked -p e2e-cucumber --test e2e -- -D warnings`
both clean; ci.yml valid YAML.

## Work Log

**2026-07-17 (idle flush):** Session idle for 10 minutes, auto-flushing WIP state.

**2026-07-17 (idle flush):** Session idle for 10 minutes, auto-flushing WIP state.

**2026-07-16 (idle flush):** Session idle for 10 minutes, auto-flushing WIP state.

**2026-07-16:** Diagnosed root cause from failing job 87668414041 logs (4.6 GB

**2026-07-16:** Diagnosed root cause from failing job 87668414041 logs (4.6 GB
per-scenario re-download → 300s serve timeout). Traced lemonade backend install path
through engines/lemonade/src/lib.rs + crates/rocm-core. Implemented harness sharing + CI
pre-warm across all 3 E2E jobs. Branch created off origin/main (worktree was behind).
Build + clippy green. Not committed/pushed yet.

## Next Steps

- Commit + push branch (needed before any dispatch — `workflow_dispatch` runs the remote
  ref).
- Scoped probe: `gh workflow run ci.yml --ref fix-e2e-share-lemonade-engine
  -f platform=strix-windows -f name_filter='<6|7|8 serve scenarios>'`. Confirm the 4.6 GB
  `therock-dist-windows-*` download appears ONCE in pre-warm, not per-scenario, and the
  scenarios pass. NOTE: `--name` matches scenario display names, not @ids — verify the
  regex shape first.
- First dispatch on a runner pays the full pre-warm once; second run should log
  "shared lemonade engine already present … skipping pre-warm".
- Full Strix-Windows dispatch (no filter) → expect green, ~15 min, 0 unexpected failures.
- Open PR. No AI refs in commit/PR/Jira (per repo convention).

## Caveats / open items

- `#[cfg(windows)]` symlink arm NOT compile-checked locally (no Windows Rust target); it's
  a verbatim copy of the Windows-proven `use_shared_runtimes` arm → safe by parity, but the
  Strix-Windows dispatch is the real proof.
- State-leak: `data/engines/lemonade` also holds mutable logs/state/locks; the suite
  asserts on serve output + `data/services` state, not engine-internal state — same accepted
  tradeoff the runtimes symlink already makes.

## Related

- [[fix-speed-up-e2e]] — broader E2E-speedup effort (distinct; this is one specific flake).
- [[persiste-app-dev-ci-runner]] — runner capacity work (2nd GPU runner, Kueue cpu quota
  18, merge-queue Build concurrency 1). Complementary: that adds GPU capacity; this reduces
  demand on the un-scalable Strix box.
- [[test-e2e-tui-cucumber]] / [[ci-manual-e2e]] — the cucumber suite + manual-dispatch loop.
