<!-- This is an AI instruction file. Use this template when creating new WIP files. Fill in the placeholders. -->

# WIP: Speed up E2E test suite

**Stage:** 0-idea
**Pipeline:** standard
**Branch:** fix-speed-up-e2e
**Last Updated:** 2026-07-16

**Token Usage:** in=490 out=136554 cache_create=649620 cache_read=37285438 calls=246

---

## Problem

The E2E suite is slow enough to be a drag on the dev loop and on CI (self-hosted GPU runners are serial, 30–75 min per cycle). Slow runs mean fewer iterations, longer PR feedback, and more contention for shared hardware.

TODO: quantify current runtime and identify the biggest contributors (model downloads, vLLM readiness waits, redundant setup, serial scenarios).

## Command Coverage & Priority

Tiering is the primary lever: **P0 runs on every PR; P1 runs less often** (nightly/on-demand).

**P0 — every PR:**
- `rocm install`, `rocm examine`, `rocm diagnose`
- Strix Halo with the latest Qwen variant supported by Lemonade

**P0 — nightly** (heavy, too expensive per-PR):
- `rocm serve` — MI300X serving Qwen3.6-27B (this is the "MI300X" P0 scenario)

**P1 (everything else):** all other commands, models, and platform combinations.

## Solution

High-level approach TBD after profiling. Candidate levers:
- Cache/warm model weights so scenarios don't re-download (HF Hub cold pulls are a known long pole).
- Reduce redundant per-scenario setup / share fixtures where safe.
- Parallelize independent scenarios where hardware allows.
- Trim or tier scenarios so the fast path runs on every PR and the heavy path runs less often.

## Scenarios

Define expected behaviors in BDD-style (Gherkin) **before** writing any tests or code.

```gherkin
Scenario: [Abstract behavior description]
  Given [situation/context]
  When [user action or event]
  Then [observable outcome]
```

**Always activate the bdd-scenarios skill** before writing or reviewing scenarios.

## Implementation Steps

### Completed ✅
- ✅ Task #1: Investigate Strix-Ubuntu ~262s timeout cluster — root cause found (hardcoded VRAM floor) and fixed.
- ✅ Profile current E2E runtime from live CI run #29472891569 (baseline quantified: mock 7.6m / MI300X 8.0m / Strix-Windows 15.8m / Strix-Ubuntu 28.4m long pole).
- ✅ Implement VRAM-floor fix: added device-total probe, scaled floor to `min(150_000, total*0.9)`, verified compile + arithmetic. Committed and pushed to origin/fix-speed-up-e2e (commit `122d2be`).

### In Progress ⏳
- ⏳ Container gate running (cold build, background task `b4rdy2tpk`). Will validate VRAM-floor fix + commit under `RUSTFLAGS="-D warnings"` after completion.

### Todo 📋
- 📋 Task #3: Add E2E coverage for `rocm diagnose` (P0 gap).
- 📋 Task #4: Confirm Strix Halo Qwen variant (latest vs smallest).
- 📋 Task #5: Reduce mock lane per-scenario overhead.
- 📋 Verify VRAM-floor fix on real Strix-Ubuntu GPU (expect ~12 min saved).

## Next Steps

- Task #2: Rework CI tiering (per-PR vs nightly) to align with P0/P1 plan and reduce wall-clock queue time (~13h).
- Task #3: Add `rocm diagnose` E2E scenarios (P0 coverage gap).
- Verify VRAM-floor fix on real GPU (expect ~12 min savings on Strix-Ubuntu lane).

## Checklist

- [ ] Scenarios written and reviewed before any implementation
- [ ] If this adds a user command, is there also a tool action for the agent?
- [ ] If this adds a tool action, are there tests covering LLM-facing semantics (description clarity, action disambiguation)?
- [ ] All scenarios have corresponding tests

## Blockers / Open Questions

- **Coverage gap on diagnose**: `rocm diagnose` command exists but zero E2E scenarios cover it — real gap vs P0 plan (Task #3).
- **Strix Halo Qwen not latest**: suite uses `Qwen3-0.6B-GGUF` (smallest GGUF recipe) on lemonade; unclear if this is the "latest variant" you intended (Task #4 decision gate).
- **Tiering already exists**: `@nightly` gate + `E2E_INCLUDE_NIGHTLY=1` env gate already separates heavy scenarios (27B serve, cold install) from per-PR runs. Task #2 is to verify/tune this alignment, not build from scratch.
- **Real bugs separate**: EAI-7423 (lemonade-on-Strix-Linux serve fails) and EAI-7052 (lemonade Vulkan instability) are tracked known bugs in `expectations.toml`, separate from the VRAM-floor waste fix (Task #1).

## Notes

Related WIPs: [[test-e2e-tui-cucumber]], [[ci-manual-e2e]], [[persiste-app-dev-ci-runner]]. See memory `reference_rocm_cli_e2e_cucumber` for suite tiers/tags and runner gotchas.

## Worktree Context

**Worktree directory**: `/Users/fres/Developer/rocm-cli-wt/fix-speed-up-e2e`
- Recreate with: `create_worktree.sh fix-speed-up-e2e`

## Work Log

### 2026-07-16

- Fetched latest main (34 commits ahead); rebased branch (fast-forward, now at abb80fa).
- Discovered E2E suite exists on updated main (`tests/e2e-cucumber/`, 4 features, ~21 scenarios).
- Audited coverage vs P0/P1 plan: `examine` ✅, `install` ⚠️ (nightly-only), `diagnose` ❌ (zero coverage), `serve` Qwen3.6-27B MI300X ✅ (nightly), Strix Halo ⚠️ (uses smallest GGUF, not latest variant).
- Identified gaps: diagnose uncovered, Strix Halo model choice unclear, per-PR vs nightly CI tiering not aligned with plan.
- Baseline from run 29472891569 (PR fix/chat-stale-url, PR event → @nightly excluded, so 27B serve did NOT run). Job compute: mock 7.6m / MI300X-GPU 8.0m / Strix-Windows 15.8m / Strix-Ubuntu 28.4m (long pole). Wall clock ~13h = serial GPU queue, not compute.
- ROOT CAUSE of Strix-Ubuntu long pole: each failing serve scenario = a fixed ~262s = ~120s `wait_for_free_vram` (ALWAYS times out) + ~60s port-drain + 90s `wait_for_model` (shortened xfail, real bug EAI-7423 lemonade-on-Strix-Linux).
  - The ~120s dead wait is an AVOIDABLE BUG: `MIN_FREE_VRAM_MIB = 150_000` (serving_steps.rs:130) is a hardcoded global floor sized for the 27B MI300X model. Strix Halo has 62 GiB *unified* RAM → can never report 150 GB free VRAM → `free >= 150_000` never true → full 2-min deadline burns on EVERY serve scenario. ~12 min of pure dead time on the lane, hits any GPU with <150 GB total VRAM.
  - FIX: make floor relative to device total, e.g. `min(150_000, total_vram * 0.9)`. Strix then needs ~90% of its real VRAM free (drains promptly); MI300X keeps the large floor for 27B.
- IMPLEMENTED VRAM-floor fix (serving_steps.rs): added `total_vram_mib`/`vram_mib` probe (amd-smi TOTAL_VRAM + rocm-smi total), `required_free_vram_mib(total) = min(150_000, total*0.9)`; `wait_for_free_vram` now uses the scaled floor. MI300X floor unchanged (150 GB); Strix ~43-55 GB (reachable). `cargo check -p e2e-cucumber --test e2e` passes.
- NOT yet verified on real GPU — behavioral confirmation needs a Strix-Ubuntu CI run (expect ~12 min saved on that lane). Compile + arithmetic verified only.
- Moved to Task #2 (CI tiering). Note: EAI-7423 (lemonade-on-Strix-Linux serve fails at 90s) is a real bug separate from the VRAM-floor waste fix.

- Committed VRAM-floor fix via `git-commit-with-fallback`, added DCO sign-off, all pre-push hooks passed (346 rocm-bin tests). Pushed to origin; branch tracking origin/fix-speed-up-e2e at commit `122d2be`.
- Read `test-add-e2e-robot-framework.md` to understand testing regimen: cucumber-rs suite auto-reconciles pass/xfail via `expectations.toml` per-platform; share-one-runtime (Task #22 on that WIP) solves per-scenario cold-install cost; `@nightly` tag + `E2E_INCLUDE_NIGHTLY=1` already gates heavy scenarios. Per-PR vs nightly tiering exists — Task #2 is to verify/tune alignment with P0 scope.
- Created 5 issue tasks to organize the work; Task #1 (VRAM-floor fix) completed; Task #2 deferred (tiering verification).
- Five levers identified: (1) VRAM-floor waste (Task #1 ✅), (2) CI tiering tune (Task #2), (3) diagnose coverage gap (Task #3), (4) Strix Qwen variant decision (Task #4), (5) mock lane overhead (Task #5).
- Reviewed memory system: `reference_rocm_cli_commit_push_workflow.md` already documents signed+signed-off + container gate + `--no-verify` (no gap, my miss not consulting it before commit). Confirmed per-project memory on disk, shared across worktrees.
- Recreated git-hidden `workspace/wip/container-test.sh` helper (was missing from fresh worktree; runs full clippy+test gate under `-D warnings`).
- Feedback saved: "act, don't ask when obvious" — for low-risk obvious work, just execute instead of asking permission.
