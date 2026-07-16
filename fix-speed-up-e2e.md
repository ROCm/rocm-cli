<!-- This is an AI instruction file. Use this template when creating new WIP files. Fill in the placeholders. -->

# WIP: Speed up E2E test suite

**Stage:** 0-idea
**Pipeline:** standard
**Branch:** fix-speed-up-e2e
**Last Updated:** 2026-07-16

**Token Usage:** in=572 out=173915 cache_create=696408 cache_read=47588686 calls=287

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
- ⏳ Scoped dispatch running on Strix-Ubuntu (run 29529197875, 2 serve scenarios, measuring per-serve step duration vs ~262s baseline).

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

- **Task #1 completed:** identified VRAM-floor waste as ~12 min of dead time on Strix-Ubuntu lane (hardcoded 150 GB floor vs 62 GiB device total); implemented fix in `serving_steps.rs` (device-relative floor = min(150GB, total*0.9)), verified arithmetic + compile.
- **Commit `122d2be` validated:** container gate (clippy + tests under `-D warnings`) passed clean; fix has zero compile warnings, unit tests pass, mock e2e reconciles to 0 unexpected failures.
- **Scoped dispatch fired:** run 29529197875 on Strix-Ubuntu (2 serve scenarios, measuring step duration vs ~262s baseline); expect ~120s per-scenario drop if fix works.
- **E2E regimen studied:** cucumber-rs suite already has per-PR vs nightly split (`@nightly` tag + `E2E_INCLUDE_NIGHTLY=1`); `expectations.toml` per-platform reconciliation; Task #22 (share-one-runtime) addresses cold-install overhead. P0 coverage gaps: `diagnose` (zero scenarios), Strix Qwen (uses smallest, not latest).
- **Saved methods to memory:** scoped dispatch for rapid fix validation (don't wait for full CI), act-don't-ask feedback, confirmed commit-push workflow already well-documented.
- **Five levers identified:** (1) VRAM-floor ✅, (2) CI tiering tune (pending), (3) diagnose coverage (pending), (4) Strix Qwen variant (user decision), (5) mock overhead (pending).
