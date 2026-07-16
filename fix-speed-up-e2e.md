<!-- This is an AI instruction file. Use this template when creating new WIP files. Fill in the placeholders. -->

# WIP: Speed up E2E test suite

**Stage:** 0-idea
**Pipeline:** standard
**Branch:** fix-speed-up-e2e
**Last Updated:** 2026-07-16

**Token Usage:** in=176 out=37358 cache_create=262209 cache_read=9155512 calls=89

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
- (none yet)

### In Progress ⏳
- (none yet)

### Todo 📋
- 📋 Profile current E2E runtime; identify the biggest time sinks
- 📋 Decide which levers to pull (caching, parallelism, tiering)
- 📋 Implement + measure improvement

## Next Steps

Profile the suite to establish a baseline runtime and find the dominant cost before changing anything.

## Checklist

- [ ] Scenarios written and reviewed before any implementation
- [ ] If this adds a user command, is there also a tool action for the agent?
- [ ] If this adds a tool action, are there tests covering LLM-facing semantics (description clarity, action disambiguation)?
- [ ] All scenarios have corresponding tests

## Blockers / Open Questions

- **Coverage gap on diagnose**: `rocm diagnose` command exists but zero E2E scenarios cover it — real gap vs P0 plan.
- **Strix Halo Qwen not latest**: suite uses `Qwen3-0.6B-GGUF` (smallest GGUF recipe) on lemonade; unclear if this is the "latest variant supported by Lemonade" you intended.
- **Per-PR vs nightly split unclear**: CI currently has no per-PR-only tier — `xtask e2e` runs everything applicable per-host, and `@nightly` is only pulled in via manual dispatch. Need to clarify how your P0-every-PR intent maps to CI workflows.
- **Baseline unknown**: need to inspect a live CI run (e.g. #29472891569) to quantify current runtimes and identify bottlenecks.

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
- Next: inspect live CI run to quantify baseline runtime, then decide scope (profiling, coverage gaps, tiering rewire).
