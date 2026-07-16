<!-- This is an AI instruction file. Use this template when creating new WIP files. Fill in the placeholders. -->

# WIP: Speed up E2E test suite

**Stage:** 0-idea
**Pipeline:** standard
**Branch:** fix-speed-up-e2e
**Last Updated:** 2026-07-16

**Token Usage:** in=716 out=200519 cache_create=3415526 cache_read=65436447 calls=360

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
**Always activate the bdd-scenarios skill** before writing or reviewing scenarios.

### Task #3 — `rocm diagnose` / `rocm fix` coverage (P0 gap) — DRAFT, awaiting review

These cover the P0 `diagnose` command (and its sibling `fix`). All are **GPU-independent**
— `diagnose` matches purely on a symptom string against a closed catalog, `fix --dry-run`
and `fix` listing change nothing — so they belong on the **fast mock lane / per-PR tier**,
not `@requires-gpu`.

```gherkin
Scenario: 1 - Diagnosing a recognised failure reports a likely cause and a fix
  Given a user who hit a known ROCm failure
  When the user asks the CLI to diagnose that symptom
  Then the CLI reports a likely cause
  And it points to a fix for that cause

Scenario: 2 - Diagnosing an unrecognised failure admits it and routes the user onward
  Given a user who hit a failure the CLI does not recognise
  When the user asks the CLI to diagnose that symptom
  Then the CLI states that no known cause matched
  And it tells the user where to report the problem

Scenario: 3 - A diagnosis is available in machine-readable form for tooling
  Given a user who hit a known ROCm failure
  When the user asks the CLI to diagnose that symptom in machine-readable form
  Then the result identifies whether a known cause matched
  And it can be consumed by other tools

Scenario: 4 - The user can see every fix the CLI knows how to apply
  When the user asks the CLI which fixes it offers
  Then the CLI lists the fixes it can apply
  And each fix indicates whether the CLI can apply it automatically

Scenario: 5 - Previewing a fix explains the change without making it
  Given a user who has chosen a known fix
  When the user previews that fix without applying it
  Then the CLI describes what the fix would change
  And nothing on the machine is changed

Scenario: 6 - Asking for a fix the CLI does not know is refused clearly
  Given a user who names a fix the CLI does not offer
  When the user asks the CLI to apply that fix
  Then the CLI refuses
  And it explains that the fix is not recognised
```

**Technical Details** (mapping — kept out of the scenarios per bdd rules):

| # | @id | invocation | key assertion | tier |
|---|-----|-----------|---------------|------|
| 1 | diagnose-matches-known-symptom | `diagnose --symptom "unable to open /dev/kfd"` | exit 0; stdout has a `#1 [HIGH/LIKELY ...]` match + an `id:`/`plan:` line | mock (per-PR) |
| 2 | diagnose-no-match-routes-upstream | `diagnose --symptom "totally unrelated gibberish"` | exit 0; "no known misconfiguration matched" + a report/route target | mock |
| 3 | diagnose-json-has-match-flag | `diagnose --symptom "unable to open /dev/kfd" --json` | exit 0; parseable JSON; `matched` non-empty / `has_match` true | mock |
| 4 | fix-lists-known-recipes | `fix` (no id) | exit 0; "Available fix-ids" + ≥1 `[AUTO]`/`[PRINT-ONLY]` row | mock |
| 5 | fix-dry-run-changes-nothing | `fix fix-4-render-group --dry-run` | exit 0; prints a plan; no mutation (isolated data dir untouched) | mock |
| 6 | fix-unknown-id-rejected | `fix fix-does-not-exist` | non-zero exit (code 2); message names the unknown id | mock |

Notes for implementation (NOT part of the spec): `diagnose` always exits 0 (it's a query —
branch on `--json` fields, never the code). `--symptom "unable to open /dev/kfd"` scores exactly
50 = `MIN_SCORE_FOR_MATCH`, a deterministic match with no GPU. `fix fix-9-igpu-dgpu` mutates —
use `fix-4-render-group` (a print-only recipe) for the dry-run scenario. All 6 need
`expectations.toml` entries (expect-pass on all platforms; none are known bugs). Candidate file:
new `tests/e2e-cucumber/features/diagnose.feature`.

## Implementation Steps

### Completed ✅
- ✅ Task #1: Investigate Strix-Ubuntu ~262s timeout cluster — root cause found (hardcoded VRAM floor) and fixed.
- ✅ Profile current E2E runtime from live CI run #29472891569 (baseline quantified: mock 7.6m / MI300X 8.0m / Strix-Windows 15.8m / Strix-Ubuntu 28.4m long pole).
- ✅ Implement VRAM-floor fix: added device-total probe, scaled floor to `min(150_000, total*0.9)`, verified compile + arithmetic. Committed and pushed to origin/fix-speed-up-e2e (commit `122d2be`).

### In Progress ⏳
- ⏳ Task #1 probe: run 29529197875 (Strix-Ubuntu, 2 serve scenarios) backlogged on serial GPU runner (still queued). Auto-loop every 10m checking status. On completion: if timing drops ~120s/serve, open PR for `122d2be`; else diagnose + iterate.
- ⏳ Task #3: BDD scenarios for `rocm diagnose`/`fix` (6 scenarios drafted, GPU-independent mock-lane, awaiting review before step implementation).

### Todo 📋
- 📋 Task #2: Rework CI tiering (per-PR vs nightly) — verify/tune existing `@nightly` split.
- 📋 Task #3: Implement diagnose E2E test steps (scenarios drafted, awaiting review).
- 📋 Task #4: Confirm Strix Halo Qwen variant (user decision: latest vs smallest).
- 📋 Task #5: Reduce mock lane per-scenario overhead (fixed overhead ~4.8s/scenario, multiply across 12).
- 📋 Verify VRAM-floor fix on real Strix-Ubuntu GPU (run 29529197875 in progress; expect ~12 min saved).

## Next Steps

- Probe run 29529197875 completes → verify VRAM-floor fix impact → if successful, open PR for `122d2be`; if not, diagnose and iterate.
- Task #3: Review & implement diagnose E2E scenarios (6 drafted, mock-lane ready).
- Task #4: Confirm with user whether Strix Qwen should be latest variant or smallest (current: smallest).
- Task #2: Tune CI tiering alignment (review existing `@nightly` structure vs P0/P1 plan).

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

### 2026-07-16 (idle flush)

Session idle for 10 minutes, auto-flushing WIP state.

### 2026-07-16 — Task #1 fix complete & probe dispatched, Task #3 scenarios drafted, container gate validated

- **Task #1 root cause + fix:** hardcoded `MIN_FREE_VRAM_MIB=150_000` (sized for MI300X only) vs Strix's 62 GiB total → every serve waited 2 min for an unreachable threshold (~12 min wasted on lane). Fix: `required_free_vram_mib(total) = min(150_000, total*0.9)`. Commit `122d2be` (signed/signed-off, pushed origin/fix-speed-up-e2e).
- **Container gate validation:** linux build cold, but gate passed — clippy `-D warnings` clean on e2e-cucumber test target, workspace + lib tests green, mock e2e reconciliation correct (4 xfail/0 unexpected). Helper script bug fixed (removed `--tags` filter that broke reconciliation).
- **Probe fired:** run 29529197875 (narrow: Strix-Ubuntu, 2 serve scenarios, nightly off). Still queued on serial box behind 3 older runs. Auto-loop every 10m; on completion → if serve durations drop ~120s, open PR for `122d2be`; else diagnose + iterate.
- **Task #3 drafted:** 6 BDD scenarios for `rocm diagnose`/`fix` (GPU-independent, mock-lane, fills P0 gap). Full Gherkin + technical table in WIP scenarios section, ready for review before step code.
- **Methods documented:** scoped dispatch for rapid validation (narrow `name_filter`), act-don't-ask feedback, container-gate + commit workflow, sync_progress.sh for WIP updates saved to memory.
- **Awaiting:** Task #1 probe to complete, Task #4 user decision (Strix Qwen: latest or smallest).
