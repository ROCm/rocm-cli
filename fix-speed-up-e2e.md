<!-- This is an AI instruction file. Use this template when creating new WIP files. Fill in the placeholders. -->

# WIP: Speed up E2E test suite

**Stage:** 6-implementing — STAGED TASK, NOT done (PR #126 shipped as milestone; roadmap R1–R8 + Task #5 remain; re-branch in place for next chunk)
**Pipeline:** standard
**Branch:** fix-speed-up-e2e
**Last Updated:** 2026-07-17 (idle flush)

**Token Usage:** in=991 out=291238 cache_create=5250327 cache_read=93853352 calls=500

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

Task #3 (`rocm diagnose`/`fix` coverage) has moved to its own WIP + branch/PR: [[test-e2e-diagnose]] (PR #127). Its 6 scenarios, technical mapping, and live-container corrections now live there.

## Implementation Steps

### Completed ✅
- ✅ Task #1: Investigate Strix-Ubuntu ~262s timeout cluster — root cause found (hardcoded VRAM floor) and fixed.
- ✅ Profile current E2E runtime from live CI run #29472891569 (baseline quantified: mock 7.6m / MI300X 8.0m / Strix-Windows 15.8m / Strix-Ubuntu 28.4m long pole).
- ✅ Implement VRAM-floor fix: added device-total probe, scaled floor to `min(150_000, total*0.9)`, verified compile + arithmetic. Committed and pushed to origin/fix-speed-up-e2e (commit `122d2be`).

### Completed ✅ (cont.)
- ✅ Task #1 probe (run 29529197875, Strix-Ubuntu): **VRAM fix CONFIRMED**. Serve step = 91s (pure 90s readiness wait) vs ~262s baseline → the ~120s `wait_for_free_vram` dead-time is GONE. The job's "regression" flag was a FALSE ALARM from `--name` scoped mode: platform.json recorded 0 expectations (scoped `--name` bypasses the `.filter_run` resolutions-population path; a full run records all 25). So the EAI-7423 lemonade xfail couldn't reconcile — NOT caused by my change. LESSON: `--name` breaks reconciliation; judge scoped probes by step TIMING/behavior, not the pass/fail verdict.
- ✅ **Task #1 SHIPPED: PR #126 MERGED** into main (merge commit `e9a4b154`, 2026-07-17 ~03:03 CEST). All blocking checks green; Strix-Ubuntu lane PASS 15m37s (was 28.4m long pole). The 2 red lanes (MI300X-GPU, Strix-Windows) are non-blocking `continue-on-error` pre-existing failures, unaffected by this change (MI300X floor unchanged). Branch was rebased onto latest main before merge (commit became `a94600f`).

- ✅ **Task #3 SHIPPED to PR #127** — moved to its own WIP: [[test-e2e-diagnose]]. Full detail (scenarios, env-dependence saga, process notes) lives there.

### Todo 📋

The old Task #2/#4 were superseded by the **Efficiency roadmap (R1–R8)** below — same work, better framed. Do not track them separately:
- Task #2 (CI tiering per-PR vs nightly) → **R7 + R8** (gate heavy serve matrix to `merge_group`, narrow paths-filter).
- Task #4 (Strix Qwen variant: latest vs smallest) → **R1–R3**, decision gate is **R2** (settle "latest vs smallest" once, apply to both lemonade + vLLM).

Still tracked here (not covered by the roadmap):
- 📋 Task #5: Reduce mock lane per-scenario overhead (fixed overhead ~4.8s/scenario, multiply across 12). Distinct from R4–R6 (which moves scenarios *off* GPU, not the mock-lane fixed cost).
- 📋 FILE separately: `fix fix-2-unset-override --dry-run` panics rc=101 (a dry-run should never panic).

## Efficiency roadmap — fundamental levers (2026-07-17, discussed with user)

Stepping back from point-fixes: E2E wall-clock is dominated by REAL model serving
(cold weight load + engine startup + GPU ready), run serially on scarce hardware.
The levers below attack that from three angles — fewer real serves, cheaper real
serves, less-frequent real serves. Ordered by leverage. (R-prefixed to avoid clashing
with Task #1–5 above; R2/R4 overlap Task #4/#2 respectively — reconcile, don't dup.)

**Cheaper serves — smallest model (R1–R3):**
- 📋 R1 — Audit every GPU serve scenario: map scenario → current model → smallest
  viable model. Baseline: lemonade already uses `Qwen3-0.6B-GGUF` (smallest recipe);
  vLLM path uses `Qwen2.5-1.5B-Instruct` but code notes `Qwen2.5-0.5B` is the smallest
  vLLM-preferred entry.
- 📋 R2 — Switch vLLM serve target 1.5B → 0.5B (host_serve_target in serving_steps.rs);
  verify it still resolves to vLLM on Instinct. **Overlaps Task #4** (Strix Qwen
  variant decision) — settle the "latest vs smallest" call once, apply to both.
- 📋 R3 — Document the policy: any GPU serve scenario uses the smallest model that
  satisfies its assertion; large-model behavior is `@nightly` only.

**Fewer real serves — mock/real split (R4–R6, biggest structural win):**
- 📋 R4 — Classify every `@requires-gpu` scenario: genuinely-needs-real-inference vs
  only-tests-CLI-behavior. Hypothesis (validate against assertions): MUST be real =
  serve-vllm-inference, serve-lemonade-inference, serve-default-engine-inference (6b),
  serve-readiness-contract (8), serve-large-model-inference (nightly), chat-end-to-end,
  chat-tool-definitions. MOCKABLE = serve-default-engine-working-endpoint (6),
  serve-vllm-default-on-instinct (9), examine-detects-gpu-and-driver (3),
  examine-distinguishes-unmanaged-rocm (4), runtime-path-not-nested (3).
- 📋 R5 — Design a faithful mock serve engine (extends existing mock_server.rs +
  register_mock_service): must mimic serve plan / /v1/models / /v1/chat/completions so
  behavioral scenarios pass identically without a GPU. Risk: mock/real drift kills E2E
  confidence — keep a small real-serve smoke set to catch it.
- 📋 R6 — Migrate mockable scenarios off GPU (drop `@requires-gpu` → hosted/parallel/
  per-push); keep only genuine real-inference scenarios on GPU/Strix. No coverage loss.
  Depends on R4+R5.

**Less-frequent real serves — schedule (R7–R8):**
- 📋 R7 — Gate the heavy real-GPU serve matrix to `merge_group` only (not per push),
  BUT keep ONE minimal real serve on `pull_request` as a pre-merge canary (user's
  mitigation, so a broken serve is caught on the PR, not after it enters the queue).
  Prereqs: fix the Strix-Windows flake first (merge-time flakes bounce good PRs);
  verify no moved job is a required check (would stall the queue). **Overlaps Task #2**
  (tiering) — same lever, reconcile.
- 📋 R8 — Add a narrow `serve` paths-filter (engines/**, apps/rocm serve code,
  crates/rocm-core, **/*.feature, e2e-cucumber + broad-dep safety nets) so Rust-but-
  not-serve PRs (dash-only, unrelated crates) skip the GPU matrix. Today the coarse
  `heavy` filter trips the whole matrix on ANY `.rs`. Err toward inclusion.

**Dropped (user, 2026-07-17):** "serve once, assert many" (shared serve fixture) —
sacrifices scenario independence for a gain the smallest-model + mock split already
capture more cleanly. **Capacity** = user adds hardware when available (near-maxed:
2nd MI300X runner added, Strix boxes physically 1-each).

## Next Steps

- Tasks #1 (merged PR #126) + #3 (PR #127 open) done. Post-merge cleanup deferred. Remaining work = the Efficiency roadmap (R1–R8) + Task #5.
- **Decision gate R2:** confirm with user whether serve targets should be latest variant or smallest (current: lemonade `Qwen3-0.6B-GGUF` smallest; vLLM 1.5B). Settle once, apply to both.
- **Biggest structural win R4–R6:** mock/real split — classify `@requires-gpu` scenarios, build a faithful mock serve engine, migrate mockable ones off GPU.
- **Schedule R7–R8:** gate heavy serve matrix to `merge_group` + narrow serve paths-filter (prereq: fix Strix-Windows flake first).

## Checklist

- [ ] Scenarios written and reviewed before any implementation
- [ ] If this adds a user command, is there also a tool action for the agent?
- [ ] If this adds a tool action, are there tests covering LLM-facing semantics (description clarity, action disambiguation)?
- [ ] All scenarios have corresponding tests

## Blockers / Open Questions

- **Coverage gap on diagnose**: addressed in [[test-e2e-diagnose]] (PR #127) — no longer tracked here.
- **Serve model: latest vs smallest** (decision gate **R2**): suite uses `Qwen3-0.6B-GGUF` (smallest GGUF recipe) on lemonade, 1.5B on vLLM; unclear if the "latest variant" is intended. Settle once, apply to both.
- **Tiering already exists** (feeds **R7**): `@nightly` gate + `E2E_INCLUDE_NIGHTLY=1` env gate already separates heavy scenarios (27B serve, cold install) from per-PR runs. R7 verifies/tunes this alignment (merge_group gating), not build from scratch.
- **Real bugs separate**: EAI-7423 (lemonade-on-Strix-Linux serve fails) and EAI-7052 (lemonade Vulkan instability) are tracked known bugs in `expectations.toml`, separate from the VRAM-floor waste fix (Task #1).

## Notes

Related WIPs: [[test-e2e-tui-cucumber]], [[ci-manual-e2e]], [[persist-app-dev-ci-runner]]. See memory `reference_rocm_cli_e2e_cucumber` for suite tiers/tags and runner gotchas.

## Worktree Context

**Worktree directory**: `/Users/fres/Developer/rocm-cli-wt/fix-speed-up-e2e`
- Recreate with: `create_worktree.sh fix-speed-up-e2e`

## Work Log

### 2026-07-17 (idle flush) — [IDLE FLUSH 6]

**Session idle for 10 minutes, auto-flushing WIP state.**

### 2026-07-17 — Session review: verified PR #126 merge, reconciled overlaps between Task list and Efficiency roadmap

- **PR #126 verified MERGED** (2026-07-17 01:02 UTC, merge commit `e9a4b154`, main branch). Strix-Ubuntu lane 28.4m → 15.6m; VRAM-floor fix live.
- **Reconciled overlaps:** Task #2/#4 superseded by Efficiency roadmap R1–R8. Remapped Task #2 → R7+R8 (merge_group gating + paths-filter); Task #4 → R1–R3, decision gate R2 (latest vs smallest). Updated Next Steps and Blockers to use R-numbering; no duplicate tracking.
- **Task #3 split to own WIP:** [[test-e2e-diagnose]] created as dedicated file (Stage 8-awaiting-pr-approval, PR #127 open).

### 2026-07-17 — Created dedicated WIP for Task #3 (test-e2e-diagnose, PR #127)

- Created `/Users/fres/Developer/rocm-cli-progress/test-e2e-diagnose.md` (Stage `8-awaiting-pr-approval`).
- Moved Task #3 context out of [[fix-speed-up-e2e]] into dedicated branch WIP: scenarios, technical mapping, env-dependence saga, process notes.
- Trimmed parent WIP (fix-speed-up-e2e) down to `[[test-e2e-diagnose]]` pointers; Task #3 full detail now lives in its own file.

### 2026-07-17 (idle flush) — [IDLE FLUSH 5]

**Session idle for 10 minutes, auto-flushing WIP state.**

### 2026-07-17 (idle flush) — [IDLE FLUSH 4]

**Session idle for 10 minutes, auto-flushing WIP state.**

### 2026-07-17 (idle flush) — [IDLE FLUSH 3]

**Session idle for 10 minutes, auto-flushing WIP state.**

### 2026-07-17 (idle flush) — [IDLE FLUSH 2]

**Session idle for 10 minutes, auto-flushing WIP state.**

### 2026-07-17 (idle flush)

**Session idle for 10 minutes, auto-flushing WIP state.**

### 2026-07-17 — Container gate validated, Task #3 live-probed, methods saved, Task #4 + #2 scoped

- **Container gate passed clean:** clippy `-D warnings` + all unit tests on e2e-cucumber (the fix itself compiles clean).
- **Task #3 live-verified:** probed `rocm diagnose/fix` in Linux container; corrected 3 assumptions (OS-gating, env-dependent match-id, rc codes). 6 scenarios ready for implementation review. Saved rule to bdd-scenarios skill + global memory (verify against running system, not just source).
- **Task #4 + #2 scoped:** confirmed P0/P1 plan vs suite; Task #4 needs user decision (Strix Qwen variant); Task #2 verified `@nightly` infra exists, just needs alignment check.
- **Probe 29529197875 still queued:** waited ~2.5h; serial GPU box has heavy contention behind 2 PR runs. No action until completion (auto-loop every 10m).

### 2026-07-16 — Task #1 fix + container gate, Task #3 scenarios drafted + live validated, methods saved

- **Task #1 root cause + fix:** hardcoded `MIN_FREE_VRAM_MIB=150_000` (MI300X) vs Strix 62 GiB → every serve waited 2 min (~12 min wasted). Fix: scale to device total `min(150_000, total*0.9)`. Commit `122d2be` signed/off, pushed.
- **Container gate (full):** clippy + workspace tests clean under `-D warnings`; e2e-cucumber clippy passed. Mock lane reconciliation 4 xfail/0 unexpected (helper bug: removed erroneous `--tags` filter).
- **Probe run 29529197875:** Strix-Ubuntu 2-scenario dispatch fired; queued behind PR/merge runs on serial GPU box (~2h backlog). Auto-loop every 10m checks; on completion: if durations drop ~120s/serve → PR ready; else iterate.
- **Task #3 scenarios drafted + validated:** 6 BDD (diagnose/fix, GPU-independent, mock-lane). Probed live in container; corrected 3 assumptions: OS-gating, env-dependence of match/rc, recipe selection. Technical table verified. Ready for implementation review.
- **Methods to memory:** scoped dispatch (narrow rapid testing), act-don't-ask, commit/push flow (signing, sign-off, container gate, --no-verify), always-Linux-container, feedback entries.
- **Awaiting:** probe completion (still queued), user decision on Task #4 (Strix Qwen: latest or smallest).
