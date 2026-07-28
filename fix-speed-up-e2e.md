<!-- This is an AI instruction file. Use this template when creating new WIP files. Fill in the placeholders. -->

# WIP: Speed up E2E test suite

**Stage:** 6-implementing — STAGED TASK, NOT done (PR #126 + #128 + #136 merged; Tasks #8–#11 remain — re-branch in place off fresh main)
**Pipeline:** standard
**Ticket:** [EAI-7484](https://amd.atlassian.net/browse/EAI-7484) — "Speed up E2E suite - mock/real split for GPU serve scenarios" (assignee Fredrik Espinoza, component rocm-cli) — tracked Tasks #5–#7 (PR #136, MERGED)
**Branch:** next chunk (Tasks #8–#11) re-branches off fresh main. Shipped: test-e2e-mock-real-split (#136), test-e2e-smallest-serve-model (#128), fix-speed-up-e2e (#126)
**Last Updated:** 2026-07-22
**Pre-PR-check:** passed — claude-opus-4.8 (reviewer agent), 2026-07-23

**Token Usage:** in=1006 out=301808 cache_create=5512288 cache_read=94907267 calls=509

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

- ✅ Diagnose/fix E2E coverage — moved to its own WIP: [[test-e2e-diagnose]] (PR #127). No longer numbered here.

### Todo 📋

One flat task list. E2E wall-clock is dominated by REAL model serving (cold weight load
+ engine startup + GPU ready), run serially on scarce hardware. Tasks #2–#7 attack that
from three angles — cheaper serves, fewer serves, less-frequent serves — ordered by
leverage. Tasks #8–#9 are smaller/independent.

**Cheaper serves — smallest model (Tasks #2–#4):**
- ✅ Task #2 — AUDIT DONE (2026-07-17). Mapping of every GPU serve scenario:
  | @id | serve step | model | engine | smallest? |
  |-----|-----------|-------|--------|-----------|
  | serve-vllm-inference (5) | host_serve_target | Qwen2.5-**1.5B** | vLLM | ❌ 0.5B avail |
  | serve-readiness-contract (8) | host_serve_target | 1.5B / 0.6B | host | ❌ vLLM side |
  | serve-default-engine-working-endpoint (6) | host_serve_target | 1.5B / 0.6B | host | ❌ vLLM side |
  | serve-default-engine-inference (6b) | host_serve_target | 1.5B / 0.6B | host | ❌ vLLM side |
  | serve-lemonade-inference (7) | setup_lemonade_model | Qwen3-0.6B-GGUF | lemonade | ✅ |
  | serve-vllm-default-on-instinct (9) | dedicated | Qwen2.5-0.5B | vLLM | ✅ |
  | serve-large-model-inference (10, nightly) | setup_large_gpu_model | Qwen3.6-27B | vLLM | ✅ intentional |
  **ONE lever:** the vLLM branch of `host_serve_target()` (serving_steps.rs:285) serves
  1.5B and feeds 4 real-serve Instinct scenarios (5, 8, 6, 6b). Code already documents
  0.5B as the smallest vLLM-preferred entry (line 458 uses it, proven to serve). Lemonade
  already at floor (0.6B); 27B nightly by design. → Task #3 = flip line 285 1.5B→0.5B.
- ✅ Task #3 — **DECISION: smallest** (user, 2026-07-17). Flipped host_serve_target vLLM
  branch 1.5B→0.5B (serving_steps.rs:285) + stale doc-comment at :695. 4-line diff,
  compiles clean. vLLM-resolution proof: scenario 9 already serves this exact 0.5B model
  and asserts vLLM selection. No CI prewarm list to update (weights lazy-download to shared
  HF cache; prewarm only installs SDK runtime). Real GPU verdict = PR CI lane.
  Branch: `test-e2e-smallest-serve-model` (off fresh main, re-branched in place).
  **SHIPPED: PR #128** (commit `7579270`, signed+signed-off). Container gate green
  (clippy -D warnings, workspace tests, e2e mock lane 4 xfail/0 unexpected). GPU CI lane
  is the real verdict for the model swap.
- ✅ Task #4 — SHIPPED to PR #128 (commit `f326e93`). Added a MODEL-SIZE POLICY
  doc-comment on host_serve_target: smallest model that satisfies the assertion; floors
  vLLM 0.5B / lemonade 0.6B; large-model = `@nightly` only. Gate green (clippy + ws tests).

- ✅ **Tasks #2–#4 SHIPPED: PR #128 MERGED** into main (squash `73e0fd1`, 2026-07-20
  ~15:58 CEST). Final landed model = **Qwen3.5-0.8B** on the vLLM `host_serve_target`
  path (evolved 1.5B→0.5B→scoped→0.8B across review). rominf APPROVED; volen-silo's
  CHANGES_REQUESTED dismissed. All blocking checks green; only non-blocking
  `E2E tests (GPU)` red (pre-existing continue-on-error lane). Also merged around it:
  PR #127 (diagnose, squash `8f67d4a`) and the GPU-required probe PR #121.

**Fewer real serves — mock/real split (Tasks #5–#7, biggest structural win) — [EAI-7484], component rocm-cli, one PR on `test-e2e-mock-real-split`:**
- ✅ Task #5 — DONE (2026-07-21). Classified all 14 GPU-tagged scenarios vs their actual
  assertions (not the hypothesis). GPU tags in use: `@requires-gpu` (primary), co-tags
  `@nightly`, `@requires-engine:vllm|lemonade`, `@requires-os:linux`, `@serve-timeout:2400`.
  **MUST-BE-REAL (8):** model_serving #5 (vLLM inference, model_ids_match), #7 (lemonade
  inference), #6b (default-engine inference), #8 (readiness contract EAI-7333), #10 (27B
  nightly), examine #3 (real gfx/amdgpu probe — no serve engine), runtime #1 (real SDK
  install, nightly), runtime #3 (real install folder layout). Note examine#3 + runtime#1/#3
  need real HARDWARE/INSTALL but exercise NO serve engine — a mock serve can't help them.
  **MOCKABLE (6):** chat #4 (tools accepted — asserts only valid choices array), chat #5
  (end-to-end — asserts only non-empty content, not model_ids_match; `given` already uses
  MockServer), model_serving #9 (vLLM-default — plan-only, comment says so), #12 + #13
  (GPU-masked / bad-index refusals — assert rc!=0 + stderr, refuse before engine start),
  examine #4 (planted unmanaged-ROCm fixture, classification+guidance only — tag over-broad).
  **HYPOTHESIS CORRECTIONS:** chat #4/#5 are MOCKABLE (hypothesis wrongly had them real);
  #12/#13 newly found mockable; examine#4 confirmed mockable.
  **Borderline:** model_serving #6 (working-endpoint) — engine-selection half is plan-only
  but "model reachable" hits real /v1/models after a real managed serve. Kept MUST-BE-REAL
  on the reachability assertion; strongest demotion candidate if intent is selection-only.
- 📋 Task #6 — Design a faithful mock serve engine (extends existing mock_server.rs +
  register_mock_service): must mimic serve plan / /v1/models / /v1/chat/completions so
  behavioral scenarios pass identically without a GPU. Risk: mock/real drift kills E2E
  confidence — keep a small real-serve smoke set to catch it.
  **REVISED after #6 investigation (2026-07-21):** lane routing is CAPABILITY-driven, not
  tag-filtered — `@requires-gpu` = skip if no AMD GPU; `@requires-no-gpu` = skip if GPU
  present (runs ONLY on the GitHub-hosted mock lane, e.g. scenario 11 today). Migration =
  RETAG, not a CI change. MockServer already serves /v1/models + /v1/chat/completions and
  echoes the requested model (so model_ids_match works); write_service_record already
  registers a managed/ready vllm record. Split of the 5 candidates:
  - #12, #13 (masked-GPU / bad-index refusal) → NO mock server needed; assert rc!=0 +
    pre-flight message. Just retag `@requires-gpu @requires-os:linux` → `@requires-no-gpu`.
    Scenario 11 already proves this message fires on the no-GPU lane.
  - #4, #5 (need a served endpoint) → point the "served in background" Given at MockServer
    + register_mock_service instead of real `rocm serve`; drop `@requires-gpu`.
  - #9 KEPT ON GPU (user decision 2026-07-21): asserts a REAL launch (rc==0 + `engine: vllm`
    on real serve stdout), keyed on seeing a real Instinct GPU — mocking would assert against
    our own output, erasing its value. Excluded from migration.
  **XPASS trap:** any migrated scenario that starts passing XPASSes its
  `effective_engine=="vllm"` xfail entry (e.g. chat-tool-definitions-accepted EAI-7223,
  expectations.toml:128) → must remove those entries or the run fails on XPASS.
- 📋 Task #7 — Migrate mockable scenarios off GPU (#4, #5, #12, #13; #9 excluded per above).
  Retag + wire #4/#5 to MockServer; scrub stale vllm xfail entries. No coverage loss.
- ✅ **Tasks #5–#7 SHIPPED: PR #136 MERGED** (squash `cae7781` on main, 2026-07-22
  18:13 CEST), [EAI-7484]. rominf APPROVED. Rebased onto fresh main before merge (resolved
  chat.feature renumber-vs-retag conflict from PR #114's TUI privacy-notice rewrite;
  container gate re-green 3 xfail/0 XPASS/0 unexpected). Landed via merge queue.
  FINAL migration set = **chat #4 + #5 only**. setup_background_model made capability-aware
  (real serve on GPU host, MockServer + register_mock_service on no-GPU). `a managed runtime
  is active` no-ops on no-GPU (SDK install needs a GPU family). EAI-7423 lemonade xfails for
  these two ids scoped `therock_family="gfx*"` so they expect-pass on the mock lane (glob
  "gfx*" vs empty gfx_target = no match). Gate: clippy -D warnings clean, 25 ws/lib tests,
  e2e mock lane 4 xfail/0 XPASS/0 unexpected; #4/#5 PASS on no-GPU lane.
  **#13 NOT migratable — proven by running system (not source):** with `--gpu 99` on a
  no-GPU host, the GPU-required pre-flight refuses with "no usable AMD GPU" BEFORE validating
  the index, so the index-specific message ("99"+"out of range"/"not available") never
  appears → assertion fails. Reverted #13 to `@requires-gpu @requires-os:linux`, kept an
  improved comment documenting why. #12 excluded earlier (would duplicate scenario 11 on the
  mock lane). #9 excluded (user: keep real launch+selection on GPU).

**Less-frequent real serves — schedule (Tasks #8–#9):**
- 📋 Task #8 — Gate the heavy real-GPU serve matrix to `merge_group` only (not per push),
  BUT keep ONE minimal real serve on `pull_request` as a pre-merge canary (so a broken
  serve is caught on the PR, not after it enters the queue). Prereqs: fix the
  Strix-Windows flake first (merge-time flakes bounce good PRs); verify no moved job is a
  required check (would stall the queue). Note: `@nightly` + `E2E_INCLUDE_NIGHTLY=1`
  tiering already exists — this verifies/tunes it, not build from scratch.
- 📋 Task #9 — Add a narrow `serve` paths-filter (engines/**, apps/rocm serve code,
  crates/rocm-core, **/*.feature, e2e-cucumber + broad-dep safety nets) so Rust-but-
  not-serve PRs (dash-only, unrelated crates) skip the GPU matrix. Today the coarse
  `heavy` filter trips the whole matrix on ANY `.rs`. Err toward inclusion.

**Smaller / independent (Tasks #10–#11):**
- 📋 Task #10 — Reduce mock lane per-scenario overhead (fixed overhead ~4.8s/scenario,
  multiply across 12). Distinct from Tasks #5–#7 (which move scenarios *off* GPU, not the
  mock-lane fixed cost).
- 📋 Task #11 — FILE separately: `fix fix-2-unset-override --dry-run` panics rc=101 (a
  dry-run should never panic). Not a speedup; a correctness bug found while probing.

**Dropped (user, 2026-07-17):** "serve once, assert many" (shared serve fixture) —
sacrifices scenario independence for a gain the smallest-model + mock split already
capture more cleanly. **Capacity** = user adds hardware when available (near-maxed:
2nd MI300X runner added, Strix boxes physically 1-each).

## Next Steps

- **PR #128 (Tasks #2–#4) MERGED** (squash `73e0fd1`, 2026-07-20). Tasks #1–#4 all done.
- **Re-branch in place** for the next chunk (Tasks #5–#7): `git checkout main && git pull &&
  git checkout -b <task5-7-branch>` off fresh main, update Branch field. Do NOT keep
  committing on the merged `test-e2e-smallest-serve-model`.
- **Biggest structural win Tasks #5–#7:** mock/real split — classify `@requires-gpu` scenarios, build a faithful mock serve engine, migrate mockable ones off GPU.
- **Schedule Tasks #8–#9:** gate heavy serve matrix to `merge_group` + narrow serve paths-filter (prereq: fix Strix-Windows flake first).
- Task #11 (dry-run panic) still to be filed separately.

## Checklist

- [ ] Scenarios written and reviewed before any implementation
- [ ] If this adds a user command, is there also a tool action for the agent?
- [ ] If this adds a tool action, are there tests covering LLM-facing semantics (description clarity, action disambiguation)?
- [ ] All scenarios have corresponding tests

## Blockers / Open Questions

- **Coverage gap on diagnose**: addressed in [[test-e2e-diagnose]] (PR #127) — no longer tracked here.
- **Serve model: latest vs smallest** (decision gate **Task #3**): suite uses `Qwen3-0.6B-GGUF` (smallest GGUF recipe) on lemonade, 1.5B on vLLM; unclear if the "latest variant" is intended. Settle once, apply to both.
- **Tiering already exists** (feeds **Task #8**): `@nightly` gate + `E2E_INCLUDE_NIGHTLY=1` env gate already separates heavy scenarios (27B serve, cold install) from per-PR runs. Task #8 verifies/tunes this alignment (merge_group gating), not build from scratch.
- **Real bugs separate**: EAI-7423 (lemonade-on-Strix-Linux serve fails) and EAI-7052 (lemonade Vulkan instability) are tracked known bugs in `expectations.toml`, separate from the VRAM-floor waste fix (Task #1).

## Notes

Related WIPs: [[test-e2e-tui-cucumber]], [[ci-manual-e2e]], [[persist-app-dev-ci-runner]]. See memory `reference_rocm_cli_e2e_cucumber` for suite tiers/tags and runner gotchas.

## Worktree Context

**Worktree directory**: `/Users/fres/Developer/rocm-cli-wt/fix-speed-up-e2e`
- Recreate with: `create_worktree.sh fix-speed-up-e2e`

## Work Log

### 2026-07-22 — PR #136 MERGED (Tasks #5–#7); rebase-conflict resolved, GPU red = EAI-7533

- **Merged** squash `cae7781` on main (18:13 CEST) via merge queue. rominf APPROVED; no
  inline/issue/bot comments (all 3 review surfaces checked).
- **chat #5 cold-cache flake, NOT a regression:** first GPU-lane run flagged
  `chat-end-to-end-local-model` FAIL — but it was the cold HF weight download blowing the
  300s readiness window (warm re-serve of the same model = ~83s XPASS). Re-ran GPU lane →
  chat #5 PASS, 0 unexpected failures. Confirms migration only touches the no-GPU path.
- **Rebase onto fresh main:** branch had conflicted (main advanced 9 commits). Only real
  conflict = `chat.feature`: PR #114 (`cfae8d3`) rewrote the privacy-notice scenario into a
  TUI/pty version + renumbered; my commit only drops `@requires-gpu` from chat #4/#5.
  Resolved by keeping main's numbering/TUI content and applying only my two tag-drops +
  rationale comments (now scenarios 5 & 6). Other 3 files auto-merged clean.
- **Container gate re-green** post-rebase: clippy -D warnings + 25 ws tests + e2e mock lane
  `31 scenarios 28 passed / 3 xfail / 0 XPASS / 0 unexpected`. Force-pushed rebased `0c2c5c0`
  via git-push-fallback --no-verify (macOS pre-push fails on 3 pre-existing `managed_stop_*`
  #[cfg(unix)] tests in apps/rocm — unrelated; diff is tests/e2e-cucumber only).
- **GPU lane red is EAI-7533, not #136:** the 4 XPASS (EAI-7333 ×2, EAI-7052 ×2) are stale
  xfail rows for bugs now fixed on the Instinct host — reproduced on main + PR #134,
  non-required (continue-on-error). Filed as [EAI-7533](https://amd.atlassian.net/browse/EAI-7533).
- **Next:** re-branch in place off fresh main for Tasks #8–#11. Also on main now: PR #142
  (`a1a8079` ci: stabilize GPU E2E and merge queue) — check whether it already addresses
  Task #8 (merge_group gating) before starting.

### 2026-07-21 — Tasks #5–#7 shipped to PR #136; created EAI-7484

- **Jira:** created EAI-7484 (Task, standalone, component rocm-cli via REST — acli create
  has no --component flag; PUT /issue with components add id 31850, HTTP 204, verified).
- **Task #6 impl:** setup_background_model capability-aware (real serve on GPU host,
  MockServer + register_mock_service on no-GPU). setup_active_runtime no-ops on no-GPU.
- **Task #7:** dropped @requires-gpu from chat #4/#5; scoped EAI-7423 lemonade xfails to
  therock_family="gfx*" (mock lane has empty gfx_target → no match → expect-pass).
- **#13 experiment FAILED (kept the finding):** container e2e run showed serve-absent-gpu-
  index-rejected fails on the no-GPU lane — GPU-required pre-flight fires "no usable AMD GPU"
  before the --gpu index is validated, so the index-specific assertion never matches.
  Reverted #13 tags, kept improved comment. LESSON: verify migrations against the RUNNING
  system — source review predicted #13 was mockable; the live run disproved it.
- **Gate green** (clippy + 25 ws tests + e2e mock 4xfail/0XPASS/0unexpected). Commit `477510d`
  signed (RSA, 1Password) + DCO. Pushed --no-verify (macOS pre-push failed on 4 pre-existing
  `managed_stop_*` #[cfg(unix)] process-identity tests in apps/rocm — unrelated to my diff,
  which touches only tests/e2e-cucumber; AGENTS.md §6 macOS unsupported). **PR #136 open.**
- **Next:** watch PR #136 CI to green (esp. the mock `e2e` lane + GPU lanes confirm #4/#5
  still real-serve on GPU). Then Tasks #8–#11.

### 2026-07-21 — Re-branched for Tasks #5–#7; Task #5 audit DONE

- **Re-branched in place:** `test-e2e-mock-real-split` reset onto fresh `origin/main`
  (`3aa64e7`; had to fetch+reset — `main` is checked out in the primary worktree so a
  plain `git checkout main` failed). Old `test-e2e-smallest-serve-model` is merged/stale.
- **Task #5 DONE (subagent audit vs live assertions):** 14 GPU-tagged scenarios → 8
  MUST-BE-REAL, 6 MOCKABLE. Hypothesis was WRONG on chat #4/#5 (actually mockable — assert
  only success/non-empty, not model_ids_match) and MISSED #12/#13 (masked-GPU refusals,
  mockable). Borderline = model_serving #6. examine#3 + runtime#1/#3 need real hardware but
  no serve engine (mock serve irrelevant). Full table in Task #5 above.
- **Next:** Task #6 — design faithful mock serve engine extending mock_server.rs.

### 2026-07-21 — PR #128 verified MERGED; Tasks #2–#4 done, re-branch pending for #5–#11

- **Live check (git fetch + gh):** PR #128 state MERGED, squash `73e0fd1` on main
  (2026-07-20 15:58 CEST). Final landed subject "serve Qwen3.5 0.8B on shared vLLM path":
  the model evolved past my local `4cd2c53` (1.5B→0.5B→scoped→**0.8B**) via extra commits
  I don't have locally. rominf APPROVED, volen-silo DISMISSED. Blocking checks all green;
  only `E2E tests (GPU)` red = pre-existing non-blocking continue-on-error lane.
- **Also confirmed merged:** PR #127 diagnose (squash `8f67d4a`), GPU-required probe #121.
- **Local branch `test-e2e-smallest-serve-model` HEAD `4cd2c53` is now stale** (behind the
  merged squash). Next actionable = re-branch off fresh main for Tasks #5–#7 (mock/real
  split), per staged-task protocol. Marked Tasks #2–#4 ✅ SHIPPED in the WIP.

### 2026-07-17 — PR #128 review (volen-silo, CHANGES_REQUESTED): scoped the swap

- **Review was CORRECT — verified vs expectations.toml + catalog, not trusted blind.** `host_serve_target().0` also feeds the default-engine step (serving_steps.rs:435, no `--engine`), where the model's own `preferred_engines` drives resolution. 1.5B (lemonade-pref) → 0.5B (vLLM-pref) flipped default-engine resolution on Instinct lemonade→vLLM, invalidating the EAI-7052 xfail for serve-default-engine-working-endpoint/-inference → XPASS → reconciliation exit 1. expectations.toml:55-63 states the dependency.
- **Fix = option (a) scope the swap** (`4cd2c53`): new `default_engine_serve_target()` stays lemonade-preferred (1.5B Instinct / 0.6B lemonade hosts) for the default-engine step; 0.5B size cut kept ONLY on explicit `--engine vllm` path (scenarios 5, 8). Default-engine resolution now byte-identical to main → xfail matrix untouched, no GPU re-verify needed. Rejected option (b) (embrace default-vLLM + rewrite matrix) as scope creep beyond "smallest model".
- Gate green (clippy + 23 ws tests), pushed, replied to review (issue-comment 5002733399). LESSON: a shared test helper can feed both explicit-engine AND engine-resolving call sites — a model swap is not always behavior-neutral.

### 2026-07-17 — Tasks #2+#3 shipped to PR #128 (smallest vLLM serve model)

- **Task #2 audit:** mapped all 7 GPU serve scenarios → only lever is host_serve_target vLLM branch (1.5B).
- **Task #3 (decision: smallest, user):** flipped serving_steps.rs:285 1.5B→0.5B + stale doc-comment :695. 4-line diff.
- **Re-branched in place** (merged fix-speed-up-e2e → new test-e2e-smallest-serve-model off fresh origin/main; picked up #90/#94/#116). Saved memory `staged-task-rebranch-on-merged`.
- **Container gate green**, committed signed `7579270`, pushed, **PR #128 open**. Awaiting GPU CI lane (real verdict for the swap).

### 2026-07-17 — Unified numbering: one flat task scheme (dropped R-prefix)

- Collapsed the separate "Efficiency roadmap R1–R8" + Todo list into ONE flat task list. Dropped the "R" namespace (it only meant "Roadmap"). Everything is now Task #N.
- Renumbering: R1–R8 → Tasks #2–#9; old Task #5 (mock-lane overhead) → Task #10; panic-bug filing → Task #11. Task #1 (VRAM, merged PR #126) keeps its number. Diagnose (old Task #3) exits this file's numbering — lives in [[test-e2e-diagnose]].
- Decision gate is now **Task #3** (was R2): latest vs smallest serve model.
- Updated Stage line, Next Steps, and Blockers to the new numbers.

### 2026-07-17 — Session: WIP restructure, overlap reconciliation, staged-task framing, awaiting decision gate

- **Moved Task #3 to own WIP:** Created [[test-e2e-diagnose]].md (PR #127 open, Stage 8-awaiting-pr-approval); trimmed parent WIP to pointer.
- **Reconciled roadmap overlaps:** Task #2/#4 superseded by Efficiency roadmap R1–R8. Remapped Task #2→R7+R8, Task #4→R1–R3 (decision gate R2). One home per piece of work.
- **Fixed done-detection:** Updated Stage line to flag "STAGED TASK, NOT done"; PR #126 shipped as milestone; re-branch in place for next chunk per wip-management skill protocol.
- **Next actionable:** R2 decision gate (latest vs smallest serve model); then R4–R6 (mock/real split, biggest structural win).

### 2026-07-17 (idle flush) — [IDLE FLUSH 7]

**Session idle for 10 minutes, auto-flushing WIP state.**

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
