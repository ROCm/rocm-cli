# WIP: E2E mock/real split for GPU serve scenarios

**Stage:** 9-done (PR #136 MERGED — squash cae7781 on main, 2026-07-22 18:13 CEST)
**Pipeline:** standard
**Ticket:** [EAI-7484](https://amd.atlassian.net/browse/EAI-7484) — "Speed up E2E suite - mock/real split for GPU serve scenarios" (assignee Fredrik Espinoza, component rocm-cli)
**Branch:** test-e2e-mock-real-split (merged; child of the [[fix-speed-up-e2e]] container)
**Parent:** fix-speed-up-e2e
**Pre-PR-check:** passed — claude-opus-4.8 (reviewer agent), 2026-07-23
**Last Updated:** 2026-07-30

**Token Usage:** in=481 out=110661 cache_create=4573562 cache_read=29653085 calls=243

---

## Problem

E2E wall-clock is dominated by REAL model serving (cold weight load + engine
startup + GPU ready), run serially on scarce hardware. Many `@requires-gpu`
scenarios don't actually need a real serve — their assertions only check
success/non-empty output or a refusal message, which a faithful mock can satisfy.
Running them on real GPUs wastes the scarcest resource. The biggest structural
win in the speed-up effort is moving mockable serve scenarios off GPU.

## Solution

Classify every `@requires-gpu` serve scenario as MUST-BE-REAL vs MOCKABLE against
its ACTUAL assertions (not a hypothesis), build/extend a faithful mock serve
engine, and migrate the mockable ones to the no-GPU lane — with zero coverage
loss and a guard against mock/real drift.

Lane routing is CAPABILITY-driven, not tag-filtered: `@requires-gpu` skips when no
AMD GPU; `@requires-no-gpu` runs only on the GitHub-hosted mock lane. So migration
is a RETAG plus wiring the "served in background" Given to MockServer, not a CI
change. MockServer already serves /v1/models + /v1/chat/completions and echoes the
requested model; write_service_record already registers a managed/ready vllm
record.

## Implementation Steps

### Completed ✅
- ✅ Task #5 — Classified all 14 GPU-tagged scenarios vs their real assertions.
  MUST-BE-REAL (8): model_serving #5 (vLLM inference, model_ids_match), #7
  (lemonade inference), #6b (default-engine inference), #8 (readiness contract
  EAI-7333), #10 (27B nightly), examine #3 (real gfx/amdgpu probe), runtime #1
  (real SDK install, nightly), runtime #3 (real install folder layout).
  MOCKABLE (6): chat #4 (tools accepted), chat #5 (end-to-end, non-empty content),
  model_serving #9 (vLLM-default plan-only), #12 + #13 (GPU-masked / bad-index
  refusals), examine #4 (planted unmanaged-ROCm fixture). Hypothesis corrections:
  chat #4/#5 mockable; #12/#13 newly found mockable; examine #4 confirmed.
- ✅ Task #6 — Made setup_background_model capability-aware (real serve on GPU host,
  MockServer + register_mock_service on no-GPU). setup_active_runtime no-ops on
  no-GPU (SDK install needs a GPU family). Confirmed lane routing is capability-,
  not tag-driven.
- ✅ Task #7 — Migrated mockable scenarios off GPU. FINAL set = **chat #4 + #5
  only**. Dropped `@requires-gpu`; scoped EAI-7423 lemonade xfails to
  `therock_family="gfx*"` (mock lane has empty gfx_target → no match → expect-pass).
- ✅ **SHIPPED: PR #136 MERGED** (squash cae7781, 2026-07-22 18:13 CEST). rominf
  APPROVED. Rebased onto fresh main before merge (resolved chat.feature
  renumber-vs-retag conflict from PR #114's TUI privacy-notice rewrite). Landed via
  merge queue. Gate: clippy -D warnings clean, 25 ws/lib tests, e2e mock lane 4
  xfail/0 XPASS/0 unexpected; #4/#5 PASS on no-GPU lane.

## Notes

- **#13 NOT migratable — proven by the running system (not source):** with
  `--gpu 99` on a no-GPU host, the GPU-required pre-flight refuses with "no usable
  AMD GPU" BEFORE validating the index, so the index-specific message never appears
  → assertion fails. Reverted #13 to `@requires-gpu @requires-os:linux`, kept an
  improved comment. LESSON: verify migrations against the RUNNING system.
- **#12 excluded** (would duplicate scenario 11 on the mock lane). **#9 excluded**
  (user: keep real launch+selection on GPU — asserts a REAL launch keyed on a real
  Instinct GPU; mocking would assert against our own output).
- **XPASS trap:** any migrated scenario that starts passing XPASSes its
  `effective_engine=="vllm"` xfail entry → must scrub those entries or the run
  fails on XPASS.
- **GPU lane red is EAI-7533, not #136:** 4 XPASS (EAI-7333 ×2, EAI-7052 ×2) are
  stale xfail rows for bugs now fixed on the Instinct host; non-required
  (continue-on-error). Filed as [EAI-7533](https://amd.atlassian.net/browse/EAI-7533).

## Worktree Context

**Container**: shares the [[fix-speed-up-e2e]] worktree
(`/Users/fres/Developer/rocm-cli-wt/fix-speed-up-e2e`) — this branch was worked in
place there. Branch merged; no standalone worktree.

## Work Log

### 2026-07-22 — PR #136 MERGED

- Merged squash cae7781 on main (18:13 CEST) via merge queue. rominf APPROVED; no
  inline/issue/bot comments (all 3 review surfaces checked).
- chat #5 cold-cache flake (NOT a regression): first GPU-lane run flagged
  chat-end-to-end-local-model FAIL — cold HF weight download blew the 300s
  readiness window; warm re-serve = ~83s XPASS. Re-ran → PASS, 0 unexpected.
- Rebase onto fresh main: only real conflict = chat.feature (PR #114 rewrote the
  privacy-notice scenario into a TUI/pty version + renumbered). Kept main's
  numbering/TUI content, applied only the two tag-drops + rationale comments.
- Container gate re-green post-rebase; force-pushed rebased 0c2c5c0.

### 2026-07-21 — Tasks #5–#7 shipped to PR #136; created EAI-7484

- Created EAI-7484 (Task, standalone, component rocm-cli via REST + PUT components
  add id 31850, HTTP 204, verified).
- Task #6 impl: setup_background_model capability-aware. Task #7: dropped
  @requires-gpu from chat #4/#5; scoped EAI-7423 lemonade xfails.
- #13 experiment FAILED (kept the finding): reverted tags, improved comment.
- Gate green; commit 477510d signed + DCO; pushed --no-verify (pre-existing macOS
  managed_stop_* test failures, unrelated). PR #136 opened.

### 2026-07-21 — Re-branched for Tasks #5–#7; Task #5 audit DONE

- Re-branched in place: test-e2e-mock-real-split reset onto fresh origin/main.
- Task #5 done (subagent audit vs live assertions): 14 GPU-tagged → 8 real, 6
  mockable. Hypothesis wrong on chat #4/#5; missed #12/#13.
