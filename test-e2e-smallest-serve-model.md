# WIP: E2E smallest serve model (cheaper serves)

**Stage:** 9-done (PR #128 MERGED — squash 73e0fd1 on main, 2026-07-20 15:58 CEST)
**Pipeline:** standard
**Ticket:** none — shipped without a dedicated Jira ticket (part of the speed-up effort).
**Branch:** test-e2e-smallest-serve-model (merged; child of the [[fix-speed-up-e2e]] container)
**Pre-PR-check:** passed
**Last Updated:** 2026-07-20

**Token Usage:** in=0 out=0 cache_create=0 cache_read=0 calls=0

---

## Problem

E2E serve scenarios were serving larger models than their assertions require (vLLM
`host_serve_target` served 1.5B), wasting cold-load + GPU time on the scarcest
resource. Cheaper (smaller) models that still satisfy the assertions cut per-serve
time.

## Solution

Serve the smallest model that satisfies each assertion: floor the vLLM
`host_serve_target` path and document a MODEL-SIZE POLICY. Keep engine resolution
byte-identical where a shared helper also feeds engine-resolving call sites (a
model swap is not always behavior-neutral).

## Implementation Steps

### Completed ✅
- ✅ Task #2 — AUDIT: mapped every GPU serve scenario. Only lever = the vLLM branch
  of `host_serve_target()` (serving_steps.rs:285), which served 1.5B and fed 4
  real-serve Instinct scenarios. Lemonade already at floor (0.6B); 27B nightly by
  design.
- ✅ Task #3 — DECISION: smallest (user, 2026-07-17). Flipped host_serve_target
  vLLM branch 1.5B→0.5B + stale doc-comment. vLLM-resolution proof: scenario 9
  already serves this exact 0.5B model and asserts vLLM selection.
- ✅ Task #4 — Added a MODEL-SIZE POLICY doc-comment on host_serve_target:
  smallest model that satisfies the assertion; floors vLLM 0.5B / lemonade 0.6B;
  large-model = `@nightly` only.
- ✅ **SHIPPED: PR #128 MERGED** (squash 73e0fd1, 2026-07-20 15:58 CEST). Final
  landed model = **Qwen3.5-0.8B** on the vLLM host_serve_target path (evolved
  1.5B→0.5B→scoped→0.8B across review). rominf APPROVED; volen-silo's
  CHANGES_REQUESTED dismissed. Only non-blocking `E2E tests (GPU)` red (pre-existing
  continue-on-error lane).

## Notes

- **Review saga (volen-silo CHANGES_REQUESTED, then scoped):** `host_serve_target().0`
  also feeds the default-engine step (no `--engine`), where the model's own
  `preferred_engines` drives resolution. 1.5B (lemonade-pref) → 0.5B (vLLM-pref)
  flipped default-engine resolution on Instinct lemonade→vLLM, invalidating the
  EAI-7052 xfail → XPASS → reconciliation exit 1. Fix: scoped the swap — a new
  `default_engine_serve_target()` stays lemonade-preferred; the 0.5B cut kept ONLY
  on the explicit `--engine vllm` path. LESSON: a shared test helper can feed both
  explicit-engine AND engine-resolving call sites.

## Worktree Context

**Container**: shares the [[fix-speed-up-e2e]] worktree
(`/Users/fres/Developer/rocm-cli-wt/fix-speed-up-e2e`) — worked in place there.
Branch merged; no standalone worktree.

## Work Log

### 2026-07-20 — PR #128 verified MERGED

- Live check: PR #128 MERGED, squash 73e0fd1 on main. Final subject "serve Qwen3.5
  0.8B on shared vLLM path"; model evolved past the local 1.5B→0.5B via review.
  rominf APPROVED, volen-silo DISMISSED.

### 2026-07-17 — scoped the swap after review; shipped

- volen-silo CHANGES_REQUESTED was CORRECT (verified vs expectations.toml). Fixed
  by scoping the swap to the explicit-vLLM path only. Gate green; PR #128 open.
- Task #2 audit + Task #3 flip (1.5B→0.5B) + Task #4 policy doc-comment. Container
  gate green; committed signed 7579270.
