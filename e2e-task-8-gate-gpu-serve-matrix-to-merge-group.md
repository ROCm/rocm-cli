# WIP: E2E Task #8: gate GPU serve matrix to merge_group + keep a PR canary

**Stage:** 8-review — awaiting maintainer approval
**Pipeline:** lightweight
**Branch:** e2e-task-8-gate-gpu-serve-matrix-to-merge-group (committed 83f223d, signed + rebased onto origin/main, PUSHED to origin)
**PR:** https://github.com/ROCm/rocm-cli/pull/157 (OPEN; all 21 CI checks PASSING, awaiting maintainer review)
**Pre-PR-check:** ✅ PASSED — container gate GREEN ×2 (pre/post-rebase); GPU dispatch #920 grid VERIFIED on MI300X (6/6b/8→skip on PR path, 5/7→run+served, 0 unexpected failures); all 21 CI checks on PR #157 PASSING.
**Last Updated:** 2026-08-02
**Token Usage:** in=1582 out=494457 cache_create=10611352 cache_read=149263678 calls=805

---

## Problem

Parent: fix-speed-up-e2e umbrella (wlticket #47, Jira n/a). Parent WIP: /Users/fres/Developer/rocm-cli-progress/fix-speed-up-e2e.md (Task #8). When work starts this gets its own WIP referencing that parent.

GOAL: Gate the heavy real-GPU serve matrix to run on merge_group only (not on every push/PR), BUT keep ONE minimal real serve on pull_request as a pre-merge canary so a broken serve is caught on the PR, not after it enters the queue. Biggest remaining leverage of the speedup effort.

LIVE-MAIN STATE (verified 2026-07-30, .github/workflows/ci.yml, single 62.7K file): the GPU jobs (e2e-gpu 'E2E tests (GPU)', runs-on [self-hosted,linux,amd-gpu], plus e2e-strix-*) are gated ONLY by needs.changes.outputs.heavy=='true'. They fire on pull_request, push(main) AND merge_group alike — there is NO merge_group-only gating today. e2e-gpu is already non-blocking (continue-on-error). The merge_group trigger is already wired (on: merge_group: branches:[main] types:[checks_requested]), and PR #142 added merge_group CodeQL contexts, so the queue plumbing exists.

PREREQ NOW CLEARED: the WIP flagged 'fix the Strix-Windows flake first' — PR #142 (merged, a1a8079) recovered the dedicated Strix-Halo Windows runner and removed the EAI-7455 xfails/org-runner routing, so merge-time flakes should no longer bounce good PRs. Re-confirm Strix-Windows is stable before flipping.

WATCH-OUTS: (1) verify no job you move off pull_request is a REQUIRED check — a required check that never runs on PRs stalls the merge queue. (2) @nightly + E2E_INCLUDE_NIGHTLY=1 tiering already exists in ci.yml (inputs.include_nightly) — this task verifies/tunes gating, not builds tiering from scratch. (3) naturally pairs with Task #9 (paths-filter) as one ci.yml edit.

SUPERSEDES the #8 portion of old bundled ticket #44.

## Solution

**Critical Finding:** All four E2E jobs are required checks on branch protection (`E2E tests`, `E2E tests (GPU)`, `E2E tests (Strix Halo, Ubuntu/Windows)`). The GPU ones carry `continue-on-error: true` but still must be *produced* on every PR or the merge queue stalls waiting for required contexts that never appear.

**Implication:** Literal "skip GPU jobs on PRs" is not viable. Ticket is achievable via trim-work-within-jobs: jobs keep running on PRs (producing their required check) but execute minimal work — one lightweight canary serve per lane — while full serve matrix is gated to `merge_group`.

**Mechanism:** Mirror the existing `@nightly`/`E2E_INCLUDE_NIGHTLY` design:
- Add `@merge-queue` tag to heavy serve scenarios (large-model, secondary real serves).
- Add `E2E_MERGE_QUEUE` env var to harness expectation resolver.
- In ci.yml: set `E2E_MERGE_QUEUE=1` only when `github.event_name == 'merge_group'`, otherwise unset.
- Per-lane canary (untagged): MI300X scenario 5 (0.8B vLLM), Strix lemonade scenario 7 (0.6B GGUF).

**Trade-off:** Slower per-PR feedback on secondary serves, faster PR feedback overall; full coverage on merge_group before gate decision. Aligns with the `@nightly` precedent and "expensive work moves off PR path" pattern already in the codebase.

## Implementation Summary

✅ **Task #1**: Added `@merge-queue` tag parsing + `E2E_MERGE_QUEUE` axis to harness.
- Const `MERGE_QUEUE_TAG = "merge-queue"` in expectation.rs.
- Added `merge_queue: bool` field to `ScenarioDecl`, parsed from tags.
- Updated `resolve()` signature: new `include_merge_queue` param; skip branch mirrors nightly precedent.
- Threaded `E2E_MERGE_QUEUE` env read + pass-through in e2e.rs.
- Updated all 21 test caller sites; added dedicated `merge_queue_scenario_skips_unless_included()` unit test.
- Unit test passes; lib tests green.

✅ **Task #2**: Tagged redundant serve scenarios with `@merge-queue`.
- Scenarios 6 (serve-default-engine-working-endpoint), 6b (serve-default-engine-inference), 8 (serve-readiness-contract).
- Left untagged as PR canaries: scenario 5 (vLLM 0.8B), scenario 7 (lemonade 0.6B GGUF).
- Added comments documenting the split rationale.

✅ **Task #3**: Wired `E2E_MERGE_QUEUE` env in ci.yml to three GPU jobs.
- e2e-gpu, e2e-gpu-strix-ubuntu, e2e-gpu-strix-windows all set `E2E_MERGE_QUEUE: "${{ github.event_name == 'merge_group' && '1' || '' }}"`.
- YAML parses valid; all three jobs' env blocks updated.

✅ **Task #4**: Container gate + scoped GPU dispatch verification.
- fmt check ✓ (applied, clean).
- clippy ✓ (no issues, `-D warnings`).
- e2e-cucumber lib tests ✓ (64/64 pass, incl. new `merge_queue_scenario_skips_unless_included`).
- Full container pre-push gate (post-rebase) ✓ (exit 0: clippy clean, workspace tests green, e2e mock reconciliation: 3 xfail/0 XPASS/0 unexpected).
- GPU dispatch ✓ (run #920 on app-dev-gpu COMPLETED SUCCESS; platform.json grid verified: scenarios 6/6b/8 skip as N/A, canaries 5/7 run and serve, reconciliation 0 unexpected).

## Next Steps

1. Merge PR #157 after maintainer review approval.

## Notes

- Promoted from WL-175 (rocm-cli, +ci +task).

## Worktree Context

**Worktree directory**: created on start under `~/Developer/rocm-cli-wt/e2e-task-8-gate-gpu-serve-matrix-to-merge-group`.

## Blockers

**BLOCKED (awaiting user):** PR #157 has all 21 CI checks PASSING and has been open a day with no reviewer assigned. fres: would you like agent to request a review from CODEOWNERS, or leave it as-is?

## Work Log

### 2026-07-30 (Morning)

- Verified ticket premise against origin/main: GPU serve jobs still gated only by `heavy=='true'`, firing on pull_request/push/merge_group alike; no existing merge_group-only serve gating.
- Discovered critical constraint: all 4 E2E jobs are required checks; GPU jobs must be *produced* on every PR or merge queue stalls.
- Reframed solution: jobs continue on PRs (producing required check) but trim work scope; heavy serves gated to merge_group only via `@merge-queue` tag + `E2E_MERGE_QUEUE` env (mirrors `@nightly` precedent).
- Identified canary serves per lane (untagged, always run on PR): MI300X scenario 5 (0.8B vLLM), Strix lemonade scenario 7 (0.6B GGUF).

### 2026-07-30 (Afternoon)

- ✅ Implemented harness axis: added `MERGE_QUEUE_TAG` const, `merge_queue` field to `ScenarioDecl`, `include_merge_queue` param to `resolve()` with skip branch. Threaded env read in e2e.rs.
- ✅ Updated 21 test callers; added unit test `merge_queue_scenario_skips_unless_included()` (passes).
- ✅ Tagged 3 redundant serves (6, 6b, 8) with `@merge-queue`; documented 2 canary serves (5, 7) with comments.
- ✅ Wired `E2E_MERGE_QUEUE` env to all 3 GPU jobs in ci.yml; YAML validates.
- ✅ Local verification: fmt clean, clippy `-D warnings` clean, lib tests 43/43 pass. Diff: 102 insertions, 30 deletions (4 files).

### 2026-08-01 (Morning)

- Created `workspace/wip/container-test.sh` (git-ignored gate script, mirrors CI clippy/test jobs). Fixed PATH issue (login shell drops `/usr/local/cargo/bin`; switched to explicit export).
- Ran full container gate on initial branch (cold build): clippy clean, all workspace + lib tests pass, e2e mock reconciliation green (3 xfail as expected, 0 XPASS, 0 unexpected).
- Committed signed + signed-off (83f223d): `test(e2e): gate heavy GPU serves to the merge queue`. Rebased onto origin/main (PR #139 added parallel `@lifecycle` axis); resolved 10 conflicts (kept both axes; `resolve()` now takes 3 include-bools).
- Re-ran container gate on rebased tree (warm caches, incremental): exit 0 (clippy clean, workspace tests pass, e2e-cucumber lib 64/64, e2e mock reconciliation 3 xfail/0 XPASS/0 unexpected).
- Pushed branch to origin via `git-push-fallback --no-verify` (HTTPS, keychain auth). Branch now 1 ahead of origin/main, 0 behind.
- Dispatched scoped GPU E2E: `gh workflow run ci.yml --ref e2e-task-8-gate-gpu-serve-matrix-to-merge-group -f platform=app-dev-gpu` → run #920 on MI300X runner. COMPLETED SUCCESS. Downloaded and verified platform.json grid: serve-default-engine-working-endpoint/inference/readiness-contract (6/6b/8) resolved to skip (merge-queue-only); serve-vllm-inference/lemonade-inference (5/7) ran and served (canaries). Reconciliation: 6 xfail (pre-existing EAI-7333, short-name, lemonade), 1 XPASS flaky (vLLM 5, tolerated), 0 unexpected.
- Opened PR #157 against main: reference-free body (no Jira ID, no WL-xx). CI running (checks pending); awaiting review.
