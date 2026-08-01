# WIP: E2E Task #8: gate GPU serve matrix to merge_group + keep a PR canary

**Stage:** 2-implementing
**Pipeline:** lightweight
**Branch:** e2e-task-8-gate-gpu-serve-matrix-to-merge-group (uncommitted, code + ci.yml + features ready)
**Pre-PR-check:** pending (container gate interrupted; needs restart + commit)
**Last Updated:** 2026-08-01 (morning)
**Token Usage:** in=1278 out=386214 cache_create=6125235 cache_read=115048755 calls=645

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
- e2e-cucumber lib tests ✓ (43/43 pass, incl. new `merge_queue_scenario_skips_unless_included`).
- Full container pre-push gate: started (dependencies downloading); interrupted before completion.
- GPU dispatch: not yet attempted.

## Next Steps

1. ⏸ Restart Apple container, run full Linux container gate locally (clippy + cargo test --workspace + lib tests + e2e mock).
2. Run scoped GPU dispatch on app-dev-gpu (manual workflow: `--ref e2e-task-8-gate-gpu-serve-matrix-to-merge-group -f platform=app-dev-gpu`).
3. Verify: PR event skips 6/6b/8, keeps 5/7; merge_group runs all. Confirm no required checks are removed.
4. Open PR against main.

## Notes

- Promoted from WL-175 (rocm-cli, +ci +task).

## Worktree Context

**Worktree directory**: created on start under `~/Developer/rocm-cli-wt/e2e-task-8-gate-gpu-serve-matrix-to-merge-group`.

## Blockers

**BLOCKED (awaiting user):** Full container gate backgrounded; running cold build (estimated 5–10 min). After completion: verify log for green `clippy + cargo test + e2e-cucumber lib + xtask e2e` results. Then run scoped GPU dispatch on app-dev-gpu (manual workflow: `--ref e2e-task-8-gate-gpu-serve-matrix-to-merge-group -f platform=app-dev-gpu`) to verify merge_group env branching. After both gates green: commit (no refs) + push to origin.

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
- Started full container gate (cold build ~5+ min); backgrounded to avoid blocking. Awaiting completion + GPU dispatch before commit/push.
- Code changes complete: 4 files modified (102 insertions, 30 deletions); locally verified via `cargo test -p e2e-cucumber --lib` + mac-local clippy (43/43 tests pass, clippy `-D warnings` clean).
