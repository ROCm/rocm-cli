# WIP: E2E Task #8: gate GPU serve matrix to merge_group + keep a PR canary

**Stage:** 1-scoping
**Pipeline:** lightweight
**Branch:** e2e-task-8-gate-gpu-serve-matrix-to-merge-group
**Pre-PR-check:** none
**Last Updated:** 2026-07-30
**Token Usage:** in=110 out=42429 cache_create=637002 cache_read=5426906 calls=55

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

## Next Steps

1. Confirm user agrees with trim-work approach and canary selection (awaiting decision).
2. If approved: design detailed ci.yml + harness changes, identify affected serve scenarios.
3. Implement: edit ci.yml (env vars, conditional logic), annotate .feature files, update expectation.rs resolver.
4. Test on mock + dispatch.
5. Open PR.

## Notes

- Promoted from WL-175 (rocm-cli, +ci +task).

## Worktree Context

**Worktree directory**: created on start under `~/Developer/rocm-cli-wt/e2e-task-8-gate-gpu-serve-matrix-to-merge-group`.

## Blockers

**BLOCKED (awaiting user):** Confirm approach (trim-work-within-jobs via `@merge-queue` tag + `E2E_MERGE_QUEUE` env) and canary selection (cheapest serve per lane: MI300X scenario 5, Strix scenario 7) before detailed design.

## Work Log

### 2026-07-30

- Verified ticket premise against origin/main: GPU serve jobs still gated only by `heavy=='true'`, firing on pull_request/push/merge_group alike; no existing merge_group-only serve gating.
- Discovered critical constraint: all 4 E2E jobs are required checks; GPU jobs must be *produced* on every PR or merge queue stalls.
- Reframed solution: jobs continue on PRs (producing required check) but trim work scope; heavy serves gated to merge_group only via `@merge-queue` tag + `E2E_MERGE_QUEUE` env (mirrors `@nightly` precedent).
- Identified canary serves per lane (untagged, always run on PR): MI300X scenario 5 (0.8B vLLM), Strix lemonade scenario 7 (0.6B GGUF).
