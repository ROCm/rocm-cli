# WIP: Make XPASS non-fatal for known-flaky xfails (flaky marker)

**Stage:** 9-done (PR #142 merged 2026-07-27 with identical feature; #138 closed as superseded)
**Pipeline:** standard
**Branch:** fix-xpass-non-fatal-flaky (worktree archived)
**PR:** #138 — **CLOSED** (superseded by #142 merged 2026-07-27). Feature shipped via #142 on main.
**Pre-PR-check:** passed — opencode (independent reviewer), 2026-07-22
**Jira:** EAI-7456 (**Done**, 2026-07-28) — commented referencing PR #142 as delivery; #138 closed as redundant.
**Last Updated:** 2026-07-28

**Token Usage:** in=2062 out=481977 cache_create=15115072 cache_read=159467295 calls=981

---

## CI status — ALL REQUIRED CHECKS GREEN (run 29910874741, commit 3dd423a)
- ✅ All 8 merge-required checks pass: Coverage, License header, clippy, build-and-test, **windows-build-and-test**, **Commit signatures + sign-off** (DCO fix worked), Lint (PowerShell), prek.
- PR state: `MERGEABLE` but `BLOCKED` on `REVIEW_REQUIRED` — needs a human approving review. Nothing else gates merge.
- **Non-required E2E lanes are advisory** (NOT in branch protection): Strix-Windows queues on an offline runner; the GPU lane may still show the orthogonal EAI-7052 XPASS / chat cold-cache FAIL. Neither blocks merge.
- **Concurrency-group finding (for later, separate from #138):** `ci.yml` uses a shared per-ref group with `cancel-in-progress: true`. Supersession can't complete when the superseded run has a job queued on an OFFLINE self-hosted runner (GitHub can't cancel it) → it holds the group and stalls the new run at `pending`/0-jobs. Fix required this time: `gh api -X POST .../force-cancel` on the old run (plain `gh run cancel` does NOT reap an offline-runner job). Recommendation: split self-hosted E2E lanes into their own workflow/concurrency group so an offline runner can never stall the merge-required lanes. Capture as a work-ledger item.

## Problem

The E2E harness (`tests/e2e-cucumber/tests/e2e.rs` ~line 779–791) exits 1 on any
XPASS or unexpected-failure. That is too strict for intermittent known bugs. When
a flaky bug happens not to reproduce on a given run, its xfail scenario passes →
counts as XPASS → suite exits 1 → the required check goes red and blocks merge —
even though 0 real failures occurred. The `expectations.toml` comments already
acknowledge this ("an occasional XPASS is expected and harmless here"), yet the
harness still treats it as fatal.

**Concrete impact (live):** PR #127 (`test-e2e-diagnose`) is blocked — its GPU
lane reconciled 9 xfail, 2 XPASS, 0 unexpected failures (all 6 diagnose scenarios
passed, everything green), but the suite exits 1 purely because two EAI-7333 vLLM
scenarios (`serve-vllm-inference`, `serve-readiness-contract`) passed when
expected to fail. Pure flake drift, orthogonal to that PR's change.

**Why not just flip those to expect-pass:** EAI-7333 is genuinely intermittent —
documented flipping across MI300X runs (29404668327 passed, 29413321046 failed).
Flipping to expect-pass would cause the opposite failure (unexpected-FAIL) on the
next host/run where the bug resurfaces. The bug is still open; the xfail is
correct.

## Solution

Add a per-condition `flaky = true` marker in `expectations.toml` (parsed in
`src/expectation.rs`) that makes a scenario's XPASS **non-fatal** — still printed
in the reconciliation line for visibility, but excluded from the exit-1 decision.
Unexpected-FAIL stays fatal always. Apply `flaky = true` to the EAI-7333 vLLM
entries and the new EAI-7455 lemonade-Windows entries.

## Scenarios

_To write with the bdd-scenarios skill before implementation. Sketch:_

```gherkin
Scenario: Flaky xfail that passes does not fail the suite
  Given an xfail scenario marked flaky = true
  When it unexpectedly passes (XPASS) with no other failures
  Then the reconciliation line reports the XPASS
  And the suite exits 0

Scenario: Unexpected failure is always fatal
  Given any scenario (flaky or not)
  When it fails when expected to pass
  Then the suite exits 1

Scenario: Non-flaky XPASS remains fatal
  Given an xfail scenario without flaky = true
  When it unexpectedly passes
  Then the suite exits 1
```

## Implementation Steps

### Done ✅
- ✅ Added `flaky` field to `XfailEntry` + `Expectation::ExpectXfail` in `src/expectation.rs`; parsed from `expectations.toml` (`#[serde(default)]`, defaults false).
- ✅ Extracted the exit classification into a pure `reconcile()` + `Reconciliation` (with `is_fatal()`) in `src/expectation.rs`; `tests/e2e.rs` now calls it. Flaky XPASS printed as "(flaky, non-fatal)" and excluded from exit-1; non-flaky XPASS + unexpected-FAIL stay fatal.
- ✅ Marked EAI-7333 vLLM entries (`serve-vllm-inference`, `serve-readiness-contract`) `flaky = true`, with grammar doc in the toml header.
- ✅ Added 5 unit tests (flaky parse/default, flaky-XPASS non-fatal, non-flaky-XPASS fatal, unexpected-FAIL always fatal, all-expected clean). `cargo test --lib` green (18 passed); clippy clean.
- ✅ Container gate green (exit 0, full mock E2E suite ran, reconciliation line correct). Code reviewed LGTM.
- ✅ Commit `5ee8341` (signed), push to origin, PR #138 open.
- ✅ DCO fix: amended `5ee8341` → `3dd423a` (added `Signed-off-by`, re-signed), force-pushed.
- ✅ CI re-triggered (run 29910874741); all 8 merge-required checks green (including `Commit signatures + sign-off`); PR `MERGEABLE` but awaiting approving review.
- ✅ CI deadlock diagnosed: old run's Windows job (`strix-halo-windows` runner, offline) held shared concurrency group; resolved via `gh api -X POST .../force-cancel` (plain cancel cannot reap offline-runner jobs).
- ✅ PR #138 rebased onto current main (was 20 behind); my changes intact, merged cleanly with main's new e2e code, no conflicts. HEAD `5393392`, signed (amd fallback key) + Signed-off-by.
- ✅ GPU E2E triage: feature verified working (`0 XPASS (0 flaky), 0 unexpected failures`). Orthogonal EAI-7052 XPASS'es are a separate decision (flip-expect-pass / mark-flaky / defer).
- ✅ Container gate re-run (post-rebase): green (31 scenarios, 0 XPASS, 0 unexpected failures).
- ✅ Force-pushed rebased branch `5393392` to origin.
- ✅ `gh pr ready 138` — flipped out of draft to ready-for-review. CI re-triggered (run 30070568396).
- ✅ Collision detected: PR #142 (rominf, broader "stabilize GPU E2E and merge queue") landed identical flaky-XPASS feature on same 3 files. #138 is now fully redundant. PR #142 merged 2026-07-27 (commit `a1a8079`).
- ✅ Closed #138 as superseded by #142 (all behavior + markers + tests now on main via #142).

### Todo 📋
- 📋 EAI-7052 decision (flip-expect-pass / mark-flaky / defer) — advisory GPU lane, non-merge-blocking.
- 📋 EAI-7455 lemonade-Windows entries: N/A here — they live on `fix-e2e-share-lemonade-engine`; that branch marks them flaky after this lands.
- 📋 PR #138 awaiting human approving review (all required checks green).

## Next Steps

1. PR #138 awaiting review/merge.
2. Once merged: rebase `fix-e2e-share-lemonade-engine` on top (it will mark EAI-7455 Windows entries flaky).

## Checklist

- [ ] Scenarios written and reviewed before implementation
- [ ] `flaky` parsed and defaulted false
- [ ] Unexpected-FAIL remains fatal in all cases
- [ ] Reconciliation line still prints flaky XPASS

## Blockers / Open Questions

- **RESOLVED: #138 superseded by #142 (merged 2026-07-27)**. PR #142 merged on main with identical flaky feature (same `flaky: bool` field, same two EAI-7333 entries marked flaky, same non-fatal exit gate). #138 is now fully redundant — all behavior + markers + test coverage on main. Recommendation: close #138 as superseded.
- **EAI-7052 mis-attribution** (independent of #138/#142): EAI-7052 = "Lemonade should use installed ROCm" (Resolved 2026-07-16) — unrelated to xfail symptom. Two scenarios XPASS'd on #138 runs but xfail'd on #142 runs → intermittent. Separate cleanup issue, not urgent.

## Notes

- **Why it matters beyond #127:** also protects `fix-e2e-share-lemonade-engine` —
  the EAI-7455 Windows xfails are themselves flaky (the lemonade daemon race is
  non-deterministic), so on a lucky run they XPASS and trip the same exit-1,
  turning the lane red in mirror image of the bug just fixed.
- Key files: `tests/e2e-cucumber/tests/e2e.rs` (~779–791, exit decision),
  `src/expectation.rs` (parse), `expectations.toml` (markers).
- Related: [[test-add-e2e-robot-framework]] (EAI-7333 context), [[fix-e2e-share-lemonade-engine]] (EAI-7455 Windows xfails).

## Worktree Context

**Worktree directory**: /Users/fres/Developer/rocm-cli-wt/fix-xpass-non-fatal-flaky (active).
- Recreate with: `create_worktree.sh fix-xpass-non-fatal-flaky`
- Container gate script copied in at `workspace/wip/container-test.sh` (gitignored).

## Work Log

### 2026-07-17

- Created WIP capturing the flaky-XPASS-non-fatal design (own branch, not started).
- Blocking #127 live; recommended delivery is a separate PR landed first, then rebase #127 and the lemonade branch on top.
- Next: user decides separate-PR-vs-bundle, then create branch/worktree and write scenarios.

**2026-07-17 (implementation & verification):**
- EAI-7456 → In Progress, assigned to Fredrik.
- Full implementation done: 3 files (+211/-38); `flaky` field + pure `reconcile()`/`Reconciliation{is_fatal()}` in expectation.rs; e2e.rs calls it; both EAI-7333 vLLM entries marked `flaky=true`.
- 5 unit tests (parse, default, fatal/non-fatal logic); `cargo test --lib` 18 passed; clippy clean.
- Container gate: warm `.cargo-container` (397M) copied from sibling; gate re-running offline. Did not complete before session end (ongoing background task).
- Next: await gate → commit → push → PR (separate, per delivery decision).

**2026-07-17 (gate verification):**
- Container gate (`container-test.sh all`) re-run with warm 397M cargo cache from sibling worktree — **GREEN** (exit 0). Full mock E2E suite ran in Linux binary; reconciliation printed: `4 xfail (failed as expected), 0 XPASS (0 flaky, non-fatal), 0 unexpected failure(s)`. Code verified.
- Ready: commit + push + open PR (separate, per recommendation).

### 2026-07-21

- **Implementation complete**: 3 files (+211/-38), `flaky` field in `XfailEntry` + `Expectation::ExpectXfail`, extracted pure `reconcile()`/`Reconciliation` in expectation.rs, e2e.rs exit decision refactored to use it.
- **Verification**: 5 unit tests (parse/default/fatal logic), `cargo test --lib` 18 passed, clippy clean, container gate green (exit 0, warm `.cargo-container` 397M from sibling, full mock E2E suite ran in Linux binary, reconciliation printed correctly: `4 xfail, 0 XPASS (0 flaky, non-fatal), 0 unexpected failure(s)`).
- **Status**: Code ready to commit/push/open PR (separate, per delivery decision). EAI-7456 In Progress, assigned Fredrik.

**2026-07-21 (final verification):**
- Reviewed by opencode: reconcile() pure fn, 4-bucket logic, flaky defaults false via serde(default), TOML run IDs cited. BTreeMap clone noted (minor, intentional for testability).
- All gates passed: unit tests (18), clippy, container gate (full mock E2E, exit 0). Ready to commit/push/PR.

### 2026-07-22

- **Commit + push + PR**: all pre-commit hooks passed (cargo fmt caught 1 line); signed commit `5ee8341`; pushed to origin; PR #138 open.
- **PR body**: full test plan (unit tests, clippy, container gate), why flaky instead of expect-pass, run IDs for intermittent behavior.
- **WIP stage**: 7-PR-open.

**2026-07-22 (independent second-agent review — opencode):**
- Reviewed the working diff of the 3 tracked files (expectations.toml, src/expectation.rs, tests/e2e.rs) against the WIP's Problem/Solution/Scenarios. **Verdict: passed** — recorded in the new `Pre-PR-check` field (this WIP predated that field, so it was backfilled). Note: PR #138 was already open, so this is effectively a post-open review, not a pre-PR gate.
- Findings: reconcile() extracted as pure testable fn with 4 bucket tests; is_fatal() gates only on non-flaky xpass + unexpected_fail; flaky defaults false via serde(default); Scenarios sketch (flaky-XPASS non-fatal / unexpected-FAIL fatal / non-flaky-XPASS fatal) all map to a passing test. Nit (non-blocking): BTreeMap rebuilt via clone in e2e.rs — minor, acceptable for testability.
- Not re-run locally this session: cargo test / clippy / container gate (WIP records them green on 2026-07-21).
- Posted the review to tmux window 0 of this session per user request.

**2026-07-22 (post-push CI analysis):**
- PR #138 pushed; CI kicked off. Two check failures: (1) missing DCO `Signed-off-by` trailer (code signed fine, just missing sign-off flag); (2) E2E GPU lane exited 1 on unrelated flake (2 flaky-XPASS of EAI-7333 vLLM now correctly non-fatal per my change; 1 non-flaky XPASS on EAI-7052 + 1 unexpected-FAIL on chat scenario — orthogonal flake drift, not caused by this PR).
- Reconciliation line on GPU run proves feature works: `"Reconciliation: 7 xfail, 3 XPASS (2 flaky, non-fatal), 1 unexpected failure(s)."` — the 2 EAI-7333 entries correctly printed as "(flaky, non-fatal)" and non-fatal (exit would be 1 only on the non-flaky XPASS + unexpected-FAIL, not the flaky ones).
- **Blocker (awaiting user):** amend commit with DCO (`git commit -s --amend`) and force-push to re-trigger clean CI.
- **Note on GPU failures:** orthogonal to this PR; recommend either (a) re-run to confirm flake vs. real, or (b) mark `serve-default-engine-working-endpoint (EAI-7052)` flaky too (separate follow-up PR, or include here if user wants).

**2026-07-22 (amend + CI re-trigger):**
- Commit `5ee8341` missing DCO; amended with `git commit -s --amend`, re-signed (good sig), force-pushed → `3dd423a`.
- CI re-triggered (run 29910874741); CodeQL green; main CI (sign-off + GPU) pending. Old run 29901541610 holds shared concurrency group (GPU job stuck on offline `app-dev-gpu`); cancelled old run but GitHub slow to reap.
- Feature confirmed working from first run: reconciliation `7 xfail, 3 XPASS (2 flaky, non-fatal), 1 unexpected failure` — EAI-7333 flaky entries correctly non-fatal.

**2026-07-22 (CI queue analysis):**
- Investigated CI deadlock: old run's GPU job completed (failure), but Windows job (`strix-halo-windows`) still queued on offline runner; cannot be cancelled → holds shared concurrency group → new run stuck pending 54+ min.
- `restore-app-dev-runner` skill checked but not applicable: bottleneck is `strix-halo-windows` (offline), not `app-dev-gpu`.
- No further progress until Windows runner back online or GitHub times out offline job.

**2026-07-22 (force-cancel + merge-ready state):**
- Old CI run 29901541610 stuck on Windows job queued on offline runner; `force-cancel` via `gh api -X POST .../force-cancel` reped it (plain `gh run cancel` cannot reap offline-runner jobs).
- New run 29910874741 freed from concurrency block, created 5 jobs, then all 8 merge-required checks completed successfully (including `Commit signatures + sign-off`).
- PR #138 now `MERGEABLE`, gated only on `REVIEW_REQUIRED` (human approving review).
- Non-required E2E lanes (GPU, Strix-Windows) are advisory, do not block merge.
- Concurrency-group root cause & recommended fix (split self-hosted E2E into separate workflow) captured in WIP CI status section for later work-ledger item.

**2026-07-22 (final nudge):**
- Re-polled PR #138: unchanged — OPEN/MERGEABLE, mergeStateStatus BLOCKED, reviewDecision REVIEW_REQUIRED, 0 reviews, head 3dd423a, all 8 required checks green.
- No progress until human approval lands. Stage stays 7-PR-open.

**2026-07-22 (final session close):**
- Re-polled PR #138: OPEN/MERGEABLE, mergeStateStatus BLOCKED, reviewDecision REVIEW_REQUIRED, 0 reviews, head 3dd423a, all 8 required checks green.
- No further progress until human approval. WIP stage remains 7-PR-open; blocker set to BLOCKED (awaiting user).
- Concurrency-group finding (split self-hosted E2E into separate workflow) captured for later work-ledger item.

**2026-07-22 (final nudge):**
- Re-polled PR #138: unchanged — OPEN/MERGEABLE, mergeStateStatus BLOCKED, reviewDecision REVIEW_REQUIRED, 0 reviews, head 3dd423a, all 8 required checks green.
- No progress until human approval lands. Stage stays 7-PR-open.

### 2026-07-23

**Session: re-check on PR #138 approval + CI.**
- Re-polled PR #138 state: OPEN/MERGEABLE, mergeStateStatus BLOCKED, reviewDecision REVIEW_REQUIRED, 0 reviews, head 3dd423a, all 8 required checks green.
- Branch now BEHIND main (main advanced); strict branch protection requires rebase before merge, but only AFTER approval (to avoid re-running CI for a PR that still can't merge).
- **Blockers (awaiting user):** (1) human approving review; (2) branch rebase/update (after approval, before merge).
- No merge possible until approval; no action items for agent.

**2026-07-23 (final nudge + merge decision):**
- PR #138 state: OPEN/MERGEABLE, mergeStateStatus BEHIND (main advanced), reviewDecision REVIEW_REQUIRED, 0 reviews, head 3dd423a, all 8 merge-required checks green.
- Asked fres: approval gate + need for rebase (held branch update until after approval to avoid re-running CI for unmergeable PR).
- **Blockers (awaiting user):** (1) human approving review on PR #138; (2) branch update/rebase (after approval, before merge). No merge possible until approval lands.

**2026-07-24 (final session):**
- Re-polled PR #138 thrice: OPEN/MERGEABLE, mergeStateStatus BLOCKED then BEHIND, reviewDecision REVIEW_REQUIRED, 0 reviews, head 3dd423a, all 8 merge-required checks green.
- Diagnosed CI queue deadlock: old run 29901541610 held shared concurrency group on Windows job queued to offline `strix-halo-windows` runner (GitHub cannot cancel); used `gh api -X POST .../force-cancel` to reap it (plain cancel fails on offline-runner jobs). Freed new run 29910874741 to create/complete all required checks.
- Concurrency-group gap identified: shared per-ref group with `cancel-in-progress: true` cannot reap jobs stuck on offline self-hosted runners → blocks merge-required lanes. Recommended fix: split self-hosted E2E into separate workflow with unique per-run concurrency group. Captured for work-ledger CI redesign item (separate from #138).
- **Blocker (final):** PR #138 awaiting human approving review (all required checks green, mergeable, but mergeState BLOCKED on REVIEW_REQUIRED). Branch now BEHIND main; will update-branch after approval, before merge. No further progress possible until approval.

**2026-07-24 (full implementation → PR ready):**
- **Corrected misreading**: PR #138 is a DRAFT, not review-gated. Ball was with me.
- **Rebased** onto current main (20 behind). My changes intact, merged cleanly, no conflicts. HEAD `5393392`, signed (amd fallback key) + Signed-off-by.
- **GPU E2E triage**: feature verified — `8 xfail, 2 XPASS (0 flaky, non-fatal), 0 unexpected failure`. EAI-7333 flaky entries correct. Two EAI-7052 entries XPASS'd (separate call, non-merge-blocking).
- **Container gate green** (31 scenarios, 0 XPASS, 0 unexpected).
- **Force-pushed** rebased branch; **`gh pr ready 138`** (out of draft). CI re-triggered (run 30070568396).
- **PR state**: OPEN/MERGEABLE, BLOCKED on genuine REVIEW_REQUIRED (0 reviews). All 8 merge-required checks green. Awaiting human approval.
- **Open**: EAI-7052 decision (flip-expect-pass / mark-flaky / defer) — advisory lane, non-blocking.

**2026-07-24 (EAI-7052 evidence + PR collision discovery):**
- Gathered EAI-7052 evidence per fres: mis-attribution confirmed (EAI-7052 = "Lemonade use installed ROCm", Done/Resolved 2026-07-16, unrelated to Vulkan hang). Two scenarios XPASS'd on my runs (07-24) but xfail'd on #142 merge-queue (07-23) → intermittent, not fixed.
- **Critical discovery**: PR #142 ("ci: stabilize GPU E2E and merge queue", rominf, OPEN) rewrites the same 3 files as #138 with parallel flaky-XPASS implementation. Created 2026-07-23 09:10Z (one day after #138 2026-07-22 07:48Z). Whichever lands first, other conflicts hard + becomes redundant.
- Verified: #138 is first (2026-07-22); #142 created second (2026-07-23) but cycling through merge queue, may be closer to landing.
- **Blockers (awaiting user)**: (1) #138/#142 collision resolution (close #138 in favor of #142 / close #142 in favor of #138 / coordinate with rominf); (2) EAI-7052 call (mark-flaky / flip-expect-pass / defer) depends on (1). No further progress on #138 until (1) decided.

### 2026-07-28

**Session: #142 merged, #138 collision resolved.**
- Verified: PR #142 merged 2026-07-27 (commit `a1a8079`). #138 now CONFLICTING/DIRTY (was READY, now MERGING off-plan).
- **Full redundancy confirmed**: main now has `flaky: bool` field (both XfailEntry + ExpectXfail), same two EAI-7333 entries marked flaky, same exit-gate logic (flaky XPASS non-fatal), unit tests in e2e-report.
- #138 fully redundant — all behavior + markers now on main. Recommendation: **close #138 as superseded by #142**.
- EAI-7052 mis-attribution remains (separate cleanup, independent of both PRs).

**2026-07-28 (collision resolution + PR close):**
- PR #142 ("ci: stabilize GPU E2E and merge queue", rominf) merged 2026-07-27 (commit `a1a8079`).
- Verified: main now has identical `flaky: bool` field (XfailEntry + ExpectXfail), same two EAI-7333 entries marked flaky, same non-fatal exit gate + unit tests.
- #138 fully redundant — all feature behavior + markers + test coverage shipped on main via #142.
- **Closed #138** with superseded note. EAI-7052 mis-attribution fixed on main (now EAI-7423, not EAI-7052).
- Task complete — feature shipped (via #142, not #138).
