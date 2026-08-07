# WIP: Fix CI self-hosted E2E lane timeout (offline runner holds concurrency group)

**Stage:** 8-in-review — PR #193 OPEN (https://github.com/ROCm/rocm-cli/pull/193); commit 9087896 pushed (rebased on main, signed+signoff), Linux gate GREEN on legion (workspace mode), CI running
**Pipeline:** lightweight
**Branch:** fix-ci-selfhosted-lane-timeout
**Pre-PR-check:** review-done — OpenCode gpt-5.6-sol reviewer, 2026-08-06, @0c884da+bfd0fb8bbaea934a
**Ticket:** EAI-7548 (Bug, component rocm-cli) — https://amd.atlassian.net/browse/EAI-7548
**Last Updated:** 2026-08-07
**Token Usage (cumulative):** in=625k→635k out=532k→535k cache_create=2142379 cache_read=71648k→71850k calls=288→292
**Token Usage:** in=625k out=532k cache_create=2142379 cache_read=71648k calls=288

---

## Problem

`ci.yml`'s shared concurrency group `CI-<ref>-shared` uses `cancel-in-progress: true`.
When a new run supersedes an older one, GitHub **cannot cancel a job queued on an
OFFLINE self-hosted runner** — so the superseded run stays "alive" holding the
concurrency group, and the new run sits `pending` with 0 jobs indefinitely.

**Impact:** seen on PR #138 (2026-07-22): a queued job on the offline `app-dev-gpu`
runner held the group; the new run never started and had to be force-cancelled by
hand (`gh api -X POST .../force-cancel`; plain `gh run cancel` does NOT reap an
offline-runner job). This blocks the merge-required lanes for anyone whose run gets
superseded while a self-hosted runner is offline.

## Solution

### DESIGN GATE RESOLVED (2026-08-03) — the planned "preferred fix" is INVALID

Two findings from the design investigation kill the original plan and reshape the task:

1. **`timeout-minutes` does NOT reap a job stuck in the QUEUED phase.** GitHub's job
   timeout timer only starts once a runner picks the job up and it is *running*; it
   never fires while a job sits "Waiting for a runner to pick up this job…" on an
   OFFLINE self-hosted runner. That queued-and-uncancellable state is exactly the
   stall we saw on PR #138, so adding/relying on `timeout-minutes` cannot clear it.
   (Sources: GitHub community discussion #50926; actions/runner #4312 — both confirm
   the timer covers running time only, not queue wait.)
2. **The lanes already have `timeout-minutes` AND dispatch is already isolated.** All
   three self-hosted E2E lanes already carry a cap (`e2e-gpu`: 90, `e2e-gpu-strix-
   ubuntu`: 35, `e2e-gpu-strix-windows`: 35) and `workflow_dispatch` already gets a
   UNIQUE concurrency group (`…-<run_id>`). Both landed in #69, before this WIP was
   written. So there is nothing to add there — the WIP's snapshot was stale.

**Remaining real exposure:** push / pull_request / merge_group still share the group
`CI-<ref>-shared` with `cancel-in-progress: true`. When a newer run supersedes an
older one whose self-hosted job is queued on an offline runner, GitHub can't cancel
that job, so the superseded run keeps holding the shared group and the new run sits
pending with 0 jobs — the #138 stall.

### Only mechanism that actually works: structural separation

Because a job queued on an offline runner is fundamentally uncancellable, the fix is
to make sure such a job never shares a run / concurrency group with the merge-required
checks. Options (scope decision needed — see Blockers):

- **(A) Split the self-hosted E2E lanes into their own workflow** with their own
  concurrency group (its own `-<ref>` key, `cancel-in-progress: true`). An offline
  runner can then only stall *that* workflow's own supersession, never the required
  lanes in `ci.yml`. Larger diff (new workflow file, move 3 jobs + the report's
  `needs`, re-wire artifact consolidation), but it's the robust structural fix.
- **(B) Scope `cancel-in-progress` so it never supersedes on the shared group in a way
  that can strand a queued self-hosted job** — e.g. keep supersession for the
  GitHub-hosted required lanes but give the self-hosted lanes a non-cancelling group.
  Requires per-job concurrency (job-level `concurrency:`), which interacts subtly with
  required-check reporting; needs careful design.
- **(C) Accept the manual force-cancel** as the documented operational remedy (the
  `gh api -X POST .../force-cancel` we already used) and close EAI-7548 as
  won't-fix-in-CI-config, since the true root cause is runner offline-ness (addressed
  by the runner-persistence work, EAI-7447 / persist-app-dev-ci-runner).

Recommendation: **(A)** is the only change that structurally guarantees the required
checks can't be stalled by an offline self-hosted runner, and it composes cleanly with
the existing report job (artifacts glob already auto-discovers platforms). But it's a
bigger diff than the ticket's original framing, so the scope needs a human call before
implementation.

## Scenarios

(Lightweight — activate bdd-scenarios skill before finalizing.)

```gherkin
Scenario: Hosted CI is not held hostage by unavailable GPU hardware
  Given the GPU hardware for a pull request's E2E check is unavailable
  And that check is still waiting to run
  When a contributor pushes a newer commit to the pull request
  Then hosted CI supersedes its previous run for that commit
  And the pull request's hosted required checks run for the new commit
  Without waiting on the unavailable GPU hardware
```

Note (implementation, kept out of the scenario): the waiting GPU job itself is
NOT reaped — it stays queued until the runner returns or GitHub reaps it — but it
can no longer hold up hosted CI, because the two now run in separate workflows
with separate concurrency groups. The stale row this replaced wrongly claimed a
timeout releases the group; the design gate proved `timeout-minutes` never fires
on a queued job. Caveat unchanged: while the self-hosted checks remain in the
required list, an offline runner can still block a MERGE (not the hosted checks'
execution) via a missing required check until they are de-required — a separate
branch-protection change.

## Implementation Steps

### Done ✅
- ✅ Resolved the design gate: `timeout-minutes` does NOT reap a QUEUED job on an
  offline runner (timer covers running time only). Planned fix invalid.
- ✅ Confirmed all 3 self-hosted lanes already have `timeout-minutes` and dispatch
  already has a unique concurrency group (landed in #69).

### Implementing (option A, "Split only") ✅ COMPLETE
- ✅ Scope decided: **A — split self-hosted lanes into their own workflow.** "Split only":
  branch protection left as-is (user handles the required-check list separately).
- ✅ Created `.github/workflows/e2e-selfhosted.yml` (652 lines): the 3 self-hosted jobs
  (`e2e-gpu`, `e2e-gpu-strix-ubuntu`, `e2e-gpu-strix-windows`) + a self-hosted-side
  `e2e-report` (renamed check `E2E consolidated report (self-hosted)` to avoid colliding
  with ci.yml's required one). Own concurrency group (`${{ github.workflow }}-…`), own
  trimmed `changes` gate (cross-workflow `needs` impossible), triggers mirror ci.yml
  (push/PR/merge_group/dispatch). Dropped the `build-and-test` dep (cross-workflow); the
  jobs `cargo build` themselves and are `continue-on-error`.
- ✅ Removed those 3 jobs from `ci.yml` (1281→750 lines); mock `e2e` (hosted, required)
  stays. Repointed `e2e-report.needs` to just `[changes, e2e]`; kept its required name
  `E2E consolidated report`. Trimmed dead dispatch inputs (GPU platform options,
  name_filter, include_nightly — mock lane uses none).
- ✅ Validated: both files `yaml.safe_load` OK; all 5 required check names still produced,
  each by exactly one workflow (no collision, no orphaned required check); ci.yml clean of
  self-hosted refs; actionlint clean except the pre-existing custom-label warnings (main's
  ci.yml already emits 8). e2e_report.rs renders whatever artifacts it finds (no hardcoded
  platform requirement), so the per-workflow report split is safe.
- ✅ Pre-PR review findings (6 total) all addressed and verified:
  - F1 (native label): rebased to main + rebuilt e2e-selfhosted.yml from current jobs
  - F2 (empty report): refuted with actions/download-artifact docs + clarifying comment
  - F3 (stale README): updated GPU dispatch commands and job table
  - F4 (missing sign-off): commit amended with `-s` flag
  - F5 (no regression test): added xtask/src/workflow_contract.rs (3 tests pass here, fail on main)
  - F6 (stale scenario): Gherkin rewritten to implemented guarantee
- ✅ Committed + signed as **0c884da** (all hooks passed: fmt, license-header, signing).
- ✅ Verification complete: signature valid (G), sign-off present, 3 contract tests pass, both workflows parse, clean tree.

### Pre-PR review — ROUND 2 (reviewer @0c884da, 2026-08-06) — findings addressed (STAGED, not committed)
- **R2-F1 (empty report — I WAS WRONG in round 1): VALID, fixed.** Verified in the ACTUAL
  pinned v8.0.1 source (`src/download-artifact.ts` line 190-192): when `artifacts.length === 1`
  the artifact extracts into `resolvedPath` (the dir root) regardless of `pattern`/
  `merge-multiple` — my round-1 docs-based refutation was wrong. After the split each report
  job has exactly one artifact, so `discover()` (immediate-subdir scan) would find nothing →
  empty report. FIX: taught `discover()` to also handle a root-level `report.json`, labeling
  it from the sibling `platform.json`'s `platform_slug` (slug→artifact-name map). Added 2
  tests (root-level w/ slug for all 4 platforms; sidecar-absent → mock fallback). Removed the
  incorrect per-artifact-singleton comment in BOTH workflows.
- **R2-F2 (contract test can false-pass): VALID, fixed.** Rewrote workflow_contract.rs to
  EXTRACT the complete `runs-on` value (joins block/flow multiline forms) and the actual
  top-level `concurrency.group` value (joins folded `>-` lines, stops at next key, ignores
  comments) — then assert on those, not whole-file substrings. Added 2 extractor-guard tests
  proving the multiline parsing works. 5 tests pass.
- **R2-F3 (docs/ci-hardware-testing.md stale): VALID, fixed.** Rewrote for the split: intro
  names e2e-selfhosted.yml + the offline-stall rationale; platform table has a workflow column
  + `native` labels; triggers section drops the build-and-test dep and uses the new dispatch;
  blocking section documents the required-check/offline caveat; report note covers the two
  report jobs + the v8 single-artifact flatten.
- **R2-F4 (WIP scenario overclaims/implementation-centric): VALID, fixed.** Rewrote the
  Gherkin behaviorally ("hosted CI is not held hostage by unavailable GPU hardware"); moved
  filenames/concurrency mechanics into a following note.
- Reviewer confirmed round-1 DCO, license header, native labels, README fixes are verified.
- ✅ Re-validated: all 71 xtask tests pass; fmt clean; both workflows parse.
- 📋 BLOCKER (same as before): this turn arrived via the reviewer's `--pre-pr-review` relay,
  whose gate blocks `git commit`. The amend must run in a turn driven directly by fres.
  Next: `git-commit-with-fallback --amend -s` (re-using the message), then request re-review.

### Pre-PR review — ROUND 3 (reviewer @0c884da, 2026-08-06) — findings assessed (STAGED)
- **R3-F1 (no-sidecar GPU failure misattributed to Mock): VALID, fixed.** Confirmed the exact
  path: e2e.rs writes report.json (line 849) then, on a parsing/hook error, exits (line 862)
  BEFORE writing platform.json (line 893). So a GPU run that errors leaves report.json with NO
  sidecar → my round-2 fallback labeled it `e2e-report` (Mock/Linux) → hardware failure shown
  as Mock in the grid. FIX: `label_for_root_report` now maps ONLY an explicit `mock` slug to
  the mock artifact; a missing/unknown slug returns a neutral `e2e-unknown-report` (renders as
  "Unknown", never a false real platform). Updated the no-sidecar test to assert neutral (not
  mock) and added an unknown-slug test. 72 xtask tests pass.
- **R3-F2 (commit uses gmail, not employer, identity): DEFERRED TO FRES — not an agent call.**
  AGENTS.md §2 says "employer's author/committer identity **(when required)**". The empirical
  signal contradicts a hard requirement: fres's OWN prior merged commit on this repo
  (#124-era) is authored `Fredrik Espinoza <fredrik.espinoza@gmail.com>` — the same identity
  as HEAD — and fres has been merging under it; only one other contributor uses @amd.com. My
  standing rules forbid changing git identity/config without explicit direction. So this is
  fres's decision: keep the established gmail identity, or re-author under an @amd.com identity
  (author + committer + Signed-off-by + signing key). Flagging, not switching.
- Reviewer confirmed all four round-2 fixes verify.
- ✅ Re-validated: 72 xtask tests pass; fmt clean; both workflows parse.

### Pre-PR review — ROUND 4 (reviewer @0c884da, 2026-08-06) — final finding fixed (STAGED)
- **R4-F1 (unknown label still falsely Linux): VALID, fixed.** My round-3 neutral label
  `e2e-unknown-report` still went through `fallback_descriptor`, which hardcodes OS = "Linux"
  — so a Windows GPU run that errored before writing platform.json would be reported as Linux.
  FIX: added an explicit `"unknown"` arm in `parse_descriptor` (crates/e2e-report/src/lib.rs)
  that renders **Unknown / Unknown** (OS not defaulted to Linux). Added two descriptor unit
  tests in the e2e-report crate: known-artifact mapping, and the unknown → Unknown/Unknown
  assertion. 36 e2e-report + 72 xtask tests pass; fmt clean.
- **Identity finding WITHDRAWN by reviewer** — confirmed my assessment: AGENTS.md qualifies it
  "when required", and merged repo history establishes the gmail identity as accepted. So the
  commit stays as-is (`Fredrik Espinoza <fredrik.espinoza@gmail.com>`).
- Reviewer confirmed all prior-round fixes verify. **No open findings remain.**
- 📋 Ready to amend-commit (blocked only by the reviewer-relay gate on this turn) then reopen.

### Pre-PR review — TERMINAL verdict (2026-08-06): review-done
- Reviewer returned **review-done** @0c884da+bfd0fb8bbaea934a — terminal, distinct from
  `passed`, recorded verbatim. No findings ≥80 confidence remain after the full-scope pass.
  Verified: 36 e2e-report + 72 xtask tests, fmt, YAML, hawkeye, diff checks all pass; the
  unknown descriptor renders Unknown/Unknown. Pre-PR gate is SATISFIED.
- NEXT (needs a fres-driven turn — reviewer relay blocks commits, and opening a PR is an
  externally-visible action): `git-commit-with-fallback --amend -s` to fold all four rounds of
  review-response changes into commit 0c884da, then open the PR. All changes are staged; the
  identity stays gmail (finding withdrawn). Separately, fres de-requires the 4 self-hosted
  checks in branch protection (the "Split only" caveat).

## KEY FINDING — required-check contradiction (drives "Split only" caveat)

Branch protection on `main` (`strict:true`, `enforce_admins:true`) lists all FOUR
self-hosted checks as **required**: `E2E tests (GPU)`, `(Strix Halo, Ubuntu)`,
`(Strix Halo, Windows)`, `E2E consolidated report`. But the jobs are coded
`continue-on-error: true` ("non-blocking"), and PRs #114/#124/#131/#134 all merged with
`E2E tests (GPU)` = FAILURE (one CANCELLED). Both are true because `continue-on-error`
neutralizes a job that RAN-and-failed, but does nothing for a job that NEVER RAN
(offline runner) — a missing required check blocks the merge.

Consequence for "Split only": the split fixes the concurrency STARVATION of the hosted
required checks (they start immediately instead of pending-with-0-jobs). It does NOT by
itself unblock merges when the GPU runner is offline — those 4 checks would still be
required-but-missing. Fully closing that needs them removed from the required list (an
admin branch-protection change), which the user opted to handle separately.

## Next Steps

**BLOCKED (awaiting user):**
- De-require the four self-hosted checks in branch protection (`E2E tests (GPU)`, `E2E tests (Strix Halo, Ubuntu)`, `E2E tests (Strix Halo, Windows)`, `E2E consolidated report (self-hosted)`). The "Split only" caveat: this PR fixes hosted-check starvation (they start immediately instead of pending-with-0-jobs), but an offline runner can still block a MERGE via missing-required-check until these four are removed from the required list (separate admin branch-protection change).

## RESOLVED
- Does `timeout-minutes` cancel a job still QUEUED on an offline runner?
  → **No.** The timer only starts once the job is running. Confirmed via GitHub
  community #50926 and actions/runner #4312.
- **(scope):** A — split into own workflow, "Split only" (no branch-protection
  change in this PR).

## Notes

- Related to the runner-reliability theme (offline runners): app-dev-gpu/Strix runner
  work and EAI-7447. Promoted from work-ledger inbox item 12.

## Worktree Context

**Worktree directory**: `/Users/fres/Developer/rocm-cli-wt/fix-ci-selfhosted-lane-timeout`
- Recreate with: `create_worktree.sh fix-ci-selfhosted-lane-timeout`

## Work Log

### 2026-07-22

- Filed EAI-7548 (Bug, component rocm-cli) for the offline-runner concurrency-group stall.
- Created this WIP (lightweight pipeline) and the worktree; promoted inbox item 12.
- Next: resolve the queued-vs-running `timeout-minutes` question, then edit `ci.yml`.

### 2026-08-03 (first session)

- Resolved the design gate. `timeout-minutes` does NOT reap a QUEUED job on an offline
  runner (timer covers running time only) — the planned "preferred fix" is invalid.
- Found the 3 self-hosted lanes ALREADY have `timeout-minutes` and dispatch already has
  a unique concurrency group (both from #69) — WIP snapshot was stale; nothing to add.
- Real fix is structural (split self-hosted lanes into their own workflow, option A) —
  bigger diff than the ticket framed. Surfaced to human for a scope decision; did NOT
  start implementation (design-gate stage, scope call is a human decision).

### 2026-08-04 (pre-PR review response)

- Reviewer (OpenCode gpt-5.6-sol) returned **changes-requested**, 6 findings. Assessment
  + action for each (all verified against the real code, not taken on faith):
  - **F1 (native label dropped) — VALID.** Root cause: my branch was built on a STALE base
    (main was 6 commits ahead; #146 added `native` to the Strix runs-on, #139 refactored the
    E2E jobs). Rebased onto origin/main and REBUILT e2e-selfhosted.yml from main's CURRENT
    job bodies, so `native` + all #139 changes carry over. Also aligned the new `changes`
    filter with main's (`xtask/**`, `tests/e2e-cucumber/**`).
  - **F2 (empty report via download-artifact) — REFUTED.** Official actions/download-artifact
    docs: with `pattern` and default `merge-multiple: false`, EVEN a single match extracts
    into a per-artifact subdirectory — exactly what discover() (immediate-subdir search)
    expects. Only by-name/by-id, or merge-multiple:true, flattens to root. No change needed;
    added a clarifying comment on the download step.
  - **F3 (README stale GPU dispatch) — VALID.** Updated tests/e2e-cucumber/README.md: job
    table now shows the workflow column, dispatch commands point to e2e-selfhosted.yml, report
    description notes the per-workflow split.
  - **F4 (missing Signed-off-by) — VALID.** Will commit with `-s` (see blocker below).
  - **F5 (no regression test) — VALID.** Added xtask/src/workflow_contract.rs (dependency-free
    text contract): asserts ci.yml schedules no self-hosted runs-on, e2e-selfhosted.yml owns
    the 3 GPU jobs, and the two workflows have distinct concurrency namespaces. 3 tests pass
    here; they FAIL on main (main's ci.yml has self-hosted runs-on lines). Caught a real stale
    comment in ci.yml (fixed).
  - **F6 (stale WIP scenario) — VALID.** Rewrote the Gherkin scenario to the implemented
    guarantee (hosted checks start despite a separately-queued self-hosted job; queued job is
    NOT reaped), retaining the required-check caveat.
- **BLOCKER:** all fixes are staged, but this turn arrived via the reviewer's `--pre-pr-review`
  tmux relay, whose gate permits source edits in this worktree but BLOCKS `git commit`. So the
  amend+`-s` commit must happen in a turn driven directly by fres (or a non-relay session).
  Next: `git-commit-with-fallback --amend -s` (wrapper passes args through), then request re-review.

### 2026-08-04

- User chose option A, "Split only". Discovered a required-check contradiction: all 4
  self-hosted checks are in main's REQUIRED list yet coded continue-on-error — so the
  split fixes hosted-check STARVATION but an offline runner can still block via
  missing-required-check until they're de-required (user's separate task). Captured in the
  "KEY FINDING" section above.
- Implemented the split: new `.github/workflows/e2e-selfhosted.yml`; removed the 3
  self-hosted jobs from `ci.yml`; repointed the hosted report. Validated YAML, required-
  check name coverage, and actionlint (only pre-existing custom-label warnings). Also
  confirmed nightly.yml has its OWN independent self-hosted jobs (no concurrency coupling),
  so it's unaffected by the split.
- Next: pre-PR review gate, then open PR.

### 2026-08-04 (second session)

- Code complete: `.github/workflows/e2e-selfhosted.yml` created (652 lines, 3 self-hosted jobs
  + self-hosted-side report with dedicated concurrency group and trimmed `changes` gate); the
  3 jobs removed from `ci.yml` (1281→750 lines); hosted report repointed to `[changes, e2e]`.
- Validation passed: both files parse; all 5 required checks produced (one per workflow, no
  collision); actionlint clean except pre-existing custom-label warnings.
- Signed and committed as **6976c7e** (commit-signing hook passed). Now awaiting mandatory
  pre-PR review gate (second-agent reviewer must run `pre-pr-review` skill and write verdict
  to WIP `Pre-PR-check` field). Author does not self-review.

### 2026-08-04 (third session, post-review remediation)

- All 6 review findings addressed: rebased to main's HEAD (F1/native label + #139 E2E changes restored), clarified artifact extraction (F2), updated README (F3), staged amended commit for `-s` sign-off (F4), added regression test contract (F5), rewrote scenario (F6).
- All fixes staged; commit pending direct session (previous relay-gate blocked `git commit`).
- Ready for amendment and re-review once user runs `git-commit-with-fallback --amend -s`.

### 2026-08-05 (commit finalization + verification)

- Amended commit with `-s` sign-off flag; all hooks passed (fmt, license-header, signing).
- Verified final tree: signature valid, sign-off present, 3 contract tests pass, both workflows parse, clean worktree.
- Commit finalized as **0c884da**; all 6 pre-PR review findings confirmed resolved.
- Awaiting re-review by second-agent reviewer before opening PR. "Split only" caveat remains — user to handle branch-protection de-require separately.

### 2026-08-06 (round-3 review findings assessment)

- **R3-F1 (neutral slug for missing platform.json): VALID, fixed.** Confirmed e2e.rs writes report.json before platform.json and exits on error without writing the sidecar. Updated `label_for_root_report` to map only explicit `mock` slug to mock artifact; missing/unknown slug → neutral `e2e-unknown-report` (renders "Unknown", never false platform). Updated no-sidecar test assertion to expect neutral, added unknown-slug test. **72 xtask tests pass; fmt clean; workflows parse.**
- **R3-F2 (commit uses gmail vs employer identity): DEFERRED — user decision required.** Empirical signal contradicts hard requirement: user's own prior merged commit on repo uses `Fredrik Espinoza <fredrik.espinoza@gmail.com>` (same identity as HEAD); AGENTS.md §2 says "when required", not unconditionally. My rules forbid changing git identity without explicit direction. Flagging for user judgment: keep gmail (established practice) or re-author under `@amd.com`?
- Changes staged; commitment pending identity decision from user.

### 2026-08-06 (round-4 review — final findings round)

- **R4-F1 (unknown label still hardcoding Linux): VALID, fixed.** Discovered round-3 neutral label `e2e-unknown-report` still went through `fallback_descriptor`, which hardcodes OS="Linux" — so a Windows GPU run that errored before writing platform.json would render as Linux. FIX: added explicit `"unknown"` arm in `parse_descriptor` (crates/e2e-report/src/lib.rs) → renders **Unknown / Unknown**. Added two descriptor unit tests (known-artifact mapping; unknown → Unknown/Unknown case). **36 e2e-report + 72 xtask tests pass; fmt clean; workflows parse.**
- **R3-F2 (identity): WITHDRAWN by reviewer.** Confirmed AGENTS.md §2 qualifies "when required" + merged repo history (user's own prior commit uses gmail identity). Commit stays as-is.
- **Terminal verdict: review-done** (verbatim, 2026-08-06, @0c884da+bfd0fb8bbaea934a). All prior-round fixes verified; no findings ≥80 confidence remain. Pre-PR gate SATISFIED. All changes staged on top of commit 0c884da.
- NEXT: User to drive a direct turn and run `git-commit-with-fallback --amend -s` (fold review-response changes into commit, then PR-ready). Separately, de-require the four self-hosted checks in branch protection.

### 2026-08-07

- Explored alternative to legion (`docker context` native-Linux target) for running `e2e` mode locally. Diagnosed blockers: acer's SSH key not accepted, and cannot confirm acer is bare-metal Linux vs another Docker-Desktop/WSL2 host. Verified WSL2 kernel detection is hardcoded (`/proc/version` read, "microsoft" match) with no override mechanism.
- Clarified the split's guarantee: it blocks e2e-selfhosted workflow's own supersession but does NOT undo the required-check contradiction (offline runner can still block merge via missing-required-check until de-required). Documented under KEY FINDING.
- PR #193 remains open, commit 9087896 is CI-live on legion. No further action by agent; awaiting user to de-require the four self-hosted checks in branch protection per "Split only" caveat.
