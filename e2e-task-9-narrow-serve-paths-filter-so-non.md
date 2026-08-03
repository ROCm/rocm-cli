# WIP: E2E Task #9: narrow 'serve' paths-filter so non-serve Rust PRs skip the GPU matrix

**Stage:** 8-in-review — PR #156 open (https://github.com/ROCm/rocm-cli/pull/156); commit d7896c6 signed+signoff, rebased on main d17fc0c, all checks GREEN, maintainer review pending
**PR:** https://github.com/ROCm/rocm-cli/pull/156
**Pipeline:** lightweight
**Branch:** e2e-task-9-narrow-serve-paths-filter-so-non
**Jira:** EAI-7746 (Task, rocm-cli, unassigned)
**Pre-PR-check:** review-done — OpenCode reviewer (gpt-5.6-sol), 2026-07-31, @ca9f297+e2748ac0b92b8d83 — PASSED after two review rounds (all issues fixed); short-name scenarios now @serves-on-gpu, full serve-step/feature sweep found no other real GPU serve escaping canary gating; focused tests + harness compile + clippy + git diff --check all pass.
**Last Updated:** 2026-08-03
**Bundles:** Task #8 (WL-175, merge_group gating + PR canary) — same branch/PR.

**Token Usage:** in=2573 out=1068987 cache_create=36204659 cache_read=267895667 calls=1270

---

## Problem

Parent: fix-speed-up-e2e umbrella (wlticket #47). Parent WIP: /Users/fres/Developer/rocm-cli-progress/fix-speed-up-e2e.md (Task #9). Gets its own WIP referencing the parent when work starts.

GOAL: Add a narrow 'serve' paths-filter so Rust-but-not-serve PRs (dash-only changes, unrelated crates) skip the heavy GPU serve matrix entirely.

LIVE-MAIN STATE (verified 2026-07-30, ci.yml 'changes' job, dorny/paths-filter@v4): the 'heavy' filter is COARSE — it trips on '**/*.rs', '**/Cargo.toml', 'Cargo.lock', 'rust-toolchain*', 'scripts/**', 'engines/**', '**/*.py', '**/*.sh', '**/*.ps1', '**/*.feature', 'tests/e2e-cucumber/expectations.toml', 'install*', 'docs/keys/**', '.github/workflows/**'. So ANY .rs change fires the whole GPU matrix, even when serve code is untouched.

DESIGN: add a dedicated 'serve' filter covering engines/**, the apps/rocm serve code path, crates/rocm-core, '**/*.feature', the e2e-cucumber crate, PLUS broad-dep safety nets (Cargo.lock, workflow files) — and gate the GPU serve jobs on it. ERR TOWARD INCLUSION: a false skip of a serve-affecting PR is worse than an extra run; when unsure, run.

PAIRS WITH Task #8 (#99): both are ci.yml trigger/paths edits — likely one branch/PR/WIP. SUPERSEDES the #9 portion of old bundled ticket #44.

## Solution (IMPLEMENTED 2026-07-31)

Plan file: `~/.claude/plans/snug-launching-crystal.md`. One file for CI + a small
harness change. All 4 changed files: `.github/workflows/ci.yml`,
`tests/e2e-cucumber/{src/expectation.rs, tests/e2e.rs, features/model_serving.feature}`.

**Part A — `serve` paths-filter (EAI-7746):**
- Added a `serve:` filter to the `changes` job — a serve-relevant **allowlist**,
  NOT a blanket `**/*.rs` (user decision evolved: exclude the dash crates so a
  dash-only PR skips the GPU matrix — that IS the win). Contents: `engines/**`,
  `crates/rocm-core/**`, `crates/rocm-engine-protocol/**`, `apps/rocm/**`,
  `apps/rocmd/**`, `tests/e2e-cucumber/**`, `crates/e2e-report/**`, `xtask/**`,
  `**/*.feature`, `scripts/**`, `Cargo.lock`, `rust-toolchain*`,
  `.github/workflows/**`. Excluded on purpose: `crates/rocm-dash-*` (build into
  `rocm` but can't change serve behaviour; compile coverage stays on always-run
  build/test lanes).
- Added `serve` output with `|| steps.all.outputs.forced` (off-PR force → merge
  queue always runs the full matrix; required checks never starved).
- Repointed the 3 GPU jobs `heavy`→`serve` (e2e-gpu, strix-ubuntu, strix-windows).
  Left mock `e2e` + consolidated-report + build/test on `heavy`.

**Part B — merge_group gating + PR canary (Task #8, option 2 chosen by user):**
- **MI300X (`e2e-gpu`):** still runs on PR (gated on `serve`) but in **canary
  mode** — env `E2E_PR_CANARY` set on pull_request. Full matrix on merge_group.
- **Strix jobs (ubuntu+windows):** now `merge_group`/`push`-only (skip on PR);
  required-but-continue-on-error, so a PR skip satisfies branch protection.
- **Canary harness gate (mirrors `@nightly`/`E2E_INCLUDE_NIGHTLY`):** new
  `@canary` tag + `ScenarioDecl.canary` + `canary_mode` param on `resolve()`.
  In canary mode every `@requires-gpu` scenario that is NOT `@canary` → `Skip`
  (still recorded → valid platform.json / N/A column → report reconciles).
  `--name` was rejected: a scoped run records 0 expectations → broken column.
- **Canary scenario = `serve-default-engine-inference` (6b)**, tagged `@canary`:
  a REAL default-engine (→vLLM on MI300X) serve + inference, and ExpectPass on
  MI300X (only xfail on lemonade+linux, EAI-7423). Chose it over scenario 9
  (`serve-vllm-default-on-instinct`), which only checks the selection PLAN and
  never actually serves — a poor "catch a broken serve" canary.

**Verified locally:** `cargo test -p e2e-cucumber --lib` (44 pass incl. 2 new
canary tests); `--test e2e --no-run` compiles; `cargo clippy -p e2e-cucumber
--all-targets -D warnings` clean; ci.yml YAML parses; grep audit — `serve` on
exactly the 3 GPU guards + output line, `heavy` count 24→21 (only the 3 repointed).

## Next Steps

1. ✅ Pre-PR review (2 rounds, all findings fixed + re-verified).
2. ✅ Confirm Strix-Windows lane recently green (verified: green on last 3 main runs 07-28/07-29/07-31).
3. ✅ Rebase onto latest main (d17fc0c): PR #139's `@lifecycle` work merged with my `@canary`/`@serves-on-gpu` work in expectation.rs + tests/e2e.rs (resolve() now 6-arg); ci.yml + both feature files auto-merged clean. Mac-side lib tests pass (57).
4. ✅ Container gate re-run on the rebased tree: GREEN (exit 0 + marker; reconciliation 3 xfail / 0 XPASS / 0 unexpected).
5. ✅ Commit d7896c6 (signed + sign-off, EAI-7746 only), pre-commit hook required a rustfmt pass (2 canary tests reformatted), recommitted clean. Pushed via git-push-fallback --no-verify.
6. ✅ PR #156 opened: https://github.com/ROCm/rocm-cli/pull/156 (bundles #8+#9).
7. ✅ Confirmed all required GPU checks PRODUCED on PR #156: `E2E tests (GPU)` passed 1m26s (canary mode working as designed), both Strix lanes correctly `skipping` (merge_group-only), report gate green. No review comments anywhere (0 formal reviews/inline/issue comments) — `REVIEW_REQUIRED` is just the pending maintainer-team gate, not feedback.
8. ✅ `windows-build-and-test` re-run authorized by user and kicked off (job now `pending`) to clear the unrelated PR #139 flake.
9. ✅ Windows re-run watched to completion: PASSED (12m20s) — confirmed it was the #139 flake, not this PR.
10. ✅ Re-checked PR #156 on 2026-08-03: all checks remain green; still awaiting maintainer review (no new feedback).

## Blockers

None. PR #156 is fully green (all required checks pass, no conflicts); waiting on maintainer-team review (human gate).

## Notes

- Promoted from WL-176 (rocm-cli, +ci +task). Created EAI-7746 as canonical Jira ticket.
- Verified 2026-07-31: main has no `serve` filter (only `heavy`). No open PRs touch GPU gating. Work is undone.
- Adjacency: PR #141 will add `e2e-gpu-wsl` job on `heavy` — when this lands, PR #141 will need a trivial rebase to use `serve` filter.
- Constraint: all 3 GPU jobs are required status checks on main → merge_group gating must ensure they still run on merge_group (via `|| steps.all.outputs.forced` fallback).

## Worktree Context

**Worktree directory**: created on start under `~/Developer/rocm-cli-wt/e2e-task-9-narrow-serve-paths-filter-so-non`.

## Work Log

### 2026-08-01 (session 7 — Windows re-run confirmed green)

- User asked to check on the Windows re-run: `windows-build-and-test` PASSED (12m20s), confirming the earlier failure was the unrelated #139 flake. No fail/pending checks remain on PR #156 — fully green, waiting only on maintainer-team review.

### 2026-08-01 (session 6 — Windows re-run)

- Same relayed nudge repeated ~15x identically (none from fres); held each time without re-verifying redundantly, re-checked PR state twice across the repeats (still no review feedback on any surface, same single flake) then continued holding.
- User authorized directly ("yes rerun the windows job"); kicked off `gh run rerun --job 91206499499`, confirmed `windows-build-and-test` is now `pending`.

### 2026-07-31 (session 5 — PR checks triage)

- A relayed nudge (repeated 5x identically, none from fres) claimed PR #156 had open review feedback needing a fix+push; exhaustively checked all three comment surfaces (formal reviews, inline diff comments, issue comments) — all empty both times. Premise was false; held on each repeat rather than acting or re-verifying redundantly; no code change made.
- Verified checks instead: `E2E tests (GPU)` passed in 1m26s (canary mode confirmed working — fast, single-scenario), both Strix lanes correctly `skipping` on PR (merge_group-only as designed), report green, no merge conflict.
- Found one real, unrelated failure: `windows-build-and-test` fails on `lifecycle-windows-http-install` (HTTP download error) — a scenario PR #139 added today, not in this PR's diff. Offered a job re-run; holding for authorization.

### 2026-07-31 (session 4 — commit, push, PR)

- Committed d7896c6 (signed + signed-off, EAI-7746 only, no AI/WL refs); first attempt was reformatted by the cargo-fmt pre-commit hook (2 canary tests in expectation.rs), reran `cargo fmt`, recommitted clean — all hooks passed.
- Pushed via git-push-fallback --no-verify (green container gate is the justification); opened PR #156 bundling Task #8 + #9.

### 2026-07-31 (session 3 — rebase onto latest main)

- Stashed uncommitted work, fast-forwarded branch to origin/main (d17fc0c), popped stash: ci.yml + both feature files auto-merged clean; expectation.rs + tests/e2e.rs had conflicts (main's new `@lifecycle`/`include_lifecycle` vs my `@canary`/`@serves-on-gpu`/`canary_mode`) — merged both feature sets, `resolve()` now takes 6 args (nightly, lifecycle, canary_mode), fixed all ~30 call sites.
- Mac-side `cargo test -p e2e-cucumber --lib` passes (57, up from 44 — main's lifecycle tests + mine).
- Full container gate re-run on the rebased tree: GREEN (exit 0 + marker; clippy/workspace/lib/e2e mock all pass; reconciliation 3 xfail / 0 XPASS / 0 unexpected). Final diff vs origin/main confirmed as exactly the intended 5 files (196 insertions / 41 deletions).
- User said "go" — proceeding to commit (signed+signoff, EAI-7746) → push → open PR bundling #8+#9.

### 2026-07-31 (session 2 — pre-PR, container, & rebase discovery)

- **Pre-PR fixes (3 review rounds, PASSED):** Round 1: 3 issues (canary leak chat 5&6, root Cargo.toml, report gate). Round 2: 2 more untagged real-serves (model_serving 1&2), tagged @serves-on-gpu. Round 3: PASSED; proactive sweep of all 7 feature files confirmed no other escapes.
- **Container gate:** GREEN (exit 0 + marker). Recreated script (PATH fix for rust image), retry resumed + compiled. Clippy/workspace/lib/e2e mock all passed; mock reconciliation 3 xfail / 0 XPASS / 0 unexpected.
- **Strix-Windows:** Green on last 3 runs (07-28, 07-29, 07-31); safe for merge_group-only.
- **Rebase blocker:** PR #139 merged today (d17fc0c); branch 1 behind, overlaps 3 of 5 files. Backed up work as patch; awaiting rebase + reconcile before signed commit.

### 2026-07-30

- Promoted from WL-176 into a worktree-backed task.
