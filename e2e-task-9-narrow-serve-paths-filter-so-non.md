# WIP: E2E Task #9: narrow 'serve' paths-filter so non-serve Rust PRs skip the GPU matrix

**Stage:** 6-implementing — code complete; pre-PR review passed; awaiting container gate completion + push
**Pipeline:** lightweight
**Branch:** e2e-task-9-narrow-serve-paths-filter-so-non
**Jira:** EAI-7746 (Task, rocm-cli, unassigned)
**Pre-PR-check:** review-done — OpenCode reviewer (gpt-5.6-sol), 2026-07-31, @ca9f297+e2748ac0b92b8d83 — PASSED after two review rounds (all issues fixed); short-name scenarios now @serves-on-gpu, full serve-step/feature sweep found no other real GPU serve escaping canary gating; focused tests + harness compile + clippy + git diff --check all pass.
**Last Updated:** 2026-07-31
**Bundles:** Task #8 (WL-175, merge_group gating + PR canary) — same branch/PR.

**Token Usage:** in=766 out=382753 cache_create=4888263 cache_read=69752956 calls=390

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
3. Container gate (Linux, repo convention) — IN PROGRESS: script recreated + running in background (transient crates.io timeout on 1st attempt; task b135xn1oq, status unknown — no completion record; check output file).
4. Commit (signed + sign-off, EAI-7746, no AI refs), push, open PR bundling #8+#9.
5. On PR: confirm all required GPU checks PRODUCED; scoped dispatch validates canary serves only 6b.

## Blockers

**BLOCKED (awaiting user):** Container gate background task (b135xn1oq) stopped without completion record — check output file `/private/tmp/claude-501/-Users-fres-Developer-rocm-cli-wt-e2e-task-9-narrow-serve-paths-filter-so-non/8c1b9f63-96ee-4631-a78f-1b22c9c40b6d/tasks/b135xn1oq.output`. Once gate is confirmed green, user authorization needed to commit/signed push. All code changes complete; pre-PR review passed.

## Notes

- Promoted from WL-176 (rocm-cli, +ci +task). Created EAI-7746 as canonical Jira ticket.
- Verified 2026-07-31: main has no `serve` filter (only `heavy`). No open PRs touch GPU gating. Work is undone.
- Adjacency: PR #141 will add `e2e-gpu-wsl` job on `heavy` — when this lands, PR #141 will need a trivial rebase to use `serve` filter.
- Constraint: all 3 GPU jobs are required status checks on main → merge_group gating must ensure they still run on merge_group (via `|| steps.all.outputs.forced` fallback).

## Worktree Context

**Worktree directory**: created on start under `~/Developer/rocm-cli-wt/e2e-task-9-narrow-serve-paths-filter-so-non`.

## Work Log

### 2026-07-31 (continued)

- **Pre-PR-review fixes (3 rounds, all passed):** Round 1: 3 issues found/fixed (canary leak on chat 5&6, root Cargo.toml gap, report gate mismatch). Round 2: 2 more untagged real-serves in model_serving (scenarios 1&2); tagged @serves-on-gpu. Round 3: PASSED after proactive sweep of all 7 feature files + serving_steps.rs confirmed no other real GPU serves escape canary gating.
- **Container gate:** Script recreated from memory (path issues on rust image); background task started but stopped without completion record. Output at `/private/tmp/claude-501/...`. Awaiting completion confirmation.
- **Strix-Windows stability:** Verified green on last 3 main runs (07-28, 07-29, 07-31); safe to move merge_group-only.
- **Awaiting:** container gate completion confirmation + user sign-off to commit/push.

### 2026-07-30

- Promoted from WL-176 into a worktree-backed task.
