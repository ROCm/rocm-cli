# WIP: E2E Task #9: narrow 'serve' paths-filter so non-serve Rust PRs skip the GPU matrix

**Stage:** 6-implementing (complete; awaiting container gate + push)
**Pipeline:** lightweight
**Branch:** e2e-task-9-narrow-serve-paths-filter-so-non
**Jira:** EAI-7746 (Task, rocm-cli, unassigned)
**Pre-PR-check:** none
**Last Updated:** 2026-07-31
**Bundles:** Task #8 (WL-175, merge_group gating + PR canary) — same branch/PR.

**Token Usage:** in=488 out=282028 cache_create=1970397 cache_read=37258699 calls=244

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

## Next Steps (awaiting user)

1. **Blocker:** Linux container gate (repo convention; user action required).
2. **Blocker:** Confirm Strix-Windows lane recently green before relying on merge_group-only (user action required).
3. Commit (signed + sign-off, EAI-7746 in msg, no AI refs), push, open PR bundling #8+#9.
4. On PR: confirm all required GPU checks are PRODUCED (skip/run, none pending);
   scoped dispatch to confirm canary path serves only 6b.

## Blockers

**BLOCKED (awaiting user):** Container gate + Strix-Windows stability check before commit/push. All code complete and locally verified.

## Notes

- Promoted from WL-176 (rocm-cli, +ci +task). Created EAI-7746 as canonical Jira ticket.
- Verified 2026-07-31: main has no `serve` filter (only `heavy`). No open PRs touch GPU gating. Work is undone.
- Adjacency: PR #141 will add `e2e-gpu-wsl` job on `heavy` — when this lands, PR #141 will need a trivial rebase to use `serve` filter.
- Constraint: all 3 GPU jobs are required status checks on main → merge_group gating must ensure they still run on merge_group (via `|| steps.all.outputs.forced` fallback).

## Worktree Context

**Worktree directory**: created on start under `~/Developer/rocm-cli-wt/e2e-task-9-narrow-serve-paths-filter-so-non`.

## Work Log

### 2026-07-31

- Mapped LIVE-MAIN state: ci.yml `heavy` filter is coarse (`**/*.rs`, etc.); all 3 GPU jobs (self-hosted, required checks) fire on every PR.
- Verified work undone: main has no `serve` filter, no open PRs add GPU gating or paths-filter. Created EAI-7746 as canonical ticket.
- Confirmed bundling: Task #8+#9 in one PR (both edit same GPU-job `if:` guards).
- Found critical constraint: all 3 GPU jobs + consolidated report are required status checks → merge_group gating must keep them PRODUCED (forced-true off-PR; strix skip-on-PR satisfies branch protection as a skipped required check).
- **Filter decision evolved during design:** allowlist that EXCLUDES dash crates (not blanket `**/*.rs`) — a dash-only PR now skips the GPU matrix (the actual win). "Conservative" reinterpreted as "err toward inclusion within the serve surface", not "keep every .rs".
- **Canary mechanism decision (user, option 2):** `--name` can't be the canary (breaks platform.json reconciliation — 0 expectations). Added a `@canary` harness gate mirroring `@nightly`; canary = scenario 6b (real serve+inference, ExpectPass on MI300X).
- IMPLEMENTED all 4 files; verified locally (lib tests 44 pass, clippy clean, ci.yml parses, grep audit). Ready for container gate → push → PR.

### 2026-07-30

- Promoted from WL-176 into a worktree-backed task.
