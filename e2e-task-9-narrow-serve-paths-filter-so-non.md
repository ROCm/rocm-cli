# WIP: E2E Task #9: narrow 'serve' paths-filter so non-serve Rust PRs skip the GPU matrix

**Stage:** 6-implementing — pre-PR review findings, awaiting user fix authorization
**Pipeline:** lightweight
**Branch:** e2e-task-9-narrow-serve-paths-filter-so-non
**Jira:** EAI-7746 (Task, rocm-cli, unassigned)
**Pre-PR-check:** changes-requested — OpenCode reviewer (gpt-5.6-sol), 2026-07-31, @ca9f297 (dirty)
  - Canary mode still runs two non-canary real GPU serves — expectation.rs:338 only skips non-canary @requires-gpu, but chat.feature:48 & :59 are untagged and serving_steps.rs:555-568 real-serves on GPU. Add/use a tag for scenarios that may real-serve on GPU so they stay mock-covered but skip in PR canary mode. → Awaiting user authorization to fix (relay gate blocks agent Edit).
  - Root Cargo.toml can bypass the serve filter — ci.yml:137-150 omits root Cargo.toml even though workspace deps affect serving crates without necessarily changing Cargo.lock. Add root Cargo.toml (not **/Cargo.toml, preserving dash exclusions). → Awaiting user authorization to fix.
  - GPU E2E can run while its consolidated report skips — ci.yml:1281 gates only on heavy, but serve includes paths not covered by heavy. Gate the report on heavy || serve || workflow_dispatch. → Awaiting user authorization to fix.
**Last Updated:** 2026-07-31 (pre-PR review findings identified; awaiting user authorization for three fixes)
**Bundles:** Task #8 (WL-175, merge_group gating + PR canary) — same branch/PR.

**Token Usage:** in=528 out=319791 cache_create=2656210 cache_read=41098790 calls=271

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

## Next Steps (awaiting user authorization)

1. **Pre-PR fixes (three code changes required):**
   - Fix 1: Add `@serves-on-gpu` tag; canary_mode skips scenarios that may real-serve on GPU (requires_gpu OR serves_on_gpu). Tag chat scenarios 5 & 6.
   - Fix 2: Add root-only `Cargo.toml` to serve filter (not `**/Cargo.toml`, to preserve dash exclusions).
   - Fix 3: Gate e2e-report on `heavy || serve || workflow_dispatch` instead of just `heavy`.
2. Container gate (Linux, repo convention).
3. Confirm Strix-Windows lane recently green before relying on merge_group-only.
4. Commit (signed + sign-off, EAI-7746, no AI refs), push, open PR bundling #8+#9.
5. On PR: confirm all required GPU checks PRODUCED; scoped dispatch validates canary serves only 6b.

## Blockers

**BLOCKED (awaiting user):** Pre-PR-check findings (changes-requested) require three code edits before proceeding to commit/push. User must authorize edits (relay gate blocks Edit from agent).

## Notes

- Promoted from WL-176 (rocm-cli, +ci +task). Created EAI-7746 as canonical Jira ticket.
- Verified 2026-07-31: main has no `serve` filter (only `heavy`). No open PRs touch GPU gating. Work is undone.
- Adjacency: PR #141 will add `e2e-gpu-wsl` job on `heavy` — when this lands, PR #141 will need a trivial rebase to use `serve` filter.
- Constraint: all 3 GPU jobs are required status checks on main → merge_group gating must ensure they still run on merge_group (via `|| steps.all.outputs.forced` fallback).

## Worktree Context

**Worktree directory**: created on start under `~/Developer/rocm-cli-wt/e2e-task-9-narrow-serve-paths-filter-so-non`.

## Work Log

### 2026-07-31

- **Full implementation (Part A + B):** Added `serve` paths-filter + merge_group gating + `@canary` harness gate. Part A: allowlist exclude dash crates; Part B: MI300X canary on PR, Strix jobs merge_group-only, canary_mode skips non-canary @requires-gpu scenarios.
- **Local verification:** cargo test 44 pass (incl. 2 canary tests), clippy -D warnings clean, ci.yml YAML parses, grep audit passes.
- **Pre-PR review (changes-requested):** Found three real issues: (1) canary skips non-canary @requires-gpu but chat scenarios 5&6 are untagged + real-serve on GPU → need @serves-on-gpu tag + canary skip both tags; (2) `serve` filter omits root Cargo.toml (to preserve dash exclusions) → transitive workspace-dep edit can bypass filter; (3) e2e-report gates only on `heavy` but `serve` matches paths `heavy` doesn't → GPU can run while report skips.
- **Three fixes needed** before user can authorize commit: (1) add @serves-on-gpu, tag chat 5&6, update canary skip logic; (2) add root Cargo.toml to serve filter; (3) gate report on heavy || serve || workflow_dispatch.

### 2026-07-30

- Promoted from WL-176 into a worktree-backed task.
