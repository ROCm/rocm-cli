# WIP: E2E Task #9: narrow 'serve' paths-filter so non-serve Rust PRs skip the GPU matrix

**Stage:** 1-planning
**Pipeline:** lightweight
**Branch:** e2e-task-9-narrow-serve-paths-filter-so-non
**Pre-PR-check:** none
**Last Updated:** 2026-07-31

**Token Usage:** in=128 out=77615 cache_create=426156 cache_read=7113298 calls=64

---

## Problem

Parent: fix-speed-up-e2e umbrella (wlticket #47). Parent WIP: /Users/fres/Developer/rocm-cli-progress/fix-speed-up-e2e.md (Task #9). Gets its own WIP referencing the parent when work starts.

GOAL: Add a narrow 'serve' paths-filter so Rust-but-not-serve PRs (dash-only changes, unrelated crates) skip the heavy GPU serve matrix entirely.

LIVE-MAIN STATE (verified 2026-07-30, ci.yml 'changes' job, dorny/paths-filter@v4): the 'heavy' filter is COARSE — it trips on '**/*.rs', '**/Cargo.toml', 'Cargo.lock', 'rust-toolchain*', 'scripts/**', 'engines/**', '**/*.py', '**/*.sh', '**/*.ps1', '**/*.feature', 'tests/e2e-cucumber/expectations.toml', 'install*', 'docs/keys/**', '.github/workflows/**'. So ANY .rs change fires the whole GPU matrix, even when serve code is untouched.

DESIGN: add a dedicated 'serve' filter covering engines/**, the apps/rocm serve code path, crates/rocm-core, '**/*.feature', the e2e-cucumber crate, PLUS broad-dep safety nets (Cargo.lock, workflow files) — and gate the GPU serve jobs on it. ERR TOWARD INCLUSION: a false skip of a serve-affecting PR is worse than an extra run; when unsure, run.

PAIRS WITH Task #8 (#99): both are ci.yml trigger/paths edits — likely one branch/PR/WIP. SUPERSEDES the #9 portion of old bundled ticket #44.

## Solution

**Approach (awaiting user decisions):**

1. Add a narrow `serve` paths-filter to ci.yml's `changes` job (dorny/paths-filter @v4)
   - **Contents:** `engines/**`, `apps/rocm/src/therock.rs`, `apps/rocm/src/serve_summary.rs`, `crates/rocm-core/**`, `crates/rocm-engine-protocol/**`, `tests/e2e-cucumber/**`, `crates/e2e-report/**`, `xtask/**`, `**/*.feature`, `Cargo.lock`, `.github/workflows/**`
   - **Rationale:** GPU e2e jobs' unique value is `@requires-gpu` scenarios (serve + GPU chat/examine/runtime-setup). Non-GPU scenarios already covered by blocking mock lane. Dash-only changes (rocm-dash-tui/core/collectors/daemon) cannot affect serve behavior → safe to exclude. Xtask/e2e-report included because consolidation job + harness depend on them.

2. Repoint the 3 GPU jobs from `needs.changes.outputs.heavy` to `needs.changes.outputs.serve`:
   - `e2e-gpu` (MI300X, line ~732)
   - `e2e-gpu-strix-ubuntu` (line ~923)
   - `e2e-gpu-strix-windows` (line ~1080)

3. Pair with Task #8 (#99): both edit the same GPU-job `if:` guards. **User choice:** land #9 alone (then rebase #8) or bundle both in one PR (merge_group gating + serve filter).

## Next Steps

1. Clarify: bundle Task #8+#9 or land #9 separately then rebase #8?
2. Confirm excluding dash crates (max speedup) vs. conservative keep-all-Rust (minimal win)?
3. Design the ci.yml paths-filter entries (line-by-line, validate no typos).
4. Implement, test (manual ci.yml syntax validation), and open PR.

## Notes

- Promoted from WL-176 (rocm-cli, +ci +task).
- Verified on 2026-07-31: main has no `serve` filter (only `heavy`). No open PRs touch GPU gating or add serve filters. Work is undone and ready.
- Adjacency: PR #141 will add `e2e-gpu-wsl` job on `heavy` — when #9 lands, #141's serve filter will need update too (trivial rebase).

## Blockers

**BLOCKED (awaiting user):** Two decisions needed before implementation:
1. **PR scope:** Task #9 alone (then rebase Task #8 later) or bundle both #8+#9 (merge_group gating + serve filter in one PR)?
2. **Exclusion:** Exclude dash crates for max speedup, or keep conservative (any `**/*.rs` fires GPU jobs, same as today)?

## Worktree Context

**Worktree directory**: created on start under `~/Developer/rocm-cli-wt/e2e-task-9-narrow-serve-paths-filter-so-non`.

## Work Log

### 2026-07-31

- Mapped LIVE-MAIN state: ci.yml 'heavy' filter is coarse (`**/*.rs`, etc.); GPU jobs (3 self-hosted, non-blocking) fire on every PR even when serve code untouched.
- Verified work is undone: main has no `serve` filter, no open PRs add GPU gating or paths-filter narrowing.
- Analyzed scope: GPU e2e jobs' unique value = `@requires-gpu` scenarios (serve, GPU chat/examine/runtime-setup); non-GPU already covered by blocking mock lane.
- Identified serve-affecting paths: `engines/` (vllm/lemonade), `apps/rocm/src/{therock,serve_summary}.rs`, `crates/rocm-core`, protocol/xtask/e2e crates, `.feature` files.
- Confirmed dash crates (tui/core/collectors/daemon) build into rocm binary but cannot affect serve behavior → safe exclusion candidate for max speedup.
- Awaiting user decision on PR scope (bundle #8 or land alone) and exclusion strategy (exclude dash or stay conservative).

### 2026-07-30

- Promoted from WL-176 into a worktree-backed task.
