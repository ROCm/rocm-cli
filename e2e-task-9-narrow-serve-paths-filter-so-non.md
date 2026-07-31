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

**Approach (user decisions made 2026-07-31):**

1. **Add a conservative `serve` paths-filter** to ci.yml's `changes` job (dorny/paths-filter @v4)
   - **Contents (inclusive, err-toward-inclusion):** `**/*.rs`, `**/Cargo.toml`, `Cargo.lock`, `rust-toolchain*`, `engines/**`, `apps/rocm/**`, `crates/rocm-core/**`, `crates/rocm-engine-protocol/**`, `crates/e2e-report/**`, `tests/e2e-cucumber/**`, `**/*.feature`, `xtask/**`, `.github/workflows/**`
   - **Note:** Includes all `**/*.rs` to stay conservative (do not exclude dash crates). False skip worse than extra run; when in doubt, run.
   - **Rationale:** GPU e2e jobs' unique value is `@requires-gpu` scenarios. Compile/test coverage stays on always-run lanes (build-and-test, test, windows-build-and-test).

2. **Bundle with Task #8 (#99):** both in one branch/PR
   - Add `serve` filter here
   - Add merge_group gating + PR canary in same PR
   - One coherent ci.yml edit, no rebase collision

3. **Repoint the 3 GPU jobs** from `needs.changes.outputs.heavy` to `needs.changes.outputs.serve`:
   - `e2e-gpu` (MI300X, line ~732)
   - `e2e-gpu-strix-ubuntu` (line ~923)
   - `e2e-gpu-strix-windows` (line ~1080)
   - Add `|| steps.all.outputs.forced` fallback to ensure merge_group still fires them (required check safety)

## Next Steps

1. Read Task #8 WIP to coordinate merge_group gating design (both edit same GPU-job `if:` guards).
2. Design the full ci.yml patch: add `serve` output, consolidate GPU job guards with merge_group + serve logic.
3. Implement, validate ci.yml syntax, test manually on branch.
4. Open PR bundling both tasks.

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
- Confirmed bundling: Task #8+#9 in one PR (both edit same GPU-job `if:` guards). User chose conservative/inclusive filter (keep `**/*.rs`, don't exclude dash).
- Found critical constraint: all 3 GPU jobs are required status checks → merge_group gating must use `|| steps.all.outputs.forced` fallback to ensure they still run in queue.

### 2026-07-30

- Promoted from WL-176 into a worktree-backed task.
