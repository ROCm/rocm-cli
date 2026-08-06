# WIP: rocm fix fix-2-unset-override --dry-run panics rc=101

**Stage:** 9-pr-open-ci-rerunning
**Pipeline:** lightweight
**Branch:** rocm-fix-fix-2-unset-override-dry-run-panics-rc
**Pre-PR-check:** review-done (terminal) — reviewer(gpt-5.6-sol pre-PR agent), 2026-08-06, @09e1c08
**Last Updated:** 2026-08-06
**Token Usage:** in=2793813 out=614761 cache_create=7240762 cache_read=104954320 calls=657

**Gate Status:** GREEN. clippy clean, workspace/lib tests pass, and the full e2e suite ran; the single unexpected failure (`dash-managed-service-metrics`, a 30s-TTFT TUI timing scenario) was CONFIRMED a flake — it passes in isolation (scoped container re-run: 1 passed, 0 unexpected). Unrelated to the SIGPIPE change (touches only the engine-stdin write path). Revised fix committed `09e1c08` (signed) and pushed to origin.

**Status:** Sign-off check now **PASSES** ✓; all standard CI green; approval PRESERVED. "E2E tests (GPU)" first FAILED in 1s with a runner INFRA error (`System.ArgumentNullException: configuredSettings` in GitHub.Runner ConfigurationStore.GetSettings) — self-hosted GPU runner misconfigured/unregistered, NOT a code failure. Once the overall run completed I re-ran the failed job (`gh run rerun 31102600401 --failed`); GPU E2E is now re-queued/pending. Waiting on that re-run. If it fails again with the same infra error, the GPU runner needs standing up (restore-app-dev-runner skill) — surface to fres, not a code issue. Merge remains fres's call.

---

## Problem

`rocm fix fix-2-unset-override --dry-run` panics rc=101 — a dry-run should never panic. Correctness bug found while probing E2E speedups (was Task #11 of the fix-speed-up-e2e umbrella, not a speedup task). Split out to its own item 2026-07-28.
Parent umbrella: fix-speed-up-e2e (wlticket #47); parent WIP /Users/fres/Developer/rocm-cli-progress/fix-speed-up-e2e.md (Task #11).

## Solution

**Root cause: broken-pipe (SIGPIPE) panic — NOT fix-2-specific.** rc=101 is a
Rust panic. Rust installs `SIG_IGN` for SIGPIPE at startup, so when a downstream
reader closes the pipe early (`rocm fix … --dry-run | head`, or the E2E harness
truncating output), the next `println!` returns EPIPE which std unwraps into
`failed printing to stdout: Broken pipe (os error 32)` → exit 101. Surfaced by
fix-2 only because that's what the E2E probe happened to pipe; ANY subcommand
that prints hits it.

**Fix:** reset SIGPIPE to `SIG_DFL` at the top of `main()` (ripgrep/fd/bat
pattern). The process then terminates on SIGPIPE (exit 128+13 = 141), the
conventional Unix behaviour, instead of panicking. Verified empirically that
this is SAFE for the serve/daemon paths: Rust std uses `MSG_NOSIGNAL` on socket
writes, so a dropped connection still yields a normal `BrokenPipe` error, not a
signal (probe in container confirmed clean EPIPE under `SIG_DFL`).

**Follow-up (2nd pre-PR reviewer, gpt-5.6-sol, 90 conf — CONFIRMED against code):**
the process-wide `SIG_DFL` regresses the ONE place rocm writes to a spawned
child's stdin — the engine stdio path (`engine_request_with_env_root`,
main.rs:15230-15237). If the engine child exits/crashes before reading, that
write would now kill rocm with SIGPIPE/141 *before* the existing exit-status
diagnostics (bail! at ~15288-15293) can run. Socket `MSG_NOSIGNAL` does NOT
cover `ChildStdin` pipes. Verified all other stdin sites are `Stdio::null()`/
`inherit()` (parent never writes), so this is the only at-risk write.
**Fix:** wrap that write in `with_sigpipe_ignored()` (temp `SIG_IGN` + restore),
so a dead engine yields EPIPE (surfaced as "failed to write engine request",
then `wait()` reports why) instead of a signal.

Files:
- `apps/rocm/src/main.rs` — `reset_sigpipe()` (cfg(unix) libc FFI + no-op
  elsewhere), called first in `main()`; NEW `with_sigpipe_ignored()` helper
  wrapping the engine child-stdin write.
- `apps/rocm/Cargo.toml` — `libc.workspace = true` under `[target.'cfg(unix)']`.
- `apps/rocm/tests/broken_pipe.rs` — regression test: close stdout early on a
  large-output command (`completions bash`, ~120 KB > 64 KB pipe buf), assert no
  panic / not exit 101.
- `apps/rocm/src/main.rs` `mod tests` — NEW `engine_stdin_write_under_sig_dfl_is_guarded_not_fatal`:
  re-execs itself under `SIG_DFL` and asserts BOTH that an unguarded write to a
  dead child's stdin is killed by SIGPIPE (sanity) AND that the guarded write
  survives with an error. Mutation-checked: removing the guard fails the test.

## Verification (Linux container)

- ✓ Pre-fix repro: `fix fix-2-unset-override --dry-run | head` → panic, rc=101.
- ✓ Post-fix: same → exit 141 (SIGPIPE), no panic. Normal runs exit 0. 
- ✓ New test `early_pipe_close_does_not_panic` passes.
- ✓ Host disk freed ~27 GB (was 100% full, causing spurious linker failure); regenerable caches cleared from merged/on-hold worktrees.

## Next Steps

1. Wait for final CI jobs to finish (all pending, none failing). Once complete, PR #185 is fully mergeable and ready for fres to merge.

## Notes

- Promoted from WL-89 (rocm-cli, +bug +wl:89).

## Worktree Context

**Worktree directory**: created on start under `~/Developer/rocm-cli-wt/rocm-fix-fix-2-unset-override-dry-run-panics-rc`.

## Work Log

### 2026-08-03

- Promoted from WL-89 into a worktree-backed task.
- Implemented and verified SIGPIPE fix in Linux container (reset to `SIG_DFL` in `main()`, safe for daemon paths via `MSG_NOSIGNAL`); regression test added and passing.
- Recovered ~27 GB disk space (host was 100% full) by clearing regenerable build caches; full container gate (clippy + tests + e2e) now running in background.

### 2026-08-05

- Gate log reviewed: clippy + all unit/lib tests passed (green); e2e stage failed on missing release binary — root-caused to the disk-full interruption during the earlier release build (`cargo xtask e2e` does build the release binaries itself, e2e.rs:45-49), not a gate-invocation gap.
- Committed the fix to the feature branch: `6b50809` (signed), 27 lines (main.rs + Cargo.toml + Cargo.lock + broken_pipe.rs).
- Re-ran the container e2e gate (release build + cucumber suite) — GREEN: 28 passed, 3 xfail (expected), 0 unexpected failures, exit 0. The prior 20-scenario failure is gone now that the release binary builds.
- Requested independent pre-PR review — verdict PASSED (reviewer opus pre-pr agent, @6b50809+1ab80552cd4d971a): reset_sigpipe() placement/FFI/cfg-gating, workspace libc dep, and regression test all check out; no serve/daemon regression (no project code sets its own SIGPIPE handler); commit message clean. No findings ≥80.
- Pushed branch to origin (`git-push-fallback --no-verify`, HTTPS/keychain; `--no-verify` justified — Mac pre-push hook can't pass by design, container gate green). Remote at 6b50809. PR is now createable; awaiting fres to open it.

### 2026-08-06

- Guidance compliance review (Check guidance compliance agent) completed: no findings ≥80 confidence. Confirmed unsafe_code usage matches project convention (libc FFI with documented exception), workspace-level libc dependency correctly resolved, test file matches integration-test conventions, build/clippy/test all pass in clean clone. No AI/Claude references introduced; commit author verified. Four independent review passes remain (scheduled background).
- Pre-PR reviewer (gpt-5.6-sol agent) issued `changes-requested` at 90 confidence. Implemented requested enhancement: `with_sigpipe_ignored()` helper to wrap engine stdin writes, allowing SIGPIPE to be temporarily ignored so stdout/stderr writes abort with SIGPIPE but stdin writes surface as `BrokenPipe` errors instead of signals. Applied to engine subprocess stdin path. Changes in working tree pending commit and re-review.
- Confirmed the finding against the code (engine write at main.rs:15230-15237; diagnostics after wait() at ~15288-15293) and verified every other stdin site is null/inherit, so the engine path is the only at-risk write. Added a re-exec'd regression test proving both directions (unguarded write killed by SIGPIPE; guarded write survives with an error). Mutation-checked: reverting the guard to a passthrough makes the test FAIL (wait_status 13 = SIGPIPE), so it is load-bearing. Test passes on Mac; running the full container gate next.
- Full container gate completed: clippy clean, all unit/lib tests passed. E2E gate hit 1 unexpected failure: `dash-managed-service-metrics` (timing-sensitive TUI metrics test, "TTFT metrics did not appear: timed out after 30s"). Confirmed this is a suspected flake, not a regression: 2026-08-05 e2e was green (0 unexpected failures) on `6b50809` which already had the global `SIG_DFL` reset; the new narrower guard only wraps the engine-stdin write path, which the dashboard metrics scenario does not exercise. Scoped container re-run of just that one scenario in progress to confirm flake vs. regression (cheapest verification).
- Scoped container re-run of that scenario PASSED (1 passed, 0 unexpected) → confirmed flake. A first clippy `large_stack_arrays` error on the 64 KB test buffer (passed on Mac, caught by container `-D warnings`) was fixed by moving it to the heap. Committed revised fix `09e1c08` (signed) and pushed to origin (`git-push-fallback --no-verify`; remote now at 09e1c08, 2 commits on the branch). Pre-PR verdict `review-done` is terminal, gate green — ready for fres to open the PR.

### 2026-08-06 (session 4)

- fres opened PR #185 for this branch. CI checks: approved by juhovainio, all CI passing (e2e still pending), mergeable but 2 commits behind main.
- Diagnosed blocking check failure: "Commit signatures + sign-off" failed. Both commits are GPG/SSH-signed but missing `Signed-off-by:` trailers; this repo requires both. 
- Prepared rebase command (`git rebase --signoff origin/main`) to add trailers and clear BEHIND state. Confirmed force-push is fres's call (irreversible, externally visible); not running without explicit go-ahead. Task parked on hold awaiting fres's instruction.

### 2026-08-06 (session 5)

- Agent nudge: PR #185's blocking check finished and CI run completed. Re-verified status — nothing changed materially. Sign-off check still blocking; GPU E2E now finished (all green). Still awaiting fres's go-ahead to force-push the rebase.
- fres said "rebase" → authorized. Verified clean tree + responsive signer (op-ssh-sign probe OK), then `git rebase --signoff --gpg-sign origin/main`: both commits re-signed AND stamped `Signed-off-by: fredespi`, replayed onto current main (0 behind). New SHAs e075ace + 9291469 (HEAD). Force-pushed with `--force-with-lease`; remote at 9291469. Confirmed juhovainio's APPROVED review survived the force-push. CI re-triggered on the new commits (sign-off check pending → expected to pass now). Merge stays fres's call.

### 2026-08-06 (session 6)

- Polled CI status on #185 after rebase — **sign-off check now PASSES** ✓. Remaining checks (Coverage, Analyze/rust, TPN, clippy) still pending, none red. Approval preserved. PR fully mergeable once final jobs finish; merge timing remains fres's decision.

### 2026-08-06 (session 7)

- Monitored final CI jobs. All standard checks (Coverage, Analyze/rust, TPN, clippy) passed GREEN ✓. "E2E tests (GPU)" initially FAILED in 1s with runner infra error (`System.ArgumentNullException: configuredSettings`), not a code issue — self-hosted GPU runner misconfigured/unregistered.
- Waited for full run to complete, then re-ran failed GPU job (`--failed` flag). GPU E2E now re-queued/pending for retry.
- If re-run fails with same infra error again, indicates runner-provisioning issue (not this fix) — flagged for manual investigation, not a code regression.
