# WIP: rocm fix fix-2-unset-override --dry-run panics rc=101

**Stage:** 7-review-ready
**Pipeline:** lightweight
**Branch:** rocm-fix-fix-2-unset-override-dry-run-panics-rc
**Pre-PR-check:** none
**Last Updated:** 2026-08-05
**Token Usage:** in=9457 out=166908 cache_create=2386652 cache_read=36748415 calls=298

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

Files:
- `apps/rocm/src/main.rs` — `reset_sigpipe()` (cfg(unix) libc FFI + no-op
  elsewhere), called first in `main()`.
- `apps/rocm/Cargo.toml` — `libc.workspace = true` under `[target.'cfg(unix)']`.
- `apps/rocm/tests/broken_pipe.rs` — regression test: close stdout early on a
  large-output command (`completions bash`, ~120 KB > 64 KB pipe buf), assert no
  panic / not exit 101.

## Verification (Linux container)

- ✓ Pre-fix repro: `fix fix-2-unset-override --dry-run | head` → panic, rc=101.
- ✓ Post-fix: same → exit 141 (SIGPIPE), no panic. Normal runs exit 0. 
- ✓ New test `early_pipe_close_does_not_panic` passes.
- ✓ Host disk freed ~27 GB (was 100% full, causing spurious linker failure); regenerable caches cleared from merged/on-hold worktrees.

## Next Steps

1. Confirm e2e re-run is green (release build + cucumber suite running in container).
2. Request independent pre-PR review; do NOT open the PR (fres's action).

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
- Re-running the container e2e gate (release build + cucumber suite) now that disk is free; pre-PR review to follow once green.
