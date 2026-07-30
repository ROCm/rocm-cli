# WIP: E2E Task #10: reduce mock-lane per-scenario fixed overhead

**Stage:** 5-investigating — awaiting user decision
**Pipeline:** lightweight
**Branch:** e2e-task-10-reduce-mock-lane-per-scenario-fixed
**Pre-PR-check:** none
**Last Updated:** 2026-07-30

**Token Usage:** in=326 out=142851 cache_create=869256 cache_read=24871324 calls=174

---

## PROFILING RESULT (2026-07-30) — premise NOT confirmed

Measured on **Linux** (Apple `container` — the authoritative env; Mac runtime is
OS-gated and misleading: a Mac `serve` wrongly resolves to lemonade and downloads
an ubuntu tarball, giving a false 33s on scenario 11 that does NOT happen on Linux).

Linux release `rocm` + prebuilt cucumber harness:
- **All 31 scenarios, harness binary run directly: 1.38s total.**
- Per scenario in isolation: ~0.07–0.14s each (scenario 11 "no-GPU fails fast" = 0.07s).
- `cargo test --test e2e` (warm, 0 compiling): **72.7s** wrapping that 1.38s run.
  `cargo test --no-run` (build-only, warm) = **25.4s**; re-run after = **2.1s**.
- CI "Run E2E tests" step (`cargo xtask e2e`): 98–345s across recent main runs —
  this BUILDS the release `rocm` binary (~3.5 min cold) + the harness, then runs.

**Conclusion:** there is no meaningful per-scenario fixed cost. Scenarios cost
~40ms each. Mock-lane wall-clock is dominated by one-time BUILD (release `rocm` +
harness) and cargo orchestration — none of it scales with scenario count. The
"~4.8s ×12" figure = total step time ÷ scenario count, i.e. build cost
misattributed as per-scenario cost.

**Options for fres:**
1. Close WL-177 as won't-fix/obsolete (premise refuted; no per-scenario cost to cut). No code change.
2. Re-scope to BUILD time — cache/reuse the release `rocm` build in the mock job
   (overlaps Tasks #8/#9 CI-caching territory). Not a per-scenario change.
3. Drop as lowest-priority P3.

Recommendation: (1) or (2). No source changes made — investigation only.

---

## Problem (as filed)

Parent: fix-speed-up-e2e umbrella (wlticket #47). Parent WIP: /Users/fres/Developer/rocm-cli-progress/fix-speed-up-e2e.md (Task #10). Gets its own WIP referencing the parent when work starts.

GOAL: Reduce the mock lane's fixed per-scenario overhead (~4.8s/scenario, multiplied across ~12 scenarios on the GitHub-hosted no-GPU lane).

DISTINCT FROM Tasks #5-#7 (mock/real split, EAI-7484, shipped PR #136): those MOVED scenarios off GPU onto the mock lane; this attacks the mock lane's own fixed cost per scenario (setup/teardown/fixture/process spin-up), which is now more impactful since more scenarios run there post-#136.

NEXT STEP when work starts: profile where the ~4.8s goes (binary launch, temp-dir config/data/cache setup, MockServer start, service-record planting) and cut the shared fixed cost. Lowest priority of the remaining speedup tasks (P3). SUPERSEDES the mock-lane-overhead portion of old bundled ticket #44.

## Solution

Investigation complete. Profiling shows WL-177 premise (per-scenario fixed overhead)
is unfounded. Wall-clock cost is one-time build + cargo orchestration, not
per-scenario. See **Blockers** below.

## Next Steps

Blocked pending fres decision on ticket direction: close as won't-fix, re-scope
to build caching, or drop as P3.

## Blockers

**BLOCKED (awaiting user):** WL-177 profiling complete. Decision needed:
1. Close ticket as won't-fix/obsolete (premise refuted; ~40ms per-scenario cost
   is not addressable without breaking test isolation).
2. Re-scope to mock-job build-caching strategy (cache/reuse release `rocm` build
   across scenarios; overlaps Tasks #8/#9 territory).
3. Drop as lowest-priority P3 (leave for future consideration).

See **PROFILING RESULT** section above for data.

## Notes

- Promoted from WL-177 (rocm-cli, +perf +task).

## Worktree Context

**Worktree directory**: created on start under `~/Developer/rocm-cli-wt/e2e-task-10-reduce-mock-lane-per-scenario-fixed`.

## Work Log

### 2026-07-30 (session 2)

- Profiled mock-lane per-scenario cost on Linux (Apple `container`; Mac runtime is OS-gated and misleading).
- Built Linux release `rocm` + prebuilt cucumber harness; ran all 31 scenarios in isolation and full suite.
- Measured: harness binary direct = 1.38s total; per-scenario ~40ms; scenario 11 ("no-GPU fails fast") = 0.07s.
- Measured: `cargo test` wrapper = 72.7s (build-only = 25.4s, re-run = 2.1s); CI step = 98–345s (includes release `rocm` build).
- **Conclusion:** no meaningful per-scenario fixed cost. Ticket premise (4.8s overhead) = build cost misattributed to per-scenario. Awaiting user decision on ticket direction.

### 2026-07-30

- Promoted from WL-177 into a worktree-backed task.
