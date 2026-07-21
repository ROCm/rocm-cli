# WIP: Make XPASS non-fatal for known-flaky xfails (flaky marker)

**Stage:** 6-implementing
**Pipeline:** standard
**Branch:** fix-xpass-non-fatal-flaky (worktree active)
**Jira:** EAI-7456 (In Progress, assigned Fredrik) — https://amd.atlassian.net/browse/EAI-7456
**Last Updated:** 2026-07-21

**Token Usage:** in=806 out=184814 cache_create=2139870 cache_read=47360294 calls=368

---

## Problem

The E2E harness (`tests/e2e-cucumber/tests/e2e.rs` ~line 779–791) exits 1 on any
XPASS or unexpected-failure. That is too strict for intermittent known bugs. When
a flaky bug happens not to reproduce on a given run, its xfail scenario passes →
counts as XPASS → suite exits 1 → the required check goes red and blocks merge —
even though 0 real failures occurred. The `expectations.toml` comments already
acknowledge this ("an occasional XPASS is expected and harmless here"), yet the
harness still treats it as fatal.

**Concrete impact (live):** PR #127 (`test-e2e-diagnose`) is blocked — its GPU
lane reconciled 9 xfail, 2 XPASS, 0 unexpected failures (all 6 diagnose scenarios
passed, everything green), but the suite exits 1 purely because two EAI-7333 vLLM
scenarios (`serve-vllm-inference`, `serve-readiness-contract`) passed when
expected to fail. Pure flake drift, orthogonal to that PR's change.

**Why not just flip those to expect-pass:** EAI-7333 is genuinely intermittent —
documented flipping across MI300X runs (29404668327 passed, 29413321046 failed).
Flipping to expect-pass would cause the opposite failure (unexpected-FAIL) on the
next host/run where the bug resurfaces. The bug is still open; the xfail is
correct.

## Solution

Add a per-condition `flaky = true` marker in `expectations.toml` (parsed in
`src/expectation.rs`) that makes a scenario's XPASS **non-fatal** — still printed
in the reconciliation line for visibility, but excluded from the exit-1 decision.
Unexpected-FAIL stays fatal always. Apply `flaky = true` to the EAI-7333 vLLM
entries and the new EAI-7455 lemonade-Windows entries.

## Scenarios

_To write with the bdd-scenarios skill before implementation. Sketch:_

```gherkin
Scenario: Flaky xfail that passes does not fail the suite
  Given an xfail scenario marked flaky = true
  When it unexpectedly passes (XPASS) with no other failures
  Then the reconciliation line reports the XPASS
  And the suite exits 0

Scenario: Unexpected failure is always fatal
  Given any scenario (flaky or not)
  When it fails when expected to pass
  Then the suite exits 1

Scenario: Non-flaky XPASS remains fatal
  Given an xfail scenario without flaky = true
  When it unexpectedly passes
  Then the suite exits 1
```

## Implementation Steps

### Done ✅
- ✅ Added `flaky` field to `XfailEntry` + `Expectation::ExpectXfail` in `src/expectation.rs`; parsed from `expectations.toml` (`#[serde(default)]`, defaults false).
- ✅ Extracted the exit classification into a pure `reconcile()` + `Reconciliation` (with `is_fatal()`) in `src/expectation.rs`; `tests/e2e.rs` now calls it. Flaky XPASS printed as "(flaky, non-fatal)" and excluded from exit-1; non-flaky XPASS + unexpected-FAIL stay fatal.
- ✅ Marked EAI-7333 vLLM entries (`serve-vllm-inference`, `serve-readiness-contract`) `flaky = true`, with grammar doc in the toml header.
- ✅ Added 5 unit tests (flaky parse/default, flaky-XPASS non-fatal, non-flaky-XPASS fatal, unexpected-FAIL always fatal, all-expected clean). `cargo test --lib` green (18 passed); clippy clean.

### Todo 📋
- 📋 **READY TO COMMIT** — Code complete, all tests green, container gate green (exit 0, full mock E2E ran in Linux binary, reconciliation printed correctly).
- 📋 Commit + push, open PR (separate, per delivery decision).
- 📋 EAI-7455 lemonade-Windows entries: N/A here — they live on `fix-e2e-share-lemonade-engine`; that branch marks them flaky after this lands.

## Next Steps

1. Decide delivery: separate PR first vs. bundle into the lemonade branch (see Blockers).
2. Create branch/worktree, write scenarios (bdd-scenarios skill), implement.
3. Land, then rebase #127 and `fix-e2e-share-lemonade-engine` on top.

## Checklist

- [ ] Scenarios written and reviewed before implementation
- [ ] `flaky` parsed and defaulted false
- [ ] Unexpected-FAIL remains fatal in all cases
- [ ] Reconciliation line still prints flaky XPASS

## Blockers / Open Questions

- **Delivery decision (yours):** separate focused PR first vs. bundle into the
  lemonade branch. **Recommendation: separate PR first** — it's a cross-cutting
  reconciliation-semantics change (affects all platforms/xfails), distinct from
  either feature branch's scope. Landing it first unblocks #127 independently
  (right dependency direction) and makes the lemonade branch robust against its
  own flaky xfails.

## Notes

- **Why it matters beyond #127:** also protects `fix-e2e-share-lemonade-engine` —
  the EAI-7455 Windows xfails are themselves flaky (the lemonade daemon race is
  non-deterministic), so on a lucky run they XPASS and trip the same exit-1,
  turning the lane red in mirror image of the bug just fixed.
- Key files: `tests/e2e-cucumber/tests/e2e.rs` (~779–791, exit decision),
  `src/expectation.rs` (parse), `expectations.toml` (markers).
- Related: [[test-add-e2e-robot-framework]] (EAI-7333 context), [[fix-e2e-share-lemonade-engine]] (EAI-7455 Windows xfails).

## Worktree Context

**Worktree directory**: /Users/fres/Developer/rocm-cli-wt/fix-xpass-non-fatal-flaky (active).
- Recreate with: `create_worktree.sh fix-xpass-non-fatal-flaky`
- Container gate script copied in at `workspace/wip/container-test.sh` (gitignored).

## Work Log

### 2026-07-17

- Created WIP capturing the flaky-XPASS-non-fatal design (own branch, not started).
- Blocking #127 live; recommended delivery is a separate PR landed first, then rebase #127 and the lemonade branch on top.
- Next: user decides separate-PR-vs-bundle, then create branch/worktree and write scenarios.

**2026-07-17 (implementation & verification):**
- EAI-7456 → In Progress, assigned to Fredrik.
- Full implementation done: 3 files (+211/-38); `flaky` field + pure `reconcile()`/`Reconciliation{is_fatal()}` in expectation.rs; e2e.rs calls it; both EAI-7333 vLLM entries marked `flaky=true`.
- 5 unit tests (parse, default, fatal/non-fatal logic); `cargo test --lib` 18 passed; clippy clean.
- Container gate: warm `.cargo-container` (397M) copied from sibling; gate re-running offline. Did not complete before session end (ongoing background task).
- Next: await gate → commit → push → PR (separate, per delivery decision).

**2026-07-17 (gate verification):**
- Container gate (`container-test.sh all`) re-run with warm 397M cargo cache from sibling worktree — **GREEN** (exit 0). Full mock E2E suite ran in Linux binary; reconciliation printed: `4 xfail (failed as expected), 0 XPASS (0 flaky, non-fatal), 0 unexpected failure(s)`. Code verified.
- Ready: commit + push + open PR (separate, per recommendation).

### 2026-07-21

- **Implementation complete**: 3 files (+211/-38), `flaky` field in `XfailEntry` + `Expectation::ExpectXfail`, extracted pure `reconcile()`/`Reconciliation` in expectation.rs, e2e.rs exit decision refactored to use it.
- **Verification**: 5 unit tests (parse/default/fatal logic), `cargo test --lib` 18 passed, clippy clean, container gate green (exit 0, warm `.cargo-container` 397M from sibling, full mock E2E suite ran in Linux binary, reconciliation printed correctly: `4 xfail, 0 XPASS (0 flaky, non-fatal), 0 unexpected failure(s)`).
- **Status**: Code ready to commit/push/open PR (separate, per delivery decision). EAI-7456 In Progress, assigned Fredrik.
