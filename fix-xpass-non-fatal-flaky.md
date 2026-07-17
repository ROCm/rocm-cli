# WIP: Make XPASS non-fatal for known-flaky xfails (flaky marker)

**Stage:** 0-idea
**Pipeline:** standard
**Branch:** fix-xpass-non-fatal-flaky (not created yet)
**Jira:** EAI-7456 (Backlog) — https://amd.atlassian.net/browse/EAI-7456
**Last Updated:** 2026-07-17

**Token Usage:** in=7 out=1602 cache_create=107869 cache_read=379532 calls=5

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

### Todo 📋
- 📋 Add `flaky` field to the expectation struct in `src/expectation.rs` and parse it from `expectations.toml`.
- 📋 Update the exit-decision logic in `tests/e2e-cucumber/tests/e2e.rs` (~779–791) to exclude flaky XPASS from the fatal set; keep it in the printed reconciliation line.
- 📋 Mark EAI-7333 vLLM entries (`serve-vllm-inference`, `serve-readiness-contract`) `flaky = true`.
- 📋 Mark the new EAI-7455 lemonade-Windows entries `flaky = true`.
- 📋 Verify: a run where a flaky xfail XPASSes exits 0; an unexpected-FAIL still exits 1.

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

**Worktree directory**: not created yet.
- Recreate with: `create_worktree.sh fix-xpass-non-fatal-flaky`

## Work Log

### 2026-07-17

- Created WIP capturing the flaky-XPASS-non-fatal design (own branch, not started).
- Blocking #127 live; recommended delivery is a separate PR landed first, then rebase #127 and the lemonade branch on top.
- Next: user decides separate-PR-vs-bundle, then create branch/worktree and write scenarios.
