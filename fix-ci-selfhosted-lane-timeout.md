# WIP: Fix CI self-hosted E2E lane timeout (offline runner holds concurrency group)

**Stage:** 4-design
**Pipeline:** lightweight
**Branch:** fix-ci-selfhosted-lane-timeout
**Pre-PR-check:** none
**Ticket:** EAI-7548 (Bug, component rocm-cli) — https://amd.atlassian.net/browse/EAI-7548
**Last Updated:** 2026-07-22

**Token Usage:** in=0 out=0 cache_create=0 cache_read=0 calls=0

---

## Problem

`ci.yml`'s shared concurrency group `CI-<ref>-shared` uses `cancel-in-progress: true`.
When a new run supersedes an older one, GitHub **cannot cancel a job queued on an
OFFLINE self-hosted runner** — so the superseded run stays "alive" holding the
concurrency group, and the new run sits `pending` with 0 jobs indefinitely.

**Impact:** seen on PR #138 (2026-07-22): a queued job on the offline `app-dev-gpu`
runner held the group; the new run never started and had to be force-cancelled by
hand (`gh api -X POST .../force-cancel`; plain `gh run cancel` does NOT reap an
offline-runner job). This blocks the merge-required lanes for anyone whose run gets
superseded while a self-hosted runner is offline.

## Solution

**Preferred fix:** add job-level `timeout-minutes` to the self-hosted E2E lanes in
`ci.yml`, so a job queued on an offline runner self-expires instead of hanging
forever — which releases the concurrency group. Small, robust, self-clears
regardless of a lane being required or advisory, and survives a lane later becoming
merge-required.

**Follow-up option (larger, separate — NOT this PR):** split the self-hosted E2E
lanes into their own workflow with their own concurrency group so an offline runner
can never stall the merge-required checks (keep `cancel-in-progress: true`). Captured
in the ticket as the bigger alternative.

Key design point: `timeout-minutes` counts from when the job starts *running*, but a
job stuck **queued** on an offline runner is what we need to reap — confirm whether
GH's job timeout applies to the queued/pending phase, or whether a separate
queue-timeout mechanism is needed. This is the main open question to resolve in
design before writing the change.

## Scenarios

(Lightweight — activate bdd-scenarios skill before finalizing.)

```gherkin
Scenario: A superseded run does not hang on an offline self-hosted runner
  Given a self-hosted E2E lane has a job queued on an offline runner
  And a newer run supersedes the current one via the shared concurrency group
  When the queued job exceeds the lane's timeout
  Then the job self-expires and releases the concurrency group
  And the newer run starts instead of sitting pending with zero jobs
```

## Implementation Steps

### Todo 📋
- 📋 Confirm whether `timeout-minutes` reaps a job stuck in the QUEUED phase on an
  offline runner (the core mechanism question), or whether GH only times out
  running jobs — determines if this fix is sufficient.
- 📋 Identify the self-hosted E2E lane jobs in `ci.yml` (1232 lines).
- 📋 Add `timeout-minutes` to those jobs with an appropriate cap.
- 📋 Validate (probe run) that a superseded run no longer hangs.

## Next Steps

Resolve the queued-vs-running timeout question first (design gate), then locate the
self-hosted lanes in `ci.yml` and add `timeout-minutes`.

## Blockers / Open Questions

- **Does `timeout-minutes` cancel a job still QUEUED on an offline runner?** If GH's
  job timeout only starts once a job is running, this fix won't reap the exact stuck
  state we saw — need to verify before committing to the approach (may push toward
  the workflow-split follow-up).

## Notes

- Related to the runner-reliability theme (offline runners): app-dev-gpu/Strix runner
  work and EAI-7447. Promoted from work-ledger inbox item 12.

## Worktree Context

**Worktree directory**: `/Users/fres/Developer/rocm-cli-wt/fix-ci-selfhosted-lane-timeout`
- Recreate with: `create_worktree.sh fix-ci-selfhosted-lane-timeout`

## Work Log

### 2026-07-22

- Filed EAI-7548 (Bug, component rocm-cli) for the offline-runner concurrency-group stall.
- Created this WIP (lightweight pipeline) and the worktree; promoted inbox item 12.
- Next: resolve the queued-vs-running `timeout-minutes` question, then edit `ci.yml`.
