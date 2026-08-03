# WIP: Fix CI self-hosted E2E lane timeout (offline runner holds concurrency group)

**Stage:** 4-design (design gate resolved — planned fix INVALID, see below; awaiting scope decision)
**Pipeline:** lightweight
**Branch:** fix-ci-selfhosted-lane-timeout
**Pre-PR-check:** none
**Ticket:** EAI-7548 (Bug, component rocm-cli) — https://amd.atlassian.net/browse/EAI-7548
**Last Updated:** 2026-08-03
**Token Usage:** in=94 out=71486 cache_create=845666 cache_read=4939709 calls=49

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

### DESIGN GATE RESOLVED (2026-08-03) — the planned "preferred fix" is INVALID

Two findings from the design investigation kill the original plan and reshape the task:

1. **`timeout-minutes` does NOT reap a job stuck in the QUEUED phase.** GitHub's job
   timeout timer only starts once a runner picks the job up and it is *running*; it
   never fires while a job sits "Waiting for a runner to pick up this job…" on an
   OFFLINE self-hosted runner. That queued-and-uncancellable state is exactly the
   stall we saw on PR #138, so adding/relying on `timeout-minutes` cannot clear it.
   (Sources: GitHub community discussion #50926; actions/runner #4312 — both confirm
   the timer covers running time only, not queue wait.)
2. **The lanes already have `timeout-minutes` AND dispatch is already isolated.** All
   three self-hosted E2E lanes already carry a cap (`e2e-gpu`: 90, `e2e-gpu-strix-
   ubuntu`: 35, `e2e-gpu-strix-windows`: 35) and `workflow_dispatch` already gets a
   UNIQUE concurrency group (`…-<run_id>`). Both landed in #69, before this WIP was
   written. So there is nothing to add there — the WIP's snapshot was stale.

**Remaining real exposure:** push / pull_request / merge_group still share the group
`CI-<ref>-shared` with `cancel-in-progress: true`. When a newer run supersedes an
older one whose self-hosted job is queued on an offline runner, GitHub can't cancel
that job, so the superseded run keeps holding the shared group and the new run sits
pending with 0 jobs — the #138 stall.

### Only mechanism that actually works: structural separation

Because a job queued on an offline runner is fundamentally uncancellable, the fix is
to make sure such a job never shares a run / concurrency group with the merge-required
checks. Options (scope decision needed — see Blockers):

- **(A) Split the self-hosted E2E lanes into their own workflow** with their own
  concurrency group (its own `-<ref>` key, `cancel-in-progress: true`). An offline
  runner can then only stall *that* workflow's own supersession, never the required
  lanes in `ci.yml`. Larger diff (new workflow file, move 3 jobs + the report's
  `needs`, re-wire artifact consolidation), but it's the robust structural fix.
- **(B) Scope `cancel-in-progress` so it never supersedes on the shared group in a way
  that can strand a queued self-hosted job** — e.g. keep supersession for the
  GitHub-hosted required lanes but give the self-hosted lanes a non-cancelling group.
  Requires per-job concurrency (job-level `concurrency:`), which interacts subtly with
  required-check reporting; needs careful design.
- **(C) Accept the manual force-cancel** as the documented operational remedy (the
  `gh api -X POST .../force-cancel` we already used) and close EAI-7548 as
  won't-fix-in-CI-config, since the true root cause is runner offline-ness (addressed
  by the runner-persistence work, EAI-7447 / persist-app-dev-ci-runner).

Recommendation: **(A)** is the only change that structurally guarantees the required
checks can't be stalled by an offline self-hosted runner, and it composes cleanly with
the existing report job (artifacts glob already auto-discovers platforms). But it's a
bigger diff than the ticket's original framing, so the scope needs a human call before
implementation.

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

### Done ✅
- ✅ Resolved the design gate: `timeout-minutes` does NOT reap a QUEUED job on an
  offline runner (timer covers running time only). Planned fix invalid.
- ✅ Confirmed all 3 self-hosted lanes already have `timeout-minutes` and dispatch
  already has a unique concurrency group (landed in #69).

### Pending scope decision 📋
- 📋 Human call: pursue (A) split-workflow, (B) per-job concurrency, or (C) close as
  won't-fix-in-CI (root cause is runner offline-ness, owned by EAI-7447).
- 📋 (If A chosen) create new workflow, move the 3 self-hosted jobs + report `needs`,
  wire its own concurrency group; validate a superseded shared-group run no longer hangs.

## Next Steps

**Awaiting scope decision** (see Blockers) — the original one-line `timeout-minutes`
fix is off the table; the real fix is structural (split workflow) and a bigger diff
than the ticket framed, so it needs a human go-ahead on scope before implementation.

## Blockers / Open Questions

- **RESOLVED:** Does `timeout-minutes` cancel a job still QUEUED on an offline runner?
  → **No.** The timer only starts once the job is running. Confirmed via GitHub
  community #50926 and actions/runner #4312.
- **OPEN (scope):** Which remedy — (A) split self-hosted lanes into their own
  workflow (robust, bigger diff), (B) per-job concurrency on the shared group
  (subtle, required-check interactions), or (C) close as won't-fix-in-CI since the
  true root cause is the runner being offline (EAI-7447 / persist-app-dev-ci-runner)?
  Needs a human decision before implementation.

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

### 2026-08-03

- Resolved the design gate. `timeout-minutes` does NOT reap a QUEUED job on an offline
  runner (timer covers running time only) — the planned "preferred fix" is invalid.
- Found the 3 self-hosted lanes ALREADY have `timeout-minutes` and dispatch already has
  a unique concurrency group (both from #69) — WIP snapshot was stale; nothing to add.
- Real fix is structural (split self-hosted lanes into their own workflow, option A) —
  bigger diff than the ticket framed. Surfaced to human for a scope decision; did NOT
  start implementation (design-gate stage, scope call is a human decision).
