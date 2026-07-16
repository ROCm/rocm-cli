<!-- This is an AI instruction file. Use this template when creating new WIP files. Fill in the placeholders. -->

# WIP: Speed up E2E test suite

**Stage:** 0-idea
**Pipeline:** standard
**Branch:** fix-speed-up-e2e
**Last Updated:** 2026-07-16

**Token Usage:** in=20 out=5587 cache_create=195033 cache_read=855738 calls=11

---

## Problem

The E2E suite is slow enough to be a drag on the dev loop and on CI (self-hosted GPU runners are serial, 30–75 min per cycle). Slow runs mean fewer iterations, longer PR feedback, and more contention for shared hardware.

TODO: quantify current runtime and identify the biggest contributors (model downloads, vLLM readiness waits, redundant setup, serial scenarios).

## Solution

High-level approach TBD after profiling. Candidate levers:
- Cache/warm model weights so scenarios don't re-download (HF Hub cold pulls are a known long pole).
- Reduce redundant per-scenario setup / share fixtures where safe.
- Parallelize independent scenarios where hardware allows.
- Trim or tier scenarios so the fast path runs on every PR and the heavy path runs less often.

## Scenarios

Define expected behaviors in BDD-style (Gherkin) **before** writing any tests or code.

```gherkin
Scenario: [Abstract behavior description]
  Given [situation/context]
  When [user action or event]
  Then [observable outcome]
```

**Always activate the bdd-scenarios skill** before writing or reviewing scenarios.

## Implementation Steps

### Completed ✅
- (none yet)

### In Progress ⏳
- (none yet)

### Todo 📋
- 📋 Profile current E2E runtime; identify the biggest time sinks
- 📋 Decide which levers to pull (caching, parallelism, tiering)
- 📋 Implement + measure improvement

## Next Steps

Profile the suite to establish a baseline runtime and find the dominant cost before changing anything.

## Checklist

- [ ] Scenarios written and reviewed before any implementation
- [ ] If this adds a user command, is there also a tool action for the agent?
- [ ] If this adds a tool action, are there tests covering LLM-facing semantics (description clarity, action disambiguation)?
- [ ] All scenarios have corresponding tests

## Blockers / Open Questions

- **Baseline unknown**: need current runtime numbers before deciding where to optimize.

## Notes

Related WIPs: [[test-e2e-tui-cucumber]], [[ci-manual-e2e]], [[persiste-app-dev-ci-runner]]. See memory `reference_rocm_cli_e2e_cucumber` for suite tiers/tags and runner gotchas.

## Worktree Context

**Worktree directory**: `/Users/fres/Developer/rocm-cli-wt/fix-speed-up-e2e`
- Recreate with: `create_worktree.sh fix-speed-up-e2e`

## Work Log

### 2026-07-16

- Created WIP file for the E2E speed-up effort.
- Established skeleton with problem statement, candidate optimization levers, and initial profiling task.
- Next: profile the suite to get a baseline and find the dominant cost.
