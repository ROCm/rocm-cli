
# WIP: Make ci.yml manually dispatchable (workflow_dispatch)

**Stage:** 8-awaiting-pr-approval
**Pipeline:** lightweight
**Branch:** ci-manual-e2e
**PR:** #98
**Last Updated:** 2026-07-10

---

## Problem

The self-hosted GPU runners (app-dev MI300X `amd-gpu`, Strix Halo Ubuntu/Windows
`strix-halo`) can only be exercised by pushing to a PR branch, which triggers the
full CI pipeline and waits on `build-and-test`. There is no way to iterate quickly
on the E2E jobs against those runners without a full push+wait cycle, and no way
to trigger a run at all without a push.

The GitHub constraint: a `workflow_dispatch` workflow is only dispatchable if the
trigger exists on the **default branch (`main`)**. Once it does, you can dispatch
against ANY ref and GitHub runs that ref's copy of the YAML + code. So a small PR
must land the trigger on main first; after that, #69 (which owns the actual E2E
jobs) can be dispatched with `--ref test/add-e2e-robot-framework`.

## Solution

Minimal enabling PR off `main`, single-source bootstrap (bootstrap stays in
ci.yml, never duplicated — see [[workflows-self-bootstrap]]):
1. Add `workflow_dispatch:` to `ci.yml` `on:` with two `choice` inputs:
   - `platform`: all / app-dev-gpu / strix-ubuntu / strix-windows
   - `tier`: both / expect-pass / known-bugs
2. `build-and-test` gets `if: github.event_name != 'workflow_dispatch'` so a
   manual run skips the heavy gate for a fast loop.

Main has NO E2E jobs (they live only on #69), and nothing on main `needs:
build-and-test`, so skipping it breaks no downstream job here. The inputs are
declared now so #69 can key its E2E `if:` guards off them.

## Implementation Steps

- ✅ Add workflow_dispatch trigger + platform/tier inputs to ci.yml on:
- ✅ Make build-and-test skip on workflow_dispatch
- ✅ Verified: nothing on main needs build-and-test; YAML + actionlint clean
- ✅ Commit `ea117c4` (signed RSA, signed-off, no AI refs) + pushed + PR #98 opened
- 📋 Merge fast → main becomes dispatchable

## The #69 counterpart ("make space")

Tracked on the #69 WIP. #69 must mirror the **byte-identical** `on:`
workflow_dispatch block + inputs, add the same build-and-test skip, and convert
each E2E job's `if:` from `needs.changes.outputs.heavy == 'true'` to a guard that
(a) tolerates a skipped build-and-test (`always()` + result checks) and
(b) honors the platform/tier inputs on dispatch. The `on:` block MUST match this
branch exactly to avoid merge collision.

## Merge Order

1. `ci-manual-e2e` → main (this PR, fast).
2. #69 lands whenever (already has the mirrored dispatch plumbing).
3. After both: `gh workflow run ci.yml --ref test/add-e2e-robot-framework
   -f platform=strix-windows -f tier=known-bugs` → fast targeted runs.

## Notes

- Related security hardening branch `ci-harden-actions` (SHA-pinning + dependabot)
  is separate and NOT urgent — does nothing for the iteration loop.
- Verification before push: run the Linux container suite per
  [[rocm-cli-e2e-cucumber]] rule (this branch touches only YAML, so container run
  is a formality, but keep the habit).
