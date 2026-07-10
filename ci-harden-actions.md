
# WIP: CI security hardening (SHA-pin actions + Dependabot)

**Stage:** 8-awaiting-pr-approval
**Pipeline:** lightweight
**Branch:** ci-harden-actions
**PR:** #99
**Last Updated:** 2026-07-10

---

## Problem

Self-hosted runners on a public repo are a supply-chain risk: actions referenced
by mutable tags (`@v6`, `@v1`) can be hijacked (tag force-moved to malicious code),
which then runs on our runners — including the self-hosted GPU boxes. Several repo
Actions settings were also more permissive than needed.

## Solution

**This PR (code):**
- SHA-pin all remaining tag-referenced actions across ci.yml/nightly.yml/release.yml
  (checkout→v6.0.3 df4cb1c, cache→v5.1.0 caa2961, setup-rust-toolchain→v1.17.0 166cdcf,
  dtolnay/rust-toolchain→1.96.0 c0e9df8), each with a `# vX` comment.
- Add `.github/dependabot.yml` (github-actions, weekly) so pins stay current.

**Already applied this session (repo settings, admin API — NOT in the diff):**
- Default GITHUB_TOKEN permissions: write → **read**
- Actions can create/approve PRs: true → **false**
- Fork-PR approval: confirmed already "all external contributors"

## Implementation Steps
- ✅ Rebased branch onto current origin/main (was 18 behind; create_worktree.sh
  branched off stale local main — same gotcha as [[ci-manual-e2e]]).
- ✅ SHA-pinned 4 unique actions across 3 workflows (commit `e605a59`).
- ✅ Added dependabot.yml (commit `855dfc7`).
- ✅ actionlint clean (only pre-existing nightly.yml shellcheck SC2129 style nit + self-hosted labels).
- ✅ Opened PR #99 with hardening-summary note (settings changes + follow-ups).

## ⚠️ FOLLOW-UP REQUIRED — enable sha_pinning_required (do NOT forget)

After #99 merges, the repo setting `sha_pinning_required` is what actually
ENFORCES pinning going forward. It is currently **off** — pinning today is
voluntary, so a future PR could reintroduce a tag ref and nothing would stop it.

**Enable order (strict):**
1. Merge #99 → main is fully pinned.
2. Rebase every other open branch onto pinned main (#69, #98) so they pick up the
   pins.
3. THEN flip it:
   `gh api -X PUT repos/ROCm/rocm-cli/actions/permissions -f enabled=true -f allowed_actions=all -F sha_pinning_required=true`

**Why the order matters:** once on, GitHub REJECTS any workflow run on a ref whose
workflows still reference a tag/branch (not a 40-char SHA) — the run fails at
startup. Flip it before a ref is pinned and that ref's CI breaks (main + all open
PR CI). Nothing is permanently damaged; runs are just rejected until pinned.
Tracked as task #4.

## Other follow-ups (NOT this PR)
- Optional: `allowed_actions: all` → `selected`.
- Runner-host hardening: ephemeral runners + non-privileged service account.
- Re-check the 4 pinned SHAs still match their intended versions if main advances
  before merge; Dependabot handles ongoing bumps once merged.

## Notes
- Only 3 workflow files, no composite actions — scope fully bounded.
- Not urgent (does nothing for the manual-dispatch iteration loop); parked behind
  [[ci-manual-e2e]] / [[test-add-e2e-robot-framework]] work.
