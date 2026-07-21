<!-- WIP file. Personal notes on the orphan progress branch — never merged into main. -->

# WIP: Close the xfail-cleanup gap for code-only bug fixes (EAI-7478)

**Stage:** 7-PR-open (PR #134, driving CI green)
**Pipeline:** standard
**Branch:** fix-xfail-cleanup-gap
**Jira:** EAI-7478 (Bug, component rocm-cli) — https://amd.atlassian.net/browse/EAI-7478
**Last Updated:** 2026-07-21

**Token Usage:** in=490 out=129710 cache_create=1206044 cache_read=33379281 calls=248

---

## Problem

**Framing corrected by CI investigation (2026-07-20):** E2E is NOT path-skipped. There's no dedicated
`e2e` filter — E2E lanes are gated on the `heavy` filter (`ci.yml:108-119`), which INCLUDES
`expectations.toml` and `**/*.feature`. A code-only `.rs`/`.py` fix still trips `heavy` and DOES run E2E.

The real gap is **platform/hardware coverage**, not path skipping:
1. **Fix PR's lane doesn't exercise the xfail's platform.** The reconciler (`tests/e2e-cucumber/tests/e2e.rs:729-772`)
   only `exit(1)`s when an expected-xfail scenario actually PASSES on the hardware currently running. If
   the xfail is keyed `when={effective_engine="vllm"}` but the fix PR's lane resolves to lemonade, it's
   never exercised → no XPASS → merges green.
2. **Nightly swallows it.** Nightly (`nightly.yml:339-412`) runs the GPU lane + same reconciler on cron
   `0 6 * * *`, but `continue-on-error: true` (line 343) discards the `exit(1)` and nothing consumes the
   uploaded report → no issue, no signal.

The stale xfail sits dormant until an **unrelated** E2E-touching PR runs the GPU/Strix lanes and goes
red with an XPASS **mis-attributed** to that unrelated PR.

**Concrete example:** PR #94 (EAI-7052, "reuse system-installed ROCm in Lemonade") merged fully green
(its lane didn't exercise vllm). Its two `effective_engine="vllm"` xfails for
`serve-default-engine-working-endpoint` + `serve-default-engine-inference` then surfaced as XPASSes on
PR #127's GPU lane.

## Solution — DECIDED (2026-07-20): (a) + (b)

**Chosen: (a) advisory PR-time grep check + (b) a single new PR-template entry.** NOT (c), NOT CONTRIBUTING.md.

### (a) Advisory CI check
New non-blocking job in `ci.yml` (`ubuntu-latest`, `if: pull_request`, seconds, no GPU). A small Python
script (`tomllib` parses `expectations.toml` natively):
- **File side:** parse `expectations.toml`, build `{ticket_id -> [rows]}` from each row's `bug` value
  VERBATIM (no `EAI` assumption — derive IDs from the file).
- **PR side:** extract candidate IDs from PR title + body + commit messages using BOTH patterns:
  `[A-Z]+-\d+` (Jira-style, today's `EAI-NNNN`) AND `#\d+` (bare GitHub issue + `Fixes/Closes/Resolves #\d+`).
- **Match:** intersection -> step-summary warning naming ticket + scenario `@id`s + `when` engine/os.
- **Always exit 0** (advisory). No hardcoded internal names.

**Why match bare `#123` too (user-confirmed):** a bare `#123` often means "PR #123" not "issue 123", so
matching risks some false-positive nudges — acceptable because the check is advisory-only. Favor recall:
a stray nudge costs a glance; a missed stale xfail costs a mis-attributed XPASS red on an unrelated PR.

### (b) PR template
New `.github/pull_request_template.md` (none exists today) with ONE entry, ticket-neutral (no EAI):
> - [ ] If this PR fixes a bug, searched `tests/e2e-cucumber/expectations.toml` for the fixed ticket ID
>   and removed/narrowed any now-stale xfail rows.

### Not doing
No CONTRIBUTING.md change. No (c) nightly-issue-filing (deferred; own ticket if wanted). No change to the
reconciler or `heavy` filter.

### Decision recorded in project memory
`~/.claude/projects/-Users-fres-Developer-rocm-cli/memory/reference_xfail_ticket_match_bare_hash.md`

## Scenarios

All four locked scenarios have been translated into `--self-test` cases in
`scripts/xfail_expectations_hint.py` (the tests are now the single source of truth; prose removed to
avoid drift, per the agreed lifecycle):
- S1 (fix referencing a live-xfail bug is warned) → 3 self-test cases: jira ref, `Fixes #123`, bare `#123`
- S2 (referenced bug with no xfail row) → self-test "scenario 2"
- S3 (references nothing tracked) → self-test "scenario 3"
- S4 (advisory, never blocks) → the `exit 0` half is asserted in every self-test case; the "not a
  required check" half is verified when the CI job is wired (job must be non-blocking, `if: pull_request`).

## Implementation Steps

### Todo 📋
- 📋 Resolve CI dispatch backlog (Actions repo-wide, not branch-specific); merge when checks pass

### Done ✅
- ✅ Added `.github/pull_request_template.md` (single ticket-neutral xfail-cleanup checkbox)
- ✅ Wired advisory `xfail-hint` job into `ci.yml` after `changes`: ubuntu, `if: pull_request`, non-blocking,
  reads title/body via `env:` + commit msgs via `git log BASE..HEAD`, reuses pinned checkout SHA (no new actions).
  Validated: ci.yml parses, ruff clean, self-test green, real-ticket run emits the note + exit 0.
- ✅ Confirmed conventions: scripts flat in `scripts/`; test via embedded `--self-test` (mirror `release_readiness.py`); workflows read `github.event.pull_request.{title,body}`
- ✅ Wrote scenarios (bdd skill) → translated all four into `--self-test` cases
- ✅ Wrote `scripts/xfail_expectations_hint.py` + self-tests; all pass. Verified real run reproduces the
  EAI-7052 → serve-default-engine-* case from the ticket; miss silent; exit 0 always; step-summary writes
- ✅ Investigated CI: no `e2e` path filter (gated on `heavy`); reconciler at `e2e.rs:729-772` (exit 1, no issue);
  nightly runs GPU lane but `continue-on-error` swallows it; no EAI-grep, no PR template, no DoD doc exist today
- ✅ Decided approach (a)+(b); recorded rationale + bare-`#123` reasoning here and in project memory
- ✅ Fixed CodeQL path-injection alert 742: removed taint sink (`os.environ.get + open()`); emit() now prints
  to stdout, workflow redirects to $GITHUB_STEP_SUMMARY (matching ci.yml:1254 convention). Removed unused `import os`,
  self-test + ruff + YAML validation all pass.
- ✅ Committed signed+DCO (`47e74a2`) and pushed to PR #134 (1Password launched → signed via op-ssh-sign;
  repo enforces `Signed-off-by`, added via `-s --amend`). CodeQL re-analyzing.
- ✅ Verified CodeQL re-run: all 4 checks pass (0 failures). Alert 742 no longer appears on `47e74a2`;
  fix confirmed. PR #134 APPROVED by rominf (non-blocking follow-up: guard missing expectations.toml).
- ✅ Fixed reviewer follow-up: guard expectations.toml read in main(), exit 0 if absent (never-fail).
  Self-test + ruff green. Rebased 3 commits onto latest main (1881294 after PR #132 merged), re-signed,
  force-pushed with lease (head `1889e21`). Approval survived. Required checks queued; CodeQL already green.
- ✅ Reopened PR #134 to retrigger CI (initial run was stuck for 3h with zero jobs dispatched).
  Fresh run `29848018755` started; approval survives reopen. Awaiting job dispatch (repo-wide Actions backlog).

## Next Steps

- Await job dispatch on fresh run `29848018755` (repo-wide Actions backlog).
- Merge once all required checks pass.

## Blockers / Open Questions

- None open.

## Notes

- Improvement/gap-closure, not a single-line fix — deliberately kept options open in both ticket and WIP.
- Inbox origin: `~/Developer/work-ledger/inbox.md` rocm-cli item 12 (from test-e2e-diagnose review, 2026-07-20).
- Related in-flight WIP: `fix-nightly-27b-preseed` (EAI-7477) also E2E-infra; and `test-e2e-diagnose` where this was surfaced.

## Worktree Context

**Worktree directory**: `/Users/fres/Developer/rocm-cli-wt/fix-xfail-cleanup-gap`
- Recreate with: `create_worktree.sh fix-xfail-cleanup-gap`
- Base: origin/main @ 73e0fd1

## Work Log

### 2026-07-20

- Created EAI-7478 (Bug, component rocm-cli) from inbox item 12.
- Set up worktree off fresh origin/main and this WIP at stage 4-design with the three options captured.
- Investigated CI gaps (path filters, reconciler, nightly behavior); decided approach (a)+(b).

### 2026-07-21 (continued)

- Diagnosed CodeQL path-injection alert 742: inline `# codeql[...]` comment ignored by GitHub; removed taint sink.
- Refactored emit() to print stdout, updated ci.yml to redirect to $GITHUB_STEP_SUMMARY (matching convention).
  Self-test, ruff, YAML all pass.
- 1Password launched, committed signed+DCO (`47e74a2`), pushed. CodeQL re-run: all 4 CI checks green, alert 742 cleared.
- Verified PR #134 APPROVED by rominf; noted non-blocking follow-up (guard missing expectations.toml in main()).
- Fixed the follow-up: guard the expectations.toml read in main(), exit 0 if absent (never-fail contract).
  Verified missing-file case exits 0; self-test + ruff green.
- Rebased 3 commits onto latest main after PR #132 merged (1881294). Re-signed all, force-pushed with lease.
  Approval survived. Head now `1889e21`, up-to-date with main, required checks queued.
- CI run stuck (0 jobs dispatched for 3h across repo). Closed + reopened PR to retrigger. Fresh run started.
  Awaiting Actions job dispatch (repo-wide backlog, not branch-specific).
