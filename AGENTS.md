---
name: rocm-cli-oss-contribution
description: Safe, high-quality contribution workflow for upstream-facing work in rocm-cli. Activate before any upstream PR, issue, review reply, public fork push, or other external-facing GitHub action.
allowed-tools: Bash(git:*), Bash(gh:*), Bash(cargo:*), Bash(python:*), Bash(make:*), Read, Grep, Glob, Edit, Write
---

# AGENTS: Safe OSS Contribution Workflow For rocm-cli

Use this workflow for any upstream-facing work — PRs, issues, review replies, or public-fork branches — in this or any other public repository.
This is a blocking requirement before upstream-facing activity.

## 0) Activation And Scope

Activate this workflow before any of these actions:

- opening or editing an upstream PR
- opening or editing an upstream issue
- posting or editing an upstream comment or review reply
- pushing a branch intended for upstream

This workflow adds OSS safety and verification guardrails on top of normal dev flow.

## 1) Core Operating Rules

- Solve the reported problem, not an adjacent symptom.
- Be explicit about what is verified versus assumed.
- Never silently skip verification; if a check cannot run, state it clearly.
- Surface options and tradeoffs when multiple approaches are valid.
- Require explicit user approval for high-blast-radius actions:
  - force-pushes
  - opening/closing/merging PRs
  - public comments/review replies
  - changes to shared or public state

## 2) Company Identity Required, Internal Content Forbidden

Identity requirements stay intact:

- use your employer's author/committer identity (when required)
- include DCO sign-off (`git commit -s`)
- keep commit signing enabled when configured; do not bypass with `--no-gpg-sign`, `-n`, or `--no-verify`

Signing and sign-off are now **enforced**, not just policy: local prek hooks
check signing config on every commit (`commit-signing-configured`) and verify
the full range on push (`verify-commits`), and a blocking CI gate
(`commit-signatures`) re-checks every PR with GitHub "Verified" status. See
`docs/commit-signatures.md`.

Content restrictions for upstream surfaces:

- do not include internal/proprietary names, aliases, URLs, hostnames, gateways, cluster names, or registry paths
- do not include internal ticket identifiers, links to internal tracking systems, or unrelated internal usernames
- apply this rule to PR titles/bodies, issue text, comments, review replies, commit messages, branch names, code comments, fixtures, and logs

Use neutral external framing (for example: "backend" or "gateway") rather than internal or vendor-specific ownership phrasing.

Leak scan before each upstream push/PR/comment batch:

```bash
INTERNAL_KEYWORDS_PATTERN='internal|confidential|proprietary|private|ticket-[0-9]+|jira|confluence|\.corp|\.internal'
git diff <upstream-base>..HEAD \
  | grep -inE "$INTERNAL_KEYWORDS_PATTERN" \
  && echo "REVIEW each hit" || echo "diff clean"
```

This default pattern is intentionally generic so the workflow runs as-is. Refine `INTERNAL_KEYWORDS_PATTERN` with your organization's internal names and systems. Also manually review non-diff text surfaces (PR body, comments, issue text, branch name).

## 3) Reproduce First, Then Fix The Actual Issue

- reproduce the reported issue before claiming root cause or fix
- verify that the same reproduction passes after the change
- test boundary paths (non-default config, larger inputs, alternate trigger paths)
- do not relabel an adjacent improvement as "the fix" if the original repro still fails

Every bug fix ships with a regression test in the same change:

- test fails before fix and passes after fix
- choose unit/integration/e2e level based on where the bug lives
- if an e2e cannot run in default CI, state the gap in PR text and cover at another CI level

## 4) Live State Verification Before Any External Claim

Before each stateful decision or public status update:

- refresh remote state (`git fetch`)
- re-check PR status and checks with live `gh` queries
- verify review context against current PR head commit

Do not rely on stale memory, partial CI views, or prior snapshots.
Subagent reports are hypotheses until directly re-verified. When re-verifying, match the verification scope to the claim: if subagent claimed "tests pass", re-run the same test suite; if it claimed "no conflicts", do the rebase locally; if it claimed "leak-free", re-run the scan.

After rebase/cherry-pick/merge, grep for conflict markers:

```bash
grep -rn "^<<<<<<<\|^=======\|^>>>>>>>" .
```

For leak scans, use upstream base `origin/main` (or upstream default branch if different):

```bash
INTERNAL_KEYWORDS_PATTERN='internal|confidential|proprietary|private|ticket-[0-9]+|jira|confluence|\.corp|\.internal'
git diff origin/main..HEAD \
  | grep -inE "$INTERNAL_KEYWORDS_PATTERN" \
  && echo "REVIEW each hit" || echo "diff clean"
```

## 5) Investigate rocm-cli Before Editing

Understand existing patterns first:

- contributor and behavior docs in `README.md`, `docs/`, and `skills/`
- workspace topology from `Cargo.toml`
- sibling implementations in `apps/`, `crates/`, and `engines/`
- existing tests and conventions in `docs/testing.md`

Fix at the correct layer (root cause), not by shrinking symptom visibility.
If approach choice is ambiguous, present alternatives and recommend one.

## 6) rocm-cli Architecture Guardrails

Current workspace members:

- apps: `apps/rocm`, `apps/rocmd`
- shared crates: `crates/rocm-core`, `crates/rocm-engine-protocol`
- engine crates: `engines/atom`, `engines/lemonade`, `engines/llama-cpp`, `engines/pytorch`, `engines/sglang`, `engines/vllm`

Guardrails:

- `crates/rocm-engine-protocol` is a contract surface; verify all impacted engines after protocol changes
- preserve strict GPU-required behavior; do not introduce silent CPU fallback
- respect platform gates (for example, native Windows handling for vLLM/SGLang)
- supported host platforms are Windows and Linux only (including WSL where documented)
- platforms outside Windows/Linux are unsupported; do not implement, debug, or "fix" unsupported-platform behavior
  - if a test fails only on unsupported platforms (e.g., macOS), skip or mark as out of scope; do not alter logic to make it pass
  - add a comment documenting why the test is skipped (e.g., `#[cfg_attr(not(target_os = "linux"), ignore)]`)

## 7) Local Assistant And Tool-Use Policy Consistency

When changing assistant-adjacent behavior, keep consistency with:

- `docs/llm-tool-use.md`
- `skills/rocm-cli-assistant/SKILL.md`

Required consistency points:

- inspect state before proposing mutation
- mutating actions require approval flow
- avoid invented shell/package-manager commands in assistant behavior paths
- preserve built-in assistant constraints and no-CPU-fallback policy

## 8) Verification Matrix For This Repo

Minimum quality gate before upstream-ready status:

```bash
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
python scripts/smoke_local.py
```

When relevant to touched behavior, also run targeted checks from `docs/testing.md`, such as:

- focused Rust test groups for touched modules
- engine-specific GPU self-tests (`--self-test`) or live GPU tests when hardware is available
- release gate checks for release-path changes (`python scripts/single_exe_release_gate.py`)

**Execution environment expectations:**

- By default, tests run on the developer's local machine (Linux or Windows, GPU optional but recommended for engine tests)
- GPU tests must pass on hardware if available locally; otherwise, they are expected to fail gracefully with clear messaging
- CI gates run on dedicated hardware and may have different results than local runs; use CI as the authoritative verification
- If a required check cannot run locally (e.g., GPU not available), say exactly what was not run and why in PR description

## 9) Release Trust And Signing Requirements

For release-affecting changes, follow `docs/release-trust.md`. In brief:

- checksum sidecars are required
- detached signatures are required when release policy mandates them
- do not bypass required signature modes
- preserve metadata/index signature verification behavior when enabled

Production signing and trust roots are owner-controlled inputs; do not substitute ad hoc keys as production trust anchors.

See `docs/release-trust.md` for the full policy, key management, and signing workflows.

## 10) Vendored Upstream Trees

If a vendored upstream tree is introduced in the future, apply the following rules:

- keep vendored changes minimal and attributable
- do not assume root workspace checks cover vendored workspace behavior
- follow sync notes in the appropriate upstream-sync documentation (e.g., `docs/upstream-codex-sync.md`) for pin updates and rebuild workflow

## 11) PR/Issue/Review Conduct

- keep each PR scoped to one logical change
- write for maintainers unfamiliar with local context
- avoid AI-generated boilerplate footers
- do not resolve reviewer threads you did not author; reply with fix commit context
- if reviewed code must be updated, explain what changed since review

**Stacked and dependent PRs:**

- Keep stacked PRs in draft until dependencies merge upstream (i.e., this repo's main branch, not just local)
- After dependencies merge, rebase and move PR out of draft
- Use `git rebase -i` for meaningful commit messages; prefer individual commits over squash unless the PR is a single logical unit
- In PR body, clearly link to dependencies (e.g., "Depends on #123") and reference the target branch

## 12) CI Ownership: Drive To Green

Do not stop at opening/updating a PR.
Watch checks to completion and drive to all-green.

- inspect failing logs directly
- fix real regressions from your change
- handle infrastructure flakes by rerun or maintainer escalation with evidence
- ensure flakes are not hiding real code failures in other checks

A red check means "not ready" until resolved.

## 13) Upstream Pre-flight Checklist

Run this checklist before any upstream push, PR update, issue update, or public reply:

1. Live state refreshed (`git fetch`, fresh PR/check status query) — *§4*
2. Project conventions re-checked for touched area — *§5*
3. Issue reproduced; fix validated against same repro plus boundaries — *§3*
4. Regression test added and passing at the appropriate level — *§3*
5. Required local gates run (or explicitly documented gaps) — *§8*
6. Leak scan completed across diff and non-diff text surfaces — *§2*
7. Employer identity (if required), DCO sign-off, and commit-signing requirements preserved — *§2*
8. Scope remains one logical change at correct layer — *§11*
9. CI watched to completion and driven to green — *§12*
10. User approval obtained for any high-blast-radius external action — *§1*

## 14) Related Internal Workflows

Use existing internal mechanics/workflows for implementation details such as commit, push, rebase, PR operations, and review-and-fix/dev-cycle automation where available.

## 15) Enforcement And Feedback

This workflow is advisory and human-enforced. Violations are addressed through:

- **Pre-push review**: Activate this workflow before pushing; review each step
- **Code review**: Upstream maintainers may request alignment if issues are found
- **CI gates**: Section 8 checks block CI; ensure all pass before opening PR
- **Incident response**: If a leak or policy violation reaches upstream, document the root cause and adjust workflow

This file defines OSS safety, verification, and publication guardrails for rocm-cli.
