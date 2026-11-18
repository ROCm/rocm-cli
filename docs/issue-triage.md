<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Issue triage

[issue-triage.yml](../.github/workflows/issue-triage.yml) gives every newly
opened issue a first pass: labels, a priority and a maintainer assignee. It
does not close, edit or comment on an issue except for one `needs-info` comment
(below). The logic is in [triage_issue.py](../scripts/triage_issue.py).

## What it does

| Step | How |
| --- | --- |
| `platform:linux`, `platform:windows`, `platform:wsl` | read from the `Platform:` line of the issue template; the unedited `Linux / WSL2 / Windows` line gets none |
| `needs-info` plus one comment | a bug report written in the bug template with no reproduction steps (the template's hint text does not count) or no `rocm --version` output; a free-form report is not checked |
| type (`bug`, `enhancement`, `question`, `documentation`) | model, only when the issue has no type label yet; the template's own label is never overridden |
| area (`area:cli`, `area:dash-tui`, `area:rocmd`, `area:assistant`, `engines`, `installation`, `documentation`) | model, only when the issue has no area label yet |
| `priority:P0` to `priority:P3` | model, against the rubric in [issue-triage.txt](../.github/prompts/issue-triage.txt) |
| assignee | round-robin over [triage-maintainers.txt](../.github/triage-maintainers.txt): the issue number modulo the list length, skipping the reporter |

Missing labels are created on first use. An issue that already has a priority
or area label, `needs-info` or an assignee is left alone for that part, so re-running
the workflow does not stack labels or reassign.

## Trust model

The issue title and body are untrusted, and a model that followed them could be
talked into using whatever credential its job holds. So the workflow is three
jobs, and the credential that reaches the model is never one that can write
issues:

| Job | Token | Does |
| --- | --- | --- |
| `prepare` | `issues: read` | fetches the issue and builds the size-capped prompt; a bot or pull request produces no prompt and the rest is skipped |
| `classify` | `COPILOT_PAT` only | the model call, with no write scope on `GITHUB_TOKEN` |
| `apply` | `issues: write` | reduces the reply to values from fixed lists (`validate_classification`), then labels, assigns and comments |

The script, not the model, decides which labels exist and who is assigned. Only
the issue number from the event reaches a script, and the model reply reaches
`apply` through an environment variable, so no issue text is interpolated into a
shell command. If the model call fails, is rate-limited or returns something
unparseable, the platform label, `needs-info` and the assignment are still
applied and a warning is logged.

The model call is the one the release-readiness narrative in
[nightly.yml](../.github/workflows/nightly.yml) makes: `actions/ai-inference`
with the Copilot provider, authenticated by the Copilot-licensed PAT in the
`COPILOT_PAT` secret. Each triaged issue spends Copilot requests on that PAT's
owner. If the PAT is revoked or its seat lapses, triage degrades to the
deterministic steps without failing. GitHub Models (`models: read`) is not used:
it is retired and its endpoint no longer returns completions.

## Changing it

- **Maintainers:** edit [triage-maintainers.txt](../.github/triage-maintainers.txt).
  Order matters, and inserting a name shifts who the next issue goes to.
- **Rubric and model:** edit [issue-triage.txt](../.github/prompts/issue-triage.txt)
  and the `model:` input in the workflow. The script's `AREAS`, `TYPES` and
  `PRIORITIES` hold the same values; the self-test fails if one is missing from
  the prompt. A new label also needs an entry in `NEW_LABELS`, which the
  self-test also checks.

## Running it by hand

Run the workflow from the Actions tab with an issue number. `dry_run` defaults
to on and prints the plan without touching the issue. Locally:

```bash
GH_TOKEN="$(gh auth token)" python scripts/triage_issue.py --repo ROCm/rocm-cli --issue 123 --dry-run
TRIAGE_MODEL_RESPONSE='{"type":"bug","area":"engines","priority":"P2"}' \
  GH_TOKEN="$(gh auth token)" python scripts/triage_issue.py --repo ROCm/rocm-cli --issue 123 --dry-run
python scripts/triage_issue.py --self-test
```

The first form shows the deterministic part alone; the second supplies a model
reply by hand. The self-test runs in the CI `python`/script step. It covers
parsing, reply parsing, round-robin selection, enum validation, the prompt cap,
idempotence and the needs-info comment. It does not call the model, which is
exercised by running the workflow itself.
