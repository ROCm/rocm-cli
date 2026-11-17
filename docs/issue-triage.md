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
| `platform:linux`, `platform:windows`, `platform:wsl` | read from the `Platform:` line of the issue template |
| `needs-info` plus one comment | a bug report with no reproduction steps or no `rocm --version` output |
| type (`bug`, `enhancement`, `question`, `documentation`) | model, only when the issue has no type label yet; the template's own label is never overridden |
| area (`area:cli`, `area:dash-tui`, `area:rocmd`, `area:assistant`, `engines`, `installation`, `documentation`) | model |
| `priority:P0` to `priority:P3` | model, against the rubric in the script |
| assignee | round-robin over [triage-maintainers.txt](../.github/triage-maintainers.txt): the issue number modulo the list length, skipping the reporter |

Missing labels are created on first use. An issue that already has a priority
label, `needs-info` or an assignee is left alone for that part, so re-running
the workflow does not stack labels or reassign.

## Trust model

The issue title and body are untrusted. The model is asked for a JSON object
and nothing else; the script keeps only values from fixed lists
(`validate_classification`) and decides itself which labels exist and who is
assigned. The model has no tools and sees no repository content. The job's
token can write issues and call GitHub Models, nothing else, and only the issue
number from the event reaches the script, so no issue text is interpolated into
a shell command. If the model call fails, the platform label, `needs-info` and
the assignment are still applied and a warning is logged.

The model is called through GitHub Models with the job's `GITHUB_TOKEN`
(`models: read`), so there is no secret to provision. GitHub Models is
rate-limited; an issue that hits the limit is still labelled and assigned,
just without type, area and priority.

## Changing it

- **Maintainers:** edit [triage-maintainers.txt](../.github/triage-maintainers.txt).
  Order matters, and inserting a name shifts who the next issue goes to.
- **Rubric, areas, model:** edit `SYSTEM_PROMPT`, `AREAS` and `MODEL` in the script.
  A new label also needs an entry in `NEW_LABELS`; the self-test fails if an
  area or priority label has no way to be created.

## Running it by hand

Run the workflow from the Actions tab with an issue number. `dry_run` defaults
to on and prints the plan without touching the issue. Locally:

```bash
GH_TOKEN="$(gh auth token)" python scripts/triage_issue.py --repo ROCm/rocm-cli --issue 123 --dry-run
python scripts/triage_issue.py --self-test
```

The self-test runs in the CI `python`/script step. It covers parsing,
round-robin selection, enum validation, idempotence and the needs-info comment.
It does not call the model.
