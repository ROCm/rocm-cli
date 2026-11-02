<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# rocm-doctor skill quality

How well [`skills/rocm-doctor`](../skills/rocm-doctor/SKILL.md) performs on a fixed
set of scenarios, where that is measured, and how to read it.

**This measures skill quality on a fixed scenario set.** It does not measure usage,
adoption or how often real users succeed. A figure can only move when the skill's
prose, its dataset ([`evals.json`](../skills/rocm-doctor/evals/evals.json)), the grading
model or the harness changes, never because of anything a user did. Do not report it
as an engagement figure.

## Where it is graded

Not in this repository. The `skill-evals` job in [`ci.yml`](../.github/workflows/ci.yml)
runs skillscope's `structural` check only: it makes no model calls and is advisory.
`routing` and `behavioral` make paid model calls and need a credential this repository
does not hold, so they run in [`amd/skills`](https://github.com/amd/skills), which
vendors this skill and pays for the runs:

1. A change to `skills/rocm-doctor/` lands on `main` here.
2. `amd/skills`' nightly `federate-skills` workflow notices that the folder's contents
   changed and opens a pull request titled "Bump `rocm-doctor` to `<commit>`". The
   vendored folder's `.federated.json` records the source commit and a content hash.
3. Its `evals` workflow runs [skillscope](https://github.com/amd/skillscope) on that
   pull request: `behavioral` for the skill, on Linux and on Windows, and `routing`
   when the skill is in the workflow's `routing_room`.
4. The result is the run's job summary.

So a figure exists for a given version of the skill when a bump pull request for it has
been graded, and not otherwise.

Running `routing` and `behavioral` from this repository's own `skill-evals` job as well,
with the credential `amd/skills` uses, is not done yet. This page describes what runs
today.

Everything in the rest of this page about what runs there was checked on 2026-10-30.
It is `amd/skills`' workflow and it can change; the files to re-read are
`.github/workflows/evals.yml`, `.github/workflows/federate-skills.yml` and
`.github/federation.json` in that repository.

## The two figures

### First-run success rate (`behavioral`)

**Rate = passed / cases**, reported with both counts.

- `cases` is every evaluation in `evals.json` that carries at least one of
  `expected_behavior`, `unexpected_behavior`, `logs_contain` or `files_exist`.
- A case passes only when the run did not error, it has at least one check, and every
  check holds.

"First run" is literal: each case runs once, in a fresh workspace, through one
`claude -p` invocation with the CLI's retry loop disabled. The harness never re-runs a
case.

Each expectation is graded one of three ways:

| Expectation | Graded by |
| --- | --- |
| `logs_contain` | case-insensitive substring match over the raw transcript |
| `files_exist` | the filesystem, matching whole path segments |
| `expected_behavior` / `unexpected_behavior` | an LLM judge, on the same model as the run |

A case is all-or-nothing, so the summary also gives the expectation-level count
(`14/14 individual expectations`). With few cases the case rate moves in large steps
and the expectation count shows the movement underneath.

### Routing accuracy (`routing`)

**Accuracy = passed / graded**, as the harness computes it: the skill's own prompts
(those that should trigger it, and the near misses) plus a set of shared negative
prompts that ship inside the harness package. A case passes when the skill fired and
should have, or when nothing fired and nothing should have.

It needs the skill in the room. A routing run installs the skills in `routing_room`
side by side and grades which one fires, so a skill that is not listed there is not
routed. `rocm-doctor` was not listed when this was checked, which is why no routing
figure exists for it yet. See [Changing what is graded](#changing-what-is-graded).

## Errored cases

The two graders count an errored case (a timeout, a judge failure or an infrastructure
fault) differently, and both rules are the harness's own:

- **Behavioral:** an errored case counts as a **failure** and stays in `cases`.
- **Routing:** an errored case is **excluded** from `graded` and `passed`.

Read an errored count beside the rate. An infrastructure failure is not a skill defect.

## What pins a run

A figure is only comparable with another when the same things were graded:

| Pin | Where it is recorded |
| --- | --- |
| The skill | `.federated.json` in the vendored folder: source commit and a content hash of the folder. The dataset is part of the folder, so a changed dataset changes the hash. |
| The harness | The ref on the `uses: amd/skillscope/...` line in `amd/skills`' `evals.yml`. Some routing cases ship inside the harness package, so changing the harness changes the denominator. |
| The model and effort | The summary line (`on opus (effort high)`). CI forces the model, and the effort is `high`. |

To read a run, open the bump pull request for the version of the skill you care about,
then its `evals` run. The behavioral jobs for `rocm-doctor` each print a summary of the
form `N/N cases passed (M/M individual expectations)` and a list of any unmet
expectations. Compare two runs only when the three pins above match.

## What is not available yet

At the time of writing:

- **A routing figure**, because the skill is not in `routing_room`.
- **A kept series.** The harness writes its JSON report on the runner, but the
  behavioral run did not upload it as an artifact, so the figure lives in the job
  summary and the run's log retention, not in a stored history.
- **A scheduled graded run.** The weekly schedule in `amd/skills` runs the free
  reference checks only, on purpose; graded runs happen when a pull request changes a
  skill.
- **A measured run-to-run spread.** The judge is non-deterministic and is the same
  model family as the run, so two runs of an unchanged skill can differ. Until the
  spread has been measured on an unchanged commit, no single movement can be called a
  regression.

## Changing what is graded

This repository owns the skill and its dataset, and a change to either is made here. It
does not own the workflows in `amd/skills` that grade them, which belong to that
repository's maintainers. So each of these is a request to them, or a pull request they
review and may decline:

- **Routing.** Add `rocm-doctor` to `routing_room` in `evals.yml`. That workflow lists
  the room by hand because adding a name moves every other skill's routing number, so
  the request should say that plainly.
- **A spread.** Run the `evals` workflow several times at one commit with
  `mode: behavioral` and `skills: rocm-doctor`, changing nothing between runs, and
  record the observed spread of the case rate and of the routing accuracy beside the
  figures. Dispatching it takes permission on that repository.
- **A kept series.** Have the report uploaded as an artifact, or collected from the
  run, so two runs from different dates can be compared without reconstructing them.

## Reading a change

- **The sample is small.** With a handful of behavioral cases, one case is a large step.
  Compare absolute counts, and do not read a one-case move as a real change.
- **These are our prompts, not the field.** Nothing here measures failures users
  actually hit.
- **The grading host has no AMD GPU and no `rocm` CLI.** The behavioral cases therefore
  grade the skill's procedure (probing first, asking for consent before anything that
  mutates, routing upstream when nothing is established), not whether a diagnosis was
  correct. A rise does not mean users get fixed faster.
- **Editing the dataset moves the figures.** A changed content hash between two runs
  means the scenario set may have changed; look at that diff before comparing them.
