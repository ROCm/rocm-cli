<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Nightly E2E on the shared GPU queue

The nightly GPU lanes can run on a shared queue of GPU workers: one lane per
GPU model in the queue's fleet. The run itself happens next to the queue, which
submits the lanes, waits for them and collects their results.
[`queue-run.yml`](../.github/workflows/queue-run.yml) starts a run of one
branch, waits for it and publishes it here: its lane matrix in the job log and
summary, the lanes' files as the `queue-run-results` artifact, and a pass or
fail. [`nightly-queue.yml`](../.github/workflows/nightly-queue.yml) publishes
a run that has already finished.

This gates nothing yet. The scheduled GPU lanes in
[`nightly.yml`](../.github/workflows/nightly.yml) remain the nightly coverage
gate described in [ci-hardware-testing.md](ci-hardware-testing.md). Three
things must happen before this could replace them: a lane must be seen to
build and run the suite from a cold start in the lane image; the published
verdict must become trustworthy enough to gate on — see "What this does not
protect yet" below; and the lanes must cover what those jobs cover. A lane is
one Linux container testing the `release` channel (see "What a run
executes"), while `nightly.yml` also tests the `nightly` channel and native
Windows and WSL.

This page is the **contract** between this repository and the queue side: what
a lane must do to be reported, and how to read the report. The queue side —
how runs are started and executed, and the `queue-publish` command the
workflows call — is maintained privately with the queue, not in this
repository.

## Starting a run

Dispatch `queue-run.yml` with the branch to test (`main` by default); it also
runs every night at 02:17 UTC on `main`. The job asks the queue side for a run
of that branch, named for the job's run id and attempt, so re-running the job
starts a new run. It waits for every lane to finish, up to about five hours,
then publishes the run as below. The job log starts with the run's inputs
(below). Only one run goes at a time: a second
dispatch waits for the first, and the queue side refuses one while another,
however started, is unfinished. GitHub keeps at most one job waiting, so a
third dispatch, or the nightly schedule, replaces a dispatch still waiting
and that one never runs. The branch must exist in this repository;
otherwise the job fails at once.

## Publishing a finished run

Dispatch `nightly-queue.yml` with a run's `request_id` — the id its
`queue-run.yml` job printed — to publish that run again, for instance when that
job was cancelled while the run went on. It never starts a run, and there is
no default: a run that did not happen cannot be published in its place. It
fails, rather than reporting green, when:

- the named run has not finished, or never existed
- the run failed, could not start, was cancelled before it finished, or
  discovered no lanes at all

Each new run removes results older than two weeks, so an older `request_id`
may no longer be publishable; an uploaded artifact follows this repository's
own artifact retention.

## What a run executes

Every run is asked for: by `queue-run.yml`, or by an operator on the queue
side. A run has six inputs; the queue side sets their defaults, and any of them
can be overridden for a single run (`queue-run.yml` sets the branch):

- the branch — the branch of this repository to test, `main` by default. The
  run resolves it to a commit once, as it starts, and every lane checks out
  that commit, so all lanes test the same code however long each waits for a
  GPU. The report's first line names the commit. Only a branch of this
  repository can be named, never a commit id. A lane that cannot check the
  commit out fails, with the reason in its `lane.log`.
- the lane image — the container every lane runs
- the lane command — a single line run by `bash` under `set -euo pipefail` in
  that image, from the root of the checkout, with the commit in
  `ROCM_CLI_COMMIT`. The default is
  [`scripts/run-e2e-lane.sh`](../scripts/run-e2e-lane.sh), which builds
  and runs the full suite from a cold start, so what a lane runs is changed
  here, by a pull request, like any other test
- the blocking lanes — which lanes can fail the run; left empty, every lane can
- the excluded lanes — GPU models to leave out of this run
- the wait — how long the run waits for its slowest lane before cancelling
  what is left, which is then reported as **incomplete**; at most 15000
  seconds, and a run given more does not start

The lane set itself is discovered, not listed: one lane per GPU model available
in the queue's fleet, minus any excluded. New hardware appears without any
edit, and hardware that is away does not produce a red lane — though a
blocking lane that was not discovered is reported as a warning, since that run
did not test it. A run that discovers no lanes at all fails.

A discovered lane that is not blocking is advisory: reported, but unable to
fail the run. Promoting it is a deliberate edit once it has run clean.

Every lane runs the same image and command; there is no per-lane variant.

## Output

**A lane must write everything it wants reported directly into `/out`.**

- Each lane's `/out` lands in the workflow's artifact, `queue-run-results`
  from `queue-run.yml` or `nightly-queue-results` from `nightly-queue.yml`,
  one directory per lane, next to that lane's `lane.log` (the command's combined
  output).
- Files keep their paths under `/out`, subdirectories included. `lane.log` is
  written by the harness; a lane must not use that name. Names ending in
  `.quarantine`, hidden files and anything under a hidden directory are never
  published.
- Anything a lane leaves elsewhere is not published. The queue records more
  than the suite asked for, including details of the machine that ran it, and
  the published results are public.
- Collected files are also scanned, and a file containing an identifier the
  queue side treats as infrastructure — any UUID, a private overlay-network
  address, a few internal names — in its content or its path is withheld
  whole, a directory so named with everything in it, counted as one; the
  report's `scanned … withheld …` line says how many. The scan is a backstop, not the
  control: only writing to `/out`, and nothing identifying, is.

Widening what is collected is not the fix for a missing file; writing the file
to `/out` is.

A lane passes when its command exits zero. Capturing the command's output into
`lane.log` does not affect that status.

### Reading the matrix

The report has one row per lane and a summary. A lane is not simply pass or
fail, and the distinction matters when deciding whether to chase something:

| Outcome | Meaning | Fails the run? |
|---|---|---|
| **passed** | the command exited zero | no |
| **failed** | the lane's command or its container failed; lanes are not retried, so one failure is final. `lane.log` shows a test failure; a lane whose container never started, for example because its image could not be pulled, has no `lane.log` | yes |
| **incomplete** | no result: cancelled at the run's deadline, or no worker in the fleet could take it | yes |
| **missing** | discovered but never submitted | yes |
| **advisory** | any outcome but passed, on a lane that is not blocking | no |

The matrix is drawn before the lanes' files are collected, so a lane that
passed but whose files did not all come back, or that came back with nothing
from `/out` (every lane that runs writes `lane.log` there), is still counted
as passed in it. A `no result` line after the matrix names each such lane, and
that line is what counts: a blocking lane without a result fails the run.

**incomplete** and **missing** fail the run because a blocking lane without a
result is not a pass, but they point at queue capacity or infrastructure
rather than at a regression.

## What the workflow requires

- a self-hosted Linux runner registered with the `queue-report` label alone,
  without GitHub's default `self-hosted`, OS and architecture labels, so only a
  job naming `queue-report` lands on it, providing
  `/opt/queue/bin/queue-publish` and read access to the queue side's results
- a registered runner name and host name that do not identify internal
  infrastructure, since GitHub prints both into every job log

Both workflows require **no secret** and no permission, and contract tests in
[workflow_contract.rs](../xtask/src/workflow_contract.rs)
(`queue_nightly_publisher_holds_no_credential`, `queue_run_holds_no_credential`)
keep it that way.

That test reads the workflow files on `main`. A workflow on another branch of
this repository can still name the runner's label, which is one reason the
runner is treated as untrusted below. On `main`, the tests also refuse any
other workflow that names the `queue-report` label or these workflows, so none
sends a job to the runner or waits on them by mistake. The runner itself
refuses, before any step runs, every job that is not one of these two
workflows as it is on `main`. Whether this
workflow's check is required before merging is a repository setting the test
cannot see. It must not be: a manual-only workflow never reports on a pull
request, so requiring it would block every merge.

## Why it works this way

**The run happens next to the queue, not in a GitHub job.** Submitting work to
the queue needs a credential, and a GitHub job is the wrong place to hold one
here. The runner is a persistent self-hosted machine on a public repository,
and runner labels are not an access control: any job that names the label can
run on it, including one from a branch or, after approval, a fork. Such a job
could read whatever the runner holds, or tamper with it to capture what the
next job receives. Approval gates on environments only govern jobs that name
the environment, not jobs that run next to one. Restricting a runner to a single
workflow, or making it ephemeral, needs organisation-level configuration or a
token-minting credential. So the queue credential stays with the queue side,
and the runner has no credential for the queue or the cluster, and no sudo.

**What this does not protect yet: the report's integrity.** The runner keeps
its own registration with GitHub between jobs, so a published verdict is only
as trustworthy as every job that has run on that runner. Such a job can also
read every retained run's results, including runs started by hand that were
never published. None of this reaches the queue or its credential. Each job also receives GitHub's own short-lived
runtime token, which a tampered runner could misuse within that job's reach,
such as writing build caches. Together these are why this workflow must not
gate anything until the runner is single-use (a fresh registration per job),
or the queue side signs what it reports and something other than this runner
checks the signature: a tampered runner would skip its own check.

**Results move by a shared volume, not a network call.** The queue side writes
each finished run's publishable files; the runner reads them read-only. No
credential is needed between the two, and the only call the runner makes is
the request for a run.

**A run is asked for, not submitted, from here.** `queue-run.yml` holds no
credential and sends nothing to the queue: the runner, which sits next to the
queue side, asks it for a run of a branch, and the queue side starts it with
its own credential. It believes the request because of where it comes from,
which is why the runner runs only these two workflows from `main`. The worst a
request can do is start a run of a branch already in this repository.

**Only `/out` is published, and a missing run is never green.** Publishing by
origin is the main control on what reaches public results, with the scan
behind it. An unfinished run fails the job, and a job only ever publishes the
run it started or was given, so a nightly that stops running cannot keep
reporting an old pass: the missing night shows as a missing scheduled job.

**Nothing public names the queue's infrastructure.** Runner names, labels,
paths, branches and this page use neutral names, per
[AGENTS.md](../AGENTS.md)'s rules for public content.
