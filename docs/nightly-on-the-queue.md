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

The queue's lanes cover the Linux GPU nightly: one lane per GPU model, each
running the full suite on both the `release` and the `nightly` channel, as the
GPU jobs of [`nightly.yml`](../.github/workflows/nightly.yml) do (see
[ci-hardware-testing.md](ci-hardware-testing.md)). Their results are trusted the
way those jobs' results are: the queue runs on the same infrastructure and under
the same access as the self-hosted runners, and neither signs what it reports.

The lane set is whatever GPU models the queue has. The 11-13 run
([43351651832](https://github.com/ROCm/rocm-cli/actions/runs/43351651832)) discovered `mi300x`, `mi325x`, `mi350p`, `mi350x`,
`mi355x`, `r9700s` and `w7900-dual`; `w7900-dual` (W7900) is advisory, and it is
the only scheduled coverage of that hardware, since `e2e-selfhosted.yml` no
longer has a W7900 lane.

**The remaining gap is Strix Halo.** The queue has no Strix lane and runs Linux
pods, so the Strix Halo nightlies (Ubuntu, Windows and WSL) stay on the shared Strix pool,
in `nightly.yml` and [`e2e-selfhosted.yml`](../.github/workflows/e2e-selfhosted.yml);
nothing here replaces them. Which checks are required before merging is a repository
setting, outside this page.

This page is the **contract** between this repository and the queue side: what
a lane must do to be reported, and how to read the report. The queue side —
how runs are started and executed, and the `queue-publish` command the
workflows call — is maintained privately with the queue, not in this
repository.

## Starting a run

Dispatch `queue-run.yml` with the branch to test (`main` by default); it also
runs every night at 02:17 UTC on `main`. The job asks the queue side for a run
of that branch, named for the job's run id and attempt, so re-running the job
starts a new run. Each lane may wait up to 4 h 10 min (15000 s, the run's wait limit below)
to be started, then run for up to 3 h (10800 s, its run limit below), however
long it waited. The job waits until every lane has finished or been
cancelled, then publishes the run as below. The job log starts with the run's inputs
(below). Only one run goes at a time: a second
dispatch waits for the first, and the queue side refuses one while another,
however started, is unfinished. GitHub keeps at most one job waiting, so a
third dispatch, or the nightly schedule, replaces a dispatch still waiting
and that one never runs. The branch must exist in this repository;
otherwise the job fails at once.

## Publishing a finished run

Dispatch `nightly-queue.yml` with a run's `request_id` — the id its
`queue-run.yml` job printed — to publish that run again, for instance when that
job was cancelled while the run went on. Such a run was never finished, so if
its lanes are done the queue side finishes it first. Do that within a day of
its lanes finishing: the queue keeps a lane's files for about a day, and a run
not finished by then loses them. It never starts a run,
and there is no default: a run that did not happen cannot be published in its
place. It
fails, rather than reporting green, when:

- the named run has not finished, or never existed
- the run failed, could not start, was cancelled before it finished, or
  discovered no lanes at all

A finished run's results are kept for two weeks: each new run removes older
ones, so a finished run's `request_id` can be published again until then. An
uploaded artifact follows this repository's own artifact retention.

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
  once and runs the full suite from a cold start on each channel, so what a
  lane runs is changed here, by a pull request, like any other test
- the blocking lanes — which lanes can fail the run; left empty, every lane can
- the excluded lanes — GPU models to leave out of this run
- the wait — how long a lane may wait to be started before it is cancelled and
  reported as **incomplete** (no capacity); at most 15000
  seconds, and a run given more does not start
- the run time — how long a started lane may run, counted from its start,
  before it is cancelled and reported as **incomplete**; at most 10800
  seconds, and a run given more does not start

The lane set itself is discovered, not listed: one lane per GPU model available
in the queue's fleet, minus any excluded. New hardware appears without any
edit, and hardware that is away does not produce a red lane — though a
blocking lane that was not discovered is reported as a warning, since that run
did not test it. A run that discovers no lanes at all fails.

A discovered lane that is not blocking is advisory: reported, but unable to
fail the run. Promoting it is a deliberate edit once it has run clean.

Every lane runs the same image and command; there is no per-lane variant. The
channels are the command's, not the queue's: the default command runs the
`release` channel and then the `nightly` one (`E2E_CHANNELS`, default `release
nightly`), each with its own runtimes and its own suite run, so a lane takes
about twice as long as one channel would. That time has not been measured on a
queue lane from a cold start, so a lane that cannot finish both channels within
its run time (counted from its start, so waiting for a GPU does not use it)
is reported as **incomplete**; the run time can be raised for a run, up to its
limit (see the run's inputs above). Each channel starts with the GPU preflight, as each
channel's job in `nightly.yml` does, so a model server the previous channel's
suite left on the GPU is reclaimed first; the lane log shows it. Both always
run, so a failing channel does not hide the other, and the lane fails if either
did.

A lane starts cold, so its first steps fetch from the network: the build
packages, the rustup installer, the toolchain and the crates for the host
being built. The lane command tries each of those up to four times, and never
retries a build or a test. The pre-warm's package downloads and the suite's
model downloads are left to the tools doing them, so a model that downloads
more slowly than its scenario waits fails that scenario. The lane fetches only
the host's crates, so it refuses to start with `CARGO_BUILD_TARGET` set.

The queue is the only scheduled run of the R9700, MI350P and W7900 lanes: their
self-hosted nightly jobs were removed from `nightly.yml`. The MI300X nightly
job is still there, and the queue's `mi300x` lane is an additional signal to it.
The lanes write only into the queue artifact, and the
`e2e-gpu-<channel>-report` artifacts that `nightly.yml`'s consolidated report
reads are not produced from them. The report fetches the queue's artifact
itself instead, so the support matrix, the `E2E-Support-Matrix` and
`Release-Readiness` wiki pages and the release-readiness verdict keep every
GPU the queue tests: see [The support matrix and the wiki](#the-support-matrix-and-the-wiki).

## Output

**A lane must write everything it wants reported directly into `/out`.** The
default command writes each channel's suite results to `/out/<channel>/results`.

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

## The support matrix and the wiki

`nightly.yml`'s `e2e-report-nightly` job builds the consolidated support
matrix, which `publish-e2e-wiki` publishes as the `E2E-Support-Matrix` wiki page
and `release-readiness-nightly` reads its GO/NO-GO verdict from (see
[ci-hardware-testing.md](ci-hardware-testing.md)). It also reads the queue's
lanes, from the queue's own results, so retiring a self-hosted nightly lane
that the queue covers drops no GPU from either page, and a GPU that only the
queue has (MI325X, MI350X, MI355X) is on them.

- **Which run.** The newest `queue-run-results` artifact of a run on this
  repository's `main` that finished, and started, in the last 24 hours,
  whatever the run's conclusion was: a red lane fails the run, and its artifact
  is still uploaded. The job finds the artifact by name, not through the
  workflow that makes it, because no other workflow may name that one (see
  below). The run also has to have tested `main`: a run started from `main` can
  be asked to test another branch (the `branch` input), and the job log names
  the branch it did. `queue-run.yml` starts at 02:17 UTC and the nightly
  starts at 06:00 UTC, its report job after its GPU lanes have finished, so a
  run that took under about three and a half hours is in the night's report. A
  run that is still going, that started on an earlier day or that tested
  another branch is not imported, and neither is a run published again by
  `nightly-queue.yml`.
- **Which lanes, and whether each blocks.** The artifact holds the lanes'
  files and nothing else about the run, so the job reads the run's job log, whose
  lane matrix is the only place the queue prints which lanes it ran and which of
  them can fail the run (the `GATE` column). Only the tested branch and commit
  and each lane's name, phase, exit status and gate are taken from it, each
  checked against a strict pattern first. A lane whose name is not a plain
  lowercase word (letters, digits and dashes, and not ending in `-advisory`,
  `-release` or `-nightly`) is shown as `unnamed-1` and so on, as a blocking
  lane with no result. Without the log nothing is imported.
- **What is imported.** For each lane and channel,
  `<lane>/<channel>/results/{report.json,platform.json,commands.jsonl}`, the
  three files the report reads, as an artifact named
  `e2e-queue-<lane>[-advisory]-<channel>-report`. The `lane.log`, the JUnit and
  HTML output and the harness diagnostics are not copied, and nothing the
  queue's scan withheld is brought back. The lane's `platform_slug` is replaced
  by `queue-<lane>-<channel>`: the slug is the report's column identity, and
  queue lanes of different models report the same one (an MI325X lane probes as
  `mi300x`, an MI350X or MI355X lane as `mi350p`). The lane command therefore
  has to keep writing those files to `/out/<channel>/results`, as
  [`scripts/run-e2e-lane.sh`](../scripts/run-e2e-lane.sh) does.
- **How it reads.** A queue row is labelled `MI325X (queue) Linux (release)`,
  so it is not confused with the self-hosted MI300X lane, and a lane the queue
  does not block on is `MI325X (queue, advisory) Linux (release)`. The page
  opens with a `Queue lanes` table: one row per lane the queue ran, with its
  gate, phase, exit status and the channels that reported.
- **What counts towards the verdict.** The same as for the queue's own run. A
  blocking lane's failures are in the `E2E` failure count. An advisory lane's are
  shown but left out of it. A lane that came back with no result, or whose
  channel is missing after it did not finish cleanly, is shown as `EMPTY` and
  counts for a blocking lane, because a blocking lane without a result is not
  a pass.
- **When there is nothing to import.** No run in the window, a run of another
  branch, an artifact or log that cannot be downloaded or an import that fails
  never fails the nightly report: the page opens with a `Queue lanes` note
  saying that no queue lanes are in it tonight and why, and the rest is built as
  before. The verdict does not take that absence for a failure, so read the note
  before reading a `GO`.

What is public follows from the queue's own rules: the artifact is public and
already scanned, only three of its files are used, and the log is the public job
log.

## What the workflow requires

- a self-hosted Linux runner registered with the `queue-report` label alone,
  without GitHub's default `self-hosted`, OS and architecture labels, so only a
  job naming `queue-report` lands on it, providing
  the queue side's publish command and read access to the queue side's results
- a registered runner name and host name that do not identify internal
  infrastructure, since GitHub prints both into every job log

Both workflows require **no secret** and no permission, and contract tests in
[workflow_contract.rs](../xtask/src/workflow_contract.rs)
(`queue_nightly_publisher_holds_no_credential`, `queue_run_holds_no_credential`)
keep it that way.

That test reads the workflow files on `main`. A workflow on another branch of
this repository can still name the runner's label, which is one reason the
runner holds no credential, below. On `main`, the tests also refuse any
other workflow that names the `queue-report` label or these workflows, so none
sends a job to the runner or waits on them by mistake. The runner is set up to
run only these two workflows as they are on `main`, by a check on the runner
that runs before any step. That check lives outside this repository, so
nothing here can verify it; the runner is still treated as untrusted, as above
and below. Whether this
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

**What the report is trusted for.** The runner keeps its own registration with
GitHub between jobs, and a job on it can read every retained run's results,
including runs started by hand that were never published. That is the same trust
the self-hosted runners already carry: their nightly results are not signed
either, and the queue runs on the same infrastructure under the same access. A
job on this runner still cannot reach the queue or its credential. Each job also
receives GitHub's own short-lived runtime token, as on any runner.

**Results move by a shared volume, not a network call.** The queue side writes
each finished run's publishable files; the runner reads them read-only. No
credential is needed between the two, and the runner's only calls ask the
queue side to start a run, how it is going, and to finish it; none carries a
file.

**A run is asked for, not submitted, from here.** `queue-run.yml` holds no
credential and sends nothing to the queue: the runner, which sits next to the
queue side, asks it for a run of a branch, and the queue side starts it with
its own credential. It believes the request because of where it comes from,
which is why the runner is set up to run only these two workflows from `main`
(a setting outside this repository). A request can start a run of a branch
already in this repository, and that branch's lane command then runs in the
lane pods. The lane spec gives those pods no queue or cluster credential and
mounts neither the results volume nor the runner's files, so the command can
publish only what it writes to `/out`, through the scan above. What else a pod
can reach is set by the queue's own isolation, which this repository cannot
verify.

**Only `/out` is published, and a missing run is never green.** Publishing by
origin is the main control on what reaches public results, with the scan
behind it. An unfinished run fails the job, and a job only ever publishes the
run it started or was given, so a nightly that stops running cannot keep
reporting an old pass: the missing night shows as a missing scheduled job.

**Nothing public names the queue's infrastructure.** Runner names, labels,
paths, branches and this page use neutral names, per
[AGENTS.md](../AGENTS.md)'s rules for public content.
