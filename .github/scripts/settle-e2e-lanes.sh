#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Wait until every nightly E2E lane has finished, or can be given up on.
#
# GitHub has no queue timeout for a job: `timeout-minutes` only starts counting
# once a runner picks the job up, so a lane whose runner pool is offline sits
# queued for the full 24 hours, and every job that `needs` it waits with it. The
# nightly's report and readiness jobs therefore depend on THIS job instead of on
# the lanes. It polls the run's own job list and settles when no lane is still
# able to produce a result:
#
#   - a lane leg that has completed is done;
#   - a leg still queued QUEUE_DEADLINE_MINUTES after it was created is
#     abandoned (no runner took it), unless another leg of the same lane has
#     started or finished: that proves the pool has a runner, and the queued
#     leg is only waiting its turn behind it on a single-runner pool, so it
#     is held to the hard deadline instead;
#   - a leg that started is waited for (its own `timeout-minutes` bounds it),
#     but anything unfinished HARD_DEADLINE_MINUTES after it was created is
#     abandoned too, so this job cannot wait unboundedly on a hung runner.
#
# If the job list cannot be read for long enough to span the hard deadline, the
# script exits 1 instead of polling until the job's own timeout (which GitHub
# reports as `cancelled`), so the run's canceller sees a failed settle job.
#
# Outputs (GITHUB_OUTPUT):
#   lane_results     one-line JSON object, lane job id -> result, shaped like the
#                    `needs` context the report job's missing-report check reads
#                    (`{"<job id>": {"result": "<result>"}}`). Results are the
#                    `needs` vocabulary (success, failure, cancelled, skipped)
#                    plus `abandoned` (never got a runner, or ran past the hard
#                    deadline) and `absent` (no job in the run carries the lane's
#                    name, so a rename cannot make a lane silently disappear).
#   abandoned_count  how many lane legs were abandoned; the job that cancels the
#                    stragglers runs only when this is not 0.
#
# Inputs (environment):
#   LANES                    lines of `<job id>=<job display name>`; a leg of a
#                            matrix lane is `<display name> (<leg>)`.
#   QUEUE_DEADLINE_MINUTES   how long a leg may stay queued.
#   HARD_DEADLINE_MINUTES    how long after creation a leg may stay unfinished.
#   LANE_RUN_MINUTES         the longest a lane may run once it starts (its
#                            timeout-minutes). QUEUE_DEADLINE_MINUTES plus this must
#                            fit in HARD_DEADLINE_MINUTES, or a lane that starts just
#                            inside the queue deadline would be abandoned while it is
#                            still legitimately running.
#   POLL_SECS                seconds between polls (default 60).
#   SETTLE_NOW_EPOCH         pin "now" (seconds since the epoch); tests only.
#   SETTLE_CALL_TIMEOUT_SECS limit on one job-list request (default 60); tests only.
#   GH_TOKEN, GITHUB_REPOSITORY, GITHUB_RUN_ID, GITHUB_OUTPUT, GITHUB_STEP_SUMMARY
#                            as GitHub provides them.
set -euo pipefail

: "${LANES:?LANES is required}"
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
: "${GITHUB_RUN_ID:?GITHUB_RUN_ID is required}"
queue_minutes="${QUEUE_DEADLINE_MINUTES:?QUEUE_DEADLINE_MINUTES is required}"
hard_minutes="${HARD_DEADLINE_MINUTES:?HARD_DEADLINE_MINUTES is required}"
lane_run_minutes="${LANE_RUN_MINUTES:?LANE_RUN_MINUTES is required}"
poll_secs="${POLL_SECS:-60}"
# Bash arithmetic reads a non-number as 0, which would give up on every lane at
# once; refuse it instead (the queue deadline is a dispatch input).
for value in "${queue_minutes}" "${hard_minutes}" "${lane_run_minutes}" "${poll_secs}"; do
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "::error::deadline and poll values must be whole numbers, got '${value}'"
    exit 1
  fi
done
# Force base 10 from here on: arithmetic reads a leading zero as octal, so `060`
# would be 48 and `008` an error.
queue_minutes=$((10#${queue_minutes}))
hard_minutes=$((10#${hard_minutes}))
lane_run_minutes=$((10#${lane_run_minutes}))
poll_secs=$((10#${poll_secs}))
if ((queue_minutes + lane_run_minutes > hard_minutes)); then
  echo "::error::a lane queued for ${queue_minutes} minutes and then run for ${lane_run_minutes} would pass the ${hard_minutes}-minute hard deadline; lower the queue deadline or raise the hard one"
  exit 1
fi
output_file="${GITHUB_OUTPUT:-/dev/null}"
summary_file="${GITHUB_STEP_SUMMARY:-/dev/null}"

lanes_json=$(
  while IFS='=' read -r id name; do
    [[ -n "${id}" ]] || continue
    # shellcheck disable=SC2016  # jq program, not shell
    jq -n --arg id "${id}" --arg name "${name}" '{id: $id, name: $name}'
  done <<<"${LANES}" | jq -s .
)

# One verdict per poll: every lane's result, whether anything is still pending,
# and the names of the abandoned legs. `pending` wins over `abandoned` within a
# lane, so a lane with one leg still able to finish is not settled early. A job
# belongs to the lane with the longest name it matches, so a lane named
# `Build` and another named `Build (extra)` each keep their own legs.
# shellcheck disable=SC2016  # jq program, not shell
evaluate='
  def age: ($now - (.created_at | fromdateiso8601));
  def leg_state($lane_started):
    if .status == "completed" then "done"
    elif age >= $hard then "abandoned"
    elif .status == "in_progress" then "running"
    elif age >= $queue and ($lane_started | not) then "abandoned"
    else "waiting" end;
  def owner($job):
    [ $lanes[] | . as $l | select($job.name == $l.name or ($job.name | startswith($l.name + " ("))) ]
    | max_by(.name | length);
  . as $jobs
  | ($jobs | map(. as $job | . + {lane_id: (owner($job).id // null)})) as $owned
  | [ $lanes[] | . as $lane
      | ($owned | map(select(.lane_id == $lane.id))) as $legs
      | ($legs | any(.status == "in_progress" or .status == "completed")) as $started
      | ($legs | map(leg_state($started))) as $states
      | { id: $lane.id,
          abandoned: [ range(0; $legs | length) | select($states[.] == "abandoned") | $legs[.].name ],
          result:
            ( if ($legs | length) == 0 then "absent"
              elif any($states[]; . == "running" or . == "waiting") then "pending"
              elif any($states[]; . == "abandoned") then "abandoned"
              elif all($legs[]; .conclusion == "skipped") then "skipped"
              elif any($legs[]; .conclusion == "failure" or .conclusion == "timed_out") then "failure"
              elif any($legs[]; .conclusion == "cancelled") then "cancelled"
              else "success" end ) } ]
  | { pending: any(.[]; .result == "pending"),
      results: (map({key: .id, value: {result: .result}}) | from_entries),
      abandoned: (map(.abandoned) | add) }
'

# A job listing that keeps failing must not leave this job polling until its own
# cap: GitHub reports a job killed by `timeout-minutes` as `cancelled`, not
# `failure`, and the hard deadline above is only judged once a listing arrives.
# So after enough consecutive failures to span the hard deadline (each attempt
# takes at most the call timeout plus the poll interval) the job fails itself,
# which lets the run's canceller end the run. The call timeout keeps one hung
# request from defeating that bound.
call_timeout_secs="${SETTLE_CALL_TIMEOUT_SECS:-60}"
attempt_secs=$((poll_secs + call_timeout_secs))
max_failures=$(((hard_minutes * 60 + attempt_secs - 1) / attempt_secs))
failures=0

while true; do
  now="${SETTLE_NOW_EPOCH:-$(date +%s)}"
  if run_jobs=$(timeout "${call_timeout_secs}" gh api --paginate \
    "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/jobs?per_page=100&filter=latest" \
    --jq '.jobs[]' | jq -s .); then
    failures=0
    verdict=$(jq -c \
      --argjson lanes "${lanes_json}" \
      --argjson now "${now}" \
      --argjson queue "$((queue_minutes * 60))" \
      --argjson hard "$((hard_minutes * 60))" \
      "${evaluate}" <<<"${run_jobs}")
    if [[ "$(jq -r .pending <<<"${verdict}")" = false ]]; then
      break
    fi
  else
    # A transient API failure must not end the wait early (the lanes would be
    # reported as having produced nothing), but a sustained one ends it.
    failures=$((failures + 1))
    if ((failures >= max_failures)); then
      echo "::error::could not list this run's jobs ${failures} times in a row; giving up so the run can be ended"
      exit 1
    fi
    echo "::warning::could not list this run's jobs; retrying"
  fi
  sleep "${poll_secs}"
done

abandoned_count=$(jq -r '.abandoned | length' <<<"${verdict}")
{
  echo "lane_results=$(jq -c .results <<<"${verdict}")"
  echo "abandoned_count=${abandoned_count}"
} >>"${output_file}"

if [[ "${abandoned_count}" -gt 0 ]]; then
  while IFS= read -r name; do
    echo "::warning::${name} was given up on: it never got a runner within ${queue_minutes} minutes (or ran past ${hard_minutes}), so it is missing from the reports"
    echo "- :hourglass: \`${name}\` was given up on: no runner took it within ${queue_minutes} minutes (or it ran past ${hard_minutes}); it is missing from the reports" >>"${summary_file}"
  done < <(jq -r '.abandoned[]' <<<"${verdict}")
fi
