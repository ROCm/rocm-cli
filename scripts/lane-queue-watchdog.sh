#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Turn a self-hosted lane that no runner ever picked up into a red check.
#
# WHY: GitHub cancels a job no runner has been assigned after 24 hours, and
# `timeout-minutes` cannot catch that because it counts from the job START. This
# script runs on a GitHub-hosted runner next to the lanes, polls this run's job
# list, and fails as soon as ANY watched job that no runner has started (`queued`,
# or the not-yet-running `waiting`, `pending` and `requested`) has waited for at
# least a budget well under 24 hours, with an annotation naming the lane.
# Which lanes are watched, why `mi300x` and `w7900` are exempt, and how this
# relates to the report job's own (later) red: docs/ci-hardware-testing.md,
# "Lanes no runner picks up".
#
# LIMIT: only the jobs this run created are watched, and only while this job
# runs. "Re-run failed jobs" creates new lane jobs after the original watchdog
# finished, so those are not watched; a full re-run starts a fresh watchdog.
#
# Usage:
#   lane-queue-watchdog.sh LABEL [LABEL...]
#
# A job is WATCHED when its runner labels (the `runs-on` list) include any LABEL.
# Labels are matched rather than job names because the jobs API reports a
# matrix job's expanded display name, never its id, and a name can be renamed
# without anyone noticing that the watchdog stopped covering the lane.
#
# Environment:
#   GITHUB_REPOSITORY, GITHUB_RUN_ID   the run to watch (set by Actions)
#   GH_TOKEN                           token for `gh` (needs `actions: read`)
#   LANE_QUEUE_BUDGET_MINUTES          queue budget in minutes; the workflows set
#                                      it, the default only applies when unset
#   LANE_QUEUE_POLL_SECONDS            poll interval, default 60
#   LANE_QUEUE_GRACE_MINUTES           how long to wait for a watched job to
#                                      appear in the job list at all, default 10
#   LANE_QUEUE_API_RETRIES             attempts per API call, default 3
#
# Seams for tests (not meant for workflows):
#   LANE_QUEUE_NOW      epoch seconds to use as "now" at the first poll; every
#                       later poll adds the seconds slept so far, so a test can
#                       watch a job cross the budget between polls
#   LANE_QUEUE_SLEEP    command used instead of `sleep`
#   `gh` is resolved from PATH, so a stub earlier on PATH replaces it.
#
# Exit status:
#   0  every watched job has started or finished (skipped, completed and running
#      jobs are never failures), or no watched job appeared within the grace
#      period (reported as a warning, since there is nothing to watch);
#   1  any watched job is still waiting for a runner at or past the budget, or has
#      no creation time to measure, or the job list could not be read, or the
#      per-poll filter failed;
#   2  usage error: no label, or a setting that is not a whole number (the poll
#      interval must also be at least 1).

set -euo pipefail

budget_minutes="${LANE_QUEUE_BUDGET_MINUTES:-300}"
poll_seconds="${LANE_QUEUE_POLL_SECONDS:-60}"
grace_minutes="${LANE_QUEUE_GRACE_MINUTES:-10}"
api_retries="${LANE_QUEUE_API_RETRIES:-3}"
sleep_cmd="${LANE_QUEUE_SLEEP:-sleep}"
test_now="${LANE_QUEUE_NOW:-}"

if [[ $# -eq 0 ]]; then
  echo "usage: lane-queue-watchdog.sh LABEL [LABEL...]" >&2
  exit 2
fi
# No leading zeros: bash reads `08` as a bad octal number inside arithmetic and
# the poll loop would then fail the same way on every pass until the job times out.
for value in "${budget_minutes}" "${poll_seconds}" "${grace_minutes}" "${api_retries}" ${test_now:+"${test_now}"}; do
  if [[ ! "${value}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "::error::lane-queue-watchdog: '${value}' is not a whole number" >&2
    exit 2
  fi
done
# A zero interval never advances the clock: the grace period would not expire and
# the API would be called back to back.
if [[ "${poll_seconds}" -lt 1 ]]; then
  echo "::error::lane-queue-watchdog: LANE_QUEUE_POLL_SECONDS must be at least 1" >&2
  exit 2
fi
: "${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is not set}"
: "${GITHUB_RUN_ID:?GITHUB_RUN_ID is not set}"

watch_labels_json=$(printf '%s\n' "$@" | jq -R . | jq -sc .)

# Print one JSON job object per line for this run, or return non-zero.
fetch_jobs() {
  gh api --paginate \
    "repos/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/jobs?per_page=100" \
    --jq '.jobs[]'
}

# fetch_jobs, retried: a transient API error must not pass the lane and must not
# fail it on the first blip either.
fetch_jobs_with_retry() {
  local attempt=1 out
  while true; do
    if out=$(fetch_jobs); then
      printf '%s\n' "${out}"
      return 0
    fi
    if [[ "${attempt}" -ge "${api_retries}" ]]; then
      return 1
    fi
    echo "::warning::could not list this run's jobs (attempt ${attempt}/${api_retries}); retrying" >&2
    attempt=$((attempt + 1))
    "${sleep_cmd}" 5
  done
}

waited_seconds=0
while true; do
  if ! jobs_json=$(fetch_jobs_with_retry); then
    echo "::error::could not list this run's jobs after ${api_retries} attempts, so no self-hosted lane was checked for being stuck in the queue"
    exit 1
  fi

  if [[ -n "${test_now}" ]]; then
    now=$((test_now + waited_seconds))
  else
    now=$(date +%s)
  fi
  # One line per watched job: status<TAB>age in whole minutes<TAB>name. Age is
  # measured from job creation, which is when it began waiting; a job with no
  # `created_at` gets the age "unknown" so it can be reported rather than aged 0.
  if ! watched=$(jq -rs --argjson watch "${watch_labels_json}" --argjson now "${now}" '
    .[]
    | select(any((.labels // [])[]; . as $l | $watch | index($l)))
    | [.status,
       (if .created_at then ((($now - (.created_at | fromdateiso8601)) / 60) | floor) else "unknown" end),
       .name]
    | @tsv' <<<"${jobs_json}"); then
    echo "::error::could not read this run's job list (the jq filter failed), so no self-hosted lane was checked for being stuck in the queue"
    exit 1
  fi

  if [[ -z "${watched}" ]]; then
    # The lanes may not be in the list yet. Wait a bounded grace, then give up
    # quietly: a run whose lanes never appear has nothing to watch.
    if [[ "$((waited_seconds / 60))" -ge "${grace_minutes}" ]]; then
      echo "::warning::no job with a watched runner label (${*}) appeared in this run; nothing to watch"
      exit 0
    fi
  else
    # A watched job is still waiting for a runner until it is in_progress or
    # completed (which includes skipped). Fail as soon as ANY waiting job is at
    # or past the budget, naming only the over-budget ones.
    waiting=0
    over_budget=0
    while IFS=$'\t' read -r status age name; do
      case "${status}" in
        in_progress | completed) continue ;;
        *) ;;
      esac
      waiting=1
      if [[ "${age}" == "unknown" ]]; then
        echo "::error::${name} is ${status} but has no creation time, so how long it has waited cannot be measured"
        over_budget=1
      elif [[ "${age}" -ge "${budget_minutes}" ]]; then
        echo "::error::${name} has been ${status} for ${age} minutes (budget ${budget_minutes}), past the budget; no runner picked it up, so this lane may be providing no coverage"
        over_budget=1
      fi
    done <<<"${watched}"
    if [[ "${over_budget}" -eq 1 ]]; then
      echo "GitHub cancels a job nobody picked up after 24 hours, and that shows as cancelled rather than failed. This check is red so the gap is visible." >&2
      exit 1
    fi
    if [[ "${waiting}" -eq 0 ]]; then
      echo "every watched self-hosted lane was scheduled or skipped"
      exit 0
    fi
  fi

  "${sleep_cmd}" "${poll_seconds}"
  waited_seconds=$((waited_seconds + poll_seconds))
done
