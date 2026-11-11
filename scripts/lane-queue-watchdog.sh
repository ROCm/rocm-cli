#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Turn a self-hosted lane that no runner ever picked up into a red check.
#
# WHY THIS EXISTS: GitHub cancels a job that no runner has been assigned after
# 24 hours. The job concludes `cancelled`, which merge gates treat as neutral, so
# a lane whose runners are offline silently drops out of coverage and nothing in
# the run turns red. `timeout-minutes` does not help: it counts from the job
# START, and a queued job has not started. This script runs on a GitHub-hosted
# runner next to the lanes, polls this run's job list, and fails once a watched
# job has sat in `queued` for longer than a budget well under 24 hours.
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
#   LANE_QUEUE_BUDGET_MINUTES          queue budget, default 300
#   LANE_QUEUE_POLL_SECONDS            poll interval, default 60
#   LANE_QUEUE_GRACE_MINUTES           how long to wait for a watched job to
#                                      appear in the job list at all, default 10
#   LANE_QUEUE_API_RETRIES             attempts per API call, default 3
#
# Seams for tests (not meant for workflows):
#   LANE_QUEUE_NOW      epoch seconds to use as "now" when ageing a job
#   LANE_QUEUE_SLEEP    command used instead of `sleep`
#   `gh` is resolved from PATH, so a stub earlier on PATH replaces it.
#
# Exit status: 0 when no watched job is left queued (skipped, completed and
# running jobs are never failures); 1 when a watched job is still queued past the
# budget, or when the job list could not be read; 2 on a usage error.

set -euo pipefail

budget_minutes="${LANE_QUEUE_BUDGET_MINUTES:-300}"
poll_seconds="${LANE_QUEUE_POLL_SECONDS:-60}"
grace_minutes="${LANE_QUEUE_GRACE_MINUTES:-10}"
api_retries="${LANE_QUEUE_API_RETRIES:-3}"
sleep_cmd="${LANE_QUEUE_SLEEP:-sleep}"

if [[ $# -eq 0 ]]; then
  echo "usage: lane-queue-watchdog.sh LABEL [LABEL...]" >&2
  exit 2
fi
for value in "${budget_minutes}" "${poll_seconds}" "${grace_minutes}" "${api_retries}"; do
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "::error::lane-queue-watchdog: '${value}' is not a whole number" >&2
    exit 2
  fi
done
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

  now="${LANE_QUEUE_NOW:-$(date +%s)}"
  # One line per watched job: status<TAB>age in whole minutes<TAB>name.
  # Age is measured from job creation, which is when it began waiting.
  watched=$(jq -rs --argjson watch "${watch_labels_json}" --argjson now "${now}" '
    .[]
    | select(any((.labels // [])[]; . as $l | $watch | index($l)))
    | [.status,
       (if .created_at then ((($now - (.created_at | fromdateiso8601)) / 60) | floor) else 0 end),
       .name]
    | @tsv' <<<"${jobs_json}")

  if [[ -z "${watched}" ]]; then
    # The lanes may not be in the list yet. Wait a bounded grace, then give up
    # quietly: a run whose lanes never appear has nothing to watch.
    if [[ "$((waited_seconds / 60))" -ge "${grace_minutes}" ]]; then
      echo "::warning::no job with a watched runner label (${*}) appeared in this run; nothing to watch"
      exit 0
    fi
  else
    queued=$(awk -F '\t' '$1 == "queued"' <<<"${watched}")
    if [[ -z "${queued}" ]]; then
      echo "no watched job is queued; every self-hosted lane was scheduled or skipped"
      exit 0
    fi
    # Keep waiting while any queued job is still inside its budget.
    if awk -F '\t' -v budget="${budget_minutes}" '$2 < budget { found = 1 } END { exit !found }' <<<"${queued}"; then
      :
    else
      while IFS=$'\t' read -r _ age name; do
        echo "::error::${name} has been queued for ${age} minutes (budget ${budget_minutes}); no runner picked it up; this lane provided no coverage"
      done <<<"${queued}"
      echo "GitHub cancels a job nobody picked up after 24 hours, and that shows as cancelled rather than failed. This check is red so the gap is visible." >&2
      exit 1
    fi
  fi

  "${sleep_cmd}" "${poll_seconds}"
  waited_seconds=$((waited_seconds + poll_seconds))
done
