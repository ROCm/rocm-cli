#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Decide whether a pull request's MI300X requirement is met.
#
# A pull request has two MI300X lanes: `e2e-gpu-queue` (the shared GPU queue, the
# primary) and `e2e-gpu` (a self-hosted runner, a bonus). Either one satisfies the
# requirement, so a self-hosted lane that never got a runner must not turn the
# pull request red when the queue lane passed.
#
# Each lane is put in one class from its job result, its `skipped` output and
# whether its report artifact was uploaded:
#
#   passed     the lane succeeded AND uploaded its report
#   failed     the lane ran and failed (whatever it uploaded)
#   no-report  the lane succeeded but exported no report: ran, proved nothing
#   absent     skipped, cancelled, never got a runner (`abandoned`), not in the
#              run, or said it skipped itself on purpose: no evidence either way
#
# A self-hosted lane that skipped itself as stale (its PR closed or its head moved
# on) leaves the artifact STALE_ARTIFACT and is absent, however it ended.
#
# When MI300X_REQUIRED is `true` (a pull request that the queue lane takes):
#
#   - any lane `failed`               -> fail: a lane that ran and failed still fails
#   - else at least one lane `passed` -> pass
#   - else                            -> fail: no lane produced a passing report
#
# Otherwise (a push, a dispatch, a pull request the queue does not take) there is
# no either-or, and the lanes are judged as the generic missing-report check
# judges every other lane: a lane that ran (succeeded, failed, or was cut off by
# its timeout, which GitHub reports as cancelled) and exported no report fails
# the check, and so does a lane that never got a runner (`abandoned`: there is
# no queue lane to cover it), while a skipped or missing one does not. A lane
# that ran, failed and did export its report is the report's to show, as before.
#
# Inputs (environment):
#   MI300X_REQUIRED       `true` for the either-or rule above.
#   QUEUE_RESULT          `e2e-gpu-queue`'s result: the `needs` vocabulary.
#   QUEUE_SKIPPED         `true` when the queue lane skipped itself on purpose.
#   QUEUE_ARTIFACT        the queue lane's report artifact name.
#   SETTLE_RESULT         `e2e-gpu-settled`'s own result. Anything but `success` (or
#                         `skipped`) leaves the self-hosted result unknown: absent
#                         where the either-or rule applies, a failure where it does not.
#   SELFHOSTED_RESULT     `e2e-gpu`'s result: the `needs` vocabulary plus
#                         `abandoned` and `absent` (see settle-e2e-lanes.sh).
#   SELFHOSTED_ARTIFACT   the self-hosted lane's report artifact name.
#   STALE_ARTIFACT        the marker artifact a stale-skipped self-hosted lane uploads.
#   ARTIFACTS             the run's artifact names, one per line.
#   GITHUB_STEP_SUMMARY   as GitHub provides it.
set -euo pipefail

required="${MI300X_REQUIRED:-false}"
artifacts="${ARTIFACTS-}"
summary_file="${GITHUB_STEP_SUMMARY:-/dev/null}"

# classify <result> <skipped output> <artifact>
classify() {
  local result="$1" skipped="$2" artifact="$3"
  case "${result}" in
    success)
      if [[ "${skipped}" = true ]]; then
        echo absent
      elif grep -qxF -- "${artifact}" <<<"${artifacts}"; then
        echo passed
      else
        echo no-report
      fi
      ;;
    failure | timed_out) echo failed ;;
    cancelled | skipped | abandoned | absent | "") echo absent ;;
    # A result nobody here has reasoned about must not read as an absent lane.
    *) echo failed ;;
  esac
}

# ran_without_report <result> <skipped output> <artifact>: did the lane run to an
# end (or be cut off) and leave no report? As the generic missing-report check
# reads a lane: only a skipped or never-started one is excused.
ran_without_report() {
  local result="$1" skipped="$2" artifact="$3"
  case "${result}" in
    skipped | abandoned | absent | "") return 1 ;;
    *) ;;
  esac
  [[ "${skipped}" = true ]] && return 1
  ! grep -qxF -- "${artifact}" <<<"${artifacts}"
}

queue=$(classify "${QUEUE_RESULT-}" "${QUEUE_SKIPPED-}" "${QUEUE_ARTIFACT:?QUEUE_ARTIFACT is required}")
selfhosted_result="${SELFHOSTED_RESULT-}"
# A lane that skipped itself as stale did not run for this head: absent.
if grep -qxF -- "${STALE_ARTIFACT:?STALE_ARTIFACT is required}" <<<"${artifacts}"; then
  selfhosted_result=skipped
fi
# `e2e-gpu-settled` hands over the self-hosted lane's result; when it failed, was
# cancelled or never ran, that result is unknown and an empty one must not read as
# an absent lane.
settle_unknown=false
if [[ "${SETTLE_RESULT:-success}" != success && "${SETTLE_RESULT}" != skipped ]]; then
  settle_unknown=true
  selfhosted_result=""
fi
selfhosted=$(classify "${selfhosted_result}" "" "${SELFHOSTED_ARTIFACT:?SELFHOSTED_ARTIFACT is required}")

verdict=0
if [[ "${required}" = true ]]; then
  if [[ "${queue}" = failed || "${selfhosted}" = failed ]]; then
    verdict=1
    reason="a lane that ran failed"
  elif [[ "${queue}" = passed || "${selfhosted}" = passed ]]; then
    reason="at least one lane passed with a report"
  else
    verdict=1
    reason="no lane produced a passing report"
  fi
else
  if [[ "${settle_unknown}" = true ]]; then
    verdict=1
    reason="the self-hosted lane's result is unknown (the settle job ${SETTLE_RESULT})"
  elif [[ "${selfhosted_result}" = abandoned ]]; then
    # No queue lane to fall back on: a lane that never got a runner is the only
    # MI300X signal missing, and it failed the check before this job gave up on it.
    verdict=1
    reason="the self-hosted lane never got a runner and no queue lane covers it"
  elif ran_without_report "${QUEUE_RESULT-}" "${QUEUE_SKIPPED-}" "${QUEUE_ARTIFACT}" ||
    ran_without_report "${selfhosted_result}" "" "${SELFHOSTED_ARTIFACT}"; then
    verdict=1
    reason="a lane ran and exported no report"
  else
    reason="no either-or for this run; a lane that ran fails only if it exported no report"
  fi
fi

mark=":white_check_mark:"
[[ "${required}" = true ]] || mark=":information_source:"
[[ "${verdict}" = 0 ]] || mark=":x:"
{
  echo "### MI300X: ${mark} ${reason}"
  echo "- queue lane (\`e2e-gpu-queue\`): ${queue} (result: ${QUEUE_RESULT:-absent})"
  echo "- self-hosted lane (\`e2e-gpu\`): ${selfhosted} (result: ${selfhosted_result:-absent})"
  if [[ "${queue}" = absent && "${required}" = true ]]; then
    echo "- The queue lane did not run for this push (switched off, or refused for the nightly window), and a skipped lane is never a pass: re-run once it can, or when the self-hosted lane has a runner."
  fi
  if [[ "${selfhosted}" = absent && "${required}" = true ]]; then
    echo "- A self-hosted lane that was cancelled, skipped or never got a runner counts as absent, not failed."
  fi
} >>"${summary_file}"
echo "MI300X queue lane: ${queue}; self-hosted lane: ${selfhosted}; ${reason}"
if [[ "${verdict}" != 0 ]]; then
  echo "::error::MI300X requirement not met: ${reason} (queue: ${queue}, self-hosted: ${selfhosted})"
fi
exit "${verdict}"
