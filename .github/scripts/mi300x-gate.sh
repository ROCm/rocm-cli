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
# When MI300X_REQUIRED is `true` (a pull request that the queue lane takes):
#
#   - any lane `failed`               -> fail: a lane that ran and failed still fails
#   - else at least one lane `passed` -> pass
#   - else                            -> fail: no lane produced a passing report
#
# Otherwise (a push, a dispatch, a pull request the queue does not take) there is
# no either-or: only a lane that ran and left no report (`no-report`) fails, as
# the generic missing-report check treats every other lane.
#
# Inputs (environment):
#   MI300X_REQUIRED       `true` for the either-or rule above.
#   QUEUE_RESULT          `e2e-gpu-queue`'s result: the `needs` vocabulary.
#   QUEUE_SKIPPED         `true` when the queue lane skipped itself on purpose.
#   QUEUE_ARTIFACT        the queue lane's report artifact name.
#   SELFHOSTED_RESULT     `e2e-gpu`'s result: the `needs` vocabulary plus
#                         `abandoned` and `absent` (see settle-e2e-lanes.sh).
#   SELFHOSTED_ARTIFACT   the self-hosted lane's report artifact name.
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

queue=$(classify "${QUEUE_RESULT-}" "${QUEUE_SKIPPED-}" "${QUEUE_ARTIFACT:?QUEUE_ARTIFACT is required}")
selfhosted=$(classify "${SELFHOSTED_RESULT-}" "" "${SELFHOSTED_ARTIFACT:?SELFHOSTED_ARTIFACT is required}")

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
  if [[ "${queue}" = no-report || "${selfhosted}" = no-report ]]; then
    verdict=1
    reason="a lane ran and exported no report"
  else
    reason="MI300X is not required of this run; only a lane that ran without a report fails"
  fi
fi

mark=":white_check_mark:"
[[ "${verdict}" = 0 ]] || mark=":x:"
{
  echo "### MI300X: ${mark} ${reason}"
  echo "- queue lane (\`e2e-gpu-queue\`): ${queue} (result: ${QUEUE_RESULT:-absent})"
  echo "- self-hosted lane (\`e2e-gpu\`): ${selfhosted} (result: ${SELFHOSTED_RESULT:-absent})"
  if [[ "${selfhosted}" = absent && "${required}" = true ]]; then
    echo "- A self-hosted lane that was cancelled, skipped or never got a runner counts as absent, not failed."
  fi
} >>"${summary_file}"
echo "MI300X queue lane: ${queue}; self-hosted lane: ${selfhosted}; ${reason}"
if [[ "${verdict}" != 0 ]]; then
  echo "::error::MI300X requirement not met: ${reason} (queue: ${queue}, self-hosted: ${selfhosted})"
fi
exit "${verdict}"
