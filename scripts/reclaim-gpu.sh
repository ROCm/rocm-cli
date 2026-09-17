#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Reclaim the GPU from E2E processes leaked by a killed or timed-out prior run,
# whose Drop teardown never executed.
#
# WHY A SCRIPT AND NOT INLINE `pkill -f`: the inline version anchored each
# pattern to a scenario temp dir (`/tmp/rocm-e2e`) or the shared *cache*
# (`e2e-shared`), but a managed serve runs the engine binary out of the shared
# *pre-warm runtime* tree — `$RUNNER_WORKSPACE/e2e-prewarm-*/data/runtimes/...`
# — which matched neither. `pkill -f` also matches the whole command line in
# order, and argv[0] is the binary path, so a `/tmp/rocm-e2e-*` model argument
# appearing later could not rescue those patterns either. A leaked llama-server
# was therefore invisible to reclaim and held the card until the runner itself
# was replaced, failing every later job on that runner at the GPU preflight.
#
# That is why a miss here is expensive rather than merely untidy: on the
# container lanes the pod is usually fresh per job, but the work volume is not
# and two jobs can share one pod's life — so a single leak the reclaim cannot
# match poisons every job scheduled to that runner until the pod is recreated.
#
# The rule here is order-independent and covers every E2E-owned tree: a process
# is reclaimed when its command line names an E2E root AND an engine/serve
# process. Both halves are required, which is what keeps a legitimate
# `/workload` manual-testing serve on a shared self-hosted runner safe — it
# names no E2E root.

set -euo pipefail

# E2E-owned roots. A process must name one of these to be considered ours.
#   /tmp/rocm-e2e   per-scenario temp dirs
#   e2e-shared      E2E_SHARED_CACHE_DIR (models, HF weights)
#   e2e-prewarm     E2E_SHARED_RUNTIMES_DIR — the pre-warmed runtime tree the
#                   engine binaries actually live in
#   e2e-target      CARGO_TARGET_DIR for the suite's own binaries
E2E_ROOTS=(
  '/tmp/rocm-e2e'
  'e2e-shared'
  'e2e-prewarm'
  'e2e-target'
)

# Engine/serve processes that can hold VRAM. Matched anywhere in the command
# line, so a wrapper or an absolute binary path both work.
ENGINE_MARKERS=(
  'llama-server'
  'vllm'
  '__engine-serve-http'
  'rocm daemon'
)

# Seconds to wait for a TERM'd process to exit before escalating to KILL.
TERM_GRACE_SECS="${RECLAIM_TERM_GRACE_SECS:-5}"

# Set by --self-test so its decoys are the only things in scope. Never set in
# CI: an empty scope means "any process matching the rules above".
SELFTEST_SCOPE="${RECLAIM_SELFTEST_SCOPE:-}"

usage() {
  cat <<'EOF'
Usage: reclaim-gpu.sh [--dry-run | --report-holders | --self-test]

  (no flags)        Terminate leaked E2E engine processes and report what was killed.
  --dry-run         List what would be terminated; kill nothing.
  --report-holders  Print current GPU holders and candidate processes, for
                    diagnosing a preflight failure. Kills nothing, never fails.
  --self-test       Verify the matching rules against decoy processes. No GPU needed.
EOF
}

# True while a pid is a live process. A zombie is NOT alive: it has already
# released its VRAM and is only waiting to be reaped, but `kill -0` still
# succeeds on it, which would read as "ignored SIGTERM".
process_alive() {
  local pid="$1"
  local state
  # /proc/<pid>/stat field 3 is the state, but field 2 (comm) may contain
  # spaces, so cut after the closing paren rather than counting fields.
  state="$(sed 's/.*) //' "/proc/${pid}/stat" 2>/dev/null | cut -d' ' -f1)" || return 1
  [[ -n "${state}" ]] || return 1
  [[ "${state}" != "Z" ]]
}

# Print "pid<TAB>command line" for every process whose command line names both
# an E2E root and an engine marker.
select_leaked() {
  local cmdline_file pid cmdline root engine has_root has_engine
  for cmdline_file in /proc/[0-9]*/cmdline; do
    pid="${cmdline_file#/proc/}"
    pid="${pid%/cmdline}"
    # Never reclaim ourselves or our own shell.
    if [[ "${pid}" == "$$" || "${pid}" == "${PPID}" ]]; then
      continue
    fi
    # A process can exit between the glob and the read; that is not an error.
    # Redirect stderr BEFORE the input redirection: the shell applies them left
    # to right, so `<file 2>/dev/null` still lets the shell's own "No such file"
    # reach the terminal when the open fails.
    cmdline="$(tr '\0' ' ' 2>/dev/null <"${cmdline_file}")" || continue
    [[ -n "${cmdline}" ]] || continue

    if [[ -n "${SELFTEST_SCOPE}" ]]; then
      case "${cmdline}" in
        *"${SELFTEST_SCOPE}"*) ;;
        *) continue ;;
      esac
    fi

    has_root=0
    for root in "${E2E_ROOTS[@]}"; do
      case "${cmdline}" in
        *"${root}"*)
          has_root=1
          break
          ;;
        *) ;;
      esac
    done
    [[ "${has_root}" == 1 ]] || continue

    has_engine=0
    for engine in "${ENGINE_MARKERS[@]}"; do
      case "${cmdline}" in
        *"${engine}"*)
          has_engine=1
          break
          ;;
        *) ;;
      esac
    done
    [[ "${has_engine}" == 1 ]] || continue

    printf '%s\t%s\n' "${pid}" "${cmdline}"
  done
}

# TERM, wait out the grace period, then KILL whatever is left. Reports every
# process it acts on: an unconditional "reclaimed" tells the next reader
# nothing, and a silent no-op is how the original defect stayed hidden.
reclaim() {
  local dry_run="$1"
  local selected pid cmdline current
  local killed=0
  local waited=0

  selected="$(select_leaked)"
  if [[ -z "${selected}" ]]; then
    echo "reclaim: no leaked E2E engine processes found"
    return 0
  fi

  while IFS=$'\t' read -r pid cmdline; do
    [[ -n "${pid}" ]] || continue
    if [[ "${dry_run}" == 1 ]]; then
      echo "reclaim: would terminate pid=${pid} cmd=${cmdline}"
    else
      echo "reclaim: terminating pid=${pid} cmd=${cmdline}"
      kill -TERM "${pid}" 2>/dev/null || true
    fi
    killed=$((killed + 1))
  done <<<"${selected}"

  if [[ "${dry_run}" == 1 ]]; then
    echo "reclaim: ${killed} process(es) would be terminated (dry run)"
    return 0
  fi

  while [[ "${waited}" -lt "${TERM_GRACE_SECS}" ]]; do
    if [[ -z "$(select_leaked)" ]]; then
      break
    fi
    sleep 1
    waited=$((waited + 1))
  done

  # Anything still alive after the grace period gets SIGKILL. VRAM is released
  # by the kernel when the process dies, so this is what actually frees the card.
  while IFS=$'\t' read -r pid cmdline; do
    [[ -n "${pid}" ]] || continue
    process_alive "${pid}" || continue
    # This loop walks the pre-TERM snapshot, and a pid freed during the grace
    # window can be handed to an unrelated process. Escalating on the pid alone
    # would SIGKILL that bystander, so require the command line to still be the
    # one we selected. Not killing a genuine holder is recoverable — the next
    # job's reclaim sees it again — where killing a bystander is not.
    current="$(tr '\0' ' ' 2>/dev/null <"/proc/${pid}/cmdline")" || continue
    if [[ "${current}" != "${cmdline}" ]]; then
      echo "reclaim: pid=${pid} was recycled during the grace period, not escalating"
      continue
    fi
    echo "reclaim: pid=${pid} ignored SIGTERM after ${TERM_GRACE_SECS}s, sending SIGKILL"
    kill -KILL "${pid}" 2>/dev/null || true
  done <<<"${selected}"

  echo "reclaim: ${killed} process(es) terminated"
}

# Diagnostics for a preflight that hit its ceiling. Never fails: it runs on the
# failure path, where masking the real error would be worse than missing output.
report_holders() {
  echo "--- rocm-smi KFD processes (per-process VRAM) ---"
  # Process names show as UNKNOWN inside a container: KFD reports host PIDs,
  # which do not resolve in the container's PID namespace. The VRAM column is
  # still the answer to "what is holding the card".
  timeout 15 rocm-smi --showpids 2>&1 | head -40 || true
  echo "--- engine/serve processes visible here ---"
  # shellcheck disable=SC2009 # pgrep cannot print elapsed time, and how long a
  # holder has been alive is what distinguishes a leak from this job's own serve.
  ps -eo pid,etimes,args 2>/dev/null |
    grep -Ei 'llama-server|vllm|__engine-serve-http|rocm daemon' |
    grep -v grep |
    head -40 || true
  echo "--- of those, E2E-owned (reclaim would take these) ---"
  select_leaked || true
}

# Spawn a decoy whose argv[0] is a real-shaped path, so the match is tested
# against the same string a live engine process would present.
spawn_decoy() {
  local path="$1"
  mkdir -p "$(dirname "${path}")"
  cp /bin/sleep "${path}"
  # Detach stdio: this runs inside a command substitution, and a background
  # child holding the capture pipe open would block the caller until it exits.
  "${path}" 300 >/dev/null 2>&1 &
  echo $!
}

# Spawn a decoy that IGNORES SIGTERM, so the TERM -> grace -> KILL escalation is
# exercised. A plain `cp /bin/sleep` decoy dies on the first TERM and leaves the
# escalation branch unreached, which is how it went untested.
spawn_stubborn_decoy() {
  local path="$1"
  mkdir -p "$(dirname "${path}")"
  # The foreground `sleep 1` children are short-lived and name no E2E root, so
  # they are never selected and leave nothing behind once the parent is killed.
  cat >"${path}" <<'DECOY'
#!/usr/bin/env bash
trap '' TERM
while :; do sleep 1; done
DECOY
  chmod +x "${path}"
  "${path}" >/dev/null 2>&1 &
  echo $!
}

self_test() {
  local tmp prewarm_decoy workload_decoy harness_decoy stubborn_decoy
  local prewarm_pid workload_pid harness_pid stubborn_pid selected reclaim_out
  local failures=0

  # Deliberately NOT under /tmp/rocm-e2e: that prefix is one of the roots the
  # old patterns did match, which would mask the regression this guards.
  tmp="$(mktemp -d /tmp/reclaim-selftest-XXXXXX)"
  export RECLAIM_SELFTEST_SCOPE="${tmp}"
  SELFTEST_SCOPE="${tmp}"
  # The stubborn decoy never exits on its own, so the grace loop always runs to
  # the ceiling. Keep it short: this is a unit-speed test, not a GPU lane.
  TERM_GRACE_SECS=2
  # shellcheck disable=SC2064 # expand ${tmp} now, at trap definition time
  trap "rm -rf '${tmp}'" EXIT

  # The real shape: lemonade's engine binary inside the shared pre-warm runtime.
  prewarm_decoy="${tmp}/e2e-prewarm-multi-arch-v2/data/runtimes/wheel/release-wheel-multi-arch-7-14-1-deadbeef/engines/lemonade/runtime/bin/llamacpp/rocm-stable/llama-b9752/llama-server"
  # A manual-testing serve on a shared runner: an engine, but no E2E root.
  workload_decoy="${tmp}/workload/manual-serve/llama-server"
  # An E2E root with NO engine marker — the suite's own test binary under
  # CARGO_TARGET_DIR. This is what the `has_engine` half exists to spare, and
  # without it in the fixtures that half can be deleted with the test still green.
  harness_decoy="${tmp}/e2e-target/release/deps/e2e-harness"
  # Same shape as the pre-warm decoy but ignores SIGTERM, so the escalation this
  # script adds is reached. Without it, TERM alone ends every decoy and both the
  # SIGKILL block and the `kill -TERM` call can be removed with the test green.
  stubborn_decoy="${tmp}/e2e-prewarm-multi-arch-v2/data/runtimes/wheel/release-wheel-multi-arch-7-14-1-deadbeef/engines/lemonade/runtime/bin/llamacpp/rocm-stable/llama-b9753/llama-server"

  prewarm_pid="$(spawn_decoy "${prewarm_decoy}")"
  workload_pid="$(spawn_decoy "${workload_decoy}")"
  harness_pid="$(spawn_decoy "${harness_decoy}")"
  stubborn_pid="$(spawn_stubborn_decoy "${stubborn_decoy}")"
  # Give the decoys a moment to appear in /proc with their full argv.
  sleep 1

  # 1. Regression guard: the patterns this script replaced could not see a
  #    pre-warm engine process. If this ever matches, the decoy stopped being
  #    representative and the rest of the self-test proves nothing.
  if pgrep -f '/tmp/rocm-e2e.*llama-server' >/dev/null 2>&1 ||
    pgrep -f 'e2e-shared.*llama-server' >/dev/null 2>&1; then
    echo "FAIL: superseded patterns matched the pre-warm decoy; decoy is unrepresentative"
    failures=$((failures + 1))
  else
    echo "ok: superseded patterns do not match a pre-warm engine process (the defect)"
  fi

  # 2. The new rule selects it.
  selected="$(select_leaked)"
  if grep -q "^${prewarm_pid}	" <<<"${selected}"; then
    echo "ok: pre-warm engine process is selected"
  else
    echo "FAIL: pre-warm engine process was not selected"
    failures=$((failures + 1))
  fi

  # 3. A manual-testing serve is left alone (engine marker, but no E2E root).
  if grep -q "^${workload_pid}	" <<<"${selected}"; then
    echo "FAIL: /workload manual serve was selected; reclaim must not touch it"
    failures=$((failures + 1))
  else
    echo "ok: /workload manual serve is not selected"
  fi

  # 4. Both halves are required: an E2E root alone must not select. Deleting the
  #    `has_engine` requirement makes exactly this check fail and nothing else.
  if grep -q "^${harness_pid}	" <<<"${selected}"; then
    echo "FAIL: E2E test binary was selected; the engine half of the rule is not enforced"
    failures=$((failures + 1))
  else
    echo "ok: an E2E root without an engine marker is not selected"
  fi

  # 5. A process that ignores SIGTERM is still selected.
  if grep -q "^${stubborn_pid}	" <<<"${selected}"; then
    echo "ok: SIGTERM-ignoring pre-warm engine process is selected"
  else
    echo "FAIL: SIGTERM-ignoring pre-warm engine process was not selected"
    failures=$((failures + 1))
  fi

  # 6. End to end: reclaim kills the leaks and spares both bystanders.
  reclaim_out="$(reclaim 0)"
  sleep 1
  if process_alive "${prewarm_pid}"; then
    echo "FAIL: pre-warm engine process survived reclaim"
    failures=$((failures + 1))
  else
    echo "ok: pre-warm engine process was reclaimed"
  fi
  if process_alive "${workload_pid}"; then
    echo "ok: /workload manual serve survived reclaim"
  else
    echo "FAIL: /workload manual serve was killed by reclaim"
    failures=$((failures + 1))
  fi
  if process_alive "${harness_pid}"; then
    echo "ok: E2E test binary survived reclaim"
  else
    echo "FAIL: E2E test binary was killed by reclaim"
    failures=$((failures + 1))
  fi

  # 7. The escalation ran, and ran only where it was needed. Asserting the
  #    stubborn decoy died covers the SIGKILL block; asserting the ordinary
  #    decoy did NOT reach escalation covers the `kill -TERM` that precedes it,
  #    which would otherwise be silently replaceable by any no-op.
  if process_alive "${stubborn_pid}"; then
    echo "FAIL: SIGTERM-ignoring process survived reclaim; escalation to SIGKILL did not happen"
    failures=$((failures + 1))
  else
    echo "ok: SIGTERM-ignoring process was escalated to SIGKILL"
  fi
  if grep -q "pid=${stubborn_pid} ignored SIGTERM" <<<"${reclaim_out}"; then
    echo "ok: escalation was reported for the process that ignored SIGTERM"
  else
    echo "FAIL: no escalation reported for the SIGTERM-ignoring process"
    failures=$((failures + 1))
  fi
  if grep -q "pid=${prewarm_pid} ignored SIGTERM" <<<"${reclaim_out}"; then
    echo "FAIL: ordinary decoy reached SIGKILL escalation; SIGTERM is not being delivered"
    failures=$((failures + 1))
  else
    echo "ok: ordinary decoy exited on SIGTERM without escalation"
  fi

  kill -KILL "${workload_pid}" "${harness_pid}" "${stubborn_pid}" 2>/dev/null || true

  if [[ "${failures}" -ne 0 ]]; then
    echo "reclaim-gpu self-test: ${failures} failure(s)"
    return 1
  fi
  echo "reclaim-gpu self-test: all checks passed"
}

main() {
  case "${1:-}" in
    '')
      reclaim 0
      # Scenario temp dirs are recreated per run; clearing them keeps a wedged
      # runner's disk from filling with dead scenario state.
      rm -rf /tmp/rocm-e2e-* 2>/dev/null || true
      ;;
    --dry-run) reclaim 1 ;;
    --report-holders) report_holders ;;
    --self-test) self_test ;;
    -h | --help) usage ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
