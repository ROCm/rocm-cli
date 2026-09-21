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

# Read a process's command line, NUL-separated in /proc, as a single string.
# Fails when the process is gone or its command line is empty (a kernel thread,
# or a zombie whose argv has already been released).
#
# Redirect stderr BEFORE the input redirection: the shell applies them left to
# right, so `<file 2>/dev/null` still lets the shell's own "No such file" reach
# the terminal when the open fails.
cmdline_of() {
  local cmdline
  cmdline="$(tr '\0' ' ' 2>/dev/null <"/proc/${1}/cmdline")" || return 1
  [[ -n "${cmdline}" ]] || return 1
  printf '%s' "${cmdline}"
}

# True when a command line names an E2E root AND an engine marker.
#
# Split out of select_leaked so the self-test can drive every root and every
# marker directly. Exercising them through spawned processes would need one
# process per list entry; leaving them undriven is how `e2e-shared`, `vllm`,
# `__engine-serve-http` and `rocm daemon` came to be removable with the
# self-test still green.
cmdline_matches_rule() {
  local cmdline="$1"
  local root engine
  local has_root=0
  local has_engine=0
  for root in "${E2E_ROOTS[@]}"; do
    case "${cmdline}" in
      *"${root}"*)
        has_root=1
        break
        ;;
      *) ;;
    esac
  done
  [[ "${has_root}" == 1 ]] || return 1
  for engine in "${ENGINE_MARKERS[@]}"; do
    case "${cmdline}" in
      *"${engine}"*)
        has_engine=1
        break
        ;;
      *) ;;
    esac
  done
  [[ "${has_engine}" == 1 ]]
}

# Whether <pid> is still running the command line it was selected with.
#   0  same process
#   1  a DIFFERENT command line — the pid was recycled
#   2  gone: exited between the liveness check and this read
# The 1/2 split matters because they are not the same event, and reporting a
# process that simply exited as "recycled" misdescribes a benign race.
same_selected_process() {
  local pid="$1"
  local expected="$2"
  local current
  current="$(cmdline_of "${pid}")" || return 2
  [[ "${current}" == "${expected}" ]]
}

# Print "pid<TAB>command line" for every process whose command line names both
# an E2E root and an engine marker.
select_leaked() {
  local cmdline_file pid cmdline
  for cmdline_file in /proc/[0-9]*/cmdline; do
    pid="${cmdline_file#/proc/}"
    pid="${pid%/cmdline}"
    # Never reclaim ourselves or our own shell.
    if [[ "${pid}" == "$$" || "${pid}" == "${PPID}" ]]; then
      continue
    fi
    # A process can exit between the glob and the read; that is not an error.
    cmdline="$(cmdline_of "${pid}")" || continue

    if [[ -n "${SELFTEST_SCOPE}" ]]; then
      case "${cmdline}" in
        *"${SELFTEST_SCOPE}"*) ;;
        *) continue ;;
      esac
    fi

    cmdline_matches_rule "${cmdline}" || continue

    printf '%s\t%s\n' "${pid}" "${cmdline}"
  done
}

# TERM, wait out the grace period, then KILL whatever is left. Reports every
# process it acts on: an unconditional "reclaimed" tells the next reader
# nothing, and a silent no-op is how the original defect stayed hidden.
reclaim() {
  local dry_run="$1"
  local selected pid cmdline rc
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
    rc=0
    same_selected_process "${pid}" "${cmdline}" || rc=$?
    case "${rc}" in
      0)
        echo "reclaim: pid=${pid} ignored SIGTERM after ${TERM_GRACE_SECS}s, sending SIGKILL"
        kill -KILL "${pid}" 2>/dev/null || true
        ;;
      2) echo "reclaim: pid=${pid} exited during the grace period" ;;
      *) echo "reclaim: pid=${pid} was recycled during the grace period, not escalating" ;;
    esac
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

# Spawn a process that becomes a ZOMBIE and stays one, plus the keeper holding
# it in that state. Echoes "<zombie pid> <keeper pid>"; the zombie pid is empty
# if it could not be produced.
#
# A zombie is the second way into cmdline_of's failure path and the only one
# that reaches its emptiness check: /proc/<pid>/cmdline still OPENS for a
# zombie, it just reads zero bytes, because the kernel has released the argv
# while the process table entry remains.
spawn_zombie() {
  local dir="$1"
  local keeper_pid zombie_pid state
  local waited=0
  mkdir -p "${dir}"
  # The subshell starts a short-lived child and then `exec`s a long sleep, so
  # the parent that would reap it is replaced by a process that never calls
  # wait(). The child therefore stays a zombie for as long as the keeper lives.
  # Bash reaps its OWN background children, which is why this needs the exec.
  (
    sleep 0.1 &
    echo $! >"${dir}/zombie.pid"
    exec sleep 300
  ) >/dev/null 2>&1 &
  keeper_pid=$!
  # Wait for the child to actually reach state Z rather than assuming it has.
  while [[ "${waited}" -lt 50 ]]; do
    if [[ -s "${dir}/zombie.pid" ]]; then
      zombie_pid="$(cat "${dir}/zombie.pid")"
      state="$(sed 's/.*) //' "/proc/${zombie_pid}/stat" 2>/dev/null | cut -d' ' -f1)" || state=''
      [[ "${state}" == "Z" ]] && break
    fi
    sleep 0.1
    waited=$((waited + 1))
  done
  # Sentinel, not an empty field: `read` skips leading whitespace, so echoing an
  # empty first field would shift the KEEPER's pid into the caller's zombie_pid
  # and make the "could not produce one" branch unreachable.
  [[ "${state:-}" == "Z" ]] || zombie_pid='none'
  echo "${zombie_pid} ${keeper_pid}"
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

# Every root and every marker, asserted individually against the rule.
#
# The expectation is written out rather than derived from the arrays: a loop
# over E2E_ROOTS cannot notice a root DELETED from E2E_ROOTS, which is exactly
# the drift that left `e2e-shared`, `vllm`, `__engine-serve-http` and
# `rocm daemon` removable with the self-test green. The duplication is the
# point — changing either list must be a deliberate edit in two places, one of
# which names the PowerShell mirrors that also have to move.
assert_rule_covers_every_list_entry() {
  local failures=0
  local root engine
  local expected_roots=('/tmp/rocm-e2e' 'e2e-shared' 'e2e-prewarm' 'e2e-target')
  local expected_markers=('llama-server' 'vllm' '__engine-serve-http' 'rocm daemon')

  if [[ "$(printf '%s\n' "${E2E_ROOTS[@]}")" != "$(printf '%s\n' "${expected_roots[@]}")" ]]; then
    echo "FAIL: E2E_ROOTS changed — update this expectation AND both PowerShell mirrors"
    echo "      script:   ${E2E_ROOTS[*]}"
    echo "      expected: ${expected_roots[*]}"
    failures=$((failures + 1))
  fi
  if [[ "$(printf '%s\n' "${ENGINE_MARKERS[@]}")" != "$(printf '%s\n' "${expected_markers[@]}")" ]]; then
    echo "FAIL: ENGINE_MARKERS changed — update this expectation AND both PowerShell mirrors"
    echo "      script:   ${ENGINE_MARKERS[*]}"
    echo "      expected: ${expected_markers[*]}"
    failures=$((failures + 1))
  fi

  for root in "${expected_roots[@]}"; do
    if ! cmdline_matches_rule "${root}/bin/llama-server --model m"; then
      echo "FAIL: root '${root}' with an engine marker does not match the rule"
      failures=$((failures + 1))
    fi
    if cmdline_matches_rule "${root}/bin/e2e-harness --exact"; then
      echo "FAIL: root '${root}' matched with NO engine marker; the AND rule is broken"
      failures=$((failures + 1))
    fi
  done

  for engine in "${expected_markers[@]}"; do
    if ! cmdline_matches_rule "e2e-prewarm/bin/${engine} --serve"; then
      echo "FAIL: marker '${engine}' under an E2E root does not match the rule"
      failures=$((failures + 1))
    fi
    if cmdline_matches_rule "workload/bin/${engine} --serve"; then
      echo "FAIL: marker '${engine}' matched with NO E2E root; a manual serve is not safe"
      failures=$((failures + 1))
    fi
  done

  if [[ "${failures}" -eq 0 ]]; then
    echo "ok: every E2E root and every engine marker is individually enforced"
  fi
  return "${failures}"
}

self_test() {
  local tmp prewarm_decoy workload_decoy harness_decoy stubborn_decoy
  local prewarm_pid workload_pid harness_pid stubborn_pid selected reclaim_out
  local escapee_pid escapee_cmd guard_rc probe_cmd
  local zombie_pid zombie_keeper_pid
  local superseded_hit=0
  local probe_failures=0
  local containment_failures=0
  local list_failures=0
  local failures=0

  assert_rule_covers_every_list_entry || list_failures=$?
  failures=$((failures + list_failures))

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
  read -r zombie_pid zombie_keeper_pid <<<"$(spawn_zombie "${tmp}/zombie")"
  # Give the decoys a moment to appear in /proc with their full argv.
  sleep 1

  # 1. Regression guard: the patterns this script replaced could not see a
  #    pre-warm engine process. If this ever matches, the decoy stopped being
  #    representative and the rest of the self-test proves nothing.
  #
  #    Asked of the decoys' own command lines, not of the machine. The obvious
  #    `pgrep -f '<pattern>'` also matches the shell that INVOKED it whenever
  #    that shell's own command line contains the pattern, so it reports a
  #    match with no such process alive; on a shared self-hosted runner an
  #    unrelated process would fail it too. Both are false FAILs in the check
  #    whose entire job is to say "the decoy is unrepresentative".
  for probe_cmd in "$(cmdline_of "${prewarm_pid}")" "$(cmdline_of "${stubborn_pid}")"; do
    # An unreadable probe would leave every pattern unmatched and print "ok"
    # having tested nothing. Same reason the zombie arm below fails loudly
    # rather than skipping: a check that cannot run must not report a pass.
    if [[ -z "${probe_cmd}" ]]; then
      echo "FAIL: could not read a decoy's command line; the regression guard tested nothing"
      probe_failures=$((probe_failures + 1))
      continue
    fi
    if [[ "${probe_cmd}" =~ /tmp/rocm-e2e.*llama-server ]] ||
      [[ "${probe_cmd}" =~ e2e-shared.*llama-server ]]; then
      superseded_hit=1
    fi
  done
  failures=$((failures + probe_failures))
  if [[ "${superseded_hit}" == 1 ]]; then
    echo "FAIL: superseded patterns matched the pre-warm decoy; decoy is unrepresentative"
    failures=$((failures + 1))
  elif [[ "${probe_failures}" -eq 0 ]]; then
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

  # 6. Containment: self_test issues REAL kills, and SELFTEST_SCOPE is the only
  #    thing keeping them inside the scratch tree. Assert that before killing
  #    rather than trusting it — this step runs on a hosted ephemeral lane
  #    today, but nothing in the script stops it being run anywhere else.
  while IFS=$'\t' read -r escapee_pid escapee_cmd; do
    [[ -n "${escapee_pid}" ]] || continue
    case "${escapee_cmd}" in
      *"${tmp}"*) ;;
      *)
        echo "FAIL: selection escaped the scratch tree: pid=${escapee_pid} cmd=${escapee_cmd}"
        containment_failures=$((containment_failures + 1))
        ;;
    esac
  done <<<"${selected}"
  failures=$((failures + containment_failures))
  # Counted on its own rather than off the running total: gating this line on
  # the total suppressed it whenever an EARLIER check had failed — which is
  # precisely the run whose output someone is reading.
  if [[ "${containment_failures}" -eq 0 ]]; then
    echo "ok: every selected process lies inside the self-test scratch tree"
  fi

  # 7. The escalation guard's comparison, in all three directions — and for
  #    "gone", by BOTH routes into it.
  #
  #    NOTE: only the COMPARISON is covered. The guard's call site is exercised
  #    solely in the always-escalate direction, because making a pid be reused
  #    by a different process on demand is not reproducible in a test — so
  #    deleting the call site still passes the self-test. Said plainly rather
  #    than implied by a green run.
  if same_selected_process "${stubborn_pid}" "$(cmdline_of "${stubborn_pid}")"; then
    echo "ok: escalation guard accepts an unchanged command line"
  else
    echo "FAIL: escalation guard rejected an unchanged command line; nothing would escalate"
    failures=$((failures + 1))
  fi
  guard_rc=0
  same_selected_process "${stubborn_pid}" "/some/other/process --unrelated" || guard_rc=$?
  if [[ "${guard_rc}" == 1 ]]; then
    echo "ok: escalation guard rejects a recycled pid"
  else
    echo "FAIL: escalation guard did not report a changed command line as recycled (rc=${guard_rc})"
    failures=$((failures + 1))
  fi
  guard_rc=0
  # A pid above /proc/sys/kernel/pid_max cannot exist, so this is the "gone" arm.
  same_selected_process 2147483647 "anything" || guard_rc=$?
  if [[ "${guard_rc}" == 2 ]]; then
    echo "ok: escalation guard reports a departed process as gone, not recycled"
  else
    echo "FAIL: escalation guard conflated a departed process with a recycled one (rc=${guard_rc})"
    failures=$((failures + 1))
  fi
  # The other route into "gone": the open SUCCEEDS and reads nothing. The arm
  # above exercises only a failed OPEN, so without this one `cmdline_of`'s
  # emptiness check can be deleted with the self-test still green — restoring
  # the "was recycled" mislabel for a process that merely exited, which is the
  # race the guard exists to describe correctly.
  if [[ "${zombie_pid}" != "none" ]]; then
    guard_rc=0
    same_selected_process "${zombie_pid}" "anything" || guard_rc=$?
    if [[ "${guard_rc}" == 2 ]]; then
      echo "ok: escalation guard reports an argv-less zombie as gone, not recycled"
    else
      echo "FAIL: zombie with an empty command line was not reported as gone (rc=${guard_rc})"
      failures=$((failures + 1))
    fi
  else
    # Failing rather than skipping: an arm that silently does not run is the
    # exact defect this check was added to close.
    echo "FAIL: could not produce a zombie decoy; the empty-cmdline arm went untested"
    failures=$((failures + 1))
  fi

  # 8. End to end: reclaim kills the leaks and spares both bystanders.
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

  # 9. The escalation ran, and ran only where it was needed. Asserting the
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

  # Includes prewarm_pid: on a GREEN run reclaim has already killed it, but on a
  # FAILED run it was not selected, and it would otherwise outlive the scratch
  # tree for its full 300s as an orphan. Killing the zombie's keeper lets init
  # reap the zombie itself.
  kill -KILL "${prewarm_pid}" "${workload_pid}" "${harness_pid}" "${stubborn_pid}" \
    "${zombie_keeper_pid}" 2>/dev/null || true

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
