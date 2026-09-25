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
#   /tmp/rocm-e2e   per-scenario temp dirs (plus the TMPDIR-derived form below)
#   e2e-shared      E2E_SHARED_CACHE_DIR (models, HF weights)
#   e2e-prewarm     E2E_SHARED_RUNTIMES_DIR — the pre-warmed runtime tree the
#                   engine binaries actually live in
#   e2e-target      CARGO_TARGET_DIR for the suite's own binaries
#
# The scenario root stays ABSOLUTE, and follows TMPDIR.
#
# Scenario dirs come from `TempDir::with_prefix("rocm-e2e-")`, which builds on
# `std::env::temp_dir()` and so follows TMPDIR; the nightly Strix lane redirects
# it to `$HOME/actions-runner/tmp`. A bare `/tmp/rocm-e2e` kept matching there
# only by accident — that directory is itself named `tmp`, so the substring
# still appeared — and would silently stop matching under any redirect not
# named `tmp`. Deriving the redirected form fixes that for ANY target.
#
# WHAT THE ANCHORING DOES AND DOES NOT BUY. Roots are matched as unanchored
# substrings of the WHOLE command line, so any root also matches a process that
# merely mentions it in an ARGUMENT. The bare segment `rocm-e2e` — what both
# PowerShell mirrors use — makes that trivially reachable: a hand-run
# `--hf-repo myorg/rocm-e2e-baseline-7b` on a shared runner is selected and
# SIGKILLed. Measured, and briefly shipped, which is why it is spelled out.
#
# Anchoring the scenario root to an absolute path NARROWS that surface. It does
# not close it, and the `/workload` promise below is correspondingly weaker than
# it reads:
#
#   - under a redirected TMPDIR, an argument naming a sibling path still
#     matches — `--extra-data-dir /tmp/mydir/rocm-e2e-notes` with
#     TMPDIR=/tmp/mydir;
#   - the three segment roots are NOT anchored at all, and "shared" and "target"
#     are ordinary words. `--model-path /home/dev/e2e-shared-models/x.gguf` and
#     `--served-model-name e2e-target-vs-baseline` are both selected today.
#
# Both predate this script and neither is fixed here; closing them properly
# means matching on a path boundary rather than a substring, across the bash
# rule and both mirrors together. Stated rather than implied, because an earlier
# revision of this comment claimed these roots "name directories no argument
# plausibly carries" — which is false, and is the same overclaim that made the
# bare-segment change look safe.
E2E_ROOTS=(
  '/tmp/rocm-e2e'
  'e2e-shared'
  'e2e-prewarm'
  'e2e-target'
)
# Appended rather than listed, because it is only known at run time. Skipped
# when TMPDIR is unset or already /tmp, so the list stays exactly the pinned one
# on every lane that does not redirect. A RELATIVE TMPDIR is skipped too: it
# would append a relative root and quietly give up the anchoring above, and no
# lane sets one.
if [[ "${TMPDIR:-}" == /* && "${TMPDIR%/}" != "/tmp" ]]; then
  E2E_ROOTS+=("${TMPDIR%/}/rocm-e2e")
fi

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
# NEWLINE is flattened along with the NUL separators the kernel inserted,
# because it is the record terminator `select_leaked` emits and `reclaim` reads
# back. A newline INSIDE an argv element is content, not a separator, and
# `--chat-template` or `--prompt` can carry one. Left intact, such a process
# emits a record that splits into two on the way back: the real pid arrives
# carrying a truncated command line, fails the identity check against its own
# full one, and is reported "recycled ... not signalling" — so the leak survives
# both TERM and KILL while the summary prints "0 process(es) terminated". That is
# the silent miss this whole script exists to end, so it is fixed at the single
# point every consumer already reads through rather than at each of them.
#
# The TAB is deliberately NOT flattened, though it is the field separator. It
# does not need to be, measured rather than reasoned:
#
#   record   "<pid>\t/path/llama-server --tmpl a\tb "
#   read -r pid cmdline  ->  pid=[<pid>]  cmdline=[/path/llama-server --tmpl a\tb ]
#
# `read` gives leftover words AND their intervening separators to the LAST name,
# so a tab EMBEDDED in the command line lands in `cmdline` verbatim. Separators
# are stripped only where the assigned field BEGINS or ENDS with them, and BOTH
# edges are ruled out here — each by its own property, so both are named:
#
#   - leading: the command line starts with argv[0], a path, so the trailing
#     field never begins with a separator. (Were it to, a leading tab WOULD be
#     stripped: `IFS=$'\t' read -r pid cmdline <<<$'1\t\tx'` yields `x`.)
#   - trailing: flattening the final NUL always leaves the string ending in a
#     space, so a tab is never the last byte. This one is load-bearing and
#     measured — with the trailing space `…arg\t ` survives intact, without it
#     the same tab is stripped. The same property is relied on again at the
#     self-test's record check; if it ever changes, both move together.
#
# An earlier revision also claimed the tab "shifts the field boundary" and
# flattened it for that reason — also false, and deleting the tab from the
# flattening set failed no check in this file. Flattening it anyway would have
# coarsened the identity comparison below for nothing.
#
# Matching is unaffected either way: no root or marker contains a newline. The
# identity comparison in same_selected_process is coarsened only for the newline
# — two command lines differing ONLY in a newline-versus-space at one position
# now read as equal. Both sides come through here, so they are flattened alike.
# That is a real cost, weighed against a silent miss any engine invocation
# carrying a prompt can reach.
#
# Redirect stderr BEFORE the input redirection: the shell applies them left to
# right, so `<file 2>/dev/null` still lets the shell's own "No such file" reach
# the terminal when the open fails.
cmdline_of() {
  local cmdline
  cmdline="$(tr '\0\n' '  ' 2>/dev/null <"/proc/${1}/cmdline")" || return 1
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
      # Same guard as the escalation below, and for the same reason. The window
      # is shorter here — no grace period — but it is not zero: select_leaked
      # walks the whole of /proc, so a pid can be freed and reissued between
      # the scan that selected it and this signal. A bystander is just as dead
      # from TERM as from KILL, so the check cannot be reserved for the latter.
      rc=0
      same_selected_process "${pid}" "${cmdline}" || rc=$?
      case "${rc}" in
        0) ;;
        2)
          echo "reclaim: pid=${pid} exited before it could be terminated"
          continue
          ;;
        *)
          echo "reclaim: pid=${pid} was recycled before it could be terminated, not signalling"
          continue
          ;;
      esac
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
  local engine
  local -a marker_args=()
  echo "--- rocm-smi KFD processes (per-process VRAM) ---"
  # Process names show as UNKNOWN inside a container: KFD reports host PIDs,
  # which do not resolve in the container's PID namespace. The VRAM column is
  # still the answer to "what is holding the card".
  timeout 15 rocm-smi --showpids 2>&1 | head -40 || true
  echo "--- engine/serve processes visible here ---"
  # Derived from ENGINE_MARKERS rather than spelled out again. A second copy of
  # the list is exactly the drift the PowerShell mirrors demonstrated and that
  # the xtask contract test exists to catch — and this call site had no such
  # guard, so a marker added above would have quietly stopped appearing here.
  # Deriving it needs no guard: there is no longer a copy that can go stale.
  #
  # -F, not an -E alternation: the markers are literal substrings everywhere
  # else in this script, and building a regex out of them would give a future
  # marker containing a metacharacter a different meaning here than in the rule.
  #
  # No empty-array guard. Given zero patterns grep prints its usage to stderr
  # and exits 2; the `|| true` below swallows the STATUS, so the section would
  # carry that usage error in place of any holders — wrong, but not silent.
  # Reaching it needs ENGINE_MARKERS to be empty, which
  # assert_rule_covers_every_list_entry already fails on: it pins the list to
  # its four entries by exact comparison, so an emptied array reds the
  # self-test before anything gets here. Guarding it again would add a branch
  # the self-test could never reach, which is worth less than the assertion
  # that already covers it.
  for engine in "${ENGINE_MARKERS[@]}"; do
    marker_args+=(-e "${engine}")
  done
  # shellcheck disable=SC2009 # pgrep cannot print elapsed time, and how long a
  # holder has been alive is what distinguishes a leak from this job's own serve.
  ps -eo pid,etimes,args 2>/dev/null |
    grep -Fi "${marker_args[@]}" |
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
# it in that state. Echoes "<zombie pid> <keeper pid>"; the zombie pid is the
# literal string `none` if one could not be produced — NOT an empty field, for
# the reason given at the sentinel itself. Test `!= "none"`, never `-z`.
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

# Spawn a decoy carrying the record delimiters INSIDE one argument — a newline
# and a tab, as a `--chat-template` or `--prompt` value does.
#
# Not spawn_decoy with an extra argument: that decoy is `cp /bin/sleep`, and
# sleep rejects a non-numeric argument and exits before it can be observed. This
# needs something that ignores what it is handed and stays alive.
#
# Ordinary in every other respect — it must die on SIGTERM, so it is the stubborn
# decoy's body without the trap. Its `sleep 1` children name no E2E root, so they
# are never selected, and none outlives the scratch tree by more than a second.
spawn_delimiter_decoy() {
  local path="$1"
  mkdir -p "$(dirname "${path}")"
  cat >"${path}" <<'DECOY'
#!/usr/bin/env bash
while :; do sleep 1; done
DECOY
  chmod +x "${path}"
  "${path}" $'--chat-template\nrole:\tuser' >/dev/null 2>&1 &
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
  # The TMPDIR-derived scenario root is appended at load time, so the pinned
  # list is the STATIC one plus that entry when it applies. Computed the same
  # way the script does rather than assumed absent: this function also runs on a
  # developer machine, and macOS sets TMPDIR for every shell.
  if [[ "${TMPDIR:-}" == /* && "${TMPDIR%/}" != "/tmp" ]]; then
    expected_roots+=("${TMPDIR%/}/rocm-e2e")
  fi

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

  # The scenario root must stay ANCHORED to an absolute path. These are argument
  # strings, not paths: a hand-run serve on a shared runner that merely NAMES an
  # E2E-suite baseline. A bare `rocm-e2e` root matches both and SIGKILLs them.
  # The `/workload` fixtures above cannot catch it — they carry no arguments at
  # all, so they stay green against exactly this mistake. It was made, and this
  # is what would have caught it.
  #
  # This pins the scenario root ONLY. The header explains why that is a
  # narrowing rather than a fix: the three segment roots stay unanchored, so
  # `--model-path /home/dev/e2e-shared-models/x.gguf` is still selected. No
  # assertion here claims otherwise, deliberately — a green run means the
  # scenario root did not regress, not that a manual serve is safe.
  for root in '--hf-repo myorg/rocm-e2e-baseline-7b' '--model-alias my-rocm-e2e-comparison'; do
    if cmdline_matches_rule "/workload/manual/llama-server ${root}"; then
      echo "FAIL: a manual serve was selected for merely NAMING '${root}'; the scenario root is not anchored"
      failures=$((failures + 1))
    fi
  done

  # The other half of the same decision: anchoring must not cost the redirect.
  # Only assertable when this shell actually has a redirected TMPDIR, so it says
  # which case ran rather than reporting a pass for a branch it skipped.
  if [[ "${TMPDIR:-}" == /* && "${TMPDIR%/}" != "/tmp" ]]; then
    if cmdline_matches_rule "${TMPDIR%/}/rocm-e2e-ab12/bin/llama-server --model m"; then
      echo "ok: a scenario tree under the redirected TMPDIR is selected"
    else
      echo "FAIL: TMPDIR is redirected but a scenario tree under it is not selected"
      failures=$((failures + 1))
    fi
  else
    echo "note: no absolute TMPDIR redirect in this shell, so that arm was not exercised"
  fi

  if [[ "${failures}" -eq 0 ]]; then
    echo "ok: every E2E root and every engine marker is individually enforced"
  fi
  return "${failures}"
}

# Decoys the self-test has spawned so far, read by its EXIT trap.
#
# Global, and read at trap FIRE time rather than expanded at definition time the
# way the scratch-dir paths are: a pid is not known until its spawn returns, so
# the one thing that must not be baked into the trap string is this list.
SELFTEST_DECOY_PIDS=()

selftest_track_decoy() {
  SELFTEST_DECOY_PIDS+=("$1")
}

# SIGKILL, not TERM: one decoy ignores TERM by design, and this runs on the way
# out with nothing left to wait for it.
selftest_kill_decoys() {
  [[ "${#SELFTEST_DECOY_PIDS[@]}" -gt 0 ]] || return 0
  kill -KILL "${SELFTEST_DECOY_PIDS[@]}" 2>/dev/null || true
}

self_test() {
  local tmp prewarm_decoy workload_decoy harness_decoy stubborn_decoy
  local prewarm_pid workload_pid harness_pid stubborn_pid selected reclaim_out
  local outside outside_decoy outside_pid outside_cmd
  local delimiter_decoy delimiter_pid delimiter_raw record_pid
  local forced_out guard_pid
  local escapee_pid escapee_cmd guard_rc probe_cmd
  local zombie_pid zombie_keeper_pid
  local decoy_pids
  local superseded_hit=0
  local probe_failures=0
  local containment_failures=0
  local fixture_failures=0
  local record_failures=0
  local list_failures=0
  local failures=0

  assert_rule_covers_every_list_entry || list_failures=$?
  failures=$((failures + list_failures))

  # Deliberately NOT under /tmp/rocm-e2e: that prefix is one of the roots the
  # old patterns did match, which would mask the regression this guards.
  tmp="$(mktemp -d /tmp/reclaim-selftest-XXXXXX)"
  # Armed the moment there is something to remove, and widened below once the
  # second tree exists. The `mktemp` that follows can fail, and under `set -e`
  # that aborts the function — with `tmp` already on disk and, if the trap were
  # installed only afterwards, nothing left to clean it up.
  # shellcheck disable=SC2064 # expand ${tmp} now, at trap definition time
  trap "selftest_kill_decoys; rm -rf '${tmp}'" EXIT
  # A SECOND tree, outside the scope, for the bystander decoy below. Everything
  # reachable through SELFTEST_SCOPE is inside `tmp` by construction, so a
  # negative case for the scope filter cannot live there.
  outside="$(mktemp -d /tmp/reclaim-selftest-out-XXXXXX)"
  # Widened in the statement immediately after the one that created it, for the
  # same reason the narrow trap above exists: anything fallible in between is a
  # window where a tree is on disk with nothing arranged to remove it.
  # shellcheck disable=SC2064 # expand both paths now, at trap definition time
  trap "selftest_kill_decoys; rm -rf '${tmp}' '${outside}'" EXIT
  export RECLAIM_SELFTEST_SCOPE="${tmp}"
  SELFTEST_SCOPE="${tmp}"
  # The stubborn decoy never exits on its own, so the grace loop always runs to
  # the ceiling. Keep it short: this is a unit-speed test, not a GPU lane.
  TERM_GRACE_SECS=2

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
  # Matches the rule in full — an E2E root AND an engine marker — but lies
  # OUTSIDE the scope. It is the only fixture the SELFTEST_SCOPE filter can be
  # observed doing anything to, and therefore the only reason check 7 can fail.
  outside_decoy="${outside}/e2e-prewarm-bystander/bin/llama-server"
  # Same pre-warm shape again, but its ARGUMENTS carry a newline and a tab — the
  # two bytes the pid<TAB>cmdline<NEWLINE> record format is built out of. Without
  # a fixture whose command line contains them, the format is only ever exercised
  # on strings that cannot break it.
  delimiter_decoy="${tmp}/e2e-prewarm-multi-arch-v2/data/runtimes/wheel/release-wheel-multi-arch-7-14-1-deadbeef/engines/lemonade/runtime/bin/llamacpp/rocm-stable/llama-b9754/llama-server"

  # Each pid is registered with the EXIT trap the moment it exists, for the same
  # reason the scratch-dir traps above are widened one statement at a time:
  # registering them only in the list further down would leave every
  # ALREADY-spawned decoy running with nothing arranged to kill it — including
  # the stubborn one, which ignores SIGTERM and loops forever. Measured: a
  # failing statement between two spawns left 3 survivors before this change and
  # 0 after.
  #
  # NOT because `set -e` aborts on a failing spawn — it does not, and an earlier
  # version of this comment said it did. `inherit_errexit` is off, so errexit
  # does not reach inside `$( )`, and each helper ends in an `echo` that
  # succeeds regardless; a hard failure injected into a spawn helper runs the
  # suite to a green exit 0. The routes that DO reach the trap mid-way are a
  # signal (a cancelled CI job), and any fallible statement between the spawns
  # here — including one a later edit adds, which is the case this guards.
  #
  # The flip side is worth knowing: because a broken spawn does not abort, it
  # yields a live-looking pid for a process that died immediately, and surfaces
  # later as a confusing assertion failure rather than as "the spawn failed".
  prewarm_pid="$(spawn_decoy "${prewarm_decoy}")"
  selftest_track_decoy "${prewarm_pid}"
  workload_pid="$(spawn_decoy "${workload_decoy}")"
  selftest_track_decoy "${workload_pid}"
  harness_pid="$(spawn_decoy "${harness_decoy}")"
  selftest_track_decoy "${harness_pid}"
  stubborn_pid="$(spawn_stubborn_decoy "${stubborn_decoy}")"
  selftest_track_decoy "${stubborn_pid}"
  outside_pid="$(spawn_decoy "${outside_decoy}")"
  selftest_track_decoy "${outside_pid}"
  delimiter_pid="$(spawn_delimiter_decoy "${delimiter_decoy}")"
  selftest_track_decoy "${delimiter_pid}"
  read -r zombie_pid zombie_keeper_pid <<<"$(spawn_zombie "${tmp}/zombie")"
  selftest_track_decoy "${zombie_keeper_pid}"
  # Every process this function spawned that can still be signalled, so cleanup
  # is one list rather than a line kept in step at each early return. The zombie
  # itself is absent deliberately: it is already dead, and killing its keeper is
  # what lets init reap it. Same set the trap holds, by construction.
  decoy_pids=("${SELFTEST_DECOY_PIDS[@]}")
  # The registration itself, pinned. Without this, dropping a
  # `selftest_track_decoy` call is silent AND leaks for real: both the EXIT trap
  # and the final kill read this same list, so an untracked decoy is signalled
  # by nothing. It survives even a GREEN run — measured, the `/workload` fixture
  # outlives the suite by its full 300s, because `reclaim` is supposed to spare
  # that one and the tracked list is the only other thing that would kill it.
  # (An earlier version of this comment claimed such a decoy "is still killed at
  # the end". It is not, and it is only incidentally true for the three fixtures
  # `reclaim` itself signals.)
  #
  # LIMITATION, since an exact literal cannot carry its own maintenance: this
  # catches a tracking call REMOVED from an existing decoy, where the count
  # drops. It does NOT catch a decoy ADDED without one — the count stays at the
  # stale literal and the run passes. Adding a spawn means bumping this number
  # in the same edit; there is no mechanism enforcing that.
  if [[ "${#SELFTEST_DECOY_PIDS[@]}" -ne 7 ]]; then
    echo "FAIL: ${#SELFTEST_DECOY_PIDS[@]} decoys registered with the EXIT trap, expected 7"
    echo "      tracked: ${SELFTEST_DECOY_PIDS[*]}"
    echo "      a spawn is not being tracked; it would outlive the run, aborted or not"
    failures=$((failures + 1))
  fi
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

  # 3. The record format survives a command line containing BOTH delimiter bytes.
  #
  #    select_leaked emits "pid<TAB>cmdline<NEWLINE>" and reclaim reads it back
  #    with `IFS=$'\t' read -r`. The two bytes are not symmetric, and this check
  #    is the only place that says so:
  #
  #    - a NEWLINE in an argv element splits one record into two. The genuine pid
  #      then arrives with a TRUNCATED command line, fails the identity check
  #      against its own full one, and is passed over as "recycled": the leak
  #      survives TERM and KILL alike while the run signs off with "0 process(es)
  #      terminated". cmdline_of flattens it for that reason, and deleting that
  #      flattening fails this check.
  #    - a TAB does NOT, even though it is the field separator, because `read`
  #      hands leftover separators to the last name and the trailing space left
  #      by the final NUL keeps a tab from ever being stripped as a trailing one.
  #      cmdline_of deliberately leaves it alone; see the reasoning there.
  #
  #    The tab fixture therefore guards the FORMAT, not the flattening: it is
  #    what fails if the record shape is ever changed to something a tab can
  #    break. Asserting it as though it pinned a flattening step would be a claim
  #    of coverage that no mutation can falsify, which is how this check read
  #    before — it announced the tab as individually asserted while deleting the
  #    tab from the flattening set left every check green.
  #
  #    Placed HERE, ahead of containment, deliberately. A split record also
  #    trips check 7's escape loop — its continuation line names no scratch
  #    tree — which arms the gate and returns before checks 8-11 ever run. The
  #    run is red either way; what this adds is the true cause, printed before
  #    the one check 7 would otherwise report in its place.
  # Read RAW, flattening only the kernel's NUL separators: cmdline_of flattens
  # the newline, so asking it would report the fixture carries one no matter what
  # the decoy was actually given.
  # Counted apart from the record loop below, for the same reason check 7 keeps
  # its empty-selection count separate: these say the FIXTURE is unusable, the
  # loop says the FORMAT is broken, and the "ok:" line must not be able to
  # affirm the second while the first has just been denied.
  # stderr redirected BEFORE the input, for the reason cmdline_of spells out:
  # the shell applies them left to right, so the input-first form lets its own
  # "No such file" reach the terminal on a failed open — here, noise printed by
  # the very branch that exists to report a dead fixture cleanly.
  if ! delimiter_raw="$(tr '\0' ' ' 2>/dev/null <"/proc/${delimiter_pid}/cmdline")"; then
    # Distinguished from "carries no delimiter": if the decoy died before this
    # read, the byte assertions below would report a format problem for what is
    # really a dead fixture, and send the next reader after the wrong thing.
    echo "FAIL: delimiter decoy's command line could not be read; the fixture is gone, not malformed"
    fixture_failures=$((fixture_failures + 1))
    delimiter_raw=''
  else
    case "${delimiter_raw}" in
      *$'\n'*) ;;
      *)
        echo "FAIL: delimiter decoy's command line carries no newline; the record-format check is vacuous"
        fixture_failures=$((fixture_failures + 1))
        ;;
    esac
    case "${delimiter_raw}" in
      *$'\t'*) ;;
      *)
        echo "FAIL: delimiter decoy's command line carries no tab; the record format is no longer exercised against one"
        fixture_failures=$((fixture_failures + 1))
        ;;
    esac
  fi
  if ! grep -q "^${delimiter_pid}	" <<<"${selected}"; then
    # Selection is NOT what this check tests — and note it survives the
    # unflattened form, because the `^pid<TAB>` grep matches the first line of a
    # split record. It is asserted only so the loop below cannot report a pass
    # having never seen a command line with a delimiter in it.
    echo "FAIL: delimiter decoy was not selected; the record-format check has no fixture"
    fixture_failures=$((fixture_failures + 1))
  fi
  failures=$((failures + fixture_failures))
  while IFS=$'\t' read -r record_pid _; do
    [[ -n "${record_pid}" ]] || continue
    if [[ ! "${record_pid}" =~ ^[0-9]+$ ]]; then
      echo "FAIL: selection record does not begin with a pid: '${record_pid}'"
      record_failures=$((record_failures + 1))
    fi
  done <<<"${selected}"
  failures=$((failures + record_failures))
  # Gated on the fixture too, not just on the record loop. A loop that examined
  # a command line with no delimiter in it proves nothing, so affirming it here
  # would print "every record survives delimiters" directly beneath a FAIL line
  # saying there were none — evidence contradicting itself on the run someone is
  # reading, which is the very thing check 7 is shaped to avoid.
  if [[ "${record_failures}" -eq 0 && "${fixture_failures}" -eq 0 ]]; then
    echo "ok: every selection record survives delimiters in a command line"
  fi

  # 4. A manual-testing serve is left alone (engine marker, but no E2E root).
  if grep -q "^${workload_pid}	" <<<"${selected}"; then
    echo "FAIL: /workload manual serve was selected; reclaim must not touch it"
    failures=$((failures + 1))
  else
    echo "ok: /workload manual serve is not selected"
  fi

  # 5. Both halves are required: an E2E root alone must not select. Deleting the
  #    `has_engine` requirement makes exactly this check fail and nothing else.
  if grep -q "^${harness_pid}	" <<<"${selected}"; then
    echo "FAIL: E2E test binary was selected; the engine half of the rule is not enforced"
    failures=$((failures + 1))
  else
    echo "ok: an E2E root without an engine marker is not selected"
  fi

  # 6. A process that ignores SIGTERM is still selected.
  if grep -q "^${stubborn_pid}	" <<<"${selected}"; then
    echo "ok: SIGTERM-ignoring pre-warm engine process is selected"
  else
    echo "FAIL: SIGTERM-ignoring pre-warm engine process was not selected"
    failures=$((failures + 1))
  fi

  # 7. Containment: self_test issues REAL kills, and SELFTEST_SCOPE is the only
  #    thing keeping them inside the scratch tree. Assert that before killing
  #    rather than trusting it — this step runs on a hosted ephemeral lane
  #    today, but nothing in the script stops it being run anywhere else.
  # The bystander decoy is what gives this check teeth. select_leaked filters
  # every candidate against SELFTEST_SCOPE, and SELFTEST_SCOPE *is* ${tmp}, so
  # the loop below — "is every selected entry inside ${tmp}?" — re-derives its
  # answer from the very filter it claims to be checking, and cannot fail on
  # any fixture that lives inside the scope. Deleting the filter outright left
  # this check green. A rule-matching process OUTSIDE the scope is the only
  # thing the filter can be caught NOT doing its job on.
  # An empty selection satisfies the loop below trivially, so without this the
  # "ok:" line would report containment verified on a run where nothing was
  # examined. Checks 2 and 6 already fail such a run, so this is not a false
  # green — but the evidence line is read by whoever is diagnosing that run.
  #
  #    Counted apart from containment_failures on purpose: that counter arms the
  #    gate below, whose message and early return are specifically about a
  #    selection reaching OUTSIDE the scratch tree. An empty selection is the
  #    opposite failure and signals nothing at all, so routing it through that
  #    gate would report a false cause and cut the run short of checks 8-11.
  #
  #    Diagnostic only, and said plainly rather than implied: an empty selection
  #    is already caught — checks 2 and 6 fail it — so deleting this check does
  #    NOT let such a run pass. What it adds is the reason, on the run someone
  #    is reading, which is also why the "ok:" line below is gated on it.
  if [[ -z "${selected}" ]]; then
    echo "FAIL: selection was empty; containment had nothing to examine"
    failures=$((failures + 1))
  fi
  outside_cmd="$(cmdline_of "${outside_pid}")" || outside_cmd=''
  if [[ -z "${outside_cmd}" ]]; then
    echo "FAIL: bystander decoy is not running; containment has no negative case"
    containment_failures=$((containment_failures + 1))
  elif ! cmdline_matches_rule "${outside_cmd}"; then
    # Asserted, not assumed: if the bystander stopped matching the rule it
    # would be excluded for that reason instead of by the scope filter, and
    # the negative case below would pass while proving nothing.
    echo "FAIL: bystander decoy no longer matches the rule; containment proves nothing"
    containment_failures=$((containment_failures + 1))
  elif grep -q "^${outside_pid}	" <<<"${selected}"; then
    echo "FAIL: a rule-matching process outside the scratch tree was selected"
    containment_failures=$((containment_failures + 1))
  fi
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
  # Gated on a non-empty selection as well as on containment: the loop above is
  # satisfied trivially by an empty one, so without this the run would print
  # "selection was empty" and then immediately claim every selected process was
  # checked — evidence contradicting itself on the run being diagnosed.
  if [[ "${containment_failures}" -eq 0 && -n "${selected}" ]]; then
    echo "ok: every selected process lies inside the self-test scratch tree"
  elif [[ "${containment_failures}" -eq 0 ]]; then
    : # empty selection: already reported above, and nothing to affirm here
  else
    # A GATE, not a score. Check 9 below calls the real `reclaim 0`, which
    # sends real signals to whatever select_leaked returns at that moment. If
    # containment has just failed, that selection reaches outside this scratch
    # tree — the exact accident this check exists to prevent — so scoring it
    # and carrying on would let the self-test do the damage it is guarding
    # against. Stop here instead, while nothing has been signalled yet.
    echo "FAIL: containment breached; refusing to run the real reclaim"
    kill -KILL "${decoy_pids[@]}" 2>/dev/null || true
    echo "reclaim-gpu self-test: ${failures} failure(s)"
    return 1
  fi

  # 8. The escalation guard's comparison, in all three directions — and for
  #    "gone", by BOTH routes into it.
  #
  #    The COMPARISON is covered here; check 9 covers the two CALL SITES by
  #    forcing the verdict, because a pid cannot be made to be reused by a
  #    different process on demand.
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

  # 9. The guard's two CALL SITES, by forcing the verdict they act on.
  #
  #    A pid cannot be made to be reused by a different process on demand, so
  #    until now both call sites were exercised only in the always-proceed
  #    direction and either could be deleted with the self-test green. Overriding
  #    the comparison for one real `reclaim` run reaches them: with every verdict
  #    "recycled", a correct reclaim signals NOTHING, so deleting EITHER call
  #    site kills decoys here and fails the survival loop below.
  #
  #    The override is declared INSIDE the command substitution, which bash runs
  #    in a subshell, so it cannot outlive this one call — no save/restore to get
  #    wrong, and no seam in the production path. The kills a broken guard would
  #    issue are still real, which is what makes the survival loop meaningful.
  forced_out="$(
    same_selected_process() { return 1; }
    reclaim 0
  )"
  sleep 1
  if grep -q "pid=${prewarm_pid} was recycled before it could be terminated" <<<"${forced_out}"; then
    echo "ok: pre-TERM guard refused to signal a pid whose identity changed"
  else
    echo "FAIL: pre-TERM guard did not act on a recycled verdict"
    failures=$((failures + 1))
  fi
  # Bound to a specific pid, like its pre-TERM sibling above: an unbound match
  # would be satisfied by this message emitted for any process. With every
  # verdict forced to "recycled" nothing is signalled, so both engine decoys
  # survive to the escalation loop and either pid would serve; the stubborn one
  # is named because it is the decoy that reaches escalation in the UNforced
  # run too, which keeps this assertion reading the same way as check 11.
  if grep -q "pid=${stubborn_pid} was recycled during the grace period, not escalating" <<<"${forced_out}"; then
    echo "ok: pre-KILL guard refused to escalate onto a recycled pid"
  else
    echo "FAIL: pre-KILL guard did not act on a recycled verdict"
    failures=$((failures + 1))
  fi
  # The point of both guards: a forced-recycled run must leave every process alive.
  for guard_pid in "${decoy_pids[@]}"; do
    if ! process_alive "${guard_pid}"; then
      echo "FAIL: pid=${guard_pid} was signalled despite a recycled verdict"
      failures=$((failures + 1))
    fi
  done
  # 10. End to end: reclaim kills the leaks and spares all three bystanders —
  #    the manual serve, the harness binary, and the out-of-scope decoy.
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
  # The end-to-end form of check 3, and the one that states the cost: with the
  # delimiters unflattened this process is selected, passed over as "recycled",
  # and still holding the card when reclaim reports success.
  if process_alive "${delimiter_pid}"; then
    echo "FAIL: engine process with delimiters in its command line survived reclaim"
    failures=$((failures + 1))
  else
    echo "ok: engine process with delimiters in its command line was reclaimed"
  fi
  # The bystander matches the rule in full, so only the scope kept it out of the
  # selection. Its survival is the end-to-end form of check 7.
  if process_alive "${outside_pid}"; then
    echo "ok: rule-matching process outside the scratch tree survived reclaim"
  else
    echo "FAIL: reclaim killed a rule-matching process outside the scratch tree"
    failures=$((failures + 1))
  fi

  # 11. The escalation ran, and ran only where it was needed. Asserting the
  #     stubborn decoy died covers the SIGKILL block; asserting the ordinary
  #     decoy did NOT reach escalation covers the `kill -TERM` that precedes it,
  #     which would otherwise be silently replaceable by any no-op.
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
  kill -KILL "${decoy_pids[@]}" 2>/dev/null || true

  if [[ "${failures}" -ne 0 ]]; then
    echo "reclaim-gpu self-test: ${failures} failure(s)"
    return 1
  fi
  echo "reclaim-gpu self-test: all checks passed"
}

main() {
  local scenario_tmp
  case "${1:-}" in
    '')
      reclaim 0
      # Scenario temp dirs are recreated per run; clearing them keeps a wedged
      # runner's disk from filling with dead scenario state.
      #
      # TMPDIR as well as /tmp, because the suite creates these through
      # `std::env::temp_dir()` and the nightly Strix lane redirects TMPDIR to
      # the runner's home. A bare `/tmp` glob removed nothing at all there, so
      # the guard was dead on the one lane that redirects. Both are listed
      # rather than just TMPDIR: the lanes that do not set it still want /tmp,
      # and a stale tree from before a redirect was added would outlive it.
      #
      # The absolute-path test is the SAME one the roots above apply, and it
      # matters most here: this is the only consumer that deletes rather than
      # merely failing to match. Interpolated raw, a relative TMPDIR makes this
      # `rm -rf relative/dir/rocm-e2e-*` resolved against the CWD — the repo
      # checkout, on a CI runner. No lane sets one, which is a reason to skip
      # the value, not a reason to hand it to `rm -rf` unchecked.
      scenario_tmp="${TMPDIR:-}"
      [[ "${scenario_tmp}" == /* ]] || scenario_tmp='/tmp'
      rm -rf "${scenario_tmp%/}"/rocm-e2e-* /tmp/rocm-e2e-* 2>/dev/null || true
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
