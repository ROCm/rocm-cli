#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# GPU preflight for the self-hosted E2E lanes: fail fast (~90s) with a named
# reason instead of hanging to the job cap when the GPU is missing, the driver
# is wedged, or the card is still held by a serve leaked by a prior run.
#
# TWO GATES, both of which must pass, because neither sees what the other does.
#
#   1. A BOUNDED WAIT on free memory. Not a one-shot check: transient
#      contention — the reclaim step's kills still draining the card — self-
#      heals within seconds, so this polls to a ceiling and succeeds the moment
#      the GPU is both responsive AND has enough free memory. Only a genuinely
#      absent, wedged or exhausted GPU reaches the ceiling.
#
#   2. `reclaim-gpu.sh --assert-clear`: no leaked E2E engine process survives.
#
# Gate 1 alone is much coarser than "catches a leftover serve" suggests, which
# is what #446 was opened about. The floor is a fixed number of GiB applied to
# whatever pool rocm-smi reports, so what it detects is set by the POOL's size,
# not by what a serve holds:
#
#   lane                        pool      floor    trips only past
#   Strix Halo (GTT aperture)   ~62 GiB    8 GiB   ~54 GiB held
#   MI300X                      192 GiB   16 GiB   ~176 GiB held
#
# A leftover llama-server holding 10-20 GiB clears every one of them. Gate 2
# catches that process by name, through the same rules the reclaim step acts on.
#
# Gate 1 is KEPT rather than replaced, because the converse case is just as real
# and gate 2 cannot see it: a non-E2E tenant on a shared self-hosted host can
# exhaust the pool while running nothing the reclaim rules match. They are
# complementary, and each is the only one that catches its own case.
#
# ONE SCRIPT, SIX LANES. This body was six near-identical copies across
# e2e-selfhosted.yml and nightly.yml (#294), differing only in the floor — so a
# fix landed six times, and the one time it landed five, nothing noticed for
# months. The only per-lane input is GPU_PREFLIGHT_MIN_FREE_GIB. The two
# PowerShell blocks on the Windows-native lanes and the two advisory bodies on
# the WSL lanes stay where they are: native Windows has no bash to run this
# with, and the advisory lanes warn where this one fails.
#
# SHELL OPTIONS. As an inline `run:` step this ran under GitHub's documented
# default `bash -e {0}` — `-e`, but, unlike an explicit `shell: bash`, NO
# pipefail. As a standalone script it sets `-euo pipefail`, which is this repo's
# convention and what shellcheck's strict profile assumes. That promotes the
# `|| true` in `field` from insurance to load-bearing; see the note there.
#
# Env:
#   GPU_PREFLIGHT_MIN_FREE_GIB  free-memory floor, GiB. Default 16 (discrete);
#                               the APU lanes pass 8.
#   GPU_PREFLIGHT_CEILING_SECS  how long to keep polling. Default 90.

set -euo pipefail

MIN_FREE_GIB="${GPU_PREFLIGHT_MIN_FREE_GIB:-16}"
CEILING_SECS="${GPU_PREFLIGHT_CEILING_SECS:-90}"

# Named relative to the checkout, like the `--report-holders` call this step has
# always made and like every other `bash scripts/...` in these workflows: GitHub
# runs each step from the workspace root. Resolving it against this file's own
# location instead would be no more correct — the reclaim reads $PWD-relative
# state of its own — and would break the contract tests that drive this script
# from a scratch tree.
RECLAIM_SCRIPT="scripts/reclaim-gpu.sh"

min_free=$((MIN_FREE_GIB * 1024 * 1024 * 1024))
deadline=$((SECONDS + CEILING_SECS))
reason="rocm-smi never returned within its timeout (driver wedged or GPU absent)"

# `<POOL> Total [Used ]Memory (B): <n>`, last GPU's row wins. Anchoring on the
# `(B)` unit stops a differently-denominated reading being compared as bytes.
#
# The trailing `|| true` yields an empty string for a pool this host does not
# report (GTT on a discrete card), which the guards below handle. Under the
# `set -o pipefail` above it is LOAD-BEARING, which it was not inline: the final
# `grep -oE` exits 1 when the pool is absent, pipefail makes that the whole
# pipeline's status, and `-e` would abort the script on the assignment rather
# than leaving `gtt_total` empty for the carveout branch to reason about.
# Inline, under `bash -e {0}` without pipefail, the pipeline's status was
# `tail`'s and `-e` never fired — so the same four characters changed from
# insurance to the thing that keeps a discrete card off the error path.
out=""
field() { printf '%s\n' "${out}" | grep -iE "$1 \(B\):" | sed 's/.*: *//' | grep -oE '[0-9]+' | tail -1 || true; }

shown=0
# Both memos below describe the HOST, not the moment, so they are answered once
# and reused. Re-asking each tick spends the ceiling on `timeout`s rather than
# on retries, which is what the ceiling is for. Neither is ever set from a call
# that was merely cut short.
combined=1
gtt_absent=0
# Check the deadline only after a poll, never before the first one. $SECONDS
# counts whole wall-clock seconds, so a short ceiling can lapse before the loop
# is entered, and the verdict below would then describe a reading that was never
# taken.
first_poll=1
while [[ "${first_poll}" -eq 1 ]] || [[ "${SECONDS}" -lt "${deadline}" ]]; do
  first_poll=0
  out=""
  rc=0
  if [[ "${combined}" -eq 1 ]]; then
    # rocm-smi itself can hang on a wedged driver — bound every call with timeout.
    out="$(timeout 15 rocm-smi --showmeminfo vram gtt 2>/dev/null)" || rc=$?
    # Read the marker the way `field` reads the same text, so a casing variant
    # costs neither a redundant query nor a missed pool.
    if [[ "${rc}" -ne 0 ]] || ! printf '%s\n' "${out}" | grep -qi 'VRAM Total Memory (B):'; then
      out=""
      # 124 is `timeout`'s own signal: the call was cut short, which says
      # nothing about whether this rocm-smi accepts the two-pool form. Only a
      # call that actually answered may retire it.
      [[ "${rc}" -eq 124 ]] || combined=0
    fi
  fi
  if [[ -z "${out}" ]]; then
    # A rocm-smi that rejects the two-pool form: fall back to the single-pool
    # call this step has always made.
    out="$(timeout 15 rocm-smi --showmeminfo vram 2>/dev/null)" || {
      sleep 5
      continue
    }
  fi
  total="$(field 'VRAM Total Memory')"
  used="$(field 'VRAM Total Used Memory')"
  gtt_total="$(field 'GTT Total Memory')"
  gtt_used="$(field 'GTT Total Used Memory')"
  if [[ -z "${total}" ]] || [[ -z "${used}" ]]; then
    reason="rocm-smi returned no VRAM figures (no AMD GPU detected)"
    sleep 5
    continue
  fi
  # Print the raw pools once: without them a failure here cannot be told apart
  # from a genuinely occupied GPU by reading this log.
  if [[ "${shown}" -eq 0 ]]; then
    echo "rocm-smi pools (used/total bytes): VRAM ${used}/${total}, GTT ${gtt_used:-n/a}/${gtt_total:-n/a}"
    shown=1
  fi
  # A zero total is a failed reading, not a carveout. Retry instead of
  # reinterpreting it as an APU and passing the gate on whatever GTT reports —
  # a wedged driver must not clear this step.
  if [[ "${total}" -le 0 ]]; then
    reason="rocm-smi reported a VRAM total of 0 (unparseable or wedged reading)"
    sleep 5
    continue
  fi
  pool=VRAM
  if [[ "${total}" -lt "${min_free}" ]]; then
    # An APU has no private VRAM: this total is the fixed BIOS carveout while
    # the engine allocates out of GTT-backed system RAM, so the floor is being
    # applied to a pool the engine never draws on — it fails healthy hosts and
    # stays quiet on starved ones. See `vram_capacity_is_meaningful` in
    # apps/rocm/src/main.rs, which withholds the product's equivalent warning
    # for the same reason.
    #
    # What the floor then means: ${MIN_FREE_GIB} GiB against the whole GTT
    # aperture, so it trips only near exhaustion. That coarseness is why the
    # process assertion below exists; it is not a reason to skip the floor.
    #
    # Discrete cards never reach this branch — their VRAM total is far above
    # the floor, which is also what lets the discrete and APU lanes share this
    # one script: the two-pool query degrades to the single-pool one on a
    # rocm-smi that rejects it, and on a card that answers it the extra GTT
    # rows are simply never consulted.
    if [[ "${gtt_absent}" -eq 0 ]] && { [[ -z "${gtt_total}" ]] || [[ -z "${gtt_used}" ]]; }; then
      # The two-pool form above may have been rejected outright, in which case
      # the fallback query carried no GTT rows. Ask for the pool on its own
      # before concluding the host has none, or a CLI quirk fails a healthy APU
      # with the very report this avoids.
      rc=0
      out="$(timeout 15 rocm-smi --showmeminfo gtt 2>/dev/null)" || rc=$?
      gtt_total="$(field 'GTT Total Memory')"
      gtt_used="$(field 'GTT Total Used Memory')"
      if [[ -n "${gtt_total}" ]] && [[ -n "${gtt_used}" ]]; then
        # The line printed above necessarily read `GTT n/a/n/a` — the combined
        # query is what failed. Report what the pool the gate now decides on
        # actually holds, or the log contradicts the verdict on precisely the
        # hosts this branch exists for.
        echo "rocm-smi GTT pool (separate query): ${gtt_used}/${gtt_total}"
      elif [[ "${rc}" -ne 0 ]]; then
        # A query that failed is not a host without the pool. Keeping them apart
        # stops a flaky call being reported as a hardware fact, and leaves the
        # memo below for the real thing.
        reason="VRAM total is $((total / 1024 / 1024)) MiB, under the ${MIN_FREE_GIB} GiB floor, and the GTT query itself failed (rocm-smi exit ${rc}) — the pool was never read"
        sleep 5
        continue
      else
        gtt_absent=1
      fi
    fi
    if [[ -z "${gtt_total}" ]] || [[ -z "${gtt_used}" ]]; then
      reason="VRAM total is $((total / 1024 / 1024)) MiB, under the ${MIN_FREE_GIB} GiB floor, and rocm-smi reported no GTT pool — the floor cannot be met from VRAM on this host"
      sleep 5
      continue
    fi
    # A zero total is a failed reading on this pool too. Without this the gate
    # reports `0 GiB free … a serve is likely still holding the GPU` — the exact
    # false report this step exists to stop, one pool over from where it was
    # stopped.
    if [[ "${gtt_total}" -le 0 ]]; then
      reason="rocm-smi reported a GTT total of 0 (unparseable or wedged reading)"
      sleep 5
      continue
    fi
    pool=GTT
    total="${gtt_total}"
    used="${gtt_used}"
  fi
  free=$((total - used))
  if [[ "${free}" -ge "${min_free}" ]]; then
    echo "GPU ready: $((free / 1024 / 1024 / 1024)) GiB ${pool} free (>= ${MIN_FREE_GIB} GiB)."
    # Gate 2. Deliberately AFTER the memory floor rather than before it: a card
    # still draining from the reclaim step's kills is the transient the wait
    # above exists to absorb, and asking this first would report the processes
    # that are on their way out as survivors.
    #
    # What that ordering does NOT establish, on these lanes, is that the drain
    # has finished. The floor is 8/16 GiB against pools of ~62 and ~192 GiB, so
    # on a healthy host it is met on the FIRST poll with no wait at all; it only
    # delays anything past ~54/~176 GiB held. On a discrete lane, therefore,
    # clearing the floor says nothing about whether a process the reclaim step
    # SIGKILLed has finished exiting — what bounds that is the reclaim's own
    # TERM grace period and kill loop, not this. The ordering is still the right
    # way round, since asking before the floor can only be worse; it is simply
    # not the proof of a settled card an earlier wording of this comment
    # claimed. A process still dying when this runs is reported as a survivor,
    # which is a false positive this gate can produce (#446).
    #
    # Shelled out to reclaim-gpu.sh rather than matched here: that script owns
    # the E2E_ROOTS / ENGINE_MARKERS rules, documents the over-match exposure,
    # mirrors them into the PowerShell reclaims under an xtask contract test,
    # and self-tests them against decoy processes. A second matcher in CI is how
    # the original `pkill -f` patterns drifted.
    #
    # BOUNDED by `timeout`, like every rocm-smi call above and for a sharper
    # reason. `--assert-clear` walks /proc/[0-9]*/cmdline and READS each entry;
    # a process wedged in uninterruptible sleep (D-state) can block that read,
    # and a wedged GPU process is precisely the condition this gate fires on.
    # The hang risk is CORRELATED with what is being detected, not independent
    # of it — so the one run that most needs a ceiling is the one that would
    # otherwise ride to the job's own cap, which is the failure this whole
    # script exists to replace with a fast, named reason.
    #
    # A TIMEOUT FAILS THE GATE. A scan that could not finish is not evidence
    # the host is clear; passing on it would turn a hang into a silent green on
    # exactly the hosts #446 is about. `timeout` reports expiry as exit 124,
    # which is why the status is captured rather than tested with `if`: the
    # outcomes — clear, survivor named, never finished, and never ran at all —
    # have to be told apart, and reporting any of them as another misdescribes
    # it. "A leaked E2E engine process is still holding it (listed above)" is
    # claimed ONLY for exit 1, the one code that means the scan ran and named
    # somebody. Anything else that is not 0 or 124 — 127 for a missing script,
    # 2 for an argument this script and reclaim-gpu.sh disagree about, a signal
    # — printed that message with nothing listed above it, sending the reader
    # to hunt for a process that was never found.
    assert_rc=0
    timeout 15 bash "${RECLAIM_SCRIPT}" --assert-clear || assert_rc=$?
    if [[ "${assert_rc}" -eq 0 ]]; then
      exit 0
    fi
    if [[ "${assert_rc}" -eq 124 ]]; then
      echo "::error::GPU preflight failed: the leaked-process scan (reclaim-gpu.sh --assert-clear) did not finish within 15s and was timed out — the host could NOT be shown clear. A process stuck in uninterruptible sleep blocks the /proc read this scan makes, which is the state a wedged GPU process is in, so this fails rather than passes."
    elif [[ "${assert_rc}" -eq 1 ]]; then
      echo "::error::GPU preflight failed: the GPU has ${MIN_FREE_GIB}+ GiB free but a leaked E2E engine process is still holding it (listed above). The free-memory floor cannot see this: against this pool it only trips near exhaustion — see #446."
    else
      echo "::error::GPU preflight failed: the leaked-process scan (${RECLAIM_SCRIPT} --assert-clear) could not be run — it exited ${assert_rc}, which is neither clear (0), a named survivor (1) nor the 15s bound (124). A missing or unreadable script reports 127 and an unrecognised flag reports 2. Nothing was shown about this host either way, so this fails rather than passes."
    fi
    # Bounded for the same reason, and on this path more sharply: having just
    # learned the scan can stall, an unbounded diagnostic would spend the
    # ceiling the bound above just saved. The rocm-smi call inside the reclaim
    # is already bounded there; its own /proc walk is what this bounds.
    timeout 15 bash "${RECLAIM_SCRIPT}" --report-holders || true
    exit 1
  fi
  reason="${pool} never dropped below the floor: only $((free / 1024 / 1024 / 1024)) GiB free (< ${MIN_FREE_GIB} GiB) — a serve is likely still holding the GPU"
  echo "waiting: $((free / 1024 / 1024 / 1024)) GiB ${pool} free (< ${MIN_FREE_GIB} GiB)…"
  sleep 5
done
echo "::error::GPU preflight failed after ${CEILING_SECS}s: ${reason}"
# Name what is holding the card instead of leaving the next reader to
# reconstruct it: --report-holders prints per-process VRAM and the engine
# processes visible here. Never fails the step itself, and bounded like the
# gate above: a wedged holder is the likeliest reason this path was reached at
# all, and a diagnostic that hangs costs the fast failure it was printing for.
timeout 15 bash "${RECLAIM_SCRIPT}" --report-holders || true
exit 1
