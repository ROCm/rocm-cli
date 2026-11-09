#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# One lane of the nightly on the shared GPU queue: the full E2E suite, nightly
# scenarios included, on the one GPU the lane was given, once for each ROCm
# channel.
#
# It mirrors the self-hosted nightly jobs (`e2e-gpu-nightly` in nightly.yml, a
# matrix over the release and nightly channels) from a cold start: the lane is a
# fresh container holding a checkout of the commit under test, with no
# toolchain, build or runtime cached from an earlier run. The build is shared;
# each channel gets its own runtimes, its own suite run, and its own directory
# under the lane's out, `<channel>/results`. Both channels always run, so one
# failing does not hide the other; the lane fails if either did. The queue side
# runs this from the checkout's root. What a lane must and must not write is in
# docs/nightly-on-the-queue.md.
#
# Inputs, all optional:
#   E2E_CHANNELS   the ROCm channels to run, in order: release, nightly, or
#                  both (default "release nightly")
#   E2E_LANE_WORK  where the build and runtimes go (default /var/tmp/e2e-lane)
#   LANE_OUT       where reported files go (default /out)

set -euo pipefail

cd "$(dirname "$0")/.."
out="${LANE_OUT:-/out}"
work="${E2E_LANE_WORK:-/var/tmp/e2e-lane}"
channels="${E2E_CHANNELS:-release nightly}"

export E2E_INCLUDE_NIGHTLY="${E2E_INCLUDE_NIGHTLY-1}"
export E2E_SERVE_TIMEOUT_SECS="${E2E_SERVE_TIMEOUT_SECS:-300}"

# The channels are checked before anything is built, so a typo costs nothing.
for channel in ${channels}; do
  case "${channel}" in
    release | nightly) ;;
    *)
      echo "E2E_CHANNELS: unknown channel '${channel}' (release or nightly)" >&2
      exit 2
      ;;
  esac
done
# Words, not characters: a tab or a newline alone is no channel either.
# shellcheck disable=SC2086
set -- ${channels}
if [[ "$#" -eq 0 ]]; then
  echo "E2E_CHANNELS names no channel" >&2
  exit 2
fi

# After each channel's run, so a failing suite still reports which scenarios
# failed. The lane's status stays the suite's: a failed copy only warns.
save_results() {
  local channel="$1"
  if [[ -d tests/e2e-cucumber/results ]]; then
    mkdir -p "${out}/${channel}/results" &&
      cp -R tests/e2e-cucumber/results/. "${out}/${channel}/results/" ||
      echo "warning: could not copy the E2E results to ${out}/${channel}" >&2
  fi
}

# One tree on one filesystem: uv only hardlinks out of its cache into a runtime
# on the same mount, and copies every file otherwise (see e2e-selfhosted.yml).
mkdir -p "${work}"
export CARGO_TARGET_DIR="${work}/target"
export E2E_SHARED_CACHE_DIR="${work}/shared"
export E2E_SHARED_UV_CACHE_DIR="${work}/uv-cache"
# What a cold Ubuntu build needs, as the WSL lane in e2e-selfhosted.yml
# installs it; the ROCm dev images carry a compiler but not these. The update is
# retried as there: a mirror mid-sync fails it for a few minutes.
if ! command -v pkg-config >/dev/null || [[ ! -e /usr/include/sys/capability.h ]]; then
  for attempt in 1 2 3 4; do
    if apt-get update -qq; then
      break
    fi
    if [[ "${attempt}" -eq 4 ]]; then
      echo "apt-get update failed after ${attempt} attempts" >&2
      exit 1
    fi
    echo "apt-get update failed (attempt ${attempt}); retrying in 60s" >&2
    sleep 60
  done
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    pkg-config build-essential libcap-dev >/dev/null
fi

if ! command -v cargo >/dev/null; then
  # A pinned rustup-init, checked against the SHA-256 recorded here, not a
  # script piped into a shell. To move to a newer rustup, replace the version
  # and both sums with those published beside its rustup-init.
  rustup_version="1.29.3"
  case "$(uname -m)" in
    x86_64)
      rustup_triple="x86_64-unknown-linux-gnu"
      rustup_sha256="b1c8cb14e30889e7e2999e6b69e62d9e9b9ffc0f4df8180564d306d9680e6c55"
      ;;
    aarch64 | arm64)
      rustup_triple="aarch64-unknown-linux-gnu"
      rustup_sha256="9f608c142e0e38a78957d0661ba1dfaf0b6fb37184cfaa4b9a0aa57af6184282"
      ;;
    *)
      echo "no pinned rustup-init for $(uname -m)" >&2
      exit 1
      ;;
  esac
  rustup_init="${work}/rustup-init"
  curl --proto '=https' --tlsv1.2 -sSf -o "${rustup_init}" \
    "https://static.rust-lang.org/rustup/archive/${rustup_version}/${rustup_triple}/rustup-init"
  if ! echo "${rustup_sha256}  ${rustup_init}" | sha256sum -c --quiet -; then
    echo "rustup-init ${rustup_version} does not match its recorded SHA-256" >&2
    exit 1
  fi
  chmod +x "${rustup_init}"
  # No default toolchain: rust-toolchain.toml pins it, and the first cargo
  # call in this checkout installs that one.
  "${rustup_init}" -y --profile minimal --default-toolchain none
  # shellcheck source=/dev/null
  . "${HOME}/.cargo/env"
fi

bash .github/scripts/gpu-preflight.sh

# The e2e-test-hooks feature must match what `cargo xtask e2e` builds for
# itself; see the e2e-gpu job in e2e-selfhosted.yml.
cargo build --release -p rocm -p rocmd --features rocm/e2e-test-hooks
export ROCM_CLI_BINARY="${CARGO_TARGET_DIR}/release/rocm"
export ROCM_CLI_ROCMD_BINARY="${CARGO_TARGET_DIR}/release/rocmd"

# One channel: its runtimes, installed in place, never moved (install sdk bakes
# absolute paths into the runtime manifest; see `xtask e2e-prewarm`), in a tree
# of its own so the two channels' runtimes never meet; then the full suite on
# the build. A failure is returned, not fatal, so the next channel still runs.
run_channel() {
  local channel="$1"
  local prewarm="${work}/prewarm-${channel}"
  export E2E_SHARED_RUNTIMES_DIR="${prewarm}/data/runtimes"
  # The suite writes here, and a channel must not report the last one's files.
  rm -rf tests/e2e-cucumber/results
  HF_HOME="${E2E_SHARED_CACHE_DIR}/huggingface" \
    UV_CACHE_DIR="${E2E_SHARED_UV_CACHE_DIR}" \
    cargo xtask e2e-prewarm --channel "${channel}" --prewarm-dir "${prewarm}" ||
    return $?
  cargo xtask e2e
}

status=0
for channel in ${channels}; do
  echo "== channel ${channel}"
  rc=0
  run_channel "${channel}" || rc=$?
  save_results "${channel}"
  if [[ "${rc}" -ne 0 ]]; then
    echo "channel ${channel} failed (status ${rc})" >&2
    [[ "${status}" -ne 0 ]] || status="${rc}"
  fi
done
exit "${status}"
