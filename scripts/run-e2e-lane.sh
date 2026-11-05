#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# One lane of the nightly on the shared GPU queue: the full E2E suite, nightly
# scenarios included, on the one GPU the lane was given.
#
# It mirrors the self-hosted nightly job (`e2e-gpu-nightly` in nightly.yml) from a
# cold start: the lane is a fresh container holding a checkout of the commit
# under test, with no toolchain, build or runtime cached from an earlier run. The
# queue side runs this from the checkout's root. What a lane must and must not
# write is in docs/nightly-on-the-queue.md.
#
# Inputs, all optional:
#   E2E_CHANNEL    the ROCm channel to pre-warm, release (default) or nightly
#   E2E_LANE_WORK  where the build and runtimes go (default /var/tmp/e2e-lane)
#   LANE_OUT       where reported files go (default /out)

set -euo pipefail

cd "$(dirname "$0")/.."
out="${LANE_OUT:-/out}"
work="${E2E_LANE_WORK:-/var/tmp/e2e-lane}"
channel="${E2E_CHANNEL:-release}"

export E2E_INCLUDE_NIGHTLY="${E2E_INCLUDE_NIGHTLY-1}"
export E2E_SERVE_TIMEOUT_SECS="${E2E_SERVE_TIMEOUT_SECS:-300}"

# On every way out, so a failing suite still reports which scenarios failed.
# The lane's status stays the suite's: a failed copy only warns.
save_results() {
  local rc=$?
  if [ -d tests/e2e-cucumber/results ]; then
    mkdir -p "$out/results" &&
      cp -R tests/e2e-cucumber/results/. "$out/results/" ||
      echo "warning: could not copy the E2E results to $out" >&2
  fi
  exit "$rc"
}
trap save_results EXIT

# One tree on one filesystem: uv only hardlinks out of its cache into a runtime
# on the same mount, and copies every file otherwise (see e2e-selfhosted.yml).
mkdir -p "$work"
export CARGO_TARGET_DIR="$work/target"
export E2E_SHARED_CACHE_DIR="$work/shared"
export E2E_SHARED_UV_CACHE_DIR="$work/uv-cache"
prewarm="$work/prewarm"
export E2E_SHARED_RUNTIMES_DIR="$prewarm/data/runtimes"

# What a cold Ubuntu build needs, as the WSL lane in e2e-selfhosted.yml
# installs it; the ROCm dev images carry a compiler but not these.
if ! command -v pkg-config >/dev/null || [ ! -e /usr/include/sys/capability.h ]; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    pkg-config build-essential libcap-dev >/dev/null
fi

if ! command -v cargo >/dev/null; then
  # No default toolchain: rust-toolchain.toml pins it, and the first cargo
  # call in this checkout installs that one.
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
    sh -s -- -y --profile minimal --default-toolchain none
  # shellcheck source=/dev/null
  . "$HOME/.cargo/env"
fi

bash .github/scripts/gpu-preflight.sh

# The e2e-test-hooks feature must match what `cargo xtask e2e` builds for
# itself; see the e2e-gpu job in e2e-selfhosted.yml.
cargo build --release -p rocm -p rocmd --features rocm/e2e-test-hooks
export ROCM_CLI_BINARY="$CARGO_TARGET_DIR/release/rocm"
export ROCM_CLI_ROCMD_BINARY="$CARGO_TARGET_DIR/release/rocmd"

# Installed in place, never moved: install sdk bakes absolute paths into the
# runtime manifest. See `xtask e2e-prewarm`.
HF_HOME="$E2E_SHARED_CACHE_DIR/huggingface" \
  UV_CACHE_DIR="$E2E_SHARED_UV_CACHE_DIR" \
  cargo xtask e2e-prewarm --channel "$channel" --prewarm-dir "$prewarm"

cargo xtask e2e
