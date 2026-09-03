#!/bin/sh
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# A stand-in for Tailscale on the *calling* machine, for the end-to-end harness.
#
# `rocm remote` shells out locally to discover peers and to check one is online
# before dialling it. This reports a single peer — the container the harness
# started — as an online, GPU-tagged machine whose address is loopback, which is
# where that container's SSH port is published.
#
# It answers `status` only. The remote half of the tailnet story (`serve`) is a
# different stub that runs inside the container.

set -eu

if [ "${1:-}" = "status" ]; then
  cat "${FAKE_LOCAL_TAILSCALE_STATUS:?FAKE_LOCAL_TAILSCALE_STATUS must point at a status document}"
  exit 0
fi

echo "fake local tailscale: unsupported command: ${1:-}" >&2
exit 2
