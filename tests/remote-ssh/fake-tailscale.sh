#!/bin/sh
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# A stand-in for Tailscale on the test remote, covering only `serve`.
#
# It keeps its forwards in a file shaped like the real ServeConfig, so publish,
# inspect and withdraw are genuinely stateful: withdrawing has to actually
# remove the entry for a test to see it gone. That is the property worth
# checking here — a publish outlives reboots, so a withdrawal that silently does
# nothing leaves an endpoint exposed with nothing tracking it.

set -eu

STATE_DIR="${FAKE_ROCM_STATE:-/var/lib/fake-rocm}"
SERVE="${STATE_DIR}/serve.json"
mkdir -p "${STATE_DIR}"
[ -f "${SERVE}" ] || echo '{}' > "${SERVE}"

if [ "${1:-}" != "serve" ]; then
  echo "fake tailscale: unsupported command: ${1:-}" >&2
  exit 2
fi
shift

if [ "${1:-}" = "status" ]; then
  cat "${SERVE}"
  exit 0
fi

port=""
target=""
off=0
for arg in "$@"; do
  case "${arg}" in
    --bg) ;;
    --tcp=*) port="${arg#--tcp=}" ;;
    tcp://*) target="${arg#tcp://}" ;;
    off) off=1 ;;
    *) ;;
  esac
done

[ -n "${port}" ] || { echo "fake tailscale: no --tcp port given" >&2; exit 2; }

if [ "${off}" -eq 1 ]; then
  jq --arg port "${port}" 'if .TCP then .TCP |= del(.[$port]) else . end' \
    "${SERVE}" > "${SERVE}.tmp"
  mv "${SERVE}.tmp" "${SERVE}"
  exit 0
fi

[ -n "${target}" ] || { echo "fake tailscale: no forward target given" >&2; exit 2; }
# Integer map keys serialize as strings, which is what the real daemon emits and
# what the parser has to match.
jq --arg port "${port}" --arg target "${target}" \
  '.TCP = ((.TCP // {}) + {($port): {TCPForward: $target}})' \
  "${SERVE}" > "${SERVE}.tmp"
mv "${SERVE}.tmp" "${SERVE}"
