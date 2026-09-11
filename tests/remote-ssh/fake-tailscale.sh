#!/bin/sh
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# A stand-in for Tailscale on the test remote, covering `serve` and `funnel`.
#
# It keeps its forwards in a file shaped like the real ServeConfig, so publish,
# inspect and withdraw are genuinely stateful: withdrawing has to actually
# remove the entry for a test to see it gone. That is the property worth
# checking here — a publish outlives reboots, so a withdrawal that silently does
# nothing leaves an endpoint exposed with nothing tracking it.
#
# `funnel` exists here only as a way to put the remote into the state the CLI
# must refuse to publish over. The CLI never runs it — it reads `AllowFunnel`
# out of the serve config and bails — but without a fake that can *write* that
# key, the classifier's exposure branch is only ever exercised against
# hand-written fixtures, never against something daemon-shaped.

set -eu

STATE_DIR="${FAKE_ROCM_STATE:-/var/lib/fake-rocm}"
SERVE="${STATE_DIR}/serve.json"
# The real daemon keys AllowFunnel by `host:port` using the node's tailnet DNS
# name. Any plausible name will do — the CLI matches on the port suffix because
# it does not track the host.
FUNNEL_HOST="${FAKE_TAILNET_HOST:-fake-gpu.tail1234.ts.net}"
mkdir -p "${STATE_DIR}"
[ -f "${SERVE}" ] || echo '{}' > "${SERVE}"

command="${1:-}"
if [ "${command}" != "serve" ] && [ "${command}" != "funnel" ]; then
  echo "fake tailscale: unsupported command: ${command}" >&2
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

# `funnel --tcp=N on|off` toggles AllowFunnel and touches nothing else, which is
# what makes it a usable fixture: a port can be Funnel-exposed with no forward
# behind it, or with one, and the CLI has to refuse in both cases.
if [ "${command}" = "funnel" ]; then
  if [ "${off}" -eq 1 ]; then
    jq --arg key "${FUNNEL_HOST}:${port}" \
      'if .AllowFunnel then .AllowFunnel |= del(.[$key]) else . end' \
      "${SERVE}" > "${SERVE}.tmp"
  else
    jq --arg key "${FUNNEL_HOST}:${port}" \
      '.AllowFunnel = ((.AllowFunnel // {}) + {($key): true})' \
      "${SERVE}" > "${SERVE}.tmp"
  fi
  mv "${SERVE}.tmp" "${SERVE}"
  exit 0
fi

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
