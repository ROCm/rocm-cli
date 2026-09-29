#!/bin/sh
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# A stand-in for the ROCm CLI on the test remote. It answers in the shapes the
# real one does for the handful of commands the control channel issues, and
# keeps its service registry in a file so a `serve` is visible to a later
# `services list`.
#
# It also records the API key it was handed on stdin, so a test can prove the
# credential arrived by that route and not through the command line.

set -eu

STATE_DIR="${FAKE_ROCM_STATE:-/var/lib/fake-rocm}"
SERVICES="${STATE_DIR}/services.json"
KEY_RECORD="${STATE_DIR}/last-api-key"
ARGV_RECORD="${STATE_DIR}/last-argv"

mkdir -p "${STATE_DIR}"
printf '%s\n' "$*" > "${ARGV_RECORD}"

case "${1:-}" in
  --version)
    echo "rocm 0.0.0-fake"
    ;;

  serve)
    shift
    port=11434
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --port) port="$2"; shift 2 ;;
        *) shift ;;
      esac
    done
    # The key reaches us as an exported environment variable, put there by the
    # remote shell reading it off stdin. Recording it is how a test tells that
    # apart from a key pasted into the command line.
    printf '%s' "${ROCM_SERVE_API_KEY:-}" > "${KEY_RECORD}"

    service_id="svc-fake-${port}"
    jq --arg id "${service_id}" --argjson port "${port}" \
       --argjson now "$(date +%s)000" '
      . + [{
        service_id: $id, engine: "fake", model_ref: "m", canonical_model_id: "m",
        host: "127.0.0.1", port: $port,
        endpoint_url: ("http://127.0.0.1:" + ($port|tostring) + "/v1"),
        mode: "managed", status: "ready", supervisor_pid: 1,
        manifest_path: "/tmp/m", log_path: "/tmp/l", engine_state_path: "/tmp/e",
        created_at_unix_ms: $now
      }]' "${SERVICES}" > "${SERVICES}.tmp"
    mv "${SERVICES}.tmp" "${SERVICES}"
    echo "started ${service_id}"
    ;;

  services)
    case "${2:-}" in
      list)
        # Only the JSON form is exercised; that is the machine-readable contract
        # the remote orchestration actually reads.
        cat "${SERVICES}"
        ;;
      stop)
        service_id="${3:-}"
        jq --arg id "${service_id}" 'map(select(.service_id != $id))' \
          "${SERVICES}" > "${SERVICES}.tmp"
        mv "${SERVICES}.tmp" "${SERVICES}"
        echo "stopped ${service_id}"
        ;;
      *)
        echo "fake rocm: unsupported services subcommand: ${2:-}" >&2
        exit 2
        ;;
    esac
    ;;

  examine)
    # A complete examination, not a plausible-looking fragment. The scoring runs
    # on the calling machine and deserializes this into a fixed struct with no
    # optional fields, so a partial document is not "close enough" — it fails to
    # parse, which is exactly the mistake an earlier version of this stub made.
    cat "${STATE_DIR}/examination.json"
    ;;

  *)
    echo "fake rocm: unsupported command: ${1:-}" >&2
    exit 2
    ;;
esac
