# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# shellcheck shell=bash
# Sourced by render.sh (NOT by the tapes themselves) before vhs starts. Starts
# the standalone mock OpenAI server (`rocm-demo-env`, which reuses the e2e test
# harness), puts the binaries on PATH, points `rocm` at an isolated config via env
# vars, and tears everything down when the sourcing shell exits. No GPU or real
# model required — output is deterministic, so the recorded screencast is
# reproducible.
#
# Honours (with defaults):
#   ROCM_DEMO_MODEL   model id shown in the demo
#   ROCM_BIN_DIR      dir holding the `rocm` and `rocm-demo-env` binaries
#
# `source` this (do not execute): it exports env into the current shell and sets
# an EXIT trap that stops the mock server.

: "${ROCM_DEMO_MODEL:=Qwen/Qwen3.5-0.6B}"
_repo="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
: "${ROCM_BIN_DIR:=${_repo}/target/release}"
export PATH="${ROCM_BIN_DIR}:${PATH}"

# A realistic canned reply so the recorded chat reads naturally; the mock server
# echoes this instead of its test-only default (see src/mock_server.rs).
export ROCM_MOCK_CHAT_REPLY="You're running on an AMD Instinct MI300X (gfx942)."

_demo_root="$(mktemp -d)"
# The mock ignores SIGINT/SIGHUP (VHS/ttyd emit some during terminal setup) and
# stops on the SIGTERM the trap below sends — so it survives until the tape runs.
rocm-demo-env --root "${_demo_root}" --model "${ROCM_DEMO_MODEL}" >"${_demo_root}/env.sh" 2>/dev/null &
_demo_pid=$!
# shellcheck disable=SC2064
trap "kill ${_demo_pid} 2>/dev/null; rm -rf '${_demo_root}'" EXIT

# Wait for the server to publish its env + readiness marker before continuing.
for _ in $(seq 1 100); do
  grep -q 'ready on' "${_demo_root}/env.sh" 2>/dev/null && break
  sleep 0.1
done
# shellcheck disable=SC1091
source "${_demo_root}/env.sh"
