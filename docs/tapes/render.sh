#!/usr/bin/env bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Render a demo tape with the binaries + mock environment already in place.
#
#   docs/tapes/render.sh docs/tapes/chat.tape
#
# Sourcing lib/demo-env.sh first (not from inside the tape) is deliberate: VHS
# types on a fixed clock and never waits for a command to return, so any setup
# done *inside* a tape races the typing. Doing it here means every tape is a pure
# command sequence that runs against a ready shell.
#
# Env overrides: ROCM_BIN_DIR (default target/release), ROCM_DEMO_MODEL.
set -u

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

tape="${1:?usage: render.sh <tape>}"

# Sets PATH to the binaries, exports ROCM_CLI_* at an isolated config, starts the
# mock server, and installs an EXIT trap that stops it when this script exits.
# shellcheck source=docs/tapes/lib/demo-env.sh
source docs/tapes/lib/demo-env.sh

vhs "$tape"
