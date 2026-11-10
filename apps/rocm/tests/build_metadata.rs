// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

// The included build-script module exposes its internal path calculation and
// `format_cli_version` (tag, branch, and unknown cases) to this integration
// test; its build entry point is intentionally unused here.
#[allow(dead_code)]
#[path = "../build.rs"]
mod build_script;
