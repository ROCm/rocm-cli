// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! A stand-in for `docker` that always fails, for scenarios proving the AIM
//! engine's Docker-unavailable gate.
//!
//! `daemon_reachable()` (engines/aim) and the harness's own Docker probe
//! (`capability::docker_available`) both just run `docker info` and check the
//! exit status, so a `docker` that always exits non-zero deterministically
//! simulates "no reachable daemon" regardless of whether the host running the
//! suite actually has Docker installed.
//!
//! A Rust binary rather than a shell script so the scenarios run on Windows
//! too.

fn main() {
    std::process::exit(1);
}
