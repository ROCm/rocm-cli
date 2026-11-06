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
//! When `FAKE_DOCKER_LOG` names a file, every invocation's arguments are
//! appended to it first (one line per call), so a scenario can prove a refused
//! `rocm serve` never reached Docker, not even for `docker info`.
//!
//! A Rust binary rather than a shell script so the scenarios run on Windows
//! too.

fn main() {
    if let Some(log) = std::env::var_os("FAKE_DOCKER_LOG") {
        use std::io::Write as _;
        let args: Vec<String> = std::env::args().skip(1).collect();
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log)
        {
            let _ = writeln!(file, "{}", args.join(" "));
        }
    }
    std::process::exit(1);
}
