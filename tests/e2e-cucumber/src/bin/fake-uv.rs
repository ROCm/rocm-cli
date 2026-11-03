// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! A stand-in for `uv`, for the ComfyUI dependency-install scenarios.
//!
//! `rocm comfyui install` shells out to `uv` to resolve ComfyUI's Python
//! dependencies. A scenario that has to be deterministic and offline plants
//! this instead and controls its outcome through two env vars:
//!
//! `FAKE_UV_STDOUT` — a line to print to stdout before exiting (unset: none).
//! `FAKE_UV_EXIT_CODE` — the exit code to return (unset: `0`).
//!
//! A Rust binary rather than a shell script so the scenarios run on Windows
//! too.

fn main() {
    if let Ok(line) = std::env::var("FAKE_UV_STDOUT") {
        println!("{line}");
    }
    let code: i32 = match std::env::var("FAKE_UV_EXIT_CODE") {
        Ok(value) => value.parse().unwrap_or_else(|e| {
            panic!("FAKE_UV_EXIT_CODE={value:?} is not a valid exit code: {e}")
        }),
        Err(_) => 0,
    };
    std::process::exit(code);
}
