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
//! `FAKE_UV_LOG` — a file to append one line per invocation to, holding the
//! arguments it was given (unset: nothing is written). A scenario that must
//! show `uv` was never run asserts the file does not exist.
//!
//! `FAKE_UV_DRAIN_STDIN` — when set, read standard input to its end before
//! answering (unset: stdin is left alone). The CLI writes the requirements to
//! `uv pip compile -` on stdin before it reads anything back, so a stand-in that
//! answers and exits without reading can make that write fail on a closed pipe,
//! depending on timing. Unset by default because a scenario that never pipes
//! anything in may leave stdin attached to something that never ends.
//!
//! A Rust binary rather than a shell script so the scenarios run on Windows
//! too.

fn main() {
    if std::env::var_os("FAKE_UV_DRAIN_STDIN").is_some() {
        use std::io::Read as _;
        let mut sink = Vec::new();
        let _ = std::io::stdin().read_to_end(&mut sink);
    }
    if let Some(log) = std::env::var_os("FAKE_UV_LOG").map(std::path::PathBuf::from) {
        use std::io::Write as _;
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log)
            .unwrap_or_else(|e| panic!("failed to open FAKE_UV_LOG {}: {e}", log.display()));
        writeln!(file, "{}", args.join(" "))
            .unwrap_or_else(|e| panic!("failed to write FAKE_UV_LOG {}: {e}", log.display()));
    }
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
