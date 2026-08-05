// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Regression test: a downstream reader closing the pipe early must not make
//! `rocm` panic. Rust installs `SIG_IGN` for SIGPIPE at startup, so without the
//! `reset_sigpipe()` in `main` the next `println!` after the pipe closes panics
//! with `failed printing to stdout: Broken pipe` and exit code 101 (found via
//! `rocm fix fix-2-unset-override --dry-run | head`). With SIGPIPE reset to
//! `SIG_DFL` the process instead terminates on the signal (exit 128+13 = 141),
//! the conventional behaviour for a Unix CLI, and never prints a panic.

#![cfg(unix)]

use std::io::Read;
use std::process::{Command, Stdio};

/// Spawn a command whose output far exceeds the OS pipe buffer (`completions
/// bash` is ~120 KB, vs a 64 KB pipe), read a single byte, then drop the read
/// end so the child's next write hits the closed pipe while it is still
/// producing output. A small-output command (e.g. `fix`, ~1.5 KB) would fit
/// entirely in the pipe buffer and let the child finish before the close,
/// never exercising the broken-pipe path. The SIGPIPE reset lives in `main`,
/// so this covers every subcommand, not just the one that first surfaced it.
#[test]
fn early_pipe_close_does_not_panic() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rocm"))
        .args(["completions", "bash"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .spawn()
        .expect("spawn rocm completions bash");

    // Read a single byte, then drop stdout to close the read end of the pipe.
    let mut stdout = child.stdout.take().expect("capture stdout");
    let mut one = [0u8; 1];
    let _ = stdout.read(&mut one);
    drop(stdout);

    let mut stderr = child.stderr.take().expect("capture stderr");
    let mut err = String::new();
    let _ = stderr.read_to_string(&mut err);

    let status = child.wait().expect("wait for rocm completions bash");

    assert!(
        !err.contains("panicked") && !err.contains("Broken pipe"),
        "rocm panicked on a closed stdout pipe instead of exiting on SIGPIPE; stderr:\n{err}"
    );
    assert_ne!(
        status.code(),
        Some(101),
        "rocm exited 101 (panic) on a closed stdout pipe; expected clean SIGPIPE termination"
    );
}
