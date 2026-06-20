// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//! Integration test proving `rocm daemon` runs the real foreground automation
//! loop (embedded `rocmd`) instead of only printing the status panel.
//!
//! The real loop prints the `rocmd run` banner and then blocks forever, so we
//! spawn the built binary, read stdout until the banner appears (or a short
//! timeout elapses), then kill the process.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// Resolve a writable temp directory unique to this test run.
fn temp_state_dir() -> PathBuf {
    let mut dir = std::env::temp_dir();
    let unique = format!(
        "rocm-daemon-run-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos()
    );
    dir.push(unique);
    std::fs::create_dir_all(&dir).expect("create temp state dir");
    dir
}

#[test]
fn daemon_runs_real_foreground_loop() {
    let state_dir = temp_state_dir();
    let config_dir = state_dir.join("config");
    let data_dir = state_dir.join("data");
    let cache_dir = state_dir.join("cache");

    let mut child = Command::new(env!("CARGO_BIN_EXE_rocm"))
        .arg("daemon")
        // Point AppPaths::discover at the temp dirs so the daemon never touches
        // the real user environment.
        .env("ROCM_CLI_CONFIG_DIR", &config_dir)
        .env("ROCM_CLI_DATA_DIR", &data_dir)
        .env("ROCM_CLI_CACHE_DIR", &cache_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .stdin(Stdio::null())
        .spawn()
        .expect("spawn rocm daemon");

    let stdout = child.stdout.take().expect("capture daemon stdout");
    let (tx, rx) = mpsc::channel::<bool>();
    let reader = thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line) {
                Ok(0) => break, // EOF: process exited.
                Ok(_) => {
                    if line.trim_end() == "rocmd run" {
                        let _ = tx.send(true);
                        return;
                    }
                }
                Err(_) => break,
            }
        }
        let _ = tx.send(false);
    });

    let saw_banner = rx.recv_timeout(Duration::from_secs(20)).unwrap_or(false);

    // The foreground loop blocks; terminate it regardless of outcome.
    let _ = child.kill();
    let _ = child.wait();
    let _ = reader.join();
    let _ = std::fs::remove_dir_all(&state_dir);

    assert!(
        saw_banner,
        "expected `rocm daemon` to print the rocmd run banner (real foreground loop), \
         but it did not appear before timeout"
    );
}
