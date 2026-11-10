// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Optional-command capture for the host probes.
//!
//! Runs a tool that may not be installed under a short timeout and keeps its
//! stdout only on success, trying each spelling of the program that
//! [`crate::probe_exec::tool_path_candidates`] gives (`PATHEXT` and the fixed
//! Windows system locations included). A missing, slow or failing tool reads as
//! `None`, never as an error.

use crate::probe_exec::tool_path_candidates;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

pub(super) const OPTIONAL_COMMAND_TIMEOUT: Duration = Duration::from_millis(1_500);

pub(super) fn capture_optional_command(program: &str, args: &[&str]) -> Option<String> {
    capture_optional_command_with_timeout(program, args, OPTIONAL_COMMAND_TIMEOUT)
}

pub(super) fn capture_optional_command_with_timeout(
    program: &str,
    args: &[&str],
    timeout: Duration,
) -> Option<String> {
    for candidate in tool_path_candidates(program) {
        if let Some(output) =
            capture_optional_command_candidate_with_timeout(Path::new(&candidate), args, timeout)
        {
            return Some(output);
        }
    }
    None
}

fn capture_optional_command_candidate_with_timeout(
    program: &Path,
    args: &[&str],
    timeout: Duration,
) -> Option<String> {
    let mut child = match Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            debug_command_capture_failure(program, "spawn", &error.to_string());
            return None;
        }
    };
    let mut stdout_reader = child.stdout.take().map(|mut stdout| {
        thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = stdout.read_to_end(&mut bytes);
            bytes
        })
    });

    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let bytes = stdout_reader
                    .take()
                    .map(|reader| reader.join().unwrap_or_default())
                    .unwrap_or_default();
                if status.success() {
                    return String::from_utf8(bytes).ok();
                }
                debug_command_capture_failure(program, "exit", &format!("status {status}"));
                return None;
            }
            Ok(None) if start.elapsed() < timeout => {
                thread::sleep(Duration::from_millis(25));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                if let Some(reader) = stdout_reader.take() {
                    let _ = reader.join();
                }
                debug_command_capture_failure(program, "timeout", "timed out");
                return None;
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                if let Some(reader) = stdout_reader.take() {
                    let _ = reader.join();
                }
                debug_command_capture_failure(program, "wait", "failed to wait");
                return None;
            }
        }
    }
}

fn debug_command_capture_failure(program: &Path, stage: &str, detail: &str) {
    if !crate::runtime::env_flag("ROCM_CLI_DEBUG_COMMAND_CAPTURE") {
        return;
    }
    eprintln!(
        "rocm debug: command capture {stage} failed for {}: {detail}",
        program.display()
    );
}
