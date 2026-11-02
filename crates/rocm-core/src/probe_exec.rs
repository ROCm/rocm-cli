// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Subprocess and file-read helpers for the host probes.
//!
//! [`run`]/[`run_with_env`] run a command under a timeout and never panic, `which`
//! resolves a program on `PATH` without executing it, and [`read_text`]/
//! [`read_host_text`] read a file as a possibly-empty string. They sit apart from
//! [`crate::examine`] so `fix`, `process`, `host_gpu` and `rocm_install` can reach
//! them without importing the module that owns the `Examination` document.

use crate::runtime_is_windows;
use std::ffi::OsString;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// Run a command with a timeout. Returns `(rc, stdout, stderr)`. `rc` is `127`
/// when the program can't be spawned and `124` on timeout.
pub(crate) fn run(program: &str, args: &[&str], timeout: Duration) -> (i32, String, String) {
    run_with_env(program, args, &[], timeout)
}

/// [`run`] with extra environment variables overlaid on the inherited ones.
pub(crate) fn run_with_env(
    program: &str,
    args: &[&str],
    envs: &[(String, OsString)],
    timeout: Duration,
) -> (i32, String, String) {
    let (rc, stdout, stderr) = run_raw(program, args, envs, timeout);
    (
        rc,
        String::from_utf8_lossy(&stdout).into_owned(),
        String::from_utf8_lossy(&stderr).into_owned(),
    )
}

/// [`run`] without the UTF-8 assumption, for output that is not UTF-8.
fn run_raw(
    program: &str,
    args: &[&str],
    envs: &[(String, OsString)],
    timeout: Duration,
) -> (i32, Vec<u8>, Vec<u8>) {
    let Ok(mut child) = Command::new(program)
        .args(args)
        .envs(envs.iter().map(|(key, value)| (key, value)))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    else {
        return (127, Vec::new(), Vec::new());
    };
    // Bytes, then a lossy conversion at the end. `read_to_string` FAILS on
    // invalid UTF-8 and the error was discarded, so a single stray byte silently
    // emptied the whole capture — which reads downstream as "the command printed
    // nothing", not as "the output could not be decoded".
    let stdout_handle = child.stdout.take().map(|mut stdout| {
        thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stdout.read_to_end(&mut buf);
            buf
        })
    });
    let stderr_handle = child.stderr.take().map(|mut stderr| {
        thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stderr.read_to_end(&mut buf);
            buf
        })
    });
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    break None;
                }
                thread::sleep(Duration::from_millis(20));
            }
            Err(_) => break None,
        }
    };
    let stdout: Vec<u8> = stdout_handle
        .and_then(|handle| handle.join().ok())
        .unwrap_or_default();
    let stderr: Vec<u8> = stderr_handle
        .and_then(|handle| handle.join().ok())
        .unwrap_or_default();
    let rc = match status {
        Some(status) => status.code().unwrap_or(-1),
        None => 124,
    };
    (rc, stdout, stderr)
}

/// Run a command whose output is UTF-16LE, as `wsl.exe`'s is.
///
/// Decoding is by declaration, not detection. Sniffing the encoding cannot work
/// here: UTF-16LE text in a Latin or Cyrillic script is made entirely of bytes
/// below 0x80, so it is *valid UTF-8* and decodes without error straight into
/// mojibake — no NUL-density or validity test can tell the two apart. The one
/// reliable fact is which program produced the bytes.
pub(crate) fn run_utf16le(program: &str, args: &[&str], timeout: Duration) -> (i32, String) {
    let (rc, stdout, _) = run_raw(program, args, &[], timeout);
    (rc, decode_utf16le(&stdout))
}

pub(crate) fn decode_utf16le(bytes: &[u8]) -> String {
    let body = bytes.strip_prefix(&[0xFF, 0xFE]).unwrap_or(bytes);
    // `chunks_exact` drops a trailing odd byte rather than panicking on it.
    let units: Vec<u16> = body
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect();
    String::from_utf16_lossy(&units)
}

pub(crate) fn read_text(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_default()
}

/// [`read_text`] for an absolute host path, read where [`crate::host_path`]
/// says the host is.
pub(crate) fn read_host_text(path: &str) -> String {
    std::fs::read_to_string(crate::host_path(path)).unwrap_or_default()
}

/// Whether `program` resolves on `PATH` (best-effort, no execution).
pub(crate) fn which(program: &str) -> bool {
    let Ok(path) = std::env::var("PATH") else {
        return false;
    };
    let (sep, exts): (char, &[&str]) = if runtime_is_windows() {
        (';', &[".exe", ".bat", ".cmd", ""])
    } else {
        (':', &[""])
    };
    for dir in path.split(sep) {
        if dir.is_empty() {
            continue;
        }
        for ext in exts {
            if Path::new(dir).join(format!("{program}{ext}")).is_file() {
                return true;
            }
        }
    }
    false
}

pub(crate) const SHORT: Duration = Duration::from_secs(5);
pub(crate) const MEDIUM: Duration = Duration::from_secs(8);

/// Truncate `value` to at most `max_chars` characters, appending a marker when
/// truncated. Slices on char boundaries (matching Python's `value[:n]`); a byte
/// slice would panic when the cut lands inside a multibyte character.
///
/// `pub(crate)` because [`crate::process`] bounds its timeout quote with the
/// same marker, and a second copy here would be free to drift from it.
/// (`host_gpu/diagnostics.rs`'s `truncate_diagnostic_line` is a different function with a
/// different marker — plain `...` — not a third copy of this one.)
pub(crate) fn truncate_to_chars(value: String, max_chars: usize) -> String {
    if value.chars().count() > max_chars {
        let truncated: String = value.chars().take(max_chars).collect();
        format!("{truncated}...[truncated]")
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_output_that_is_not_utf8_is_kept_rather_than_dropped() {
        // `read_to_string` fails on invalid UTF-8, and the error was discarded --
        // so one stray byte emptied a whole capture, which every caller then read
        // as "the command printed nothing".
        let (rc, out, _) = run("printf", &["ok\\xffdone"], SHORT);
        if rc == 127 {
            return; // no `printf` binary on this host
        }
        assert!(
            out.starts_with("ok") && out.ends_with("done"),
            "the undecodable byte must not take the rest of the output with it: {out:?}"
        );
    }

    #[test]
    fn env_truncation_is_char_safe_across_multibyte_boundary() {
        // 5000 'é' chars = 10000 bytes; byte 4000 lands inside a char, which a
        // byte slice would panic on. Must truncate cleanly to 4000 chars.
        let value = "é".repeat(5000);
        let out = truncate_to_chars(value, 4000);
        assert!(out.ends_with("...[truncated]"));
        assert_eq!(out.chars().filter(|&c| c == 'é').count(), 4000);
    }

    #[test]
    fn env_truncation_leaves_short_values_untouched() {
        assert_eq!(truncate_to_chars("short".to_owned(), 4000), "short");
    }
}
