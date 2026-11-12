// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! A stand-in engine child for tests.
//!
//! It is a `/bin/sh` script that records how it was launched (arguments,
//! environment, stdin, and every request it was sent over all its runs) into a
//! scratch directory and then runs a caller-supplied
//! body, so a test can assert what an engine IPC client put on the child's
//! command line, environment and stdin, and shape what comes back (a response
//! envelope, a crash, stderr output).
//!
//! Test-only: compiled for this crate's own tests and for the `rocm` and
//! `rocmd` test builds through the `test-support` feature, so the two engine
//! IPC callers are exercised against the same fake rather than each growing a
//! private one. Unix-only, because the fake is a shell script; the pure parts
//! of the client are tested on every platform without it.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

static NEXT_ENGINE: AtomicU32 = AtomicU32::new(0);

/// `value` as a single-quoted POSIX shell word. Inside single quotes nothing is
/// special except the quote itself, which cannot be escaped there: close the
/// literal, emit an escaped quote, and reopen it (`'\''`). An apostrophe in a
/// response (`can't stop`) or in `TMPDIR` would otherwise end the literal early.
fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', r"'\''"))
}

pub struct FakeEngine {
    dir: PathBuf,
    script: PathBuf,
}

impl FakeEngine {
    /// Write the script. `reads_stdin` makes the fake drain its stdin into
    /// `stdin` first (what a well-behaved engine does); pass `false` for an
    /// engine that exits without reading its request. `body` runs after the
    /// recording and may use `$dir`, the scratch directory.
    ///
    /// The script is written through [`rocm_core::fake_executable`], so running
    /// it straight away cannot fail with `ETXTBSY` while other tests on parallel
    /// threads spawn children of their own.
    pub fn new(label: &str, reads_stdin: bool, body: &str) -> Self {
        Self::new_in(&std::env::temp_dir(), label, reads_stdin, body)
    }

    /// [`FakeEngine::new`] with its scratch directory created under `base`
    /// instead of the system temporary directory.
    pub fn new_in(base: &Path, label: &str, reads_stdin: bool, body: &str) -> Self {
        let dir = base.join(format!(
            "rocm-fake-engine-{label}-{}-{}",
            std::process::id(),
            NEXT_ENGINE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).expect("create the fake engine's scratch directory");
        let script = dir.join("engine.sh");
        let read_stdin = if reads_stdin {
            "cat > \"$dir/stdin\"\ncat \"$dir/stdin\" >> \"$dir/requests\"\n"
        } else {
            ""
        };
        rocm_core::fake_executable::write_fake_executable(
            &script,
            format!(
                "#!/bin/sh\ndir={}\nprintf '%s\\n' \"$@\" > \"$dir/argv\"\nenv > \"$dir/env\"\n{read_stdin}{body}\n",
                shell_quote(&dir.to_string_lossy())
            ),
        )
        .expect("write the fake engine script");
        Self { dir, script }
    }

    /// A fake that reads its request and answers with `response` (a complete
    /// response envelope as JSON text), exiting with `status`.
    pub fn answering(label: &str, response: &str, status: i32) -> Self {
        Self::new(
            label,
            true,
            &format!("printf '%s' {}\nexit {status}", shell_quote(response)),
        )
    }

    pub fn path(&self) -> &Path {
        &self.script
    }

    /// The arguments the fake was started with, one per entry.
    pub fn argv(&self) -> Vec<String> {
        fs::read_to_string(self.dir.join("argv"))
            .expect("the fake engine never recorded its arguments")
            .lines()
            .map(str::to_owned)
            .collect()
    }

    /// Every request the fake read, one per time it was run, oldest first. A
    /// fake that was never run (or never read its stdin) has an empty list.
    pub fn requests(&self) -> Vec<serde_json::Value> {
        fs::read_to_string(self.dir.join("requests"))
            .unwrap_or_default()
            .lines()
            .map(|line| {
                serde_json::from_str(line)
                    .unwrap_or_else(|error| panic!("a request is not JSON ({error}): {line:?}"))
            })
            .collect()
    }

    /// The `method` of every request the fake read, oldest first.
    pub fn methods(&self) -> Vec<String> {
        self.requests()
            .iter()
            .map(|request| request["method"].as_str().unwrap_or_default().to_owned())
            .collect()
    }

    /// Everything the fake read from its stdin the last time it was run,
    /// verbatim.
    pub fn stdin(&self) -> String {
        fs::read_to_string(self.dir.join("stdin")).expect("the fake engine never read its stdin")
    }

    /// The value of `key` in the fake's environment, if it was set.
    pub fn env(&self, key: &str) -> Option<String> {
        let prefix = format!("{key}=");
        fs::read_to_string(self.dir.join("env"))
            .expect("the fake engine never recorded its environment")
            .lines()
            .find_map(|line| line.strip_prefix(&prefix).map(str::to_owned))
    }
}

impl Drop for FakeEngine {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use std::process::{Command, Stdio};

    /// Run `engine` the way a client does: request on stdin, answer on stdout.
    fn run(engine: &FakeEngine) -> (std::process::ExitStatus, String, String) {
        let mut child = Command::new(engine.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(b"request\n").unwrap();
        let output = child.wait_with_output().unwrap();
        (
            output.status,
            String::from_utf8_lossy(&output.stdout).into_owned(),
            String::from_utf8_lossy(&output.stderr).into_owned(),
        )
    }

    #[test]
    fn shell_quote_survives_each_character_a_literal_can_hold() {
        assert_eq!(shell_quote("plain"), "'plain'");
        assert_eq!(shell_quote("can't"), r"'can'\''t'");
        assert_eq!(shell_quote("''"), r"''\'''\'''");
        assert_eq!(shell_quote(""), "''");
    }

    #[test]
    fn a_response_with_an_apostrophe_is_printed_verbatim() {
        let response = r#"{"ok":false,"error":{"code":"x","message":"can't stop"}}"#;
        let engine = FakeEngine::answering("fake-apostrophe-response", response, 0);
        let (status, stdout, stderr) = run(&engine);
        assert!(status.success(), "{status:?}: {stderr}");
        assert_eq!(stdout, response);
        assert_eq!(stderr, "", "the script must parse");
    }

    #[test]
    fn a_scratch_directory_with_an_apostrophe_still_records_the_launch() {
        let base =
            std::env::temp_dir().join(format!("rocm-fake-engine-it's-{}", std::process::id()));
        let engine = FakeEngine::new_in(&base, "fake-apostrophe-dir", true, "exit 0");
        let (status, _, stderr) = run(&engine);
        assert!(status.success(), "{status:?}: {stderr}");
        assert_eq!(stderr, "", "the script must parse");
        assert_eq!(engine.stdin(), "request\n");
        drop(engine);
        let _ = fs::remove_dir_all(&base);
    }
}
