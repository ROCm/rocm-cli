// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! A stand-in engine child for tests.
//!
//! It is a `/bin/sh` script that records how it was launched (arguments,
//! environment, stdin) into a scratch directory and then runs a caller-supplied
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
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

static NEXT_ENGINE: AtomicU32 = AtomicU32::new(0);

/// `errno` for "text file busy" (`ETXTBSY`): another thread's `fork` briefly
/// held the script's write descriptor open when this one tried to execute it.
const TEXT_FILE_BUSY: i32 = 26;

pub struct FakeEngine {
    dir: PathBuf,
    script: PathBuf,
}

impl FakeEngine {
    /// Write the script. `reads_stdin` makes the fake drain its stdin into
    /// `stdin` first (what a well-behaved engine does); pass `false` for an
    /// engine that exits without reading its request. `body` runs after the
    /// recording and may use `$dir`, the scratch directory.
    pub fn new(label: &str, reads_stdin: bool, body: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "rocm-fake-engine-{label}-{}-{}",
            std::process::id(),
            NEXT_ENGINE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).expect("create the fake engine's scratch directory");
        let script = dir.join("engine.sh");
        let read_stdin = if reads_stdin {
            "cat > \"$dir/stdin\"\n"
        } else {
            ""
        };
        fs::write(
            &script,
            format!(
                "#!/bin/sh\ndir='{}'\nprintf '%s\\n' \"$@\" > \"$dir/argv\"\nenv > \"$dir/env\"\n{read_stdin}{body}\n",
                dir.display()
            ),
        )
        .expect("write the fake engine script");
        fs::set_permissions(&script, fs::Permissions::from_mode(0o755))
            .expect("make the fake engine executable");
        Self { dir, script }
    }

    /// A fake that reads its request and answers with `response` (a complete
    /// response envelope as JSON text), exiting with `status`.
    pub fn answering(label: &str, response: &str, status: i32) -> Self {
        Self::new(
            label,
            true,
            &format!("printf '%s' '{response}'\nexit {status}"),
        )
    }

    pub fn path(&self) -> &Path {
        &self.script
    }

    /// The scratch directory, for a test that wants to point `$dir` somewhere.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The arguments the fake was started with, one per entry.
    pub fn argv(&self) -> Vec<String> {
        fs::read_to_string(self.dir.join("argv"))
            .expect("the fake engine never recorded its arguments")
            .lines()
            .map(str::to_owned)
            .collect()
    }

    /// Everything the fake read from its stdin, verbatim.
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

    /// Run `launch`, retrying while the kernel refuses to execute the script
    /// because a concurrent `fork` in this test process still holds its write
    /// descriptor. A fresh script is executed the instant it is written, and the
    /// test harness runs tests on parallel threads, so the window is real.
    pub fn run<R>(&self, mut launch: impl FnMut() -> anyhow::Result<R>) -> anyhow::Result<R> {
        for attempt in 0..50 {
            match launch() {
                Err(error) if attempt < 49 && is_text_file_busy(&error) => {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
                other => return other,
            }
        }
        unreachable!("the final attempt always returns")
    }
}

fn is_text_file_busy(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .downcast_ref::<io::Error>()
            .is_some_and(|io| io.raw_os_error() == Some(TEXT_FILE_BUSY))
    })
}

impl Drop for FakeEngine {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}
