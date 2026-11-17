// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The shell script a scenario plants as a stand-in `uname`.
//!
//! `-v` answers a fixed string, so a probe that also asks for the kernel version
//! has nothing to wait for; `-r` runs a fragment the scenario chooses; and every
//! call appends its arguments to a log, so a scenario can count how often the
//! release was asked for. The log lives under the scenario's isolated root, whose
//! path the harness does not control, so it is quoted for the shell rather than
//! put between bare single quotes.

use std::path::Path;

/// `value` as one single-quoted shell word: each embedded single quote closes the
/// quotes, is written escaped and reopens them. (`rocm_core::quoting` has the same
/// quoting, but `rocm-core` is only a dev-dependency of this library.)
fn single_quoted(value: &str) -> String {
    format!("'{}'", value.replace('\'', r"'\''"))
}

/// The text of the stand-in `uname`: log the arguments to `log`, answer `-v`, and
/// otherwise run `release_body` (a shell fragment).
#[must_use]
pub fn stand_in_uname_script(release_body: &str, log: &Path) -> String {
    let log = single_quoted(&log.to_string_lossy());
    format!(
        "#!/bin/sh\necho \"$*\" >> {log}\nif [ \"$1\" = -v ]; then echo '#1 SMP stand-in'; exit 0; fi\n{release_body}\n"
    )
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::process::Command;

    /// Plant the script in a directory whose name holds a single quote and a
    /// space, run it, and read back what it logged and printed.
    fn run_in_awkward_directory(args: &[&str], body: &str) -> (String, String) {
        let scratch = tempfile::tempdir().expect("scratch directory");
        let dir = scratch.path().join("it's a dir");
        std::fs::create_dir_all(&dir).expect("create the awkward directory");
        let log = dir.join("uname calls.log");
        let script = dir.join("uname");
        rocm_core::fake_executable::write_fake_executable(
            &script,
            stand_in_uname_script(body, &log),
        )
        .expect("plant the stand-in");
        let output = Command::new(&script).args(args).output().expect("run it");
        assert!(output.status.success(), "{output:?}");
        (
            String::from_utf8_lossy(&output.stdout).into_owned(),
            std::fs::read_to_string(&log).expect("the stand-in logged its call"),
        )
    }

    #[test]
    fn the_log_path_survives_a_single_quote_and_a_space_in_it() {
        let (stdout, log) = run_in_awkward_directory(&["-r"], "echo 6.8.0-stand-in");
        assert_eq!(stdout.trim(), "6.8.0-stand-in");
        assert_eq!(log, "-r\n");
    }

    #[test]
    fn dash_v_answers_a_fixed_string_and_is_logged_too() {
        let (stdout, log) = run_in_awkward_directory(&["-v"], "echo not-reached");
        assert_eq!(stdout.trim(), "#1 SMP stand-in");
        assert_eq!(log, "-v\n");
    }
}
