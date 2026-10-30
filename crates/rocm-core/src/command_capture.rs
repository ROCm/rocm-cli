// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Running the `rocm` CLI as a child process and capturing what it printed.
//!
//! Both apps do this — `rocmd`'s sandbox and MCP clusters, and `rocm`'s chat
//! tool bridge when it replays an approved mutating command — so both use the
//! one [`CommandCapture`], spawn wrapper and text formatter here.
//!
//! The bounded-execution half — draining both pipes, the deadline, quoting a
//! timed-out child's output — is [`run_with_timeout`]'s, not this module's.
//!
//! The environment the child gets is the caller's to choose: each app passes
//! the variables it wants set (in practice the `ROCM_CLI_*_DIR` overrides that
//! keep the child on the same config/data/cache roots as its parent) rather
//! than this crate deciding for both.

use crate::binary_paths::daemon_binary_path;
use crate::process::run_with_timeout;
use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;
use std::time::Duration;

/// What a `rocm` command printed, and how it exited.
///
/// Usually filled in by [`run_rocm_capture_with_env`] from a real child. The
/// chat tool bridge also builds one by hand for read-only commands it answers
/// in-process, with no child at all, so the shape is "a command's result", not
/// proof that anything was spawned.
#[derive(Debug)]
pub struct CommandCapture {
    /// The program followed by its arguments.
    pub argv: Vec<String>,
    /// The exit code; `1` when a child had none (killed by a signal).
    pub exit_status: i32,
    pub stdout: String,
    pub stderr: String,
}

/// Render a capture as the text shown to a tool caller: stdout alone when
/// stderr is blank, stderr alone (labelled) when stdout is blank, and both
/// labelled otherwise. Each stream is trimmed.
pub fn command_capture_text(output: &CommandCapture) -> String {
    if output.stderr.trim().is_empty() {
        output.stdout.trim().to_owned()
    } else if output.stdout.trim().is_empty() {
        format!("stderr:\n{}", output.stderr.trim())
    } else {
        format!(
            "stdout:\n{}\n\nstderr:\n{}",
            output.stdout.trim(),
            output.stderr.trim()
        )
    }
}

/// Run the `rocm` binary ([`daemon_binary_path`]) with `args`, `env` set on
/// top of the inherited environment, bounded by `timeout`.
///
/// A non-zero exit is not an error: it comes back as a [`CommandCapture`]
/// carrying that exit status. `Err` means the call itself failed — the binary
/// could not be located or spawned, or the deadline passed.
///
/// # Errors
///
/// Fails when [`daemon_binary_path`] cannot resolve the binary, or when
/// [`run_with_timeout`] fails to spawn it or the child outlives `timeout`.
pub fn run_rocm_capture_with_env(
    args: &[&str],
    env: &[(&str, &Path)],
    timeout: Duration,
) -> Result<CommandCapture> {
    capture_program(&daemon_binary_path()?, args, env, timeout)
}

fn capture_program(
    program: &Path,
    args: &[&str],
    env: &[(&str, &Path)],
    timeout: Duration,
) -> Result<CommandCapture> {
    let mut command = Command::new(program);
    // Stdio left to `run_with_timeout`, which sets it itself.
    command.args(args);
    for (key, value) in env {
        command.env(key, value);
    }
    let output = run_with_timeout(command, timeout, "process")
        .with_context(|| format!("failed to run {}", program.display()))?;
    Ok(CommandCapture {
        argv: std::iter::once(program.display().to_string())
            .chain(args.iter().map(|value| (*value).to_owned()))
            .collect(),
        exit_status: output.status.code().unwrap_or(1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_capture_text_prefers_stdout_only_when_stderr_is_blank() {
        let output = CommandCapture {
            argv: vec!["rocm".to_owned()],
            exit_status: 0,
            stdout: "  all good  \n".to_owned(),
            stderr: "  \n".to_owned(),
        };
        assert_eq!(command_capture_text(&output), "all good");
    }

    #[test]
    fn command_capture_text_falls_back_to_stderr_when_stdout_is_blank() {
        let output = CommandCapture {
            argv: vec!["rocm".to_owned()],
            exit_status: 1,
            stdout: "   ".to_owned(),
            stderr: "boom".to_owned(),
        };
        assert_eq!(command_capture_text(&output), "stderr:\nboom");
    }

    #[test]
    fn command_capture_text_joins_both_streams_when_present() {
        let output = CommandCapture {
            argv: vec!["rocm".to_owned()],
            exit_status: 1,
            stdout: "out".to_owned(),
            stderr: "err".to_owned(),
        };
        assert_eq!(
            command_capture_text(&output),
            "stdout:\nout\n\nstderr:\nerr"
        );
    }

    /// The caller-supplied `env` reaches the child, and a non-zero exit comes
    /// back as a capture — argv, exit status and both streams — rather than an
    /// error. Exercised through `capture_program` because the public entry point
    /// spawns `daemon_binary_path()`, which from this crate's unit tests is the
    /// built `rocm` next to the harness when there is one and the harness itself
    /// when there is not — neither a child this test can script.
    #[test]
    fn a_failing_child_is_captured_with_the_env_it_was_given() {
        const VAR: &str = "ROCM_CLI_COMMAND_CAPTURE_TEST_VALUE";
        let value = Path::new("marker-value");
        let (program, args): (&Path, &[&str]) = if cfg!(windows) {
            // No space around `&`: cmd would carry a trailing one into the
            // echoed text.
            (
                Path::new("cmd"),
                &[
                    "/C",
                    "echo %ROCM_CLI_COMMAND_CAPTURE_TEST_VALUE%&echo oops 1>&2&exit 3",
                ],
            )
        } else {
            (
                Path::new("sh"),
                &[
                    "-c",
                    "echo \"$ROCM_CLI_COMMAND_CAPTURE_TEST_VALUE\"; echo oops >&2; exit 3",
                ],
            )
        };

        let output = capture_program(program, args, &[(VAR, value)], Duration::from_secs(10))
            .expect("a non-zero exit must be captured, not returned as an error");

        assert_eq!(
            output.argv,
            std::iter::once(program.display().to_string())
                .chain(args.iter().map(|arg| (*arg).to_owned()))
                .collect::<Vec<_>>()
        );
        assert_eq!(output.exit_status, 3);
        assert_eq!(output.stdout.trim(), "marker-value");
        assert_eq!(output.stderr.trim(), "oops");
    }
}
