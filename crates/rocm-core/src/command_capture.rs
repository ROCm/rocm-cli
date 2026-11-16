// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The result of running a command, and running the `rocm` CLI as a child
//! process and capturing what it printed.
//!
//! [`CommandCapture`] is the one carrier of a command's result: the raw bytes
//! of both streams, a typed [`CommandOutcome`] and the status, for the host
//! probes the bounded executor ([`crate::process`]) runs as well as for the
//! `rocm` captures below. Both apps run `rocm` as a child — `rocmd`'s sandbox
//! and MCP clusters, and `rocm`'s chat tool bridge when it replays an approved
//! mutating command — so both use the one [`CommandCapture`], spawn wrapper and
//! text formatter here.
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
use std::io::ErrorKind;
use std::path::Path;
use std::process::{Command, ExitStatus, Output};
use std::time::Duration;

/// How a command ended, as far as the bounded executor
/// ([`crate::process::run_with_timeout`]'s core) can tell.
///
/// The variants are the ways a run can end that a caller handles differently:
/// a probe treats "not installed" quietly, "timed out" as a finding, and "exited
/// with a code" by reading the code. Telling a child killed by a signal from one
/// that exited with `1` is what the old integer `exit_status` could not do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandOutcome {
    /// The program does not exist ([`ErrorKind::NotFound`] from the spawn).
    NotFound,
    /// The program could not be started for another reason, for example
    /// [`ErrorKind::PermissionDenied`].
    SpawnFailed(ErrorKind),
    /// The deadline passed and the child was killed.
    TimedOut,
    /// The executor could not poll or reap the child, killed it and gave up.
    WaitFailed,
    /// The child exited with this code.
    Exited(i32),
    /// The child ended without an exit code: killed by a signal. The number is
    /// the signal where the platform reports one (Unix), else `None`.
    Signaled(Option<i32>),
}

impl CommandOutcome {
    /// The outcome of a spawn that failed with `kind`.
    #[must_use]
    pub fn from_spawn_error(kind: ErrorKind) -> Self {
        if kind == ErrorKind::NotFound {
            Self::NotFound
        } else {
            Self::SpawnFailed(kind)
        }
    }

    /// The outcome of a child that was waited for and not timed out: an exit
    /// code when there is one, else a signal. Takes the two parts rather than an
    /// [`ExitStatus`] so every host can check it with the values a Windows or a
    /// Unix status would give.
    #[must_use]
    pub const fn from_exit(code: Option<i32>, signal: Option<i32>) -> Self {
        match code {
            Some(code) => Self::Exited(code),
            None => Self::Signaled(signal),
        }
    }

    /// The exit code when the child exited with one.
    #[must_use]
    pub const fn exit_code(self) -> Option<i32> {
        match self {
            Self::Exited(code) => Some(code),
            _ => None,
        }
    }
}

/// The outcome of a child that was waited for to the end of its life.
pub(crate) fn outcome_of_status(status: ExitStatus) -> CommandOutcome {
    #[cfg(unix)]
    let signal = std::os::unix::process::ExitStatusExt::signal(&status);
    #[cfg(not(unix))]
    let signal = None;
    CommandOutcome::from_exit(status.code(), signal)
}

/// What a command printed, and how it ended.
///
/// Filled in by [`run_rocm_capture_with_env`] from a real child, and by the
/// bounded executor for the host probes. The chat tool bridge also builds one by
/// hand ([`Self::from_text`]) for read-only commands it answers in-process, with
/// no child at all (the daemon's tests build them the same way), so the shape is
/// "a command's result", not proof that anything was spawned.
///
/// The text fields are the lossy form of the raw ones, kept because most callers
/// read text; a caller that must decode the bytes itself (UTF-16LE from
/// `wsl.exe`, or a strict UTF-8 check) reads `stdout_raw` and `stderr_raw`.
#[derive(Debug)]
pub struct CommandCapture {
    /// The program followed by its arguments.
    pub argv: Vec<String>,
    /// The exit code; `1` when there was none (a signal, a timeout, a program
    /// that could not be started). Read [`Self::outcome`] to tell those apart.
    pub exit_status: i32,
    /// Standard output, lossily decoded from [`Self::stdout_raw`].
    pub stdout: String,
    /// Standard error, lossily decoded from [`Self::stderr_raw`].
    pub stderr: String,
    /// Standard output as the child wrote it. On a timeout, whatever arrived
    /// before the deadline and the drain grace.
    pub stdout_raw: Vec<u8>,
    /// Standard error as the child wrote it.
    pub stderr_raw: Vec<u8>,
    /// How the command ended.
    pub outcome: CommandOutcome,
    /// The status the operating system reported for a child that was reaped, for
    /// a caller that prints it. `None` for a program that was never started or a
    /// capture built by hand.
    pub status: Option<ExitStatus>,
    /// What the executor saw go wrong, in words: the spawn or wait error, or the
    /// read error that cut a pipe short (the outcome is then still what the
    /// child did, and the bytes are what arrived before the error).
    pub detail: Option<String>,
}

impl CommandCapture {
    /// A capture of a command that was answered without a child.
    #[must_use]
    pub fn from_text(argv: Vec<String>, exit_status: i32, stdout: String, stderr: String) -> Self {
        Self {
            argv,
            exit_status,
            stdout_raw: stdout.clone().into_bytes(),
            stderr_raw: stderr.clone().into_bytes(),
            stdout,
            stderr,
            outcome: CommandOutcome::Exited(exit_status),
            status: None,
            detail: None,
        }
    }

    /// A capture of a finished child's [`Output`].
    fn from_output(argv: Vec<String>, output: Output) -> Self {
        let outcome = outcome_of_status(output.status);
        Self::from_run(
            argv,
            outcome,
            Some(output.status),
            output.stdout,
            output.stderr,
            None,
        )
    }

    /// The capture the bounded executor builds from what a run left behind.
    #[must_use]
    pub(crate) fn from_run(
        argv: Vec<String>,
        outcome: CommandOutcome,
        status: Option<ExitStatus>,
        stdout_raw: Vec<u8>,
        stderr_raw: Vec<u8>,
        detail: Option<String>,
    ) -> Self {
        Self {
            argv,
            exit_status: outcome.exit_code().unwrap_or(1),
            stdout: String::from_utf8_lossy(&stdout_raw).into_owned(),
            stderr: String::from_utf8_lossy(&stderr_raw).into_owned(),
            stdout_raw,
            stderr_raw,
            outcome,
            status,
            detail,
        }
    }
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
    Ok(CommandCapture::from_output(
        std::iter::once(program.display().to_string())
            .chain(args.iter().map(|value| (*value).to_owned()))
            .collect(),
        output,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_capture_text_prefers_stdout_only_when_stderr_is_blank() {
        let output = CommandCapture::from_text(
            vec!["rocm".to_owned()],
            0,
            "  all good  \n".to_owned(),
            "  \n".to_owned(),
        );
        assert_eq!(command_capture_text(&output), "all good");
    }

    #[test]
    fn command_capture_text_falls_back_to_stderr_when_stdout_is_blank() {
        let output = CommandCapture::from_text(
            vec!["rocm".to_owned()],
            1,
            "   ".to_owned(),
            "boom".to_owned(),
        );
        assert_eq!(command_capture_text(&output), "stderr:\nboom");
    }

    #[test]
    fn command_capture_text_joins_both_streams_when_present() {
        let output = CommandCapture::from_text(
            vec!["rocm".to_owned()],
            1,
            "out".to_owned(),
            "err".to_owned(),
        );
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

    #[test]
    fn an_exit_code_outranks_a_signal_and_no_code_is_a_signal() {
        assert_eq!(
            CommandOutcome::from_exit(Some(3), Some(9)),
            CommandOutcome::Exited(3)
        );
        assert_eq!(
            CommandOutcome::from_exit(Some(0), None),
            CommandOutcome::Exited(0)
        );
        assert_eq!(
            CommandOutcome::from_exit(None, Some(9)),
            CommandOutcome::Signaled(Some(9))
        );
        assert_eq!(
            CommandOutcome::from_exit(None, None),
            CommandOutcome::Signaled(None),
            "a status with neither (no signal number on Windows) is still not an exit"
        );
        assert_eq!(CommandOutcome::Exited(3).exit_code(), Some(3));
        for outcome in [
            CommandOutcome::NotFound,
            CommandOutcome::TimedOut,
            CommandOutcome::WaitFailed,
            CommandOutcome::Signaled(Some(9)),
            CommandOutcome::SpawnFailed(ErrorKind::PermissionDenied),
        ] {
            assert_eq!(outcome.exit_code(), None, "{outcome:?}");
        }
    }

    #[test]
    fn only_a_missing_program_is_not_found() {
        assert_eq!(
            CommandOutcome::from_spawn_error(ErrorKind::NotFound),
            CommandOutcome::NotFound
        );
        assert_eq!(
            CommandOutcome::from_spawn_error(ErrorKind::PermissionDenied),
            CommandOutcome::SpawnFailed(ErrorKind::PermissionDenied)
        );
    }

    #[test]
    fn a_status_is_an_exit_code_or_a_signal() {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt as _;
            // The raw wait status: the exit code in the second byte, or the
            // signal number in the low bits.
            assert_eq!(
                outcome_of_status(ExitStatus::from_raw(3 << 8)),
                CommandOutcome::Exited(3)
            );
            assert_eq!(
                outcome_of_status(ExitStatus::from_raw(9)),
                CommandOutcome::Signaled(Some(9))
            );
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::ExitStatusExt as _;
            assert_eq!(
                outcome_of_status(ExitStatus::from_raw(3)),
                CommandOutcome::Exited(3)
            );
        }
    }

    #[test]
    fn a_hand_built_capture_is_an_exit_whose_bytes_are_its_text() {
        let capture = CommandCapture::from_text(
            vec!["rocm".to_owned()],
            2,
            "out".to_owned(),
            "err".to_owned(),
        );

        assert_eq!(capture.outcome, CommandOutcome::Exited(2));
        assert_eq!(capture.exit_status, 2);
        assert_eq!(capture.stdout_raw, b"out");
        assert_eq!(capture.stderr_raw, b"err");
        assert!(capture.status.is_none() && capture.detail.is_none());
    }

    #[test]
    fn a_run_keeps_its_raw_bytes_and_reports_text_from_them_lossily() {
        let capture = CommandCapture::from_run(
            vec!["tool".to_owned()],
            CommandOutcome::TimedOut,
            None,
            b"ok\xffdone".to_vec(),
            Vec::new(),
            None,
        );

        assert_eq!(capture.stdout_raw, b"ok\xffdone");
        assert_eq!(capture.stdout, "ok\u{FFFD}done");
        assert_eq!(capture.exit_status, 1, "no exit code to report");

        let exited = CommandCapture::from_run(
            Vec::new(),
            CommandOutcome::Exited(7),
            None,
            Vec::new(),
            Vec::new(),
            None,
        );
        assert_eq!(exited.exit_status, 7);
    }

    /// Subprocess-runners pair of `docs/host-probing.md`: a child killed by a
    /// signal has no exit code, so the legacy `exit_status` reads `1`, which is
    /// also what a child that exits with `1` reads; the typed `outcome` is what
    /// tells them apart.
    #[cfg(unix)]
    #[test]
    fn a_child_killed_by_a_signal_reads_as_exit_status_1() {
        let killed = capture_program(
            Path::new("sh"),
            &["-c", "echo partial; kill -9 $$"],
            &[],
            Duration::from_secs(10),
        )
        .expect("a signal death is captured, not an error");
        let exited = capture_program(
            Path::new("sh"),
            &["-c", "echo partial; exit 1"],
            &[],
            Duration::from_secs(10),
        )
        .expect("a non-zero exit is captured, not an error");

        assert_eq!(killed.exit_status, 1);
        assert_eq!(killed.stdout.trim(), "partial");
        assert_eq!(
            killed.exit_status, exited.exit_status,
            "the two cannot be told apart from the exit status"
        );
        assert_eq!(killed.outcome, CommandOutcome::Signaled(Some(9)));
        assert_eq!(exited.outcome, CommandOutcome::Exited(1));
    }

    /// A program that cannot be started is an error carrying the program, not a
    /// capture, so a caller cannot tell "not installed" from a timeout except by
    /// the text of the message.
    #[test]
    fn a_program_that_cannot_be_started_is_an_error_naming_it() {
        let missing = Path::new("/nonexistent-dir-for-the-runner-pair/tool");
        let error = capture_program(missing, &[], &[], Duration::from_secs(10))
            .expect_err("a spawn failure is an error here");

        let rendered = format!("{error:#}");
        assert!(
            rendered.contains(&format!("failed to run {}", missing.display())),
            "{rendered}"
        );
        assert!(rendered.contains("failed to spawn process"), "{rendered}");
    }
}
