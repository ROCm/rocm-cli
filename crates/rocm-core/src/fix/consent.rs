// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Consent and the shared execution helpers of `rocm fix`.
//!
//! The consent decision (`consent_without_prompt`, `confirm`), the timeouts
//! the runners run host commands under, the running-OS family a recipe is
//! selected by (`current_os`), and the stderr/stdout helpers (`fail!`,
//! `relay_output`) that cannot themselves panic when a pipe is closed. Used
//! by `fix.rs` and its runners.
//!
//! Also the one way a runner runs a host command ([`run_step`]) and reports
//! how it ended ([`Step`], [`StepFailure`]): the line printed and the exit
//! code returned come from the same value, so a timeout cannot be reported as
//! an exit code, nor an unfinished change as "NOT changed".

use super::options::exit;
use crate::command_capture::{CommandCapture, CommandOutcome};
use crate::{runtime_is_linux, runtime_is_windows};
use std::io::{IsTerminal, Write};
use std::process::Command;
use std::time::Duration;

pub(super) const RUN_TIMEOUT: Duration = Duration::from_mins(1);
pub(super) const QUERY_TIMEOUT: Duration = Duration::from_secs(8);

/// Print a failure explanation to stderr, ignoring write failures (closed
/// stderr, full disk) so an I/O error while explaining a failure can't itself
/// panic the process.
macro_rules! fail {
    ($($arg:tt)*) => {{
        let _ = writeln!(std::io::stderr(), $($arg)*);
    }};
}
pub(super) use fail;

/// Relay a captured command's stdout/stderr, ignoring write failures for the
/// same reason `fail!` does — relaying subprocess output can't itself panic
/// the process if the pipe on the other end is closed.
pub(super) fn relay_output(out: &str, err: &str) {
    let _ = write!(std::io::stdout(), "{out}");
    let _ = write!(std::io::stderr(), "{err}");
}

/// Run one host command of a fix under `timeout` and say how it ended.
///
/// The runners take this as an injected
/// `FnMut(&str, &[&str], Duration) -> CommandCapture`, so their tests drive
/// every [`CommandOutcome`] without a real `setx`, `PowerShell` or `sudo`. It
/// calls the bounded executor directly rather than the `probe_exec::run`
/// adapter, which folds a timeout and a failed wait into exit code `124` and
/// a program that cannot start into `127` -- which is how a timed-out
/// `usermod` used to be reported as "usermod exited 124".
///
/// The executor stops the direct child only (see `process/bounded.rs`), so a
/// command run under `sudo` may still finish after a timeout; [`Step::failure`]
/// says so rather than claiming nothing changed.
pub(super) fn run_step(program: &str, args: &[&str], timeout: Duration) -> CommandCapture {
    let mut command = Command::new(program);
    command.args(args);
    crate::process::run_bounded(command, timeout, program)
}

/// How a command ended, as the first half of a sentence: "`label` timed out
/// after 60s and was stopped". `timeout` is the bound the command ran under.
fn outcome_phrase(label: &str, outcome: CommandOutcome, timeout: Duration) -> String {
    match outcome {
        CommandOutcome::Exited(code) => format!("{label} exited {code}"),
        CommandOutcome::NotFound => format!("{label} could not be started: it was not found"),
        CommandOutcome::SpawnFailed(kind) => format!("{label} could not be started: {kind}"),
        CommandOutcome::TimedOut => format!(
            "{label} timed out after {}s and was stopped",
            timeout.as_secs()
        ),
        CommandOutcome::WaitFailed => {
            format!("{label} could not be waited for and was stopped")
        }
        CommandOutcome::Signaled(Some(signal)) => {
            format!("{label} was killed by signal {signal} before it finished")
        }
        CommandOutcome::Signaled(None) => format!("{label} was killed before it finished"),
    }
}

/// Whether a command that ended this way may have made its change anyway.
///
/// A command that ran to its own exit with a failure code, or never started,
/// did not. One that was stopped from outside -- the deadline, a failed wait,
/// a signal -- may have been part-way through, or (under `sudo`) may still be
/// running, so the honest answer is "may or may not".
const fn may_have_acted(outcome: CommandOutcome) -> bool {
    matches!(
        outcome,
        CommandOutcome::TimedOut | CommandOutcome::WaitFailed | CommandOutcome::Signaled(_)
    )
}

/// One host command a fix runs to make its change, and how to recover when
/// the command is stopped before it reports.
pub(super) struct Step<'a> {
    /// The program run.
    pub(super) program: &'a str,
    /// Its arguments.
    pub(super) args: &'a [&'a str],
    /// The command as messages name it (`setx`, `PowerShell`, `sudo usermod`).
    pub(super) label: &'a str,
    /// What the command changes, as the subject of "... NOT changed."
    pub(super) what: &'a str,
    /// A command the user can run to see whether the change was made.
    pub(super) check: &'a str,
    /// Where `check` runs, which the line names when it is not a plain shell.
    pub(super) check_in: CheckIn,
    /// The `rocm fix` invocation that makes the change if it was not made.
    pub(super) rerun: &'a str,
    /// The bound the command runs under; also the number the timeout line names.
    pub(super) timeout: Duration,
}

/// Where a [`Step`]'s `check` command is run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CheckIn {
    /// Any shell: "Check with `id -nG alice`".
    Shell,
    /// A PowerShell expression, which `cmd` cannot run: "Check in PowerShell
    /// with `[Environment]::GetEnvironmentVariable(...)`".
    PowerShell,
}

impl CheckIn {
    /// The words before the backticked command.
    const fn lead(self) -> &'static str {
        match self {
            Self::Shell => "Check with",
            Self::PowerShell => "Check in PowerShell with",
        }
    }
}

/// A failed step: the line to print on stderr and the exit code to return,
/// decided together so the two cannot disagree.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct StepFailure {
    pub(super) code: i32,
    pub(super) line: String,
}

impl Step<'_> {
    /// `None` when the command exited `0`; otherwise what to print and return.
    pub(super) fn failure(&self, outcome: CommandOutcome) -> Option<StepFailure> {
        if outcome == CommandOutcome::Exited(0) {
            return None;
        }
        let phrase = outcome_phrase(self.label, outcome, self.timeout);
        let line = if may_have_acted(outcome) {
            format!(
                "{phrase}; {} may or may not have changed. {} `{}`, then re-run `{}` if it did not.",
                self.what,
                self.check_in.lead(),
                self.check,
                self.rerun
            )
        } else {
            format!("{phrase}; {} NOT changed.", self.what)
        };
        Some(StepFailure {
            code: exit::FAILED,
            line,
        })
    }

    /// Run the command through `run`, relay what it printed, and on failure
    /// write the [`StepFailure`] line to `err` and return its code.
    pub(super) fn run(
        &self,
        run: &mut impl FnMut(&str, &[&str], Duration) -> CommandCapture,
        err: &mut impl Write,
    ) -> Result<(), i32> {
        let capture = run(self.program, self.args, self.timeout);
        relay_output(&capture.stdout, &capture.stderr);
        match self.failure(capture.outcome) {
            None => Ok(()),
            Some(failure) => {
                let _ = writeln!(err, "{}", failure.line);
                Err(failure.code)
            }
        }
    }
}

/// A read that a fix depends on did not complete: nothing has been changed
/// yet, and the fix stops rather than guess what the value was.
pub(super) fn read_failure(
    subject: &str,
    label: &str,
    outcome: CommandOutcome,
    timeout: Duration,
) -> StepFailure {
    StepFailure {
        code: exit::FAILED,
        line: format!(
            "Could not read {subject}: {}; nothing was changed.",
            outcome_phrase(label, outcome, timeout)
        ),
    }
}

/// The platform family a recipe is selected by.
///
/// WSL is its own family, which is why recipes valid on both have to name `wsl`
/// explicitly. Not `const fn`: unlike the OS, WSL has to be probed at runtime.
pub(super) fn current_os() -> &'static str {
    if runtime_is_windows() {
        "windows"
    } else if runtime_is_linux() {
        if crate::is_wsl_host() { "wsl" } else { "linux" }
    } else {
        "other"
    }
}

/// The half of the consent decision that needs no I/O: `Some(verdict)` when the
/// answer is already determined, `None` when the user must actually be asked.
///
/// Split out from [`confirm`] so the two rules that matter — `--yes` approves,
/// and a non-interactive shell *refuses* rather than silently proceeding — are
/// testable without a controlled stdin. A test that drove `confirm` directly
/// would block on `read_line` whenever the suite happened to run with a terminal
/// attached.
const fn consent_without_prompt(assume_yes: bool, stdin_is_terminal: bool) -> Option<bool> {
    if assume_yes {
        return Some(true);
    }
    if !stdin_is_terminal {
        return Some(false);
    }
    None
}

pub(super) fn confirm(prompt: &str, assume_yes: bool) -> bool {
    if let Some(verdict) = consent_without_prompt(assume_yes, std::io::stdin().is_terminal()) {
        if !verdict {
            fail!("Non-interactive shell and --yes not passed; refusing to apply.");
        }
        return verdict;
    }
    let _ = crate::confirm::print_prompt(&mut std::io::stdout(), &format!("{prompt} [y/N]: "));
    let confirmed =
        crate::confirm::read_answer(&mut std::io::stdin().lock(), &mut std::io::stdout())
            .unwrap_or(false);
    if !confirmed {
        fail!("Not confirmed; refusing to apply.");
    }
    confirmed
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Restored, not new. This PR made the `wsl` arm of the platform lookup
    /// carry more weight, not less -- it now drives the per-platform class, the
    /// plan, the recipe printer and the listing -- so the guard tying that arm
    /// to the real host detector matters more here than it did before.
    #[test]
    fn current_os_reports_wsl_exactly_when_is_wsl_host_does() {
        // `current_os()`'s wsl branch is `crate::is_wsl_host()`, which is
        // `crate::wsl_signals_indicate_wsl()` against the real `/dev/dxg` and
        // `/proc/version` -- `$WSL_DISTRO_NAME` plays no part any more, so
        // there is nothing left to force portably here. The table-driven
        // coverage of the predicate itself lives with
        // `wsl_signals_indicate_wsl` in lib.rs; this test only checks that
        // `current_os()` reports the same answer `is_wsl_host()` does on
        // whatever machine actually runs it, whether that is a bare-metal CI
        // runner, Windows, or a real WSL host.
        //
        // In an `e2e-test-hooks` build both calls read through the simulated
        // host root, which another test may be setting and clearing, so hold
        // its lock to compare two answers about the same host. Without the
        // feature the lock is uncontended and costs nothing.
        let _guard = crate::hardware_root::HOST_ROOT_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(
            current_os() == "wsl",
            crate::is_wsl_host(),
            "current_os() must agree with is_wsl_host()"
        );
    }

    const STEP: Step<'static> = Step {
        program: "setx",
        args: &[],
        label: "setx",
        what: "User-scope X",
        check: "check-x",
        check_in: CheckIn::Shell,
        rerun: "rocm fix fix-x",
        timeout: RUN_TIMEOUT,
    };

    /// What [`Step::failure`] reports for `outcome`, written out per variant.
    /// The `match` is exhaustive on purpose: a new [`CommandOutcome`] does not
    /// compile here until someone decides what a stopped fix says about it.
    fn expected(outcome: CommandOutcome) -> Option<String> {
        let maybe = "User-scope X may or may not have changed. Check with `check-x`, then \
                     re-run `rocm fix fix-x` if it did not.";
        let not_changed = "User-scope X NOT changed.";
        Some(match outcome {
            CommandOutcome::Exited(0) => return None,
            CommandOutcome::Exited(_) => format!("setx exited 1; {not_changed}"),
            CommandOutcome::NotFound => {
                format!("setx could not be started: it was not found; {not_changed}")
            }
            CommandOutcome::SpawnFailed(_) => {
                format!("setx could not be started: permission denied; {not_changed}")
            }
            CommandOutcome::TimedOut => {
                format!("setx timed out after 60s and was stopped; {maybe}")
            }
            CommandOutcome::WaitFailed => {
                format!("setx could not be waited for and was stopped; {maybe}")
            }
            CommandOutcome::Signaled(Some(_)) => {
                format!("setx was killed by signal 9 before it finished; {maybe}")
            }
            CommandOutcome::Signaled(None) => {
                format!("setx was killed before it finished; {maybe}")
            }
        })
    }

    /// The line and the code come from one value: every outcome but a clean
    /// exit is exit 4, a timeout says "timed out" with the bound it ran under,
    /// and only a command that ran to its own end (or never started) is
    /// reported as having changed nothing.
    #[test]
    fn step_failure_reports_each_outcome_with_its_own_wording() {
        for outcome in [
            CommandOutcome::Exited(0),
            CommandOutcome::Exited(1),
            CommandOutcome::NotFound,
            CommandOutcome::SpawnFailed(std::io::ErrorKind::PermissionDenied),
            CommandOutcome::TimedOut,
            CommandOutcome::WaitFailed,
            CommandOutcome::Signaled(Some(9)),
            CommandOutcome::Signaled(None),
        ] {
            let got = STEP.failure(outcome);
            match expected(outcome) {
                None => assert_eq!(got, None, "{outcome:?} is success"),
                Some(line) => assert_eq!(
                    got,
                    Some(StepFailure {
                        code: exit::FAILED,
                        line
                    }),
                    "{outcome:?}"
                ),
            }
        }
    }

    /// The runner the production paths use reports a program that is not
    /// there as `NotFound`, not as the legacy adapter's `127`, so the line a
    /// runner prints for it is "could not be started".
    #[test]
    fn run_step_reports_a_missing_program_as_not_found() {
        let capture = run_step("rocm-fix-no-such-program-345", &[], QUERY_TIMEOUT);
        assert_eq!(capture.outcome, CommandOutcome::NotFound);
    }

    /// A PowerShell expression is not something `cmd` can run, so the line
    /// says where to run it; a plain shell command keeps "Check with".
    #[test]
    fn a_powershell_check_says_to_run_it_in_powershell() {
        let step = Step {
            check_in: CheckIn::PowerShell,
            ..STEP
        };
        let line = step
            .failure(CommandOutcome::TimedOut)
            .expect("a failure")
            .line;
        assert!(
            line.contains(" Check in PowerShell with `check-x`, then re-run"),
            "{line}"
        );
    }

    #[test]
    fn a_timeout_names_the_bound_the_step_ran_under() {
        let step = Step {
            timeout: QUERY_TIMEOUT,
            ..STEP
        };
        let line = step
            .failure(CommandOutcome::TimedOut)
            .expect("a failure")
            .line;
        assert!(line.starts_with("setx timed out after 8s "), "{line}");
    }

    #[test]
    fn a_failed_read_says_nothing_was_changed() {
        assert_eq!(
            read_failure(
                "the current User PATH",
                "PowerShell",
                CommandOutcome::Exited(1),
                QUERY_TIMEOUT
            ),
            StepFailure {
                code: exit::FAILED,
                line: "Could not read the current User PATH: PowerShell exited 1; nothing was \
                       changed."
                    .to_owned()
            }
        );
    }

    #[test]
    fn yes_flag_approves_without_prompting() {
        assert_eq!(consent_without_prompt(true, false), Some(true));
        assert_eq!(consent_without_prompt(true, true), Some(true));
    }

    #[test]
    fn non_interactive_shell_refuses_instead_of_proceeding() {
        // The rule the skill relies on: without --yes and with nothing to prompt,
        // the answer is a definite NO, never an implicit yes.
        assert_eq!(consent_without_prompt(false, false), Some(false));
    }

    #[test]
    fn interactive_shell_without_yes_must_actually_ask() {
        assert_eq!(consent_without_prompt(false, true), None);
    }
}
