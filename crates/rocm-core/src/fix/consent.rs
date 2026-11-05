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

use crate::{runtime_is_linux, runtime_is_windows};
use std::io::{IsTerminal, Write};
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
    let confirmed = crate::confirm::read_answer(&mut std::io::stdin().lock()).unwrap_or(false);
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
