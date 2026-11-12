// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! What `tasklist` says about one process, read so that a failed query cannot
//! pass for an answer.
//!
//! A scenario that wants to know whether a process a stop was meant to take down
//! is still running asks `tasklist /FI "PID eq N" /FO CSV /NH`. A match prints a
//! CSV row naming the PID and no match prints an INFO line without one, but the
//! command can also fail (it cannot be started under that name, it is refused, it
//! times out) and print nothing useful. Reading "no row" from such a run as "the
//! process exited" lets the scenario pass while the process may still be running,
//! so the exit status is judged before the text. The read is a pure function of
//! the command's status and output, so a table test covers every case on every
//! host.

/// Whether `tasklist` reported process `pid` running.
///
/// `success` is whether the command exited successfully and `stdout` is what it
/// printed. `Ok(true)` is a CSV row for `pid`, `Ok(false)` is a successful run
/// with no such row, and `Err` is a run that did not succeed: its output says
/// nothing about the process, so there is no answer.
///
/// # Errors
///
/// Returns a message naming the PID and what the failed run printed when the
/// command did not exit successfully.
pub fn tasklist_reports_running(
    success: bool,
    stdout: &str,
    stderr: &str,
    pid: u32,
) -> Result<bool, String> {
    if !success {
        return Err(format!(
            "`tasklist` did not exit successfully, so it cannot say whether process {pid} \
             is still running (stdout: {:?}, stderr: {:?})",
            stdout.trim(),
            stderr.trim()
        ));
    }
    Ok(stdout.contains(&format!("\"{pid}\"")))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROW: &str = "\"ping.exe\",\"4242\",\"Console\",\"1\",\"3,044 K\"\r\n";
    const NO_MATCH: &str = "INFO: No tasks are running which match the specified criteria.\r\n";

    #[test]
    fn a_successful_run_with_a_row_for_the_pid_is_running() {
        assert_eq!(tasklist_reports_running(true, ROW, "", 4242), Ok(true));
    }

    #[test]
    fn a_successful_run_without_a_row_for_the_pid_is_not_running() {
        assert_eq!(
            tasklist_reports_running(true, NO_MATCH, "", 4242),
            Ok(false)
        );
        // A row for another PID is not a row for this one, nor is a PID that
        // only contains it.
        assert_eq!(tasklist_reports_running(true, ROW, "", 424), Ok(false));
        assert_eq!(tasklist_reports_running(true, ROW, "", 7), Ok(false));
    }

    #[test]
    fn a_failed_run_is_no_answer_whatever_it_printed() {
        // No output to read: "no row" must not become "exited".
        let err = tasklist_reports_running(false, "", "Access is denied.", 4242)
            .expect_err("a failed query is not an answer");
        assert!(
            err.contains("4242") && err.contains("Access is denied."),
            "{err}"
        );
        // Nor does a row printed before the failure make it one.
        assert!(tasklist_reports_running(false, ROW, "", 4242).is_err());
        assert!(tasklist_reports_running(false, NO_MATCH, "", 4242).is_err());
    }
}
