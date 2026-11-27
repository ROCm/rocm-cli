// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The markers of a fatal vLLM startup failure, shared by the engine's readiness
//! wait and the `rocm serve` failure summary.
//!
//! vLLM's API-server process stays alive and silent after its engine core dies,
//! so a readiness wait that only watches the child process finds out at its
//! timeout. The engine log is where the failure is visible at once. Two crates
//! read it and have to agree on what counts as fatal: `rocm-engine-vllm` scans
//! new log bytes on every poll to stop waiting, and `apps/rocm` names the cause
//! in the failure it prints afterwards.

use crate::{terminal, vllm_oom};

/// Lead-in of the sentence the engine's readiness wait fails with when it stops
/// waiting on a fatal log line.
///
/// Shared for the reason [`crate::VLLM_OOM_DETECTED_PREFIX`] is: the engine's bail
/// is written back into the shared log, and the line it quotes would otherwise
/// be found there a second time and read as the engine's own output.
/// [`vllm_oom::is_rocm_authored_log_line`] keys on it.
pub const VLLM_FATAL_STARTUP_DETECTED_PREFIX: &str = "Detected a fatal engine error (";

/// Case-insensitive substrings that mark a log line as a fatal vLLM startup
/// failure: the engine core giving up, or the GPU runtime faulting under it.
///
/// A line here is a reason to stop waiting for readiness, so each entry has to
/// be something a server that is still coming up never prints.
const FATAL_STARTUP_MARKERS: [&str; 5] = [
    "enginecore failed to start",
    "hsa_status_error_",
    "gpu stall detected",
    "hiperrorillegaladdress",
    "illegal memory access",
];

/// The longest fatal line [`vllm_fatal_startup_line`] returns, in characters.
const MAX_FATAL_LINE_CHARS: usize = 300;

/// The first line of `log_text` that marks a fatal startup failure, printable
/// and bounded, or `None` when there is none.
///
/// The first marker line rather than the last, because it is the root cause: the
/// engine core's own `EngineCore failed to start` follows the GPU fault that
/// made it fail. The line is subprocess output that reaches the user's terminal,
/// so it is returned as the [`terminal::rendered_lines`] row that matched
/// (control bytes removed without welding two drawn rows into one) and cut to
/// [`MAX_FATAL_LINE_CHARS`]. Lines rocm-cli wrote into the log itself are
/// skipped, so the bail this crate's readiness wait writes back is never read as
/// a second failure.
#[must_use]
pub fn vllm_fatal_startup_line(log_text: &str) -> Option<String> {
    log_text
        .lines()
        .flat_map(|line| line.split('\r'))
        .filter(|segment| !vllm_oom::is_rocm_authored_log_line(segment))
        .flat_map(terminal::rendered_lines)
        .find(|row| {
            let lower = row.to_ascii_lowercase();
            FATAL_STARTUP_MARKERS
                .iter()
                .any(|marker| lower.contains(marker))
        })
        .map(|row| {
            let row = row.trim();
            match row.char_indices().nth(MAX_FATAL_LINE_CHARS) {
                Some((end, _)) => format!("{}...", &row[..end]),
                None => row.to_owned(),
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_marker_is_found_whatever_its_case() {
        for line in [
            "(EngineCore pid=7) ERROR core.py:1 EngineCore failed to start.",
            "HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception",
            "(EngineCore pid=7) GPU stall detected on queue 3",
            "RuntimeError: HIP error: hipErrorIllegalAddress",
            "HIP error: an Illegal Memory Access was encountered",
        ] {
            assert_eq!(vllm_fatal_startup_line(line).as_deref(), Some(line));
        }
    }

    #[test]
    fn a_healthy_startup_log_has_no_fatal_line() {
        let log = "INFO Starting vLLM API server\n\
                   INFO Loading weights: 40%\n\
                   WARNING Using default chat template\n\
                   INFO Application startup complete.\n";
        assert_eq!(vllm_fatal_startup_line(log), None);
        assert_eq!(vllm_fatal_startup_line(""), None);
    }

    #[test]
    fn the_root_cause_wins_over_the_wrapper_that_follows_it() {
        let log = "INFO loading\n\
                   ERROR HSA_STATUS_ERROR_EXCEPTION: hardware exception\n\
                   ERROR EngineCore failed to start.\n";
        assert_eq!(
            vllm_fatal_startup_line(log).as_deref(),
            Some("ERROR HSA_STATUS_ERROR_EXCEPTION: hardware exception")
        );
    }

    #[test]
    fn the_line_is_the_row_the_terminal_drew_not_a_welded_one() {
        let log = "Loading weights: 90%\r\x1b[2KGPU stall detected\n";
        assert_eq!(
            vllm_fatal_startup_line(log).as_deref(),
            Some("GPU stall detected")
        );
    }

    #[test]
    fn terminal_control_bytes_never_reach_the_returned_line() {
        let log = "\x1b[31mEngineCore failed to start\x1b[0m\x07\n";
        let line = vllm_fatal_startup_line(log).expect("the marker is present");
        assert!(!line.chars().any(char::is_control), "{line:?}");
        assert!(line.contains("EngineCore failed to start"));
    }

    #[test]
    fn a_very_long_line_is_cut() {
        let log = format!("EngineCore failed to start {}\n", "x".repeat(5_000));
        let line = vllm_fatal_startup_line(&log).expect("the marker is present");
        assert!(line.chars().count() <= MAX_FATAL_LINE_CHARS + 3, "{line}");
        assert!(line.ends_with("..."));
    }

    #[test]
    fn the_engines_own_bail_is_not_read_back_as_a_second_failure() {
        let bail = format!(
            "{VLLM_FATAL_STARTUP_DETECTED_PREFIX}HSA_STATUS_ERROR_EXCEPTION: hardware exception)."
        );
        assert_eq!(vllm_fatal_startup_line(&bail), None);
    }
}
