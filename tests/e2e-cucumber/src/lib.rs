// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

pub mod capability;
pub mod expectation;
pub mod http_server;
pub mod installer_fixture;
pub mod loopback_http;
pub mod mock_server;
pub mod model_id;
pub mod panic_capture;
pub mod reader_failure;
pub mod send_until;
pub mod serve_log;

/// Render everything known about a failed `rocm` invocation.
///
/// Both streams are always shown, each labelled and each with an explicit
/// `(empty)` marker. A bare `(empty)` is a finding in itself — it says the CLI
/// died without explaining itself — whereas an omitted section just looks like
/// the harness lost the output.
///
/// Exists because a step that asserts on the exit code while printing only
/// stdout leaves a failed step undiagnosable: the panic reads `rocm serve
/// failed:` followed by nothing at all, which is what EAI-8031 hit on the
/// MI300X lane. The CLI reports its errors on stderr.
pub fn cli_failure_report(args: &[&str], rc: i32, stdout: &str, stderr: &str) -> String {
    fn section(label: &str, body: &str) -> String {
        let body = body.trim_end();
        if body.is_empty() {
            format!("--- {label}: (empty) ---")
        } else {
            format!("--- {label} ---\n{body}")
        }
    }
    format!(
        "`rocm {}` failed (rc={rc})\n{}\n{}",
        args.join(" "),
        section("stdout", stdout),
        section("stderr", stderr),
    )
}

pub fn chat_response_is_successful(response: &serde_json::Value) -> bool {
    response
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|choices| !choices.is_empty())
}

// The report generator lives in its own lean crate (only maud + serde) so xtask
// can reuse it without pulling this harness's heavy tree. Re-export it under the
// original path so `e2e_cucumber::report::{generate, evaluate_xfail}` call sites
// keep working.
pub use e2e_report as report;

#[cfg(test)]
mod tests {
    use super::{chat_response_is_successful, cli_failure_report};

    /// The regression this whole helper exists for: a serve that dies with
    /// nothing on stdout must still show the reason, which is on stderr.
    #[test]
    fn failure_report_shows_stderr_when_stdout_is_empty() {
        let report = cli_failure_report(
            &["serve", "unsloth/Qwen3-0.6B-GGUF:Q4_0", "--managed"],
            1,
            "",
            "error: no llama-server backend found",
        );
        assert!(
            report.contains("no llama-server backend found"),
            "the reason must survive into the panic message:\n{report}"
        );
        assert!(
            report.contains("rc=1"),
            "exit code must be shown:\n{report}"
        );
        assert!(
            report.contains("serve unsloth/Qwen3-0.6B-GGUF:Q4_0 --managed"),
            "the failing invocation must be identifiable:\n{report}"
        );
    }

    /// An empty stream is labelled rather than omitted: "the CLI said nothing"
    /// and "the harness dropped the output" are different diagnoses.
    #[test]
    fn failure_report_marks_empty_streams_explicitly() {
        let report = cli_failure_report(&["examine"], 2, "", "");
        assert!(report.contains("stdout: (empty)"), "{report}");
        assert!(report.contains("stderr: (empty)"), "{report}");
    }

    #[test]
    fn failure_report_keeps_both_streams_when_both_are_present() {
        let report = cli_failure_report(&["install", "sdk"], 3, "plan line", "boom");
        assert!(report.contains("plan line"), "{report}");
        assert!(report.contains("boom"), "{report}");
    }

    #[test]
    fn chat_success_requires_non_empty_choices_array() {
        assert!(!chat_response_is_successful(&serde_json::json!({})));
        assert!(!chat_response_is_successful(
            &serde_json::json!({"choices": null})
        ));
        assert!(!chat_response_is_successful(
            &serde_json::json!({"choices": {}})
        ));
        assert!(!chat_response_is_successful(
            &serde_json::json!({"choices": []})
        ));
        assert!(chat_response_is_successful(
            &serde_json::json!({"choices": [{}]})
        ));
    }
}
