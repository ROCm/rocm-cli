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
pub mod serve_log;

use std::path::{Path, PathBuf};

/// Select the single binary directory consumed by lifecycle packaging.
///
/// `xtask package` accepts one `ROCM_BIN_DIR`, so prebuilt `rocm` and `rocmd`
/// paths must name siblings. Reject a split layout instead of silently taking
/// whichever `rocmd` happens to sit next to `rocm`.
pub fn lifecycle_binary_dir(
    rocm: &Path,
    rocmd: Option<&Path>,
    fallback: &Path,
) -> Result<PathBuf, String> {
    let parent_or_fallback = |path: &Path| {
        path.parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .map_or_else(|| fallback.to_path_buf(), Path::to_path_buf)
    };
    let rocm_dir = parent_or_fallback(rocm);
    let rocmd = rocmd.ok_or_else(|| {
        "ROCM_CLI_ROCMD_BINARY must be set for lifecycle packaging so the packaged rocmd is explicit"
            .to_owned()
    })?;
    let rocmd_dir = parent_or_fallback(rocmd);
    if rocm_dir != rocmd_dir {
        return Err(format!(
            "lifecycle packaging requires ROCM_CLI_BINARY and ROCM_CLI_ROCMD_BINARY in the same directory; rocm uses {}, rocmd uses {}",
            rocm_dir.display(),
            rocmd_dir.display()
        ));
    }
    Ok(rocm_dir)
}

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
    use super::{chat_response_is_successful, cli_failure_report, lifecycle_binary_dir};
    use std::path::Path;

    #[test]
    fn lifecycle_packaging_rejects_rocm_and_rocmd_from_different_directories() {
        let error = lifecycle_binary_dir(
            Path::new("/prebuilt/cli/rocm"),
            Some(Path::new("/prebuilt/daemon/rocmd")),
            Path::new("/workspace/target/release"),
        )
        .expect_err("packaging cannot consume binaries from separate directories");

        assert!(
            error.contains("/prebuilt/cli"),
            "missing rocm directory: {error}"
        );
        assert!(
            error.contains("/prebuilt/daemon"),
            "missing rocmd directory: {error}"
        );
    }

    #[test]
    fn lifecycle_packaging_uses_the_directory_containing_both_configured_binaries() {
        let selected = lifecycle_binary_dir(
            Path::new("/prebuilt/bin/rocm"),
            Some(Path::new("/prebuilt/bin/rocmd")),
            Path::new("/workspace/target/release"),
        )
        .expect("sibling binaries must be accepted");

        assert_eq!(selected, Path::new("/prebuilt/bin"));
    }

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
