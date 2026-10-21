// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

pub mod capability;
pub mod chat_request;
pub mod expectation;
pub mod fixture_binary;
pub mod harness_diagnostics;
pub mod http_server;
pub mod installer_fixture;
pub mod loopback_http;
pub mod mock_server;
pub mod model_id;
pub mod monotonic_clock;
pub mod paced_download;
pub mod panic_capture;
pub mod quit_and_wait;
pub mod reader_failure;
pub mod send_until;
pub mod serve_log;
pub mod shared_runtime;

use std::path::{Path, PathBuf};

/// Whether `id` names a `rocm fix` catalog entry.
///
/// A remediable `fix-`/`fix-wsl-` one, or a `DiagnoseOnly` `info-` one.
/// Shared so a future third prefix (or an `info-` rename) is a one-line
/// change instead of four.
#[must_use]
pub fn is_catalog_id(id: &str) -> bool {
    id.starts_with("fix-") || id.starts_with("info-")
}

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

/// One labelled block of a diagnostic bundle.
///
/// An empty body is marked `(empty)` rather than omitted, because "the CLI said
/// nothing" and "the harness lost the output" are different diagnoses and only
/// an explicit marker tells them apart. Shared by [`cli_failure_report`] and
/// [`serve_log::serve_attempt_report`] so every bundle reads the same way.
pub(crate) fn section(label: &str, body: &str) -> String {
    let body = body.trim_end();
    if body.is_empty() {
        format!("--- {label}: (empty) ---")
    } else {
        format!("--- {label} ---\n{body}")
    }
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
///
/// Covers only a NON-ZERO exit. A `--managed` serve that exits 0 and then never
/// serves needs [`serve_log::serve_attempt_report`] instead — same streams, plus
/// the engine log and device state that are the only evidence in that case.
pub fn cli_failure_report(args: &[&str], rc: i32, stdout: &str, stderr: &str) -> String {
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

/// Opening anchor of the region of `rocm fix fix-6-path` output that describes
/// the User PATH write: the first line `path_export_windows` prints once it has
/// read the existing User PATH and decided what to write.
const PATH_WRITE_PLAN_PREFIX: &str = "Plan: prepend ";

/// Closing anchor: the confirmation `path_export_windows` prints after the
/// write has succeeded.
const PATH_WRITE_CONFIRMATION: &str = "This only takes effect in NEW shells";

/// The slice of `rocm fix fix-6-path` output that is the PATH *write*.
///
/// That is the `Plan: prepend ...` line, the PowerShell script line printed
/// under it, anything that run relayed, and the confirmation line that
/// follows. `None` when either anchor is missing, which is itself a failure
/// the caller should report.
///
/// Scoped rather than matched against the whole output on purpose.
/// `rocm fix <id>` prints the catalog recipe — rationale, command block,
/// notes — *before* it runs anything, so the whole output legitimately
/// contains prose about the fix as well as the fix's own actions. An
/// assertion that `setx` appears nowhere in all of that answers a question
/// about the documentation, not about what the CLI did: it fails when the
/// catalog merely *mentions* `setx` (including, as `fix-6-path`'s notes now
/// do, to warn a user off it), while still being unable to distinguish the
/// write from any other line. This region is the span in which `setx` would
/// mean the regression the scenario is guarding against, and the only span in
/// which the PowerShell call proves the write went the right way.
#[must_use]
pub fn user_path_write_region(output: &str) -> Option<&str> {
    let plan = output.find(PATH_WRITE_PLAN_PREFIX)?;
    let region = &output[plan..];
    let confirmation = region.find(PATH_WRITE_CONFIRMATION)?;
    let end = region[confirmation..]
        .find('\n')
        .map_or(region.len(), |newline| confirmation + newline);
    Some(&region[..end])
}

// The report generator lives in its own lean crate (only maud + serde) so xtask
// can reuse it without pulling this harness's heavy tree. Re-export it under the
// original path so `e2e_cucumber::report::{generate, evaluate_xfail}` call sites
// keep working.
pub use e2e_report as report;

#[cfg(test)]
mod tests {
    use super::{
        chat_response_is_successful, cli_failure_report, lifecycle_binary_dir,
        user_path_write_region,
    };
    use std::path::Path;

    /// What `rocm fix fix-6-path --yes` prints *before* it does anything: the
    /// catalog recipe, reproduced here as `print_recipe` lays it out. The
    /// notes deliberately name `setx` — warning a Windows user off the
    /// `setx PATH "%PATH%;..."` one-liner is the point of them — which is
    /// exactly the text a whole-output `setx` ban cannot tell apart from a
    /// regressed write.
    const PATH_FIX_CATALOG_BLOCK: &str = concat!(
        "Fix:        fix-6-path  -- Add the ROCm/HIP bin directory to PATH\n",
        "OS scope:   linux, windows, wsl\n",
        "Rationale:  Windows: the HIP SDK is installed but its bin directory isn't on the User PATH.\n",
        "Commands:\n",
        "  $ # Linux:\n",
        "  $ echo 'export PATH=\"/opt/rocm/bin:$PATH\"' >> ~/.bashrc\n",
        "  $ # Windows (PowerShell):\n",
        "  $ [Environment]::SetEnvironmentVariable('PATH', 'C:\\Program Files\\AMD\\ROCm\\<version>\\bin;' + [Environment]::GetEnvironmentVariable('PATH', 'User'), 'User')\n",
        "Flags:      rocm fix can run it\n",
        "Note:       Windows: the PowerShell line reads and rewrites the User scope only. Do not substitute `setx PATH \"%PATH%;...\"`: setx silently truncates the value at 1024 characters.\n",
        "Note:       Windows: a PATH change only takes effect in NEW shells.\n",
        "Verify:     rocminfo | head -n 5 && hipcc --version\n",
        "\n",
    );

    /// What `path_export_windows` prints when the write goes the way
    /// `cb9f748d` made it go.
    const PATH_WRITE_THROUGH_POWERSHELL: &str = concat!(
        "Plan: prepend C:\\ROCm\\bin to your User PATH:\n",
        "  [Environment]::SetEnvironmentVariable('PATH', 'C:\\ROCm\\bin;C:\\Windows', 'User')\n",
        "Added C:\\ROCm\\bin to your User PATH. This only takes effect in NEW shells -- close this terminal and reopen it before re-running hipInfo.\n",
    );

    /// The regression the scenario exists to catch: the same plan/confirmation
    /// framing, with the write itself back on `setx`.
    const PATH_WRITE_REGRESSED_TO_SETX: &str = concat!(
        "Plan: prepend C:\\ROCm\\bin to your User PATH:\n",
        "  setx PATH \"%PATH%;C:\\ROCm\\bin\"\n",
        "Added C:\\ROCm\\bin to your User PATH. This only takes effect in NEW shells -- close this terminal and reopen it before re-running hipInfo.\n",
    );

    /// The narrowing this helper exists for. The catalog block above the write
    /// says `setx` twice, on purpose; the write below it does not. A
    /// whole-output `!contains("setx ")` — what diagnose-38 asserted before —
    /// fails on this output even though the CLI did exactly the right thing,
    /// which is what left `main`'s Windows E2E lane permanently red.
    #[test]
    fn the_write_region_excludes_the_catalog_prose_that_merely_names_setx() {
        let output = format!("{PATH_FIX_CATALOG_BLOCK}{PATH_WRITE_THROUGH_POWERSHELL}");

        assert!(
            output.contains("setx "),
            "the fixture must reproduce the catalog text the old assertion tripped on"
        );

        let region = user_path_write_region(&output).expect("a successful write has both anchors");

        assert!(
            region.contains("[Environment]::SetEnvironmentVariable("),
            "the write itself must be the PowerShell call:\n{region}"
        );
        assert!(
            !region.contains("setx "),
            "no part of the write mentions setx; only the catalog above it does:\n{region}"
        );
    }

    /// ...and the narrowing must not have narrowed the guard away: the same
    /// output with the write back on `setx` still fails.
    #[test]
    fn the_write_region_still_catches_a_write_that_regressed_to_setx() {
        let output = format!("{PATH_FIX_CATALOG_BLOCK}{PATH_WRITE_REGRESSED_TO_SETX}");

        let region = user_path_write_region(&output).expect("the plan line is still printed");

        assert!(
            region.contains("setx "),
            "a write that went back to setx must land inside the asserted region:\n{region}"
        );
        assert!(
            !region.contains("[Environment]::SetEnvironmentVariable("),
            "the PowerShell call is gone in the regressed output:\n{region}"
        );
    }

    /// A run that printed the catalog and then never reached the write has no
    /// region at all, rather than an empty one that would vacuously satisfy
    /// both halves of the assertion.
    #[test]
    fn there_is_no_write_region_when_the_write_never_ran() {
        assert!(user_path_write_region(PATH_FIX_CATALOG_BLOCK).is_none());
    }

    /// The plan was printed but the write failed or was aborted: no
    /// confirmation, so no region. Half a write must not read as a whole one.
    #[test]
    fn there_is_no_write_region_when_the_write_was_not_confirmed() {
        let output = format!(
            "{PATH_FIX_CATALOG_BLOCK}Plan: prepend C:\\ROCm\\bin to your User PATH:\n  [Environment]::SetEnvironmentVariable('PATH', 'C:\\ROCm\\bin', 'User')\n(dry-run; not executed)\n"
        );

        assert!(user_path_write_region(&output).is_none());
    }

    /// The region stops at the end of the confirmation line, so output printed
    /// after the write is outside it for the same reason the catalog above is.
    #[test]
    fn the_write_region_ends_at_the_confirmation_line() {
        let output = format!(
            "{PATH_WRITE_THROUGH_POWERSHELL}Tip: `setx` is not used for PATH; see the note above.\n"
        );

        let region = user_path_write_region(&output).expect("both anchors are present");

        assert!(region.starts_with("Plan: prepend C:\\ROCm\\bin"));
        assert!(region.ends_with("before re-running hipInfo."), "{region}");
        assert!(!region.contains("setx "), "{region}");
    }

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
