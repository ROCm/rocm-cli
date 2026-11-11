// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The text `rocm install sdk` prints when it cannot resolve a GPU: what the
//! detection saw, and on Windows the raw output of each inventory query.

use super::detect_host_gpu_summary;
use super::linux::{detect_linux_primary_gpu_name, detect_linux_sysfs_gfx_target};
#[cfg(windows)]
use super::windows_inventory::{
    WINDOWS_INVENTORY_QUERY_TIMEOUT, WINDOWS_PNP_ENTITY_INVENTORY_SCRIPT,
    WINDOWS_VIDEO_CONTROLLER_INVENTORY_SCRIPT, WindowsExamineInventory,
    parse_windows_examine_inventory, parse_windows_pnputil_display_inventory,
};
use super::wsl::{detect_wsl_windows_display_probe_text, is_wsl_host};
#[cfg(any(windows, test))]
use crate::command_capture::{CommandCapture, CommandOutcome};
#[cfg(windows)]
use crate::gfx_target::{
    amd_pci_device_id_from_pnp_id, gfx_target_from_amd_marketing_name,
    gfx_target_from_amd_pci_device_id,
};
#[cfg(windows)]
use crate::probe_exec::tool_path_candidates;
#[cfg(windows)]
use crate::process::run_bounded;
use crate::{runtime_is_linux, runtime_is_windows, runtime_os_name};
#[cfg(any(windows, test))]
use std::path::{Path, PathBuf};
#[cfg(windows)]
use std::process::Command;
#[cfg(windows)]
use std::time::Duration;

pub fn detect_host_gpu_diagnostics() -> String {
    use std::fmt::Write as _;
    let mut output = String::new();
    let _ = writeln!(output, "GPU detection diagnostics");
    let _ = writeln!(output, "  runtime_os: {}", runtime_os_name());
    let summary = detect_host_gpu_summary(None);
    let _ = writeln!(
        output,
        "  detected_name: {}",
        summary.name.as_deref().unwrap_or("<unknown>")
    );
    let _ = writeln!(
        output,
        "  detected_gfx_target: {}",
        summary.gfx_target.as_deref().unwrap_or("<unknown>")
    );
    let _ = writeln!(
        output,
        "  detected_therock_family: {}",
        summary.therock_family.as_deref().unwrap_or("<unknown>")
    );

    if runtime_is_windows() {
        append_windows_gpu_probe_diagnostics(&mut output);
    } else if runtime_is_linux() {
        let _ = writeln!(
            output,
            "  linux_sysfs_gfx_target: {}",
            detect_linux_sysfs_gfx_target()
                .as_deref()
                .unwrap_or("<not found>")
        );
        let _ = writeln!(
            output,
            "  linux_primary_gpu_name: {}",
            detect_linux_primary_gpu_name()
                .as_deref()
                .unwrap_or("<not found>")
        );
        if is_wsl_host() {
            let wsl_probe = detect_wsl_windows_display_probe_text().unwrap_or_default();
            let _ = writeln!(
                output,
                "  wsl_windows_display_probe_lines: {}",
                wsl_probe.lines().count()
            );
            for line in wsl_probe.lines().take(8) {
                let _ = writeln!(output, "    {line}");
            }
        }
    }

    output
}

#[cfg(windows)]
fn append_windows_gpu_probe_diagnostics(output: &mut String) {
    append_windows_probe_diagnostics(
        output,
        "pnputil display devices",
        "pnputil",
        &["/enum-devices", "/class", "Display"],
        WINDOWS_INVENTORY_QUERY_TIMEOUT,
        parse_windows_pnputil_display_inventory,
    );
    append_windows_probe_diagnostics(
        output,
        "Win32_VideoController",
        "powershell",
        &[
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            WINDOWS_VIDEO_CONTROLLER_INVENTORY_SCRIPT,
        ],
        WINDOWS_INVENTORY_QUERY_TIMEOUT,
        parse_windows_examine_inventory,
    );
    append_windows_probe_diagnostics(
        output,
        "Win32_PnPEntity",
        "powershell",
        &[
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            WINDOWS_PNP_ENTITY_INVENTORY_SCRIPT,
        ],
        WINDOWS_INVENTORY_QUERY_TIMEOUT,
        parse_windows_examine_inventory,
    );
}

#[cfg(not(windows))]
const fn append_windows_gpu_probe_diagnostics(_output: &mut String) {}

#[cfg(windows)]
fn append_windows_probe_diagnostics(
    output: &mut String,
    label: &str,
    program: &str,
    args: &[&str],
    timeout: Duration,
    parse: fn(&str) -> WindowsExamineInventory,
) {
    use std::fmt::Write as _;
    let result = capture_diagnostic_command(program, args, timeout);
    let _ = writeln!(output, "  probe: {label}");
    let _ = writeln!(
        output,
        "    command: {} {}",
        result
            .program
            .as_ref()
            .map_or_else(|| program.to_owned(), |path| path.display().to_string()),
        args.join(" ")
    );
    if let Some(error) = result.error.as_deref() {
        let _ = writeln!(output, "    error: {error}");
    }
    if result.timed_out {
        let _ = writeln!(output, "    error: timed out");
    }
    if let Some(status) = result.status.as_deref() {
        let _ = writeln!(output, "    status: {status}");
    }

    let inventory = parse(&result.stdout);
    let _ = writeln!(output, "    display_count: {}", inventory.displays.len());
    for display in inventory.displays.iter().take(8) {
        let gfx = display
            .pnp_device_id
            .as_deref()
            .and_then(amd_pci_device_id_from_pnp_id)
            .and_then(|device_id| gfx_target_from_amd_pci_device_id(&device_id).map(str::to_owned))
            .or_else(|| gfx_target_from_amd_marketing_name(&display.name).map(str::to_owned))
            .unwrap_or_else(|| "<unknown>".to_owned());
        let _ = writeln!(
            output,
            "      gpu: name={} pnp={} driver={} gfx={}",
            crate::placeholder::trimmed_or_unknown(&display.name),
            display.pnp_device_id.as_deref().unwrap_or("<unknown>"),
            display.driver_version.as_deref().unwrap_or("<unknown>"),
            gfx
        );
    }
    append_diagnostic_stream(output, "stdout", &result.stdout);
    append_diagnostic_stream(output, "stderr", &result.stderr);
}

#[derive(Debug)]
#[cfg(any(windows, test))]
struct DiagnosticCommandResult {
    program: Option<PathBuf>,
    status: Option<String>,
    stdout: String,
    stderr: String,
    error: Option<String>,
    timed_out: bool,
}

/// Run `program` under every spelling [`tool_path_candidates`] gives, through the
/// executor the other host probes share ([`run_bounded`]), and describe the
/// first run that started.
#[cfg(windows)]
fn capture_diagnostic_command(
    program: &str,
    args: &[&str],
    timeout: Duration,
) -> DiagnosticCommandResult {
    capture_diagnostic_with(program, &tool_path_candidates(program), |path| {
        let mut command = Command::new(path);
        command.args(args);
        run_bounded(command, timeout, "process")
    })
}

/// [`capture_diagnostic_command`] with the candidates and the runner passed in,
/// so a test on any host can say how each run ends.
///
/// The next spelling is tried only when a program could not be started. Any run
/// that started is final: an exit (non-zero included) reports its status and
/// both streams, a timeout reports the output that arrived and says so, and a
/// failure to wait reports that, with no output.
#[cfg(any(windows, test))]
fn capture_diagnostic_with(
    program: &str,
    candidates: &[String],
    mut run: impl FnMut(&Path) -> CommandCapture,
) -> DiagnosticCommandResult {
    let mut last_error = None;
    for candidate in candidates {
        let path = PathBuf::from(candidate);
        let capture = run(&path);
        match capture.outcome {
            CommandOutcome::NotFound | CommandOutcome::SpawnFailed(_) => {
                last_error = Some(format!(
                    "failed to launch {}: {}",
                    path.display(),
                    capture.detail.as_deref().unwrap_or("unknown error")
                ));
            }
            CommandOutcome::Exited(_) | CommandOutcome::Signaled(_) => {
                return DiagnosticCommandResult {
                    program: Some(path),
                    status: capture.status.map(|status| status.to_string()),
                    stdout: capture.stdout,
                    stderr: capture.stderr,
                    error: None,
                    timed_out: false,
                };
            }
            CommandOutcome::TimedOut => {
                return DiagnosticCommandResult {
                    program: Some(path),
                    status: None,
                    stdout: capture.stdout,
                    stderr: capture.stderr,
                    error: None,
                    timed_out: true,
                };
            }
            CommandOutcome::WaitFailed => {
                return DiagnosticCommandResult {
                    program: Some(path),
                    status: None,
                    stdout: String::new(),
                    stderr: String::new(),
                    error: Some(format!(
                        "failed to wait: {}",
                        capture.detail.as_deref().unwrap_or("unknown error")
                    )),
                    timed_out: false,
                };
            }
        }
    }

    DiagnosticCommandResult {
        program: None,
        status: None,
        stdout: String::new(),
        stderr: String::new(),
        error: last_error.or_else(|| Some(format!("{program} was not found"))),
        timed_out: false,
    }
}

#[cfg(windows)]
fn append_diagnostic_stream(output: &mut String, name: &str, text: &str) {
    use std::fmt::Write as _;
    let mut lines = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .peekable();
    if lines.peek().is_none() {
        return;
    }
    let _ = writeln!(output, "    {name}:");
    for line in lines.take(12) {
        let _ = writeln!(
            output,
            "      {}",
            truncate_diagnostic_line(line.trim(), 220)
        );
    }
}

#[cfg(windows)]
fn truncate_diagnostic_line(line: &str, max_chars: usize) -> String {
    let mut chars = line.chars();
    let truncated = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() {
        format!("{truncated}...")
    } else {
        truncated
    }
}

/// The Windows-only diagnostic capture's choice of candidate and its report of
/// each way a run can end, driven through an injected runner so it runs on every
/// host, the Windows job included.
#[cfg(test)]
mod capture_diagnostic_tests {
    use super::{CommandCapture, CommandOutcome, capture_diagnostic_with};
    use std::path::Path;
    use std::process::ExitStatus;

    fn status(code: i32) -> ExitStatus {
        #[cfg(unix)]
        use std::os::unix::process::ExitStatusExt as _;
        #[cfg(windows)]
        use std::os::windows::process::ExitStatusExt as _;
        #[cfg(unix)]
        let raw = code << 8;
        #[cfg(windows)]
        let raw = code.unsigned_abs();
        ExitStatus::from_raw(raw)
    }

    fn candidates(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_owned()).collect()
    }

    fn finished(
        outcome: CommandOutcome,
        code: Option<i32>,
        detail: Option<&str>,
    ) -> CommandCapture {
        CommandCapture::from_run(
            vec!["tool".to_owned()],
            outcome,
            code.map(status),
            b"out\n".to_vec(),
            b"err\n".to_vec(),
            detail.map(str::to_owned),
        )
    }

    /// A runner that gives each candidate the next scripted capture, and records
    /// which candidates it was asked to start.
    fn scripted(
        mut script: Vec<CommandCapture>,
        asked: &mut Vec<String>,
    ) -> impl FnMut(&Path) -> CommandCapture + '_ {
        script.reverse();
        move |path| {
            asked.push(path.display().to_string());
            script
                .pop()
                .expect("more candidates were started than scripted")
        }
    }

    #[test]
    fn the_next_spelling_is_tried_only_when_a_program_could_not_be_started() {
        let mut asked = Vec::new();
        let result = capture_diagnostic_with(
            "tool",
            &candidates(&["a", "b", "c"]),
            scripted(
                vec![
                    finished(CommandOutcome::NotFound, None, Some("no a")),
                    finished(
                        CommandOutcome::SpawnFailed(std::io::ErrorKind::PermissionDenied),
                        None,
                        Some("no b"),
                    ),
                    finished(CommandOutcome::Exited(0), Some(0), None),
                ],
                &mut asked,
            ),
        );

        assert_eq!(asked, ["a", "b", "c"]);
        assert_eq!(result.program.as_deref(), Some(Path::new("c")));
        assert_eq!(result.status, Some(status(0).to_string()));
        assert_eq!(
            (result.stdout.as_str(), result.stderr.as_str()),
            ("out\n", "err\n")
        );
        assert!(result.error.is_none() && !result.timed_out);
    }

    #[test]
    fn a_program_that_never_starts_reports_the_last_launch_error() {
        let mut asked = Vec::new();
        let result = capture_diagnostic_with(
            "tool",
            &candidates(&["a", "b"]),
            scripted(
                vec![
                    finished(CommandOutcome::NotFound, None, Some("no a")),
                    finished(CommandOutcome::NotFound, None, Some("no b")),
                ],
                &mut asked,
            ),
        );

        assert!(result.program.is_none() && result.status.is_none());
        assert_eq!(result.error.as_deref(), Some("failed to launch b: no b"));
        assert!(result.stdout.is_empty() && !result.timed_out);

        let none = capture_diagnostic_with("tool", &[], |_| unreachable!("no candidate"));
        assert_eq!(none.error.as_deref(), Some("tool was not found"));
    }

    #[test]
    fn a_non_zero_exit_is_final_and_reports_its_status_and_both_streams() {
        let mut asked = Vec::new();
        let result = capture_diagnostic_with(
            "tool",
            &candidates(&["a", "b"]),
            scripted(
                vec![finished(CommandOutcome::Exited(3), Some(3), None)],
                &mut asked,
            ),
        );

        assert_eq!(asked, ["a"], "a run that started must not be retried");
        assert_eq!(result.program.as_deref(), Some(Path::new("a")));
        assert_eq!(result.status, Some(status(3).to_string()));
        #[cfg(windows)]
        assert_eq!(result.status.as_deref(), Some("exit code: 3"));
        assert_eq!(
            (result.stdout.as_str(), result.stderr.as_str()),
            ("out\n", "err\n")
        );
        assert!(result.error.is_none() && !result.timed_out);
    }

    #[test]
    fn a_timeout_is_final_keeps_the_output_that_arrived_and_says_so() {
        let mut asked = Vec::new();
        let result = capture_diagnostic_with(
            "tool",
            &candidates(&["a", "b"]),
            scripted(
                vec![finished(CommandOutcome::TimedOut, Some(1), None)],
                &mut asked,
            ),
        );

        assert_eq!(asked, ["a"], "a hung program must not be started again");
        assert!(result.timed_out);
        assert!(
            result.status.is_none(),
            "a killed child has no status to report"
        );
        assert_eq!(
            (result.stdout.as_str(), result.stderr.as_str()),
            ("out\n", "err\n")
        );
        assert!(result.error.is_none());
    }

    #[test]
    fn a_failure_to_wait_is_final_and_reports_no_output() {
        let mut asked = Vec::new();
        let result = capture_diagnostic_with(
            "tool",
            &candidates(&["a", "b"]),
            scripted(
                vec![finished(
                    CommandOutcome::WaitFailed,
                    None,
                    Some("the os said no"),
                )],
                &mut asked,
            ),
        );

        assert_eq!(asked, ["a"]);
        assert_eq!(result.program.as_deref(), Some(Path::new("a")));
        assert_eq!(
            result.error.as_deref(),
            Some("failed to wait: the os said no")
        );
        assert!(result.stdout.is_empty() && result.stderr.is_empty());
        assert!(!result.timed_out && result.status.is_none());
    }
}
