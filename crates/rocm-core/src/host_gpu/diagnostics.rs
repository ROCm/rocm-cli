// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The text `rocm install sdk` prints when it cannot resolve a GPU: what the
//! detection saw, and on Windows the raw output of each inventory query.

#[cfg(windows)]
use super::capture::tool_path_candidates;
use super::detect_host_gpu_summary;
use super::linux::{detect_linux_primary_gpu_name, detect_linux_sysfs_gfx_target};
#[cfg(windows)]
use super::windows_inventory::{
    WINDOWS_INVENTORY_QUERY_TIMEOUT, WINDOWS_PNP_ENTITY_INVENTORY_SCRIPT,
    WINDOWS_VIDEO_CONTROLLER_INVENTORY_SCRIPT, WindowsExamineInventory,
    parse_windows_examine_inventory, parse_windows_pnputil_display_inventory,
};
use super::wsl::{detect_wsl_windows_display_probe_text, is_wsl_host};
#[cfg(windows)]
use crate::gfx_target::{
    amd_pci_device_id_from_pnp_id, gfx_target_from_amd_marketing_name,
    gfx_target_from_amd_pci_device_id,
};
use crate::{runtime_is_linux, runtime_is_windows, runtime_os_name};
#[cfg(windows)]
use std::io::Read;
#[cfg(windows)]
use std::path::PathBuf;
#[cfg(windows)]
use std::process::{Command, Stdio};
#[cfg(windows)]
use std::thread;
#[cfg(windows)]
use std::time::{Duration, Instant};

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
#[cfg(windows)]
struct DiagnosticCommandResult {
    program: Option<PathBuf>,
    status: Option<String>,
    stdout: String,
    stderr: String,
    error: Option<String>,
    timed_out: bool,
}

#[cfg(windows)]
fn capture_diagnostic_command(
    program: &str,
    args: &[&str],
    timeout: Duration,
) -> DiagnosticCommandResult {
    let candidates = tool_path_candidates(program);
    let mut last_error = None;
    for candidate in candidates {
        let path = PathBuf::from(&candidate);
        let mut child = match Command::new(&path)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(error) => {
                last_error = Some(format!("failed to launch {}: {error}", path.display()));
                continue;
            }
        };
        let stdout_reader = child.stdout.take().map(|mut stdout| {
            thread::spawn(move || {
                let mut bytes = Vec::new();
                let _ = stdout.read_to_end(&mut bytes);
                bytes
            })
        });
        let stderr_reader = child.stderr.take().map(|mut stderr| {
            thread::spawn(move || {
                let mut bytes = Vec::new();
                let _ = stderr.read_to_end(&mut bytes);
                bytes
            })
        });

        let start = Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let stdout = stdout_reader
                        .map(|reader| reader.join().unwrap_or_default())
                        .unwrap_or_default();
                    let stderr = stderr_reader
                        .map(|reader| reader.join().unwrap_or_default())
                        .unwrap_or_default();
                    return DiagnosticCommandResult {
                        program: Some(path),
                        status: Some(status.to_string()),
                        stdout: String::from_utf8_lossy(&stdout).into_owned(),
                        stderr: String::from_utf8_lossy(&stderr).into_owned(),
                        error: None,
                        timed_out: false,
                    };
                }
                Ok(None) if start.elapsed() < timeout => {
                    thread::sleep(Duration::from_millis(25));
                }
                Ok(None) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    let stdout = stdout_reader
                        .map(|reader| reader.join().unwrap_or_default())
                        .unwrap_or_default();
                    let stderr = stderr_reader
                        .map(|reader| reader.join().unwrap_or_default())
                        .unwrap_or_default();
                    return DiagnosticCommandResult {
                        program: Some(path),
                        status: None,
                        stdout: String::from_utf8_lossy(&stdout).into_owned(),
                        stderr: String::from_utf8_lossy(&stderr).into_owned(),
                        error: None,
                        timed_out: true,
                    };
                }
                Err(error) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return DiagnosticCommandResult {
                        program: Some(path),
                        status: None,
                        stdout: String::new(),
                        stderr: String::new(),
                        error: Some(format!("failed to wait: {error}")),
                        timed_out: false,
                    };
                }
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
