// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! GPU/host hardware detection.
//!
//! `ExamineSummary`'s `HostGpuSummary`/`DriverSummary`/`WslSummary` aggregation,
//! Windows WMI/pnputil inventory parsing, kernel/distro/CPU/RAM detection, and
//! gfx-target detection (PCI-device-ID, marketing-name, KFD, DRM-ip-discovery,
//! and sysfs). Also holds the TheRock-managed-SDK gfx-target probe
//! (`detect_managed_therock_sdk_gfx_target`) rather than `managed_runtime`,
//! since its only caller is this module's own gfx-target fallback chain.
//!
//! `examine.rs` and `fix.rs` still reach several items here through `lib.rs`'s
//! flat `crate::` re-export paths rather than `crate::host_gpu::` — a
//! deliberate transitional step from the zero-diff extraction, not an
//! oversight.

use crate::{
    AppPaths, LegacyRocmSummary, TheRockFamilyManifest, detect_legacy_rocm_summary,
    detect_managed_therock_family, discover_rocm_installs, env_flag, examine,
    managed_sdk_tool_path, runtime_is_linux, runtime_is_windows, runtime_os_name, unix_time_millis,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::fs;
use std::io::{IsTerminal, Read, stdin, stdout};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const OPTIONAL_COMMAND_TIMEOUT: Duration = Duration::from_millis(1_500);
const WINDOWS_INVENTORY_QUERY_TIMEOUT: Duration = Duration::from_secs(5);
const WINDOWS_VIDEO_CONTROLLER_INVENTORY_SCRIPT: &str = r#"$gpus = Get-CimInstance -ClassName Win32_VideoController -Property Name,DriverVersion,PNPDeviceID,AdapterCompatibility | Where-Object { $_.PNPDeviceID -match 'VEN_1002' -or $_.AdapterCompatibility -match 'AMD|Advanced Micro Devices' -or $_.Name -match 'AMD|Radeon|Instinct' }; foreach ($gpu in $gpus) { "GPU`t$($gpu.Name)`t$($gpu.DriverVersion)`t$($gpu.PNPDeviceID)" }"#;
#[cfg(windows)]
const WINDOWS_PNP_ENTITY_INVENTORY_SCRIPT: &str = r#"$displayGuid = '{4d36e968-e325-11ce-bfc1-08002be10318}'; $gpus = Get-CimInstance -ClassName Win32_PnPEntity -Property Name,DeviceID,PNPClass,ClassGuid,Manufacturer | Where-Object { (($_.PNPClass -eq 'Display' -or $_.ClassGuid -eq $displayGuid) -and ($_.DeviceID -match 'VEN_1002' -or $_.Name -match 'AMD|Radeon|Instinct|Graphics' -or $_.Manufacturer -match 'AMD|Advanced Micro Devices')) -or ($_.DeviceID -match 'PCI\\VEN_1002' -and $_.Name -match 'Radeon|Instinct|Graphics') }; foreach ($gpu in $gpus) { "GPU`t$($gpu.Name)`t`t$($gpu.DeviceID)" }"#;
#[cfg(windows)]
const WINDOWS_SYSTEM_INVENTORY_SCRIPT: &str = r#"$cpu = Get-CimInstance -ClassName Win32_Processor -Property Name | Select-Object -First 1 -ExpandProperty Name; if ($cpu) { "CPU`t$cpu" }; $ram = Get-CimInstance -ClassName Win32_ComputerSystem -Property TotalPhysicalMemory | Select-Object -First 1 -ExpandProperty TotalPhysicalMemory; if ($ram) { "RAM`t$ram" }"#;

pub fn detect_host_gfx_target() -> Option<String> {
    let paths = AppPaths::discover().ok();
    detect_host_gpu_summary_fast(paths.as_ref()).gfx_target
}

fn detect_examine_gfx_target_fast(
    windows_inventory: Option<&WindowsExamineInventory>,
) -> Option<String> {
    if runtime_is_windows() {
        return detect_windows_display_gfx_target_with_inventory(windows_inventory);
    }

    if runtime_is_linux() {
        return detect_linux_sysfs_gfx_target().or_else(detect_wsl_windows_display_gfx_target_fast);
    }

    None
}

#[allow(dead_code)]
fn detect_host_gfx_target_with_context(
    windows_inventory: Option<&WindowsExamineInventory>,
    wsl: Option<&WslSummary>,
    paths: Option<&AppPaths>,
) -> Option<String> {
    if runtime_is_windows() {
        return detect_windows_display_gfx_target_with_inventory(windows_inventory)
            .or_else(|| {
                capture_optional_command("rocm_agent_enumerator", &[])
                    .and_then(|output| extract_first_gfx_token(&output))
            })
            .or_else(|| {
                capture_optional_command("rocminfo", &[])
                    .and_then(|output| extract_first_gfx_token(&output))
            });
    }

    detect_linux_sysfs_gfx_target()
        .or_else(|| {
            capture_optional_command("rocm_agent_enumerator", &[])
                .and_then(|output| extract_first_gfx_token(&output))
        })
        .or_else(|| {
            capture_optional_command("rocminfo", &[])
                .and_then(|output| extract_first_gfx_token(&output))
        })
        .or_else(|| paths.and_then(detect_managed_therock_sdk_gfx_target))
        .or_else(|| detect_wsl_windows_display_gfx_target(wsl))
        .or_else(|| detect_windows_display_gfx_target_with_inventory(windows_inventory))
}

pub fn extract_first_gfx_token(text: &str) -> Option<String> {
    text.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '-' || ch == '_'))
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .find_map(|token| {
            let normalized = token.to_ascii_lowercase();
            if normalized.starts_with("gfx") {
                Some(normalized)
            } else {
                None
            }
        })
}

pub fn normalize_therock_family(value: &str) -> Option<String> {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }

    let target = extract_first_gfx_token(&normalized).unwrap_or(normalized);
    match target.as_str() {
        "gfx101x-dgpu" => Some("gfx101X-dgpu".to_owned()),
        "gfx103x-dgpu" => Some("gfx103X-dgpu".to_owned()),
        "gfx110x-all" => Some("gfx110X-all".to_owned()),
        "gfx120x-all" => Some("gfx120X-all".to_owned()),
        "gfx90x-dgpu" => Some("gfx90X-dgpu".to_owned()),
        "gfx94x-dcgpu" => Some("gfx94X-dcgpu".to_owned()),
        "gfx950-dcgpu" => Some("gfx950-dcgpu".to_owned()),
        value if value.starts_with("gfx101") => Some("gfx101X-dgpu".to_owned()),
        value if value.starts_with("gfx103") => Some("gfx103X-dgpu".to_owned()),
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103" => Some("gfx110X-all".to_owned()),
        value if value.starts_with("gfx1150") => Some("gfx1150".to_owned()),
        value if value.starts_with("gfx1151") => Some("gfx1151".to_owned()),
        value if value.starts_with("gfx1152") => Some("gfx1152".to_owned()),
        value if value.starts_with("gfx1153") => Some("gfx1153".to_owned()),
        "gfx1200" | "gfx1201" => Some("gfx120X-all".to_owned()),
        value if value.starts_with("gfx125") => Some("gfx125X-dcgpu".to_owned()),
        value if value.starts_with("gfx900") => Some("gfx900".to_owned()),
        value if value.starts_with("gfx906") => Some("gfx906".to_owned()),
        value if value.starts_with("gfx908") => Some("gfx908".to_owned()),
        value if value.starts_with("gfx90a") => Some("gfx90a".to_owned()),
        value if value.starts_with("gfx950") => Some("gfx950-dcgpu".to_owned()),
        value
            if value.starts_with("gfx942")
                || value.starts_with("gfx94")
                || value.starts_with("gfx9-4") =>
        {
            Some("gfx94X-dcgpu".to_owned())
        }
        value if value.starts_with("gfx90") => Some("gfx90X-dcgpu".to_owned()),
        _ => None,
    }
}

/// The TheRock package families the CLI recognizes.
///
/// This is the full set of values [`normalize_therock_family`] can produce. Used
/// to tell the user which `--family` values are valid when GPU auto-detection
/// cannot resolve an installable runtime. Whether a given family currently has
/// published wheels depends on the channel and release; recognition here does
/// not guarantee availability.
///
/// Kept in sync with [`normalize_therock_family`] by
/// `known_therock_families_all_round_trip` — every entry must normalize back to
/// itself.
pub const fn known_therock_families() -> &'static [&'static str] {
    &[
        "gfx90X-dgpu",
        "gfx90X-dcgpu",
        "gfx900",
        "gfx906",
        "gfx908",
        "gfx90a",
        "gfx94X-dcgpu",
        "gfx950-dcgpu",
        "gfx101X-dgpu",
        "gfx103X-dgpu",
        "gfx110X-all",
        "gfx1150",
        "gfx1151",
        "gfx1152",
        "gfx1153",
        "gfx120X-all",
        "gfx125X-dcgpu",
    ]
}

fn capture_optional_command(program: &str, args: &[&str]) -> Option<String> {
    capture_optional_command_with_timeout(program, args, OPTIONAL_COMMAND_TIMEOUT)
}

fn capture_optional_command_with_timeout(
    program: &str,
    args: &[&str],
    timeout: Duration,
) -> Option<String> {
    for candidate in tool_path_candidates(program) {
        if let Some(output) =
            capture_optional_command_candidate_with_timeout(Path::new(&candidate), args, timeout)
        {
            return Some(output);
        }
    }
    None
}

fn capture_optional_command_candidate_with_timeout(
    program: &Path,
    args: &[&str],
    timeout: Duration,
) -> Option<String> {
    let mut child = match Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            debug_command_capture_failure(program, "spawn", &error.to_string());
            return None;
        }
    };
    let mut stdout_reader = child.stdout.take().map(|mut stdout| {
        thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = stdout.read_to_end(&mut bytes);
            bytes
        })
    });

    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let bytes = stdout_reader
                    .take()
                    .map(|reader| reader.join().unwrap_or_default())
                    .unwrap_or_default();
                if status.success() {
                    return String::from_utf8(bytes).ok();
                }
                debug_command_capture_failure(program, "exit", &format!("status {status}"));
                return None;
            }
            Ok(None) if start.elapsed() < timeout => {
                thread::sleep(Duration::from_millis(25));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                if let Some(reader) = stdout_reader.take() {
                    let _ = reader.join();
                }
                debug_command_capture_failure(program, "timeout", "timed out");
                return None;
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                if let Some(reader) = stdout_reader.take() {
                    let _ = reader.join();
                }
                debug_command_capture_failure(program, "wait", "failed to wait");
                return None;
            }
        }
    }
}

fn debug_command_capture_failure(program: &Path, stage: &str, detail: &str) {
    if !env_flag("ROCM_CLI_DEBUG_COMMAND_CAPTURE") {
        return;
    }
    eprintln!(
        "rocm debug: command capture {stage} failed for {}: {detail}",
        program.display()
    );
}

fn capture_optional_path_command_with_env(
    program: &Path,
    args: &[&str],
    envs: &[(&str, OsString)],
    timeout: Duration,
) -> Option<String> {
    let output_path = std::env::temp_dir().join(format!(
        "rocm-cli-command-{}-{}.out",
        std::process::id(),
        unix_time_millis()
    ));
    let output_file = fs::File::create(&output_path).ok()?;
    let mut command = Command::new(program);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(output_file))
        .stderr(Stdio::null());
    for (key, value) in envs {
        command.env(key, value);
    }
    let Ok(mut child) = command.spawn() else {
        let _ = fs::remove_file(&output_path);
        return None;
    };

    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let bytes = if status.success() {
                    fs::read(&output_path).ok()
                } else {
                    None
                };
                let _ = fs::remove_file(&output_path);
                return bytes.and_then(|bytes| String::from_utf8(bytes).ok());
            }
            Ok(None) if start.elapsed() < timeout => {
                thread::sleep(Duration::from_millis(25));
            }
            Ok(None) | Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = fs::remove_file(&output_path);
                return None;
            }
        }
    }
}

fn tool_on_path(program: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|path| {
        std::env::split_paths(&path).any(|dir| {
            tool_path_candidates(program)
                .into_iter()
                .any(|name| dir.join(name).is_file())
        })
    })
}

fn tool_path_candidates(program: &str) -> Vec<String> {
    let path = Path::new(program);
    if path.extension().is_some() || !runtime_is_windows() {
        return vec![program.to_owned()];
    }
    let mut names = Vec::new();
    names.push(program.to_owned());
    let pathext = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_owned());
    for ext in pathext
        .split(';')
        .map(str::trim)
        .filter(|ext| !ext.is_empty())
    {
        names.push(format!("{program}{ext}"));
        names.push(format!("{program}{}", ext.to_ascii_lowercase()));
    }
    names.extend(windows_absolute_tool_candidates(program));
    names.sort();
    names.dedup();
    names
}

fn windows_absolute_tool_candidates(program: &str) -> Vec<String> {
    if !runtime_is_windows() {
        return Vec::new();
    }
    let program = program.trim().to_ascii_lowercase();
    let system_root = std::env::var("SystemRoot")
        .or_else(|_| std::env::var("WINDIR"))
        .unwrap_or_else(|_| r"C:\Windows".to_owned());
    match program.as_str() {
        "cmd" | "cmd.exe" => vec![format!(r"{system_root}\System32\cmd.exe")],
        "pnputil" | "pnputil.exe" => vec![format!(r"{system_root}\System32\pnputil.exe")],
        "powershell" | "powershell.exe" => vec![
            format!(r"{system_root}\System32\WindowsPowerShell\v1.0\powershell.exe"),
            "powershell.exe".to_owned(),
        ],
        "pwsh" | "pwsh.exe" => vec!["pwsh.exe".to_owned()],
        _ => Vec::new(),
    }
}

#[cfg(windows)]
fn detect_windows_display_gfx_target() -> Option<String> {
    if !runtime_is_windows() {
        return None;
    }
    capture_optional_command_with_timeout(
        "powershell",
        &[
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            WINDOWS_VIDEO_CONTROLLER_INVENTORY_SCRIPT,
        ],
        WINDOWS_INVENTORY_QUERY_TIMEOUT,
    )
    .map(|output| parse_windows_examine_inventory(&output).display_gfx_probe_text())
    .and_then(|output| parse_windows_display_gfx_target(&output))
}

#[cfg(not(windows))]
const fn detect_windows_display_gfx_target() -> Option<String> {
    None
}

fn detect_windows_display_gfx_target_with_inventory(
    windows_inventory: Option<&WindowsExamineInventory>,
) -> Option<String> {
    if runtime_is_windows() {
        return windows_inventory
            .and_then(WindowsExamineInventory::display_gfx_target)
            .or_else(|| {
                if windows_inventory.is_none() {
                    detect_windows_display_gfx_target()
                } else {
                    None
                }
            });
    }

    detect_windows_display_gfx_target()
}

fn detect_wsl_windows_display_gfx_target(wsl: Option<&WslSummary>) -> Option<String> {
    if !runtime_is_linux() || wsl.is_none() {
        return None;
    }

    detect_wsl_windows_display_gfx_target_fast()
}

fn detect_wsl_windows_display_gfx_target_fast() -> Option<String> {
    detect_wsl_windows_display_probe_text()
        .as_deref()
        .and_then(parse_windows_display_gfx_target)
}

fn detect_wsl_windows_display_name(wsl: Option<&WslSummary>) -> Option<String> {
    if !runtime_is_linux() || !wsl.is_some_and(|summary| summary.is_wsl) {
        return None;
    }

    detect_wsl_windows_display_name_fast()
}

fn detect_wsl_windows_display_name_fast() -> Option<String> {
    detect_wsl_windows_display_probe_text()
        .as_deref()
        .and_then(parse_windows_display_name)
}

/// What the guest was able to learn about the Windows host's AMD display driver.
///
/// Three states, not two. Reaching the host requires WSL interop, which the user
/// can switch off and which is absent entirely inside a container running on WSL.
/// Collapsing "could not ask" into "no driver found" would make the catalog blame
/// a Windows driver on every locked-down or containerised host, so the two stay
/// distinct and the check abstains on [`Unreachable`](Self::Unreachable).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WslHostDriverProbe {
    /// WSL interop is unavailable, so the host was never asked.
    Unreachable,
    /// The host answered but reported no AMD display adapter.
    NoAmdDisplay,
    /// The host's AMD display driver version.
    Version(String),
}

/// The AMD display driver of the machine this is running on.
///
/// The host-side counterpart to [`detect_wsl_host_driver`]: when `rocm` runs on
/// Windows and inspects a WSL distribution, the driver is a local question and
/// needs no interop to answer.
///
/// Returns the same tri-state, and for the same reason. An earlier version
/// collapsed it to `Option<String>` and defaulted the `None`, so "this is not
/// Windows" and "the inventory query failed" both arrived as an empty version --
/// which the catalog reads as "the host has no AMD adapter" and reports as a
/// missing driver on a machine it never managed to look at.
pub(crate) fn detect_local_windows_host_driver() -> WslHostDriverProbe {
    if !runtime_is_windows() {
        return WslHostDriverProbe::Unreachable;
    }
    let Some(inventory) = detect_windows_examine_inventory() else {
        return WslHostDriverProbe::Unreachable;
    };
    inventory
        .preferred_amd_display()
        .and_then(|display| display.driver_version.as_deref())
        .map(str::trim)
        .filter(|version| !version.is_empty())
        .map_or(WslHostDriverProbe::NoAmdDisplay, |version| {
            WslHostDriverProbe::Version(version.to_owned())
        })
}

/// Ask the Windows host, from inside the distro, which AMD display driver it runs.
pub(crate) fn detect_wsl_host_driver() -> WslHostDriverProbe {
    if !is_wsl_host() {
        return WslHostDriverProbe::Unreachable;
    }
    let Some(output) = capture_optional_command_with_timeout(
        "powershell.exe",
        &[
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            WINDOWS_VIDEO_CONTROLLER_INVENTORY_SCRIPT,
        ],
        WINDOWS_INVENTORY_QUERY_TIMEOUT,
    ) else {
        return WslHostDriverProbe::Unreachable;
    };
    parse_windows_examine_inventory(&output)
        .preferred_amd_display()
        .and_then(|display| display.driver_version.as_deref())
        .map(str::trim)
        .filter(|version| !version.is_empty())
        .map_or(WslHostDriverProbe::NoAmdDisplay, |version| {
            WslHostDriverProbe::Version(version.to_owned())
        })
}

fn detect_wsl_windows_display_probe_text() -> Option<String> {
    if !is_wsl_host() {
        return None;
    }

    capture_optional_command_with_timeout(
        "powershell.exe",
        &[
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            WINDOWS_VIDEO_CONTROLLER_INVENTORY_SCRIPT,
        ],
        WINDOWS_INVENTORY_QUERY_TIMEOUT,
    )
    .map(|output| {
        parse_windows_examine_inventory(&output)
            .display_gfx_probe_text()
            .trim()
            .to_owned()
    })
    .filter(|output| !output.is_empty())
}

#[cfg(target_os = "linux")]
fn detect_linux_primary_gpu_name() -> Option<String> {
    if !runtime_is_linux() {
        return None;
    }

    let entries = fs::read_dir("/sys/class/drm").ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let device_dir = entry.path().join("device");
        if !is_amdgpu_device(&device_dir) {
            continue;
        }
        for file_name in ["product_name", "product", "model"] {
            let Some(value) = fs::read_to_string(device_dir.join(file_name)).ok() else {
                continue;
            };
            let value = value.trim();
            if !value.is_empty() {
                return Some(value.to_owned());
            }
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
const fn detect_linux_primary_gpu_name() -> Option<String> {
    None
}

fn parse_windows_display_gfx_target(text: &str) -> Option<String> {
    let mut name_fallback = None;
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let (name, pnp_id) = line.split_once('\t').unwrap_or((line, ""));
        if let Some(device_id) = amd_pci_device_id_from_pnp_id(pnp_id)
            && let Some(target) = gfx_target_from_amd_pci_device_id(&device_id)
        {
            return Some(target.to_owned());
        }
        if name_fallback.is_none() {
            name_fallback = gfx_target_from_amd_marketing_name(name).map(str::to_owned);
        }
    }
    name_fallback
}

fn parse_windows_display_name(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .find_map(|line| {
            let (name, _) = line.split_once('\t').unwrap_or((line, ""));
            let name = name.trim();
            (!name.is_empty()).then(|| name.to_owned())
        })
}

fn amd_pci_device_id_from_pnp_id(pnp_id: &str) -> Option<String> {
    let upper = pnp_id.to_ascii_uppercase();
    if !upper.contains("VEN_1002") {
        return None;
    }
    let start = upper.find("DEV_")? + "DEV_".len();
    let device_id = upper[start..]
        .chars()
        .take_while(char::is_ascii_hexdigit)
        .take(4)
        .collect::<String>();
    if device_id.len() == 4 {
        Some(device_id.to_ascii_lowercase())
    } else {
        None
    }
}

fn gfx_target_from_amd_pci_device_id(device_id: &str) -> Option<&'static str> {
    match device_id.to_ascii_lowercase().as_str() {
        // Navi 21 / 22 / 23 / 24: Radeon RX 6000 desktop and mobile ASICs.
        "73a0" | "73a1" | "73a2" | "73a3" | "73a5" | "73a8" | "73a9" | "73ab" | "73ac" | "73ad"
        | "73ae" | "73af" => Some("gfx1030"),
        "73c0" | "73c1" | "73c3" => Some("gfx1031"),
        "73e0" | "73e1" | "73e2" | "73e3" | "73e8" | "73e9" | "73ea" | "73eb" | "73ec" | "73ed"
        | "73ef" => Some("gfx1032"),
        "7420" | "7421" | "7422" | "7423" | "7424" | "743f" => Some("gfx1034"),
        // RDNA2 APUs.
        "163f" => Some("gfx1033"),
        "164d" | "1681" => Some("gfx1035"),
        "164e" => Some("gfx1036"),
        // RDNA3 APUs.
        "15bf" | "164f" | "1900" | "1901" => Some("gfx1103"),
        // RDNA3.5 APUs with public PCI IDs that map cleanly to one gfx target.
        "1114" => Some("gfx1152"),
        // Navi 48: Radeon RX 9070 / 9070 XT / 9070 GRE.
        "7550" => Some("gfx1201"),
        _ => None,
    }
}

fn gfx_target_from_amd_marketing_name(name: &str) -> Option<&'static str> {
    let lower = name.to_ascii_lowercase();
    let normalized = normalize_marketing_name_for_match(&lower);
    for entry in AMD_MARKETING_GFX_TARGETS {
        if marketing_name_contains(&normalized, entry.pattern) {
            return Some(entry.gfx_target);
        }
    }
    None
}

#[derive(Debug, Clone, Copy)]
struct AmdMarketingGfxTarget {
    pattern: &'static str,
    gfx_target: &'static str,
}

const AMD_MARKETING_GFX_TARGETS: &[AmdMarketingGfxTarget] = &[
    // RDNA4 discrete.
    AmdMarketingGfxTarget {
        pattern: "ai pro r9700",
        gfx_target: "gfx1201",
    },
    AmdMarketingGfxTarget {
        pattern: "ai pro r9600",
        gfx_target: "gfx1201",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 9070",
        gfx_target: "gfx1201",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 9060",
        gfx_target: "gfx1200",
    },
    // RDNA3 discrete.
    AmdMarketingGfxTarget {
        pattern: "pro w7900",
        gfx_target: "gfx1100",
    },
    AmdMarketingGfxTarget {
        pattern: "pro w7800",
        gfx_target: "gfx1100",
    },
    AmdMarketingGfxTarget {
        pattern: "pro w7700",
        gfx_target: "gfx1101",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 7900",
        gfx_target: "gfx1100",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 7800",
        gfx_target: "gfx1101",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 7700",
        gfx_target: "gfx1101",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 7600",
        gfx_target: "gfx1102",
    },
    // RDNA2 discrete. Mobile names that share number prefixes are listed before desktop.
    AmdMarketingGfxTarget {
        pattern: "pro w6800",
        gfx_target: "gfx1030",
    },
    AmdMarketingGfxTarget {
        pattern: "pro w6600",
        gfx_target: "gfx1032",
    },
    AmdMarketingGfxTarget {
        pattern: "pro v620",
        gfx_target: "gfx1030",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6850m",
        gfx_target: "gfx1031",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6800m",
        gfx_target: "gfx1031",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6700m",
        gfx_target: "gfx1031",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6700s",
        gfx_target: "gfx1032",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6650m",
        gfx_target: "gfx1032",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6600m",
        gfx_target: "gfx1032",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6600s",
        gfx_target: "gfx1032",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6500m",
        gfx_target: "gfx1034",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6400m",
        gfx_target: "gfx1034",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6300m",
        gfx_target: "gfx1034",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6950",
        gfx_target: "gfx1030",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6900",
        gfx_target: "gfx1030",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6800",
        gfx_target: "gfx1030",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6750",
        gfx_target: "gfx1031",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6700",
        gfx_target: "gfx1031",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6650",
        gfx_target: "gfx1032",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6600",
        gfx_target: "gfx1032",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6500",
        gfx_target: "gfx1034",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6400",
        gfx_target: "gfx1034",
    },
    // RDNA3.5 / Strix Halo APUs.
    AmdMarketingGfxTarget {
        pattern: "8060s",
        gfx_target: "gfx1151",
    },
    AmdMarketingGfxTarget {
        pattern: "8050s",
        gfx_target: "gfx1151",
    },
    AmdMarketingGfxTarget {
        pattern: "8040s",
        gfx_target: "gfx1151",
    },
    // RDNA3.5 APUs.
    AmdMarketingGfxTarget {
        pattern: "890m",
        gfx_target: "gfx1150",
    },
    AmdMarketingGfxTarget {
        pattern: "880m",
        gfx_target: "gfx1150",
    },
    AmdMarketingGfxTarget {
        pattern: "860m",
        gfx_target: "gfx1152",
    },
    AmdMarketingGfxTarget {
        pattern: "840m",
        gfx_target: "gfx1152",
    },
    AmdMarketingGfxTarget {
        pattern: "820m",
        gfx_target: "gfx1153",
    },
    // RDNA3 APUs.
    AmdMarketingGfxTarget {
        pattern: "780m",
        gfx_target: "gfx1103",
    },
    AmdMarketingGfxTarget {
        pattern: "760m",
        gfx_target: "gfx1103",
    },
    AmdMarketingGfxTarget {
        pattern: "740m",
        gfx_target: "gfx1103",
    },
    // RDNA2 APUs.
    AmdMarketingGfxTarget {
        pattern: "680m",
        gfx_target: "gfx1035",
    },
    AmdMarketingGfxTarget {
        pattern: "660m",
        gfx_target: "gfx1035",
    },
    AmdMarketingGfxTarget {
        pattern: "610m",
        gfx_target: "gfx1036",
    },
    AmdMarketingGfxTarget {
        pattern: "steam deck",
        gfx_target: "gfx1033",
    },
    AmdMarketingGfxTarget {
        pattern: "van gogh",
        gfx_target: "gfx1033",
    },
];

fn normalize_marketing_name_for_match(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            'a'..='z' | '0'..='9' => ch,
            'A'..='Z' => ch.to_ascii_lowercase(),
            _ => ' ',
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn marketing_name_contains(normalized_name: &str, pattern: &str) -> bool {
    normalized_name
        .split_whitespace()
        .collect::<Vec<_>>()
        .windows(pattern.split_whitespace().count())
        .any(|window| window.join(" ") == pattern)
}

#[cfg(target_os = "linux")]
pub(crate) fn detect_linux_sysfs_gfx_target() -> Option<String> {
    if !runtime_is_linux() {
        return None;
    }

    detect_linux_kfd_gfx_target().or_else(detect_linux_drm_ip_discovery_gfx_target)
}

#[cfg(not(target_os = "linux"))]
pub(crate) const fn detect_linux_sysfs_gfx_target() -> Option<String> {
    None
}

#[cfg(target_os = "linux")]
fn detect_linux_kfd_gfx_target() -> Option<String> {
    detect_kfd_gfx_target_in(Path::new("/sys/class/kfd/kfd/topology/nodes"))
}

/// One GPU as the kernel's KFD topology describes it.
///
/// This is the *kernel's* answer to "which GPUs exist here", which is a
/// different question from the one the PCI bus answers. In a container the bus
/// still carries every card the host has, while KFD carries only the devices
/// passed through — so this is what `examine` must count, and the PCI scan is
/// only good for naming what it finds here.
#[cfg(any(target_os = "linux", test))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KfdGpuNode {
    /// The node's PCI address as `lspci -D` spells it (`0000:11:00.0`), or
    /// empty when the node does not state a usable one.
    pub(crate) pci_id: String,
    /// The node's own gfx target (`gfx942`), or empty when unparseable.
    ///
    /// Per-node, so it can be attributed to a particular device: an APU+dGPU
    /// host reports two different targets and each belongs to exactly one card.
    pub(crate) gfx_target: String,
}

/// Every GPU the KFD topology describes, in node order, read from a
/// caller-supplied nodes directory. `None` when the topology could not be read
/// at all, which callers must treat as "cannot say" rather than as "no GPUs".
///
/// Same planted-directory seam and cfg gating as [`detect_kfd_gfx_target_in`].
/// The `/sys` path lives in the probe that calls this
/// (`examine::probe_gpus_kernel_membership`), so that probe's own wiring is
/// drivable from a test too rather than only the reconcile it hands off to.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn kfd_gpu_nodes_in(nodes_dir: &Path) -> Option<Vec<KfdGpuNode>> {
    let mut nodes: Vec<((u64, String), KfdGpuNode)> = fs::read_dir(nodes_dir)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let version = kfd_node_gfx_target_version(&path)?;
            if !kfd_gfx_target_version_is_gpu(version.trim()) {
                return None;
            }
            let properties = fs::read_to_string(path.join("properties")).unwrap_or_default();
            Some((
                natural_node_order(&entry.file_name().to_string_lossy()),
                KfdGpuNode {
                    pci_id: kfd_node_pci_id(&properties).unwrap_or_default(),
                    gfx_target: parse_linux_kfd_gfx_target(version.trim()).unwrap_or_default(),
                },
            ))
        })
        .collect();
    // `read_dir` order is filesystem-defined, so sort on the node number for the
    // same reason `detect_kfd_gfx_target_in` does: node 0 is HIP ordinal 0.
    nodes.sort_by(|(left, _), (right, _)| left.cmp(right));
    Some(nodes.into_iter().map(|(_, node)| node).collect())
}

/// The PCI address a KFD topology node reports, spelled as `lspci -D` spells it.
///
/// KFD states the address as two decimal properties. `domain` is the PCI domain;
/// `location_id` is the kernel's `pci_dev_id()`, i.e. `(bus << 8) | devfn`, with
/// `devfn` packing the device number in bits 3..8 and the function in bits 0..3.
/// So `location_id 4352` (`0x1100`) in domain 0 is `0000:11:00.0`.
///
/// Verified against an 8-GPU MI300X host: nodes 2..9 report `location_id` 4352,
/// 12032, 17920, 23808, 35584, 43520, 49664 and 55808, which decode to exactly
/// the eight addresses `lspci -D` lists for its accelerators (EAI-8449).
///
/// `None` when the property is absent or zero. Zero is refused rather than
/// decoded: `0000:00:00.0` is the host bridge, so emitting it would be a
/// wrong-but-plausible address that could match an unrelated PCI entry.
#[cfg(any(target_os = "linux", test))]
fn kfd_node_pci_id(properties: &str) -> Option<String> {
    let location = kfd_property_value(properties, "location_id")?
        .parse::<u32>()
        .ok()?;
    if location == 0 {
        return None;
    }
    let domain = kfd_property_value(properties, "domain")
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0);
    let bus = (location >> 8) & 0xff;
    let device = (location >> 3) & 0x1f;
    let function = location & 0x7;
    Some(format!("{domain:04x}:{bus:02x}:{device:02x}.{function}"))
}

/// The KFD-topology read, against a caller-supplied nodes directory.
///
/// Split out so it can be driven against a planted directory: the hosts where
/// this matters most (an Instinct box with no `lspci`) are exactly the ones a
/// test cannot run on. Same seam as `discover_rocm_installs_in`.
///
/// Gated the same way as `parse_linux_kfd_gfx_target`, which it calls: present
/// on Linux and under `cfg(test)` everywhere, so the tests run on every platform
/// without the function existing in a Windows release build that can never use
/// it. (`target_os` alone would have left the tests Linux-only; `test` alone
/// would not compile on Windows CI, which is how this was found.)
#[cfg(any(target_os = "linux", test))]
pub(crate) fn detect_kfd_gfx_target_in(nodes_dir: &Path) -> Option<String> {
    let mut targets: Vec<(String, String)> = fs::read_dir(nodes_dir)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let value = kfd_node_gfx_target_version(&entry.path())?;
            let token = parse_linux_kfd_gfx_target(value.trim())?;
            Some((entry.file_name().to_string_lossy().into_owned(), token))
        })
        .collect();
    // `read_dir` order is filesystem-defined, so a multi-node box could report a
    // different GPU run to run. Node directories are numbered (`0`, `1`, ... on
    // a real KFD; `node0`, `node1`, ... in planted fixtures), so sorting on the
    // trailing number makes the answer stable and picks the lowest-numbered
    // node, which is the one HIP ordinal 0 refers to.
    targets.sort_by_key(|(name, _)| natural_node_order(name));
    targets.into_iter().next().map(|(_, token)| token)
}

/// Sort key for a KFD node directory name: its trailing number when it has one,
/// so `node9` precedes `node10` rather than following it lexicographically.
#[cfg(any(target_os = "linux", test))]
fn natural_node_order(name: &str) -> (u64, String) {
    let digits: String = name
        .chars()
        .skip_while(|ch| !ch.is_ascii_digit())
        .take_while(char::is_ascii_digit)
        .collect();
    (
        digits.parse::<u64>().unwrap_or(u64::MAX),
        name.to_ascii_lowercase(),
    )
}

/// AMD GPU device ordinals usable for a GPU-required launch on this host, after
/// applying the runtime visibility mask (`HIP_VISIBLE_DEVICES`, then
/// `ROCR_VISIBLE_DEVICES`).
///
/// `Some(indices)` is an authoritative answer: an empty vector means no GPU is
/// usable, so a GPU-required launch must fail rather than fall back to CPU or
/// assume device 0. `None` means availability could not be probed on this
/// platform (e.g. the KFD device exists but its topology is unreadable, or a
/// non-Linux target) and callers should not block a launch on this basis.
#[must_use]
pub fn usable_amd_gpu_indices() -> Option<Vec<u32>> {
    probe_usable_amd_gpu_indices()
}

/// Whether at least one AMD GPU is usable for a GPU-required launch.
///
/// `false` only when the probe authoritatively reports zero usable devices; an
/// unprobeable platform reports `true` so it does not block launches (see
/// [`usable_amd_gpu_indices`]).
#[must_use]
pub fn has_usable_amd_gpu() -> bool {
    usable_amd_gpu_indices().is_none_or(|indices| !indices.is_empty())
}

#[cfg(target_os = "linux")]
fn probe_usable_amd_gpu_indices() -> Option<Vec<u32>> {
    // WSL2 reaches the GPU through /dev/dxg and the Windows host driver. It has
    // no KFD topology and no amdgpu DRM card, and `linux_kfd_gpu_node_count`
    // reads "topology unreadable AND no /dev/kfd" as an authoritative zero
    // rather than as unknown -- which is exactly the WSL2 shape. So a machine
    // whose `examine` said `wsl_rocdxg_ready` was refused a GPU-required launch
    // for having none.
    //
    // Answer from the plumbing that platform actually uses, so `serve` and
    // `examine` cannot contradict each other in either direction: ready means a
    // device, not-ready means none, and both match what the report prints.
    if is_wsl_host() {
        let ready = detect_wsl_summary().is_some_and(|wsl| wsl.rocdxg_ready());
        // One device: WSL2 exposes no per-device topology to enumerate, and the
        // visibility mask still applies on top so an explicit
        // HIP_VISIBLE_DEVICES="" is honoured.
        return usable_amd_gpu_indices_from(usize::from(ready), visibility_mask_from_env());
    }
    let present =
        combine_amd_gpu_counts(linux_kfd_gpu_node_count(), linux_drm_amdgpu_card_count())?;
    usable_amd_gpu_indices_from(present, visibility_mask_from_env())
}

#[cfg(not(target_os = "linux"))]
fn probe_usable_amd_gpu_indices() -> Option<Vec<u32>> {
    None
}

/// Combine the KFD-topology and DRM-card AMD GPU counts into one "GPUs present"
/// figure. KFD counts *compute* nodes and is authoritative for HIP ordinals, so
/// it wins whenever it reports at least one GPU. DRM is used only as the
/// zero-KFD fallback, so that a host KFD does not account for is not wrongly
/// told it has no GPU and blocked from serving. The fallback was added when the
/// KFD count read a node file that no kernel exposes and so came back zero
/// everywhere; how often a correctly-read KFD still reports zero has not been
/// surveyed, so treat this as a hedge rather than a description of known
/// hardware.
///
/// DRM must not *raise* a nonzero KFD count: a display/render-only AMD DRM card
/// with no KFD compute node (e.g. KFD=1, DRM=2) would otherwise invent a usable
/// HIP ordinal that passes `--gpu` validation but fails later inside HIP.
///
/// `None` only when NEITHER surface could be read (availability truly unknown).
#[cfg(any(target_os = "linux", test))]
fn combine_amd_gpu_counts(kfd: Option<usize>, drm: Option<usize>) -> Option<usize> {
    match kfd {
        // KFD is compute-authoritative: prefer it whenever it sees a GPU.
        Some(k) if k > 0 => Some(k),
        // Zero KFD compute nodes: let DRM answer rather than block serving.
        Some(_) => Some(drm.unwrap_or(0)),
        // KFD unreadable: use DRM if it could be read, else availability unknown.
        None => drm,
    }
}

/// Count AMD (`amdgpu`) primary DRM cards under `/sys/class/drm` (`card0`,
/// `card1`, …), skipping connector sub-nodes like `card0-DP-1`. `None` when the
/// DRM class dir can't be read; `Some(0)` when it is readable with no AMD card.
#[cfg(target_os = "linux")]
fn linux_drm_amdgpu_card_count() -> Option<usize> {
    let entries = fs::read_dir(Path::new("/sys/class/drm")).ok()?;
    let count = entries
        .flatten()
        .filter(|entry| {
            let card_path = entry.path();
            let Some(name) = card_path.file_name().and_then(|value| value.to_str()) else {
                return false;
            };
            name.starts_with("card")
                && !name.contains('-')
                && is_amdgpu_device(&card_path.join("device"))
        })
        .count();
    Some(count)
}

/// Count AMD GPU nodes in the KFD topology. `Some(0)` is an authoritative "no
/// GPU" (topology readable with no GPU node, or no KFD device at all); `None`
/// means the topology could not be read even though `/dev/kfd` exists, so
/// availability is unknown and must not be treated as zero.
#[cfg(target_os = "linux")]
fn linux_kfd_gpu_node_count() -> Option<usize> {
    linux_kfd_gpu_node_count_in(Path::new("/sys/class/kfd/kfd/topology/nodes"))
}

/// The KFD GPU-node count, against a caller-supplied nodes directory.
///
/// Split out for the same reason as [`detect_kfd_gfx_target_in`]: the hosts
/// where this matters are the ones a test cannot run on. Only the readable
/// branch is driven by tests -- the unreadable branches key off the real
/// `/dev/kfd`, which a planted directory cannot stand in for.
#[cfg(any(target_os = "linux", test))]
fn linux_kfd_gpu_node_count_in(nodes_dir: &Path) -> Option<usize> {
    match fs::read_dir(nodes_dir) {
        Ok(entries) => Some(
            entries
                .flatten()
                .filter(|entry| kfd_node_is_gpu(&entry.path()))
                .count(),
        ),
        Err(_) if Path::new("/dev/kfd").exists() => None,
        Err(_) => Some(0),
    }
}

/// A KFD topology node is a GPU (not the CPU node) when its
/// `gfx_target_version` is a nonzero value; CPU nodes report `0`.
#[cfg(any(target_os = "linux", test))]
fn kfd_node_is_gpu(node_dir: &Path) -> bool {
    kfd_node_gfx_target_version(node_dir)
        .is_some_and(|value| kfd_gfx_target_version_is_gpu(value.trim()))
}

/// A KFD topology node's `gfx_target_version`, or `None` when the node does not
/// state one.
///
/// The kernel exposes it as a **line inside the node's `properties` file**
/// (`gfx_target_version 90402`), not as a standalone file. Reading only the
/// standalone path found nothing on every real KFD host, so target detection
/// silently fell through to the DRM ip-discovery route, which decodes a GC IP
/// version and is wrong-but-plausible on the GC 9.4.x line: an MI300X reported
/// `gfx943` where KFD plainly said `90402` (gfx942). Hosts whose ip-discovery
/// route also came up empty got `<unknown>` instead.
///
/// The standalone file is still read as a fallback, so any layout that does
/// expose it keeps working.
#[cfg(any(target_os = "linux", test))]
fn kfd_node_gfx_target_version(node_dir: &Path) -> Option<String> {
    let from_properties = fs::read_to_string(node_dir.join("properties"))
        .ok()
        .and_then(|text| kfd_property_value(&text, "gfx_target_version"));
    if from_properties.is_some() {
        return from_properties;
    }
    fs::read_to_string(node_dir.join("gfx_target_version"))
        .ok()
        .map(|value| value.trim().to_owned())
}

/// The value of one `key value` line in a KFD `properties` body.
#[cfg(any(target_os = "linux", test))]
fn kfd_property_value(text: &str, key: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let mut parts = line.split_whitespace();
        if parts.next()? != key {
            return None;
        }
        parts.next().map(str::to_owned)
    })
}

#[cfg(any(target_os = "linux", test))]
fn kfd_gfx_target_version_is_gpu(value: &str) -> bool {
    value
        .trim()
        .parse::<u64>()
        .is_ok_and(|version| version != 0)
}

/// The GPU visibility mask read from the environment: both variables, kept
/// separately because they apply at different layers and in a fixed order.
/// `ROCR_VISIBLE_DEVICES` masks at the ROCr level and HIP then re-indexes the
/// survivors as `0..N`; `HIP_VISIBLE_DEVICES` selects *within* that re-indexed
/// set. Collapsing the two into one "winning" value loses the composition and
/// makes HIP tokens look like physical ordinals. See
/// [`usable_amd_gpu_indices_from`].
#[cfg(any(target_os = "linux", test))]
#[derive(Debug, Clone)]
struct GpuVisibilityMask {
    /// `ROCR_VISIBLE_DEVICES`, when set. Its tokens are physical ordinals.
    rocr: Option<String>,
    /// `HIP_VISIBLE_DEVICES`, when set. Its tokens are ordinals in the space ROCr
    /// leaves behind — the same as physical ordinals only when `rocr` is unset.
    hip: Option<String>,
}

/// The active GPU visibility mask: whichever of `ROCR_VISIBLE_DEVICES` and
/// `HIP_VISIBLE_DEVICES` are set, and both when both are. `None` when neither is
/// set; an explicitly empty value is carried through as an empty string so
/// callers can distinguish "unset" (all visible) from "set to nothing" (all
/// masked out).
///
/// Linux-only: its sole caller is the Linux probe. (Not `+ test` — no test
/// references it directly, so compiling it into a non-Linux test build would be
/// dead code, which the workspace lints deny.)
#[cfg(target_os = "linux")]
fn visibility_mask_from_env() -> Option<GpuVisibilityMask> {
    let read = |key: &str| std::env::var_os(key).map(|value| value.to_string_lossy().into_owned());
    let mask = GpuVisibilityMask {
        rocr: read("ROCR_VISIBLE_DEVICES"),
        hip: read("HIP_VISIBLE_DEVICES"),
    };
    if mask.rocr.is_none() && mask.hip.is_none() {
        return None;
    }
    Some(mask)
}

/// The tokens of `mask` naming a device in `0..count`, de-duplicated and in mask
/// order. An empty mask yields an empty set (every device hidden). `None` when
/// the mask cannot be interpreted by ordinal alone — a UUID token, or an ordinal
/// outside `0..count` — so callers must not mistake it for "no GPU".
#[cfg(any(target_os = "linux", test))]
fn mask_tokens_within(count: usize, mask: &str) -> Option<Vec<u32>> {
    if mask.is_empty() {
        return Some(Vec::new());
    }
    let mut visible = Vec::new();
    for token in mask.split(',') {
        let index = token.trim().parse::<u32>().ok()?;
        if (index as usize) >= count {
            return None;
        }
        if !visible.contains(&index) {
            visible.push(index);
        }
    }
    Some(visible)
}

/// Apply the visibility `mask` to the present device ordinals (`0..present`).
/// A `None` `mask` means no mask is set (every present device is visible); a
/// `None` *return* means the mask could not be interpreted authoritatively, so
/// callers must not mistake it for "no GPU".
///
/// The returned ordinals are always in HIP space — the space rocm-cli pins its
/// selection through `HIP_VISIBLE_DEVICES`. The two variables are therefore
/// composed in the order the runtime applies them, not treated as alternatives:
///
/// 1. `ROCR_VISIBLE_DEVICES` hides physical devices *below* HIP, which then
///    re-indexes the survivors as `0..N`. Returning the raw physical tokens would
///    make `--gpu` validation reject the ordinals that actually bind and accept
///    ones that do not.
/// 2. `HIP_VISIBLE_DEVICES` then selects within that `0..N` space. Its tokens are
///    already HIP ordinals and are kept as-is — but they must be range-checked
///    against `N`, not against `present`: under an active ROCR mask a HIP token
///    can sit below the physical count and still name no device HIP can see
///    (EAI-7194). Checking it against `present` accepted a `--gpu` ordinal that
///    cannot bind, and steered `--gpu auto` onto it — the exact failure this
///    composition exists to prevent, reached through the other variable.
#[cfg(any(target_os = "linux", test))]
fn usable_amd_gpu_indices_from(
    present: usize,
    mask: Option<GpuVisibilityMask>,
) -> Option<Vec<u32>> {
    let Some(mask) = mask else {
        return Some((0..present as u32).collect());
    };
    // How many devices HIP can see at all: the ROCr survivors, or every present
    // device when no ROCR mask is set.
    let hip_space = match mask.rocr.as_deref() {
        None => present,
        Some(rocr) => {
            let survivors = mask_tokens_within(present, rocr)?;
            if survivors.is_empty() {
                // ROCr hid every device, so there is nothing for HIP to select
                // from whatever HIP_VISIBLE_DEVICES names. Authoritatively empty.
                return Some(Vec::new());
            }
            survivors.len()
        }
    };
    let Some(hip) = mask.hip.as_deref() else {
        // No HIP mask: every device HIP can see is selectable, numbered 0..N.
        return Some((0..hip_space as u32).collect());
    };
    mask_tokens_within(hip_space, hip)
}

#[cfg(any(target_os = "linux", test))]
fn parse_linux_kfd_gfx_target(value: &str) -> Option<String> {
    if let Some(token) = extract_first_gfx_token(value) {
        return Some(token);
    }
    let digits = value.trim();
    if digits.is_empty() || !digits.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    match digits.len() {
        3 | 4 => Some(format!("gfx{digits}")),
        // A 5/6-digit value is KFD's *packed* version (major·10000 + minor·100 +
        // step), not a target name, so there is no `gfx{digits}` fallback here.
        // Fabricating one from a version that failed to decode fed
        // `gfx90010`-shaped tokens into `normalize_therock_family`, where the
        // loose `gfx90` arm mapped them to a plausible-looking but wrong family.
        // Yielding `None` instead lets detection try the next KFD node and then
        // `ip_discovery`, and otherwise report the target as unknown — which is
        // recoverable with `--family`, where a wrong family silently installs
        // the wrong runtime wheel.
        5 | 6 => {
            let raw: u32 = digits.parse().ok()?;
            let major = raw / 10_000;
            let minor = (raw / 100) % 100;
            let revision = raw % 100;
            gfx_target_from_gc_version(major, minor, revision)
        }
        _ => None,
    }
}

#[cfg(target_os = "linux")]
fn detect_linux_drm_ip_discovery_gfx_target() -> Option<String> {
    let drm_dir = Path::new("/sys/class/drm");
    let entries = fs::read_dir(drm_dir).ok()?;
    for entry in entries.flatten() {
        let card_path = entry.path();
        let Some(card_name) = card_path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !card_name.starts_with("card") || card_name.contains('-') {
            continue;
        }
        let device_dir = card_path.join("device");
        if !is_amdgpu_device(&device_dir) {
            continue;
        }
        let gc_root = device_dir.join("ip_discovery");
        let token = detect_ip_discovery_gc_target(&gc_root);
        if token.is_some() {
            return token;
        }
    }
    None
}

#[cfg(any(target_os = "linux", test))]
fn is_amdgpu_device(device_dir: &Path) -> bool {
    if let Ok(vendor) = fs::read_to_string(device_dir.join("vendor"))
        && vendor.trim().eq_ignore_ascii_case("0x1002")
    {
        return true;
    }
    if let Ok(uevent) = fs::read_to_string(device_dir.join("uevent")) {
        return uevent.lines().any(|line| line.trim() == "DRIVER=amdgpu");
    }
    false
}

#[cfg(any(target_os = "linux", test))]
fn detect_ip_discovery_gc_target(ip_discovery_dir: &Path) -> Option<String> {
    let die_entries = fs::read_dir(ip_discovery_dir.join("die")).ok()?;
    for die in die_entries.flatten() {
        let Some(gc_entries) = fs::read_dir(die.path().join("GC")).ok() else {
            continue;
        };
        for gc in gc_entries.flatten() {
            let block = gc.path();
            let Some(major) = fs::read_to_string(block.join("major"))
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok())
            else {
                continue;
            };
            let Some(minor) = fs::read_to_string(block.join("minor"))
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok())
            else {
                continue;
            };
            let Some(revision) = fs::read_to_string(block.join("revision"))
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok())
            else {
                continue;
            };
            if let Some(token) = gfx_target_from_gc_version(major, minor, revision) {
                return Some(token);
            }
        }
    }
    None
}

#[cfg(any(target_os = "linux", test))]
/// Render gfx target *components* as an LLVM target token.
///
/// The token is `gfx` + major in decimal + minor and revision as **single hex
/// digits** — the `a` in `gfx90a` is revision `10`. Concatenating the components
/// as decimal agrees with hex only while every component is below 10, so it
/// silently produced `gfx9010` for gfx90a hardware (MI210/MI250), which then
/// normalized to the wrong TheRock family.
///
/// **The caller owns whether its numbers are target components at all.** KFD's
/// `gfx_target_version` packs exactly this triple, so decoding it and calling
/// here is sound. A GC (Graphics Core) IP version is a *different* quantity that
/// merely coincides with the target on many parts: it does not on the GC 9.4.x
/// line, where GC 9.4.0/9.4.1/9.4.2/9.4.3 are gfx906/gfx908/gfx90a/gfx942. So
/// [`detect_ip_discovery_gc_target`] can still yield a wrong-but-plausible token
/// for those parts — pre-existing, unchanged by the hex encoding, and not
/// something this function can detect, since the components it receives are
/// well-formed either way.
///
/// A minor or revision that cannot be a single hex digit does not describe any
/// gfx target, so it yields `None` rather than a fabricated token: the caller
/// tries the next detection source and otherwise reports the target as unknown,
/// which is recoverable with `--family`, where a fabricated one silently
/// installs the wrong runtime wheel. `major` is only checked for zero — it is
/// printed in decimal and has no single-digit bound (`gfx1030`, `gfx1250`), so
/// an implausible major still concatenates into a token.
fn gfx_target_from_gc_version(major: u32, minor: u32, revision: u32) -> Option<String> {
    if major == 0 || minor > 0xf || revision > 0xf {
        return None;
    }
    Some(format!("gfx{major}{minor:x}{revision:x}"))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExamineSummary {
    pub os: String,
    pub arch: String,
    pub kernel: Option<String>,
    pub distro: Option<String>,
    pub cpu: Option<String>,
    pub system_ram_gib: Option<f64>,
    /// Whether *this* `rocm` process was given a terminal — not a property of
    /// the machine, unlike every other field here.
    ///
    /// False whenever stdout is captured, which includes the dashboard running
    /// `rocm examine` as a child process. That is why the same machine reports
    /// `true` from a shell and `false` from the dashboard: both are correct.
    /// See [`interactive_terminal`] for what it gates.
    pub interactive_terminal: bool,
    pub default_engine: String,
    pub detected_gfx_target: Option<String>,
    #[serde(default)]
    pub compatible_therock_family: Option<String>,
    #[serde(default)]
    pub detected_therock_family: Option<String>,
    pub driver: DriverSummary,
    pub legacy_rocm: LegacyRocmSummary,
    #[serde(default)]
    pub wsl: Option<WslSummary>,
    pub managed_runtime_count: usize,
    pub managed_service_count: usize,
    pub model_cache_entries: usize,
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
    pub cache_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriverSummary {
    pub policy: String,
    pub status: String,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WslSummary {
    pub is_wsl: bool,
    pub dxg_device: bool,
    pub dxcore: bool,
    pub librocdxg: bool,
    pub rocdxg_dids: bool,
    pub ldconfig_librocdxg: bool,
    pub rocminfo: bool,
    pub cargo: bool,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct HostGpuSummary {
    pub name: Option<String>,
    pub gfx_target: Option<String>,
    pub therock_family: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct WindowsExamineInventory {
    cpu_model: Option<String>,
    system_ram_gib: Option<f64>,
    displays: Vec<WindowsDisplayAdapter>,
}

#[derive(Debug, Clone)]
struct WindowsDisplayAdapter {
    name: String,
    driver_version: Option<String>,
    pnp_device_id: Option<String>,
}

impl WindowsExamineInventory {
    #[cfg(windows)]
    fn is_empty(&self) -> bool {
        self.cpu_model.is_none() && self.system_ram_gib.is_none() && self.displays.is_empty()
    }

    #[cfg(windows)]
    fn merge_missing_from(&mut self, mut other: WindowsExamineInventory) {
        if self.cpu_model.is_none() {
            self.cpu_model = other.cpu_model.take();
        }
        if self.system_ram_gib.is_none() {
            self.system_ram_gib = other.system_ram_gib.take();
        }
        for display in other.displays {
            let duplicate = self.displays.iter_mut().find(|existing| {
                match (
                    existing.pnp_device_id.as_deref(),
                    display.pnp_device_id.as_deref(),
                ) {
                    (Some(left), Some(right)) => left.eq_ignore_ascii_case(right),
                    _ => {
                        !existing.name.trim().is_empty()
                            && !display.name.trim().is_empty()
                            && existing.name.eq_ignore_ascii_case(&display.name)
                    }
                }
            });
            if let Some(existing) = duplicate {
                if existing.name.trim().is_empty() && !display.name.trim().is_empty() {
                    existing.name = display.name;
                }
                if existing.driver_version.is_none() {
                    existing.driver_version = display.driver_version;
                }
                if existing.pnp_device_id.is_none() {
                    existing.pnp_device_id = display.pnp_device_id;
                }
            } else {
                self.displays.push(display);
            }
        }
    }

    fn amd_display_driver_detail(&self) -> Option<String> {
        let display = self.preferred_amd_display()?;
        let name = display.name.trim();
        if name.is_empty() {
            return None;
        }
        let detail = format!(
            "{name} driver {}",
            display.driver_version.as_deref().unwrap_or("")
        );
        Some(detail.trim().to_owned())
    }

    fn amd_display_name(&self) -> Option<String> {
        self.preferred_amd_display()
            .map(|display| display.name.trim())
            .filter(|name| !name.is_empty())
            .map(str::to_owned)
    }

    fn preferred_amd_display(&self) -> Option<&WindowsDisplayAdapter> {
        self.displays
            .iter()
            .find(|display| {
                display
                    .pnp_device_id
                    .as_deref()
                    .and_then(amd_pci_device_id_from_pnp_id)
                    .and_then(|device_id| gfx_target_from_amd_pci_device_id(&device_id))
                    .is_some()
            })
            .or_else(|| {
                self.displays
                    .iter()
                    .find(|display| gfx_target_from_amd_marketing_name(&display.name).is_some())
            })
            .or_else(|| {
                self.displays
                    .iter()
                    .find(|display| !display.name.trim().is_empty())
            })
    }

    fn display_gfx_target(&self) -> Option<String> {
        parse_windows_display_gfx_target(&self.display_gfx_probe_text())
    }

    fn display_gfx_probe_text(&self) -> String {
        self.displays
            .iter()
            .map(|display| {
                format!(
                    "{}\t{}",
                    display.name,
                    display.pnp_device_id.as_deref().unwrap_or("")
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl ExamineSummary {
    pub fn gather() -> Result<Self> {
        let paths = AppPaths::discover()?;
        let windows_inventory = detect_windows_examine_inventory();
        let wsl = detect_wsl_summary();
        let detected_gfx_target = detect_examine_gfx_target_fast(windows_inventory.as_ref());
        let compatible_therock_family = detected_gfx_target
            .as_deref()
            .and_then(normalize_therock_family);
        let detected_therock_family = detect_managed_therock_family(&paths);
        // Report the engine this GPU actually serves on, not the platform
        // constant. `compatible_therock_family` is the right input: it is
        // normalised from the real GPU, whereas `detected_therock_family`
        // describes the installed managed runtime and is absent before one
        // exists — which would silently downgrade the answer to the constant on
        // a fresh machine.
        let host_gpu = HostGpuSummary {
            name: None,
            gfx_target: detected_gfx_target.clone(),
            therock_family: compatible_therock_family.clone(),
        };
        Ok(Self {
            os: runtime_os_name().to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
            kernel: detect_kernel_version(),
            distro: detect_distro_name(),
            cpu: detect_cpu_model_with_windows_inventory(windows_inventory.as_ref()),
            system_ram_gib: detect_system_ram_gib_with_windows_inventory(
                windows_inventory.as_ref(),
            ),
            interactive_terminal: interactive_terminal(),
            default_engine: default_engine_for_host(&host_gpu).to_owned(),
            detected_gfx_target,
            compatible_therock_family,
            detected_therock_family,
            driver: detect_driver_summary_with_windows_inventory(
                windows_inventory.as_ref(),
                wsl.as_ref(),
            ),
            legacy_rocm: detect_legacy_rocm_summary(),
            wsl,
            managed_runtime_count: count_json_files(
                &paths.data_dir.join("runtimes").join("registry"),
            ),
            managed_service_count: count_json_files(&paths.services_dir()),
            model_cache_entries: count_dir_entries(&paths.data_dir.join("models")),
            config_dir: paths.config_dir,
            data_dir: paths.data_dir,
            cache_dir: paths.cache_dir,
        })
    }

    pub fn render_text(&self) -> String {
        let legacy_paths = if self.legacy_rocm.paths.is_empty() {
            "<none>".to_owned()
        } else {
            self.legacy_rocm
                .paths
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let wsl = self.wsl.as_ref();
        // Every other field here describes the MACHINE; this one describes the
        // invocation, which is what made it ambiguous -- `true` from a terminal
        // and `false` under the dashboard both look like claims about the host.
        // Say which it is on the line itself, so pasted output explains itself.
        let interactive_terminal = if self.interactive_terminal {
            "true (this run has a terminal; the CLI may prompt)"
        } else {
            "false (this run's output is captured, so the CLI will not prompt)"
        };
        format!(
            "rocm examine\n  os: {}\n  arch: {}\n  kernel: {}\n  distro: {}\n  cpu: {}\n  system_ram: {}\n  interactive_terminal: {}\n  default_engine: {}\n  detected_gfx_target: {}\n  compatible_therock_family: {}\n  detected_therock_family: {}\n  driver_policy: {}\n  driver_status: {}\n  driver_detail: {}\n  legacy_rocm_status: {}\n  legacy_rocm_paths: {}\n  legacy_rocm_version: {}\n  legacy_rocm_detail: {}\n  legacy_rocm_guidance: {}\n  wsl: {}\n  wsl_dxg_device: {}\n  wsl_dxcore: {}\n  wsl_librocdxg: {}\n  wsl_rocdxg_dids: {}\n  wsl_ldconfig_librocdxg: {}\n  wsl_global_rocminfo: {}\n  wsl_cargo: {}\n  wsl_detail: {}\n  managed_runtimes: {}\n  managed_services: {}\n  model_cache_entries: {}\n  config_dir: {}\n  data_dir: {}\n  cache_dir: {}\n",
            self.os,
            self.arch,
            self.kernel.as_deref().unwrap_or("<unknown>"),
            self.distro.as_deref().unwrap_or("<unknown>"),
            self.cpu.as_deref().unwrap_or("<unknown>"),
            self.system_ram_gib
                .map_or_else(|| "<unknown>".to_owned(), format_gib_value),
            interactive_terminal,
            self.default_engine,
            self.detected_gfx_target.as_deref().unwrap_or("<unknown>"),
            self.compatible_therock_family
                .as_deref()
                .unwrap_or("<unknown>"),
            self.detected_therock_family
                .as_deref()
                .unwrap_or("<not detected>"),
            self.driver.policy,
            self.driver.status,
            self.driver.detail.as_deref().unwrap_or("<unknown>"),
            self.legacy_rocm.status,
            legacy_paths,
            self.legacy_rocm.version.as_deref().unwrap_or("<unknown>"),
            self.legacy_rocm.detail.as_deref().unwrap_or("<unknown>"),
            self.legacy_rocm_guidance(),
            wsl.is_some_and(|summary| summary.is_wsl),
            wsl.is_some_and(|summary| summary.dxg_device),
            wsl.is_some_and(|summary| summary.dxcore),
            wsl.is_some_and(|summary| summary.librocdxg),
            wsl.is_some_and(|summary| summary.rocdxg_dids),
            wsl.is_some_and(|summary| summary.ldconfig_librocdxg),
            wsl.is_some_and(|summary| summary.rocminfo),
            wsl.is_some_and(|summary| summary.cargo),
            wsl.and_then(|summary| summary.detail.as_deref())
                .unwrap_or("<not WSL>"),
            self.managed_runtime_count,
            self.managed_service_count,
            self.model_cache_entries,
            self.config_dir.display(),
            self.data_dir.display(),
            self.cache_dir.display(),
        )
    }

    const fn legacy_rocm_guidance(&self) -> &'static str {
        if self.legacy_rocm.paths.is_empty() {
            return "none";
        }
        if self.managed_runtime_count == 0 {
            return "legacy ROCm detected; install a managed TheRock runtime with `rocm install sdk --channel release --format wheel` and keep legacy ROCm unmanaged";
        }
        "legacy ROCm detected; keep it side-by-side and use rocm-cli managed TheRock runtimes for local engines"
    }
}

/// Whether this process can hold an interactive exchange with a user.
///
/// Both streams must be a terminal: stdin so an answer can be read, stdout so
/// the question is seen. Anything that captures either — a pipe, a CI step, the
/// dashboard spawning `rocm` as a child — makes this false, and callers then
/// skip the prompt rather than block on input nobody can supply.
///
/// A property of the invocation, not of the host. `rocm examine` reports it so a
/// pasted report explains why prompts were skipped.
pub fn interactive_terminal() -> bool {
    stdin().is_terminal() && stdout().is_terminal()
}

pub const fn default_engine_for_platform() -> &'static str {
    "lemonade"
}

/// The engine this host serves on by default, absent an explicit choice.
///
/// [`default_engine_for_platform`] alone answers "what does this OS fall back
/// to", which is not the same question: on Instinct data-center parts serving
/// goes through vLLM, and reporting the platform constant there contradicts what
/// `serve` actually selects. Use this wherever the CLI *tells the user* what the
/// default engine is; `default_engine_for_platform` remains correct as the
/// last-resort fallback once GPU and recipe preferences have been exhausted.
///
/// A value the user configured still outranks this — callers that have a
/// configured engine must prefer it, mirroring `select_serve_engine`.
#[must_use]
pub fn default_engine_for_host(summary: &HostGpuSummary) -> &'static str {
    preferred_serve_engine_for_host_gpu_summary(summary).unwrap_or_else(default_engine_for_platform)
}

const VLLM_PREFERRED_THEROCK_FAMILIES: &[&str] = &["gfx906", "gfx908", "gfx90a"];

pub fn preferred_serve_engine_for_host_gpu_summary(
    summary: &HostGpuSummary,
) -> Option<&'static str> {
    // The vLLM engine adapter bails out on native Windows builds, so never prefer it
    // there. WSL builds as a Linux target and therefore remains eligible.
    if cfg!(windows) {
        return None;
    }
    preferred_serve_engine_for_therock_family(
        summary
            .therock_family
            .as_deref()
            .or(summary.gfx_target.as_deref()),
    )
}

fn preferred_serve_engine_for_therock_family(family: Option<&str>) -> Option<&'static str> {
    let family = family?.trim();
    if family.is_empty() {
        return None;
    }

    let family = normalize_therock_family(family)
        .as_deref()
        .unwrap_or(family)
        .to_ascii_lowercase();
    if family.ends_with("-dcgpu")
        || VLLM_PREFERRED_THEROCK_FAMILIES
            .iter()
            .any(|candidate| *candidate == family)
    {
        Some("vllm")
    } else {
        None
    }
}

fn detect_kernel_version() -> Option<String> {
    if runtime_is_windows() {
        capture_optional_command("cmd", &["/C", "ver"])
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    } else {
        capture_optional_command("uname", &["-r"])
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    }
}

fn detect_distro_name() -> Option<String> {
    if runtime_is_windows() {
        return Some("Windows".to_owned());
    }

    if runtime_is_linux() {
        return parse_os_release_pretty_name(&fs::read_to_string("/etc/os-release").ok()?)
            .or_else(|| Some("Linux".to_owned()));
    }

    None
}

fn parse_os_release_pretty_name(text: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let value = line.strip_prefix("PRETTY_NAME=")?.trim();
        let value = value.trim_matches('"').trim_matches('\'').trim();
        (!value.is_empty()).then(|| value.to_owned())
    })
}

fn detect_cpu_model_with_windows_inventory(
    windows_inventory: Option<&WindowsExamineInventory>,
) -> Option<String> {
    if runtime_is_windows()
        && let Some(inventory) = windows_inventory
    {
        return inventory.cpu_model.clone();
    }

    detect_cpu_model()
}

fn detect_cpu_model() -> Option<String> {
    if runtime_is_windows() {
        let script =
            "Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name";
        return capture_optional_command_with_timeout(
            "powershell",
            &["-NoProfile", "-Command", script],
            OPTIONAL_COMMAND_TIMEOUT,
        )
        .map(|value| normalize_cpu_model(&value))
        .filter(|value| !value.is_empty());
    }

    if runtime_is_linux()
        && let Some(model) = fs::read_to_string("/proc/cpuinfo").ok().and_then(|text| {
            text.lines().find_map(|line| {
                let value = line
                    .strip_prefix("model name")
                    .and_then(|rest| rest.split_once(':').map(|(_, value)| value))
                    .or_else(|| {
                        line.strip_prefix("Hardware")
                            .and_then(|rest| rest.split_once(':').map(|(_, value)| value))
                    })?;
                let value = normalize_cpu_model(value);
                (!value.is_empty()).then_some(value)
            })
        })
    {
        return Some(model);
    }

    None
}

fn detect_system_ram_gib_with_windows_inventory(
    windows_inventory: Option<&WindowsExamineInventory>,
) -> Option<f64> {
    if runtime_is_windows()
        && let Some(inventory) = windows_inventory
    {
        return inventory.system_ram_gib;
    }

    detect_system_ram_gib()
}

pub fn detect_system_ram_gib() -> Option<f64> {
    if runtime_is_windows() {
        let script = "(Get-CimInstance -ClassName Win32_ComputerSystem -Property TotalPhysicalMemory).TotalPhysicalMemory";
        return capture_optional_command_with_timeout(
            "powershell",
            &["-NoProfile", "-Command", script],
            OPTIONAL_COMMAND_TIMEOUT,
        )
        .and_then(|value| bytes_text_to_gib(&value));
    }

    if runtime_is_linux()
        && let Some(kib) = fs::read_to_string("/proc/meminfo").ok().and_then(|text| {
            text.lines().find_map(|line| {
                let value = line.strip_prefix("MemTotal:")?.trim();
                let number = value.split_whitespace().next()?.parse::<f64>().ok()?;
                Some(number)
            })
        })
    {
        return Some(kib / 1024.0 / 1024.0);
    }

    if cfg!(target_os = "macos") {
        return capture_optional_command("sysctl", &["-n", "hw.memsize"])
            .and_then(|value| bytes_text_to_gib(&value));
    }

    None
}

fn bytes_text_to_gib(value: &str) -> Option<f64> {
    let bytes = value.trim().parse::<f64>().ok()?;
    (bytes > 0.0).then_some(bytes / 1024.0 / 1024.0 / 1024.0)
}

fn format_gib_value(value: f64) -> String {
    if value >= 10.0 {
        format!("{value:.0} GiB")
    } else {
        format!("{value:.1} GiB")
    }
}

fn normalize_cpu_model(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Whether this host is WSL2 — the one answer, for every caller.
///
/// There were three of these, and they disagreed. The install summary asked for
/// `/dev/dxg` or `microsoft` in `/proc/version`; the JSON probe asked for
/// `microsoft` or `wsl` in `/proc/version`, or `$WSL_DISTRO_NAME`. So a host
/// with `/dev/dxg` but a kernel string naming neither was WSL to one and not the
/// other, and the same command could contradict itself between its two output
/// forms. Worse, the e2e harness derives `is_wsl` for its whole expectation
/// matrix by reading one of them.
///
/// `/dev/dxg` is trusted on its own, and so is the kernel's own build string in
/// `/proc/version` — nothing else stamps a kernel `-microsoft-standard-WSL2` or
/// `-Microsoft`. `$WSL_DISTRO_NAME` is not trusted at all: it is an ordinary
/// environment variable that can survive into a shell that merely inherited it
/// (over `ssh`, in a `systemd` unit, under `sudo` without `-E`, under `env -i` or
/// cron) without corroborating anything. See [`wsl_signals_indicate_wsl`] for why
/// a false positive is no longer cheap.
#[must_use]
pub fn is_wsl_host() -> bool {
    runtime_is_linux()
        && wsl_signals_indicate_wsl(
            Path::new("/dev/dxg").exists(),
            &fs::read_to_string("/proc/version").unwrap_or_default(),
        )
}

/// Whether the host runs WSL 1 rather than WSL 2.
///
/// WSL 1 translates syscalls instead of running a real kernel, so it has no
/// `/dev/dxg` and no GPU path at all. Without this the catalog would tell a WSL 1
/// user to update a Windows driver that could never help them.
///
/// WSL 1 reports a kernel ending in `-Microsoft`, as in `4.4.0-19041-Microsoft`.
/// WSL 2 builds all carry `microsoft-standard`, with the `-WSL2` suffix added
/// later — `4.19.104-microsoft-standard` was the original and has no `WSL2` in
/// it at all.
///
/// So the test is the `standard` marker and the trailing position, not the
/// absence of `WSL2`. Keying on `WSL2` alone called every early WSL 2 kernel
/// "WSL 1", which is the asymmetric error [`crate::examine::WslFacts::version`]
/// documents as the one to avoid: it tells the user to convert a distribution
/// that is already converted, at high confidence, while suppressing every other
/// check. Anything unrecognised is read as WSL 2 for the same reason.
#[must_use]
pub(crate) fn is_wsl1_kernel(kernel_release: &str) -> bool {
    let kernel = kernel_release.trim().to_ascii_lowercase();
    kernel.ends_with("-microsoft") && !kernel.contains("standard")
}

/// The dynamic linker cache, or `None` when `ldconfig` could not be run.
///
/// `ldconfig` lives in `/sbin`, which is not on a non-root user's `PATH` on
/// Debian and derivatives. Looking it up by bare name there yields nothing, and
/// an empty cache is indistinguishable from a cache that does not list the
/// library — so a correctly installed ROCDXG read as "not registered with the
/// linker" and the catalog told the user to run `ldconfig` on a working install.
///
/// Search the conventional locations, and report "could not ask" as `None`
/// rather than as an empty answer.
pub(crate) fn ldconfig_cache() -> Option<String> {
    for program in ["ldconfig", "/sbin/ldconfig", "/usr/sbin/ldconfig"] {
        if let Some(text) = capture_optional_command(program, &["-p"]) {
            return Some(text);
        }
    }
    None
}

/// Whether the linker cache lists ROCDXG, or `None` if it could not be read.
pub(crate) fn ldconfig_lists_librocdxg() -> Option<bool> {
    ldconfig_cache().map(|text| text.contains("librocdxg.so"))
}

/// Whether `relative` exists under any ROCm install on this host.
///
/// The WSL probe used to hardcode `/opt/rocm`, so a versioned install at
/// `/opt/rocm-7.x` reported ROCDXG missing and the catalog would then blame a
/// package that was in fact installed. Ask the same resolver the rest of the CLI
/// uses, and keep the conventional root as a fallback for the case where
/// discovery finds nothing.
fn rocm_relative_file_exists(relative: &str) -> bool {
    if Path::new("/opt/rocm").join(relative).exists() {
        return true;
    }
    discover_rocm_installs()
        .iter()
        .any(|install| install.path.join(relative).exists())
}

/// The predicate itself, separated from reading the machine so the union can be
/// tested — including the two cases that used to split the old implementations.
///
/// `/dev/dxg` alone is trusted outright — nothing but WSLg's GPU passthrough
/// creates that device node. The kernel's own build string in `/proc/version` is
/// also trusted alone: only a WSL kernel is built `-microsoft-standard[-WSL2]` or
/// `-Microsoft`, and that string cannot be inherited, forwarded, or left behind
/// by an unrelated shell the way `$WSL_DISTRO_NAME` can. `$WSL_DISTRO_NAME` plays
/// no part here at all — an ordinary bare-metal host that merely inherited it
/// (over `ssh`, from a parent shell, under `sudo` without `-E`) has no
/// `/proc/version` match to go with it, so it still reads as Linux. A false
/// positive the other way no longer costs only a route-out note — this catalog
/// now runs the WSL diagnosis and fix set directly, so a bare-metal host
/// misread as WSL would have its entire bare-metal catalog silently disabled.
fn wsl_signals_indicate_wsl(dxg_device: bool, proc_version: &str) -> bool {
    if dxg_device {
        return true;
    }
    let proc_version = proc_version.to_ascii_lowercase();
    proc_version.contains("microsoft") || proc_version.contains("wsl")
}

pub(crate) fn detect_wsl_summary() -> Option<WslSummary> {
    if !runtime_is_linux() || !is_wsl_host() {
        return None;
    }

    let dxg_device = Path::new("/dev/dxg").exists();
    let is_wsl = true;

    let dxcore = Path::new("/usr/lib/wsl/lib/libdxcore.so").exists();
    let librocdxg = rocm_relative_file_exists("lib/librocdxg.so");
    let rocdxg_dids = rocm_relative_file_exists("share/rocdxg/dids.conf");
    let ldconfig_text = ldconfig_cache();
    let ldconfig_librocdxg = ldconfig_text
        .as_deref()
        .is_some_and(|text| text.contains("librocdxg.so"));
    let rocminfo = tool_on_path("rocminfo");
    let cargo = tool_on_path("cargo");
    let mut missing = Vec::new();
    if !dxg_device {
        missing.push("/dev/dxg");
    }
    if !dxcore {
        missing.push("/usr/lib/wsl/lib/libdxcore.so");
    }
    if !librocdxg {
        // Named without a directory: the file is looked up across every ROCm
        // install, so quoting one root would misreport where it was not found.
        missing.push("librocdxg.so");
    }
    if !ldconfig_librocdxg {
        missing.push("ldconfig:librocdxg.so");
    }
    let detail = if missing.is_empty() {
        Some("WSL DXCore and ROCDXG plumbing detected".to_owned())
    } else {
        Some(format!("missing {}", missing.join(", ")))
    };

    Some(WslSummary {
        is_wsl,
        dxg_device,
        dxcore,
        librocdxg,
        rocdxg_dids,
        ldconfig_librocdxg,
        rocminfo,
        cargo,
        detail,
    })
}

fn detect_driver_summary_with_windows_inventory(
    windows_inventory: Option<&WindowsExamineInventory>,
    wsl: Option<&WslSummary>,
) -> DriverSummary {
    if runtime_is_windows() {
        let detail = windows_inventory
            .and_then(WindowsExamineInventory::amd_display_driver_detail)
            .or_else(|| {
                if windows_inventory.is_none() {
                    detect_windows_amd_display_driver()
                } else {
                    None
                }
            });
        return windows_driver_summary(detail);
    }

    if let Some(wsl) = wsl {
        return wsl_driver_summary(wsl);
    }

    detect_driver_summary()
}

fn detect_driver_summary() -> DriverSummary {
    if runtime_is_windows() {
        let detail = detect_windows_amd_display_driver();
        return windows_driver_summary(detail);
    }

    if runtime_is_linux() {
        let module_detected = Path::new("/sys/module/amdgpu").exists();
        return DriverSummary {
            policy: "linux_official_amd_dkms_wrapper".to_owned(),
            status: if module_detected {
                "amdgpu_available".to_owned()
            } else {
                "not_detected".to_owned()
            },
            detail: if Path::new("/dev/kfd").exists() {
                Some("/dev/kfd is present".to_owned())
            } else if module_detected {
                Some("amdgpu module metadata is present".to_owned())
            } else {
                None
            },
        };
    }

    DriverSummary {
        policy: "inspection_only".to_owned(),
        status: "unsupported_platform".to_owned(),
        detail: None,
    }
}

/// The amdgpu kernel module's version on Linux, if it reports one.
///
/// Prefers the sysfs attribute (a plain file read, no subprocess) that DKMS
/// builds of amdgpu expose. Falls back to `modinfo`, which is the only source
/// for the in-tree kernel module -- it doesn't populate
/// `/sys/module/amdgpu/version` at all.
fn detect_linux_amdgpu_driver_version() -> Option<String> {
    if let Ok(text) = fs::read_to_string("/sys/module/amdgpu/version") {
        let version = text.trim();
        if !version.is_empty() {
            return Some(version.to_owned());
        }
    }
    let (rc, out, _) = examine::run("modinfo", &["amdgpu"], examine::SHORT);
    if rc != 0 {
        return None;
    }
    out.lines()
        .find_map(|line| line.strip_prefix("version:"))
        .map(str::trim)
        .filter(|version| !version.is_empty())
        .map(str::to_owned)
}

/// The GPU driver version for this machine.
///
/// Sourced however the current platform exposes it: the amdgpu kernel module
/// on Linux, the AMD display driver on Windows, or -- inside WSL -- the
/// Windows host's display driver, since that's the driver a WSL guest's GPU
/// workloads actually depend on, not its own (driver-less) amdgpu module.
pub fn detect_gpu_driver_version() -> Option<String> {
    if is_wsl_host() {
        return match detect_wsl_host_driver() {
            WslHostDriverProbe::Version(version) => Some(version),
            WslHostDriverProbe::Unreachable | WslHostDriverProbe::NoAmdDisplay => None,
        };
    }
    if runtime_is_windows() {
        return detect_windows_amd_display_driver();
    }
    if runtime_is_linux() {
        return detect_linux_amdgpu_driver_version();
    }
    None
}

impl WslSummary {
    /// Whether the ROCDXG plumbing a GPU workload needs is actually in place.
    ///
    /// Extracted so `serve` can act on the same answer `examine` prints, rather
    /// than reaching its own conclusion from a different source. That split is
    /// what let `examine` report `wsl_rocdxg_ready` while `serve` refused on the
    /// same machine for want of a GPU.
    #[must_use]
    pub const fn rocdxg_ready(&self) -> bool {
        self.dxg_device && self.dxcore && self.librocdxg && self.ldconfig_librocdxg
    }
}

fn wsl_driver_summary(wsl: &WslSummary) -> DriverSummary {
    let ready = wsl.rocdxg_ready();
    let status = if ready {
        "wsl_rocdxg_ready"
    } else if wsl.dxg_device && wsl.dxcore {
        "wsl_rocdxg_missing"
    } else {
        "wsl_gpu_plumbing_missing"
    };
    DriverSummary {
        policy: "wsl_rocdxg".to_owned(),
        status: status.to_owned(),
        detail: wsl.detail.clone(),
    }
}

fn windows_driver_summary(detail: Option<String>) -> DriverSummary {
    DriverSummary {
        policy: "windows_validate_only".to_owned(),
        status: if detail.is_some() {
            "amd_display_driver_detected".to_owned()
        } else {
            "not_detected".to_owned()
        },
        detail,
    }
}

#[cfg(windows)]
fn detect_windows_amd_display_driver() -> Option<String> {
    if !runtime_is_windows() {
        return None;
    }
    let script = "$gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.AdapterCompatibility -match 'AMD|Advanced Micro Devices' -or $_.Name -match 'AMD|Radeon|Instinct' } | Select-Object -First 1 -Property Name,DriverVersion; if ($gpu) { \"$($gpu.Name) driver $($gpu.DriverVersion)\" }";
    capture_optional_command("powershell", &["-NoProfile", "-Command", script])
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

#[cfg(not(windows))]
const fn detect_windows_amd_display_driver() -> Option<String> {
    None
}

#[cfg(windows)]
fn detect_windows_examine_inventory() -> Option<WindowsExamineInventory> {
    if !runtime_is_windows() {
        return None;
    }
    let mut inventory = WindowsExamineInventory::default();
    if let Some(pnp_util) = detect_windows_examine_inventory_from_pnputil() {
        inventory.merge_missing_from(pnp_util);
    }
    if inventory.displays.is_empty()
        && let Some(video) = detect_windows_examine_inventory_from_video_controller()
    {
        inventory.merge_missing_from(video);
    }
    if inventory.displays.is_empty()
        && let Some(pnp) = detect_windows_examine_inventory_from_pnp_entity()
    {
        inventory.merge_missing_from(pnp);
    }
    if (inventory.cpu_model.is_none() || inventory.system_ram_gib.is_none())
        && let Some(system) = detect_windows_system_inventory_from_cim()
    {
        inventory.merge_missing_from(system);
    }

    (!inventory.is_empty()).then_some(inventory)
}

#[cfg(windows)]
fn detect_windows_examine_inventory_from_pnputil() -> Option<WindowsExamineInventory> {
    if !runtime_is_windows() {
        return None;
    }
    capture_optional_command_with_timeout(
        "pnputil",
        &["/enum-devices", "/class", "Display"],
        WINDOWS_INVENTORY_QUERY_TIMEOUT,
    )
    .map(|output| parse_windows_pnputil_display_inventory(&output))
}

#[cfg(windows)]
fn detect_windows_examine_inventory_from_video_controller() -> Option<WindowsExamineInventory> {
    if !runtime_is_windows() {
        return None;
    }
    capture_optional_command_with_timeout(
        "powershell",
        &[
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            WINDOWS_VIDEO_CONTROLLER_INVENTORY_SCRIPT,
        ],
        WINDOWS_INVENTORY_QUERY_TIMEOUT,
    )
    .map(|output| parse_windows_examine_inventory(&output))
}

#[cfg(windows)]
fn detect_windows_system_inventory_from_cim() -> Option<WindowsExamineInventory> {
    if !runtime_is_windows() {
        return None;
    }
    capture_optional_command_with_timeout(
        "powershell",
        &[
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            WINDOWS_SYSTEM_INVENTORY_SCRIPT,
        ],
        OPTIONAL_COMMAND_TIMEOUT,
    )
    .map(|output| parse_windows_examine_inventory(&output))
}

#[cfg(windows)]
fn detect_windows_examine_inventory_from_pnp_entity() -> Option<WindowsExamineInventory> {
    if !runtime_is_windows() {
        return None;
    }
    capture_optional_command_with_timeout(
        "powershell",
        &[
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            WINDOWS_PNP_ENTITY_INVENTORY_SCRIPT,
        ],
        WINDOWS_INVENTORY_QUERY_TIMEOUT,
    )
    .map(|output| parse_windows_examine_inventory(&output))
}

#[cfg(not(windows))]
const fn detect_windows_examine_inventory() -> Option<WindowsExamineInventory> {
    None
}

#[cfg(any(windows, test))]
fn clean_windows_display_name(value: &str) -> String {
    let value = value.trim();
    let value = value.rsplit_once(';').map_or(value, |(_, name)| name);
    value.trim().to_owned()
}

#[cfg_attr(not(windows), allow(dead_code))]
fn parse_windows_examine_inventory(text: &str) -> WindowsExamineInventory {
    let mut inventory = WindowsExamineInventory::default();

    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let mut fields = line.split('\t');
        match fields.next() {
            Some("CPU") => {
                let cpu_model = fields.collect::<Vec<_>>().join("\t");
                let cpu_model = normalize_cpu_model(&cpu_model);
                if !cpu_model.is_empty() {
                    inventory.cpu_model = Some(cpu_model);
                }
            }
            Some("RAM") => {
                let bytes = fields.next().unwrap_or("").trim();
                inventory.system_ram_gib = bytes_text_to_gib(bytes);
            }
            Some("GPU") => {
                let name = fields.next().unwrap_or("").trim().to_owned();
                let driver_version = fields
                    .next()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_owned);
                let pnp_device_id = fields
                    .next()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_owned);
                if !name.is_empty() || driver_version.is_some() || pnp_device_id.is_some() {
                    inventory.displays.push(WindowsDisplayAdapter {
                        name,
                        driver_version,
                        pnp_device_id,
                    });
                }
            }
            _ => {}
        }
    }

    inventory
}

#[cfg(any(windows, test))]
fn parse_windows_pnputil_display_inventory(text: &str) -> WindowsExamineInventory {
    let mut inventory = WindowsExamineInventory::default();
    let mut name: Option<String> = None;
    let mut instance_id: Option<String> = None;
    let mut driver_version: Option<String> = None;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            push_windows_pnputil_display(
                &mut inventory,
                &mut name,
                &mut instance_id,
                &mut driver_version,
            );
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim().to_ascii_lowercase();
        let value = value.trim();
        if value.is_empty() {
            continue;
        }
        match key.as_str() {
            "instance id" | "device instance id" => {
                instance_id = Some(value.to_owned());
            }
            "device description" | "friendly name" | "name" => {
                name = Some(clean_windows_display_name(value));
            }
            "driver version" => {
                driver_version = Some(value.to_owned());
            }
            _ => {}
        }
    }
    push_windows_pnputil_display(
        &mut inventory,
        &mut name,
        &mut instance_id,
        &mut driver_version,
    );

    inventory
}

#[cfg(any(windows, test))]
fn push_windows_pnputil_display(
    inventory: &mut WindowsExamineInventory,
    name: &mut Option<String>,
    instance_id: &mut Option<String>,
    driver_version: &mut Option<String>,
) {
    let pnp = instance_id.take();
    let display_name = name.take().unwrap_or_default();
    let driver = driver_version.take();
    let has_amd_id = pnp
        .as_deref()
        .is_some_and(|value| value.to_ascii_uppercase().contains("VEN_1002"));
    let has_amd_name = display_name
        .to_ascii_lowercase()
        .split_whitespace()
        .any(|token| matches!(token, "amd" | "radeon" | "instinct"));
    if !has_amd_id && !has_amd_name {
        return;
    }
    inventory.displays.push(WindowsDisplayAdapter {
        name: display_name,
        driver_version: driver,
        pnp_device_id: pnp,
    });
}

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
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| program.to_owned()),
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
            empty_as_unknown(&display.name),
            display.pnp_device_id.as_deref().unwrap_or("<unknown>"),
            display.driver_version.as_deref().unwrap_or("<unknown>"),
            gfx
        );
    }
    append_diagnostic_stream(output, "stdout", &result.stdout);
    append_diagnostic_stream(output, "stderr", &result.stderr);
}

#[cfg(windows)]
fn empty_as_unknown(value: &str) -> &str {
    let value = value.trim();
    if value.is_empty() { "<unknown>" } else { value }
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

fn count_json_files(dir: &Path) -> usize {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|entry| entry.path().extension().and_then(|value| value.to_str()) == Some("json"))
        .count()
}

fn count_dir_entries(dir: &Path) -> usize {
    fs::read_dir(dir).map_or(0, |entries| entries.flatten().count())
}

pub fn detect_host_gpu_summary(paths: Option<&AppPaths>) -> HostGpuSummary {
    detect_host_gpu_summary_fast(paths)
}

#[cfg(windows)]
fn detect_host_gpu_summary_fast(_paths: Option<&AppPaths>) -> HostGpuSummary {
    let windows_inventory = detect_windows_examine_inventory();
    let gfx_target = detect_windows_display_gfx_target_with_inventory(windows_inventory.as_ref());
    let therock_family = gfx_target.as_deref().and_then(normalize_therock_family);
    let name = windows_inventory
        .as_ref()
        .and_then(WindowsExamineInventory::amd_display_name);
    HostGpuSummary {
        name,
        gfx_target,
        therock_family,
    }
}

#[cfg(target_os = "linux")]
fn detect_host_gpu_summary_fast(_paths: Option<&AppPaths>) -> HostGpuSummary {
    if runtime_is_windows() {
        let windows_inventory = detect_windows_examine_inventory();
        let gfx_target =
            detect_windows_display_gfx_target_with_inventory(windows_inventory.as_ref());
        let therock_family = gfx_target.as_deref().and_then(normalize_therock_family);
        let name = windows_inventory
            .as_ref()
            .and_then(WindowsExamineInventory::amd_display_name);
        return HostGpuSummary {
            name,
            gfx_target,
            therock_family,
        };
    }

    let linux_gfx_target = detect_linux_sysfs_gfx_target();
    let linux_name = detect_linux_primary_gpu_name();
    let wsl_display_probe = if linux_gfx_target.is_none() || linux_name.is_none() {
        detect_wsl_windows_display_probe_text()
    } else {
        None
    };
    let gfx_target = linux_gfx_target.or_else(|| {
        wsl_display_probe
            .as_deref()
            .and_then(parse_windows_display_gfx_target)
    });
    let therock_family = gfx_target.as_deref().and_then(normalize_therock_family);
    let name = linux_name.or_else(|| {
        wsl_display_probe
            .as_deref()
            .and_then(parse_windows_display_name)
    });
    HostGpuSummary {
        name,
        gfx_target,
        therock_family,
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
fn detect_host_gpu_summary_fast(_paths: Option<&AppPaths>) -> HostGpuSummary {
    HostGpuSummary::default()
}

#[allow(dead_code)]
fn detect_host_gpu_summary_full(paths: Option<&AppPaths>) -> HostGpuSummary {
    let windows_inventory = detect_windows_examine_inventory();
    let wsl = detect_wsl_summary();
    let gfx_target =
        detect_host_gfx_target_with_context(windows_inventory.as_ref(), wsl.as_ref(), paths);
    let therock_family = gfx_target.as_deref().and_then(normalize_therock_family);
    let name = detect_host_gpu_name_with_context(windows_inventory.as_ref(), wsl.as_ref());
    HostGpuSummary {
        name,
        gfx_target,
        therock_family,
    }
}

fn detect_host_gpu_name_with_context(
    windows_inventory: Option<&WindowsExamineInventory>,
    wsl: Option<&WslSummary>,
) -> Option<String> {
    windows_inventory
        .and_then(WindowsExamineInventory::amd_display_name)
        .or_else(detect_linux_primary_gpu_name)
        .or_else(|| detect_wsl_windows_display_name(wsl))
}

fn detect_managed_therock_sdk_gfx_target(paths: &AppPaths) -> Option<String> {
    managed_therock_sdk_probe_candidates(&paths.data_dir.join("runtimes").join("registry"))
        .into_iter()
        .find_map(|candidate| {
            let tool = managed_sdk_tool_path(&candidate.bin_path, "rocm_agent_enumerator")?;
            let mut envs = Vec::new();
            if let Some(ld_library_path) = managed_sdk_ld_library_path(&candidate) {
                envs.push(("LD_LIBRARY_PATH", ld_library_path));
            }
            capture_optional_path_command_with_env(&tool, &[], &envs, OPTIONAL_COMMAND_TIMEOUT)
                .and_then(|output| extract_first_gfx_token(&output))
        })
}

pub(crate) fn managed_therock_sdk_probe_candidates(
    registry_dir: &Path,
) -> Vec<TheRockSdkProbeCandidate> {
    let Ok(entries) = fs::read_dir(registry_dir) else {
        return Vec::new();
    };
    let mut candidates = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Some(record) = fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<TheRockFamilyManifest>(&bytes).ok())
        else {
            continue;
        };
        if !record.looks_like_therock() {
            continue;
        }
        let Some(sdk) = record.rocm_sdk else {
            continue;
        };
        if !sdk.import_ok {
            continue;
        }
        let Some(root_path) = sdk.root_path else {
            continue;
        };
        let Some(bin_path) = sdk.bin_path else {
            continue;
        };
        candidates.push(TheRockSdkProbeCandidate {
            installed_at_unix_ms: record.installed_at_unix_ms.unwrap_or(0),
            site_packages: sdk.site_packages,
            root_path,
            bin_path,
            library_paths: sdk.library_paths,
        });
    }
    candidates.sort_by_key(|candidate| std::cmp::Reverse(candidate.installed_at_unix_ms));
    candidates
}

fn managed_sdk_ld_library_path(candidate: &TheRockSdkProbeCandidate) -> Option<OsString> {
    let mut paths = Vec::new();
    collect_managed_runtime_library_paths(
        &candidate.root_path,
        candidate.site_packages.as_deref(),
        &mut paths,
    );
    let wsl_lib = PathBuf::from("/usr/lib/wsl/lib");
    if wsl_lib.is_dir() {
        paths.push(wsl_lib);
    }
    if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH")
        && !existing.is_empty()
    {
        paths.extend(std::env::split_paths(&existing));
    }
    if paths.is_empty() {
        None
    } else {
        std::env::join_paths(paths).ok()
    }
}

/// Every library directory a managed runtime keeps, given its root and the
/// `site-packages` its SDK recorded.
///
/// One description of the layout, deliberately. A wheel-format runtime does not
/// keep its ROCm libraries under the root: they sit in a sibling `_rocm_sdk_*`
/// package inside `site-packages`, and a caller that walks the root alone sees
/// an empty runtime rather than a populated one. That is not a difference a
/// caller should have to remember, so it lives here and every search shares it.
pub(crate) fn collect_managed_runtime_library_paths(
    root: &Path,
    site_packages: Option<&Path>,
    paths: &mut Vec<PathBuf>,
) {
    collect_sdk_library_paths(root, paths);
    if let Some(recorded) = site_packages {
        collect_sdk_package_library_paths(recorded, paths);
    }
}

/// Library directories of the `_rocm_sdk_*` packages inside `site_packages`.
///
/// They belong to the runtime that contains them, not to themselves.
fn collect_sdk_package_library_paths(site_packages: &Path, paths: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(site_packages) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|name| name.starts_with("_rocm_sdk_"))
        {
            collect_sdk_library_paths(&path, paths);
        }
    }
}

pub(crate) fn collect_sdk_library_paths(root: &Path, paths: &mut Vec<PathBuf>) {
    for path in [
        root.join("bin"),
        root.join("lib"),
        root.join("lib64"),
        root.join("lib").join("rocm_sysdeps").join("lib"),
    ] {
        if path.is_dir() {
            paths.push(path);
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TheRockSdkProbeCandidate {
    installed_at_unix_ms: u128,
    pub(crate) site_packages: Option<PathBuf>,
    pub(crate) root_path: PathBuf,
    bin_path: PathBuf,
    pub(crate) library_paths: Vec<PathBuf>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{temp_app_paths, workspace_test_artifact_dir};
    use std::path::PathBuf;

    fn write_fake_rocm_agent_enumerator(bin_dir: &Path, target: &str) -> Result<()> {
        if cfg!(windows) {
            let path = bin_dir.join("rocm_agent_enumerator.cmd");
            fs::write(path, format!("@echo off\r\necho {target}\r\n"))?;
        } else {
            let path = bin_dir.join("rocm_agent_enumerator");
            fs::write(&path, format!("#!/bin/sh\necho {target}\n"))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&path, fs::Permissions::from_mode(0o755))?;
            }
        }
        Ok(())
    }

    #[test]
    fn managed_sdk_probe_detects_gfx_from_therock_tool() -> Result<()> {
        let (root, paths) = temp_app_paths("managed-sdk-gfx");
        let registry = paths.data_dir.join("runtimes").join("registry");
        let site_packages = root.join("site-packages");
        let sdk_root = site_packages.join("_rocm_sdk_devel");
        let sdk_bin = sdk_root.join("bin");
        fs::create_dir_all(&sdk_bin)?;
        write_fake_rocm_agent_enumerator(&sdk_bin, "gfx1201")?;
        fs::create_dir_all(&registry)?;
        fs::write(
            registry.join("runtime.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "runtime_id": "therock-release:gfx120X-all",
                "family": "gfx120X-all",
                "installed_at_unix_ms": 10,
                "rocm_sdk": {
                    "import_ok": true,
                    "site_packages": site_packages,
                    "root_path": sdk_root,
                    "bin_path": sdk_bin
                }
            }))?,
        )?;

        assert_eq!(
            detect_managed_therock_sdk_gfx_target(&paths),
            Some("gfx1201".to_owned())
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    /// `managed_therock_sdk_probe_candidates` surfaces the SDK's own recorded
    /// `library_paths` rather than dropping them.
    ///
    /// Those are what `examine`'s comgr/HIP search now reads to find a managed
    /// runtime's libraries (see `known_install_roots`/`install_library_dirs` in
    /// `examine.rs`), in place of re-deriving the layout from `root_path` and
    /// `site_packages` after the fact -- a guess that does not hold for every
    /// real install shape, which is what left a managed runtime's own code
    /// object manager library unseen on a real host
    /// (`examine-finds-the-managed-runtimes-own-compilation-library`). A
    /// candidate whose `library_paths` came back empty would defeat that fix
    /// silently, so this pins the field surviving the read.
    #[test]
    fn managed_sdk_probe_candidate_carries_recorded_library_paths() -> Result<()> {
        let (root, paths) = temp_app_paths("managed-sdk-library-paths");
        let registry = paths.data_dir.join("runtimes").join("registry");
        let site_packages = root.join("site-packages");
        let sdk_root = site_packages.join("_rocm_sdk_devel");
        let sdk_bin = sdk_root.join("bin");
        let comgr_dir = site_packages.join("_rocm_sdk_core").join("lib");
        fs::create_dir_all(&sdk_bin)?;
        fs::create_dir_all(&comgr_dir)?;
        fs::create_dir_all(&registry)?;
        fs::write(
            registry.join("runtime.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "runtime_id": "therock-release:gfx120X-all",
                "family": "gfx120X-all",
                "installed_at_unix_ms": 10,
                "rocm_sdk": {
                    "import_ok": true,
                    "site_packages": site_packages,
                    "root_path": sdk_root,
                    "bin_path": sdk_bin,
                    "library_paths": [comgr_dir]
                }
            }))?,
        )?;

        let candidates = managed_therock_sdk_probe_candidates(&registry);
        assert_eq!(candidates.len(), 1, "expected exactly one candidate");
        assert_eq!(
            candidates[0].library_paths,
            vec![comgr_dir],
            "the recorded library_paths must survive into the candidate, or the \
             comgr/HIP search has nowhere else reliable to find them"
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn managed_sdk_probe_skips_non_therock_manifests() -> Result<()> {
        let (root, paths) = temp_app_paths("managed-sdk-skip-non-therock");
        let registry = paths.data_dir.join("runtimes").join("registry");
        let site_packages = root.join("site-packages");
        let sdk_root = site_packages.join("_rocm_sdk_devel");
        let sdk_bin = sdk_root.join("bin");
        fs::create_dir_all(&sdk_bin)?;
        write_fake_rocm_agent_enumerator(&sdk_bin, "gfx9999")?;
        fs::create_dir_all(&registry)?;
        fs::write(
            registry.join("runtime.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "runtime_id": "external-runtime",
                "family": "gfx120X-all",
                "installed_at_unix_ms": 10,
                "rocm_sdk": {
                    "import_ok": true,
                    "site_packages": site_packages,
                    "root_path": sdk_root,
                    "bin_path": sdk_bin
                }
            }))?,
        )?;

        assert_eq!(detect_managed_therock_sdk_gfx_target(&paths), None);
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn normalize_therock_family_maps_gfx1101_to_gfx110x_all() {
        assert_eq!(
            normalize_therock_family("gfx1101"),
            Some("gfx110X-all".to_owned())
        );
    }

    #[test]
    fn normalize_therock_family_maps_gfx1103_to_gfx110x_all() {
        assert_eq!(
            normalize_therock_family("gfx1103"),
            Some("gfx110X-all".to_owned())
        );
    }

    #[test]
    fn normalize_therock_family_maps_gfx1201_to_gfx120x_all() {
        assert_eq!(
            normalize_therock_family("gfx1201"),
            Some("gfx120X-all".to_owned())
        );
    }

    #[test]
    fn normalize_therock_family_accepts_canonical_family_labels() {
        assert_eq!(
            normalize_therock_family("gfx120X-all"),
            Some("gfx120X-all".to_owned())
        );
        assert_eq!(
            normalize_therock_family("gfx110X-all"),
            Some("gfx110X-all".to_owned())
        );
        assert_eq!(
            normalize_therock_family("gfx94X-dcgpu"),
            Some("gfx94X-dcgpu".to_owned())
        );
    }

    /// ROCm 10's next layout publishes a `gfx125X-dcgpu` family; the raw
    /// `gfx1250` arch a host reports has to land on it.
    #[test]
    fn normalize_therock_family_maps_gfx1250_to_gfx125x_dcgpu() {
        assert_eq!(
            normalize_therock_family("gfx1250"),
            Some("gfx125X-dcgpu".to_owned())
        );
    }

    #[test]
    fn known_therock_families_all_round_trip() {
        for family in known_therock_families() {
            assert_eq!(
                normalize_therock_family(family).as_deref(),
                Some(*family),
                "known family `{family}` must normalize back to itself"
            );
        }
    }

    #[test]
    fn known_therock_families_is_not_empty() {
        assert!(!known_therock_families().is_empty());
    }

    #[test]
    fn windows_display_parser_maps_rx_9070_xt_device_id_to_gfx1201() {
        let text = "ASPEED Graphics Family(WDDM)\tPCI\\VEN_1A03&DEV_2000\nAMD Radeon RX 9070 XT\tPCI\\VEN_1002&DEV_7550&SUBSYS_2435148C&REV_C0";
        assert_eq!(
            parse_windows_display_gfx_target(text),
            Some("gfx1201".to_owned())
        );
    }

    #[test]
    fn windows_display_parser_maps_known_amd_pci_ids() {
        for (device_id, expected) in [
            ("73A0", "gfx1030"),
            ("73C0", "gfx1031"),
            ("73E0", "gfx1032"),
            ("163F", "gfx1033"),
            ("743F", "gfx1034"),
            ("1681", "gfx1035"),
            ("164E", "gfx1036"),
            ("15BF", "gfx1103"),
            ("164F", "gfx1103"),
            ("1900", "gfx1103"),
            ("1114", "gfx1152"),
        ] {
            assert_eq!(
                parse_windows_display_gfx_target(&format!(
                    "AMD Display Adapter\tPCI\\VEN_1002&DEV_{device_id}"
                )),
                Some(expected.to_owned()),
                "{device_id}"
            );
        }
    }

    #[test]
    fn windows_display_parser_falls_back_to_name_when_pci_id_is_uncertain() {
        assert_eq!(
            parse_windows_display_gfx_target("AMD Radeon 820M\tPCI\\VEN_1002&DEV_1902"),
            Some("gfx1153".to_owned())
        );
    }

    #[test]
    fn windows_display_name_parser_uses_first_nonempty_adapter_name() {
        assert_eq!(
            parse_windows_display_name("\nAMD Radeon RX 9070 XT\tPCI\\VEN_1002&DEV_7550\n"),
            Some("AMD Radeon RX 9070 XT".to_owned())
        );
    }

    #[test]
    fn windows_display_parser_maps_known_marketing_names() {
        for (name, expected) in [
            ("AMD Radeon RX 9070 XT\t", "gfx1201"),
            ("AMD Radeon RX 9060 XT\t", "gfx1200"),
            ("AMD Radeon RX 7900 XTX\t", "gfx1100"),
            ("AMD Radeon RX 7800 XT\t", "gfx1101"),
            ("AMD Radeon RX 7600\t", "gfx1102"),
            ("AMD Radeon RX 6800 XT\t", "gfx1030"),
            ("AMD Radeon RX 6800M\t", "gfx1031"),
            ("AMD Radeon RX 6700 XT\t", "gfx1031"),
            ("AMD Radeon RX 6600\t", "gfx1032"),
            ("AMD Radeon RX 6500 XT\t", "gfx1034"),
            ("AMD Radeon 680M\t", "gfx1035"),
            ("AMD Radeon 660M\t", "gfx1035"),
            ("AMD Radeon 610M\t", "gfx1036"),
            ("AMD Radeon 780M\t", "gfx1103"),
            ("AMD Radeon 760M\t", "gfx1103"),
            ("AMD Radeon 740M\t", "gfx1103"),
            ("AMD Radeon 8060S\t", "gfx1151"),
            ("AMD Radeon 890M\t", "gfx1150"),
            ("AMD Radeon 860M\t", "gfx1152"),
            ("AMD Radeon 820M\t", "gfx1153"),
            ("Steam Deck\t", "gfx1033"),
        ] {
            assert_eq!(
                parse_windows_display_gfx_target(name),
                Some(expected.to_owned()),
                "{name}"
            );
        }
    }

    #[test]
    fn amd_pci_device_id_parser_requires_amd_vendor() {
        assert_eq!(
            amd_pci_device_id_from_pnp_id("PCI\\VEN_1002&DEV_7550&SUBSYS_2435148C"),
            Some("7550".to_owned())
        );
        assert_eq!(
            amd_pci_device_id_from_pnp_id("PCI\\VEN_1A03&DEV_2000"),
            None
        );
    }

    #[test]
    fn windows_examine_gfx_detection_uses_inventory_without_rocm_tools() {
        if !cfg!(windows) {
            return;
        }
        let inventory = parse_windows_examine_inventory(
            "GPU\tAMD Radeon RX 9070 XT\t32.0.23033.1002\tPCI\\VEN_1002&DEV_7550",
        );

        assert_eq!(
            detect_host_gfx_target_with_context(Some(&inventory), None, None),
            Some("gfx1201".to_owned())
        );
    }

    #[test]
    fn gc_version_converts_to_gfx_target() {
        assert_eq!(
            gfx_target_from_gc_version(11, 0, 1),
            Some("gfx1101".to_owned())
        );
        assert_eq!(
            gfx_target_from_gc_version(11, 0, 3),
            Some("gfx1103".to_owned())
        );
        // Two digits of major stay decimal.
        assert_eq!(
            gfx_target_from_gc_version(12, 5, 0),
            Some("gfx1250".to_owned())
        );
    }

    #[test]
    fn gc_version_encodes_components_above_nine_as_hex_digits() {
        // GC 9.0.10 is gfx90a (MI210/MI250), not "gfx9010": minor and revision
        // are single hex digits. Decimal concatenation agreed with hex only
        // while every component stayed below 10.
        assert_eq!(
            gfx_target_from_gc_version(9, 0, 10),
            Some("gfx90a".to_owned())
        );
        assert_eq!(
            gfx_target_from_gc_version(9, 4, 12),
            Some("gfx94c".to_owned())
        );

        // A component that is not a single hex digit describes no gfx target,
        // so detection falls through rather than acting on a fabricated one.
        assert_eq!(gfx_target_from_gc_version(12, 16, 0), None);
        assert_eq!(gfx_target_from_gc_version(12, 0, 16), None);
        assert_eq!(gfx_target_from_gc_version(0, 0, 1), None);
    }

    #[test]
    fn gfx90a_gc_version_normalizes_to_its_own_therock_family() {
        // The point of the fix, end to end: "gfx9010" missed every specific arm
        // of `normalize_therock_family` and fell through to the loose `gfx90`
        // one, yielding "gfx90X-dcgpu" — the wrong runtime wheel, and outside
        // `VLLM_PREFERRED_THEROCK_FAMILIES`, so engine selection changed too.
        let target = gfx_target_from_gc_version(9, 0, 10).expect("gfx90a target");
        assert_eq!(normalize_therock_family(&target), Some("gfx90a".to_owned()));
    }

    #[test]
    fn linux_kfd_gfx_target_parser_accepts_numeric_and_direct_tokens() {
        assert_eq!(
            parse_linux_kfd_gfx_target("110003"),
            Some("gfx1103".to_owned())
        );
        assert_eq!(
            parse_linux_kfd_gfx_target("120001"),
            Some("gfx1201".to_owned())
        );
        assert_eq!(
            parse_linux_kfd_gfx_target("gfx1103"),
            Some("gfx1103".to_owned())
        );
        // KFD packs gfx90a as 9·10000 + 0·100 + 10.
        assert_eq!(
            parse_linux_kfd_gfx_target("90010"),
            Some("gfx90a".to_owned())
        );
        // A packed version whose components are not single hex digits is not a
        // target; it must not be passed through as `gfx{digits}`, which used to
        // normalize to a plausible-looking but wrong family.
        assert_eq!(parse_linux_kfd_gfx_target("121600"), None);
        assert_eq!(parse_linux_kfd_gfx_target("not-a-target"), None);
    }

    #[test]
    fn linux_ip_discovery_gc_fixture_maps_to_gfx_target() -> Result<()> {
        let (root, _paths) = temp_app_paths("linux-ip-discovery");
        let gc = root
            .join("ip_discovery")
            .join("die")
            .join("0")
            .join("GC")
            .join("0");
        fs::create_dir_all(&gc)?;
        fs::write(gc.join("major"), "11")?;
        fs::write(gc.join("minor"), "0")?;
        fs::write(gc.join("revision"), "3")?;

        assert_eq!(
            detect_ip_discovery_gc_target(&root.join("ip_discovery")),
            Some("gfx1103".to_owned())
        );

        // KNOWN WRONG, and pinned so the gap stays visible: on the GC 9.4.x line
        // the GC IP version is not the LLVM target. Aldebaran (MI200/MI250)
        // reports GC 9.4.2 here but is gfx90a, so this path yields MI300's target
        // instead and resolves to the wrong TheRock family — the right wheel for
        // this host is `gfx90a`. Pre-existing: the decimal encoding produced
        // `gfx942` for 9/4/2 too, so the hex fix neither caused nor closes it.
        // Fixing it needs a GC-IP-version → target table for GC 9.4.x
        // (9.4.0/9.4.1/9.4.3 are gfx906/gfx908/gfx942), not a different encoding.
        fs::write(gc.join("major"), "9")?;
        fs::write(gc.join("minor"), "4")?;
        fs::write(gc.join("revision"), "2")?;
        let aldebaran = detect_ip_discovery_gc_target(&root.join("ip_discovery"));
        assert_eq!(aldebaran, Some("gfx942".to_owned()));
        assert_eq!(
            normalize_therock_family(&aldebaran.expect("target")),
            Some("gfx94X-dcgpu".to_owned())
        );

        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn linux_amdgpu_device_fixture_accepts_vendor_or_uevent_driver() -> Result<()> {
        let (root, _paths) = temp_app_paths("linux-amdgpu-device");
        let vendor = root.join("vendor");
        fs::create_dir_all(&root)?;
        fs::write(&vendor, "0x1002\n")?;
        assert!(is_amdgpu_device(&root));
        fs::remove_file(&vendor)?;
        fs::write(root.join("uevent"), "DRIVER=amdgpu\n")?;
        assert!(is_amdgpu_device(&root));
        fs::write(root.join("uevent"), "DRIVER=i915\n")?;
        assert!(!is_amdgpu_device(&root));
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn kfd_topology_names_the_gpu_when_no_tooling_is_installed() -> Result<()> {
        // Covers the standalone-file fallback; see
        // `kfd_topology_reads_the_target_from_the_properties_file` for the
        // layout a real KFD host has. `Examination` enumerates GPUs by shelling out to
        // lspci and rocminfo; where neither is reachable it reported no AMD GPU
        // on a machine that has one, while the human report -- which reads this
        // topology instead -- named the target correctly. Planted here because
        // the hosts where it matters are the ones a test cannot run on.
        let (root, _) = temp_app_paths("kfd-topology");
        let nodes = root.join("nodes");
        fs::create_dir_all(nodes.join("node0"))?;
        fs::write(nodes.join("node0").join("gfx_target_version"), "90402\n")?;

        let found = detect_kfd_gfx_target_in(&nodes);
        fs::remove_dir_all(&root).ok();

        assert_eq!(found.as_deref(), Some("gfx942"));
        Ok(())
    }

    #[test]
    fn kfd_topology_answer_does_not_depend_on_directory_order() -> Result<()> {
        // `read_dir` order is filesystem-defined, so a multi-node box could name
        // a different GPU run to run. node9 must not beat node10 lexically
        // either -- the lowest-numbered node is the one HIP ordinal 0 means.
        let (root, _) = temp_app_paths("kfd-topology-order");
        let nodes = root.join("nodes");
        for (node, version) in [("node10", "110100"), ("node9", "90402"), ("node0", "90400")] {
            fs::create_dir_all(nodes.join(node))?;
            fs::write(nodes.join(node).join("gfx_target_version"), version)?;
        }

        let found = detect_kfd_gfx_target_in(&nodes);
        fs::remove_dir_all(&root).ok();

        assert_eq!(found.as_deref(), Some("gfx940"));
        Ok(())
    }

    #[test]
    fn kfd_topology_skips_nodes_that_name_no_target() -> Result<()> {
        // A CPU node carries a zero version. Reporting `gfx0` off the first
        // directory encountered would be worse than reporting nothing.
        let (root, _) = temp_app_paths("kfd-topology-cpu-node");
        let nodes = root.join("nodes");
        fs::create_dir_all(nodes.join("node0"))?;
        fs::write(nodes.join("node0").join("gfx_target_version"), "0\n")?;
        fs::create_dir_all(nodes.join("node1"))?;
        fs::write(nodes.join("node1").join("gfx_target_version"), "110000\n")?;

        let found = detect_kfd_gfx_target_in(&nodes);
        fs::remove_dir_all(&root).ok();

        assert_eq!(found.as_deref(), Some("gfx1100"));
        Ok(())
    }

    #[test]
    fn kfd_topology_reads_the_target_from_the_properties_file() -> Result<()> {
        // The layout every real KFD host actually has: the value is a line in
        // the node's `properties`, and there is no standalone file. Reading only
        // the standalone path found nothing here, so an MI300X fell through to
        // ip-discovery and was reported as `gfx943` while KFD said `90402`.
        // Node directories are bare integers on real hardware, not `nodeN`.
        let (root, _) = temp_app_paths("kfd-topology-properties");
        let nodes = root.join("nodes");
        // Trimmed from a live MI300X; ordering and neighbours are as found.
        for (node, properties) in [
            (
                "0",
                "cpu_cores_count 56\nsimd_count 0\ngfx_target_version 0\n",
            ),
            (
                "5",
                "cpu_cores_count 0\nsimd_count 1216\nmax_waves_per_simd 8\ngfx_target_version 90402\n",
            ),
        ] {
            fs::create_dir_all(nodes.join(node))?;
            fs::write(nodes.join(node).join("properties"), properties)?;
        }

        let found = detect_kfd_gfx_target_in(&nodes);
        fs::remove_dir_all(&root).ok();

        assert_eq!(found.as_deref(), Some("gfx942"));
        Ok(())
    }

    #[test]
    fn kfd_gpu_node_count_reads_properties_and_skips_cpu_nodes() -> Result<()> {
        // The count used the same phantom standalone file, so it saw zero GPUs
        // on every real host and let the DRM card count stand in for it. Two
        // CPU nodes and two GPU nodes, shaped like a live Instinct topology.
        let (root, _) = temp_app_paths("kfd-node-count-properties");
        let nodes = root.join("nodes");
        for (node, properties) in [
            ("0", "cpu_cores_count 56\ngfx_target_version 0\n"),
            ("1", "cpu_cores_count 56\ngfx_target_version 0\n"),
            ("5", "simd_count 1216\ngfx_target_version 90402\n"),
            ("6", "simd_count 1216\ngfx_target_version 90402\n"),
        ] {
            fs::create_dir_all(nodes.join(node))?;
            fs::write(nodes.join(node).join("properties"), properties)?;
        }

        let count = linux_kfd_gpu_node_count_in(&nodes);
        fs::remove_dir_all(&root).ok();

        assert_eq!(count, Some(2));
        Ok(())
    }

    #[test]
    fn kfd_nodes_carry_a_decoded_pci_address_and_their_own_target() -> Result<()> {
        // The first three GPU nodes of a real 8-GPU MI300X host, verbatim from
        // its `location_id` values, behind the two CPU nodes it also reports.
        // `lspci -D` lists those cards at 0000:11:00.0, 0000:2f:00.0 and
        // 0000:46:00.0, so the decode is checked against the machine rather
        // than against itself.
        let (root, _) = temp_app_paths("kfd-nodes-pci");
        let nodes = root.join("nodes");
        for node in ["0", "1"] {
            fs::create_dir_all(nodes.join(node))?;
            fs::write(
                nodes.join(node).join("properties"),
                "cpu_cores_count 56\nsimd_count 0\ngfx_target_version 0\nlocation_id 0\ndomain 0\n",
            )?;
        }
        for (node, location) in [("2", 4352), ("3", 12032), ("4", 17920)] {
            fs::create_dir_all(nodes.join(node))?;
            fs::write(
                nodes.join(node).join("properties"),
                format!(
                    "cpu_cores_count 0\nsimd_count 1216\ngfx_target_version 90402\n\
                     location_id {location}\ndomain 0\n"
                ),
            )?;
        }
        let read = kfd_gpu_nodes_in(&nodes);
        fs::remove_dir_all(&root).ok();

        let expected: Vec<KfdGpuNode> = ["0000:11:00.0", "0000:2f:00.0", "0000:46:00.0"]
            .into_iter()
            .map(|pci_id| KfdGpuNode {
                pci_id: pci_id.to_owned(),
                gfx_target: "gfx942".to_owned(),
            })
            .collect();
        // The CPU nodes are excluded, and node order is preserved so entry 0 is
        // the device HIP calls ordinal 0.
        assert_eq!(read, Some(expected));
        Ok(())
    }

    #[test]
    fn kfd_nodes_keep_their_targets_apart_and_refuse_an_unusable_address() -> Result<()> {
        // An APU + a discrete card. Each node states its own target, which is
        // the whole point of reading them per-node: a single host-wide answer
        // would stamp the APU's gfx1103 onto the discrete card.
        //
        // The dGPU node here also reports `location_id 0`. That decodes to
        // 0000:00:00.0 -- the host bridge -- so it must come back empty rather
        // than as an address that could match an unrelated PCI entry.
        let (root, _) = temp_app_paths("kfd-nodes-mixed");
        let nodes = root.join("nodes");
        fs::create_dir_all(nodes.join("0"))?;
        fs::write(
            nodes.join("0").join("properties"),
            "cpu_cores_count 16\ngfx_target_version 0\nlocation_id 0\ndomain 0\n",
        )?;
        fs::create_dir_all(nodes.join("1"))?;
        fs::write(
            nodes.join("1").join("properties"),
            "simd_count 256\ngfx_target_version 110003\nlocation_id 25600\ndomain 0\n",
        )?;
        fs::create_dir_all(nodes.join("2"))?;
        fs::write(
            nodes.join("2").join("properties"),
            "simd_count 768\ngfx_target_version 110000\nlocation_id 0\ndomain 0\n",
        )?;
        let read = kfd_gpu_nodes_in(&nodes);
        // The host-wide answer is still the APU, which is why it must not be
        // attributed to the discrete card.
        let lowest = detect_kfd_gfx_target_in(&nodes);
        fs::remove_dir_all(&root).ok();

        assert_eq!(
            read,
            Some(vec![
                KfdGpuNode {
                    pci_id: "0000:64:00.0".to_owned(),
                    gfx_target: "gfx1103".to_owned(),
                },
                KfdGpuNode {
                    pci_id: String::new(),
                    gfx_target: "gfx1100".to_owned(),
                },
            ])
        );
        assert_eq!(lowest.as_deref(), Some("gfx1103"));
        Ok(())
    }

    #[test]
    fn a_kfd_node_decodes_a_nonzero_function_and_domain() {
        // devfn packs the device in bits 3..8 and the function in bits 0..3, and
        // the domain is a separate property -- so neither is assumed to be zero.
        // 0x8ffa = bus 0x8f, device 0x1f, function 2.
        assert_eq!(
            kfd_node_pci_id("location_id 36858\ndomain 5\n").as_deref(),
            Some("0005:8f:1f.2")
        );
        // A node that states no location at all cannot be placed on the bus.
        assert_eq!(kfd_node_pci_id("gfx_target_version 90402\n"), None);
    }

    #[test]
    fn kfd_properties_win_over_a_standalone_file() -> Result<()> {
        // Both present: `properties` is what the kernel maintains, so it decides.
        let (root, _) = temp_app_paths("kfd-topology-properties-precedence");
        let nodes = root.join("nodes");
        fs::create_dir_all(nodes.join("0"))?;
        fs::write(
            nodes.join("0").join("properties"),
            "gfx_target_version 90402\n",
        )?;
        fs::write(nodes.join("0").join("gfx_target_version"), "110000\n")?;

        let found = detect_kfd_gfx_target_in(&nodes);
        fs::remove_dir_all(&root).ok();

        assert_eq!(found.as_deref(), Some("gfx942"));
        Ok(())
    }

    #[test]
    fn kfd_topology_bare_integer_node_order_is_numeric_not_lexical() -> Result<()> {
        // Real KFD node directories are bare integers, and the only ordering that
        // separates numeric from lexical sorting is a multi-digit one: `10` must
        // not come before `2`. The other topology tests use single digits, which
        // sort the same either way, so this is the case that pins the behaviour.
        let (root, _) = temp_app_paths("kfd-topology-bare-integer-order");
        let nodes = root.join("nodes");
        for (node, version) in [("10", "110100\n"), ("2", "90402\n")] {
            fs::create_dir_all(nodes.join(node))?;
            fs::write(
                nodes.join(node).join("properties"),
                format!("gfx_target_version {version}"),
            )?;
        }

        let found = detect_kfd_gfx_target_in(&nodes);
        fs::remove_dir_all(&root).ok();

        // Node 2 is the lowest-numbered GPU node, so it is the one HIP ordinal 0
        // refers to.
        assert_eq!(found.as_deref(), Some("gfx942"));
        Ok(())
    }

    #[test]
    fn kfd_property_value_reads_only_its_own_key() {
        let body = "simd_count 1216\ngfx_target_version 90402\nnum_gws 64\n";
        assert_eq!(
            kfd_property_value(body, "gfx_target_version").as_deref(),
            Some("90402")
        );
        assert_eq!(
            kfd_property_value(body, "simd_count").as_deref(),
            Some("1216")
        );
        // A key that only appears as a prefix of another must not match, and an
        // absent key yields nothing rather than an empty string.
        assert_eq!(kfd_property_value(body, "gfx_target"), None);
        assert_eq!(kfd_property_value(body, "vram_size"), None);
    }

    #[test]
    fn absent_kfd_topology_names_nothing_rather_than_guessing() {
        let missing = workspace_test_artifact_dir().join("rocm-core-kfd-absent-nodes");
        fs::remove_dir_all(&missing).ok();
        assert_eq!(detect_kfd_gfx_target_in(&missing), None);
    }

    #[test]
    fn kfd_gfx_target_version_distinguishes_gpu_from_cpu_nodes() {
        // CPU topology nodes report a zero gfx target version; GPU nodes report a
        // nonzero one.
        assert!(!kfd_gfx_target_version_is_gpu("0"));
        assert!(!kfd_gfx_target_version_is_gpu(""));
        assert!(!kfd_gfx_target_version_is_gpu("not-a-number"));
        assert!(kfd_gfx_target_version_is_gpu("90402"));
        assert!(kfd_gfx_target_version_is_gpu("110000"));
    }

    #[test]
    fn a_ready_wsl_host_offers_a_device_and_an_unready_one_does_not() {
        // The EAI-7944 shape. `serve` used to count KFD nodes and DRM cards,
        // neither of which exists on WSL2, and read the result as an
        // authoritative zero -- refusing to launch on a machine whose own
        // `examine` reported the GPU ready.
        assert_eq!(
            usable_amd_gpu_indices_from(usize::from(true), None),
            Some(vec![0])
        );
        assert_eq!(
            usable_amd_gpu_indices_from(usize::from(false), None),
            Some(vec![])
        );
        // An explicit empty mask still wins, so a user can opt out on WSL as
        // anywhere — HIP_VISIBLE_DEVICES="" hides the device.
        assert_eq!(
            usable_amd_gpu_indices_from(usize::from(true), hip_mask("")),
            Some(vec![])
        );
    }

    #[test]
    fn combine_amd_gpu_counts_prefers_compute_authoritative_kfd() {
        // KFD is compute-authoritative: a nonzero KFD count wins, and DRM must not
        // raise it. A display/render-only AMD DRM card (KFD=1, DRM=2) must NOT
        // invent a second usable HIP ordinal, or an explicit `--gpu 1` would pass
        // validation and then fail inside HIP.
        assert_eq!(combine_amd_gpu_counts(Some(1), Some(2)), Some(1));
        // Same principle with more DRM cards: still bounded by the KFD count.
        assert_eq!(combine_amd_gpu_counts(Some(2), Some(8)), Some(2));
        // KFD larger than DRM (e.g. multi-partition compute nodes): KFD still wins.
        assert_eq!(combine_amd_gpu_counts(Some(3), Some(1)), Some(3));
        // Strix Halo shape: KFD reports 0 GPU nodes, DRM sees the iGPU → 1 present.
        assert_eq!(combine_amd_gpu_counts(Some(0), Some(1)), Some(1));
        // Discrete GPUs: both surfaces agree.
        assert_eq!(combine_amd_gpu_counts(Some(8), Some(8)), Some(8));
        // One surface unreadable → use the other.
        assert_eq!(combine_amd_gpu_counts(None, Some(1)), Some(1));
        assert_eq!(combine_amd_gpu_counts(Some(2), None), Some(2));
        // Neither readable → unknown (caller must not treat as zero).
        assert_eq!(combine_amd_gpu_counts(None, None), None);
        // Both agree on zero → authoritative no-GPU.
        assert_eq!(combine_amd_gpu_counts(Some(0), Some(0)), Some(0));
    }

    #[test]
    fn usable_gpu_indices_unset_mask_returns_all_present_devices() {
        assert_eq!(usable_amd_gpu_indices_from(0, None), Some(Vec::new()));
        assert_eq!(usable_amd_gpu_indices_from(1, None), Some(vec![0]));
        assert_eq!(usable_amd_gpu_indices_from(3, None), Some(vec![0, 1, 2]));
    }

    /// Only `HIP_VISIBLE_DEVICES` set: its ordinals are already in HIP space and
    /// are used as-is.
    fn hip_mask(value: &str) -> Option<GpuVisibilityMask> {
        Some(GpuVisibilityMask {
            rocr: None,
            hip: Some(value.to_owned()),
        })
    }

    /// Only `ROCR_VISIBLE_DEVICES` set: its physical ordinals are re-indexed into
    /// HIP space (survivors become `0..N`).
    fn rocr_mask(value: &str) -> Option<GpuVisibilityMask> {
        Some(GpuVisibilityMask {
            rocr: Some(value.to_owned()),
            hip: None,
        })
    }

    /// Both variables set. ROCr applies first and HIP re-indexes the survivors,
    /// so `hip`'s tokens are ordinals *within* `rocr`'s survivor list.
    fn rocr_then_hip_mask(rocr: &str, hip: &str) -> Option<GpuVisibilityMask> {
        Some(GpuVisibilityMask {
            rocr: Some(rocr.to_owned()),
            hip: Some(hip.to_owned()),
        })
    }

    #[test]
    fn usable_gpu_indices_empty_mask_hides_every_device() {
        // The masked-device path: GPUs are present but fully masked out.
        assert_eq!(
            usable_amd_gpu_indices_from(2, hip_mask("")),
            Some(Vec::new())
        );
        // An empty ROCR mask hides every device too.
        assert_eq!(
            usable_amd_gpu_indices_from(2, rocr_mask("")),
            Some(Vec::new())
        );
    }

    #[test]
    fn usable_gpu_indices_honors_valid_ordinal_masks() {
        assert_eq!(
            usable_amd_gpu_indices_from(4, hip_mask("2,0")),
            Some(vec![2, 0])
        );
        // Duplicates are collapsed.
        assert_eq!(
            usable_amd_gpu_indices_from(2, hip_mask("1,1")),
            Some(vec![1])
        );
    }

    #[test]
    fn usable_gpu_indices_treats_unsupported_masks_as_unprobeable() {
        assert_eq!(usable_amd_gpu_indices_from(2, hip_mask("0,5")), None);
        assert_eq!(
            usable_amd_gpu_indices_from(2, hip_mask("GPU-deadbeef")),
            None
        );
    }

    #[test]
    fn a_hip_mask_keeps_its_ordinals_but_a_rocr_mask_is_reindexed_to_hip_space() {
        // A HIP_VISIBLE_DEVICES mask is already in the HIP-ordinal space rocm-cli
        // exports through, so its tokens are used as-is.
        assert_eq!(
            usable_amd_gpu_indices_from(4, hip_mask("2,3")),
            Some(vec![2, 3])
        );
        // A ROCR_VISIBLE_DEVICES mask hides physical devices below HIP, which then
        // re-indexes the survivors as 0..N. On a 4-GPU host, ROCR=2,3 leaves two
        // devices that HIP sees as ordinals 0 and 1 — the values that actually
        // bind when exported via HIP_VISIBLE_DEVICES — not the physical 2 and 3.
        assert_eq!(
            usable_amd_gpu_indices_from(4, rocr_mask("2,3")),
            Some(vec![0, 1])
        );
        // A single-device ROCR mask re-indexes to just ordinal 0.
        assert_eq!(
            usable_amd_gpu_indices_from(4, rocr_mask("3")),
            Some(vec![0])
        );
        // An out-of-range token is still "cannot interpret", regardless of source.
        assert_eq!(usable_amd_gpu_indices_from(2, rocr_mask("5")), None);
    }

    #[test]
    fn a_hip_mask_under_a_rocr_mask_is_bounded_by_the_rocr_survivors() {
        // EAI-7194, second half: both variables set. ROCr applies first and HIP
        // re-indexes the survivors as 0..N, so HIP tokens are ordinals within that
        // reduced set — NOT physical ordinals. Range-checking them against the
        // physical `present` accepted ordinals that cannot bind.
        //
        // 4 GPUs present, ROCR=2,3 leaves two devices HIP numbers 0 and 1.
        // HIP=3 names nothing HIP can see, even though 3 < 4 physically: the probe
        // cannot resolve the visible set and must say "unknown", not confidently
        // hand back [3] for `--gpu 3` to be accepted against and `--gpu auto` to be
        // steered onto.
        assert_eq!(
            usable_amd_gpu_indices_from(4, rocr_then_hip_mask("2,3", "3")),
            None
        );
        // The same shape one ordinal lower is a real device: HIP ordinal 1 is the
        // second ROCr survivor (physical 3), and it stays selectable.
        assert_eq!(
            usable_amd_gpu_indices_from(4, rocr_then_hip_mask("2,3", "1")),
            Some(vec![1])
        );
        // Selecting every survivor keeps both re-indexed ordinals.
        assert_eq!(
            usable_amd_gpu_indices_from(4, rocr_then_hip_mask("2,3", "1,0")),
            Some(vec![1, 0])
        );
        // A one-device ROCr set leaves only HIP ordinal 0; ordinal 1 is unknown,
        // not physical ordinal 1.
        assert_eq!(
            usable_amd_gpu_indices_from(4, rocr_then_hip_mask("3", "0")),
            Some(vec![0])
        );
        assert_eq!(
            usable_amd_gpu_indices_from(4, rocr_then_hip_mask("3", "1")),
            None
        );
        // Either variable hiding everything is authoritative: an empty ROCR mask
        // leaves HIP nothing to select, so a HIP mask cannot resurrect a device.
        assert_eq!(
            usable_amd_gpu_indices_from(4, rocr_then_hip_mask("", "0")),
            Some(Vec::new())
        );
        assert_eq!(
            usable_amd_gpu_indices_from(4, rocr_then_hip_mask("2,3", "")),
            Some(Vec::new())
        );
        // An uninterpretable ROCR mask is still "unknown" whatever HIP says.
        assert_eq!(
            usable_amd_gpu_indices_from(4, rocr_then_hip_mask("GPU-deadbeef", "0")),
            None
        );
        assert_eq!(
            usable_amd_gpu_indices_from(2, rocr_then_hip_mask("5", "0")),
            None
        );
    }
    #[test]
    fn default_engine_is_always_usable_on_windows() {
        if cfg!(windows) {
            assert_eq!(default_engine_for_platform(), "lemonade");
        }
    }

    #[test]
    fn instinct_dcgpu_family_prefers_vllm() {
        // On Instinct data-center GPUs (TheRock `*-dcgpu` families, e.g. the
        // MI300X's gfx94X-dcgpu) the default serving engine is vLLM. This is the
        // GPU-family preference the serve engine selection honors before falling
        // back to a recipe's own preferred engine. vLLM is Linux-only, so the
        // preference does not apply on native Windows.
        let summary = HostGpuSummary {
            name: Some("AMD Instinct MI300X".to_owned()),
            gfx_target: Some("gfx942".to_owned()),
            therock_family: Some("gfx94X-dcgpu".to_owned()),
        };
        let preferred = preferred_serve_engine_for_host_gpu_summary(&summary);
        if cfg!(windows) {
            assert_eq!(preferred, None, "vLLM is not preferred on native Windows");
        } else {
            assert_eq!(preferred, Some("vllm"));
        }
    }

    #[test]
    fn host_default_engine_is_vllm_on_instinct() {
        // What `rocm examine` and `rocm engines list` report on an MI300X. The
        // platform constant said "lemonade" here while serve picked vLLM, so the
        // reported default contradicted the actual behaviour.
        let summary = HostGpuSummary {
            name: Some("AMD Instinct MI300X".to_owned()),
            gfx_target: Some("gfx942".to_owned()),
            therock_family: Some("gfx94X-dcgpu".to_owned()),
        };
        if cfg!(windows) {
            assert_eq!(default_engine_for_host(&summary), "lemonade");
        } else {
            assert_eq!(default_engine_for_host(&summary), "vllm");
        }
    }

    #[test]
    fn host_default_engine_covers_every_vllm_preferred_family() {
        // Guards the whole preferred set, not just the dcgpu branch, so adding a
        // family to VLLM_PREFERRED_THEROCK_FAMILIES cannot leave the reported
        // default behind.
        for family in VLLM_PREFERRED_THEROCK_FAMILIES {
            let summary = HostGpuSummary {
                name: None,
                gfx_target: Some((*family).to_owned()),
                therock_family: Some((*family).to_owned()),
            };
            let expected = if cfg!(windows) { "lemonade" } else { "vllm" };
            assert_eq!(
                default_engine_for_host(&summary),
                expected,
                "unexpected default for {family}"
            );
        }
    }

    #[test]
    fn host_default_engine_is_lemonade_without_a_vllm_preference() {
        // Strix Halo (gfx1151), a consumer family, and a machine whose GPU has not
        // been identified at all must all keep the platform default.
        for summary in [
            HostGpuSummary {
                name: Some("AMD Radeon 8060S".to_owned()),
                gfx_target: Some("gfx1151".to_owned()),
                therock_family: Some("gfx1151".to_owned()),
            },
            HostGpuSummary {
                name: Some("AMD Radeon".to_owned()),
                gfx_target: Some("gfx1100".to_owned()),
                therock_family: Some("gfx110X-all".to_owned()),
            },
            HostGpuSummary::default(),
        ] {
            assert_eq!(default_engine_for_host(&summary), "lemonade");
        }
    }

    #[test]
    fn host_default_engine_never_reports_vllm_on_native_windows() {
        // The vLLM adapter bails on native Windows, so no GPU may talk the
        // reported default into vLLM there — including an Instinct part.
        if !cfg!(windows) {
            return;
        }
        let summary = HostGpuSummary {
            name: Some("AMD Instinct MI300X".to_owned()),
            gfx_target: Some("gfx942".to_owned()),
            therock_family: Some("gfx94X-dcgpu".to_owned()),
        };
        assert_eq!(default_engine_for_host(&summary), "lemonade");
    }

    #[test]
    fn consumer_gpu_family_has_no_vllm_preference() {
        // A non-dcgpu consumer family (e.g. gfx110X-all) has no GPU-level vLLM
        // preference, so serve selection falls through to the recipe/platform
        // default rather than forcing vLLM.
        let summary = HostGpuSummary {
            name: Some("AMD Radeon".to_owned()),
            gfx_target: Some("gfx1100".to_owned()),
            therock_family: Some("gfx110X-all".to_owned()),
        };
        assert_eq!(preferred_serve_engine_for_host_gpu_summary(&summary), None);
    }

    #[test]
    fn preferred_serve_engine_uses_vllm_for_supported_therock_families() {
        assert_eq!(
            preferred_serve_engine_for_therock_family(Some("gfx90a")),
            Some("vllm")
        );
        assert_eq!(
            preferred_serve_engine_for_therock_family(Some("gfx950")),
            Some("vllm")
        );
        assert_eq!(
            preferred_serve_engine_for_therock_family(Some("gfx999-dcgpu")),
            Some("vllm")
        );
        assert_eq!(preferred_serve_engine_for_therock_family(None), None);
    }

    #[test]
    fn preferred_serve_engine_host_summary_respects_platform_and_fields() {
        // `gfx_target` is consulted as a fallback when `therock_family` is absent.
        let summary = HostGpuSummary {
            gfx_target: Some("gfx950".to_owned()),
            ..HostGpuSummary::default()
        };
        // The vLLM adapter is unsupported on native Windows, so the preference is
        // gated off there while remaining active on Linux/WSL builds.
        let expected = if cfg!(windows) { None } else { Some("vllm") };
        assert_eq!(
            preferred_serve_engine_for_host_gpu_summary(&summary),
            expected
        );

        // No GPU information never resolves to a vLLM preference on any platform.
        assert_eq!(
            preferred_serve_engine_for_host_gpu_summary(&HostGpuSummary::default()),
            None
        );
    }

    #[test]
    fn windows_display_name_cleaner_removes_inf_resource_prefix() {
        assert_eq!(
            clean_windows_display_name("@oem40.inf,%amd7550.23%;AMD Radeon RX 9070 XT"),
            "AMD Radeon RX 9070 XT"
        );
        assert_eq!(
            clean_windows_display_name("AMD Radeon RX 9070 XT"),
            "AMD Radeon RX 9070 XT"
        );
    }

    #[test]
    fn windows_examine_inventory_parser_feeds_cpu_driver_and_gfx_detection() {
        let inventory = parse_windows_examine_inventory(
            "CPU\t  AMD Ryzen 9 9950X  16-Core Processor  \nRAM\t68719476736\nGPU\tAMD Radeon RX 9070 XT\t32.0.13031.9001\tPCI\\VEN_1002&DEV_7550&SUBSYS_2435148C&REV_C0\n",
        );

        assert_eq!(
            inventory.cpu_model.as_deref(),
            Some("AMD Ryzen 9 9950X 16-Core Processor")
        );
        assert_eq!(inventory.system_ram_gib, Some(64.0));
        assert_eq!(
            inventory.amd_display_driver_detail().as_deref(),
            Some("AMD Radeon RX 9070 XT driver 32.0.13031.9001")
        );
        assert_eq!(inventory.display_gfx_target(), Some("gfx1201".to_owned()));
    }

    #[test]
    fn windows_pnputil_inventory_parser_detects_780m_device_id() {
        let inventory = parse_windows_pnputil_display_inventory(
            "\
Instance ID:                PCI\\VEN_1002&DEV_15BF&SUBSYS_15021025&REV_C1\\4&2F6D7E4A&0&0041
Device Description:        AMD Radeon 780M Graphics
Class Name:                Display
Class GUID:                {4d36e968-e325-11ce-bfc1-08002be10318}
Manufacturer Name:         Advanced Micro Devices, Inc.
Status:                    Started
Driver Name:               oem42.inf
",
        );

        assert_eq!(
            inventory.amd_display_name().as_deref(),
            Some("AMD Radeon 780M Graphics")
        );
        assert_eq!(inventory.display_gfx_target(), Some("gfx1103".to_owned()));
    }

    #[test]
    fn windows_pnputil_inventory_parser_ignores_non_amd_display() {
        let inventory = parse_windows_pnputil_display_inventory(
            "\
Instance ID:                PCI\\VEN_8086&DEV_9A49&SUBSYS_00000000
Device Description:        Intel UHD Graphics
Class Name:                Display
",
        );

        assert!(inventory.displays.is_empty());
    }

    #[test]
    fn windows_examine_inventory_prefers_real_gpu_over_noisy_amd_pnp_entries() {
        let inventory = parse_windows_examine_inventory(
            "GPU\tAMD Bluetooth Capture Audio Device\t\t{2101C4C0-2C15-4035-A0D0-EEC3C2277B11}\\CAPTURE&CP_111215637\nGPU\tAMD-OpenGL User Mode Driver\t\tSWD\\DRIVERENUM\\AMDOGL&5&BAA66E4&0\nGPU\tAMD Radeon 780M Graphics\t\tPCI\\VEN_1002&DEV_1900&SUBSYS_50EE17AA&REV_D0\\4&EB5E2B6&0&0041\n",
        );

        assert_eq!(
            inventory.amd_display_name().as_deref(),
            Some("AMD Radeon 780M Graphics")
        );
        assert_eq!(inventory.display_gfx_target(), Some("gfx1103".to_owned()));
    }

    #[test]
    fn counts_json_files_and_model_cache_entries_for_examine() -> Result<()> {
        let (root, paths) = temp_app_paths("examine-counts");
        let registry = paths.data_dir.join("runtimes").join("registry");
        let models = paths.data_dir.join("models");
        fs::create_dir_all(&registry)?;
        fs::create_dir_all(&models)?;
        fs::write(registry.join("runtime-a.json"), "{}")?;
        fs::write(registry.join("runtime-b.json"), "{}")?;
        fs::write(registry.join("notes.txt"), "skip")?;
        fs::create_dir_all(models.join("hf"))?;
        fs::write(models.join("local.bin"), "model")?;

        assert_eq!(count_json_files(&registry), 2);
        assert_eq!(count_dir_entries(&models), 2);
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn examine_render_includes_driver_and_state_counts() {
        let summary = ExamineSummary {
            os: "windows".to_owned(),
            arch: "x86_64".to_owned(),
            kernel: Some("10.0.26100".to_owned()),
            distro: Some("Windows".to_owned()),
            cpu: Some("AMD Ryzen".to_owned()),
            system_ram_gib: Some(64.0),
            interactive_terminal: false,
            default_engine: "vllm".to_owned(),
            detected_gfx_target: None,
            compatible_therock_family: Some("gfx120X-all".to_owned()),
            detected_therock_family: None,
            driver: DriverSummary {
                policy: "windows_validate_only".to_owned(),
                status: "amd_display_driver_detected".to_owned(),
                detail: Some("AMD Radeon driver 1.2.3".to_owned()),
            },
            legacy_rocm: LegacyRocmSummary {
                status: "detected_unmanaged".to_owned(),
                paths: vec![PathBuf::from("C:\\Program Files\\AMD\\ROCm")],
                detail: Some("legacy install".to_owned()),
                version: Some("6.4.1".to_owned()),
            },
            wsl: None,
            managed_runtime_count: 2,
            managed_service_count: 1,
            model_cache_entries: 3,
            config_dir: PathBuf::from("config"),
            data_dir: PathBuf::from("data"),
            cache_dir: PathBuf::from("cache"),
        };

        let rendered = summary.render_text();
        assert!(rendered.contains("distro: Windows"));
        assert!(rendered.contains("cpu: AMD Ryzen"));
        assert!(rendered.contains("system_ram: 64 GiB"));
        assert!(rendered.contains("compatible_therock_family: gfx120X-all"));
        assert!(rendered.contains("detected_therock_family: <not detected>"));
        assert!(rendered.contains("driver_policy: windows_validate_only"));
        assert!(rendered.contains("driver_status: amd_display_driver_detected"));
        assert!(rendered.contains("legacy_rocm_status: detected_unmanaged"));
        assert!(rendered.contains("legacy_rocm_paths: C:\\Program Files\\AMD\\ROCm"));
        assert!(
            rendered.contains("legacy_rocm_guidance: legacy ROCm detected; keep it side-by-side")
        );
        assert!(rendered.contains("wsl: false"));
        assert!(rendered.contains("managed_runtimes: 2"));
        assert!(rendered.contains("managed_services: 1"));
        assert!(rendered.contains("model_cache_entries: 3"));
    }

    #[test]
    fn examine_render_explains_what_interactive_terminal_means() {
        // The same machine reports `true` from a shell and `false` under the
        // dashboard, because the field describes the invocation rather than the
        // host. Both are correct, and the line has to say so on its own — a
        // pasted report is usually all a reader gets.
        let mut summary = ExamineSummary {
            os: "linux".to_owned(),
            arch: "x86_64".to_owned(),
            kernel: None,
            distro: None,
            cpu: None,
            system_ram_gib: None,
            interactive_terminal: true,
            default_engine: "vllm".to_owned(),
            detected_gfx_target: None,
            compatible_therock_family: None,
            detected_therock_family: None,
            driver: DriverSummary {
                policy: "linux_official_amd_dkms_wrapper".to_owned(),
                status: "amdgpu_available".to_owned(),
                detail: None,
            },
            legacy_rocm: LegacyRocmSummary {
                status: "not_detected".to_owned(),
                paths: Vec::new(),
                detail: None,
                version: None,
            },
            wsl: None,
            managed_runtime_count: 0,
            managed_service_count: 0,
            model_cache_entries: 0,
            config_dir: PathBuf::from("config"),
            data_dir: PathBuf::from("data"),
            cache_dir: PathBuf::from("cache"),
        };

        let interactive = summary.render_text();
        assert!(
            interactive.contains("interactive_terminal: true (this run has a terminal"),
            "the true case must say it is about this run:\n{interactive}"
        );

        summary.interactive_terminal = false;
        let captured = summary.render_text();
        assert!(
            captured.contains("interactive_terminal: false (this run's output is captured"),
            "the false case must explain why, not just report it:\n{captured}"
        );
        // The reason it matters to the reader: it is why they saw no prompt.
        assert!(
            captured.contains("will not prompt"),
            "the false case must connect to the visible consequence:\n{captured}"
        );
    }

    #[test]
    fn examine_render_guides_managed_runtime_install_when_only_legacy_rocm_exists() {
        let summary = ExamineSummary {
            os: "linux".to_owned(),
            arch: "x86_64".to_owned(),
            kernel: None,
            distro: None,
            cpu: None,
            system_ram_gib: None,
            interactive_terminal: false,
            default_engine: "vllm".to_owned(),
            detected_gfx_target: None,
            compatible_therock_family: None,
            detected_therock_family: None,
            driver: DriverSummary {
                policy: "linux_official_amd_dkms_wrapper".to_owned(),
                status: "amdgpu_available".to_owned(),
                detail: None,
            },
            legacy_rocm: LegacyRocmSummary {
                status: "detected_unmanaged".to_owned(),
                paths: vec![PathBuf::from("/opt/rocm")],
                detail: Some("legacy install".to_owned()),
                version: Some("7.14.0".to_owned()),
            },
            wsl: None,
            managed_runtime_count: 0,
            managed_service_count: 0,
            model_cache_entries: 0,
            config_dir: PathBuf::from("config"),
            data_dir: PathBuf::from("data"),
            cache_dir: PathBuf::from("cache"),
        };

        let rendered = summary.render_text();

        assert!(rendered.contains(
            "legacy_rocm_guidance: legacy ROCm detected; install a managed TheRock runtime"
        ));
        assert!(rendered.contains("rocm install sdk --channel release --format wheel"));
    }

    #[test]
    fn wsl_driver_summary_reports_missing_rocdxg_without_amdgpu_fallback() {
        let summary = WslSummary {
            is_wsl: true,
            dxg_device: true,
            dxcore: true,
            librocdxg: false,
            rocdxg_dids: false,
            ldconfig_librocdxg: false,
            rocminfo: false,
            cargo: true,
            detail: Some("missing /opt/rocm/lib/librocdxg.so".to_owned()),
        };

        let driver = wsl_driver_summary(&summary);

        assert_eq!(driver.policy, "wsl_rocdxg");
        assert_eq!(driver.status, "wsl_rocdxg_missing");
        assert!(
            driver
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("librocdxg"))
        );
    }

    #[test]
    fn parses_os_release_pretty_name() {
        assert_eq!(
            parse_os_release_pretty_name("NAME=Ubuntu\nPRETTY_NAME=\"Ubuntu 24.04.2 LTS\"\n"),
            Some("Ubuntu 24.04.2 LTS".to_owned())
        );
    }

    #[test]
    fn dev_dxg_is_believed_on_its_own() {
        // Nothing but WSLg's GPU passthrough creates this device node, so it is
        // trusted without corroboration.
        assert!(wsl_signals_indicate_wsl(true, ""), "/dev/dxg");
        assert!(
            wsl_signals_indicate_wsl(true, "Linux version 6.8.0-51-generic"),
            "/dev/dxg overrides an otherwise ordinary kernel string"
        );
    }

    #[test]
    fn proc_version_alone_is_believed() {
        // Only a WSL kernel is built `-microsoft-standard[-WSL2]` or
        // `-Microsoft` -- unlike $WSL_DISTRO_NAME, that string cannot be
        // inherited or forwarded into an unrelated shell, so it needs no
        // corroboration. This is also what makes a WSL2 container correctly
        // read as WSL even when it was not started with /dev/dxg passed in:
        // it shares the host kernel, so /proc/version still carries the
        // marker even though the container has no $WSL_DISTRO_NAME of its own.
        assert!(
            wsl_signals_indicate_wsl(false, "Linux version 6.6.87.2-microsoft-standard-WSL2"),
            "microsoft in /proc/version is enough on its own"
        );
        assert!(
            wsl_signals_indicate_wsl(false, "Linux version 5.15.0 wsl2"),
            "wsl in /proc/version is enough on its own"
        );
    }

    #[test]
    fn an_ordinary_kernel_with_no_signals_is_not_wsl() {
        assert!(
            !wsl_signals_indicate_wsl(false, "Linux version 6.8.0-51-generic"),
            "an ordinary kernel is not WSL"
        );
        assert!(
            !wsl_signals_indicate_wsl(false, ""),
            "no device and no /proc/version to read is not WSL"
        );
    }

    #[test]
    fn real_wsl2_and_wsl1_hosts_are_recognised_by_proc_version_alone() {
        // Its own doc comment above records that WSL 1 kernels always end in
        // "-microsoft" and WSL 2 kernels always carry "microsoft-standard" --
        // so every real WSL host is recognised without needing $WSL_DISTRO_NAME
        // or /dev/dxg at all.
        let wsl2 = "Linux version 5.15.167.4-microsoft-standard-WSL2";
        let wsl2_early = "Linux version 4.19.104-microsoft-standard";
        let wsl1 = "Linux version 4.4.0-19041-Microsoft";
        for proc_version in [wsl2, wsl2_early, wsl1] {
            assert!(
                wsl_signals_indicate_wsl(false, proc_version),
                "a real WSL host was not recognised: {proc_version:?}"
            );
        }
        // WSL 2 with GPU passthrough enabled also has /dev/dxg, which is
        // believed regardless of /proc/version.
        assert!(wsl_signals_indicate_wsl(true, wsl2));
    }

    #[test]
    fn wsl_case_folding_does_not_depend_on_the_kernel_string_casing() {
        assert!(wsl_signals_indicate_wsl(false, "MICROSOFT-STANDARD-WSL2"));
    }

    #[test]
    fn wsl1_is_told_apart_from_wsl2_by_the_kernel_release() {
        // The two are the same string family, distinguished only by the WSL2
        // marker. Getting this backwards would send a WSL 1 user chasing a
        // Windows driver update that can never give them a GPU, or hide the
        // conversion advice from the one platform that needs it.
        for wsl1 in [
            "4.4.0-19041-Microsoft",
            "4.4.0-18362-MICROSOFT",
            "4.4.0-17763-microsoft",
        ] {
            assert!(is_wsl1_kernel(wsl1), "{wsl1} is a WSL 1 kernel");
        }
        for wsl2 in [
            "6.6.87.2-microsoft-standard-WSL2",
            "5.15.167.4-microsoft-standard-WSL2",
            "6.18.33.2-MICROSOFT-STANDARD-WSL2",
            // The `-WSL2` suffix is not the marker. These are the earlier WSL 2
            // kernels, which carry `microsoft-standard` and no `WSL2` at all --
            // testing for the absence of `WSL2` called every one of them WSL 1.
            "4.19.104-microsoft-standard",
            "4.19.128-microsoft-standard",
            "5.10.16.3-microsoft-standard",
        ] {
            assert!(!is_wsl1_kernel(wsl2), "{wsl2} is a WSL 2 kernel");
        }
        // A bare-metal kernel is neither, and must not read as WSL 1 -- the
        // caller only asks on a host already known to be WSL, but answering
        // "yes" here would be wrong if that ever changed.
        assert!(!is_wsl1_kernel("6.8.0-51-generic"));
        assert!(!is_wsl1_kernel(""));
    }

    #[test]
    fn rocdxg_is_ready_only_when_the_whole_chain_is_present() {
        let ready = WslSummary {
            is_wsl: true,
            dxg_device: true,
            dxcore: true,
            librocdxg: true,
            rocdxg_dids: false,
            ldconfig_librocdxg: true,
            rocminfo: false,
            cargo: false,
            detail: None,
        };
        assert!(ready.rocdxg_ready());
        // Each link is load-bearing: drop any one and a GPU launch cannot work,
        // so `serve` must not be told it can.
        for break_one in 0..4 {
            let mut partial = ready.clone();
            match break_one {
                0 => partial.dxg_device = false,
                1 => partial.dxcore = false,
                2 => partial.librocdxg = false,
                _ => partial.ldconfig_librocdxg = false,
            }
            assert!(
                !partial.rocdxg_ready(),
                "a broken link at {break_one} must not read as ready"
            );
        }
    }
}
