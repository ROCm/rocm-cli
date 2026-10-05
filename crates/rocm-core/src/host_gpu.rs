// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use crate::{
    AppPaths, OPTIONAL_COMMAND_TIMEOUT, WINDOWS_INVENTORY_QUERY_TIMEOUT,
    WINDOWS_VIDEO_CONTROLLER_INVENTORY_SCRIPT, WindowsExamineInventory, WslSummary,
    detect_host_gpu_summary_fast, detect_managed_therock_sdk_gfx_target,
    detect_windows_examine_inventory, detect_wsl_summary, env_flag, is_wsl_host,
    parse_windows_examine_inventory, runtime_is_linux, runtime_is_windows, unix_time_millis,
};
#[cfg(test)]
use anyhow::Result;
use std::ffi::OsString;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

pub fn detect_host_gfx_target() -> Option<String> {
    let paths = AppPaths::discover().ok();
    detect_host_gpu_summary_fast(paths.as_ref()).gfx_target
}

pub(crate) fn detect_examine_gfx_target_fast(
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
pub(crate) fn detect_host_gfx_target_with_context(
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

pub(crate) fn capture_optional_command(program: &str, args: &[&str]) -> Option<String> {
    capture_optional_command_with_timeout(program, args, OPTIONAL_COMMAND_TIMEOUT)
}

pub(crate) fn capture_optional_command_with_timeout(
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

pub(crate) fn capture_optional_path_command_with_env(
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

pub(crate) fn tool_on_path(program: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|path| {
        std::env::split_paths(&path).any(|dir| {
            tool_path_candidates(program)
                .into_iter()
                .any(|name| dir.join(name).is_file())
        })
    })
}

pub(crate) fn tool_path_candidates(program: &str) -> Vec<String> {
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

pub(crate) fn detect_windows_display_gfx_target_with_inventory(
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

pub(crate) fn detect_wsl_windows_display_name(wsl: Option<&WslSummary>) -> Option<String> {
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

pub(crate) fn detect_wsl_windows_display_probe_text() -> Option<String> {
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
pub(crate) fn detect_linux_primary_gpu_name() -> Option<String> {
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
pub(crate) const fn detect_linux_primary_gpu_name() -> Option<String> {
    None
}

pub(crate) fn parse_windows_display_gfx_target(text: &str) -> Option<String> {
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

pub(crate) fn parse_windows_display_name(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .find_map(|line| {
            let (name, _) = line.split_once('\t').unwrap_or((line, ""));
            let name = name.trim();
            (!name.is_empty()).then(|| name.to_owned())
        })
}

pub(crate) fn amd_pci_device_id_from_pnp_id(pnp_id: &str) -> Option<String> {
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

pub(crate) fn gfx_target_from_amd_pci_device_id(device_id: &str) -> Option<&'static str> {
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

pub(crate) fn gfx_target_from_amd_marketing_name(name: &str) -> Option<&'static str> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_app_paths(name: &str) -> (PathBuf, AppPaths) {
        let root = workspace_test_artifact_dir().join(format!(
            "rocm-core-{name}-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        (root, paths)
    }

    fn workspace_test_artifact_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(".rocm-work")
            .join("tests")
            .join("core")
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
}
