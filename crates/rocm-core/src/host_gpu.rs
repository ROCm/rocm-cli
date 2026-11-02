// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! GPU/host hardware detection.
//!
//! `ExamineSummary`'s `HostGpuSummary`/`DriverSummary`/`WslSummary` aggregation
//! and the driver-summary and gfx-target fallback chains that feed it. The
//! gfx-target and TheRock-family tables those chains consult live in
//! [`crate::gfx_target`], and this module re-exports the public ones so their
//! `crate::host_gpu::` paths keep resolving. The TheRock-managed-SDK gfx-target
//! probe (`detect_managed_therock_sdk_gfx_target`) stays here, since its only
//! caller is this module's own gfx-target fallback chain; the registry reading
//! it uses lives in [`crate::managed_runtime`].
//!
//! The probes live in child modules: `capture` (optional-command capture and the
//! tool-on-`PATH` lookup), `os_facts` (kernel, distro, CPU and RAM), `linux`
//! (sysfs, KFD and DRM), `gpu_visibility` (which GPUs this process may use),
//! `windows_inventory` (the Windows display-adapter inventory) and `wsl` (WSL
//! detection and the Windows host driver as seen from the guest).
//!
//! `examine.rs` and `fix.rs` still reach several items here through `lib.rs`'s
//! flat `crate::` re-export paths rather than `crate::host_gpu::` — a
//! deliberate transitional step from the zero-diff extraction, not an
//! oversight.

#[cfg(windows)]
use crate::gfx_target::{
    amd_pci_device_id_from_pnp_id, gfx_target_from_amd_marketing_name,
    gfx_target_from_amd_pci_device_id,
};
pub use crate::gfx_target::{
    extract_first_gfx_token, gfx_target_from_pnp_device_id, gfx_target_has_no_rocm_build,
    known_therock_families, normalize_therock_family,
};
#[cfg(target_os = "linux")]
use crate::gfx_target::{parse_windows_display_gfx_target, parse_windows_display_name};
use crate::managed_runtime::{managed_sdk_ld_library_path, managed_therock_sdk_probe_candidates};
use crate::{
    AppPaths, LegacyRocmSummary, detect_legacy_rocm_summary, detect_managed_therock_family,
    host_path, managed_sdk_tool_path, runtime_is_linux, runtime_is_windows, runtime_os_name,
    unix_time_millis,
};
use anyhow::Result;
#[cfg(windows)]
use capture::tool_path_candidates;
use capture::{OPTIONAL_COMMAND_TIMEOUT, capture_optional_command};
pub(crate) use os_facts::detect_distro_name;
pub use os_facts::detect_system_ram_gib;
use os_facts::{detect_cpu_model, detect_kernel_version, format_gib_value};
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::fs;
#[cfg(windows)]
use std::io::Read;
use std::io::{IsTerminal, stdin, stdout};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

mod capture;
mod gpu_visibility;
pub(crate) mod linux;
mod os_facts;
mod windows_inventory;
mod wsl;

pub use gpu_visibility::{has_usable_amd_gpu, usable_amd_gpu_indices};
pub(crate) use linux::detect_linux_sysfs_gfx_target;
pub use linux::is_amdgpu_device;
#[cfg(any(target_os = "linux", test))]
pub(crate) use linux::{KfdGpuNode, kfd_gpu_nodes_in};
use linux::{detect_linux_amdgpu_driver_version, detect_linux_primary_gpu_name};
pub(crate) use windows_inventory::detect_windows_amd_driver_info;
#[cfg(any(windows, test))]
use windows_inventory::parse_windows_examine_inventory;
#[cfg(windows)]
use windows_inventory::parse_windows_pnputil_display_inventory;
#[cfg(windows)]
use windows_inventory::{
    WINDOWS_INVENTORY_QUERY_TIMEOUT, WINDOWS_PNP_ENTITY_INVENTORY_SCRIPT,
    WINDOWS_VIDEO_CONTROLLER_INVENTORY_SCRIPT,
};
use windows_inventory::{
    WindowsExamineInventory, detect_windows_amd_display_driver,
    detect_windows_display_gfx_target_with_inventory, detect_windows_examine_inventory,
    detect_windows_examine_inventory_with_driver_facts,
};
#[cfg(test)]
use wsl::local_windows_host_driver_probe;
pub use wsl::{WslHostDriverProbe, is_wsl_host};
pub(crate) use wsl::{
    detect_local_windows_host_driver, detect_wsl_host_driver, detect_wsl_summary, is_wsl1_kernel,
    ldconfig_cache, ldconfig_lists_librocdxg,
};
use wsl::{
    detect_wsl_windows_display_gfx_target, detect_wsl_windows_display_gfx_target_fast,
    detect_wsl_windows_display_name_fast, detect_wsl_windows_display_probe_text,
    wsl_driver_summary,
};

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

fn detect_wsl_windows_display_name(wsl: Option<&WslSummary>) -> Option<String> {
    if !runtime_is_linux() || !wsl.is_some_and(|summary| summary.is_wsl) {
        return None;
    }

    detect_wsl_windows_display_name_fast()
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
    /// The Windows AMD display driver's version and date, when both the OS is
    /// Windows and a version was read. `None` everywhere else.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub windows: Option<crate::windows_driver::WindowsDriverInfo>,
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

impl ExamineSummary {
    /// The full host summary, including the Windows display driver version and
    /// date. Everything that prints the summary, or serializes it for a client
    /// (`rocm examine`, the MCP and bridge snapshots), wants this one.
    pub fn gather() -> Result<Self> {
        Self::gather_scoped(true)
    }

    /// The host summary without the Windows driver version and date, for a
    /// caller that reads neither. On an AMD Windows host that skips the
    /// PowerShell video-controller query `gather` adds for them; the GPU
    /// target, CPU, RAM and every other field are the same.
    pub fn gather_without_driver_facts() -> Result<Self> {
        Self::gather_scoped(false)
    }

    fn gather_scoped(driver_facts: bool) -> Result<Self> {
        let paths = AppPaths::discover()?;
        let windows_inventory = if driver_facts {
            detect_windows_examine_inventory_with_driver_facts()
        } else {
            detect_windows_examine_inventory()
        };
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

    /// The `rocm_support` line of the examine text, or an empty string.
    ///
    /// Present only when a GPU was detected but no ROCm package family covers
    /// it (for example the gfx90c Vega iGPU in Renoir / Cezanne APUs). Without
    /// it `compatible_therock_family: <unknown>` reads like a probe failure, when
    /// the GPU was in fact recognized and simply has no ROCm build.
    fn rocm_support_line(&self) -> String {
        match (
            self.detected_gfx_target.as_deref(),
            self.compatible_therock_family.as_deref(),
        ) {
            (Some(gfx), None) if gfx_target_has_no_rocm_build(gfx) => format!(
                "  rocm_support: GPU detected ({gfx}) but not supported by ROCm; \
                 the Vulkan backend can still be used\n"
            ),
            _ => String::new(),
        }
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
            "rocm examine\n  os: {}\n  arch: {}\n  kernel: {}\n  distro: {}\n  cpu: {}\n  system_ram: {}\n  interactive_terminal: {}\n  default_engine: {}\n  detected_gfx_target: {}\n  compatible_therock_family: {}\n{}  detected_therock_family: {}\n  driver_policy: {}\n  driver_status: {}\n  driver_detail: {}\n{}  legacy_rocm_status: {}\n  legacy_rocm_paths: {}\n  legacy_rocm_version: {}\n  legacy_rocm_detail: {}\n  legacy_rocm_guidance: {}\n  wsl: {}\n  wsl_dxg_device: {}\n  wsl_dxcore: {}\n  wsl_librocdxg: {}\n  wsl_rocdxg_dids: {}\n  wsl_ldconfig_librocdxg: {}\n  wsl_global_rocminfo: {}\n  wsl_cargo: {}\n  wsl_detail: {}\n  managed_runtimes: {}\n  managed_services: {}\n  model_cache_entries: {}\n  config_dir: {}\n  data_dir: {}\n  cache_dir: {}\n",
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
            self.rocm_support_line(),
            self.detected_therock_family
                .as_deref()
                .unwrap_or("<not detected>"),
            self.driver.policy,
            self.driver.status,
            self.driver.detail.as_deref().unwrap_or("<unknown>"),
            self.windows_driver_lines(),
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

    /// The `driver_version`, `driver_date` and `driver_age` lines. Empty unless
    /// this is a Windows host where a driver version was read, so other
    /// platforms print exactly what they printed before.
    fn windows_driver_lines(&self) -> String {
        let Some(info) = self.driver.windows.as_ref() else {
            return String::new();
        };
        format!(
            "  driver_version: {}\n  driver_date: {}\n  driver_age: {}\n",
            info.version,
            info.date
                .map_or_else(|| "<unknown>".to_owned(), |d| d.to_string()),
            crate::windows_driver::age_summary(info),
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

fn detect_driver_summary_with_windows_inventory(
    windows_inventory: Option<&WindowsExamineInventory>,
    wsl: Option<&WslSummary>,
) -> DriverSummary {
    if runtime_is_windows() {
        return windows_driver_summary_from_inventory(
            windows_inventory,
            detect_windows_amd_display_driver,
        );
    }

    if let Some(wsl) = wsl {
        return wsl_driver_summary(wsl);
    }

    detect_driver_summary()
}

/// The Windows driver summary for an inventory: the detail line and the driver
/// version and date, both from the same adapter.
///
/// Pure apart from `fallback_detail`, which runs only when there is no
/// inventory at all. The runtime check stays at the one production call site;
/// everything it feeds is decided here, so tests cover it on every platform.
fn windows_driver_summary_from_inventory(
    windows_inventory: Option<&WindowsExamineInventory>,
    fallback_detail: impl FnOnce() -> Option<String>,
) -> DriverSummary {
    let detail = windows_inventory
        .and_then(WindowsExamineInventory::amd_display_driver_detail)
        .or_else(|| {
            if windows_inventory.is_none() {
                fallback_detail()
            } else {
                None
            }
        });
    let info = windows_inventory.and_then(WindowsExamineInventory::amd_display_driver_info);
    windows_driver_summary(detail, info)
}

fn detect_driver_summary() -> DriverSummary {
    if runtime_is_windows() {
        let detail = detect_windows_amd_display_driver();
        return windows_driver_summary(detail, None);
    }

    if runtime_is_linux() {
        let module_detected = host_path("/sys/module/amdgpu").exists();
        return DriverSummary {
            policy: "linux_official_amd_dkms_wrapper".to_owned(),
            status: if module_detected {
                "amdgpu_available".to_owned()
            } else {
                "not_detected".to_owned()
            },
            detail: if host_path("/dev/kfd").exists() {
                Some("/dev/kfd is present".to_owned())
            } else if module_detected {
                Some("amdgpu module metadata is present".to_owned())
            } else {
                None
            },
            windows: None,
        };
    }

    DriverSummary {
        policy: "inspection_only".to_owned(),
        status: "unsupported_platform".to_owned(),
        detail: None,
        windows: None,
    }
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

fn windows_driver_summary(
    detail: Option<String>,
    windows: Option<crate::windows_driver::WindowsDriverInfo>,
) -> DriverSummary {
    DriverSummary {
        policy: "windows_validate_only".to_owned(),
        status: if detail.is_some() {
            "amd_display_driver_detected".to_owned()
        } else {
            "not_detected".to_owned()
        },
        detail,
        windows,
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_app_paths;
    use std::path::PathBuf;
    use windows_inventory::WindowsInventorySources;

    fn write_fake_rocm_agent_enumerator(bin_dir: &Path, target: &str) -> Result<()> {
        if cfg!(windows) {
            let path = bin_dir.join("rocm_agent_enumerator.cmd");
            fs::write(path, format!("@echo off\r\necho {target}\r\n"))?;
        } else {
            let path = bin_dir.join("rocm_agent_enumerator");
            crate::fake_executable::write_fake_executable(
                &path,
                format!("#!/bin/sh\necho {target}\n"),
            )?;
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
                windows: None,
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

    fn examine_summary_with_gpu(
        detected_gfx_target: Option<&str>,
        compatible_therock_family: Option<&str>,
    ) -> ExamineSummary {
        ExamineSummary {
            os: "windows".to_owned(),
            arch: "x86_64".to_owned(),
            kernel: None,
            distro: None,
            cpu: None,
            system_ram_gib: None,
            interactive_terminal: false,
            default_engine: "lemonade".to_owned(),
            detected_gfx_target: detected_gfx_target.map(str::to_owned),
            compatible_therock_family: compatible_therock_family.map(str::to_owned),
            detected_therock_family: None,
            driver: DriverSummary {
                policy: "windows_validate_only".to_owned(),
                status: "amd_display_driver_detected".to_owned(),
                detail: None,
                windows: None,
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
        }
    }

    fn windows_summary_with_driver(
        info: Option<crate::windows_driver::WindowsDriverInfo>,
    ) -> ExamineSummary {
        ExamineSummary {
            os: "windows".to_owned(),
            arch: "x86_64".to_owned(),
            kernel: None,
            distro: None,
            cpu: None,
            system_ram_gib: None,
            interactive_terminal: false,
            default_engine: "lemonade".to_owned(),
            detected_gfx_target: None,
            compatible_therock_family: None,
            detected_therock_family: None,
            driver: windows_driver_summary(
                Some("AMD Radeon(TM) Graphics driver 30.0.13014.8".to_owned()),
                info,
            ),
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
        }
    }

    #[test]
    fn examine_render_says_a_gpu_without_a_rocm_family_is_unsupported() {
        // gfx90c is detected from its PCI ID but has no ROCm package family.
        let family = normalize_therock_family("gfx90c");
        let summary = examine_summary_with_gpu(Some("gfx90c"), family.as_deref());
        let rendered = summary.render_text();
        assert!(
            rendered.contains("detected_gfx_target: gfx90c"),
            "{rendered}"
        );
        assert!(
            rendered.contains("rocm_support: GPU detected (gfx90c) but not supported by ROCm"),
            "{rendered}"
        );
        assert!(
            rendered.contains("the Vulkan backend can still be used"),
            "{rendered}"
        );
    }

    #[test]
    fn examine_render_omits_the_support_note_for_supported_or_missing_gpus() {
        let family = normalize_therock_family("gfx1103");
        let supported = examine_summary_with_gpu(Some("gfx1103"), family.as_deref());
        let rendered = supported.render_text();
        assert!(!rendered.contains("rocm_support"), "{rendered}");

        let missing = examine_summary_with_gpu(None, None);
        let rendered = missing.render_text();
        assert!(!rendered.contains("rocm_support"), "{rendered}");

        // An unmapped target the CLI knows nothing about is not claimed to be
        // unsupported; only targets known to have no ROCm build are.
        let unknown = examine_summary_with_gpu(Some("gfx803"), None);
        let rendered = unknown.render_text();
        assert!(!rendered.contains("rocm_support"), "{rendered}");
    }

    #[test]
    fn examine_shows_the_windows_driver_version_date_and_age() {
        let old = windows_summary_with_driver(
            crate::windows_driver::WindowsDriverInfo::from_probe("30.0.13014.8", "2021-08-04"),
        )
        .render_text();
        assert!(old.contains("  driver_version: 30.0.13014.8\n"));
        assert!(old.contains("  driver_date: 2021-08-04\n"));
        assert!(old.contains("  driver_age: older than the first HIP SDK for Windows"));
        assert!(old.contains(crate::windows_driver::DISPLAY_DRIVER_OLD_FIX_ID));
        assert!(old.contains("not established"));

        let current = windows_summary_with_driver(
            crate::windows_driver::WindowsDriverInfo::from_probe("32.0.23033.1002", "2026-05-02"),
        )
        .render_text();
        assert!(current.contains("  driver_version: 32.0.23033.1002\n"));
        assert!(current.contains("  driver_age: not older than"));
        assert!(!current.contains(crate::windows_driver::DISPLAY_DRIVER_OLD_FIX_ID));

        let undated = windows_summary_with_driver(
            crate::windows_driver::WindowsDriverInfo::from_probe("30.0.13014.8", ""),
        )
        .render_text();
        assert!(undated.contains("  driver_date: <unknown>\n"));
        assert!(undated.contains("  driver_age: unknown"));
        assert!(!undated.contains(crate::windows_driver::DISPLAY_DRIVER_OLD_FIX_ID));
    }

    #[test]
    fn examine_prints_no_driver_age_lines_when_no_version_was_read() {
        let rendered = windows_summary_with_driver(None).render_text();
        assert!(!rendered.contains("driver_version:"));
        assert!(!rendered.contains("driver_age:"));
        assert!(rendered.contains("driver_detail: AMD Radeon(TM) Graphics driver"));
    }

    // --- the one probe result behind examine, install driver and diagnose ---

    const IGPU_OLD: &str =
        "GPU\tAMD Radeon(TM) Graphics\t30.0.13014.8\tPCI\\VEN_1002&DEV_1636\t2021-08-04\n";
    pub(super) const DGPU_CURRENT: &str =
        "GPU\tAMD Radeon RX 6600\t32.0.23033.1002\tPCI\\VEN_1002&DEV_73E0\t2026-05-02\n";

    /// Counts the queries an inventory run starts, and answers each with canned
    /// text. `pnputil` answers like the real one: adapters, no driver facts.
    #[derive(Default)]
    pub(super) struct CountingSources {
        pub(super) pnputil: Option<&'static str>,
        pub(super) video: Option<&'static str>,
        pub(super) pnp_entity: Option<&'static str>,
        pub(super) system: Option<&'static str>,
        pub(super) calls: Vec<&'static str>,
    }

    impl CountingSources {
        fn answer(text: Option<&'static str>) -> Option<WindowsExamineInventory> {
            text.map(parse_windows_examine_inventory)
        }
    }

    impl WindowsInventorySources for CountingSources {
        fn pnputil(&mut self) -> Option<WindowsExamineInventory> {
            self.calls.push("pnputil");
            Self::answer(self.pnputil)
        }
        fn video_controller(&mut self) -> Option<WindowsExamineInventory> {
            self.calls.push("video");
            Self::answer(self.video)
        }
        fn pnp_entity(&mut self) -> Option<WindowsExamineInventory> {
            self.calls.push("pnp_entity");
            Self::answer(self.pnp_entity)
        }
        fn system(&mut self) -> Option<WindowsExamineInventory> {
            self.calls.push("system");
            Self::answer(self.system)
        }
    }

    pub(super) const PNPUTIL_ONLY: &str = "GPU\tAMD Radeon RX 6600\t\tPCI\\VEN_1002&DEV_73E0\n";

    pub(super) fn pnputil_host() -> CountingSources {
        CountingSources {
            pnputil: Some(PNPUTIL_ONLY),
            video: Some(DGPU_CURRENT),
            system: Some("CPU\tAMD Ryzen 5 5600\nRAM\t17179869184\n"),
            ..CountingSources::default()
        }
    }

    /// `gather_without_driver_facts` is `gather` minus the driver version and
    /// date: the GPU target a caller reads is the same, and it never carries a
    /// driver record the full summary lacks. That it starts no PowerShell query
    /// for them is the `the_startup_paths_do_not_start_the_driver_fact_query`
    /// test, which counts the sources the same scope runs.
    #[test]
    fn the_summary_without_driver_facts_keeps_the_gpu_target() {
        let full = ExamineSummary::gather().expect("full summary");
        let lean = ExamineSummary::gather_without_driver_facts().expect("lean summary");
        assert_eq!(lean.os, full.os);
        assert_eq!(lean.detected_gfx_target, full.detected_gfx_target);
        assert_eq!(
            lean.compatible_therock_family,
            full.compatible_therock_family
        );
        assert!(
            lean.driver.windows.is_none() || lean.driver.windows == full.driver.windows,
            "{:?} vs {:?}",
            lean.driver.windows,
            full.driver.windows
        );
    }

    /// The runtime check stays at the call site; what it feeds is decided here,
    /// so it is covered on every platform. Replacing the driver info with `None`
    /// in this function fails this test.
    #[test]
    fn the_windows_summary_carries_the_driver_from_the_inventory() {
        let inventory = parse_windows_examine_inventory(IGPU_OLD);
        let summary = windows_driver_summary_from_inventory(Some(&inventory), || {
            panic!("an inventory was given, so the fallback must not run")
        });
        let info = summary.windows.expect("the driver reaches the summary");
        assert_eq!(info.version, "30.0.13014.8");
        assert!(matches!(
            info.age(),
            crate::windows_driver::DriverAge::Predates(_)
        ));
        assert_eq!(summary.status, "amd_display_driver_detected");

        // No inventory: the fallback detail is used and there is no version.
        let none = windows_driver_summary_from_inventory(None, || {
            Some("AMD Radeon(TM) Graphics driver".to_owned())
        });
        assert_eq!(none.windows, None);
        assert_eq!(none.status, "amd_display_driver_detected");
    }

    /// An integrated GPU with an old driver and a supported discrete GPU with a
    /// recent one: `preferred_amd_display` picks the discrete card, and the
    /// examine text, the install text and the diagnose result all report it.
    #[test]
    fn examine_install_and_diagnose_read_the_same_adapter_on_a_two_adapter_host() {
        // The iGPU is listed first, which is where a first-match query stops.
        let inventory = parse_windows_examine_inventory(&format!("{IGPU_OLD}{DGPU_CURRENT}"));
        let summary = windows_driver_summary_from_inventory(Some(&inventory), || None);
        let info = summary.windows.expect("a driver");
        assert_eq!(info.version, "32.0.23033.1002", "the discrete GPU's driver");

        // examine
        let examine_text = windows_summary_with_driver(Some(info.clone())).render_text();
        assert!(
            examine_text.contains("  driver_date: 2026-05-02\n"),
            "{examine_text}"
        );
        assert!(!examine_text.contains(crate::windows_driver::DISPLAY_DRIVER_OLD_FIX_ID));

        // install driver
        let install = crate::windows_driver::install_driver_summary(Some(&info));
        assert!(install.contains("outdated: no"), "{install}");

        // diagnose, fed the same probe result
        let mut exam = crate::examine::Examination {
            os_family: "windows".to_owned(),
            ..crate::examine::Examination::default()
        };
        exam.set_windows_driver(inventory.amd_display_driver_info().as_ref());
        assert_eq!(exam.adrenalin_version, "32.0.23033.1002");
        assert_eq!(exam.adrenalin_date, "2026-05-02");
        let report = crate::run_diagnose(&exam, "");
        assert!(
            report
                .matched
                .iter()
                .all(|d| d.id != crate::windows_driver::DISPLAY_DRIVER_OLD_FIX_ID),
            "the old iGPU driver must not be raised when the discrete GPU is chosen"
        );
    }

    /// Windows loads the Microsoft Basic Display Adapter when no AMD driver is
    /// installed. It keeps the AMD vendor id, and its placeholder 2006 date is
    /// after the year-2000 guard. Inputs are recalled from how Windows reports
    /// it, not observed on a Windows machine.
    #[test]
    fn the_microsoft_basic_display_adapter_is_not_an_old_amd_driver() {
        for text in [
            "GPU\tMicrosoft Basic Display Adapter\t10.0.26100.1\tPCI\\VEN_1002&DEV_73E0\t2006-06-21\n",
            // Named like an AMD adapter, with the operating system's version.
            "GPU\tAMD Radeon RX 6600\t10.0.19041.3636\tPCI\\VEN_1002&DEV_73E0\t2006-06-21\n",
            // The name alone, with a version that does not look like the OS's.
            "GPU\tMicrosoft Basic Display Adapter\t1.0.0.0\tPCI\\VEN_1002&DEV_73E0\t2006-06-21\n",
        ] {
            let inventory = parse_windows_examine_inventory(text);
            assert!(inventory.preferred_amd_display().is_some(), "{text}");
            assert_eq!(inventory.amd_display_driver_info(), None, "{text}");
            // The WSL and local host-driver probes read the same exclusion, so
            // a fallback driver is "no AMD display", not an installed version.
            assert_eq!(
                local_windows_host_driver_probe(Some(inventory.clone())),
                WslHostDriverProbe::NoAmdDisplay,
                "{text}"
            );
            let summary = windows_driver_summary_from_inventory(Some(&inventory), || None);
            assert_eq!(summary.windows, None, "{text}");
            // The legacy status and detail agree: no AMD driver is detected.
            assert_eq!(summary.status, "not_detected", "{text}");
            assert_eq!(summary.detail, None, "{text}");
            let rendered = windows_summary_with_driver(summary.windows).render_text();
            assert!(!rendered.contains("driver_age:"), "{rendered}");
            assert!(!rendered.contains(crate::windows_driver::DISPLAY_DRIVER_OLD_FIX_ID));
        }
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
                windows: None,
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
                windows: None,
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
}
