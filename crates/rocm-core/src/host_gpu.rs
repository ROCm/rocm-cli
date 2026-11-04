// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! GPU/host hardware detection.
//!
//! `ExamineSummary`'s `HostGpuSummary`/`DriverSummary`/`WslSummary` aggregation
//! and the driver-summary and gfx-target fallback chains that feed it. The
//! gfx-target and TheRock-family tables those chains consult live in
//! [`crate::gfx_target`], and this module re-exports the public ones so their
//! `crate::host_gpu::` paths keep resolving. `interactive_terminal` lives in
//! [`crate::confirm`] and is re-exported here too, so the
//! `crate::host_gpu::interactive_terminal` path keeps resolving.
//!
//! The probes live in child modules: `capture` (optional-command capture and the
//! tool-on-`PATH` lookup), `os_facts` (kernel, distro, CPU and RAM), `linux`
//! (sysfs, KFD and DRM), `gpu_visibility` (which GPUs this process may use),
//! `windows_inventory` (the Windows display-adapter inventory), `wsl` (WSL
//! detection and the Windows host driver as seen from the guest), `render` (the
//! `rocm examine` text of an `ExamineSummary`), `diagnostics` (the GPU
//! detection diagnostics text) and `engine_default` (the default serve engine).
//!
//! The `examine` child modules and `fix.rs` still reach several items here
//! through `lib.rs`'s flat `crate::` re-export paths rather than
//! `crate::host_gpu::` — a deliberate transitional step from the mechanical
//! extraction, not an oversight.

pub use crate::confirm::interactive_terminal;
pub use crate::gfx_target::{
    extract_first_gfx_token, gfx_target_from_pnp_device_id, gfx_target_has_no_rocm_build,
    known_therock_families, normalize_therock_family,
};
#[cfg(target_os = "linux")]
use crate::gfx_target::{parse_windows_display_gfx_target, parse_windows_display_name};
use crate::{
    AppPaths, LegacyRocmSummary, detect_legacy_rocm_summary, detect_managed_therock_family,
    host_path, runtime_is_linux, runtime_is_windows, runtime_os_name,
};
use anyhow::Result;
pub(crate) use os_facts::detect_distro_name;
pub use os_facts::detect_system_ram_gib;
use os_facts::{detect_cpu_model, detect_kernel_version};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

mod capture;
mod diagnostics;
mod engine_default;
mod gpu_visibility;
pub(crate) mod linux;
mod os_facts;
mod render;
mod windows_inventory;
mod wsl;

pub use diagnostics::detect_host_gpu_diagnostics;
pub use engine_default::{
    default_engine_for_host, default_engine_for_platform,
    preferred_serve_engine_for_host_gpu_summary,
};

pub use gpu_visibility::{has_usable_amd_gpu, usable_amd_gpu_indices};
pub(crate) use linux::detect_linux_sysfs_gfx_target;
pub use linux::is_amdgpu_device;
#[cfg(any(target_os = "linux", test))]
pub(crate) use linux::{KfdGpuNode, kfd_gpu_nodes_in};
use linux::{detect_linux_amdgpu_driver_version, detect_linux_primary_gpu_name};
pub(crate) use windows_inventory::detect_windows_amd_driver_info;
#[cfg(test)]
use windows_inventory::parse_windows_examine_inventory;
use windows_inventory::{
    WindowsExamineInventory, detect_windows_amd_display_driver,
    detect_windows_display_gfx_target_with_inventory, detect_windows_examine_inventory,
    detect_windows_examine_inventory_with_driver_facts,
};
#[cfg(target_os = "linux")]
use wsl::detect_wsl_windows_display_probe_text;
#[cfg(test)]
use wsl::local_windows_host_driver_probe;
pub use wsl::{WslHostDriverProbe, is_wsl_host};
pub(crate) use wsl::{
    detect_local_windows_host_driver, detect_wsl_host_driver, detect_wsl_summary, is_wsl1_kernel,
    ldconfig_cache, ldconfig_lists_librocdxg,
};
use wsl::{detect_wsl_windows_display_gfx_target_fast, wsl_driver_summary};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_app_paths;
    use std::path::PathBuf;
    use windows_inventory::WindowsInventorySources;

    #[test]
    fn windows_examine_gfx_detection_uses_the_display_inventory() {
        if !cfg!(windows) {
            return;
        }
        let inventory = parse_windows_examine_inventory(
            "GPU\tAMD Radeon RX 9070 XT\t32.0.23033.1002\tPCI\\VEN_1002&DEV_7550",
        );

        assert_eq!(
            detect_windows_display_gfx_target_with_inventory(Some(&inventory)),
            Some("gfx1201".to_owned())
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

    pub(super) fn windows_summary_with_driver(
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
}
