// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Whether this host can run a model through Vulkan.
//!
//! The CLI has always *asserted* this for a GPU with no ROCm build -- `examine`
//! told a gfx90c user "the Vulkan backend can still be used" without ever
//! looking. This probe is what makes that sentence evidence rather than a
//! promise, and it is what lets a host outside the ROCm compatibility matrix be
//! described by what it can do instead of only by what it cannot.
//!
//! Two sources, in increasing order of strength:
//!
//! 1. An **installable driver**: an ICD manifest on Linux (`/usr/share/vulkan/
//!    icd.d` and friends), or the Khronos registry key on Windows. This proves a
//!    driver is *installed*, not that a device enumerates.
//! 2. An **enumerated device**: `vulkaninfo --summary`, when it is on PATH.
//!    This proves a device actually answers.
//!
//! The two are kept apart by [`VulkanStatus`] rather than collapsed, because
//! collapsing them would replace one unverified promise with another.
//!
//! **A software rasteriser is not a GPU.** Mesa's lavapipe (`lvp_icd.*.json`,
//! `PHYSICAL_DEVICE_TYPE_CPU`) presents a complete Vulkan implementation that
//! runs entirely on the CPU. Counting it here would route a user onto CPU
//! execution while reporting a GPU path -- precisely the silent CPU fallback
//! `AGENTS.md` §6 forbids -- so it is recognised in order to be excluded, and
//! named in `detail` so the exclusion is visible rather than mysterious.

use super::capture::capture_optional_command;
#[cfg(target_os = "linux")]
use crate::host_path;
use serde::{Deserialize, Serialize};

/// The directories a Linux Vulkan loader reads ICD manifests from.
///
/// Read through [`host_path`] so a simulated host root can plant them, which is
/// what lets the no-GPU CI lane cover this probe at all.
#[cfg(target_os = "linux")]
const LINUX_ICD_DIRS: [&str; 4] = [
    "/usr/share/vulkan/icd.d",
    "/usr/local/share/vulkan/icd.d",
    "/etc/vulkan/icd.d",
    "/usr/lib/x86_64-linux-gnu/vulkan/icd.d",
];

/// How sure the CLI is that a model can run through Vulkan here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VulkanStatus {
    /// A Vulkan device answered and it is a real GPU.
    Available,
    /// A driver is installed, but no device was enumerated -- `vulkaninfo` is
    /// not on PATH, so nothing confirmed a device answers. Likely to work;
    /// not established.
    DriverPresentDevicesUnverified,
    /// The probe looked and found no GPU Vulkan driver. A software rasteriser
    /// alone lands here, not in [`Self::Available`].
    Absent,
    /// The probe could not look (an unsupported platform).
    ///
    /// The default: a value nobody has filled in is one nobody has probed, and
    /// defaulting to `Absent` would let an unset field read as a fact about the
    /// machine.
    #[default]
    Unknown,
}

impl VulkanStatus {
    /// Whether a model may be routed through Vulkan on the strength of this.
    #[must_use]
    pub const fn is_usable(self) -> bool {
        matches!(self, Self::Available | Self::DriverPresentDevicesUnverified)
    }
}

/// One physical device as `vulkaninfo --summary` reported it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanDevice {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driver_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driver_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vendor_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_id: Option<String>,
    /// The raw `deviceType`, e.g. `PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_type: Option<String>,
}

impl VulkanDevice {
    /// AMD's PCI vendor id, as `vulkaninfo` prints it.
    const AMD_VENDOR_ID: &'static str = "0x1002";

    #[must_use]
    pub fn is_amd(&self) -> bool {
        self.vendor_id
            .as_deref()
            .is_some_and(|id| id.eq_ignore_ascii_case(Self::AMD_VENDOR_ID))
    }

    /// Whether this device is real hardware rather than a software rasteriser.
    ///
    /// A missing `deviceType` reads as hardware: the field is optional in the
    /// summary output, and treating an unreported type as CPU would discard a
    /// working GPU. Lavapipe always reports its type, so the only entries this
    /// lets through unverified are ones that named no type at all.
    #[must_use]
    pub fn is_hardware(&self) -> bool {
        !self
            .device_type
            .as_deref()
            .is_some_and(|kind| kind.to_ascii_uppercase().contains("TYPE_CPU"))
    }
}

/// What the host offers for Vulkan, and the evidence behind it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VulkanSupport {
    pub status: VulkanStatus,
    /// The ICD manifests or registry entries found, as the loader names them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub icds: Vec<String>,
    /// The devices `vulkaninfo` enumerated, when it was available.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub devices: Vec<VulkanDevice>,
    /// A one-line human summary, including why anything was excluded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl VulkanSupport {
    /// The driver this host would use, for the `examine` line.
    ///
    /// Prefers what a device actually reported (`AMD proprietary driver`,
    /// `radv`) over what an ICD filename implies, because the filename is a
    /// guess about a vendor and the device's own `driverName` is not.
    #[must_use]
    pub fn driver_label(&self) -> Option<String> {
        self.devices
            .iter()
            .find(|device| device.is_amd() && device.is_hardware())
            .and_then(|device| device.driver_name.clone())
            .or_else(|| {
                self.icds
                    .iter()
                    .find_map(|icd| icd_driver_label(icd))
                    .map(str::to_owned)
            })
    }
}

/// The driver an ICD manifest's file name implies, or `None` when the name is
/// not one this table knows.
///
/// Deliberately includes the software rasterisers so a caller can *exclude*
/// them by name; [`icd_is_gpu_driver`] is the predicate that does so.
#[must_use]
pub fn icd_driver_label(icd: &str) -> Option<&'static str> {
    let name = icd
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(icd)
        .to_ascii_lowercase();
    if name.starts_with("radeon_icd") {
        return Some("Mesa RADV");
    }
    if name.starts_with("amd_icd") || name.starts_with("amdvlk") {
        return Some("AMDVLK");
    }
    if name.starts_with("lvp_icd") {
        return Some("Mesa lavapipe (software)");
    }
    if name.starts_with("vk_swiftshader") {
        return Some("SwiftShader (software)");
    }
    if name.starts_with("nvidia_icd") {
        return Some("NVIDIA");
    }
    if name.starts_with("intel_icd") || name.starts_with("intel_hasvk") {
        return Some("Intel");
    }
    None
}

/// Whether an ICD manifest names a driver that runs on a GPU.
///
/// An unrecognised manifest counts as a GPU driver: a vendor the table does not
/// list yet is far more likely to be real hardware than a new software
/// rasteriser, and the cost of the two errors is not symmetric -- missing a
/// working GPU path sends the user to CPU, while admitting an unknown one is
/// corrected by the device enumeration that follows.
#[must_use]
pub fn icd_is_gpu_driver(icd: &str) -> bool {
    !icd_driver_label(icd).is_some_and(|label| label.contains("(software)"))
}

/// Parse `vulkaninfo --summary` output into its physical devices.
///
/// Tolerates the loader warnings `vulkaninfo` prints to stdout before the
/// report, and the `Instance Extensions` / `Instance Layers` sections ahead of
/// `Devices:` -- only `GPUn:` blocks and their indented `key = value` lines are
/// read.
#[must_use]
pub fn parse_vulkaninfo_summary(output: &str) -> Vec<VulkanDevice> {
    let mut devices = Vec::new();
    let mut current: Option<VulkanDevice> = None;
    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("GPU") && trimmed.ends_with(':') {
            if let Some(device) = current.take() {
                devices.push(device);
            }
            current = Some(VulkanDevice::default());
            continue;
        }
        let Some(device) = current.as_mut() else {
            continue;
        };
        let Some((key, value)) = trimmed.split_once('=') else {
            // A non-indented, non-`key = value` line ends the block: the
            // summary's device section is the last thing printed, but a caller
            // may hand us trailing output.
            if !trimmed.is_empty() && !line.starts_with([' ', '\t']) {
                devices.push(current.take().unwrap_or_default());
            }
            continue;
        };
        let value = value.trim().to_owned();
        if value.is_empty() {
            continue;
        }
        match key.trim() {
            "deviceName" => device.name = value,
            "driverName" => device.driver_name = Some(value),
            "driverID" => device.driver_id = Some(value),
            "apiVersion" => device.api_version = Some(value),
            "vendorID" => device.vendor_id = Some(value),
            "deviceID" => device.device_id = Some(value),
            "deviceType" => device.device_type = Some(value),
            _ => {}
        }
    }
    if let Some(device) = current.take() {
        devices.push(device);
    }
    devices.retain(|device| !device.name.is_empty());
    devices
}

/// The ICD manifest paths listed under the Windows Khronos registry key, as
/// [`WINDOWS_VULKAN_ICD_SCRIPT`] prints them (one `ICD\t<path>` row each).
#[must_use]
pub fn parse_windows_vulkan_icd_listing(output: &str) -> Vec<String> {
    output
        .lines()
        .filter_map(|line| line.trim().strip_prefix("ICD\t"))
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Enumerate both Khronos ICD keys. The WOW6432Node key carries the 32-bit
/// drivers; a 64-bit-only install lists nothing there, which is not an error.
#[cfg(any(windows, test))]
pub(super) const WINDOWS_VULKAN_ICD_SCRIPT: &str = r#"foreach ($key in 'HKLM:\SOFTWARE\Khronos\Vulkan\Drivers','HKLM:\SOFTWARE\WOW6432Node\Khronos\Vulkan\Drivers') { if (Test-Path $key) { foreach ($name in (Get-Item $key).GetValueNames()) { "ICD`t$name" } } }"#;

/// Every Vulkan ICD manifest this host's loader would read.
#[cfg(target_os = "linux")]
fn detect_icds() -> Vec<String> {
    let mut icds = Vec::new();
    for dir in LINUX_ICD_DIRS {
        let Ok(entries) = std::fs::read_dir(host_path(dir)) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "json")
                && let Some(name) = path.file_name().and_then(|name| name.to_str())
            {
                icds.push(name.to_owned());
            }
        }
    }
    icds.sort();
    icds.dedup();
    icds
}

#[cfg(windows)]
fn detect_icds() -> Vec<String> {
    capture_optional_command(
        "powershell",
        &[
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            WINDOWS_VULKAN_ICD_SCRIPT,
        ],
    )
    .map(|output| parse_windows_vulkan_icd_listing(&output))
    .unwrap_or_default()
}

#[cfg(not(any(target_os = "linux", windows)))]
const fn detect_icds() -> Vec<String> {
    Vec::new()
}

/// The devices `vulkaninfo --summary` reports, or an empty list when the tool
/// is not installed. Absence of the tool is not absence of a driver, which is
/// why this cannot decide [`VulkanStatus`] on its own.
fn detect_vulkan_devices() -> Vec<VulkanDevice> {
    capture_optional_command("vulkaninfo", &["--summary"])
        .map(|output| parse_vulkaninfo_summary(&output))
        .unwrap_or_default()
}

/// Whether `vulkaninfo` is on this host at all, regardless of what it said.
///
/// The distinction the classification turns on is "could we ask" versus "we
/// asked and nothing was there" — and an empty device list cannot tell those
/// apart, because a missing binary and one that exits non-zero both yield it.
/// A host that genuinely has no Vulkan driver is the second case: `vulkaninfo`
/// is installed and fails. Only the first is a gap in the probe.
fn vulkan_enumeration_available() -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| {
        crate::probe_exec::tool_path_candidates("vulkaninfo")
            .iter()
            .any(|name| dir.join(name).is_file())
    })
}

/// Whether this platform's Vulkan surfaces can be read at all.
const fn vulkan_is_probeable() -> bool {
    cfg!(any(target_os = "linux", windows))
}

/// What this host offers for Vulkan.
#[must_use]
pub fn detect_vulkan_support() -> VulkanSupport {
    classify_vulkan_support(
        detect_icds(),
        detect_vulkan_devices(),
        vulkan_is_probeable(),
        vulkan_enumeration_available(),
    )
}

/// The pure decision behind [`detect_vulkan_support`], split out so every
/// combination of evidence can be tested without a Vulkan install.
#[must_use]
pub fn classify_vulkan_support(
    icds: Vec<String>,
    devices: Vec<VulkanDevice>,
    probeable: bool,
    enumerated: bool,
) -> VulkanSupport {
    if !probeable {
        return VulkanSupport {
            status: VulkanStatus::Unknown,
            icds,
            devices,
            detail: Some("Vulkan support is not probed on this platform".to_owned()),
        };
    }

    let amd_gpu = devices
        .iter()
        .find(|device| device.is_amd() && device.is_hardware());
    if let Some(device) = amd_gpu {
        let driver = device.driver_name.as_deref().unwrap_or("an unnamed driver");
        return VulkanSupport {
            detail: Some(format!("{} through {driver}", device.name)),
            status: VulkanStatus::Available,
            icds,
            devices,
        };
    }

    let software_only = !devices.is_empty() && devices.iter().all(|device| !device.is_hardware());
    if software_only {
        return VulkanSupport {
            status: VulkanStatus::Absent,
            detail: Some(
                "only a software (CPU) Vulkan device was found; that is not a GPU path".to_owned(),
            ),
            icds,
            devices,
        };
    }

    let gpu_icds: Vec<&String> = icds.iter().filter(|icd| icd_is_gpu_driver(icd)).collect();
    if gpu_icds.is_empty() {
        // Nothing found at all, and nothing that could have found it: the ICD
        // search is not exhaustive (a fixed directory list on Linux that
        // ignores `VK_ICD_FILENAMES`/`VK_DRIVER_FILES`; only the legacy
        // Khronos key on Windows), so without `vulkaninfo` to ask, an empty
        // result is a gap in the probe rather than a fact about the host.
        // `Absent` renders as "no Vulkan GPU driver was found - install one",
        // which would then be told to a host that has one.
        //
        // `enumerated` is deliberately "the tool is installed", not "the tool
        // returned devices": a host genuinely without a driver HAS
        // `vulkaninfo` and gets nothing from it, and that is a real answer.
        if icds.is_empty() && devices.is_empty() && !enumerated {
            return VulkanSupport {
                status: VulkanStatus::Unknown,
                detail: Some(
                    "no Vulkan driver was found, but `vulkaninfo` is not installed either, so \
                     nothing here confirms one is absent"
                        .to_owned(),
                ),
                icds,
                devices,
            };
        }
        let detail = if icds.is_empty() {
            "no Vulkan driver found".to_owned()
        } else {
            "only a software (CPU) Vulkan driver is installed; that is not a GPU path".to_owned()
        };
        return VulkanSupport {
            status: VulkanStatus::Absent,
            detail: Some(detail),
            icds,
            devices,
        };
    }

    let named = gpu_icds
        .iter()
        .find_map(|icd| icd_driver_label(icd))
        .map_or_else(
            || "a Vulkan driver is installed".to_owned(),
            |label| format!("{label} is installed"),
        );
    VulkanSupport {
        status: VulkanStatus::DriverPresentDevicesUnverified,
        detail: Some(format!(
            "{named}, but no device was enumerated (install `vulkaninfo` to confirm)"
        )),
        icds,
        devices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real `vulkaninfo --summary` output from a Ryzen 7 4700U (gfx90c,
    /// Renoir) on Windows 11 with Adrenalin 31.0.21925.1001 -- the machine
    /// ROCMAI-517 is about. Note the loader warnings ahead of the report and
    /// the instance sections before `Devices:`; both must be skipped.
    const RENOIR_WINDOWS_SUMMARY: &str = "\
WARNING: [Loader Message] Code 0 : windows_read_data_files_in_registry: Registry lookup failed to get layer manifest files.
==========
VULKANINFO
==========

Vulkan Instance Version: 1.3.301


Instance Extensions: count = 13
-------------------------------
VK_EXT_debug_report                    : extension revision 10
VK_KHR_surface                         : extension revision 25

Instance Layers: count = 1
--------------------------
VK_LAYER_AMD_switchable_graphics AMD switchable graphics layer 1.3.260  version 1

Devices:
========
GPU0:
\tapiVersion         = 1.3.260
\tdriverVersion      = 2.0.279
\tvendorID           = 0x1002
\tdeviceID           = 0x1636
\tdeviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
\tdeviceName         = AMD Radeon(TM) Graphics
\tdriverID           = DRIVER_ID_AMD_PROPRIETARY
\tdriverName         = AMD proprietary driver
\tdriverInfo         = 26.5.2 (AMD proprietary shader compiler)
\tconformanceVersion = 1.3.3.1
";

    #[test]
    fn the_renoir_igpu_is_read_out_of_a_real_vulkaninfo_summary() {
        let devices = parse_vulkaninfo_summary(RENOIR_WINDOWS_SUMMARY);

        assert_eq!(devices.len(), 1, "{devices:?}");
        let gpu = &devices[0];
        assert_eq!(gpu.name, "AMD Radeon(TM) Graphics");
        assert_eq!(gpu.vendor_id.as_deref(), Some("0x1002"));
        // The same device id the gfx-target table maps to gfx90c.
        assert_eq!(gpu.device_id.as_deref(), Some("0x1636"));
        assert_eq!(gpu.driver_name.as_deref(), Some("AMD proprietary driver"));
        assert_eq!(gpu.driver_id.as_deref(), Some("DRIVER_ID_AMD_PROPRIETARY"));
        assert_eq!(gpu.api_version.as_deref(), Some("1.3.260"));
        assert!(gpu.is_amd(), "vendor 0x1002 is AMD");
        assert!(gpu.is_hardware(), "an integrated GPU is hardware");
    }

    #[test]
    fn a_real_renoir_host_reports_vulkan_available_and_names_its_driver() {
        let devices = parse_vulkaninfo_summary(RENOIR_WINDOWS_SUMMARY);

        let support = classify_vulkan_support(Vec::new(), devices, true, true);

        assert_eq!(support.status, VulkanStatus::Available);
        assert!(support.status.is_usable());
        assert_eq!(
            support.driver_label().as_deref(),
            Some("AMD proprietary driver"),
            "the device's own driverName must win over any ICD filename guess"
        );
    }

    /// The defect this guards against: lavapipe is a complete Vulkan
    /// implementation that runs on the CPU. Reporting it as a usable Vulkan
    /// path would route a GPU-required launch onto CPU execution while telling
    /// the user a GPU path was chosen -- the silent CPU fallback AGENTS.md §6
    /// forbids, reached through a backend rather than a policy bug.
    #[test]
    fn a_software_rasteriser_is_never_a_usable_vulkan_path() {
        let lavapipe = "\
GPU0:
\tapiVersion   = 1.3.255
\tvendorID     = 0x10005
\tdeviceType   = PHYSICAL_DEVICE_TYPE_CPU
\tdeviceName   = llvmpipe (LLVM 15.0.7, 256 bits)
\tdriverName   = llvmpipe
";
        let devices = parse_vulkaninfo_summary(lavapipe);
        assert_eq!(devices.len(), 1);
        assert!(!devices[0].is_hardware());

        let support =
            classify_vulkan_support(vec!["lvp_icd.x86_64.json".to_owned()], devices, true, true);

        assert_eq!(
            support.status,
            VulkanStatus::Absent,
            "a CPU rasteriser must not read as a Vulkan GPU path: {support:?}"
        );
        assert!(!support.status.is_usable());
        assert!(
            support
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("not a GPU path")),
            "{support:?}"
        );
    }

    #[test]
    fn a_lavapipe_only_install_is_absent_even_before_any_device_is_enumerated() {
        let support = classify_vulkan_support(
            vec!["lvp_icd.x86_64.json".to_owned()],
            Vec::new(),
            true,
            true,
        );

        assert_eq!(support.status, VulkanStatus::Absent);
        assert!(!icd_is_gpu_driver("lvp_icd.x86_64.json"));
        assert!(icd_is_gpu_driver("radeon_icd.x86_64.json"));
    }

    #[test]
    fn an_installed_driver_with_no_enumerated_device_stays_unverified() {
        let support = classify_vulkan_support(
            vec!["radeon_icd.x86_64.json".to_owned()],
            Vec::new(),
            true,
            true,
        );

        assert_eq!(support.status, VulkanStatus::DriverPresentDevicesUnverified);
        assert!(
            support.status.is_usable(),
            "an installed GPU driver is still worth routing to"
        );
        assert_eq!(support.driver_label().as_deref(), Some("Mesa RADV"));
        assert!(
            support
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("no device was enumerated")),
            "the gap must be stated, not papered over: {support:?}"
        );
    }

    /// No ICD and no enumerated device is an absence of evidence, not evidence
    /// of absence — neither source is exhaustive (the Linux ICD search ignores
    /// `VK_ICD_FILENAMES`, the Windows one reads only the legacy registry key,
    /// and `vulkaninfo` is often not installed). `Absent` renders as a
    /// confirmed "no Vulkan GPU driver was found - install one", so a host with
    /// a working driver would be told to install one it already has.
    #[test]
    fn no_evidence_either_way_is_not_reported_as_a_confirmed_absence() {
        // `vulkaninfo` absent: nothing could have found a driver, so nothing
        // may claim there is none.
        let unasked = classify_vulkan_support(Vec::new(), Vec::new(), true, false);
        assert_eq!(
            unasked.status,
            VulkanStatus::Unknown,
            "with no ICD and no way to enumerate, absence is unproven"
        );
        assert!(
            unasked
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("not installed")),
            "the detail must name the missing tool: {unasked:?}"
        );

        // `vulkaninfo` present and it found nothing: that IS the answer, and
        // it is the shape a host genuinely without a driver takes.
        let nothing = classify_vulkan_support(Vec::new(), Vec::new(), true, true);
        assert_eq!(
            nothing.status,
            VulkanStatus::Absent,
            "a probe that could look and found nothing is a real absence"
        );
        assert!(nothing.driver_label().is_none());

        // What `Absent` is now reserved for: something WAS found and ruled out.
        let software_only = classify_vulkan_support(
            vec!["lvp_icd.x86_64.json".to_owned()],
            Vec::new(),
            true,
            true,
        );
        assert_eq!(software_only.status, VulkanStatus::Absent);

        // macOS and friends: no evidence either way, which must not read as
        // "no Vulkan" any more than it reads as "Vulkan works".
        let unprobeable = classify_vulkan_support(Vec::new(), Vec::new(), false, false);
        assert_eq!(unprobeable.status, VulkanStatus::Unknown);
        assert!(!unprobeable.status.is_usable());
    }

    #[test]
    fn an_unknown_vendors_icd_is_assumed_to_be_hardware() {
        // A vendor the table does not list yet must not be mistaken for a
        // software rasteriser; the device enumeration corrects it if wrong.
        assert!(icd_is_gpu_driver("some_new_vendor_icd.x86_64.json"));
        assert_eq!(icd_driver_label("some_new_vendor_icd.x86_64.json"), None);

        let support = classify_vulkan_support(
            vec!["some_new_vendor_icd.x86_64.json".to_owned()],
            Vec::new(),
            true,
            true,
        );
        assert_eq!(support.status, VulkanStatus::DriverPresentDevicesUnverified);
    }

    #[test]
    fn a_discrete_amd_card_is_preferred_over_a_software_device_in_the_same_report() {
        let mixed = "\
GPU0:
\tvendorID     = 0x10005
\tdeviceType   = PHYSICAL_DEVICE_TYPE_CPU
\tdeviceName   = llvmpipe
\tdriverName   = llvmpipe
GPU1:
\tvendorID     = 0x1002
\tdeviceType   = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
\tdeviceName   = AMD Radeon RX 7900 XTX
\tdriverName   = radv
";
        let support =
            classify_vulkan_support(Vec::new(), parse_vulkaninfo_summary(mixed), true, true);

        assert_eq!(support.status, VulkanStatus::Available);
        assert_eq!(support.driver_label().as_deref(), Some("radv"));
        assert!(
            support
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("AMD Radeon RX 7900 XTX")),
            "{support:?}"
        );
    }

    #[test]
    fn a_non_amd_gpu_does_not_make_vulkan_an_amd_path() {
        let nvidia = "\
GPU0:
\tvendorID     = 0x10de
\tdeviceType   = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
\tdeviceName   = NVIDIA GeForce GTX 1050 Ti
\tdriverName   = NVIDIA
";
        let support = classify_vulkan_support(
            vec!["nvidia_icd.json".to_owned()],
            parse_vulkaninfo_summary(nvidia),
            true,
            true,
        );

        // The driver is installed and is not software, so the honest answer is
        // "unverified for AMD", not "available".
        assert_eq!(support.status, VulkanStatus::DriverPresentDevicesUnverified);
    }

    #[test]
    fn the_windows_registry_listing_is_read_into_icd_paths() {
        let output = "ICD\tC:\\Windows\\System32\\DriverStore\\FileRepository\\u0397342.inf_amd64_x\\amdvlk64.json\nICD\tC:\\Windows\\SysWOW64\\amdvlk32.json\n";

        let icds = parse_windows_vulkan_icd_listing(output);

        assert_eq!(icds.len(), 2, "{icds:?}");
        assert!(icds[0].ends_with("amdvlk64.json"));
        assert_eq!(icd_driver_label(&icds[0]), Some("AMDVLK"));
        assert!(
            icd_is_gpu_driver(&icds[0]),
            "a Windows path separator must not defeat the filename table"
        );
    }

    /// Both Khronos keys must be queried. A 32-bit AMDVLK registration lives
    /// only under `WOW6432Node`, so reading the 64-bit key alone would report
    /// "no Vulkan driver" on a host that has one.
    #[test]
    fn the_windows_icd_query_reads_both_the_native_and_the_wow6432_key() {
        assert!(
            WINDOWS_VULKAN_ICD_SCRIPT.contains(r"HKLM:\SOFTWARE\Khronos\Vulkan\Drivers"),
            "{WINDOWS_VULKAN_ICD_SCRIPT}"
        );
        assert!(
            WINDOWS_VULKAN_ICD_SCRIPT
                .contains(r"HKLM:\SOFTWARE\WOW6432Node\Khronos\Vulkan\Drivers"),
            "{WINDOWS_VULKAN_ICD_SCRIPT}"
        );
        // A missing key is ordinary on a host with no Vulkan at all, so the
        // script must test for it rather than let PowerShell error out.
        assert!(
            WINDOWS_VULKAN_ICD_SCRIPT.contains("Test-Path"),
            "{WINDOWS_VULKAN_ICD_SCRIPT}"
        );
        // The rows the parser consumes.
        assert!(
            WINDOWS_VULKAN_ICD_SCRIPT.contains("\"ICD`t$name\""),
            "{WINDOWS_VULKAN_ICD_SCRIPT}"
        );
    }

    #[test]
    fn an_empty_registry_key_yields_no_icds_rather_than_a_blank_entry() {
        assert!(parse_windows_vulkan_icd_listing("").is_empty());
        assert!(parse_windows_vulkan_icd_listing("ICD\t\n").is_empty());
        assert!(parse_windows_vulkan_icd_listing("unrelated output\n").is_empty());
    }

    #[test]
    fn a_vulkaninfo_report_with_no_devices_yields_none() {
        assert!(parse_vulkaninfo_summary("Vulkan Instance Version: 1.3.301\n").is_empty());
        assert!(parse_vulkaninfo_summary("").is_empty());
    }
}
