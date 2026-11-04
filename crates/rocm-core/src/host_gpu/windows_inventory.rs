// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The Windows display-adapter inventory: the PowerShell queries, the parsers for
//! their output, and the assembly of one inventory from several sources.

#[cfg(windows)]
use super::capture::{
    OPTIONAL_COMMAND_TIMEOUT, capture_optional_command, capture_optional_command_with_timeout,
};
use super::os_facts::{bytes_text_to_gib, normalize_cpu_model};
use crate::gfx_target::{
    amd_pci_device_id_from_pnp_id, gfx_target_from_amd_marketing_name,
    gfx_target_from_amd_pci_device_id, gfx_target_has_no_rocm_build,
    parse_windows_display_gfx_target,
};
use crate::runtime_is_windows;
use std::time::Duration;

pub(super) const WINDOWS_INVENTORY_QUERY_TIMEOUT: Duration = Duration::from_secs(5);

pub(super) const WINDOWS_VIDEO_CONTROLLER_INVENTORY_SCRIPT: &str = r#"$gpus = Get-CimInstance -ClassName Win32_VideoController -Property Name,DriverVersion,DriverDate,PNPDeviceID,AdapterCompatibility | Where-Object { $_.PNPDeviceID -match 'VEN_1002' -or $_.AdapterCompatibility -match 'AMD|Advanced Micro Devices' -or $_.Name -match 'AMD|Radeon|Instinct' }; foreach ($gpu in $gpus) { $date = ''; if ($gpu.DriverDate) { $date = $gpu.DriverDate.ToUniversalTime().ToString('yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture) }; "GPU`t$($gpu.Name)`t$($gpu.DriverVersion)`t$($gpu.PNPDeviceID)`t$date" }"#;

#[cfg(windows)]
pub(super) const WINDOWS_PNP_ENTITY_INVENTORY_SCRIPT: &str = r#"$displayGuid = '{4d36e968-e325-11ce-bfc1-08002be10318}'; $gpus = Get-CimInstance -ClassName Win32_PnPEntity -Property Name,DeviceID,PNPClass,ClassGuid,Manufacturer | Where-Object { (($_.PNPClass -eq 'Display' -or $_.ClassGuid -eq $displayGuid) -and ($_.DeviceID -match 'VEN_1002' -or $_.Name -match 'AMD|Radeon|Instinct|Graphics' -or $_.Manufacturer -match 'AMD|Advanced Micro Devices')) -or ($_.DeviceID -match 'PCI\\VEN_1002' -and $_.Name -match 'Radeon|Instinct|Graphics') }; foreach ($gpu in $gpus) { "GPU`t$($gpu.Name)`t`t$($gpu.DeviceID)" }"#;

#[cfg(windows)]
const WINDOWS_SYSTEM_INVENTORY_SCRIPT: &str = r#"$cpu = Get-CimInstance -ClassName Win32_Processor -Property Name | Select-Object -First 1 -ExpandProperty Name; if ($cpu) { "CPU`t$cpu" }; $ram = Get-CimInstance -ClassName Win32_ComputerSystem -Property TotalPhysicalMemory | Select-Object -First 1 -ExpandProperty TotalPhysicalMemory; if ($ram) { "RAM`t$ram" }"#;

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

pub(super) fn detect_windows_display_gfx_target_with_inventory(
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

#[derive(Debug, Clone, Default)]
pub(super) struct WindowsExamineInventory {
    pub(super) cpu_model: Option<String>,
    pub(super) system_ram_gib: Option<f64>,
    pub(super) displays: Vec<WindowsDisplayAdapter>,
}

#[derive(Debug, Clone)]
pub(super) struct WindowsDisplayAdapter {
    pub(super) name: String,
    pub(super) driver_version: Option<String>,
    /// `yyyy-MM-dd`, only when the probe gave a date that parses.
    driver_date: Option<String>,
    pub(super) pnp_device_id: Option<String>,
}

impl WindowsExamineInventory {
    #[cfg(any(windows, test))]
    const fn is_empty(&self) -> bool {
        self.cpu_model.is_none() && self.system_ram_gib.is_none() && self.displays.is_empty()
    }

    #[cfg(any(windows, test))]
    fn merge_missing_from(&mut self, mut other: Self) {
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
                if existing.driver_date.is_none() {
                    existing.driver_date = display.driver_date;
                }
                if existing.pnp_device_id.is_none() {
                    existing.pnp_device_id = display.pnp_device_id;
                }
            } else {
                self.displays.push(display);
            }
        }
    }

    pub(super) fn amd_display_driver_detail(&self) -> Option<String> {
        let display = self.preferred_amd_display()?;
        let name = display.name.trim();
        if name.is_empty() {
            return None;
        }
        // The Microsoft fallback driver is not an AMD driver, so the legacy
        // `driver_status` and `driver_detail` fields must not report it as one
        // either (see `amd_display_driver_info`).
        if crate::windows_driver::is_microsoft_fallback_driver(
            name,
            display.driver_version.as_deref().unwrap_or(""),
        ) {
            return None;
        }
        let detail = format!(
            "{name} driver {}",
            display.driver_version.as_deref().unwrap_or("")
        );
        Some(detail.trim().to_owned())
    }

    /// The preferred AMD adapter's driver version and date, when a version was
    /// read. `None` when there is no adapter or its version is unknown.
    ///
    /// The one place all three commands (`rocm examine`, `rocm install driver`
    /// and `rocm diagnose`) read the driver from. The Microsoft fallback driver
    /// is not an AMD driver, so it gives `None` (see
    /// [`crate::windows_driver::is_microsoft_fallback_driver`]).
    pub(super) fn amd_display_driver_info(
        &self,
    ) -> Option<crate::windows_driver::WindowsDriverInfo> {
        let display = self.preferred_amd_display()?;
        let version = display.driver_version.as_deref()?;
        if crate::windows_driver::is_microsoft_fallback_driver(&display.name, version) {
            return None;
        }
        crate::windows_driver::WindowsDriverInfo::from_probe(
            version,
            display.driver_date.as_deref().unwrap_or(""),
        )
    }

    /// The preferred AMD adapter's driver version, trimmed, or `None` when it has
    /// none or it is the Microsoft fallback driver, which is not an AMD driver.
    /// The host-driver probes read it from here so they apply the same
    /// exclusion as [`Self::amd_display_driver_info`].
    pub(super) fn amd_display_driver_version(&self) -> Option<String> {
        let display = self.preferred_amd_display()?;
        let version = display.driver_version.as_deref()?.trim();
        (!version.is_empty()
            && !crate::windows_driver::is_microsoft_fallback_driver(&display.name, version))
        .then(|| version.to_owned())
    }

    /// Whether the preferred AMD adapter still lacks a version or a date, so a
    /// second query is worth making.
    #[cfg(any(windows, test))]
    fn preferred_amd_display_misses_driver_facts(&self) -> bool {
        self.preferred_amd_display()
            .is_some_and(|d| d.driver_version.is_none() || d.driver_date.is_none())
    }

    pub(super) fn amd_display_name(&self) -> Option<String> {
        self.preferred_amd_display()
            .map(|display| display.name.trim())
            .filter(|name| !name.is_empty())
            .map(str::to_owned)
    }

    pub(super) fn preferred_amd_display(&self) -> Option<&WindowsDisplayAdapter> {
        let pci_target = |display: &WindowsDisplayAdapter| {
            display
                .pnp_device_id
                .as_deref()
                .and_then(amd_pci_device_id_from_pnp_id)
                .and_then(|device_id| gfx_target_from_amd_pci_device_id(&device_id))
        };
        self.displays
            .iter()
            .find(|display| pci_target(display).is_some_and(|t| !gfx_target_has_no_rocm_build(t)))
            .or_else(|| {
                self.displays
                    .iter()
                    .find(|display| gfx_target_from_amd_marketing_name(&display.name).is_some())
            })
            // A recognized-but-unsupported iGPU (gfx90c) ranks below every
            // other recognized adapter, so it cannot hide a discrete card.
            .or_else(|| {
                self.displays
                    .iter()
                    .find(|display| pci_target(display).is_some())
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

    pub(super) fn display_gfx_probe_text(&self) -> String {
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

#[cfg(windows)]
pub(super) fn detect_windows_amd_display_driver() -> Option<String> {
    if !runtime_is_windows() {
        return None;
    }
    let script = "$gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.AdapterCompatibility -match 'AMD|Advanced Micro Devices' -or $_.Name -match 'AMD|Radeon|Instinct' } | Select-Object -First 1 -Property Name,DriverVersion; if ($gpu) { \"$($gpu.Name) driver $($gpu.DriverVersion)\" }";
    capture_optional_command("powershell", &["-NoProfile", "-Command", script])
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

#[cfg(not(windows))]
pub(super) const fn detect_windows_amd_display_driver() -> Option<String> {
    None
}

/// Which optional queries one inventory run may start.
///
/// The display list always comes from the cheapest source that has it. These
/// choose what is added on top, so a caller that never shows the driver version
/// or date does not start a process for it.
#[cfg(any(windows, test))]
#[derive(Debug, Clone, Copy)]
pub(super) struct InventoryScope {
    /// Fill a missing CPU model or RAM size from a system query.
    system: bool,
    /// Ask the video controller for the preferred adapter's driver version and
    /// date when the first source lacks them.
    driver_facts: bool,
}

#[cfg(any(windows, test))]
impl InventoryScope {
    /// Startup and engine-selection paths (`detect_host_gpu_summary_fast`, chat
    /// host facts): no driver facts. `pnputil`, the first
    /// source, never reports them, so asking for them would start a PowerShell
    /// process on every AMD host each time.
    const FAST: Self = Self {
        system: true,
        driver_facts: false,
    };
    /// `rocm examine` and `rocm install driver`, which print the driver.
    pub(super) const SUMMARY: Self = Self {
        system: true,
        driver_facts: true,
    };
    /// `rocm diagnose`, which needs the driver but not the CPU or RAM.
    pub(super) const DRIVER_ONLY: Self = Self {
        system: false,
        driver_facts: true,
    };
}

/// The queries an inventory is built from, so a test can count which ones run.
#[cfg(any(windows, test))]
pub(super) trait WindowsInventorySources {
    fn pnputil(&mut self) -> Option<WindowsExamineInventory>;
    fn video_controller(&mut self) -> Option<WindowsExamineInventory>;
    fn pnp_entity(&mut self) -> Option<WindowsExamineInventory>;
    fn system(&mut self) -> Option<WindowsExamineInventory>;
}

#[cfg(windows)]
struct LiveWindowsInventorySources;

#[cfg(windows)]
impl WindowsInventorySources for LiveWindowsInventorySources {
    fn pnputil(&mut self) -> Option<WindowsExamineInventory> {
        detect_windows_examine_inventory_from_pnputil()
    }
    fn video_controller(&mut self) -> Option<WindowsExamineInventory> {
        detect_windows_examine_inventory_from_video_controller()
    }
    fn pnp_entity(&mut self) -> Option<WindowsExamineInventory> {
        detect_windows_examine_inventory_from_pnp_entity()
    }
    fn system(&mut self) -> Option<WindowsExamineInventory> {
        detect_windows_system_inventory_from_cim()
    }
}

#[cfg(any(windows, test))]
pub(super) fn assemble_windows_inventory(
    sources: &mut impl WindowsInventorySources,
    scope: InventoryScope,
) -> Option<WindowsExamineInventory> {
    let mut inventory = WindowsExamineInventory::default();
    if let Some(pnp_util) = sources.pnputil() {
        inventory.merge_missing_from(pnp_util);
    }
    let mut video_asked = false;
    if inventory.displays.is_empty() {
        video_asked = true;
        if let Some(video) = sources.video_controller() {
            inventory.merge_missing_from(video);
        }
    }
    if inventory.displays.is_empty()
        && let Some(pnp) = sources.pnp_entity()
    {
        inventory.merge_missing_from(pnp);
    }
    // `pnputil` lists the adapter but not its driver version or date. Only a
    // caller that shows them asks the video controller for the two facts, and
    // only when something is still missing and that query has not just run.
    if scope.driver_facts
        && !video_asked
        && inventory.preferred_amd_display_misses_driver_facts()
        && let Some(video) = sources.video_controller()
    {
        inventory.merge_missing_from(video);
    }
    if scope.system
        && (inventory.cpu_model.is_none() || inventory.system_ram_gib.is_none())
        && let Some(system) = sources.system()
    {
        inventory.merge_missing_from(system);
    }

    (!inventory.is_empty()).then_some(inventory)
}

/// The inventory for startup and engine selection. Starts no query for the
/// driver version or date; see [`InventoryScope::FAST`].
#[cfg(windows)]
pub(super) fn detect_windows_examine_inventory() -> Option<WindowsExamineInventory> {
    if !runtime_is_windows() {
        return None;
    }
    assemble_windows_inventory(&mut LiveWindowsInventorySources, InventoryScope::FAST)
}

/// The inventory for `rocm examine` and `rocm install driver`, with the
/// driver version and date filled in where the first source lacks them.
#[cfg(windows)]
pub(super) fn detect_windows_examine_inventory_with_driver_facts() -> Option<WindowsExamineInventory>
{
    if !runtime_is_windows() {
        return None;
    }
    assemble_windows_inventory(&mut LiveWindowsInventorySources, InventoryScope::SUMMARY)
}

/// The inventory for a caller that needs the display driver but not the CPU
/// or RAM.
#[cfg(windows)]
fn detect_windows_driver_inventory() -> Option<WindowsExamineInventory> {
    if !runtime_is_windows() {
        return None;
    }
    assemble_windows_inventory(
        &mut LiveWindowsInventorySources,
        InventoryScope::DRIVER_ONLY,
    )
}

#[cfg(not(windows))]
const fn detect_windows_driver_inventory() -> Option<WindowsExamineInventory> {
    None
}

/// The AMD display driver as `rocm examine` reports it, for `rocm diagnose`.
///
/// Uses the inventory pipeline and adapter selection of `ExamineSummary::gather`.
/// Separate calls can observe different inventory results or probe failures.
pub(crate) fn detect_windows_amd_driver_info() -> Option<crate::windows_driver::WindowsDriverInfo> {
    detect_windows_driver_inventory()?.amd_display_driver_info()
}

#[cfg(not(windows))]
pub(super) const fn detect_windows_examine_inventory_with_driver_facts()
-> Option<WindowsExamineInventory> {
    None
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
pub(super) const fn detect_windows_examine_inventory() -> Option<WindowsExamineInventory> {
    None
}

#[cfg(any(windows, test))]
fn clean_windows_display_name(value: &str) -> String {
    let value = value.trim();
    let value = value.rsplit_once(';').map_or(value, |(_, name)| name);
    value.trim().to_owned()
}

#[cfg_attr(not(windows), allow(dead_code))]
pub(super) fn parse_windows_examine_inventory(text: &str) -> WindowsExamineInventory {
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
                // Optional fifth field. A value that is not a date is dropped,
                // so an unreadable date is never mistaken for an old one.
                let driver_date = fields
                    .next()
                    .and_then(crate::windows_driver::CalendarDate::parse)
                    .map(|date| date.to_string());
                if !name.is_empty() || driver_version.is_some() || pnp_device_id.is_some() {
                    inventory.displays.push(WindowsDisplayAdapter {
                        name,
                        driver_version,
                        driver_date,
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
pub(super) fn parse_windows_pnputil_display_inventory(text: &str) -> WindowsExamineInventory {
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
        driver_date: None,
        pnp_device_id: pnp,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host_gpu::tests::{CountingSources, DGPU_CURRENT, PNPUTIL_ONLY, pnputil_host};

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

    /// The CIM driver date is a local-time `DateTime` for an instant that is
    /// midnight UTC, so formatting it unconverted gives the previous day west of
    /// UTC and would flag a driver dated exactly on the cutoff. This pins the
    /// script text only; the behaviour was checked by converting a real
    /// `DriverDate` into a UTC-8 zone in PowerShell (previous day without the
    /// conversion, the INF day with it).
    #[test]
    fn the_inventory_script_formats_the_driver_date_in_utc() {
        assert!(WINDOWS_VIDEO_CONTROLLER_INVENTORY_SCRIPT.contains(
            "$gpu.DriverDate.ToUniversalTime().ToString('yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)"
        ));
    }

    #[test]
    fn the_inventory_parser_reads_the_driver_date_and_drops_a_bad_one() {
        let inventory = parse_windows_examine_inventory(
            "GPU\tAMD Radeon(TM) Graphics\t30.0.13014.8\tPCI\\VEN_1002&DEV_1636\t2021-08-04\n",
        );
        let info = inventory.amd_display_driver_info().expect("info");
        assert_eq!(info.version, "30.0.13014.8");
        assert_eq!(
            info.date.map(|d| d.to_string()).as_deref(),
            Some("2021-08-04")
        );

        let bad = parse_windows_examine_inventory(
            "GPU\tAMD Radeon(TM) Graphics\t30.0.13014.8\tPCI\\VEN_1002&DEV_1636\tnot a date\n",
        );
        assert_eq!(bad.amd_display_driver_info().expect("info").date, None);

        let no_date = parse_windows_examine_inventory(
            "GPU\tAMD Radeon(TM) Graphics\t30.0.13014.8\tPCI\\VEN_1002&DEV_1636\n",
        );
        assert_eq!(no_date.amd_display_driver_info().expect("info").date, None);
    }

    #[test]
    fn an_adapter_missing_its_version_asks_for_a_second_query() {
        let from_pnputil = parse_windows_examine_inventory(
            "GPU\tAMD Radeon(TM) Graphics\t\tPCI\\VEN_1002&DEV_1636\n",
        );
        assert!(from_pnputil.preferred_amd_display_misses_driver_facts());
        let complete = parse_windows_examine_inventory(
            "GPU\tAMD Radeon(TM) Graphics\t30.0.13014.8\tPCI\\VEN_1002&DEV_1636\t2021-08-04\n",
        );
        assert!(!complete.preferred_amd_display_misses_driver_facts());
    }

    #[test]
    fn a_version_less_adapter_gives_no_driver_info_and_is_filled_from_the_video_controller() {
        // The pnputil shape: the adapter is listed with no driver version.
        let mut inventory = parse_windows_examine_inventory(
            "GPU\tAMD Radeon(TM) Graphics\t\tPCI\\VEN_1002&DEV_1636\n",
        );
        assert_eq!(inventory.amd_display_driver_info(), None);
        assert_eq!(
            inventory.amd_display_driver_detail().as_deref(),
            Some("AMD Radeon(TM) Graphics driver")
        );
        // The video controller answers with the same adapter id.
        inventory.merge_missing_from(parse_windows_examine_inventory(
            "GPU\tAMD Radeon(TM) Graphics\t30.0.13014.8\tPCI\\VEN_1002&DEV_1636\t2021-08-04\n",
        ));
        assert_eq!(
            inventory.amd_display_driver_detail().as_deref(),
            Some("AMD Radeon(TM) Graphics driver 30.0.13014.8")
        );
        let info = inventory.amd_display_driver_info().expect("filled");
        assert_eq!(
            info.date.map(|d| d.to_string()).as_deref(),
            Some("2021-08-04")
        );
        assert_eq!(
            inventory.displays.len(),
            1,
            "the adapter must not duplicate"
        );
    }

    #[test]
    fn the_startup_paths_do_not_start_the_driver_fact_query() {
        let mut sources = pnputil_host();
        let inventory =
            assemble_windows_inventory(&mut sources, InventoryScope::FAST).expect("an inventory");
        assert!(
            !sources.calls.contains(&"video"),
            "the fast paths run on every `rocm serve`, chat start and engine pick, and must not start a PowerShell query for a date they never print: {:?}",
            sources.calls
        );
        // The behaviour that follows: no version, so nothing to flag or print.
        assert_eq!(inventory.amd_display_driver_info(), None);
    }

    #[test]
    fn the_summary_path_starts_the_driver_fact_query_once() {
        let mut sources = pnputil_host();
        let inventory = assemble_windows_inventory(&mut sources, InventoryScope::SUMMARY)
            .expect("an inventory");
        assert_eq!(
            sources.calls.iter().filter(|c| **c == "video").count(),
            1,
            "{:?}",
            sources.calls
        );
        let info = inventory.amd_display_driver_info().expect("filled");
        assert_eq!(info.version, "32.0.23033.1002");
    }

    #[test]
    fn the_diagnose_path_gets_the_driver_without_the_cpu_and_ram_query() {
        let mut sources = pnputil_host();
        let inventory = assemble_windows_inventory(&mut sources, InventoryScope::DRIVER_ONLY)
            .expect("an inventory");
        assert!(!sources.calls.contains(&"system"), "{:?}", sources.calls);
        assert!(inventory.amd_display_driver_info().is_some());
    }

    #[test]
    fn a_complete_first_source_needs_no_second_query_and_none_repeats_one() {
        let mut complete = CountingSources {
            pnputil: Some(DGPU_CURRENT),
            ..CountingSources::default()
        };
        assemble_windows_inventory(&mut complete, InventoryScope::SUMMARY);
        assert!(!complete.calls.contains(&"video"), "{:?}", complete.calls);

        // No adapter from `pnputil`: the video controller is the first source,
        // and a second ask for the same facts would repeat it.
        let mut empty_first = CountingSources {
            video: Some(PNPUTIL_ONLY),
            ..CountingSources::default()
        };
        assemble_windows_inventory(&mut empty_first, InventoryScope::SUMMARY);
        assert_eq!(
            empty_first.calls.iter().filter(|c| **c == "video").count(),
            1,
            "{:?}",
            empty_first.calls
        );
    }

    /// A version without a date is the shape the date-only branch exists for:
    /// `|| d.driver_date.is_none()` in the predicate.
    #[test]
    fn an_adapter_with_a_version_and_no_date_still_asks_for_the_date() {
        let version_only = parse_windows_examine_inventory(
            "GPU\tAMD Radeon RX 6600\t32.0.23033.1002\tPCI\\VEN_1002&DEV_73E0\n",
        );
        assert!(version_only.preferred_amd_display_misses_driver_facts());

        let mut sources = CountingSources {
            pnputil: Some("GPU\tAMD Radeon RX 6600\t32.0.23033.1002\tPCI\\VEN_1002&DEV_73E0\n"),
            video: Some(DGPU_CURRENT),
            ..CountingSources::default()
        };
        let inventory = assemble_windows_inventory(&mut sources, InventoryScope::SUMMARY)
            .expect("an inventory");
        assert!(sources.calls.contains(&"video"), "{:?}", sources.calls);
        let info = inventory.amd_display_driver_info().expect("info");
        assert_eq!(
            info.date.map(|d| d.to_string()).as_deref(),
            Some("2026-05-02")
        );
    }
}
