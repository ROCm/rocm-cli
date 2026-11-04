// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! WSL detection and the Windows host driver as seen from the guest: the WSL2
//! signals, the librocdxg chain, and the local and WSL host-driver probes.

use super::capture::{
    capture_optional_command, capture_optional_command_with_timeout, tool_on_path,
};
use super::windows_inventory::{
    WINDOWS_INVENTORY_QUERY_TIMEOUT, WINDOWS_VIDEO_CONTROLLER_INVENTORY_SCRIPT,
    WindowsExamineInventory, detect_windows_examine_inventory_with_driver_facts,
    parse_windows_examine_inventory,
};
use super::{DriverSummary, WslSummary};
use crate::gfx_target::parse_windows_display_gfx_target;
use crate::{discover_rocm_installs, host_path, runtime_is_linux, runtime_is_windows};
use std::fs;

pub(super) fn detect_wsl_windows_display_gfx_target_fast() -> Option<String> {
    detect_wsl_windows_display_probe_text()
        .as_deref()
        .and_then(parse_windows_display_gfx_target)
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
    // This one reads the driver version, which `pnputil` never reports, so it
    // asks for the driver facts. It also keeps the CPU and RAM query: on a host
    // with no AMD adapter that is what makes the inventory non-empty, and an
    // empty inventory reads as "could not ask" rather than "no adapter". It is
    // not a startup path.
    local_windows_host_driver_probe(detect_windows_examine_inventory_with_driver_facts())
}

/// Map an inventory result to the host-driver answer. `None` means no source
/// answered; an inventory without an AMD display means the host answered and has
/// none.
pub(super) fn local_windows_host_driver_probe(
    inventory: Option<WindowsExamineInventory>,
) -> WslHostDriverProbe {
    let Some(inventory) = inventory else {
        return WslHostDriverProbe::Unreachable;
    };
    inventory.amd_display_driver_version().map_or(
        WslHostDriverProbe::NoAmdDisplay,
        WslHostDriverProbe::Version,
    )
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
        .amd_display_driver_version()
        .map_or(
            WslHostDriverProbe::NoAmdDisplay,
            WslHostDriverProbe::Version,
        )
}

pub(super) fn detect_wsl_windows_display_probe_text() -> Option<String> {
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
            host_path("/dev/dxg").exists(),
            &fs::read_to_string(host_path("/proc/version")).unwrap_or_default(),
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
///
/// Deliberately not routed through [`host_path`]: these are programs this
/// process executes, not files it reads to describe the machine, and a
/// simulated host root holds no working linker to run.
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
    if host_path("/opt/rocm").join(relative).exists() {
        return true;
    }
    // Discovery reports real install paths; reading them through `host_path`
    // keeps a simulated host's answer from being decided by the real machine.
    discover_rocm_installs()
        .iter()
        .any(|install| host_path(&install.path).join(relative).exists())
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

    let dxg_device = host_path("/dev/dxg").exists();
    let is_wsl = true;

    let dxcore = host_path("/usr/lib/wsl/lib/libdxcore.so").exists();
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

pub(super) fn wsl_driver_summary(wsl: &WslSummary) -> DriverSummary {
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
        windows: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host_gpu::tests::{CountingSources, pnputil_host};
    use crate::host_gpu::windows_inventory::{InventoryScope, assemble_windows_inventory};

    /// A host that answers but has no AMD adapter reads as `NoAmdDisplay`; a host
    /// where no source answers reads as `Unreachable`. The CPU and RAM answer is
    /// what makes the first inventory non-empty, so the local host-driver check
    /// has to keep that query (a `DRIVER_ONLY` run on the same host is empty and
    /// reads as unreachable).
    #[test]
    fn the_local_host_driver_probe_tells_no_adapter_from_no_answer() {
        let no_adapter = || CountingSources {
            system: Some("CPU\tIntel Core i7\nRAM\t17179869184\n"),
            ..CountingSources::default()
        };
        let mut answered = no_adapter();
        assert_eq!(
            local_windows_host_driver_probe(assemble_windows_inventory(
                &mut answered,
                InventoryScope::SUMMARY
            )),
            WslHostDriverProbe::NoAmdDisplay
        );
        let mut driver_only = no_adapter();
        assert_eq!(
            assemble_windows_inventory(&mut driver_only, InventoryScope::DRIVER_ONLY).map(|_| ()),
            None,
            "DRIVER_ONLY asks for no system data, so this host reads as empty"
        );
        let mut silent = CountingSources::default();
        assert_eq!(
            local_windows_host_driver_probe(assemble_windows_inventory(
                &mut silent,
                InventoryScope::SUMMARY
            )),
            WslHostDriverProbe::Unreachable
        );
        let mut with_adapter = pnputil_host();
        assert_eq!(
            local_windows_host_driver_probe(assemble_windows_inventory(
                &mut with_adapter,
                InventoryScope::SUMMARY
            )),
            WslHostDriverProbe::Version("32.0.23033.1002".to_owned())
        );
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
