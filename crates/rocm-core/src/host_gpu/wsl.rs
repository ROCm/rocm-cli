// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! WSL detection and the Windows host driver as seen from the guest: the WSL2
//! signals, the librocdxg chain, and the local and WSL host-driver probes.

use super::capture::{capture_optional_command, capture_optional_command_with_timeout};
use super::windows_inventory::{
    WINDOWS_INVENTORY_QUERY_TIMEOUT, WINDOWS_VIDEO_CONTROLLER_INVENTORY_SCRIPT,
    WindowsExamineInventory, detect_windows_examine_inventory_with_driver_facts,
    parse_windows_examine_inventory,
};
use super::{DriverSummary, WslSummary};
use crate::gfx_target::parse_windows_display_gfx_target;
use crate::probe_exec::which;
use crate::{discover_rocm_installs, host_path, runtime_is_linux, runtime_is_windows};
use std::fs;
use std::path::{Path, PathBuf};

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
/// simulated host root holds no working linker to run. The e2e suite answers
/// the bare `ldconfig` with a stand-in on `PATH` instead; the absolute
/// fallbacks still reach the real host's linker when `PATH` has none.
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

/// The ROCm install roots on this host: the conventional `/opt/rocm` first, then
/// every install discovery reports.
///
/// The WSL probe used to hardcode `/opt/rocm`, so a versioned install at
/// `/opt/rocm-7.x` reported ROCDXG missing and the catalog would then blame a
/// package that was in fact installed. Ask the same resolver the rest of the CLI
/// uses, and keep the conventional root as a fallback for the case where
/// discovery finds nothing.
fn rocm_roots() -> Vec<PathBuf> {
    // Discovery reports real install paths; reading them through `host_path`
    // keeps a simulated host's answer from being decided by the real machine.
    std::iter::once(host_path("/opt/rocm"))
        .chain(
            discover_rocm_installs()
                .iter()
                .map(|install| host_path(&install.path)),
        )
        .collect()
}

/// Whether `relative` exists under any of `roots`.
fn rocm_relative_file_exists(roots: &[PathBuf], relative: &str) -> bool {
    roots.iter().any(|root| root.join(relative).exists())
}

/// Whether `librocdxg.so` sits in a directory on `ld_library_path`.
///
/// The dynamic loader searches `LD_LIBRARY_PATH` before it consults the linker
/// cache, so a library there is loadable whether or not `ldconfig` ever heard of
/// it. A TheRock tarball under `/opt/rocm-<version>` is the case that matters:
/// nothing registers it with the cache, so a shell that has put the install on
/// `LD_LIBRARY_PATH` (see `docs/wsl.md`) reaches ROCm only through this
/// variable. The value read is the invoking process's own, not the managed
/// environment the CLI applies to the commands it launches.
///
/// Takes the variable's value and an existence check rather than reading the
/// machine, so the path-splitting rules can be tested without one. Empty entries
/// are skipped: the loader reads an empty entry as the current directory, which
/// says nothing about where ROCm lives.
fn librocdxg_on_loader_path(ld_library_path: Option<&str>, exists: impl Fn(&Path) -> bool) -> bool {
    ld_library_path.is_some_and(|value| {
        std::env::split_paths(value)
            .filter(|dir| !dir.as_os_str().is_empty())
            .any(|dir| exists(&dir.join("librocdxg.so")))
    })
}

/// Whether some ROCm install holds `librocdxg.so` in the same `lib` directory as
/// a `libhsa-runtime64`.
///
/// The runtime opens `librocdxg.so` by name, and TheRock builds it with
/// `RUNPATH $ORIGIN`, so the loader looks in the runtime's own directory before it
/// asks the linker cache or `LD_LIBRARY_PATH`. A tarball under
/// `/opt/rocm-<version>` keeps the two side by side and registers neither, which
/// is why a working install read as missing while those were the only sources
/// asked (ROCM-30386). The library alone in a `lib` directory does not count: with
/// no runtime beside it, only the cache or `LD_LIBRARY_PATH` can reach it.
///
/// Takes the install roots and an existence check rather than reading the
/// machine, so the layouts can be tested without one.
fn librocdxg_beside_runtime(roots: &[PathBuf], exists: impl Fn(&Path) -> bool) -> bool {
    roots.iter().any(|root| {
        let lib = root.join("lib");
        exists(&lib.join("librocdxg.so"))
            && ["libhsa-runtime64.so.1", "libhsa-runtime64.so"]
                .iter()
                .any(|runtime| exists(&lib.join(runtime)))
    })
}

/// Whether the loader can resolve `librocdxg.so`: registered in the linker cache,
/// on `LD_LIBRARY_PATH`, or beside the runtime that opens it.
///
/// The one place the sources are combined, so `rocdxg_ready`, the `missing`
/// detail and the diagnose guards cannot disagree about it.
const fn librocdxg_loadable(
    in_linker_cache: bool,
    on_ld_library_path: bool,
    beside_runtime: bool,
) -> bool {
    in_linker_cache || on_ld_library_path || beside_runtime
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

/// The entries of the `missing ...` detail, separated from reading the machine so
/// the list can be tested against each combination of sources.
///
/// `librocdxg_found` is the file existing under a ROCm install or in a directory
/// on `LD_LIBRARY_PATH`; `loadable` is [`librocdxg_loadable`]. The cache entry is
/// named only when no source resolves the library: one on `LD_LIBRARY_PATH` or
/// beside the runtime that opens it loads without a cache entry, so naming the
/// cache would send the user after a registration the workload does not need.
#[allow(clippy::fn_params_excessive_bools)] // four independent probe answers, named at the one call site
fn missing_plumbing(
    dxg_device: bool,
    dxcore: bool,
    librocdxg_found: bool,
    loadable: bool,
) -> Vec<&'static str> {
    let mut missing = Vec::new();
    if !dxg_device {
        missing.push("/dev/dxg");
    }
    if !dxcore {
        missing.push("/usr/lib/wsl/lib/libdxcore.so");
    }
    if !librocdxg_found {
        // Named without a directory: the file is looked up across every ROCm
        // install and `LD_LIBRARY_PATH`, so quoting one root would misreport where
        // it was not found.
        missing.push("librocdxg.so");
    }
    if !loadable {
        missing.push("ldconfig:librocdxg.so");
    }
    missing
}

pub(crate) fn detect_wsl_summary() -> Option<WslSummary> {
    if !runtime_is_linux() || !is_wsl_host() {
        return None;
    }

    let dxg_device = host_path("/dev/dxg").exists();
    let is_wsl = true;

    let dxcore = host_path("/usr/lib/wsl/lib/libdxcore.so").exists();
    let roots = rocm_roots();
    let librocdxg = rocm_relative_file_exists(&roots, "lib/librocdxg.so");
    let rocdxg_dids = rocm_relative_file_exists(&roots, "share/rocdxg/dids.conf");
    let ldconfig_text = ldconfig_cache();
    let ldconfig_librocdxg = ldconfig_text
        .as_deref()
        .is_some_and(|text| text.contains("librocdxg.so"));
    // Read through `host_path`, like every other file this probe looks for, so a
    // simulated host decides the answer rather than the real machine.
    let librocdxg_on_ld_library_path =
        librocdxg_on_loader_path(std::env::var("LD_LIBRARY_PATH").ok().as_deref(), |file| {
            host_path(file).exists()
        });
    // The roots are already re-rooted through `host_path`, so this existence check
    // takes the path as given.
    let librocdxg_beside_runtime = librocdxg_beside_runtime(&roots, Path::exists);
    let rocminfo = which("rocminfo");
    let cargo = which("cargo");
    let missing = missing_plumbing(
        dxg_device,
        dxcore,
        librocdxg || librocdxg_on_ld_library_path,
        librocdxg_loadable(
            ldconfig_librocdxg,
            librocdxg_on_ld_library_path,
            librocdxg_beside_runtime,
        ),
    );
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
        librocdxg_on_ld_library_path,
        librocdxg_beside_runtime,
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
        self.dxg_device
            && self.dxcore
            && self.librocdxg_present()
            && librocdxg_loadable(
                self.ldconfig_librocdxg,
                self.librocdxg_on_ld_library_path,
                self.librocdxg_beside_runtime,
            )
    }
}

/// The `driver_status` `rocm examine` reports when the WSL GPU path is usable.
///
/// One constant for the producer ([`wsl_driver_summary`]) and for the advice that
/// tells a refused `rocm serve` user which status to look for, so the two cannot
/// drift apart.
pub const WSL_ROCDXG_READY: &str = "wsl_rocdxg_ready";

pub(super) fn wsl_driver_summary(wsl: &WslSummary) -> DriverSummary {
    let ready = wsl.rocdxg_ready();
    let status = if ready {
        WSL_ROCDXG_READY
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
            librocdxg_on_ld_library_path: false,
            librocdxg_beside_runtime: false,
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
    fn the_loader_path_finds_the_library_in_any_listed_directory() {
        // The helper is not Linux-gated, so this runs on the Windows job too. The
        // variable is built with the platform's own separator, and the library is
        // matched by its parent directory rather than by a `/`-joined string.
        let on_disk = |file: &Path| {
            file.file_name() == Some(std::ffi::OsStr::new("librocdxg.so"))
                && file.parent() == Some(Path::new("/opt/rocm-10.1.0/lib"))
        };
        let joined = |dirs: &[&str]| {
            std::env::join_paths(dirs)
                .expect("directories hold no separator")
                .into_string()
                .expect("directories are UTF-8")
        };
        // Not the first entry: the search is over every directory, in the order
        // the loader reads them.
        assert!(librocdxg_on_loader_path(
            Some(&joined(&[
                "/usr/lib/wsl/lib",
                "/opt/rocm-10.1.0/lib",
                "/other"
            ])),
            on_disk
        ));
        assert!(!librocdxg_on_loader_path(
            Some(&joined(&["/usr/lib/wsl/lib", "/opt/rocm-9.0.0/lib"])),
            on_disk
        ));
    }

    #[test]
    fn an_unset_or_empty_loader_path_finds_nothing() {
        let everywhere = |_: &Path| true;
        assert!(!librocdxg_on_loader_path(None, everywhere));
        assert!(!librocdxg_on_loader_path(Some(""), everywhere));
        // An empty entry is the current directory to the loader, which says
        // nothing about where ROCm lives, so a variable of nothing but empty
        // entries must not count as a hit even when a file of that name would
        // exist relative to it. Built with the platform's separator.
        for entries in [&["", ""][..], &["", "", ""][..]] {
            let value = std::env::join_paths(entries)
                .expect("empty entries hold no separator")
                .into_string()
                .expect("empty entries are UTF-8");
            assert!(
                !librocdxg_on_loader_path(Some(&value), everywhere),
                "{value:?}"
            );
        }
    }

    /// An existence check over a fixed set of files, compared component-wise so a
    /// path built with the platform separator matches one written with `/`.
    fn exists_among(files: &[&str]) -> impl Fn(&Path) -> bool {
        let files: Vec<PathBuf> = files.iter().map(PathBuf::from).collect();
        move |path| {
            files
                .iter()
                .any(|file| file.components().eq(path.components()))
        }
    }

    #[test]
    fn the_library_beside_the_runtime_that_opens_it_counts() {
        // The ROCM-30386 layout: a TheRock tarball under a versioned root, the two
        // libraries side by side, neither registered anywhere.
        let roots = [PathBuf::from("/opt/rocm-10.1.0")];
        for runtime in ["libhsa-runtime64.so.1", "libhsa-runtime64.so"] {
            let runtime = format!("/opt/rocm-10.1.0/lib/{runtime}");
            assert!(
                librocdxg_beside_runtime(
                    &roots,
                    exists_among(&["/opt/rocm-10.1.0/lib/librocdxg.so", &runtime])
                ),
                "{runtime}"
            );
        }
    }

    #[test]
    fn the_library_alone_or_the_runtime_alone_does_not_count() {
        let roots = [PathBuf::from("/opt/rocm")];
        // The package layout: librocdxg in a `lib` directory with no runtime
        // beside it, reached only through the cache or LD_LIBRARY_PATH.
        assert!(!librocdxg_beside_runtime(
            &roots,
            exists_among(&["/opt/rocm/lib/librocdxg.so"])
        ));
        assert!(!librocdxg_beside_runtime(
            &roots,
            exists_among(&["/opt/rocm/lib/libhsa-runtime64.so.1"])
        ));
        assert!(!librocdxg_beside_runtime(&[], exists_among(&[])));
    }

    #[test]
    fn beside_means_the_same_directory_not_the_same_machine() {
        // The runtime in one install and the library in another: the runtime's
        // `$ORIGIN` is its own install, so the other one's library is not found.
        let roots = [
            PathBuf::from("/opt/rocm"),
            PathBuf::from("/opt/rocm-10.1.0"),
        ];
        assert!(!librocdxg_beside_runtime(
            &roots,
            exists_among(&[
                "/opt/rocm/lib/librocdxg.so",
                "/opt/rocm-10.1.0/lib/libhsa-runtime64.so.1",
            ])
        ));
        // A runtime elsewhere under the same root is not beside it either.
        assert!(!librocdxg_beside_runtime(
            &roots[..1],
            exists_among(&[
                "/opt/rocm/lib/librocdxg.so",
                "/opt/rocm/lib/llvm/lib/libhsa-runtime64.so.1",
            ])
        ));
    }

    #[test]
    fn the_library_is_loadable_from_any_one_source() {
        assert!(librocdxg_loadable(true, false, false));
        assert!(librocdxg_loadable(false, true, false));
        assert!(librocdxg_loadable(false, false, true));
        assert!(librocdxg_loadable(true, true, true));
        assert!(!librocdxg_loadable(false, false, false));
    }

    #[test]
    fn rocdxg_is_ready_for_an_unregistered_library_beside_the_runtime() {
        // The ROCM-30386 layout with no environment help at all: the tarball's
        // runtime opens librocdxg from its own directory.
        let mut summary = WslSummary {
            is_wsl: true,
            dxg_device: true,
            dxcore: true,
            librocdxg: true,
            rocdxg_dids: false,
            ldconfig_librocdxg: false,
            librocdxg_on_ld_library_path: false,
            librocdxg_beside_runtime: true,
            rocminfo: true,
            cargo: false,
            detail: None,
        };
        assert!(summary.rocdxg_ready());
        assert_eq!(wsl_driver_summary(&summary).status, "wsl_rocdxg_ready");
        // Without the sibling runtime nothing reaches the library.
        summary.librocdxg_beside_runtime = false;
        assert!(!summary.rocdxg_ready());
        assert_eq!(wsl_driver_summary(&summary).status, "wsl_rocdxg_missing");
    }

    #[test]
    fn a_rocdxg_only_on_the_loader_path_is_ready() {
        // A source build in a non-ROCm directory: no ROCm root holds it, so
        // `librocdxg` is false, and the cache does not list it. The variable alone
        // reaches it, which is what docs/wsl.md tells the user to arrange.
        let mut summary = WslSummary {
            is_wsl: true,
            dxg_device: true,
            dxcore: true,
            librocdxg: false,
            rocdxg_dids: false,
            ldconfig_librocdxg: false,
            librocdxg_on_ld_library_path: true,
            librocdxg_beside_runtime: false,
            rocminfo: true,
            cargo: false,
            detail: None,
        };
        assert!(summary.librocdxg_present());
        assert!(summary.rocdxg_ready());
        let driver = wsl_driver_summary(&summary);
        assert_eq!(driver.status, "wsl_rocdxg_ready");
        // Not ready without the variable: the library is then nowhere.
        summary.librocdxg_on_ld_library_path = false;
        assert!(!summary.librocdxg_present());
        assert!(!summary.rocdxg_ready());
    }

    #[test]
    fn a_rocdxg_on_the_loader_path_is_not_listed_as_missing() {
        // found via LD_LIBRARY_PATH only: loadable, so neither entry appears.
        assert!(missing_plumbing(true, true, true, true).is_empty());
        // Nowhere: both the file and the way to load it are named.
        assert_eq!(
            missing_plumbing(true, true, false, false),
            ["librocdxg.so", "ldconfig:librocdxg.so"]
        );
        // On disk under a root but reached by nothing: only the loader entry.
        assert_eq!(
            missing_plumbing(true, true, true, false),
            ["ldconfig:librocdxg.so"]
        );
        assert_eq!(
            missing_plumbing(false, false, true, true),
            ["/dev/dxg", "/usr/lib/wsl/lib/libdxcore.so"]
        );
    }

    #[test]
    fn rocdxg_is_ready_for_an_unregistered_library_on_the_loader_path() {
        // The ROCM-30386 layout: a versioned tarball install the linker cache has
        // never heard of, reachable through LD_LIBRARY_PATH.
        let mut summary = WslSummary {
            is_wsl: true,
            dxg_device: true,
            dxcore: true,
            librocdxg: true,
            rocdxg_dids: true,
            ldconfig_librocdxg: false,
            librocdxg_on_ld_library_path: true,
            librocdxg_beside_runtime: false,
            rocminfo: true,
            cargo: false,
            detail: None,
        };
        assert!(summary.rocdxg_ready());
        assert_eq!(wsl_driver_summary(&summary).status, "wsl_rocdxg_ready");
        // The same machine without the variable is still not ready: the loader
        // has no way to find the library.
        summary.librocdxg_on_ld_library_path = false;
        assert!(!summary.rocdxg_ready());
        assert_eq!(wsl_driver_summary(&summary).status, "wsl_rocdxg_missing");
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
            librocdxg_on_ld_library_path: false,
            librocdxg_beside_runtime: false,
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
