// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Host environment probes: the OS and distro, the CPU, the tracked environment
//! variables, the shared-memory filesystem, the container check and the kernel log.

use super::Examination;
use super::library_copies::libraries_on_ld_path;
use crate::probe_exec::{MEDIUM, SHORT, read_host_text, run, truncate_to_chars};
use crate::{runtime_is_linux, runtime_is_windows};

/// Environment variables that commonly steer (or break) ROCm/HIP runtime
/// behavior. Captured verbatim into `Examination::env`.
/// Cap on captured `PATH` / `LD_LIBRARY_PATH` length, to keep the Examination
/// JSON bounded. Generous on purpose: the stored value is read by the catalog
/// (e.g. the `fix-6-path` PATH check), so the cut must sit well past any
/// realistic ROCm/HIP bin entry to avoid false "not on PATH" diagnoses.
const ENV_VALUE_MAX_CHARS: usize = 16_000;

const TRACKED_ENV_VARS: &[&str] = &[
    "HSA_OVERRIDE_GFX_VERSION",
    // Legacy ROCm releases need this to find the GPU through DXG under WSL.
    "HSA_ENABLE_DXG_DETECTION",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "ROCM_PATH",
    "ROCM_HOME",
    "HIP_PATH",
    "HIP_PLATFORM",
    "PYTORCH_ROCM_ARCH",
    "HCC_AMDGPU_TARGET",
    "AMDGPU_TARGETS",
    "LD_LIBRARY_PATH",
    "PATH",
];

pub(super) fn probe_os(e: &mut Examination) {
    // The OS *version*, mirroring examine.py's `platform.version()`. This used
    // to hold `std::env::consts::OS`, which is the OS *name* and so merely
    // repeated `os_family`, telling a reader of the report nothing.
    //
    // Left empty on platforms the CLI does not support rather than guessed at:
    // an empty field reads as "not collected", while a wrong one would be
    // reasoned over by the diagnosis catalog.
    e.os_version = if runtime_is_linux() {
        run("uname", &["-v"], SHORT).1.trim().to_owned()
    } else if runtime_is_windows() {
        run("cmd", &["/C", "ver"], SHORT).1.trim().to_owned()
    } else {
        String::new()
    };
    if runtime_is_linux() {
        e.os_family = "linux".to_owned();
        e.kernel_release = run("uname", &["-r"], SHORT).1.trim().to_owned();
        e.kernel_cmdline = read_host_text("/proc/cmdline").trim().to_owned();
        record_distro(
            e,
            &crate::hardware_root::read_os_release(crate::hardware_root::host_path_for_str),
        );
        if let Some(param) = parse_iommu_param(&e.kernel_cmdline) {
            e.iommu_kernel_param = param;
        }
        // One shared answer. This used to be its own predicate, and it differed
        // from the install summary's — so `rocm examine` and `rocm examine
        // --json` could disagree about the platform they were describing.
        e.is_wsl = crate::is_wsl_host();
    } else if runtime_is_windows() {
        e.os_family = "windows".to_owned();
    } else {
        e.os_family = "other".to_owned();
    }
}

/// Fill the distro fields from `read`, through the same parser the driver
/// plan uses, so the two report the same distro.
///
/// An unreadable line leaves both fields empty and says why in
/// `probe_failures`, naming the file actually read and the line — an empty
/// distro with no explanation would read as "nothing to report" rather than
/// "something to fix". A missing file reads as empty text, which is not a
/// failure; a file that exists but could not be read is, and is reported as
/// itself rather than as an unreadable line it never reached.
fn record_distro(e: &mut Examination, read: &crate::hardware_root::OsRelease) {
    if let crate::hardware_root::OsRelease::Unreadable { path, error } = read {
        e.probe_failures.push(format!(
            "{}; the distro is not reported.",
            crate::hardware_root::unreadable_prefix(path, error)
        ));
        return;
    }
    match crate::os_release::parse(read.text()) {
        Ok(mut fields) => {
            e.distro_id = fields.remove("ID").unwrap_or_default();
            e.distro_id_like = fields.remove("ID_LIKE").unwrap_or_default();
            e.distro_version = fields.remove("VERSION_ID").unwrap_or_default();
        }
        Err(unreadable) => e.probe_failures.push(format!(
            "{} {unreadable}; the distro is not reported.",
            read.path()
        )),
    }
}

/// Extract the value of `iommu=<value>` from a kernel cmdline string.
pub(super) fn parse_iommu_param(cmdline: &str) -> Option<String> {
    cmdline.split_whitespace().find_map(|token| {
        token
            .strip_prefix("iommu=")
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    })
}

pub(super) fn probe_cpu_linux(e: &mut Examination) {
    let txt = read_host_text("/proc/cpuinfo");
    for line in txt.lines() {
        if (e.cpu_vendor == "unknown")
            && line.starts_with("vendor_id")
            && let Some((_, value)) = line.split_once(':')
        {
            let value = value.trim();
            e.cpu_vendor = if value.contains("AMD") {
                "amd".to_owned()
            } else if value.contains("Intel") {
                "intel".to_owned()
            } else {
                value.to_lowercase()
            };
        }
        if e.cpu_model.is_empty()
            && line.starts_with("model name")
            && let Some((_, value)) = line.split_once(':')
        {
            e.cpu_model = value.trim().to_owned();
        }
        if e.cpu_vendor != "unknown" && !e.cpu_model.is_empty() {
            break;
        }
    }
}

pub(super) fn probe_cpu_windows(e: &mut Examination) {
    let (rc, out, _) = run(
        "powershell",
        &[
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name",
        ],
        MEDIUM,
    );
    if rc == 0 && !out.trim().is_empty() {
        e.cpu_model = out
            .trim()
            .lines()
            .next()
            .unwrap_or_default()
            .trim()
            .to_owned();
        let lname = e.cpu_model.to_lowercase();
        e.cpu_vendor = if lname.contains("amd") {
            "amd".to_owned()
        } else if lname.contains("intel") {
            "intel".to_owned()
        } else {
            "unknown".to_owned()
        };
    } else {
        e.probe_failures
            .push("Get-CimInstance Win32_Processor failed; cannot identify CPU.".to_owned());
    }
}

pub(super) fn probe_env(e: &mut Examination) {
    for key in TRACKED_ENV_VARS {
        let Ok(value) = std::env::var(key) else {
            continue;
        };
        let value = if matches!(*key, "PATH" | "LD_LIBRARY_PATH") {
            truncate_to_chars(value, ENV_VALUE_MAX_CHARS)
        } else {
            value
        };
        e.env.insert((*key).to_owned(), value);
    }
    let ld = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    let hit = libraries_on_ld_path(&ld, "libamdhip64").into_iter().next();
    if let Some(hit) = hit {
        e.hip_libs_on_ld_path = Some(true);
        e.notes
            .push(format!("libamdhip64 visible via LD_LIBRARY_PATH: {hit}"));
    } else {
        e.hip_libs_on_ld_path = if ld.is_empty() { None } else { Some(false) };
    }
}

/// The shared-memory filesystem a serving workload uses.
const SHM_PATH: &str = "/dev/shm";

/// Measure `/dev/shm`.
///
/// A direct `statvfs` rather than the shared disk-space helper, and that is not
/// an oversight in the helper. `sysinfo` omits tmpfs, and `disk_space` guards
/// against the consequence by comparing device ids -- so asking it about
/// `/dev/shm` correctly returns "unknown" instead of confidently reporting the
/// root filesystem's free space. The guard is right; this needs the number it
/// declines to guess at.
pub(super) fn probe_shared_memory(e: &mut Examination) {
    probe_shared_memory_at(e, SHM_PATH);
}

/// The body of [`probe_shared_memory`], with the path as a parameter so the
/// unmeasurable branch is reachable from a test. A hard-coded `/dev/shm` cannot
/// be made to fail on a host that has one.
fn probe_shared_memory_at(e: &mut Examination, path: &str) {
    let Some((total, available)) = filesystem_size(path) else {
        // Recorded, not swallowed. The fields stay `None`, which the catalog
        // reads as "not measured" rather than "no shortage" -- but without a
        // trace here, an examination that could not measure is byte-identical
        // to one that measured a healthy machine, and nothing tells a reader
        // which they are looking at. `probe_failures` is the documented channel
        // for exactly this (see `Examination::probe`), and a non-empty one also
        // degrades `status` from `ok`, which is the signal a caller branches on.
        e.probe_failures.push(format!(
            "could not query {path}; shared-memory allowance unknown"
        ));
        return;
    };
    e.shm_total_bytes = Some(total);
    e.shm_available_bytes = Some(available);
}

/// Total and available bytes for the filesystem mounted at `path`.
///
/// `None` when the path does not exist or the call fails, which callers must
/// keep distinct from zero: a machine that could not be measured is not a
/// machine with no space.
#[cfg(target_os = "linux")]
#[allow(unsafe_code)] // statvfs FFI; the same pattern as the Win32 calls in process.rs
fn filesystem_size(path: &str) -> Option<(u64, u64)> {
    let c_path = std::ffi::CString::new(path).ok()?;
    // SAFETY: `c_path` is a valid NUL-terminated C string that outlives the
    // call, and `stats` is a correctly sized, writable `statvfs` this thread
    // owns. The call only reads the path and writes the struct.
    let stats = unsafe {
        let mut stats = std::mem::MaybeUninit::<libc::statvfs>::zeroed();
        if libc::statvfs(c_path.as_ptr(), stats.as_mut_ptr()) != 0 {
            return None;
        }
        stats.assume_init()
    };
    // `f_frsize` is the fragment size the block counts are expressed in.
    // `checked_mul` rather than a plain product: these are values the kernel
    // hands back, and a probe has no business panicking on a surprising one.
    let block = stats.f_frsize;
    Some((
        block.checked_mul(stats.f_blocks)?,
        block.checked_mul(stats.f_bavail)?,
    ))
}

#[cfg(not(target_os = "linux"))]
// Platform stub. The `cfg` sibling calls non-const code, so making only this
// arm `const` would give the two platforms different signatures and push
// `missing_const_for_fn` onto every caller in turn.
#[allow(clippy::missing_const_for_fn)]
fn filesystem_size(_path: &str) -> Option<(u64, u64)> {
    None
}

pub(super) fn probe_container(e: &mut Examination) {
    for (marker, kind) in [("/.dockerenv", "docker"), ("/run/.containerenv", "podman")] {
        if crate::host_path(marker).exists() {
            e.in_container = true;
            e.container_kind = kind.to_owned();
            return;
        }
    }
    let cg = read_host_text("/proc/1/cgroup");
    if !cg.is_empty()
        && ["docker", "containerd", "lxc", "kubepods", "podman"]
            .iter()
            .any(|x| cg.contains(x))
    {
        e.in_container = true;
        if e.container_kind.is_empty() {
            e.container_kind = "container".to_owned();
        }
    }
}

pub(super) fn probe_dmesg_amdgpu(e: &mut Examination) {
    let (rc, out, _) = run("journalctl", &["-k", "--no-pager", "-n", "400"], MEDIUM);
    let text = if rc == 0 && !out.is_empty() {
        out
    } else {
        let (rc2, out2, _) = run("dmesg", &[], SHORT);
        if rc2 == 0 { out2 } else { String::new() }
    };
    if text.is_empty() {
        return;
    }
    let interesting = [
        "page fault",
        "ras controller",
        "vm_fault",
        "amdgpu_device_init",
        "out_of_registers",
        "ring",
        "gpu reset",
        "psp",
        "hw_fault",
    ];
    let mut hits: Vec<String> = Vec::new();
    for line in text.lines() {
        if !line.contains("amdgpu") && !line.contains("amdkfd") {
            continue;
        }
        let lower = line.to_lowercase();
        if interesting.iter().any(|s| lower.contains(s)) {
            hits.push(line.trim().chars().take(300).collect());
        }
    }
    let start = hits.len().saturating_sub(15);
    e.dmesg_amdgpu_tail = hits.split_off(start);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Linux host with an AMD GPU, so `compute_status` turns on the probe
    /// failures alone.
    fn linux_host_with_a_gpu() -> Examination {
        Examination {
            os_family: "linux".to_owned(),
            has_amd_gpu: true,
            ..Examination::default()
        }
    }

    /// The message and the state it describes, asserted together: an
    /// unreadable os-release names its line in `probe_failures`, leaves the
    /// distro unreported, and makes the examination `degraded` — and a readable
    /// one reports the distro with no failure recorded and stays `ok`.
    #[test]
    fn an_unreadable_os_release_names_its_line_instead_of_a_distro() {
        let mut e = linux_host_with_a_gpu();
        record_distro(
            &mut e,
            &crate::hardware_root::OsRelease::found(
                "ID=ubuntu\nVERSION_ID=\"24.04\"\nexport ID=debian\n",
            ),
        );
        assert_eq!(
            e.probe_failures,
            vec![
                "/etc/os-release line 3 is not a plain assignment: \"export ID=debian\"; \
                 the distro is not reported."
                    .to_owned()
            ]
        );
        assert_eq!((e.distro_id.as_str(), e.distro_version.as_str()), ("", ""));
        assert_eq!(e.compute_status(), "degraded");

        let mut e = linux_host_with_a_gpu();
        record_distro(
            &mut e,
            &crate::hardware_root::OsRelease::found("ID='ubuntu'\nVERSION_ID='24.04' # LTS\n"),
        );
        assert!(e.probe_failures.is_empty(), "{:?}", e.probe_failures);
        assert_eq!(
            (e.distro_id.as_str(), e.distro_version.as_str()),
            ("ubuntu", "24.04")
        );
        assert_eq!(e.compute_status(), "ok");

        // No file at all is not a failure: there is nothing to fix.
        let mut e = Examination::default();
        record_distro(&mut e, &crate::hardware_root::OsRelease::Absent);
        assert!(e.probe_failures.is_empty(), "{:?}", e.probe_failures);
    }

    /// `ID_LIKE` is parsed into `distro_id_like` for
    /// `repo_pkg_family`/`openmpi::resolve_package_manager`'s fallback
    /// matching, same as `ID`/`VERSION_ID`, and round-trips through JSON the
    /// same way so `rocm remote doctor` gets it from a deserialized remote
    /// `examine --json` output too.
    #[test]
    fn id_like_is_parsed_into_distro_id_like() {
        let mut e = linux_host_with_a_gpu();
        record_distro(
            &mut e,
            &crate::hardware_root::OsRelease::found(
                "ID=amzn\nID_LIKE=\"fedora\"\nVERSION_ID=\"2023\"\n",
            ),
        );
        assert_eq!(e.distro_id_like, "fedora");
    }

    /// A file that exists but could not be read is reported as itself — not
    /// folded into an empty, unreported distro the way `Absent` is, and not
    /// pointing at the wrong candidate when the fallback was the one that
    /// failed.
    #[test]
    fn an_unreadable_file_names_itself_instead_of_a_distro() {
        let mut e = linux_host_with_a_gpu();
        record_distro(
            &mut e,
            &crate::hardware_root::OsRelease::Unreadable {
                path: "/usr/lib/os-release",
                error: std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied"),
            },
        );
        assert_eq!(
            e.probe_failures,
            vec![
                "/usr/lib/os-release could not be read: denied; the distro is not reported."
                    .to_owned()
            ]
        );
        assert_eq!((e.distro_id.as_str(), e.distro_version.as_str()), ("", ""));
        assert_eq!(e.compute_status(), "degraded");
    }

    /// The same malformed line as
    /// [`an_unreadable_os_release_names_its_line_instead_of_a_distro`], but
    /// read through the fallback path: `probe_failures` must name
    /// `/usr/lib/os-release`, not `/etc/os-release`. That test only exercises
    /// [`crate::hardware_root::OsRelease::found`], which hardcodes
    /// `/etc/os-release`, so a regression that hardcoded the path back into
    /// `record_distro`'s message would still pass it.
    #[test]
    fn a_fallback_reads_malformed_line_names_the_fallback_path_in_probe_failures() {
        let mut e = linux_host_with_a_gpu();
        record_distro(
            &mut e,
            &crate::hardware_root::OsRelease::Found {
                path: "/usr/lib/os-release",
                text: "ID=ubuntu\nVERSION_ID=\"24.04\"\nexport ID=debian\n".to_owned(),
            },
        );
        assert_eq!(
            e.probe_failures,
            vec![
                "/usr/lib/os-release line 3 is not a plain assignment: \"export ID=debian\"; \
                 the distro is not reported."
                    .to_owned()
            ]
        );
        assert_eq!((e.distro_id.as_str(), e.distro_version.as_str()), ("", ""));
        assert_eq!(e.compute_status(), "degraded");
    }

    #[test]
    fn an_unmeasurable_shared_memory_allowance_leaves_a_trace() {
        // The distinction the fields exist to preserve, tested at the layer that
        // can erase it. The catalog already refuses to report a shortage it
        // could not measure; this is the other half. Without a record here, an
        // examination that failed to measure is indistinguishable from one that
        // measured a healthy machine, and a reader cannot tell which.
        // A machine the CLI supports and has found a GPU on, so `compute_status`
        // reaches the probe-failure branch instead of short-circuiting on a
        // platform or hardware verdict first.
        let mut e = Examination {
            os_family: "linux".to_owned(),
            has_amd_gpu: true,
            ..Examination::default()
        };
        probe_shared_memory_at(&mut e, "/nonexistent-shm-for-this-test");

        assert_eq!(
            e.shm_total_bytes, None,
            "an unreadable path must not invent a measurement"
        );
        assert!(
            e.probe_failures.iter().any(|f| f.contains("shared-memory")),
            "the failure has to be recorded, not swallowed: {:?}",
            e.probe_failures
        );
        // `probe_failures` is what degrades the overall verdict, so the trace is
        // load-bearing rather than decorative.
        assert_eq!(
            e.compute_status(),
            "degraded",
            "a probe that could not run must not leave the machine looking clean"
        );
    }

    #[test]
    fn iommu_param_parsed_from_cmdline() {
        assert_eq!(
            parse_iommu_param("BOOT_IMAGE=/vmlinuz iommu=pt amd_iommu=on quiet"),
            Some("pt".to_owned())
        );
        assert_eq!(parse_iommu_param("BOOT_IMAGE=/vmlinuz quiet"), None);
    }

    #[test]
    fn os_version_carries_a_version_not_the_os_name() {
        // This field used to hold `std::env::consts::OS`, so it repeated
        // `os_family` and told a reader of the report nothing.
        let mut e = Examination::default();
        probe_os(&mut e);

        // Holds on every platform: a real version on the supported ones, and
        // empty on the rest -- neither of which is the OS name.
        assert_ne!(
            e.os_version,
            std::env::consts::OS,
            "os_version must not merely repeat the OS name"
        );

        if runtime_is_linux() || runtime_is_windows() {
            assert!(
                !e.os_version.is_empty(),
                "a supported host must report an OS version"
            );
            assert_ne!(e.os_version, e.os_family);
        } else {
            // Blank by choice on a host the CLI does not support, rather than
            // a guess the diagnosis catalog would then reason over.
            assert!(
                e.os_version.is_empty(),
                "unsupported hosts report no version, got {:?}",
                e.os_version
            );
        }
    }

    /// Both `/etc/os-release` reads `rocm examine` makes — the distro fields
    /// and the distro name — answer from a simulated host root, and through the
    /// shared parser: a planted file naming a distro no runner has reads as
    /// that distro, as `sh` reads it. A read that bypassed the root would
    /// report the runner's distro. One that bypassed the parser would read the
    /// file as the old per-reader parsers did: the distro fields with their
    /// single quotes kept, and the name from the first of two `PRETTY_NAME`s
    /// rather than the last, which is the one `sh` assigns.
    #[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
    #[test]
    fn os_release_reads_follow_the_simulated_host_root() {
        use crate::hardware_root::TEST_HOST_ROOT_ENV;
        use crate::test_env::RestoredEnvVar;

        let root = std::env::temp_dir().join(format!(
            "rocm-core-os-release-root-{}",
            crate::test_support::unique_suffix()
        ));
        std::fs::create_dir_all(root.join("etc")).expect("create planted /etc");
        std::fs::write(
            root.join("etc/os-release"),
            "ID='planted'\nVERSION_ID='0.1'\nPRETTY_NAME='Stale 0.0'\nPRETTY_NAME='Planted 0.1'\n",
        )
        .expect("write planted os-release");

        let (examined, name) = {
            let _guard = crate::hardware_root::HOST_ROOT_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let _root = RestoredEnvVar::set(TEST_HOST_ROOT_ENV, &root);
            let mut examined = Examination::default();
            probe_os(&mut examined);
            (examined, crate::host_gpu::detect_distro_name())
        };
        std::fs::remove_dir_all(&root).ok();

        assert_eq!(
            (
                examined.distro_id.as_str(),
                examined.distro_version.as_str()
            ),
            ("planted", "0.1")
        );
        assert_eq!(name.as_deref(), Some("Planted 0.1"));
    }

    /// `os-release(5)` falls back to `/usr/lib/os-release` when `/etc/os-release`
    /// is absent. Both `rocm examine` reads must still answer from a host that
    /// ships only the fallback path — a probe that stopped after the first
    /// candidate would read as an unknown distro instead.
    #[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
    #[test]
    fn os_release_reads_fall_back_to_usr_lib_when_etc_is_absent() {
        use crate::hardware_root::TEST_HOST_ROOT_ENV;
        use crate::test_env::RestoredEnvVar;

        let root = std::env::temp_dir().join(format!(
            "rocm-core-os-release-fallback-root-{}",
            crate::test_support::unique_suffix()
        ));
        std::fs::create_dir_all(root.join("usr/lib")).expect("create planted /usr/lib");
        std::fs::write(
            root.join("usr/lib/os-release"),
            "ID='fallback'\nVERSION_ID='1.2'\nPRETTY_NAME='Fallback 1.2'\n",
        )
        .expect("write planted os-release");

        let (examined, name) = {
            let _guard = crate::hardware_root::HOST_ROOT_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let _root = RestoredEnvVar::set(TEST_HOST_ROOT_ENV, &root);
            let mut examined = Examination::default();
            probe_os(&mut examined);
            (examined, crate::host_gpu::detect_distro_name())
        };
        std::fs::remove_dir_all(&root).ok();

        assert_eq!(
            (
                examined.distro_id.as_str(),
                examined.distro_version.as_str()
            ),
            ("fallback", "1.2")
        );
        assert_eq!(name.as_deref(), Some("Fallback 1.2"));
    }
}
