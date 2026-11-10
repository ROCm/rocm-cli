// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Host environment probes: the OS and distro, the CPU, the tracked environment
//! variables, the shared-memory filesystem, the container check and the kernel log.

use super::Examination;
use super::library_copies::libraries_on_ld_path;
use crate::host_gpu::{
    CpuFacts, read_linux_cpu_facts, run_windows_system_query, windows_cpu_facts_with,
};
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
        // The summary's read, so `kernel:` and `kernel_release` come from one
        // `uname -r`. A failed read is empty here, with no `probe_failures`
        // entry, as before: a host without a usable `uname` is not a degraded one.
        e.kernel_release = crate::host_gpu::detect_kernel_version().unwrap_or_default();
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

/// Fill the CPU fields from the shared reader's answer. A CPU that could not be
/// read leaves the model empty and the vendor `unknown`, which is how the JSON
/// has always said "not collected".
pub(crate) fn record_cpu(e: &mut Examination, facts: CpuFacts) {
    e.cpu_model = facts.model.unwrap_or_default();
    e.cpu_vendor = facts.vendor;
}

/// The CPU on Linux, from `/proc/cpuinfo` through the reader the host summary
/// uses. A cpuinfo with no CPU in it is not a probe failure.
pub(super) fn probe_cpu_linux(e: &mut Examination) {
    record_cpu(e, read_linux_cpu_facts());
}

/// The CPU on Windows, from the CPU row of the system inventory through the
/// reader the host summary uses.
pub(super) fn probe_cpu_windows(e: &mut Examination) {
    probe_cpu_windows_with(e, run_windows_system_query);
}

/// [`probe_cpu_windows`] with the query injected, so a test on any host drives
/// the examination's whole Windows read, the failure entry included. `query` is
/// handed the inventory's script and time limit; see `windows_cpu_facts_with`.
/// [`probe_cpu_windows`] passes the real PowerShell runner, which is the one
/// argument no test on a Linux host can reach.
pub(crate) fn probe_cpu_windows_with(
    e: &mut Examination,
    query: impl FnOnce(&str, std::time::Duration) -> Option<String>,
) {
    record_cpu_windows(e, windows_cpu_facts_with(query));
}

/// The `probe_failures` entry for a Windows read that found no CPU. It names the
/// query that ran, the system inventory's CPU and RAM script, and fires whether
/// that query failed, timed out, printed nothing or printed no CPU row.
pub(crate) const WINDOWS_CPU_NOT_FOUND: &str =
    "The Windows system inventory query (CPU and RAM) reported no CPU; cannot identify CPU.";

/// [`record_cpu`], and a failure entry when no CPU could be read: on Windows the
/// CPU comes from a query that can fail or time out, and an empty model alone
/// would not say so. Crate-visible so a test on any host can feed it the answer
/// of an injected query.
pub(crate) fn record_cpu_windows(e: &mut Examination, facts: CpuFacts) {
    if facts.model.is_none() {
        e.probe_failures.push(WINDOWS_CPU_NOT_FOUND.to_owned());
    }
    record_cpu(e, facts);
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

    /// Plant `files` (path under the root, body) in a fresh directory and run
    /// `read` with it as the simulated host root.
    #[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
    fn with_planted_host_root<T>(files: &[(&str, &str)], read: impl FnOnce() -> T) -> T {
        use crate::hardware_root::TEST_HOST_ROOT_ENV;
        use crate::test_env::RestoredEnvVar;

        let root = std::env::temp_dir().join(format!(
            "rocm-core-cpu-root-{}",
            crate::test_support::unique_suffix()
        ));
        for (relative, body) in files {
            let path = root.join(relative.trim_start_matches('/'));
            std::fs::create_dir_all(path.parent().expect("planted path has a parent"))
                .expect("create planted directory");
            std::fs::write(path, body).expect("write planted file");
        }
        let result = {
            let _guard = crate::hardware_root::HOST_ROOT_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let _root = RestoredEnvVar::set(TEST_HOST_ROOT_ENV, &root);
            read()
        };
        std::fs::remove_dir_all(&root).ok();
        result
    }

    /// CPU-model pair of docs/host-probing.md, native Linux. Both reads take
    /// `/proc/cpuinfo` from the simulated host root, so a planted CPU no runner
    /// has is what each reports, and both report the model with its whitespace
    /// collapsed (the examination used to keep the runs of spaces).
    #[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
    #[test]
    fn cpu_reads_follow_the_simulated_host_root() {
        let (summary, examined) = with_planted_host_root(
            &[(
                "proc/cpuinfo",
                "processor\t: 0\nvendor_id\t: AuthenticAMD\nmodel name\t: Planted  CPU 9000   Processor\n",
            )],
            || {
                let mut examined = Examination::default();
                probe_cpu_linux(&mut examined);
                (crate::host_gpu::detect_cpu_model(), examined)
            },
        );
        assert_eq!(summary.as_deref(), Some("Planted CPU 9000 Processor"));
        assert_eq!(examined.cpu_model, "Planted CPU 9000 Processor");
        assert_eq!(examined.cpu_vendor, "amd");
    }

    /// CPU-model pair, WSL: the summary reads the CPU and the examination does
    /// not. `probe_with_interpreter` returns from its WSL branch before
    /// `probe_cpu_linux`, so on every WSL host `cpu_model` is empty and
    /// `cpu_vendor` is `unknown` while the summary shows the CPU. Pinned as an
    /// existing difference of the CPU-model pair, which leaves it as it is.
    #[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
    #[test]
    fn on_wsl_the_summary_reads_the_cpu_and_the_examination_does_not() {
        let (summary, examined) = with_planted_host_root(
            &[
                ("dev/dxg", ""),
                (
                    "proc/cpuinfo",
                    "processor\t: 0\nvendor_id\t: AuthenticAMD\nmodel name\t: Planted WSL CPU\n",
                ),
            ],
            || {
                (
                    crate::host_gpu::detect_cpu_model(),
                    Examination::probe(crate::examine::FrameworkProbe::Skip),
                )
            },
        );
        assert!(examined.is_wsl, "the planted /dev/dxg reads as WSL");
        assert_eq!(summary.as_deref(), Some("Planted WSL CPU"));
        assert_eq!(examined.cpu_model, "");
        assert_eq!(examined.cpu_vendor, "unknown");
    }

    /// `probe_cpu_windows` runs the system inventory's query and nothing else:
    /// off Windows there is no PowerShell for the real runner, so the read finds
    /// no CPU and the entry says which query that was, and the injected-query
    /// form given a CPU row fills the fields the same way. Replacing the runner
    /// `probe_cpu_windows` passes, or the failure entry, fails this test. On
    /// Windows the runner is live, so it is skipped there.
    #[cfg(not(windows))]
    #[test]
    fn probe_cpu_windows_runs_the_inventory_query_and_reports_no_cpu_without_one() {
        let mut examined = Examination::default();
        probe_cpu_windows(&mut examined);
        assert_eq!(
            (examined.cpu_model.as_str(), examined.cpu_vendor.as_str()),
            ("", "unknown")
        );
        assert_eq!(examined.probe_failures, [WINDOWS_CPU_NOT_FOUND]);

        let mut with_cpu = Examination::default();
        probe_cpu_windows_with(&mut with_cpu, |_, _| {
            Some("CPU\tIntel(R) Core(TM) i5-12400\nRAM\t17179869184\n".to_owned())
        });
        assert_eq!(
            (with_cpu.cpu_model.as_str(), with_cpu.cpu_vendor.as_str()),
            ("Intel(R) Core(TM) i5-12400", "intel")
        );
        assert!(with_cpu.probe_failures.is_empty());
    }

    // The kernel-release read, the second pair of `docs/host-probing.md`
    // ("Kernel version"). It used to be two readers: `detect_kernel_version` in
    // the summary stack (1.5 s, strict UTF-8, `None` on any failure or non-zero
    // exit) and `probe_os` here (5 s, lossy, exit code ignored, empty on
    // failure). `probe_os` now calls `detect_kernel_version`, so these tests pin
    // the one reader's behaviour through both entry points: a release is read
    // as printed, a non-zero exit, output that is not UTF-8, a silent `uname`
    // and one that never answers read as nothing, an answer within the 5 s
    // limit is read, and a failed read leaves no `probe_failures` entry. Each
    // case plants a fake `uname` first on `PATH` and asks both entry points in
    // a child copy of this test binary, because the lookup of `uname` follows
    // the process's own `PATH`, which a test thread cannot change without
    // disturbing every other test. The child's entry point,
    // [`kernel_pair_child_reports_both_answers`], does nothing in a normal run.
    // Linux only, because `probe_os` reads `uname -r` only when
    // `runtime_is_linux()`, and the fake is a shell script; other Unix hosts are
    // not supported platforms.

    #[cfg(target_os = "linux")]
    mod kernel_pair {
        use super::*;
        use std::process::Command;
        use std::time::Instant;

        const CHILD_ENV: &str = "ROCM_CORE_TEST_KERNEL_PAIR";
        const ANSWER_PREFIX: &str = "kernel-pair-answers:";

        /// What each stack read for `uname -r`.
        #[derive(Debug, PartialEq, Eq)]
        struct Pair {
            /// `detect_kernel_version`, the summary's `kernel`.
            summary: Option<String>,
            /// `probe_os`'s `kernel_release`.
            examination: String,
        }

        fn pair(summary: Option<&str>, examination: &str) -> Pair {
            Pair {
                summary: summary.map(str::to_owned),
                examination: examination.to_owned(),
            }
        }

        /// The child's side of [`ask`]: print both answers as JSON.
        #[test]
        fn kernel_pair_child_reports_both_answers() {
            let Some(mode) = std::env::var_os(CHILD_ENV) else {
                return;
            };
            if mode == "inspect" {
                // One inspection through the entry point `rocm examine --json` uses.
                let (e, summary) =
                    Examination::probe_with_summary(crate::FrameworkProbe::Skip, None)
                        .expect("probe and gather");
                let answers = serde_json::json!({
                    "summary": summary.kernel,
                    "examination": e.kernel_release,
                    "probe_failures": e.probe_failures,
                });
                println!("{ANSWER_PREFIX}{answers}");
                return;
            }
            let mut e = Examination::default();
            probe_os(&mut e);
            let answers = serde_json::json!({
                "summary": crate::host_gpu::detect_kernel_version(),
                "examination": e.kernel_release,
                "probe_failures": e.probe_failures,
            });
            println!("{ANSWER_PREFIX}{answers}");
        }

        /// Plant a `uname` whose `-r` output is `body` (a shell fragment) and ask
        /// both stacks. `uname -v` answers a fixed string, so `probe_os` has
        /// nothing else to wait for.
        fn ask(body: &str) -> Pair {
            let (json, _release_reads) = run_child(body, "pair");
            // A failed read is empty, not a failure: `probe_os` records no
            // `probe_failures` entry at all in any case (whatever its wording
            // would be), so a host with a missing or broken `uname` does not read
            // as degraded. The fake host's os-release reads cleanly, so the list
            // is empty rather than compared with a baseline.
            assert_eq!(json["probe_failures"], serde_json::json!([]), "{json}");
            Pair {
                summary: json["summary"].as_str().map(str::to_owned),
                examination: json["examination"]
                    .as_str()
                    .expect("the examination answer is a string")
                    .to_owned(),
            }
        }

        /// Run the child in `mode` against a fake `uname` whose `-r` output is
        /// `body`, and return its answers and how many times `-r` was asked.
        /// `uname -v` answers a fixed string, so `probe_os` has nothing else to
        /// wait for.
        fn run_child(body: &str, mode: &str) -> (serde_json::Value, usize) {
            let scratch = tempfile::tempdir().expect("scratch directory");
            let bin = scratch.path().join("bin");
            std::fs::create_dir_all(&bin).expect("create the fake bin directory");
            let calls = scratch.path().join("release-reads");
            crate::fake_executable::write_fake_executable(
                &bin.join("uname"),
                format!(
                    "#!/bin/sh\nif [ \"$1\" = -v ]; then echo '#1 SMP fake'; exit 0; fi\necho x >> '{}'\n{body}\n",
                    calls.display()
                ),
            )
            .expect("plant the fake uname");
            let this_test = module_path!()
                .split_once("::")
                .expect("module path has a crate prefix")
                .1;
            let output = Command::new(std::env::current_exe().expect("path of the test binary"))
                .args([
                    "--exact",
                    &format!("{this_test}::kernel_pair_child_reports_both_answers"),
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD_ENV, mode)
                // The fake first; the system directories so `sleep` and `printf`
                // resolve inside it.
                .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
                .output()
                .expect("start the test binary again");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let line = stdout
                .lines()
                .find_map(|line| line.split_once(ANSWER_PREFIX).map(|(_, json)| json))
                .unwrap_or_else(|| {
                    panic!(
                        "the child printed no answers (status {}):\n{stdout}\n{}",
                        output.status,
                        String::from_utf8_lossy(&output.stderr)
                    )
                });
            let json: serde_json::Value = serde_json::from_str(line).expect("answers are JSON");
            let reads = std::fs::read_to_string(&calls).map_or(0, |log| log.lines().count());
            (json, reads)
        }

        /// One inspection reads the release once, for the examination and the
        /// summary together, so the two fields agree and a hung `uname` costs
        /// one timeout. The reads are counted by the fake `uname` itself, so a
        /// second read shows even when both return the same string. A failed
        /// read is shared too: both are empty and it is still one read.
        #[test]
        fn one_inspection_reads_the_kernel_release_once() {
            let (json, reads) = run_child("echo 6.8.0-fake", "inspect");
            assert_eq!(json["summary"], "6.8.0-fake", "{json}");
            assert_eq!(json["examination"], "6.8.0-fake", "{json}");
            assert_eq!(reads, 1, "{json}");

            let (json, reads) = run_child("echo 6.8.0-partial; exit 3", "inspect");
            assert_eq!(json["summary"], serde_json::Value::Null, "{json}");
            assert_eq!(json["examination"], "", "{json}");
            assert_eq!(reads, 1, "{json}");
        }

        #[test]
        fn both_read_the_release_a_working_uname_prints() {
            assert_eq!(
                ask("echo 6.8.0-fake"),
                pair(Some("6.8.0-fake"), "6.8.0-fake")
            );
        }

        /// A WSL 1 and a WSL 2 style release are read as printed by both, so
        /// `is_wsl1_kernel` (which reads `kernel_release`) sees what the summary
        /// shows. The consumer's answer is pinned in `wsl.rs`.
        #[test]
        fn both_read_wsl_style_releases_as_printed() {
            for release in [
                "4.4.0-19041-Microsoft",
                "5.15.167.4-microsoft-standard-WSL2",
            ] {
                assert_eq!(
                    ask(&format!("echo {release}")),
                    pair(Some(release), release),
                    "{release}"
                );
            }
        }

        /// A `uname` that prints a release and exits non-zero is dropped by
        /// both. The examination used to keep the release.
        #[test]
        fn a_non_zero_exit_with_output_is_dropped_by_both() {
            assert_eq!(ask("echo 6.8.0-partial; exit 3"), pair(None, ""));
        }

        /// Both read a successful but silent `uname` as nothing: `None` in the
        /// summary (shown as `<unknown>`), empty in the examination, with no
        /// `probe_failures` entry in either case.
        #[test]
        fn a_silent_uname_reads_as_nothing_in_both() {
            assert_eq!(ask("exit 0"), pair(None, ""));
        }

        /// Output that is not UTF-8 is dropped by both. The examination used to
        /// keep it with the bad byte replaced.
        #[test]
        fn output_that_is_not_utf8_is_dropped_by_both() {
            assert_eq!(ask("printf '6.8.0-\\377\\n'"), pair(None, ""));
        }

        /// A `uname` that answers after 2 s is read by both: the one timeout is
        /// the longer of the two old ones, 5 s. The summary used to give up at
        /// 1.5 s, so 2 s is past that old limit and leaves 3 s of margin to the
        /// new one on a loaded runner.
        #[test]
        fn an_answer_after_two_seconds_is_read_by_both() {
            let started = Instant::now();
            assert_eq!(
                ask("sleep 2; echo 6.8.0-slow"),
                pair(Some("6.8.0-slow"), "6.8.0-slow")
            );
            assert!(started.elapsed() >= std::time::Duration::from_secs(2));
        }

        /// A `uname` that never answers is read as nothing by both, each after
        /// the one timeout. `exec` leaves no grandchild holding the pipe.
        ///
        /// The stand-in would exit successfully with nothing after 30 s, which is
        /// also what a read with no timeout would turn into `None`, so the
        /// answers alone cannot tell a timeout from no timeout. The elapsed time
        /// can: the child reads twice in a row, each bounded by the 5 s limit, so
        /// it takes about 10 s, and at least 9 s shows the stand-in really hung.
        /// With the limit raised past the stand-in's 30 s it takes 60 s and the
        /// upper bound fails.
        #[test]
        fn a_uname_that_never_answers_reads_as_nothing_in_both_within_the_timeout() {
            let started = Instant::now();
            assert_eq!(ask("exec sleep 30"), pair(None, ""));
            let elapsed = started.elapsed();
            assert!(
                elapsed >= std::time::Duration::from_secs(9),
                "two reads of a hung uname returned after {elapsed:?}"
            );
            assert!(
                elapsed < std::time::Duration::from_secs(25),
                "the reads outlived the timeout: {elapsed:?}"
            );
        }
    }
}
