// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Linux driver access: the kernel modules, the device nodes and who may open
//! them, and Secure Boot.

use super::{Device, Examination};
use crate::probe_exec::{SHORT, read_host_text, read_text, run, which};
use std::time::Duration;

/// Whether the kernel's kfd is present on this host, however it got there.
///
/// On current kernels kfd is not a module of its own: `HSA_AMD` is a `bool`
/// Kconfig option and amdkfd is linked into `amdgpu`, so no `amdkfd` line
/// appears in the module list, built in or not. (Before 4.20 it was a separate
/// `amdkfd` module, which the module-list check still covers.) The kernel's own
/// interfaces are the signal: the `/sys/class/kfd/kfd` class device, or the
/// `/dev/kfd` node.
fn kfd_present_on_host() -> bool {
    crate::host_path("/sys/class/kfd/kfd").exists() || crate::host_path("/dev/kfd").exists()
}

/// Whether the amdgpu driver is present on this host, loaded or built in.
///
/// A built-in driver is absent from the module list. `/sys/module/amdgpu` is
/// no answer: the kernel creates it for a built-in driver at an early initcall,
/// before `amdgpu_init` can bail out (`nomodeset`, `amdgpu.modeset=0`). The PCI
/// driver directory exists only once `amdgpu_init` registered the driver.
fn amdgpu_present_on_host() -> bool {
    crate::host_path("/sys/bus/pci/drivers/amdgpu").exists()
}

/// What the host says about `amdgpu_loaded`: the amdgpu PCI driver is
/// registered, or kfd is present. kfd is linked into amdgpu, so kfd on the
/// host proves amdgpu too: a container that passes `/dev/kfd` through without
/// `/sys/bus/pci` must not report amdkfd up and amdgpu down.
fn amdgpu_implied_by_host() -> bool {
    amdgpu_present_on_host() || kfd_present_on_host()
}

/// Whether `name` is loaded or built in: listed in the module list, or present
/// on the host.
///
/// `modules` is `None` when the module list could not be read; the host
/// answering `true` still answers `true` then, and nothing else is known. The
/// host probe runs only when the module list has not already answered.
fn module_or_host_state(
    modules: Option<&[&str]>,
    name: &str,
    on_host: impl FnOnce() -> bool,
) -> Option<bool> {
    match modules {
        Some(modules) => Some(modules.contains(&name) || on_host()),
        None => on_host().then_some(true),
    }
}

/// `amdgpu_loaded` and `amdkfd_loaded`, in that order, from the module list
/// and the host. kfd on the host answers both; the amdgpu PCI driver answers
/// only `amdgpu_loaded`.
fn driver_states(modules: Option<&[&str]>) -> (Option<bool>, Option<bool>) {
    (
        module_or_host_state(modules, "amdgpu", amdgpu_implied_by_host),
        module_or_host_state(modules, "amdkfd", kfd_present_on_host),
    )
}

/// The kernel's module list, one module per line with its name first: `lsmod`
/// without its header, else `/proc/modules`. `None` when neither can be read.
fn read_module_list() -> Option<String> {
    let (rc, out, _) = run("lsmod", &[], SHORT);
    if rc == 0 {
        Some(out.lines().skip(1).collect::<Vec<_>>().join("\n"))
    } else {
        let txt = read_host_text("/proc/modules");
        if txt.is_empty() { None } else { Some(txt) }
    }
}

pub(super) fn probe_modules(e: &mut Examination) {
    probe_modules_from(e, read_module_list().as_deref());
}

/// `probe_modules` given the module list, so a test can hand one in instead
/// of reading the machine's own.
fn probe_modules_from(e: &mut Examination, module_text: Option<&str>) {
    let modules: Option<Vec<&str>> = module_text.map(|text| {
        text.lines()
            .filter_map(|line| line.split_whitespace().next())
            .collect()
    });
    (e.amdgpu_loaded, e.amdkfd_loaded) = driver_states(modules.as_deref());

    for dir in ["/etc/modprobe.d", "/usr/lib/modprobe.d", "/run/modprobe.d"] {
        let Ok(entries) = std::fs::read_dir(crate::host_path(dir)) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("conf") {
                continue;
            }
            let body = read_text(&path.to_string_lossy());
            if body.lines().any(line_blacklists_amdgpu) {
                // The logical path, not where the probe read it.
                e.amdgpu_blacklisted_in
                    .push(format!("{dir}/{}", entry.file_name().to_string_lossy()));
            }
        }
    }
}

/// Matches `^\s*blacklist\s+amdgpu\b`.
pub(super) fn line_blacklists_amdgpu(line: &str) -> bool {
    let rest = line.trim_start();
    let Some(rest) = rest.strip_prefix("blacklist") else {
        return false;
    };
    let rest = rest.trim_start();
    rest == "amdgpu"
        || rest.strip_prefix("amdgpu").is_some_and(|tail| {
            tail.is_empty() || !tail.starts_with(|c: char| c.is_alphanumeric() || c == '_')
        })
}

pub(super) fn probe_devices(e: &mut Examination) {
    e.kfd = Some(stat_device("/dev/kfd", &e.user_name, &e.user_groups));
    if let Ok(entries) = std::fs::read_dir(crate::host_path("/dev/dri")) {
        let mut render: Vec<String> = entries
            .flatten()
            .filter_map(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                name.starts_with("renderD")
                    .then(|| format!("/dev/dri/{name}"))
            })
            .collect();
        render.sort();
        for path in render {
            e.render_devices
                .push(stat_device(&path, &e.user_name, &e.user_groups));
        }
    }
}

/// Stat the device node at the logical host path `path` (`/dev/kfd`), reading
/// it where [`crate::host_path`] puts it but reporting the logical path.
pub(super) fn stat_device(path: &str, user_name: &str, user_groups: &[String]) -> Device {
    let host = crate::host_path(path);
    let mut device = Device {
        path: path.to_owned(),
        exists: host.exists(),
        ..Device::default()
    };
    if !device.exists {
        return device;
    }
    let (rc, out, _) = run("stat", &["-c", "%A|%U|%G", &host.to_string_lossy()], SHORT);
    if rc == 0 {
        let fields: Vec<&str> = out.trim().split('|').collect();
        if fields.len() == 3 {
            device.mode = fields[0].to_owned();
            device.owner_user = fields[1].to_owned();
            device.owner_group = fields[2].to_owned();
            let (can_read, can_write) = mode_access(
                &device.mode,
                &device.owner_user,
                &device.owner_group,
                user_name,
                user_groups,
            );
            device.user_can_read = can_read;
            device.user_can_write = can_write;
        }
    }
    device
}

/// Derive read/write access from a `stat`-style mode string and group
/// membership, following POSIX precedence (owner, then group, then other).
pub(super) fn mode_access(
    mode: &str,
    owner_user: &str,
    owner_group: &str,
    user_name: &str,
    user_groups: &[String],
) -> (Option<bool>, Option<bool>) {
    let bytes = mode.as_bytes();
    if bytes.len() < 10 {
        return (None, None);
    }
    let class = if !user_name.is_empty() && user_name == owner_user {
        1
    } else if user_groups.iter().any(|g| g == owner_group) {
        4
    } else {
        7
    };
    let read = bytes[class] == b'r';
    let write = bytes[class + 1] == b'w';
    (Some(read), Some(write))
}

pub(super) fn probe_user(e: &mut Examination) {
    e.user_name = std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_default();
    let (rc, out, _) = run("id", &["-Gn"], Duration::from_secs(3));
    if rc == 0 {
        e.user_groups = out.split_whitespace().map(str::to_owned).collect();
    }
    e.in_render_group = Some(e.user_groups.iter().any(|g| g == "render"));
    e.in_video_group = Some(e.user_groups.iter().any(|g| g == "video"));
}

pub(super) fn probe_secure_boot(e: &mut Examination) {
    if !which("mokutil") {
        return;
    }
    let (rc, out, _) = run("mokutil", &["--sb-state"], Duration::from_secs(3));
    if rc == 0 {
        let o = out.to_lowercase();
        if o.contains("enabled") {
            e.secure_boot = "enabled".to_owned();
        } else if o.contains("disabled") {
            e.secure_boot = "disabled".to_owned();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blacklist_amdgpu_detection() {
        assert!(line_blacklists_amdgpu("blacklist amdgpu"));
        assert!(line_blacklists_amdgpu("  blacklist   amdgpu"));
        assert!(!line_blacklists_amdgpu("blacklist amdgpufoo"));
        assert!(!line_blacklists_amdgpu("# blacklist amdgpu"));
    }

    #[test]
    fn mode_access_owner_group_other_precedence() {
        // crw-rw---- root render: a render member can write, others cannot.
        let groups = vec!["render".to_owned()];
        let (r, w) = mode_access("crw-rw----", "root", "render", "alice", &groups);
        assert_eq!((r, w), (Some(true), Some(true)));
        let (r, w) = mode_access("crw-rw----", "root", "render", "alice", &[]);
        assert_eq!((r, w), (Some(false), Some(false)));
    }

    #[test]
    fn a_module_is_present_when_listed() {
        assert_eq!(
            module_or_host_state(Some(&["amdgpu", "amdkfd"]), "amdkfd", || false),
            Some(true)
        );
        assert_eq!(
            module_or_host_state(Some(&["amdgpu", "drm"]), "amdgpu", || false),
            Some(true)
        );
    }

    /// Built into the kernel, a driver never appears in the module list.
    #[test]
    fn a_module_is_present_when_built_in_and_absent_from_the_module_list() {
        assert_eq!(
            module_or_host_state(Some(&["drm"]), "amdkfd", || true),
            Some(true)
        );
        assert_eq!(
            module_or_host_state(Some(&[]), "amdgpu", || true),
            Some(true)
        );
    }

    #[test]
    fn a_module_is_absent_with_neither_a_listing_nor_the_host() {
        assert_eq!(
            module_or_host_state(Some(&["drm"]), "amdkfd", || false),
            Some(false)
        );
        assert_eq!(
            module_or_host_state(Some(&[]), "amdgpu", || false),
            Some(false)
        );
    }

    /// With no module list, the host still answers; without it the field
    /// stays unknown rather than claiming `false`.
    #[test]
    fn an_unreadable_module_list_follows_the_host() {
        assert_eq!(module_or_host_state(None, "amdgpu", || true), Some(true));
        assert_eq!(module_or_host_state(None, "amdgpu", || false), None);
    }

    /// The host probe is skipped once the module list has answered.
    #[test]
    fn the_host_probe_is_skipped_when_the_module_list_already_answers() {
        assert_eq!(
            module_or_host_state(Some(&["amdgpu"]), "amdgpu", || unreachable!()),
            Some(true)
        );
    }

    /// The host side of the check, against a planted root.
    #[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
    #[test]
    fn amdgpu_present_on_host_reads_the_pci_driver_dir() {
        use crate::hardware_root::TEST_HOST_ROOT_ENV;
        use crate::test_env::RestoredEnvVar;

        let root = std::env::temp_dir().join(format!(
            "rocm-core-amdgpu-present-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        std::fs::create_dir_all(&root).expect("create root");
        let (neither, module_dir_only, planted) = {
            let _guard = crate::hardware_root::HOST_ROOT_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let _root = RestoredEnvVar::set(TEST_HOST_ROOT_ENV, &root);
            let neither = amdgpu_present_on_host();
            // The module directory a built-in driver keeps even when it never
            // bound: not evidence.
            std::fs::create_dir_all(root.join("sys/module/amdgpu")).expect("plant sysfs");
            let module_dir_only = amdgpu_present_on_host();
            std::fs::create_dir_all(root.join("sys/bus/pci/drivers/amdgpu")).expect("plant sysfs");
            (neither, module_dir_only, amdgpu_present_on_host())
        };
        std::fs::remove_dir_all(&root).ok();
        assert!(!neither, "an empty root has no amdgpu");
        assert!(
            !module_dir_only,
            "/sys/module/amdgpu alone is not a bound driver"
        );
        assert!(planted, "/sys/bus/pci/drivers/amdgpu counts");
    }

    /// A module list, as `read_module_list` returns it, naming neither
    /// `amdgpu` nor `amdkfd`.
    #[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
    const NO_AMD_MODULES: &str = "snd_hda_intel 61440 3 - Live 0x0000000000000000\n\
                                  nvme 61440 2 - Live 0x0000000000000000";

    /// The module list alone, against an empty host root: on a machine where
    /// the drivers really are modules, each field must come from its own name
    /// in the list. Both named answers both fields; `amdgpu` alone answers
    /// only `amdgpu_loaded`, and `amdkfd` alone only `amdkfd_loaded`, so a
    /// misspelt or swapped name in `driver_states` fails here.
    #[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
    #[test]
    fn probe_modules_reports_each_driver_its_module_list_names() {
        use crate::hardware_root::TEST_HOST_ROOT_ENV;
        use crate::test_env::RestoredEnvVar;

        let root = tempfile::tempdir().expect("create host root");
        let probe = |module_text: &str| {
            let mut e = Examination::default();
            probe_modules_from(&mut e, Some(module_text));
            (e.amdgpu_loaded, e.amdkfd_loaded)
        };
        let (both, amdgpu_only, amdkfd_only) = {
            let _guard = crate::hardware_root::HOST_ROOT_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let _root = RestoredEnvVar::set(TEST_HOST_ROOT_ENV, root.path());
            (
                probe(
                    "amdgpu 19378176 12 - Live 0x0000000000000000\n\
                     amdkfd 380928 1 amdgpu, Live 0x0000000000000000\n\
                     nvme 61440 2 - Live 0x0000000000000000",
                ),
                probe(
                    "snd_hda_intel 61440 3 - Live 0x0000000000000000\n\
                     amdgpu 19378176 12 - Live 0x0000000000000000",
                ),
                probe("amdkfd 380928 0 - Live 0x0000000000000000"),
            )
        };
        assert_eq!(both, (Some(true), Some(true)), "both drivers listed");
        assert_eq!(amdgpu_only, (Some(true), Some(false)), "only amdgpu listed");
        assert_eq!(amdkfd_only, (Some(false), Some(true)), "only amdkfd listed");
    }

    /// The wiring, not just the helpers: `probe_modules_from` must consult the
    /// host root. The module list it is handed names neither driver, so
    /// `amdgpu_loaded` can only come from the planted
    /// `/sys/bus/pci/drivers/amdgpu`, and that directory must not make
    /// `amdkfd_loaded` true.
    #[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
    #[test]
    fn probe_modules_reports_amdgpu_when_it_is_on_the_host() {
        use crate::hardware_root::TEST_HOST_ROOT_ENV;
        use crate::test_env::RestoredEnvVar;

        let root = std::env::temp_dir().join(format!(
            "rocm-core-probe-amdgpu-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        std::fs::create_dir_all(root.join("sys/bus/pci/drivers/amdgpu")).expect("plant sysfs");
        let (amdgpu_loaded, amdkfd_loaded) = {
            let _guard = crate::hardware_root::HOST_ROOT_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let _root = RestoredEnvVar::set(TEST_HOST_ROOT_ENV, &root);
            let mut e = Examination::default();
            probe_modules_from(&mut e, Some(NO_AMD_MODULES));
            (e.amdgpu_loaded, e.amdkfd_loaded)
        };
        std::fs::remove_dir_all(&root).ok();
        assert_eq!(amdgpu_loaded, Some(true));
        assert_eq!(
            amdkfd_loaded,
            Some(false),
            "the amdgpu PCI driver does not imply kfd"
        );
    }

    /// The host side of the check, against a planted root: the sysfs class
    /// device alone, the device node alone, and neither.
    #[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
    #[test]
    fn kfd_present_on_host_reads_sysfs_and_the_device_node() {
        use crate::hardware_root::TEST_HOST_ROOT_ENV;
        use crate::test_env::RestoredEnvVar;

        let root = std::env::temp_dir().join(format!(
            "rocm-core-kfd-present-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        std::fs::create_dir_all(&root).expect("create root");
        let (neither, sysfs_only, dev_only) = {
            let _guard = crate::hardware_root::HOST_ROOT_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let _root = RestoredEnvVar::set(TEST_HOST_ROOT_ENV, &root);
            let neither = kfd_present_on_host();
            std::fs::create_dir_all(root.join("sys/class/kfd/kfd")).expect("plant sysfs");
            let sysfs_only = kfd_present_on_host();
            std::fs::remove_dir_all(root.join("sys")).expect("unplant sysfs");
            std::fs::create_dir_all(root.join("dev")).expect("plant dev");
            std::fs::write(root.join("dev/kfd"), "").expect("plant /dev/kfd");
            (neither, sysfs_only, kfd_present_on_host())
        };
        std::fs::remove_dir_all(&root).ok();
        assert!(!neither, "an empty root has no kfd");
        assert!(sysfs_only, "/sys/class/kfd/kfd alone counts");
        assert!(dev_only, "/dev/kfd alone counts");
    }

    /// kfd on the host alone answers `amdgpu_loaded`: no amdgpu PCI driver
    /// directory, only a kfd signal, either one.
    #[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
    #[test]
    fn kfd_alone_on_the_host_implies_amdgpu() {
        use crate::hardware_root::TEST_HOST_ROOT_ENV;
        use crate::test_env::RestoredEnvVar;

        let root = std::env::temp_dir().join(format!(
            "rocm-core-kfd-implies-amdgpu-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        std::fs::create_dir_all(&root).expect("create root");
        let (neither, sysfs_only, dev_only) = {
            let _guard = crate::hardware_root::HOST_ROOT_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let _root = RestoredEnvVar::set(TEST_HOST_ROOT_ENV, &root);
            let neither = amdgpu_implied_by_host();
            std::fs::create_dir_all(root.join("sys/class/kfd/kfd")).expect("plant sysfs");
            let sysfs_only = amdgpu_implied_by_host();
            std::fs::remove_dir_all(root.join("sys")).expect("unplant sysfs");
            std::fs::create_dir_all(root.join("dev")).expect("plant dev");
            std::fs::write(root.join("dev/kfd"), "").expect("plant /dev/kfd");
            (neither, sysfs_only, amdgpu_implied_by_host())
        };
        std::fs::remove_dir_all(&root).ok();
        assert!(!neither, "an empty root implies no amdgpu");
        assert!(sysfs_only, "/sys/class/kfd/kfd alone implies amdgpu");
        assert!(dev_only, "/dev/kfd alone implies amdgpu");
    }

    /// Which host signal answers which field. Each case plants one signal into
    /// an otherwise empty root and hands over the module list directly, so the
    /// answer cannot come from the machine the test runs on: kfd alone answers
    /// both fields, the amdgpu PCI driver alone answers only `amdgpu_loaded`,
    /// and with the module list unreadable the host still answers.
    #[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
    #[test]
    fn driver_states_answer_each_field_from_its_own_host_signal() {
        use crate::hardware_root::TEST_HOST_ROOT_ENV;
        use crate::test_env::RestoredEnvVar;

        let root = std::env::temp_dir().join(format!(
            "rocm-core-driver-states-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        std::fs::create_dir_all(&root).expect("create root");
        let (neither, unread_neither, kfd_only, unread_kfd_only, pci_only, unread_pci_only) = {
            let _guard = crate::hardware_root::HOST_ROOT_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let _root = RestoredEnvVar::set(TEST_HOST_ROOT_ENV, &root);
            let neither = driver_states(Some(&[]));
            let unread_neither = driver_states(None);
            std::fs::create_dir_all(root.join("dev")).expect("plant dev");
            std::fs::write(root.join("dev/kfd"), "").expect("plant /dev/kfd");
            let kfd_only = driver_states(Some(&[]));
            let unread_kfd_only = driver_states(None);
            std::fs::remove_dir_all(root.join("dev")).expect("unplant dev");
            std::fs::create_dir_all(root.join("sys/bus/pci/drivers/amdgpu")).expect("plant sysfs");
            (
                neither,
                unread_neither,
                kfd_only,
                unread_kfd_only,
                driver_states(Some(&[])),
                driver_states(None),
            )
        };
        std::fs::remove_dir_all(&root).ok();
        assert_eq!(
            neither,
            (Some(false), Some(false)),
            "nothing listed, nothing on the host"
        );
        assert_eq!(
            unread_neither,
            (None, None),
            "nothing read, nothing on the host"
        );
        assert_eq!(kfd_only, (Some(true), Some(true)), "kfd implies amdgpu");
        assert_eq!(
            unread_kfd_only,
            (Some(true), Some(true)),
            "kfd answers without a module list"
        );
        assert_eq!(
            pci_only,
            (Some(true), Some(false)),
            "amdgpu does not imply kfd"
        );
        assert_eq!(
            unread_pci_only,
            (Some(true), None),
            "amdgpu does not imply kfd"
        );
    }

    /// The wiring, not just the helpers: `probe_modules_from` must consult the
    /// host root. The module list it is handed names neither driver, so
    /// `Some(true)` can only come from the planted `/dev/kfd`, which answers
    /// `amdgpu_loaded` too.
    #[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
    #[test]
    fn probe_modules_reports_amdkfd_when_kfd_is_on_the_host() {
        use crate::hardware_root::TEST_HOST_ROOT_ENV;
        use crate::test_env::RestoredEnvVar;

        let root = std::env::temp_dir().join(format!(
            "rocm-core-probe-kfd-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        std::fs::create_dir_all(root.join("dev")).expect("plant dev");
        std::fs::write(root.join("dev/kfd"), "").expect("plant /dev/kfd");
        let (amdgpu_loaded, amdkfd_loaded) = {
            let _guard = crate::hardware_root::HOST_ROOT_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let _root = RestoredEnvVar::set(TEST_HOST_ROOT_ENV, &root);
            let mut e = Examination::default();
            probe_modules_from(&mut e, Some(NO_AMD_MODULES));
            (e.amdgpu_loaded, e.amdkfd_loaded)
        };
        std::fs::remove_dir_all(&root).ok();
        assert_eq!(amdkfd_loaded, Some(true));
        assert_eq!(amdgpu_loaded, Some(true), "kfd implies amdgpu");
    }
}
