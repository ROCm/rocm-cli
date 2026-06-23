//! OpenMPI detection and cross-distro install planning.
//!
//! vLLM's ROCm wheels link against the OpenMPI runtime (`libmpi.so`) and use
//! `mpirun` for multi-process/multi-node execution. OpenMPI is a system library
//! that has no reliable, no-sudo, cross-distro Python wheel, so this module owns:
//!
//! * detecting whether a usable OpenMPI runtime is already present, and
//! * building an approval-gated, distro-aware system-package install plan.
//!
//! The plan is never executed here; callers present it for explicit approval and
//! run it through their normal privileged-command flow (mirroring the native
//! driver-install plan). Detection is Linux/WSL only — native Windows vLLM is
//! unsupported, so the caller skips this path there.

use std::path::PathBuf;
#[cfg(target_os = "linux")]
use std::process::{Command, Stdio};

/// Result of probing the host for an OpenMPI runtime.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenMpiStatus {
    /// Whether a usable `libmpi.so*` was located (what vLLM needs at import time).
    pub present: bool,
    /// Path to the discovered `libmpi.so*`, when known.
    pub libmpi_path: Option<PathBuf>,
    /// Path to the discovered `mpirun` launcher, when known.
    pub mpirun_path: Option<PathBuf>,
}

/// Cross-distro plan for installing the OpenMPI runtime via the system package
/// manager. Commands are advisory until explicitly approved and executed by the
/// caller.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenMpiInstallPlan {
    /// Whether the host distribution maps to a known package manager.
    pub supported: bool,
    /// Detected package manager (`apt`, `dnf`, `zypper`, `pacman`), when known.
    pub package_manager: Option<String>,
    /// System packages the plan installs.
    pub packages: Vec<String>,
    /// Ordered shell commands to run (each requires root via `sudo`).
    pub commands: Vec<String>,
    /// Human-readable preconditions to verify before approving execution.
    pub preflight_checks: Vec<String>,
    /// Explanation of what the plan does and any distro-specific caveats.
    pub reason: String,
}

/// Candidate directories that hold OpenMPI shared libraries.
///
/// Covers distributions that install OpenMPI outside the default loader path
/// (notably RHEL-family layouts under `/usr/lib64/openmpi/lib`). Only existing
/// directories are returned.
pub fn openmpi_library_dirs() -> Vec<PathBuf> {
    const CANDIDATES: &[&str] = &[
        "/usr/lib64/openmpi/lib",
        "/usr/lib/x86_64-linux-gnu/openmpi/lib",
        "/usr/lib/openmpi/lib",
        "/usr/lib/aarch64-linux-gnu/openmpi/lib",
    ];
    CANDIDATES
        .iter()
        .map(PathBuf::from)
        .filter(|path| path.is_dir())
        .collect()
}

/// Probe the host for a usable OpenMPI runtime. Returns a default (not present)
/// status on non-Linux hosts, where this path is not exercised.
#[cfg(target_os = "linux")]
pub fn detect_openmpi() -> OpenMpiStatus {
    let libmpi_path = ldconfig_libmpi_path().or_else(scan_libmpi_paths);
    let mpirun_path = find_mpirun_path();
    OpenMpiStatus {
        present: libmpi_path.is_some(),
        libmpi_path,
        mpirun_path,
    }
}

/// Non-Linux hosts do not exercise the OpenMPI install path.
#[cfg(not(target_os = "linux"))]
pub fn detect_openmpi() -> OpenMpiStatus {
    OpenMpiStatus::default()
}

/// Whether the current process is running as root (effective UID 0).
#[cfg(target_os = "linux")]
#[allow(unsafe_code)] // libc FFI
pub fn running_as_root() -> bool {
    // SAFETY: `geteuid` takes no arguments and is always safe to call.
    unsafe { libc::geteuid() == 0 }
}

/// Whether the current process is running as root. Always false off Linux.
#[cfg(not(target_os = "linux"))]
pub fn running_as_root() -> bool {
    false
}

/// Whether privileged package installs can run without an interactive prompt:
/// either the process is already root, or passwordless `sudo` is configured.
///
/// Used to decide whether the OpenMPI install can proceed automatically without
/// an explicit `--yes` approval (which exists to authorize an interactive sudo
/// password prompt).
#[cfg(target_os = "linux")]
pub fn can_autoinstall() -> bool {
    running_as_root() || sudo_noninteractive_available()
}

/// Privileged auto-install is never attempted off Linux.
#[cfg(not(target_os = "linux"))]
pub fn can_autoinstall() -> bool {
    false
}

#[cfg(target_os = "linux")]
fn sudo_noninteractive_available() -> bool {
    Command::new("sudo")
        .args(["-n", "true"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

#[cfg(target_os = "linux")]
fn ldconfig_libmpi_path() -> Option<PathBuf> {
    let output = Command::new("ldconfig").arg("-p").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    parse_ldconfig_libmpi(&text)
}

#[cfg(target_os = "linux")]
fn scan_libmpi_paths() -> Option<PathBuf> {
    const DIRS: &[&str] = &[
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
        "/usr/lib64/openmpi/lib",
        "/usr/lib/x86_64-linux-gnu/openmpi/lib",
        "/usr/lib/openmpi/lib",
    ];
    for dir in DIRS {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("libmpi.so") {
                return Some(entry.path());
            }
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn find_mpirun_path() -> Option<PathBuf> {
    const EXTRA: &[&str] = &[
        "/usr/lib64/openmpi/bin/mpirun",
        "/usr/lib/x86_64-linux-gnu/openmpi/bin/mpirun",
    ];
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            let candidate = dir.join("mpirun");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    EXTRA
        .iter()
        .map(PathBuf::from)
        .find(|candidate| candidate.is_file())
}

/// Parse the output of `ldconfig -p` and return the path of the first `libmpi.so*`
/// entry, if any.
pub(crate) fn parse_ldconfig_libmpi(output: &str) -> Option<PathBuf> {
    for line in output.lines() {
        let line = line.trim();
        if !line.starts_with("libmpi.so") {
            continue;
        }
        let Some((_, path)) = line.split_once("=>") else {
            continue;
        };
        let path = path.trim();
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackageManager {
    Apt,
    Dnf,
    Zypper,
    Pacman,
}

impl PackageManager {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Apt => "apt",
            Self::Dnf => "dnf",
            Self::Zypper => "zypper",
            Self::Pacman => "pacman",
        }
    }
}

/// Build an approval-gated OpenMPI install plan for a distribution identity.
///
/// `os_id` is the `/etc/os-release` `ID` field, `id_like` is the (possibly
/// empty) `ID_LIKE` field, and `version_id` is `VERSION_ID`. Matching is
/// case-insensitive.
pub fn build_openmpi_install_plan(
    os_id: &str,
    id_like: &str,
    version_id: &str,
) -> OpenMpiInstallPlan {
    let _ = version_id;
    let os_id = os_id.trim().to_ascii_lowercase();
    let id_like = id_like.trim().to_ascii_lowercase();

    let Some(manager) = resolve_package_manager(&os_id, &id_like) else {
        return OpenMpiInstallPlan {
            supported: false,
            package_manager: None,
            packages: Vec::new(),
            commands: Vec::new(),
            preflight_checks: Vec::new(),
            reason: format!(
                "Could not map distribution (ID={}) to a known package manager; install the OpenMPI runtime manually (e.g. your distro's `openmpi` / `libopenmpi` package providing libmpi.so and mpirun).",
                if os_id.is_empty() {
                    "<unknown>"
                } else {
                    &os_id
                }
            ),
        };
    };

    let (packages, commands, reason) = match manager {
        PackageManager::Apt => (
            vec!["libopenmpi3".to_owned(), "openmpi-bin".to_owned()],
            vec![
                "sudo apt-get update".to_owned(),
                "sudo apt-get install -y libopenmpi3 openmpi-bin".to_owned(),
            ],
            "Install the OpenMPI runtime (libmpi.so) and mpirun that vLLM's ROCm wheels require. Requires root via apt; approve before execution.".to_owned(),
        ),
        PackageManager::Dnf => (
            vec!["openmpi".to_owned(), "openmpi-devel".to_owned()],
            vec!["sudo dnf install -y openmpi openmpi-devel".to_owned()],
            "Install the OpenMPI runtime that vLLM's ROCm wheels require. Requires root via dnf; approve before execution. Note: RHEL-family packages install OpenMPI under /usr/lib64/openmpi (use `module load mpi/openmpi-x86_64` for an interactive shell); rocm-cli adds this directory to the vLLM launch library path automatically.".to_owned(),
        ),
        PackageManager::Zypper => (
            vec!["openmpi4".to_owned(), "openmpi4-devel".to_owned()],
            vec!["sudo zypper install -y openmpi4 openmpi4-devel".to_owned()],
            "Install the OpenMPI runtime that vLLM's ROCm wheels require. Requires root via zypper; approve before execution. If openmpi4 is unavailable, substitute your distribution's current openmpi package.".to_owned(),
        ),
        PackageManager::Pacman => (
            vec!["openmpi".to_owned()],
            vec!["sudo pacman -S --needed --noconfirm openmpi".to_owned()],
            "Install the OpenMPI runtime that vLLM's ROCm wheels require. Requires root via pacman; approve before execution.".to_owned(),
        ),
    };

    OpenMpiInstallPlan {
        supported: true,
        package_manager: Some(manager.as_str().to_owned()),
        packages,
        commands,
        preflight_checks: vec![
            "root access: run as root, or ensure `sudo -v` succeeds before approval".to_owned(),
            "`sudo` command is available when not running as root".to_owned(),
            format!("`{}` package manager is available", manager.as_str()),
        ],
        reason,
    }
}

fn resolve_package_manager(os_id: &str, id_like: &str) -> Option<PackageManager> {
    const APT: &[&str] = &[
        "ubuntu",
        "debian",
        "linuxmint",
        "pop",
        "raspbian",
        "elementary",
    ];
    const DNF: &[&str] = &[
        "rhel",
        "centos",
        "fedora",
        "rocky",
        "almalinux",
        "ol",
        "oracle",
        "amzn",
    ];
    const ZYPPER: &[&str] = &[
        "sles",
        "sle",
        "opensuse",
        "opensuse-leap",
        "opensuse-tumbleweed",
    ];
    const PACMAN: &[&str] = &["arch", "manjaro", "endeavouros", "arcolinux"];

    let matches = |ids: &[&str]| {
        ids.iter().any(|candidate| {
            os_id == *candidate || id_like.split_whitespace().any(|like| like == *candidate)
        })
    };

    if matches(APT) {
        Some(PackageManager::Apt)
    } else if matches(DNF) {
        Some(PackageManager::Dnf)
    } else if matches(ZYPPER) {
        Some(PackageManager::Zypper)
    } else if matches(PACMAN) {
        Some(PackageManager::Pacman)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_libmpi_from_ldconfig() {
        let output = "\tlibm.so.6 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libm.so.6\n\tlibmpi.so.40 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libmpi.so.40\n";
        assert_eq!(
            parse_ldconfig_libmpi(output),
            Some(PathBuf::from("/usr/lib/x86_64-linux-gnu/libmpi.so.40"))
        );
    }

    #[test]
    fn ldconfig_without_libmpi_returns_none() {
        let output = "\tlibm.so.6 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libm.so.6\n";
        assert_eq!(parse_ldconfig_libmpi(output), None);
    }

    #[test]
    fn apt_plan_for_ubuntu() {
        let plan = build_openmpi_install_plan("ubuntu", "debian", "24.04");
        assert!(plan.supported);
        assert_eq!(plan.package_manager.as_deref(), Some("apt"));
        assert!(plan.packages.contains(&"libopenmpi3".to_owned()));
        assert!(
            plan.commands
                .iter()
                .any(|cmd| cmd.contains("apt-get install -y libopenmpi3 openmpi-bin"))
        );
    }

    #[test]
    fn apt_plan_via_id_like() {
        let plan = build_openmpi_install_plan("linuxmint", "ubuntu debian", "21");
        assert_eq!(plan.package_manager.as_deref(), Some("apt"));
    }

    #[test]
    fn dnf_plan_for_rhel_family() {
        for id in ["rhel", "fedora", "rocky", "almalinux", "ol"] {
            let plan = build_openmpi_install_plan(id, "", "9");
            assert_eq!(
                plan.package_manager.as_deref(),
                Some("dnf"),
                "expected dnf for {id}"
            );
            assert!(plan.commands.iter().any(|cmd| cmd.contains("dnf install")));
        }
    }

    #[test]
    fn dnf_plan_via_id_like_rhel() {
        let plan = build_openmpi_install_plan("oraclelinux", "fedora", "9");
        assert_eq!(plan.package_manager.as_deref(), Some("dnf"));
    }

    #[test]
    fn zypper_plan_for_suse() {
        let plan = build_openmpi_install_plan("opensuse-leap", "suse opensuse", "15.6");
        assert_eq!(plan.package_manager.as_deref(), Some("zypper"));
        assert!(
            plan.commands
                .iter()
                .any(|cmd| cmd.contains("zypper install"))
        );
    }

    #[test]
    fn pacman_plan_for_arch() {
        let plan = build_openmpi_install_plan("arch", "", "");
        assert_eq!(plan.package_manager.as_deref(), Some("pacman"));
        assert!(plan.commands.iter().any(|cmd| cmd.contains("pacman -S")));
    }

    #[test]
    fn unknown_distro_is_unsupported() {
        let plan = build_openmpi_install_plan("void", "", "");
        assert!(!plan.supported);
        assert!(plan.package_manager.is_none());
        assert!(plan.commands.is_empty());
        assert!(plan.reason.contains("manually"));
    }
}
