// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Generic distro package-manager planning.
//!
//! The parts of a system-package install plan that do not depend on which
//! library is being installed: the package-manager model (`PackageManager`
//! and its resolution from an os-release `ID`/`ID_LIKE`), the plan and command
//! types (`SystemPackageInstallPlan`, `InstallCommand`), reading the host's
//! distribution to plan with a caller-supplied builder
//! (`plan_for_os_release_at`/`plan_for_os_release_read`), the install hints
//! built on those plans (`InstallHint`), the root/`sudo` checks that decide
//! whether an install can run unprompted, and the `apt-get -s` simulation
//! helpers that keep an automatic install from removing the ROCm stack.
//!
//! `openmpi.rs` owns the per-library builders and detection — OpenMPI,
//! libatomic and libnuma — and re-exports the public items here, so
//! `rocm_core::openmpi::...` paths are unchanged. A plan is never executed
//! here; callers present it for explicit approval and run it themselves.

#[cfg(target_os = "linux")]
use std::process::{Command, Stdio};

/// A single system-package install step, modeled as an argv vector rather than a
/// shell line.
///
/// Keeping the program and its arguments as separate tokens lets the runner
/// execute the command without a shell and decide *structurally* whether to
/// prepend `sudo` (see [`InstallCommand::resolved_argv`]) instead of
/// string-munging a `"sudo "` prefix. Every command and package name is a
/// hardcoded constant; nothing user-controlled is interpolated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallCommand {
    /// Program followed by its arguments, e.g.
    /// `["apt-get", "install", "-y", "libatomic1"]`.
    pub argv: Vec<String>,
    /// Whether the command must run as root. When the current process is not
    /// already root, the runner prepends `sudo`.
    pub needs_root: bool,
}

impl InstallCommand {
    /// Build a root-requiring command from borrowed argv tokens.
    pub(crate) fn sudo(argv: &[&str]) -> Self {
        Self {
            argv: argv.iter().map(|token| (*token).to_owned()).collect(),
            needs_root: true,
        }
    }

    /// The argv to execute, given whether the current process is already root.
    /// Prepends `sudo` only when the command needs root and we are not root (so
    /// it does not assume `sudo` exists when already running as root).
    pub fn resolved_argv(&self, root: bool) -> Vec<String> {
        if self.needs_root && !root {
            std::iter::once("sudo".to_owned())
                .chain(self.argv.iter().cloned())
                .collect()
        } else {
            self.argv.clone()
        }
    }
}

impl std::fmt::Display for InstallCommand {
    /// Render the command as it would run on a non-root host (with `sudo`), for
    /// display in plans, install hints, and error context.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.resolved_argv(false).join(" "))
    }
}

/// Cross-distro plan for installing a system package via the system package
/// manager.
///
/// Used for the OpenMPI runtime, libatomic, and libnuma. Commands are advisory
/// until explicitly approved and executed by the caller.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SystemPackageInstallPlan {
    /// Whether the host distribution maps to a known package manager.
    pub supported: bool,
    /// Detected package manager (`apt`, `dnf`, `zypper`, `pacman`), when known.
    pub package_manager: Option<String>,
    /// System packages the plan installs.
    pub packages: Vec<String>,
    /// Ordered install commands to run (each requires root).
    pub commands: Vec<InstallCommand>,
    /// Human-readable preconditions to verify before approving execution.
    pub preflight_checks: Vec<String>,
    /// Explanation of what the plan does and any distro-specific caveats.
    pub reason: String,
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
// Platform stub. The `cfg` sibling calls non-const code, so making only this
// arm `const` would give the two platforms different signatures and push
// `missing_const_for_fn` onto every caller in turn.
#[allow(clippy::missing_const_for_fn)]
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
pub const fn can_autoinstall() -> bool {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PackageManager {
    Apt,
    Dnf,
    Zypper,
    Pacman,
}

impl PackageManager {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Apt => "apt",
            Self::Dnf => "dnf",
            Self::Zypper => "zypper",
            Self::Pacman => "pacman",
        }
    }
}

/// One of the `build_*_install_plan` functions: a distribution's `ID` and
/// `ID_LIKE` to the plan that installs one runtime on it.
pub type InstallPlanBuilder = fn(&str, &str) -> SystemPackageInstallPlan;

/// `build`'s plan for the distribution the os-release `text` at `path`
/// declares, naming `path` instead of always `/etc/os-release` in an
/// unreadable-line message.
///
/// Unit tests that fixture bare `text` (never the real host) construct an
/// [`crate::hardware_root::OsRelease`] with
/// [`crate::hardware_root::OsRelease::found`] and call
/// [`plan_for_os_release_read`]; every production call site knows the real
/// path it read and calls this function or [`plan_for_os_release_read`]
/// directly.
///
/// An unreadable file (see [`crate::os_release`]) plans nothing, and the
/// reason names the line that made it unreadable and what to change on it,
/// ahead of `build`'s own manual-install advice for an unknown distribution.
/// Without that, the plan would blame the distribution for what is a fixable
/// line in the file.
#[must_use]
pub fn plan_for_os_release_at(
    path: &str,
    text: &str,
    build: InstallPlanBuilder,
) -> SystemPackageInstallPlan {
    match crate::os_release::id_and_id_like(text) {
        Ok((os_id, id_like)) => build(&os_id, &id_like),
        Err(unreadable) => {
            let mut plan = build("", "");
            plan.reason = format!(
                "{path} {unreadable}, so the distribution cannot be read; to fix it, {}. {}",
                unreadable.fix(),
                plan.reason
            );
            plan
        }
    }
}

/// [`plan_for_os_release_at`] for `read`.
///
/// Names the file actually read in an unreadable-line message, and reports a
/// file that exists but could not be read as itself rather than as an unknown
/// distribution. Without the latter, a host whose `/etc/os-release` is merely
/// unreadable (permission denied, a directory in its place) would say only
/// "Could not map distribution (ID=)" — the same message an actually unknown
/// distribution gets, sending the user looking for a support gap instead of
/// a filesystem problem. The real cause is now prepended ahead of that
/// generic message, not substituted for it — `build("", "")` still produces
/// it, since nothing else identifies the distribution either.
#[must_use]
pub fn plan_for_os_release_read(
    read: &crate::hardware_root::OsRelease,
    build: InstallPlanBuilder,
) -> SystemPackageInstallPlan {
    if let crate::hardware_root::OsRelease::Unreadable { path, error } = read {
        let mut plan = build("", "");
        plan.reason = format!(
            "{}, so the distribution cannot be read. {}",
            crate::hardware_root::unreadable_prefix(path, error),
            plan.reason
        );
        return plan;
    }
    plan_for_os_release_at(read.path(), read.text(), build)
}

/// What a user missing a runtime is told to do, and whether the plan behind
/// the advice installs anything.
///
/// `rocm engines install vllm --yes` runs the same plan
/// ([`plan_for_os_release_read`] over the same builder and the same real
/// file), so
/// a message may offer that command only when [`InstallHint::plan_installs`]
/// is true; [`InstallHint::or`] does the offering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallHint {
    /// The commands the plan runs, or the manual-install advice.
    pub text: String,
    /// Whether the plan is supported and has commands to run.
    pub plan_installs: bool,
}

impl InstallHint {
    /// The hint, followed by `, or {alternative}` only when the plan installs
    /// something. For an unknown distribution or an unreadable file, an
    /// installer that would plan nothing is not offered.
    #[must_use]
    pub fn or(&self, alternative: &str) -> String {
        if self.plan_installs {
            format!("{}, or {alternative}", self.text)
        } else {
            self.text.clone()
        }
    }
}

impl std::fmt::Display for InstallHint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text)
    }
}

/// The hint for os-release `text` read from `path`, naming that path in any
/// unreadable-line message: the commands [`plan_for_os_release_at`] plans
/// with `build` when it plans any, otherwise `manual` — followed, for an
/// unreadable file, by the line that made it so and what to change on it.
///
/// Off Linux this has no non-test caller — [`install_hint_for_read`],
/// its only production caller, is itself dead there for the same reason.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn install_hint_for_at(
    path: &str,
    text: &str,
    build: InstallPlanBuilder,
    manual: &str,
) -> InstallHint {
    if let Err(unreadable) = crate::os_release::id_and_id_like(text) {
        return InstallHint {
            text: format!(
                "{manual}; {path} {unreadable}, so the distribution cannot be read; \
                 to fix it, {}",
                unreadable.fix()
            ),
            plan_installs: false,
        };
    }
    let plan = plan_for_os_release_at(path, text, build);
    if plan.supported && !plan.commands.is_empty() {
        let rendered = plan
            .commands
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(" && ");
        return InstallHint {
            text: format!("install it with `{rendered}`"),
            plan_installs: true,
        };
    }
    InstallHint {
        text: manual.to_owned(),
        plan_installs: false,
    }
}

/// [`install_hint_for_at`] for `read`, naming the file actually read, and
/// reporting a file that exists but could not be read as itself rather than
/// as an unknown distribution — the same reasoning as
/// [`plan_for_os_release_read`], for the hint rather than the plan.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn install_hint_for_read(
    read: &crate::hardware_root::OsRelease,
    build: InstallPlanBuilder,
    manual: &str,
) -> InstallHint {
    if let crate::hardware_root::OsRelease::Unreadable { path, error } = read {
        return InstallHint {
            text: format!(
                "{manual}; {}, so the distribution cannot be read",
                crate::hardware_root::unreadable_prefix(path, error)
            ),
            plan_installs: false,
        };
    }
    install_hint_for_at(read.path(), read.text(), build, manual)
}

/// Build a non-mutating apt invocation for a planned `apt-get install` command.
///
/// Returns `None` for other commands. Any `sudo` prefix is dropped along with
/// assume-yes flags: `apt-get -s` changes nothing, so it needs neither root nor
/// approval, and keeping the resulting argv visibly non-destructive makes it
/// safe to log and test.
pub fn apt_simulate_argv(argv: &[String]) -> Option<Vec<String>> {
    let argv = match argv.split_first() {
        Some((program, rest)) if program == "sudo" => rest,
        _ => argv,
    };
    if argv.first().map(String::as_str) != Some("apt-get")
        || !argv.iter().any(|arg| arg == "install")
    {
        return None;
    }

    let mut simulated = Vec::with_capacity(argv.len() + 1);
    simulated.push("apt-get".to_owned());
    simulated.push("-s".to_owned());
    simulated.extend(
        argv.iter()
            .skip(1)
            .filter(|arg| !matches!(arg.as_str(), "-y" | "--yes" | "--assume-yes"))
            .cloned(),
    );
    Some(simulated)
}

/// Parse package removals from `apt-get -s` output.
///
/// Apt emits one `Remv <package> ...` operation per package after its readable
/// summary. Parsing those operation records avoids depending on summary wrapping
/// and strips apt's optional trailing `*` marker.
pub fn parse_apt_simulate_removals(output: &str) -> Vec<String> {
    output
        .lines()
        .filter_map(|line| {
            let package = line
                .trim_start()
                .strip_prefix("Remv ")?
                .split_whitespace()
                .next()?;
            Some(package.trim_end_matches('*').to_owned())
        })
        .collect()
}

/// Whether an apt package belongs to the ROCm/AMDGPU system stack and must not
/// be removed as a side effect of automatic dependency setup.
///
/// Matching is by *family prefix* rather than an exhaustive package list: ROCm
/// ships hundreds of packages and adds more every release, so enumerating exact
/// names would silently stop protecting whatever was added last. The prefixes
/// below cover the component families ROCm publishes — note that many are
/// dashless (`hipblas`, `rocminfo`, `hsakmt-roct`, `llvm-amdgpu`), so a
/// dash-separated match would miss them.
///
/// This deliberately favors recall over precision. A false positive only makes
/// the CLI refuse an automatic install and ask the user to resolve it by hand,
/// whereas a false negative silently breaks their ROCm installation. The
/// prefixes are still specific enough not to match the packages these plans
/// legitimately install or replace (`openmpi-bin`, `libnuma1`, `mpich`, ...).
pub fn is_protected_rocm_package(package: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "amd-smi",
        "amdgpu",
        "comgr",
        // hip, hipcc, hipblas, hipfft, hipsolver, hip-runtime-amd, hipify-clang
        "hip",
        // hsa-rocr, hsakmt-roct, hsa-amd-aqlprofile
        "hsa",
        "llvm-amdgpu",
        "migraphx",
        "miopen",
        "mivisionx",
        "openmp-extras",
        "rccl",
        // rocm*, plus the dashless libraries: rocblas, rocfft, rocrand,
        // rocsolver, rocsparse, rocthrust, rocprim, rocprofiler, roctracer,
        // rocalution, rocwmma, rocdecode, rocjpeg, rocdbgapi, rocminfo, rocr-runtime
        "roc",
        "rpp",
    ];
    // Strip any architecture qualifier (`amdgpu-core:amd64`) before matching.
    let package = package.split(':').next().unwrap_or(package);
    PREFIXES.iter().any(|prefix| package.starts_with(prefix))
}

pub(crate) fn resolve_package_manager(os_id: &str, id_like: &str) -> Option<PackageManager> {
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
    fn apt_simulate_argv_drops_assume_yes_and_simulates() {
        let install = InstallCommand::sudo(&["apt-get", "install", "-y", "openmpi-bin"]);
        assert_eq!(
            apt_simulate_argv(&install.argv).unwrap(),
            vec!["apt-get", "-s", "install", "openmpi-bin"]
        );
        // The executor passes the resolved argv, which carries a `sudo` prefix
        // when not already root; the simulation must still be recognized.
        assert_eq!(
            apt_simulate_argv(&install.resolved_argv(false)).unwrap(),
            vec!["apt-get", "-s", "install", "openmpi-bin"]
        );
        // Non-install apt commands and other package managers are not simulated.
        assert!(apt_simulate_argv(&InstallCommand::sudo(&["apt-get", "update"]).argv).is_none());
        assert!(
            apt_simulate_argv(&InstallCommand::sudo(&["dnf", "install", "-y", "openmpi"]).argv)
                .is_none()
        );
    }

    #[test]
    fn parses_removals_from_apt_simulation() {
        // Mirrors the EAI-7957 report: OpenMPI setup pulling older distro
        // toolchain packages while marking the ROCm stack for removal.
        let output = "\
Reading package lists...
The following packages will be REMOVED:
  rocm rocm-hip rocm-hip-runtime-dev mivisionx-dev rpp-dev
The following NEW packages will be installed:
  openmpi-bin libopenmpi-dev
Remv rocm [7.14.0]
Remv rocm-hip [7.14.0]
Remv rocm-hip-runtime-dev* [7.14.0]
Remv mivisionx-dev [7.14.0]
Remv rpp-dev [7.14.0]
Inst openmpi-bin (4.1.6 Ubuntu:24.04 [amd64])
";
        assert_eq!(
            parse_apt_simulate_removals(output),
            vec![
                "rocm",
                "rocm-hip",
                "rocm-hip-runtime-dev",
                "mivisionx-dev",
                "rpp-dev"
            ]
        );
        // A purely additive transaction removes nothing.
        assert!(parse_apt_simulate_removals("Inst libnuma1 (2.0.18 [amd64])\n").is_empty());
    }

    #[test]
    fn protects_rocm_stack_packages_only() {
        for package in [
            // The packages the EAI-7957 report saw apt mark for removal.
            "rocm",
            "rocm-hip",
            "rocm-hip-runtime-dev",
            "mivisionx-dev",
            "rpp-dev",
            // Core packages whose names do not follow a `<family>-` shape, so a
            // dash-separated match would miss them and let apt remove the ROCm
            // stack anyway.
            "hsakmt-roct",
            "llvm-amdgpu",
            "rocminfo",
            "hipblas",
            "rocblas",
            "hipcc",
            "comgr",
            "hsa-rocr",
            "openmp-extras-runtime",
            // Driver packages and architecture-qualified names.
            "amdgpu-dkms",
            "hip-runtime-amd",
            "amdgpu-core:amd64",
        ] {
            assert!(
                is_protected_rocm_package(package),
                "{package} must be protected"
            );
        }
        // The packages these plans legitimately install, and the MPI
        // implementation apt may swap out to satisfy them, must stay removable
        // or the guard would block its own install.
        for package in [
            "openmpi-bin",
            "libopenmpi-dev",
            "openmpi4",
            "mpich",
            "libmpich-dev",
            "libnuma1",
            "numactl",
            "libatomic1",
        ] {
            assert!(
                !is_protected_rocm_package(package),
                "{package} must not be protected"
            );
        }
    }

    #[test]
    fn install_command_prepends_sudo_only_when_not_root() {
        let command = InstallCommand::sudo(&["dnf", "install", "-y", "openmpi"]);
        // Non-root: sudo is prepended, both in the executed argv and the display.
        assert_eq!(
            command.resolved_argv(false),
            vec!["sudo", "dnf", "install", "-y", "openmpi"]
        );
        assert_eq!(command.to_string(), "sudo dnf install -y openmpi");
        // Root: sudo (which may be absent) is dropped from the executed argv.
        assert_eq!(
            command.resolved_argv(true),
            vec!["dnf", "install", "-y", "openmpi"]
        );
    }
}
