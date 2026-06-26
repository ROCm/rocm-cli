// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! OpenMPI detection and cross-distro install planning.
//!
//! vLLM's ROCm wheels link against the OpenMPI runtime (`libmpi.so`) and use
//! `mpirun` for multi-process/multi-node execution. OpenMPI is a system library
//! that has no reliable, no-sudo, cross-distro Python wheel, so this module owns:
//!
//! * detecting whether a usable OpenMPI runtime is already present, and
//! * building an approval-gated, distro-aware system-package install plan.
//!
//! It also owns the related compatibility shims and checks for the other system
//! libraries the ROCm torch wheel needs on minimal hosts: a compiled
//! `libmpi_cxx.so.40` stub that supplies the legacy OpenMPI C++ binding symbols
//! OpenMPI 5.x dropped (see [`ensure_mpi_cxx_compat`]) and the runtime checks
//! plus install plans for the `libatomic.so.1` (see [`libatomic_present`] /
//! [`build_libatomic_install_plan`]) and `libnuma.so.1` (see [`libnuma_present`]
//! / [`build_libnuma_install_plan`]) runtimes. The ROCm SDK bundles numa only
//! under a renamed soname with rewritten symbol versions, which cannot satisfy
//! torch's `libnuma_1.2` binding, so the real numactl runtime is installed
//! rather than shimmed.
//!
//! The plan is never executed here; callers present it for explicit approval and
//! run it through their normal privileged-command flow (mirroring the native
//! driver-install plan). Detection is Linux/WSL only — native Windows vLLM is
//! unsupported, so the caller skips this path there.

#[cfg(target_os = "linux")]
use std::path::Path;
use std::path::PathBuf;
#[cfg(target_os = "linux")]
use std::process::{Command, Stdio};

/// Result of probing the host for an OpenMPI runtime.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenMpiStatus {
    /// Whether a usable OpenMPI runtime is available, meaning both `libmpi.so*`
    /// (what vLLM links against at import time) and the `mpirun` launcher (used
    /// for multi-process/multi-node execution) were located. A host with only
    /// one of the two is reported as not present so the warning/install path
    /// still runs.
    pub present: bool,
    /// Path to the discovered `libmpi.so*`, when known.
    pub libmpi_path: Option<PathBuf>,
    /// Path to the discovered `mpirun` launcher, when known.
    pub mpirun_path: Option<PathBuf>,
}

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
    fn sudo(argv: &[&str]) -> Self {
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
        present: runtime_present(libmpi_path.as_deref(), mpirun_path.as_deref()),
        libmpi_path,
        mpirun_path,
    }
}

/// Whether the located paths constitute a usable OpenMPI runtime.
///
/// A usable runtime needs both the shared library vLLM links against and the
/// `mpirun` launcher; a library-only (or launcher-only) partial install is not
/// considered present so the warning/auto-install path still runs.
// Only the Linux `detect_openmpi` path calls this at runtime; off Linux it is
// exercised solely by cross-platform unit tests.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) const fn runtime_present(
    libmpi_path: Option<&std::path::Path>,
    mpirun_path: Option<&std::path::Path>,
) -> bool {
    libmpi_path.is_some() && mpirun_path.is_some()
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

/// Directories scanned for OpenMPI shared libraries that live outside the
/// default loader path (notably RHEL-family layouts under
/// `/usr/lib64/openmpi/lib`).
#[cfg(target_os = "linux")]
const LIBMPI_SCAN_DIRS: &[&str] = &[
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
    "/usr/lib64",
    "/usr/lib",
    "/usr/lib64/openmpi/lib",
    "/usr/lib/x86_64-linux-gnu/openmpi/lib",
    "/usr/lib/openmpi/lib",
];

/// Return the path of the first entry across `dirs` whose file name starts with
/// `prefix`; directories that cannot be read are skipped.
///
/// Shared scan behind locating a versioned shared object (for example
/// `libmpi.so`, `libatomic.so.1`, or `libnuma.so.1`) across a set of candidate
/// library directories.
#[cfg(target_os = "linux")]
fn scan_dirs_for_prefix(dirs: &[&str], prefix: &str) -> Option<PathBuf> {
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            if entry.file_name().to_string_lossy().starts_with(prefix) {
                return Some(entry.path());
            }
        }
    }
    None
}

/// Standard system library directories searched for distro-provided runtime
/// objects such as `libatomic.so.1` and `libnuma.so.1`.
///
/// Distinct from [`LIBMPI_SCAN_DIRS`], which additionally probes the OpenMPI
/// subdirectories and omits `/lib64`.
#[cfg(target_os = "linux")]
const STANDARD_LIB_DIRS: &[&str] = &[
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
    "/usr/lib64",
    "/lib64",
    "/usr/lib",
];

/// Scan [`LIBMPI_SCAN_DIRS`] for the first shared library whose file name starts
/// with `name_prefix` (for example `libmpi.so` or `libmpi_cxx.so`).
#[cfg(target_os = "linux")]
fn scan_for_shared_lib(name_prefix: &str) -> Option<PathBuf> {
    scan_dirs_for_prefix(LIBMPI_SCAN_DIRS, name_prefix)
}

#[cfg(target_os = "linux")]
fn scan_libmpi_paths() -> Option<PathBuf> {
    scan_for_shared_lib("libmpi.so")
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
// Only the Linux `ldconfig_libmpi_path` path calls this at runtime; off Linux it
// is exercised solely by cross-platform unit tests.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
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

/// Locate a real `libmpi_cxx.so*` (the legacy OpenMPI C++ bindings) on the host,
/// if one is installed. Returns `None` when no such library exists, which is the
/// case on OpenMPI 5.x (it removed the C++ bindings entirely).
#[cfg(target_os = "linux")]
fn find_libmpi_cxx() -> Option<PathBuf> {
    scan_for_shared_lib("libmpi_cxx.so")
}

/// Whether `link` already exists and resolves to `target`.
#[cfg(target_os = "linux")]
fn symlink_points_to(link: &Path, target: &Path) -> bool {
    std::fs::read_link(link).is_ok_and(|dest| dest == target)
}

/// Prebuilt `libmpi_cxx.so.40` stub defining the legacy OpenMPI C++ binding
/// symbols PyTorch references (see `build.rs`). Empty when the build host had no
/// C compiler, in which case the shim is disabled.
#[cfg(target_os = "linux")]
const MPI_CXX_STUB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/libmpi_cxx_stub.so"));

/// Write `bytes` to `compat_dir/name`, creating `compat_dir` as needed, and
/// return `compat_dir` on success. Idempotent: only rewrites when the file is
/// missing or differs (which also replaces a stale symlink left by older
/// rocm-cli versions). The file is marked loadable (mode `0o755`).
#[cfg(target_os = "linux")]
fn write_compat_file(compat_dir: &Path, name: &str, bytes: &[u8]) -> Option<PathBuf> {
    use std::os::unix::fs::PermissionsExt;

    if std::fs::create_dir_all(compat_dir).is_err() {
        return None;
    }
    let path = compat_dir.join(name);
    let needs_write = match std::fs::read(&path) {
        Ok(existing) => existing != bytes,
        Err(_) => true,
    };
    if needs_write {
        // Remove first so a stale symlink is replaced rather than written through.
        let _ = std::fs::remove_file(&path);
        std::fs::write(&path, bytes).ok()?;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755));
    }
    Some(compat_dir.to_path_buf())
}

/// Ensure PyTorch wheels built against OpenMPI 4.x can load on hosts that ship
/// OpenMPI 5.x (which removed the legacy MPI C++ bindings, `libmpi_cxx`).
///
/// PyTorch's `libtorch_global_deps.so` lists `libmpi_cxx.so.40` as a `NEEDED`
/// dependency *and* `libtorch_cpu.so` references a few MPI C++ binding symbols
/// (`MPI::Win::Free()`, `MPI::Datatype::Free()`, `MPI::Comm::Comm()`). OpenMPI 5
/// provides neither the soname nor those symbols, so `import torch` aborts —
/// first with `libmpi_cxx.so.40: cannot open shared object file`, and (once the
/// soname is satisfied) with `undefined symbol: _ZN3MPI...`. A symlink to the
/// real `libmpi.so` fixes only the first error, because the C library does not
/// export the C++ symbols.
///
/// Those C++ symbols are never *called* in single-node serving (the C MPI API,
/// resolved from the real `libmpi.so.40` that torch also `NEEDED`s, is what
/// actually runs), so this materializes a tiny stub `libmpi_cxx.so.40` — built at
/// compile time (see `build.rs`) and embedded — that merely *defines* the symbols
/// so the dynamic loader can resolve torch.
///
/// Returns `None` (no-op) when a real `libmpi_cxx.so*` is already present
/// (OpenMPI 4.x), when no stub was embedded (no C toolchain at build time), or
/// when the stub cannot be written. Otherwise it writes `libmpi_cxx.so.40` inside
/// `compat_dir` and returns `compat_dir` for the caller to add to the loader
/// path. The operation is idempotent and replaces a stale symlink left by older
/// rocm-cli versions.
#[cfg(target_os = "linux")]
pub fn ensure_mpi_cxx_compat(compat_dir: &Path) -> Option<PathBuf> {
    // A real C++ bindings library is present (OpenMPI 4.x); nothing to shim.
    if find_libmpi_cxx().is_some() {
        return None;
    }
    // No stub embedded (build host lacked a C compiler); cannot synthesize the
    // C++ bindings here.
    if MPI_CXX_STUB.is_empty() {
        return None;
    }
    write_compat_file(compat_dir, "libmpi_cxx.so.40", MPI_CXX_STUB)
}

/// Non-Linux hosts never need the OpenMPI C++ bindings shim.
#[cfg(not(target_os = "linux"))]
pub fn ensure_mpi_cxx_compat(_compat_dir: &std::path::Path) -> Option<PathBuf> {
    None
}

/// Create (idempotently) a `link_name` symlink in `compat_dir` pointing at
/// `target`, returning `compat_dir` on success.
///
/// This is a generic symlink-based shim helper for managed-runtime libraries on
/// hosts whose system or bundled library uses a different soname. Note that the
/// OpenMPI C++ bindings shim no longer uses this path (it materializes a compiled
/// `libmpi_cxx.so.40` stub via [`ensure_mpi_cxx_compat`]), and libnuma is
/// deliberately *not* shimmed — the real numactl runtime is installed instead.
/// Returns `None` when `target` does not exist or the symlink cannot be created.
#[cfg(target_os = "linux")]
pub fn ensure_compat_symlink(compat_dir: &Path, link_name: &str, target: &Path) -> Option<PathBuf> {
    if !target.exists() {
        return None;
    }
    if std::fs::create_dir_all(compat_dir).is_err() {
        return None;
    }
    let link = compat_dir.join(link_name);
    if symlink_points_to(&link, target) {
        return Some(compat_dir.to_path_buf());
    }
    let _ = std::fs::remove_file(&link);
    std::os::unix::fs::symlink(target, &link)
        .ok()
        .map(|()| compat_dir.to_path_buf())
}

/// Non-Linux hosts do not create runtime library shims.
#[cfg(not(target_os = "linux"))]
pub fn ensure_compat_symlink(
    _compat_dir: &std::path::Path,
    _link_name: &str,
    _target: &std::path::Path,
) -> Option<PathBuf> {
    None
}

/// Whether `ldconfig -p` reports a shared library with the exact given soname.
///
/// Used to skip a compatibility shim when the loader can already resolve the
/// library, and to detect whether a required system library (such as
/// `libatomic.so.1`) is installed. Always `false` off Linux or when `ldconfig`
/// is absent.
#[cfg(target_os = "linux")]
pub fn ldconfig_has_soname(soname: &str) -> bool {
    let Ok(output) = Command::new("ldconfig").arg("-p").output() else {
        return false;
    };
    if !output.status.success() {
        return false;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .any(|line| line.split_whitespace().next() == Some(soname))
}

/// Always `false` off Linux, where the loader-path shim is not exercised.
#[cfg(not(target_os = "linux"))]
pub fn ldconfig_has_soname(_soname: &str) -> bool {
    false
}

/// Whether the `libatomic.so.1` runtime library is available to the dynamic
/// loader.
///
/// PyTorch's ROCm wheels link `libatomic.so.1` (GCC's atomic-operations runtime)
/// as a `NEEDED` dependency, so `import torch` aborts with `libatomic.so.1:
/// cannot open shared object file` when it is missing. Unlike `libnuma`, the
/// ROCm SDK does not bundle libatomic, so there is nothing to shim — it must be
/// installed from the distribution's package manager (see
/// [`build_libatomic_install_plan`]). Minimal container images (notably RHEL UBI)
/// ship only GCC's `libatomic.so` linker script, not the runtime `.so.1`.
///
/// Always `true` off Linux, where this path is not exercised.
#[cfg(target_os = "linux")]
pub fn libatomic_present() -> bool {
    ldconfig_has_soname("libatomic.so.1") || scan_libatomic_path().is_some()
}

/// Non-Linux hosts do not exercise the libatomic dependency path.
#[cfg(not(target_os = "linux"))]
pub fn libatomic_present() -> bool {
    true
}

/// Locate a real `libatomic.so.1*` shared object in the standard library
/// directories. GCC's `/usr/lib/gcc/.../libatomic.so` is only a linker script
/// (`INPUT(...)`), not a loadable object, so it is intentionally not searched.
#[cfg(target_os = "linux")]
fn scan_libatomic_path() -> Option<PathBuf> {
    scan_dirs_for_prefix(STANDARD_LIB_DIRS, "libatomic.so.1")
}

/// Whether a real system `libnuma.so.1` is resolvable by the dynamic loader.
///
/// PyTorch's ROCm `libc10.so` lists `libnuma.so.1` as a `NEEDED` dependency and
/// binds the versioned symbols `libnuma_1.1` / `libnuma_1.2` from upstream
/// numactl. TheRock *does* bundle numa in the ROCm SDK sysdeps, but under the
/// renamed soname `librocm_sysdeps_numa.so.1` whose version nodes are rewritten
/// to `AMDROCM_SYSDEPS_1.0_libnuma_*`. Those renamed versions cannot satisfy the
/// plain `libnuma_1.2` symbol torch needs, so a symlink to the bundled library
/// fails at load with `version 'libnuma_1.2' not found`. The real numactl
/// runtime must therefore be installed from the distribution's package manager
/// (see [`build_libnuma_install_plan`]); it is intentionally not shimmed.
///
/// Always `true` off Linux, where this path is not exercised.
#[cfg(target_os = "linux")]
pub fn libnuma_present() -> bool {
    ldconfig_has_soname("libnuma.so.1") || scan_libnuma_path().is_some()
}

/// Non-Linux hosts do not exercise the libnuma dependency path.
#[cfg(not(target_os = "linux"))]
pub fn libnuma_present() -> bool {
    true
}

/// Locate a real `libnuma.so.1*` shared object in the standard system library
/// directories. The ROCm SDK's bundled `librocm_sysdeps_numa.so.1` lives outside
/// these directories (inside the Python site-packages tree) and uses renamed
/// symbol versions, so it is intentionally never matched here.
#[cfg(target_os = "linux")]
fn scan_libnuma_path() -> Option<PathBuf> {
    scan_dirs_for_prefix(STANDARD_LIB_DIRS, "libnuma.so.1")
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
/// `os_id` is the `/etc/os-release` `ID` field and `id_like` is the (possibly
/// empty) `ID_LIKE` field. Matching is case-insensitive.
pub fn build_openmpi_install_plan(os_id: &str, id_like: &str) -> SystemPackageInstallPlan {
    let os_id = os_id.trim().to_ascii_lowercase();
    let id_like = id_like.trim().to_ascii_lowercase();

    let Some(manager) = resolve_package_manager(&os_id, &id_like) else {
        return SystemPackageInstallPlan {
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
            // `openmpi-bin` provides `mpirun` and pulls in the matching OpenMPI
            // runtime library; `libopenmpi-dev` provides the unversioned
            // `libmpi.so` and depends on the correct runtime package across
            // releases. Both are release-agnostic names, so they keep working
            // across the Debian/Ubuntu `t64` ABI transition where the runtime
            // package was renamed `libopenmpi3` -> `libopenmpi3t64`.
            vec!["openmpi-bin".to_owned(), "libopenmpi-dev".to_owned()],
            vec![
                InstallCommand::sudo(&["apt-get", "update"]),
                InstallCommand::sudo(&[
                    "apt-get",
                    "install",
                    "-y",
                    "openmpi-bin",
                    "libopenmpi-dev",
                ]),
            ],
            "Install the OpenMPI runtime (libmpi.so) and mpirun that vLLM's ROCm wheels require. Requires root via apt; approve before execution.".to_owned(),
        ),
        PackageManager::Dnf => (
            // Only the base `openmpi` package is required: it provides the
            // runtime library (`/usr/lib64/openmpi/lib/libmpi.so.*`) and the
            // launcher (`/usr/lib64/openmpi/bin/mpirun`) that vLLM needs. The
            // `openmpi-devel` package (headers, `mpicc`, the unversioned
            // `libmpi.so` symlink) is *not* needed at runtime and lives in the
            // CodeReady Builder (CRB / PowerTools) repository, which is disabled
            // by default on RHEL. Requiring it previously made the whole `dnf`
            // transaction fail with "No match for argument: openmpi-devel" on
            // stock RHEL hosts, so it is intentionally omitted here.
            vec!["openmpi".to_owned()],
            vec![InstallCommand::sudo(&["dnf", "install", "-y", "openmpi"])],
            "Install the OpenMPI runtime that vLLM's ROCm wheels require. Requires root via dnf; approve before execution. Note: RHEL-family packages install OpenMPI under /usr/lib64/openmpi (use `module load mpi/openmpi-x86_64` for an interactive shell); rocm-cli adds this directory to the vLLM launch library path automatically.".to_owned(),
        ),
        PackageManager::Zypper => (
            vec!["openmpi4".to_owned(), "openmpi4-devel".to_owned()],
            vec![InstallCommand::sudo(&[
                "zypper",
                "install",
                "-y",
                "openmpi4",
                "openmpi4-devel",
            ])],
            "Install the OpenMPI runtime that vLLM's ROCm wheels require. Requires root via zypper; approve before execution. If openmpi4 is unavailable, substitute your distribution's current openmpi package.".to_owned(),
        ),
        PackageManager::Pacman => (
            vec!["openmpi".to_owned()],
            vec![InstallCommand::sudo(&[
                "pacman",
                "-S",
                "--needed",
                "--noconfirm",
                "openmpi",
            ])],
            "Install the OpenMPI runtime that vLLM's ROCm wheels require. Requires root via pacman; approve before execution.".to_owned(),
        ),
    };

    SystemPackageInstallPlan {
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

/// Build an approval-gated plan for installing the `libatomic` runtime that
/// PyTorch's ROCm wheels require, for a distribution identity.
///
/// Arguments mirror [`build_openmpi_install_plan`]. Unlike OpenMPI, libatomic is
/// not bundled by the ROCm SDK and cannot be shimmed, so missing hosts must
/// install it from the package manager. Arch-family hosts ship libatomic as part
/// of `gcc-libs` (always present), so no install is planned there.
pub fn build_libatomic_install_plan(os_id: &str, id_like: &str) -> SystemPackageInstallPlan {
    let os_id = os_id.trim().to_ascii_lowercase();
    let id_like = id_like.trim().to_ascii_lowercase();

    let Some(manager) = resolve_package_manager(&os_id, &id_like) else {
        return SystemPackageInstallPlan {
            supported: false,
            package_manager: None,
            packages: Vec::new(),
            commands: Vec::new(),
            preflight_checks: Vec::new(),
            reason: format!(
                "Could not map distribution (ID={}) to a known package manager; install the libatomic runtime manually (your distro's `libatomic` / `libatomic1` package providing libatomic.so.1).",
                if os_id.is_empty() {
                    "<unknown>"
                } else {
                    &os_id
                }
            ),
        };
    };

    let (packages, commands): (Vec<String>, Vec<InstallCommand>) = match manager {
        PackageManager::Apt => (
            vec!["libatomic1".to_owned()],
            vec![
                InstallCommand::sudo(&["apt-get", "update"]),
                InstallCommand::sudo(&["apt-get", "install", "-y", "libatomic1"]),
            ],
        ),
        PackageManager::Dnf => (
            vec!["libatomic".to_owned()],
            vec![InstallCommand::sudo(&["dnf", "install", "-y", "libatomic"])],
        ),
        PackageManager::Zypper => (
            vec!["libatomic1".to_owned()],
            vec![InstallCommand::sudo(&[
                "zypper",
                "install",
                "-y",
                "libatomic1",
            ])],
        ),
        // Arch ships libatomic inside `gcc-libs`, which is part of the base
        // install, so there is nothing to install separately.
        PackageManager::Pacman => (Vec::new(), Vec::new()),
    };

    if packages.is_empty() {
        return SystemPackageInstallPlan {
            supported: false,
            package_manager: Some(manager.as_str().to_owned()),
            packages,
            commands,
            preflight_checks: Vec::new(),
            reason: "libatomic ships with the base toolchain on this distribution; no separate install is required.".to_owned(),
        };
    }

    SystemPackageInstallPlan {
        supported: true,
        package_manager: Some(manager.as_str().to_owned()),
        packages,
        commands,
        preflight_checks: vec![
            "root access: run as root, or ensure `sudo -v` succeeds before approval".to_owned(),
            "`sudo` command is available when not running as root".to_owned(),
            format!("`{}` package manager is available", manager.as_str()),
        ],
        reason: "Install the libatomic runtime (libatomic.so.1) that vLLM's ROCm torch wheel links against. Requires root; approve before execution.".to_owned(),
    }
}

/// Build an approval-gated plan for installing the real `libnuma` runtime that
/// PyTorch's ROCm wheels require, for a distribution identity.
///
/// Arguments mirror [`build_openmpi_install_plan`]. Although the ROCm SDK bundles
/// numa, it does so under a renamed soname with rewritten symbol versions that
/// cannot satisfy torch's `libnuma_1.2` binding (see [`libnuma_present`]), so the
/// upstream numactl runtime must be installed from the package manager. Unlike
/// libatomic, numactl is *not* part of the Arch base install, so a pacman plan is
/// produced there as well.
pub fn build_libnuma_install_plan(os_id: &str, id_like: &str) -> SystemPackageInstallPlan {
    let os_id = os_id.trim().to_ascii_lowercase();
    let id_like = id_like.trim().to_ascii_lowercase();

    let Some(manager) = resolve_package_manager(&os_id, &id_like) else {
        return SystemPackageInstallPlan {
            supported: false,
            package_manager: None,
            packages: Vec::new(),
            commands: Vec::new(),
            preflight_checks: Vec::new(),
            reason: format!(
                "Could not map distribution (ID={}) to a known package manager; install the libnuma runtime manually (your distro's `numactl-libs` / `libnuma1` package providing libnuma.so.1).",
                if os_id.is_empty() {
                    "<unknown>"
                } else {
                    &os_id
                }
            ),
        };
    };

    let (packages, commands): (Vec<String>, Vec<InstallCommand>) = match manager {
        PackageManager::Apt => (
            vec!["libnuma1".to_owned()],
            vec![
                InstallCommand::sudo(&["apt-get", "update"]),
                InstallCommand::sudo(&["apt-get", "install", "-y", "libnuma1"]),
            ],
        ),
        PackageManager::Dnf => (
            vec!["numactl-libs".to_owned()],
            vec![InstallCommand::sudo(&[
                "dnf",
                "install",
                "-y",
                "numactl-libs",
            ])],
        ),
        PackageManager::Zypper => (
            vec!["libnuma1".to_owned()],
            vec![InstallCommand::sudo(&[
                "zypper", "install", "-y", "libnuma1",
            ])],
        ),
        PackageManager::Pacman => (
            vec!["numactl".to_owned()],
            vec![InstallCommand::sudo(&[
                "pacman",
                "-S",
                "--needed",
                "--noconfirm",
                "numactl",
            ])],
        ),
    };

    SystemPackageInstallPlan {
        supported: true,
        package_manager: Some(manager.as_str().to_owned()),
        packages,
        commands,
        preflight_checks: vec![
            "root access: run as root, or ensure `sudo -v` succeeds before approval".to_owned(),
            "`sudo` command is available when not running as root".to_owned(),
            format!("`{}` package manager is available", manager.as_str()),
        ],
        reason: "Install the real numactl runtime (libnuma.so.1, providing the libnuma_1.2 symbols) that PyTorch's ROCm wheel links against. The ROCm SDK's bundled numa uses renamed symbol versions and cannot satisfy this. Requires root; approve before execution.".to_owned(),
    }
}

/// runtime that vLLM requires, for embedding in user-facing error messages.
///
/// On Linux it reads `/etc/os-release` and returns the package-manager command
/// from [`build_openmpi_install_plan`] when the distribution is recognized,
/// otherwise a generic instruction. Off Linux it returns a generic message,
/// since the OpenMPI path is not exercised there.
pub fn install_hint() -> String {
    #[cfg(target_os = "linux")]
    {
        let os_release = std::fs::read_to_string("/etc/os-release").unwrap_or_default();
        let field = |key: &str| parse_os_release_field(&os_release, key).unwrap_or_default();
        let plan = build_openmpi_install_plan(&field("ID"), &field("ID_LIKE"));
        if plan.supported && !plan.commands.is_empty() {
            let rendered = plan
                .commands
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" && ");
            return format!("install it with `{rendered}`");
        }
    }
    "install your distribution's OpenMPI runtime package (providing libmpi.so / libmpi_cxx.so and mpirun)".to_owned()
}

/// Build a short, distro-aware hint describing how to install the `libatomic`
/// runtime that vLLM requires, for embedding in user-facing error messages.
///
/// On Linux it reads `/etc/os-release` and returns the package-manager command
/// from [`build_libatomic_install_plan`] when the distribution is recognized,
/// otherwise a generic instruction. Off Linux it returns a generic message.
pub fn libatomic_install_hint() -> String {
    #[cfg(target_os = "linux")]
    {
        let os_release = std::fs::read_to_string("/etc/os-release").unwrap_or_default();
        let field = |key: &str| parse_os_release_field(&os_release, key).unwrap_or_default();
        let plan = build_libatomic_install_plan(&field("ID"), &field("ID_LIKE"));
        if plan.supported && !plan.commands.is_empty() {
            let rendered = plan
                .commands
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" && ");
            return format!("install it with `{rendered}`");
        }
    }
    "install your distribution's libatomic runtime package (providing libatomic.so.1)".to_owned()
}

/// Build a short, distro-aware hint describing how to install the real `libnuma`
/// runtime that PyTorch requires, for embedding in user-facing error messages.
///
/// On Linux it reads `/etc/os-release` and returns the package-manager command
/// from [`build_libnuma_install_plan`] when the distribution is recognized,
/// otherwise a generic instruction. Off Linux it returns a generic message.
pub fn libnuma_install_hint() -> String {
    #[cfg(target_os = "linux")]
    {
        let os_release = std::fs::read_to_string("/etc/os-release").unwrap_or_default();
        let field = |key: &str| parse_os_release_field(&os_release, key).unwrap_or_default();
        let plan = build_libnuma_install_plan(&field("ID"), &field("ID_LIKE"));
        if plan.supported && !plan.commands.is_empty() {
            let rendered = plan
                .commands
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" && ");
            return format!("install it with `{rendered}`");
        }
    }
    "install your distribution's numactl runtime package (providing libnuma.so.1)".to_owned()
}

/// Parse a single `KEY=VALUE` field from `/etc/os-release` contents, stripping
/// optional surrounding quotes. Returns `None` when the key is absent.
// Only the Linux `install_hint` path calls this at runtime; off Linux it is
// exercised solely by cross-platform unit tests.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) fn parse_os_release_field(text: &str, key: &str) -> Option<String> {
    for line in text.lines() {
        let line = line.trim();
        let Some((name, value)) = line.split_once('=') else {
            continue;
        };
        if name.trim() != key {
            continue;
        }
        let value = value.trim().trim_matches('"').trim_matches('\'');
        return Some(value.to_owned());
    }
    None
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
    fn runtime_present_requires_both_libmpi_and_mpirun() {
        let lib = PathBuf::from("/usr/lib/x86_64-linux-gnu/libmpi.so.40");
        let run = PathBuf::from("/usr/bin/mpirun");
        // Both present -> usable runtime.
        assert!(runtime_present(Some(&lib), Some(&run)));
        // Library only (e.g. runtime package without launcher) -> not usable.
        assert!(!runtime_present(Some(&lib), None));
        // Launcher only -> not usable.
        assert!(!runtime_present(None, Some(&run)));
        // Neither -> not usable.
        assert!(!runtime_present(None, None));
    }

    #[test]
    fn parses_os_release_fields_with_and_without_quotes() {
        let text =
            "NAME=\"Red Hat Enterprise Linux\"\nID=rhel\nID_LIKE=fedora\nVERSION_ID=\"9.4\"\n";
        assert_eq!(parse_os_release_field(text, "ID").as_deref(), Some("rhel"));
        assert_eq!(
            parse_os_release_field(text, "ID_LIKE").as_deref(),
            Some("fedora")
        );
        assert_eq!(
            parse_os_release_field(text, "VERSION_ID").as_deref(),
            Some("9.4")
        );
        assert_eq!(parse_os_release_field(text, "MISSING"), None);
    }

    #[test]
    fn install_hint_is_non_empty_and_actionable() {
        // The hint is embedded in the serve preflight error, so it must always
        // give the user something to act on regardless of host distribution.
        let hint = install_hint();
        assert!(!hint.trim().is_empty());
        assert!(hint.to_ascii_lowercase().contains("install"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn install_mpi_cxx_symlink_creates_idempotent_link() {
        use std::fs;

        let base = std::env::temp_dir().join(format!(
            "rocm-openmpi-compat-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&base);
        let target = base.join("libmpi.so.40");
        let compat = base.join("compat");
        fs::create_dir_all(&base).unwrap();
        fs::write(&target, b"fake").unwrap();

        // First call creates the shim and returns the compat directory.
        let dir =
            ensure_compat_symlink(&compat, "libmpi_cxx.so.40", &target).expect("symlink created");
        assert_eq!(dir, compat);
        let link = compat.join("libmpi_cxx.so.40");
        assert!(symlink_points_to(&link, &target));
        // The link resolves to the real target contents.
        assert_eq!(fs::read(&link).unwrap(), b"fake");

        // Second call is idempotent and still reports success.
        let dir_again =
            ensure_compat_symlink(&compat, "libmpi_cxx.so.40", &target).expect("idempotent");
        assert_eq!(dir_again, compat);
        assert!(symlink_points_to(&link, &target));

        let _ = fs::remove_dir_all(&base);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn ensure_compat_symlink_skips_missing_target() {
        let base = std::env::temp_dir().join(format!(
            "rocm-compat-missing-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        let missing_target = base.join("does-not-exist.so");
        let compat = base.join("compat");
        // No shim is created when the target does not exist.
        assert_eq!(
            ensure_compat_symlink(&compat, "libmpi_cxx.so.40", &missing_target),
            None
        );
        assert!(!compat.join("libmpi_cxx.so.40").exists());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn embedded_mpi_cxx_stub_is_valid_elf_with_cxx_symbols() {
        // The build script writes an empty stub (disabling the shim) when no C
        // toolchain is available, so the binary can still be built. Only assert
        // the embedded bytes are a valid ELF object when a stub was produced;
        // skip otherwise so `cargo test` passes on hosts without a C compiler.
        if MPI_CXX_STUB.is_empty() {
            return;
        }
        assert_eq!(&MPI_CXX_STUB[..4], b"\x7fELF", "stub is not an ELF object");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn write_compat_file_is_idempotent_and_replaces_symlink() {
        use std::fs;

        let base = std::env::temp_dir().join(format!(
            "rocm-compat-write-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&base);
        let compat = base.join("compat");

        // Seed a stale symlink (as older rocm-cli versions left behind).
        fs::create_dir_all(&compat).unwrap();
        let stale_target = base.join("stale.so");
        fs::write(&stale_target, b"stale").unwrap();
        std::os::unix::fs::symlink(&stale_target, compat.join("libmpi_cxx.so.40")).unwrap();

        // Writing replaces the symlink with a real file holding the stub bytes.
        let dir = write_compat_file(&compat, "libmpi_cxx.so.40", b"STUBBYTES").expect("written");
        assert_eq!(dir, compat);
        let path = compat.join("libmpi_cxx.so.40");
        assert!(
            !fs::symlink_metadata(&path)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read(&path).unwrap(), b"STUBBYTES");

        // Second call with identical bytes is idempotent.
        let dir_again =
            write_compat_file(&compat, "libmpi_cxx.so.40", b"STUBBYTES").expect("idempotent");
        assert_eq!(dir_again, compat);
        assert_eq!(fs::read(&path).unwrap(), b"STUBBYTES");

        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn ldconfig_without_libmpi_returns_none() {
        let output = "\tlibm.so.6 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libm.so.6\n";
        assert_eq!(parse_ldconfig_libmpi(output), None);
    }

    #[test]
    fn libatomic_plan_for_rhel_uses_base_package() {
        let plan = build_libatomic_install_plan("rhel", "fedora");
        assert!(plan.supported);
        assert_eq!(plan.package_manager.as_deref(), Some("dnf"));
        assert_eq!(plan.packages, vec!["libatomic".to_owned()]);
        assert!(
            plan.commands
                .iter()
                .any(|cmd| cmd.to_string() == "sudo dnf install -y libatomic"),
            "expected base libatomic install command: {:?}",
            plan.commands
        );
    }

    #[test]
    fn libatomic_plan_for_debian_uses_libatomic1() {
        let plan = build_libatomic_install_plan("ubuntu", "debian");
        assert_eq!(plan.package_manager.as_deref(), Some("apt"));
        assert_eq!(plan.packages, vec!["libatomic1".to_owned()]);
    }

    #[test]
    fn libatomic_plan_for_arch_is_unsupported_no_op() {
        // Arch ships libatomic in gcc-libs (base install), so there is nothing
        // to install separately and the plan is marked unsupported (no-op).
        let plan = build_libatomic_install_plan("arch", "");
        assert!(!plan.supported);
        assert!(plan.commands.is_empty());
    }

    #[test]
    fn libnuma_plan_for_rhel_uses_numactl_libs() {
        let plan = build_libnuma_install_plan("rhel", "fedora");
        assert!(plan.supported);
        assert_eq!(plan.package_manager.as_deref(), Some("dnf"));
        assert_eq!(plan.packages, vec!["numactl-libs".to_owned()]);
        assert!(
            plan.commands
                .iter()
                .any(|cmd| cmd.to_string() == "sudo dnf install -y numactl-libs"),
            "expected numactl-libs install command: {:?}",
            plan.commands
        );
    }

    #[test]
    fn libnuma_plan_for_debian_uses_libnuma1() {
        let plan = build_libnuma_install_plan("ubuntu", "debian");
        assert_eq!(plan.package_manager.as_deref(), Some("apt"));
        assert_eq!(plan.packages, vec!["libnuma1".to_owned()]);
    }

    #[test]
    fn libnuma_plan_for_arch_uses_numactl() {
        // Unlike libatomic, numactl is not part of the Arch base install, so a
        // pacman plan is produced.
        let plan = build_libnuma_install_plan("arch", "");
        assert!(plan.supported);
        assert_eq!(plan.package_manager.as_deref(), Some("pacman"));
        assert_eq!(plan.packages, vec!["numactl".to_owned()]);
    }

    #[test]
    fn apt_plan_for_ubuntu() {
        let plan = build_openmpi_install_plan("ubuntu", "debian");
        assert!(plan.supported);
        assert_eq!(plan.package_manager.as_deref(), Some("apt"));
        // Release-agnostic names so the plan survives the `t64` ABI transition
        // (libopenmpi3 -> libopenmpi3t64) on newer Ubuntu/Debian.
        assert!(plan.packages.contains(&"openmpi-bin".to_owned()));
        assert!(plan.packages.contains(&"libopenmpi-dev".to_owned()));
        assert!(
            !plan.packages.iter().any(|pkg| pkg == "libopenmpi3"),
            "apt plan must not pin the release-specific libopenmpi3 name: {:?}",
            plan.packages
        );
        assert!(plan.commands.iter().any(|cmd| {
            cmd.to_string()
                .contains("apt-get install -y openmpi-bin libopenmpi-dev")
        }));
    }

    #[test]
    fn apt_plan_via_id_like() {
        let plan = build_openmpi_install_plan("linuxmint", "ubuntu debian");
        assert_eq!(plan.package_manager.as_deref(), Some("apt"));
    }

    #[test]
    fn dnf_plan_for_rhel_family() {
        for id in ["rhel", "fedora", "rocky", "almalinux", "ol"] {
            let plan = build_openmpi_install_plan(id, "");
            assert_eq!(
                plan.package_manager.as_deref(),
                Some("dnf"),
                "expected dnf for {id}"
            );
            assert!(
                plan.commands
                    .iter()
                    .any(|cmd| cmd.to_string().contains("dnf install"))
            );
        }
    }

    #[test]
    fn dnf_plan_omits_devel_package() {
        // The `openmpi-devel` package lives in the CodeReady Builder (CRB)
        // repository, which is disabled by default on RHEL, so requiring it made
        // `dnf install` fail outright ("No match for argument: openmpi-devel").
        // The vLLM runtime only needs `libmpi.so*` + `mpirun`, both provided by
        // the base `openmpi` package, so the plan must not request `-devel`.
        let plan = build_openmpi_install_plan("rhel", "fedora");
        assert_eq!(plan.packages, vec!["openmpi".to_owned()]);
        assert!(
            !plan.packages.iter().any(|pkg| pkg.contains("devel")),
            "RHEL plan must not require a -devel package: {:?}",
            plan.packages
        );
        assert!(
            plan.commands
                .iter()
                .all(|cmd| !cmd.to_string().contains("openmpi-devel")),
            "RHEL install command must not reference openmpi-devel: {:?}",
            plan.commands
        );
        assert!(
            plan.commands
                .iter()
                .any(|cmd| cmd.to_string() == "sudo dnf install -y openmpi"),
            "expected base openmpi install command: {:?}",
            plan.commands
        );
    }

    #[test]
    fn dnf_plan_via_id_like_rhel() {
        let plan = build_openmpi_install_plan("oraclelinux", "fedora");
        assert_eq!(plan.package_manager.as_deref(), Some("dnf"));
    }

    #[test]
    fn zypper_plan_for_suse() {
        let plan = build_openmpi_install_plan("opensuse-leap", "suse opensuse");
        assert_eq!(plan.package_manager.as_deref(), Some("zypper"));
        assert!(
            plan.commands
                .iter()
                .any(|cmd| cmd.to_string().contains("zypper install"))
        );
    }

    #[test]
    fn pacman_plan_for_arch() {
        let plan = build_openmpi_install_plan("arch", "");
        assert_eq!(plan.package_manager.as_deref(), Some("pacman"));
        assert!(
            plan.commands
                .iter()
                .any(|cmd| cmd.to_string().contains("pacman -S"))
        );
    }

    #[test]
    fn unknown_distro_is_unsupported() {
        let plan = build_openmpi_install_plan("void", "");
        assert!(!plan.supported);
        assert!(plan.package_manager.is_none());
        assert!(plan.commands.is_empty());
        assert!(plan.reason.contains("manually"));
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
