// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Where the hardware probes read the host from.
//!
//! The probes that decide what GPU, driver and platform the CLI is running on
//! resolve their path through [`host_path`] instead of naming the absolute path
//! directly. That is every one of these reads, and only these:
//!
//! - device nodes: `/dev/kfd`, the render nodes under `/dev/dri`, `/dev/dxg`;
//! - `sysfs`: the KFD topology under `/sys/class/kfd/kfd/topology/nodes`, the
//!   DRM cards under `/sys/class/drm`, `/sys/module/amdgpu` and
//!   `/sys/module/amdgpu/version`;
//! - `procfs`: `/proc/version`, `/proc/cpuinfo`, `/proc/meminfo`,
//!   `/proc/cmdline`, `/proc/modules` (the fallback when `lsmod` cannot run)
//!   and `/proc/1/cgroup`;
//! - the `/etc/os-release` behind the distro fields and name `rocm examine`
//!   reports, falling back to `/usr/lib/os-release` when the former is absent,
//!   per `os-release(5)`;
//! - the modprobe configuration directories `/etc/modprobe.d`,
//!   `/usr/lib/modprobe.d` and `/run/modprobe.d`;
//! - the WSL plumbing: the `/usr/lib/wsl/lib` directory check and
//!   `/usr/lib/wsl/lib/libdxcore.so`;
//! - the WSL ROCDXG check: `lib/librocdxg.so` and `share/rocdxg/dids.conf`,
//!   looked for under `/opt/rocm` and under each discovered ROCm install;
//! - the container markers `/.dockerenv` and `/run/.containerenv`.
//!
//! Deliberately NOT routed: anything a real process acts on rather than reads
//! to describe the machine. ROCm install discovery itself — which installs
//! exist under `/opt` and `/usr/local`, their paths and versions — reads the
//! real host, because the CLI prints those paths and writes them into shell rc
//! files; only the ROCDXG existence check above looks under them through the
//! root. The `/usr/lib/wsl/lib` loader entry goes into a real engine's
//! `LD_LIBRARY_PATH`; re-rooting it would put the simulated directory in front
//! of a real loader. Programs the probes execute, such as `ldconfig`, stay on
//! the real host too: a simulated root holds no binary to run. Every
//! package-manager plan reads the real `/etc/os-release` (with the same
//! `/usr/lib/os-release` fallback) — the OpenMPI, libatomic and libnuma install
//! hints in `openmpi.rs`, and the driver, OpenMPI and runtime-library installs
//! in the `rocm` binary — because the commands it builds run against the real
//! machine. Process liveness under `/proc/<pid>` and `/dev/shm` sizing are
//! about this process's environment, not the hardware.
//!
//! Routed, but with the real device keeping a veto: the `rocm dash` `amd-smi`
//! pre-flight. It decides whether a real `amd-smi` process may start, so
//! `apps/rocm/src/dash.rs` passes `host_path("/dev/kfd")` to the dash daemon
//! and the collector requires both that path and the real `/dev/kfd` to be
//! readable. A simulated root can hide the GPU from the dashboard but cannot
//! start `amd-smi` on a machine without one. The WSL reachability verdict
//! still substitutes for the whole device check, as it always has, so the veto
//! does not cover it.
//!
//! In a normal build [`host_path`] is the identity: the path is returned as
//! written and the probes read the real host. Only a build compiled with the
//! `e2e-test-hooks` Cargo feature honours [`TEST_HOST_ROOT_ENV`], which re-roots
//! those absolute paths under a directory the E2E suite populated with a
//! simulated host — a KFD topology for a GPU that is not there, a `/dev/dxg`
//! that makes a bare-metal runner read as WSL. Without the feature the
//! override logic does not exist, so a stray environment variable can never
//! change what a release build believes about its hardware.
//!
//! Callers keep reporting the logical path (`/dev/kfd`), never the re-rooted
//! one: the fake root decides what the probes *find*, not what the CLI *says*.

use std::path::{Path, PathBuf};

/// Directory that stands in for `/` for every hardware probe, honoured only in
/// builds with the `e2e-test-hooks` feature.
pub const TEST_HOST_ROOT_ENV: &str = "ROCM_CLI_TEST_HOST_ROOT";

/// The path a hardware probe should read for the absolute host path `path`.
///
/// The identity unless this is an `e2e-test-hooks` build with
/// [`TEST_HOST_ROOT_ENV`] set; see the module docs.
#[must_use]
pub fn host_path(path: impl AsRef<Path>) -> PathBuf {
    host_path_under(hardware_root().as_deref(), path.as_ref())
}

/// [`host_path`], fixed to `&str`.
///
/// [`read_os_release`]'s `resolve` parameter needs a concrete
/// `Fn(&str) -> PathBuf`, which a bare reference to the generic `host_path`
/// cannot satisfy (the borrow checker rejects it as "not general enough"
/// over the resolved lifetime), so every probe that must honour a simulated
/// root passes this instead of `|p| host_path(p)`.
#[must_use]
pub fn host_path_for_str(path: &str) -> PathBuf {
    host_path(path)
}

/// [`read_os_release`]'s `resolve` parameter for a caller that must always
/// read the real machine.
///
/// Never routed through [`host_path`], because it builds a package-manager
/// plan or install hint that runs against the real host, so a simulated
/// host root must not choose its distro (see each caller's own doc comment
/// for why). A bare `PathBuf::from` is rejected the same way a bare
/// `host_path` is (see [`host_path_for_str`]), so every such caller passes
/// this instead of `|p| PathBuf::from(p)`.
#[must_use]
pub fn identity_path_for_str(path: &str) -> PathBuf {
    PathBuf::from(path)
}

#[cfg(feature = "e2e-test-hooks")]
fn hardware_root() -> Option<PathBuf> {
    std::env::var_os(TEST_HOST_ROOT_ENV)
        .filter(|root| !root.is_empty())
        .map(PathBuf::from)
}

#[cfg(not(feature = "e2e-test-hooks"))]
const fn hardware_root() -> Option<PathBuf> {
    None
}

/// [`host_path`] with the root supplied by the caller, so the re-rooting can be
/// tested without touching the process environment.
///
/// A relative `path` is returned unchanged: only absolute host paths name the
/// machine, and joining a relative one under the root would invent a location
/// nobody asked for.
///
/// `path` must not contain `..`: [`Path::join`] does not normalise, so a `..`
/// after the leading `/` would walk back out of the root. This is a documented
/// precondition rather than a check: callers pass fixed literals, apart from
/// the ROCDXG check in `rocm_relative_file_exists`, which re-roots each
/// discovered ROCm install path (a `$ROCM_PATH` is taken verbatim there, so a
/// `..` in it escapes the root; a *relative* `$ROCM_PATH` is a different case:
/// `strip_prefix("/")` fails, it is returned unchanged by the catch-all arm,
/// and the simulated root silently does not apply — that ROCDXG lookup reads
/// relative to the working directory instead).
fn host_path_under(root: Option<&Path>, path: &Path) -> PathBuf {
    match (root, path.strip_prefix("/")) {
        (Some(root), Ok(relative)) => root.join(relative),
        _ => path.to_path_buf(),
    }
}

/// What reading `/etc/os-release` (falling back to `/usr/lib/os-release` per
/// `os-release(5)`) produced.
///
/// Kept as three states, not collapsed to `Option<String>`, so a caller can
/// name the file it actually read in a parse-error message (`Found`'s `path`)
/// and can tell "nothing to report" (`Absent`, the documented fallback
/// contract: a missing file is not a failure) apart from "something is
/// wrong" (`Unreadable`: a candidate exists but could not be read). Before
/// this distinction existed, every caller folded both into the same empty
/// text and reported a host with a merely-unreadable file as an unsupported
/// distribution.
#[derive(Debug)]
pub enum OsRelease {
    /// Read from `path` (`/etc/os-release` or `/usr/lib/os-release`).
    Found { path: &'static str, text: String },
    /// Neither candidate exists.
    Absent,
    /// `path` exists but could not be read, for a reason other than absence
    /// (permission denied, a directory in its place, ...).
    Unreadable {
        path: &'static str,
        error: std::io::Error,
    },
}

impl OsRelease {
    /// The text to parse: the file's contents when one was read, empty for
    /// [`Self::Absent`] or [`Self::Unreadable`] — an absent file declares
    /// nothing, and an unreadable one must be reported as itself by the
    /// caller rather than parsed as an empty, unknown distribution.
    #[must_use]
    pub fn text(&self) -> &str {
        match self {
            Self::Found { text, .. } => text,
            Self::Absent | Self::Unreadable { .. } => "",
        }
    }

    /// The path a parse-error message should name: the one actually read, or
    /// `/etc/os-release` for [`Self::Absent`] — which never reaches such a
    /// message, since empty text never fails to parse.
    #[must_use]
    pub const fn path(&self) -> &'static str {
        match self {
            Self::Found { path, .. } | Self::Unreadable { path, .. } => path,
            Self::Absent => "/etc/os-release",
        }
    }

    /// For tests: a [`Self::Found`] at the default `/etc/os-release` path.
    #[cfg(test)]
    pub(crate) fn found(text: impl Into<String>) -> Self {
        Self::Found {
            path: "/etc/os-release",
            text: text.into(),
        }
    }
}

/// The shared opening every [`OsRelease::Unreadable`] message builds on.
///
/// Names the candidate path and the I/O error that made it unreadable. Each
/// call site appends its own clause describing what that means for it (no
/// commands planned, a probe failure, an unreported distro, ...), so only
/// this shared part can drift if changed in one place and not another.
#[must_use]
pub fn unreadable_prefix(path: &str, error: &std::io::Error) -> String {
    format!("{path} could not be read: {error}")
}

/// Reads `/etc/os-release`, falling back to `/usr/lib/os-release` per
/// `os-release(5)` when the former is absent.
///
/// `resolve` maps each candidate to where it should actually be read from —
/// [`host_path`] for a probe that must honour a simulated test root, or the
/// identity for one that must always read the real machine (see the module
/// doc above).
///
/// Falls back only when `/etc/os-release` does not exist, never for another
/// read error (permission denied, a directory in its place, ...): `os-release(5)`
/// defines the fallback as existence-based, and treating any error the same
/// way would silently build commands for the wrong distro from a stale
/// `/usr/lib/os-release` on a host where `/etc/os-release` exists but is
/// merely unreadable.
pub fn read_os_release(resolve: impl Fn(&str) -> PathBuf) -> OsRelease {
    match std::fs::read_to_string(resolve("/etc/os-release")) {
        Ok(text) => OsRelease::Found {
            path: "/etc/os-release",
            text,
        },
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            match std::fs::read_to_string(resolve("/usr/lib/os-release")) {
                Ok(text) => OsRelease::Found {
                    path: "/usr/lib/os-release",
                    text,
                },
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => OsRelease::Absent,
                Err(error) => OsRelease::Unreadable {
                    path: "/usr/lib/os-release",
                    error,
                },
            }
        }
        Err(error) => OsRelease::Unreadable {
            path: "/etc/os-release",
            error,
        },
    }
}

/// Serialises every test that sets [`TEST_HOST_ROOT_ENV`], including the
/// downstream probe test in `examine.rs`, so two of them never see each other's
/// simulated root.
#[cfg(test)]
pub(crate) static HOST_ROOT_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn without_a_root_the_host_path_is_unchanged() {
        assert_eq!(
            host_path_under(None, Path::new("/dev/kfd")),
            PathBuf::from("/dev/kfd")
        );
    }

    #[test]
    fn a_root_re_roots_an_absolute_host_path() {
        assert_eq!(
            host_path_under(
                Some(Path::new("/tmp/fake-host")),
                Path::new("/sys/class/kfd/kfd/topology/nodes")
            ),
            PathBuf::from("/tmp/fake-host/sys/class/kfd/kfd/topology/nodes")
        );
    }

    #[test]
    fn a_root_leaves_a_relative_path_alone() {
        assert_eq!(
            host_path_under(Some(Path::new("/tmp/fake-host")), Path::new("lib/x.so")),
            PathBuf::from("lib/x.so")
        );
    }

    /// The release build must not honour the variable at all — this is the
    /// property that makes it safe to ship the hook in the product's code. The
    /// variable is really set, so a build that started reading it fails here.
    #[cfg(not(feature = "e2e-test-hooks"))]
    #[test]
    fn a_build_without_the_hook_ignores_the_variable() {
        let _guard = HOST_ROOT_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _root =
            crate::test_env::RestoredEnvVar::set(TEST_HOST_ROOT_ENV, Path::new("/tmp/fake-host"));
        assert_eq!(host_path("/dev/kfd"), PathBuf::from("/dev/kfd"));
    }

    /// The hook build re-roots under the variable, and treats an empty value as
    /// unset rather than as the current directory.
    #[cfg(feature = "e2e-test-hooks")]
    #[test]
    fn a_hook_build_honours_the_variable_and_ignores_an_empty_one() {
        let _guard = HOST_ROOT_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        {
            let _root = crate::test_env::RestoredEnvVar::set(
                TEST_HOST_ROOT_ENV,
                Path::new("/tmp/fake-host"),
            );
            assert_eq!(
                host_path("/dev/kfd"),
                PathBuf::from("/tmp/fake-host/dev/kfd")
            );
        }
        let _empty = crate::test_env::RestoredEnvVar::set(TEST_HOST_ROOT_ENV, Path::new(""));
        assert_eq!(host_path("/dev/kfd"), PathBuf::from("/dev/kfd"));
    }

    /// Maps `/etc/os-release` to `primary` and `/usr/lib/os-release` to
    /// `fallback` under `dir`, so [`read_os_release`]'s fallback can be tested
    /// without touching the real host's files.
    fn candidate(dir: &Path, name: &str) -> PathBuf {
        match name {
            "/etc/os-release" => dir.join("primary"),
            "/usr/lib/os-release" => dir.join("fallback"),
            other => panic!("unexpected os-release candidate: {other}"),
        }
    }

    /// Extracts `(path, text)` from a [`OsRelease::Found`], panicking on any
    /// other variant — the shape every "a candidate was read" assertion below
    /// wants, written once so each test states only what differs.
    fn found(read: OsRelease) -> (&'static str, String) {
        match read {
            OsRelease::Found { path, text } => (path, text),
            other => panic!("expected OsRelease::Found, got {other:?}"),
        }
    }

    #[test]
    fn read_os_release_prefers_the_primary_candidate() {
        let dir = tempfile::tempdir().expect("create temp dir");
        std::fs::write(dir.path().join("primary"), "ID=primary\n").expect("write primary");
        std::fs::write(dir.path().join("fallback"), "ID=fallback\n").expect("write fallback");
        assert_eq!(
            found(read_os_release(|name| candidate(dir.path(), name))),
            ("/etc/os-release", "ID=primary\n".to_owned())
        );
    }

    #[test]
    fn read_os_release_falls_back_when_the_primary_is_absent() {
        let dir = tempfile::tempdir().expect("create temp dir");
        std::fs::write(dir.path().join("fallback"), "ID=fallback\n").expect("write fallback");
        assert_eq!(
            found(read_os_release(|name| candidate(dir.path(), name))),
            ("/usr/lib/os-release", "ID=fallback\n".to_owned())
        );
    }

    #[test]
    fn read_os_release_is_absent_when_neither_candidate_exists() {
        let dir = tempfile::tempdir().expect("create temp dir");
        assert!(matches!(
            read_os_release(|name| candidate(dir.path(), name)),
            OsRelease::Absent
        ));
    }

    /// A primary that exists but fails to read for a reason other than
    /// absence (here, a directory in its place) must not fall back: that
    /// would silently build commands for the wrong distro from a fallback
    /// file that may not describe this host at all. A directory read as a
    /// file is a portable, root-free way to get a non-`NotFound` `io::Error`.
    /// It must also be reported as itself, not folded into `Absent`: an
    /// unreadable file is not "nothing to report".
    #[test]
    fn read_os_release_does_not_fall_back_when_the_primary_errors_for_another_reason() {
        let dir = tempfile::tempdir().expect("create temp dir");
        std::fs::create_dir(dir.path().join("primary")).expect("create primary as a directory");
        std::fs::write(dir.path().join("fallback"), "ID=fallback\n").expect("write fallback");
        match read_os_release(|name| candidate(dir.path(), name)) {
            OsRelease::Unreadable { path, .. } => assert_eq!(path, "/etc/os-release"),
            other => panic!("expected OsRelease::Unreadable, got {other:?}"),
        }
    }

    /// The same non-absence distinction, but on the fallback candidate: a
    /// primary that is genuinely absent and a fallback that exists but
    /// cannot be read must not read as `Absent` either.
    #[test]
    fn read_os_release_reports_the_fallback_as_unreadable_when_it_errors_for_another_reason() {
        let dir = tempfile::tempdir().expect("create temp dir");
        std::fs::create_dir(dir.path().join("fallback")).expect("create fallback as a directory");
        match read_os_release(|name| candidate(dir.path(), name)) {
            OsRelease::Unreadable { path, .. } => assert_eq!(path, "/usr/lib/os-release"),
            other => panic!("expected OsRelease::Unreadable, got {other:?}"),
        }
    }
}
