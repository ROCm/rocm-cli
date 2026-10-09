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
//! - the `/etc/os-release` behind the distro name `rocm examine` reports;
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
//! package-manager plan reads the real `/etc/os-release` — the install hints in
//! `openmpi.rs`, and the driver, OpenMPI and runtime-library installs in the
//! `rocm` binary — because the commands it builds run against the real
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
}
