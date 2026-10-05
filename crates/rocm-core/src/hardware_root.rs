// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Where the hardware probes read the host from.
//!
//! The probes that decide what GPU, driver and platform the CLI is running on —
//! device nodes under `/dev`, the KFD topology and DRM cards in `sysfs`,
//! `/proc/version`, `/proc/cpuinfo`, `/etc/os-release`, the WSL plumbing under
//! `/usr/lib/wsl`, the container markers (`/.dockerenv`, `/run/.containerenv`,
//! `/proc/1/cgroup`) — resolve their path through [`host_path`] instead of
//! naming the absolute path directly.
//!
//! Deliberately NOT routed: anything a real process acts on rather than reads
//! to describe the machine. ROCm install discovery under `/opt` and
//! `/usr/local` reports paths the CLI prints and writes into shell rc files, and
//! the `/usr/lib/wsl/lib` loader entry goes into a real engine's
//! `LD_LIBRARY_PATH`; re-rooting either would put the simulated directory in
//! front of a user or a real loader. Programs the probes execute, such as
//! `ldconfig`, stay on the real host too: a simulated root holds no binary to
//! run. Process liveness under `/proc/<pid>` and `/dev/shm` sizing are about
//! this process's environment, not the hardware. The package-manager install
//! hints in `openmpi.rs` still read the real `/etc/os-release`.
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
/// the discovered ROCm install paths `rocm_relative_file_exists` probes under.
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
