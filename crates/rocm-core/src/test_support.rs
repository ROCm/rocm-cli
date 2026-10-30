// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Shared fixtures for building a throwaway [`AppPaths`] under a per-test,
//! per-process directory.
//!
//! Used from every module split out of this crate's former `lib.rs` god file
//! so the naming/uniqueness scheme for test artifact directories stays in one
//! place rather than drifting across copies.

use crate::{AppPaths, ManagedServiceRecord, unix_time_millis};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// The volume serial number paired with the 64-bit file index: this
/// platform's file identity, the way an inode is on Unix. Shared by every
/// test that tells a publish-by-rename apart from a rewrite in place.
///
/// Read through `GetFileInformationByHandle` rather than
/// `std::os::windows::fs::MetadataExt`, whose `volume_serial_number` and
/// `file_index` accessors are unstable behind the `windows_by_handle`
/// feature gate and so cannot be called on the pinned stable toolchain.
#[cfg(windows)]
#[allow(unsafe_code)] // Win32 FFI
pub(crate) fn windows_file_identity(path: &std::path::Path) -> anyhow::Result<(u32, u64)> {
    use anyhow::{Context, bail};
    use std::fs;
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::Storage::FileSystem::{
        BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
    };

    let file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    // All-POD `repr(C)` record, fully overwritten by the call below on
    // success and never read on failure.
    let mut info: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
    let handle = file.as_raw_handle() as HANDLE;
    if unsafe { GetFileInformationByHandle(handle, &raw mut info) } == 0 {
        bail!(
            "failed to read the file identity of {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        );
    }
    Ok((
        info.dwVolumeSerialNumber,
        (u64::from(info.nFileIndexHigh) << 32) | u64::from(info.nFileIndexLow),
    ))
}

pub(crate) fn temp_app_paths(name: &str) -> (PathBuf, AppPaths) {
    let root = workspace_test_artifact_dir().join(format!("rocm-core-{name}-{}", unique_suffix()));
    let paths = AppPaths {
        config_dir: root.join("config"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
    };
    (root, paths)
}

/// A name suffix no other call in this test process returns.
///
/// The pid separates processes and the timestamp keeps names readable across
/// runs, but neither separates two tests in one process: `cargo test` runs
/// them on parallel threads, and two that start in the same millisecond with
/// the same label got the same root, so one test's cleanup deleted the other's
/// files. The counter is per-process and monotonic, so no two calls here can
/// agree.
pub(crate) fn unique_suffix() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}-{}-{}",
        std::process::id(),
        unix_time_millis(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

pub(crate) fn workspace_test_artifact_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(".rocm-work")
        .join("tests")
        .join("core")
}

pub(crate) fn probe_test_record(port: u16) -> ManagedServiceRecord {
    let root = PathBuf::from("/tmp/rocm-inference-probe-test");
    let paths = AppPaths {
        config_dir: root.join("config"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
    };
    ManagedServiceRecord::new(
        &paths,
        "svc-probe",
        "vllm",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-0.6B",
        "127.0.0.1",
        port,
        "serve",
        4242,
        None,
        None,
        None,
    )
}

/// Call `make_root` from many threads at once, remove what it made, and
/// assert no two calls returned the same root.
///
/// Concurrent rather than sequential, because that is how tests call these
/// helpers: a run of calls made one after another can let the clock tick
/// between them and pass by luck, while threads released together all read the
/// same millisecond, so a root keyed on pid and time alone repeats.
pub(crate) fn assert_each_call_gets_its_own_root(make_root: impl Fn() -> PathBuf + Sync) {
    const CALLERS: usize = 16;
    let start = std::sync::Barrier::new(CALLERS);
    let roots: Vec<PathBuf> = std::thread::scope(|scope| {
        let callers: Vec<_> = (0..CALLERS)
            .map(|_| {
                scope.spawn(|| {
                    start.wait();
                    make_root()
                })
            })
            .collect();
        callers
            .into_iter()
            .map(|caller| caller.join().expect("a caller panicked"))
            .collect()
    });
    for root in &roots {
        let _ = std::fs::remove_dir_all(root);
    }
    let distinct: std::collections::HashSet<_> = roots.iter().collect();
    assert_eq!(
        distinct.len(),
        CALLERS,
        "two calls with one label must not share a root"
    );
}

/// The temp name the retired `write_atomic` picked for `destination` at
/// `millis`: the full file name plus `.tmp-<millis>-<pid>`, with this
/// process's pid. Unlike the names `atomic_write` picks, it was fixed by the
/// clock and the pid alone, so a test can occupy it ahead of time.
pub(crate) fn retired_temp_path(destination: &std::path::Path, millis: u128) -> PathBuf {
    let mut name = destination
        .file_name()
        .expect("a destination has a file name")
        .to_os_string();
    name.push(format!(".tmp-{millis}-{}", std::process::id()));
    destination.with_file_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_app_paths_gives_each_call_its_own_root_even_for_one_label() {
        assert_each_call_gets_its_own_root(|| temp_app_paths("same-label").0);
    }
}
