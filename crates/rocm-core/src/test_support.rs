// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Shared fixtures for building a throwaway [`AppPaths`] under a per-test,
//! per-process directory, and for seeding the automation audit store that both
//! `automation.rs`'s and `jsonl_store.rs`'s tests drive.
//!
//! Used from every module split out of this crate's former `lib.rs` god file
//! so the naming/uniqueness scheme for test artifact directories stays in one
//! place rather than drifting across copies.

use crate::{AppPaths, AuditEventRecord, ManagedServiceRecord, unix_time_millis};
use anyhow::Result;
use std::fs;
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

pub(crate) fn audit_record(tag: &str) -> AuditEventRecord {
    AuditEventRecord {
        at_unix_ms: 1,
        source: "test".to_owned(),
        category: "concurrency".to_owned(),
        actor: "writer".to_owned(),
        level: "info".to_owned(),
        action: "append".to_owned(),
        message: tag.to_owned(),
        watcher_id: None,
        service_id: None,
    }
}

/// Write `contents` as the audit store, creating its directory.
pub(crate) fn seed_audit_store(paths: &AppPaths, contents: &str) -> Result<()> {
    paths.ensure()?;
    fs::write(paths.audit_events_path(), contents)?;
    Ok(())
}

pub(crate) fn audit_line(tag: &str) -> String {
    serde_json::to_string(&audit_record(tag)).expect("serialize audit record")
}

pub(crate) fn audit_messages(records: &[AuditEventRecord]) -> Vec<&str> {
    records.iter().map(|r| r.message.as_str()).collect()
}

/// Case count for a property test that sets its own count: `PROPTEST_CASES`
/// (proptest's own knob, which an explicit count would otherwise override)
/// when it is set and above `default`, otherwise `default`.
///
/// Never below `default`: the counts are chosen so a property catches the
/// defects it targets with near certainty, and some of those are hit by a
/// small share of cases, so a lower count would pass with them present. `0`
/// would pass vacuously. A value that is ignored, because it is lower or does
/// not parse, is reported on stderr. `archive_props::run_property` applies the
/// same rule with its own copy, since other crates include that file by
/// `#[path]`.
pub(crate) fn proptest_cases(default: u32) -> u32 {
    let requested = std::env::var("PROPTEST_CASES").ok();
    let cases = raised_case_count(requested.as_deref(), default);
    if let Some(requested) = requested
        && requested.parse::<u32>().ok() != Some(cases)
    {
        eprintln!(
            "PROPTEST_CASES={requested} ignored: this property runs at least {default} cases"
        );
    }
    cases
}

/// The pure half of [`proptest_cases`]: `requested` raises `default` and never
/// lowers it.
fn raised_case_count(requested: Option<&str>, default: u32) -> u32 {
    requested
        .and_then(|value| value.parse().ok())
        .map_or(default, |requested: u32| requested.max(default))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_app_paths_gives_each_call_its_own_root_even_for_one_label() {
        assert_each_call_gets_its_own_root(|| temp_app_paths("same-label").0);
    }

    #[test]
    fn proptest_cases_are_raised_but_never_lowered() {
        assert_eq!(raised_case_count(None, 256), 256);
        assert_eq!(raised_case_count(Some("4096"), 256), 4096);
        assert_eq!(raised_case_count(Some("16"), 256), 256);
        assert_eq!(raised_case_count(Some("0"), 256), 256);
        assert_eq!(raised_case_count(Some("lots"), 256), 256);
    }
}
