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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_app_paths_gives_each_call_its_own_root_even_for_one_label() {
        assert_each_call_gets_its_own_root(|| temp_app_paths("same-label").0);
    }
}
