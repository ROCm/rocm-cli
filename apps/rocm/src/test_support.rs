// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Shared test-only fixtures for `rocm`'s module test suites.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// A name suffix no other call in this test process returns.
///
/// Test scratch roots are named `<label>-<suffix>`. The pid separates
/// processes and the timestamp keeps names readable across runs, but neither
/// separates two tests in one process: `cargo test` runs them on parallel
/// threads, and two that start in the same millisecond with the same label got
/// the same root, so one test's cleanup deleted the other's files. The counter
/// is per-process and monotonic, so no two calls here can agree.
pub(crate) fn unique_suffix() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}-{}-{}",
        std::process::id(),
        rocm_core::unix_time_millis(),
        NEXT.fetch_add(1, Ordering::Relaxed)
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

mod tests {
    use super::*;

    #[test]
    fn unique_suffix_never_repeats_within_a_process() {
        let suffixes: std::collections::HashSet<String> =
            (0..64).map(|_| unique_suffix()).collect();
        assert_eq!(suffixes.len(), 64, "two calls must never share a suffix");
    }
}
