// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Shared test-only fixtures for `rocm-engine-lemonade`'s module test suites.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// A fresh, empty scratch directory under the crate's `target/`, named
/// `lemonade-fs-test-<tag>-<pid>-<millis>-<counter>`.
///
/// The base is `CARGO_MANIFEST_DIR`, a compile-time constant, so the path never
/// derives from a runtime environment read. The tag keeps the name readable;
/// it does not keep it unique. A tag reused by two tests running on parallel
/// threads, or one test running in two `cargo test` processes against the same
/// checkout, would otherwise share the directory, and each call wipes it
/// first, so one run would delete the other's files. The pid separates
/// processes and the per-process counter separates calls within one, so no two
/// calls agree whatever the tag.
pub(crate) fn scratch_dir(tag: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!(
            "lemonade-fs-test-{tag}-{}-{}-{}",
            std::process::id(),
            rocm_core::unix_time_millis(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

mod tests {
    use super::*;

    /// Concurrent rather than sequential, because that is how tests call this
    /// helper: calls made one after another can let the clock tick between
    /// them and pass by luck, while threads released together all read the
    /// same millisecond, so a name keyed on pid and time alone repeats.
    #[test]
    fn scratch_dir_gives_each_call_its_own_directory_even_for_one_tag() {
        const CALLERS: usize = 16;
        let start = std::sync::Barrier::new(CALLERS);
        let dirs: Vec<PathBuf> = std::thread::scope(|scope| {
            let callers: Vec<_> = (0..CALLERS)
                .map(|_| {
                    scope.spawn(|| {
                        start.wait();
                        scratch_dir("same-tag")
                    })
                })
                .collect();
            callers
                .into_iter()
                .map(|caller| caller.join().expect("a caller panicked"))
                .collect()
        });
        for dir in &dirs {
            let _ = fs::remove_dir_all(dir);
        }
        let distinct: std::collections::HashSet<_> = dirs.iter().collect();
        assert_eq!(
            distinct.len(),
            CALLERS,
            "two calls with one tag must not share a directory"
        );
    }
}
