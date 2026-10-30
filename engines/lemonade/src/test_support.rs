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
/// calls agree whatever the tag. The `remove_dir_all` wipe stays anyway: after
/// pid reuse, it clears a leftover directory of the same name that an earlier
/// run left behind.
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

/// The temp name the retired `rocm_core::write_atomic` picked when it
/// replaced `backend_versions.json`: `with_extension("pins.tmp-<millis>-<pid>")`
/// swapped the `json` extension for `pins`, so the name was
/// `backend_versions.pins.tmp-<millis>-<pid>`, with this process's pid. It was
/// fixed by the clock and the pid alone, so a test can occupy it ahead of time.
pub(crate) fn retired_pin_temp_path(pin_file: &Path, millis: u128) -> PathBuf {
    pin_file.with_extension(format!("pins.tmp-{millis}-{}", std::process::id()))
}

/// Runs `write` while a foreign file occupies every name `foreign_name`
/// returns for each millisecond of a window starting now, and asserts that
/// none of them was written through, truncated or moved. Mirrors the helper of
/// the same name in `rocm-core`'s test support, which is private to that crate.
///
/// This is what tells a writer that reserves its own temp file
/// (`rocm_core::atomic_write`) from one that writes to a clock-and-pid name
/// with `fs::write`: the second opens the foreign file, truncates it and
/// renames it away, taking another writer's staged bytes with it.
///
/// `write` runs once per attempt, so it must set up whatever it needs itself.
/// One name covers one millisecond, so when creating the names is slower than
/// one per millisecond, or the write finishes after the window, the attempt
/// proves nothing; it is retried with a wider window, and a host too slow for
/// any window says so on stderr and passes rather than failing correct code.
/// The directory the names live in must exist already.
pub(crate) fn assert_write_spares_foreign_temp_files(
    foreign_name: impl Fn(u128) -> PathBuf,
    mut write: impl FnMut() -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    const FOREIGN: &[u8] = b"another writer's staged bytes";
    const ATTEMPTS: usize = 3;
    const MAX_WINDOW_MS: u128 = 16_000;

    let mut window_ms: u128 = 2_000;
    for attempt in 1..=ATTEMPTS {
        let start = rocm_core::unix_time_millis();
        let foreign: Vec<PathBuf> = (start..start + window_ms).map(&foreign_name).collect();
        for path in &foreign {
            fs::write(path, FOREIGN)?;
        }
        let setup_ms = rocm_core::unix_time_millis() - start;

        write()?;
        let finished = rocm_core::unix_time_millis();

        for path in &foreign {
            assert_eq!(
                fs::read(path).ok().as_deref(),
                Some(FOREIGN),
                "{} belongs to another writer but was written through or moved",
                path.display()
            );
        }
        for path in &foreign {
            fs::remove_file(path)?;
        }
        if finished < start + window_ms {
            return Ok(());
        }
        let next_ms = 4 * (finished - start);
        if setup_ms * 2 >= window_ms || next_ms > MAX_WINDOW_MS {
            eprintln!(
                "inconclusive: pre-creating {window_ms} names took {setup_ms} ms and the write \
                 finished after them; this filesystem is too slow for the test to say anything"
            );
            return Ok(());
        }
        window_ms = window_ms.max(next_ms);
        eprintln!(
            "attempt {attempt}/{ATTEMPTS}: the write finished after the pre-created names ran \
             out; retrying with a {window_ms} ms window"
        );
    }
    eprintln!(
        "inconclusive: on every attempt the write ran after the window of pre-created names, \
         so this host is too slow for this test to say anything"
    );
    Ok(())
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
