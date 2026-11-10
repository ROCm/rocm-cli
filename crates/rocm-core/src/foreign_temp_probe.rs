// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! A test probe for the claim "this writer never touches another writer's
//! temp file", shared by this crate's tests and by the tests of the crates
//! that publish through [`crate::atomic_write`].
//!
//! Compiled only for this crate's own tests and under the `test-support`
//! feature, which a dependent crate enables from its `[dev-dependencies]`
//! entry, so it is absent from every shipped build.

use crate::unix_time_millis;
use std::path::PathBuf;

const FOREIGN: &[u8] = b"another writer's staged bytes";
const ATTEMPTS: usize = 3;
const INITIAL_WINDOW_MS: u128 = 400;
/// Ceiling for both the window and the lead before the probe gives up.
const MAX_MS: u128 = 16_000;

/// What one run of the probe established.
#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    /// The write ran inside the window of occupied names and left every one
    /// of them alone.
    Spared,
    /// The host was too slow for any window to cover the write, so a name may
    /// not have been in its way and the run says nothing about the writer.
    Inconclusive(String),
}

/// Runs `write` while a foreign file occupies every name `foreign_name`
/// returns for each millisecond of a window, and asserts that none of them was
/// written through, truncated or moved.
///
/// This is what tells a writer that reserves its own temp file
/// (`atomic_write`) from one that writes to a clock-and-pid name with
/// `fs::write`: the second opens the foreign file, truncates it and renames
/// it away, taking another writer's staged bytes with it. Every name in the
/// window is taken, so a writer that picks one of them hits a foreign file
/// whatever millisecond its write runs in.
///
/// The window starts a "lead" after the names are first created, and `write`
/// only runs once it has begun, so creating the names can take as long as the
/// host needs (about a millisecond per file is common) without the clock
/// running past them. `write` runs once per attempt, so it must set up
/// whatever it needs itself, inside the window. When the creation outlasts
/// the lead or the write outlasts the window, the attempt is retried with a
/// longer lead or window. A host too slow for any of them fails the test with
/// an "inconclusive" panic, because a probe that quietly passes there is a
/// test that does not test. The one exception is Windows, where the cost of
/// creating files has not been measured: there the probe says so on stderr
/// and passes, so a run on a Windows host can pass without the probe having
/// proven anything. The directory the names live in must exist already.
pub fn assert_write_spares_foreign_temp_files(
    foreign_name: impl Fn(u128) -> PathBuf,
    write: impl FnMut() -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let outcome = probe(foreign_name, write, INITIAL_WINDOW_MS, MAX_MS, ATTEMPTS)?;
    require_conclusive(outcome, cfg!(windows));
    Ok(())
}

/// Fails the test for an inconclusive `outcome` unless `tolerate_inconclusive`
/// is set, in which case it only says so on stderr. A parameter rather than a
/// `cfg!` inside, so both branches run in a test on every platform.
fn require_conclusive(outcome: Outcome, tolerate_inconclusive: bool) {
    if let Outcome::Inconclusive(why) = outcome {
        if tolerate_inconclusive {
            eprintln!("inconclusive (tolerated on Windows): {why}");
        } else {
            panic!(
                "the foreign-temp-file probe is inconclusive, so it cannot show the writer is \
                 safe: {why}"
            );
        }
    }
}

fn probe(
    foreign_name: impl Fn(u128) -> PathBuf,
    mut write: impl FnMut() -> anyhow::Result<()>,
    initial_window_ms: u128,
    max_ms: u128,
    attempts: usize,
) -> anyhow::Result<Outcome> {
    let mut window_ms = initial_window_ms;
    let mut lead_ms = 2 * initial_window_ms;
    for attempt in 1..=attempts {
        let created_from = unix_time_millis();
        let window_start = created_from + lead_ms;
        let window_end = window_start + window_ms;
        let foreign: Vec<PathBuf> = (window_start..window_end).map(&foreign_name).collect();
        for path in &foreign {
            std::fs::write(path, FOREIGN)?;
        }
        let setup_ms = unix_time_millis() - created_from;
        while unix_time_millis() < window_start {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        let write_started = unix_time_millis();
        write()?;
        let finished = unix_time_millis();

        for path in &foreign {
            assert_eq!(
                std::fs::read(path).ok().as_deref(),
                Some(FOREIGN),
                "{} belongs to another writer but was written through or moved",
                path.display()
            );
        }
        for path in &foreign {
            std::fs::remove_file(path)?;
        }
        if finished < window_end {
            return Ok(Outcome::Spared);
        }
        window_ms = window_ms.max(4 * (finished - write_started));
        lead_ms = lead_ms.max(2 * setup_ms);
        if window_ms.max(lead_ms) > max_ms {
            return Ok(Outcome::Inconclusive(format!(
                "creating {} names took {setup_ms} ms and the write took {} ms, past the \
                 window; this host is too slow for the test to say anything",
                foreign.len(),
                finished - write_started
            )));
        }
        eprintln!(
            "attempt {attempt}/{attempts}: the write finished after the pre-created names ran \
             out; retrying with a {window_ms} ms window after a {lead_ms} ms lead"
        );
    }
    Ok(Outcome::Inconclusive(
        "on every attempt the write ran after the window of pre-created names, so this host \
         is too slow for this test to say anything"
            .to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn occupied_name(dir: &Path, millis: u128) -> PathBuf {
        dir.join(format!("target.tmp-{millis}"))
    }

    /// How many times a self-test below reruns the probe when the host was too
    /// slow for it to be conclusive, before accepting that it cannot say.
    const SELF_TEST_RUNS: usize = 3;

    /// Reruns `run` until it reaches a verdict about the writer, `None` when
    /// every run was inconclusive. A panic from the probe propagates.
    fn run_until_conclusive(mut run: impl FnMut() -> anyhow::Result<Outcome>) -> Option<Outcome> {
        for _ in 0..SELF_TEST_RUNS {
            match run().expect("probe ran") {
                Outcome::Inconclusive(why) => eprintln!("self-test run inconclusive: {why}"),
                Outcome::Spared => return Some(Outcome::Spared),
            }
        }
        None
    }

    fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| {
                payload
                    .downcast_ref::<&str>()
                    .map(|text| (*text).to_owned())
            })
            .unwrap_or_default()
    }

    #[test]
    fn a_writer_that_reuses_a_retired_name_is_caught() {
        let dir = tempfile::tempdir().expect("scratch directory");
        // A writer shaped like the retired `write_atomic`: a temp name fixed
        // by the clock alone, opened with `fs::write`, which truncates
        // whatever is already there.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_until_conclusive(|| {
                probe(
                    |millis| occupied_name(dir.path(), millis),
                    || {
                        std::fs::write(
                            occupied_name(dir.path(), unix_time_millis()),
                            b"clobbered",
                        )?;
                        Ok(())
                    },
                    INITIAL_WINDOW_MS,
                    MAX_MS,
                    ATTEMPTS,
                )
            })
        }));
        match result {
            Err(payload) => {
                let text = panic_text(payload.as_ref());
                assert!(
                    text.contains("belongs to another writer"),
                    "the probe failed for another reason: {text}"
                );
            }
            Ok(Some(verdict)) => {
                panic!("a writer that truncates a foreign temp file was reported {verdict:?}")
            }
            // A host too slow for any window to cover the write says nothing
            // about the probe; the run above already printed why.
            Ok(None) => {}
        }
    }

    #[test]
    fn a_writer_that_picks_its_own_name_is_spared() {
        let dir = tempfile::tempdir().expect("scratch directory");
        let verdict = run_until_conclusive(|| {
            probe(
                |millis| occupied_name(dir.path(), millis),
                || {
                    std::fs::write(dir.path().join("unrelated.tmp"), b"staged")?;
                    Ok(())
                },
                INITIAL_WINDOW_MS,
                MAX_MS,
                ATTEMPTS,
            )
        });
        // `None` is a host too slow to say; a panic inside would be the probe
        // wrongly blaming a writer that never touched a foreign name.
        assert!(
            matches!(verdict, None | Some(Outcome::Spared)),
            "{verdict:?}"
        );
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .expect("list scratch directory")
            .map(|entry| entry.expect("directory entry").file_name())
            .collect();
        assert_eq!(
            leftovers,
            [std::ffi::OsString::from("unrelated.tmp")],
            "the probe must remove the names it occupied"
        );
    }

    #[test]
    fn a_write_slower_than_every_window_is_reported_inconclusive() {
        let dir = tempfile::tempdir().expect("scratch directory");
        let outcome = probe(
            |millis| occupied_name(dir.path(), millis),
            || {
                std::thread::sleep(std::time::Duration::from_millis(60));
                Ok(())
            },
            5,
            20,
            ATTEMPTS,
        )
        .expect("probe ran");
        assert!(
            matches!(outcome, Outcome::Inconclusive(_)),
            "a write that outlasts the window proves nothing: {outcome:?}"
        );
    }

    #[test]
    #[should_panic(expected = "inconclusive")]
    fn an_inconclusive_probe_fails_when_not_tolerated() {
        require_conclusive(Outcome::Inconclusive("too slow".to_owned()), false);
    }

    #[test]
    fn an_inconclusive_probe_is_tolerated_when_allowed() {
        // Returning without a panic is the whole claim. The public entry point
        // passes `cfg!(windows)` for this flag.
        require_conclusive(Outcome::Inconclusive("too slow".to_owned()), true);
    }

    #[test]
    fn a_conclusive_probe_passes_either_way() {
        require_conclusive(Outcome::Spared, false);
        require_conclusive(Outcome::Spared, true);
    }
}
