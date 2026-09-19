// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Drift guard for the `.feature` files' scenario naming and ids.
//!
//! The report groups its expectation grid by feature and orders rows by the
//! `<feature-key>-<NN>` index in each scenario's name, so that convention is
//! load-bearing, not cosmetic. It had already drifted once — indexes restarting
//! at 1 in every file, `examine` numbered 1, 2, 5, 3, 4, a stray `6b` in
//! `model_serving`, and no indexes at all in `install_lifecycle`.
//!
//! This runs in the ordinary `cargo test` set (unlike the `e2e` target, which
//! needs a real `rocm` binary), so a mis-numbered scenario is caught without a
//! full suite run.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The short key each feature file's scenarios and ids are prefixed with.
/// Adding a `.feature` file means adding its key here — deliberately explicit,
/// so a new file can't quietly opt out of the convention.
const FEATURE_KEYS: &[(&str, &str)] = &[
    ("artifact_prefetch.feature", "artifact-prefetch"),
    ("automations.feature", "automations"),
    ("bench.feature", "bench"),
    ("chat.feature", "chat"),
    ("comfyui.feature", "comfyui"),
    ("config.feature", "config"),
    ("dash.feature", "dash"),
    ("dependency_guard.feature", "deps-guard"),
    ("diagnose.feature", "diagnose"),
    ("engine_shell.feature", "engine-shell"),
    ("examine.feature", "examine"),
    ("install_lifecycle.feature", "lifecycle"),
    ("logs.feature", "logs"),
    ("model_serving.feature", "serve"),
    ("networking.feature", "networking"),
    // Not `runtime`: `runtime_setup.feature` owns that key, and two files
    // sharing one key would collide on every index (`runtime-01` in both).
    ("runtime_lifecycle.feature", "runtime-lifecycle"),
    ("runtime_setup.feature", "runtime"),
    ("service_record_cleanup.feature", "service-cleanup"),
    ("therock_next_generation.feature", "therock-next"),
    ("update.feature", "update"),
];

fn features_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("features")
}

/// Every `.feature` file actually present, by file name.
///
/// Flat, not recursive — and it refuses to run rather than quietly covering
/// less if that stops being true. `feature_files` in `src/expectation.rs` makes
/// the same assumption and its refusal message names THIS scan as the other
/// place to fix, so the two guards belong together: a subdirectory that made
/// one of them fail loudly while the other silently skipped it would be the
/// worst of both.
fn feature_files() -> Vec<String> {
    feature_files_in(&features_dir())
}

/// The scan itself, split out so the subdirectory refusal can be exercised
/// against a temporary tree — against the real `features/` it could only fire
/// by someone breaking the repository.
///
/// Deliberately the same shape as `feature_files_in` in `src/expectation.rs`:
/// same refusal, the same `Path::extension()` test rather than a `.feature`
/// string suffix, and the same non-empty assertion. The extension test is not
/// interchangeable with the suffix one — a file named exactly `.feature` has no
/// extension and would have been taken by one scan and skipped by the other,
/// which is the divergence this pair exists to prevent.
fn feature_files_in(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(dir).expect("features dir") {
        let path = entry.expect("dir entry").path();
        assert!(
            !path.is_dir(),
            "features/ has grown a subdirectory ({}), which this scan does not descend \
             into — make it recursive, here and in src/expectation.rs, before moving any \
             .feature file into one",
            path.display()
        );
        if path.extension().is_some_and(|ext| ext == "feature") {
            let name = path.file_name().expect("dir entry name").to_string_lossy();
            names.push(name.into_owned());
        }
    }
    assert!(
        !names.is_empty(),
        "found no .feature files in {}",
        dir.display()
    );
    names.sort();
    names
}

#[test]
fn the_feature_scan_takes_the_flat_files_it_finds() {
    // Pins FILTERING and MEMBERSHIP, not sort order. `read_dir` order is
    // unspecified and happens to come back alphabetical here, so removing
    // `names.sort()` leaves this green — verified. Nothing can force a real
    // directory to yield entries out of order, so the ordering is asserted for
    // a stable comparison rather than because this test proves it.
    let dir = tempfile::tempdir().expect("no temp dir");
    std::fs::write(dir.path().join("b.feature"), "Feature: b\n").unwrap();
    std::fs::write(dir.path().join("a.feature"), "Feature: a\n").unwrap();
    std::fs::write(dir.path().join("notes.md"), "ignored\n").unwrap();
    // A file named exactly `.feature` has no extension, so neither this scan nor
    // its sibling takes it. Present here so the two stay agreed on that.
    std::fs::write(dir.path().join(".feature"), "not a feature file\n").unwrap();
    assert_eq!(feature_files_in(dir.path()), ["a.feature", "b.feature"]);
}

#[test]
#[should_panic(expected = "found no .feature files")]
fn the_feature_scan_refuses_a_directory_with_no_feature_files() {
    // The assertion this pins turns a `features/` that stopped yielding files
    // into ONE clear failure naming the directory. It is not what stops the
    // other checks passing vacuously — they already fail on their own, just
    // confusingly: `feature_files_and_declared_keys_agree` reports the first
    // FEATURE_KEYS entry as one that "does not exist", and the other three
    // panic inside `scenarios_of` with a read error. Verified by pointing
    // `features_dir()` at an empty directory with the assertion deleted: four
    // of the five checks fail, none of them saying the directory is empty.
    //
    // Unreachable against the real directory, so pinned here — and pinned the
    // same way in `src/expectation.rs`, whose scan carries the same assertion.
    //
    // The fixture writes a non-`.feature` file on purpose: what the scan
    // refuses is an empty RESULT, not an empty directory, and this covers the
    // stronger case.
    let dir = tempfile::tempdir().expect("no temp dir");
    std::fs::write(dir.path().join("notes.md"), "ignored\n").unwrap();
    let _ = feature_files_in(dir.path());
}

#[test]
#[should_panic(expected = "has grown a subdirectory")]
fn the_feature_scan_refuses_a_subdirectory_rather_than_skipping_it() {
    // The branch that stops this scan quietly covering less than it claims.
    // Unreachable against the real `features/`, so it is pinned here — the same
    // way `src/expectation.rs` pins its twin.
    let dir = tempfile::tempdir().expect("no temp dir");
    std::fs::write(dir.path().join("a.feature"), "Feature: a\n").unwrap();
    std::fs::create_dir(dir.path().join("nested")).unwrap();
    let _ = feature_files_in(dir.path());
}

/// The `@id:` tags and scenario names in one feature file, paired in declaration
/// order. Tags precede their scenario, so the most recent id seen belongs to the
/// next scenario line.
///
/// `Scenario Outline:` counts too — the suite has none today, but an outline
/// added later would otherwise slip past every check in this file silently.
fn scenarios_of(file: &str) -> Vec<(Option<String>, String)> {
    let path = features_dir().join(file);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e} — stale FEATURE_KEYS entry?", path.display()));
    let mut pending_id = None;
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('@') {
            // Strip the `@` per TAG, not just off the head of the line: on a
            // multi-tag line every token after the first keeps its own `@`, so
            // `@requires-os:linux @id:x` would hide the id. This mirrors
            // `ScenarioDecl::from_tags` in src/expectation.rs — the guard must
            // read tags exactly as the harness does, or it rejects Gherkin the
            // harness accepts.
            for tag in line.split_whitespace() {
                let tag = tag.strip_prefix('@').unwrap_or(tag);
                if let Some(id) = tag.strip_prefix("id:") {
                    pending_id = Some(id.to_owned());
                }
            }
        } else if let Some(name) = line
            .strip_prefix("Scenario: ")
            .or_else(|| line.strip_prefix("Scenario Outline: "))
        {
            out.push((pending_id.take(), name.to_owned()));
        }
    }
    out
}

#[test]
fn feature_files_and_declared_keys_agree() {
    let declared: Vec<&str> = FEATURE_KEYS.iter().map(|(f, _)| *f).collect();
    let present = feature_files();
    for file in &present {
        assert!(
            declared.contains(&file.as_str()),
            "{file} has no key in FEATURE_KEYS — add one so its scenarios are \
             indexed and its ids are qualified like every other feature",
        );
    }
    // The reciprocal: a key left behind after its file was deleted or renamed
    // would otherwise surface as an unexplained read error deep in another test.
    for file in declared {
        assert!(
            present.iter().any(|p| p == file),
            "FEATURE_KEYS lists {file}, which does not exist — drop the entry",
        );
    }
    // Every other check in this file is a `for … in scenarios_of(file)` loop, so
    // a file the parser reads as having NO scenarios passes them all vacuously.
    // A mangled `Scenario:` keyword — precisely what a bad bulk find-replace
    // does — would then hide a whole feature from the guard while the report
    // silently renders its rows unsorted.
    for file in &present {
        assert!(
            !scenarios_of(file).is_empty(),
            "{file}: no scenarios parsed — the naming checks would pass \
             vacuously. Is a `Scenario:` keyword malformed?",
        );
    }
}

#[test]
fn scenario_names_are_indexed_sequentially_per_feature() {
    for (file, key) in FEATURE_KEYS {
        for (n, (_id, name)) in scenarios_of(file).iter().enumerate() {
            let expected = format!("{key}-{:02} - ", n + 1);
            assert!(
                name.starts_with(&expected),
                "{file}: scenario {} is named {name:?} but must start with \
                 {expected:?} — indexes are per-feature, sequential, and in \
                 declaration order (the report sorts grid rows by them)",
                n + 1,
            );
        }
    }
}

#[test]
fn scenario_indexes_are_unique_across_the_suite() {
    // The whole point of the feature key: an index must name exactly one
    // scenario suite-wide. Before the key, "1" named eight different scenarios.
    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    for (file, _key) in FEATURE_KEYS {
        for (_id, name) in scenarios_of(file) {
            let index = name
                .split(" - ")
                .next()
                .expect("split always yields one part")
                .to_owned();
            if let Some(prev) = seen.insert(index.clone(), (*file).to_owned()) {
                panic!("index {index:?} is used by both {prev} and {file}");
            }
        }
    }
}

#[test]
fn every_scenario_has_a_feature_qualified_id() {
    let mut seen: BTreeMap<String, String> = BTreeMap::new();
    for (file, key) in FEATURE_KEYS {
        for (id, name) in scenarios_of(file) {
            let id = id.unwrap_or_else(|| {
                panic!("{file}: scenario {name:?} has no @id: tag — the report grid keys on it")
            });
            assert!(
                id.starts_with(&format!("{key}-")),
                "{file}: @id:{id} must start with {key:?} so the id alone says \
                 which feature it belongs to",
            );
            if let Some(prev) = seen.insert(id.clone(), (*file).to_owned()) {
                panic!("duplicate @id:{id} in both {prev} and {file}");
            }
        }
    }
}
