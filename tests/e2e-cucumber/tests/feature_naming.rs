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
    ("bench.feature", "bench"),
    ("chat.feature", "chat"),
    ("dash.feature", "dash"),
    ("diagnose.feature", "diagnose"),
    ("engine_shell.feature", "engine-shell"),
    ("examine.feature", "examine"),
    ("install_lifecycle.feature", "lifecycle"),
    ("model_serving.feature", "serve"),
    ("networking.feature", "networking"),
    ("runtime_setup.feature", "runtime"),
];

fn features_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("features")
}

/// Every `.feature` file actually present, by file name.
fn feature_files() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(features_dir())
        .expect("features dir")
        .map(|e| e.expect("dir entry").file_name().to_string_lossy().into())
        .filter(|n: &String| n.ends_with(".feature"))
        .collect();
    names.sort();
    names
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
