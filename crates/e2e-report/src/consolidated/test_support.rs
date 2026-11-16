// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Shared `#[cfg(test)]` fixtures for the `consolidated` module tree:
//! report.json builders, temp-file writers and the grid-order probe.

use std::path::PathBuf;

use super::{Grid, load_reports};

// One feature with scenarios; each scenario is (tags, step-statuses).
pub(super) fn feature_json(scenarios: &[(&[&str], &[&str])]) -> String {
    let els: Vec<String> = scenarios
        .iter()
        .enumerate()
        .map(|(i, (tags, statuses))| {
            let tags = tags
                .iter()
                .map(|t| format!(r#"{{"name":"{t}"}}"#))
                .collect::<Vec<_>>()
                .join(",");
            let steps = statuses
                .iter()
                .map(|s| {
                    format!(r#"{{"keyword":"Given ","name":"x","result":{{"status":"{s}"}}}}"#)
                })
                .collect::<Vec<_>>()
                .join(",");
            format!(r#"{{"name":"s{i}","tags":[{tags}],"steps":[{steps}]}}"#)
        })
        .collect();
    format!(
        r#"[{{"name":"F","uri":"f.feature","elements":[{}]}}]"#,
        els.join(",")
    )
}

pub(super) fn write_report(features_json: &str) -> tempfile::NamedTempFile {
    use std::io::Write as _;
    let mut f = tempfile::NamedTempFile::new().expect("temp file");
    f.write_all(features_json.as_bytes()).expect("write json");
    f
}

/// Write a report.json + platform.json pair into a fresh dir and return the
/// report.json path (the input the grid keys on).
pub(super) fn write_platform(
    report_json: &str,
    platform_json: &str,
) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("tempdir");
    let report = dir.path().join("report.json");
    std::fs::write(&report, report_json).expect("write report");
    std::fs::write(dir.path().join("platform.json"), platform_json).expect("write platform");
    (dir, report)
}

/// The order the grid renders rows in, flattened across feature groups.
pub(super) fn grid_order(inputs: &[(String, PathBuf)]) -> Vec<(String, String)> {
    Grid::build(&load_reports(inputs))
        .groups
        .iter()
        .flat_map(|g| g.rows.iter().map(|r| (g.feature.clone(), r.id.clone())))
        .collect()
}
