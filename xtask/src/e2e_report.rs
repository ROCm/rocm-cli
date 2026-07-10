// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Consolidate per-platform E2E `report.json` files into one cross-platform
//! HTML report + a `$GITHUB_STEP_SUMMARY` matrix.
//!
//! Auto-discovers platforms: every immediate subdirectory of `artifacts_dir`
//! that contains a `report.json` becomes one platform, labeled from its
//! directory name. Adding a new platform to CI needs no change here — its
//! `e2e-*-report` artifact simply shows up as a new subdir.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Discover platform reports under `artifacts_dir`, write the consolidated HTML
/// to `html_out`, and print the summary matrix to stdout.
pub fn run(artifacts_dir: &Path, html_out: &Path) -> Result<()> {
    let inputs = discover(artifacts_dir)
        .with_context(|| format!("scanning artifacts dir {}", artifacts_dir.display()))?;

    if inputs.is_empty() {
        eprintln!(
            "warning: no per-platform report.json found under {} — writing an empty report",
            artifacts_dir.display()
        );
    }

    if let Some(parent) = html_out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating output dir {}", parent.display()))?;
    }

    e2e_report::generate_consolidated(&inputs, html_out)
        .with_context(|| format!("writing consolidated report to {}", html_out.display()))?;

    // Printed to stdout so CI can redirect it into $GITHUB_STEP_SUMMARY.
    print!("{}", e2e_report::consolidated_summary_markdown(&inputs));

    eprintln!(
        "Consolidated {} platform report(s) -> {}",
        inputs.len(),
        html_out.display()
    );
    Ok(())
}

/// Return `(label, report_json_path)` for each immediate subdir of `dir` that
/// contains a `report.json`, sorted by label for stable output.
fn discover(dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut inputs = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        // A missing artifacts dir means no platforms ran — treat as empty, not
        // an error, so the aggregator's `if: always()` never hard-fails.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(inputs),
        Err(e) => return Err(e).context("reading artifacts dir")?,
    };

    for entry in entries {
        let entry = entry.context("reading dir entry")?;
        if !entry.file_type().context("stat dir entry")?.is_dir() {
            continue;
        }
        let subdir = entry.path();
        let report = subdir.join("report.json");
        if report.is_file() {
            let raw = entry.file_name().to_string_lossy().into_owned();
            inputs.push((label_from_artifact(&raw), report));
        }
    }

    inputs.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(inputs)
}

/// Turn an artifact/dir name like `e2e-gpu-strix-windows-known-bugs-report`
/// into a human label like `Gpu Strix Windows Known Bugs`.
fn label_from_artifact(name: &str) -> String {
    let without_prefix = name.strip_prefix("e2e-").unwrap_or(name);
    let core = without_prefix
        .strip_suffix("-report")
        .unwrap_or(without_prefix);

    // The bare mock expect-pass artifact is `e2e-report`; after stripping it
    // collapses to `report` (or empty). Label that tier "Mock".
    if core.is_empty() || core == "report" {
        return "Mock".to_string();
    }

    core.split('-')
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut c = w.chars();
            c.next().map_or_else(String::new, |first| {
                first.to_uppercase().collect::<String>() + c.as_str()
            })
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_strips_prefix_and_suffix_and_titlecases() {
        assert_eq!(
            label_from_artifact("e2e-gpu-strix-windows-known-bugs-report"),
            "Gpu Strix Windows Known Bugs"
        );
        // The bare mock expect-pass artifact collapses to "Mock".
        assert_eq!(label_from_artifact("e2e-report"), "Mock");
        assert_eq!(label_from_artifact("e2e-gpu-report"), "Gpu");
        assert_eq!(label_from_artifact("e2e-known-bugs-report"), "Known Bugs");
    }

    #[test]
    fn discover_missing_dir_is_empty() {
        let got = discover(Path::new("/no/such/dir")).expect("ok");
        assert!(got.is_empty());
    }

    #[test]
    fn discover_finds_subdirs_with_report_json() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();

        // Two platform dirs with report.json, one without, one loose file.
        for name in ["e2e-report", "e2e-gpu-report"] {
            let d = root.join(name);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("report.json"), "[]").unwrap();
        }
        std::fs::create_dir_all(root.join("e2e-empty-report")).unwrap(); // no report.json
        std::fs::write(root.join("loose.txt"), "x").unwrap();

        let got = discover(root).expect("discover");
        let labels: Vec<&str> = got.iter().map(|(l, _)| l.as_str()).collect();
        // Sorted by label: "Gpu" (from e2e-gpu-report) then "Mock" (from e2e-report).
        assert_eq!(labels, vec!["Gpu", "Mock"]);
    }
}
