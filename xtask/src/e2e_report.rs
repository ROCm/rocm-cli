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

    let meta = e2e_report::RunMeta {
        commit: std::env::var("GITHUB_SHA").ok(),
        // GITHUB_REF_NAME is the short branch/tag name; fall back to GITHUB_REF.
        branch: std::env::var("GITHUB_REF_NAME")
            .ok()
            .or_else(|| std::env::var("GITHUB_REF").ok()),
        run_number: std::env::var("GITHUB_RUN_NUMBER").ok(),
        event: std::env::var("GITHUB_EVENT_NAME").ok(),
    };

    e2e_report::generate_consolidated(&inputs, html_out, &meta)
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

/// Return `(label, report_json_path)` for each platform report under `dir`,
/// sorted by label for stable output.
///
/// Two layouts are handled, because `actions/download-artifact@v8` does NOT
/// always create a per-artifact subdirectory:
///   * multi-artifact download → `dir/<artifact-name>/report.json` (one subdir
///     per artifact); the subdir name is the label.
///   * single-artifact download → `dir/report.json` at the ROOT. When exactly
///     one artifact matches the download `pattern`, v8 extracts it straight into
///     the `path` (its source picks `resolvedPath` when `artifacts.length === 1`,
///     regardless of `pattern`/`merge-multiple`). After the ci.yml ⇄
///     e2e-selfhosted.yml split each report job has exactly ONE artifact, so this
///     is the normal case there. The root file has no name to label from, so we
///     recover the label from the sibling `platform.json`'s `platform_slug`.
fn discover(dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut inputs = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        // A missing artifacts dir means no platforms ran — treat as empty, not
        // an error, so the aggregator's `if: always()` never hard-fails.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(inputs),
        Err(e) => return Err(e).context("reading artifacts dir")?,
    };

    // Single-artifact layout: a report.json sits directly in `dir`.
    let root_report = dir.join("report.json");
    if root_report.is_file() {
        inputs.push((label_for_root_report(dir), root_report));
    }

    for entry in entries {
        let entry = entry.context("reading dir entry")?;
        if !entry.file_type().context("stat dir entry")?.is_dir() {
            continue;
        }
        let subdir = entry.path();
        let report = subdir.join("report.json");
        if report.is_file() {
            // Pass the raw artifact/dir name through; the e2e-report crate parses
            // it into Platform / OS / Tier and owns all display formatting.
            let raw = entry.file_name().to_string_lossy().into_owned();
            inputs.push((raw, report));
        }
    }

    inputs.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(inputs)
}

/// Label for a root-level (single-artifact) `report.json`. Recovers the platform
/// from the sibling `platform.json`'s `platform_slug` and maps it to the same
/// artifact-name shape `e2e_report::parse_descriptor` expects, so a flattened
/// download renders identically to a per-subdir one.
///
/// The sidecar can be ABSENT even for a GPU run: the harness writes `report.json`
/// first and, on a parsing/hook error, exits BEFORE writing `platform.json` (see
/// tests/e2e-cucumber/tests/e2e.rs). So a missing/unrecognized slug must NOT be
/// labeled `mock` — that would misattribute a hardware failure to Mock/Linux in
/// the grid. Only an explicit `mock` slug maps to the mock artifact; anything
/// missing or unknown gets a neutral `e2e-unknown-report`, which
/// `parse_descriptor` renders as "Unknown" rather than claiming a real platform.
fn label_for_root_report(dir: &Path) -> String {
    let slug = std::fs::read_to_string(dir.join("platform.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| {
            v.get("platform_slug")
                .and_then(|s| s.as_str())
                .map(str::to_owned)
        });
    match slug.as_deref() {
        Some("mock") => "e2e-report".to_owned(),
        Some("mi300x") => "e2e-gpu-report".to_owned(),
        Some("strix-halo-linux") => "e2e-gpu-strix-ubuntu-report".to_owned(),
        Some("strix-halo-windows") => "e2e-gpu-strix-windows-report".to_owned(),
        // Missing sidecar (e.g. a GPU run that errored before writing it) or an
        // unrecognized slug → neutral identity, never a false "Mock".
        _ => "e2e-unknown-report".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Raw artifact names are passed through, sorted; the e2e-report crate
        // turns them into Platform / OS / Tier.
        let names: Vec<&str> = got.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["e2e-gpu-report", "e2e-report"]);
    }

    // Single-artifact download flattens report.json into the dir root (no
    // per-artifact subdir). Discovery must still find it and label it from the
    // platform.json sidecar's slug — otherwise the split's one-artifact report
    // jobs silently produce an empty report.
    #[test]
    fn discover_finds_root_level_report_labeled_from_slug() {
        for (slug, expected) in [
            ("mi300x", "e2e-gpu-report"),
            ("strix-halo-linux", "e2e-gpu-strix-ubuntu-report"),
            ("strix-halo-windows", "e2e-gpu-strix-windows-report"),
            ("mock", "e2e-report"),
        ] {
            let tmp = tempfile::tempdir().expect("tempdir");
            let root = tmp.path();
            std::fs::write(root.join("report.json"), "[]").unwrap();
            std::fs::write(
                root.join("platform.json"),
                format!(r#"{{"platform_slug":"{slug}"}}"#),
            )
            .unwrap();

            let got = discover(root).expect("discover");
            let names: Vec<&str> = got.iter().map(|(n, _)| n.as_str()).collect();
            assert_eq!(names, vec![expected], "slug {slug}");
        }
    }

    // A root report.json with no platform.json still resolves (rather than being
    // dropped) — but to a NEUTRAL "unknown" label, never "mock". This is the
    // GPU-run-that-errored case: the harness writes report.json, then exits on a
    // parsing/hook error BEFORE writing platform.json. Labeling it "mock" would
    // misattribute a hardware failure to Mock/Linux in the consolidated grid.
    #[test]
    fn discover_root_report_without_sidecar_is_neutral_not_mock() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        std::fs::write(root.join("report.json"), "[]").unwrap();

        let got = discover(root).expect("discover");
        let names: Vec<&str> = got.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["e2e-unknown-report"]);
        // Must NOT masquerade as the mock platform.
        assert_ne!(names, vec!["e2e-report"]);
    }

    // An unrecognized slug is likewise neutral, never a false real platform.
    #[test]
    fn discover_root_report_unknown_slug_is_neutral() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        std::fs::write(root.join("report.json"), "[]").unwrap();
        std::fs::write(
            root.join("platform.json"),
            r#"{"platform_slug":"some-future-gpu"}"#,
        )
        .unwrap();

        let got = discover(root).expect("discover");
        let names: Vec<&str> = got.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["e2e-unknown-report"]);
    }
}
