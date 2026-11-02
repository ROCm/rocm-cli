// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! HTML renderer for the consolidated report: [`generate_consolidated`] and the
//! matrix table, legend, expectation grid and per-platform sections it
//! assembles, plus the [`RunMeta`] header line. A child of `consolidated`, so it
//! reads the model types defined there (`PlatformReport`, `Grid`,
//! `CellOutcome`) without those being widened. Re-exported from the crate root
//! by `lib.rs`.

use std::path::Path;

use maud::{Markup, html};

use super::{CellOutcome, Grid, LEGEND_ENTRIES, PlatformReport, PlatformReports, platform_cell};
use crate::components::{feature_group, now_utc, page_shell, stats_bar};

/// Run-level metadata shown in the report header so a downloaded report can be
/// traced back to the CI run that produced it. All optional — populated from CI
/// env vars, absent for a local run.
#[derive(Default)]
pub struct RunMeta {
    pub commit: Option<String>,
    pub branch: Option<String>,
    pub run_number: Option<String>,
    pub event: Option<String>,
}

impl RunMeta {
    fn line(&self) -> Option<String> {
        let mut parts = Vec::new();
        if let Some(c) = &self.commit {
            parts.push(format!("commit {}", &c[..c.len().min(7)]));
        }
        if let Some(b) = &self.branch {
            parts.push(format!("branch {b}"));
        }
        if let Some(n) = &self.run_number {
            parts.push(format!("run #{n}"));
        }
        if let Some(e) = &self.event {
            parts.push(e.clone());
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(" · "))
        }
    }
}

/// Build one consolidated HTML report from a [`PlatformReports`] bundle.
///
/// The bundle is built from `(label, json_path)` pairs passed to
/// [`load_platform_reports`](super::load_platform_reports); the label
/// identifies the platform/tier (e.g. "GPU Strix Windows (known bugs)"). New
/// platforms need no code change — the caller just passes more inputs there.
///
/// Reads no input from disk; everything it renders comes from the bundle
/// (see [`load_platform_reports`](super::load_platform_reports)).
pub fn generate_consolidated(
    bundle: &PlatformReports,
    html_out: &Path,
    meta: &RunMeta,
) -> std::io::Result<()> {
    let reports = &bundle.reports;
    let now = now_utc();
    let all_ok = reports.iter().all(PlatformReport::ok);
    let overall = if reports.is_empty() {
        ("status-fail", "No platform reports found".to_string())
    } else if all_ok {
        ("status-pass", "All platforms in expected state".to_string())
    } else {
        let bad = reports.iter().filter(|r| !r.ok()).count();
        ("status-fail", format!("{bad} platform(s) need attention"))
    };

    let body = html! {
        div.header {
            h1 { "Consolidated E2E Report" }
            div.generated {
                @if let Some(line) = meta.line() { (line) br; }
                "Generated " (now)
            }
        }

        h2 { "Summary Information" }
        table.summary-table {
            tr {
                td { "Status:" }
                td class=(overall.0) { (overall.1) }
            }
            tr { td { "Rows:" } td { (reports.len()) } }
        }

        @if reports.is_empty() {
            p { "No per-platform report.json files were found to consolidate." }
        } @else {
            h2 { "Platforms" }
            (matrix_table(reports))
            (legend())

            (expectation_grid_html(&bundle.grid))

            h2 { "Per-platform Details" }
            div.details {
                @for report in reports {
                    (platform_section(report))
                }
            }
        }
    };
    let markup = page_shell("Consolidated E2E Report", body);

    std::fs::write(html_out, markup.into_string())
}

/// Render the reconciled expectation grid as an HTML section (for the standalone
/// report). Empty markup only when there is truly nothing to report — no
/// column/row AND no excluded platform to caption (see
/// [`Grid::nothing_to_report`]); a report where every platform lacks a usable
/// `platform.json` still renders a heading and caption naming them.
pub(super) fn expectation_grid_html(grid: &Grid) -> Markup {
    // Not `grid.is_empty()`: a report where every platform lacks a
    // `platform.json` has no columns/groups either, but still has an
    // `excluded_note` worth showing — otherwise that case regresses to a
    // silently missing section, the exact complaint issue #702 raises (just
    // for every platform at once instead of some of them).
    if grid.nothing_to_report() {
        return html! {};
    }
    let problems = grid.problems();
    let excluded_note = grid.excluded_note();
    html! {
        h2 { "Expectation grid (scenario × platform)" }
        @if !grid.is_empty() {
            p.grid-legend {
                @for (i, (outcome, text)) in LEGEND_ENTRIES.iter().copied().enumerate() {
                    @if i > 0 { " · " }
                    @if outcome.grid_class().is_empty() {
                        (outcome.glyph())
                    } @else {
                        span class=(outcome.grid_class()) { (outcome.glyph()) }
                    }
                    " " (text)
                }
            }
        }
        @if let Some(note) = &excluded_note {
            p.empty-note { (note) }
        }
        // One table per feature, so a reader can scan a single area of the CLI
        // instead of one undivided 60-row block.
        @for group in &grid.groups {
            h3.feature-heading { (group.feature) }
            table.stats {
                thead {
                    tr {
                        th { "Scenario" }
                        @for col in &grid.columns {
                            @let versions = col.versions.summary();
                            th {
                                (col.label())
                                @if !col.engine.is_empty() { br; small { (col.engine) } }
                                @if !versions.is_empty() { br; small.versions { (versions) } }
                            }
                        }
                    }
                }
                tbody {
                    @for row in &group.rows {
                        tr {
                            td {
                                @if !row.name.is_empty() { (row.name) br; }
                                code { (row.id) }
                            }
                            @for col in &grid.columns {
                                @let outcome = col.outcomes.get(&row.id).copied().unwrap_or(CellOutcome::Missing);
                                // One combined class attr — `td.num class=(..)` would emit
                                // two `class` attributes, and the browser keeps only the
                                // first ("num"), dropping the status colour.
                                td class=(format!("num {}", outcome.grid_class())) {
                                    (outcome.glyph())
                                }
                            }
                        }
                    }
                }
            }
        }
        @if !problems.is_empty() {
            h3 { "Needs attention" }
            ul {
                @for (platform_label, id, outcome, detail) in &problems {
                    li {
                        b {
                            @match outcome {
                                CellOutcome::Xpass => "XPASS",
                                CellOutcome::UnexpectedFail => "unexpected failure",
                                CellOutcome::RanWhenNa => "ran despite n/a",
                                _ => "issue",
                            }
                        }
                        " on " code { (platform_label) } ": " code { (id) }
                        @if let Some(d) = detail {
                            @if let Some(b) = &d.bug { " (" (b) ")" }
                            @if !d.effective_engine.is_empty() { " [engine: " (d.effective_engine) "]" }
                            @if let Some(r) = &d.reason { " — " (r) }
                        }
                    }
                }
            }
        }
    }
}

fn matrix_table(reports: &[PlatformReport]) -> Markup {
    let (mut t_total, mut t_pass, mut t_fail, mut t_skip, mut t_xfail) = (0, 0, 0, 0, 0);
    for r in reports {
        let (total, pass, fail, skip, xf) = r.display_counts();
        t_total += total;
        t_pass += pass;
        t_fail += fail;
        t_skip += skip;
        t_xfail += xf;
    }
    html! {
        table.stats {
            tr {
                th { "Platform" } th { "OS" }
                th { "Total" } th { "Pass" } th { "Fail" } th { "Skip" }
                th { "Xfail" } th { "Status" }
                th { "Pass / Fail / Skip" }
            }
            @for r in reports {
                @let (total, pass, fail, skip, xf) = r.display_counts();
                tr {
                    td { (platform_cell(r)) }
                    td { (r.desc.os) }
                    td.num { (total) }
                    td.num { (pass) }
                    td.num { (fail) }
                    td.num { (skip) }
                    td.num { @if xf > 0 || r.is_known_bugs { (xf) } @else { "n/a" } }
                    td class=(if r.ok() { "status-pass" } else { "status-fail" }) {
                        (r.status_text())
                    }
                    td { (stats_bar(&r.stats)) }
                }
            }
            tr.total-row {
                td { "Total" } td {}
                td.num { (t_total) }
                td.num { (t_pass) }
                td.num { (t_fail) }
                td.num { (t_skip) }
                td.num { (t_xfail) }
                td {} td {}
            }
        }
    }
}

/// Explain the non-obvious columns/terms so the report is self-contained.
fn legend() -> Markup {
    html! {
        div.legend {
            h3 { "Legend" }
            ul {
                li {
                    b { "Mock" }
                    " — no GPU. The CLI runs against a fake in-process model "
                    "server (a planted service record), validating CLI behaviour "
                    "and wiring without hardware. Runs on a GitHub-hosted runner "
                    "and gates the PR."
                }
                li {
                    b { "Every other platform" }
                    " — real self-hosted GPU hardware; non-blocking while proven out."
                }
                li {
                    b { "Status" }
                    " — PASS when the platform is in its expected state (every "
                    "expect-pass scenario passed and every known bug failed as "
                    "expected); FAIL on an unexpected failure or an XPASS (a known "
                    "bug that unexpectedly passed — remove its @expected-failure tag)."
                }
                li { b { "Xfail" } " — count of known-bug scenarios that failed as expected." }
                li { b { "Skip" } " — scenarios not applicable on this platform (not run)." }
            }
        }
    }
}

fn platform_section(report: &PlatformReport) -> Markup {
    let badge_class = if report.ok() {
        "badge-pass"
    } else {
        "badge-fail"
    };
    html! {
        details.platform open[!report.ok()] {
            summary.platform-row {
                span class=(format!("badge {badge_class}")) { (report.status_text()) }
                span.platform-name { (report.label) }
                span.elapsed { (report.stats.total) " scenarios" }
            }
            @if report.features.is_empty() {
                p.empty-note { "No report.json data for this platform." }
            } @else {
                @for feature in &report.features {
                    (feature_group(feature))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consolidated::test_support::{feature_json, write_platform};
    use crate::consolidated::{load_platform_reports, load_reports};

    #[test]
    fn generate_consolidated_renders_the_expectation_grid() {
        // Review finding (6f130efe round, back when `generate_consolidated`
        // was `generate_consolidated_from_reports` and took `reports` and
        // `inputs` as two loose parameters): passing `&[]` instead of `inputs`
        // to the `expectation_grid_html(inputs)` call inside passed every
        // existing test at the time, because the fixtures used elsewhere had
        // no platform.json, so the grid was empty either way. The same slip
        // now would be building the bundle's grid from `&[]` in
        // `load_platform_reports`. Give this one a platform.json via
        // write_platform and assert the grid heading actually renders.
        let report = feature_json(&[(&["id:serve-a"], &["passed"])]);
        let platform = r#"{
            "platform_slug": "mock",
            "capability": {"effective_serve_engine": "none"},
            "expectations": [
                {"id":"serve-a","feature":"Serving","scenario":"serve-01 - first","expected":"pass"}
            ]
        }"#;
        let (_d, path) = write_platform(&report, platform);
        let inputs = vec![("mock".to_string(), path)];
        let bundle = load_platform_reports(&inputs);

        let out = tempfile::NamedTempFile::new().expect("temp");
        generate_consolidated(&bundle, out.path(), &RunMeta::default()).expect("generate");
        let html = std::fs::read_to_string(out.path()).expect("read");
        assert!(
            html.contains("Expectation grid (scenario × platform)"),
            "grid heading missing from output:\n{html}"
        );
    }

    #[test]
    fn generate_consolidated_writes_html() {
        // Same platform_slug, two channels: the HTML matrix (matrix_table) must
        // carry the channel suffix too, not just its markdown twin.
        let release_platform = r#"{
            "platform_slug": "mi300x",
            "capability": {"effective_serve_engine": "vllm"},
            "versions": {"channel": "release"}
        }"#;
        let nightly_platform = r#"{
            "platform_slug": "mi300x",
            "capability": {"effective_serve_engine": "vllm"},
            "versions": {"channel": "nightly"}
        }"#;
        let report = feature_json(&[(&[], &["passed"])]);
        let (_d1, release_path) = write_platform(&report, release_platform);
        let (_d2, nightly_path) = write_platform(&report, nightly_platform);
        let out = tempfile::NamedTempFile::new().expect("temp");
        let inputs = vec![
            ("mi300x-release".to_string(), release_path),
            ("mi300x-nightly".to_string(), nightly_path),
        ];
        generate_consolidated(
            &load_platform_reports(&inputs),
            out.path(),
            &RunMeta::default(),
        )
        .expect("generate");
        let html = std::fs::read_to_string(out.path()).expect("read");
        assert!(html.contains("Consolidated E2E Report"));
        assert!(
            html.contains("<title>Consolidated E2E Report</title>"),
            "the page_shell title argument must reach the <title> tag, not just the <h1>"
        );
        assert!(html.contains("Platforms"));
        assert!(html.contains("Legend"));
        assert!(
            html.contains("Mi300x (release)"),
            "HTML matrix must carry the release channel suffix"
        );
        assert!(
            html.contains("Mi300x (nightly)"),
            "HTML matrix must carry the nightly channel suffix"
        );

        // The sort key's channel element only matters if it actually drives
        // order: nightly's lexical head start ("n" < "r") must place it before
        // release in the rendered matrix, or this element of the tuple is dead.
        let nightly_pos = html.find("Mi300x (nightly)").expect("nightly cell present");
        let release_pos = html.find("Mi300x (release)").expect("release cell present");
        assert!(
            nightly_pos < release_pos,
            "HTML matrix must order nightly before release for the same platform"
        );
    }

    #[test]
    fn expectation_grid_html_legend_colors_every_red_glyph() {
        // Every outcome that renders red in the grid (`grid_class() ==
        // "status-fail"`) must also render red in the legend, not just the
        // original Xfail/UnexpectedFail pair — Xpass, RanWhenNa, and Absent
        // are equally `status-fail` in the grid and were previously left as
        // plain, uncoloured glyphs in the legend.
        let report = feature_json(&[(&["id:serve-x"], &["passed"])]);
        let platform = r#"{
            "platform_slug": "mi300x",
            "capability": {"effective_serve_engine": "vllm"},
            "expectations": [
                {"id":"serve-x","effective_engine":"vllm","expected":"pass"}
            ]
        }"#;
        let (_d, path) = write_platform(&report, platform);
        let inputs = vec![("mi300x".to_string(), path)];

        let html = expectation_grid_html(&Grid::build(&load_reports(&inputs))).into_string();
        for glyph in ["⚠️XPASS", "⚠️n/a-ran", "⚠️no-result", "❌FAIL"] {
            let needle = format!("class=\"status-fail\">{glyph}</span>");
            assert!(
                html.contains(&needle),
                "legend must colour {glyph:?} status-fail:\n{html}"
            );
        }
        // The known-bug xfail glyph is styled grey (status-xfail), not red.
        assert!(
            html.contains("class=\"status-xfail\">✗</span>"),
            "legend must colour the xfail glyph status-xfail:\n{html}"
        );
        // Genuinely unstyled outcomes stay bare glyphs, not empty-class spans.
        assert!(
            !html.contains("class=\"\">"),
            "unstyled legend glyphs must not be wrapped in an empty-class span:\n{html}"
        );
    }
}
