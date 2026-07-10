// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! HTML/markdown reporting for the cucumber E2E suite.
//!
//! Lives in its own lean crate (only `maud` + `serde`) so both the
//! `e2e-cucumber` test harness and `xtask` can depend on it without pulling the
//! harness's heavy tree (cucumber/axum/reqwest/tokio) into `xtask`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use maud::{DOCTYPE, Markup, PreEscaped, html};
use serde::Deserialize;

#[derive(Deserialize)]
struct Feature {
    name: String,
    uri: String,
    #[serde(default)]
    elements: Vec<Element>,
}

#[derive(Deserialize)]
struct Element {
    name: String,
    #[serde(default)]
    tags: Vec<Tag>,
    #[serde(default)]
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Tag {
    name: String,
}

#[derive(Deserialize)]
struct Step {
    keyword: String,
    name: String,
    #[serde(default)]
    result: StepResult,
}

#[derive(Deserialize, Default)]
struct StepResult {
    #[serde(default)]
    status: String,
    #[serde(default)]
    duration: u64,
    #[serde(default)]
    error_message: Option<String>,
}

struct Stats {
    total: u32,
    passed: u32,
    failed: u32,
    skipped: u32,
    elapsed_ns: u64,
}

impl Stats {
    const fn new() -> Self {
        Self {
            total: 0,
            passed: 0,
            failed: 0,
            skipped: 0,
            elapsed_ns: 0,
        }
    }

    fn add(&mut self, status: &str, duration_ns: u64) {
        self.total += 1;
        self.elapsed_ns += duration_ns;
        match status {
            "passed" => self.passed += 1,
            "skipped" => self.skipped += 1,
            // `failed`, `undefined`, `ambiguous`, `pending` — anything that isn't
            // an outright pass or skip is a failure. Counting `undefined`/
            // `ambiguous` as passed would greenwash a broken step definition.
            _ => self.failed += 1,
        }
    }

    fn elapsed_str(&self) -> String {
        let ms = self.elapsed_ns / 1_000_000;
        let s = ms / 1000;
        let m = s / 60;
        format!("{:02}:{:02}:{:02}.{:03}", m / 60, m % 60, s % 60, ms % 1000)
    }

    /// Percentage widths for the pass/fail/skip bar. Returns `None` when there
    /// are no scenarios (no bar to render).
    const fn bar_widths(&self) -> Option<(u32, u32, u32)> {
        if self.total == 0 {
            return None;
        }
        let pw = self.passed * 100 / self.total;
        let fw = self.failed * 100 / self.total;
        let sw = 100 - pw - fw;
        Some((pw, fw, sw))
    }

    const fn status_text(&self) -> &'static str {
        if self.failed > 0 {
            "FAIL"
        } else if self.total == 0 {
            "SKIP"
        } else {
            "PASS"
        }
    }
}

fn stats_bar(stats: &Stats) -> Markup {
    html! {
        @if let Some((pw, fw, sw)) = stats.bar_widths() {
            div.bar {
                span.bar-pass style=(format!("width:{pw}%")) {}
                span.bar-fail style=(format!("width:{fw}%")) {}
                span.bar-skip style=(format!("width:{sw}%")) {}
            }
        }
    }
}

fn scenario_status(el: &Element) -> &'static str {
    // Any non-pass, non-skip step status (failed, undefined, ambiguous, pending)
    // fails the scenario — an undefined step must not report as passed.
    for s in &el.steps {
        if !matches!(s.result.status.as_str(), "passed" | "skipped") {
            return "failed";
        }
    }
    for s in &el.steps {
        if s.result.status == "skipped" {
            return "skipped";
        }
    }
    "passed"
}

fn scenario_duration(el: &Element) -> u64 {
    el.steps.iter().map(|s| s.result.duration).sum()
}

/// Read and parse a cucumber `report.json` into its feature list. A missing or
/// malformed file yields an empty list rather than an error, so a single bad
/// platform report never sinks a consolidated run.
fn parse_features(json_path: &Path) -> Vec<Feature> {
    std::fs::read_to_string(json_path)
        .ok()
        .and_then(|json| serde_json::from_str(&json).ok())
        .unwrap_or_default()
}

fn stats_of(features: &[Feature]) -> Stats {
    let mut stats = Stats::new();
    for f in features {
        for el in &f.elements {
            stats.add(scenario_status(el), scenario_duration(el));
        }
    }
    stats
}

/// Outcome of a known-bugs ("expect failures") run.
///
/// In this mode a tagged scenario failing is the *expected* result (the bug
/// still reproduces), and a tagged scenario passing is the alarming one — the
/// bug was silently fixed and its `@expected-failure` tag should be removed so
/// the scenario moves into the blocking suite.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct XfailReport {
    /// Scenarios tagged `@expected-failure` that failed as expected (xfail).
    pub xfail: u32,
    /// Scenarios tagged `@expected-failure` that unexpectedly passed (XPASS) —
    /// these make the run fail so the stale tag gets noticed.
    pub xpass: Vec<String>,
    /// Scenarios NOT tagged `@expected-failure` that failed — a known-bugs run
    /// should only contain tagged scenarios, so an untagged failure is a real
    /// regression and also fails the run.
    pub untagged_failures: Vec<String>,
}

impl XfailReport {
    /// The run is healthy when every expected-failure scenario failed and there
    /// were no XPASS scenarios or untagged failures.
    pub const fn is_ok(&self) -> bool {
        self.xpass.is_empty() && self.untagged_failures.is_empty()
    }
}

const EXPECTED_FAILURE_TAG: &str = "expected-failure";

fn evaluate_xfail_features(features: &[Feature]) -> XfailReport {
    let mut report = XfailReport::default();
    for f in features {
        for el in &f.elements {
            let tagged = el.tags.iter().any(|t| t.name == EXPECTED_FAILURE_TAG);
            let failed = scenario_status(el) == "failed";
            match (tagged, failed) {
                (true, true) => report.xfail += 1,
                (true, false) => report.xpass.push(el.name.clone()),
                (false, true) => report.untagged_failures.push(el.name.clone()),
                (false, false) => {}
            }
        }
    }
    report
}

/// Evaluate a completed known-bugs run from its `report.json`, applying xfail
/// inversion: expected-failure scenarios are meant to fail.
///
/// Tag names in the cucumber JSON are stored without the leading `@`.
pub fn evaluate_xfail(json_path: &Path) -> std::io::Result<XfailReport> {
    let json = std::fs::read_to_string(json_path)?;
    let features: Vec<Feature> = serde_json::from_str(&json).unwrap_or_default();
    Ok(evaluate_xfail_features(&features))
}

pub fn generate(json_path: &Path, html_path: &Path) -> std::io::Result<()> {
    let json = std::fs::read_to_string(json_path)?;
    let features: Vec<Feature> = serde_json::from_str(&json).unwrap_or_default();

    let mut all = Stats::new();
    let mut by_tag: BTreeMap<String, Stats> = BTreeMap::new();
    let mut by_feature: BTreeMap<String, Stats> = BTreeMap::new();

    for f in &features {
        for el in &f.elements {
            let status = scenario_status(el);
            let dur = scenario_duration(el);
            all.add(status, dur);
            by_feature
                .entry(f.name.clone())
                .or_insert_with(Stats::new)
                .add(status, dur);
            for tag in &el.tags {
                by_tag
                    .entry(tag.name.clone())
                    .or_insert_with(Stats::new)
                    .add(status, dur);
            }
        }
    }

    let now = now_utc();

    let overall_status = all.status_text();
    let status_class = overall_status.to_lowercase();
    let status_msg = if all.failed == 0 && all.total > 0 {
        "All tests passed".to_string()
    } else if all.failed > 0 {
        format!("{} test(s) failed", all.failed)
    } else {
        "No tests executed".to_string()
    };

    let by_tag_rows: Vec<(String, &Stats)> = by_tag.iter().map(|(k, v)| (k.clone(), v)).collect();
    let by_feature_rows: Vec<(String, &Stats)> =
        by_feature.iter().map(|(k, v)| (k.clone(), v)).collect();

    let markup = html! {
        (DOCTYPE)
        html lang="en" {
            head {
                meta charset="utf-8";
                title { "E2E Test Report" }
                style { (PreEscaped(STYLE)) }
            }
            body {
                div.header {
                    h1 { "E2E Test Report" }
                    div.generated { "Generated" br; (now) }
                }

                h2 { "Summary Information" }
                table.summary-table {
                    tr {
                        td { "Status:" }
                        td class=(format!("status-{status_class}")) { (status_msg) }
                    }
                    tr { td { "Elapsed Time:" } td { (all.elapsed_str()) } }
                    tr { td { "Features:" } td { (features.len()) } }
                    tr { td { "Scenarios:" } td { (all.total) } }
                }

                h2 { "Test Statistics" }
                (stats_table("Total Statistics", &[("All Tests".to_string(), &all)]))
                @if !by_tag_rows.is_empty() {
                    (stats_table("Statistics by Tag", &by_tag_rows))
                }
                (stats_table("Statistics by Feature", &by_feature_rows))

                h2 { "Test Details" }
                div.details {
                    @for feature in &features {
                        (feature_group(feature))
                    }
                }
            }
        }
    };

    std::fs::write(html_path, markup.into_string())
}

/// The platform/OS/tier a report belongs to, parsed from its artifact name.
///
/// Splitting these into separate fields (rather than one mashed
/// "Gpu Strix Ubuntu Known Bugs" label) is what lets the matrix show distinct
/// Platform / OS / Tier columns.
struct Descriptor {
    platform: String,
    os: String,
    /// True for the `known-bugs` tier (xfail-inverted), false for expect-pass.
    known_bugs: bool,
}

impl Descriptor {
    const fn tier(&self) -> &'static str {
        if self.known_bugs {
            "known bugs"
        } else {
            "expect-pass"
        }
    }
}

/// Parse an artifact/dir name like `e2e-gpu-strix-windows-known-bugs-report`
/// into its Platform / OS / Tier. Unknown shapes fall back to a titlecased
/// platform on Linux so a new artifact still renders sensibly.
fn parse_descriptor(name: &str) -> Descriptor {
    // Strip prefix, then suffix, each relative to the prior result (not `name`),
    // so `e2e-report` correctly reduces to the empty core, not back to itself.
    let core = name.strip_prefix("e2e-").unwrap_or(name);
    let core = core.strip_suffix("-report").unwrap_or(core);

    // Tier: the `-known-bugs` suffix (or the bare `known-bugs` mock artifact).
    let (core, known_bugs) = match core.strip_suffix("known-bugs") {
        Some(rest) => (rest.trim_end_matches('-'), true),
        None => (core, false),
    };

    let (platform, os) = match core {
        // The bare mock expect-pass artifact is `e2e-report` → core "report" or "".
        "" | "report" => ("Mock", "Linux"),
        "gpu" => ("MI300X", "Linux"),
        "gpu-strix-ubuntu" => ("Strix Halo", "Ubuntu"),
        "gpu-strix-windows" => ("Strix Halo", "Windows"),
        other => return fallback_descriptor(other, known_bugs),
    };

    Descriptor {
        platform: platform.to_string(),
        os: os.to_string(),
        known_bugs,
    }
}

fn fallback_descriptor(core: &str, known_bugs: bool) -> Descriptor {
    let platform = core
        .split('-')
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut c = w.chars();
            c.next().map_or_else(String::new, |f| {
                f.to_uppercase().collect::<String>() + c.as_str()
            })
        })
        .collect::<Vec<_>>()
        .join(" ");
    Descriptor {
        platform: if platform.is_empty() {
            "Unknown".to_string()
        } else {
            platform
        },
        os: "Linux".to_string(),
        known_bugs,
    }
}

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

/// A single platform/job's parsed report plus its derived health.
///
/// One of these corresponds to one uploaded `*-report` artifact (a
/// platform × tier combination, e.g. "GPU Strix Ubuntu (known bugs)").
struct PlatformReport {
    desc: Descriptor,
    /// Human label kept for the per-platform detail sections.
    label: String,
    features: Vec<Feature>,
    stats: Stats,
    xfail: XfailReport,
    /// True when the report contains any `@expected-failure` scenario — i.e. it
    /// is a known-bugs run, whose health follows xfail inversion rather than a
    /// plain zero-failures rule.
    is_known_bugs: bool,
}

impl PlatformReport {
    fn load(artifact: String, json_path: &Path) -> Self {
        let features = parse_features(json_path);
        let stats = stats_of(&features);
        let xfail = evaluate_xfail_features(&features);
        let is_known_bugs = features
            .iter()
            .flat_map(|f| &f.elements)
            .any(|el| el.tags.iter().any(|t| t.name == EXPECTED_FAILURE_TAG));
        let desc = parse_descriptor(&artifact);
        let label = format!("{} {} ({})", desc.platform, desc.os, desc.tier());
        Self {
            desc,
            label,
            features,
            stats,
            xfail,
            is_known_bugs,
        }
    }

    /// A row is healthy (green) when it is in its expected state: for a normal
    /// tier, no failures; for a known-bugs tier, no XPASS and no untagged
    /// failures (the known bugs are supposed to fail).
    const fn ok(&self) -> bool {
        if self.is_known_bugs {
            self.xfail.is_ok()
        } else {
            self.stats.failed == 0 && self.stats.total > 0
        }
    }

    const fn status_text(&self) -> &'static str {
        if self.stats.total == 0 {
            "EMPTY"
        } else if self.ok() {
            "PASS"
        } else {
            "FAIL"
        }
    }
}

/// Build one consolidated HTML report from several per-platform `report.json`
/// files.
///
/// `inputs` is `(label, json_path)` pairs; the label identifies the
/// platform/tier (e.g. "GPU Strix Windows (known bugs)"). New platforms need no
/// code change — the caller just passes more inputs.
pub fn generate_consolidated(
    inputs: &[(String, PathBuf)],
    html_out: &Path,
    meta: &RunMeta,
) -> std::io::Result<()> {
    let mut reports: Vec<PlatformReport> = inputs
        .iter()
        .map(|(label, path)| PlatformReport::load(label.clone(), path))
        .collect();
    // Group each platform's rows together and order tiers expect-pass → known
    // bugs, instead of the alphabetical mash of the old single-label sort.
    reports.sort_by(|a, b| {
        (&a.desc.platform, &a.desc.os, a.desc.known_bugs).cmp(&(
            &b.desc.platform,
            &b.desc.os,
            b.desc.known_bugs,
        ))
    });

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

    let markup = html! {
        (DOCTYPE)
        html lang="en" {
            head {
                meta charset="utf-8";
                title { "Consolidated E2E Report" }
                style { (PreEscaped(STYLE)) }
            }
            body {
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
                    (matrix_table(&reports))
                    (legend())

                    h2 { "Per-platform Details" }
                    div.details {
                        @for report in &reports {
                            (platform_section(report))
                        }
                    }
                }
            }
        }
    };

    std::fs::write(html_out, markup.into_string())
}

/// Render the consolidated result as a GitHub-flavoured markdown table, for
/// piping into `$GITHUB_STEP_SUMMARY`. Same inputs as
/// [`generate_consolidated`].
pub fn consolidated_summary_markdown(inputs: &[(String, PathBuf)]) -> String {
    use std::fmt::Write as _;

    let reports: Vec<PlatformReport> = inputs
        .iter()
        .map(|(label, path)| PlatformReport::load(label.clone(), path))
        .collect();

    let mut reports = reports;
    reports.sort_by(|a, b| {
        (&a.desc.platform, &a.desc.os, a.desc.known_bugs).cmp(&(
            &b.desc.platform,
            &b.desc.os,
            b.desc.known_bugs,
        ))
    });

    let mut out = String::from("## E2E consolidated report\n\n");
    if reports.is_empty() {
        out.push_str("_No per-platform report.json files were found to consolidate._\n");
        return out;
    }

    out.push_str("| Platform | OS | Tier | Total | Pass | Fail | Skip | Xfail | Status |\n");
    out.push_str("|---|---|---|--:|--:|--:|--:|--:|:--|\n");
    let (mut t_total, mut t_pass, mut t_fail, mut t_skip, mut t_xfail) = (0, 0, 0, 0, 0);
    for r in &reports {
        let xfail = if r.is_known_bugs {
            r.xfail.xfail.to_string()
        } else {
            "—".to_string()
        };
        t_total += r.stats.total;
        t_pass += r.stats.passed;
        t_fail += r.stats.failed;
        t_skip += r.stats.skipped;
        if r.is_known_bugs {
            t_xfail += r.xfail.xfail;
        }
        // `writeln!` into a String never fails; the discard keeps clippy happy.
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            r.desc.platform,
            r.desc.os,
            r.desc.tier(),
            r.stats.total,
            r.stats.passed,
            r.stats.failed,
            r.stats.skipped,
            xfail,
            r.status_text(),
        );
    }
    let _ = writeln!(
        out,
        "| **Total** | | | {t_total} | {t_pass} | {t_fail} | {t_skip} | {t_xfail} | |",
    );

    out.push_str(
        "\n**Mock** = no GPU (fake in-process server, gates the PR); \
         **MI300X / Strix Halo** = self-hosted GPU (non-blocking). \
         **expect-pass** must all pass; **known bugs** are xfail-inverted \
         (failing as expected = PASS; FAIL only on XPASS or an untagged failure).\n",
    );

    // Call out anything that needs a human: XPASS (fixed bug, stale tag) and
    // untagged failures in a known-bugs run.
    let mut notes = Vec::new();
    for r in &reports {
        for name in &r.xfail.xpass {
            notes.push(format!(
                "- **XPASS** in _{}_: `{}` is tagged `@expected-failure` but passed — remove the tag.",
                r.label, name
            ));
        }
        for name in &r.xfail.untagged_failures {
            if r.is_known_bugs {
                notes.push(format!(
                    "- **Regression** in _{}_: `{}` failed but is not tagged `@expected-failure`.",
                    r.label, name
                ));
            }
        }
    }
    if !notes.is_empty() {
        out.push_str("\n### Needs attention\n\n");
        for n in notes {
            out.push_str(&n);
            out.push('\n');
        }
    }

    out
}

fn matrix_table(reports: &[PlatformReport]) -> Markup {
    let (mut t_total, mut t_pass, mut t_fail, mut t_skip, mut t_xfail) = (0, 0, 0, 0, 0);
    for r in reports {
        t_total += r.stats.total;
        t_pass += r.stats.passed;
        t_fail += r.stats.failed;
        t_skip += r.stats.skipped;
        if r.is_known_bugs {
            t_xfail += r.xfail.xfail;
        }
    }
    html! {
        table.stats {
            tr {
                th { "Platform" } th { "OS" } th { "Tier" }
                th { "Total" } th { "Pass" } th { "Fail" } th { "Skip" }
                th { "Xfail" } th { "Status" }
                th { "Pass / Fail / Skip" }
            }
            @for r in reports {
                tr {
                    td { (r.desc.platform) }
                    td { (r.desc.os) }
                    td { (r.desc.tier()) }
                    td.num { (r.stats.total) }
                    td.num { (r.stats.passed) }
                    td.num { (r.stats.failed) }
                    td.num { (r.stats.skipped) }
                    td.num { @if r.is_known_bugs { (r.xfail.xfail) } @else { "—" } }
                    td class=(if r.ok() { "status-pass" } else { "status-fail" }) {
                        (r.status_text())
                    }
                    td { (stats_bar(&r.stats)) }
                }
            }
            tr.total-row {
                td { "Total" } td {} td {}
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
                    b { "MI300X / Strix Halo" }
                    " — real self-hosted GPU hardware; non-blocking while proven out."
                }
                li {
                    b { "Tier — expect-pass" }
                    " — every scenario must pass; any failure fails the row."
                }
                li {
                    b { "Tier — known bugs" }
                    " — scenarios tagged @expected-failure. They are expected to "
                    "fail, so the result is inverted: failing as expected → PASS. "
                    "The row goes FAIL only on an XPASS (a known bug unexpectedly "
                    "passed — remove its tag) or an untagged failure."
                }
                li { b { "Xfail" } " — count of known-bug scenarios that failed as expected." }
                li { b { "Skip" } " — scenarios not run." }
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

fn feature_group(feature: &Feature) -> Markup {
    html! {
        div.feature-group {
            div.feature-title {
                "Feature: " (feature.name) " "
                span.elapsed { "(" (feature.uri) }
                ")"
            }
            @for scenario in &feature.elements {
                (scenario_block(scenario))
            }
        }
    }
}

fn scenario_block(scenario: &Element) -> Markup {
    let status = scenario_status(scenario);
    let dur_ms = scenario_duration(scenario) / 1_000_000;
    let badge_class = match status {
        "failed" => "badge-fail",
        "skipped" => "badge-skip",
        _ => "badge-pass",
    };
    html! {
        details.scenario {
            summary.scenario-row {
                span class=(format!("badge {badge_class}")) { (status.to_uppercase()) }
                span.scenario-name { (scenario.name) }
                @for tag in &scenario.tags {
                    span.tag { (tag.name) }
                }
                span.elapsed { (dur_ms) "ms" }
            }
            div.steps {
                @for step in &scenario.steps {
                    (step_row(step))
                }
            }
        }
    }
}

fn step_row(step: &Step) -> Markup {
    let (icon, icon_class) = match step.result.status.as_str() {
        "passed" => ("\u{2714}", "pass"),
        "failed" => ("\u{2718}", "fail"),
        _ => ("\u{25CB}", ""),
    };
    let step_ms = step.result.duration / 1_000_000;
    html! {
        div.step {
            span class=(format!("step-icon {icon_class}")) { (icon) }
            span.step-keyword { (step.keyword) }
            (step.name)
            span.step-duration { (step_ms) "ms" }
        }
        @if let Some(err) = &step.result.error_message {
            div.error-box { (err) }
        }
    }
}

fn stats_table(title: &str, rows: &[(String, &Stats)]) -> Markup {
    html! {
        table.stats {
            tr {
                th { (title) }
                th { "Total" } th { "Pass" } th { "Fail" } th { "Skip" }
                th { "Elapsed" } th { "Pass / Fail / Skip" }
            }
            @for (label, stats) in rows {
                tr {
                    td { a href="#" { (label) } }
                    td.num { (stats.total) }
                    td.num { (stats.passed) }
                    td.num { (stats.failed) }
                    td.num { (stats.skipped) }
                    td.num { (stats.elapsed_str()) }
                    td { (stats_bar(stats)) }
                }
            }
        }
    }
}

/// Current wall-clock time formatted as `YYYY-MM-DD HH:MM:SS UTC`, or an empty
/// string if the clock is before the Unix epoch.
fn now_utc() -> String {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| format_utc(d.as_secs()))
        .unwrap_or_default()
}

/// Format Unix epoch seconds as `YYYY-MM-DD HH:MM:SS UTC` without pulling in a
/// date crate. Uses Howard Hinnant's civil-from-days algorithm (valid for all
/// Gregorian dates), so the report shows a real timestamp rather than a stub.
fn format_utc(secs: u64) -> String {
    let days = i64::try_from(secs / 86_400).unwrap_or(0);
    let rem = secs % 86_400;
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);

    // Days since 1970-01-01 → civil (year, month, day).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!("{y:04}-{m:02}-{d:02} {hh:02}:{mm:02}:{ss:02} UTC")
}

const STYLE: &str = r#"
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 2rem; color: #1a1a1a; background: #fff; }
  h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
  h2 { font-size: 1.2rem; margin: 1.5rem 0 0.75rem; border-bottom: 2px solid #333; padding-bottom: 0.25rem; }
  h3 { font-size: 1rem; margin: 1rem 0 0.5rem; }

  .header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 1.5rem; }
  .generated { text-align: right; color: #666; font-size: 0.85rem; }

  .summary-table { width: 100%; border-collapse: collapse; margin-bottom: 1rem; }
  .summary-table td { padding: 4px 12px; }
  .summary-table td:first-child { font-weight: 600; width: 140px; }
  .status-pass { color: #2e7d32; font-weight: 600; }
  .status-fail { color: #c62828; font-weight: 600; }

  table.stats { width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; font-size: 0.9rem; }
  table.stats th { background: #f5f5f5; padding: 6px 12px; text-align: left; border: 1px solid #ddd;
                   font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.03em; }
  table.stats td { padding: 6px 12px; border: 1px solid #ddd; }
  table.stats td.num { text-align: right; font-variant-numeric: tabular-nums; }
  table.stats tr:hover { background: #fafafa; }
  table.stats tr.total-row { font-weight: 600; background: #f5f5f5; }
  table.stats tr.total-row:hover { background: #f5f5f5; }
  table.stats a { color: #1565c0; text-decoration: none; }
  table.stats a:hover { text-decoration: underline; }

  .legend { font-size: 0.85rem; color: #444; background: #fafafa; border: 1px solid #e0e0e0;
            border-radius: 4px; padding: 8px 16px; margin-bottom: 1.5rem; }
  .legend h3 { margin: 0.4rem 0; font-size: 0.9rem; }
  .legend ul { margin: 0.4rem 0; padding-left: 1.2rem; }
  .legend li { margin: 0.25rem 0; }

  .bar { display: inline-flex; width: 120px; height: 14px; border-radius: 2px; overflow: hidden; vertical-align: middle; }
  .bar-pass { background: #8bc34a; }
  .bar-fail { background: #e53935; }
  .bar-skip { background: #bdbdbd; }

  .details { margin-top: 0.5rem; }
  .feature-group { margin-bottom: 1.5rem; }
  .feature-title { font-weight: 600; font-size: 1rem; padding: 8px 12px; background: #e3f2fd;
                   border-left: 4px solid #1565c0; margin-bottom: 0; }
  .platform { border: 1px solid #cfcfcf; border-radius: 4px; padding: 12px; margin-bottom: 1rem; background: #fafafa; }
  .platform-row { display: flex; align-items: center; gap: 8px; cursor: pointer; font-size: 1.05rem; }
  .platform-name { font-weight: 700; flex: 1; }
  .empty-note { color: #999; font-style: italic; padding: 8px 0; }
  .scenario { border: 1px solid #e0e0e0; border-top: none; padding: 12px; background: #fff; }
  .scenario:last-child { border-radius: 0 0 4px 4px; }
  .scenario-row { display: flex; align-items: center; gap: 8px; cursor: pointer; }
  .scenario-row:hover { background: #f5f5f5; margin: -12px; padding: 12px; }

  .badge { font-size: 0.7rem; padding: 2px 8px; border-radius: 3px; font-weight: 700;
           text-transform: uppercase; letter-spacing: 0.04em; }
  .badge-pass { background: #c8e6c9; color: #2e7d32; }
  .badge-fail { background: #ffcdd2; color: #c62828; }
  .badge-skip { background: #e0e0e0; color: #616161; }

  .scenario-name { font-weight: 600; flex: 1; }
  .tag { font-size: 0.7rem; padding: 2px 6px; border-radius: 3px; background: #e8eaf6; color: #3949ab; }
  .elapsed { color: #999; font-size: 0.8rem; font-variant-numeric: tabular-nums; }

  .steps { margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee; }
  .step { font-family: "SF Mono", "Fira Code", "Consolas", monospace; font-size: 0.82rem;
          padding: 3px 0; display: flex; gap: 6px; align-items: baseline; }
  .step-icon { width: 1.2rem; text-align: center; flex-shrink: 0; }
  .step-icon.pass { color: #2e7d32; }
  .step-icon.fail { color: #c62828; }
  .step-keyword { color: #7b1fa2; font-weight: 600; }
  .step-duration { color: #bbb; margin-left: auto; font-size: 0.75rem; }

  .error-box { background: #fff3f3; border: 1px solid #ffcdd2; border-radius: 4px; padding: 10px;
               margin-top: 6px; font-family: monospace; font-size: 0.78rem; color: #b71c1c;
               white-space: pre-wrap; word-break: break-word; max-height: 300px; overflow-y: auto; }

  details summary { cursor: pointer; user-select: none; }
  details summary:hover { text-decoration: underline; }
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario_from(statuses: &[&str]) -> Element {
        let steps = statuses
            .iter()
            .map(|s| format!(r#"{{"keyword":"Given ","name":"x","result":{{"status":"{s}"}}}}"#))
            .collect::<Vec<_>>()
            .join(",");
        let json = format!(r#"{{"name":"s","steps":[{steps}]}}"#);
        serde_json::from_str(&json).expect("valid element json")
    }

    #[test]
    fn stats_add_counts_passed_and_skipped() {
        let mut s = Stats::new();
        s.add("passed", 0);
        s.add("skipped", 0);
        assert_eq!((s.passed, s.skipped, s.failed), (1, 1, 0));
    }

    #[test]
    fn stats_add_counts_undefined_and_ambiguous_as_failures() {
        // Regression: these were previously miscounted as passed, greenwashing
        // broken step definitions.
        let mut s = Stats::new();
        s.add("failed", 0);
        s.add("undefined", 0);
        s.add("ambiguous", 0);
        s.add("pending", 0);
        assert_eq!(s.failed, 4);
        assert_eq!(s.passed, 0);
    }

    #[test]
    fn scenario_status_undefined_step_is_not_passed() {
        // Regression: an undefined step must fail the scenario, not pass it.
        assert_eq!(
            scenario_status(&scenario_from(&["passed", "undefined"])),
            "failed"
        );
        assert_eq!(
            scenario_status(&scenario_from(&["passed", "ambiguous"])),
            "failed"
        );
    }

    #[test]
    fn scenario_status_failed_wins_over_skipped() {
        assert_eq!(
            scenario_status(&scenario_from(&["passed", "failed", "skipped"])),
            "failed"
        );
    }

    #[test]
    fn scenario_status_skipped_when_no_failures() {
        assert_eq!(
            scenario_status(&scenario_from(&["passed", "skipped"])),
            "skipped"
        );
    }

    #[test]
    fn scenario_status_all_passed() {
        assert_eq!(
            scenario_status(&scenario_from(&["passed", "passed"])),
            "passed"
        );
    }

    fn write_report(features_json: &str) -> tempfile::NamedTempFile {
        use std::io::Write as _;
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        f.write_all(features_json.as_bytes()).expect("write json");
        f
    }

    // One feature with scenarios; each scenario is (tags, step-statuses).
    fn feature_json(scenarios: &[(&[&str], &[&str])]) -> String {
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

    #[test]
    fn xfail_all_tagged_failing_is_ok() {
        let f = write_report(&feature_json(&[
            (&["expected-failure"], &["failed"]),
            (
                &["expected-failure", "expected-failure-EAI-7219"],
                &["passed", "failed"],
            ),
        ]));
        let r = evaluate_xfail(f.path()).expect("evaluate");
        assert_eq!(r.xfail, 2);
        assert!(r.is_ok());
    }

    #[test]
    fn xfail_tagged_passing_is_xpass_and_not_ok() {
        // A known bug that now passes must fail the run so the stale tag is noticed.
        let f = write_report(&feature_json(&[
            (&["expected-failure"], &["failed"]),
            (&["expected-failure"], &["passed", "passed"]),
        ]));
        let r = evaluate_xfail(f.path()).expect("evaluate");
        assert_eq!(r.xfail, 1);
        assert_eq!(r.xpass, vec!["s1".to_string()]);
        assert!(!r.is_ok());
    }

    #[test]
    fn xfail_untagged_failure_is_not_ok() {
        // An untagged scenario shouldn't be in a known-bugs run; if it fails,
        // that's a real regression.
        let f = write_report(&feature_json(&[(&[], &["failed"])]));
        let r = evaluate_xfail(f.path()).expect("evaluate");
        assert_eq!(r.untagged_failures, vec!["s0".to_string()]);
        assert!(!r.is_ok());
    }

    #[test]
    fn format_utc_renders_real_date_not_placeholder() {
        // 2021-01-01 00:00:00 UTC = 1_609_459_200. Regression guard against the
        // old "-xx-xx" placeholder.
        assert_eq!(format_utc(1_609_459_200), "2021-01-01 00:00:00 UTC");
        // A time-of-day sample: 2026-07-08 13:30:45 UTC.
        assert_eq!(format_utc(1_783_517_445), "2026-07-08 13:30:45 UTC");
    }

    #[test]
    fn platform_report_normal_tier_ok_only_when_no_failures() {
        let pass = write_report(&feature_json(&[(&[], &["passed"])]));
        let r = PlatformReport::load("mock".into(), pass.path());
        assert!(!r.is_known_bugs);
        assert!(r.ok());
        assert_eq!(r.status_text(), "PASS");

        let fail = write_report(&feature_json(&[(&[], &["failed"])]));
        let r = PlatformReport::load("mock".into(), fail.path());
        assert!(!r.ok());
        assert_eq!(r.status_text(), "FAIL");
    }

    #[test]
    fn platform_report_known_bugs_tier_ok_when_bugs_still_fail() {
        // Known-bug tier: tagged scenarios failing is the healthy state.
        let f = write_report(&feature_json(&[
            (&["expected-failure"], &["failed"]),
            (&["expected-failure"], &["failed"]),
        ]));
        let r = PlatformReport::load("gpu (known bugs)".into(), f.path());
        assert!(r.is_known_bugs);
        assert!(r.ok());
        assert_eq!(r.status_text(), "PASS");
        assert_eq!(r.xfail.xfail, 2);
    }

    #[test]
    fn platform_report_known_bugs_tier_fails_on_xpass() {
        let f = write_report(&feature_json(&[
            (&["expected-failure"], &["failed"]),
            (&["expected-failure"], &["passed"]),
        ]));
        let r = PlatformReport::load("gpu (known bugs)".into(), f.path());
        assert!(!r.ok());
        assert_eq!(r.status_text(), "FAIL");
    }

    #[test]
    fn missing_report_json_is_empty_not_error() {
        let r = PlatformReport::load("gone".into(), Path::new("/no/such/report.json"));
        assert_eq!(r.stats.total, 0);
        assert_eq!(r.status_text(), "EMPTY");
    }

    #[test]
    fn consolidated_summary_markdown_has_a_row_per_platform() {
        let a = write_report(&feature_json(&[(&[], &["passed"]), (&[], &["passed"])]));
        let b = write_report(&feature_json(&[(&["expected-failure"], &["failed"])]));
        let inputs = vec![
            ("e2e-report".to_string(), a.path().to_path_buf()),
            (
                "e2e-gpu-known-bugs-report".to_string(),
                b.path().to_path_buf(),
            ),
        ];
        let md = consolidated_summary_markdown(&inputs);
        assert!(md.contains("| Mock | Linux | expect-pass | 2 | 2 | 0 | 0 | — | PASS |"));
        assert!(md.contains("| MI300X | Linux | known bugs | 1 | 0 | 1 | 0 | 1 | PASS |"));
    }

    #[test]
    fn consolidated_summary_markdown_flags_xpass() {
        let b = write_report(&feature_json(&[(&["expected-failure"], &["passed"])]));
        let inputs = vec![(
            "e2e-gpu-known-bugs-report".to_string(),
            b.path().to_path_buf(),
        )];
        let md = consolidated_summary_markdown(&inputs);
        assert!(md.contains("Needs attention"));
        assert!(md.contains("XPASS"));
    }

    #[test]
    fn consolidated_summary_markdown_empty_inputs() {
        let md = consolidated_summary_markdown(&[]);
        assert!(md.contains("No per-platform report.json files"));
    }

    #[test]
    fn generate_consolidated_writes_html() {
        let a = write_report(&feature_json(&[(&[], &["passed"])]));
        let out = tempfile::NamedTempFile::new().expect("temp");
        let inputs = vec![("e2e-report".to_string(), a.path().to_path_buf())];
        generate_consolidated(&inputs, out.path(), &RunMeta::default()).expect("generate");
        let html = std::fs::read_to_string(out.path()).expect("read");
        assert!(html.contains("Consolidated E2E Report"));
        assert!(html.contains("Mock"));
        assert!(html.contains("Platforms"));
        assert!(html.contains("Legend"));
    }
}
