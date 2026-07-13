// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! HTML/markdown reporting for the cucumber E2E suite.
//!
//! Lives in its own lean crate (only `maud` + `serde`/`serde_json`) so both the
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
    /// Before-scenario hooks (cucumber JSON `before`). A failing Before hook
    /// leaves `steps` empty, so it must be inspected too or the scenario scores
    /// as passed despite never running.
    #[serde(default)]
    before: Vec<Hook>,
    /// After-scenario hooks (cucumber JSON `after`).
    #[serde(default)]
    after: Vec<Hook>,
}

/// A cucumber before/after hook entry — we only need its result status.
#[derive(Deserialize)]
struct Hook {
    #[serde(default)]
    result: StepResult,
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
        // Derive the skip width from the actual skipped count, not `100 - pw - fw`
        // — the latter dumped the integer-division remainder into the skip
        // segment, rendering a grey sliver even when there are zero skips.
        let sw = self.skipped * 100 / self.total;
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
    // A failing before/after hook fails the scenario even when `steps` is empty
    // (a Before-hook failure prevents steps from running), so it must be checked
    // — otherwise a hook-failed scenario falls through to "passed".
    for h in el.before.iter().chain(el.after.iter()) {
        if !matches!(h.result.status.as_str(), "" | "passed" | "skipped") {
            return "failed";
        }
    }
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

/// The single source of truth for "did this scenario pass" across BOTH the CI
/// gate (`scenario_results_by_id`) and the report grid (`id_pass_map`/tally).
///
/// A scenario counts as passed ONLY when every step passed — a `skipped` status
/// (steps skipped after an early bail, or an undefined step) is NOT a pass. The
/// gate and the grid previously disagreed on this (gate: `== "passed"`, grid:
/// `!= "failed"`), so the same `report.json` could fail the job yet render green
/// in the consolidated grid. Route both through here so they can never diverge.
fn scenario_passed(el: &Element) -> bool {
    scenario_status(el) == "passed"
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

/// Tag prefix carrying a scenario's stable id (`@id:<slug>`, stored without `@`).
const ID_TAG_PREFIX: &str = "id:";

/// The stable `@id:` slug of a scenario, if it has one.
fn scenario_id(el: &Element) -> Option<String> {
    el.tags
        .iter()
        .find_map(|t| t.name.strip_prefix(ID_TAG_PREFIX).map(str::to_owned))
}

/// Map each scenario's stable `@id` → whether it passed.
///
/// Read from a completed run's `report.json`. Scenarios without an `@id` tag are
/// skipped (the new system requires every scenario to carry one). Used by the
/// harness to reconcile actual results against per-scenario expectations.
pub fn scenario_results_by_id(json_path: &Path) -> std::io::Result<Vec<(String, bool)>> {
    let json = std::fs::read_to_string(json_path)?;
    let features: Vec<Feature> = serde_json::from_str(&json).unwrap_or_default();
    let mut out = Vec::new();
    for f in &features {
        for el in &f.elements {
            if let Some(id) = scenario_id(el) {
                out.push((id, scenario_passed(el)));
            }
        }
    }
    Ok(out)
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

/// The platform/OS a report belongs to, parsed from its artifact name.
///
/// Splitting these into separate fields (rather than one mashed
/// "Gpu Strix Ubuntu" label) is what lets the matrix show distinct
/// Platform / OS columns.
struct Descriptor {
    platform: String,
    os: String,
    /// True for a legacy `known-bugs` artifact (xfail-inverted). With the
    /// one-job-per-platform model these no longer exist, but the flag is retained
    /// as a stable secondary sort key so old artifacts still order predictably.
    known_bugs: bool,
}

/// Parse an artifact/dir name like `e2e-gpu-strix-windows-report`
/// into its Platform / OS. Unknown shapes fall back to a titlecased
/// platform on Linux so a new artifact still renders sensibly.
fn parse_descriptor(name: &str) -> Descriptor {
    // Strip prefix, then suffix, each relative to the prior result (not `name`),
    // so `e2e-report` correctly reduces to the empty core, not back to itself.
    let core = name.strip_prefix("e2e-").unwrap_or(name);
    let core = core.strip_suffix("-report").unwrap_or(core);

    // Legacy `-known-bugs` suffix (retained only as a stable secondary sort key).
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
    /// Recorded `rocm` invocations from this platform's `commands.jsonl`.
    commands: Vec<CommandRecord>,
    /// Expectation-reconciled outcome (`platform.json` × `report.json` by `@id`).
    /// `None` for pre-expectation artifacts, which fall back to the junit status.
    tally: Option<ReconciledTally>,
}

/// One recorded `rocm` invocation from a platform's `commands.jsonl` sidecar.
#[derive(Deserialize)]
struct CommandRecord {
    scenario: Option<String>,
    subcommand: String,
    /// Full command as executed (e.g. "rocm serve Qwen/... --engine vllm").
    /// Falls back to `subcommand` for older artifacts that predate this field.
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    engine: Option<String>,
    /// True when `engine` was the CLI's own default choice (no `--engine` flag),
    /// so the report can show it as "<engine> (default)".
    #[serde(default)]
    engine_is_default: bool,
}

/// Read a platform's `commands.jsonl` (sibling of `report.json`). Missing file =
/// no records (older artifacts, or a platform that recorded none).
fn parse_commands(json_path: &Path) -> Vec<CommandRecord> {
    let path = json_path.with_file_name("commands.jsonl");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

/// The `platform.json` sidecar written by the harness: the probed host
/// capability plus every scenario's resolved expectation (including skips, which
/// never appear in `report.json`). This is the source of truth for a platform's
/// identity and for the expected-vs-actual reconciliation.
#[derive(Deserialize)]
struct PlatformManifest {
    #[serde(default)]
    platform_slug: String,
    #[serde(default)]
    capability: Option<ManifestCapability>,
    #[serde(default)]
    expectations: Vec<ManifestExpectation>,
}

#[derive(Deserialize)]
struct ManifestCapability {
    #[serde(default)]
    effective_serve_engine: String,
}

/// One scenario's resolved expectation, keyed by its stable `@id`.
#[derive(Deserialize, Clone)]
struct ManifestExpectation {
    id: String,
    #[serde(default)]
    effective_engine: String,
    /// "pass" | "xfail" | "skip".
    expected: String,
    #[serde(default)]
    bug: Option<String>,
    #[serde(default)]
    reason: Option<String>,
}

/// Read a platform's `platform.json` (sibling of `report.json`). Missing =
/// `None` (older artifacts predating the expectation system).
fn parse_platform_manifest(json_path: &Path) -> Option<PlatformManifest> {
    let path = json_path.with_file_name("platform.json");
    let text = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&text).ok()
}

/// How a scenario's actual result compared to its expectation on one platform.
/// Drives both the grid glyph and the "needs attention" list.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CellOutcome {
    /// Expected pass, passed.
    Pass,
    /// Expected xfail, failed as expected.
    Xfail,
    /// Not applicable here (skipped) — the engine/hardware can't exercise it.
    Skip,
    /// Expected pass, but FAILED — a real regression.
    UnexpectedFail,
    /// Expected xfail, but PASSED — stale entry (bug fixed here?).
    Xpass,
    /// Expected skip, yet a result exists — harness/resolver disagreement.
    RanWhenNa,
    /// No expectation and no result recorded for this id on this platform.
    Missing,
}

impl CellOutcome {
    /// Reconcile one scenario's expectation against its actual result.
    /// `actual` is `Some(passed)` when the scenario ran, `None` when it did not
    /// appear in `report.json` (filtered out / skipped).
    fn reconcile(expected: &str, actual: Option<bool>) -> Self {
        match (expected, actual) {
            ("pass", Some(true)) => Self::Pass,
            ("pass", Some(false)) => Self::UnexpectedFail,
            ("pass", None) => Self::Missing,
            ("xfail", Some(false)) => Self::Xfail,
            ("xfail", Some(true)) => Self::Xpass,
            ("xfail", None) => Self::Missing,
            ("skip", None) => Self::Skip,
            ("skip", Some(_)) => Self::RanWhenNa,
            _ => Self::Missing,
        }
    }

    /// True when this cell needs human attention (report FAILs on any).
    const fn is_problem(self) -> bool {
        matches!(self, Self::UnexpectedFail | Self::Xpass | Self::RanWhenNa)
    }

    const fn glyph(self) -> &'static str {
        match self {
            Self::Pass => "✅",
            Self::Xfail => "xfail",
            Self::Skip => "n/a",
            Self::UnexpectedFail => "❌FAIL",
            Self::Xpass => "⚠️XPASS",
            Self::RanWhenNa => "⚠️n/a-ran",
            Self::Missing => "·",
        }
    }
}

/// One platform column of the reconciled (scenario-id × platform) grid.
struct GridColumn {
    /// Platform identity from the manifest (e.g. "mi300x", "strix-halo", "mock").
    slug: String,
    /// Effective serve engine on this host (for the column subheading).
    engine: String,
    /// scenario id → reconciled outcome.
    outcomes: std::collections::BTreeMap<String, CellOutcome>,
    /// Per-id bug/reason, surfaced in the "needs attention" list.
    details: std::collections::BTreeMap<String, ManifestExpectation>,
}

/// The reconciled grid: ordered scenario ids × platform columns. Built from each
/// input's `platform.json` (expected) joined with its `report.json` (actual) by
/// stable `@id`. Inputs without a `platform.json` (pre-expectation artifacts) are
/// skipped here — they still appear in the legacy platform×tier matrix.
struct Grid {
    /// Scenario ids in first-seen order across all columns.
    ids: Vec<String>,
    columns: Vec<GridColumn>,
}

impl Grid {
    fn build(inputs: &[(String, PathBuf)]) -> Self {
        let mut ids: Vec<String> = Vec::new();
        let mut columns: Vec<GridColumn> = Vec::new();

        for (_label, json_path) in inputs {
            let Some(manifest) = parse_platform_manifest(json_path) else {
                continue;
            };
            // Actual results by id from this platform's report.json.
            let actual = id_pass_map(json_path);

            // Merge into an existing column with the same slug (defensive; with
            // one job per platform there is exactly one input per slug).
            let col_idx = columns
                .iter()
                .position(|c| c.slug == manifest.platform_slug)
                .unwrap_or_else(|| {
                    columns.push(GridColumn {
                        slug: manifest.platform_slug.clone(),
                        engine: manifest
                            .capability
                            .as_ref()
                            .map(|c| c.effective_serve_engine.clone())
                            .unwrap_or_default(),
                        outcomes: std::collections::BTreeMap::new(),
                        details: std::collections::BTreeMap::new(),
                    });
                    columns.len() - 1
                });

            for exp in &manifest.expectations {
                if !ids.contains(&exp.id) {
                    ids.push(exp.id.clone());
                }
                let outcome = CellOutcome::reconcile(&exp.expected, actual.get(&exp.id).copied());
                // A real result supersedes a defensive Missing on merge.
                columns[col_idx]
                    .outcomes
                    .entry(exp.id.clone())
                    .and_modify(|o| {
                        if *o == CellOutcome::Missing {
                            *o = outcome;
                        }
                    })
                    .or_insert(outcome);
                columns[col_idx].details.insert(exp.id.clone(), exp.clone());
            }
        }

        ids.sort();
        Self { ids, columns }
    }

    /// Every problem cell across the grid, as `(slug, id, outcome, detail)`.
    fn problems(&self) -> Vec<(&str, &str, CellOutcome, Option<&ManifestExpectation>)> {
        let mut out = Vec::new();
        for col in &self.columns {
            for (id, outcome) in &col.outcomes {
                if outcome.is_problem() {
                    out.push((
                        col.slug.as_str(),
                        id.as_str(),
                        *outcome,
                        col.details.get(id),
                    ));
                }
            }
        }
        out
    }

    const fn is_empty(&self) -> bool {
        self.columns.is_empty() || self.ids.is_empty()
    }
}

/// Map each scenario's stable `@id` → whether it passed, from a `report.json`.
/// (Internal sibling of the public [`scenario_results_by_id`], returning a map.)
fn id_pass_map(json_path: &Path) -> std::collections::HashMap<String, bool> {
    let features = parse_features(json_path);
    features
        .iter()
        .flat_map(|f| &f.elements)
        .filter_map(|el| scenario_id(el).map(|id| (id, scenario_passed(el))))
        .collect()
}

/// A platform's outcome after reconciling each scenario's declared expectation
/// (`platform.json`) against its actual result (`report.json`) by stable `@id`.
///
/// This is the trustworthy per-platform signal in the one-job-per-platform model:
/// a known bug that fails is `xfail` (healthy), not a failure. The coarse junit
/// [`Stats`]/`XfailReport` path counts an xfail scenario as a raw failure, so it
/// wrongly reds a clean platform — this tally is what the Status column uses.
#[derive(Default, Clone, Copy)]
struct ReconciledTally {
    pass: u32,
    xfail: u32,
    skip: u32,
    /// Cells that need a human: unexpected-fail, XPASS, or ran-when-N/A.
    problems: u32,
}

impl ReconciledTally {
    /// `true` when nothing needs attention (no unexpected-fail / XPASS / ran-when-NA).
    const fn ok(&self) -> bool {
        self.problems == 0
    }

    const fn status_text(&self) -> &'static str {
        if self.ok() { "PASS" } else { "FAIL" }
    }
}

/// Reconcile one platform's `platform.json` expectations against its
/// `report.json` results. `None` when the artifact predates the expectation
/// system (no `platform.json`) — callers fall back to the legacy junit status.
fn reconciled_tally(json_path: &Path) -> Option<ReconciledTally> {
    let manifest = parse_platform_manifest(json_path)?;
    let actual = id_pass_map(json_path);
    let mut t = ReconciledTally::default();
    for exp in &manifest.expectations {
        match CellOutcome::reconcile(&exp.expected, actual.get(&exp.id).copied()) {
            CellOutcome::Pass => t.pass += 1,
            CellOutcome::Xfail => t.xfail += 1,
            CellOutcome::Skip => t.skip += 1,
            CellOutcome::UnexpectedFail | CellOutcome::Xpass | CellOutcome::RanWhenNa => {
                t.problems += 1;
            }
            // A declared expectation with no result is a missing data point, not a
            // pass or a problem — it neither greenwashes nor reds the platform.
            CellOutcome::Missing => {}
        }
    }
    Some(t)
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
        let label = format!("{} {}", desc.platform, desc.os);
        let commands = parse_commands(json_path);
        let tally = reconciled_tally(json_path);
        Self {
            desc,
            label,
            features,
            stats,
            xfail,
            is_known_bugs,
            commands,
            tally,
        }
    }

    /// Map each scenario name → whether it passed (all steps passed/skipped).
    fn scenario_pass_map(&self) -> std::collections::HashMap<String, bool> {
        self.features
            .iter()
            .flat_map(|f| &f.elements)
            .map(|el| (el.name.clone(), scenario_status(el) != "failed"))
            .collect()
    }

    /// A row is healthy (green) when it is in its expected state.
    ///
    /// Prefers the expectation-reconciled tally (a known bug failing is `xfail`,
    /// not a failure). Falls back to the legacy junit rule only for artifacts
    /// without a `platform.json`: for a normal tier, no failures; for a
    /// known-bugs tier, no XPASS and no untagged failures.
    const fn ok(&self) -> bool {
        if let Some(tally) = &self.tally {
            return tally.ok();
        }
        if self.is_known_bugs {
            self.xfail.is_ok()
        } else {
            self.stats.failed == 0 && self.stats.total > 0
        }
    }

    const fn status_text(&self) -> &'static str {
        if let Some(tally) = &self.tally {
            return tally.status_text();
        }
        if self.stats.total == 0 {
            "EMPTY"
        } else if self.ok() {
            "PASS"
        } else {
            "FAIL"
        }
    }

    /// Display counts for the summary table, reconciliation-aware so the numbers
    /// agree with [`Self::status_text`]. Returns `(total, pass, fail, skip,
    /// xfail)`. With a `platform.json`, `fail` counts only cells needing
    /// attention (unexpected-fail / XPASS / ran-when-NA) — a known bug failing as
    /// expected lands in `xfail`, not `fail`. Without one, falls back to raw
    /// junit stats.
    const fn display_counts(&self) -> (u32, u32, u32, u32, u32) {
        if let Some(t) = &self.tally {
            let total = t.pass + t.xfail + t.skip + t.problems;
            (total, t.pass, t.problems, t.skip, t.xfail)
        } else {
            let xfail = if self.is_known_bugs {
                self.xfail.xfail
            } else {
                0
            };
            (
                self.stats.total,
                self.stats.passed,
                self.stats.failed,
                self.stats.skipped,
                xfail,
            )
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

                    (expectation_grid_html(inputs))

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

    out.push_str("| Platform | OS | Total | Pass | Fail | Skip | Xfail | Status |\n");
    out.push_str("|---|---|--:|--:|--:|--:|--:|:--|\n");
    let (mut t_total, mut t_pass, mut t_fail, mut t_skip, mut t_xfail) = (0, 0, 0, 0, 0);
    for r in &reports {
        let (total, pass, fail, skip, xf) = r.display_counts();
        // Xfail column only applies where there are known bugs to invert; a plain
        // expect-pass platform (no xfail entries) shows N/A.
        let xfail = if xf > 0 || r.is_known_bugs {
            xf.to_string()
        } else {
            "n/a".to_string()
        };
        t_total += total;
        t_pass += pass;
        t_fail += fail;
        t_skip += skip;
        t_xfail += xf;
        // `writeln!` into a String never fails; the discard keeps clippy happy.
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            r.desc.platform,
            r.desc.os,
            total,
            pass,
            fail,
            skip,
            xfail,
            r.status_text(),
        );
    }
    let _ = writeln!(
        out,
        "| **Total** | | {t_total} | {t_pass} | {t_fail} | {t_skip} | {t_xfail} | |",
    );

    out.push_str(
        "\n**Mock** runs the real `rocm` CLI, but fakes the inference backend: instead \
         of downloading a runtime and launching a real engine (vLLM/lemonade) on a \
         GPU, a tiny in-process HTTP server stands in for the OpenAI-compatible \
         endpoint that engine would expose (`/v1/models`, `/v1/chat/completions`). \
         This exercises CLI behaviour — alias resolution, service discovery, chat \
         forwarding — with no GPU, no model download, and no engine process, so it \
         runs on a GitHub-hosted runner. It **gates the PR**: it runs on every push, \
         and if it fails the PR's required check goes red and the PR cannot merge. \
         **MI300X / Strix Halo** run on real self-hosted GPU hardware with real \
         engines. They are **non-blocking**: they still run and are reported here, but \
         a failure does NOT block the PR from merging (the hardware/runners are still \
         being proven out, so their results are informational rather than a merge \
         gate).\n\n\
         Column meanings: **Pass** = scenarios that passed as expected; \
         **Xfail** = known bugs that failed as expected (healthy — the bug still \
         reproduces); **Skip** = not applicable on this platform (e.g. a GPU-only \
         scenario on Mock); **Fail** = unexpected — a scenario that should pass but \
         failed, or a known bug that unexpectedly passed (its tag is now stale). \
         **Status** is PASS unless a platform has any Fail.\n",
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

    // The per-(scenario × platform) expectation grid, when platform.json
    // sidecars are present (the new expectation system). Placed before the
    // command-coverage table; it supersedes the coarse platform×tier matrix for
    // "where should each test pass / not matter / not run".
    out.push_str(&expectation_grid_markdown(inputs));

    out.push_str(&command_coverage_markdown(&reports));

    out
}

/// Render the reconciled expectation grid as an HTML section (for the standalone
/// report). Empty markup when no `platform.json` sidecars are present.
fn expectation_grid_html(inputs: &[(String, PathBuf)]) -> Markup {
    let grid = Grid::build(inputs);
    if grid.is_empty() {
        return html! {};
    }
    let problems = grid.problems();
    html! {
        h2 { "Expectation grid (scenario × platform)" }
        table.stats {
            thead {
                tr {
                    th { "Scenario" }
                    @for col in &grid.columns {
                        th {
                            (col.slug)
                            @if !col.engine.is_empty() { br; small { (col.engine) } }
                        }
                    }
                }
            }
            tbody {
                @for id in &grid.ids {
                    tr {
                        td { code { (id) } }
                        @for col in &grid.columns {
                            @let outcome = col.outcomes.get(id).copied().unwrap_or(CellOutcome::Missing);
                            td.num
                              class=(if outcome.is_problem() { "status-fail" } else { "" }) {
                                (outcome.glyph())
                            }
                        }
                    }
                }
            }
        }
        @if !problems.is_empty() {
            h3 { "Needs attention" }
            ul {
                @for (slug, id, outcome, detail) in &problems {
                    li {
                        b {
                            @match outcome {
                                CellOutcome::Xpass => "XPASS",
                                CellOutcome::UnexpectedFail => "unexpected failure",
                                CellOutcome::RanWhenNa => "ran despite n/a",
                                _ => "issue",
                            }
                        }
                        " on " code { (slug) } ": " code { (id) }
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

/// Render the reconciled (scenario-id × platform) expectation grid as markdown,
/// plus a "needs attention" list of every XPASS / unexpected-fail / ran-when-NA.
/// Empty string when no `platform.json` sidecars are present.
fn expectation_grid_markdown(inputs: &[(String, PathBuf)]) -> String {
    use std::fmt::Write as _;

    let grid = Grid::build(inputs);
    if grid.is_empty() {
        return String::new();
    }

    let mut out = String::from("\n### Expectation grid (scenario × platform)\n\n");
    out.push_str(
        "_✅ pass · `xfail` known bug (failed as expected) · n/a not applicable here · \
         ❌FAIL regression · ⚠️XPASS bug fixed here (stale entry) · · no data._\n\n",
    );

    // Header: one column per platform, with its effective engine as context.
    out.push_str("| Scenario |");
    for col in &grid.columns {
        let eng = if col.engine.is_empty() {
            String::new()
        } else {
            format!("<br><sub>{}</sub>", col.engine)
        };
        let _ = write!(out, " {}{} |", col.slug, eng);
    }
    out.push('\n');
    out.push_str("|---|");
    for _ in &grid.columns {
        out.push_str(":--:|");
    }
    out.push('\n');

    for id in &grid.ids {
        let _ = write!(out, "| `{id}` |");
        for col in &grid.columns {
            let g = col
                .outcomes
                .get(id)
                .copied()
                .unwrap_or(CellOutcome::Missing);
            let _ = write!(out, " {} |", g.glyph());
        }
        out.push('\n');
    }

    // Needs-attention list from reconciliation problems.
    let problems = grid.problems();
    if !problems.is_empty() {
        out.push_str("\n### Needs attention\n\n");
        for (slug, id, outcome, detail) in problems {
            let bug = detail
                .and_then(|d| d.bug.as_deref())
                .map(|b| format!(" ({b})"))
                .unwrap_or_default();
            let engine = detail
                .map(|d| d.effective_engine.as_str())
                .filter(|e| !e.is_empty())
                .map(|e| format!(" [engine: {e}]"))
                .unwrap_or_default();
            let reason = detail
                .and_then(|d| d.reason.as_deref())
                .map(|r| format!(" — {r}"))
                .unwrap_or_default();
            let kind = match outcome {
                CellOutcome::Xpass => "XPASS",
                CellOutcome::UnexpectedFail => "unexpected failure",
                CellOutcome::RanWhenNa => "ran despite n/a",
                _ => "issue",
            };
            let _ = writeln!(out, "- **{kind}** on `{slug}`: `{id}`{bug}{engine}{reason}");
        }
    }

    out
}

/// A command signature: what we group invocations by in the coverage table.
///
/// `command` is the full invocation as executed; `engine` is the engine that
/// actually ran, with a "(default)" suffix when the CLI chose it itself. Grouping
/// on both keeps an explicit `--engine vllm` distinct from a default that
/// resolved to vLLM, and distinct from the same command resolving to lemonade on
/// another platform.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone)]
struct CommandKey {
    command: String,
    engine: String,
}

/// Build the "which rocm commands are exercised, with which models/engines, on
/// which platform, and do they work" coverage table.
///
/// For each (command, model, engine) × platform cell: ✅ if every scenario that
/// ran that command on that platform passed, ❌ if any failed, blank if the
/// command was never run there. "Passed" follows the scenario's own result, so a
/// command that is *supposed* to be rejected (its scenario asserts the failure)
/// still reads as ✅ — the tested behaviour held.
/// The `rocm` command surface we measure coverage against — the denominator.
///
/// Curated from the CLI's own `--help` tree (top-level subcommands and their
/// meaningful second-level subcommands), normalized to the `rocm <base>` shape
/// that `record_command`'s signature produces (see `derive_subcommand`). Pure
/// `help`/`completions` plumbing is intentionally excluded — they aren't product
/// behaviour worth an E2E. When the CLI gains a command, add it here so the
/// coverage % reflects the real surface (a deliberate, reviewable denominator
/// beats silently drifting).
const KNOWN_COMMAND_SURFACE: &[&str] = &[
    "rocm examine",
    "rocm diagnose",
    "rocm fix",
    "rocm version",
    "rocm setup status",
    "rocm setup reset",
    "rocm chat",
    "rocm install sdk",
    "rocm install driver",
    "rocm update",
    "rocm runtimes list",
    "rocm runtimes activate",
    "rocm runtimes rollback",
    "rocm runtimes uninstall",
    "rocm runtimes import",
    "rocm runtimes adopt",
    "rocm engines list",
    "rocm engines install",
    "rocm engines shell",
    "rocm model",
    "rocm serve",
    "rocm comfyui status",
    "rocm comfyui install",
    "rocm comfyui start",
    "rocm comfyui stop",
    "rocm comfyui logs",
    "rocm comfyui models-path",
    "rocm services list",
    "rocm services logs",
    "rocm services stop",
    "rocm services restart",
    "rocm automations list",
    "rocm automations enable",
    "rocm automations disable",
    "rocm config show",
    "rocm config set-engine",
    "rocm config set-default-engine",
    "rocm config set-default-runtime",
    "rocm config set-telemetry",
    "rocm config set-permissions",
    "rocm logs",
    "rocm dash",
    "rocm uninstall",
];

/// Normalize a recorded command signature to its base `rocm <base>` form for
/// matching against `KNOWN_COMMAND_SURFACE` — drops the behaviour-shaping
/// suffixes `record_command` appends (` --engine`, ` (default engine)`).
fn command_base(sig: &str) -> &str {
    sig.split(" --engine")
        .next()
        .unwrap_or(sig)
        .split(" (default engine)")
        .next()
        .unwrap_or(sig)
        .trim()
}

/// The `KNOWN_COMMAND_SURFACE` entry a recorded command exercises, if any.
///
/// A recorded base carries positionals the surface entry does not — e.g.
/// `rocm serve Qwen/Qwen2.5-1.5B-Instruct` exercises the surface command
/// `rocm serve`. Match by the LONGEST surface entry that is a word-boundary
/// prefix of the base, so a two-word command (`rocm install sdk`) wins over any
/// shorter prefix and `rocm serve <model>` maps to `rocm serve`.
fn matched_surface_command(base: &str) -> Option<&'static str> {
    KNOWN_COMMAND_SURFACE
        .iter()
        .copied()
        .filter(|cmd| base == *cmd || base.starts_with(&format!("{cmd} ")))
        .max_by_key(|cmd| cmd.len())
}

/// Coverage of the known command surface: `(covered, total, uncovered_sorted)`.
/// A command counts as covered if any platform ran a matching invocation.
fn command_coverage_summary(reports: &[PlatformReport]) -> (usize, usize, Vec<&'static str>) {
    use std::collections::BTreeSet;
    let mut exercised: BTreeSet<&'static str> = BTreeSet::new();
    for r in reports {
        for c in &r.commands {
            if let Some(cmd) = matched_surface_command(command_base(&c.subcommand)) {
                exercised.insert(cmd);
            }
        }
    }
    let uncovered: Vec<&'static str> = KNOWN_COMMAND_SURFACE
        .iter()
        .copied()
        .filter(|cmd| !exercised.contains(*cmd))
        .collect();
    let total = KNOWN_COMMAND_SURFACE.len();
    (total - uncovered.len(), total, uncovered)
}

fn command_coverage_markdown(reports: &[PlatformReport]) -> String {
    use std::collections::BTreeMap;
    use std::fmt::Write as _;

    // Platform columns in matrix order (platform+os), de-duplicated across tiers.
    let mut columns: Vec<String> = Vec::new();
    for r in reports {
        let col = format!("{} {}", r.desc.platform, r.desc.os);
        if !columns.contains(&col) {
            columns.push(col);
        }
    }

    // key → (column → all-passed-so-far). None = not run in that column.
    let mut cells: BTreeMap<CommandKey, BTreeMap<String, bool>> = BTreeMap::new();
    for r in reports {
        let col = format!("{} {}", r.desc.platform, r.desc.os);
        let passed = r.scenario_pass_map();
        for c in &r.commands {
            // Full command as executed; fall back to the stripped signature for
            // older artifacts that predate the `command` field.
            let command = c.command.clone().unwrap_or_else(|| c.subcommand.clone());
            // Engine actually used, with a "(default)" marker when the CLI chose
            // it (no explicit --engine flag).
            let engine = match c.engine.as_deref() {
                Some(e) if c.engine_is_default => format!("{e} (default)"),
                Some(e) => e.to_string(),
                None => String::new(),
            };
            let key = CommandKey { command, engine };
            // A command's cell is healthy only if EVERY scenario that ran it on
            // this platform passed; an unknown scenario is treated as passed
            // (the command ran and we have no failing evidence).
            let ok = c
                .scenario
                .as_deref()
                .and_then(|s| passed.get(s).copied())
                .unwrap_or(true);
            let entry = cells
                .entry(key)
                .or_default()
                .entry(col.clone())
                .or_insert(true);
            *entry = *entry && ok;
        }
    }

    if cells.is_empty() {
        return String::new();
    }

    let (covered, total, uncovered) = command_coverage_summary(reports);
    let pct = (covered * 100).checked_div(total).unwrap_or(0);

    let mut out = String::from("\n### Command coverage\n\n");
    let _ = writeln!(
        out,
        "**CLI surface coverage: {covered}/{total} commands ({pct}%)** exercised by \
         at least one platform.\n"
    );
    out.push_str("_Which `rocm` commands are exercised, with which engine, per platform. ");
    out.push_str(
        "✅ tested & behaved as expected · ❌ failed · blank = this command was not run on \
         this platform — usually because its scenario is not applicable here (skipped for \
         GPU/OS/engine), or the command is platform-specific by construction (e.g. a \
         host serves a different model/engine)._\n\n",
    );

    out.push_str("| Command | Engine |");
    for col in &columns {
        let _ = write!(out, " {col} |");
    }
    out.push('\n');
    out.push_str("|---|---|");
    for _ in &columns {
        out.push_str(":--:|");
    }
    out.push('\n');

    for (key, per_col) in &cells {
        let engine = if key.engine.is_empty() {
            "n/a"
        } else {
            &key.engine
        };
        let _ = write!(out, "| `{}` | {} |", key.command, engine);
        for col in &columns {
            let mark = match per_col.get(col) {
                Some(true) => " ✅ |",
                Some(false) => " ❌ |",
                None => " |",
            };
            out.push_str(mark);
        }
        out.push('\n');
    }

    // Fold-out list of the command surface NOT yet exercised by any platform, so
    // the coverage % is actionable rather than just a number.
    if !uncovered.is_empty() {
        let _ = write!(
            out,
            "\n<details><summary>Uncovered commands ({})</summary>\n\n",
            uncovered.len()
        );
        for cmd in &uncovered {
            let _ = writeln!(out, "- `{cmd}`");
        }
        out.push_str("\n</details>\n");
    }

    out
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
                    td { (r.desc.platform) }
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
                    b { "MI300X / Strix Halo" }
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

    #[test]
    fn scenario_passed_is_strict_and_shared() {
        // The unified predicate: only an all-steps-passed scenario counts as
        // passed. A skipped scenario is NOT a pass — both the CI gate and the
        // grid go through scenario_passed, so they can't diverge on this.
        assert!(scenario_passed(&scenario_from(&["passed", "passed"])));
        assert!(!scenario_passed(&scenario_from(&["passed", "skipped"])));
        assert!(!scenario_passed(&scenario_from(&["failed"])));
    }

    #[test]
    fn before_hook_failure_scores_scenario_failed() {
        // A failing Before hook leaves steps empty; without checking hooks the
        // scenario would fall through to "passed". It must score failed.
        let el: Element = serde_json::from_str(
            r#"{"name":"s","steps":[],"before":[{"result":{"status":"failed"}}]}"#,
        )
        .expect("valid element json");
        assert_eq!(scenario_status(&el), "failed");
        assert!(!scenario_passed(&el));

        // A passed Before hook + passed steps is still a pass.
        let ok: Element = serde_json::from_str(
            r#"{"name":"s","steps":[{"keyword":"Given ","name":"x","result":{"status":"passed"}}],"before":[{"result":{"status":"passed"}}]}"#,
        )
        .expect("valid element json");
        assert!(scenario_passed(&ok));
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
        assert!(md.contains("| Mock | Linux | 2 | 2 | 0 | 0 | n/a | PASS |"));
        assert!(md.contains("| MI300X | Linux | 1 | 0 | 1 | 0 | 1 | PASS |"));
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

    #[test]
    fn command_coverage_ties_to_scenario_not_rc() {
        // A scenario that PASSES while its command exited non-zero (e.g. an
        // adoption that is supposed to be rejected) must read ✅ — coverage
        // follows the scenario result, not the raw rc.
        let dir = tempfile::tempdir().expect("tempdir");
        let report = dir.path().join("report.json");
        // One scenario "s0" that passed, one "s1" that failed.
        std::fs::write(
            &report,
            feature_json(&[(&[], &["passed"]), (&[], &["failed"])]),
        )
        .expect("write report");
        // s0 ran `runtimes adopt` (rc=1 but scenario passed) → ✅.
        // s1 ran an explicit-engine `serve` (scenario failed) → ❌.
        // s0 also ran a default-engine `serve` whose engine the CLI resolved.
        std::fs::write(
            dir.path().join("commands.jsonl"),
            concat!(
                r#"{"scenario":"s0","subcommand":"rocm runtimes adopt","command":"rocm runtimes adopt","model":null,"engine":null,"rc":1}"#,
                "\n",
                r#"{"scenario":"s1","subcommand":"rocm serve --engine","command":"rocm serve Qwen --engine vllm","model":"Qwen","engine":"vllm","engine_is_default":false,"rc":0}"#,
                "\n",
                r#"{"scenario":"s0","subcommand":"rocm serve (default engine)","command":"rocm serve Qwen","model":"Qwen","engine":"lemonade","engine_is_default":true,"rc":0}"#,
                "\n",
            ),
        )
        .expect("write commands");

        let inputs = vec![("e2e-gpu-report".to_string(), report)];
        let md = consolidated_summary_markdown(&inputs);
        assert!(md.contains("### Command coverage"));
        // adoption: rc=1 but scenario passed → ✅
        assert!(
            md.contains("| `rocm runtimes adopt` | n/a | ✅ |"),
            "adopt should be ✅ (scenario passed despite rc=1):\n{md}"
        );
        // explicit-engine serve: full command shown, engine surfaced, scenario failed → ❌
        assert!(
            md.contains("| `rocm serve Qwen --engine vllm` | vllm | ❌ |"),
            "serve should show full command + engine and be ❌:\n{md}"
        );
        // default-engine serve: the CLI-resolved engine shows as "<engine> (default)"
        assert!(
            md.contains("| `rocm serve Qwen` | lemonade (default) | ✅ |"),
            "default serve should show resolved engine marked (default):\n{md}"
        );
    }

    #[test]
    fn cell_outcome_reconciliation() {
        use CellOutcome as C;
        assert_eq!(C::reconcile("pass", Some(true)), C::Pass);
        assert_eq!(C::reconcile("pass", Some(false)), C::UnexpectedFail);
        assert_eq!(C::reconcile("xfail", Some(false)), C::Xfail);
        assert_eq!(C::reconcile("xfail", Some(true)), C::Xpass);
        assert_eq!(C::reconcile("skip", None), C::Skip);
        assert_eq!(C::reconcile("skip", Some(true)), C::RanWhenNa);
        assert!(C::UnexpectedFail.is_problem());
        assert!(C::Xpass.is_problem());
        assert!(C::RanWhenNa.is_problem());
        assert!(!C::Pass.is_problem());
        assert!(!C::Xfail.is_problem());
        assert!(!C::Skip.is_problem());
    }

    /// Write a report.json + platform.json pair into a fresh dir and return the
    /// report.json path (the input the grid keys on).
    fn write_platform(report_json: &str, platform_json: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let report = dir.path().join("report.json");
        std::fs::write(&report, report_json).expect("write report");
        std::fs::write(dir.path().join("platform.json"), platform_json).expect("write platform");
        (dir, report)
    }

    #[test]
    fn grid_reconciles_xfail_and_pass_by_id() {
        // Scenario s0 tagged @id:serve-x, expected xfail, actually failed → xfail (good).
        // Scenario s1 tagged @id:examine-y, expected pass, actually passed → pass.
        let report = feature_json(&[
            (&["id:serve-x"], &["failed"]),
            (&["id:examine-y"], &["passed"]),
        ]);
        let platform = r#"{
            "platform_slug": "mi300x",
            "capability": {"effective_serve_engine": "vllm"},
            "expectations": [
                {"id":"serve-x","effective_engine":"vllm","expected":"xfail","bug":"EAI-7333","reason":"readiness gap"},
                {"id":"examine-y","effective_engine":"vllm","expected":"pass"}
            ]
        }"#;
        let (_d, path) = write_platform(&report, platform);
        let inputs = vec![("mi300x".to_string(), path)];
        let md = consolidated_summary_markdown(&inputs);
        assert!(md.contains("### Expectation grid"), "grid missing:\n{md}");
        assert!(
            md.contains("| `serve-x` | xfail |"),
            "serve-x should be xfail:\n{md}"
        );
        assert!(
            md.contains("| `examine-y` | ✅ |"),
            "examine-y should be pass:\n{md}"
        );
        // No problems → no needs-attention from the grid.
        assert!(!md.contains("**XPASS**"), "should have no XPASS:\n{md}");
    }

    #[test]
    fn grid_flags_xpass_when_known_bug_passes() {
        // s0 expected xfail but PASSED → XPASS (the run #543 Strix-Windows case).
        let report = feature_json(&[(&["id:serve-default"], &["passed"])]);
        let platform = r#"{
            "platform_slug": "strix-halo",
            "capability": {"effective_serve_engine": "lemonade"},
            "expectations": [
                {"id":"serve-default","effective_engine":"lemonade","expected":"xfail","bug":"EAI-7333","reason":"readiness gap"}
            ]
        }"#;
        let (_d, path) = write_platform(&report, platform);
        let inputs = vec![("strix-halo".to_string(), path)];
        let md = consolidated_summary_markdown(&inputs);
        assert!(md.contains("⚠️XPASS"), "grid cell should show XPASS:\n{md}");
        assert!(
            md.contains("**XPASS** on `strix-halo`: `serve-default` (EAI-7333)"),
            "needs-attention should list the XPASS with bug:\n{md}"
        );
    }

    #[test]
    fn grid_shows_skip_as_not_applicable() {
        // Scenario is skip on this host; report.json has no entry for it.
        let report = feature_json(&[(&["id:ran-here"], &["passed"])]);
        let platform = r#"{
            "platform_slug": "mock",
            "capability": {"effective_serve_engine": "lemonade"},
            "expectations": [
                {"id":"ran-here","effective_engine":"lemonade","expected":"pass"},
                {"id":"gpu-only","effective_engine":"vllm","expected":"skip","reason":"requires an AMD GPU"}
            ]
        }"#;
        let (_d, path) = write_platform(&report, platform);
        let inputs = vec![("mock".to_string(), path)];
        let md = consolidated_summary_markdown(&inputs);
        assert!(
            md.contains("| `gpu-only` | n/a |"),
            "skip should render as n/a:\n{md}"
        );
        assert!(md.contains("| `ran-here` | ✅ |"));
    }

    #[test]
    fn command_base_strips_suffixes() {
        assert_eq!(command_base("rocm serve --engine"), "rocm serve");
        assert_eq!(command_base("rocm serve (default engine)"), "rocm serve");
        assert_eq!(command_base("rocm install sdk"), "rocm install sdk");
    }

    #[test]
    fn matched_surface_command_maps_positionals_and_prefers_longest() {
        // Regression: a serve command embeds the model in its base, so it must
        // still map to the surface entry `rocm serve` (was counted uncovered).
        assert_eq!(
            matched_surface_command("rocm serve Qwen/Qwen2.5-1.5B-Instruct"),
            Some("rocm serve")
        );
        assert_eq!(
            matched_surface_command("rocm serve Qwen3-0.6B-GGUF"),
            Some("rocm serve")
        );
        // Longest-prefix wins: a two-word surface command is not shadowed by a
        // shorter one, and `rocm install sdk` maps to itself, not `rocm install`.
        assert_eq!(
            matched_surface_command("rocm install sdk"),
            Some("rocm install sdk")
        );
        // A bare exact match still works; an unknown command matches nothing.
        assert_eq!(
            matched_surface_command("rocm version"),
            Some("rocm version")
        );
        assert_eq!(matched_surface_command("rocm bogus"), None);
    }

    #[test]
    fn command_coverage_counts_against_known_surface() {
        // A report whose commands.jsonl exercised examine + serve (+ a serve
        // variant) → those count once against the known surface; total is the
        // full catalog; uncovered excludes what ran.
        let dir = tempfile::tempdir().expect("tempdir");
        let report = dir.path().join("report.json");
        std::fs::write(&report, feature_json(&[(&[], &["passed"])])).expect("write report");
        std::fs::write(
            dir.path().join("commands.jsonl"),
            concat!(
                r#"{"scenario":"s0","subcommand":"rocm examine","model":null,"engine":null,"rc":0}"#,
                "\n",
                r#"{"scenario":"s0","subcommand":"rocm serve Qwen/Qwen2.5-1.5B-Instruct --engine","model":"Qwen/Qwen2.5-1.5B-Instruct","engine":"vllm","rc":0}"#,
                "\n",
                r#"{"scenario":"s0","subcommand":"rocm serve Qwen3-0.6B-GGUF (default engine)","model":"Qwen3-0.6B-GGUF","engine":null,"rc":0}"#,
                "\n",
            ),
        )
        .expect("write commands");

        let reports = vec![PlatformReport::load("e2e-gpu-report".to_string(), &report)];
        let (covered, total, uncovered) = command_coverage_summary(&reports);
        assert_eq!(total, KNOWN_COMMAND_SURFACE.len());
        // examine + serve (both variants normalize to "rocm serve") = 2 covered.
        assert_eq!(covered, 2, "expected examine + serve covered");
        assert_eq!(total - covered, uncovered.len());
        assert!(uncovered.contains(&"rocm dash"), "dash should be uncovered");
        assert!(!uncovered.contains(&"rocm examine"));
        assert!(!uncovered.contains(&"rocm serve"));

        // The rendered markdown surfaces the % and the fold-out.
        let md = consolidated_summary_markdown(&[("e2e-gpu-report".to_string(), report)]);
        assert!(
            md.contains("CLI surface coverage:"),
            "coverage line missing:\n{md}"
        );
        assert!(
            md.contains("Uncovered commands ("),
            "uncovered fold missing:\n{md}"
        );
    }

    #[test]
    fn grid_absent_without_platform_json() {
        // Old-style artifact (report.json only) → no grid section.
        let report = write_report(&feature_json(&[(&[], &["passed"])]));
        let inputs = vec![("e2e-report".to_string(), report.path().to_path_buf())];
        let md = consolidated_summary_markdown(&inputs);
        assert!(
            !md.contains("### Expectation grid"),
            "no grid expected:\n{md}"
        );
    }

    #[test]
    fn summary_status_reconciles_xfail_as_pass() {
        // Regression (run 29209242248): a platform whose only junit failures are
        // known-bug xfails was shown FAIL in the summary while the grid said
        // clean. With a platform.json, the summary Status must reconcile: an
        // xfail is healthy, so the row is PASS and its Fail count is 0.
        let report = feature_json(&[
            (&["id:serve-short-name"], &["failed"]), // known bug, expected xfail
            (&["id:examine-version"], &["passed"]),
        ]);
        let platform = r#"{
            "platform_slug": "mock",
            "capability": {"effective_serve_engine": "lemonade"},
            "expectations": [
                {"id":"serve-short-name","effective_engine":"lemonade","expected":"xfail","bug":"EAI-7219","reason":"alias not forwarded"},
                {"id":"examine-version","effective_engine":"lemonade","expected":"pass"}
            ]
        }"#;
        let (_d, path) = write_platform(&report, platform);
        let r = PlatformReport::load("e2e-report".into(), &path);
        assert!(r.ok(), "xfail-only platform should be ok");
        assert_eq!(r.status_text(), "PASS");
        // Fail column is the reconciled problem count (0), not raw junit (1);
        // the failed known bug is counted as xfail instead.
        let (total, pass, fail, _skip, xfail) = r.display_counts();
        assert_eq!((total, pass, fail, xfail), (2, 1, 0, 1));

        let inputs = vec![("e2e-report".to_string(), path)];
        let md = consolidated_summary_markdown(&inputs);
        assert!(
            md.contains("| Mock | Linux | 2 | 1 | 0 | 0 | 1 | PASS |"),
            "summary row should be reconciled PASS with 0 fails:\n{md}"
        );
    }

    #[test]
    fn summary_status_reconciles_unexpected_fail_as_fail() {
        // The honest-red half: an expect-pass scenario that actually failed is an
        // unexpected failure → the platform's summary Status is FAIL.
        let report = feature_json(&[(&["id:serve-default-engine"], &["failed"])]);
        let platform = r#"{
            "platform_slug": "strix-halo-linux",
            "capability": {"effective_serve_engine": "lemonade"},
            "expectations": [
                {"id":"serve-default-engine","effective_engine":"lemonade","expected":"pass"}
            ]
        }"#;
        let (_d, path) = write_platform(&report, platform);
        let r = PlatformReport::load("e2e-gpu-strix-ubuntu-report".into(), &path);
        assert!(!r.ok(), "an unexpected fail must red the platform");
        assert_eq!(r.status_text(), "FAIL");
        let (_total, _pass, fail, _skip, _xfail) = r.display_counts();
        assert_eq!(fail, 1);
    }
}
