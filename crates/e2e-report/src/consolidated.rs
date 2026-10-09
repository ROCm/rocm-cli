// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Multi-platform consolidated report: the `PlatformReport`/manifest/expectation
//! model, the reconciled (scenario × platform) `Grid`, and the HTML/markdown
//! generation built on top of them.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use maud::{Markup, html};
use serde::Deserialize;

use crate::command_coverage::command_coverage_markdown;
use crate::components::{feature_group, now_utc, page_shell, stats_bar};
use crate::parse::{
    EXPECTED_FAILURE_TAG, Feature, Stats, XfailReport, evaluate_xfail_features, parse_features,
    scenario_id, scenario_passed, stats_of,
};

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
    /// The channel suffix stripped from the artifact name, if any. Only a
    /// fallback: [`PlatformReport::effective_channel`] prefers the manifest's
    /// `versions.channel` and uses this solely for the errored-run case where
    /// no `platform.json` was ever written.
    channel_from_name: Option<&'static str>,
}

/// Parse an artifact/dir name like `e2e-gpu-strix-windows-report`
/// into its Platform / OS. Unknown shapes fall back to a titlecased
/// platform on Linux so a new artifact still renders sensibly.
fn parse_descriptor(name: &str) -> Descriptor {
    // Strip prefix, then suffix, each relative to the prior result (not `name`),
    // so `e2e-report` correctly reduces to the empty core, not back to itself.
    let core = name.strip_prefix("e2e-").unwrap_or(name);
    let core = core.strip_suffix("-report").unwrap_or(core);

    // The nightly workflow always appends the channel as the final segment
    // before `-report` (e.g. `e2e-gpu-strix-windows-nightly-report`). Captured
    // as `channel_from_name` — a fallback for the errored-run case where no
    // `platform.json` was written, so the manifest has no channel to read.
    let (core, channel_from_name) = match core.strip_suffix("-release") {
        Some(rest) => (rest, Some("release")),
        None => match core.strip_suffix("-nightly") {
            Some(rest) => (rest, Some("nightly")),
            None => (core, None),
        },
    };

    // Legacy `-known-bugs` suffix (retained only as a stable secondary sort key).
    let (core, known_bugs) = match core.strip_suffix("known-bugs") {
        Some(rest) => (rest.trim_end_matches('-'), true),
        None => (core, false),
    };

    let (platform, os) = match core {
        // The bare mock expect-pass artifact is `e2e-report` → core "report" or "".
        "" | "report" => ("Mock", "Linux"),
        "gpu" => ("MI300X", "Linux"),
        "gpu-rad3" => ("R9700", "Linux"),
        "gpu-mi350p" => ("MI350P", "Linux"),
        "gpu-strix-ubuntu" => ("Strix Halo", "Ubuntu"),
        "gpu-strix-windows" => ("Strix Halo", "Windows"),
        // Same silicon again, third host boundary: an Ubuntu distro under WSL2 on
        // the Windows box. It is neither the native Ubuntu nor the Windows lane —
        // GPU access goes through the WSL passthrough — so it needs its own OS
        // value, not a reuse of "Ubuntu", or the grid would show two rows that
        // claim to be the same host and disagree.
        "gpu-strix-wsl" => ("Strix Halo", "WSL2"),
        // `e2e-unknown-report`: a report whose platform.json sidecar was missing
        // or unrecognized (e.g. a GPU run that errored before writing it). The OS
        // is genuinely unknown here — a Windows GPU run that erupted early must NOT
        // be reported as Linux — so render Unknown / Unknown rather than defaulting
        // OS to Linux the way `fallback_descriptor` does for a titlecased platform.
        "unknown" => ("Unknown", "Unknown"),
        other => return fallback_descriptor(other, known_bugs, channel_from_name),
    };

    Descriptor {
        platform: platform.to_string(),
        os: os.to_string(),
        known_bugs,
        channel_from_name,
    }
}

fn fallback_descriptor(
    core: &str,
    known_bugs: bool,
    channel_from_name: Option<&'static str>,
) -> Descriptor {
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
        channel_from_name,
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
pub(crate) struct PlatformReport {
    desc: Descriptor,
    /// Human label for the per-platform detail sections and the markdown
    /// XPASS/Regression notes. Also the de-duplication key for
    /// command-coverage columns across a platform's tiers (see
    /// `command_coverage.rs`) — a display-only change to this field (e.g. a
    /// known-bugs marker) would silently split or merge those columns, so
    /// keep that consumer in mind alongside the other two. `pub(crate)` so
    /// `command_coverage.rs` can reuse it rather than reaching into
    /// `desc`/`versions` itself.
    pub(crate) label: String,
    features: Vec<Feature>,
    stats: Stats,
    xfail: XfailReport,
    /// True when the report contains any `@expected-failure` scenario — i.e. it
    /// is a known-bugs run, whose health follows xfail inversion rather than a
    /// plain zero-failures rule.
    is_known_bugs: bool,
    /// Recorded `rocm` invocations from this platform's `commands.jsonl`.
    pub(crate) commands: Vec<CommandRecord>,
    /// Expectation-reconciled outcome (`platform.json` × `report.json` by `@id`).
    /// `None` for pre-expectation artifacts, which fall back to the junit status.
    tally: Option<ReconciledTally>,
    /// Component versions (OS/ROCm/vLLM/lemonade) from `platform.json`, for the
    /// summary-matrix Platform/OS cells. Default (all `None`) for older artifacts.
    versions: PlatformVersions,
}

/// One recorded `rocm` invocation from a platform's `commands.jsonl` sidecar.
#[derive(Deserialize)]
pub(crate) struct CommandRecord {
    pub(crate) scenario: Option<String>,
    pub(crate) subcommand: String,
    /// Full command as executed (e.g. "rocm serve Qwen/... --engine vllm").
    /// Falls back to `subcommand` for older artifacts that predate this field.
    #[serde(default)]
    pub(crate) command: Option<String>,
    #[serde(default)]
    pub(crate) engine: Option<String>,
    /// True when `engine` was the CLI's own default choice (no `--engine` flag),
    /// so the report can show it as "<engine> (default)".
    #[serde(default)]
    pub(crate) engine_is_default: bool,
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
    /// Component versions (OS/ROCm/vLLM/lemonade); absent in older artifacts.
    #[serde(default)]
    versions: PlatformVersions,
    #[serde(default)]
    expectations: Vec<ManifestExpectation>,
}

#[derive(Deserialize)]
struct ManifestCapability {
    #[serde(default)]
    effective_serve_engine: String,
}

/// Per-platform component versions, mirrored from the harness `platform.json`.
/// All optional — a source not present on a platform is simply omitted.
#[derive(Deserialize, Default, Clone)]
struct PlatformVersions {
    #[serde(default)]
    os: Option<String>,
    #[serde(default)]
    rocm: Option<String>,
    #[serde(default)]
    vllm: Option<String>,
    #[serde(default)]
    lemonade: Option<String>,
    /// Release channel ("release" | "nightly") the platform's active runtime was
    /// installed under. `None` on artifacts predating the field, or when no
    /// managed runtime was active — those render as a single unlabelled column.
    #[serde(default)]
    channel: Option<String>,
}

impl PlatformVersions {
    /// Rendered "os X · ROCm Y · vLLM Z · lemonade W" line for a column heading,
    /// skipping absent components. Empty when nothing is known.
    fn summary(&self) -> String {
        [
            self.os.as_deref().map(|v| format!("OS {v}")),
            self.rocm.as_deref().map(|v| format!("ROCm {v}")),
            self.vllm.as_deref().map(|v| format!("vLLM {v}")),
            self.lemonade.as_deref().map(|v| format!("lemonade {v}")),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" · ")
    }

    /// The software-stack versions (ROCm/vLLM/lemonade, WITHOUT the OS — that goes
    /// in its own cell), for the summary-matrix Platform cell. Empty when none are
    /// known (e.g. mock, which has no installed runtime).
    fn platform_stack(&self) -> String {
        [
            self.rocm.as_deref().map(|v| format!("ROCm {v}")),
            self.vllm.as_deref().map(|v| format!("vLLM {v}")),
            self.lemonade.as_deref().map(|v| format!("lemonade {v}")),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" · ")
    }
}

/// One scenario's resolved expectation, keyed by its stable `@id`.
#[derive(Deserialize, Clone)]
struct ManifestExpectation {
    id: String,
    /// The `Feature:` this scenario belongs to. Absent in artifacts predating
    /// the grouped grid — see [`feature_of`] for the fallback.
    #[serde(default)]
    feature: String,
    /// The scenario's own name (`<key>-<NN> - <description>`), carrying the
    /// per-feature index rows are sorted by. Absent in older artifacts.
    #[serde(default)]
    scenario: String,
    #[serde(default)]
    effective_engine: String,
    /// "pass" | "xfail" | "skip".
    expected: String,
    #[serde(default)]
    bug: Option<String>,
    #[serde(default)]
    reason: Option<String>,
    /// Non-deterministic known bug: an XPASS is expected and non-fatal.
    #[serde(default)]
    flaky: bool,
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
    /// Expected deterministic xfail, but PASSED — stale entry (bug fixed here?).
    Xpass,
    /// Flaky xfail passed this run — expected for a non-deterministic known bug.
    FlakyXpass,
    /// Expected skip, yet a result exists — harness/resolver disagreement.
    RanWhenNa,
    /// Expected to run (pass or xfail) but NO result was recorded — the scenario
    /// never reported (hung / cancelled / lost report.json). A problem: a
    /// lost-results run must not be indistinguishable from all-pass.
    Absent,
    /// No expectation and no result recorded for this id on this platform.
    Missing,
}

impl CellOutcome {
    /// Reconcile one scenario's expectation against its actual result.
    /// `actual` is `Some(passed)` when the scenario ran, `None` when it did not
    /// appear in `report.json` (filtered out / skipped).
    fn reconcile(expected: &str, flaky: bool, actual: Option<bool>) -> Self {
        match (expected, actual) {
            ("pass", Some(true)) => Self::Pass,
            ("pass", Some(false)) => Self::UnexpectedFail,
            ("pass", None) => Self::Absent,
            ("xfail", Some(false)) => Self::Xfail,
            ("xfail", Some(true)) if flaky => Self::FlakyXpass,
            ("xfail", Some(true)) => Self::Xpass,
            ("xfail", None) => Self::Absent,
            ("skip", None) => Self::Skip,
            ("skip", Some(_)) => Self::RanWhenNa,
            _ => Self::Missing,
        }
    }

    /// True when this cell needs human attention (report FAILs on any).
    const fn is_problem(self) -> bool {
        matches!(
            self,
            Self::UnexpectedFail | Self::Xpass | Self::RanWhenNa | Self::Absent
        )
    }

    const fn glyph(self) -> &'static str {
        match self {
            Self::Pass => "✅",
            // A grey ✗ (styled `status-xfail` in HTML): a known bug that failed as
            // expected — visually a muted sibling of the red ❌ regression mark.
            Self::Xfail => "✗",
            Self::Skip => "n/a",
            Self::UnexpectedFail => "❌FAIL",
            Self::Xpass => "⚠️XPASS",
            Self::FlakyXpass => "✅XPASS (flaky)",
            Self::RanWhenNa => "⚠️n/a-ran",
            Self::Absent => "⚠️no-result",
            Self::Missing => "·",
        }
    }

    /// CSS class for this cell in the HTML grid: red for a problem, grey for an
    /// xfail (known bug, healthy), none otherwise.
    const fn grid_class(self) -> &'static str {
        if self.is_problem() {
            "status-fail"
        } else if matches!(self, Self::Xfail) {
            "status-xfail"
        } else {
            ""
        }
    }
}

/// One (outcome, human explanation) entry per glyph, in the grid legend's
/// display order. Shared by the HTML and Markdown legends so a new
/// `CellOutcome` variant can't update one copy and leave the other stale.
const LEGEND_ENTRIES: &[(CellOutcome, &str)] = &[
    (CellOutcome::Pass, "pass"),
    (CellOutcome::Xfail, "known bug, failed as expected (xfail)"),
    (CellOutcome::Skip, "not applicable here"),
    (CellOutcome::UnexpectedFail, "regression"),
    (CellOutcome::Xpass, "bug fixed here (stale entry)"),
    (CellOutcome::FlakyXpass, "known flaky bug passed this run"),
    (CellOutcome::RanWhenNa, "ran despite being marked n/a"),
    (
        CellOutcome::Absent,
        "expected to run but no result recorded",
    ),
    (CellOutcome::Missing, "no data."),
];

/// One platform column of the reconciled (scenario-id × platform) grid.
struct GridColumn {
    /// Platform identity from the manifest (e.g. "mi300x", "strix-halo", "mock").
    slug: String,
    /// Effective serve engine on this host (for the column subheading).
    engine: String,
    /// Component versions (OS/ROCm/vLLM/lemonade) for the column heading.
    versions: PlatformVersions,
    /// scenario id → reconciled outcome.
    outcomes: std::collections::BTreeMap<String, CellOutcome>,
    /// Per-id bug/reason, surfaced in the "needs attention" list.
    details: std::collections::BTreeMap<String, ManifestExpectation>,
}

/// Append a "(channel)" suffix to `base` when known.
fn with_channel_suffix(base: &str, channel: Option<&str>) -> String {
    match channel {
        Some(c) => format!("{base} ({c})"),
        None => base.to_string(),
    }
}

impl GridColumn {
    /// Column heading label: the slug, plus its channel when known — the part of
    /// column identity that doesn't already show up in `versions.summary()`.
    fn label(&self) -> String {
        with_channel_suffix(&self.slug, self.versions.channel.as_deref())
    }
}

/// One row of the grid: a scenario, with the identity used to place and order it.
struct GridRow {
    id: String,
    /// The scenario's human name, or empty when unknown (an artifact predating
    /// the `scenario` field whose scenario ran nowhere).
    name: String,
    /// Per-feature index parsed from the `<key>-<NN>` name prefix. `None` when
    /// the name is absent or unindexed — those rows sort last, by id.
    index: Option<u32>,
}

/// Scenarios of one `Feature:`, in display order.
struct FeatureGroup {
    feature: String,
    rows: Vec<GridRow>,
}

/// The reconciled grid: scenario rows grouped by feature × platform columns.
/// Built from each input's `platform.json` (expected) joined with its
/// `report.json` (actual) by stable `@id`. Inputs without a `platform.json`
/// (pre-expectation artifacts) are skipped here — they still appear in the legacy
/// platform×tier matrix.
struct Grid {
    /// Feature groups, alphabetical by feature name; rows within a group ordered
    /// by their `<key>-<NN>` index (i.e. feature-file order).
    groups: Vec<FeatureGroup>,
    columns: Vec<GridColumn>,
}

/// The per-feature index in a scenario name like `serve-07 - Something happens`.
/// `None` for a name that doesn't carry one (older artifact, or a scenario
/// renamed out of the convention).
fn scenario_index(name: &str) -> Option<u32> {
    let head = name.split(" - ").next()?;
    let (_key, digits) = head.rsplit_once('-')?;
    digits.parse().ok()
}

/// The feature a scenario belongs to, best-effort.
///
/// 1. `platform.json`'s own `feature` — the honest source, and the only one that
///    covers a scenario skipped on every platform.
/// 2. The feature name from a `report.json` that ran it.
/// 3. Failing both, the id's leading segment (`serve-vllm-inference` → `serve`),
///    so a pre-expectation artifact still groups sensibly instead of collapsing
///    into one bucket.
fn feature_of(exp: &ManifestExpectation, from_reports: Option<&str>) -> String {
    if !exp.feature.is_empty() {
        return exp.feature.clone();
    }
    if let Some(name) = from_reports.filter(|n| !n.is_empty()) {
        return name.to_owned();
    }
    exp.id
        .split_once('-')
        .map_or_else(|| exp.id.clone(), |(key, _)| key.to_owned())
}

impl Grid {
    fn build(inputs: &[(String, PathBuf)]) -> Self {
        // Feature names for ids that ran somewhere, as a fallback for artifacts
        // whose platform.json predates the `feature` field.
        let features_from_reports = id_features(inputs);
        // id → (feature, scenario name), merged across inputs. A later input can
        // fill in identity an earlier (older) artifact lacked.
        let mut identity: BTreeMap<String, (String, String)> = BTreeMap::new();
        let mut columns: Vec<GridColumn> = Vec::new();

        for (_label, json_path) in inputs {
            let Some(manifest) = parse_platform_manifest(json_path) else {
                continue;
            };
            // Actual results by id from this platform's report.json.
            let actual = id_pass_map(json_path);

            // Merge into an existing column with the same (slug, channel) —
            // channel is part of column identity so a nightly and a release run
            // of the same platform land in separate columns instead of one
            // colliding into the other (defensive; with one job per platform per
            // channel there is exactly one input per (slug, channel)).
            #[allow(clippy::suspicious_operation_groupings)]
            // field names legitimately differ: GridColumn::slug vs PlatformManifest::platform_slug
            let col_idx = columns
                .iter()
                .position(|c| {
                    c.slug == manifest.platform_slug
                        && c.versions.channel == manifest.versions.channel
                })
                .unwrap_or_else(|| {
                    columns.push(GridColumn {
                        slug: manifest.platform_slug.clone(),
                        engine: manifest
                            .capability
                            .as_ref()
                            .map(|c| c.effective_serve_engine.clone())
                            .unwrap_or_default(),
                        versions: manifest.versions.clone(),
                        outcomes: std::collections::BTreeMap::new(),
                        details: std::collections::BTreeMap::new(),
                    });
                    columns.len() - 1
                });

            for exp in &manifest.expectations {
                let entry = identity
                    .entry(exp.id.clone())
                    .or_insert_with(|| (String::new(), String::new()));
                // An artifact that names the feature itself is authoritative and
                // OVERWRITES whatever a fallback guessed — `feature_of` always
                // yields something (worst case the id's prefix), so a mere
                // is-empty check would let the first input's guess stick and the
                // later, better-informed artifact be ignored.
                if exp.feature.is_empty() {
                    if entry.0.is_empty() {
                        entry.0 =
                            feature_of(exp, features_from_reports.get(&exp.id).map(String::as_str));
                    }
                } else {
                    entry.0.clone_from(&exp.feature);
                }
                // The display name needs no fallback — unlike `feature` it is
                // never synthesized, so an absent one is simply empty and the
                // first artifact that HAS a name supplies it either way.
                // The only case the two rules differ on is two artifacts naming
                // the same id differently (mixing vintages across a rename):
                // take the last, matching `feature` above. The name carries the
                // sort index, so the rule should at least be stated rather than
                // falling out of map iteration order.
                if !exp.scenario.is_empty() {
                    entry.1.clone_from(&exp.scenario);
                }
                let outcome =
                    CellOutcome::reconcile(&exp.expected, exp.flaky, actual.get(&exp.id).copied());
                // A real result supersedes a defensive Missing on merge, and a
                // problem outcome is never silently displaced by a clean one for
                // the same id in the same column — a second, conflicting result
                // must stay visible rather than being masked by the first. A
                // second, *differing* problem outcome for the same id/column
                // (e.g. Failed then Flaky) still keeps whichever arrived first,
                // since neither `and_modify` branch matches once `o` is already
                // a problem.
                columns[col_idx]
                    .outcomes
                    .entry(exp.id.clone())
                    .and_modify(|o| {
                        if *o == CellOutcome::Missing || (outcome.is_problem() && !o.is_problem()) {
                            *o = outcome;
                        }
                    })
                    .or_insert(outcome);
                columns[col_idx].details.insert(exp.id.clone(), exp.clone());
            }
        }

        // Group by feature, then order rows by their per-feature index so the
        // grid reads in feature-file order rather than the alphabetical-by-id
        // mash the flat table used to show. Unindexed rows sort last, by id.
        let mut by_feature: BTreeMap<String, Vec<GridRow>> = BTreeMap::new();
        for (id, (feature, name)) in identity {
            let index = scenario_index(&name);
            by_feature
                .entry(feature)
                .or_default()
                .push(GridRow { id, name, index });
        }
        let groups = by_feature
            .into_iter()
            .map(|(feature, mut rows)| {
                rows.sort_by(|a, b| {
                    (a.index.is_none(), a.index, &a.id).cmp(&(b.index.is_none(), b.index, &b.id))
                });
                FeatureGroup { feature, rows }
            })
            .collect();
        Self { groups, columns }
    }

    /// Every problem cell across the grid, as `(slug, id, outcome, detail)`.
    fn problems(&self) -> Vec<(String, &str, CellOutcome, Option<&ManifestExpectation>)> {
        let mut out = Vec::new();
        for col in &self.columns {
            for (id, outcome) in &col.outcomes {
                if outcome.is_problem() {
                    out.push((col.label(), id.as_str(), *outcome, col.details.get(id)));
                }
            }
        }
        out
    }

    const fn is_empty(&self) -> bool {
        self.columns.is_empty() || self.groups.is_empty()
    }
}

/// Map each scenario `@id` → the name of the feature that ran it, from every
/// input's `report.json`. The fallback used when a `platform.json` predates the
/// `feature` field.
fn id_features(inputs: &[(String, PathBuf)]) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for (_label, json_path) in inputs {
        for f in parse_features(json_path) {
            for el in &f.elements {
                if let Some(id) = scenario_id(el) {
                    map.entry(id).or_insert_with(|| f.name.clone());
                }
            }
        }
    }
    map
}

/// Map each scenario's stable `@id` → whether it passed, from a `report.json`.
/// (Internal sibling of the public [`crate::parse::scenario_results_by_id`],
/// returning a map.)
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
        match CellOutcome::reconcile(&exp.expected, exp.flaky, actual.get(&exp.id).copied()) {
            CellOutcome::Pass => t.pass += 1,
            CellOutcome::Xfail => t.xfail += 1,
            CellOutcome::Skip => t.skip += 1,
            CellOutcome::FlakyXpass => t.pass += 1,
            CellOutcome::UnexpectedFail
            | CellOutcome::Xpass
            | CellOutcome::RanWhenNa
            | CellOutcome::Absent => {
                t.problems += 1;
            }
            // No expectation AND no result — nothing declared to reconcile. (A
            // DECLARED expectation with no result is `Absent`, counted above, so a
            // lost-results run reds the platform instead of silently passing.)
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
        let commands = parse_commands(json_path);
        let tally = reconciled_tally(json_path);
        let versions = parse_platform_manifest(json_path)
            .map(|m| m.versions)
            .unwrap_or_default();
        // Manifest channel wins when known; the artifact-name channel is only a
        // fallback for the errored-run case where no `platform.json` was written.
        let effective_channel = versions.channel.as_deref().or(desc.channel_from_name);
        let label =
            with_channel_suffix(&format!("{} {}", desc.platform, desc.os), effective_channel);
        Self {
            desc,
            label,
            features,
            stats,
            xfail,
            is_known_bugs,
            commands,
            tally,
            versions,
        }
    }

    /// The channel that governs this platform's rendering: the manifest's
    /// `versions.channel` when known, falling back to the channel captured
    /// from the artifact name for the errored-run case where no
    /// `platform.json` was ever written.
    fn effective_channel(&self) -> Option<&str> {
        self.versions
            .channel
            .as_deref()
            .or(self.desc.channel_from_name)
    }

    /// Map each scenario name → whether it passed. Uses the canonical
    /// `scenario_passed` predicate (every step passed) so the command-coverage
    /// table can never diverge from the CI gate / grid — a `skipped` scenario is
    /// NOT a pass (see `scenario_passed`).
    pub(crate) fn scenario_pass_map(&self) -> std::collections::HashMap<String, bool> {
        self.features
            .iter()
            .flat_map(|f| &f.elements)
            .map(|el| (el.name.clone(), scenario_passed(el)))
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

/// Order platforms by platform/OS, then known-bugs flag, then channel — shared
/// between the HTML report and the markdown summary so a future ordering
/// tweak can't land in only one of them.
fn platform_sort_key(r: &PlatformReport) -> (&str, &str, bool, Option<&str>) {
    (
        &r.desc.platform,
        &r.desc.os,
        r.desc.known_bugs,
        r.effective_channel(),
    )
}

/// Platform cell for a table that renders OS in its own column (so this omits
/// `desc.os`, unlike `PlatformReport::label`) — shared between the markdown
/// summary and `matrix_table` so the two can't drift the way they already
/// have once (the markdown/HTML drift fixed by #460).
fn platform_cell(r: &PlatformReport) -> String {
    with_channel_suffix(&r.desc.platform, r.effective_channel())
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
    reports.sort_by(|a, b| platform_sort_key(a).cmp(&platform_sort_key(b)));

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
    };
    let markup = page_shell("Consolidated E2E Report", body);

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
    reports.sort_by(|a, b| platform_sort_key(a).cmp(&platform_sort_key(b)));

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
        // Component versions in the Platform/OS cells: ROCm/vLLM/lemonade under the
        // platform, the OS version under the OS. Absent components are omitted (mock
        // has no runtime; a not-yet-probed source is simply skipped).
        let plat_base = platform_cell(r);
        let plat_cell = match r.versions.platform_stack() {
            s if s.is_empty() => plat_base,
            s => format!("{plat_base}<br><sub>{s}</sub>"),
        };
        let os_cell = match r.versions.os.as_deref() {
            Some(v) if v != r.desc.os => format!("{}<br><sub>{}</sub>", r.desc.os, v),
            _ => r.desc.os.clone(),
        };
        // `writeln!` into a String never fails; the discard keeps clippy happy.
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            plat_cell,
            os_cell,
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
         **Every other platform** is a real self-hosted GPU host running real \
         engines — the rows below are the list, so naming them here would only go \
         stale as lanes are added. They are **non-blocking**: they still run and \
         are reported here, but \
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

    // id → (scenario name, Gherkin steps), built once from report.json across all
    // inputs. Used both for the grid's Scenario column (name + id link) and the
    // Scenario reference section at the very end.
    let scenarios = scenario_reference(inputs);

    // The per-(scenario × platform) expectation grid, when platform.json
    // sidecars are present (the new expectation system). Placed before the
    // command-coverage table; it supersedes the coarse platform×tier matrix for
    // "where should each test pass / not matter / not run".
    out.push_str(&expectation_grid_markdown(inputs, &scenarios));

    out.push_str(&command_coverage_markdown(&reports));

    // Scenario reference LAST (after command coverage): each id's actual Gherkin
    // scenario, anchored so the grid's id links resolve here.
    out.push_str(&scenario_reference_markdown(inputs, &scenarios));

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
fn expectation_grid_markdown(
    inputs: &[(String, PathBuf)],
    scenarios: &std::collections::BTreeMap<String, (String, Vec<String>)>,
) -> String {
    use std::fmt::Write as _;

    let grid = Grid::build(inputs);
    if grid.is_empty() {
        return String::new();
    }

    let mut out = String::from("\n### Expectation grid (scenario × platform)\n\n");
    let legend = LEGEND_ENTRIES
        .iter()
        .map(|(outcome, text)| format!("{} {text}", outcome.glyph()))
        .collect::<Vec<_>>()
        .join(" · ");
    let _ = writeln!(out, "_{legend}_\n");

    // One table per feature, under its own heading — a single undivided table of
    // every scenario in the suite is unreadable, and gives no clue where one area
    // of the CLI ends and the next begins.
    for group in &grid.groups {
        let _ = writeln!(out, "#### {}\n", group.feature);

        // Header: one column per platform, with its effective engine. Component
        // versions live in the summary matrix above, not here.
        out.push_str("| Scenario |");
        for col in &grid.columns {
            let eng = if col.engine.is_empty() {
                String::new()
            } else {
                format!("<br><sub>{}</sub>", col.engine)
            };
            let _ = write!(out, " {}{} |", col.label(), eng);
        }
        out.push('\n');
        out.push_str("|---|");
        for _ in &grid.columns {
            out.push_str(":--:|");
        }
        out.push('\n');

        for row in &group.rows {
            // Scenario cell: human name on top, the @id below as a link to its
            // entry in the Scenario reference section (GitHub anchors
            // `##### <id>` to `#<id>`). Prefer the name recorded in platform.json
            // — it covers a scenario that was skipped everywhere and so has no
            // report.json entry to read a name from.
            let name = if row.name.is_empty() {
                scenarios.get(&row.id).map_or("", |(n, _)| n.as_str())
            } else {
                row.name.as_str()
            };
            let id = &row.id;
            let _ = write!(out, "| {name}<br>[`{id}`](#{id}) |");
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

/// Render the Scenario reference section: each scenario `@id` with its actual
/// Gherkin scenario (name + steps), anchored by the id (`##### <id>` → GitHub
/// anchor `#<id>`) so the grid's id links resolve. Grouped by feature and ordered
/// like the grid. Empty when no expectation grid exists.
fn scenario_reference_markdown(
    inputs: &[(String, PathBuf)],
    scenarios: &std::collections::BTreeMap<String, (String, Vec<String>)>,
) -> String {
    use std::fmt::Write as _;

    // Only emit when there is a grid to reference (platform.json sidecars present),
    // matching where the id links are generated.
    let grid = Grid::build(inputs);
    if grid.is_empty() {
        return String::new();
    }

    // Walk the grid's own order so the reference is laid out feature by feature,
    // matching the tables that link into it. Every grid row gets an entry — a
    // scenario skipped on every platform has no steps to show, but still needs
    // its anchor or the grid's link to it dangles.
    //
    // Scoped to grid rows deliberately: this section exists to back the grid's
    // links, so it documents exactly what the grid shows. A scenario present only
    // in an artifact with no `platform.json` sidecar has no grid row and so gets
    // no entry — nothing links to it either.
    let mut out = String::from("\n### Scenario reference\n\n");
    for group in &grid.groups {
        let _ = writeln!(out, "#### {}\n", group.feature);
        for row in &group.rows {
            let entry = scenarios.get(&row.id);
            // Prefer the platform.json name, as the grid does — it is the only
            // source that covers a scenario which ran nowhere.
            let name = if row.name.is_empty() {
                entry.map_or("", |(n, _)| n.as_str())
            } else {
                row.name.as_str()
            };
            let _ = writeln!(out, "##### {}\n", row.id);
            if !name.is_empty() {
                let _ = writeln!(out, "_{name}_\n");
            }
            match entry {
                Some((_, steps)) => {
                    for step in steps {
                        let _ = writeln!(out, "- {step}");
                    }
                }
                // Ran nowhere (n/a on every platform), so no steps were recorded.
                None => out.push_str("_Not run on any platform in this run._\n"),
            }
            out.push('\n');
        }
    }
    out
}

/// Build a map of scenario `@id` → (human name, Gherkin steps) by scanning every
/// input's `report.json`. Used for the grid's Scenario reference section. A
/// scenario appears on the first input that ran it; later duplicates are ignored
/// (the name/steps are identical across platforms).
fn scenario_reference(
    inputs: &[(String, PathBuf)],
) -> std::collections::BTreeMap<String, (String, Vec<String>)> {
    let mut map = std::collections::BTreeMap::new();
    for (_label, json_path) in inputs {
        for f in parse_features(json_path) {
            for el in f.elements {
                let Some(id) = scenario_id(&el) else { continue };
                if map.contains_key(&id) {
                    continue;
                }
                let steps = el
                    .steps
                    .iter()
                    .map(|s| format!("{}{}", s.keyword, s.name))
                    .collect();
                map.insert(id, (el.name.clone(), steps));
            }
        }
    }
    map
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
    use crate::command_coverage::{KNOWN_COMMAND_SURFACE, command_coverage_summary};

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

    fn write_report(features_json: &str) -> tempfile::NamedTempFile {
        use std::io::Write as _;
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        f.write_all(features_json.as_bytes()).expect("write json");
        f
    }

    #[test]
    fn parse_descriptor_maps_known_artifacts() {
        for (name, platform, os) in [
            ("e2e-report", "Mock", "Linux"),
            ("e2e-gpu-report", "MI300X", "Linux"),
            ("e2e-gpu-rad3-report", "R9700", "Linux"),
            ("e2e-gpu-mi350p-report", "MI350P", "Linux"),
            ("e2e-gpu-strix-ubuntu-report", "Strix Halo", "Ubuntu"),
            ("e2e-gpu-strix-windows-report", "Strix Halo", "Windows"),
            // Must not fall through to `fallback_descriptor`, which would render
            // "Gpu Strix Wsl" on Linux — a WSL2 host reported as native Linux.
            ("e2e-gpu-strix-wsl-report", "Strix Halo", "WSL2"),
        ] {
            let d = parse_descriptor(name);
            assert_eq!(
                (d.platform.as_str(), d.os.as_str()),
                (platform, os),
                "{name}"
            );
        }
    }

    #[test]
    fn parse_descriptor_strips_nightly_workflow_channel_suffix() {
        // Real artifact names from `.github/workflows/nightly.yml`, which always
        // inserts the channel as the final segment before `-report`. A synthetic
        // label like "mi300x-release" wouldn't reproduce the bug this guards
        // against: nightly-channel artifacts falling through to
        // `fallback_descriptor` and being mislabeled "Linux".
        for (name, platform, os) in [
            ("e2e-gpu-release-report", "MI300X", "Linux"),
            ("e2e-gpu-nightly-report", "MI300X", "Linux"),
            ("e2e-gpu-rad3-release-report", "R9700", "Linux"),
            ("e2e-gpu-rad3-nightly-report", "R9700", "Linux"),
            ("e2e-gpu-mi350p-nightly-report", "MI350P", "Linux"),
            (
                "e2e-gpu-strix-ubuntu-nightly-report",
                "Strix Halo",
                "Ubuntu",
            ),
            (
                "e2e-gpu-strix-windows-nightly-report",
                "Strix Halo",
                "Windows",
            ),
            ("e2e-gpu-strix-wsl-nightly-report", "Strix Halo", "WSL2"),
        ] {
            let d = parse_descriptor(name);
            assert_eq!(
                (d.platform.as_str(), d.os.as_str()),
                (platform, os),
                "{name}"
            );
        }
    }

    #[test]
    fn parse_descriptor_unknown_is_not_falsely_linux() {
        // A report whose platform.json was missing (e.g. a GPU run that errored
        // before writing the sidecar) is labeled `e2e-unknown-report`. Its OS is
        // genuinely unknown — a Windows GPU run must NOT be reported as Linux — so
        // both platform AND os render "Unknown", never a default "Linux".
        let d = parse_descriptor("e2e-unknown-report");
        assert_eq!(d.platform, "Unknown");
        assert_eq!(d.os, "Unknown");
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
        generate_consolidated(&inputs, out.path(), &RunMeta::default()).expect("generate");
        let html = std::fs::read_to_string(out.path()).expect("read");
        assert!(html.contains("Consolidated E2E Report"));
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
        assert_eq!(C::reconcile("pass", false, Some(true)), C::Pass);
        assert_eq!(C::reconcile("pass", false, Some(false)), C::UnexpectedFail);
        assert_eq!(C::reconcile("xfail", false, Some(false)), C::Xfail);
        assert_eq!(C::reconcile("xfail", false, Some(true)), C::Xpass);
        assert_eq!(C::reconcile("xfail", true, Some(true)), C::FlakyXpass);
        assert_eq!(C::reconcile("skip", false, None), C::Skip);
        assert_eq!(C::reconcile("skip", false, Some(true)), C::RanWhenNa);
        // A declared expect-pass/xfail with NO result is Absent (a problem), so a
        // hung/lost-results run reds the platform instead of greenwashing to PASS.
        assert_eq!(C::reconcile("pass", false, None), C::Absent);
        assert_eq!(C::reconcile("xfail", false, None), C::Absent);
        // Flaky does not excuse a missing result either: it licenses an XPASS,
        // not the absence of any outcome for a scenario the matrix declares.
        assert_eq!(C::reconcile("xfail", true, None), C::Absent);
        assert!(C::UnexpectedFail.is_problem());
        assert!(C::Xpass.is_problem());
        assert!(!C::FlakyXpass.is_problem());
        assert!(C::RanWhenNa.is_problem());
        assert!(C::Absent.is_problem());
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

    /// The order the grid renders rows in, flattened across feature groups.
    fn grid_order(inputs: &[(String, PathBuf)]) -> Vec<(String, String)> {
        Grid::build(inputs)
            .groups
            .iter()
            .flat_map(|g| g.rows.iter().map(|r| (g.feature.clone(), r.id.clone())))
            .collect()
    }

    #[test]
    fn scenario_index_parses_the_per_feature_number() {
        assert_eq!(scenario_index("serve-07 - A model responds"), Some(7));
        assert_eq!(scenario_index("lifecycle-01 - Linux - a bundle"), Some(1));
        // Unindexed names (older artifacts) have no number to sort by.
        assert_eq!(scenario_index("A model responds"), None);
        assert_eq!(scenario_index(""), None);
    }

    #[test]
    fn grid_groups_by_feature_and_orders_by_index() {
        // Declared deliberately out of order (and interleaved across features)
        // so a passing assertion can only come from real grouping + sorting,
        // not from the input order.
        let report = feature_json(&[
            (&["id:serve-b"], &["passed"]),
            (&["id:examine-a"], &["passed"]),
        ]);
        let platform = r#"{
            "platform_slug": "mock",
            "capability": {"effective_serve_engine": "none"},
            "expectations": [
                {"id":"serve-b","feature":"Serving","scenario":"serve-02 - second","expected":"pass"},
                {"id":"examine-a","feature":"Examine","scenario":"examine-09 - ninth","expected":"pass"},
                {"id":"serve-a","feature":"Serving","scenario":"serve-01 - first","expected":"skip"},
                {"id":"examine-b","feature":"Examine","scenario":"examine-10 - tenth","expected":"skip"}
            ]
        }"#;
        let (_d, path) = write_platform(&report, platform);
        let inputs = vec![("mock".to_string(), path)];

        assert_eq!(
            grid_order(&inputs),
            vec![
                ("Examine".to_string(), "examine-a".to_string()),
                ("Examine".to_string(), "examine-b".to_string()),
                ("Serving".to_string(), "serve-a".to_string()),
                ("Serving".to_string(), "serve-b".to_string()),
            ],
            "rows must group by feature and sort by index (09 before 10, not \
             lexically), regardless of declaration order",
        );

        // Each feature gets its own table under its own heading.
        let md = consolidated_summary_markdown(&inputs);
        assert!(
            md.contains("#### Examine"),
            "missing feature heading:\n{md}"
        );
        assert!(
            md.contains("#### Serving"),
            "missing feature heading:\n{md}"
        );
        // The human name is shown alongside the id, sourced from platform.json —
        // serve-a was skipped everywhere, so report.json has no name for it.
        assert!(
            md.contains("serve-01 - first<br>[`serve-a`](#serve-a)"),
            "skipped scenario should still show its name:\n{md}"
        );
    }

    #[test]
    fn grid_falls_back_to_report_feature_then_id_prefix() {
        // A platform.json predating the `feature`/`scenario` fields. `serve-x`
        // ran (so report.json knows its feature); `dash-y` was skipped
        // everywhere, leaving only its id to group by.
        let report = feature_json(&[(&["id:serve-x"], &["passed"])]);
        let platform = r#"{
            "platform_slug": "mock",
            "capability": {"effective_serve_engine": "none"},
            "expectations": [
                {"id":"serve-x","effective_engine":"","expected":"pass"},
                {"id":"dash-y","effective_engine":"","expected":"skip"}
            ]
        }"#;
        let (_d, path) = write_platform(&report, platform);
        let inputs = vec![("mock".to_string(), path)];

        // `feature_json` names its feature "F"; the id prefix covers the rest.
        assert_eq!(
            grid_order(&inputs),
            vec![
                ("F".to_string(), "serve-x".to_string()),
                ("dash".to_string(), "dash-y".to_string()),
            ],
            "an artifact with no feature field must still group, not vanish",
        );
    }

    #[test]
    fn a_named_feature_overrides_an_older_artifact_s_guess() {
        // Mixing artifact vintages: the old one has no `feature` field, so the
        // id prefix is all there is to go on; the new one names the feature.
        // The named one must win regardless of input order, or the scenario
        // splits across two groups sorted far apart.
        let old = r#"{
            "platform_slug": "mi300x",
            "capability": {"effective_serve_engine": "vllm"},
            "expectations": [{"id":"serve-x","expected":"skip"}]
        }"#;
        let new = r#"{
            "platform_slug": "mock",
            "capability": {"effective_serve_engine": "none"},
            "expectations": [
                {"id":"serve-x","feature":"Model serving","scenario":"serve-01 - a","expected":"skip"}
            ]
        }"#;
        let report = feature_json(&[]);
        let (_d1, old_path) = write_platform(&report, old);
        let (_d2, new_path) = write_platform(&report, new);

        // Old artifact first: its id-prefix guess must not stick.
        let inputs = vec![
            ("mi300x".to_string(), old_path.clone()),
            ("mock".to_string(), new_path.clone()),
        ];
        assert_eq!(
            grid_order(&inputs),
            vec![("Model serving".to_string(), "serve-x".to_string())],
            "a later artifact that names the feature must override the guess",
        );

        // And the reverse order must not regress it back to the guess.
        let inputs = vec![
            ("mock".to_string(), new_path),
            ("mi300x".to_string(), old_path),
        ];
        assert_eq!(
            grid_order(&inputs),
            vec![("Model serving".to_string(), "serve-x".to_string())],
            "an older artifact must not overwrite a named feature",
        );
    }

    #[test]
    fn the_last_naming_artifact_sets_the_sort_position() {
        // Two artifacts naming the same ids differently — a `Scenario:` renamed
        // between vintages. The name carries the sort index, so which one wins
        // decides row order; pin the documented rule (last wins) rather than
        // leaving it to fall out of iteration order.
        let older = r#"{
            "platform_slug": "mi300x",
            "capability": {"effective_serve_engine": "vllm"},
            "expectations": [
                {"id":"serve-a","feature":"Model serving","scenario":"serve-01 - a","expected":"skip"},
                {"id":"serve-b","feature":"Model serving","scenario":"serve-02 - b","expected":"skip"}
            ]
        }"#;
        let newer = r#"{
            "platform_slug": "mock",
            "capability": {"effective_serve_engine": "none"},
            "expectations": [
                {"id":"serve-a","feature":"Model serving","scenario":"serve-02 - a moved","expected":"skip"},
                {"id":"serve-b","feature":"Model serving","scenario":"serve-01 - b moved","expected":"skip"}
            ]
        }"#;
        let report = feature_json(&[]);
        let (_d1, older_path) = write_platform(&report, older);
        let (_d2, newer_path) = write_platform(&report, newer);

        let inputs = vec![
            ("mi300x".to_string(), older_path),
            ("mock".to_string(), newer_path),
        ];
        assert_eq!(
            grid_order(&inputs)
                .into_iter()
                .map(|(_, id)| id)
                .collect::<Vec<_>>(),
            vec!["serve-b".to_string(), "serve-a".to_string()],
            "the last artifact to name a scenario sets its sort position",
        );
    }

    #[test]
    fn two_channels_same_platform_yield_two_columns() {
        // Same platform_slug, different channel: must not collide into one
        // column, or a nightly run's results silently overwrite a release run's.
        let release = r#"{
            "platform_slug": "mi300x",
            "capability": {"effective_serve_engine": "vllm"},
            "versions": {"channel": "release"},
            "expectations": [{"id":"serve-x","feature":"Serving","expected":"pass"}]
        }"#;
        let nightly = r#"{
            "platform_slug": "mi300x",
            "capability": {"effective_serve_engine": "vllm"},
            "versions": {"channel": "nightly"},
            "expectations": [{"id":"serve-x","feature":"Serving","expected":"pass"}]
        }"#;
        let report = feature_json(&[(&["id:serve-x"], &["passed"])]);
        let (_d1, release_path) = write_platform(&report, release);
        let (_d2, nightly_path) = write_platform(&report, nightly);
        let inputs = vec![
            ("mi300x-release".to_string(), release_path),
            ("mi300x-nightly".to_string(), nightly_path),
        ];

        let grid = Grid::build(&inputs);
        assert_eq!(
            grid.columns.len(),
            2,
            "same slug, different channel, must be two columns"
        );
        let labels: Vec<_> = grid.columns.iter().map(GridColumn::label).collect();
        assert!(
            labels.contains(&"mi300x (release)".to_string()),
            "{labels:?}"
        );
        assert!(
            labels.contains(&"mi300x (nightly)".to_string()),
            "{labels:?}"
        );
    }

    #[test]
    fn platform_sort_key_orders_by_platform_then_os_then_known_bugs() {
        // Pins all four elements of the sort-key tuple, in precedence order:
        // platform, os, known_bugs, channel. Three Strix Halo rows alone
        // can't pin `platform` — they all share it, so dropping or swapping
        // that element would not change their order — so a fourth row on a
        // different platform ("Unknown"/"Unknown", via the explicit
        // "e2e-unknown-report" artifact name) is added: "Strix Halo" sorts
        // before "Unknown" by platform, but "Unknown" alone (as an OS) sorts
        // before "Windows", so dropping the platform element from the key, or
        // swapping it with os, both move this row ahead of the Windows row
        // instead of after it. Within the Strix Halo rows: "Strix Halo"/Ubuntu
        // before "Strix Halo"/Windows pins os; the two Ubuntu rows, identical
        // except for the known-bugs artifact suffix, pin known_bugs ordering
        // (false < true) once os and channel are held equal. A fifth and
        // sixth row on two more distinct platforms with opposite channels —
        // MI300X/release and R9700/nightly — pin channel's precedence:
        // "nightly" sorts before "release", so if channel outranked platform
        // in the key, R9700 would sort first; "MI300X" sorts before "R9700"
        // by platform, so the correct key keeps MI300X first regardless.
        // generate_consolidated_writes_html and
        // command_coverage_distinguishes_channels_on_same_platform pin channel
        // only as a tie-breaker between two rows that already share a
        // platform — not where it sits in this tuple, which is what the
        // MI300X/R9700 rows pin here. No platform.json sidecar, so
        // display_counts falls back to raw junit stats and each row's
        // distinct scenario count (1/2/3/4/5/6) tells the rows apart in the
        // rendered table.
        let ubuntu_plain_report = write_report(&feature_json(&[(&[], &["passed"])]));
        let ubuntu_known_bugs_report =
            write_report(&feature_json(&[(&[], &["passed"]), (&[], &["passed"])]));
        let windows_plain_report = write_report(&feature_json(&[
            (&[], &["passed"]),
            (&[], &["passed"]),
            (&[], &["passed"]),
        ]));
        let unknown_report = write_report(&feature_json(&[
            (&[], &["passed"]),
            (&[], &["passed"]),
            (&[], &["passed"]),
            (&[], &["passed"]),
        ]));
        let mi300x_release_report = write_report(&feature_json(&[
            (&[], &["passed"]),
            (&[], &["passed"]),
            (&[], &["passed"]),
            (&[], &["passed"]),
            (&[], &["passed"]),
        ]));
        let r9700_nightly_report = write_report(&feature_json(&[
            (&[], &["passed"]),
            (&[], &["passed"]),
            (&[], &["passed"]),
            (&[], &["passed"]),
            (&[], &["passed"]),
            (&[], &["passed"]),
        ]));
        let inputs = vec![
            (
                "e2e-gpu-strix-windows-report".to_string(),
                windows_plain_report.path().to_path_buf(),
            ),
            (
                "e2e-gpu-strix-ubuntu-known-bugs-report".to_string(),
                ubuntu_known_bugs_report.path().to_path_buf(),
            ),
            (
                "e2e-gpu-strix-ubuntu-report".to_string(),
                ubuntu_plain_report.path().to_path_buf(),
            ),
            (
                "e2e-unknown-report".to_string(),
                unknown_report.path().to_path_buf(),
            ),
            (
                "e2e-gpu-release-report".to_string(),
                mi300x_release_report.path().to_path_buf(),
            ),
            (
                "e2e-gpu-rad3-nightly-report".to_string(),
                r9700_nightly_report.path().to_path_buf(),
            ),
        ];

        let md = consolidated_summary_markdown(&inputs);
        let ubuntu_plain_pos = md
            .find("| Strix Halo | Ubuntu | 1 |")
            .expect("plain Ubuntu row");
        let ubuntu_known_bugs_pos = md
            .find("| Strix Halo | Ubuntu | 2 |")
            .expect("known-bugs Ubuntu row");
        let windows_pos = md
            .find("| Strix Halo | Windows | 3 |")
            .expect("Windows row");
        let unknown_pos = md
            .find("| Unknown | Unknown | 4 |")
            .expect("Unknown-platform row");
        let mi300x_pos = md
            .find("| MI300X (release) | Linux | 5 |")
            .expect("MI300X release row");
        let r9700_pos = md
            .find("| R9700 (nightly) | Linux | 6 |")
            .expect("R9700 nightly row");
        assert!(
            ubuntu_plain_pos < ubuntu_known_bugs_pos,
            "same platform/os: known_bugs=false must sort before known_bugs=true:\n{md}"
        );
        assert!(
            ubuntu_known_bugs_pos < windows_pos,
            "same platform: Ubuntu must sort before Windows regardless of known_bugs:\n{md}"
        );
        assert!(
            windows_pos < unknown_pos,
            "\"Strix Halo\" must sort before \"Unknown\" by platform, even though \"Unknown\" as an OS sorts before \"Windows\":\n{md}"
        );
        assert!(
            mi300x_pos < r9700_pos,
            "\"MI300X\" must sort before \"R9700\" by platform, even though its channel \
             (\"release\") sorts after R9700's (\"nightly\"):\n{md}"
        );
    }

    #[test]
    fn nightly_failure_reaches_needs_attention_in_its_own_column() {
        // Pins that a release pass and a nightly fail on the same slug land in
        // two separate columns (not one collision that a merge clause has to
        // arbitrate) and that the nightly failure surfaces in the needs-attention
        // list. It does NOT exercise the same-column merge-protection clause —
        // see `a_later_failure_is_not_masked_within_one_column` for that.
        let release = r#"{
            "platform_slug": "mi300x",
            "capability": {"effective_serve_engine": "vllm"},
            "versions": {"channel": "release"},
            "expectations": [{"id":"serve-x","feature":"Serving","expected":"pass"}]
        }"#;
        let nightly = r#"{
            "platform_slug": "mi300x",
            "capability": {"effective_serve_engine": "vllm"},
            "versions": {"channel": "nightly"},
            "expectations": [{"id":"serve-x","feature":"Serving","expected":"pass"}]
        }"#;
        let passed = feature_json(&[(&["id:serve-x"], &["passed"])]);
        let failed = feature_json(&[(&["id:serve-x"], &["failed"])]);
        let (_d1, release_path) = write_platform(&passed, release);
        let (_d2, nightly_path) = write_platform(&failed, nightly);
        let inputs = vec![
            ("mi300x-release".to_string(), release_path),
            ("mi300x-nightly".to_string(), nightly_path),
        ];

        let grid = Grid::build(&inputs);
        let nightly_col = grid
            .columns
            .iter()
            .find(|c| c.versions.channel.as_deref() == Some("nightly"))
            .expect("nightly column present");
        let outcome = *nightly_col
            .outcomes
            .get("serve-x")
            .expect("serve-x outcome");
        assert_eq!(outcome, CellOutcome::UnexpectedFail, "{outcome:?}");
        assert!(outcome.is_problem());
        assert!(
            grid.problems()
                .iter()
                .any(|(label, id, _, _)| label == "mi300x (nightly)" && *id == "serve-x"),
            "the nightly failure must surface in the needs-attention list",
        );
    }

    #[test]
    fn a_later_failure_is_not_masked_within_one_column() {
        // Same platform_slug AND same channel: both artifacts land in ONE column,
        // so this (unlike `nightly_failure_reaches_needs_attention_in_its_own_column`)
        // actually exercises the `and_modify` merge-protection clause in
        // `Grid::build` — a later, conflicting result for the same id must not be
        // silently dropped in favor of an earlier clean one.
        let release_a = r#"{
            "platform_slug": "mi300x",
            "capability": {"effective_serve_engine": "vllm"},
            "versions": {"channel": "release"},
            "expectations": [{"id":"serve-x","feature":"Serving","expected":"pass"}]
        }"#;
        let release_b = release_a;
        let passed = feature_json(&[(&["id:serve-x"], &["passed"])]);
        let failed = feature_json(&[(&["id:serve-x"], &["failed"])]);
        let (_d1, first_path) = write_platform(&passed, release_a);
        let (_d2, second_path) = write_platform(&failed, release_b);
        let inputs = vec![
            ("mi300x-release-a".to_string(), first_path),
            ("mi300x-release-b".to_string(), second_path),
        ];

        let grid = Grid::build(&inputs);
        assert_eq!(
            grid.columns.len(),
            1,
            "same slug and channel must collapse into one column, not two",
        );
        let outcome = *grid.columns[0]
            .outcomes
            .get("serve-x")
            .expect("serve-x outcome");
        assert_eq!(
            outcome,
            CellOutcome::UnexpectedFail,
            "a later failure must not be masked by an earlier pass in the same column: {outcome:?}",
        );
    }

    #[test]
    fn single_channel_artifacts_render_unchanged() {
        // No channel anywhere in the input (older artifact / no runtime active):
        // exactly one column, and its heading is the bare slug, unchanged.
        let platform = r#"{
            "platform_slug": "mock",
            "capability": {"effective_serve_engine": "none"},
            "expectations": [{"id":"serve-x","feature":"Serving","expected":"pass"}]
        }"#;
        let report = feature_json(&[(&["id:serve-x"], &["passed"])]);
        let (_d, path) = write_platform(&report, platform);
        let inputs = vec![("mock".to_string(), path)];

        let grid = Grid::build(&inputs);
        assert_eq!(grid.columns.len(), 1);
        assert_eq!(grid.columns[0].label(), "mock");
    }

    #[test]
    fn scenario_reference_anchors_every_grid_row() {
        // A scenario that is n/a everywhere has no report.json entry — it still
        // needs an anchor, or the grid's link to it dangles.
        let report = feature_json(&[(&["id:serve-x"], &["passed"])]);
        let platform = r#"{
            "platform_slug": "mock",
            "capability": {"effective_serve_engine": "none"},
            "expectations": [
                {"id":"serve-x","feature":"Serving","scenario":"serve-01 - ran","expected":"pass"},
                {"id":"serve-z","feature":"Serving","scenario":"serve-02 - skipped","expected":"skip"}
            ]
        }"#;
        let (_d, path) = write_platform(&report, platform);
        let md = consolidated_summary_markdown(&[("mock".to_string(), path)]);
        assert!(md.contains("##### serve-x"), "missing anchor:\n{md}");
        assert!(
            md.contains("##### serve-z"),
            "a never-run scenario still needs its anchor:\n{md}"
        );
        assert!(
            md.contains("_Not run on any platform in this run._"),
            "a never-run scenario should say so instead of showing no steps:\n{md}"
        );
    }

    #[test]
    fn scenario_reference_anchors_grid_when_report_has_no_scenarios() {
        let report = feature_json(&[]);
        let platform = r#"{
            "platform_slug": "mock",
            "capability": {"effective_serve_engine": "none"},
            "expectations": [
                {"id":"serve-a","feature":"Serving","scenario":"serve-01 - first skipped","expected":"skip"},
                {"id":"serve-b","feature":"Serving","scenario":"serve-02 - second skipped","expected":"skip"}
            ]
        }"#;
        let (_d, path) = write_platform(&report, platform);
        let md = consolidated_summary_markdown(&[("mock".to_string(), path)]);

        for (id, name) in [
            ("serve-a", "serve-01 - first skipped"),
            ("serve-b", "serve-02 - second skipped"),
        ] {
            assert!(
                md.contains(&format!("{name}<br>[`{id}`](#{id})")),
                "grid link should use the manifest name for {id}:\n{md}"
            );
            assert!(
                md.contains(&format!("##### {id}")),
                "grid link for {id} should have a Scenario reference anchor:\n{md}"
            );
        }
        assert_eq!(
            md.matches("_Not run on any platform in this run._").count(),
            2,
            "each all-skipped reference entry should explain that it was not run:\n{md}"
        );
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
            md.contains("[`serve-x`](#serve-x) | ✗ |"),
            "serve-x should be xfail (grey ✗):\n{md}"
        );
        assert!(
            md.contains("[`examine-y`](#examine-y) | ✅ |"),
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
    fn grid_tolerates_flaky_xpass() {
        let report = feature_json(&[(&["id:serve-flaky"], &["passed"])]);
        let platform = r#"{
            "platform_slug": "mi300x",
            "capability": {"effective_serve_engine": "vllm"},
            "expectations": [
                {"id":"serve-flaky","effective_engine":"vllm","expected":"xfail","bug":"EAI-7333","reason":"readiness race","flaky":true}
            ]
        }"#;
        let (_d, path) = write_platform(&report, platform);
        let inputs = vec![("mi300x".to_string(), path)];
        let md = consolidated_summary_markdown(&inputs);
        assert!(
            md.contains("✅XPASS (flaky)"),
            "grid should show tolerated flaky XPASS:\n{md}"
        );
        assert!(
            !md.contains("**XPASS** on"),
            "flaky XPASS must not need attention:\n{md}"
        );
    }

    #[test]
    fn expectation_grid_legend_explains_every_glyph() {
        // The legend must explain every glyph `CellOutcome::glyph()` can emit,
        // not just the original pass/xfail/n-a/FAIL/XPASS/no-data set — flaky
        // XPASS, ran-when-n/a, and absent results are real states a reader can
        // land on and must not be left silently unexplained.
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

        let md = consolidated_summary_markdown(&inputs);
        for glyph in ["✅XPASS (flaky)", "⚠️n/a-ran", "⚠️no-result"] {
            assert!(
                md.contains(glyph),
                "markdown grid legend must explain {glyph:?}:\n{md}"
            );
        }

        let html = expectation_grid_html(&inputs).into_string();
        for glyph in ["✅XPASS (flaky)", "⚠️n/a-ran", "⚠️no-result"] {
            assert!(
                html.contains(glyph),
                "html grid legend must explain {glyph:?}:\n{html}"
            );
        }
    }

    #[test]
    fn legend_entries_stay_exhaustive_with_cell_outcome() {
        // `expected_text`'s match is exhaustive, so adding a `CellOutcome`
        // variant without adding an arm here is a compile error — unlike the
        // hardcoded 3-glyph check above, this can't silently miss a 10th
        // variant the way `LEGEND_ENTRIES` alone could.
        fn expected_text(outcome: CellOutcome) -> &'static str {
            match outcome {
                CellOutcome::Pass => "pass",
                CellOutcome::Xfail => "known bug, failed as expected (xfail)",
                CellOutcome::Skip => "not applicable here",
                CellOutcome::UnexpectedFail => "regression",
                CellOutcome::Xpass => "bug fixed here (stale entry)",
                CellOutcome::FlakyXpass => "known flaky bug passed this run",
                CellOutcome::RanWhenNa => "ran despite being marked n/a",
                CellOutcome::Absent => "expected to run but no result recorded",
                CellOutcome::Missing => "no data.",
            }
        }
        const ALL: [CellOutcome; 9] = [
            CellOutcome::Pass,
            CellOutcome::Xfail,
            CellOutcome::Skip,
            CellOutcome::UnexpectedFail,
            CellOutcome::Xpass,
            CellOutcome::FlakyXpass,
            CellOutcome::RanWhenNa,
            CellOutcome::Absent,
            CellOutcome::Missing,
        ];

        assert_eq!(
            LEGEND_ENTRIES.len(),
            ALL.len(),
            "LEGEND_ENTRIES must have exactly one entry per CellOutcome variant"
        );
        for outcome in ALL {
            let entry = LEGEND_ENTRIES
                .iter()
                .find(|(o, _)| *o == outcome)
                .unwrap_or_else(|| panic!("{outcome:?} has no LEGEND_ENTRIES explanation"));
            assert_eq!(
                entry.1,
                expected_text(outcome),
                "{outcome:?}'s LEGEND_ENTRIES text drifted from the source of truth"
            );
        }
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

        let html = expectation_grid_html(&inputs).into_string();
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

    #[test]
    fn grid_heading_shows_component_versions() {
        // platform.json carries OS/ROCm/vLLM/lemonade versions → they render in the
        // summary matrix Platform cell (ROCm/vLLM/lemonade) and OS cell (distro).
        // (Versions were removed from the expectation-grid header.)
        let report = feature_json(&[(&["id:serve-x"], &["passed"])]);
        let platform = r#"{
            "platform_slug": "mi300x",
            "capability": {"effective_serve_engine": "vllm"},
            "versions": {"os":"Ubuntu 24.04.3 LTS","rocm":"7.13.0","vllm":"0.23.0+rocm723","lemonade":"11.5.1"},
            "expectations": [
                {"id":"serve-x","effective_engine":"vllm","expected":"pass"}
            ]
        }"#;
        let (_d, path) = write_platform(&report, platform);
        let inputs = vec![("mi300x".to_string(), path)];
        let md = consolidated_summary_markdown(&inputs);
        for token in [
            "Ubuntu 24.04.3 LTS",
            "ROCm 7.13.0",
            "vLLM 0.23.0+rocm723",
            "lemonade 11.5.1",
        ] {
            assert!(md.contains(token), "matrix cell missing {token:?}:\n{md}");
        }
    }

    #[test]
    fn grid_heading_omits_absent_versions() {
        // A platform.json with only OS known renders just the distro (in the matrix
        // OS cell); absent ROCm/vLLM/lemonade are omitted from the Platform cell.
        let report = feature_json(&[(&["id:serve-x"], &["passed"])]);
        let platform = r#"{
            "platform_slug": "mock",
            "capability": {"effective_serve_engine": "lemonade"},
            "versions": {"os":"Debian 12"},
            "expectations": [
                {"id":"serve-x","effective_engine":"lemonade","expected":"pass"}
            ]
        }"#;
        let (_d, path) = write_platform(&report, platform);
        let inputs = vec![("mock".to_string(), path)];
        let md = consolidated_summary_markdown(&inputs);
        assert!(md.contains("Debian 12"), "should show OS distro:\n{md}");
        assert!(!md.contains("vLLM "), "no vLLM version known → omit:\n{md}");
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
            md.contains("[`gpu-only`](#gpu-only) | n/a |"),
            "skip should render as n/a:\n{md}"
        );
        assert!(md.contains("[`ran-here`](#ran-here) | ✅ |"));
    }

    // Pure helper unit tests live with their code in `command_coverage.rs`;
    // the command-coverage tests here stay because they build `PlatformReport`
    // through the private `PlatformReport::load` and the shared `feature_json`
    // helper, both local to this module.

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
    fn command_coverage_distinguishes_channels_on_same_platform() {
        // Same platform_slug, release passing and nightly failing the same
        // command: the coverage table must carry both channels as separate
        // columns, not collapse them into one where the failure masks the pass.
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
        let commands = concat!(
            r#"{"scenario":"s0","subcommand":"rocm serve Qwen/Qwen2.5-1.5B-Instruct --engine","model":"Qwen/Qwen2.5-1.5B-Instruct","engine":"vllm","rc":0}"#,
            "\n",
        );

        let release_dir = tempfile::tempdir().expect("tempdir");
        let release_report = release_dir.path().join("report.json");
        std::fs::write(&release_report, feature_json(&[(&[], &["passed"])])).expect("write report");
        std::fs::write(release_dir.path().join("platform.json"), release_platform)
            .expect("write platform");
        std::fs::write(release_dir.path().join("commands.jsonl"), commands)
            .expect("write commands");

        let nightly_dir = tempfile::tempdir().expect("tempdir");
        let nightly_report = nightly_dir.path().join("report.json");
        std::fs::write(&nightly_report, feature_json(&[(&[], &["failed"])])).expect("write report");
        std::fs::write(nightly_dir.path().join("platform.json"), nightly_platform)
            .expect("write platform");
        std::fs::write(nightly_dir.path().join("commands.jsonl"), commands)
            .expect("write commands");

        let inputs = vec![
            ("mi300x-release".to_string(), release_report),
            ("mi300x-nightly".to_string(), nightly_report),
        ];
        let md = consolidated_summary_markdown(&inputs);

        // Assert structure before glyphs: a column-collapse regression must
        // fail loudly here rather than passing by reading the wrong cell.
        let header = md
            .lines()
            .find(|l| l.contains("| Command | Engine |"))
            .expect("coverage header row present");
        let header_cells: Vec<&str> = header.split('|').map(str::trim).collect();
        assert!(
            header_cells.contains(&"Mi300x Linux (release)")
                && header_cells.contains(&"Mi300x Linux (nightly)"),
            "coverage header must carry both channels distinctly:\n{header}"
        );

        let row = md
            .lines()
            .find(|l| l.contains("rocm serve"))
            .expect("serve command row present");
        let row_cells: Vec<&str> = row.split('|').map(str::trim).collect();
        let release_i = header_cells
            .iter()
            .position(|c| *c == "Mi300x Linux (release)")
            .unwrap();
        let nightly_i = header_cells
            .iter()
            .position(|c| *c == "Mi300x Linux (nightly)")
            .unwrap();
        assert_eq!(
            row_cells[release_i], "✅",
            "release column must show a pass:\n{row}"
        );
        assert_eq!(
            row_cells[nightly_i], "❌",
            "nightly failure must not be masked by release's pass:\n{row}"
        );

        // Platform/OS summary table: same two channels must get separate rows
        // there too, not just in the coverage sub-table above.
        assert!(
            md.contains("| Mi300x (release) | Linux |"),
            "summary row must carry the release channel suffix:\n{md}"
        );
        assert!(
            md.contains("| Mi300x (nightly) | Linux |"),
            "summary row must carry the nightly channel suffix:\n{md}"
        );

        // The 4-element sort tuple only matters if it actually drives order
        // regardless of input order: feed nightly before release and confirm
        // the summary rows land in the same relative order either way.
        let reversed_inputs = vec![inputs[1].clone(), inputs[0].clone()];
        let md_reversed = consolidated_summary_markdown(&reversed_inputs);
        let rows: Vec<&str> = md.lines().filter(|l| l.starts_with("| Mi300x (")).collect();
        let rows_reversed: Vec<&str> = md_reversed
            .lines()
            .filter(|l| l.starts_with("| Mi300x ("))
            .collect();
        assert_eq!(
            rows, rows_reversed,
            "platform row order must be deterministic regardless of input order, \
             which only holds once channel is in the sort key"
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

    #[test]
    fn manifest_channel_wins_over_artifact_name_channel() {
        // The manifest's `versions.channel` is authoritative; an artifact-name
        // channel, when present, must not override it.
        let platform = r#"{
            "platform_slug": "mi300x",
            "capability": {"effective_serve_engine": "vllm"},
            "versions": {"channel": "release"},
            "expectations": [{"id":"serve-x","feature":"Serving","expected":"pass"}]
        }"#;
        let report = feature_json(&[(&["id:serve-x"], &["passed"])]);
        let (_d, path) = write_platform(&report, platform);
        // Artifact name (deliberately) disagrees with the manifest, to prove
        // precedence rather than coincidentally matching.
        let r = PlatformReport::load("e2e-gpu-nightly-report".into(), &path);
        assert_eq!(
            r.effective_channel(),
            Some("release"),
            "{:?}",
            r.effective_channel()
        );
        assert!(r.label.contains("(release)"), "{}", r.label);
        assert!(!r.label.contains("(nightly)"), "{}", r.label);
    }

    #[test]
    fn artifact_name_channel_is_a_fallback_when_manifest_is_absent() {
        // An errored run that never wrote `platform.json` has no manifest channel
        // to read; the artifact-name channel is the only signal available.
        let report = write_report(&feature_json(&[(&[], &["passed"])]));
        let r = PlatformReport::load("e2e-gpu-strix-windows-nightly-report".into(), report.path());
        assert_eq!(
            r.effective_channel(),
            Some("nightly"),
            "{:?}",
            r.effective_channel()
        );
        assert!(r.label.contains("(nightly)"), "{}", r.label);
    }
}
