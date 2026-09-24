// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Cucumber `report.json` data model, parsing, and xfail (`@expected-failure`)
//! evaluation.

use std::path::Path;

use serde::Deserialize;

#[derive(Deserialize)]
pub(crate) struct Feature {
    pub(crate) name: String,
    pub(crate) uri: String,
    #[serde(default)]
    pub(crate) elements: Vec<Element>,
}

#[derive(Deserialize)]
pub(crate) struct Element {
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) tags: Vec<Tag>,
    #[serde(default)]
    pub(crate) steps: Vec<Step>,
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
pub(crate) struct Tag {
    pub(crate) name: String,
}

#[derive(Deserialize)]
pub(crate) struct Step {
    pub(crate) keyword: String,
    pub(crate) name: String,
    #[serde(default)]
    pub(crate) result: StepResult,
}

#[derive(Deserialize, Default)]
pub(crate) struct StepResult {
    #[serde(default)]
    pub(crate) status: String,
    #[serde(default)]
    pub(crate) duration: u64,
    #[serde(default)]
    pub(crate) error_message: Option<String>,
}

pub(crate) struct Stats {
    pub(crate) total: u32,
    pub(crate) passed: u32,
    pub(crate) failed: u32,
    pub(crate) skipped: u32,
    elapsed_ns: u64,
}

impl Stats {
    pub(crate) const fn new() -> Self {
        Self {
            total: 0,
            passed: 0,
            failed: 0,
            skipped: 0,
            elapsed_ns: 0,
        }
    }

    pub(crate) fn add(&mut self, status: &str, duration_ns: u64) {
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

    pub(crate) fn elapsed_str(&self) -> String {
        let ms = self.elapsed_ns / 1_000_000;
        let s = ms / 1000;
        let m = s / 60;
        format!("{:02}:{:02}:{:02}.{:03}", m / 60, m % 60, s % 60, ms % 1000)
    }

    /// Percentage widths for the pass/fail/skip bar. Returns `None` when there
    /// are no scenarios (no bar to render).
    pub(crate) const fn bar_widths(&self) -> Option<(u32, u32, u32)> {
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

    pub(crate) const fn status_text(&self) -> &'static str {
        if self.failed > 0 {
            "FAIL"
        } else if self.total == 0 {
            "SKIP"
        } else {
            "PASS"
        }
    }
}

pub(crate) fn scenario_status(el: &Element) -> &'static str {
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
pub(crate) fn scenario_passed(el: &Element) -> bool {
    scenario_status(el) == "passed"
}

pub(crate) fn scenario_duration(el: &Element) -> u64 {
    el.steps.iter().map(|s| s.result.duration).sum()
}

/// Read and parse a cucumber `report.json` into its feature list. A missing or
/// malformed file yields an empty list rather than an error, so a single bad
/// platform report never sinks a consolidated run.
pub(crate) fn parse_features(json_path: &Path) -> Vec<Feature> {
    std::fs::read_to_string(json_path)
        .ok()
        .and_then(|json| serde_json::from_str(&json).ok())
        .unwrap_or_default()
}

pub(crate) fn stats_of(features: &[Feature]) -> Stats {
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

pub(crate) const EXPECTED_FAILURE_TAG: &str = "expected-failure";

pub(crate) fn evaluate_xfail_features(features: &[Feature]) -> XfailReport {
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
pub(crate) fn scenario_id(el: &Element) -> Option<String> {
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
}
