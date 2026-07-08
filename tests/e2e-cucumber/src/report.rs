// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use std::collections::BTreeMap;
use std::path::Path;
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

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| format_utc(d.as_secs()))
        .unwrap_or_default();

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
            }
        }
    };

    std::fs::write(html_path, markup.into_string())
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
  table.stats a { color: #1565c0; text-decoration: none; }
  table.stats a:hover { text-decoration: underline; }

  .bar { display: inline-flex; width: 120px; height: 14px; border-radius: 2px; overflow: hidden; vertical-align: middle; }
  .bar-pass { background: #8bc34a; }
  .bar-fail { background: #e53935; }
  .bar-skip { background: #bdbdbd; }

  .details { margin-top: 0.5rem; }
  .feature-group { margin-bottom: 1.5rem; }
  .feature-title { font-weight: 600; font-size: 1rem; padding: 8px 12px; background: #e3f2fd;
                   border-left: 4px solid #1565c0; margin-bottom: 0; }
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
    fn format_utc_renders_real_date_not_placeholder() {
        // 2021-01-01 00:00:00 UTC = 1_609_459_200. Regression guard against the
        // old "-xx-xx" placeholder.
        assert_eq!(format_utc(1_609_459_200), "2021-01-01 00:00:00 UTC");
        // A time-of-day sample: 2026-07-08 13:30:45 UTC.
        assert_eq!(format_utc(1_783_517_445), "2026-07-08 13:30:45 UTC");
    }
}
