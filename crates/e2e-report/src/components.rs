// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Shared maud HTML fragment rendering, plus the CSS and timestamp helpers used
//! by both the single-platform and consolidated report generators. Depends only
//! on [`crate::parse`] types so both generators can use it without a cycle.

use std::time::SystemTime;

use maud::{Markup, html};

use crate::parse::{Element, Feature, Stats, Step, scenario_duration, scenario_status};

pub(crate) fn stats_bar(stats: &Stats) -> Markup {
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

pub(crate) fn feature_group(feature: &Feature) -> Markup {
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

pub(crate) fn stats_table(title: &str, rows: &[(String, &Stats)]) -> Markup {
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
pub(crate) fn now_utc() -> String {
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

pub(crate) const STYLE: &str = r#"
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
  /* xfail: a known bug that failed as expected — a muted grey ✗, sibling to the red ✗. */
  .status-xfail { color: #9e9e9e; font-weight: 600; }
  .grid-legend { font-size: 0.82rem; color: #555; margin: -0.5rem 0 0.75rem; }
  /* Feature heading above each sub-table of the expectation grid. */
  .feature-heading { margin: 1.25rem 0 0.4rem; padding-bottom: 0.2rem; border-bottom: 1px solid #e0e0e0;
                     color: #1565c0; }

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

    #[test]
    fn format_utc_renders_real_date_not_placeholder() {
        // 2021-01-01 00:00:00 UTC = 1_609_459_200. Regression guard against the
        // old "-xx-xx" placeholder.
        assert_eq!(format_utc(1_609_459_200), "2021-01-01 00:00:00 UTC");
        // A time-of-day sample: 2026-07-08 13:30:45 UTC.
        assert_eq!(format_utc(1_783_517_445), "2026-07-08 13:30:45 UTC");
    }
}
