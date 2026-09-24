// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Single-platform HTML report generation.

use std::collections::BTreeMap;
use std::path::Path;

use maud::{DOCTYPE, PreEscaped, html};

use crate::components::{STYLE, feature_group, now_utc, stats_table};
use crate::parse::{Feature, Stats, scenario_duration, scenario_status};

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
