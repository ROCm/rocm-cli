// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Single-platform HTML report generation.

use std::collections::BTreeMap;
use std::path::Path;

use maud::html;

use crate::components::{feature_group, now_utc, page_shell, stats_table};
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

    let body = html! {
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
    };
    let markup = page_shell("E2E Test Report", body);

    std::fs::write(html_path, markup.into_string())
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use super::generate;

    #[test]
    fn generate_renders_the_body_inside_the_shared_page_shell() {
        // `single_report::generate` is the only other caller of
        // `components::page_shell` besides the consolidated generator.
        // `page_shell_carries_title_and_style` (in components.rs) pins the
        // shell with a synthetic body; this test pins that this call site's
        // real body — the scenario name parsed from the report JSON —
        // actually reaches that shell.
        let json = r#"[{"name":"F","uri":"f.feature","elements":[
            {"name":"a lonely scenario","tags":[],"steps":[
                {"keyword":"Given ","name":"x","result":{"status":"passed"}}
            ]}
        ]}]"#;
        let mut json_file = tempfile::NamedTempFile::new().expect("temp json");
        json_file.write_all(json.as_bytes()).expect("write json");
        let html_file = tempfile::NamedTempFile::new().expect("temp html");

        generate(json_file.path(), html_file.path()).expect("generate succeeds");

        let out = std::fs::read_to_string(html_file.path()).expect("read html");
        assert!(out.contains("<!DOCTYPE html>"), "{out}");
        assert!(out.contains("<title>E2E Test Report</title>"), "{out}");
        assert!(
            out.contains("a lonely scenario"),
            "the body must reach the rendered page, not just the shell:\n{out}"
        );
    }
}
