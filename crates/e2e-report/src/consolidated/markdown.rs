// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Markdown renderer for the consolidated report: [`consolidated_summary_markdown`]
//! (the GitHub-flavoured table piped into `$GITHUB_STEP_SUMMARY`), the
//! expectation grid and the Scenario reference section it appends. A child of
//! `consolidated`, so it reads the model types defined there (`PlatformReport`,
//! `Grid`, `CellOutcome`) without those being widened, and it pulls in the
//! command-coverage table from `command_coverage.rs`. Re-exported from the
//! crate root by `lib.rs`.

use super::{CellOutcome, Grid, LEGEND_ENTRIES, PlatformReports, ScenarioReference, platform_cell};
use crate::command_coverage::command_coverage_markdown;

/// Render the consolidated result from a [`PlatformReports`] bundle as a
/// GitHub-flavoured markdown table, for piping into `$GITHUB_STEP_SUMMARY`.
/// Same bundle as [`generate_consolidated`].
///
/// Reads nothing from disk; everything it renders comes from the bundle
/// (see [`load_platform_reports`]).
///
/// [`generate_consolidated`]: super::html::generate_consolidated
/// [`load_platform_reports`]: super::load_platform_reports
pub fn consolidated_summary_markdown(bundle: &PlatformReports) -> String {
    use std::fmt::Write as _;

    let reports = &bundle.reports;
    let mut out = String::from("## E2E consolidated report\n\n");
    if reports.is_empty() {
        out.push_str("_No per-platform report.json files were found to consolidate._\n");
        return out;
    }

    out.push_str("| Platform | OS | Total | Pass | Fail | Skip | Xfail | Status |\n");
    out.push_str("|---|---|--:|--:|--:|--:|--:|:--|\n");
    let (mut t_total, mut t_pass, mut t_fail, mut t_skip, mut t_xfail) = (0, 0, 0, 0, 0);
    for r in reports {
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
    for r in reports {
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
    // "where should each test pass / not matter / not run". The scenario
    // map supplies names for the grid's Scenario column (name + id link).
    out.push_str(&expectation_grid_markdown(&bundle.grid, &bundle.scenarios));

    out.push_str(&command_coverage_markdown(reports));

    // Scenario reference LAST (after command coverage): each id's actual Gherkin
    // scenario, anchored so the grid's id links resolve here.
    out.push_str(&scenario_reference_markdown(
        &bundle.grid,
        &bundle.scenarios,
    ));

    out
}

/// Render the reconciled (scenario-id × platform) expectation grid as markdown,
/// plus a "needs attention" list of every XPASS / unexpected-fail / ran-when-NA.
/// Empty string only when there is truly nothing to report — no column/row
/// AND no excluded platform to caption (see [`Grid::nothing_to_report`]); a
/// report where every platform lacks a usable `platform.json` still renders a
/// heading and caption naming them.
fn expectation_grid_markdown(grid: &Grid, scenarios: &ScenarioReference) -> String {
    use std::fmt::Write as _;

    // Not `grid.is_empty()`: see the matching comment in `expectation_grid_html`
    // — a report where every platform lacks a `platform.json` still has an
    // `excluded_note` worth showing, even with no columns/groups to table.
    if grid.nothing_to_report() {
        return String::new();
    }

    let mut out = String::from("\n### Expectation grid (scenario × platform)\n\n");
    if !grid.is_empty() {
        let legend = LEGEND_ENTRIES
            .iter()
            .map(|(outcome, text)| format!("{} {text}", outcome.glyph()))
            .collect::<Vec<_>>()
            .join(" · ");
        let _ = writeln!(out, "_{legend}_\n");
    }
    if let Some(note) = grid.excluded_note() {
        let _ = writeln!(out, "_{note}_\n");
    }

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
        for (platform_label, id, outcome, detail) in problems {
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
            let _ = writeln!(
                out,
                "- **{kind}** on `{platform_label}`: `{id}`{bug}{engine}{reason}"
            );
        }
    }

    out
}

/// Render the Scenario reference section: each scenario `@id` with its actual
/// Gherkin scenario (name + steps), anchored by the id (`##### <id>` → GitHub
/// anchor `#<id>`) so the grid's id links resolve. Grouped by feature and ordered
/// like the grid. Empty when no expectation grid exists.
fn scenario_reference_markdown(grid: &Grid, scenarios: &ScenarioReference) -> String {
    use std::fmt::Write as _;

    // Only emit when there is a grid to reference (platform.json sidecars present),
    // matching where the id links are generated.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consolidated::load_platform_reports;
    use crate::consolidated::test_support::{
        feature_json, grid_order, write_platform, write_report,
    };

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
        let md = consolidated_summary_markdown(&load_platform_reports(&inputs));
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
        let md = consolidated_summary_markdown(&load_platform_reports(&inputs));
        assert!(md.contains("Needs attention"));
        assert!(md.contains("XPASS"));
    }

    #[test]
    fn consolidated_summary_markdown_empty_inputs() {
        let md = consolidated_summary_markdown(&load_platform_reports(&[]));
        assert!(md.contains("No per-platform report.json files"));
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
        let md = consolidated_summary_markdown(&load_platform_reports(&inputs));
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
    fn consolidated_summary_markdown_orders_rows_by_report_sort_key() {
        // report_sort_key_orders_by_every_dimension already pins all four
        // tuple elements via a direct call to report_sort_key with synthetic
        // PlatformReports. This test instead goes through the public
        // consolidated_summary_markdown entry point with realistic
        // artifact-name fixtures, so a regression in load_platform_reports'
        // use of report_sort_key — e.g. calling it with the wrong field order,
        // or not calling it at all — would still be caught even if
        // report_sort_key itself stayed correct in isolation.
        //
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
        // MI300X/release and R9700/nightly — pin channel's precedence below
        // platform: "nightly" sorts before "release", so if channel outranked
        // platform in the key, R9700 would sort first; "MI300X" sorts before
        // "R9700" by platform, so the correct key keeps MI300X first
        // regardless. A seventh row, same platform/os as the MI300X row but
        // known_bugs=true and channel=nightly, pins that known_bugs outranks
        // channel too: "nightly" sorts before "release", so if channel
        // outranked known_bugs, this row would sort before the MI300X/release
        // row despite known_bugs=false < true. Channel's own effect on
        // ordering — as a tie-breaker once platform/os/known_bugs are equal —
        // is pinned by generate_consolidated_writes_html and
        // command_coverage_distinguishes_channels_on_same_platform, not
        // repeated here.
        // No platform.json sidecar, so display_counts falls back to raw junit
        // stats and each row's distinct scenario count (1/2/3/4/5/6/7) tells
        // the rows apart in the rendered table.
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
        let mi300x_known_bugs_nightly_report = write_report(&feature_json(&[
            (&[], &["passed"]),
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
            (
                "e2e-gpu-known-bugs-nightly-report".to_string(),
                mi300x_known_bugs_nightly_report.path().to_path_buf(),
            ),
        ];

        let md = consolidated_summary_markdown(&load_platform_reports(&inputs));
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
        let mi300x_known_bugs_nightly_pos = md
            .find("| MI300X (nightly) | Linux | 7 |")
            .expect("MI300X known-bugs nightly row");
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
        assert!(
            mi300x_pos < mi300x_known_bugs_nightly_pos,
            "same platform/os: known_bugs=false must sort before known_bugs=true, even though \
             the known_bugs=true row's channel (\"nightly\") sorts before the known_bugs=false \
             row's (\"release\"):\n{md}"
        );
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
        let md =
            consolidated_summary_markdown(&load_platform_reports(&[("mock".to_string(), path)]));
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
        let md =
            consolidated_summary_markdown(&load_platform_reports(&[("mock".to_string(), path)]));

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
        let md = consolidated_summary_markdown(&load_platform_reports(&inputs));
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
        let md = consolidated_summary_markdown(&load_platform_reports(&inputs));
        assert!(md.contains("⚠️XPASS"), "grid cell should show XPASS:\n{md}");
        // Canonical label ("Strix Halo Linux", via parse_descriptor's fallback
        // titlecasing — fallback_descriptor always assigns Linux), not the raw
        // "strix-halo" slug, and WITH the OS (the grid has no separate OS
        // column, unlike the summary matrix) — see issue #702.
        assert!(
            md.contains("**XPASS** on `Strix Halo Linux`: `serve-default` (EAI-7333)"),
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
        let md = consolidated_summary_markdown(&load_platform_reports(&inputs));
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
        let md = consolidated_summary_markdown(&load_platform_reports(&inputs));
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
        let md = consolidated_summary_markdown(&load_platform_reports(&inputs));
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
        let md = consolidated_summary_markdown(&load_platform_reports(&inputs));
        assert!(
            md.contains("[`gpu-only`](#gpu-only) | n/a |"),
            "skip should render as n/a:\n{md}"
        );
        assert!(md.contains("[`ran-here`](#ran-here) | ✅ |"));
    }
}
