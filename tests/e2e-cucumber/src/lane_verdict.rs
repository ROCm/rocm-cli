// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Whether a run's outcome says anything about the lane it ran on.
//!
//! Scenario results alone cannot say that. Scenarios the capability probe
//! resolves to skip never reach `report.json`, so a run in which every scenario
//! was skipped, or none was selected, has no failure to report and passes.
//! That is what a GPU lane does when its probe cannot read the binary under
//! test, or when the GPU is not visible: every GPU scenario (`@requires-gpu` or
//! `@requires-multi-gpu`) resolves to skip, and the lane is green without
//! having tested a GPU.
//!
//! Two checks close that gap, both pure so every row of their state matrix is a
//! unit test:
//!
//! - [`preflight_verdict`], before any scenario runs: a probe that failed is
//!   fatal on every lane, and so is a lane that declares a GPU (`1` in
//!   [`EXPECT_GPU_ENV`]) where the probe found none.
//! - [`post_run_verdict`], after the run: a run that executed no scenario is
//!   fatal on every lane, and so is a GPU-declaring lane on which no GPU
//!   scenario (`@requires-gpu` or `@requires-multi-gpu`) executed.
//!
//! The module also holds the decisions `tests/e2e.rs` takes around those two
//! checks, so they are unit-tested too: [`preflight`] (the declaration plus
//! the pre-run verdict), [`needs_gpu`] and [`gpu_scenario_counts`] (the input to
//! the post-run check), [`run_failed`] (the final exit, combining the
//! harness's own XPASS and failure classification with the post-run verdict),
//! [`finish_run`] (the post-run verdict and the `platform.json` manifest that
//! records it), and [`failure_lines`] and [`note_line`] (the exact lines
//! printed, with the annotation escaped by [`workflow_command`]). The harness
//! itself still sets up the run, probes the host, reads its other environment
//! variables, classifies the scenario results, writes the files, and does the
//! printing and exiting; that glue has no test of its own.

use std::ffi::OsStr;

pub use e2e_report::EXPECT_GPU_ENV;

/// What a lane says about its GPU in [`EXPECT_GPU_ENV`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDeclaration {
    /// `1`: the lane runs on GPU hardware, so a run that found or used none is
    /// broken.
    Expected,
    /// `0`: the lane deliberately has no usable GPU, so `@requires-gpu`
    /// scenarios are not applicable on it.
    NotExpected,
    /// Unset or empty: no claim either way (GitHub-hosted lanes, local runs).
    Undeclared,
}

/// Read [`EXPECT_GPU_ENV`]'s value.
///
/// Anything other than `1`, `0`, empty or unset is an error rather than a
/// guess, since a misspelt declaration that read as "undeclared" would switch
/// the GPU check off without anyone noticing.
pub fn parse_gpu_declaration(value: Option<&OsStr>) -> Result<GpuDeclaration, String> {
    match value.map(OsStr::to_str) {
        None | Some(Some("")) => Ok(GpuDeclaration::Undeclared),
        Some(Some("1")) => Ok(GpuDeclaration::Expected),
        Some(Some("0")) => Ok(GpuDeclaration::NotExpected),
        Some(_) => Err(format!(
            "invalid {EXPECT_GPU_ENV} value {:?}: use 1 (this lane has a GPU), 0 (it \
             deliberately has none), or leave it unset",
            value.unwrap_or_default().to_string_lossy()
        )),
    }
}

/// What a check concluded: a reason to fail the run, and lines to print either
/// way.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Verdict {
    /// Why the run must fail; `None` when it may proceed or pass.
    pub fatal: Option<String>,
    /// Informational lines that do not change the outcome.
    pub notes: Vec<String>,
}

/// The verdict before any scenario runs, from the capability probe and the
/// lane's declaration.
///
/// `probe_error` is [`crate::capability::HostCapability::probe_error`];
/// `probe_facts` is the one-line summary of the facts the GPU decision was made
/// from ([`crate::capability::probe_facts_summary`]), quoted when a declared GPU
/// is missing.
#[must_use]
pub fn preflight_verdict(
    probe_error: Option<&str>,
    has_amd_gpu: bool,
    probe_facts: &str,
    declaration: GpuDeclaration,
) -> Verdict {
    let mut verdict = Verdict::default();
    if let Some(error) = probe_error {
        verdict.fatal = Some(format!(
            "the host capability probe could not read the rocm binary under test, so every \
             capability-gated scenario would resolve to skip and the run would pass without \
             testing them: {error}"
        ));
        return verdict;
    }
    match (declaration, has_amd_gpu) {
        (GpuDeclaration::Expected, false) => {
            verdict.fatal = Some(format!(
                "this lane declares a GPU ({EXPECT_GPU_ENV}=1) but the capability probe found \
                 no usable AMD GPU ({probe_facts}), so every @requires-gpu scenario would \
                 resolve to skip"
            ));
        }
        (GpuDeclaration::NotExpected, true) => verdict.notes.push(format!(
            "this lane declares no usable GPU ({EXPECT_GPU_ENV}=0) but the capability probe \
             found one ({probe_facts}); the declaration is stale, and @requires-gpu scenarios \
             will run"
        )),
        (GpuDeclaration::NotExpected, false) => verdict.notes.push(format!(
            "this lane declares no usable GPU ({EXPECT_GPU_ENV}=0); @requires-gpu scenarios \
             are not applicable here"
        )),
        (GpuDeclaration::Expected, true) | (GpuDeclaration::Undeclared, _) => {}
    }
    verdict
}

/// How many GPU scenarios (`@requires-gpu` or `@requires-multi-gpu`) the run
/// resolved and how many of those executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuScenarioCounts {
    /// Scenarios tagged `@requires-gpu` or `@requires-multi-gpu` that the
    /// harness resolved, whether or not they ran.
    pub resolved: usize,
    /// Of those, how many reached `report.json`.
    pub executed: usize,
}

/// The verdict after the run.
///
/// `executed` counts the scenarios that reached `report.json`. `gpu` is `None`
/// when the harness's scenario filter never ran — a cucumber `--name`/`--tags`
/// filter on the command line replaces it — so nothing is known about which
/// scenarios needed a GPU and only the zero-executed check applies.
#[must_use]
pub fn post_run_verdict(
    executed: usize,
    gpu: Option<GpuScenarioCounts>,
    declaration: GpuDeclaration,
) -> Verdict {
    let mut verdict = Verdict::default();
    if executed == 0 {
        verdict.fatal = Some(
            "0 scenarios executed: a run that executes nothing tests nothing. Check the \
             cucumber --name/--tags filter, E2E_ONLY_LIFECYCLE and E2E_ALSO_RUN_IDS"
                .to_owned(),
        );
        return verdict;
    }
    let Some(gpu) = gpu else {
        return verdict;
    };
    match declaration {
        GpuDeclaration::Expected if gpu.executed == 0 => {
            verdict.fatal = Some(format!(
                "this lane declares a GPU ({EXPECT_GPU_ENV}=1) but no @requires-gpu (or \
                 @requires-multi-gpu) scenario executed ({} resolved, none ran)",
                gpu.resolved
            ));
        }
        GpuDeclaration::NotExpected => verdict.notes.push(format!(
            "{} of {} GPU scenario(s) (@requires-gpu or @requires-multi-gpu) did not run on this \
             lane ({EXPECT_GPU_ENV}=0)",
            gpu.resolved.saturating_sub(gpu.executed),
            gpu.resolved
        )),
        GpuDeclaration::Expected | GpuDeclaration::Undeclared => {}
    }
    verdict
}

/// What the harness carries forward from the pre-run check when it passes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Preflight {
    /// The lane's parsed [`EXPECT_GPU_ENV`] declaration, for the post-run check.
    pub declaration: GpuDeclaration,
    /// Lines to print before the run.
    pub notes: Vec<String>,
}

/// The pre-run lane check, from the raw [`EXPECT_GPU_ENV`] value and the
/// probe's results: `Err` with the reason when the run must not start.
///
/// An invalid declaration is reported before anything the probe found, since
/// the verdict depends on the declaration. The harness only prints the notes
/// or the refusal and exits.
pub fn preflight(
    declaration_value: Option<&OsStr>,
    probe_error: Option<&str>,
    has_amd_gpu: bool,
    probe_facts: &str,
) -> Result<Preflight, String> {
    let declaration = parse_gpu_declaration(declaration_value)?;
    let verdict = preflight_verdict(probe_error, has_amd_gpu, probe_facts, declaration);
    match verdict.fatal {
        Some(fatal) => Err(fatal),
        None => Ok(Preflight {
            declaration,
            notes: verdict.notes,
        }),
    }
}

/// Whether a scenario counts toward the post-run GPU check.
///
/// It does when tagged `@requires-gpu` or `@requires-multi-gpu`. `@requires-gfx-target` does not
/// count, since such a scenario reads the chip name without using the GPU.
#[must_use]
pub const fn needs_gpu(decl: &crate::expectation::ScenarioDecl) -> bool {
    decl.requires_gpu || decl.requires_multi_gpu
}

/// Count the GPU scenarios the harness resolved and how many of them executed.
///
/// `resolved` is each scenario the harness's filter saw, as `(id, needs_gpu)`;
/// `executed` is the id of each scenario in `report.json`. `None` when nothing
/// was resolved: a cucumber `--name`/`--tags` filter replaced the harness's
/// filter, so which scenarios need a GPU is unknown (see [`post_run_verdict`]).
/// An executed id the harness did not resolve as a GPU scenario is not counted.
pub fn gpu_scenario_counts<'a>(
    resolved: impl IntoIterator<Item = (&'a str, bool)>,
    executed: impl IntoIterator<Item = &'a str>,
) -> Option<GpuScenarioCounts> {
    let mut any_resolved = false;
    let gpu_ids: std::collections::BTreeSet<&str> = resolved
        .into_iter()
        .inspect(|_| any_resolved = true)
        .filter_map(|(id, gpu)| gpu.then_some(id))
        .collect();
    if !any_resolved {
        return None;
    }
    Some(GpuScenarioCounts {
        resolved: gpu_ids.len(),
        executed: executed
            .into_iter()
            .filter(|id| gpu_ids.contains(id))
            .count(),
    })
}

/// Whether the run exits non-zero after reconciliation.
///
/// Any one of a stale XPASS, an unexpected failure (both counted by the
/// harness) or a fatal post-run verdict is enough. Parse and hook errors exit
/// earlier, in the harness.
#[must_use]
pub const fn run_failed(stale_xpass: usize, unexpected_fail: usize, post_run: &Verdict) -> bool {
    stale_xpass > 0 || unexpected_fail > 0 || post_run.fatal.is_some()
}

/// A GitHub Actions workflow command (`::error::…`, `::notice::…`) carrying
/// `message` as its data.
///
/// The data is escaped as the runner expects — `%` first, then CR and LF — so a
/// probe's stderr tail can neither cut the annotation short at a newline nor
/// have a literal `%0A` in it read back as one.
#[must_use]
pub fn workflow_command(command: &str, message: &str) -> String {
    let data = message
        .replace('%', "%25")
        .replace('\r', "%0D")
        .replace('\n', "%0A");
    format!("::{command}::{data}")
}

/// One scenario the harness's filter resolved, as [`finish_run`] needs it.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedRun<'s> {
    pub id: &'s str,
    pub feature: &'s str,
    pub scenario: &'s str,
    pub engine: &'s str,
    pub expectation: &'s crate::expectation::Expectation,
    /// [`needs_gpu`] of the scenario's tags.
    pub needs_gpu: bool,
}

/// The post-run lane check, and the `platform.json` manifest that records its
/// failure as `lane_failure` so the consolidated report shows the lane red.
///
/// `resolved` is every scenario the harness's filter saw (empty when a CLI
/// filter replaced it); `executed` is the id of each scenario in
/// `report.json`.
#[must_use]
pub fn finish_run<'a>(
    cap: &'a crate::capability::HostCapability,
    versions: crate::capability::PlatformVersions,
    resolved: &[ResolvedRun<'_>],
    executed: &[&str],
    declaration: GpuDeclaration,
) -> (Verdict, crate::expectation::PlatformManifest<'a>) {
    let gpu = gpu_scenario_counts(
        resolved.iter().map(|r| (r.id, r.needs_gpu)),
        executed.iter().copied(),
    );
    let verdict = post_run_verdict(executed.len(), gpu, declaration);
    let manifest = crate::expectation::PlatformManifest {
        platform_slug: &cap.platform_slug,
        capability: cap,
        versions,
        lane_failure: verdict.fatal.clone(),
        expectations: resolved
            .iter()
            .map(|r| {
                crate::expectation::ResolvedScenario::new(
                    r.id,
                    r.feature,
                    r.scenario,
                    r.engine,
                    r.expectation,
                )
            })
            .collect(),
    };
    (verdict, manifest)
}

/// The lines the harness prints when a lane check fails: a GitHub `::error::`
/// annotation, then the same message as a plain line for anyone reading the
/// log outside Actions.
#[must_use]
pub fn failure_lines(message: &str) -> [String; 2] {
    let line = format!("E2E lane check failed: {message}");
    [workflow_command("error", &line), line]
}

/// The `::notice::` line the harness prints for a lane-check note.
#[must_use]
pub fn note_line(note: &str) -> String {
    workflow_command("notice", &format!("E2E lane: {note}"))
}

#[cfg(test)]
mod tests {
    use super::{
        GpuDeclaration, GpuScenarioCounts, Preflight, ResolvedRun, Verdict, failure_lines,
        finish_run, gpu_scenario_counts, needs_gpu, note_line, parse_gpu_declaration,
        post_run_verdict, preflight, preflight_verdict, run_failed, workflow_command,
    };
    use std::ffi::OsStr;

    const FACTS: &str =
        "detected_gfx_target=<none>, wsl=false, driver_status=<none>, no visibility mask";
    const PROBE_ERROR: &str = "`rocm examine` exited with exit status: 1";

    const ALL: [GpuDeclaration; 3] = [
        GpuDeclaration::Expected,
        GpuDeclaration::NotExpected,
        GpuDeclaration::Undeclared,
    ];

    #[test]
    fn only_one_zero_and_unset_or_empty_are_declarations() {
        assert_eq!(parse_gpu_declaration(None), Ok(GpuDeclaration::Undeclared));
        assert_eq!(
            parse_gpu_declaration(Some(OsStr::new(""))),
            Ok(GpuDeclaration::Undeclared)
        );
        assert_eq!(
            parse_gpu_declaration(Some(OsStr::new("1"))),
            Ok(GpuDeclaration::Expected)
        );
        assert_eq!(
            parse_gpu_declaration(Some(OsStr::new("0"))),
            Ok(GpuDeclaration::NotExpected)
        );
    }

    #[test]
    fn a_misspelt_declaration_is_an_error_not_undeclared() {
        for value in ["true", "yes", " 1", "2"] {
            assert_eq!(
                parse_gpu_declaration(Some(OsStr::new(value))),
                Err(format!(
                    "invalid E2E_EXPECT_GPU value {value:?}: use 1 (this lane has a GPU), 0 (it \
                     deliberately has none), or leave it unset"
                )),
            );
        }
    }

    // Row: probe failed (non-zero exit, spawn failure, empty output, or
    // `engines list` failing after `examine` succeeded all reach here as one
    // `probe_error`). Fatal under every declaration, GPU or not.
    #[test]
    fn a_failed_probe_is_fatal_under_every_declaration() {
        for declaration in ALL {
            for has_amd_gpu in [false, true] {
                let verdict = preflight_verdict(Some(PROBE_ERROR), has_amd_gpu, FACTS, declaration);
                assert_eq!(
                    verdict,
                    Verdict {
                        fatal: Some(format!(
                            "the host capability probe could not read the rocm binary under \
                             test, so every capability-gated scenario would resolve to skip and \
                             the run would pass without testing them: {PROBE_ERROR}"
                        )),
                        notes: vec![],
                    },
                    "{declaration:?}, gpu={has_amd_gpu}"
                );
            }
        }
    }

    // Row: probe succeeded, no usable GPU, declared 1.
    #[test]
    fn a_declared_gpu_the_probe_did_not_find_is_fatal_and_names_the_facts() {
        let verdict = preflight_verdict(None, false, FACTS, GpuDeclaration::Expected);
        assert_eq!(
            verdict.fatal.as_deref(),
            Some(
                "this lane declares a GPU (E2E_EXPECT_GPU=1) but the capability probe found no \
                 usable AMD GPU (detected_gfx_target=<none>, wsl=false, driver_status=<none>, \
                 no visibility mask), so every @requires-gpu scenario would resolve to skip"
            )
        );
        assert!(verdict.notes.is_empty());
    }

    // Row: probe succeeded with a GPU, declared 1 — runs, nothing to say.
    #[test]
    fn a_declared_gpu_that_is_present_runs_without_comment() {
        assert_eq!(
            preflight_verdict(None, true, FACTS, GpuDeclaration::Expected),
            Verdict::default()
        );
    }

    // Row: probe succeeded with a GPU, declared 0 — runs, with a stale note.
    #[test]
    fn a_declared_absent_gpu_that_is_present_is_a_note_not_a_failure() {
        let verdict = preflight_verdict(None, true, FACTS, GpuDeclaration::NotExpected);
        assert_eq!(verdict.fatal, None);
        // Pins the wording only. That @requires-gpu scenarios do run on a host
        // whose probe found a GPU, whatever the declaration, is proved by
        // `expectation::tests::requires_gpu_runs_where_the_probe_found_a_gpu_and_skips_where_it_did_not`.
        assert_eq!(
            verdict.notes,
            [format!(
                "this lane declares no usable GPU (E2E_EXPECT_GPU=0) but the capability probe \
                 found one ({FACTS}); the declaration is stale, and @requires-gpu scenarios will \
                 run"
            )]
        );
    }

    // Row: probe succeeded, no GPU, declared 0 — runs, says why GPU scenarios
    // are not applicable.
    #[test]
    fn a_declared_absent_gpu_that_is_absent_runs_with_a_note() {
        let verdict = preflight_verdict(None, false, FACTS, GpuDeclaration::NotExpected);
        assert_eq!(verdict.fatal, None);
        // Pins the wording only. That @requires-gpu scenarios resolve to skip
        // (not applicable) where the probe found no GPU is proved by
        // `expectation::tests::requires_gpu_runs_where_the_probe_found_a_gpu_and_skips_where_it_did_not`.
        assert_eq!(
            verdict.notes,
            [
                "this lane declares no usable GPU (E2E_EXPECT_GPU=0); @requires-gpu scenarios are \
              not applicable here"
            ]
        );
    }

    // Rows: probe succeeded, undeclared — today's behaviour, GPU or not.
    #[test]
    fn an_undeclared_lane_runs_without_comment_either_way() {
        for has_amd_gpu in [false, true] {
            assert_eq!(
                preflight_verdict(None, has_amd_gpu, FACTS, GpuDeclaration::Undeclared),
                Verdict::default(),
                "gpu={has_amd_gpu}"
            );
        }
    }

    // Post-run row: nothing executed — fatal under every declaration, whether
    // or not the harness's own filter ran (a `-n` filter matching nothing is
    // the `None` case).
    #[test]
    fn a_run_that_executed_nothing_is_fatal_under_every_declaration() {
        let gpu_cases = [
            None,
            Some(GpuScenarioCounts {
                resolved: 0,
                executed: 0,
            }),
            Some(GpuScenarioCounts {
                resolved: 4,
                executed: 0,
            }),
        ];
        for declaration in ALL {
            for gpu in gpu_cases {
                let verdict = post_run_verdict(0, gpu, declaration);
                assert_eq!(
                    verdict.fatal.as_deref(),
                    Some(
                        "0 scenarios executed: a run that executes nothing tests nothing. Check \
                         the cucumber --name/--tags filter, E2E_ONLY_LIFECYCLE and \
                         E2E_ALSO_RUN_IDS"
                    ),
                    "{declaration:?}, {gpu:?}"
                );
            }
        }
    }

    // Post-run row: declared 1, scenarios ran but no GPU one did.
    #[test]
    fn a_gpu_lane_on_which_no_gpu_scenario_ran_is_fatal() {
        let verdict = post_run_verdict(
            12,
            Some(GpuScenarioCounts {
                resolved: 7,
                executed: 0,
            }),
            GpuDeclaration::Expected,
        );
        assert_eq!(
            verdict.fatal.as_deref(),
            Some(
                "this lane declares a GPU (E2E_EXPECT_GPU=1) but no @requires-gpu (or \
                 @requires-multi-gpu) scenario executed (7 resolved, none ran)"
            )
        );
    }

    // Post-run row: declared 1, scenarios ran, and the harness's filter ran but
    // resolved no GPU scenario at all (none selected): still fatal, since the
    // lane promised GPU coverage it did not get.
    #[test]
    fn a_gpu_lane_that_resolved_no_gpu_scenario_is_fatal() {
        let verdict = post_run_verdict(
            5,
            Some(GpuScenarioCounts {
                resolved: 0,
                executed: 0,
            }),
            GpuDeclaration::Expected,
        );
        assert_eq!(
            verdict,
            Verdict {
                fatal: Some(
                    "this lane declares a GPU (E2E_EXPECT_GPU=1) but no @requires-gpu (or \
                     @requires-multi-gpu) scenario executed (0 resolved, none ran)"
                        .to_owned()
                ),
                notes: vec![],
            }
        );
    }

    #[test]
    fn a_gpu_lane_on_which_a_gpu_scenario_ran_passes() {
        assert_eq!(
            post_run_verdict(
                12,
                Some(GpuScenarioCounts {
                    resolved: 7,
                    executed: 1,
                }),
                GpuDeclaration::Expected,
            ),
            Verdict::default()
        );
    }

    // Post-run row: a CLI filter bypassed the harness's filter, so which
    // scenarios need a GPU is unknown — only the zero check applies.
    #[test]
    fn a_cli_filtered_run_is_not_held_to_the_gpu_check() {
        for declaration in ALL {
            assert_eq!(
                post_run_verdict(3, None, declaration),
                Verdict::default(),
                "{declaration:?}"
            );
        }
    }

    #[test]
    fn a_no_gpu_lane_says_how_many_gpu_scenarios_did_not_run() {
        let verdict = post_run_verdict(
            12,
            Some(GpuScenarioCounts {
                resolved: 7,
                executed: 0,
            }),
            GpuDeclaration::NotExpected,
        );
        assert_eq!(verdict.fatal, None);
        assert_eq!(
            verdict.notes,
            [
                "7 of 7 GPU scenario(s) (@requires-gpu or @requires-multi-gpu) did not run on \
                 this lane (E2E_EXPECT_GPU=0)"
            ]
        );
        // Some ran: the note counts only the ones that did not.
        let verdict = post_run_verdict(
            12,
            Some(GpuScenarioCounts {
                resolved: 7,
                executed: 3,
            }),
            GpuDeclaration::NotExpected,
        );
        assert_eq!(
            verdict.notes,
            [
                "4 of 7 GPU scenario(s) (@requires-gpu or @requires-multi-gpu) did not run on \
                 this lane (E2E_EXPECT_GPU=0)"
            ]
        );
    }

    #[test]
    fn an_undeclared_lane_is_not_held_to_the_gpu_check() {
        assert_eq!(
            post_run_verdict(
                12,
                Some(GpuScenarioCounts {
                    resolved: 7,
                    executed: 0,
                }),
                GpuDeclaration::Undeclared,
            ),
            Verdict::default()
        );
    }

    // --- the harness glue, as pure functions ---

    #[test]
    fn preflight_refuses_an_invalid_declaration_before_reading_the_probe() {
        let invalid = Err(
            "invalid E2E_EXPECT_GPU value \"yes\": use 1 (this lane has a GPU), \
                           0 (it deliberately has none), or leave it unset"
                .to_owned(),
        );
        // A probe that also failed, or found no GPU, must not displace the
        // declaration error: the declaration is checked first.
        for (probe_error, has_amd_gpu) in [(None, true), (Some(PROBE_ERROR), false), (None, false)]
        {
            assert_eq!(
                preflight(Some(OsStr::new("yes")), probe_error, has_amd_gpu, FACTS),
                invalid,
                "probe_error={probe_error:?}, gpu={has_amd_gpu}"
            );
        }
    }

    #[test]
    fn preflight_refuses_a_failed_probe_and_a_missing_declared_gpu() {
        let refused = preflight(None, Some(PROBE_ERROR), true, FACTS).expect_err("probe failed");
        assert!(refused.ends_with(PROBE_ERROR), "{refused}");
        let refused =
            preflight(Some(OsStr::new("1")), None, false, FACTS).expect_err("declared GPU missing");
        assert!(
            refused.starts_with("this lane declares a GPU (E2E_EXPECT_GPU=1)"),
            "{refused}"
        );
    }

    #[test]
    fn preflight_carries_the_declaration_and_notes_into_the_run() {
        assert_eq!(
            preflight(Some(OsStr::new("0")), None, false, FACTS),
            Ok(Preflight {
                declaration: GpuDeclaration::NotExpected,
                notes: vec![
                    "this lane declares no usable GPU (E2E_EXPECT_GPU=0); @requires-gpu \
                     scenarios are not applicable here"
                        .to_owned()
                ],
            })
        );
        assert_eq!(
            preflight(Some(OsStr::new("1")), None, true, FACTS),
            Ok(Preflight {
                declaration: GpuDeclaration::Expected,
                notes: vec![],
            })
        );
    }

    #[test]
    fn gpu_and_multi_gpu_scenarios_need_a_gpu_and_a_gfx_target_one_does_not() {
        let decl = crate::expectation::ScenarioDecl::from_tags;
        assert!(needs_gpu(&decl(&["id:a", "requires-gpu"])));
        assert!(needs_gpu(&decl(&["id:b", "requires-multi-gpu"])));
        assert!(!needs_gpu(&decl(&["id:c", "requires-gfx-target"])));
        assert!(!needs_gpu(&decl(&["id:d"])));
    }

    #[test]
    fn a_cli_filtered_run_resolved_nothing_so_has_no_gpu_counts() {
        assert_eq!(gpu_scenario_counts([], ["serve-a", "serve-b"]), None);
    }

    #[test]
    fn gpu_counts_take_resolved_from_the_filter_and_executed_from_the_report() {
        let resolved = [
            ("gpu-ran", true),
            ("multi-gpu-only", true),
            ("gpu-skipped", true),
            ("cpu-ran", false),
        ];
        // `cpu-ran` executed but is not a GPU scenario, and `untagged` was
        // never resolved: neither counts as an executed GPU scenario.
        let executed = ["gpu-ran", "multi-gpu-only", "cpu-ran", "untagged"];
        assert_eq!(
            gpu_scenario_counts(resolved, executed),
            Some(GpuScenarioCounts {
                resolved: 3,
                executed: 2,
            })
        );
    }

    #[test]
    fn gpu_counts_with_nothing_executed_are_zero_not_none() {
        assert_eq!(
            gpu_scenario_counts([("gpu", true), ("cpu", false)], []),
            Some(GpuScenarioCounts {
                resolved: 1,
                executed: 0,
            })
        );
        assert_eq!(
            gpu_scenario_counts([("cpu", false)], ["cpu"]),
            Some(GpuScenarioCounts {
                resolved: 0,
                executed: 0,
            })
        );
    }

    #[test]
    fn the_run_fails_on_any_one_of_its_three_reasons() {
        let fatal = Verdict {
            fatal: Some("0 scenarios executed".to_owned()),
            notes: vec![],
        };
        let clean = Verdict::default();
        assert!(!run_failed(0, 0, &clean));
        assert!(run_failed(1, 0, &clean));
        assert!(run_failed(0, 1, &clean));
        assert!(run_failed(0, 0, &fatal));
        // A note alone never fails the run.
        let noted = Verdict {
            fatal: None,
            notes: vec!["stale".to_owned()],
        };
        assert!(!run_failed(0, 0, &noted));
    }

    #[test]
    fn workflow_command_data_escapes_percent_cr_and_lf() {
        assert_eq!(
            workflow_command("error", "100% failed\r\nstderr: a%0Ab"),
            "::error::100%25 failed%0D%0Astderr: a%250Ab"
        );
        assert_eq!(workflow_command("notice", "plain"), "::notice::plain");
    }

    #[test]
    fn a_failure_prints_an_escaped_annotation_then_the_plain_message() {
        assert_eq!(
            failure_lines("probe failed\n--- stderr ---\n100% full"),
            [
                "::error::E2E lane check failed: probe failed%0A--- stderr ---%0A100%25 full"
                    .to_owned(),
                "E2E lane check failed: probe failed\n--- stderr ---\n100% full".to_owned(),
            ]
        );
    }

    #[test]
    fn a_note_prints_as_an_escaped_notice() {
        assert_eq!(
            note_line("7 of 7 GPU scenario(s) did not run (100%)"),
            "::notice::E2E lane: 7 of 7 GPU scenario(s) did not run (100%25)"
        );
    }

    /// `finish_run`'s manifest, written as the harness writes it and read back
    /// with the consolidated report's own loader: the post-run lane failure
    /// it records makes the lane red there, and a lane that passed the check
    /// stays green. Ties the producer to the consumer through the function
    /// `tests/e2e.rs` calls.
    #[test]
    fn finish_runs_manifest_reads_back_as_the_verdict_in_the_report() {
        let cap = crate::capability::HostCapability {
            os_family: "linux".into(),
            is_wsl: false,
            gfx_target: Some("gfx1151".into()),
            has_amd_gpu: true,
            amd_gpu_count: Some(1),
            available_engines: vec!["lemonade".into()],
            effective_serve_engine: "lemonade".into(),
            platform_slug: "strix-halo-linux".into(),
            oom_fault_injection: false,
            case_sensitive_fs: true,
            dir_permissions_bind: true,
            driver_status: String::new(),
            probe_error: None,
        };
        let pass = crate::expectation::Expectation::ExpectPass;
        let skip = crate::expectation::Expectation::Skip {
            reason: "no usable AMD GPU".to_owned(),
        };
        let resolved_with = |gpu_expectation| {
            [
                ResolvedRun {
                    id: "examine-version",
                    feature: "Examine",
                    scenario: "examine-01",
                    engine: "lemonade",
                    expectation: &pass,
                    needs_gpu: false,
                },
                ResolvedRun {
                    id: "serve-gpu",
                    feature: "Serving",
                    scenario: "serve-01",
                    engine: "vllm",
                    expectation: gpu_expectation,
                    needs_gpu: true,
                },
            ]
        };
        // Writes what the harness writes and reads it back: the verdict, the
        // report's status, and `platform.json`'s id -> expected mapping.
        // `executed` is also what report.json holds; every listed id passed.
        let status_of = |resolved: &[ResolvedRun<'_>], executed: &[&str]| {
            let (verdict, manifest) = finish_run(
                &cap,
                crate::capability::PlatformVersions::default(),
                resolved,
                executed,
                GpuDeclaration::Expected,
            );
            let dir = tempfile::TempDir::with_prefix("rocm-e2e-manifest-").expect("temp dir");
            let elements: Vec<String> = executed
                .iter()
                .map(|id| {
                    format!(
                        r#"{{"name":"{id}","tags":[{{"name":"id:{id}"}}],"steps":[{{"keyword":"Given ","name":"x","result":{{"status":"passed"}}}}]}}"#
                    )
                })
                .collect();
            let report = dir.path().join("report.json");
            std::fs::write(
                &report,
                format!(
                    r#"[{{"name":"F","uri":"f.feature","elements":[{}]}}]"#,
                    elements.join(",")
                ),
            )
            .expect("report.json");
            let platform = dir.path().join("platform.json");
            std::fs::write(
                &platform,
                serde_json::to_string_pretty(&manifest).expect("serializes"),
            )
            .expect("platform.json");
            let written: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&platform).expect("read back"))
                    .expect("valid json");
            let mapping: Vec<Vec<String>> = written["expectations"]
                .as_array()
                .expect("an expectations array")
                .iter()
                .map(|e| {
                    ["id", "feature", "scenario", "effective_engine", "expected"]
                        .iter()
                        .map(|key| e[*key].as_str().unwrap_or_default().to_owned())
                        .collect()
                })
                .collect();
            (
                verdict,
                e2e_report::platform_status("e2e-report", &report),
                mapping,
            )
        };
        // id, feature, scenario, effective_engine, expected — per expectation.
        let mapping_of = |rows: &[[&str; 5]]| -> Vec<Vec<String>> {
            rows.iter()
                .map(|row| row.iter().map(|field| (*field).to_owned()).collect())
                .collect()
        };

        // Every cell is pass or skip, so only the recorded lane failure can
        // make this row FAIL.
        let reason = "this lane declares a GPU (E2E_EXPECT_GPU=1) but no @requires-gpu (or \
                      @requires-multi-gpu) scenario executed (1 resolved, none ran)";
        let (verdict, failed, mapping) = status_of(&resolved_with(&skip), &["examine-version"]);
        assert_eq!(verdict.fatal.as_deref(), Some(reason));
        assert_eq!(
            mapping,
            mapping_of(&[
                [
                    "examine-version",
                    "Examine",
                    "examine-01",
                    "lemonade",
                    "pass"
                ],
                ["serve-gpu", "Serving", "serve-01", "vllm", "skip"],
            ])
        );
        assert_eq!(
            failed,
            e2e_report::PlatformStatus {
                ok: false,
                status: "FAIL",
                lane_failure: Some(reason.to_owned()),
            }
        );

        let (verdict, passed, mapping) =
            status_of(&resolved_with(&pass), &["examine-version", "serve-gpu"]);
        assert_eq!(verdict.fatal, None);
        assert_eq!(
            mapping,
            mapping_of(&[
                [
                    "examine-version",
                    "Examine",
                    "examine-01",
                    "lemonade",
                    "pass"
                ],
                ["serve-gpu", "Serving", "serve-01", "vllm", "pass"],
            ])
        );
        assert_eq!(
            passed,
            e2e_report::PlatformStatus {
                ok: true,
                status: "PASS",
                lane_failure: None,
            }
        );
    }
}
