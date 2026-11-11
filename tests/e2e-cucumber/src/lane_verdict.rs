// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Whether a run's outcome says anything about the lane it ran on.
//!
//! Scenario results alone cannot say that. Scenarios the capability probe
//! resolves to skip never reach `report.json`, so a run in which every scenario
//! was skipped, or none was selected, has no failure to report and passes.
//! That is what a GPU lane does when its probe cannot read the binary under
//! test, or when the GPU is not visible: every `@requires-gpu` scenario
//! resolves to skip, and the lane is green without having tested a GPU.
//!
//! Two checks close that gap, both pure so every row of their state matrix is a
//! unit test:
//!
//! - [`preflight_verdict`], before any scenario runs: a probe that failed is
//!   fatal on every lane, and so is a lane that declares a GPU (`1` in
//!   [`EXPECT_GPU_ENV`]) where the probe found none.
//! - [`post_run_verdict`], after the run: a run that executed no scenario is
//!   fatal on every lane, and so is a GPU-declaring lane on which no
//!   `@requires-gpu` scenario executed.

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
/// Anything other than `1`, `0`, empty or unset is an error rather than a guess, since a misspelt declaration that read
/// as "undeclared" would switch the GPU check off without anyone noticing.
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

/// How many `@requires-gpu` scenarios the run resolved and how many of those
/// executed.
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
             cucumber --name/--tags filter, E2E_ONLY_LIFECYCLE and E2E_ALSO_RUN_IDS, and the \
             host capability line above"
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
                "this lane declares a GPU ({EXPECT_GPU_ENV}=1) but no @requires-gpu scenario \
                 executed ({} resolved, none ran)",
                gpu.resolved
            ));
        }
        GpuDeclaration::NotExpected => verdict.notes.push(format!(
            "{} of {} @requires-gpu scenario(s) did not run on this lane ({EXPECT_GPU_ENV}=0)",
            gpu.resolved.saturating_sub(gpu.executed),
            gpu.resolved
        )),
        GpuDeclaration::Expected | GpuDeclaration::Undeclared => {}
    }
    verdict
}

#[cfg(test)]
mod tests {
    use super::{
        EXPECT_GPU_ENV, GpuDeclaration, GpuScenarioCounts, Verdict, parse_gpu_declaration,
        post_run_verdict, preflight_verdict,
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
    fn the_variable_name_is_the_one_the_workflows_set() {
        assert_eq!(EXPECT_GPU_ENV, "E2E_EXPECT_GPU");
    }

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
                         E2E_ALSO_RUN_IDS, and the host capability line above"
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
                "this lane declares a GPU (E2E_EXPECT_GPU=1) but no @requires-gpu scenario \
                 executed (7 resolved, none ran)"
            )
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
            ["7 of 7 @requires-gpu scenario(s) did not run on this lane (E2E_EXPECT_GPU=0)"]
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
}
