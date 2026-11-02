// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use rocm_core::{
    AppPaths, DependencyViolation, check_dependencies, split_local_version, violation_subject,
    violations_requiring,
};

use crate::ENGINE_NAME;
use crate::runtime::{VllmRuntime, recorded_sdk_torch_build, runtime_is_managed};

/// Whether a resolvable vLLM environment still needs an install pass, and what to tell
/// the user about why.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct RepairAssessment {
    /// The environment holds vLLM but not the dependencies vLLM declares, so the
    /// install must run even though `reinstall` was not requested.
    pub(crate) needed: bool,
    /// Findings to surface with the install response.
    pub(crate) notes: Vec<String>,
}

/// Decide whether an already-resolvable vLLM environment must be reinstalled.
///
/// Only managed environments are assessed. An external environment belongs to the user:
/// rocm-cli reports on it but does not rewrite its packages, which is the very failure
/// mode this check exists to catch.
pub(crate) fn assess_runtime_repair(runtime: &VllmRuntime) -> RepairAssessment {
    if !runtime_is_managed(runtime) {
        return RepairAssessment::default();
    }
    let Some(python) = runtime.python_executable.as_ref() else {
        return RepairAssessment::default();
    };
    let paths = match AppPaths::discover() {
        Ok(paths) => paths,
        Err(error) => return unverified_repair(&error.to_string()),
    };
    match check_dependencies(&paths, python) {
        Ok(violations) => repair_from_violations(
            &violations,
            recorded_sdk_torch_build(&runtime.runtime_id).as_deref(),
            torch_alignment_disabled(),
        ),
        // An unusable `uv` or an offline host must not block an install that would
        // otherwise succeed; report that the check did not run and carry on as before.
        Err(error) => unverified_repair(&error.to_string()),
    }
}

/// The package whose build the SDK and the engine both have an opinion about.
const TORCH_PACKAGE: &str = "torch";

/// Whether the user has opted out of rocm-cli choosing this runtime's torch.
///
/// The CLI's own opt-out is the same call, not a matching one: the engine cannot
/// call into the binary that owns the alignment, and a duplicated read is a
/// contract that drifts. [`rocm_core::torch_alignment_disabled`] carries the rest.
fn torch_alignment_disabled() -> bool {
    rocm_core::torch_alignment_disabled()
}

/// Whether this violation is the torch divergence rocm-cli deliberately leaves behind.
///
/// After an engine install, rocm-cli puts back the SDK's *build* of the torch release
/// the engine pins, because the engine's build cannot open a device against the
/// installed SDK libraries. The engine's metadata pins an exact version and cannot
/// express "same release, the SDK's build", so `uv pip check` reports the result as
/// unsatisfied forever. Treating that as a defect makes the two mechanisms fight: the
/// engine reinstalls torch to its own build, rocm-cli puts the SDK's back, and the
/// next invocation starts over — two full torch-stack flips a run, and a warning
/// claiming a repair that undid the intended state.
///
/// With the alignment disabled the rule is simply "any torch pin": see below.
///
/// Otherwise three conditions, all required, and they are exactly the rule rocm-cli
/// applies: take the *release* from the engine's pin and the *build* from the SDK.
///
/// The violation must be about torch — any other unmet requirement is real. The
/// installed release must be the one the engine pins; an SDK torch of a *different*
/// release is the separate bug where the engine cannot accept what the SDK installed,
/// and a reinstall is the right answer there. And the installed build must be the one
/// the runtime's manifest records for the SDK; a torch from neither side is the
/// breakage this check exists to catch. Miss any one and the engine either fights the
/// alignment or silently accepts a runtime that cannot serve.
fn is_intended_torch_divergence(
    detail: &str,
    sdk_torch_build: Option<&str>,
    torch_alignment_disabled: bool,
) -> bool {
    let Some(subject) = violation_subject(detail) else {
        return false;
    };
    if !subject.package.eq_ignore_ascii_case(TORCH_PACKAGE) {
        return false;
    }
    // Opted out, so rocm-cli does not choose this runtime's torch and no build it
    // holds can be wrong *here*: the pin is unmet because the user meant it to be.
    // The build and release tests below are the aligned-case rule — asking a
    // hand-installed torch to match the SDK's build would fail every time, and the
    // reinstall that followed would install the engine's build over exactly the torch
    // the opt-out exists to keep. Only torch is spared: the check above already
    // rejected every other package, so an unrelated vLLM-owned defect still repairs.
    if torch_alignment_disabled {
        return true;
    }
    let Some(sdk_torch_build) = sdk_torch_build else {
        return false;
    };
    let (Some(required), Some(installed)) =
        (subject.required.as_deref(), subject.installed.as_deref())
    else {
        return false;
    };
    let (installed_release, Some(installed_build)) = split_local_version(installed) else {
        return false;
    };
    installed_build == sdk_torch_build && installed_release == split_local_version(required).0
}

/// The repair decision for a set of violations found in the environment.
///
/// `sdk_torch_build` is the build identifier the runtime's manifest records for the
/// SDK's torch, or `None` when it cannot be determined — in which case nothing is
/// treated as intended and the previous behaviour stands.
///
/// `torch_alignment_disabled` is the user's opt-out. It changes which violations count
/// as defects, never whether defects are acted on: a torch pin stops being one, and
/// everything else vLLM requires is assessed exactly as before. Returning early on the
/// opt-out instead would hide a broken torchvision behind an unrelated preference.
fn repair_from_violations(
    violations: &[DependencyViolation],
    sdk_torch_build: Option<&str>,
    torch_alignment_disabled: bool,
) -> RepairAssessment {
    let owned = violations_requiring(violations, ENGINE_NAME);
    if owned.is_empty() {
        return RepairAssessment::default();
    }
    let (intended, defects): (Vec<&DependencyViolation>, Vec<&DependencyViolation>) =
        owned.into_iter().partition(|violation| {
            is_intended_torch_divergence(
                &violation.detail,
                sdk_torch_build,
                torch_alignment_disabled,
            )
        });

    if defects.is_empty() {
        // Nothing to repair. Reinstalling here would replace that torch with the
        // engine's build and hand back a runtime nobody asked for.
        //
        // Which sentence is true depends on whose torch this is. Under the alignment it
        // is the SDK's and rocm-cli put it there; under the opt-out it is the user's and
        // rocm-cli never touched it. Reusing the first line for the second case would
        // tell a user who hand-installed torch that the CLI had installed it for them.
        let headline = if torch_alignment_disabled {
            "torch alignment is disabled by ROCM_CLI_DISABLE_TORCH_ALIGNMENT; the torch this runtime holds is the user's and a reinstall would replace it"
        } else {
            "the runtime holds the SDK's build of the torch vLLM pins; that divergence is intended and a reinstall would undo it"
        };
        let mut notes = vec![headline.to_owned()];
        notes.extend(
            intended
                .iter()
                .map(|violation| format!("expected divergence: {}", violation.detail)),
        );
        return RepairAssessment {
            needed: false,
            notes,
        };
    }

    let mut notes = vec![
        "the runtime environment did not satisfy vLLM's pinned dependencies; vLLM was reinstalled to restore them".to_owned(),
    ];
    // One note per violation rather than one joined line. The real failure is the whole
    // torch stack — torch, torchvision and torchaudio move together when the SDK writes
    // over the engine's pins — so joining them produced a single ~380-character line that
    // is unreadable in a terminal. Mirrors the per-finding `violation:` lines the
    // CLI-side renderer already emits.
    notes.extend(
        defects
            .iter()
            .map(|violation| format!("violation: {}", violation.detail)),
    );
    // An intended divergence alongside a real one is still worth naming, so the reader
    // is not left thinking the reinstall was about torch when it was not.
    notes.extend(
        intended
            .iter()
            .map(|violation| format!("expected divergence: {}", violation.detail)),
    );
    // The hint blames `rocm install sdk` for writing the SDK torch stack over vLLM's
    // pins, which stops being a live theory once the manifest names the SDK's build:
    // the alignment then identifies that stack and settles it, so a defect surviving
    // to here is something else. It stays on under the opt-out, which spares only the
    // package named `torch` — a `torchvision` or `torchaudio` defect is still the SDK
    // stack written over vLLM's pins, and the opt-out has turned off the step that
    // would have corrected it, so the hint is more use there rather than less.
    if sdk_torch_build.is_none() {
        notes.push(
            "if this recurs after `rocm install sdk`, the SDK torch stack is being written over vLLM's pinned torch".to_owned(),
        );
    }
    RepairAssessment {
        needed: true,
        notes,
    }
}

fn unverified_repair(reason: &str) -> RepairAssessment {
    RepairAssessment {
        needed: false,
        notes: vec![format!(
            "vLLM's pinned dependencies could not be verified in this environment: {reason}"
        )],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{
        RocmSdkRuntimeProbe, TheRockRuntimeManifest, sdk_torch_build_from_manifest,
    };

    fn violation(requiring: &str, detail: &str) -> DependencyViolation {
        DependencyViolation {
            requiring: requiring.to_owned(),
            detail: detail.to_owned(),
        }
    }
    /// The build identifier the SDK recorded in the runtimes used by these tests.
    const SDK_BUILD: &str = "rocm7.13.0";
    /// `repair_from_violations`'s opt-out argument, named at the call sites so a bare
    /// `false`/`true` does not have to be decoded against the signature.
    const ALIGNED: bool = false;
    const OPTED_OUT: bool = true;

    #[test]
    fn a_consistent_environment_is_not_reinstalled() {
        // The other settled state: the SDK published no build of the release vLLM
        // pins, so the runtime kept the engine's own build and the exact pin is
        // satisfied. `uv pip check` reports nothing at all, and the recorded SDK
        // build must not manufacture a finding out of that silence.
        assert_eq!(
            repair_from_violations(&[], Some(SDK_BUILD), ALIGNED),
            RepairAssessment::default()
        );
    }
    #[test]
    fn a_torch_of_the_wrong_release_still_forces_a_reinstall() {
        // The other direction of the same problem: the SDK installed a torch
        // *release* the engine does not accept. The build is the SDK's, but the
        // release is not the engine's, so this is not the intended divergence and
        // the reinstall that restores the engine's release must still happen.
        let assessment = repair_from_violations(
            &[violation(
                "vllm",
                "The package `vllm` requires `torch==2.10.0+git8514f05`, but `2.9.1+rocm7.14.0a20260611` is installed",
            )],
            Some("rocm7.14.0a20260611"),
            ALIGNED,
        );

        assert!(assessment.needed);
        assert!(
            assessment
                .notes
                .iter()
                .any(|note| note.contains("2.10.0+git8514f05")),
            "the reported note names the pin that was violated: {:?}",
            assessment.notes
        );
    }
    #[test]
    fn the_intended_torch_divergence_alone_does_not_force_a_reinstall() {
        // The steady state this change exists to stop churning. rocm-cli put the
        // SDK's build of the release vLLM pins back after the engine install; the
        // engine's exact pin cannot express that, so `uv pip check` reports it
        // forever. Reinstalling would replace it with the build that opens no
        // device, and the next invocation would do the whole thing again.
        let assessment = repair_from_violations(
            &[violation(
                "vllm",
                "The package `vllm` requires `torch==2.11.0+gitd0c8b1f`, but `2.11.0+rocm7.13.0` is installed",
            )],
            Some(SDK_BUILD),
            ALIGNED,
        );

        assert!(
            !assessment.needed,
            "the intended divergence must not trigger a reinstall: {:?}",
            assessment.notes
        );
        assert!(
            assessment
                .notes
                .iter()
                .all(|note| !note.contains("was reinstalled")),
            "no note may claim a repair that did not happen: {:?}",
            assessment.notes
        );
    }
    #[test]
    fn a_torch_from_neither_side_still_forces_a_reinstall() {
        // Same release the engine pins, but a build belonging to neither the SDK nor
        // the engine — someone installed a torch by hand, or a resolver picked one
        // off PyPI. Nothing about that is intended.
        let assessment = repair_from_violations(
            &[violation(
                "vllm",
                "The package `vllm` requires `torch==2.11.0+gitd0c8b1f`, but `2.11.0+cpu` is installed",
            )],
            Some(SDK_BUILD),
            ALIGNED,
        );

        assert!(assessment.needed);
    }
    #[test]
    fn an_unidentified_sdk_build_keeps_the_previous_behaviour() {
        // Without a recorded build there is no way to tell the intended divergence
        // from a defect, and guessing in the permissive direction would leave a
        // genuinely broken runtime alone. Fall back to repairing.
        let assessment = repair_from_violations(
            &[violation(
                "vllm",
                "The package `vllm` requires `torch==2.11.0+gitd0c8b1f`, but `2.11.0+rocm7.13.0` is installed",
            )],
            None,
            ALIGNED,
        );

        assert!(assessment.needed);
        assert!(
            assessment
                .notes
                .iter()
                .any(|note| note.contains("rocm install sdk")),
            "the SDK-overwrite hint belongs to exactly this un-identifiable case: {:?}",
            assessment.notes
        );
    }
    #[test]
    fn the_whole_replaced_torch_stack_is_reported_one_finding_per_line() {
        // What the failure looks like on hardware right after `rocm install sdk`:
        // the SDK moves torch, torchvision and torchaudio together, so all three
        // pins are violated at once. Joining them into a single note produced one
        // ~380-character line; each finding gets its own so a terminal can show it.
        //
        // Only torch is realigned, so only torch's divergence is intended. The other
        // two are genuine and still drive the reinstall — which is what restores all
        // three to the engine's builds before rocm-cli puts torch back.
        let assessment = repair_from_violations(
            &[
                violation(
                    "vllm",
                    "The package `vllm` requires `torch==2.11.0+gitd0c8b1f`, but `2.11.0+rocm7.13.0` is installed",
                ),
                violation(
                    "vllm",
                    "The package `vllm` requires `torchvision==0.24.1+d801a34`, but `0.26.0+rocm7.13.0` is installed",
                ),
                violation(
                    "vllm",
                    "The package `vllm` requires `torchaudio==2.9.0+eaa9e4e`, but `2.11.0+rocm7.13.0` is installed",
                ),
            ],
            Some(SDK_BUILD),
            ALIGNED,
        );

        assert!(assessment.needed);
        let violation_notes: Vec<&String> = assessment
            .notes
            .iter()
            .filter(|note| note.starts_with("violation: "))
            .collect();
        assert_eq!(
            violation_notes.len(),
            2,
            "every genuinely violated pin gets its own note: {:?}",
            assessment.notes
        );
        for package in ["torchvision==", "torchaudio=="] {
            assert!(
                violation_notes.iter().any(|note| note.contains(package)),
                "{package} is missing from the reported notes: {:?}",
                assessment.notes
            );
        }
        assert!(
            violation_notes
                .iter()
                .all(|note| !note.contains("torch==2.11.0")),
            "the realigned torch is a divergence, not a violation: {:?}",
            assessment.notes
        );
        assert!(
            assessment
                .notes
                .iter()
                .any(|note| note.starts_with("expected divergence: ")
                    && note.contains("torch==2.11.0+gitd0c8b1f")),
            "the intended divergence is still named, so the reader is not left \
             thinking the reinstall was about torch: {:?}",
            assessment.notes
        );
        assert!(
            assessment.notes.iter().all(|note| note.len() < 200),
            "no note should be a wall of joined findings: {:?}",
            assessment.notes
        );
    }
    #[test]
    fn an_sdk_built_torchvision_is_still_a_violation() {
        // The SDK writes the whole torch stack, so torchvision can carry the same
        // build as torch and — when the releases happen to line up — look exactly
        // like the intended divergence. rocm-cli realigns torch and nothing else, so
        // this is a real violation the engine must repair. The stack test above
        // cannot catch a regression here: its torchvision release differs too, so
        // the release check alone would still reject it.
        let assessment = repair_from_violations(
            &[violation(
                "vllm",
                "The package `vllm` requires `torchvision==0.24.1+d801a34`, but `0.24.1+rocm7.13.0` is installed",
            )],
            Some(SDK_BUILD),
            ALIGNED,
        );

        assert!(
            assessment.needed,
            "only torch is realigned; another package at the SDK's build is a genuine violation: {:?}",
            assessment.notes
        );
    }
    #[test]
    fn unrelated_upstream_conflicts_do_not_force_a_reinstall() {
        // These environments routinely carry conflicts between third-party packages.
        // Reinstalling vLLM would not resolve them, so they must not trigger one.
        let assessment = repair_from_violations(
            &[
                violation(
                    "tilelang",
                    "The package `tilelang` requires `cloudpickle>=3.0`, but `2.2.1` is installed",
                ),
                violation(
                    "torch",
                    "The package `torch` requires `sympy>=1.13`, but `1.12` is installed",
                ),
            ],
            Some(SDK_BUILD),
            ALIGNED,
        );

        assert_eq!(assessment, RepairAssessment::default());
    }
    #[test]
    fn an_opted_out_custom_torch_alone_does_not_force_a_reinstall() {
        // The runtime the opt-out exists to produce: the user set
        // ROCM_CLI_DISABLE_TORCH_ALIGNMENT, rocm-cli left their torch alone, and the
        // engine's exact pin is therefore unmet. The build belongs to neither the SDK
        // nor the engine — it is whatever the user chose — so the aligned-case rule
        // would call it a defect and reinstall vLLM, which installs the engine's torch
        // over the one the opt-out was set to keep. That is the CLI-side fight moved
        // into the engine, and it would make the opt-out worthless on any managed
        // runtime.
        let assessment = repair_from_violations(
            &[violation(
                "vllm",
                "The package `vllm` requires `torch==2.11.0+gitd0c8b1f`, but `2.9.1+cu128` is installed",
            )],
            Some(SDK_BUILD),
            OPTED_OUT,
        );

        assert!(
            !assessment.needed,
            "the opt-out must spare a hand-installed torch: {:?}",
            assessment.notes
        );
        assert!(
            assessment
                .notes
                .iter()
                .all(|note| !note.contains("was reinstalled")),
            "no note may claim a repair that did not happen: {:?}",
            assessment.notes
        );
        assert!(
            assessment
                .notes
                .iter()
                .any(|note| note.contains("ROCM_CLI_DISABLE_TORCH_ALIGNMENT")),
            "the reason given must be the opt-out, not a divergence rocm-cli produced: {:?}",
            assessment.notes
        );
        assert!(
            assessment
                .notes
                .iter()
                .all(|note| !note.contains("the runtime holds the SDK's build")),
            "rocm-cli did not install this torch and must not say it did: {:?}",
            assessment.notes
        );
    }
    #[test]
    fn an_opted_out_custom_torch_still_repairs_an_unrelated_defect() {
        // The opt-out is about torch, not about the environment. A vLLM-owned pin that
        // has nothing to do with torch is broken the same way it was before, and
        // reinstalling vLLM is still what fixes it. Returning early on the opt-out
        // would hide this defect behind a preference about a different package, and the
        // runtime would stay unable to serve with nothing said about why.
        let assessment = repair_from_violations(
            &[
                violation(
                    "vllm",
                    "The package `vllm` requires `torch==2.11.0+gitd0c8b1f`, but `2.9.1+cu128` is installed",
                ),
                violation(
                    "vllm",
                    "The package `vllm` requires `torchvision==0.24.1+d801a34`, but `0.20.0+cu128` is installed",
                ),
            ],
            Some(SDK_BUILD),
            OPTED_OUT,
        );

        assert!(
            assessment.needed,
            "an unrelated vLLM pin is still a defect under the opt-out: {:?}",
            assessment.notes
        );
        let violation_notes: Vec<&String> = assessment
            .notes
            .iter()
            .filter(|note| note.starts_with("violation: "))
            .collect();
        assert_eq!(
            violation_notes.len(),
            1,
            "only the unrelated pin is a violation: {:?}",
            assessment.notes
        );
        assert!(
            violation_notes[0].contains("torchvision=="),
            "the defect named must be the unrelated one: {:?}",
            assessment.notes
        );
        assert!(
            assessment
                .notes
                .iter()
                .any(|note| note.starts_with("expected divergence: ")
                    && note.contains("torch==2.11.0+gitd0c8b1f")),
            "the spared torch is still named, so the reader is not left thinking the \
             reinstall was about torch: {:?}",
            assessment.notes
        );
    }
    #[test]
    fn a_settled_runtime_converges_on_the_build_its_own_manifest_records() {
        // The convergence proof the tests above cannot give on their own. They hand
        // the classification a build literal, so a change to what
        // `sdk_torch_build_from_manifest` yields — `7.13.0` where the local segment
        // reads `rocm7.13.0`, say — would leave every one of them passing while the
        // real pipeline churned forever: the engine would call the realigned torch a
        // defect, reinstall its own build, rocm-cli would put the SDK's back, and the
        // next invocation would start over. Feeding the classification the value the
        // manifest actually produces is what ties the two halves together.
        //
        // The SDK's own torch release is deliberately not the one vLLM pins, because
        // that is the case realignment exists for: the release comes from the engine,
        // only the build comes from the SDK.
        let settled = violation(
            "vllm",
            "The package `vllm` requires `torch==2.11.0+gitd0c8b1f`, but `2.11.0+rocm7.13.0` is installed",
        );
        let recorded = TheRockRuntimeManifest {
            sdk_torch: Some("2.9.1+rocm7.13.0".to_owned()),
            ..TheRockRuntimeManifest::default()
        };
        // Written before `sdk_torch` was recorded. These runtimes are already on real
        // machines, so they have to settle too rather than churn forever.
        let reconstructed = TheRockRuntimeManifest {
            rocm_sdk: Some(RocmSdkRuntimeProbe {
                rocm_sdk_version: Some("7.13.0".to_owned()),
                ..RocmSdkRuntimeProbe::default()
            }),
            ..TheRockRuntimeManifest::default()
        };

        for manifest in [recorded, reconstructed] {
            let build = sdk_torch_build_from_manifest(&manifest)
                .expect("both manifest generations identify the SDK's build");
            let assessment =
                repair_from_violations(std::slice::from_ref(&settled), Some(&build), ALIGNED);

            assert!(
                !assessment.needed,
                "the state rocm-cli settles on must survive the engine's own check: {:?}",
                assessment.notes
            );
        }
    }
    #[test]
    fn an_unrunnable_check_reports_itself_without_forcing_a_reinstall() {
        let assessment = unverified_repair("uv binary is unavailable");

        assert!(!assessment.needed);
        assert_eq!(assessment.notes.len(), 1);
        assert!(
            assessment.notes[0].contains("could not be verified"),
            "{:?}",
            assessment.notes
        );
    }
}
