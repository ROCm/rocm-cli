// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! ROCm failure-mode diagnosis.
//!
//! Rust port of the `rocm-doctor` skill's `diagnose.py`. It matches an
//! [`Examination`] (plus optional user symptom text) against a **closed list**
//! of known misconfigurations and returns ranked [`Diagnosis`] results, each
//! with the evidence it used and a [`Fix`] (plan + verify step). When nothing
//! matches it routes the user upstream rather than guessing.
//!
//! The catalog is deliberately closed: new failure modes are added here, not
//! invented at runtime. Keyword tables, thresholds, and tracker URLs are the
//! data; the per-check logic mirrors `diagnose.py` field-for-field so the two
//! stay behaviorally identical.

use crate::examine::Examination;
use engine::{
    check_16_vllm_oom, check_17_torch_dlpack_cuda_variant, check_18_comgr_conflict,
    check_19_shm_too_small,
};
use host::{
    check_1_arch_not_in_wheel, check_2_hsa_override_unneeded, check_6_path_missing,
    check_8_wheel_rocm_mismatch, check_9_igpu_dgpu_collision, check_20_gpu_unsupported_by_rocm,
};
use linux::{
    check_3_rocm_kernel_unsupported, check_4_render_group, check_5_amdgpu_blacklisted,
    check_7_stale_repos, check_10_container_devices, check_11_iommu_hang,
    check_12_repo_native_broken,
};
use report::route_when_no_match;
use serde::{Deserialize, Serialize};
use windows::{
    check_13_hip_sdk_missing, check_14_adrenalin_too_old, check_15_msvc_redist,
    check_21_msvc_redist_outdated, check_22_display_driver_old,
};
use wsl::{
    check_wsl_1_gpu_not_exposed, check_wsl_2_dxcore_missing, check_wsl_3_rocdxg_missing,
    check_wsl_4_rocdxg_not_linked, check_wsl_5_distro_too_old, check_wsl_6_host_driver_too_old,
    check_wsl_7_wsl1,
};

mod engine;
mod host;
mod linux;
mod report;
mod scoring;
#[cfg(test)]
mod test_support;
mod windows;
mod wsl;

pub use engine::{
    VLLM_IMPORT_CANONICAL_SYMPTOM, VLLM_OOM_CANONICAL_SYMPTOM, vllm_import_symptom_is_diagnosable,
    vllm_oom_symptom_is_diagnosable,
};
pub use linux::dkms_driver_plan_is_supported;
pub use report::{render_report_text, upstream_route};

/// At/above this score a diagnosis is treated as a real match.
pub const MIN_SCORE_FOR_MATCH: i32 = 50;

/// At/above this score the agent may propose the fix immediately.
pub const HIGH_CONFIDENCE: i32 = 75;

/// A proposed remediation for a [`Diagnosis`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Fix {
    pub summary: String,
    pub commands: Vec<String>,
    pub needs_sudo: bool,
    pub needs_reboot: bool,
    pub needs_relogin: bool,
    pub fix_id: String,
    /// Whether `rocm fix <fix_id>`, run with no extra arguments, will change
    /// the machine.
    ///
    /// Kept alongside [`Fix::class`], which it is derived from, because it is
    /// the field published consumers already read. It is not redundant detail
    /// so much as a narrower question: `class` says *what* the CLI will do,
    /// this says only whether the machine is about to change.
    pub auto_applicable: bool,
    /// What the CLI will do with this entry on the examined machine.
    ///
    /// `#[serde(default)]` so a payload written before this field existed still
    /// deserializes; the default understates rather than overstates.
    #[serde(default)]
    pub class: crate::fix::FixClass,
    /// The argument this entry is waiting for, when `class` is
    /// [`crate::fix::FixClass::NeedsArgument`]. Read from the catalog beside
    /// `class` so the flags line renders identically from `rocm fix` and here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub needs: Option<String>,
    pub notes: Vec<String>,
    pub verify: String,
}

/// A single scored match against the failure-mode catalog.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Diagnosis {
    pub id: String,
    pub title: String,
    pub score: i32,
    pub evidence: Vec<String>,
    pub fix: Option<Fix>,
}

/// Where to send a report when no catalog entry matches.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Route {
    pub target: String,
    pub url: String,
}

/// The full diagnosis output (mirrors `diagnose.py --json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnoseReport {
    /// All nonzero-score diagnoses, highest score first.
    ///
    /// Every checker that fired at all lands here, including ones scoring below
    /// [`MIN_SCORE_FOR_MATCH`] — so a non-empty `matched` does NOT mean a cause
    /// was established. Read [`DiagnoseReport::has_match`] for that.
    pub matched: Vec<Diagnosis>,
    /// Whether any entry in `matched` cleared [`MIN_SCORE_FOR_MATCH`].
    ///
    /// Serialized because a JSON consumer cannot otherwise tell a real cause
    /// from a weak signal, and the obvious substitute — "is `matched` empty?" —
    /// is wrong. Several checkers open with a nonzero base score for a
    /// situation that is merely *potentially* relevant (being in a container,
    /// having an APU alongside a discrete GPU), so a perfectly healthy host
    /// produces a non-empty `matched` full of sub-threshold entries. A caller
    /// gating on emptiness proposes a fix for a machine with nothing wrong, and
    /// never routes the user to `route_when_no_match`.
    ///
    /// Computed at construction; [`DiagnoseReport::has_match`] recomputes from
    /// `matched` and stays the authority for Rust callers.
    #[serde(default)]
    pub has_match: bool,
    pub min_score_for_match: i32,
    pub high_confidence_threshold: i32,
    pub route_when_no_match: Route,
    /// Set when the catalog has no entries at all for the running platform, and
    /// carrying the routing note that says so.
    ///
    /// Decided purely by whether the catalog covers the running platform — never
    /// by a score. When this is `Some`, no checker ran, so `matched` is empty and
    /// `has_match` is false; a consumer must not wait for sub-threshold rows
    /// alongside it. Pinned by
    /// `an_out_of_scope_report_never_carries_matched_entries`.
    #[serde(default)]
    pub out_of_scope: Option<String>,
    /// The verdict on a model, when `--model` named one.
    ///
    /// Attached by the caller after the fact rather than produced by
    /// [`diagnose`]: answering it needs the host's GPU memory and the engine
    /// `serve` would select, neither of which an [`Examination`] carries. It
    /// rides on this report rather than replacing it because the environment
    /// answer stays true and useful either way — a blocked model on a host whose
    /// driver is also misconfigured is two findings, not one.
    #[serde(default)]
    pub model: Option<crate::model_readiness::ModelReadiness>,
}

/// Whether any diagnosis cleared [`MIN_SCORE_FOR_MATCH`].
///
/// One place the rule is written, so the serialized `has_match` field and the
/// [`DiagnoseReport::has_match`] accessor cannot answer differently.
fn any_cleared_threshold(matched: &[Diagnosis]) -> bool {
    matched.iter().any(|d| d.score >= MIN_SCORE_FOR_MATCH)
}

impl DiagnoseReport {
    /// Whether at least one diagnosis cleared [`MIN_SCORE_FOR_MATCH`].
    #[must_use]
    pub fn has_match(&self) -> bool {
        any_cleared_threshold(&self.matched)
    }
}

// ---------------------------------------------------------------------------
// Per-misconfiguration checkers (1:1 with diagnose.py)
//
// A new checker goes at the END of this section, next to where its `CHECKERS`
// entry was appended. Inserting one between an existing function and the doc
// comment above it silently rebinds that comment to the new item: Rust joins a
// run of `///` lines with no blank line between them, so the old rationale ends
// up documenting the new constant and the old function is left with none. That
// compiles, and clippy, rustfmt and every test pass.
// ---------------------------------------------------------------------------

/// A checker plus the OS families it applies to.
type Checker = (fn(&Examination, &str) -> Diagnosis, &'static [&'static str]);

// The `wsl` entries below run on less evidence than on bare metal: WSL collects
// no GPU topology (`gpus` is empty, because the probes that fill it read KFD and
// DRM), so the parts of those checks that compare a detected gfx target against a
// wheel's arch list cannot contribute. They still fire on the evidence WSL does
// have -- environment, ROCm install, symptom keywords -- which is strictly better
// than the previous behaviour of not running at all.
//
// The degradation is one-directional and must stay that way: less evidence means
// a check may miss a real fault, never that it invents one. Pinned by
// `the_shared_checks_under_report_on_wsl_rather_than_over_report`.
const CHECKERS: &[Checker] = &[
    (check_1_arch_not_in_wheel, &["linux", "windows", "wsl"]),
    (check_2_hsa_override_unneeded, &["linux", "windows", "wsl"]),
    (check_3_rocm_kernel_unsupported, &["linux"]),
    (check_4_render_group, &["linux"]),
    (check_5_amdgpu_blacklisted, &["linux"]),
    (check_6_path_missing, &["linux", "windows", "wsl"]),
    (check_7_stale_repos, &["linux"]),
    (check_8_wheel_rocm_mismatch, &["linux", "windows", "wsl"]),
    // Not "wsl": WSL2 exposes no per-device topology to collide over.
    (check_9_igpu_dgpu_collision, &["linux", "windows"]),
    (check_10_container_devices, &["linux"]),
    (check_11_iommu_hang, &["linux"]),
    (check_12_repo_native_broken, &["linux"]),
    (check_13_hip_sdk_missing, &["windows"]),
    (check_14_adrenalin_too_old, &["windows"]),
    (check_15_msvc_redist, &["windows"]),
    (check_16_vllm_oom, &["linux", "wsl"]),
    // Linux and WSL2: `libtorch_cuda.so` is an ELF name, and the vLLM engine is
    // gated off native Windows but runs on WSL2. The engine's startup-failure
    // hint fires on both, so the entry it points at must be reachable on both.
    // 17 rather than 16 because the vLLM out-of-memory entry above already
    // holds 16; the number is a stable handle, so the two do not get to share
    // one.
    (check_17_torch_dlpack_cuda_variant, &["linux", "wsl"]),
    (check_wsl_1_gpu_not_exposed, WSL_ONLY),
    (check_wsl_2_dxcore_missing, WSL_ONLY),
    (check_wsl_3_rocdxg_missing, WSL_ONLY),
    (check_wsl_4_rocdxg_not_linked, WSL_ONLY),
    (check_wsl_5_distro_too_old, WSL_ONLY),
    (check_wsl_6_host_driver_too_old, WSL_ONLY),
    (check_wsl_7_wsl1, WSL_ONLY),
    // Both families, opting in explicitly as the platform split requires. Which
    // copy of a library the loader picks is not a question about the amdgpu
    // module or the Windows host driver -- a wheel copy and a system copy
    // collide on WSL2 exactly as they do on bare metal. Windows is deferred
    // until this has proved the approach.
    (check_18_comgr_conflict, &["linux", "wsl"]),
    // Both families, for the same reason. The shortage has nothing to do with
    // `amdgpu` or `/dev/kfd` -- it is the size of a tmpfs -- and WSL2 ships the
    // same 64 MB default a container does, so leaving this tagged `linux` alone
    // would silence it on one of the two platforms most likely to have it.
    (check_19_shm_too_small, &["linux", "wsl"]),
    // Not "wsl": WSL2 reports no GPU topology (`gpus` is empty), so this check
    // has nothing to read there and would never fire.
    (check_20_gpu_unsupported_by_rocm, &["linux", "windows"]),
    // Windows only: the runtime is read from the Windows System32 folder. A WSL
    // distro is examined as a distro, so it has no such runtime to read.
    (check_21_msvc_redist_outdated, &["windows"]),
    // Windows only: the driver date comes from the Windows display adapter.
    // WSL2 reads the host driver through its own entries.
    (check_22_display_driver_old, &["windows"]),
];

const WSL_ONLY: &[&str] = &["wsl"];

/// The platform family a catalog entry is selected by.
///
/// This is deliberately NOT `Examination::os_family`. WSL2 reports an `os_family`
/// of `linux` and must keep doing so — `install`, `serve` and the engine crates
/// branch on it — but it is a different platform for diagnosis: it reaches the
/// GPU through `/dev/dxg` and the Windows host driver, with no `amdgpu` module
/// and no `/dev/kfd`.
///
/// Resolving the family here rather than in each check is what makes the split
/// safe. Every bare-metal entry is tagged `linux` only, so it stops applying on
/// WSL automatically; an entry that genuinely applies to both opts in by naming
/// `wsl` as well. That preserves the property the old wholesale WSL short-circuit
/// bought — no `fix-4-render-group` on a healthy WSL box — without also
/// suppressing the checks that were always valid there.
const fn platform_family(e: &Examination) -> &str {
    if e.is_wsl {
        return "wsl";
    }
    if e.os_family.is_empty() {
        return "linux";
    }
    e.os_family.as_str()
}

/// Run every applicable checker, drop zero-score results, sort by score
/// descending (stable, so ties keep catalog order).
fn run_all_checks(e: &Examination, symptom: &str) -> Vec<Diagnosis> {
    let family = platform_family(e);
    let mut results: Vec<Diagnosis> = CHECKERS
        .iter()
        .filter(|(_, applicable)| applicable.contains(&family))
        .map(|(check, _)| check(e, symptom))
        .filter(|d| d.score > 0)
        .collect();
    // Stable sort by score descending: ties keep catalog order.
    results.sort_by_key(|d| std::cmp::Reverse(d.score));
    results
}

/// Whether any catalog entry at all applies to this platform.
fn catalog_covers(e: &Examination) -> bool {
    let family = platform_family(e);
    CHECKERS
        .iter()
        .any(|(_, applicable)| applicable.contains(&family))
}

/// Diagnose an examination against the closed catalog.
#[must_use]
pub fn diagnose(e: &Examination, symptom: &str) -> DiagnoseReport {
    // WSL2 used to be short-circuited here as out of scope. It is a real platform
    // in the catalog now, with its own entries; what keeps the bare-metal checks
    // off it is `platform_family`, not a special case at this level.
    //
    // `fix-16-vllm-oom` and `fix-17-torch-dlpack` opt into `wsl` in CHECKERS for
    // that reason: they are keyword-only checks whose failure modes are just as
    // real under WSL2, so they still answer there alongside the WSL entries.
    //
    // `out_of_scope` still exists, for the platforms that genuinely have no
    // entries. That case used to fall through to an empty catalog and report "no
    // known misconfiguration", which reads as "your machine looks fine" when the
    // truth is that nothing was ever checked.
    let out_of_scope = uncovered_platform_message(e);
    let mut matched = if out_of_scope.is_some() {
        Vec::new()
    } else {
        run_all_checks(e, symptom)
    };
    take_applicability_from_the_catalog(&mut matched, platform_family(e));
    DiagnoseReport {
        has_match: any_cleared_threshold(&matched),
        matched,
        min_score_for_match: MIN_SCORE_FOR_MATCH,
        high_confidence_threshold: HIGH_CONFIDENCE,
        route_when_no_match: route_when_no_match(e),
        out_of_scope,
        model: None,
    }
}

/// Fill each finding's applicability from the catalog, for the operating system
/// of the machine that was examined.
///
/// One place, deliberately. Every checker used to state `auto_applicable` by
/// hand, which meant the catalog's answer existed twice in two modules with
/// nothing comparing them — and they had already drifted: `fix-9-igpu-dgpu` was
/// auto-applicable in the catalog and, in the Linux arm of its own checker,
/// not. Deriving it here makes that disagreement unrepresentable rather than
/// something a test has to go looking for.
///
/// Keyed on the examined machine's platform *family*, not the running host and
/// not `os_family`. WSL reports an `os_family` of `linux` but is its own family
/// for catalog purposes, so reading `os_family` here would look up every WSL
/// recipe under the wrong platform and silently report them all as print-only.
fn take_applicability_from_the_catalog(matched: &mut [Diagnosis], platform_family: &str) {
    for diagnosis in matched.iter_mut() {
        let Some(fix) = diagnosis.fix.as_mut() else {
            continue;
        };
        // An entry the catalog does not list for this OS keeps the default:
        // the CLI will not act on it here, which is exactly what the OS gate in
        // `fix::apply` would tell the user.
        let class = crate::fix::class_on(&fix.fix_id, platform_family).unwrap_or_default();
        fix.class = class;
        fix.auto_applicable = class.applies_itself();
        fix.needs = crate::fix::needs_on(&fix.fix_id, platform_family).map(ToOwned::to_owned);
    }
}

/// Say so when the catalog has no entries for the running platform.
///
/// Returning `None` here means the platform is covered, not that it is healthy.
fn uncovered_platform_message(e: &Examination) -> Option<String> {
    if catalog_covers(e) {
        return None;
    }
    Some(format!(
        "rocm diagnose covers Linux, Windows and WSL2. This host reports '{}', which the \
         catalog has no entries for, so nothing was checked -- this is not a clean bill of \
         health. Run `rocm examine --json` and report the platform upstream.",
        e.os_family
    ))
}

#[cfg(test)]
mod tests {
    use super::test_support::{igpu_dgpu_host, linux_base, windows_driver_exam, wsl_base};
    use super::windows::DISPLAY_DRIVER_OLD_SCORE;
    use super::*;
    use crate::examine::Gpu;

    /// `take_applicability_from_the_catalog` is the only place that may fill in
    /// `Fix::auto_applicable`. Every checker used to set it by hand instead --
    /// twenty of them were converted to leave it at `Fix::default()` and let the
    /// catalog fill it in, but the shared-memory and WSL checkers kept the
    /// inline `auto_applicable: false,` a while longer. It never produced wrong
    /// output, because the catalog pass overwrites whatever a checker sets --
    /// which is exactly the problem: a checker disagreeing with the catalog
    /// again (the way `fix-9-igpu-dgpu` once did) would be silently masked
    /// rather than caught. A behavioral test can't see this, since the output
    /// is correct either way; this reads the checkers' own source instead, so a
    /// hand-set boolean is caught before it can start drifting unnoticed.
    #[test]
    fn no_checker_hand_sets_auto_applicable() {
        // The checkers live in this file and in the modules under `diagnose/`.
        // Every file that holds one is listed here: a checker that moves into a
        // file missing from this list would leave the guard passing vacuously.
        let sources = [
            include_str!("diagnose.rs"),
            include_str!("diagnose/engine.rs"),
            include_str!("diagnose/host.rs"),
            include_str!("diagnose/linux.rs"),
            include_str!("diagnose/report.rs"),
            include_str!("diagnose/scoring.rs"),
            include_str!("diagnose/windows.rs"),
            include_str!("diagnose/wsl.rs"),
        ];
        let offenders: Vec<&str> = sources
            .into_iter()
            .flat_map(|source| source.lines())
            .filter(|line| {
                let trimmed = line.trim_start();
                trimmed.starts_with("auto_applicable: true")
                    || trimmed.starts_with("auto_applicable: false")
            })
            .collect();
        assert!(
            offenders.is_empty(),
            "a Fix literal is hand-setting `auto_applicable` again -- only \
             `take_applicability_from_the_catalog` may set this field; every checker must \
             leave it at `Fix::default()` and let the catalog fill it in, or the catalog and \
             the checker can silently drift the way fix-9-igpu-dgpu once did:\n{offenders:#?}"
        );
    }

    /// A pasted HIP-loading error still opens fix-13 when no AMD GPU was
    /// detected at all — deliberately, and this is the test that says so.
    ///
    /// `symptom_names_hip_loading` is the one of `needs_hip`'s three disjuncts
    /// not gated on `e.has_amd_gpu`, and the asymmetry is the point. The other
    /// two read a *probe* ("PyTorch here is a HIP build"), which is only
    /// meaningful beside a GPU. This one reads a *fact the user supplied*: a
    /// process on this machine tried to load HIP and failed. A GPU probe
    /// answering "nothing detected" is itself one of the things a broken HIP
    /// install causes, so letting it veto the user's own error would silence
    /// the entry on the hosts most likely to need it — a headless box whose
    /// driver did not load, or a WSL guest where the probe sees no card.
    /// Installing the HIP SDK is the right next step either way; if there
    /// genuinely is no AMD GPU, fix-13 ranks beside the other causes rather
    /// than suppressing them.
    #[test]
    fn fix_13_still_answers_a_pasted_hip_error_when_no_amd_gpu_was_detected() {
        let no_gpu = Examination {
            os_family: "windows".to_owned(),
            // The probe found nothing: no `has_amd_gpu`, no `gpus` entries.
            ..Examination::default()
        };
        let finding = check_13_hip_sdk_missing(&no_gpu, "amdhip64.dll was not found");
        assert_eq!(
            finding.score, 85,
            "the same score as the host with a detected GPU: the pasted error is the \
             evidence, and an absent-GPU probe does not weaken it: {finding:?}"
        );
        assert!(
            finding
                .evidence
                .iter()
                .any(|ev| ev.contains("No HIP SDK install found")),
            "and the SDK's absence is still named as the thing to put right: {finding:?}"
        );
        // Nothing here is an unsupported-GPU host, so the entry that outranks
        // fix-13 on the ticket's laptop must stay silent rather than firing on
        // a machine whose hardware was never identified.
        assert_eq!(
            check_20_gpu_unsupported_by_rocm(&no_gpu, "").score,
            0,
            "an undetected GPU is not a confirmed-unsupported one"
        );
    }

    /// The gate the rocm-doctor skill tells an agent to read, and the reason it
    /// is not "is `matched` empty?".
    ///
    /// This cannot be asserted from the e2e suite: `diagnose` scores several
    /// checkers from host state alone, with no symptom keyword involved, so on a
    /// runner that happens to have (say) `amdgpu` blacklisted the catalog
    /// explains the host no matter what symptom is passed. Constructing the
    /// `Examination` is the only way to hold the premise still.
    #[test]
    fn sub_threshold_causes_leave_has_match_false_and_route_upstream() {
        // Missing both groups scores 45 -- worth listing, not enough to
        // establish. `matched` is NOT empty here, which is the whole point: an
        // agent gating on emptiness would propose this fix for a host where
        // nothing was established, and never route the user anywhere.
        let mut e = linux_base();
        e.in_render_group = Some(false);
        e.in_video_group = Some(false);
        let report = diagnose(&e, "the office printer keeps jamming on page three");

        assert!(
            !report.matched.is_empty(),
            "this test is pointless unless something sub-threshold was listed"
        );
        assert!(
            report
                .matched
                .iter()
                .all(|d| d.score < report.min_score_for_match),
            "expected every cause below {}, got {:?}",
            report.min_score_for_match,
            report
                .matched
                .iter()
                .map(|d| (&d.id, d.score))
                .collect::<Vec<_>>()
        );
        assert!(
            !report.has_match,
            "nothing cleared the threshold, so has_match must be false -- \
             skills/rocm-doctor/ tells an agent to route upstream on exactly this"
        );
        assert!(
            report.route_when_no_match.url.starts_with("http"),
            "the skill's rule is to hand over this tracker when nothing was \
             established, so it must name somewhere to go: {:?}",
            report.route_when_no_match
        );
    }

    /// The other half: the flag has to discriminate, or asserting it is free.
    #[test]
    fn an_established_cause_sets_has_match() {
        let mut e = linux_base();
        e.in_render_group = Some(false);
        e.in_video_group = Some(false);
        let report = diagnose(&e, "cannot open /dev/kfd: permission denied");
        assert!(
            report.has_match,
            "a symptom that matches the catalog must set has_match: {:?}",
            report
                .matched
                .iter()
                .map(|d| (&d.id, d.score))
                .collect::<Vec<_>>()
        );
    }

    /// Every diagnosis must agree with the fix catalog about whether the CLI
    /// can apply the fix itself.
    ///
    /// An agent following the rocm-doctor skill branches on `auto_applicable`
    /// from `diagnose --json` to decide whether to offer to run `rocm fix` or
    /// merely print the plan, while `fix::apply` dispatches on `RECIPES`. The
    /// two are written in different files and nothing but this test holds them
    /// together, so a drift makes the CLI contradict itself: it would advertise
    /// a fix as one it can run and then refuse, or the reverse.
    ///
    /// Both OS families are exercised because the checkers build their `Fix`
    /// per-OS and only one branch is taken per run, so a Linux-only test would
    /// leave the Windows branch free to drift unobserved.
    #[test]
    fn every_diagnosis_agrees_with_the_fix_catalog_on_auto_applicability() {
        for os in ["linux", "windows"] {
            let mut e = Examination {
                os_family: os.to_owned(),
                ..Examination::default()
            };
            // Trip several checkers at once so this covers more of the catalog
            // than a single fix.
            e.in_render_group = Some(false);
            e.in_video_group = Some(false);
            e.has_apu = true;
            e.has_discrete_amd = true;
            e.gpus = vec![
                Gpu {
                    gfx_target: "gfx1103".to_owned(),
                    is_amd: true,
                    is_apu: Some(true),
                    ..Gpu::default()
                },
                Gpu {
                    gfx_target: "gfx1100".to_owned(),
                    is_amd: true,
                    is_apu: Some(false),
                    ..Gpu::default()
                },
            ];

            let report = diagnose(&e, "torch crashes with a segfault");
            assert!(
                !report.matched.is_empty(),
                "{os}: expected at least one diagnosis to compare against the catalog"
            );
            for d in &report.matched {
                let fix = d
                    .fix
                    .as_ref()
                    .unwrap_or_else(|| panic!("{os}/{}: matched with no fix", d.id));
                let catalog = crate::fix::class_on(&fix.fix_id, os)
                    .unwrap_or_else(|| {
                        panic!("{os}/{}: emitted a fix-id not in the catalog", fix.fix_id)
                    })
                    .applies_itself();
                assert_eq!(
                    fix.auto_applicable, catalog,
                    "{os}/{}: diagnose reports auto_applicable={}, but the fix catalog \
                     (what `rocm fix` actually does) says {catalog}",
                    fix.fix_id, fix.auto_applicable
                );
            }
        }
    }

    /// ROCMAI-543: every string `rocm diagnose` can put on a redirectable
    /// stream has to be plain ASCII, or a Windows console on the OEM code page
    /// (437) renders UTF-8 punctuation as mojibake like `ΓÇö`.
    ///
    /// Driven off `CHECKERS` rather than off a list of individual messages.
    /// The catalog *is* the registry of everything this command can say, so a
    /// check added later is covered the day it lands and no single sentence
    /// needs a pinned copy here -- a test that restates the constant it checks
    /// protects very little. Both surfaces are asserted because both are
    /// redirectable: the rendered text report, and the `--json` payload, which
    /// `serde_json` emits as raw UTF-8 rather than `\u` escapes.
    ///
    /// Two of this fix's conversions sit on branches a default `Examination`
    /// never reaches -- the dGPU/APU pairing in `check_9_igpu_dgpu_collision`
    /// and `check_16_vllm_oom`'s "does not fit" wording -- so the corpus lights
    /// them deliberately and the coverage guard at the end fails if it stops.
    #[test]
    fn every_message_the_diagnose_catalog_can_emit_is_ascii() {
        let corpus: Vec<(&str, Examination, &str)> = vec![
            ("linux, no symptom", linux_base(), ""),
            ("wsl, no symptom", wsl_base(), ""),
            (
                "linux apu+dgpu, crash",
                igpu_dgpu_host("linux"),
                "torch crashes with a segfault",
            ),
            (
                "windows apu+dgpu, crash",
                igpu_dgpu_host("windows"),
                "torch crashes with a segfault",
            ),
            ("linux vllm oom", linux_base(), VLLM_OOM_CANONICAL_SYMPTOM),
            ("wsl vllm oom", wsl_base(), VLLM_OOM_CANONICAL_SYMPTOM),
        ];

        let mut everything_seen = String::new();
        for (label, examination, symptom) in &corpus {
            // Every catalog entry, not only the ones that scored high enough to
            // be reported on this host: a message is still shipped text when the
            // host it describes is not the one under test.
            for (check, _) in CHECKERS {
                let diagnosis = check(examination, symptom);
                let json = serde_json::to_string(&diagnosis).expect("a Diagnosis serializes");
                assert!(
                    json.is_ascii(),
                    "{label}: diagnosis `{}` carries non-ASCII punctuation, which renders as \
                     mojibake on a Windows console using the OEM code page (ROCMAI-543): {json}",
                    diagnosis.id
                );
                everything_seen.push_str(&json);
            }

            let report = diagnose(examination, symptom);
            let text = render_report_text(&report, report.matched.len());
            assert!(
                text.is_ascii(),
                "{label}: `rocm diagnose` text output must be ASCII-only (ROCMAI-543):\n{text}"
            );
            let json = serde_json::to_string(&report).expect("a DiagnoseReport serializes");
            assert!(
                json.is_ascii(),
                "{label}: `rocm diagnose --json` output must be ASCII-only (ROCMAI-543): {json}"
            );
            everything_seen.push_str(&text);
        }

        // An invariant test that quietly stops reaching the branches it was
        // written for passes for the wrong reason. These markers sit beside the
        // converted punctuation without containing it, so they stay true
        // whichever way the dash goes and only fail if the corpus rots.
        assert!(
            everything_seen.contains("Discrete GPU(s):"),
            "the corpus no longer reaches check_9's dGPU/APU branch, so its wording is unguarded"
        );
        assert!(
            everything_seen.contains("reservation will not help"),
            "the corpus no longer reaches check_16's does-not-fit wording, so it is unguarded"
        );
    }

    #[test]
    fn wsl2_never_runs_the_bare_metal_catalog() {
        // The property the old wholesale WSL short-circuit bought, kept after
        // WSL became a real platform in the catalog. Every signal below WOULD
        // fire a bare-metal check; none of them mean anything on WSL2, where
        // there is no amdgpu module, no /dev/kfd and no render group.
        let mut e = wsl_base();
        e.in_render_group = Some(false);
        e.in_video_group = Some(false);
        e.amdgpu_loaded = Some(false);
        e.rocm_version = "6.4.1".to_owned();
        e.amdgpu_blacklisted_in = vec!["/etc/modprobe.d/blacklist.conf".to_owned()];
        e.rocm_repos_seen = vec![
            "repo.radeon.com/rocm/6.2".to_owned(),
            "repo.radeon.com/rocm/6.4".to_owned(),
        ];
        let report = diagnose(&e, "unable to open /dev/kfd permission denied");
        for bare_metal in [
            "fix-3-rocm-kernel",
            "fix-4-render-group",
            "fix-5-amdgpu-load",
            "fix-7-stale-repos",
            "fix-10-container",
            "fix-11-iommu",
            "fix-12-installer",
        ] {
            assert!(
                !report.matched.iter().any(|d| d.id == bare_metal),
                "{bare_metal} must not fire on WSL2"
            );
        }
    }

    #[test]
    fn wsl_checks_never_fire_on_bare_metal() {
        // The converse guard. A bare-metal host has no WSL facts at all, and the
        // WSL checks read those facts as false -- so without the family gate they
        // would report a missing /dev/dxg on every ordinary Linux box.
        let mut e = linux_base();
        e.in_render_group = Some(false);
        let report = diagnose(&e, "no ROCm-capable device is detected");
        assert!(
            !report.matched.iter().any(|d| d.id.starts_with("fix-wsl-")),
            "no WSL entry may fire on bare metal: {:?}",
            report.matched.iter().map(|d| &d.id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn the_shared_checks_under_report_on_wsl_rather_than_over_report() {
        // WSL does not collect GPU topology: `gpus` stays empty because the
        // probes that fill it read KFD and DRM, which do not exist there. The
        // cross-platform checks enabled on WSL read those fields, so they run on
        // less evidence here than on bare metal.
        //
        // That degradation has to be one-directional. Missing topology must mean
        // a check cannot reach its threshold on structure alone -- never that it
        // invents a fault. This pins the direction: on a healthy host with a
        // framework installed, no shared check may fire at all.
        let mut e = wsl_base();
        e.framework = "pytorch".to_owned();
        e.framework_version = "2.6.0".to_owned();
        e.framework_rocm_version = "6.4".to_owned();
        e.rocm_version = "6.4.1".to_owned();
        e.framework_arch_list = vec!["gfx1100".to_owned(), "gfx1151".to_owned()];
        e.rocm_path = "/opt/rocm".to_owned();
        e.env
            .insert("PATH".to_owned(), "/opt/rocm/bin:/usr/bin".to_owned());
        assert!(e.gpus.is_empty(), "WSL collects no GPU topology");

        let report = diagnose(&e, "");
        assert!(
            report.matched.is_empty(),
            "no shared check may fire without topology: {:?}",
            report
                .matched
                .iter()
                .map(|d| (&d.id, d.score))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_platform_with_no_catalog_entries_says_nothing_was_checked() {
        // Previously only WSL set this. An unsupported OS fell through to an
        // empty catalog and reported "no known misconfiguration", which reads as
        // a clean bill of health when in truth nothing ran.
        let mut e = linux_base();
        e.os_family = "other".to_owned();
        let report = diagnose(&e, "anything at all");
        let reason = report
            .out_of_scope
            .as_deref()
            .expect("an uncovered platform must say so");
        assert!(reason.contains("other"), "must name the platform: {reason}");
        assert!(
            reason.contains("not a clean bill of health"),
            "must not be mistaken for a pass: {reason}"
        );
        assert!(report.matched.is_empty());
        assert!(!report.has_match());
    }

    #[test]
    fn an_out_of_scope_report_never_carries_matched_entries() {
        // The serialized JSON contract on `out_of_scope`: when it is set the
        // catalog did not run, so `matched` cannot hold anything -- not even a
        // weak sub-threshold signal. One direction only: a covered platform with
        // nothing wrong also has an empty `matched`, with `out_of_scope` unset.
        // Nothing else in the tree pins the pair, which is how the field's own
        // doc comment came to claim the opposite. Two independent mechanisms
        // uphold it today -- `diagnose` clears `matched` when `out_of_scope` is
        // set, and `run_all_checks` filters on the same platform family
        // `catalog_covers` consults -- so this fails once both are gone, which
        // is exactly the "a checker somehow ran on an uncovered host" shape the
        // JSON contract must not develop.
        let mut uncovered = linux_base();
        uncovered.os_family = "other".to_owned();
        let mut windows = linux_base();
        windows.os_family = "windows".to_owned();
        for e in [linux_base(), wsl_base(), windows, uncovered] {
            // A symptom that scores for at least one entry where the catalog
            // applies, so "matched is empty" is a real consequence of the
            // platform being uncovered rather than of a silent symptom.
            let report = diagnose(&e, VLLM_OOM_CANONICAL_SYMPTOM);
            assert!(
                report.out_of_scope.is_none() || report.matched.is_empty(),
                "out_of_scope set with matched entries on {}: {:?}",
                e.os_family,
                report.matched.iter().map(|d| &d.id).collect::<Vec<_>>()
            );
            assert!(
                report.out_of_scope.is_none() || !report.has_match(),
                "out_of_scope set alongside has_match on {}",
                e.os_family
            );
        }
    }

    #[test]
    fn non_wsl_still_diagnoses_normally() {
        let mut e = linux_base();
        e.in_render_group = Some(false);
        let report = diagnose(&e, "");
        assert!(report.out_of_scope.is_none());
        assert_eq!(report.matched[0].id, "fix-4-render-group");
    }

    #[test]
    fn the_serialized_verdict_agrees_with_the_accessor() {
        // Rust callers read the method, tooling reads the field. They answer the
        // same question, so a host where they disagree would hand the two
        // audiences different verdicts.
        let mut in_container = linux_base();
        in_container.in_container = true;
        let mut render_group_missing = linux_base();
        render_group_missing.in_render_group = Some(false);
        render_group_missing.in_video_group = Some(false);

        // The fixtures above all land BELOW the threshold, so on their own they
        // would only ever exercise the `false` branch -- and a field that is
        // always false serializes correctly by accident. The last one carries a
        // symptom that pushes the same finding past HIGH_CONFIDENCE, so `true`
        // reaches the wire here too and not only on a bare-metal e2e lane.
        let mut real_fault = linux_base();
        real_fault.in_render_group = Some(false);

        let cases = [
            (linux_base(), ""),
            (in_container, ""),
            (render_group_missing, ""),
            (real_fault, "RuntimeError: unable to open /dev/kfd"),
        ];
        assert!(
            cases
                .iter()
                .any(|(e, symptom)| diagnose(e, symptom).has_match),
            "at least one fixture must clear the threshold, or this only ever \
             proves the false branch"
        );

        for (e, symptom) in cases {
            let report = diagnose(&e, symptom);
            assert_eq!(
                report.has_match,
                report.has_match(),
                "field and accessor disagree for {:?}",
                report.matched.iter().map(|d| &d.id).collect::<Vec<_>>()
            );
            let json: serde_json::Value =
                serde_json::from_str(&serde_json::to_string(&report).unwrap()).unwrap();
            assert_eq!(
                json.get("has_match").and_then(serde_json::Value::as_bool),
                Some(report.has_match),
                "the verdict must survive into the emitted document"
            );
        }
    }

    #[test]
    fn windows_only_checks_skipped_on_linux() {
        let e = linux_base();
        let report = diagnose(&e, "vcruntime140.dll is missing");
        // fix-15 is windows-only; must not appear on a linux exam.
        assert!(report.matched.iter().all(|d| d.id != "fix-15-msvc-redist"));
    }

    #[test]
    fn every_checker_platform_is_covered_by_its_recipe() {
        // `render_report_text` ends a matched diagnosis with `apply with: rocm
        // fix {id}`, and `fix::apply` then gates that id on the recipe's
        // `applies_on` against the *running* OS. So any platform family a
        // checker is registered for but its recipe omits is a platform where the
        // tool names a command and then refuses to run it -- which is exactly
        // what `fix-16-vllm-oom` did on WSL2, where the checker opts into `wsl`
        // and the recipe was LINUX_ONLY.
        //
        // Only one direction is an error. A recipe may legitimately apply more
        // widely than its checker answers (a user can reach `rocm fix <id>`
        // directly, without a diagnosis), so this asserts containment rather
        // than equality.
        //
        // No exemptions, including for the informational `info-20` entry: it is
        // a `DIAGNOSE-ONLY` member of the same catalog, so `rocm fix` answers
        // for it on both platforms its checker is registered for, and the
        // `apply with:` line diagnose prints for it is live like every other.
        for (check, families) in CHECKERS {
            let id = check(&Examination::default(), "").id;
            let applies_on = crate::fix::recipe_applies_on(&id).unwrap_or_else(|| {
                panic!(
                    "checker id `{id}` has no recipe in the fix catalog, so the \
                     `apply with: rocm fix {id}` line diagnose prints is dead"
                )
            });
            for family in *families {
                assert!(
                    applies_on.contains(family),
                    "`{id}` is diagnosed on `{family}` but its recipe only applies on \
                     {applies_on:?}; `rocm fix {id}` would print the plan and then exit 3 there"
                );
            }
        }
    }

    #[test]
    fn diagnosis_remediation_matches_the_fix_catalog_for_shared_fix_ids() {
        // fix-13, fix-14, fix-15, fix-16, fix-21, info-20 and fix-22 live in two places: the `rocm fix`
        // catalog (crate::fix::RECIPES) and the Fix these checks embed in a
        // Diagnosis. The prose *framing* legitimately differs by surface (a
        // catalog rationale vs a diagnosis summary), but the actionable
        // remediation must not silently diverge -- the verify command, the
        // notes, and the executable command lines have to agree. This is the
        // guard that they do; it caught a two-vs-three-space `verify` and a
        // reworded note.
        //
        // info-20 is the same dual-representation shape even though it has
        // nothing to run: its notes are the whole of its remedy (that
        // HSA_OVERRIDE_GFX_VERSION is not a workaround, and that `rocm serve`
        // already picks Vulkan), so the two copies drifting would leave one
        // surface giving advice the other contradicts. Its empty `commands` and
        // `verify` have to stay agreed on too, since a command appearing on one
        // side only is how a DIAGNOSE-ONLY entry silently stops being one.
        let vllm_oom = check_16_vllm_oom(
            &Examination {
                os_family: "linux".to_owned(),
                ..Examination::default()
            },
            VLLM_OOM_CANONICAL_SYMPTOM,
        );
        let msvc = check_15_msvc_redist(
            &Examination {
                os_family: "windows".to_owned(),
                msvc_redist_present: Some(false),
                ..Examination::default()
            },
            "ImportError: DLL load failed: vcruntime140_1.dll not found",
        );
        let msvc_outdated = check_21_msvc_redist_outdated(
            &Examination {
                os_family: "windows".to_owned(),
                msvc_redist_version: "14.29.30133.0".to_owned(),
                ..Examination::default()
            },
            "",
        );
        let unsupported_gpu = check_20_gpu_unsupported_by_rocm(
            &Examination {
                gpus: vec![Gpu {
                    gfx_target: "gfx90c".to_owned(),
                    is_amd: true,
                    ..Gpu::default()
                }],
                ..Examination::default()
            },
            "",
        );
        // fix-13 and fix-14 print a Windows verify that the Windows-only tests
        // in `fix.rs` run from the catalog copy, so this copy has to match it.
        let hip_sdk_missing = check_13_hip_sdk_missing(
            &Examination {
                os_family: "windows".to_owned(),
                ..Examination::default()
            },
            "ImportError: DLL load failed: amdhip64_6.dll was not found",
        );
        let adrenalin_old = check_14_adrenalin_too_old(
            &Examination {
                os_family: "windows".to_owned(),
                hip_sdk_path: r"C:\Program Files\AMD\ROCm\6.4".to_owned(),
                hipinfo_present: true,
                hipinfo_status: "error".to_owned(),
                ..Examination::default()
            },
            "driver too old",
        );
        // fix-22 is advice only: it scores below the match threshold on purpose,
        // yet a user meets both of its copies, so it is held to the same guard.
        let display_driver_old =
            check_22_display_driver_old(&windows_driver_exam("30.0.13014.8", "2021-08-04"), "");
        for diag in [
            vllm_oom,
            msvc,
            msvc_outdated,
            unsupported_gpu,
            hip_sdk_missing,
            adrenalin_old,
            display_driver_old,
        ] {
            if diag.id == crate::windows_driver::DISPLAY_DRIVER_OLD_FIX_ID {
                assert_eq!(diag.score, DISPLAY_DRIVER_OLD_SCORE);
            } else {
                assert!(
                    diag.score >= MIN_SCORE_FOR_MATCH,
                    "test setup must yield a real match for {}",
                    diag.id
                );
            }
            let fix = diag.fix.expect("a matched diagnosis must carry a Fix");
            let (verify, notes, run_commands) =
                crate::fix::recipe_verify_notes_and_run_commands(&fix.fix_id)
                    .expect("a shared fix_id must exist in the recipe catalog");
            assert_eq!(fix.verify, verify, "verify diverged for {}", fix.fix_id);
            assert_eq!(
                fix.notes,
                notes.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>(),
                "notes diverged for {}",
                fix.fix_id
            );
            let emitted_run = fix
                .commands
                .iter()
                .map(String::as_str)
                .filter(|line| !line.trim_start().starts_with('#'))
                .collect::<Vec<_>>();
            assert_eq!(
                emitted_run, run_commands,
                "executable command lines diverged for {}",
                fix.fix_id
            );
        }
    }

    /// The `Fix` that each checker builds for itself is shown to users of
    /// Windows PowerShell 5.1 as well as cmd, just like the `rocm fix` catalog.
    /// This runs every checker that answers on Windows and applies the same
    /// rule as the catalog test in `fix.rs` (`fails_in_windows_powershell`), so a
    /// hand-built verify string or command that breaks one of the rule's known
    /// constructs fails here. A construct the rule does not list passes.
    ///
    /// Like that test it pins the text only. Nothing here runs a command in a
    /// shell. The Windows-only tests named on `fails_in_windows_powershell` in
    /// `fix.rs` run the fix-2, fix-13, fix-14, fix-15, fix-21 and fix-22 lines,
    /// and `the_igpu_verify_pins_the_variable_it_names` runs fix-9's, all on
    /// Windows.
    #[test]
    fn windows_diagnosis_remediation_passes_the_powershell_text_rule() {
        // Host state that wakes the checkers which answer from the machine and
        // not from the symptom: an override set, a HIP SDK with no `hipInfo`,
        // an APU beside a discrete GPU, and a runtime that is missing.
        let e = Examination {
            os_family: "windows".to_owned(),
            msvc_redist_present: Some(false),
            hip_sdk_path: r"C:\Program Files\AMD\ROCm\6.4".to_owned(),
            hipinfo_present: false,
            has_apu: true,
            has_discrete_amd: true,
            env: [
                ("HSA_OVERRIDE_GFX_VERSION".to_owned(), "11.0.0".to_owned()),
                ("PATH".to_owned(), r"C:\Windows".to_owned()),
            ]
            .into_iter()
            .collect(),
            ..Examination::default()
        };
        // Wide enough to wake the keyword-driven checkers, so the test sees
        // the Fix they build rather than an empty answer. fix-13 only scores
        // when the paste names HIP failing to load, hence `was not found`.
        let symptom = "hipErrorNoBinaryForGpu HSA_STATUS_ERROR_INVALID_ISA invalid device function \
                       HSA_OVERRIDE_GFX_VERSION vcruntime140_1.dll amdhip64_6.dll was not found \
                       hipblas.dll driver too old";
        let mut seen: Vec<String> = Vec::new();
        for (check, families) in CHECKERS {
            if !families.contains(&"windows") {
                continue;
            }
            let Some(fix) = check(&e, symptom).fix else {
                continue;
            };
            let printed = fix
                .commands
                .iter()
                .map(String::as_str)
                .filter(|c| !c.trim_start().starts_with('#'))
                .chain(std::iter::once(fix.verify.as_str()));
            for line in printed {
                if let Some(why) = crate::fix::fails_in_windows_powershell(line) {
                    panic!("{}: `{line}` uses {why}", fix.fix_id);
                }
            }
            seen.push(fix.fix_id);
        }
        // Guard the test itself: it must have looked at the Windows checkers.
        for expected in [
            "fix-2-unset-override",
            "fix-6-path",
            "fix-9-igpu-dgpu",
            "fix-13-hip-sdk-missing",
            "fix-14-adrenalin-too-old",
            "fix-15-msvc-redist",
        ] {
            assert!(
                seen.iter().any(|id| id == expected),
                "the sweep never saw {expected}; it saw {seen:?}"
            );
        }
    }

    #[test]
    fn report_serializes_expected_shape() {
        let report = diagnose(&linux_base(), "");
        let v = serde_json::to_value(&report).unwrap();
        for key in [
            "matched",
            "has_match",
            "min_score_for_match",
            "high_confidence_threshold",
            "route_when_no_match",
        ] {
            assert!(v.get(key).is_some(), "missing {key}");
        }
        assert_eq!(v["min_score_for_match"], 50);
        assert_eq!(v["high_confidence_threshold"], 75);
    }
}
