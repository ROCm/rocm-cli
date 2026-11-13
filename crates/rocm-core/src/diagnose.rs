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
//!
//! This file keeps the public types, the two score thresholds, [`diagnose`]
//! itself and the two `vllm_*_symptom_is_diagnosable` classifiers that run the
//! whole catalog. The rest is in child modules: `catalog` holds the `CHECKERS` table,
//! `scoring` the keyword tables and scoring helpers, `report` the routing and the
//! text report, and `linux`, `windows`, `wsl`, `host` and `engine` the checks.

use crate::examine::Examination;
use catalog::{
    platform_family, run_all_checks, take_applicability_from_the_catalog,
    uncovered_platform_message,
};
use engine::scores_on_every_engine_host;
use report::route_when_no_match;
use serde::{Deserialize, Serialize};

mod catalog;
mod engine;
mod host;
mod linux;
mod report;
mod scoring;
#[cfg(test)]
mod test_support;
mod windows;
mod wsl;

pub use engine::{VLLM_IMPORT_CANONICAL_SYMPTOM, VLLM_OOM_CANONICAL_SYMPTOM};
pub use linux::{AMDGPU_MISSING_EVIDENCE, dkms_driver_plan_is_supported};
pub use report::{SYMPTOM_DID_NOT_CONTRIBUTE_NOTE, render_report_text, upstream_route};

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
    /// Whether the symptom text the user supplied contributed to this entry.
    ///
    /// `true` when, without the symptom, this entry would have a different score
    /// or different evidence, or would not be listed at all; `false` when it
    /// comes from the examined host's state alone. A host-state finding can
    /// outrank the entry the symptom added to, so a reader asking "did the text
    /// I pasted contribute to this entry?" reads this rather than the ranking.
    ///
    /// It records a contribution, not a verdict that the entry explains the
    /// error: any keyword hit counts, however weak. Read `score` and `evidence`
    /// for how much the text added.
    ///
    /// Always serialized, and `false` when no symptom was given: nothing was
    /// pasted, so nothing in it contributed. Omitting it in that case would make
    /// "no symptom" indistinguishable from "a CLI too old to report this field".
    /// `#[serde(default)]` so a payload written before the field existed still
    /// deserializes. Set by [`diagnose`], never by a checker.
    #[serde(default)]
    pub symptom_matched: bool,
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

/// Whether `symptom` clears [`MIN_SCORE_FOR_MATCH`] for the vLLM-OOM failure
/// mode.
///
/// This is the OOM *classifier* the whole stack scores against. It is not what
/// the engine calls directly any more: `rocm_core::vllm_oom_diagnose_symptom`
/// owns line selection for both surfaces and reaches this per rendered row, and
/// what those surfaces then branch on before printing is
/// `quotable_in_single_quotes` — a line this function accepts is still replaced
/// by [`VLLM_OOM_CANONICAL_SYMPTOM`] when it cannot be rendered as one intact
/// single-quoted argument. Scoring here and admissibility there are deliberately
/// separate questions; keeping them straight is what stops a colourised line
/// being printed as a runnable lookalike of the user's error.
#[must_use]
pub fn vllm_oom_symptom_is_diagnosable(symptom: &str) -> bool {
    scores_on_every_engine_host(symptom, "fix-16-vllm-oom", run_all_checks)
}

/// Whether `symptom` clears [`MIN_SCORE_FOR_MATCH`] for the engine-startup
/// import failure.
///
/// The vLLM engine calls this to decide whether the user's actual failing line
/// is worth routing into a `--symptom` example, or whether to fall back to
/// [`VLLM_IMPORT_CANONICAL_SYMPTOM`] so the printed command is guaranteed to
/// report a cause rather than reading as "the tool checked and found nothing".
///
/// The hint fires on every host the vLLM engine runs on, which is Linux and
/// WSL2. The symptom therefore has to clear the threshold on each of those
/// families, run through the same registry `rocm diagnose` uses, so a check
/// that is registered for one family and not the other cannot make the engine
/// name a command that finds nothing.
#[must_use]
pub fn vllm_import_symptom_is_diagnosable(symptom: &str) -> bool {
    scores_on_every_engine_host(symptom, "fix-17-torch-dlpack", run_all_checks)
}

/// Set [`Diagnosis::symptom_matched`] on every entry in `matched`.
///
/// Decided by running the same catalog again on the same machine with no
/// symptom and comparing, rather than by each checker reporting what it read.
/// Checkers fold the symptom in several ways (keyword tables, single patterns,
/// a keyword whose own weight is replaced by a fixed one, a symptom that lifts a
/// negative host score above zero), and a flag each of them had to set by hand
/// is one a new checker can forget. The comparison asks the question itself:
/// would this entry look the same without the text? Evidence is compared as
/// well as the score because a score is capped at 100, so a keyword hit on an
/// entry already at the cap changes its evidence and not its number.
///
/// The checkers are pure functions of the examination and the symptom, so the
/// second run reads no more of the machine than the first.
fn mark_symptom_contribution(matched: &mut [Diagnosis], e: &Examination, symptom: &str) {
    if symptom.is_empty() {
        return;
    }
    let host_only = run_all_checks(e, "");
    for d in matched {
        d.symptom_matched = contributed(d, &host_only);
    }
}

/// Whether the symptom contributed to `d`, given `host_only`, the same catalog
/// run on the same host with no symptom.
///
/// It did when `d` is absent from that run, or present with a different score
/// or different evidence. Either difference alone is enough: a keyword hit on
/// an entry already at the 100 cap changes only its evidence.
fn contributed(d: &Diagnosis, host_only: &[Diagnosis]) -> bool {
    host_only
        .iter()
        .find(|h| h.id == d.id)
        .is_none_or(|h| h.score != d.score || h.evidence != d.evidence)
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
    mark_symptom_contribution(&mut matched, e, symptom);
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

#[cfg(test)]
mod tests {
    use super::test_support::linux_base;
    use super::*;

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
        // The directory is walked recursively rather than listed, so a new child
        // module, nested or not, is scanned without anyone remembering to add it
        // here. Only the test-only `test_support.rs` is skipped: it builds
        // `Examination` fixtures and no `Fix`.
        fn collect(dir: &std::path::Path, out: &mut Vec<String>) {
            for entry in std::fs::read_dir(dir).expect("a readable directory") {
                let path = entry.expect("a readable directory entry").path();
                if path.is_dir() {
                    collect(&path, out);
                    continue;
                }
                let is_rust = path.extension().is_some_and(|ext| ext == "rs");
                let is_test_support = path
                    .file_name()
                    .is_some_and(|name| name == "test_support.rs");
                if is_rust && !is_test_support {
                    out.push(std::fs::read_to_string(&path).expect("a readable source file"));
                }
            }
        }
        let mut sources = vec![include_str!("diagnose.rs").to_owned()];
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/diagnose");
        let mut children = Vec::new();
        collect(std::path::Path::new(dir), &mut children);
        let scanned_children = children.len();
        sources.extend(children);
        assert!(
            scanned_children > 0,
            "no module under {dir} was scanned, so this guard would pass vacuously"
        );
        let offenders: Vec<&str> = sources
            .iter()
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

    #[test]
    fn canonical_vllm_import_symptom_is_diagnosable() {
        // The engine's fallback symptom must always clear the threshold, so the
        // `rocm diagnose` command it prints never reports nothing.
        assert!(vllm_import_symptom_is_diagnosable(
            VLLM_IMPORT_CANONICAL_SYMPTOM
        ));
        // A frame that names only the shim's module, and not the missing library,
        // is not enough on its own, and the engine has to know that.
        assert!(!vllm_import_symptom_is_diagnosable(
            "File \"tvm_ffi/_optional_torch_c_dlpack.py\", line 42, in <module>"
        ));
        assert!(!vllm_import_symptom_is_diagnosable("an ordinary failure"));
    }

    #[test]
    fn the_engine_host_guard_requires_the_entry_to_score_on_every_host_family() {
        // A registry stand-in: `listed_on` says which hosts list the entry.
        fn registry(
            id: &'static str,
            listed_on_wsl: bool,
            listed_on_linux: bool,
        ) -> impl Fn(&Examination, &str) -> Vec<Diagnosis> {
            move |host, _symptom| {
                let listed = if host.is_wsl {
                    listed_on_wsl
                } else {
                    listed_on_linux
                };
                if !listed {
                    return Vec::new();
                }
                vec![Diagnosis {
                    id: id.to_owned(),
                    score: 100,
                    ..Diagnosis::default()
                }]
            }
        }
        let guard = |run| scores_on_every_engine_host("x", "fix-17-torch-dlpack", run);
        assert!(guard(registry("fix-17-torch-dlpack", true, true)));
        assert!(
            !guard(registry("fix-17-torch-dlpack", false, true)),
            "an entry missing on WSL must not count"
        );
        assert!(
            !guard(registry("fix-17-torch-dlpack", true, false)),
            "an entry missing on Linux must not count"
        );
        assert!(
            !guard(registry("fix-16-vllm-oom", true, true)),
            "a different entry scoring is not this entry scoring"
        );
        // And the real registry lists both entries on both families.
        assert!(vllm_import_symptom_is_diagnosable(
            VLLM_IMPORT_CANONICAL_SYMPTOM
        ));
        assert!(vllm_oom_symptom_is_diagnosable(VLLM_OOM_CANONICAL_SYMPTOM));
        // An ISA error scores on another check, not on this one.
        assert!(!vllm_import_symptom_is_diagnosable(
            "hipErrorNoBinaryForGpu: Unable to find code object for all current devices"
        ));
    }

    #[test]
    fn canonical_vllm_oom_symptom_is_diagnosable() {
        // The engine's fallback symptom must always clear the threshold, so the
        // `rocm diagnose` command it prints never reports nothing.
        assert!(vllm_oom_symptom_is_diagnosable(VLLM_OOM_CANONICAL_SYMPTOM));
        // And a bare framework mention the engine would reject is not diagnosable.
        assert!(!vllm_oom_symptom_is_diagnosable("vllm: out of memory"));
    }

    /// The entry with `id` in `report`, or a panic naming what was listed.
    fn entry<'a>(report: &'a DiagnoseReport, id: &str) -> &'a Diagnosis {
        report
            .matched
            .iter()
            .find(|d| d.id == id)
            .unwrap_or_else(|| {
                panic!(
                    "{id} was not listed: {:?}",
                    report
                        .matched
                        .iter()
                        .map(|d| (&d.id, d.score))
                        .collect::<Vec<_>>()
                )
            })
    }

    fn with_gpu_override() -> Examination {
        let mut e = linux_base();
        e.env
            .insert("HSA_OVERRIDE_GFX_VERSION".to_owned(), "11.0.0".to_owned());
        e
    }

    const ISA_PASTE: &str = "hipErrorNoBinaryForGpu: HSA_STATUS_ERROR_INVALID_ISA";

    /// The case the field exists for: a finding about the machine sits beside
    /// the entry the paste scores, and the paste contributed to only one of them.
    #[test]
    fn a_host_state_finding_is_not_credited_to_a_paste_about_something_else() {
        let report = diagnose(&with_gpu_override(), ISA_PASTE);

        let host = entry(&report, "fix-2-unset-override");
        assert!(
            !host.symptom_matched,
            "the override finding comes from the environment alone: {host:?}"
        );
        let pasted = entry(&report, "fix-1-arch");
        assert!(
            pasted.symptom_matched,
            "the ISA entry is scored from the paste: {pasted:?}"
        );

        // And both answers reach the document a consumer reads.
        let json = serde_json::to_value(&report).unwrap();
        let field = |id: &str| {
            json["matched"]
                .as_array()
                .unwrap()
                .iter()
                .find(|d| d["id"] == id)
                .map(|d| d["symptom_matched"].clone())
        };
        assert_eq!(field("fix-2-unset-override"), Some(false.into()));
        assert_eq!(field("fix-1-arch"), Some(true.into()));
    }

    #[test]
    fn a_paste_that_raises_a_host_finding_is_credited() {
        // The override alone scores 30; a page fault in the paste adds to it.
        let report = diagnose(&with_gpu_override(), "memory access page fault");
        let raised = entry(&report, "fix-2-unset-override");
        assert!(
            raised.score > 30,
            "the paste must have raised it: {raised:?}"
        );
        assert!(raised.symptom_matched, "{raised:?}");
    }

    #[test]
    fn a_paste_that_lifts_an_entry_the_host_alone_ruled_out_is_credited() {
        // The framework already carries this GPU's target, which pulls the
        // arch entry below zero on its own, so it is not listed without the
        // paste at all. With it, the entry is listed, and only because of it.
        let mut e = linux_base();
        e.gpus = vec![crate::examine::Gpu {
            gfx_target: "gfx1100".to_owned(),
            is_amd: true,
            ..crate::examine::Gpu::default()
        }];
        e.framework_arch_list = vec!["gfx1100".to_owned()];
        assert!(
            diagnose(&e, "")
                .matched
                .iter()
                .all(|d| d.id != "fix-1-arch"),
            "the premise: without the paste this entry is not listed"
        );
        let report = diagnose(&e, ISA_PASTE);
        assert!(entry(&report, "fix-1-arch").symptom_matched);
    }

    #[test]
    fn with_no_symptom_every_entry_says_so_and_the_field_is_still_written() {
        for symptom in ["", "   "] {
            let report = diagnose(&with_gpu_override(), symptom);
            assert!(
                !report.matched.is_empty(),
                "this test is pointless unless something was listed"
            );
            assert!(
                report.matched.iter().all(|d| !d.symptom_matched),
                "nothing was pasted for {symptom:?}, so nothing in it contributed: {:?}",
                report.matched
            );
            let json = serde_json::to_value(&report).unwrap();
            for d in json["matched"].as_array().unwrap() {
                assert_eq!(
                    d.get("symptom_matched"),
                    Some(&serde_json::Value::Bool(false)),
                    "the field is written, not omitted, when no symptom was given: {d}"
                );
            }
        }
    }

    #[test]
    fn a_payload_written_before_the_field_existed_still_reads() {
        let old =
            r#"{"id":"fix-2-unset-override","title":"t","score":30,"evidence":[],"fix":null}"#;
        let d: Diagnosis = serde_json::from_str(old).expect("an older payload deserializes");
        assert!(!d.symptom_matched);
    }

    fn finding(id: &str, score: i32, evidence: &[&str]) -> Diagnosis {
        Diagnosis {
            id: id.to_owned(),
            score,
            evidence: evidence.iter().map(|&e| e.to_owned()).collect(),
            ..Diagnosis::default()
        }
    }

    /// The three ways the symptom can show in an entry, one at a time, so each
    /// half of the comparison has a case that only it decides.
    #[test]
    fn contribution_is_any_difference_from_the_run_without_the_symptom() {
        let host_only = [finding("fix-a", 30, &["host fact"])];

        // Same entry, same evidence, higher score: the score alone differs.
        assert!(contributed(
            &finding("fix-a", 55, &["host fact"]),
            &host_only
        ));
        // Same score, extra evidence: what a keyword hit does to an entry
        // already at the 100 cap. The evidence alone differs.
        assert!(contributed(
            &finding("fix-a", 30, &["host fact", "error mentions X"]),
            &host_only
        ));
        // Not listed at all without the symptom.
        assert!(contributed(
            &finding("fix-b", 50, &["error mentions Y"]),
            &host_only
        ));
        // Identical to the run without it: nothing contributed.
        assert!(!contributed(
            &finding("fix-a", 30, &["host fact"]),
            &host_only
        ));
    }
}
