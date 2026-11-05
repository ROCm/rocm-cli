// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Routing and the text report.
//!
//! [`upstream_route`] is the one place a [`Route`] is built, so every command that
//! has to say "I don't recognise this" sends the user to the same tracker for the
//! same target. [`render_report_text`] renders a [`DiagnoseReport`] for people.

use super::{DiagnoseReport, HIGH_CONFIDENCE, MIN_SCORE_FOR_MATCH, Route};
use crate::examine::Examination;

/// Upstream tracker for a routing target.
///
/// Only the targets [`route_when_no_match`] can actually produce are listed.
/// Arms for lemonade / ollama / lm-studio / amdgpu-install were unreachable —
/// the host probe never reports those frameworks — so they described a
/// capability the CLI does not have. Routing an app that merely appears in the
/// symptom text is the caller's job, not the probe's.
fn upstream_tracker(target: &str) -> &'static str {
    match target {
        "pytorch" => "https://github.com/pytorch/pytorch/issues  (tag with rocm label)",
        "llama-cpp" => "https://github.com/ggml-org/llama.cpp/issues",
        _ => "https://github.com/ROCm/ROCm/issues",
    }
}

/// Where to report something this CLI could not answer.
///
/// The one place a `Route` is built, so every command that has to say "I don't
/// recognise this" sends the user to the same tracker for the same target —
/// `rocm diagnose --model` has the same problem for a model the catalog does not
/// carry as `rocm diagnose` has for a symptom it does not recognise.
#[must_use]
pub fn upstream_route(target: &str) -> Route {
    Route {
        target: target.to_owned(),
        url: upstream_tracker(target).to_owned(),
    }
}

/// Where to send a user when nothing in the catalog matched.
///
/// Keyed off the *host-detected* framework, which `Examination::probe` only
/// ever sets to `pytorch`, `llama-cpp`, `unknown` or `skipped` — so those two
/// named arms plus the ROCm-core default are the whole reachable set.
///
/// Adding a framework to `examine/framework.rs`'s probe means adding its arm here, its
/// tracker in [`upstream_tracker`], and its name to the hand-maintained list in
/// `routing_targets_cover_every_framework_the_probe_reports`. That test reads
/// its own list rather than deriving one from `examine/framework.rs`, so it cannot notice
/// a new framework on its own — all three edits are manual.
pub(super) fn route_when_no_match(e: &Examination) -> Route {
    let target = match e.framework.as_str() {
        "pytorch" => "pytorch",
        "llama-cpp" => "llama-cpp",
        _ => "rocm-core",
    };
    upstream_route(target)
}

/// Whether a catalog entry is advice about the host rather than a candidate
/// cause of the reported symptom.
///
/// Such an entry reads only evidence, never the symptom, so it fires whatever
/// the user reported. The text report must not call it "the highest-scoring
/// match" nor let it stand in for the upstream escalation.
fn is_advice_only(fix_id: &str) -> bool {
    fix_id == crate::windows_driver::DISPLAY_DRIVER_OLD_FIX_ID
}

/// Render the human-facing diagnosis view (mirrors `diagnose.py`'s text output).
#[must_use]
pub fn render_report_text(report: &DiagnoseReport, top: usize) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    if let Some(reason) = &report.out_of_scope {
        out.push_str("rocm diagnose: out of scope for this platform.\n\n");
        out.push_str(reason);
        out.push('\n');
        return out;
    }
    // An advice-only row says something true about the host whatever the user
    // reported, so it is never an answer to the symptom. When it is all there is,
    // the upstream escalation still applies: it is printed first and the advice
    // follows. JSON is unaffected, `matched` and `has_match` stay as built.
    let answers_symptom = report.matched.iter().any(|d| !is_advice_only(&d.id));
    if !answers_symptom {
        let route = &report.route_when_no_match;
        out.push_str("rocm diagnose: no known misconfiguration matched.\n\n");
        out.push_str("This is the explicit 'I don't recognise this failure mode' case. Do not speculate; file the symptom + this examination output upstream:\n");
        let _ = writeln!(out, "  {:>12}: {}", route.target, route.url);
        out.push('\n');
        out.push_str("Include the JSON from `rocm examine --json` in your report.\n");
        if report.matched.is_empty() {
            return out;
        }
        out.push_str(
            "\nAdvice about this host follows. It does not explain the symptom, and nothing in it is applied automatically.\n\n",
        );
    }
    for (i, d) in report.matched.iter().take(top).enumerate() {
        let tier = if d.score >= HIGH_CONFIDENCE {
            "HIGH"
        } else if d.score >= MIN_SCORE_FOR_MATCH {
            "LIKELY"
        } else {
            "WEAK"
        };
        // The ordinal is a ranking position, not a handle -- naming the fix-id on
        // the same line stops it reading like something `rocm fix` would accept.
        // Users were typing `rocm fix #1` because the two were shown apart, with
        // only the ordinal looking like an identifier.
        let _ = writeln!(
            out,
            "#{} ({}) [{tier} score={}/100] {}",
            i + 1,
            d.id,
            d.score,
            d.title
        );
        let _ = writeln!(out, "   id: {}", d.id);
        for ev in &d.evidence {
            let _ = writeln!(out, "   - {ev}");
        }
        if let Some(fix) = &d.fix {
            let _ = writeln!(out, "   plan: {}", fix.summary);
            for c in &fix.commands {
                let _ = writeln!(out, "     $ {c}");
            }
            let flags = crate::fix::format_flags(
                fix.needs_sudo,
                fix.needs_reboot,
                fix.needs_relogin,
                fix.class,
                fix.needs.as_deref(),
            );
            let _ = writeln!(out, "   flags: {}", flags.join(", "));
            for n in &fix.notes {
                let _ = writeln!(out, "   note: {n}");
            }
            if !fix.verify.is_empty() {
                let _ = writeln!(out, "   verify after fix: {}", fix.verify);
            }
        }
        // Every finding carries the command that acts on it. Previously only the
        // summary at the end named one, and only when some match cleared
        // HIGH_CONFIDENCE -- so a report of low-confidence matches showed ids but
        // never said what to run, which is what sent users guessing at the
        // ordinal.
        //
        // Unconditional, including for a `DIAGNOSE-ONLY` cause: `rocm fix` has
        // an answer for every id in the catalog, and for that class the answer
        // is to name the problem and change nothing. The `flags:` line above
        // already said so, so the command is not a promise of a repair.
        let _ = writeln!(out, "   apply with: rocm fix {}", d.id);
        out.push('\n');
    }

    // A truncated list must say so: otherwise the matches shown read as
    // everything the CLI found.
    let shown = report.matched.len().min(top);
    if report.matched.len() > shown {
        let _ = writeln!(
            out,
            "Showing {shown} of {} matches; pass `--top {}` to see the rest.\n",
            report.matched.len(),
            report.matched.len()
        );
    }

    if let Some(high) = report.matched.iter().find(|d| d.score >= HIGH_CONFIDENCE) {
        let _ = writeln!(out, "Next step: run `rocm fix {}`.", high.id);
    } else if let Some(best) = report.matched.iter().find(|d| !is_advice_only(&d.id)) {
        // Below the threshold the caution still stands, but it is advice about
        // confidence -- not a reason to withhold the command. Ending here was the
        // bug: the user was told to confirm more evidence and given nothing to
        // run once they had.
        out.push_str(
            "Highest-scoring match is below the HIGH_CONFIDENCE threshold. Confirm one more piece of evidence before applying.\n",
        );
        let _ = writeln!(
            out,
            "When you are ready, run `rocm fix {}` (the highest-scoring match), or use the `apply with:` line of whichever cause fits.",
            best.id
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnose::test_support::linux_base;
    use crate::diagnose::{Diagnosis, Fix, diagnose};

    #[test]
    fn rendering_offers_a_command_for_a_diagnose_only_cause_that_ranks_first() {
        // The reason this entry is a `DIAGNOSE-ONLY` catalog member rather than
        // a `fix: None`: the report promises an `apply with:` line for every
        // cause, README and `--help` repeat that promise, and `rocm fix` has to
        // keep it for this id too. What marks it as unfixable is the flags
        // line, not a missing command.
        let report = DiagnoseReport {
            matched: vec![
                Diagnosis {
                    id: "info-20-gpu-no-rocm-support".to_owned(),
                    title: "GPU has no ROCm/HIP support".to_owned(),
                    score: 60,
                    evidence: vec!["gfx90c has no ROCm/HIP runtime build".to_owned()],
                    fix: Some(Fix {
                        fix_id: "info-20-gpu-no-rocm-support".to_owned(),
                        summary: "No ROCm/HIP runtime build exists for this architecture."
                            .to_owned(),
                        class: crate::fix::FixClass::DiagnoseOnly,
                        ..Fix::default()
                    }),
                },
                Diagnosis {
                    id: "fix-9-igpu-dgpu".to_owned(),
                    title: "iGPU+dGPU collision".to_owned(),
                    score: 55,
                    evidence: vec!["both an AMD APU and an AMD discrete GPU".to_owned()],
                    fix: Some(Fix {
                        fix_id: "fix-9-igpu-dgpu".to_owned(),
                        summary: "Pin HIP_VISIBLE_DEVICES.".to_owned(),
                        ..Fix::default()
                    }),
                },
            ],
            has_match: true,
            min_score_for_match: MIN_SCORE_FOR_MATCH,
            high_confidence_threshold: HIGH_CONFIDENCE,
            route_when_no_match: Route {
                target: "github".to_owned(),
                url: "https://example.invalid".to_owned(),
            },
            out_of_scope: None,
            model: None,
        };
        let out = render_report_text(&report, 5);
        assert!(
            out.contains("apply with: rocm fix info-20-gpu-no-rocm-support"),
            "every cause carries the command that acts on it, this one included:\n{out}"
        );
        assert!(
            out.contains("no reliable fix (`rocm fix` will NOT change anything)"),
            "and the flags line is what says the command will not repair anything:\n{out}"
        );
        assert!(
            out.contains("`apply with:` line"),
            "the closing pointer to the runnable causes survives unchanged:\n{out}"
        );
    }

    #[test]
    fn a_low_confidence_report_still_names_a_command_to_run() {
        // The reported case: every match below HIGH_CONFIDENCE. The report used
        // to end at "confirm one more piece of evidence" and never say what to
        // run, so users guessed at the `#1` ordinal.
        let mut e = linux_base();
        e.in_render_group = Some(false);
        e.in_video_group = Some(false);
        let report = diagnose(&e, "");
        assert!(
            report.matched.iter().all(|d| d.score < HIGH_CONFIDENCE),
            "fixture must stay below the threshold for this test to mean anything"
        );

        let out = render_report_text(&report, 5);
        assert!(
            out.contains("below the HIGH_CONFIDENCE threshold"),
            "the confidence caution must survive:\n{out}"
        );
        assert!(
            out.contains("rocm fix fix-4-render-group"),
            "a below-threshold report must still name a runnable command:\n{out}"
        );
    }

    #[test]
    fn every_reported_cause_carries_its_own_apply_command() {
        let mut e = linux_base();
        e.in_render_group = Some(false);
        e.in_video_group = Some(false);
        let report = diagnose(&e, "");
        let out = render_report_text(&report, 5);

        let applies = out.matches("apply with: rocm fix ").count();
        let shown = report.matched.len().min(5);
        assert_eq!(
            applies, shown,
            "each of the {shown} shown causes needs its own command:\n{out}"
        );
        for d in report.matched.iter().take(shown) {
            assert!(
                out.contains(&format!("apply with: rocm fix {}", d.id)),
                "no command for {}:\n{out}",
                d.id
            );
        }
    }

    #[test]
    fn the_ranking_position_is_shown_with_the_id_it_stands_for() {
        // `#1` alone looked like a handle `rocm fix` would take. Pairing it with
        // the id on the same line is what stops that read.
        let mut e = linux_base();
        e.in_render_group = Some(false);
        let report = diagnose(&e, "");
        let out = render_report_text(&report, 5);
        let top = &report.matched[0];
        assert!(
            out.contains(&format!("#1 ({})", top.id)),
            "the ordinal must name the id it refers to:\n{out}"
        );
    }

    #[test]
    fn a_truncated_report_says_it_was_truncated() {
        let mut e = linux_base();
        e.in_render_group = Some(false);
        e.in_video_group = Some(false);
        let report = diagnose(&e, "");
        if report.matched.len() < 2 {
            return; // nothing to truncate on this fixture
        }
        let out = render_report_text(&report, 1);
        assert!(
            out.contains("Showing 1 of"),
            "a truncated list must not read as the complete set:\n{out}"
        );
    }

    /// Routing must stay defined for every framework the probe can report, and
    /// must not claim targets it can never reach.
    ///
    /// The catalog docs previously advertised lemonade / ollama / lm-studio
    /// routing that no probe could ever trigger; this pins the reachable set so
    /// a re-added arm has to come with a probe that reaches it.
    ///
    /// The list below is **hand-maintained**: its first four entries mirror what
    /// `Examination::probe` sets in `examine/framework.rs`, and the rest are names the
    /// probe never returns, kept so a re-added arm for one of them shows up.
    /// Neither half is derived from that code, so adding a fifth framework to
    /// the probe will not fail this test — see the note on
    /// [`route_when_no_match`] for the three places to edit.
    #[test]
    fn routing_targets_cover_every_framework_the_probe_reports() {
        let mut targets = Vec::new();
        // The first four are what `rocm examine` can actually report. The last
        // three never come back from the probe, and are here precisely for
        // that reason: `route_when_no_match` once carried arms for them, and
        // removing those arms is what this test pins. Without these values in
        // the loop, restoring `"lemonade" => "lemonade"` changes nothing the
        // assertion below observes, and the CLI could go back to advertising
        // routing the probe can never reach with every test still green. With
        // them, an unreachable arm grows the target set and fails loudly.
        for framework in [
            "skipped",
            "pytorch",
            "llama-cpp",
            "unknown",
            "lemonade",
            "ollama",
            "lm-studio",
        ] {
            let e = Examination {
                framework: framework.to_owned(),
                ..Examination::default()
            };
            let route = route_when_no_match(&e);
            assert!(
                route.url.starts_with("http"),
                "{framework}: routed to a non-URL {:?}",
                route.url
            );
            targets.push(route.target);
        }
        targets.sort_unstable();
        targets.dedup();
        assert_eq!(
            targets,
            vec!["llama-cpp", "pytorch", "rocm-core"],
            "the reachable routing targets changed; update the docs in \
             skills/rocm-doctor/reference.md (and the amd/skills copy) to match"
        );
    }

    #[test]
    fn no_match_routes_upstream() {
        let mut e = linux_base();
        e.framework = "pytorch".to_owned();
        let report = diagnose(&e, "");
        assert!(report.matched.is_empty());
        assert!(!report.has_match());
        assert_eq!(report.route_when_no_match.target, "pytorch");
        assert!(report.route_when_no_match.url.contains("pytorch/pytorch"));
    }

    #[test]
    fn no_match_default_route_is_rocm_core() {
        let e = linux_base();
        let report = diagnose(&e, "");
        assert_eq!(report.route_when_no_match.target, "rocm-core");
    }
}
