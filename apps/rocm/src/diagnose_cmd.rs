// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm diagnose` and `rocm fix`: the command handlers (`diagnose`, `fix`) and
//! the report step behind `rocm diagnose --report` — `show_prepared_report`
//! prints what a report would carry and offers it through `perform_delivery`,
//! which acts on a delivery decision with an injected `Opener`, and
//! `established_entry` picks the catalog entry a report names.
//!
//! Mechanically relocated from `main.rs` with no behavior change; see
//! `docs/architecture.md`. `diagnose` and `fix` are `pub(crate)`: `dispatch()`
//! calls them through the re-import `use crate::diagnose_cmd::{diagnose,
//! fix};`. The other three functions have no caller outside this module and
//! stay private. `diagnose` gets its `--model` verdict from
//! `model_fit.rs`'s `assess_model_on_this_host`, and `fix` reports a failing
//! exit code to `main()` through `FixExitCode`, which stays in `main.rs` with
//! the rest of the exit-code plumbing and is reached as `crate::FixExitCode`.
//!
//! The report and delivery regression tests live in this module's own `mod
//! tests`. The `exit_code_for_*` tests and
//! `dispatch_carries_fixs_exit_code_through_to_exit_code_for`, whose subject is
//! that exit-code plumbing, stay in `main.rs`, as does
//! `the_dashboard_approval_and_the_cli_give_the_same_sender_notice`, which
//! compares two constants and calls nothing from this module.

use anyhow::{Result, bail};
use rocm_core::browser::Opener;
use rocm_core::{AppPaths, RocmCliConfig};
use serde::Serialize;

use crate::FixExitCode;
use crate::cli_report::{JsonOrText, print_json_or_text};
use crate::model_fit::assess_model_on_this_host;

pub(crate) fn diagnose(
    symptom: Option<String>,
    top: usize,
    json: bool,
    distro: Option<String>,
    model: Option<String>,
    report_requested: bool,
    send: bool,
) -> Result<()> {
    // The model verdict is about THIS machine, always. `--distro` retargets the
    // environment examination at another one, but the GPU memory and engine
    // selection behind a model verdict are read locally -- so answering for a
    // model here would describe the wrong host under a heading naming another.
    // Refuse rather than substitute, the same way `--distro` itself does when it
    // cannot reach a machine.
    //
    // Keyed on whether `--distro` was passed, not on anything the probe below
    // populates. An examination-derived signal (e.g. whether `examination.wsl`
    // ended up set, or `locally_probed`) would make this refusal depend on
    // probe internals instead of on the flag the user actually typed --
    // exactly the kind of coupling that lets a future change to the probe
    // silently let a `--distro` run fall through and answer for the local host
    // under a heading naming another machine.
    if model.is_some() && distro.is_some() {
        bail!(
            "--model answers for the machine running this command, and --distro points the \
             examination at a different one. Run `rocm diagnose --model <model>` inside the \
             distribution instead."
        );
    }
    // `rocm diagnose` is a query: it exits 0 whether it matched, found nothing,
    // or is out of scope. Callers read `has_match` / `out_of_scope` /
    // `route_when_no_match` from `--json` rather than branching on the exit code.
    //
    // `--distro` is the exception that still errors: the user named a machine to
    // inspect, and silently reporting on a different one would be worse than
    // failing. `--distro` with no value means "the only one installed".
    let examination = if let Some(name) = distro {
        let selected = (!name.is_empty()).then_some(name);
        rocm_core::probe_wsl_distro_from_host(selected.as_deref())
            .map_err(|reason| anyhow::anyhow!("{reason}"))?
    } else {
        // Same reasoning as `examine --json`: the catalog reasons over the torch
        // the engines will load, which is the active runtime's.
        //
        // Best-effort, unlike `examine`'s copy: this command already promises to
        // answer on a degraded host, so a path-discovery failure must cost only
        // the managed-runtime lookup, never the diagnosis.
        let interpreter = AppPaths::discover().ok().and_then(|paths| {
            let config = RocmCliConfig::load(&paths).unwrap_or_default();
            rocm_core::active_managed_framework_interpreter(&paths, &config)
        });
        rocm_core::Examination::probe_with_interpreter(
            rocm_core::FrameworkProbe::Auto,
            interpreter.as_ref(),
        )
    };
    let inspected_remotely = examination
        .wsl
        .as_ref()
        .is_some_and(|wsl| !wsl.locally_probed);
    let mut report = rocm_core::run_diagnose(&examination, &symptom.unwrap_or_default());
    if let Some(model_ref) = &model {
        report.model = Some(assess_model_on_this_host(model_ref, &examination));
    }
    if report_requested {
        return show_prepared_report(&examination, &report, json, send);
    }
    print_json_or_text(
        json,
        &DiagnoseOutput {
            report: &report,
            top,
            inspected_remotely,
        },
    )
}

/// What `rocm diagnose` prints: the report as JSON, or the report as text with
/// the parts that depend on how it was gathered (`--top`, a remote inspection).
struct DiagnoseOutput<'a> {
    report: &'a rocm_core::DiagnoseReport,
    top: usize,
    inspected_remotely: bool,
}

impl Serialize for DiagnoseOutput<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        self.report.serialize(serializer)
    }
}

impl JsonOrText for DiagnoseOutput<'_> {
    fn render_text(&self) -> String {
        let mut text = rocm_core::render_diagnose_text(self.report, self.top);
        if let Some(readiness) = &self.report.model {
            text.push('\n');
            text.push_str(&rocm_core::model_readiness::render_model_readiness_text(readiness));
        }
        // Inspecting a distribution from outside it collects no environment, so
        // the checks that read one never run. Without saying so, "no known
        // misconfiguration matched" reads as a clean bill of health for the
        // whole installation, when it only covers the WSL GPU stack.
        if self.inspected_remotely {
            text.push_str(
                "\nNote: inspected from outside the distribution, which sees only its \
                 WSL GPU stack.\nChecks that read the environment (HSA_OVERRIDE_GFX_VERSION, \
                 PATH, the framework/ROCm pairing)\ndid not run. For those, run `rocm diagnose` \
                 inside the distribution.\n",
            );
        }
        text
    }
}

/// The catalog entry a report should name, and whether a fix was offered for it.
///
/// Reads `has_match` rather than taking the head of `matched`. Several checkers
/// open with a nonzero score for a situation that is merely *potentially*
/// relevant, so `matched` is rarely empty even on a healthy machine — taking its
/// head regardless would publish a sub-threshold signal as though it were an
/// established cause, and the counts built on those reports would be wrong in a
/// way nothing downstream could detect.
fn established_entry(report: &rocm_core::DiagnoseReport) -> (Option<&str>, bool) {
    if !report.has_match {
        return (None, false);
    }
    report.matched.first().map_or((None, false), |top| {
        (Some(top.id.as_str()), top.fix.is_some())
    })
}

/// Act on a delivery decision, and say what happened.
///
/// Takes the decision rather than making it, and takes the opener rather than
/// being one. Both for the same reason: the decision is tested in `rocm-core`
/// against every environment, and this half has to be tested against an opener
/// that does not exist, on a machine with no browser. A function that decided
/// and opened could be verified on neither.
fn perform_delivery(
    delivery: &rocm_core::report_delivery::Delivery,
    opener: &dyn Opener,
) -> String {
    use rocm_core::report_delivery::{DESTINATION, Delivery, SENDER_NOTICE};
    let outcome = match delivery {
        // No mail client is started here on purpose, and the reason is worth
        // the line: this is the branch for a machine held over SSH, or a
        // server with no mail client at all, where starting one would open on
        // somebody else's desktop or fail silently. The address is named as
        // well as the link, because a machine in this state often cannot act
        // on a `mailto:` at all and the user has to send the mail by hand.
        Delivery::Show(url) => format!(
            "Nothing has been sent. To send this yourself, mail the report above to \
             {DESTINATION}, or open:\n  {url}"
        ),
        Delivery::Open(url) => match opener.open(url) {
            Ok(()) => format!(
                "Nothing has been sent yet. A prefilled mail to {DESTINATION} was opened, and \
                 it is sent only when you send it:\n  {url}"
            ),
            // A failed open is not a failed command. The user still has the
            // address and the link, which is the whole of what this offers.
            Err(error) => format!(
                "Nothing has been sent. A mail client could not be started ({error}). To send \
                 this yourself, mail the report above to {DESTINATION}, or open:\n  {url}"
            ),
        },
    };
    // Every branch offers a mail the user may send, so every branch says what
    // that mail reveals about them. First, straight under the report: after the
    // link, which wraps over many lines, it scrolls out of sight.
    format!("{SENDER_NOTICE}\n{outcome}")
}

/// Print the report this machine would contribute, and send nothing.
fn show_prepared_report(
    examination: &rocm_core::Examination,
    report: &rocm_core::DiagnoseReport,
    json: bool,
    send: bool,
) -> Result<()> {
    let (entry, fix_offered) = established_entry(report);
    // Exit 0 either way. A refusal is this command working, not failing: it
    // decided correctly and said why, and a nonzero code would send a caller
    // looking for a fault. Anything scripting this reads the outcome from
    // `--json` rather than from the exit code, exactly as `rocm diagnose` itself
    // already asks callers to do.
    match rocm_core::prepare_report(examination, entry, fix_offered) {
        Ok(prepared) => {
            if json {
                println!("{}", serde_json::to_string_pretty(&prepared)?);
            } else {
                println!("This is the whole of what a report would carry:");
                println!();
                println!("{}", serde_json::to_string_pretty(&prepared)?);
                println!();
                // The content is printed above before this decides anything,
                // so a report is always read before its form is offered. That
                // ordering is the promise `--send` makes, and `--send`
                // requires `--report` so it cannot be skipped.
                let delivery = rocm_core::report_delivery::deliver(&prepared, send, &|key| {
                    std::env::var(key).ok()
                });
                println!(
                    "{}",
                    perform_delivery(&delivery, &rocm_core::browser::SystemOpener)
                );
            }
            Ok(())
        }
        Err(refusal) => {
            let explanation = match refusal {
                rocm_core::ReportRefusal::UnreleasedHardware => {
                    // "Doctor" is what the epic calls this capability; the CLI
                    // has no such command, so a user reading this has nothing
                    // to run and nothing to look up.
                    "This machine holds hardware that is not on AMD's published ROCm \
                     compatibility matrix, so no report was prepared. A report describes only \
                     hardware the compatibility matrix lists as supported."
                }
                rocm_core::ReportRefusal::ArchitectureUnreadable => {
                    "No AMD GPU architecture could be read here, so nothing confirms this \
                     hardware is on the ROCm compatibility matrix. No report was prepared."
                }
                rocm_core::ReportRefusal::PlatformNotProbed => {
                    // Says what happened rather than dressing it as a finding
                    // about the machine. The earlier wording told a healthy WSL
                    // user their GPU could not be read, when nothing had looked.
                    "This CLI does not inspect the GPU on WSL yet, so it cannot confirm whether \
                     this hardware is on the ROCm compatibility matrix. No report was prepared. \
                     This is a gap in the tool, not a problem with the machine."
                }
            };
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&rocm_core::refusal_envelope(
                        refusal,
                        explanation
                    ))?
                );
            } else {
                println!("{explanation}");
            }
            Ok(())
        }
    }
}

pub(crate) fn fix(
    fix_id: Option<String>,
    yes: bool,
    dry_run: bool,
    device_index: Option<i64>,
    json: bool,
) -> Result<()> {
    let Some(fix_id) = fix_id else {
        if json {
            print!("{}", rocm_core::catalog_manifest_json()?);
        } else {
            print!("{}", rocm_core::list_fix_recipes());
        }
        return Ok(());
    };
    // Refused rather than ignored. "Apply this fix, as JSON" has no meaning, and
    // quietly dropping the flag would let a caller believe it had asked for
    // machine-readable output and got it.
    if json {
        anyhow::bail!(
            "`--json` describes the whole catalog, so it cannot be combined with a fix id. \
             Run `rocm fix --json` to read the catalog, or `rocm fix {fix_id}` to apply this fix."
        );
    }
    let opts = rocm_core::FixOptions {
        yes,
        dry_run,
        device_index,
    };
    let code = rocm_core::apply_fix(&fix_id, &opts);
    if code != 0 {
        return Err(FixExitCode(code).into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use rocm_core::report_delivery::Delivery;

    use super::*;

    /// An opener that records rather than opens, and can be told to fail.
    ///
    /// The whole reason the opener is a trait: the real one spawns a browser
    /// against whatever desktop exists, so neither "it was opened" nor "it was
    /// deliberately not opened" can be observed in CI without this.
    struct RecordingOpener {
        opened: RefCell<Vec<String>>,
        fails: bool,
    }

    impl RecordingOpener {
        fn working() -> Self {
            Self {
                opened: RefCell::new(Vec::new()),
                fails: false,
            }
        }
        fn broken() -> Self {
            Self {
                opened: RefCell::new(Vec::new()),
                fails: true,
            }
        }
        fn opened(&self) -> Vec<String> {
            self.opened.borrow().clone()
        }
    }

    impl Opener for RecordingOpener {
        fn open(&self, url: &str) -> anyhow::Result<()> {
            self.opened.borrow_mut().push(url.to_owned());
            if self.fails {
                anyhow::bail!("no browser here");
            }
            Ok(())
        }
    }

    /// Nothing is opened unless the decision was to open.
    ///
    /// The assertion that matters is on the opener, not on the wording. A
    /// message saying no browser was started is satisfied by any string; an
    /// opener that recorded nothing is the actual claim.
    #[test]
    fn a_delivery_that_is_not_an_open_never_reaches_the_browser() {
        let delivery = Delivery::Show("mailto:nobody@example.invalid".to_owned());
        let opener = RecordingOpener::working();
        let said = perform_delivery(&delivery, &opener);

        assert!(
            opener.opened().is_empty(),
            "a mail client was started for {delivery:?}, which is the one thing this path must \
             not do on a machine the user is holding over SSH"
        );
        assert!(
            said.contains("Nothing has been sent"),
            "the user has to be told nothing left the machine: {said}"
        );
        // A machine in this state often cannot act on a `mailto:` at all, so
        // the address has to be readable on its own, not only inside the link.
        assert!(
            said.contains(rocm_core::report_delivery::DESTINATION),
            "a user who has to send the mail by hand needs the address: {said}"
        );
    }

    /// Opening is what an open decision does, and the user is told it is not
    /// filed yet.
    #[test]
    fn an_open_decision_reaches_the_browser_and_is_still_not_a_send() {
        let url = "https://example.invalid/new?body=x";
        let opener = RecordingOpener::working();
        let said = perform_delivery(&Delivery::Open(url.to_owned()), &opener);

        assert_eq!(
            opener.opened(),
            vec![url.to_owned()],
            "premise failed: an open decision must reach the opener, otherwise the cases above \
             are satisfied by never opening anything"
        );
        assert!(
            said.contains("only when you send it"),
            "opening a prefilled mail is not sending it, and the user has to know which one \
             happened: {said}"
        );
    }

    /// A browser that will not start still leaves the user the link.
    #[test]
    fn a_browser_that_fails_to_start_still_hands_the_user_the_link() {
        let url = "https://example.invalid/new?body=x";
        let said = perform_delivery(&Delivery::Open(url.to_owned()), &RecordingOpener::broken());

        assert!(
            said.contains(url),
            "the link is the whole of what this offers, so a failed browser must not lose it: \
             {said}"
        );
        assert!(said.contains("Nothing has been sent"));
    }

    /// Every way a mail is offered says it carries the sender's address.
    ///
    /// This pins that the notice is printed on every branch. Its second
    /// sentence, that the report names no user, host, or file path, is proven
    /// against a real machine by the scenarios `diagnose-30`
    /// (`@id:diagnose-report-carries-no-identifying-detail`) and `diagnose-32`
    /// (`@id:diagnose-send-on-a-headless-machine-prints-instead-of-opening`),
    /// which sweep the report for exactly those.
    #[test]
    fn every_offered_mail_says_it_carries_the_senders_address() {
        let url = "mailto:nobody@example.invalid";
        for (delivery, opener) in [
            (Delivery::Show(url.to_owned()), RecordingOpener::working()),
            (Delivery::Open(url.to_owned()), RecordingOpener::working()),
            (Delivery::Open(url.to_owned()), RecordingOpener::broken()),
        ] {
            let said = perform_delivery(&delivery, &opener);
            assert!(
                said.starts_with(rocm_core::report_delivery::SENDER_NOTICE),
                "the user was offered a mail without being told first that it carries their \
                 address: {said}"
            );
            // Before the link, which wraps over many lines and would push a
            // notice after it out of sight.
            assert!(said.find(url) > said.find(rocm_core::report_delivery::SENDER_NOTICE));
        }
    }

    /// A diagnosis report holding exactly one finding.
    ///
    /// `has_match` is passed independently of the score on purpose: the point
    /// under test is that the two are read together, so a fixture that derived
    /// one from the other could not express the case being guarded against.
    fn report_of(
        has_match: bool,
        id: &str,
        score: i32,
        fix: Option<rocm_core::Fix>,
    ) -> rocm_core::DiagnoseReport {
        rocm_core::DiagnoseReport {
            has_match,
            matched: vec![rocm_core::Diagnosis {
                id: id.to_owned(),
                title: "under test".to_owned(),
                score,
                evidence: Vec::new(),
                fix,
            }],
            min_score_for_match: 50,
            high_confidence_threshold: 80,
            route_when_no_match: rocm_core::diagnose::Route {
                target: String::new(),
                url: String::new(),
            },
            out_of_scope: None,
            model: None,
        }
    }

    /// The entry a report names is one the diagnosis established, not merely
    /// the strongest signal it saw.
    ///
    /// This is a wiring test, not a logic one. `established_entry` is correct in
    /// itself; what it could get wrong is being handed `matched.first()`
    /// unconditionally. Several checkers open with a nonzero score for a
    /// situation that is only potentially relevant, so a healthy machine
    /// produces a `matched` list full of sub-threshold entries — and a report
    /// naming one of those would look like an established cause to every
    /// counter downstream, with nothing able to tell the difference afterwards.
    #[test]
    fn a_report_names_an_established_cause_and_not_the_loudest_weak_signal() {
        let weak_only = report_of(false, "fix-10-container", 25, None);
        assert_eq!(
            established_entry(&weak_only),
            (None, false),
            "nothing cleared the bar, so the report has no entry to name"
        );

        // Non-vacuity: an established cause must come through, or the assertion
        // above is satisfied by never naming anything.
        let established = report_of(true, "fix-6-path", 90, Some(rocm_core::Fix::default()));
        assert_eq!(
            established_entry(&established),
            (Some("fix-6-path"), true),
            "an established cause with a fix is exactly what a report is for"
        );
    }
}
