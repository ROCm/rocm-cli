// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Installing ROCm on a machine nobody is watching.
//!
//! This is the riskiest thing `rocm remote` can do, and the parts that make it
//! safe are refusals rather than capability. The install itself already exists
//! and is already non-interactive; what is new here is deciding when it is
//! allowed to run.
//!
//! Three guards, all of them about a human not being present:
//!
//! The machine's state is scored against the failure-mode catalog *first*. That
//! catalog exists because ROCm installs go wrong in specific, recognisable
//! ways — a half-configured driver, a DKMS build against the wrong kernel,
//! Secure Boot refusing an unsigned module, repo files left over from a
//! previous attempt. Each of those needs a person to decide, and the wizard
//! that walks a human through them cannot do so over SSH. So anything the
//! catalog recognises stops the install and prints what it found. Only a
//! machine that is plainly missing ROCm, with nothing else wrong, proceeds.
//!
//! And a machine the catalog could not score at all is refused too, which is a
//! different refusal and not a subtlety. On a platform the catalog carries no
//! entries for it runs nothing and returns no findings — the same empty result
//! a healthy machine produces. "Nothing was checked" and "nothing is wrong" are
//! opposite facts wearing one shape, and only reading them apart keeps the
//! gating above from being satisfied by a machine nobody looked at.
//!
//! And privilege escalation is checked before it is needed. The driver install
//! runs commands through `sudo`, while the control channel refuses to answer
//! prompts by design — so a machine asking for a password does not fail, it
//! hangs. Establishing that sudo is passwordless first turns the most likely
//! real-world failure into a sentence instead of a stall.

use anyhow::{Result, bail};
use rocm_core::diagnose::DiagnoseReport;

use super::bullets;
use super::doctor;
use super::transport::Transport;

/// Why an unattended install was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// The catalog has no entries for this platform, so it scored nothing.
    /// Carries the catalog's own explanation.
    NotEvaluated { reason: String },
    /// The catalog recognised something needing a person, and at least one of
    /// those things is something a person can act on.
    ///
    /// `without_a_fix` is carried beside `findings` rather than folded into it
    /// because the message tells the user to resolve what it lists, and an
    /// entry the catalog has no remedy for cannot be resolved. Naming those
    /// apart is what keeps that instruction true of everything it is said
    /// about.
    NeedsAHuman {
        findings: Vec<String>,
        without_a_fix: Vec<String>,
    },
    /// Everything the catalog recognised is a `DIAGNOSE-ONLY` statement about
    /// the machine, with no remedy for any of it.
    ///
    /// Still a refusal — the install would be proceeding on a machine the
    /// catalog has already said something about — but a different one. There
    /// is nothing to resolve and running it again reaches the same answer, so
    /// "resolve them first, then run this again" would be an instruction with
    /// no way to carry it out. AGENTS.md §3: remediation text has to be able to
    /// clear the condition that printed it.
    NothingToInstall { findings: Vec<String> },
    /// Privileged commands would block on a password prompt.
    NeedsPasswordlessSudo,
}

/// Decide whether ROCm may be installed on this machine unattended.
///
/// Split from the doing so the judgement is testable on its own — the part
/// worth being sure about is what gets refused, not what gets run.
pub(crate) fn assess(report: &DiagnoseReport, passwordless_sudo: bool) -> Option<Refusal> {
    // Asked first, and deliberately not by reading `matched`. When the catalog
    // has no entries for a platform it does not run and returns an *empty*
    // `matched` — the same shape a machine with nothing wrong produces. The two
    // are opposite facts: one is a verdict, the other is the absence of one, and
    // only `out_of_scope` tells them apart. Gating on `matched` alone therefore
    // reads "nothing was checked" as "nothing else is wrong", which is the one
    // way to reach an unattended, privileged, multi-minute install on a machine
    // this module's premise says must have been looked at first.
    if let Some(reason) = &report.out_of_scope {
        return Some(Refusal::NotEvaluated {
            reason: reason.clone(),
        });
    }
    let (findings, without_a_fix) = recognised_problems(report);
    if findings.is_empty() && !without_a_fix.is_empty() {
        return Some(Refusal::NothingToInstall {
            findings: without_a_fix,
        });
    }
    if !findings.is_empty() {
        return Some(Refusal::NeedsAHuman {
            findings,
            without_a_fix,
        });
    }
    if !passwordless_sudo {
        return Some(Refusal::NeedsPasswordlessSudo);
    }
    None
}

/// Catalog findings strong enough to be treated as real, split by whether
/// anything can be done about them: `(actionable, without_a_fix)`.
///
/// Weak signals are excluded deliberately: several checks open with a low score
/// for a situation that is merely *potentially* relevant, and treating those as
/// blockers would refuse installs on healthy machines until nobody trusted the
/// refusal.
///
/// The split is by [`FixClass::DiagnoseOnly`] rather than by `fix.is_none()`:
/// every catalog entry carries a `Fix`, and the class is where "detected,
/// explained, no remedy" is recorded. A missing `Fix` falls on the *same* side
/// — `is_none_or` — because the question this answers is "has the user anything
/// to apply", and no `Fix` is no answer either. That arm is unreachable from
/// today's catalog, which is why
/// `a_diagnosis_with_no_fix_at_all_is_not_called_resolvable` pins it: nothing
/// else would notice it inverting, and inverted it files a no-remedy diagnosis
/// under "Resolve them first — the suggested fixes are below".
///
/// Both halves still refuse the install; they differ only in what the refusal
/// can honestly ask the user to do next.
fn recognised_problems(report: &DiagnoseReport) -> (Vec<String>, Vec<String>) {
    let mut actionable = Vec::new();
    let mut without_a_fix = Vec::new();
    for diagnosis in &report.matched {
        if diagnosis.score < rocm_core::diagnose::MIN_SCORE_FOR_MATCH {
            continue;
        }
        let fixless = diagnosis
            .fix
            .as_ref()
            .is_none_or(|fix| fix.class == rocm_core::fix::FixClass::DiagnoseOnly);
        if fixless {
            without_a_fix.push(diagnosis.title.clone());
        } else {
            actionable.push(diagnosis.title.clone());
        }
    }
    (actionable, without_a_fix)
}

/// Can privileged commands run without a prompt nobody will answer?
pub(crate) fn has_passwordless_sudo(transport: &dyn Transport) -> Result<bool> {
    // `-n` makes sudo fail rather than prompt, which is the whole question.
    Ok(transport.exec("sudo -n true")?.success)
}

/// Install ROCm on the remote, having decided it is safe to.
pub(crate) fn install(transport: &dyn Transport, target: &str, remote_cli: &str) -> Result<()> {
    println!("Installing ROCm on {target}. This can take several minutes.");

    let driver = transport.exec(&format!("{remote_cli} install driver --yes"))?;
    if !driver.success {
        bail!(
            "the driver install failed on {target}: {}\n\
             Nothing further was attempted. Check the machine with \
             `rocm remote doctor {target}`.",
            driver.stderr.trim()
        );
    }
    print_indented(&driver.stdout);

    let sdk = transport.exec(&format!("{remote_cli} install sdk --yes"))?;
    if !sdk.success {
        bail!(
            "the driver installed on {target} but the ROCm SDK did not: {}\n\
             The machine is part-way through a setup; check it with \
             `rocm remote doctor {target}` before retrying.",
            sdk.stderr.trim()
        );
    }
    print_indented(&sdk.stdout);

    // The driver install records that a reboot is needed rather than performing
    // one. Saying so matters more here than locally: nobody is sitting at this
    // machine to notice it behaving as though the install did not take.
    if mentions_reboot(&driver.stdout) || mentions_reboot(&sdk.stdout) {
        println!();
        println!("{target} needs a reboot before it can serve.");
        println!("  reboot it, then run: rocm remote doctor {target}");
    }
    Ok(())
}

fn mentions_reboot(output: &str) -> bool {
    output.lines().any(|line| {
        let line = line.to_ascii_lowercase();
        line.contains("reboot_required: true") || line.contains("reboot required")
    })
}

fn print_indented(output: &str) {
    for line in output.lines().filter(|line| !line.trim().is_empty()) {
        println!("  {line}");
    }
}

/// Explain a refusal, including what to run instead.
pub(crate) fn describe_refusal(
    refusal: &Refusal,
    target: &str,
    report: &DiagnoseReport,
    top: usize,
) -> String {
    match refusal {
        // Worded so it cannot be read as a clean bill of health, which is the
        // whole hazard: an empty findings list looks identical to a healthy
        // machine's. The catalog's own sentence is carried too, because it names
        // the platform the remote reported and this one does not.
        Refusal::NotEvaluated { reason } => format!(
            "{target} runs a platform the ROCm failure catalog has no entries for, so nothing \
             was checked there and there is no verdict to install on.\n\n\
             What the check reported:\n  {reason}\n\n\
             An unattended install is only allowed on a machine the catalog looked at and \
             found nothing wrong with. This is a machine it never looked at, which is a \
             different thing. Set it up yourself if you are sure it is supported:\n  \
             ssh {target} -- rocm bootstrap setup"
        ),
        Refusal::NeedsAHuman {
            findings,
            without_a_fix,
        } => format!(
            "{target} is not in a state that can be set up unattended.\n\n\
             What was found:\n{}\n\n\
             These are the situations the setup wizard exists to walk a person through, \
             and it cannot do that over a connection with nobody watching. Resolve them \
             first - the suggested fixes are below.{}\n\n{}",
            bullets(findings),
            if without_a_fix.is_empty() {
                // Only here is a rerun promised, because only here is resolving
                // the list above enough to let one through: with nothing in
                // `without_a_fix`, the next `assess` on a resolved machine falls
                // past every refusal.
                "\n\nOnce they are, run this again.".to_owned()
            } else {
                // Kept out of the list above on purpose: "resolve them first"
                // is addressed to that list, and these cannot be resolved.
                //
                // And no rerun is offered. Resolving the actionable half leaves
                // these behind, `assess` then takes its `findings.is_empty() &&
                // !without_a_fix.is_empty()` branch, and the second run refuses
                // again — so "then run this again" would be describing a path
                // that does not exist. AGENTS.md §3.
                format!(
                    "\n\nAlso reported, with no fix to apply - these describe the machine \
                     rather than something to put right, and will still be true afterwards:\n{}\
                     \n\nBecause of them, an unattended install will still be refused once the \
                     list above is resolved. Set this machine up yourself if you want ROCm on \
                     it regardless of them.",
                    bullets(without_a_fix)
                )
            },
            doctor::render_report(target, report, top)
        ),
        // Names no install command. Every member of this class is a statement
        // about what the machine *is*, and `rocm bootstrap setup` — the command
        // this used to offer — cannot clear one: over ssh without a terminal it
        // only prints that it needs one, and with a terminal the SDK install
        // reaches `therock`'s own refusal for the same hardware. Pointing at a
        // command that is guaranteed to decline is worse than pointing at none.
        Refusal::NothingToInstall { findings } => format!(
            "ROCm will not be installed on {target}, and no rerun of this changes that.\n\n\
             What was found:\n{}\n\n\
             The catalog has no fix for any of this: it describes what the machine is, not \
             something that went wrong. There is nothing to resolve, a second run reaches the \
             same answer, and no install command clears it - running one on that machine \
             yourself meets the same facts. What is left is a decision about the machine; \
             the report below is what there is to make it on.\n\n{}",
            bullets(findings),
            doctor::render_report(target, report, top)
        ),
        Refusal::NeedsPasswordlessSudo => format!(
            "installing ROCm on {target} needs administrator rights, and this connection \
             cannot answer a password prompt - it would hang rather than fail.\n\n\
             Either set up passwordless sudo there, or install it yourself:\n  \
             ssh {target} -- rocm bootstrap setup"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::transport::{ScriptedStep, ScriptedTransport};
    use rocm_core::diagnose::{Diagnosis, MIN_SCORE_FOR_MATCH};

    /// The refusal's own `What was found:` list, with the appended
    /// `doctor::render_report` excluded.
    ///
    /// Asserting `message.contains(title)` is NOT a check on that list. Every
    /// refusal ends with the rendered report, which prints each matched
    /// diagnosis by title, so the whole-message form holds with
    /// [`bullets`] returning nothing at all — a refusal whose own summary is a
    /// bare header, leaving the reader to recover the findings from a report
    /// underneath it. The list is what this module composes, so the list is what
    /// gets looked at.
    ///
    /// Bounded by the blank line that ends the section rather than by the next
    /// known heading, so a later insertion between the two does not quietly
    /// widen it.
    fn what_was_found(message: &str) -> String {
        let after = message
            .split_once("What was found:\n")
            .map_or_else(String::new, |(_, rest)| rest.to_owned());
        after
            .lines()
            .take_while(|line| !line.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The `with no fix to apply` aside's own list, on the same terms and for
    /// the same reason.
    fn also_reported(message: &str) -> String {
        let after = message
            .split_once("will still be true afterwards:\n")
            .map_or_else(String::new, |(_, rest)| rest.to_owned());
        after
            .lines()
            .take_while(|line| !line.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// What this module wrote, with the rendered doctor report it appends taken
    /// off the end.
    ///
    /// The report is `rocm remote doctor`'s output, verbatim, and it carries its
    /// own remediation lines for whatever it matched. Searching the whole
    /// message for a command would therefore be reading that output and
    /// attributing it here.
    fn refusal_prose(message: &str, target: &str, report: &DiagnoseReport) -> String {
        let rendered = doctor::render_report(target, report, 5);
        message
            .strip_suffix(&rendered)
            .unwrap_or_else(|| panic!("every refusal ends with the rendered report:\n{message}"))
            .to_owned()
    }

    fn report_with(diagnoses: Vec<Diagnosis>) -> DiagnoseReport {
        let mut report = rocm_core::diagnose::diagnose(
            &rocm_core::Examination::probe(rocm_core::FrameworkProbe::Skip),
            "",
        );
        report.matched = diagnoses;
        report
    }

    /// A finding with something to apply, which is what most of the catalog is.
    fn finding(title: &str, score: i32) -> Diagnosis {
        Diagnosis {
            id: "x".to_owned(),
            title: title.to_owned(),
            score,
            fix: Some(rocm_core::diagnose::Fix {
                fix_id: "x".to_owned(),
                class: rocm_core::fix::FixClass::PrintOnly,
                ..rocm_core::diagnose::Fix::default()
            }),
            ..Diagnosis::default()
        }
    }

    /// A finding the catalog classes `DIAGNOSE-ONLY`: real, scored, and with
    /// nothing a person could do about it. `info-20-gpu-no-rocm-support` is the
    /// first one, at a flat 60 — above `MIN_SCORE_FOR_MATCH`, so it reaches
    /// this module as a refusal and not as a weak signal.
    fn finding_without_a_fix(title: &str, score: i32) -> Diagnosis {
        Diagnosis {
            fix: Some(rocm_core::diagnose::Fix {
                fix_id: "info-20-gpu-no-rocm-support".to_owned(),
                class: rocm_core::fix::FixClass::DiagnoseOnly,
                ..rocm_core::diagnose::Fix::default()
            }),
            ..finding(title, score)
        }
    }

    /// An examination of a machine the catalog has no entries for.
    ///
    /// `platform_family` reads `os_family` (and `is_wsl`), and the catalog
    /// carries checkers for `linux`, `windows` and `wsl` only — so anything else
    /// is a host on which nothing is ever run.
    fn out_of_scope_examination() -> rocm_core::Examination {
        let mut examination = rocm_core::Examination::probe(rocm_core::FrameworkProbe::Skip);
        examination.is_wsl = false;
        examination.os_family = "darwin".to_owned();
        examination
    }

    /// A report for such a machine, produced by the real [`rocm_core::diagnose`]
    /// rather than hand-built.
    ///
    /// What makes a platform out of scope is the catalog's own coverage rule.
    /// Constructing a `DiagnoseReport { out_of_scope: Some(..), .. }` by hand
    /// would assert against this test's idea of that rule instead of against the
    /// catalog's, and would keep passing if the two ever diverged.
    fn out_of_scope_report() -> DiagnoseReport {
        let report = rocm_core::diagnose::diagnose(&out_of_scope_examination(), "");
        assert!(
            report.out_of_scope.is_some(),
            "the fixture must actually be a platform the catalog declines to score"
        );
        assert!(
            report.matched.is_empty(),
            "an out-of-scope report carries no findings, which is the whole hazard"
        );
        report
    }

    #[test]
    fn a_machine_the_catalog_never_scored_is_not_installed_on_unattended() {
        // `matched` is empty here not because the machine is healthy but because
        // no checker was ever run against it. Reading that emptiness as "nothing
        // else wrong" is the one way to reach an unattended install with no
        // catalog verdict at all — which is what this module says gates it.
        let report = out_of_scope_report();
        assert!(
            assess(&report, true).is_some(),
            "a platform the catalog never evaluated must not be installed on unattended"
        );
    }

    #[test]
    fn an_unscored_platform_says_nothing_was_checked_rather_than_naming_a_finding() {
        // The refusal a person reads has to say which of the two it is: the
        // catalog found something, or the catalog was never run. Reporting an
        // empty findings list would read as "nothing is wrong, but no".
        let report = out_of_scope_report();
        let refusal =
            assess(&report, true).expect("an unevaluated platform must produce a refusal");
        let message = describe_refusal(&refusal, "mac-mini", &report, 5);
        assert!(
            message.contains("nothing was checked"),
            "the refusal must say the catalog never ran: {message}"
        );
        assert!(
            message.contains("mac-mini"),
            "and name the machine it is about: {message}"
        );
    }

    #[test]
    fn an_unscored_platform_outranks_a_sudo_problem() {
        // Same reasoning as a recognised finding outranking one: fixing sudo
        // would not make this machine installable, so leading with sudo sends
        // someone down a path that ends nowhere.
        let report = out_of_scope_report();
        assert_ne!(
            assess(&report, false),
            Some(Refusal::NeedsPasswordlessSudo),
            "an unevaluated platform is not a sudo problem"
        );
    }

    #[test]
    fn a_clean_machine_with_working_sudo_may_be_installed() {
        assert_eq!(assess(&report_with(vec![]), true), None);
    }

    #[test]
    fn a_recognised_problem_stops_the_install_and_names_itself() {
        // These are exactly the cases the interactive wizard exists for, and it
        // cannot walk anybody through them over a connection nobody is watching.
        let report = report_with(vec![finding("DKMS built against another kernel", 90)]);
        assert_eq!(
            assess(&report, true),
            Some(Refusal::NeedsAHuman {
                findings: vec!["DKMS built against another kernel".to_owned()],
                without_a_fix: vec![],
            })
        );

        // The aside is conditional, and nothing else pins the empty case: a
        // guard that always took the non-empty branch would print the "Also
        // reported, with no fix to apply" header with nothing under it, telling
        // a reader this machine has an unfixable problem it does not have.
        let message = describe_refusal(
            &assess(&report, true).expect("refused"),
            "gpu-box",
            &report,
            5,
        );
        assert!(
            !message.contains("with no fix to apply"),
            "every finding here is actionable, so the fixless aside must not appear \
             at all -- an empty one is a dangling header: {message}"
        );
    }

    /// A machine whose only finding has no fix is refused, and told so in words
    /// it can act on.
    ///
    /// `info-20-gpu-no-rocm-support` is a flat 60, so it clears
    /// `MIN_SCORE_FOR_MATCH` on its own and reaches this module as a refusal.
    /// Sending "Resolve them first — the suggested fixes are below — then run
    /// this again" for it would be an instruction with no way to carry it out:
    /// the catalog has no fix, and the second run reaches the same answer.
    #[test]
    fn a_finding_with_no_fix_is_refused_without_being_called_resolvable() {
        let report = report_with(vec![finding_without_a_fix(
            "GPU has no ROCm/HIP support (Vulkan backend still works)",
            60,
        )]);
        assert_eq!(
            assess(&report, true),
            Some(Refusal::NothingToInstall {
                findings: vec![
                    "GPU has no ROCm/HIP support (Vulkan backend still works)".to_owned()
                ],
            }),
            "a fixless finding is its own refusal, not one that asks for it to be resolved"
        );

        let message = describe_refusal(
            &assess(&report, true).expect("refused"),
            "lemonade-laptop",
            &report,
            5,
        );
        assert!(
            what_was_found(&message).contains("GPU has no ROCm/HIP support"),
            "the refusal's own list must name what it found, not leave it to the \
             report appended below: {message}"
        );
        assert!(
            !message.contains("Resolve them first"),
            "nothing here can be resolved, so the message must not ask for it: {message}"
        );
        assert!(
            message.contains("no fix for any of this"),
            "and it must say why there is nothing to do: {message}"
        );
        // AGENTS.md §3. `rocm bootstrap setup` was offered here and cannot clear
        // this: over ssh with no terminal it prints only that it needs one, and
        // with a terminal the SDK install refuses on the same hardware fact.
        // Any install command named in this refusal is in the same position, so
        // the check is on the shape and not on that one string.
        let prose = refusal_prose(&message, "lemonade-laptop", &report);
        for named in ["bootstrap setup", "rocm install", "install sdk"] {
            assert!(
                !prose.contains(named),
                "the refusal names `{named}`, which cannot clear a finding the catalog \
                 has no fix for: {prose}"
            );
        }
    }

    /// A diagnosis carrying no `Fix` at all is counted as fixless, not as
    /// something a person can resolve.
    ///
    /// `recognised_problems` reads `.is_none_or(|fix| fix.class ==
    /// DiagnoseOnly)`, and the `is_none` half had nothing standing on it:
    /// turning it into `is_some_and` left every test in this module passing,
    /// while routing a no-`Fix` diagnosis into "Resolve them first — the
    /// suggested fixes are below" with no suggested fix to be below it.
    ///
    /// The catalog gives every entry a `Fix` today, which is exactly why this is
    /// pinned here: the fallback is unreachable from the catalog, so nothing
    /// else would ever notice it inverting.
    #[test]
    fn a_diagnosis_with_no_fix_at_all_is_not_called_resolvable() {
        let mut fixless = finding("something the catalog carries no remedy for", 90);
        fixless.fix = None;
        let report = report_with(vec![fixless]);
        assert_eq!(
            assess(&report, true),
            Some(Refusal::NothingToInstall {
                findings: vec!["something the catalog carries no remedy for".to_owned()],
            }),
            "a diagnosis with no Fix has nothing to suggest, so it cannot be put under \
             an instruction to apply the suggestion"
        );
    }

    /// A fixable finding beside a fixless one keeps "resolve them first" true
    /// of the list it is said about.
    ///
    /// The mixed case is the one a single list gets wrong: naming both under
    /// "Resolve them first" makes the instruction false for one of them, and
    /// dropping the fixless one hides a fact about the machine that is still
    /// going to be true afterwards.
    #[test]
    fn a_fixless_finding_beside_a_fixable_one_is_listed_apart_from_it() {
        let report = report_with(vec![
            finding("DKMS built against another kernel", 90),
            finding_without_a_fix(
                "GPU has no ROCm/HIP support (Vulkan backend still works)",
                60,
            ),
        ]);
        assert_eq!(
            assess(&report, true),
            Some(Refusal::NeedsAHuman {
                findings: vec!["DKMS built against another kernel".to_owned()],
                without_a_fix: vec![
                    "GPU has no ROCm/HIP support (Vulkan backend still works)".to_owned()
                ],
            })
        );

        let message = describe_refusal(
            &assess(&report, true).expect("refused"),
            "gpu-box",
            &report,
            5,
        );
        let resolve = message
            .find("Resolve them first")
            .expect("the actionable half still says to resolve what it listed");
        let aside = message
            .find("with no fix to apply")
            .expect("the fixless half is called out separately");
        assert!(
            message[..resolve].contains("DKMS built against another kernel"),
            "the resolvable finding belongs above the instruction to resolve it: {message}"
        );
        assert!(
            resolve < aside,
            "the aside belongs after the instruction: {message}"
        );
        // Each half's own list, not `message.contains`: the rendered report at
        // the foot prints every matched title and sits after both, so the
        // whole-message form holds with either list emptied.
        assert_eq!(
            what_was_found(&message).trim(),
            "- DKMS built against another kernel",
            "the resolvable half lists exactly what it asks to be resolved: {message}"
        );
        assert_eq!(
            also_reported(&message).trim(),
            "- GPU has no ROCm/HIP support (Vulkan backend still works)",
            "and the fixless half lists exactly what cannot be: {message}"
        );
        // AGENTS.md §3: resolving the list above will NOT let a rerun through —
        // `assess` then takes its `findings.is_empty() && !without_a_fix
        // .is_empty()` branch and refuses again — so the mixed case must not
        // promise one. The empty case, which may, is pinned below.
        assert!(
            !refusal_prose(&message, "gpu-box", &report).contains("run this again"),
            "a rerun is offered that `assess` would refuse a second time: {message}"
        );
    }

    /// With nothing fixless beside them, resolving the listed findings really
    /// does let the next run through — so that case, and only that case, says
    /// so.
    ///
    /// The two halves are pinned together because the risk is one wording
    /// serving both: "then run this again" is true here and false in the mixed
    /// case above, and a single unconditional sentence would read as correct
    /// against either test alone.
    #[test]
    fn resolvable_findings_on_their_own_do_promise_a_rerun() {
        let report = report_with(vec![finding("DKMS built against another kernel", 90)]);
        let message = describe_refusal(
            &assess(&report, true).expect("refused"),
            "gpu-box",
            &report,
            5,
        );
        assert!(
            message.contains("run this again"),
            "nothing here blocks a second run once the list is resolved, so the \
             instruction has to say so: {message}"
        );
        assert_eq!(
            what_was_found(&message).trim(),
            "- DKMS built against another kernel",
            "and the list it says that about is the refusal's own: {message}"
        );
        // Proof that the resolved machine is in fact installable, rather than a
        // claim about it: the same report with the finding cleared.
        assert_eq!(
            assess(&report_with(vec![]), true),
            None,
            "the promise above is only true if a cleared report installs"
        );
    }

    #[test]
    fn a_weak_signal_does_not_block_an_otherwise_healthy_machine() {
        // Several checks open with a low score for something merely potentially
        // relevant. Treating those as blockers would refuse healthy machines
        // until nobody believed the refusal.
        let report = report_with(vec![finding(
            "possibly in a container",
            MIN_SCORE_FOR_MATCH - 1,
        )]);
        assert_eq!(assess(&report, true), None);
    }

    #[test]
    fn a_machine_that_would_prompt_for_a_password_is_refused_not_hung() {
        // The control channel never answers prompts, so without this the install
        // stalls instead of failing — the worst outcome of the three.
        assert_eq!(
            assess(&report_with(vec![]), false),
            Some(Refusal::NeedsPasswordlessSudo)
        );
    }

    #[test]
    fn a_fixless_finding_outranks_a_sudo_problem() {
        // Same reasoning as the other two refusal variants, and the one this
        // change introduces: sorting out passwordless sudo would not make this
        // machine installable, because the catalog has nothing to install for
        // its GPU either way. Leading with sudo would send someone to configure
        // it and then meet this refusal anyway.
        let report = report_with(vec![finding_without_a_fix(
            "GPU has no ROCm/HIP support (Vulkan backend still works)",
            60,
        )]);
        assert!(
            matches!(
                assess(&report, false),
                Some(Refusal::NothingToInstall { .. })
            ),
            "a machine with nothing to install is not a sudo problem: {:?}",
            assess(&report, false)
        );
    }

    #[test]
    fn a_recognised_problem_outranks_a_sudo_problem() {
        // Fixing sudo would not make this machine installable, so saying so
        // first would send someone down the wrong path.
        let report = report_with(vec![finding("Secure Boot is blocking the module", 80)]);
        assert!(matches!(
            assess(&report, false),
            Some(Refusal::NeedsAHuman { .. })
        ));
    }

    #[test]
    fn passwordless_sudo_is_detected_without_triggering_a_prompt() {
        let allowed = ScriptedTransport::new(vec![ScriptedStep::ok("sudo -n true", "")]);
        assert!(has_passwordless_sudo(&allowed).unwrap());

        let denied = ScriptedTransport::new(vec![ScriptedStep::fails(
            "sudo -n true",
            1,
            "a password is required",
        )]);
        assert!(!has_passwordless_sudo(&denied).unwrap());
    }

    #[test]
    fn a_refusal_says_what_was_wrong_and_what_to_do() {
        let report = report_with(vec![finding("stale repository files", 70)]);
        let refusal = assess(&report, true).expect("refused");
        let message = describe_refusal(&refusal, "gpu-box", &report, 5);
        assert!(message.contains("stale repository files"), "{message}");
        assert!(message.contains("Health of gpu-box"), "{message}");
    }

    #[test]
    fn a_sudo_refusal_offers_the_manual_route() {
        let message = describe_refusal(
            &Refusal::NeedsPasswordlessSudo,
            "gpu-box",
            &report_with(vec![]),
            5,
        );
        assert!(message.contains("would hang rather than fail"), "{message}");
        assert!(
            message.contains("ssh gpu-box -- rocm bootstrap setup"),
            "{message}"
        );
    }

    #[test]
    fn the_sdk_is_not_attempted_when_the_driver_fails() {
        // Stacking a second install onto a failed one leaves a machine in a
        // state neither step can describe.
        let transport = ScriptedTransport::new(vec![
            ScriptedStep::fails("install driver --yes", 1, "no supported GPU"),
            ScriptedStep::ok("install sdk --yes", ""),
        ]);
        let error = install(&transport, "gpu-box", "rocm")
            .unwrap_err()
            .to_string();
        assert!(error.contains("Nothing further was attempted"), "{error}");
    }

    #[test]
    fn a_half_finished_install_says_so_rather_than_reporting_success() {
        let transport = ScriptedTransport::new(vec![
            ScriptedStep::ok("install driver --yes", "status: completed"),
            ScriptedStep::fails("install sdk --yes", 1, "no wheels for this platform"),
        ]);
        let error = install(&transport, "gpu-box", "rocm")
            .unwrap_err()
            .to_string();
        assert!(error.contains("part-way through"), "{error}");
    }

    #[test]
    fn a_required_reboot_is_surfaced_because_nobody_is_watching_the_machine() {
        assert!(mentions_reboot("execution:\n  reboot_required: true\n"));
        assert!(mentions_reboot("A reboot required before use"));
        assert!(!mentions_reboot("status: completed"));
    }
}
