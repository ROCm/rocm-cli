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
//! Two guards, both of them about a human not being present:
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
//! And privilege escalation is checked before it is needed. The driver install
//! runs commands through `sudo`, while the control channel refuses to answer
//! prompts by design — so a machine asking for a password does not fail, it
//! hangs. Establishing that sudo is passwordless first turns the most likely
//! real-world failure into a sentence instead of a stall.

use anyhow::{Result, bail};
use rocm_core::diagnose::DiagnoseReport;

use super::doctor;
use super::transport::Transport;

/// Why an unattended install was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// The catalog recognised something needing a person.
    NeedsAHuman { findings: Vec<String> },
    /// Privileged commands would block on a password prompt.
    NeedsPasswordlessSudo,
}

/// Decide whether ROCm may be installed on this machine unattended.
///
/// Split from the doing so the judgement is testable on its own — the part
/// worth being sure about is what gets refused, not what gets run.
pub(crate) fn assess(report: &DiagnoseReport, passwordless_sudo: bool) -> Option<Refusal> {
    let findings = recognised_problems(report);
    if !findings.is_empty() {
        return Some(Refusal::NeedsAHuman { findings });
    }
    if !passwordless_sudo {
        return Some(Refusal::NeedsPasswordlessSudo);
    }
    None
}

/// Catalog findings strong enough to be treated as real.
///
/// Weak signals are excluded deliberately: several checks open with a low score
/// for a situation that is merely *potentially* relevant, and treating those as
/// blockers would refuse installs on healthy machines until nobody trusted the
/// refusal.
fn recognised_problems(report: &DiagnoseReport) -> Vec<String> {
    report
        .matched
        .iter()
        .filter(|diagnosis| diagnosis.score >= rocm_core::diagnose::MIN_SCORE_FOR_MATCH)
        .map(|diagnosis| diagnosis.title.clone())
        .collect()
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
        Refusal::NeedsAHuman { findings } => format!(
            "{target} is not in a state that can be set up unattended.\n\n\
             What was found:\n{}\n\n\
             These are the situations the setup wizard exists to walk a person through, \
             and it cannot do that over a connection with nobody watching. Resolve them \
             first — the suggested fixes are below — then run this again.\n\n{}",
            findings
                .iter()
                .map(|finding| format!("  - {finding}"))
                .collect::<Vec<_>>()
                .join("\n"),
            doctor::render_report(target, report, top)
        ),
        Refusal::NeedsPasswordlessSudo => format!(
            "installing ROCm on {target} needs administrator rights, and this connection \
             cannot answer a password prompt — it would hang rather than fail.\n\n\
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

    fn report_with(diagnoses: Vec<Diagnosis>) -> DiagnoseReport {
        let mut report = rocm_core::diagnose::diagnose(
            &rocm_core::Examination::probe(rocm_core::FrameworkProbe::Skip),
            "",
        );
        report.matched = diagnoses;
        report
    }

    fn finding(title: &str, score: i32) -> Diagnosis {
        Diagnosis {
            id: "x".to_owned(),
            title: title.to_owned(),
            score,
            ..Diagnosis::default()
        }
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
                findings: vec!["DKMS built against another kernel".to_owned()]
            })
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
