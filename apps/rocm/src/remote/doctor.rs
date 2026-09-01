// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Checking a remote machine's health from here.
//!
//! This needs almost no new logic, because of how the local checks are already
//! built: gathering facts about a machine produces a plain serializable
//! snapshot, and scoring that snapshot against the failure-mode catalog reads
//! nothing but the snapshot. Neither half touches the local filesystem while
//! deciding. So the fetch happens on the remote and the scoring happens here,
//! with the same catalog the local command uses and no remote-side code at all.
//!
//! The snapshot is the contract, deliberately — not the human report, which
//! mixes in local paths, cache directories and engine inventory that describe
//! whichever machine rendered it. Reading that from a remote and printing it
//! here would produce a report that is subtly about the wrong computer.
//!
//! Fixes are rewritten to name the target. A command that repairs a machine you
//! are not sitting at is not a command you can paste, and printing it bare
//! invites running it against your own.

use std::fmt::Write as _;

use anyhow::{Context, Result};
use rocm_core::Examination;
use rocm_core::diagnose::{DiagnoseReport, diagnose};

use super::transport::Transport;

/// Fetch the remote's own view of itself and score it here.
pub(crate) fn examine_remote(
    transport: &dyn Transport,
    remote_cli: &str,
    symptom: Option<&str>,
) -> Result<(Examination, DiagnoseReport)> {
    let json = transport
        .run(&format!("{remote_cli} examine --json"))
        .context("could not read the remote machine's system state")?;
    let examination = parse_examination(&json)?;
    let report = diagnose(&examination, symptom.unwrap_or_default());
    Ok((examination, report))
}

/// Read an examination out of what the remote printed.
///
/// The remote wraps its examination in a document carrying extra rendering
/// fields; ignoring what we do not recognise is what lets a remote on a
/// different CLI version still be understood. A field we *do* need being absent
/// is the opposite case, and says so.
fn parse_examination(json: &str) -> Result<Examination> {
    serde_json::from_str::<Examination>(json.trim()).context(
        "could not understand the remote machine's system state. The remote CLI is \
         probably a different version than this one — update whichever is older.",
    )
}

/// Render the findings, with every fix aimed at the machine they are about.
pub(crate) fn render_report(target: &str, report: &DiagnoseReport, top: usize) -> String {
    let local = rocm_core::diagnose::render_report_text(report, top);
    let mut output = format!("Health of {target}\n\n");
    output.push_str(&redirect_fix_commands(&local, target));
    output
}

/// The exact prefixes the report renderer puts in front of a runnable command.
///
/// Matching on these rather than on "the line looks like a command" is what
/// keeps prose out. The report opens with lines such as
/// `rocm diagnose: no known misconfiguration matched.` — a sentence that begins
/// with a command name, and which a looser rule rewrites into nonsense.
///
/// Coupled to the renderer on purpose, and the tests below run the real
/// renderer so that a change to its layout fails here rather than silently
/// turning this back into a no-op.
const COMMAND_PREFIXES: &[&str] = &["$ ", "apply with: ", "verify after fix: "];

/// Aim every suggested command at the machine it repairs.
///
/// Two shapes, because the renderer uses both: a command on its own line behind
/// a known prefix, and a command quoted inline inside a sentence — "Next step:
/// run `rocm fix …`". The second is easy to miss and just as actionable; a user
/// copies what is between the backticks.
fn redirect_fix_commands(rendered: &str, target: &str) -> String {
    let mut output = rendered
        .lines()
        .map(|line| match split_command(line) {
            Some((lead, command)) => format!("{lead}ssh {target} -- {command}"),
            None => line.to_owned(),
        })
        .collect::<Vec<_>>()
        .join("\n");
    output.push('\n');
    redirect_quoted_commands(&output, target)
}

/// Rewrite every backtick-quoted `rocm …` so it names the target machine.
///
/// Matched by the quoting rather than by listing each phrase the renderer might
/// wrap around one. The phrases have changed before and will again; what does
/// not change is that a command a user is meant to run is put in backticks.
fn redirect_quoted_commands(rendered: &str, target: &str) -> String {
    let mut output = String::with_capacity(rendered.len());
    let mut rest = rendered;

    while let Some(open) = rest.find('`') {
        let (before, from_open) = rest.split_at(open);
        output.push_str(before);
        let after_open = &from_open[1..];
        let Some(close) = after_open.find('`') else {
            // An unpaired backtick is prose, not a quote. Leave the remainder be.
            output.push_str(from_open);
            return output;
        };
        let (quoted, remainder) = after_open.split_at(close);
        if quoted.starts_with("rocm ") {
            let _ = write!(output, "`ssh {target} -- {quoted}`");
        } else {
            let _ = write!(output, "`{quoted}`");
        }
        rest = &remainder[1..];
    }
    output.push_str(rest);
    output
}

/// Split a rendered line into everything up to and including its command
/// prefix, and the command itself.
fn split_command(line: &str) -> Option<(&str, &str)> {
    let trimmed = line.trim_start();
    let prefix = COMMAND_PREFIXES
        .iter()
        .find(|prefix| trimmed.starts_with(**prefix))?;
    let command = trimmed[prefix.len()..].trim();
    if command.is_empty() {
        return None;
    }
    let lead_len = line.len() - trimmed.len() + prefix.len();
    Some((&line[..lead_len], command))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::transport::{ScriptedStep, ScriptedTransport};

    /// The shape `rocm examine --json` prints: an examination, flattened
    /// together with a rendering summary that belongs to whoever printed it.
    fn examine_json() -> String {
        let examination = Examination::probe(rocm_core::FrameworkProbe::Skip);
        let mut value = serde_json::to_value(&examination).unwrap();
        value.as_object_mut().unwrap().insert(
            "summary".to_owned(),
            serde_json::json!({"default_engine": "vllm"}),
        );
        serde_json::to_string(&value).unwrap()
    }

    #[test]
    fn a_remote_examination_is_read_through_the_snapshot_not_the_human_report() {
        // The extra rendering fields describe the machine that printed them, so
        // they are ignored rather than adopted.
        let transport =
            ScriptedTransport::new(vec![ScriptedStep::ok("examine --json", &examine_json())]);
        let (examination, _) = examine_remote(&transport, "rocm", None).expect("examined");
        // Round-tripping the snapshot is the contract; the value itself is
        // whatever this machine happens to be.
        assert!(!examination.os_family.is_empty());
    }

    #[test]
    fn a_remote_on_another_version_says_so_instead_of_failing_obscurely() {
        let error = parse_examination(r#"{"unexpected": true}"#)
            .unwrap_err()
            .to_string();
        assert!(error.contains("different version"), "{error}");
    }

    #[test]
    fn the_container_fixture_is_something_this_code_can_actually_read() {
        // The stub that stands in for a remote CLI has to answer with a document
        // this deserializer accepts. An earlier version returned a two-field
        // fragment that looked plausible and could never have parsed — the
        // container lane never noticed, because nothing there ran this code.
        // Pinning it here means a drift in either direction fails a unit test.
        let fixture = include_str!("../../../../tests/remote-ssh/examination.json");
        let parsed =
            parse_examination(fixture).expect("the container stub's examination must deserialize");
        assert_eq!(parsed.os_family, "linux");
    }

    #[test]
    fn a_newer_remote_adding_fields_is_still_understood() {
        // Version skew between the machine driving and the machine driven is
        // normal; unknown fields must not break the read.
        let mut value: serde_json::Value = serde_json::from_str(&examine_json()).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("added_in_a_later_release".to_owned(), serde_json::json!(1));
        assert!(parse_examination(&value.to_string()).is_ok());
    }

    /// A report the renderer will render fully, built without asking this
    /// machine anything.
    ///
    /// Deliberately not `diagnose(&Examination::probe(..))`: on a host the
    /// catalog considers out of scope — WSL2, for one — that returns an
    /// out-of-scope report and the renderer short-circuits before printing a
    /// single command. Tests built that way pass or fail depending on the
    /// developer's machine, which is the opposite of what these need to prove.
    fn report(matched: Vec<rocm_core::diagnose::Diagnosis>) -> DiagnoseReport {
        DiagnoseReport {
            has_match: matched.iter().any(|d| d.score >= 50),
            matched,
            min_score_for_match: rocm_core::diagnose::MIN_SCORE_FOR_MATCH,
            high_confidence_threshold: rocm_core::diagnose::HIGH_CONFIDENCE,
            route_when_no_match: rocm_core::diagnose::Route {
                target: "rocm-cli".to_owned(),
                url: "https://example.invalid/issues".to_owned(),
            },
            out_of_scope: None,
        }
    }

    /// A report containing a fix, in the shape the renderer prints in full.
    ///
    /// The previous version of these tests invented an output shape the renderer
    /// never produces, so they passed while the rewriting matched nothing at all.
    fn real_report_with_a_fix() -> DiagnoseReport {
        use rocm_core::diagnose::{Diagnosis, Fix};
        report(vec![Diagnosis {
            id: "dkms-mismatch".to_owned(),
            title: "DKMS built against another kernel".to_owned(),
            score: 90,
            evidence: vec!["dkms status reports a stale build".to_owned()],
            fix: Some(Fix {
                summary: "rebuild the module".to_owned(),
                commands: vec!["sudo dkms autoinstall".to_owned()],
                needs_sudo: true,
                needs_reboot: true,
                fix_id: "dkms-mismatch".to_owned(),
                verify: "rocm doctor".to_owned(),
                ..Fix::default()
            }),
        }])
    }

    #[test]
    fn every_command_the_real_renderer_emits_is_aimed_at_the_remote() {
        let report = real_report_with_a_fix();
        let rendered = render_report("gpu-box", &report, 5);

        // The three shapes the renderer actually produces: a fix command, the
        // handle that applies it, and the check to run afterwards. Each is a
        // command a user would otherwise paste into their own terminal.
        assert!(
            rendered.contains("$ ssh gpu-box -- sudo dkms autoinstall"),
            "fix command not redirected:\n{rendered}"
        );
        assert!(
            rendered.contains("apply with: ssh gpu-box -- rocm fix dkms-mismatch"),
            "apply-with command not redirected:\n{rendered}"
        );
        assert!(
            rendered.contains("verify after fix: ssh gpu-box -- rocm doctor"),
            "verify command not redirected:\n{rendered}"
        );
    }

    #[test]
    fn the_rewriting_is_not_silently_a_no_op() {
        // The failure this guards against is the one that already happened: the
        // renderer's layout and this module's expectations drifted apart, and
        // nothing noticed because the tests supplied their own input. If the
        // renderer stops emitting these prefixes, this fails.
        let local = rocm_core::diagnose::render_report_text(&real_report_with_a_fix(), 5);
        let redirected = redirect_fix_commands(&local, "gpu-box");
        assert_ne!(
            local, redirected,
            "no command in a real report was recognised:\n{local}"
        );
    }

    #[test]
    fn prose_that_opens_with_a_command_name_is_left_intact() {
        // The report's own headers start with `rocm diagnose: …`. A rule that
        // recognised commands by their first word would turn each into
        // `ssh gpu-box -- rocm diagnose: no known misconfiguration matched.`
        let rendered = render_report("gpu-box", &report(vec![]), 5);
        assert!(
            !rendered.contains("ssh gpu-box -- rocm diagnose:"),
            "a sentence was rewritten as a command:\n{rendered}"
        );
    }

    #[test]
    fn a_command_quoted_inside_a_sentence_is_redirected_too() {
        // The renderer closes with "Next step: run `rocm fix <id>`." — no prefix,
        // so a line-prefix rule misses it entirely, and it is exactly the line a
        // user acts on.
        let rendered = render_report("gpu-box", &real_report_with_a_fix(), 5);
        assert!(
            rendered.contains("run `ssh gpu-box -- rocm fix dkms-mismatch`"),
            "the closing instruction still points at the local machine:\n{rendered}"
        );
        assert!(
            !rendered.contains("run `rocm fix"),
            "no bare local command should remain:\n{rendered}"
        );
    }

    #[test]
    fn a_below_threshold_report_redirects_its_closing_advice_as_well() {
        // The other trailing branch, reached when nothing clears the confidence
        // threshold. It names a command too.
        let mut low = real_report_with_a_fix();
        low.matched[0].score = 40;
        low.has_match = false;
        let rendered = render_report("gpu-box", &low, 5);
        assert!(
            !rendered.contains("run `rocm fix"),
            "the low-confidence branch still points locally:\n{rendered}"
        );
    }

    #[test]
    fn ordinary_quoted_text_is_left_alone() {
        // Only commands get redirected; backticks around anything else stay put.
        let left = redirect_quoted_commands("see the `apply with:` line and `/dev/kfd`", "gpu-box");
        assert_eq!(left, "see the `apply with:` line and `/dev/kfd`");
        // An unpaired backtick is prose, not a quote.
        assert_eq!(
            redirect_quoted_commands("a ` stray tick", "gpu-box"),
            "a ` stray tick"
        );
    }

    #[test]
    fn the_quoted_examine_command_asks_for_the_remote_machines_state() {
        // Run locally it reports the wrong computer, and the user never learns
        // why the answer looked irrelevant.
        let rendered = render_report("gpu-box", &report(vec![]), 5);
        assert!(
            rendered.contains("`ssh gpu-box -- rocm examine --json`"),
            "{rendered}"
        );
    }

    #[test]
    fn indentation_survives_so_the_report_still_reads_as_one() {
        let rendered = redirect_fix_commands("     $ sudo dkms autoinstall\n", "gpu-box");
        assert!(
            rendered.starts_with("     $ ssh gpu-box -- "),
            "{rendered:?}"
        );
    }

    #[test]
    fn a_prefix_with_nothing_after_it_is_not_a_command() {
        assert_eq!(
            redirect_fix_commands("   apply with: \n", "gpu-box").trim_end(),
            "   apply with:"
        );
    }

    #[test]
    fn the_report_names_the_machine_it_describes() {
        // Without this the output is indistinguishable from a local report, and
        // acting on it means fixing the wrong computer.
        let rendered = render_report("gpu-box", &report(vec![]), 5);
        assert!(rendered.starts_with("Health of gpu-box"), "{rendered}");
    }
}
