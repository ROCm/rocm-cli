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
///
/// The aiming happens on the *report*, before rendering, not on the rendered
/// text afterwards. A `Fix.commands` entry is one command, but it is not one
/// line: the catalog holds backslash-continued `pip install` and `docker run`
/// entries spanning up to seven lines apiece. Rewriting line by line cut those
/// at the first newline and left the remainder as a bare local command — for the
/// `pip` entry, that silently dropped the `--index-url` and installed the wrong
/// wheels. Rewriting per entry keeps a command whole by construction.
pub(crate) fn render_report(target: &str, report: &DiagnoseReport, top: usize) -> String {
    let local = rocm_core::diagnose::render_report_text(&aim_at(target, report), top);
    let mut output = format!("Health of {target}\n\n");
    output.push_str(&redirect_quoted_commands(
        &redirect_apply_with(&local, target),
        target,
    ));
    output
}

/// Prefix of the one runnable command the renderer synthesises itself.
///
/// Everything else under a fix comes from a `Fix` field that [`aim_at`] can
/// rewrite before rendering. This line does not: the renderer builds it from
/// `fix_id` (`   apply with: rocm fix <id>`), so the only place to aim it is
/// after the fact. Kept deliberately narrow — one prefix, one synthesised shape
/// — rather than the general line-matching this module used to do, which is what
/// cut multi-line catalog commands in half.
const APPLY_WITH: &str = "apply with: ";

/// Aim the renderer's own `apply with:` line at the target machine.
fn redirect_apply_with(rendered: &str, target: &str) -> String {
    let mut output = rendered
        .lines()
        .map(|line| {
            let trimmed = line.trim_start();
            let Some(command) = trimmed.strip_prefix(APPLY_WITH) else {
                return line.to_owned();
            };
            let command = command.trim();
            if command.is_empty() {
                return line.to_owned();
            }
            let lead = &line[..line.len() - trimmed.len() + APPLY_WITH.len()];
            format!("{lead}{}", remote_invocation(target, command))
        })
        .collect::<Vec<_>>()
        .join("\n");
    output.push('\n');
    output
}

/// Copy `report` with every runnable command aimed at `target`.
fn aim_at(target: &str, report: &DiagnoseReport) -> DiagnoseReport {
    let mut aimed = report.clone();
    for diagnosis in &mut aimed.matched {
        let Some(fix) = diagnosis.fix.as_mut() else {
            continue;
        };
        // A sequence that only works as a sequence cannot be split across
        // separate ssh invocations, each with its own shell and no shared state.
        // Rewriting it would hand the user commands that look runnable and are
        // not, which is worse than saying so.
        if is_stateful_sequence(fix) {
            fix.notes.push(format!(
                "These steps share one shell session, so they cannot be run over separate \
                 connections. Open a session first with `ssh {target}`, then run them there."
            ));
            continue;
        }
        fix.commands = join_continuations(&fix.commands)
            .into_iter()
            .map(|command| {
                if is_comment(&command) {
                    command
                } else {
                    remote_invocation(target, &command)
                }
            })
            .collect();
        if !fix.verify.is_empty() {
            fix.verify = remote_invocation(target, &fix.verify);
        }
    }
    aimed
}

/// Merge backslash-continued entries into the single command they spell.
///
/// The catalog stores a multi-line command in two different shapes, and they are
/// not interchangeable. `fix-1-arch`'s `pip install` is **one** entry with an
/// embedded newline. `fix-10-container`'s `docker run` is **seven** entries,
/// each ending in a trailing `\`, meant to read as one continued invocation.
///
/// Rewriting per entry is right for the first shape and wrong for the second: it
/// turns one `docker run` into seven independent `ssh` calls, each carrying a
/// dangling backslash inside its quoting. Joining first makes both shapes the
/// same thing — one entry holding the whole command — before anything rewrites
/// it, and the far shell then reads the continuation exactly as a local one
/// would.
///
/// A line that does not end in `\` closes the group, so an entry that stands
/// alone passes through untouched.
fn join_continuations(commands: &[String]) -> Vec<String> {
    let mut joined: Vec<String> = Vec::new();
    let mut pending: Option<String> = None;

    for command in commands {
        let continues = command.trim_end().ends_with('\\');
        match pending.as_mut() {
            Some(open) => {
                open.push('\n');
                open.push_str(command);
            }
            None => pending = Some(command.clone()),
        }
        if !continues && let Some(complete) = pending.take() {
            joined.push(complete);
        }
    }
    // A trailing `\` with nothing after it is malformed, but dropping the text
    // would be worse than emitting it as it stands.
    if let Some(unterminated) = pending {
        joined.push(unterminated);
    }
    joined
}

/// Whether a catalog entry is prose rather than something to run.
///
/// The renderer prints every `commands` element behind `$ `, including the
/// `# Recommended: …` annotations the catalog uses to explain a choice. Wrapping
/// one in `ssh … -- '# …'` presents a comment as a command to run.
fn is_comment(command: &str) -> bool {
    command.trim_start().starts_with('#')
}

/// Whether a fix's steps depend on each other's shell state.
///
/// The catalog has one such entry today — a three-stage fix whose later steps
/// run *inside* a subshell the first one opens — and it labels its own steps.
/// Matching on that label rather than guessing at the commands keeps this
/// honest: an entry that stops saying so stops being treated as one.
fn is_stateful_sequence(fix: &rocm_core::diagnose::Fix) -> bool {
    fix.commands
        .iter()
        .any(|command| command.contains("step 2 of") || command.contains("INSIDE the subshell"))
}

/// The command a user should paste to run `command` on `target`.
///
/// The quoting is load-bearing. A suggested fix is copied into the user's own
/// shell, and an interpolated command splits across the two machines at its
/// first metacharacter: `ssh box -- echo x | sudo tee /etc/f` runs the `echo`
/// there and the privileged `tee` **here**, rewriting the operator's own system
/// while they believe they are repairing someone else's.
///
/// That is the common case, not an exotic one. Most commands in the fix catalog
/// contain a pipe, a `&&` or a redirect — `rocminfo | grep …`, `sudo apt update
/// 2>&1 | tail …`, `echo … | sudo tee …`.
///
/// One layer of quoting, not two, and no `sh -c`: the local shell strips the
/// quotes, `ssh` sends what is left as the command, and the *remote* login shell
/// is what parses the pipeline — on the machine it is meant to run on. Wrapping
/// it in a second layer would only make an already-correct line unreadable, and
/// these lines exist to be read and pasted.
fn remote_invocation(target: &str, command: &str) -> String {
    format!("ssh {target} -- {}", super::shell_quote(command))
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
            // Quoted the same way as a `$ ` line: these carry `#` comments and
            // shell operators too, and a backticked command is copied just as
            // readily as one on its own line.
            let _ = write!(output, "`{}`", remote_invocation(target, quoted));
        } else {
            let _ = write!(output, "`{quoted}`");
        }
        rest = &remainder[1..];
    }
    output.push_str(rest);
    output
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
    ///
    /// The commands are taken verbatim from the real catalog rather than
    /// invented, for the same class of reason one step in: the invented ones were
    /// all single words with no shell operators, while most catalog entries
    /// contain a pipe, a `&&` or a redirect. A fixture that cannot express the
    /// failing input cannot fail, which is how the unquoted rewriting survived
    /// five review rounds.
    fn real_report_with_a_fix() -> DiagnoseReport {
        use rocm_core::diagnose::{Diagnosis, Fix};
        report(vec![Diagnosis {
            id: "dkms-mismatch".to_owned(),
            title: "DKMS built against another kernel".to_owned(),
            score: 90,
            evidence: vec!["dkms status reports a stale build".to_owned()],
            fix: Some(Fix {
                summary: "rebuild the module".to_owned(),
                commands: vec![
                    "sudo dkms autoinstall".to_owned(),
                    // `fix-wsl-2-dxcore-missing`, verbatim. The privileged half
                    // is behind the pipe, which is what made the unquoted form
                    // rewrite the operator's own machine.
                    "echo /usr/lib/wsl/lib | sudo tee /etc/ld.so.conf.d/wsl.conf".to_owned(),
                ],
                needs_sudo: true,
                needs_reboot: true,
                fix_id: "dkms-mismatch".to_owned(),
                verify: "lsmod | grep amdgpu && rocminfo | head -n 5".to_owned(),
                ..Fix::default()
            }),
        }])
    }

    /// Paste `line` into a real shell, with `ssh` replaced by a stub, and report
    /// the command text `ssh` was handed.
    ///
    /// Asking a shell rather than inspecting the string is the whole point. What
    /// decides which machine a command runs on is what the *local* shell does
    /// with it before `ssh` ever sees it, and no assertion about the rendered
    /// text can observe that.
    fn what_ssh_would_send(line: &str) -> String {
        // Drop the destination and the `--` guard; what remains is the command
        // ssh transmits. `$*` rejoins it exactly as ssh does.
        let stub = r#"ssh() { shift 2; printf %s "$*"; }"#;
        let output = std::process::Command::new("sh")
            .arg("-c")
            .arg(format!("{stub}\n{line}"))
            .output()
            .expect("sh should run");
        String::from_utf8_lossy(&output.stdout).into_owned()
    }

    #[test]
    fn a_piped_fix_command_is_sent_whole_rather_than_split_across_two_machines() {
        // The defect this replaces: `ssh box -- echo x | sudo tee /etc/f` runs
        // the echo on the remote and the privileged tee *locally*. Nothing about
        // the rendered string reveals that — only running it through a shell.
        const PIPED: &str = "echo /usr/lib/wsl/lib | sudo tee /etc/ld.so.conf.d/wsl.conf";
        let rendered = render_report("gpu-box", &real_report_with_a_fix(), 5);
        let line = rendered
            .lines()
            .map(str::trim)
            .find(|line| line.starts_with("$ ") && line.contains("ld.so.conf.d"))
            .expect("the piped fix command must still be rendered")
            .trim_start_matches("$ ")
            .to_owned();

        assert_eq!(
            what_ssh_would_send(&line),
            PIPED,
            "ssh must be handed the whole pipeline; anything missing here is a \
             command the local machine ran instead: {line}"
        );

        // The same property for the `verify after fix:` line, which carries `&&`
        // and a second pipe and is copied just as readily.
        let verify = rendered
            .lines()
            .map(str::trim)
            .find(|line| line.starts_with("verify after fix: "))
            .expect("the verify command must still be rendered")
            .trim_start_matches("verify after fix: ")
            .to_owned();
        assert_eq!(
            what_ssh_would_send(&verify),
            "lsmod | grep amdgpu && rocminfo | head -n 5",
            "the verify command split across two machines: {verify}"
        );
    }

    #[test]
    fn every_command_the_real_renderer_emits_is_aimed_at_the_remote() {
        let report = real_report_with_a_fix();
        let rendered = render_report("gpu-box", &report, 5);

        // The three shapes the renderer actually produces: a fix command, the
        // handle that applies it, and the check to run afterwards. Each is a
        // command a user would otherwise paste into their own terminal.
        assert!(
            rendered.contains("$ ssh gpu-box -- 'sudo dkms autoinstall'"),
            "fix command not redirected:\n{rendered}"
        );
        assert!(
            rendered.contains("apply with: ssh gpu-box -- 'rocm fix dkms-mismatch'"),
            "apply-with command not redirected:\n{rendered}"
        );
        assert!(
            rendered.contains(
                "verify after fix: ssh gpu-box -- 'lsmod | grep amdgpu && rocminfo | head -n 5'"
            ),
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
        let redirected = render_report("gpu-box", &real_report_with_a_fix(), 5);
        assert!(
            !redirected.contains(&format!(
                "$ {}",
                local
                    .lines()
                    .find_map(|line| line.trim().strip_prefix("$ "))
                    .expect("a real report emits at least one command")
            )),
            "a command in a real report was left aimed at this machine:\n{redirected}"
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
            rendered.contains("run `ssh gpu-box -- 'rocm fix dkms-mismatch'`"),
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
            rendered.contains("`ssh gpu-box -- 'rocm examine --json'`"),
            "{rendered}"
        );
    }

    #[test]
    fn indentation_survives_so_the_report_still_reads_as_one() {
        let rendered = render_report("gpu-box", &real_report_with_a_fix(), 5);
        assert!(
            rendered
                .lines()
                .any(|line| line.starts_with("     $ ssh gpu-box -- ")),
            "{rendered}"
        );
    }

    #[test]
    fn a_comment_in_a_fix_is_not_presented_as_a_command_to_run() {
        // The catalog annotates its steps with `#` lines, and the renderer puts
        // every element behind `$ `. Wrapping one in ssh tells the user to run a
        // comment on another machine.
        use rocm_core::diagnose::{Diagnosis, Fix};
        let rendered = render_report(
            "gpu-box",
            &report(vec![Diagnosis {
                id: "annotated".to_owned(),
                title: "a fix that explains itself".to_owned(),
                score: 90,
                evidence: vec!["something".to_owned()],
                fix: Some(Fix {
                    summary: "do the thing".to_owned(),
                    commands: vec![
                        "# Recommended: the nightly wheels".to_owned(),
                        "sudo dkms autoinstall".to_owned(),
                    ],
                    fix_id: "annotated".to_owned(),
                    ..Fix::default()
                }),
            }]),
            5,
        );
        assert!(
            rendered.contains("$ # Recommended: the nightly wheels"),
            "the comment should be left as prose:\n{rendered}"
        );
        assert!(
            !rendered.contains("ssh gpu-box -- '#"),
            "a comment was presented as a command:\n{rendered}"
        );
        assert!(
            rendered.contains("$ ssh gpu-box -- 'sudo dkms autoinstall'"),
            "the real command must still be aimed at the remote:\n{rendered}"
        );
    }

    #[test]
    fn a_backslash_continued_command_stored_as_many_entries_becomes_one_invocation() {
        // `fix-10-container` stores one `docker run` as seven entries, each
        // ending in a trailing `\`. Rewriting per entry — correct for the
        // *other* multi-line shape — turns it into seven independent ssh calls,
        // each with a dangling backslash inside its quoting.
        use rocm_core::diagnose::{Diagnosis, Fix};
        let rendered = render_report(
            "gpu-box",
            &report(vec![Diagnosis {
                id: "container".to_owned(),
                title: "container missing devices".to_owned(),
                score: 90,
                evidence: vec!["no /dev/kfd in the container".to_owned()],
                fix: Some(Fix {
                    summary: "re-launch with the devices passed through".to_owned(),
                    commands: vec![
                        "# Docker / Podman flags AMD-recommends:".to_owned(),
                        "docker run --rm -it \\".to_owned(),
                        "  --device=/dev/kfd \\".to_owned(),
                        "  --group-add render \\".to_owned(),
                        "  rocm/pytorch:latest".to_owned(),
                    ],
                    fix_id: "container".to_owned(),
                    ..Fix::default()
                }),
            }]),
            5,
        );

        // Exactly one rewritten command line for the whole docker run, not four.
        // Counted over `$ `-prefixed lines only, so the `apply with:` and the
        // closing "Next step" line — both legitimately rewritten — do not count.
        assert_eq!(
            rendered
                .lines()
                .filter(|line| line.trim_start().starts_with("$ ssh gpu-box --"))
                .count(),
            1,
            "the docker run must be one ssh call, not one per entry:\n{rendered}"
        );
        // The comment stays prose.
        assert!(rendered.contains("$ # Docker / Podman flags"), "{rendered}");
        // And the far shell receives the whole continued command as one piece.
        let line = rendered
            .lines()
            .map(str::trim_start)
            .find(|line| line.starts_with("$ ssh gpu-box -- 'docker run"))
            .expect("the docker run must be rewritten as a single command")
            .trim_start_matches("$ ")
            .to_owned();
        // The remaining fragments follow inside the same quoting, so the line
        // the report prints is only the first physical line of one command.
        let whole = rendered
            .split_once("ssh gpu-box -- '")
            .expect("rewritten")
            .1
            .split_once("'\n")
            .map_or_else(String::new, |(body, _)| body.to_owned());
        for fragment in [
            "--device=/dev/kfd",
            "--group-add render",
            "rocm/pytorch:latest",
        ] {
            assert!(
                whole.contains(fragment),
                "`{fragment}` was split into its own ssh call:\n{rendered}"
            );
        }
        assert!(line.starts_with("ssh gpu-box -- 'docker run"), "{line}");
    }

    #[test]
    fn a_multi_line_fix_command_is_kept_whole_rather_than_cut_at_its_first_newline() {
        // `fix-1-arch` is one `commands` element spanning two lines via a
        // backslash continuation. Rewriting line by line aimed the first half at
        // the remote and left `--index-url …` behind as a bare local fragment —
        // so the user installed wheels from the default index.
        use rocm_core::diagnose::{Diagnosis, Fix};
        const CONTINUED: &str =
            "pip install --pre torch \\\n  --index-url https://example.invalid/rocm6.4";
        let rendered = render_report(
            "gpu-box",
            &report(vec![Diagnosis {
                id: "wheels".to_owned(),
                title: "wrong wheels".to_owned(),
                score: 90,
                evidence: vec!["something".to_owned()],
                fix: Some(Fix {
                    summary: "reinstall".to_owned(),
                    commands: vec![CONTINUED.to_owned()],
                    fix_id: "wheels".to_owned(),
                    ..Fix::default()
                }),
            }]),
            5,
        );
        // The index URL must be inside the quoting, not stranded after it.
        let quoted = rendered
            .split_once("ssh gpu-box -- '")
            .expect("the command is aimed at the remote")
            .1;
        let body = quoted
            .split_once("'\n")
            .map_or(quoted, |(body, _)| body)
            .to_owned();
        assert!(
            body.contains("--index-url https://example.invalid/rocm6.4"),
            "the continuation was cut off, so the remote would install the wrong \
             wheels:\n{rendered}"
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
