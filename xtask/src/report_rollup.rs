// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Count received Doctor reports: `cargo xtask report-rollup <paths>`.
//!
//! A maintainer exports the report mailbox to `.eml` files and runs this over
//! them. It reads, it counts, and it prints; it sends nothing and files
//! nothing. Promoting a group to an issue is a person's decision, made after
//! reading this.
//!
//! Every input lands in exactly one bucket, and every bucket is printed: a
//! report counted in the rollup, a message that is not a report, a duplicate
//! export of a message already counted, or a file in a format this does not
//! read. An input that disappeared without a trace would make the counts look
//! complete while being wrong.
//!
//! What each output may carry from the input differs, on purpose. Nothing from
//! a report's own mail headers reaches any output; the adapter drops it while
//! parsing. `--json` carries counts and group values, and `read_report` only
//! accepts group values from the CLI's own vocabulary; a report holding any
//! other value reads as unread. No free text from the input reaches `--json`,
//! so it is safe to pass on. The text output names the files it could not read, and
//! `--verbose` adds the file name and subject of each message that is not a
//! report. Those are the maintainer's own inputs, needed to find what was
//! skipped, and a subject or a file name can name a person, so that output is
//! for the maintainer's terminal, not for passing on.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rocm_core::{ReadOutcome, ReportRollup};
use serde_json::json;

use crate::report_mailbox::{self, Message};

/// What was read, beside what it counted to.
#[derive(Debug, Default)]
struct Collected {
    outcomes: Vec<ReadOutcome>,
    files: usize,
    /// Messages that are not reports, by file name and subject. Printed only
    /// with `--verbose`, and never in `--json`.
    not_reports: Vec<(String, String)>,
    duplicates: usize,
    /// Reports with no `Message-ID`, which cannot be deduplicated: the same
    /// one exported twice is counted twice. Said, so the total is read with it.
    no_message_id: usize,
    /// Reports that arrived as a forward. Counted, not removed: a forward
    /// carries a new `Message-ID`, so one also received directly is counted
    /// twice, and nothing in a report can tell the two apart.
    forwards: usize,
    unsupported: Vec<String>,
}

impl Collected {
    fn render_text(&self, rollup: &ReportRollup, verbose: bool) -> String {
        let mut out = rollup.render_text();
        let _ = writeln!(
            out,
            "\nInput: {} files read; {} reports; {} not a report; {} duplicate exports removed; \
             {} unsupported.",
            self.files,
            self.outcomes.len(),
            self.not_reports.len(),
            self.duplicates,
            self.unsupported.len(),
        );
        if self.forwards > 0 {
            let _ = writeln!(
                out,
                "{} of the reports arrived as forwards. A forward of a report that was also \
                 received directly is counted twice.",
                self.forwards
            );
        }
        if self.no_message_id > 0 {
            let _ = writeln!(
                out,
                "{} of the reports had no Message-ID and could not be deduplicated. One exported \
                 twice is counted twice.",
                self.no_message_id
            );
        }
        if verbose {
            for (file, subject) in &self.not_reports {
                let _ = writeln!(out, "  not a report: {file}: {subject}");
            }
        }
        for file in &self.unsupported {
            let _ = writeln!(out, "  unsupported, not read: {file}");
        }
        out
    }

    fn render_json(&self, rollup: &ReportRollup) -> serde_json::Value {
        json!({
            "input": {
                "files": self.files,
                "reports": self.outcomes.len(),
                "not_a_report": self.not_reports.len(),
                "duplicates": self.duplicates,
                "forwards": self.forwards,
                "no_message_id": self.no_message_id,
                // A count, not the names: `--json` carries no free text from the input.
                "unsupported": self.unsupported.len(),
            },
            "rollup": rollup.to_json(),
        })
    }
}

/// Count the reports in `paths` and print the rollup.
///
/// # Errors
/// When a path cannot be read, or when nothing readable was given at all: an
/// empty input must never look like a mailbox that received no reports.
pub fn run(paths: &[PathBuf], json: bool, verbose: bool) -> Result<()> {
    let collected = collect(paths)?;
    let rollup = ReportRollup::of(&collected.outcomes);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&collected.render_json(&rollup))?
        );
    } else {
        print!("{}", collected.render_text(&rollup, verbose));
    }
    Ok(())
}

fn collect(paths: &[PathBuf]) -> Result<Collected> {
    let mut collected = Collected::default();
    let mut files = Vec::new();
    for path in paths {
        if path.is_dir() {
            let mut entries: Vec<PathBuf> = fs::read_dir(path)
                .with_context(|| format!("failed to read {}", path.display()))?
                .map(|entry| entry.map(|entry| entry.path()))
                .collect::<std::io::Result<_>>()
                .with_context(|| format!("failed to read {}", path.display()))?;
            entries.sort();
            // Read one level deep. A folder inside is named rather than
            // dropped, because an export with folders would otherwise print a
            // plausible total that is too low.
            for entry in entries {
                if entry.is_dir() {
                    collected.unsupported.push(format!(
                        "{}/ (a directory; only one level is read)",
                        file_name(&entry)
                    ));
                } else {
                    files.push(entry);
                }
            }
        } else {
            files.push(path.clone());
        }
    }

    let mut seen = HashSet::new();
    for file in &files {
        let name = file_name(file);
        let extension = file
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase());
        match extension.as_deref() {
            Some("eml") => {
                let message = report_mailbox::parse_eml(&read(file)?);
                collected.files += 1;
                add_message(&mut collected, &mut seen, &name, &message);
            }
            Some("json") => {
                let body = String::from_utf8_lossy(&read(file)?).into_owned();
                collected.files += 1;
                collected.outcomes.push(report_mailbox::read_body(&body));
            }
            _ => collected.unsupported.push(name),
        }
    }
    if collected.files == 0 {
        bail!(
            "no .eml or .json files were found in the given paths, so there is nothing to count \
             (export the mailbox as .eml files; see docs/testing.md, \"Counting received reports\")"
        );
    }
    Ok(collected)
}

fn add_message(
    collected: &mut Collected,
    seen: &mut HashSet<String>,
    name: &str,
    message: &Message,
) {
    // Duplicates first, for every message: the same reply exported from two
    // inboxes is one reply, just as the same report is one report.
    if let Some(digest) = &message.id_digest
        && !seen.insert(digest.clone())
    {
        collected.duplicates += 1;
        return;
    }
    if !message.is_report() {
        collected
            .not_reports
            .push((name.to_owned(), message.subject.clone()));
        return;
    }
    if message.id_digest.is_none() {
        collected.no_message_id += 1;
    }
    if message.is_forward() {
        collected.forwards += 1;
    }
    // A report mail with no text part is still a report: it is counted as
    // unread rather than dropped.
    collected.outcomes.push(report_mailbox::read_body(
        message.body.as_deref().unwrap_or_default(),
    ));
}

fn read(file: &Path) -> Result<Vec<u8>> {
    fs::read(file).with_context(|| format!("failed to read {}", file.display()))
}

fn file_name(file: &Path) -> String {
    file.file_name().map_or_else(
        || file.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocm_core::{REPORT_SCHEMA_VERSION, Report};

    fn report(entry: &str) -> Report {
        Report {
            schema: REPORT_SCHEMA_VERSION,
            architecture: "gfx942".to_owned(),
            architecture_matrix: rocm_core::APPROVED_ARCHITECTURES_SOURCE.to_owned(),
            entry: entry.to_owned(),
            os_family: "linux".to_owned(),
            os_major: "22".to_owned(),
            distro: "ubuntu".to_owned(),
            rocm: "7.1".to_owned(),
            engine: "pytorch".to_owned(),
            engine_version: "2.5".to_owned(),
            cli_version: "0.1.0".to_owned(),
            fix_offered: true,
        }
    }

    fn percent_decode(value: &str) -> String {
        let bytes = value.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'%'
                && let Some(byte) = std::str::from_utf8(&bytes[i + 1..i + 3])
                    .ok()
                    .and_then(|hex| u8::from_str_radix(hex, 16).ok())
            {
                out.push(byte);
                i += 3;
                continue;
            }
            out.push(bytes[i]);
            i += 1;
        }
        String::from_utf8(out).expect("the link is UTF-8")
    }

    /// The mail a user's client would send from the link the CLI built, so the
    /// writer and this reader are checked against each other rather than
    /// against a hand-written copy of the format.
    fn mail_from_link(report: &Report, message_id: &str) -> String {
        let link = rocm_core::report_delivery::report_url(report);
        let query = link.split_once('?').expect("a mailto link has a query").1;
        let field = |name: &str| {
            query
                .split('&')
                .find_map(|pair| pair.strip_prefix(&format!("{name}=")))
                .map(percent_decode)
                .expect("the link carries the field")
        };
        format!(
            "From: Someone <someone@example.com>\r\nMessage-ID: <{message_id}@example.com>\r\n\
             Subject: {}\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n{}\r\n",
            field("subject"),
            field("body").replace('\n', "\r\n"),
        )
    }

    #[test]
    fn mail_built_from_the_cli_link_is_counted_and_every_other_input_is_accounted_for() {
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, content: &str| fs::write(dir.path().join(name), content).unwrap();
        write("a.eml", &mail_from_link(&report("fix-6-path"), "a"));
        write("b.eml", &mail_from_link(&report("fix-6-path"), "b"));
        write(
            "c.eml",
            &mail_from_link(&report(rocm_core::report::UNRECOGNISED), "c"),
        );
        // The same message exported twice.
        write("a-copy.eml", &mail_from_link(&report("fix-6-path"), "a"));
        // A reply quoting a report.
        write(
            "reply.eml",
            "Subject: Re: [rocm-doctor] fix-6-path on gfx942 / ubuntu-22\r\n\r\n> {}\r\n",
        );
        // A report from a newer CLI.
        write(
            "newer.json",
            &format!(
                r#"{{"schema": {}, "entry": "x"}}"#,
                REPORT_SCHEMA_VERSION + 1
            ),
        );
        write("old.msg", "binary");

        let collected = collect(&[dir.path().to_path_buf()]).unwrap();
        let rollup = ReportRollup::of(&collected.outcomes);

        assert_eq!(collected.files, 6);
        assert_eq!(collected.duplicates, 1);
        assert_eq!(collected.not_reports.len(), 1);
        assert_eq!(collected.unsupported, ["old.msg"]);
        assert_eq!(rollup.understood.count, 3);
        assert_eq!(rollup.recognised(), 2);
        assert_eq!(rollup.unrecognised(), 1);
        assert_eq!(rollup.unread[&(REPORT_SCHEMA_VERSION + 1)], 1);

        let text = collected.render_text(&rollup, true);
        assert!(text.contains("unsupported, not read: old.msg"), "{text}");
        assert!(text.contains("1 duplicate exports removed"), "{text}");
        let value = collected.render_json(&rollup);
        assert_eq!(value["input"]["reports"], 4);
        assert_eq!(value["rollup"]["total"], 4);
    }

    /// Nothing from a report's mail headers or signature reaches any output,
    /// `--verbose` included.
    #[test]
    fn neither_output_carries_the_sender() {
        let dir = tempfile::tempdir().unwrap();
        let mail = mail_from_link(&report("fix-6-path"), "id-marker")
            .replace("someone@example.com", "sender-marker@example.com")
            .replacen(
                "Subject:",
                "Received: from host-marker.example.com\r\nX-Mailer: Mailer-Marker\r\nSubject:",
                1,
            )
            .replace("\r\n}\r\n", "\r\n}\r\n\r\n--\r\nSignature-Marker\r\n");
        fs::write(dir.path().join("a.eml"), mail).unwrap();

        let collected = collect(&[dir.path().to_path_buf()]).unwrap();
        let rollup = ReportRollup::of(&collected.outcomes);
        assert_eq!(rollup.understood.count, 1);

        let outputs = format!(
            "{}{}",
            collected.render_text(&rollup, true),
            collected.render_json(&rollup)
        );
        for marker in [
            "sender-marker",
            "host-marker",
            "Mailer-Marker",
            "id-marker",
            "Signature-Marker",
        ] {
            assert!(
                !outputs.contains(marker),
                "{marker} reached the output:\n{outputs}"
            );
        }
    }

    /// The input's own text, file names and the subjects of non-reports, stays
    /// out of `--json` entirely and out of the text output unless `--verbose`
    /// asks for it. Only the names of unreadable files are always printed,
    /// because a maintainer cannot act on a file without its name.
    #[test]
    fn input_text_reaches_only_the_outputs_documented_to_carry_it() {
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, content: &str| fs::write(dir.path().join(name), content).unwrap();
        write(
            "report-file-marker.eml",
            &mail_from_link(&report("fix-6-path"), "a"),
        );
        write(
            "reply-file-marker.eml",
            "Subject: Re: question from Subject-Marker\r\n\r\nhello\r\n",
        );
        write("unsupported-file-marker.msg", "binary");

        let collected = collect(&[dir.path().to_path_buf()]).unwrap();
        let rollup = ReportRollup::of(&collected.outcomes);
        let json = collected.render_json(&rollup).to_string();
        let text = collected.render_text(&rollup, false);
        let verbose = collected.render_text(&rollup, true);

        for marker in [
            "report-file-marker",
            "reply-file-marker",
            "Subject-Marker",
            "unsupported-file-marker",
        ] {
            assert!(!json.contains(marker), "{marker} reached --json:\n{json}");
        }
        for marker in ["report-file-marker", "reply-file-marker", "Subject-Marker"] {
            assert!(
                !text.contains(marker),
                "{marker} reached the default text:\n{text}"
            );
        }
        assert!(
            text.contains("unsupported-file-marker"),
            "an unreadable file must be named:\n{text}"
        );
        assert!(verbose.contains("reply-file-marker") && verbose.contains("Subject-Marker"));
        // A report's own file name is never printed, verbose or not.
        assert!(!verbose.contains("report-file-marker"), "{verbose}");
    }

    /// Text a forged report puts in any of its fields reaches neither output.
    ///
    /// The report address takes mail from anyone. The reader accepts only
    /// values this CLI writes, so a forged field makes the report unread, and
    /// a field the rollup never prints stays unprinted either way.
    #[test]
    fn a_marker_in_any_report_field_reaches_neither_output() {
        let genuine = serde_json::to_value(report("fix-6-path")).unwrap();
        let fields: Vec<String> = genuine.as_object().unwrap().keys().cloned().collect();
        assert_eq!(
            fields.len(),
            12,
            "every Report field is covered: {fields:?}"
        );
        let marker = "Jane-Doe-Marker\u{1b}[31m";
        for field in fields
            .iter()
            .filter(|f| *f != "schema" && *f != "fix_offered")
        {
            let dir = tempfile::tempdir().unwrap();
            let mut forged = genuine.clone();
            forged[field.as_str()] = serde_json::Value::String(marker.to_owned());
            fs::write(dir.path().join("forged.json"), forged.to_string()).unwrap();

            let collected = collect(&[dir.path().to_path_buf()]).unwrap();
            let rollup = ReportRollup::of(&collected.outcomes);
            let outputs = format!(
                "{}{}",
                collected.render_text(&rollup, true),
                collected.render_json(&rollup)
            );
            assert!(
                !outputs.contains("Jane-Doe-Marker") && !outputs.contains('\u{1b}'),
                "a forged `{field}` reached an output:\n{outputs}"
            );
        }
    }

    /// The same reply exported from two inboxes is listed once, not twice.
    #[test]
    fn a_non_report_exported_twice_is_one_message() {
        let dir = tempfile::tempdir().unwrap();
        let reply = "Message-ID: <r@example.com>\r\nSubject: Re: [rocm-doctor] x\r\n\r\nthanks\r\n";
        fs::write(dir.path().join("a.eml"), reply).unwrap();
        fs::write(dir.path().join("b.eml"), reply).unwrap();
        fs::write(
            dir.path().join("c.eml"),
            mail_from_link(&report("fix-6-path"), "c"),
        )
        .unwrap();

        let collected = collect(&[dir.path().to_path_buf()]).unwrap();
        assert_eq!(collected.not_reports.len(), 1);
        assert_eq!(collected.duplicates, 1);
        assert_eq!(collected.outcomes.len(), 1);
    }

    /// A report without a `Message-ID` cannot be deduplicated, and the output
    /// says how many there were rather than double counting in silence.
    #[test]
    fn reports_without_a_message_id_are_counted_as_such() {
        let dir = tempfile::tempdir().unwrap();
        let without_id = mail_from_link(&report("fix-6-path"), "x")
            .lines()
            .filter(|line| !line.starts_with("Message-ID:"))
            .collect::<Vec<_>>()
            .join("\r\n");
        fs::write(dir.path().join("a.eml"), &without_id).unwrap();
        fs::write(dir.path().join("a-copy.eml"), &without_id).unwrap();

        let collected = collect(&[dir.path().to_path_buf()]).unwrap();
        let rollup = ReportRollup::of(&collected.outcomes);
        assert_eq!(
            rollup.understood.count, 2,
            "no id, so nothing to deduplicate on"
        );
        assert_eq!(collected.no_message_id, 2);
        let text = collected.render_text(&rollup, false);
        assert!(
            text.contains("2 of the reports had no Message-ID"),
            "{text}"
        );
        assert_eq!(collected.render_json(&rollup)["input"]["no_message_id"], 2);
    }

    /// A folder inside the export is named, not silently dropped, and a
    /// forwarded report is counted and said to be one.
    #[test]
    fn folders_are_named_and_forwards_are_counted_visibly() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("a.eml"),
            mail_from_link(&report("fix-6-path"), "a"),
        )
        .unwrap();
        let forwarded =
            mail_from_link(&report("fix-6-path"), "b").replace("Subject: ", "Subject: Fw: ");
        fs::write(dir.path().join("b.eml"), forwarded).unwrap();
        fs::create_dir(dir.path().join("Archive")).unwrap();
        fs::write(
            dir.path().join("Archive").join("c.eml"),
            mail_from_link(&report("fix-6-path"), "c"),
        )
        .unwrap();

        let collected = collect(&[dir.path().to_path_buf()]).unwrap();
        let rollup = ReportRollup::of(&collected.outcomes);

        assert_eq!(
            rollup.understood.count, 2,
            "the folder is not read, and the forward is"
        );
        assert_eq!(collected.forwards, 1);
        assert_eq!(collected.unsupported.len(), 1);
        let text = collected.render_text(&rollup, false);
        assert!(text.contains("Archive/"), "{text}");
        assert!(
            text.contains("1 of the reports arrived as forwards"),
            "{text}"
        );
        assert_eq!(collected.render_json(&rollup)["input"]["forwards"], 1);
    }

    #[test]
    fn an_input_with_nothing_readable_is_an_error_not_an_empty_count() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("old.msg"), "binary").unwrap();
        let error = collect(&[dir.path().to_path_buf()]).unwrap_err();
        assert!(error.to_string().contains("nothing to count"), "{error}");
        assert!(collect(&[dir.path().join("missing.eml")]).is_err());
    }
}
