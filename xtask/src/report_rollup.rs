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
//! a report's own mail headers reaches any output, except the week derived from
//! its `Date`, which is what the trend shows; the adapter drops the rest while
//! parsing. `--json` carries counts and group values, and `read_report` only
//! accepts group values from the CLI's own vocabulary; a report holding any
//! other value reads as unread. No free text from the input reaches `--json`,
//! so it is safe to pass on. The text output names the files it could not read, and
//! `--verbose` adds the file name and subject of each message that is not a
//! report. Those are the maintainer's own inputs, needed to find what was
//! skipped, and a subject or a file name can name a person, so that output is
//! for the maintainer's terminal, not for passing on.
//!
//! The weekly trend places each report in the week its mail was sent. The mail's
//! `Date` is reduced to that week in the adapter, so the time of day, the zone
//! and the day never reach a count or an output. A report with no usable date is
//! counted and is in no week; the output says how many.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use rocm_core::{ReadOutcome, ReportRiseRule, ReportRollup, ReportTrendView, ReportWeek};
use serde_json::json;

use crate::report_mailbox::{self, Message};

/// What was read, beside what it counted to.
#[derive(Debug, Default)]
struct Collected {
    /// Each report with the week its mail was sent in, when that is known.
    outcomes: Vec<(Option<ReportWeek>, ReadOutcome)>,
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
    /// Count every report, each in its own week.
    fn rollup(&self) -> ReportRollup {
        let mut rollup = ReportRollup::default();
        for (week, outcome) in &self.outcomes {
            rollup.add_in_week(*week, outcome);
        }
        rollup
    }

    /// With the default view of the trend, which is what most tests want.
    #[cfg(test)]
    fn render_text(&self, rollup: &ReportRollup, verbose: bool) -> String {
        self.render_text_with(rollup, &ReportTrendView::default(), verbose)
    }

    fn render_text_with(
        &self,
        rollup: &ReportRollup,
        view: &ReportTrendView,
        verbose: bool,
    ) -> String {
        let mut out = rollup.render_text_with(view);
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

    /// With the default view of the trend, which is what most tests want.
    #[cfg(test)]
    fn render_json(&self, rollup: &ReportRollup) -> serde_json::Value {
        self.render_json_with(rollup, &ReportTrendView::default())
    }

    fn render_json_with(&self, rollup: &ReportRollup, view: &ReportTrendView) -> serde_json::Value {
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
            "rollup": rollup.to_json_with(view),
        })
    }
}

/// The command line of `report-rollup`.
///
/// Defined here, beside the code that reads it, and flattened into the command
/// in `main.rs`. The defaults are the library's own, so the CLI and the library
/// cannot disagree about them.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// `.eml` or `.json` files, or directories holding them (read one level
    /// deep).
    #[arg(required = true)]
    pub paths: Vec<PathBuf>,
    /// Print the rollup as JSON.
    #[arg(long)]
    pub json: bool,
    /// Also list each message skipped as not a report, by file and subject.
    #[arg(long)]
    pub verbose: bool,
    /// How many of the newest weeks the trend table shows (at least 1).
    #[arg(long, default_value_t = ReportTrendView::default().weeks)]
    pub weeks: usize,
    /// A week is a rise when its count for a known problem is at least this many
    /// times the average of the weeks before it (at least 2).
    #[arg(long, default_value_t = ReportRiseRule::default().factor)]
    pub rise_factor: usize,
    /// How many weeks before a week make up its average (at least 1). The first
    /// week of the export is never one of them, so a rise needs this plus two
    /// weeks of reports.
    #[arg(long, default_value_t = ReportRiseRule::default().baseline_weeks)]
    pub baseline_weeks: usize,
    /// The fewest reports a week needs to be called a rise.
    #[arg(long, default_value_t = ReportRiseRule::default().min_reports)]
    pub min_reports: usize,
}

impl Args {
    /// What the trend shows and what counts as a rise, as the flags set them.
    const fn view(&self) -> ReportTrendView {
        ReportTrendView {
            weeks: self.weeks,
            rise: ReportRiseRule {
                factor: self.rise_factor,
                baseline_weeks: self.baseline_weeks,
                min_reports: self.min_reports,
            },
        }
    }

    /// Refuse a flag that could not mean anything, each with its own message.
    fn check(&self) -> Result<()> {
        if self.rise_factor < 2 {
            bail!(
                "--rise-factor must be at least 2: a week at the same rate as before is not a rise"
            );
        }
        if self.baseline_weeks < 1 {
            bail!(
                "--baseline-weeks must be at least 1: a rise is judged against the weeks before it"
            );
        }
        if self.weeks < 1 {
            bail!("--weeks must be at least 1: the table needs a week to show");
        }
        Ok(())
    }
}

/// Count the reports in `args.paths` and print the rollup.
///
/// # Errors
/// See [`render`].
pub fn run(args: &Args) -> Result<()> {
    print!("{}", render(args, today())?);
    Ok(())
}

/// The rollup as it is printed, on the day `today` (days since 1970-01-01).
///
/// # Errors
/// When a flag could not mean anything, which is checked before anything is
/// read; when a path cannot be read; or when nothing readable was given at all
/// (an empty input must never look like a mailbox that received no reports).
fn render(args: &Args, today: i64) -> Result<String> {
    args.check()?;
    let view = args.view();
    let collected = collect(&args.paths, today)?;
    let rollup = collected.rollup();
    Ok(if args.json {
        let value = collected.render_json_with(&rollup, &view);
        format!("{}\n", serde_json::to_string_pretty(&value)?)
    } else {
        collected.render_text_with(&rollup, &view, args.verbose)
    })
}

/// Today, as days since 1970-01-01 in UTC. A clock before 1970 reads as day 0,
/// which makes every date look like the future, so every report reads as
/// undated: the output says so instead of placing reports in the wrong weeks.
fn today() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| {
            i64::try_from(since.as_secs() / 86_400).unwrap_or(0)
        })
}

fn collect(paths: &[PathBuf], today: i64) -> Result<Collected> {
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
                let message = report_mailbox::parse_eml(&read(file)?, today);
                collected.files += 1;
                add_message(&mut collected, &mut seen, &name, &message);
            }
            Some("json") => {
                let body = String::from_utf8_lossy(&read(file)?).into_owned();
                collected.files += 1;
                // A raw body has no mail around it, so no date.
                collected
                    .outcomes
                    .push((None, report_mailbox::read_body(&body)));
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
    collected.outcomes.push((
        message.week,
        report_mailbox::read_body(message.body.as_deref().unwrap_or_default()),
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

    /// 2027-01-15, as days since 1970-01-01: the day these tests run "on".
    const TODAY: i64 = 20_833;

    /// `today()` reads the real clock, which the other tests avoid by passing a
    /// day in. It must be a day number: not zero (a clock before 1970, which would
    /// make every date look like the future), and not seconds or hours, which a
    /// wrong divisor would give and which would switch off the "more than a day
    /// ahead" check.
    #[test]
    fn today_is_a_day_number_in_a_sane_range() {
        let day = today();
        // From the day the code that writes reports first landed, to 2100-01-01.
        assert!(
            (report_mailbox::EARLIEST_PLAUSIBLE_DAY..47_482).contains(&day),
            "today() = {day}"
        );
    }

    /// The flags as a command line gives them, parsed by the same definition the
    /// command uses.
    #[derive(clap::Parser)]
    struct TestCli {
        #[command(flatten)]
        args: Args,
    }

    fn args(flags: &[&str]) -> Args {
        use clap::Parser as _;
        TestCli::try_parse_from(std::iter::once("report-rollup").chain(flags.iter().copied()))
            .expect("the flags parse")
            .args
    }

    /// `mail` with a `Date` header, which `mail_from_link` does not write.
    fn dated(mail: &str, date: &str) -> String {
        mail.replacen("Subject:", &format!("Date: {date}\r\nSubject:"), 1)
    }

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

        let collected = collect(&[dir.path().to_path_buf()], TODAY).unwrap();
        let rollup = collected.rollup();

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

        let collected = collect(&[dir.path().to_path_buf()], TODAY).unwrap();
        let rollup = collected.rollup();
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

        let collected = collect(&[dir.path().to_path_buf()], TODAY).unwrap();
        let rollup = collected.rollup();
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

            let collected = collect(&[dir.path().to_path_buf()], TODAY).unwrap();
            let rollup = collected.rollup();
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

        let collected = collect(&[dir.path().to_path_buf()], TODAY).unwrap();
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

        let collected = collect(&[dir.path().to_path_buf()], TODAY).unwrap();
        let rollup = collected.rollup();
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

        let collected = collect(&[dir.path().to_path_buf()], TODAY).unwrap();
        let rollup = collected.rollup();

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

    /// Reports sent in six consecutive weeks for one known problem: 1 in the
    /// first, a partial week because the export began late in it, then four weeks
    /// of 4, then a week of 12, which is exactly three times the average.
    /// Written as real mail, from the link the CLI builds.
    fn six_weeks_with_a_rise(dir: &Path) {
        // The Thursday of each week, which is in the week of the Monday before it.
        // The first week starts on 2026-10-05, the earliest day a report can have
        // been sent.
        let thursdays = [
            "Thu, 08 Oct 2026 10:00:00 +0000",
            "Thu, 15 Oct 2026 10:00:00 +0000",
            "Thu, 22 Oct 2026 10:00:00 +0000",
            "Thu, 29 Oct 2026 10:00:00 +0000",
            "Thu, 05 Nov 2026 10:00:00 +0000",
            "Thu, 12 Nov 2026 10:00:00 +0000",
        ];
        for (week, date) in thursdays.iter().enumerate() {
            let (count, rocm) = match week {
                0 => (1, "7.1"),
                5 => (12, "7.2"),
                _ => (4, "7.1"),
            };
            for nth in 0..count {
                let report = Report {
                    rocm: rocm.to_owned(),
                    ..report("fix-6-path")
                };
                let mail = dated(&mail_from_link(&report, &format!("w{week}n{nth}")), date);
                fs::write(dir.join(format!("w{week}-{nth}.eml")), mail).unwrap();
            }
        }
    }

    /// The weekly trend, and the rise in it, from mail as the CLI sends it.
    #[test]
    fn reports_are_placed_in_the_week_they_were_sent_and_a_rise_is_found() {
        let dir = tempfile::tempdir().unwrap();
        six_weeks_with_a_rise(dir.path());

        let collected = collect(&[dir.path().to_path_buf()], TODAY).unwrap();
        let rollup = collected.rollup();
        assert_eq!(rollup.understood.count, 29);
        assert_eq!(
            rollup.understood.count - rollup.trend.undated,
            29,
            "all dated"
        );
        assert_eq!(rollup.trend.undated, 0);

        let rises = rollup.trend.rises(&ReportRiseRule::default());
        assert_eq!(rises.len(), 1, "{rises:?}");
        assert_eq!(rises[0].entry, "fix-6-path");
        assert_eq!(rises[0].week.to_string(), "2026-11-09");
        assert_eq!((rises[0].count, rises[0].baseline_reports), (12, 16));
        assert_eq!(rises[0].rocm, [("7.2".to_owned(), 12)]);

        let text = collected.render_text(&rollup, false);
        // The row in the trend table, not the one in the tree above it, which
        // also starts with the entry and holds its total.
        let trend = text.split("Weekly trend").nth(1).expect("a trend section");
        let row = trend.lines().find(|l| l.starts_with("fix-6-path")).unwrap();
        assert!(
            row.ends_with("     1     4     4     4     4    12"),
            "{row:?}"
        );
        assert!(
            text.contains(
                "fix-6-path: week of 2026-11-09, 12 reports against 4.0 a week; ROCm 7.2 (12)"
            ),
            "{text}"
        );
        let value = collected.render_json(&rollup);
        assert_eq!(value["rollup"]["trend"]["rises"][0]["week"], "2026-11-09");
        assert_eq!(
            value["rollup"]["trend"]["entries"][0]["counts"],
            json!([1, 4, 4, 4, 4, 12])
        );
    }

    /// A message's date reaches no output finer than its week: not the time of
    /// day, not the zone, and not the day of the month.
    #[test]
    fn a_message_date_reaches_no_output_finer_than_its_week() {
        let dir = tempfile::tempdir().unwrap();
        // Thursday 29 October, 23:59:59 at +05:30, which is 18:29:59 in UTC. The
        // week it falls in starts on Monday 26 October, and that Monday is the
        // one date that may appear.
        let mail = dated(
            &mail_from_link(&report("fix-6-path"), "x"),
            "Thu, 29 Oct 2026 23:59:59 +0530",
        );
        fs::write(dir.path().join("a.eml"), mail).unwrap();

        let collected = collect(&[dir.path().to_path_buf()], TODAY).unwrap();
        let rollup = collected.rollup();
        assert_eq!(
            rollup.understood.count - rollup.trend.undated,
            1,
            "the date must have been read"
        );

        let outputs = format!(
            "{}{}{collected:?}{rollup:?}",
            collected.render_text(&rollup, true),
            collected.render_json(&rollup)
        );
        assert!(
            outputs.contains("2026-10-26"),
            "the week is shown:\n{outputs}"
        );
        for finer in ["23:59", "18:29", "+0530", "29 Oct", "10-29", "Thu,"] {
            assert!(
                !outputs.contains(finer),
                "{finer} reached an output:\n{outputs}"
            );
        }
    }

    /// A report with no usable date is counted, is in no week, and the output
    /// says so. It is not a quiet week.
    #[test]
    fn a_report_with_no_usable_date_is_counted_and_said_to_be_undated() {
        let dir = tempfile::tempdir().unwrap();
        let mail = mail_from_link(&report("fix-6-path"), "a");
        fs::write(dir.path().join("no-date.eml"), &mail).unwrap();
        fs::write(
            dir.path().join("forged.eml"),
            dated(
                &mail_from_link(&report("fix-6-path"), "b"),
                "Thu, 29 Oct 2099 10:00:00 +0000",
            ),
        )
        .unwrap();
        fs::write(
            dir.path().join("raw.json"),
            serde_json::to_string(&report("fix-6-path")).unwrap(),
        )
        .unwrap();

        let collected = collect(&[dir.path().to_path_buf()], TODAY).unwrap();
        let rollup = collected.rollup();
        assert_eq!(rollup.understood.count, 3, "all three are counted");
        assert_eq!(rollup.trend.undated, 3);
        assert_eq!(rollup.understood.count - rollup.trend.undated, 0);
        let text = collected.render_text(&rollup, false);
        assert!(
            text.contains("none of the 3 reports carried a usable date"),
            "{text}"
        );
        assert!(!text.contains("Rises"), "no week, no rise:\n{text}");
    }

    /// Each flag reaches its own field. The values are all different, so two
    /// flags swapped on the way to the view would show.
    #[test]
    fn each_flag_reaches_its_own_part_of_the_view() {
        let parsed = args(&[
            "x",
            "--json",
            "--verbose",
            "--weeks",
            "3",
            "--rise-factor",
            "2",
            "--baseline-weeks",
            "5",
            "--min-reports",
            "7",
        ]);
        assert!(parsed.json && parsed.verbose);
        assert_eq!(
            parsed.view(),
            ReportTrendView {
                weeks: 3,
                rise: ReportRiseRule {
                    factor: 2,
                    baseline_weeks: 5,
                    min_reports: 7,
                },
            }
        );
    }

    /// With no flags the view is the library's own default: the CLI reads its
    /// defaults from there, so the two cannot drift.
    #[test]
    fn without_flags_the_view_is_the_librarys_default() {
        let parsed = args(&["x"]);
        assert!(!parsed.json && !parsed.verbose);
        assert_eq!(parsed.view(), ReportTrendView::default());
    }

    /// A flag that could not mean anything is refused, each with its own message,
    /// and before anything is read.
    #[test]
    fn flags_that_could_not_mean_anything_are_refused_before_anything_is_read() {
        let dir = tempfile::tempdir().unwrap();
        // A path that cannot be read. If a flag were checked only after the
        // paths were read, the error would be about this file and not the flag.
        let missing = dir.path().join("does-not-exist.eml");
        let missing = missing.to_str().unwrap();

        // The premise: with valid flags, reading the path is what fails.
        let error = render(&args(&[missing]), TODAY).unwrap_err().to_string();
        assert!(error.contains("failed to read"), "{error}");

        for (flag, value, said) in [
            ("--rise-factor", "1", "--rise-factor must be at least 2"),
            (
                "--baseline-weeks",
                "0",
                "--baseline-weeks must be at least 1",
            ),
            ("--weeks", "0", "--weeks must be at least 1"),
        ] {
            let error = render(&args(&[missing, flag, value]), TODAY)
                .unwrap_err()
                .to_string();
            assert!(error.contains(said), "{flag} {value}: {error}");
            assert!(!error.contains("failed to read"), "{flag} {value}: {error}");
        }
    }

    /// The smallest value each flag accepts is accepted: a check that is one too
    /// strict would refuse a value the docs themselves use.
    #[test]
    fn the_smallest_meaningful_value_of_each_flag_is_accepted() {
        let dir = tempfile::tempdir().unwrap();
        six_weeks_with_a_rise(dir.path());
        let path = dir.path().to_str().unwrap();
        for (flag, value) in [
            ("--rise-factor", "2"),
            ("--baseline-weeks", "1"),
            ("--weeks", "1"),
        ] {
            let out = render(&args(&[path, flag, value]), TODAY);
            assert!(out.is_ok(), "{flag} {value}: {out:?}");
        }
    }

    /// The flags reach the output, not only the view: each changes what is printed.
    #[test]
    fn non_default_flags_change_what_is_printed() {
        let dir = tempfile::tempdir().unwrap();
        six_weeks_with_a_rise(dir.path());
        let path = dir.path().to_str().unwrap();
        let json = |flags: &[&str]| -> serde_json::Value {
            let mut all = vec![path, "--json"];
            all.extend_from_slice(flags);
            serde_json::from_str(&render(&args(&all), TODAY).unwrap()).unwrap()
        };
        let trend = |value: &serde_json::Value| value["rollup"]["trend"].clone();

        // By default: the whole six weeks, and the one rise.
        let by_default = trend(&json(&[]));
        assert_eq!(by_default["weeks"].as_array().unwrap().len(), 6);
        assert_eq!(by_default["rises"].as_array().unwrap().len(), 1);

        // --weeks shows the newest weeks only. The rise is judged over all of them.
        let three = trend(&json(&["--weeks", "3"]));
        assert_eq!(three["weeks"].as_array().unwrap().len(), 3);
        assert_eq!(three["entries"][0]["counts"], serde_json::json!([4, 4, 12]));
        assert_eq!(three["rises"].as_array().unwrap().len(), 1);

        // --rise-factor 4: 12 against an average of 4 is 3x, not 4x.
        assert!(
            trend(&json(&["--rise-factor", "4"]))["rises"]
                .as_array()
                .unwrap()
                .is_empty()
        );

        // --min-reports 13: the week holds 12.
        assert!(
            trend(&json(&["--min-reports", "13"]))["rises"]
                .as_array()
                .unwrap()
                .is_empty()
        );

        // --baseline-weeks 2: the average is over the two weeks before, 8 reports.
        let two = trend(&json(&["--baseline-weeks", "2"]));
        assert_eq!(two["rises"][0]["baseline_weeks"], 2);
        assert_eq!(two["rises"][0]["baseline_reports"], 8);
        assert_eq!(
            two["rule"],
            serde_json::json!({"factor": 3, "baseline_weeks": 2, "min_reports": 5})
        );

        // And the text names the rule it used.
        let text = render(
            &args(&[path, "--rise-factor", "2", "--min-reports", "6"]),
            TODAY,
        )
        .unwrap();
        assert!(
            text.contains(
                "at least 2x the average of the 4 weeks before it, with at least 6 reports"
            ),
            "{text}"
        );
    }

    #[test]
    fn an_input_with_nothing_readable_is_an_error_not_an_empty_count() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("old.msg"), "binary").unwrap();
        let error = collect(&[dir.path().to_path_buf()], TODAY).unwrap_err();
        assert!(error.to_string().contains("nothing to count"), "{error}");
        assert!(collect(&[dir.path().join("missing.eml")], TODAY).is_err());
    }
}
