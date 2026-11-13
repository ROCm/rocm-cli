// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Bring a finished run of the shared GPU queue into the nightly's consolidated
//! E2E report, so the support matrix and the release-readiness verdict derived
//! from it keep covering the GPUs the queue tests.
//!
//! `nightly.yml`'s `e2e-report-nightly` merges one `report.json` per
//! `e2e-*-report` artifact. The queue publishes a different layout: one
//! directory per lane in its `queue-run-results` artifact, holding
//! `<channel>/results/{report.json,platform.json,commands.jsonl,...}` as
//! `scripts/run-e2e-lane.sh` writes them (docs/nightly-on-the-queue.md). This
//! turns the one into the other:
//!
//! * every lane's reports are copied to `<artifacts-dir>/e2e-queue-<lane>[-advisory]-<channel>-report/`,
//!   named so `e2e_report` labels them by lane and gate. Only the three files the
//!   report reads are copied; `lane.log`, the JUnit and HTML output and the
//!   harness diagnostics stay where they are.
//! * the report's `platform_slug` is replaced by the lane's name and channel. The
//!   slug is what the report keys a grid column by, and queue lanes of different
//!   models report one slug (an MI325X lane probes as `mi300x`, an MI355X as
//!   `mi350p`), which would fold them, and the self-hosted MI300X lane, into one
//!   column. The channel is there because a lane's manifest names the channel of
//!   the runtime it found, which need not be the one the lane was told to run.
//! * which lanes exist and which of them the queue blocks on is read from the
//!   run's lane matrix in its job log, the only place either is published. A lane
//!   the queue marks advisory is labelled so, and `release-readiness` leaves it
//!   out of its verdict. Anything this cannot read counts as blocking.
//! * a lane that reported nothing, or that did not finish cleanly and lacks a
//!   channel, gets an empty report for it, which the report shows as `EMPTY` and
//!   `release-readiness` counts as a failure for a blocking lane: a lane without
//!   a result is not a pass.
//! * a short status table of every lane is written for the top of the page.
//!
//! Nothing is imported unless the caller found a run: with `--absent <reason>`
//! only the status section is written, saying so, and the nightly report goes on
//! without queue lanes. The repository is public, so everything written is either
//! copied from the queue's own scanned files or built from fields this checks
//! against a strict pattern first.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use regex::Regex;

/// The channels `scripts/run-e2e-lane.sh` runs, in the order it runs them.
const CHANNELS: [&str; 2] = ["release", "nightly"];

/// The files of a lane's `<channel>/results` the consolidated report reads.
const REPORT_FILES: [&str; 3] = ["report.json", "platform.json", "commands.jsonl"];

#[derive(clap::Args)]
pub struct Args {
    /// Directory the queue run's `queue-run-results` artifact was downloaded
    /// into: one subdirectory per lane.
    #[arg(long, requires = "run_url", conflicts_with = "absent")]
    queue_dir: Option<PathBuf>,
    /// The queue run's job log, which prints the branch the run tested and its
    /// lane matrix. Without it, or when it names a branch other than `main`,
    /// nothing is imported.
    #[arg(long, requires = "queue_dir")]
    log: Option<PathBuf>,
    /// The queue run's URL, linked from the status section.
    #[arg(long)]
    run_url: Option<String>,
    /// When the queue run started, as the API reports it.
    #[arg(long, requires = "queue_dir")]
    run_started: Option<String>,
    /// Import nothing and say why: no queue run was found, or its results could
    /// not be fetched.
    #[arg(
        long,
        conflicts_with = "queue_dir",
        required_unless_present = "queue_dir"
    )]
    absent: Option<String>,
    /// Where the per-lane report directories go: the directory
    /// `xtask e2e-report` reads.
    #[arg(long)]
    artifacts_dir: PathBuf,
    /// Where to write the status section, for the top of the support matrix.
    #[arg(long)]
    section_out: PathBuf,
}

pub fn run(args: &Args) -> Result<()> {
    let section = match (&args.queue_dir, &args.absent) {
        (Some(queue_dir), _) => {
            let log = args.log.as_ref().and_then(|path| {
                std::fs::read_to_string(path)
                    .map_err(|e| eprintln!("warning: queue job log {}: {e}", path.display()))
                    .ok()
            });
            let log = parse_log(log.as_deref());
            if log.branch.as_deref() == Some("main") {
                // A log that yielded no lanes is as good as none for the lanes'
                // state: every lane is then read as blocking.
                let have_matrix = !log.lanes.is_empty();
                let plan = plan(&log, scan(queue_dir)?, have_matrix);
                write_reports(&plan, queue_dir, &args.artifacts_dir)?;
                render_section(
                    &plan,
                    args.run_url.as_deref(),
                    args.run_started.as_deref(),
                    have_matrix,
                )
            } else {
                // The run is of the workflow on `main`, but may have been asked
                // to test another branch, which is no night's result for `main`.
                // The log is the only place that says which, so without it
                // nothing is imported.
                render_absent(
                    "the newest queue run did not test main, or its job log could not be read \
                     to tell",
                )
            }
        }
        (None, Some(reason)) => render_absent(reason),
        (None, None) => bail!("one of --queue-dir and --absent is required"),
    };
    if let Some(parent) = args.section_out.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(&args.section_out, section)
        .with_context(|| format!("writing {}", args.section_out.display()))?;
    Ok(())
}

/// What the queue run's job log says about the run.
#[derive(Debug, Default, PartialEq, Eq)]
struct QueueLog {
    /// The branch the run tested and the commit every lane tested, from the
    /// report's first line. The branch is not the one the workflow ran on: a run
    /// started from `main` can be asked to test another branch.
    branch: Option<String>,
    commit: Option<String>,
    lanes: Vec<MatrixRow>,
}

/// One row of the lane matrix the queue prints.
#[derive(Debug, PartialEq, Eq)]
struct MatrixRow {
    lane: String,
    /// The lane's phase (`Done`, `Failed`, ...), when the row had the usual shape.
    phase: Option<String>,
    /// The lane command's exit status, or `-` when it never ran.
    exit: Option<String>,
    /// Whether the queue lets this lane fail the run. Anything but `advisory` is
    /// read as blocking.
    blocking: bool,
}

impl MatrixRow {
    /// A lane the queue reports as finished with exit status zero.
    fn is_clean(&self) -> bool {
        self.phase.as_deref() == Some("Done") && self.exit.as_deref() == Some("0")
    }
}

/// A lane name is a path segment and a word in an artifact name: lowercase
/// letters, digits and dashes only.
///
/// A name ending in a word the artifact names use for something else
/// (`-advisory`, `-release`, `-nightly`) is refused too: it would be read back
/// as a gate or a channel.
fn is_lane_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 40
        && !name.starts_with('-')
        && !["-advisory", "-release", "-nightly"]
            .iter()
            .any(|suffix| name.ends_with(suffix))
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// Read the lane matrix and the tested commit out of the run's job log.
///
/// The publisher prints its report with a `| ` prefix on every line, which the
/// runner puts a timestamp in front of. Only the fields that match a strict
/// pattern are kept; a line that does not is ignored, so a log that is not what
/// was expected yields no lanes rather than text to publish.
fn parse_log(log: Option<&str>) -> QueueLog {
    let mut out = QueueLog::default();
    let Some(log) = log else { return out };
    let tested = Regex::new(r"^rocm-cli (\S+) at ([0-9a-f]{40})$").expect("static pattern");
    let mut in_matrix = false;
    let mut unnamed = 0;
    for line in log.lines() {
        let line = line.trim_start_matches('\u{feff}');
        // `<timestamp> | <text>`: the part after the first `| `.
        let Some((_, body)) = line.split_once("| ") else {
            // A blank report line is `|` alone; either way the matrix is over.
            if line.trim_end().ends_with('|') {
                in_matrix = false;
            }
            continue;
        };
        let body = body.trim_end();
        if let Some(caps) = tested.captures(body) {
            out.branch = Some(caps[1].to_owned());
            out.commit = Some(caps[2].to_owned());
        }
        let tokens: Vec<&str> = body.split_whitespace().collect();
        if tokens.first() == Some(&"LANE") && tokens.last() == Some(&"GATE") {
            in_matrix = true;
            continue;
        }
        if !in_matrix {
            continue;
        }
        if body.is_empty() || tokens.first() == Some(&"SUMMARY:") {
            in_matrix = false;
            continue;
        }
        let (Some(lane), Some(gate)) = (tokens.first(), tokens.last()) else {
            continue;
        };
        if out.lanes.iter().any(|row| row.lane == *lane) {
            continue;
        }
        // A row whose lane cannot be named safely still counts: as a lane that
        // reported nothing, and blocking, since its gate cannot be trusted either.
        if !is_lane_name(lane) {
            unnamed += 1;
            out.lanes.push(MatrixRow {
                lane: format!("unnamed-{unnamed}"),
                phase: None,
                exit: None,
                blocking: true,
            });
            continue;
        }
        // LANE WORKLOAD PHASE EXIT EXPORT WAITED GATE
        let shaped = tokens.len() == 7;
        let phase = shaped
            .then(|| tokens[2])
            .filter(|p| p.bytes().all(|b| b.is_ascii_alphabetic()))
            .map(str::to_owned);
        let exit = shaped
            .then(|| tokens[3])
            .filter(|e| *e == "-" || (e.len() <= 4 && e.bytes().all(|b| b.is_ascii_digit())))
            .map(str::to_owned);
        out.lanes.push(MatrixRow {
            lane: (*lane).to_owned(),
            phase,
            exit,
            blocking: *gate != "advisory",
        });
    }
    out
}

/// The channels with a `results/report.json` under each lane directory of the
/// downloaded artifact. A directory (or the final file) that is a symlink is not
/// followed. A lane directory with a name that is not a lane name is kept as an
/// `unnamed-N` lane with no reports, so that it shows as a lane without a result
/// instead of dropping out unseen.
fn scan(queue_dir: &Path) -> Result<BTreeMap<String, Vec<&'static str>>> {
    let mut lanes = BTreeMap::new();
    let mut unnamed = 0;
    let entries = std::fs::read_dir(queue_dir)
        .with_context(|| format!("reading the queue results in {}", queue_dir.display()))?;
    for entry in entries {
        let entry = entry.context("reading a queue results entry")?;
        if !entry.file_type().is_ok_and(|t| t.is_dir()) {
            continue;
        }
        let Some(lane) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if !is_lane_name(&lane) {
            eprintln!("warning: a queue lane directory has an unexpected name; shown as unnamed");
            unnamed += 1;
            lanes.insert(format!("unnamed-{unnamed}"), Vec::new());
            continue;
        }
        let channels = CHANNELS
            .into_iter()
            .filter(|channel| regular_file(&entry.path().join(channel).join("results/report.json")))
            .collect();
        lanes.insert(lane, channels);
    }
    Ok(lanes)
}

fn regular_file(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok_and(|m| m.is_file())
}

/// What one lane contributes to the report.
#[derive(Debug, PartialEq, Eq)]
struct LanePlan {
    lane: String,
    blocking: bool,
    phase: Option<String>,
    exit: Option<String>,
    /// Channels whose reports are copied.
    reported: Vec<&'static str>,
    /// Channels (or, for `None`, the lane as a whole) an empty report stands in for.
    placeholders: Vec<Option<&'static str>>,
}

#[derive(Debug, PartialEq, Eq)]
struct Plan {
    commit: Option<String>,
    lanes: Vec<LanePlan>,
}

/// Decide, lane by lane, what goes into the report. `have_matrix` says whether the
/// lane matrix could be read at all: without it nothing is known of a lane's
/// state beyond its files.
fn plan(log: &QueueLog, found: BTreeMap<String, Vec<&'static str>>, have_matrix: bool) -> Plan {
    let mut names: Vec<&String> = found
        .keys()
        .chain(log.lanes.iter().map(|r| &r.lane))
        .collect();
    names.sort();
    names.dedup();
    let mut lanes = Vec::new();
    for name in names {
        let row = log.lanes.iter().find(|r| &r.lane == name);
        let reported = found.get(name).cloned().unwrap_or_default();
        let clean = row.is_some_and(MatrixRow::is_clean);
        let mut placeholders = Vec::new();
        if reported.is_empty() {
            // Nothing came back from this lane, whatever it was told to run.
            placeholders.push(None);
        } else if have_matrix && !clean {
            // It did not finish cleanly, and a channel it should have run left
            // no report. A lane that finished cleanly may have been told to run
            // one channel only.
            placeholders.extend(
                CHANNELS
                    .into_iter()
                    .filter(|channel| !reported.contains(channel))
                    .map(Some),
            );
        }
        lanes.push(LanePlan {
            lane: name.clone(),
            blocking: row.is_none_or(|r| r.blocking),
            phase: row.and_then(|r| r.phase.clone()),
            exit: row.and_then(|r| r.exit.clone()),
            reported,
            placeholders,
        });
    }
    Plan {
        commit: log.commit.clone(),
        lanes,
    }
}

/// Write each lane's report directories under `artifacts_dir`.
fn write_reports(plan: &Plan, queue_dir: &Path, artifacts_dir: &Path) -> Result<()> {
    for lane in &plan.lanes {
        for channel in &lane.reported {
            let dest = artifacts_dir.join(e2e_report::queue_artifact_name(
                &lane.lane,
                !lane.blocking,
                Some(channel),
            ));
            std::fs::create_dir_all(&dest)
                .with_context(|| format!("creating {}", dest.display()))?;
            let src = queue_dir.join(&lane.lane).join(channel).join("results");
            for file in REPORT_FILES {
                let from = src.join(file);
                if !regular_file(&from) {
                    continue;
                }
                if file == "platform.json" {
                    // A manifest that does not parse is left out: the report then
                    // names the lane as having no usable manifest.
                    let text = std::fs::read_to_string(&from)
                        .with_context(|| format!("reading {}", from.display()))?;
                    if let Some(text) = with_lane_slug(&text, &lane.lane, channel) {
                        std::fs::write(dest.join(file), text)
                            .with_context(|| format!("writing {}", dest.join(file).display()))?;
                    }
                } else {
                    std::fs::copy(&from, dest.join(file))
                        .with_context(|| format!("copying {}", from.display()))?;
                }
            }
        }
        for channel in &lane.placeholders {
            let dest = artifacts_dir.join(e2e_report::queue_artifact_name(
                &lane.lane,
                !lane.blocking,
                *channel,
            ));
            std::fs::create_dir_all(&dest)
                .with_context(|| format!("creating {}", dest.display()))?;
            // No scenarios at all: the report shows the lane as EMPTY.
            std::fs::write(dest.join("report.json"), "[]\n")
                .with_context(|| format!("writing {}", dest.join("report.json").display()))?;
        }
    }
    Ok(())
}

/// `platform.json` with its `platform_slug` (and the copy under `capability`)
/// set to the lane's name and channel, or `None` when it is not a JSON object.
fn with_lane_slug(text: &str, lane: &str, channel: &str) -> Option<String> {
    let mut manifest: serde_json::Value = serde_json::from_str(text).ok()?;
    let slug = serde_json::Value::String(format!("queue-{lane}-{channel}"));
    let object = manifest.as_object_mut()?;
    object.insert("platform_slug".to_owned(), slug.clone());
    if let Some(capability) = object
        .get_mut("capability")
        .and_then(serde_json::Value::as_object_mut)
    {
        capability.insert("platform_slug".to_owned(), slug);
    }
    serde_json::to_string_pretty(&manifest).ok()
}

/// Text that fits in a table cell, from a reason this repository wrote itself.
fn one_line(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_ascii_graphic() || c == ' ' {
                c
            } else {
                ' '
            }
        })
        .filter(|c| *c != '|' && *c != '`')
        .take(200)
        .collect::<String>()
        .trim()
        .to_owned()
}

const SECTION_HEADING: &str = "## Queue lanes\n\n";

/// The section for a nightly with no queue results to import.
fn render_absent(reason: &str) -> String {
    format!(
        "{SECTION_HEADING}_No queue lanes are in this report: {}. The GPUs the shared queue \
         tests are not shown tonight._\n\n",
        one_line(reason)
    )
}

fn is_run_url(url: &str) -> bool {
    Regex::new(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/actions/runs/[0-9]+$")
        .expect("static pattern")
        .is_match(url)
}

/// The status section: which run was imported, and one row per lane.
fn render_section(
    plan: &Plan,
    run_url: Option<&str>,
    run_started: Option<&str>,
    have_matrix: bool,
) -> String {
    let mut out = String::from(SECTION_HEADING);
    let mut source = String::from("Latest finished run of the shared GPU queue on `main`");
    if let Some(url) = run_url.filter(|u| is_run_url(u)) {
        let id = url.rsplit('/').next().unwrap_or_default();
        let _ = write!(source, ": [run {id}]({url})");
    }
    if let Some(started) = run_started.filter(|s| {
        Regex::new(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}Z$")
            .expect("static pattern")
            .is_match(s)
    }) {
        let _ = write!(source, ", started {started}");
    }
    if let Some(commit) = &plan.commit {
        let _ = write!(source, ", tested commit `{}`", &commit[..12]);
    }
    let _ = writeln!(out, "{source}.\n");
    if !have_matrix {
        out.push_str(
            "_The run's lane matrix could not be read, so every lane is shown as blocking._\n\n",
        );
    }
    out.push_str("| Lane | GPU | Gate | Phase | Exit | Channels reported |\n");
    out.push_str("|---|---|---|---|---|---|\n");
    for lane in &plan.lanes {
        let channels = if lane.reported.is_empty() {
            "none".to_owned()
        } else {
            lane.reported.join(", ")
        };
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {channels} |",
            lane.lane,
            e2e_report::queue_lane_gpu(&lane.lane),
            if lane.blocking {
                "blocking"
            } else {
                "advisory"
            },
            lane.phase.as_deref().unwrap_or("unknown"),
            lane.exit.as_deref().unwrap_or("unknown"),
        );
    }
    let _ = write!(
        out,
        "\nThe queue's own gate decides which lanes count: a lane it marks advisory is labelled \
         `{}` below and is left out of the release-readiness verdict, and a blocking lane that \
         reported no result is shown as `EMPTY` and counts as a failure there.\n\n",
        e2e_report::ADVISORY_QUEUE_MARKER
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lane matrix of the 2026-11-13 scheduled run, as its public job log
    /// printed it.
    const LOG: &str = "\u{feff}2026-11-13T02:22:14.0268419Z started gh-43351651832-1; waiting for it to finish
2026-11-13T03:45:26.1317642Z gh-43351651832-1 still running (10 min, 0 of 7 lane(s) finished)
2026-11-13T03:45:26.1323838Z request gh-43351651832-1, finished 2026-11-13T03:45:26Z
2026-11-13T03:45:26.1361153Z | image: \"rocm/dev-ubuntu-24.04:7.2.4\"
2026-11-13T03:45:26.1361751Z | blocking: \"mi300x,r9700s\"
2026-11-13T03:45:26.1362299Z | rocm-cli main at 691834a04540743d0dced173bd2ef70c1244b351
2026-11-13T03:45:26.1362503Z | lanes: mi300x mi325x mi350p mi350x mi355x r9700s w7900-dual
2026-11-13T03:45:26.1362739Z |
2026-11-13T03:45:26.1362910Z | LANE         WORKLOAD                       PHASE                EXIT  EXPORT    WAITED  GATE
2026-11-13T03:45:26.1363201Z | mi300x       rocm-cli-gh-43351651832-1-mi300x Failed               1     Exported  0m      block
2026-11-13T03:45:26.1363511Z | mi325x       rocm-cli-gh-43351651832-1-mi325x Failed               1     Exported  0m      advisory
2026-11-13T03:45:26.1364685Z | r9700s       rocm-cli-gh-43351651832-1-r9700s Failed               -     -         0m      block
2026-11-13T03:45:26.1364991Z | w7900-dual   rocm-cli-gh-43351651832-1-w7900-dual Done                 0     Exported  0m      advisory
2026-11-13T03:45:26.1365442Z |
2026-11-13T03:45:26.1365634Z | SUMMARY: 7 lane(s), 1 passed, 2 failed, 0 incomplete, 0 missing, 4 advisory
2026-11-13T03:45:26.1365895Z | lane mi300x: 7 file(s) collected
2026-11-13T03:45:26.1365895Z | mi999x       not a row after the matrix                      Failed 1 Exported 0m block
";

    fn row(lane: &str, phase: &str, exit: &str, blocking: bool) -> MatrixRow {
        MatrixRow {
            lane: lane.to_owned(),
            phase: Some(phase.to_owned()),
            exit: Some(exit.to_owned()),
            blocking,
        }
    }

    #[test]
    fn the_matrix_and_the_tested_commit_are_read_from_the_job_log() {
        let log = parse_log(Some(LOG));
        assert_eq!(log.branch.as_deref(), Some("main"));
        assert_eq!(
            log.commit.as_deref(),
            Some("691834a04540743d0dced173bd2ef70c1244b351")
        );
        assert_eq!(
            log.lanes,
            vec![
                row("mi300x", "Failed", "1", true),
                row("mi325x", "Failed", "1", false),
                row("r9700s", "Failed", "-", true),
                row("w7900-dual", "Done", "0", false),
            ],
            "a line after the matrix is not a lane"
        );
    }

    #[test]
    fn a_log_that_is_not_the_report_yields_nothing_to_publish() {
        assert_eq!(parse_log(None), QueueLog::default());
        assert_eq!(
            parse_log(Some(
                "<html>ERR_ACCESS_DENIED</html>\nLANE GATE\nmi300x x\n"
            )),
            QueueLog::default()
        );
        // A row of an unexpected shape keeps its lane and its gate, and nothing else.
        let odd = "2026-11-13T03:45:26Z | LANE WORKLOAD PHASE GATE\n\
                   2026-11-13T03:45:26Z | mi300x some workload Pending now advisory\n";
        assert_eq!(
            parse_log(Some(odd)).lanes,
            vec![MatrixRow {
                lane: "mi300x".to_owned(),
                phase: None,
                exit: None,
                blocking: false,
            }]
        );
        // Text that is not a lane name is never kept, whatever the row says; the
        // lane is counted, as blocking.
        let hostile = "2026-11-13T03:45:26Z | LANE GATE\n\
                       2026-11-13T03:45:26Z | ../../etc/passwd advisory\n\
                       2026-11-13T03:45:26Z | Uppercase block\n";
        let lanes = parse_log(Some(hostile)).lanes;
        assert_eq!(
            lanes.iter().map(|l| l.lane.as_str()).collect::<Vec<_>>(),
            ["unnamed-1", "unnamed-2"]
        );
        assert!(lanes.iter().all(|l| l.blocking));
    }

    fn found(entries: &[(&str, &[&'static str])]) -> BTreeMap<String, Vec<&'static str>> {
        entries
            .iter()
            .map(|(lane, channels)| ((*lane).to_owned(), channels.to_vec()))
            .collect()
    }

    #[test]
    fn a_blocking_lane_with_no_result_gets_an_empty_report_and_an_advisory_one_stays_advisory() {
        let log = parse_log(Some(LOG));
        let plan = plan(
            &log,
            found(&[
                ("mi300x", &["release"]),
                ("mi325x", &["release", "nightly"]),
                ("w7900-dual", &["release", "nightly"]),
            ]),
            true,
        );
        let by_lane = |name: &str| {
            plan.lanes
                .iter()
                .find(|l| l.lane == name)
                .expect("lane planned")
        };
        // Failed, and the nightly channel left nothing: one stands in for it.
        assert_eq!(by_lane("mi300x").placeholders, vec![Some("nightly")]);
        assert!(by_lane("mi300x").blocking);
        // Nothing at all came back from r9700s, which the queue never listed a file for.
        assert_eq!(by_lane("r9700s").placeholders, vec![None]);
        assert!(by_lane("r9700s").reported.is_empty());
        // Advisory stays advisory, and a lane with every channel needs no stand-in.
        assert!(!by_lane("mi325x").blocking);
        assert!(by_lane("mi325x").placeholders.is_empty());
        assert!(by_lane("w7900-dual").placeholders.is_empty());
        assert_eq!(
            plan.commit.as_deref(),
            Some("691834a04540743d0dced173bd2ef70c1244b351")
        );
    }

    #[test]
    fn a_lane_that_finished_cleanly_may_have_run_one_channel_only() {
        let log = QueueLog {
            branch: Some("main".to_owned()),
            commit: None,
            lanes: vec![row("mi300x", "Done", "0", true)],
        };
        let plan = plan(&log, found(&[("mi300x", &["release"])]), true);
        assert!(plan.lanes[0].placeholders.is_empty());
    }

    #[test]
    fn without_the_matrix_every_lane_is_blocking_and_only_a_lane_with_no_files_is_missing() {
        let plan = plan(
            &QueueLog::default(),
            found(&[("mi325x", &["release"]), ("r9700s", &[])]),
            false,
        );
        assert!(plan.lanes.iter().all(|l| l.blocking));
        assert!(plan.lanes[0].placeholders.is_empty());
        assert_eq!(plan.lanes[1].placeholders, vec![None]);
    }

    #[test]
    fn the_lane_and_channel_replace_the_slug_that_two_lanes_share() {
        let manifest = r#"{"platform_slug":"mi300x","capability":{"platform_slug":"mi300x",
            "gfx_target":"gfx942"},"versions":{"channel":"release"},"expectations":[]}"#;
        let rewritten = with_lane_slug(manifest, "mi325x", "nightly").expect("an object");
        let value: serde_json::Value = serde_json::from_str(&rewritten).expect("json");
        assert_eq!(value["platform_slug"], "queue-mi325x-nightly");
        assert_eq!(value["capability"]["platform_slug"], "queue-mi325x-nightly");
        assert_eq!(value["capability"]["gfx_target"], "gfx942");
        assert_eq!(value["versions"]["channel"], "release");
        assert_eq!(with_lane_slug("[1]", "mi325x", "release"), None);
        assert_eq!(with_lane_slug("not json", "mi325x", "release"), None);
    }

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        std::fs::write(path, text).expect("write");
    }

    /// A queue artifact as the run above published it, cut down to two lanes.
    fn queue_artifact(dir: &Path) {
        for (lane, slug, channels) in [
            ("mi300x", "mi300x", &["release"][..]),
            ("mi325x", "mi300x", &["release", "nightly"][..]),
        ] {
            write(&dir.join(lane).join("lane.log"), "the lane's own log\n");
            for channel in channels {
                let results = dir.join(lane).join(channel).join("results");
                write(&results.join("report.json"), "[]\n");
                write(&results.join("commands.jsonl"), "");
                write(&results.join("junit.xml"), "<testsuites/>");
                write(&results.join("harness-diagnostics.log"), "diagnostics");
                write(
                    &results.join("platform.json"),
                    &format!(
                        r#"{{"platform_slug":"{slug}","versions":{{"channel":"{channel}"}},"expectations":[]}}"#
                    ),
                );
            }
        }
        write(&dir.join("r9700s").join("lane.log"), "no files came back\n");
    }

    #[test]
    fn the_import_writes_only_what_the_report_reads_under_names_that_say_lane_and_gate() {
        let queue = tempfile::tempdir().expect("tempdir");
        queue_artifact(queue.path());
        let log = queue.path().join("job.log");
        // The log sits beside the lanes here; a stray file is no lane.
        write(&log, LOG);
        let out = tempfile::tempdir().expect("tempdir");
        let section = out.path().join("consolidated/queue-lanes.md");
        let args = Args {
            queue_dir: Some(queue.path().to_owned()),
            log: Some(log),
            run_url: Some("https://github.com/ROCm/rocm-cli/actions/runs/43351651832".to_owned()),
            run_started: Some("2026-11-13T02:21:58Z".to_owned()),
            absent: None,
            artifacts_dir: out.path().join("e2e-artifacts"),
            section_out: section.clone(),
        };
        run(&args).expect("import");

        let mut dirs: Vec<String> = std::fs::read_dir(out.path().join("e2e-artifacts"))
            .expect("artifacts")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        dirs.sort();
        assert_eq!(
            dirs,
            [
                "e2e-queue-mi300x-nightly-report",
                "e2e-queue-mi300x-release-report",
                "e2e-queue-mi325x-advisory-nightly-report",
                "e2e-queue-mi325x-advisory-release-report",
                "e2e-queue-r9700s-report",
                // Listed in the log, with no files at all: an empty stand-in.
                "e2e-queue-w7900-dual-advisory-report",
            ]
        );
        let copied = out
            .path()
            .join("e2e-artifacts/e2e-queue-mi325x-advisory-release-report");
        let mut files: Vec<String> = std::fs::read_dir(&copied)
            .expect("lane dir")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        files.sort();
        assert_eq!(files, ["commands.jsonl", "platform.json", "report.json"]);
        let manifest = std::fs::read_to_string(copied.join("platform.json")).expect("manifest");
        assert!(manifest.contains("queue-mi325x-release"), "{manifest}");
        assert_eq!(
            std::fs::read_to_string(
                out.path()
                    .join("e2e-artifacts/e2e-queue-r9700s-report/report.json")
            )
            .expect("placeholder"),
            "[]\n"
        );

        let section = std::fs::read_to_string(section).expect("section");
        assert!(section.starts_with("## Queue lanes\n"), "{section}");
        assert!(
            section.contains(
                "[run 43351651832](https://github.com/ROCm/rocm-cli/actions/runs/43351651832), \
                 started 2026-11-13T02:21:58Z, tested commit `691834a04540`"
            ),
            "{section}"
        );
        assert!(
            section.contains("| mi325x | MI325X | advisory | Failed | 1 | release, nightly |"),
            "{section}"
        );
        assert!(
            section.contains("| r9700s | R9700 | blocking | Failed | - | none |"),
            "{section}"
        );
        // Nothing of the lane's own log or the workload names reaches the page.
        assert!(!section.contains("rocm-cli-gh-"), "{section}");
        assert!(!section.contains("lane's own log"), "{section}");
    }

    #[test]
    fn an_absent_queue_run_is_a_section_that_says_so_and_nothing_else() {
        let out = tempfile::tempdir().expect("tempdir");
        let section = out.path().join("queue-lanes.md");
        run(&Args {
            queue_dir: None,
            log: None,
            run_url: None,
            run_started: None,
            absent: Some("no queue run finished on main in the last 24 hours | `x`\n".to_owned()),
            artifacts_dir: out.path().join("e2e-artifacts"),
            section_out: section.clone(),
        })
        .expect("an absent run is not an error");
        assert!(!out.path().join("e2e-artifacts").exists());
        let text = std::fs::read_to_string(section).expect("section");
        assert_eq!(
            text,
            "## Queue lanes\n\n_No queue lanes are in this report: no queue run finished on \
             main in the last 24 hours  x. The GPUs the shared queue tests are not shown \
             tonight._\n\n"
        );
    }

    fn import_with_log(log_text: Option<&str>) -> (tempfile::TempDir, String) {
        let queue = tempfile::tempdir().expect("tempdir");
        queue_artifact(queue.path());
        let out = tempfile::tempdir().expect("tempdir");
        let log = log_text.map(|text| {
            let path = out.path().join("job.log");
            write(&path, text);
            path
        });
        let section = out.path().join("queue-lanes.md");
        run(&Args {
            queue_dir: Some(queue.path().to_owned()),
            log,
            run_url: Some("https://github.com/ROCm/rocm-cli/actions/runs/1".to_owned()),
            run_started: None,
            absent: None,
            artifacts_dir: out.path().join("e2e-artifacts"),
            section_out: section.clone(),
        })
        .expect("a log that cannot be used is not an error");
        let text = std::fs::read_to_string(section).expect("section");
        (out, text)
    }

    #[test]
    fn a_run_that_tested_another_branch_or_whose_log_is_unreadable_is_not_imported() {
        // A run started from main can be asked to test any branch of the repository.
        let other = LOG.replace("rocm-cli main at", "rocm-cli some-branch at");
        for log in [
            Some(other.as_str()),
            // What the egress proxy answers when it refuses the request.
            Some("<html>ERR_ACCESS_DENIED</html>\n"),
            None,
        ] {
            let (out, text) = import_with_log(log);
            assert!(
                text.contains("did not test main, or its job log could not be read"),
                "{text}"
            );
            assert!(!text.contains("| Lane |"), "{text}");
            assert!(!out.path().join("e2e-artifacts").exists());
        }
    }

    #[test]
    fn a_log_without_a_matrix_leaves_every_lane_blocking_and_says_so() {
        let (out, text) = import_with_log(Some(
            "2026-11-13T03:45:26Z | rocm-cli main at 691834a04540743d0dced173bd2ef70c1244b351\n",
        ));
        assert!(text.contains("every lane is shown as blocking"), "{text}");
        assert!(!text.contains("advisory |"), "{text}");
        // The lane with no files is still a missing result; the others are not
        // second-guessed about a channel they never ran.
        let arts = out.path().join("e2e-artifacts");
        assert!(arts.join("e2e-queue-r9700s-report").is_dir());
        assert!(!arts.join("e2e-queue-mi300x-nightly-report").exists());
    }

    #[test]
    fn a_lane_that_cannot_be_named_still_shows_as_a_blocking_lane_with_no_result() {
        for name in [
            "Mi300X",
            "mi300x_2",
            "x-advisory",
            "x-release",
            "x-nightly",
            "-x",
        ] {
            assert!(!is_lane_name(name), "{name}");
        }
        let log = parse_log(Some(
            "2026-11-13T03:45:26Z | LANE WORKLOAD PHASE EXIT EXPORT WAITED GATE\n\
             2026-11-13T03:45:26Z | x-advisory w Done 0 Exported 0m advisory\n",
        ));
        assert_eq!(
            log.lanes,
            vec![MatrixRow {
                lane: "unnamed-1".to_owned(),
                phase: None,
                exit: None,
                blocking: true,
            }]
        );
        // The same for a directory of the artifact.
        let queue = tempfile::tempdir().expect("tempdir");
        write(&queue.path().join("mi300x_2/lane.log"), "log\n");
        write(
            &queue.path().join("mi300x/release/results/report.json"),
            "[]",
        );
        let found = scan(queue.path()).expect("scan");
        assert_eq!(found.get("unnamed-1"), Some(&Vec::new()));
        assert_eq!(found.get("mi300x"), Some(&vec!["release"]));
        let plan = plan(&QueueLog::default(), found, false);
        let unnamed = plan
            .lanes
            .iter()
            .find(|l| l.lane == "unnamed-1")
            .expect("lane");
        assert!(unnamed.blocking);
        assert_eq!(unnamed.placeholders, vec![None]);
    }

    #[test]
    fn a_run_url_that_is_not_a_run_is_not_linked() {
        let plan = Plan {
            commit: None,
            lanes: Vec::new(),
        };
        let text = render_section(&plan, Some("https://example.com/x)[y]("), None, true);
        assert!(!text.contains("example.com"), "{text}");
    }
}
