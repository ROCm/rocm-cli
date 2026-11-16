// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The `--follow` tail behind `rocm logs --service <id> --follow` and
//! `rocm services logs <id> --follow`.
//!
//! Split out of [`services_text.rs`](../services_text.rs) as a pure move when
//! that module outgrew the file-size limit (see `docs/architecture.md`); this
//! is the same code, not a rewrite. It is new code to begin with, not one of
//! the clusters mechanically relocated from `main.rs`.
//!
//! This module owns `follow_service_logs` and the seams its tests drive
//! (`follow_service_logs_with`, `follow_service_logs_via`, `follow_loop`), the
//! growing-file cursor and the pieces it yields (`FollowCursor`, `LogSegment`,
//! `write_log_segments`), the terminal footer that keeps the "how to stop"
//! hint under the newest output (`FollowFooter`, `follow_hint`,
//! `terminal_cols`), and the two small helpers the loop uses to say something
//! of its own (`newly_ready`, `write_note`).
//!
//! A `FollowCursor` follows a growing file; it is not `rocm_core::LogTail`,
//! the result of a bounded tail read, which is what the log *views* in
//! [`services_text.rs`](../services_text.rs) use.
//!
//! Only `follow_service_logs` is `pub(crate)`: `dispatch()` calls it for `rocm
//! logs --service <id> --follow` and `services_cmd.rs` calls it for `rocm
//! services logs <id> --follow`. Everything else here has no caller outside
//! this module and stays private.
//!
//! The reads it needs stay in [`services_text.rs`](../services_text.rs) and are
//! called from here: `load_managed_service` for the liveness re-read and
//! `render_service_logs_text` for the snapshot printed before the follow
//! starts. Nothing in `services_text.rs` calls back into this module.
//!
//! The regression tests for everything above moved here with it, reaching
//! shared fixtures (`test_paths`) through `crate::tests::` the same way the
//! other split-out modules do.

use std::fs;
use std::io::{self, Read as _, Seek as _, SeekFrom};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use anyhow::{Result, bail};
use rocm_core::{AppPaths, ManagedServiceRecord};

use crate::managed_service_status::managed_service_is_live;
use crate::services_text::{load_managed_service, render_service_logs_text};

/// How often `--follow` looks for new log output.
const LOG_FOLLOW_POLL: Duration = Duration::from_millis(500);
/// How often `--follow` re-reads the service record to see whether it stopped.
const LOG_FOLLOW_LIVENESS_EVERY: Duration = Duration::from_secs(2);

/// Reads whatever a growing log file gained since the last [`Self::poll`].
/// Not `rocm_core::LogTail`, which is what a bounded read of the end of a log
/// returns.
///
/// Download bars (tqdm, huggingface_hub) redraw in place with `\r`, so a
/// finished segment ends at either `\r` or `\n`; a trailing partial segment is
/// held back until its terminator arrives.
struct FollowCursor {
    path: PathBuf,
    offset: u64,
    pending: Vec<u8>,
}

/// One completed piece of log output and whether it ended in a carriage return.
#[derive(Debug, PartialEq, Eq)]
struct LogSegment {
    text: String,
    redraw: bool,
}

impl FollowCursor {
    /// Start following at the file's current end.
    fn at_end(path: &Path) -> Self {
        let offset = fs::metadata(path).map_or(0, |meta| meta.len());
        Self {
            path: path.to_path_buf(),
            offset,
            pending: Vec::new(),
        }
    }

    fn poll(&mut self) -> Vec<LogSegment> {
        let Ok(mut file) = fs::File::open(&self.path) else {
            return Vec::new();
        };
        let len = file.metadata().map_or(0, |meta| meta.len());
        if len < self.offset {
            // Truncated or replaced (a restart reuses the path): start over.
            self.offset = 0;
            self.pending.clear();
        }
        if file.seek(SeekFrom::Start(self.offset)).is_err() {
            return Vec::new();
        }
        let mut fresh = Vec::new();
        if file.read_to_end(&mut fresh).is_err() {
            return Vec::new();
        }
        self.offset += fresh.len() as u64;
        self.pending.extend_from_slice(&fresh);

        let mut segments = Vec::new();
        let mut start = 0;
        let mut index = 0;
        while index < self.pending.len() {
            let byte = self.pending[index];
            // A `\r` at the very end of the data may be the first half of a `\r\n`
            // that the next poll completes; hold it back rather than emit an
            // empty segment (which would erase a final redraw frame on a tty).
            if byte == b'\r' && index + 1 == self.pending.len() {
                break;
            }
            if byte == b'\n' || byte == b'\r' {
                let mut redraw = byte == b'\r';
                // The footer's row bookkeeping assumes the log text moves no cursor.
                let text = rocm_core::terminal::strip_terminal_control_sequences(
                    &String::from_utf8_lossy(&self.pending[start..index]),
                );
                // `\r\n` is one line break, not a redraw followed by an empty line.
                if redraw && self.pending.get(index + 1) == Some(&b'\n') {
                    redraw = false;
                    index += 1;
                }
                segments.push(LogSegment { text, redraw });
                start = index + 1;
            }
            index += 1;
        }
        self.pending.drain(..start);
        segments
    }

    /// The unterminated output still held back, for when following ends: a
    /// last write cut short by a crash is otherwise never shown.
    fn finish(&mut self) -> Option<LogSegment> {
        let mut bytes = std::mem::take(&mut self.pending);
        if bytes.last() == Some(&b'\r') {
            bytes.pop();
        }
        // Same stripping as `poll`: a cut-off write can end mid-escape.
        (!bytes.is_empty()).then(|| LogSegment {
            text: rocm_core::terminal::strip_terminal_control_sequences(&String::from_utf8_lossy(
                &bytes,
            )),
            redraw: false,
        })
    }
}

/// Writes `segments`; returns whether the last one was a redraw, which leaves the
/// cursor on that row (the next segment erases it first) instead of a fresh one.
fn write_log_segments(
    out: &mut impl io::Write,
    segments: &[LogSegment],
    tty: bool,
) -> Result<bool> {
    for segment in segments {
        match (tty, segment.redraw) {
            // Redraw the same terminal row, as the engine's own bar does.
            (true, true) => write!(out, "\r\x1b[2K{}", segment.text)?,
            (true, false) => writeln!(out, "\r\x1b[2K{}", segment.text)?,
            (false, _) => writeln!(out, "{}", segment.text)?,
        }
    }
    out.flush()?;
    Ok(tty && segments.last().is_some_and(|segment| segment.redraw))
}

/// The "how to stop" hint kept on the row just below the newest log output of a
/// terminal follow, so it stays in view however much the log scrolls.
///
/// While drawn, the cursor sits on the footer row. [`Self::erase`] puts it back
/// where the log output left off: on a fresh row, or, after a redraw segment
/// (a download bar), at the end of that still-open row, one row up.
struct FollowFooter {
    text: String,
    drawn: bool,
    /// The row above the footer is an unfinished redraw row.
    open_row: bool,
}

/// Led by the part that says the server keeps running: service ids run 40+
/// characters, so on a narrow terminal the end of the hint is what gets cut.
fn follow_hint(service_id: &str) -> String {
    format!("Ctrl-C stops following, the server keeps running. Following {service_id}")
}

impl FollowFooter {
    fn new(service_id: &str) -> Self {
        Self {
            text: follow_hint(service_id),
            drawn: false,
            open_row: false,
        }
    }

    fn erase(&mut self, out: &mut impl io::Write) -> Result<()> {
        if self.drawn {
            write!(out, "\r\x1b[2K")?;
            if self.open_row {
                write!(out, "\x1b[1A")?;
            }
            self.drawn = false;
        }
        Ok(())
    }

    /// Clear the footer and end an unfinished redraw row, leaving the cursor on
    /// a fresh row.
    fn finish(&mut self, out: &mut impl io::Write) -> Result<()> {
        self.erase(out)?;
        if self.open_row {
            writeln!(out)?;
            self.open_row = false;
        }
        out.flush()?;
        Ok(())
    }

    /// One line of the follow's own, written below the log output with the
    /// footer redrawn under it. An unfinished download bar row is ended first,
    /// not overwritten.
    fn write_notice(
        &mut self,
        out: &mut impl io::Write,
        text: &str,
        cols: Option<usize>,
    ) -> Result<()> {
        self.finish(out)?;
        writeln!(out, "{text}")?;
        self.draw(out, false, cols)
    }

    /// `cols` truncates the text so it never wraps onto a second row, which the
    /// one-row erase could not clear.
    fn draw(
        &mut self,
        out: &mut impl io::Write,
        open_row: bool,
        cols: Option<usize>,
    ) -> Result<()> {
        if open_row {
            writeln!(out)?;
        }
        let shown: String = match cols {
            Some(cols) => {
                crate::cli_progress::truncate_to_width(&self.text, cols.saturating_sub(1))
            }
            None => self.text.clone(),
        };
        write!(out, "\r\x1b[2K{shown}")?;
        out.flush()?;
        self.drawn = true;
        self.open_row = open_row;
        Ok(())
    }

    /// Write a batch of new output with the footer kept below it. A batch with
    /// nothing in it leaves the screen alone (no flicker on an idle log).
    fn write_batch(
        &mut self,
        out: &mut impl io::Write,
        segments: &[LogSegment],
        cols: Option<usize>,
    ) -> Result<()> {
        if segments.is_empty() {
            return Ok(());
        }
        self.erase(out)?;
        let open_row = write_log_segments(out, segments, true)?;
        self.draw(out, open_row, cols)
    }
}

fn terminal_cols() -> Option<usize> {
    crossterm::terminal::size()
        .ok()
        .map(|(cols, _)| usize::from(cols))
}

/// `rocm logs --service <id> --follow`: print the recent output, then keep
/// printing new output until the service is no longer starting or running.
/// Ctrl-C ends the follow and leaves the service running.
pub(crate) fn follow_service_logs(paths: &AppPaths, service_id: &str) -> Result<()> {
    use std::io::IsTerminal as _;
    let mut out = io::stdout();
    let tty = out.is_terminal();
    follow_service_logs_with(
        paths,
        service_id,
        &mut out,
        &mut io::stderr(),
        tty,
        LOG_FOLLOW_POLL,
        LOG_FOLLOW_LIVENESS_EVERY,
        &terminal_cols,
    )
}

/// True once, the first time `status` reads `ready`.
fn newly_ready(announced: &mut bool, status: &str) -> bool {
    let ready = status == "ready" && !*announced;
    *announced |= ready;
    ready
}

/// A line about the follow itself (the hint, "is ready.", "stopped following.").
/// On a terminal it shares the log's stream and row layout; piped, it goes to
/// `err` so `rocm logs -f | grep error` sees log text only.
fn write_note(
    out: &mut impl io::Write,
    err: &mut impl io::Write,
    tty: bool,
    line: &str,
) -> Result<()> {
    if tty {
        writeln!(out, "{line}")?;
    } else {
        writeln!(err, "{line}")?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn follow_service_logs_with(
    paths: &AppPaths,
    service_id: &str,
    out: &mut impl io::Write,
    err: &mut impl io::Write,
    tty: bool,
    poll: Duration,
    liveness_every: Duration,
    cols: &dyn Fn() -> Option<usize>,
) -> Result<()> {
    follow_service_logs_via(
        paths,
        service_id,
        &|| load_managed_service(paths, service_id),
        out,
        err,
        tty,
        poll,
        liveness_every,
        cols,
    )
}

/// [`follow_service_logs_with`] with the record read injected: a `ready` status
/// only sticks while the endpoint answers, so a test cannot plant one on disk.
#[allow(clippy::too_many_arguments)]
fn follow_service_logs_via(
    paths: &AppPaths,
    service_id: &str,
    load: &dyn Fn() -> Result<ManagedServiceRecord>,
    out: &mut impl io::Write,
    err: &mut impl io::Write,
    tty: bool,
    poll: Duration,
    liveness_every: Duration,
    cols: &dyn Fn() -> Option<usize>,
) -> Result<()> {
    let record = load()?;
    // Open the tail before the snapshot: output written in between is then shown
    // twice (in the snapshot and by the first poll) rather than lost, and the
    // tail can start mid-line.
    let mut tail = FollowCursor::at_end(&record.log_path);
    write!(out, "{}", render_service_logs_text(paths, service_id)?)?;
    out.flush()?;
    if !managed_service_is_live(&record) {
        write_note(
            out,
            err,
            tty,
            &format!(
                "\nService {service_id} is {}; not following.",
                record.status
            ),
        )?;
        // Same outcome as a follow that sees the server fail: not a success.
        if record.status == "failed" {
            bail!("service {service_id} failed; see the log output above");
        }
        return Ok(());
    }
    // On a terminal the hint is a footer that stays under the newest output; a
    // pipe or log file gets it once, since it has no row to keep it on.
    let mut footer = tty.then(|| FollowFooter::new(service_id));
    match footer.as_mut() {
        Some(footer) => footer.draw(out, false, cols())?,
        None => write_note(out, err, tty, &format!("\n{}", follow_hint(service_id)))?,
    }
    out.flush()?;

    let result = follow_loop(
        load,
        &record,
        &mut tail,
        &mut footer,
        out,
        err,
        tty,
        poll,
        liveness_every,
        cols,
    );
    if result.is_err()
        && let Some(footer) = footer.as_mut()
    {
        // Leave the terminal on a clean row so the caller's `Error:` is not glued
        // to the footer or to an unfinished download bar.
        let _ = footer.finish(out);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn follow_loop(
    load: &dyn Fn() -> Result<ManagedServiceRecord>,
    record: &ManagedServiceRecord,
    tail: &mut FollowCursor,
    footer: &mut Option<FollowFooter>,
    out: &mut impl io::Write,
    err: &mut impl io::Write,
    tty: bool,
    poll: Duration,
    liveness_every: Duration,
    cols: &dyn Fn() -> Option<usize>,
) -> Result<()> {
    let service_id = record.service_id.as_str();
    let mut announced_ready = record.status == "ready";
    let mut since_check = Duration::ZERO;
    loop {
        thread::sleep(poll);
        match footer.as_mut() {
            Some(footer) => footer.write_batch(out, &tail.poll(), cols())?,
            None => {
                write_log_segments(out, &tail.poll(), tty)?;
            }
        }
        since_check += poll;
        if since_check < liveness_every {
            continue;
        }
        since_check = Duration::ZERO;
        let record = load()?;
        if newly_ready(&mut announced_ready, &record.status) {
            let line = format!("Service {service_id} is ready.");
            match footer.as_mut() {
                Some(footer) => footer.write_notice(out, &line, cols())?,
                None => write_note(out, err, tty, &line)?,
            }
        }
        if !managed_service_is_live(&record) {
            // Output written between the last poll and the exit.
            let mut last = tail.poll();
            last.extend(tail.finish());
            let mut open_row = false;
            if let Some(footer) = footer.as_mut() {
                footer.erase(out)?;
                open_row = footer.open_row;
            }
            open_row = write_log_segments(out, &last, tty)? || (open_row && last.is_empty());
            if open_row {
                writeln!(out)?;
            }
            write_note(
                out,
                err,
                tty,
                &format!(
                    "Service {service_id} is {}; stopped following.",
                    record.status
                ),
            )?;
            // Same outcome as `serve --wait`: a server that failed is not a success.
            if record.status == "failed" {
                bail!("service {service_id} failed; see the log output above");
            }
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::thread;
    use std::time::Duration;

    use super::*;
    use crate::tests::test_paths;

    #[test]
    fn log_tail_yields_only_new_complete_segments() -> Result<()> {
        let (root, paths) = test_paths("log-tail");
        paths.ensure()?;
        let log = root.join("tail.log");
        fs::write(&log, "old line\n")?;
        let mut tail = FollowCursor::at_end(&log);
        assert!(tail.poll().is_empty());

        // tqdm-style bar redraws with `\r`; the last piece has no terminator yet.
        fs::write(&log, "old line\n 28%|##\r 33%|###\rdone\r\nhalf")?;
        let segments = tail.poll();
        assert_eq!(
            segments,
            vec![
                LogSegment {
                    text: " 28%|##".into(),
                    redraw: true
                },
                LogSegment {
                    text: " 33%|###".into(),
                    redraw: true
                },
                LogSegment {
                    text: "done".into(),
                    redraw: false
                },
            ]
        );
        // The held-back partial segment completes on the next poll.
        let mut file = fs::OpenOptions::new().append(true).open(&log)?;
        std::io::Write::write_all(&mut file, b" line\n")?;
        assert_eq!(
            tail.poll(),
            vec![LogSegment {
                text: "half line".into(),
                redraw: false
            }]
        );

        // A truncated log (restart reusing the path) is read from the start.
        fs::write(&log, "new\n")?;
        assert_eq!(
            tail.poll(),
            vec![LogSegment {
                text: "new".into(),
                redraw: false
            }]
        );
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn log_tail_holds_back_a_trailing_cr_and_flushes_the_rest_on_finish() -> Result<()> {
        let (root, paths) = test_paths("log-tail-edges");
        paths.ensure()?;
        let log = root.join("edges.log");
        fs::write(&log, "")?;
        let mut tail = FollowCursor::at_end(&log);

        // A `\r` ending the data may be half of a `\r\n`: it is not a redraw yet.
        fs::write(&log, " 99%|#\r")?;
        assert!(tail.poll().is_empty());
        let mut file = fs::OpenOptions::new().append(true).open(&log)?;
        std::io::Write::write_all(&mut file, b"\n")?;
        assert_eq!(
            tail.poll(),
            vec![LogSegment {
                text: " 99%|#".into(),
                redraw: false
            }],
            "the held `\\r` joins the next `\\n` instead of yielding an empty segment"
        );

        // Output cut short before any terminator is shown when following ends.
        std::io::Write::write_all(&mut file, b"crashed mid-wri")?;
        assert!(tail.poll().is_empty());
        assert_eq!(
            tail.finish(),
            Some(LogSegment {
                text: "crashed mid-wri".into(),
                redraw: false
            })
        );
        assert_eq!(tail.finish(), None);
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn write_log_segments_redraws_in_place_only_on_a_tty() -> Result<()> {
        let segments = [
            LogSegment {
                text: "28%".into(),
                redraw: true,
            },
            LogSegment {
                text: "done".into(),
                redraw: false,
            },
        ];
        let mut tty = Vec::new();
        write_log_segments(&mut tty, &segments, true)?;
        // A redraw segment leaves the cursor on its row; the next segment (or
        // line) erases the row first, and a plain line ends with a newline.
        assert_eq!(String::from_utf8(tty)?, "\r\x1b[2K28%\r\x1b[2Kdone\n");

        let mut piped = Vec::new();
        write_log_segments(&mut piped, &segments, false)?;
        assert_eq!(String::from_utf8(piped)?, "28%\ndone\n");
        Ok(())
    }

    fn seg(text: &str, redraw: bool) -> LogSegment {
        LogSegment {
            text: text.into(),
            redraw,
        }
    }

    const ERASE: &str = "\r\x1b[2K";

    #[test]
    fn follow_footer_is_redrawn_under_each_batch_of_new_lines() -> Result<()> {
        let mut footer = FollowFooter::new("svc_a");
        let hint = footer.text.clone();
        let mut out = Vec::new();
        footer.draw(&mut out, false, None)?;
        assert_eq!(String::from_utf8(out.clone())?, format!("{ERASE}{hint}"));

        out.clear();
        footer.write_batch(&mut out, &[seg("one", false), seg("two", false)], None)?;
        // Footer row cleared, lines written over it, footer redrawn on the next row.
        assert_eq!(
            String::from_utf8(out)?,
            format!("{ERASE}{ERASE}one\n{ERASE}two\n{ERASE}{hint}")
        );
        Ok(())
    }

    #[test]
    fn follow_footer_leaves_an_idle_screen_alone() -> Result<()> {
        let mut footer = FollowFooter::new("svc_a");
        let mut out = Vec::new();
        footer.draw(&mut out, false, None)?;
        out.clear();
        footer.write_batch(&mut out, &[], None)?;
        assert!(out.is_empty(), "{out:?}");
        Ok(())
    }

    #[test]
    fn follow_footer_keeps_a_download_bar_on_its_own_row_above_it() -> Result<()> {
        let mut footer = FollowFooter::new("svc_a");
        let hint = footer.text.clone();
        let mut out = Vec::new();
        footer.draw(&mut out, false, None)?;
        out.clear();

        footer.write_batch(&mut out, &[seg("28%", true)], None)?;
        // The bar row stays open; the footer goes on the row below it.
        assert_eq!(
            String::from_utf8(out.clone())?,
            format!("{ERASE}{ERASE}28%\n{ERASE}{hint}")
        );

        out.clear();
        footer.write_batch(&mut out, &[seg("43%", true)], None)?;
        // Erase the footer, step back up onto the bar row, redraw it in place.
        assert_eq!(
            String::from_utf8(out.clone())?,
            format!("{ERASE}\x1b[1A{ERASE}43%\n{ERASE}{hint}")
        );

        out.clear();
        footer.write_batch(&mut out, &[seg("done", false)], None)?;
        // A finished line replaces the bar, and the footer is back on a fresh row.
        assert_eq!(
            String::from_utf8(out)?,
            format!("{ERASE}\x1b[1A{ERASE}done\n{ERASE}{hint}")
        );
        Ok(())
    }

    #[test]
    fn follow_footer_is_cut_to_one_terminal_row() -> Result<()> {
        let mut footer = FollowFooter::new("svc_a");
        let mut out = Vec::new();
        footer.draw(&mut out, false, Some(30))?;
        // One column short of the width, so it never wraps, and what is kept is
        // the head of the hint: the part saying the server keeps running.
        let drawn = String::from_utf8(out)?;
        let shown = drawn.strip_prefix(ERASE).expect("erase prefix");
        assert_eq!(unicode_width::UnicodeWidthStr::width(shown), 29, "{shown}");
        assert!(shown.starts_with("Ctrl-C stops following, the"), "{shown}");
        Ok(())
    }

    #[test]
    fn log_tail_strips_cursor_escapes_from_engine_output() -> Result<()> {
        let (root, _paths) = test_paths("tail-escapes");
        fs::create_dir_all(&root)?;
        let log = root.join("engine.log");
        fs::write(&log, "")?;
        let mut tail = FollowCursor::at_end(&log);
        fs::write(&log, "before\x1b[1Aafter\n")?;
        let segments = tail.poll();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "beforeafter");
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn log_tail_finish_strips_cursor_escapes_from_a_cut_off_write() -> Result<()> {
        let (root, _paths) = test_paths("tail-finish-escapes");
        fs::create_dir_all(&root)?;
        let log = root.join("engine.log");
        fs::write(&log, "")?;
        let mut tail = FollowCursor::at_end(&log);
        fs::write(&log, "last\x1b[1Aword")?;
        assert!(tail.poll().is_empty());
        let segment = tail.finish().expect("unterminated output is returned");
        assert_eq!(segment.text, "lastword");
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn follow_service_logs_keeps_the_hint_under_new_output_on_a_tty() -> Result<()> {
        let (root, paths) = test_paths("follow-tty");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc_follow_tty",
            "vllm",
            "qwen3.5",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11438,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        record.status = "starting".to_owned();
        record.write()?;
        fs::write(&record.log_path, "before\n")?;

        let log_path = record.log_path;
        let manifest_paths = paths.clone();
        let writer = thread::spawn(move || -> Result<()> {
            let appended = (|| -> Result<()> {
                thread::sleep(Duration::from_millis(150));
                let mut file = fs::OpenOptions::new().append(true).open(&log_path)?;
                std::io::Write::write_all(&mut file, b"Downloading 43%\n")?;
                thread::sleep(Duration::from_millis(150));
                Ok(())
            })();
            let mut record = load_managed_service(&manifest_paths, "svc_follow_tty")?;
            record.status = "stopped".to_owned();
            record.write()?;
            appended
        });
        let mut out = Vec::new();
        let mut err = Vec::new();
        follow_service_logs_with(
            &paths,
            "svc_follow_tty",
            &mut out,
            &mut err,
            true,
            Duration::from_millis(20),
            Duration::from_millis(40),
            &|| None,
        )?;
        writer.join().expect("writer thread")?;
        let out = String::from_utf8(out)?;
        let hint = FollowFooter::new("svc_follow_tty").text;
        // The hint is drawn again right under the line that scrolled it away,
        // and only the stop message follows it, after the footer row is cleared.
        assert!(
            out.contains(&format!("Downloading 43%\n{ERASE}{hint}")),
            "{out:?}"
        );
        assert!(
            out.ends_with(&format!(
                "{hint}{ERASE}Service svc_follow_tty is stopped; stopped following.\n"
            )),
            "{out:?}"
        );
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn follow_service_logs_ends_an_open_download_bar_row_before_the_stop_message() -> Result<()> {
        let (root, paths) = test_paths("follow-tty-bar");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc_follow_bar",
            "vllm",
            "qwen3.5",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11440,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        record.status = "starting".to_owned();
        record.write()?;
        fs::write(&record.log_path, "")?;

        let log_path = record.log_path;
        let manifest_paths = paths.clone();
        let writer = thread::spawn(move || -> Result<()> {
            let appended = (|| -> Result<()> {
                thread::sleep(Duration::from_millis(150));
                let mut file = fs::OpenOptions::new().append(true).open(&log_path)?;
                std::io::Write::write_all(&mut file, b"28%\r43%\r\r")?;
                thread::sleep(Duration::from_millis(150));
                Ok(())
            })();
            let mut record = load_managed_service(&manifest_paths, "svc_follow_bar")?;
            record.status = "stopped".to_owned();
            record.write()?;
            appended
        });
        let mut out = Vec::new();
        let mut err = Vec::new();
        follow_service_logs_with(
            &paths,
            "svc_follow_bar",
            &mut out,
            &mut err,
            true,
            Duration::from_millis(20),
            Duration::from_millis(40),
            &|| None,
        )?;
        writer.join().expect("writer thread")?;
        let out = String::from_utf8(out)?;
        // The footer is erased and the cursor goes back up onto the bar's row; that row
        // is then ended with a newline so the stop message starts on its own row.
        assert!(
            out.ends_with(&format!(
                "{ERASE}\x1b[1A\nService svc_follow_bar is stopped; stopped following.\n"
            )),
            "{out:?}"
        );
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn follow_service_logs_draws_no_footer_for_a_stopped_service_on_a_tty() -> Result<()> {
        let (root, paths) = test_paths("follow-tty-stopped");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc_follow_tty_stopped",
            "vllm",
            "qwen3.5",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11439,
            "managed",
            0,
            None,
            None,
            None,
        );
        record.status = "stopped".to_owned();
        record.write()?;
        let mut out = Vec::new();
        let mut err = Vec::new();
        follow_service_logs_with(
            &paths,
            "svc_follow_tty_stopped",
            &mut out,
            &mut err,
            true,
            Duration::from_millis(1),
            Duration::from_millis(1),
            &|| None,
        )?;
        let out = String::from_utf8(out)?;
        assert!(!out.contains("Following"), "{out:?}");
        assert!(out.contains("not following."), "{out:?}");
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn follow_service_logs_prints_snapshot_and_returns_for_a_stopped_service() -> Result<()> {
        let (root, paths) = test_paths("follow-stopped");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc_follow_stopped",
            "vllm",
            "qwen3.5",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11436,
            "managed",
            0,
            None,
            None,
            None,
        );
        record.status = "stopped".to_owned();
        record.write()?;
        fs::write(&record.log_path, "Downloading 86%\n")?;

        let mut out = Vec::new();
        let mut err = Vec::new();
        follow_service_logs_with(
            &paths,
            "svc_follow_stopped",
            &mut out,
            &mut err,
            false,
            Duration::from_millis(1),
            Duration::from_millis(1),
            &|| None,
        )?;
        let out = String::from_utf8(out)?;
        let err = String::from_utf8(err)?;
        assert!(out.contains("Downloading 86%"), "{out}");
        // The notice is not log text, so a pipe keeps it off stdout.
        assert!(!out.contains("not following"), "{out}");
        assert!(
            err.contains("Service svc_follow_stopped is stopped; not following."),
            "{err}"
        );
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn newly_ready_fires_once_on_the_first_ready_status() {
        let mut announced = false;
        assert!(!newly_ready(&mut announced, "starting"));
        assert!(newly_ready(&mut announced, "ready"));
        assert!(!newly_ready(&mut announced, "ready"));
        // A follow that began on an already-ready server has announced nothing new.
        let mut announced = true;
        assert!(!newly_ready(&mut announced, "ready"));
    }

    #[test]
    fn follow_service_logs_fails_when_the_service_failed() -> Result<()> {
        let (root, paths) = test_paths("follow-ready-failed");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc_follow_fail",
            "vllm",
            "qwen3.5",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11438,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        record.status = "starting".to_owned();
        record.write()?;
        fs::write(&record.log_path, "before\n")?;

        let manifest_paths = paths.clone();
        let writer = thread::spawn(move || -> Result<()> {
            thread::sleep(Duration::from_millis(200));
            let mut record = load_managed_service(&manifest_paths, "svc_follow_fail")?;
            record.status = "failed".to_owned();
            record.write()
        });
        let mut out = Vec::new();
        let mut err = Vec::new();
        let result = follow_service_logs_with(
            &paths,
            "svc_follow_fail",
            &mut out,
            &mut err,
            false,
            Duration::from_millis(20),
            Duration::from_millis(40),
            &|| None,
        );
        writer.join().expect("writer thread")?;
        let err = String::from_utf8(err)?;
        assert!(err.contains("is failed; stopped following"), "{err}");
        let error = result.expect_err("a failed service must not exit 0");
        assert!(
            error.to_string().contains("svc_follow_fail failed"),
            "{error}"
        );
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn follow_service_logs_streams_new_output_until_the_service_stops() -> Result<()> {
        let (root, paths) = test_paths("follow-live");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc_follow_live",
            "vllm",
            "qwen3.5",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11437,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        record.status = "starting".to_owned();
        record.write()?;
        fs::write(&record.log_path, "before\n")?;

        let log_path = record.log_path;
        let manifest_paths = paths.clone();
        let writer = thread::spawn(move || -> Result<()> {
            let appended = (|| -> Result<()> {
                thread::sleep(Duration::from_millis(150));
                let mut file = fs::OpenOptions::new().append(true).open(&log_path)?;
                std::io::Write::write_all(&mut file, b"Downloading 43%\n")?;
                thread::sleep(Duration::from_millis(150));
                Ok(())
            })();
            // Stop the service even when the append failed, so the follow loop
            // under test always ends instead of hanging the run.
            let mut record = load_managed_service(&manifest_paths, "svc_follow_live")?;
            record.status = "stopped".to_owned();
            record.write()?;
            appended
        });
        let mut out = Vec::new();
        let mut err = Vec::new();
        follow_service_logs_with(
            &paths,
            "svc_follow_live",
            &mut out,
            &mut err,
            false,
            Duration::from_millis(20),
            Duration::from_millis(40),
            &|| None,
        )?;
        writer.join().expect("writer thread")?;
        let out = String::from_utf8(out)?;
        let err = String::from_utf8(err)?;
        // Piped: stdout is the log text alone, with no terminal escapes, and the
        // follow's own lines go to stderr, the hint exactly once.
        assert!(out.contains("Downloading 43%"), "{out}");
        assert!(!out.contains('\x1b'), "{out:?}");
        assert!(!out.contains("Following"), "{out}");
        assert!(!out.contains("stopped following"), "{out}");
        assert_eq!(err.matches("Following svc_follow_live").count(), 1, "{err}");
        assert!(err.contains("stopped following"), "{err}");
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    fn starting_record(paths: &AppPaths, id: &str, port: u16) -> Result<ManagedServiceRecord> {
        let mut record = ManagedServiceRecord::new(
            paths,
            id,
            "vllm",
            "qwen3.5",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            port,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        record.status = "starting".to_owned();
        record.write()?;
        Ok(record)
    }

    /// A record read that walks through `statuses` (the last one repeats).
    fn scripted_statuses(
        base: ManagedServiceRecord,
        statuses: &'static [&'static str],
    ) -> impl Fn() -> Result<ManagedServiceRecord> {
        let calls = std::cell::Cell::new(0_usize);
        move || {
            let call = calls.get();
            calls.set(call + 1);
            let mut record = base.clone();
            record.status = statuses[call.min(statuses.len() - 1)].to_owned();
            Ok(record)
        }
    }

    #[test]
    fn follow_service_logs_announces_ready_once_when_the_server_comes_up() -> Result<()> {
        let (root, paths) = test_paths("follow-ready-line");
        paths.ensure()?;
        let record = starting_record(&paths, "svc_follow_ready", 11441)?;
        fs::write(&record.log_path, "before\n")?;
        // Initial read, then one read per liveness check.
        let load = scripted_statuses(
            record,
            &["starting", "starting", "ready", "ready", "stopped"],
        );
        let mut out = Vec::new();
        let mut err = Vec::new();
        follow_service_logs_via(
            &paths,
            "svc_follow_ready",
            &load,
            &mut out,
            &mut err,
            false,
            Duration::from_millis(5),
            Duration::from_millis(5),
            &|| None,
        )?;
        let err = String::from_utf8(err)?;
        assert_eq!(
            err.matches("Service svc_follow_ready is ready.").count(),
            1,
            "{err}"
        );
        assert!(!String::from_utf8(out)?.contains("is ready."));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn follow_service_logs_ends_a_download_bar_row_before_the_ready_line_on_a_tty() -> Result<()> {
        let (root, paths) = test_paths("follow-ready-bar");
        paths.ensure()?;
        let record = starting_record(&paths, "svc_follow_rbar", 11442)?;
        fs::write(&record.log_path, "")?;
        let log_path = record.log_path.clone();
        let appender = thread::spawn(move || -> Result<()> {
            thread::sleep(Duration::from_millis(50));
            let mut file = fs::OpenOptions::new().append(true).open(&log_path)?;
            std::io::Write::write_all(&mut file, b"28%\r43%\r")?;
            Ok(())
        });
        // The bar is on screen when the status turns ready (liveness check 3 on).
        let calls = std::cell::Cell::new(0_usize);
        let base = record;
        let load = move || -> Result<ManagedServiceRecord> {
            let call = calls.get();
            calls.set(call + 1);
            if call >= 3 {
                thread::sleep(Duration::from_millis(30));
            }
            let mut record = base.clone();
            record.status = match call {
                0..=7 => "starting",
                8..=9 => "ready",
                _ => "stopped",
            }
            .to_owned();
            Ok(record)
        };
        let mut out = Vec::new();
        let mut err = Vec::new();
        follow_service_logs_via(
            &paths,
            "svc_follow_rbar",
            &load,
            &mut out,
            &mut err,
            true,
            Duration::from_millis(10),
            Duration::from_millis(10),
            &|| None,
        )?;
        appender.join().expect("appender thread")?;
        let out = String::from_utf8(out)?;
        // The bar's row is ended with a newline, then the ready line gets its own
        // row; it never replaces the bar.
        assert!(
            out.contains(&format!(
                "{ERASE}\x1b[1A\nService svc_follow_rbar is ready.\n"
            )),
            "{out:?}"
        );
        assert!(err.is_empty(), "a terminal follow writes nothing to stderr");
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn follow_service_logs_clears_the_footer_when_the_follow_errors() -> Result<()> {
        let (root, paths) = test_paths("follow-error");
        paths.ensure()?;
        let record = starting_record(&paths, "svc_follow_gone", 11443)?;
        fs::write(&record.log_path, "")?;
        let manifest = record.manifest_path;
        // The service record disappears mid-follow, so the next liveness read fails.
        let remover = thread::spawn(move || -> Result<()> {
            thread::sleep(Duration::from_millis(100));
            fs::remove_file(manifest)?;
            Ok(())
        });
        let mut out = Vec::new();
        let mut err = Vec::new();
        let result = follow_service_logs_with(
            &paths,
            "svc_follow_gone",
            &mut out,
            &mut err,
            true,
            Duration::from_millis(20),
            Duration::from_millis(40),
            &|| None,
        );
        remover.join().expect("remover thread")?;
        assert!(result.is_err(), "a vanished record must end the follow");
        let out = String::from_utf8(out)?;
        // The footer row is erased, so `Error:` starts on a clean row.
        assert!(out.ends_with(ERASE), "{out:?}");
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn follow_service_logs_fails_for_a_service_that_is_already_failed() -> Result<()> {
        let (root, paths) = test_paths("follow-already-failed");
        paths.ensure()?;
        let mut record = starting_record(&paths, "svc_follow_dead", 11444)?;
        record.status = "failed".to_owned();
        record.supervisor_pid = 0;
        record.write()?;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let result = follow_service_logs_with(
            &paths,
            "svc_follow_dead",
            &mut out,
            &mut err,
            false,
            Duration::from_millis(1),
            Duration::from_millis(1),
            &|| None,
        );
        let error = result.expect_err("a failed service must not exit 0");
        assert!(
            error.to_string().contains("svc_follow_dead failed"),
            "{error}"
        );
        assert!(String::from_utf8(err)?.contains("is failed; not following."));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }
}
