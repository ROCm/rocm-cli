// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Reading back the log a managed serve wrote, for failure diagnostics.
//!
//! When `rocm serve --managed` returns `readiness: starting` and the model never
//! appears on `/v1/models`, the harness sees only a timeout: the engine's own
//! reason (an out-of-memory at engine init, a stalled weight download, a runtime
//! that failed to import) is written to the service log in a per-scenario temp
//! directory that no report artifact captures. Quoting its tail into the failure
//! is what makes such a stall diagnosable from CI output alone.
//!
//! `--managed` is what makes a stall this opaque: `rocm serve` returns as soon as
//! the supervisor is launched, so an engine that dies afterwards leaves a **zero**
//! exit code and no CLI-side error at all. Nothing about the failure is visible
//! from the exit status, which is why this module collects the evidence from
//! elsewhere — the CLI's own output, the engine's log, and the device state.
//!
//! Two things come out of here, and a failing serve step wants both:
//!
//! - [`service_log_tail`] — the last lines, quoted straight into the panic so the
//!   job log explains itself without downloading anything;
//! - [`archive_service_log`] — the whole file, copied into the results directory
//!   CI uploads, because the tail cannot show the engine's STARTUP banner (which
//!   backend and device it chose), and the temp directory holding the original is
//!   deleted with the scenario.

use std::path::Path;

/// How many trailing lines of a stalled service's log to quote in a failure.
const DEFAULT_TAIL_LINES: usize = 40;

/// Subdirectory of the results directory that archived service logs land in.
/// Its parent is what CI uploads, so this path is also how the log is addressed
/// inside the artifact.
const ARCHIVE_SUBDIR: &str = "service-logs";

/// Cap on one archived service log.
///
/// A runaway engine log — a stalled download's progress bars, a crash loop —
/// must not inflate the CI artifact. Past the cap the head and tail are kept and
/// the middle elided: llama.cpp and vLLM announce the backend and device they
/// selected in their FIRST lines and fail in their LAST, and it takes both halves
/// to tell "picked the wrong backend" apart from "died loading the weights".
const MAX_ARCHIVED_LOG_BYTES: u64 = 4 * 1024 * 1024;

/// The `log_path:` a `rocm serve --managed` plan reports for the service it
/// launched, if the output carries one.
fn parse_log_path(serve_stdout: &str) -> Option<&str> {
    serve_stdout
        .lines()
        .find_map(|line| line.trim().strip_prefix("log_path:"))
        .map(str::trim)
        .filter(|path| !path.is_empty())
}

/// The last `max` lines of `text`.
fn tail_lines(text: &str, max: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(max)..].join("\n")
}

/// The tail of the log written by the managed service `serve_stdout` launched.
///
/// Never fails: every reason the log can't be quoted (no `log_path` line, the
/// file is missing or unreadable, the engine wrote nothing) becomes part of the
/// returned text, because this only ever runs while reporting another failure.
#[must_use]
pub fn service_log_tail(serve_stdout: &str) -> String {
    let Some(path) = parse_log_path(serve_stdout) else {
        return "<no log_path in serve output>".to_owned();
    };
    // Read bytes, not a `String`: an engine log carries progress bars and ANSI
    // and can hold a partially written multi-byte sequence, so a strict UTF-8
    // read would discard the whole tail over one bad byte — in precisely the
    // failure this exists to explain. Replace the bad bytes and quote the rest.
    match std::fs::read(path) {
        Ok(bytes) => {
            let log = String::from_utf8_lossy(&bytes);
            if log.trim().is_empty() {
                format!("<{path} is empty>")
            } else {
                tail_lines(&log, DEFAULT_TAIL_LINES)
            }
        }
        Err(error) => format!("<failed to read {path}: {error}>"),
    }
}

/// A filename-safe form of a scenario name: lowercase, non-alphanumerics folded
/// to single dashes. Only ever used to LABEL a file whose uniqueness comes from
/// the service id already in its name, so collisions here cost nothing.
fn slugify(name: &str) -> String {
    let mut slug = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
        } else if !slug.ends_with('-') {
            slug.push('-');
        }
    }
    let slug = slug.trim_matches('-');
    if slug.is_empty() {
        "scenario".to_owned()
    } else {
        slug.to_owned()
    }
}

/// Read at most `max` bytes of `path` for archiving, plus a note describing any
/// elision. A file that fits is returned verbatim; a larger one is returned as
/// its head and tail with the middle skipped and the gap marked.
///
/// Bounded by SEEKING rather than by reading the file and trimming it after,
/// because the input is an engine log that has already misbehaved: a crash loop
/// or a stuck progress bar can leave one arbitrarily large, and an allocation
/// failure ABORTS the process rather than unwinding — destroying the very report
/// this is collecting. Seeking keeps the cost fixed however big the log grows.
///
/// The log may still be being written, so `len` is a snapshot: the head and tail
/// are always real, and only the elided count can be slightly stale.
fn read_clamped(path: &Path, max: u64) -> std::io::Result<(Vec<u8>, String)> {
    use std::io::{Read as _, Seek as _, SeekFrom};

    let mut file = std::fs::File::open(path)?;
    let len = file.metadata()?.len();
    if len <= max {
        let mut body = Vec::new();
        file.read_to_end(&mut body)?;
        return Ok((body, String::new()));
    }
    // `len > max >= 2 * half`, so the two halves can never overlap.
    let half = max / 2;
    let mut head = vec![0u8; half as usize];
    file.read_exact(&mut head)?;
    file.seek(SeekFrom::Start(len - half))?;
    let mut tail = vec![0u8; half as usize];
    file.read_exact(&mut tail)?;

    let elided = len - 2 * half;
    let marker = format!("\n\n<<< {elided} bytes elided by the E2E harness >>>\n\n");
    let mut body = head;
    body.extend_from_slice(marker.as_bytes());
    body.extend_from_slice(&tail);
    Ok((body, format!(" ({elided} bytes elided from the middle)")))
}

/// Copy the log written by the managed service `serve_stdout` launched into
/// `results_dir`, and say where it landed.
///
/// The original lives in the scenario's isolated temp data dir and is deleted
/// with that `TempDir` when the scenario ends, while CI uploads only the results
/// directory — so today a run keeps nothing of the one file that holds the
/// engine's own account of itself. Copying it there is what lets the NEXT red
/// run be root-caused, instead of the run after that.
///
/// Returns the archived path relative to `results_dir` (which is how it is
/// addressed inside the uploaded artifact), or a bracketed reason it could not
/// be archived.
///
/// Never fails, for the same reason [`service_log_tail`] never fails: it only
/// ever runs while another failure is already being reported, and panicking here
/// would replace that report with a double panic.
#[must_use]
pub fn archive_service_log(serve_stdout: &str, results_dir: &Path, scenario: &str) -> String {
    let Some(source) = parse_log_path(serve_stdout) else {
        return "<not archived: no log_path in serve output>".to_owned();
    };
    let source = Path::new(source);
    let (body, note) = match read_clamped(source, MAX_ARCHIVED_LOG_BYTES) {
        Ok(read) => read,
        Err(error) => {
            return format!(
                "<not archived: failed to read {}: {error}>",
                source.display()
            );
        }
    };
    let dir = results_dir.join(ARCHIVE_SUBDIR);
    if let Err(error) = std::fs::create_dir_all(&dir) {
        return format!(
            "<not archived: failed to create {}: {error}>",
            dir.display()
        );
    }
    // The service id already makes the source name unique per launch; the
    // scenario slug in front says which scenario owns it without opening it.
    let name = format!(
        "{}--{}",
        slugify(scenario),
        source
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("service.log")
    );
    match std::fs::write(dir.join(&name), body) {
        Ok(()) => format!("archived to {ARCHIVE_SUBDIR}/{name}{note}"),
        Err(error) => format!(
            "<not archived: failed to write {}: {error}>",
            dir.join(&name).display()
        ),
    }
}

/// Everything one `rocm serve --managed` attempt left behind, in the order a
/// reader needs it.
///
/// Collected as a struct rather than passed positionally because the fields are
/// six interchangeable strings, and getting two of them the wrong way round
/// would silently mislabel the evidence in a report nobody can cross-check.
#[derive(Debug, Clone, Copy)]
pub struct ServeAttempt<'a> {
    /// One line saying what went wrong, including the invocation and exit code.
    /// The caller owns this because only it knows whether the serve failed
    /// outright or exited 0 and then never served.
    pub headline: &'a str,
    /// What the device looked like before this attempt started (see the serve
    /// steps' `ensure_serve_port_free`). An undrained GPU is a common cause of a
    /// serve that never becomes ready, and is invisible from anything else here.
    pub device_state: &'a str,
    pub stdout: &'a str,
    pub stderr: &'a str,
    /// [`service_log_tail`], read BEFORE the service was stopped so it reflects
    /// what the engine wrote on its own rather than what the stop provoked.
    pub log_tail: &'a str,
    /// [`archive_service_log`]'s account of where the full log was saved.
    pub archived_log: &'a str,
    /// How stopping this attempt's service went. A stop that found no record, or
    /// failed, means the engine may still hold the port and the device — which
    /// changes how every later failure in the run should be read.
    pub stop_status: &'a str,
}

/// Render one serve attempt's evidence as a failure message.
///
/// Every section is always present and empty ones are marked, on the same
/// reasoning as [`crate::cli_failure_report`]: a silent engine and a harness that
/// dropped the output look identical otherwise, and they call for opposite next
/// steps.
#[must_use]
pub fn serve_attempt_report(attempt: &ServeAttempt<'_>) -> String {
    use crate::section;
    format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}",
        attempt.headline,
        section("device state", attempt.device_state),
        section("stdout", attempt.stdout),
        section("stderr", attempt.stderr),
        section("service log (tail)", attempt.log_tail),
        section("service log (full)", attempt.archived_log),
        section("stop", attempt.stop_status),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_ARCHIVED_LOG_BYTES, ServeAttempt, archive_service_log, parse_log_path, read_clamped,
        serve_attempt_report, service_log_tail, slugify, tail_lines,
    };

    const SERVE_PLAN: &str = "\
serve plan
  requested model: Qwen/Qwen3.5-0.8B
  engine: vllm
managed service launched
  service_id: vllm-qwen-qwen3-5-0-8b-1785145816856
  log_path: /tmp/rocm-e2e-abc123/data/services/vllm-qwen.log
  readiness: starting
";

    #[test]
    fn log_path_is_read_from_the_indented_plan_line() {
        assert_eq!(
            parse_log_path(SERVE_PLAN),
            Some("/tmp/rocm-e2e-abc123/data/services/vllm-qwen.log")
        );
    }

    #[test]
    fn output_without_a_log_path_yields_none() {
        assert_eq!(parse_log_path("serve plan\n  engine: vllm\n"), None);
        assert_eq!(parse_log_path("  log_path:   \n"), None);
    }

    #[test]
    fn tail_keeps_the_last_lines_and_never_panics_on_short_input() {
        let text = (1..=5)
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(tail_lines(&text, 2), "4\n5");
        assert_eq!(tail_lines(&text, 99), text);
        assert_eq!(tail_lines("", 3), "");
    }

    #[test]
    fn tail_of_a_real_file_is_quoted_verbatim() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("service.log");
        std::fs::write(&path, "boot\nENGINE ERROR: out of memory\n").expect("write log");
        let stdout = format!("managed service launched\n  log_path: {}\n", path.display());
        assert_eq!(
            service_log_tail(&stdout),
            "boot\nENGINE ERROR: out of memory"
        );
    }

    #[test]
    fn invalid_utf8_still_yields_the_tail() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("service.log");
        // A truncated multi-byte sequence, as a killed engine's part-written
        // progress bar leaves behind. The readable lines must survive it.
        let mut bytes = b"loading weights \xff\xfe\nENGINE ERROR: out of memory\n".to_vec();
        bytes.extend_from_slice(b"\xe2\x82");
        std::fs::write(&path, &bytes).expect("write log");
        let stdout = format!("  log_path: {}\n", path.display());
        let reported = service_log_tail(&stdout);
        assert!(
            reported.contains("ENGINE ERROR: out of memory"),
            "tail lost to invalid UTF-8: {reported}"
        );
        assert!(
            reported.starts_with("loading weights "),
            "unexpected tail: {reported}"
        );
    }

    #[test]
    fn every_unreadable_case_explains_itself_instead_of_failing() {
        assert_eq!(
            service_log_tail("managed service launched\n"),
            "<no log_path in serve output>"
        );

        let dir = tempfile::tempdir().expect("tempdir");
        let empty = dir.path().join("empty.log");
        std::fs::write(&empty, "  \n").expect("write log");
        assert_eq!(
            service_log_tail(&format!("  log_path: {}\n", empty.display())),
            format!("<{} is empty>", empty.display())
        );

        let missing = dir.path().join("missing.log");
        let reported = service_log_tail(&format!("  log_path: {}\n", missing.display()));
        assert!(
            reported.starts_with(&format!("<failed to read {}", missing.display())),
            "unexpected message: {reported}"
        );
    }

    /// Write a service log and return the serve stdout that points at it.
    fn planted_log(dir: &std::path::Path, name: &str, body: &[u8]) -> String {
        let path = dir.join(name);
        std::fs::write(&path, body).expect("write log");
        format!("managed service launched\n  log_path: {}\n", path.display())
    }

    /// The whole point: the file the scenario's `TempDir` is about to delete ends
    /// up under the results directory CI uploads, byte for byte.
    #[test]
    fn the_full_log_is_copied_into_the_results_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let results = tempfile::tempdir().expect("tempdir");
        // A startup banner the 40-line tail could not have shown, plus the
        // failure at the end — archiving has to preserve both.
        let body = b"ggml_cuda_init: found 1 ROCm device\nload: failed to load model\n";
        let stdout = planted_log(dir.path(), "lemonade-qwen3-0-6b-1785145816856.log", body);

        let reported =
            archive_service_log(&stdout, results.path(), "14 - A canonical HF checkpoint");

        let expected_name = "14-a-canonical-hf-checkpoint--lemonade-qwen3-0-6b-1785145816856.log";
        assert_eq!(
            reported,
            format!("archived to service-logs/{expected_name}"),
            "the report must name the path as it appears inside the artifact"
        );
        let archived = results.path().join("service-logs").join(expected_name);
        assert_eq!(
            std::fs::read(&archived).expect("archived log"),
            body,
            "the archived copy must be the log verbatim"
        );
    }

    /// Read a log of `body` under an explicit cap. The cap is a parameter so the
    /// boundary can be probed with bytes instead of megabytes.
    fn read_log(body: &[u8], max: u64) -> (Vec<u8>, String) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("service.log");
        std::fs::write(&path, body).expect("write log");
        read_clamped(&path, max).expect("read log")
    }

    /// A runaway log is bounded, but never at the cost of the startup banner:
    /// both ends survive and the gap says so.
    #[test]
    fn an_oversized_log_keeps_both_ends_and_marks_the_gap() {
        // 26 bytes under a 10-byte cap: 5 from each end, 16 elided.
        let body = b"HEAD-rocm-backend-selected";
        let (clamped, note) = read_log(body, 10);

        let text = String::from_utf8_lossy(&clamped);
        assert!(text.starts_with("HEAD-"), "head lost: {text}");
        assert!(text.ends_with("ected"), "tail lost: {text}");
        assert!(
            text.contains("<<< 16 bytes elided by the E2E harness >>>"),
            "the elision must be stated, not silent: {text}"
        );
        assert_eq!(note, " (16 bytes elided from the middle)");
    }

    /// The cap itself: at exactly the limit nothing may be dropped, and one byte
    /// over must drop exactly that byte — the halves must never overlap and so
    /// never duplicate content into the archive.
    #[test]
    fn the_cap_boundary_neither_over_nor_under_trims() {
        let body = b"abcdefgh";

        let (at_cap, note) = read_log(body, 8);
        assert_eq!(at_cap, body, "a log exactly at the cap must be verbatim");
        assert!(note.is_empty(), "no elision note expected: {note}");

        // One byte over: half = 3, so "abc" and "fgh" survive and "de" is the gap.
        // Asserted whole, so an overlap (which would duplicate bytes) cannot pass.
        let (over_cap, note) = read_log(body, 7);
        assert_eq!(
            String::from_utf8_lossy(&over_cap),
            "abc\n\n<<< 2 bytes elided by the E2E harness >>>\n\nfgh"
        );
        assert_eq!(note, " (2 bytes elided from the middle)");
    }

    /// Reading is bounded by seeking, not by loading the log and trimming it, so
    /// a log far larger than the cap costs no more than a log at the cap.
    #[test]
    fn a_log_far_over_the_cap_is_still_read_in_bounded_space() {
        let body = vec![b'x'; 1024];
        let (clamped, note) = read_log(&body, 16);
        assert_eq!(clamped.len(), 16 + note_marker_len(1008));
        assert_eq!(note, " (1008 bytes elided from the middle)");
    }

    fn note_marker_len(elided: u64) -> usize {
        format!("\n\n<<< {elided} bytes elided by the E2E harness >>>\n\n").len()
    }

    #[test]
    fn every_unarchivable_case_explains_itself_instead_of_failing() {
        let results = tempfile::tempdir().expect("tempdir");
        assert_eq!(
            archive_service_log("managed service launched\n", results.path(), "s"),
            "<not archived: no log_path in serve output>"
        );

        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("missing.log");
        let reported = archive_service_log(
            &format!("  log_path: {}\n", missing.display()),
            results.path(),
            "s",
        );
        assert!(
            reported.starts_with("<not archived: failed to read"),
            "unexpected message: {reported}"
        );
    }

    /// An elided archive still says so in the one line that reaches the failure
    /// message, so nobody reads a truncated log as the whole story.
    #[test]
    fn an_elided_archive_says_so_in_the_report_line() {
        let dir = tempfile::tempdir().expect("tempdir");
        let results = tempfile::tempdir().expect("tempdir");
        let body = vec![b'x'; MAX_ARCHIVED_LOG_BYTES as usize + 32];
        let stdout = planted_log(dir.path(), "svc.log", &body);

        let reported = archive_service_log(&stdout, results.path(), "s");

        assert_eq!(
            reported,
            "archived to service-logs/s--svc.log (32 bytes elided from the middle)"
        );
    }

    #[test]
    fn scenario_names_become_filename_safe_slugs() {
        assert_eq!(
            slugify("14 - A canonical Hugging Face checkpoint"),
            "14-a-canonical-hugging-face-checkpoint"
        );
        assert_eq!(slugify("serve/model:Q4_0"), "serve-model-q4-0");
        assert_eq!(slugify("   "), "scenario");
        assert_eq!(slugify(""), "scenario");
    }

    /// The regression #260 is about: a serve that exits 0 and never serves must
    /// carry the engine's own reason, the device it started on, and where the
    /// full log went — none of which the plain readiness timeout reported.
    #[test]
    fn a_stalled_serve_report_carries_every_piece_of_evidence() {
        let report = serve_attempt_report(&ServeAttempt {
            headline: "`rocm serve unsloth/Qwen3-0.6B-GGUF:Q4_0 --engine lemonade --managed` \
                       exited 0, but the endpoint never served `Qwen3-0.6B` within 600s",
            device_state: "device state: drained (48000 MiB free of 49000 MiB, floor 44100 MiB)",
            stdout: "managed service launched\n  readiness: starting",
            stderr: "",
            log_tail: "load: failed to load model",
            archived_log: "archived to service-logs/scenario--svc.log",
            stop_status: "lemonade-qwen: stopped",
        });

        for expected in [
            "exited 0",
            "device state: drained",
            "readiness: starting",
            "load: failed to load model",
            "service-logs/scenario--svc.log",
            "lemonade-qwen: stopped",
        ] {
            assert!(
                report.contains(expected),
                "'{expected}' missing from the report:\n{report}"
            );
        }
        // A silent stream is a finding, so it is labelled rather than dropped.
        assert!(report.contains("stderr: (empty)"), "{report}");
    }
}
