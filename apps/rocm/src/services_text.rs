// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Local-server log rendering and the `rocm services` status/log surfaces,
//! plus the managed-service record reads and the `stop_server` answer shape the
//! stop and restart paths build on.
//!
//! Mechanically relocated from `main.rs` with no behavior change, and later
//! narrowed: the process-control cluster (stop, restart and the internal
//! sandbox-tool runner) lives in `service_control.rs` (see
//! `docs/architecture.md`).
//!
//! This module owns: the CLI/TUI logs browser (`render_logs_text`,
//! `render_logs_browser_text`, `render_logs_browser_page_text`,
//! `render_logs_browser_page_text_with_options`, `logs_browser_total_pages`,
//! the `LogBrowserEntry` type, `collect_log_browser_entries`,
//! `filter_log_browser_entries`, `log_browser_source_label`,
//! `write_past_attempts_hint`), the `rocm services`/`rocm services logs`
//! status surfaces (`render_services_text`, `render_services_json`,
//! `render_service_logs_text`, `render_service_logs_text_with_options`,
//! `render_service_action_result`, `read_managed_service_manifest`,
//! `load_managed_service`), the internal `rocmd`-compatible status surface
//! (`render_internal_status_text`), the `stop_server` answer shape
//! (`internal_stop_server_value`, `service_stop_unconfirmed`) and its renderer
//! (`render_unconfirmed_service_stop`, `process_check_command`), plus
//! `validate_service_id`, `list_log_files`, `read_optional_tail_lines`, and the
//! CLI-lifecycle-log tail formatting helpers (`format_cli_lifecycle_tail_line`,
//! `lifecycle_event_label`, `humanize_log_label`, `lifecycle_field`,
//! `read_tail_lines`).
//!
//! The `LogBrowserEntry` struct moved here with the logs browser and stays
//! private.
//!
//! `dispatch()` calls directly into this module at several points —
//! `render_internal_status_text`, `render_service_logs_text`,
//! `render_logs_browser_text`, and `render_logs_text` — re-imported via
//! `use crate::services_text::{...};`. `services_cmd.rs`'s `services()` calls
//! `render_services_text`, `render_services_json` and `render_service_logs_text`
//! through its own imports, and its `run_approved_service_action` and
//! `finish_service_action` also read `render_service_action_result`,
//! `service_stop_unconfirmed` and `validate_service_id`. `service_records.rs`
//! reads `validate_service_id` (in `service_record_artifacts` and
//! `remove_managed_service_record`) and
//! `load_managed_service` (in `remove_managed_service_record` and
//! `apply_service_prune_plan`).
//! `load_managed_service` is `pub(crate)` and read from `runtime_services.rs`
//! (reconciling live servers against a runtime activation), from
//! `service_control.rs` and from `managed_service_status.rs`'s own test module;
//! `read_managed_service_manifest`, `internal_stop_server_value` and
//! `process_check_command` are `pub(crate)` for `service_control.rs` (the last
//! one only for its tests); `read_tail_lines`, `render_logs_text`,
//! `render_logs_browser_text`, `render_services_text`, `render_service_logs_text`,
//! and `validate_service_id` are also `pub(crate)` and read from `chat.rs`'s
//! deterministic tool-output summaries; `read_optional_tail_lines` and
//! `render_service_action_result` are also `pub(crate)` and read from
//! `service_launch.rs`'s managed-spawn/attach path. Every other item in this
//! list has no caller outside this module and stays private.
//!
//! The regression tests for what stays here live in this module's own
//! `mod tests`, reaching shared fixtures (`test_paths`) through `crate::tests::`
//! the same way other mechanically relocated modules do. The tests that drive
//! the stop and restart paths moved to `service_control.rs`; `assert_no_retry_line`
//! stays here as a `pub(crate)` helper because it checks this module's
//! unconfirmed-stop rendering and `service_launch.rs`'s tests read it too. Two
//! tests that call `load_managed_service` only incidentally — their primary
//! subject is `runtime_services::pin_service_record_to_runtime`/
//! `restore_service_record_pin` and
//! `runtime_services::restart_service_onto_runtime_with`'s outcome
//! classification respectively — stay in `main.rs`'s shared `mod tests`
//! instead.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, ManagedServiceRecord};
use rocm_engine_protocol::DEFAULT_LOG_TAIL_LINES;

use crate::managed_service_status::{
    load_managed_services, local_server_sidebar_status, managed_service_is_live,
    managed_service_sidebar_counts, refresh_managed_service_runtime_liveness,
};
use crate::service_hints::{self, service_action_past_tense};
use crate::shell_quote::shell_quote;
use crate::{SandboxToolArg, cli_lifecycle_log_path};

pub(crate) fn render_logs_text(paths: &AppPaths) -> String {
    render_logs_browser_text(paths, None)
}

pub(crate) fn render_logs_browser_text(paths: &AppPaths, query: Option<&str>) -> String {
    render_logs_browser_page_text(paths, query, 0, 24)
}

fn render_logs_browser_page_text(
    paths: &AppPaths,
    query: Option<&str>,
    page: usize,
    page_size: usize,
) -> String {
    render_logs_browser_page_text_with_options(paths, query, page, page_size, true)
}

fn render_logs_browser_page_text_with_options(
    paths: &AppPaths,
    query: Option<&str>,
    page: usize,
    page_size: usize,
    show_file_locations: bool,
) -> String {
    let mut output = String::new();
    let lifecycle_path = cli_lifecycle_log_path(paths);
    let action_dir = paths.data_dir.join("logs").join("cli");
    let screen_dir = paths.data_dir.join("logs").join("tui");
    let query = query.map(str::trim).filter(|value| !value.is_empty());
    let page_size = page_size.max(1);
    let _ = writeln!(output, "Logs");
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "File locations: {}",
        if show_file_locations {
            "shown"
        } else {
            "hidden"
        }
    );
    if show_file_locations {
        let _ = writeln!(
            output,
            "  Folder: {}",
            paths.data_dir.join("logs").display()
        );
        let _ = writeln!(output, "  Activity log: {}", lifecycle_path.display());
        let _ = writeln!(output, "  Command logs: {}", action_dir.display());
        let _ = writeln!(output, "  Screen command logs: {}", screen_dir.display());
        let _ = writeln!(
            output,
            "  Audit events: {}",
            paths.audit_events_path().display()
        );
    } else {
        let _ = writeln!(output, "  Choose Show file locations to see exact paths.");
    }
    let action_logs = list_log_files(&action_dir, 12);
    let screen_logs = list_log_files(&screen_dir, 12);
    if show_file_locations {
        if action_logs.is_empty() && screen_logs.is_empty() {
            let _ = writeln!(output, "  Recent command files: none yet");
        } else {
            let _ = writeln!(output, "  Recent command files:");
            for log in action_logs {
                let display = log.strip_prefix(&action_dir).unwrap_or(&log);
                let _ = writeln!(output, "    cli/{}", display.display());
            }
            for log in screen_logs {
                let display = log.strip_prefix(&screen_dir).unwrap_or(&log);
                let _ = writeln!(output, "    screen/{}", display.display());
            }
        }
    }
    let _ = writeln!(output);
    if query.is_some() {
        let _ = writeln!(output, "Recent activity: filtered by search");
    } else {
        let recent_lines = read_optional_tail_lines(&lifecycle_path, 8, "CLI lifecycle log");
        if recent_lines.is_empty() {
            let _ = writeln!(output, "Recent activity: no activity yet");
        } else {
            let _ = writeln!(
                output,
                "Recent activity: last {} line(s)",
                recent_lines.len()
            );
            for line in recent_lines {
                let _ = writeln!(output, "  {}", format_cli_lifecycle_tail_line(&line));
            }
        }
    }

    let entries = collect_log_browser_entries(paths);
    let matching_entries = filter_log_browser_entries(&entries, query);
    let total_pages = logs_browser_total_pages(matching_entries.len(), page_size);
    let page = page.min(total_pages.saturating_sub(1));
    let start = page.saturating_mul(page_size);
    let end = start.saturating_add(page_size).min(matching_entries.len());
    let _ = writeln!(output);
    let _ = writeln!(output, "Matching lines");
    let _ = writeln!(output, "  Search: {}", query.unwrap_or("none"));
    let _ = writeln!(
        output,
        "  Lines: {} of {} recent line(s)",
        matching_entries.len(),
        entries.len()
    );
    let _ = writeln!(output, "  Page: {} of {}", page + 1, total_pages);
    if matching_entries.is_empty() {
        let _ = writeln!(output, "  Showing: 0 of 0");
    } else {
        let _ = writeln!(
            output,
            "  Showing: {}-{} of {}",
            start + 1,
            end,
            matching_entries.len()
        );
    }
    if entries.is_empty() {
        let _ = writeln!(output, "  No logs found yet.");
    } else if matching_entries.is_empty() {
        let _ = writeln!(output, "  No matching lines.");
    } else {
        let _ = writeln!(output, "  Lines:");
        for entry in matching_entries.into_iter().skip(start).take(page_size) {
            let _ = writeln!(
                output,
                "    {}: {}",
                log_browser_source_label(&entry.source, show_file_locations),
                entry.line
            );
        }
    }
    output
}

fn logs_browser_total_pages(item_count: usize, page_size: usize) -> usize {
    item_count.div_ceil(page_size).max(1)
}

#[derive(Debug, Clone)]
struct LogBrowserEntry {
    source: String,
    line: String,
}

fn collect_log_browser_entries(paths: &AppPaths) -> Vec<LogBrowserEntry> {
    let lifecycle_path = cli_lifecycle_log_path(paths);
    let mut entries = Vec::new();
    for line in read_optional_tail_lines(&lifecycle_path, 8, "CLI lifecycle log") {
        entries.push(LogBrowserEntry {
            source: "lifecycle".to_owned(),
            line: format_cli_lifecycle_tail_line(&line),
        });
    }

    let action_dir = paths.data_dir.join("logs").join("cli");
    for path in list_log_files(&action_dir, 12) {
        let display = path
            .strip_prefix(&action_dir)
            .unwrap_or(&path)
            .display()
            .to_string();
        for line in read_optional_tail_lines(&path, 12, "CLI action log") {
            entries.push(LogBrowserEntry {
                source: format!("action/{display}"),
                line,
            });
        }
    }

    let screen_dir = paths.data_dir.join("logs").join("tui");
    for path in list_log_files(&screen_dir, 12) {
        let display = path
            .strip_prefix(&screen_dir)
            .unwrap_or(&path)
            .display()
            .to_string();
        for line in read_optional_tail_lines(&path, 12, "screen command log") {
            entries.push(LogBrowserEntry {
                source: format!("screen/{display}"),
                line,
            });
        }
    }
    entries
}

fn filter_log_browser_entries<'a>(
    entries: &'a [LogBrowserEntry],
    query: Option<&str>,
) -> Vec<&'a LogBrowserEntry> {
    let Some(query) = query else {
        return entries.iter().collect();
    };
    let query = query.to_ascii_lowercase();
    entries
        .iter()
        .filter(|entry| {
            entry.source.to_ascii_lowercase().contains(&query)
                || entry.line.to_ascii_lowercase().contains(&query)
        })
        .collect()
}

fn log_browser_source_label(source: &str, show_file_locations: bool) -> String {
    if source == "lifecycle" {
        "recent activity".to_owned()
    } else if let Some(path) = source.strip_prefix("action/") {
        if show_file_locations {
            format!("command log {path}")
        } else {
            "command output".to_owned()
        }
    } else if let Some(path) = source.strip_prefix("screen/") {
        if show_file_locations {
            format!("screen command log {path}")
        } else {
            "screen command output".to_owned()
        }
    } else {
        source.to_owned()
    }
}

/// Tell the default (`!all`) view about local server records it is hiding.
///
/// The default view lists only live servers, so a host whose serves all failed
/// saw an empty list with no hint that anything was recorded. Name the command
/// that lists them, the one that reads a specific log (with a real id, so that
/// line can be pasted as-is), and the one that deletes them.
///
/// `prune` is named last and without qualification on purpose. Every record
/// carries an unrotated engine log, so on a host that has served real models
/// this block is pointing at the only thing on disk the user can get back — and
/// the other two lines only ever let them *look*. Nothing is said here about
/// how `prune` decides what to remove: that is its own command's output, and
/// its behaviour is under change on a sibling branch.
fn write_past_attempts_hint(output: &mut String, past_attempts: usize, newest_id: Option<&str>) {
    if past_attempts == 0 {
        return;
    }
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "Past attempts: {past_attempts} local server record(s) that are no longer running."
    );
    let _ = writeln!(output, "  See them: rocm services list --all");
    if let Some(id) = newest_id {
        // Quoted like every other `rocm services … <id>` hint, so the line is
        // the argv it names even for an id that is not one inert word.
        let _ = writeln!(
            output,
            "  Read the newest: rocm services logs {}",
            shell_quote(id)
        );
    }
    let _ = writeln!(output, "  Reclaim the space: rocm services prune");
}

pub(crate) fn render_services_text(paths: &AppPaths, all: bool) -> Result<String> {
    // Counts describe the WHOLE registry, not the rows this view keeps. The
    // default view hides records that are not live, and counting the filtered
    // list made the header say "none ready" on a host whose only local servers
    // had failed - the records were on disk and nothing mentioned them. Rows
    // stay filtered; only the header and the hint below see everything.
    let all_records = load_managed_services(paths)?;
    let counts = managed_service_sidebar_counts(&all_records);
    // `load_managed_services` sorts newest-first, so the first record that is
    // not live is the newest one - the id worth putting in a pasteable hint.
    let newest_past_attempt = all_records
        .iter()
        .find(|record| !managed_service_is_live(record))
        .map(|record| record.service_id.clone());
    let records = all_records
        .into_iter()
        .filter(|record| all || managed_service_is_live(record))
        .collect::<Vec<_>>();
    let mut output = String::new();
    let _ = writeln!(output, "Local Servers");
    let _ = writeln!(output);
    let _ = writeln!(output, "Status: {}", local_server_sidebar_status(&counts));
    let _ = writeln!(output);
    if records.is_empty() {
        let _ = if all {
            writeln!(output, "No local server records yet.")
        } else {
            writeln!(output, "No local servers are running.")
        };
        // Gate the separator on the same condition as the hint, not just on
        // `!all`: `write_past_attempts_hint` early-returns at zero, so a host
        // that has never served would otherwise gain a blank line between "No
        // local servers are running." and the next line - a visible change to
        // the one output this PR is not meant to touch.
        if !all && counts.past_attempts > 0 {
            write_past_attempts_hint(
                &mut output,
                counts.past_attempts,
                newest_past_attempt.as_deref(),
            );
            let _ = writeln!(output);
        }
        let _ = writeln!(
            output,
            "Start one with `rocm serve <model> --managed`, or run `rocm` and choose Serve."
        );
        return Ok(output);
    }

    let _ = writeln!(output, "Servers");
    for record in records {
        let _ = writeln!(output, "- {}", record.service_id);
        let _ = writeln!(output, "  status: {}", record.status);
        let _ = writeln!(output, "  engine: {}", record.engine);
        let _ = writeln!(output, "  model: {}", record.model_ref);
        let _ = writeln!(output, "  endpoint: {}", record.endpoint_url);
        let _ = writeln!(
            output,
            "  logs: rocm services logs {}",
            shell_quote(&record.service_id)
        );
        if matches!(
            record.status.as_str(),
            "ready" | "running" | "starting" | "recovering"
        ) {
            let _ = writeln!(
                output,
                "  stop: {}",
                service_hints::service_stop_hint(&record.service_id)
            );
        } else {
            let _ = writeln!(
                output,
                "  restart: {}",
                service_hints::service_action_hint("restart", &record.service_id)
            );
        }
    }
    // A host can have one server ready and three failed, so the populated
    // branch needs the hint just as much as the empty one.
    if !all {
        write_past_attempts_hint(
            &mut output,
            counts.past_attempts,
            newest_past_attempt.as_deref(),
        );
    }
    Ok(output)
}

/// The machine-readable counterpart to [`render_services_text`].
///
/// Applies the same liveness filter as the text form so `--json` and the table
/// agree on which services they consider current — the two must not disagree
/// about what is running. Emits the `ManagedServiceRecord`s verbatim rather than
/// a bespoke projection: `rocm remote` deserializes them back into the same type
/// on the other side of its control channel, so any field this dropped would be
/// a field the remote orchestration could never see.
pub(crate) fn render_services_json(paths: &AppPaths, all: bool) -> Result<String> {
    let records = load_managed_services(paths)?
        .into_iter()
        .filter(|record| all || managed_service_is_live(record))
        .collect::<Vec<_>>();
    let mut output = serde_json::to_string_pretty(&records)
        .context("failed to serialize the managed service records as JSON")?;
    output.push('\n');
    Ok(output)
}

pub(crate) fn render_service_logs_text(paths: &AppPaths, service_id: &str) -> Result<String> {
    render_service_logs_text_with_options(paths, service_id, true)
}

fn render_service_logs_text_with_options(
    paths: &AppPaths,
    service_id: &str,
    show_file_locations: bool,
) -> Result<String> {
    let record = load_managed_service(paths, service_id)?;
    let recent_lines =
        read_optional_tail_lines(&record.log_path, DEFAULT_LOG_TAIL_LINES, "service log");

    let mut output = String::new();
    let _ = writeln!(output, "Service Log");
    let _ = writeln!(output);
    let _ = writeln!(output, "Service: {}", record.service_id);
    let _ = writeln!(output, "Engine: {}", record.engine);
    let _ = writeln!(output, "Status: {}", record.status);
    let _ = writeln!(output, "Endpoint: {}", record.endpoint_url);
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "File locations: {}",
        if show_file_locations {
            "shown"
        } else {
            "hidden"
        }
    );
    if show_file_locations {
        let _ = writeln!(output, "  Details file: {}", record.manifest_path.display());
        let _ = writeln!(output, "  Log file: {}", record.log_path.display());
    } else {
        let _ = writeln!(output, "  Choose Show file locations to see exact paths.");
    }
    let _ = writeln!(output);
    if recent_lines.is_empty() {
        let _ = writeln!(output, "Recent output: no output yet");
    } else {
        let _ = writeln!(output, "Recent output: last {} line(s)", recent_lines.len());
        for line in recent_lines {
            let _ = writeln!(output, "  {line}");
        }
    }
    Ok(output)
}

/// [`render_service_action_result`], plus — for a stop — the note about the
/// background automation helper when it outlives the stopped service.
pub(crate) fn render_service_action_result_with_helper_note(
    paths: &AppPaths,
    tool: &str,
    value: &serde_json::Value,
) -> String {
    let mut text = render_service_action_result(tool, value);
    if tool == "stop_server"
        && let Some(note) = crate::service_launch::background_helper_note(paths)
    {
        let _ = writeln!(text, "  note: {}", note.text);
    }
    text
}

pub(crate) fn render_service_action_result(tool: &str, value: &serde_json::Value) -> String {
    let output = value.get("output").unwrap_or(value);
    if tool == "stop_server" && service_stop_unconfirmed(value) {
        return render_unconfirmed_service_stop(output);
    }
    let action = service_action_past_tense(tool);
    let service = output.get("service").or_else(|| {
        output
            .get("result")
            .and_then(|result| result.get("service"))
    });
    let mut text = String::new();
    let _ = writeln!(text, "Local server {action}");
    if let Some(service) = service {
        if let Some(service_id) = service
            .get("service_id")
            .and_then(serde_json::Value::as_str)
        {
            let _ = writeln!(text, "  service: {service_id}");
        }
        if let Some(status) = service.get("status").and_then(serde_json::Value::as_str) {
            let _ = writeln!(text, "  status: {status}");
        }
        if let Some(endpoint) = service
            .get("endpoint_url")
            .and_then(serde_json::Value::as_str)
        {
            let _ = writeln!(text, "  endpoint: {endpoint}");
        }
    }
    if let Some(result) = output.get("result")
        && let Some(count) = result
            .get("signaled_pids")
            .and_then(serde_json::Value::as_array)
            .map(Vec::len)
    {
        let _ = writeln!(text, "  stopped processes: {count}");
    }
    text
}

/// A `stop_server` answer whose stop could not confirm that every recorded
/// process is gone. The stop kept the record's processes, its stop request and
/// any endpoint key (see `stop_internal_managed_service_confirmed_with` in
/// `service_control.rs`), so this says so instead of claiming a stop, with a
/// line for each reason it found:
///
/// - a process that was signalled and that, or a process under it, outlived
///   the deadline (`timed_out`);
/// - a process that was not signalled because its identity could not be
///   verified (`unverified`), with how to look at it. It may have exited
///   meanwhile: the identity read can fail because the process is going away;
/// - the record changed while the stop ran, typically because a restart
///   recorded new processes, so the stop's verdict says nothing about what the
///   record names now.
///
/// It names no command to run the stop again, like `rocmd`'s MCP answer: a
/// second stop cannot be relied on to finish the job. When the recorded
/// process is gone and only a process under it survived, the second stop finds
/// the recorded one gone and confirms without reaching the survivor.
fn render_unconfirmed_service_stop(output: &serde_json::Value) -> String {
    let result = output.get("result");
    let service_id = result
        .and_then(|result| result.get("service_id"))
        .and_then(serde_json::Value::as_str);
    let outcomes = result
        .and_then(|result| result.get("pid_outcomes"))
        .and_then(serde_json::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let with_outcome = |label: &str| -> Vec<(u64, &str)> {
        outcomes
            .iter()
            .filter(|entry| entry.get("outcome").and_then(serde_json::Value::as_str) == Some(label))
            .filter_map(|entry| {
                Some((
                    entry.get("pid").and_then(serde_json::Value::as_u64)?,
                    entry
                        .get("role")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("process"),
                ))
            })
            .collect()
    };
    let timed_out = with_outcome(rocm_core::TerminationOutcome::TimedOut.as_str());
    let unverified = with_outcome(rocm_core::TerminationOutcome::Unverified.as_str());

    let mut text = String::new();
    let _ = writeln!(text, "Local server stop not confirmed");
    if let Some(service_id) = service_id {
        let _ = writeln!(text, "  service: {service_id}");
    }
    // No `status:` line: a record the liveness refresh already read as
    // `stopped` keeps that status through a stop that confirmed nothing, so
    // printing it here would contradict the header.
    if let Some(count) = result
        .and_then(|result| result.get("signaled_pids"))
        .and_then(serde_json::Value::as_array)
        .map(Vec::len)
    {
        let _ = writeln!(text, "  signaled processes: {count}");
    }
    for (pid, role) in &timed_out {
        let _ = writeln!(
            text,
            "  still running: process {pid} ({role}) or a process under it did not exit after it was signaled"
        );
    }
    for (pid, role) in &unverified {
        let _ = writeln!(
            text,
            "  not signaled: process {pid} ({role}) could not be verified as this server's; it may still be running"
        );
    }
    let record_changed = result
        .and_then(|result| result.get("record_changed"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    if record_changed {
        let _ = writeln!(
            text,
            "  note: the record changed while the stop ran, so it may now name processes this stop did not handle"
        );
    }
    if result
        .and_then(|result| result.get("engine_stop_unconfirmed"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        let why = result
            .and_then(|result| result.pointer("/engine_stop/error"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("it did not answer");
        let _ = writeln!(text, "  not confirmed: the engine's own stop failed: {why}");
    }
    let endpoint_key_kept = result
        .and_then(|result| result.get("endpoint_key_kept"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    if endpoint_key_kept {
        let _ = writeln!(
            text,
            "  kept: the server record, which still names its processes, and its endpoint key, which the server may still be enforcing"
        );
    } else {
        let _ = writeln!(
            text,
            "  kept: the server record, which still names its processes"
        );
    }
    for (pid, _) in &unverified {
        let _ = writeln!(
            text,
            "  check: `{}` shows whether process {pid} is still running; stop it yourself only if it belongs to this server",
            process_check_command(*pid)
        );
    }
    if record_changed {
        let _ = writeln!(
            text,
            "  check: `rocm services` lists what the record names now"
        );
    }
    text
}

/// A stock OS command that shows whether `pid` is running, for a process a
/// stop left alone because it could not verify it. Not a `rocm` command: the
/// CLI deliberately does not act on that process, so the user decides.
///
/// A Linux command because only Linux can produce that outcome: it needs a
/// recorded start-time that can no longer be read back, and
/// `rocm_core::process_start_ticks` records one only on Linux.
pub(crate) fn process_check_command(pid: u64) -> String {
    format!("ps -p {pid} -o pid,args")
}

/// A managed service's record exactly as it is on disk, without the engine-state
/// and liveness refresh [`load_managed_service`] applies (and may persist).
pub(crate) fn read_managed_service_manifest(
    paths: &AppPaths,
    service_id: &str,
) -> Result<(ManagedServiceRecord, PathBuf)> {
    validate_service_id(service_id)?;
    let manifest_path = paths.service_manifest_path(service_id);
    let mut record = ManagedServiceRecord::read_manifest(paths, service_id)?;
    record.normalize_paths_for_host();
    Ok((record, manifest_path))
}

pub(crate) fn load_managed_service(
    paths: &AppPaths,
    service_id: &str,
) -> Result<ManagedServiceRecord> {
    let (mut record, manifest_path) = read_managed_service_manifest(paths, service_id)?;
    let refreshed_from_engine = record.refresh_from_engine_state().unwrap_or(false);
    let refreshed_liveness = refresh_managed_service_runtime_liveness(paths, &mut record);
    if refreshed_from_engine || refreshed_liveness {
        let _ = record.write();
    }
    if record.service_id != service_id {
        bail!(
            "managed service manifest {} contains service_id `{}`, expected `{service_id}`",
            manifest_path.display(),
            record.service_id
        );
    }
    Ok(record)
}

pub(crate) fn render_internal_status_text(paths: &AppPaths) -> Result<String> {
    let services = load_managed_services(paths)?;
    let mut output = String::new();
    let _ = writeln!(output, "rocmd status");
    let _ = writeln!(output, "  config dir: {}", paths.config_dir.display());
    let _ = writeln!(output, "  data dir: {}", paths.data_dir.display());
    let _ = writeln!(
        output,
        "  policy: built into rocm; no separate rocmd binary is required"
    );
    let _ = writeln!(output, "  services: {}", services.len());
    Ok(output)
}

/// Whether a `stop_internal_managed_service` report (see `service_control.rs`)
/// confirms that every recorded process is gone. A report without the verdict
/// counts as unconfirmed, so a missing field can never read as a completed
/// stop. Same rule as `rocmd`'s `stop_report_confirmed`.
fn stop_report_confirmed(report: &serde_json::Value) -> bool {
    report.get("stopped").and_then(serde_json::Value::as_bool) == Some(true)
}

/// The `stop_server` sandbox tool's answer for a stop report: `stopped` only
/// when the stop confirmed it, `stop_unconfirmed` otherwise — the same status
/// `rocmd`'s `stop_server` tool reports, taken from the same verdict that
/// decided what the stop wrote to the record.
pub(crate) fn internal_stop_server_value(report: serde_json::Value) -> serde_json::Value {
    let status = if stop_report_confirmed(&report) {
        "stopped"
    } else {
        "stop_unconfirmed"
    };
    serde_json::json!({
        "tool": SandboxToolArg::StopServer.as_cli_value(),
        "status": status,
        "mutating": true,
        "result": report,
    })
}

/// Whether a `stop_server` tool answer reports a stop it could not confirm.
pub(crate) fn service_stop_unconfirmed(value: &serde_json::Value) -> bool {
    let output = value.get("output").unwrap_or(value);
    output.get("status").and_then(serde_json::Value::as_str) == Some("stop_unconfirmed")
}

pub(crate) fn validate_service_id(service_id: &str) -> Result<()> {
    // Single source of truth for what makes an id safe as a filesystem path
    // component (rejects empty, path separators, `..`, control chars).
    rocm_core::ServiceId::new(service_id)?;
    Ok(())
}

fn list_log_files(dir: &Path, limit: usize) -> Vec<PathBuf> {
    if limit == 0 || !dir.is_dir() {
        return Vec::new();
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut paths = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("log"))
        .collect::<Vec<_>>();
    paths.sort();
    paths.into_iter().take(limit).collect()
}

pub(crate) fn read_optional_tail_lines(path: &Path, limit: usize, label: &str) -> Vec<String> {
    if !path.is_file() {
        return Vec::new();
    }
    match read_tail_lines(path, limit, label) {
        Ok(lines) => lines,
        Err(error) => vec![format!("<failed to read {label}: {error}>")],
    }
}

fn format_cli_lifecycle_tail_line(line: &str) -> String {
    let level = lifecycle_field(line, "level").unwrap_or("info");
    let category = lifecycle_field(line, "category").unwrap_or("cli");
    let action = lifecycle_field(line, "action").unwrap_or("event");
    let message = line
        .split_once(" message=")
        .map_or(line, |(_, message)| message)
        .trim();
    let label = lifecycle_event_label(category, action);
    if level == "info" {
        format!("{label}: {message}")
    } else {
        format!("{label} ({level}): {message}")
    }
}

fn lifecycle_event_label(category: &str, action: &str) -> String {
    match action {
        "runtime_activate" => "Runtime changed".to_owned(),
        "runtime_import" | "runtime_adopt" => "Runtime added".to_owned(),
        "runtime_uninstall" => "Runtime removed".to_owned(),
        "runtime_update" | "update_apply" | "update_check" => "Update check".to_owned(),
        "install_sdk" | "install_driver" => "Install".to_owned(),
        "engine_install" | "engine_switch" => "Engine changed".to_owned(),
        "service_start" | "service_stop" | "service_restart" | "serve" => {
            "Service event".to_owned()
        }
        "automation" | "watcher" => "Automation".to_owned(),
        "event" => humanize_log_label(category),
        other => humanize_log_label(other),
    }
}

fn humanize_log_label(value: &str) -> String {
    let mut words = value
        .split(['_', '-', '/'])
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    if words.is_empty() {
        return "Activity".to_owned();
    }
    let first = words.remove(0);
    let mut output = String::new();
    let mut chars = first.chars();
    if let Some(ch) = chars.next() {
        output.extend(ch.to_uppercase());
        output.push_str(chars.as_str());
    }
    for word in words {
        output.push(' ');
        output.push_str(word);
    }
    output
}

fn lifecycle_field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let prefix = format!("{key}=");
    line.split_whitespace()
        .find_map(|part| part.strip_prefix(&prefix))
        .filter(|value| !value.is_empty())
}

pub(crate) fn read_tail_lines(path: &Path, limit: usize, label: &str) -> Result<Vec<String>> {
    if limit == 0 {
        return Ok(Vec::new());
    }

    let file = fs::File::open(path)
        .with_context(|| format!("failed to open {label} {}", path.display()))?;
    let reader = io::BufReader::new(file);
    let mut lines = VecDeque::with_capacity(limit);
    for line in reader.lines() {
        let line = line.with_context(|| format!("failed to read {label} {}", path.display()))?;
        if lines.len() == limit {
            lines.pop_front();
        }
        lines.push_back(line);
    }
    Ok(lines.into_iter().collect())
}
#[cfg(test)]
pub(crate) mod tests {
    use std::thread;
    use std::time::Duration;

    use super::*;
    use crate::providers;
    use crate::service_control::run_internal_sandbox_tool;
    use crate::tests::test_paths;

    #[test]
    fn render_logs_text_preserves_directory_summary() {
        let (_root, paths) = test_paths("logs-summary");
        let rendered = render_logs_text(&paths);

        assert!(rendered.contains("Logs"));
        assert!(rendered.contains("File locations: shown"));
        assert!(rendered.contains(&format!(
            "  Folder: {}",
            paths.data_dir.join("logs").display()
        )));
        assert!(rendered.contains(&format!(
            "  Activity log: {}",
            cli_lifecycle_log_path(&paths).display()
        )));
        assert!(rendered.contains("  Command logs:"));
        assert!(rendered.contains("  Screen command logs:"));
        assert!(rendered.contains(&format!(
            "  Audit events: {}",
            paths.audit_events_path().display()
        )));
        assert!(rendered.contains("  Recent command files: none yet"));
        assert!(rendered.contains("Recent activity: no activity yet"));
        assert!(rendered.contains("Matching lines"));
        assert!(rendered.contains("  Search: none"));
        assert!(rendered.contains("  No logs found yet."));
    }

    #[test]
    fn render_logs_text_lists_action_logs_and_recent_lifecycle_tail() -> Result<()> {
        let (root, paths) = test_paths("logs-navigation");
        fs::create_dir_all(paths.data_dir.join("logs").join("cli"))?;
        fs::write(
            cli_lifecycle_log_path(&paths),
            (0..10).fold(String::new(), |mut acc, index| {
                let _ = writeln!(
                    acc,
                    "{index} level=info category=runtime action=install_sdk message=event-{index}"
                );
                acc
            }),
        )?;
        fs::write(
            paths
                .data_dir
                .join("logs")
                .join("cli")
                .join("runtime-install_sdk.log"),
            "install event\n",
        )?;
        fs::write(
            paths
                .data_dir
                .join("logs")
                .join("cli")
                .join("update-update_check.log"),
            "update event\n",
        )?;

        let rendered = render_logs_text(&paths);

        assert!(rendered.contains("  Recent command files:"));
        assert!(rendered.contains("runtime-install_sdk.log"));
        assert!(rendered.contains("update-update_check.log"));
        assert!(rendered.contains("Recent activity: last 8 line(s)"));
        assert!(!rendered.contains("event-0"));
        assert!(!rendered.contains("event-1"));
        assert!(rendered.contains("Install: event-2"));
        assert!(rendered.contains("event-2"));
        assert!(rendered.contains("event-9"));
        assert!(rendered.contains("  Lines: 10 of 10 recent line(s)"));
        assert!(rendered.contains("    command log runtime-install_sdk.log: install event"));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn render_logs_text_lists_screen_command_logs() -> Result<()> {
        let (root, paths) = test_paths("logs-screen-command");
        let screen_dir = paths.data_dir.join("logs").join("tui");
        fs::create_dir_all(&screen_dir)?;
        let screen_log = screen_dir.join("12345-install-the-rock-sdk.log");
        fs::write(
            &screen_log,
            "title: Install TheRock SDK\n\
             recent_live_output:\n\
             Output: resolving torch wheels\n\
             command_output:\n\
             stdout:\n\
             resolved torch\n",
        )?;

        let rendered = render_logs_text(&paths);

        assert!(rendered.contains("  Screen command logs:"));
        assert!(rendered.contains("screen/12345-install-the-rock-sdk.log"));
        assert!(rendered.contains("screen command log 12345-install-the-rock-sdk.log"));
        assert!(rendered.contains("Output: resolving torch wheels"));
        assert!(rendered.contains("  Lines: 6 of 6 recent line(s)"));
        let filtered = render_logs_browser_text(&paths, Some("torch wheels"));
        assert!(filtered.contains("Search: torch wheels"));
        assert!(filtered.contains("Output: resolving torch wheels"));
        assert!(filtered.contains("screen command log 12345-install-the-rock-sdk.log"));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn render_logs_browser_text_filters_lifecycle_and_action_logs() -> Result<()> {
        let (root, paths) = test_paths("logs-browser-search");
        fs::create_dir_all(paths.data_dir.join("logs").join("cli"))?;
        fs::write(
            cli_lifecycle_log_path(&paths),
            "1 level=info category=runtime action=install_sdk message=installed sdk\n\
             2 level=info category=service action=serve message=server ready\n",
        )?;
        fs::write(
            paths
                .data_dir
                .join("logs")
                .join("cli")
                .join("service-serve.log"),
            "server ready\nmodel warmed\n",
        )?;

        let rendered = render_logs_browser_text(&paths, Some("server"));

        assert!(rendered.contains("  Search: server"));
        assert!(rendered.contains("  Lines: 2 of 4 recent line(s)"));
        assert!(rendered.contains("    recent activity: Service event: server ready"));
        assert!(rendered.contains("    command log service-serve.log: server ready"));
        assert!(!rendered.contains("installed sdk"));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn render_logs_browser_page_text_paginates_matching_lines() -> Result<()> {
        let (root, paths) = test_paths("logs-browser-pages");
        let action_dir = paths.data_dir.join("logs").join("cli");
        fs::create_dir_all(&action_dir)?;
        fs::write(action_dir.join("a.log"), "alpha-1\nalpha-2\nalpha-3\n")?;
        fs::write(action_dir.join("b.log"), "alpha-4\nalpha-5\nalpha-6\n")?;

        let rendered = render_logs_browser_page_text(&paths, Some("alpha"), 1, 4);

        assert!(rendered.contains("  Page: 2 of 2"));
        assert!(rendered.contains("  Showing: 5-6 of 6"));
        assert!(!rendered.contains("alpha-1"));
        assert!(rendered.contains("alpha-5"));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn cli_lifecycle_tail_lines_render_compactly() {
        let rendered = format_cli_lifecycle_tail_line(
            "42 level=error category=runtime action=install_sdk service_id=<none> message=line one",
        );

        assert_eq!(rendered, "Install (error): line one");
    }

    /// `logs`, `stop` and `restart` for an id with no record all fail with the
    /// shared not-found error alone — one plain line, no OS cause under it.
    /// `remove` and `rocm logs --service` reach the same `load_managed_service`.
    #[test]
    fn service_commands_report_an_unknown_id_without_an_os_cause() -> Result<()> {
        let (root, paths) = test_paths("service-unknown-id");
        paths.ensure()?;
        let expected = "No local server named `nope`. See `rocm services list --all`.";
        let errors = [
            render_service_logs_text(&paths, "nope").map(drop),
            run_internal_sandbox_tool(
                &paths,
                SandboxToolArg::StopServer,
                Some("nope".to_owned()),
                true,
            )
            .map(drop),
            run_internal_sandbox_tool(
                &paths,
                SandboxToolArg::RestartServer,
                Some("nope".to_owned()),
                true,
            )
            .map(drop),
        ];
        for error in errors {
            let error = error.expect_err("no record exists for `nope`");
            assert!(
                error
                    .downcast_ref::<rocm_core::ManagedServiceNotFound>()
                    .is_some(),
                "{error:?}"
            );
            // anyhow appends a backtrace when RUST_BACKTRACE is set, as CI does.
            let rendered = format!("{error:?}");
            let rendered = rendered
                .split_once("\n\nStack backtrace:")
                .map_or(rendered.as_str(), |(message, _)| message);
            assert_eq!(rendered, expected);
        }
        assert!(!paths.service_manifest_path("nope").exists());
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn render_service_logs_text_tails_manifest_log() -> Result<()> {
        let (root, paths) = test_paths("service-logs");
        paths.ensure()?;

        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc_qwen35_primary",
            "vllm",
            "qwen3.5",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            std::process::id(),
            Some("therock-release".to_owned()),
            None,
            Some("gpu_preferred".to_owned()),
        );
        record.status = "ready".to_owned();
        record.write()?;

        let mut log = String::new();
        for index in 1..=90 {
            let _ = writeln!(log, "entry-{index:03}");
        }
        fs::write(&record.log_path, log)?;

        let rendered = render_service_logs_text(&paths, "svc_qwen35_primary")?;
        assert!(rendered.contains("Service Log"));
        assert!(rendered.contains("Service: svc_qwen35_primary"));
        assert!(rendered.contains("Engine: vllm"));
        assert!(rendered.contains("Status: starting"));
        assert!(rendered.contains("File locations: shown"));
        assert!(rendered.contains(&format!(
            "  Details file: {}",
            record.manifest_path.display()
        )));
        assert!(rendered.contains(&format!("  Log file: {}", record.log_path.display())));
        assert!(!rendered.contains("entry-010"));
        assert!(rendered.contains("entry-011"));
        assert!(rendered.contains("entry-090"));

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn render_services_text_lists_live_services_by_default_and_all_on_request() -> Result<()> {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let (root, paths) = test_paths("services-list");
        paths.ensure()?;
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let ready_port = listener.local_addr()?.port();
        // The ready service is listed and then asked to complete a request:
        // readiness is not granted on the model listing alone.
        let server = thread::spawn(move || -> Result<()> {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept()?;
                stream.set_read_timeout(Some(Duration::from_secs(2)))?;
                let mut request = [0_u8; 512];
                let read = stream.read(&mut request)?;
                let request = String::from_utf8_lossy(&request[..read]).into_owned();
                let body = if request.starts_with("POST /v1/chat/completions ") {
                    r#"{"choices":[{"message":{"content":"ok"}}]}"#
                } else {
                    r#"{"data":[{"id":"Qwen/Qwen3.5"}]}"#
                };
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )?;
            }
            Ok(())
        });
        let current_pid = std::process::id();

        for (service_id, status, port) in [
            ("svc-ready", "ready", ready_port),
            ("svc-starting", "starting", 11436_u16),
            ("svc-failed", "failed", 11437_u16),
        ] {
            let mut record = ManagedServiceRecord::new(
                &paths,
                service_id,
                "vllm",
                "qwen",
                "Qwen/Qwen3.5",
                "127.0.0.1",
                port,
                "managed",
                current_pid,
                Some("therock-release".to_owned()),
                None,
                Some("gpu_required".to_owned()),
            );
            record.status = status.to_owned();
            record.write()?;
        }

        let rendered = render_services_text(&paths, false)?;
        server
            .join()
            .expect("fake models server should not panic")?;
        let all = render_services_text(&paths, true)?;
        let _ = fs::remove_dir_all(root);

        assert!(rendered.contains("Local Servers"));
        // The header counts the whole registry, so the hidden `failed` record
        // is admitted to rather than silently dropped from the default view.
        assert!(
            rendered.contains("Status: 1 ready, 1 starting, 1 not running"),
            "{rendered}"
        );
        assert!(
            rendered
                .contains("Past attempts: 1 local server record(s) that are no longer running."),
            "{rendered}"
        );
        assert!(
            rendered.contains("  Read the newest: rocm services logs svc-failed"),
            "{rendered}"
        );
        assert!(rendered.contains("- svc-ready"));
        assert!(rendered.contains("  stop: rocm services stop svc-ready --yes"));
        assert!(!rendered.contains("- svc-failed"));
        assert!(all.contains("- svc-failed"));
        assert!(all.contains("  restart: rocm services restart svc-failed --yes"));
        assert!(!rendered.contains("running servers"));
        Ok(())
    }

    /// The default view lists only live servers. The counts used to be taken
    /// from that already-filtered list, so a host whose local servers had all
    /// failed was told "none ready" and shown an empty list - the records were
    /// on disk, unmentioned, and nothing on screen said how to reach them.
    /// Counts now come from the unfiltered registry; the rows stay filtered.
    #[test]
    fn render_services_text_surfaces_records_the_default_view_hides() -> Result<()> {
        let (root, paths) = test_paths("services-past-attempts-only");
        paths.ensure()?;
        // Not live, so `refresh_managed_service_runtime_liveness` returns before
        // any endpoint probe or pid check: no network, no timing dependence.
        for (service_id, created_at) in [("svc-older", 1_000_u128), ("svc-newest", 2_000_u128)] {
            let mut record = ManagedServiceRecord::new(
                &paths,
                service_id,
                "vllm",
                "qwen",
                "Qwen/Qwen3.5",
                "127.0.0.1",
                11500,
                "managed",
                999_999_999,
                None,
                None,
                None,
            );
            record.status = "failed".to_owned();
            record.created_at_unix_ms = created_at;
            record.write()?;
        }

        let rendered = render_services_text(&paths, false)?;
        let all = render_services_text(&paths, true)?;
        let _ = fs::remove_dir_all(root);

        assert!(rendered.contains("Status: 2 not running"), "{rendered}");
        assert!(!rendered.contains("none ready"), "{rendered}");
        assert!(
            rendered.contains("No local servers are running."),
            "{rendered}"
        );
        assert!(
            rendered
                .contains("Past attempts: 2 local server record(s) that are no longer running."),
            "{rendered}"
        );
        assert!(
            rendered.contains("  See them: rocm services list --all"),
            "{rendered}"
        );
        // A real id, newest first, so the line can be pasted as-is.
        assert!(
            rendered.contains("  Read the newest: rocm services logs svc-newest"),
            "{rendered}"
        );
        // The other two lines only let the user look. Each record keeps an
        // unrotated engine log, so `prune` is the only line here that gets any
        // of that space back - without it the block is a dead end on exactly
        // the host that needs it most.
        assert!(
            rendered.contains("  Reclaim the space: rocm services prune"),
            "{rendered}"
        );
        // The `--all` header is fixed by the same change; it lists the rows, so
        // it does not repeat the pointer.
        assert!(all.contains("Status: 2 not running"), "{all}");
        assert!(all.contains("- svc-newest"), "{all}");
        assert!(!all.contains("Past attempts:"), "{all}");
        Ok(())
    }

    /// Both `rocm services logs <id>` lines the list prints — the per-row
    /// `logs:` line and the default view's "Read the newest" pointer — are
    /// commands a user is told to paste, so each must split back to exactly
    /// that argv whatever the id holds. Mutation: interpolating the id bare in
    /// either line fails this.
    #[test]
    fn render_services_text_logs_hints_denote_the_argv_they_name() -> Result<()> {
        let service_id = "svc 'a b'";
        let (root, paths) = test_paths("services-logs-hint-awkward-id");
        paths.ensure()?;
        // Not live, so no endpoint probe or pid check runs.
        let mut record = ManagedServiceRecord::new(
            &paths,
            service_id,
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11500,
            "managed",
            999_999_999,
            None,
            None,
            None,
        );
        record.status = "failed".to_owned();
        record.write()?;

        let default_view = render_services_text(&paths, false);
        let all_view = render_services_text(&paths, true);
        let _ = fs::remove_dir_all(root);
        let (default_view, all_view) = (default_view?, all_view?);

        let expected = Some(
            ["rocm", "services", "logs", service_id]
                .map(str::to_owned)
                .to_vec(),
        );
        let newest = default_view
            .lines()
            .find_map(|line| line.strip_prefix("  Read the newest: "))
            .expect("the default view points at the newest past attempt");
        assert_eq!(shlex::split(newest), expected, "{default_view}");
        let row = all_view
            .lines()
            .find_map(|line| line.strip_prefix("  logs: "))
            .expect("the --all view lists the record with its logs command");
        assert_eq!(shlex::split(row), expected, "{all_view}");
        Ok(())
    }

    /// The host this change is NOT meant to touch: nothing has ever been served,
    /// so there is no record to point at and the output must read exactly as it
    /// did before. Every other test here plants at least one record, so without
    /// this one the zero case has no coverage at all - and it is the case the
    /// hint's separator regressed, by printing a blank line the old output never
    /// had between "No local servers are running." and the next line.
    #[test]
    fn render_services_text_leaves_a_host_that_never_served_untouched() -> Result<()> {
        let (root, paths) = test_paths("services-never-served");
        paths.ensure()?;

        let rendered = render_services_text(&paths, false)?;
        let _ = fs::remove_dir_all(root);

        assert_eq!(
            rendered,
            "Local Servers\n\
             \n\
             Status: none ready\n\
             \n\
             No local servers are running.\n\
             Start one with `rocm serve <model> --managed`, or run `rocm` and choose Serve.\n",
            "{rendered}"
        );
        Ok(())
    }

    /// A host can have one server running and several that failed, so the
    /// populated branch of the default view needs the same pointer as the empty
    /// one - the failed records are hidden there too.
    #[test]
    fn render_services_text_points_at_past_attempts_beside_a_live_server() -> Result<()> {
        let (root, paths) = test_paths("services-past-attempts-mixed");
        paths.ensure()?;
        let current_pid = std::process::id();
        // `starting` skips the endpoint probe and this process is a
        // guaranteed-live pid, so the record stays live without any network.
        let mut live = ManagedServiceRecord::new(
            &paths,
            "svc-live",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11501,
            "managed",
            current_pid,
            None,
            None,
            None,
        );
        live.status = "starting".to_owned();
        live.engine_pid = Some(current_pid);
        live.created_at_unix_ms = 3_000;
        live.write()?;
        for (service_id, created_at) in [
            ("svc-failed-older", 1_000_u128),
            ("svc-failed-newest", 2_000_u128),
        ] {
            let mut record = ManagedServiceRecord::new(
                &paths,
                service_id,
                "vllm",
                "qwen",
                "Qwen/Qwen3.5",
                "127.0.0.1",
                11502,
                "managed",
                999_999_999,
                None,
                None,
                None,
            );
            record.status = "failed".to_owned();
            record.created_at_unix_ms = created_at;
            record.write()?;
        }

        let rendered = render_services_text(&paths, false)?;
        let _ = fs::remove_dir_all(root);

        assert!(
            rendered.contains("Status: 1 starting, 2 not running"),
            "{rendered}"
        );
        // Rows stay filtered: only the live server is listed.
        assert!(rendered.contains("- svc-live"), "{rendered}");
        assert!(!rendered.contains("- svc-failed-newest"), "{rendered}");
        assert!(
            rendered
                .contains("Past attempts: 2 local server record(s) that are no longer running."),
            "{rendered}"
        );
        assert!(
            rendered.contains("  See them: rocm services list --all"),
            "{rendered}"
        );
        assert!(
            rendered.contains("  Read the newest: rocm services logs svc-failed-newest"),
            "{rendered}"
        );
        Ok(())
    }

    #[test]
    fn services_json_round_trips_and_applies_the_same_liveness_filter_as_the_table() -> Result<()> {
        // This JSON is a contract, not a convenience: `rocm remote` parses it
        // back over its control channel to learn which service a remote serve
        // just started. Two things have to hold — every record survives the
        // round trip, and `--json` agrees with the table about what is live. If
        // they disagreed, the remote orchestration would act on a different set
        // of services than the operator sees.
        let (root, paths) = test_paths("services-json");
        paths.ensure()?;
        let current_pid = std::process::id();
        for (service_id, status, port) in [
            ("svc-live", "starting", 11440_u16),
            ("svc-past", "failed", 11441_u16),
        ] {
            let mut record = ManagedServiceRecord::new(
                &paths,
                service_id,
                "vllm",
                "qwen",
                "Qwen/Qwen3.5",
                "127.0.0.1",
                port,
                "managed",
                current_pid,
                Some("therock-release".to_owned()),
                None,
                Some("gpu_required".to_owned()),
            );
            record.status = status.to_owned();
            record.write()?;
        }

        let live = render_services_json(&paths, false)?;
        let every = render_services_json(&paths, true)?;
        let _ = fs::remove_dir_all(root);

        let live: Vec<ManagedServiceRecord> = serde_json::from_str(&live)?;
        let every: Vec<ManagedServiceRecord> = serde_json::from_str(&every)?;

        assert_eq!(
            live.iter()
                .map(|r| r.service_id.as_str())
                .collect::<Vec<_>>(),
            vec!["svc-live"],
            "the default listing must hide past attempts, exactly as the table does"
        );
        let mut every_ids = every
            .iter()
            .map(|r| r.service_id.as_str())
            .collect::<Vec<_>>();
        every_ids.sort_unstable();
        assert_eq!(every_ids, vec!["svc-live", "svc-past"]);

        // The fields remote orchestration actually reads must survive intact.
        let record = &live[0];
        assert_eq!(record.port, 11440);
        assert_eq!(record.status, "starting");
        // Note for remote orchestration: the recorded endpoint is already the
        // OpenAI-compatible base, `/v1` suffix included — not a bare origin.
        assert_eq!(record.endpoint_url, "http://127.0.0.1:11440/v1");
        assert_eq!(record.canonical_model_id, "Qwen/Qwen3.5");
        Ok(())
    }

    #[test]
    fn render_services_text_demotes_stale_ready_record() -> Result<()> {
        let (root, paths) = test_paths("services-stale-ready");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-stale-ready",
            "lemonade",
            "qwen",
            providers::BUILTIN_ASSISTANT_MODEL_ID,
            "127.0.0.1",
            9,
            "managed",
            999_999_999,
            Some("therock-release".to_owned()),
            None,
            Some("gpu_required".to_owned()),
        );
        record.status = "ready".to_owned();
        record.engine_pid = Some(999_999_999);
        record.write()?;

        let rendered = render_services_text(&paths, false)?;
        let all = render_services_text(&paths, true)?;
        let reloaded = load_managed_service(&paths, "svc-stale-ready")?;
        let _ = fs::remove_dir_all(root);

        assert!(rendered.contains("No local servers are running."));
        // The transition this test is named for reaches the header too: once the
        // record is demoted it is a past attempt, so the header says so instead
        // of the old "none ready" - which described the live set correctly and
        // the disk misleadingly.
        assert!(rendered.contains("Status: 1 not running"), "{rendered}");
        assert!(all.contains("- svc-stale-ready"));
        assert!(all.contains("  status: stopped"));
        assert_eq!(reloaded.status, "stopped");
        Ok(())
    }

    /// No line of an unconfirmed-stop report offers a command to run the stop
    /// again — no `retry:` line, and no `services stop` anywhere: a second stop
    /// can confirm without reaching a surviving process under the recorded one.
    pub(crate) fn assert_no_retry_line(rendered: &str) {
        assert!(
            !rendered.lines().any(|line| {
                line.trim_start().starts_with("retry:") || line.contains("services stop")
            }),
            "an unconfirmed stop names no stop command: {rendered}"
        );
    }

    #[test]
    fn services_stop_reports_a_report_without_a_verdict_as_unconfirmed() {
        let answer = internal_stop_server_value(serde_json::json!({
            "service_id": "svc-legacy",
            "status": "ready",
            "signaled_pids": [],
        }));
        assert_eq!(answer["status"], "stop_unconfirmed", "{answer}");
    }
}
