// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Local-server log rendering, the `rocm services` status/log surfaces, and
//! the internal sandbox-tool execution (stop/restart/list) behind the
//! chat/LLM tool-calling path and the daemon-free managed-service lifecycle.
//!
//! Mechanically relocated from `main.rs` with no behavior change — Phase 6f
//! (ROCMAI-209) of the `main.rs` modularization (see `docs/architecture.md`).
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
//! `load_managed_service`), the internal `rocmd`-compatible status/sandbox
//! surface (`render_internal_status_text`, `run_internal_sandbox_tool`), and
//! the managed-service stop/restart implementation behind both the sandbox
//! tool and `rocm services stop|restart` (`terminate_recorded_service_pids`,
//! the `MANAGED_STOP_GRACE` constant, `stop_internal_managed_service`,
//! `stop_internal_managed_service_with`,
//! `stop_internal_managed_service_confirmed_with`,
//! `unload_lemonade_service_model`, `apply_restart_runtime_pin`,
//! `load_service_for_restart`, `restart_child_argv`,
//! `restart_internal_managed_service`, `restart_internal_managed_service_with`,
//! `update_restarting_record`, `record_restarted_child`,
//! `record_restart_readiness`), plus `validate_service_id`, `list_log_files`,
//! `read_optional_tail_lines`, and the CLI-lifecycle-log tail formatting
//! helpers (`format_cli_lifecycle_tail_line`, `lifecycle_event_label`,
//! `humanize_log_label`, `lifecycle_field`, `read_tail_lines`).
//!
//! Two items physically interleaved in this span in `main.rs` moved with it
//! rather than staying behind, since neither has a caller outside this
//! cluster: the `LogBrowserEntry` struct (used only by
//! `collect_log_browser_entries`/`filter_log_browser_entries`) and the
//! `MANAGED_STOP_GRACE` constant (used only by
//! `stop_internal_managed_service_confirmed_with`). Both stay private.
//!
//! `dispatch()` calls directly into this module at several points —
//! `render_internal_status_text`, `run_internal_sandbox_tool`,
//! `render_service_logs_text`, `render_logs_browser_text`, `render_logs_text`,
//! `render_services_text`, and `render_services_json` — re-imported via `use
//! crate::services_text::{...};`. `render_service_action_result`,
//! `run_internal_sandbox_tool`, and `validate_service_id` are also read from
//! `main.rs`'s own `run_approved_service_action` (dispatch-adjacent, but not
//! itself part of this cluster — see `service_launch.rs`'s doc comment for
//! why the broader service-record-management cluster stays at the crate
//! root); `validate_service_id` and `load_managed_service` are also read from
//! `main.rs`'s `remove_managed_service_record`/`apply_service_prune_plan`.
//! `load_managed_service` and `restart_internal_managed_service` are
//! `pub(crate)` and read from `runtime_services.rs` (reconciling live servers
//! against a runtime activation) and from `managed_service_status.rs`'s own
//! test module; `read_tail_lines`, `render_logs_text`, `render_logs_browser_text`,
//! `render_services_text`, `render_service_logs_text`, and
//! `validate_service_id` are also `pub(crate)` and read from `chat.rs`'s
//! deterministic tool-output summaries; `read_optional_tail_lines` and
//! `render_service_action_result` are also `pub(crate)` and read from
//! `service_launch.rs`'s managed-spawn/attach path. Every other item in this
//! list has no caller outside this module and stays private — including
//! `restart_internal_managed_service_with`, `restart_child_argv`,
//! `load_service_for_restart`, `apply_restart_runtime_pin`,
//! `unload_lemonade_service_model`, `record_restarted_child`, and
//! `record_restart_readiness`: only this module's own wrappers and tests call
//! them, even though `main.rs`'s doc comments on sibling modules
//! (`engine_runtime_select.rs`, `gpu_policy.rs`) still name them as callers of
//! *those* modules' functions.
//!
//! The cluster's regression tests live in this module's own `mod tests`,
//! reaching shared fixtures (`test_paths`, `plant_service_record_on_runtime`,
//! `OLD_RUNTIME_KEY`, `NEW_RUNTIME_KEY`) through `crate::tests::` the same way
//! other mechanically relocated modules do, and `runtime_services::` through
//! its own `use crate::runtime_services;` for the handful of tests that drive
//! this cluster's restart/stop behavior through `runtime_services`'s public
//! entry points rather than calling it directly. Three tests that call this
//! cluster's `load_managed_service`/`render_service_action_result` only
//! incidentally — their primary subject is `runtime_services::
//! pin_service_record_to_runtime`/`restore_service_record_pin`,
//! `runtime_services::restart_service_onto_runtime_with`'s outcome
//! classification, and `main.rs`'s own `run_approved_service_action`
//! respectively — stay in `main.rs`'s shared `mod tests` instead.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};
#[cfg(not(windows))]
use std::process::Stdio;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, EndpointReadiness, ManagedServiceRecord};
use rocm_engine_protocol::{DEFAULT_LOG_TAIL_LINES, EngineMethod, StopRequest, StopResponse};

use crate::endpoint_keys;
use crate::engine_ipc::{builtin_engine_serve_http_args, engine_request};
#[cfg(not(windows))]
use crate::engine_runtime_select::apply_app_path_env;
#[cfg(windows)]
use crate::engine_runtime_select::{app_path_env_var_refs, app_path_env_var_values};
use crate::engine_runtime_select::{
    http_post_local_service_json, managed_service_launcher_path, parse_engine_recipe_json_arg,
    wait_for_service_http_ready,
};
use crate::engines_cmd::env_root_for_service;
use crate::gpu_policy::parse_device_policy;
use crate::managed_service_status::{
    load_managed_services, local_server_sidebar_status, managed_service_is_live,
    managed_service_sidebar_counts, refresh_managed_service_runtime_liveness,
};
use crate::service_hints;
use crate::service_launch::ensure_public_service_has_endpoint_key;
#[cfg(not(windows))]
use crate::service_launch::{
    attach_background_stdio, detach_background_command, managed_engine_startup_failure_detail,
    managed_service_process_command,
};
use crate::{
    SandboxToolArg, cli_lifecycle_log_path, service_action_past_tense, status_for_readiness,
};

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
        let _ = writeln!(output, "  Read the newest: rocm services logs {id}");
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
        let _ = writeln!(output, "  logs: rocm services logs {}", record.service_id);
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

pub(crate) fn render_service_action_result(tool: &str, value: &serde_json::Value) -> String {
    let output = value.get("output").unwrap_or(value);
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

/// A managed service's record exactly as it is on disk, without the engine-state
/// and liveness refresh [`load_managed_service`] applies (and may persist).
fn read_managed_service_manifest(
    paths: &AppPaths,
    service_id: &str,
) -> Result<(ManagedServiceRecord, PathBuf)> {
    validate_service_id(service_id)?;
    let manifest_path = paths.service_manifest_path(service_id);
    let bytes = fs::read(&manifest_path).with_context(|| {
        format!(
            "managed service `{service_id}` not found at {}",
            manifest_path.display()
        )
    })?;
    let mut record = serde_json::from_slice::<ManagedServiceRecord>(&bytes)
        .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
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

pub(crate) fn run_internal_sandbox_tool(
    paths: &AppPaths,
    tool: SandboxToolArg,
    service_id: Option<String>,
    allow_native_fallback: bool,
) -> Result<serde_json::Value> {
    if !allow_native_fallback {
        bail!(
            "isolated sandbox runner is unavailable in the single-binary build; pass --allow-native-fallback to run the restricted internal tool API"
        );
    }
    let output = match tool {
        SandboxToolArg::ListServers => {
            let services = load_managed_services(paths)?;
            serde_json::json!({
                "tool": tool.as_cli_value(),
                "status": "listed",
                "mutating": false,
                "count": services.len(),
                "services": services,
            })
        }
        SandboxToolArg::StopServer => {
            let service_id = service_id.context("stop_server requires --service-id")?;
            let result = stop_internal_managed_service(paths, &service_id)?;
            serde_json::json!({
                "tool": tool.as_cli_value(),
                "status": "stopped",
                "mutating": true,
                "result": result,
            })
        }
        SandboxToolArg::RestartServer => {
            let service_id = service_id.context("restart_server requires --service-id")?;
            let service = restart_internal_managed_service(paths, &service_id, None)?;
            serde_json::json!({
                "tool": tool.as_cli_value(),
                "status": "restarted",
                "mutating": true,
                "service": service,
            })
        }
    };
    Ok(serde_json::json!({
        "protocol": "rocmd-sandbox-run-v0",
        "tool": tool.as_cli_value(),
        "ok": true,
        "ok_meaning": "sandbox wrapper completed; inspect output.status for the restricted tool result",
        "isolation": "native_restricted",
        "output": output,
    }))
}

/// How long a managed stop waits for each recorded process to actually exit
/// after `SIGTERM` before escalating to `SIGKILL` and then reporting a timeout.
const MANAGED_STOP_GRACE: Duration = Duration::from_secs(10);

/// Terminate the processes recorded for a managed service, verifying each PID's
/// identity first so a recycled PID never causes an unrelated process tree to be
/// killed.
///
/// The supervisor (launcher) and the engine server are distinct processes, so
/// each PID is paired with **its own** start-time token (`supervisor_start_ticks`
/// vs `engine_start_ticks`). When they resolve to the same PID (e.g. before the
/// engine state has been refreshed) the entry is de-duplicated, preferring a
/// known token. Returns the PIDs actually signalled and whether every recorded
/// process is confirmed no longer running.
fn terminate_recorded_service_pids(record: &ManagedServiceRecord) -> (Vec<u32>, bool) {
    // Build the (pid, own-token) work list, de-duplicating on PID and preferring
    // an entry that carries a verifiable start-time.
    let mut entries: Vec<(u32, Option<u64>)> = Vec::new();
    for (pid, ticks) in [
        (Some(record.supervisor_pid), record.supervisor_start_ticks),
        (record.engine_pid, record.engine_start_ticks),
    ] {
        let Some(pid) = pid.filter(|pid| *pid != 0 && *pid != std::process::id()) else {
            continue;
        };
        if let Some(existing) = entries.iter_mut().find(|(seen, _)| *seen == pid) {
            if existing.1.is_none() {
                existing.1 = ticks;
            }
        } else {
            entries.push((pid, ticks));
        }
    }

    let mut signaled_pids = Vec::new();
    let mut all_stopped = true;
    for (pid, ticks) in entries {
        let identity = rocm_core::ProcessIdentity::new(pid, ticks);
        // The managed stop contract is definitive: after the adapter gets its
        // graceful opportunity, this cleanup pass forces any verified survivor
        // so the command does not report success while a GPU worker remains.
        let outcome = rocm_core::terminate_verified(
            &identity,
            rocm_core::KillScope::Tree,
            MANAGED_STOP_GRACE,
            true,
        );
        if !outcome.stopped() {
            all_stopped = false;
        }
        // Report every PID that received a signal. TimedOut belongs here because
        // signalling was attempted even though the process was not confirmed
        // stopped; `all_stopped` separately carries the truthful completion state.
        if matches!(
            outcome,
            rocm_core::TerminationOutcome::Graceful
                | rocm_core::TerminationOutcome::Forced
                | rocm_core::TerminationOutcome::TimedOut
        ) {
            signaled_pids.push(pid);
        }
    }
    (signaled_pids, all_stopped)
}

fn stop_internal_managed_service(paths: &AppPaths, service_id: &str) -> Result<serde_json::Value> {
    stop_internal_managed_service_with(paths, service_id, terminate_recorded_service_pids)
}

/// [`stop_internal_managed_service`], with the termination of the recorded
/// processes supplied by the caller. Production passes
/// [`terminate_recorded_service_pids`]; a test passes a stand-in that observes
/// what is on disk at the moment the processes would be signalled. Everything
/// else — the engine stop, the manifest writes, the key cleanup — runs for real.
///
/// `pub(crate)` so `service_launch`'s tests can drive the real stop path too,
/// rather than writing a stopped status to disk themselves.
pub(crate) fn stop_internal_managed_service_with(
    paths: &AppPaths,
    service_id: &str,
    terminate: impl FnOnce(&ManagedServiceRecord) -> (Vec<u32>, bool),
) -> Result<serde_json::Value> {
    stop_internal_managed_service_confirmed_with(paths, service_id, terminate)
        .map(|(report, _)| report)
}

/// [`stop_internal_managed_service_with`], also returning whether the stop is
/// confirmed: every process it handled is gone and the record still names
/// them. The report's `status` cannot carry that on its own — a record that
/// already read `stopped` keeps that status through a stop that confirmed
/// nothing.
fn stop_internal_managed_service_confirmed_with(
    paths: &AppPaths,
    service_id: &str,
    terminate: impl FnOnce(&ManagedServiceRecord) -> (Vec<u32>, bool),
) -> Result<(serde_json::Value, bool)> {
    let mut stopping = load_managed_service(paths, service_id)?;
    // Record the request on disk before anything is stopped, not only after.
    // From the engine `Stop` below until the last recorded process is confirmed
    // gone, the record still reads `ready`/`running` while its endpoint stops
    // answering — exactly what the daemon's `server-recover` watcher restarts.
    // That watcher skips a record carrying this marker, so it has to be on disk
    // for the whole stop. Same order as `rocmd`'s `stop_managed_service_with`.
    let requested_at = rocm_core::unix_time_millis();
    stopping.stop_requested_unix_ms = Some(requested_at);
    stopping.write()?;
    let engine_stop = if stopping.engine == "lemonade" {
        unload_lemonade_service_model(&stopping).map(|()| StopResponse {
            stopped: true,
            graceful: true,
        })
    } else {
        engine_request::<_, StopResponse>(
            Some(paths),
            &stopping.engine,
            EngineMethod::Stop,
            &StopRequest {
                service_id: stopping.service_id.clone(),
                force: true,
            },
        )
    };
    let (signaled_pids, terminated_all) = terminate(&stopping);
    // Re-read the record rather than write back the copy loaded before the
    // stop, which can be some forty seconds old by now: `write` replaces the
    // whole file, and a supervisor, the daemon's restart or another stop may
    // have written in the meantime. Only this stop's own changes are applied to
    // what is there now — the same discipline as `rocmd`'s
    // `stop_managed_service_with`. Read as-is: `load_managed_service`'s
    // liveness refresh would act on this stop's own marker before the verdict
    // below has been reached.
    let (mut record, _) = read_managed_service_manifest(paths, service_id)?;
    // Only claim "stopped" when every recorded process is confirmed gone, and
    // only for the processes this stop handled. If the record names different
    // ones now — a restart recorded its new supervisor while the stop ran —
    // the verdict says nothing about them, and claiming the service stopped or
    // dropping the key it serves with would both be false.
    //
    // When a stop cannot confirm termination (rare: SIGKILL-resistant or
    // unverifiable PID), leave the status so the standard liveness refresh
    // reconciles it to "stopped" once the process actually dies — rather than
    // asserting a stop that did not happen.
    // A container engine is stopped by its engine `Stop`, not by the PID of the
    // supervisor that follows it, so a failed `Stop` leaves the stop unconfirmed.
    let engine_stop_confirms =
        !rocm_core::engine_stop_decides_confirmation(&stopping.engine) || engine_stop.is_ok();
    let all_stopped =
        terminated_all && stopping.names_same_processes(&record) && engine_stop_confirms;
    if all_stopped {
        record.status = "stopped".to_owned();
        record.stop_requested_unix_ms = None;
    } else {
        // The marker records that a stop was *asked for* even though it could
        // not be confirmed. `refresh_managed_service_runtime_liveness` needs it
        // to tell a service the operator stopped from one that merely crashed:
        // only the former should lose its endpoint key once its processes are
        // gone. Re-stamped, because a restart that ran meanwhile clears it.
        record.stop_requested_unix_ms = Some(requested_at);
    }
    record.write()?;
    // Drop the endpoint key with the service by deleting its 0600 key file.
    // Best-effort — a stopped service must not fail to stop just because key
    // cleanup did.
    //
    // Gated on `all_stopped` for the same reason the status is: an unconfirmed
    // stop may have left the engine alive and still enforcing the key, and
    // discarding our only copy would lock the CLI's own probes, chat, and
    // service discovery out of a service that is otherwise fine. The marker
    // written above hands that cleanup to the liveness refresh instead.
    if all_stopped {
        endpoint_keys::clear_endpoint_api_key(paths, &record.service_id);
    }
    let engine_stop = match engine_stop {
        Ok(response) => serde_json::json!({
            "attempted": true,
            "stopped": response.stopped,
            "graceful": response.graceful,
        }),
        Err(error) => serde_json::json!({
            "attempted": true,
            "error": error.to_string(),
        }),
    };
    let report = serde_json::json!({
        "service_id": service_id,
        "status": record.status,
        "engine_stop": engine_stop,
        "signaled_pids": signaled_pids,
    });
    Ok((report, all_stopped))
}

fn unload_lemonade_service_model(record: &ManagedServiceRecord) -> Result<()> {
    let body = serde_json::json!({
        "model_name": record.canonical_model_id,
    });
    let (status, response_body) = http_post_local_service_json(
        &record.host,
        record.port,
        "/v1/unload",
        &body,
        Duration::from_secs(5),
    )?;
    if status == 200 {
        thread::sleep(Duration::from_millis(500));
        Ok(())
    } else {
        bail!("lemonade unload returned HTTP {status}: {response_body}");
    }
}

/// Re-apply a caller's runtime pin to a record that was just loaded for a
/// restart, in the same shape `pin_service_record_to_runtime` writes it: the
/// exact runtime key, and no env pin.
///
/// Split out from [`restart_internal_managed_service`] so the pin can be
/// asserted without launching an engine — see
/// `pinning_the_record_is_what_puts_the_new_runtime_in_the_restart_argv`.
fn apply_restart_runtime_pin(record: &mut ManagedServiceRecord, runtime_pin: Option<&str>) {
    if let Some(runtime_key) = runtime_pin {
        record.runtime_id = Some(runtime_key.to_owned());
        record.env_id = None;
    }
}

/// Load the record a restart relaunches from, with the caller's runtime pin
/// applied after the load's own refresh.
///
/// Everything a restart can say about WHICH runtime the child comes back on is
/// decided here and in [`restart_child_argv`], and neither spawns anything, so
/// `restart_internal_managed_service` is only the stop, the spawn and the
/// readiness wait around them. That split is what lets a test reach the part of
/// a restart that matters for a move without an engine: dropping the pin from
/// this function, or from the caller that supplies it, changes the argv a test
/// reads off [`restart_child_argv`].
fn load_service_for_restart(
    paths: &AppPaths,
    service_id: &str,
    runtime_pin: Option<&str>,
) -> Result<ManagedServiceRecord> {
    let mut record = load_managed_service(paths, service_id)?;
    apply_restart_runtime_pin(&mut record, runtime_pin);
    Ok(record)
}

/// The argv a restarted child is launched with, read entirely off the record.
///
/// Every field comes from the record rather than from a parameter on purpose: a
/// restart's runtime is whatever the record says after
/// [`load_service_for_restart`], and an input here that bypassed it would let a
/// test assert a runtime the restart never uses.
fn restart_child_argv(record: &ManagedServiceRecord) -> Result<Vec<String>> {
    let policy = parse_device_policy(record.device_policy.as_deref())?;
    let recipe = parse_engine_recipe_json_arg(record.engine_recipe_json.clone())?;
    builtin_engine_serve_http_args(
        &record.engine,
        &record.service_id,
        &record.canonical_model_id,
        &record.host,
        record.port,
        &policy,
        &record.gpu_indices,
        record.runtime_id.as_deref(),
        record.env_id.as_deref(),
        recipe.as_ref(),
        &record.engine_state_path,
        Some(&record.log_path),
    )
}

/// Stop a managed service and bring it back up.
///
/// `runtime_pin` is for callers whose whole reason for restarting is to MOVE the
/// service — `runtime_services::restart_service_onto_runtime`. Passing the key
/// here is not a convenience over writing the record first: `load_managed_service`
/// below refreshes the record from the engine state file written by the process
/// being replaced, which re-adopts THAT process's `runtime_id` and `env_id`. For
/// a plain restart that is correct, since the record should describe what is
/// actually running. For a move it silently undoes the re-pin — the previous
/// `env_id` comes back, and `builtin_engine_serve_http_args` omits `--runtime-id`
/// whenever `env_id` is set, so the child would resolve a runtime for itself and
/// come back up on the one it was already using while the record claimed the new
/// key. `None` leaves the refreshed record exactly as it was.
pub(crate) fn restart_internal_managed_service(
    paths: &AppPaths,
    service_id: &str,
    runtime_pin: Option<&str>,
) -> Result<ManagedServiceRecord> {
    restart_internal_managed_service_with(
        paths,
        service_id,
        runtime_pin,
        terminate_recorded_service_pids,
    )
}

/// [`restart_internal_managed_service`], with the termination of the recorded
/// processes supplied by the caller — see [`stop_internal_managed_service_with`].
///
/// The respawn is serialized on the same cross-process launch lock a fresh
/// `rocm serve` launch and `rocmd` recovery both use: the confirmed stop just
/// released this service's (host, port), and nothing reserves it again until
/// the new child binds and its record is published, so a concurrent launch or
/// recovery could otherwise claim the exact port this restart is about to
/// reuse.
fn restart_internal_managed_service_with(
    paths: &AppPaths,
    service_id: &str,
    runtime_pin: Option<&str>,
    terminate: impl FnOnce(&ManagedServiceRecord) -> (Vec<u32>, bool),
) -> Result<ManagedServiceRecord> {
    restart_internal_managed_service_inner(paths, service_id, runtime_pin, terminate, None)
}

/// Replaces the child `spawn()` of a restart (non-Windows only; Windows spawns
/// through `rocm_core::spawn_detached_no_inherit`). Test-only seam, so a test
/// can observe the leased port at the exact instant production would spawn.
type RestartSpawnOverride<'a> =
    dyn Fn(&mut std::process::Command) -> io::Result<std::process::Child> + 'a;

fn restart_internal_managed_service_inner(
    paths: &AppPaths,
    service_id: &str,
    runtime_pin: Option<&str>,
    terminate: impl FnOnce(&ManagedServiceRecord) -> (Vec<u32>, bool),
    spawn_override: Option<&RestartSpawnOverride<'_>>,
) -> Result<ManagedServiceRecord> {
    #[cfg(windows)]
    let _ = spawn_override;
    let record = load_service_for_restart(paths, service_id, runtime_pin)?;
    // Preserve the endpoint key across the restart: `stop_internal_managed_service`
    // deletes the key file (correct on a real stop), but a restart must bring the
    // service back on the same public host with the same auth. Capture it first and
    // re-store it after the stop so the spawn below hands the child the same key.
    let preserved_endpoint_key = endpoint_keys::endpoint_api_key(paths, service_id);
    // Checked before the stop, so a refused restart leaves a running service
    // running instead of stopping it and then failing to bring it back.
    ensure_public_service_has_endpoint_key(
        &record.host,
        preserved_endpoint_key.is_some(),
        record.requires_api_key,
    )?;
    let (stop_report, stopped) =
        stop_internal_managed_service_confirmed_with(paths, service_id, terminate)
            .with_context(|| format!("failed to stop `{service_id}` before restarting it"))?;
    // Starting a new engine is only safe once the old one is confirmed gone: a
    // survivor still holds the port and the GPU, so the new child would fail to
    // bind or contend for the device, and recording it would hide the survivor
    // from the next stop. The unconfirmed stop has already re-stamped its
    // marker and kept the endpoint key, so the liveness refresh finishes the
    // stop once the process does exit.
    if !stopped {
        bail!(
            "did not restart `{service_id}`: its stop could not confirm that every recorded \
             process exited (signalled pids: {}), so no new engine was started; the stop \
             request stays recorded on the service",
            stop_report["signaled_pids"]
        );
    }
    if let Some(key) = preserved_endpoint_key.as_deref() {
        endpoint_keys::store_endpoint_api_key(paths, service_id, key)?;
    }
    // What the confirmed stop left on disk. Every later write is applied to the
    // record as it is then, guarded on it still being this — never a write-back
    // of the copy loaded before the stop, which would erase a stop marker or
    // anything else written while this restart ran. The confirmed stop cleared
    // its own marker, so a restart supersedes every stop asked for before it;
    // one asked for since is someone else's, and none of the writes below
    // touches it.
    let (stopped_record, _) = read_managed_service_manifest(paths, service_id)?;
    // Still after the stop, like the device-policy refusal inside it always was:
    // the callers' "stopped by the attempt" report and pin restore both cover a
    // restart that gets this far and fails. Pure computation — it parses the
    // record, it does not touch the network — so it stays ahead of the launch
    // lock below and a refused device policy still fails before anything binds.
    let serve_args = restart_child_argv(&record)?;
    // The confirmed stop just released this (host, port): nothing holds it again
    // until the new child binds and `record_restarted_child` publishes below.
    // Close that gap the same way a fresh launch and `rocmd` recovery do —
    // under the shared launch lock, with a re-check against the disk and a
    // loopback preflight — so a concurrent launch or recovery can't claim this
    // exact port out from under the restart. Unbounded, like every other
    // CLI-side acquire of this lock (see `FileLock::acquire_bounded`'s doc
    // comment for why only `rocmd`'s recovery bounds its wait).
    let launch_lock = rocm_core::FileLock::acquire(paths.managed_launch_lock_path())?;
    if let Some(collision) = load_managed_services(paths)
        .context("failed to re-read managed services under the launch lock")?
        .into_iter()
        .find(|other| {
            other.service_id != stopped_record.service_id && other.blocks_port(stopped_record.port)
        })
    {
        bail!(
            "cannot restart `{service_id}` on {}:{}: managed service {} is recorded on that \
             port (it may have claimed that port while the restart waited for the launch lock)",
            stopped_record.host,
            stopped_record.port,
            collision.service_id
        );
    }
    let port_lease = if rocm_core::is_canonical_loopback_host(&stopped_record.host) {
        Some(rocm_core::lease_loopback_port_or_bind_error(
            stopped_record.port,
            format_args!(
                "cannot restart `{service_id}` on {}:{}: its port is in use",
                stopped_record.host, stopped_record.port
            ),
        )?)
    } else {
        None
    };
    fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&record.log_path)
        .with_context(|| format!("failed to open {}", record.log_path.display()))?;
    if let Some(parent) = record.engine_state_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let current_exe = managed_service_launcher_path()
        .context("failed to resolve current rocm executable path")?;
    let engine_envs_root = env_root_for_service(
        paths,
        &record.engine,
        record.runtime_id.as_deref(),
        record.env_id.as_deref(),
    )?;
    // Hand the port to the respawned child: close the probe socket immediately
    // before the spawn, still holding the launch lock.
    if let Some(lease) = port_lease {
        lease.release();
    }
    // A restarted public service comes back authenticated: its 0600 key file
    // persists across restarts (removed only on stop), so hand the child the same
    // file and reuse the key for the readiness probe.
    let endpoint_key_file = endpoint_keys::endpoint_key_file_if_present(paths, &record.service_id);
    let endpoint_api_key = endpoint_keys::endpoint_api_key(paths, &record.service_id);
    #[cfg(windows)]
    let child_pid = {
        let env_values = app_path_env_var_values(paths, engine_envs_root.as_deref());
        let mut env_refs = app_path_env_var_refs(&env_values);
        if let Some(key_file) = endpoint_key_file.as_deref() {
            env_refs.push((rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV, key_file));
        }
        rocm_core::spawn_detached_no_inherit(&current_exe, &serve_args, &env_refs)
            .context("failed to restart managed engine process")?
    };
    #[cfg(not(windows))]
    let child_pid = {
        let mut command = managed_service_process_command(&current_exe, &serve_args);
        command.stdin(Stdio::null());
        attach_background_stdio(&mut command, Some(&record.log_path))?;
        detach_background_command(&mut command);
        apply_app_path_env(&mut command, paths);
        if let Some(engine_envs_root) = engine_envs_root.as_deref() {
            command.env("ROCM_CLI_ENGINE_ENVS_ROOT", engine_envs_root);
        }
        if let Some(key_file) = endpoint_key_file.as_deref() {
            command.env(rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV, key_file);
        }
        let mut child = match spawn_override {
            Some(spawn) => spawn(&mut command),
            None => command.spawn(),
        }
        .context("failed to restart managed engine process")?;
        thread::sleep(Duration::from_millis(200));
        if let Some(status) = child
            .try_wait()
            .context("failed to check restarted engine startup state")?
        {
            update_restarting_record(paths, &stopped_record, |current| {
                current.status = "failed".to_owned();
            })?;
            bail!(
                "{}",
                managed_engine_startup_failure_detail(
                    &format!(" with status {status}"),
                    &record.log_path
                )
            );
        }
        child.id()
    };
    #[cfg(windows)]
    thread::sleep(Duration::from_millis(200));
    // Recorded before the readiness wait, so a stop issued during it finds the
    // new child rather than the processes already stopped.
    let started = record_restarted_child(paths, &stopped_record, runtime_pin, child_pid)?;
    // Published; a concurrent launch or recovery now sees this service as live
    // on its port again, so there is nothing left for the lock to protect, and
    // the readiness wait below can block for many seconds.
    drop(launch_lock);
    let readiness = wait_for_service_http_ready(
        &started.engine,
        &started.host,
        started.port,
        started.served_model_name(),
        endpoint_api_key.as_deref(),
        Duration::from_secs(45),
    );
    record_restart_readiness(paths, &started, readiness)
}

/// Apply one of a restart's own changes to the record as it is on disk now,
/// rather than writing back a copy the restart holds. `write` replaces the
/// whole file, and a stop records `stop_requested_unix_ms` before it signals
/// anything — the same hazard `rocmd`'s `update_supervised_record` guards
/// against for the supervisor's writes.
///
/// Nothing is written, and `None` returned, when the record no longer reads
/// as `expected` left it — it is gone, names other processes, or has another
/// status: whoever wrote since holds a truer view than this restart. `update`
/// never touches the stop marker, so a stop asked for meanwhile survives.
fn update_restarting_record(
    paths: &AppPaths,
    expected: &ManagedServiceRecord,
    update: impl FnOnce(&mut ManagedServiceRecord),
) -> Result<Option<ManagedServiceRecord>> {
    if !paths.service_manifest_path(&expected.service_id).exists() {
        return Ok(None);
    }
    let (mut current, _) = read_managed_service_manifest(paths, &expected.service_id)?;
    if !current.names_same_processes(expected) || current.status != expected.status {
        return Ok(None);
    }
    update(&mut current);
    current.write()?;
    Ok(Some(current))
}

/// Record the child a restart just started, over the record its confirmed
/// stop left (`stopped`). Fails when that record has changed since: the child
/// is then running but unrecorded, and the caller has to hear that rather than
/// a claim that the service restarted.
///
/// `runtime_pin` is re-applied here because the record on disk does not carry
/// it: the restart's own load refreshed the record from the outgoing process's
/// engine state and persisted that, re-adopting the previous `runtime_id` and
/// `env_id` over the pin `runtime_services::pin_service_record_to_runtime`
/// wrote. The child was launched with the pin (see [`restart_child_argv`]), so
/// the record that names it has to say the same runtime.
fn record_restarted_child(
    paths: &AppPaths,
    stopped: &ManagedServiceRecord,
    runtime_pin: Option<&str>,
    child_pid: u32,
) -> Result<ManagedServiceRecord> {
    update_restarting_record(paths, stopped, |current| {
        apply_restart_runtime_pin(current, runtime_pin);
        current.status = "running".to_owned();
        // Both roles move to the restarted child, each with its own token —
        // see `spawn_managed_engine_child`.
        current.record_supervisor_identity(child_pid);
        current.record_engine_identity(child_pid);
        // Counts the restart and drops the previous run's inference
        // verification — the new child has an unloaded model, so the old
        // verdict says nothing about it.
        current.reset_for_restart();
    })?
    .with_context(|| {
        format!(
            "service `{}` changed on disk while it was restarting, so the restarted engine \
             (pid {child_pid}) was not recorded on it",
            stopped.service_id
        )
    })
}

/// The restart's last write: what the readiness wait found. Applied only while
/// the record still reads as [`record_restarted_child`] left it; otherwise a
/// stop or another restart has written since, and the record as it is now is
/// returned untouched.
fn record_restart_readiness(
    paths: &AppPaths,
    started: &ManagedServiceRecord,
    readiness: EndpointReadiness,
) -> Result<ManagedServiceRecord> {
    let updated = update_restarting_record(paths, started, |current| {
        current.status = status_for_readiness(readiness).to_owned();
        if readiness == EndpointReadiness::Serving {
            current.inference_verified_at_unix_ms = Some(rocm_core::unix_time_millis() as u64);
        }
    })?;
    match updated {
        Some(record) => Ok(record),
        None => Ok(read_managed_service_manifest(paths, &started.service_id)
            .map_or_else(|_| started.clone(), |(record, _)| record)),
    }
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
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::providers;
    use crate::runtime_services;
    use crate::tests::{
        NEW_RUNTIME_KEY, OLD_RUNTIME_KEY, ScopedTestEnv, plant_service_record_on_runtime,
        test_paths,
    };

    #[test]
    fn lemonade_stop_unloads_selected_model_over_http() -> Result<()> {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::sync::mpsc;

        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        let (sender, receiver) = mpsc::channel();
        let handle = thread::spawn(move || -> Result<()> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(2)))?;
            let mut request = Vec::new();
            let mut buffer = [0_u8; 512];
            loop {
                let read = stream.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                let text = String::from_utf8_lossy(&request);
                if let Some((headers, body)) = text.split_once("\r\n\r\n") {
                    let expected = headers
                        .lines()
                        .find_map(|line| line.strip_prefix("Content-Length: "))
                        .and_then(|value| value.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                    if body.len() >= expected {
                        break;
                    }
                }
            }
            let text = String::from_utf8(request).context("request was not utf-8")?;
            sender.send(text).ok();
            stream.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: close\r\n\r\n{\"status\":\"success\",\"message\":\"ok\"}",
            )?;
            Ok(())
        });

        let (_root, paths) = test_paths("lemonade-stop-unload");
        let record = ManagedServiceRecord::new(
            &paths,
            "svc-qwen",
            "lemonade",
            "qwen",
            "Qwen3-0.6B-GGUF",
            "127.0.0.1",
            port,
            "managed",
            123,
            Some("therock-release".to_owned()),
            Some("lemonade-embeddable-10.6.0".to_owned()),
            Some("gpu_required".to_owned()),
        );
        unload_lemonade_service_model(&record)?;
        handle.join().expect("listener thread panicked")?;
        let request = receiver.recv_timeout(Duration::from_secs(1))?;
        assert!(request.starts_with("POST /v1/unload HTTP/1.1"));
        assert!(request.contains("\"model_name\":\"Qwen3-0.6B-GGUF\""));
        Ok(())
    }

    #[test]
    fn lemonade_stop_unload_is_bounded_by_the_request_timeout() -> Result<()> {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        use std::time::Instant;

        // Regression test for the stall this PR fixes: a peer that trickles the
        // response one byte at a time, never framing or closing, used to stall
        // `read_tcp_stream_to_string`'s read-to-EOF loop indefinitely. That hung
        // `unload_lemonade_service_model` past its 5s timeout during scenario
        // teardown, showing up as an unexplained multi-minute gap. The unload
        // call must now return an error at (not far past) its 5s budget.
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let port = listener.local_addr()?.port();
        thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
            let mut buffer = [0_u8; 512];
            let _ = stream.read(&mut buffer);
            let body = b"{\"status\":\"success\",\"message\":\"ok\"}";
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            // One byte every 300ms never finishes framing the 35-byte body
            // inside the 5s unload timeout below, so the bound under test is
            // the deadline firing, not the response completing early.
            for byte in body {
                if stream.write_all(&[*byte]).is_err() {
                    return;
                }
                thread::sleep(Duration::from_millis(300));
            }
        });

        let (_root, paths) = test_paths("lemonade-stop-unload-dribble");
        let record = ManagedServiceRecord::new(
            &paths,
            "svc-qwen",
            "lemonade",
            "qwen",
            "Qwen3-0.6B-GGUF",
            "127.0.0.1",
            port,
            "managed",
            123,
            Some("therock-release".to_owned()),
            Some("lemonade-embeddable-10.6.0".to_owned()),
            Some("gpu_required".to_owned()),
        );
        let started = Instant::now();
        assert!(unload_lemonade_service_model(&record).is_err());
        let elapsed = started.elapsed();
        assert!(
            elapsed >= Duration::from_secs(4),
            "bounded BY the 5s deadline, not failing early: {elapsed:?}"
        );
        assert!(elapsed < Duration::from_secs(8), "{elapsed:?}");
        Ok(())
    }

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

    #[test]
    fn restarting_a_keyless_public_service_names_the_public_bind_not_the_key_flag() -> Result<()> {
        // The same defect at a real call site rather than through the guard's own
        // arguments. `restart` reads `requires_api_key` off the record on disk, so
        // a record written by an ordinary public-bind launch must not claim the
        // service asked for `--require-api-key` — the branch order means a record
        // that claims it wins, and its remediation drops `--allow-public-bind`.
        let (root, paths) = test_paths("restart-public-no-key");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "vllm-public-2000",
            "vllm",
            "qwen",
            "qwen-canonical",
            "0.0.0.0",
            12000,
            "managed",
            std::process::id(),
            None,
            None,
            None,
        );
        // What a plain `--host 0.0.0.0 --allow-public-bind` launch records: the
        // bind is public, and the flag was never passed.
        record.requires_api_key = false;
        record.write()?;

        // No key file: the situation after a stop, which drops it.
        let error = restart_internal_managed_service(&paths, "vllm-public-2000", None)
            .expect_err("a keyless public service must not be restarted");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("--allow-public-bind"),
            "the refusal must name the command that brings it back public: {rendered}"
        );
        assert!(
            !rendered.contains("--require-api-key"),
            "a service that never passed the flag must not be told it did: {rendered}"
        );

        // The refusal happens before the stop, so the service is left alone.
        assert!(
            load_managed_service(&paths, "vllm-public-2000").is_ok(),
            "a refused restart must not have removed the record"
        );
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn restart_refuses_a_public_service_without_a_key_before_stopping_it() {
        // The guard runs before the stop, so a refused restart must leave the
        // record exactly as it was rather than taking down a running service.
        let (root, paths) = test_paths("restart-public-no-key");
        let service_id = "svc-public-nokey";
        let mut record = ManagedServiceRecord::new(
            &paths,
            service_id,
            "vllm",
            "model-ref",
            "canonical/model",
            "0.0.0.0",
            11435,
            "managed",
            std::process::id(),
            None,
            None,
            Some("gpu_required".to_owned()),
        );
        record.status = "running".to_owned();
        record.write().unwrap();

        let error = restart_internal_managed_service(&paths, service_id, None).unwrap_err();
        assert!(
            error.to_string().contains("without authentication"),
            "{error:#}"
        );

        // Proves the *ordering*, not just the refusal: had the guard run after
        // `stop_internal_managed_service`, the stop would have written
        // "stopped". It reaches that state here because the record's only pid is
        // the test's own (`engine_pid` is None and `terminate_recorded_service_pids`
        // skips the caller's pid), so the stop confirms termination trivially.
        let after = load_managed_service(&paths, service_id).unwrap();
        assert_ne!(
            after.status, "stopped",
            "a refused restart must not stop the service"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn stop_records_its_marker_on_disk_before_any_process_is_signalled() {
        // The daemon's `server-recover` watcher restarts a managed service whose
        // endpoint stops answering unless the record carries a stop marker. A
        // stop that wrote the marker only after terminating the processes would
        // leave that whole window — the engine `Stop`, then up to the grace
        // period per PID — open for the daemon to bring the service back. So at
        // the moment the processes are signalled the marker must already be on
        // disk, not just in the stop's in-memory copy.
        let (root, paths) = test_paths("stop-marker-before-signal");
        paths.ensure().unwrap();
        let service_id = "svc-stop-marker-order";
        // A port nothing listens on, so the lemonade unload fails fast instead
        // of reaching a real server.
        let port = std::net::TcpListener::bind(("127.0.0.1", 0))
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let mut record = ManagedServiceRecord::new(
            &paths,
            service_id,
            "lemonade",
            "model-ref",
            "canonical/model",
            "127.0.0.1",
            port,
            "managed",
            999_999_999,
            None,
            None,
            None,
        );
        record.status = "ready".to_owned();
        record.write().unwrap();

        // Read the file as-is: `load_managed_service` runs the liveness refresh,
        // which would consume the marker of a record whose PIDs are dead.
        let on_disk = || {
            serde_json::from_slice::<ManagedServiceRecord>(
                &fs::read(paths.service_manifest_path(service_id)).unwrap(),
            )
            .unwrap()
        };
        let seen_at_signal = std::cell::Cell::new(None);
        let report = stop_internal_managed_service_with(&paths, service_id, |_| {
            let on_disk = on_disk();
            seen_at_signal.set(Some(on_disk.stop_requested_unix_ms));
            (Vec::new(), false)
        })
        .unwrap();

        assert!(
            matches!(seen_at_signal.get(), Some(Some(_))),
            "the stop marker must be persisted before the processes are signalled, \
             saw {:?}",
            seen_at_signal.get()
        );
        // The final-state gating is unchanged: an unconfirmed stop keeps the
        // marker it wrote up front.
        assert!(on_disk().stop_requested_unix_ms.is_some());
        assert_eq!(report["signaled_pids"], serde_json::json!([]));
        let _ = fs::remove_dir_all(root);
    }

    /// A stopped-lemonade record on a port nothing listens on, so the stop's
    /// unload fails fast instead of reaching a real server.
    fn stop_test_record(paths: &AppPaths, service_id: &str) -> ManagedServiceRecord {
        let port = std::net::TcpListener::bind(("127.0.0.1", 0))
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let mut record = ManagedServiceRecord::new(
            paths,
            service_id,
            "lemonade",
            "model-ref",
            "canonical/model",
            "127.0.0.1",
            port,
            "managed",
            999_999_999,
            None,
            None,
            None,
        );
        record.status = "ready".to_owned();
        record
    }

    /// The manifest as it is on disk, without `load_managed_service`'s refresh.
    fn raw_manifest(paths: &AppPaths, service_id: &str) -> ManagedServiceRecord {
        serde_json::from_slice(&fs::read(paths.service_manifest_path(service_id)).unwrap()).unwrap()
    }

    #[test]
    fn stop_does_not_confirm_when_a_restart_recorded_new_processes_meanwhile() {
        // A stop's verdict covers the processes it read and signalled. A restart
        // landing while it ran records a new supervisor and clears the marker;
        // claiming that service stopped, or deleting the key it serves with,
        // would be false. Here every process the stop handled is confirmed gone,
        // so only the re-read can tell.
        let (root, paths) = test_paths("stop-restart-meanwhile");
        paths.ensure().unwrap();
        let service_id = "svc-stop-restart-meanwhile";
        stop_test_record(&paths, service_id).write().unwrap();
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "key-123").unwrap();

        let report = stop_internal_managed_service_with(&paths, service_id, |_| {
            let mut restarted = raw_manifest(&paths, service_id);
            restarted.status = "recovering".to_owned();
            restarted.supervisor_pid = 4_242_424;
            restarted.supervisor_start_ticks = Some(7);
            restarted.stop_requested_unix_ms = None;
            restarted.write().unwrap();
            (Vec::new(), true)
        })
        .unwrap();

        let on_disk = raw_manifest(&paths, service_id);
        let key = endpoint_keys::endpoint_api_key(&paths, service_id);
        let _ = fs::remove_dir_all(root);
        assert_eq!(report["status"], "recovering", "{report}");
        assert_eq!(on_disk.status, "recovering");
        assert_eq!(on_disk.supervisor_pid, 4_242_424, "the restart's PID stays");
        assert!(
            on_disk.stop_requested_unix_ms.is_some(),
            "an unconfirmed stop re-stamps the marker the restart cleared"
        );
        assert_eq!(key.as_deref(), Some("key-123"), "the key is kept");
    }

    #[test]
    fn stop_of_a_container_engine_is_unconfirmed_when_the_engine_stop_fails() {
        // For `aim` the recorded PID is only the supervisor; killing it says
        // nothing about the container. With no aim engine to answer `Stop`, the
        // PID verdict alone must not mark the service stopped or drop its key.
        let (root, paths) = test_paths("stop-aim-engine-stop-fails");
        paths.ensure().unwrap();
        // The built-in aim engine answers `Stop` through `docker`; a dead daemon
        // address makes that fail the same way on a host with or without docker.
        let mut env = ScopedTestEnv::new();
        env.set("DOCKER_HOST", "unix:///nonexistent/rocm-test-docker.sock");
        let service_id = "svc-stop-aim-engine-stop-fails";
        let mut record = stop_test_record(&paths, service_id);
        record.engine = "aim".to_owned();
        record.write().unwrap();
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "key-123").unwrap();

        let report =
            stop_internal_managed_service_with(&paths, service_id, |_| (Vec::new(), true)).unwrap();
        drop(env);

        let on_disk = raw_manifest(&paths, service_id);
        let key = endpoint_keys::endpoint_api_key(&paths, service_id);
        let _ = fs::remove_dir_all(root);
        assert!(report["engine_stop"]["error"].is_string(), "{report}");
        // Unconfirmed: the marker is re-stamped and the key kept for the liveness
        // refresh to settle, rather than cleared as for a confirmed stop.
        assert!(on_disk.stop_requested_unix_ms.is_some());
        assert_eq!(key.as_deref(), Some("key-123"), "the key is kept");
    }

    #[test]
    fn stop_final_write_keeps_what_others_wrote_while_it_ran() {
        // The stop's last write applies its own fields to the record as it is
        // now, not to the copy it loaded before the engine stop and the wait.
        let (root, paths) = test_paths("stop-final-write-rereads");
        paths.ensure().unwrap();
        let service_id = "svc-stop-final-write-rereads";
        stop_test_record(&paths, service_id).write().unwrap();
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "key-123").unwrap();

        let report = stop_internal_managed_service_with(&paths, service_id, |_| {
            let mut meanwhile = raw_manifest(&paths, service_id);
            meanwhile.restart_count = 7;
            meanwhile.write().unwrap();
            (Vec::new(), true)
        })
        .unwrap();

        let on_disk = raw_manifest(&paths, service_id);
        let key = endpoint_keys::endpoint_api_key(&paths, service_id);
        let _ = fs::remove_dir_all(root);
        assert_eq!(report["status"], "stopped", "{report}");
        assert_eq!(on_disk.status, "stopped");
        assert_eq!(on_disk.stop_requested_unix_ms, None);
        assert_eq!(on_disk.restart_count, 7, "a concurrent write survives");
        assert_eq!(key, None, "a confirmed stop drops the key");
    }

    #[test]
    fn restart_does_not_start_an_engine_over_an_unconfirmed_stop() {
        // A restart used to discard its stop's result and start the new engine
        // regardless, next to a survivor still holding the port and the GPU.
        // Here the stop cannot confirm termination: the restart must say so,
        // start nothing, and leave the stop's own state standing.
        let (root, paths) = test_paths("restart-unconfirmed-stop");
        paths.ensure().unwrap();
        let service_id = "svc-restart-unconfirmed-stop";
        let record = stop_test_record(&paths, service_id);
        record.write().unwrap();
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "key-123").unwrap();

        let result = restart_internal_managed_service_with(&paths, service_id, None, |_| {
            (vec![999_999_999], false)
        });

        let on_disk = raw_manifest(&paths, service_id);
        let key = endpoint_keys::endpoint_api_key(&paths, service_id);
        let log_created = record.log_path.exists();
        let _ = fs::remove_dir_all(root);
        let error = result.expect_err("an unconfirmed stop must not be restarted over");
        let rendered = format!("{error:#}");
        assert!(rendered.contains("could not confirm"), "{rendered}");
        assert!(rendered.contains("no new engine was started"), "{rendered}");
        assert!(rendered.contains("999999999"), "names the pids: {rendered}");
        // The claims in that message, checked against the state they describe.
        assert!(
            !log_created,
            "nothing was started: the restart never opened the new run's log"
        );
        assert_eq!(on_disk.restart_count, 0, "no restart was recorded");
        assert_eq!(
            on_disk.supervisor_pid, 999_999_999,
            "the record still names the survivor"
        );
        assert!(
            on_disk.stop_requested_unix_ms.is_some(),
            "the stop request stays recorded on the service"
        );
        assert_eq!(
            key.as_deref(),
            Some("key-123"),
            "the key stays with the survivor"
        );
    }

    /// A record as a restart's confirmed stop leaves it, on disk.
    fn stopped_restart_record(paths: &AppPaths, service_id: &str) -> ManagedServiceRecord {
        let mut record = stop_test_record(paths, service_id);
        record.status = "stopped".to_owned();
        record.write().unwrap();
        raw_manifest(paths, service_id)
    }

    #[test]
    fn restart_readiness_write_keeps_a_stop_asked_for_while_it_waited() {
        // The restart's last write lands after a readiness wait of up to 45
        // seconds. A stop asked for in that window persists its marker before
        // it signals anything; writing the restart's own copy back erased it.
        let (root, paths) = test_paths("restart-readiness-keeps-marker");
        paths.ensure().unwrap();
        let service_id = "svc-restart-readiness-keeps-marker";
        let stopped = stopped_restart_record(&paths, service_id);
        let started = record_restarted_child(&paths, &stopped, None, std::process::id()).unwrap();

        let mut stopping = raw_manifest(&paths, service_id);
        stopping.stop_requested_unix_ms = Some(1_234);
        stopping.write().unwrap();

        let returned =
            record_restart_readiness(&paths, &started, EndpointReadiness::Unreachable).unwrap();
        let on_disk = raw_manifest(&paths, service_id);
        let _ = fs::remove_dir_all(root);
        assert_eq!(
            on_disk.stop_requested_unix_ms,
            Some(1_234),
            "the stop survives"
        );
        assert_eq!(on_disk.supervisor_pid, std::process::id());
        assert_eq!(on_disk.restart_count, 1);
        assert_eq!(returned.stop_requested_unix_ms, Some(1_234));
    }

    #[test]
    fn restart_readiness_write_leaves_a_stop_that_finished_meanwhile() {
        // A stop that confirmed the new child gone during the wait wrote
        // "stopped"; the restart's verdict on a process that no longer exists
        // must not overwrite it.
        let (root, paths) = test_paths("restart-readiness-after-stop");
        paths.ensure().unwrap();
        let service_id = "svc-restart-readiness-after-stop";
        let stopped = stopped_restart_record(&paths, service_id);
        let started = record_restarted_child(&paths, &stopped, None, std::process::id()).unwrap();

        let mut finished = raw_manifest(&paths, service_id);
        finished.status = "stopped".to_owned();
        finished.write().unwrap();

        let returned =
            record_restart_readiness(&paths, &started, EndpointReadiness::Serving).unwrap();
        let on_disk = raw_manifest(&paths, service_id);
        let _ = fs::remove_dir_all(root);
        assert_eq!(on_disk.status, "stopped");
        assert_eq!(on_disk.inference_verified_at_unix_ms, None);
        assert_eq!(
            returned.status, "stopped",
            "the caller sees the record as it is"
        );
    }

    #[test]
    fn restarted_child_record_names_the_runtime_the_restart_was_pinned_to() {
        // A move (`rocm runtimes activate --restart-services`) launches the
        // child on the pinned runtime, but the restart's own load re-adopted the
        // outgoing process's `runtime_id`/`env_id` from its engine state and
        // persisted them over the pin. The record the restart writes for its
        // child is built from disk, so it has to carry the pin again or it
        // names the runtime the service just left.
        let (root, paths) = test_paths("restart-child-runtime-pin");
        paths.ensure().unwrap();
        let service_id = "svc-restart-child-runtime-pin";
        let mut stopped = stop_test_record(&paths, service_id);
        stopped.status = "stopped".to_owned();
        stopped.runtime_id = Some("runtime-old".to_owned());
        stopped.env_id = Some("env-old".to_owned());
        stopped.write().unwrap();
        let stopped = raw_manifest(&paths, service_id);

        let started =
            record_restarted_child(&paths, &stopped, Some("runtime-new"), std::process::id())
                .unwrap();
        let on_disk = raw_manifest(&paths, service_id);
        let _ = fs::remove_dir_all(root);
        assert_eq!(on_disk.runtime_id.as_deref(), Some("runtime-new"));
        assert_eq!(
            on_disk.env_id, None,
            "the pin drops the env, as a move does"
        );
        assert_eq!(started.runtime_id.as_deref(), Some("runtime-new"));
        assert_eq!(on_disk.supervisor_pid, std::process::id());
    }

    #[test]
    fn restart_reports_a_child_it_could_not_record_and_writes_nothing() {
        // Between the restart's stop and its spawn, another restart recorded
        // its own processes. Writing this child over them would hide theirs;
        // the restart has to report its child unrecorded instead.
        let (root, paths) = test_paths("restart-child-unrecorded");
        paths.ensure().unwrap();
        let service_id = "svc-restart-child-unrecorded";
        let stopped = stopped_restart_record(&paths, service_id);

        let mut other = raw_manifest(&paths, service_id);
        other.status = "running".to_owned();
        other.supervisor_pid = 4_242_424;
        other.supervisor_start_ticks = Some(7);
        other.write().unwrap();

        let result = record_restarted_child(&paths, &stopped, None, std::process::id());
        let on_disk = raw_manifest(&paths, service_id);
        let _ = fs::remove_dir_all(root);
        let rendered = format!(
            "{:#}",
            result.expect_err("a changed record must be reported")
        );
        assert!(rendered.contains("was not recorded"), "{rendered}");
        assert!(
            rendered.contains(&std::process::id().to_string()),
            "{rendered}"
        );
        assert_eq!(
            on_disk.supervisor_pid, 4_242_424,
            "the other restart's PID stays"
        );
        assert_eq!(on_disk.restart_count, 0, "nothing was written");
    }

    /// The gap a manual restart used to leave open: the confirmed stop just
    /// released the service's port, and nothing reserves it again until the
    /// new child binds and `record_restarted_child` publishes — a concurrent
    /// `rocm serve` launch could claim that exact port in between. Recreate
    /// the race under the shared launch lock and require the restart to
    /// refuse rather than spawn a competitor onto it, following the same
    /// wait-then-publish-then-release shape as
    /// `services_prune_waits_for_the_managed_launch_lock_before_sweeping`.
    #[test]
    fn restart_refuses_a_port_a_launch_claimed_while_it_waited_for_the_lock() -> Result<()> {
        use std::sync::mpsc;

        const WAIT_PROOF: Duration = Duration::from_millis(500);

        let (root, paths) = test_paths("restart-launch-claimed-port");
        paths.ensure()?;
        let service_id = "svc-restart-launch-claimed-port";
        let record = stop_test_record(&paths, service_id);
        let host = record.host.clone();
        let port = record.port;
        record.write()?;

        let launch_lock = rocm_core::FileLock::acquire(paths.managed_launch_lock_path())?;

        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let (waited, finished) = std::thread::scope(|scope| {
            scope.spawn(|| {
                started_tx.send(()).expect("signal about to restart");
                let outcome =
                    restart_internal_managed_service_with(&paths, service_id, None, |_| {
                        (Vec::new(), true)
                    });
                let _ = done_tx.send(outcome.map_err(|error| format!("{error:#}")));
            });
            started_rx
                .recv_timeout(Duration::from_secs(10))
                .expect("restart thread started");
            let early = done_rx.recv_timeout(WAIT_PROOF);
            let waited = early.is_err();
            // Simulate a fresh `rocm serve --gpu auto` publishing a brand-new
            // service on this exact port while the restart was still waiting
            // for the lock it itself now holds.
            let mut competitor = ManagedServiceRecord::new(
                &paths,
                "svc-fresh-launch",
                "lemonade",
                "other-model",
                "other-model-canonical",
                &host,
                port,
                "managed",
                0,
                None,
                None,
                None,
            );
            competitor.status = "starting".to_owned();
            competitor.write().expect("publish the competing record");
            drop(launch_lock);
            let finished = match early {
                Ok(finished) => finished,
                Err(_) => done_rx
                    .recv_timeout(Duration::from_secs(10))
                    .expect("restart completes once the launch lock is released"),
            };
            (waited, finished)
        });
        let _ = fs::remove_dir_all(root);

        assert!(
            waited,
            "the restart must serialize on the shared launch lock"
        );
        let error =
            finished.expect_err("a live competitor on the same port must refuse the restart");
        assert!(error.contains("svc-fresh-launch"), "{error}");
        assert!(error.contains("claimed that port"), "{error}");
        Ok(())
    }

    /// The disk re-check above only catches a recorded competitor. Something
    /// bound to the exact port without ever publishing a manifest for it — a
    /// stray process, a one-off `nc -l` — would sail through that check; only
    /// the loopback lease actually binding the port catches it. A real
    /// listener held on the port is what proves that lease ran, not just a
    /// record on disk.
    #[test]
    fn restart_refuses_a_port_something_is_actually_bound_to() -> Result<()> {
        let (root, paths) = test_paths("restart-port-actually-bound");
        paths.ensure()?;
        let service_id = "svc-restart-port-actually-bound";
        let record = stop_test_record(&paths, service_id);
        let port = record.port;
        record.write()?;

        let _blocker = std::net::TcpListener::bind(("127.0.0.1", port))?;

        let result =
            restart_internal_managed_service_with(&paths, service_id, None, |_| (Vec::new(), true));

        let _ = fs::remove_dir_all(root);
        let error =
            result.expect_err("a port actually bound by someone else must refuse the restart");
        let rendered = format!("{error:#}");
        assert!(rendered.contains("its port is in use"), "{rendered}");
        Ok(())
    }

    /// Mirrors `launch_releases_the_leased_port_before_spawning_the_child` and
    /// its rocmd twin: the restart's probe lease must be closed by the time the
    /// child is spawned, or the child's own bind of the port fails. Without
    /// the `lease.release()` call every loopback restart would die at the
    /// engine bind, and nothing else here would notice.
    #[cfg(not(windows))]
    #[test]
    fn restart_releases_the_leased_port_before_spawning_the_child() -> Result<()> {
        let (root, paths) = test_paths("restart-releases-lease");
        paths.ensure()?;
        let service_id = "svc-restart-releases-lease";
        let record = stop_test_record(&paths, service_id);
        let port = record.port;
        record.write()?;

        let bind_succeeded = std::cell::Cell::new(false);
        let lock_still_held = std::cell::Cell::new(false);
        let lock_path = paths.managed_launch_lock_path();
        let spawn: &RestartSpawnOverride<'_> = &|_command| {
            // The launch lock must outlive the spawn: it is what keeps a
            // concurrent launch from claiming the port before this record is
            // republished.
            lock_still_held
                .set(rocm_core::FileLock::acquire_bounded(&lock_path, Duration::ZERO).is_err());
            // Retry briefly: a sibling test's concurrent fork can hold a copy of
            // the just-released socket until its exec closes it.
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while !bind_succeeded.get() && std::time::Instant::now() < deadline {
                bind_succeeded.set(std::net::TcpListener::bind(("127.0.0.1", port)).is_ok());
            }
            // Fail the spawn on purpose: the observation above is all this
            // test needs, and no real engine should start.
            Err(io::Error::other("spawn intercepted by test"))
        };
        let result = restart_internal_managed_service_inner(
            &paths,
            service_id,
            None,
            |_| (Vec::new(), true),
            Some(spawn),
        );
        let _ = fs::remove_dir_all(root);

        assert!(
            bind_succeeded.get(),
            "the leased port must already be bindable (released) by the time the child is spawned"
        );
        assert!(
            lock_still_held.get(),
            "the launch lock must still be held when the restart spawns"
        );
        let rendered = format!("{:#}", result.expect_err("the intercepted spawn must fail"));
        assert!(
            rendered.contains("spawn intercepted by test"),
            "the restart must have reached the spawn: {rendered}"
        );
        Ok(())
    }

    #[cfg(not(windows))]
    #[test]
    fn restart_is_not_blocked_by_a_stopped_or_failed_record_on_the_same_port() -> Result<()> {
        // Mirror of `restart_refuses_a_port_a_launch_claimed...`: `blocks_port` is
        // `is_live() && port == port`, so a `stopped`/`failed` sibling on the same
        // port must not trip the "claimed that port" refusal. Mutation-proof: if
        // `blocks_port` regressed to ignore status, the restart would bail there
        // instead of reaching the real (and, in this test harness, doomed) child
        // spawn below.
        let (root, paths) = test_paths("restart-non-live-same-port");
        paths.ensure()?;
        let service_id = "svc-restart-non-live-same-port";
        let record = stop_test_record(&paths, service_id);
        let host = record.host.clone();
        let port = record.port;
        record.write()?;

        let mut stopped_sibling = ManagedServiceRecord::new(
            &paths,
            "svc-stopped-sibling",
            "lemonade",
            "other-model",
            "other-model-canonical",
            &host,
            port,
            "managed",
            0,
            None,
            None,
            None,
        );
        stopped_sibling.status = "stopped".to_owned();
        stopped_sibling.write()?;

        let mut failed_sibling = ManagedServiceRecord::new(
            &paths,
            "svc-failed-sibling",
            "lemonade",
            "other-model-2",
            "other-model-2-canonical",
            &host,
            port,
            "managed",
            0,
            None,
            None,
            None,
        );
        failed_sibling.status = "failed".to_owned();
        failed_sibling.write()?;

        // Fail the spawn on purpose: reaching it is what proves neither sibling
        // blocked the restart, so any other error is a real failure here.
        let spawn: &RestartSpawnOverride<'_> =
            &|_command| Err(io::Error::other("spawn intercepted by test"));
        let result = restart_internal_managed_service_inner(
            &paths,
            service_id,
            None,
            |_| (Vec::new(), true),
            Some(spawn),
        );

        let _ = fs::remove_dir_all(root);
        let rendered = format!("{:#}", result.expect_err("the intercepted spawn must fail"));
        assert!(
            rendered.contains("spawn intercepted by test"),
            "a stopped or failed sibling on the same port must not block the restart: {rendered}"
        );
        Ok(())
    }

    /// The value following `flag` in an argv, or `None` when the flag is absent.
    fn argv_flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
        let index = args.iter().position(|arg| arg == flag)?;
        args.get(index + 1).map(String::as_str)
    }

    /// The runtime pin a `--restart-services` restart moves a server with, at
    /// the only place it is observable without launching an engine: the argv the
    /// restarted child is given.
    ///
    /// The sibling test above asserts the record's fields; that is not the same
    /// claim. `builtin_engine_serve_http_args` omits `--runtime-id` whenever
    /// `env_id` is set, so what the record says and what the child is told can
    /// differ — and it is the child's argv that decides which runtime comes back
    /// up.
    ///
    /// The setup is what makes this non-vacuous, and it is the state a live
    /// restart really sees rather than an invented one. The service being moved is
    /// still running when the restart loads its record, so its engine state file
    /// still describes it, and `refresh_from_engine_state` re-adopts that
    /// process's `runtime_id` and `env_id` over the pin
    /// `pin_service_record_to_runtime` had just written. The `None` load at the
    /// end is that load: the argv carries no `--runtime-id` at all because the
    /// revived `env_id` suppresses it.
    ///
    /// The restart is the production `restart_service_onto_runtime_with`, handed
    /// a stand-in that does what `restart_internal_managed_service` does up to
    /// the spawn — `load_service_for_restart`, then `restart_child_argv` — and
    /// then fails. Two things are guarded:
    ///
    /// - that `load_service_for_restart` applies the pin it is given: removing
    ///   the `apply_restart_runtime_pin` call inside it leaves the revived
    ///   `env_id` in place, and the `--runtime-id` and `--env-id` assertions
    ///   fail;
    /// - that `restart_service_onto_runtime_with` hands the key to the restart:
    ///   passing `None` instead of `Some(runtime_key)` fails the assertion on the
    ///   pin the stand-in was given, and with it the `--runtime-id` assertion.
    ///
    /// The failing stand-in also drives the restore path, so the record on disk
    /// is read back to be on the runtime it ran on.
    ///
    /// What it does not reach is that a real child launched with this argv loads
    /// the runtime, which needs a GPU: `runtime-lifecycle-12` observes that.
    #[test]
    fn pinning_the_record_is_what_puts_the_new_runtime_in_the_restart_argv() -> Result<()> {
        let (root, paths) = test_paths("service-runtime-pin-argv");
        paths.ensure()?;
        let planted = plant_service_record_on_runtime(
            &paths,
            "svc-argv",
            "vllm",
            "starting",
            std::process::id(),
            Some(OLD_RUNTIME_KEY),
            Some("custom-env"),
        )?;
        // The outgoing process's own state, in the shape vLLM writes it: the
        // exact key it was launched with, the family form it resolved that to,
        // and an `env_id` it records on every launch whether or not the user
        // asked for one.
        fs::write(
            &planted.engine_state_path,
            serde_json::to_vec(&serde_json::json!({
                "status": "starting",
                "requested_runtime_id": OLD_RUNTIME_KEY,
                "runtime_id": "therock-release:gfx120X-all",
                "env_id": "custom-env",
            }))?,
        )?;
        let handed_pin = RefCell::new(None::<String>);
        let observed_argv = RefCell::new(None::<Vec<String>>);

        let error = runtime_services::restart_service_onto_runtime_with(
            &paths,
            "svc-argv",
            NEW_RUNTIME_KEY,
            |restart_paths, service_id, pin| -> Result<ManagedServiceRecord> {
                *handed_pin.borrow_mut() = pin.map(ToOwned::to_owned);
                let record = load_service_for_restart(restart_paths, service_id, pin)?;
                *observed_argv.borrow_mut() = Some(restart_child_argv(&record)?);
                bail!("stop here: the argv is what is under test");
            },
        )
        .expect_err("the stand-in restart always fails");
        let handed_pin = handed_pin.into_inner();
        let observed_argv = observed_argv.into_inner();
        // Read the file itself: a load would refresh the record from the engine
        // state again and hide what the restore wrote.
        let on_disk: ManagedServiceRecord =
            serde_json::from_slice(&fs::read(paths.service_manifest_path("svc-argv"))?)?;
        let unpinned = load_service_for_restart(&paths, "svc-argv", None)?;
        let unpinned_argv = restart_child_argv(&unpinned)?;
        let _ = fs::remove_dir_all(&root);

        assert!(
            format!("{error:#}").contains("stop here"),
            "the stand-in's own failure must reach the caller:\n{error:#}"
        );
        assert_eq!(
            handed_pin.as_deref(),
            Some(NEW_RUNTIME_KEY),
            "the key has to be passed to the restart as well as written to the \
             record: the restart's own load refreshes the record from the \
             outgoing process's state and undoes the write"
        );
        let argv = observed_argv.context("the stand-in restart never built an argv")?;
        assert_eq!(
            argv_flag_value(&argv, "--runtime-id"),
            Some(NEW_RUNTIME_KEY),
            "re-applying the pin after the refresh is what makes the restarted \
             child load the runtime that was just activated: {argv:?}"
        );
        assert_eq!(
            argv_flag_value(&argv, "--env-id"),
            None,
            "clearing `env_id` is half the pin, not tidying: leaving it set \
             would suppress the `--runtime-id` asserted above: {argv:?}"
        );
        assert_eq!(
            argv_flag_value(&unpinned_argv, "--runtime-id"),
            None,
            "without the pin the same record is restarted with no `--runtime-id`, \
             because the outgoing process's `env_id` is restored over it; this is \
             what makes the assertions above mean something: {unpinned_argv:?}"
        );
        assert_eq!(
            argv_flag_value(&unpinned_argv, "--env-id"),
            Some("custom-env"),
            "which is the engine's own env coming back, not the user's: {unpinned_argv:?}"
        );
        assert_eq!(
            on_disk.runtime_id.as_deref(),
            Some(OLD_RUNTIME_KEY),
            "the failed restart must put the record back on the runtime it ran on"
        );
        assert_eq!(on_disk.env_id.as_deref(), Some("custom-env"));
        Ok(())
    }

    #[test]
    fn a_refused_service_restart_leaves_the_record_on_the_runtime_it_ran_on() -> Result<()> {
        let (root, paths) = test_paths("service-restart-refused");
        paths.ensure()?;
        let mut record = plant_service_record_on_runtime(
            &paths,
            "svc-public",
            "vllm",
            "starting",
            std::process::id(),
            Some(OLD_RUNTIME_KEY),
            Some("custom-env"),
        )?;
        // A public bind with no endpoint key is refused by
        // `restart_internal_managed_service` before it stops anything, which
        // makes the failure branch reachable without spawning a process — and,
        // just as importantly, without signalling the pid the planted record
        // carries, which is this test process.
        record.host = "0.0.0.0".to_owned();
        record.write()?;
        endpoint_keys::clear_endpoint_api_key(&paths, "svc-public");

        let error =
            runtime_services::restart_service_onto_runtime(&paths, "svc-public", NEW_RUNTIME_KEY)
                .expect_err("a refused restart must fail the move");
        let after = load_managed_service(&paths, "svc-public")?;
        let _ = fs::remove_dir_all(&root);

        assert!(
            format!("{error:#}").contains("endpoint API key"),
            "the restart's own refusal must survive to the caller:\n{error:#}"
        );
        assert_eq!(
            after.runtime_id.as_deref(),
            Some(OLD_RUNTIME_KEY),
            "a service that did not move must not be described by a runtime it \
             never loaded"
        );
        assert_eq!(after.env_id.as_deref(), Some("custom-env"));
        Ok(())
    }

    #[test]
    fn a_restart_that_fails_after_the_stop_leaves_the_server_stopped_and_repinned() -> Result<()> {
        let (root, paths) = test_paths("service-restart-fails-after-stop");
        paths.ensure()?;
        let mut record = plant_service_record_on_runtime(
            &paths,
            "svc-cpu-policy",
            "vllm",
            "starting",
            // Skipped by `terminate_recorded_service_pids`, which never
            // signals this process — so the stop runs for real without the
            // test killing itself.
            std::process::id(),
            Some(OLD_RUNTIME_KEY),
            None,
        )?;
        // A device policy the managed serve path refuses. It is parsed AFTER
        // `restart_internal_managed_service` has already stopped the service,
        // which is the failure the report's "stopped by the attempt" claim and
        // the pin restore are both written for — and which the pre-stop
        // refusal test deliberately does not reach.
        record.device_policy = Some("cpu_only".to_owned());
        record.write()?;

        let error = runtime_services::restart_service_onto_runtime(
            &paths,
            "svc-cpu-policy",
            NEW_RUNTIME_KEY,
        )
        .expect_err("a refused device policy must fail the move");
        let after = load_managed_service(&paths, "svc-cpu-policy")?;
        let _ = fs::remove_dir_all(&root);

        let error = format!("{error:#}");
        assert!(
            error.contains("CPU mode is not a fallback path"),
            "moving a server onto a new runtime goes through the same managed \
             serve path as a fresh launch, so it must refuse CPU execution \
             exactly as that path does:\n{error}"
        );
        assert_eq!(
            after.runtime_id.as_deref(),
            Some(OLD_RUNTIME_KEY),
            "the restart got past the stop but never came back up, so the \
             record must not describe a runtime the server never loaded"
        );
        assert_eq!(
            after.status, "stopped",
            "the report and the docs both tell the user a failed restart left \
             the server down; if the stop had not happened that sends them \
             away from a server that is still serving"
        );
        Ok(())
    }

    #[cfg(unix)]
    fn managed_record_for_pid(
        paths: &AppPaths,
        pid: u32,
        start_ticks: Option<u64>,
    ) -> ManagedServiceRecord {
        let mut record = ManagedServiceRecord::new(
            paths,
            "svc-managed-stop",
            "vllm",
            "m",
            "m",
            "127.0.0.1",
            9,
            "managed",
            pid,
            None,
            None,
            None,
        );
        record.engine_pid = Some(pid);
        record.supervisor_pid = pid;
        record.supervisor_start_ticks = start_ticks;
        record
    }

    #[cfg(unix)]
    #[test]
    fn managed_stop_refuses_recycled_pid() {
        // A live process whose recorded identity does not match: this is exactly
        // what a recycled PID looks like. The stop must NOT signal it.
        let mut child = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn child");
        let pid = child.id();
        let (root, paths) = test_paths("managed-stop-recycled");
        let real = rocm_core::process_start_ticks(pid).expect("start-ticks");
        let record = managed_record_for_pid(&paths, pid, Some(real.wrapping_add(1)));

        let (signaled, all_stopped) = terminate_recorded_service_pids(&record);

        assert!(signaled.is_empty(), "must not signal a mismatched identity");
        // A recycled PID means the recorded service process itself is gone.
        assert!(all_stopped);
        assert!(
            rocm_core::process_is_running(pid),
            "the unrelated live process must survive"
        );
        let _ = child.kill();
        let _ = child.wait();
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn managed_stop_verifies_distinct_supervisor_and_engine_pids() {
        // The launcher and the engine server are different processes; each PID
        // must be verified with its OWN start-time token and terminated.
        let sup = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn supervisor");
        let eng = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn engine");
        let (sup_pid, eng_pid) = (sup.id(), eng.id());
        let (root, paths) = test_paths("managed-stop-distinct");
        let mut record =
            managed_record_for_pid(&paths, sup_pid, rocm_core::process_start_ticks(sup_pid));
        record.engine_pid = Some(eng_pid);
        record.engine_start_ticks = rocm_core::process_start_ticks(eng_pid);

        let (mut signaled, all_stopped) = terminate_recorded_service_pids(&record);
        signaled.sort_unstable();
        let mut expected = vec![sup_pid, eng_pid];
        expected.sort_unstable();

        assert_eq!(signaled, expected);
        assert!(all_stopped);
        let (mut sup, mut eng) = (sup, eng);
        let _ = sup.wait();
        let _ = eng.wait();
        assert!(!rocm_core::process_is_running(sup_pid));
        assert!(!rocm_core::process_is_running(eng_pid));
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn managed_stop_terminates_verified_pid() {
        // Correct identity: the recorded process is really ours, so it is stopped.
        let child = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn child");
        let pid = child.id();
        let (root, paths) = test_paths("managed-stop-verified");
        let real = rocm_core::process_start_ticks(pid).expect("start-ticks");
        let record = managed_record_for_pid(&paths, pid, Some(real));

        let (signaled, all_stopped) = terminate_recorded_service_pids(&record);

        assert_eq!(signaled, vec![pid]);
        assert!(all_stopped);
        // Reap our own child (a detached managed process would be reaped by init)
        // so the liveness check does not observe a not-yet-reaped zombie.
        let mut child = child;
        let _ = child.wait();
        assert!(
            !rocm_core::process_is_running(pid),
            "the verified process must be terminated"
        );
        let _ = fs::remove_dir_all(root);
    }
}
