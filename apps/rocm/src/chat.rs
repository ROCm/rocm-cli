// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Chat/assistant CLI rendering, local-tool-calling heuristics and
//! validation, deterministic tool-output summarization, and the MCP
//! tool-bridge (`run_internal_mcp_call`), reached from `dispatch()`'s
//! `Command::Chat`/`Command::McpCall` arms and from `dash_seam.rs`.
//!
//! Phase 6d (ROCMAI-88) of the `main.rs` modularization (see
//! `docs/architecture.md`). This first PR moves the cluster's private types
//! and the bottom-of-call-graph pure helpers that have no callers inside the
//! still-to-move remainder of the cluster; the chat/MCP-bridge function
//! bodies and the `dispatch()`/`dash_seam.rs` call-site updates land in a
//! follow-up PR.

use crate::providers;
use anyhow::{Context, Result};
use rocm_engine_protocol::DEFAULT_LOG_TAIL_LINES;
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) enum ChatRocmCommandAction {
    ReadOnly(Vec<String>),
    Approval {
        args: Vec<String>,
        pending_title: String,
        command_title: String,
    },
}

#[derive(Debug)]
pub(crate) struct CommandCapture {
    pub(crate) argv: Vec<String>,
    pub(crate) exit_status: i32,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
}

pub(crate) fn command_capture_text(output: &CommandCapture) -> String {
    if output.stderr.trim().is_empty() {
        output.stdout.trim().to_owned()
    } else if output.stdout.trim().is_empty() {
        format!("stderr:\n{}", output.stderr.trim())
    } else {
        format!(
            "stdout:\n{}\n\nstderr:\n{}",
            output.stdout.trim(),
            output.stderr.trim()
        )
    }
}

pub(crate) fn mcp_tool_result_text(value: &serde_json::Value) -> String {
    value
        .get("content")
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    (item.get("type").and_then(serde_json::Value::as_str) == Some("text"))
                        .then(|| item.get("text").and_then(serde_json::Value::as_str))
                        .flatten()
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .filter(|text| !text.trim().is_empty())
        .unwrap_or_else(|| value.to_string())
}

pub(crate) fn mcp_tool_result_is_error(value: &serde_json::Value) -> bool {
    value
        .get("isError")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

pub(crate) fn chat_tool_display_label(name: &str) -> String {
    match name {
        "examine" => "Checked this computer".to_owned(),
        "gpu_snapshot" => "Checked GPU status".to_owned(),
        "engines" => "Checked local engines".to_owned(),
        "services" => "Checked model servers".to_owned(),
        "bridge_snapshot" => "Checked ROCm state".to_owned(),
        "service_logs" => "Read server logs".to_owned(),
        "automations" => "Checked automations".to_owned(),
        "natural_language_plan" => "Planned ROCm request".to_owned(),
        "path_exists" => "Checked folder path".to_owned(),
        "port_status" => "Checked local port".to_owned(),
        "update_check" => "Checked for ROCm updates".to_owned(),
        "install_sdk_dry_run" => "Previewed ROCm install".to_owned(),
        "install_sdk" => "Install ROCm".to_owned(),
        "install_engine" => "Install engine".to_owned(),
        "launch_server" => "Start local model server".to_owned(),
        "stop_server" => "Stop local model server".to_owned(),
        "watcher_enable" => "Enable automation".to_owned(),
        "watcher_disable" => "Disable automation".to_owned(),
        other => other.replace('_', " "),
    }
}

pub(crate) fn chat_tool_call_display_label(call: &providers::ChatToolCall) -> String {
    if call.name != "rocm_command" {
        return chat_tool_display_label(&call.name);
    }
    let Some(args) = crate::normalized_chat_rocm_command_args(call).ok() else {
        return "rocm command".to_owned();
    };
    match args.as_slice() {
        [command] if command.eq_ignore_ascii_case("examine") => "Checked this computer".to_owned(),
        [command]
            if command.eq_ignore_ascii_case("model") || command.eq_ignore_ascii_case("models") =>
        {
            "Checked model recipes".to_owned()
        }
        [command, subcommand]
            if command.eq_ignore_ascii_case("engines")
                && subcommand.eq_ignore_ascii_case("list") =>
        {
            "Checked local engines".to_owned()
        }
        [command] if command.eq_ignore_ascii_case("services") => "Checked model servers".to_owned(),
        [command, subcommand, rest @ ..]
            if command.eq_ignore_ascii_case("services")
                && subcommand.eq_ignore_ascii_case("list")
                && rest
                    .iter()
                    .all(|arg| matches!(arg.as_str(), "-a" | "--all")) =>
        {
            "Checked model servers".to_owned()
        }
        [command] if command.eq_ignore_ascii_case("comfyui") => "Checked ComfyUI".to_owned(),
        [command, subcommand]
            if command.eq_ignore_ascii_case("comfyui")
                && subcommand.eq_ignore_ascii_case("status") =>
        {
            "Checked ComfyUI".to_owned()
        }
        [command, subcommand, ..]
            if command.eq_ignore_ascii_case("comfyui")
                && matches!(subcommand.to_ascii_lowercase().as_str(), "logs" | "log") =>
        {
            "Read ComfyUI logs".to_owned()
        }
        [command, subcommand]
            if command.eq_ignore_ascii_case("config")
                && subcommand.eq_ignore_ascii_case("show") =>
        {
            "Checked ROCm config".to_owned()
        }
        _ => "rocm command".to_owned(),
    }
}

pub(crate) fn loopback_tcp_port_is_reachable(host: &str, port: u16) -> bool {
    let Ok(addresses) = (host, port).to_socket_addrs() else {
        return false;
    };
    addresses
        .into_iter()
        .any(|address| TcpStream::connect_timeout(&address, Duration::from_millis(200)).is_ok())
}

pub(crate) fn parse_optional_query(args: &[String]) -> Result<Option<String>> {
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--query" | "-q" => {
                let value = args
                    .get(index + 1)
                    .context("logs query flag requires a value")?;
                return Ok(Some(value.to_owned()));
            }
            value if value.starts_with("--") => {}
            value => return Ok(Some(value.to_owned())),
        }
        index += 1;
    }
    Ok(None)
}

pub(crate) fn parse_optional_lines(args: &[String]) -> Result<usize> {
    let mut index = 0;
    while index < args.len() {
        if matches!(args[index].as_str(), "--lines" | "-n") {
            let value = args.get(index + 1).context("lines flag requires a value")?;
            return value
                .parse::<usize>()
                .context("lines flag must be a positive number");
        }
        index += 1;
    }
    Ok(DEFAULT_LOG_TAIL_LINES)
}

pub(crate) fn push_optional_json_cli_arg(
    args: &mut Vec<String>,
    object: &serde_json::Map<String, serde_json::Value>,
    flag: &str,
    key: &str,
) {
    if let Some(value) = json_string(object, key) {
        args.push(flag.to_owned());
        args.push(value);
    }
}

pub(crate) fn json_string(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<String> {
    object
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .filter(|value| !value.trim().is_empty())
}

#[derive(Debug, Default)]
pub(crate) struct DeterministicModelRecipe {
    pub(crate) canonical_id: String,
    pub(crate) min_gpu_mem_gib: Option<u32>,
    pub(crate) engines: Vec<String>,
    pub(crate) engine_statuses: Vec<String>,
    pub(crate) warnings: Vec<String>,
}

pub(crate) fn parse_deterministic_model_recipes(tool_text: &str) -> Vec<DeterministicModelRecipe> {
    let mut recipes = Vec::new();
    let mut current: Option<DeterministicModelRecipe> = None;
    for line in tool_text.lines() {
        let trimmed = line.trim();
        if let Some(recipe) = parse_model_recipe_header(trimmed) {
            if let Some(previous) = current.replace(recipe) {
                recipes.push(previous);
            }
            continue;
        }
        let Some(recipe) = current.as_mut() else {
            continue;
        };
        if is_engine_status_line(trimmed) {
            recipe.engine_statuses.push(trimmed.to_owned());
        } else if let Some(warning) = trimmed.strip_prefix("warning:") {
            recipe.warnings.push(warning.trim().to_owned());
        }
    }
    if let Some(recipe) = current {
        recipes.push(recipe);
    }
    recipes
}

fn parse_model_recipe_header(line: &str) -> Option<DeterministicModelRecipe> {
    if !line.contains(" aliases=[") || !line.contains("min_gpu_mem=") {
        return None;
    }
    let canonical_id = line.split_whitespace().next()?.to_owned();
    let engines = parse_bracketed_csv(line, "engines=[", "]");
    let min_gpu_mem_gib = line
        .split_once("min_gpu_mem=")
        .and_then(|(_, rest)| rest.split_whitespace().next())
        .and_then(|value| value.parse::<u32>().ok());
    Some(DeterministicModelRecipe {
        canonical_id,
        min_gpu_mem_gib,
        engines,
        engine_statuses: Vec::new(),
        warnings: Vec::new(),
    })
}

fn parse_bracketed_csv(line: &str, start: &str, end: &str) -> Vec<String> {
    line.split_once(start)
        .and_then(|(_, rest)| rest.split_once(end))
        .map(|(values, _)| {
            values
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn is_engine_status_line(line: &str) -> bool {
    matches!(
        line.split_once(':').map(|(engine, _)| engine),
        Some("lemonade" | "vllm")
    )
}

#[derive(Debug, Default)]
pub(crate) struct DeterministicServiceRow {
    pub(crate) service_id: String,
    pub(crate) engine: String,
    pub(crate) model: String,
    pub(crate) status: String,
    pub(crate) running_state: String,
    pub(crate) endpoint: String,
}

pub(crate) fn parse_deterministic_service_rows(tool_text: &str) -> Vec<DeterministicServiceRow> {
    let mut rows = Vec::new();
    let mut current: Option<DeterministicServiceRow> = None;
    for line in tool_text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("- service_id=") {
            if let Some(row) = current.take() {
                rows.push(row);
            }
            let mut row = DeterministicServiceRow::default();
            for token in rest.split_whitespace() {
                if let Some((key, value)) = token.split_once('=') {
                    match key {
                        "service_id" => row.service_id = value.to_owned(),
                        "engine" => row.engine = value.to_owned(),
                        "model" => row.model = value.to_owned(),
                        "status" => row.status = value.to_owned(),
                        "running_state" => row.running_state = value.to_owned(),
                        "endpoint" => row.endpoint = value.to_owned(),
                        _ => {}
                    }
                }
            }
            if row.running_state.is_empty() {
                row.running_state = crate::managed_service_running_state(&row.status).to_owned();
            }
            current = Some(row);
            continue;
        }
        if let Some(service_id) = trimmed.strip_prefix("- ") {
            if let Some(row) = current.take() {
                rows.push(row);
            }
            current = Some(DeterministicServiceRow {
                service_id: service_id.to_owned(),
                ..Default::default()
            });
            continue;
        }
        let Some(row) = current.as_mut() else {
            continue;
        };
        if let Some(value) = trimmed.strip_prefix("status:") {
            row.status = value.trim().to_owned();
            row.running_state = crate::managed_service_running_state(&row.status).to_owned();
        } else if let Some(value) = trimmed.strip_prefix("engine:") {
            row.engine = value.trim().to_owned();
        } else if let Some(value) = trimmed.strip_prefix("model:") {
            row.model = value.trim().to_owned();
        } else if let Some(value) = trimmed.strip_prefix("endpoint:") {
            row.endpoint = value.trim().to_owned();
        }
    }
    if let Some(row) = current {
        rows.push(row);
    }
    rows
}

#[derive(Debug, Default)]
pub(crate) struct DeterministicEngineRow {
    pub(crate) engine: String,
    pub(crate) runtime: String,
    pub(crate) note: String,
}

pub(crate) fn parse_deterministic_engine_rows(tool_text: &str) -> Vec<DeterministicEngineRow> {
    let mut rows = Vec::new();
    let mut current: Option<DeterministicEngineRow> = None;
    for line in tool_text.lines() {
        let trimmed = line.trim().trim_start_matches("* ").trim_start();
        if let Some(engine) = engine_name_from_inventory_line(trimmed) {
            if let Some(row) = current.take() {
                rows.push(row);
            }
            current = Some(DeterministicEngineRow {
                engine: engine.to_owned(),
                ..Default::default()
            });
            continue;
        }
        let Some(row) = current.as_mut() else {
            continue;
        };
        if let Some(runtime) = trimmed.strip_prefix("runtime:") {
            row.runtime = runtime.trim().to_owned();
        } else if let Some(note) = trimmed.strip_prefix("note:") {
            row.note = note.trim().to_owned();
        }
    }
    if let Some(row) = current {
        rows.push(row);
    }
    rows
}

fn engine_name_from_inventory_line(line: &str) -> Option<&'static str> {
    ["lemonade", "vllm"].into_iter().find(|engine| {
        line == *engine
            || line
                .strip_prefix(*engine)
                .is_some_and(|rest| rest.starts_with(char::is_whitespace))
    })
}
