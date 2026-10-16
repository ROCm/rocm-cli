// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Chat/assistant tool-call support: the chat/MCP-bridge cluster's
//! `pub(crate)` types (`ChatRocmCommandAction`, `CommandCapture`,
//! `DeterministicModelRecipe`, `DeterministicServiceRow`,
//! `DeterministicEngineRow`) and its leaf helpers — tool-call argument
//! normalization and display labels, deterministic tool-output parsing, and
//! the MCP-result/loopback-port/service-running-state helpers. `main.rs`
//! constructs `ChatRocmCommandAction` and `CommandCapture` directly; it only
//! receives `DeterministicModelRecipe` as a parameter and
//! `DeterministicServiceRow`/`DeterministicEngineRow` as return values
//! (neither of the latter two is ever named there directly). The helper
//! functions are `pub(crate)` and called from `main.rs` via qualified
//! `chat::` paths, except for a handful of private internals used only
//! within this file; nothing here calls back into `main.rs`.
//!
//! Full domain extraction, mirroring `providers.rs`. Phase 6d (ROCMAI-88) of
//! the `main.rs` modularization (see `docs/architecture.md`), split across
//! three PRs given the cluster's size: this module's current contents are
//! part 1's slice. The cluster's orchestration bodies
//! (`run_internal_mcp_call`, the `validate_chat_*` family,
//! `deterministic_rocm_tool_summary`) and the `dispatch()`/`dash_seam.rs`
//! call-site qualification are still at the crate root.

use crate::providers;
use anyhow::{Context, Result, bail};
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

fn chat_tool_display_label(name: &str) -> String {
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
    let Some(args) = normalized_chat_rocm_command_args(call).ok() else {
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

pub(crate) fn normalized_chat_rocm_command_args(
    call: &providers::ChatToolCall,
) -> Result<Vec<String>> {
    let object = call
        .arguments
        .as_object()
        .context("rocm_command arguments must be a JSON object")?;
    let values = object
        .get("args")
        .and_then(serde_json::Value::as_array)
        .context("rocm_command requires `args`")?;
    if values.is_empty() || values.len() > 64 {
        bail!("rocm_command `args` must contain 1 to 64 strings");
    }
    let mut args = Vec::with_capacity(values.len());
    for value in values {
        let arg = value
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .context("rocm_command `args` entries must be non-empty strings")?;
        if arg.contains('\0') || arg.contains('\n') || arg.contains('\r') {
            bail!("rocm_command arguments must not contain control characters");
        }
        if arg.len() > 512 {
            bail!("rocm_command argument is too long");
        }
        args.push(arg.to_owned());
    }
    if args
        .first()
        .is_some_and(|arg| arg.eq_ignore_ascii_case("rocm"))
    {
        args.remove(0);
    }
    if args.is_empty() {
        bail!("rocm_command args should omit the leading `rocm` program name");
    }
    if let Some(reason) = object.get("reason")
        && !reason.is_string()
    {
        bail!("rocm_command `reason` must be a string when present");
    }
    Ok(args)
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

pub(crate) fn managed_service_running_state(status: &str) -> &'static str {
    match status {
        "ready" | "running" => "running",
        "starting" | "recovering" => "starting",
        "failed" | "stopped" => "not_running",
        _ => "unknown",
    }
}

pub(crate) fn parse_deterministic_service_rows(tool_text: &str) -> Vec<DeterministicServiceRow> {
    let mut rows = Vec::new();
    let mut current: Option<DeterministicServiceRow> = None;
    for line in tool_text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("- ") {
            if rest.starts_with("service_id=") {
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
                    row.running_state = managed_service_running_state(&row.status).to_owned();
                }
                current = Some(row);
                continue;
            }
            if let Some(row) = current.take() {
                rows.push(row);
            }
            current = Some(DeterministicServiceRow {
                service_id: rest.to_owned(),
                ..Default::default()
            });
            continue;
        }
        let Some(row) = current.as_mut() else {
            continue;
        };
        if let Some(value) = trimmed.strip_prefix("status:") {
            row.status = value.trim().to_owned();
            row.running_state = managed_service_running_state(&row.status).to_owned();
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn rocm_command_call(args: &[&str]) -> providers::ChatToolCall {
        providers::ChatToolCall {
            id: Some("test".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({ "args": args }),
        }
    }

    #[test]
    fn command_capture_text_prefers_stdout_only_when_stderr_is_blank() {
        let output = CommandCapture {
            argv: vec!["rocm".to_owned()],
            exit_status: 0,
            stdout: "  all good  \n".to_owned(),
            stderr: "  \n".to_owned(),
        };
        assert_eq!(command_capture_text(&output), "all good");
    }

    #[test]
    fn command_capture_text_falls_back_to_stderr_when_stdout_is_blank() {
        let output = CommandCapture {
            argv: vec!["rocm".to_owned()],
            exit_status: 1,
            stdout: "   ".to_owned(),
            stderr: "boom".to_owned(),
        };
        assert_eq!(command_capture_text(&output), "stderr:\nboom");
    }

    #[test]
    fn command_capture_text_joins_both_streams_when_present() {
        let output = CommandCapture {
            argv: vec!["rocm".to_owned()],
            exit_status: 1,
            stdout: "out".to_owned(),
            stderr: "err".to_owned(),
        };
        assert_eq!(
            command_capture_text(&output),
            "stdout:\nout\n\nstderr:\nerr"
        );
    }

    #[test]
    fn mcp_tool_result_text_joins_text_content_items() {
        let value = serde_json::json!({
            "content": [
                {"type": "text", "text": "line one"},
                {"type": "image", "text": "ignored"},
                {"type": "text", "text": "line two"},
            ]
        });
        assert_eq!(mcp_tool_result_text(&value), "line one\nline two");
    }

    #[test]
    fn mcp_tool_result_text_falls_back_to_raw_value_without_text_content() {
        let value = serde_json::json!({"content": []});
        assert_eq!(mcp_tool_result_text(&value), value.to_string());
    }

    #[test]
    fn mcp_tool_result_is_error_reads_the_is_error_flag() {
        assert!(mcp_tool_result_is_error(
            &serde_json::json!({"isError": true})
        ));
        assert!(!mcp_tool_result_is_error(
            &serde_json::json!({"isError": false})
        ));
        assert!(!mcp_tool_result_is_error(&serde_json::json!({})));
    }

    #[test]
    fn chat_tool_display_label_maps_known_tools_and_humanizes_others() {
        assert_eq!(chat_tool_display_label("examine"), "Checked this computer");
        assert_eq!(
            chat_tool_display_label("gpu_snapshot"),
            "Checked GPU status"
        );
        assert_eq!(chat_tool_display_label("install_sdk"), "Install ROCm");
        assert_eq!(
            chat_tool_display_label("some_unmapped_tool"),
            "some unmapped tool"
        );
    }

    #[test]
    fn chat_tool_call_display_label_delegates_for_non_rocm_command_tools() {
        let call = providers::ChatToolCall {
            id: None,
            name: "examine".to_owned(),
            arguments: serde_json::json!({}),
        };
        assert_eq!(chat_tool_call_display_label(&call), "Checked this computer");
    }

    #[test]
    fn chat_tool_call_display_label_recognizes_rocm_command_shapes() {
        assert_eq!(
            chat_tool_call_display_label(&rocm_command_call(&["examine"])),
            "Checked this computer"
        );
        assert_eq!(
            chat_tool_call_display_label(&rocm_command_call(&["models"])),
            "Checked model recipes"
        );
        assert_eq!(
            chat_tool_call_display_label(&rocm_command_call(&["model"])),
            "Checked model recipes"
        );
        assert_eq!(
            chat_tool_call_display_label(&rocm_command_call(&["ENGINES", "List"])),
            "Checked local engines"
        );
        assert_eq!(
            chat_tool_call_display_label(&rocm_command_call(&["services"])),
            "Checked model servers"
        );
        assert_eq!(
            chat_tool_call_display_label(&rocm_command_call(&["services", "list", "-a"])),
            "Checked model servers"
        );
        assert_eq!(
            chat_tool_call_display_label(&rocm_command_call(&["services", "list", "--all"])),
            "Checked model servers"
        );
        assert_eq!(
            chat_tool_call_display_label(&rocm_command_call(&["comfyui"])),
            "Checked ComfyUI"
        );
        assert_eq!(
            chat_tool_call_display_label(&rocm_command_call(&["comfyui", "status"])),
            "Checked ComfyUI"
        );
        assert_eq!(
            chat_tool_call_display_label(&rocm_command_call(&["comfyui", "logs", "-n", "50"])),
            "Read ComfyUI logs"
        );
        assert_eq!(
            chat_tool_call_display_label(&rocm_command_call(&["config", "show"])),
            "Checked ROCm config"
        );
    }

    #[test]
    fn chat_tool_call_display_label_falls_back_for_unparseable_args() {
        let call = providers::ChatToolCall {
            id: None,
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({"args": []}),
        };
        assert_eq!(chat_tool_call_display_label(&call), "rocm command");
    }

    #[test]
    fn loopback_tcp_port_is_reachable_detects_listening_and_closed_ports() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
        let port = listener.local_addr().expect("local addr").port();
        assert!(loopback_tcp_port_is_reachable("127.0.0.1", port));
        drop(listener);

        // Port 0 can never be connected to; stands in for "closed" without
        // depending on a specific unused port staying unused.
        assert!(!loopback_tcp_port_is_reachable("127.0.0.1", 0));
    }

    #[test]
    fn parse_optional_query_reads_flag_and_positional_forms() {
        assert_eq!(
            parse_optional_query(&["--query".to_owned(), "gpu".to_owned()]).unwrap(),
            Some("gpu".to_owned())
        );
        assert_eq!(
            parse_optional_query(&["-q".to_owned(), "gpu".to_owned()]).unwrap(),
            Some("gpu".to_owned())
        );
        assert_eq!(
            parse_optional_query(&["gpu".to_owned()]).unwrap(),
            Some("gpu".to_owned())
        );
        assert_eq!(parse_optional_query(&[]).unwrap(), None);
    }

    #[test]
    fn parse_optional_query_rejects_a_dangling_flag() {
        assert!(parse_optional_query(&["--query".to_owned()]).is_err());
    }

    #[test]
    fn parse_optional_query_skips_unknown_flags_to_find_the_positional_value() {
        assert_eq!(
            parse_optional_query(&["--unknown-flag".to_owned(), "gpu".to_owned()]).unwrap(),
            Some("gpu".to_owned())
        );
    }

    #[test]
    fn parse_optional_lines_reads_flag_or_defaults() {
        assert_eq!(
            parse_optional_lines(&["--lines".to_owned(), "25".to_owned()]).unwrap(),
            25
        );
        assert_eq!(
            parse_optional_lines(&["-n".to_owned(), "25".to_owned()]).unwrap(),
            25
        );
        assert_eq!(parse_optional_lines(&[]).unwrap(), DEFAULT_LOG_TAIL_LINES);
    }

    #[test]
    fn parse_optional_lines_rejects_a_non_numeric_value() {
        assert!(parse_optional_lines(&["--lines".to_owned(), "nope".to_owned()]).is_err());
    }

    #[test]
    fn push_optional_json_cli_arg_pushes_present_values_only() {
        let object = serde_json::json!({"reason": "because"});
        let object = object.as_object().unwrap();
        let mut args = Vec::new();
        push_optional_json_cli_arg(&mut args, object, "--reason", "reason");
        push_optional_json_cli_arg(&mut args, object, "--missing", "missing");
        assert_eq!(args, vec!["--reason".to_owned(), "because".to_owned()]);
    }

    #[test]
    fn json_string_filters_missing_and_blank_values() {
        let object = serde_json::json!({"reason": "  ", "note": "ok"});
        let object = object.as_object().unwrap();
        assert_eq!(json_string(object, "note"), Some("ok".to_owned()));
        assert_eq!(json_string(object, "reason"), None);
        assert_eq!(json_string(object, "missing"), None);
    }

    #[test]
    fn parse_deterministic_model_recipes_reads_header_fields_and_notes() {
        let recipes = parse_deterministic_model_recipes(
            "\
  Qwen3-4B-Instruct-2507-GGUF aliases=[qwen, lemonade-qwen] task=chat dtype=gguf device=gpu_required min_gpu_mem=4 GiB engines=[lemonade]
lemonade: ready
warning: slow on this GPU
  Qwen/Qwen3.5-4B aliases=[qwen3.5] task=chat dtype=bfloat16 device=gpu_preferred min_gpu_mem=12 GiB engines=[vllm]
vllm: not_installed
",
        );
        assert_eq!(recipes.len(), 2);
        assert_eq!(recipes[0].canonical_id, "Qwen3-4B-Instruct-2507-GGUF");
        assert_eq!(recipes[0].min_gpu_mem_gib, Some(4));
        assert_eq!(recipes[0].engines, vec!["lemonade".to_owned()]);
        assert_eq!(
            recipes[0].engine_statuses,
            vec!["lemonade: ready".to_owned()]
        );
        assert_eq!(recipes[0].warnings, vec!["slow on this GPU".to_owned()]);
        assert_eq!(recipes[1].canonical_id, "Qwen/Qwen3.5-4B");
        assert_eq!(recipes[1].min_gpu_mem_gib, Some(12));
    }

    #[test]
    fn parse_deterministic_model_recipes_requires_aliases_bracket_in_header() {
        let recipes = parse_deterministic_model_recipes(
            "\
  NotAHeader task=chat dtype=gguf device=gpu_required min_gpu_mem=4 GiB engines=[lemonade]
  Qwen/Qwen3.5-4B aliases=[qwen3.5] task=chat dtype=bfloat16 device=gpu_preferred min_gpu_mem=12 GiB engines=[vllm]
",
        );
        assert_eq!(recipes.len(), 1);
        assert_eq!(recipes[0].canonical_id, "Qwen/Qwen3.5-4B");
    }

    #[test]
    fn parse_deterministic_service_rows_reads_inline_key_value_form() {
        let rows = parse_deterministic_service_rows(
            "  - service_id=svc-1 engine=vllm model=qwen status=ready running_state=running endpoint=127.0.0.1:8000",
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].service_id, "svc-1");
        assert_eq!(rows[0].engine, "vllm");
        assert_eq!(rows[0].status, "ready");
        assert_eq!(rows[0].running_state, "running");
        assert_eq!(rows[0].endpoint, "127.0.0.1:8000");
    }

    #[test]
    fn parse_deterministic_service_rows_derives_running_state_when_absent() {
        let rows = parse_deterministic_service_rows(
            "\
- svc-2
status: starting
engine: lemonade
model: qwen
endpoint: 127.0.0.1:8001
",
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].service_id, "svc-2");
        assert_eq!(rows[0].running_state, "starting");
    }

    #[test]
    fn parse_deterministic_service_rows_derives_running_state_for_inline_form_when_absent() {
        let rows = parse_deterministic_service_rows(
            "  - service_id=svc-3 engine=vllm model=qwen status=starting endpoint=127.0.0.1:8002",
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].running_state, "starting");
    }

    #[test]
    fn parse_deterministic_engine_rows_reads_runtime_and_note() {
        let rows = parse_deterministic_engine_rows(
            "\
* lemonade
  runtime: gfx120X-all
  note: default engine
* vllm
  runtime: gfx110X-all
",
        );
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].engine, "lemonade");
        assert_eq!(rows[0].runtime, "gfx120X-all");
        assert_eq!(rows[0].note, "default engine");
        assert_eq!(rows[1].engine, "vllm");
        assert_eq!(rows[1].note, "");
    }

    #[test]
    fn normalized_chat_rocm_command_args_strips_leading_program_name() {
        let call = rocm_command_call(&["rocm", "engines", "list"]);
        assert_eq!(
            normalized_chat_rocm_command_args(&call).unwrap(),
            vec!["engines".to_owned(), "list".to_owned()]
        );
    }

    #[test]
    fn normalized_chat_rocm_command_args_rejects_empty_and_control_characters() {
        assert_eq!(
            normalized_chat_rocm_command_args(&rocm_command_call(&[]))
                .unwrap_err()
                .to_string(),
            "rocm_command `args` must contain 1 to 64 strings"
        );
        assert!(normalized_chat_rocm_command_args(&rocm_command_call(&["rocm"])).is_err());

        for control_arg in ["engines\nlist", "engines\rlist", "engines\0list"] {
            let call = providers::ChatToolCall {
                id: None,
                name: "rocm_command".to_owned(),
                arguments: serde_json::json!({"args": [control_arg]}),
            };
            assert_eq!(
                normalized_chat_rocm_command_args(&call)
                    .unwrap_err()
                    .to_string(),
                "rocm_command arguments must not contain control characters"
            );
        }
    }

    #[test]
    fn normalized_chat_rocm_command_args_accepts_exactly_the_count_and_length_limits() {
        let max_args = vec!["engines"; 64];
        assert_eq!(
            normalized_chat_rocm_command_args(&rocm_command_call(&max_args))
                .unwrap()
                .len(),
            64
        );

        let max_len_arg = "a".repeat(512);
        let call = providers::ChatToolCall {
            id: None,
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({"args": [max_len_arg.as_str()]}),
        };
        assert_eq!(
            normalized_chat_rocm_command_args(&call).unwrap(),
            vec![max_len_arg]
        );
    }

    #[test]
    fn normalized_chat_rocm_command_args_rejects_over_limit_count_and_length() {
        let too_many_args = vec!["engines"; 65];
        assert_eq!(
            normalized_chat_rocm_command_args(&rocm_command_call(&too_many_args))
                .unwrap_err()
                .to_string(),
            "rocm_command `args` must contain 1 to 64 strings"
        );

        let too_long_arg = "a".repeat(513);
        let call = providers::ChatToolCall {
            id: None,
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({"args": [too_long_arg]}),
        };
        assert_eq!(
            normalized_chat_rocm_command_args(&call)
                .unwrap_err()
                .to_string(),
            "rocm_command argument is too long"
        );
    }

    #[test]
    fn normalized_chat_rocm_command_args_rejects_a_non_string_reason() {
        let call = providers::ChatToolCall {
            id: None,
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({"args": ["engines", "list"], "reason": 42}),
        };
        assert!(normalized_chat_rocm_command_args(&call).is_err());
    }

    #[test]
    fn managed_service_running_state_maps_every_known_status() {
        assert_eq!(managed_service_running_state("ready"), "running");
        assert_eq!(managed_service_running_state("running"), "running");
        assert_eq!(managed_service_running_state("starting"), "starting");
        assert_eq!(managed_service_running_state("recovering"), "starting");
        assert_eq!(managed_service_running_state("failed"), "not_running");
        assert_eq!(managed_service_running_state("stopped"), "not_running");
        assert_eq!(managed_service_running_state("anything-else"), "unknown");
    }
}
