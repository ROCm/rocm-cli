// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Chat/assistant tool-call support: the chat/MCP-bridge cluster's types
//! (`ChatRocmCommandAction`, `CommandCapture`, `DeterministicModelRecipe`,
//! `DeterministicServiceRow`, `DeterministicEngineRow`, and others — all
//! module-private except `ChatToolApprovalRequest`, reached by
//! `dash_seam.rs` as `chat_tool_approval_request`'s return type), its
//! tool-call validation/approval/execution logic (`validate_chat_tool_call`
//! and the `validate_chat_*_tool_call` family), deterministic tool-output
//! summarization, and the MCP tool-bridge entry points
//! (`run_internal_mcp_call` and its helpers). Every call site in `main.rs`,
//! `dash.rs`, `dash_seam.rs`, `comfyui.rs`, and `runtime_install_finalize.rs`
//! goes through the `chat::` module boundary.
//!
//! Full domain extraction, mirroring `providers.rs`. Phase 6d (ROCMAI-88) of
//! the `main.rs` modularization (see `docs/architecture.md`), split across
//! three PRs given the cluster's size: this module's contents are the
//! combined slice of all three (types and leaf helpers, the orchestration
//! bodies, and the cluster's regression tests moved out of `main.rs`'s
//! shared `mod tests` block), plus the third PR's own visibility audit —
//! every function still `pub(crate)` has a confirmed caller outside this
//! module; none of the module's types need to be, bar the one exception
//! above.

use crate::runtime_install_finalize::empty_as_unknown;
use crate::runtimes_cmd::render_runtimes_text;
use crate::service_launch::is_loopback_host;
use crate::{
    ChatInferenceParams, apply_app_path_env, build_codex_bridge_gpu_snapshot,
    build_codex_bridge_snapshot, builtin_codex_bridge_engine_inventory, cli_version_string,
    format_structured_tool_call, freeform_plan_next_action_with_context, friendly_engine_label,
    load_managed_services, managed_service_is_live, read_tail_lines, record_cli_audit_event,
    render_automations_text, render_config_text, render_daemon_text,
    render_engine_inventory_text_with_paths, render_examine_text, render_examine_text_with_paths,
    render_freeform_plan, render_logs_browser_text, render_logs_text,
    render_model_registry_verbose_text_with_context_and_host, render_service_logs_text,
    render_services_text, render_setup_status_text, render_uninstall_dry_run, render_update_text,
    serve_args_request_cpu_device, service_model_names_match, therock_install_version_selector,
    validate_service_id,
};
use crate::{chat_host_facts, comfyui, providers, therock};
use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, AutomationRuntimeState, DEFAULT_LOCAL_HOST, ExamineSummary, ManagedServiceRecord,
    RocmCliConfig, builtin_watcher, daemon_binary_path, load_recent_automation_events,
    runtime_install_root_is_protected,
};
use rocm_engine_protocol::DEFAULT_LOG_TAIL_LINES;
use std::fmt::Write as _;
use std::io::Read as _;
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;
use std::time::Duration;

#[derive(Debug, Clone, Eq, PartialEq)]
enum ChatRocmCommandAction {
    ReadOnly(Vec<String>),
    Approval {
        args: Vec<String>,
        pending_title: String,
        command_title: String,
    },
}

#[derive(Debug)]
struct CommandCapture {
    pub(crate) argv: Vec<String>,
    pub(crate) exit_status: i32,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
}

fn command_capture_text(output: &CommandCapture) -> String {
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

fn mcp_tool_result_text(value: &serde_json::Value) -> String {
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

fn mcp_tool_result_is_error(value: &serde_json::Value) -> bool {
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

fn chat_tool_call_display_label(call: &providers::ChatToolCall) -> String {
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

fn loopback_tcp_port_is_reachable(host: &str, port: u16) -> bool {
    let Ok(addresses) = (host, port).to_socket_addrs() else {
        return false;
    };
    addresses
        .into_iter()
        .any(|address| TcpStream::connect_timeout(&address, Duration::from_millis(200)).is_ok())
}

fn parse_optional_query(args: &[String]) -> Result<Option<String>> {
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

fn parse_optional_lines(args: &[String]) -> Result<usize> {
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

fn push_optional_json_cli_arg(
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

fn json_string(object: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<String> {
    object
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .filter(|value| !value.trim().is_empty())
}

#[derive(Debug, Default)]
struct DeterministicModelRecipe {
    pub(crate) canonical_id: String,
    pub(crate) min_gpu_mem_gib: Option<u32>,
    pub(crate) engines: Vec<String>,
    pub(crate) engine_statuses: Vec<String>,
    pub(crate) warnings: Vec<String>,
}

fn parse_deterministic_model_recipes(tool_text: &str) -> Vec<DeterministicModelRecipe> {
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
struct DeterministicServiceRow {
    pub(crate) service_id: String,
    pub(crate) engine: String,
    pub(crate) model: String,
    pub(crate) status: String,
    pub(crate) running_state: String,
    pub(crate) endpoint: String,
}

fn parse_deterministic_service_rows(tool_text: &str) -> Vec<DeterministicServiceRow> {
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
struct DeterministicEngineRow {
    pub(crate) engine: String,
    pub(crate) runtime: String,
    pub(crate) note: String,
}

fn parse_deterministic_engine_rows(tool_text: &str) -> Vec<DeterministicEngineRow> {
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

/// Read a one-shot chat prompt from standard input when it is piped in.
///
/// Backs the documented `echo "…" | rocm chat` path: when `--prompt` is omitted
/// and stdin is not a terminal, the piped text becomes the prompt. Returns
/// `None` when stdin is an interactive TTY or the piped input is blank, so the
/// caller falls back to the status screen instead of blocking on input nobody
/// can supply.
///
/// The read runs to EOF — the conventional filter contract that
/// `read_provider_key_from_user` already follows. A caller that hands us a pipe
/// it never writes to and never closes therefore waits, exactly as `cat` would;
/// the TTY guard is what keeps that off an interactive user, and a
/// non-interactive caller with no prompt to send should pass `/dev/null` (as
/// `scripts/smoke_local.py` does) rather than an idle pipe. A read error — a
/// closed or non-UTF-8 fd 0 — is reported instead of being folded into "no
/// prompt", so text that was piped but could not be decoded fails loudly rather
/// than silently becoming a status screen and a zero exit.
pub(crate) fn read_piped_prompt() -> Result<Option<String>> {
    use std::io::IsTerminal as _;

    if std::io::stdin().is_terminal() {
        return Ok(None);
    }
    let mut buf = String::new();
    std::io::stdin()
        .read_to_string(&mut buf)
        .context("failed to read chat prompt from standard input")?;
    Ok(piped_prompt_from_input(&buf))
}

/// Turn raw piped stdin into a chat prompt, or `None` when there is nothing to
/// send.
///
/// `--prompt` reaches the send path verbatim, so a piped prompt must too:
/// leading indentation and trailing spaces or tabs carry meaning for a model and
/// are preserved byte for byte. The only thing removed is the line ending the
/// writer appends — a single trailing `\n`, plus the `\r` in front of it on
/// Windows — because `echo "…" |` and `printf '…\n' |` add it, not the user.
/// Further blank lines stay: the second newline of `printf 'a\n\n'` is authored
/// content, not shell punctuation.
///
/// `trim()` is used only to classify the input: whitespace-only stdin (an empty
/// pipe, or a bare newline) holds no prompt and yields `None`. That is the one
/// deliberate divergence from `--prompt`, which forwards `"   "` as written: a
/// bare `rocm chat` under any redirect — `< /dev/null`, a closed heredoc, a CI
/// step with no stdin — has to keep printing the status screen rather than send
/// a blank turn to a model, and a caller who really means to send whitespace
/// can still say so with `--prompt`.
fn piped_prompt_from_input(input: &str) -> Option<String> {
    if input.trim().is_empty() {
        return None;
    }
    let prompt = match input.strip_suffix('\n') {
        Some(rest) => rest.strip_suffix('\r').unwrap_or(rest),
        None => input,
    };
    Some(prompt.to_owned())
}

pub(crate) fn render_chat_text(paths: &AppPaths, provider: &str) -> Result<String> {
    let status = providers::provider_status(paths, provider)?;
    let mut output = String::new();
    let _ = writeln!(output, "Chat assistant");
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "Assistant source: {}",
        if status.provider == "local" {
            "local model on this computer"
        } else {
            status.provider.as_str()
        }
    );
    let _ = writeln!(
        output,
        "Status: {}",
        plain_provider_auth_status(&status.auth_status)
    );
    let _ = writeln!(
        output,
        "ROCm help: {}",
        if status.tool_call_schema.is_empty() {
            "not available"
        } else {
            "available"
        }
    );
    if status.models.is_empty() {
        let _ = writeln!(output, "Models: none found yet");
    } else {
        let _ = writeln!(output, "Models:");
        for model in status.models {
            let _ = writeln!(output, "  - {model}");
        }
    }
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "Choose Chat source to switch between local and remote assistants."
    );
    let _ = writeln!(output, "Choose Settings to manage saved keys and defaults.");
    Ok(output)
}

fn plain_provider_auth_status(status: &str) -> String {
    if status == "ready" {
        "Ready".to_owned()
    } else if status == "no_ready_local_service" {
        "No local model server is ready".to_owned()
    } else if status.starts_with("disabled:") {
        "Disabled until you enable this provider".to_owned()
    } else {
        status.replace('_', " ")
    }
}

pub(crate) fn render_chat_prompt_text(
    paths: &AppPaths,
    provider: &str,
    model: Option<&str>,
    prompt: &str,
    rocm_tools: bool,
    params: ChatInferenceParams,
) -> Result<String> {
    Ok(render_chat_prompt_result_with_progress(
        paths, provider, model, prompt, rocm_tools, params, None,
    )?
    .rendered)
}

/// Parse a `--temperature` value: a finite number greater than or equal to 0.
pub(crate) fn parse_non_negative_temperature(value: &str) -> Result<f32, String> {
    let parsed: f32 = value
        .parse()
        .map_err(|_| format!("`{value}` is not a valid number"))?;
    if !parsed.is_finite() || parsed < 0.0 {
        return Err("temperature must be a finite value >= 0.0".to_owned());
    }
    Ok(parsed)
}

/// Parse a `--top-p` value: a finite number in the inclusive range 0.0..=1.0.
pub(crate) fn parse_unit_interval(value: &str) -> Result<f32, String> {
    let parsed: f32 = value
        .parse()
        .map_err(|_| format!("`{value}` is not a valid number"))?;
    if !parsed.is_finite() || !(0.0..=1.0).contains(&parsed) {
        return Err("top-p must be between 0.0 and 1.0".to_owned());
    }
    Ok(parsed)
}

pub(crate) fn parse_positive_u32(value: &str) -> Result<u32, String> {
    let parsed: u32 = value
        .parse()
        .map_err(|_| format!("`{value}` is not a valid positive integer"))?;
    if parsed == 0 {
        return Err("max-tokens must be greater than 0".to_owned());
    }
    Ok(parsed)
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct ChatToolApprovalRequest {
    pub pending_title: String,
    pub command_title: String,
    pub args: Vec<String>,
    pub display_command: Option<String>,
    pub explanation: Option<String>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct ChatPromptResult {
    pub rendered: String,
    pub approval: Option<ChatToolApprovalRequest>,
}

struct ChatToolRunResult {
    pub(crate) approval: Option<ChatToolApprovalRequest>,
    pub(crate) follow_up_text: String,
    pub(crate) ran_read_only_tool: bool,
    pub(crate) read_only_tool_error: bool,
    pub(crate) needs_install_folder: bool,
}

/// Convenience wrapper used by tests that do not exercise sampling controls.
#[cfg(test)]
fn render_chat_prompt_result(
    paths: &AppPaths,
    provider: &str,
    model: Option<&str>,
    prompt: &str,
    rocm_tools: bool,
) -> Result<ChatPromptResult> {
    render_chat_prompt_result_with_progress(
        paths,
        provider,
        model,
        prompt,
        rocm_tools,
        ChatInferenceParams::default(),
        None,
    )
}

fn render_chat_prompt_result_with_progress(
    paths: &AppPaths,
    provider: &str,
    model: Option<&str>,
    prompt: &str,
    rocm_tools: bool,
    params: ChatInferenceParams,
    progress: Option<&mut dyn FnMut(String)>,
) -> Result<ChatPromptResult> {
    let mut progress = progress;
    let user_prompt = latest_user_chat_message(prompt);
    let assistant_model = local_rocm_tools_assistant_model(provider, rocm_tools).or(model);
    let service_needed_model = if local_rocm_tools_assistant_model(provider, rocm_tools).is_some() {
        None
    } else {
        model
    };
    if rocm_tools && let Some(approval) = install_sdk_without_prefix_chat_approval(user_prompt) {
        report_chat_tool_progress(&mut progress, "Asking the assistant.");
        report_chat_tool_progress(&mut progress, "Review needed: Install ROCm");
        report_chat_tool_progress(&mut progress, "Waiting for the ROCm install folder.");
        return Ok(ChatPromptResult {
            rendered: render_install_sdk_folder_needed_chat_text(user_prompt, &approval.args),
            approval: Some(approval),
        });
    }
    let mut messages = Vec::new();
    if rocm_tools {
        messages.push(providers::ChatMessage {
            role: "system".to_owned(),
            content: rocm_chat_tool_system_prompt_for_host(Some(paths)),
        });
    }
    messages.push(providers::ChatMessage {
        role: "user".to_owned(),
        content: prompt.to_owned(),
    });
    let mut response = if rocm_tools {
        report_chat_tool_progress(&mut progress, "Asking the assistant.");
        if let Some(call) = deterministic_mutating_tool_call_for_prompt(user_prompt)? {
            report_chat_tool_progress(&mut progress, "Preparing a review card.");
            providers::ChatResponse {
                provider: provider.to_owned(),
                model: assistant_model.unwrap_or("local").to_owned(),
                content: deterministic_mutating_tool_intro(&call),
                tool_calls: vec![call],
            }
        } else if !local_assistant_service_ready_for_chat(paths, assistant_model)
            && let Some(response) =
                local_read_only_fallback_response_for_prompt(provider, assistant_model, user_prompt)
        {
            report_chat_tool_progress(&mut progress, "Running ROCm status check.");
            response
        } else {
            match providers::provider_chat(
                paths,
                provider,
                &providers::ChatRequest {
                    model: assistant_model.map(str::to_owned),
                    messages: messages.clone(),
                    max_tokens: params.max_tokens,
                    temperature: params.temperature,
                    top_p: params.top_p,
                    rocm_tools,
                },
            ) {
                Ok(response) => response,
                Err(error)
                    if provider == "local" && local_provider_missing_service_error(&error) =>
                {
                    if let Some(response) = local_read_only_fallback_response_for_prompt(
                        provider,
                        assistant_model,
                        user_prompt,
                    ) {
                        report_chat_tool_progress(&mut progress, "Running ROCm status check.");
                        response
                    } else {
                        return Ok(ChatPromptResult {
                            rendered: local_chat_service_needed_text(
                                service_needed_model,
                                user_prompt,
                                rocm_tools,
                            ),
                            approval: None,
                        });
                    }
                }
                Err(error) => return Err(error),
            }
        }
    } else {
        report_chat_tool_progress(&mut progress, "Asking the assistant.");
        match providers::provider_chat(
            paths,
            provider,
            &providers::ChatRequest {
                model: assistant_model.map(str::to_owned),
                messages: messages.clone(),
                max_tokens: params.max_tokens,
                temperature: params.temperature,
                top_p: params.top_p,
                rocm_tools,
            },
        ) {
            Ok(response) => response,
            Err(error) if provider == "local" && local_provider_missing_service_error(&error) => {
                return Ok(ChatPromptResult {
                    rendered: local_chat_service_needed_text(
                        service_needed_model,
                        user_prompt,
                        rocm_tools,
                    ),
                    approval: None,
                });
            }
            Err(error) => return Err(error),
        }
    };
    let fallback_tool_call = if rocm_tools && response.tool_calls.is_empty() {
        fallback_rocm_tool_call_for_prompt(user_prompt)
    } else {
        None
    };
    let fallback_tool_call_used = fallback_tool_call.is_some();
    if let Some(call) = fallback_tool_call {
        response.tool_calls.push(call);
    }
    if rocm_tools
        && let Some(call) =
            supplemental_read_only_tool_call_for_prompt(user_prompt, &response.tool_calls)
    {
        response.tool_calls.push(call);
    }
    let mut output = String::new();
    let _ = writeln!(output, "chat response");
    let _ = writeln!(output, "  provider: {}", response.provider);
    let _ = writeln!(output, "  model: {}", response.model);
    let _ = writeln!(
        output,
        "  rocm tools: {}",
        if rocm_tools { "enabled" } else { "off" }
    );
    let _ = writeln!(output);
    let initial_content_is_intermediate = fallback_tool_call_used
        || local_tool_call_content_is_intermediate(provider, rocm_tools, &response);
    let initial_content = visible_chat_content(&response.content);
    if initial_content_is_intermediate {
        let _ = writeln!(output, "Assistant is preparing the next step.");
    } else if !initial_content.trim().is_empty() {
        let _ = writeln!(output, "{initial_content}");
    }
    let tool_result = if rocm_tools {
        let explanation = (!fallback_tool_call_used && !initial_content.trim().is_empty())
            .then_some(initial_content.as_str());
        append_chat_tool_results(paths, &response, &mut output, explanation, &mut progress)?
    } else {
        ChatToolRunResult {
            approval: None,
            follow_up_text: String::new(),
            ran_read_only_tool: false,
            read_only_tool_error: false,
            needs_install_folder: false,
        }
    };
    let deterministic_summary = deterministic_chat_tool_summary(&tool_result.follow_up_text);
    if let Some(summary) = deterministic_summary.as_deref() {
        let _ = writeln!(output);
        let _ = writeln!(output, "ROCm CLI summary");
        let _ = writeln!(output, "{summary}");
    }
    let follow_up_blocking_summary = deterministic_summary
        .as_deref()
        .filter(|_| deterministic_summary_can_stand_alone(&tool_result.follow_up_text));
    if should_request_local_tool_follow_up(provider, &tool_result, follow_up_blocking_summary) {
        let follow_up_messages = vec![
            providers::ChatMessage {
                role: "system".to_owned(),
                content: "You are ROCm CLI's local assistant. ROCm tools have already been checked for this turn. Use the supplied tool results for local facts about this machine. For service results, ready/running means running, starting/recovering means starting, failed/stopped means not running, and no matching service row means ROCm CLI is not managing that service as running. If the tool results do not contain enough information to answer the whole question, say what is known from the results and then answer the rest normally from your own model knowledge. Do not request another tool call. Do not invent local paths, GPU names, versions, or install state that are not shown in the tool results. Keep the answer concise.".to_owned(),
            },
            providers::ChatMessage {
                role: "user".to_owned(),
                content: format!(
                    "Original request:\n{}\n\nROCm tool results:\n{}\n\nAnswer the original request. Use the ROCm tool results for this machine's facts; if they are incomplete, answer the remaining part generally.",
                    user_prompt.trim(),
                    tool_result.follow_up_text.trim()
                ),
            },
        ];
        let follow_up = providers::provider_chat(
            paths,
            provider,
            &providers::ChatRequest {
                model: assistant_model.map(str::to_owned),
                messages: follow_up_messages,
                max_tokens: params.max_tokens.or(Some(256)),
                temperature: params.temperature,
                top_p: params.top_p,
                rocm_tools: false,
            },
        )?;
        if follow_up.tool_calls.is_empty() {
            let follow_up_content = visible_chat_content(&follow_up.content);
            if local_follow_up_content_is_final(&follow_up) && !follow_up_content.trim().is_empty()
            {
                let _ = writeln!(output);
                let _ = writeln!(output, "Assistant after ROCm checks");
                let _ = writeln!(output, "{}", follow_up_content.trim());
            }
        } else {
            let _ = writeln!(output);
            let _ = writeln!(
                output,
                "Assistant asked for another ROCm check after the first tool results."
            );
            let _ = writeln!(
                output,
                "rocm-cli stopped there so the answer is not based on another guess. Nothing was changed."
            );
        }
    } else if tool_result.read_only_tool_error {
        let _ = writeln!(output);
        let _ = writeln!(
            output,
            "Assistant did not answer from this result because a ROCm check reported an error."
        );
        let _ = writeln!(output, "Nothing was changed.");
    } else if tool_result.needs_install_folder {
        let _ = writeln!(output);
        let _ = writeln!(output, "Assistant after ROCm checks");
        let _ = writeln!(
            output,
            "I can install ROCm/TheRock, but I need the install folder first. Type the folder you want to use, for example D:\\ROCm\\therock_venvs. Nothing will download until ROCm CLI shows a review card and you approve it."
        );
    }
    if tool_result.approval.is_none()
        && tool_result.ran_read_only_tool
        && !output.contains("Nothing was changed.")
    {
        let _ = writeln!(output);
        let _ = writeln!(output, "Nothing was changed.");
    }
    Ok(ChatPromptResult {
        rendered: output,
        approval: tool_result.approval,
    })
}

fn report_chat_tool_progress(progress: &mut Option<&mut dyn FnMut(String)>, message: &str) {
    if let Some(progress) = progress.as_deref_mut() {
        progress(message.to_owned());
    }
}

fn latest_user_chat_message(prompt: &str) -> &str {
    const MARKER: &str = "\nNew message:\n";
    prompt
        .rfind(MARKER)
        .map_or(prompt, |index| &prompt[index + MARKER.len()..])
        .trim()
}

fn deterministic_mutating_tool_call_for_prompt(
    prompt: &str,
) -> Result<Option<providers::ChatToolCall>> {
    let Some(call) = fallback_rocm_tool_call_for_prompt(prompt) else {
        return Ok(None);
    };
    validate_chat_tool_call(&call)?;
    if chat_tool_call_is_read_only(&call) {
        Ok(None)
    } else {
        Ok(Some(call))
    }
}

fn supplemental_read_only_tool_call_for_prompt(
    prompt: &str,
    existing_calls: &[providers::ChatToolCall],
) -> Option<providers::ChatToolCall> {
    let normalized = latest_user_chat_message(prompt).to_ascii_lowercase();
    if prompt_mentions_serving_engine_or_service(&normalized)
        && prompt_asks_running_or_status(&normalized)
        && prompt_asks_engine_install_state(&normalized)
    {
        for call in [
            fallback_services_list_tool_call(),
            fallback_engine_list_tool_call(),
        ] {
            if !chat_tool_calls_include_equivalent(existing_calls, &call) {
                return Some(call);
            }
        }
        return None;
    }
    let call = fallback_rocm_tool_call_for_prompt(prompt)?;
    if !chat_tool_call_is_read_only(&call)
        || chat_tool_calls_include_equivalent(existing_calls, &call)
    {
        return None;
    }
    Some(call)
}

fn chat_tool_calls_include_equivalent(
    existing_calls: &[providers::ChatToolCall],
    required: &providers::ChatToolCall,
) -> bool {
    existing_calls.iter().any(|existing| {
        if existing.name != required.name {
            return false;
        }
        if required.name == "rocm_command" {
            return normalized_chat_rocm_command_args(existing).ok()
                == normalized_chat_rocm_command_args(required).ok();
        }
        if required.name == "port_status" {
            return chat_port_status_tool_calls_equivalent(existing, required);
        }
        existing.arguments == required.arguments
    })
}

fn chat_port_status_tool_calls_equivalent(
    left: &providers::ChatToolCall,
    right: &providers::ChatToolCall,
) -> bool {
    let Some(left_object) = left.arguments.as_object() else {
        return false;
    };
    let Some(right_object) = right.arguments.as_object() else {
        return false;
    };
    let left_port = left_object.get("port").and_then(serde_json::Value::as_u64);
    let right_port = right_object.get("port").and_then(serde_json::Value::as_u64);
    if left_port != right_port {
        return false;
    }
    let left_host =
        json_string(left_object, "host").unwrap_or_else(|| DEFAULT_LOCAL_HOST.to_owned());
    let right_host =
        json_string(right_object, "host").unwrap_or_else(|| DEFAULT_LOCAL_HOST.to_owned());
    loopback_host_key(&left_host) == loopback_host_key(&right_host)
}

fn loopback_host_key(host: &str) -> String {
    match host.trim().to_ascii_lowercase().as_str() {
        "localhost" | "127.0.0.1" => "127.0.0.1".to_owned(),
        "::1" | "[::1]" => "::1".to_owned(),
        other => other.to_owned(),
    }
}

fn deterministic_mutating_tool_intro(call: &providers::ChatToolCall) -> String {
    match chat_tool_approval_request(call, None) {
        Ok(approval) if approval.pending_title == "Install ROCm" => {
            "I can install ROCm/TheRock into the folder you chose. Review the card before anything downloads or changes."
                .to_owned()
        }
        Ok(approval) if approval.pending_title == "Install ComfyUI" => {
            "I can install ComfyUI into ROCm CLI's managed app folder. Review the card before anything downloads or changes."
                .to_owned()
        }
        Ok(approval) if approval.pending_title == "Start ComfyUI" => {
            "I can start ComfyUI for you. Review the card before ROCm CLI launches it."
                .to_owned()
        }
        Ok(approval) if approval.pending_title == "Start local model server" => {
            "I can start the recommended local model server on the GPU. Review the card before ROCm CLI launches it."
                .to_owned()
        }
        Ok(approval) => format!(
            "I can prepare this ROCm change: {}. Review the card before anything runs.",
            approval.pending_title
        ),
        Err(_) => {
            "I can prepare this ROCm change. Review the card before anything runs.".to_owned()
        }
    }
}

fn local_tool_call_content_is_intermediate(
    provider: &str,
    rocm_tools: bool,
    response: &providers::ChatResponse,
) -> bool {
    provider == "local"
        && rocm_tools
        && !response.tool_calls.is_empty()
        && !response.content.trim().is_empty()
}

fn local_follow_up_content_is_final(response: &providers::ChatResponse) -> bool {
    response.tool_calls.is_empty() && !response.content.trim().is_empty()
}

fn local_read_only_fallback_response_for_prompt(
    provider: &str,
    assistant_model: Option<&str>,
    prompt: &str,
) -> Option<providers::ChatResponse> {
    if provider != "local" || !prompt_can_use_read_only_without_local_assistant(prompt) {
        return None;
    }
    let call = fallback_rocm_tool_call_for_prompt(prompt)?;
    if !chat_tool_call_is_read_only(&call) {
        return None;
    }
    Some(providers::ChatResponse {
        provider: provider.to_owned(),
        model: assistant_model.unwrap_or("local").to_owned(),
        content: String::new(),
        tool_calls: vec![call],
    })
}

fn local_assistant_service_ready_for_chat(paths: &AppPaths, assistant_model: Option<&str>) -> bool {
    let model = assistant_model.unwrap_or(providers::BUILTIN_ASSISTANT_MODEL_ID);
    load_managed_services(paths).is_ok_and(|records| {
        records.iter().any(|record| {
            matches!(record.status.as_str(), "ready" | "running")
                && (service_model_names_match(&record.canonical_model_id, model)
                    || service_model_names_match(&record.model_ref, model))
        })
    })
}

fn prompt_can_use_read_only_without_local_assistant(prompt: &str) -> bool {
    let normalized = latest_user_chat_message(prompt).to_ascii_lowercase();
    let mentions_status_subject = any_substring(
        &normalized,
        &[
            "comfyui",
            "comfy ui",
            "comfy",
            "vllm",
            "lemonade",
            "llama.cpp",
            "llama cpp",
            "qwen",
            "model server",
            "local server",
            "local model server",
            "assistant server",
            "port",
            "8188",
            "therock",
            "rocm",
        ],
    );
    mentions_status_subject
        && (prompt_asks_running_or_status(&normalized)
            || any_substring(
                &normalized,
                &["installed", "available", "detected", "engine status"],
            ))
}

fn prompt_mentions_serving_engine_or_service(normalized_prompt: &str) -> bool {
    any_substring(
        normalized_prompt,
        &[
            "vllm",
            "lemonade",
            "llama.cpp",
            "llama cpp",
            "qwen",
            "model server",
            "local server",
            "local model server",
            "assistant server",
        ],
    )
}

fn prompt_asks_engine_install_state(normalized_prompt: &str) -> bool {
    any_substring(
        normalized_prompt,
        &["installed", "available", "detected", "engine status"],
    )
}

fn fallback_engine_list_tool_call() -> providers::ChatToolCall {
    providers::ChatToolCall {
        id: Some("fallback-engine-list".to_owned()),
        name: "rocm_command".to_owned(),
        arguments: serde_json::json!({ "args": ["engines", "list"] }),
    }
}

fn fallback_services_list_tool_call() -> providers::ChatToolCall {
    providers::ChatToolCall {
        id: Some("fallback-services-list".to_owned()),
        name: "rocm_command".to_owned(),
        arguments: serde_json::json!({ "args": ["services", "list", "--all"] }),
    }
}

fn fallback_rocm_tool_call_for_prompt(prompt: &str) -> Option<providers::ChatToolCall> {
    let prompt = latest_user_chat_message(prompt);
    let normalized = prompt.to_ascii_lowercase();
    let asks_running_or_status = prompt_asks_running_or_status(&normalized);
    let mentions_comfyui = any_substring(&normalized, &["comfyui", "comfy ui", "comfy"]);
    if asks_running_or_status
        && !mentions_comfyui
        && any_substring(&normalized, &["8188", "port 8188"])
    {
        return Some(providers::ChatToolCall {
            id: Some("fallback-port-8188-status".to_owned()),
            name: "port_status".to_owned(),
            arguments: serde_json::json!({ "host": DEFAULT_LOCAL_HOST, "port": 8188 }),
        });
    }
    let mentions_serving_engine_or_service = prompt_mentions_serving_engine_or_service(&normalized);
    if mentions_serving_engine_or_service && prompt_asks_engine_install_state(&normalized) {
        return Some(fallback_engine_list_tool_call());
    }
    if mentions_serving_engine_or_service && asks_running_or_status {
        return Some(fallback_services_list_tool_call());
    }
    let mentions_llm_or_model = any_substring(
        &normalized,
        &["llm", "llms", "model", "models", "assistant"],
    );
    let asks_support_or_fit = any_substring(
        &normalized,
        &[
            "support",
            "supported",
            "run",
            "runs",
            "fit",
            "fits",
            "can my machine",
            "can this machine",
        ],
    );
    if mentions_llm_or_model && asks_support_or_fit {
        return Some(providers::ChatToolCall {
            id: Some("fallback-rocm-model".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({ "args": ["model"] }),
        });
    }

    if mentions_comfyui {
        if any_substring(&normalized, &["log", "logs"]) {
            return Some(providers::ChatToolCall {
                id: Some("fallback-comfyui-logs".to_owned()),
                name: "rocm_command".to_owned(),
                arguments: serde_json::json!({ "args": ["comfyui", "logs"] }),
            });
        }
        if asks_running_or_status {
            return Some(providers::ChatToolCall {
                id: Some("fallback-comfyui-status".to_owned()),
                name: "rocm_command".to_owned(),
                arguments: serde_json::json!({ "args": ["comfyui", "status"] }),
            });
        }
        if any_substring(&normalized, &["start", "run", "launch", "open"]) {
            return Some(providers::ChatToolCall {
                id: Some("fallback-comfyui-start".to_owned()),
                name: "rocm_command".to_owned(),
                arguments: serde_json::json!({
                    "args": ["comfyui", "start"],
                    "reason": "Start ComfyUI locally after the user approves it."
                }),
            });
        }
        if any_substring(
            &normalized,
            &[
                "can you setup",
                "can you set up",
                "please setup",
                "please set up",
                "setup comfyui for me",
                "set up comfyui for me",
                "install comfyui",
                "download comfyui",
            ],
        ) {
            return Some(providers::ChatToolCall {
                id: Some("fallback-comfyui-install".to_owned()),
                name: "rocm_command".to_owned(),
                arguments: serde_json::json!({
                    "args": ["comfyui", "install"],
                    "reason": "Install ComfyUI into ROCm CLI's managed app folder after the user approves it."
                }),
            });
        }
        return Some(providers::ChatToolCall {
            id: Some("fallback-comfyui-status".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({ "args": ["comfyui", "status"] }),
        });
    }

    if let Some(call) = fallback_config_tool_call_for_prompt(&normalized) {
        return Some(call);
    }

    if mentions_llm_or_model
        && any_substring(
            &normalized,
            &[
                "serve",
                "server",
                "start",
                "setup and serve",
                "set up and serve",
                "run locally",
                "local model",
                "local assistant",
            ],
        )
    {
        return Some(providers::ChatToolCall {
            id: Some("fallback-serve-qwen".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["serve", "qwen", "--engine", "lemonade", "--device", "gpu_required", "--managed"],
                "reason": "Start the recommended local assistant after the user approves it."
            }),
        });
    }

    let mentions_setup = any_substring(&normalized, &["setup", "set up", "install"]);
    let asks_how = any_substring(&normalized, &["how", "help", "what do i need"]);
    let mentions_rocm_or_therock = any_substring(&normalized, &["therock", "rocm"]);
    let requested_install_prefix = requested_install_prefix_from_prompt(prompt);
    if requested_install_prefix.is_none()
        && let Some(approval) = install_sdk_without_prefix_chat_approval(prompt)
    {
        return Some(providers::ChatToolCall {
            id: Some("fallback-therock-install-folder".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": approval.args,
                "reason": "Ask the user to choose the ROCm/TheRock install folder before installing."
            }),
        });
    }
    if mentions_rocm_or_therock
        && prompt_requests_install_action(&normalized)
        && let Some(prefix) = requested_install_prefix
    {
        let mut args = vec![
            "install".to_owned(),
            "sdk".to_owned(),
            "--channel".to_owned(),
            "release".to_owned(),
            "--format".to_owned(),
            "wheel".to_owned(),
            "--prefix".to_owned(),
            prefix,
        ];
        if let Some(build_date) = requested_therock_build_date_from_prompt(&normalized) {
            args.push("--build-date".to_owned());
            args.push(build_date);
        } else if let Some(version) = requested_therock_version_from_prompt(&normalized) {
            args.push("--version".to_owned());
            args.push(version);
        }
        return Some(providers::ChatToolCall {
            id: Some("fallback-therock-install".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": args,
                "reason": "Install TheRock ROCm into the user-selected folder after the user approves it."
            }),
        });
    }
    if asks_how && mentions_setup && any_substring(&normalized, &["therock", "rocm"]) {
        return Some(providers::ChatToolCall {
            id: Some("fallback-therock-examine".to_owned()),
            name: "examine".to_owned(),
            arguments: serde_json::json!({}),
        });
    }

    let asks_where_installed = any_substring(
        &normalized,
        &[
            "where is rocm",
            "where's rocm",
            "where is therock",
            "where's therock",
            "where did rocm",
            "where did therock",
            "where rocm is installed",
            "where therock is installed",
            "rocm install folder",
            "therock install folder",
            "rocm installed at",
            "therock installed at",
        ],
    );
    let asks_status = asks_where_installed
        || any_substring(
            &normalized,
            &[
                "is rocm installed",
                "is therock installed",
                "is therock setup",
                "is therock set up",
                "rocm installed",
                "therock installed",
                "check this rocm setup",
                "which gpu",
                "what gpu",
                "gpu is on",
                "gpu do i have",
                "my machine",
            ],
        );
    if asks_status && any_substring(&normalized, &["gpu", "rocm", "therock", "setup"]) {
        return Some(providers::ChatToolCall {
            id: Some("fallback-examine".to_owned()),
            name: "examine".to_owned(),
            arguments: serde_json::json!({}),
        });
    }

    None
}

fn prompt_asks_running_or_status(normalized: &str) -> bool {
    any_substring(
        normalized,
        &[
            "running",
            "is it up",
            "is this up",
            "is there",
            "are there",
            "status",
            "started",
            "listening",
            "on port",
            "port ",
        ],
    )
}

fn fallback_config_tool_call_for_prompt(normalized: &str) -> Option<providers::ChatToolCall> {
    if !any_substring(
        normalized,
        &[
            "config",
            "setting",
            "settings",
            "default engine",
            "telemetry",
        ],
    ) {
        return None;
    }
    if any_substring(normalized, &["show", "check", "what", "current", "list"]) {
        return Some(providers::ChatToolCall {
            id: Some("fallback-config-show".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({ "args": ["config", "show"] }),
        });
    }
    if any_substring(normalized, &["default engine", "set engine", "use engine"]) {
        for engine in ["lemonade", "vllm"] {
            if normalized.contains(engine) {
                return Some(providers::ChatToolCall {
                    id: Some("fallback-config-default-engine".to_owned()),
                    name: "rocm_command".to_owned(),
                    arguments: serde_json::json!({
                        "args": ["config", "set-default-engine", engine],
                        "reason": "Change ROCm CLI's default engine after the user approves it."
                    }),
                });
            }
        }
    }
    if normalized.contains("telemetry") {
        if any_substring(normalized, &["off", "disable", "disabled"]) {
            return Some(providers::ChatToolCall {
                id: Some("fallback-config-telemetry-off".to_owned()),
                name: "rocm_command".to_owned(),
                arguments: serde_json::json!({
                    "args": ["config", "set-telemetry", "off"],
                    "reason": "Turn ROCm CLI telemetry off after the user approves it."
                }),
            });
        }
        if any_substring(normalized, &["local", "on", "enable", "enabled"]) {
            return Some(providers::ChatToolCall {
                id: Some("fallback-config-telemetry-local".to_owned()),
                name: "rocm_command".to_owned(),
                arguments: serde_json::json!({
                    "args": ["config", "set-telemetry", "local"],
                    "reason": "Enable local-only ROCm inspection after the user approves it."
                }),
            });
        }
    }
    None
}

pub(crate) fn any_substring(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn install_sdk_without_prefix_chat_approval(prompt: &str) -> Option<ChatToolApprovalRequest> {
    install_sdk_chat_approval_for_prompt(prompt).and_then(|approval| {
        chat_cli_arg_value(&approval.args, "--prefix")
            .is_none()
            .then_some(approval)
    })
}

fn install_sdk_chat_approval_for_prompt(prompt: &str) -> Option<ChatToolApprovalRequest> {
    let prompt = latest_user_chat_message(prompt);
    let normalized = prompt.to_ascii_lowercase();
    if !prompt_requests_rocm_install_or_setup(&normalized) {
        return None;
    }
    let mut args = vec![
        "install".to_owned(),
        "sdk".to_owned(),
        "--channel".to_owned(),
        if normalized.contains("nightly") {
            "nightly".to_owned()
        } else {
            "release".to_owned()
        },
        "--format".to_owned(),
        "wheel".to_owned(),
    ];
    if let Some(prefix) = requested_install_prefix_from_prompt(prompt) {
        args.push("--prefix".to_owned());
        args.push(prefix);
    }
    if let Some(build_date) = requested_therock_build_date_from_prompt(&normalized) {
        args.push("--build-date".to_owned());
        args.push(build_date);
    } else if let Some(version) = requested_therock_version_from_prompt(&normalized) {
        args.push("--version".to_owned());
        args.push(version);
    }
    Some(ChatToolApprovalRequest {
        pending_title: "Install ROCm".to_owned(),
        command_title: "Install".to_owned(),
        display_command: Some(format_structured_tool_call("rocm", &args)),
        args,
        explanation: Some(
            "Install ROCm/TheRock after the user chooses the install folder.".to_owned(),
        ),
    })
}

fn render_install_sdk_folder_needed_chat_text(prompt: &str, args: &[String]) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "ROCm install");
    let _ = writeln!(output);
    let _ = writeln!(output, "I can install ROCm/TheRock for you.");
    let _ = writeln!(
        output,
        "First choose the folder where ROCm CLI should put the Python environment."
    );
    let _ = writeln!(
        output,
        "Nothing will download or change until you review and approve the install."
    );
    if let Some(date) = chat_cli_arg_value(args, "--build-date") {
        let _ = writeln!(output);
        let _ = writeln!(output, "Requested build date: {date}");
    }
    if let Some(version) = chat_cli_arg_value(args, "--version") {
        let _ = writeln!(output);
        let _ = writeln!(output, "Requested version: {version}");
    }
    let _ = writeln!(output);
    let _ = writeln!(output, "Your request");
    let _ = writeln!(output, "  {}", prompt.trim());
    output.trim_end().to_owned()
}

fn prompt_requests_rocm_install_or_setup(normalized: &str) -> bool {
    let mentions_rocm_stack = any_substring(
        normalized,
        &[
            "rocm",
            "therock",
            "the rock",
            "amd gpu package",
            "amd gpu ready",
            "local ai",
        ],
    );
    if !mentions_rocm_stack {
        return false;
    }
    prompt_requests_install_action(normalized)
        || any_substring(
            normalized,
            &[
                "need rocm",
                "need therock",
                "need the rock",
                "need to get rocm",
                "need to get therock",
                "get rocm installed",
                "get therock installed",
                "get the rock installed",
                "make rocm work",
                "make therock work",
                "make the rock work",
                "prepare rocm",
                "prepare therock",
                "prepare the rock",
                "set up my amd gpu",
                "setup my amd gpu",
                "set up local ai",
                "setup local ai",
                "get local ai working",
                "make local ai work",
                "rocm please",
                "make my amd gpu ready",
                "make my gpu ready",
                "get my amd gpu ready",
                "get my gpu ready",
            ],
        )
}

fn prompt_requests_install_action(normalized: &str) -> bool {
    any_substring(
        normalized,
        &[
            "install this",
            "install specific",
            "install the",
            "install rocm",
            "install therock",
            "install the rock",
            "install amd gpu package",
            "install local ai",
            "can you install",
            "please install",
            "rocm please",
            "therock please",
            "the rock please",
            "need to install",
            "want to install",
            "i need rocm",
            "i need therock",
            "i need the rock",
            "get me rocm",
            "get me therock",
            "get my gpu ready",
            "make my gpu ready",
            "make my amd gpu ready",
            "get installed",
            "setup for me",
            "set up for me",
            "setup rocm",
            "set up rocm",
            "setup therock",
            "set up therock",
            "setup the rock",
            "set up the rock",
            "setup my gpu",
            "set up my gpu",
            "setup amd gpu",
            "set up amd gpu",
        ],
    )
}

pub(crate) fn requested_install_prefix_from_prompt(prompt: &str) -> Option<String> {
    let lower = prompt.to_ascii_lowercase();
    for phrase in [
        "--prefix=",
        "--prefix ",
        " into folder ",
        " into ",
        " in folder ",
        " in ",
        " to folder ",
        " to ",
        " under ",
        " at ",
        " use folder ",
        " use ",
        "use ",
        " folder is ",
        " folder: ",
        "folder:",
    ] {
        let mut search_start = 0;
        while let Some(relative_index) = lower[search_start..].find(phrase) {
            let value_start = search_start + relative_index + phrase.len();
            if let Some(prefix) = clean_requested_install_prefix(&prompt[value_start..]) {
                return Some(prefix);
            }
            search_start = value_start;
        }
    }
    requested_bare_install_prefix_from_prompt(prompt)
}

fn requested_bare_install_prefix_from_prompt(prompt: &str) -> Option<String> {
    for quote in ['"', '\''] {
        let mut remaining = prompt;
        while let Some(start) = remaining.find(quote) {
            let after_start = &remaining[start + quote.len_utf8()..];
            let Some(end) = after_start.find(quote) else {
                break;
            };
            if let Some(prefix) = clean_requested_install_prefix(&after_start[..end]) {
                return Some(prefix);
            }
            remaining = &after_start[end + quote.len_utf8()..];
        }
    }
    prompt.split_whitespace().find_map(|token| {
        clean_requested_install_prefix(
            token.trim_matches(|ch: char| {
                matches!(ch, '"' | '\'' | '`' | ',' | ';' | '.' | ')' | ']')
            }),
        )
    })
}

fn clean_requested_install_prefix(candidate: &str) -> Option<String> {
    let trimmed = candidate.trim_start();
    if trimmed.is_empty() {
        return None;
    }
    let mut chars = trimmed.chars();
    let first = chars.next()?;
    let extracted = if first == '"' || first == '\'' {
        let close = trimmed[first.len_utf8()..].find(first)?;
        &trimmed[first.len_utf8()..first.len_utf8().saturating_add(close)]
    } else {
        trimmed
            .split(['\r', '\n', ',', ';'])
            .next()
            .unwrap_or_default()
            .split(" and ")
            .next()
            .unwrap_or_default()
            .split(" with ")
            .next()
            .unwrap_or_default()
            .split(" then ")
            .next()
            .unwrap_or_default()
            .trim_end_matches(['.', ')', ']'])
    };
    let prefix = extracted.trim();
    if prefix.is_empty() || !looks_like_user_path(prefix) {
        return None;
    }
    Some(prefix.to_owned())
}

fn looks_like_user_path(value: &str) -> bool {
    value.starts_with('/')
        || value.starts_with('~')
        || value.starts_with("\\\\")
        || value.contains(":\\")
        || value.contains(":/")
        || value.contains('\\')
}

pub(crate) fn requested_therock_build_date_from_prompt(normalized: &str) -> Option<String> {
    normalized
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|token| token.len() == 8 && token.chars().all(|ch| ch.is_ascii_digit()))
        .find_map(
            |token| match therock::RuntimeVersionSelector::build_date(token).ok()? {
                therock::RuntimeVersionSelector::BuildDate(date) => Some(date),
                therock::RuntimeVersionSelector::Version(_) => None,
            },
        )
        .or_else(|| {
            normalized
                .split_whitespace()
                .map(|token| {
                    token.trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '-')
                })
                .find_map(|token| {
                    match therock::RuntimeVersionSelector::build_date(token).ok()? {
                        therock::RuntimeVersionSelector::BuildDate(date) => Some(date),
                        therock::RuntimeVersionSelector::Version(_) => None,
                    }
                })
        })
}

fn requested_therock_version_from_prompt(normalized: &str) -> Option<String> {
    normalized
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || matches!(ch, '.' | '+' | '-' | '_')))
        .filter(|token| looks_like_therock_runtime_version(token))
        .find_map(|token| {
            therock::RuntimeVersionSelector::version(token)
                .ok()
                .map(|_| token.to_owned())
        })
}

fn looks_like_therock_runtime_version(token: &str) -> bool {
    let Some((prefix, date)) = token.rsplit_once('a') else {
        return false;
    };
    !prefix.is_empty()
        && prefix.chars().next().is_some_and(|ch| ch.is_ascii_digit())
        && prefix.contains('.')
        && date.len() == 8
        && date.chars().all(|ch| ch.is_ascii_digit())
        && therock::RuntimeVersionSelector::build_date(date).is_ok()
}

fn visible_chat_content(content: &str) -> String {
    let mut output = String::new();
    let mut remaining = content;
    while let Some(start) = find_ascii_case_insensitive(remaining, "<think>") {
        output.push_str(&remaining[..start]);
        let after_start = &remaining[start + "<think>".len()..];
        let Some(end) = find_ascii_case_insensitive(after_start, "</think>") else {
            remaining = "";
            break;
        };
        remaining = &after_start[end + "</think>".len()..];
    }
    output.push_str(remaining);
    output.trim().to_owned()
}

fn find_ascii_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    let needle = needle.as_bytes();
    haystack
        .as_bytes()
        .windows(needle.len())
        .position(|window| window.eq_ignore_ascii_case(needle))
}

/// The host-independent half of the assistant prompt.
///
/// Statements that are only true on *some* hosts do not belong here — they used
/// to, and a WSL user was told "on native Windows, vLLM is skipped" while vLLM
/// was in fact their supported path. Anything host-dependent now comes from
/// [`chat_host_facts::HostFacts`], which knows which machine it is describing.
const ROCM_CHAT_TOOL_SYSTEM_PROMPT: &str = "You are ROCm CLI's local assistant. Speak in simple English for non-technical users. Use the provided ROCm tools when you need to inspect this machine, preview setup, read service logs, check updates, inspect automations, install or start ROCm-managed apps, or request ROCm/TheRock, config, engine, app, and local model server changes. For simple greetings or thanks like hello, hi, hey, ok, or thank you, reply normally; do not inspect ROCm, do not call tools, and do not launch or propose a model server. Tool-use rules: inspect first with read-only tools; call rocm_command only with argv-style args and no shell text; use natural_language_plan for ROCm requests that do not fit another read-only tool; ask for a mutating tool call only after explaining why it is needed; summarize tool results after they are returned. Read-only tools may run immediately. Tools that install, launch, stop, delete, or change state require user approval; request rocm_command and explain why. For 'is X running?', 'what is running?', status, or port questions, inspect before answering and do not start, stop, install, or serve anything. For ComfyUI or port 8188 use [\"comfyui\",\"status\"] or port_status. For vLLM, Lemonade, qwen, or local model servers use [\"services\",\"list\",\"--all\"] for running state and [\"engines\",\"list\"] for installed/available engine state. Treat ready/running as running, starting/recovering as starting, failed/stopped as not running, and no matching record as unknown or not managed by ROCm CLI. Interpret Examine carefully: active_runtime_status=ready means ROCm CLI has an active managed TheRock/ROCm runtime; legacy_rocm_status=not_detected only means no global system ROCm install was found. If active_runtime_status=ready, tell the user ROCm/TheRock is installed and active for ROCm CLI. For 'is TheRock installed', 'is ROCm installed', or 'which GPU is on this machine', use examine or gpu_snapshot before answering. For 'how do I setup TheRock' or install/setup requests, guide the user to choose an install folder first; do not answer with only a status check. For 'which LLMs can this machine support', use rocm_command args [\"model\"] or natural_language_plan before answering. For TheRock installs, always let the user choose the install folder. If the user names a folder or prefix, preserve that exact folder with [\"--prefix\",\"PATH\"]; you may call path_exists first to check whether that user-provided folder or its parent exists. If the user asks you to install TheRock/ROCm but has not named a folder, ask for the folder or let the guided setup folder picker collect it; do not invent a hidden default folder and do not request an install command without --prefix. Use rocm_command args [\"install\",\"sdk\",\"--channel\",\"release\",\"--format\",\"wheel\",\"--prefix\",\"PATH\"] only when the user asks you to install it and a folder is known; for a requested build date add [\"--build-date\",\"YYYY-MM-DD\"] and for a requested exact version add [\"--version\",\"VERSION\"]. For config changes, inspect with [\"config\",\"show\"] first when useful, then request config subcommands such as [\"config\",\"set-default-engine\",\"lemonade\"], [\"config\",\"set-default-runtime\",\"RUNTIME_KEY\"], or [\"config\",\"set-telemetry\",\"local\"] only after explaining why. For ComfyUI, use rocm_command with args like [\"comfyui\",\"status\"], [\"comfyui\",\"logs\"], [\"comfyui\",\"install\"], [\"comfyui\",\"start\"], or [\"comfyui\",\"stop\"]. First-time setup is the same thing as bootstrap in ROCm CLI; it is a deterministic ROCm setup flow, not a separate model chat. The built-in local assistant is fixed to qwen, which maps to Qwen3-4B-Instruct-2507-GGUF served by Lemonade with gpu_required. vLLM and Lemonade are the general serving engines; inspect or manage them when the user asks about general model serving, but do not switch the built-in assistant away from Lemonade. Use qwen-smoke only for a quick server smoke test. For vLLM management, inspect engines first and use [\"engines\",\"install\",\"vllm\"] or [\"serve\",\"MODEL\",\"--engine\",\"vllm\",\"--device\",\"gpu_required\",\"--managed\"] only where the host supports it. Do not invent shell commands and do not request CPU fallback.";
const ROCM_CHAT_TOOL_SKILL: &str = include_str!("../../../skills/rocm-cli-assistant/SKILL.md");

fn rocm_chat_tool_system_prompt() -> String {
    format!("{ROCM_CHAT_TOOL_SYSTEM_PROMPT}\n\nROCm CLI assistant skill:\n{ROCM_CHAT_TOOL_SKILL}")
}

/// The assistant prompt grounded in the machine it will answer for.
///
/// The single composition point: both chat surfaces (`rocm chat --prompt
/// --tools` in this bin, and the dashboard chat via
/// [`crate::dash::resolved_args`]) send this, so neither can drift into
/// answering platform questions from pretraining alone.
pub(crate) fn rocm_chat_tool_system_prompt_for_host(paths: Option<&AppPaths>) -> String {
    let facts = chat_host_facts::HostFacts::detect(paths);
    format!("{}\n\n{}", rocm_chat_tool_system_prompt(), facts.render())
}

fn local_provider_missing_service_error(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        cause
            .to_string()
            .contains("local provider has no ready managed service")
    })
}

fn local_rocm_tools_assistant_model(provider: &str, rocm_tools: bool) -> Option<&'static str> {
    (provider == "local" && rocm_tools).then_some(providers::BUILTIN_ASSISTANT_MODEL_ID)
}

fn local_chat_service_needed_text(model: Option<&str>, prompt: &str, rocm_tools: bool) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "No local assistant is running yet.");
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "To use LLM-assisted ROCm commands, start a ROCm GPU local model server first."
    );
    let _ = writeln!(
        output,
        "First-time ROCm setup does not need an LLM; run `rocm` and use Set Up ROCm for that."
    );
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "Recommended path:\n  run `rocm`, choose Start a local model, use the recommended assistant model, then start it"
    );
    let _ = writeln!(output);
    if let Some(model) = model.filter(|value| !value.trim().is_empty()) {
        let args = vec![
            "serve".to_owned(),
            model.to_owned(),
            "--device".to_owned(),
            "gpu_required".to_owned(),
            "--managed".to_owned(),
        ];
        let _ = writeln!(
            output,
            "Advanced manual command for the selected model:\n  {}",
            format_structured_tool_call("rocm", &args)
        );
    } else {
        let example_args = vec![
            "serve".to_owned(),
            providers::LEMONADE_ASSISTANT_MODEL_ID.to_owned(),
            "--engine".to_owned(),
            "lemonade".to_owned(),
            "--device".to_owned(),
            "gpu_required".to_owned(),
            "--managed".to_owned(),
        ];
        let _ = writeln!(
            output,
            "Advanced manual command for the recommended assistant model:\n  {}",
            format_structured_tool_call("rocm", &example_args)
        );
    }
    let mut chat_args = vec!["chat".to_owned()];
    if rocm_tools {
        chat_args.push("--tools".to_owned());
    }
    chat_args.push("--provider".to_owned());
    chat_args.push("local".to_owned());
    if let Some(model) = model.filter(|value| !value.trim().is_empty()) {
        chat_args.push("--model".to_owned());
        chat_args.push(model.to_owned());
    }
    if !prompt.trim().is_empty() {
        chat_args.push("--prompt".to_owned());
        chat_args.push(prompt.to_owned());
    }
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "After the server is ready, run:\n  {}",
        format_structured_tool_call("rocm", &chat_args)
    );
    let _ = writeln!(output);
    let _ = writeln!(
        output,
        "For the guided flow, run `rocm`, choose Start a local model, start a model, then choose Chat."
    );
    let _ = writeln!(output, "Nothing was changed.");
    output.trim_end().to_owned()
}

fn append_chat_tool_results(
    paths: &AppPaths,
    response: &providers::ChatResponse,
    output: &mut String,
    assistant_explanation: Option<&str>,
    progress: &mut Option<&mut dyn FnMut(String)>,
) -> Result<ChatToolRunResult> {
    if response.tool_calls.is_empty() {
        let _ = writeln!(output, "ROCm checks used");
        let _ = writeln!(output, "  none requested");
        return Ok(ChatToolRunResult {
            approval: None,
            follow_up_text: String::new(),
            ran_read_only_tool: false,
            read_only_tool_error: false,
            needs_install_folder: false,
        });
    }

    let _ = writeln!(output, "ROCm checks used");
    let mut approval = None;
    let mut follow_up_text = String::new();
    let mut ran_read_only_tool = false;
    let mut read_only_tool_error = false;
    let mut needs_install_folder = false;
    for call in &response.tool_calls {
        if chat_tool_call_requests_therock_install_without_prefix(call) {
            needs_install_folder = true;
            report_chat_tool_progress(progress, "Waiting for the ROCm install folder.");
            let _ = writeln!(output, "  Install ROCm: needs install folder");
            let _ = writeln!(
                output,
                "    not run: choose an install folder before the review card"
            );
            continue;
        }
        validate_chat_tool_call(call)?;
        let label = chat_tool_call_display_label(call);
        if chat_tool_call_is_read_only(call) {
            report_chat_tool_progress(progress, &format!("Running ROCm check: {label}."));
            let result = run_chat_read_only_tool(paths, call)?;
            ran_read_only_tool = true;
            let is_error = mcp_tool_result_is_error(&result);
            read_only_tool_error |= is_error;
            report_chat_tool_progress(
                progress,
                &format!(
                    "ROCm check finished: {label} ({})",
                    chat_read_only_tool_status_label(is_error)
                ),
            );
            let _ = writeln!(
                output,
                "  {}: {}",
                label,
                chat_read_only_tool_status_label(is_error)
            );
            let result_text = mcp_tool_result_text(&result);
            let _ = writeln!(follow_up_text, "{}:", call.name);
            let _ = writeln!(follow_up_text, "{result_text}");
            for line in result_text.lines() {
                let _ = writeln!(output, "    {line}");
            }
        } else {
            report_chat_tool_progress(progress, &format!("Review needed: {label}."));
            let _ = writeln!(output, "  {label}: needs your review");
            let _ = writeln!(
                output,
                "    not run: review the approval card before anything runs"
            );
            if let Some(command) = rocm_chat_tool_requested_command(call) {
                let _ = writeln!(output, "    advanced manual command: {command}");
            }
            if approval.is_none() {
                approval = Some(chat_tool_approval_request(call, assistant_explanation)?);
            }
        }
    }
    Ok(ChatToolRunResult {
        approval,
        follow_up_text,
        ran_read_only_tool,
        read_only_tool_error,
        needs_install_folder,
    })
}

fn chat_tool_call_requests_therock_install_without_prefix(call: &providers::ChatToolCall) -> bool {
    let Some(object) = call.arguments.as_object() else {
        return false;
    };
    match call.name.as_str() {
        "install_sdk" => json_string(object, "prefix").is_none(),
        "rocm_command" => rocm_command_args_install_sdk_without_prefix(object),
        _ => false,
    }
}

fn rocm_command_args_install_sdk_without_prefix(
    object: &serde_json::Map<String, serde_json::Value>,
) -> bool {
    let Some(args) = object.get("args").and_then(serde_json::Value::as_array) else {
        return false;
    };
    let args = args
        .iter()
        .filter_map(serde_json::Value::as_str)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    args.first()
        .is_some_and(|arg| arg.eq_ignore_ascii_case("install"))
        && args
            .get(1)
            .is_some_and(|arg| arg.eq_ignore_ascii_case("sdk"))
        && chat_cli_arg_value(&args, "--prefix").is_none()
}

pub(crate) fn chat_tool_approval_request(
    call: &providers::ChatToolCall,
    assistant_explanation: Option<&str>,
) -> Result<ChatToolApprovalRequest> {
    validate_chat_tool_call(call)?;
    if call.name == "rocm_command" {
        let ChatRocmCommandAction::Approval {
            args,
            pending_title,
            command_title,
        } = chat_rocm_command_action(call)?
        else {
            bail!("ROCm command tool is read-only and does not need approval");
        };
        // `pending_title` is deliberately short (e.g. "Install driver"); don't
        // read that as all the approver sees. The full `rocm` argv rides
        // along via `display_command` below and is what the approval body
        // actually renders (see `dash_seam.rs`'s `RocmToolExecutor::execute`).
        return Ok(ChatToolApprovalRequest {
            pending_title,
            command_title,
            display_command: Some(format_structured_tool_call("rocm", &args)),
            args,
            explanation: assistant_explanation
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned),
        });
    }
    // `proposal_action` (approve/reject) executes in-process in
    // `run_internal_mcp_call` — it has NO CLI argv, so build its approval
    // request directly rather than through `rocm_chat_tool_requested_args`.
    if call.name == "proposal_action" {
        let object = call
            .arguments
            .as_object()
            .context("proposal_action arguments must be a JSON object")?;
        let proposal_id = json_string(object, "proposal_id")
            .context("proposal_action requires non-empty `proposal_id`")?;
        let action =
            json_string(object, "action").context("proposal_action requires non-empty `action`")?;
        let pending_title = match action.as_str() {
            "approve" => "Approve proposal",
            "reject" => "Reject proposal",
            other => bail!("proposal_action `{other}` does not require approval"),
        };
        return Ok(ChatToolApprovalRequest {
            pending_title: pending_title.to_owned(),
            command_title: "Reviews".to_owned(),
            display_command: Some(format!("proposal {proposal_id}")),
            args: Vec::new(),
            explanation: assistant_explanation
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned),
        });
    }
    let args = rocm_chat_tool_requested_args(call)
        .with_context(|| format!("ROCm tool `{}` is missing required arguments", call.name))?;
    let (pending_title, command_title) = match call.name.as_str() {
        "install_sdk" => ("Install ROCm", "Install"),
        "install_engine" => ("Install engine", "Engine"),
        "launch_server" => ("Start local model server", "Serve"),
        "stop_server" => ("Stop local model server", "Services"),
        "watcher_enable" => ("Enable automation", "Automations"),
        "watcher_disable" => ("Disable automation", "Automations"),
        other => bail!("ROCm tool `{other}` is read-only or unsupported for approval"),
    };
    // Same as the `rocm_command` arm above: `pending_title` is short by
    // design — the full command line rides along in `display_command` and
    // is what the approval body renders, not just this title.
    Ok(ChatToolApprovalRequest {
        pending_title: pending_title.to_owned(),
        command_title: command_title.to_owned(),
        display_command: rocm_chat_tool_requested_command(call),
        args,
        explanation: assistant_explanation
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned),
    })
}

pub(crate) fn validate_chat_tool_call(call: &providers::ChatToolCall) -> Result<()> {
    if !call.arguments.is_object() {
        bail!("ROCm tool `{}` arguments must be a JSON object", call.name);
    }
    match call.name.as_str() {
        "examine"
        // `doctor` is the dash-side LLM tool + `/doctor` overlay name for the
        // same machine inspection the bin exposes as `examine`. Accept it as an
        // alias so a model-issued `doctor` call resolves end-to-end.
        | "doctor"
        | "bridge_snapshot"
        | "gpu_snapshot"
        | "engines"
        | "services"
        | "service_logs"
        | "automations"
        | "natural_language_plan"
        | "path_exists"
        | "port_status"
        | "rocm_command"
        | "update_check"
        | "install_sdk_dry_run"
        | "install_sdk"
        | "install_engine"
        | "launch_server"
        | "stop_server"
        | "watcher_enable"
        | "watcher_disable"
        | "proposal_action" => {}
        other => bail!("local assistant requested unsupported ROCm tool `{other}`"),
    }
    match call.name.as_str() {
        "install_sdk" => validate_chat_install_sdk_tool_call(call)?,
        "install_engine" => validate_required_chat_string(call, "engine")?,
        "launch_server" => validate_chat_launch_server_tool_call(call)?,
        "service_logs" | "stop_server" => validate_chat_service_tool_call(call)?,
        "automations" => validate_optional_chat_integer(call, "event_limit", 1, 64)?,
        "natural_language_plan" => validate_required_chat_string(call, "request")?,
        "path_exists" => validate_required_chat_string(call, "path")?,
        "port_status" => validate_chat_port_status_tool_call(call)?,
        "rocm_command" => validate_chat_rocm_command_tool_call(call)?,
        "watcher_enable" => validate_chat_watcher_tool_call(call, true)?,
        "watcher_disable" => validate_chat_watcher_tool_call(call, false)?,
        "proposal_action" => validate_chat_proposal_action_tool_call(call)?,
        _ => {}
    }
    Ok(())
}

/// Validate a `proposal_action` chat-tool call: `proposal_id` must be a
/// non-empty string and `action` must be one of show | approve | reject.
fn validate_chat_proposal_action_tool_call(call: &providers::ChatToolCall) -> Result<()> {
    let object = call
        .arguments
        .as_object()
        .context("proposal_action arguments must be a JSON object")?;
    let proposal_id = json_string(object, "proposal_id")
        .context("proposal_action requires non-empty `proposal_id`")?;
    if proposal_id.len() > 128 {
        bail!("proposal_id too long");
    }
    let action =
        json_string(object, "action").context("proposal_action requires non-empty `action`")?;
    if !matches!(action.as_str(), "show" | "approve" | "reject") {
        bail!("proposal_action `action` must be one of show, approve, reject");
    }
    Ok(())
}

fn validate_chat_install_sdk_tool_call(call: &providers::ChatToolCall) -> Result<()> {
    let object = call
        .arguments
        .as_object()
        .context("install_sdk arguments must be a JSON object")?;
    let channel = json_string(object, "channel").unwrap_or_else(|| "release".to_owned());
    if !matches!(channel.as_str(), "release" | "nightly") {
        bail!("local assistant requested unsupported TheRock channel `{channel}`");
    }
    let format = json_string(object, "format").unwrap_or_else(|| "wheel".to_owned());
    if !matches!(format.as_str(), "wheel" | "tarball") {
        bail!("local assistant requested unsupported TheRock install format `{format}`");
    }
    if rocm_core::runtime_is_windows() && format != "wheel" {
        bail!("local assistant cannot request `{format}` installs on Windows; use wheel");
    }
    let version = json_string(object, "version");
    let build_date = json_string(object, "build_date");
    if version.is_some() && build_date.is_some() {
        bail!("local assistant cannot request both `version` and `build_date`");
    }
    if format != "wheel" && (version.is_some() || build_date.is_some()) {
        bail!(
            "local assistant can only request specific TheRock wheel versions for wheel installs"
        );
    }
    if let Some(version) = version {
        therock::RuntimeVersionSelector::version(version)?;
    }
    if let Some(build_date) = build_date {
        therock::RuntimeVersionSelector::build_date(build_date)?;
    }
    let Some(prefix) = json_string(object, "prefix") else {
        bail!(
            "local assistant must ask the user for a ROCm/TheRock install folder before requesting install_sdk"
        );
    };
    let prefix_path = Path::new(&prefix);
    if chat_install_prefix_is_system(prefix_path) {
        bail!(
            "local assistant cannot request system install folder `{}`",
            prefix_path.display()
        );
    }
    Ok(())
}

fn validate_chat_launch_server_tool_call(call: &providers::ChatToolCall) -> Result<()> {
    let object = call
        .arguments
        .as_object()
        .context("launch_server arguments must be a JSON object")?;
    if object
        .get("allow_public_bind")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        bail!("local assistant cannot request public network binding");
    }
    let host = json_string(object, "host").unwrap_or_else(|| DEFAULT_LOCAL_HOST.to_owned());
    if !is_loopback_host(&host) {
        bail!("local assistant cannot request non-local host `{host}`");
    }
    if object
        .get("device")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|device| device.to_ascii_lowercase().contains("cpu"))
    {
        bail!("local assistant cannot request CPU execution; ROCm GPU execution is required");
    }
    Ok(())
}

fn validate_chat_service_tool_call(call: &providers::ChatToolCall) -> Result<()> {
    let object = call
        .arguments
        .as_object()
        .context("service tool arguments must be a JSON object")?;
    let service_id =
        json_string(object, "service_id").context("service tool requires `service_id`")?;
    validate_service_id(&service_id)?;
    if call.name == "service_logs" {
        validate_optional_chat_integer(call, "lines", 1, 500)?;
    }
    Ok(())
}

fn validate_chat_watcher_tool_call(call: &providers::ChatToolCall, allow_mode: bool) -> Result<()> {
    let object = call
        .arguments
        .as_object()
        .context("watcher tool arguments must be a JSON object")?;
    let watcher = json_string(object, "watcher").context("watcher tool requires `watcher`")?;
    if watcher.len() > 128 {
        bail!("watcher id too long");
    }
    if builtin_watcher(&watcher).is_none() {
        bail!("local assistant requested unknown watcher `{watcher}`");
    }
    if let Some(mode) = json_string(object, "mode") {
        if !allow_mode {
            bail!("local assistant cannot set `mode` when disabling a watcher");
        }
        if !matches!(mode.as_str(), "observe" | "propose" | "contained") {
            bail!("local assistant requested unsupported watcher mode `{mode}`");
        }
    }
    Ok(())
}

fn validate_chat_port_status_tool_call(call: &providers::ChatToolCall) -> Result<()> {
    let object = call
        .arguments
        .as_object()
        .context("port_status arguments must be a JSON object")?;
    let host = json_string(object, "host").unwrap_or_else(|| DEFAULT_LOCAL_HOST.to_owned());
    if !is_loopback_host(&host) {
        bail!("local assistant cannot inspect non-local host `{host}`");
    }
    let Some(port) = object.get("port").and_then(serde_json::Value::as_u64) else {
        bail!("ROCm tool `port_status` requires integer `port`");
    };
    if !(1..=u64::from(u16::MAX)).contains(&port) {
        bail!("ROCm tool `port_status` argument `port` must be between 1 and 65535");
    }
    Ok(())
}

fn validate_required_chat_string(call: &providers::ChatToolCall, key: &str) -> Result<()> {
    let object = call
        .arguments
        .as_object()
        .context("tool arguments must be a JSON object")?;
    json_string(object, key)
        .with_context(|| format!("ROCm tool `{}` requires non-empty `{key}`", call.name))?;
    Ok(())
}

fn validate_optional_chat_integer(
    call: &providers::ChatToolCall,
    key: &str,
    min: u64,
    max: u64,
) -> Result<()> {
    let object = call
        .arguments
        .as_object()
        .context("tool arguments must be a JSON object")?;
    let Some(value) = object.get(key) else {
        return Ok(());
    };
    let Some(value) = value.as_u64() else {
        bail!(
            "ROCm tool `{}` argument `{key}` must be an integer",
            call.name
        );
    };
    if value < min || value > max {
        bail!(
            "ROCm tool `{}` argument `{key}` must be between {min} and {max}",
            call.name
        );
    }
    Ok(())
}

fn validate_chat_rocm_command_tool_call(call: &providers::ChatToolCall) -> Result<()> {
    chat_rocm_command_action(call).map(|_| ())
}

fn chat_rocm_command_action(call: &providers::ChatToolCall) -> Result<ChatRocmCommandAction> {
    let args = normalized_chat_rocm_command_args(call)?;
    chat_rocm_command_action_from_args(args)
}

fn normalized_chat_rocm_command_args(call: &providers::ChatToolCall) -> Result<Vec<String>> {
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

fn managed_service_running_state(status: &str) -> &'static str {
    match status {
        "ready" | "running" => "running",
        "starting" | "recovering" => "starting",
        "failed" | "stopped" => "not_running",
        _ => "unknown",
    }
}

fn render_services_tool_result_text(records: &[ManagedServiceRecord]) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "managed_services: {}", records.len());
    let _ = writeln!(
        output,
        "status_meaning: ready/running = running; starting/recovering = starting; failed/stopped = not running; no matching row = not managed by ROCm CLI"
    );
    if records.is_empty() {
        let _ = writeln!(output, "services: none");
        return output;
    }
    let _ = writeln!(output, "services:");
    for record in records {
        let _ = writeln!(
            output,
            "  - service_id={} engine={} model={} canonical_model={} status={} running_state={} endpoint={}",
            record.service_id,
            record.engine,
            record.model_ref,
            record.canonical_model_id,
            record.status,
            managed_service_running_state(&record.status),
            record.endpoint_url
        );
    }
    output
}

fn chat_rocm_command_action_from_args(mut args: Vec<String>) -> Result<ChatRocmCommandAction> {
    canonicalize_chat_rocm_command(&mut args)?;
    validate_chat_rocm_command_safety(&args)?;
    let first = args.first().map(|value| value.to_ascii_lowercase());
    let second = args.get(1).map(|value| value.to_ascii_lowercase());
    match first.as_deref() {
        Some("examine" | "version" | "model" | "models" | "daemon" | "logs") => {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        Some("update") if !args.iter().any(|arg| arg == "--apply") => {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        Some("runtimes") if second.as_deref().is_none_or(|value| value == "list") => {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        Some("engines") if second.as_deref().is_some_and(|value| value == "list") => {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        Some("services")
            if second
                .as_deref()
                .is_none_or(|value| matches!(value, "list" | "logs")) =>
        {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        Some("automations") if second.as_deref().is_none_or(|value| value == "list") => {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        Some("config") if second.as_deref() == Some("show") => {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        Some("comfyui")
            if second
                .as_deref()
                .is_none_or(|value| matches!(value, "status" | "logs" | "log")) =>
        {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        Some("install") if second.as_deref() == Some("sdk") => {
            // The chat/MCP surfaces spawn `rocm` with null stdin, so
            // `interactive_terminal()` is false and the consent prompt would
            // refuse with a "re-run with `--approve-replacing-active-default`"
            // error, which the schema-constrained `install_sdk` tool gives the
            // user no way to answer.
            //
            // Not `--yes`: that flag also approves system-package installs, and
            // this spawn has no terminal on which to answer the `sudo` password
            // prompt such an install can raise. Granting it here would make the
            // vLLM/OpenMPI step run a sudo it cannot complete and abort the
            // engine auto-install that previously warned and continued. The
            // narrow flag grants exactly the consent the prompt is asking for.
            //
            // `--yes` is stripped rather than merely not added, because the
            // generic `rocm_command` tool takes a model-supplied argv: a
            // model-emitted `--yes` would otherwise reach this null-stdin spawn
            // and re-grant the system-package consent `76c6aa3c` removed. The
            // strip runs before the argv is rendered for human approval, so what
            // is shown is still what runs.
            //
            // The `--yes=...` form is stripped too. Clap rejects an attached
            // value on this flag today, so an exact-match strip happens to be
            // airtight — but only by borrowing a property of clap's error
            // taxonomy that nothing here owns. Giving `--yes` `num_args`, or a
            // clap release that starts accepting `--yes=true` on a bare `bool`,
            // would silently restore the sudo consent this strip exists to
            // remove. Matching the prefix keeps the guarantee local to this
            // function.
            args.retain(|arg| arg != "--yes" && !arg.starts_with("--yes="));
            // Withheld on `--dry-run`, which returns before the consent gate and
            // so needs no consent: `rocmd` and dash-tui omit it there for the
            // same reason, and a preview should not be recorded as carrying an
            // approval it never used.
            if !args.iter().any(|arg| arg == "--dry-run") {
                ensure_flag(&mut args, "--approve-replacing-active-default");
            }
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Install ROCm".to_owned(),
                command_title: "Install".to_owned(),
            })
        }
        Some("install") if second.as_deref() == Some("driver") => {
            ensure_flag(&mut args, "--yes");
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Install driver".to_owned(),
                command_title: "Install".to_owned(),
            })
        }
        Some("update") if args.iter().any(|arg| arg == "--apply") => {
            ensure_flag(&mut args, "--yes");
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Apply ROCm update".to_owned(),
                command_title: "Update".to_owned(),
            })
        }
        Some("runtimes")
            if second
                .as_deref()
                .is_some_and(|value| value == "uninstall" || value == "remove")
                && args.iter().any(|arg| arg == "--dry-run") =>
        {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        Some("runtimes")
            if second
                .as_deref()
                .is_some_and(|value| value == "uninstall" || value == "remove") =>
        {
            ensure_flag(&mut args, "--yes");
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Remove ROCm install".to_owned(),
                command_title: "Runtimes".to_owned(),
            })
        }
        Some("runtimes") => Ok(ChatRocmCommandAction::Approval {
            args,
            pending_title: "Change ROCm runtime".to_owned(),
            command_title: "Runtimes".to_owned(),
        }),
        Some("engines") if second.as_deref() == Some("install") => {
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Install engine".to_owned(),
                command_title: "Engine".to_owned(),
            })
        }
        Some("serve") => Ok(ChatRocmCommandAction::Approval {
            args,
            pending_title: "Start local model server".to_owned(),
            command_title: "Serve".to_owned(),
        }),
        Some("services")
            if second
                .as_deref()
                .is_some_and(|value| matches!(value, "stop" | "restart")) =>
        {
            ensure_flag(&mut args, "--yes");
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Change local model server".to_owned(),
                command_title: "Services".to_owned(),
            })
        }
        Some("automations")
            if second
                .as_deref()
                .is_some_and(|value| matches!(value, "enable" | "disable")) =>
        {
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Change automation".to_owned(),
                command_title: "Automations".to_owned(),
            })
        }
        Some("config") => Ok(ChatRocmCommandAction::Approval {
            args,
            pending_title: "Change settings".to_owned(),
            command_title: "Config".to_owned(),
        }),
        Some("uninstall") if args.iter().any(|arg| arg == "--dry-run") => {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        Some("uninstall") => {
            ensure_flag(&mut args, "--yes");
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Uninstall ROCm CLI".to_owned(),
                command_title: "Uninstall".to_owned(),
            })
        }
        // `storage report` (the default subcommand) only measures folders, so
        // the assistant can run it directly. The two `remove-*` verbs delete
        // multi-gigabyte trees and go through the approval UI.
        Some("storage") if second.as_deref().is_none_or(|value| value == "report") => {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        Some("storage") => {
            ensure_flag(&mut args, "--yes");
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Free up disk space".to_owned(),
                command_title: "Storage".to_owned(),
            })
        }
        Some("comfyui") if second.as_deref() == Some("install") => {
            ensure_flag(&mut args, "--yes");
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Install ComfyUI".to_owned(),
                command_title: "ComfyUI".to_owned(),
            })
        }
        Some("comfyui") if second.as_deref() == Some("start") => {
            ensure_flag(&mut args, "--yes");
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Start ComfyUI".to_owned(),
                command_title: "ComfyUI".to_owned(),
            })
        }
        Some("comfyui") if second.as_deref() == Some("stop") => {
            ensure_flag(&mut args, "--yes");
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Stop ComfyUI".to_owned(),
                command_title: "ComfyUI".to_owned(),
            })
        }
        Some("setup") if second.as_deref().is_none_or(|value| value == "status") => {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        Some("setup") if second.as_deref() == Some("reset") => {
            ensure_flag(&mut args, "--yes");
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: "Reset first-time setup".to_owned(),
                command_title: "Setup".to_owned(),
            })
        }
        // Mirrors the daemon's `ensure_rocm_command_is_read_only`, which admits
        // the same three verbs. The daemon's arm carries the reasoning; the
        // invariant is that the two agree, and this PR added the arm there
        // first, so the two disagreed until this was written.
        Some("remote")
            if second
                .as_deref()
                .is_some_and(|value| matches!(value, "targets" | "doctor" | "status")) =>
        {
            Ok(ChatRocmCommandAction::ReadOnly(args))
        }
        // `serve`, `attach` and `stop` start, publish or tear down something on
        // another machine, so they go through approval like any other mutation.
        Some("remote")
            if second
                .as_deref()
                .is_some_and(|value| matches!(value, "serve" | "attach" | "stop")) =>
        {
            let verb = second.as_deref().unwrap_or_default().to_owned();
            Ok(ChatRocmCommandAction::Approval {
                args,
                pending_title: format!("Remote {verb}"),
                command_title: "Remote".to_owned(),
            })
        }
        Some(command) => bail!("local assistant cannot use unsupported rocm command `{command}`"),
        None => bail!("rocm_command requires at least one argument"),
    }
}

fn canonicalize_chat_rocm_command(args: &mut [String]) -> Result<()> {
    if args
        .first()
        .is_some_and(|arg| arg.eq_ignore_ascii_case("comfy"))
    {
        args[0] = "comfyui".to_owned();
    }
    if args
        .first()
        .is_some_and(|arg| arg.eq_ignore_ascii_case("engine"))
    {
        bail!("use `engines` for rocm engine commands");
    }
    Ok(())
}

pub(crate) fn validate_chat_rocm_command_safety(args: &[String]) -> Result<()> {
    if let Some(first) = args.first()
        && first.eq_ignore_ascii_case("serve")
    {
        if chat_cli_has_flag(args, "--allow-public-bind") {
            bail!("local assistant cannot request public network binding");
        }
        if serve_args_request_cpu_device(args) {
            bail!("local assistant cannot request CPU execution; ROCm GPU execution is required");
        }
        if let Some(host) = chat_cli_arg_value(args, "--host")
            && !is_loopback_host(host)
        {
            bail!("local assistant cannot request non-local host `{host}`");
        }
        if chat_cli_has_flag(args, "--foreground") || !chat_cli_has_flag(args, "--managed") {
            bail!("local assistant must request managed serving with --managed");
        }
    }
    if args
        .first()
        .is_some_and(|arg| arg.eq_ignore_ascii_case("comfyui"))
        && args
            .get(1)
            .is_some_and(|arg| arg.eq_ignore_ascii_case("start"))
        && let Some(host) = chat_cli_arg_value(args, "--host")
        && !is_loopback_host(host)
    {
        bail!("local assistant cannot start ComfyUI on non-local host `{host}`");
    }
    if args
        .first()
        .is_some_and(|arg| arg.eq_ignore_ascii_case("install"))
        && args
            .get(1)
            .is_some_and(|arg| arg.eq_ignore_ascii_case("sdk"))
    {
        if rocm_core::runtime_is_windows()
            && chat_cli_arg_value(args, "--format")
                .is_some_and(|value| !value.eq_ignore_ascii_case("wheel"))
        {
            bail!("local assistant cannot request non-wheel ROCm installs on Windows");
        }
        let version = chat_cli_arg_value_checked(args, "--version")?;
        let build_date = chat_cli_arg_value_checked(args, "--build-date")?;
        if version.is_some() && build_date.is_some() {
            bail!("local assistant cannot request both --version and --build-date");
        }
        if version.is_some() || build_date.is_some() {
            let format = chat_cli_arg_value(args, "--format").unwrap_or("wheel");
            if !format.eq_ignore_ascii_case("wheel") {
                bail!(
                    "local assistant can only request specific TheRock wheel versions for wheel installs"
                );
            }
        }
        if let Some(version) = version {
            therock::RuntimeVersionSelector::version(version)?;
        }
        if let Some(build_date) = build_date {
            therock::RuntimeVersionSelector::build_date(build_date)?;
        }
        let Some(prefix) = chat_cli_arg_value_checked(args, "--prefix")? else {
            bail!(
                "local assistant must ask the user for a ROCm/TheRock install folder before requesting `rocm install sdk`"
            );
        };
        let prefix_path = Path::new(prefix);
        if chat_install_prefix_is_system(prefix_path) {
            bail!(
                "local assistant cannot request system install folder `{}`",
                prefix_path.display()
            );
        }
    }
    Ok(())
}

pub(crate) fn ensure_flag(args: &mut Vec<String>, flag: &str) {
    if !args.iter().any(|arg| arg == flag) {
        args.push(flag.to_owned());
    }
}

pub(crate) fn chat_cli_arg_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        if let Some((option, value)) = arg.split_once('=')
            && option == name
        {
            return Some(value);
        }
        if arg == name {
            return args.get(index + 1).map(String::as_str);
        }
        index += 1;
    }
    None
}

fn chat_cli_arg_value_checked<'a>(args: &'a [String], name: &str) -> Result<Option<&'a str>> {
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].as_str();
        if let Some((option, value)) = arg.split_once('=')
            && option == name
        {
            if value.trim().is_empty() {
                bail!("local assistant provided empty value for {name}");
            }
            return Ok(Some(value));
        }
        if arg == name {
            let Some(value) = args.get(index + 1).map(String::as_str) else {
                bail!("local assistant omitted the value for {name}");
            };
            if value.starts_with("--") || value.trim().is_empty() {
                bail!("local assistant omitted the value for {name}");
            }
            return Ok(Some(value));
        }
        index += 1;
    }
    Ok(None)
}

pub(crate) fn chat_cli_has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| {
        arg == name
            || arg
                .strip_prefix(name)
                .is_some_and(|rest| rest.starts_with('='))
    })
}

fn chat_install_prefix_is_system(prefix: &Path) -> bool {
    prefix.as_os_str().is_empty() || runtime_install_root_is_protected(prefix)
}

pub(crate) fn chat_tool_call_is_read_only(call: &providers::ChatToolCall) -> bool {
    if call.name == "rocm_command" {
        return matches!(
            chat_rocm_command_action(call),
            Ok(ChatRocmCommandAction::ReadOnly(_))
        );
    }
    // `proposal_action` is read-only only when showing a proposal; approve/reject
    // mutate the proposal status and must route through approval.
    if call.name == "proposal_action" {
        return call
            .arguments
            .as_object()
            .and_then(|object| json_string(object, "action"))
            .as_deref()
            == Some("show");
    }
    matches!(
        call.name.as_str(),
        "examine"
            | "doctor"
            | "bridge_snapshot"
            | "gpu_snapshot"
            | "engines"
            | "services"
            | "service_logs"
            | "automations"
            | "natural_language_plan"
            | "path_exists"
            | "port_status"
            | "update_check"
            | "install_sdk_dry_run"
    )
}

fn deterministic_rocm_tool_summary(tool_text: &str) -> Option<String> {
    if !tool_text.contains("examine:") {
        return None;
    }
    let mut lines = Vec::new();
    let gpu = chat_tool_value(tool_text, "driver_detail")
        .filter(|value| value != "<unknown>")
        .or_else(|| {
            let target = chat_tool_value(tool_text, "detected_gfx_target")?;
            Some(format!("AMD GPU target {target}"))
        });
    if let Some(gpu) = gpu {
        lines.push(format!("  GPU: {gpu}"));
    }

    let active_status = chat_tool_value(tool_text, "active_runtime_status");
    if active_status.as_deref() == Some("ready") {
        let family = chat_tool_value(tool_text, "active_runtime_family")
            .filter(|value| value != "<unset>" && value != "<unknown>");
        let version = chat_tool_value(tool_text, "active_runtime_version")
            .filter(|value| value != "<unset>" && value != "<unknown>");
        let mut detail = "installed and active for ROCm CLI".to_owned();
        if let Some(family) = family {
            let _ = write!(detail, " ({family})");
        }
        if let Some(version) = version {
            let _ = write!(detail, ", {version}");
        }
        lines.push(format!("  ROCm/TheRock: {detail}"));
        if let Some(root) = chat_tool_value(tool_text, "active_runtime_root")
            .or_else(|| chat_tool_value(tool_text, "setup_runtime_root"))
            .filter(|value| value != "<unset>" && value != "<unknown>" && value != "<none>")
        {
            lines.push(format!("  Install folder: {root}"));
        }
        if let Some(cache) = chat_tool_value(tool_text, "active_runtime_pip_cache_dir")
            .or_else(|| chat_tool_value(tool_text, "setup_runtime_pip_cache_dir"))
            .filter(|value| value != "<unset>" && value != "<unknown>" && value != "<none>")
        {
            lines.push(format!("  Downloads/cache: {cache}"));
        }
    } else if let Some(status) = active_status.as_deref() {
        lines.push(format!("  ROCm/TheRock: active runtime status is {status}"));
        if let Some(root) = chat_tool_value(tool_text, "setup_runtime_root")
            .filter(|value| value != "<unset>" && value != "<unknown>" && value != "<none>")
        {
            lines.push(format!("  Selected setup folder: {root}"));
        }
        if let Some(cache) = chat_tool_value(tool_text, "setup_runtime_pip_cache_dir")
            .filter(|value| value != "<unset>" && value != "<unknown>" && value != "<none>")
        {
            lines.push(format!("  Downloads/cache: {cache}"));
        }
    }

    if chat_tool_value(tool_text, "legacy_rocm_status").as_deref() == Some("not_detected")
        && active_status.as_deref() == Some("ready")
    {
        lines.push(
            "  Note: no global legacy ROCm install was found; ROCm CLI is using its managed TheRock runtime."
                .to_owned(),
        );
    }

    (!lines.is_empty()).then(|| lines.join("\n"))
}

fn deterministic_chat_tool_summary(tool_text: &str) -> Option<String> {
    deterministic_rocm_tool_summary(tool_text)
        .or_else(|| deterministic_model_tool_summary(tool_text))
        .or_else(|| deterministic_combined_status_tool_summary(tool_text))
}

fn deterministic_combined_status_tool_summary(tool_text: &str) -> Option<String> {
    let summaries = [
        deterministic_comfyui_tool_summary(tool_text),
        deterministic_port_status_tool_summary(tool_text),
        deterministic_engine_inventory_tool_summary(tool_text),
        deterministic_services_tool_summary(tool_text),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    (!summaries.is_empty()).then(|| summaries.join("\n"))
}

fn deterministic_summary_can_stand_alone(tool_text: &str) -> bool {
    !tool_text.contains("model recipes")
}

fn deterministic_model_tool_summary(tool_text: &str) -> Option<String> {
    if !tool_text.contains("model recipes") {
        return None;
    }
    let recipes = parse_deterministic_model_recipes(tool_text);
    if recipes.is_empty() {
        return None;
    }

    let mut lines = Vec::new();
    if let Some(recipe) = recipes
        .iter()
        .find(|recipe| recipe.canonical_id == providers::LEMONADE_ASSISTANT_MODEL_ID)
    {
        lines.push(format!(
            "  Recommended local assistant: qwen ({})",
            recipe.canonical_id
        ));
        lines.push(format!(
            "    Fit: needs about {} GPU memory.",
            recipe
                .min_gpu_mem_gib
                .map_or_else(|| "unknown".to_owned(), |value| format!("{value} GiB"))
        ));
        lines.push(format!(
            "    Engine: {}",
            deterministic_engine_fit_summary(recipe, "lemonade")
        ));
    }

    if let Some(recipe) = recipes
        .iter()
        .find(|recipe| recipe.canonical_id == "Qwen3-0.6B-GGUF")
    {
        lines.push(format!(
            "  Tiny smoke test: qwen-smoke ({})",
            recipe.canonical_id
        ));
        lines.push(
            "    Use this only to check that GPU serving starts; it is not the default assistant."
                .to_owned(),
        );
    }

    if let Some(recipe) = recipes
        .iter()
        .find(|recipe| recipe.canonical_id == "meta-llama/Llama-3.2-3B-Instruct")
    {
        lines.push(format!(
            "  8 GiB-class option: llama ({})",
            recipe.canonical_id
        ));
        lines.push(format!(
            "    Fit: asks for {}; this can be tight on APUs and depends on available shared GPU memory.",
            recipe
                .min_gpu_mem_gib.map_or_else(|| "unknown GPU memory".to_owned(), |value| format!("{value} GiB"))
        ));
        lines.push(format!(
            "    Engines: {}",
            deterministic_available_engine_names(recipe).join(", ")
        ));
    }

    let larger = recipes
        .iter()
        .filter(|recipe| recipe.min_gpu_mem_gib.is_some_and(|value| value > 8))
        .filter(|recipe| recipe.canonical_id != providers::BUILTIN_ASSISTANT_MODEL_ID)
        .map(|recipe| {
            format!(
                "{} asks for {}",
                recipe.canonical_id,
                recipe.min_gpu_mem_gib.map_or_else(
                    || "more GPU memory".to_owned(),
                    |value| format!("{value} GiB")
                )
            )
        })
        .collect::<Vec<_>>();
    if !larger.is_empty() {
        lines.push(format!(
            "  Larger recipes are not low-VRAM defaults: {}.",
            larger.join("; ")
        ));
    }

    let linux_wsl_only = recipes
        .iter()
        .flat_map(|recipe| {
            recipe
                .engine_statuses
                .iter()
                .filter(|status| {
                    status.contains("unsupported_native_windows")
                        || status.to_ascii_lowercase().contains("wsl/linux")
                        || status.to_ascii_lowercase().contains("linux/wsl")
                })
                .filter_map(|status| {
                    let (engine, _) = status.split_once(':')?;
                    Some(format!(
                        "{} uses {} through WSL/Linux on Windows",
                        recipe.canonical_id, engine
                    ))
                })
        })
        .collect::<Vec<_>>();
    if !linux_wsl_only.is_empty() {
        lines.push(format!(
            "  Native Windows note: {}.",
            linux_wsl_only.join("; ")
        ));
    }

    if !lines.is_empty() {
        lines.push(
            "  Run `rocm examine` to refresh GPU memory details before starting anything large."
                .to_owned(),
        );
    }
    (!lines.is_empty()).then(|| lines.join("\n"))
}

fn deterministic_engine_fit_summary(recipe: &DeterministicModelRecipe, engine: &str) -> String {
    let status = recipe
        .engine_statuses
        .iter()
        .find(|status| status.starts_with(&format!("{engine}:")))
        .map(std::string::String::as_str);
    match status {
        Some(status) if status.contains("unsupported_native_windows") => {
            format!("{engine}: WSL/Linux only on Windows")
        }
        Some(status)
            if status.split_once(':').is_some_and(|(_, rest)| {
                rest.trim_start().starts_with("available")
                    || rest.trim_start().starts_with("adapter_available")
            }) =>
        {
            format!("{engine}: available")
        }
        Some(status) => status.to_owned(),
        None if recipe.engines.iter().any(|candidate| candidate == engine) => {
            format!("{engine}: listed")
        }
        None => format!("{engine}: not listed"),
    }
}

fn deterministic_available_engine_names(recipe: &DeterministicModelRecipe) -> Vec<String> {
    let engines = recipe
        .engine_statuses
        .iter()
        .filter_map(|status| {
            let (engine, rest) = status.split_once(':')?;
            (rest.trim_start().starts_with("available")
                || rest.trim_start().starts_with("adapter_available"))
            .then_some(engine.to_owned())
        })
        .collect::<Vec<_>>();
    if engines.is_empty() {
        recipe.engines.clone()
    } else {
        engines
    }
}

pub(crate) fn deterministic_comfyui_tool_summary(tool_text: &str) -> Option<String> {
    if !tool_text.contains("ComfyUI") || !tool_text.contains("Running") {
        return None;
    }
    let mut lines = Vec::new();
    match chat_tool_value(tool_text, "installed").as_deref() {
        Some("yes") => lines.push("  ComfyUI: installed.".to_owned()),
        Some("no") => lines.push("  ComfyUI: not installed.".to_owned()),
        Some(value) => lines.push(format!("  ComfyUI: installed status is {value}.")),
        None => {}
    }
    if let Some(status) = chat_tool_value(tool_text, "status") {
        let running_text = if status.eq_ignore_ascii_case("running") {
            "running".to_owned()
        } else if status.eq_ignore_ascii_case("starting") {
            "starting".to_owned()
        } else {
            format!("not running ({status})")
        };
        lines.push(format!("  Running: {running_text}."));
    }
    if let Some(models_path) = chat_tool_value(tool_text, "models path") {
        lines.push(format!("  Models folder: {models_path}."));
    }
    // A note about the install itself (an interrupted reinstall, which names
    // the command that finishes it) sits between `status`'s `ComfyUI` heading
    // and its `Running` section, whose own notes the `Running:` line sums up.
    for line in tool_text
        .lines()
        .skip_while(|line| line.trim() != "ComfyUI")
        .take_while(|line| line.trim() != "Running")
    {
        if let Some(note) = line.trim().strip_prefix("note:").map(str::trim)
            && !note.is_empty()
        {
            lines.push(format!("  Note: {note}."));
        }
    }
    (!lines.is_empty()).then(|| lines.join("\n"))
}

fn deterministic_services_tool_summary(tool_text: &str) -> Option<String> {
    if !tool_text.contains("Local Servers")
        && !tool_text.contains("managed_services:")
        && !tool_text.contains("services:")
    {
        return None;
    }
    let rows = parse_deterministic_service_rows(tool_text);
    let mut lines = Vec::new();
    let live = rows
        .iter()
        .filter(|row| matches!(row.running_state.as_str(), "running" | "starting"))
        .collect::<Vec<_>>();
    if live.is_empty() {
        lines.push("  Local model servers: none running under ROCm CLI.".to_owned());
    } else {
        lines.push(format!(
            "  Local model servers running/starting: {}.",
            live.len()
        ));
        for row in live.iter().take(4) {
            lines.push(format!(
                "    {} {} ({}) at {}.",
                row.engine,
                row.model,
                row.running_state,
                empty_as_unknown(&row.endpoint)
            ));
        }
    }
    let past = rows
        .iter()
        .filter(|row| !matches!(row.running_state.as_str(), "running" | "starting"))
        .take(3)
        .map(|row| {
            format!(
                "{} {} {}",
                empty_as_unknown(&row.engine),
                empty_as_unknown(&row.model),
                empty_as_unknown(&row.status)
            )
        })
        .collect::<Vec<_>>();
    if !past.is_empty() {
        lines.push(format!("  Past/non-running records: {}.", past.join("; ")));
    }
    (!lines.is_empty()).then(|| lines.join("\n"))
}

fn deterministic_port_status_tool_summary(tool_text: &str) -> Option<String> {
    if !tool_text.contains("port_status:") && !tool_text.contains("listening:") {
        return None;
    }
    let port = chat_tool_value(tool_text, "port")?;
    let listening = chat_tool_value(tool_text, "listening").unwrap_or_else(|| "unknown".to_owned());
    let mut lines = Vec::new();
    let state = match listening.as_str() {
        "true" => "listening",
        "false" => "not listening",
        _ => "unknown",
    };
    lines.push(format!("  Port {port}: {state}."));
    if let Some(hint) = chat_tool_value(tool_text, "hint") {
        lines.push(format!("  Hint: {hint}."));
    }
    if chat_tool_value(tool_text, "managed_service").as_deref() == Some("none") {
        lines.push("  Managed service: none on that endpoint.".to_owned());
    }
    Some(lines.join("\n"))
}

fn deterministic_engine_inventory_tool_summary(tool_text: &str) -> Option<String> {
    if !tool_text.contains("Local model engines") {
        return None;
    }
    let rows = parse_deterministic_engine_rows(tool_text);
    if rows.is_empty() {
        return None;
    }
    let mut lines = vec!["  Engine runtimes:".to_owned()];
    for row in rows {
        let mut line = format!(
            "    {}: {}",
            friendly_engine_label(&row.engine),
            empty_as_unknown(&row.runtime)
        );
        if !row.note.trim().is_empty() {
            let _ = write!(line, " ({})", row.note.trim());
        }
        line.push('.');
        lines.push(line);
    }
    Some(lines.join("\n"))
}

fn should_request_local_tool_follow_up(
    provider: &str,
    tool_result: &ChatToolRunResult,
    deterministic_summary: Option<&str>,
) -> bool {
    provider == "local"
        && deterministic_summary.is_none()
        && tool_result.approval.is_none()
        && tool_result.ran_read_only_tool
        && !tool_result.read_only_tool_error
        && !tool_result.follow_up_text.trim().is_empty()
}

fn chat_tool_value(tool_text: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}:");
    tool_text.lines().find_map(|line| {
        line.trim()
            .strip_prefix(&prefix)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    })
}

// pub(crate): called from main.rs's own MCP dispatch and from the dash
// execution seam (dash_seam.rs)
pub(crate) fn run_internal_mcp_call(
    paths: &AppPaths,
    name: &str,
    arguments: serde_json::Value,
    allow_mutation: bool,
) -> Result<serde_json::Value> {
    let arguments = internal_mcp_arguments(arguments);
    let call = providers::ChatToolCall {
        id: None,
        name: name.to_owned(),
        arguments: serde_json::Value::Object(arguments.clone()),
    };
    validate_chat_tool_call(&call)?;
    if !chat_tool_call_is_read_only(&call) && !allow_mutation {
        bail!(
            "MCP tool `{name}` changes local ROCm state; rerun `rocm mcp-call {name}` with --allow-mutation only after explicit user approval"
        );
    }

    match name {
        "examine" | "doctor" => {
            let examine = ExamineSummary::gather()?;
            let text = render_examine_text()?;
            Ok(internal_mcp_tool_success(text, serde_json::json!(examine)))
        }
        "bridge_snapshot" => {
            let snapshot = build_codex_bridge_snapshot(paths)?;
            Ok(internal_mcp_tool_success(
                format!(
                    "Captured bridge snapshot for {} / {} with default engine `{}`.",
                    snapshot.examine.os, snapshot.examine.arch, snapshot.examine.default_engine
                ),
                serde_json::json!(snapshot),
            ))
        }
        "gpu_snapshot" => {
            let config = RocmCliConfig::load(paths).unwrap_or_default();
            let gpu = build_codex_bridge_gpu_snapshot(&config);
            let status = if !config.telemetry.local_inspection_enabled() {
                "GPU telemetry is disabled by rocm-cli config."
            } else if gpu.amd_smi_available {
                "Captured amd-smi GPU snapshot."
            } else {
                "Use `rocm examine` for the current local AMD GPU summary."
            };
            Ok(internal_mcp_tool_success(
                status.to_owned(),
                serde_json::json!(gpu),
            ))
        }
        "engines" => {
            let engines = builtin_codex_bridge_engine_inventory();
            Ok(internal_mcp_tool_success(
                format!("Found {} engine entries.", engines.len()),
                serde_json::json!({ "engines": engines }),
            ))
        }
        "services" => {
            let services = load_managed_services(paths)?;
            Ok(internal_mcp_tool_success(
                render_services_tool_result_text(&services),
                serde_json::json!({ "services": services }),
            ))
        }
        "port_status" => run_chat_port_status_tool(paths, &call),
        "service_logs" => {
            let service_id = json_string(&arguments, "service_id")
                .context("service_logs requires `service_id`")?;
            let lines = arguments
                .get("lines")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(80)
                .clamp(1, 500) as usize;
            let record = load_managed_services(paths)?
                .into_iter()
                .find(|service| service.service_id == service_id)
                .with_context(|| format!("managed service `{service_id}` not found"))?;
            let tail = read_tail_lines(&record.log_path, lines, "service log")?.join("\n");
            Ok(internal_mcp_tool_success(
                format!(
                    "Read the last {} line(s) from service `{}`.",
                    lines, record.service_id
                ),
                serde_json::json!({
                    "service": record,
                    "lines": lines,
                    "tail": tail,
                }),
            ))
        }
        "automations" => {
            let event_limit = arguments
                .get("event_limit")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(10)
                .clamp(1, 64) as usize;
            let runtime = AutomationRuntimeState::load(paths)?;
            let events = load_recent_automation_events(paths, event_limit)?;
            Ok(internal_mcp_tool_success(
                format!(
                    "Loaded automation runtime and {} recent events.",
                    events.len()
                ),
                serde_json::json!({
                    "runtime": runtime,
                    "recent_events": events,
                }),
            ))
        }
        "natural_language_plan" => {
            let request = json_string(&arguments, "request")
                .context("natural_language_plan requires `request`")?;
            let config = RocmCliConfig::load(paths).unwrap_or_default();
            let text = render_freeform_plan(&request, paths, &config);
            let action =
                freeform_plan_next_action_with_context(&request, paths, &config).map(|action| {
                    serde_json::json!({
                        "args": action.args,
                        "approval_required": action.approval_required,
                        "has_placeholders": action.has_placeholders,
                        "provider_assisted": action.provider_assisted,
                    })
                });
            Ok(internal_mcp_tool_success(
                "Planned the ROCm request.".to_owned(),
                serde_json::json!({
                    "request": request,
                    "text": text,
                    "action": action,
                }),
            ))
        }
        "rocm_command" => {
            let action = chat_rocm_command_action(&call)?;
            let output = match action {
                ChatRocmCommandAction::ReadOnly(args) => {
                    run_rocm_command_for_paths(paths, &args, Duration::from_mins(2))?
                }
                ChatRocmCommandAction::Approval { args, .. } if allow_mutation => {
                    let refs = args.iter().map(String::as_str).collect::<Vec<_>>();
                    run_rocm_capture_for_paths(paths, &refs, Duration::from_mins(2))?
                }
                ChatRocmCommandAction::Approval { .. } => {
                    bail!("rocm_command changes local ROCm state and needs approval")
                }
            };
            Ok(internal_mcp_tool_result_from_command(
                "Ran `rocm` command.",
                output,
                false,
            ))
        }
        "update_check" => {
            let output =
                run_rocm_command_for_paths(paths, &["update".to_owned()], Duration::from_mins(1))?;
            Ok(internal_mcp_tool_result_from_command(
                "Ran `rocm update`.",
                output,
                false,
            ))
        }
        "install_sdk_dry_run" => {
            let args = internal_mcp_install_sdk_args(&arguments, true)?;
            let output = run_rocm_command_for_paths(paths, &args, Duration::from_mins(2))?;
            Ok(internal_mcp_tool_result_from_command(
                "Ran `rocm install sdk --dry-run`.",
                output,
                false,
            ))
        }
        "path_exists" => run_chat_path_exists_tool(&call),
        "proposal_action" => {
            // Executes IN-PROCESS (no CLI subprocess): show loads a proposal;
            // approve/reject mutate its status (allow_mutation already enforced
            // by the read-only split above). proposal_action executes in-process
            // (via update_automation_proposal_status); it never delegates to the
            // subprocess arm.
            let proposal_id = json_string(&arguments, "proposal_id")
                .context("proposal_action requires `proposal_id`")?;
            let action =
                json_string(&arguments, "action").context("proposal_action requires `action`")?;
            match action.as_str() {
                "show" => {
                    let proposal = rocm_core::find_automation_proposal(paths, &proposal_id)
                        .with_context(|| {
                            format!("automation proposal `{proposal_id}` not found")
                        })?;
                    Ok(internal_mcp_tool_success(
                        format!(
                            "Proposal {} ({}): {}",
                            proposal.proposal_id, proposal.status, proposal.title
                        ),
                        serde_json::json!({
                            "id": proposal.proposal_id,
                            "status": proposal.status,
                            "summary": proposal.title,
                            "reason": proposal.message,
                        }),
                    ))
                }
                "approve" | "reject" => {
                    let status = if action == "approve" {
                        "approved"
                    } else {
                        "rejected"
                    };
                    let updated =
                        rocm_core::update_automation_proposal_status(paths, &proposal_id, status)?;
                    record_cli_audit_event(
                        paths,
                        "automations",
                        if action == "approve" {
                            "proposal_approved"
                        } else {
                            "proposal_rejected"
                        },
                        "info",
                        format!("proposal {proposal_id} {status}"),
                        None,
                    );
                    Ok(internal_mcp_tool_success(
                        format!("Proposal {proposal_id} {status}."),
                        serde_json::json!({
                            "id": updated.proposal_id,
                            "status": updated.status,
                            "summary": updated.title,
                            "reason": updated.message,
                        }),
                    ))
                }
                other => bail!("proposal_action `{other}` is not supported"),
            }
        }
        "install_sdk" | "install_engine" | "launch_server" | "stop_server" | "watcher_enable"
        | "watcher_disable" => {
            let args = rocm_chat_tool_requested_args(&call)
                .with_context(|| format!("MCP tool `{name}` is missing required arguments"))?;
            let refs = args.iter().map(String::as_str).collect::<Vec<_>>();
            let output = run_rocm_capture_for_paths(paths, &refs, Duration::from_mins(2))?;
            Ok(internal_mcp_tool_result_from_command(
                "Ran approved `rocm` command.",
                output,
                false,
            ))
        }
        other => bail!("unsupported MCP tool `{other}`"),
    }
}

fn internal_mcp_arguments(value: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
    if let Some(arguments) = value
        .get("arguments")
        .and_then(serde_json::Value::as_object)
    {
        return arguments.clone();
    }
    value.as_object().cloned().unwrap_or_default()
}

fn internal_mcp_tool_success(text: String, structured: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
        "structuredContent": structured,
        "isError": false,
    })
}

fn internal_mcp_tool_result_from_command(
    prefix: &str,
    output: CommandCapture,
    is_error: bool,
) -> serde_json::Value {
    let text = format!("{prefix}\n\n{}", command_capture_text(&output));
    serde_json::json!({
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
        "structuredContent": {
            "argv": output.argv,
            "exit_status": output.exit_status,
            "stdout": output.stdout,
            "stderr": output.stderr,
        },
        "isError": is_error || output.exit_status != 0,
    })
}

fn run_rocm_capture_for_paths(
    paths: &AppPaths,
    args: &[&str],
    timeout: Duration,
) -> Result<CommandCapture> {
    let rocm_binary = daemon_binary_path()?;
    let mut command = ProcessCommand::new(&rocm_binary);
    // Stdio left to `run_with_timeout`, which sets it itself.
    command.args(args);
    apply_app_path_env(&mut command, paths);
    let output = rocm_core::process::run_with_timeout(command, timeout, "process")
        .with_context(|| format!("failed to run {}", rocm_binary.display()))?;
    Ok(CommandCapture {
        argv: std::iter::once(rocm_binary.display().to_string())
            .chain(args.iter().map(|value| (*value).to_owned()))
            .collect(),
        exit_status: output.status.code().unwrap_or(1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    })
}

fn run_rocm_command_for_paths(
    paths: &AppPaths,
    args: &[String],
    _timeout: Duration,
) -> Result<CommandCapture> {
    match run_rocm_read_only_in_process(paths, args) {
        Ok(stdout) => Ok(CommandCapture {
            argv: std::iter::once("rocm".to_owned())
                .chain(args.iter().cloned())
                .collect(),
            exit_status: 0,
            stdout,
            stderr: String::new(),
        }),
        Err(error) => Err(error).with_context(|| {
            format!(
                "read-only assistant command is not implemented in-process: {}",
                format_structured_tool_call("rocm", args)
            )
        }),
    }
}

fn run_rocm_read_only_in_process(paths: &AppPaths, args: &[String]) -> Result<String> {
    let config = RocmCliConfig::load(paths).unwrap_or_default();
    match args {
        [] => bail!("rocm command requires at least one argument"),
        [command] if command.eq_ignore_ascii_case("examine") => {
            render_examine_text_with_paths(paths, &config)
        }
        [command]
            if command.eq_ignore_ascii_case("version")
                || command == "--version"
                || command == "-V" =>
        {
            Ok(format!("rocm-cli {}\n", cli_version_string()))
        }
        [command]
            if command.eq_ignore_ascii_case("model") || command.eq_ignore_ascii_case("models") =>
        {
            Ok(render_model_registry_verbose_text_with_context_and_host(
                Some(paths),
                None,
                None,
            ))
        }
        [command] if command.eq_ignore_ascii_case("daemon") => {
            Ok(render_daemon_text(paths, &config))
        }
        [command] if command.eq_ignore_ascii_case("logs") => Ok(render_logs_text(paths)),
        [command, rest @ ..] if command.eq_ignore_ascii_case("logs") => {
            let query = parse_optional_query(rest)?;
            Ok(render_logs_browser_text(paths, query.as_deref()))
        }
        [command] if command.eq_ignore_ascii_case("runtimes") => {
            render_runtimes_text(paths, &config)
        }
        [command, subcommand]
            if command.eq_ignore_ascii_case("runtimes")
                && subcommand.eq_ignore_ascii_case("list") =>
        {
            render_runtimes_text(paths, &config)
        }
        [command, subcommand]
            if command.eq_ignore_ascii_case("engines")
                && subcommand.eq_ignore_ascii_case("list") =>
        {
            Ok(render_engine_inventory_text_with_paths(Some(paths)))
        }
        [command] if command.eq_ignore_ascii_case("services") => render_services_text(paths, false),
        [command, subcommand]
            if command.eq_ignore_ascii_case("services")
                && subcommand.eq_ignore_ascii_case("list") =>
        {
            render_services_text(paths, false)
        }
        [command, subcommand, flag]
            if command.eq_ignore_ascii_case("services")
                && subcommand.eq_ignore_ascii_case("list")
                && matches!(flag.as_str(), "-a" | "--all") =>
        {
            render_services_text(paths, true)
        }
        [command, subcommand, service_id]
            if command.eq_ignore_ascii_case("services")
                && matches!(subcommand.to_ascii_lowercase().as_str(), "logs" | "log") =>
        {
            render_service_logs_text(paths, service_id)
        }
        [command] if command.eq_ignore_ascii_case("automations") => {
            render_automations_text(paths, &config)
        }
        [command, subcommand]
            if command.eq_ignore_ascii_case("automations")
                && subcommand.eq_ignore_ascii_case("list") =>
        {
            render_automations_text(paths, &config)
        }
        [command, subcommand]
            if command.eq_ignore_ascii_case("config")
                && subcommand.eq_ignore_ascii_case("show") =>
        {
            Ok(render_config_text(paths, &config))
        }
        [command] if command.eq_ignore_ascii_case("comfyui") => {
            comfyui::render_status(paths, &config)
        }
        [command, subcommand]
            if command.eq_ignore_ascii_case("comfyui")
                && subcommand.eq_ignore_ascii_case("status") =>
        {
            comfyui::render_status(paths, &config)
        }
        [command, subcommand, rest @ ..]
            if command.eq_ignore_ascii_case("comfyui")
                && matches!(subcommand.to_ascii_lowercase().as_str(), "logs" | "log") =>
        {
            comfyui::render_logs(
                paths,
                parse_optional_lines(rest).unwrap_or(DEFAULT_LOG_TAIL_LINES),
            )
        }
        [command, rest @ ..]
            if command.eq_ignore_ascii_case("update")
                && !rest.iter().any(|arg| arg.eq_ignore_ascii_case("--apply")) =>
        {
            render_update_text(paths)
        }
        [command, subcommand, rest @ ..]
            if command.eq_ignore_ascii_case("install")
                && subcommand.eq_ignore_ascii_case("sdk")
                && rest.iter().any(|arg| arg.eq_ignore_ascii_case("--dry-run")) =>
        {
            render_install_sdk_dry_run_for_args(paths, rest)
        }
        [command, rest @ ..]
            if command.eq_ignore_ascii_case("uninstall")
                && rest.iter().any(|arg| arg.eq_ignore_ascii_case("--dry-run")) =>
        {
            render_uninstall_dry_run(paths)
        }
        [command] if command.eq_ignore_ascii_case("setup") => {
            render_setup_status_text(paths, &config)
        }
        [command, subcommand]
            if command.eq_ignore_ascii_case("setup")
                && subcommand.eq_ignore_ascii_case("status") =>
        {
            render_setup_status_text(paths, &config)
        }
        _ => bail!(
            "unsupported in-process read-only rocm command: {}",
            format_structured_tool_call("rocm", args)
        ),
    }
}

/// Map the parsed `install sdk` arguments onto the install request.
///
/// Extracted from the command arm so this mapping has a seam. `include_devel`
/// is the field that most needs one: nothing downstream re-derives it, so if
/// the flag stopped being forwarded here the install would silently go back to
/// pulling the compiler toolchain and every other assertion would still pass.
#[allow(clippy::too_many_arguments)]
pub(crate) const fn sdk_install_request<'a>(
    channel: &'a str,
    format: &'a str,
    prefix: Option<PathBuf>,
    version_selector: Option<therock::RuntimeVersionSelector>,
    family_override: Option<&'a str>,
    dry_run: bool,
    devel: bool,
    consent: therock::SdkInstallConsent,
) -> therock::SdkInstallRequest<'a> {
    therock::SdkInstallRequest {
        channel,
        format,
        prefix,
        version_selector,
        family_override,
        dry_run,
        include_devel: devel,
        consent,
    }
}

fn render_install_sdk_dry_run_for_args(paths: &AppPaths, args: &[String]) -> Result<String> {
    let channel = chat_cli_arg_value(args, "--channel").unwrap_or("release");
    let format = chat_cli_arg_value(args, "--format").unwrap_or("wheel");
    let prefix = chat_cli_arg_value(args, "--prefix").map(PathBuf::from);
    let version = chat_cli_arg_value(args, "--version").map(str::to_owned);
    let build_date = chat_cli_arg_value(args, "--build-date").map(str::to_owned);
    let selector = therock_install_version_selector(version, build_date)?;
    let devel = chat_cli_has_flag(args, "--devel");
    // Dry run, so nothing is displaced and the consent gate is never reached;
    // the narrow consent is what this chat surface would pass for a real
    // install, and passing `--yes`'s source here would be a lie waiting to be
    // printed if the preview ever grew a gate.
    Ok(therock::install_sdk(
        paths,
        therock::SdkInstallRequest {
            channel,
            format,
            prefix,
            version_selector: selector,
            dry_run: true,
            include_devel: devel,
            consent: therock::SdkInstallConsent::Preapproved(
                therock::SdkInstallApprovalSource::ApproveReplacingActiveDefault,
            ),
            ..therock::SdkInstallRequest::default()
        },
    )?
    .output)
}

fn internal_mcp_install_sdk_args(
    arguments: &serde_json::Map<String, serde_json::Value>,
    dry_run: bool,
) -> Result<Vec<String>> {
    let channel = json_string(arguments, "channel").unwrap_or_else(|| "release".to_owned());
    let format = json_string(arguments, "format").unwrap_or_else(|| "wheel".to_owned());
    let mut argv = vec![
        "install".to_owned(),
        "sdk".to_owned(),
        "--channel".to_owned(),
        channel,
        "--format".to_owned(),
        format,
    ];
    if let Some(prefix) = json_string(arguments, "prefix") {
        let prefix_path = Path::new(&prefix);
        if chat_install_prefix_is_system(prefix_path) {
            bail!(
                "install_sdk prefix `{}` is a system folder; choose a user folder instead",
                prefix_path.display()
            );
        }
        argv.push("--prefix".to_owned());
        argv.push(prefix);
    }
    if let Some(version) = json_string(arguments, "version") {
        argv.push("--version".to_owned());
        argv.push(version);
    }
    if let Some(build_date) = json_string(arguments, "build_date") {
        argv.push("--build-date".to_owned());
        argv.push(build_date);
    }
    if dry_run {
        argv.push("--dry-run".to_owned());
    }
    Ok(argv)
}

fn run_chat_read_only_tool(
    paths: &AppPaths,
    call: &providers::ChatToolCall,
) -> Result<serde_json::Value> {
    match call.name.as_str() {
        "path_exists" => run_chat_path_exists_tool(call),
        "port_status" => run_chat_port_status_tool(paths, call),
        "rocm_command" => {
            let action = chat_rocm_command_action(call)?;
            let ChatRocmCommandAction::ReadOnly(args) = action else {
                bail!("assistant read-only path cannot run mutating rocm_command");
            };
            let output = run_rocm_command_for_paths(paths, &args, Duration::from_mins(2))?;
            Ok(internal_mcp_tool_result_from_command(
                "Ran `rocm` command.",
                output,
                false,
            ))
        }
        "update_check" => {
            let output =
                run_rocm_command_for_paths(paths, &["update".to_owned()], Duration::from_mins(1))?;
            Ok(internal_mcp_tool_result_from_command(
                "Ran `rocm update`.",
                output,
                false,
            ))
        }
        "install_sdk_dry_run" => {
            let arguments = internal_mcp_arguments(call.arguments.clone());
            let args = internal_mcp_install_sdk_args(&arguments, true)?;
            let output = run_rocm_command_for_paths(paths, &args, Duration::from_mins(2))?;
            Ok(internal_mcp_tool_result_from_command(
                "Ran `rocm install sdk --dry-run`.",
                output,
                false,
            ))
        }
        _ => run_internal_mcp_call(paths, &call.name, call.arguments.clone(), false),
    }
}

fn run_chat_path_exists_tool(call: &providers::ChatToolCall) -> Result<serde_json::Value> {
    let object = call
        .arguments
        .as_object()
        .context("path_exists arguments must be a JSON object")?;
    let path = json_string(object, "path").context("path_exists requires path")?;
    let path = Path::new(&path);
    let metadata = path.metadata().ok();
    let path_kind = metadata.as_ref().map_or("missing", |metadata| {
        if metadata.is_dir() {
            "directory"
        } else if metadata.is_file() {
            "file"
        } else {
            "other"
        }
    });
    let parent = path.parent();
    let parent_exists = parent.is_some_and(Path::exists);
    let parent_display = parent.map_or_else(
        || "<none>".to_owned(),
        |parent| parent.display().to_string(),
    );
    let text = format!(
        "path: {}\nexists: {}\nkind: {}\nparent: {}\nparent_exists: {}",
        path.display(),
        metadata.is_some(),
        path_kind,
        parent_display,
        parent_exists
    );
    Ok(serde_json::json!({
        "content": [{
            "type": "text",
            "text": text,
        }]
    }))
}

fn run_chat_port_status_tool(
    paths: &AppPaths,
    call: &providers::ChatToolCall,
) -> Result<serde_json::Value> {
    let object = call
        .arguments
        .as_object()
        .context("port_status arguments must be a JSON object")?;
    let host = json_string(object, "host").unwrap_or_else(|| DEFAULT_LOCAL_HOST.to_owned());
    let port = object
        .get("port")
        .and_then(serde_json::Value::as_u64)
        .context("port_status requires port")? as u16;
    let reachable = loopback_tcp_port_is_reachable(&host, port);
    let host_key = loopback_host_key(&host);
    let matching_services = load_managed_services(paths)?
        .into_iter()
        .filter(|record| {
            record.port == port
                && loopback_host_key(&record.host) == host_key
                && managed_service_is_live(record)
        })
        .collect::<Vec<_>>();
    let app_hint = if port == comfyui::default_port() {
        Some("ComfyUI default port")
    } else if port == rocm_core::DEFAULT_LOCAL_PORT {
        Some("ROCm local model server default port")
    } else {
        None
    };
    let mut text = String::new();
    let _ = writeln!(text, "host: {host}");
    let _ = writeln!(text, "port: {port}");
    let _ = writeln!(text, "listening: {reachable}");
    if let Some(app_hint) = app_hint {
        let _ = writeln!(text, "hint: {app_hint}");
    }
    if matching_services.is_empty() {
        let _ = writeln!(text, "managed_service: none");
    } else {
        let _ = writeln!(text, "managed_services:");
        for service in &matching_services {
            let _ = writeln!(
                text,
                "  - service_id={} engine={} model={} status={} running_state={} endpoint={}",
                service.service_id,
                service.engine,
                service.model_ref,
                service.status,
                managed_service_running_state(&service.status),
                service.endpoint_url
            );
        }
    }
    Ok(serde_json::json!({
        "content": [{
            "type": "text",
            "text": text,
        }],
        "structuredContent": {
            "host": host,
            "port": port,
            "listening": reachable,
            "hint": app_hint,
            "managed_services": matching_services,
        },
        "isError": false,
    }))
}

const fn chat_read_only_tool_status_label(is_error: bool) -> &'static str {
    if is_error {
        "reported an error"
    } else {
        "done"
    }
}

fn rocm_chat_tool_requested_command(call: &providers::ChatToolCall) -> Option<String> {
    let args = rocm_chat_tool_requested_args(call)?;
    Some(format_structured_tool_call("rocm", &args))
}

pub(crate) fn rocm_chat_tool_requested_args(call: &providers::ChatToolCall) -> Option<Vec<String>> {
    let object = call.arguments.as_object()?;
    match call.name.as_str() {
        "install_sdk" => {
            let mut args = vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                json_string(object, "channel").unwrap_or_else(|| "release".to_owned()),
                "--format".to_owned(),
                json_string(object, "format").unwrap_or_else(|| "wheel".to_owned()),
                // The MCP surface runs `rocm` with null stdin, so the consent
                // prompt would refuse. This keeps the tool non-interactive,
                // matching the chat `install sdk` arm. Deliberately not `--yes`:
                // that would additionally approve a `sudo` system-package
                // install this spawn has no terminal to answer.
                "--approve-replacing-active-default".to_owned(),
            ];
            if let Some(prefix) = json_string(object, "prefix") {
                args.push("--prefix".to_owned());
                args.push(prefix);
            }
            if let Some(version) = json_string(object, "version") {
                args.push("--version".to_owned());
                args.push(version);
            }
            if let Some(build_date) = json_string(object, "build_date") {
                args.push("--build-date".to_owned());
                args.push(build_date);
            }
            Some(args)
        }
        "install_engine" => {
            let engine = json_string(object, "engine")?;
            let mut args = vec!["engines".to_owned(), "install".to_owned(), engine];
            if let Some(runtime_id) = json_string(object, "runtime_id") {
                args.push("--runtime-id".to_owned());
                args.push(runtime_id);
            }
            if let Some(python_version) = json_string(object, "python_version") {
                args.push("--python-version".to_owned());
                args.push(python_version);
            }
            if object
                .get("reinstall")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
            {
                args.push("--reinstall".to_owned());
            }
            Some(args)
        }
        "launch_server" => {
            let model = json_string(object, "model")?;
            let mut args = vec!["serve".to_owned(), model, "--managed".to_owned()];
            push_optional_json_cli_arg(&mut args, object, "--engine", "engine");
            push_optional_json_cli_arg(&mut args, object, "--device", "device");
            push_optional_json_cli_arg(&mut args, object, "--runtime-id", "runtime_id");
            push_optional_json_cli_arg(&mut args, object, "--env-id", "env_id");
            push_optional_json_cli_arg(&mut args, object, "--host", "host");
            if let Some(port) = object.get("port").and_then(serde_json::Value::as_u64) {
                args.push("--port".to_owned());
                args.push(port.to_string());
            }
            Some(args)
        }
        "stop_server" => {
            let service_id = json_string(object, "service_id")?;
            Some(vec![
                "services".to_owned(),
                "stop".to_owned(),
                service_id,
                "--yes".to_owned(),
            ])
        }
        "watcher_enable" => {
            let watcher = json_string(object, "watcher")?;
            let mut args = vec!["automations".to_owned(), "enable".to_owned(), watcher];
            push_optional_json_cli_arg(&mut args, object, "--mode", "mode");
            Some(args)
        }
        "watcher_disable" => {
            let watcher = json_string(object, "watcher")?;
            Some(vec![
                "automations".to_owned(),
                "disable".to_owned(),
                watcher,
            ])
        }
        "rocm_command" => match chat_rocm_command_action(call).ok()? {
            ChatRocmCommandAction::Approval { args, .. }
            | ChatRocmCommandAction::ReadOnly(args) => Some(args),
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_install_finalize::tests::consents_for_install_sdk_argv;
    use crate::tests::test_paths;
    use crate::{Cli, Command, InstallTarget};
    use clap::Parser;
    use rocm_core::ManagedServiceRecord;
    use std::fs;
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
    fn runtimes_activate_approval_modal_uses_the_runtime_noun() {
        // `comfyui install`'s selection errors steer the user to `rocm runtimes`
        // and say "ROCm runtime". Running a mutating `rocm runtimes …` from chat
        // raises an approval modal whose title is written here, not by clap, so
        // `main.rs`'s `runtimes_help_uses_the_runtime_noun_throughout` (which
        // covers `--help` text) cannot reach it. It said "Change ROCm install"
        // until this rename.
        let action = chat_rocm_command_action_from_args(vec![
            "runtimes".to_owned(),
            "activate".to_owned(),
            "some-runtime-key".to_owned(),
        ])
        .expect("a mutating runtimes command classifies");
        let ChatRocmCommandAction::Approval { pending_title, .. } = action else {
            panic!("`rocm runtimes activate` must require approval, got {action:?}");
        };
        assert!(
            pending_title.contains("runtime") && !pending_title.contains("install"),
            "the `runtimes` approval modal must use the `runtime` noun, got {pending_title:?}"
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

    #[test]
    fn services_tool_result_text_includes_running_interpretation() {
        let (_root, paths) = test_paths("services-tool-text");
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-vllm",
            "vllm",
            "Qwen/Qwen3.5",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            std::process::id(),
            Some("therock-release".to_owned()),
            None,
            Some("gpu_required".to_owned()),
        );
        record.status = "ready".to_owned();

        let rendered = render_services_tool_result_text(&[record]);

        assert!(rendered.contains("status_meaning: ready/running = running"));
        assert!(rendered.contains("engine=vllm"));
        assert!(rendered.contains("running_state=running"));
    }

    #[test]
    fn chat_tool_call_mutating_install_maps_to_reviewable_rocm_command() {
        let call = providers::ChatToolCall {
            id: Some("call-1".to_owned()),
            name: "install_sdk".to_owned(),
            arguments: serde_json::json!({
                "channel": "release",
                "format": "wheel",
                "prefix": "D:\\ROCm\\therock_venvs"
            }),
        };

        assert!(!chat_tool_call_is_read_only(&call));
        validate_chat_tool_call(&call).expect("install request should validate for review");
        assert_eq!(
            rocm_chat_tool_requested_command(&call).as_deref(),
            Some(
                "rocm install sdk --channel release --format wheel --approve-replacing-active-default --prefix D:\\ROCm\\therock_venvs"
            )
        );
        let approval = chat_tool_approval_request(
            &call,
            Some("TheRock is not installed yet, so I need to install ROCm first."),
        )
        .expect("approval should be built");
        assert_eq!(approval.pending_title, "Install ROCm");
        assert_eq!(approval.command_title, "Install");
        assert_eq!(
            approval.explanation.as_deref(),
            Some("TheRock is not installed yet, so I need to install ROCm first.")
        );
        assert_eq!(
            approval.args,
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                "release".to_owned(),
                "--format".to_owned(),
                "wheel".to_owned(),
                "--approve-replacing-active-default".to_owned(),
                "--prefix".to_owned(),
                "D:\\ROCm\\therock_venvs".to_owned(),
            ]
        );
    }

    #[test]
    fn chat_install_sdk_strips_model_supplied_yes_and_skips_consent_on_dry_run() {
        // `rocm_command` carries a model-supplied argv that nothing else filters,
        // so `--yes` can arrive here. This arm exists to grant the narrow consent
        // only; letting `--yes` through would re-grant the system-package/sudo
        // consent on a spawn with no terminal to answer a password prompt.
        let classify = |args: &[&str]| -> Vec<String> {
            let action = chat_rocm_command_action_from_args(
                args.iter().copied().map(str::to_owned).collect(),
            )
            .expect("install sdk should classify");
            let ChatRocmCommandAction::Approval { args, .. } = action else {
                panic!("install sdk is a mutating command");
            };
            args
        };

        assert_eq!(
            classify(&["install", "sdk", "--prefix", "/tmp/therock", "--yes"]),
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--prefix".to_owned(),
                "/tmp/therock".to_owned(),
                "--approve-replacing-active-default".to_owned(),
            ]
        );

        // A dry run returns before the consent gate, so it is not given a consent
        // it never uses — matching the dry-run-aware sibling arms.
        assert_eq!(
            classify(&["install", "sdk", "--prefix", "/tmp/therock", "--dry-run"]),
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--prefix".to_owned(),
                "/tmp/therock".to_owned(),
                "--dry-run".to_owned(),
            ]
        );

        // Both together: the strip is unconditional, so `--yes` still does not
        // survive into a preview spawn, and the dry run still gains no consent.
        // A model that emits both must not end up with either flag.
        assert_eq!(
            classify(&[
                "install",
                "sdk",
                "--prefix",
                "/tmp/therock",
                "--yes",
                "--dry-run",
            ]),
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--prefix".to_owned(),
                "/tmp/therock".to_owned(),
                "--dry-run".to_owned(),
            ]
        );
    }

    #[test]
    fn chat_install_sdk_strips_a_model_supplied_yes_in_both_its_bare_and_attached_forms() {
        // `--yes` and `--yes=true` are both model-supplied argv that reach the
        // chat arm intact: neither `canonicalize_chat_rocm_command` nor
        // `validate_chat_rocm_command_safety` splits or rejects either. Whichever
        // form survives re-grants into a null-stdin spawn the system-package/sudo
        // consent `76c6aa3c` removed, with no terminal to answer the password
        // prompt, so the strip has to catch both.
        let classify = |args: &[&str]| -> Vec<String> {
            let action = chat_rocm_command_action_from_args(
                args.iter().copied().map(str::to_owned).collect(),
            )
            .expect("install sdk should classify");
            let ChatRocmCommandAction::Approval { args, .. } = action else {
                panic!("install sdk is a mutating command");
            };
            args
        };

        // Both terms of `arg != "--yes" && !arg.starts_with("--yes=")` are driven
        // here, and each alone: the bare form is caught only by the first, the
        // `=` forms only by the second, so dropping either term reddens this test
        // on its own rather than leaving one half to a sibling.
        for supplied in ["--yes", "--yes=true", "--yes=1", "--yes=false"] {
            let args = classify(&["install", "sdk", "--prefix", "/tmp/therock", supplied]);
            assert!(
                !args.iter().any(|arg| arg.starts_with("--yes")),
                "`{supplied}` must not survive the chat strip, got {args:?}"
            );
            assert_eq!(
                args,
                vec![
                    "install".to_owned(),
                    "sdk".to_owned(),
                    "--prefix".to_owned(),
                    "/tmp/therock".to_owned(),
                    "--approve-replacing-active-default".to_owned(),
                ],
                "stripping `{supplied}` must leave the rest of the argv and the narrow consent alone"
            );
        }

        // Future-proofing, not a guard on the `--yes=` term this test's other
        // assertions pin: `--yes-not-a-flag` survives both the exact-match strip
        // that preceded that term and the two-term strip that replaced it, so it
        // would pass on either. What it does catch is the next edit — widening
        // the second term to `starts_with("--yes")` to "simplify" it would start
        // eating every argv token that merely begins the same way, silently
        // dropping arguments the model legitimately sent.
        let args = classify(&[
            "install",
            "sdk",
            "--prefix",
            "/tmp/therock",
            "--yes-not-a-flag",
        ]);
        assert!(
            args.iter().any(|arg| arg == "--yes-not-a-flag"),
            "the strip must match `--yes` and `--yes=…`, not every token starting with \
             `--yes`, got {args:?}"
        );

        // Second layer, and only the second: clap also refuses an attached value
        // on this flag, so even an unstripped `--yes=true` would not parse today.
        // That is what the strip above deliberately stops depending on — pinned
        // here so a later `num_args` on `--yes` shows up as a failure of the
        // backstop rather than passing unnoticed.
        for attached in ["--yes=true", "--yes=1", "--yes=false"] {
            let error = match Cli::try_parse_from(["rocm", "install", "sdk", attached]) {
                Ok(cli) => panic!("`{attached}` must not parse, got {cli:?}"),
                Err(error) => error,
            };
            assert_eq!(
                error.kind(),
                clap::error::ErrorKind::TooManyValues,
                "expected clap to reject the attached value on {attached}: {error}"
            );
        }

        // Control: the bare form does parse, so the assertions above are about
        // the `=`-form and not about `--yes` being rejected outright.
        let cli = Cli::try_parse_from(["rocm", "install", "sdk", "--yes"])
            .expect("the bare flag is the form the chat arm strips");
        match cli.command {
            Some(Command::Install {
                target: InstallTarget::Sdk { yes, .. },
            }) => assert!(yes),
            other => panic!("expected `install sdk`, got {other:?}"),
        }
    }

    #[test]
    fn chat_tool_call_mutating_install_accepts_requested_build_date() {
        let call = providers::ChatToolCall {
            id: Some("call-date".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["install", "sdk", "--channel", "release", "--format", "wheel", "--prefix", "D:\\ROCm\\therock_venvs", "--build-date", "06052026"],
                "reason": "The user asked for the TheRock build from 2026-06-05."
            }),
        };

        validate_chat_tool_call(&call).expect("date-specific install should validate for review");
        assert!(!chat_tool_call_is_read_only(&call));
        assert_eq!(
            rocm_chat_tool_requested_command(&call).as_deref(),
            Some(
                "rocm install sdk --channel release --format wheel --prefix D:\\ROCm\\therock_venvs --build-date 06052026 --approve-replacing-active-default"
            )
        );
        let approval =
            chat_tool_approval_request(&call, Some("Install the requested TheRock build."))
                .expect("approval should be built");
        assert_eq!(approval.pending_title, "Install ROCm");
        assert_eq!(
            approval.args,
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                "release".to_owned(),
                "--format".to_owned(),
                "wheel".to_owned(),
                "--prefix".to_owned(),
                "D:\\ROCm\\therock_venvs".to_owned(),
                "--build-date".to_owned(),
                "06052026".to_owned(),
                "--approve-replacing-active-default".to_owned(),
            ]
        );
    }

    #[test]
    fn chat_tool_call_rejects_mutating_install_without_user_folder() {
        let structured = providers::ChatToolCall {
            id: Some("call-missing-prefix".to_owned()),
            name: "install_sdk".to_owned(),
            arguments: serde_json::json!({
                "channel": "release",
                "format": "wheel"
            }),
        };
        let error = validate_chat_tool_call(&structured)
            .unwrap_err()
            .to_string();
        assert!(error.contains("ask the user"), "unexpected error: {error}");

        let command = providers::ChatToolCall {
            id: Some("call-command-missing-prefix".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["install", "sdk", "--channel", "release", "--format", "wheel"],
                "reason": "Install ROCm."
            }),
        };
        let error = validate_chat_tool_call(&command).unwrap_err().to_string();
        assert!(error.contains("ask the user"), "unexpected error: {error}");
    }

    #[test]
    fn chat_tool_call_service_and_watcher_changes_map_to_reviewable_rocm_commands() {
        let stop = providers::ChatToolCall {
            id: Some("call-stop".to_owned()),
            name: "stop_server".to_owned(),
            arguments: serde_json::json!({ "service_id": "svc-qwen" }),
        };
        validate_chat_tool_call(&stop).expect("stop request should validate for review");
        assert_eq!(
            rocm_chat_tool_requested_command(&stop).as_deref(),
            Some("rocm services stop svc-qwen --yes")
        );
        let approval =
            chat_tool_approval_request(&stop, Some("This server is using memory we need."))
                .expect("stop approval should be built");
        assert_eq!(approval.pending_title, "Stop local model server");
        assert_eq!(approval.command_title, "Services");
        assert_eq!(
            approval.args,
            vec![
                "services".to_owned(),
                "stop".to_owned(),
                "svc-qwen".to_owned(),
                "--yes".to_owned(),
            ]
        );
        assert_eq!(
            approval.explanation.as_deref(),
            Some("This server is using memory we need.")
        );

        let enable = providers::ChatToolCall {
            id: Some("call-watch".to_owned()),
            name: "watcher_enable".to_owned(),
            arguments: serde_json::json!({
                "watcher": "server-recover",
                "mode": "propose"
            }),
        };
        validate_chat_tool_call(&enable).expect("watcher enable should validate for review");
        assert_eq!(
            rocm_chat_tool_requested_command(&enable).as_deref(),
            Some("rocm automations enable server-recover --mode propose")
        );
        let approval =
            chat_tool_approval_request(&enable, Some("Recovering failed servers would help."))
                .expect("watcher approval should be built");
        assert_eq!(approval.pending_title, "Enable automation");
        assert_eq!(approval.command_title, "Automations");
        assert_eq!(
            approval.args,
            vec![
                "automations".to_owned(),
                "enable".to_owned(),
                "server-recover".to_owned(),
                "--mode".to_owned(),
                "propose".to_owned(),
            ]
        );

        let disable = providers::ChatToolCall {
            id: Some("call-disable".to_owned()),
            name: "watcher_disable".to_owned(),
            arguments: serde_json::json!({ "watcher": "server-recover" }),
        };
        validate_chat_tool_call(&disable).expect("watcher disable should validate for review");
        assert_eq!(
            rocm_chat_tool_requested_command(&disable).as_deref(),
            Some("rocm automations disable server-recover")
        );
    }

    #[test]
    fn proposal_action_rejects_over_long_proposal_id() {
        let call = providers::ChatToolCall {
            id: None,
            name: "proposal_action".to_owned(),
            arguments: serde_json::json!({
                "proposal_id": "p".repeat(129),
                "action": "show"
            }),
        };
        let err = validate_chat_proposal_action_tool_call(&call)
            .expect_err("over-long proposal_id must be rejected");
        assert!(err.to_string().contains("proposal_id too long"));
    }

    #[test]
    fn chat_tool_call_accepts_expanded_read_only_bridge_tools() {
        for call in [
            providers::ChatToolCall {
                id: None,
                name: "bridge_snapshot".to_owned(),
                arguments: serde_json::json!({}),
            },
            providers::ChatToolCall {
                id: None,
                name: "service_logs".to_owned(),
                arguments: serde_json::json!({
                    "service_id": "svc-qwen",
                    "lines": 120
                }),
            },
            providers::ChatToolCall {
                id: None,
                name: "automations".to_owned(),
                arguments: serde_json::json!({ "event_limit": 12 }),
            },
            providers::ChatToolCall {
                id: None,
                name: "natural_language_plan".to_owned(),
                arguments: serde_json::json!({ "request": "check whether ROCm needs an update" }),
            },
            providers::ChatToolCall {
                id: None,
                name: "port_status".to_owned(),
                arguments: serde_json::json!({ "host": "127.0.0.1", "port": 8188 }),
            },
            providers::ChatToolCall {
                id: None,
                name: "update_check".to_owned(),
                arguments: serde_json::json!({}),
            },
        ] {
            validate_chat_tool_call(&call).expect("read-only bridge tool should validate");
            assert!(
                chat_tool_call_is_read_only(&call),
                "{} should be read-only",
                call.name
            );
        }
    }

    #[test]
    fn local_assistant_prompt_instructions_cover_core_support_questions() {
        let prompt = rocm_chat_tool_system_prompt();
        for expected in [
            "is TheRock installed",
            "which GPU is on this machine",
            "active_runtime_status=ready",
            "legacy_rocm_status=not_detected",
            "[\"model\"]",
            "--build-date",
            "always let the user choose the install folder",
            "--prefix",
            "do not invent a hidden default folder",
            "config",
            "comfyui",
            "First-time setup is the same thing as bootstrap",
            "vllm",
            "Qwen3-4B-Instruct-2507-GGUF",
            "fixed to qwen",
            "served by Lemonade",
            "port_status",
            "[\"services\",\"list\",\"--all\"]",
            "qwen-smoke",
            "Do not invent shell commands",
            "ROCm CLI Assistant Skill",
            "Treat `localhost` and `127.0.0.1` as the same loopback endpoint",
        ] {
            assert!(
                prompt.contains(expected),
                "system prompt should mention {expected}"
            );
        }
    }

    /// The reported bug: asked "What can ROCm do on Windows?", the assistant
    /// answered that ROCm is Windows-incompatible and suggested CUDA/DirectX —
    /// because nothing ever told it which machine it was on. The prompt the CLI
    /// actually sends must carry the host.
    #[test]
    fn assistant_prompt_states_the_host_it_is_answering_for() {
        let prompt = rocm_chat_tool_system_prompt_for_host(None);

        // The tool-use rules survive the composition (this is the same prompt,
        // grounded — not a replacement for it).
        assert!(
            prompt.contains("You are ROCm CLI's local assistant"),
            "the ROCm tool-use prompt must still be there:\n{prompt}"
        );

        // …and it now names this machine's OS and GPU state.
        let expected_os = if cfg!(windows) {
            "- Operating system: Windows"
        } else {
            "- Operating system: Linux"
        };
        assert!(
            prompt.contains(expected_os),
            "the prompt must state this machine's operating system:\n{prompt}"
        );
        assert!(
            prompt.contains("- AMD GPU: "),
            "the prompt must state what GPU was detected (or that none was):\n{prompt}"
        );
        assert!(
            prompt.contains("Never tell the user ROCm is unavailable on their platform"),
            "the prompt must refuse the reported answer:\n{prompt}"
        );
    }

    /// The static prompt asserted two Windows-only facts at every host, which is
    /// how a WSL user — where vLLM IS the supported path — was told to go use
    /// WSL. Platform claims now come from the detected facts instead.
    #[test]
    fn assistant_prompt_makes_no_unconditional_windows_claims() {
        let prompt = rocm_chat_tool_system_prompt();
        assert!(
            !prompt.contains("non-technical Windows users"),
            "the audience is not assumed to be on Windows:\n{prompt}"
        );
        assert!(
            !prompt.contains("On native Windows, vLLM is skipped"),
            "the vLLM caveat belongs in the host facts, not the static prompt:\n{prompt}"
        );

        // Stated only where it is true: present on Windows, absent elsewhere.
        let grounded = rocm_chat_tool_system_prompt_for_host(None);
        assert_eq!(
            grounded.contains("vLLM is skipped on native Windows"),
            cfg!(windows),
            "the vLLM caveat must track the host:\n{grounded}"
        );
    }

    /// The prompt tells the model to "use examine … before answering". The dash
    /// registers its machine check as `doctor`, so before the alias that
    /// sentence named a tool absent from the dash's schema.
    #[test]
    fn every_tool_the_prompt_names_exists_in_the_dash_schema() {
        let prompt = rocm_chat_tool_system_prompt();
        for named in [
            "examine",
            "gpu_snapshot",
            "port_status",
            "natural_language_plan",
        ] {
            assert!(prompt.contains(named), "prompt should mention {named}");
            assert!(
                rocm_dash_tui::agent::ROCM_READ_TOOL_NAMES.contains(&named),
                "the prompt names `{named}` but the dash never registers it, so a \
                 model that obeys the prompt calls a tool the schema does not offer"
            );
        }
    }

    #[test]
    fn deterministic_rocm_tool_summary_interprets_managed_runtime_as_installed() {
        let summary = deterministic_rocm_tool_summary(
            "\
examine:
  driver_detail: AMD Radeon RX 9070 XT driver 32.0.23033.1002
  legacy_rocm_status: not_detected
runtime_state:
  active_runtime_status: ready
  active_runtime_root: D:\\ROCm\\therock_venvs
  active_runtime_pip_cache_dir: D:\\ROCm\\therock_venvs\\pip-cache
  active_runtime_version: 7.13.0a20260511 (build 2026-05-11)
  active_runtime_family: gfx120X-all
",
        )
        .expect("examine output should summarize");

        assert!(summary.contains("GPU: AMD Radeon RX 9070 XT driver 32.0.23033.1002"));
        assert!(summary.contains("ROCm/TheRock: installed and active for ROCm CLI"));
        assert!(summary.contains("gfx120X-all"));
        assert!(summary.contains(r"Install folder: D:\ROCm\therock_venvs"));
        assert!(summary.contains(r"Downloads/cache: D:\ROCm\therock_venvs\pip-cache"));
        assert!(summary.contains("no global legacy ROCm install was found"));
    }

    #[test]
    fn fallback_tool_call_routes_where_installed_to_read_only_examine() {
        for prompt in [
            "where is rocm installed?",
            "where is TheRock installed?",
            "what is the ROCm install folder?",
            "where did rocm install to?",
        ] {
            let call = fallback_rocm_tool_call_for_prompt(prompt).unwrap();
            assert_eq!(call.name, "examine", "{prompt}");
            assert!(chat_tool_call_is_read_only(&call), "{prompt}");
        }
    }

    #[test]
    fn deterministic_rocm_tool_summary_suppresses_extra_local_model_follow_up() {
        let tool_result = ChatToolRunResult {
            approval: None,
            follow_up_text: "\
examine:
  legacy_rocm_status: not_detected
runtime_state:
  active_runtime_status: ready
"
            .to_owned(),
            ran_read_only_tool: true,
            read_only_tool_error: false,
            needs_install_folder: false,
        };
        let summary = deterministic_rocm_tool_summary(&tool_result.follow_up_text);

        assert!(summary.is_some());
        assert!(!should_request_local_tool_follow_up(
            "local",
            &tool_result,
            summary.as_deref()
        ));

        let mut model_list_result = tool_result;
        model_list_result.follow_up_text = "rocm_command:\nmodel recipes\n  qwen\n".to_owned();
        assert!(should_request_local_tool_follow_up(
            "local",
            &model_list_result,
            None
        ));
    }

    #[test]
    fn deterministic_model_tool_summary_identifies_low_vram_assistant() {
        let summary = deterministic_model_tool_summary(
            "\
rocm_command:
model recipes
  Qwen3-4B-Instruct-2507-GGUF aliases=[qwen, lemonade-qwen] task=chat dtype=gguf device=gpu_required min_gpu_mem=4 GiB engines=[lemonade]
      engine_support:
        lemonade: available path=D:\\rocm\\rocm-engine-lemonade.exe
      warning: recommended Lemonade GGUF assistant for ROCm machines
  Qwen3-0.6B-GGUF aliases=[qwen-smoke, lemonade-tiny] task=chat dtype=gguf device=gpu_required min_gpu_mem=2 GiB engines=[lemonade]
      engine_support:
        lemonade: available path=D:\\rocm\\rocm-engine-lemonade.exe
      warning: tiny Lemonade GGUF smoke-test model; not the default assistant
  Qwen/Qwen2.5-0.5B-Instruct aliases=[qwen-tiny] task=chat dtype=float16 device=gpu_required min_gpu_mem=4 GiB engines=[lemonade]
      engine_support:
        lemonade: available path=D:\\rocm\\rocm-engine-lemonade.exe
  Qwen/Qwen3.5-4B aliases=[qwen3.5] task=chat dtype=bfloat16 device=gpu_preferred min_gpu_mem=12 GiB engines=[vllm]
      engine_support:
        vllm: adapter_available path=D:\\rocm\\rocm-engine-vllm.exe runtime_status=unsupported_native_windows reason=native Windows skipped; use WSL/Linux vLLM ROCm
  meta-llama/Llama-3.2-3B-Instruct aliases=[llama] task=chat dtype=bfloat16 device=gpu_preferred min_gpu_mem=8 GiB engines=[lemonade, vllm]
      engine_support:
        lemonade: available path=D:\\rocm\\rocm-engine-lemonade.exe
        vllm: available path=D:\\rocm\\rocm-engine-vllm.exe
",
        )
        .expect("model output should summarize");

        assert!(summary.contains("Recommended local assistant: qwen"));
        assert!(summary.contains("Qwen3-4B-Instruct-2507-GGUF"));
        assert!(summary.contains("4 GiB"));
        assert!(summary.contains("Tiny smoke test: qwen-smoke"));
        assert!(summary.contains("Qwen3-0.6B-GGUF"));
        assert!(summary.contains("8 GiB-class option: llama"));
        assert!(summary.contains("lemonade, vllm"));
        assert!(summary.contains("Qwen/Qwen3.5-4B asks for 12 GiB"));
        assert!(summary.contains("Native Windows note"));
        assert!(summary.contains("Run `rocm examine`"));
    }

    #[test]
    fn deterministic_model_tool_summary_suppresses_extra_local_model_follow_up() {
        let tool_result = ChatToolRunResult {
            approval: None,
            follow_up_text: "\
rocm_command:
model recipes
  Qwen3-4B-Instruct-2507-GGUF aliases=[qwen] task=chat dtype=gguf device=gpu_required min_gpu_mem=4 GiB engines=[lemonade]
      engine_support:
        lemonade: available path=D:\\rocm\\rocm-engine-lemonade.exe
"
            .to_owned(),
            ran_read_only_tool: true,
            read_only_tool_error: false,
            needs_install_folder: false,
        };
        let summary = deterministic_chat_tool_summary(&tool_result.follow_up_text);

        assert!(summary.is_some());
        assert!(!should_request_local_tool_follow_up(
            "local",
            &tool_result,
            summary.as_deref()
        ));
    }

    #[test]
    fn chat_tool_call_accepts_assistant_support_command_shapes() {
        for (call, expected_command, read_only) in [
            (
                providers::ChatToolCall {
                    id: None,
                    name: "rocm_command".to_owned(),
                    arguments: serde_json::json!({ "args": ["examine"] }),
                },
                Some("rocm examine"),
                true,
            ),
            (
                providers::ChatToolCall {
                    id: None,
                    name: "gpu_snapshot".to_owned(),
                    arguments: serde_json::json!({}),
                },
                None,
                true,
            ),
            (
                providers::ChatToolCall {
                    id: None,
                    name: "rocm_command".to_owned(),
                    arguments: serde_json::json!({ "args": ["model"] }),
                },
                Some("rocm model"),
                true,
            ),
            (
                providers::ChatToolCall {
                    id: None,
                    name: "install_sdk".to_owned(),
                    arguments: serde_json::json!({
                        "channel": "release",
                        "format": "wheel",
                        "prefix": "D:\\ROCm\\therock_venvs"
                    }),
                },
                Some(
                    "rocm install sdk --channel release --format wheel --approve-replacing-active-default --prefix D:\\ROCm\\therock_venvs",
                ),
                false,
            ),
            (
                providers::ChatToolCall {
                    id: None,
                    name: "rocm_command".to_owned(),
                    arguments: serde_json::json!({ "args": ["comfyui", "install"] }),
                },
                Some("rocm comfyui install --yes"),
                false,
            ),
            (
                providers::ChatToolCall {
                    id: None,
                    name: "launch_server".to_owned(),
                    arguments: serde_json::json!({
                        "model": "qwen",
                        "engine": "vllm",
                        "device": "gpu_required"
                    }),
                },
                Some("rocm serve qwen --managed --engine vllm --device gpu_required"),
                false,
            ),
        ] {
            validate_chat_tool_call(&call).expect("assistant support tool should validate");
            assert_eq!(
                chat_tool_call_is_read_only(&call),
                read_only,
                "{}",
                call.name
            );
            if let Some(expected_command) = expected_command {
                assert_eq!(
                    rocm_chat_tool_requested_command(&call).as_deref(),
                    Some(expected_command)
                );
            }
        }
    }

    #[test]
    fn chat_rocm_command_routes_comfyui_and_engine_actions() {
        let comfy_install = providers::ChatToolCall {
            id: Some("call-comfy".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["comfyui", "install"],
                "reason": "The user asked me to install ComfyUI."
            }),
        };
        validate_chat_tool_call(&comfy_install).expect("ComfyUI install should validate");
        assert!(!chat_tool_call_is_read_only(&comfy_install));
        assert_eq!(
            rocm_chat_tool_requested_command(&comfy_install).as_deref(),
            Some("rocm comfyui install --yes")
        );
        let approval = chat_tool_approval_request(&comfy_install, Some("Install ComfyUI now."))
            .expect("approval should be built");
        assert_eq!(approval.pending_title, "Install ComfyUI");
        assert_eq!(approval.command_title, "ComfyUI");
        assert_eq!(
            approval.args,
            vec![
                "comfyui".to_owned(),
                "install".to_owned(),
                "--yes".to_owned()
            ]
        );

        let lemonade = providers::ChatToolCall {
            id: Some("call-lemonade".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["engines", "install", "lemonade"]
            }),
        };
        validate_chat_tool_call(&lemonade).expect("lemonade engine install should validate");
        assert!(!chat_tool_call_is_read_only(&lemonade));
        assert_eq!(
            rocm_chat_tool_requested_command(&lemonade).as_deref(),
            Some("rocm engines install lemonade")
        );
        let approval =
            chat_tool_approval_request(&lemonade, Some("Install Lemonade for local serving."))
                .expect("approval should be built");
        assert_eq!(approval.pending_title, "Install engine");
        assert_eq!(approval.command_title, "Engine");

        let vllm = providers::ChatToolCall {
            id: Some("call-vllm".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["engines", "install", "vllm"]
            }),
        };
        validate_chat_tool_call(&vllm).expect("vLLM engine install should validate");
        assert!(!chat_tool_call_is_read_only(&vllm));
        assert_eq!(
            rocm_chat_tool_requested_command(&vllm).as_deref(),
            Some("rocm engines install vllm")
        );
        let approval = chat_tool_approval_request(&vllm, Some("Install vLLM for Linux/WSL."))
            .expect("approval should be built");
        assert_eq!(approval.pending_title, "Install engine");
        assert_eq!(approval.command_title, "Engine");

        let comfy_start = providers::ChatToolCall {
            id: Some("call-comfy-start".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["comfyui", "start"]
            }),
        };
        validate_chat_tool_call(&comfy_start).expect("ComfyUI start should validate");
        assert!(!chat_tool_call_is_read_only(&comfy_start));
        let approval = chat_tool_approval_request(&comfy_start, Some("Start ComfyUI locally."))
            .expect("approval should be built");
        assert_eq!(approval.pending_title, "Start ComfyUI");
        assert_eq!(approval.command_title, "ComfyUI");

        let serve = providers::ChatToolCall {
            id: Some("call-serve".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["serve", "qwen", "--engine", "vllm", "--device", "gpu_required", "--managed"]
            }),
        };
        validate_chat_tool_call(&serve).expect("managed serve should validate");
        assert!(!chat_tool_call_is_read_only(&serve));
        assert_eq!(
            rocm_chat_tool_requested_command(&serve).as_deref(),
            Some("rocm serve qwen --engine vllm --device gpu_required --managed")
        );
        let approval = chat_tool_approval_request(&serve, Some("Start the recommended assistant."))
            .expect("approval should be built");
        assert_eq!(approval.pending_title, "Start local model server");
        assert_eq!(approval.command_title, "Serve");

        let vllm_serve = providers::ChatToolCall {
            id: Some("call-vllm-serve".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["serve", "Qwen/Qwen3.5-4B", "--engine", "vllm", "--device", "gpu_required", "--managed"]
            }),
        };
        validate_chat_tool_call(&vllm_serve).expect("managed vLLM serve should validate");
        assert!(!chat_tool_call_is_read_only(&vllm_serve));
        assert_eq!(
            rocm_chat_tool_requested_command(&vllm_serve).as_deref(),
            Some("rocm serve Qwen/Qwen3.5-4B --engine vllm --device gpu_required --managed")
        );

        let config = providers::ChatToolCall {
            id: Some("call-config".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["config", "set-default-engine", "vllm"]
            }),
        };
        validate_chat_tool_call(&config).expect("config change should validate");
        assert!(!chat_tool_call_is_read_only(&config));
        let approval = chat_tool_approval_request(&config, Some("Use vLLM as the default engine."))
            .expect("approval should be built");
        assert_eq!(approval.pending_title, "Change settings");
        assert_eq!(approval.command_title, "Config");
    }

    #[test]
    fn setup_status_is_read_only() {
        for args in [
            vec!["setup".to_owned()],
            vec!["setup".to_owned(), "status".to_owned()],
        ] {
            let action =
                chat_rocm_command_action_from_args(args.clone()).expect("setup status classifies");
            assert!(
                matches!(action, ChatRocmCommandAction::ReadOnly(_)),
                "setup {args:?} should be read-only, got {action:?}"
            );
        }
    }

    #[test]
    fn storage_report_is_read_only_and_removal_requires_approval() {
        for args in [
            vec!["storage".to_owned()],
            vec!["storage".to_owned(), "report".to_owned()],
        ] {
            let action = chat_rocm_command_action_from_args(args.clone())
                .expect("storage report classifies");
            assert!(
                matches!(action, ChatRocmCommandAction::ReadOnly(_)),
                "storage {args:?} only measures folders, so it should be read-only, got {action:?}"
            );
        }

        for verb in ["remove-old-installs", "remove-downloads"] {
            let action =
                chat_rocm_command_action_from_args(vec!["storage".to_owned(), verb.to_owned()])
                    .expect("storage removal classifies");
            match action {
                ChatRocmCommandAction::Approval { args, .. } => {
                    assert!(
                        args.iter().any(|arg| arg == "--yes"),
                        "approved removal runs non-interactively: {args:?}"
                    );
                }
                other @ ChatRocmCommandAction::ReadOnly(_) => {
                    panic!("storage {verb} must require approval, got {other:?}")
                }
            }
        }
    }

    #[test]
    fn setup_reset_requires_approval() {
        let action =
            chat_rocm_command_action_from_args(vec!["setup".to_owned(), "reset".to_owned()])
                .expect("setup reset classifies");
        match action {
            ChatRocmCommandAction::Approval {
                pending_title,
                command_title,
                ..
            } => {
                assert_eq!(pending_title, "Reset first-time setup");
                assert_eq!(command_title, "Setup");
            }
            other @ ChatRocmCommandAction::ReadOnly(_) => {
                panic!("setup reset should require approval, got {other:?}")
            }
        }
    }

    #[test]
    fn proposal_action_show_is_read_only() {
        let call = providers::ChatToolCall {
            id: None,
            name: "proposal_action".to_owned(),
            arguments: serde_json::json!({ "proposal_id": "p1", "action": "show" }),
        };
        validate_chat_tool_call(&call).expect("show validates");
        assert!(
            chat_tool_call_is_read_only(&call),
            "proposal_action show must be read-only"
        );
    }

    #[test]
    fn proposal_action_approve_requires_approval() {
        for action in ["approve", "reject"] {
            let call = providers::ChatToolCall {
                id: None,
                name: "proposal_action".to_owned(),
                arguments: serde_json::json!({ "proposal_id": "p1", "action": action }),
            };
            validate_chat_tool_call(&call).expect("approve/reject validates");
            assert!(
                !chat_tool_call_is_read_only(&call),
                "proposal_action {action} must NOT be read-only"
            );
            let req = chat_tool_approval_request(&call, None).unwrap_or_else(|err| {
                panic!("proposal_action {action} should need approval: {err}")
            });
            assert_eq!(req.command_title, "Reviews");
            assert!(
                req.pending_title.contains("proposal") || req.pending_title.contains("Proposal")
            );
            assert!(
                req.display_command
                    .as_deref()
                    .unwrap_or_default()
                    .contains("p1"),
                "display command should show the proposal id"
            );
        }
    }

    #[test]
    fn proposal_action_rejects_unknown_action() {
        let call = providers::ChatToolCall {
            id: None,
            name: "proposal_action".to_owned(),
            arguments: serde_json::json!({ "proposal_id": "p1", "action": "delete" }),
        };
        assert!(
            validate_chat_tool_call(&call).is_err(),
            "unknown proposal_action `action` must be rejected"
        );
    }

    #[test]
    fn proposal_action_approve_updates_status() {
        let (root, paths) = test_paths("proposal-approve");
        // Seed a pending proposal.
        let proposal = rocm_core::AutomationProposalRecord {
            at_unix_ms: rocm_core::unix_time_millis(),
            proposal_id: "prop-approve-1".to_owned(),
            watcher_id: "therock-update".to_owned(),
            action: "prepare_driver_plan".to_owned(),
            title: "Apply driver plan".to_owned(),
            message: "A reviewed driver plan is ready.".to_owned(),
            status: "pending".to_owned(),
            service_id: None,
            tool: None,
            arguments: serde_json::Value::Null,
            reviewed_at_unix_ms: None,
        };
        rocm_core::append_automation_proposal(&paths, &proposal).expect("seed proposal");

        // show is read-only and returns the proposal.
        let shown = run_internal_mcp_call(
            &paths,
            "proposal_action",
            serde_json::json!({ "proposal_id": "prop-approve-1", "action": "show" }),
            false,
        )
        .expect("show ok");
        assert_eq!(shown["structuredContent"]["status"], "pending");

        // approve requires allow_mutation.
        assert!(
            run_internal_mcp_call(
                &paths,
                "proposal_action",
                serde_json::json!({ "proposal_id": "prop-approve-1", "action": "approve" }),
                false,
            )
            .is_err(),
            "approve without allow_mutation must bail"
        );

        // approve with allow_mutation sets status to approved.
        let approved = run_internal_mcp_call(
            &paths,
            "proposal_action",
            serde_json::json!({ "proposal_id": "prop-approve-1", "action": "approve" }),
            true,
        )
        .expect("approve ok");
        assert_eq!(approved["structuredContent"]["status"], "approved");
        let stored = rocm_core::find_automation_proposal(&paths, "prop-approve-1")
            .expect("proposal still present");
        assert_eq!(stored.status, "approved");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn proposal_action_reject_updates_status() {
        let (root, paths) = test_paths("proposal-reject");
        let proposal = rocm_core::AutomationProposalRecord {
            at_unix_ms: rocm_core::unix_time_millis(),
            proposal_id: "prop-reject-1".to_owned(),
            watcher_id: "server-recover".to_owned(),
            action: "queue_stop_server_proposal".to_owned(),
            title: "Stop overheating server".to_owned(),
            message: "GPU thermal pressure detected.".to_owned(),
            status: "pending".to_owned(),
            service_id: None,
            tool: None,
            arguments: serde_json::Value::Null,
            reviewed_at_unix_ms: None,
        };
        rocm_core::append_automation_proposal(&paths, &proposal).expect("seed proposal");

        let rejected = run_internal_mcp_call(
            &paths,
            "proposal_action",
            serde_json::json!({ "proposal_id": "prop-reject-1", "action": "reject" }),
            true,
        )
        .expect("reject ok");
        assert_eq!(rejected["structuredContent"]["status"], "rejected");
        let stored = rocm_core::find_automation_proposal(&paths, "prop-reject-1")
            .expect("proposal still present");
        assert_eq!(stored.status, "rejected");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn proposal_action_show_missing_proposal_errors() {
        let (root, paths) = test_paths("proposal-missing");
        assert!(
            run_internal_mcp_call(
                &paths,
                "proposal_action",
                serde_json::json!({ "proposal_id": "nope", "action": "show" }),
                false,
            )
            .is_err(),
            "showing a missing proposal must error"
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn config_set_permissions_classifies_as_approval() {
        // Permission escalation MUST route through approval — the classifier
        // routes any `config <sub>` (catch-all) to Approval. Verify both modes.
        for mode in ["full_access", "ask"] {
            let action = chat_rocm_command_action_from_args(vec![
                "config".to_owned(),
                "set-permissions".to_owned(),
                mode.to_owned(),
            ])
            .expect("config set-permissions classifies");
            match action {
                ChatRocmCommandAction::Approval { command_title, .. } => {
                    assert_eq!(command_title, "Config");
                }
                other @ ChatRocmCommandAction::ReadOnly(_) => {
                    panic!("config set-permissions {mode} must need approval, got {other:?}")
                }
            }
        }
    }

    #[test]
    fn watcher_validator_rejects_unknown_and_invalid_mode() {
        // Unknown watcher id → rejected.
        let unknown = providers::ChatToolCall {
            id: None,
            name: "watcher_enable".to_owned(),
            arguments: serde_json::json!({ "watcher": "no-such-watcher" }),
        };
        assert!(
            validate_chat_watcher_tool_call(&unknown, true).is_err(),
            "unknown watcher must be rejected"
        );
        // Invalid mode → rejected.
        let bad_mode = providers::ChatToolCall {
            id: None,
            name: "watcher_enable".to_owned(),
            arguments: serde_json::json!({ "watcher": "therock-update", "mode": "rampage" }),
        };
        assert!(
            validate_chat_watcher_tool_call(&bad_mode, true).is_err(),
            "invalid watcher mode must be rejected"
        );
        // Valid watcher + valid mode → accepted.
        let ok = providers::ChatToolCall {
            id: None,
            name: "watcher_enable".to_owned(),
            arguments: serde_json::json!({ "watcher": "therock-update", "mode": "observe" }),
        };
        validate_chat_watcher_tool_call(&ok, true).expect("valid watcher+mode accepted");
        // Disable must reject a `mode`.
        let disable_with_mode = providers::ChatToolCall {
            id: None,
            name: "watcher_disable".to_owned(),
            arguments: serde_json::json!({ "watcher": "therock-update", "mode": "observe" }),
        };
        assert!(
            validate_chat_watcher_tool_call(&disable_with_mode, false).is_err(),
            "disable must reject a mode argument"
        );
    }

    #[test]
    fn lifecycle_read_mutate_split_is_honest() {
        let read_only = [
            vec!["update".to_owned()],
            vec!["comfyui".to_owned(), "status".to_owned()],
            vec!["comfyui".to_owned(), "logs".to_owned()],
            vec!["uninstall".to_owned(), "--dry-run".to_owned()],
            vec!["setup".to_owned(), "status".to_owned()],
            vec![
                "runtimes".to_owned(),
                "uninstall".to_owned(),
                "old-runtime".to_owned(),
                "--dry-run".to_owned(),
            ],
            vec![
                "runtimes".to_owned(),
                "remove".to_owned(),
                "old-runtime".to_owned(),
                "--dry-run".to_owned(),
            ],
            // Must agree with the daemon's `ensure_rocm_command_is_read_only`.
            // Its comment claims the two are mirrored, and nothing enforced
            // that — the arm was added there and not here, and the classifiers
            // disagreed until a reviewer noticed.
            vec!["remote".to_owned(), "targets".to_owned()],
            vec![
                "remote".to_owned(),
                "doctor".to_owned(),
                "gpu-box".to_owned(),
            ],
            vec!["remote".to_owned(), "status".to_owned()],
        ];
        for args in read_only {
            let action = chat_rocm_command_action_from_args(args.clone())
                .unwrap_or_else(|err| panic!("{args:?} should classify: {err}"));
            assert!(
                matches!(action, ChatRocmCommandAction::ReadOnly(_)),
                "{args:?} should be read-only, got {action:?}"
            );
        }

        // Paired with whether the command has a `--yes` to inject. The flag
        // exists to keep a consent prompt from hanging a null-stdin spawn, so
        // the demand only makes sense for commands that would prompt — and
        // injecting it where clap defines no such flag would make the spawn fail
        // to parse rather than succeed unattended.
        let mutating = [
            (vec!["update".to_owned(), "--apply".to_owned()], true),
            // These start, publish or tear down on another machine. They take no
            // `--yes` because they never prompt: consent is the approval step
            // itself, and every destructive choice they make is already settled
            // by an explicit flag (`remote stop --force`). Give any of them an
            // interactive prompt and it needs a consent flag here too.
            (
                vec![
                    "remote".to_owned(),
                    "serve".to_owned(),
                    "gpu-box".to_owned(),
                    "m".to_owned(),
                ],
                false,
            ),
            (
                vec!["remote".to_owned(), "attach".to_owned(), "sess".to_owned()],
                false,
            ),
            (
                vec!["remote".to_owned(), "stop".to_owned(), "sess".to_owned()],
                false,
            ),
            (vec!["comfyui".to_owned(), "install".to_owned()], true),
            (vec!["comfyui".to_owned(), "start".to_owned()], true),
            (vec!["comfyui".to_owned(), "stop".to_owned()], true),
            (vec!["uninstall".to_owned()], true),
            (vec!["setup".to_owned(), "reset".to_owned()], true),
            (
                vec![
                    "runtimes".to_owned(),
                    "uninstall".to_owned(),
                    "old-runtime".to_owned(),
                ],
                true,
            ),
            (
                vec![
                    "runtimes".to_owned(),
                    "remove".to_owned(),
                    "old-runtime".to_owned(),
                ],
                true,
            ),
        ];
        for (args, expects_yes) in mutating {
            let action = chat_rocm_command_action_from_args(args.clone())
                .unwrap_or_else(|err| panic!("{args:?} should classify: {err}"));
            match &action {
                ChatRocmCommandAction::Approval { args, .. } => {
                    assert_eq!(
                        args.iter().any(|arg| arg == "--yes"),
                        expects_yes,
                        "{args:?} disagrees with whether the approval path should \
                         carry --yes"
                    );
                }
                other @ ChatRocmCommandAction::ReadOnly(_) => {
                    panic!("{args:?} should require approval, got {other:?}")
                }
            }
        }
    }

    #[test]
    fn install_sdk_chat_and_mcp_args_approve_the_replacement_for_a_non_interactive_spawn() {
        // The chat/MCP surfaces spawn `rocm` with null stdin, so the consent
        // prompt would refuse with "re-run with
        // `--approve-replacing-active-default`" — a flag the user has no way to
        // supply from chat or the dashboard. Both the chat classifier
        // arm and the MCP tool-args builder must inject the consent so an
        // install over the active default runtime is not silently refused.
        let action = chat_rocm_command_action_from_args(vec![
            "install".to_owned(),
            "sdk".to_owned(),
            "--channel".to_owned(),
            "release".to_owned(),
            // The classifier requires a user-chosen (non-system) install folder.
            "--prefix".to_owned(),
            "/home/tester/rocm-managed".to_owned(),
        ])
        .expect("install sdk classifies");
        let chat_args = match action {
            ChatRocmCommandAction::Approval { args, .. } => args,
            other @ ChatRocmCommandAction::ReadOnly(_) => {
                panic!("install sdk must require approval, got {other:?}")
            }
        };

        // The MCP `install_sdk` tool builds its own argv (it does not route
        // through the classifier above), so it must add the flag independently.
        let call = providers::ChatToolCall {
            id: None,
            name: "install_sdk".to_owned(),
            arguments: serde_json::json!({ "channel": "release", "format": "wheel" }),
        };
        let mcp_args = rocm_chat_tool_requested_args(&call).expect("install_sdk tool builds args");

        for args in [&chat_args, &mcp_args] {
            assert_eq!(
                consents_for_install_sdk_argv(args).replace_active_default,
                therock::SdkInstallConsent::Preapproved(
                    therock::SdkInstallApprovalSource::ApproveReplacingActiveDefault
                ),
                "the spawn must approve replacing the active default, and be credited \
                 to the flag it actually passed rather than to --yes, got {args:?}"
            );
        }
    }

    #[test]
    fn chat_rocm_command_runs_read_only_and_rejects_risky_shapes() {
        let status = providers::ChatToolCall {
            id: None,
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["rocm", "comfy", "status"]
            }),
        };
        validate_chat_tool_call(&status).expect("ComfyUI status should validate");
        assert!(chat_tool_call_is_read_only(&status));
        assert_eq!(
            rocm_chat_tool_requested_command(&status).as_deref(),
            Some("rocm comfyui status")
        );

        let logs = providers::ChatToolCall {
            id: None,
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["comfyui", "logs"]
            }),
        };
        validate_chat_tool_call(&logs).expect("ComfyUI logs should validate");
        assert!(chat_tool_call_is_read_only(&logs));
        assert_eq!(
            rocm_chat_tool_requested_command(&logs).as_deref(),
            Some("rocm comfyui logs")
        );

        let cpu = providers::ChatToolCall {
            id: None,
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["serve", "tiny.gguf", "--engine", "vllm", "--device", "cpu"]
            }),
        };
        let error = validate_chat_tool_call(&cpu).unwrap_err().to_string();
        assert!(error.contains("CPU execution"));

        let public_flag = providers::ChatToolCall {
            id: None,
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["serve", "tiny.gguf", "--engine", "vllm", "--allow-public-bind", "--managed"]
            }),
        };
        let error = validate_chat_tool_call(&public_flag)
            .unwrap_err()
            .to_string();
        assert!(error.contains("public network binding"));

        let foreground = providers::ChatToolCall {
            id: None,
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["serve", "qwen", "--engine", "vllm", "--foreground"]
            }),
        };
        let error = validate_chat_tool_call(&foreground)
            .unwrap_err()
            .to_string();
        assert!(error.contains("--managed"));

        let shell = providers::ChatToolCall {
            id: None,
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({
                "args": ["powershell", "-Command", "whoami"]
            }),
        };
        let error = validate_chat_tool_call(&shell).unwrap_err().to_string();
        assert!(error.contains("unsupported rocm command"));
    }

    #[test]
    fn assistant_read_only_rocm_commands_do_not_fallback_to_child_process() {
        let (_root, paths) = test_paths("readonly-rocm-in-process-only");
        let args = vec!["services".to_owned(), "status".to_owned()];

        let error = run_rocm_command_for_paths(&paths, &args, Duration::from_secs(1))
            .unwrap_err()
            .to_string();

        assert!(error.contains("read-only assistant command is not implemented in-process"));
        assert!(error.contains("rocm services status"));
        assert!(!paths.data_dir.join("logs").exists());
    }

    #[test]
    fn internal_mcp_read_only_rocm_command_runs_in_process() {
        let (_root, paths) = test_paths("mcp-readonly-rocm-in-process");

        let result = run_internal_mcp_call(
            &paths,
            "rocm_command",
            serde_json::json!({ "args": ["version"] }),
            false,
        )
        .expect("read-only rocm mcp-call should run");

        assert_eq!(
            result.get("isError").and_then(serde_json::Value::as_bool),
            Some(false)
        );
        assert_eq!(
            result
                .pointer("/structuredContent/argv/0")
                .and_then(serde_json::Value::as_str),
            Some("rocm")
        );
        assert!(mcp_tool_result_text(&result).contains(env!("CARGO_PKG_VERSION")));
    }

    #[test]
    fn natural_language_plan_returns_structured_mutating_action() {
        let (_root, paths) = test_paths("nl-plan-structured-mutating");

        let result = run_internal_mcp_call(
            &paths,
            "natural_language_plan",
            serde_json::json!({
                "request": "install TheRock into /opt/rocm-target"
            }),
            false,
        )
        .expect("natural_language_plan should plan the request");

        assert_eq!(
            result.get("isError").and_then(serde_json::Value::as_bool),
            Some(false)
        );
        // text (rendered plan) preserved.
        assert!(
            !result
                .pointer("/structuredContent/text")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .trim()
                .is_empty()
        );
        // A known install folder yields a mutating, placeholder-free action.
        assert_eq!(
            result
                .pointer("/structuredContent/action/approval_required")
                .and_then(serde_json::Value::as_bool),
            Some(true)
        );
        assert_eq!(
            result
                .pointer("/structuredContent/action/has_placeholders")
                .and_then(serde_json::Value::as_bool),
            Some(false)
        );
        let args = result
            .pointer("/structuredContent/action/args")
            .and_then(serde_json::Value::as_array)
            .expect("action args should be present");
        assert!(args.iter().any(|arg| arg.as_str() == Some("install")));
        assert!(args.iter().any(|arg| arg.as_str() == Some("--prefix")));
    }

    #[test]
    fn natural_language_plan_returns_placeholder_action_when_incomplete() {
        let (_root, paths) = test_paths("nl-plan-structured-placeholder");

        let result = run_internal_mcp_call(
            &paths,
            "natural_language_plan",
            serde_json::json!({ "request": "serve" }),
            false,
        )
        .expect("natural_language_plan should plan the request");

        assert_eq!(
            result
                .pointer("/structuredContent/action/has_placeholders")
                .and_then(serde_json::Value::as_bool),
            Some(true)
        );
    }

    #[test]
    fn setup_status_renders_in_process() {
        let (_root, paths) = test_paths("setup-status-in-process");

        for args in [
            vec!["setup".to_owned()],
            vec!["setup".to_owned(), "status".to_owned()],
        ] {
            let text = run_rocm_read_only_in_process(&paths, &args)
                .expect("setup status read path should render in-process");
            assert!(
                !text.trim().is_empty(),
                "setup status text should be non-empty for {args:?}"
            );
            assert!(
                text.contains("ROCm setup"),
                "setup status text should be reached for {args:?}: {text}"
            );
        }
    }

    #[test]
    fn chat_tool_call_rejects_bad_service_and_watcher_suggestions() {
        for (call, expected) in [
            (
                providers::ChatToolCall {
                    id: None,
                    name: "service_logs".to_owned(),
                    arguments: serde_json::json!({ "service_id": "bad/name" }),
                },
                "must not contain path separators",
            ),
            (
                providers::ChatToolCall {
                    id: None,
                    name: "automations".to_owned(),
                    arguments: serde_json::json!({ "event_limit": 1000 }),
                },
                "between 1 and 64",
            ),
            (
                providers::ChatToolCall {
                    id: None,
                    name: "watcher_enable".to_owned(),
                    arguments: serde_json::json!({
                        "watcher": "unknown",
                        "mode": "propose"
                    }),
                },
                "unknown watcher",
            ),
            (
                providers::ChatToolCall {
                    id: None,
                    name: "watcher_disable".to_owned(),
                    arguments: serde_json::json!({
                        "watcher": "server-recover",
                        "mode": "propose"
                    }),
                },
                "cannot set `mode`",
            ),
        ] {
            let error = validate_chat_tool_call(&call).unwrap_err().to_string();
            assert!(error.contains(expected), "unexpected error: {error}");
        }
    }

    #[test]
    fn chat_tool_call_rejects_cpu_and_public_bind_server_requests() {
        let cpu = providers::ChatToolCall {
            id: None,
            name: "launch_server".to_owned(),
            arguments: serde_json::json!({
                "model": "tiny.gguf",
                "engine": "vllm",
                "device": "cpu"
            }),
        };
        let error = validate_chat_tool_call(&cpu).unwrap_err().to_string();
        assert!(error.contains("CPU execution"));

        let public = providers::ChatToolCall {
            id: None,
            name: "launch_server".to_owned(),
            arguments: serde_json::json!({
                "model": "tiny.gguf",
                "engine": "vllm",
                "host": "0.0.0.0",
                "allow_public_bind": true
            }),
        };
        let error = validate_chat_tool_call(&public).unwrap_err().to_string();
        assert!(error.contains("public network binding"));

        let host = providers::ChatToolCall {
            id: None,
            name: "launch_server".to_owned(),
            arguments: serde_json::json!({
                "model": "tiny.gguf",
                "engine": "vllm",
                "host": "0.0.0.0"
            }),
        };
        let error = validate_chat_tool_call(&host).unwrap_err().to_string();
        assert!(error.contains("non-local host"));
    }

    #[test]
    fn chat_tool_call_rejects_bad_install_suggestions() {
        for (arguments, expected) in [
            (
                serde_json::json!({ "channel": "stable" }),
                "unsupported TheRock channel",
            ),
            (
                serde_json::json!({ "format": "zip" }),
                "unsupported TheRock install format",
            ),
            (
                serde_json::json!({
                    "prefix": if cfg!(windows) { "C:\\Windows\\rocm" } else { "/opt/rocm" }
                }),
                "system install folder",
            ),
            (
                serde_json::json!({
                    "version": "7.13.0a20260605",
                    "build_date": "2026-06-05"
                }),
                "both `version` and `build_date`",
            ),
            (
                serde_json::json!({ "build_date": "not-a-date" }),
                "build date",
            ),
            (
                serde_json::json!({
                    "format": "tarball",
                    "build_date": "2026-06-05"
                }),
                if cfg!(windows) {
                    "tarball` installs on Windows"
                } else {
                    "specific TheRock wheel versions"
                },
            ),
        ] {
            let call = providers::ChatToolCall {
                id: None,
                name: "install_sdk".to_owned(),
                arguments,
            };
            let error = validate_chat_tool_call(&call).unwrap_err().to_string();
            assert!(error.contains(expected), "unexpected error: {error}");
        }
    }

    /// Guard: every read-only tool the dash side registers must have a home in
    /// the bin's accept-list. A name-only tool (the `doctor` regression caught
    /// in review) would otherwise fail end-to-end as "unsupported ROCm tool"
    /// while passing the dash-side, name-against-itself completeness checks.
    ///
    /// We assert the NAME is accepted — not that empty args validate — so the
    /// check stays hermetic (no real `BinToolExecutor::execute` / live I/O) and
    /// independent of each tool's argument schema.
    #[test]
    fn read_tool_names_are_subset_of_bin_accept_list() {
        for name in rocm_dash_tui::agent::ROCM_READ_TOOL_NAMES {
            let call = providers::ChatToolCall {
                id: None,
                name: name.to_owned(),
                arguments: serde_json::json!({}),
            };
            if let Err(error) = validate_chat_tool_call(&call) {
                let message = error.to_string();
                assert!(
                    !message.contains("unsupported ROCm tool"),
                    "ROCM_READ_TOOL_NAMES advertises `{name}` but the bin rejects \
                     the name as unsupported: {message}"
                );
            }
        }
    }

    #[test]
    fn chat_tool_call_refusal_prioritizes_safe_review_wording() -> Result<()> {
        let mut output = String::new();
        let (_root, paths) = test_paths("chat-tool-refusal");
        let response = providers::ChatResponse {
            provider: "local".to_owned(),
            model: "tiny.gguf".to_owned(),
            content: "I can install ROCm.".to_owned(),
            tool_calls: vec![providers::ChatToolCall {
                id: None,
                name: "install_sdk".to_owned(),
                arguments: serde_json::json!({
                    "channel": "release",
                    "format": "wheel",
                    "prefix": "D:\\ROCm\\therock_venvs"
                }),
            }],
        };

        let mut progress = None;
        append_chat_tool_results(
            &paths,
            &response,
            &mut output,
            Some("I can install ROCm."),
            &mut progress,
        )?;

        assert!(output.contains("Install ROCm: needs your review"));
        assert!(!output.contains("install_sdk: approval required"));
        assert!(output.contains("not run: review the approval card before anything runs"));
        assert!(output.contains("advanced manual command: rocm install sdk"));
        assert!(!output.contains("run the shown ROCm command"));
        Ok(())
    }

    #[test]
    fn fallback_tool_call_runs_model_support_checks() {
        let call =
            fallback_rocm_tool_call_for_prompt("Which LLMs can this machine support?").unwrap();
        assert_eq!(call.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&call).unwrap(),
            vec!["model".to_owned()]
        );

        assert!(fallback_rocm_tool_call_for_prompt("Which LLM are you using?").is_none());
    }

    #[test]
    fn local_rocm_tools_chat_uses_fixed_lemonade_qwen_assistant() {
        assert_eq!(
            local_rocm_tools_assistant_model("local", true),
            Some(providers::BUILTIN_ASSISTANT_MODEL_ID)
        );
        assert_eq!(local_rocm_tools_assistant_model("local", false), None);
        assert_eq!(local_rocm_tools_assistant_model("openai", true), None);
    }

    #[test]
    fn fallback_tool_call_routes_running_questions_to_status_tools() {
        let comfy = fallback_rocm_tool_call_for_prompt("Is ComfyUI running?").unwrap();
        assert_eq!(comfy.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&comfy).unwrap(),
            vec!["comfyui".to_owned(), "status".to_owned()]
        );
        assert!(chat_tool_call_is_read_only(&comfy));

        let comfy_port =
            fallback_rocm_tool_call_for_prompt("Is ComfyUI running on port 8188?").unwrap();
        assert_eq!(comfy_port.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&comfy_port).unwrap(),
            vec!["comfyui".to_owned(), "status".to_owned()]
        );
        assert!(chat_tool_call_is_read_only(&comfy_port));

        for prompt in [
            "Is vLLM running?",
            "is lemonade running?",
            "is the model server running?",
            "is qwen running?",
        ] {
            let call = fallback_rocm_tool_call_for_prompt(prompt).unwrap();
            assert_eq!(call.name, "rocm_command", "{prompt}");
            assert_eq!(
                normalized_chat_rocm_command_args(&call).unwrap(),
                vec!["services".to_owned(), "list".to_owned(), "--all".to_owned(),],
                "{prompt}"
            );
            assert!(chat_tool_call_is_read_only(&call), "{prompt}");
        }

        let port = fallback_rocm_tool_call_for_prompt("what is running on port 8188?").unwrap();
        assert_eq!(port.name, "port_status");
        assert_eq!(
            port.arguments,
            serde_json::json!({ "host": DEFAULT_LOCAL_HOST, "port": 8188 })
        );
        assert!(chat_tool_call_is_read_only(&port));
    }

    #[test]
    fn fallback_tool_call_routes_engine_install_state_to_engines_list() {
        for prompt in ["is vLLM installed?", "is Lemonade available?"] {
            let call = fallback_rocm_tool_call_for_prompt(prompt).unwrap();
            assert_eq!(call.name, "rocm_command", "{prompt}");
            assert_eq!(
                normalized_chat_rocm_command_args(&call).unwrap(),
                vec!["engines".to_owned(), "list".to_owned()],
                "{prompt}"
            );
            assert!(chat_tool_call_is_read_only(&call), "{prompt}");
        }
    }

    #[test]
    fn supplemental_tool_call_adds_missing_specific_status_check() {
        let generic_services = providers::ChatToolCall {
            id: Some("model-picked-services".to_owned()),
            name: "services".to_owned(),
            arguments: serde_json::json!({}),
        };

        let call =
            supplemental_read_only_tool_call_for_prompt("Is ComfyUI running?", &[generic_services])
                .unwrap();

        assert_eq!(call.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&call).unwrap(),
            vec!["comfyui".to_owned(), "status".to_owned()]
        );
        assert!(chat_tool_call_is_read_only(&call));
    }

    #[test]
    fn supplemental_tool_call_does_not_duplicate_equivalent_status_check() {
        let comfy_status = fallback_rocm_tool_call_for_prompt("Is ComfyUI running?").unwrap();

        assert!(
            supplemental_read_only_tool_call_for_prompt("Is ComfyUI running?", &[comfy_status])
                .is_none()
        );
    }

    #[test]
    fn supplemental_tool_call_adds_running_state_for_engine_install_question() {
        let engine_inventory =
            fallback_rocm_tool_call_for_prompt("Is vLLM installed and is it running?").unwrap();
        assert_eq!(
            normalized_chat_rocm_command_args(&engine_inventory).unwrap(),
            vec!["engines".to_owned(), "list".to_owned()]
        );

        let services = supplemental_read_only_tool_call_for_prompt(
            "Is vLLM installed and is it running?",
            &[engine_inventory],
        )
        .unwrap();

        assert_eq!(services.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&services).unwrap(),
            vec!["services".to_owned(), "list".to_owned(), "--all".to_owned()]
        );
        assert!(chat_tool_call_is_read_only(&services));
    }

    #[test]
    fn supplemental_tool_call_treats_loopback_port_checks_as_equivalent() {
        let model_port_check = providers::ChatToolCall {
            id: Some("model-picked-port".to_owned()),
            name: "port_status".to_owned(),
            arguments: serde_json::json!({ "host": "localhost", "port": 8188 }),
        };

        assert!(
            supplemental_read_only_tool_call_for_prompt(
                "what is running on port 8188?",
                &[model_port_check]
            )
            .is_none()
        );
    }

    #[test]
    fn fallback_tool_call_runs_status_checks() {
        for prompt in [
            "Which GPU is on my machine, and is ROCm installed?",
            "Is TheRock setup on my machine?",
            "Check this ROCm setup.",
        ] {
            let call = fallback_rocm_tool_call_for_prompt(prompt).unwrap();
            assert_eq!(call.name, "examine");
            assert_eq!(call.arguments, serde_json::json!({}));
        }
    }

    #[test]
    fn fallback_tool_call_routes_therock_setup_to_install_folder_flow() {
        let call = fallback_rocm_tool_call_for_prompt("How do I setup TheRock?").unwrap();
        assert_eq!(call.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&call).unwrap(),
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                "release".to_owned(),
                "--format".to_owned(),
                "wheel".to_owned(),
            ]
        );
        assert!(!chat_tool_call_is_read_only(&call));
    }

    #[test]
    fn fallback_tool_call_preserves_requested_therock_install_prefix() {
        for (prompt, expected_prefix) in [
            (
                "install TheRock for me in D:\\ROCm\\therock_venvs",
                "D:\\ROCm\\therock_venvs",
            ),
            ("install ROCm to D:\\ROCm\\temp", "D:\\ROCm\\temp"),
        ] {
            let call = fallback_rocm_tool_call_for_prompt(prompt).unwrap();
            assert_eq!(call.name, "rocm_command");
            assert_eq!(
                normalized_chat_rocm_command_args(&call).unwrap(),
                vec![
                    "install".to_owned(),
                    "sdk".to_owned(),
                    "--channel".to_owned(),
                    "release".to_owned(),
                    "--format".to_owned(),
                    "wheel".to_owned(),
                    "--prefix".to_owned(),
                    expected_prefix.to_owned(),
                ]
            );
            assert!(!chat_tool_call_is_read_only(&call));
        }
    }

    #[test]
    fn fallback_tool_call_routes_requested_therock_build_date_install() {
        for (prompt, expected_prefix) in [
            (
                "Install this specific TheRock wheel from date 06052026 into D:\\ROCm\\therock_venvs",
                "D:\\ROCm\\therock_venvs",
            ),
            (
                "install ROCm at D:\\ROCm\\temp with build date 2026-06-05",
                "D:\\ROCm\\temp",
            ),
        ] {
            let call = fallback_rocm_tool_call_for_prompt(prompt).unwrap();
            assert_eq!(call.name, "rocm_command");
            assert_eq!(
                normalized_chat_rocm_command_args(&call).unwrap(),
                vec![
                    "install".to_owned(),
                    "sdk".to_owned(),
                    "--channel".to_owned(),
                    "release".to_owned(),
                    "--format".to_owned(),
                    "wheel".to_owned(),
                    "--prefix".to_owned(),
                    expected_prefix.to_owned(),
                    "--build-date".to_owned(),
                    "2026-06-05".to_owned(),
                ],
                "{prompt}"
            );
            assert!(!chat_tool_call_is_read_only(&call));
        }
    }

    #[test]
    fn fallback_tool_call_does_not_install_therock_without_folder() {
        let call = fallback_rocm_tool_call_for_prompt(
            "Install this specific TheRock wheel from date 06052026",
        )
        .unwrap();
        assert_eq!(call.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&call).unwrap(),
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                "release".to_owned(),
                "--format".to_owned(),
                "wheel".to_owned(),
                "--build-date".to_owned(),
                "2026-06-05".to_owned(),
            ]
        );
        assert!(!chat_tool_call_is_read_only(&call));
    }

    #[test]
    fn chat_install_intent_without_folder_uses_folder_picker_not_rocm_check() -> Result<()> {
        let (_root, paths) = test_paths("chat-install-intent-folder-picker");
        for prompt in [
            "i need to install rocm",
            "How do I setup TheRock?",
            "install therock",
            "How do I get it to install therock?",
            "rocm please",
            "make my AMD GPU ready",
            "get TheRock installed for local AI",
            "set up my AMD GPU for local AI",
            "install this specific TheRock wheel from date 06052026",
        ] {
            let result = render_chat_prompt_result(&paths, "local", None, prompt, true)?;
            let approval = result
                .approval
                .as_ref()
                .expect("install should need folder");
            assert_eq!(approval.pending_title, "Install ROCm", "{prompt}");
            assert_eq!(
                approval.args[..6],
                [
                    "install".to_owned(),
                    "sdk".to_owned(),
                    "--channel".to_owned(),
                    "release".to_owned(),
                    "--format".to_owned(),
                    "wheel".to_owned(),
                ],
                "{prompt}"
            );
            assert!(
                !approval.args.iter().any(|arg| arg == "--prefix"),
                "{prompt}"
            );
            assert!(
                result.rendered.contains("First choose the folder"),
                "{prompt}: {}",
                result.rendered
            );
            assert!(
                !result.rendered.contains("I checked ROCm"),
                "{prompt}: {}",
                result.rendered
            );
            assert!(
                !result.rendered.contains("ROCm CLI summary"),
                "{prompt}: {}",
                result.rendered
            );
            if prompt.contains("06052026") {
                assert!(approval.args.contains(&"--build-date".to_owned()));
                assert!(approval.args.contains(&"2026-06-05".to_owned()));
            }
        }
        Ok(())
    }

    #[test]
    fn chat_install_intent_ignores_old_conversation_words() -> Result<()> {
        let (_root, paths) = test_paths("chat-install-intent-latest-message");
        let prompt = "\
Conversation so far:
Assistant: Use /examine to refresh actual GPU memory fit before starting anything large.
Assistant: Native Windows note: models may use WSL/Linux through Windows.

New message:
install therock";

        let result = render_chat_prompt_result(&paths, "local", None, prompt, true)?;
        let approval = result
            .approval
            .as_ref()
            .expect("direct latest install request should need a folder");

        assert_eq!(approval.pending_title, "Install ROCm");
        assert!(!approval.args.iter().any(|arg| arg == "--prefix"));
        assert!(result.rendered.contains("I can install ROCm/TheRock"));
        assert!(result.rendered.contains("First choose the folder"));
        assert!(!result.rendered.contains("I checked ROCm"));
        assert!(!result.rendered.contains("ROCm CLI summary"));
        Ok(())
    }

    #[test]
    fn chat_how_to_setup_question_opens_install_folder_flow() {
        assert!(
            install_sdk_without_prefix_chat_approval("How do I setup TheRock?").is_some(),
            "a setup question should ask for the install folder"
        );
        assert!(
            install_sdk_without_prefix_chat_approval("install therock").is_some(),
            "a direct install command should open the folder picker"
        );
    }

    #[test]
    fn chat_install_intent_preserves_bare_folder_path() {
        let approval =
            install_sdk_chat_approval_for_prompt("install therock D:\\ROCm\\therock_venvs")
                .expect("direct install prompt should be recognized");

        assert_eq!(
            approval.args,
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                "release".to_owned(),
                "--format".to_owned(),
                "wheel".to_owned(),
                "--prefix".to_owned(),
                "D:\\ROCm\\therock_venvs".to_owned(),
            ]
        );
    }

    #[test]
    fn fallback_tool_call_routes_requested_therock_exact_version_install() {
        let call = fallback_rocm_tool_call_for_prompt(
            "Install the TheRock ROCm wheel version 7.13.0a20260605 into D:\\ROCm\\therock_venvs",
        )
        .unwrap();
        assert_eq!(call.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&call).unwrap(),
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                "release".to_owned(),
                "--format".to_owned(),
                "wheel".to_owned(),
                "--prefix".to_owned(),
                "D:\\ROCm\\therock_venvs".to_owned(),
                "--version".to_owned(),
                "7.13.0a20260605".to_owned(),
            ]
        );
        assert!(!chat_tool_call_is_read_only(&call));
    }

    #[test]
    fn path_exists_chat_tool_is_read_only() {
        let call = providers::ChatToolCall {
            id: Some("path-check".to_owned()),
            name: "path_exists".to_owned(),
            arguments: serde_json::json!({ "path": "D:\\ROCm\\temp" }),
        };

        validate_chat_tool_call(&call).unwrap();
        assert!(chat_tool_call_is_read_only(&call));
    }

    #[test]
    fn port_status_chat_tool_is_read_only_and_loopback_only() {
        let call = providers::ChatToolCall {
            id: Some("port-check".to_owned()),
            name: "port_status".to_owned(),
            arguments: serde_json::json!({ "host": "127.0.0.1", "port": 8188 }),
        };

        validate_chat_tool_call(&call).unwrap();
        assert!(chat_tool_call_is_read_only(&call));

        let public = providers::ChatToolCall {
            id: Some("public-port-check".to_owned()),
            name: "port_status".to_owned(),
            arguments: serde_json::json!({ "host": "192.168.1.10", "port": 8188 }),
        };
        let error = validate_chat_tool_call(&public).unwrap_err().to_string();
        assert!(error.contains("non-local host"), "{error}");
    }

    #[test]
    fn port_status_matches_loopback_managed_services() -> Result<()> {
        let (root, paths) = test_paths("port-status-loopback");
        paths.ensure()?;
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-comfyui",
            "comfyui",
            "ComfyUI",
            "ComfyUI",
            "127.0.0.1",
            18188,
            "managed",
            std::process::id(),
            Some("therock-release".to_owned()),
            None,
            Some("gpu_required".to_owned()),
        );
        record.status = "ready".to_owned();
        record.write()?;

        let call = providers::ChatToolCall {
            id: Some("port-check".to_owned()),
            name: "port_status".to_owned(),
            arguments: serde_json::json!({ "host": "localhost", "port": 18188 }),
        };
        let result = run_chat_port_status_tool(&paths, &call)?;
        let text = mcp_tool_result_text(&result);
        let managed_service_count = result
            .get("structuredContent")
            .and_then(|content| content.get("managed_services"))
            .and_then(serde_json::Value::as_array)
            .map_or(0, Vec::len);
        let _ = fs::remove_dir_all(root);

        assert_eq!(managed_service_count, 1);
        assert!(text.contains("managed_services:"), "{text}");
        assert!(text.contains("service_id=svc-comfyui"), "{text}");
        assert!(text.contains("running_state=starting"), "{text}");
        Ok(())
    }

    #[test]
    fn fallback_tool_call_routes_simple_config_changes() {
        let show = fallback_rocm_tool_call_for_prompt("Show current ROCm CLI config").unwrap();
        assert_eq!(
            normalized_chat_rocm_command_args(&show).unwrap(),
            vec!["config".to_owned(), "show".to_owned()]
        );
        assert!(chat_tool_call_is_read_only(&show));

        let engine = fallback_rocm_tool_call_for_prompt("Set the default engine to vllm").unwrap();
        assert_eq!(
            normalized_chat_rocm_command_args(&engine).unwrap(),
            vec![
                "config".to_owned(),
                "set-default-engine".to_owned(),
                "vllm".to_owned(),
            ]
        );
        assert!(!chat_tool_call_is_read_only(&engine));

        let telemetry =
            fallback_rocm_tool_call_for_prompt("Disable telemetry in settings").unwrap();
        assert_eq!(
            normalized_chat_rocm_command_args(&telemetry).unwrap(),
            vec![
                "config".to_owned(),
                "set-telemetry".to_owned(),
                "off".to_owned(),
            ]
        );
        assert!(!chat_tool_call_is_read_only(&telemetry));
    }

    #[test]
    fn fallback_tool_call_routes_comfyui_support_and_actions() {
        let status = fallback_rocm_tool_call_for_prompt("How do I setup ComfyUI?").unwrap();
        assert_eq!(status.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&status).unwrap(),
            vec!["comfyui".to_owned(), "status".to_owned()]
        );
        assert!(chat_tool_call_is_read_only(&status));

        let install = fallback_rocm_tool_call_for_prompt("Can you setup ComfyUI for me?").unwrap();
        assert_eq!(
            normalized_chat_rocm_command_args(&install).unwrap(),
            vec!["comfyui".to_owned(), "install".to_owned()]
        );
        assert!(!chat_tool_call_is_read_only(&install));
        let approval =
            chat_tool_approval_request(&install, Some("Install ComfyUI after approval.")).unwrap();
        assert_eq!(approval.pending_title, "Install ComfyUI");

        let start = fallback_rocm_tool_call_for_prompt("Can you start ComfyUI?").unwrap();
        assert_eq!(
            normalized_chat_rocm_command_args(&start).unwrap(),
            vec!["comfyui".to_owned(), "start".to_owned()]
        );
        assert!(!chat_tool_call_is_read_only(&start));
        let approval =
            chat_tool_approval_request(&start, Some("Start ComfyUI after approval.")).unwrap();
        assert_eq!(approval.pending_title, "Start ComfyUI");
    }

    #[test]
    fn fallback_tool_call_routes_local_llm_serve_requests() {
        let call =
            fallback_rocm_tool_call_for_prompt("Can you setup and serve an LLM for me?").unwrap();
        assert_eq!(call.name, "rocm_command");
        assert_eq!(
            normalized_chat_rocm_command_args(&call).unwrap(),
            vec![
                "serve".to_owned(),
                "qwen".to_owned(),
                "--engine".to_owned(),
                "lemonade".to_owned(),
                "--device".to_owned(),
                "gpu_required".to_owned(),
                "--managed".to_owned(),
            ]
        );
        assert!(!chat_tool_call_is_read_only(&call));
        let approval =
            chat_tool_approval_request(&call, Some("Start qwen after approval.")).unwrap();
        assert_eq!(approval.pending_title, "Start local model server");
        assert_eq!(
            rocm_chat_tool_requested_command(&call).as_deref(),
            Some("rocm serve qwen --engine lemonade --device gpu_required --managed")
        );
    }

    #[test]
    fn local_chat_tool_call_content_is_treated_as_intermediate() {
        let response = providers::ChatResponse {
            provider: "local".to_owned(),
            model: "Qwen/Qwen3-0.6B".to_owned(),
            content: "The active runtime root is /opt/rocm.".to_owned(),
            tool_calls: vec![providers::ChatToolCall {
                id: Some("call-1".to_owned()),
                name: "examine".to_owned(),
                arguments: serde_json::json!({}),
            }],
        };

        assert!(local_tool_call_content_is_intermediate(
            "local", true, &response
        ));
        assert!(!local_tool_call_content_is_intermediate(
            "openai", true, &response
        ));
        assert!(!local_tool_call_content_is_intermediate(
            "local", false, &response
        ));

        let without_tools = providers::ChatResponse {
            tool_calls: Vec::new(),
            ..response
        };
        assert!(!local_tool_call_content_is_intermediate(
            "local",
            true,
            &without_tools
        ));
    }

    #[test]
    fn local_chat_follow_up_with_tool_call_is_not_final_answer() {
        let response = providers::ChatResponse {
            provider: "local".to_owned(),
            model: "Qwen/Qwen3-0.6B".to_owned(),
            content: "The runtime root is /opt/rocml.".to_owned(),
            tool_calls: vec![providers::ChatToolCall {
                id: Some("call-2".to_owned()),
                name: "examine".to_owned(),
                arguments: serde_json::json!({}),
            }],
        };

        assert!(!local_follow_up_content_is_final(&response));

        let final_answer = providers::ChatResponse {
            tool_calls: Vec::new(),
            content: "The runtime root is D:\\ROCm\\therock_venvs.".to_owned(),
            ..response
        };
        assert!(local_follow_up_content_is_final(&final_answer));
    }

    #[test]
    fn visible_chat_content_removes_reasoning_blocks() {
        assert_eq!(
            visible_chat_content(
                "<think>\nchecking the tool output\n</think>\nThe runtime root is D:\\ROCm\\temp."
            ),
            "The runtime root is D:\\ROCm\\temp."
        );
        assert_eq!(
            visible_chat_content("Before\n<THINK>hidden</THINK>\nAfter"),
            "Before\n\nAfter"
        );
        assert_eq!(visible_chat_content("<think>unfinished"), "");
    }

    #[test]
    fn chat_tool_result_errors_use_plain_failure_wording() {
        assert_eq!(chat_read_only_tool_status_label(false), "done");
        assert_eq!(chat_read_only_tool_status_label(true), "reported an error");
        assert_eq!(chat_tool_display_label("examine"), "Checked this computer");
        assert_eq!(
            chat_tool_display_label("gpu_snapshot"),
            "Checked GPU status"
        );
        assert_eq!(chat_tool_display_label("install_sdk"), "Install ROCm");
        assert!(mcp_tool_result_is_error(&serde_json::json!({
            "isError": true
        })));
        assert!(!mcp_tool_result_is_error(&serde_json::json!({
            "isError": false
        })));
        assert!(!mcp_tool_result_is_error(&serde_json::json!({})));
    }

    #[test]
    fn local_chat_without_service_explains_serve_before_chat_without_llm_setup() {
        let (_root, paths) = test_paths("local-chat-no-service-guidance");
        let result =
            render_chat_prompt_result(&paths, "local", None, "Check this ROCm setup", true)
                .expect("missing local assistant should render guidance");
        assert!(result.approval.is_none());
        let rendered = result.rendered;

        assert!(rendered.contains("No local assistant is running yet."));
        assert!(rendered.contains("First-time ROCm setup does not need an LLM"));
        assert!(rendered.contains("Recommended path:"));
        assert!(rendered.contains("Advanced manual command"));
        assert!(rendered.contains(
            "rocm serve Qwen3-4B-Instruct-2507-GGUF --engine lemonade --device gpu_required --managed"
        ));
        assert!(!rendered.contains("sshleifer/tiny-gpt2"));
        assert!(rendered.contains("rocm chat --tools --provider local --prompt"));
        assert!(rendered.contains("Nothing was changed."));
        assert!(!rendered.contains("install sdk"));
        assert!(!rendered.contains("setup TheRock with an LLM"));
    }

    #[test]
    fn local_chat_status_prompts_use_read_only_tools_without_assistant() -> Result<()> {
        let (_root, paths) = test_paths("local-chat-status-fallback");

        let running =
            render_chat_prompt_result(&paths, "local", None, "Is vLLM running?", true)?.rendered;
        assert!(!running.contains("No local assistant is running yet."));
        assert!(running.contains("Checked model servers: done"), "{running}");
        assert!(running.contains("ROCm CLI summary"), "{running}");
        assert!(
            running.contains("Local model servers: none running under ROCm CLI."),
            "{running}"
        );
        assert!(running.contains("Nothing was changed."));

        let installed =
            render_chat_prompt_result(&paths, "local", None, "Is vLLM installed?", true)?.rendered;
        assert!(!installed.contains("No local assistant is running yet."));
        assert!(installed.contains("Engine runtimes:"), "{installed}");
        assert!(installed.contains("vLLM:"), "{installed}");

        let installed_and_running = render_chat_prompt_result(
            &paths,
            "local",
            None,
            "Is vLLM installed and is it running?",
            true,
        )?
        .rendered;
        assert!(
            installed_and_running.contains("Checked local engines: done"),
            "{installed_and_running}"
        );
        assert!(
            installed_and_running.contains("Checked model servers: done"),
            "{installed_and_running}"
        );
        assert!(
            installed_and_running.contains("Engine runtimes:"),
            "{installed_and_running}"
        );
        assert!(
            installed_and_running.contains("Local model servers: none running under ROCm CLI."),
            "{installed_and_running}"
        );

        let port = render_chat_prompt_result(
            &paths,
            "local",
            None,
            "What is running on port 8188?",
            true,
        )?
        .rendered;
        assert!(!port.contains("No local assistant is running yet."));
        assert!(port.contains("Checked local port: done"), "{port}");
        assert!(port.contains("Port 8188:"), "{port}");
        Ok(())
    }

    #[test]
    fn chat_tools_anthropic_reaches_provider_opt_in_boundary() {
        let (_root, paths) = test_paths("anthropic-chat-tools-opt-in");

        let error = render_chat_prompt_result(
            &paths,
            "anthropic",
            Some("claude-test"),
            "Check this ROCm setup",
            true,
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("cloud provider `anthropic` is disabled"));
        assert!(error.contains("rocm config enable-provider anthropic"));
        assert!(!error.contains("OpenAI-compatible provider"));
    }

    /// The half the parse test above cannot reach: that the parsed flag is what
    /// the install request carries.
    ///
    /// `install()` is not callable here — it needs `uv`, a live index and a real
    /// probe — so the mapping is extracted into `sdk_install_request` and pinned
    /// directly. Hardcoding `include_devel` at that mapping is the change this
    /// catches and the clap test does not.
    #[test]
    fn install_sdk_request_forwards_the_parsed_devel_flag() {
        for devel in [false, true] {
            let request = sdk_install_request(
                "release",
                "wheel",
                None,
                None,
                None,
                false,
                devel,
                therock::SdkInstallConsent::Ask,
            );
            assert_eq!(
                request.include_devel, devel,
                "the parsed --devel flag must reach the install request"
            );
        }

        // And the flag must not be confused with the neighbouring bool.
        let dry_run_only = sdk_install_request(
            "release",
            "wheel",
            None,
            None,
            None,
            true,
            false,
            therock::SdkInstallConsent::Ask,
        );
        assert!(dry_run_only.dry_run);
        assert!(!dry_run_only.include_devel);
    }

    /// End to end across the two seams a `rocm install sdk --devel` traverses:
    /// clap parse, then the request mapping. Neither alone proves the flag
    /// survives the trip.
    #[test]
    fn parsed_install_sdk_arguments_reach_the_request_with_devel_intact() {
        for (args, expected) in [
            (vec!["rocm", "install", "sdk"], false),
            (vec!["rocm", "install", "sdk", "--devel"], true),
        ] {
            let cli = Cli::try_parse_from(&args).expect("install sdk should parse");
            let Some(Command::Install {
                target: InstallTarget::Sdk { devel, .. },
            }) = cli.command
            else {
                panic!("expected an install sdk target for {args:?}");
            };
            let request = sdk_install_request(
                "release",
                "wheel",
                None,
                None,
                None,
                false,
                devel,
                therock::SdkInstallConsent::Ask,
            );
            assert_eq!(
                request.include_devel, expected,
                "--devel did not survive parse -> request for {args:?}"
            );
        }
    }

    /// The same question — "is this a system location?" — also keeps the local
    /// assistant from installing into one, and there the folder arrives as
    /// free-form JSON from a model rather than from a file the user wrote. A
    /// trailing separator is exactly the sort of thing generated text carries,
    /// so the refusal has to survive every spelling here too.
    #[test]
    #[cfg(unix)]
    fn the_assistant_cannot_name_a_system_install_folder_by_respelling_it() {
        let home = rocm_core::runtime_home_dir().expect("a home directory");
        let home = home.display().to_string();
        let spellings = [
            "/usr/".to_owned(),
            "/opt/".to_owned(),
            "//usr".to_owned(),
            "/./opt".to_owned(),
            format!("{home}/../../usr"),
        ];

        let accepted: Vec<String> = spellings
            .into_iter()
            .filter(|text| !chat_install_prefix_is_system(Path::new(text)))
            .collect();

        assert!(
            accepted.is_empty(),
            "accepted as a user install folder: {accepted:?}"
        );
    }

    #[test]
    fn piped_prompt_keeps_everything_but_the_trailing_line_ending() {
        // A piped prompt must reach the send path byte-identical to the same
        // text passed with `--prompt`, which is forwarded verbatim. Only the
        // line ending the writer appends is dropped; indentation and trailing
        // spaces or tabs are content and have to survive.
        assert_eq!(
            piped_prompt_from_input("    indented line\n"),
            Some("    indented line".to_owned()),
            "leading indentation must survive; only the trailing newline goes"
        );
        assert_eq!(
            piped_prompt_from_input("trailing spaces matter   \n"),
            Some("trailing spaces matter   ".to_owned()),
            "trailing spaces must survive the trailing-newline strip"
        );
        assert_eq!(
            piped_prompt_from_input("\tleading tab"),
            Some("\tleading tab".to_owned()),
            "input without a trailing newline must pass through unchanged"
        );
        assert_eq!(
            piped_prompt_from_input("crlf line\r\n"),
            Some("crlf line".to_owned()),
            "a Windows line ending must be stripped as one unit"
        );
        assert_eq!(
            piped_prompt_from_input("fn main() {\n    body\n}\n"),
            Some("fn main() {\n    body\n}".to_owned()),
            "interior newlines and indentation must survive"
        );
        assert_eq!(
            piped_prompt_from_input("blank line below\n\n"),
            Some("blank line below\n".to_owned()),
            "only one trailing line ending is shell punctuation; the rest is content"
        );
    }

    #[test]
    fn piped_prompt_treats_whitespace_only_input_as_no_prompt() {
        // The fallback that keeps the no-argument behavior intact: an empty
        // pipe, a bare newline, or blank padding carries no prompt, so the
        // caller must see `None` and render the status screen instead of
        // sending whitespace to a model.
        for input in ["", "\n", "\r\n", "   ", "  \t \n\n", "\n\n\n"] {
            assert_eq!(
                piped_prompt_from_input(input),
                None,
                "whitespace-only stdin must yield no prompt; input: {input:?}"
            );
        }
    }
}
