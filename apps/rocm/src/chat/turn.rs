// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The chat assistant's turn loop and transcript: one prompt in, the rendered
//! reply (and the approval request, if the assistant asked for a change) out.
//! `render_chat_prompt_result_with_progress` is the entry point. It builds the
//! assistant's system prompt (`ROCM_CHAT_TOOL_SYSTEM_PROMPT` with the compiled-in
//! assistant skill, grounded in the host by `rocm_chat_tool_system_prompt_for_host`),
//! asks the provider, runs the read-only tool calls the reply requests through
//! the bridge, summarizes their output, and renders the transcript
//! (`append_chat_tool_results`, the `local_*` and display-label helpers). It
//! owns the turn's result types, `ChatPromptResult` and `ChatToolRunResult`.
//!
//! A child of the `chat` module and the top of its call graph: it calls into
//! the prompt-intent heuristics (`chat/intent.rs`), the MCP bridge
//! (`chat/mcp_bridge.rs`), the summaries (`chat/summaries.rs`) and
//! `tool_policy.rs`, and none of them calls back into it. `chat.rs` reads
//! `render_chat_prompt_result_with_progress` from here, and re-exports
//! `rocm_chat_tool_system_prompt_for_host` for `dash.rs`.

use super::intent::{
    fallback_engine_list_tool_call, fallback_rocm_tool_call_for_prompt,
    fallback_services_list_tool_call, install_sdk_without_prefix_chat_approval,
    latest_user_chat_message, prompt_asks_engine_install_state, prompt_asks_running_or_status,
    prompt_can_use_read_only_without_local_assistant, prompt_mentions_serving_engine_or_service,
};
use super::mcp_bridge::{loopback_host_key, run_chat_read_only_tool};
use super::summaries::{deterministic_chat_tool_summary, deterministic_summary_can_stand_alone};
use crate::engine_runtime_select::service_model_names_match;
use crate::freeform_planner::format_structured_tool_call;
use crate::managed_service_status::load_managed_services;
use crate::rocm_argv::chat_cli_arg_value;
use crate::tool_policy::{
    ChatToolApprovalRequest, chat_tool_approval_request, chat_tool_call_is_read_only, json_string,
    normalized_chat_rocm_command_args, rocm_chat_tool_requested_command, validate_chat_tool_call,
};
use crate::{ChatInferenceParams, chat_host_facts, providers};
use anyhow::Result;
use rocm_core::{AppPaths, DEFAULT_LOCAL_HOST};
use std::fmt::Write as _;

pub(super) fn mcp_tool_result_text(value: &serde_json::Value) -> String {
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

#[derive(Debug, Clone, Eq, PartialEq)]
pub(super) struct ChatPromptResult {
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

pub(super) fn render_chat_prompt_result_with_progress(
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
const ROCM_CHAT_TOOL_SYSTEM_PROMPT: &str = "You are ROCm CLI's local assistant. Speak in simple English for non-technical users. Use the provided ROCm tools when you need to inspect this machine, preview setup, read service logs, check updates, inspect automations, install or start ROCm-managed apps, or request ROCm/TheRock, config, engine, app, and local model server changes. For simple greetings or thanks like hello, hi, hey, ok, or thank you, reply normally; do not inspect ROCm, do not call tools, and do not launch or propose a model server. Tool-use rules: inspect first with read-only tools; call rocm_command only with argv-style args and no shell text; use natural_language_plan for ROCm requests that do not fit another read-only tool; ask for a mutating tool call only after explaining why it is needed; summarize tool results after they are returned. Read-only tools may run immediately. Tools that install, launch, stop, delete, or change state require user approval; request rocm_command and explain why. For 'is X running?', 'what is running?', status, or port questions, inspect before answering and do not start, stop, install, or serve anything. For ComfyUI or port 8188 use [\"comfyui\",\"status\"] or port_status. For vLLM, Lemonade, AIM, qwen, or local model servers use [\"services\",\"list\",\"--all\"] for running state and [\"engines\",\"list\"] for installed/available engine state. Treat ready/running as running, starting/recovering as starting, failed/stopped as not running, and no matching record as unknown or not managed by ROCm CLI. Interpret Examine carefully: active_runtime_status=ready means ROCm CLI has an active managed TheRock/ROCm runtime; legacy_rocm_status=not_detected only means no global system ROCm install was found. If active_runtime_status=ready, tell the user ROCm/TheRock is installed and active for ROCm CLI. For 'is TheRock installed', 'is ROCm installed', or 'which GPU is on this machine', use examine or gpu_snapshot before answering. For 'how do I setup TheRock' or install/setup requests, guide the user to choose an install folder first; do not answer with only a status check. For 'which LLMs can this machine support', use rocm_command args [\"model\"] or natural_language_plan before answering. For TheRock installs, always let the user choose the install folder. If the user names a folder or prefix, preserve that exact folder with [\"--prefix\",\"PATH\"]; you may call path_exists first to check whether that user-provided folder or its parent exists. If the user asks you to install TheRock/ROCm but has not named a folder, ask for the folder or let the guided setup folder picker collect it; do not invent a hidden default folder and do not request an install command without --prefix. Use rocm_command args [\"install\",\"sdk\",\"--channel\",\"release\",\"--format\",\"wheel\",\"--prefix\",\"PATH\"] only when the user asks you to install it and a folder is known; for a requested build date add [\"--build-date\",\"YYYY-MM-DD\"] and for a requested exact version add [\"--version\",\"VERSION\"]. For config changes, inspect with [\"config\",\"show\"] first when useful, then request config subcommands such as [\"config\",\"set-default-engine\",\"lemonade\"], [\"config\",\"set-default-runtime\",\"RUNTIME_KEY\"], or [\"config\",\"set-telemetry\",\"local\"] only after explaining why. For ComfyUI, use rocm_command with args like [\"comfyui\",\"status\"], [\"comfyui\",\"logs\"], [\"comfyui\",\"install\"], [\"comfyui\",\"start\"], or [\"comfyui\",\"stop\"]. First-time setup is the same thing as bootstrap in ROCm CLI; it is a deterministic ROCm setup flow, not a separate model chat. The built-in local assistant is fixed to qwen, which maps to Qwen3-4B-Instruct-2507-GGUF served by Lemonade with gpu_required. vLLM, Lemonade, and AIM (Docker-based) are the general serving engines; inspect or manage them when the user asks about general model serving, but do not switch the built-in assistant away from Lemonade. Use qwen-smoke only for a quick server smoke test. For vLLM management, inspect engines first and use [\"engines\",\"install\",\"vllm\"] or [\"serve\",\"MODEL\",\"--engine\",\"vllm\",\"--device\",\"gpu_required\",\"--managed\"] only where the host supports it. Do not invent shell commands and do not request CPU fallback.";

const ROCM_CHAT_TOOL_SKILL: &str = include_str!("../../../../skills/rocm-cli-assistant/SKILL.md");

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

const fn chat_read_only_tool_status_label(is_error: bool) -> &'static str {
    if is_error {
        "reported an error"
    } else {
        "done"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::summaries::deterministic_rocm_tool_summary;
    use crate::tests::test_paths;

    fn rocm_command_call(args: &[&str]) -> providers::ChatToolCall {
        providers::ChatToolCall {
            id: Some("test".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({ "args": args }),
        }
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
            grounded.contains("vLLM and AIM are skipped on native Windows"),
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
    fn local_rocm_tools_chat_uses_fixed_lemonade_qwen_assistant() {
        assert_eq!(
            local_rocm_tools_assistant_model("local", true),
            Some(providers::BUILTIN_ASSISTANT_MODEL_ID)
        );
        assert_eq!(local_rocm_tools_assistant_model("local", false), None);
        assert_eq!(local_rocm_tools_assistant_model("openai", true), None);
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
}
