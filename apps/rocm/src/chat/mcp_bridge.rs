// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The chat assistant's MCP bridge and in-process command execution: the
//! entry point `run_internal_mcp_call` (the tool dispatch behind `rocm
//! mcp-call`, the dash execution seam and the assistant's read-only tool
//! path), the in-process runner for the read-only `rocm` commands the assistant
//! may ask for (`run_rocm_read_only_in_process`), the helpers that wrap a
//! command's output as an MCP tool result, and the `path_exists` and
//! `port_status` tools. `render_services_tool_result_text`, the text of the
//! `services` tool, lives here too, as does `mcp_tool_result_text`, which reads
//! the text back out of an MCP tool result for the turn loop.
//!
//! A child of the `chat` module. It calls into `tool_policy.rs` (the validators
//! and the approval split), `rocm_argv.rs`, `chat/summaries.rs`
//! (`managed_service_running_state`) and the crate-root renderers; nothing here
//! calls back into the turn loop in `chat/turn.rs`, tests included; the turn
//! loop reads this module's items through `super::mcp_bridge`.
//! `run_internal_mcp_call` stays reachable as `chat::run_internal_mcp_call` for
//! `main.rs` and `dash_seam.rs`.

use super::summaries::managed_service_running_state;
use crate::automations::{render_automations_text, render_daemon_text};
use crate::engine_runtime_select::app_path_env_vars;
use crate::freeform_planner::{format_structured_tool_call, plan_freeform_request_with_context};
use crate::managed_service_status::{load_managed_services, managed_service_is_live};
use crate::rocm_argv::{chat_cli_arg_value, chat_cli_has_flag};
use crate::runtimes_cmd::render_runtimes_text;
use crate::services_text::{
    read_tail_lines, render_logs_browser_text, render_logs_text, render_service_logs_text,
    render_services_text,
};
use crate::tool_policy::{
    ChatRocmCommandAction, chat_install_prefix_is_system, chat_rocm_command_action,
    chat_tool_call_is_read_only, json_string, rocm_chat_tool_requested_args,
    validate_chat_tool_call,
};
use crate::uninstall_plan::render_uninstall_dry_run;
use crate::update_cmd::render_update_text;
use crate::{
    build_codex_bridge_gpu_snapshot, build_codex_bridge_snapshot,
    builtin_codex_bridge_engine_inventory, cli_version_string, record_cli_audit_event,
    render_config_text, render_engine_inventory_text_with_paths, render_examine_text,
    render_examine_text_with_paths, render_model_registry_verbose_text_with_context_and_host,
    render_setup_status_text, therock_install_version_selector,
};
use crate::{comfyui, providers, therock};
use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, AutomationRuntimeState, CommandCapture, DEFAULT_LOCAL_HOST, ExamineSummary,
    ManagedServiceRecord, RocmCliConfig, command_capture_text, load_recent_automation_events,
};
use rocm_engine_protocol::DEFAULT_LOG_TAIL_LINES;
use std::fmt::Write as _;
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::time::Duration;

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

pub(super) fn loopback_host_key(host: &str) -> String {
    match host.trim().to_ascii_lowercase().as_str() {
        "localhost" | "127.0.0.1" => "127.0.0.1".to_owned(),
        "::1" | "[::1]" => "::1".to_owned(),
        other => other.to_owned(),
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
            // Built exactly once: the text shown to the operator and the
            // `action` reported as what would run next both come from the
            // same `StructuredRequestPlan`, so a provider-assisted plan (not
            // guaranteed deterministic across calls) cannot describe one
            // command in `text` and a different one in `action`.
            let (text, action, planned_args) =
                plan_freeform_request_with_context(&request, paths, &config);
            let action = action.map(|action| {
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
                    // Every action's argv, as its `tool_call:` line in `text`
                    // prints it, so a caller can hold a model's request to the
                    // model-only rules (`tool_policy::validate_model_chosen_plan`).
                    "planned_args": planned_args,
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

fn run_rocm_capture_for_paths(
    paths: &AppPaths,
    args: &[&str],
    timeout: Duration,
) -> Result<CommandCapture> {
    rocm_core::run_rocm_capture_with_env(args, &app_path_env_vars(paths), timeout)
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
        // `install driver` is a read-only call where the command only reports
        // (`chat_install_driver_action`). It runs as a dry run that never
        // executes anything, so a call that reaches it elsewhere cannot change
        // the host either. `--dry-run` is accepted because it is what this arm
        // already does; `--reconcile` is not, since it rewrites the saved driver
        // install state and the command refuses to combine it with the other
        // flags.
        [command, subcommand, rest @ ..]
            if command.eq_ignore_ascii_case("install")
                && subcommand.eq_ignore_ascii_case("driver")
                && rest.iter().all(|arg| {
                    arg.eq_ignore_ascii_case("--dkms") || arg.eq_ignore_ascii_case("--dry-run")
                }) =>
        {
            let dkms = rest.iter().any(|arg| arg.eq_ignore_ascii_case("--dkms"));
            crate::driver_install::install_driver(paths, dkms, false, true)
                .map(|result| result.output)
                .map_err(|error| error.source)
        }
        _ => bail!(
            "unsupported in-process read-only rocm command: {}",
            format_structured_tool_call("rocm", args)
        ),
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

pub(super) fn run_chat_read_only_tool(
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

pub(super) fn run_chat_path_exists_tool(
    call: &providers::ChatToolCall,
) -> Result<serde_json::Value> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::test_paths;
    use std::fs;
    use std::net::TcpListener;

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

    /// On Windows `rocm install driver` only reports, and chat plans it as a
    /// read-only call. That call must run in process: before this arm existed it
    /// failed with "unsupported in-process read-only rocm command". It is a dry
    /// run, so it never executes anything on any platform.
    #[test]
    fn install_driver_read_only_call_runs_in_process() {
        let (_root, paths) = test_paths("install-driver-in-process");
        let args = vec!["install".to_owned(), "driver".to_owned()];
        let text = run_rocm_read_only_in_process(&paths, &args)
            .expect("the read-only install driver call should run in process");
        assert!(text.contains("approval:"), "{text}");
        assert!(!text.contains("driver install completed"), "{text}");
        assert!(
            run_rocm_read_only_in_process(
                &paths,
                &[
                    "install".to_owned(),
                    "driver".to_owned(),
                    "--yes".to_owned()
                ]
            )
            .is_err(),
            "a flag the read-only path does not know is still refused"
        );
    }

    /// `--dry-run` is what the in-process arm already does, so it is accepted
    /// and the call reports a dry run. `--reconcile` writes driver install state
    /// and stays refused, as does any flag the arm does not know.
    #[test]
    fn install_driver_read_only_call_accepts_dry_run_and_refuses_reconcile() {
        let (_root, paths) = test_paths("install-driver-in-process-dry-run");
        let run = |extra: &[&str]| {
            let mut args = vec!["install".to_owned(), "driver".to_owned()];
            args.extend(extra.iter().map(|arg| (*arg).to_owned()));
            run_rocm_read_only_in_process(&paths, &args)
        };
        for extra in [&["--dry-run"][..], &["--dkms", "--dry-run"], &["--DRY-RUN"]] {
            let text = run(extra).unwrap_or_else(|error| panic!("{extra:?} must run: {error}"));
            assert!(text.contains("dry_run: true"), "{extra:?}: {text}");
            assert!(
                !text.contains("driver install completed"),
                "{extra:?}: {text}"
            );
        }
        for extra in [&["--reconcile"][..], &["--dry-run", "--reconcile"]] {
            assert!(run(extra).is_err(), "{extra:?} must stay refused");
        }
        // Refusing `--reconcile` changed nothing on disk: the state file is
        // written only by a reconcile or an executed install.
        assert!(
            !crate::driver_install::driver_install_state_path(&paths).exists(),
            "a refused call must not create driver install state"
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
}
