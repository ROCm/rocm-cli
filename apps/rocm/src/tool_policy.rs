// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The chat assistant's tool-call policy: which `rocm` commands and bridge
//! tools a model may request, which of them run immediately, and which need
//! the user's approval first.
//!
//! `validate_chat_tool_call` and the `validate_chat_*_tool_call` family
//! reject malformed or unsafe requests (a CPU or public-bind `serve`, a
//! system install folder, a service id or watcher mode that does not parse);
//! `chat_rocm_command_action` and `chat_rocm_command_action_from_args` sort a
//! `rocm` argv into a read-only command or one that needs approval
//! (`ChatRocmCommandAction`); `chat_tool_approval_request` builds the approval
//! prompt (`ChatToolApprovalRequest`) the dashboard shows;
//! `chat_tool_call_is_read_only` is the one yes/no a caller asks before
//! running a call; and `rocm_chat_tool_requested_args` and
//! `rocm_chat_tool_requested_command` turn a tool call into the argv and the
//! one-line command it stands for.
//!
//! The module was relocated verbatim from `chat.rs`, which calls into it and
//! re-exports the items other modules reach as `chat::<item>`. It calls
//! nothing in `chat.rs` and nothing in `freeform_planner.rs`, which is what
//! lets the planner call into it without importing `chat`.

use crate::rocm_argv::{
    chat_cli_arg_value, chat_cli_arg_value_checked, chat_cli_has_flag, ensure_flag,
    format_structured_tool_call, serve_args_request_cpu_device,
};
use crate::service_launch::is_loopback_host;
use crate::services_text::validate_service_id;
use crate::{providers, therock};
use anyhow::{Context, Result, bail};
use rocm_core::{DEFAULT_LOCAL_HOST, builtin_watcher, runtime_install_root_is_protected};
use std::path::Path;

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) enum ChatRocmCommandAction {
    ReadOnly(Vec<String>),
    Approval {
        args: Vec<String>,
        pending_title: String,
        command_title: String,
    },
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

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct ChatToolApprovalRequest {
    pub pending_title: String,
    pub command_title: String,
    pub args: Vec<String>,
    pub display_command: Option<String>,
    pub explanation: Option<String>,
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

pub(crate) fn chat_rocm_command_action(
    call: &providers::ChatToolCall,
) -> Result<ChatRocmCommandAction> {
    let args = normalized_chat_rocm_command_args(call)?;
    chat_rocm_command_action_from_args(args)
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

/// The chat action for `rocm install driver`. Where the command only reports
/// (`only_reports`, the same decision the command itself makes) it is a
/// read-only call with no `--yes`; elsewhere it needs approval and carries
/// `--yes`.
///
/// A model-supplied `--yes` is stripped in both forms, bare and `--yes=...`,
/// as the `install sdk` arm does: the approval path appends its own bare
/// `--yes`, so a stray attached form would otherwise ride along and clap would
/// refuse the whole command, and the read-only path has no consent to grant.
pub(crate) fn chat_install_driver_action(
    mut args: Vec<String>,
    only_reports: bool,
) -> ChatRocmCommandAction {
    args.retain(|arg| arg != "--yes" && !arg.starts_with("--yes="));
    if only_reports {
        return ChatRocmCommandAction::ReadOnly(args);
    }
    ensure_flag(&mut args, "--yes");
    ChatRocmCommandAction::Approval {
        args,
        pending_title: "Install driver".to_owned(),
        command_title: "Install".to_owned(),
    }
}

pub(crate) fn chat_rocm_command_action_from_args(
    mut args: Vec<String>,
) -> Result<ChatRocmCommandAction> {
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
        Some("install") if second.as_deref() == Some("driver") => Ok(chat_install_driver_action(
            args,
            crate::driver_install::host_install_driver_only_reports(),
        )),
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

pub(crate) fn chat_install_prefix_is_system(prefix: &Path) -> bool {
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

pub(crate) fn rocm_chat_tool_requested_command(call: &providers::ChatToolCall) -> Option<String> {
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
    use crate::{Cli, Command, InstallTarget};
    use clap::Parser;

    fn rocm_command_call(args: &[&str]) -> providers::ChatToolCall {
        providers::ChatToolCall {
            id: Some("test".to_owned()),
            name: "rocm_command".to_owned(),
            arguments: serde_json::json!({ "args": args }),
        }
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

    /// The `install driver` chat action strips `--yes` in the attached form too,
    /// like the `install sdk` arm: read-only where the command only reports, and
    /// a single bare `--yes` where it installs.
    #[test]
    fn chat_install_driver_strips_a_model_supplied_yes_in_both_forms() {
        for supplied in ["--yes", "--yes=true", "--yes=1", "--yes=false"] {
            let argv = |extra: &str| -> Vec<String> {
                ["install", "driver", "--dry-run", extra]
                    .iter()
                    .map(|arg| (*arg).to_owned())
                    .collect()
            };
            let ChatRocmCommandAction::ReadOnly(args) =
                chat_install_driver_action(argv(supplied), true)
            else {
                panic!("a report-only platform must stay read-only");
            };
            assert_eq!(args, ["install", "driver", "--dry-run"], "{supplied}");

            let ChatRocmCommandAction::Approval { args, .. } =
                chat_install_driver_action(argv(supplied), false)
            else {
                panic!("an installing platform must ask approval");
            };
            assert_eq!(
                args,
                ["install", "driver", "--dry-run", "--yes"],
                "{supplied}"
            );
        }
        let ChatRocmCommandAction::ReadOnly(args) = chat_install_driver_action(
            ["install", "driver", "--yes-not-a-flag"]
                .iter()
                .map(|arg| (*arg).to_owned())
                .collect(),
            true,
        ) else {
            panic!("read-only expected");
        };
        assert!(args.iter().any(|arg| arg == "--yes-not-a-flag"), "{args:?}");
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
}
