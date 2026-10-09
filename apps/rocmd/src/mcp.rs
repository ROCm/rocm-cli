// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use crate::common::{self, CommandCapture};
use crate::persistence;
use anyhow::{Context, Result, bail};
#[cfg(test)]
use rocm_core::unix_time_millis;
use rocm_core::{
    AppPaths, AutomationRuntimeState, DEFAULT_LOCAL_HOST, ExamineSummary, RocmCliConfig,
    load_recent_automation_events,
};
use serde::Serialize;
use serde_json::Value;
use serde_json::json;
use std::collections::VecDeque;
use std::fs;
use std::io::{self, BufRead, Write};

pub(crate) fn run_mcp_server(paths: &AppPaths) -> Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = stdin.lock();
    let mut writer = stdout.lock();
    let mut line = String::new();

    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }

        let message: Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(error) => {
                write_json_line(
                    &mut writer,
                    &json!({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": format!("parse error: {error}"),
                        }
                    }),
                )?;
                continue;
            }
        };

        let Some(method) = message.get("method").and_then(Value::as_str) else {
            if message.get("id").is_some() {
                write_json_line(
                    &mut writer,
                    &json!({
                        "jsonrpc": "2.0",
                        "id": message.get("id").cloned().unwrap_or(Value::Null),
                        "error": {
                            "code": -32600,
                            "message": "invalid request: missing method",
                        }
                    }),
                )?;
            }
            continue;
        };

        let id = message.get("id").cloned();
        let params = message.get("params").cloned().unwrap_or(Value::Null);
        match method {
            "initialize" => {
                let protocol_version = params
                    .get("protocolVersion")
                    .and_then(Value::as_str)
                    .unwrap_or("2025-03-26");
                if let Some(id) = id {
                    write_json_line(
                        &mut writer,
                        &json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "protocolVersion": protocol_version,
                                "capabilities": {
                                    "tools": {
                                        "listChanged": true,
                                    }
                                },
                                "serverInfo": {
                                    "name": "rocmd-mcp-server",
                                    "title": "ROCm AI Command Center",
                                    "version": env!("CARGO_PKG_VERSION"),
                                }
                            }
                        }),
                    )?;
                }
            }
            "ping" => {
                if let Some(id) = id {
                    write_json_line(
                        &mut writer,
                        &json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {}
                        }),
                    )?;
                }
            }
            "notifications/initialized" => {}
            "tools/list" => {
                if let Some(id) = id {
                    write_json_line(
                        &mut writer,
                        &json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "tools": rocm_mcp_tools(),
                                "nextCursor": Value::Null,
                            }
                        }),
                    )?;
                }
            }
            "tools/call" => {
                if let Some(id) = id {
                    let result = match handle_mcp_tool_call(paths, &params) {
                        Ok(result) => result,
                        Err(error) => tool_error(
                            format!("ROCm MCP tool call failed: {error:#}"),
                            json!({
                                "tool": params.get("name").cloned().unwrap_or(Value::Null),
                            }),
                        ),
                    };
                    write_json_line(
                        &mut writer,
                        &json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": result,
                        }),
                    )?;
                }
            }
            notification if notification.starts_with("notifications/") => {}
            other => {
                if let Some(id) = id {
                    write_json_line(
                        &mut writer,
                        &json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": {
                                "code": -32601,
                                "message": format!("method not found: {other}"),
                            }
                        }),
                    )?;
                }
            }
        }
    }

    Ok(())
}

fn write_json_line(writer: &mut impl Write, value: &Value) -> Result<()> {
    writer.write_all(serde_json::to_string(value)?.as_bytes())?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

pub(crate) fn print_json<T: Serialize>(value: &T) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(value).context("failed to serialize json output")?
    );
    Ok(())
}

pub(crate) fn rocm_mcp_tools() -> Vec<Value> {
    vec![
        rocm_mcp_tool(
            "examine",
            "Read the current ROCm AI Command Center host summary.",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "bridge_snapshot",
            "Read the full ROCm bridge snapshot including examine data, engines, services, automations, and gpu telemetry.",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "gpu_snapshot",
            "Read the current amd-smi GPU telemetry snapshot if available.",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "engines",
            "List available ROCm serving engines and whether each one is installed.",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "services",
            "List managed model services and their current status.",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "service_logs",
            "Read the tail of a managed service log file.",
            json!({
                "type": "object",
                "properties": {
                    "service_id": {
                        "type": "string"
                    },
                    "lines": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500
                    }
                },
                "required": ["service_id"],
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "automations",
            "List automation runtime status, watcher events, and local webhook events.",
            json!({
                "type": "object",
                "properties": {
                    "event_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 64
                    }
                },
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "natural_language_plan",
            "Ask `rocm` to translate a natural-language ROCm request into a visible plan without executing privileged work.",
            json!({
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string"
                    }
                },
                "required": ["request"],
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "rocm_command",
            "Run a supported read-only ROCm CLI command with argv-style arguments. Commands that change ROCm state are rejected here and must go through the ROCm CLI approval UI.",
            json!({
                "type": "object",
                "properties": {
                    "args": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "minItems": 1,
                        "maxItems": 64
                    },
                    "reason": {
                        "type": "string"
                    }
                },
                "required": ["args"],
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "update_check",
            "Run `rocm update` and return the current TheRock update status.",
            json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "install_sdk_dry_run",
            "Run a dry-run TheRock SDK install plan.",
            json!({
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "enum": ["release", "nightly"]
                    },
                    "format": {
                        "type": "string",
                        "enum": ["wheel", "tarball"]
                    },
                    "prefix": {
                        "type": "string"
                    },
                    "version": {
                        "type": "string"
                    },
                    "build_date": {
                        "type": "string"
                    }
                },
                "additionalProperties": false
            }),
            true,
            false,
        ),
        rocm_mcp_tool(
            "install_sdk",
            "Install a TheRock SDK into the managed runtime area or an explicitly approved prefix.",
            json!({
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "enum": ["release", "nightly"]
                    },
                    "format": {
                        "type": "string",
                        "enum": ["wheel", "tarball"]
                    },
                    "prefix": {
                        "type": "string"
                    },
                    "version": {
                        "type": "string"
                    },
                    "build_date": {
                        "type": "string"
                    },
                    "allow_system_prefix": {
                        "type": "boolean"
                    }
                },
                "additionalProperties": false
            }),
            false,
            true,
        ),
        rocm_mcp_tool(
            "install_engine",
            "Install or refresh a managed serving engine environment.",
            json!({
                "type": "object",
                "properties": {
                    "engine": {
                        "type": "string"
                    },
                    "runtime_id": {
                        "type": "string"
                    },
                    "python_version": {
                        "type": "string"
                    },
                    "reinstall": {
                        "type": "boolean"
                    }
                },
                "required": ["engine"],
                "additionalProperties": false
            }),
            false,
            false,
        ),
        rocm_mcp_tool(
            "launch_server",
            "Launch a managed local model server through `rocm serve --managed`.",
            json!({
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string"
                    },
                    "engine": {
                        "type": "string"
                    },
                    "device": {
                        "type": "string"
                    },
                    "runtime_id": {
                        "type": "string"
                    },
                    "env_id": {
                        "type": "string"
                    },
                    "host": {
                        "type": "string"
                    },
                    "port": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 65535
                    },
                    "allow_public_bind": {
                        "type": "boolean"
                    }
                },
                "required": ["model"],
                "additionalProperties": false
            }),
            false,
            true,
        ),
        rocm_mcp_tool(
            "stop_server",
            "Stop a managed service by service id and update its manifest status.",
            json!({
                "type": "object",
                "properties": {
                    "service_id": {
                        "type": "string"
                    }
                },
                "required": ["service_id"],
                "additionalProperties": false
            }),
            false,
            true,
        ),
        rocm_mcp_tool(
            "watcher_enable",
            "Enable a watcher and optionally set its mode.",
            json!({
                "type": "object",
                "properties": {
                    "watcher": {
                        "type": "string"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["observe", "propose", "contained"]
                    }
                },
                "required": ["watcher"],
                "additionalProperties": false
            }),
            false,
            false,
        ),
        rocm_mcp_tool(
            "watcher_disable",
            "Disable a watcher.",
            json!({
                "type": "object",
                "properties": {
                    "watcher": {
                        "type": "string"
                    }
                },
                "required": ["watcher"],
                "additionalProperties": false
            }),
            false,
            false,
        ),
    ]
}

fn rocm_mcp_tool(
    name: &str,
    description: &str,
    input_schema: Value,
    read_only: bool,
    destructive: bool,
) -> Value {
    json!({
        "name": name,
        "title": name.replace('_', " "),
        "description": description,
        "annotations": {
            "readOnlyHint": read_only,
            "destructiveHint": destructive,
            "openWorldHint": false,
        },
        "inputSchema": input_schema,
    })
}

fn mcp_tool_requires_direct_approval(name: &str) -> bool {
    matches!(
        name,
        "install_sdk"
            | "install_engine"
            | "launch_server"
            | "stop_server"
            | "watcher_enable"
            | "watcher_disable"
    )
}

pub(crate) fn ensure_direct_mcp_call_allowed(name: &str, allow_mutation: bool) -> Result<()> {
    if mcp_tool_requires_direct_approval(name) && !allow_mutation {
        bail!(
            "MCP tool `{name}` changes local ROCm state; rerun `rocmd mcp-call {name}` with --allow-mutation only after an explicit user approval"
        );
    }
    Ok(())
}

pub(crate) fn handle_mcp_tool_call(paths: &AppPaths, params: &Value) -> Result<Value> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let arguments = params
        .get("arguments")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    match name {
        "examine" => {
            let examine = ExamineSummary::gather()?;
            let output = common::run_rocm_capture(&["examine"])?;
            let text = command_capture_text(&output);
            if output.exit_status == 0 {
                Ok(tool_success(text, json!(examine)))
            } else {
                Ok(tool_error(
                    text,
                    json!({
                        "examine": examine,
                        "argv": output.argv,
                        "exit_status": output.exit_status,
                        "stderr": output.stderr,
                    }),
                ))
            }
        }
        "bridge_snapshot" => {
            let snapshot = common::build_bridge_snapshot(paths)?;
            Ok(tool_success(
                format!(
                    "Captured bridge snapshot for {} / {} with default engine `{}`.",
                    snapshot.examine.os, snapshot.examine.arch, snapshot.examine.default_engine
                ),
                json!(snapshot),
            ))
        }
        "gpu_snapshot" => {
            let config = RocmCliConfig::load(paths).unwrap_or_default();
            let gpu = common::gather_gpu_snapshot_for_config(&config);
            let status = if !config.telemetry.local_inspection_enabled() {
                "GPU telemetry is disabled by rocm-cli config."
            } else if gpu.amd_smi_available {
                "Captured amd-smi GPU snapshot."
            } else {
                "amd-smi is unavailable on this host."
            };
            Ok(tool_success(status.to_owned(), json!(gpu)))
        }
        "engines" => {
            let engines = common::bridge_engine_inventory();
            Ok(tool_success(
                format!("Found {} engine entries.", engines.len()),
                json!({ "engines": engines }),
            ))
        }
        "services" => {
            let services = persistence::load_managed_services(paths)?;
            Ok(tool_success(
                format!("Found {} managed services.", services.len()),
                json!({ "services": services }),
            ))
        }
        "service_logs" => {
            let service_id = arguments
                .get("service_id")
                .and_then(Value::as_str)
                .context("service_logs requires `service_id`")?;
            let lines = arguments
                .get("lines")
                .and_then(Value::as_u64)
                .unwrap_or(80)
                .clamp(1, 500) as usize;
            let record = persistence::load_managed_services(paths)?
                .into_iter()
                .find(|service| service.service_id == service_id)
                .with_context(|| format!("managed service `{service_id}` not found"))?;
            let tail = read_tail_lines(&record.log_path, lines)?;
            Ok(tool_success(
                format!(
                    "Read the last {} line(s) from service `{}`.",
                    lines, record.service_id
                ),
                json!({
                    "service": record,
                    "lines": lines,
                    "tail": tail,
                }),
            ))
        }
        "automations" => {
            let event_limit = arguments
                .get("event_limit")
                .and_then(Value::as_u64)
                .unwrap_or(10)
                .clamp(1, 64) as usize;
            let runtime = AutomationRuntimeState::load(paths)?;
            let events = load_recent_automation_events(paths, event_limit)?;
            Ok(tool_success(
                format!(
                    "Loaded automation runtime and {} recent events.",
                    events.len()
                ),
                json!({
                    "runtime": runtime,
                    "recent_events": events,
                }),
            ))
        }
        "natural_language_plan" => {
            let request = arguments
                .get("request")
                .and_then(Value::as_str)
                .context("natural_language_plan requires `request`")?;
            let output = common::run_rocm_capture(&[request])?;
            Ok(tool_result_from_command(
                "Ran natural-language planning through `rocm`.",
                output,
                false,
            ))
        }
        "rocm_command" => {
            let argv = normalized_rocm_command_args(&arguments)?;
            ensure_rocm_command_is_read_only(&argv)?;
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = common::run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran read-only `rocm` command.",
                output,
                false,
            ))
        }
        "update_check" => {
            let output = common::run_rocm_capture(&["update"])?;
            Ok(tool_result_from_command(
                "Ran `rocm update`.",
                output,
                false,
            ))
        }
        "install_sdk_dry_run" => {
            let argv = build_install_sdk_args(&arguments, true)?;
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = common::run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran `rocm install sdk --dry-run`.",
                output,
                false,
            ))
        }
        "install_sdk" => {
            let argv = build_install_sdk_args(&arguments, false)?;
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = common::run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran `rocm install sdk`.",
                output,
                false,
            ))
        }
        "install_engine" => {
            let argv = build_install_engine_args(&arguments)?;
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = common::run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran `rocm engines install`.",
                output,
                false,
            ))
        }
        "launch_server" => {
            let argv = build_launch_server_args(&arguments)?;
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = common::run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran `rocm serve --managed`.",
                output,
                false,
            ))
        }
        "stop_server" => {
            let service_id = arguments
                .get("service_id")
                .and_then(Value::as_str)
                .context("stop_server requires `service_id`")?;
            let stopped = crate::service::stop_managed_service(paths, service_id)?;
            Ok(tool_success(
                format!("Stopped managed service `{service_id}`."),
                stopped,
            ))
        }
        "watcher_enable" => {
            let argv = build_watcher_enable_args(&arguments)?;
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = common::run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran `rocm automations enable`.",
                output,
                false,
            ))
        }
        "watcher_disable" => {
            let watcher = arguments
                .get("watcher")
                .and_then(Value::as_str)
                .context("watcher_disable requires `watcher`")?;
            let argv = [
                "automations".to_owned(),
                "disable".to_owned(),
                watcher.to_owned(),
            ];
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = common::run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran `rocm automations disable`.",
                output,
                false,
            ))
        }
        other => Ok(tool_error(
            format!("Unknown ROCm MCP tool `{other}`."),
            json!({ "tool": other }),
        )),
    }
}

fn tool_success(text: String, structured: Value) -> Value {
    json!({
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

fn tool_error(text: String, structured: Value) -> Value {
    json!({
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
        "structuredContent": structured,
        "isError": true,
    })
}

fn tool_result_from_command(prefix: &str, output: CommandCapture, is_error: bool) -> Value {
    let text = format!("{prefix}\n\n{}", command_capture_text(&output));
    json!({
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

fn read_tail_lines(path: &std::path::Path, limit: usize) -> Result<String> {
    let content =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut lines = VecDeque::with_capacity(limit);
    for line in content.lines() {
        if lines.len() == limit {
            lines.pop_front();
        }
        lines.push_back(line.to_owned());
    }
    Ok(lines.into_iter().collect::<Vec<_>>().join("\n"))
}

fn normalized_rocm_command_args(arguments: &serde_json::Map<String, Value>) -> Result<Vec<String>> {
    let values = arguments
        .get("args")
        .and_then(Value::as_array)
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
    if args
        .first()
        .is_some_and(|arg| arg.eq_ignore_ascii_case("comfy"))
    {
        args[0] = "comfyui".to_owned();
    }
    if args.is_empty() {
        bail!("rocm_command args should omit the leading `rocm` program name");
    }
    Ok(args)
}

fn ensure_rocm_command_is_read_only(args: &[String]) -> Result<()> {
    let first = args.first().map(|value| value.to_ascii_lowercase());
    let second = args.get(1).map(|value| value.to_ascii_lowercase());
    let read_only = match first.as_deref() {
        Some("examine" | "version" | "model" | "models" | "daemon" | "logs") => true,
        Some("update") => !args.iter().any(|arg| arg == "--apply"),
        Some("runtimes") => {
            second.as_deref().is_none_or(|value| value == "list")
                || (second
                    .as_deref()
                    .is_some_and(|value| value == "uninstall" || value == "remove")
                    && args.iter().any(|arg| arg == "--dry-run"))
        }
        Some("engines") => second.as_deref().is_some_and(|value| value == "list"),
        Some("services") => second
            .as_deref()
            .is_none_or(|value| matches!(value, "list" | "logs")),
        Some("automations") => second.as_deref().is_none_or(|value| value == "list"),
        Some("config") => second.as_deref() == Some("show"),
        Some("comfyui") => second
            .as_deref()
            .is_none_or(|value| matches!(value, "status" | "logs" | "log")),
        Some("uninstall") => args.iter().any(|arg| arg == "--dry-run"),
        // `storage report` (the default subcommand) only measures folders. The
        // two `remove-*` verbs delete, so they stay off the read-only list.
        Some("storage") => second.as_deref().is_none_or(|value| value == "report"),
        // `setup status` reports first-time setup state (read-only); `setup reset`
        // clears the completion/dismissal state and is mutating (it does not by
        // itself reopen onboarding). Mirrors the bin's rocm_command classifier so
        // the read-only allowlist is consistent across binaries.
        Some("setup") => second.as_deref().is_none_or(|value| value == "status"),
        // `remote targets` reads the local tailnet, `doctor` fetches another
        // machine's state and scores it here, `status` probes sessions that
        // already exist. None of them change anything on either machine.
        // `serve`, `attach` and `stop` start, publish or tear down, so they stay
        // off the list and go through the approval UI like any other mutation.
        Some("remote") => second
            .as_deref()
            .is_some_and(|value| matches!(value, "targets" | "doctor" | "status")),
        _ => false,
    };
    if read_only {
        return Ok(());
    }
    bail!(
        "rocm_command changes local ROCm state or is unsupported here; request it through the ROCm CLI approval UI instead"
    )
}

fn build_install_sdk_args(
    arguments: &serde_json::Map<String, Value>,
    dry_run: bool,
) -> Result<Vec<String>> {
    let channel = arguments
        .get("channel")
        .and_then(Value::as_str)
        .unwrap_or("release");
    let format = arguments
        .get("format")
        .and_then(Value::as_str)
        .unwrap_or("wheel");
    let prefix = arguments.get("prefix").and_then(Value::as_str);
    let version = arguments.get("version").and_then(Value::as_str);
    let build_date = arguments.get("build_date").and_then(Value::as_str);
    let allow_system_prefix = arguments
        .get("allow_system_prefix")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if version.is_some() && build_date.is_some() {
        bail!("install_sdk accepts either `version` or `build_date`, not both");
    }

    let mut argv = vec![
        "install".to_owned(),
        "sdk".to_owned(),
        "--channel".to_owned(),
        channel.to_owned(),
        "--format".to_owned(),
        format.to_owned(),
    ];
    if let Some(prefix) = prefix {
        let prefix_path = std::path::Path::new(prefix);
        if system_prefix_requires_ack(prefix_path) && !allow_system_prefix {
            bail!(
                "install_sdk prefix `{}` is outside the user home; require `allow_system_prefix=true` before using system paths",
                prefix_path.display()
            );
        }
        argv.push("--prefix".to_owned());
        argv.push(prefix.to_owned());
    }
    if let Some(version) = version {
        if version.trim().is_empty() {
            bail!("install_sdk `version` cannot be empty");
        }
        argv.push("--version".to_owned());
        argv.push(version.to_owned());
    }
    if let Some(build_date) = build_date {
        if build_date.trim().is_empty() {
            bail!("install_sdk `build_date` cannot be empty");
        }
        argv.push("--build-date".to_owned());
        argv.push(build_date.to_owned());
    }
    if dry_run {
        argv.push("--dry-run".to_owned());
    } else {
        // `run_rocm_capture_for_paths` spawns `rocm` with null stdin, so
        // `interactive_terminal()` is false in the child and an active default
        // managed runtime would make the approval gate refuse with "re-run with
        // `--approve-replacing-active-default`" — a flag no MCP caller of this
        // tool can supply.
        //
        // Not `--yes` itself: that flag carries a second, unrelated consent —
        // approving required system-package installs, which run `sudo`. This
        // spawn has no terminal, so it could never answer a sudo password
        // prompt; granting that consent would make the vLLM/OpenMPI step attempt
        // an install it cannot complete and abort the engine auto-install that
        // previously warned and continued. `--approve-replacing-active-default`
        // grants only the runtime-displacement consent the gate asks for.
        //
        // Consent is not bypassed: `install_sdk` is in
        // `mcp_tool_requires_direct_approval`, so a direct `rocmd mcp-call`
        // needs `--allow-mutation` after an explicit user approval, and over the
        // MCP protocol the tool is annotated `destructiveHint` for the client's
        // approval UI. Mirrors the chat/MCP arm in `apps/rocm`. The dry-run
        // branch never reaches the gate (it returns earlier), so it stays bare.
        argv.push("--approve-replacing-active-default".to_owned());
    }
    Ok(argv)
}

fn build_install_engine_args(arguments: &serde_json::Map<String, Value>) -> Result<Vec<String>> {
    let engine = arguments
        .get("engine")
        .and_then(Value::as_str)
        .context("install_engine requires `engine`")?;
    let runtime_id = arguments
        .get("runtime_id")
        .and_then(Value::as_str)
        .unwrap_or("therock-release");
    let python_version = arguments.get("python_version").and_then(Value::as_str);
    let reinstall = arguments
        .get("reinstall")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let mut argv = vec![
        "engines".to_owned(),
        "install".to_owned(),
        engine.to_owned(),
        "--runtime-id".to_owned(),
        runtime_id.to_owned(),
    ];
    if let Some(python_version) = python_version {
        argv.push("--python-version".to_owned());
        argv.push(python_version.to_owned());
    }
    if reinstall {
        argv.push("--reinstall".to_owned());
    }
    Ok(argv)
}

fn build_launch_server_args(arguments: &serde_json::Map<String, Value>) -> Result<Vec<String>> {
    let model = arguments
        .get("model")
        .and_then(Value::as_str)
        .context("launch_server requires `model`")?;
    let host = arguments
        .get("host")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_LOCAL_HOST);
    let allow_public_bind = arguments
        .get("allow_public_bind")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if !is_loopback_host(host) && !allow_public_bind {
        bail!(
            "launch_server host `{host}` is not loopback; require `allow_public_bind=true` before binding a non-local interface"
        );
    }

    let mut argv = vec!["serve".to_owned(), model.to_owned(), "--managed".to_owned()];
    if let Some(engine) = arguments.get("engine").and_then(Value::as_str) {
        argv.push("--engine".to_owned());
        argv.push(engine.to_owned());
    }
    if let Some(device) = arguments.get("device").and_then(Value::as_str) {
        argv.push("--device".to_owned());
        argv.push(device.to_owned());
    }
    if let Some(runtime_id) = arguments.get("runtime_id").and_then(Value::as_str) {
        argv.push("--runtime-id".to_owned());
        argv.push(runtime_id.to_owned());
    }
    if let Some(env_id) = arguments.get("env_id").and_then(Value::as_str) {
        argv.push("--env-id".to_owned());
        argv.push(env_id.to_owned());
    }
    argv.push("--host".to_owned());
    argv.push(host.to_owned());
    if allow_public_bind {
        argv.push("--allow-public-bind".to_owned());
    }
    if let Some(port) = arguments.get("port").and_then(Value::as_u64) {
        argv.push("--port".to_owned());
        argv.push(port.to_string());
    }
    Ok(argv)
}

fn build_watcher_enable_args(arguments: &serde_json::Map<String, Value>) -> Result<Vec<String>> {
    let watcher = arguments
        .get("watcher")
        .and_then(Value::as_str)
        .context("watcher_enable requires `watcher`")?;
    let mut argv = vec![
        "automations".to_owned(),
        "enable".to_owned(),
        watcher.to_owned(),
    ];
    if let Some(mode) = arguments.get("mode").and_then(Value::as_str) {
        argv.push("--mode".to_owned());
        argv.push(mode.to_owned());
    }
    Ok(argv)
}

/// Inverse of [`rocm_engine_protocol::is_public_bind_host`], which owns the
/// policy so `rocm` and `rocmd` never classify the same host differently.
fn is_loopback_host(host: &str) -> bool {
    !rocm_engine_protocol::is_public_bind_host(host)
}

fn system_prefix_requires_ack(prefix: &std::path::Path) -> bool {
    match rocm_core::runtime_home_dir() {
        Some(home) => !rocm_core::runtime_path_is_same_or_inside(prefix, &home),
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::workspace_test_artifact_dir;
    use std::fs;
    use std::path::PathBuf;

    fn unique_test_path(label: &str) -> PathBuf {
        let root = workspace_test_artifact_dir();
        fs::create_dir_all(&root).expect("create workspace-local test dir");
        root.join(label)
    }

    #[test]
    fn remote_read_only_verbs_are_allowed_and_mutating_ones_are_not() {
        let allow = |args: &[&str]| {
            let owned = args.iter().map(|a| (*a).to_owned()).collect::<Vec<_>>();
            super::ensure_rocm_command_is_read_only(&owned)
        };

        // These read: the local tailnet, another machine's state, sessions that
        // already exist. Rejecting them made the whole family unusable here even
        // though none of them change anything.
        for args in [
            &["remote", "targets"][..],
            &["remote", "targets", "--tag", "gpu"][..],
            &["remote", "doctor", "gpu-box"][..],
            &["remote", "status"][..],
        ] {
            allow(args).unwrap_or_else(|error| panic!("{args:?} should be read-only: {error:#}"));
        }

        // These start, publish or tear down, so they go through approval.
        for args in [
            &["remote", "serve", "gpu-box", "a-model"][..],
            &["remote", "attach", "sess"][..],
            &["remote", "stop", "sess"][..],
            &["remote"][..],
        ] {
            assert!(allow(args).is_err(), "{args:?} must not be read-only");
        }
    }
    #[test]
    fn rocm_mcp_tools_include_bridge_gaps() {
        let tools = rocm_mcp_tools();
        let names = tools
            .iter()
            .filter_map(|tool| tool.get("name").and_then(Value::as_str).map(str::to_owned))
            .collect::<Vec<_>>();
        assert!(names.contains(&"gpu_snapshot".to_owned()));
        assert!(names.contains(&"service_logs".to_owned()));
        assert!(names.contains(&"natural_language_plan".to_owned()));
        assert!(names.contains(&"rocm_command".to_owned()));
        assert!(names.contains(&"install_sdk".to_owned()));
        assert!(names.contains(&"install_engine".to_owned()));
        assert!(names.contains(&"launch_server".to_owned()));
        assert!(names.contains(&"stop_server".to_owned()));
        assert!(names.contains(&"watcher_enable".to_owned()));
        assert!(names.contains(&"watcher_disable".to_owned()));
        let automations = tools
            .iter()
            .find(|tool| tool.get("name").and_then(Value::as_str) == Some("automations"))
            .expect("automations tool should be present");
        assert!(
            automations
                .get("description")
                .and_then(Value::as_str)
                .is_some_and(|description| description.contains("local webhook events"))
        );
    }

    #[test]
    fn direct_mcp_call_requires_approval_for_every_mutating_tool() {
        for tool in rocm_mcp_tools() {
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .expect("tool should have a name");
            let read_only = tool
                .get("annotations")
                .and_then(|annotations| annotations.get("readOnlyHint"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            assert_eq!(
                mcp_tool_requires_direct_approval(name),
                !read_only,
                "hidden direct MCP helper approval classification drifted for `{name}`"
            );
        }
    }

    #[test]
    fn direct_mcp_call_guard_blocks_mutation_without_explicit_ack() {
        ensure_direct_mcp_call_allowed("examine", false)
            .expect("read-only direct MCP helper calls should not need mutation approval");

        let error = ensure_direct_mcp_call_allowed("install_sdk", false)
            .expect_err("mutating direct MCP helper calls should require approval");
        assert!(error.to_string().contains("--allow-mutation"), "{error:#}");

        ensure_direct_mcp_call_allowed("install_sdk", true)
            .expect("explicitly approved direct MCP mutation should pass the helper guard");
    }

    #[test]
    fn rocm_command_helper_allows_only_read_only_rocm_commands() -> Result<()> {
        let status_args = normalized_rocm_command_args(
            serde_json::json!({
                "args": ["rocm", "comfy", "status"]
            })
            .as_object()
            .expect("json object"),
        )?;
        assert_eq!(status_args, vec!["comfyui".to_owned(), "status".to_owned()]);
        ensure_rocm_command_is_read_only(&status_args).expect("ComfyUI status should be read-only");

        let log_args = normalized_rocm_command_args(
            serde_json::json!({
                "args": ["comfyui", "logs"]
            })
            .as_object()
            .expect("json object"),
        )?;
        ensure_rocm_command_is_read_only(&log_args).expect("ComfyUI logs should be read-only");

        let install_args = normalized_rocm_command_args(
            serde_json::json!({
                "args": ["comfyui", "install"]
            })
            .as_object()
            .expect("json object"),
        )?;
        let error = ensure_rocm_command_is_read_only(&install_args)
            .expect_err("ComfyUI install must go through approval");
        assert!(error.to_string().contains("approval UI"));

        let shell_args = normalized_rocm_command_args(
            serde_json::json!({
                "args": ["powershell", "-Command", "whoami"]
            })
            .as_object()
            .expect("json object"),
        )?;
        let error = ensure_rocm_command_is_read_only(&shell_args)
            .expect_err("non-rocm shell commands should be rejected");
        assert!(error.to_string().contains("approval UI"));
        Ok(())
    }

    #[test]
    fn rocm_command_helper_treats_setup_status_as_read_only_and_reset_as_mutating() -> Result<()> {
        // Mirrors the bin's rocm_command classifier so `setup status` is read-only
        // on every binary's tool surface while `setup reset` stays approval-gated.
        let bare_args = normalized_rocm_command_args(
            serde_json::json!({ "args": ["setup"] })
                .as_object()
                .expect("json object"),
        )?;
        ensure_rocm_command_is_read_only(&bare_args).expect("bare setup should be read-only");

        let status_args = normalized_rocm_command_args(
            serde_json::json!({ "args": ["setup", "status"] })
                .as_object()
                .expect("json object"),
        )?;
        ensure_rocm_command_is_read_only(&status_args).expect("setup status should be read-only");

        let reset_args = normalized_rocm_command_args(
            serde_json::json!({ "args": ["setup", "reset"] })
                .as_object()
                .expect("json object"),
        )?;
        let error = ensure_rocm_command_is_read_only(&reset_args)
            .expect_err("setup reset must go through approval");
        assert!(error.to_string().contains("approval UI"));
        Ok(())
    }

    #[test]
    fn rocm_command_helper_treats_runtimes_uninstall_dry_run_as_read_only() -> Result<()> {
        // Mirrors the bin's chat_rocm_command_action_from_args classifier so a
        // dry-run preview stays read-only on every binary's tool surface while
        // an actual uninstall/remove still requires approval.
        for verb in ["uninstall", "remove"] {
            let dry_run_args = normalized_rocm_command_args(
                serde_json::json!({ "args": ["runtimes", verb, "--dry-run"] })
                    .as_object()
                    .expect("json object"),
            )?;
            ensure_rocm_command_is_read_only(&dry_run_args)
                .unwrap_or_else(|_| panic!("runtimes {verb} --dry-run should be read-only"));

            let mutating_args = normalized_rocm_command_args(
                serde_json::json!({ "args": ["runtimes", verb] })
                    .as_object()
                    .expect("json object"),
            )?;
            let error = match ensure_rocm_command_is_read_only(&mutating_args) {
                Ok(()) => panic!("runtimes {verb} without --dry-run must go through approval"),
                Err(error) => error,
            };
            assert!(error.to_string().contains("approval UI"));
        }
        Ok(())
    }

    #[test]
    fn storage_report_is_read_only_but_removal_is_not() -> Result<()> {
        for args in [vec!["storage"], vec!["storage", "report"]] {
            let normalized = normalized_rocm_command_args(
                serde_json::json!({ "args": args })
                    .as_object()
                    .expect("json object"),
            )?;
            ensure_rocm_command_is_read_only(&normalized)
                .unwrap_or_else(|_| panic!("storage {args:?} only measures folders"));
        }

        for verb in ["remove-old-installs", "remove-downloads"] {
            let normalized = normalized_rocm_command_args(
                serde_json::json!({ "args": ["storage", verb] })
                    .as_object()
                    .expect("json object"),
            )?;
            let error = ensure_rocm_command_is_read_only(&normalized)
                .expect_err("storage removal must go through approval");
            assert!(error.to_string().contains("approval UI"));
        }
        Ok(())
    }

    #[test]
    fn read_tail_lines_returns_last_lines_only() -> Result<()> {
        let path = unique_test_path(&format!(
            "rocmd-tail-test-{}-{}.log",
            std::process::id(),
            unix_time_millis()
        ));
        fs::write(&path, "line1\nline2\nline3\nline4\n")?;
        let tail = read_tail_lines(&path, 2)?;
        fs::remove_file(&path)?;
        assert_eq!(tail, "line3\nline4");
        Ok(())
    }

    #[test]
    fn launch_server_rejects_public_bind_without_ack() {
        let arguments = serde_json::Map::from_iter([
            ("model".to_owned(), Value::String("tiny-gpt2".to_owned())),
            ("host".to_owned(), Value::String("0.0.0.0".to_owned())),
        ]);
        let error = build_launch_server_args(&arguments).unwrap_err();
        assert!(
            error.to_string().contains("allow_public_bind=true"),
            "{error:#}"
        );
    }

    #[test]
    fn launch_server_forwards_public_bind_ack() -> Result<()> {
        let arguments = serde_json::Map::from_iter([
            ("model".to_owned(), Value::String("tiny-gpt2".to_owned())),
            ("host".to_owned(), Value::String("0.0.0.0".to_owned())),
            ("allow_public_bind".to_owned(), Value::Bool(true)),
        ]);
        let args = build_launch_server_args(&arguments)?;
        assert!(args.contains(&"--allow-public-bind".to_owned()));
        Ok(())
    }

    #[test]
    fn install_sdk_rejects_system_prefix_without_ack() {
        let arguments = serde_json::Map::from_iter([(
            "prefix".to_owned(),
            Value::String("/opt/rocm".to_owned()),
        )]);
        let error = build_install_sdk_args(&arguments, false).unwrap_err();
        assert!(
            error.to_string().contains("allow_system_prefix=true"),
            "{error:#}"
        );
    }

    /// The test above only ever hands `system_prefix_requires_ack` an
    /// already-canonical path, so it cannot catch the bug this crate's fix
    /// addresses: a `..`-respelled prefix that escapes `$HOME` used to compare
    /// equal to a path still inside it (`Path::ancestors()` treats `..` as an
    /// ordinary component), so acknowledgement was never required. Drive the
    /// same check with a prefix built by walking `..` out of the real home
    /// directory, which is exactly the shape the original bug let through.
    #[test]
    #[cfg(unix)]
    fn install_sdk_rejects_system_prefix_reached_by_escaping_home() {
        let home = rocm_core::runtime_home_dir().expect("a home directory");
        let escaped_prefix = format!("{}/../../usr", home.display());

        let arguments =
            serde_json::Map::from_iter([("prefix".to_owned(), Value::String(escaped_prefix))]);
        let error = build_install_sdk_args(&arguments, false).unwrap_err();
        assert!(
            error.to_string().contains("allow_system_prefix=true"),
            "{error:#}"
        );
    }

    /// The `install_sdk` MCP tool spawns `rocm` with null stdin, so a real
    /// install over an active default managed runtime would hit the approval
    /// gate's non-interactive refusal and bail asking for a flag no MCP caller
    /// can pass. The real-install argv must therefore carry the consent flag;
    /// the dry-run argv must not, because a dry run never reaches the gate and
    /// the flag there would claim an approval the caller did not give.
    ///
    /// It must be `--approve-replacing-active-default` and never `--yes`:
    /// `--yes` additionally approves running `sudo` for required system
    /// packages, and a null-stdin spawn has no terminal on which that password
    /// prompt could be answered.
    #[test]
    fn install_sdk_real_install_args_approve_only_the_runtime_replacement() -> Result<()> {
        let arguments = serde_json::Map::new();

        let real = build_install_sdk_args(&arguments, false)?;
        assert!(
            real.contains(&"--approve-replacing-active-default".to_owned()),
            "real install argv must approve the replacement for the null-stdin spawn: {real:?}"
        );
        assert!(
            !real.contains(&"--yes".to_owned()),
            "real install argv must not grant the system-package consent it cannot answer: {real:?}"
        );
        assert!(
            !real.contains(&"--dry-run".to_owned()),
            "real install argv must not be a dry run: {real:?}"
        );

        let dry = build_install_sdk_args(&arguments, true)?;
        assert!(
            !dry.contains(&"--approve-replacing-active-default".to_owned())
                && !dry.contains(&"--yes".to_owned()),
            "dry-run argv must not carry a consent flag: {dry:?}"
        );
        assert!(
            dry.contains(&"--dry-run".to_owned()),
            "dry-run argv must carry --dry-run: {dry:?}"
        );
        Ok(())
    }

    #[test]
    fn install_sdk_forwards_requested_build_date_and_rejects_conflict() -> Result<()> {
        let arguments = serde_json::Map::from_iter([(
            "build_date".to_owned(),
            Value::String("2026-06-05".to_owned()),
        )]);
        let argv = build_install_sdk_args(&arguments, true)?;
        assert_eq!(
            argv,
            vec![
                "install".to_owned(),
                "sdk".to_owned(),
                "--channel".to_owned(),
                "release".to_owned(),
                "--format".to_owned(),
                "wheel".to_owned(),
                "--build-date".to_owned(),
                "2026-06-05".to_owned(),
                "--dry-run".to_owned(),
            ]
        );

        let conflicting = serde_json::Map::from_iter([
            (
                "version".to_owned(),
                Value::String("7.13.0a20260605".to_owned()),
            ),
            (
                "build_date".to_owned(),
                Value::String("2026-06-05".to_owned()),
            ),
        ]);
        let error = build_install_sdk_args(&conflicting, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("either `version` or `build_date`"));
        Ok(())
    }

    #[test]
    fn watcher_enable_builds_mode_args() -> Result<()> {
        let arguments = serde_json::Map::from_iter([
            (
                "watcher".to_owned(),
                Value::String("server-recover".to_owned()),
            ),
            ("mode".to_owned(), Value::String("contained".to_owned())),
        ]);
        let argv = build_watcher_enable_args(&arguments)?;
        assert_eq!(
            argv,
            vec![
                "automations".to_owned(),
                "enable".to_owned(),
                "server-recover".to_owned(),
                "--mode".to_owned(),
                "contained".to_owned()
            ]
        );
        Ok(())
    }
}
