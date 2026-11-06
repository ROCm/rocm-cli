// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use crate::common;
use crate::host_snapshot;
use crate::mcp_argv::{
    build_install_engine_args, build_install_sdk_args, build_launch_server_args,
    build_watcher_enable_args, ensure_rocm_command_is_read_only, normalized_rocm_command_args,
};
use crate::persistence;
use anyhow::{Context, Result, bail};
#[cfg(test)]
use rocm_core::unix_time_millis;
use rocm_core::{
    AppPaths, AutomationRuntimeState, CommandCapture, ExamineSummary, RocmCliConfig,
    command_capture_text, load_recent_automation_events,
};
use serde_json::Value;
use serde_json::json;
use std::io::{self, BufRead, Write};
use std::time::Duration;

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
            let output = run_rocm_capture(&["examine"])?;
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
            let snapshot = host_snapshot::build_bridge_snapshot(paths)?;
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
            let gpu = host_snapshot::gather_gpu_snapshot_for_config(&config);
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
            let engines = host_snapshot::bridge_engine_inventory();
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
            let output = run_rocm_capture(&[request])?;
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
            let output = run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran read-only `rocm` command.",
                output,
                false,
            ))
        }
        "update_check" => {
            let output = run_rocm_capture(&["update"])?;
            Ok(tool_result_from_command(
                "Ran `rocm update`.",
                output,
                false,
            ))
        }
        "install_sdk_dry_run" => {
            let argv = build_install_sdk_args(&arguments, true)?;
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran `rocm install sdk --dry-run`.",
                output,
                false,
            ))
        }
        "install_sdk" => {
            let argv = build_install_sdk_args(&arguments, false)?;
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran `rocm install sdk`.",
                output,
                false,
            ))
        }
        "install_engine" => {
            let argv = build_install_engine_args(&arguments)?;
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = run_rocm_capture(&refs)?;
            Ok(tool_result_from_command(
                "Ran `rocm engines install`.",
                output,
                false,
            ))
        }
        "launch_server" => {
            let argv = build_launch_server_args(&arguments)?;
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = run_rocm_capture(&refs)?;
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
            Ok(mcp_stop_server_reply(service_id, stopped))
        }
        "watcher_enable" => {
            let argv = build_watcher_enable_args(&arguments)?;
            let refs = argv.iter().map(String::as_str).collect::<Vec<_>>();
            let output = run_rocm_capture(&refs)?;
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
            let output = run_rocm_capture(&refs)?;
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

fn run_rocm_capture(args: &[&str]) -> Result<CommandCapture> {
    let paths = AppPaths::discover()?;
    common::run_rocm_capture_for_paths(&paths, args, Duration::from_mins(2))
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

/// The `stop_server` MCP tool's answer for a [`crate::service::stop_managed_service`] report.
fn mcp_stop_server_reply(service_id: &str, stopped: Value) -> Value {
    if crate::service::stop_report_confirmed(&stopped) {
        tool_success(format!("Stopped managed service `{service_id}`."), stopped)
    } else {
        // Not a success: a recorded process may still be running and holding
        // the device. `stop_managed_service` keeps the record's PIDs and any
        // endpoint key for exactly that case, and `pid_outcomes` in the
        // structured result names the PID that could not be confirmed.
        tool_error(
            format!(
                "Could not confirm that managed service `{service_id}` stopped; a recorded process may still be running. Its record, which names the processes it last recorded, and any endpoint key, which the server may still be enforcing, were kept; see `pid_outcomes`."
            ),
            stopped,
        )
    }
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

/// The last `limit` lines of a service log joined with newlines, read through
/// the shared bounded tail routine.
fn read_tail_lines(path: &std::path::Path, limit: usize) -> Result<String> {
    let lines = rocm_core::tail_lines(path, limit, rocm_core::LOG_VIEW_MAX_TAIL_READ)
        .with_context(|| format!("failed to read {}", path.display()))?;
    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::load_service_record;
    use crate::test_support::{
        assert_unconfirmed_stop_kept_the_service, seed_keyed_service, stop_with_outcome,
        temp_app_paths, workspace_test_artifact_dir,
    };
    use std::fs;
    use std::path::PathBuf;

    fn unique_test_path(label: &str) -> PathBuf {
        let root = workspace_test_artifact_dir();
        fs::create_dir_all(&root).expect("create workspace-local test dir");
        root.join(label)
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

    fn tail_fixture(label: &str, bytes: &[u8]) -> Result<PathBuf> {
        let path = unique_test_path(&format!(
            "rocmd-tail-{label}-{}.log",
            crate::test_support::unique_suffix()
        ));
        fs::write(&path, bytes)?;
        Ok(path)
    }

    /// 5 MiB: a marked first line, 5120 lines of 1 KiB, a marked last line.
    fn large_log_with_head_marker() -> Vec<u8> {
        let mut bytes = b"HEAD-MARKER\n".to_vec();
        let filler = format!("{}\n", "x".repeat(1023));
        for _ in 0..5120 {
            bytes.extend_from_slice(filler.as_bytes());
        }
        bytes.extend_from_slice(b"TAIL-MARKER\n");
        bytes
    }

    #[test]
    fn read_tail_lines_a_large_file_is_read_through_a_bounded_window() -> Result<()> {
        // A line far outside the tail window is not returned however large the
        // limit: the log is not read whole into one string.
        let path = tail_fixture("large", &large_log_with_head_marker())?;

        let tail = read_tail_lines(&path, 10_000)?;
        fs::remove_file(&path)?;

        assert!(!tail.lines().any(|line| line == "HEAD-MARKER"));
        assert!(tail.ends_with("\nTAIL-MARKER"));
        assert!(tail.lines().count() < 5122);
        assert!(tail.lines().count() > 3500, "the window holds about 4 MiB");
        Ok(())
    }

    #[test]
    fn read_tail_lines_shows_a_sixteen_bit_log_as_lossy_text_instead_of_failing() -> Result<()> {
        // The tool reads the child's own bytes as UTF-8; a UTF-16 log is not
        // detected here (only the ComfyUI log view does that), but it no
        // longer fails the whole tool call.
        let mut bytes = vec![0xff, 0xfe];
        for unit in "[INFO] one\r\n".encode_utf16() {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }
        let path = tail_fixture("utf16", &bytes)?;

        let tail = read_tail_lines(&path, 5)?;
        fs::remove_file(&path)?;

        assert!(
            tail.starts_with("\u{FFFD}\u{FFFD}["),
            "the byte order mark is replaced, the rest kept: {tail:?}"
        );
        Ok(())
    }

    #[test]
    fn read_tail_lines_keeps_ansi_colour_codes() -> Result<()> {
        let path = tail_fixture("ansi", b"\x1b[32mgreen\x1b[0m done\n")?;

        let tail = read_tail_lines(&path, 5)?;
        fs::remove_file(&path)?;

        assert_eq!(tail, "\u{1b}[32mgreen\u{1b}[0m done");
        Ok(())
    }

    #[test]
    fn read_tail_lines_decodes_invalid_eight_bit_text_lossily() -> Result<()> {
        let path = tail_fixture("invalid", b"ok\nbad\xffbyte\nlast\n")?;

        let tail = read_tail_lines(&path, 5)?;
        fs::remove_file(&path)?;

        assert_eq!(tail, "ok\nbad\u{FFFD}byte\nlast");
        Ok(())
    }

    #[test]
    fn read_tail_lines_missing_file_is_a_labelled_read_error() {
        let path = unique_test_path(&format!(
            "rocmd-tail-missing-{}.log",
            crate::test_support::unique_suffix()
        ));

        let error = read_tail_lines(&path, 5).expect_err("a missing log must not read");

        assert_eq!(
            error.to_string(),
            format!("failed to read {}", path.display())
        );
        assert!(
            format!("{error:#}").contains(&format!("failed to open {}", path.display())),
            "the cause names the failing step: {error:#}"
        );
    }

    #[test]
    fn read_tail_lines_limit_zero_returns_nothing() -> Result<()> {
        // With limit 0 the old reader never evicted and returned the whole
        // file. The tool handler clamps the limit to 1..=500, so only a
        // direct call could see it.
        let path = tail_fixture("zero", b"a\nb\nc\n")?;

        let tail = read_tail_lines(&path, 0)?;
        fs::remove_file(&path)?;

        assert_eq!(tail, "");
        Ok(())
    }

    #[test]
    fn read_tail_lines_returns_every_line_when_fewer_than_the_limit() -> Result<()> {
        let path = tail_fixture("few", b"a\nb\n")?;

        let tail = read_tail_lines(&path, 50)?;
        fs::remove_file(&path)?;

        assert_eq!(tail, "a\nb");
        Ok(())
    }

    #[test]
    fn read_tail_lines_keeps_a_final_line_without_a_trailing_newline() -> Result<()> {
        let path = tail_fixture("no-newline", b"a\nb\nc")?;

        let tail = read_tail_lines(&path, 2)?;
        fs::remove_file(&path)?;

        assert_eq!(tail, "b\nc");
        Ok(())
    }

    #[test]
    fn read_tail_lines_strips_crlf_line_endings() -> Result<()> {
        let path = tail_fixture("crlf", b"a\r\nb\r\n")?;

        let tail = read_tail_lines(&path, 5)?;
        fs::remove_file(&path)?;

        assert_eq!(tail, "a\nb");
        Ok(())
    }

    fn mcp_stop_server(paths: &AppPaths, service_id: &str) -> Result<Value> {
        handle_mcp_tool_call(
            paths,
            &json!({
                "name": "stop_server",
                "arguments": { "service_id": service_id },
            }),
        )
    }

    fn mcp_result_text(value: &Value) -> &str {
        value
            .get("content")
            .and_then(Value::as_array)
            .and_then(|content| content.first())
            .and_then(|entry| entry.get("text"))
            .and_then(Value::as_str)
            .unwrap_or_default()
    }

    /// The MCP tool used to answer "Stopped managed service" with
    /// `isError: false` whatever the stop found. A client acting on that would
    /// assume the device was free while the engine might still hold it.
    #[test]
    fn mcp_stop_server_reports_an_unconfirmed_stop_as_an_error() -> Result<()> {
        let (root, paths) = temp_app_paths("mcp-stop-unconfirmed");
        paths.ensure()?;
        let service_id = "svc-mcp-stop-unconfirmed";
        let key_path = seed_keyed_service(&paths, service_id, 11446)?;

        // The handler is `stop_managed_service` feeding `mcp_stop_server_reply`;
        // the confirmed-stop test below drives that whole handler for real.
        let result = stop_with_outcome(&paths, service_id, rocm_core::TerminationOutcome::TimedOut)
            .map(|stopped| mcp_stop_server_reply(service_id, stopped));
        let key_kept = key_path.exists();
        let reloaded = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let value = result?;
        assert_eq!(
            value.get("isError").and_then(Value::as_bool),
            Some(true),
            "{value}"
        );
        let text = mcp_result_text(&value);
        assert!(
            text.starts_with("Could not confirm that managed service"),
            "{text}"
        );
        assert!(!text.contains("Stopped managed service"), "{text}");
        assert_eq!(
            value
                .pointer("/structuredContent/stopped")
                .and_then(Value::as_bool),
            Some(false),
            "{value}"
        );
        assert_eq!(
            value
                .pointer("/structuredContent/pid_outcomes/0/outcome")
                .and_then(Value::as_str),
            Some("timed_out"),
            "the PID the message points at must be named: {value}"
        );
        // The message claims the PIDs and key were kept; prove it.
        assert_unconfirmed_stop_kept_the_service(&reloaded?, key_kept);
        Ok(())
    }

    #[test]
    fn mcp_stop_server_reports_a_confirmed_stop_as_success() -> Result<()> {
        let (root, paths) = temp_app_paths("mcp-stop-confirmed");
        paths.ensure()?;
        let service_id = "svc-mcp-stop-confirmed";
        // Not running, so the real stop path confirms it without signalling.
        let key_path = seed_keyed_service(&paths, service_id, 11447)?;

        let result = mcp_stop_server(&paths, service_id);
        let key_removed = !key_path.exists();
        let reloaded = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let value = result?;
        assert_eq!(
            value.get("isError").and_then(Value::as_bool),
            Some(false),
            "{value}"
        );
        assert_eq!(
            mcp_result_text(&value),
            format!("Stopped managed service `{service_id}`.")
        );
        let reloaded = reloaded?;
        assert_eq!(reloaded.status, "stopped");
        assert_eq!(reloaded.supervisor_pid, 0);
        assert!(key_removed, "a confirmed stop must drop the endpoint key");
        Ok(())
    }
}
