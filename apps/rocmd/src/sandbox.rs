// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use crate::artifact_prefetch::prefetch_artifact_value_with_policy;
use crate::common;
use crate::persistence::load_managed_services;
use crate::sandbox_tool::{SandboxToolArg, SandboxToolPolicy};
use crate::service::{stop_managed_service, stop_report_confirmed};
use crate::supervise::restart_managed_service;
use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, AuditEventRecord, CommandCapture, ExamineSummary, append_audit_event,
    resolve_model_recipe_artifact, unix_time_millis,
};
use serde_json::Value;
use serde_json::json;
#[cfg(any(test, target_os = "linux"))]
use std::fs;
#[cfg(target_os = "linux")]
use std::process::{Command as ProcessCommand, Stdio};
use std::time::Duration;

pub(crate) fn run_sandbox_runner(
    paths: &AppPaths,
    tool: SandboxToolArg,
    service_id: Option<String>,
    artifact_ref: Option<String>,
    message: Option<String>,
    allow_native_fallback: bool,
    policy: SandboxToolPolicy,
) -> Result<Value> {
    #[cfg(target_os = "linux")]
    {
        if command_available("bwrap") {
            return run_bubblewrap_sandbox(paths, tool, service_id, artifact_ref, message, policy);
        }
    }

    if allow_native_fallback {
        return run_native_restricted_sandbox(
            paths,
            tool,
            service_id,
            artifact_ref,
            message,
            policy,
        );
    }

    bail!(
        "isolated sandbox runner is unavailable on this host; pass --allow-native-fallback to run only the restricted internal tool API"
    )
}

#[cfg(target_os = "linux")]
fn run_bubblewrap_sandbox(
    paths: &AppPaths,
    tool: SandboxToolArg,
    service_id: Option<String>,
    artifact_ref: Option<String>,
    message: Option<String>,
    policy: SandboxToolPolicy,
) -> Result<Value> {
    paths.ensure()?;

    let current_exe = std::env::current_exe().context("failed to resolve rocmd executable")?;
    let exe_dir = current_exe
        .parent()
        .context("rocmd executable has no parent directory")?
        .to_path_buf();

    let mut command = ProcessCommand::new("bwrap");
    command
        .arg("--die-with-parent")
        .arg("--new-session")
        .arg("--unshare-ipc")
        .arg("--unshare-uts")
        .arg("--proc")
        .arg("/proc")
        .arg("--dev")
        .arg("/dev")
        .arg("--tmpfs")
        .arg("/tmp")
        .arg("--tmpfs")
        .arg("/run");
    if !(matches!(tool, SandboxToolArg::PrefetchArtifact) && policy.allow_artifact_download) {
        command.arg("--unshare-net");
    }

    for path in ["/usr", "/bin", "/lib", "/lib64", "/etc"] {
        let path = std::path::Path::new(path);
        if path.exists() {
            command.arg("--ro-bind").arg(path).arg(path);
        }
    }

    command.arg("--ro-bind").arg(&exe_dir).arg(&exe_dir);
    bind_app_path_for_sandbox(&mut command, &paths.config_dir, false)?;
    bind_app_path_for_sandbox(&mut command, &paths.data_dir, tool.writes_data())?;
    bind_app_path_for_sandbox(&mut command, &paths.cache_dir, tool.writes_cache())?;
    append_sandbox_tool_command_args(
        &mut command,
        &current_exe,
        tool,
        service_id.as_deref(),
        artifact_ref.as_deref(),
        message.as_deref(),
        policy,
    );

    let output =
        rocm_core::process::run_with_timeout(command, Duration::from_mins(1), "sandbox process")?;
    let value = parse_sandbox_child_output(output, "bubblewrap sandbox")?;
    record_sandbox_audit(paths, tool, "bubblewrap", true, service_id.as_deref())?;
    Ok(sandbox_report(tool, "bubblewrap", value))
}

#[cfg(target_os = "linux")]
fn bind_app_path_for_sandbox(
    command: &mut ProcessCommand,
    path: &std::path::Path,
    writable: bool,
) -> Result<()> {
    fs::create_dir_all(path).with_context(|| format!("failed to create {}", path.display()))?;
    if writable {
        command.arg("--bind");
    } else {
        command.arg("--ro-bind");
    }
    command.arg(path).arg(path);
    Ok(())
}

fn run_native_restricted_sandbox(
    paths: &AppPaths,
    tool: SandboxToolArg,
    service_id: Option<String>,
    artifact_ref: Option<String>,
    message: Option<String>,
    policy: SandboxToolPolicy,
) -> Result<Value> {
    let value = run_sandbox_tool(
        paths,
        tool,
        service_id.clone(),
        artifact_ref,
        message,
        policy,
    )?;
    record_sandbox_audit(
        paths,
        tool,
        "native_restricted",
        true,
        service_id.as_deref(),
    )?;
    Ok(sandbox_report(tool, "native_restricted", value))
}

pub(crate) fn run_sandbox_tool(
    paths: &AppPaths,
    tool: SandboxToolArg,
    service_id: Option<String>,
    artifact_ref: Option<String>,
    message: Option<String>,
    policy: SandboxToolPolicy,
) -> Result<Value> {
    match tool {
        SandboxToolArg::CheckUpdates => {
            let output =
                common::run_rocm_capture_for_paths(paths, &["update"], Duration::from_mins(1))?;
            Ok(sandbox_check_updates_value(output))
        }
        SandboxToolArg::DriverPlan => {
            let output = common::run_rocm_capture_for_paths(
                paths,
                &["install", "driver", "--dkms", "--dry-run"],
                Duration::from_mins(1),
            )?;
            Ok(sandbox_driver_plan_value(output))
        }
        SandboxToolArg::ExamineSnapshot => {
            let examine = ExamineSummary::gather()?;
            Ok(json!({
                "tool": tool.as_cli_value(),
                "status": "captured",
                "mutating": false,
                "examine": examine,
            }))
        }
        SandboxToolArg::ListServers => {
            let services = load_managed_services(paths)?;
            Ok(json!({
                "tool": tool.as_cli_value(),
                "status": "listed",
                "mutating": false,
                "count": services.len(),
                "services": services,
            }))
        }
        SandboxToolArg::RestartServer => {
            let service_id = service_id.context("restart_server requires `--service-id`")?;
            let mut record = load_managed_services(paths)?
                .into_iter()
                .find(|record| record.service_id == service_id)
                .with_context(|| format!("managed service `{service_id}` not found"))?;
            restart_managed_service(paths, &mut record)?;
            Ok(json!({
                "tool": tool.as_cli_value(),
                "status": "restarted",
                "mutating": true,
                "service": record,
            }))
        }
        SandboxToolArg::StopServer => {
            let service_id = service_id.context("stop_server requires `--service-id`")?;
            let stopped = stop_managed_service(paths, &service_id)?;
            Ok(sandbox_stop_server_value(stopped))
        }
        SandboxToolArg::PrefetchArtifact => {
            let artifact_ref =
                artifact_ref.context("prefetch_artifact requires `--artifact-ref`")?;
            let resolved = resolve_model_recipe_artifact(&artifact_ref)?.with_context(|| {
                format!("artifact_ref `{artifact_ref}` was not found in the model recipe registry")
            })?;
            let (recipe, artifact) = resolved;
            prefetch_artifact_value_with_policy(
                paths,
                &artifact_ref,
                &recipe.canonical_model_id,
                artifact,
                policy,
            )
        }
        SandboxToolArg::NotifyUser => {
            let message = message.unwrap_or_else(|| "sandbox notification".to_owned());
            record_notification_audit(paths, "sandbox:notify_user", "notify_user", None, &message)?;
            Ok(json!({
                "tool": tool.as_cli_value(),
                "status": "notified",
                "mutating": false,
                "message": message,
                "notification_recorded": true,
            }))
        }
    }
}

/// The `stop_server` sandbox tool's answer for a [`stop_managed_service`]
/// report. The status is taken from the same verdict that decided what the
/// stop wrote to the manifest, so the two cannot disagree.
fn sandbox_stop_server_value(stopped: Value) -> Value {
    let status = if stop_report_confirmed(&stopped) {
        "stopped"
    } else {
        "stop_unconfirmed"
    };
    json!({
        "tool": SandboxToolArg::StopServer.as_cli_value(),
        "status": status,
        "mutating": true,
        "result": stopped,
    })
}

pub(crate) fn record_notification_audit(
    paths: &AppPaths,
    actor: &str,
    action: &str,
    watcher_id: Option<&str>,
    message: &str,
) -> Result<()> {
    append_audit_event(
        paths,
        &AuditEventRecord {
            at_unix_ms: unix_time_millis(),
            source: "rocmd".to_owned(),
            category: "notification".to_owned(),
            actor: actor.to_owned(),
            level: "info".to_owned(),
            action: action.to_owned(),
            message: message.to_owned(),
            watcher_id: watcher_id.map(ToOwned::to_owned),
            service_id: None,
        },
    )
}

pub(crate) fn sandbox_check_updates_value(output: CommandCapture) -> Value {
    let status = update_check_status(&output);
    let update_available =
        output.exit_status == 0 && update_output_reports_update_available(&output.stdout);
    let message = common::update_check_message(status);
    json!({
        "tool": SandboxToolArg::CheckUpdates.as_cli_value(),
        "status": status,
        "update_available": update_available,
        "mutating": false,
        "message": message,
        "argv": output.argv,
        "exit_status": output.exit_status,
        "stdout": output.stdout,
        "stderr": output.stderr,
    })
}

fn update_check_status(output: &CommandCapture) -> &'static str {
    if output.exit_status != 0 {
        "error"
    // A newer version outranks a composition repair: reporting the repair while
    // some runtime is a whole release behind would understate the tree.
    } else if update_output_reports_status(&output.stdout, "update_available") {
        "update_available"
    } else if update_output_reports_status(&output.stdout, "repair_available") {
        "repair_available"
    } else {
        "checked"
    }
}

fn update_output_reports_status(stdout: &str, status: &str) -> bool {
    let expected = format!("status={status}");
    stdout.split_whitespace().any(|part| part == expected)
}

fn update_output_reports_update_available(stdout: &str) -> bool {
    update_output_reports_status(stdout, "update_available")
        || update_output_reports_status(stdout, "repair_available")
        || stdout
            .split_whitespace()
            .any(|part| part == "update_available=true")
}

pub(crate) fn sandbox_driver_plan_value(output: CommandCapture) -> Value {
    let status = if output.exit_status == 0 {
        "planned"
    } else {
        "error"
    };
    json!({
        "tool": SandboxToolArg::DriverPlan.as_cli_value(),
        "status": status,
        "mutating": false,
        "message": "ran read-only `rocm install driver --dkms --dry-run`; no driver commands were executed",
        "argv": output.argv,
        "exit_status": output.exit_status,
        "stdout": output.stdout,
        "stderr": output.stderr,
    })
}

fn sandbox_report(tool: SandboxToolArg, isolation: &str, output: Value) -> Value {
    json!({
        "protocol": "rocmd-sandbox-run-v0",
        "tool": tool.as_cli_value(),
        "isolation": isolation,
        "ok": true,
        "ok_meaning": "sandbox wrapper completed; inspect output.status and output.exit_status for the restricted tool result",
        "output": output,
    })
}

fn record_sandbox_audit(
    paths: &AppPaths,
    tool: SandboxToolArg,
    isolation: &str,
    ok: bool,
    service_id: Option<&str>,
) -> Result<()> {
    append_audit_event(
        paths,
        &AuditEventRecord {
            at_unix_ms: unix_time_millis(),
            source: "rocmd".to_owned(),
            category: "sandbox".to_owned(),
            actor: "sandbox-runner".to_owned(),
            level: if ok { "info" } else { "error" }.to_owned(),
            action: tool.as_cli_value().to_owned(),
            message: format!(
                "sandbox tool `{}` completed with isolation `{isolation}`",
                tool.as_cli_value()
            ),
            watcher_id: None,
            service_id: service_id.map(str::to_owned),
        },
    )
}

#[cfg(target_os = "linux")]
fn append_sandbox_tool_command_args(
    command: &mut ProcessCommand,
    rocmd_binary: &std::path::Path,
    tool: SandboxToolArg,
    service_id: Option<&str>,
    artifact_ref: Option<&str>,
    message: Option<&str>,
    policy: SandboxToolPolicy,
) {
    command
        .arg("--")
        .arg(rocmd_binary)
        .arg("sandbox-tool")
        .arg(tool.as_cli_value());
    if let Some(service_id) = service_id {
        command.arg("--service-id").arg(service_id);
    }
    if let Some(artifact_ref) = artifact_ref {
        command.arg("--artifact-ref").arg(artifact_ref);
    }
    if policy.allow_artifact_download {
        command.arg("--allow-artifact-download");
    }
    if policy.allow_huggingface_download {
        command.arg("--allow-huggingface-download");
    }
    if let Some(max_bytes) = policy.artifact_max_bytes {
        command
            .arg("--artifact-max-bytes")
            .arg(max_bytes.to_string());
    }
    if let Some(message) = message {
        command.arg("--message").arg(message);
    }
}

#[cfg(target_os = "linux")]
fn parse_sandbox_child_output(output: std::process::Output, label: &str) -> Result<Value> {
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !output.status.success() {
        bail!(
            "{label} failed: {}",
            if !stderr.is_empty() {
                stderr
            } else if !stdout.is_empty() {
                stdout
            } else {
                format!("exit status {}", output.status)
            }
        );
    }
    serde_json::from_str(&stdout).with_context(|| format!("failed to parse {label} json output"))
}

#[cfg(target_os = "linux")]
fn command_available(name: &str) -> bool {
    ProcessCommand::new(name)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::load_service_record;
    use crate::test_support::{
        assert_unconfirmed_stop_kept_the_service, seed_keyed_service, stop_with_outcome,
        temp_app_paths,
    };
    use rocm_core::ManagedServiceRecord;
    #[cfg(target_os = "linux")]
    use std::path::Path;

    #[test]
    fn sandbox_check_updates_value_is_read_only_and_preserves_output() {
        let value = sandbox_check_updates_value(CommandCapture {
            argv: vec!["rocm".to_owned(), "update".to_owned()],
            exit_status: 0,
            stdout: "update\n  runtime release status=up_to_date\n".to_owned(),
            stderr: String::new(),
        });

        assert_eq!(
            value.get("tool").and_then(Value::as_str),
            Some("check_updates")
        );
        assert_eq!(value.get("status").and_then(Value::as_str), Some("checked"));
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("no updates were applied"))
        );
        assert!(
            value
                .get("stdout")
                .and_then(Value::as_str)
                .is_some_and(|stdout| stdout.contains("status=up_to_date"))
        );
    }

    #[test]
    fn sandbox_check_updates_value_marks_runtime_update_available() {
        let value = sandbox_check_updates_value(CommandCapture {
            argv: vec!["rocm".to_owned(), "update".to_owned()],
            exit_status: 0,
            stdout: "update\n  runtime release-pip-gfx120x-all status=update_available installed=7.13.0 latest=7.14.0\n".to_owned(),
            stderr: String::new(),
        });

        assert_eq!(
            value.get("tool").and_then(Value::as_str),
            Some("check_updates")
        );
        assert_eq!(
            value.get("status").and_then(Value::as_str),
            Some("update_available")
        );
        assert_eq!(
            value.get("update_available").and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(
                    |message| message.contains("a ROCm runtime update is available")
                        && message.contains("no updates were applied")
                )
        );
        assert!(
            value
                .get("stdout")
                .and_then(Value::as_str)
                .is_some_and(|stdout| stdout.contains("status=update_available"))
        );
    }

    #[test]
    fn sandbox_check_updates_value_marks_runtime_repair_available() {
        let value = sandbox_check_updates_value(CommandCapture {
            argv: vec!["rocm".to_owned(), "update".to_owned()],
            exit_status: 0,
            stdout: "update\n  runtime release-wheel-multi-arch-7-14-0 status=repair_available installed=7.14.0 latest=7.14.0\n".to_owned(),
            stderr: String::new(),
        });

        assert_eq!(
            value.get("status").and_then(Value::as_str),
            Some("repair_available")
        );
        // The watcher's notify decision reads this flag, so a repair the user
        // has to apply by hand must not be reported as nothing to do.
        assert_eq!(
            value.get("update_available").and_then(Value::as_bool),
            Some(true)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("runtime repair is available")
                    && message.contains("no updates were applied"))
        );
    }

    #[test]
    fn a_newer_version_outranks_a_composition_repair_in_the_watcher_report() {
        let value = sandbox_check_updates_value(CommandCapture {
            argv: vec!["rocm".to_owned(), "update".to_owned()],
            exit_status: 0,
            stdout: "update\n  runtime old status=repair_available installed=7.14.0 latest=7.14.0\n  runtime stale status=update_available installed=7.13.0 latest=7.14.0\n".to_owned(),
            stderr: String::new(),
        });

        assert_eq!(
            value.get("status").and_then(Value::as_str),
            Some("update_available")
        );
    }

    #[test]
    fn sandbox_driver_plan_value_is_read_only_and_preserves_output() {
        let value = sandbox_driver_plan_value(CommandCapture {
            argv: vec![
                "rocm".to_owned(),
                "install".to_owned(),
                "driver".to_owned(),
                "--dkms".to_owned(),
                "--dry-run".to_owned(),
            ],
            exit_status: 0,
            stdout: "driver install plan\n  supported: true\n".to_owned(),
            stderr: String::new(),
        });

        assert_eq!(
            value.get("tool").and_then(Value::as_str),
            Some("driver_plan")
        );
        assert_eq!(value.get("status").and_then(Value::as_str), Some("planned"));
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("no driver commands were executed"))
        );
        assert!(
            value
                .get("stdout")
                .and_then(Value::as_str)
                .is_some_and(|stdout| stdout.contains("driver install plan"))
        );
    }

    #[test]
    fn sandbox_tool_cli_values_cover_restricted_plan_api() {
        let names = [
            SandboxToolArg::CheckUpdates,
            SandboxToolArg::ExamineSnapshot,
            SandboxToolArg::ListServers,
            SandboxToolArg::RestartServer,
            SandboxToolArg::StopServer,
            SandboxToolArg::PrefetchArtifact,
            SandboxToolArg::NotifyUser,
            SandboxToolArg::DriverPlan,
        ]
        .into_iter()
        .map(SandboxToolArg::as_cli_value)
        .collect::<Vec<_>>();

        for expected in [
            "check_updates",
            "examine_snapshot",
            "list_servers",
            "restart_server",
            "stop_server",
            "prefetch_artifact",
            "notify_user",
            "driver_plan",
        ] {
            assert!(names.contains(&expected), "missing sandbox tool {expected}");
        }
    }

    #[test]
    fn sandbox_tool_examine_snapshot_is_read_only() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-examine-snapshot");
        let value = run_sandbox_tool(
            &paths,
            SandboxToolArg::ExamineSnapshot,
            None,
            None,
            None,
            SandboxToolPolicy::default(),
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(
            value.get("tool").and_then(Value::as_str),
            Some("examine_snapshot")
        );
        assert_eq!(
            value.get("status").and_then(Value::as_str),
            Some("captured")
        );
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert!(value.get("examine").is_some());
        Ok(())
    }

    #[test]
    fn sandbox_tool_list_servers_returns_records() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-list-servers");
        paths.ensure()?;
        let record = rocm_core::ManagedServiceRecord::new(
            &paths,
            "svc-1",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            123,
            None,
            None,
            None,
        );
        record.write()?;

        let value = run_sandbox_tool(
            &paths,
            SandboxToolArg::ListServers,
            None,
            None,
            None,
            SandboxToolPolicy::default(),
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(
            value.get("tool").and_then(Value::as_str),
            Some("list_servers")
        );
        assert_eq!(value.get("status").and_then(Value::as_str), Some("listed"));
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert_eq!(value.get("count").and_then(Value::as_u64), Some(1));
        Ok(())
    }

    #[test]
    fn sandbox_tool_list_servers_first_run_returns_empty_list() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-list-servers-empty");
        let value = run_sandbox_tool(
            &paths,
            SandboxToolArg::ListServers,
            None,
            None,
            None,
            SandboxToolPolicy::default(),
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(
            value.get("tool").and_then(Value::as_str),
            Some("list_servers")
        );
        assert_eq!(value.get("count").and_then(Value::as_u64), Some(0));
        assert_eq!(
            value
                .get("services")
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(0)
        );
        Ok(())
    }

    #[test]
    fn sandbox_tool_requires_service_id_for_restart() {
        let (root, paths) = temp_app_paths("sandbox-restart-requires-service");
        let error = run_sandbox_tool(
            &paths,
            SandboxToolArg::RestartServer,
            None,
            None,
            None,
            SandboxToolPolicy::default(),
        )
        .unwrap_err();
        fs::remove_dir_all(root).ok();

        assert!(
            error.to_string().contains("restart_server requires"),
            "{error:#}"
        );
    }

    #[test]
    fn sandbox_tool_restart_server_reports_missing_service() {
        let (root, paths) = temp_app_paths("sandbox-restart-missing-service");
        let error = run_sandbox_tool(
            &paths,
            SandboxToolArg::RestartServer,
            Some("missing-service".to_owned()),
            None,
            None,
            SandboxToolPolicy::default(),
        )
        .unwrap_err();
        fs::remove_dir_all(root).ok();

        assert!(
            error
                .to_string()
                .contains("managed service `missing-service` not found"),
            "{error:#}"
        );
    }

    #[test]
    fn sandbox_tool_requires_service_id_for_stop() {
        let (root, paths) = temp_app_paths("sandbox-stop-requires-service");
        let error = run_sandbox_tool(
            &paths,
            SandboxToolArg::StopServer,
            None,
            None,
            None,
            SandboxToolPolicy::default(),
        )
        .unwrap_err();
        fs::remove_dir_all(root).ok();

        assert!(
            error.to_string().contains("stop_server requires"),
            "{error:#}"
        );
    }

    #[test]
    fn sandbox_tool_stop_server_reports_missing_service() {
        let (root, paths) = temp_app_paths("sandbox-stop-missing-service");
        let error = run_sandbox_tool(
            &paths,
            SandboxToolArg::StopServer,
            Some("missing-service".to_owned()),
            None,
            None,
            SandboxToolPolicy::default(),
        )
        .unwrap_err();
        fs::remove_dir_all(root).ok();

        assert!(
            error
                .to_string()
                .contains("managed service `missing-service` not found"),
            "{error:#}"
        );
    }

    #[test]
    fn sandbox_tool_notify_user_is_read_only() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-notify-user");
        let value = run_sandbox_tool(
            &paths,
            SandboxToolArg::NotifyUser,
            None,
            None,
            Some("ROCm setup is ready.".to_owned()),
            SandboxToolPolicy::default(),
        )?;
        let audit_text = fs::read_to_string(paths.audit_events_path())?;
        fs::remove_dir_all(root).ok();

        assert_eq!(
            value.get("tool").and_then(Value::as_str),
            Some("notify_user")
        );
        assert_eq!(
            value.get("status").and_then(Value::as_str),
            Some("notified")
        );
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert_eq!(
            value.get("message").and_then(Value::as_str),
            Some("ROCm setup is ready.")
        );
        assert!(audit_text.contains("\"category\":\"notification\""));
        assert!(audit_text.contains("ROCm setup is ready."));
        Ok(())
    }

    #[test]
    fn sandbox_runner_native_fallback_records_audit() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-native-audit");
        paths.ensure()?;

        let value = run_native_restricted_sandbox(
            &paths,
            SandboxToolArg::NotifyUser,
            None,
            None,
            Some("hello".to_owned()),
            SandboxToolPolicy::default(),
        )?;
        let audit_text = fs::read_to_string(paths.audit_events_path())?;
        fs::remove_dir_all(root).ok();

        assert_eq!(
            value.get("isolation").and_then(Value::as_str),
            Some("native_restricted")
        );
        assert!(audit_text.contains("\"category\":\"sandbox\""));
        assert!(audit_text.contains("\"action\":\"notify_user\""));
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn bubblewrap_command_separates_child_args_from_bwrap_options() {
        let mut command = ProcessCommand::new("bwrap");
        append_sandbox_tool_command_args(
            &mut command,
            Path::new("/tmp/rocmd"),
            SandboxToolArg::NotifyUser,
            None,
            None,
            Some("hello"),
            SandboxToolPolicy::default(),
        );
        let args = command
            .get_args()
            .map(|value| value.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        let separator = args
            .iter()
            .position(|arg| arg == "--")
            .expect("bubblewrap command should separate child command args");

        assert_eq!(
            args.get(separator + 1).map(String::as_str),
            Some("/tmp/rocmd")
        );
        assert_eq!(
            args.get(separator + 4).map(String::as_str),
            Some("--message")
        );
        assert_eq!(args.get(separator + 5).map(String::as_str), Some("hello"));
    }

    #[test]
    fn sandbox_prefetch_requires_artifact_ref() {
        let (root, paths) = temp_app_paths("sandbox-prefetch-requires-ref");
        let error = run_sandbox_tool(
            &paths,
            SandboxToolArg::PrefetchArtifact,
            None,
            None,
            None,
            SandboxToolPolicy::default(),
        )
        .unwrap_err();
        fs::remove_dir_all(root).ok();

        assert!(
            error.to_string().contains("prefetch_artifact requires"),
            "{error:#}"
        );
    }

    #[test]
    fn sandbox_prefetch_unknown_artifact_ref_errors() {
        let (root, paths) = temp_app_paths("sandbox-prefetch-unknown-ref");
        let error = run_sandbox_tool(
            &paths,
            SandboxToolArg::PrefetchArtifact,
            None,
            Some("Qwen/Missing#hf-main".to_owned()),
            None,
            SandboxToolPolicy::default(),
        )
        .unwrap_err();
        fs::remove_dir_all(root).ok();

        assert!(error.to_string().contains("was not found"), "{error:#}");
    }

    #[test]
    fn sandbox_tool_stop_server_updates_manifest_and_skips_current_pid() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-stop-current-pid");
        paths.ensure()?;
        let current_pid = std::process::id();
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-current",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            current_pid,
            None,
            None,
            None,
        );
        record.engine_pid = Some(current_pid);
        record.status = "ready".to_owned();
        record.write()?;

        let value = run_sandbox_tool(
            &paths,
            SandboxToolArg::StopServer,
            Some("svc-current".to_owned()),
            None,
            None,
            SandboxToolPolicy::default(),
        )?;
        let reloaded = load_service_record(&paths, "svc-current")?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("stopped"));
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(true));
        assert_eq!(reloaded.status, "stopped");
        assert!(
            value
                .get("result")
                .and_then(|result| result.get("skipped_pids"))
                .and_then(Value::as_array)
                .is_some_and(|pids| pids
                    .iter()
                    .any(|pid| pid.as_u64() == Some(u64::from(current_pid))))
        );
        Ok(())
    }

    /// The sandbox tool used to report `status: stopped` unconditionally.
    #[test]
    fn sandbox_tool_stop_server_reports_an_unconfirmed_stop() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-stop-unconfirmed");
        paths.ensure()?;
        let service_id = "svc-sandbox-stop-unconfirmed";
        let key_path = seed_keyed_service(&paths, service_id, 11448)?;

        // The tool is `stop_managed_service` feeding `sandbox_stop_server_value`;
        // `sandbox_tool_stop_server_updates_manifest_and_skips_current_pid`
        // drives that whole tool for real.
        let result = stop_with_outcome(
            &paths,
            service_id,
            rocm_core::TerminationOutcome::Unverified,
        )
        .map(sandbox_stop_server_value);
        let key_kept = key_path.exists();
        let reloaded = load_service_record(&paths, service_id);
        fs::remove_dir_all(root).ok();

        let value = result?;
        assert_eq!(
            value.get("status").and_then(Value::as_str),
            Some("stop_unconfirmed"),
            "{value}"
        );
        assert_eq!(
            value.pointer("/result/stopped").and_then(Value::as_bool),
            Some(false),
            "{value}"
        );
        assert_unconfirmed_stop_kept_the_service(&reloaded?, key_kept);
        Ok(())
    }
}
