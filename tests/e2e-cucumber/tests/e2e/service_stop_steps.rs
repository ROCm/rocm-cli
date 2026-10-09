// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocmd sandbox-tool stop_server`.
//!
//! Black-box: the scenario starts a real process, plants a managed-service
//! record naming it as plain JSON — the on-disk shape a managed launch writes —
//! then runs the real `rocmd` and checks its JSON report against the process,
//! the record and the key file it describes.

use std::path::PathBuf;
use std::process::{Command, Stdio};

use cucumber::{given, then, when};

use crate::E2eWorld;

/// Id of the record the scenario plants.
const SERVICE_ID: &str = "vllm-e2e-stop";

fn services_dir(world: &E2eWorld) -> PathBuf {
    world
        .isolated_root
        .as_ref()
        .expect("scenario has no isolated root")
        .path()
        .join("data")
        .join("services")
}

fn manifest_path(world: &E2eWorld) -> PathBuf {
    services_dir(world).join(format!("{SERVICE_ID}.json"))
}

fn endpoint_key_path(world: &E2eWorld) -> PathBuf {
    services_dir(world).join(format!("{SERVICE_ID}.endpoint-key"))
}

fn read_record(world: &E2eWorld) -> serde_json::Value {
    let bytes = std::fs::read(manifest_path(world)).expect("failed to read service record");
    serde_json::from_slice(&bytes).expect("service record is not JSON")
}

fn recorded_pid(world: &E2eWorld) -> u32 {
    world
        .recorded_process
        .as_ref()
        .expect("scenario recorded no process")
        .id()
}

/// The kernel's start time for `pid`, in clock ticks since boot: field 22 of
/// `/proc/<pid>/stat`. The fields are counted from after the LAST `)`, because
/// the command name before it may itself contain spaces or parentheses.
fn start_ticks(pid: u32) -> u64 {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .unwrap_or_else(|error| panic!("failed to read /proc/{pid}/stat: {error}"));
    let after_name = &stat[stat.rfind(')').expect("stat has no command name") + 1..];
    // `after_name` starts at field 3 (the state), so field 22 is index 19.
    after_name
        .split_whitespace()
        .nth(19)
        .and_then(|field| field.parse().ok())
        .unwrap_or_else(|| panic!("no start time in /proc/{pid}/stat: {stat}"))
}

// ── Given ──────────────────────────────────────────────────────────

/// A ready, publicly bound server whose recorded supervisor is a live process
/// this scenario owns, recorded with its own start-time token and with an
/// endpoint key on disk beside the record.
#[given("a managed server whose recorded process is running")]
async fn managed_server_with_running_process(world: &mut E2eWorld) {
    let process = Command::new("sleep")
        .arg("300")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to start the process to record");
    let pid = process.id();
    world.recorded_process = Some(process);
    let ticks = start_ticks(pid);

    let services = services_dir(world);
    std::fs::create_dir_all(&services).expect("failed to create services dir");
    let manifest = manifest_path(world);
    let record = serde_json::json!({
        "service_id": SERVICE_ID,
        "engine": "vllm",
        "model_ref": "qwen",
        "canonical_model_id": "Qwen/Qwen3.5",
        "host": "0.0.0.0",
        // Discard port: closed, so nothing can answer on the server's behalf.
        "port": 9,
        "endpoint_url": "http://0.0.0.0:9/v1",
        "mode": "managed",
        "status": "ready",
        "supervisor_pid": pid,
        "supervisor_start_ticks": ticks,
        "engine_pid": null,
        "manifest_path": manifest,
        "log_path": services.join(format!("{SERVICE_ID}.log")),
        "engine_state_path": services.join(format!("{SERVICE_ID}.state.json")),
        "created_at_unix_ms": 1_700_000_000_000_u64,
    });
    std::fs::write(
        &manifest,
        serde_json::to_vec_pretty(&record).expect("failed to serialize record"),
    )
    .expect("failed to write service record");
    std::fs::write(endpoint_key_path(world), "e2e-endpoint-key")
        .expect("failed to write endpoint key");
}

// ── When ───────────────────────────────────────────────────────────

#[when("the daemon's stop_server tool stops that server")]
async fn stop_server_tool(world: &mut E2eWorld) {
    let mut command = Command::new(crate::rocmd_binary());
    command.args(["sandbox-tool", "stop_server", "--service-id", SERVICE_ID]);
    world.isolate_cmd(&mut command);
    let output = command.output().expect("failed to run rocmd");
    world.cli_output = Some(String::from_utf8_lossy(&output.stdout).into_owned());
    world.cli_stderr = Some(String::from_utf8_lossy(&output.stderr).into_owned());
    world.cli_rc = Some(output.status.code().unwrap_or(-1));
}

// ── Then ───────────────────────────────────────────────────────────

#[then("the tool reports the server stopped, naming the process it stopped")]
async fn tool_reports_stopped(world: &mut E2eWorld) {
    let stdout = world.cli_output.as_deref().unwrap_or_default();
    assert_eq!(
        world.cli_rc,
        Some(0),
        "rocmd failed:\n{stdout}\n{}",
        world.cli_stderr.as_deref().unwrap_or_default()
    );
    let report: serde_json::Value =
        serde_json::from_str(stdout).unwrap_or_else(|error| panic!("{error}: {stdout}"));
    assert_eq!(report["status"], "stopped", "{report}");
    assert_eq!(report["result"]["stopped"], true, "{report}");
    let pid = u64::from(recorded_pid(world));
    let outcomes = report["result"]["pid_outcomes"]
        .as_array()
        .unwrap_or_else(|| panic!("no pid_outcomes: {report}"));
    let outcome = outcomes
        .iter()
        .find(|entry| entry["pid"].as_u64() == Some(pid))
        .unwrap_or_else(|| panic!("the recorded process is not named: {report}"));
    assert_eq!(outcome["role"], "supervisor", "{report}");
    // `sleep` exits on SIGTERM, so the stop never needs to escalate.
    assert_eq!(outcome["outcome"], "graceful", "{report}");
    assert!(
        report["result"]["signaled_pids"]
            .as_array()
            .is_some_and(|pids| pids.iter().any(|p| p.as_u64() == Some(pid))),
        "{report}"
    );
}

#[then("the recorded process is no longer running")]
async fn recorded_process_gone(world: &mut E2eWorld) {
    let process = world
        .recorded_process
        .as_mut()
        .expect("scenario recorded no process");
    // The report's claim, checked against the process itself: it has exited
    // (and is reaped here), rather than merely having been signalled.
    let exited = process
        .try_wait()
        .expect("failed to check the recorded process");
    assert!(
        exited.is_some(),
        "the stop reported the process gone, but it is still running"
    );
}

#[then("the server's record reads stopped and names no process")]
async fn record_reads_stopped(world: &mut E2eWorld) {
    let record = read_record(world);
    assert_eq!(record["status"], "stopped", "{record}");
    assert_eq!(record["supervisor_pid"], 0, "{record}");
    assert!(record["supervisor_start_ticks"].is_null(), "{record}");
    assert!(record["engine_pid"].is_null(), "{record}");
    assert!(
        record["stop_requested_unix_ms"].is_null(),
        "a confirmed stop leaves no stop pending: {record}"
    );
}

#[then("the server's endpoint key file is gone")]
async fn endpoint_key_gone(world: &mut E2eWorld) {
    let key = endpoint_key_path(world);
    assert!(!key.exists(), "{} survived the stop", key.display());
}
