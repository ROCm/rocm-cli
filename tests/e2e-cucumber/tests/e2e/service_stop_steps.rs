// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for stopping a managed service: `rocm services stop <id> --yes`, and
//! `rocmd sandbox-tool stop_server`.
//!
//! Black-box: plants a managed-service record as plain JSON in its own isolated
//! data dir — the same on-disk shape `rocm serve --managed` writes — then runs
//! the real binary and checks what it reports against the record, and, for the
//! daemon's tool, against the process and the key file the record describes. No
//! GPU, no network — mock lane.

use std::path::PathBuf;
use std::process::{Command, Stdio};

use cucumber::{given, then, when};

use crate::E2eWorld;

/// Id of the record this scenario plants.
const SERVICE_ID: &str = "vllm-e2e-stop";
/// Engine name, which also picks the engine state directory the record's
/// third file lives in (`<data>/engines/<engine>/state/`).
const ENGINE: &str = "vllm";

fn data_dir(world: &E2eWorld) -> PathBuf {
    world
        .isolated_root
        .as_ref()
        .expect("scenario has no isolated root")
        .path()
        .join("data")
}

fn services_dir(world: &E2eWorld) -> PathBuf {
    data_dir(world).join("services")
}

fn engine_state_dir(world: &E2eWorld) -> PathBuf {
    data_dir(world).join("engines").join(ENGINE).join("state")
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

/// Does the default `services list` output carry a *live server row* for the
/// planted record? Mirrors `service_cleanup_steps::lists_live_server_row`: the
/// default view lists only servers it considers running, each as `- <id>` on
/// its own line.
fn lists_live_server_row(listed: &str) -> bool {
    listed
        .lines()
        .any(|line| line.trim_end().strip_prefix("- ") == Some(SERVICE_ID))
}

// ── Given ──────────────────────────────────────────────────────────

/// Plant a record with `supervisor_pid: 0` — the documented placeholder for
/// "no process recorded yet" — so the CLI's liveness overlay reads `starting`
/// at face value instead of checking a PID. Recording a *real* live PID would
/// be actively harmful here: the World's teardown runs `rocm services stop
/// <id> --yes` for every record left in the isolated tree, which would
/// terminate a PID naming this test process.
#[given("a managed service that is running")]
async fn plant_running_service(world: &mut E2eWorld) {
    let services = services_dir(world);
    let states = engine_state_dir(world);
    std::fs::create_dir_all(&services).expect("failed to create services dir");
    std::fs::create_dir_all(&states).expect("failed to create engine state dir");

    let manifest = manifest_path(world);
    let record = serde_json::json!({
        "service_id": SERVICE_ID,
        "engine": ENGINE,
        "model_ref": "qwen",
        "canonical_model_id": "Qwen/Qwen3.5",
        "host": "127.0.0.1",
        // Discard port: closed, so nothing here depends on a real server
        // answering on it.
        "port": 9,
        "endpoint_url": "http://127.0.0.1:9/v1",
        "mode": "managed",
        "status": "starting",
        "supervisor_pid": 0,
        "engine_pid": null,
        "manifest_path": manifest,
        "log_path": services.join(format!("{SERVICE_ID}.log")),
        "engine_state_path": states.join(format!("{SERVICE_ID}.json")),
        "created_at_unix_ms": 1_700_000_000_000_u64,
    });
    std::fs::write(
        &manifest,
        serde_json::to_vec_pretty(&record).expect("failed to serialize record"),
    )
    .expect("failed to write service record");
    std::fs::write(
        states.join(format!("{SERVICE_ID}.json")),
        serde_json::json!({ "status": "starting" }).to_string(),
    )
    .expect("failed to write engine state");

    // Guard the premise: the default (live-only) list must show it, or this
    // scenario would be exercising a stop on something not actually running.
    let listed = crate::run_rocm_ok(world, &["services", "list"]);
    assert!(
        lists_live_server_row(&listed),
        "premise: the planted record must read as running:\n{listed}"
    );
}

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

/// Runs the pinned, directly-runnable form `serve_summary.rs`'s `stop` row
/// prints: `rocm services stop <id> --yes`. Built from the known
/// [`SERVICE_ID`] rather than parsed out of any printed hint — nothing here
/// ties this literally back to `serve_summary.rs`'s own output; that
/// equivalence is what the hint-specific scenarios (`serve-24`, remote-15)
/// prove, by asserting the hint text first and then running it.
#[when("the user stops it with --yes")]
async fn stop_with_yes(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["services", "stop", SERVICE_ID, "--yes"]);
    record(world, stdout, stderr, rc);
}

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

// `render_service_action_result` only prints the `service:`/`status:`/
// `endpoint:` lines when `output.result` carries a nested `service` object
// (as `restart_server` does); `stop_server`'s result is the flat
// `service_id`/`status` shape `stop_internal_managed_service` returns, so none
// of those lines ever print for a stop today. That mismatch is a pre-existing
// gap in that renderer, unrelated to the summary hint this PR fixes, so this
// assertion covers only what the command actually prints; the on-disk status
// below is what proves the stop itself.
#[then("the CLI reports the service as stopped")]
async fn reports_stopped(world: &mut E2eWorld) {
    assert_succeeded(world);
    let combined = combined_output(world);
    assert!(
        combined.contains("Local server stopped"),
        "the stop command must say it stopped the server, got:\n{combined}"
    );
}

#[then("the service record on disk is marked stopped")]
async fn record_marked_stopped(world: &mut E2eWorld) {
    let bytes = std::fs::read(manifest_path(world)).expect("failed to read service record");
    let record: serde_json::Value =
        serde_json::from_slice(&bytes).expect("failed to parse service record");
    assert_eq!(
        record.get("status").and_then(serde_json::Value::as_str),
        Some("stopped"),
        "the on-disk record must persist the stop, got:\n{record}"
    );
}

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

// ── Helpers ────────────────────────────────────────────────────────

fn record(world: &mut E2eWorld, stdout: String, stderr: String, rc: i32) {
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

fn combined_output(world: &E2eWorld) -> String {
    format!(
        "{}\n{}",
        world.cli_output.as_deref().unwrap_or(""),
        world.cli_stderr.as_deref().unwrap_or("")
    )
}

fn assert_succeeded(world: &E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    assert_eq!(
        rc,
        0,
        "expected success, got rc={rc}:\n{}",
        combined_output(world)
    );
}
