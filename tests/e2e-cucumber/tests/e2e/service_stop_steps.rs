// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocm services stop <id> --yes` actually stopping a service.
//!
//! Black-box: plants a managed-service record as plain JSON in its own isolated
//! data dir — the same on-disk shape `rocm serve --managed` writes — then runs
//! the real binary and asserts the record is left `stopped`. No GPU, no
//! network — mock lane.

use std::path::PathBuf;

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

// ── When ───────────────────────────────────────────────────────────

/// Runs the exact line `serve_summary.rs`'s `stop` row prints: `rocm services
/// stop <id> --yes`.
#[when("the user stops it with --yes")]
async fn stop_with_yes(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["services", "stop", SERVICE_ID, "--yes"]);
    record(world, stdout, stderr, rc);
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
