// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocm services remove` / `rocm services prune`.
//!
//! Black-box: each scenario plants a managed-service record as plain JSON in its
//! own isolated data dir — the same on-disk shape `rocm serve --managed` writes —
//! then runs the real binary and asserts on the files that survive. Asserting on
//! the filesystem rather than the command's summary line is deliberate: the bug
//! this feature exists to fix is a file being *left behind*, which a summary
//! claiming success cannot reveal. No GPU, no network — mock lane.

use std::path::PathBuf;

use cucumber::{given, then, when};

use crate::E2eWorld;

/// Id of the record every scenario here plants.
const SERVICE_ID: &str = "vllm-e2e-cleanup";
/// Engine name, which also picks the engine state directory the record's third
/// file lives in (`<data>/engines/vllm/state/`).
const ENGINE: &str = "vllm";
/// A record whose manifest a user already deleted by hand, leaving its engine
/// state file stranded — the state this feature has to be able to clean up.
const ORPHAN_ID: &str = "vllm-e2e-hand-deleted";

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

/// The four files one record owns, in the order the CLI reports them.
fn record_files(world: &E2eWorld) -> Vec<PathBuf> {
    vec![
        services_dir(world).join(format!("{SERVICE_ID}.json")),
        services_dir(world).join(format!("{SERVICE_ID}.log")),
        engine_state_dir(world).join(format!("{SERVICE_ID}.json")),
        services_dir(world).join(format!("{SERVICE_ID}.endpoint-key")),
    ]
}

/// Write a record plus all four of its files.
///
/// `status` alone decides whether the CLI's liveness overlay reads the record as
/// running, because `supervisor_pid` is 0 — the documented placeholder for "no
/// process recorded yet", which is the real state of a record between `rocm
/// serve` writing it and the engine reporting a PID. With no PID to check, the
/// overlay leaves `starting` alone and leaves `failed` alone, so both premises
/// are deterministic on every platform.
///
/// Recording a *real* live PID here would be actively harmful: the World's
/// teardown runs `rocm services stop <id> --yes` for every record left in the
/// isolated tree, and that terminates the recorded PID — so a record naming this
/// test process makes the harness kill itself at end of scenario.
///
/// The engine state file must agree with `status`: the CLI adopts whatever that
/// file says before it decides anything else.
fn plant_record(world: &E2eWorld, status: &str) {
    let services = services_dir(world);
    let states = engine_state_dir(world);
    std::fs::create_dir_all(&services).expect("failed to create services dir");
    std::fs::create_dir_all(&states).expect("failed to create engine state dir");

    let manifest = services.join(format!("{SERVICE_ID}.json"));
    let log = services.join(format!("{SERVICE_ID}.log"));
    let engine_state = states.join(format!("{SERVICE_ID}.json"));
    let record = serde_json::json!({
        "service_id": SERVICE_ID,
        "engine": ENGINE,
        "model_ref": "qwen",
        "canonical_model_id": "Qwen/Qwen3.5",
        "host": "127.0.0.1",
        // Discard port: closed, so the CLI's readiness probe fails fast instead
        // of waiting out its timeout against something that might answer.
        "port": 9,
        "endpoint_url": "http://127.0.0.1:9/v1",
        "mode": "managed",
        "status": status,
        "supervisor_pid": 0,
        "engine_pid": null,
        "manifest_path": manifest,
        "log_path": log,
        "engine_state_path": engine_state,
        "created_at_unix_ms": 1_700_000_000_000_u64,
    });
    std::fs::write(
        &manifest,
        serde_json::to_vec_pretty(&record).expect("failed to serialize record"),
    )
    .expect("failed to write service record");
    std::fs::write(&log, "ERROR: engine exited during model load\n").expect("failed to write log");
    std::fs::write(
        &engine_state,
        serde_json::json!({ "status": status }).to_string(),
    )
    .expect("failed to write engine state");
    std::fs::write(
        services.join(format!("{SERVICE_ID}.endpoint-key")),
        "e2e-endpoint-key",
    )
    .expect("failed to write endpoint key");

    // Shared by every managed launch rather than owned by one service; a
    // per-service cleanup that deleted it would break the next `rocm serve`.
    std::fs::write(services.join("launch.lock"), "").expect("failed to write launch lock");
}

// ── Given ──────────────────────────────────────────────────────────

#[given("a local server record that is no longer running")]
async fn record_not_running(world: &mut E2eWorld) {
    plant_record(world, "failed");
    // Guard the premise: the default (live-only) list must NOT show it, or the
    // scenario would be exercising the running branch without saying so.
    let listed = crate::run_rocm_ok(world, &["services", "list"]);
    assert!(
        !listed.contains(SERVICE_ID),
        "premise: the planted record must read as not running:\n{listed}"
    );
}

#[given("a local server record that is still running")]
async fn record_still_running(world: &mut E2eWorld) {
    plant_record(world, "starting");
    // Guard the premise the other way: the default list shows only live
    // servers, so the record appearing here is the CLI's own statement that it
    // considers this server to be running.
    let listed = crate::run_rocm_ok(world, &["services", "list"]);
    assert!(
        listed.contains(SERVICE_ID),
        "premise: the planted record must read as running:\n{listed}"
    );
}

/// A record still claiming `ready` whose server is long gone, and whose files
/// have not been touched since.
///
/// No `services list` premise guard here, unlike the steps above: listing is
/// exactly what refreshes and rewrites the record, and "never observed since it
/// died" is the premise. The status is `ready` so the CLI's first look at it
/// corrects the status and persists that correction mid-prune — the rewrite the
/// age gate must not mistake for the record being new.
#[given("a local server record whose server died long ago and was never listed since")]
async fn record_died_long_ago(world: &mut E2eWorld) {
    plant_record(world, "ready");
    let month_ago = std::time::SystemTime::now() - std::time::Duration::from_hours(24 * 30);
    for path in record_files(world) {
        let handle = std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap_or_else(|error| panic!("failed to open {} : {error}", path.display()));
        handle
            .set_modified(month_ago)
            .unwrap_or_else(|error| panic!("failed to backdate {} : {error}", path.display()));
    }
}

/// Put something `remove_file` refuses to delete where the engine state file
/// belongs: a directory that is not empty.
///
/// The portable way to manufacture an unremovable path without root or
/// filesystem attributes — `unlink` on a directory fails on Linux (EISDIR) and
/// on Windows alike. The real-world shapes are a permission-locked path or a
/// file another process holds open; what the CLI sees is the same errno either
/// way.
#[given("that record's engine state file cannot be deleted")]
async fn engine_state_cannot_be_deleted(world: &mut E2eWorld) {
    let stuck = engine_state_dir(world).join(format!("{SERVICE_ID}.json"));
    std::fs::remove_file(&stuck).expect("failed to remove planted engine state");
    std::fs::create_dir(&stuck).expect("failed to create directory in its place");
    std::fs::write(stuck.join("held.json"), "{}").expect("failed to fill the directory");
    assert!(
        std::fs::remove_file(&stuck).is_err(),
        "premise: {} must be undeletable",
        stuck.display()
    );
}

#[given("an engine state file whose local server record was deleted by hand")]
async fn orphaned_engine_state(world: &mut E2eWorld) {
    let states = engine_state_dir(world);
    std::fs::create_dir_all(&states).expect("failed to create engine state dir");
    std::fs::write(
        states.join(format!("{ORPHAN_ID}.json")),
        serde_json::json!({ "status": "failed" }).to_string(),
    )
    .expect("failed to write orphaned engine state");
}

// ── When ───────────────────────────────────────────────────────────

#[when("the user removes that local server record")]
async fn remove_record(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["services", "remove", SERVICE_ID, "--yes"]);
    record(world, stdout, stderr, rc);
}

#[when("the user tries to remove that local server record")]
async fn try_remove_record(world: &mut E2eWorld) {
    remove_record(world).await;
}

#[when("the user previews a prune of every record that is not running")]
async fn preview_prune(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &["services", "prune", "--older-than-hours", "0", "--dry-run"],
    );
    record(world, stdout, stderr, rc);
}

#[when("the user prunes every record that is not running")]
async fn run_prune(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &["services", "prune", "--older-than-hours", "0", "--yes"],
    );
    record(world, stdout, stderr, rc);
}

/// No age argument at all — the invocation a user reaches for first, and the one
/// whose behaviour the README and the command's own summary describe.
#[when("the user prunes with the default age rule")]
async fn run_prune_default_age(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["services", "prune", "--yes"]);
    record(world, stdout, stderr, rc);
}

/// `--any-age`, spelled the way the summary tells the user to spell it, not the
/// `--older-than-hours 0` equivalent: the point is that the advertised flag
/// reaches the same code path.
#[when("the user prunes every record whatever its age")]
async fn run_prune_any_age(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["services", "prune", "--any-age", "--yes"]);
    record(world, stdout, stderr, rc);
}

// ── Then ───────────────────────────────────────────────────────────

#[then("the CLI names the log file before deleting it")]
async fn names_the_log(world: &mut E2eWorld) {
    assert_succeeded(world);
    let log = services_dir(world).join(format!("{SERVICE_ID}.log"));
    let combined = combined_output(world);
    assert!(
        combined.contains(&log.display().to_string()),
        "the removal must print the log path while it still exists, got:\n{combined}"
    );
}

#[then("every file belonging to that record is gone")]
async fn record_files_gone(world: &mut E2eWorld) {
    for path in record_files(world) {
        assert!(
            !path.exists(),
            "{} should have been deleted, but it is still there:\n{}",
            path.display(),
            combined_output(world)
        );
    }
}

#[then("every file belonging to that record is still there")]
async fn record_files_present(world: &mut E2eWorld) {
    for path in record_files(world) {
        assert!(
            path.exists(),
            "{} must not have been deleted:\n{}",
            path.display(),
            combined_output(world)
        );
    }
}

#[then("the CLI says it kept the record for being recent and names --any-age")]
async fn kept_for_being_recent(world: &mut E2eWorld) {
    assert_succeeded(world);
    let combined = combined_output(world);
    assert!(
        combined.contains("too recent, kept: 1"),
        "a silent keep is indistinguishable from finding nothing, got:\n{combined}"
    );
    assert!(
        combined.contains("rocm services prune --any-age --yes"),
        "the summary must name the flag that includes the kept record, got:\n{combined}"
    );
}

#[then("the CLI does not claim it kept anything for being recent")]
async fn kept_nothing_for_being_recent(world: &mut E2eWorld) {
    assert_succeeded(world);
    let combined = combined_output(world);
    assert!(
        combined.contains("too recent, kept: 0"),
        "a record untouched for a month is not recent, got:\n{combined}"
    );
}

#[then("the shared launch lock is still there")]
async fn launch_lock_present(world: &mut E2eWorld) {
    let lock = services_dir(world).join("launch.lock");
    assert!(
        lock.exists(),
        "the shared launch lock must survive a per-service removal:\n{}",
        combined_output(world)
    );
}

#[then("the record no longer appears in the full list")]
async fn record_not_listed(world: &mut E2eWorld) {
    let listed = crate::run_rocm_ok(world, &["services", "list", "--all"]);
    assert!(
        !listed.contains(SERVICE_ID),
        "the removed record must be gone from `rocm services list --all`:\n{listed}"
    );
}

#[then("the CLI refuses and tells the user to stop the server first")]
async fn refuses_running(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    let combined = combined_output(world);
    assert!(
        rc != 0,
        "removing a running local server must fail, but it exited 0:\n{combined}"
    );
    assert!(
        combined.contains("cannot be removed while it is running"),
        "the refusal must say why, got:\n{combined}"
    );
    assert!(
        combined.contains(&format!("rocm services stop {SERVICE_ID} --yes")),
        "the refusal must name the stop command, got:\n{combined}"
    );
}

#[then("the preview lists the record and the leftover file")]
async fn preview_lists_both(world: &mut E2eWorld) {
    assert_succeeded(world);
    let combined = combined_output(world);
    assert!(
        combined.contains(SERVICE_ID),
        "the preview must name the record it would remove:\n{combined}"
    );
    assert!(
        combined.contains(ORPHAN_ID),
        "the preview must name the leftover engine state file:\n{combined}"
    );
    assert!(
        combined.contains("Nothing was removed. Re-run without --dry-run to remove."),
        "the preview must say it removed nothing:\n{combined}"
    );
}

#[then("the leftover engine state file is gone")]
async fn orphan_gone(world: &mut E2eWorld) {
    let orphan = engine_state_dir(world).join(format!("{ORPHAN_ID}.json"));
    assert!(
        !orphan.exists(),
        "the leftover engine state file must be swept:\n{}",
        combined_output(world)
    );
}

#[then("the CLI names the file it could not remove and exits non-zero")]
async fn names_the_unremovable_file(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    let combined = combined_output(world);
    assert!(
        rc != 0,
        "a prune that could not delete a file must fail the command, got rc=0:\n{combined}"
    );
    let stuck = engine_state_dir(world).join(format!("{SERVICE_ID}.json"));
    assert!(
        combined.contains("file(s) could not be removed"),
        "the failure must be reported, not swallowed:\n{combined}"
    );
    assert!(
        combined.contains(&stuck.display().to_string()),
        "the report must name the path that survived:\n{combined}"
    );
    assert!(
        combined.contains("Re-running is safe: everything already removed stays removed."),
        "a half-done destructive run must say whether repeating it is safe:\n{combined}"
    );
    // The plan is printed before anything is deleted; a failure after that must
    // not cost the user the account of what the run was going to do.
    assert!(
        combined.contains("local server record(s) would be removed"),
        "the plan must still be on screen:\n{combined}"
    );
}

/// Everything except the one stuck path: a single unremovable file must not
/// strand the other three, which is the whole reason the removal collects
/// failures instead of returning at the first one.
#[then("the record's other files are gone")]
async fn other_record_files_gone(world: &mut E2eWorld) {
    let stuck = engine_state_dir(world).join(format!("{SERVICE_ID}.json"));
    for path in record_files(world) {
        if path == stuck {
            continue;
        }
        assert!(
            !path.exists(),
            "{} should have been deleted despite the failure elsewhere:\n{}",
            path.display(),
            combined_output(world)
        );
    }
}

/// The audit line is written before the non-zero exit is raised. A refactor that
/// bailed on the first failure instead would delete files and leave no record of
/// having done so.
#[then("the prune is still recorded in the audit log")]
async fn prune_is_audited(world: &mut E2eWorld) {
    let log = data_dir(world).join("logs").join("cli-lifecycle.log");
    let text = std::fs::read_to_string(&log).unwrap_or_else(|error| {
        panic!(
            "failed to read {}: {error}\n{}",
            log.display(),
            combined_output(world)
        )
    });
    assert!(
        text.contains("action=prune_records"),
        "a run that deleted files must be audited even when it exits non-zero, got:\n{text}"
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
