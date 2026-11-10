// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for stopping a managed service: `rocm services stop <id> --yes`
//! (confirmed, and unconfirmed through the `ROCM_E2E_STOP_OUTCOME` seam), and
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
/// The port the planted record in `plant_server_record` was started on.
const SERVER_PORT: u16 = 9;
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
    plant_server_with_running_process(world, "0.0.0.0", true);
}

/// The same server bound to loopback and started without an API key, so no
/// endpoint key file exists for it.
#[given("a managed server without an endpoint key whose recorded process is running")]
async fn keyless_managed_server_with_running_process(world: &mut E2eWorld) {
    plant_server_with_running_process(world, "127.0.0.1", false);
}

/// The server a stop leaves behind when it is bound publicly: its endpoint key
/// is dropped by the stop, so a record read back later has none.
#[given("a public managed server whose endpoint key is gone and whose recorded process is running")]
async fn public_managed_server_without_its_key(world: &mut E2eWorld) {
    plant_server_with_running_process(world, "0.0.0.0", false);
}

/// A loopback server that was launched with `--require-api-key`, which a
/// `rocm serve` of a loopback host does not otherwise ask for, and whose key is
/// gone: the other way a service needs a key besides a public bind. The record
/// says it required one; nothing else about it differs from the public server's.
#[given(
    "a loopback managed server that required an API key, whose key is gone and whose recorded process is running"
)]
async fn loopback_server_that_required_a_key_without_it(world: &mut E2eWorld) {
    plant_server_with_running_process(world, "127.0.0.1", false);
    let manifest = manifest_path(world);
    let mut record = read_record(world);
    record["requires_api_key"] = serde_json::json!(true);
    std::fs::write(
        &manifest,
        serde_json::to_vec_pretty(&record).expect("failed to serialize record"),
    )
    .expect("failed to rewrite service record");
}

fn plant_server_with_running_process(world: &mut E2eWorld, host: &str, with_key: bool) {
    let process = Command::new("sleep")
        .arg("300")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to start the process to record");
    let ticks = start_ticks(process.id());
    plant_server_record(world, host, with_key, process, Some(ticks));
}

/// A ready, publicly bound server whose recorded PID is a live process that is
/// not the server: what a recycled PID looks like. The record's start-time token
/// is not the process's own, so the stop has to refuse the PID rather than signal
/// it. The token is the process's real one offset by one on Linux, and `1` (a
/// `FILETIME` in 1601, which no live process has) on Windows.
#[given("a managed server whose recorded process id now belongs to a different process")]
async fn managed_server_with_a_recycled_pid(world: &mut E2eWorld) {
    let (program, args): (&str, &[&str]) = if cfg!(windows) {
        ("ping", &["-n", "300", "127.0.0.1"])
    } else {
        ("sleep", &["300"])
    };
    let bystander = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to start the process that holds the recorded PID");
    let stale_ticks = if cfg!(windows) {
        1
    } else {
        start_ticks(bystander.id()).wrapping_add(1)
    };
    plant_server_record(world, "0.0.0.0", true, bystander, Some(stale_ticks));
}

/// The stop's promise about the process that holds the recorded PID, checked on
/// the process: it was not signalled, so it is still running.
#[then("the process now holding the recorded process id is still running")]
async fn process_holding_the_pid_still_runs(world: &mut E2eWorld) {
    let process = world
        .recorded_process
        .as_mut()
        .expect("scenario recorded no process");
    assert!(
        process
            .try_wait()
            .expect("failed to check the recorded process")
            .is_none(),
        "the stop signalled a process that is not the recorded server"
    );
}

/// Where the parent the Windows scenario records writes the PID of the process
/// it started, for the scenario to read.
fn descendant_pid_path(world: &E2eWorld) -> PathBuf {
    services_dir(world).join("descendant.pid")
}

/// A ready, publicly bound server whose recorded supervisor is a parent process
/// that has started a child of its own, recorded the way a Windows record was
/// before Windows recorded a creation time: the PID alone, with no start-time
/// token. (A record written now carries one; this scenario plants the tokenless
/// shape on purpose, the one a PID cannot be refuted by.)
///
/// The parent is PowerShell, the child `ping` for five minutes: both ship with
/// Windows, need nothing installed, and are disposable. PowerShell starts the
/// child itself, so the child's parent is the recorded PID, which is the
/// relationship `taskkill /T` follows; it writes the child's PID where the
/// scenario can read it and then stays alive. Both outlive the scenario only if
/// the stop fails to take them, and the World's teardown reaps the parent.
#[given("a managed server whose recorded process has started another process")]
async fn managed_server_whose_process_started_another(world: &mut E2eWorld) {
    let services = services_dir(world);
    std::fs::create_dir_all(&services).expect("failed to create services dir");
    let pid_file = descendant_pid_path(world);
    let process = spawn_parent_with_descendant(&pid_file);
    let pid = process.id();
    plant_server_record(world, "0.0.0.0", true, process, None);

    // The premise, so the Then step cannot pass for the wrong reason: the
    // descendant exists and is running, and so is the parent.
    let descendant = wait_for_descendant_pid(&pid_file);
    // Kept first, with its creation time where the host reads one, so teardown
    // can end the child on its own even when the stop (or a step) fails and the
    // parent is gone: a later stop cannot recover a child from a dead parent.
    world.recorded_descendant = Some(rocm_core::ProcessIdentity::capture(descendant));
    assert_ne!(descendant, pid, "the child must be a different process");
    assert!(
        windows_process_is_running(descendant),
        "premise: the process the recorded one started must be running"
    );
    assert!(
        world
            .recorded_process
            .as_mut()
            .expect("scenario recorded no process")
            .try_wait()
            .expect("failed to check the recorded process")
            .is_none(),
        "premise: the recorded process must be running"
    );
}

fn spawn_parent_with_descendant(pid_file: &std::path::Path) -> std::process::Child {
    // A single-quoted PowerShell string: an apostrophe in the path is doubled.
    let pid_file = pid_file.display().to_string().replace('\'', "''");
    let script = format!(
        "$child = Start-Process -FilePath ping.exe -ArgumentList '-n','300','127.0.0.1' \
         -WindowStyle Hidden -PassThru; \
         Set-Content -LiteralPath '{pid_file}' -Value $child.Id; \
         Start-Sleep -Seconds 300"
    );
    Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &script])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to start the process to record")
}

/// The child's PID, once the parent has written it.
fn wait_for_descendant_pid(pid_file: &std::path::Path) -> u32 {
    let deadline = std::time::Instant::now() + std::time::Duration::from_mins(1);
    loop {
        if let Some(pid) = std::fs::read_to_string(pid_file)
            .ok()
            .and_then(|text| text.trim().parse().ok())
        {
            return pid;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the recorded process never wrote the PID of the process it started"
        );
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}

/// Whether Windows has a process `pid`, by `tasklist`: a match prints a CSV
/// row naming it, and no match prints an INFO line without one. A `tasklist`
/// that fails has said nothing about the process, so the scenario fails rather
/// than read it as "exited" (`e2e_cucumber::tasklist`).
fn windows_process_is_running(pid: u32) -> bool {
    let output = Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
        .output()
        .expect("failed to run tasklist");
    e2e_cucumber::tasklist::tasklist_reports_running(
        output.status.success(),
        &String::from_utf8_lossy(&output.stdout),
        &String::from_utf8_lossy(&output.stderr),
        pid,
    )
    .unwrap_or_else(|message| panic!("{message}"))
}

/// Write the record for a ready server whose supervisor is `process`, with
/// `ticks` as its start-time token when there is one, and take `process` into
/// the World so the scenario (and its teardown) can check and reap it.
fn plant_server_record(
    world: &mut E2eWorld,
    host: &str,
    with_key: bool,
    process: std::process::Child,
    ticks: Option<u64>,
) {
    let pid = process.id();
    world.recorded_process = Some(process);

    let services = services_dir(world);
    std::fs::create_dir_all(&services).expect("failed to create services dir");
    let manifest = manifest_path(world);
    let record = serde_json::json!({
        "service_id": SERVICE_ID,
        "engine": "vllm",
        "model_ref": "qwen",
        "canonical_model_id": "Qwen/Qwen3.5",
        "host": host,
        // Discard port: closed, so nothing can answer on the server's behalf, and
        // outside the range `rocm serve` chooses from itself, as a server that was
        // started on an explicit port and published behind a mapping would be.
        "port": SERVER_PORT,
        "endpoint_url": format!("http://{host}:{SERVER_PORT}/v1"),
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
    if with_key {
        std::fs::write(endpoint_key_path(world), "e2e-endpoint-key")
            .expect("failed to write endpoint key");
    }
}

/// Script the next `rocm` stop's verified termination to report `outcome` for
/// the recorded process, without signalling it — the `e2e-test-hooks` seam
/// `ROCM_E2E_STOP_OUTCOME`. Consumed by that one invocation.
fn script_next_stop_outcome(world: &mut E2eWorld, outcome: &str) {
    world
        .command_env
        .push(("ROCM_E2E_STOP_OUTCOME", outcome.into()));
}

#[given("the next stop cannot confirm that process exited")]
async fn next_stop_times_out(world: &mut E2eWorld) {
    script_next_stop_outcome(world, "timed_out");
}

#[given("the next stop cannot verify that process as the server's")]
async fn next_stop_unverified(world: &mut E2eWorld) {
    script_next_stop_outcome(world, "unverified");
}

// ── Given: an engine child the scenario controls ────────────────────

/// The standalone daemon's file name: `rocmd`, with the platform's suffix.
const STANDALONE_ROCMD: &str = if cfg!(windows) { "rocmd.exe" } else { "rocmd" };
/// What the stand-in engine launcher records, one line per time it is run.
const ENGINE_INVOCATIONS: &str = "engine-invocations";
/// What the stand-in engine launcher read on its stdin, over every run.
const ENGINE_REQUESTS: &str = "engine-requests";

/// The directory holding the standalone daemon copy and the stand-in `rocm`.
fn standalone_dir(world: &E2eWorld) -> PathBuf {
    world
        .isolated_root
        .as_ref()
        .expect("scenario has no isolated root")
        .path()
        .join("standalone-daemon")
}

/// Plant a copy of the real `rocmd` with a stand-in `rocm` beside it.
///
/// A `rocmd` named exactly that reaches an engine through the `rocm` next to it,
/// so the engine child the daemon runs for its `Stop` request is `behaviour`: a
/// shell script that first records its argv and what it read on stdin (into
/// [`ENGINE_INVOCATIONS`] and [`ENGINE_REQUESTS`] beside it), then does what the
/// scenario says. The copy is a plain file, not a link, because the daemon looks
/// for its sibling next to where its executable really is.
fn install_standalone_daemon(world: &E2eWorld, behaviour: &str) {
    let dir = standalone_dir(world);
    std::fs::create_dir_all(&dir).expect("failed to create the standalone daemon dir");
    let rocmd = std::fs::read(crate::rocmd_binary()).expect("failed to read rocmd");
    rocm_core::fake_executable::write_fake_executable(&dir.join(STANDALONE_ROCMD), rocmd)
        .expect("failed to plant the standalone daemon");
    let script = format!(
        "#!/bin/sh\n\
         dir=$(dirname \"$0\")\n\
         echo \"$*\" >> \"$dir/{ENGINE_INVOCATIONS}\"\n\
         cat >> \"$dir/{ENGINE_REQUESTS}\"\n\
         {behaviour}\n"
    );
    rocm_core::fake_executable::write_fake_executable(&dir.join("rocm"), script)
        .expect("failed to plant the stand-in rocm");
}

#[given("the engine answers the stop request and then exits with a failing status")]
async fn engine_answers_then_fails(world: &mut E2eWorld) {
    install_standalone_daemon(
        world,
        "echo '{\"ok\":true,\"data\":{\"stopped\":true,\"graceful\":true}}'\n\
         echo 'stand-in engine: exiting with a failing status after answering' >&2\n\
         exit 3",
    );
}

#[given(
    "the engine prints something that is not an answer, complains on stderr and exits with a failing status"
)]
async fn engine_prints_garbage_and_fails(world: &mut E2eWorld) {
    install_standalone_daemon(
        world,
        "printf 'this is not an answer'\n\
         echo 'engine exploded: no container runtime' >&2\n\
         exit 1",
    );
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
    let (stdout, stderr, rc) =
        crate::run_rocm_with_scenario_env(world, &["services", "stop", SERVICE_ID, "--yes"]);
    record(world, stdout, stderr, rc);
}

#[when("the user restarts it with --yes")]
async fn restart_with_yes(world: &mut E2eWorld) {
    let (stdout, stderr, rc) =
        crate::run_rocm_with_scenario_env(world, &["services", "restart", SERVICE_ID, "--yes"]);
    record(world, stdout, stderr, rc);
}

#[when("the daemon's stop_server tool stops that server")]
async fn stop_server_tool(world: &mut E2eWorld) {
    run_stop_server_tool(world, crate::rocmd_binary());
}

/// The same tool, run from the copy of `rocmd` that has a stand-in `rocm` beside
/// it (see [`install_standalone_daemon`]), so the engine's `Stop` request reaches
/// a child the scenario controls.
#[when("the standalone daemon's stop_server tool stops that server")]
async fn standalone_stop_server_tool(world: &mut E2eWorld) {
    let rocmd = standalone_dir(world).join(STANDALONE_ROCMD);
    assert!(
        rocmd.is_file(),
        "no standalone daemon was installed: {}",
        rocmd.display()
    );
    run_stop_server_tool(world, rocmd);
}

fn run_stop_server_tool(world: &mut E2eWorld, rocmd: PathBuf) {
    let mut command = Command::new(rocmd);
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
// gap in that renderer, so this assertion covers only what the command
// actually prints; the on-disk status below is what proves the stop itself.
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

#[then(
    "the CLI refuses the restart because it would reopen the public server without authentication"
)]
async fn restart_refused_without_authentication(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    let combined = combined_output(world);
    assert_ne!(
        rc, 0,
        "a restart that would reopen the server anonymously must fail:\n{combined}"
    );
    assert!(
        combined.contains(
            "managed service is bound to the public host `0.0.0.0` but has no endpoint API key, \
             so restarting it would reopen it without authentication."
        ),
        "{combined}"
    );
}

#[then(
    "the CLI refuses the restart because it would reopen the server that required an API key without authentication"
)]
async fn restart_refused_for_a_required_key(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    let combined = combined_output(world);
    assert_ne!(
        rc, 0,
        "a restart that would reopen the server anonymously must fail:\n{combined}"
    );
    assert!(
        combined.contains(
            "managed service was launched with `--require-api-key` but has no endpoint API key, \
             so restarting it would reopen it without authentication."
        ),
        "{combined}"
    );
    // It is the key requirement that is refused, not a public bind: the server is
    // on loopback, and is not told it is bound to a public host.
    assert!(!combined.contains("public host"), "{combined}");
}

/// What the refused restart must leave of the record's own claim: that the
/// server required a key. The restart changed nothing, so it still does.
#[then("the server's record still says it required an API key")]
async fn record_still_requires_the_key(world: &mut E2eWorld) {
    let record = read_record(world);
    assert_eq!(record["requires_api_key"], true, "{record}");
    assert_eq!(record["host"], "127.0.0.1", "{record}");
}

/// The `rocm serve` command the refusal names, run against the real parser:
/// its `<model>` placeholder filled in, and `--help` last, so the parse ends in
/// help only if every word before it was accepted. A command with a flag `rocm`
/// does not have is shown to be rejected the same way, so the check cannot pass
/// by `--help` alone.
#[then("the refusal names a serve command that rocm accepts")]
async fn refusal_names_an_accepted_serve_command(world: &mut E2eWorld) {
    check_advised_serve_command(
        world,
        &format!("rocm serve <model> --host 0.0.0.0 --port {SERVER_PORT} --allow-public-bind"),
    );
}

/// The same check for the server that asked for a key on a loopback bind: the
/// command carries the flag it asked for and nothing that moves it off the
/// default host, which is where it was.
#[then("the refusal names a serve command that asks for the key again and that rocm accepts")]
async fn refusal_names_an_accepted_serve_command_with_the_key_flag(world: &mut E2eWorld) {
    let command = check_advised_serve_command(
        world,
        &format!("rocm serve <model> --port {SERVER_PORT} --require-api-key"),
    );
    assert!(
        !command.contains("--host") && !command.contains("--allow-public-bind"),
        "a loopback server is not told to bind elsewhere: {command}"
    );
}

/// Runs the `rocm serve` command the refusal names against the real parser, and
/// returns it. `expected` is what it must contain to bring the server back where
/// it was.
fn check_advised_serve_command(world: &mut E2eWorld, expected: &str) -> String {
    let combined = combined_output(world);
    let advised: Vec<&str> = combined
        .split('`')
        .skip(1)
        .step_by(2)
        .filter(|span| span.starts_with("rocm serve <model>"))
        .collect();
    assert_eq!(
        advised.len(),
        1,
        "the refusal must name exactly one `rocm serve <model>` command:\n{combined}"
    );
    assert!(
        advised[0].starts_with(expected),
        "the command must bring the server back where it was, on the port it recorded ({expected}): {}",
        advised[0]
    );
    let mut args: Vec<String> = advised[0]
        .split_whitespace()
        .skip(1)
        .map(|word| match word {
            "<model>" => "Qwen/Qwen3-0.6B".to_owned(),
            other => other.to_owned(),
        })
        .collect();
    args.push("--help".to_owned());
    let words: Vec<&str> = args.iter().map(String::as_str).collect();
    let (stdout, stderr, rc) = crate::run_rocm(world, &words);
    assert_eq!(rc, 0, "rocm rejected {words:?}:\n{stdout}\n{stderr}");

    let mut bogus = words.clone();
    bogus.insert(bogus.len() - 1, "--no-such-flag");
    let (_, _, rc) = crate::run_rocm(world, &bogus);
    assert_ne!(
        rc, 0,
        "a flag rocm does not have must be rejected before --help: {bogus:?}"
    );
    advised[0].to_owned()
}

/// The stop the refusal names comes before the launch it names, is the exact
/// command for this server, and is one `rocm` accepts: run with `--help` last,
/// with a bogus flag as the negative control, so nothing is stopped by it. That
/// the server is still running afterwards is what the next step checks.
#[then("the refusal names the stop that has to come first")]
async fn refusal_names_the_stop_first(world: &mut E2eWorld) {
    let combined = combined_output(world);
    let stops: Vec<&str> = combined
        .split('`')
        .skip(1)
        .step_by(2)
        .filter(|span| span.starts_with("rocm services stop"))
        .collect();
    assert_eq!(
        stops.len(),
        1,
        "the refusal must name exactly one `rocm services stop` command:\n{combined}"
    );
    assert_eq!(
        stops[0],
        format!("rocm services stop {SERVICE_ID} --yes"),
        "{combined}"
    );
    assert!(
        combined.find(stops[0]).unwrap() < combined.find("rocm serve <model>").unwrap(),
        "the stop must come before the launch:\n{combined}"
    );
    let mut args: Vec<String> = stops[0]
        .split_whitespace()
        .skip(1)
        .map(str::to_owned)
        .collect();
    args.push("--help".to_owned());
    let words: Vec<&str> = args.iter().map(String::as_str).collect();
    let (stdout, stderr, rc) = crate::run_rocm(world, &words);
    assert_eq!(rc, 0, "rocm rejected {words:?}:\n{stdout}\n{stderr}");
    let mut bogus = words.clone();
    bogus.insert(bogus.len() - 1, "--no-such-flag");
    let (_, _, rc) = crate::run_rocm(world, &bogus);
    assert_ne!(
        rc, 0,
        "a flag rocm does not have must be rejected before --help: {bogus:?}"
    );
}

#[then("the server is left running, with no stop recorded and no key file")]
async fn server_left_as_it_was(world: &mut E2eWorld) {
    let pid = recorded_pid(world);
    let process = world
        .recorded_process
        .as_mut()
        .expect("scenario recorded no process");
    assert!(
        process
            .try_wait()
            .expect("failed to check the recorded process")
            .is_none(),
        "the refused restart stopped the running server"
    );
    // Not `ready` any more: the CLI's liveness refresh reads the planted server's
    // closed port as `starting` before it gets as far as the restart. What a
    // refused restart must not do is mark it stopped or drop its process.
    let record = read_record(world);
    assert_ne!(record["status"], "stopped", "{record}");
    assert_eq!(record["supervisor_pid"], u64::from(pid), "{record}");
    assert!(
        record["stop_requested_unix_ms"].is_null(),
        "a refused restart stops nothing, so it records no stop request: {record}"
    );
    let key = endpoint_key_path(world);
    assert!(
        !key.exists(),
        "a refused restart must not create a key: {}",
        key.display()
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

/// The standalone `rocmd` reaches an engine through the `rocm` beside it (it has
/// no engine verb of its own), so its `Stop` request is answered rather than
/// failing with the standalone binary's "unrecognized subcommand". The report
/// carries the engine's answer or the error that took its place.
///
/// The answer is `stopped: false`, and that is the right one here, not a gap: the
/// planted record names no engine state with a process id, so the real `vllm`
/// engine has nothing it can confirm stopping (`stop_service` answers `false`
/// when no PID is recorded). The recorded supervisor, which the daemon itself
/// signals, is what the other steps check. Asserting the whole answer, not just
/// that it is a boolean, is what tells it from an engine that was never reached.
#[then("the tool asked the engine to stop and the engine answered")]
async fn engine_answered_the_stop(world: &mut E2eWorld) {
    let stdout = world.cli_output.as_deref().unwrap_or_default();
    let report: serde_json::Value =
        serde_json::from_str(stdout).unwrap_or_else(|error| panic!("{error}: {stdout}"));
    let engine_stop = &report["result"]["engine_stop"];
    assert_eq!(engine_stop["attempted"], true, "{report}");
    assert!(
        engine_stop.get("error").is_none(),
        "the engine's answer was an error instead: {report}"
    );
    assert_eq!(
        *engine_stop,
        serde_json::json!({"attempted": true, "stopped": false, "graceful": false}),
        "{report}"
    );
}

/// The stop report `rocmd sandbox-tool stop_server` printed, with the tool's own
/// exit status checked first: both engine endings below leave the stop itself
/// succeeding.
fn tool_report(world: &E2eWorld) -> serde_json::Value {
    let stdout = world.cli_output.as_deref().unwrap_or_default();
    assert_eq!(
        world.cli_rc,
        Some(0),
        "rocmd failed:\n{stdout}\n{}",
        world.cli_stderr.as_deref().unwrap_or_default()
    );
    serde_json::from_str(stdout).unwrap_or_else(|error| panic!("{error}: {stdout}"))
}

/// The stand-in engine was run once, as `__engine-stdio vllm`, and was sent one
/// line: an engine `stop` request for the planted server. Reads what the
/// stand-in itself recorded, so it holds for the child the daemon ran, not for
/// what the daemon says it ran.
#[then("the engine received one stop request for that server over stdio")]
async fn engine_received_one_stop_request(world: &mut E2eWorld) {
    let dir = standalone_dir(world);
    let invocations = std::fs::read_to_string(dir.join(ENGINE_INVOCATIONS))
        .expect("the stand-in engine was never run");
    assert_eq!(
        invocations,
        format!("__engine-stdio {ENGINE}\n"),
        "the daemon must run the rocm beside it once, as `__engine-stdio <engine>`"
    );
    let requests = std::fs::read_to_string(dir.join(ENGINE_REQUESTS))
        .expect("the stand-in engine recorded no request");
    assert_eq!(
        requests.lines().count(),
        1,
        "one request, one line: {requests:?}"
    );
    let request: serde_json::Value = serde_json::from_str(&requests)
        .unwrap_or_else(|error| panic!("the request is not JSON ({error}): {requests:?}"));
    assert_eq!(request["method"], "stop", "{request}");
    assert_eq!(request["payload"]["service_id"], SERVICE_ID, "{request}");
}

/// What the engine printed before it exited non-zero is what the report carries:
/// a stop it answered, with no error taken from the exit status.
#[then("the tool reports the engine's answer to the stop, not the status it exited with")]
async fn engine_answer_survives_a_failing_exit(world: &mut E2eWorld) {
    let report = tool_report(world);
    assert_eq!(
        report["result"]["engine_stop"],
        serde_json::json!({"attempted": true, "stopped": true, "graceful": true}),
        "{report}"
    );
    assert_eq!(report["status"], "stopped", "{report}");
    assert_eq!(report["result"]["stopped"], true, "{report}");
}

/// An answer that cannot be read is reported as that, together with the status
/// the engine exited with and what it said on stderr: the status says the answer
/// was cut short, the stderr why.
#[then("the tool reports the stop request failed, with the engine's exit status and stderr")]
async fn unreadable_answer_reported_with_stderr(world: &mut E2eWorld) {
    let report = tool_report(world);
    let engine_stop = &report["result"]["engine_stop"];
    assert_eq!(engine_stop["attempted"], true, "{report}");
    assert_eq!(
        engine_stop["error"],
        "failed to parse engine response envelope \
         (engine stdio process exited with code 1); \
         stderr: engine exploded: no container runtime",
        "{report}"
    );
    assert!(engine_stop.get("stopped").is_none(), "{report}");
}

/// For a `vllm` server the engine's `Stop` is advisory: the recorded process
/// decides, so a failed request does not leave the server unconfirmed.
#[then("the tool still reports the server stopped")]
async fn tool_still_reports_stopped(world: &mut E2eWorld) {
    let report = tool_report(world);
    assert_eq!(report["status"], "stopped", "{report}");
    assert_eq!(report["result"]["stopped"], true, "{report}");
}

#[then("the recorded process is no longer running")]
async fn recorded_process_gone(world: &mut E2eWorld) {
    assert_recorded_process_exited(world);
}

/// The Windows tree kill, observed: the recorded process AND the one it started
/// are gone after the stop. Without the `taskkill /T` step the stop ends only
/// the recorded process (`TerminateProcess` on the root), and the child keeps
/// running, which is what the second check catches.
#[then("the recorded process and the process it started are no longer running")]
async fn recorded_process_and_its_child_gone(world: &mut E2eWorld) {
    assert_recorded_process_exited(world);
    let pid = wait_for_descendant_pid(&descendant_pid_path(world));
    // `taskkill` returns once it has asked for the termination; give the exit
    // a moment to land before calling the process a survivor.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    while windows_process_is_running(pid) {
        assert!(
            std::time::Instant::now() < deadline,
            "the stop took the recorded process down but left the process it \
             started (PID {pid}) running: the tree kill did not reach it"
        );
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
}

/// The report says one process was stopped (the recorded one: a tree kill that
/// already took the root is reported as forced, so it counts), checked next to
/// the processes themselves rather than instead of them.
#[then("the report counts one stopped process")]
async fn report_counts_one_stopped_process(world: &mut E2eWorld) {
    let combined = combined_output(world);
    assert!(combined.contains("stopped processes: 1"), "{combined}");
}

fn assert_recorded_process_exited(world: &mut E2eWorld) {
    let process = world
        .recorded_process
        .as_mut()
        .expect("scenario recorded no process");
    // The report's claim, checked against the process itself: it has exited
    // (and is reaped here), rather than merely having been signalled. The stop
    // confirms an exit from the OS's exit code, and this handle is signalled by
    // the process object's teardown, so allow that teardown a few seconds on a
    // loaded host before calling a process a survivor.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let exited = process
            .try_wait()
            .expect("failed to check the recorded process");
        if exited.is_some() {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the stop reported the process gone, but it is still running"
        );
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
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

/// An unconfirmed stop fails, on stderr, and never claims the server stopped.
fn assert_unconfirmed_stop_reported(world: &E2eWorld) -> String {
    let rc = world.cli_rc.expect("no command rc recorded");
    let combined = combined_output(world);
    assert_eq!(rc, 1, "an unconfirmed stop must exit 1:\n{combined}");
    let stderr = world.cli_stderr.as_deref().unwrap_or_default();
    assert!(
        stderr.contains("Local server stop not confirmed"),
        "the stop must say, on stderr, that it was not confirmed:\n{combined}"
    );
    assert!(
        !combined.contains("Local server stopped"),
        "an unconfirmed stop must not claim the server stopped:\n{combined}"
    );
    combined
}

/// No line of an unconfirmed-stop report offers a command to run the stop
/// again — no `retry:` line, and no `services stop` anywhere: a second stop
/// can confirm without reaching a surviving process under the recorded one.
fn assert_no_retry_line(combined: &str) {
    assert!(
        !combined.lines().any(|line| {
            line.trim_start().starts_with("retry:") || line.contains("services stop")
        }),
        "an unconfirmed stop names no stop command: {combined}"
    );
}

#[then("the CLI reports the stop as not confirmed for a process that did not exit")]
async fn reports_unconfirmed_timed_out(world: &mut E2eWorld) {
    let combined = assert_unconfirmed_stop_reported(world);
    let pid = recorded_pid(world);
    assert!(
        combined.contains(&format!(
            "still running: process {pid} (supervisor) or a process under it did not exit"
        )),
        "{combined}"
    );
    assert_no_retry_line(&combined);
}

#[then("the CLI reports the stop as not confirmed for a process it could not verify")]
async fn reports_unconfirmed_unverified(world: &mut E2eWorld) {
    let combined = assert_unconfirmed_stop_reported(world);
    let pid = recorded_pid(world);
    assert!(
        combined.contains(&format!("not signaled: process {pid} (supervisor)")),
        "{combined}"
    );
    assert!(
        combined.contains(&format!("check: `ps -p {pid} -o pid,args`")),
        "{combined}"
    );
    assert_no_retry_line(&combined);
}

#[then("the server's record still names its process and the stop request")]
async fn record_keeps_process_and_marker(world: &mut E2eWorld) {
    let record = read_record(world);
    assert_eq!(
        record["supervisor_pid"],
        u64::from(recorded_pid(world)),
        "{record}"
    );
    assert!(
        !record["stop_requested_unix_ms"].is_null(),
        "the unconfirmed stop keeps its request recorded: {record}"
    );
    assert_ne!(
        record["status"], "stopped",
        "a stop that confirmed nothing must not mark the record stopped: {record}"
    );
}

/// The report's `kept:` line, checked against the key file it describes.
#[then(
    "the report says the record and the endpoint key were kept, and the key file is still there"
)]
async fn kept_line_names_the_key(world: &mut E2eWorld) {
    let combined = combined_output(world);
    assert!(
        combined.lines().any(|line| line
            == "  kept: the server record, which still names its processes, and its endpoint key, which the server may still be enforcing"),
        "{combined}"
    );
    let key = endpoint_key_path(world);
    assert!(
        key.exists(),
        "the report says the key was kept, but {} is gone",
        key.display()
    );
}

#[then("the report says only the record was kept, and there is no key file")]
async fn kept_line_omits_the_key(world: &mut E2eWorld) {
    let combined = combined_output(world);
    assert!(
        combined
            .lines()
            .any(|line| line == "  kept: the server record, which still names its processes"),
        "{combined}"
    );
    assert!(!combined.contains("endpoint key"), "{combined}");
    let key = endpoint_key_path(world);
    assert!(
        !key.exists(),
        "a keyless server has a key file: {}",
        key.display()
    );
}

#[then("the CLI reports the stop as not confirmed and names no stop command")]
async fn reports_unconfirmed_no_command(world: &mut E2eWorld) {
    let combined = assert_unconfirmed_stop_reported(world);
    assert_no_retry_line(&combined);
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
