// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocm logs`. Plant deterministic command-log files in the isolated
//! data dir, then assert on the search output. `logs` reads only the TAIL of each
//! file, so the matching lines are planted WITHIN the tail window and the
//! assertion is on the MATCH COUNT (deterministic), not the recent-line total
//! (which also counts other log sources). Contracts verified against the running
//! Linux binary (EAI-8072). No GPU or network — mock lane.

use std::fmt::Write as _;

use cucumber::{given, then, when};

use crate::E2eWorld;

/// The topic term planted into the log and searched for. Distinctive so it can't
/// collide with anything the CLI itself writes into the isolated log dir.
const TOPIC: &str = "E2ENEEDLE";
/// Number of matching lines planted — asserted exactly in the search result.
const MATCH_COUNT: usize = 9;

#[given("recorded command logs containing several lines about a topic")]
async fn plant_command_logs(world: &mut E2eWorld) {
    plant_logs(world, false);
}

#[given(
    "recorded command logs containing several lines about a topic and one byte that is not valid text"
)]
async fn plant_command_logs_with_an_invalid_byte(world: &mut E2eWorld) {
    plant_logs(world, true);
}

/// Plant one action log. With `invalid_byte`, an unrelated leading line carries
/// a lone `0xff` — a byte no UTF-8 text contains — which the matching lines
/// after it must survive.
fn plant_logs(world: &E2eWorld, invalid_byte: bool) {
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let cli_logs = root.path().join("data").join("logs").join("cli");
    std::fs::create_dir_all(&cli_logs).expect("failed to create cli logs dir");
    // Write the matching lines LAST so they fall within the per-file tail window
    // `rocm logs` reads (a few leading non-matching lines are harmless context).
    let mut body: Vec<u8> = Vec::new();
    for i in 0..3 {
        let mut line = String::new();
        let _ = writeln!(line, "2026-01-01T00:00:0{i} unrelated startup line");
        if invalid_byte && i == 1 {
            body.extend_from_slice(b"2026-01-01T00:00:01 unrelated line with a stray \xff byte\n");
        } else {
            body.extend_from_slice(line.as_bytes());
        }
    }
    for i in 1..=MATCH_COUNT {
        let mut line = String::new();
        let _ = writeln!(
            line,
            "2026-01-01T00:01:00 event {TOPIC} occurred number {i}"
        );
        body.extend_from_slice(line.as_bytes());
    }
    std::fs::write(cli_logs.join("e2e-probe.log"), body).expect("failed to write log file");
}

/// How far past the 4 MiB a log view reads (`LOG_VIEW_MAX_TAIL_READ` in
/// `rocm-core`) the planted line reaches.
const LOG_VIEW_BOUND_BYTES: usize = 4 * 1024 * 1024;
const OVERSIZE_LINE_BYTES: usize = LOG_VIEW_BOUND_BYTES + 1024;

/// Plant the record of a stopped local server (the same planting the record
/// cleanup scenarios use) and replace its log with `bytes`.
fn plant_server_record_with_log(world: &E2eWorld, bytes: &[u8]) {
    crate::e2e::service_cleanup_steps::plant_record(world, "failed");
    std::fs::write(server_log_path(world), bytes).expect("failed to write the server log");
}

fn server_log_path(world: &E2eWorld) -> std::path::PathBuf {
    crate::e2e::service_cleanup_steps::services_dir(world).join(format!(
        "{}.log",
        crate::e2e::service_cleanup_steps::SERVICE_ID
    ))
}

/// The one saved ComfyUI log the ComfyUI log scenarios plant.
fn comfyui_log_path(world: &E2eWorld) -> std::path::PathBuf {
    let root = world.isolated_root.as_ref().expect("no isolated root");
    root.path()
        .join("data")
        .join("apps")
        .join("comfyui")
        .join("logs")
        .join("start-e2e.log")
}

fn plant_comfyui_log(world: &E2eWorld, bytes: &[u8]) {
    let log = comfyui_log_path(world);
    std::fs::create_dir_all(log.parent().expect("log has a parent"))
        .expect("failed to create the ComfyUI logs dir");
    std::fs::write(&log, bytes).expect("failed to write the ComfyUI log");
}

#[given("a ComfyUI log holding one line longer than the log view reads")]
async fn comfyui_with_an_oversize_line(world: &mut E2eWorld) {
    plant_comfyui_log(world, &vec![b'x'; OVERSIZE_LINE_BYTES]);
}

#[given("a ComfyUI log file that is empty")]
async fn comfyui_with_an_empty_log(world: &mut E2eWorld) {
    plant_comfyui_log(world, b"");
}

#[given("a local server record whose log holds one line longer than the log view reads")]
async fn server_with_an_oversize_line(world: &mut E2eWorld) {
    // No final newline: a server mid-line when it died, or one that logs
    // without line breaks.
    plant_server_record_with_log(world, &vec![b'x'; OVERSIZE_LINE_BYTES]);
}

#[given("a local server record whose log file is empty")]
async fn server_with_an_empty_log(world: &mut E2eWorld) {
    plant_server_record_with_log(world, b"");
}

// ── When ───────────────────────────────────────────────────────────

#[when("the user reads that server's log")]
async fn read_server_log(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &[
            "services",
            "logs",
            crate::e2e::service_cleanup_steps::SERVICE_ID,
        ],
    );
    record(world, stdout, stderr, rc);
}

#[when("a tool-using client reads that server's log through the service_logs tool")]
async fn read_server_log_through_the_tool(world: &mut E2eWorld) {
    // The tool is read-only, so no `--allow-mutation`. The exit code is checked
    // by the `Then` steps that parse the reply.
    let arguments =
        serde_json::json!({ "service_id": crate::e2e::service_cleanup_steps::SERVICE_ID })
            .to_string();
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &["mcp-call", "service_logs", "--arguments-json", &arguments],
    );
    record(world, stdout, stderr, rc);
}

#[when("the user reads the ComfyUI logs")]
async fn read_comfyui_logs(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["comfyui", "logs"]);
    record(world, stdout, stderr, rc);
}

#[when("the user searches the logs for that topic")]
async fn search_topic(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["logs", "--search", TOPIC]);
    record(world, stdout, stderr, rc);
}

#[when("the user searches the logs for a term that appears nowhere")]
async fn search_absent(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["logs", "--search", "e2e-no-such-term"]);
    record(world, stdout, stderr, rc);
}

#[when("the user asks for one service's logs and a search term together")]
async fn service_and_search(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &["logs", "--service", "some-service", "--search", TOPIC],
    );
    record(world, stdout, stderr, rc);
}

// ── Then ───────────────────────────────────────────────────────────

#[then("the CLI reports the matching recent lines")]
async fn reports_matches(world: &mut E2eWorld) {
    let out = ok_output(world);
    // The count of matching lines is deterministic (we planted exactly nine within
    // the tail); the "of <total>" denominator is not, so assert only the match side.
    assert!(
        out.contains(&format!("Lines: {MATCH_COUNT} of ")),
        "expected {MATCH_COUNT} matching lines, got:\n{out}"
    );
    assert!(
        out.contains(&format!("Showing: 1-{MATCH_COUNT} of {MATCH_COUNT}")),
        "expected the {MATCH_COUNT} matches to be listed, got:\n{out}"
    );
}

#[then("the CLI says the newest line is too long to show instead of saying there is no output")]
async fn says_the_line_is_too_long(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains(
            "Recent output: none shown; the newest line is longer than the 4 MiB a log view reads"
        ),
        "expected the too-long notice, got:\n{out}"
    );
    assert!(
        !out.contains("no output yet"),
        "a log with content must not be reported as having no output:\n{out}"
    );
    // The claim and the state it describes: the log on disk holds exactly one
    // line, longer than the bound the notice names.
    let on_disk = std::fs::read(server_log_path(world)).expect("the server log is still there");
    assert!(
        on_disk.len() > LOG_VIEW_BOUND_BYTES && !on_disk.contains(&b'\n'),
        "the log on disk should be one line past the bound, is {} bytes",
        on_disk.len()
    );
}

#[then(
    "the CLI says the ComfyUI log's newest line is too long to show instead of saying it is empty"
)]
async fn comfyui_says_the_line_is_too_long(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("No output shown: the newest line is longer than the 4 MiB a log view reads."),
        "expected the too-long notice, got:\n{out}"
    );
    assert!(
        !out.contains("is empty"),
        "a log with content must not be reported as empty:\n{out}"
    );
    let on_disk = std::fs::read(comfyui_log_path(world)).expect("the ComfyUI log is still there");
    assert!(
        on_disk.len() > LOG_VIEW_BOUND_BYTES && !on_disk.contains(&b'\n'),
        "the log on disk should be one line past the bound, is {} bytes",
        on_disk.len()
    );
}

#[then("the CLI says the ComfyUI log is empty")]
async fn comfyui_says_empty(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(out.contains("The saved log is empty."), "{out}");
    assert!(!out.contains("longer than"), "{out}");
    let on_disk = std::fs::read(comfyui_log_path(world)).expect("the ComfyUI log is still there");
    assert!(on_disk.is_empty(), "the log on disk should be empty");
}

#[then("the CLI says there is no output yet")]
async fn says_no_output_yet(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(out.contains("Recent output: no output yet"), "{out}");
    assert!(!out.contains("longer than"), "{out}");
    let on_disk = std::fs::read(server_log_path(world)).expect("the server log is still there");
    assert!(on_disk.is_empty(), "the log on disk should be empty");
}

/// The reply `rocm mcp-call service_logs` printed, parsed. Asserts the call
/// succeeded and reported no error first, so a failure reads as the CLI's own.
fn service_logs_tool_reply(world: &E2eWorld) -> serde_json::Value {
    let out = ok_output(world);
    let reply: serde_json::Value = serde_json::from_str(out.trim())
        .unwrap_or_else(|error| panic!("rocm mcp-call did not print JSON ({error}):\n{out}"));
    assert_eq!(
        reply["isError"],
        serde_json::json!(false),
        "the tool reported an error:\n{out}"
    );
    reply
}

#[then(
    "the tool says the newest line is too long to show and flags it instead of returning an empty tail unexplained"
)]
async fn tool_says_the_line_is_too_long(world: &mut E2eWorld) {
    let reply = service_logs_tool_reply(world);
    let text = reply["content"][0]["text"].as_str().unwrap_or_default();
    assert!(
        text.contains("none is shown: the newest line is longer than the 4 MiB a log view reads"),
        "expected the too-long notice, got:\n{reply}"
    );
    assert!(
        !text.contains("no log output yet") && !text.contains("Read the last"),
        "a log with content must not be described as empty or as read:\n{reply}"
    );
    let content = &reply["structuredContent"];
    assert_eq!(content["tail"], "", "{reply}");
    assert_eq!(content["newest_line_exceeds_read_bound"], true, "{reply}");
    // The claim and the state it describes: the log on disk holds exactly one
    // line, longer than the bound the notice names.
    let on_disk = std::fs::read(server_log_path(world)).expect("the server log is still there");
    assert!(
        on_disk.len() > LOG_VIEW_BOUND_BYTES && !on_disk.contains(&b'\n'),
        "the log on disk should be one line past the bound, is {} bytes",
        on_disk.len()
    );
}

#[then("the tool says the server has no log output yet and does not flag an over-long line")]
async fn tool_says_no_output_yet(world: &mut E2eWorld) {
    let reply = service_logs_tool_reply(world);
    let text = reply["content"][0]["text"].as_str().unwrap_or_default();
    assert!(text.contains("has no log output yet"), "{reply}");
    assert!(!text.contains("longer than"), "{reply}");
    let content = &reply["structuredContent"];
    assert_eq!(content["tail"], "", "{reply}");
    assert_eq!(content["newest_line_exceeds_read_bound"], false, "{reply}");
    let on_disk = std::fs::read(server_log_path(world)).expect("the server log is still there");
    assert!(on_disk.is_empty(), "the log on disk should be empty");
}

#[then("the CLI reports no matching lines")]
async fn reports_no_matches(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("Lines: 0 of ") && out.contains("Showing: 0 of 0"),
        "expected no matching lines, got:\n{out}"
    );
}

#[then("the CLI refuses and explains only one may be used")]
async fn refuses_conflict(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    assert!(rc != 0, "expected refusal, got rc=0:\n{}", combined(world));
    assert!(
        combined(world)
            .contains("accepts either --service <service-id> or a search query, not both"),
        "expected the service/search conflict message, got:\n{}",
        combined(world)
    );
}

// ── Helpers ────────────────────────────────────────────────────────

fn record(world: &mut E2eWorld, stdout: String, stderr: String, rc: i32) {
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

fn combined(world: &E2eWorld) -> String {
    format!(
        "{}\n{}",
        world.cli_output.as_deref().unwrap_or(""),
        world.cli_stderr.as_deref().unwrap_or("")
    )
}

fn ok_output(world: &E2eWorld) -> String {
    let rc = world.cli_rc.expect("no command rc recorded");
    assert_eq!(rc, 0, "expected success, got rc={rc}:\n{}", combined(world));
    world.cli_output.clone().unwrap_or_default()
}

// ── --follow ───────────────────────────────────────────────────────

const FOLLOW_SERVICE_ID: &str = "vllm-e2e-follow";

#[given("a stopped local server whose log shows a download at 43 percent")]
async fn plant_stopped_server_with_log(world: &mut E2eWorld) {
    plant_server_with_log(world, "stopped");
}

#[given("a failed local server whose log shows a download at 43 percent")]
async fn plant_failed_server_with_log(world: &mut E2eWorld) {
    plant_server_with_log(world, "failed");
}

fn plant_server_with_log(world: &mut E2eWorld, status: &str) {
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let data = root.path().join("data");
    let services = data.join("services");
    let states = data.join("engines").join("vllm").join("state");
    std::fs::create_dir_all(&services).expect("failed to create services dir");
    std::fs::create_dir_all(&states).expect("failed to create engine state dir");
    let log_path = services.join(format!("{FOLLOW_SERVICE_ID}.log"));
    std::fs::write(&log_path, "Downloading model.safetensors: 43%\n").expect("failed to write log");
    let record = serde_json::json!({
        "service_id": FOLLOW_SERVICE_ID,
        "engine": "vllm",
        "model_ref": "qwen",
        "canonical_model_id": "Qwen/Qwen3.5",
        "host": "127.0.0.1",
        "port": 9,
        "endpoint_url": "http://127.0.0.1:9/v1",
        "mode": "managed",
        "status": status,
        "supervisor_pid": 0,
        "engine_pid": null,
        "manifest_path": services.join(format!("{FOLLOW_SERVICE_ID}.json")),
        "log_path": log_path,
        "engine_state_path": states.join(format!("{FOLLOW_SERVICE_ID}.json")),
        "created_at_unix_ms": 1_700_000_000_000_u64,
    });
    std::fs::write(
        services.join(format!("{FOLLOW_SERVICE_ID}.json")),
        serde_json::to_vec_pretty(&record).expect("failed to serialize record"),
    )
    .expect("failed to write service record");
}

#[when("the user follows that server's logs")]
async fn follow_stopped_server(world: &mut E2eWorld) {
    let (stdout, stderr, rc) =
        crate::run_rocm(world, &["logs", "--service", FOLLOW_SERVICE_ID, "--follow"]);
    record(world, stdout, stderr, rc);
}

#[when("the user follows that server's logs with rocm services logs -f")]
async fn follow_stopped_server_through_services(world: &mut E2eWorld) {
    let (stdout, stderr, rc) =
        crate::run_rocm(world, &["services", "logs", FOLLOW_SERVICE_ID, "-f"]);
    record(world, stdout, stderr, rc);
}

#[when("the user asks to follow logs without naming a service")]
async fn follow_without_service(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["logs", "--follow"]);
    record(world, stdout, stderr, rc);
}

#[then("the CLI prints the download line, says it is not following, and exits")]
async fn prints_and_exits(world: &mut E2eWorld) {
    let out = ok_output(world);
    let err = world.cli_stderr.clone().unwrap_or_default();
    assert!(
        out.contains("Downloading model.safetensors: 43%"),
        "expected the planted log line, got:\n{out}"
    );
    // Piped, stdout is log text only; the notice about the follow goes to stderr.
    let note = format!("Service {FOLLOW_SERVICE_ID} is stopped; not following.");
    assert!(!out.contains(&note), "notice leaked onto stdout:\n{out}");
    assert!(
        err.contains(&note),
        "expected the not-following line, got:\n{err}"
    );
}

#[then("the CLI prints the download line, says it is not following, and exits with an error")]
async fn prints_and_fails(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    let out = world.cli_output.clone().unwrap_or_default();
    let err = world.cli_stderr.clone().unwrap_or_default();
    assert_ne!(
        rc, 0,
        "a failed server must not follow as success:\n{out}\n{err}"
    );
    assert!(
        out.contains("Downloading model.safetensors: 43%"),
        "expected the planted log line, got:\n{out}"
    );
    assert!(
        err.contains(&format!(
            "Service {FOLLOW_SERVICE_ID} is failed; not following."
        )),
        "expected the not-following line, got:\n{err}"
    );
    assert!(
        err.contains(&format!("service {FOLLOW_SERVICE_ID} failed")),
        "expected the failure error, got:\n{err}"
    );
}

#[then("the CLI refuses because --follow needs --service")]
async fn refuses_follow(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    assert!(rc != 0, "expected refusal, got rc=0:\n{}", combined(world));
    assert!(
        combined(world).contains("--service"),
        "expected the error to name --service, got:\n{}",
        combined(world)
    );
}
