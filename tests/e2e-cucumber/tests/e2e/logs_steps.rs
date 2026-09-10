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
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let cli_logs = root.path().join("data").join("logs").join("cli");
    std::fs::create_dir_all(&cli_logs).expect("failed to create cli logs dir");
    // Write the matching lines LAST so they fall within the per-file tail window
    // `rocm logs` reads (a few leading non-matching lines are harmless context).
    let mut body = String::new();
    for i in 0..3 {
        let _ = writeln!(body, "2026-01-01T00:00:0{i} unrelated startup line");
    }
    for i in 1..=MATCH_COUNT {
        let _ = writeln!(
            body,
            "2026-01-01T00:01:00 event {TOPIC} occurred number {i}"
        );
    }
    std::fs::write(cli_logs.join("e2e-probe.log"), body).expect("failed to write log file");
}

// ── When ───────────────────────────────────────────────────────────

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
