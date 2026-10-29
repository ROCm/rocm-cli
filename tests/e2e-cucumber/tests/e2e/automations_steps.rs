// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocm automations enable/disable` and `rocm automations` (bare,
//! which lists the most recent review requests of any status — only the
//! per-entry approve/reject controls are gated on "pending"). Black-box
//! against the isolated config dir. `automations enable` would otherwise spawn
//! a detached background daemon (`rocm daemon`) on first enable, which both
//! adds a nondeterministic `helper:` line and leaks a process past the
//! scenario. To keep the mock lane hermetic, every enable/disable scenario
//! first plants an automation runtime-state marking the daemon already
//! running under THIS test process's (live) pid, so the CLI's double-spawn
//! guard skips the spawn. Contracts verified against the running Linux binary
//! (EAI-8072). Scoped to the enable/disable/mode slice, the proposal-history
//! listing's unit formatting, and the bare listing's tolerance of a torn
//! audit-log tail; the watcher/daemon trigger behaviour itself is covered
//! separately.

use cucumber::{given, then, when};
use rocm_core::{AuditEventRecord, AutomationProposalRecord, append_automation_proposal};

use crate::E2eWorld;

/// A built-in watcher id (`BUILTIN_WATCHERS` in rocm-core); stable and always
/// present, so scenarios can enable/disable it without depending on host state.
const WATCHER: &str = "therock-update";

/// Plant an automation runtime-state that marks the background daemon already
/// running under the test harness's own (live) pid. `automations enable` guards
/// against a second spawn when the recorded daemon pid is a live process, so this
/// suppresses the detached-daemon spawn — no `helper:` line, no leaked process.
///
/// Shared with the `a fresh CLI configuration` Given (in `config_steps`), which
/// calls this so every automations scenario starting from a fresh config is also
/// spawn-suppressed. Exposed `pub(crate)` for that single caller.
pub(crate) fn suppress_daemon_spawn(world: &E2eWorld) {
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let dir = root.path().join("data").join("automations");
    std::fs::create_dir_all(&dir).expect("failed to create automations dir");
    let state = serde_json::json!({
        "running": true,
        "automations_enabled": true,
        "daemon_pid": std::process::id(),
        "started_at_unix_ms": 1_700_000_000_000u64,
        "last_tick_unix_ms": 1_700_000_000_000u64,
        "active_watchers": [],
    });
    std::fs::write(
        dir.join("runtime-state.json"),
        serde_json::to_vec_pretty(&state).expect("failed to serialize automation state"),
    )
    .expect("failed to write automation runtime state");
}

/// Plant an autostart *claim* — the harness's own (live) pid and a fresh spawn
/// time — WITHOUT planting the daemon runtime-state. Unlike `suppress_daemon_spawn`
/// (which marks the daemon already *running*, so the CLI's pre-check short-circuits
/// before the claim is ever consulted), this reproduces the spawn→publish window
/// the fix guards: no `running` state is published yet, so the "already running?"
/// check reads false, but a spawn is genuinely in flight. A correct autostart
/// holder that acquires the lock in this window must see the live, recent claim
/// and defer instead of spawning a duplicate daemon. Written in the same
/// `"<pid> <ms>"` shape the CLI's `write_autostart_claim` uses.
fn plant_inflight_autostart_claim(world: &E2eWorld) {
    let path = autostart_claim_path(world);
    std::fs::create_dir_all(path.parent().expect("claim path has a parent"))
        .expect("failed to create automations dir");
    write_harness_autostart_claim(&path);
}

/// Where the CLI reads its autostart claim inside the scenario's isolated root.
fn autostart_claim_path(world: &E2eWorld) -> std::path::PathBuf {
    let root = world.isolated_root.as_ref().expect("no isolated root");
    root.path()
        .join("data")
        .join("automations")
        .join("autostart.claim")
}

/// Write a claim for this harness process, stamped now.
fn write_harness_autostart_claim(path: &std::path::Path) {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_millis());
    std::fs::write(path, format!("{} {now_ms}", std::process::id()))
        .expect("failed to write autostart claim");
}

/// Re-stamp a claim `plant_inflight_autostart_claim` planted, right before the
/// CLI runs. The CLI honours a claim for only 30 seconds
/// (`AUTOSTART_CLAIM_TTL_MS` in `apps/rocm/src/service_launch.rs`). The mock
/// lane runs up to 64 scenarios at once, and every step blocks a runtime
/// worker while its `rocm` child runs, so this scenario's When can start more
/// than 30 seconds after its Given. The CLI then rightly reads the claim as
/// stale and spawns, and the scenario fails without any defect in the CLI.
/// Only a claim holding this process's pid is touched: that is the one the
/// Given wrote, never one the CLI wrote for a daemon it spawned.
fn refresh_harness_autostart_claim(world: &E2eWorld) {
    let path = autostart_claim_path(world);
    let Ok(text) = std::fs::read_to_string(&path) else {
        return;
    };
    let ours = text
        .split_whitespace()
        .next()
        .is_some_and(|pid| pid == std::process::id().to_string());
    if ours {
        write_harness_autostart_claim(&path);
    }
}

#[given("a daemon spawn is already in flight")]
async fn daemon_spawn_in_flight(world: &mut E2eWorld) {
    plant_inflight_autostart_claim(world);
}

#[given("a pending artifact-download proposal approved for a size just under 1 MiB")]
async fn pending_artifact_download_proposal(world: &mut E2eWorld) {
    append_automation_proposal(
        &world.app_paths(),
        &AutomationProposalRecord {
            at_unix_ms: 1_700_000_000_000,
            proposal_id: String::new(),
            watcher_id: "cache-warm".to_owned(),
            action: "queue_prefetch_proposal".to_owned(),
            title: "queue prefetch proposal".to_owned(),
            message: "cache warm requested an artifact prefetch".to_owned(),
            status: "pending".to_owned(),
            service_id: None,
            tool: Some("prefetch_artifact".to_owned()),
            arguments: serde_json::json!({
                "artifact_ref": "tiny/model#gguf",
                "allow_artifact_download": true,
                // One byte under 1 MiB: `{:.1}` rounds it up to a full 1024
                // KiB, so the approved cap must read as "1.0 MiB", not
                // "1024.0 KiB" — and in binary units, not "1.0 MB" (#190).
                "artifact_max_bytes": 1_048_575,
            }),
            reviewed_at_unix_ms: None,
        },
    )
    .expect("failed to write automation proposal fixture");
}

/// Seed the audit-event store with one clean record followed by a cut-off
/// JSON fragment with no trailing newline — the shape a writer killed
/// mid-append leaves. See the scenario's own comment in `automations.feature`
/// for why this matters and who reads this store; see
/// `scan_recent_jsonl_records`/`is_torn_record` in `rocm-core` for the read
/// path this exercises (not `inspect_tail`, which is the write path's own,
/// separate torn-tail check).
#[given("the audit event log ends in a torn, unterminated line")]
async fn torn_audit_tail(world: &mut E2eWorld) {
    let paths = world.app_paths();
    paths.ensure().expect("failed to create app dirs");
    let clean = AuditEventRecord {
        at_unix_ms: 1_700_000_000_000,
        source: "e2e".to_owned(),
        category: "concurrency".to_owned(),
        actor: "writer".to_owned(),
        level: "info".to_owned(),
        action: "append".to_owned(),
        message: "clean".to_owned(),
        watcher_id: None,
        service_id: None,
    };
    let clean_line = serde_json::to_string(&clean).expect("serialize audit record");
    // Just the first 20 bytes of the same record: short enough that the cut
    // falls well before the `message` field, so what it says doesn't matter.
    let torn_fragment = &clean_line[..20];
    std::fs::write(
        paths.audit_events_path(),
        format!("{clean_line}\n{torn_fragment}"),
    )
    .expect("failed to seed a torn audit-event tail");
}

#[given("an enabled automation watcher")]
async fn enabled_watcher(world: &mut E2eWorld) {
    suppress_daemon_spawn(world);
    crate::run_rocm_ok(
        world,
        &["automations", "enable", WATCHER, "--mode", "observe"],
    );
}

// ── When ───────────────────────────────────────────────────────────

#[when("the user enables an automation watcher in observe mode")]
async fn enable_observe(world: &mut E2eWorld) {
    refresh_harness_autostart_claim(world);
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &["automations", "enable", WATCHER, "--mode", "observe"],
    );
    record(world, stdout, stderr, rc);
}

#[when("the user re-enables the same watcher in propose mode")]
async fn enable_propose(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &["automations", "enable", WATCHER, "--mode", "propose"],
    );
    record(world, stdout, stderr, rc);
}

#[when("the user disables the watcher")]
async fn disable_watcher(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["automations", "disable", WATCHER]);
    record(world, stdout, stderr, rc);
}

#[when("the user lists automations")]
async fn list_automations(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["automations"]);
    record(world, stdout, stderr, rc);
}

#[when("the user tries to enable a watcher that does not exist")]
async fn enable_unknown(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &[
            "automations",
            "enable",
            "e2e-no-such-watcher",
            "--mode",
            "observe",
        ],
    );
    record(world, stdout, stderr, rc);
}

// ── Then ───────────────────────────────────────────────────────────

#[then("the CLI still lists the record before the torn tail")]
async fn still_lists_record_before_torn_tail(world: &mut E2eWorld) {
    // The count, not rc == 0, is the real assertion here (see
    // `automations.feature` for why). `audit_event_plain_summary` renders
    // every audit event's category to one of a handful of fixed strings
    // (never the raw message), so the fixture's "concurrency" category falls
    // through to the generic one; exactly one occurrence confirms the one
    // clean record before the torn tail was actually read and rendered, not
    // silently dropped along with it (zero) or duplicated (more than one).
    let out = ok_output(world);
    let activity_lines = out.matches("Background activity was recorded.").count();
    assert_eq!(
        activity_lines, 1,
        "expected exactly the one valid record before the torn tail to be listed, got:\n{out}"
    );
}

#[then(regex = r"^the CLI confirms the watcher is enabled in (observe|propose) mode$")]
async fn confirm_enabled(world: &mut E2eWorld, mode: String) {
    let out = ok_output(world);
    assert!(
        out.contains("automation watcher enabled"),
        "expected an enable confirmation, got:\n{out}"
    );
    assert!(
        out.contains(&format!("watcher: {WATCHER}")),
        "expected the watcher id, got:\n{out}"
    );
    assert!(
        out.contains(&format!("mode: {mode}")),
        "expected mode {mode}, got:\n{out}"
    );
}

#[then("the CLI does not start a second background daemon")]
async fn no_second_daemon(world: &mut E2eWorld) {
    // The enable itself still succeeds and confirms the watcher: deferring to the
    // in-flight spawn is invisible to the user's requested action.
    let out = ok_output(world);
    assert!(
        out.contains("automation watcher enabled"),
        "expected the enable to still succeed, got:\n{out}"
    );
    // But it must have DEFERRED to the in-flight claim: the autostart holder only
    // prints a `helper:` daemon line ("started ..." on success, "could not start
    // ..." on failure) once it actually reaches the spawn. Its absence is the
    // observable proof that no second daemon was launched — the exact regression
    // a duplicate spawn in the spawn→publish window would introduce.
    let combined = combined(world);
    assert!(
        !combined.contains("background automation daemon"),
        "expected the enable to defer to the in-flight spawn without launching a \
         second daemon, but it reached the spawn:\n{combined}"
    );
}

#[then("the CLI confirms the watcher is disabled")]
async fn confirm_disabled(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("automation watcher disabled") && out.contains(&format!("watcher: {WATCHER}")),
        "expected a disable confirmation for {WATCHER}, got:\n{out}"
    );
}

#[then("the CLI reports the approved size in binary units")]
async fn reports_binary_units(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("download: approved up to 1.0 MiB"),
        "expected the approved cap in binary (KiB/MiB/GiB) units, got:\n{out}"
    );
    assert!(
        !out.contains("1.0 MB"),
        "expected a binary MiB label, not a decimal-looking MB one:\n{out}"
    );
}

#[then("the CLI refuses and names it as unknown")]
async fn refuse_unknown(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    assert!(rc != 0, "expected refusal, got rc=0:\n{}", combined(world));
    assert!(
        combined(world).contains("unknown watcher: e2e-no-such-watcher"),
        "expected an unknown-watcher error, got:\n{}",
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
