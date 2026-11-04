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
//! listing's unit formatting, the bare listing's tolerance of a torn
//! audit-log tail, the repair of one by the next audited command, and the
//! size cap on the audit log and compaction of the
//! proposal history (ROCMAI-729). One scenario (automations-12) starts a real
//! `rocmd run` and feeds it GPU thermal alerts through its loopback webhook; the
//! rest of the watcher/daemon trigger behaviour is covered separately.
//!
//! Two slices live here. The enable/disable/mode steps act on a watcher id the
//! test already knows. The listing steps (scenario 13) assert the complementary
//! discoverability contract: everything the listing shows a user must also be
//! able to act on, so they derive each check's identifier FROM the listing rather
//! than knowing it in advance — hard-coding the real ids would keep passing
//! against a listing that publishes none of them, which is the defect pinned.

use std::io::{Read, Write};
use std::process::Stdio;
use std::time::Duration;

use cucumber::{given, then, when};
use e2e_cucumber::polled_deadline::{PolledDeadline, RetryTiming};
use rocm_core::{
    AuditEventRecord, AutomationProposalRecord, EVENT_LOG_ROTATE_BYTES,
    EVENT_LOG_ROTATED_GENERATIONS, REVIEWED_PROPOSALS_RETAINED, append_automation_proposal,
    jsonl_lock_path, load_automation_proposals, load_recent_automation_events,
};

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

/// Slugify a display name: lowercase, words joined by dashes.
///
/// This is the guess a user is forced into when the listing exposes no real
/// identifier — the current-bug fallback, not the contract. It deliberately does
/// NOT recover the true ids ("Server recovery" → `server-recovery`, not the real
/// `server-recover`), which is exactly why the listing has to publish them.
fn slug_of(display_name: &str) -> String {
    display_name
        .split_whitespace()
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join("-")
}

/// The identifiers a check block plausibly EXPOSES, most specific first.
///
/// The contract is "enable-able from what the listing shows", so the moment the
/// product publishes an identifier this must pick it up, or a correct fix would
/// stay xfailed forever (the row would never go stale). These are CANDIDATES, not
/// a verdict: the step tries each and counts the check reachable if any works, so
/// a wrong guess here costs nothing and a missing one is the only risk.
///   - a detail line keyed `id:` / `identifier:` / `slug:`, or any key ending in
///     ` id` (`command id: server-recover`)
///   - bracketed on the header — `Server recovery [server-recover]`
///   - a single parenthesised token beside the label — `Server recovery (server-recover)`
///
/// Empty only when the block names no identifier at all, which is today's defect.
fn exposed_identifiers(block: &[&str]) -> Vec<String> {
    let mut found = Vec::new();
    let mut add = |id: &str| {
        let id = id.trim().trim_matches(|c| c == '"' || c == '`');
        if !id.is_empty() && !id.contains(char::is_whitespace) && !found.iter().any(|f| f == id) {
            found.push(id.to_owned());
        }
    };
    for line in block {
        if let Some((key, rest)) = line.trim().split_once(':') {
            let key = key.trim().to_ascii_lowercase();
            if matches!(key.as_str(), "id" | "identifier" | "slug") || key.ends_with(" id") {
                add(rest);
            }
        }
    }
    if let Some(header) = block.first().map(|h| h.trim()) {
        for (open, close) in [('[', ']'), ('(', ')')] {
            if let (Some(o), Some(c)) = (header.rfind(open), header.rfind(close))
                && o < c
            {
                add(&header[o + 1..c]);
            }
        }
    }
    found
}

/// The checks the listing advertises, as `(displayed name, identifier to try)`.
///
/// A check is a two-space-indented header line carrying a parenthesised on/off
/// state — `  Server recovery (off)`. Its own detail lines (`setting:`, `does:`,
/// and potentially an `id:`) are indented further; the report's other sections
/// list events rather than checks. Each header plus the deeper-indented lines
/// under it forms one block, so an identifier the fix exposes on a detail line
/// is seen. The identifier to invoke is the one the block explicitly exposes,
/// falling back to the display-name slug only when it exposes none (today's bug).
fn listed_checks(listing: &str) -> Vec<(String, Vec<String>)> {
    let lines: Vec<&str> = listing.lines().collect();
    let is_header = |line: &str| line.starts_with("  ") && !line.starts_with("   ");
    let mut checks = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if !is_header(line) {
            continue;
        }
        let Some(name) = line
            .trim()
            .strip_suffix("(on)")
            .or_else(|| line.trim().strip_suffix("(off)"))
            .map(str::trim)
            .filter(|name| !name.is_empty())
        else {
            continue;
        };
        // The block is this header plus every following more-indented line, up
        // to the next header or a less-indented line.
        let mut block = vec![name];
        for detail in &lines[i + 1..] {
            if detail.starts_with("    ") {
                block.push(detail);
            } else {
                break;
            }
        }
        // Exposed identifiers first, then the label exactly as listed (a fix may make
        // `enable` accept it), then its slug — today's fallback.
        let mut candidates = exposed_identifiers(&block);
        for fallback in [name.to_owned(), slug_of(name)] {
            if !candidates.contains(&fallback) {
                candidates.push(fallback);
            }
        }
        checks.push((name.to_owned(), candidates));
    }
    checks
}

// ── Given ──────────────────────────────────────────────────────────

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

#[given("no automation watcher is enabled")]
async fn no_watcher_enabled(_world: &mut E2eWorld) {
    // The World already gives each scenario an isolated, empty config dir, where
    // every watcher is off. Deliberately plants NO spawn-suppression marker: the
    // scenario drives the real daemon spawn.
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

/// Make the audit log's directory refuse new files while the log inside it
/// stays writable, so the repair of a torn tail cannot write the clean copy it
/// renames into place — the failure a full disk causes, without filling one.
///
/// The CLI's lock file for the log is created first: it is reused when it
/// exists but cannot be created in a directory that takes no new files. Fails
/// if the directory still takes new files afterwards: directory permissions do
/// not bind root, and the scenario's `@requires-dir-permissions` tag skips it
/// there, so this is the backstop that keeps it from passing without
/// exercising the fallback should the tag be dropped.
/// [`restore_audit_dir_permissions`] undoes it so the scenario's temp root can
/// be removed.
#[given("the audit log's directory takes no new files")]
async fn audit_dir_takes_no_new_files(world: &mut E2eWorld) {
    let paths = world.app_paths();
    let dir = paths.audit_dir();
    std::fs::File::create(jsonl_lock_path(&paths.audit_events_path()))
        .expect("failed to create the lock file");
    forbid_new_files(&dir);
    let probe = dir.join("probe");
    if std::fs::File::create(&probe).is_ok() {
        let _ = std::fs::remove_file(&probe);
        panic!(
            "{} still takes new files, so this scenario cannot make the audit log's \
             replacement fail: directory permissions do not bind this user (root?)",
            dir.display()
        );
    }
}

#[cfg(unix)]
fn forbid_new_files(dir: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o555))
        .expect("failed to make the audit directory read-only");
}

#[cfg(not(unix))]
// Platform stub for a `@requires-os:linux` scenario. The Unix arm does I/O and
// cannot be `const`, so making only this arm `const` would give the two
// platforms different signatures.
#[allow(clippy::missing_const_for_fn)]
fn forbid_new_files(_dir: &std::path::Path) {
    panic!("this step needs Unix directory permissions; its scenario is @requires-os:linux");
}

/// Undo [`audit_dir_takes_no_new_files`] on `dir`, best effort, from the
/// world's teardown.
#[cfg(unix)]
pub(crate) fn restore_audit_dir_permissions(dir: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    if dir.is_dir() {
        let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o755));
    }
}

#[then("the CLI prints no warning about the audit log")]
async fn no_audit_warning(world: &mut E2eWorld) {
    let stderr = world.cli_stderr.clone().unwrap_or_default();
    assert!(
        !stderr.to_lowercase().contains("audit"),
        "expected no audit-log warning, got stderr:\n{stderr}"
    );
}

#[then(
    "the audit log keeps its earlier record, drops the torn line and records the permissions change"
)]
async fn audit_log_repaired_and_appended(world: &mut E2eWorld) {
    let path = world.app_paths().audit_events_path();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    // `audit_records` parses every line, so a left-over torn fragment fails here.
    let records = audit_records(&path);
    let seen: Vec<(&str, &str)> = records
        .iter()
        .map(|record| (record.message.as_str(), record.action.as_str()))
        .collect();
    assert_eq!(
        seen,
        [
            ("clean", "append"),
            ("permissions mode set to ask", "set_mode")
        ],
        "unexpected audit log:\n{text}"
    );
    assert!(text.ends_with('\n'), "the last record must be terminated");
}

#[given("an enabled automation watcher")]
async fn enabled_watcher(world: &mut E2eWorld) {
    suppress_daemon_spawn(world);
    crate::run_rocm_ok(
        world,
        &["automations", "enable", WATCHER, "--mode", "observe"],
    );
}

/// Action of the audit records seeded by `audit_log_at_cap`.
const SEEDED_AUDIT_ACTION: &str = "e2e_seeded";

#[given("an audit log that has reached its size cap")]
async fn audit_log_at_cap(world: &mut E2eWorld) {
    let paths = world.app_paths();
    std::fs::create_dir_all(paths.audit_dir()).expect("failed to create audit dir");
    let record = AuditEventRecord {
        at_unix_ms: 1_700_000_000_000,
        source: "rocm".to_owned(),
        // `rocm automations` summarizes this category as "Install activity was
        // recorded.", unlike the permissions change made afterwards.
        category: "install".to_owned(),
        actor: "cli".to_owned(),
        level: "info".to_owned(),
        action: SEEDED_AUDIT_ACTION.to_owned(),
        message: "x".repeat(1024),
        watcher_id: None,
        service_id: None,
    };
    let line = serde_json::to_string(&record).expect("serialize audit record") + "\n";
    let lines = EVENT_LOG_ROTATE_BYTES.div_ceil(line.len() as u64);
    std::fs::write(
        paths.audit_events_path(),
        line.repeat(usize::try_from(lines).expect("line count fits usize")),
    )
    .expect("failed to seed the audit log");
}

/// Action tag for the already-rotated `generation`-th file seeded by
/// `audit_log_with_full_rotation_chain`, so after one more rotation each
/// generation's new content can be traced back to which generation it came
/// from.
fn seeded_generation_action(generation: usize) -> String {
    format!("e2e_seeded_gen{generation}")
}

/// Action tag for the live file seeded by `audit_log_with_full_rotation_chain`,
/// distinct from every `seeded_generation_action` tag.
const SEEDED_LIVE_BEFORE_CHAIN_ROTATION: &str = "e2e_seeded_live_before_chain_rotation";

#[given("an audit log already holding a full chain of four rotated generations")]
async fn audit_log_with_full_rotation_chain(world: &mut E2eWorld) {
    let paths = world.app_paths();
    std::fs::create_dir_all(paths.audit_dir()).expect("failed to create audit dir");
    let live = paths.audit_events_path();

    // An already-rotated generation is never resized by a further rotation —
    // only shifted or renamed over — so these only need to exist and be
    // distinguishable, not reach the cap themselves.
    for generation in 1..=EVENT_LOG_ROTATED_GENERATIONS {
        let record = AuditEventRecord {
            at_unix_ms: 1_700_000_000_000,
            source: "rocm".to_owned(),
            category: "install".to_owned(),
            actor: "cli".to_owned(),
            level: "info".to_owned(),
            action: seeded_generation_action(generation),
            message: format!("generation {generation}"),
            watcher_id: None,
            service_id: None,
        };
        let line = serde_json::to_string(&record).expect("serialize audit record") + "\n";
        std::fs::write(
            live.with_file_name(format!("events.jsonl.{generation}")),
            line,
        )
        .unwrap_or_else(|error| panic!("failed to seed generation {generation}: {error}"));
    }

    // The live file itself must be at the cap, or the change made afterwards
    // would not rotate anything at all.
    let live_record = AuditEventRecord {
        at_unix_ms: 1_700_000_000_000,
        source: "rocm".to_owned(),
        category: "install".to_owned(),
        actor: "cli".to_owned(),
        level: "info".to_owned(),
        action: SEEDED_LIVE_BEFORE_CHAIN_ROTATION.to_owned(),
        message: "x".repeat(1024),
        watcher_id: None,
        service_id: None,
    };
    let live_line = serde_json::to_string(&live_record).expect("serialize audit record") + "\n";
    let lines = EVENT_LOG_ROTATE_BYTES.div_ceil(live_line.len() as u64);
    std::fs::write(
        &live,
        live_line.repeat(usize::try_from(lines).expect("line count fits usize")),
    )
    .expect("failed to seed the live audit log");
}

const OLD_PENDING_PROPOSAL: &str = "e2e-pending-old";
const NEW_PENDING_PROPOSAL: &str = "e2e-pending-new";
/// Reviewed proposals seeded beyond the retention cap, which compaction drops.
const EXCESS_REVIEWED: usize = 5;

fn e2e_proposal(proposal_id: &str, reviewed_at: Option<u128>) -> AutomationProposalRecord {
    AutomationProposalRecord {
        at_unix_ms: 1_700_000_000_000,
        proposal_id: proposal_id.to_owned(),
        watcher_id: "therock-update".to_owned(),
        action: "queue_update_proposal".to_owned(),
        title: "check for ROCm updates".to_owned(),
        message: "a newer ROCm runtime may be available".to_owned(),
        status: if reviewed_at.is_some() {
            "approved"
        } else {
            "pending"
        }
        .to_owned(),
        service_id: None,
        tool: Some("check_updates".to_owned()),
        arguments: serde_json::json!({}),
        reviewed_at_unix_ms: reviewed_at,
    }
}

#[given("a pending proposal queued before a long history of reviewed proposals")]
async fn pending_before_review_history(world: &mut E2eWorld) {
    let paths = world.app_paths();
    let mut proposals = vec![e2e_proposal(OLD_PENDING_PROPOSAL, None)];
    proposals.extend((0..REVIEWED_PROPOSALS_RETAINED + EXCESS_REVIEWED).map(|n| {
        e2e_proposal(
            &format!("e2e-reviewed-{n}"),
            Some(1_700_000_000_000 + n as u128),
        )
    }));
    proposals.push(e2e_proposal(NEW_PENDING_PROPOSAL, None));
    for proposal in &proposals {
        append_automation_proposal(&paths, proposal).expect("failed to seed a proposal");
    }
}

#[given("a machine with no background checks turned on")]
async fn no_background_checks(world: &mut E2eWorld) {
    // The scenario's isolated config starts empty, so every check is already off
    // and nothing needs planting for the precondition itself. What DOES need
    // planting is the running-daemon marker, for the same reason the sibling
    // scenarios plant it: `automations enable` would otherwise launch a detached
    // `rocm daemon` that outlives the scenario and accumulates on a persistent
    // runner.
    suppress_daemon_spawn(world);
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

/// Run the hidden `rocm mcp-call proposal_action` entry point the dashboard
/// and chat review controls use.
fn proposal_action(world: &mut E2eWorld, proposal_id: &str, action: &str) {
    let arguments = serde_json::json!({ "proposal_id": proposal_id, "action": action });
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &[
            "mcp-call",
            "proposal_action",
            "--arguments-json",
            &arguments.to_string(),
            "--allow-mutation",
        ],
    );
    record(world, stdout, stderr, rc);
}

#[when("the user approves the newest pending proposal")]
async fn approve_newest_pending(world: &mut E2eWorld) {
    proposal_action(world, NEW_PENDING_PROPOSAL, "approve");
}

#[when("the user looks up the older pending proposal")]
async fn show_old_pending(world: &mut E2eWorld) {
    proposal_action(world, OLD_PENDING_PROPOSAL, "show");
}

#[when("the user lists the background checks")]
async fn user_lists_checks(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["automations", "list"]);
    assert_eq!(
        rc, 0,
        "listing the background checks failed (rc={rc}):\n{stdout}\n{stderr}"
    );
    world.cli_output = Some(stdout);
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

/// How long the helper gets to publish its runtime state after the spawn, or
/// to exit after its last watcher is disabled (it re-reads the config on a 5 s
/// tick). Generous because the mock lane runs many scenarios at once.
const HELPER_LIFECYCLE_TIMEOUT: std::time::Duration = std::time::Duration::from_mins(1);

/// The helper's recorded pid and `running` flag, read from the runtime state
/// the daemon itself publishes — the same file `rocm daemon --status` reads.
fn helper_runtime_state(world: &E2eWorld) -> Option<(u32, bool)> {
    rocm_core::AutomationRuntimeState::load(&world.app_paths())
        .ok()
        .flatten()
        .map(|state| (state.daemon_pid, state.running))
}

/// Teardown for a scenario that started a real helper: stop it if it is still
/// alive, whatever step failed. The helper is found from the runtime state it
/// published (when it says `running`) or, if it never got that far, from the
/// autostart claim the spawn writes synchronously (honoured for the CLI's own
/// 30 s). It is signalled only through `terminate_verified`, against the identity
/// captured when the scenario saw it live, or captured now where the state or
/// claim names a live pid — never by a bare pid. Best-effort: teardown must not
/// panic.
pub(crate) fn stop_leftover_helper(
    paths: &rocm_core::AppPaths,
    seen: Option<rocm_core::ProcessIdentity>,
) {
    let from_state = rocm_core::AutomationRuntimeState::load(paths)
        .ok()
        .flatten()
        .filter(|state| state.running)
        .map(|state| state.daemon_pid);
    let from_claim = || {
        let text = std::fs::read_to_string(paths.automation_autostart_claim_path()).ok()?;
        let mut parts = text.split_whitespace();
        let pid: u32 = parts.next()?.parse().ok()?;
        let at_ms: u128 = parts.next()?.parse().ok()?;
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_millis();
        (now_ms.saturating_sub(at_ms) < 30_000).then_some(pid)
    };
    let identity = seen.or_else(|| {
        from_state
            .or_else(from_claim)
            // Scenarios that suppress the spawn plant a state and claim naming
            // this very harness; it is never the helper.
            .filter(|pid| *pid != std::process::id())
            .filter(|pid| rocm_core::process_is_running(*pid))
            .map(rocm_core::ProcessIdentity::capture)
    });
    if let Some(identity) = identity {
        let _ = rocm_core::terminate_verified(
            &identity,
            rocm_core::KillScope::Single,
            std::time::Duration::from_secs(5),
            true,
        );
    }
}

#[then("the CLI reports starting the background helper and it is running")]
async fn helper_started_and_running(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("helper: started background automation daemon"),
        "expected the enable to report starting the helper, got:
{}",
        combined(world)
    );
    // The claim is checked against the process: the helper publishes a running
    // state under its own pid, and that pid is a live process.
    let deadline = std::time::Instant::now() + HELPER_LIFECYCLE_TIMEOUT;
    loop {
        if let Some((pid, true)) = helper_runtime_state(world)
            && rocm_core::process_is_running(pid)
        {
            world.helper_identity = Some(rocm_core::ProcessIdentity::capture(pid));
            let status = crate::run_rocm_ok(world, &["daemon", "--status"]);
            assert!(
                status.contains("Status: running"),
                "expected `rocm daemon --status` to report the helper running, got:
{status}"
            );
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the helper never published a running state under a live pid; state: {:?}",
            helper_runtime_state(world)
        );
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
}

#[then("the background helper stops on its own")]
async fn helper_stops_on_its_own(world: &mut E2eWorld) {
    let (pid, _) = helper_runtime_state(world).expect("the helper recorded no runtime state");
    let deadline = std::time::Instant::now() + HELPER_LIFECYCLE_TIMEOUT;
    loop {
        let running = helper_runtime_state(world).is_some_and(|(_, running)| running);
        if !running && !rocm_core::process_is_running(pid) {
            let status = crate::run_rocm_ok(world, &["daemon", "--status"]);
            assert!(
                status.contains("Status: not running"),
                "expected `rocm daemon --status` to report the helper stopped, got:
{status}"
            );
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the helper (pid {pid}) was still running {}s after its last watcher was disabled",
            HELPER_LIFECYCLE_TIMEOUT.as_secs()
        );
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
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

/// The audit records in `path`, which must all parse.
fn audit_records(path: &std::path::Path) -> Vec<AuditEventRecord> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
        .lines()
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str(line).unwrap_or_else(|error| {
                panic!(
                    "{}:{}: audit line failed to parse: {error}\n{line}",
                    path.display(),
                    index + 1
                )
            })
        })
        .collect()
}

#[then("the full audit log is kept aside as its first rotated generation")]
async fn audit_log_rotated(world: &mut E2eWorld) {
    ok_output(world);
    let live = world.app_paths().audit_events_path();
    let rotated = live.with_file_name("events.jsonl.1");
    let size = std::fs::metadata(&rotated)
        .unwrap_or_else(|error| panic!("expected {} to exist: {error}", rotated.display()))
        .len();
    assert!(
        size >= EVENT_LOG_ROTATE_BYTES,
        "expected the whole capped log in {}, found {size} bytes",
        rotated.display()
    );
    assert!(
        audit_records(&rotated)
            .iter()
            .all(|record| record.action == SEEDED_AUDIT_ACTION),
        "the rotated generation must hold exactly the seeded records"
    );
    assert!(
        !live.with_file_name("events.jsonl.2").exists(),
        "a single rollover makes a single rotated generation"
    );
}

#[then("the permissions change starts a fresh audit log")]
async fn fresh_audit_log(world: &mut E2eWorld) {
    let records = audit_records(&world.app_paths().audit_events_path());
    let actions: Vec<&str> = records
        .iter()
        .map(|record| record.action.as_str())
        .collect();
    assert_eq!(
        actions,
        ["set_mode"],
        "expected the live audit log to hold only the permissions change"
    );
}

#[then("the recent background activity still includes entries from before the rotation")]
async fn recent_activity_spans_rotation(world: &mut E2eWorld) {
    let out = ok_output(world);
    let Some((_, activity)) = out.split_once("recent background activity:") else {
        panic!("expected a recent background activity section:\n{out}");
    };
    // Five most recent: four seeded records, which now live only in the
    // rotated generation, then the permissions change.
    assert_eq!(
        activity.matches("Install activity was recorded.").count(),
        4,
        "expected the four newest seeded records from the rotated log:\n{out}"
    );
    assert_eq!(
        activity
            .matches("Background activity was recorded.")
            .count(),
        1,
        "expected the permissions change from the fresh log:\n{out}"
    );
}

#[then("each rotated generation holds what the previous one held")]
async fn chain_shifted_by_one(world: &mut E2eWorld) {
    ok_output(world);
    let live = world.app_paths().audit_events_path();
    assert!(
        audit_records(&live.with_file_name("events.jsonl.1"))
            .iter()
            .all(|record| record.action == SEEDED_LIVE_BEFORE_CHAIN_ROTATION),
        "expected .1 to hold only what was the live file before this rotation"
    );
    for generation in 2..=EVENT_LOG_ROTATED_GENERATIONS {
        let actions: Vec<String> =
            audit_records(&live.with_file_name(format!("events.jsonl.{generation}")))
                .into_iter()
                .map(|record| record.action)
                .collect();
        assert_eq!(
            actions,
            [seeded_generation_action(generation - 1)],
            "expected .{generation} to hold what .{} held before this rotation",
            generation - 1
        );
    }
}

#[then("the oldest previous generation is gone")]
async fn oldest_generation_dropped(world: &mut E2eWorld) {
    let live = world.app_paths().audit_events_path();
    assert!(
        !live
            .with_file_name(format!(
                "events.jsonl.{}",
                EVENT_LOG_ROTATED_GENERATIONS + 1
            ))
            .exists(),
        "rotation must never keep more than {EVENT_LOG_ROTATED_GENERATIONS} generations"
    );
    let dropped_action = seeded_generation_action(EVENT_LOG_ROTATED_GENERATIONS);
    for generation in 1..=EVENT_LOG_ROTATED_GENERATIONS {
        assert!(
            !audit_records(&live.with_file_name(format!("events.jsonl.{generation}")))
                .iter()
                .any(|record| record.action == dropped_action),
            "the oldest generation's content ({dropped_action}) must not survive in .{generation}"
        );
    }
}

#[then("the CLI confirms the approval")]
async fn confirms_approval(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains(&format!("Proposal {NEW_PENDING_PROPOSAL} approved.")),
        "expected an approval confirmation, got:\n{out}"
    );
}

#[then("only the most recently reviewed proposals are kept")]
async fn reviewed_history_compacted(world: &mut E2eWorld) {
    let proposals =
        load_automation_proposals(&world.app_paths()).expect("proposal history reads back");
    let reviewed: Vec<&str> = proposals
        .iter()
        .filter(|proposal| proposal.status != "pending")
        .map(|proposal| proposal.proposal_id.as_str())
        .collect();
    assert_eq!(
        reviewed.len(),
        REVIEWED_PROPOSALS_RETAINED,
        "expected the review history capped at {REVIEWED_PROPOSALS_RETAINED}"
    );
    // The approval just made is the newest review, so the oldest reviews give
    // way to it: the excess plus one.
    for n in 0..=EXCESS_REVIEWED {
        let dropped = format!("e2e-reviewed-{n}");
        assert!(
            !reviewed.contains(&dropped.as_str()),
            "expected {dropped} dropped"
        );
    }
    assert_eq!(reviewed.last(), Some(&NEW_PENDING_PROPOSAL));
}

#[then("the CLI shows it still pending")]
async fn shows_still_pending(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains(&format!("Proposal {OLD_PENDING_PROPOSAL} (pending)")),
        "expected the older pending proposal to still be found, got:\n{out}"
    );
}

// ── Thermal alerts through a running daemon ────────────────────────

/// The managed server the thermal alerts name.
const HOT_SERVICE: &str = "e2e-hot";
/// How often a wait re-reads the stores the daemon writes.
const DAEMON_POLL: Duration = Duration::from_millis(100);
/// Generous: the daemon answers in milliseconds, but the mock lane runs many
/// scenarios at once and each blocks a worker while its `rocm` child runs.
const DAEMON_WAIT: Duration = Duration::from_secs(30);

/// A `rocmd run` the scenario started, and the local webhook it listens on.
/// Killed and reaped on drop, so it never outlives the scenario.
#[derive(Debug)]
pub struct AutomationDaemon {
    child: std::process::Child,
    /// `http://127.0.0.1:<port>/automation-events`.
    endpoint: String,
}

impl Drop for AutomationDaemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Re-check `ready` every [`DAEMON_POLL`] until it holds, counting only the time
/// this step was actually polled (see `polled_deadline`).
async fn wait_until(what: &str, mut ready: impl FnMut() -> bool) {
    let mut deadline = PolledDeadline::start(RetryTiming {
        timeout: DAEMON_WAIT,
        resend_interval: DAEMON_POLL,
        poll_interval: DAEMON_POLL,
    });
    while !ready() {
        if let Some(expiry) = deadline.expired() {
            panic!("gave up after {expiry} waiting for {what}");
        }
        let budget = deadline.budget(DAEMON_POLL);
        deadline.run(tokio::time::sleep(budget)).await;
    }
}

#[given("the GPU thermal-protection watcher is enabled in propose mode")]
async fn thermal_watcher_in_propose_mode(world: &mut E2eWorld) {
    crate::run_rocm_ok(
        world,
        &[
            "automations",
            "enable",
            "gpu-thermal-protect",
            "--mode",
            "propose",
        ],
    );
}

/// The server's recorded process is a disposable `sleep` the scenario owns, not
/// this harness: teardown stops every recorded service, and that signals the
/// recorded pid.
#[given("a ready managed model server")]
async fn ready_managed_server(world: &mut E2eWorld) {
    let process = std::process::Command::new("sleep")
        .arg("300")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to start the process to record");
    let pid = process.id();
    world.recorded_process = Some(process);

    let paths = world.app_paths();
    paths
        .ensure()
        .expect("failed to create the data directories");
    // Planted as plain JSON, not through the record type: the harness exercises
    // the CLI's own reading of the record.
    let manifest = paths.service_manifest_path(HOT_SERVICE);
    let record = serde_json::json!({
        "service_id": HOT_SERVICE,
        "engine": "vllm",
        "model_ref": "tiny",
        "canonical_model_id": "Tiny/Test",
        "host": "127.0.0.1",
        "port": 11435,
        "endpoint_url": "http://127.0.0.1:11435/v1",
        "mode": "managed",
        "status": "ready",
        "supervisor_pid": pid,
        "supervisor_start_ticks": rocm_core::process_start_ticks(pid),
        "engine_pid": null,
        "manifest_path": manifest,
        "log_path": paths.service_log_path(HOT_SERVICE),
        "engine_state_path": paths.service_engine_state_path("vllm", HOT_SERVICE),
        "created_at_unix_ms": 1_700_000_000_000_u64,
    });
    std::fs::create_dir_all(manifest.parent().expect("manifest path has a parent"))
        .expect("failed to create the services directory");
    std::fs::write(
        &manifest,
        serde_json::to_vec_pretty(&record).expect("failed to serialize the record"),
    )
    .expect("failed to write the managed server record");
}

#[given("the automation daemon is running with its local webhook")]
async fn daemon_running_with_webhook(world: &mut E2eWorld) {
    let root = world
        .isolated_root
        .as_ref()
        .expect("no isolated root")
        .path()
        .to_path_buf();
    // No `amd-smi` on PATH, so the daemon's own telemetry polling raises no
    // alert of its own: the only alerts are the ones the scenario posts.
    let empty_path = root.join("empty-path");
    std::fs::create_dir_all(&empty_path).expect("failed to create the isolated PATH directory");
    let (stdout_log, stderr_log) = (root.join("rocmd.out"), root.join("rocmd.err"));
    let mut command = std::process::Command::new(crate::rocmd_binary());
    // Port 0 binds an ephemeral port; the startup banner names the real one.
    command.args(["run", "--automations-enabled", "--local-webhook-port", "0"]);
    world.isolate_cmd(&mut command);
    command
        .env("PATH", &empty_path)
        .stdin(Stdio::null())
        .stdout(std::fs::File::create(&stdout_log).expect("failed to create the rocmd stdout log"))
        .stderr(std::fs::File::create(&stderr_log).expect("failed to create the rocmd stderr log"));
    let child = command.spawn().expect("failed to start rocmd");
    let mut daemon = AutomationDaemon {
        child,
        endpoint: String::new(),
    };

    let mut endpoint = None;
    wait_until("rocmd to print its webhook endpoint", || {
        let banner = std::fs::read_to_string(&stdout_log).unwrap_or_default();
        endpoint = banner
            .lines()
            .find_map(|line| line.trim().strip_prefix("local_webhook_endpoint: "))
            .filter(|value| value.starts_with("http://"))
            .map(str::to_owned);
        if endpoint.is_none()
            && let Ok(Some(status)) = daemon.child.try_wait()
        {
            panic!(
                "rocmd exited with {status} before it was ready:\n{banner}\n{}",
                std::fs::read_to_string(&stderr_log).unwrap_or_default()
            );
        }
        endpoint.is_some()
    })
    .await;
    daemon.endpoint = endpoint.expect("the wait returned without an endpoint");
    world.automation_daemon = Some(daemon);
}

/// POST the thermal alert the dashboard's telemetry would raise for
/// [`HOT_SERVICE`] to the daemon's local webhook, as a plain HTTP/1.1 request.
fn post_thermal_alert(endpoint: &str) {
    let (authority, path) = endpoint
        .strip_prefix("http://")
        .and_then(|rest| rest.split_once('/'))
        .unwrap_or_else(|| panic!("unexpected webhook endpoint `{endpoint}`"));
    let body = serde_json::json!({
        "watcher_hint": "gpu-thermal-protect",
        "kind": "gpu.thermal_pressure",
        "service_id": HOT_SERVICE,
        "reason": "hotspot_temperature_threshold",
        "payload": { "summary": "GPU 0 hotspot temperature is 96 C (limit 95 C)" },
    })
    .to_string();
    let request = format!(
        "POST /{path} HTTP/1.1\r\nHost: {authority}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let mut stream = std::net::TcpStream::connect(authority)
        .unwrap_or_else(|error| panic!("failed to reach the webhook at {authority}: {error}"));
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .expect("failed to set the webhook read timeout");
    stream
        .write_all(request.as_bytes())
        .expect("failed to send the thermal alert");
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .expect("failed to read the webhook's answer");
    assert!(
        response.starts_with("HTTP/1.1 202"),
        "the daemon did not accept the thermal alert:\n{response}"
    );
}

#[when(regex = r"^a GPU thermal alert for the server reaches the daemon(?: again)?$")]
async fn thermal_alert_reaches_daemon(world: &mut E2eWorld) {
    let daemon = world
        .automation_daemon
        .as_ref()
        .expect("no automation daemon is running");
    post_thermal_alert(&daemon.endpoint);
}

#[when("more proposals for other servers are queued after it than the history keeps")]
async fn newer_proposals_for_other_servers(world: &mut E2eWorld) {
    let paths = world.app_paths();
    for n in 0..=REVIEWED_PROPOSALS_RETAINED {
        append_automation_proposal(
            &paths,
            &AutomationProposalRecord {
                at_unix_ms: 1_700_000_000_000,
                proposal_id: String::new(),
                watcher_id: "gpu-thermal-protect".to_owned(),
                action: "queue_stop_server_proposal".to_owned(),
                title: "Review GPU pressure stop".to_owned(),
                message: "asking before stopping another server".to_owned(),
                status: "pending".to_owned(),
                service_id: Some(format!("e2e-other-{n}")),
                tool: Some("stop_server".to_owned()),
                arguments: serde_json::json!({}),
                reviewed_at_unix_ms: None,
            },
        )
        .expect("failed to queue a proposal for another server");
    }
}

/// Stop requests for [`HOT_SERVICE`] still waiting for review.
fn pending_stop_requests(paths: &rocm_core::AppPaths) -> usize {
    load_automation_proposals(paths)
        .expect("proposal history reads back")
        .iter()
        .filter(|proposal| {
            proposal.status == "pending"
                && proposal.tool.as_deref() == Some("stop_server")
                && proposal.service_id.as_deref() == Some(HOT_SERVICE)
        })
        .count()
}

fn stop_already_waiting_recorded(paths: &rocm_core::AppPaths) -> bool {
    load_recent_automation_events(paths, 50)
        .expect("automation events read back")
        .iter()
        .any(|event| {
            event.action == "stop_proposal_already_pending"
                && event.service_id.as_deref() == Some(HOT_SERVICE)
        })
}

#[then("one stop request for the server is waiting for review")]
async fn one_stop_request_waiting(world: &mut E2eWorld) {
    let paths = world.app_paths();
    wait_until("the daemon to queue a stop request", || {
        pending_stop_requests(&paths) >= 1
    })
    .await;
    assert_eq!(pending_stop_requests(&paths), 1);
}

#[then("the daemon records that a stop request is already waiting for the server")]
async fn daemon_records_stop_already_waiting(world: &mut E2eWorld) {
    let paths = world.app_paths();
    // A second stop request ends the wait too: that is the defect, and it is
    // reported below rather than after a timeout.
    wait_until("the daemon to handle the repeated alert", || {
        stop_already_waiting_recorded(&paths) || pending_stop_requests(&paths) >= 2
    })
    .await;
    assert!(
        stop_already_waiting_recorded(&paths),
        "the daemon queued another stop request instead of noting the first was still waiting; \
         {} are now waiting",
        pending_stop_requests(&paths)
    );
}

#[then("only one stop request for the server is waiting for review")]
async fn still_one_stop_request_waiting(world: &mut E2eWorld) {
    assert_eq!(
        pending_stop_requests(&world.app_paths()),
        1,
        "the repeated alert must not queue a second stop request for {HOT_SERVICE}"
    );
}

#[then("every listed check can be turned on from what the listing shows")]
async fn assert_listed_checks_enableable(world: &mut E2eWorld) {
    let listing = world
        .cli_output
        .clone()
        .expect("the background checks were never listed");
    let checks = listed_checks(&listing);
    // Guard the guard: a listing this step failed to parse would otherwise
    // "prove" the contract by checking nothing at all.
    assert!(
        !checks.is_empty(),
        "no background checks were found in the listing:\n{listing}"
    );

    let mut unreachable = Vec::new();
    for (name, candidates) in &checks {
        let mut last = String::new();
        let mut reached = false;
        for candidate in candidates {
            let (stdout, stderr, rc) =
                crate::run_rocm(world, &["automations", "enable", candidate]);
            if rc == 0 {
                reached = true;
                break;
            }
            last = format!(
                "tried {candidate:?}: rc={rc} {}",
                stderr
                    .trim()
                    .lines()
                    .next()
                    .unwrap_or_else(|| stdout.trim())
            );
        }
        if !reached {
            unreachable.push(format!(
                "{name:?} → none of {} candidate(s) worked; last {last}",
                candidates.len()
            ));
        }
    }
    assert!(
        unreachable.is_empty(),
        "the listing names {} background check(s) that cannot be turned on from what it \
         shows:\n  {}\nfull listing:\n{listing}",
        unreachable.len(),
        unreachable.join("\n  "),
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
