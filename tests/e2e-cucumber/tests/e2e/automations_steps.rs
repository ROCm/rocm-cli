// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocm automations enable/disable/list`. Black-box against the
//! isolated config dir. `automations enable` would otherwise spawn a detached
//! background daemon (`rocm daemon`) on first enable, which both adds a
//! nondeterministic `helper:` line and leaks a process past the scenario. To keep
//! the mock lane hermetic, every scenario first plants an automation
//! runtime-state marking the daemon already running under THIS test process's
//! (live) pid, so the CLI's double-spawn guard skips the spawn. Contracts
//! verified against the running Linux binary (EAI-8072, EAI-8047).
//!
//! Two slices live here. The enable/disable/mode steps act on a watcher id the
//! test already knows. The listing steps (scenario 4) assert the complementary
//! discoverability contract: everything the listing shows a user must also be
//! able to act on, so they derive each check's identifier FROM the listing rather
//! than knowing it in advance — hard-coding the real ids would keep passing
//! against a listing that publishes none of them, which is the defect pinned.

use cucumber::{given, then, when};

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

/// The identifier a check block explicitly EXPOSES, if any.
///
/// The contract is "enable-able from what the listing shows", so the moment the
/// product publishes a real identifier — inline on the header (`Server recovery
/// [server-recover]` or `... (server-recover)`) or on an indented detail line
/// (`id: server-recover`) — this must pick THAT up, or a correct fix would stay
/// xfailed forever (the row would never go stale). Returns `None` only when the
/// block names no identifier at all, which is today's defect.
fn exposed_identifier(block: &[&str]) -> Option<String> {
    // A detail line that names the id outright, in the obvious shapes a fix
    // might use: "id: server-recover", "identifier = server-recover".
    for line in block {
        let line = line.trim();
        for key in ["id:", "id =", "identifier:", "identifier ="] {
            if let Some(rest) = line.strip_prefix(key) {
                let id = rest.trim().trim_matches(|c| c == '"' || c == '`');
                if !id.is_empty() {
                    return Some(id.to_owned());
                }
            }
        }
    }
    // An id printed inline on the header, in brackets or parens after the name:
    // "Server recovery [server-recover]". The state suffix "(on)"/"(off)" has
    // already been stripped from `header` before this is called.
    let header = block.first()?.trim();
    for (open, close) in [('[', ']'), ('(', ')')] {
        if let (Some(o), Some(c)) = (header.rfind(open), header.rfind(close))
            && o < c
        {
            let inner = header[o + 1..c].trim();
            // A single token with no spaces is an identifier; a phrase is
            // still part of the display name, not an id.
            if !inner.is_empty() && !inner.contains(char::is_whitespace) {
                return Some(inner.to_owned());
            }
        }
    }
    None
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
fn listed_checks(listing: &str) -> Vec<(String, String)> {
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
        let identifier = exposed_identifier(&block).unwrap_or_else(|| slug_of(name));
        checks.push((name.to_owned(), identifier));
    }
    checks
}

// ── Given ──────────────────────────────────────────────────────────

#[given("an enabled automation watcher")]
async fn enabled_watcher(world: &mut E2eWorld) {
    suppress_daemon_spawn(world);
    crate::run_rocm_ok(
        world,
        &["automations", "enable", WATCHER, "--mode", "observe"],
    );
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

#[then("the CLI confirms the watcher is disabled")]
async fn confirm_disabled(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("automation watcher disabled") && out.contains(&format!("watcher: {WATCHER}")),
        "expected a disable confirmation for {WATCHER}, got:\n{out}"
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

#[then("every listed check can be turned on by name")]
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
    for (name, identifier) in &checks {
        let (stdout, stderr, rc) = crate::run_rocm(world, &["automations", "enable", identifier]);
        if rc != 0 {
            unreachable.push(format!(
                "{name:?} → tried {identifier:?}: rc={rc} {}",
                stderr
                    .trim()
                    .lines()
                    .next()
                    .unwrap_or_else(|| stdout.trim())
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
