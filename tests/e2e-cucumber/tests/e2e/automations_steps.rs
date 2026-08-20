// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocm automations`.
//!
//! The scenario asserts a discoverability contract: everything the listing shows
//! a user must also be able to act on. So these steps deliberately derive each
//! check's identifier FROM the listing rather than knowing it in advance —
//! hard-coding the real ids would keep passing against a listing that publishes
//! none of them, which is the defect being pinned.

use cucumber::{given, then, when};
use serde_json::json;

use crate::E2eWorld;

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

#[given("a machine with no background checks turned on")]
async fn no_background_checks(world: &mut E2eWorld) {
    // The scenario's isolated config starts empty, so every check is already off
    // and nothing needs planting for the precondition itself.
    //
    // What DOES need planting is a background helper: `automations enable`
    // launches a detached `rocm daemon` unless the recorded one is alive, and
    // that child would outlive the scenario, survive its temp dir, and pile up on
    // a persistent runner. Recording this test process as the running helper
    // makes the liveness check find one and skip the spawn — the same trick, for
    // the same reason, as pointing a planted service record's pids at this
    // process (see `E2eWorld::register_mock_service_with`). Black-box: plain JSON
    // matching the CLI's on-disk schema, not a typed import from the product.
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let automations = root.path().join("data").join("automations");
    std::fs::create_dir_all(&automations).expect("failed to create the automations dir");
    let state = json!({
        "running": true,
        "automations_enabled": false,
        "daemon_pid": std::process::id(),
        "started_at_unix_ms": 1_700_000_000_000_u64,
        "last_tick_unix_ms": 1_700_000_000_000_u64,
        "active_watchers": [],
    });
    std::fs::write(
        automations.join("runtime-state.json"),
        serde_json::to_vec_pretty(&state).expect("runtime state serialises"),
    )
    .expect("failed to write the automation runtime state");
}

// ── When ───────────────────────────────────────────────────────────

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
