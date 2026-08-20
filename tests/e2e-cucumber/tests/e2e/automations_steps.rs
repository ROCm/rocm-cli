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

/// Turn one line of the listing into the identifier a user would try.
///
/// A reader has only the displayed name to go on, so this applies the obvious
/// reading of it: lowercase, words joined by dashes. That is the *charitable*
/// derivation — if a real identifier were printed the parse would pick it up
/// verbatim, and if a user could reasonably guess it this reproduces the guess.
/// Anything the listing never shows is, by definition, not derivable here.
fn identifier_from(listed_name: &str) -> String {
    listed_name
        .split_whitespace()
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join("-")
}

/// The checks the listing advertises, as `(displayed name, derived identifier)`.
///
/// A check is a two-space-indented line carrying a parenthesised on/off state —
/// `  Server recovery (off)`. The surrounding report lines are either less
/// indented (the header block) or more (a check's own `setting:` / `does:`
/// detail), and the report's other sections list events rather than checks, so
/// the state suffix is what distinguishes a check from anything else.
fn listed_checks(listing: &str) -> Vec<(String, String)> {
    listing
        .lines()
        .filter(|line| line.starts_with("  ") && !line.starts_with("   "))
        .filter_map(|line| {
            let line = line.trim();
            let name = line
                .strip_suffix("(on)")
                .or_else(|| line.strip_suffix("(off)"))?
                .trim();
            (!name.is_empty()).then(|| (name.to_owned(), identifier_from(name)))
        })
        .collect()
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
                stderr.trim().lines().next().unwrap_or(stdout.trim())
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
