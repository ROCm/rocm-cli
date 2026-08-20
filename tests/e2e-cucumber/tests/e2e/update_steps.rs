// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocm update` (report only). Run with NO managed runtimes so the
//! report needs no network (with a runtime present, `update` reaches the TheRock
//! index to resolve the latest version). The report's update-feed status block is
//! host-invariant and is what pins the "distinguishes configured from
//! not-configured feeds" behaviour. Contracts verified against the running Linux
//! binary (WL-502). Mock lane.

use cucumber::{given, then, when};

use crate::E2eWorld;

#[given("a machine with no managed runtimes")]
async fn no_managed_runtimes(_world: &mut E2eWorld) {
    // The World's isolated data dir starts with an empty runtimes registry, so
    // `update` has nothing to check against the network. No setup required.
}

#[when("the user checks for updates")]
async fn check_updates(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["update"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the report shows there are no managed runtimes to update")]
async fn no_runtimes_to_update(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("managed runtimes: none"),
        "expected 'managed runtimes: none', got:\n{out}"
    );
}

#[then("it reports each update feed's status, marking unpublished feeds as not configured")]
async fn reports_feed_status(world: &mut E2eWorld) {
    let out = ok_output(world);
    // The update_surfaces block reports one line per feed; assert each feed appears
    // with its host-invariant status. The CLI feed is not published yet, so it must
    // read not_configured — the "not-configured" side of the distinction; the
    // engines/recipes feeds report their own stable states.
    for needle in [
        "cli: installed=",
        "status=not_configured",
        "engines:",
        "status=package_managed",
        "model_recipes:",
        "status=built_in",
        "runtimes:",
    ] {
        assert!(
            out.contains(needle),
            "expected update feed detail {needle:?}, got:\n{out}"
        );
    }
}

// ── Helpers ────────────────────────────────────────────────────────

fn ok_output(world: &E2eWorld) -> String {
    let rc = world.cli_rc.expect("no command rc recorded");
    let combined = format!(
        "{}\n{}",
        world.cli_output.as_deref().unwrap_or(""),
        world.cli_stderr.as_deref().unwrap_or("")
    );
    assert_eq!(rc, 0, "expected success, got rc={rc}:\n{combined}");
    world.cli_output.clone().unwrap_or_default()
}
