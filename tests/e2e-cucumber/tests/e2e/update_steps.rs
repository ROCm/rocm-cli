// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocm update` (report only). Run with NO managed runtimes so the
//! report needs no network (with a runtime present, `update` reaches the TheRock
//! index to resolve the latest version). The report's update-feed status block is
//! host-invariant and is what pins the "distinguishes configured from
//! not-configured feeds" behaviour. Contracts verified against the running Linux
//! binary (EAI-8072). Mock lane.
//!
//! The preview steps below are argument handling only: nothing there contacts a
//! package index or changes the machine either, so they run on every lane too.

use cucumber::{given, then, when};

use crate::E2eWorld;

/// The exit code a CLI uses to reject the way it was CALLED, as opposed to
/// failing at the work it was asked to do. Anything the command decides about
/// the machine — no runtime registered, nothing to update — is a different
/// outcome and not what that scenario is about.
const USAGE_ERROR: i32 = 2;

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
    // The update_surfaces block reports one line per feed. Assert each feed's status
    // ON ITS OWN LINE, so a status attributed to the wrong feed fails — a check that
    // only looked for the substrings anywhere would pass even if `not_configured`
    // and `package_managed` were swapped between the cli and engines feeds. The CLI
    // feed is not published yet (the "not configured" side of the distinction);
    // engines and recipes report their own stable states.
    for (feed, status) in [
        ("cli:", "status=not_configured"),
        ("engines:", "status=package_managed"),
        ("model_recipes:", "status=built_in"),
    ] {
        let line = out
            .lines()
            .map(str::trim)
            .find(|line| line.starts_with(feed));
        match line {
            Some(line) => assert!(
                line.contains(status),
                "update feed {feed:?} did not report {status:?} on its own line; got {line:?}\n\nfull output:\n{out}"
            ),
            None => panic!("no update feed line for {feed:?} in:\n{out}"),
        }
    }
}

#[when("the user asks to see what updating would do without asking for it to be done")]
async fn user_previews_update(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["update", "--dry-run"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the request is accepted rather than refused as a misuse")]
async fn assert_preview_accepted(world: &mut E2eWorld) {
    let combined = format!(
        "{}{}",
        world.cli_output.as_deref().unwrap_or(""),
        world.cli_stderr.as_deref().unwrap_or("")
    );
    // Deliberately NOT "exits 0": a host with no ROCm install registered has
    // nothing to check and says so, which is a legitimate answer to a legitimate
    // question. The contract is only that asking was allowed.
    assert_ne!(
        world.cli_rc,
        Some(USAGE_ERROR),
        "asking to preview an update was rejected as a misuse of the command:\n{combined}"
    );
}

#[then("the machine still manages no runtimes")]
async fn assert_still_manages_no_runtimes(world: &mut E2eWorld) {
    // Scoped deliberately narrowly, because the obvious stronger claim is one
    // this fixture CANNOT make. It would be natural to read this as "the
    // preview did not perform the update", but on an empty registry that
    // outcome is unreachable whether the product is honest or not: `--apply`
    // resolves a runtime to upgrade through `select_runtime_update_source`,
    // which bails with "no managed runtimes are registered" when none is
    // (`apps/rocm/src/main.rs:15939`). `update` upgrades a managed runtime; it
    // never installs a first one. So "manages none" here is guaranteed by the
    // fixture, and an assertion resting on it would be satisfied forever.
    //
    // What it does hold is the readback path: after a preview, `update` still
    // answers, exits 0, and reports the same machine `update-01` pins. That is
    // worth asserting and it can fail — but it is not the non-mutation contract.
    //
    // Proving THAT needs a scenario with a managed runtime registered, where a
    // performed update is observable. It is not done here on purpose: with a
    // manifest present, `render_update_report` resolves the latest version per
    // runtime (`apps/rocm/src/therock.rs:810` → `resolve_latest_for_manifest`),
    // which reaches the TheRock index — and this scenario runs on the mock lane,
    // which has no network. Tracked on EAI-8010 rather than forced in here.
    //
    // Today this never runs: the step above fails first on the usage error, so
    // it neither weakens nor satisfies the row that pins EAI-8010.
    let (stdout, stderr, rc) = crate::run_rocm(world, &["update"]);
    assert_eq!(
        rc, 0,
        "`update` stopped answering after the preview:\n{stdout}{stderr}"
    );
    assert!(
        stdout.contains("managed runtimes: none"),
        "reading the machine back after the preview did not report the empty registry \
         this scenario runs against:\n{stdout}"
    );
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
