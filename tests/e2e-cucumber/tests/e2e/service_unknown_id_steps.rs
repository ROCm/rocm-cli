// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for commands that name a local server with no record.
//!
//! Black-box: nothing is planted, so the isolated data dir has no record for
//! the id; the steps run the real binary and check its exit code, its error
//! line, and that the listing command it points at agrees. No GPU, no
//! network — mock lane.

use cucumber::{then, when};

use crate::E2eWorld;

#[when(expr = "the user runs {string} naming a local server that does not exist")]
async fn run_with_unknown_id(world: &mut E2eWorld, command: String) {
    let args: Vec<&str> = command.split_whitespace().collect();
    let (stdout, stderr, rc) = crate::run_rocm(world, &args);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then(expr = "the CLI says there is no local server named {string} and points at the full list")]
async fn says_not_found(world: &mut E2eWorld, service_id: String) {
    let rc = world.cli_rc.expect("no command rc recorded");
    let stdout = world.cli_output.as_deref().unwrap_or_default();
    let stderr = world.cli_stderr.as_deref().unwrap_or_default();
    assert_ne!(
        rc, 0,
        "naming a missing server must fail:\n{stdout}\n{stderr}"
    );
    // Everything from `Error:` on is what `main` prints for the failure; with a
    // cause chained under it, `Caused by:` lines would follow.
    let error = stderr.find("Error:").map_or_else(
        || panic!("no error was printed:\n{stdout}\n{stderr}"),
        |start| stderr[start..].trim_end(),
    );
    assert_eq!(
        error,
        format!("Error: No local server named `{service_id}`. See `rocm services list --all`."),
        "the error must be the one plain line, with no OS cause chained under it"
    );
    // The old message's tells, checked by name so a regression says which.
    for leaked in ["Caused by", "os error", ".json"] {
        assert!(
            !error.contains(leaked),
            "`{leaked}` leaked into the error:\n{error}"
        );
    }
}

/// The advice is only true if `rocm services list --all` runs as named and
/// really has no row for the id.
#[then(expr = "the full list really has no local server named {string}")]
async fn full_list_lacks(world: &mut E2eWorld, service_id: String) {
    let listed = crate::run_rocm_ok(world, &["services", "list", "--all"]);
    assert!(
        !listed.contains(&service_id),
        "the list the error points at names `{service_id}`:\n{listed}"
    );
}
