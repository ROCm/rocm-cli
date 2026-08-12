// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocm update`.
//!
//! Argument handling only: nothing here contacts a package index or changes the
//! machine, so these run on every lane.

use cucumber::{then, when};

use crate::E2eWorld;

/// The exit code a CLI uses to reject the way it was CALLED, as opposed to
/// failing at the work it was asked to do. Anything the command decides about
/// the machine — no runtime registered, nothing to update — is a different
/// outcome and not what this scenario is about.
const USAGE_ERROR: i32 = 2;

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
