// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocm bootstrap setup`'s non-interactive fallback. Piped, like
//! `examine_steps.rs` — this path is plain stdout, not the crossterm TUI, so it
//! needs no PTY.

use cucumber::{then, when};

use crate::E2eWorld;

#[when("the user runs bootstrap setup without a terminal")]
async fn run_bootstrap_setup_no_tty(world: &mut E2eWorld) {
    // `run_rocm` pipes the child's stdio, so `interactive_terminal()` sees no
    // TTY — the same shape every CI runner or script invocation has.
    let (stdout, stderr, rc) = crate::run_rocm(world, &["bootstrap", "setup"]);
    assert_eq!(
        rc, 0,
        "bootstrap setup without a terminal must not fail:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    world.cli_output = Some(stdout);
}

#[then("the CLI tells the user how to choose an install folder")]
async fn bootstrap_setup_message_mentions_folder_choice(world: &mut E2eWorld) {
    let stdout = world
        .cli_output
        .as_deref()
        .expect("no bootstrap setup output recorded");
    assert!(
        stdout.contains("choose an install folder"),
        "bootstrap setup's non-interactive message dropped the install-folder mention:\n{stdout}"
    );
}
