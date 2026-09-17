// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocm storage report` — the command whose job is to tell a user
//! what ROCm CLI has put on their disk.
//!
//! Black-box throughout: the scenario plants a local server record the same way
//! `rocm serve --managed` would, then reads the real binary's rendered report.
//! No GPU and no network, so this runs on the mock lane every PR.

use crate::E2eWorld;
use cucumber::{then, when};

#[when("the user asks what ROCm CLI is keeping on disk")]
async fn run_storage_report(world: &mut E2eWorld) {
    let stdout = crate::run_rocm_ok(world, &["storage", "report"]);
    world.cli_output = Some(stdout);
}

fn report(world: &E2eWorld) -> &str {
    world
        .cli_output
        .as_deref()
        .expect("no CLI output captured - did the When step run?")
}

#[then("the report names the folder holding local server records")]
async fn assert_local_server_records_row(world: &mut E2eWorld) {
    let stdout = report(world);
    // The row: label, and the real path under the scenario's isolated data dir
    // (so this cannot pass against some other host's services folder).
    let lines: Vec<&str> = stdout.lines().collect();
    let row_at = lines
        .iter()
        .position(|line| line.trim_start().starts_with("- local server records:"))
        .unwrap_or_else(|| {
            panic!("the report must name the local server records folder:\n{stdout}")
        });
    // The label, path and note all render whether or not a record exists - only
    // the size field changes (`not present` for a missing folder). Pin that
    // field, or the scenario's `Given a local server attempt has failed` is
    // inert and every assertion below passes with no record on disk at all.
    assert!(
        !lines[row_at].contains("not present"),
        "the planted record must make the folder present, not `not present`:\n{stdout}"
    );
    let services = world
        .isolated_root
        .as_ref()
        .expect("no isolated root")
        .path()
        .join("data")
        .join("services");
    assert!(
        stdout.contains(&services.display().to_string()),
        "the report must print the real services path ({}):\n{stdout}",
        services.display()
    );
    // The note has to explain what the size is made of. The engine's redirected
    // stdout/stderr lives in this folder too and nothing rotates it, so a note
    // naming only the record would contradict the number printed beside it.
    assert!(
        stdout.contains("one record plus the engine log per local server launch"),
        "the report must say what is in the folder:\n{stdout}"
    );
    // And it must route the user somewhere real: `--all` is the only listing
    // that shows a record which is no longer running.
    assert!(
        stdout.contains("rocm services list --all"),
        "the report must say how to list the records:\n{stdout}"
    );
}

#[then("the report says which of its folders can be downloaded again")]
async fn assert_re_downloadable_notes(world: &mut E2eWorld) {
    let stdout = report(world);
    // Printing the note above required the `ROCm CLI folders` loop to render
    // notes at all, which it never did. The two archive rows have always carried
    // this note in the data and in `--json`; now they show it in the text report
    // too. It is the most actionable line in the output - `rocm storage
    // remove-downloads` acts on exactly these - so pin it here rather than leave
    // it as an unwitnessed side effect.
    //
    // Asserted against the row rather than merely somewhere in the output
    // because the note is per-row and both rows carry the *same* string: a
    // `contains` check is satisfied by either one, so dropping the note from
    // `downloaded helper tools` would leave this green. Anchoring also pins
    // the shape the user reads - the `note: ` line directly under its own row
    // - so notes rendered detached from their rows, or beside the wrong
    // label, fail here rather than ship.
    let lines: Vec<&str> = stdout.lines().collect();
    for label in ["downloaded ROCm archives", "downloaded helper tools"] {
        let at = lines
            .iter()
            .position(|line| line.trim_start().starts_with(&format!("- {label}:")))
            .unwrap_or_else(|| panic!("no `{label}` row in the report:\n{stdout}"));
        assert_eq!(
            lines.get(at + 1).map(|line| line.trim()),
            Some("note: can be downloaded again; safe to remove"),
            "`{label}` must be marked as re-downloadable:\n{stdout}"
        );
    }
}
