// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `lemonade_installs.feature`: what the user is told about the
//! Lemonade copies ROCm CLI puts on disk (ROCMAI-825).
//!
//! Black-box throughout, and nothing is downloaded: the refusal scenarios
//! reach the engine's free-space check through the `rocm/e2e-test-hooks`
//! override instead of a full drive.

use std::path::PathBuf;

use cucumber::{given, then, when};

use crate::E2eWorld;

/// Free space the refusal scenario reports for the install drive: well below
/// the multi-gigabyte download, so the engine refuses before fetching anything.
const NEARLY_FULL_BYTES: &str = "1073741824";

fn data_dir(world: &E2eWorld) -> PathBuf {
    world
        .isolated_root
        .as_ref()
        .expect("no isolated root")
        .path()
        .join("data")
}

/// Where Lemonade installs when no ROCm runtime is active.
fn data_folder_tree(world: &E2eWorld) -> PathBuf {
    data_dir(world).join("engines").join("lemonade")
}

/// Lemonade's install record, in the engine's own location.
fn install_record(world: &E2eWorld) -> PathBuf {
    data_folder_tree(world)
        .join("manifests")
        .join("runtime.json")
}

fn combined_output(world: &E2eWorld) -> String {
    format!(
        "{}\n{}",
        world.cli_output.as_deref().unwrap_or_default(),
        world.cli_stderr.as_deref().unwrap_or_default()
    )
}

#[given("the drive for a new Lemonade install cannot hold its download")]
async fn install_drive_nearly_full(world: &mut E2eWorld) {
    world.command_env.push((
        "ROCM_E2E_LEMONADE_FREE_SPACE_BYTES",
        NEARLY_FULL_BYTES.into(),
    ));
}

#[when("the user installs the Lemonade engine")]
async fn user_installs_lemonade(world: &mut E2eWorld) {
    let (stdout, stderr, rc) =
        crate::run_rocm_with_scenario_env(world, &["engines", "install", "lemonade", "--yes"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the Lemonade install is refused before anything is downloaded")]
async fn assert_lemonade_install_refused(world: &mut E2eWorld) {
    let output = combined_output(world);
    // Without the hook the engine sees the real drive and starts a real
    // multi-gigabyte install. Name that cause rather than a baffling mismatch.
    assert!(
        output.contains("Installing Lemonade"),
        "no install notice: the binary under test was probably built without the \
         `rocm/e2e-test-hooks` feature, which `cargo xtask e2e` enables:\n{output}"
    );
    assert_ne!(
        world.cli_rc,
        Some(0),
        "the install must fail when the drive cannot hold it:\n{output}"
    );
    let tree = data_folder_tree(world);
    // The claim "Nothing was downloaded" is checked against the disk: no
    // archive cache and no extracted runtime.
    assert!(
        !tree.join("downloads").exists(),
        "an archive was downloaded despite the refusal:\n{output}"
    );
    assert!(
        !tree.join("runtime").exists(),
        "a runtime was extracted despite the refusal:\n{output}"
    );
    assert!(
        !install_record(world).exists(),
        "a refused install must not record Lemonade as installed:\n{output}"
    );
}

#[then("the user is told what the Lemonade install needs and where it would go")]
async fn assert_lemonade_install_notice(world: &mut E2eWorld) {
    let output = combined_output(world);
    let tree = data_folder_tree(world);
    for expected in [
        format!("and its llama.cpp ROCm backend into {}", tree.display()),
        "This downloads about".to_owned(),
        "of disk space (1.0 GiB free on that drive)".to_owned(),
        "not enough free disk space to install Lemonade and its llama.cpp ROCm backend".to_owned(),
        "Nothing was downloaded.".to_owned(),
    ] {
        assert!(
            output.contains(&expected),
            "missing `{expected}`:\n{output}"
        );
    }
}
