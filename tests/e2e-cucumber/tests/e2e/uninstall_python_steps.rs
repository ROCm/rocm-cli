// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for the `uninstall.feature` scenarios about a `rocm` that a Python
//! package manager installed (`pip install rocm-cli`).
//!
//! Black-box: a copy of the binary under test is placed where each package
//! manager would put it and a dry run is made from that copy, so the run judges
//! the copy's own folder. Only `--dry-run` is ever used, so nothing here
//! removes the copy or anything else; the copy is checked to still be there.

use std::path::{Path, PathBuf};

use cucumber::{given, then, when};

use crate::E2eWorld;
use crate::e2e::uninstall_steps::{output, root, run_binary_in};

/// The folder a Python environment keeps its scripts in on this host.
const fn scripts_folder() -> &'static str {
    if cfg!(windows) { "Scripts" } else { "bin" }
}

const fn binary_name() -> &'static str {
    if cfg!(windows) { "rocm.exe" } else { "rocm" }
}

fn python_env_copy(world: &E2eWorld) -> PathBuf {
    root(world)
        .join("python-env")
        .join(scripts_folder())
        .join(binary_name())
}

fn plain_folder_copy(world: &E2eWorld) -> PathBuf {
    root(world)
        .join("plain-install")
        .join("bin")
        .join(binary_name())
}

/// Copy the `rocm` under test to `target`, where its folder decides what an
/// uninstall run from it does.
fn place_copy(target: &Path) {
    let source = crate::rocm_binary();
    let folder = target.parent().expect("a copy has a folder");
    std::fs::create_dir_all(folder).expect("failed to create the copy's folder");
    std::fs::copy(&source, target)
        .unwrap_or_else(|error| panic!("failed to copy {source} to {}: {error}", target.display()));
}

#[given("a copy of the CLI inside a Python virtual environment")]
async fn copy_in_python_env(world: &mut E2eWorld) {
    let target = python_env_copy(world);
    place_copy(&target);
    // `python -m venv`, `pipx` and `uv tool` all write this at the root.
    let env_root = target
        .parent()
        .and_then(|folder| folder.parent())
        .expect("a script folder has an environment root");
    std::fs::write(env_root.join("pyvenv.cfg"), "home = /usr/bin\n")
        .expect("failed to write pyvenv.cfg");
}

#[given("a copy of the CLI in a plain install folder with no Python environment around it")]
async fn copy_in_plain_folder(world: &mut E2eWorld) {
    place_copy(&plain_folder_copy(world));
}

fn dry_run_from(world: &mut E2eWorld, copy: &Path) {
    // No --keep-binaries: whether the binaries are kept is the thing under
    // test. --dry-run is what keeps the run from removing the copy.
    run_binary_in(
        world,
        &copy.to_string_lossy(),
        &[
            "uninstall",
            "--dry-run",
            "--keep-config",
            "--keep-data",
            "--keep-cache",
        ],
        None,
    );
}

#[when("the user dry-runs an uninstall from the copy in the Python environment")]
async fn dry_run_from_python_env(world: &mut E2eWorld) {
    let copy = python_env_copy(world);
    dry_run_from(world, &copy);
}

#[when("the user dry-runs an uninstall from the copy in the plain install folder")]
async fn dry_run_from_plain_folder(world: &mut E2eWorld) {
    let copy = plain_folder_copy(world);
    dry_run_from(world, &copy);
}

#[then("the uninstall leaves the binaries to the package manager and names its commands")]
async fn leaves_binaries_to_the_package_manager(world: &mut E2eWorld) {
    let stdout = output(world);
    assert_eq!(world.cli_rc, Some(0), "the dry run must succeed:\n{stdout}");
    for expected in [
        "binary removal skipped because",
        "sits in a Python environment",
        "pip uninstall rocm-cli",
        "uv tool uninstall rocm-cli",
    ] {
        assert!(stdout.contains(expected), "missing {expected:?}:\n{stdout}");
    }
    // The state the message describes: no binary is planned for removal.
    assert!(
        !stdout.contains("- binary:"),
        "a pip-owned binary must not be planned for removal:\n{stdout}"
    );
}

#[then("the uninstall plans to remove the binary")]
async fn plans_to_remove_the_binary(world: &mut E2eWorld) {
    let stdout = output(world);
    assert_eq!(world.cli_rc, Some(0), "the dry run must succeed:\n{stdout}");
    assert!(
        stdout.contains("- binary:"),
        "an installer-managed binary is still planned for removal:\n{stdout}"
    );
    assert!(
        !stdout.contains("binary removal skipped because"),
        "no Python notice outside a Python environment:\n{stdout}"
    );
}

fn assert_copy_still_there(copy: &Path) {
    assert!(
        copy.is_file(),
        "a dry run removed the copy it was run from: {}",
        copy.display()
    );
}

#[then("the copy in the Python environment is still there")]
async fn the_python_env_copy_is_still_there(world: &mut E2eWorld) {
    assert_copy_still_there(&python_env_copy(world));
}

#[then("the copy in the plain install folder is still there")]
async fn the_plain_folder_copy_is_still_there(world: &mut E2eWorld) {
    assert_copy_still_there(&plain_folder_copy(world));
}
