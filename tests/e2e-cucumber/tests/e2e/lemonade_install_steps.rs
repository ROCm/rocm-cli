// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `lemonade_installs.feature`: what the user is told about the
//! Lemonade copies ROCm CLI puts on disk (ROCMAI-825).
//!
//! Black-box throughout. The two Lemonade trees and the install record are
//! planted as plain files in the layout the engine writes, so no scenario
//! downloads anything; the refusal scenario reaches the engine's free-space
//! check through the `rocm/e2e-test-hooks` override instead of a full drive,
//! and the host-library refusal through the same build's library override.

use std::path::PathBuf;

use cucumber::{given, then, when};

use crate::E2eWorld;

/// Registry key of the planted runtime. Production-shaped (slug, no `:`), so it
/// is also a valid registry file name on Windows.
const RUNTIME_KEY: &str = "release-tarball-gfx1151";
/// Payload planted in each tree, chosen so the two render as distinct sizes.
const DATA_FOLDER_TREE_BYTES: usize = 3 * 1024 * 1024;
const RUNTIME_TREE_BYTES: usize = 5 * 1024 * 1024;
/// Free space the refusal scenarios report for the install drive: 1 MiB, below
/// what any Lemonade install needs (its archive and extracted files), so the
/// engine refuses before fetching anything. Room for that but not for the
/// multi-gigabyte ROCm backend only warns, and is unit-tested.
const NEARLY_FULL_BYTES: &str = "1048576";

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

fn runtime_install_root(world: &E2eWorld) -> PathBuf {
    data_dir(world)
        .join("runtimes")
        .join("tarball")
        .join(RUNTIME_KEY)
}

/// Where Lemonade installs while the planted runtime is active.
fn runtime_tree(world: &E2eWorld) -> PathBuf {
    runtime_install_root(world).join("engines").join("lemonade")
}

/// Lemonade's install record, in the engine's own location.
fn install_record(world: &E2eWorld) -> PathBuf {
    data_folder_tree(world)
        .join("manifests")
        .join("runtime.json")
}

const fn lemond_name() -> &'static str {
    if cfg!(windows) {
        "lemond.exe"
    } else {
        "lemond"
    }
}

/// An extracted tree: the server binary plus a llama.cpp backend payload.
fn plant_lemonade_tree(tree: &std::path::Path, payload_bytes: usize) {
    let runtime_dir = tree.join("runtime");
    let backend = runtime_dir.join("bin").join("llamacpp").join("rocm-stable");
    std::fs::create_dir_all(&backend).expect("failed to create backend dir");
    std::fs::write(runtime_dir.join(lemond_name()), b"").expect("failed to write lemond");
    std::fs::write(backend.join("payload.bin"), vec![0_u8; payload_bytes])
        .expect("failed to write backend payload");
}

fn stdout(world: &E2eWorld) -> &str {
    world
        .cli_output
        .as_deref()
        .expect("no CLI output captured - did the When step run?")
}

fn combined_output(world: &E2eWorld) -> String {
    format!(
        "{}\n{}",
        world.cli_output.as_deref().unwrap_or_default(),
        world.cli_stderr.as_deref().unwrap_or_default()
    )
}

#[given("the drive for a new Lemonade install cannot hold even Lemonade itself")]
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

/// Host library versions older than any Lemonade build needs. Reported through
/// the `rocm/e2e-test-hooks` override, so the refusal is reached on a host
/// whose real glibc and OpenSSL are new.
const OLD_HOST_GLIBC: &str = "2.17";
const OLD_HOST_OPENSSL: &str = "1.1.1";

#[given("this host's glibc and OpenSSL are older than the pinned Lemonade server needs")]
async fn host_libraries_too_old(world: &mut E2eWorld) {
    world
        .command_env
        .push(("ROCM_E2E_LEMONADE_HOST_GLIBC", OLD_HOST_GLIBC.into()));
    world
        .command_env
        .push(("ROCM_E2E_LEMONADE_HOST_OPENSSL", OLD_HOST_OPENSSL.into()));
}

/// The refusal names, for each library, the version needed and the version
/// the host has, says nothing was downloaded, and that is true on disk.
#[then("the Lemonade install is refused for the host's libraries before anything is downloaded")]
async fn assert_lemonade_install_refused_for_host_libraries(world: &mut E2eWorld) {
    let output = combined_output(world);
    // Without the hook the engine sees this host's real libraries and starts a
    // real install. Name that cause rather than a baffling mismatch.
    assert!(
        output.contains("this host's libraries are too old for the Lemonade "),
        "no host-library refusal: the binary under test was probably built without \
         the `rocm/e2e-test-hooks` feature, which `cargo xtask e2e` enables:\n{output}"
    );
    assert_ne!(
        world.cli_rc,
        Some(0),
        "the install must fail on a host too old for the pinned Lemonade server:\n{output}"
    );
    for expected in [
        format!("or newer (this host has glibc {OLD_HOST_GLIBC})"),
        format!("or newer (this host has OpenSSL {OLD_HOST_OPENSSL})"),
        "it needs glibc ".to_owned(),
        "Nothing was downloaded.".to_owned(),
    ] {
        assert!(
            output.contains(&expected),
            "missing `{expected}`:\n{output}"
        );
    }
    let tree = data_folder_tree(world);
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
        format!("into {}", tree.display()),
        "a download of up to about".to_owned(),
        "of disk space (1.0 MiB free on that drive)".to_owned(),
        "not enough free disk space to install Lemonade:".to_owned(),
        "Nothing was downloaded.".to_owned(),
    ] {
        assert!(
            output.contains(&expected),
            "missing `{expected}`:\n{output}"
        );
    }
}

/// Register the planted runtime: a removable install ROCm CLI made itself
/// (registry entry plus the in-tree copy `runtimes uninstall` checks before
/// deleting the folder), or, with `read_only`, an external one that uninstall
/// only unregisters.
fn plant_runtime(world: &E2eWorld, read_only: bool) {
    let install_root = runtime_install_root(world);
    std::fs::create_dir_all(&install_root).expect("failed to create runtime root");
    std::fs::write(install_root.join("payload.txt"), "payload")
        .expect("failed to write runtime payload");
    let manifest = serde_json::to_string_pretty(&serde_json::json!({
        "runtime_key": RUNTIME_KEY,
        "runtime_id": "therock-release:gfx1151",
        "channel": "release",
        "format": "tarball",
        "family": "gfx1151",
        "family_source": "manual",
        "version": "1.0.0",
        "install_root": install_root,
        "selected_artifact_url": format!("https://example.invalid/{RUNTIME_KEY}.tar.gz"),
        "read_only": read_only,
        "installed_at_unix_ms": 1_700_000_000_000u64,
    }))
    .expect("failed to serialize runtime manifest");
    let registry = data_dir(world).join("runtimes").join("registry");
    std::fs::create_dir_all(&registry).expect("failed to create registry");
    std::fs::write(registry.join(format!("{RUNTIME_KEY}.json")), &manifest)
        .expect("failed to write registry entry");
    if !read_only {
        std::fs::write(install_root.join(".rocm-cli-runtime.json"), &manifest)
            .expect("failed to write in-tree manifest");
    }
}

/// Write Lemonade's install record naming `runtime_dir`, whether or not
/// anything is there.
fn record_install_at(world: &E2eWorld, runtime_dir: &std::path::Path) {
    let record = install_record(world);
    std::fs::create_dir_all(record.parent().expect("record has a parent"))
        .expect("failed to create record dir");
    std::fs::write(
        &record,
        serde_json::to_string_pretty(&serde_json::json!({
            "env_id": "lemonade-embeddable-e2e",
            "version": "e2e",
            "runtime_dir": runtime_dir,
            "lemond": runtime_dir.join(lemond_name()),
            "lemonade": runtime_dir.join("lemonade"),
            "backend_recipe": "llamacpp",
            "backend_name": "rocm",
            "installed_at_unix_ms": 0,
        }))
        .expect("failed to serialize install record"),
    )
    .expect("failed to write install record");
}

#[given("Lemonade was installed in the data folder and again inside a ROCm install")]
async fn plant_two_lemonade_installs(world: &mut E2eWorld) {
    plant_runtime(world, false);
    // `rocm engines install lemonade` before any runtime, then `rocm serve`
    // with the runtime active: one tree each, the record naming the second.
    plant_lemonade_tree(&data_folder_tree(world), DATA_FOLDER_TREE_BYTES);
    plant_lemonade_tree(&runtime_tree(world), RUNTIME_TREE_BYTES);
    record_install_at(world, &runtime_tree(world).join("runtime"));
}

#[given("the ROCm install that held Lemonade is gone and only the data folder's copy is left")]
async fn plant_surviving_data_folder_copy(world: &mut E2eWorld) {
    // The record still names the removed runtime's folder, as it does after a
    // runtime was uninstalled by a build that left the record behind.
    plant_lemonade_tree(&data_folder_tree(world), DATA_FOLDER_TREE_BYTES);
    record_install_at(world, &runtime_tree(world).join("runtime"));
}

#[given("a read-only ROCm install holds a Lemonade copy")]
async fn plant_lemonade_in_read_only_runtime(world: &mut E2eWorld) {
    plant_runtime(world, true);
    plant_lemonade_tree(&runtime_tree(world), RUNTIME_TREE_BYTES);
    record_install_at(world, &runtime_tree(world).join("runtime"));
}

/// The report row for the Lemonade tree at `tree`, and the lines under it.
fn lemonade_row<'a>(report: &'a str, tree: &std::path::Path) -> Vec<&'a str> {
    let lines: Vec<&str> = report.lines().collect();
    let path = format!("({})", tree.display());
    let at = lines
        .iter()
        .position(|line| line.trim_start().starts_with("- lemonade (") && line.ends_with(&path))
        .unwrap_or_else(|| panic!("no Lemonade row for {}:\n{report}", tree.display()));
    lines[at..]
        .iter()
        .take_while(|line| {
            *line == &lines[at]
                || line.trim_start().starts_with("status:")
                || line.trim_start().starts_with("note:")
        })
        .copied()
        .collect()
}

#[then("the report lists both Lemonade installs with their sizes")]
async fn assert_both_lemonade_rows(world: &mut E2eWorld) {
    let report = stdout(world).to_owned();
    let data_row = lemonade_row(&report, &data_folder_tree(world));
    assert!(
        data_row[0].contains("- lemonade (data folder): 3.0 MiB"),
        "the data folder's copy must be listed with its size:\n{report}"
    );
    let runtime_row = lemonade_row(&report, &runtime_tree(world));
    assert!(
        runtime_row[0].contains(&format!(
            "- lemonade (inside ROCm install {RUNTIME_KEY}): 5.0 MiB"
        )),
        "the runtime's copy must be listed with its size:\n{report}"
    );
    assert!(
        runtime_row
            .iter()
            .any(|line| line.trim() == "status: in use"),
        "the copy the record points at is the one in use:\n{report}"
    );
}

#[then("the report flags the data folder's Lemonade install as unused")]
async fn assert_data_folder_copy_unused(world: &mut E2eWorld) {
    let report = stdout(world).to_owned();
    let data_row = lemonade_row(&report, &data_folder_tree(world));
    assert!(
        data_row
            .iter()
            .any(|line| line.trim()
                == "status: not used right now; Lemonade runs from another install"),
        "the data folder's copy must be flagged unused:\n{report}"
    );
    assert!(
        data_row
            .iter()
            .any(|line| line.trim_start().starts_with("note: reclaimable")),
        "an unused copy must say its space can be reclaimed:\n{report}"
    );
    // Reporting only: the tree is still on disk after the report.
    assert!(data_folder_tree(world).join("runtime").exists());
}

#[then("the report does not call the surviving Lemonade copy unused or reclaimable")]
async fn assert_survivor_not_called_reclaimable(world: &mut E2eWorld) {
    let report = stdout(world).to_owned();
    let row = lemonade_row(&report, &data_folder_tree(world));
    assert!(
        row.iter().any(|line| line.trim()
            == "status: not known to be used; the install record names a folder with no install"),
        "a record naming a folder with no install leaves no copy known to be in use:\n{report}"
    );
    assert!(
        row.iter()
            .all(|line| !line.contains("reclaimable") || line.contains("not marked reclaimable")),
        "the only working copy must not be offered for deletion:\n{report}"
    );
    assert!(
        data_folder_tree(world).join("runtime").exists(),
        "reporting must not delete anything"
    );
}

#[then("the report does not say that copy is removed with its ROCm install")]
async fn assert_read_only_copy_not_removed_with_runtime(world: &mut E2eWorld) {
    let report = stdout(world).to_owned();
    let row = lemonade_row(&report, &runtime_tree(world));
    assert!(
        row.iter().all(|line| !line.contains("removed with it")),
        "uninstalling an external ROCm install only unregisters it:\n{report}"
    );
    assert!(
        row.iter()
            .any(|line| line
                .contains("does not delete that install's folder (external or imported)")),
        "the row must say why the copy outlives an uninstall:\n{report}"
    );
}

#[then("that Lemonade copy is still on disk")]
async fn assert_lemonade_copy_still_on_disk(world: &mut E2eWorld) {
    assert!(
        runtime_tree(world).join("runtime").exists(),
        "uninstalling an external ROCm install must leave its Lemonade copy"
    );
}

#[when("the user uninstalls the ROCm install holding Lemonade")]
async fn user_uninstalls_runtime_holding_lemonade(world: &mut E2eWorld) {
    let stdout = crate::run_rocm_ok(world, &["runtimes", "uninstall", RUNTIME_KEY, "--yes"]);
    world.cli_output = Some(stdout);
}

#[then("the CLI says Lemonade's install was removed with that ROCm install")]
async fn assert_lemonade_removed_with_runtime(world: &mut E2eWorld) {
    let output = stdout(world).to_owned();
    let runtime_dir = runtime_tree(world).join("runtime");
    assert!(
        output.contains(&format!(
            "its install at {} is inside this runtime folder and would be removed with it",
            runtime_dir.display()
        )),
        "the plan must warn that Lemonade goes with the runtime:\n{output}"
    );
    assert!(
        output.contains(&format!(
            "its install at {} was inside this runtime and was removed with it",
            runtime_dir.display()
        )),
        "the result must say Lemonade was removed with the runtime:\n{output}"
    );
    assert!(
        !runtime_install_root(world).exists(),
        "the runtime folder, Lemonade tree included, must be gone:\n{output}"
    );
}

#[then("Lemonade is no longer recorded as installed")]
async fn assert_lemonade_record_cleared(world: &mut E2eWorld) {
    assert!(
        !install_record(world).exists(),
        "the record still names a deleted tree"
    );
    // The CLI's own view agrees: the remaining copy is no longer called unused
    // (that needs a record pointing elsewhere), and the deleted one is not listed.
    let report = crate::run_rocm_ok(world, &["storage", "report"]);
    let data_row = lemonade_row(&report, &data_folder_tree(world));
    assert!(
        data_row
            .iter()
            .any(|line| line.trim() == "status: not recorded as installed"),
        "with no record, the remaining copy is not recorded as installed:\n{report}"
    );
    assert!(
        !report.contains(&runtime_tree(world).display().to_string()),
        "the removed copy must not be listed any more:\n{report}"
    );
}
