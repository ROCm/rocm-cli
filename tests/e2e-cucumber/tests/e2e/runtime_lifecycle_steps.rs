// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for the runtime state machine: `runtimes activate/rollback/uninstall/
//! import`. Black-box against the isolated runtimes registry. A real SDK runtime
//! needs a multi-GiB download and a GPU family, so instead these plant READ-ONLY
//! `tarball` runtime manifests: the CLI validates a read-only tarball runtime by
//! only requiring its `install_root` to be a directory holding a non-dot payload
//! file — no python, no GPU, no download. That makes the whole state machine
//! exercisable on the mock lane. Contracts verified against the running Linux
//! binary (WL-502). Related EAI-7404.

use std::path::{Path, PathBuf};

use cucumber::{given, then, when};

use crate::E2eWorld;

const FIRST_KEY: &str = "therock-release:gfx942";
const SECOND_KEY: &str = "therock-release:gfx1100";
const IMPORT_KEY: &str = "therock-release:gfx1151";

/// Write a read-only `tarball` runtime manifest into the isolated registry and
/// create its `install_root` (a dir with a payload file) so it validates as usable.
/// Returns the install_root so a scenario can assert the folder's fate.
fn plant_runtime(world: &E2eWorld, key: &str, family: &str) -> PathBuf {
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let install_root = root.path().join(format!("runtime-{family}"));
    std::fs::create_dir_all(&install_root).expect("failed to create install root");
    std::fs::write(install_root.join("payload.txt"), "payload")
        .expect("failed to write runtime payload");

    let registry = root.path().join("data").join("runtimes").join("registry");
    std::fs::create_dir_all(&registry).expect("failed to create registry dir");
    let manifest = runtime_manifest_json(key, family, &install_root);
    std::fs::write(registry.join(format!("{key}.json")), manifest)
        .expect("failed to write runtime manifest");
    install_root
}

/// A minimal valid read-only tarball runtime manifest (matches the CLI's on-disk
/// schema). Written as plain JSON — black-box, not a typed import from the crates.
fn runtime_manifest_json(key: &str, family: &str, install_root: &Path) -> String {
    serde_json::to_string_pretty(&serde_json::json!({
        "runtime_key": key,
        "runtime_id": key,
        "channel": "release",
        "format": "tarball",
        "family": family,
        "family_source": "manual",
        "version": "1.0.0",
        "install_root": install_root,
        "selected_artifact_url": format!("https://example.invalid/{key}.tar.gz"),
        "read_only": true,
        "installed_at_unix_ms": 1_700_000_000_000u64,
    }))
    .expect("failed to serialize runtime manifest")
}

/// Path to an importable manifest file (not yet in the registry) for the import
/// scenario, with its install_root created so the import validates.
fn write_import_manifest(world: &E2eWorld) -> PathBuf {
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let install_root = root.path().join("runtime-import");
    std::fs::create_dir_all(&install_root).expect("failed to create import install root");
    std::fs::write(install_root.join("payload.txt"), "payload")
        .expect("failed to write import payload");
    let manifest_path = root.path().join("import-manifest.json");
    std::fs::write(
        &manifest_path,
        runtime_manifest_json(IMPORT_KEY, "gfx1151", &install_root),
    )
    .expect("failed to write import manifest");
    manifest_path
}

// ── Given ──────────────────────────────────────────────────────────

#[given("two registered runtimes and none active")]
async fn two_runtimes(world: &mut E2eWorld) {
    plant_runtime(world, FIRST_KEY, "gfx942");
    plant_runtime(world, SECOND_KEY, "gfx1100");
}

#[given("two registered runtimes with the second active after the first")]
async fn two_runtimes_second_active(world: &mut E2eWorld) {
    plant_runtime(world, FIRST_KEY, "gfx942");
    plant_runtime(world, SECOND_KEY, "gfx1100");
    // Activate first, then second, so `previous_runtime_key` records the first —
    // the state rollback must return to.
    crate::run_rocm_ok(world, &["runtimes", "activate", FIRST_KEY]);
    crate::run_rocm_ok(world, &["runtimes", "activate", SECOND_KEY]);
}

#[given("a registered read-only runtime")]
async fn one_readonly_runtime(world: &mut E2eWorld) {
    let install_root = plant_runtime(world, FIRST_KEY, "gfx942");
    // Stash the install_root path so the uninstall scenario can assert it survives.
    world.model_name = Some(install_root.to_string_lossy().into_owned());
}

#[given("a runtime manifest to import")]
async fn manifest_to_import(world: &mut E2eWorld) {
    let path = write_import_manifest(world);
    world.model_name = Some(path.to_string_lossy().into_owned());
}

// ── When ───────────────────────────────────────────────────────────

#[when("the user activates the first runtime")]
async fn activate_first(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["runtimes", "activate", FIRST_KEY]);
    record(world, stdout, stderr, rc);
}

#[when("the user activates the second runtime")]
async fn activate_second(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["runtimes", "activate", SECOND_KEY]);
    record(world, stdout, stderr, rc);
}

#[when("the user rolls back")]
async fn rollback(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["runtimes", "rollback"]);
    record(world, stdout, stderr, rc);
}

#[when("the user uninstalls that runtime")]
async fn uninstall(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["runtimes", "uninstall", FIRST_KEY]);
    record(world, stdout, stderr, rc);
}

#[when("the user imports the runtime")]
async fn import(world: &mut E2eWorld) {
    let path = world.model_name.clone().expect("no import manifest path");
    let (stdout, stderr, rc) = crate::run_rocm(world, &["runtimes", "import", &path]);
    record(world, stdout, stderr, rc);
}

#[when("the user imports the same runtime again")]
async fn import_again(world: &mut E2eWorld) {
    let path = world.model_name.clone().expect("no import manifest path");
    let (stdout, stderr, rc) = crate::run_rocm(world, &["runtimes", "import", &path]);
    record(world, stdout, stderr, rc);
}

#[when("the user imports it again allowing replacement")]
async fn import_replace(world: &mut E2eWorld) {
    let path = world.model_name.clone().expect("no import manifest path");
    let (stdout, stderr, rc) = crate::run_rocm(world, &["runtimes", "import", &path, "--replace"]);
    record(world, stdout, stderr, rc);
}

// ── Then ───────────────────────────────────────────────────────────

#[then("that runtime becomes active having changed from nothing")]
async fn active_changed_from_nothing(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("runtime activated") && out.contains(&format!("runtime_key: {FIRST_KEY}")),
        "expected {FIRST_KEY} activated, got:\n{out}"
    );
    assert!(
        out.contains("changed_from_runtime_key: <unset>"),
        "expected no previous runtime, got:\n{out}"
    );
}

#[then("that runtime becomes active having changed from the first")]
async fn active_changed_from_first(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("runtime activated") && out.contains(&format!("runtime_key: {SECOND_KEY}")),
        "expected {SECOND_KEY} activated, got:\n{out}"
    );
    assert!(
        out.contains(&format!("changed_from_runtime_key: {FIRST_KEY}")),
        "expected previous runtime {FIRST_KEY}, got:\n{out}"
    );
}

#[then("the first runtime is active again")]
async fn first_active_again(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("runtime rolled back") && out.contains(&format!("runtime_key: {FIRST_KEY}")),
        "expected rollback to {FIRST_KEY}, got:\n{out}"
    );
}

#[then("its registry entry is removed")]
async fn registry_removed(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("runtime removed") && out.contains("registry_removed:"),
        "expected the registry entry removed, got:\n{out}"
    );
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let entry = root
        .path()
        .join("data")
        .join("runtimes")
        .join("registry")
        .join(format!("{FIRST_KEY}.json"));
    assert!(
        !entry.exists(),
        "registry entry still present: {}",
        entry.display()
    );
}

#[then("its external folder is left in place")]
async fn folder_left(world: &mut E2eWorld) {
    let out = world.cli_output.clone().unwrap_or_default();
    assert!(
        out.contains("folder_removed: no")
            && out.contains("existing external runtime folder was left untouched"),
        "expected the external folder to be left, got:\n{out}"
    );
    let install_root = world
        .model_name
        .as_deref()
        .expect("no install root recorded");
    assert!(
        Path::new(install_root).is_dir(),
        "external runtime folder was removed: {install_root}"
    );
}

#[then("the runtime is registered as read-only")]
async fn imported_readonly(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("runtime imported") && out.contains("mode: read-only"),
        "expected a read-only import, got:\n{out}"
    );
}

#[then("the CLI refuses because it already exists")]
async fn import_duplicate_refused(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    assert!(rc != 0, "expected refusal, got rc=0:\n{}", combined(world));
    assert!(
        combined(world).contains("already exists") && combined(world).contains("--replace"),
        "expected a duplicate-registry error mentioning --replace, got:\n{}",
        combined(world)
    );
}

#[then("the import succeeds")]
async fn import_succeeds(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("runtime imported"),
        "expected the replace import to succeed, got:\n{out}"
    );
}

// ── Helpers ────────────────────────────────────────────────────────

fn record(world: &mut E2eWorld, stdout: String, stderr: String, rc: i32) {
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

fn combined(world: &E2eWorld) -> String {
    format!(
        "{}\n{}",
        world.cli_output.as_deref().unwrap_or(""),
        world.cli_stderr.as_deref().unwrap_or("")
    )
}

fn ok_output(world: &E2eWorld) -> String {
    let rc = world.cli_rc.expect("no command rc recorded");
    assert_eq!(rc, 0, "expected success, got rc={rc}:\n{}", combined(world));
    world.cli_output.clone().unwrap_or_default()
}
