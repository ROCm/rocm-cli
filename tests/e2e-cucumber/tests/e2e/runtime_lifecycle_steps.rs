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
//! binary (EAI-8072). Related EAI-7404.

use std::path::{Path, PathBuf};

use cucumber::{given, then, when};
use e2e_cucumber::mock_server::{ServiceRecordOptions, write_service_record_with};

use crate::E2eWorld;

// Runtime KEYS double as registry filenames (`<key>.json`), exactly as production
// does. Production derives the key by slugifying, so a real key is filename-safe on
// every OS; the fixtures must use the same shape. A raw `therock-release:gfx942`
// (the runtime_ID form) contains a `:`, which on Windows names an NTFS alternate
// data stream instead of a normal file — the planted runtime is then undiscoverable
// and activate/rollback/uninstall fail with "installed runtime not found". Keep the
// `:` form only in `runtime_id`, which is a manifest field value, never a filename.
const FIRST_KEY: &str = "release-tarball-gfx942";
const SECOND_KEY: &str = "release-tarball-gfx1100";
const IMPORT_KEY: &str = "release-tarball-gfx1151";

/// Service id of the planted local server. Fixed by `write_service_record_with`,
/// which names both the record and its `service_id` `e2e-mock`; the activation
/// report prints it, so the assertion has to use the same literal.
const SERVICE_ID: &str = "e2e-mock";
/// Engine of the planted record, also fixed by `write_service_record_with`.
/// Deliberately NOT `lemonade`: an engine that manages its own runtime records
/// an engine-private key and is never counted as left behind, so planting one
/// would assert nothing.
const SERVICE_ENGINE: &str = "vllm";
/// Model the planted record claims to serve. Never loaded — no engine runs — it
/// only has to be a plausible id, since the record is read, not served.
const SERVICE_MODEL: &str = "TestModel/E2E-1B";
/// Port recorded for the planted local server. Nothing ever listens on it and
/// nothing ever connects to it: the record is planted as `starting`, and the
/// CLI only probes the endpoint of a `ready`/`running` record. Keeping the
/// scenario off the network is why no mock HTTP server is needed here at all.
const SERVICE_PORT: u16 = 58_921;

/// Write a read-only `tarball` runtime manifest into the isolated registry and
/// create its `install_root` (a dir with a payload file) so it validates as usable.
/// Returns the install_root so a scenario can assert the folder's fate.
fn plant_runtime(world: &E2eWorld, key: &str, family: &str) -> PathBuf {
    assert!(
        key.chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-'),
        "runtime key must be production-style and safe as a registry filename: {key}"
    );
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
        // The human-facing identifier keeps the `therock-release:<family>` form (the
        // `:` is safe here — this is a field value, never a filename, unlike `key`).
        "runtime_id": format!("therock-release:{family}"),
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

/// Plant a managed-service record the CLI reads as a LIVE local server that was
/// launched against `recorded_runtime_key`, so activation has something real to
/// report on. Uses the shared record schema (`write_service_record_with`) rather
/// than a second hand-written JSON shape, so this cannot drift from what `rocm
/// serve --managed` writes.
///
/// Three details make it count, and all three are load-bearing:
///
/// - `runtime_id` holds a runtime KEY, not the `therock-release:<family>`
///   runtime_id form the manifest carries. The on-disk field is named
///   `runtime_id` for historical reasons, but every launch path resolves its
///   selector to an exact key before recording it, and the CLI compares it
///   against the key being activated — so a record holding the `:` form would
///   simply never match and every activation would report it stale.
/// - `supervisor_pid` is this test process, which is guaranteed alive. The CLI
///   overlays real process liveness on every record it loads and demotes one
///   with no live pid to `stopped`, which is not live and so is never reported.
///   (Teardown is unaffected: `stop_managed_services` skips the `e2e-mock`
///   record precisely because it has no real process behind it.)
/// - `starting` rather than `ready`: both are live as far as the report is
///   concerned, but only `ready`/`running` records get an HTTP readiness probe,
///   which here would reach for a port nothing serves. Planting mid-startup
///   keeps the scenario hermetic and off the network.
fn plant_live_service_on_runtime(world: &E2eWorld, recorded_runtime_key: &'static str) {
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let services = root.path().join("data").join("services");
    write_service_record_with(
        &services,
        SERVICE_MODEL,
        SERVICE_PORT,
        ServiceRecordOptions {
            status: "starting",
            startup_phase: Some("loading"),
            supervisor_pid: std::process::id(),
            runtime_id: Some(recorded_runtime_key),
            ..ServiceRecordOptions::default()
        },
    );
}

/// The runtime key recorded in the isolated active-runtime marker
/// (`<data>/runtimes/active.json`) — what the next serve would actually use.
///
/// Parsed as JSON rather than substring-matched on the file: once a second
/// runtime has been activated the marker also carries `previous_runtime_key`,
/// so a `contains` check would be satisfied by either key and could not tell a
/// refused switch from a completed one.
fn active_runtime_key(world: &E2eWorld) -> String {
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let marker = root
        .path()
        .join("data")
        .join("runtimes")
        .join("active.json");
    let text = std::fs::read_to_string(&marker)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", marker.display()));
    let value: serde_json::Value = serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("{} is not valid JSON: {error}", marker.display()));
    value
        .get("runtime_key")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_else(|| panic!("no runtime_key in {}:\n{text}", marker.display()))
        .to_owned()
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

#[given("two registered runtimes and a local server recorded on the first")]
async fn two_runtimes_and_service_on_first(world: &mut E2eWorld) {
    plant_runtime(world, FIRST_KEY, "gfx942");
    plant_runtime(world, SECOND_KEY, "gfx1100");
    // Planted BEFORE either activation, so the same record covers both halves of
    // the scenario: while the first runtime is the one being activated the
    // server is on it (nothing left behind), and activating the second leaves it
    // behind without anything about the server itself having changed.
    plant_live_service_on_runtime(world, FIRST_KEY);
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

#[when("the user tries to activate the first runtime restarting services without confirming")]
async fn activate_restart_services_without_yes(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &["runtimes", "activate", FIRST_KEY, "--restart-services"],
    );
    record(world, stdout, stderr, rc);
}

#[when("the user rolls back")]
async fn rollback(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["runtimes", "rollback"]);
    record(world, stdout, stderr, rc);
}

#[when("the user lists the registered runtimes")]
async fn list_runtimes(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["runtimes", "list"]);
    record(world, stdout, stderr, rc);
}

#[when("the user uninstalls that runtime")]
async fn uninstall(world: &mut E2eWorld) {
    let (stdout, stderr, rc) =
        crate::run_rocm(world, &["runtimes", "uninstall", FIRST_KEY, "--yes"]);
    record(world, stdout, stderr, rc);
}

#[when("the user tries to uninstall that runtime without confirming")]
async fn uninstall_without_yes(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["runtimes", "uninstall", FIRST_KEY]);
    record(world, stdout, stderr, rc);
}

#[when("the user dry-runs an uninstall of that runtime")]
async fn uninstall_dry_run(world: &mut E2eWorld) {
    let (stdout, stderr, rc) =
        crate::run_rocm(world, &["runtimes", "uninstall", FIRST_KEY, "--dry-run"]);
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
    assert!(
        !out.contains("rocm runtimes rollback"),
        "no previous runtime is recorded, so rollback would hard-error; \
         must not hint at a command that immediately fails:\n{out}"
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
    assert!(
        out.contains("next step: if this causes problems, run `rocm runtimes rollback`"),
        "a previous runtime is recorded, so the built binary should hint at rollback as a \
         recovery path, got:\n{out}"
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

#[then("the listing explains the active and rollback markers")]
async fn listing_explains_markers(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("legend: * = active, - = rollback target"),
        "expected the marker legend, got:\n{out}"
    );
}

#[then("the first runtime is marked active")]
async fn first_marked_active(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains(&format!("* {FIRST_KEY}")),
        "expected {FIRST_KEY} marked active, got:\n{out}"
    );
}

#[then("the second runtime is marked as the rollback target")]
async fn second_marked_rollback(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains(&format!("- {SECOND_KEY}")),
        "expected {SECOND_KEY} marked as rollback target, got:\n{out}"
    );
}

#[then("the CLI refuses and requires --yes")]
async fn uninstall_refused_without_yes(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    assert!(rc != 0, "expected refusal, got rc=0:\n{}", combined(world));
    assert!(
        combined(world).contains("requires --yes"),
        "expected a --yes-required error, got:\n{}",
        combined(world)
    );
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let entry = root
        .path()
        .join("data")
        .join("runtimes")
        .join("registry")
        .join(format!("{FIRST_KEY}.json"));
    assert!(
        entry.exists(),
        "registry entry must survive a refused uninstall: {}",
        entry.display()
    );
}

#[then("the dry run reports the plan without confirming or changing anything")]
async fn uninstall_dry_run_reports_plan(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("runtime uninstall plan") && out.contains("dry run: no changes made"),
        "expected a dry-run plan with no changes made, got:\n{out}"
    );
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let entry = root
        .path()
        .join("data")
        .join("runtimes")
        .join("registry")
        .join(format!("{FIRST_KEY}.json"));
    assert!(
        entry.exists(),
        "registry entry must survive a dry-run uninstall: {}",
        entry.display()
    );
}

#[then("the activation reports no local server left on a previous runtime")]
async fn activation_reports_no_stale_services(world: &mut E2eWorld) {
    let out = ok_output(world);
    // The count is printed even at zero, which is the whole point: "looked, and
    // there is nothing on the old runtime" has to be distinguishable from a
    // report that never looked — the failure mode of the fixed note this
    // replaced.
    assert!(
        out.contains("  services_on_previous_runtime: 0"),
        "the planted server records the runtime being activated, so nothing is \
         left behind and the count must be 0, got:\n{out}"
    );
    assert!(
        !out.contains(SERVICE_ID),
        "a server already on the activated runtime must not be listed as left \
         behind, got:\n{out}"
    );
}

#[then("the activation names the local server left on the first runtime")]
async fn activation_names_stale_service(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("  services_on_previous_runtime: 1"),
        "one live server still records {FIRST_KEY}, got:\n{out}"
    );
    // The regression in full: not just a count, but WHICH server, on WHICH
    // engine, still serving on WHICH runtime — none of which the old fixed note
    // could say.
    assert!(
        out.contains(&format!(
            "    - {SERVICE_ID} engine={SERVICE_ENGINE} recorded_runtime={FIRST_KEY}"
        )),
        "expected the left-behind server to be named with its engine and \
         recorded runtime, got:\n{out}"
    );
    assert!(
        out.contains("note: those keep serving on their recorded runtime until they are restarted"),
        "expected the note to follow the named server, got:\n{out}"
    );
}

#[then("the activation does not print the old fixed note about running services")]
async fn activation_drops_the_old_fixed_note(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        !out.contains("running services keep their recorded runtime"),
        "the fixed note was printed whether or not any server existed and said \
         nothing about real state; it must not come back alongside the \
         state-derived report, got:\n{out}"
    );
}

#[then("the CLI refuses the restart and the second runtime stays active")]
async fn restart_services_refused_without_yes(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    assert!(rc != 0, "expected refusal, got rc=0:\n{}", combined(world));
    assert!(
        combined(world).contains("requires --yes"),
        "expected a --yes-required error, got:\n{}",
        combined(world)
    );
    assert!(
        combined(world)
            .contains("Try: rocm runtimes activate <runtime_key> --restart-services --yes"),
        "expected the refusal to spell out the approved command, got:\n{}",
        combined(world)
    );
    // The refusal is checked before anything is written, so the switch itself
    // must not have happened: a user who is told "no" must not find the runtime
    // already changed under them, with only the restarts declined.
    let active = active_runtime_key(world);
    assert_eq!(
        active,
        SECOND_KEY,
        "a refused --restart-services activation must leave the previously \
         active runtime in place:\n{}",
        combined(world)
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
