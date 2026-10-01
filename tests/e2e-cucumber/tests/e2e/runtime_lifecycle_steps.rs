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
/// The runtime key recorded in the isolated `config.json`.
///
/// The marker sibling above cannot be used by the scenario that breaks the
/// marker on purpose. The config is the right file to read there anyway: it is
/// written first, so it is the half that would be left naming the new runtime
/// if the restore did not run.
fn config_active_runtime_key(world: &E2eWorld) -> Option<String> {
    let config_dir = world
        .isolate_env()
        .into_iter()
        .find(|(key, _)| *key == "ROCM_CLI_CONFIG_DIR")
        .map(|(_, value)| PathBuf::from(value))
        .expect("isolate_env did not set ROCM_CLI_CONFIG_DIR");
    let config = config_dir.join("config.json");
    let text = std::fs::read_to_string(&config)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", config.display()));
    let value: serde_json::Value = serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("{} is not valid JSON: {error}", config.display()));
    value
        .get("active_runtime_key")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}

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

#[given("two registered runtimes and a local server recording no runtime")]
async fn two_runtimes_and_service_without_runtime(world: &mut E2eWorld) {
    plant_runtime(world, FIRST_KEY, "gfx942");
    plant_runtime(world, SECOND_KEY, "gfx1100");
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let services = root.path().join("data").join("services");
    // `runtime_id: None` is the whole setup — a record that predates runtime
    // pinning and so says nothing about which runtime its server loaded.
    write_service_record_with(
        &services,
        SERVICE_MODEL,
        SERVICE_PORT,
        ServiceRecordOptions {
            status: "starting",
            startup_phase: Some("loading"),
            supervisor_pid: std::process::id(),
            ..ServiceRecordOptions::default()
        },
    );
}

/// Make the forward marker write fail, by planting a DIRECTORY at the marker
/// path.
///
/// Same reasoning as [`services_cannot_be_read`]: permissions are no obstacle
/// to a root CI lane, while `create_new` against an existing directory fails on
/// both supported platforms. Planted AFTER the first activation, so the marker
/// it replaces was written normally.
#[given("the active runtime marker cannot be written")]
async fn marker_cannot_be_written(world: &mut E2eWorld) {
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let marker = root
        .path()
        .join("data")
        .join("runtimes")
        .join("active.json");
    std::fs::remove_file(&marker)
        .unwrap_or_else(|error| panic!("failed to remove {}: {error}", marker.display()));
    std::fs::create_dir_all(&marker)
        .unwrap_or_else(|error| panic!("failed to create {}: {error}", marker.display()));
}

#[given("the first runtime is active")]
async fn first_runtime_active(world: &mut E2eWorld) {
    crate::run_rocm_ok(world, &["runtimes", "activate", FIRST_KEY]);
}

/// Make `load_managed_services` fail, by planting a DIRECTORY where it expects a
/// service record.
///
/// It skips a services folder that is not a directory and treats a record it
/// cannot parse as absent, so neither of those reaches the error path. A `.json`
/// entry it must read and cannot is the one input that does: the extension gets
/// it past the filter, and `fs::read` on a directory fails on both supported
/// platforms without needing permissions this test cannot rely on having (CI
/// lanes run as root, where a read-only directory is no obstacle at all).
#[given("the local service records cannot be read")]
async fn services_cannot_be_read(world: &mut E2eWorld) {
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let unreadable = root
        .path()
        .join("data")
        .join("services")
        .join("unreadable.json");
    std::fs::create_dir_all(&unreadable)
        .unwrap_or_else(|error| panic!("failed to create {}: {error}", unreadable.display()));
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

#[given("the running service is recorded on a different runtime")]
async fn mark_service_on_other_runtime(world: &mut E2eWorld) {
    // Rewrite the service record and the engine state file so that the CLI's
    // `load_managed_services` (which refreshes the record from the state file via
    // `refresh_from_engine_state`) sees the service as stale when the current
    // runtime is activated with `--restart-services --yes`.
    mark_services_on_other_runtime(world);
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

/// Same command as [`activate_second`], under the name a refusal scenario needs:
/// "activates" claims an outcome the scenario is there to deny.
#[when("the user tries to activate the second runtime")]
async fn try_activate_second(world: &mut E2eWorld) {
    activate_second(world).await;
}

#[when("the user tries to activate the first runtime restarting services without confirming")]
async fn activate_restart_services_without_yes(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &["runtimes", "activate", FIRST_KEY, "--restart-services"],
    );
    record(world, stdout, stderr, rc);
}

#[when("the user activates the current runtime restarting services with confirmation")]
async fn activate_current_restart_services(world: &mut E2eWorld) {
    let key = active_runtime_key(world);
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &["runtimes", "activate", &key, "--restart-services", "--yes"],
    );
    record(world, stdout, stderr, rc);
}

#[when("the user tries to roll back restarting services without confirming")]
async fn rollback_restart_services_without_yes(world: &mut E2eWorld) {
    let (stdout, stderr, rc) =
        crate::run_rocm(world, &["runtimes", "rollback", "--restart-services"]);
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

#[then("the activation exits 0 and names the service under services_restarted")]
async fn activation_names_restarted_service(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("  services_restarted: 1"),
        "expected one service reported restarted under services_restarted, got:\n{out}"
    );
}

#[then("the restarted service list is empty for services_on_previous_runtime")]
async fn activation_previous_runtime_count_is_zero(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("  services_on_previous_runtime: 0"),
        "expected zero services left on the previous runtime after restart, got:\n{out}"
    );
}

#[then("the model endpoint responds after the restart")]
async fn model_endpoint_responds_after_restart(world: &mut E2eWorld) {
    let model = world
        .model_name
        .as_deref()
        .expect("no model name recorded by setup_gpu_model")
        .to_owned();
    let endpoint = world
        .endpoint
        .as_deref()
        .expect("no endpoint recorded by setup_gpu_model")
        .to_owned();
    // The endpoint itself, not `rocm services list`: the list prints the record
    // whatever state the process is in, so it stays green for a server the
    // restart took down — which is the one outcome this scenario exists to rule
    // out. Waiting on the same `/v1/models` signal every serve scenario waits on
    // asserts the respawned engine actually reloaded the model and is serving it.
    let models_url = format!("{endpoint}/models");
    let ready = crate::e2e::serving_steps::model_is_ready(
        &models_url,
        Some(crate::e2e::serving_steps::ready_substr_for(&model)),
        crate::e2e::serving_steps::serve_timeout_for(world),
    )
    .await;
    let (services, _, _) = crate::run_rocm(world, &["services", "list"]);
    assert!(
        ready,
        "{models_url} never served {model} after the restart moved it onto the \
         newly active runtime; services list:\n{services}"
    );
}

#[then("the activation is refused and the first runtime is still active")]
async fn activation_refused_leaves_runtime_alone(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    assert!(
        rc != 0,
        "reading the services is what fails, and it happens before either \
         write, so the activation must be refused outright:\n{}",
        combined(world)
    );
    // Both files, not the printed output. The marker is what the next serve
    // resolves through, but asserting it alone would not notice the failure this
    // ordering exists to prevent: move the services read below the config write
    // and the config names the new runtime while the marker still names the old
    // one, which is precisely the half-applied state — and a marker-only check
    // stays green through it.
    assert_eq!(
        active_runtime_key(world),
        FIRST_KEY,
        "a refused activation must leave the previous runtime fully in place — \
         a marker naming the new runtime while the switch failed is exactly the \
         half-applied state this path exists to prevent:\n{}",
        combined(world)
    );
    assert_eq!(
        config_active_runtime_key(world).as_deref(),
        Some(FIRST_KEY),
        "the services are read before either write, so a refusal must leave the \
         config untouched too — a config naming the new runtime while the marker \
         names the old one is the pair disagreeing:\n{}",
        combined(world)
    );
}

#[then("the CLI refuses the restart naming rollback and the second runtime stays active")]
async fn rollback_restart_refused_without_yes(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    assert!(rc != 0, "expected refusal, got rc=0:\n{}", combined(world));
    // The `rollback` form, not the `activate` one. The hint is built from a
    // per-command string, so naming the wrong command is the one error it can
    // make — and re-running the `activate` form here would switch the runtime
    // rather than restore it.
    assert!(
        combined(world).contains("Try: rocm runtimes rollback --restart-services --yes"),
        "expected the refusal to name the command the user actually ran, got:\n{}",
        combined(world)
    );
    // Refused before the switch: the guard runs above `rollback_runtime`, and
    // moving it below would roll back and then decline the restarts.
    let active = active_runtime_key(world);
    assert_eq!(
        active,
        SECOND_KEY,
        "a refused --restart-services rollback must leave the active runtime \
         where it was:\n{}",
        combined(world)
    );
}

#[then("the first runtime is still the rollback target")]
async fn first_is_still_the_rollback_target(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains(&format!("changed_from_runtime_key: {FIRST_KEY}")),
        "re-activating the runtime that is already active must keep the \
         rollback target — the report's own note tells the user to run exactly \
         that command to move the servers left behind, one line above the \
         `rocm runtimes rollback` hint:\n{out}"
    );
}

#[then("rolling back still reaches the first runtime")]
async fn rollback_after_reactivation_reaches_first(world: &mut E2eWorld) {
    // Read back through the command rather than the report: a target that is
    // printed but not usable would still fail the user.
    let (stdout, stderr, rc) = crate::run_rocm(world, &["runtimes", "rollback"]);
    record(world, stdout, stderr, rc);
    let out = ok_output(world);
    assert!(
        out.contains(&format!("runtime_key: {FIRST_KEY}")),
        "expected rollback to reach {FIRST_KEY}, got:\n{}",
        combined(world)
    );
}

#[then("re-activating that runtime reports the service already on it")]
async fn reactivating_reports_the_service_already_on_the_runtime(world: &mut E2eWorld) {
    // The endpoint check above proves a server is serving; it cannot prove WHICH
    // runtime that server loaded, and the failure this path exists to prevent is
    // silent — `restart_service_onto_runtime` re-pins before restarting because
    // `restart_internal_managed_service` rebuilds argv from the record on disk,
    // so transposing the two brings the engine back up on the runtime it was
    // already using and still reports success.
    //
    // Re-running the activation is what distinguishes them: the reconciler reads
    // every record through `load_managed_services`, whose
    // `refresh_from_engine_state` adopts the runtime the ENGINE actually
    // launched with, overwriting the pin. So a respawn onto the old runtime
    // surfaces here as a service left behind, while a correct one is `Matches`
    // and reported nowhere.
    //
    // Not vacuous through a dead service: a service that is not live is skipped
    // by the reconciler and would also produce a zero count — but the preceding
    // step fails first if the endpoint is not serving.
    let key = active_runtime_key(world);
    let (stdout, stderr, rc) = crate::run_rocm(world, &["runtimes", "activate", &key]);
    record(world, stdout, stderr, rc);
    let out = ok_output(world);
    assert!(
        out.contains("services_on_previous_runtime: 0"),
        "the restarted server must be on the runtime that was just activated; a \
         non-zero count here is the respawn having come back on the runtime it \
         was already using:\n{}",
        combined(world)
    );
    assert!(
        !out.contains("services_with_unrecorded_runtime"),
        "the restart re-pins the record, so the runtime it names must be \
         readable back — an unrecorded runtime here means the respawn lost \
         it:\n{}",
        combined(world)
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
    // The runtime the user typed, not a literal `<runtime_key>`: the hint is
    // only useful if it can be pasted straight back.
    assert!(
        combined(world).contains(&format!(
            "Try: rocm runtimes activate {FIRST_KEY} --restart-services --yes"
        )),
        "expected the refusal to spell out the approved command with the \
         runtime the user asked for, got:\n{}",
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

/// Rewrite every managed-service record's `runtime_id` (and the matching field
/// in the engine state file) to a placeholder that does not match any installed
/// runtime key.
///
/// Two files carry the runtime key for each service:
///
/// - The service record JSON (`<data>/services/<id>.json`) holds `runtime_id`.
/// - The engine state file (`<data>/services/<id>.state.json`) holds
///   `requested_runtime_id`, which `refresh_from_engine_state` adopts into the
///   record on every `load_managed_services` call, overwriting `runtime_id`.
///
/// Both must be updated so the modification survives the refresh that
/// `reconcile_services_for_runtime` triggers before deciding which services are
/// stale. The service record JSON's `engine_state_path` field names the state
/// file's absolute path, so there is no need to guess it.
fn mark_services_on_other_runtime(world: &E2eWorld) {
    const PLACEHOLDER: &str = "other-runtime";
    let root = world.isolated_root.as_ref().expect("no isolated root");
    let services_dir = root.path().join("data").join("services");
    let entries = std::fs::read_dir(&services_dir)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", services_dir.display()));
    for entry in entries.flatten() {
        let path = entry.path();
        // Only process service record JSON files; skip engine state files
        // (*.state.json) and log files (*.log).
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.contains(".state."))
        {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(mut value) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        let Some(obj) = value.as_object_mut() else {
            continue;
        };
        // The state file is rewritten from the path the record names, so it is
        // read out of `obj` before the record is rewritten below.
        if let Some(state_path) = obj.get("engine_state_path").and_then(|v| v.as_str()) {
            rewrite_engine_state_runtime(std::path::Path::new(state_path), PLACEHOLDER);
        }
        // Rewrite the service record.
        obj.insert(
            "runtime_id".to_owned(),
            serde_json::Value::String(PLACEHOLDER.to_owned()),
        );
        if let Ok(json) = serde_json::to_string_pretty(&value) {
            std::fs::write(&path, json)
                .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
        }
    }
}

/// Point one engine state file's recorded runtime at `placeholder`.
///
/// Split out of `mark_services_on_other_runtime` rather than nested inside it:
/// the chain of reads is four deep, and `clippy::collapsible_if` rejects it as
/// written. Best effort — a state file that is missing, unparseable, or not a
/// JSON object is left alone, and the caller's record rewrite still stands. The
/// scenario's own assertions report the resulting mismatch with the CLI output
/// attached, which is more useful than a panic here that would hide it.
fn rewrite_engine_state_runtime(state_path: &std::path::Path, placeholder: &str) {
    let Ok(state_text) = std::fs::read_to_string(state_path) else {
        return;
    };
    let Ok(mut state_val) = serde_json::from_str::<serde_json::Value>(&state_text) else {
        return;
    };
    let Some(state_obj) = state_val.as_object_mut() else {
        return;
    };
    state_obj.insert(
        "requested_runtime_id".to_owned(),
        serde_json::Value::String(placeholder.to_owned()),
    );
    state_obj.insert(
        "runtime_id".to_owned(),
        serde_json::Value::String(placeholder.to_owned()),
    );
    if let Ok(json) = serde_json::to_string_pretty(&state_val) {
        let _ = std::fs::write(state_path, json);
    }
}

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

#[then("the activation is refused and the config still names the first runtime")]
async fn failed_marker_write_rolls_the_config_back(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    assert!(
        rc != 0,
        "the marker write failed, so the activation did not complete and must \
         not report success:\n{}",
        combined(world)
    );
    // The config is written before the marker. If the restore did not run it
    // would still name the second runtime while the marker names neither —
    // the half-applied state the snapshot exists to undo.
    assert_eq!(
        config_active_runtime_key(world).as_deref(),
        Some(FIRST_KEY),
        "a failed marker write must roll the config back to the runtime that \
         was active before it:\n{}",
        combined(world)
    );
}

#[then("the activation counts the server under services_with_unrecorded_runtime")]
async fn activation_counts_unrecorded_runtime_service(world: &mut E2eWorld) {
    // `ok_output`, not `combined`: an unrecorded runtime is reported, never a
    // reason to refuse, so the activation must still succeed.
    let output = ok_output(world);
    assert!(
        output.contains("services_with_unrecorded_runtime: 1"),
        "a live server whose record names no runtime must be counted apart \
         rather than folded into the stale count:\n{output}"
    );
    assert!(
        output.contains("services_on_previous_runtime: 0"),
        "a server with no recorded runtime is not known to be on the previous \
         one, so it must not be counted there:\n{output}"
    );
    assert!(
        output.contains("recorded_runtime=<unset>"),
        "the entry must say the runtime is unset rather than naming one:\n{output}"
    );
}
