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
/// Two keys that differ only in letter case. Each is its own `<key>.json` in the
/// registry, so the pair can only coexist on a case-sensitive filesystem — which
/// is why the scenario using them carries `@requires-case-sensitive-fs`.
const LOWER_TWIN_KEY: &str = "release-tarball-gfx942";
const UPPER_TWIN_KEY: &str = "RELEASE-TARBALL-GFX942";
/// Matches neither twin exactly and both case-insensitively — the shape the
/// resolver used to settle silently by picking one.
const CASE_ONLY_SELECTOR: &str = "Release-Tarball-Gfx942";
const IMPORT_KEY: &str = "release-tarball-gfx1151";

/// Service id of the planted local server. Fixed by `write_service_record_with`,
/// which names both the record and its `service_id` `e2e-mock`; the activation
/// report prints it, so the assertion has to use the same literal.
/// The GPU scenario's server is launched by the CLI and has a generated id
/// instead, which is why its steps read ids back from the records.
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

    let registry = registry_dir(world);
    std::fs::create_dir_all(&registry).expect("failed to create registry dir");
    let manifest = runtime_manifest_json(key, family, &install_root);
    std::fs::write(registry.join(format!("{key}.json")), manifest)
        .expect("failed to write runtime manifest");
    install_root
}

/// The runtime registry under the scenario's isolated data dir — where the CLI
/// reads `<key>.json` manifests from. One definition, so a layout change cannot
/// leave the planting and the counting looking in different places.
fn registry_dir(world: &E2eWorld) -> PathBuf {
    world
        .isolated_root
        .as_ref()
        .expect("no isolated root")
        .path()
        .join("data")
        .join("runtimes")
        .join("registry")
}

/// The managed-service record folder (`<data>/services/`) under the scenario's
/// isolated data dir.
fn services_dir(world: &E2eWorld) -> PathBuf {
    world
        .isolated_root
        .as_ref()
        .expect("no isolated root")
        .path()
        .join("data")
        .join("services")
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
/// Three details make it count, and all three are load-bearing. `host` is the
/// one other thing a caller varies: `None` is loopback, as for any local server.
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
fn plant_live_service_on_runtime(
    world: &E2eWorld,
    recorded_runtime_key: &'static str,
    host: Option<&'static str>,
) {
    write_service_record_with(
        &services_dir(world),
        SERVICE_MODEL,
        SERVICE_PORT,
        ServiceRecordOptions {
            status: "starting",
            startup_phase: Some("loading"),
            supervisor_pid: std::process::id(),
            runtime_id: Some(recorded_runtime_key),
            host,
            ..ServiceRecordOptions::default()
        },
    );
}

/// The `service_id` of every service record in the isolated services folder.
///
/// For the GPU scenario, whose server is launched by the CLI under a generated
/// id the feature cannot name. Files that are not service records (logs, key
/// files, anything unparseable) are skipped.
fn recorded_service_ids(world: &E2eWorld) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(services_dir(world)) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("json"))
        .filter_map(|entry| std::fs::read_to_string(entry.path()).ok())
        .filter_map(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .filter_map(|value| {
            value
                .get("service_id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        })
        .collect()
}

/// A planted service record read back as JSON, to see what the CLI left on disk.
fn read_service_record(world: &E2eWorld, service_id: &str) -> serde_json::Value {
    let path = services_dir(world).join(format!("{service_id}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("{} is not valid JSON: {error}", path.display()))
}

/// The service ids named under one section of the activation report, in order.
///
/// `section` is the header without its leading indent or colon
/// (`services_restarted`). The entries are the `    - <id>` lines directly
/// beneath the header; the id is the first word of each, minus the `:` the
/// failure section puts after it. The header is matched with its colon, so
/// `services_restarted` does not pick up `services_restarted_not_yet_serving`.
fn ids_listed_under<'a>(report: &'a str, section: &str) -> Vec<&'a str> {
    let header = format!("  {section}:");
    report
        .lines()
        .skip_while(|line| !line.starts_with(&header))
        .skip(1)
        .map_while(|line| line.strip_prefix("    - "))
        .filter_map(|entry| entry.split_whitespace().next())
        .map(|id| id.trim_end_matches(':'))
        .collect()
}

/// The runtime key recorded in the isolated `config.json`.
///
/// The marker (see [`active_runtime_key`]) cannot be used by the scenario that
/// breaks the marker on purpose. The config is the right file to read there
/// anyway: it is written first, so it is the half that would be left naming the
/// new runtime if the restore did not run.
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

#[given("two registered runtimes whose keys differ only in letter case")]
async fn two_case_twin_runtimes(world: &mut E2eWorld) {
    // Distinct families only so each twin gets its own install root.
    plant_runtime(world, LOWER_TWIN_KEY, "gfx942");
    plant_runtime(world, UPPER_TWIN_KEY, "gfx1100");

    // The two keys are two registry files only on a case-sensitive filesystem.
    // `@requires-case-sensitive-fs` skips hosts whose temp root folds case, so
    // this guard should never fire — except that cucumber's own `-n`/`--tags`
    // selection replaces the suite's filter and with it every gate. Should the
    // twins collapse anyway, the second write overwrote the first, activation
    // would succeed on a single case-insensitive match, and the refusal step
    // would blame the resolver for a fixture that never existed. Counting the
    // entries proves the premise; testing for the lower-case file would not,
    // because on such a filesystem that path resolves to the upper-case one.
    let registry = registry_dir(world);
    let planted = std::fs::read_dir(&registry)
        .unwrap_or_else(|err| panic!("cannot read the registry at {}: {err}", registry.display()))
        .count();
    assert_eq!(
        planted,
        2,
        "the case twins collapsed into one registry entry in {}: this filesystem is \
         case-insensitive, so the @requires-case-sensitive-fs gate should have skipped \
         this scenario on this host (cucumber's -n/--tags selection bypasses the gates)",
        registry.display()
    );
}

#[given("two registered runtimes and a local server recorded on the first")]
async fn two_runtimes_and_service_on_first(world: &mut E2eWorld) {
    plant_runtime(world, FIRST_KEY, "gfx942");
    plant_runtime(world, SECOND_KEY, "gfx1100");
    // Planted BEFORE either activation, so the same record covers both halves of
    // the scenario: while the first runtime is the one being activated the
    // server is on it (nothing left behind), and activating the second leaves it
    // behind without anything about the server itself having changed.
    plant_live_service_on_runtime(world, FIRST_KEY, None);
}

#[given("two registered runtimes and a publicly bound local server recorded on the first")]
async fn two_runtimes_and_public_service_on_first(world: &mut E2eWorld) {
    plant_runtime(world, FIRST_KEY, "gfx942");
    plant_runtime(world, SECOND_KEY, "gfx1100");
    // `0.0.0.0` with no endpoint key file planted is the whole setup. Restarting
    // a public bind that has no key is refused before the stop, so the
    // activation's restart fails while the server is left up on the first
    // runtime — a failed restart that needs no engine to produce.
    plant_live_service_on_runtime(world, FIRST_KEY, Some("0.0.0.0"));
}

#[given("two registered runtimes and a local server recording no runtime")]
async fn two_runtimes_and_service_without_runtime(world: &mut E2eWorld) {
    plant_runtime(world, FIRST_KEY, "gfx942");
    plant_runtime(world, SECOND_KEY, "gfx1100");
    // `runtime_id: None` is the whole setup — a record that predates runtime
    // pinning and so says nothing about which runtime its server loaded.
    write_service_record_with(
        &services_dir(world),
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
/// to a root CI lane. The marker is written to a temp file and then published by
/// renaming it onto the marker path; the temp file is created without trouble,
/// but a rename onto a path occupied by a directory is refused on both supported
/// platforms. Planted AFTER the first activation, so the marker it replaces was
/// written normally.
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
    let unreadable = services_dir(world).join("unreadable.json");
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

#[when("the user activates the second runtime again")]
async fn activate_second_again(world: &mut E2eWorld) {
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

#[when("the user activates the second runtime restarting services with confirmation")]
async fn activate_second_restart_services(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &[
            "runtimes",
            "activate",
            SECOND_KEY,
            "--restart-services",
            "--yes",
        ],
    );
    record(world, stdout, stderr, rc);
}

/// `--yes` approves a restart, so on its own it has nothing to approve; clap
/// refuses it at parse time (`requires = "restart_services"`).
#[when("the user tries to activate the first runtime with only --yes")]
async fn activate_first_with_only_yes(world: &mut E2eWorld) {
    let (stdout, stderr, rc) =
        crate::run_rocm(world, &["runtimes", "activate", FIRST_KEY, "--yes"]);
    record(world, stdout, stderr, rc);
}

#[when("the user tries to roll back with only --yes")]
async fn rollback_with_only_yes(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["runtimes", "rollback", "--yes"]);
    record(world, stdout, stderr, rc);
}

/// Re-runs the plain activation of whichever runtime is active now: the one
/// invocation that reads every server's record back without moving the runtime.
///
/// The endpoint check in the GPU scenario proves a server is serving; it cannot
/// prove WHICH runtime that server loaded. The reconciler refreshes every record
/// from the engine state before classifying it, and `refresh_from_engine_state`
/// adopts the runtime the ENGINE actually launched with, overwriting the pin the
/// restart wrote. So a respawn that came back on a runtime that does not resolve
/// to the activated one is named as left behind by this report, and one that came
/// back recording no runtime is named under `services_with_unrecorded_runtime` —
/// neither of which the restart's own `services_restarted: 1` line, written
/// before the record is re-read, can tell apart from a clean move. The
/// pin-before-restart order is not proven here; see the scenario's comment.
///
/// Not vacuous through a dead service: a service that is not live is skipped by
/// the reconciler and would also read as zero, but the scenario's endpoint check
/// fails first if the server is not serving.
#[when("the user re-activates that runtime")]
async fn reactivate_current_runtime(world: &mut E2eWorld) {
    let key = active_runtime_key(world);
    let (stdout, stderr, rc) = crate::run_rocm(world, &["runtimes", "activate", &key]);
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

#[when("the user activates a runtime with a selector that matches both only by letter case")]
async fn activate_by_case_only_selector(world: &mut E2eWorld) {
    let (stdout, stderr, rc) =
        crate::run_rocm(world, &["runtimes", "activate", CASE_ONLY_SELECTOR]);
    record(world, stdout, stderr, rc);
}

#[when("the user activates one of them by its exact key")]
async fn activate_upper_twin_exactly(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["runtimes", "activate", UPPER_TWIN_KEY]);
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

#[then("the report says the runtime did not change")]
async fn report_says_runtime_unchanged(world: &mut E2eWorld) {
    let out = ok_output(world);
    // The rollback target is retained, so the report could be tempted to name it
    // here; it must not, because the user has not left the runtime they are on.
    assert!(
        out.contains("  changed_from_runtime_key: <unchanged>"),
        "re-selecting the runtime already in use changed nothing, and the report \
         must say so, got:\n{out}"
    );
    assert!(
        !out.contains(&format!("changed_from_runtime_key: {FIRST_KEY}")),
        "the report must not claim the user came from {FIRST_KEY}, got:\n{out}"
    );
}

#[then("the second runtime is still the one in use")]
async fn second_still_active(world: &mut E2eWorld) {
    let listing = crate::run_rocm_ok(world, &["runtimes", "list"]);
    assert!(
        listing.contains(&format!("* {SECOND_KEY}")),
        "expected {SECOND_KEY} to still be the runtime in use, got:\n{listing}"
    );
}

#[then("the first runtime is still the rollback target")]
async fn first_still_rollback_target(world: &mut E2eWorld) {
    let listing = crate::run_rocm_ok(world, &["runtimes", "list"]);
    assert!(
        listing.contains(&format!("- {FIRST_KEY}")),
        "re-selecting the runtime already in use is a no-op for the user, so it must not \
         discard the rollback target the previous activation promised; expected {FIRST_KEY} \
         still marked as the rollback target, got:\n{listing}"
    );
}

#[then("its registry entry is removed")]
async fn registry_removed(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("runtime removed") && out.contains("registry_removed:"),
        "expected the registry entry removed, got:\n{out}"
    );
    let entry = registry_dir(world).join(format!("{FIRST_KEY}.json"));
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

#[then("the CLI refuses and names both runtimes")]
async fn refuses_naming_both_twins(world: &mut E2eWorld) {
    let out = combined(world);
    assert_ne!(
        world.cli_rc,
        Some(0),
        "a selector matching two runtimes only by letter case must not succeed:\n{out}"
    );
    assert!(
        out.contains("differ only in letter case"),
        "expected the case-ambiguity refusal, got:\n{out}"
    );
    // The advice is "name one of them exactly", so both exact keys must be on
    // screen for the user to follow it.
    for key in [LOWER_TWIN_KEY, UPPER_TWIN_KEY] {
        assert!(
            out.contains(key),
            "the refusal must name `{key}` so the user can pick it exactly:\n{out}"
        );
    }
}

/// The refusal has to leave the registry as it was. Read back through
/// `runtimes list` rather than trusting the error alone: a refusal printed after
/// a partial activation would still read as a refusal.
#[then("no runtime is active")]
async fn no_runtime_active(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["runtimes", "list"]);
    assert_eq!(rc, 0, "runtimes list failed:\n{stdout}\n{stderr}");
    assert!(
        !stdout
            .lines()
            .any(|line| line.trim_start().starts_with("* ")),
        "the refused activation must not have activated anything:\n{stdout}"
    );
}

/// The remediation the refusal names actually works: the exact key activates
/// that runtime, and only that one.
#[then("that exact runtime becomes active")]
async fn exact_twin_active(world: &mut E2eWorld) {
    let _ = ok_output(world);
    let (stdout, stderr, rc) = crate::run_rocm(world, &["runtimes", "list"]);
    assert_eq!(rc, 0, "runtimes list failed:\n{stdout}\n{stderr}");
    assert!(
        stdout.contains(&format!("* {UPPER_TWIN_KEY}")),
        "expected {UPPER_TWIN_KEY} active after naming it exactly:\n{stdout}"
    );
    assert!(
        !stdout.contains(&format!("* {LOWER_TWIN_KEY}")),
        "its case twin must not be the active one:\n{stdout}"
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
    let entry = registry_dir(world).join(format!("{FIRST_KEY}.json"));
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
    let entry = registry_dir(world).join(format!("{FIRST_KEY}.json"));
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
    // there is nothing on the previous runtime" has to be distinguishable from a
    // report that never looked.
    assert!(
        out.contains("  services_on_previous_runtime: 0"),
        "every live server is on the runtime being activated, so nothing is \
         left behind and the count must be 0, got:\n{out}"
    );
    assert!(
        ids_listed_under(&out, "services_on_previous_runtime").is_empty(),
        "a server already on the activated runtime must not be listed as left \
         behind, got:\n{out}"
    );
}

#[then("the activation reports no local server with an unrecorded runtime")]
async fn activation_reports_no_unrecorded_services(world: &mut E2eWorld) {
    let out = ok_output(world);
    // The line is only printed when a live server's record names no runtime, so
    // its absence is the claim.
    assert!(
        !out.contains("services_with_unrecorded_runtime"),
        "a restart re-pins the record, so the runtime it names must be readable \
         back — an unrecorded runtime here means the respawn lost it:\n{out}"
    );
}

#[then("the activation names the local server left on the first runtime")]
async fn activation_names_stale_service(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("  services_on_previous_runtime: 1"),
        "one live server still records {FIRST_KEY}, got:\n{out}"
    );
    // Not just a count, but WHICH server, on WHICH engine, still serving on
    // WHICH runtime — none of which a fixed note could say.
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

#[then("the activation does not print a fixed note about running services")]
async fn activation_drops_the_old_fixed_note(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        !out.contains("running services keep their recorded runtime"),
        "a fixed note is printed whether or not any server exists and says \
         nothing about real state; the report must come from the service \
         records alone, got:\n{out}"
    );
}

#[then("the activation exits 0 and reports the service as restarted")]
async fn activation_names_restarted_service(world: &mut E2eWorld) {
    let out = ok_output(world);
    // The respawn is what this step asserts. Whether the server had already
    // answered an inference request when the restart stopped waiting depends on
    // how long the model takes to load, so a server still loading is listed under
    // `services_restarted_not_yet_serving` rather than `services_restarted`; the
    // next steps are what wait for it to serve.
    let confirmed = ids_listed_under(&out, "services_restarted");
    let loading = ids_listed_under(&out, "services_restarted_not_yet_serving");
    let listed: Vec<&str> = confirmed.into_iter().chain(loading).collect();
    // The count alone would be satisfied by any id. The CLI generated this
    // server's id, so it is read back from the service records and the one entry
    // listed has to be that one.
    let recorded = recorded_service_ids(world);
    assert!(
        matches!(listed.as_slice(), [id] if recorded.iter().any(|known| known.as_str() == *id)),
        "expected the one restarted entry to be the recorded service \
         (recorded: {recorded:?}, listed: {listed:?}), got:\n{out}"
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

#[then(
    "the activation is refused, saying the switch was not made, and the first runtime is still active"
)]
async fn activation_refused_leaves_runtime_alone(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    assert!(
        rc != 0,
        "reading the services is what fails, and it happens before either \
         write, so the activation must be refused outright:\n{}",
        combined(world)
    );
    // The refusal has to tell the user what did not happen. Without the context
    // the bare read error ("failed to read ...unreadable.json") says nothing about
    // whether the switch was made; the files below are what prove it was not.
    let expected = format!(
        "the local server records could not be read, so the switch to `{SECOND_KEY}` was \
         not made and the previously active runtime is unchanged"
    );
    assert!(
        combined(world).contains(&expected),
        "expected the refusal to say the switch was not made, got:\n{}",
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
/// - The engine state file (`<data>/engines/<engine>/state/<id>.json`) holds
///   `requested_runtime_id`, which `refresh_from_engine_state` adopts into the
///   record on every `load_managed_services` call, overwriting `runtime_id`.
///
/// Both must be updated so the modification survives the refresh that
/// `reconcile_services_for_runtime` triggers before deciding which services are
/// stale. The service record JSON's `engine_state_path` field names the state
/// file's absolute path, so there is no need to guess it.
fn mark_services_on_other_runtime(world: &E2eWorld) {
    const PLACEHOLDER: &str = "other-runtime";
    let folder = services_dir(world);
    let entries = std::fs::read_dir(&folder)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", folder.display()));
    for entry in entries.flatten() {
        let path = entry.path();
        // Only JSON files can be service records; skip logs and key files. The
        // engine state files live outside this folder, so a `service_id` is what
        // tells a record from any other JSON that happens to be here.
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
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
        if !obj.contains_key("service_id") {
            continue;
        }
        // The state file is rewritten from the path the record names, so it is
        // read out of `obj` before the record is rewritten below.
        if let Some(state_path) = obj.get("engine_state_path").and_then(|v| v.as_str()) {
            rewrite_engine_state_runtime(Path::new(state_path), PLACEHOLDER);
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
fn rewrite_engine_state_runtime(state_path: &Path, placeholder: &str) {
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

#[then("the CLI refuses the bare --yes and names --restart-services")]
async fn bare_yes_refused(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    assert!(
        rc != 0,
        "`--yes` with nothing to approve must be refused, got rc=0:\n{}",
        combined(world)
    );
    assert!(
        combined(world).contains("--restart-services"),
        "the refusal must name the flag `--yes` depends on, got:\n{}",
        combined(world)
    );
    // The state the callers' next steps check is what proves nothing ran: a
    // refusal printed after a switch would read the same.
    assert!(
        !combined(world).contains("runtime activated")
            && !combined(world).contains("runtime rolled back"),
        "a refused command must not report a switch:\n{}",
        combined(world)
    );
}

#[then("the activation fails naming the server it could not restart")]
async fn activation_fails_naming_unrestarted_server(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    let out = world.cli_output.clone().unwrap_or_default();
    assert!(
        rc != 0,
        "a requested restart that did not happen must fail the command:\n{}",
        combined(world)
    );
    assert!(
        out.contains("  services_restart_failed: 1"),
        "expected the report to count one failed restart, got:\n{out}"
    );
    assert_eq!(
        ids_listed_under(&out, "services_restart_failed"),
        [SERVICE_ID],
        "the failed restart must name the server, got:\n{out}"
    );
    // The reason is the one the planted record triggers: a public bind with no
    // endpoint key is refused before anything is stopped.
    assert!(
        out.contains("has no endpoint API key"),
        "expected the report to carry the restart's own refusal, got:\n{out}"
    );
    // The closing error is where the exit code gets its explanation, and it must
    // not tell the user a live server is down.
    assert!(
        combined(world).contains(&format!(
            "These are still serving from the runtime they recorded, and stay counted under \
             services_on_previous_runtime: {SERVICE_ID}."
        )),
        "expected the error to say the server is still serving from its recorded \
         runtime, got:\n{}",
        combined(world)
    );
    assert!(
        !out.contains("  services_restarted:"),
        "nothing was restarted, so nothing may be listed as restarted, got:\n{out}"
    );
}

#[then("the server is still counted as left on the first runtime")]
async fn server_still_counted_on_first_runtime(world: &mut E2eWorld) {
    let out = world.cli_output.clone().unwrap_or_default();
    assert!(
        out.contains("  services_on_previous_runtime: 1"),
        "the refused restart left the server up on {FIRST_KEY}, so it is still \
         counted as left behind, got:\n{out}"
    );
    assert!(
        out.contains(&format!(
            "    - {SERVICE_ID} engine={SERVICE_ENGINE} recorded_runtime={FIRST_KEY}"
        )),
        "expected the server named with the runtime it is still on, got:\n{out}"
    );
    // The advice to add `--restart-services` is for a user who has not tried it.
    assert!(
        !out.contains("note: those keep serving on their recorded runtime"),
        "a restart was just attempted, so the report must not tell the user to \
         attempt it, got:\n{out}"
    );
}

#[then("the switch to the second runtime was kept")]
async fn switch_kept_despite_failed_restart(world: &mut E2eWorld) {
    // A restart failure does not undo the switch; both files must agree on it.
    assert_eq!(
        active_runtime_key(world),
        SECOND_KEY,
        "the runtime was switched before the restart was attempted:\n{}",
        combined(world)
    );
    assert_eq!(
        config_active_runtime_key(world).as_deref(),
        Some(SECOND_KEY),
        "the config and the marker must agree on the switch:\n{}",
        combined(world)
    );
}

#[then("the server record still names the first runtime")]
async fn server_record_still_names_first_runtime(world: &mut E2eWorld) {
    let record = read_service_record(world, SERVICE_ID);
    assert_eq!(
        record.get("runtime_id").and_then(serde_json::Value::as_str),
        Some(FIRST_KEY),
        "a server that did not move must not be described by a runtime it never \
         loaded:\n{record}"
    );
    assert_eq!(
        record.get("status").and_then(serde_json::Value::as_str),
        Some("starting"),
        "the restart was refused before the stop, so the record must still be \
         the live one it was:\n{record}"
    );
}
