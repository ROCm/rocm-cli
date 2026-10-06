// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Steps for `rocm update` (report only). Run with NO managed runtimes so the
//! report needs no network (with a runtime present, `update` reaches the TheRock
//! index to resolve the latest version). The report's update-feed status block is
//! host-invariant and is what pins the "distinguishes configured from
//! not-configured feeds" behaviour. Contracts verified against the running Linux
//! binary (EAI-8072). Mock lane.

use cucumber::{given, then, when};
use e2e_cucumber::loopback_http::LoopbackServer;

use crate::E2eWorld;

#[given("a machine with no managed runtimes")]
async fn no_managed_runtimes(_world: &mut E2eWorld) {
    // The World's isolated data dir starts with an empty runtimes registry, so
    // `update` has nothing to check against the network. No setup required.
}

#[when("the user checks for updates")]
async fn check_updates(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["update"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[when("the user checks for updates as machine-readable JSON")]
async fn check_updates_json(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["update", "--json"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[when("the user checks for updates as machine-readable JSON with a 5 second timeout")]
async fn check_updates_json_with_timeout(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["update", "--json", "--timeout-secs", "5"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the report shows there are no managed runtimes to update")]
async fn no_runtimes_to_update(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert!(
        out.contains("managed runtimes: none"),
        "expected 'managed runtimes: none', got:\n{out}"
    );
}

#[then("it reports each update feed's status, marking unpublished feeds as not configured")]
async fn reports_feed_status(world: &mut E2eWorld) {
    let out = ok_output(world);
    // The update_surfaces block reports one line per feed. Assert each feed's status
    // ON ITS OWN LINE, so a status attributed to the wrong feed fails — a check that
    // only looked for the substrings anywhere would pass even if `not_configured`
    // and `package_managed` were swapped between the cli and engines feeds. The CLI
    // feed is not published yet (the "not configured" side of the distinction);
    // engines and recipes report their own stable states.
    for (feed, status) in [
        ("cli:", "status=not_configured"),
        ("engines:", "status=package_managed"),
        ("model_recipes:", "status=built_in"),
    ] {
        let line = out
            .lines()
            .map(str::trim)
            .find(|line| line.starts_with(feed));
        match line {
            Some(line) => assert!(
                line.contains(status),
                "update feed {feed:?} did not report {status:?} on its own line; got {line:?}\n\nfull output:\n{out}"
            ),
            None => panic!("no update feed line for {feed:?} in:\n{out}"),
        }
    }
}

// Covers the JSON envelope shape only (single line, `runtimes: []`).
// Suppressing wheel-resolution progress output ahead of the JSON contract is
// a separate concern with no managed runtimes here to trigger it — that's
// covered by `render_update_json_installs_the_suppression_guard_around_resolution`
// in `apps/rocm/src/therock.rs`, which exercises a real progress_line call
// reachable during resolution.
#[then("the machine-readable check reports no runtimes to update")]
async fn json_reports_empty_runtimes(world: &mut E2eWorld) {
    let out = ok_output(world);
    let mut lines = out.lines();
    let line = lines
        .next()
        .unwrap_or_else(|| panic!("expected a line of JSON on stdout, got empty output"));
    assert!(
        lines.next().is_none(),
        "expected exactly one stdout line (JSON must not share stdout with other output), got:\n{out}"
    );
    let doc: serde_json::Value = serde_json::from_str(line)
        .unwrap_or_else(|e| panic!("stdout line is not valid JSON: {e}\nline: {line}"));
    let runtimes = doc
        .get("runtimes")
        .and_then(serde_json::Value::as_array)
        .unwrap_or_else(|| panic!("expected a `runtimes` array in JSON, got: {doc}"));
    assert!(
        runtimes.is_empty(),
        "expected an empty `runtimes` array, got: {runtimes:?}"
    );
}

#[when("the user previews an update")]
async fn preview_update(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["update", "--dry-run"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the CLI refuses because no managed runtimes are registered")]
async fn refuses_no_managed_runtimes(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    let combined = format!(
        "{}\n{}",
        world.cli_output.as_deref().unwrap_or(""),
        world.cli_stderr.as_deref().unwrap_or("")
    );
    assert!(rc != 0, "expected a non-zero exit, got {rc}:\n{combined}");
    assert!(
        combined.contains("no managed runtimes are registered"),
        "expected the real 'no managed runtimes are registered' bail (not a \
         clap usage error), got:\n{combined}"
    );
}

#[when("the user requests updating a specific runtime without --apply or --dry-run")]
async fn update_runtime_without_apply_or_dry_run(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["update", "--runtime", "some-runtime"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the CLI refuses because --apply or --dry-run is required with --runtime or --activate")]
async fn refuses_apply_or_dry_run_required(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    let combined = format!(
        "{}\n{}",
        world.cli_output.as_deref().unwrap_or(""),
        world.cli_stderr.as_deref().unwrap_or("")
    );
    assert!(rc != 0, "expected a non-zero exit, got {rc}:\n{combined}");
    assert!(
        combined.contains("--runtime and --activate require --apply or --dry-run"),
        "expected the --runtime/--activate no-op guard bail, got:\n{combined}"
    );
}

#[when("the user checks for updates as JSON with --dry-run")]
async fn check_updates_json_with_dry_run(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(world, &["update", "--dry-run", "--json"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the CLI refuses because --dry-run and --json cannot be combined")]
async fn refuses_dry_run_json_conflict(world: &mut E2eWorld) {
    let rc = world.cli_rc.expect("no command rc recorded");
    let combined = format!(
        "{}\n{}",
        world.cli_output.as_deref().unwrap_or(""),
        world.cli_stderr.as_deref().unwrap_or("")
    );
    assert!(rc != 0, "expected a non-zero exit, got {rc}:\n{combined}");
    assert!(
        combined.contains("cannot be used with"),
        "expected a clap conflict error naming --dry-run/--json, got:\n{combined}"
    );
}

// ── update-07: version ordering decides the verdict ────────────────

/// The runtime that is a release *ahead* of what the catalog publishes.
const AHEAD_RUNTIME_KEY: &str = "nightly-tarball-gfx120x-all-7-10-0";
/// The runtime that is a release *behind* what the catalog publishes.
const BEHIND_RUNTIME_KEY: &str = "nightly-tarball-gfx110x-all-7-9";

/// Serves one nightly tarball catalog covering two families, publishing a
/// version older than one registered runtime and newer than the other.
///
/// The catalog deliberately uses the two-component `7.9`, which is what made
/// the old comparator fall back to a text compare: `"7.10.0" < "7.9"` as text,
/// the reverse of the real order. Both halves of the inversion are reachable
/// from this single fixture, since `select_tarball_candidate` filters the
/// catalog by a per-family file-name prefix.
///
/// Nightly, because the release channel filters out versions it cannot parse
/// before any comparison happens and so never showed the defect.
#[given("a nightly tarball catalog older than one registered runtime and newer than another")]
async fn nightly_catalog_straddling_two_runtimes(world: &mut E2eWorld) {
    let root = world
        .isolated_root
        .as_ref()
        .expect("scenario has no isolated root")
        .path()
        .to_path_buf();

    plant_nightly_tarball_runtime(&root, AHEAD_RUNTIME_KEY, "gfx120X-all", "7.10.0", 2_000);
    plant_nightly_tarball_runtime(&root, BEHIND_RUNTIME_KEY, "gfx110X-all", "7.9", 1_000);
    serve_nightly_tarball_catalog(
        world,
        &root,
        &[("gfx120X-all", "7.9"), ("gfx110X-all", "7.10.0")],
    );
}

/// Serves a nightly tarball catalog publishing one `(family, version)` file per
/// entry, and points the CLI at it.
fn serve_nightly_tarball_catalog(
    world: &mut E2eWorld,
    root: &std::path::Path,
    published: &[(&str, &str)],
) {
    // `platform_tarball_token` picks this from the host, so the fixture has to
    // match or the prefix filter finds nothing on the Windows lane.
    let platform = if cfg!(windows) { "windows" } else { "linux" };
    let entries = published
        .iter()
        .map(|(family, version)| {
            format!(
                "{{\"name\": \"therock-dist-{platform}-{family}-{version}.tar.gz\", \"mtime\": {:?}}}",
                1_787_000_000.0_f64
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let served = root.join("nightly-tarball-catalog");
    let catalog_dir = served.join("tarball-nightly");
    std::fs::create_dir_all(&catalog_dir).expect("failed to create catalog dir");
    std::fs::write(
        catalog_dir.join("index.html"),
        format!("<html><body><script>const files = [{entries}];</script></body></html>"),
    )
    .expect("failed to write tarball catalog");

    let server = LoopbackServer::start(&served);
    let base = format!("{}/tarball-nightly/", server.base_url());
    world.artifact_server = Some(server);
    // Without the trust opt-in the override is ignored and the CLI would reach
    // the real nightly catalog over the network.
    world
        .command_env
        .push(("ROCM_CLI_THEROCK_ALLOW_BASE_OVERRIDE", "1".into()));
    world
        .command_env
        .push(("ROCM_CLI_THEROCK_NIGHTLY_TARBALL_BASE", base.into()));
}

/// Writes one installed nightly tarball runtime into the registry.
///
/// `installed_at_unix_ms` differs per runtime so the registry order is fixed by
/// the timestamp rather than by the directory read order.
fn plant_nightly_tarball_runtime(
    root: &std::path::Path,
    runtime_key: &str,
    family: &str,
    version: &str,
    installed_at_unix_ms: u64,
) {
    let install_root = root.join(format!("runtime-{runtime_key}"));
    std::fs::create_dir_all(&install_root).expect("failed to create install root");
    let registry = root.join("data").join("runtimes").join("registry");
    std::fs::create_dir_all(&registry).expect("failed to create registry dir");
    let manifest = serde_json::to_string_pretty(&serde_json::json!({
        "runtime_key": runtime_key,
        "runtime_id": format!("therock-nightly:{family}"),
        "channel": "nightly",
        "format": "tarball",
        "family": family,
        "family_source": "manual",
        "version": version,
        "install_root": install_root,
        "selected_artifact_url": "https://example.invalid/rocm",
        "read_only": false,
        "installed_at_unix_ms": installed_at_unix_ms,
    }))
    .expect("failed to serialize runtime manifest");
    std::fs::write(registry.join(format!("{runtime_key}.json")), manifest)
        .expect("failed to write runtime manifest");
}

#[when("the user checks for updates against that catalog")]
async fn check_updates_against_catalog(world: &mut E2eWorld) {
    // `run_rocm` does not forward `command_env`, so the base override would be
    // dropped and the check would hit the real catalog.
    let (stdout, stderr, rc) = crate::run_rocm_with_scenario_env(world, &["update"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the report offers no update for the runtime that is ahead of the catalog")]
async fn no_update_for_ahead_runtime(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert_runtime_status(&out, AHEAD_RUNTIME_KEY, "ahead_of_index");
    // Assert the user-facing consequence together with the status: an
    // `update_available` verdict is what prints this line, and following it
    // would have installed 7.9 over the installed 7.10.0.
    let offer = format!("run `rocm update --apply --runtime {AHEAD_RUNTIME_KEY}`");
    assert!(
        !out.contains(&offer),
        "the report offered to install an older catalog version over a newer \
         installed runtime:\n{out}"
    );
}

#[then("the report offers an update for the runtime that is behind the catalog")]
async fn update_offered_for_behind_runtime(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert_runtime_status(&out, BEHIND_RUNTIME_KEY, "update_available");
    let offer = format!(
        "run `rocm update --apply --runtime {BEHIND_RUNTIME_KEY}` to install the newer runtime side-by-side"
    );
    assert!(
        out.contains(&offer),
        "the report did not offer the newer catalog version to a runtime a \
         release behind it; expected {offer:?} in:\n{out}"
    );
}

// ── update-08: a catalog with nothing newer offers nothing ─────────

/// Installed as `7.0.0-rc1`; the catalog publishes the normalised `7.0.0rc1`.
const RESPELLED_RUNTIME_KEY: &str = "nightly-tarball-gfx120x-all-7-0-0-rc1";
/// Installed as a four-component release, `7.10.0.71000`; the catalog
/// publishes `7.9.0`. As text `7.10.0.71000` sorts below `7.9.0`, and the old
/// comparator fell back to text for any four-component version, so this pair
/// is one it got wrong (it offered 7.9.0 as an update).
const FOUR_COMPONENT_RUNTIME_KEY: &str = "nightly-tarball-gfx110x-all-7-10-0-71000";

/// Serves a nightly tarball catalog that has nothing newer for either
/// registered runtime: one it publishes under a different spelling of the same
/// version, and one it is a minor release behind.
#[given("a nightly tarball catalog that has no newer version for either registered runtime")]
async fn nightly_catalog_with_nothing_newer(world: &mut E2eWorld) {
    let root = world
        .isolated_root
        .as_ref()
        .expect("scenario has no isolated root")
        .path()
        .to_path_buf();

    plant_nightly_tarball_runtime(
        &root,
        RESPELLED_RUNTIME_KEY,
        "gfx120X-all",
        "7.0.0-rc1",
        2_000,
    );
    plant_nightly_tarball_runtime(
        &root,
        FOUR_COMPONENT_RUNTIME_KEY,
        "gfx110X-all",
        "7.10.0.71000",
        1_000,
    );
    serve_nightly_tarball_catalog(
        world,
        &root,
        &[("gfx120X-all", "7.0.0rc1"), ("gfx110X-all", "7.9.0")],
    );
}

#[then("the report calls the runtime the catalog spells differently up to date")]
async fn respelled_runtime_is_up_to_date(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert_runtime_status(&out, RESPELLED_RUNTIME_KEY, "up_to_date");
    // The status and its consequence together: an `update_available` verdict
    // prints this offer, and following it re-downloads the installed runtime.
    let offer = format!("run `rocm update --apply --runtime {RESPELLED_RUNTIME_KEY}`");
    assert!(
        !out.contains(&offer),
        "the report offered to reinstall the version already installed, spelled \
         differently by the catalog:\n{out}"
    );
}

#[then("the report offers no update for the four-component runtime that is ahead of the catalog")]
async fn no_update_for_four_component_runtime(world: &mut E2eWorld) {
    let out = ok_output(world);
    assert_runtime_status(&out, FOUR_COMPONENT_RUNTIME_KEY, "ahead_of_index");
    let offer = format!("run `rocm update --apply --runtime {FOUR_COMPONENT_RUNTIME_KEY}`");
    assert!(
        !out.contains(&offer),
        "the report offered to install the catalog's 7.9.0 over an installed \
         7.10.0.71000:\n{out}"
    );
}

/// Assert the per-runtime report line names `status`, matching on the line for
/// `runtime_key` so a status belonging to the sibling runtime cannot satisfy it.
fn assert_runtime_status(out: &str, runtime_key: &str, status: &str) {
    let marker = format!("runtime {runtime_key} ");
    let line = out
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with(&marker))
        .unwrap_or_else(|| panic!("no report line for runtime {runtime_key:?} in:\n{out}"));
    assert!(
        line.contains(&format!("status={status}")),
        "runtime {runtime_key:?} reported the wrong freshness; expected \
         status={status}, got {line:?}\n\nfull report:\n{out}"
    );
}

// ── Helpers ────────────────────────────────────────────────────────

fn ok_output(world: &E2eWorld) -> String {
    let rc = world.cli_rc.expect("no command rc recorded");
    let combined = format!(
        "{}\n{}",
        world.cli_output.as_deref().unwrap_or(""),
        world.cli_stderr.as_deref().unwrap_or("")
    );
    assert_eq!(rc, 0, "expected success, got rc={rc}:\n{combined}");
    world.cli_output.clone().unwrap_or_default()
}
