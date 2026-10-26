// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use cucumber::{given, then, when};
use e2e_cucumber::fixture_binary::fixture_target_name;

use crate::E2eWorld;

#[when("the user asks for the model catalog in verbose detail")]
async fn user_asks_for_verbose_catalog(world: &mut E2eWorld) {
    // `run_rocm_with_scenario_env` so a `Given` step that armed `command_env`
    // (e.g. a fake-docker PATH) takes effect; it is a no-op otherwise.
    let (stdout, _, rc) = crate::run_rocm_with_scenario_env(world, &["model", "--verbose"]);
    world.cli_output = Some(stdout);
    world.cli_rc = Some(rc);
}

#[then("the catalog lists an AIM engine recipe")]
async fn catalog_lists_aim_recipe(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no model catalog output");
    assert_eq!(
        world.cli_rc,
        Some(0),
        "listing the catalog is not a failure:\n{output}"
    );
    assert!(
        output.contains("aim-qwen3.5-9b") && output.contains("engines=[aim]"),
        "expected the AIM recipe listed by canonical id and engine:\n{output}"
    );
}

#[then("the catalog's AIM entry matches this host's real Docker availability")]
async fn catalog_aim_entry_matches_docker_availability(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no model catalog output");
    let aim_available = e2e_cucumber::capability::host_capability().engine_available("aim");
    if aim_available {
        assert!(
            !output.contains("unsupported_docker_unavailable"),
            "Docker is reachable on this host, so the AIM entry should not be flagged unavailable:\n{output}"
        );
    } else {
        assert!(
            output.contains("unsupported_docker_unavailable")
                && output.contains("Docker daemon not reachable"),
            "Docker is unreachable on this host, so the AIM entry should name the reason:\n{output}"
        );
    }
}

/// Plant a `docker` on `PATH` that always fails, so the catalog's
/// Docker-unavailable gate is exercised deterministically regardless of
/// whether this host actually has Docker.
#[given("no Docker daemon is reachable")]
async fn no_docker_daemon_reachable(world: &mut E2eWorld) {
    let root = world
        .isolated_root
        .as_ref()
        .expect("scenario root")
        .path()
        .to_path_buf();
    let bin_dir = root.join("fake-bin");
    let installed = bin_dir.join(fixture_target_name("docker"));
    crate::install_fixture_binary("fake-docker", &installed);

    let mut path = std::ffi::OsString::from(&bin_dir);
    path.push(if cfg!(windows) { ";" } else { ":" });
    path.push(std::env::var_os("PATH").unwrap_or_default());
    world.command_env.push(("PATH", path));
}

#[then("the catalog's AIM entry is flagged unavailable for lack of Docker")]
async fn catalog_aim_entry_flagged_unavailable(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no model catalog output");
    assert!(
        output.contains("unsupported_docker_unavailable")
            && output.contains("Docker daemon not reachable"),
        "expected the AIM entry to name the Docker-unavailable reason:\n{output}"
    );
}
