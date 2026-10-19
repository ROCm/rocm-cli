// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use cucumber::{then, when};

use crate::E2eWorld;

#[when("the user asks for the model catalog in verbose detail")]
async fn user_asks_for_verbose_catalog(world: &mut E2eWorld) {
    let (stdout, _, rc) = crate::run_rocm(world, &["model", "--verbose"]);
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
        output.contains("Qwen/Qwen3.5-9B") && output.contains("engines=[aim]"),
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
