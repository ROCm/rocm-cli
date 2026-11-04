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
    // The recipe is listed by its model name (`aim-qwen3.5-9b` is an alias) and
    // shares the entry with vLLM, so match the AIM engine within that one line.
    assert!(
        output
            .lines()
            .any(|line| line.contains("Qwen/Qwen3.5-9B") && line.contains("engines=[aim")),
        "expected the AIM recipe listed by canonical id and engine:\n{output}"
    );
    // The host-fit fields of the same entry: which GPU families it covers, that
    // it needs no Hugging Face token, and the profile pinned for a family.
    assert!(
        output.lines().any(|line| line.contains("engine_recipe aim")
            && line.contains("supported_gfx_families=[gfx110X-all,gfx120X-all]")
            && line.contains("requires_hf_token=no")
            && line.contains("gfx110X-all=vllm-w7900-bf16-tp1-latency")),
        "expected the AIM recipe's host-fit line in the verbose listing:\n{output}"
    );
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

/// Unix ms for 2001-09-09, far more than 180 days before any run of this suite.
const LONG_AGO_MS: u64 = 1_000_000_000_000;

/// Point the CLI at a signed recipe index holding the built-in `Qwen/Qwen3.5-9B`
/// AIM entry with `generated_at_unix_ms` set `LONG_AGO_MS`, the only difference
/// from the shipped entry. Signed because that is the only way the CLI loads a
/// recipe index (same xtask keygen/sign path as `diagnose_steps`).
#[given("a recipe catalog whose AIM entry was last verified long ago")]
async fn catalog_with_stale_aim_entry(world: &mut E2eWorld) {
    let mut recipe = rocm_core::resolve_builtin_model_recipe("Qwen/Qwen3.5-9B")
        .expect("the built-in AIM recipe");
    for engine_recipe in &mut recipe.engine_recipes {
        if engine_recipe.engine == "aim" {
            engine_recipe.generated_at_unix_ms = Some(u128::from(LONG_AGO_MS));
        }
    }
    let root = world
        .isolated_root
        .as_ref()
        .expect("scenario has no isolated root")
        .path()
        .to_path_buf();
    let index = root.join("stale-aim-recipes.json");
    let signature = root.join("stale-aim-recipes.json.sig");
    let public_key = root.join("stale-aim-public.pem");
    let private_key = root.join("stale-aim-private.pem");
    let document = serde_json::json!({
        "schema_version": 1,
        "source": "e2e-stale-aim",
        "recipes": [recipe],
    });
    std::fs::write(
        &index,
        serde_json::to_vec_pretty(&document).expect("failed to serialize the recipe fixture"),
    )
    .expect("failed to write the recipe fixture");

    let keygen = crate::e2e::diagnose_steps::xtask_command()
        .args(["keygen", "--private-out"])
        .arg(&private_key)
        .arg("--public-out")
        .arg(&public_key)
        .current_dir(crate::e2e::diagnose_steps::workspace_root())
        .status()
        .expect("failed to run xtask keygen");
    assert!(keygen.success(), "xtask keygen failed");
    let sign = crate::e2e::diagnose_steps::xtask_command()
        .args(["sign", "--private-key"])
        .arg(&private_key)
        .arg("--in")
        .arg(&index)
        .arg("--out")
        .arg(&signature)
        .current_dir(crate::e2e::diagnose_steps::workspace_root())
        .status()
        .expect("failed to run xtask sign");
    assert!(sign.success(), "xtask sign failed");

    world
        .command_env
        .push(("ROCM_CLI_MODEL_RECIPE_INDEX_PATH", index.into_os_string()));
    world.command_env.push((
        "ROCM_CLI_MODEL_RECIPE_INDEX_SIGNATURE_PATH",
        signature.into_os_string(),
    ));
    world.command_env.push((
        "ROCM_CLI_MODEL_RECIPE_INDEX_PUBLIC_KEY_PATH",
        public_key.into_os_string(),
    ));
}

#[then("the catalog warns that the AIM entry may no longer be current")]
async fn catalog_warns_aim_entry_is_stale(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no model catalog output");
    assert_eq!(
        world.cli_rc,
        Some(0),
        "listing the catalog failed:\n{output}"
    );
    assert!(
        output.contains("warning: aim entry was last verified")
            && output.contains("days ago")
            && output.contains("entries are refreshed by hand"),
        "expected a staleness warning for the AIM entry, got:\n{output}"
    );
}
