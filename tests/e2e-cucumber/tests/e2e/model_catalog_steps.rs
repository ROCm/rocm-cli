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

/// Ask the `rocm_command` tool for the catalog, the way the dashboard's Chat
/// tab and any MCP client do. `mcp-call` is the CLI's own entry to that tool,
/// so the same dispatch the dash drives is exercised without a model or a TUI.
fn ask_the_tool_for_the_catalog(world: &mut E2eWorld, args: &[&str]) {
    let arguments = serde_json::json!({ "args": args }).to_string();
    let (stdout, _, rc) = crate::run_rocm(
        world,
        &["mcp-call", "rocm_command", "--arguments-json", &arguments],
    );
    world.cli_output = Some(stdout);
    world.cli_rc = Some(rc);
}

/// The catalog text inside the tool's reply, with the call's success asserted
/// first so a failure reads as the CLI's own rather than as a missing field.
fn catalog_text_from_the_tool(world: &E2eWorld) -> String {
    let output = world.cli_output.as_ref().expect("no tool reply");
    assert_eq!(world.cli_rc, Some(0), "the tool call failed:\n{output}");
    let reply: serde_json::Value = serde_json::from_str(output.trim())
        .unwrap_or_else(|error| panic!("the tool reply is not JSON ({error}):\n{output}"));
    assert_eq!(reply["isError"], serde_json::json!(false), "{reply}");
    reply["content"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("the tool reply carries no text:\n{output}"))
        .to_owned()
}

#[when("a tool-using client asks for the model catalog")]
async fn tool_client_asks_for_catalog(world: &mut E2eWorld) {
    ask_the_tool_for_the_catalog(world, &["model"]);
}

#[when("a tool-using client asks for the model catalog in verbose detail")]
async fn tool_client_asks_for_verbose_catalog(world: &mut E2eWorld) {
    ask_the_tool_for_the_catalog(world, &["model", "--verbose"]);
}

#[then("the client is given the recommended models without the per-recipe diagnostics")]
async fn tool_client_gets_the_recommendations(world: &mut E2eWorld) {
    let text = catalog_text_from_the_tool(world);
    // The same two lines the recommendation listing ends with for the built-in
    // catalog, so this pins the curated renderer rather than merely "some
    // text with a model id in it".
    assert!(
        text.contains(
            "These are recommendations - you can serve any compatible Hugging Face model"
        ) && text.contains("rocm serve <owner/repo>"),
        "expected the recommendation listing:\n{text}"
    );
    // `aliases=[` opens every row of the verbose listing and appears nowhere
    // in the curated one.
    assert!(
        !text.contains("aliases=["),
        "the per-recipe diagnostics must not be the default:\n{text}"
    );
}

#[then("what the client is given is small enough for a locally served model to read")]
async fn tool_catalog_is_small_enough_to_read(world: &mut E2eWorld) {
    let text = catalog_text_from_the_tool(world);
    // 16 KiB is the dash's per-result budget (`TOOL_RESULT_CHAR_BUDGET` in
    // `crates/rocm-dash-tui/src/agent/tools.rs`), chosen to leave room in a
    // 32k-token context for the system prompt and several turns. Asserted as
    // a literal rather than imported so the e2e suite does not take a
    // dependency on a dash internal; the listing measured under 2 KB.
    assert!(
        text.len() < 16 * 1024,
        "the catalog handed to a tool client is {} characters, past the dash's budget:\n{text}",
        text.len()
    );
}

#[then("the client is given the per-recipe diagnostics")]
async fn tool_client_gets_the_diagnostics(world: &mut E2eWorld) {
    let text = catalog_text_from_the_tool(world);
    assert!(
        text.contains("model recipes") && text.contains("aliases=["),
        "expected the verbose per-recipe listing:\n{text}"
    );
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

/// Plant the GPU capacity the recommendation is selected on.
///
/// `ROCM_E2E_FORCE_GPU_TOTAL_GIB` is the `e2e-test-hooks` seam for this: the
/// no-GPU lane measures no capacity and the GPU lanes cannot be made small, so a
/// planted figure is the only way a narrowed listing is reachable from a test.
fn plant_gpu_capacity_gib(world: &mut E2eWorld, total_gib: u32) {
    world
        .command_env
        .push(("ROCM_E2E_FORCE_GPU_TOTAL_GIB", total_gib.to_string().into()));
}

/// Above the four small GGUF/safetensors recipes, below every recipe the
/// `featured` flag used to select, which is the defect these scenarios are
/// about.
const SMALL_GPU_TOTAL_GIB: u32 = 8;

#[given("a machine whose GPU has room for only the smaller models")]
async fn a_small_gpu_machine(world: &mut E2eWorld) {
    plant_gpu_capacity_gib(world, SMALL_GPU_TOTAL_GIB);
}

#[given("a machine whose GPU is too small for any curated model")]
async fn a_tiny_gpu_machine(world: &mut E2eWorld) {
    plant_gpu_capacity_gib(world, 1);
}

/// The premise is stated, not left to the lane: a lane whose GPU memory
/// `gpu_vram_usage` can read (`amd-smi`, else the DRM sysfs fallback) measures
/// its real capacity when nothing is planted and is narrowed, which is how this
/// scenario failed on the MI300X lanes. The hook is told the memory is
/// unreadable instead.
#[given("a machine whose GPU memory cannot be read")]
async fn an_unmeasured_machine(world: &mut E2eWorld) {
    world
        .command_env
        .push(("ROCM_E2E_FORCE_GPU_TOTAL_GIB", "unreadable".into()));
}

/// An integrated GPU's shared pool, sized to the same host the small-GPU
/// scenarios describe, in bytes.
///
/// Matches [`SMALL_GPU_TOTAL_GIB`] so the narrowing asserted here is the one
/// `model-catalog-06` already asserts, and the only thing differing between the
/// two scenarios is what kind of memory produced it.
const SMALL_SHARED_POOL_BYTES: u64 = SMALL_GPU_TOTAL_GIB as u64 * 1024 * 1024 * 1024;

/// Plant a real integrated-GPU machine rather than stating the verdict for one.
///
/// `plant_gpu_capacity_gib` cannot describe this host. That hook supplies a
/// figure and says nothing about where the figure came from, so a scenario built
/// on it asserts the classification instead of running it: production could map
/// a shared pool to "dedicated" and such a scenario would stay green. This
/// plants the sysfs a Strix Halo actually presents -- an APU `gfx_target_version`
/// on the KFD node, the BIOS carve-out as the VRAM counter, the aperture as
/// `mem_info_gtt_total` -- under `ROCM_CLI_TEST_HOST_ROOT`, so
/// `classify_accelerator_memory` runs and the scenario fails when it is wrong.
///
/// `aperture_bytes` is `None` for the host whose pool could not be read. That is
/// a different machine from one nobody measured: the GPU is present and known to
/// share memory, and only the amount is missing.
fn plant_integrated_gpu(world: &mut E2eWorld, aperture_bytes: Option<u64>) {
    crate::e2e::diagnose_steps::plant_single_gpu_with_memory(
        world,
        crate::e2e::diagnose_steps::GFX1151_TARGET_VERSION,
        Some(crate::e2e::diagnose_steps::CARVE_OUT_BYTES),
        aperture_bytes,
    );
}

#[given(
    "a machine whose graphics share memory with the rest of the system, with room for only the smaller models"
)]
async fn a_small_shared_pool_machine(world: &mut E2eWorld) {
    plant_integrated_gpu(world, Some(SMALL_SHARED_POOL_BYTES));
}

#[given(
    "a machine whose graphics share memory with the rest of the system, and that amount cannot be read"
)]
async fn an_unreadable_shared_pool_machine(world: &mut E2eWorld) {
    plant_integrated_gpu(world, None);
}

/// The ordinary shape of an integrated GPU: no VRAM counters at all, because
/// there is no VRAM. `plant_integrated_gpu` plants the carve-out as a counter,
/// which is the shape a host *with* a ROCm build reports; this is the other one,
/// and it is the one that decides whether the listing's evidence-of-a-GPU test
/// is the same one `rocm diagnose --model` uses.
#[given(
    "a machine whose graphics share memory with the rest of the system and report no memory of their own"
)]
async fn a_shared_pool_machine_with_no_vram_counters(world: &mut E2eWorld) {
    crate::e2e::diagnose_steps::plant_single_gpu_with_memory(
        world,
        crate::e2e::diagnose_steps::GFX1151_TARGET_VERSION,
        None,
        Some(SMALL_SHARED_POOL_BYTES),
    );
}

/// Pin the *figure* the narrowing used, not merely that it happened.
///
/// Every adapter query on an APU reports the BIOS carve-out, and reaching a fit
/// verdict against that instead of the aperture is the headline defect this
/// whole area exists to prevent — but it is invisible from the assertions above,
/// because a listing narrowed against either figure still offers something and
/// still says it fits. So name a model that fits the pool and does not fit the
/// carve-out: it can only be on offer if the aperture was what was weighed.
#[then("the offer includes a model too large for the carve-out the adapter reports")]
async fn assert_offer_reaches_past_the_carve_out(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    let carve_out_gib = crate::e2e::diagnose_steps::CARVE_OUT_BYTES as f64 / 1024.0_f64.powi(3);
    let pool_gib = SMALL_SHARED_POOL_BYTES as f64 / 1024.0_f64.powi(3);
    let discriminating = rocm_core::builtin_model_recipe_registry()
        .recipes
        .iter()
        .filter(|recipe| {
            recipe
                .min_gpu_mem_gb
                .is_some_and(|min| f64::from(min) > carve_out_gib && f64::from(min) <= pool_gib)
        })
        .map(|recipe| recipe.canonical_model_id.clone())
        .collect::<Vec<_>>();
    assert!(
        !discriminating.is_empty(),
        "the catalog ships no recipe between the {carve_out_gib} GiB carve-out and the \
         {pool_gib} GiB pool, so this scenario can no longer tell the two figures apart"
    );
    assert!(
        discriminating.iter().any(|id| output.contains(id.as_str())),
        "none of {discriminating:?} is offered, so the listing was narrowed against the \
         carve-out rather than the shared pool:\n{output}"
    );
}

/// The caveat is false of a discrete card, and the e2e hook builds its host by
/// hand rather than through the classifier — so without this, flipping that
/// literal would print "this GPU has no dedicated VRAM" for every forced host
/// with nothing to catch it.
#[then("the user is not told anything about shared memory")]
async fn assert_no_shared_pool_caveat(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    assert!(
        !output.contains("ceiling on memory this GPU shares"),
        "this machine's memory is its own, so saying it is shared would be false:\n{output}"
    );
}

#[then("each recipe's memory verdict names the shared pool rather than VRAM")]
async fn assert_verbose_names_the_shared_pool(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    assert!(
        output.contains("reason: shared GPU memory pool"),
        "the per-recipe verdicts must name the pool they were weighed against:\n{output}"
    );
    assert!(
        !output.contains("aggregate GPU VRAM"),
        "an integrated GPU has no VRAM, so no row may call its pool that:\n{output}"
    );
    // The fitting rows are the ones a user acts on, and this renderer has no
    // closing note to carry the qualification the curated listing ends with.
    // Without it `--verbose` says a bare "supported" about a pair the diagnosis
    // will only ever call DEGRADED.
    assert!(
        output.contains("gpu_fit: supported"),
        "the planted host must fit something, or this asserts nothing:\n{output}"
    );
    assert!(
        output.contains("that pool is shared with the operating system"),
        "a row reported as fitting must say the pool it cleared is shared:\n{output}"
    );
}

/// The remedy printed beside a verdict that did not fit. It used to read "use a
/// GPU with at least N", which this machine's owner cannot act on: the figure is
/// the pool the system allows the graphics, not a card with a slot behind it.
#[then("the remedy it offers is one this machine can act on")]
async fn assert_remedy_is_actionable(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    assert!(
        output.contains("not a card that can be swapped for a larger one"),
        "a row that does not fit must offer a remedy this machine can act on:\n{output}"
    );
    assert!(
        !output.contains("or use a GPU with at least"),
        "there is no larger card to fit on a machine whose graphics have no memory of \
         their own:\n{output}"
    );
}

/// The same question through the assistant's tool, on the same machine.
///
/// The capacity is planted again because `command_env` is for the *next* `rocm`
/// invocation and the user's own run already consumed it; without this the
/// assistant would be asked on an unmeasured host, which matches an unnarrowed
/// reply whether or not the tool path measures anything.
#[when("an assistant on the same machine asks the same question through its tools")]
async fn assistant_asks_the_same_question(world: &mut E2eWorld) {
    let user_saw = world
        .cli_output
        .take()
        .expect("the user's own listing must be captured first");
    plant_gpu_capacity_gib(world, SMALL_GPU_TOTAL_GIB);
    let arguments = serde_json::json!({ "args": ["model"] }).to_string();
    let (stdout, _, rc) = crate::run_rocm_with_scenario_env(
        world,
        &["mcp-call", "rocm_command", "--arguments-json", &arguments],
    );
    world.cli_output = Some(stdout);
    world.cli_rc = Some(rc);
    world.cli_outputs = Some(vec![user_saw]);
}

#[then("the assistant is given the same recommendations the user saw")]
async fn assert_the_assistant_matches_the_user(world: &mut E2eWorld) {
    let user_saw = world
        .cli_outputs
        .as_ref()
        .and_then(|outputs| outputs.first())
        .expect("the user's listing was not captured")
        .clone();
    let assistant_saw = catalog_text_from_the_tool(world);
    assert!(
        assistant_saw.contains(user_saw.trim()),
        "an assistant asked what this machine can run must be given the answer the user was \
         given.\nthe user saw:\n{user_saw}\nthe assistant saw:\n{assistant_saw}"
    );
}

/// The listing this captures becomes the substring target that
/// `assert_the_assistant_matches_the_user` compares the assistant's answer
/// against, so its success is asserted here rather than left implied. An empty
/// stdout from a crash would make that comparison `contains("")` -- trivially
/// true -- and `model-catalog-09` would pass having compared nothing.
#[when("the user asks which models are recommended")]
async fn user_asks_for_recommendations(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm_with_scenario_env(world, &["model"]);
    assert_eq!(rc, 0, "`rocm model` failed:\n{stderr}");
    assert!(
        !stdout.trim().is_empty(),
        "`rocm model` printed nothing, so there is no answer to compare against:\n{stderr}"
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

/// Every offered row carries its own fit verdict, so the listing states the
/// claim this asserts rather than the test inferring it from model names --
/// which would go stale the moment the catalog changes.
///
/// What this cannot reach: a recipe declaring no GPU minimum carries no verdict
/// at all, and the per-platform cap drops those recipes anyway, since they sort
/// last. So reading an undeclared minimum as "fits anything" is invisible from
/// here and is pinned by
/// `a_host_too_small_for_anything_is_offered_nothing_rather_than_the_wrong_thing`
/// instead -- naming a large model here would assert something the cap already
/// made true, which passes whether the fit rule is right or not.
#[then("every model offered is one this machine can run")]
async fn assert_every_offer_fits(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    // Either label, because the row names the kind of memory it was weighed
    // against: a discrete card's own VRAM, or an integrated part's shared pool.
    // Which of the two is right for a given host is not this step's business --
    // `the rows say the memory is shared` is where that is pinned, because a
    // permissive check here would let the card wording pass on an APU.
    assert!(
        output.contains("fits this GPU") || output.contains("fits this shared pool"),
        "a narrowed listing must offer something and say it fits:\n{output}"
    );
    assert!(
        !output.contains("needs a larger GPU") && !output.contains("larger than this shared pool"),
        "a model this machine cannot run must not be recommended to it:\n{output}"
    );
}

/// The row label, which is the only qualifier a row carries on its own.
///
/// The closing note describes the list; a reader scanning rows sees neither it
/// nor `--verbose`. And the signed-index path prints rows with no closing note
/// at all, so there the label is the whole of it.
#[then("each offered row says the memory it was weighed against is shared")]
async fn assert_rows_say_the_memory_is_shared(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    assert!(
        output.contains("fits this shared pool"),
        "an offered row must name the pool it cleared, not a card:\n{output}"
    );
    assert!(
        !output.contains("fits this GPU"),
        "\"fits this GPU\" claims of an integrated part what the readiness check refuses to \
         claim about it -- clearing a shared ceiling is Degraded, never Ready:\n{output}"
    );
    // ROCMAI-543: the same guard the renderer's own tests carry. These strings
    // are only reachable on a shared-pool host, so the existing `is_ascii`
    // assertions -- which render with no host memory at all -- never see them.
    assert!(
        output.is_ascii(),
        "non-ASCII in the listing mojibakes on a Windows console:\n{output}"
    );
}

#[then("the user is told the offer was narrowed to this machine")]
async fn assert_narrowing_stated(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    assert!(
        output.contains("Narrowed to what this machine can run"),
        "a short list with nothing said about it reads as a small catalog:\n{output}"
    );
}

/// The narrowing on its own would read as a promise that what is left fits
/// whatever else the machine is doing. On an APU it is a ceiling the desktop
/// draws on too, and the listing has to say so.
#[then("the user is told that memory is shared with everything else running")]
async fn assert_shared_pool_stated(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    assert!(
        output.contains("ceiling on memory this GPU shares with the operating system")
            && output.contains("not memory set aside for it"),
        "a narrowed shared pool must be named as shared, or the list reads as memory set \
         aside for the model:\n{output}"
    );
}

#[then("the user is told none of the curated models fit")]
async fn assert_nothing_fits_is_stated(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    assert!(
        output.contains("None of the curated models fit"),
        "an empty list must say why, not render as a broken command:\n{output}"
    );
}

#[then("no model is offered")]
async fn assert_no_model_offered(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    // Both labels, so this stays true if a shared-pool host is ever given a
    // nothing-fits scenario; checking only the card wording would pass there
    // while a row was being offered.
    assert!(
        !output.contains("fits this GPU") && !output.contains("fits this shared pool"),
        "nothing fits, so nothing may be offered as fitting:\n{output}"
    );
}

/// Asserts what was *offered*, not only what was said about it.
///
/// Checking the notes alone is not enough, and that is not hypothetical: under
/// a sabotaged fit predicate this step still passed, because the notes are
/// driven by whether a figure exists rather than by what the filter did with
/// one. The evidence that nothing was narrowed is that the large models are
/// still on offer -- they need far more than any small host could supply, so
/// their presence is only possible on an unfiltered list.
#[then("the recommendations are not narrowed to this machine")]
async fn assert_not_narrowed(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    assert!(
        output.contains("~54 GiB") || output.contains("~62 GiB"),
        "an unfiltered listing still offers the large curated models; their absence means the \
         list was narrowed after all:\n{output}"
    );
    assert!(
        !output.contains("Narrowed to what this machine can run"),
        "nothing could be measured, so nothing may be claimed to have been narrowed:\n{output}"
    );
    assert!(
        !output.contains("None of the curated models fit"),
        "an unmeasured machine must never be told nothing fits it:\n{output}"
    );
}

#[then("the user is told the list is not specific to it")]
async fn assert_unmeasured_is_stated(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no command was run");
    assert!(
        output.contains("could not be read, so these are not filtered"),
        "the listing must say the recommendations are not host-specific:\n{output}"
    );
}
