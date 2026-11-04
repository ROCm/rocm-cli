// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm serve` command handler and the AIM pre-flight that runs inside it.
//!
//! The engine-recipe overrides `serve()` applies to the launch hint
//! (`--tool-call-parser`, `--gpu-memory-utilization`, generation defaults and
//! `--engine-env`) live in `serve_recipe.rs`; this file imports them by name.
//!
//! Originally a mechanical relocation from `main.rs` with no behavior change —
//! the `dispatch()` call site stays byte-identical (`serve(ServeArgs { .. })`,
//! re-imported via `use crate::serve_cmd::{ServeArgs, serve};`; `model_fit.rs`'s
//! `assess_model_for_host` calls `select_serve_engine` directly).
//! `Cli` is not referenced from this file; `DevicePolicyArg` (part of the clap arg tree)
//! is, via `use crate::DevicePolicyArg`.
//!
//! That "no behavior change" describes the move itself, not the file's history
//! since: EAI-8059 added the model-keyed reuse pre-gate below, which is new
//! serve behaviour and the one thing that can run ahead of the no-usable-GPU
//! bail. Read the comment at the pre-gate, not this paragraph, for what the
//! serve path does today.
//!
//! `collect_serve_notes`, the GPU/device warnings folded into the interactive
//! summary, was relocated here from `main.rs` verbatim, with its two tests and
//! their `vram`/`host_gpu` fixtures; `serve` is its only production caller.
//! `detect_gpu_count`, `gpu_vram_usage`,
//! `parse_device_policy`, `parse_gpu_selection`,
//! `select_gpu_indices_under_launch_lock`, `serve_gpu_low_memory_warning`,
//! and `validate_pinned_gpu_index` moved out of `main.rs` into
//! `gpu_policy.rs` in ROCMAI-214 (Phase 6f); `resolve_engine_selection` and
//! `validate_engine_selection_runtime` moved out of `main.rs` into
//! `engine_runtime_select.rs` in ROCMAI-215 (Phase 6f); this file's `use`
//! list below points to both now instead of at the crate root. The
//! managed-service-spawning tail this file calls into — `start_managed_service`,
//! `run_attached_service`, `resolve_endpoint_auth`,
//! `ensure_public_bind_engine_supported`,
//! `drop_orphaned_endpoint_key_on_already_running`, `print_managed_launch_plain`,
//! `validate_bind_host`, `ensure_background_helper_running_quiet` — was relocated
//! to `service_launch.rs` in ROCMAI-91 (Phase 6b), along with its test module;
//! this file's `use` list below points there now instead of at the crate root.

use anyhow::{Context as _, Result, bail};
use rocm_core::{
    AppPaths, GfxSupport, ModelRecipeRecord, RocmCliConfig, default_engine_for_platform,
    detect_host_gfx_target, detect_host_gpu_summary, engine_recipe_gfx_support,
    generate_service_id, normalize_therock_family, preferred_serve_engine_for_host_gpu_summary,
    resolve_model_recipe,
};
use rocm_engine_protocol::{
    DevicePolicy, EngineMethod, EngineRecipeHint, GpuSelection, ResolveModelRequest,
    ResolveModelResponse,
};

use crate::DevicePolicyArg;
use crate::endpoint_keys;
use crate::engine_ipc::engine_request;
use crate::engine_runtime_select::{
    EngineSelection, resolve_engine_selection, validate_engine_selection_runtime,
};
use crate::engines_cmd::{engine_manages_own_runtime, ensure_self_managed_engine_ready};
use crate::gpu_policy::{
    GpuVramUsage, active_visibility_mask, detect_gpu_count, gpu_vram_usage, parse_device_policy,
    parse_gpu_selection, pinned_index_bound, select_gpu_indices_under_launch_lock,
    serve_gpu_low_memory_warning, validate_pinned_gpu_index, write_unchecked_index_warning,
};
use crate::managed_service_status::{find_live_managed_service, live_managed_services_for_model};
use crate::serve_recipe::{
    ServeGenerationDefaults, engine_env_ignored_note, engine_recipe_enables_tool_choice,
    engine_recipe_with_extra_env, engine_recipe_with_generation_defaults,
    engine_recipe_with_gpu_memory_utilization_override, engine_recipe_with_tool_call_override,
    parse_gpu_memory_utilization, protocol_engine_recipe_hint, render_serve_engine_recipe_lines,
};
use crate::serve_summary;
use crate::service_launch::{
    drop_orphaned_endpoint_key_on_already_running, e2e_simulate_oom_launch,
    ensure_background_helper_running_quiet, ensure_public_bind_engine_supported,
    managed_service_reuse_refusal, no_usable_gpu_refusal, print_managed_launch_plain,
    resolve_endpoint_auth, run_attached_service, start_managed_service, validate_bind_host,
};
use crate::{
    GatedEngineAction, cli_progress, device_policy_name, ensure_engine_allowed_on_os,
    ensure_engine_allowed_on_platform,
};

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct ServeEngineSelection {
    pub(crate) engine: String,
    pub(crate) source: &'static str,
}

pub(crate) fn select_serve_engine(
    explicit_engine: Option<&str>,
    configured_default: Option<&str>,
    recipe: Option<&ModelRecipeRecord>,
    host_gpu_summary: Option<&rocm_core::HostGpuSummary>,
) -> ServeEngineSelection {
    if let Some(engine) = explicit_engine.filter(|value| !value.trim().is_empty()) {
        return ServeEngineSelection {
            engine: engine.to_owned(),
            source: "explicit --engine",
        };
    }

    if let Some(engine) = configured_default.filter(|value| !value.trim().is_empty()) {
        return ServeEngineSelection {
            engine: engine.to_owned(),
            source: "configured default_engine",
        };
    }

    if let Some(engine) = host_gpu_summary.and_then(preferred_serve_engine_for_host_gpu_summary) {
        // Only honor the GPU preference when the model's recipe can actually run on
        // that engine. A recipe that exists but does not support the preferred engine
        // (for example a GGUF model that only Lemonade can serve) must fall through to
        // its own preferred engine instead of being forced onto an incompatible engine.
        let recipe_supports_preferred =
            recipe.is_none_or(|recipe| model_recipe_supports_engine(recipe, engine));
        if recipe_supports_preferred {
            return ServeEngineSelection {
                engine: engine.to_owned(),
                source: "detected ROCm GPU family prefers vLLM",
            };
        }
    }

    if let Some(engine) = recipe
        .and_then(|recipe| recipe_preferred_engine(recipe, host_gpu_summary, cfg!(windows)))
        .filter(|value| !value.trim().is_empty())
    {
        return ServeEngineSelection {
            engine: engine.to_owned(),
            source: "recipe preferred engine; pass --engine <engine> to override; no automatic fallback",
        };
    }

    ServeEngineSelection {
        engine: default_engine_for_platform().to_owned(),
        source: "platform default",
    }
}

/// The recipe's first preferred engine that can run on this host. AIM is only
/// chosen on its own when the detected GPU family is one the image has a profile
/// for and the host is Linux or WSL, so an undetected GPU or native Windows is not,
/// and on native Windows vLLM is not an option once AIM has been passed over. When no listed engine fits, the
/// platform default applies, as it did before the recipe was catalogued. An
/// explicit `--engine aim` never comes through here, and still gets the refusal
/// that names the supported GPUs.
fn recipe_preferred_engine<'a>(
    recipe: &'a ModelRecipeRecord,
    host_gpu_summary: Option<&rocm_core::HostGpuSummary>,
    windows: bool,
) -> Option<&'a str> {
    let is_aim = |engine: &str| engine.eq_ignore_ascii_case("aim");
    let aim_usable = !windows
        && recipe
            .engine_recipes
            .iter()
            .find(|record| is_aim(&record.engine))
            .is_none_or(|record| {
                matches!(
                    engine_recipe_gfx_support(
                        record,
                        host_gpu_summary.and_then(|summary| summary.therock_family.as_deref()),
                    ),
                    GfxSupport::Supported(_)
                ) || record.supported_gfx_families.is_empty()
            });
    // Native Windows has no vLLM, but a recipe whose first engine is vLLM keeps
    // it there, as it does without the AIM catalog: the skip applies only when
    // falling past an AIM first choice that was not usable.
    let mut passed_unusable_aim = false;
    recipe
        .preferred_engines
        .iter()
        .map(String::as_str)
        .find(|engine| {
            if is_aim(engine) {
                passed_unusable_aim |= !aim_usable;
                aim_usable
            } else {
                !(windows && passed_unusable_aim && engine.eq_ignore_ascii_case("vllm"))
            }
        })
}

fn model_recipe_supports_engine(recipe: &ModelRecipeRecord, engine: &str) -> bool {
    recipe
        .preferred_engines
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(engine))
        || recipe
            .engine_recipes
            .iter()
            .any(|candidate| candidate.engine.eq_ignore_ascii_case(engine))
}

fn serve_model_ref_for_engine(
    model: &str,
    recipe: Option<&ModelRecipeRecord>,
    selected_engine: &str,
) -> String {
    let Some(recipe) =
        recipe.filter(|recipe| model_recipe_supports_engine(recipe, selected_engine))
    else {
        return model.to_owned();
    };
    let engine_recipe = recipe
        .engine_recipes
        .iter()
        .find(|engine_recipe| engine_recipe.engine.eq_ignore_ascii_case(selected_engine));
    let base = engine_recipe
        .and_then(|engine_recipe| engine_recipe.model_id_override.as_deref())
        .filter(|value| !value.trim().is_empty())
        .map_or_else(|| recipe.canonical_model_id.clone(), ToOwned::to_owned);
    // Pins a container-based engine (e.g. AIM) to the catalog's `image_digest`
    // rather than a floating tag, so a `docker pull`/`run` keyed on this ref
    // actually enforces it instead of the field being purely informational.
    match engine_recipe.and_then(|engine_recipe| engine_recipe.image_digest.as_deref()) {
        Some(digest) if !digest.trim().is_empty() => format!("{base}@{digest}"),
        _ => base,
    }
}

fn serve_engine_selection_line(selection: &ServeEngineSelection) -> String {
    format!("  engine_selection: {}", selection.source)
}

/// For AIM, refuses a GPU the catalog has no profile for and pins the catalog's
/// `AIM_PROFILE_ID` for the detected GPU family. `host_family` is a normalized
/// TheRock family, `None` when it could not be detected (nothing is enforced
/// then). Runs before `--engine-env`, so an explicit override still wins.
fn engine_recipe_with_aim_gfx_profile(
    engine: &str,
    recipe: Option<&ModelRecipeRecord>,
    hint: Option<EngineRecipeHint>,
    host_family: impl FnOnce() -> Option<String>,
) -> Result<Option<EngineRecipeHint>> {
    let Some(aim) = recipe
        .filter(|_| engine.eq_ignore_ascii_case("aim"))
        .and_then(|recipe| {
            recipe
                .engine_recipes
                .iter()
                .find(|record| record.engine.eq_ignore_ascii_case("aim"))
                .map(|record| (recipe, record))
        })
        .filter(|(_, record)| !record.supported_gfx_families.is_empty())
    else {
        return Ok(hint);
    };
    let (recipe, record) = aim;
    let family = host_family();
    match engine_recipe_gfx_support(record, family.as_deref()) {
        GfxSupport::Unrestricted => Ok(hint),
        GfxSupport::Unsupported => bail!(
            "the AIM image for `{}` has no profile for this GPU ({}); it supports: {}. Choose a model \
             or engine that supports this GPU (`rocm model --verbose`)",
            recipe.canonical_model_id,
            family.as_deref().unwrap_or("unknown"),
            record.supported_gfx_families.join(", ")
        ),
        GfxSupport::Supported(profile) => Ok(match (hint, profile) {
            (Some(mut hint), Some(profile)) => {
                hint.extra_env
                    .insert("AIM_PROFILE_ID".to_owned(), profile.to_owned());
                Some(hint)
            }
            (hint, _) => hint,
        }),
    }
}

/// The AIM checks that depend on the host GPU, in the order `serve` applies
/// them: refuse a family the image has no profile for, pin the profile the
/// catalog names for the detected family, then refuse an image that needs more
/// than one GPU. The host is only probed for an AIM record that restricts GPU
/// families, and not at all for `--device cpu_only`, since these checks are about
/// the GPU the container would use. An undetected family enforces nothing.
fn aim_gpu_preflight(
    engine: &str,
    model: &str,
    recipe: Option<&ModelRecipeRecord>,
    hint: Option<EngineRecipeHint>,
    cpu_only: bool,
    detect_family: impl FnOnce() -> Option<String>,
) -> Result<Option<EngineRecipeHint>> {
    let family = aim_engine_record(engine, recipe)
        .filter(|record| !record.supported_gfx_families.is_empty() && !cpu_only)
        .and_then(|_| detect_family());
    let hint = engine_recipe_with_aim_gfx_profile(engine, recipe, hint, || family.clone())?;
    if let Some(needed) = aim_min_gpu_count(engine, recipe, family.as_deref()) {
        ensure_aim_multi_gpu_supported(model, needed)?;
    }
    Ok(hint)
}

fn aim_engine_record<'a>(
    engine: &str,
    recipe: Option<&'a ModelRecipeRecord>,
) -> Option<&'a rocm_core::ModelRecipeEngineRecord> {
    recipe
        .filter(|_| engine.eq_ignore_ascii_case("aim"))?
        .engine_recipes
        .iter()
        .find(|record| record.engine.eq_ignore_ascii_case("aim"))
}

/// Refuses to launch a gated AIM image before the container is pulled when no
/// Hugging Face token will reach it: `--engine-env HF_TOKEN=<value>`, or the bare
/// `--engine-env HF_TOKEN` with `HF_TOKEN` exported (the form that is never
/// stored). `hint` carries the `--engine-env` entries.
fn ensure_aim_hf_token(
    engine: &str,
    recipe: Option<&ModelRecipeRecord>,
    hint: Option<&EngineRecipeHint>,
    env: impl Fn(&str) -> Option<String>,
) -> Result<()> {
    if !aim_engine_record(engine, recipe).is_some_and(|record| record.requires_hf_token) {
        return Ok(());
    }
    let supplied = hint
        .and_then(|hint| hint.extra_env.get("HF_TOKEN"))
        .is_some_and(|value| {
            !value.trim().is_empty() || env("HF_TOKEN").is_some_and(|v| !v.trim().is_empty())
        });
    if supplied {
        return Ok(());
    }
    let model = recipe.map_or("<model>", |r| r.canonical_model_id.as_str());
    bail!(
        "`{model}` is a gated model and needs a Hugging Face token. Export it and pass the \
         bare name so it is forwarded and never stored: `export HF_TOKEN=<token>`, then \
         `rocm serve {model} --engine aim --engine-env HF_TOKEN`"
    )
}

/// GPUs an AIM image needs on `family`, when more than one (see
/// `min_gpu_counts`). `None` when it needs one, or the family is unknown.
fn aim_min_gpu_count(
    engine: &str,
    recipe: Option<&ModelRecipeRecord>,
    family: Option<&str>,
) -> Option<u32> {
    let family = family?;
    aim_engine_record(engine, recipe)?
        .min_gpu_counts
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(family))
        .map(|(_, count)| *count)
}

/// Refuses an AIM image that needs `needed` GPUs before anything is pulled:
/// `rocm serve` selects a single GPU, so the container would start with too few
/// devices and fail inside the image.
fn ensure_aim_multi_gpu_supported(model: &str, needed: u32) -> Result<()> {
    bail!(
        "`{model}` needs {needed} GPUs for this GPU family, but `rocm serve` runs an engine on \
         one GPU and does not support multi-GPU serving yet, so this model cannot be started \
         here however many GPUs the host has. Choose a model with a single-GPU profile \
         (`rocm model --verbose`)"
    )
}

/// Note shown when the AIM profile that was pinned for this serve targets boards
/// other than the detected GPU (a gfx family can cover cards the profile was not
/// built for). `hint` is the final engine hint: the note needs the catalog's
/// profile to be what will actually run, so it stays quiet when no profile was
/// pinned (GPU family undetected, `--device cpu_only`) or `--engine-env
/// AIM_PROFILE_ID` replaced it.
fn aim_board_note(
    engine: &str,
    recipe: Option<&ModelRecipeRecord>,
    gpu_name: Option<&str>,
    hint: Option<&EngineRecipeHint>,
) -> Option<String> {
    let record = aim_engine_record(engine, recipe)?;
    let name = gpu_name?;
    if record.supported_boards.is_empty() {
        return None;
    }
    let pinned = hint?.extra_env.get("AIM_PROFILE_ID")?;
    if !record.profile_ids.values().any(|id| id == pinned) {
        return None;
    }
    let squash = |value: &str| {
        value
            .chars()
            .filter(char::is_ascii_alphanumeric)
            .collect::<String>()
            .to_ascii_lowercase()
    };
    let name_key = squash(name);
    if record
        .supported_boards
        .iter()
        .any(|board| name_key.contains(&squash(board)))
    {
        return None;
    }
    Some(format!(
        "the pinned AIM profile `{pinned}` targets {}; `{name}` is not one of them, so it may not run well. Override with `--engine-env AIM_PROFILE_ID=<id>`",
        record.supported_boards.join(" / ")
    ))
}

/// Whether the reuse pre-gate may run its engine/runtime probe.
///
/// The pre-gate exists for exactly one purpose: to answer the no-usable-GPU
/// bail in `serve()`. Everything it does has a side effect that must not
/// precede that bail — a runtime-registry write, a self-managed engine install,
/// a `ResolveModel` round-trip — and the only consumer of its answer
/// (`reuse_existing`) is a term of that bail's condition. So it must not open
/// any wider than the bail's own host precondition: under a GPU-required
/// policy, on a host the probe *authoritatively* reports as having no usable
/// AMD GPU.
///
/// Not "exactly when the bail fires", and the difference is only in test-hook
/// builds: the bail also requires `!scripted_backend_failure` (see the `if`
/// below it), which is `cfg!(feature = "e2e-test-hooks")`/env-gated. Under that
/// scenario the gate can open while the bail cannot fire. That is deliberate —
/// the hook stands in for the GPU the lane does not have, so the reuse path it
/// exercises must still behave as it would on a real GPU-less host — and it
/// costs nothing in a shipped binary, where the term can never be true.
///
/// `host_has_no_usable_gpu` is therefore the bail's own predicate, computed once
/// by the caller and handed to both, so the two cannot drift apart. They did:
/// keying the gate on the live-service check alone pulled all of that work ahead
/// of `validate_pinned_gpu_index` on every host where a live service's model
/// name happened to overlap — including hosts with a perfectly good GPU, where
/// the bail can never fire and reuse therefore buys nothing. #267 (`945d81b`)
/// put `--gpu` validation first deliberately, so an out-of-range ordinal gets a
/// GPU-specific refusal rather than whatever unrelated runtime error the
/// resolution happens to raise first.
///
/// `live_services_for_model` is taken as a closure, not a value, because it is
/// not free either: it goes through `load_managed_services`, which refreshes
/// liveness and can cost a probe and a `record.write()` per live record. On a
/// host with a GPU that cost is not paid at all.
///
/// It yields the refreshed records rather than a `bool` so the caller's
/// canonical-id lookup can run over them instead of repeating the scan; see
/// [`live_managed_services_for_model`].
fn reuse_pregate_opens<T>(
    cpu_only: bool,
    host_has_no_usable_gpu: bool,
    live_services_for_model: impl FnOnce() -> Option<T>,
) -> Option<T> {
    (!cpu_only && host_has_no_usable_gpu)
        .then(live_services_for_model)
        .flatten()
}

/// The no-usable-GPU bail's host precondition, read from the visibility-resolved
/// set of usable AMD GPU ordinals.
///
/// A named function rather than an inline expression because three things have
/// to answer it identically: the bail, the reuse pre-gate whose only product is
/// a term of that bail, and the unit test that pins the pre-gate. The test used
/// to spell the expression out a second time, so changing the real one left it
/// green and only serve-31 — on a self-hosted GPU lane — could notice.
///
/// `None` is "could not probe on this platform", not "no GPU", and stays
/// permissive; only an authoritative empty set is "no usable GPU". See the call
/// site in [`serve`] for what answers `None`.
const fn host_has_no_usable_gpu(visible: Option<&[u32]>) -> bool {
    matches!(visible, Some([]))
}

/// Parsed `rocm serve` arguments. Grouped into a struct to keep the dispatcher
/// and `serve()` readable now that the verb carries verbose/smoke-test controls.
pub(crate) struct ServeArgs {
    pub(crate) model: String,
    pub(crate) engine: Option<String>,
    pub(crate) device: Option<DevicePolicyArg>,
    pub(crate) gpu: Option<String>,
    pub(crate) runtime_id: Option<String>,
    pub(crate) env_id: Option<String>,
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) foreground: bool,
    pub(crate) managed: bool,
    pub(crate) verbose: bool,
    pub(crate) no_smoke_test: bool,
    pub(crate) allow_public_bind: bool,
    pub(crate) require_api_key: bool,
    pub(crate) tool_call_parser: Option<String>,
    pub(crate) gpu_memory_utilization: Option<String>,
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) api_key: Option<String>,
    pub(crate) engine_env: Vec<String>,
}

pub(crate) fn serve(args: ServeArgs) -> Result<()> {
    let ServeArgs {
        model,
        engine,
        device,
        gpu,
        runtime_id,
        env_id,
        host,
        port,
        foreground,
        managed,
        verbose,
        no_smoke_test,
        allow_public_bind,
        require_api_key,
        tool_call_parser,
        gpu_memory_utilization,
        temperature,
        top_p,
        max_tokens,
        api_key,
        engine_env,
    } = args;
    let _ = managed; // background is now the default; --managed is accepted as an explicit synonym.
    validate_bind_host(&host, allow_public_bind)?;
    // Loopback stays credential-free; a public bind must be authenticated. Resolve
    // (or generate) the endpoint key now so every downstream path — engine spawn,
    // readiness probe, smoke test, and the client-config we print — shares one value.
    // The `--api-key` flag wins; otherwise fall back to `ROCM_SERVE_API_KEY` (read
    // here rather than via clap's `env` so it works without clap's `env` feature).
    let supplied_key = api_key.or_else(|| {
        std::env::var("ROCM_SERVE_API_KEY")
            .ok()
            .filter(|value| !value.trim().is_empty())
    });
    let endpoint_auth = resolve_endpoint_auth(&host, supplied_key.as_deref(), require_api_key)?;
    let paths = AppPaths::discover()?;
    let mut config = RocmCliConfig::load(&paths)?;
    // Host GPU detection can involve sysfs/WSL probing, so only run it when engine
    // selection would actually consult it: no explicit `--engine` and no non-empty
    // configured `default_engine`.
    let host_gpu_summary = if engine
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty())
        || config
            .default_engine
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
    {
        None
    } else {
        Some(detect_host_gpu_summary(Some(&paths)))
    };
    let shared_recipe = resolve_model_recipe(&model)?;
    let serve_engine = select_serve_engine(
        engine.as_deref(),
        config.default_engine.as_deref(),
        shared_recipe.as_ref(),
        host_gpu_summary.as_ref(),
    );
    let selected_engine = serve_engine.engine.clone();
    // Before runtime resolution and any launch: an engine the platform gate rules
    // out (vLLM on native Windows) must not reach a "no active runtime" error or
    // a background server that dies with no output.
    ensure_engine_allowed_on_os(&selected_engine, GatedEngineAction::Serve)?;
    // Fail closed: a public bind must be authenticated, but Windows managed
    // Lemonade cannot receive the key (see `ensure_public_bind_engine_supported`),
    // so refuse rather than launch an open public server.
    ensure_public_bind_engine_supported(&selected_engine, endpoint_auth.is_some(), cfg!(windows))?;
    let engine_model_ref =
        serve_model_ref_for_engine(&model, shared_recipe.as_ref(), &selected_engine);
    let recipe_hint = shared_recipe
        .as_ref()
        .filter(|recipe| model_recipe_supports_engine(recipe, &selected_engine))
        .and_then(|recipe| protocol_engine_recipe_hint(recipe, &selected_engine));
    let device_policy = parse_device_policy(device.as_ref().map(|policy| policy.as_policy_str()))?;
    // CPU-only serving never pins a GPU, so skip GPU resolution entirely and
    // surface the explicit `--gpu` as ignored rather than printing a device the
    // server will not use.
    //
    // Always `false` today, and kept anyway: `parse_device_policy` rejects
    // `cpu`/`cpu_only` outright (pinned by
    // `parse_device_policy_defaults_to_gpu_required_without_cpu_fallback`), so
    // the only policy that reaches here is `GpuRequired`. The term is what makes
    // every `!cpu_only` guard below read as a guard on the *policy* rather than
    // on a constant, so admitting the policy later cannot silently turn CPU-only
    // serving into GPU pre-flight failures. It is not evidence that the CPU path
    // is exercised; nothing in the suite can reach it.
    let cpu_only = matches!(device_policy, DevicePolicy::CpuOnly);
    // The token needs nothing from the GPU, so it is checked first: a gated model
    // is refused for its missing token on any host, not only on one whose GPU the
    // image supports. Only `--engine-env` can supply it, so a hint built from
    // that alone is enough; the real hint below still gets the same env applied
    // after the catalog's profile pin, which is what lets the flag override it.
    ensure_aim_hf_token(
        &selected_engine,
        shared_recipe.as_ref(),
        engine_recipe_with_extra_env(&selected_engine, None, &engine_env)?.as_ref(),
        |key| std::env::var(key).ok(),
    )?;
    let recipe_hint = aim_gpu_preflight(
        &selected_engine,
        &model,
        shared_recipe.as_ref(),
        recipe_hint,
        cpu_only,
        || {
            detect_host_gfx_target()
                .as_deref()
                .and_then(normalize_therock_family)
        },
    )?;
    // vLLM rejects the TUI chat tab's tool-bearing requests with HTTP 400 unless it
    // is launched with `--enable-auto-tool-choice`/`--tool-call-parser`. The parser
    // is model-specific and vLLM does not auto-detect it, so it is never guessed: it
    // comes from authored catalog recipe metadata or an explicit `--tool-call-parser`
    // override, applied here for vLLM only.
    let engine_serves_vllm = selected_engine.eq_ignore_ascii_case("vllm");
    let engine_recipe = engine_recipe_with_tool_call_override(
        &selected_engine,
        recipe_hint,
        tool_call_parser.as_deref(),
    );
    // Validated before anything is launched so a typo fails immediately rather
    // than surfacing as a vLLM argparse error deep in the engine log.
    let gpu_memory_utilization = parse_gpu_memory_utilization(gpu_memory_utilization.as_deref())?;
    let engine_recipe = engine_recipe_with_gpu_memory_utilization_override(
        &selected_engine,
        engine_recipe,
        gpu_memory_utilization,
    );
    // Stored without a `note:` prefix so it can feed both output paths: the plan
    // path adds the prefix inline, the interactive summary adds it when rendering.
    let gpu_memory_utilization_note = (gpu_memory_utilization.is_some() && !engine_serves_vllm)
        .then(|| {
            format!(
                "--gpu-memory-utilization applies only to vLLM; ignored for engine '{selected_engine}'"
            )
        });
    // Translate the engine-neutral CLI controls into each adapter's server-wide
    // defaults: vLLM generation config or Lemonade llama.cpp launch flags.
    let generation_defaults = ServeGenerationDefaults {
        temperature,
        top_p,
        max_tokens,
    };
    let engine_recipe = engine_recipe_with_generation_defaults(
        &selected_engine,
        engine_recipe,
        generation_defaults,
    )?;
    let engine_recipe = engine_recipe_with_extra_env(&selected_engine, engine_recipe, &engine_env)?;
    let aim_gpu_name = aim_engine_record(&selected_engine, shared_recipe.as_ref())
        .filter(|record| !record.profile_ids.is_empty())
        .and_then(|_| {
            host_gpu_summary
                .clone()
                .unwrap_or_else(|| detect_host_gpu_summary(Some(&paths)))
                .name
        });
    let engine_env_notes: Vec<String> = [
        engine_env_ignored_note(&selected_engine, &engine_env),
        aim_board_note(
            &selected_engine,
            shared_recipe.as_ref(),
            aim_gpu_name.as_deref(),
            engine_recipe.as_ref(),
        ),
    ]
    .into_iter()
    .flatten()
    .collect();
    let tool_call_note = if tool_call_parser.is_some() && !engine_serves_vllm {
        Some(format!(
            "note: --tool-call-parser applies only to vLLM; ignored for engine '{selected_engine}'"
        ))
    } else if engine_serves_vllm && !engine_recipe_enables_tool_choice(engine_recipe.as_ref()) {
        Some(
            "note: tool calling is disabled for this model; pass `--tool-call-parser <name>` (e.g. hermes, llama3_json, mistral) to enable it".to_owned(),
        )
    } else {
        None
    };
    let gpu_selection = parse_gpu_selection(gpu.as_deref())?;
    // AMD GPU ordinals still usable after the active visibility mask
    // (`HIP_VISIBLE_DEVICES`, then `ROCR_VISIBLE_DEVICES`) is applied, in HIP
    // ordinal space — the space `--gpu` is validated and exported through. A
    // `ROCR_VISIBLE_DEVICES` mask hides devices below HIP, which re-indexes the
    // survivors as `0..N`, so those HIP positions are what comes back here, not the
    // physical ROCR token values. `None` means availability could not be probed
    // (a non-Linux target, both KFD and DRM unreadable on Linux, or a mask this
    // ordinal-only probe cannot interpret such as one naming UUIDs) — NOT WSL,
    // which answers authoritatively via `detect_wsl_summary`. On `None` selection
    // stays permissive and defers device validation to the engine. An empty set is
    // the authoritative "no usable GPU", not "unknown". Computed once and reused
    // for the fail-fast check below and for mask-aware GPU selection, so serve
    // never auto-selects — or accepts an explicit `--gpu` for — a hidden device.
    let visible_gpu_indices = if cpu_only {
        None
    } else {
        rocm_core::usable_amd_gpu_indices()
    };
    // The no-usable-GPU bail's host predicate, computed once here because two
    // things depend on it and they must not disagree: the bail itself, and the
    // reuse pre-gate below whose only product is a term of that bail. `None` is
    // "could not probe on this platform", not "no GPU", so it does not open
    // either one.
    let host_has_no_usable_gpu = host_has_no_usable_gpu(visible_gpu_indices.as_deref());
    // Reusing an already-running managed service launches nothing and pins no
    // GPU, so it must bypass the GPU-required pre-flight below — the reused
    // service was already vetted at its own launch, and this invocation does no
    // GPU work. Detect that here, but only when a live managed service that
    // plausibly serves *this* model already exists, and when the model ref can be
    // canonicalized (a runtime/env is available, or the engine manages its own).
    // Without a runtime we cannot resolve, so we fall through and the pre-flight
    // refuses the no-GPU / no-runtime case with its usual message.
    //
    // The whole block is gated on `host_has_no_usable_gpu` — the bail's own
    // predicate, computed once above and handed to `reuse_pregate_opens` — so on
    // a host that has a GPU nothing here runs at all and `--gpu` validation
    // keeps the position #267 gave it. The live-service check is passed as a
    // closure for the same reason: it is decided from the managed-service
    // records alone, with no engine round-trip and no install, but it is not
    // literally record reads and nothing else — `load_managed_services`
    // refreshes liveness first, so each live record can cost a probe and a
    // `record.write()`. The bail below is where that full cost is stated, and it
    // is the qualified claim, not this one. Everything these conditions guard
    // has side effects that must not precede the bail —
    //
    //   - the pre-gate matches on the model, not just the engine, because the
    //     probe body is real engine work (a `ResolveModel` round-trip, and for a
    //     self-managing engine an `ensure_self_managed_engine_ready` that can
    //     print "Preparing <engine> for GPU serving..." and install). Gating on
    //     the engine alone let a live service for an *unrelated* model — one this
    //     invocation can never reuse — drag that work ahead of the no-usable-GPU
    //     bail on a GPU-less host.
    //
    //   - resolving the engine/runtime selection is not free either.
    //     `validate_engine_selection_runtime` takes `single_ready_runtime_key`
    //     whenever neither `--runtime-id` nor `--env-id` is given (the default
    //     path), and that calls `recover_setup_runtime_registration`, which can
    //     `fs::create_dir_all` + `fs::write` a runtime registry manifest and can
    //     fail with an unrelated runtime-manifest error. So it stays *inside*
    //     this block, behind the record-only check, and is computed below the
    //     bail on every other path.
    let mut resolved_selection: Option<EngineSelection> = None;
    let mut resolved_model: Option<ResolveModelResponse> = None;
    let mut reuse_existing = false;
    if let Some(live_records) = reuse_pregate_opens(cpu_only, host_has_no_usable_gpu, || {
        live_managed_services_for_model(&paths, &selected_engine, &engine_model_ref)
    }) {
        let selection = validate_engine_selection_runtime(
            &paths,
            resolve_engine_selection(
                &config,
                &selected_engine,
                runtime_id.as_deref(),
                env_id.as_deref(),
            ),
        )?;
        let can_resolve_model = selection.runtime_id.is_some()
            || selection.env_id.is_some()
            || engine_manages_own_runtime(&selected_engine);
        if can_resolve_model {
            if engine_manages_own_runtime(&selected_engine) {
                ensure_self_managed_engine_ready(&paths, &mut config, &selected_engine)?;
            }
            let probe = engine_request::<_, ResolveModelResponse>(
                Some(&paths),
                &selected_engine,
                EngineMethod::ResolveModel,
                &ResolveModelRequest {
                    model_ref: engine_model_ref.clone(),
                    runtime_id: selection.runtime_id.clone(),
                    device_policy: Some(device_policy.clone()),
                    recipe_override: None,
                    engine_recipe: engine_recipe.clone(),
                },
            )?;
            // Liveness + engine + canonical id says a service *matches*; it does
            // not say this invocation can reuse it. `spawn_managed_engine_child`
            // still refuses on a recipe mismatch or an unmet `--require-api-key`,
            // so deciding the bypass on the weaker test sent the user to a dead
            // end: on a GPU-less host the refusal was skipped, the spawn told
            // them to stop the running server and re-run, and the re-run — with
            // nothing left to reuse — hit the no-GPU refusal. The advice named a
            // command that could not clear the condition.
            //
            // Every argument here is load-bearing, and each has a scenario that
            // fails if it is dropped: serve-30 for `require_api_key`, serve-32
            // for `requested_recipe_json`. The device policy cannot be dropped
            // at all — the predicate takes it by value, so passing nothing is a
            // compile error (see `managed_service_reuse_refusal`). A `cpu_only`
            // service must not buy a `gpu_required` serve a bypass of the bail.
            let requested_recipe_json = engine_recipe
                .as_ref()
                .map(serde_json::to_string)
                .transpose()
                .context("failed to encode engine recipe hint")?;
            // Over the records the pre-gate already loaded and liveness-refreshed,
            // not a second scan of the same directory.
            reuse_existing = find_live_managed_service(
                &live_records,
                &selected_engine,
                &probe.canonical_model_id,
            )
            .is_some_and(|existing| {
                managed_service_reuse_refusal(
                    existing,
                    requested_recipe_json.as_deref(),
                    require_api_key,
                    &device_policy,
                )
                .is_none()
            });
            resolved_model = Some(probe);
        }
        resolved_selection = Some(selection);
    }
    // Fail fast under a GPU-required policy when the host has no usable AMD GPU,
    // before preparing or launching any engine for *this* model (no wasted engine
    // download, and an actionable message instead of a late engine crash). The
    // engine enforces the same rule as a backstop.
    //
    // The precise contract, since the reuse detection above is the one thing that
    // can precede this bail: everything with a side effect — the runtime-selection
    // resolution (which can write a runtime registry manifest), the `ResolveModel`
    // round-trip, and any self-managed engine install — runs first only when
    // this bail's own host predicate already holds *and* a live managed service
    // already matches this engine and model. On a host with a usable GPU, and
    // whenever no such service matches (the ordinary launch, and every no-GPU
    // refusal path that has no overlapping live service), that block's body is
    // skipped and this is the first thing that runs.
    //
    // "Matches" is weaker than "is about to be reused", and the difference is
    // visible to users. `any_live_managed_service_for_model` matches model names
    // leniently (a two-way substring compare), so a live `qwen3-8b-instruct`
    // opens the gate for `rocm serve qwen` — an invocation that then fails
    // `managed_service_reuse_refusal` and still gets this bail, having already
    // paid for the engine preparation. Worse, a `?` on
    // `validate_engine_selection_runtime`, `ensure_self_managed_engine_ready` or
    // `engine_request` above can return an unrelated error *instead of* this
    // refusal on such a host. Both are the documented price of leniency (README
    // and `--help` state it); narrowing the match would silently break reuse by
    // short name, which is the worse failure. serve-29 is the regression test
    // for the non-overlapping case, where the ordering does hold strictly.
    //
    // Its *condition* is not free either, though it is paid only on a host this
    // bail already applies to: `any_live_managed_service_for_model`
    // goes through `load_managed_services`, which refreshes every service record —
    // including records for unrelated models, since the model filter is applied
    // afterwards — so a stale `ready`/`running` record costs an endpoint listing
    // probe and possibly an inference probe, plus a `record.write()`, before this
    // bail is reached. That is bounded and paid only when such records exist, but
    // it is not "nothing runs before the pre-flight".
    //
    // Skipped for cpu_only and when reusing an already-running service (nothing is
    // launched); permissive when availability cannot be probed on this platform
    // (probe returns `None`). The E2E-only
    // backend-failure scenario bypasses this host precondition so the black-box
    // test reaches Lemonade's backend boundary without real GPU hardware, and the
    // E2E-only OOM-launch fault injection likewise stands in for the GPU it does
    // not have.
    let scripted_backend_failure = cfg!(feature = "e2e-test-hooks")
        && std::env::var_os("ROCM_E2E_LEMONADE_BACKEND_INSTALL_FAILURE").is_some();
    if !cpu_only
        && !scripted_backend_failure
        && !e2e_simulate_oom_launch()
        && !reuse_existing
        && host_has_no_usable_gpu
    {
        bail!("{}", no_usable_gpu_refusal(&device_policy));
    }
    // The Docker probe comes after every refusal that needs no daemon (the AIM
    // token and GPU pre-flight, the no-usable-GPU bail): a request they refuse
    // never reaches `docker info`.
    ensure_engine_allowed_on_platform(&selected_engine, GatedEngineAction::Serve)?;
    // Whether the refusal above was waived *only* because a live service was
    // expected to be reused. `reuse_existing` already implies `!cpu_only` and
    // `host_has_no_usable_gpu` (the pre-gate requires both), so the scripted
    // backend failure is the one remaining term. Handed to the spawn, which
    // re-checks reuse authoritatively under the launch lock and raises the
    // refusal itself if the service is gone by then.
    let gpu_refusal_waived_for_reuse = reuse_existing && !scripted_backend_failure;
    // `--gpu` selects by the amd-smi `gpu` ordinal but is exported via
    // `HIP_VISIBLE_DEVICES`; those orderings can diverge when
    // `ROCR_VISIBLE_DEVICES`/partitioning is in play, so warn at serve time.
    let rocr_visible_devices_set = std::env::var_os("ROCR_VISIBLE_DEVICES").is_some();
    // Which visibility variables name a mask that acts on this platform, or
    // `None` when none does. The visible set alone cannot say: with no mask it
    // is just `0..present`, indistinguishable from a HIP mask that happens to
    // list the low ordinals. `validate_pinned_gpu_index` uses this only to word
    // its rejection — "under the active visibility mask" when a mask is set,
    // "not present on this host" when none is — so an out-of-range `--gpu` on an
    // unmasked host is not blamed on a mask the user never set. Native Windows
    // HIP has no ROCr layer, so `ROCR_VISIBLE_DEVICES` is no mask there.
    let visibility_mask = active_visibility_mask(
        rocr_visible_devices_set,
        std::env::var_os("HIP_VISIBLE_DEVICES").is_some(),
        rocm_core::runtime_is_windows(),
    );
    let gpu_vram = if cpu_only { None } else { gpu_vram_usage() };
    // Validate an explicit `--gpu <index>` up front — before engine/runtime
    // resolution — so an out-of-range or masked-out ordinal produces a
    // GPU-specific refusal even when no ROCm runtime is configured. Otherwise the
    // "no active ROCm runtime is configured" bail-out below pre-empts it and the
    // user sees a generic runtime error for what is really a bad `--gpu` value.
    // This is pure validation (no service-state read), so it needs no lock;
    // `--gpu auto` reads live busy-GPU state and stays under `launch_lock` below.
    //
    // "Up front" is exact on every host with a usable GPU: `reuse_pregate_opens`
    // is false there, so nothing above has touched the engine or the registry.
    // The single exception is the reuse exemption, which by construction only
    // runs on a host the bail above already applies to — where `visible_gpu_indices`
    // is `Some([])` and so *every* `--gpu <index>` is out of range anyway.
    //
    // On native Windows `visible_gpu_indices` is always `None` and amd-smi is
    // usually missing or unusable, so the index had nothing to be checked
    // against and `--gpu 7` on a one-GPU machine was accepted (ROCMAI-824).
    // There the AMD display adapters Windows lists bound the index instead.
    // That listing is only queried here, for an explicit index: it can count an
    // adapter HIP does not support, so it may refuse an index but must not pick
    // one for `--gpu auto`.
    let pinned_gpu_indices = if !cpu_only && let GpuSelection::Index(index) = &gpu_selection {
        let detected = detect_gpu_count();
        let bound = pinned_index_bound(
            visible_gpu_indices.clone(),
            detected,
            visibility_mask.is_some(),
            rocm_core::windows_amd_gpu_listing,
        );
        let pinned = validate_pinned_gpu_index(
            *index,
            bound.count,
            bound.visible.as_deref(),
            visibility_mask,
            bound.mask_unresolved,
        )?;
        write_unchecked_index_warning(&mut std::io::stderr(), &pinned).ok();
        Some(pinned.indices)
    } else {
        None
    };
    // Resolve which runtime/env this serve will use, now that the host GPU
    // pre-condition and `--gpu` have both been checked (or, on the reuse path,
    // the pre-gate already resolved it under the exemption). This must not move back
    // above the bail: `validate_engine_selection_runtime` reaches
    // `recover_setup_runtime_registration` on the default (no `--runtime-id` /
    // `--env-id`) path, which writes a runtime registry manifest and can fail
    // with a runtime-manifest error that has nothing to do with the GPU — so a
    // GPU-less `rocm serve <model>` would mutate disk, or report the wrong
    // problem, instead of failing fast. The reuse pre-gate above needs the answer
    // early and already paid for it when a reusable service exists; take that
    // answer rather than repeating the registry recovery.
    //
    // The ordering is enforced by `@id:serve-unrelated-live-service-still-fails-fast`
    // (serve-29), which plants a configured-but-unregistered setup venv — the one
    // state that makes the recovery's write reachable — and asserts the runtime
    // registry is still empty after the refusal. Without that fixture the call
    // could be hoisted above the bail with the whole suite green, because
    // `recover_setup_runtime_registration` returns early when no setup venv is
    // configured and the other e2e roots carry none.
    let resolved_selection = match resolved_selection {
        Some(selection) => selection,
        None => validate_engine_selection_runtime(
            &paths,
            resolve_engine_selection(
                &config,
                &selected_engine,
                runtime_id.as_deref(),
                env_id.as_deref(),
            ),
        )?,
    };
    if !matches!(device_policy, DevicePolicy::CpuOnly)
        && resolved_selection.runtime_id.is_none()
        && resolved_selection.env_id.is_none()
        && !engine_manages_own_runtime(&selected_engine)
    {
        bail!(
            "device_policy: {}; no active ROCm runtime is configured; run `rocm runtimes list` and `rocm runtimes activate <runtime_key>`, or pass --runtime-id/--env-id",
            device_policy_name(&device_policy)
        );
    }
    if !matches!(device_policy, DevicePolicy::CpuOnly)
        && engine_manages_own_runtime(&selected_engine)
        && resolved_model.is_none()
    {
        ensure_self_managed_engine_ready(&paths, &mut config, &selected_engine)?;
    }
    // Reuse the `ResolveModel` answer the reuse pre-gate already paid for, so a
    // reusing invocation does not make the same round-trip twice.
    let resolve = match resolved_model {
        Some(resolve) => resolve,
        None => engine_request::<_, ResolveModelResponse>(
            Some(&paths),
            &selected_engine,
            EngineMethod::ResolveModel,
            &ResolveModelRequest {
                model_ref: engine_model_ref,
                runtime_id: resolved_selection.runtime_id.clone(),
                device_policy: Some(device_policy),
                recipe_override: None,
                engine_recipe,
            },
        )?,
    };
    // Serialize GPU auto-selection with the managed-service claim: the busy-GPU
    // read and the claiming record write inside `spawn_managed_engine_child` must
    // be atomic, or two concurrent `rocm serve --gpu auto` can both read the same
    // GPU as free and launch on it. Taken here — after engine resolution,
    // self-managed runtime prep, and the `ResolveModel` RPC have all completed
    // unlocked — so a slow first-use install (e.g. the Lemonade embeddable
    // download/extract) never blocks an unrelated serve.
    let (gpu_indices, launch_lock) = select_gpu_indices_under_launch_lock(
        &paths,
        cpu_only,
        pinned_gpu_indices,
        detect_gpu_count,
        visible_gpu_indices.as_deref(),
        gpu_vram.as_deref(),
    )?;
    let service_id = generate_service_id(&selected_engine, &resolve.canonical_model_id);

    // Attached foreground streaming is the debugging path, selected by `--verbose`
    // or `--foreground`. Everything else backgrounds the server and, when writing
    // to an interactive terminal, shows a progress spinner + deployment summary
    // instead of a raw log stream. Piped/captured output (CI, the chat assistant)
    // keeps the plain line-by-line form.
    let use_foreground = foreground || verbose;
    let background = !use_foreground;
    let summary_mode = background && std::io::IsTerminal::is_terminal(&std::io::stdout());

    if !summary_mode {
        println!("serve plan");
        println!("  requested model: {model}");
        println!("  resolved model: {}", resolve.canonical_model_id);
        println!("  engine: {selected_engine}");
        println!("{}", serve_engine_selection_line(&serve_engine));
        println!("  host: {host}");
        println!("  port: {port}");
        if let Some(runtime_id) = resolved_selection.runtime_id.as_deref() {
            println!("  runtime_id: {runtime_id}");
        }
        if let Some(env_id) = resolved_selection.env_id.as_deref() {
            println!("  env_id: {env_id}");
        }
        if let Some(source) = resolved_selection.source.as_deref() {
            println!("  selection_source: {source}");
        }
        println!(
            "  device_policy: {}",
            device_policy_name(&resolve.device_policy)
        );
        if cpu_only {
            if matches!(gpu_selection, GpuSelection::Index(_)) {
                println!(
                    "  warning: --gpu was ignored because --device cpu_only runs the model on CPU"
                );
            }
        } else {
            match &gpu_selection {
                GpuSelection::Auto => {
                    let csv = rocm_engine_protocol::gpu_indices_to_csv(&gpu_indices)
                        .unwrap_or_else(|| "none".to_owned());
                    println!("  gpu: auto (selected {csv})");
                }
                GpuSelection::Index(_) => {
                    let csv = rocm_engine_protocol::gpu_indices_to_csv(&gpu_indices)
                        .unwrap_or_else(|| "none".to_owned());
                    println!("  gpu: {csv}");
                }
            }
            if rocr_visible_devices_set {
                println!(
                    "  warning: ROCR_VISIBLE_DEVICES is set; the selected amd-smi ordinal is exported \
                     via HIP_VISIBLE_DEVICES, which the runtime interprets relative to the \
                     ROCR-visible set, so the device the engine binds may differ. Verify the \
                     selected GPU or unset ROCR_VISIBLE_DEVICES."
                );
            }
            if let Some(warning) = serve_gpu_low_memory_warning(
                &gpu_indices,
                gpu_vram.as_deref(),
                host_gpu_summary.as_ref(),
            ) {
                println!("  {warning}");
                if engine_serves_vllm {
                    println!("  note: {}", rocm_core::VLLM_GPU_MEMORY_UTILIZATION_HINT);
                }
            }
        }
        if let Some(engine_recipe) = &resolve.engine_recipe {
            print!("{}", render_serve_engine_recipe_lines(engine_recipe));
        }
        if let Some(note) = &tool_call_note {
            println!("  {note}");
        }
        if let Some(note) = &gpu_memory_utilization_note {
            println!("  note: {note}");
        }
        for note in &engine_env_notes {
            println!("  note: {note}");
        }
    }

    let managed_runtime_id = resolved_selection.runtime_id.clone();
    let managed_env_id = resolved_selection.env_id.clone();

    // Persist the endpoint key (public bind only) in a 0600 file so the engine
    // child, the restart/recovery path, and inspection commands can retrieve it by
    // service id. Loopback binds resolve to `None` and store nothing.
    if let Some(key) = endpoint_auth.as_deref() {
        endpoint_keys::store_endpoint_api_key(&paths, &service_id, key)?;
    }

    if background {
        let mut spinner =
            cli_progress::Spinner::new(format!("Starting {model} on {selected_engine}…"));
        spinner.tick();
        let report = start_managed_service(
            &selected_engine,
            &service_id,
            &model,
            &resolve,
            &host,
            port,
            &resolve.device_policy,
            &gpu_indices,
            managed_runtime_id.as_deref(),
            managed_env_id.as_deref(),
            resolve.engine_recipe.as_ref(),
            endpoint_auth.as_deref(),
            launch_lock,
            require_api_key,
            gpu_refusal_waived_for_reuse,
            &mut |_elapsed| spinner.tick(),
        )?;
        // Not when the launch was fabricated. `simulate_oom_managed_launch`
        // writes a `starting` record for a process that never existed, so the
        // helper would be left watching a pid that will never appear, with
        // nothing in the scenario's teardown to stop it: observed as `rocm
        // daemon` processes outliving their e2e checkout by days, still
        // carrying `ROCM_E2E_SIMULATE_OOM_LAUNCH=1`. The helper is for
        // supervising a real spawned server, and there is none here.
        //
        // That — not ordering — is the whole justification. This suppression
        // runs *before* the OOM note is computed in the `summary_mode` block
        // below, so the previous claim here, that the note "is computed before
        // it would have run", was backwards and could not have excused it.
        // serve-40's `no background automation daemon was left behind` is what
        // holds the branch: deleting it left the scenario green with a `rocm
        // daemon` still running.
        //
        // `e2e_simulate_oom_launch` is a `const fn` returning `false` without
        // the test-only feature, so this branch folds away in shipped builds.
        if !e2e_simulate_oom_launch() {
            ensure_background_helper_running_quiet(summary_mode)?;
        }

        // An equivalent service was already running, so nothing was spawned and the
        // freshly generated key is unused — drop it rather than leave it orphaned in
        // storage. The existing service keeps its own key.
        if report.already_running {
            drop_orphaned_endpoint_key_on_already_running(
                &paths,
                &service_id,
                endpoint_auth.as_deref(),
            );
        }
        // Safe to move `endpoint_auth` here: this branch always returns, so the
        // fall-through (attached) path below never observes it moved.
        let launched_key = if report.already_running {
            None
        } else {
            endpoint_auth
        };

        if summary_mode {
            // Best-effort inference smoke test, on by default (opt out with
            // `--no-smoke-test`). Only meaningful for a freshly-ready server we
            // just launched; skipped when metrics could not be shown anyway.
            let metrics = if !no_smoke_test && !report.already_running && report.status == "ready" {
                spinner.set_label("Running smoke test…");
                // The local provider resolves the endpoint key from the per-service
                // 0600 key file by service id, so the smoke test authenticates
                // against a protected public endpoint without threading the secret
                // through here.
                serve_summary::run_smoke_test(&paths, resolve.served_model_name())
            } else {
                serve_summary::SmokeMetrics::default()
            };
            spinner.clear();

            let mut notes = collect_serve_notes(
                cpu_only,
                &gpu_selection,
                rocr_visible_devices_set,
                &gpu_indices,
                gpu_vram.as_deref(),
                gpu_memory_utilization_note.as_deref(),
                host_gpu_summary.as_ref(),
                engine_serves_vllm,
            );
            notes.extend(engine_env_notes.iter().cloned());
            // A managed serve that failed on VRAM lands here not-ready with the OOM
            // traceback only in its own log. Read that log tail so the summary can
            // name the memory knobs, since the pre-launch low-VRAM warning cannot
            // fire without amd-smi/rocm-smi telemetry.
            //
            // Note the asymmetry this sits inside: `summary_mode` is
            // `background && stdout().is_terminal()`, so the guidance reaches an
            // interactive user only. A scripted / CI / assistant-driven run takes
            // `print_managed_launch_plain`, which ends at `readiness: {status}`
            // and carries no *post-launch* note. It is not noteless: the
            // `!summary_mode` branch above still prints the pre-launch planning
            // lines, low-VRAM `note:` included, so the asymmetry is precisely
            // that the OOM confirmation and the `rocm diagnose --symptom`
            // command are missing — not that the path prints nothing. That path
            // is machine-readable by design and is not the place to grow prose,
            // but it does mean the same failed launch is explained in one
            // invocation and not the other.
            // The engine *name*, not `engine_serves_vllm`: the note's gate is
            // derived inside `append_oom_serve_note` from the value it is about,
            // because a boolean here was a call-site seam no test could hold.
            let notes = serve_summary::append_oom_serve_note(
                notes,
                &selected_engine,
                report.already_running,
                &report.status,
                report.log_path.as_deref(),
            );
            let summary = serve_summary::DeploymentSummary {
                engine: selected_engine.clone(),
                requested_model: model,
                api_model: resolve.served_model_name().to_owned(),
                chat_endpoint: format!("{}/chat/completions", report.endpoint_url),
                service_id: report.service_id.clone(),
                status: report.status.clone(),
                already_running: report.already_running,
                metrics,
                api_key: launched_key,
                notes,
            };
            // Same intentional one-time key display as `print_managed_launch_plain`
            // in service_launch.rs; see its rationale for why the tag below is currently inert.
            // codeql[rust/cleartext-logging]
            print!("{}", serve_summary::render_summary(&summary));
        } else {
            spinner.clear();
            print_managed_launch_plain(&report, launched_key.as_deref());
        }
        return Ok(());
    }

    run_attached_service(
        &selected_engine,
        &service_id,
        &model,
        &resolve,
        &host,
        port,
        &gpu_indices,
        resolved_selection.runtime_id.as_deref(),
        resolved_selection.env_id.as_deref(),
        endpoint_auth.as_deref(),
        launch_lock,
        require_api_key,
        gpu_refusal_waived_for_reuse,
    )
}

/// GPU/device warnings folded into the interactive deployment summary. Mirrors the
/// inline warnings printed in the plain serve plan, in the same order.
#[allow(clippy::too_many_arguments)]
fn collect_serve_notes(
    cpu_only: bool,
    gpu_selection: &GpuSelection,
    rocr_visible_devices_set: bool,
    gpu_indices: &[u32],
    gpu_vram: Option<&[GpuVramUsage]>,
    engine_flag_note: Option<&str>,
    host_gpu_summary: Option<&rocm_core::HostGpuSummary>,
    engine_is_vllm: bool,
) -> Vec<String> {
    let mut notes = Vec::new();
    // An engine-scoped flag the selected engine cannot honor must be reported here
    // too: the summary path is what an interactive `rocm serve` actually prints, so
    // a note only emitted on the plan path would never reach that user.
    if let Some(note) = engine_flag_note {
        notes.push(note.to_owned());
    }
    if cpu_only {
        if matches!(gpu_selection, GpuSelection::Index(_)) {
            notes.push(
                "--gpu was ignored because --device cpu_only runs the model on CPU".to_owned(),
            );
        }
    } else {
        if rocr_visible_devices_set {
            notes.push(
                "ROCR_VISIBLE_DEVICES is set; the selected amd-smi ordinal is exported via \
                 HIP_VISIBLE_DEVICES, which the runtime interprets relative to the ROCR-visible \
                 set, so the device the engine binds may differ. Verify the selected GPU or unset \
                 ROCR_VISIBLE_DEVICES."
                    .to_owned(),
            );
        }
        if let Some(warning) = serve_gpu_low_memory_warning(gpu_indices, gpu_vram, host_gpu_summary)
        {
            notes.push(warning);
            // vLLM's total-VRAM reservation is what turns a busy card into an OOM;
            // pair the generic warning with the concrete knob that avoids it.
            if engine_is_vllm {
                notes.push(rocm_core::VLLM_GPU_MEMORY_UTILIZATION_HINT.to_owned());
            }
        }
    }
    notes
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocm_core::resolve_builtin_model_recipe;

    /// Regression coverage for the pre-gate's ordering (ROCMAI-419 review).
    ///
    /// The gate used to be `!cpu_only && live_managed_services_for_model(..)`,
    /// with no reference to the host's GPU at all. On any host where a live
    /// service's model name overlapped the one asked for — including a host with
    /// a perfectly usable GPU, where the no-usable-GPU bail can never fire and
    /// reuse therefore buys nothing — that pulled `validate_engine_selection_runtime`
    /// (which can write the runtime registry), `ensure_self_managed_engine_ready`
    /// (which can install Lemonade) and the `ResolveModel` round-trip in front of
    /// `validate_pinned_gpu_index`, so a `?` from any of them pre-empted the
    /// GPU-specific refusal #267 (`945d81b`) deliberately put first.
    ///
    /// Both halves are asserted, because the probe is a cost as well as a term:
    /// on a host with a GPU the gate must be shut *without* consulting the
    /// managed-service records at all.
    ///
    /// The host predicate comes from the production
    /// [`host_has_no_usable_gpu`], not a copy of its expression — this test
    /// used to restate it, so changing the real one left the test green.
    #[test]
    fn the_reuse_pregate_stays_shut_on_a_host_that_has_a_usable_gpu() {
        let mut probed = false;
        assert!(
            reuse_pregate_opens(false, host_has_no_usable_gpu(Some(&[0])), || {
                probed = true;
                Some(())
            })
            .is_none(),
            "a host with a usable GPU must not open the reuse pre-gate: the bail \
             it answers cannot fire there, so its engine and registry work would \
             run ahead of `--gpu` validation for nothing"
        );
        assert!(
            !probed,
            "the managed-service probe must not even be consulted on a host with \
             a GPU — `load_managed_services` refreshes liveness and can cost a \
             probe and a `record.write()` per live record"
        );

        // An unprobeable platform is "unknown", not "no GPU". The bail stays
        // permissive there, so the pre-gate must too.
        let mut probed_unknown = false;
        assert!(
            reuse_pregate_opens(false, host_has_no_usable_gpu(None), || {
                probed_unknown = true;
                Some(())
            })
            .is_none(),
            "an unprobeable host must not open the pre-gate either"
        );
        assert!(!probed_unknown);

        // The exemption itself still works: authoritative empty set, live match.
        assert!(
            reuse_pregate_opens(false, host_has_no_usable_gpu(Some(&[])), || Some(())).is_some(),
            "a host with no usable GPU and a matching live service must still \
             reach the reuse probe, or the exemption serve-28 covers is gone"
        );
        // ...and `--device cpu_only` never had a GPU bail to exempt. No CLI
        // invocation can produce that policy today (`parse_device_policy`
        // rejects `cpu_only`), so this asserts the helper's own guard, not a
        // reachable `serve()` input — see the `cpu_only` binding in `serve`.
        assert!(
            reuse_pregate_opens(true, host_has_no_usable_gpu(Some(&[])), || Some(())).is_none()
        );
    }

    #[test]
    fn serve_engine_selection_uses_shared_recipe_when_no_override_exists() {
        let recipe = resolve_builtin_model_recipe("qwen32b").expect("qwen32b recipe");

        let selection = select_serve_engine(None, None, Some(&recipe), None);

        assert_eq!(
            selection,
            ServeEngineSelection {
                engine: "vllm".to_owned(),
                source: "recipe preferred engine; pass --engine <engine> to override; no automatic fallback",
            }
        );
        assert_eq!(
            serve_engine_selection_line(&selection),
            "  engine_selection: recipe preferred engine; pass --engine <engine> to override; no automatic fallback"
        );
        assert_eq!(
            serve_model_ref_for_engine("qwen32b", Some(&recipe), "vllm"),
            "Qwen/Qwen3-32B-FP8"
        );
    }

    #[test]
    fn serve_engine_selection_prefers_vllm_for_supported_gpus() {
        let summary = rocm_core::HostGpuSummary {
            therock_family: Some("gfx90a".to_owned()),
            ..rocm_core::HostGpuSummary::default()
        };

        let selection = select_serve_engine(None, None, None, Some(&summary));

        // vLLM is unsupported on native Windows, so the GPU-family preference is gated
        // off there and selection falls back to the platform default.
        let expected = if cfg!(windows) {
            ServeEngineSelection {
                engine: "lemonade".to_owned(),
                source: "platform default",
            }
        } else {
            ServeEngineSelection {
                engine: "vllm".to_owned(),
                source: "detected ROCm GPU family prefers vLLM",
            }
        };
        assert_eq!(selection, expected);
    }

    #[test]
    fn serve_engine_selection_keeps_recipe_engine_when_gpu_preference_is_incompatible() {
        // qwen-smoke is a tiny GGUF model that only Lemonade can serve and has no vLLM
        // recipe. Even on a vLLM-preferred GPU it must stay on Lemonade rather than being
        // forced onto vLLM (which cannot load the GGUF and fails to locate the model).
        let recipe = resolve_builtin_model_recipe("qwen-smoke").expect("qwen-smoke recipe");
        let summary = rocm_core::HostGpuSummary {
            therock_family: Some("gfx90a".to_owned()),
            ..rocm_core::HostGpuSummary::default()
        };

        let selection = select_serve_engine(None, None, Some(&recipe), Some(&summary));

        assert_eq!(
            selection,
            ServeEngineSelection {
                engine: "lemonade".to_owned(),
                source: "recipe preferred engine; pass --engine <engine> to override; no automatic fallback",
            }
        );
    }

    #[test]
    fn aim_catalogued_hf_ids_pick_aim_only_on_a_gpu_it_supports() {
        // These ids had no recipe before the AIM catalog; cataloguing them must not
        // move an Instinct host off vLLM, nor send a host the image has no profile
        // for (or an undetected GPU, or native Windows) to an engine that refuses it.
        let host = |family: Option<&str>| rocm_core::HostGpuSummary {
            therock_family: family.map(str::to_owned),
            ..rocm_core::HostGpuSummary::default()
        };
        let instinct = host(Some("gfx94X-dcgpu"));
        let radeon = host(Some("gfx110X-all"));
        let strix = host(Some("gfx1151"));
        let undetected = host(None);

        // An Instinct-only image (`Qwen/Qwen3-32B`) and a Radeon one
        // (`Qwen/Qwen3.5-9B`) each get AIM on their own family and their next
        // listed engine (vLLM) elsewhere, or the platform default on Windows.
        let instinct_image = resolve_builtin_model_recipe("Qwen/Qwen3-32B").unwrap();
        let radeon_image = resolve_builtin_model_recipe("Qwen/Qwen3.5-9B").unwrap();
        for (recipe, family, windows, expected) in [
            (&instinct_image, &instinct, false, Some("aim")),
            (&instinct_image, &instinct, true, None),
            (&instinct_image, &radeon, false, Some("vllm")),
            (&instinct_image, &strix, false, Some("vllm")),
            (&instinct_image, &undetected, false, Some("vllm")),
            (&radeon_image, &radeon, false, Some("aim")),
            (&radeon_image, &radeon, true, None),
            (&radeon_image, &instinct, false, Some("vllm")),
            (&radeon_image, &strix, false, Some("vllm")),
        ] {
            assert_eq!(
                recipe_preferred_engine(recipe, Some(family), windows),
                expected,
                "{} on {:?} (windows: {windows})",
                recipe.canonical_model_id,
                family.therock_family
            );
        }
        // No GPU summary at all is treated like an undetected GPU.
        assert_eq!(
            recipe_preferred_engine(&radeon_image, None, false),
            Some("vllm")
        );
        // Recipes that do not prefer AIM are untouched.
        let vllm_only = resolve_builtin_model_recipe("meta-llama/Llama-3.2-3B-Instruct").unwrap();
        assert_eq!(
            recipe_preferred_engine(&vllm_only, Some(&strix), false),
            Some("vllm")
        );
        // A recipe that lists vLLM first keeps it on native Windows (the skip
        // applies only past an unusable AIM first choice), as it does on main.
        assert_eq!(
            recipe_preferred_engine(&vllm_only, Some(&strix), true),
            Some("vllm")
        );

        // End to end through the selector. A vLLM-preferred host keeps vLLM; the
        // rest follow the table above (the platform default is OS-specific, so
        // compare against it rather than a literal).
        let on_instinct = select_serve_engine(None, None, Some(&instinct_image), Some(&instinct));
        if !cfg!(windows) {
            assert_eq!(on_instinct.engine, "vllm");
            let on_radeon = select_serve_engine(None, None, Some(&radeon_image), Some(&radeon));
            assert_eq!(on_radeon.engine, "aim");
        }
        let on_strix = select_serve_engine(None, None, Some(&radeon_image), Some(&strix));
        // AIM has no profile for Strix Halo, so the recipe's next engine is used;
        // native Windows has no vLLM either, so it takes the platform default.
        assert_eq!(
            on_strix.engine,
            if cfg!(windows) {
                default_engine_for_platform()
            } else {
                "vllm"
            }
        );
        // An explicit engine is never overridden.
        let explicit = select_serve_engine(Some("aim"), None, Some(&radeon_image), Some(&strix));
        assert_eq!(explicit.engine, "aim");
    }

    #[test]
    fn serve_qwen_uses_vllm_with_hf_repo_on_vllm_preferred_gpu() {
        // The qwen alias serves the GGUF via Lemonade by default, but on a vLLM-preferred
        // GPU it must serve the non-GGUF Hugging Face repo through vLLM.
        let recipe = resolve_builtin_model_recipe("qwen").expect("qwen recipe");
        let summary = rocm_core::HostGpuSummary {
            therock_family: Some("gfx94X-dcgpu".to_owned()),
            ..rocm_core::HostGpuSummary::default()
        };

        let selection = select_serve_engine(None, None, Some(&recipe), Some(&summary));
        // On native Windows the vLLM preference is gated off, so the qwen recipe stays on
        // its own preferred engine (Lemonade) instead of being routed to vLLM.
        let expected = if cfg!(windows) {
            ServeEngineSelection {
                engine: "lemonade".to_owned(),
                source: "recipe preferred engine; pass --engine <engine> to override; no automatic fallback",
            }
        } else {
            ServeEngineSelection {
                engine: "vllm".to_owned(),
                source: "detected ROCm GPU family prefers vLLM",
            }
        };
        assert_eq!(selection, expected);
        assert_eq!(
            serve_model_ref_for_engine("qwen", Some(&recipe), "vllm"),
            "Qwen/Qwen3-4B-Instruct-2507"
        );
        // Lemonade keeps the GGUF canonical id.
        assert_eq!(
            serve_model_ref_for_engine("qwen", Some(&recipe), "lemonade"),
            "Qwen3-4B-Instruct-2507-GGUF"
        );
    }

    #[test]
    fn serve_model_ref_for_engine_pins_the_shipped_aim_image_to_its_digest() {
        let recipe = resolve_builtin_model_recipe("aim-qwen3.5-9b").expect("aim catalog entry");
        let pinned = serve_model_ref_for_engine("aim-qwen3.5-9b", Some(&recipe), "aim");
        let (image, digest) = pinned
            .split_once('@')
            .expect("the AIM entry must be pinned");
        assert_eq!(
            image,
            "amdenterpriseai/aim-radeon-qwen-qwen3-5-9b:0.12.0-preview"
        );
        assert!(
            digest.starts_with("sha256:") && digest.len() == "sha256:".len() + 64,
            "{digest}"
        );
        // Other engines are unaffected by the AIM entry's pin.
        let qwen32b = resolve_builtin_model_recipe("qwen32b").expect("qwen32b recipe");
        assert_eq!(
            serve_model_ref_for_engine("qwen32b", Some(&qwen32b), "vllm"),
            "Qwen/Qwen3-32B-FP8"
        );
    }

    #[test]
    fn catalog_profile_pin_reaches_the_aim_hint_and_engine_env_overrides_it() {
        let recipe = resolve_builtin_model_recipe("aim-qwen3.5-9b").expect("aim catalog entry");
        let hint = protocol_engine_recipe_hint(&recipe, "aim");
        let pinned = |family: &'static str| {
            engine_recipe_with_aim_gfx_profile("aim", Some(&recipe), hint.clone(), || {
                Some(family.to_owned())
            })
            .unwrap()
            .unwrap()
        };
        let w7900 = pinned("gfx110X-all");
        assert_eq!(
            w7900.extra_env.get("AIM_PROFILE_ID").map(String::as_str),
            Some("vllm-w7900-bf16-tp1-latency")
        );
        assert_eq!(
            pinned("gfx120X-all")
                .extra_env
                .get("AIM_PROFILE_ID")
                .map(String::as_str),
            Some("vllm-r9700-bf16-tp1-latency")
        );

        let overridden = engine_recipe_with_extra_env(
            "aim",
            Some(w7900),
            &[
                "AIM_PROFILE_ID=vllm-other".to_owned(),
                "HF_TOKEN".to_owned(),
            ],
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            overridden
                .extra_env
                .get("AIM_PROFILE_ID")
                .map(String::as_str),
            Some("vllm-other"),
            "`--engine-env` must beat the catalog's value"
        );
        assert_eq!(overridden.extra_env.get("HF_TOKEN"), Some(&String::new()));
    }

    #[test]
    fn aim_serve_refuses_a_gpu_the_image_has_no_profile_for() {
        let radeon = resolve_builtin_model_recipe("aim-qwen3.5-9b").expect("radeon aim entry");
        let hint = protocol_engine_recipe_hint(&radeon, "aim");
        let err = engine_recipe_with_aim_gfx_profile("aim", Some(&radeon), hint, || {
            Some("gfx94X-dcgpu".to_owned())
        })
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("no profile for this GPU (gfx94X-dcgpu)"),
            "{err}"
        );
        assert!(err.contains("gfx110X-all, gfx120X-all"), "{err}");

        // Strix Halo has no AIM profile either.
        let hint = protocol_engine_recipe_hint(&radeon, "aim");
        assert!(
            engine_recipe_with_aim_gfx_profile("aim", Some(&radeon), hint, || Some(
                "gfx1151".to_owned()
            ))
            .is_err()
        );

        // Instinct images serve on Instinct without pinning a profile (AIM
        // auto-selects), and are refused on a Radeon.
        let instinct = resolve_builtin_model_recipe("Qwen/Qwen3-32B").expect("instinct aim entry");
        let hint = protocol_engine_recipe_hint(&instinct, "aim");
        let served =
            engine_recipe_with_aim_gfx_profile("aim", Some(&instinct), hint.clone(), || {
                Some("gfx94X-dcgpu".to_owned())
            })
            .unwrap()
            .unwrap();
        assert!(!served.extra_env.contains_key("AIM_PROFILE_ID"));
        assert!(
            engine_recipe_with_aim_gfx_profile("aim", Some(&instinct), hint, || Some(
                "gfx110X-all".to_owned()
            ))
            .is_err()
        );
    }

    #[test]
    fn gated_aim_model_needs_an_hf_token_before_launch() {
        let gated = resolve_builtin_model_recipe("meta-llama/Llama-3.2-3B-Instruct")
            .expect("gated aim entry");
        let hint = |engine_env: &[&str]| {
            engine_recipe_with_extra_env(
                "aim",
                protocol_engine_recipe_hint(&gated, "aim"),
                &engine_env
                    .iter()
                    .map(|s| (*s).to_owned())
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        };
        let no_env = |_: &str| None;
        let exported = |_: &str| Some("hf_x".to_owned());

        // Without the flag, or with the bare flag but nothing exported, refuse and
        // name the command that clears it.
        for env in [&no_env as &dyn Fn(&str) -> Option<String>, &exported] {
            let err = ensure_aim_hf_token("aim", Some(&gated), hint(&[]).as_ref(), env)
                .unwrap_err()
                .to_string();
            assert!(err.contains("--engine-env HF_TOKEN"), "{err}");
        }
        assert!(
            ensure_aim_hf_token("aim", Some(&gated), hint(&["HF_TOKEN"]).as_ref(), no_env).is_err()
        );
        // The advised form (bare flag with the variable exported) or an explicit
        // value clears it, and other engines are never checked.
        assert!(
            ensure_aim_hf_token("aim", Some(&gated), hint(&["HF_TOKEN"]).as_ref(), exported)
                .is_ok()
        );
        assert!(
            ensure_aim_hf_token(
                "aim",
                Some(&gated),
                hint(&["HF_TOKEN=hf_y"]).as_ref(),
                no_env
            )
            .is_ok()
        );
        assert!(ensure_aim_hf_token("vllm", Some(&gated), None, no_env).is_ok());
        let open = resolve_builtin_model_recipe("Qwen/Qwen3-32B").expect("ungated aim entry");
        assert!(ensure_aim_hf_token("aim", Some(&open), None, no_env).is_ok());
    }

    #[test]
    fn multi_gpu_aim_image_is_refused_before_launch() {
        let id = "deepseek-ai/DeepSeek-R1";
        let r1 = resolve_builtin_model_recipe(id).expect("multi-gpu aim entry");
        assert_eq!(
            aim_min_gpu_count("aim", Some(&r1), Some("gfx94X-dcgpu")),
            Some(8)
        );
        assert_eq!(aim_min_gpu_count("aim", Some(&r1), Some("gfx1151")), None);
        assert_eq!(aim_min_gpu_count("aim", Some(&r1), None), None);
        assert_eq!(
            aim_min_gpu_count("vllm", Some(&r1), Some("gfx94X-dcgpu")),
            None
        );
        // A single-GPU image has no requirement.
        let small = resolve_builtin_model_recipe("Qwen/Qwen3-32B").expect("single-gpu entry");
        assert_eq!(
            aim_min_gpu_count("aim", Some(&small), Some("gfx94X-dcgpu")),
            None
        );
        // Radeon needs differ per family.
        let flash = resolve_builtin_model_recipe("zai-org/GLM-4.7-Flash").expect("flash entry");
        assert_eq!(
            aim_min_gpu_count("aim", Some(&flash), Some("gfx110X-all")),
            Some(2)
        );
        assert_eq!(
            aim_min_gpu_count("aim", Some(&flash), Some("gfx120X-all")),
            Some(4)
        );

        let err = ensure_aim_multi_gpu_supported(id, 8)
            .unwrap_err()
            .to_string();
        assert!(err.contains("needs 8 GPUs") && err.contains("does not support multi-GPU"));
    }

    #[test]
    fn aim_board_note_only_fires_when_the_catalog_profile_is_what_runs() {
        let radeon = resolve_builtin_model_recipe("aim-qwen3.5-9b").expect("radeon aim entry");
        // The hint the serve path ends up with once a GPU family was detected.
        let pinned = engine_recipe_with_aim_gfx_profile(
            "aim",
            Some(&radeon),
            protocol_engine_recipe_hint(&radeon, "aim"),
            || Some("gfx110X-all".to_owned()),
        )
        .unwrap();
        let note = |gpu: Option<&str>, hint: Option<&EngineRecipeHint>| {
            aim_board_note("aim", Some(&radeon), gpu, hint)
        };
        assert!(note(Some("AMD Radeon PRO W7900"), pinned.as_ref()).is_none());
        assert!(note(Some("AMD Radeon AI PRO R9700"), pinned.as_ref()).is_none());
        let other = note(Some("AMD Radeon RX 7900 XTX"), pinned.as_ref()).expect("warns");
        assert!(
            other.contains("vllm-w7900-bf16-tp1-latency")
                && other.contains("W7900 / R9700")
                && other.contains("--engine-env AIM_PROFILE_ID"),
            "{other}"
        );
        assert!(note(None, pinned.as_ref()).is_none());

        // No pin applied (family undetected, or `--device cpu_only`): no claim about it.
        let unpinned = engine_recipe_with_aim_gfx_profile(
            "aim",
            Some(&radeon),
            protocol_engine_recipe_hint(&radeon, "aim"),
            || None,
        )
        .unwrap();
        assert!(note(Some("AMD Radeon RX 7900 XTX"), unpinned.as_ref()).is_none());
        assert!(note(Some("AMD Radeon RX 7900 XTX"), None).is_none());
        // An explicit AIM_PROFILE_ID replaced the catalog's: its board is the user's call.
        let overridden =
            engine_recipe_with_extra_env("aim", pinned, &["AIM_PROFILE_ID=vllm-other".to_owned()])
                .unwrap();
        assert!(note(Some("AMD Radeon RX 7900 XTX"), overridden.as_ref()).is_none());
        // Instinct images pin no profile, so there is nothing to mismatch.
        let instinct = resolve_builtin_model_recipe("Qwen/Qwen3-32B").expect("instinct entry");
        assert!(aim_board_note("aim", Some(&instinct), Some("anything"), None).is_none());
    }

    #[test]
    fn aim_gpu_preflight_probes_only_for_a_restricted_aim_record_off_cpu_only() {
        let radeon = resolve_builtin_model_recipe("aim-qwen3.5-9b").expect("radeon aim entry");
        let hint = protocol_engine_recipe_hint(&radeon, "aim");
        let preflight = |engine: &str, cpu_only: bool, family: fn() -> Option<String>| {
            aim_gpu_preflight(engine, "m", Some(&radeon), hint.clone(), cpu_only, family)
        };
        let no_probe = || -> Option<String> { panic!("host GPU must not be probed") };

        // An undetected family enforces nothing.
        assert_eq!(preflight("aim", false, || None).unwrap(), hint);
        // The probe runs for AIM off `cpu_only`, and the result is enforced.
        assert!(preflight("aim", false, || Some("gfx94X-dcgpu".to_owned())).is_err());
        // Neither a different engine nor `--device cpu_only` reaches the probe,
        // so an unsupported family cannot refuse them.
        assert_eq!(preflight("vllm", false, no_probe).unwrap(), hint);
        assert_eq!(preflight("aim", true, no_probe).unwrap(), hint);
    }

    #[test]
    fn aim_gpu_preflight_pins_the_profile_and_then_refuses_multi_gpu_images() {
        let radeon = resolve_builtin_model_recipe("aim-qwen3.5-9b").expect("radeon aim entry");
        let pinned = aim_gpu_preflight(
            "aim",
            "m",
            Some(&radeon),
            protocol_engine_recipe_hint(&radeon, "aim"),
            false,
            || Some("gfx110X-all".to_owned()),
        )
        .unwrap()
        .expect("hint");
        assert_eq!(
            pinned.extra_env.get("AIM_PROFILE_ID").map(String::as_str),
            Some("vllm-w7900-bf16-tp1-latency")
        );

        // The multi-GPU refusal comes from the same pre-flight, and only for a
        // family whose count exceeds one.
        let multi = resolve_builtin_model_recipe("zai-org/GLM-4.7-Flash").expect("multi-GPU entry");
        let hint = || protocol_engine_recipe_hint(&multi, "aim");
        let err = aim_gpu_preflight("aim", "m", Some(&multi), hint(), false, || {
            Some("gfx120X-all".to_owned())
        })
        .unwrap_err();
        assert!(err.to_string().contains("multi-GPU serving yet"), "{err:#}");
        assert!(
            aim_gpu_preflight("aim", "m", Some(&multi), hint(), true, || None).is_ok(),
            "cpu_only skips the multi-GPU check"
        );
    }

    #[test]
    fn recipe_preferred_engine_treats_unrestricted_and_missing_aim_records_as_usable() {
        let host = rocm_core::HostGpuSummary {
            therock_family: Some("gfx1151".to_owned()),
            ..rocm_core::HostGpuSummary::default()
        };
        let mut recipe = resolve_builtin_model_recipe("Qwen/Qwen3-32B").unwrap();
        assert_eq!(
            recipe_preferred_engine(&recipe, Some(&host), false),
            Some("vllm")
        );
        // Only a signed external index reaches these two arms, so they are built
        // here: an AIM record that restricts no family is usable on any GPU, and
        // a recipe that lists AIM first but carries no AIM record has nothing to
        // gate it.
        for record in &mut recipe.engine_recipes {
            if record.engine == "aim" {
                record.supported_gfx_families.clear();
            }
        }
        assert_eq!(
            recipe_preferred_engine(&recipe, Some(&host), false),
            Some("aim")
        );
        recipe
            .engine_recipes
            .retain(|record| record.engine != "aim");
        assert_eq!(
            recipe_preferred_engine(&recipe, Some(&host), false),
            Some("aim")
        );
        assert_eq!(recipe_preferred_engine(&recipe, Some(&host), true), None);
    }

    #[test]
    fn explicit_engine_override_keeps_alias_when_shared_recipe_is_for_another_engine() {
        // `qwen-smoke` is a Lemonade-only GGUF recipe (no vLLM engine recipe).
        let recipe = resolve_builtin_model_recipe("qwen-smoke").expect("qwen-smoke recipe");

        // Served under the engine it targets, the alias resolves to the canonical id.
        assert_eq!(
            serve_model_ref_for_engine("qwen-smoke", Some(&recipe), "lemonade"),
            "Qwen3-0.6B-GGUF"
        );
        // Under an engine the recipe does not support, the raw alias flows through unchanged.
        assert_eq!(
            serve_model_ref_for_engine("qwen-smoke", Some(&recipe), "vllm"),
            "qwen-smoke"
        );
    }

    #[test]
    fn serve_model_ref_for_engine_ignores_blank_image_digest() {
        let mut recipe = resolve_builtin_model_recipe("qwen32b").expect("qwen32b recipe");
        recipe
            .engine_recipes
            .iter_mut()
            .find(|engine_recipe| engine_recipe.engine == "vllm")
            .expect("vllm engine recipe")
            .image_digest = Some("   ".to_owned());

        assert_eq!(
            serve_model_ref_for_engine("qwen32b", Some(&recipe), "vllm"),
            "Qwen/Qwen3-32B-FP8"
        );
    }

    #[test]
    fn serve_engine_selection_respects_explicit_and_configured_engines() {
        let recipe = resolve_builtin_model_recipe("qwen32b").expect("qwen32b recipe");

        let explicit = select_serve_engine(Some("vllm"), Some("lemonade"), Some(&recipe), None);
        let configured = select_serve_engine(None, Some("lemonade"), Some(&recipe), None);

        assert_eq!(
            explicit,
            ServeEngineSelection {
                engine: "vllm".to_owned(),
                source: "explicit --engine",
            }
        );
        assert_eq!(
            configured,
            ServeEngineSelection {
                engine: "lemonade".to_owned(),
                source: "configured default_engine",
            }
        );
    }

    #[test]
    fn serve_notes_surface_the_ignored_engine_flag_in_summary_mode() {
        // The interactive summary is what a default `rocm serve` prints, so a flag
        // the selected engine cannot honor has to be reported through this path —
        // not only on the plan path that an interactive run never takes.
        let note = "--gpu-memory-utilization applies only to vLLM; ignored for engine 'lemonade'";
        let notes = collect_serve_notes(
            false,
            &GpuSelection::Auto,
            false,
            &[0],
            None,
            Some(note),
            None,
            false,
        );
        assert!(
            notes.iter().any(|entry| entry == note),
            "the ignored-flag note must reach the summary: {notes:?}"
        );

        let quiet = collect_serve_notes(
            false,
            &GpuSelection::Auto,
            false,
            &[0],
            None,
            None,
            None,
            false,
        );
        assert!(
            !quiet
                .iter()
                .any(|entry| entry.contains("--gpu-memory-utilization")),
            "nothing to report when the flag was honored: {quiet:?}"
        );
    }

    /// Duplicated from `gpu_policy::tests` (ROCMAI-214): both this test module
    /// and `gpu_policy.rs`'s own build `GpuVramUsage` fixtures, and this helper
    /// has no other home now that `gpu_policy.rs` owns the type.
    fn vram(index: u32, used_mb: u64, total_mb: u64) -> GpuVramUsage {
        GpuVramUsage {
            index,
            used_mb,
            total_mb,
        }
    }

    /// Duplicated from `gpu_policy::tests` (ROCMAI-214) for the same reason as
    /// `vram` above.
    fn host_gpu(gfx_target: &str) -> rocm_core::HostGpuSummary {
        rocm_core::HostGpuSummary {
            gfx_target: Some(gfx_target.to_owned()),
            ..rocm_core::HostGpuSummary::default()
        }
    }

    #[test]
    fn serve_notes_pair_low_vram_with_the_vllm_utilization_hint() {
        // A busy discrete card that trips the low-VRAM warning: on vLLM the note
        // must also carry the concrete `--gpu-memory-utilization` workaround, so
        // the interactive summary tells the user how to avoid the OOM.
        let busy = [vram(0, 182_000, 192_000)];
        let notes = collect_serve_notes(
            false,
            &GpuSelection::Auto,
            false,
            &[0],
            Some(&busy),
            None,
            Some(&host_gpu("gfx1100")),
            true,
        );
        assert!(
            notes.iter().any(|entry| entry.contains("has only")),
            "the low-VRAM warning must be present: {notes:?}"
        );
        assert!(
            notes
                .iter()
                .any(|entry| entry.contains("--gpu-memory-utilization")),
            "vLLM low-VRAM notes must hint the utilization workaround: {notes:?}"
        );

        // The literal fragments the GPU-lane scenario
        // `@id:serve-vllm-low-vram-oom-guidance` matches on. That scenario runs
        // only where a real card exists, so pin the wording here too: a rename
        // then fails on every lane rather than silently on the one lane that can
        // observe it. `GPU 0 has only` is what makes the warning provably about
        // the *pinned* device; `vLLM reserves ~90%` is what distinguishes the
        // pre-launch hint from any other line that merely names the flag.
        assert!(
            notes.iter().any(|entry| entry.contains("GPU 0 has only")),
            "the warning must name the selected GPU: {notes:?}"
        );
        assert!(
            notes
                .iter()
                .any(|entry| entry.contains("vLLM reserves ~90%")),
            "the hint must explain vLLM's total-VRAM reservation: {notes:?}"
        );

        // The same busy card on a non-vLLM engine keeps the warning but omits the
        // vLLM-only knob, which that engine cannot honor.
        let lemonade = collect_serve_notes(
            false,
            &GpuSelection::Auto,
            false,
            &[0],
            Some(&busy),
            None,
            Some(&host_gpu("gfx1100")),
            false,
        );
        assert!(
            lemonade.iter().any(|entry| entry.contains("has only")),
            "the low-VRAM warning still fires for other engines: {lemonade:?}"
        );
        assert!(
            !lemonade
                .iter()
                .any(|entry| entry.contains("--gpu-memory-utilization")),
            "non-vLLM engines must not be told to pass a vLLM-only flag: {lemonade:?}"
        );
    }
}
