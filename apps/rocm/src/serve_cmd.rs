// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm serve` command handler and engine-recipe overrides.
//!
//! Originally a mechanical relocation from `main.rs` with no behavior change —
//! the `dispatch()` call site stays byte-identical (`serve(ServeArgs { .. })`,
//! re-imported via `use crate::serve_cmd::{ServeArgs, select_serve_engine, serve};`
//! — `select_serve_engine` rides along for `assess_model_for_host`, not dispatch).
//! `Cli` is not referenced from this file; `DevicePolicyArg` (part of the clap arg tree)
//! is, via `use crate::DevicePolicyArg`.
//!
//! That "no behavior change" describes the move itself, not the file's history
//! since: EAI-8059 added the model-keyed reuse pre-gate below, which is new
//! serve behaviour and the one thing that can run ahead of the no-usable-GPU
//! bail. Read the comment at the pre-gate, not this paragraph, for what the
//! serve path does today.
//!
//! `collect_serve_notes` and `validate_pinned_gpu_index` stay in `main.rs`
//! (no entanglement; a later pass can relocate them). The
//! managed-service-spawning tail this file calls into — `start_managed_service`,
//! `run_attached_service`, `resolve_endpoint_auth`,
//! `ensure_public_bind_engine_supported`,
//! `drop_orphaned_endpoint_key_on_already_running`, `print_managed_launch_plain`,
//! `validate_bind_host`, `ensure_background_helper_running_quiet` — was relocated
//! to `service_launch.rs` in ROCMAI-91 (Phase 6b), along with its test module;
//! this file's `use` list below points there now instead of at the crate root.

use std::fmt::Write as _;

use anyhow::{Context as _, Result, bail};
use rocm_core::{
    AppPaths, ModelRecipeRecord, RocmCliConfig, default_engine_for_platform,
    detect_host_gpu_summary, generate_service_id, preferred_serve_engine_for_host_gpu_summary,
    resolve_model_recipe,
};
use rocm_engine_protocol::{
    DevicePolicy, ENGINE_RECIPE_CONTRACT_VERSION, EngineMethod, EngineRecipeEndpointHint,
    EngineRecipeHint, EngineRecipeUnsupportedCombinationHint, GpuSelection, ResolveModelRequest,
    ResolveModelResponse,
};

use crate::DevicePolicyArg;
use crate::endpoint_keys;
use crate::engines_cmd::{engine_manages_own_runtime, ensure_self_managed_engine_ready};
use crate::serve_summary;
use crate::service_launch::{
    drop_orphaned_endpoint_key_on_already_running, ensure_background_helper_running_quiet,
    ensure_public_bind_engine_supported, managed_service_reuse_refusal, print_managed_launch_plain,
    resolve_endpoint_auth, run_attached_service, start_managed_service, validate_bind_host,
};
use crate::{
    EngineSelection, any_live_managed_service_for_model, cli_progress, collect_serve_notes,
    detect_gpu_count, device_policy_name, engine_request, existing_live_managed_service,
    gpu_vram_usage, parse_device_policy, parse_gpu_selection, resolve_engine_selection,
    select_gpu_indices_under_launch_lock, serve_gpu_low_memory_warning,
    validate_engine_selection_runtime, validate_pinned_gpu_index,
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
        .and_then(|recipe| recipe.preferred_engines.first())
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
    if let Some(override_id) = recipe
        .engine_recipes
        .iter()
        .find(|engine_recipe| engine_recipe.engine.eq_ignore_ascii_case(selected_engine))
        .and_then(|engine_recipe| engine_recipe.model_id_override.as_deref())
        .filter(|value| !value.trim().is_empty())
    {
        return override_id.to_owned();
    }
    recipe.canonical_model_id.clone()
}

fn serve_engine_selection_line(selection: &ServeEngineSelection) -> String {
    format!("  engine_selection: {}", selection.source)
}

fn render_serve_engine_recipe_lines(engine_recipe: &EngineRecipeHint) -> String {
    let mut output = String::new();
    let _ = writeln!(
        output,
        "  engine_recipe_contract: {}",
        engine_recipe.contract_version
    );
    let _ = writeln!(
        output,
        "  engine_recipe_policy: selected-engine required_flags are applied at launch; parser/endpoint metadata is forwarded to the adapter"
    );
    let _ = writeln!(output, "  engine_recipe_engine: {}", engine_recipe.engine);
    if !engine_recipe.required_flags.is_empty() {
        let _ = writeln!(
            output,
            "  engine_recipe_required_flags: {}",
            engine_recipe.required_flags.join(" ")
        );
    }
    output
}

fn protocol_engine_recipe_hint(
    recipe: &ModelRecipeRecord,
    engine: &str,
) -> Option<EngineRecipeHint> {
    recipe
        .engine_recipes
        .iter()
        .find(|engine_recipe| engine_recipe.engine == engine)
        .map(|engine_recipe| EngineRecipeHint {
            contract_version: ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
            engine: engine_recipe.engine.clone(),
            required_flags: engine_recipe.required_flags.clone(),
            parser_settings: engine_recipe.parser_settings.clone(),
            preferred_endpoint: engine_recipe.preferred_endpoint.as_ref().map(|endpoint| {
                EngineRecipeEndpointHint {
                    endpoint_mode: endpoint.endpoint_mode.clone(),
                    settings: endpoint.settings.clone(),
                }
            }),
            unsupported_combinations: engine_recipe
                .unsupported_combinations
                .iter()
                .map(|combination| EngineRecipeUnsupportedCombinationHint {
                    combination: combination.combination.clone(),
                    reason: combination.reason.clone(),
                })
                .collect(),
            notes: engine_recipe.notes.clone(),
        })
}

/// Applies an explicit `--tool-call-parser` override to a vLLM engine recipe hint.
///
/// The TUI chat tab always attaches tool definitions to non-streaming chat
/// requests (`tool_choice: "auto"`). vLLM rejects those with HTTP 400 unless it was
/// started with `--enable-auto-tool-choice` *and* a matching `--tool-call-parser`.
/// The correct parser is model-specific and vLLM does not auto-detect it, so it is
/// never guessed from the model ref: it comes either from authored catalog recipe
/// metadata (already carried in `required_flags`) or from the explicit
/// `--tool-call-parser` serve flag, which this applies.
///
/// Only vLLM is affected. When an override is supplied it wins over any
/// recipe-authored parser (a single `--tool-call-parser`, no duplication) and a
/// minimal hint is synthesized when none exists (arbitrary HF repos, or a catalog
/// model forced onto a non-preferred engine). With no override the hint passes
/// through unchanged.
fn engine_recipe_with_tool_call_override(
    engine: &str,
    hint: Option<EngineRecipeHint>,
    tool_call_parser: Option<&str>,
) -> Option<EngineRecipeHint> {
    if !engine.eq_ignore_ascii_case("vllm") {
        return hint;
    }
    let Some(parser) = tool_call_parser
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return hint;
    };
    let mut hint = hint.unwrap_or_else(|| EngineRecipeHint {
        contract_version: ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
        engine: engine.to_owned(),
        ..EngineRecipeHint::default()
    });
    set_vllm_tool_call_parser(&mut hint.required_flags, parser);
    Some(hint)
}

/// Sampling defaults a `rocm serve` invocation can push into a vLLM launch.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct ServeGenerationDefaults {
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
}

impl ServeGenerationDefaults {
    const fn is_empty(&self) -> bool {
        self.temperature.is_none() && self.top_p.is_none() && self.max_tokens.is_none()
    }
}

/// Applies `rocm serve` generation defaults (`--temperature`/`--top-p`/`--max-tokens`)
/// to the selected engine's launch recipe.
///
/// vLLM `serve` has no raw `--temperature`/`--top-p` flags; `--override-generation-config`
/// is the supported way to set server-wide sampling defaults, so `--max-tokens` is
/// mapped onto vLLM's `max_new_tokens` output cap. Only supplied values are written,
/// and any values already carried by an authored recipe's
/// `--override-generation-config` are preserved (the CLI-supplied keys win).
///
/// Lemonade's llama.cpp backend accepts the equivalent `--temperature`, `--top-p`,
/// and `--n-predict` launch flags. A minimal hint is synthesized when none exists.
/// With no defaults supplied the hint passes through unchanged.
fn engine_recipe_with_generation_defaults(
    engine: &str,
    hint: Option<EngineRecipeHint>,
    defaults: ServeGenerationDefaults,
) -> Result<Option<EngineRecipeHint>> {
    if defaults.is_empty() {
        return Ok(hint);
    }
    let mut hint = hint.unwrap_or_else(|| EngineRecipeHint {
        contract_version: ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
        engine: engine.to_owned(),
        ..EngineRecipeHint::default()
    });
    if engine.eq_ignore_ascii_case("vllm") {
        let mut overrides = serde_json::Map::new();
        if let Some(temperature) = defaults.temperature {
            overrides.insert("temperature".to_owned(), serde_json::json!(temperature));
        }
        if let Some(top_p) = defaults.top_p {
            overrides.insert("top_p".to_owned(), serde_json::json!(top_p));
        }
        if let Some(max_tokens) = defaults.max_tokens {
            overrides.insert("max_new_tokens".to_owned(), serde_json::json!(max_tokens));
        }
        set_vllm_override_generation_config(&mut hint.required_flags, &overrides);
    } else if engine.eq_ignore_ascii_case("lemonade") {
        set_lemonade_generation_defaults(&mut hint.required_flags, defaults);
    } else {
        bail!(
            "generation defaults are not supported by engine `{engine}`; omit --temperature/--top-p/--max-tokens or select vllm/lemonade"
        );
    }
    Ok(Some(hint))
}

fn set_lemonade_generation_defaults(flags: &mut Vec<String>, defaults: ServeGenerationDefaults) {
    // Only touch a flag pair when the caller actually supplied that control —
    // an unset field must leave any authored recipe value in place rather than
    // deleting it, mirroring the vLLM merge semantics in
    // `set_vllm_override_generation_config`.
    for (name, value) in [
        (
            "--temperature",
            defaults.temperature.map(|value| value.to_string()),
        ),
        ("--top-p", defaults.top_p.map(|value| value.to_string())),
        (
            "--n-predict",
            defaults.max_tokens.map(|value| value.to_string()),
        ),
    ] {
        let Some(value) = value else {
            continue;
        };
        let mut rewritten = Vec::with_capacity(flags.len() + 2);
        let mut skip_value = false;
        for flag in std::mem::take(flags) {
            if skip_value {
                skip_value = false;
                continue;
            }
            if flag == name {
                skip_value = true;
            } else {
                rewritten.push(flag);
            }
        }
        rewritten.extend([name.to_owned(), value]);
        *flags = rewritten;
    }
}

/// Rewrites `flags` so vLLM's `--override-generation-config` carries exactly one
/// merged JSON object: any existing `--override-generation-config <value>` pair is
/// removed, its keys are used as a base, and `overrides` are layered on top (CLI
/// values win). Emits a single flag pair with the merged, stably-ordered config.
fn set_vllm_override_generation_config(
    flags: &mut Vec<String>,
    overrides: &serde_json::Map<String, serde_json::Value>,
) {
    let existing = std::mem::take(flags);
    let mut rewritten: Vec<String> = Vec::with_capacity(existing.len() + 2);
    let mut merged = serde_json::Map::new();
    let mut take_value = false;
    for flag in existing {
        if take_value {
            take_value = false;
            if let Ok(serde_json::Value::Object(existing_config)) =
                serde_json::from_str::<serde_json::Value>(&flag)
            {
                for (key, value) in existing_config {
                    merged.insert(key, value);
                }
            } else {
                eprintln!(
                    "warning: existing --override-generation-config value is not valid JSON; discarding it"
                );
            }
            continue;
        }
        if flag == "--override-generation-config" {
            take_value = true;
            continue;
        }
        rewritten.push(flag);
    }
    for (key, value) in overrides {
        merged.insert(key.clone(), value.clone());
    }
    rewritten.push("--override-generation-config".to_owned());
    rewritten.push(serde_json::Value::Object(merged).to_string());
    *flags = rewritten;
}

/// Rewrites `flags` so vLLM tool calling uses exactly `parser`: drops any existing
/// `--tool-call-parser <value>` pair, ensures `--enable-auto-tool-choice` is
/// present, then appends the new parser flag.
fn set_vllm_tool_call_parser(flags: &mut Vec<String>, parser: &str) {
    let existing = std::mem::take(flags);
    let mut rewritten: Vec<String> = Vec::with_capacity(existing.len() + 3);
    let mut skip_value = false;
    for flag in existing {
        if skip_value {
            // Drop the value that followed the removed `--tool-call-parser`.
            skip_value = false;
            continue;
        }
        if flag == "--tool-call-parser" {
            skip_value = true;
            continue;
        }
        rewritten.push(flag);
    }
    if !rewritten
        .iter()
        .any(|flag| flag == "--enable-auto-tool-choice")
    {
        rewritten.push("--enable-auto-tool-choice".to_owned());
    }
    rewritten.push("--tool-call-parser".to_owned());
    rewritten.push(parser.to_owned());
    *flags = rewritten;
}

/// Applies an explicit `--gpu-memory-utilization` to the vLLM engine recipe.
///
/// rocm-cli intentionally ships no default for this: vLLM sizes its KV cache as
/// a fraction of the device's TOTAL VRAM, and any number rocm-cli picked would
/// silently override upstream's and drift from it. So the flag is passed through
/// only when the user asked for one, via `required_flags` (the same channel the
/// `--tool-call-parser` override uses — no protocol change needed).
///
/// Only vLLM is affected. An explicit value wins over any recipe-authored one,
/// and a minimal hint is synthesized when none exists.
fn engine_recipe_with_gpu_memory_utilization_override(
    engine: &str,
    hint: Option<EngineRecipeHint>,
    gpu_memory_utilization: Option<f64>,
) -> Option<EngineRecipeHint> {
    if !engine.eq_ignore_ascii_case("vllm") {
        return hint;
    }
    let Some(value) = gpu_memory_utilization else {
        return hint;
    };
    let mut hint = hint.unwrap_or_else(|| EngineRecipeHint {
        contract_version: ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
        engine: engine.to_owned(),
        ..EngineRecipeHint::default()
    });
    set_vllm_gpu_memory_utilization(&mut hint.required_flags, value);
    Some(hint)
}

/// Rewrites `flags` so vLLM receives exactly one `--gpu-memory-utilization
/// <value>` pair: drops any existing pair, then appends the new one.
fn set_vllm_gpu_memory_utilization(flags: &mut Vec<String>, value: f64) {
    let existing = std::mem::take(flags);
    let mut rewritten: Vec<String> = Vec::with_capacity(existing.len() + 2);
    let mut skip_value = false;
    for flag in existing {
        if skip_value {
            // Drop the value that followed the removed flag.
            skip_value = false;
            continue;
        }
        if flag == "--gpu-memory-utilization" {
            skip_value = true;
            continue;
        }
        rewritten.push(flag);
    }
    rewritten.push("--gpu-memory-utilization".to_owned());
    rewritten.push(format!("{value}"));
    *flags = rewritten;
}

/// Parse `rocm serve --gpu-memory-utilization`. Unlike the env-var overrides
/// elsewhere in this file, an explicit CLI value is never silently ignored: a
/// user who types a bad fraction is told so.
fn parse_gpu_memory_utilization(value: Option<&str>) -> Result<Option<f64>> {
    let Some(raw) = value else {
        return Ok(None);
    };
    let trimmed = raw.trim();
    let parsed: f64 = trimmed.parse().map_err(|_| {
        anyhow::anyhow!(
            "--gpu-memory-utilization expects a fraction greater than 0 and at most 1 \
             (e.g. 0.5); got `{trimmed}`"
        )
    })?;
    if !parsed.is_finite() || parsed <= 0.0 || parsed > 1.0 {
        bail!(
            "--gpu-memory-utilization must be greater than 0 and at most 1 (a fraction of \
             the GPU's TOTAL VRAM, e.g. 0.5); got `{trimmed}`"
        );
    }
    Ok(Some(parsed))
}

/// Whether the resolved engine recipe launches vLLM with tool calling enabled.
fn engine_recipe_enables_tool_choice(hint: Option<&EngineRecipeHint>) -> bool {
    hint.is_some_and(|hint| {
        hint.required_flags
            .iter()
            .any(|flag| flag == "--enable-auto-tool-choice")
    })
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
/// `live_service_matches` is taken as a closure, not a `bool`, because it is not
/// free either: it goes through `load_managed_services`, which refreshes
/// liveness and can cost a probe and a `record.write()` per live record. On a
/// host with a GPU that cost is not paid at all.
fn reuse_pregate_opens(
    cpu_only: bool,
    host_has_no_usable_gpu: bool,
    live_service_matches: impl FnOnce() -> bool,
) -> bool {
    !cpu_only && host_has_no_usable_gpu && live_service_matches()
}

/// The no-usable-GPU bail's host precondition, read from the visibility-resolved
/// set of usable AMD GPU ordinals.
///
/// A named function rather than an inline expression because three things have
/// to answer it identically: the bail, the reuse pre-gate whose only product is
/// a term of that bail, and the unit test that pins the pre-gate. The test used
/// to spell the expression out a second time, so changing the real one left it
/// green and only serve-29 — on a self-hosted GPU lane — could notice.
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
    let device_policy = parse_device_policy(device.as_ref().map(|policy| policy.as_policy_str()))?;
    let gpu_selection = parse_gpu_selection(gpu.as_deref())?;
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
    if reuse_pregate_opens(cpu_only, host_has_no_usable_gpu, || {
        any_live_managed_service_for_model(&paths, &selected_engine, &engine_model_ref)
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
            // fails if it is dropped: serve-28 for `require_api_key`, serve-30
            // for `requested_recipe_json`. The device policy cannot be dropped
            // at all — the predicate takes it by value, so passing nothing is a
            // compile error (see `managed_service_reuse_refusal`). A `cpu_only`
            // service must not buy a `gpu_required` serve a bypass of the bail.
            let requested_recipe_json = engine_recipe
                .as_ref()
                .map(serde_json::to_string)
                .transpose()
                .context("failed to encode engine recipe hint")?;
            reuse_existing =
                existing_live_managed_service(&paths, &selected_engine, &probe.canonical_model_id)
                    .is_some_and(|existing| {
                        managed_service_reuse_refusal(
                            &existing,
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
    // short name, which is the worse failure. serve-27 is the regression test
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
    // test reaches Lemonade's backend boundary without real GPU hardware.
    let scripted_backend_failure = cfg!(feature = "e2e-test-hooks")
        && std::env::var_os("ROCM_E2E_LEMONADE_BACKEND_INSTALL_FAILURE").is_some();
    if !cpu_only && !scripted_backend_failure && !reuse_existing && host_has_no_usable_gpu {
        bail!(
            "no usable AMD GPU detected; `rocm serve` requires a GPU under the {policy} \
             policy and does not fall back to CPU. Check the driver with `rocm examine`, \
             confirm /dev/kfd is present, and ensure HIP_VISIBLE_DEVICES / \
             ROCR_VISIBLE_DEVICES are not masking every device.",
            policy = device_policy_name(&device_policy)
        );
    }
    // `--gpu` selects by the amd-smi `gpu` ordinal but is exported via
    // `HIP_VISIBLE_DEVICES`; those orderings can diverge when
    // `ROCR_VISIBLE_DEVICES`/partitioning is in play, so warn at serve time.
    let rocr_visible_devices_set = std::env::var_os("ROCR_VISIBLE_DEVICES").is_some();
    // Whether *any* visibility mask is active. The visible set alone cannot say:
    // with no mask it is just `0..present`, indistinguishable from a HIP mask
    // that happens to list the low ordinals. `validate_pinned_gpu_index` uses
    // this only to word its rejection — "under the active visibility mask" when a
    // mask is set, "not present on this host" when none is — so an out-of-range
    // `--gpu` on an unmasked host is not blamed on a mask the user never set.
    let visibility_mask_active =
        rocr_visible_devices_set || std::env::var_os("HIP_VISIBLE_DEVICES").is_some();
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
    let pinned_gpu_indices = if !cpu_only && let GpuSelection::Index(index) = &gpu_selection {
        Some(validate_pinned_gpu_index(
            *index,
            detect_gpu_count(),
            visible_gpu_indices.as_deref(),
            visibility_mask_active,
        )?)
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
    // (serve-27), which plants a configured-but-unregistered setup venv — the one
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
            &mut |_elapsed| spinner.tick(),
        )?;
        ensure_background_helper_running_quiet(summary_mode)?;

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
                serve_summary::run_smoke_test(&paths, &resolve.canonical_model_id)
            } else {
                serve_summary::SmokeMetrics::default()
            };
            spinner.clear();

            let notes = collect_serve_notes(
                cpu_only,
                &gpu_selection,
                rocr_visible_devices_set,
                &gpu_indices,
                gpu_vram.as_deref(),
                gpu_memory_utilization_note.as_deref(),
                host_gpu_summary.as_ref(),
                engine_serves_vllm,
            );
            let summary = serve_summary::DeploymentSummary {
                engine: selected_engine.clone(),
                requested_model: model,
                api_model: resolve.canonical_model_id,
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
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use rocm_core::resolve_builtin_model_recipe;

    /// Regression coverage for the pre-gate's ordering (ROCMAI-419 review).
    ///
    /// The gate used to be `!cpu_only && any_live_managed_service_for_model(..)`,
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
            !reuse_pregate_opens(false, host_has_no_usable_gpu(Some(&[0])), || {
                probed = true;
                true
            }),
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
            !reuse_pregate_opens(false, host_has_no_usable_gpu(None), || {
                probed_unknown = true;
                true
            }),
            "an unprobeable host must not open the pre-gate either"
        );
        assert!(!probed_unknown);

        // The exemption itself still works: authoritative empty set, live match.
        assert!(
            reuse_pregate_opens(false, host_has_no_usable_gpu(Some(&[])), || true),
            "a host with no usable GPU and a matching live service must still \
             reach the reuse probe, or the exemption serve-26 covers is gone"
        );
        // ...and `--device cpu_only` never had a GPU bail to exempt. No CLI
        // invocation can produce that policy today (`parse_device_policy`
        // rejects `cpu_only`), so this asserts the helper's own guard, not a
        // reachable `serve()` input — see the `cpu_only` binding in `serve`.
        assert!(!reuse_pregate_opens(
            true,
            host_has_no_usable_gpu(Some(&[])),
            || true
        ));
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
    fn protocol_engine_recipe_hint_maps_selected_engine_metadata() {
        let mut recipe = resolve_builtin_model_recipe("qwen").expect("qwen recipe");
        recipe.engine_recipes = vec![
            rocm_core::ModelRecipeEngineRecord {
                engine: "vllm".to_owned(),
                required_flags: vec!["--enable-auto-tool-choice".to_owned()],
                parser_settings: BTreeMap::from([(
                    "reasoning_parser".to_owned(),
                    "qwen3".to_owned(),
                )]),
                preferred_endpoint: Some(rocm_core::ModelRecipeEndpointRecord {
                    endpoint_mode: "openai".to_owned(),
                    settings: BTreeMap::from([("streaming".to_owned(), "true".to_owned())]),
                }),
                unsupported_combinations: vec![
                    rocm_core::ModelRecipeUnsupportedCombinationRecord {
                        combination: "native Windows GPU serving".to_owned(),
                        reason: "vLLM ROCm serving is Linux/WSL only".to_owned(),
                    },
                ],
                notes: vec!["adapter hint".to_owned()],
                model_id_override: None,
            },
            rocm_core::ModelRecipeEngineRecord {
                engine: "lemonade".to_owned(),
                required_flags: vec!["--reasoning-parser".to_owned(), "qwen3".to_owned()],
                parser_settings: BTreeMap::new(),
                preferred_endpoint: None,
                unsupported_combinations: Vec::new(),
                notes: Vec::new(),
                model_id_override: None,
            },
        ];

        let hint = protocol_engine_recipe_hint(&recipe, "vllm").expect("vllm hint");

        assert_eq!(hint.contract_version, ENGINE_RECIPE_CONTRACT_VERSION);
        assert_eq!(hint.engine, "vllm");
        assert_eq!(
            hint.required_flags,
            vec!["--enable-auto-tool-choice".to_owned()]
        );
        assert_eq!(
            hint.parser_settings
                .get("reasoning_parser")
                .map(String::as_str),
            Some("qwen3")
        );
        assert_eq!(
            hint.preferred_endpoint
                .as_ref()
                .map(|endpoint| endpoint.endpoint_mode.as_str()),
            Some("openai")
        );
        assert_eq!(
            hint.preferred_endpoint
                .as_ref()
                .and_then(|endpoint| endpoint.settings.get("streaming"))
                .map(String::as_str),
            Some("true")
        );
        assert_eq!(hint.unsupported_combinations.len(), 1);
        assert_eq!(hint.notes, vec!["adapter hint".to_owned()]);
        let serve_lines = render_serve_engine_recipe_lines(&hint);
        assert!(serve_lines.contains(
            "engine_recipe_policy: selected-engine required_flags are applied at launch"
        ));
        assert!(serve_lines.contains("engine_recipe_required_flags: --enable-auto-tool-choice"));
        assert!(protocol_engine_recipe_hint(&recipe, "unknown-engine").is_none());
    }

    #[test]
    fn tool_call_override_synthesizes_hint_for_vllm_without_recipe() {
        // Arbitrary HF repo with no catalog recipe: the explicit override is the
        // only source of the parser, and a minimal hint is synthesized to carry it.
        let hint = engine_recipe_with_tool_call_override("vllm", None, Some("hermes"))
            .expect("an override should synthesize a vllm tool-choice hint");
        assert_eq!(hint.engine, "vllm");
        assert_eq!(hint.contract_version, ENGINE_RECIPE_CONTRACT_VERSION);
        assert_eq!(
            hint.required_flags,
            vec![
                "--enable-auto-tool-choice".to_owned(),
                "--tool-call-parser".to_owned(),
                "hermes".to_owned(),
            ]
        );
    }

    #[test]
    fn tool_call_override_replaces_recipe_authored_parser() {
        // Override wins over an authored parser: exactly one `--tool-call-parser`,
        // set to the override value, with unrelated flags preserved in order.
        let existing = EngineRecipeHint {
            contract_version: ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
            engine: "vllm".to_owned(),
            required_flags: vec![
                "--reasoning-parser".to_owned(),
                "qwen3".to_owned(),
                "--enable-auto-tool-choice".to_owned(),
                "--tool-call-parser".to_owned(),
                "llama3_json".to_owned(),
            ],
            ..EngineRecipeHint::default()
        };
        let hint =
            engine_recipe_with_tool_call_override("vllm", Some(existing), Some("hermes")).unwrap();
        assert_eq!(
            hint.required_flags,
            vec![
                "--reasoning-parser".to_owned(),
                "qwen3".to_owned(),
                "--enable-auto-tool-choice".to_owned(),
                "--tool-call-parser".to_owned(),
                "hermes".to_owned(),
            ]
        );
        assert_eq!(
            hint.required_flags
                .iter()
                .filter(|flag| *flag == "--tool-call-parser")
                .count(),
            1
        );
    }

    #[test]
    fn tool_call_override_absent_preserves_recipe_flags_without_guessing() {
        // No override: authored recipe metadata flows through unchanged and no
        // parser is ever guessed from the model ref.
        let authored = EngineRecipeHint {
            contract_version: ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
            engine: "vllm".to_owned(),
            required_flags: vec![
                "--enable-auto-tool-choice".to_owned(),
                "--tool-call-parser".to_owned(),
                "hermes".to_owned(),
            ],
            ..EngineRecipeHint::default()
        };
        let hint =
            engine_recipe_with_tool_call_override("vllm", Some(authored.clone()), None).unwrap();
        assert_eq!(hint.required_flags, authored.required_flags);

        // Unknown model, no recipe, no override: nothing is injected.
        assert!(engine_recipe_with_tool_call_override("vllm", None, None).is_none());
        // A blank override is treated as absent.
        assert!(engine_recipe_with_tool_call_override("vllm", None, Some("  ")).is_none());
    }

    #[test]
    fn tool_call_override_leaves_non_vllm_engines_untouched() {
        // The override is vLLM-specific: other engines are never rewritten.
        assert!(engine_recipe_with_tool_call_override("lemonade", None, Some("hermes")).is_none());
        let existing = EngineRecipeHint {
            contract_version: ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
            engine: "lemonade".to_owned(),
            required_flags: vec!["--some-flag".to_owned()],
            ..EngineRecipeHint::default()
        };
        let hint = engine_recipe_with_tool_call_override(
            "lemonade",
            Some(existing.clone()),
            Some("hermes"),
        )
        .unwrap();
        assert_eq!(hint.required_flags, existing.required_flags);
    }

    #[test]
    fn gpu_memory_utilization_absent_without_explicit_flag() {
        // rocm-cli ships no default: with nothing supplied the recipe is left
        // alone, so vLLM applies its own default rather than one rocm-cli owns.
        assert_eq!(parse_gpu_memory_utilization(None).unwrap(), None);
        assert!(engine_recipe_with_gpu_memory_utilization_override("vllm", None, None).is_none());
        let authored = EngineRecipeHint {
            contract_version: ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
            engine: "vllm".to_owned(),
            required_flags: vec!["--enable-auto-tool-choice".to_owned()],
            ..EngineRecipeHint::default()
        };
        let hint = engine_recipe_with_gpu_memory_utilization_override("vllm", Some(authored), None)
            .unwrap();
        assert!(
            !hint
                .required_flags
                .iter()
                .any(|flag| flag == "--gpu-memory-utilization"),
            "no default may be injected: {:?}",
            hint.required_flags
        );
    }

    #[test]
    fn gpu_memory_utilization_override_reaches_required_flags() {
        let value = parse_gpu_memory_utilization(Some("0.35")).unwrap();
        let hint = engine_recipe_with_gpu_memory_utilization_override("vllm", None, value)
            .expect("an explicit value should synthesize a vllm hint");
        assert_eq!(hint.engine, "vllm");
        assert_eq!(hint.contract_version, ENGINE_RECIPE_CONTRACT_VERSION);
        assert_eq!(
            hint.required_flags,
            vec!["--gpu-memory-utilization".to_owned(), "0.35".to_owned()]
        );
    }

    #[test]
    fn gpu_memory_utilization_override_replaces_authored_value_and_keeps_others() {
        let existing = EngineRecipeHint {
            contract_version: ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
            engine: "vllm".to_owned(),
            required_flags: vec![
                "--enable-auto-tool-choice".to_owned(),
                "--gpu-memory-utilization".to_owned(),
                "0.8".to_owned(),
                "--tool-call-parser".to_owned(),
                "hermes".to_owned(),
            ],
            ..EngineRecipeHint::default()
        };
        let hint = engine_recipe_with_gpu_memory_utilization_override(
            "vllm",
            Some(existing),
            parse_gpu_memory_utilization(Some("1.0")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            hint.required_flags,
            vec![
                "--enable-auto-tool-choice".to_owned(),
                "--tool-call-parser".to_owned(),
                "hermes".to_owned(),
                "--gpu-memory-utilization".to_owned(),
                "1".to_owned(),
            ]
        );
    }

    #[test]
    fn gpu_memory_utilization_override_leaves_non_vllm_engines_untouched() {
        // The override is vLLM-specific: other engines are never rewritten, with
        // or without a recipe of their own.
        assert!(
            engine_recipe_with_gpu_memory_utilization_override("lemonade", None, Some(0.5))
                .is_none()
        );
        let existing = EngineRecipeHint {
            contract_version: ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
            engine: "lemonade".to_owned(),
            required_flags: vec!["--some-flag".to_owned()],
            ..EngineRecipeHint::default()
        };
        let hint = engine_recipe_with_gpu_memory_utilization_override(
            "lemonade",
            Some(existing.clone()),
            Some(0.5),
        )
        .unwrap();
        assert_eq!(hint.required_flags, existing.required_flags);
    }

    #[test]
    fn gpu_memory_utilization_rejects_out_of_range_and_unparsable_values() {
        // An explicit CLI value is never silently ignored (unlike the env-var
        // overrides elsewhere): each bad value must produce an actionable error.
        for bad in ["0", "0.0", "1.5", "-0.2", "abc", "", "NaN", "inf"] {
            let Err(error) = parse_gpu_memory_utilization(Some(bad)) else {
                panic!("`{bad}` must be rejected, not silently ignored");
            };
            let message = error.to_string();
            assert!(
                message.contains("--gpu-memory-utilization"),
                "error for `{bad}` should name the flag: {message}"
            );
        }
        assert_eq!(
            parse_gpu_memory_utilization(Some(" 0.5 ")).unwrap(),
            Some(0.5)
        );
        assert_eq!(parse_gpu_memory_utilization(Some("1")).unwrap(), Some(1.0));
    }

    #[test]
    fn generation_defaults_inject_override_generation_config_for_vllm() {
        // vLLM has no raw sampling flags: all three controls collapse into a single
        // `--override-generation-config` JSON with `--max-tokens` mapped to the
        // engine's `max_new_tokens` output cap.
        let hint = engine_recipe_with_generation_defaults(
            "vllm",
            None,
            ServeGenerationDefaults {
                temperature: Some(0.5),
                top_p: Some(0.25),
                max_tokens: Some(128),
            },
        )
        .expect("vllm defaults are supported")
        .expect("supplied defaults should synthesize a vllm hint");
        assert_eq!(hint.engine, "vllm");
        assert_eq!(hint.required_flags.len(), 2);
        assert_eq!(hint.required_flags[0], "--override-generation-config");
        let config: serde_json::Value =
            serde_json::from_str(&hint.required_flags[1]).expect("config is valid JSON");
        assert_eq!(config["temperature"], 0.5);
        assert_eq!(config["top_p"], 0.25);
        assert_eq!(config["max_new_tokens"], 128);
    }

    #[test]
    fn generation_defaults_include_only_supplied_values() {
        // Unset controls are omitted so the engine keeps its own defaults.
        let hint = engine_recipe_with_generation_defaults(
            "vllm",
            None,
            ServeGenerationDefaults {
                temperature: Some(0.25),
                top_p: None,
                max_tokens: None,
            },
        )
        .expect("vllm defaults are supported")
        .expect("a single supplied default still synthesizes a hint");
        let config: serde_json::Value =
            serde_json::from_str(&hint.required_flags[1]).expect("config is valid JSON");
        assert_eq!(config["temperature"], 0.25);
        assert!(config.get("top_p").is_none());
        assert!(config.get("max_new_tokens").is_none());
    }

    #[test]
    fn generation_defaults_merge_with_recipe_authored_config() {
        // CLI values win, but authored keys the CLI does not set are preserved and
        // exactly one `--override-generation-config` pair remains.
        let authored = EngineRecipeHint {
            contract_version: ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
            engine: "vllm".to_owned(),
            required_flags: vec![
                "--enable-auto-tool-choice".to_owned(),
                "--override-generation-config".to_owned(),
                "{\"temperature\":0.9,\"repetition_penalty\":1.1}".to_owned(),
            ],
            ..EngineRecipeHint::default()
        };
        let hint = engine_recipe_with_generation_defaults(
            "vllm",
            Some(authored),
            ServeGenerationDefaults {
                temperature: Some(0.25),
                top_p: Some(0.5),
                max_tokens: None,
            },
        )
        .expect("vllm defaults are supported")
        .unwrap();
        assert_eq!(
            hint.required_flags
                .iter()
                .filter(|flag| *flag == "--override-generation-config")
                .count(),
            1
        );
        assert_eq!(hint.required_flags[0], "--enable-auto-tool-choice");
        let config: serde_json::Value =
            serde_json::from_str(hint.required_flags.last().unwrap()).unwrap();
        assert_eq!(config["temperature"], 0.25);
        assert_eq!(config["top_p"], 0.5);
        assert_eq!(config["repetition_penalty"], 1.1);
    }

    #[test]
    fn generation_defaults_absent_or_non_vllm_pass_through() {
        // No controls supplied: the hint flows through unchanged.
        assert!(
            engine_recipe_with_generation_defaults(
                "vllm",
                None,
                ServeGenerationDefaults::default()
            )
            .unwrap()
            .is_none()
        );
        assert!(
            engine_recipe_with_generation_defaults(
                "unknown",
                None,
                ServeGenerationDefaults {
                    temperature: Some(0.5),
                    top_p: Some(0.5),
                    max_tokens: Some(64),
                },
            )
            .is_err()
        );
    }

    #[test]
    fn generation_defaults_translate_to_lemonade_llama_server_flags() {
        let hint = engine_recipe_with_generation_defaults(
            "lemonade",
            None,
            ServeGenerationDefaults {
                temperature: Some(0.5),
                top_p: Some(0.25),
                max_tokens: Some(128),
            },
        )
        .expect("lemonade defaults are supported")
        .expect("defaults synthesize a recipe");
        assert_eq!(
            hint.required_flags,
            [
                "--temperature",
                "0.5",
                "--top-p",
                "0.25",
                "--n-predict",
                "128"
            ]
        );
    }

    #[test]
    fn generation_defaults_preserve_unset_lemonade_recipe_flags() {
        // Only --temperature is supplied via CLI; an authored --top-p already
        // present in the recipe must survive untouched, mirroring the vLLM
        // merge behavior instead of being deleted.
        let authored = EngineRecipeHint {
            contract_version: ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
            engine: "lemonade".to_owned(),
            required_flags: vec!["--top-p".to_owned(), "0.9".to_owned()],
            ..EngineRecipeHint::default()
        };
        let hint = engine_recipe_with_generation_defaults(
            "lemonade",
            Some(authored),
            ServeGenerationDefaults {
                temperature: Some(0.5),
                top_p: None,
                max_tokens: None,
            },
        )
        .expect("lemonade defaults are supported")
        .expect("supplied defaults should synthesize a hint");
        assert_eq!(
            hint.required_flags,
            ["--top-p", "0.9", "--temperature", "0.5"]
        );
    }

    #[test]
    fn engine_recipe_enables_tool_choice_reflects_flags() {
        assert!(!engine_recipe_enables_tool_choice(None));
        let without = EngineRecipeHint {
            contract_version: ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
            engine: "vllm".to_owned(),
            required_flags: vec!["--reasoning-parser".to_owned(), "qwen3".to_owned()],
            ..EngineRecipeHint::default()
        };
        assert!(!engine_recipe_enables_tool_choice(Some(&without)));
        let with = engine_recipe_with_tool_call_override("vllm", None, Some("hermes"));
        assert!(engine_recipe_enables_tool_choice(with.as_ref()));
    }
}
