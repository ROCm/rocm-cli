// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The `rocm model` (alias `models`) catalog renderer: the curated list of
//! recommended models grouped by hardware target
//! (`render_model_registry_text_with_context_and_host`) and the verbose
//! per-recipe diagnostics behind `rocm model --verbose`
//! (`render_model_registry_verbose_text_with_context_and_host`), which cover
//! artifact and download-rule metadata, host-RAM and GPU fit, manual-alternative
//! recommendations, and per-engine support and adapter-availability notes.
//! `format_bytes`, the human-readable size formatter the verbose listing uses,
//! lives here too and is also read from `storage.rs` and `automations.rs`.
//!
//! Relocated verbatim from `examine_report_text.rs`, which keeps the
//! `rocm examine` report and the engine-inventory listing. The two halves do not
//! call each other: this module reaches only the engine-availability helpers in
//! `engine_ipc.rs`, `PlatformRuleOut` and the platform-gate functions that
//! `main.rs` re-imports from `model_fit.rs`, and `serve_cmd.rs`'s
//! `select_serve_engine`.
//!
//! The two renderers and `format_bytes` are `pub(crate)`; `main.rs` imports them
//! and `chat/mcp_bridge.rs`, `storage.rs` and `automations.rs` reach them as
//! `crate::<name>`. Every other item is private.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use rocm_core::{
    AppPaths, ModelRecipeRecord, ModelRecipeRegistry, ModelRecipeRegistrySource,
    builtin_model_recipes, detect_host_gfx_target, load_model_recipe_registry,
    model_artifact_cache_status, model_catalog_platforms, model_recipe_featured,
    model_recipe_target_platform_label, normalize_therock_family, platform_matches_gfx_family,
};
use rocm_engine_protocol::ENGINE_RECIPE_CONTRACT_VERSION;

use crate::engine_ipc::{builtin_engine_available, resolve_engine_binary_path_with_paths};
use crate::{PlatformRuleOut, engine_ruled_out_by_platform_with};

/// The per-hardware-group heading in `rocm model`, carrying the "this is the
/// box you are on" marker when the group matches the detected host GPU.
///
/// Split out of [`render_model_registry_text_with_context_and_host`] only so
/// the marked branch is reachable from a test. Taking it in situ needs a host
/// whose real gfx target matches a catalog platform, which the default no-GPU
/// CI lane never provides — leaving the marker (`←` before ROCMAI-543) as the
/// one ASCII conversion on this surface that nothing guards.
fn model_catalog_platform_heading(label: &str, your_gpu: bool) -> String {
    if your_gpu {
        format!("{label} <- your GPU")
    } else {
        label.to_owned()
    }
}

/// How many models each hardware group offers.
///
/// `rocm model` is a recommendation, not an inventory: the built-in catalog
/// showed two per group before this, and `--verbose` is where the whole list
/// lives. Selecting by fit widens the candidate pool from the four `featured`
/// recipes to every recipe that runs here, so without a cap a large GPU would
/// get forty rows where it used to get two.
const MAX_RECOMMENDATIONS_PER_PLATFORM: usize = 3;

/// Whether this host's GPU memory is known to cover the recipe's minimum.
///
/// Same arithmetic as [`model_recipe_gpu_fit_label`], which words it for one
/// row; this answers it for selection.
///
/// **A recipe with no declared minimum does not fit.** That reads backwards
/// until you look at which recipes those are: every one of the seven in the
/// built-in catalog is a model too large to size for a single GPU --
/// Llama-3.1-405B, DeepSeek-V4-Flash, MiniMax-M3, GLM-5.2 and their like. The
/// field is absent because nobody could put a single number on them, not
/// because they are free. Reading the gap as "fits anything" recommended a 405B
/// model to a quarter-gigabyte host, which is the same
/// unknown-treated-as-a-fact error this listing exists to stop -- committed
/// about the recipe instead of about the machine.
///
/// They stay reachable: `rocm serve` takes any model, and `rocm model
/// --verbose` lists every recipe with what it needs. What they do not do is
/// get *recommended* on the strength of a figure nobody supplied.
fn recipe_fits_gpu(recipe: &ModelRecipeRecord, aggregate_gpu_vram_gib: f64) -> bool {
    recipe
        .min_gpu_mem_gb
        .is_some_and(|required| aggregate_gpu_vram_gib >= f64::from(required))
}

/// The recommendation order within one hardware group: most capable first.
///
/// `featured` leads, because a curator marking a recipe still means something —
/// but it no longer *selects*, only breaks ties. After that the largest model
/// that fits comes first, since a host that can run it generally should.
fn recommendation_rank(recipe: &ModelRecipeRecord) -> (bool, u32) {
    (
        !model_recipe_featured(recipe),
        u32::MAX - recipe.min_gpu_mem_gb.unwrap_or(0),
    )
}

/// What a hardware group offers, and how much it left out.
///
/// `offered` is capped at [`MAX_RECOMMENDATIONS_PER_PLATFORM`]; `also_fit` is
/// how many further recipes this host could run but the cap dropped. Reporting
/// that count is what keeps the narrowing legible: a short list with nothing
/// said about it reads as "the catalog is small" rather than "there is more
/// behind `--verbose`".
struct PlatformOffer<'a> {
    offered: Vec<&'a ModelRecipeRecord>,
    also_fit: usize,
}

/// Choose what one hardware group recommends for this host.
///
/// `fit` is `None` on a host whose GPU memory could not be read. That case
/// falls back to the curated `featured` short list, uncapped and unnarrowed by
/// memory, because a machine the CLI could not measure must not be told its
/// options are narrow. It is the same `Unknown`-is-not-`Absent` rule
/// `model_readiness` applies to a verdict, applied to a listing.
///
/// The engine gate still applies there. Whether vLLM can run is decided by
/// [`engine_ruled_out_by_platform_with`] from the OS and Docker alone, so an
/// unreadable memory figure says nothing about it — and two of the four
/// `featured` recipes are vLLM-only, which native Windows cannot serve at any
/// size.
fn platform_offer<'a>(
    candidates: &[&'a ModelRecipeRecord],
    fit: Option<f64>,
    engine_allowed: &dyn Fn(&ModelRecipeRecord) -> bool,
) -> PlatformOffer<'a> {
    let Some(available) = fit else {
        return PlatformOffer {
            offered: candidates
                .iter()
                .copied()
                .filter(|recipe| model_recipe_featured(recipe))
                .filter(|recipe| engine_allowed(recipe))
                .collect(),
            also_fit: 0,
        };
    };
    let mut runnable = candidates
        .iter()
        .copied()
        .filter(|recipe| recipe_fits_gpu(recipe, available))
        .filter(|recipe| engine_allowed(recipe))
        .collect::<Vec<_>>();
    runnable.sort_by_key(|recipe| recommendation_rank(recipe));
    let also_fit = runnable
        .len()
        .saturating_sub(MAX_RECOMMENDATIONS_PER_PLATFORM);
    runnable.truncate(MAX_RECOMMENDATIONS_PER_PLATFORM);
    PlatformOffer {
        offered: runnable,
        also_fit,
    }
}

/// Whether the platform gate lets this recipe's engine run on this host.
///
/// Without it a native-Windows host is recommended vLLM models it cannot serve,
/// which is the same class of wrong answer as recommending one that does not
/// fit — the listing would just be confidently wrong about the engine instead
/// of the memory.
fn recipe_engine_allowed(recipe: &ModelRecipeRecord, docker_reachable: &dyn Fn() -> bool) -> bool {
    recipe.preferred_engines.first().is_none_or(|engine| {
        engine_ruled_out_by_platform_with(engine, docker_reachable) == PlatformRuleOut::Allowed
    })
}

#[allow(dead_code)]
pub(crate) fn render_model_registry_text_with_context_and_host(
    _paths: Option<&AppPaths>,
    aggregate_gpu_vram_gib: Option<f64>,
    _host_ram_gib: Option<f64>,
) -> String {
    let mut output = String::new();
    let registry = match load_model_recipe_registry() {
        Ok(registry) => registry,
        Err(error) => {
            let _ = writeln!(output, "Recommended models unavailable: {error}");
            return output;
        }
    };
    let _ = writeln!(output, "{}", model_catalog_header(&registry));
    // A configured index is already curated by whoever signed it, so it is shown
    // whole and not re-filtered here. The built-in catalog is the one this
    // selection is for.
    let show_all = !matches!(registry.source, ModelRecipeRegistrySource::BuiltIn);
    // Only the built-in catalog is narrowed to what runs here; `None` also means
    // "do not filter", which is what an unmeasured host gets.
    let fit = if show_all {
        None
    } else {
        aggregate_gpu_vram_gib
    };
    let visible = registry.recipes.iter().collect::<Vec<_>>();
    let platforms = model_catalog_platforms(&registry);
    let host_gfx_family = detect_host_gfx_target()
        .as_deref()
        .and_then(normalize_therock_family);
    // One `docker info` answers every AIM recipe in the listing, as the verbose
    // renderer already does.
    let docker_reachable = once_per_call(rocm_engine_aim::builtin_docker_daemon_reachable);
    let engine_allowed =
        |recipe: &ModelRecipeRecord| recipe_engine_allowed(recipe, &docker_reachable);
    let mut offered_any = false;
    let mut also_fit_total = 0usize;
    for platform in &platforms {
        let candidates = visible
            .iter()
            .copied()
            .filter(|recipe| rocm_core::model_recipe_targets_platform(recipe, platform))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            continue;
        }
        let offer = if show_all {
            PlatformOffer {
                offered: candidates.clone(),
                also_fit: 0,
            }
        } else {
            platform_offer(&candidates, fit, &engine_allowed)
        };
        if offer.offered.is_empty() {
            continue;
        }
        offered_any = true;
        also_fit_total += offer.also_fit;
        let your_gpu = host_gfx_family
            .as_deref()
            .is_some_and(|family| platform_matches_gfx_family(platform, family));
        let heading = model_catalog_platform_heading(&platform.label, your_gpu);
        let _ = writeln!(output, "\n{heading}");
        for recipe in offer.offered {
            // Show the canonical Hugging Face id (the reliable serve target) and
            // the quant that fits this hardware. Append a fit verdict only where
            // both sides of the comparison are known — on an unmeasured host
            // every row would read "GPU fit unknown", which is noise, and on a
            // recipe with no declared minimum there is nothing to compare.
            let detail = recipe
                .quantization
                .clone()
                .unwrap_or_else(|| model_recipe_memory_label(recipe));
            let fit_label = aggregate_gpu_vram_gib
                .and_then(|available| model_recipe_gpu_fit_label(recipe, available))
                .map_or_else(String::new, |label| format!("  {label}"));
            let _ = writeln!(
                output,
                "  {}  {}{}",
                recipe.canonical_model_id, detail, fit_label
            );
        }
    }
    // Say why the list is the length it is. Without this a narrowed list reads
    // as a small catalog, and an empty one reads as a broken command.
    if !show_all {
        if fit.is_none() {
            let _ = writeln!(
                output,
                "\nThis machine's GPU memory could not be read, so these are not filtered to what \
                 it can run. `rocm diagnose --model <model>` checks one against this host."
            );
        } else if !offered_any {
            let _ = writeln!(
                output,
                "\nNone of the curated models fit this machine's GPU memory. You can still serve \
                 a smaller model of your own, and `rocm model --verbose` lists every curated \
                 recipe with what it needs."
            );
        } else if also_fit_total > 0 {
            let _ = writeln!(
                output,
                "\nNarrowed to what this machine can run; {also_fit_total} more curated model(s) \
                 also fit. `rocm model --verbose` lists them all."
            );
        } else {
            let _ = writeln!(output, "\nNarrowed to what this machine can run.");
        }
    }
    if matches!(registry.source, ModelRecipeRegistrySource::BuiltIn) {
        let _ = writeln!(
            output,
            "\nThese are recommendations - you can serve any compatible Hugging Face model:\n  \
             rocm serve <owner/repo>          # vLLM, e.g. Qwen/Qwen3.6-27B\n  \
             rocm serve <owner/repo>:<quant>  # Lemonade GGUF, e.g. unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M\n\
             \nUse `rocm model --verbose` for details."
        );
    } else {
        let _ = writeln!(output, "\nUse `rocm model --verbose` for details.");
    }
    output
}

fn model_recipe_memory_label(recipe: &ModelRecipeRecord) -> String {
    recipe.min_gpu_mem_gb.map_or_else(
        || "no GPU minimum".to_owned(),
        |value| format!("{value} GiB GPU"),
    )
}

/// How a recipe's declared GPU minimum compares with this host's memory.
///
/// `None` where the recipe declares no minimum: the comparison has no subject,
/// and saying "fits" on the strength of a figure nobody wrote down is the same
/// unknown-read-as-a-fact that [`recipe_fits_gpu`] exists to refuse.
fn model_recipe_gpu_fit_label(
    recipe: &ModelRecipeRecord,
    available_gib: f64,
) -> Option<&'static str> {
    let required = recipe.min_gpu_mem_gb?;
    Some(if available_gib >= f64::from(required) {
        "fits this GPU"
    } else {
        "needs a larger GPU"
    })
}

pub(crate) fn render_model_registry_verbose_text_with_context_and_host(
    paths: Option<&AppPaths>,
    aggregate_gpu_vram_gib: Option<f64>,
    host_ram_gib: Option<f64>,
) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "model recipes");
    let registry = match load_model_recipe_registry() {
        Ok(registry) => registry,
        Err(error) => {
            let _ = writeln!(output, "  source_error: {error}");
            let _ = writeln!(
                output,
                "  source_action: configure or fix the signed recipe index before using /model"
            );
            return output;
        }
    };
    // Every AIM recipe asks whether Docker is reachable; one `docker info` per
    // listing answers all of them.
    let docker_reachable = once_per_call(rocm_engine_aim::builtin_docker_daemon_reachable);
    // Detected only if a recipe's first engine is AIM, whose usability depends on it.
    let host_gpu = std::cell::OnceCell::new();
    for recipe in &registry.recipes {
        let aliases = if recipe.aliases.is_empty() {
            "<none>".to_owned()
        } else {
            recipe.aliases.join(", ")
        };
        let engines = if recipe.preferred_engines.is_empty() {
            "<none>".to_owned()
        } else {
            recipe.preferred_engines.join(", ")
        };
        let memory = recipe.min_gpu_mem_gb.map_or_else(
            || "<not required>".to_owned(),
            |value| format!("{value} GiB"),
        );
        let _ = writeln!(
            output,
            "  {} aliases=[{}] task={} dtype={} device={} min_gpu_mem={} engines=[{}]",
            recipe.canonical_model_id,
            aliases,
            recipe.task,
            recipe.dtype,
            recipe.device_policy,
            memory,
            engines
        );
        let hidden_from_builtin = matches!(registry.source, ModelRecipeRegistrySource::BuiltIn)
            && !model_recipe_featured(recipe);
        if hidden_from_builtin {
            let _ = writeln!(
                output,
                "      catalog: hidden (resolvable via rocm serve, not in the curated list)"
            );
        } else {
            let platforms = model_catalog_platforms(&registry);
            let _ = writeln!(
                output,
                "      catalog: {}",
                model_recipe_target_platform_label(recipe, &platforms)
            );
        }
        append_model_recipe_metadata_lines(&mut output, recipe, paths);
        append_model_host_ram_fit_lines(&mut output, recipe, host_ram_gib);
        append_model_fit_lines(
            &mut output,
            recipe,
            aggregate_gpu_vram_gib,
            &host_gpu,
            paths,
        );
        append_model_engine_support_lines(&mut output, recipe, paths, &docker_reachable);
        if recipe.trust_remote_code {
            let _ = writeln!(output, "      trust_remote_code: true");
        }
        for warning in &recipe.warnings {
            let _ = writeln!(output, "      warning: {warning}");
        }
        append_model_recipe_staleness_lines(&mut output, recipe, rocm_core::unix_time_millis());
    }
    append_model_recipe_registry_source(&mut output, &registry);
    output
}

fn append_model_recipe_staleness_lines(
    output: &mut String,
    recipe: &ModelRecipeRecord,
    now_ms: u128,
) {
    for engine_recipe in &recipe.engine_recipes {
        if let Some(warning) = rocm_core::model_recipe_staleness_warning(engine_recipe, now_ms) {
            let _ = writeln!(output, "      warning: {warning}");
        }
    }
}

/// The `rocm model` header. For the default built-in list it just names the
/// action; a configured recipe index instead advertises its provenance (the
/// only case where the source differs from the default and is worth surfacing).
fn model_catalog_header(registry: &ModelRecipeRegistry) -> String {
    match &registry.source {
        ModelRecipeRegistrySource::BuiltIn => {
            "Recommended models - run one with `rocm serve <model>`".to_owned()
        }
        ModelRecipeRegistrySource::SignedIndex { index_path, .. } => {
            format!(
                "Recommended models - from recipe index {}",
                index_path.display()
            )
        }
    }
}

#[allow(dead_code)]
fn append_model_recipe_registry_source(output: &mut String, registry: &ModelRecipeRegistry) {
    match &registry.source {
        ModelRecipeRegistrySource::BuiltIn => {
            let _ = writeln!(
                output,
                "  source: built-in recipe registry; external signed recipe index is not configured yet"
            );
        }
        ModelRecipeRegistrySource::SignedIndex {
            index_path,
            signature_path,
            public_key_path,
        } => {
            let _ = writeln!(
                output,
                "  source: signed model recipe index path={} signature={} public_key={}",
                index_path.display(),
                signature_path.display(),
                public_key_path.display()
            );
        }
    }
}

#[allow(dead_code)]
fn append_model_recipe_metadata_lines(
    output: &mut String,
    recipe: &ModelRecipeRecord,
    paths: Option<&AppPaths>,
) {
    let _ = writeln!(
        output,
        "      recommended_system_ram: {}",
        recipe
            .recommended_system_ram_gb
            .map_or_else(|| "<unknown>".to_owned(), |value| format!("{value} GiB"))
    );
    let _ = writeln!(
        output,
        "      quantization: {}",
        recipe.quantization.as_deref().unwrap_or("<unspecified>")
    );
    append_model_engine_recipe_settings_lines(output, recipe);
    append_model_artifact_lines(output, recipe, paths);
}

#[allow(dead_code)]
fn append_model_engine_recipe_settings_lines(output: &mut String, recipe: &ModelRecipeRecord) {
    if recipe.engine_recipes.is_empty() {
        let _ = writeln!(output, "      engine_recipes: <none>");
        return;
    }
    let _ = writeln!(
        output,
        "      engine_recipes_policy: protocol_contract={ENGINE_RECIPE_CONTRACT_VERSION} selected-engine hint is passed to adapters during model resolution and required flags are forwarded at launch"
    );
    for engine_recipe in &recipe.engine_recipes {
        let required_flags = format_list_or_none(&engine_recipe.required_flags);
        let parser_settings = format_string_map_or_none(&engine_recipe.parser_settings);
        let endpoint = engine_recipe.preferred_endpoint.as_ref().map_or_else(
            || "<none>".to_owned(),
            |endpoint| {
                let settings = format_string_map_or_none(&endpoint.settings);
                format!("mode={} settings=[{}]", endpoint.endpoint_mode, settings)
            },
        );
        let unsupported = if engine_recipe.unsupported_combinations.is_empty() {
            "<none>".to_owned()
        } else {
            engine_recipe
                .unsupported_combinations
                .iter()
                .map(|item| format!("{} ({})", item.combination, item.reason))
                .collect::<Vec<_>>()
                .join("; ")
        };
        let notes = format_list_or_none(&engine_recipe.notes);
        if !engine_recipe.supported_gfx_families.is_empty()
            || engine_recipe.requires_hf_token
            || !engine_recipe.profile_ids.is_empty()
            || !engine_recipe.min_gpu_counts.is_empty()
        {
            let _ = writeln!(
                output,
                "      engine_recipe {} supported_gfx_families=[{}] requires_hf_token={} profile_ids=[{}] min_gpu_counts=[{}]",
                engine_recipe.engine,
                format_list_or_none(&engine_recipe.supported_gfx_families),
                if engine_recipe.requires_hf_token {
                    "yes"
                } else {
                    "no"
                },
                format_string_map_or_none(&engine_recipe.profile_ids),
                format_string_map_or_none(
                    &engine_recipe
                        .min_gpu_counts
                        .iter()
                        .map(|(family, count)| (family.clone(), count.to_string()))
                        .collect()
                )
            );
        }
        let _ = writeln!(
            output,
            "      engine_recipe {} required_flags=[{}] parser_settings=[{}] preferred_endpoint={} unsupported_combinations=[{}] notes=[{}]",
            engine_recipe.engine, required_flags, parser_settings, endpoint, unsupported, notes
        );
    }
}

#[allow(dead_code)]
fn format_list_or_none(values: &[String]) -> String {
    if values.is_empty() {
        "<none>".to_owned()
    } else {
        values.join(",")
    }
}

#[allow(dead_code)]
fn format_string_map_or_none(values: &BTreeMap<String, String>) -> String {
    if values.is_empty() {
        "<none>".to_owned()
    } else {
        values
            .iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect::<Vec<_>>()
            .join(",")
    }
}

#[allow(dead_code)]
fn append_model_artifact_lines(
    output: &mut String,
    recipe: &ModelRecipeRecord,
    paths: Option<&AppPaths>,
) {
    if recipe.artifacts.is_empty() {
        let _ = writeln!(output, "      artifact_check: not_checked");
        let _ = writeln!(
            output,
            "      artifact_reason: {}",
            recipe
                .artifact_hint
                .as_deref()
                .unwrap_or("recipe does not declare artifact requirements")
        );
        return;
    }

    let gated = recipe
        .artifacts
        .iter()
        .any(|artifact| artifact.gated.unwrap_or(false));
    let artifact_check = if gated {
        "blocked"
    } else {
        "metadata_available"
    };
    let artifact_reason = if gated {
        "signed recipe index declares one or more gated artifacts"
    } else {
        "signed recipe index declares artifact metadata"
    };
    let _ = writeln!(output, "      artifact_check: {artifact_check}");
    let _ = writeln!(
        output,
        "      artifact_reason: {artifact_reason}; this is not a live availability or cache check"
    );
    let _ = writeln!(output, "      artifact_count: {}", recipe.artifacts.len());
    for artifact in &recipe.artifacts {
        let engines = if artifact.engines.is_empty() {
            "<unspecified>".to_owned()
        } else {
            artifact.engines.join(",")
        };
        let size = artifact
            .size_bytes
            .map_or_else(|| "<unknown>".to_owned(), format_bytes);
        let gated = artifact.gated.unwrap_or(false);
        let _ = writeln!(
            output,
            "      artifact {} kind={} uri={} revision={} size={} sha256={} license={} gated={} quantization={} engines=[{}]",
            artifact.artifact_id,
            artifact.kind,
            artifact.uri,
            artifact.revision.as_deref().unwrap_or("<unspecified>"),
            size,
            artifact.sha256.as_deref().unwrap_or("<unspecified>"),
            artifact.license.as_deref().unwrap_or("<unspecified>"),
            gated,
            artifact.quantization.as_deref().unwrap_or("<unspecified>"),
            engines
        );
        if let Some(source_policy) = &artifact.source_policy {
            append_artifact_source_policy_lines(output, source_policy);
        }
        if let Some(paths) = paths {
            let cache = model_artifact_cache_status(paths, &recipe.canonical_model_id, artifact);
            let _ = writeln!(
                output,
                "      artifact_cache {} status={} marker={} reason={}",
                cache.artifact_id,
                cache.status,
                cache.marker_path.display(),
                cache.reason
            );
        } else {
            let _ = writeln!(
                output,
                "      artifact_cache {} status=unknown marker=<unavailable> reason=app paths unavailable; no live cache check performed",
                artifact.artifact_id
            );
        }
    }
}

fn append_artifact_source_policy_lines(
    output: &mut String,
    source_policy: &rocm_core::ModelRecipeArtifactSourcePolicyRecord,
) {
    let _ = writeln!(
        output,
        "      download rule: {}",
        artifact_source_policy_label(&source_policy.policy)
    );
    for host in &source_policy.required_hosts {
        let _ = writeln!(output, "        allowed site: {host}");
    }
    for note in &source_policy.notes {
        let _ = writeln!(output, "        note: {note}");
    }
}

fn artifact_source_policy_label(policy: &str) -> &str {
    match policy {
        "direct_https_sha256" => "Direct HTTPS download with checksum",
        "huggingface_public" => "Public Hugging Face download",
        "huggingface_authenticated" => "Hugging Face download, token required",
        "manual_only" => "Manual download only",
        _ => "Unknown download rule",
    }
}

/// Human-readable size, for `rocm storage`'s cache report and the automations
/// proposal history. Both report values routinely in the tens of megabytes,
/// so the intermediate tiers matter: a bare byte count is unreadable there.
pub(crate) fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    // `{:.1}` rounds, so from 1023.95 of a unit upwards the text already reads
    // as a full 1024 of it. Step up there rather than printing "1024.0 KiB".
    const ROUNDS_UP_TO_NEXT_UNIT: f64 = 1023.95;

    let value = bytes as f64;
    if value < KIB {
        return format!("{bytes} bytes");
    }
    let kib = value / KIB;
    if kib < ROUNDS_UP_TO_NEXT_UNIT {
        return format!("{kib:.1} KiB");
    }
    let mib = kib / KIB;
    if mib < ROUNDS_UP_TO_NEXT_UNIT {
        return format!("{mib:.1} MiB");
    }
    format!("{:.1} GiB", mib / KIB)
}

#[allow(dead_code)]
fn append_model_host_ram_fit_lines(
    output: &mut String,
    recipe: &ModelRecipeRecord,
    host_ram_gib: Option<f64>,
) {
    let _ = writeln!(output, "      system_ram_policy: advisory");
    let Some(required) = recipe.recommended_system_ram_gb else {
        let _ = writeln!(output, "      system_ram_fit: unknown");
        let _ = writeln!(
            output,
            "      system_ram_reason: recipe does not declare a RAM recommendation"
        );
        return;
    };

    match host_ram_gib {
        Some(available) if available >= f64::from(required) => {
            let _ = writeln!(output, "      system_ram_fit: supported");
            let _ = writeln!(
                output,
                "      system_ram_reason: host RAM {} meets recipe recommendation {}",
                format_gib(available),
                format_gib(f64::from(required))
            );
        }
        Some(available) => {
            let _ = writeln!(output, "      system_ram_fit: below_recommendation");
            let _ = writeln!(
                output,
                "      system_ram_reason: host RAM {} is below recipe recommendation {}",
                format_gib(available),
                format_gib(f64::from(required))
            );
            let _ = writeln!(
                output,
                "      system_ram_action: consider a smaller recipe or a host with at least {} system RAM for smoother serving",
                format_gib(f64::from(required))
            );
        }
        None => {
            let _ = writeln!(output, "      system_ram_fit: unknown");
            let _ = writeln!(
                output,
                "      system_ram_reason: current host RAM telemetry is unavailable"
            );
            let _ = writeln!(
                output,
                "      system_ram_action: run /examine to refresh host telemetry"
            );
        }
    }
}

#[allow(dead_code)]
fn append_model_fit_lines(
    output: &mut String,
    recipe: &ModelRecipeRecord,
    aggregate_gpu_vram_gib: Option<f64>,
    host_gpu: &std::cell::OnceCell<rocm_core::HostGpuSummary>,
    paths: Option<&AppPaths>,
) {
    let engine_target = preferred_engine_action_target(recipe, host_gpu, paths);
    match recipe.min_gpu_mem_gb {
        None => {
            let _ = writeln!(output, "      gpu_fit: supported");
            let _ = writeln!(
                output,
                "      reason: recipe device policy `{}` does not require GPU VRAM",
                recipe.device_policy
            );
            let _ = writeln!(
                output,
                "      action: use /plan serve {} with {}",
                recipe_display_ref(recipe),
                engine_target
            );
        }
        Some(required) => match aggregate_gpu_vram_gib {
            Some(available) if available >= f64::from(required) => {
                let _ = writeln!(output, "      gpu_fit: supported");
                let _ = writeln!(
                    output,
                    "      reason: aggregate GPU VRAM {} meets recipe minimum {}",
                    format_gib(available),
                    format_gib(f64::from(required))
                );
                let _ = writeln!(
                    output,
                    "      action: use /plan serve {} with {}",
                    recipe_display_ref(recipe),
                    engine_target
                );
            }
            Some(available) => {
                let _ = writeln!(output, "      gpu_fit: unsupported");
                let _ = writeln!(
                    output,
                    "      reason: aggregate GPU VRAM {} is below recipe minimum {}",
                    format_gib(available),
                    format_gib(f64::from(required))
                );
                let _ = writeln!(
                    output,
                    "      action: choose a recipe with min_gpu_mem <= {} or use a GPU with at least {} before serving",
                    format_gib(available),
                    format_gib(f64::from(required))
                );
                append_manual_alternative_lines(output, recipe, aggregate_gpu_vram_gib);
            }
            None => {
                let _ = writeln!(output, "      gpu_fit: unknown");
                let _ = writeln!(
                    output,
                    "      reason: current telemetry has no aggregate GPU VRAM reading"
                );
                // Not `/examine`: an `Examination` is a static host snapshot and
                // carries no VRAM figure at all, so sending the user there for a
                // missing VRAM reading is a confident dead end. Nor `amd-smi
                // metric --json`: reaching this branch means the measurement
                // this path already attempted came back empty, so repeating it
                // raw answers nothing. `rocm diagnose --model` classifies the
                // same reading against one model (`host_accelerator_memory`) and
                // says why it is missing, which is what the user needs here.
                let _ = writeln!(
                    output,
                    "      action: run `rocm diagnose --model {}` for a live reading",
                    recipe_display_ref(recipe)
                );
            }
        },
    }
}

#[allow(dead_code)]
fn append_manual_alternative_lines(
    output: &mut String,
    recipe: &ModelRecipeRecord,
    aggregate_gpu_vram_gib: Option<f64>,
) {
    let alternatives = manual_alternative_recommendations(recipe, aggregate_gpu_vram_gib);
    if alternatives.is_empty() {
        let _ = writeln!(output, "      manual_alternatives: <none declared>");
    } else {
        let _ = writeln!(
            output,
            "      manual_alternatives: {}",
            alternatives.join(", ")
        );
        let _ = writeln!(
            output,
            "      manual_alternative_policy: user must choose one explicitly; none is selected automatically"
        );
    }
}

/// Curated models to suggest in place of one that does not fit, for `/model`.
///
/// The *selection policy* — declared alternatives first, else same-task curated
/// recipes, capped — lives in `rocm_core::model_readiness::curated_alternatives`
/// and is shared with `rocm diagnose --model`, so the two commands cannot come
/// to answer "what should I run instead" differently. What stays local is the
/// predicate and the wording: `/model` asks only whether the VRAM minimum is
/// met, while `diagnose --model` asks the whole readiness question, and each is
/// right for its own report.
#[allow(dead_code)]
fn manual_alternative_recommendations(
    recipe: &ModelRecipeRecord,
    aggregate_gpu_vram_gib: Option<f64>,
) -> Vec<String> {
    let catalog = builtin_model_recipes();
    rocm_core::model_readiness::curated_alternatives(Some(recipe), &catalog, &|candidate| {
        recipe_is_manual_fit(candidate, aggregate_gpu_vram_gib)
    })
    .into_iter()
    .map(|(candidate_ref, candidate)| {
        // A declared alternative is shown with its requirement, a fallback with
        // its name alone. The two branches are exclusive -- a fallback only runs
        // when no declared candidate survived the predicate, and a candidate
        // that failed it in one branch fails it in the other -- so testing the
        // declared list here recovers exactly which branch produced this pick.
        if recipe
            .manual_alternatives
            .iter()
            .any(|declared| declared == candidate_ref)
        {
            format!(
                "{} ({})",
                candidate_ref,
                candidate.min_gpu_mem_gb.map_or_else(
                    || "CPU-only".to_owned(),
                    |value| format!("{} min GPU", format_gib(f64::from(value)))
                )
            )
        } else {
            candidate_ref.to_owned()
        }
    })
    .collect()
}

#[allow(dead_code)]
fn recipe_is_manual_fit(recipe: &ModelRecipeRecord, aggregate_gpu_vram_gib: Option<f64>) -> bool {
    match recipe.min_gpu_mem_gb {
        None => true,
        Some(required) => {
            aggregate_gpu_vram_gib.is_some_and(|available| available >= f64::from(required))
        }
    }
}

/// Wraps `probe` so it runs at most once, however many times the result is asked
/// for, for as long as the returned closure lives.
fn once_per_call(probe: impl Fn() -> bool) -> impl Fn() -> bool {
    let cached = std::cell::OnceCell::new();
    move || *cached.get_or_init(&probe)
}

#[allow(dead_code)]
fn append_model_engine_support_lines(
    output: &mut String,
    recipe: &ModelRecipeRecord,
    paths: Option<&AppPaths>,
    docker_reachable: &dyn Fn() -> bool,
) {
    if recipe.preferred_engines.is_empty() {
        let _ = writeln!(output, "      engine_support: unknown");
        let _ = writeln!(
            output,
            "      engine_action: add a preferred engine to the signed recipe index before serving"
        );
        return;
    }

    let _ = writeln!(output, "      engine_support:");
    for engine in &recipe.preferred_engines {
        if builtin_engine_available(engine) {
            if let Some(note) = model_registry_adapter_availability_note(engine, docker_reachable) {
                let _ = writeln!(
                    output,
                    "        {engine}: adapter_available path=<built-in> {note}"
                );
            } else {
                let _ = writeln!(output, "        {engine}: built_in");
            }
        } else if let Some(paths) = paths {
            match resolve_engine_binary_path_with_paths(engine, paths) {
                Ok(path) => {
                    if let Some(note) =
                        model_registry_adapter_availability_note(engine, docker_reachable)
                    {
                        let _ = writeln!(
                            output,
                            "        {engine}: adapter_available path={} {note}",
                            path.display()
                        );
                    } else {
                        let _ = writeln!(
                            output,
                            "        {engine}: available path={}",
                            path.display()
                        );
                    }
                }
                Err(error) => {
                    let _ = writeln!(
                        output,
                        "        {engine}: unavailable reason={}",
                        model_registry_reason(error.to_string())
                    );
                }
            }
        } else {
            let _ = writeln!(output, "        {engine}: not_checked");
        }
    }
    let _ = writeln!(
        output,
        "      engine_action: use /engine install <engine> for an unavailable preferred engine, or select an available listed engine explicitly; approval is still required before install or serve"
    );
}

#[allow(dead_code)]
fn model_registry_adapter_availability_note(
    engine: &str,
    docker_reachable: &dyn Fn() -> bool,
) -> Option<&'static str> {
    match engine_ruled_out_by_platform_with(engine, docker_reachable) {
        PlatformRuleOut::Allowed => None,
        PlatformRuleOut::AimDockerUnavailable => Some(
            "runtime_status=unsupported_docker_unavailable reason=Docker daemon not reachable; start Docker or choose a different engine; run /engine for adapter details",
        ),
        PlatformRuleOut::NativeWindowsVllm => Some(
            "runtime_status=unsupported_native_windows reason=native Windows skipped; use WSL/Linux vLLM ROCm; gpu_execution_required=true; run /engine for adapter details",
        ),
        PlatformRuleOut::NativeWindowsAim => Some(
            "runtime_status=unsupported_native_windows reason=native Windows skipped; AIM GPU serving is Linux/WSL2 only; run /engine for adapter details",
        ),
    }
}

/// One definition, shared with `rocm diagnose --model`: a model named in a
/// suggestion has to be a string the user can paste back into `rocm serve`, and
/// two answers to "what do I call this recipe" is one too many.
#[allow(dead_code)]
fn recipe_display_ref(recipe: &ModelRecipeRecord) -> &str {
    rocm_core::model_readiness::recipe_display_ref(recipe)
}

#[allow(dead_code)]
fn preferred_engine_action_target(
    recipe: &ModelRecipeRecord,
    host_gpu: &std::cell::OnceCell<rocm_core::HostGpuSummary>,
    paths: Option<&AppPaths>,
) -> String {
    let first = recipe
        .preferred_engines
        .first()
        .map_or("<engine>", String::as_str);
    if !first.eq_ignore_ascii_case("aim") {
        return first.to_owned();
    }
    // AIM is host-gated (GPU family, platform), so name the engine `rocm serve`
    // would pick on this host instead of assuming AIM.
    let host = host_gpu.get_or_init(|| rocm_core::detect_host_gpu_summary(paths));
    crate::serve_cmd::select_serve_engine(None, None, Some(recipe), Some(host)).engine
}

#[allow(dead_code)]
fn model_registry_reason(reason: String) -> String {
    reason
        .replace(
            " No CPU fallback is used.",
            " Serving stays blocked until a matching GPU engine adapter is installed.",
        )
        .replace(
            " No fallback engine is used.",
            " Serving stays blocked until the requested engine adapter is installed.",
        )
}

#[allow(dead_code)]
fn format_gib(value: f64) -> String {
    if (value.fract()).abs() < f64::EPSILON {
        format!("{value:.0} GiB")
    } else {
        format!("{value:.1} GiB")
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use anyhow::Result;
    use rocm_core::resolve_builtin_model_recipe;

    use super::*;
    use crate::tests::test_paths;

    /// Just below a unit boundary the value rounds up to a full 1024 of the
    /// SMALLER unit, which has to be reported as 1.0 of the larger one — the
    /// same defect `rocm_core::format_bytes` had. Each pair is the last input
    /// that still belongs to the smaller unit and the first that `{:.1}` rounds
    /// up to 1024.0 of it; the second used to print "1024.0 KiB" / "1024.0 MiB".
    #[test]
    fn format_bytes_steps_up_instead_of_printing_1024_of_the_smaller_unit() {
        assert_eq!(format_bytes(1023), "1023 bytes");
        assert_eq!(format_bytes(1024), "1.0 KiB");
        assert_eq!(format_bytes(1_048_524), "1023.9 KiB");
        assert_eq!(format_bytes(1_048_525), "1.0 MiB");
        assert_eq!(format_bytes(1_048_575), "1.0 MiB");
        assert_eq!(format_bytes(1_048_576), "1.0 MiB");
        assert_eq!(format_bytes(1_073_689_395), "1023.9 MiB");
        assert_eq!(format_bytes(1_073_689_396), "1.0 GiB");
        assert_eq!(format_bytes(1_073_741_823), "1.0 GiB");
        // GiB is the top unit: nothing to promote to, so 1024 GiB stays in it.
        assert_eq!(format_bytes(1_099_511_627_776), "1024.0 GiB");
    }

    /// The units `format_bytes` prints, smallest first.
    const BYTE_UNITS: [&str; 4] = ["bytes", "KiB", "MiB", "GiB"];

    /// Byte counts that actually visit the unit boundaries. A uniform `u64`
    /// almost always lands far above the top unit, so on its own it never
    /// samples the band where `{:.1}` rounding reaches 1024.0. The other arms
    /// draw uniformly within one unit's range, and from a window just below
    /// each rounded boundary (KiB→MiB, MiB→GiB) that scales with the boundary,
    /// as the band does — see `rocm_core::disk_space`'s generator for the full
    /// reasoning.
    fn byte_count_strategy() -> impl proptest::strategy::Strategy<Value = u64> {
        use proptest::prelude::*;
        prop_oneof![
            any::<u64>(),
            (0u32..=3).prop_flat_map(|exponent| {
                let low = if exponent == 0 {
                    0
                } else {
                    1024u64.pow(exponent)
                };
                low..1024u64.pow(exponent + 1)
            }),
            (2u32..=3).prop_flat_map(|exponent| {
                let boundary = 1024u64.pow(exponent);
                (boundary - boundary / 16384)..=(boundary + 1)
            }),
        ]
    }

    proptest::proptest! {
        /// A size is rendered in the unit it belongs to, which has two edges.
        ///
        /// Upper: below the top unit, the printed mantissa is under 1024.0 —
        /// otherwise the size is shown in a unit it has outgrown.
        ///
        /// Lower: above `bytes`, the printed mantissa is at least 1.0, and the
        /// next smaller unit would have printed 1024.0 or more — otherwise the
        /// size was promoted before it reached a whole unit.
        ///
        /// Both edges compare the mantissa as printed, in tenths: that is the
        /// quantity a reader sees, so it is the one the scaling has to decide on.
        #[test]
        fn format_bytes_renders_a_size_in_its_own_unit(
            bytes in byte_count_strategy(),
        ) {
            let rendered = format_bytes(bytes);
            let (value, unit) = rendered
                .split_once(' ')
                .expect("rendered size is `<number> <unit>`");
            let value: f64 = value.parse().expect("numeric part parses");
            let tenths = (value * 10.0).round();
            let exponent = BYTE_UNITS
                .iter()
                .position(|name| *name == unit)
                .expect("rendered unit is one of the known units");
            if exponent + 1 < BYTE_UNITS.len() {
                proptest::prop_assert!(
                    tenths < 10_240.0,
                    "{bytes} rendered as {rendered}, which should have been \
                     promoted to the next unit",
                );
            }
            if exponent > 0 {
                proptest::prop_assert!(
                    tenths >= 10.0,
                    "{bytes} rendered as {rendered}, which was promoted before \
                     it reached a whole unit",
                );
                // Dividing by a power of two is exact, so this is the value the
                // smaller unit would have printed, not an approximation of it.
                let smaller = (1..exponent).fold(bytes as f64, |value, _| value / 1024.0);
                proptest::prop_assert!(
                    (smaller * 10.0).round() >= 10_240.0,
                    "{bytes} rendered as {rendered}, but still fits the smaller \
                     unit as {smaller:.1} {}",
                    BYTE_UNITS[exponent - 1],
                );
            }
        }
    }

    #[test]
    fn model_catalog_platform_heading_marks_the_host_gpu_in_ascii() {
        assert_eq!(
            model_catalog_platform_heading("AMD Instinct - MI300X", false),
            "AMD Instinct - MI300X",
            "an unmatched platform heading is the bare label"
        );

        // ROCMAI-543: this marker was `←` (U+2190), which a Windows console on
        // the OEM code page renders as mojibake. `render_model_registry_text_*`
        // only reaches this branch on a host whose real gfx target matches a
        // catalog platform, so the default no-GPU CI lane never covers it.
        // Asserting the exact string (not `contains`) is what makes a revert of
        // the punctuation alone fail, which a substring check would not.
        let marked = model_catalog_platform_heading("AMD Instinct - MI300X", true);
        assert_eq!(marked, "AMD Instinct - MI300X <- your GPU");
        assert!(
            marked.is_ascii(),
            "the host-GPU marker must be ASCII (ROCMAI-543): {marked}"
        );
    }

    #[test]
    fn render_model_registry_text_lists_builtin_recipes() {
        let rendered = render_model_registry_text_with_context_and_host(None, None, None);

        // ROCMAI-543: a Windows console on the OEM code page (437) renders a
        // UTF-8 em dash (and other non-ASCII punctuation) as mojibake like
        // "ΓÇö" instead of transliterating it, so user-facing CLI output must
        // stick to ASCII. `is_ascii()` is strictly stronger than a single
        // em-dash check: it also catches smart quotes, ellipses, and arrows
        // reintroduced anywhere in this output (a new catalog label, a new
        // hint line, a "your GPU" heading), not just U+2014.
        assert!(
            rendered.is_ascii(),
            "rocm model output must be ASCII-only; non-ASCII punctuation renders as \
             mojibake on a Windows console using the OEM code page (ROCMAI-543): {rendered}"
        );

        // Clean header: what the list is + the one action, no implementation jargon.
        assert!(rendered.contains("Recommended models - run one with `rocm serve <model>`"));
        // Featured current models are grouped under their hardware target, each
        // with the quant that fits a single GPU.
        assert!(rendered.contains("AMD Ryzen AI - Strix Halo (Lemonade / llama.cpp)"));
        assert!(rendered.contains("AMD Instinct - MI300X, MI350X, MI355X (vLLM / AIM)"));
        // Strix Halo GGUF entries carry the servable owner/repo:variant ref.
        assert!(rendered.contains("unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M"));
        assert!(rendered.contains("Q4_K_M GGUF"));
        assert!(rendered.contains("Qwen/Qwen3.6-27B"));
        assert!(rendered.contains("BF16"));
        // Multi-GPU-only models are not featured (single-GPU serving only).
        assert!(!rendered.contains("GLM-5.2"));
        assert!(!rendered.contains("DeepSeek-V4-Flash"));
        // Without a known host GPU, no misleading per-row fit verdict is shown.
        assert!(!rendered.contains("GPU fit unknown"));
        // Superseded / smoke / assistant recipes are hidden from the curated list.
        assert!(!rendered.contains("tiny-gpt2"));
        assert!(!rendered.contains("Qwen/Qwen3.5-4B"));
        assert!(!rendered.contains("Qwen3-4B-Instruct-2507-GGUF"));
        // Disclaimer: the list is a starting point; any compatible HF model works.
        assert!(rendered.contains("you can serve any compatible Hugging Face model"));
        assert!(rendered.contains("rocm serve <owner/repo>:<quant>"));
        assert!(rendered.contains("rocm model --verbose"));
        assert!(!rendered.contains("recommended_system_ram:"));
        assert!(!rendered.contains("engine_support:"));
        assert!(!rendered.contains("artifact_check:"));
    }

    #[test]
    fn render_model_registry_text_shows_fit_when_gpu_known() {
        // With a known aggregate VRAM the listing is narrowed to what this host
        // can run, so every row that survives says so. 60 GiB fits Qwen3.6-27B
        // (~54) but not Gemma-4-31B (~62), and the one that does not fit is no
        // longer offered rather than offered with a refusal beside it --
        // "recommended" and "needs a larger GPU" cannot both be true of a row.
        let rendered = render_model_registry_text_with_context_and_host(None, Some(60.0), None);

        assert!(rendered.contains("fits this GPU"));
        assert!(
            !rendered.contains("needs a larger GPU"),
            "a model this host cannot run must not be recommended to it: {rendered}"
        );
        assert!(!rendered.contains("GPU fit unknown"));
        assert!(
            rendered.contains("Narrowed to what this machine can run"),
            "the narrowing must be stated, or a short list reads as a small catalog: {rendered}"
        );
    }

    /// A signed index is shown whole rather than narrowed, so its rows are the
    /// one place a recipe with no declared minimum still reaches the fit label
    /// on a measured host. A bare "fits" there would assert something nobody
    /// measured — the same reading [`recipe_fits_gpu`] refuses.
    #[test]
    fn a_recipe_with_no_declared_minimum_claims_no_fit() {
        let recipes = builtin_model_recipes();
        let undeclared = recipes
            .iter()
            .find(|recipe| recipe.min_gpu_mem_gb.is_none())
            .expect("the catalog ships recipes with no declared GPU minimum");

        assert_eq!(
            model_recipe_gpu_fit_label(undeclared, 0.25),
            None,
            "a model with no declared minimum must not be labelled against a host's memory"
        );
    }

    /// The defect this whole change exists for: `featured` is a static flag, and
    /// on the current catalog its four recipes need 17, 22, 54 and 62 GiB while
    /// the eight recipes at or below 8 GiB carry it on none of them. A small or
    /// integrated GPU was therefore recommended four models it cannot run, and
    /// shown none of the ones it can.
    #[test]
    fn a_small_host_is_offered_models_it_can_actually_run() {
        let recipes = builtin_model_recipes();
        let candidates = recipes.iter().collect::<Vec<_>>();

        let offer = platform_offer(&candidates, Some(8.0), &|_| true);

        assert!(
            !offer.offered.is_empty(),
            "the catalog ships recipes an 8 GiB host can run; offering none would be wrong"
        );
        for recipe in &offer.offered {
            assert!(
                recipe_fits_gpu(recipe, 8.0),
                "offered a model this host cannot run: {} needs {:?} GiB",
                recipe.canonical_model_id,
                recipe.min_gpu_mem_gb
            );
        }
    }

    #[test]
    fn a_large_host_still_sees_the_large_models() {
        let recipes = builtin_model_recipes();
        let candidates = recipes.iter().collect::<Vec<_>>();

        let offer = platform_offer(&candidates, Some(640.0), &|_| true);

        assert!(
            offer
                .offered
                .iter()
                .any(|recipe| recipe.min_gpu_mem_gb.is_some_and(|min| min >= 48)),
            "a host with plenty of memory must still be offered the large models"
        );
    }

    /// A machine the CLI could not measure must not be told its options are
    /// narrow. This is the `Unknown`-is-not-`Absent` rule `model_readiness`
    /// applies to a verdict, applied to a listing -- and it is the behaviour
    /// most likely to regress silently, because an unmeasured host looks the
    /// same as a host with no memory until you read what it was offered.
    #[test]
    fn an_unmeasured_host_is_not_filtered_at_all() {
        let recipes = builtin_model_recipes();
        let candidates = recipes.iter().collect::<Vec<_>>();

        let offer = platform_offer(&candidates, None, &|_| true);

        assert_eq!(
            offer.also_fit, 0,
            "nothing was narrowed, so nothing was left out"
        );
        assert!(
            offer
                .offered
                .iter()
                .all(|recipe| model_recipe_featured(recipe)),
            "an unmeasured host keeps the curated short list"
        );
        assert!(
            !offer.offered.is_empty(),
            "the curated list is what an unmeasured host still gets"
        );
        assert!(
            offer
                .offered
                .iter()
                .any(|recipe| recipe.min_gpu_mem_gb.is_some_and(|needed| needed > 50)),
            "a model far larger than any small host is still offered when no figure ruled it out"
        );
    }

    /// An unreadable memory figure says nothing about whether vLLM can run, so
    /// the relaxation in [`platform_offer`] must be to the memory filter alone.
    /// Native Windows cannot serve a vLLM recipe at any size, and two of the
    /// four `featured` recipes are vLLM-only.
    #[test]
    fn an_unmeasured_host_still_has_its_engine_gate_applied() {
        let recipes = builtin_model_recipes();
        let candidates = recipes.iter().collect::<Vec<_>>();
        let not_lemonade = |recipe: &ModelRecipeRecord| {
            recipe
                .preferred_engines
                .first()
                .is_some_and(|engine| engine == "lemonade")
        };

        let offer = platform_offer(&candidates, None, &not_lemonade);

        assert!(
            !offer.offered.is_empty(),
            "the recipes the gate allows are still offered"
        );
        assert!(
            offer.offered.iter().all(|recipe| not_lemonade(recipe)),
            "a recipe the platform rules out is not offered to an unmeasured host either: {:?}",
            offer
                .offered
                .iter()
                .map(|recipe| &recipe.canonical_model_id)
                .collect::<Vec<_>>()
        );
    }

    /// Also the regression test for reading an absent `min_gpu_mem_gb` as "fits
    /// anything": the seven recipes without one are the largest models in the
    /// catalog, so that reading offered Llama-3.1-405B and DeepSeek-V4-Flash to
    /// a quarter-gigabyte host.
    #[test]
    fn a_host_too_small_for_anything_is_offered_nothing_rather_than_the_wrong_thing() {
        let recipes = builtin_model_recipes();
        let candidates = recipes.iter().collect::<Vec<_>>();

        let offer = platform_offer(&candidates, Some(0.25), &|_| true);

        assert!(
            offer.offered.is_empty(),
            "a machine too small for every curated model must be offered none of them: {:?}",
            offer
                .offered
                .iter()
                .map(|recipe| &recipe.canonical_model_id)
                .collect::<Vec<_>>()
        );
    }

    /// The same rule stated from the other side, so it survives a catalog that
    /// one day sizes those models: a recipe with no declared minimum is never
    /// recommended on a fit-filtered host, however much memory it has.
    #[test]
    fn a_recipe_with_no_declared_minimum_is_never_recommended_by_fit() {
        let recipes = builtin_model_recipes();
        let unsized_ids = recipes
            .iter()
            .filter(|recipe| recipe.min_gpu_mem_gb.is_none())
            .map(|recipe| recipe.canonical_model_id.clone())
            .collect::<Vec<_>>();
        assert!(
            !unsized_ids.is_empty(),
            "the catalog still ships recipes with no minimum; otherwise this test guards nothing"
        );
        let candidates = recipes.iter().collect::<Vec<_>>();

        let offer = platform_offer(&candidates, Some(100_000.0), &|_| true);

        for recipe in &offer.offered {
            assert!(
                !unsized_ids.contains(&recipe.canonical_model_id),
                "{} declares no GPU minimum, so no host can be told it fits",
                recipe.canonical_model_id
            );
        }
    }

    /// Without the platform gate a native-Windows host is recommended vLLM
    /// models it cannot serve -- confidently wrong about the engine instead of
    /// the memory, which is the same defect wearing a different hat.
    #[test]
    fn an_engine_the_platform_rules_out_is_never_offered() {
        let recipes = builtin_model_recipes();
        let candidates = recipes.iter().collect::<Vec<_>>();

        let offer = platform_offer(&candidates, Some(640.0), &|_| false);

        assert!(
            offer.offered.is_empty(),
            "no engine can run here, so nothing can be recommended"
        );
    }

    #[test]
    fn models_beyond_the_cap_are_counted_rather_than_silently_dropped() {
        let recipes = builtin_model_recipes();
        let candidates = recipes.iter().collect::<Vec<_>>();

        let offer = platform_offer(&candidates, Some(640.0), &|_| true);

        assert_eq!(offer.offered.len(), MAX_RECOMMENDATIONS_PER_PLATFORM);
        assert!(
            offer.also_fit > 0,
            "a host that can run far more than the cap shows must be told how many more"
        );
    }

    #[test]
    fn model_catalog_header_only_names_source_for_configured_index() {
        // Default built-in list: no implementation jargon, just the action.
        let builtin = ModelRecipeRegistry {
            recipes: Vec::new(),
            platforms: Vec::new(),
            source: ModelRecipeRegistrySource::BuiltIn,
        };
        let header = model_catalog_header(&builtin);
        assert_eq!(
            header,
            "Recommended models - run one with `rocm serve <model>`"
        );
        assert!(!header.contains("built-in"));
        assert!(!header.contains("fallback"));

        // A configured index is the only case worth advertising provenance for.
        let configured = ModelRecipeRegistry {
            recipes: Vec::new(),
            platforms: Vec::new(),
            source: ModelRecipeRegistrySource::SignedIndex {
                index_path: PathBuf::from("/etc/rocm/recipes.json"),
                signature_path: PathBuf::from("/etc/rocm/recipes.json.sig"),
                public_key_path: PathBuf::from("/etc/rocm/recipes.pub"),
            },
        };
        assert_eq!(
            model_catalog_header(&configured),
            "Recommended models - from recipe index /etc/rocm/recipes.json"
        );
        // ROCMAI-543: both header variants are printed verbatim, so neither may
        // carry non-ASCII punctuation that mojibakes on a Windows OEM code page.
        assert!(model_catalog_header(&builtin).is_ascii());
        assert!(model_catalog_header(&configured).is_ascii());
    }

    #[test]
    fn render_model_registry_verbose_text_keeps_diagnostics() {
        let rendered = render_model_registry_verbose_text_with_context_and_host(None, None, None);

        assert!(rendered.contains("model recipes"));
        assert!(rendered.contains("aliases=[qwen"));
        assert!(rendered.contains("recommended_system_ram: 16 GiB"));
        assert!(rendered.contains("system_ram_fit: unknown"));
        assert!(rendered.contains("gpu_fit: unknown"));
        // This branch is what a host whose VRAM could not be read reaches, and
        // its remediation must name a command that explains the gap rather than
        // the raw measurement this path already tried and got nothing from.
        assert!(rendered.contains("action: run `rocm diagnose --model"));
        assert!(!rendered.contains("amd-smi metric --json"));
        assert!(rendered.contains("engine_support:"));
        assert!(rendered.contains("engine_action: use /engine install <engine>"));
        assert!(rendered.contains("source: built-in recipe registry"));
        // Verbose keeps every recipe (including hidden ones) and annotates each
        // with its curated catalog placement.
        assert!(rendered.contains("catalog: AMD Instinct - MI300X, MI350X, MI355X (vLLM / AIM)"));
        assert!(rendered.contains("catalog: hidden (resolvable via rocm serve"));
        assert!(rendered.contains("sshleifer/tiny-gpt2"));
        // ROCMAI-543: `rocm model --verbose` output must stick to ASCII too, not
        // only the default non-verbose listing.
        assert!(
            rendered.is_ascii(),
            "rocm model --verbose output must be ASCII-only; non-ASCII punctuation renders \
             as mojibake on a Windows console using the OEM code page (ROCMAI-543): {rendered}"
        );
    }

    #[test]
    fn model_recipe_artifact_lines_surface_signed_index_metadata() -> Result<()> {
        let (root, paths) = test_paths("model-artifact-cache-lines");
        let mut recipe = resolve_builtin_model_recipe("qwen").expect("qwen recipe");
        recipe.artifacts = vec![rocm_core::ModelRecipeArtifactRecord {
            artifact_id: "hf-main".to_owned(),
            kind: "huggingface".to_owned(),
            uri: "https://huggingface.co/Qwen/Qwen3.5-4B/resolve/main/model.safetensors".to_owned(),
            revision: Some("main".to_owned()),
            sha256: Some("b".repeat(64)),
            size_bytes: Some(2 * 1024 * 1024 * 1024),
            license: Some("apache-2.0".to_owned()),
            gated: Some(false),
            quantization: Some("bfloat16".to_owned()),
            engines: vec!["vllm".to_owned()],
            source_policy: Some(rocm_core::ModelRecipeArtifactSourcePolicyRecord {
                policy: "huggingface_public".to_owned(),
                required_hosts: vec!["huggingface.co".to_owned()],
                notes: vec!["test metadata only".to_owned()],
            }),
        }];
        recipe.engine_recipes = vec![rocm_core::ModelRecipeEngineRecord {
            engine: "vllm".to_owned(),
            required_flags: vec!["--enable-auto-tool-choice".to_owned()],
            parser_settings: BTreeMap::from([("reasoning_parser".to_owned(), "qwen3".to_owned())]),
            preferred_endpoint: Some(rocm_core::ModelRecipeEndpointRecord {
                endpoint_mode: "openai".to_owned(),
                settings: BTreeMap::from([("streaming".to_owned(), "true".to_owned())]),
            }),
            unsupported_combinations: vec![rocm_core::ModelRecipeUnsupportedCombinationRecord {
                combination: "native Windows GPU serving".to_owned(),
                reason: "vLLM ROCm serving is Linux/WSL only".to_owned(),
            }],
            notes: vec!["metadata only; not applied to launches yet".to_owned()],
            model_id_override: None,
            image_digest: None,
            generated_at_unix_ms: None,
            supported_gfx_families: Vec::new(),
            profile_ids: BTreeMap::new(),
            requires_hf_token: false,
            supported_boards: Vec::new(),
            min_gpu_counts: BTreeMap::new(),
            extra_env: BTreeMap::new(),
        }];
        let mut output = String::new();

        append_model_recipe_metadata_lines(&mut output, &recipe, Some(&paths));
        let _ = fs::remove_dir_all(root);

        assert!(output.contains("artifact_check: metadata_available"));
        assert!(output.contains("artifact_count: 1"));
        assert!(output.contains("artifact hf-main kind=huggingface"));
        assert!(output.contains("download rule: Public Hugging Face download"));
        assert!(output.contains("allowed site: huggingface.co"));
        assert!(output.contains("note: test metadata only"));
        assert!(!output.contains("source_policy=huggingface_public"));
        assert!(output.contains("size=2.0 GiB"));
        assert!(output.contains("engines=[vllm]"));
        assert!(output.contains("artifact_cache hf-main status=missing"));
        assert!(output.contains("prefetch requires an approved source policy"));
        assert!(output.contains(
            "engine_recipes_policy: protocol_contract=0.1.0 selected-engine hint is passed to adapters during model resolution and required flags are forwarded at launch"
        ));
        assert!(output.contains("engine_recipe vllm required_flags=[--enable-auto-tool-choice]"));
        assert!(output.contains("parser_settings=[reasoning_parser=qwen3]"));
        assert!(output.contains("preferred_endpoint=mode=openai settings=[streaming=true]"));
        assert!(output.contains(
            "unsupported_combinations=[native Windows GPU serving (vLLM ROCm serving is Linux/WSL only)]"
        ));
        Ok(())
    }

    #[test]
    fn render_model_registry_text_reports_host_ram_fit() {
        let rendered =
            render_model_registry_verbose_text_with_context_and_host(None, None, Some(32.0));

        assert!(rendered.contains("system_ram_policy: advisory"));
        assert!(rendered.contains("system_ram_fit: supported"));
        assert!(rendered.contains("host RAM 32 GiB meets recipe recommendation 16 GiB"));
        assert!(rendered.contains("system_ram_fit: below_recommendation"));
        assert!(rendered.contains("host RAM 32 GiB is below recipe recommendation 64 GiB"));
        assert!(rendered.contains("host with at least 64 GiB system RAM for smoother serving"));
    }

    #[test]
    fn render_model_registry_text_reports_supported_and_unsupported_gpu_fit() {
        let rendered =
            render_model_registry_verbose_text_with_context_and_host(None, Some(16.0), None);

        assert!(rendered.contains("gpu_fit: supported"));
        assert!(rendered.contains("aggregate GPU VRAM 16 GiB meets recipe minimum 12 GiB"));
        assert!(rendered.contains("gpu_fit: unsupported"));
        assert!(rendered.contains("aggregate GPU VRAM 16 GiB is below recipe minimum 48 GiB"));
        assert!(rendered.contains("choose a recipe with min_gpu_mem <= 16 GiB"));
        assert!(rendered.contains("or use a GPU with at least 48 GiB before serving"));
        assert!(rendered.contains("manual_alternatives: qwen3.5-4b (12 GiB min GPU)"));
        assert!(rendered.contains(
            "manual_alternative_policy: user must choose one explicitly; none is selected automatically"
        ));
        assert!(rendered.contains("llama-3.2-3b-instruct (8 GiB min GPU)"));
        assert!(rendered.contains("tiny-gpt2 (2 GiB min GPU)"));
    }

    #[test]
    fn verbose_model_listing_shows_aim_gpu_families_and_token_requirement() {
        let engine_lines = |id: &str| {
            let recipe = rocm_core::resolve_builtin_model_recipe(id).expect("aim recipe");
            let mut output = String::new();
            append_model_engine_recipe_settings_lines(&mut output, &recipe);
            output
        };
        let gated = engine_lines("meta-llama/Llama-3.2-3B-Instruct");
        assert!(
            gated.contains(
                "engine_recipe aim supported_gfx_families=[gfx94X-dcgpu,gfx950-dcgpu] \
                 requires_hf_token=yes profile_ids=[<none>] min_gpu_counts=[<none>]"
            ),
            "{gated}"
        );
        // Radeon images list the profile pinned per family and the GPU counts.
        let radeon = engine_lines("zai-org/GLM-4.7-Flash");
        assert!(
            radeon.contains(
                "supported_gfx_families=[gfx110X-all,gfx120X-all] requires_hf_token=no \
                 profile_ids=[gfx110X-all="
            ) && radeon.contains("min_gpu_counts=[gfx110X-all=2,gfx120X-all=4]"),
            "{radeon}"
        );
        // A recipe with no AIM entry prints no such line.
        let lemonade = engine_lines("qwen");
        assert!(!lemonade.contains("supported_gfx_families"), "{lemonade}");
    }

    #[test]
    fn engine_action_target_follows_what_serve_would_pick_on_this_host() {
        let target = |recipe_id: &str, family: &str| {
            let recipe = rocm_core::resolve_builtin_model_recipe(recipe_id).expect("recipe");
            let host = std::cell::OnceCell::new();
            let _ = host.set(rocm_core::HostGpuSummary {
                therock_family: Some(family.to_owned()),
                ..rocm_core::HostGpuSummary::default()
            });
            preferred_engine_action_target(&recipe, &host, None)
        };
        // A recipe that lists no AIM is untouched by the host.
        assert_eq!(target("tiny-gpt2", "gfx1151"), "vllm");
        if !cfg!(windows) {
            // AIM is named only where `rocm serve` would pick it.
            assert_eq!(target("Qwen/Qwen3.5-9B", "gfx110X-all"), "aim");
            assert_ne!(target("Qwen/Qwen3.5-9B", "gfx1151"), "aim");
            // Instinct hosts prefer vLLM, which `rocm serve` picks over AIM.
            assert_eq!(target("Qwen/Qwen3-32B", "gfx94X-dcgpu"), "vllm");
        }
    }

    #[test]
    fn once_per_call_runs_the_probe_once() {
        let calls = std::cell::Cell::new(0);
        let cached = once_per_call(|| {
            calls.set(calls.get() + 1);
            false
        });
        assert!(!cached() && !cached() && !cached());
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn verbose_model_listing_shows_a_token_requirement_without_gfx_families() {
        // A schema-valid record can carry `requires_hf_token` alone; the listing
        // must still say the image needs a token.
        let mut recipe =
            rocm_core::resolve_builtin_model_recipe("meta-llama/Llama-3.2-3B-Instruct")
                .expect("aim recipe");
        for record in &mut recipe.engine_recipes {
            record.supported_gfx_families.clear();
        }
        let mut output = String::new();
        append_model_engine_recipe_settings_lines(&mut output, &recipe);
        assert!(
            output.contains(
                "engine_recipe aim supported_gfx_families=[<none>] requires_hf_token=yes"
            ),
            "{output}"
        );
    }

    #[test]
    fn model_recipe_staleness_lines_warn_only_for_stale_engine_recipes() {
        let recipe = rocm_core::resolve_builtin_model_recipe("Qwen/Qwen3-32B").expect("aim entry");
        let verified = recipe.engine_recipes[0]
            .generated_at_unix_ms
            .expect("aim entries record when they were verified");
        let stale_ms = rocm_core::MODEL_RECIPE_STALE_AFTER_MS;
        let mut fresh = String::new();
        append_model_recipe_staleness_lines(&mut fresh, &recipe, verified + stale_ms);
        assert!(fresh.is_empty(), "{fresh}");
        let mut stale = String::new();
        append_model_recipe_staleness_lines(&mut stale, &recipe, verified + stale_ms + 1);
        assert!(
            stale.contains("warning: aim entry was last verified 180 days ago"),
            "{stale}"
        );
    }

    #[test]
    fn render_model_registry_text_marks_tiny_recipe_as_gpu_smoke_support() {
        let rendered = render_model_registry_verbose_text_with_context_and_host(None, None, None);

        assert!(rendered.contains("sshleifer/tiny-gpt2"));
        assert!(rendered.contains("device=gpu_required"));
        assert!(rendered.contains("min_gpu_mem=2 GiB"));
        assert!(!rendered.contains("recipe device policy `cpu_only`"));
    }

    #[test]
    fn model_registry_marks_windows_vllm_adapter_as_runtime_unsupported() -> Result<()> {
        let (root, paths) = test_paths("model-vllm-support");
        let plugin_dir = paths.primary_engine_plugin_dir();
        fs::create_dir_all(&plugin_dir)?;
        let plugin_path = plugin_dir.join(
            rocm_engine_protocol::platform_engine_plugin_binary_name("vllm"),
        );
        fs::write(&plugin_path, "vllm")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&plugin_path, fs::Permissions::from_mode(0o755))?;
        }
        let recipe = resolve_builtin_model_recipe("qwen32b").expect("qwen32b recipe");
        let mut output = String::new();

        append_model_engine_support_lines(
            &mut output,
            &recipe,
            Some(&paths),
            &rocm_engine_aim::builtin_docker_daemon_reachable,
        );
        let _ = fs::remove_dir_all(root);

        if cfg!(windows) {
            assert!(output.contains("vllm: adapter_available"), "{output}");
            assert!(
                output.contains("runtime_status=unsupported_native_windows"),
                "{output}"
            );
            assert!(output.contains("gpu_execution_required=true"), "{output}");
            assert!(!output.contains("CPU fallback"), "{output}");
        } else {
            assert!(output.contains("vllm: built_in"), "{output}");
        }
        Ok(())
    }

    #[test]
    fn model_registry_engine_reasons_do_not_mention_cpu_fallbacks() {
        let rendered = render_model_registry_verbose_text_with_context_and_host(None, None, None);

        assert!(!rendered.contains("CPU fallback"));
        assert!(rendered.contains("engine_action: use /engine install <engine>"));
    }

    /// The adapter-availability note the verbose listing appends for vLLM: a
    /// native Windows host gets the explicit `unsupported_native_windows`
    /// status, every other host gets no note. Moved here from the
    /// `engine_runtime_status_label` test in `examine_report_text.rs`, which
    /// keeps the `runtime:` label half of the same check.
    #[test]
    fn vllm_adapter_availability_note_is_explicit_on_native_windows() {
        if cfg!(windows) {
            assert!(
                model_registry_adapter_availability_note("vllm", &|| true)
                    .is_some_and(|note| note.contains("unsupported_native_windows"))
            );
        } else {
            assert!(model_registry_adapter_availability_note("vllm", &|| true).is_none());
        }
    }
}
