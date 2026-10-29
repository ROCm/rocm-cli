// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Examine/model-catalog/engine-inventory rendering: the `rocm examine`
//! report (both its JSON-adjacent text path and its plain-language human
//! report), the `rocm engines`/`examine` engine-inventory listing, and the
//! `rocm models`/`examine` model-catalog rendering (built-in recipe
//! listing, verbose per-recipe diagnostics, artifact/source-policy
//! metadata, host-RAM/GPU fit, manual-alternative recommendations, and
//! per-engine support/adapter-availability notes).
//!
//! Mechanically relocated from `main.rs` with no behavior change — Phase 6f
//! (ROCMAI-208), the last sub-phase of this split, folding the
//! `main.rs` modularization epic (ROCMAI-27/ROCMAI-86) closed (see
//! `docs/architecture.md`).
//!
//! This module owns: the launch-summary/examine text surfaces
//! (`render_launch_summary`, `render_examine_text`,
//! `render_examine_text_with_paths`, `examine_human_report`,
//! `render_examine_plain_header`, `plain_status_label`), the engine-inventory
//! listing shared by `rocm engines` and `rocm examine`
//! (`write_default_engine_legend`, the `DEFAULT_ENGINE_MARKER` constant,
//! `render_engine_inventory_text_with_paths`, `append_examine_runtime_state`,
//! `append_examine_engine_inventory`), and the model-catalog rendering
//! (`model_catalog_platform_heading`,
//! `render_model_registry_text_with_context_and_host`,
//! `model_recipe_memory_label`, `model_recipe_gpu_fit_label`,
//! `render_model_registry_verbose_text_with_context_and_host`,
//! `model_catalog_header`, `append_model_recipe_registry_source`,
//! `append_model_recipe_metadata_lines`,
//! `append_model_engine_recipe_settings_lines`, `format_list_or_none`,
//! `format_string_map_or_none`, `append_model_artifact_lines`,
//! `append_artifact_source_policy_lines`, `artifact_source_policy_label`,
//! `format_bytes`, `append_model_host_ram_fit_lines`,
//! `append_model_fit_lines`, `append_manual_alternative_lines`,
//! `manual_alternative_recommendations`, `recipe_is_manual_fit`,
//! `append_model_engine_support_lines`, `recipe_display_ref`,
//! `preferred_engine_action_target`, `model_registry_reason`, `format_gib`,
//! `append_engine_detect_summary`, `friendly_engine_runtime_kind`,
//! `friendly_engine_detect_notes`, `friendly_engine_detect_note_fallback`,
//! `engine_runtime_status_label`, and
//! `engine_runtime_is_native_windows_unsupported`).
//!
//! Two items physically interleaved in this span in `main.rs` moved with it
//! rather than staying behind, since neither has a caller outside this
//! cluster: the `DEFAULT_ENGINE_MARKER` constant (shared by both
//! engine-inventory renderers so the two copies cannot drift apart) and
//! `model_registry_adapter_availability_note` (read only from
//! `append_model_engine_support_lines` and this module's own test). Both
//! stay private.
//!
//! `render_config_text` and `telemetry_policy_summary` sit immediately
//! adjacent to this cluster's end boundary in `main.rs` (they render the
//! `rocm config` command's own output) but are explicitly out of scope —
//! they don't fit any of the 8 named Phase 6f clusters, a gap already
//! flagged on epic ROCMAI-27 and on ROCMAI-179 ("Phase 8: validate
//! module-split sufficiency"). They are left exactly where they are,
//! untouched by this change.
//!
//! `dispatch()` calls directly into this module at three points — the
//! `models`/`examine --verbose` branches' calls to
//! `render_model_registry_verbose_text_with_context_and_host` and
//! `render_model_registry_text_with_context_and_host`, and the human-readable
//! `Command::Examine` arm's call to `examine_human_report` — all three
//! re-imported via `use crate::examine_report_text::{...};`; `main.rs`'s own
//! `launch_default` calls `render_launch_summary` directly, also re-imported
//! there. `render_examine_text`, `render_examine_text_with_paths`, and
//! `render_model_registry_verbose_text_with_context_and_host` are also
//! `pub(crate)`, read from [`chat.rs`](../apps/rocm/src/chat.rs)'s
//! deterministic tool-output summaries; `render_engine_inventory_text_with_paths`
//! is also read from `chat.rs` and from
//! [`engines_cmd.rs`](../apps/rocm/src/engines_cmd.rs)'s `engines()`;
//! `append_examine_runtime_state` is also `pub(crate)`, read only from
//! [`runtimes_cmd.rs`](../apps/rocm/src/runtimes_cmd.rs)'s own test module —
//! `main.rs`'s re-import of it is itself `#[cfg(test)]`, since it would
//! otherwise be an unused import in a non-test build; `format_bytes` is also
//! `pub(crate)`, read from
//! [`storage.rs`](../apps/rocm/src/storage.rs)'s cache-size reporting and
//! [`automations.rs`](../apps/rocm/src/automations.rs)'s plain-language
//! summaries. Every other item in this list has no caller outside this
//! module and stays private.
//!
//! This cluster reaches three items that stayed behind in `main.rs` because
//! they already had callers outside it: `engine_inventory()` (the built-in
//! engine list, also read from `main.rs`'s own serve-engine resolution),
//! `engine_ruled_out_by_platform` (also read from `main.rs`'s engine-serve
//! dispatch arm; the private `PlatformRuleOut` enum it returns, which
//! `model_registry_adapter_availability_note` matches on, is reached the same
//! way), and `missing_packaged_engine_reason` (already noted in
//! [`engine_ipc.rs`](../apps/rocm/src/engine_ipc.rs)'s doc comment as having
//! callers outside that cluster, plus its own test — this cluster is one of
//! them) — all three reached via `use crate::{...};`, the same way other
//! mechanically relocated modules reach a left-behind crate-root item.
//!
//! The cluster's regression tests live in this module's own `mod tests`,
//! reaching the shared `test_paths` fixture through `crate::tests::` and
//! `write_test_pip_runtime` through `crate::test_support::`, the same way
//! other mechanically relocated modules do. Four tests interleaved among
//! this cluster's tests in `main.rs`'s old shared `mod tests` stayed behind
//! because their primary subject is unrelated: `missing_packaged_engine_reason`'s
//! own test (`missing_packaged_engine_reason_has_no_deferred_first_party_engines`),
//! two CLI-lifecycle-audit-log tests
//! (`record_cli_audit_event_writes_cli_lifecycle_record`,
//! `cli_lifecycle_log_sanitizes_components_and_newlines`), and the
//! whole-command-tree ASCII-help walk (`every_help_text_in_the_command_tree_is_ascii`).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::Result;
use rocm_core::{
    AppPaths, ExamineSummary, ModelRecipeRecord, ModelRecipeRegistry, ModelRecipeRegistrySource,
    RocmCliConfig, builtin_model_recipes, detect_host_gfx_target, engine_plugin_dirs,
    load_model_recipe_registry, managed_pip_cache_dir, model_artifact_cache_status,
    model_catalog_platforms, model_recipe_featured, model_recipe_target_platform_label,
    normalize_therock_family, platform_matches_gfx_family,
};
use rocm_engine_protocol::{
    DetectRequest, DetectResponse, ENGINE_RECIPE_CONTRACT_VERSION, EngineMethod,
};

use crate::engine_ipc::{
    builtin_engine_available, engine_request, resolve_engine_binary_path,
    resolve_engine_binary_path_with_paths,
};
use crate::freeform_planner;
use crate::runtimes_cmd::{
    current_runtime_manifest, default_runtime_id_matches, recover_setup_runtime_registration,
    runtime_keys_text, runtime_usability_status, setup_runtime_paths, toolchain_state_text,
};
use crate::therock;
use crate::{
    PlatformRuleOut, engine_inventory, engine_ruled_out_by_platform, missing_packaged_engine_reason,
};

pub(crate) fn render_launch_summary(paths: &AppPaths, config: &RocmCliConfig) -> String {
    // User-facing text, so it must agree with what `serve` would pick — the same
    // contradiction `examine` used to show on Instinct.
    let selected_default_engine = config
        .default_engine
        .as_deref()
        .unwrap_or_else(|| freeform_planner::host_default_engine(Some(paths)));
    let mut output = String::new();
    let _ = writeln!(output, "rocm interactive shell");
    let _ = writeln!(output, "  terminal: non-interactive");
    let _ = writeln!(output, "  default engine: {selected_default_engine}");
    let _ = writeln!(
        output,
        "  default runtime: {}",
        config.default_runtime_id.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(
        output,
        "  active runtime key: {}",
        config.active_runtime_key.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(output, "  config dir: {}", paths.config_dir.display());
    let _ = writeln!(output, "  config file: {}", paths.config_path().display());
    let _ = writeln!(output, "  data dir: {}", paths.data_dir.display());
    let _ = writeln!(output, "  cache dir: {}", paths.cache_dir.display());
    let _ = writeln!(
        output,
        "  note: launch from an interactive terminal to enter the TUI."
    );
    output
}

pub(crate) fn render_examine_text() -> Result<String> {
    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths).unwrap_or_default();
    render_examine_text_with_paths(&paths, &config)
}

pub(crate) fn render_examine_text_with_paths(
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<String> {
    Ok(examine_human_report(paths, config)?.0)
}

/// Build the human examine report and return it alongside the `ExamineSummary`
/// it was built from, so callers can derive scope/exit-code without re-probing.
pub(crate) fn examine_human_report(
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<(String, ExamineSummary)> {
    recover_setup_runtime_registration(paths, config)?;
    let summary = ExamineSummary::gather()?;
    let mut output = render_examine_plain_header(&summary);
    output.push_str(&summary.render_text());
    append_examine_runtime_state(&mut output, paths, config)?;
    append_examine_engine_inventory(&mut output, paths, config, &summary.default_engine);
    Ok((output, summary))
}

fn render_examine_plain_header(summary: &ExamineSummary) -> String {
    let gpu = if summary.detected_gfx_target.is_some() {
        "AMD GPU detected"
    } else {
        "AMD GPU not detected yet"
    };
    // This line counted only CLI-managed runtimes, so a machine with ROCm
    // already installed was greeted with "No ROCm installs saved yet" -- read,
    // reasonably, as "nothing is installed". Now that the resolver finds those
    // installs and their versions, say what is actually on the machine, and that
    // a pre-existing install is left alone on purpose rather than missed.
    let managed = match summary.managed_runtime_count {
        0 => None,
        1 => Some("1 ROCm install saved".to_owned()),
        count => Some(format!("{count} ROCm installs saved")),
    };
    let existing = (summary.legacy_rocm.status != "not_detected").then(|| {
        summary.legacy_rocm.version.as_deref().map_or_else(
            || "existing ROCm install found, left unmanaged by design".to_owned(),
            |version| format!("existing ROCm {version} found, left unmanaged by design"),
        )
    });
    let runtime = match (managed, existing) {
        (Some(managed), Some(existing)) => format!("{managed}; {existing}"),
        (Some(managed), None) => managed,
        (None, Some(existing)) => format!("No ROCm installs saved yet; {existing}"),
        (None, None) => "No ROCm installs saved yet".to_owned(),
    };
    format!(
        "ROCm setup check\n  {gpu}\n  {runtime}\n  Driver: {}\n\nDetails\n",
        plain_status_label(&summary.driver.status)
    )
}

fn plain_status_label(status: &str) -> String {
    status.replace('_', " ")
}

/// Marker shown beside the engine `serve`/CLI commands default to. Every
/// place that renders this glyph MUST use this constant so the rendered
/// character and the legend text stay in sync.
const DEFAULT_ENGINE_MARKER: &str = "*";

/// Writes the legend line explaining [`DEFAULT_ENGINE_MARKER`]. Shared by both
/// engine-inventory renderers so the two copies cannot drift apart.
fn write_default_engine_legend(output: &mut String) {
    let _ = writeln!(output, "  legend: {DEFAULT_ENGINE_MARKER} = default engine");
}

pub(crate) fn render_engine_inventory_text_with_paths(paths: Option<&AppPaths>) -> String {
    // Mark the engine this GPU actually serves on as primary. Using the platform
    // constant put the `*` on Lemonade even on Instinct, where `serve` picks vLLM.
    let host_gpu = rocm_core::detect_host_gpu_summary(paths);
    let host_default_engine = rocm_core::default_engine_for_host(&host_gpu);
    // A configured `default_engine` still wins over the host preference, mirroring
    // `select_serve_engine` (and `append_examine_engine_inventory`). Without this,
    // the legend could mark an engine `serve` will not actually default to.
    let configured_default_engine = paths.and_then(|paths| {
        RocmCliConfig::load(paths)
            .ok()
            .and_then(|config| config.default_engine)
    });
    // Mirrors `select_serve_engine`'s guard: a blank configured value must not
    // out-rank the host preference, or the marker would land on an engine name
    // that is empty rather than falling back.
    let default_engine = configured_default_engine
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(host_default_engine);
    // A configured default naming an external plugin (or a stale/typo'd name)
    // won't appear in `engine_inventory()`'s built-ins list below — printing
    // the legend in that case would explain a marker that lands on zero rows.
    let any_marked = engine_inventory()
        .iter()
        .any(|(name, _)| *name == default_engine);
    let mut output = String::new();
    let _ = writeln!(output, "Local model engines");
    let _ = writeln!(
        output,
        "  Built-in engines are included with rocm-cli. External plugins are optional."
    );
    let _ = writeln!(output, "  ROCm GPU execution is required.");
    if let Some(paths) = paths {
        let _ = writeln!(output, "  Plugin folders:");
        for (index, path) in engine_plugin_dirs(paths).iter().enumerate() {
            let note = if index == 0 { "primary" } else { "legacy" };
            let _ = writeln!(output, "    {}. {} ({note})", index + 1, path.display());
        }
    } else {
        let _ = writeln!(output, "  Plugin folders: not checked");
    }
    if any_marked {
        write_default_engine_legend(&mut output);
    }
    for (name, note) in engine_inventory() {
        let marker = if *name == default_engine {
            DEFAULT_ENGINE_MARKER
        } else {
            " "
        };
        let _ = writeln!(output, "{marker} {name:10} {note}");
        append_engine_detect_summary(&mut output, name, paths);
    }
    let _ = writeln!(
        output,
        "  protocol: {}",
        rocm_engine_protocol::ENGINE_PROTOCOL_VERSION
    );
    output
}

pub(crate) fn append_examine_runtime_state(
    output: &mut String,
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<()> {
    let manifests = therock::load_runtime_manifests(paths)?;
    let active = current_runtime_manifest(config, &manifests);
    let default_runtime_matches = default_runtime_id_matches(config, &manifests);
    let ambiguous_default_keys =
        if config.active_runtime_key.is_none() && default_runtime_matches.len() > 1 {
            Some(runtime_keys_text(&default_runtime_matches))
        } else {
            None
        };
    let _ = writeln!(output, "runtime_state:");
    let _ = writeln!(
        output,
        "  active_runtime_id: {}",
        config.default_runtime_id.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(
        output,
        "  active_runtime_key: {}",
        config.active_runtime_key.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(
        output,
        "  previous_runtime_key: {}",
        config.previous_runtime_key.as_deref().unwrap_or("<unset>")
    );
    let active_status = match active {
        Some(manifest) => runtime_usability_status(manifest),
        None if ambiguous_default_keys.is_some() => "ambiguous_runtime_id".to_owned(),
        None if config.active_runtime_key.is_some() || config.default_runtime_id.is_some() => {
            "missing_manifest".to_owned()
        }
        None => "unset".to_owned(),
    };
    let _ = writeln!(output, "  active_runtime_status: {active_status}");
    if let Some(keys) = ambiguous_default_keys {
        let _ = writeln!(output, "  active_runtime_matches: {keys}");
        let _ = writeln!(
            output,
            "  active_runtime_action: rocm runtimes activate <runtime_key>"
        );
    }
    if let Some(manifest) = active {
        let _ = writeln!(
            output,
            "  active_runtime_root: {}",
            manifest.install_root.display()
        );
        let pip_cache_dir = manifest
            .pip_cache_dir
            .clone()
            .unwrap_or_else(|| managed_pip_cache_dir(&manifest.install_root));
        let _ = writeln!(
            output,
            "  active_runtime_pip_cache_dir: {}",
            pip_cache_dir.display()
        );
        let _ = writeln!(
            output,
            "  active_runtime_version: {}",
            therock::runtime_version_display(&manifest.version)
        );
        let _ = writeln!(output, "  active_runtime_family: {}", manifest.family);
        // The same fact `rocm runtimes list` reports as `toolchain=`, for the
        // one runtime that is actually in use. `examine` is where a user looks
        // when a build fails on a missing `hipcc`, and without this line the
        // command that exists to answer "what is my ROCm state" could not say
        // whether the active runtime has a compiler at all.
        let _ = writeln!(
            output,
            "  active_runtime_toolchain: {}",
            toolchain_state_text(manifest.includes_devel())
        );
        let mode = if manifest.read_only {
            "read-only"
        } else {
            "managed"
        };
        let _ = writeln!(output, "  active_runtime_mode: {mode}");
    }
    if let Some((setup_root, pip_cache_dir)) = setup_runtime_paths(config) {
        let _ = writeln!(output, "  setup_runtime_root: {}", setup_root.display());
        let _ = writeln!(
            output,
            "  setup_runtime_pip_cache_dir: {}",
            pip_cache_dir.display()
        );
    }
    let keys = if manifests.is_empty() {
        "<none>".to_owned()
    } else {
        manifests
            .iter()
            .map(|manifest| manifest.runtime_key.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let _ = writeln!(output, "  registered_runtime_keys: {keys}");
    Ok(())
}

/// `host_default_engine` is the engine this GPU serves on, already resolved by
/// the caller from the `ExamineSummary` it holds — passed in rather than
/// re-probed so the two halves of one `rocm examine` run cannot disagree.
fn append_examine_engine_inventory(
    output: &mut String,
    paths: &AppPaths,
    config: &RocmCliConfig,
    host_default_engine: &str,
) {
    // Mirrors `select_serve_engine`'s guard: a blank configured value must not
    // out-rank the host preference, or `effective_default` below would become
    // an engine name that is empty rather than falling back.
    let configured_default = config
        .default_engine
        .as_deref()
        .filter(|value| !value.trim().is_empty());
    // A configured value still wins, mirroring `select_serve_engine`. Only the
    // fallback becomes GPU-aware: it used to be the platform constant, which
    // reported Lemonade on Instinct where serve picks vLLM.
    let effective_default = configured_default.unwrap_or(host_default_engine);
    // A configured default naming an external plugin (or a stale/typo'd name)
    // won't appear in `engine_inventory()`'s built-ins list below — printing
    // the legend in that case would explain a marker that lands on zero rows.
    let any_marked = engine_inventory()
        .iter()
        .any(|(name, _)| *name == effective_default);
    let _ = writeln!(output, "engine_inventory:");
    // Unchanged on purpose: this line means "what the user configured", so an
    // unset value must keep reading as unset rather than borrowing the host default.
    let _ = writeln!(
        output,
        "  configured_default_engine: {}",
        configured_default.unwrap_or("<platform default>")
    );
    let _ = writeln!(output, "  effective_default_engine: {effective_default}");
    // Say so when a configured engine is holding this host off the one its GPU
    // would pick. Installs provisioned before the installer stopped seeding
    // `default_engine` still carry `lemonade` on disk, and nothing distinguishes
    // that seed from a deliberate choice -- so point at the fix rather than
    // rewriting a file the user owns on a guess. This also catches a genuinely
    // stale choice made by hand, which no migration would have covered.
    if let Some(configured) = configured_default
        && configured != host_default_engine
    {
        let _ = writeln!(
            output,
            "  configured_default_note: overrides this host's {host_default_engine} preference; run `rocm config clear-default-engine` to follow the host"
        );
    }
    let _ = writeln!(
        output,
        "  plugin_policy: first-party engines are built in; external data-dir plugins are optional overrides"
    );
    let _ = writeln!(
        output,
        "  external_plugin_policy: optional overrides only; no fallback engine is selected automatically"
    );
    let _ = writeln!(
        output,
        "  plugin_dirs: {}",
        engine_plugin_dirs(paths)
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    if any_marked {
        write_default_engine_legend(output);
    }
    for (engine, note) in engine_inventory() {
        let marker = if *engine == effective_default {
            DEFAULT_ENGINE_MARKER
        } else {
            " "
        };
        let adapter = if builtin_engine_available(engine) {
            "built_in".to_owned()
        } else {
            match resolve_engine_binary_path_with_paths(engine, paths) {
                Ok(path) => format!("external path={}", path.display()),
                Err(error) => format!("missing reason={error}"),
            }
        };
        let runtime_pref = config
            .engine_config(engine)
            .and_then(|entry| {
                entry
                    .preferred_runtime_id
                    .as_deref()
                    .or(entry.preferred_env_id.as_deref())
                    .or(entry.last_installed_runtime_id.as_deref())
                    .or(entry.last_installed_env_id.as_deref())
            })
            .unwrap_or("<unset>");
        let _ = writeln!(
            output,
            "  {marker} {engine} adapter={adapter} runtime_pref={runtime_pref} note={note}"
        );
    }
}

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
    // Curated view: for the built-in catalog show only the featured short list;
    // a configured index is already curated, so show all of it. Group either way
    // by the hardware path each model targets.
    let show_all = !matches!(registry.source, ModelRecipeRegistrySource::BuiltIn);
    let visible = registry
        .recipes
        .iter()
        .filter(|recipe| show_all || model_recipe_featured(recipe))
        .collect::<Vec<_>>();
    let platforms = model_catalog_platforms(&registry);
    let host_gfx_family = detect_host_gfx_target()
        .as_deref()
        .and_then(normalize_therock_family);
    for platform in &platforms {
        let mut group = visible
            .iter()
            .filter(|recipe| {
                recipe.preferred_engines.first().is_some_and(|engine| {
                    platform
                        .engines
                        .iter()
                        .any(|e| e.eq_ignore_ascii_case(engine))
                })
            })
            .peekable();
        if group.peek().is_none() {
            continue;
        }
        let your_gpu = host_gfx_family
            .as_deref()
            .is_some_and(|family| platform_matches_gfx_family(platform, family));
        let heading = model_catalog_platform_heading(&platform.label, your_gpu);
        let _ = writeln!(output, "\n{heading}");
        for recipe in group {
            // Show the canonical Hugging Face id (the reliable serve target) and
            // the quant that fits this hardware. Append a fit verdict only when the
            // host GPU VRAM is known — otherwise every row would read "GPU fit
            // unknown", which is noise.
            let detail = recipe
                .quantization
                .clone()
                .unwrap_or_else(|| model_recipe_memory_label(recipe));
            let fit = if aggregate_gpu_vram_gib.is_some() {
                format!(
                    "  {}",
                    model_recipe_gpu_fit_label(recipe, aggregate_gpu_vram_gib)
                )
            } else {
                String::new()
            };
            let _ = writeln!(output, "  {}  {}{}", recipe.canonical_model_id, detail, fit);
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

fn model_recipe_gpu_fit_label(
    recipe: &ModelRecipeRecord,
    aggregate_gpu_vram_gib: Option<f64>,
) -> &'static str {
    match (recipe.min_gpu_mem_gb, aggregate_gpu_vram_gib) {
        (Some(required), Some(available)) if available >= f64::from(required) => "fits this GPU",
        (Some(_), Some(_)) => "needs a larger GPU",
        (Some(_), None) => "GPU fit unknown",
        (None, _) => "fits",
    }
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
        append_model_fit_lines(&mut output, recipe, aggregate_gpu_vram_gib);
        append_model_engine_support_lines(&mut output, recipe, paths);
        if recipe.trust_remote_code {
            let _ = writeln!(output, "      trust_remote_code: true");
        }
        for warning in &recipe.warnings {
            let _ = writeln!(output, "      warning: {warning}");
        }
        for engine_recipe in &recipe.engine_recipes {
            if let Some(warning) = rocm_core::model_recipe_staleness_warning(
                engine_recipe,
                rocm_core::unix_time_millis(),
            ) {
                let _ = writeln!(output, "      warning: {warning}");
            }
        }
    }
    append_model_recipe_registry_source(&mut output, &registry);
    output
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
) {
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
                preferred_engine_action_target(recipe)
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
                    preferred_engine_action_target(recipe)
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
                // missing VRAM reading is a confident dead end. But naming
                // `amd-smi metric --json` directly is just as much of one here:
                // every production caller of this function hardcodes
                // `aggregate_gpu_vram_gib = None`, so the reading this function
                // sees never changes no matter what the user runs. `rocm
                // diagnose --model` is the command that actually threads a live
                // reading through (`host_accelerator_memory`), so it is pointed
                // at instead of a command this one cannot act on.
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

#[allow(dead_code)]
fn append_model_engine_support_lines(
    output: &mut String,
    recipe: &ModelRecipeRecord,
    paths: Option<&AppPaths>,
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
            if let Some(note) = model_registry_adapter_availability_note(engine) {
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
                    if let Some(note) = model_registry_adapter_availability_note(engine) {
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
        } else if let Some(reason) = missing_packaged_engine_reason(engine) {
            let _ = writeln!(
                output,
                "        {engine}: unavailable reason={}",
                model_registry_reason(reason)
            );
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
fn model_registry_adapter_availability_note(engine: &str) -> Option<&'static str> {
    match engine_ruled_out_by_platform(engine) {
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
fn preferred_engine_action_target(recipe: &ModelRecipeRecord) -> &str {
    recipe
        .preferred_engines
        .first()
        .map_or("<engine>", String::as_str)
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

fn append_engine_detect_summary(output: &mut String, engine: &str, paths: Option<&AppPaths>) {
    let engine_binary = if builtin_engine_available(engine) {
        Ok(PathBuf::new())
    } else {
        match paths {
            Some(paths) => resolve_engine_binary_path_with_paths(engine, paths),
            None => resolve_engine_binary_path(engine),
        }
    };
    let binary_status = if builtin_engine_available(engine) {
        "adapter: built-in".to_owned()
    } else {
        match &engine_binary {
            Ok(_) => "adapter: available".to_owned(),
            Err(_) => "adapter: not found".to_owned(),
        }
    };
    let _ = writeln!(output, "    {binary_status}");
    if !builtin_engine_available(engine)
        && engine_binary.is_err()
        && let Some(reason) = missing_packaged_engine_reason(engine)
    {
        let _ = writeln!(output, "    note: {reason}");
        return;
    }

    let Ok(detect) = engine_request::<_, DetectResponse>(
        paths,
        engine,
        EngineMethod::Detect,
        &DetectRequest {
            runtime_id: None,
            device_filter: None,
        },
    ) else {
        return;
    };

    let _ = writeln!(
        output,
        "    runtime: {}",
        engine_runtime_status_label(engine, &detect)
    );
    if let Some(kind) = detect.runtime_kind.as_deref() {
        let _ = writeln!(
            output,
            "    runtime kind: {}",
            friendly_engine_runtime_kind(kind)
        );
    }
    if detect.runtime_executable.is_some() {
        let _ = writeln!(output, "    runtime executable: available");
    }
    if let Some(note) = friendly_engine_detect_notes(engine, &detect.notes) {
        let _ = writeln!(output, "    note: {note}");
    }
}

fn friendly_engine_runtime_kind(kind: &str) -> String {
    kind.replace('_', " ")
}

fn friendly_engine_detect_notes(engine: &str, notes: &[String]) -> Option<String> {
    if notes.is_empty() {
        return None;
    }
    let combined = notes.join("; ");
    let lower = combined.to_ascii_lowercase();
    if engine.eq_ignore_ascii_case("lemonade") {
        if lower.contains("not installed") || lower.contains("not found") {
            return Some("Lemonade is not installed yet.".to_owned());
        }
        if lower.contains("lemonade embeddable")
            || lower.contains("llamacpp:rocm")
            || lower.contains("configured")
        {
            return Some("Lemonade is ready on your AMD GPU.".to_owned());
        }
    }
    if engine.eq_ignore_ascii_case("vllm")
        && (lower.contains("not installed") || lower.contains("command was not found"))
    {
        return Some("vLLM is not installed in a Linux/WSL ROCm Python environment.".to_owned());
    }
    if rocm_core::runtime_is_windows()
        && (lower.contains("unsupported_native_windows")
            || lower.contains("native windows")
            || lower.contains("linux/wsl")
            || lower.contains("wsl/linux"))
    {
        return Some(format!("{engine} is available from WSL/Linux on Windows."));
    }
    Some(friendly_engine_detect_note_fallback(&combined))
}

fn friendly_engine_detect_note_fallback(note: &str) -> String {
    let lower = note.to_ascii_lowercase();
    if rocm_core::runtime_is_windows()
        && (lower.contains("unsupported_native_windows")
            || lower.contains("native windows")
            || lower.contains("linux/wsl")
            || lower.contains("wsl/linux"))
    {
        return "This engine is available from WSL/Linux on Windows.".to_owned();
    }
    note.to_owned()
}

fn engine_runtime_status_label(engine: &str, detect: &DetectResponse) -> &'static str {
    if detect.installed {
        "ready"
    } else if engine_runtime_is_native_windows_unsupported(engine, detect) {
        "unsupported_native_windows"
    } else {
        "not found"
    }
}

fn engine_runtime_is_native_windows_unsupported(engine: &str, detect: &DetectResponse) -> bool {
    if !rocm_core::runtime_is_windows() || !engine.eq_ignore_ascii_case("vllm") {
        return false;
    }

    detect
        .notes
        .iter()
        .chain(
            detect
                .available_devices
                .iter()
                .filter_map(|device| device.reason.as_ref()),
        )
        .any(|note| {
            let normalized = note.to_ascii_lowercase();
            normalized.contains("linux/wsl") && normalized.contains("native windows")
        })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use rocm_core::resolve_builtin_model_recipe;

    use super::*;
    use crate::runtimes_cmd::runtime_manifest_path;
    use crate::test_support::write_test_pip_runtime;
    use crate::tests::test_paths;

    /// An `ExamineSummary` with the install-reporting fields under test and
    /// everything else inert.
    fn summary_with_legacy(
        managed_runtime_count: usize,
        legacy_status: &str,
        legacy_version: Option<&str>,
    ) -> ExamineSummary {
        ExamineSummary {
            os: "linux".to_owned(),
            arch: "x86_64".to_owned(),
            kernel: None,
            distro: None,
            cpu: None,
            system_ram_gib: None,
            interactive_terminal: false,
            default_engine: "lemonade".to_owned(),
            detected_gfx_target: None,
            compatible_therock_family: None,
            detected_therock_family: None,
            driver: rocm_core::DriverSummary {
                policy: "linux_official_amd_dkms_wrapper".to_owned(),
                status: "amdgpu_available".to_owned(),
                detail: None,
                windows: None,
            },
            legacy_rocm: rocm_core::LegacyRocmSummary {
                status: legacy_status.to_owned(),
                paths: Vec::new(),
                detail: None,
                version: legacy_version.map(str::to_owned),
            },
            wsl: None,
            managed_runtime_count,
            managed_service_count: 0,
            model_cache_entries: 0,
            config_dir: PathBuf::from("config"),
            data_dir: PathBuf::from("data"),
            cache_dir: PathBuf::from("cache"),
        }
    }

    #[test]
    fn header_does_not_claim_nothing_is_installed_when_rocm_is_present() {
        // The reported case: ROCm on the machine, none of it CLI-managed. The
        // header counted only managed runtimes, so it said "No ROCm installs
        // saved yet" — read as "nothing is installed".
        let header = render_examine_plain_header(&summary_with_legacy(
            0,
            "detected_unmanaged",
            Some("7.14.0"),
        ));
        assert!(
            header.contains("existing ROCm 7.14.0 found"),
            "the detected install and its version must appear:\n{header}"
        );
        assert!(
            header.contains("left unmanaged by design"),
            "it must read as a deliberate choice, not an oversight:\n{header}"
        );
    }

    #[test]
    fn header_reports_a_detected_install_whose_version_is_unknown() {
        // Degrades to naming the install without inventing a version.
        let header =
            render_examine_plain_header(&summary_with_legacy(0, "detected_unmanaged", None));
        assert!(
            header.contains("existing ROCm install found"),
            "the install must still be reported:\n{header}"
        );
        assert!(
            !header.contains("<unknown>"),
            "an unknown version must not leak a placeholder into the summary:\n{header}"
        );
    }

    #[test]
    fn header_accounts_for_managed_and_existing_installs_together() {
        let header = render_examine_plain_header(&summary_with_legacy(
            2,
            "detected_unmanaged",
            Some("6.4.1"),
        ));
        assert!(
            header.contains("2 ROCm installs saved"),
            "the managed count must survive:\n{header}"
        );
        assert!(
            header.contains("existing ROCm 6.4.1 found"),
            "the unmanaged install must be reported alongside it:\n{header}"
        );
    }

    #[test]
    fn header_still_says_nothing_is_installed_when_nothing_is() {
        // The wording only changes when there is something to report.
        let header = render_examine_plain_header(&summary_with_legacy(0, "not_detected", None));
        assert!(
            header.contains("No ROCm installs saved yet"),
            "an empty machine must keep the original wording:\n{header}"
        );
        assert!(
            !header.contains("existing ROCm"),
            "nothing should be claimed on an empty machine:\n{header}"
        );
    }

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
    fn render_engine_inventory_text_surfaces_external_plugin_policy() {
        let (root, paths) = test_paths("engine-plugin-policy");

        let rendered = render_engine_inventory_text_with_paths(Some(&paths));
        let _ = fs::remove_dir_all(root);

        assert!(rendered.contains("Local model engines"));
        assert!(rendered.contains("Built-in engines are included with rocm-cli"));
        assert!(rendered.contains("ROCm GPU execution is required"));
        assert!(rendered.contains("Plugin folders:"));
        assert!(rendered.contains(&paths.primary_engine_plugin_dir().display().to_string()));
    }

    #[test]
    fn render_engine_inventory_text_includes_marker_legend() {
        let (root, paths) = test_paths("engine-inventory-marker-legend");
        let host_default =
            rocm_core::default_engine_for_host(&rocm_core::detect_host_gpu_summary(Some(&paths)));

        let rendered = render_engine_inventory_text_with_paths(Some(&paths));
        let _ = fs::remove_dir_all(root);

        let legend = format!("legend: {DEFAULT_ENGINE_MARKER} = default engine");
        let legend_pos = rendered.find(&legend).expect("legend line present");
        // Anchor on the marked row itself (not just `"{DEFAULT_ENGINE_MARKER} "`,
        // which also matches inside the legend line and would pass even if no
        // row were actually marked).
        let marker_pos = rendered
            .find(&format!("{DEFAULT_ENGINE_MARKER} {host_default}"))
            .expect("the default engine entry present");
        assert!(
            legend_pos < marker_pos,
            "legend must appear before the entries it explains:\n{rendered}"
        );
    }

    #[test]
    fn render_engine_inventory_text_omits_legend_when_configured_default_matches_nothing() {
        // A configured default naming an external plugin (or a stale/typo'd
        // name) matches zero rows in `engine_inventory()`'s built-ins list —
        // the legend would then explain a marker that appears nowhere.
        let (root, paths) = test_paths("engine-inventory-unmatched-default");
        let config = RocmCliConfig {
            default_engine: Some("totally-unknown-plugin".to_owned()),
            ..RocmCliConfig::default()
        };
        config.save(&paths).expect("save config");

        let rendered = render_engine_inventory_text_with_paths(Some(&paths));
        let _ = fs::remove_dir_all(root);

        assert!(
            !rendered.contains("legend:"),
            "legend must be absent when the configured default matches no built-in engine; got:\n{rendered}"
        );
    }

    #[test]
    fn render_engine_inventory_text_honors_configured_default_engine() {
        // Regression: this renderer used to mark only `default_engine_for_host`,
        // ignoring a configured `default_engine` — the same host-vs-configured
        // precedence `select_serve_engine` and `append_examine_engine_inventory`
        // already honor. Pick whichever engine the host does NOT prefer so the
        // configured value is guaranteed to actually change the marked engine.
        let (root, paths) = test_paths("engine-inventory-configured-default");
        let host_default =
            rocm_core::default_engine_for_host(&rocm_core::detect_host_gpu_summary(Some(&paths)));
        let configured = if host_default == "vllm" {
            "lemonade"
        } else {
            "vllm"
        };
        let config = RocmCliConfig {
            default_engine: Some(configured.to_owned()),
            ..RocmCliConfig::default()
        };
        config.save(&paths).expect("save config");

        let rendered = render_engine_inventory_text_with_paths(Some(&paths));
        let _ = fs::remove_dir_all(root);

        assert!(
            rendered.contains(&format!("{DEFAULT_ENGINE_MARKER} {configured}")),
            "configured default engine {configured} must be marked; got:\n{rendered}"
        );
        assert!(
            !rendered.contains(&format!("{DEFAULT_ENGINE_MARKER} {host_default}")),
            "host default {host_default} must not be marked once a different engine is configured; got:\n{rendered}"
        );
    }

    #[test]
    fn examine_treats_a_blank_configured_engine_as_unset() {
        // Mirrors `select_serve_engine`'s guard: a config file with
        // `default_engine = ""` must fall back to the host preference rather
        // than reporting an empty engine name as "effective" and marking none
        // of the real ones.
        let (root, paths) = test_paths("examine-engine-inventory-blank-configured");
        let config = RocmCliConfig {
            default_engine: Some(String::new()),
            ..RocmCliConfig::default()
        };
        let mut output = String::new();

        append_examine_engine_inventory(&mut output, &paths, &config, "vllm");

        assert!(
            output.contains("configured_default_engine: <platform default>"),
            "a blank configured value must read as unset:\n{output}"
        );
        assert!(
            output.contains("effective_default_engine: vllm"),
            "a blank configured value must fall back to the host default:\n{output}"
        );
        assert!(
            output.contains("  * vllm "),
            "the '*' marker must land on the host's default, not an empty name:\n{output}"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn friendly_engine_detect_notes_hide_probe_and_path_noise() {
        let lemonade = friendly_engine_detect_notes(
            "lemonade",
            &["Lemonade embeddable 10.6.0 is installed at D:/ROCm/temp/runtime; Lemonade is configured for llamacpp:rocm; no CPU fallback is used".to_owned()],
        )
        .expect("lemonade note");
        assert_eq!(lemonade, "Lemonade is ready on your AMD GPU.");

        let vllm = friendly_engine_detect_notes(
            "vllm",
            &["vLLM is not installed in a Linux/WSL ROCm Python environment. Native Windows is skipped; no CPU fallback is used.".to_owned()],
        )
        .expect("vllm note");
        assert_eq!(
            vllm,
            "vLLM is not installed in a Linux/WSL ROCm Python environment."
        );
        // Raw install paths from the probe body must not leak into the friendly note.
        assert!(!lemonade.contains("D:/"));
    }

    #[test]
    fn linux_only_engine_runtime_status_is_explicit_on_native_windows() {
        let detect = DetectResponse {
            installed: false,
            env_id: None,
            runtime_kind: Some("external_vllm".to_owned()),
            runtime_executable: None,
            managed_env: Some(false),
            python_version: None,
            torch_version: None,
            transformers_version: None,
            available_devices: vec![rocm_engine_protocol::EngineDeviceAvailability {
                kind: "rocm_gpu".to_owned(),
                available: false,
                reason: Some(
                    "vLLM ROCm serving is supported by rocm-cli only on Linux/WSL; native Windows vLLM is skipped. No CPU fallback is used."
                        .to_owned(),
                ),
            }],
            capabilities: rocm_engine_protocol::EngineCapabilities {
                cpu: false,
                rocm_gpu: false,
                openai_compatible: true,
                tool_calling: false,
                quantized_models: "vllm-supported".to_owned(),
                reasoning_parser: false,
            },
            notes: Vec::new(),
        };

        if cfg!(windows) {
            assert_eq!(
                engine_runtime_status_label("vllm", &detect),
                "unsupported_native_windows"
            );
            assert!(
                model_registry_adapter_availability_note("vllm")
                    .is_some_and(|note| note.contains("unsupported_native_windows"))
            );
        } else {
            assert_eq!(engine_runtime_status_label("vllm", &detect), "not found");
            assert!(model_registry_adapter_availability_note("vllm").is_none());
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
        assert!(rendered.contains("AMD Ryzen AI - Strix Halo (Lemonade / llama.cpp / AIM)"));
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
        // With a known aggregate VRAM, each row gains a concrete fit verdict.
        // 60 GiB fits Qwen3.6-27B (~54) but not Gemma-4-31B (~62).
        let rendered = render_model_registry_text_with_context_and_host(None, Some(60.0), None);

        assert!(rendered.contains("fits this GPU"));
        assert!(rendered.contains("needs a larger GPU"));
        assert!(!rendered.contains("GPU fit unknown"));
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
        // Every production caller of this function hardcodes
        // `aggregate_gpu_vram_gib = None` (see `Command::Model` and the `/model`
        // assistant tool), so this branch is the only one `rocm model` ever
        // actually reaches -- its remediation must name a command this same
        // path can act on, not one whose reading this function never sees.
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

        append_model_engine_support_lines(&mut output, &recipe, Some(&paths));
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

    #[test]
    fn examine_runtime_state_reports_active_runtime_key_and_status() -> Result<()> {
        let (root, paths) = test_paths("examine-runtime-state");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all",
            "therock-release:gfx120X-all",
            "7.13.0a20260416",
            1,
        )?;
        let config = RocmCliConfig {
            default_runtime_id: Some(manifest.runtime_id.clone()),
            active_runtime_key: Some(manifest.runtime_key),
            previous_runtime_key: Some("older-release".to_owned()),
            ..RocmCliConfig::default()
        };
        let mut output = String::new();

        append_examine_runtime_state(&mut output, &paths, &config)?;

        assert!(output.contains("runtime_state:"));
        assert!(output.contains("active_runtime_id: therock-release:gfx120X-all"));
        assert!(output.contains("active_runtime_key: release-pip-gfx120x-all"));
        assert!(output.contains("previous_runtime_key: older-release"));
        assert!(output.contains("active_runtime_status: ready"));
        assert!(output.contains("active_runtime_family: gfx120X-all"));
        assert!(output.contains("registered_runtime_keys: release-pip-gfx120x-all"));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    /// `rocm examine` is the first command a user runs when something ROCm is
    /// wrong, and "my build cannot find `hipcc`" is now a reachable state by
    /// design. Reporting the toolchain only from `rocm runtimes list` leaves
    /// the primary diagnostic unable to answer the question the opt-in
    /// created. Both polarities are pinned, because a field hardcoded to either
    /// word reads as working from a single run.
    #[test]
    fn examine_runtime_state_reports_whether_the_active_runtime_has_the_toolchain() -> Result<()> {
        let (root, paths) = test_paths("examine-runtime-toolchain");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-7-13-0",
            "therock-release:gfx120X-all",
            "7.13.0",
            10,
        )?;
        let config = RocmCliConfig {
            active_runtime_key: Some(manifest.runtime_key.clone()),
            ..RocmCliConfig::default()
        };

        // The fixture records `devel: true` with no composition.
        let mut output = String::new();
        append_examine_runtime_state(&mut output, &paths, &config)?;
        assert!(
            output.contains("active_runtime_toolchain: included"),
            "a toolchain runtime must say so:\n{output}"
        );

        // A runtime-only install reports the other way. Written through the
        // recorded specs, which is what `includes_devel` actually reads, so
        // this exercises the same path a real `rocm install sdk` produces.
        let runtime_only = therock::InstalledRuntimeManifest {
            devel: false,
            wheel_composition: Some(therock::WheelRuntimeComposition {
                source_layout_generation: "canonical".to_owned(),
                package_specs: vec!["rocm[libraries,device-gfx1201]==7.13.0".to_owned()],
                rocm_sdk_target: Some("gfx1201".to_owned()),
            }),
            ..manifest
        };
        fs::write(
            runtime_manifest_path(&paths, &runtime_only.runtime_key),
            serde_json::to_vec_pretty(&runtime_only)?,
        )?;
        output.clear();
        append_examine_runtime_state(&mut output, &paths, &config)?;
        assert!(
            output.contains("active_runtime_toolchain: excluded"),
            "a runtime-only runtime must say so:\n{output}"
        );

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn examine_runtime_state_reports_ambiguous_default_runtime_id() -> Result<()> {
        let (root, paths) = test_paths("examine-runtime-ambiguous");
        write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all",
            "therock-release:gfx120X-all",
            "7.13.0a20260416",
            1,
        )?;
        write_test_pip_runtime(
            &paths,
            "vllm-source-pip-gfx120x-all",
            "therock-release:gfx120X-all",
            "7.13.0a20260416",
            2,
        )?;
        let mut config = RocmCliConfig {
            default_runtime_id: Some("therock-release:gfx120X-all".to_owned()),
            ..RocmCliConfig::default()
        };
        let mut output = String::new();

        append_examine_runtime_state(&mut output, &paths, &config)?;

        assert!(output.contains("active_runtime_status: ambiguous_runtime_id"));
        assert!(output.contains("active_runtime_matches:"));
        assert!(output.contains("release-pip-gfx120x-all"));
        assert!(output.contains("vllm-source-pip-gfx120x-all"));
        assert!(output.contains("active_runtime_action: rocm runtimes activate <runtime_key>"));
        assert!(!output.contains("active_runtime_status: missing_manifest"));

        config.active_runtime_key = Some("release-pip-gfx120x-all".to_owned());
        output.clear();
        append_examine_runtime_state(&mut output, &paths, &config)?;

        assert!(output.contains("active_runtime_status: ready"));
        assert!(!output.contains("ambiguous_runtime_id"));
        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn examine_engine_inventory_reports_config_without_engine_detect() {
        let (root, paths) = test_paths("examine-engine-inventory");
        let mut config = RocmCliConfig {
            default_engine: Some("vllm".to_owned()),
            ..RocmCliConfig::default()
        };
        config.engine_config_mut("vllm").preferred_runtime_id =
            Some("therock-release:gfx120X-all".to_owned());
        let mut output = String::new();

        // Host default deliberately disagrees with the configured value: an
        // engine the user chose must outrank whatever the GPU would prefer.
        append_examine_engine_inventory(&mut output, &paths, &config, "lemonade");

        assert!(output.contains("engine_inventory:"));
        assert!(output.contains("configured_default_engine: vllm"));
        assert!(output.contains("effective_default_engine: vllm"));
        assert!(output.contains("* vllm"));
        assert!(output.contains("runtime_pref=therock-release:gfx120X-all"));
        assert!(output.contains("plugin_dirs:"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn examine_engine_inventory_includes_marker_legend() {
        let (root, paths) = test_paths("examine-engine-inventory-marker-legend");
        let config = RocmCliConfig::default();
        let mut output = String::new();

        append_examine_engine_inventory(&mut output, &paths, &config, "vllm");
        let _ = fs::remove_dir_all(root);

        let legend = format!("legend: {DEFAULT_ENGINE_MARKER} = default engine");
        let legend_pos = output.find(&legend).expect("legend line present");
        let marker_pos = output
            .find(&format!("{DEFAULT_ENGINE_MARKER} vllm"))
            .expect("the default engine entry present");
        assert!(
            legend_pos < marker_pos,
            "legend must appear before the entries it explains:\n{output}"
        );
    }

    #[test]
    fn examine_engine_inventory_omits_legend_when_effective_default_matches_nothing() {
        // Same gap as `render_engine_inventory_text_with_paths`: an effective
        // default naming an external plugin (or a stale/typo'd name) matches
        // zero rows in `engine_inventory()`'s built-ins list.
        let (root, paths) = test_paths("examine-engine-inventory-unmatched-default");
        let config = RocmCliConfig {
            default_engine: Some("totally-unknown-plugin".to_owned()),
            ..RocmCliConfig::default()
        };
        let mut output = String::new();

        append_examine_engine_inventory(&mut output, &paths, &config, "vllm");
        let _ = fs::remove_dir_all(root);

        assert!(
            !output.contains("legend:"),
            "legend must be absent when the effective default matches no built-in engine; got:\n{output}"
        );
    }

    // ---------- engine shell prompt shim ----------

    #[test]
    fn examine_engine_inventory_falls_back_to_the_host_engine_when_unconfigured() {
        // With nothing configured, the reported default must be the engine this
        // GPU actually serves on. On an Instinct host that is vLLM; reporting the
        // platform constant here is what made `examine` contradict `serve`.
        let (root, paths) = test_paths("examine-engine-inventory-host");
        let config = RocmCliConfig::default();
        let mut output = String::new();

        append_examine_engine_inventory(&mut output, &paths, &config, "vllm");

        // The configured line still reads as unset — the host default must not be
        // mistaken for something the user chose.
        assert!(output.contains("configured_default_engine: <platform default>"));
        assert!(output.contains("effective_default_engine: vllm"));
        assert!(
            output.contains("* vllm"),
            "the primary marker should follow the host engine:\n{output}"
        );
        // Nothing is being overridden, so there is nothing to warn about.
        assert!(
            !output.contains("configured_default_note"),
            "an unconfigured host must not be warned about an override:\n{output}"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn examine_flags_a_configured_engine_that_overrides_the_host() {
        // Installs provisioned before the installer stopped seeding
        // `default_engine` still carry `lemonade` on disk. That seed is
        // indistinguishable from a deliberate choice, so it is not rewritten --
        // but the user is told why their Instinct box is not on vLLM, and how to
        // change it.
        let (root, paths) = test_paths("examine-engine-inventory-override");
        let config = RocmCliConfig {
            default_engine: Some("lemonade".to_owned()),
            ..RocmCliConfig::default()
        };
        let mut output = String::new();

        append_examine_engine_inventory(&mut output, &paths, &config, "vllm");

        assert!(
            output.contains("effective_default_engine: lemonade"),
            "the configured engine still wins:\n{output}"
        );
        assert!(
            output.contains("configured_default_note:"),
            "the override must be called out:\n{output}"
        );
        assert!(
            output.contains("rocm config clear-default-engine"),
            "the note must name the remedy, not just the problem:\n{output}"
        );
        assert!(
            output.contains("  * lemonade "),
            "the '*' marker must land on the configured engine, not the host's:\n{output}"
        );
        assert!(
            !output.contains("  * vllm "),
            "the host's preferred engine must not also be marked once overridden:\n{output}"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn examine_is_silent_when_the_configured_engine_agrees_with_the_host() {
        let (root, paths) = test_paths("examine-engine-inventory-agrees");
        let config = RocmCliConfig {
            default_engine: Some("vllm".to_owned()),
            ..RocmCliConfig::default()
        };
        let mut output = String::new();

        append_examine_engine_inventory(&mut output, &paths, &config, "vllm");

        assert!(
            !output.contains("configured_default_note"),
            "there is no override to report when the two agree:\n{output}"
        );
        let _ = fs::remove_dir_all(root);
    }
}
