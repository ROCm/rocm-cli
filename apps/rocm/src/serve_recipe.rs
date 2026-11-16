// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Engine-recipe overrides for `rocm serve`.
//!
//! These are the pure transforms `serve()` applies, in order, to the selected
//! engine's launch recipe hint: mapping the catalog's engine record onto the
//! protocol hint, then layering the explicit `--tool-call-parser`,
//! `--gpu-memory-utilization`, `--temperature`/`--top-p`/`--max-tokens` and
//! `--engine-env` controls on top.
//! `serve()` is the only production caller; nothing here calls back into it.

use std::fmt::Write as _;

use anyhow::{Result, bail};
use rocm_core::ModelRecipeRecord;
use rocm_engine_protocol::{
    ENGINE_RECIPE_CONTRACT_VERSION, EngineRecipeEndpointHint, EngineRecipeHint,
    EngineRecipeUnsupportedCombinationHint,
};

pub(crate) fn render_serve_engine_recipe_lines(engine_recipe: &EngineRecipeHint) -> String {
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

pub(crate) fn protocol_engine_recipe_hint(
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
            extra_env: engine_recipe.extra_env.clone(),
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
pub(crate) fn engine_recipe_with_tool_call_override(
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
pub(crate) struct ServeGenerationDefaults {
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) max_tokens: Option<u32>,
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
pub(crate) fn engine_recipe_with_generation_defaults(
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
pub(crate) fn engine_recipe_with_gpu_memory_utilization_override(
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
/// in `serve_cmd.rs`, an explicit CLI value is never silently ignored: a
/// user who types a bad fraction is told so.
pub(crate) fn parse_gpu_memory_utilization(value: Option<&str>) -> Result<Option<f64>> {
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

/// Env vars the engine sets itself, which `--engine-env` may not override.
const RESERVED_ENGINE_ENV_KEYS: [&str; 2] = ["HIP_VISIBLE_DEVICES", "VLLM_API_KEY"];

/// Applies `--engine-env` overrides to the engine recipe's `extra_env` map, for
/// the one engine that consumes it (`aim`, e.g. `AIM_PROFILE_ID`): `KEY=VALUE`
/// sets a value, and a bare `KEY` forwards `KEY` from the engine process's own
/// environment so a secret such as `HF_TOKEN` never lands in argv or the
/// service record (recorded as an empty value, meaning "forward"). Other engines
/// ignore the flag (see [`engine_env_ignored_note`]) and get their hint back
/// unchanged. A minimal hint is synthesized when none exists.
pub(crate) fn engine_recipe_with_extra_env(
    engine: &str,
    hint: Option<EngineRecipeHint>,
    engine_env: &[String],
) -> Result<Option<EngineRecipeHint>> {
    if engine_env.is_empty() || !engine.eq_ignore_ascii_case("aim") {
        return Ok(hint);
    }
    let mut hint = hint.unwrap_or_else(|| EngineRecipeHint {
        contract_version: ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
        engine: engine.to_owned(),
        ..EngineRecipeHint::default()
    });
    for entry in engine_env {
        let (key, value) = entry.split_once('=').unwrap_or((entry.as_str(), ""));
        // A bare `KEY` forwards it from the environment; the hint cannot carry an
        // explicitly empty value, so `KEY=` would silently mean the same thing.
        if entry.contains('=') && value.is_empty() {
            bail!(
                "--engine-env `{entry}` sets an empty value; pass `{key}` alone to forward it \
                 from your environment, or `{key}=VALUE`"
            );
        }
        if key.is_empty() || !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            bail!(
                "--engine-env expects `KEY=VALUE` (or a bare `KEY` to forward it from the \
                 environment) with a non-empty name of letters, digits and `_`; got `{entry}`"
            );
        }
        if RESERVED_ENGINE_ENV_KEYS.contains(&key) {
            bail!("--engine-env cannot set `{key}`; rocm-cli manages it for the engine");
        }
        if value.contains(['\n', '\r']) {
            bail!("--engine-env value for `{key}` cannot contain newlines");
        }
        hint.extra_env.insert(key.to_owned(), value.to_owned());
    }
    Ok(Some(hint))
}

/// `--engine-env` note shown when the flag was passed but the selected engine
/// never consumes `EngineRecipeHint::extra_env` (only `aim` does; see
/// `engine_recipe_with_extra_env`), so the override would otherwise be silently
/// dropped with no indication to the user.
pub(crate) fn engine_env_ignored_note(engine: &str, engine_env: &[String]) -> Option<String> {
    (!engine_env.is_empty() && !engine.eq_ignore_ascii_case("aim")).then(|| {
        format!("--engine-env applies only to engine 'aim'; ignored for engine '{engine}'")
    })
}

/// Whether the resolved engine recipe launches vLLM with tool calling enabled.
pub(crate) fn engine_recipe_enables_tool_choice(hint: Option<&EngineRecipeHint>) -> bool {
    hint.is_some_and(|hint| {
        hint.required_flags
            .iter()
            .any(|flag| flag == "--enable-auto-tool-choice")
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use rocm_core::resolve_builtin_model_recipe;

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
                image_digest: None,
                generated_at_unix_ms: None,
                supported_gfx_families: Vec::new(),
                profile_ids: BTreeMap::new(),
                requires_hf_token: false,
                supported_boards: Vec::new(),
                min_gpu_counts: BTreeMap::new(),
                extra_env: BTreeMap::new(),
            },
            rocm_core::ModelRecipeEngineRecord {
                engine: "lemonade".to_owned(),
                required_flags: vec!["--reasoning-parser".to_owned(), "qwen3".to_owned()],
                parser_settings: BTreeMap::new(),
                preferred_endpoint: None,
                unsupported_combinations: Vec::new(),
                notes: Vec::new(),
                model_id_override: None,
                image_digest: None,
                generated_at_unix_ms: None,
                supported_gfx_families: Vec::new(),
                profile_ids: BTreeMap::new(),
                requires_hf_token: false,
                supported_boards: Vec::new(),
                min_gpu_counts: BTreeMap::new(),
                extra_env: BTreeMap::new(),
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
    fn generation_defaults_reject_aim_instead_of_silently_dropping() {
        // AIM's engine code only ever reads `extra_env`, never `required_flags` — a
        // container launch has no argv to carry `--override-generation-config`
        // into, unlike vLLM's own host process. Writing it into `required_flags`
        // anyway would silently drop `--temperature`/`--top-p`/`--max-tokens` for
        // `rocm serve --engine aim`, so this must fail fast instead.
        let error = engine_recipe_with_generation_defaults(
            "aim",
            None,
            ServeGenerationDefaults {
                temperature: Some(0.5),
                top_p: None,
                max_tokens: None,
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("aim"));
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
    fn extra_env_absent_passes_hint_through_unchanged() {
        assert!(
            engine_recipe_with_extra_env("aim", None, &[])
                .unwrap()
                .is_none()
        );

        let authored = EngineRecipeHint {
            contract_version: ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
            engine: "aim".to_owned(),
            ..EngineRecipeHint::default()
        };
        let hint = engine_recipe_with_extra_env("aim", Some(authored.clone()), &[]).unwrap();
        assert_eq!(hint, Some(authored));
    }

    #[test]
    fn extra_env_synthesizes_hint_and_merges_into_authored_one() {
        let hint = engine_recipe_with_extra_env("aim", None, &["AIM_PROFILE_ID=foo".to_owned()])
            .unwrap()
            .expect("--engine-env should synthesize a hint");
        assert_eq!(hint.engine, "aim");
        assert_eq!(
            hint.extra_env.get("AIM_PROFILE_ID"),
            Some(&"foo".to_owned())
        );

        let authored = EngineRecipeHint {
            contract_version: ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
            engine: "aim".to_owned(),
            extra_env: BTreeMap::from([("EXISTING".to_owned(), "kept".to_owned())]),
            ..EngineRecipeHint::default()
        };
        let hint =
            engine_recipe_with_extra_env("aim", Some(authored), &["AIM_PROFILE_ID=foo".to_owned()])
                .unwrap()
                .unwrap();
        assert_eq!(hint.extra_env.get("EXISTING"), Some(&"kept".to_owned()));
        assert_eq!(
            hint.extra_env.get("AIM_PROFILE_ID"),
            Some(&"foo".to_owned())
        );
    }

    #[test]
    fn extra_env_bare_key_forwards_from_the_environment_without_recording_a_value() {
        let hint = engine_recipe_with_extra_env("aim", None, &["HF_TOKEN".to_owned()])
            .unwrap()
            .unwrap();
        assert_eq!(hint.extra_env.get("HF_TOKEN"), Some(&String::new()));
    }

    #[test]
    fn extra_env_rejects_entry_with_empty_or_malformed_key() {
        for entry in ["=some-value", "", "BAD KEY=1", "A-B=1"] {
            let error = engine_recipe_with_extra_env("aim", None, &[entry.to_owned()]).unwrap_err();
            assert!(error.to_string().contains("KEY=VALUE"), "{entry}: {error}");
        }
    }

    #[test]
    fn extra_env_rejects_reserved_keys_and_newlines() {
        for entry in ["VLLM_API_KEY=x", "HIP_VISIBLE_DEVICES=0", "VLLM_API_KEY"] {
            let error = engine_recipe_with_extra_env("aim", None, &[entry.to_owned()]).unwrap_err();
            assert!(error.to_string().contains("manages it"), "{entry}: {error}");
        }
        assert!(engine_recipe_with_extra_env("aim", None, &["A=b\nc".to_owned()]).is_err());
    }

    #[test]
    fn extra_env_leaves_the_hint_alone_for_engines_that_ignore_it() {
        assert_eq!(
            engine_recipe_with_extra_env("vllm", None, &["FOO=bar".to_owned()]).unwrap(),
            None,
            "a flag reported as ignored must not change what the engine is sent"
        );
    }

    #[test]
    fn engine_env_distinguishes_a_bare_key_from_an_empty_value() {
        let env = |entry: &str| engine_recipe_with_extra_env("aim", None, &[entry.to_owned()]);
        let bare = env("HF_TOKEN").unwrap().expect("hint");
        assert_eq!(bare.extra_env.get("HF_TOKEN").map(String::as_str), Some(""));
        let err = env("HF_TOKEN=").unwrap_err().to_string();
        assert!(
            err.contains("empty value") && err.contains("`HF_TOKEN` alone"),
            "{err}"
        );
        let set = env("A=b").unwrap().expect("hint");
        assert_eq!(set.extra_env.get("A").map(String::as_str), Some("b"));
    }

    #[test]
    fn engine_env_ignored_note_is_none_for_aim() {
        assert_eq!(
            engine_env_ignored_note("aim", &["AIM_PROFILE_ID=foo".to_owned()]),
            None
        );
    }

    #[test]
    fn engine_env_ignored_note_is_none_when_engine_env_is_empty() {
        assert_eq!(engine_env_ignored_note("vllm", &[]), None);
    }

    #[test]
    fn engine_env_ignored_note_warns_for_engines_that_do_not_consume_it() {
        let note = engine_env_ignored_note("vllm", &["FOO=bar".to_owned()])
            .expect("vllm does not consume extra_env");
        assert!(note.contains("--engine-env"));
        assert!(note.contains("'aim'"));
        assert!(note.contains("'vllm'"));
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
