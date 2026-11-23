// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The AIM checks `rocm serve` runs before it launches an AIM container: the
//! GPU-family profile pin and refusal, the Hugging Face token check for a gated
//! image, the multi-GPU refusal, and the board note.
//!
//! Relocated verbatim from `serve_cmd.rs`, with its tests, to keep that file
//! under the module-size limit; `serve` is the only production caller.

use anyhow::{Result, bail};
use rocm_core::{GfxSupport, ModelRecipeRecord, engine_recipe_gfx_support};
use rocm_engine_protocol::EngineRecipeHint;

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
pub(crate) fn aim_gpu_preflight(
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

pub(crate) fn aim_engine_record<'a>(
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
pub(crate) fn ensure_aim_hf_token(
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
pub(crate) fn aim_board_note(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve_recipe::{engine_recipe_with_extra_env, protocol_engine_recipe_hint};
    use rocm_core::resolve_builtin_model_recipe;

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
}
