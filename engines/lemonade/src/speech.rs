// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Serving Lemonade's speech models: speech-to-text through its `whispercpp`
//! recipe and text-to-speech through its `openmoss` recipe.
//!
//! These models always run behind Lemonade's router (`lemond`): the direct
//! `llama-server` path the chat flow can take on Linux has no audio routes. The
//! backend is chosen the same way as for chat — ROCm first, Vulkan when Lemonade
//! reports ROCm unsupported, never CPU — and is named explicitly on load,
//! because Lemonade's `auto` resolves to whatever build is installed, which for
//! whisper.cpp can be the CPU one. (OpenMOSS has no CPU build at all.)
//!
//! What this can and cannot verify: Lemonade 11.5.1 reports a loaded model's
//! `device`, but derives it from the backend *name* — a `rocm` Whisper is
//! reported as `cpu`, and a `vulkan` one as `gpu` even where whisper.cpp found
//! no Vulkan device and ran on the CPU. So the field proves nothing either way
//! and is not consulted. What is checked is the backend Lemonade actually loaded
//! (`recipe_options.<recipe>_backend`), exactly as the chat path checks
//! `llamacpp_backend`. `rocm serve`'s usable-AMD-GPU gate, which runs before
//! any engine is started, ensures an AMD GPU is present, but not that a Vulkan
//! build of the speech backend can use it: with no Vulkan device it may run on
//! the CPU unreported. That gap is shared with the chat path's Vulkan llama.cpp
//! and is documented in docs/engine-plugins.md.

use anyhow::{Context, Result, anyhow, bail};
use rocm_core::{ModelTask, format_http_base_url};
use rocm_engine_protocol::{DevicePolicy, EngineRecipeHint, ResolveModelResponse};
use serde_json::{Value, json};
use std::path::Path;
use std::time::Duration;

use crate::backend_alignment::LemonadeRuntime;
use crate::process::LemonadeProcessEnvironment;
use crate::{DEFAULT_HOST, ServeHttpRequest};

/// How Lemonade serves one kind of speech model.
#[derive(Debug)]
pub(crate) struct SpeechRecipe {
    pub(crate) task: ModelTask,
    /// Lemonade's recipe name, as `lemonade backends` and `/v1/health` report it.
    pub(crate) recipe: &'static str,
    /// The `lemonade load` flag that names the backend.
    load_flag: &'static str,
    /// The `recipe_options` key that reports the backend a model was loaded on.
    backend_option: &'static str,
    /// The weight format, as the catalog records it.
    dtype: &'static str,
    /// What the user is told about the model's memory needs.
    estimated_memory: &'static str,
    /// Shown when the backend fails to install twice in a row. The download is
    /// retried by every serve that finds the backend missing, so running the
    /// same command again is the step that installs it.
    install_advice: &'static str,
}

pub(crate) const TRANSCRIPTION: SpeechRecipe = SpeechRecipe {
    task: ModelTask::Transcription,
    recipe: "whispercpp",
    load_flag: "--whispercpp",
    backend_option: "whispercpp_backend",
    dtype: "ggml",
    estimated_memory: "under 2 GiB for Whisper Large-v3-Turbo; well under 1 GiB for the \
                       smaller Whisper models",
    install_advice: "Lemonade could not install its speech-to-text (whisper.cpp) backend; \
                     check the network connection and run the same `rocm serve` command \
                     again, which retries the download",
};

pub(crate) const SPEECH: SpeechRecipe = SpeechRecipe {
    task: ModelTask::Speech,
    recipe: "openmoss",
    load_flag: "--openmoss",
    backend_option: "openmoss_backend",
    dtype: "gguf",
    estimated_memory: "about 14 GiB of GPU memory for OpenMOSS-TTS (8-bit)",
    install_advice: "Lemonade could not install its text-to-speech (OpenMOSS) backend; \
                     check the network connection and run the same `rocm serve` command \
                     again, which retries the download",
};

impl SpeechRecipe {
    pub(crate) const fn health_keys(&self) -> (&'static str, &'static str) {
        (self.recipe, self.backend_option)
    }
}

/// The recipe a speech task is served with, or `None` for one that is not a
/// speech task.
pub(crate) const fn speech_recipe(task: ModelTask) -> Option<&'static SpeechRecipe> {
    match task {
        ModelTask::Transcription => Some(&TRANSCRIPTION),
        ModelTask::Speech => Some(&SPEECH),
        ModelTask::Chat | ModelTask::Unknown => None,
    }
}

/// GPU backends only, best first. `cpu` is deliberately absent: under the
/// GPU-required policy a host without a usable GPU backend fails instead of
/// running the model on the CPU.
const SPEECH_BACKEND_PRIORITY: [&str; 2] = ["rocm", "vulkan"];

/// How long `lemond` gets to answer its status query after it is spawned.
const LEMOND_STARTUP_TIMEOUT: Duration = Duration::from_secs(30);

pub(crate) fn resolve_speech_model(
    recipe: &SpeechRecipe,
    canonical_model_id: String,
    device_policy: DevicePolicy,
    engine_recipe: Option<EngineRecipeHint>,
) -> ResolveModelResponse {
    ResolveModelResponse {
        canonical_model_id,
        task: recipe.task.as_str().to_owned(),
        source: "lemonade".to_owned(),
        revision: "main".to_owned(),
        loader: recipe.recipe.to_owned(),
        trust_remote_code: false,
        chat_template_mode: "none".to_owned(),
        dtype: recipe.dtype.to_owned(),
        device_policy,
        estimated_memory: recipe.estimated_memory.to_owned(),
        launch_defaults: json!({
            "host": DEFAULT_HOST,
            "port": rocm_core::DEFAULT_LOCAL_PORT,
            "endpoint_mode": "openai"
        }),
        engine_recipe,
        warnings: vec![format!(
            "Lemonade runs this model on its ROCm {recipe} backend, or Vulkan when ROCm is \
             unsupported on this GPU; a CPU build is never selected",
            recipe = recipe.recipe
        )],
    }
}

/// Serve a speech model until the router exits.
pub(crate) fn serve_speech_model(
    recipe: &SpeechRecipe,
    request: &ServeHttpRequest,
    runtime: &LemonadeRuntime,
    process_env: &LemonadeProcessEnvironment,
    log_path: Option<&Path>,
) -> Result<()> {
    let manifest = &runtime.manifest;
    let mut child =
        crate::process::spawn_lemond(manifest, &request.host, request.port, log_path, process_env)?;
    crate::state::write_running_state(
        request,
        runtime,
        std::process::id(),
        Some(child.id()),
        "running",
    )?;
    let prepared = prepare_speech_model(recipe, request, runtime, process_env, log_path);
    let backend = match prepared {
        Ok(backend) => backend,
        Err(error) => {
            let _ = crate::state::terminate_pid(child.id(), true);
            let _ = child.wait();
            return Err(error);
        }
    };
    crate::state::merge_json_state(
        &request.state_path,
        &json!({
            "status": "ready",
            // The post-load smoke test was a real request answered with 200,
            // so the CLI need not repeat it before calling the server ready.
            rocm_core::INFERENCE_VERIFIED_STATE_KEY: rocm_core::unix_time_millis() as u64,
            "server_pid": child.id(),
            "server_start_ticks": rocm_core::process_start_ticks(child.id()),
            "backend_state": "ready",
            "backend_requested": backend,
            "load_response": {
                "status": "loaded",
                "method": "lemonade-cli",
                "model_name": request.model_ref,
                recipe.backend_option: backend
            },
        }),
    )?;
    let status = child.wait().context("failed waiting for Lemonade server")?;
    crate::state::mark_json_status(
        &request.state_path,
        if status.success() {
            "stopped"
        } else {
            "failed"
        },
    )?;
    if status.success() {
        Ok(())
    } else {
        bail!("Lemonade server exited with status {status}")
    }
}

/// Bring the router to the point of serving: pick and install a GPU backend,
/// load the model on it, and prove one real request succeeds. Returns the
/// backend in use.
fn prepare_speech_model(
    recipe: &SpeechRecipe,
    request: &ServeHttpRequest,
    runtime: &LemonadeRuntime,
    process_env: &LemonadeProcessEnvironment,
    log_path: Option<&Path>,
) -> Result<String> {
    let manifest = &runtime.manifest;
    crate::process::wait_for_lemonade_cli_status(
        manifest,
        &request.host,
        request.port,
        LEMOND_STARTUP_TIMEOUT,
        log_path,
        process_env,
    )
    .context("Lemonade server did not become ready")?;
    let listing = crate::backend_alignment::run_lemonade_backends_list(manifest, process_env)?;
    let backend = select_backend(recipe, &listing)?;
    if backend.already_installed {
        eprintln!(
            "Using installed Lemonade {}:{} backend.",
            recipe.recipe, backend.name
        );
    } else {
        eprintln!(
            "Installing Lemonade {}:{} backend...",
            recipe.recipe, backend.name
        );
        crate::backend_alignment::install_backend_with_retry(
            || {
                crate::process::run_lemonade_recipe_backend_install(
                    manifest,
                    &request.host,
                    request.port,
                    recipe.recipe,
                    &backend.name,
                    process_env,
                )
            },
            recipe.install_advice,
        )?;
    }
    crate::process::run_lemonade_load_command(
        manifest,
        &load_args(
            recipe,
            &request.host,
            request.port,
            &request.model_ref,
            &backend.name,
        ),
        log_path,
        process_env,
    )
    .with_context(|| {
        format!(
            "failed to load {} with Lemonade {}:{}",
            request.model_ref, recipe.recipe, backend.name
        )
    })?;
    let health = crate::process::query_health_json(&request.host, request.port)?;
    if !health_has_model(recipe, &health, &request.model_ref, &backend.name) {
        bail!(
            "Lemonade load completed but the server did not report {} loaded on its {}:{} \
             backend",
            request.model_ref,
            recipe.recipe,
            backend.name
        );
    }
    let status = rocm_core::openai_inference_status(
        recipe.task,
        &format_http_base_url(&request.host, request.port),
        &request.model_ref,
        rocm_engine_protocol::resolve_endpoint_api_key().as_deref(),
        rocm_core::inference_probe_timeout(recipe.task),
    )
    .with_context(|| {
        format!(
            "the post-load {} smoke test did not complete",
            recipe.task.display_name()
        )
    })?;
    if status != 200 {
        bail!(
            "{} loaded, but a test {} request returned HTTP {status}",
            request.model_ref,
            recipe.task.display_name()
        );
    }
    Ok(backend.name)
}

#[derive(Debug, PartialEq, Eq)]
struct SelectedBackend {
    name: String,
    already_installed: bool,
}

/// Pick the recipe's backend from `lemonade backends` output.
fn select_backend(recipe: &SpeechRecipe, listing: &str) -> Result<SelectedBackend> {
    let statuses = crate::backend_alignment::parse_backend_statuses(listing, recipe.recipe);
    let (name, already_installed) =
        crate::backend_alignment::select_best_backend(&statuses, &SPEECH_BACKEND_PRIORITY)
            .ok_or_else(|| {
                anyhow!(
                    "Lemonade reports no supported GPU {} backend for this host (status: {}). \
                     The GPU-required policy does not fall back to CPU; check the GPU driver \
                     with `rocm examine`.",
                    recipe.task.display_name(),
                    crate::backend_alignment::describe_backend_statuses(&statuses)
                )
            })?;
    Ok(SelectedBackend {
        name,
        already_installed,
    })
}

fn load_args(
    recipe: &SpeechRecipe,
    host: &str,
    port: u16,
    model_ref: &str,
    backend: &str,
) -> Vec<String> {
    vec![
        "--host".to_owned(),
        host.to_owned(),
        "--port".to_owned(),
        port.to_string(),
        "load".to_owned(),
        model_ref.to_owned(),
        recipe.load_flag.to_owned(),
        backend.to_owned(),
        "--save-options".to_owned(),
    ]
}

/// Whether Lemonade reports `model_ref` loaded through the recipe on exactly
/// `backend`.
fn health_has_model(recipe: &SpeechRecipe, health: &Value, model_ref: &str, backend: &str) -> bool {
    let model_ref = model_ref.trim();
    if model_ref.is_empty() {
        return false;
    }
    health
        .get("all_models_loaded")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|model| {
            let str_field = |key: &str| model.get(key).and_then(Value::as_str);
            str_field("model_name").is_some_and(|name| name.eq_ignore_ascii_case(model_ref))
                && str_field("recipe") == Some(recipe.recipe)
                && model
                    .get("recipe_options")
                    .and_then(|options| options.get(recipe.backend_option))
                    .and_then(Value::as_str)
                    == Some(backend)
        })
}

/// The healthcheck's "listed" signal for a speech service.
pub(crate) fn query_loaded_speech_model(
    recipe: &SpeechRecipe,
    endpoint_url: &str,
    model_ref: &str,
    backend: &str,
) -> Result<bool> {
    let (host, port) = crate::state::parse_http_endpoint(endpoint_url)
        .with_context(|| format!("unsupported endpoint URL `{endpoint_url}`"))?;
    let health = crate::process::query_health_json(&host, port)?;
    Ok(health_has_model(recipe, &health, model_ref, backend))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `lemonade backends --all` output captured from Lemonade 11.5.1 on a
    /// Linux host where ROCm is unsupported, trimmed to the rows around
    /// `whispercpp` plus the recipes 11.5.1 added before it.
    const BACKENDS_NO_ROCM: &str = "\
Recipe              Backend     Status          Message/Version                               Action
----------------------------------------------------------------------------------------------------
acestep             cuda        unsupported     Unsupported GPU                                -
                    rocm        unsupported     Unsupported GPU                                -
                    vulkan      installable     Backend is supported but not installed.        lemonade backends install acestep:vulkan
kokoro              cpu         installable     Backend is supported but not installed.        lemonade backends install kokoro:cpu
llamacpp            cpu         installable     Backend is supported but not installed.        lemonade backends install llamacpp:cpu
                    rocm        unsupported     Unsupported GPU                                -
                    vulkan      installed       b9747                                          -
openmoss            rocm        unsupported     Unsupported GPU                                -
                    vulkan      installable     Backend is supported but not installed.        lemonade backends install openmoss:vulkan
vllm                rocm        unsupported     Unsupported GPU                                -
whispercpp          cpu         installed       v1.8.4                                         -
                    metal       unsupported     Requires macOS                                 -
                    npu         unsupported     Requires Windows                               -
                    rocm        unsupported     Unsupported GPU                                -
                    vulkan      installable     Backend is supported but not installed.        lemonade backends install whispercpp:vulkan
";

    #[test]
    fn rocm_unsupported_selects_vulkan_even_when_the_cpu_build_is_installed() {
        assert_eq!(
            select_backend(&TRANSCRIPTION, BACKENDS_NO_ROCM).expect("vulkan is usable"),
            SelectedBackend {
                name: "vulkan".to_owned(),
                already_installed: false,
            }
        );
    }

    #[test]
    fn rocm_is_preferred_when_lemonade_supports_it() {
        let listing = BACKENDS_NO_ROCM.replace(
            "                    rocm        unsupported     Unsupported GPU                                -\n                    vulkan      installable     Backend is supported but not installed.        lemonade backends install whispercpp:vulkan",
            "                    rocm        installed       v1.8.4                                         -\n                    vulkan      installable     Backend is supported but not installed.        lemonade backends install whispercpp:vulkan",
        );
        assert_eq!(
            select_backend(&TRANSCRIPTION, &listing).expect("rocm is usable"),
            SelectedBackend {
                name: "rocm".to_owned(),
                already_installed: true,
            }
        );
    }

    #[test]
    fn only_a_cpu_backend_is_refused_rather_than_used() {
        let listing = "\
Recipe              Backend     Status          Message/Version                               Action
whispercpp          cpu         installed       v1.8.4                                         -
                    rocm        unsupported     Unsupported GPU                                -
                    vulkan      unsupported     Unsupported GPU                                -
";
        let error = select_backend(&TRANSCRIPTION, listing)
            .expect_err("CPU must never be selected")
            .to_string();
        assert!(error.contains("does not fall back to CPU"), "{error}");
        assert!(error.contains("cpu=installed"), "{error}");
    }

    #[test]
    fn rows_of_recipes_listed_before_whispercpp_are_not_mistaken_for_its_backends() {
        // `acestep` and `openmoss` are 11.5.1 recipes with a usable `vulkan`
        // row; were they unknown to the parser, those rows would be read as
        // continuation rows of the recipe above them.
        let statuses = crate::backend_alignment::parse_backend_statuses(
            BACKENDS_NO_ROCM,
            TRANSCRIPTION.recipe,
        );
        assert_eq!(
            statuses,
            vec![
                ("cpu".to_owned(), "installed".to_owned()),
                ("metal".to_owned(), "unsupported".to_owned()),
                ("npu".to_owned(), "unsupported".to_owned()),
                ("rocm".to_owned(), "unsupported".to_owned()),
                ("vulkan".to_owned(), "installable".to_owned()),
            ]
        );
    }

    #[test]
    fn rows_of_recipes_listed_after_llamacpp_are_not_attached_to_it() {
        // 11.5.1 lists `openmoss` and `vllm` after `llamacpp`. Before those
        // recipes were known, their rows read as more llama.cpp backends.
        let statuses = crate::backend_alignment::parse_backend_statuses(
            BACKENDS_NO_ROCM,
            crate::LLAMACPP_RECIPE,
        );
        assert_eq!(
            statuses,
            vec![
                ("cpu".to_owned(), "installable".to_owned()),
                ("rocm".to_owned(), "unsupported".to_owned()),
                ("vulkan".to_owned(), "installed".to_owned()),
            ]
        );
    }

    #[test]
    fn a_failed_whisper_backend_install_is_retried_once_then_names_a_working_remedy() {
        let mut attempts = 0;
        let error = crate::backend_alignment::install_backend_with_retry(
            || {
                attempts += 1;
                anyhow::bail!("download interrupted")
            },
            TRANSCRIPTION.install_advice,
        )
        .expect_err("both attempts fail");
        assert_eq!(attempts, 2);
        assert!(
            error
                .to_string()
                .contains("run the same `rocm serve` command again"),
            "{error}"
        );
        // The remedy holds because a serve that finds the backend missing
        // installs it: an `installable` backend is selected as not yet
        // installed, which is what sends `prepare_speech_model` down the
        // install path again.
        assert!(
            !select_backend(&TRANSCRIPTION, BACKENDS_NO_ROCM)
                .expect("vulkan is usable")
                .already_installed
        );
    }

    #[test]
    fn text_to_speech_uses_the_openmoss_gpu_backend_and_its_own_load_flag() {
        assert_eq!(
            select_backend(&SPEECH, BACKENDS_NO_ROCM).expect("vulkan is usable"),
            SelectedBackend {
                name: "vulkan".to_owned(),
                already_installed: false,
            }
        );
        assert_eq!(
            load_args(&SPEECH, "127.0.0.1", 13305, "OpenMOSS-TTS", "vulkan")[6..8],
            ["--openmoss", "vulkan"]
        );
        let loaded = json!({"all_models_loaded": [{
            "model_name": "OpenMOSS-TTS",
            "recipe": "openmoss",
            "recipe_options": {"openmoss_backend": "vulkan"}
        }]});
        assert!(health_has_model(&SPEECH, &loaded, "OpenMOSS-TTS", "vulkan"));
        assert!(
            !health_has_model(&TRANSCRIPTION, &loaded, "OpenMOSS-TTS", "vulkan"),
            "a text-to-speech model is not a loaded speech-to-text one"
        );
    }

    #[test]
    fn each_speech_task_reads_its_own_recipe_row() {
        // ROCm usable for OpenMOSS but not for whisper.cpp: each task must pick
        // from its own recipe's rows, not the other's.
        let listing = BACKENDS_NO_ROCM.replace(
            "openmoss            rocm        unsupported     Unsupported GPU                                -",
            "openmoss            rocm        installable     Backend is supported but not installed.        lemonade backends install openmoss:rocm",
        );
        assert_ne!(listing, BACKENDS_NO_ROCM, "fixture edit applied");
        assert_eq!(
            select_backend(&SPEECH, &listing)
                .expect("rocm is usable")
                .name,
            "rocm"
        );
        assert_eq!(
            select_backend(&TRANSCRIPTION, &listing)
                .expect("vulkan is usable")
                .name,
            "vulkan"
        );
    }

    #[test]
    fn only_speech_tasks_have_a_speech_recipe() {
        assert_eq!(
            speech_recipe(ModelTask::Transcription).map(|r| r.recipe),
            Some("whispercpp")
        );
        assert_eq!(
            speech_recipe(ModelTask::Speech).map(|r| r.recipe),
            Some("openmoss")
        );
        assert!(speech_recipe(ModelTask::Chat).is_none());
        assert!(speech_recipe(ModelTask::Unknown).is_none());
    }

    #[test]
    fn load_names_the_backend_explicitly_instead_of_auto() {
        assert_eq!(
            load_args(&TRANSCRIPTION, "127.0.0.1", 13305, "Whisper-Tiny", "rocm"),
            vec![
                "--host",
                "127.0.0.1",
                "--port",
                "13305",
                "load",
                "Whisper-Tiny",
                "--whispercpp",
                "rocm",
                "--save-options",
            ]
        );
    }

    /// `all_models_loaded` as Lemonade 11.5.1 reported it for Whisper-Tiny
    /// loaded with `--whispercpp vulkan` (captured, then trimmed).
    fn loaded(backend: &str, device: &str) -> Value {
        json!({
            "all_models_loaded": [{
                "model_name": "Whisper-Tiny",
                "recipe": "whispercpp",
                "device": device,
                "type": "transcription",
                "recipe_options": {"ctx_size": 4096, "whispercpp_backend": backend}
            }]
        })
    }

    #[test]
    fn loaded_model_must_be_on_the_requested_backend() {
        assert!(health_has_model(
            &TRANSCRIPTION,
            &loaded("vulkan", "gpu"),
            "Whisper-Tiny",
            "vulkan"
        ));
        assert!(
            !health_has_model(
                &TRANSCRIPTION,
                &loaded("cpu", "cpu"),
                "Whisper-Tiny",
                "vulkan"
            ),
            "a model Lemonade loaded on its CPU build is not the one we asked for"
        );
        assert!(!health_has_model(
            &TRANSCRIPTION,
            &loaded("vulkan", "gpu"),
            "Whisper-Base",
            "vulkan"
        ));
    }

    #[test]
    fn the_device_label_is_not_trusted_either_way() {
        // Lemonade 11.5.1 labels a ROCm whisper.cpp model `cpu`; the backend
        // decides, not the label.
        assert!(health_has_model(
            &TRANSCRIPTION,
            &loaded("rocm", "cpu"),
            "Whisper-Tiny",
            "rocm"
        ));
    }
}
