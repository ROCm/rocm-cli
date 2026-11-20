// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Lemonade's Gufo backend: whether it can run on this host, and getting it
//! installed before a Gufo model is loaded.
//!
//! Gufo is an inference engine with model-specific kernels for AMD Strix Halo
//! (gfx1151). Lemonade runs it from a container image (Podman or Docker) as the
//! `gufo` recipe's only backend, `rocm`. Lemonade itself decides whether that
//! backend can run here and reports one of these states for it in
//! `/api/v1/system-info`:
//!
//! - `installed` (or `update_available`): the image is on disk; load the model.
//! - `installable` (or `update_required`): pull the image, then load.
//! - `action_required`: the container setup is incomplete (no Podman or
//!   Docker, Podman on runc instead of crun, the account not in `render`).
//!   Lemonade's documentation warns that a container started under runc cannot
//!   open the GPU and runs on the CPU, so this is a refusal, never a load.
//! - `unsupported`: not a Strix Halo GPU, or not Linux.
//!
//! The answer is read from Lemonade rather than re-derived here, so the GPU and
//! container checks stay the ones the backend itself applies.

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use rocm_engine_protocol::{DevicePolicy, EngineRecipeHint, ResolveModelResponse};

use crate::install::LemonadeInstallManifest;
use crate::process::LemonadeProcessEnvironment;
use crate::runtime_dir::ParentRuntimeEnvironment;

pub use crate::recipe::GUFO_RECIPE;
/// What a flags refusal on the `ResolveModel` round trip left undone.
pub(crate) const REFUSED_BEFORE_ANY_SERVE: &str = "nothing was started";
/// What a flags refusal at backend selection left undone: Lemonade's server is
/// up by then, but nothing was installed or loaded on it.
pub(crate) const REFUSED_BEFORE_BACKEND: &str =
    "the Gufo image was not pulled and the model was not loaded";
/// How the engine's detect note about Gufo starts. `rocm engines` and
/// `rocm examine` print a note with this prefix on a line of its own instead of
/// folding it into their one-line summary of the engine.
pub const GUFO_DETECT_NOTE_PREFIX: &str = "Lemonade Gufo backend";
/// The Gufo recipe's only backend.
pub(crate) const GUFO_BACKEND: &str = "rocm";

/// How long a `/api/v1/system-info` request may take. Lemonade probes the
/// hardware and the container engine to answer it, so allow more than a
/// health check.
const SYSTEM_INFO_TIMEOUT: Duration = Duration::from_secs(30);

/// A JSON file whose contents stand in for `lemond`'s `/api/v1/system-info`
/// answer, so a scenario can drive every Gufo state without a Strix Halo, a
/// container engine or a running `lemond`.
#[cfg(feature = "e2e-test-hooks")]
pub(crate) const SYSTEM_INFO_TEST_ENV: &str = "ROCM_E2E_LEMONADE_SYSTEM_INFO";

/// What Lemonade reports for the `gufo:rocm` backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct GufoBackendStatus {
    /// Lemonade's state word for the backend (see the module docs).
    pub(crate) state: String,
    /// Lemonade's explanation, one problem per line; may be empty.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub(crate) message: String,
    /// The steps Lemonade says complete the setup, one per line; may be empty.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub(crate) action: String,
}

/// What a serve of a Gufo model must do before loading it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GufoReadiness {
    /// The backend is installed; load the model.
    Installed,
    /// The backend can run here but its image is not on disk yet.
    NeedsInstall,
}

/// Read the `gufo:rocm` row out of a `/api/v1/system-info` document. `None`
/// when this Lemonade has no Gufo recipe at all.
pub(crate) fn status_from_system_info(system_info: &Value) -> Option<GufoBackendStatus> {
    let backend = system_info
        .get("recipes")?
        .get(GUFO_RECIPE)?
        .get("backends")?
        .get(GUFO_BACKEND)?;
    let text = |field: &str| {
        backend
            .get(field)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_owned()
    };
    Some(GufoBackendStatus {
        // Lemonade's own CLI prints an absent state as `unsupported`.
        state: Some(text("state"))
            .filter(|state| !state.is_empty())
            .unwrap_or_else(|| "unsupported".to_owned()),
        message: text("message"),
        action: text("action"),
    })
}

/// Decide what a serve of a Gufo model does with the backend Lemonade reports,
/// or refuse it with the reason and what to do about it. Neither the model nor the
/// Gufo image is downloaded before this answers (Lemonade itself may have been
/// installed by then: `rocm serve` installs it before asking), and no answer here
/// ever leads to a CPU run.
pub(crate) fn readiness(status: Option<&GufoBackendStatus>) -> Result<GufoReadiness> {
    let Some(status) = status else {
        bail!(
            "the installed Lemonade reports no Gufo backend, so it cannot serve Gufo models \
             on this machine; the model and the Gufo image were not downloaded, and the \
             model is not run on any other backend or on the CPU"
        );
    };
    match status.state.as_str() {
        "installed" => Ok(GufoReadiness::Installed),
        state if state.starts_with("update_available") => Ok(GufoReadiness::Installed),
        "installable" | "update_required" => Ok(GufoReadiness::NeedsInstall),
        "action_required" => bail!("{}", setup_incomplete_message(status)),
        "unsupported" => bail!("{}", unsupported_message(status)),
        other => bail!(
            "Lemonade reports the Gufo backend as `{other}`, a state rocm-cli does not \
             know how to serve from; Gufo was not started{}",
            reported_suffix(&status.message)
        ),
    }
}

fn unsupported_message(status: &GufoBackendStatus) -> String {
    format!(
        "Gufo models run only on an AMD Strix Halo GPU (gfx1151) under Linux, and \
         Lemonade reports the Gufo backend as unsupported on this machine{}. The model \
         and the Gufo image were not downloaded, and the model is not run on any other \
         backend or on the CPU. `rocm model` lists the models that serve on other AMD GPUs.",
        reported_suffix(&status.message)
    )
}

fn setup_incomplete_message(status: &GufoBackendStatus) -> String {
    let mut message = String::from(
        "Lemonade runs Gufo from a container image, and the container setup on this \
         machine is not complete, so a Gufo container could not reach the GPU. The model \
         and the Gufo image were not downloaded, Gufo was not started, and the model is \
         not run on the CPU instead.",
    );
    let problems = lines(&status.message);
    if !problems.is_empty() {
        message.push_str("\n\nLemonade reports:");
        for line in problems {
            message.push_str("\n  ");
            message.push_str(line);
        }
    }
    let steps = lines(&status.action);
    if steps.is_empty() {
        // What Lemonade's container-backend guide (docs/guide/configuration/
        // container-backends.md at v2026.47.1) asks of each engine. Its own
        // `lemonade` CLI is inside rocm-cli's runtime tree, not on the user's
        // PATH, so it is not a command to point at.
        message.push_str(
            "\n\nLemonade gave no setup steps. It needs Podman with crun and your account \
             in the `render` group, or Docker with your account in the `docker` group.",
        );
    } else {
        message.push_str("\n\nTo finish the setup:");
        for line in steps {
            message.push_str("\n  ");
            message.push_str(line);
        }
    }
    message.push_str("\n\nThen run the same `rocm serve` command again.");
    message
}

fn lines(text: &str) -> Vec<&str> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && *line != "-")
        .collect()
}

fn reported_suffix(message: &str) -> String {
    let message = lines(message).join("; ");
    if message.is_empty() {
        String::new()
    } else {
        format!(" (Lemonade: {message})")
    }
}

/// One line for `rocm engines` and `rocm examine`, from the status Lemonade last
/// reported (at install, or at the last Gufo serve) and the Gufo models this
/// Lemonade's registry lists.
pub(crate) fn detect_note(
    recorded: Option<&GufoBackendStatus>,
    models: &[String],
) -> Option<String> {
    if models.is_empty() && recorded.is_none() {
        return None;
    }
    let availability = match recorded {
        None => {
            "not checked yet; `rocm serve` checks it before downloading a Gufo model".to_owned()
        }
        Some(status) => match readiness(Some(status)) {
            Ok(GufoReadiness::Installed) => "installed".to_owned(),
            Ok(GufoReadiness::NeedsInstall) => {
                "available, image not downloaded yet (pulled on the first Gufo serve)".to_owned()
            }
            Err(_) if status.state == "action_required" => format!(
                "needs container setup before it can run{}",
                reported_suffix(&status.message)
            ),
            Err(_) if status.state == "unsupported" => format!(
                "unsupported on this machine (Strix Halo gfx1151 under Linux only){}",
                reported_suffix(&status.message)
            ),
            Err(_) => format!("`{}`", status.state),
        },
    };
    let checked = if recorded.is_some() {
        " (as Lemonade last reported it)"
    } else {
        ""
    };
    let models = if models.is_empty() {
        String::new()
    } else {
        format!("; Gufo models: {}", models.join(", "))
    };
    Some(format!(
        "{GUFO_DETECT_NOTE_PREFIX} ({GUFO_RECIPE}:{GUFO_BACKEND}, Strix Halo container): \
         {availability}{checked}{models}"
    ))
}

/// The file in the runtime directory that keeps the last Gufo status Lemonade
/// reported, for `rocm engines` and `rocm examine`, which start no `lemond`.
pub(crate) const RECORDED_STATUS_FILE: &str = "rocm-cli-gufo-status.json";

/// Keep `status` as the last one Lemonade reported. Best effort: a runtime tree
/// that cannot be written (a read-only ROCm install) leaves the last record,
/// and the next serve asks Lemonade again anyway.
pub(crate) fn record_status(runtime_dir: &Path, status: Option<&GufoBackendStatus>) {
    let path = runtime_dir.join(RECORDED_STATUS_FILE);
    let _ = match status {
        Some(status) => serde_json::to_vec_pretty(status)
            .map_err(std::io::Error::other)
            .and_then(|bytes| std::fs::write(&path, bytes)),
        None => std::fs::remove_file(&path),
    };
}

/// The last Gufo status [`record_status`] kept, if any.
pub(crate) fn recorded_status(runtime_dir: &Path) -> Option<GufoBackendStatus> {
    let bytes = std::fs::read(runtime_dir.join(RECORDED_STATUS_FILE)).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Ask the `lemond` an install already started about Gufo and record the
/// answer. Best effort: Gufo is not part of what an install sets up, so a
/// failed question records nothing and fails nothing.
pub(crate) fn record_status_from(runtime_dir: &Path, host: &str, port: u16) {
    if let Ok(status) = query_status(host, port) {
        record_status(runtime_dir, status.as_ref());
    }
}

/// Ask a running `lemond` what it reports for the Gufo backend.
pub(crate) fn query_status(host: &str, port: u16) -> Result<Option<GufoBackendStatus>> {
    let endpoint = rocm_core::format_http_base_url(host, port);
    let api_key = rocm_engine_protocol::resolve_endpoint_api_key();
    let body = rocm_core::http_get_text_with_auth(
        &endpoint,
        "/api/v1/system-info",
        api_key.as_deref(),
        SYSTEM_INFO_TIMEOUT,
    )
    .with_context(|| format!("failed to ask Lemonade at {endpoint} about its Gufo backend"))?;
    let system_info: Value =
        serde_json::from_str(&body).context("failed to parse Lemonade system-info JSON")?;
    Ok(status_from_system_info(&system_info))
}

/// The Gufo backend status, asked of a short-lived `lemond` started for the
/// purpose. Used before a serve launches anything, so a refusal reaches the
/// user's terminal instead of a background log.
pub(crate) fn probe_status(
    manifest: &LemonadeInstallManifest,
) -> Result<Option<GufoBackendStatus>> {
    #[cfg(feature = "e2e-test-hooks")]
    if let Some(path) = std::env::var_os(SYSTEM_INFO_TEST_ENV) {
        let status = scripted_status(Path::new(&path))?;
        record_status(&manifest.runtime_dir, status.as_ref());
        return Ok(status);
    }
    let port = crate::state::free_local_port()?;
    let log_path = manifest.runtime_dir.join("gufo-probe-lemond.log");
    let process_env = crate::process::lemonade_process_environment()?;
    let mut child = crate::process::spawn_lemond(
        manifest,
        crate::DEFAULT_HOST,
        port,
        Some(&log_path),
        &process_env,
    )?;
    let result = (|| {
        crate::process::wait_for_lemonade_cli_status(
            manifest,
            crate::DEFAULT_HOST,
            port,
            Duration::from_secs(30),
            Some(&log_path),
            &process_env,
        )?;
        query_status(crate::DEFAULT_HOST, port)
    })();
    let _ = crate::state::terminate_pid(child.id(), true);
    let _ = child.wait();
    if let Ok(status) = &result {
        record_status(&manifest.runtime_dir, status.as_ref());
    }
    result
}

#[cfg(feature = "e2e-test-hooks")]
fn scripted_status(path: &Path) -> Result<Option<GufoBackendStatus>> {
    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let system_info: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(status_from_system_info(&system_info))
}

/// Make sure the Gufo backend can serve and is installed on the running
/// `lemond` at `host:port`, installing its image when Lemonade says it can.
/// Refuses (see [`readiness`]) when it cannot run here.
pub(crate) fn ensure_backend(
    manifest: &LemonadeInstallManifest,
    host: &str,
    port: u16,
    process_env: &LemonadeProcessEnvironment,
    parent: &ParentRuntimeEnvironment,
) -> Result<String> {
    let status = query_status(host, port)?;
    record_status(&manifest.runtime_dir, status.as_ref());
    match readiness(status.as_ref())? {
        GufoReadiness::Installed => {
            eprintln!("Using installed Lemonade {GUFO_RECIPE}:{GUFO_BACKEND} backend.");
        }
        GufoReadiness::NeedsInstall => {
            eprintln!(
                "Installing Lemonade {GUFO_RECIPE}:{GUFO_BACKEND} backend (pulls its container image)..."
            );
            crate::process::run_lemonade_recipe_backend_install(
                manifest,
                host,
                port,
                GUFO_RECIPE,
                GUFO_BACKEND,
                process_env,
                parent,
            )
            .context(
                "Lemonade could not install the Gufo backend's container image, so the model \
                 was not loaded; check that Podman or Docker can pull images, then retry \
                 `rocm serve`",
            )?;
            // So `rocm engines` stops saying the image is still to be pulled:
            // the install succeeded, so it is installed, and Lemonade's own
            // answer replaces that when it gives one.
            record_status(
                &manifest.runtime_dir,
                Some(&GufoBackendStatus {
                    state: "installed".to_owned(),
                    message: String::new(),
                    action: String::new(),
                }),
            );
            record_status_from(&manifest.runtime_dir, host, port);
        }
    }
    Ok(GUFO_BACKEND.to_owned())
}

/// The `ResolveModel` answer for a Gufo model: refused when Gufo cannot run
/// here (asked of Lemonade, see [`probe_status`]), or when the request carries
/// llama-server flags Gufo would not apply; otherwise a description of the
/// Gufo serve the launch will run.
pub(crate) fn resolve_gufo_model(
    manifest: &LemonadeInstallManifest,
    canonical_model_id: String,
    device_policy: DevicePolicy,
    engine_recipe: Option<EngineRecipeHint>,
) -> Result<ResolveModelResponse> {
    refuse_llama_server_flags(
        &canonical_model_id,
        engine_recipe.as_ref(),
        REFUSED_BEFORE_ANY_SERVE,
    )?;
    let status = probe_status(manifest)?;
    let readiness = readiness(status.as_ref())?;
    let mut warnings = vec![
        "Lemonade serves this model with Gufo, its Strix Halo (gfx1151) engine, from a \
         container image; the GPU-required policy never runs it on the CPU"
            .to_owned(),
    ];
    if readiness == GufoReadiness::NeedsInstall {
        warnings.push(
            "the Gufo container image is not downloaded yet; the first serve pulls it \
             before loading the model"
                .to_owned(),
        );
    }
    Ok(ResolveModelResponse {
        canonical_model_id,
        served_model_id: None,
        task: "chat-completions".to_owned(),
        source: "lemonade".to_owned(),
        revision: "main".to_owned(),
        loader: GUFO_RECIPE.to_owned(),
        trust_remote_code: false,
        chat_template_mode: "lemonade".to_owned(),
        dtype: "gguf".to_owned(),
        device_policy,
        estimated_memory: "the model's GGUF size plus context, in Strix Halo shared memory"
            .to_owned(),
        launch_defaults: serde_json::json!({
            "host": crate::DEFAULT_HOST,
            "port": rocm_core::DEFAULT_LOCAL_PORT,
            "endpoint_mode": "openai"
        }),
        engine_recipe,
        warnings,
    })
}

/// The line a ready Gufo serve prints and logs, so the user can tell the model
/// is served by Gufo and not by Lemonade's llama.cpp backend.
pub(crate) fn serving_notice(model: &str) -> String {
    format!(
        "Lemonade is serving {model} with Gufo ({GUFO_RECIPE}:{GUFO_BACKEND}), its Strix Halo \
         engine, from a container on the GPU; no CPU fallback is used."
    )
}

/// `rocm serve`'s sampling flags and a catalog recipe's flags reach Lemonade as
/// llama-server arguments (`--llamacpp-args`). Gufo is not llama-server, so it
/// would not apply them; refuse rather than serve with them silently dropped.
/// `outcome` says what the refusal left undone where it was raised, which
/// differs: before any serve starts on the `ResolveModel` round trip, after
/// Lemonade's server is up at backend selection.
pub(crate) fn refuse_llama_server_flags(
    model: &str,
    engine_recipe: Option<&EngineRecipeHint>,
    outcome: &str,
) -> Result<()> {
    let Some(flags) = engine_recipe
        .map(|recipe| &recipe.required_flags)
        .filter(|flags| !flags.is_empty())
    else {
        return Ok(());
    };
    bail!(
        "`{model}` is a Gufo model, and Gufo does not take the llama-server arguments \
         this serve would pass it ({}); {outcome}. Serve it without \
         --temperature, --top-p and --max-tokens (or, starting the engine directly, \
         with no llama-server flags in its recipe), and set sampling per request \
         instead.",
        flags.join(" ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn system_info(state: &str, message: &str, action: &str) -> Value {
        json!({
            "recipes": {
                "llamacpp": {"backends": {"rocm": {"state": "installed"}}},
                "gufo": {"backends": {"rocm": {
                    "state": state,
                    "message": message,
                    "action": action,
                    "tier": "guest",
                    "format": "container"
                }}}
            }
        })
    }

    fn status(state: &str, message: &str, action: &str) -> GufoBackendStatus {
        status_from_system_info(&system_info(state, message, action)).unwrap()
    }

    /// Every state Lemonade defines, and what a serve does with it. A state
    /// missing from this table is a state nobody decided about.
    #[test]
    fn every_lemonade_state_maps_to_a_load_an_install_or_a_refusal() {
        let cases = [
            ("installed", Some(GufoReadiness::Installed)),
            ("update_available", Some(GufoReadiness::Installed)),
            ("installable", Some(GufoReadiness::NeedsInstall)),
            ("update_required", Some(GufoReadiness::NeedsInstall)),
            ("action_required", None),
            ("unsupported", None),
            ("something_new", None),
        ];
        for (state, expected) in cases {
            let result = readiness(Some(&status(state, "", "")));
            assert_eq!(
                result.as_ref().ok().copied(),
                expected,
                "{state}: {result:?}"
            );
        }
    }

    #[test]
    fn a_lemonade_without_the_gufo_recipe_is_refused_without_a_command_that_cannot_help() {
        let info = json!({"recipes": {"llamacpp": {"backends": {}}}});
        assert_eq!(status_from_system_info(&info), None);
        let error = readiness(None).unwrap_err().to_string();
        assert!(error.contains("reports no Gufo backend"), "{error}");
        assert!(error.contains("not run on any other backend"), "{error}");
        // A reinstall of the same pin gets the same answer, so none is advised.
        assert!(!error.contains("rocm engines install"), "{error}");
    }

    #[test]
    fn an_unsupported_gpu_is_refused_naming_strix_halo_and_no_cpu_run() {
        let error = readiness(Some(&status("unsupported", "Unsupported GPU", "")))
            .unwrap_err()
            .to_string();
        assert!(error.contains("Strix Halo GPU (gfx1151)"), "{error}");
        assert!(error.contains("(Lemonade: Unsupported GPU)"), "{error}");
        assert!(
            error.contains("The model and the Gufo image were not downloaded"),
            "{error}"
        );
        assert!(
            error.contains("not run on any other backend or on the CPU"),
            "{error}"
        );
    }

    #[test]
    fn incomplete_container_setup_is_refused_with_lemonades_own_steps() {
        let error = readiness(Some(&status(
            "action_required",
            "Podman uses runc\nAccount is not in render",
            "sudo apt install crun\nsudo usermod -aG render $USER",
        )))
        .unwrap_err()
        .to_string();
        assert!(error.contains("not run on the CPU instead"), "{error}");
        assert!(
            error.contains("Lemonade reports:\n  Podman uses runc\n  Account is not in render"),
            "{error}"
        );
        assert!(
            error.contains(
                "To finish the setup:\n  sudo apt install crun\n  sudo usermod -aG render $USER"
            ),
            "{error}"
        );
        assert!(error.ends_with("Then run the same `rocm serve` command again."));
    }

    #[test]
    fn incomplete_setup_without_steps_names_what_lemonade_needs() {
        let error = readiness(Some(&status("action_required", "", "-")))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(
                "It needs Podman with crun and your account in the `render` group, or Docker \
                 with your account in the `docker` group."
            ),
            "{error}"
        );
        // Lemonade's own CLI is not on the user's PATH, so it is not advised.
        assert!(!error.contains("`lemonade "), "{error}");
        assert!(!error.contains("Lemonade reports:"), "{error}");
    }

    #[test]
    fn an_absent_state_reads_as_unsupported_as_lemonades_cli_prints_it() {
        let info = json!({"recipes": {"gufo": {"backends": {"rocm": {}}}}});
        assert_eq!(status_from_system_info(&info).unwrap().state, "unsupported");
    }

    #[test]
    fn llama_server_flags_on_a_gufo_serve_are_refused_not_dropped() {
        assert!(
            refuse_llama_server_flags("Qwen3.8-27B-Gufo", None, REFUSED_BEFORE_ANY_SERVE).is_ok()
        );
        let empty = EngineRecipeHint::default();
        assert!(
            refuse_llama_server_flags("Qwen3.8-27B-Gufo", Some(&empty), REFUSED_BEFORE_ANY_SERVE)
                .is_ok()
        );
        // Each of `rocm serve`'s sampling flags, as the llama-server argument it
        // reaches Lemonade as (README: `--max-tokens` is llama.cpp's
        // `--n-predict`), is refused on its own.
        for flag in [
            ["--temperature", "0.5"],
            ["--top-p", "0.9"],
            ["--n-predict", "128"],
        ] {
            let sampling = EngineRecipeHint {
                required_flags: flag.iter().map(|part| (*part).to_owned()).collect(),
                ..EngineRecipeHint::default()
            };
            let error = refuse_llama_server_flags(
                "Qwen3.8-27B-Gufo",
                Some(&sampling),
                REFUSED_BEFORE_ANY_SERVE,
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains(&format!("({})", flag.join(" "))), "{error}");
            assert!(error.contains("nothing was started"), "{error}");
        }
    }

    #[test]
    fn a_recorded_status_round_trips_and_none_clears_it() {
        let runtime = tempfile::tempdir().unwrap();
        assert_eq!(recorded_status(runtime.path()), None);
        let reported = status(
            "action_required",
            "Podman uses runc",
            "sudo apt install crun",
        );
        record_status(runtime.path(), Some(&reported));
        assert_eq!(recorded_status(runtime.path()), Some(reported));
        record_status(runtime.path(), None);
        assert_eq!(recorded_status(runtime.path()), None);
    }

    #[test]
    fn the_detect_note_names_the_state_when_it_was_checked_and_the_models() {
        let models = vec!["Qwen3.8-27B-Gufo".to_owned()];
        let cases = [
            (None, "not checked yet"),
            (
                Some(status("installed", "", "")),
                ": installed (as Lemonade last reported it)",
            ),
            (
                Some(status("installable", "", "")),
                "image not downloaded yet",
            ),
            (
                Some(status("action_required", "Podman uses runc", "")),
                "needs container setup before it can run (Lemonade: Podman uses runc)",
            ),
            (
                Some(status("unsupported", "Unsupported GPU", "")),
                "unsupported on this machine (Strix Halo gfx1151 under Linux only)",
            ),
        ];
        for (recorded, expected) in cases {
            let note = detect_note(recorded.as_ref(), &models).unwrap();
            assert!(note.starts_with(GUFO_DETECT_NOTE_PREFIX), "{note}");
            assert!(note.contains(expected), "{note}");
            assert!(note.ends_with("; Gufo models: Qwen3.8-27B-Gufo"), "{note}");
        }
        assert_eq!(detect_note(None, &[]), None);
    }
}
