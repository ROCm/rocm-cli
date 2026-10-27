// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, format_http_base_url, http_get_text_with_auth, runtime_home_dir};
use serde_json::{Value, json};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command as ProcessCommand, ExitStatus, Stdio};
use std::time::Duration;

use crate::install::{LemonadeInstallManifest, LemonadeRuntime};
use crate::llamacpp_backend::{backend_server_dirs, find_llama_server_binary_for_backend};
use crate::process::{
    LemonadeProcessEnvironment, apply_environment_vars, hide_child_console_window,
    lemonade_process_environment_vars, query_chat_smoke_endpoint, run_lemonade_pull_with_parent,
    spawn_lemond_with_parent, wait_for_lemonade_cli_status_with_parent,
};
use crate::runtime_dir::ParentRuntimeEnvironment;
use crate::state::{
    free_local_port, mark_json_status, merge_json_state, resolve_lemonade_model_ref, terminate_pid,
    write_running_state,
};
use crate::{DEFAULT_HOST, ROCM_BACKEND_NAME, ServeHttpRequest};

/// Direct serving is only available for a local GGUF or an exact checkpoint.
/// Unsupported registered recipes stay on the Lemonade router path.
#[derive(Debug)]
pub(crate) enum DirectLlamaModel {
    Local(PathBuf),
    Checkpoint(HfCheckpoint),
}

#[derive(Debug)]
pub(crate) enum ServeRoute {
    Direct {
        model: DirectLlamaModel,
        server: PathBuf,
    },
    Router {
        model_ref: String,
    },
}

/// Choose the serving path before downloading or launching a server.
pub(crate) fn serving_route(
    manifest: &LemonadeInstallManifest,
    request: &ServeHttpRequest,
    is_linux: bool,
) -> Result<ServeRoute> {
    crate::device_policy::require_gpu_required(&request.device_policy)?;
    let explicit_checkpoint = parse_hf_checkpoint(&request.model_ref).is_some();
    // Windows serves a registered name through the router on purpose: direct
    // serving passes `-m`, `--n-gpu-layers`, `--alias` and the recipe flags, and does
    // not apply the router's per-model launch defaults, so only an explicit
    // checkpoint is served directly there.
    if (is_linux || explicit_checkpoint)
        && let Some(model) = resolve_direct_llama_model(manifest, &request.model_ref)?
        && let Some(server) = direct_serve_binary(manifest)
    {
        return Ok(ServeRoute::Direct { model, server });
    }
    if explicit_checkpoint {
        if manifest.backend_recipe != crate::LLAMACPP_RECIPE {
            bail!(
                "selected Lemonade recipe `{}` does not support direct HF checkpoint serving; a GPU llama.cpp recipe is required",
                manifest.backend_recipe
            );
        }
        if backend_server_dirs(&manifest.backend_name).is_none() {
            bail!(
                "selected Lemonade backend `{}` does not support GPU direct serving; no CPU fallback is used",
                manifest.backend_name
            );
        }
        // The backend is a supported GPU one, so only its binary is absent. A
        // reinstall re-extracts the runtime and reinstalls the best backend, which
        // restores the file whatever state Lemonade recorded for it.
        bail!(
            "no selected GPU llama-server backend `{}` is installed under {} for direct serving of `{}`; run `rocm engines install lemonade --reinstall` to restore it",
            manifest.backend_name,
            manifest.runtime_dir.display(),
            request.model_ref
        );
    }
    Ok(ServeRoute::Router {
        model_ref: resolve_lemonade_model_ref(&request.model_ref),
    })
}

/// The direct route still open after the router failed to load or verify a
/// model: Linux only, a model the registry resolves to a safe checkpoint, and the
/// llama-server of the backend Lemonade selected (read from the manifest the
/// router serve updated). A CPU or unknown backend never yields a server, so this
/// cannot become a CPU fallback.
pub(crate) fn direct_fallback_route(
    manifest: &LemonadeInstallManifest,
    request: &ServeHttpRequest,
    is_linux: bool,
) -> Option<(DirectLlamaModel, PathBuf)> {
    if !is_linux {
        return None;
    }
    let model = resolve_direct_llama_model(manifest, &request.model_ref)
        .ok()
        .flatten()?;
    let server = direct_serve_binary(manifest)?;
    Some((model, server))
}

/// Serve directly after the router failed to load or verify the model, or return
/// the router's error unchanged when no direct route is open. The line it prints
/// and logs names the backend taken from the server path it launches.
#[allow(clippy::too_many_arguments)]
pub(crate) fn serve_direct_after_router_failure(
    paths: &AppPaths,
    request: &ServeHttpRequest,
    runtime: &LemonadeRuntime,
    process_env: &LemonadeProcessEnvironment,
    log_path: Option<&Path>,
    parent: &ParentRuntimeEnvironment,
    is_linux: bool,
    router_error: anyhow::Error,
) -> Result<()> {
    let Some((model, server)) = direct_fallback_route(&runtime.manifest, request, is_linux) else {
        // No direct route is open, so the router's failure is final. The router
        // left its state for this function to settle.
        let _ = crate::state::mark_json_status(&request.state_path, "failed");
        return Err(router_error);
    };
    let backend = llama_server_backend_label(&server);
    let reason = format!(
        "Lemonade router could not serve {} ({router_error:#}); serving it directly with the selected {backend} GPU llama-server",
        request.model_ref
    );
    eprintln!("{reason}");
    // The router left the state at `running`, with a server that is already
    // stopped. A reader that sees it while the download or the spawn is still
    // running would report a server that is gone, and one that sees `failed`
    // would record the service as failed for good (`refresh_from_engine_state`
    // ignores later states once a record is `failed`), so the state says
    // `starting` for the whole fallback and goes to `failed` if the fallback
    // does not produce a running server.
    crate::state::merge_json_state(
        &request.state_path,
        &serde_json::json!({ "engine": crate::ENGINE_NAME, "status": "starting" }),
    )
    .with_context(|| {
        format!(
            "failed to record that the direct serving fallback is starting (the router had failed: {router_error:#})"
        )
    })?;
    let outcome = ensure_model_in(
        paths,
        request,
        runtime,
        process_env,
        log_path,
        &model,
        parent,
    )
    .with_context(|| format!("{router_error:#}; the direct serving fallback also failed"))
    .and_then(|model_path| {
        serve_direct_llama_server(
            request,
            runtime,
            process_env,
            &server,
            &model_path,
            log_path,
            &reason,
            parent,
        )
    });
    if outcome.is_err() {
        let _ = crate::state::mark_json_status(&request.state_path, "failed");
    }
    outcome
}

pub(crate) fn resolve_direct_llama_model(
    manifest: &LemonadeInstallManifest,
    model_ref: &str,
) -> Result<Option<DirectLlamaModel>> {
    let path = PathBuf::from(model_ref);
    if path.is_file()
        && path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    {
        return Ok(Some(DirectLlamaModel::Local(path)));
    }
    Ok(resolve_direct_llama_checkpoint(manifest, model_ref)?.map(DirectLlamaModel::Checkpoint))
}

/// Resolve the cached file without choosing an arbitrary quantization. The same
/// resolution is used before downloading and immediately before server launch.
pub(crate) fn direct_llama_model_path(
    paths: &AppPaths,
    model: &DirectLlamaModel,
) -> Result<Option<PathBuf>> {
    match model {
        DirectLlamaModel::Local(path) => Ok(path.is_file().then(|| path.clone())),
        DirectLlamaModel::Checkpoint(checkpoint) => {
            let matches = hf_checkpoint_gguf_matches(paths, checkpoint);
            // A split model is one model: serve it from its first shard, which
            // llama-server uses to load the remaining shards beside it.
            if let Some(first) = first_shard_of_complete_set(&matches) {
                return Ok(Some(first));
            }
            // Some but not all shards of one model (an interrupted download): treat
            // the model as absent so it is downloaded again, never served partial.
            if is_incomplete_shard_set(&matches) {
                return Ok(None);
            }
            if matches.len() > 1 {
                let names = matches
                    .iter()
                    .filter_map(|path| path.file_name())
                    .map(|name| name.to_string_lossy())
                    .collect::<Vec<_>>()
                    .join(", ");
                let example = matches
                    .first()
                    .and_then(|path| path.file_name())
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default();
                bail!(
                    "variant `{}` matches multiple files ({names}); specify an exact quantization or filename, e.g. `{}/{}:{example}`",
                    checkpoint.variant.as_deref().unwrap_or_default(),
                    checkpoint.owner,
                    checkpoint.repo
                );
            }
            Ok(matches.into_iter().next())
        }
    }
}

/// Return the GGUF for `model`, downloading it first when absent. The cache
/// location and parent runtime environment are supplied by the caller. The hub
/// lookup also reads HF cache environment variables and the user's home directory;
/// tests use unique checkpoint names to avoid matching unrelated cached models.
pub(crate) fn ensure_model_in(
    paths: &AppPaths,
    request: &ServeHttpRequest,
    runtime: &LemonadeRuntime,
    process_env: &LemonadeProcessEnvironment,
    log_path: Option<&Path>,
    model: &DirectLlamaModel,
    parent: &ParentRuntimeEnvironment,
) -> Result<PathBuf> {
    ensure_model_with_download(paths, &request.model_ref, model, |checkpoint_ref| {
        let download_port = free_local_port()?;
        let mut child = spawn_lemond_with_parent(
            &runtime.manifest,
            DEFAULT_HOST,
            download_port,
            log_path,
            process_env,
            parent,
        )?;
        let result = (|| -> Result<()> {
            wait_for_lemonade_cli_status_with_parent(
                &runtime.manifest,
                DEFAULT_HOST,
                download_port,
                Duration::from_secs(30),
                log_path,
                process_env,
                parent,
            )?;
            // Pull downloads without a model load, so this path neither guesses a
            // backend nor masks load errors.
            run_lemonade_pull_with_parent(
                &runtime.manifest,
                DEFAULT_HOST,
                download_port,
                checkpoint_ref,
                log_path,
                process_env,
                parent,
            )
        })();
        let _ = terminate_pid(child.id(), true);
        let _ = child.wait();
        result
    })
}

/// Return the cached GGUF, downloading it first when absent. `download` receives
/// the validated checkpoint reference (`owner/repo:variant`), never the alias the
/// user typed, and its error is returned as is.
fn ensure_model_with_download(
    paths: &AppPaths,
    requested: &str,
    model: &DirectLlamaModel,
    download: impl FnOnce(&str) -> Result<()>,
) -> Result<PathBuf> {
    if let Some(path) = direct_llama_model_path(paths, model)? {
        return Ok(path);
    }
    let DirectLlamaModel::Checkpoint(checkpoint) = model else {
        bail!("local GGUF `{requested}` is missing");
    };
    download(&checkpoint.model_ref())?;
    direct_llama_model_path(paths, model)?.with_context(|| {
        format!(
            "Lemonade did not download `{requested}` for direct serving (searched {})",
            describe_hf_cache_roots(paths)
        )
    })
}

/// Run one of Lemonade's packaged `llama-server` binaries (`server`) directly on a
/// GGUF file, exposing the model under `request.model_ref` via `--alias`. This is how
/// a canonical Hugging Face name is served under exactly that name — Lemonade's router
/// renames registered models, but an llama-server alias is a free-form string.
#[allow(clippy::too_many_arguments)]
pub(crate) fn serve_direct_llama_server(
    request: &ServeHttpRequest,
    runtime: &LemonadeRuntime,
    process_env: &LemonadeProcessEnvironment,
    server: &Path,
    model_path: &Path,
    log_path: Option<&Path>,
    reason: &str,
    parent: &ParentRuntimeEnvironment,
) -> Result<()> {
    serve_direct_llama_server_within(
        request,
        runtime,
        process_env,
        server,
        model_path,
        log_path,
        reason,
        parent,
        DIRECT_SERVER_READY_TIMEOUT,
    )
}

/// How long a direct llama-server may take to answer `/v1/models` and a chat smoke
/// test before the serve is treated as failed.
const DIRECT_SERVER_READY_TIMEOUT: Duration = Duration::from_mins(2);

#[allow(clippy::too_many_arguments)]
fn serve_direct_llama_server_within(
    request: &ServeHttpRequest,
    runtime: &LemonadeRuntime,
    process_env: &LemonadeProcessEnvironment,
    server: &Path,
    model_path: &Path,
    log_path: Option<&Path>,
    reason: &str,
    parent: &ParentRuntimeEnvironment,
    ready_timeout: Duration,
) -> Result<()> {
    if !model_path.is_file() {
        bail!(
            "GGUF for `{}` is missing at {}",
            request.model_ref,
            model_path.display()
        );
    }
    if !server.is_file() {
        bail!("Lemonade llama-server is missing at {}", server.display());
    }
    let backend = llama_server_backend_label(server);

    if let Some(log_path) = log_path
        && let Some(parent) = log_path.parent()
    {
        fs::create_dir_all(parent)?;
    }
    if let Some(log_path) = log_path {
        let mut log = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .with_context(|| format!("failed to open {}", log_path.display()))?;
        writeln!(
            log,
            "\nLaunching Lemonade packaged {backend} llama-server directly: {reason}"
        )
        .ok();
    }

    let mut direct_env = process_env.clone();
    if let Some(server_dir) = server.parent() {
        push_existing_path(&mut direct_env.path_entries, server_dir.to_path_buf());
        push_existing_path(&mut direct_env.library_entries, server_dir.to_path_buf());
    }

    let mut command = direct_llama_server_command(request, server, model_path);
    // Packaged llama-server is raw llama.cpp — it does not read `LEMONADE_API_KEY`.
    // When `rocm serve` protects a public endpoint, hand llama-server the *existing*
    // CLI-managed 0600 key file via `--api-key-file` (a path, not the value, so the
    // secret never lands in the process table). Reusing the managed file rather than
    // writing a copy keeps the key's lifecycle owned by `rocm serve` — created before
    // spawn, deleted on stop — with no stale plaintext copy left behind on teardown.
    if let Some(key_file) = rocm_engine_protocol::resolve_endpoint_api_key_file() {
        command.arg("--api-key-file").arg(&key_file);
    }
    if let Some(log_path) = log_path {
        let log = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .with_context(|| format!("failed to open {}", log_path.display()))?;
        command.stdout(Stdio::from(log.try_clone()?));
        command.stderr(Stdio::from(log));
    } else {
        command.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    }
    let vars = lemonade_process_environment_vars(&direct_env, parent)?;
    apply_environment_vars(&mut command, &vars);
    hide_child_console_window(&mut command);
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to start {}", server.display()))?;
    let mut steps = RealDirectServerSteps {
        request,
        runtime,
        backend: &backend,
        model_path,
        log_path,
        ready_timeout,
    };
    supervise_direct_server(&mut child, &request.state_path, &mut steps)
}

/// The fallible steps that follow spawning a direct llama-server, so the failure
/// handling in [`supervise_direct_server`] is exercised without a live server.
trait DirectServerSteps {
    fn record_running(&mut self, server_pid: u32) -> Result<()>;
    fn wait_ready(&mut self) -> Result<()>;
    fn record_ready(&mut self, server_pid: u32) -> Result<()>;
    /// Announce what is now true. Called only once the server answered, passed its
    /// chat smoke test and the `ready` state is written: an earlier announcement
    /// would claim a serve that a failed readiness check then contradicts.
    fn announce_serving(&mut self);
    fn wait_exit(&mut self, child: &mut Child) -> std::io::Result<ExitStatus> {
        child.wait()
    }
}

struct RealDirectServerSteps<'a> {
    request: &'a ServeHttpRequest,
    runtime: &'a LemonadeRuntime,
    backend: &'a str,
    model_path: &'a Path,
    log_path: Option<&'a Path>,
    ready_timeout: Duration,
}

impl DirectServerSteps for RealDirectServerSteps<'_> {
    fn record_running(&mut self, server_pid: u32) -> Result<()> {
        write_running_state(
            self.request,
            self.runtime,
            std::process::id(),
            Some(server_pid),
            "running",
        )
    }

    fn wait_ready(&mut self) -> Result<()> {
        let request = self.request;
        wait_for_openai_models_ready(
            &request.host,
            request.port,
            &request.model_ref,
            self.ready_timeout,
        )?;
        if !query_chat_smoke_endpoint(&request.host, request.port, &request.model_ref)? {
            bail!("Lemonade packaged llama-server did not pass a chat-completion smoke test");
        }
        Ok(())
    }

    fn record_ready(&mut self, server_pid: u32) -> Result<()> {
        let request = self.request;
        merge_json_state(
            &request.state_path,
            &json!({
                "status": "ready",
                "server_pid": server_pid,
                // Identity token for the server PID, captured while the child is alive.
                "server_start_ticks": rocm_core::process_start_ticks(server_pid),
                "backend_state": "ready",
                "backend_requested": self.backend,
                "backend_mode": "lemonade-packaged-llama-server",
                "load_response": {
                    "status": "loaded",
                    "method": "lemonade-packaged-llama-server",
                    "model_name": request.model_ref,
                    "model_path": self.model_path,
                    "llamacpp_backend": self.backend
                },
            }),
        )
    }

    fn announce_serving(&mut self) {
        crate::device_policy::announce_non_rocm_backend(self.backend, self.log_path);
    }
}

/// Own a just-spawned direct llama-server. Every step after the spawn, state
/// writes and the final wait included, runs under one guard: any failure stops and
/// reaps the child (`Child` does not kill on drop) and marks the state `failed`,
/// so the server can never keep running with no state that stop and health can
/// reach.
fn supervise_direct_server<S: DirectServerSteps>(
    child: &mut Child,
    state_path: &Path,
    steps: &mut S,
) -> Result<()> {
    let server_pid = child.id();
    let stop_and_fail = |child: &mut Child| {
        let _ = terminate_pid(server_pid, true);
        let _ = child.wait();
        let _ = mark_json_status(state_path, "failed");
    };
    let started = (|| -> Result<()> {
        steps.record_running(server_pid)?;
        steps.wait_ready()?;
        steps.record_ready(server_pid)?;
        steps.announce_serving();
        Ok(())
    })();
    if let Err(error) = started {
        stop_and_fail(child);
        return Err(error);
    }
    let status = match steps.wait_exit(child) {
        Ok(status) => status,
        Err(error) => {
            stop_and_fail(child);
            return Err(anyhow::Error::new(error)
                .context("failed waiting for Lemonade packaged llama-server"));
        }
    };
    mark_json_status(
        state_path,
        if status.success() {
            "stopped"
        } else {
            "failed"
        },
    )?;
    if status.success() {
        Ok(())
    } else {
        bail!("Lemonade packaged llama-server exited with status {status}")
    }
}

pub(crate) fn direct_llama_server_command(
    request: &ServeHttpRequest,
    server: &Path,
    model_path: &Path,
) -> ProcessCommand {
    let mut command = ProcessCommand::new(server);
    command
        .arg("-m")
        .arg(model_path)
        .arg("--host")
        .arg(&request.host)
        .arg("--port")
        .arg(request.port.to_string())
        .arg("--n-gpu-layers")
        .arg("999")
        .arg("--alias")
        .arg(&request.model_ref)
        .stdin(Stdio::null());
    if let Some(recipe) = request.engine_recipe.as_ref() {
        command.args(&recipe.required_flags);
    }
    command
}

fn wait_for_openai_models_ready(
    host: &str,
    port: u16,
    model_ref: &str,
    timeout: Duration,
) -> Result<()> {
    let endpoint = format_http_base_url(host, port);
    let endpoint_api_key = rocm_engine_protocol::resolve_endpoint_api_key();
    let start = std::time::Instant::now();
    let mut last_error = None;
    while start.elapsed() < timeout {
        match http_get_text_with_auth(
            &endpoint,
            "/v1/models",
            endpoint_api_key.as_deref(),
            Duration::from_secs(3),
        )
        .and_then(|body| parse_models_ready(&body, model_ref))
        {
            Ok(true) => return Ok(()),
            Ok(false) => last_error = Some("model was not reported by /v1/models".to_owned()),
            Err(error) => last_error = Some(error.to_string()),
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    bail!(
        "Lemonade packaged llama-server did not become ready: {}",
        last_error.unwrap_or_else(|| "not checked".to_owned())
    )
}

fn parse_models_ready(body: &str, model_ref: &str) -> Result<bool> {
    let value = serde_json::from_str::<Value>(body.trim())
        .context("failed to parse /v1/models response")?;
    Ok(models_payload_has_loaded_model(&value, model_ref))
}

fn models_payload_has_loaded_model(value: &Value, model_ref: &str) -> bool {
    value
        .get("data")
        .or_else(|| value.get("models"))
        .and_then(Value::as_array)
        .is_some_and(|models| {
            models.iter().any(|model| {
                ["id", "model", "name"]
                    .into_iter()
                    .filter_map(|field| model.get(field).and_then(Value::as_str))
                    .any(|loaded| loaded == model_ref)
            })
        })
}

fn push_existing_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !path.exists() || paths.iter().any(|existing| existing == &path) {
        return;
    }
    paths.push(path);
}

/// A canonical Hugging Face checkpoint reference: `owner/repo` with an optional
/// `:variant` suffix (a quantization label such as `BF16`/`Q4_0`, or an exact
/// `file.gguf`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HfCheckpoint {
    owner: String,
    repo: String,
    variant: Option<String>,
}

impl HfCheckpoint {
    fn model_ref(&self) -> String {
        format!(
            "{}/{}:{}",
            self.owner,
            self.repo,
            self.variant.as_deref().unwrap_or_default()
        )
    }
}

/// Parse a model reference as a canonical Hugging Face checkpoint. Returns `None` for
/// anything not of the form `owner/repo[:variant]` — built-in Lemonade aliases and
/// bare registered names all fall through unchanged.
///
/// Every component is validated against the Hugging Face id / quantization charset
/// (`[A-Za-z0-9._-]`, and never `.`/`..`). This is a security boundary as well as a
/// correctness one: the components are used to build a filesystem path into the model
/// cache, so rejecting path separators and traversal tokens prevents a crafted model
/// reference from escaping the cache directory.
pub(crate) fn parse_hf_checkpoint(model_ref: &str) -> Option<HfCheckpoint> {
    let trimmed = model_ref.trim();
    if trimmed.is_empty() {
        return None;
    }
    // Split off an optional `:variant` suffix on the first colon; the owner/repo pair
    // never contains a colon, so any remainder is the variant.
    let (repo_path, variant) = match trimmed.split_once(':') {
        Some((path, variant)) => (path, Some(variant)),
        None => (trimmed, None),
    };
    let (owner, repo) = repo_path.split_once('/')?;
    if !is_safe_hf_component(owner) || !is_safe_hf_component(repo) {
        return None;
    }
    if variant.is_some_and(|variant| !is_safe_hf_component(variant)) {
        return None;
    }
    Some(HfCheckpoint {
        owner: owner.to_owned(),
        repo: repo.to_owned(),
        variant: variant.map(str::to_owned),
    })
}

/// A safe Hugging Face path component: a non-empty run of `[A-Za-z0-9._-]` that is not
/// a `.`/`..` traversal token. Matches the Hugging Face id and quantization-label
/// charset and, by excluding `/`, `\`, and traversal tokens, keeps a user-supplied
/// component from escaping the model cache directory when used to build a path.
fn is_safe_hf_component(component: &str) -> bool {
    !component.is_empty()
        && component != "."
        && component != ".."
        && component
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// Directory names under `bin/llamacpp/` that [`llama_server_backend_label`]
/// recognizes when it names the backend of a packaged `llama-server`. GPU backends
/// only; `cpu` is intentionally absent so a CPU build is never labelled as, or
/// mistaken for, a direct-serve backend. This is only a label lookup: which
/// directories are searched for a selected backend is decided by
/// [`crate::llamacpp_backend::backend_server_dirs`].
const DIRECT_LLAMA_SERVER_BACKENDS: [&str; 4] = ["rocm-stable", "rocm-nightly", "rocm", "vulkan"];

/// The backend label for a packaged llama-server, taken from the nearest ancestor
/// directory matching a known backend name (e.g. `vulkan`, `rocm-stable`) so it still
/// resolves correctly when Lemonade nests the binary under a build-numbered
/// subdirectory; falls back to the ROCm backend name.
pub(crate) fn llama_server_backend_label(server: &Path) -> String {
    server
        .ancestors()
        .filter_map(|dir| dir.file_name())
        .filter_map(|name| name.to_str())
        .find(|name| DIRECT_LLAMA_SERVER_BACKENDS.contains(name))
        .unwrap_or(ROCM_BACKEND_NAME)
        .to_owned()
}

/// Registered model names are exact keys after known shorthand normalization.
/// Metadata is untrusted: it must name a safe HF checkpoint with an explicit
/// variant, never a local path or a model guessed from the registered name.
pub(crate) fn resolve_direct_llama_checkpoint(
    manifest: &LemonadeInstallManifest,
    model_ref: &str,
) -> Result<Option<HfCheckpoint>> {
    if let Some(checkpoint) = parse_hf_checkpoint(model_ref) {
        return require_checkpoint_variant(checkpoint, model_ref).map(Some);
    }
    let registry_path = manifest.runtime_dir.join("resources/server_models.json");
    let Ok(bytes) = fs::read(&registry_path) else {
        return Ok(None);
    };
    let Ok(registry) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return Ok(None);
    };
    let Some(entries) = registry.as_object() else {
        return Ok(None);
    };
    let name = resolve_lemonade_model_ref(model_ref);
    let Some(metadata) = entries.get(&name) else {
        return Ok(None);
    };
    if metadata.get("recipe").and_then(serde_json::Value::as_str) != Some(crate::LLAMACPP_RECIPE) {
        return Ok(None);
    }
    let Some(checkpoint_ref) = metadata
        .get("checkpoint")
        .and_then(serde_json::Value::as_str)
    else {
        return Ok(None);
    };
    // A registered checkpoint that is not a plain `owner/repo:variant` is not
    // this module's to serve: Lemonade's catalog writes `*` for "every file", for
    // one. Nothing here builds a path from it, so it goes to the router, which
    // served it before. Only a reference the user typed is an error (above).
    let Some(checkpoint) = parse_hf_checkpoint(checkpoint_ref) else {
        return Ok(None);
    };
    let Some(fields) = metadata.as_object() else {
        return Ok(None);
    };
    // Only the single-checkpoint shape is supported directly. Extra files,
    // multimodal inputs, options and unknown fields belong to Lemonade's router.
    // In the 11.5.1 source, size/suggested describe catalog entries and these
    // labels classify chat models. Other labels can change launch flags (MTP).
    if checkpoint.variant.is_none()
        || fields.iter().any(|(key, value)| match key.as_str() {
            "recipe" | "checkpoint" => false,
            "size" => !value.as_f64().is_some_and(|size| size >= 0.0),
            "suggested" => !value.is_boolean(),
            "labels" => !value.as_array().is_some_and(|labels| {
                labels.iter().all(|label| {
                    matches!(label.as_str(), Some("reasoning" | "tool-calling" | "tools"))
                })
            }),
            _ => true,
        })
    {
        return Ok(None);
    }
    Ok(Some(checkpoint))
}

fn require_checkpoint_variant(checkpoint: HfCheckpoint, model_ref: &str) -> Result<HfCheckpoint> {
    if checkpoint.variant.is_none() {
        bail!(
            "`{model_ref}` needs an explicit quantization variant to serve; use `owner/repo:variant`, e.g. `{}/{}:Q4_K_M` (see the repo's GGUF files on Hugging Face for the available variants)",
            checkpoint.owner,
            checkpoint.repo
        );
    }
    Ok(checkpoint)
}

pub(crate) fn direct_serve_binary(manifest: &LemonadeInstallManifest) -> Option<PathBuf> {
    if manifest.backend_recipe != crate::LLAMACPP_RECIPE {
        return None;
    }
    find_llama_server_binary_for_backend(manifest, &manifest.backend_name)
}

fn hf_cache_roots(paths: &AppPaths) -> Vec<PathBuf> {
    hf_cache_roots_from(
        paths,
        |name| std::env::var_os(name).map(PathBuf::from),
        runtime_home_dir,
    )
}

/// `home_dir` is only consulted when `HOME` is unset, which is always true on Windows
/// (Lemonade still caches under `%USERPROFILE%\.cache\huggingface\hub`); it's injected
/// so the fallback can be exercised without depending on the host's actual platform.
fn hf_cache_roots_from<F, H>(paths: &AppPaths, mut env_path: F, home_dir: H) -> Vec<PathBuf>
where
    F: FnMut(&str) -> Option<PathBuf>,
    H: FnOnce() -> Option<PathBuf>,
{
    let mut roots = Vec::new();
    if let Some(hub_cache) = env_path("HUGGINGFACE_HUB_CACHE") {
        push_hf_cache_root(&mut roots, hub_cache);
    }
    if let Some(hf_home) = env_path("HF_HOME") {
        push_hf_cache_root(&mut roots, hf_home.join("hub"));
    }
    push_hf_cache_root(&mut roots, paths.cache_dir.join("huggingface").join("hub"));
    if let Some(home) = env_path("HOME").or_else(home_dir) {
        push_hf_cache_root(
            &mut roots,
            home.join(".cache").join("huggingface").join("hub"),
        );
    }
    roots
}

/// Render the cache directories `hf_cache_roots` searches, for use in an error message
/// when none of them held the expected file.
fn describe_hf_cache_roots(paths: &AppPaths) -> String {
    render_hf_cache_roots(&hf_cache_roots(paths))
}

/// The rendering half of [`describe_hf_cache_roots`], split out so the error text can be
/// asserted against the same root list `hf_cache_roots_from` produces, without depending
/// on the host's real environment.
fn render_hf_cache_roots(roots: &[PathBuf]) -> String {
    if roots.is_empty() {
        return "no Hugging Face cache directories could be determined".to_owned();
    }
    roots
        .iter()
        .map(|root| root.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn push_hf_cache_root(roots: &mut Vec<PathBuf>, path: PathBuf) {
    if path.as_os_str().is_empty() || roots.iter().any(|existing| existing == &path) {
        return;
    }
    roots.push(path);
}

/// All cached GGUF files for a checkpoint's repo under one hub cache root (unfiltered
/// by variant).
fn collect_hf_checkpoint_ggufs(cache_root: &Path, checkpoint: &HfCheckpoint) -> Vec<PathBuf> {
    let repo_dir = format!("models--{}--{}", checkpoint.owner, checkpoint.repo);
    let snapshots = cache_root.join(repo_dir).join("snapshots");
    let mut ggufs = Vec::new();
    if let Ok(entries) = fs::read_dir(snapshots) {
        for entry in entries.flatten() {
            collect_gguf_files(&entry.path(), &mut ggufs, 0);
        }
    }
    ggufs
}

/// Every cached GGUF matching a checkpoint's variant, across all hub cache roots,
/// deduplicated by file name. Used to detect an ambiguous variant (more than one match)
/// so the caller can refuse to pick arbitrarily.
fn hf_checkpoint_gguf_matches(paths: &AppPaths, checkpoint: &HfCheckpoint) -> Vec<PathBuf> {
    hf_checkpoint_gguf_matches_in_roots(&hf_cache_roots(paths), checkpoint)
}

fn hf_checkpoint_gguf_matches_in_roots(
    roots: &[PathBuf],
    checkpoint: &HfCheckpoint,
) -> Vec<PathBuf> {
    // Keep root priority and path order, but prefer a complete split model over
    // partial copies of the same files. Deduplicating first could combine shards
    // from two revisions, hiding a usable complete download behind an older one.
    let mut candidates = roots
        .iter()
        .flat_map(|root| {
            filter_ggufs_by_variant(
                collect_hf_checkpoint_ggufs(root, checkpoint),
                checkpoint.variant.as_deref(),
            )
        })
        .collect::<Vec<_>>();
    let mut groups = std::collections::HashMap::new();
    for path in &candidates {
        if let Some((stem, _, total)) = parse_gguf_shard(path) {
            groups
                .entry((path.parent().map(Path::to_path_buf), stem, total))
                .or_insert_with(Vec::new)
                .push(path.clone());
        }
    }
    let complete = groups
        .into_iter()
        .filter_map(|(key, paths)| first_shard_of_complete_set(&paths).map(|_| key))
        .collect::<std::collections::HashSet<_>>();
    candidates.sort_by_key(|path| {
        parse_gguf_shard(path).is_some_and(|(stem, _, total)| {
            !complete.contains(&(path.parent().map(Path::to_path_buf), stem, total))
        })
    });
    let mut seen_names = std::collections::HashSet::new();
    candidates.retain(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| seen_names.insert(name.to_owned()))
    });
    candidates
}

/// Split a split-GGUF file name (`<stem>-00001-of-00002.gguf`) into its stem,
/// 1-based shard number and shard count.
fn parse_gguf_shard(path: &Path) -> Option<(String, u32, u32)> {
    let name = path.file_name()?.to_str()?;
    if !name.to_ascii_lowercase().ends_with(".gguf") {
        return None;
    }
    // The suffix is ASCII, so this slice is on a character boundary.
    let stem = &name[..name.len() - ".gguf".len()];
    let (rest, total) = stem.rsplit_once("-of-")?;
    let (stem, index) = rest.rsplit_once('-')?;
    let digits = |text: &str| text.len() == 5 && text.chars().all(|c| c.is_ascii_digit());
    if stem.is_empty() || !digits(index) || !digits(total) {
        return None;
    }
    Some((stem.to_owned(), index.parse().ok()?, total.parse().ok()?))
}

/// The first shard when `matches` is exactly one complete split model (every shard
/// of one stem present once), `None` otherwise. Several unrelated files, or an
/// incomplete set, are still ambiguous or missing and are handled by the caller.
fn first_shard_of_complete_set(matches: &[PathBuf]) -> Option<PathBuf> {
    let shards = matches
        .iter()
        .map(|path| parse_gguf_shard(path).map(|shard| (path, shard)))
        .collect::<Option<Vec<_>>>()?;
    let (first_path, (stem, _, total)) = shards.first()?;
    // llama-server finds the remaining shards beside the first one, so shards
    // spread over several cache roots or snapshot revisions are not a usable set.
    let directory = first_path.parent();
    let complete = shards.len() == usize::try_from(*total).ok()?
        && shards.iter().all(|(path, (other, _, other_total))| {
            other == stem && other_total == total && path.parent() == directory
        })
        && (1..=*total).all(|wanted| shards.iter().any(|(_, (_, index, _))| *index == wanted));
    if !complete {
        return None;
    }
    shards
        .iter()
        .find(|(_, (_, index, _))| *index == 1)
        .map(|(path, _)| (*path).clone())
}

/// Whether `matches` are only shards of a single split model, with some missing.
fn is_incomplete_shard_set(matches: &[PathBuf]) -> bool {
    let Some(shards) = matches
        .iter()
        .map(|path| parse_gguf_shard(path))
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let Some((stem, _, total)) = shards.first() else {
        return false;
    };
    shards
        .iter()
        .all(|(other, _, other_total)| other == stem && other_total == total)
}

/// Collect `*.gguf` files under a snapshot revision directory. GGUFs usually sit at the
/// snapshot root, but sharded variants live one folder down, so recurse a few levels.
fn collect_gguf_files(dir: &Path, out: &mut Vec<PathBuf>, depth: usize) {
    if depth > 3 {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_gguf_files(&path, out, depth + 1);
        } else if path.is_file()
            && path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            out.push(path);
        }
    }
}

/// All GGUFs matching a requested variant, sorted by path. With no variant, every file
/// matches. With a variant, a case-insensitive exact filename match (`:model.gguf`)
/// wins outright; otherwise every file whose name contains the quantization label
/// (`:BF16`, `:Q4_0`) matches — more than one indicates an ambiguous variant.
fn filter_ggufs_by_variant(mut ggufs: Vec<PathBuf>, variant: Option<&str>) -> Vec<PathBuf> {
    ggufs.sort();
    ggufs.dedup();
    let Some(variant) = variant else {
        return ggufs;
    };
    let file_name = |path: &Path| -> Option<String> {
        path.file_name()
            .and_then(|name| name.to_str())
            .map(str::to_owned)
    };
    let exact_matches = ggufs
        .iter()
        .filter(|path| file_name(path).is_some_and(|name| name.eq_ignore_ascii_case(variant)))
        .collect::<Vec<_>>();
    if !exact_matches.is_empty() {
        // A shard names its whole split model: serving needs every shard, and the
        // ambiguity error suggests a shard's exact filename. The same shard name
        // can exist in several snapshot revisions; prefer one with the full set.
        let shard_set = |exact: &PathBuf| -> Option<Vec<PathBuf>> {
            let (stem, _, total) = parse_gguf_shard(exact)?;
            Some(
                ggufs
                    .iter()
                    .filter(|path| {
                        path.parent() == exact.parent()
                            && parse_gguf_shard(path).is_some_and(|(other, _, other_total)| {
                                other == stem && other_total == total
                            })
                    })
                    .cloned()
                    .collect(),
            )
        };
        let sets = exact_matches
            .iter()
            .filter_map(|exact| shard_set(exact))
            .collect::<Vec<_>>();
        if let Some(set) = sets
            .iter()
            .find(|set| first_shard_of_complete_set(set).is_some())
            .or_else(|| sets.first())
        {
            return set.clone();
        }
        return vec![exact_matches[0].clone()];
    }
    let variant_lower = variant.to_ascii_lowercase();
    // An explicit filename is not a quantization substring. If absent, download
    // that exact file rather than serve another file with the same suffix.
    if Path::new(variant)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
    {
        return Vec::new();
    }
    ggufs
        .into_iter()
        .filter(|path| {
            file_name(path).is_some_and(|name| name.to_ascii_lowercase().contains(&variant_lower))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::install::platform_binary_name;
    use crate::process::models_payload_has_ready_model;
    use crate::test_support::scratch_dir;
    use crate::{DEFAULT_MODEL, LLAMACPP_RECIPE};
    use rocm_engine_protocol::EngineRecipeHint;
    use std::fs;

    fn write_registry(manifest: &LemonadeInstallManifest, registry: serde_json::Value) {
        let resources = manifest.runtime_dir.join("resources");
        fs::create_dir_all(&resources).unwrap();
        fs::write(
            resources.join("server_models.json"),
            serde_json::to_vec(&registry).unwrap(),
        )
        .unwrap();
    }

    fn serve_request(model_ref: &str) -> ServeHttpRequest {
        ServeHttpRequest {
            service_id: "test".to_owned(),
            model_ref: model_ref.to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 11435,
            device_policy: rocm_engine_protocol::DevicePolicy::GpuRequired,
            gpu_indices: vec![1],
            runtime_id: None,
            env_id: None,
            state_path: PathBuf::from("state.json"),
            log_path: None,
            engine_recipe: None,
        }
    }

    #[test]
    fn serving_route_covers_platform_recipe_binary_and_policy_boundaries() {
        let dir = scratch_dir("serving-routes");
        let mut manifest = test_manifest(dir.join("runtime"));
        manifest.backend_name = "vulkan".to_owned();
        write_registry(
            &manifest,
            serde_json::json!({
                "Qwen3-0.6B-GGUF": {"recipe": "llamacpp", "checkpoint": "owner/repo:Q4_0", "suggested": true, "size": 0.38, "labels": ["reasoning"]},
                "Complex": {"recipe": "llamacpp", "checkpoint": "owner/repo:Q4_0", "mmproj": "owner/repo:mmproj.gguf"},
                "OtherRecipe": {"recipe": "oga", "checkpoint": "owner/repo"}
            }),
        );
        let binary = manifest
            .runtime_dir
            .join("bin/llamacpp/vulkan")
            .join(platform_binary_name("llama-server"));
        fs::create_dir_all(binary.parent().unwrap()).unwrap();
        fs::write(&binary, b"x").unwrap();
        for is_linux in [true, false] {
            let requested = serve_request("qwen-smoke");
            let route = serving_route(&manifest, &requested, is_linux).unwrap();
            if is_linux {
                let ServeRoute::Direct { server, .. } = route else {
                    panic!("a registered alias with a selected-backend binary serves directly");
                };
                assert_eq!(server, binary);
                // The `--alias` the server is started with is asserted through the
                // production serve path in
                // `direct_serve_through_serve_http_uses_the_requested_alias` (lib.rs).
            } else {
                assert!(
                    matches!(route, ServeRoute::Router { model_ref } if model_ref == "Qwen3-0.6B-GGUF")
                );
            }
            assert!(matches!(
                serving_route(&manifest, &serve_request("owner/repo:Q4_0"), is_linux).unwrap(),
                ServeRoute::Direct { .. }
            ));
            assert!(serving_route(&manifest, &serve_request("owner/repo"), is_linux).is_err());
            for name in ["Complex", "Unknown", "OtherRecipe"] {
                assert!(
                    matches!(serving_route(&manifest, &serve_request(name), is_linux).unwrap(), ServeRoute::Router { model_ref } if model_ref == name)
                );
            }
            let mut cpu_request = serve_request("qwen-smoke");
            cpu_request.device_policy = rocm_engine_protocol::DevicePolicy::CpuOnly;
            assert!(
                serving_route(&manifest, &cpu_request, is_linux)
                    .unwrap_err()
                    .to_string()
                    .contains("no CPU fallback")
            );
        }
        fs::remove_file(&binary).unwrap();
        for is_linux in [true, false] {
            assert!(matches!(
                serving_route(&manifest, &serve_request("qwen-smoke"), is_linux).unwrap(),
                ServeRoute::Router { .. }
            ));
            // The selected backend is supported but its binary is gone: the error is
            // returned (no route, so nothing is started) and names the command that
            // restores it. `rocm engines install lemonade --reinstall` is a real
            // invocation: the `engines install` subcommand takes `--reinstall`, which
            // re-extracts the runtime and reinstalls the best backend.
            let error = serving_route(&manifest, &serve_request("owner/repo:Q4_0"), is_linux)
                .unwrap_err()
                .to_string();
            assert!(error.contains("no selected GPU llama-server"), "{error}");
            // This pins the wording only. That the command exists and takes `--reinstall`
            // is exercised by `@id:runtime-lemonade-backend-alignment-reported` and
            // `@id:runtime-lemonade-backend-alignment-opt-out` (runtime_setup.feature),
            // which run `rocm engines install lemonade --reinstall` and require it to
            // succeed.
            // That first scenario (runtime-16) deletes the selected backend's
            // llama-server, reinstalls, and requires the file to be back, using the
            // same layout as `direct_serve_binary`.
            // That a reinstall re-extracts the runtime over an existing one is
            // `a_version_change_forces_re_extraction_over_the_existing_runtime`
            // (install.rs).
            assert!(
                error.contains("rocm engines install lemonade --reinstall"),
                "{error}"
            );
        }
        // Recipe/backend refusal is independent of a missing packaged binary.
        fs::write(&binary, b"x").unwrap();
        let cpu_binary = manifest
            .runtime_dir
            .join("bin/llamacpp/cpu")
            .join(platform_binary_name("llama-server"));
        fs::create_dir_all(cpu_binary.parent().unwrap()).unwrap();
        fs::write(cpu_binary, b"x").unwrap();
        for (recipe, backend) in [
            ("oga", "vulkan"),
            (LLAMACPP_RECIPE, "cpu"),
            (LLAMACPP_RECIPE, "unknown"),
        ] {
            manifest.backend_recipe = recipe.to_owned();
            manifest.backend_name = backend.to_owned();
            for is_linux in [true, false] {
                assert!(matches!(
                    serving_route(&manifest, &serve_request("qwen-smoke"), is_linux).unwrap(),
                    ServeRoute::Router { .. }
                ));
                let error = serving_route(&manifest, &serve_request("owner/repo:Q4_0"), is_linux)
                    .unwrap_err()
                    .to_string();
                let expected = if recipe == LLAMACPP_RECIPE {
                    "does not support GPU direct serving"
                } else {
                    "does not support direct HF checkpoint serving"
                };
                assert!(error.contains(expected), "{error}");
                // A reinstall cannot make a CPU or non-llama.cpp backend servable, and
                // their binary may well be present, so neither claims it is missing.
                assert!(!error.contains("is installed under"), "{error}");
                assert!(!error.contains("--reinstall"), "{error}");
            }
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn serving_variant_errors_carry_concrete_examples() {
        let dir = scratch_dir("variant-examples");
        let manifest = test_manifest(dir.join("runtime"));
        let error = resolve_direct_llama_model(&manifest, "owner/repo")
            .unwrap_err()
            .to_string();
        assert!(error.contains("`owner/repo` needs an explicit"), "{error}");
        assert!(error.contains("e.g. `owner/repo:Q4_K_M`"), "{error}");
        // The same failure through the serve entry point returns the error and
        // no route, so nothing is downloaded or started.
        let error = serving_route(&manifest, &serve_request("owner/repo"), true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("e.g. `owner/repo:Q4_K_M`"), "{error}");

        let paths = AppPaths {
            config_dir: dir.join("config"),
            data_dir: dir.join("data"),
            cache_dir: dir.join("cache"),
        };
        let snapshot = paths
            .cache_dir
            .join("huggingface/hub/models--owner--repo/snapshots/revision");
        fs::create_dir_all(&snapshot).unwrap();
        for file in ["model-Q4_0.gguf", "model-Q4_K_M.gguf"] {
            fs::write(snapshot.join(file), b"x").unwrap();
        }
        let ambiguous = parse_hf_checkpoint("owner/repo:Q4").unwrap();
        let error = direct_llama_model_path(&paths, &DirectLlamaModel::Checkpoint(ambiguous))
            .unwrap_err()
            .to_string();
        assert!(error.contains("matches multiple files"), "{error}");
        // The example is a real, unambiguous reference: it resolves to one file.
        assert!(
            error.contains("e.g. `owner/repo:model-Q4_0.gguf`"),
            "{error}"
        );
        let example = parse_hf_checkpoint("owner/repo:model-Q4_0.gguf").unwrap();
        assert!(
            direct_llama_model_path(&paths, &DirectLlamaModel::Checkpoint(example))
                .unwrap()
                .is_some()
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn serving_regression_complex_registered_models_use_router() {
        let dir = scratch_dir("registered-complex");
        let manifest = test_manifest(dir.join("runtime"));
        for (key, value) in [
            ("mmproj", serde_json::json!("owner/repo:mmproj.gguf")),
            (
                "checkpoints",
                serde_json::json!(["owner/repo:Q4_0", "owner/repo:mmproj.gguf"]),
            ),
            ("extra_files", serde_json::json!(["tokenizer.json"])),
            ("options", serde_json::json!({"ctx_size": 8192})),
            ("source", serde_json::json!("alternate")),
            ("unknown_option", serde_json::json!(true)),
            ("recipe_options", serde_json::json!({"ctx_size": 8192})),
            ("labels", serde_json::json!(["vision"])),
            ("labels", serde_json::json!(["mtp"])),
            ("labels", serde_json::json!(["unknown"])),
            ("labels", serde_json::json!([123])),
            ("labels", serde_json::json!("reasoning")),
            ("size", serde_json::json!("0.38")),
            ("suggested", serde_json::json!(1)),
        ] {
            let mut metadata =
                serde_json::json!({"recipe": "llamacpp", "checkpoint": "owner/repo:Q4_0"});
            metadata[key] = value;
            write_registry(&manifest, serde_json::json!({"Complex-GGUF": metadata}));
            assert!(
                resolve_direct_llama_model(&manifest, "Complex-GGUF")
                    .unwrap()
                    .is_none(),
                "{key}"
            );
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn serving_regression_safe_variantless_registered_model_uses_router() {
        let dir = scratch_dir("registered-variantless");
        let manifest = test_manifest(dir.join("runtime"));
        write_registry(
            &manifest,
            serde_json::json!({
                "Variantless": {"recipe": "llamacpp", "checkpoint": "owner/repo"}
            }),
        );
        assert!(
            resolve_direct_llama_model(&manifest, "Variantless")
                .unwrap()
                .is_none()
        );
        assert!(resolve_direct_llama_model(&manifest, "owner/repo").is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn serving_regression_unusable_catalog_uses_router() {
        let dir = scratch_dir("registered-unusable");
        let manifest = test_manifest(dir.join("runtime"));
        let catalog = manifest.runtime_dir.join("resources/server_models.json");
        fs::create_dir_all(catalog.parent().unwrap()).unwrap();
        for bytes in [b"[]".as_slice(), b"null".as_slice(), b"invalid".as_slice()] {
            fs::write(&catalog, bytes).unwrap();
            assert!(
                resolve_direct_llama_model(&manifest, "qwen-smoke")
                    .unwrap()
                    .is_none()
            );
        }
        fs::remove_file(&catalog).unwrap();
        fs::create_dir(&catalog).unwrap();
        assert!(
            resolve_direct_llama_model(&manifest, "qwen-smoke")
                .unwrap()
                .is_none()
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn serving_regression_registered_alias_resolves_exact_checkpoint() {
        let dir = scratch_dir("registered-alias");
        let manifest = test_manifest(dir.join("runtime"));
        write_registry(
            &manifest,
            serde_json::json!({
                "Qwen3-0.6B-GGUF": {"recipe": "llamacpp", "checkpoint": "unsloth/Qwen3-0.6B-GGUF:Q4_0"},
                "Other-GGUF": {"recipe": "llamacpp", "checkpoint": "owner/Other-GGUF:Q8_0"}
            }),
        );
        for alias in ["qwen-smoke", "tiny", " QWEN3-0.6B-GGUF "] {
            assert_eq!(
                resolve_direct_llama_checkpoint(&manifest, alias).unwrap(),
                parse_hf_checkpoint("unsloth/Qwen3-0.6B-GGUF:Q4_0")
            );
        }
        assert_eq!(
            resolve_direct_llama_checkpoint(&manifest, "Other-GGUF").unwrap(),
            parse_hf_checkpoint("owner/Other-GGUF:Q8_0")
        );
        assert_eq!(
            resolve_direct_llama_checkpoint(&manifest, "Other").unwrap(),
            None
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn serving_regression_routes_unsafe_registered_metadata_to_the_router() {
        let dir = scratch_dir("registered-unsafe");
        let manifest = test_manifest(dir.join("runtime"));
        for checkpoint in [
            "../../etc/passwd:Q4_0",
            "owner/repo:../secret",
            "/tmp/model.gguf",
        ] {
            write_registry(
                &manifest,
                serde_json::json!({
                    "Unsafe-GGUF": {"recipe": "llamacpp", "checkpoint": checkpoint}
                }),
            );
            // Never served directly, and never turned into a path: the router
            // gets the name, as it did before direct serving existed.
            assert_eq!(
                resolve_direct_llama_checkpoint(&manifest, "Unsafe-GGUF").unwrap(),
                None,
                "{checkpoint}"
            );
        }
        fs::remove_dir_all(dir).unwrap();
    }

    /// Lemonade's own catalog writes `*` for "every file". It is not a safe
    /// filename, so direct serving cannot take it, but it is not malformed either:
    /// the serve must go to the router, and the fallback must agree.
    #[test]
    fn a_registered_wildcard_variant_routes_to_the_router_not_an_error() {
        let dir = scratch_dir("registered-wildcard");
        let mut manifest = test_manifest(dir.join("runtime"));
        manifest.backend_name = "vulkan".to_owned();
        write_registry(
            &manifest,
            serde_json::json!({
                "Wildcard-GGUF": {
                    "checkpoint": "owner/repo:*",
                    "recipe": "llamacpp",
                    "suggested": true,
                    "labels": ["reasoning"],
                    "size": 1.0
                }
            }),
        );
        let binary = manifest
            .runtime_dir
            .join("bin/llamacpp/vulkan")
            .join(platform_binary_name("llama-server"));
        fs::create_dir_all(binary.parent().unwrap()).unwrap();
        fs::write(&binary, b"x").unwrap();
        for is_linux in [true, false] {
            assert!(
                matches!(
                    serving_route(&manifest, &serve_request("Wildcard-GGUF"), is_linux).unwrap(),
                    ServeRoute::Router { ref model_ref } if model_ref == "Wildcard-GGUF"
                ),
                "is_linux {is_linux}"
            );
            assert!(
                direct_fallback_route(&manifest, &serve_request("Wildcard-GGUF"), is_linux)
                    .is_none(),
                "is_linux {is_linux}"
            );
        }
        fs::remove_dir_all(dir).unwrap();
    }

    /// `size` is part of the single-checkpoint shape: a negative one is not a
    /// catalog entry this module understands, and zero is.
    #[test]
    fn a_registered_entry_with_a_negative_size_stays_with_the_router() {
        let dir = scratch_dir("registered-size");
        let manifest = test_manifest(dir.join("runtime"));
        for (size, direct) in [(-1.0, false), (0.0, true), (0.38, true)] {
            write_registry(
                &manifest,
                serde_json::json!({
                    "Sized-GGUF": {
                        "checkpoint": "owner/repo:Q4_0",
                        "recipe": "llamacpp",
                        "size": size
                    }
                }),
            );
            assert_eq!(
                resolve_direct_llama_checkpoint(&manifest, "Sized-GGUF")
                    .unwrap()
                    .is_some(),
                direct,
                "size {size}"
            );
        }
        fs::remove_dir_all(dir).unwrap();
    }

    /// Two different split models with the same shard count are not one set,
    /// whatever their indexes: serving the first would leave llama-server looking
    /// for shards of a model that is not there.
    #[test]
    fn shards_of_two_different_models_are_not_a_complete_set() {
        let dir = Path::new("/snapshots/rev");
        let paths = |names: [&str; 2]| names.map(|name| dir.join(name)).to_vec();
        assert_eq!(
            first_shard_of_complete_set(&paths(["a-00001-of-00002.gguf", "b-00002-of-00002.gguf"])),
            None
        );
        assert_eq!(
            first_shard_of_complete_set(&paths(["a-00002-of-00002.gguf", "a-00001-of-00002.gguf"])),
            Some(dir.join("a-00001-of-00002.gguf"))
        );
    }

    /// A variant that names a `.gguf` file is an exact filename, not a quantization
    /// label: when no file has that name, nothing matches, even though another
    /// file's name ends with the same text.
    #[test]
    fn an_exact_gguf_variant_that_is_absent_does_not_match_a_longer_file_name() {
        let ggufs = vec![PathBuf::from("/snapshots/rev/model-Q4_0.gguf")];
        assert_eq!(
            filter_ggufs_by_variant(ggufs.clone(), Some("Q4_0.gguf")),
            Vec::<PathBuf>::new()
        );
        // The same text as a label still matches by substring.
        assert_eq!(filter_ggufs_by_variant(ggufs.clone(), Some("Q4_0")), ggufs);
    }

    #[test]
    fn serving_regression_selected_backend_ignores_stale_rocm() {
        let dir = scratch_dir("selected-backend");
        let mut manifest = test_manifest(dir.join("runtime"));
        manifest.backend_name = "vulkan".to_owned();
        let base = manifest.runtime_dir.join("bin/llamacpp");
        for backend in ["rocm-stable", "vulkan", "cpu"] {
            fs::create_dir_all(base.join(backend)).unwrap();
            fs::write(
                base.join(backend)
                    .join(platform_binary_name("llama-server")),
                b"x",
            )
            .unwrap();
        }
        assert_eq!(
            direct_serve_binary(&manifest),
            Some(
                base.join("vulkan")
                    .join(platform_binary_name("llama-server"))
            )
        );
        fs::remove_dir_all(base.join("vulkan")).unwrap();
        assert_eq!(direct_serve_binary(&manifest), None);
        manifest.backend_name = "cpu".to_owned();
        assert_eq!(direct_serve_binary(&manifest), None);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn serving_registered_recipe_boundaries_and_missing_registry() {
        let dir = scratch_dir("registry-boundaries");
        let manifest = test_manifest(dir.join("runtime"));
        assert!(
            resolve_direct_llama_model(&manifest, "qwen-smoke")
                .unwrap()
                .is_none()
        );
        write_registry(
            &manifest,
            serde_json::json!({
                "RecipeModel": {"recipe": "oga", "checkpoint": "not/a/gguf"},
                "NoCheckpoint": {"recipe": "llamacpp"},
                "MalformedCheckpoint": {"recipe": "llamacpp", "checkpoint": 123},
                "Exact-GGUF": {"recipe": "llamacpp", "checkpoint": "owner/repo:Q8_0"}
            }),
        );
        for name in [
            "RecipeModel",
            "Unknown-GGUF",
            "exact-gguf",
            "Exact",
            "../../escape",
            "NoCheckpoint",
            "MalformedCheckpoint",
        ] {
            assert!(
                resolve_direct_llama_model(&manifest, name)
                    .unwrap()
                    .is_none(),
                "{name}"
            );
        }
        assert!(resolve_direct_llama_model(&manifest, "owner/repo").is_err());
        for json in [b"[]".as_slice(), b"invalid".as_slice()] {
            fs::write(
                manifest.runtime_dir.join("resources/server_models.json"),
                json,
            )
            .unwrap();
            assert!(
                resolve_direct_llama_model(&manifest, "Exact-GGUF")
                    .unwrap()
                    .is_none()
            );
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn serving_registered_checkpoint_cache_rejects_ambiguous_variants() {
        let dir = scratch_dir("registry-cache");
        let manifest = test_manifest(dir.join("runtime"));
        let roots = vec![dir.join("hub-one"), dir.join("hub-two")];
        for root in &roots {
            let snapshot = root.join("models--owner--repo/snapshots/revision");
            fs::create_dir_all(&snapshot).unwrap();
            for file in ["model-Q4_0.gguf", "model-Q4_K_M.gguf"] {
                fs::write(snapshot.join(file), b"x").unwrap();
            }
            // A GGUF suffix on a directory is not a downloaded model file. This
            // fixture is handled by the `is_dir()` branch of `collect_gguf_files`; the
            // `is_file()` filter is covered by
            // `collect_gguf_files_skips_dangling_symlinks`.
            fs::create_dir_all(snapshot.join("directory-Q4.gguf")).unwrap();
        }
        write_registry(
            &manifest,
            serde_json::json!({
                "Exact-GGUF": {"recipe": "llamacpp", "checkpoint": "owner/repo:model-Q4_0.gguf"},
                "Ambiguous-GGUF": {"recipe": "llamacpp", "checkpoint": "owner/repo:Q4"}
            }),
        );
        let exact = resolve_direct_llama_checkpoint(&manifest, "Exact-GGUF")
            .unwrap()
            .unwrap();
        let matches = hf_checkpoint_gguf_matches_in_roots(&roots, &exact);
        assert_eq!(
            matches.len(),
            1,
            "same filename across cache roots is one quantization"
        );
        assert_eq!(matches[0].file_name().unwrap(), "model-Q4_0.gguf");
        let ambiguous = resolve_direct_llama_checkpoint(&manifest, "Ambiguous-GGUF")
            .unwrap()
            .unwrap();
        assert_eq!(
            hf_checkpoint_gguf_matches_in_roots(&roots, &ambiguous).len(),
            2
        );
        // Path resolution must reject ambiguous matching, not pick sorted first.
        let paths = AppPaths {
            config_dir: dir.join("config"),
            data_dir: dir.join("data"),
            cache_dir: dir.join("cache"),
        };
        let snapshot = paths
            .cache_dir
            .join("huggingface/hub/models--owner--repo/snapshots/revision");
        fs::create_dir_all(&snapshot).unwrap();
        for file in ["model-Q4_0.gguf", "model-Q4_K_M.gguf"] {
            fs::write(snapshot.join(file), b"x").unwrap();
        }
        assert!(direct_llama_model_path(&paths, &DirectLlamaModel::Checkpoint(ambiguous)).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn serving_local_gguf_is_explicit_not_registry_path_metadata() {
        let dir = scratch_dir("local-gguf");
        let manifest = test_manifest(dir.join("runtime"));
        let model = dir.join("model.GGUF");
        fs::write(&model, b"x").unwrap();
        let local = resolve_direct_llama_model(&manifest, model.to_str().unwrap())
            .unwrap()
            .unwrap();
        assert!(matches!(local, DirectLlamaModel::Local(ref path) if path == &model));
        write_registry(
            &manifest,
            serde_json::json!({
                "Path-GGUF": {"recipe": "llamacpp", "checkpoint": model}
            }),
        );
        // A registry entry never names a local file to serve: it is left to the
        // router, not read as a path.
        assert!(
            resolve_direct_llama_model(&manifest, "Path-GGUF")
                .unwrap()
                .is_none()
        );
        let non_gguf = dir.join("model.txt");
        fs::write(&non_gguf, b"x").unwrap();
        assert!(
            resolve_direct_llama_model(&manifest, non_gguf.to_str().unwrap())
                .unwrap()
                .is_none()
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn parses_canonical_hugging_face_checkpoints() {
        assert_eq!(
            parse_hf_checkpoint("LiquidAI/LFM2.5-230M-GGUF:Q4_0"),
            Some(HfCheckpoint {
                owner: "LiquidAI".to_owned(),
                repo: "LFM2.5-230M-GGUF".to_owned(),
                variant: Some("Q4_0".to_owned()),
            })
        );
        assert_eq!(
            parse_hf_checkpoint("unsloth/Qwen3-0.6B-GGUF"),
            Some(HfCheckpoint {
                owner: "unsloth".to_owned(),
                repo: "Qwen3-0.6B-GGUF".to_owned(),
                variant: None,
            })
        );
    }

    #[test]
    fn rejects_non_checkpoint_model_refs() {
        // Built-in aliases and bare registered names have no owner/repo pair.
        assert!(parse_hf_checkpoint("qwen").is_none());
        assert!(parse_hf_checkpoint(DEFAULT_MODEL).is_none());
        assert!(parse_hf_checkpoint("").is_none());
        assert!(parse_hf_checkpoint("/model.gguf").is_none());
        assert!(parse_hf_checkpoint("owner/").is_none());
        assert!(parse_hf_checkpoint("owner/repo/extra").is_none());
        assert!(parse_hf_checkpoint("owner/repo:").is_none());
    }

    #[test]
    fn checkpoint_matches_prefer_the_highest_priority_cache_root() {
        let dir = scratch_dir("root-priority");
        // The higher-priority root sorts after the lower one alphabetically, so a
        // merged-then-sorted list would pick the wrong file.
        let high = dir.join("zz-high");
        let low = dir.join("aa-low");
        let roots = vec![high.clone(), low.clone()];
        let checkpoint = parse_hf_checkpoint("owner/repo:model-Q4_0.gguf").unwrap();
        let snapshot_file =
            |root: &Path| root.join("models--owner--repo/snapshots/revision/model-Q4_0.gguf");
        for root in [&high, &low] {
            let file = snapshot_file(root);
            fs::create_dir_all(file.parent().unwrap()).unwrap();
            fs::write(&file, b"x").unwrap();
        }
        assert_eq!(
            hf_checkpoint_gguf_matches_in_roots(&roots, &checkpoint),
            vec![snapshot_file(&high)],
            "the highest-priority root wins, not the alphabetically first path"
        );

        // A file present only in the lower-priority root is still found.
        fs::remove_file(snapshot_file(&high)).unwrap();
        assert_eq!(
            hf_checkpoint_gguf_matches_in_roots(&roots, &checkpoint),
            vec![snapshot_file(&low)]
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_path_traversal_in_checkpoint_components() {
        // A crafted reference must not be able to escape the model cache directory.
        assert!(parse_hf_checkpoint("../../etc/passwd").is_none());
        assert!(parse_hf_checkpoint("..:Q4_0").is_none());
        assert!(parse_hf_checkpoint("owner/..").is_none());
        assert!(parse_hf_checkpoint("owner/repo:../secret").is_none());
        assert!(parse_hf_checkpoint(r"owner\..\..--x/repo").is_none());
        assert!(parse_hf_checkpoint("owner/repo:a/b.gguf").is_none());
    }

    #[test]
    fn serving_regression_exact_filename_never_substitutes_suffix_match() {
        assert!(
            filter_ggufs_by_variant(
                vec![PathBuf::from("/cache/other-model.gguf")],
                Some("model.gguf")
            )
            .is_empty()
        );
    }

    #[test]
    fn selects_gguf_by_variant() {
        // Real filenames use the quant label without the `-GGUF` repo suffix, so a
        // substring match on the `:variant` selects the right file.
        let ggufs = vec![
            PathBuf::from("/cache/LFM2.5-230M-Q4_0.gguf"),
            PathBuf::from("/cache/LFM2.5-230M-Q8_0.gguf"),
        ];
        assert_eq!(
            filter_ggufs_by_variant(ggufs.clone(), Some("q4_0"))
                .into_iter()
                .next(),
            Some(PathBuf::from("/cache/LFM2.5-230M-Q4_0.gguf"))
        );
        assert_eq!(
            filter_ggufs_by_variant(ggufs.clone(), Some("LFM2.5-230M-Q8_0.gguf"))
                .into_iter()
                .next(),
            Some(PathBuf::from("/cache/LFM2.5-230M-Q8_0.gguf"))
        );
        assert_eq!(
            filter_ggufs_by_variant(ggufs.clone(), None)
                .into_iter()
                .next(),
            Some(PathBuf::from("/cache/LFM2.5-230M-Q4_0.gguf"))
        );
        assert_eq!(
            filter_ggufs_by_variant(ggufs, Some("Q2_K"))
                .into_iter()
                .next(),
            None
        );
    }

    #[test]
    fn ambiguous_variant_matches_multiple_ggufs() {
        // A partial label like `Q4` matches several quants; the caller must refuse to
        // pick. A full label or exact filename resolves to exactly one.
        let ggufs = vec![
            PathBuf::from("/cache/LFM2.5-230M-Q4_0.gguf"),
            PathBuf::from("/cache/LFM2.5-230M-Q4_K_M.gguf"),
        ];
        assert_eq!(filter_ggufs_by_variant(ggufs.clone(), Some("Q4")).len(), 2);
        assert_eq!(
            filter_ggufs_by_variant(ggufs.clone(), Some("Q4_0")).len(),
            1
        );
        assert_eq!(
            filter_ggufs_by_variant(ggufs, Some("LFM2.5-230M-Q4_K_M.gguf")).len(),
            1
        );
    }

    #[test]
    fn direct_serve_binary_uses_selected_gpu_backend_and_skips_cpu() {
        let dir = scratch_dir("find-server");
        let runtime_dir = dir.join("runtime");
        let llamacpp = runtime_dir.join("bin").join("llamacpp");
        for backend in ["cpu", "vulkan"] {
            let backend_dir = llamacpp.join(backend);
            fs::create_dir_all(&backend_dir).unwrap();
            fs::write(backend_dir.join(platform_binary_name("llama-server")), b"x").unwrap();
        }
        let mut manifest = test_manifest(runtime_dir);
        manifest.backend_name = "vulkan".to_owned();
        // Both cpu and vulkan exist; the GPU backend is chosen and cpu is never selected.
        assert_eq!(
            direct_serve_binary(&manifest),
            Some(
                llamacpp
                    .join("vulkan")
                    .join(platform_binary_name("llama-server"))
            )
        );
        // With only cpu present, nothing is returned (no silent CPU fallback).
        fs::remove_dir_all(llamacpp.join("vulkan")).unwrap();
        assert_eq!(direct_serve_binary(&manifest), None);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn direct_serve_binary_finds_build_numbered_nesting() {
        // Some Lemonade llama.cpp releases extract one level deeper than
        // `<backend>/llama-server`, e.g. `rocm-stable/llama-b9752/llama-server`.
        let dir = scratch_dir("find-server-nested");
        let runtime_dir = dir.join("runtime");
        let llamacpp = runtime_dir.join("bin").join("llamacpp");
        let nested = llamacpp.join("rocm-stable").join("llama-b9752");
        fs::create_dir_all(&nested).unwrap();
        let server = nested.join(platform_binary_name("llama-server"));
        fs::write(&server, b"x").unwrap();
        let manifest = test_manifest(runtime_dir);
        assert_eq!(direct_serve_binary(&manifest), Some(server.clone()));
        assert_eq!(llama_server_backend_label(&server), "rocm-stable");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn direct_serve_binary_for_rocm_follows_family_priority() {
        let dir = scratch_dir("find-server-priority");
        let runtime_dir = dir.join("runtime");
        let llamacpp = runtime_dir.join("bin").join("llamacpp");
        let server = platform_binary_name("llama-server");
        let install = |backend: &str| {
            let backend_dir = llamacpp.join(backend);
            fs::create_dir_all(&backend_dir).unwrap();
            fs::write(backend_dir.join(&server), b"x").unwrap();
        };
        let manifest = test_manifest(runtime_dir);

        // The manifest selects `rocm`. A host whose ROCm build sits in the nightly
        // directory still serves from it, and a vulkan install is never used.
        install("rocm-nightly");
        install("vulkan");
        assert_eq!(
            direct_serve_binary(&manifest),
            Some(llamacpp.join("rocm-nightly").join(&server))
        );

        // With rocm-stable also present, it wins: it is first in the ROCm family order.
        install("rocm-stable");
        assert_eq!(
            direct_serve_binary(&manifest),
            Some(llamacpp.join("rocm-stable").join(&server))
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn collect_gguf_files_recurses_and_filters() {
        let dir = scratch_dir("collect-gguf");
        fs::create_dir_all(dir.join("shard")).unwrap();
        fs::write(dir.join("model-Q4_0.gguf"), b"x").unwrap();
        fs::write(dir.join("notes.txt"), b"x").unwrap();
        fs::write(dir.join("shard").join("model-Q8_0.gguf"), b"x").unwrap();
        let mut found = Vec::new();
        collect_gguf_files(&dir, &mut found, 0);
        let mut names = found
            .iter()
            .filter_map(|path| path.file_name().and_then(|name| name.to_str()))
            .map(str::to_owned)
            .collect::<Vec<_>>();
        names.sort();
        assert_eq!(names, vec!["model-Q4_0.gguf", "model-Q8_0.gguf"]);
        fs::remove_dir_all(&dir).ok();
    }

    /// Hugging Face snapshots hold symlinks into `blobs/`; one whose blob is gone (an
    /// interrupted download) is neither a directory nor a file, so only the `is_file()`
    /// check keeps it out of the model list.
    #[cfg(unix)]
    #[test]
    fn collect_gguf_files_skips_dangling_symlinks() {
        let dir = scratch_dir("collect-gguf-dangling");
        fs::write(dir.join("real-Q4_0.gguf"), b"x").unwrap();
        std::os::unix::fs::symlink(dir.join("missing-blob"), dir.join("dangling-Q8_0.gguf"))
            .unwrap();
        assert!(!dir.join("dangling-Q8_0.gguf").is_dir());
        let mut found = Vec::new();
        collect_gguf_files(&dir, &mut found, 0);
        assert_eq!(found, vec![dir.join("real-Q4_0.gguf")]);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn download_uses_validated_checkpoint_ref_not_the_alias_and_returns_the_file() {
        let dir = scratch_dir("download-ref");
        let paths = AppPaths {
            config_dir: dir.join("config"),
            data_dir: dir.join("data"),
            cache_dir: dir.join("cache"),
        };
        let checkpoint = parse_hf_checkpoint("pull-test-owner/pull-test-repo:Q4_0").unwrap();
        let model = DirectLlamaModel::Checkpoint(checkpoint);
        let snapshot = paths
            .cache_dir
            .join("huggingface/hub/models--pull-test-owner--pull-test-repo/snapshots/revision");
        let mut pulled = Vec::new();
        let path = ensure_model_with_download(&paths, "qwen-smoke", &model, |reference| {
            pulled.push(reference.to_owned());
            fs::create_dir_all(&snapshot).unwrap();
            fs::write(snapshot.join("Qwen3-0.6B-Q4_0.gguf"), b"x").unwrap();
            Ok(())
        })
        .unwrap();
        assert_eq!(pulled, ["pull-test-owner/pull-test-repo:Q4_0"]);
        assert_eq!(path, snapshot.join("Qwen3-0.6B-Q4_0.gguf"));
        // A cached file is served without pulling again.
        let path = ensure_model_with_download(&paths, "qwen-smoke", &model, |_| {
            panic!("cached model must not be pulled")
        })
        .unwrap();
        assert_eq!(path, snapshot.join("Qwen3-0.6B-Q4_0.gguf"));
        fs::remove_dir_all(&dir).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn real_download_closure_pulls_the_checkpoint_not_the_requested_alias() {
        use crate::test_http::fake_runtime;
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        // The fake `lemonade pull` drops the GGUF into the cache, as the real one does.
        let manifest = fake_runtime::manifest(
            dir,
            0,
            Some("models--pull-test-owner--pull-test-repo/snapshots/rev/model-Q4_0.gguf"),
        );
        let runtime = LemonadeRuntime { manifest };
        let paths = AppPaths {
            config_dir: dir.join("config"),
            data_dir: dir.join("data"),
            cache_dir: dir.join("cache"),
        };
        // A request whose text is an alias, resolved to a registered checkpoint.
        let request = serve_request("qwen-smoke");
        let model = DirectLlamaModel::Checkpoint(
            parse_hf_checkpoint("pull-test-owner/pull-test-repo:Q4_0").unwrap(),
        );
        let path = ensure_model_in(
            &paths,
            &request,
            &runtime,
            &LemonadeProcessEnvironment::default(),
            None,
            &model,
            &ParentRuntimeEnvironment::rooted_at(dir),
        )
        .unwrap();
        assert!(path.is_file(), "{}", path.display());
        let pulls = fake_runtime::calls(dir)
            .into_iter()
            .filter(|call| call.contains(" pull "))
            .collect::<Vec<_>>();
        assert_eq!(pulls.len(), 1, "{pulls:?}");
        let argv = pulls[0].split_whitespace().collect::<Vec<_>>();
        assert_eq!(
            argv[argv.len() - 2..],
            ["pull", "pull-test-owner/pull-test-repo:Q4_0"],
            "{pulls:?}"
        );
        assert!(!pulls[0].contains("qwen-smoke"), "{pulls:?}");
    }

    #[test]
    fn split_gguf_variant_resolves_to_its_first_shard() {
        let dir = scratch_dir("sharded");
        let paths = AppPaths {
            config_dir: dir.join("config"),
            data_dir: dir.join("data"),
            cache_dir: dir.join("cache"),
        };
        let model = DirectLlamaModel::Checkpoint(
            parse_hf_checkpoint("shard-test-owner/shard-test-repo:Q4_K_M").unwrap(),
        );
        let folder = paths
            .cache_dir
            .join("huggingface/hub/models--shard-test-owner--shard-test-repo/snapshots/rev/Q4_K_M");
        fs::create_dir_all(&folder).unwrap();
        // One shard of two: the set is incomplete, so it counts as not downloaded
        // (and is pulled again) rather than being served partial.
        fs::write(folder.join("model-Q4_K_M-00002-of-00002.gguf"), b"x").unwrap();
        assert_eq!(direct_llama_model_path(&paths, &model).unwrap(), None);
        // Both shards present: one model, served from the first shard.
        fs::write(folder.join("model-Q4_K_M-00001-of-00002.gguf"), b"x").unwrap();
        assert_eq!(
            direct_llama_model_path(&paths, &model).unwrap(),
            Some(folder.join("model-Q4_K_M-00001-of-00002.gguf"))
        );
        // An unrelated file with the same label makes it ambiguous again.
        fs::write(folder.join("other-Q4_K_M.gguf"), b"x").unwrap();
        let error = direct_llama_model_path(&paths, &model)
            .unwrap_err()
            .to_string();
        assert!(error.contains("matches multiple files"), "{error}");
        fs::remove_dir_all(&dir).ok();
    }

    fn shard_test_paths(dir: &Path) -> AppPaths {
        AppPaths {
            config_dir: dir.join("config"),
            data_dir: dir.join("data"),
            cache_dir: dir.join("cache"),
        }
    }

    fn shard_checkpoint(variant: &str) -> DirectLlamaModel {
        DirectLlamaModel::Checkpoint(
            parse_hf_checkpoint(&format!("shard-test-owner/shard-test-repo:{variant}")).unwrap(),
        )
    }

    fn shard_snapshot(paths: &AppPaths, revision: &str) -> PathBuf {
        let folder = paths.cache_dir.join(format!(
            "huggingface/hub/models--shard-test-owner--shard-test-repo/snapshots/{revision}"
        ));
        fs::create_dir_all(&folder).unwrap();
        folder
    }

    #[test]
    fn split_set_ambiguity_suggestion_resolves_to_the_whole_set() {
        let dir = scratch_dir("shard-suggestion");
        let paths = shard_test_paths(&dir);
        let folder = shard_snapshot(&paths, "rev");
        for file in [
            "model-Q4_K_M-00001-of-00002.gguf",
            "model-Q4_K_M-00002-of-00002.gguf",
            "other-Q4_K_M.gguf",
        ] {
            fs::write(folder.join(file), b"x").unwrap();
        }
        let error = direct_llama_model_path(&paths, &shard_checkpoint("Q4_K_M"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("matches multiple files"), "{error}");
        let suggestion = error
            .split_once("e.g. `shard-test-owner/shard-test-repo:")
            .and_then(|(_, rest)| rest.split_once('`'))
            .map_or_else(
                || panic!("no suggestion in {error}"),
                |(variant, _)| variant.to_owned(),
            );
        assert!(suggestion.contains("-of-"), "{suggestion}");
        // The suggestion, fed back, serves the split model from its first shard.
        assert_eq!(
            direct_llama_model_path(&paths, &shard_checkpoint(&suggestion)).unwrap(),
            Some(folder.join("model-Q4_K_M-00001-of-00002.gguf"))
        );
        // A shard name alone never serves a partial set.
        fs::remove_file(folder.join("model-Q4_K_M-00002-of-00002.gguf")).unwrap();
        assert_eq!(
            direct_llama_model_path(&paths, &shard_checkpoint(&suggestion)).unwrap(),
            None
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn shards_spread_over_cache_roots_are_not_a_complete_set() {
        let dir = scratch_dir("shard-roots");
        let checkpoint = parse_hf_checkpoint("shard-test-owner/shard-test-repo:Q4_K_M").unwrap();
        let mut roots = Vec::new();
        for (name, shard) in [("h1", 1), ("h2", 2)] {
            let root = dir.join(name);
            let folder = root.join("models--shard-test-owner--shard-test-repo/snapshots/rev");
            fs::create_dir_all(&folder).unwrap();
            fs::write(
                folder.join(format!("model-Q4_K_M-0000{shard}-of-00002.gguf")),
                b"x",
            )
            .unwrap();
            roots.push(root);
        }
        let matches = hf_checkpoint_gguf_matches_in_roots(&roots, &checkpoint);
        assert_eq!(matches.len(), 2, "{matches:?}");
        assert_eq!(first_shard_of_complete_set(&matches), None);
        assert!(is_incomplete_shard_set(&matches));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn shards_spread_over_snapshot_revisions_are_not_a_complete_set() {
        let dir = scratch_dir("shard-revisions");
        let paths = shard_test_paths(&dir);
        for (revision, shard) in [("revA", 1), ("revB", 2)] {
            fs::write(
                shard_snapshot(&paths, revision)
                    .join(format!("model-Q4_K_M-0000{shard}-of-00002.gguf")),
                b"x",
            )
            .unwrap();
        }
        assert_eq!(
            direct_llama_model_path(&paths, &shard_checkpoint("Q4_K_M")).unwrap(),
            None
        );
        // The same two shards in one revision are served.
        let folder = shard_snapshot(&paths, "revA");
        fs::write(folder.join("model-Q4_K_M-00002-of-00002.gguf"), b"x").unwrap();
        assert_eq!(
            direct_llama_model_path(&paths, &shard_checkpoint("Q4_K_M")).unwrap(),
            Some(folder.join("model-Q4_K_M-00001-of-00002.gguf"))
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn complete_snapshot_is_not_shadowed_by_an_older_partial_download() {
        let temporary = tempfile::tempdir().unwrap();
        let paths = shard_test_paths(temporary.path());
        let partial = shard_snapshot(&paths, "aaa");
        let complete = shard_snapshot(&paths, "bbb");
        let first = "model-Q4_K_M-00001-of-00002.gguf";
        fs::write(partial.join(first), b"x").unwrap();
        for file in [first, "model-Q4_K_M-00002-of-00002.gguf"] {
            fs::write(complete.join(file), b"x").unwrap();
        }
        for variant in ["Q4_K_M", first] {
            let model = shard_checkpoint(variant);
            let file = ensure_model_with_download(&paths, "req", &model, |_| {
                panic!("a complete cached revision must not trigger a pull")
            })
            .unwrap();
            assert_eq!(file, complete.join(first));
        }
        // A second unrelated model remains ambiguous; preferring a complete
        // revision must not turn a broad quantization into an arbitrary choice.
        fs::write(complete.join("other-Q4_K_M.gguf"), b"x").unwrap();
        assert!(
            direct_llama_model_path(&paths, &shard_checkpoint("Q4_K_M"))
                .unwrap_err()
                .to_string()
                .contains("matches multiple files")
        );
    }

    #[test]
    fn shard_indexes_outside_the_declared_range_are_not_a_complete_set() {
        let dir = scratch_dir("shard-range");
        let paths = shard_test_paths(&dir);
        let folder = shard_snapshot(&paths, "rev");
        for file in [
            "model-Q4_K_M-00001-of-00002.gguf",
            "model-Q4_K_M-00003-of-00002.gguf",
        ] {
            fs::write(folder.join(file), b"x").unwrap();
        }
        assert_eq!(
            direct_llama_model_path(&paths, &shard_checkpoint("Q4_K_M")).unwrap(),
            None
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn shard_parsing_requires_the_split_naming_scheme() {
        let parse = |name: &str| parse_gguf_shard(Path::new(name));
        assert_eq!(
            parse("gpt-oss-120b-Q4_K_M-00001-of-00002.gguf"),
            Some(("gpt-oss-120b-Q4_K_M".to_owned(), 1, 2))
        );
        assert_eq!(parse("model-Q4_K_M.gguf"), None);
        assert_eq!(parse("model-1-of-2.gguf"), None);
        assert_eq!(parse("model-00001-of-00002.bin"), None);
        assert_eq!(parse("-00001-of-00002.gguf"), None);
    }

    #[test]
    fn download_error_is_returned_and_missing_file_after_pull_is_an_error() {
        let dir = scratch_dir("download-error");
        let paths = AppPaths {
            config_dir: dir.join("config"),
            data_dir: dir.join("data"),
            cache_dir: dir.join("cache"),
        };
        let model = DirectLlamaModel::Checkpoint(
            parse_hf_checkpoint("pull-test-owner/pull-test-repo:model-Q4_0.gguf").unwrap(),
        );
        let error = ensure_model_with_download(&paths, "alias", &model, |_| {
            anyhow::bail!("Lemonade pull of `pull-test-owner/pull-test-repo:model-Q4_0.gguf` failed with status 7")
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("failed with status 7"), "{error}");
        // A pull that exits zero without leaving the file is still an error.
        let error = ensure_model_with_download(&paths, "alias", &model, |_| Ok(()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("did not download `alias`"), "{error}");
        // A missing local GGUF is never pulled.
        let local = DirectLlamaModel::Local(dir.join("gone.gguf"));
        let error = ensure_model_with_download(&paths, "gone.gguf", &local, |_| {
            panic!("a local GGUF has nothing to pull")
        })
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("local GGUF `gone.gguf` is missing"),
            "{error}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    /// Real entries from the released 11.5.1 `server_models.json`. Other released
    /// entries carry labels such as `hot`, `coding` and `mtp`; those are outside
    /// the labels direct serving accepts, so they intentionally route to the router.
    fn released_catalog_entries() -> serde_json::Value {
        serde_json::json!({
            "Qwen3-4B-Instruct-2507-GGUF": {
                "checkpoint": "unsloth/Qwen3-4B-Instruct-2507-GGUF:Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                "recipe": "llamacpp",
                "suggested": true,
                "labels": ["tool-calling"],
                "size": 2.5
            },
            "Qwen3-0.6B-GGUF": {
                "checkpoint": "unsloth/Qwen3-0.6B-GGUF:Q4_0",
                "recipe": "llamacpp",
                "suggested": true,
                "labels": ["reasoning"],
                "size": 0.38
            },
            "Qwen3.5-0.8B-GGUF": {
                "checkpoint": "unsloth/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-UD-Q4_K_XL.gguf",
                "mmproj": "mmproj-F16.gguf",
                "recipe": "llamacpp",
                "suggested": true,
                "labels": ["vision", "tool-calling"],
                "size": 0.764
            },
            "gpt-oss-120b-mxfp-GGUF": {
                "checkpoint": "ggml-org/gpt-oss-120b-GGUF:*",
                "recipe": "llamacpp",
                "suggested": true,
                "labels": ["reasoning", "tool-calling"],
                "size": 63.0
            }
        })
    }

    #[test]
    fn released_catalog_default_model_routes_direct_with_binary_and_router_without() {
        let dir = scratch_dir("released-catalog");
        let mut manifest = test_manifest(dir.join("runtime"));
        manifest.backend_name = "vulkan".to_owned();
        write_registry(&manifest, released_catalog_entries());
        let binary = manifest
            .runtime_dir
            .join("bin/llamacpp/vulkan")
            .join(platform_binary_name("llama-server"));

        // The engine default is the first released entry.
        assert_eq!(DEFAULT_MODEL, "Qwen3-4B-Instruct-2507-GGUF");
        let direct_checkpoints = [
            (
                DEFAULT_MODEL,
                "unsloth/Qwen3-4B-Instruct-2507-GGUF:Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
            ),
            ("Qwen3-0.6B-GGUF", "unsloth/Qwen3-0.6B-GGUF:Q4_0"),
        ];

        // No selected-backend binary: every entry stays on the router.
        for name in [DEFAULT_MODEL, "Qwen3-0.6B-GGUF", "Qwen3.5-0.8B-GGUF"] {
            assert!(
                matches!(
                    serving_route(&manifest, &serve_request(name), true).unwrap(),
                    ServeRoute::Router { ref model_ref } if model_ref == name
                ),
                "{name}"
            );
        }

        fs::create_dir_all(binary.parent().unwrap()).unwrap();
        fs::write(&binary, b"x").unwrap();
        for (name, checkpoint) in direct_checkpoints {
            let route = serving_route(&manifest, &serve_request(name), true).unwrap();
            let ServeRoute::Direct { model, server } = route else {
                panic!("{name} should serve directly with a selected-backend binary");
            };
            assert_eq!(server, binary);
            assert!(
                matches!(&model, DirectLlamaModel::Checkpoint(found) if found.model_ref() == checkpoint),
                "{name}: {model:?}"
            );
        }
        // The multimodal entry carries `mmproj`, which only the router handles.
        assert!(matches!(
            serving_route(&manifest, &serve_request("Qwen3.5-0.8B-GGUF"), true).unwrap(),
            ServeRoute::Router { model_ref } if model_ref == "Qwen3.5-0.8B-GGUF"
        ));
        // `ggml-org/gpt-oss-120b-GGUF:*` is how the catalog says "every file". Its
        // variant is not a filename, so it is never served directly; with the
        // binary present it still goes to the router on Linux and elsewhere, and
        // is not an error on either. The fallback after a router failure agrees.
        let wildcard = "gpt-oss-120b-mxfp-GGUF";
        for is_linux in [true, false] {
            assert!(
                matches!(
                    serving_route(&manifest, &serve_request(wildcard), is_linux).unwrap(),
                    ServeRoute::Router { ref model_ref } if model_ref == wildcard
                ),
                "is_linux {is_linux}"
            );
            assert!(
                direct_fallback_route(&manifest, &serve_request(wildcard), is_linux).is_none(),
                "is_linux {is_linux}"
            );
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rocm_channel_names_are_not_manifest_backends() {
        // Installs record `rocm` or `vulkan`; `rocm-stable` and `rocm-nightly` are only
        // directory names inside the `rocm` family, so a manifest naming one finds
        // nothing and is refused as an unsupported backend, not as a missing install.
        let dir = scratch_dir("channel-names");
        let mut manifest = test_manifest(dir.join("runtime"));
        let stable = manifest.runtime_dir.join("bin/llamacpp/rocm-stable");
        fs::create_dir_all(&stable).unwrap();
        fs::write(stable.join(platform_binary_name("llama-server")), b"x").unwrap();
        assert!(direct_serve_binary(&manifest).is_some());
        for name in ["rocm-stable", "rocm-nightly"] {
            manifest.backend_name = name.to_owned();
            assert_eq!(direct_serve_binary(&manifest), None, "{name}");
            let error = serving_route(&manifest, &serve_request("owner/repo:Q4_0"), true)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("does not support GPU direct serving"),
                "{error}"
            );
        }
        fs::remove_dir_all(dir).unwrap();
    }

    fn test_manifest(runtime_dir: PathBuf) -> LemonadeInstallManifest {
        LemonadeInstallManifest {
            env_id: "test".to_owned(),
            version: rocm_deps::LEMONADE_VERSION.to_owned(),
            runtime_dir,
            lemond: PathBuf::from("lemond"),
            lemonade: PathBuf::from("lemonade"),
            backend_recipe: LLAMACPP_RECIPE.to_owned(),
            backend_name: ROCM_BACKEND_NAME.to_owned(),
            installed_at_unix_ms: 0,
        }
    }

    #[test]
    fn direct_qwen_lookup_checks_huggingface_cache_env() {
        let paths = AppPaths {
            config_dir: PathBuf::from("config"),
            data_dir: PathBuf::from("data"),
            cache_dir: PathBuf::from("rocm-cache"),
        };
        let roots = hf_cache_roots_from(
            &paths,
            |name| match name {
                "HUGGINGFACE_HUB_CACHE" => Some(PathBuf::from("hf-hub")),
                "HF_HOME" => Some(PathBuf::from("hf-home")),
                "HOME" => Some(PathBuf::from("home")),
                _ => None,
            },
            // `HOME` is set above, so the fallback must never be consulted.
            || panic!("home_dir fallback must not run when HOME is set"),
        );

        assert_eq!(
            roots,
            vec![
                PathBuf::from("hf-hub"),
                PathBuf::from("hf-home").join("hub"),
                PathBuf::from("rocm-cache").join("huggingface").join("hub"),
                PathBuf::from("home")
                    .join(".cache")
                    .join("huggingface")
                    .join("hub"),
            ]
        );
    }

    #[test]
    fn hf_cache_roots_falls_back_to_platform_home_when_home_unset() {
        // Windows never sets `HOME`, but Lemonade still caches under
        // `%USERPROFILE%\.cache\huggingface\hub`; the lookup must fall back to the
        // platform home directory (here standing in for `runtime_home_dir`, which
        // itself resolves `USERPROFILE` on Windows) rather than skip the hub cache.
        let paths = AppPaths {
            config_dir: PathBuf::from("config"),
            data_dir: PathBuf::from("data"),
            cache_dir: PathBuf::from("rocm-cache"),
        };
        let roots = hf_cache_roots_from(
            &paths,
            |_name| None,
            || Some(PathBuf::from(r"C:\Users\dev")),
        );

        assert_eq!(
            roots,
            vec![
                PathBuf::from("rocm-cache").join("huggingface").join("hub"),
                PathBuf::from(r"C:\Users\dev")
                    .join(".cache")
                    .join("huggingface")
                    .join("hub"),
            ]
        );
    }

    #[test]
    fn hf_cache_roots_skips_home_fallback_when_platform_home_unavailable() {
        let paths = AppPaths {
            config_dir: PathBuf::from("config"),
            data_dir: PathBuf::from("data"),
            cache_dir: PathBuf::from("rocm-cache"),
        };
        let roots = hf_cache_roots_from(&paths, |_name| None, || None);

        assert_eq!(
            roots,
            vec![PathBuf::from("rocm-cache").join("huggingface").join("hub")]
        );
    }

    /// The "did not download" errors claim they searched a specific set of directories.
    /// Assert that claim against the roots `hf_cache_roots_from` actually returns, so the
    /// message cannot drift from the lookup it describes — in particular, that the
    /// Windows `USERPROFILE` fallback this fix adds is named in the text a user sees.
    #[test]
    fn hf_cache_root_description_names_every_searched_root() {
        let paths = AppPaths {
            config_dir: PathBuf::from("config"),
            data_dir: PathBuf::from("data"),
            cache_dir: PathBuf::from("rocm-cache"),
        };
        let roots = hf_cache_roots_from(
            &paths,
            |_name| None,
            || Some(PathBuf::from(r"C:\Users\dev")),
        );
        let described = render_hf_cache_roots(&roots);

        for root in &roots {
            assert!(
                described.contains(&root.display().to_string()),
                "error text {described:?} omits searched root {}",
                root.display()
            );
        }
        assert!(
            described.contains(&PathBuf::from(r"C:\Users\dev").display().to_string()),
            "error text {described:?} must name the platform-home fallback root"
        );
    }

    #[test]
    fn hf_cache_root_description_reports_when_no_roots_resolve() {
        assert_eq!(
            render_hf_cache_roots(&[]),
            "no Hugging Face cache directories could be determined"
        );
    }

    /// A stand-in llama-server that stays up until it is signalled.
    #[cfg(target_os = "linux")]
    fn fake_llama_server(dir: &Path) -> PathBuf {
        let server = dir.join("bin/llamacpp/vulkan/llama-server");
        fs::create_dir_all(server.parent().unwrap()).unwrap();
        // The script records its own PID before `exec`: afterwards the process
        // is `sleep 30` and its command line no longer names the script, so the
        // PID is the only reliable way to find it again.
        rocm_core::fake_executable::write_fake_executable(
            &server,
            "#!/bin/sh\nprintf '%s' \"$$\" > \"$(dirname \"$0\")/server.pid\"\nexec sleep 30\n",
        )
        .unwrap();
        server
    }

    /// The PID the fake server recorded, if it got as far as running its script.
    /// Waits briefly for it, so a server that was NOT stopped is seen.
    #[cfg(target_os = "linux")]
    fn recorded_server_pid(server: &Path) -> Option<u32> {
        let pid_file = server.parent().unwrap().join("server.pid");
        for _ in 0..40 {
            if let Some(pid) = fs::read_to_string(&pid_file)
                .ok()
                .and_then(|text| text.trim().parse().ok())
            {
                return Some(pid);
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        None
    }

    /// Kills and reaps a spawned child when dropped, so a failing assertion
    /// cannot leave a fake server running.
    #[cfg(target_os = "linux")]
    struct KillOnDrop(Child);

    #[cfg(target_os = "linux")]
    impl Drop for KillOnDrop {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    #[cfg(target_os = "linux")]
    fn process_exists(pid: u32) -> bool {
        Path::new(&format!("/proc/{pid}")).exists()
    }

    #[cfg(target_os = "linux")]
    #[derive(Clone, Copy, Debug)]
    enum DirectFail {
        RecordRunning,
        NotReady,
        RecordReady,
        Wait,
    }

    #[cfg(target_os = "linux")]
    struct FailingSteps(DirectFail);

    #[cfg(target_os = "linux")]
    impl DirectServerSteps for FailingSteps {
        fn record_running(&mut self, _server_pid: u32) -> Result<()> {
            if matches!(self.0, DirectFail::RecordRunning) {
                bail!("injected record_running failure");
            }
            Ok(())
        }
        fn wait_ready(&mut self) -> Result<()> {
            if matches!(self.0, DirectFail::NotReady) {
                bail!("injected readiness failure");
            }
            Ok(())
        }
        fn record_ready(&mut self, _server_pid: u32) -> Result<()> {
            if matches!(self.0, DirectFail::RecordReady) {
                bail!("injected record_ready failure");
            }
            Ok(())
        }
        fn announce_serving(&mut self) {
            // Only a failure after the server became ready (the final wait) may
            // follow an announcement; every earlier failure must not announce.
            assert!(
                matches!(self.0, DirectFail::Wait),
                "{:?}: announced serving although a step before ready failed",
                self.0
            );
        }
        fn wait_exit(&mut self, _child: &mut Child) -> std::io::Result<ExitStatus> {
            Err(std::io::Error::other("injected wait failure"))
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn direct_server_step_failure_stops_the_child_and_marks_state_failed() {
        for fail in [
            DirectFail::RecordRunning,
            DirectFail::NotReady,
            DirectFail::RecordReady,
            DirectFail::Wait,
        ] {
            let dir = crate::test_support::scratch_dir("direct-supervise");
            let server = fake_llama_server(&dir);
            let state_path = dir.join("state.json");
            fs::write(&state_path, r#"{"status":"running"}"#).unwrap();
            let mut guard = KillOnDrop(ProcessCommand::new(&server).spawn().unwrap());
            let pid = guard.0.id();
            assert!(process_exists(pid), "{fail:?}: fake server did not start");
            let error = supervise_direct_server(&mut guard.0, &state_path, &mut FailingSteps(fail))
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("injected") || error.contains("waiting"),
                "{error}"
            );
            assert!(
                !process_exists(pid),
                "{fail:?}: server {pid} is still running after the failure"
            );
            let state = fs::read_to_string(&state_path).unwrap();
            assert!(
                state.contains(r#""status": "failed""#) || state.contains(r#""status":"failed""#),
                "{fail:?}: {state}"
            );
            fs::remove_dir_all(dir).unwrap();
        }
    }

    /// A direct serve whose server never becomes ready must not leave a log line
    /// claiming Lemonade is serving: the log is where a managed serve is read, and
    /// the state next to it says `failed`.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_direct_serve_that_never_becomes_ready_does_not_log_that_it_is_serving() {
        let dir = crate::test_support::scratch_dir("direct-not-ready");
        let server = fake_llama_server(&dir);
        let model = dir.join("model.gguf");
        fs::write(&model, b"x").unwrap();
        let state_path = dir.join("state.json");
        fs::write(&state_path, r#"{"status":"starting"}"#).unwrap();
        let log_path = dir.join("serve.log");
        let request = ServeHttpRequest {
            service_id: "test".to_owned(),
            model_ref: "owner/repo:Q4_0".to_owned(),
            host: "127.0.0.1".to_owned(),
            // Nothing listens here, so the readiness probe can only time out.
            port: 11438,
            device_policy: rocm_engine_protocol::DevicePolicy::GpuRequired,
            gpu_indices: vec![0],
            runtime_id: None,
            env_id: None,
            state_path: state_path.clone(),
            log_path: Some(log_path.clone()),
            engine_recipe: None,
        };
        let runtime = LemonadeRuntime {
            manifest: test_manifest(dir.join("runtime")),
        };
        let error = serve_direct_llama_server_within(
            &request,
            &runtime,
            &LemonadeProcessEnvironment::default(),
            &server,
            &model,
            Some(&log_path),
            "test",
            &ParentRuntimeEnvironment::rooted_at(&dir),
            Duration::from_secs(1),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("did not become ready"), "{error}");
        let still_running = recorded_server_pid(&server)
            .is_some_and(|pid| Path::new(&format!("/proc/{pid}")).exists());
        if still_running && let Some(pid) = recorded_server_pid(&server) {
            let _ = terminate_pid(pid, true);
        }
        assert!(!still_running, "the server outlived the failed serve");
        let state = fs::read_to_string(&state_path).unwrap();
        assert!(state.contains("failed"), "{state}");
        let log = fs::read_to_string(&log_path).unwrap();
        assert!(log.contains("Launching Lemonade packaged vulkan"), "{log}");
        assert!(!log.contains("is serving on"), "{log}");
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn direct_server_stops_when_state_cannot_be_written_after_spawn() {
        let dir = crate::test_support::scratch_dir("direct-unwritable");
        let server = fake_llama_server(&dir);
        let model = dir.join("model.gguf");
        fs::write(&model, b"x").unwrap();
        // A directory at the state path makes the first post-spawn write fail.
        let state_path = dir.join("state.json");
        fs::create_dir_all(&state_path).unwrap();
        let request = ServeHttpRequest {
            service_id: "test".to_owned(),
            model_ref: "owner/repo:Q4_0".to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 11437,
            device_policy: rocm_engine_protocol::DevicePolicy::GpuRequired,
            gpu_indices: vec![0],
            runtime_id: None,
            env_id: None,
            state_path,
            log_path: None,
            engine_recipe: None,
        };
        let runtime = LemonadeRuntime {
            manifest: test_manifest(dir.join("runtime")),
        };
        let result = serve_direct_llama_server(
            &request,
            &runtime,
            &LemonadeProcessEnvironment::default(),
            &server,
            &model,
            None,
            "test",
            &ParentRuntimeEnvironment::rooted_at(&dir),
        );
        // A server that was not stopped runs its script and records its PID; one
        // that was stopped first either never got that far or is gone by now.
        let still_running = recorded_server_pid(&server)
            .is_some_and(|pid| Path::new(&format!("/proc/{pid}")).exists());
        // Clean up before asserting so a failure cannot leave the server behind.
        if still_running && let Some(pid) = recorded_server_pid(&server) {
            let _ = terminate_pid(pid, true);
        }
        assert!(result.is_err());
        assert!(!still_running, "the direct server was left running");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn serving_direct_command_preserves_alias_model_path_and_gpu_backend() {
        let request = ServeHttpRequest {
            service_id: "test".to_owned(),
            model_ref: "qwen-smoke".to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 11435,
            device_policy: rocm_engine_protocol::DevicePolicy::GpuRequired,
            gpu_indices: vec![0],
            runtime_id: None,
            env_id: None,
            state_path: PathBuf::from("test-state"),
            log_path: None,
            engine_recipe: Some(EngineRecipeHint {
                required_flags: vec!["--ctx-size".to_owned(), "8192".to_owned()],
                ..EngineRecipeHint::default()
            }),
        };
        let server = Path::new("runtime/bin/llamacpp/vulkan/llama-build/llama-server");
        let model = Path::new("cache/model-Q4_0.gguf");
        let command = direct_llama_server_command(&request, server, model);
        assert_eq!(command.get_program(), server.as_os_str());
        let args = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            args,
            [
                "-m",
                "cache/model-Q4_0.gguf",
                "--host",
                "127.0.0.1",
                "--port",
                "11435",
                "--n-gpu-layers",
                "999",
                "--alias",
                "qwen-smoke",
                "--ctx-size",
                "8192"
            ]
        );
        assert_eq!(llama_server_backend_label(server), "vulkan");
        assert!(
            parse_models_ready(r#"{"data":[{"id":"qwen-smoke"}]}"#, &request.model_ref).unwrap()
        );
        assert!(
            !parse_models_ready(r#"{"data":[{"id":"Qwen3-0.6B-GGUF"}]}"#, &request.model_ref)
                .unwrap()
        );
    }

    #[test]
    fn serving_regression_direct_readiness_requires_requested_alias() {
        assert!(parse_models_ready(r#"{"data":[{"id":"qwen-smoke"}]}"#, "qwen-smoke").unwrap());
        assert!(
            !parse_models_ready(r#"{"data":[{"id":"Qwen3-0.6B-GGUF"}]}"#, "qwen-smoke").unwrap()
        );
        // The router deliberately normalizes known aliases, unlike direct serving.
        assert!(models_payload_has_ready_model(
            &json!({"data": [{
                "id": "Qwen3-0.6B-GGUF", "recipe_options": {"llamacpp_backend": "vulkan"}
            }]}),
            "qwen-smoke",
            "vulkan"
        ));
    }
}
