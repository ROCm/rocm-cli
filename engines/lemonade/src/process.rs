// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, RocmCliConfig, active_managed_therock_environment, format_http_base_url,
    http_get_text_with_auth, prepend_runtime_paths, runtime_is_linux, runtime_is_windows,
};
use rocm_engine_protocol::EngineRecipeHint;
use serde_json::{Value, json};
use std::ffi::OsString;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::time::Duration;

use crate::backend_alignment::LemonadeRuntime;
use crate::direct_llama::{direct_llama_model_path, llama_server_backend_label};
use crate::install::LemonadeInstallManifest;
use crate::runtime_dir::{ParentRuntimeEnvironment, child_runtime_dir_var};
use crate::state::{
    mark_json_status, merge_json_state, parse_http_endpoint, resolve_lemonade_model_ref,
    tail_lines, terminate_pid, write_running_state,
};
use crate::{LLAMACPP_RECIPE, ServeHttpRequest};

const STARTUP_FAILURE_LOG_TAIL_LINES: usize = 80;

#[derive(Debug, Clone, Default)]
pub(crate) struct LemonadeProcessEnvironment {
    pub(crate) rocm_root: Option<PathBuf>,
    pub(crate) path_entries: Vec<PathBuf>,
    pub(crate) library_entries: Vec<PathBuf>,
    pub(crate) gpu_indices: Vec<u32>,
}

fn windows_child_path(path: &Path) -> String {
    let raw = path.display().to_string();
    let normalized = raw.replace('\\', "/");
    let bytes = normalized.as_bytes();
    if bytes.len() >= 3 && bytes[0] == b'/' && bytes[1].is_ascii_alphabetic() && bytes[2] == b'/' {
        let drive = (bytes[1] as char).to_ascii_uppercase();
        let rest = normalized[3..].replace('/', "\\");
        return format!("{drive}:\\{rest}");
    }
    if bytes.len() == 2 && bytes[0] == b'/' && bytes[1].is_ascii_alphabetic() {
        let drive = (bytes[1] as char).to_ascii_uppercase();
        return format!("{drive}:\\");
    }
    raw
}

pub(crate) fn spawn_lemond(
    manifest: &LemonadeInstallManifest,
    host: &str,
    port: u16,
    log_path: Option<&Path>,
    process_env: &LemonadeProcessEnvironment,
) -> Result<LemondChild> {
    #[cfg(windows)]
    if let Some(log_path) = log_path {
        let args = vec![
            child_process_path(&manifest.runtime_dir),
            "--host".to_owned(),
            host.to_owned(),
            "--port".to_owned(),
            port.to_string(),
        ];
        let pid = rocm_core::spawn_hidden_console_with_log(&manifest.lemond, &args, &[], log_path)
            .with_context(|| format!("failed to start {}", manifest.lemond.display()))?;
        return Ok(LemondChild::Pid(pid));
    }

    let mut command = ProcessCommand::new(&manifest.lemond);
    command
        .arg(child_process_path(&manifest.runtime_dir))
        .arg("--host")
        .arg(host)
        .arg("--port")
        .arg(port.to_string())
        .stdin(Stdio::null());
    apply_lemonade_process_environment(&mut command, process_env)?;
    if let Some(log_path) = log_path {
        if let Some(parent) = log_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let log = fs::File::create(log_path)
            .with_context(|| format!("failed to create {}", log_path.display()))?;
        command.stdout(Stdio::from(log.try_clone()?));
        command.stderr(Stdio::from(log));
    } else {
        command.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    }
    hide_child_console_window(&mut command);
    command
        .spawn()
        .with_context(|| format!("failed to start {}", manifest.lemond.display()))
        .map(LemondChild::Child)
}

pub(crate) enum LemondChild {
    Child(std::process::Child),
    #[cfg(windows)]
    Pid(u32),
}

impl LemondChild {
    pub(crate) fn id(&self) -> u32 {
        match self {
            Self::Child(child) => child.id(),
            #[cfg(windows)]
            Self::Pid(pid) => *pid,
        }
    }

    pub(crate) fn wait(&mut self) -> Result<LemondExitStatus> {
        match self {
            Self::Child(child) => {
                let status = child.wait().context("failed waiting for Lemonade server")?;
                Ok(LemondExitStatus {
                    success: status.success(),
                    description: status.to_string(),
                })
            }
            #[cfg(windows)]
            Self::Pid(pid) => {
                let code = rocm_core::wait_for_process_exit(*pid)?;
                Ok(LemondExitStatus {
                    success: code == 0,
                    description: format!("exit code {code}"),
                })
            }
        }
    }
}

pub(crate) struct LemondExitStatus {
    success: bool,
    description: String,
}

impl LemondExitStatus {
    pub(crate) const fn success(&self) -> bool {
        self.success
    }
}

impl std::fmt::Display for LemondExitStatus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.description)
    }
}

#[cfg(windows)]
pub(crate) fn hide_child_console_window(command: &mut ProcessCommand) {
    use std::os::windows::process::CommandExt;

    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
pub(crate) const fn hide_child_console_window(_command: &mut ProcessCommand) {}

fn child_process_path(path: &Path) -> String {
    if runtime_is_windows() {
        windows_child_path(path)
    } else {
        path.display().to_string()
    }
}

pub(crate) fn lemonade_process_environment() -> Result<LemonadeProcessEnvironment> {
    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths).unwrap_or_default();
    let Some(env) = active_managed_therock_environment(&paths, &config)? else {
        return Ok(LemonadeProcessEnvironment::default());
    };
    Ok(LemonadeProcessEnvironment {
        rocm_root: env.rocm_root,
        path_entries: env.path_entries,
        library_entries: env.library_entries,
        gpu_indices: Vec::new(),
    })
}

pub(crate) fn apply_lemonade_process_environment(
    command: &mut ProcessCommand,
    env: &LemonadeProcessEnvironment,
) -> Result<()> {
    let vars = lemonade_process_environment_vars(env, &ParentRuntimeEnvironment::current())?;
    apply_environment_vars(command, &vars);
    Ok(())
}

/// Apply an already-resolved variable set. Callers that spawn repeatedly (the
/// readiness poll) resolve once and reuse, so preparing the runtime directory
/// does not repeat its syscalls on every attempt.
fn apply_environment_vars(command: &mut ProcessCommand, vars: &[(&'static str, OsString)]) {
    for (key, value) in vars {
        command.env(key, value);
    }
}

/// The parts of *this* process's environment that decide where the child's
/// runtime directory goes.
///
/// Captured into a value instead of being read inside
/// [`lemonade_process_environment_vars`] so the precedence is testable without
/// mutating process-global env vars, which is `unsafe` and racy under parallel
/// tests in edition 2024.
pub(crate) fn lemonade_process_environment_vars(
    env: &LemonadeProcessEnvironment,
    parent: &ParentRuntimeEnvironment,
) -> Result<Vec<(&'static str, OsString)>> {
    let mut vars = Vec::new();
    if let Some(rocm_root) = env.rocm_root.as_ref() {
        vars.push(("ROCM_PATH", rocm_root.as_os_str().to_owned()));
    }
    if let Some(runtime_dir) = child_runtime_dir_var(parent)? {
        vars.push(runtime_dir);
    }
    let mut path_entries = env.path_entries.clone();
    if runtime_is_windows() {
        path_entries.extend(env.library_entries.iter().cloned());
    }
    if let Some(path) = prepend_runtime_paths(&path_entries, std::env::var_os("PATH"))? {
        vars.push(("PATH", path));
    }
    if runtime_is_linux()
        && let Some(ld_library_path) =
            prepend_runtime_paths(&env.library_entries, std::env::var_os("LD_LIBRARY_PATH"))?
    {
        vars.push(("LD_LIBRARY_PATH", ld_library_path));
    }
    if let Some(csv) = rocm_engine_protocol::gpu_indices_to_csv(&env.gpu_indices) {
        vars.push(("HIP_VISIBLE_DEVICES", OsString::from(csv)));
    }
    // When `rocm serve` protects a public endpoint, the key arrives via
    // `ROCM_SERVE_API_KEY[_FILE]`. Lemonade's server gates /api,/v0,/v1 on
    // `LEMONADE_API_KEY`, and its own CLI clients (used here for the readiness
    // `status` probe) auto-present the same env var — so translating it once here
    // both secures the server and keeps our readiness checks authenticated.
    if let Some(api_key) = rocm_engine_protocol::resolve_endpoint_api_key() {
        vars.push(("LEMONADE_API_KEY", OsString::from(api_key)));
    }
    Ok(vars)
}

fn push_existing_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !path.exists() || paths.iter().any(|existing| existing == &path) {
        return;
    }
    paths.push(path);
}

pub(crate) fn wait_for_lemonade_cli_status(
    manifest: &LemonadeInstallManifest,
    host: &str,
    port: u16,
    timeout: Duration,
    log_path: Option<&Path>,
    process_env: &LemonadeProcessEnvironment,
) -> Result<()> {
    let start = std::time::Instant::now();
    let mut last_status = None;
    // Resolved once: the poll runs twice a second, and resolving also prepares
    // the child's runtime directory on disk.
    let env_vars =
        lemonade_process_environment_vars(process_env, &ParentRuntimeEnvironment::current())?;
    while start.elapsed() < timeout {
        let mut command = ProcessCommand::new(&manifest.lemonade);
        command
            .arg("--host")
            .arg(host)
            .arg("--port")
            .arg(port.to_string())
            .arg("status")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        apply_environment_vars(&mut command, &env_vars);
        hide_child_console_window(&mut command);
        match command.status() {
            Ok(status) if status.success() => return Ok(()),
            Ok(status) => last_status = Some(status.to_string()),
            Err(error) => last_status = Some(error.to_string()),
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    let startup_log_summary = summarize_startup_log_tail(log_path, STARTUP_FAILURE_LOG_TAIL_LINES);
    bail!(
        "Lemonade server did not become ready: {}; {}",
        last_status.unwrap_or_else(|| "not checked".to_owned()),
        startup_log_summary
    )
}

fn summarize_startup_log_tail(log_path: Option<&Path>, limit: usize) -> String {
    let Some(log_path) = log_path else {
        return "no Lemonade startup log path was configured".to_owned();
    };
    if !log_path.is_file() {
        return format!("Lemonade startup log not found at {}", log_path.display());
    }
    match tail_lines(log_path, limit) {
        Ok(lines) if lines.is_empty() => {
            format!("Lemonade startup log {} is empty", log_path.display())
        }
        Ok(lines) => format!(
            "Lemonade startup log tail ({}):\n{}",
            log_path.display(),
            lines.join("\n")
        ),
        Err(error) => format!(
            "failed to read Lemonade startup log {}: {error}",
            log_path.display()
        ),
    }
}

pub(crate) fn run_lemonade_backend_install(
    manifest: &LemonadeInstallManifest,
    host: &str,
    port: u16,
    backend: &str,
    process_env: &LemonadeProcessEnvironment,
) -> Result<()> {
    let mut command = ProcessCommand::new(&manifest.lemonade);
    command
        .arg("--host")
        .arg(host)
        .arg("--port")
        .arg(port.to_string())
        .arg("backends")
        .arg("install")
        .arg(format!("{LLAMACPP_RECIPE}:{backend}"))
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    apply_lemonade_process_environment(&mut command, process_env)?;
    hide_child_console_window(&mut command);
    let status = command
        .status()
        .with_context(|| format!("failed to run {}", manifest.lemonade.display()))?;
    if !status.success() {
        bail!("Lemonade backend install failed with status {status}");
    }
    Ok(())
}

fn lemonade_pull_args(host: &str, port: u16, checkpoint_ref: &str) -> Vec<String> {
    vec![
        "--host".to_owned(),
        host.to_owned(),
        "--port".to_owned(),
        port.to_string(),
        "pull".to_owned(),
        checkpoint_ref.to_owned(),
    ]
}

/// Download a canonical Hugging Face checkpoint (`owner/repo:variant`) through Lemonade
/// so its GGUF lands in the HF hub cache. Runs non-interactively, so a `:variant` is
/// required — a bare `owner/repo` triggers Lemonade's interactive variant menu, which
/// cannot be answered here.
pub(crate) fn run_lemonade_pull(
    manifest: &LemonadeInstallManifest,
    host: &str,
    port: u16,
    checkpoint_ref: &str,
    log_path: Option<&Path>,
    process_env: &LemonadeProcessEnvironment,
) -> Result<()> {
    let mut command = ProcessCommand::new(&manifest.lemonade);
    command
        .args(lemonade_pull_args(host, port, checkpoint_ref))
        .stdin(Stdio::null());
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
    apply_lemonade_process_environment(&mut command, process_env)?;
    hide_child_console_window(&mut command);
    let status = command
        .status()
        .with_context(|| format!("failed to run {}", manifest.lemonade.display()))?;
    if !status.success() {
        bail!("Lemonade pull of `{checkpoint_ref}` failed with status {status}");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_lemonade_model_load(
    manifest: &LemonadeInstallManifest,
    host: &str,
    port: u16,
    model_ref: &str,
    backend: &str,
    engine_recipe: Option<&EngineRecipeHint>,
    log_path: Option<&Path>,
    process_env: &LemonadeProcessEnvironment,
) -> Result<()> {
    let mut command = ProcessCommand::new(&manifest.lemonade);
    command
        .args(lemonade_model_load_args(
            host,
            port,
            model_ref,
            backend,
            engine_recipe,
        ))
        .stdin(Stdio::null());
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
    apply_lemonade_process_environment(&mut command, process_env)?;
    hide_child_console_window(&mut command);
    let status = command
        .status()
        .with_context(|| format!("failed to run {}", manifest.lemonade.display()))?;
    if !status.success() {
        bail!("Lemonade model load failed with status {status}");
    }
    Ok(())
}

/// Run one of Lemonade's packaged `llama-server` binaries (`server`) directly on a
/// GGUF file, exposing the model under `request.model_ref` via `--alias`. This is how
/// a canonical Hugging Face name is served under exactly that name — Lemonade's router
/// renames registered models, but an llama-server alias is a free-form string.
pub(crate) fn serve_direct_llama_server(
    request: &ServeHttpRequest,
    runtime: &LemonadeRuntime,
    process_env: &LemonadeProcessEnvironment,
    server: &Path,
    log_path: Option<&Path>,
    reason: &anyhow::Error,
) -> Result<()> {
    let paths = AppPaths::discover()?;
    let model_path = direct_llama_model_path(&paths, &request.model_ref).with_context(|| {
        format!(
            "Lemonade downloaded model `{}` was not found for direct serving",
            request.model_ref
        )
    })?;
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
            "\nLaunching Lemonade packaged {backend} llama-server directly: {reason:#}"
        )
        .ok();
    }

    let mut direct_env = process_env.clone();
    if let Some(server_dir) = server.parent() {
        push_existing_path(&mut direct_env.path_entries, server_dir.to_path_buf());
        push_existing_path(&mut direct_env.library_entries, server_dir.to_path_buf());
    }

    let mut command = ProcessCommand::new(server);
    command
        .arg("-m")
        .arg(&model_path)
        .arg("--host")
        .arg(&request.host)
        .arg("--port")
        .arg(request.port.to_string())
        .arg("--n-gpu-layers")
        .arg("999")
        .arg("--alias")
        .arg(&request.model_ref)
        .stdin(Stdio::null());
    if let Some(engine_recipe) = request.engine_recipe.as_ref() {
        command.args(&engine_recipe.required_flags);
    }
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
    apply_lemonade_process_environment(&mut command, &direct_env)?;
    hide_child_console_window(&mut command);
    let mut child = command
        .spawn()
        .with_context(|| format!("failed to start {}", server.display()))?;
    write_running_state(
        request,
        runtime,
        std::process::id(),
        Some(child.id()),
        "running",
    )?;
    // If the server never becomes ready (or fails the smoke test), don't leak the child
    // or leave the state stuck at `running`: kill it and mark the service failed.
    let readiness = (|| -> Result<()> {
        wait_for_openai_models_ready(
            &request.host,
            request.port,
            &request.model_ref,
            Duration::from_mins(2),
        )?;
        if !query_chat_smoke_endpoint(&request.host, request.port, &request.model_ref)? {
            bail!("Lemonade packaged llama-server did not pass a chat-completion smoke test");
        }
        Ok(())
    })();
    if let Err(error) = readiness {
        let _ = terminate_pid(child.id(), true);
        let _ = child.wait();
        let _ = mark_json_status(&request.state_path, "failed");
        return Err(error);
    }
    merge_json_state(
        &request.state_path,
        &json!({
            "status": "ready",
            "server_pid": child.id(),
            // Identity token for the server PID, captured while the child is alive.
            "server_start_ticks": rocm_core::process_start_ticks(child.id()),
            "backend_state": "ready",
            "backend_requested": backend,
            "backend_mode": "lemonade-packaged-llama-server",
            "load_response": {
                "status": "loaded",
                "method": "lemonade-packaged-llama-server",
                "model_name": request.model_ref,
                "model_path": model_path,
                "llamacpp_backend": backend
            },
        }),
    )?;
    let status = child
        .wait()
        .context("failed waiting for Lemonade packaged llama-server")?;
    mark_json_status(
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
        bail!("Lemonade packaged llama-server exited with status {status}")
    }
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
                    .any(|loaded| model_names_match(loaded, model_ref))
            })
        })
}

/// Whether `/v1/models` advertises `model_ref` as ready to serve on GPU. A stock
/// `llama-server` (direct-serve) entry has no `recipe_options` and is accepted by name,
/// since the direct-serve path only runs GPU backends. A Lemonade-router entry carries
/// `recipe_options`, so its `llamacpp_backend` must match — this keeps a merely
/// registered-but-unloaded model (empty `recipe_options`) from reading as ready.
fn models_payload_has_ready_model(value: &Value, model_ref: &str, backend: &str) -> bool {
    value
        .get("data")
        .or_else(|| value.get("models"))
        .and_then(Value::as_array)
        .is_some_and(|models| {
            models.iter().any(|model| {
                let name_matches = ["id", "model", "name"]
                    .into_iter()
                    .filter_map(|field| model.get(field).and_then(Value::as_str))
                    .any(|loaded| model_names_match(loaded, model_ref));
                name_matches && model_reports_ready_backend(model, backend)
            })
        })
}

/// A `/v1/models` entry is servable on GPU when it carries no `recipe_options` (a stock
/// `llama-server` direct-serve entry — GPU-only by construction) or its reported
/// `llamacpp_backend` matches the expected backend.
fn model_reports_ready_backend(model: &Value, backend: &str) -> bool {
    match model.get("recipe_options") {
        None => true,
        Some(options) => options
            .get("llamacpp_backend")
            .and_then(Value::as_str)
            .is_some_and(|loaded| lemonade_backend_matches(loaded, backend)),
    }
}

fn lemonade_model_load_args(
    host: &str,
    port: u16,
    model_ref: &str,
    backend: &str,
    engine_recipe: Option<&EngineRecipeHint>,
) -> Vec<String> {
    let mut args = vec![
        "--host".to_owned(),
        host.to_owned(),
        "--port".to_owned(),
        port.to_string(),
        "load".to_owned(),
        model_ref.to_owned(),
        "--llamacpp".to_owned(),
        backend.to_owned(),
        "--save-options".to_owned(),
    ];
    if let Some(engine_recipe) = engine_recipe.filter(|recipe| !recipe.required_flags.is_empty()) {
        args.extend([
            "--llamacpp-args".to_owned(),
            engine_recipe.required_flags.join(" "),
        ]);
    }
    args
}

fn query_health_json(host: &str, port: u16) -> Result<Value> {
    let endpoint = format_http_base_url(host, port);
    let endpoint_api_key = rocm_engine_protocol::resolve_endpoint_api_key();
    let body = http_get_text_with_auth(
        &endpoint,
        "/v1/health",
        endpoint_api_key.as_deref(),
        Duration::from_secs(3),
    )
    .with_context(|| format!("failed to query Lemonade health at {endpoint}/v1/health"))?;
    serde_json::from_str(&body).context("failed to parse Lemonade health JSON")
}

pub(crate) fn query_loaded_model_endpoint(
    endpoint_url: &str,
    model_ref: &str,
    backend: &str,
) -> Result<bool> {
    let (host, port) = parse_http_endpoint(endpoint_url)
        .with_context(|| format!("unsupported endpoint URL `{endpoint_url}`"))?;
    // Lemonade's router reports readiness via `/v1/health` (`all_models_loaded`). The
    // direct-serve path runs a stock `llama-server` that has no such field, and instead
    // advertises the model in `/v1/models`; fall back to that (name + backend) so both
    // serving modes are recognized as ready.
    if let Ok(health) = query_health_json(&host, port)
        && health_has_loaded_model(&health, model_ref, backend)
    {
        return Ok(true);
    }
    let endpoint = format_http_base_url(&host, port);
    let endpoint_api_key = rocm_engine_protocol::resolve_endpoint_api_key();
    let body = http_get_text_with_auth(
        &endpoint,
        "/v1/models",
        endpoint_api_key.as_deref(),
        Duration::from_secs(3),
    )
    .with_context(|| format!("failed to query Lemonade models at {endpoint}/v1/models"))?;
    let models =
        serde_json::from_str::<Value>(&body).context("failed to parse Lemonade /v1/models JSON")?;
    Ok(models_payload_has_ready_model(&models, model_ref, backend))
}

/// Post-load smoke test: the freshly loaded model must actually complete a chat
/// request. Deliberately stricter than the readiness probe — only a `200` counts,
/// so a refusal (wrong model name, unsupported request) fails the serve instead
/// of being reported as a working service.
pub(crate) fn query_chat_smoke_endpoint(host: &str, port: u16, model_ref: &str) -> Result<bool> {
    let status = rocm_core::openai_chat_completion_status(
        &format_http_base_url(host, port),
        model_ref,
        rocm_engine_protocol::resolve_endpoint_api_key().as_deref(),
        rocm_core::INFERENCE_PROBE_TIMEOUT,
    )?;
    Ok(status == 200)
}

fn health_has_loaded_model(health: &Value, model_ref: &str, backend: &str) -> bool {
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
            let name_matches = ["model_name", "id", "name"]
                .into_iter()
                .filter_map(|field| model.get(field).and_then(Value::as_str))
                .any(|loaded| model_names_match(loaded, model_ref));
            let backend_matches = model
                .get("recipe_options")
                .and_then(|options| options.get("llamacpp_backend"))
                .and_then(Value::as_str)
                .is_some_and(|loaded| lemonade_backend_matches(loaded, backend));
            name_matches && backend_matches
        })
}

fn model_names_match(left: &str, right: &str) -> bool {
    left.eq_ignore_ascii_case(right)
        || resolve_lemonade_model_ref(left).eq_ignore_ascii_case(&resolve_lemonade_model_ref(right))
}

fn lemonade_backend_matches(value: &str, backend: &str) -> bool {
    value
        .trim()
        .to_ascii_lowercase()
        .starts_with(&backend.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_MODEL;

    #[test]
    fn models_ready_accepts_direct_serve_and_gates_router_backend() {
        // Direct-serve (stock llama-server): no `recipe_options` → ready by name.
        let direct = json!({"data": [{"id": "LiquidAI/LFM2.5-230M-GGUF:Q4_0"}]});
        assert!(models_payload_has_ready_model(
            &direct,
            "LiquidAI/LFM2.5-230M-GGUF:Q4_0",
            "vulkan"
        ));
        // Router entry loaded on the matching backend → ready.
        let loaded = json!({"data": [{
            "id": "Qwen3-0.6B-GGUF",
            "recipe_options": {"llamacpp_backend": "vulkan"}
        }]});
        assert!(models_payload_has_ready_model(
            &loaded,
            "Qwen3-0.6B-GGUF",
            "vulkan"
        ));
        // Registered but not loaded (empty `recipe_options`) → not ready.
        let registered = json!({"data": [{
            "id": "Qwen3-0.6B-GGUF",
            "recipe_options": {}
        }]});
        assert!(!models_payload_has_ready_model(
            &registered,
            "Qwen3-0.6B-GGUF",
            "vulkan"
        ));
    }

    #[test]
    fn lemonade_pull_builds_checkpoint_command() {
        assert_eq!(
            lemonade_pull_args("127.0.0.1", 11435, "LiquidAI/LFM2.5-230M-GGUF:Q4_0"),
            vec![
                "--host",
                "127.0.0.1",
                "--port",
                "11435",
                "pull",
                "LiquidAI/LFM2.5-230M-GGUF:Q4_0",
            ]
        );
    }

    #[test]
    fn windows_child_path_maps_ape_drive_paths() {
        assert_eq!(
            windows_child_path(Path::new("/D/path/to/rocm-cli/file.zip")),
            r"D:\path\to\rocm-cli\file.zip"
        );
        assert_eq!(windows_child_path(Path::new("/c")), r"C:\");
    }

    #[test]
    fn lemonade_model_load_uses_selected_backend() {
        let args = lemonade_model_load_args("127.0.0.1", 11435, DEFAULT_MODEL, "vulkan", None);
        assert_eq!(
            args,
            vec![
                "--host",
                "127.0.0.1",
                "--port",
                "11435",
                "load",
                DEFAULT_MODEL,
                "--llamacpp",
                "vulkan",
                "--save-options",
            ]
        );
    }

    #[test]
    fn lemonade_model_load_forwards_llamacpp_recipe_flags() {
        let recipe = EngineRecipeHint {
            required_flags: vec![
                "--temperature".to_owned(),
                "0.5".to_owned(),
                "--top-p".to_owned(),
                "0.25".to_owned(),
                "--n-predict".to_owned(),
                "128".to_owned(),
            ],
            ..EngineRecipeHint::default()
        };
        let args =
            lemonade_model_load_args("127.0.0.1", 11435, DEFAULT_MODEL, "vulkan", Some(&recipe));
        assert_eq!(
            args[args.len() - 2..],
            [
                "--llamacpp-args",
                "--temperature 0.5 --top-p 0.25 --n-predict 128"
            ]
        );
    }

    #[test]
    fn health_parser_requires_loaded_requested_model() {
        let unloaded = json!({
            "status": "ok",
            "model_loaded": null,
            "all_models_loaded": []
        });
        assert!(!health_has_loaded_model(&unloaded, DEFAULT_MODEL, "vulkan"));

        let loaded = json!({
            "status": "ok",
            "model_loaded": DEFAULT_MODEL,
            "all_models_loaded": [{
                "model_name": DEFAULT_MODEL,
                "recipe": "llamacpp",
                "recipe_options": {
                    "llamacpp_backend": "vulkan"
                }
            }]
        });
        assert!(health_has_loaded_model(&loaded, "lemonade-qwen", "vulkan"));
        assert!(!health_has_loaded_model(&loaded, "lemonade-qwen", "rocm"));
    }
}
