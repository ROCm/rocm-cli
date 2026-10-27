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
use std::process::{Child, Command as ProcessCommand, ExitStatus, Stdio};
use std::time::Duration;

use crate::backend_alignment::LemonadeRuntime;
use crate::direct_llama::llama_server_backend_label;
use crate::install::LemonadeInstallManifest;
use crate::runtime_dir::{ParentRuntimeEnvironment, child_runtime_dir_var};
use crate::state::{
    MAX_TAIL_READ, mark_json_status, merge_json_state, parse_http_endpoint,
    resolve_lemonade_model_ref, terminate_pid, write_running_state,
};
use crate::{LLAMACPP_RECIPE, ServeHttpRequest};

const STARTUP_FAILURE_LOG_TAIL_LINES: usize = 80;

#[derive(Debug, Clone, Default)]
pub(crate) struct LemonadeProcessEnvironment {
    rocm_root: Option<PathBuf>,
    path_entries: Vec<PathBuf>,
    library_entries: Vec<PathBuf>,
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
    spawn_lemond_with_parent(
        manifest,
        host,
        port,
        log_path,
        process_env,
        &ParentRuntimeEnvironment::current(),
    )
}

/// `spawn_lemond` with the parent runtime environment supplied by the caller, so
/// tests need not read or create anything under the real user dirs.
pub(crate) fn spawn_lemond_with_parent(
    manifest: &LemonadeInstallManifest,
    host: &str,
    port: u16,
    log_path: Option<&Path>,
    process_env: &LemonadeProcessEnvironment,
    parent: &ParentRuntimeEnvironment,
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
    let vars = lemonade_process_environment_vars(process_env, parent)?;
    apply_environment_vars(&mut command, &vars);
    if let Some(log_path) = log_path {
        if let Some(log_dir) = log_path.parent() {
            fs::create_dir_all(log_dir)?;
        }
        let log = open_service_log(log_path)?;
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

/// Start a fresh service log and open it in append mode. The child writes
/// through this handle while this process appends its own lines (for example the
/// non-ROCm backend notice) through another; without `O_APPEND` the child's next
/// write would land at its own offset, on top of what was appended.
fn open_service_log(log_path: &Path) -> Result<fs::File> {
    fs::File::create(log_path)
        .with_context(|| format!("failed to create {}", log_path.display()))?;
    fs::OpenOptions::new()
        .append(true)
        .open(log_path)
        .with_context(|| format!("failed to open {}", log_path.display()))
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
    #[cfg(test)]
    pub(crate) fn for_test(success: bool, description: &str) -> Self {
        Self {
            success,
            description: description.to_owned(),
        }
    }

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

/// Apply the Lemonade process environment to `command`, with the parent runtime
/// environment supplied by the caller so a test can confine what is read and created.
pub(crate) fn apply_lemonade_process_environment_with_parent(
    command: &mut ProcessCommand,
    env: &LemonadeProcessEnvironment,
    parent: &ParentRuntimeEnvironment,
) -> Result<()> {
    let vars = lemonade_process_environment_vars(env, parent)?;
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
    wait_for_lemonade_cli_status_with_parent(
        manifest,
        host,
        port,
        timeout,
        log_path,
        process_env,
        &ParentRuntimeEnvironment::current(),
    )
}

/// `wait_for_lemonade_cli_status` with the parent runtime environment supplied by
/// the caller, so tests need not read or create anything under the real user dirs.
pub(crate) fn wait_for_lemonade_cli_status_with_parent(
    manifest: &LemonadeInstallManifest,
    host: &str,
    port: u16,
    timeout: Duration,
    log_path: Option<&Path>,
    process_env: &LemonadeProcessEnvironment,
    parent: &ParentRuntimeEnvironment,
) -> Result<()> {
    let start = std::time::Instant::now();
    let mut last_status = None;
    // Resolved once: the poll runs twice a second, and resolving also prepares
    // the child's runtime directory on disk.
    let env_vars = lemonade_process_environment_vars(process_env, parent)?;
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
    match rocm_core::tail_lines(log_path, limit, MAX_TAIL_READ) {
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

/// Install a llama.cpp backend through the `lemonade` CLI. The parent runtime
/// environment is supplied by the caller, so a test can confine what is read and
/// created.
pub(crate) fn run_lemonade_backend_install(
    manifest: &LemonadeInstallManifest,
    host: &str,
    port: u16,
    backend: &str,
    process_env: &LemonadeProcessEnvironment,
    parent: &ParentRuntimeEnvironment,
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
    apply_lemonade_process_environment_with_parent(&mut command, process_env, parent)?;
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
/// The parent runtime environment is supplied by the caller (production passes
/// `ParentRuntimeEnvironment::current()`), so tests need not read or create anything
/// under the real user dirs.
pub(crate) fn run_lemonade_pull_with_parent(
    manifest: &LemonadeInstallManifest,
    host: &str,
    port: u16,
    checkpoint_ref: &str,
    log_path: Option<&Path>,
    process_env: &LemonadeProcessEnvironment,
    parent: &ParentRuntimeEnvironment,
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
    let vars = lemonade_process_environment_vars(process_env, parent)?;
    apply_environment_vars(&mut command, &vars);
    hide_child_console_window(&mut command);
    let status = command
        .status()
        .with_context(|| format!("failed to run {}", manifest.lemonade.display()))?;
    if !status.success() {
        bail!("Lemonade pull of `{checkpoint_ref}` failed with status {status}");
    }
    Ok(())
}

/// Load a model through the Lemonade CLI. The parent runtime environment is supplied
/// by the caller (production passes `ParentRuntimeEnvironment::current()`), so tests
/// need not read or create anything under the real user dirs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_lemonade_model_load_with_parent(
    manifest: &LemonadeInstallManifest,
    host: &str,
    port: u16,
    model_ref: &str,
    backend: &str,
    engine_recipe: Option<&EngineRecipeHint>,
    log_path: Option<&Path>,
    process_env: &LemonadeProcessEnvironment,
    parent: &ParentRuntimeEnvironment,
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
    let vars = lemonade_process_environment_vars(process_env, parent)?;
    apply_environment_vars(&mut command, &vars);
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
        crate::state::announce_non_rocm_backend(self.backend, self.log_path);
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

    #[cfg(target_os = "linux")]
    #[test]
    fn serving_gpu_model_load_process_error_is_returned() {
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        let executable = dir.join("lemonade");
        let args_path = dir.join("args.txt");
        rocm_core::fake_executable::write_fake_executable(&executable, "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$(dirname \"$0\")/args.txt\"\nprintf 'GPU backend initialization failed\\n' >&2\nexit 23\n").unwrap();
        let manifest = LemonadeInstallManifest {
            env_id: "test".to_owned(),
            version: "test".to_owned(),
            runtime_dir: dir.join("runtime"),
            lemond: dir.join("lemond"),
            lemonade: executable,
            backend_recipe: crate::LLAMACPP_RECIPE.to_owned(),
            backend_name: "vulkan".to_owned(),
            installed_at_unix_ms: 0,
        };
        let log = dir.join("load.log");
        let parent = ParentRuntimeEnvironment::rooted_at(dir);
        let error = run_lemonade_model_load_with_parent(
            &manifest,
            "127.0.0.1",
            11435,
            "Qwen3-0.6B-GGUF",
            "vulkan",
            None,
            Some(&log),
            &LemonadeProcessEnvironment::default(),
            &parent,
        )
        .unwrap_err();
        assert!(error.to_string().contains("Lemonade model load failed"));
        assert!(error.to_string().contains("23"));
        assert!(
            fs::read_to_string(log)
                .unwrap()
                .contains("GPU backend initialization failed")
        );
        let args = fs::read_to_string(args_path).unwrap();
        assert!(args.contains("load\nQwen3-0.6B-GGUF\n--llamacpp\nvulkan\n"));
        assert!(!args.contains("cpu"));
    }

    #[cfg(target_os = "linux")]
    fn fake_pull_manifest(dir: &Path, exit_code: i32) -> LemonadeInstallManifest {
        let executable = dir.join("lemonade");
        rocm_core::fake_executable::write_fake_executable(
            &executable,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$(dirname \"$0\")/args.txt\"\nprintf 'pull output line\\n'\nprintf 'pull failure detail\\n' >&2\nexit {exit_code}\n"
            ),
        )
        .unwrap();
        LemonadeInstallManifest {
            env_id: "test".to_owned(),
            version: "test".to_owned(),
            runtime_dir: dir.join("runtime"),
            lemond: dir.join("lemond"),
            lemonade: executable,
            backend_recipe: crate::LLAMACPP_RECIPE.to_owned(),
            backend_name: "vulkan".to_owned(),
            installed_at_unix_ms: 0,
        }
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
            manifest: fake_pull_manifest(&dir, 0),
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

    /// The child's handle and a second appender share one log: a line appended
    /// after the child's first write must survive the child's next write.
    #[test]
    fn a_line_appended_beside_the_childs_log_handle_survives_its_next_write() {
        use std::io::Write as _;
        let dir = crate::test_support::scratch_dir("service-log-append");
        let log_path = dir.join("serve.log");
        let mut child_handle = open_service_log(&log_path).unwrap();
        child_handle.write_all(b"child line one\n").unwrap();
        let notice = crate::state::announce_non_rocm_backend("vulkan", Some(&log_path)).unwrap();
        child_handle.write_all(b"child line two\n").unwrap();
        // The whole file, not a substring: without append the child's second write
        // lands on top of the start of the notice and a `contains` on its tail
        // would still match.
        let log = fs::read_to_string(&log_path).unwrap();
        assert_eq!(
            log,
            format!("child line one\n{notice}\nchild line two\n"),
            "{log}"
        );
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
            manifest: fake_pull_manifest(&dir, 0),
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

    #[cfg(target_os = "linux")]
    #[test]
    fn pull_runs_exact_checkpoint_argv() {
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        let manifest = fake_pull_manifest(dir, 0);
        let log = dir.join("pull.log");
        run_lemonade_pull_with_parent(
            &manifest,
            "127.0.0.1",
            11436,
            "owner/repo:Q4_0",
            Some(&log),
            &LemonadeProcessEnvironment::default(),
            &ParentRuntimeEnvironment::rooted_at(dir),
        )
        .unwrap();
        assert_eq!(
            fs::read_to_string(dir.join("args.txt")).unwrap(),
            "--host\n127.0.0.1\n--port\n11436\npull\nowner/repo:Q4_0\n"
        );
        assert!(
            fs::read_to_string(log)
                .unwrap()
                .contains("pull output line")
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn pull_failure_is_returned_with_checkpoint_status_and_logged_output() {
        let temporary = tempfile::tempdir().unwrap();
        let dir = temporary.path();
        let manifest = fake_pull_manifest(dir, 7);
        let log = dir.join("pull.log");
        let error = run_lemonade_pull_with_parent(
            &manifest,
            "127.0.0.1",
            11436,
            "owner/repo:Q4_0",
            Some(&log),
            &LemonadeProcessEnvironment::default(),
            &ParentRuntimeEnvironment::rooted_at(dir),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("owner/repo:Q4_0"), "{error}");
        assert!(error.contains('7'), "{error}");
        // The process output is streamed to the log rather than embedded in the error.
        assert!(
            fs::read_to_string(log)
                .unwrap()
                .contains("pull failure detail")
        );
    }

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
