// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, ManagedRuntimeEnvironment, RocmCliConfig, active_managed_therock_environment,
    format_http_base_url, http_get_text_with_auth, prepend_runtime_paths, runtime_is_linux,
    runtime_is_windows,
};
use rocm_engine_protocol::EngineRecipeHint;
use serde_json::Value;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::time::Duration;

use crate::device_policy::device_policy_name;
use crate::install::LemonadeInstallManifest;
use crate::runtime_dir::{ParentRuntimeEnvironment, child_runtime_dir_var};
use crate::sdk_root::OfferedSdk;
use crate::state::{MAX_TAIL_READ, parse_http_endpoint, resolve_lemonade_model_ref};
use crate::{LLAMACPP_RECIPE, ServeHttpRequest};

const STARTUP_FAILURE_LOG_TAIL_LINES: usize = 80;

/// Environment variables to set on a child: a name and a value.
#[cfg(any(windows, test))]
type EnvironmentVars = Vec<(&'static str, OsString)>;

#[derive(Debug, Clone, Default)]
pub(crate) struct LemonadeProcessEnvironment {
    /// What `ROCM_PATH` is set to: the managed SDK's root, or a root staged for it
    /// (see [`crate::sdk_root`]).
    rocm_root: Option<PathBuf>,
    /// The managed SDK, when `rocm_root` is a root Lemonade's own probe accepts,
    /// so it can reuse the SDK instead of downloading a ROCm runtime of its own.
    pub(crate) offered_sdk: Option<OfferedSdk>,
    /// Why the managed SDK could not be offered, when it could not: the install
    /// notice says so. `rocm_root` is then the SDK's own root.
    pub(crate) offer_failure: Option<String>,
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
    spawn_lemond_with_parent(
        manifest,
        host,
        port,
        log_path,
        process_env,
        &ParentRuntimeEnvironment::current(),
        LogMode::Fresh,
    )
}

/// How a spawned `lemond` treats the service log it is given.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum LogMode {
    /// Start the log over: this is the serve's own server.
    Fresh,
    /// Keep what is there and append: a helper `lemond` (the one that pulls a
    /// model) shares the log of a server that already wrote the reason it is needed.
    Append,
}

/// `spawn_lemond` with the parent runtime environment supplied by the caller, so
/// tests need not read or create anything under the real user dirs. Every
/// spawn goes through here, so the host-library refusal below covers the
/// router serve and the helper `lemond` too.
pub(crate) fn spawn_lemond_with_parent(
    manifest: &LemonadeInstallManifest,
    host: &str,
    port: u16,
    log_path: Option<&Path>,
    process_env: &LemonadeProcessEnvironment,
    parent: &ParentRuntimeEnvironment,
    log_mode: LogMode,
) -> Result<LemondChild> {
    #[cfg(target_os = "linux")]
    return spawn_lemond_on(
        &crate::host_libraries::probe_host_libraries(),
        crate::host_libraries::host_library_check_disabled(),
        manifest,
        host,
        port,
        log_path,
        process_env,
        parent,
        log_mode,
    );
    #[cfg(not(target_os = "linux"))]
    spawn_lemond_unchecked(
        manifest,
        host,
        port,
        log_path,
        process_env,
        parent,
        log_mode,
    )
}

/// [`spawn_lemond_with_parent`] on Linux, with the host's libraries and the
/// opt-out given so a test can drive the refusal on a host whose libraries are
/// new.
#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn spawn_lemond_on(
    host_libraries: &crate::host_libraries::HostLibraries,
    check_disabled: bool,
    manifest: &LemonadeInstallManifest,
    host: &str,
    port: u16,
    log_path: Option<&Path>,
    process_env: &LemonadeProcessEnvironment,
    parent: &ParentRuntimeEnvironment,
    log_mode: LogMode,
) -> Result<LemondChild> {
    refuse_lemond_on_too_old_host(manifest, host_libraries, check_disabled)?;
    spawn_lemond_unchecked(
        manifest,
        host,
        port,
        log_path,
        process_env,
        parent,
        log_mode,
    )
}

/// Refuse to start the pinned Linux `lemond` on a host whose glibc or OpenSSL
/// is older than it needs (see [`crate::host_libraries`]), rather than let it
/// exit with a loader error that reads as a readiness timeout.
///
/// Applies only to a tree of the pinned version: the recorded needs belong to
/// that archive, and a tree left by an older pin is replaced by the next
/// install, which runs the same check before downloading.
#[cfg(target_os = "linux")]
fn refuse_lemond_on_too_old_host(
    manifest: &LemonadeInstallManifest,
    host_libraries: &crate::host_libraries::HostLibraries,
    check_disabled: bool,
) -> Result<()> {
    if manifest.version != rocm_deps::LEMONADE_VERSION {
        return Ok(());
    }
    crate::host_libraries::refuse_if_host_too_old(
        &manifest.version,
        &crate::install::LINUX_LEMOND_NEEDS,
        host_libraries,
        "Lemonade was not started.",
        check_disabled,
    )
}

#[allow(clippy::too_many_arguments)]
fn spawn_lemond_unchecked(
    manifest: &LemonadeInstallManifest,
    host: &str,
    port: u16,
    log_path: Option<&Path>,
    process_env: &LemonadeProcessEnvironment,
    parent: &ParentRuntimeEnvironment,
    log_mode: LogMode,
) -> Result<LemondChild> {
    #[cfg(windows)]
    if let Some(log_path) = log_path {
        let (args, vars) = windows_logged_start(manifest, host, port, process_env, parent)?;
        return spawn_lemond_logged(&manifest.lemond, &args, &vars, log_path);
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
        let log = open_service_log(log_path, log_mode)?;
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

/// The arguments and the environment a Windows `lemond` start that logs its
/// output runs with. The environment is the one the other start applies:
/// `ROCM_PATH` and the SDK's directories on `PATH` are how `lemond` finds the
/// managed SDK, `HIP_VISIBLE_DEVICES` pins the GPUs, and `LEMONADE_API_KEY` is how
/// it is told to require a key.
#[cfg(any(windows, test))]
fn windows_logged_start(
    manifest: &LemonadeInstallManifest,
    host: &str,
    port: u16,
    process_env: &LemonadeProcessEnvironment,
    parent: &ParentRuntimeEnvironment,
) -> Result<(Vec<String>, EnvironmentVars)> {
    let args = vec![
        child_process_path(&manifest.runtime_dir),
        "--host".to_owned(),
        host.to_owned(),
        "--port".to_owned(),
        port.to_string(),
    ];
    let vars = lemonade_process_environment_vars(process_env, parent)?;
    Ok((args, vars))
}

/// The variables in the form the Windows spawn helpers take them: a name and a
/// value that is a path only in being an `OsStr`. The helper builds the
/// child's environment block from the parent's plus these, replacing a variable
/// of the same name whatever its case.
#[cfg(any(windows, test))]
fn windows_env_overrides<'a>(
    vars: &'a [(&'static str, OsString)],
) -> Vec<(&'static str, &'a Path)> {
    vars.iter()
        .map(|(name, value)| (*name, Path::new(value)))
        .collect()
}

/// Start `program` on Windows with its output going to `log_path` and `vars`
/// in its environment.
#[cfg(windows)]
fn spawn_lemond_logged(
    program: &Path,
    args: &[String],
    vars: &[(&'static str, OsString)],
    log_path: &Path,
) -> Result<LemondChild> {
    let pid = rocm_core::spawn_hidden_console_with_log(
        program,
        args,
        &windows_env_overrides(vars),
        log_path,
    )
    .with_context(|| format!("failed to start {}", program.display()))?;
    Ok(LemondChild::Pid(pid))
}

/// Open the service log in append mode, starting it over first for
/// [`LogMode::Fresh`]. The child writes
/// through this handle while this process appends its own lines (for example the
/// non-ROCm backend notice) through another; without `O_APPEND` the child's next
/// write would land at its own offset, on top of what was appended.
fn open_service_log(log_path: &Path, log_mode: LogMode) -> Result<fs::File> {
    if log_mode == LogMode::Fresh {
        fs::File::create(log_path)
            .with_context(|| format!("failed to create {}", log_path.display()))?;
    }
    fs::OpenOptions::new()
        .append(true)
        .create(true)
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

/// Which of this process's standard output streams are a console.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OutputConsoles {
    pub(crate) stdout: bool,
    pub(crate) stderr: bool,
}

impl OutputConsoles {
    pub(crate) fn detect() -> Self {
        use std::io::IsTerminal;
        Self {
            stdout: std::io::stdout().is_terminal(),
            stderr: std::io::stderr().is_terminal(),
        }
    }

    /// Whether a child that inherits both streams may be started without a
    /// console window.
    ///
    /// `CREATE_NO_WINDOW` starts the child with no console at all, and a child
    /// with no console writes nothing to a console handle it inherited: run
    /// from a real Windows console, `cmd /c echo`, a Python child and
    /// `lemonade.exe --version` all printed their output without the flag and
    /// printed nothing with it. That is how `lemonade backends install`'s
    /// progress for a multi-gigabyte download disappeared from an interactive
    /// `rocm serve` while still showing up in a redirected log. When either
    /// stream is a console this process has one, so the child shares it and no
    /// window opens; when neither is (a pipe, a file, or no console at all) the
    /// flag is still applied, so a console-less parent never flashes a window.
    pub(crate) const fn child_may_hide_its_console(self) -> bool {
        !(self.stdout || self.stderr)
    }
}

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
    Ok(process_environment_for(
        &crate::install::lemonade_root(&paths, None),
        env,
        runtime_is_windows(),
    ))
}

/// The Lemonade process environment for the active managed runtime `env`,
/// offering its SDK to Lemonade as the ROCm runtime to use ([`crate::sdk_root`]).
///
/// A root that cannot be staged is not an error: `ROCM_PATH` is then the SDK's own
/// root, which is what it was before the SDK was offered, and the reason is kept in
/// `offer_failure` for the install notice. Printing it here would repeat it for
/// every environment built, and would claim a download in places (a server start)
/// where none follows.
fn process_environment_for(
    lemonade_root: &Path,
    env: ManagedRuntimeEnvironment,
    windows: bool,
) -> LemonadeProcessEnvironment {
    let mut offer_failure = None;
    let (rocm_root, offered_sdk) = match env.rocm_root {
        None => (None, None),
        Some(sdk_root) => {
            match crate::sdk_root::offer_sdk(
                lemonade_root,
                &sdk_root,
                &env.library_entries,
                windows,
            ) {
                Ok(offer) => (Some(offer.rocm_root), offer.offered),
                Err(error) => {
                    offer_failure = Some(format!("{error:#}"));
                    (Some(sdk_root), None)
                }
            }
        }
    };
    LemonadeProcessEnvironment {
        rocm_root,
        offered_sdk,
        offer_failure,
        path_entries: env.path_entries,
        library_entries: env.library_entries,
        gpu_indices: Vec::new(),
    }
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
pub(crate) fn apply_environment_vars(
    command: &mut ProcessCommand,
    vars: &[(&'static str, OsString)],
) {
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
    let mut command = backend_install_command(
        manifest,
        host,
        port,
        backend,
        process_env,
        parent,
        OutputConsoles::detect(),
        hide_child_console_window,
    )?;
    let status = command
        .status()
        .with_context(|| format!("failed to run {}", manifest.lemonade.display()))?;
    if !status.success() {
        bail!("Lemonade backend install failed with status {status}");
    }
    Ok(())
}

/// The `lemonade backends install` command, ready to run.
///
/// It inherits stdout and stderr: the backend download's progress is the child's
/// own output. `consoles` and `hide_window` are parameters so a test can assert
/// whether the window is hidden for each console state; the one production
/// caller passes the real streams and [`hide_child_console_window`].
#[allow(clippy::too_many_arguments)]
fn backend_install_command(
    manifest: &LemonadeInstallManifest,
    host: &str,
    port: u16,
    backend: &str,
    process_env: &LemonadeProcessEnvironment,
    parent: &ParentRuntimeEnvironment,
    consoles: OutputConsoles,
    hide_window: impl FnOnce(&mut ProcessCommand),
) -> Result<ProcessCommand> {
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
    if consoles.child_may_hide_its_console() {
        hide_window(&mut command);
    }
    Ok(command)
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

/// Whether `/v1/models` advertises `model_ref` as ready to serve on GPU. A stock
/// `llama-server` (direct-serve) entry has no `recipe_options` and is accepted by name,
/// since the direct-serve path only runs GPU backends. A Lemonade-router entry carries
/// `recipe_options`, so its `llamacpp_backend` must match — this keeps a merely
/// registered-but-unloaded model (empty `recipe_options`) from reading as ready.
pub(crate) fn models_payload_has_ready_model(
    value: &Value,
    model_ref: &str,
    backend: &str,
) -> bool {
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

pub(crate) fn lemonade_model_load_args(
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
    if let Some(engine_recipe) = engine_recipe {
        let (context_size, llamacpp_flags) = split_context_size_flag(&engine_recipe.required_flags);
        if let Some(context_size) = context_size {
            args.extend(["--ctx-size".to_owned(), context_size]);
        }
        if !llamacpp_flags.is_empty() {
            args.extend(["--llamacpp-args".to_owned(), llamacpp_flags.join(" ")]);
        }
    }
    args
}

/// Separate a recipe's `--ctx-size N` from the rest of its flags.
///
/// Lemonade owns the context size: `lemonade load` has its own `--ctx-size`,
/// and without it the router auto-tunes the value (it resolved to 4096 on a
/// Vega iGPU for a 4B model, less than the assistant's own prompt). A
/// `--ctx-size` inside `--llamacpp-args` would be a second, conflicting
/// source, so the recipe's value goes to the option Lemonade reads and only
/// the remaining flags are passed through to llama-server. The `--ctx-size=N`
/// spelling is accepted too.
///
/// Only a value that parses as a count is taken, in either spelling. A
/// `--ctx-size` that names anything else is left in place for llama-server to
/// reject rather than guessed at: last in the list with nothing after it,
/// `--ctx-size=` with nothing or nonsense after the `=`, or followed by a token
/// that is not a number. The last is why the next token is inspected and not
/// merely counted — swallowing it would send Lemonade `--ctx-size --temperature`
/// and hand llama-server the orphaned `0.5`. The test is that the token parses
/// as a count, not that it lacks a leading `--`: llama-server's short flags
/// (`-ngl`, `-fa`, `-b`, `-np`) carry a single dash and would otherwise be
/// swallowed by the same mistake in a different spelling. The two spellings are
/// held to the same test deliberately — a `--ctx-size=-1` taken as a value would
/// reach `lemonade load` as one *and* leave
/// [`crate::context_size::recipe_pins_context`] true, so the host sizing would
/// stand down for a window the recipe never actually named. Everything left in
/// place leaves that predicate false, so the sizing still runs.
///
/// `-c`, llama-server's short spelling, is not recognized in either form; no
/// catalog recipe uses it, and validating flag spellings at catalog-load time is
/// the place to close that rather than a second spelling here.
pub(crate) fn split_context_size_flag(flags: &[String]) -> (Option<String>, Vec<&str>) {
    let mut context_size = None;
    let mut rest = Vec::with_capacity(flags.len());
    let mut iter = flags.iter().map(String::as_str).peekable();
    while let Some(flag) = iter.next() {
        if let Some(value) = flag.strip_prefix("--ctx-size=") {
            if value.parse::<u64>().is_ok() {
                context_size = Some(value.to_owned());
            } else {
                rest.push(flag);
            }
        } else if flag == "--ctx-size"
            && iter.peek().is_some_and(|next| next.parse::<u64>().is_ok())
        {
            context_size = iter.next().map(str::to_owned);
        } else {
            rest.push(flag);
        }
    }
    (context_size, rest)
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

pub(crate) fn model_names_match(left: &str, right: &str) -> bool {
    left.eq_ignore_ascii_case(right)
        || resolve_lemonade_model_ref(left).eq_ignore_ascii_case(&resolve_lemonade_model_ref(right))
}

fn lemonade_backend_matches(value: &str, backend: &str) -> bool {
    value
        .trim()
        .to_ascii_lowercase()
        .starts_with(&backend.to_ascii_lowercase())
}

pub(crate) fn serve_http_command_args(request: &ServeHttpRequest) -> Result<Vec<String>> {
    let mut args = vec![
        "serve-http".to_owned(),
        request.service_id.clone(),
        request.model_ref.clone(),
        "--host".to_owned(),
        request.host.clone(),
        "--port".to_owned(),
        request.port.to_string(),
        "--device-policy".to_owned(),
        device_policy_name(&request.device_policy).to_owned(),
        "--state-path".to_owned(),
        request.state_path.display().to_string(),
    ];
    if let Some(runtime_id) = request.runtime_id.as_deref() {
        args.extend(["--runtime-id".to_owned(), runtime_id.to_owned()]);
    }
    if let Some(env_id) = request.env_id.as_deref() {
        args.extend(["--env-id".to_owned(), env_id.to_owned()]);
    }
    if let Some(log_path) = request.log_path.as_ref() {
        args.extend(["--log-path".to_owned(), log_path.display().to_string()]);
    }
    if let Some(csv) = rocm_engine_protocol::gpu_indices_to_csv(&request.gpu_indices) {
        args.extend(["--gpu".to_owned(), csv]);
    }
    if let Some(engine_recipe) = request.engine_recipe.as_ref() {
        args.extend([
            "--engine-recipe-json".to_owned(),
            serde_json::to_string(engine_recipe)
                .context("failed to serialize the engine recipe for serve-http")?,
        ]);
    }
    Ok(args)
}

#[cfg(windows)]
pub(crate) fn spawn_serve_http_background(
    current_exe: &Path,
    serve_args: &[String],
) -> Result<u32> {
    rocm_core::spawn_detached_no_inherit(current_exe, serve_args, &[])
        .context("failed to launch Lemonade serve-http background process")
}

#[cfg(not(windows))]
pub(crate) fn spawn_serve_http_background(
    current_exe: &Path,
    serve_args: &[String],
) -> Result<u32> {
    let child = ProcessCommand::new(current_exe)
        .args(serve_args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("failed to launch Lemonade serve-http background process")?;
    Ok(child.id())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_MODEL;
    use rocm_engine_protocol::DevicePolicy;
    use serde_json::json;

    fn test_manifest() -> LemonadeInstallManifest {
        LemonadeInstallManifest {
            env_id: "lemonade-embeddable-test".to_owned(),
            version: "test".to_owned(),
            runtime_dir: PathBuf::from("rt"),
            lemond: PathBuf::from("rt").join("lemond"),
            lemonade: PathBuf::from("rt").join("lemonade"),
            backend_recipe: "llamacpp".to_owned(),
            backend_name: "rocm".to_owned(),
            installed_at_unix_ms: 0,
        }
    }

    /// A `lemond` stand-in that records it was started, in a fresh directory.
    #[cfg(target_os = "linux")]
    fn marker_lemond(version: &str) -> (tempfile::TempDir, LemonadeInstallManifest, PathBuf) {
        use std::os::unix::fs::PermissionsExt as _;
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("started");
        let lemond = dir.path().join("lemond");
        fs::write(
            &lemond,
            format!("#!/bin/sh\ntouch '{}'\n", marker.display()),
        )
        .unwrap();
        fs::set_permissions(&lemond, fs::Permissions::from_mode(0o755)).unwrap();
        let manifest = LemonadeInstallManifest {
            version: version.to_owned(),
            runtime_dir: dir.path().to_path_buf(),
            lemond,
            ..test_manifest()
        };
        (dir, manifest, marker)
    }

    /// Start the stand-in through `spawn_lemond`'s Linux path, with the host's
    /// libraries given rather than probed.
    ///
    /// A script written a moment ago can fail to exec with `ETXTBSY` when
    /// another test thread forks while the write handle is still open; that,
    /// and only that, is retried.
    #[cfg(target_os = "linux")]
    fn start_on(
        manifest: &LemonadeInstallManifest,
        host_libraries: &crate::host_libraries::HostLibraries,
        check_disabled: bool,
    ) -> Result<()> {
        let mut attempts = 0;
        let parent_root = tempfile::tempdir().unwrap();
        let mut child = loop {
            match spawn_lemond_on(
                host_libraries,
                check_disabled,
                manifest,
                "127.0.0.1",
                1,
                None,
                &LemonadeProcessEnvironment::default(),
                &ParentRuntimeEnvironment::rooted_at(parent_root.path()),
                LogMode::Fresh,
            ) {
                Err(error)
                    if attempts < 50
                        && error.chain().any(|cause| {
                            cause
                                .downcast_ref::<std::io::Error>()
                                .and_then(std::io::Error::raw_os_error)
                                == Some(libc::ETXTBSY)
                        }) =>
                {
                    attempts += 1;
                    std::thread::sleep(Duration::from_millis(10));
                }
                result => break result?,
            }
        };
        child.wait()?;
        Ok(())
    }

    /// Host libraries as a test writes them: a version, or `missing` for a
    /// library the loader cache does not list.
    #[cfg(target_os = "linux")]
    fn libraries(glibc: &str, openssl: &str) -> crate::host_libraries::HostLibraries {
        use crate::host_libraries::{HostLibrary, LibVersion};
        let library = |text: &str| match text {
            "missing" => HostLibrary::Missing,
            text => HostLibrary::Version(LibVersion::parse(text).unwrap()),
        };
        crate::host_libraries::HostLibraries {
            glibc: library(glibc),
            openssl: library(openssl),
        }
    }

    /// On a host older than the pinned `lemond` needs (Ubuntu 20.04: glibc 2.31
    /// and OpenSSL 1.1 only), the refusal names what each short library needs
    /// and what the host has, says `lemond` was not started, and it really was
    /// not: the stand-in never ran.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_pinned_lemond_is_not_started_on_a_host_with_too_old_libraries() {
        let (_dir, manifest, marker) = marker_lemond(rocm_deps::LEMONADE_VERSION);
        let error = start_on(&manifest, &libraries("2.31", "missing"), false)
            .expect_err("a host older than the pinned lemond needs is refused")
            .to_string();
        let needs = &crate::install::LINUX_LEMOND_NEEDS;
        for expected in [
            format!(
                "glibc {} or newer (this host's libc.so.6 provides up to GLIBC_2.31)",
                needs.glibc
            ),
            format!(
                "OpenSSL {} or newer (this host's loader cache has no libssl.so.3)",
                needs.openssl
            ),
            format!("the Lemonade {} server", rocm_deps::LEMONADE_VERSION),
        ] {
            assert!(error.contains(&expected), "missing `{expected}`: {error}");
        }
        assert!(
            error.contains("Lemonade was not started. If this host provides"),
            "{error}"
        );
        assert!(!marker.exists(), "lemond ran despite the refusal");
    }

    /// The other rows: a host that meets the needs starts `lemond`, and so does
    /// a tree of another version, whose needs this pin does not record.
    #[cfg(target_os = "linux")]
    #[test]
    fn lemond_starts_on_a_new_enough_host_and_for_a_tree_of_another_version() {
        let needs = &crate::install::LINUX_LEMOND_NEEDS;
        let (_dir, manifest, marker) = marker_lemond(rocm_deps::LEMONADE_VERSION);
        start_on(&manifest, &libraries(needs.glibc, needs.openssl), false).unwrap();
        assert!(
            marker.exists(),
            "lemond did not start on a host meeting its needs"
        );

        let (_dir, manifest, marker) = marker_lemond("0.0.1");
        start_on(&manifest, &libraries("2.31", "missing"), false).unwrap();
        assert!(
            marker.exists(),
            "a tree of another version was refused by this pin's needs"
        );
    }

    /// With `ROCM_CLI_DISABLE_LEMONADE_HOST_LIBRARY_CHECK` set, the pinned
    /// `lemond` starts on a host the check would refuse, as the refusal's own
    /// advice promises.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_disabled_check_starts_the_pinned_lemond_on_a_host_it_would_refuse() {
        let (_dir, manifest, marker) = marker_lemond(rocm_deps::LEMONADE_VERSION);
        start_on(&manifest, &libraries("2.31", "missing"), true).unwrap();
        assert!(
            marker.exists(),
            "the disabled check still kept lemond from starting"
        );
    }

    /// The Windows spawn helper takes the variables as a separate argument, and
    /// the start that logs `lemond`'s output was passing it none. This is the
    /// conversion of the variable set into that argument: every variable, with
    /// its value, in order. The start itself is covered by
    /// `a_logged_windows_start_is_given_the_lemonade_environment` (the arguments
    /// and variables it passes, Linux) and the Windows-only
    /// `a_logged_lemond_start_on_windows_receives_the_lemonade_environment` (the
    /// spawn). The one line that hands the first to the second, in
    /// `spawn_lemond_unchecked`, compiles only on Windows and has no test of its own.
    #[test]
    fn every_variable_becomes_a_windows_override_with_its_value() {
        let vars = vec![
            ("ROCM_PATH", OsString::from(r"C:\sdk\_rocm_sdk_core")),
            ("PATH", OsString::from(r"C:\sdk\bin;C:\Windows")),
            ("LEMONADE_API_KEY", OsString::from("secret")),
        ];
        let overrides = windows_env_overrides(&vars);
        assert_eq!(
            overrides,
            [
                ("ROCM_PATH", Path::new(r"C:\sdk\_rocm_sdk_core")),
                ("PATH", Path::new(r"C:\sdk\bin;C:\Windows")),
                ("LEMONADE_API_KEY", Path::new("secret")),
            ]
        );
        assert!(windows_env_overrides(&[]).is_empty());
    }

    /// The real Windows spawn, with the real helper: a child started with a log
    /// sees the variables it was given, replacing the parent's of the same name.
    /// The variables are the ones `windows_logged_start` builds, which is what
    /// `spawn_lemond_unchecked` passes. The SDK's directory is a real one,
    /// because the `PATH` the variables carry keeps only entries that exist. Runs
    /// on the Windows CI job, the only place this code is compiled.
    #[cfg(windows)]
    #[test]
    fn a_logged_lemond_start_on_windows_receives_the_lemonade_environment() {
        let dir = crate::test_support::scratch_dir("lemond-logged-env");
        let script = dir.join("emit.bat");
        // A .bat run through cmd with one path argument, as rocm-core's own
        // Windows log test does: `CreateProcessW` cannot run a .bat directly.
        fs::write(
            &script,
            "@echo off\r\necho rocm_path=%ROCM_PATH%\r\necho path=%PATH%\r\n",
        )
        .unwrap();
        let log = dir.join("lemond.log");
        let sdk_bin = dir.join("sdk-bin");
        fs::create_dir_all(&sdk_bin).unwrap();
        let env = LemonadeProcessEnvironment {
            rocm_root: Some(PathBuf::from(r"C:\sdk\_rocm_sdk_core")),
            path_entries: vec![sdk_bin.clone()],
            ..LemonadeProcessEnvironment::default()
        };
        let (_, vars) = windows_logged_start(
            &test_manifest(),
            "127.0.0.1",
            8000,
            &env,
            &ParentRuntimeEnvironment::current(),
        )
        .unwrap();
        let comspec = std::env::var_os("COMSPEC")
            .unwrap_or_else(|| OsString::from(r"C:\Windows\System32\cmd.exe"));
        let mut child = spawn_lemond_logged(
            Path::new(&comspec),
            &["/c".to_owned(), script.to_string_lossy().into_owned()],
            &vars,
            &log,
        )
        .unwrap();
        let status = child.wait().unwrap();
        let output = fs::read_to_string(&log).unwrap();
        fs::remove_dir_all(&dir).ok();
        assert!(status.success(), "{status}: {output}");
        assert!(
            output.contains(r"rocm_path=C:\sdk\_rocm_sdk_core"),
            "ROCM_PATH did not reach the child:\n{output}"
        );
        assert!(
            output
                .to_lowercase()
                .contains(&format!("path={};", sdk_bin.display()).to_lowercase()),
            "the SDK's directory is not first on the child's PATH:\n{output}"
        );
    }

    /// The Windows start that logs `lemond`'s output is given the arguments of
    /// the server and the whole Lemonade environment: `ROCM_PATH`, the SDK's
    /// `PATH` entries, the GPU pinning. It was given none of the variables.
    /// Runs where a confined parent environment exists (Linux); the Windows-only
    /// spawn test below builds the same variables on its own gate.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_logged_windows_start_is_given_the_lemonade_environment() {
        let dir = crate::test_support::scratch_dir("windows-logged-start");
        let sdk_bin = dir.join("bin");
        fs::create_dir_all(&sdk_bin).unwrap();
        let env = LemonadeProcessEnvironment {
            rocm_root: Some(dir.join("_rocm_sdk_core")),
            path_entries: vec![sdk_bin.clone()],
            gpu_indices: vec![1],
            ..LemonadeProcessEnvironment::default()
        };
        let parent = ParentRuntimeEnvironment::rooted_at(&dir);
        let (args, vars) =
            windows_logged_start(&test_manifest(), "127.0.0.1", 8123, &env, &parent).unwrap();
        fs::remove_dir_all(&dir).ok();

        assert_eq!(args, ["rt", "--host", "127.0.0.1", "--port", "8123"]);
        let var = |name: &str| {
            vars.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string_lossy().into_owned())
        };
        assert_eq!(
            var("ROCM_PATH").as_deref(),
            Some(&*dir.join("_rocm_sdk_core").to_string_lossy())
        );
        assert!(
            var("PATH").is_some_and(|path| path.starts_with(&*sdk_bin.to_string_lossy())),
            "{vars:?}"
        );
        assert_eq!(var("HIP_VISIBLE_DEVICES").as_deref(), Some("1"));
    }

    #[cfg(unix)]
    fn managed_environment(
        rocm_root: &Path,
        library_entries: Vec<PathBuf>,
    ) -> ManagedRuntimeEnvironment {
        ManagedRuntimeEnvironment {
            rocm_root: Some(rocm_root.to_path_buf()),
            path_entries: vec![rocm_root.join("bin")],
            library_entries,
        }
    }

    /// With a Linux wheel-layout SDK active, `ROCM_PATH` is the root staged for
    /// it, and that is what Lemonade is told: the environment variables carry
    /// the staged root, the loader path still carries the SDK's libraries, and
    /// the SDK is recorded as offered.
    #[cfg(target_os = "linux")]
    #[test]
    fn rocm_path_is_the_root_staged_for_the_active_sdk() {
        let dir = crate::test_support::scratch_dir("process-env-staged");
        let sdk = dir.join("_rocm_sdk_core");
        fs::create_dir_all(sdk.join("lib")).unwrap();
        fs::create_dir_all(sdk.join(".info")).unwrap();
        fs::write(sdk.join("lib").join("libamdhip64.so.7"), b"hip").unwrap();
        fs::write(sdk.join(".info").join("version"), "10.1.0\n").unwrap();
        let lemonade_root = dir.join("lemonade");

        let env = process_environment_for(
            &lemonade_root,
            managed_environment(&sdk, vec![sdk.join("lib")]),
            false,
        );
        let staged = env.rocm_root.clone().expect("a ROCm root is set");
        assert!(
            staged.starts_with(lemonade_root.join("sdk-root")),
            "{staged:?}"
        );
        assert_eq!(
            env.offered_sdk,
            Some(OfferedSdk {
                version: Some("10.1.0".to_owned())
            })
        );

        let parent = ParentRuntimeEnvironment::rooted_at(&dir);
        let vars = lemonade_process_environment_vars(&env, &parent).unwrap();
        let var = |name: &str| {
            vars.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.clone())
        };
        assert_eq!(var("ROCM_PATH"), Some(staged.into_os_string()));
        let loader_path =
            var("LD_LIBRARY_PATH").expect("the SDK's libraries are on the loader path");
        assert!(
            loader_path
                .to_string_lossy()
                .starts_with(&*sdk.join("lib").to_string_lossy()),
            "{loader_path:?}"
        );
        fs::remove_dir_all(&dir).ok();
    }

    /// A root that cannot be staged falls back to the SDK's own, which is what
    /// `ROCM_PATH` was before the SDK was offered, and the SDK is not recorded
    /// as offered: the install notice must not claim a reuse that will not
    /// happen.
    #[cfg(unix)]
    #[test]
    fn a_root_that_cannot_be_staged_leaves_rocm_path_at_the_sdk() {
        let dir = crate::test_support::scratch_dir("process-env-unstaged");
        let sdk = dir.join("_rocm_sdk_core");
        fs::create_dir_all(sdk.join("lib")).unwrap();
        fs::write(sdk.join("lib").join("libamdhip64.so.7"), b"hip").unwrap();
        // The folder that would hold the staged root is a file.
        let lemonade_root = dir.join("lemonade");
        fs::write(&lemonade_root, b"not a folder").unwrap();
        fs::create_dir_all(sdk.join(".info")).unwrap();
        fs::write(sdk.join(".info").join("version"), "10.1.0\n").unwrap();

        let env = process_environment_for(&lemonade_root, managed_environment(&sdk, vec![]), false);
        assert_eq!(env.rocm_root, Some(sdk));
        assert_eq!(env.offered_sdk, None);
        // The reason is kept for the install notice, which prints it (see
        // `the_notice_says_when_the_active_sdk_could_not_be_offered` in
        // `footprint.rs`); nothing is printed here, where it would repeat for
        // every environment built.
        assert!(
            env.offer_failure
                .as_deref()
                .is_some_and(|reason| !reason.is_empty()),
            "{:?}",
            env.offer_failure
        );
        fs::remove_dir_all(&dir).ok();
    }

    /// No managed runtime means no `ROCM_PATH` and nothing offered.
    #[test]
    fn no_managed_runtime_offers_nothing() {
        let env = process_environment_for(
            Path::new("unused"),
            ManagedRuntimeEnvironment::default(),
            false,
        );
        assert_eq!(env.rocm_root, None);
        assert_eq!(env.offered_sdk, None);
    }

    /// Build the real backend-install command for a console state and report
    /// whether the window was hidden, along with the command itself.
    fn backend_install_for(consoles: OutputConsoles) -> (ProcessCommand, bool) {
        let mut hidden = false;
        let root = crate::test_support::scratch_dir("backend-install-command");
        // Confined to a scratch root where a confined parent exists (Linux), so
        // building the command creates nothing in the real runtime directory.
        #[cfg(target_os = "linux")]
        let parent = ParentRuntimeEnvironment::rooted_at(&root);
        #[cfg(not(target_os = "linux"))]
        let parent = ParentRuntimeEnvironment::current();
        let command = backend_install_command(
            &test_manifest(),
            "127.0.0.1",
            8123,
            "rocm",
            &LemonadeProcessEnvironment::default(),
            &parent,
            consoles,
            |_| hidden = true,
        )
        .expect("the command builds");
        fs::remove_dir_all(&root).ok();
        (command, hidden)
    }

    /// The backend install's progress is the child's own output. Hiding its
    /// console while that output is a console discards it (see
    /// `OutputConsoles::child_may_hide_its_console`), so the window is hidden
    /// only when neither stream is one. The command is the real one: a dropped
    /// call to the hide step, a hide step run unconditionally, or a console
    /// check that ignores one of the streams each turn a case below red.
    #[test]
    fn the_backend_install_hides_its_window_only_when_no_output_is_a_console() {
        for (stdout, stderr, expect_hidden) in [
            (true, true, false),
            (true, false, false),
            (false, true, false),
            (false, false, true),
        ] {
            let (command, hidden) = backend_install_for(OutputConsoles { stdout, stderr });
            assert_eq!(
                hidden, expect_hidden,
                "stdout console={stdout}, stderr console={stderr}"
            );
            assert_eq!(command.get_program(), PathBuf::from("rt").join("lemonade"));
            let args: Vec<_> = command
                .get_args()
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect();
            assert_eq!(
                args,
                [
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8123",
                    "backends",
                    "install",
                    "llamacpp:rocm"
                ]
            );
        }
    }

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

    /// The child's handle and a second appender share one log: a line appended
    /// after the child's first write must survive the child's next write.
    #[test]
    fn a_line_appended_beside_the_childs_log_handle_survives_its_next_write() {
        use std::io::Write as _;
        let dir = crate::test_support::scratch_dir("service-log-append");
        let log_path = dir.join("serve.log");
        let mut child_handle = open_service_log(&log_path, LogMode::Fresh).unwrap();
        child_handle.write_all(b"child line one\n").unwrap();
        let notice =
            crate::device_policy::announce_non_rocm_backend("vulkan", Some(&log_path)).unwrap();
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

    /// A helper `lemond` appends to the log of the server whose failure made it
    /// necessary; only the serve's own server starts the log over.
    #[test]
    fn a_helper_server_keeps_the_log_it_is_given() {
        use std::io::Write as _;
        let dir = crate::test_support::scratch_dir("service-log-modes");
        let log_path = dir.join("serve.log");
        fs::write(
            &log_path,
            "router failed to load the model
",
        )
        .unwrap();
        let mut helper = open_service_log(&log_path, LogMode::Append).unwrap();
        helper
            .write_all(
                b"helper line
",
            )
            .unwrap();
        assert_eq!(
            fs::read_to_string(&log_path).unwrap(),
            "router failed to load the model
helper line
"
        );
        drop(open_service_log(&log_path, LogMode::Fresh).unwrap());
        assert_eq!(fs::read_to_string(&log_path).unwrap(), "");
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
    fn lemonade_model_load_routes_recipe_context_size_to_its_own_option() {
        let recipe = EngineRecipeHint {
            required_flags: vec![
                "--temperature".to_owned(),
                "0.5".to_owned(),
                "--ctx-size".to_owned(),
                "8192".to_owned(),
            ],
            ..EngineRecipeHint::default()
        };
        let args =
            lemonade_model_load_args("127.0.0.1", 11435, DEFAULT_MODEL, "vulkan", Some(&recipe));
        assert_eq!(
            args[args.len() - 4..],
            ["--ctx-size", "8192", "--llamacpp-args", "--temperature 0.5"]
        );
        assert!(
            !args.iter().any(|arg| arg.contains("--ctx-size 8192")),
            "the context size must not also reach llama-server: {args:?}"
        );

        let only_ctx = EngineRecipeHint {
            required_flags: vec!["--ctx-size=4096".to_owned()],
            ..EngineRecipeHint::default()
        };
        let args =
            lemonade_model_load_args("127.0.0.1", 11435, DEFAULT_MODEL, "vulkan", Some(&only_ctx));
        assert_eq!(args[args.len() - 2..], ["--ctx-size", "4096"]);
        assert!(!args.contains(&"--llamacpp-args".to_owned()));
    }

    #[test]
    fn split_context_size_flag_leaves_a_valueless_flag_for_llama_server() {
        let owned = |list: &[&str]| {
            list.iter()
                .map(|flag| (*flag).to_owned())
                .collect::<Vec<String>>()
        };

        // Last in the list, with nothing after it.
        let flags = owned(&["--ctx-size"]);
        let (context_size, rest) = split_context_size_flag(&flags);
        assert_eq!(context_size, None);
        assert_eq!(rest, ["--ctx-size"]);

        // Followed by another flag rather than a number. Counting the next token
        // instead of looking at it would send Lemonade `--ctx-size --temperature`
        // and hand llama-server the orphaned `0.5`.
        let flags = owned(&["--ctx-size", "--temperature", "0.5"]);
        let (context_size, rest) = split_context_size_flag(&flags);
        assert_eq!(context_size, None);
        assert_eq!(rest, ["--ctx-size", "--temperature", "0.5"]);

        // Followed by a *single*-dash llama-server flag. Excluding only `--`
        // would consume `-ngl` as the window and orphan its `999`, which is the
        // `--temperature` bug above wearing llama-server's short spelling.
        let flags = owned(&["--ctx-size", "-ngl", "999"]);
        let (context_size, rest) = split_context_size_flag(&flags);
        assert_eq!(context_size, None);
        assert_eq!(rest, ["--ctx-size", "-ngl", "999"]);

        // A negative number is not a count either, and llama-server owns the
        // refusal.
        let flags = owned(&["--ctx-size", "-1"]);
        let (context_size, rest) = split_context_size_flag(&flags);
        assert_eq!(context_size, None);
        assert_eq!(rest, ["--ctx-size", "-1"]);

        // `--ctx-size=` names no value, and must not become `--ctx-size ""`.
        let flags = owned(&["--ctx-size="]);
        let (context_size, rest) = split_context_size_flag(&flags);
        assert_eq!(context_size, None);
        assert_eq!(rest, ["--ctx-size="]);

        // The `=` spelling is held to the same test as the bare one. Taking
        // these would both forward nonsense to `lemonade load` and leave
        // `recipe_pins_context` true, standing the host sizing down for a window
        // the recipe never named.
        for flag in ["--ctx-size=abc", "--ctx-size=-1", "--ctx-size=8192x"] {
            let flags = owned(&[flag]);
            let (context_size, rest) = split_context_size_flag(&flags);
            assert_eq!(context_size, None, "{flag}");
            assert_eq!(rest, [flag], "{flag}");
        }

        // A real value is still taken, in either spelling, and does not reach
        // llama-server a second time.
        let flags = owned(&["--ctx-size", "8192"]);
        let (context_size, rest) = split_context_size_flag(&flags);
        assert_eq!(context_size, Some("8192".to_owned()));
        assert!(rest.is_empty(), "{rest:?}");

        let flags = owned(&["--temperature", "0.5", "--ctx-size=8192"]);
        let (context_size, rest) = split_context_size_flag(&flags);
        assert_eq!(context_size, Some("8192".to_owned()));
        assert_eq!(rest, ["--temperature", "0.5"]);
    }

    /// The valueless spellings must not reach `lemonade load` as a `--ctx-size`
    /// with a flag or an empty string after it.
    ///
    /// The last case carries a *usable* window alongside other flags, so the
    /// whole test turns on the split actually happening: pass everything through
    /// untouched — which is what this function did before the split existed, and
    /// what satisfies the three valueless cases on its own — and that case fails.
    #[test]
    fn a_valueless_context_size_is_passed_through_to_llama_server_untouched() {
        // A real value is extracted instead of passed through, so "do nothing"
        // is not an answer that satisfies this test.
        let pinned = EngineRecipeHint {
            required_flags: vec![
                "--temperature".to_owned(),
                "0.5".to_owned(),
                "--ctx-size".to_owned(),
                "8192".to_owned(),
            ],
            ..EngineRecipeHint::default()
        };
        let args =
            lemonade_model_load_args("127.0.0.1", 11435, DEFAULT_MODEL, "vulkan", Some(&pinned));
        assert_eq!(
            args[args.len() - 5..],
            [
                "--save-options".to_owned(),
                "--ctx-size".to_owned(),
                "8192".to_owned(),
                "--llamacpp-args".to_owned(),
                "--temperature 0.5".to_owned()
            ],
            "{args:?}"
        );

        for flags in [
            vec!["--ctx-size".to_owned()],
            vec![
                "--ctx-size".to_owned(),
                "--temperature".to_owned(),
                "0.5".to_owned(),
            ],
            vec!["--ctx-size=".to_owned()],
        ] {
            let recipe = EngineRecipeHint {
                required_flags: flags.clone(),
                ..EngineRecipeHint::default()
            };
            let args = lemonade_model_load_args(
                "127.0.0.1",
                11435,
                DEFAULT_MODEL,
                "vulkan",
                Some(&recipe),
            );
            // The whole recipe is passed through to llama-server and nothing is
            // inserted between `--save-options` and it, so Lemonade is given no
            // `--ctx-size` of its own. (Looking for the string alone would not
            // say this: the pass-through *value* is `--ctx-size ...` here.)
            assert_eq!(
                args[args.len() - 3..],
                [
                    "--save-options".to_owned(),
                    "--llamacpp-args".to_owned(),
                    flags.join(" ")
                ],
                "{flags:?}"
            );
        }
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

    #[test]
    fn serve_http_args_preserve_runtime_selection() {
        let request = ServeHttpRequest {
            service_id: "svc".to_owned(),
            model_ref: DEFAULT_MODEL.to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 11435,
            device_policy: DevicePolicy::GpuRequired,
            gpu_indices: Vec::new(),
            runtime_id: Some("runtime".to_owned()),
            env_id: Some("env".to_owned()),
            state_path: PathBuf::from("state.json"),
            log_path: Some(PathBuf::from("service.log")),
            engine_recipe: None,
        };
        let args = serve_http_command_args(&request).expect("the request serializes");
        assert!(args.contains(&"--runtime-id".to_owned()));
        assert!(args.contains(&"runtime".to_owned()));
        assert!(args.contains(&"--env-id".to_owned()));
        assert!(args.contains(&"env".to_owned()));
        assert!(!args.iter().any(|arg| arg == "cpu"));
    }
}
