// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use crate::therock;
use anyhow::{Context, Result, bail};
use process::{comfyui_run_state_cli_label, endpoint_is_reachable, evaluate_running_state};
use rocm_core::{
    AppPaths, RocmCliConfig, format_http_base_url, runtime_is_windows,
    runtime_path_for_windows_child, unix_time_millis,
};
use runtime_env::apply_runtime_environment;
use serde::{Deserialize, Serialize};
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::time::SystemTime;

mod install;
mod process;
mod runtime_env;
mod source_swap;

pub(crate) use install::install;
pub(crate) use process::{start, stop};

const APP_ID: &str = "comfyui";
const APP_NAME: &str = "ComfyUI";
const COMFYUI_SOURCE_ARCHIVE_URL: &str =
    "https://github.com/comfyanonymous/ComfyUI/archive/refs/heads/master.tar.gz";
const COMFYUI_SOURCE_ARCHIVE_NAME: &str = "ComfyUI-master.tar.gz";
const COMFYUI_DEFAULT_HOST: &str = "127.0.0.1";
const COMFYUI_DEFAULT_PORT: u16 = 8188;

/// The ComfyUI source archive URL, overridable only in `e2e-test-hooks`
/// builds so a fixture server can exercise the real download path.
#[cfg(feature = "e2e-test-hooks")]
fn comfyui_source_archive_url() -> String {
    std::env::var("ROCM_CLI_COMFYUI_SOURCE_ARCHIVE_URL_OVERRIDE")
        .unwrap_or_else(|_| COMFYUI_SOURCE_ARCHIVE_URL.to_owned())
}
#[cfg(not(feature = "e2e-test-hooks"))]
fn comfyui_source_archive_url() -> String {
    COMFYUI_SOURCE_ARCHIVE_URL.to_owned()
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct ComfyUiInstallOptions {
    pub runtime_id: Option<String>,
    pub reinstall: bool,
    pub dry_run: bool,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct ComfyUiStartOptions {
    pub host: String,
    pub port: u16,
    pub no_open_browser: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComfyUiManifest {
    app_id: String,
    runtime_key: String,
    runtime_id: String,
    runtime_version: String,
    runtime_root: PathBuf,
    python_executable: PathBuf,
    source_url: String,
    source_path: PathBuf,
    requirements_path: PathBuf,
    pip_cache_dir: Option<PathBuf>,
    log_path: PathBuf,
    torch_version: Option<String>,
    torch_cuda_available: bool,
    installed_at_unix_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ComfyUiState {
    app_id: String,
    url: String,
    host: String,
    port: u16,
    pid: u32,
    source_path: PathBuf,
    python_executable: PathBuf,
    log_path: PathBuf,
    started_at_unix_ms: u128,
}

#[derive(Debug, Clone, Deserialize)]
struct ComfyUiProbe {
    torch_version: Option<String>,
    torch_cuda_available: bool,
    device_count: i64,
    devices: Vec<String>,
}

#[derive(Debug, Clone)]
struct SelectedRuntime {
    manifest: therock::InstalledRuntimeManifest,
    python: PathBuf,
}

#[derive(Debug, Clone, Default)]
struct ComfyUiRuntimeEnvironment {
    venv_root: Option<PathBuf>,
    rocm_root: Option<PathBuf>,
    path_entries: Vec<PathBuf>,
    library_entries: Vec<PathBuf>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum ComfyUiRunState {
    Running,
    Starting,
    Stopped,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct ComfyUiRunReport {
    state: ComfyUiRunState,
    process_running: bool,
    endpoint_reachable: bool,
}

pub(crate) const fn default_host() -> &'static str {
    COMFYUI_DEFAULT_HOST
}

pub(crate) const fn default_port() -> u16 {
    COMFYUI_DEFAULT_PORT
}

/// The facts the chat's ComfyUI status answer is built from, read the way
/// [`render_status`] reads them rather than scraped from its text.
struct StatusSummary {
    installed: bool,
    /// The URL of a ComfyUI that answers, whether or not rocm-cli started it.
    running_url: Option<String>,
    starting: bool,
    /// What `status` and `start` say about a reinstall that stopped part-way.
    interrupted_reinstall: Option<String>,
}

fn status_summary(paths: &AppPaths) -> Result<StatusSummary> {
    let manifest = load_manifest(paths)?;
    let interrupted_reinstall = manifest
        .as_ref()
        .filter(|manifest| source_swap::swap_interrupted(&manifest.source_path))
        .map(|manifest| interrupted_reinstall_advice(&manifest.runtime_key));
    let (running_url, starting) = match load_state(paths)? {
        Some(state) => match evaluate_running_state(&state).state {
            ComfyUiRunState::Running => (Some(state.url), false),
            ComfyUiRunState::Starting => (None, true),
            ComfyUiRunState::Stopped => (None, false),
        },
        None => (default_unmanaged_running_url(), false),
    };
    Ok(StatusSummary {
        installed: manifest.is_some(),
        running_url,
        starting,
        interrupted_reinstall,
    })
}

/// The chat's answer to "what is ComfyUI's status". It advises `start` only
/// when nothing is running or starting and `start` would not refuse; it passes
/// on the interrupted-reinstall note, which names the command to run instead.
pub(crate) fn render_chat_status_answer(paths: &AppPaths) -> Result<String> {
    let summary = status_summary(paths)?;
    let mut output = String::new();
    writeln!(output, "ComfyUI status")?;
    writeln!(output)?;
    if summary.installed {
        writeln!(output, "ComfyUI: installed")?;
    } else {
        writeln!(output, "ComfyUI: not installed yet")?;
    }
    if let Some(url) = &summary.running_url {
        writeln!(output, "Running: yes")?;
        writeln!(output, "URL: {url}")?;
    } else if summary.starting {
        writeln!(output, "Running: starting")?;
    } else {
        writeln!(output, "Running: no")?;
    }
    if let Some(note) = &summary.interrupted_reinstall {
        writeln!(output, "Note: {note}.")?;
    }
    if !summary.installed {
        writeln!(
            output,
            "To install it, ask `can you setup ComfyUI for me` or run `rocm comfyui install`."
        )?;
    } else if summary.running_url.is_none()
        && !summary.starting
        && summary.interrupted_reinstall.is_none()
    {
        writeln!(
            output,
            "To open it, ask `can you start ComfyUI` or run `rocm comfyui start`."
        )?;
    }
    writeln!(output)?;
    writeln!(output, "Nothing was changed.")?;
    Ok(output)
}

pub(crate) fn render_status(paths: &AppPaths, config: &RocmCliConfig) -> Result<String> {
    let mut output = String::new();
    writeln!(output, "{APP_NAME}")?;
    writeln!(output)?;

    let mut interrupted = false;
    if let Some(manifest) = load_manifest(paths)? {
        writeln!(output, "  installed: yes")?;
        writeln!(
            output,
            "  ROCm runtime: {}",
            therock::runtime_version_display(&manifest.runtime_version)
        )?;
        writeln!(output, "  folder: {}", manifest.source_path.display())?;
        writeln!(
            output,
            "  models path: {}",
            models_folder_for_manifest(&manifest).display()
        )?;
        writeln!(
            output,
            "  AMD GPU check: {}",
            if manifest.torch_cuda_available {
                "ready"
            } else {
                "failed"
            }
        )?;
        if source_swap::swap_interrupted(&manifest.source_path) {
            interrupted = true;
            writeln!(
                output,
                "  note: {}",
                interrupted_reinstall_advice(&manifest.runtime_key)
            )?;
        }
    } else {
        writeln!(output, "  installed: no")?;
        writeln!(output, "  next step: rocm comfyui install")?;
    }

    if let Some(state) = load_state(paths)? {
        let run_report = evaluate_running_state(&state);
        writeln!(output)?;
        writeln!(output, "Running")?;
        writeln!(
            output,
            "  status: {}",
            comfyui_run_state_cli_label(run_report.state)
        )?;
        writeln!(output, "  url: {}", state.url)?;
        match run_report.state {
            ComfyUiRunState::Running => {}
            ComfyUiRunState::Starting => {
                writeln!(output, "  note: starting")?;
            }
            // `start` refuses an interrupted reinstall; the note above names
            // the command to run first.
            ComfyUiRunState::Stopped if !interrupted => {
                writeln!(output, "  next step: rocm comfyui start")?;
            }
            ComfyUiRunState::Stopped => {}
        }
    } else {
        writeln!(output)?;
        writeln!(output, "Running")?;
        if let Some(url) = default_unmanaged_running_url() {
            writeln!(output, "  status: running outside rocm-cli")?;
            writeln!(output, "  url: {url}")?;
            writeln!(
                output,
                "  note: ROCm CLI did not start this ComfyUI process"
            )?;
        } else {
            writeln!(output, "  status: not started by rocm-cli")?;
        }
    }

    let runtimes = therock::load_runtime_manifests(paths)?;
    if runtimes.is_empty() {
        writeln!(output)?;
        writeln!(output, "ROCm")?;
        writeln!(
            output,
            "  Install ROCm first: run `rocm` and pick \"Set up this system\"."
        )?;
    } else if let Some(active) = config.active_runtime_key.as_deref() {
        writeln!(output)?;
        writeln!(output, "Default ROCm runtime")?;
        writeln!(output, "  {active}")?;
    }

    Ok(output)
}

pub(crate) fn render_logs(paths: &AppPaths, line_limit: usize) -> Result<String> {
    render_logs_with_options(paths, line_limit, true)
}

pub(crate) fn models_folder(paths: &AppPaths) -> Result<Option<PathBuf>> {
    Ok(load_manifest(paths)?.map(|manifest| models_folder_for_manifest(&manifest)))
}

pub(crate) fn render_models_path(paths: &AppPaths) -> Result<String> {
    let Some(path) = models_folder(paths)? else {
        bail!("ComfyUI is not installed yet. Run `rocm comfyui install` first.");
    };
    Ok(format!("{}\n", path.display()))
}

fn default_unmanaged_running_url() -> Option<String> {
    unmanaged_running_url(COMFYUI_DEFAULT_HOST, COMFYUI_DEFAULT_PORT)
}

fn unmanaged_running_url(host: &str, port: u16) -> Option<String> {
    endpoint_is_reachable(host, port).then(|| format_http_base_url(host, port))
}

fn render_logs_with_options(
    paths: &AppPaths,
    line_limit: usize,
    show_file_locations: bool,
) -> Result<String> {
    let limit = line_limit.clamp(1, 400);
    let logs = recent_log_paths(paths)?;
    let mut output = String::new();
    writeln!(output, "{APP_NAME} logs")?;
    writeln!(output)?;
    if logs.is_empty() {
        writeln!(output, "No ComfyUI logs yet.")?;
        writeln!(output)?;
        writeln!(output, "Start with: rocm comfyui install")?;
        return Ok(output);
    }

    for (index, path) in logs.iter().take(4).enumerate() {
        if index > 0 {
            writeln!(output)?;
        }
        writeln!(output, "{}", comfyui_log_title(path))?;
        if show_file_locations {
            writeln!(output, "  saved file: {}", path.display())?;
        }
        let lines = read_tail_lines(path, limit)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if lines.is_empty() {
            writeln!(output, "  The saved log is empty.")?;
        } else {
            writeln!(output, "  latest output:")?;
            for line in lines {
                writeln!(output, "    {line}")?;
            }
        }
    }
    Ok(output)
}

/// What `start` and `status` say about a reinstall that stopped part-way. The
/// command names the runtime the install belongs to, so it finishes that
/// install rather than one in whichever runtime is the default now; an install
/// started there finds the swap's marker and completes it.
fn interrupted_reinstall_advice(runtime_key: &str) -> String {
    format!(
        "a reinstall was interrupted; run `{}` to finish it",
        finishing_install_command(runtime_key)
    )
}

/// What a failure after the swap says: the new code is in place, what it
/// changed (`changes`, already rendered with its leading space, or empty), and
/// the install that finishes it.
fn unfinished_install_advice(changes: &str, runtime_key: &str) -> String {
    format!(
        "the new ComfyUI code is in place{changes}, but the install did not finish; once the \
         cause below is fixed, run `{}` to finish it",
        finishing_install_command(runtime_key)
    )
}

/// The `rocm comfyui install` line both pieces of advice above tell the user
/// to run. The key is quoted like every other value in a line the CLI offers
/// to be copied (see [`crate::shell_quote`]), so the line names exactly the
/// argv that finishes the install.
fn finishing_install_command(runtime_key: &str) -> String {
    format!(
        "rocm comfyui install --runtime-id {}",
        crate::shell_quote::shell_quote(runtime_key)
    )
}

fn app_root(paths: &AppPaths) -> PathBuf {
    paths.data_dir.join("apps").join(APP_ID)
}

#[cfg(test)]
fn source_path(paths: &AppPaths) -> PathBuf {
    source_path_from_app_root(&app_root(paths))
}

fn runtime_app_root(runtime: &therock::InstalledRuntimeManifest) -> PathBuf {
    runtime.install_root.join("apps").join(APP_ID)
}

fn source_path_from_app_root(app_root: &Path) -> PathBuf {
    app_root.join("source")
}

fn models_folder_for_source(source_path: &Path) -> PathBuf {
    source_path.join("models")
}

fn models_folder_for_manifest(manifest: &ComfyUiManifest) -> PathBuf {
    models_folder_for_source(&manifest.source_path)
}

fn manifest_path(paths: &AppPaths) -> PathBuf {
    app_root(paths).join("manifests").join("current.json")
}

fn state_path(paths: &AppPaths) -> PathBuf {
    app_root(paths).join("state").join("running.json")
}

fn install_log_path_from_app_root(app_root: &Path) -> PathBuf {
    app_root
        .join("logs")
        .join(format!("install-{}.log", unix_time_millis()))
}

fn start_log_path_for_manifest(manifest: &ComfyUiManifest) -> PathBuf {
    manifest
        .source_path
        .parent()
        .map_or_else(
            || manifest.runtime_root.join("apps").join(APP_ID),
            Path::to_path_buf,
        )
        .join("logs")
        .join(format!("start-{}.log", unix_time_millis()))
}

fn recent_log_paths(paths: &AppPaths) -> Result<Vec<PathBuf>> {
    let mut paths_seen = Vec::<PathBuf>::new();
    if let Some(manifest) = load_manifest(paths)? {
        push_log_path(&mut paths_seen, manifest.log_path);
    }
    if let Some(state) = load_state(paths)? {
        push_log_path(&mut paths_seen, state.log_path);
    }
    let logs_root = app_root(paths).join("logs");
    if logs_root.is_dir() {
        for entry in fs::read_dir(&logs_root)
            .with_context(|| format!("failed to read {}", logs_root.display()))?
        {
            let path = entry?.path();
            push_log_path(&mut paths_seen, path);
        }
    }
    paths_seen.retain(|path| path.is_file());
    paths_seen.sort_by(|left, right| {
        log_modified_time(right)
            .cmp(&log_modified_time(left))
            .then_with(|| right.cmp(left))
    });
    Ok(paths_seen)
}

fn push_log_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.iter().any(|seen| seen == &path) {
        paths.push(path);
    }
}

fn log_modified_time(path: &Path) -> SystemTime {
    path.metadata()
        .and_then(|metadata| metadata.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

fn comfyui_log_title(path: &Path) -> &'static str {
    let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
        return "Saved log";
    };
    if name.starts_with("install-") {
        "Install log"
    } else if name.starts_with("start-") {
        "Run log"
    } else {
        "Saved log"
    }
}

fn read_tail_lines(path: &Path, limit: usize) -> Result<Vec<String>> {
    let bytes = fs::read(path)?;
    let text = strip_ansi_sequences(&decode_log_bytes(&bytes));
    let mut lines = text
        .lines()
        .rev()
        .take(limit)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    lines.reverse();
    Ok(lines)
}

fn decode_log_bytes(bytes: &[u8]) -> String {
    if bytes.starts_with(&[0xff, 0xfe]) || looks_like_utf16_le(bytes) {
        let start = if bytes.starts_with(&[0xff, 0xfe]) {
            2
        } else {
            0
        };
        let units = bytes[start..]
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect::<Vec<_>>();
        return String::from_utf16_lossy(&units);
    }
    String::from_utf8_lossy(bytes).into_owned()
}

fn looks_like_utf16_le(bytes: &[u8]) -> bool {
    if bytes.len() < 8 {
        return false;
    }
    let sample_len = bytes.len().min(512);
    let zero_odd_bytes = bytes[..sample_len]
        .iter()
        .enumerate()
        .filter(|(index, byte)| index % 2 == 1 && **byte == 0)
        .count();
    zero_odd_bytes * 4 >= sample_len
}

fn strip_ansi_sequences(text: &str) -> String {
    let mut cleaned = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\x1b' {
            cleaned.push(ch);
            continue;
        }
        if chars.peek() == Some(&'[') {
            let _ = chars.next();
            for next in chars.by_ref() {
                if ('@'..='~').contains(&next) {
                    break;
                }
            }
        }
    }
    cleaned
}

fn load_manifest(paths: &AppPaths) -> Result<Option<ComfyUiManifest>> {
    let path = manifest_path(paths);
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map(Some)
        .with_context(|| format!("failed to parse {}", path.display()))
}

/// Reuse whatever URL the manifest already on disk recorded, rather than the
/// current `comfyui_source_archive_url()` — that folder was produced by
/// *some* prior install, which may have run under a different
/// source-archive override than this one. Falls back to the current URL if
/// there's no prior manifest that can be read (e.g. it was deleted out from
/// under an otherwise-intact source folder, or is unreadable/unparseable) —
/// this path must not abort on a broken manifest, since it's otherwise the
/// one command that recovers from one.
fn reused_source_url(paths: &AppPaths) -> String {
    load_manifest(paths)
        .ok()
        .flatten()
        .map_or_else(comfyui_source_archive_url, |manifest| manifest.source_url)
}

fn save_manifest(paths: &AppPaths, manifest: &ComfyUiManifest) -> Result<()> {
    let path = manifest_path(paths);
    fs::create_dir_all(
        path.parent()
            .context("ComfyUI manifest path has no parent directory")?,
    )?;
    fs::write(
        &path,
        serde_json::to_vec_pretty(manifest).context("failed to serialize ComfyUI manifest")?,
    )
    .with_context(|| format!("failed to write {}", path.display()))
}

fn load_state(paths: &AppPaths) -> Result<Option<ComfyUiState>> {
    let path = state_path(paths);
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map(Some)
        .with_context(|| format!("failed to parse {}", path.display()))
}

fn save_state(paths: &AppPaths, state: &ComfyUiState) -> Result<()> {
    let path = state_path(paths);
    fs::create_dir_all(
        path.parent()
            .context("ComfyUI state path has no parent directory")?,
    )?;
    fs::write(
        &path,
        serde_json::to_vec_pretty(state).context("failed to serialize ComfyUI state")?,
    )
    .with_context(|| format!("failed to write {}", path.display()))
}

fn probe_comfyui(
    python: &Path,
    source_path: &Path,
    runtime_env: Option<&ComfyUiRuntimeEnvironment>,
) -> Result<ComfyUiProbe> {
    let probe_dir = create_temp_dir_under(source_path, "probe")?;
    let probe_path = probe_dir.join("probe.py");
    let result_path = probe_dir.join("result.json");
    let source_literal = serde_json::to_string(&child_path_string(source_path))?;
    let script = format!(
        r#"
import json, sys
sys.path.insert(0, {source_literal})
import torch
result = {{
    "torch_version": getattr(torch, "__version__", None),
    "torch_cuda_available": bool(torch.cuda.is_available()),
    "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
}}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(result, handle)
"#
    );
    fs::write(&probe_path, script)
        .with_context(|| format!("failed to write {}", probe_path.display()))?;
    let mut command = Command::new(python);
    command
        .arg(child_path_string(&probe_path))
        .arg(child_path_string(&result_path));
    if let Some(runtime_env) = runtime_env {
        apply_runtime_environment(&mut command, runtime_env)?;
    }
    let output = capture_configured_command(
        command,
        &format!("run ComfyUI GPU check with {}", python.display()),
    )?;
    let result_text = fs::read_to_string(&result_path).unwrap_or_default();
    let parsed_result = serde_json::from_str::<ComfyUiProbe>(result_text.trim());
    if !output.status.success() {
        if let Ok(probe) = parsed_result {
            let _ = fs::remove_dir_all(&probe_dir);
            return Ok(probe);
        }
        let _ = fs::remove_dir_all(&probe_dir);
        bail!(
            "ComfyUI GPU check failed with status {}.\nstdout: {}\nstderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stdout).trim(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let probe = match parsed_result {
        Ok(probe) => probe,
        Err(error) => {
            let _ = fs::remove_dir_all(&probe_dir);
            return Err(error).context("failed to parse ComfyUI GPU check output");
        }
    };
    let _ = fs::remove_dir_all(&probe_dir);
    Ok(probe)
}

struct CapturedCommandOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn capture_configured_command(
    mut command: Command,
    context_label: &str,
) -> Result<CapturedCommandOutput> {
    let temp_dir = create_capture_temp_dir()?;
    let stdout_path = temp_dir.join("stdout.txt");
    let stderr_path = temp_dir.join("stderr.txt");
    let stdout_file = fs::File::create(&stdout_path)
        .with_context(|| format!("failed to create {}", stdout_path.display()))?;
    let stderr_file = fs::File::create(&stderr_path)
        .with_context(|| format!("failed to create {}", stderr_path.display()))?;
    let status_result = command
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .status()
        .with_context(|| format!("{context_label}: failed to start command"));
    let stdout = fs::read(&stdout_path).unwrap_or_default();
    let stderr = fs::read(&stderr_path).unwrap_or_default();
    let _ = fs::remove_dir_all(&temp_dir);
    Ok(CapturedCommandOutput {
        status: status_result?,
        stdout,
        stderr,
    })
}

fn create_capture_temp_dir() -> Result<PathBuf> {
    create_temp_dir("capture")
}

fn create_temp_dir(label: &str) -> Result<PathBuf> {
    create_temp_dir_under(&std::env::temp_dir(), label)
}

fn create_temp_dir_under(root: &Path, label: &str) -> Result<PathBuf> {
    fs::create_dir_all(root).with_context(|| format!("failed to create {}", root.display()))?;
    for attempt in 0..1000 {
        let path = root.join(format!(
            "rocm-comfyui-{label}-{}-{}-{attempt}",
            std::process::id(),
            unix_time_millis()
        ));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(error).with_context(|| format!("failed to create {}", path.display()));
            }
        }
    }
    bail!("failed to allocate a temporary command capture directory")
}

/// Quotes `value` for log text only, using Rust's own `{value:?}`
/// Debug-format escaping; the real args passed to `Command::args` alongside
/// this are unquoted. Deliberately kept separate from `rocm_core::quoting`
/// (ROCMAI-727); see that module's doc comment for why.
fn quote_log_arg(value: &str) -> String {
    if value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || "-_./:=\\".contains(ch))
    {
        value.to_owned()
    } else {
        format!("{value:?}")
    }
}

fn child_path_string(path: &Path) -> String {
    if runtime_is_windows() {
        runtime_path_for_windows_child(path)
    } else {
        path.display().to_string()
    }
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use super::process::ensure_source_is_startable;
    use super::*;
    #[cfg(unix)]
    use crate::chat;
    use rocm_core::{runtime_python_executable_in_env, runtime_rocm_library_filename};
    use std::net::TcpListener;
    use std::time::{SystemTime, UNIX_EPOCH};

    pub(super) fn unused_local_port() -> Result<u16> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        Ok(listener.local_addr()?.port())
    }

    #[test]
    fn status_without_install_is_plain() -> Result<()> {
        let paths = test_paths("comfyui-status");
        let config = RocmCliConfig::default();
        let rendered = render_status(&paths, &config)?;
        assert!(rendered.contains("installed: no"));
        assert!(rendered.contains("next step: rocm comfyui install"));
        Ok(())
    }

    #[test]
    fn status_labels_managed_rocm_as_runtime_not_install() -> Result<()> {
        // `render_status` shares the ROCm noun with `select_runtime`'s errors.
        // With a ComfyUI install and an active runtime present it must render
        // "ROCm runtime:" and "Default ROCm runtime", never the stale
        // "ROCm install" labels the rename left behind.
        let paths = test_paths("comfyui-status-runtime-label");
        let runtime = ready_runtime_manifest(&paths, "release-wheel-gfx94x-dcgpu-7-13-0")?;
        write_runtime_manifest(&paths, &runtime)?;
        // Built from `runtime_app_root`, the same helper `install()` uses, rather
        // than the test-only `source_path(&paths)` (which hangs off a different,
        // data-dir-rooted `app_root`): this way the planted manifest's
        // `source_path` is the layout a real install actually produces, so this
        // test can show status's `folder:` line agrees with install's.
        let managed_source_path = source_path_from_app_root(&runtime_app_root(&runtime));
        save_manifest(
            &paths,
            &ComfyUiManifest {
                app_id: APP_ID.to_owned(),
                runtime_key: runtime.runtime_key.clone(),
                runtime_id: runtime.runtime_id.clone(),
                runtime_version: runtime.version.clone(),
                runtime_root: runtime.install_root.clone(),
                python_executable: paths.data_dir.join("runtimes").join("python.exe"),
                source_url: comfyui_source_archive_url(),
                source_path: managed_source_path.clone(),
                requirements_path: managed_source_path.join("requirements.txt"),
                pip_cache_dir: None,
                log_path: app_root(&paths).join("logs").join("install-100.log"),
                torch_version: Some("2.10.0".to_owned()),
                torch_cuda_available: true,
                installed_at_unix_ms: 100,
            },
        )?;
        let config = RocmCliConfig {
            active_runtime_key: Some(runtime.runtime_key),
            ..Default::default()
        };

        let rendered = render_status(&paths, &config)?;

        assert!(
            rendered.contains("ROCm runtime:"),
            "status should label the managed ROCm as `ROCm runtime:`, got: {rendered}"
        );
        assert!(
            rendered.contains("Default ROCm runtime"),
            "status should title the active default `Default ROCm runtime`, got: {rendered}"
        );
        assert!(
            !rendered.contains("ROCm install"),
            "status must not reintroduce the `ROCm install` label, got: {rendered}"
        );
        assert!(
            rendered.contains(&format!("  folder: {}\n", managed_source_path.display())),
            "status `folder:` should report the ComfyUI source checkout, not the runtime root, got: {rendered}"
        );
        assert!(
            !rendered.contains(&format!("  folder: {}\n", runtime.install_root.display())),
            "status `folder:` must not report the bare runtime root, got: {rendered}"
        );
        Ok(())
    }

    fn test_manifest_with_source_url(paths: &AppPaths, source_url: &str) -> ComfyUiManifest {
        ComfyUiManifest {
            app_id: APP_ID.to_owned(),
            runtime_key: "test-runtime".to_owned(),
            runtime_id: "test-runtime-id".to_owned(),
            runtime_version: "1.0.0".to_owned(),
            runtime_root: paths.data_dir.join("runtimes").join("test-runtime"),
            python_executable: paths.data_dir.join("runtimes").join("python.exe"),
            source_url: source_url.to_owned(),
            source_path: source_path(paths),
            requirements_path: source_path(paths).join("requirements.txt"),
            pip_cache_dir: None,
            log_path: app_root(paths).join("logs").join("install-100.log"),
            torch_version: None,
            torch_cuda_available: false,
            installed_at_unix_ms: 100,
        }
    }

    #[test]
    fn reused_source_url_falls_back_to_current_url_when_no_manifest_on_disk() {
        let paths = test_paths("comfyui-reused-url-no-manifest");

        let url = reused_source_url(&paths);

        assert_eq!(url, comfyui_source_archive_url());
    }

    #[test]
    fn reused_source_url_returns_recorded_url_from_valid_manifest() -> Result<()> {
        let paths = test_paths("comfyui-reused-url-valid-manifest");
        let recorded_url = "https://example.invalid/prior-comfyui-source.tar.gz";
        save_manifest(&paths, &test_manifest_with_source_url(&paths, recorded_url))?;

        let url = reused_source_url(&paths);

        assert_eq!(url, recorded_url);
        Ok(())
    }

    #[test]
    fn reused_source_url_falls_back_to_current_url_on_unparseable_manifest() -> Result<()> {
        // Mutation-sensitive: fails the instant `reused_source_url` reverts
        // from `.ok().flatten()` to propagating `load_manifest`'s error,
        // which is the exact regression this PR shipped and then fixed.
        let paths = test_paths("comfyui-reused-url-corrupt-manifest");
        let path = manifest_path(&paths);
        fs::create_dir_all(path.parent().expect("manifest path has a parent"))?;
        fs::write(&path, b"not valid json")?;

        let url = reused_source_url(&paths);

        assert_eq!(url, comfyui_source_archive_url());
        Ok(())
    }

    #[test]
    fn logs_render_inline_latest_output() -> Result<()> {
        let paths = test_paths("comfyui-logs");
        let logs = app_root(&paths).join("logs");
        fs::create_dir_all(&logs)?;
        let install_log = logs.join("install-100.log");
        let run_log = logs.join("start-200.log");
        fs::write(&install_log, "install 1\ninstall 2\ninstall 3\n")?;
        fs::write(&run_log, "run 1\nrun 2\nrun 3\n")?;
        save_manifest(
            &paths,
            &ComfyUiManifest {
                app_id: APP_ID.to_owned(),
                runtime_key: "therock-release:gfx120X-all".to_owned(),
                runtime_id: "therock-release".to_owned(),
                runtime_version: "7.13.0a20260511".to_owned(),
                runtime_root: paths.data_dir.join("runtimes").join("runtime"),
                python_executable: paths.data_dir.join("runtimes").join("python.exe"),
                source_url: comfyui_source_archive_url(),
                source_path: source_path(&paths),
                requirements_path: source_path(&paths).join("requirements.txt"),
                pip_cache_dir: None,
                log_path: install_log.clone(),
                torch_version: Some("2.10.0".to_owned()),
                torch_cuda_available: true,
                installed_at_unix_ms: 100,
            },
        )?;
        save_state(
            &paths,
            &ComfyUiState {
                app_id: APP_ID.to_owned(),
                url: "http://127.0.0.1:8188".to_owned(),
                host: "127.0.0.1".to_owned(),
                port: 8188,
                pid: 42,
                source_path: source_path(&paths),
                python_executable: paths.data_dir.join("runtimes").join("python.exe"),
                log_path: run_log.clone(),
                started_at_unix_ms: 200,
            },
        )?;

        let rendered = render_logs(&paths, 2)?;

        assert!(rendered.contains("ComfyUI logs"));
        assert!(rendered.contains("Install log"));
        assert!(rendered.contains("Run log"));
        assert!(rendered.contains("saved file:"));
        assert!(rendered.contains(&install_log.display().to_string()));
        assert!(rendered.contains(&run_log.display().to_string()));
        assert!(rendered.contains("install 2"));
        assert!(rendered.contains("install 3"));
        assert!(!rendered.contains("install 1"));
        assert!(rendered.contains("run 2"));
        assert!(rendered.contains("run 3"));
        Ok(())
    }

    #[test]
    fn log_tail_decodes_utf16_and_strips_ansi() -> Result<()> {
        let paths = test_paths("comfyui-log-cleanup");
        let log = app_root(&paths).join("logs").join("start-utf16.log");
        fs::create_dir_all(log.parent().context("test log has no parent")?)?;
        let text = "\u{feff}\u{1b}[32m[INFO]\u{1b}[0m Device: AMD Radeon RX 9070 XT\n";
        let mut bytes = Vec::new();
        for unit in text.encode_utf16() {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }
        fs::write(&log, bytes)?;

        let lines = read_tail_lines(&log, 5)?;

        assert_eq!(lines, vec!["[INFO] Device: AMD Radeon RX 9070 XT"]);
        Ok(())
    }

    fn tail_fixture(name: &str, bytes: &[u8]) -> Result<PathBuf> {
        let paths = test_paths(name);
        let dir = app_root(&paths).join("logs");
        fs::create_dir_all(&dir)?;
        let path = dir.join("start-fixture.log");
        fs::write(&path, bytes)?;
        Ok(path)
    }

    fn utf16_le(text: &str, bom: bool) -> Vec<u8> {
        let mut bytes = if bom { vec![0xff, 0xfe] } else { Vec::new() };
        for unit in text.encode_utf16() {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }
        bytes
    }

    /// 5 MiB: a marked first line, 5120 lines of 1 KiB, a marked last line.
    fn large_log_with_head_marker() -> Vec<u8> {
        let mut bytes = b"HEAD-MARKER\n".to_vec();
        let filler = format!("{}\n", "x".repeat(1023));
        for _ in 0..5120 {
            bytes.extend_from_slice(filler.as_bytes());
        }
        bytes.extend_from_slice(b"TAIL-MARKER\n");
        bytes
    }

    #[test]
    fn read_tail_lines_a_large_file_is_read_whole_today() -> Result<()> {
        // Characterisation of the whole-file read: a line far outside any
        // tail window is still returned when the limit is large enough.
        let log = tail_fixture("comfyui-tail-large", &large_log_with_head_marker())?;

        let lines = read_tail_lines(&log, 10_000)?;

        assert_eq!(lines.first().map(String::as_str), Some("HEAD-MARKER"));
        assert_eq!(lines.last().map(String::as_str), Some("TAIL-MARKER"));
        assert_eq!(lines.len(), 5122);
        Ok(())
    }

    #[test]
    fn read_tail_lines_decodes_a_bomless_utf16_log_and_strips_ansi() -> Result<()> {
        let text = "\u{1b}[32m[INFO]\u{1b}[0m one\r\n\u{1b}[1;31mtwo\u{1b}[0m\r\n";
        let log = tail_fixture("comfyui-tail-utf16-nobom", &utf16_le(text, false))?;

        let lines = read_tail_lines(&log, 5)?;

        assert_eq!(lines, vec!["[INFO] one", "two"]);
        Ok(())
    }

    #[test]
    fn read_tail_lines_strips_ansi_from_an_eight_bit_log() -> Result<()> {
        let log = tail_fixture("comfyui-tail-ansi-utf8", b"\x1b[32mgreen\x1b[0m done\n")?;

        assert_eq!(read_tail_lines(&log, 5)?, vec!["green done"]);
        Ok(())
    }

    #[test]
    fn read_tail_lines_decodes_invalid_eight_bit_text_lossily() -> Result<()> {
        let log = tail_fixture("comfyui-tail-invalid", b"ok\nbad\xffbyte\nlast\n")?;

        assert_eq!(
            read_tail_lines(&log, 5)?,
            vec!["ok", "bad\u{FFFD}byte", "last"]
        );
        Ok(())
    }

    #[test]
    fn read_tail_lines_missing_file_is_a_not_found_error() {
        let paths = test_paths("comfyui-tail-missing");
        let missing = app_root(&paths).join("logs").join("absent.log");

        let error = read_tail_lines(&missing, 5).expect_err("a missing log must not read");

        let io_error = error
            .downcast_ref::<io::Error>()
            .expect("the bare io error is returned, without a path label");
        assert_eq!(io_error.kind(), io::ErrorKind::NotFound);
        // The same applies to limit 0: the file is opened before the limit is
        // looked at.
        assert!(read_tail_lines(&missing, 0).is_err());
    }

    #[test]
    fn read_tail_lines_limit_zero_returns_nothing_for_an_existing_log() -> Result<()> {
        let log = tail_fixture("comfyui-tail-zero", b"a\nb\n")?;

        assert!(read_tail_lines(&log, 0)?.is_empty());
        Ok(())
    }

    #[test]
    fn read_tail_lines_returns_every_line_when_fewer_than_the_limit() -> Result<()> {
        let log = tail_fixture("comfyui-tail-few", b"a\nb\n")?;

        assert_eq!(read_tail_lines(&log, 50)?, vec!["a", "b"]);
        Ok(())
    }

    #[test]
    fn read_tail_lines_keeps_a_final_line_without_a_trailing_newline() -> Result<()> {
        let log = tail_fixture("comfyui-tail-no-newline", b"a\nb\nc")?;

        assert_eq!(read_tail_lines(&log, 2)?, vec!["b", "c"]);
        Ok(())
    }

    #[test]
    fn read_tail_lines_strips_crlf_line_endings() -> Result<()> {
        let log = tail_fixture("comfyui-tail-crlf", b"a\r\nb\r\n")?;

        assert_eq!(read_tail_lines(&log, 5)?, vec!["a", "b"]);
        Ok(())
    }

    #[test]
    fn unmanaged_running_url_reports_reachable_loopback_endpoint() -> Result<()> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();

        let url = unmanaged_running_url("127.0.0.1", port);

        assert_eq!(url, Some(format!("http://127.0.0.1:{port}")));
        Ok(())
    }

    #[test]
    fn status_reports_stopped_when_saved_comfyui_pid_is_gone() -> Result<()> {
        let paths = test_paths("comfyui-stale-state");
        let logs = app_root(&paths).join("logs");
        fs::create_dir_all(&logs)?;
        let port = unused_local_port()?;
        save_state(
            &paths,
            &ComfyUiState {
                app_id: APP_ID.to_owned(),
                url: format!("http://127.0.0.1:{port}"),
                host: "127.0.0.1".to_owned(),
                port,
                pid: 0,
                source_path: source_path(&paths),
                python_executable: paths.data_dir.join("runtimes").join("python.exe"),
                log_path: logs.join("start-200.log"),
                started_at_unix_ms: 200,
            },
        )?;

        let rendered = render_status(&paths, &RocmCliConfig::default())?;

        assert!(rendered.contains("status: stopped"));
        assert!(rendered.contains("next step: rocm comfyui start"));
        assert!(!rendered.contains("status: starting or running"));
        Ok(())
    }

    pub(super) fn ready_runtime_manifest(
        paths: &AppPaths,
        runtime_key: &str,
    ) -> Result<therock::InstalledRuntimeManifest> {
        let runtime_root = paths.data_dir.join("runtime-root").join(runtime_key);
        let python = runtime_python_executable_in_env(&runtime_root);
        let python_bin = python
            .parent()
            .context("runtime Python path has no parent")?
            .to_path_buf();
        let sdk_root = runtime_root.join("sdk");
        let sdk_bin = sdk_root.join("bin");
        fs::create_dir_all(&python_bin)?;
        fs::create_dir_all(&sdk_bin)?;
        fs::write(runtime_root.join(".rocm-cli-runtime.json"), "{}")?;
        fs::write(&python, "python")?;
        let amdhip = sdk_bin.join(runtime_rocm_library_filename("amdhip64"));
        let hipblas = sdk_bin.join(runtime_rocm_library_filename("hipblas"));
        fs::write(&amdhip, "amdhip")?;
        fs::write(&hipblas, "hipblas")?;

        Ok(therock::InstalledRuntimeManifest {
            runtime_key: runtime_key.to_owned(),
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            channel: "release".to_owned(),
            format: "wheel".to_owned(),
            family: "gfx120X-all".to_owned(),
            family_source: "test".to_owned(),
            version: "7.13.0a20260511".to_owned(),
            install_root: runtime_root,
            selected_artifact_url: "https://example.invalid/simple".to_owned(),
            source_layout_generation: None,
            index_url: None,
            tarball_file_name: None,
            python_launcher: None,
            python_executable: Some(python.display().to_string()),
            pip_cache_dir: None,
            rocm_sdk: Some(therock::RocmSdkPythonProbe {
                import_ok: true,
                root_path: Some(sdk_root),
                bin_path: Some(sdk_bin),
                resolved_libraries: vec![
                    therock::RocmSdkLibraryProbe {
                        shortname: "amdhip64".to_owned(),
                        paths: vec![amdhip],
                    },
                    therock::RocmSdkLibraryProbe {
                        shortname: "hipblas".to_owned(),
                        paths: vec![hipblas],
                    },
                ],
                ..therock::RocmSdkPythonProbe::default()
            }),
            sdk_torch: None,
            wheel_composition: None,
            read_only: false,
            imported_from: None,
            devel: true,
            installed_at_unix_ms: 100,
        })
    }

    pub(super) fn write_runtime_manifest(
        paths: &AppPaths,
        manifest: &therock::InstalledRuntimeManifest,
    ) -> Result<()> {
        let registry_dir = paths.data_dir.join("runtimes").join("registry");
        fs::create_dir_all(&registry_dir)?;
        fs::write(
            registry_dir.join(format!("{}.json", manifest.runtime_key)),
            serde_json::to_vec_pretty(manifest)?,
        )?;
        Ok(())
    }

    pub(super) fn test_paths(name: &str) -> AppPaths {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("rocm-cli-{name}-{nonce}"));
        AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        }
    }

    /// User content an existing ComfyUI folder holds, keyed by path relative to
    /// `source/`. Every kept top-level entry appears at least once, so dropping
    /// any of them from the preserved list turns a reinstall test red.
    const USER_FILES: [(&str, &str); 10] = [
        ("models/checkpoints/my-model.safetensors", "user model"),
        ("user/default/workflows/my-workflow.json", "user workflow"),
        ("output/ComfyUI_00001_.png", "user image"),
        ("input/my-upload.png", "user upload"),
        ("custom_nodes/my-node/__init__.py", "user node"),
        ("datasets/my-set/0001.png", "user dataset"),
        ("extra_model_paths.yaml", "user model paths"),
        // The user's copy of a file the new release also ships wins.
        ("custom_nodes/websocket_image_save.py", "old sample node"),
        // No release ships these: left in place.
        ("styles.csv", "user styles"),
        ("temp/scratch.png", "scratch"),
    ];

    /// A ComfyUI install that has been used: release code plus user content.
    fn plant_used_comfyui_source(source: &Path) -> Result<()> {
        for (relative, contents) in [
            ("main.py", "old main"),
            ("requirements.txt", "torch\n"),
            ("comfy/removed_upstream.py", "old code"),
        ]
        .into_iter()
        .chain(USER_FILES)
        {
            let path = source.join(relative);
            fs::create_dir_all(path.parent().context("fixture path has no parent")?)?;
            fs::write(&path, contents)?;
        }
        Ok(())
    }

    pub(super) fn assert_user_files_intact(source: &Path) {
        for (relative, contents) in USER_FILES {
            let path = source.join(relative);
            assert_eq!(
                fs::read_to_string(&path).ok().as_deref(),
                Some(contents),
                "a reinstall must keep {}",
                path.display()
            );
        }
    }

    /// Writes the archive `download_and_extract_source` would otherwise
    /// download, so a reinstall runs offline. Its top-level folder mirrors
    /// GitHub's `ComfyUI-master/`. Only the unix-only install tests use it.
    #[cfg(unix)]
    pub(super) fn plant_cached_source_archive(
        app_root: &Path,
        files: &[(&str, &str)],
    ) -> Result<()> {
        let archive_path = app_root.join("downloads").join(COMFYUI_SOURCE_ARCHIVE_NAME);
        fs::create_dir_all(
            archive_path
                .parent()
                .context("archive path has no parent")?,
        )?;
        let encoder = flate2::write::GzEncoder::new(
            fs::File::create(&archive_path)?,
            flate2::Compression::fast(),
        );
        let mut builder = tar::Builder::new(encoder);
        for (relative, contents) in files {
            let mut header = tar::Header::new_gnu();
            header.set_size(contents.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append_data(
                &mut header,
                format!("ComfyUI-master/{relative}"),
                contents.as_bytes(),
            )?;
        }
        builder.into_inner()?.finish()?;
        Ok(())
    }

    pub(super) fn reinstall_fixture(name: &str) -> Result<(AppPaths, RocmCliConfig, PathBuf)> {
        let paths = test_paths(name);
        let runtime = ready_runtime_manifest(&paths, "reinstall-runtime")?;
        write_runtime_manifest(&paths, &runtime)?;
        let config = RocmCliConfig {
            active_runtime_key: Some(runtime.runtime_key.clone()),
            ..Default::default()
        };
        let app_root = runtime_app_root(&runtime);
        plant_used_comfyui_source(&source_path_from_app_root(&app_root))?;
        Ok((paths, config, app_root))
    }

    pub(super) fn reinstall_options(dry_run: bool) -> ComfyUiInstallOptions {
        ComfyUiInstallOptions {
            runtime_id: None,
            reinstall: true,
            dry_run,
        }
    }

    pub(super) fn remove_test_root(paths: &AppPaths) {
        if let Some(root) = paths.data_dir.parent() {
            let _ = fs::remove_dir_all(root);
        }
    }

    /// A cached release archive and a runtime Python that passes the GPU
    /// check, so `install` runs to completion offline.
    #[cfg(unix)]
    pub(super) fn plant_installable_release(paths: &AppPaths, app_root: &Path) -> Result<()> {
        plant_cached_source_archive(
            app_root,
            &[
                ("main.py", "new main"),
                ("requirements.txt", "torch\n"),
                ("comfy/added_upstream.py", "new code"),
            ],
        )?;
        let runtime = ready_runtime_manifest(paths, "reinstall-runtime")?;
        let python = PathBuf::from(runtime.python_executable.context("runtime has no Python")?);
        rocm_core::fake_executable::write_fake_executable(
            &python,
            "#!/bin/sh\nprintf '%s' '{\"torch_version\": \"2.4.0\", \"torch_cuda_available\": true, \"device_count\": 1, \"devices\": [\"Fake GPU\"]}' > \"$2\"\n",
        )?;
        Ok(())
    }

    /// `start` refuses a folder whose reinstall was interrupted, and `status`
    /// says so. The command both name is then run exactly as printed and
    /// clears it: the marker is gone and `start`'s checks pass.
    #[cfg(unix)]
    #[test]
    fn start_refuses_an_interrupted_reinstall_until_install_finishes_it() -> Result<()> {
        let (paths, config, app_root) = reinstall_fixture("comfyui-start-mid-swap")?;
        let source = source_path_from_app_root(&app_root);
        plant_installable_release(&paths, &app_root)?;
        install(&paths, &config, reinstall_options(false))?;
        // As an interrupted reinstall leaves it.
        fs::write(source.join(source_swap::SWAP_MARKER), "models\n")?;
        fs::remove_file(source.join("main.py"))?;
        let advice = "a reinstall was interrupted; run `rocm comfyui install --runtime-id reinstall-runtime` to finish it";

        let refused = start(
            &paths,
            ComfyUiStartOptions {
                host: "127.0.0.1".to_owned(),
                port: unused_local_port()?,
                no_open_browser: true,
            },
        )
        .expect_err("start must refuse a half-replaced folder");
        assert!(format!("{refused:#}").contains(advice), "{refused:#}");
        let status = render_status(&paths, &config)?;
        assert!(status.contains(&format!("  note: {advice}\n")), "{status}");

        // `rocm comfyui install --runtime-id reinstall-runtime`, as printed.
        install(
            &paths,
            &config,
            ComfyUiInstallOptions {
                runtime_id: Some("reinstall-runtime".to_owned()),
                reinstall: false,
                dry_run: false,
            },
        )?;

        assert!(!source.join(source_swap::SWAP_MARKER).exists());
        let manifest = load_manifest(&paths)?.context("manifest")?;
        ensure_source_is_startable(&manifest)?;
        assert!(!render_status(&paths, &config)?.contains("interrupted"));
        assert_user_files_intact(&source);
        remove_test_root(&paths);
        Ok(())
    }

    /// A long-lived process standing in for a running ComfyUI.
    pub(super) fn spawn_stand_in_process() -> Result<std::process::Child> {
        let mut command = if cfg!(windows) {
            let mut command = Command::new("ping");
            command.args(["-n", "60", "127.0.0.1"]);
            command
        } else {
            let mut command = Command::new("sleep");
            command.arg("60");
            command
        };
        command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("failed to spawn a stand-in process")
    }

    /// The chat's status answer passes on the interrupted-reinstall note and
    /// does not advise `start`, which would refuse; `status` drops its
    /// `start` next step too. Once the named install finishes the swap, both
    /// advise `start` again.
    #[cfg(unix)]
    #[test]
    fn chat_status_answer_names_the_install_that_finishes_an_interrupted_reinstall() -> Result<()> {
        let (paths, config, app_root) = reinstall_fixture("comfyui-chat-status-mid-swap")?;
        let source = source_path_from_app_root(&app_root);
        plant_installable_release(&paths, &app_root)?;
        install(&paths, &config, reinstall_options(false))?;
        // A saved state for a ComfyUI that has exited: status reports it
        // stopped.
        let mut exited = spawn_stand_in_process()?;
        let dead_pid = exited.id();
        exited.kill()?;
        exited.wait()?;
        let port = unused_local_port()?;
        save_state(
            &paths,
            &ComfyUiState {
                app_id: APP_ID.to_owned(),
                url: format!("http://127.0.0.1:{port}"),
                host: "127.0.0.1".to_owned(),
                port,
                pid: dead_pid,
                source_path: source.clone(),
                python_executable: paths.data_dir.join("python"),
                log_path: paths.data_dir.join("start.log"),
                started_at_unix_ms: 1,
            },
        )?;
        // As an interrupted reinstall leaves it.
        fs::write(source.join(source_swap::SWAP_MARKER), "models\n")?;
        fs::remove_file(source.join("main.py"))?;
        let advice = "a reinstall was interrupted; run `rocm comfyui install --runtime-id reinstall-runtime` to finish it";

        let answer = render_chat_status_answer(&paths)?;
        assert!(answer.contains("Running: no\n"), "{answer}");
        assert!(answer.contains(&format!("Note: {advice}.\n")), "{answer}");
        assert!(!answer.contains("comfyui start"), "{answer}");
        let status = render_status(&paths, &config)?;
        assert!(!status.contains("comfyui start"), "{status}");
        // The summary the assistant answers with after a status tool call
        // passes the note on too.
        // Another tool's output ahead of it in the same text keeps its notes.
        let summary = chat::deterministic_comfyui_tool_summary(&format!(
            "Engines\n  note: another tool's note\n\n{status}"
        ))
        .context("a status summary")?;
        assert!(summary.contains(&format!("  Note: {advice}.")), "{summary}");
        assert!(!summary.contains("another tool"), "{summary}");
        // What the note claims: start refuses until that install runs.
        let manifest = load_manifest(&paths)?.context("manifest")?;
        assert!(ensure_source_is_startable(&manifest).is_err());

        install(
            &paths,
            &config,
            ComfyUiInstallOptions {
                runtime_id: Some("reinstall-runtime".to_owned()),
                reinstall: false,
                dry_run: false,
            },
        )?;

        let answer = render_chat_status_answer(&paths)?;
        assert!(!answer.contains("interrupted"), "{answer}");
        assert!(answer.contains("run `rocm comfyui start`"), "{answer}");
        let summary = chat::deterministic_comfyui_tool_summary(&render_status(&paths, &config)?)
            .context("a status summary")?;
        assert!(!summary.contains("Note:"), "{summary}");
        assert!(render_status(&paths, &config)?.contains("next step: rocm comfyui start"));
        ensure_source_is_startable(&load_manifest(&paths)?.context("manifest")?)?;
        remove_test_root(&paths);
        Ok(())
    }

    /// An installed ComfyUI with a saved state for the given process and
    /// port, as `start` leaves it.
    #[cfg(unix)]
    fn installed_with_state(name: &str, pid: u32, port: u16) -> Result<(AppPaths, String)> {
        let (paths, config, app_root) = reinstall_fixture(name)?;
        let source = source_path_from_app_root(&app_root);
        plant_installable_release(&paths, &app_root)?;
        install(&paths, &config, reinstall_options(false))?;
        let url = format_http_base_url("127.0.0.1", port);
        save_state(
            &paths,
            &ComfyUiState {
                app_id: APP_ID.to_owned(),
                url: url.clone(),
                host: "127.0.0.1".to_owned(),
                port,
                pid,
                source_path: source,
                python_executable: paths.data_dir.join("python"),
                log_path: paths.data_dir.join("start.log"),
                started_at_unix_ms: 1,
            },
        )?;
        Ok((paths, url))
    }

    /// A ComfyUI that answers on its port is reported running, with its URL,
    /// and `start` is not advised.
    #[cfg(unix)]
    #[test]
    fn chat_status_answer_reports_a_running_comfyui_with_its_url() -> Result<()> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let (paths, url) =
            installed_with_state("comfyui-chat-status-running", std::process::id(), port)?;

        let answer = render_chat_status_answer(&paths)?;

        assert!(
            answer.contains(&format!("Running: yes\nURL: {url}\n")),
            "{answer}"
        );
        assert!(!answer.contains("comfyui start"), "{answer}");
        // What the answer claims: the URL answers.
        assert!(std::net::TcpStream::connect(("127.0.0.1", port)).is_ok());
        drop(listener);
        remove_test_root(&paths);
        Ok(())
    }

    /// A ComfyUI whose process is up but whose port does not answer yet is
    /// reported starting, and `start`, which would launch a second one, is
    /// not advised.
    #[cfg(unix)]
    #[test]
    fn chat_status_answer_reports_a_starting_comfyui() -> Result<()> {
        let mut child = spawn_stand_in_process()?;
        let port = unused_local_port()?;
        let result = installed_with_state("comfyui-chat-status-starting", child.id(), port)
            .and_then(|(paths, _)| {
                let status = render_status(&paths, &RocmCliConfig::default())?;
                Ok((render_chat_status_answer(&paths)?, status, paths))
            });
        let _ = child.kill();
        let _ = child.wait();

        let (answer, status, paths) = result?;
        assert!(answer.contains("Running: starting\n"), "{answer}");
        // `status` says `note: starting` under `Running`; the tool summary
        // sums that up as its running state, not as a note.
        assert!(status.contains("  note: starting\n"), "{status}");
        let summary =
            chat::deterministic_comfyui_tool_summary(&status).context("a status summary")?;
        assert!(summary.contains("  Running: starting."), "{summary}");
        assert!(!summary.contains("Note:"), "{summary}");
        assert!(!answer.contains("URL:"), "{answer}");
        assert!(!answer.contains("comfyui start"), "{answer}");
        remove_test_root(&paths);
        Ok(())
    }

    /// Both pieces of advice name an install to run as printed, so the line
    /// must split back to exactly that argv whatever the runtime key holds.
    /// The key is read back by stripping the fixed text around it rather than
    /// splitting on the backticks, which a hostile key may itself contain.
    /// Mutation: interpolating the key bare in `finishing_install_command`
    /// fails this for both.
    #[test]
    fn finishing_advice_names_the_install_as_its_argv() {
        use proptest::prelude::*;
        use proptest::test_runner::{Config, TestRunner};

        let mut runner = TestRunner::new(Config {
            cases: 256,
            ..Config::default()
        });
        runner
            .run(&crate::shell_quote::hostile::arg(), |runtime_key| {
                let expected = ["rocm", "comfyui", "install", "--runtime-id", &runtime_key]
                    .map(str::to_owned)
                    .to_vec();
                for (message, prefix, suffix) in [
                    (
                        interrupted_reinstall_advice(&runtime_key),
                        "a reinstall was interrupted; run `",
                        "` to finish it",
                    ),
                    (
                        unfinished_install_advice(" (replaced: comfy)", &runtime_key),
                        "the new ComfyUI code is in place (replaced: comfy), but the install \
                         did not finish; once the cause below is fixed, run `",
                        "` to finish it",
                    ),
                ] {
                    let command = message
                        .strip_prefix(prefix)
                        .and_then(|rest| rest.strip_suffix(suffix))
                        .unwrap_or_else(|| panic!("unexpected advice: {message}"));
                    prop_assert_eq!(shlex::split(command), Some(expected.clone()), "{}", message);
                }
                Ok(())
            })
            .unwrap();
    }
}
