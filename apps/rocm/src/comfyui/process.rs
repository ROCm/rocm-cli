// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Starting, stopping and watching the ComfyUI process this CLI launches.
//!
//! [`start`] checks the installed folder, runs the GPU check, launches ComfyUI
//! in the background (a detached child on Linux, a hidden PowerShell launcher on
//! Windows) and waits for its port to answer; [`stop`] ends the process tree the
//! saved state names. [`evaluate_running_state`] is how `status` and the
//! reinstall guard ([`running_from`], [`refuse_while_running`],
//! [`running_advice`]) tell a running, a still-loading and a stopped ComfyUI
//! apart. The saved state itself, and the types in it, stay in the parent
//! module.

use super::runtime_env::{
    apply_runtime_environment, runtime_environment_assignments, runtime_environment_for_manifest,
    same_path_text,
};
use super::{
    APP_ID, APP_NAME, ComfyUiManifest, ComfyUiRunReport, ComfyUiRunState,
    ComfyUiRuntimeEnvironment, ComfyUiStartOptions, ComfyUiState, app_root,
    capture_configured_command, child_path_string, interrupted_reinstall_advice, load_manifest,
    load_state, models_folder_for_manifest, probe_comfyui, save_state, source_swap,
    start_log_path_for_manifest, state_path,
};
use anyhow::{Context, Result, bail};
use rocm_core::browser::{Opener, SystemOpener};
use rocm_core::{
    AppPaths, format_http_base_url, powershell_quote, runtime_is_linux, runtime_is_windows,
    unix_time_millis,
};
use std::fmt::Write as FmtWrite;
use std::fs;
use std::net::{TcpStream, ToSocketAddrs};
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

pub(crate) fn start(paths: &AppPaths, options: ComfyUiStartOptions) -> Result<String> {
    let manifest = load_manifest(paths)?.context("ComfyUI is not installed yet")?;
    ensure_source_is_startable(&manifest)?;
    let runtime_env = runtime_environment_for_manifest(paths, &manifest)?;
    let models_folder = models_folder_for_manifest(&manifest);
    fs::create_dir_all(&models_folder)
        .with_context(|| format!("failed to create {}", models_folder.display()))?;
    let probe = probe_comfyui(
        &manifest.python_executable,
        &manifest.source_path,
        Some(&runtime_env),
    )?;
    if !probe.torch_cuda_available {
        bail!("ComfyUI cannot start because the AMD GPU check failed. No CPU mode was used.");
    }
    let url = format_http_base_url(&options.host, options.port);
    let log_path = start_log_path_for_manifest(&manifest);
    fs::create_dir_all(
        log_path
            .parent()
            .context("ComfyUI start log path has no parent directory")?,
    )?;
    fs::File::create(&log_path)
        .with_context(|| format!("failed to create {}", log_path.display()))?;
    let pid = spawn_comfyui_background(paths, &manifest, &options, &runtime_env, &log_path)?;
    let state = ComfyUiState {
        app_id: APP_ID.to_owned(),
        url: url.clone(),
        host: options.host.clone(),
        port: options.port,
        pid,
        source_path: manifest.source_path.clone(),
        python_executable: manifest.python_executable,
        log_path: log_path.clone(),
        started_at_unix_ms: unix_time_millis(),
    };
    save_state(paths, &state)?;
    let run_report = wait_for_running_state(&state, Duration::from_mins(3));
    if run_report.state == ComfyUiRunState::Stopped {
        bail!(
            "ComfyUI stopped before the local URL was ready. Check the log at {}",
            log_path.display()
        );
    }
    let browser_status = if options.no_open_browser {
        "not opened (--no-open-browser)".to_owned()
    } else {
        match SystemOpener.open(&url) {
            Ok(()) => "opened".to_owned(),
            Err(error) => format!("not opened ({error})"),
        }
    };

    let mut output = String::new();
    writeln!(output, "{APP_NAME}")?;
    writeln!(
        output,
        "  status: {}",
        comfyui_run_state_cli_label(run_report.state)
    )?;
    writeln!(
        output,
        "  AMD GPU check: ready ({} device{})",
        probe.device_count,
        if probe.device_count == 1 { "" } else { "s" }
    )?;
    if !probe.devices.is_empty() {
        writeln!(output, "  GPU: {}", probe.devices.join(", "))?;
    }
    writeln!(output, "  URL: {url}")?;
    writeln!(output, "  models path: {}", models_folder.display())?;
    writeln!(output, "  browser: {browser_status}")?;
    if run_report.state == ComfyUiRunState::Starting {
        writeln!(output, "  note: still loading")?;
    }
    Ok(output)
}

/// The checks `start` makes on the installed folder before launching it.
pub(super) fn ensure_source_is_startable(manifest: &ComfyUiManifest) -> Result<()> {
    if source_swap::swap_interrupted(&manifest.source_path) {
        bail!(
            "{}; it would start a half-replaced ComfyUI",
            interrupted_reinstall_advice(&manifest.runtime_key)
        );
    }
    if !manifest.source_path.join("main.py").is_file() {
        bail!(
            "ComfyUI main.py is missing from {}; reinstall ComfyUI",
            manifest.source_path.display()
        );
    }
    Ok(())
}

/// The saved state of the ComfyUI this CLI started, when it is still running
/// from `source_path`: it keeps the old code loaded and keeps writing into the
/// folder a reinstall changes, so the reinstall waits for it. `rocm comfyui
/// stop` stops it and removes the saved state this reads, which is what clears
/// it. A saved state that cannot be read counts as not running, and
/// `warnings` says so: `stop` cannot read it either, so a refusal would have
/// nothing to clear it. What a running ComfyUI risks is the code it runs; the
/// preserved folders are left alone whatever is running.
pub(super) fn running_from(
    paths: &AppPaths,
    source_path: &Path,
    warnings: &mut Vec<String>,
) -> Option<ComfyUiState> {
    let state = match load_state(paths) {
        Ok(state) => state?,
        Err(error) => {
            warnings.push(format!(
                "{error:#}; whether a ComfyUI started by rocm-cli is running from {} is unknown, \
                 so the reinstall is not refused for it",
                source_path.display()
            ));
            return None;
        }
    };
    (same_path_text(&state.source_path, source_path)
        && evaluate_running_state(&state).state != ComfyUiRunState::Stopped)
        .then_some(state)
}

/// The refusal of a reinstall while the ComfyUI [`running_from`] found runs.
pub(super) fn refuse_while_running(
    source_path: &Path,
    running: Option<ComfyUiState>,
) -> Result<()> {
    match running {
        Some(state) => bail!(
            "{}, then reinstall. Nothing was changed.",
            running_advice(source_path, &state)
        ),
        None => Ok(()),
    }
}

/// What a reinstall says about a ComfyUI found by [`running_from`].
pub(super) fn running_advice(source_path: &Path, state: &ComfyUiState) -> String {
    format!(
        "ComfyUI is running from {} (process {}, {}). Stop it with `rocm comfyui stop`",
        source_path.display(),
        state.pid,
        state.url
    )
}

pub(crate) fn stop(paths: &AppPaths) -> Result<String> {
    let Some(state) = load_state(paths)? else {
        return Ok(format!(
            "{APP_NAME}\n  status: stopped\n  note: ComfyUI was not running\n"
        ));
    };
    let report = evaluate_running_state(&state);
    if report.process_running {
        terminate_process_tree(state.pid)?;
        wait_until_stopped(state.pid);
    }
    let _ = fs::remove_file(state_path(paths));

    let mut output = String::new();
    writeln!(output, "{APP_NAME}")?;
    writeln!(output, "  status: stopped")?;
    writeln!(output, "  url: {}", state.url)?;
    Ok(output)
}

fn spawn_comfyui_background(
    paths: &AppPaths,
    manifest: &ComfyUiManifest,
    options: &ComfyUiStartOptions,
    runtime_env: &ComfyUiRuntimeEnvironment,
    log_path: &Path,
) -> Result<u32> {
    if runtime_is_windows() {
        return spawn_comfyui_background_windows(paths, manifest, options, runtime_env, log_path);
    }
    spawn_comfyui_background_unix(manifest, options, runtime_env, log_path)
}

fn spawn_comfyui_background_unix(
    manifest: &ComfyUiManifest,
    options: &ComfyUiStartOptions,
    runtime_env: &ComfyUiRuntimeEnvironment,
    log_path: &Path,
) -> Result<u32> {
    let log = fs::OpenOptions::new()
        .append(true)
        .open(log_path)
        .with_context(|| format!("failed to open {}", log_path.display()))?;
    let stdout = log
        .try_clone()
        .context("failed to clone ComfyUI log file")?;
    let stderr = log
        .try_clone()
        .context("failed to clone ComfyUI log file")?;
    let mut command = Command::new(&manifest.python_executable);
    command
        .current_dir(&manifest.source_path)
        .arg("main.py")
        .arg("--listen")
        .arg(&options.host)
        .arg("--port")
        .arg(options.port.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    apply_runtime_environment(&mut command, runtime_env)?;
    let child = command.spawn().with_context(|| {
        format!(
            "failed to start ComfyUI with {}",
            manifest.python_executable.display()
        )
    })?;
    Ok(child.id())
}

fn spawn_comfyui_background_windows(
    paths: &AppPaths,
    manifest: &ComfyUiManifest,
    options: &ComfyUiStartOptions,
    runtime_env: &ComfyUiRuntimeEnvironment,
    log_path: &Path,
) -> Result<u32> {
    let state_dir = app_root(paths).join("state");
    fs::create_dir_all(&state_dir)
        .with_context(|| format!("failed to create {}", state_dir.display()))?;
    let runner_path = state_dir.join("run-comfyui.ps1");
    let launcher_path = state_dir.join("launch-comfyui.ps1");
    let pid_path = state_dir.join("comfyui-launch.pid");

    let mut runner = String::new();
    writeln!(runner, "$ErrorActionPreference = 'Continue'")?;
    writeln!(
        runner,
        "Set-Location -LiteralPath {}",
        powershell_quote(&child_path_string(&manifest.source_path))
    )?;
    for (key, value) in runtime_environment_assignments(runtime_env)? {
        if key == "PATH" {
            continue;
        }
        writeln!(
            runner,
            "$env:{} = {}",
            key,
            powershell_quote(&value.to_string_lossy())
        )?;
    }
    let mut path_prefix_entries = runtime_env.path_entries.clone();
    path_prefix_entries.extend(runtime_env.library_entries.iter().cloned());
    let path_prefix = path_prefix_entries
        .iter()
        .map(|path| child_path_string(path))
        .collect::<Vec<_>>()
        .join(";");
    if !path_prefix.is_empty() {
        writeln!(
            runner,
            "$env:PATH = {} + ';' + $env:PATH",
            powershell_quote(&path_prefix)
        )?;
    }
    let mut command_line = powershell_quote(&child_path_string(&manifest.python_executable));
    for arg in [
        "main.py",
        "--listen",
        options.host.as_str(),
        "--port",
        &options.port.to_string(),
    ] {
        write!(command_line, " {}", powershell_quote(arg))?;
    }
    writeln!(
        runner,
        "& {} *>> {}",
        command_line,
        powershell_quote(&child_path_string(log_path))
    )?;
    fs::write(&runner_path, runner)
        .with_context(|| format!("failed to write {}", runner_path.display()))?;

    let launcher = format!(
        "$p = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',{}) -WindowStyle Hidden -PassThru\nSet-Content -LiteralPath {} -Value $p.Id -Encoding ASCII\n",
        powershell_quote(&child_path_string(&runner_path)),
        powershell_quote(&child_path_string(&pid_path))
    );
    fs::write(&launcher_path, launcher)
        .with_context(|| format!("failed to write {}", launcher_path.display()))?;

    let mut command = Command::new("powershell.exe");
    command
        .arg("-NoProfile")
        .arg("-ExecutionPolicy")
        .arg("Bypass")
        .arg("-File")
        .arg(&launcher_path);
    let output = capture_configured_command(
        command,
        &format!("run ComfyUI launcher {}", launcher_path.display()),
    )?;
    let pid_text = fs::read_to_string(&pid_path).unwrap_or_default();
    let _ = fs::remove_file(&pid_path);
    if !output.status.success() && pid_text.trim().is_empty() {
        bail!(
            "failed to launch ComfyUI in the background: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let pid = pid_text
        .split_whitespace()
        .next()
        .context("PowerShell launcher did not return a process id")?
        .parse::<u32>()
        .context("PowerShell launcher returned an invalid process id")?;
    Ok(pid)
}

pub(super) fn evaluate_running_state(state: &ComfyUiState) -> ComfyUiRunReport {
    let process_running = process_is_running(state.pid);
    let endpoint_reachable = endpoint_is_reachable(&state.host, state.port);
    let state = if endpoint_reachable {
        ComfyUiRunState::Running
    } else if process_running {
        ComfyUiRunState::Starting
    } else {
        ComfyUiRunState::Stopped
    };
    ComfyUiRunReport {
        state,
        process_running,
        endpoint_reachable,
    }
}

fn wait_for_running_state(state: &ComfyUiState, timeout: Duration) -> ComfyUiRunReport {
    let deadline = std::time::Instant::now() + timeout;
    let mut last_report = evaluate_running_state(state);
    while std::time::Instant::now() < deadline {
        if last_report.state == ComfyUiRunState::Running {
            return last_report;
        }
        thread::sleep(Duration::from_millis(500));
        last_report = evaluate_running_state(state);
    }
    last_report
}

pub(super) const fn comfyui_run_state_cli_label(state: ComfyUiRunState) -> &'static str {
    match state {
        ComfyUiRunState::Running => "running",
        ComfyUiRunState::Starting => "starting",
        ComfyUiRunState::Stopped => "stopped",
    }
}

pub(super) fn endpoint_is_reachable(host: &str, port: u16) -> bool {
    let Ok(addresses) = (host, port).to_socket_addrs() else {
        return false;
    };
    addresses
        .into_iter()
        .any(|address| TcpStream::connect_timeout(&address, Duration::from_millis(200)).is_ok())
}

fn process_is_running(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    if runtime_is_windows() {
        return windows_process_is_running(pid);
    }
    if runtime_is_linux() {
        return unix_process_is_running(pid);
    }
    true
}

fn windows_process_is_running(pid: u32) -> bool {
    let filter = format!("PID eq {pid}");
    let mut command = Command::new("tasklist");
    command.args(["/FI", &filter, "/FO", "CSV", "/NH"]);
    let Ok(output) = capture_configured_command(command, "check ComfyUI process") else {
        return false;
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.contains(&format!("\"{pid}\"")) {
        return true;
    }
    if !output.status.success() {
        return false;
    }
    false
}

fn unix_process_is_running(pid: u32) -> bool {
    Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn wait_until_stopped(pid: u32) {
    for _ in 0..20 {
        if !process_is_running(pid) {
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }
}

fn terminate_process_tree(pid: u32) -> Result<()> {
    if !process_is_running(pid) {
        return Ok(());
    }
    if runtime_is_windows() {
        return terminate_windows_process_tree(pid);
    }
    if runtime_is_linux() {
        return terminate_unix_process(pid);
    }
    Ok(())
}

fn terminate_windows_process_tree(pid: u32) -> Result<()> {
    let status = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .context("failed to run taskkill for ComfyUI")?;
    if !status.success() && process_is_running(pid) {
        bail!("failed to stop ComfyUI process {pid}");
    }
    Ok(())
}

fn terminate_unix_process(pid: u32) -> Result<()> {
    let status = Command::new("kill")
        .arg(pid.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .context("failed to run kill for ComfyUI")?;
    if !status.success() && process_is_running(pid) {
        bail!("failed to stop ComfyUI process {pid}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::path::PathBuf;

    #[test]
    fn running_state_requires_saved_process_and_reachable_port() -> Result<()> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let state = ComfyUiState {
            app_id: APP_ID.to_owned(),
            url: format!("http://127.0.0.1:{port}"),
            host: "127.0.0.1".to_owned(),
            port,
            pid: std::process::id(),
            source_path: PathBuf::from("source"),
            python_executable: PathBuf::from("python"),
            log_path: PathBuf::from("start.log"),
            started_at_unix_ms: 200,
        };

        let report = evaluate_running_state(&state);

        assert_eq!(report.state, ComfyUiRunState::Running);
        assert!(report.process_running);
        assert!(report.endpoint_reachable);
        Ok(())
    }
}
