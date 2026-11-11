// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Starting, stopping and watching the ComfyUI process this CLI launches.
//!
//! [`start`] checks the installed folder, runs the GPU check, launches ComfyUI
//! in the background (a detached child on Linux, a hidden PowerShell launcher on
//! Windows) and waits for its port to answer; [`stop`] ends the process tree the
//! saved state names, or sets aside a saved state that cannot be parsed
//! without stopping anything. [`evaluate_running_state`] is how `status` and the
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
    ComfyUiRuntimeEnvironment, ComfyUiStartOptions, ComfyUiState, DefaultAddressCheck,
    SavedRunState, UnreadableRunState, app_root, capture_configured_command, check_default_address,
    child_path_string, interrupted_reinstall_advice, load_manifest, models_folder_for_manifest,
    probe_comfyui, read_saved_state, save_state, something_answers_note, source_swap,
    start_log_path_for_manifest, state_path, unreadable_next_step, unreadable_state_aside_path,
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
/// it. A saved state that cannot be parsed counts as not running, and
/// `warnings` says so: it names no process, so `stop` cannot stop one either
/// (it only sets the record aside), and a refusal would point at nothing that
/// could end. A saved state that cannot be read at all is an error. What a
/// running ComfyUI risks is the code it runs; the preserved folders are left
/// alone whatever is running.
pub(super) fn running_from(
    paths: &AppPaths,
    source_path: &Path,
    warnings: &mut Vec<String>,
) -> Result<Option<ComfyUiState>> {
    let state = match read_saved_state(paths)? {
        SavedRunState::Saved(state) => state,
        SavedRunState::Absent => return Ok(None),
        SavedRunState::Unreadable(unreadable) => {
            warnings.push(format!(
                "{}; it names no process `rocm comfyui stop` could stop, so the reinstall of {} \
                 is not refused for it, and a ComfyUI still running from there may have its code \
                 replaced under it",
                unreadable.note(),
                source_path.display()
            ));
            return Ok(None);
        }
    };
    Ok((same_path_text(&state.source_path, source_path)
        && evaluate_running_state(&state).state != ComfyUiRunState::Stopped)
        .then_some(state))
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

/// What `rocm comfyui stop` prints, and the error it exits with after
/// printing it when the stop did not reach "stopped".
#[derive(Debug)]
pub(crate) struct StopReport {
    pub(crate) text: String,
    pub(crate) failure: Option<String>,
}

pub(crate) fn stop(paths: &AppPaths) -> Result<StopReport> {
    stop_with(paths, check_default_address)
}

/// [`stop`] with the check of ComfyUI's default address passed in, so tests
/// decide what answers there.
pub(super) fn stop_with(
    paths: &AppPaths,
    default_address_check: impl FnOnce() -> DefaultAddressCheck,
) -> Result<StopReport> {
    let state = match read_saved_state(paths)? {
        SavedRunState::Saved(state) => state,
        SavedRunState::Absent => {
            return Ok(StopReport {
                text: format!("{APP_NAME}\n  status: stopped\n  note: ComfyUI was not running\n"),
                failure: None,
            });
        }
        SavedRunState::Unreadable(unreadable) => {
            return stop_over_unreadable_state(
                &unreadable,
                &unreadable_state_aside_path(paths),
                default_address_check(),
            );
        }
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
    Ok(StopReport {
        text: output,
        failure: None,
    })
}

/// A record that cannot be parsed names no process, so nothing is stopped: a
/// guess could end a process rocm-cli did not start. Only the default address
/// is checked. If something answers there, the record is left in place and the
/// stop fails, naming what to do; otherwise the record is moved to `aside`
/// (replacing one moved there before), which stops `status` reporting an
/// unknown running state while keeping it to look at.
fn stop_over_unreadable_state(
    unreadable: &UnreadableRunState,
    aside: &Path,
    default_address: DefaultAddressCheck,
) -> Result<StopReport> {
    let note = unreadable.note();
    let mut output = String::new();
    writeln!(output, "{APP_NAME}")?;
    writeln!(output, "  status: unknown")?;
    writeln!(output, "  note: {note}")?;
    if default_address.answers {
        let url = default_address.url;
        let answering = something_answers_note(&url);
        writeln!(output, "  url: {url}")?;
        writeln!(output, "  note: {answering}")?;
        writeln!(
            output,
            "  note: no process was stopped, and the record was left in place"
        )?;
        let next_step = unreadable_next_step(&unreadable.path, Some(&url));
        writeln!(output, "  next step: {next_step}")?;
        return Ok(StopReport {
            text: output,
            failure: Some(format!(
                "{answering}; no process was stopped and the record was left in place: \
                 {next_step}"
            )),
        });
    }
    // A rename by path: a `start` that saves a new record between the read
    // above and this rename would have that record moved aside instead. That
    // race is accepted; `start` takes no lock.
    let moved = match fs::rename(&unreadable.path, aside) {
        Ok(()) => {
            writeln!(output, "  moved aside to: {}", aside.display())?;
            true
        }
        // Another `stop` (or the user) got there first.
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            writeln!(
                output,
                "  moved aside: nothing; the record was already gone"
            )?;
            false
        }
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "{note}; failed to move it to {}, and no process was stopped",
                    aside.display()
                )
            });
        }
    };
    write!(
        output,
        "  note: no process was stopped; rocm-cli checked only {}, where nothing answers, so a \
         ComfyUI started on another address is not covered",
        default_address.url
    )?;
    if moved {
        write!(
            output,
            "; the moved-aside record is kept at {}",
            aside.display()
        )?;
    }
    writeln!(output)?;
    Ok(StopReport {
        text: output,
        failure: None,
    })
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
    let mut command = Command::new(rocm_core::windows_system_tool("tasklist.exe"));
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
    let status = Command::new(rocm_core::windows_system_tool("taskkill.exe"))
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

    /// Over an unreadable record with nothing answering on the default
    /// address, `stop` stops nothing, moves the record aside (replacing one
    /// moved there before) and succeeds. The first truncated record names a
    /// live stand-in process before it was cut off, and that process must
    /// still be running afterwards. It is also what `status`'s next step
    /// promises: afterwards `status` no longer reports the record.
    #[test]
    fn stop_sets_an_unreadable_record_aside_and_stops_no_process() -> Result<()> {
        use crate::comfyui::render_status_with;
        use crate::comfyui::tests::{
            UNREADABLE_RECORDS, plant_unreadable_record, quiet_default_address, remove_test_root,
            spawn_stand_in_process, test_paths,
        };
        use rocm_core::RocmCliConfig;

        fn stop_over_records_naming(pid: u32) -> Result<()> {
            let truncated_naming_this_process = format!(
                "{{\"pid\": {pid}, {}",
                UNREADABLE_RECORDS[0].trim_start_matches('{')
            );
            for (index, text) in [
                truncated_naming_this_process.as_str(),
                UNREADABLE_RECORDS[0],
                UNREADABLE_RECORDS[1],
            ]
            .into_iter()
            .enumerate()
            {
                let paths = test_paths(&format!("comfyui-stop-unreadable-{index}"));
                let path = plant_unreadable_record(&paths, text)?;
                let aside = unreadable_state_aside_path(&paths);
                if index == 1 {
                    // One moved aside by an earlier stop is replaced.
                    fs::write(&aside, "an earlier record")?;
                }
                let config = RocmCliConfig::default();
                let outcome = render_status_with(&paths, &config, quiet_default_address).and_then(
                    |status_before| {
                        let report = stop_with(&paths, quiet_default_address)?;
                        let status_after =
                            render_status_with(&paths, &config, quiet_default_address)?;
                        Ok((
                            status_before,
                            report,
                            status_after,
                            path.exists(),
                            fs::read_to_string(&aside).ok(),
                        ))
                    },
                );
                remove_test_root(&paths);
                let (status_before, report, status_after, record_left, aside_text) = outcome?;

                assert!(
                status_before.contains(
                    "next step: run `rocm comfyui stop`, which moves the unreadable record aside and \
                     stops no process\n"
                ),
                "{status_before}"
            );
                assert!(report.failure.is_none(), "{report:?}");
                let expected = format!(
                    "{APP_NAME}\n  status: unknown\n  note: rocm-cli's record of the ComfyUI it \
                 started cannot be parsed ({}: ",
                    path.display()
                );
                assert!(report.text.starts_with(&expected), "{}", report.text);
                assert!(
                report.text.ends_with(&format!(
                    "  moved aside to: {aside}\n  note: no process was stopped; rocm-cli checked \
                     only http://127.0.0.1:8188, where nothing answers, so a ComfyUI started on \
                     another address is not covered; the moved-aside record is kept at {aside}\n",
                    aside = aside.display()
                )),
                "{}",
                report.text
            );
                assert!(!report.text.contains("status: stopped"), "{}", report.text);
                assert!(!record_left, "stop must move the unreadable record away");
                assert_eq!(
                    aside_text.as_deref(),
                    Some(text),
                    "the record is kept aside"
                );
                assert!(!status_after.contains("status: unknown"), "{status_after}");
                assert!(!status_after.contains("cannot be parsed"), "{status_after}");
            }
            Ok(())
        }

        let mut stand_in = spawn_stand_in_process()?;
        let outcome = stop_over_records_naming(stand_in.id());
        let still_running = stand_in.try_wait().map(|status| status.is_none());
        let _ = stand_in.kill();
        let _ = stand_in.wait();
        outcome?;
        assert!(
            still_running?,
            "stop must not end the process a cut-off record named"
        );
        Ok(())
    }

    /// When something answers on the default address, `stop` over an
    /// unreadable record leaves the record in place, says only that address
    /// was checked, does not claim it stopped anything, and fails. Its advice
    /// is then true either way: once nothing answers there, running `stop`
    /// again sets the record aside, and a record the user moves aside
    /// themselves is no longer reported.
    #[test]
    fn stop_over_an_unreadable_record_keeps_it_while_something_answers() -> Result<()> {
        use crate::comfyui::tests::{
            UNREADABLE_RECORDS, answering_default_address, plant_unreadable_record,
            quiet_default_address, remove_test_root, test_paths,
        };

        for (index, text) in UNREADABLE_RECORDS.iter().enumerate() {
            let paths = test_paths(&format!("comfyui-stop-unreadable-answering-{index}"));
            let path = plant_unreadable_record(&paths, text)?;
            let aside = unreadable_state_aside_path(&paths);

            let outcome = stop_with(&paths, answering_default_address).and_then(|refused| {
                let kept = fs::read_to_string(&path).ok();
                let aside_existed = aside.exists();
                // Way out one: what answered was the ComfyUI, the user
                // stopped it, and `stop` runs again.
                let again = stop_with(&paths, quiet_default_address)?;
                let after_again = (
                    path.exists(),
                    fs::read_to_string(&aside).ok(),
                    crate::comfyui::render_status_with(
                        &paths,
                        &rocm_core::RocmCliConfig::default(),
                        quiet_default_address,
                    )?,
                );
                // Way out two: what answers is not it, so the user moves the
                // record aside by hand while it still answers.
                let by_hand = path.with_file_name("moved-by-hand.json");
                plant_unreadable_record(&paths, text)?;
                let before_by_hand = crate::comfyui::render_status_with(
                    &paths,
                    &rocm_core::RocmCliConfig::default(),
                    answering_default_address,
                )?;
                fs::rename(&path, &by_hand)?;
                let after_by_hand = crate::comfyui::render_status_with(
                    &paths,
                    &rocm_core::RocmCliConfig::default(),
                    answering_default_address,
                )?;
                Ok((
                    refused,
                    kept,
                    aside_existed,
                    again,
                    after_again,
                    before_by_hand,
                    after_by_hand,
                ))
            });
            remove_test_root(&paths);
            let (
                refused,
                kept,
                aside_existed,
                again,
                (left_after_again, aside_after_again, status_after_again),
                before_by_hand,
                after_by_hand,
            ) = outcome?;

            let failure = refused.failure.as_deref().context("the stop must fail")?;
            let answering = "something answers at http://127.0.0.1:8188, the only address \
                             rocm-cli checked; it cannot tell whether that is the ComfyUI it \
                             started";
            let next_step = format!(
                "stop what answers at http://127.0.0.1:8188 yourself, then run `rocm comfyui \
                 stop` again; if what answers there is not the ComfyUI rocm-cli started, move {} \
                 aside yourself instead",
                path.display()
            );
            assert_eq!(
                failure,
                format!(
                    "{answering}; no process was stopped and the record was left in place: \
                     {next_step}"
                )
            );
            assert!(
                refused.text.ends_with(&format!(
                    "  url: http://127.0.0.1:8188\n  note: {answering}\n  note: no process was \
                     stopped, and the record was left in place\n  next step: {next_step}\n"
                )),
                "{}",
                refused.text
            );
            assert!(
                refused.text.contains("  status: unknown\n"),
                "{}",
                refused.text
            );
            assert_eq!(kept.as_deref(), Some(*text), "the record stays in place");
            assert!(!aside_existed, "nothing is moved aside");
            // Way out one: the second stop sets the record aside, bytes intact.
            assert!(again.failure.is_none(), "{again:?}");
            assert!(!left_after_again, "the second stop moves the record away");
            assert_eq!(
                aside_after_again.as_deref(),
                Some(*text),
                "the second stop keeps the record aside, unchanged"
            );
            assert!(
                status_after_again.contains("  status: not started by rocm-cli\n"),
                "{status_after_again}"
            );
            // Way out two: unknown until the user moves it, then not reported.
            assert!(
                before_by_hand.contains("  status: unknown\n"),
                "{before_by_hand}"
            );
            assert!(
                !after_by_hand.contains("status: unknown")
                    && !after_by_hand.contains("cannot be parsed"),
                "{after_by_hand}"
            );
            assert!(
                after_by_hand.contains("  status: running outside rocm-cli\n"),
                "{after_by_hand}"
            );
        }
        Ok(())
    }

    /// A record that cannot be moved aside (a directory stands where it
    /// would go, which stops `root` too) makes `stop` fail naming the record
    /// and the destination, and the record stays in place.
    #[test]
    fn stop_reports_an_unreadable_record_it_cannot_move_aside() -> Result<()> {
        use crate::comfyui::tests::{
            UNREADABLE_RECORDS, plant_unreadable_record, quiet_default_address, remove_test_root,
            test_paths,
        };

        let paths = test_paths("comfyui-stop-unreadable-blocked");
        let path = plant_unreadable_record(&paths, UNREADABLE_RECORDS[0])?;
        let aside = unreadable_state_aside_path(&paths);
        fs::create_dir_all(aside.join("occupied"))?;

        let outcome = stop_with(&paths, quiet_default_address);
        let kept = fs::read_to_string(&path).ok();
        remove_test_root(&paths);

        let error = outcome.expect_err("a record that cannot be moved aside fails the stop");
        let message = format!("{error:#}");
        assert!(
            message.starts_with(&format!(
                "rocm-cli's record of the ComfyUI it started cannot be parsed ({}: ",
                path.display()
            )),
            "{message}"
        );
        assert!(
            message.contains(&format!(
                "; failed to move it to {}, and no process was stopped",
                aside.display()
            )),
            "{message}"
        );
        assert_eq!(kept.as_deref(), Some(UNREADABLE_RECORDS[0]));
        Ok(())
    }

    /// A second `stop` racing the first finds the record already moved; it
    /// says so and succeeds rather than failing on the missing file.
    #[test]
    fn stop_over_an_unreadable_record_already_moved_says_so_and_succeeds() -> Result<()> {
        use crate::comfyui::tests::{quiet_default_address, remove_test_root, test_paths};

        let paths = test_paths("comfyui-stop-unreadable-gone");
        let path = state_path(&paths);
        let aside = unreadable_state_aside_path(&paths);
        fs::create_dir_all(path.parent().context("no parent")?)?;
        let unreadable = UnreadableRunState {
            path,
            cause: "EOF while parsing a value at line 1 column 0".to_owned(),
        };

        let report = stop_over_unreadable_state(&unreadable, &aside, quiet_default_address());
        let aside_exists = aside.exists();
        remove_test_root(&paths);

        let report = report?;
        assert!(report.failure.is_none(), "{report:?}");
        assert!(
            report
                .text
                .contains("  moved aside: nothing; the record was already gone\n"),
            "{}",
            report.text
        );
        assert!(
            report.text.ends_with("another address is not covered\n"),
            "nothing was moved, so no kept record is named: {}",
            report.text
        );
        assert!(!aside_exists);
        Ok(())
    }
}
