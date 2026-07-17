// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[cfg(windows)]
use std::ffi::OsStr;
use std::ffi::OsString;
use std::fs;
use std::io::{IsTerminal, Read, Write, stdin, stdout};
use std::net::{IpAddr, TcpStream, ToSocketAddrs};
#[cfg(windows)]
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
#[cfg(windows)]
use windows_sys::Win32::System::Threading::{
    CREATE_NEW_CONSOLE, CREATE_NEW_PROCESS_GROUP, CREATE_NO_WINDOW, CREATE_UNICODE_ENVIRONMENT,
    CreateProcessW, DETACHED_PROCESS, GetExitCodeProcess, INFINITE, OpenProcess,
    PROCESS_INFORMATION, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE, PROCESS_TERMINATE,
    STARTF_USESHOWWINDOW, STARTF_USESTDHANDLES, STARTUPINFOW, TerminateProcess,
    WaitForSingleObject,
};

pub mod diagnose;
pub mod examine;
pub mod fix;
pub mod openmpi;
pub mod runtime;
pub mod uv;
pub use diagnose::{
    DiagnoseReport, Diagnosis, Fix, diagnose as run_diagnose,
    render_report_text as render_diagnose_text,
};
pub use examine::{Examination, FrameworkProbe, WSL_ROUTE_OUT_NOTE};
pub use fix::{FixOptions, apply as apply_fix, list_recipes as list_fix_recipes};
use runtime::env_path_override;
#[cfg(test)]
use runtime::home_rocm_dir;
pub use runtime::{
    RuntimeHost, RuntimePlatform, current_executable_path, default_cache_dir, default_config_dir,
    default_data_dir, default_interactive_shell_program, managed_logs_dir, managed_pip_cache_dir,
    managed_runtime_cache_dir, managed_tools_dir, normalize_runtime_path_for_host,
    normalize_runtime_path_for_storage, normalize_runtime_path_text_for_host,
    normalize_runtime_path_text_for_platform, normalize_runtime_path_text_for_storage,
    platform_binary_name, prepend_runtime_path, runtime_directory_label,
    runtime_drive_root_for_key, runtime_drive_roots, runtime_exe_suffix, runtime_home_dir,
    runtime_install_root_is_protected, runtime_is_linux, runtime_is_windows, runtime_os_name,
    runtime_path_for_child, runtime_path_for_windows_child, runtime_path_is_same_or_inside,
    runtime_path_list_join, runtime_path_list_split, runtime_path_sort_key,
    runtime_path_text_is_absolute_for_host, runtime_path_text_is_absolute_for_platform,
    runtime_paths_equivalent, runtime_python_activation_hint, runtime_python_activation_script,
    runtime_python_bin_dir_name, runtime_python_env_bin_dir, runtime_python_executable_in_env,
    runtime_python_executable_name, runtime_rocm_library_filename, shell_command_for_host,
};
pub use uv::{
    DEFAULT_UV_TIMEOUT_SECS, ensure_uv_binary, uv_binary_name, uv_command_env,
    uv_http_timeout_secs, uv_pip_freeze_args, uv_pip_install_base, uv_venv_args,
};

pub const DEFAULT_LOCAL_PORT: u16 = 11_435;
pub const DEFAULT_LOCAL_HOST: &str = "127.0.0.1";
const OPTIONAL_COMMAND_TIMEOUT: Duration = Duration::from_millis(1_500);
const WINDOWS_INVENTORY_QUERY_TIMEOUT: Duration = Duration::from_secs(5);
const WINDOWS_VIDEO_CONTROLLER_INVENTORY_SCRIPT: &str = r#"$gpus = Get-CimInstance -ClassName Win32_VideoController -Property Name,DriverVersion,PNPDeviceID,AdapterCompatibility | Where-Object { $_.PNPDeviceID -match 'VEN_1002' -or $_.AdapterCompatibility -match 'AMD|Advanced Micro Devices' -or $_.Name -match 'AMD|Radeon|Instinct' }; foreach ($gpu in $gpus) { "GPU`t$($gpu.Name)`t$($gpu.DriverVersion)`t$($gpu.PNPDeviceID)" }"#;
#[cfg(windows)]
const WINDOWS_PNP_ENTITY_INVENTORY_SCRIPT: &str = r#"$displayGuid = '{4d36e968-e325-11ce-bfc1-08002be10318}'; $gpus = Get-CimInstance -ClassName Win32_PnPEntity -Property Name,DeviceID,PNPClass,ClassGuid,Manufacturer | Where-Object { (($_.PNPClass -eq 'Display' -or $_.ClassGuid -eq $displayGuid) -and ($_.DeviceID -match 'VEN_1002' -or $_.Name -match 'AMD|Radeon|Instinct|Graphics' -or $_.Manufacturer -match 'AMD|Advanced Micro Devices')) -or ($_.DeviceID -match 'PCI\\VEN_1002' -and $_.Name -match 'Radeon|Instinct|Graphics') }; foreach ($gpu in $gpus) { "GPU`t$($gpu.Name)`t`t$($gpu.DeviceID)" }"#;
#[cfg(windows)]
const WINDOWS_SYSTEM_INVENTORY_SCRIPT: &str = r#"$cpu = Get-CimInstance -ClassName Win32_Processor -Property Name | Select-Object -First 1 -ExpandProperty Name; if ($cpu) { "CPU`t$cpu" }; $ram = Get-CimInstance -ClassName Win32_ComputerSystem -Property TotalPhysicalMemory | Select-Object -First 1 -ExpandProperty TotalPhysicalMemory; if ($ram) { "RAM`t$ram" }"#;

pub fn format_host_for_url(host: &str) -> String {
    let trimmed = host.trim();
    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        return trimmed.to_owned();
    }
    match trimmed.parse::<IpAddr>() {
        Ok(IpAddr::V6(_)) => format!("[{trimmed}]"),
        _ => trimmed.to_owned(),
    }
}

pub fn format_host_port(host: &str, port: u16) -> String {
    format!("{}:{port}", format_host_for_url(host))
}

pub fn format_http_base_url(host: &str, port: u16) -> String {
    format!("http://{}", format_host_port(host, port))
}

pub fn parse_http_endpoint(endpoint_url: &str) -> Option<(String, u16)> {
    let without_scheme = endpoint_url.trim().strip_prefix("http://")?;
    let authority = without_scheme.split('/').next()?.trim();
    if authority.is_empty() {
        return None;
    }
    if let Some(rest) = authority.strip_prefix('[') {
        let end = rest.find(']')?;
        let host = rest[..end].to_owned();
        let port = rest[end + 1..].strip_prefix(':')?.parse().ok()?;
        return Some((host, port));
    }
    let (host, port) = authority.rsplit_once(':')?;
    Some((host.to_owned(), port.parse().ok()?))
}

pub fn download_file_to_path(url: &str, destination: &Path, timeout: Duration) -> Result<()> {
    let response = ureq::get(url)
        .timeout(timeout)
        .call()
        .with_context(|| format!("failed to download {url}"))?;
    if response.status() != 200 {
        bail!("HTTP {} while downloading {url}", response.status());
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let mut reader = response.into_reader();
    let mut file = fs::File::create(destination)
        .with_context(|| format!("failed to create {}", destination.display()))?;
    std::io::copy(&mut reader, &mut file)
        .with_context(|| format!("failed to write {}", destination.display()))?;
    Ok(())
}

pub fn http_get_text(endpoint_url: &str, path: &str, timeout: Duration) -> Result<String> {
    let (host, port) = parse_http_endpoint(endpoint_url)
        .with_context(|| format!("unsupported endpoint URL `{endpoint_url}`"))?;
    let mut stream = connect_tcp_stream(&host, port, timeout)?;
    let host_header = format_host_port(&host, port);
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {host_header}\r\nAccept: application/json\r\nConnection: close\r\n\r\n"
    );
    write_all_tcp_stream(&mut stream, request.as_bytes())
        .with_context(|| format!("failed to write HTTP GET {path}"))?;
    let response = read_tcp_stream_to_string(&mut stream)
        .with_context(|| format!("failed to read HTTP GET {path}"))?;
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .context("HTTP response was missing a body")?;
    let status_line = headers.lines().next().unwrap_or_default();
    if !status_line.contains(" 200 ") {
        bail!("HTTP endpoint returned {status_line}");
    }
    Ok(body.to_owned())
}

pub fn openai_models_endpoint_has_model(
    endpoint_url: &str,
    expected_model: Option<&str>,
    timeout: Duration,
) -> Result<bool> {
    let body = http_get_text(endpoint_url, "/v1/models", timeout)?;
    let value = serde_json::from_str::<serde_json::Value>(body.trim())
        .context("failed to parse /v1/models JSON")?;
    let loaded_models = openai_loaded_model_ids(&value);
    if loaded_models.is_empty() {
        return Ok(false);
    }
    let Some(expected_model) = expected_model.filter(|value| !value.trim().is_empty()) else {
        return Ok(true);
    };
    Ok(loaded_models
        .iter()
        .any(|loaded| model_refs_match(loaded, expected_model)))
}

pub fn managed_service_endpoint_model_ready(
    record: &ManagedServiceRecord,
    timeout: Duration,
) -> Result<bool> {
    if record.endpoint_url.trim().is_empty() {
        return Ok(false);
    }
    let expected = if !record.canonical_model_id.trim().is_empty() {
        Some(record.canonical_model_id.as_str())
    } else if !record.model_ref.trim().is_empty() {
        Some(record.model_ref.as_str())
    } else {
        None
    };
    openai_models_endpoint_has_model(&record.endpoint_url, expected, timeout)
}

fn openai_loaded_model_ids(value: &serde_json::Value) -> Vec<String> {
    value
        .get("data")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| {
            ["id", "model", "name"]
                .into_iter()
                .filter_map(|field| item.get(field).and_then(serde_json::Value::as_str))
                .find(|value| !value.trim().is_empty())
                .map(str::to_owned)
        })
        .collect()
}

fn model_refs_match(left: &str, right: &str) -> bool {
    let left = left.trim();
    let right = right.trim();
    if left.eq_ignore_ascii_case(right) || model_ref_basename(left).eq_ignore_ascii_case(right) {
        return true;
    }
    if model_ref_basename(right).eq_ignore_ascii_case(left)
        || model_ref_basename(left).eq_ignore_ascii_case(model_ref_basename(right))
    {
        return true;
    }
    builtin_model_recipes().into_iter().any(|recipe| {
        (recipe.matches_ref(left) || recipe.matches_ref(right))
            && (recipe.matches_ref(left) && recipe.matches_ref(right))
    }) || model_ref_family_matches(left, right)
        || model_ref_family_matches(right, left)
}

fn model_ref_basename(value: &str) -> &str {
    value
        .trim()
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or_else(|| value.trim())
}

fn model_ref_family_matches(reported: &str, expected_family: &str) -> bool {
    let expected = normalize_model_ref_family(expected_family);
    if expected.len() < 3 || expected.chars().any(|ch| ch.is_ascii_digit()) {
        return false;
    }
    model_ref_tokens(reported)
        .into_iter()
        .any(|token| token == expected || token.starts_with(&expected))
}

fn normalize_model_ref_family(value: &str) -> String {
    value
        .trim()
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .flat_map(char::to_lowercase)
        .collect()
}

fn model_ref_tokens(value: &str) -> Vec<String> {
    value
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter_map(|token| {
            let token = normalize_model_ref_family(token);
            (!token.is_empty()).then_some(token)
        })
        .collect()
}

pub fn connect_tcp_stream(host: &str, port: u16, timeout: Duration) -> Result<TcpStream> {
    let addr = (host, port)
        .to_socket_addrs()
        .with_context(|| format!("failed to resolve {host}:{port}"))?
        .next()
        .with_context(|| format!("no socket addresses resolved for {host}:{port}"))?;
    let stream =
        TcpStream::connect(addr).with_context(|| format!("failed to connect to {host}:{port}"))?;
    stream.set_read_timeout(Some(timeout)).ok();
    stream.set_write_timeout(Some(timeout)).ok();
    Ok(stream)
}

pub fn write_all_tcp_stream(stream: &mut TcpStream, bytes: &[u8]) -> Result<()> {
    stream
        .write_all(bytes)
        .context("failed to write to TCP stream")
}

pub fn read_tcp_stream_to_string(stream: &mut TcpStream) -> Result<String> {
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .context("failed to read TCP stream")?;
    Ok(response)
}

#[cfg(windows)]
pub fn spawn_detached_no_inherit(
    program: &Path,
    args: &[String],
    env_overrides: &[(&str, &Path)],
) -> Result<u32> {
    spawn_windows_no_inherit(
        program,
        args,
        env_overrides,
        DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT,
        false,
        None,
    )
}

#[cfg(windows)]
pub fn spawn_hidden_console_no_inherit(
    program: &Path,
    args: &[String],
    env_overrides: &[(&str, &Path)],
) -> Result<u32> {
    spawn_windows_no_inherit(
        program,
        args,
        env_overrides,
        CREATE_NEW_CONSOLE | CREATE_NEW_PROCESS_GROUP | CREATE_UNICODE_ENVIRONMENT,
        true,
        None,
    )
}

#[cfg(windows)]
#[allow(unsafe_code)] // Win32 FFI
pub fn spawn_hidden_console_with_log(
    program: &Path,
    args: &[String],
    env_overrides: &[(&str, &Path)],
    log_path: &Path,
) -> Result<u32> {
    use std::os::windows::io::AsRawHandle;
    use std::ptr::null_mut;
    use windows_sys::Win32::Foundation::{
        CloseHandle, DUPLICATE_SAME_ACCESS, DuplicateHandle, HANDLE,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let log_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .with_context(|| format!("failed to open {}", log_path.display()))?;
    let current_process = unsafe { GetCurrentProcess() };
    let source = log_file.as_raw_handle() as HANDLE;
    let mut stdout_handle: HANDLE = null_mut();
    let mut stderr_handle: HANDLE = null_mut();
    unsafe {
        if DuplicateHandle(
            current_process,
            source,
            current_process,
            &mut stdout_handle,
            0,
            1,
            DUPLICATE_SAME_ACCESS,
        ) == 0
        {
            bail!(
                "failed to duplicate stdout log handle for {}: {}",
                log_path.display(),
                std::io::Error::last_os_error()
            );
        }
        if DuplicateHandle(
            current_process,
            source,
            current_process,
            &mut stderr_handle,
            0,
            1,
            DUPLICATE_SAME_ACCESS,
        ) == 0
        {
            CloseHandle(stdout_handle);
            bail!(
                "failed to duplicate stderr log handle for {}: {}",
                log_path.display(),
                std::io::Error::last_os_error()
            );
        }
    }
    let result = spawn_windows_no_inherit(
        program,
        args,
        env_overrides,
        CREATE_NEW_CONSOLE | CREATE_NEW_PROCESS_GROUP | CREATE_UNICODE_ENVIRONMENT,
        true,
        Some((stdout_handle, stderr_handle)),
    );
    unsafe {
        CloseHandle(stdout_handle);
        CloseHandle(stderr_handle);
    }
    result
}

#[cfg(windows)]
#[allow(unsafe_code)] // Win32 FFI
pub fn wait_for_process_exit(pid: u32) -> Result<u32> {
    use windows_sys::Win32::Foundation::CloseHandle;

    let handle = unsafe {
        OpenProcess(
            PROCESS_SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
            0,
            pid,
        )
    };
    if handle.is_null() {
        bail!(
            "failed to open process {pid} for wait: {}",
            std::io::Error::last_os_error()
        );
    }
    unsafe {
        WaitForSingleObject(handle, INFINITE);
        let mut exit_code = 0;
        if GetExitCodeProcess(handle, &mut exit_code) == 0 {
            CloseHandle(handle);
            bail!(
                "failed to read process {pid} exit code: {}",
                std::io::Error::last_os_error()
            );
        }
        CloseHandle(handle);
        Ok(exit_code)
    }
}

#[cfg(windows)]
#[allow(unsafe_code)] // Win32 FFI
pub fn terminate_process(pid: u32) -> Result<()> {
    use windows_sys::Win32::Foundation::CloseHandle;

    let handle = unsafe { OpenProcess(PROCESS_TERMINATE, 0, pid) };
    if handle.is_null() {
        bail!(
            "failed to open process {pid} for termination: {}",
            std::io::Error::last_os_error()
        );
    }
    let terminated = unsafe { TerminateProcess(handle, 1) };
    unsafe {
        CloseHandle(handle);
    }
    if terminated == 0 {
        bail!(
            "failed to terminate process {pid}: {}",
            std::io::Error::last_os_error()
        );
    }
    Ok(())
}

#[cfg(not(windows))]
#[allow(unsafe_code)] // libc FFI
pub fn terminate_process(pid: u32) -> Result<()> {
    let status = unsafe { libc::kill(pid.cast_signed(), libc::SIGTERM) };
    if status == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
            .with_context(|| format!("failed to terminate process {pid}"))
    }
}

/// Terminate `pid` together with every transitive child process.
///
/// Long-running engines such as vLLM spawn helper subprocesses (for example the
/// `EngineCore` worker that holds the GPU allocation). Signalling only the
/// launcher PID leaves those workers reparented to init, where they keep the
/// model resident and the device memory pinned. Walking the descendant tree and
/// signalling each process avoids that leak.
#[cfg(not(windows))]
#[allow(unsafe_code)] // libc FFI
pub fn terminate_process_tree(pid: u32) -> Result<()> {
    let mut last_error: Option<(u32, std::io::Error)> = None;
    for target in collect_process_tree(pid) {
        let status = unsafe { libc::kill(target.cast_signed(), libc::SIGTERM) };
        if status != 0 {
            let error = std::io::Error::last_os_error();
            // A process that already exited (ESRCH) is not a failure here.
            if error.raw_os_error() != Some(libc::ESRCH) {
                last_error = Some((target, error));
            }
        }
    }
    if let Some((target, error)) = last_error {
        return Err(error).with_context(|| format!("failed to terminate process {target}"));
    }
    Ok(())
}

/// Collect `root` plus all of its transitive descendants by reading `/proc`.
///
/// On platforms without `/proc` (for example macOS) only `root` is returned, so
/// callers degrade to single-process termination rather than failing.
#[cfg(not(windows))]
fn collect_process_tree(root: u32) -> Vec<u32> {
    let mut children: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
    if let Ok(entries) = fs::read_dir("/proc") {
        for entry in entries.flatten() {
            let Ok(pid) = entry.file_name().to_string_lossy().parse::<u32>() else {
                continue;
            };
            if let Some(ppid) = read_parent_pid(pid) {
                children.entry(ppid).or_default().push(pid);
            }
        }
    }

    let mut order = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![root];
    while let Some(pid) = stack.pop() {
        if !seen.insert(pid) {
            continue;
        }
        order.push(pid);
        if let Some(kids) = children.get(&pid) {
            stack.extend(kids.iter().copied());
        }
    }
    order
}

/// Read the parent PID of `pid` from `/proc/<pid>/stat`.
///
/// The `comm` field can contain spaces and parentheses, so the parent PID is
/// parsed from the text after the final `)`.
#[cfg(not(windows))]
fn read_parent_pid(pid: u32) -> Option<u32> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let after_comm = stat.get(stat.rfind(')')? + 1..)?;
    let mut fields = after_comm.split_whitespace();
    let _state = fields.next()?;
    fields.next()?.parse::<u32>().ok()
}

/// Terminate `pid` together with every transitive child process.
///
/// The Windows implementation falls back to terminating the single process; the
/// engines that rely on descendant cleanup are Unix-only.
#[cfg(windows)]
pub fn terminate_process_tree(pid: u32) -> Result<()> {
    terminate_process(pid)
}

#[cfg(windows)]
#[allow(unsafe_code)] // Win32 FFI
pub fn process_is_running(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::CloseHandle;

    if pid == 0 {
        return false;
    }
    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if handle.is_null() {
        return false;
    }
    let mut exit_code = 0;
    let ok = unsafe { GetExitCodeProcess(handle, &mut exit_code) != 0 };
    unsafe {
        CloseHandle(handle);
    }
    ok && exit_code == 259
}

#[cfg(not(windows))]
#[allow(unsafe_code)] // libc FFI
pub fn process_is_running(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    let Ok(pid) = libc::pid_t::try_from(pid) else {
        return false;
    };
    let status = unsafe { libc::kill(pid, 0) };
    if status == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(unix)]
#[allow(unsafe_code)] // libc FFI (pre_exec/setsid)
pub fn detach_command_session(command: &mut Command) {
    use std::os::unix::process::CommandExt;

    unsafe {
        command.pre_exec(|| {
            if libc::setsid() < 0 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(())
            }
        });
    }
}

#[cfg(not(unix))]
pub fn detach_command_session(_command: &mut Command) {}

#[cfg(windows)]
#[allow(unsafe_code)] // Win32 FFI
fn spawn_windows_no_inherit(
    program: &Path,
    args: &[String],
    env_overrides: &[(&str, &Path)],
    creation_flags: u32,
    hide_window: bool,
    std_handles: Option<(
        windows_sys::Win32::Foundation::HANDLE,
        windows_sys::Win32::Foundation::HANDLE,
    )>,
) -> Result<u32> {
    use std::ptr::{null, null_mut};
    use windows_sys::Win32::Foundation::CloseHandle;

    let mut command_line = windows_command_line(program.as_os_str(), args);
    let application_name = nul_terminated_wide(program.as_os_str());
    let mut environment = windows_environment_block(env_overrides);
    let mut startup_info = STARTUPINFOW {
        cb: std::mem::size_of::<STARTUPINFOW>() as u32,
        ..Default::default()
    };
    if hide_window {
        const SW_HIDE: u16 = 0;
        startup_info.dwFlags |= STARTF_USESHOWWINDOW;
        startup_info.wShowWindow = SW_HIDE;
    }
    if let Some((stdout_handle, stderr_handle)) = std_handles {
        startup_info.dwFlags |= STARTF_USESTDHANDLES;
        startup_info.hStdInput = null_mut();
        startup_info.hStdOutput = stdout_handle;
        startup_info.hStdError = stderr_handle;
    }
    let mut process_info = PROCESS_INFORMATION::default();
    let created = unsafe {
        CreateProcessW(
            application_name.as_ptr(),
            command_line.as_mut_ptr(),
            null(),
            null(),
            if std_handles.is_some() { 1 } else { 0 },
            creation_flags,
            environment.as_mut_ptr().cast(),
            null(),
            &startup_info,
            &mut process_info,
        )
    };
    if created == 0 {
        bail!(
            "failed to launch detached process {}: {}",
            program.display(),
            std::io::Error::last_os_error()
        );
    }
    unsafe {
        CloseHandle(process_info.hThread);
        CloseHandle(process_info.hProcess);
    }
    Ok(process_info.dwProcessId)
}

#[cfg(windows)]
fn nul_terminated_wide(value: &OsStr) -> Vec<u16> {
    value.encode_wide().chain(std::iter::once(0)).collect()
}

#[cfg(windows)]
fn windows_command_line(program: &OsStr, args: &[String]) -> Vec<u16> {
    let mut command = quote_windows_arg(&program.to_string_lossy());
    for arg in args {
        command.push(' ');
        command.push_str(&quote_windows_arg(arg));
    }
    OsStr::new(&command)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect()
}

#[cfg(windows)]
fn quote_windows_arg(arg: &str) -> String {
    if !arg.is_empty()
        && !arg
            .chars()
            .any(|ch| matches!(ch, ' ' | '\t' | '\n' | '\r' | '"'))
    {
        return arg.to_owned();
    }
    let mut quoted = String::from("\"");
    let mut backslashes = 0usize;
    for ch in arg.chars() {
        match ch {
            '\\' => backslashes += 1,
            '"' => {
                quoted.extend(std::iter::repeat_n('\\', backslashes * 2 + 1));
                quoted.push('"');
                backslashes = 0;
            }
            _ => {
                quoted.extend(std::iter::repeat_n('\\', backslashes));
                backslashes = 0;
                quoted.push(ch);
            }
        }
    }
    quoted.extend(std::iter::repeat_n('\\', backslashes * 2));
    quoted.push('"');
    quoted
}

#[cfg(windows)]
fn windows_environment_block(env_overrides: &[(&str, &Path)]) -> Vec<u16> {
    let mut env = BTreeMap::<String, OsString>::new();
    for (key, value) in std::env::vars_os() {
        let key_string = key.to_string_lossy().to_string();
        env.insert(
            key_string.to_ascii_uppercase(),
            OsString::from(format!("{}={}", key_string, value.to_string_lossy())),
        );
    }
    for (key, value) in env_overrides {
        env.insert(
            key.to_ascii_uppercase(),
            OsString::from(format!("{}={}", key, value.display())),
        );
    }
    let mut block = Vec::new();
    for entry in env.values() {
        block.extend(entry.encode_wide());
        block.push(0);
    }
    block.push(0);
    block
}

#[derive(Debug, Clone, Serialize)]
pub struct AppPaths {
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
    pub cache_dir: PathBuf,
}

impl AppPaths {
    pub fn discover() -> Result<Self> {
        let data_dir_override = env_path_override("ROCM_CLI_DATA_DIR");
        let cache_dir_override = env_path_override("ROCM_CLI_CACHE_DIR");
        let mut paths = Self {
            config_dir: env_path_override("ROCM_CLI_CONFIG_DIR")
                .or_else(default_config_dir)
                .context("unable to determine config directory for rocm-cli")?,
            data_dir: data_dir_override
                .clone()
                .or_else(default_data_dir)
                .context("unable to determine data directory for rocm-cli")?,
            cache_dir: cache_dir_override
                .clone()
                .or_else(default_cache_dir)
                .context("unable to determine cache directory for rocm-cli")?,
        }
        .normalize_for_host();
        if data_dir_override.is_none()
            && let Some(managed_root) = configured_managed_root_from_config(&paths)
        {
            paths = paths.with_managed_root(managed_root, cache_dir_override.is_some());
        }
        Ok(paths.normalize_for_host())
    }

    fn normalize_for_host(mut self) -> Self {
        self.config_dir = normalize_runtime_path_for_host(&self.config_dir);
        self.data_dir = normalize_runtime_path_for_host(&self.data_dir);
        self.cache_dir = normalize_runtime_path_for_host(&self.cache_dir);
        self
    }

    #[must_use]
    pub fn with_managed_root(mut self, root: impl Into<PathBuf>, keep_cache_dir: bool) -> Self {
        self.data_dir = normalize_runtime_path_for_host(&root.into());
        if !keep_cache_dir {
            self.cache_dir = managed_runtime_cache_dir(&self.data_dir);
        }
        self.normalize_for_host()
    }

    pub fn ensure(&self) -> Result<()> {
        for dir in [
            &self.config_dir,
            &self.data_dir,
            &self.cache_dir,
            &self.audit_dir(),
            &self.automations_dir(),
            &self.data_dir.join("engines"),
            &self.data_dir.join("envs"),
            &self.data_dir.join("logs"),
            &self.data_dir.join("services"),
            &self.data_dir.join("models"),
            &self.data_dir.join("runtimes"),
            &self.telemetry_state_dir(),
        ] {
            fs::create_dir_all(dir)
                .with_context(|| format!("failed to create {}", dir.display()))?;
        }
        Ok(())
    }

    pub fn engine_dir(&self, engine: &str) -> PathBuf {
        self.data_dir.join("engines").join(engine)
    }

    pub fn primary_engine_plugin_dir(&self) -> PathBuf {
        self.data_dir.join("engines").join("plugins")
    }

    pub fn engine_logs_dir(&self, engine: &str) -> PathBuf {
        self.engine_dir(engine).join("logs")
    }

    pub fn engine_envs_root(&self) -> PathBuf {
        env_path_override("ROCM_CLI_ENGINE_ENVS_ROOT").map_or_else(
            || self.data_dir.join("engines"),
            |root| normalize_runtime_path_for_host(&root),
        )
    }

    pub fn engine_envs_dir(&self, engine: &str) -> PathBuf {
        self.engine_envs_root().join(engine).join("envs")
    }

    pub fn engine_locks_dir(&self, engine: &str) -> PathBuf {
        self.engine_dir(engine).join("locks")
    }

    pub fn engine_manifests_dir(&self, engine: &str) -> PathBuf {
        self.engine_dir(engine).join("manifests")
    }

    pub fn engine_state_dir(&self, engine: &str) -> PathBuf {
        self.engine_dir(engine).join("state")
    }

    pub fn config_path(&self) -> PathBuf {
        self.config_dir.join("config.json")
    }

    pub fn services_dir(&self) -> PathBuf {
        self.data_dir.join("services")
    }

    pub fn audit_dir(&self) -> PathBuf {
        self.data_dir.join("audit")
    }

    pub fn audit_events_path(&self) -> PathBuf {
        self.audit_dir().join("events.jsonl")
    }

    pub fn automations_dir(&self) -> PathBuf {
        self.data_dir.join("automations")
    }

    pub fn automation_state_path(&self) -> PathBuf {
        self.automations_dir().join("runtime-state.json")
    }

    pub fn automation_events_path(&self) -> PathBuf {
        self.automations_dir().join("events.jsonl")
    }

    pub fn automation_proposals_path(&self) -> PathBuf {
        self.automations_dir().join("proposals.jsonl")
    }

    pub fn service_manifest_path(&self, service_id: &str) -> PathBuf {
        self.services_dir().join(format!("{service_id}.json"))
    }

    pub fn service_log_path(&self, service_id: &str) -> PathBuf {
        self.services_dir().join(format!("{service_id}.log"))
    }

    pub fn service_engine_state_path(&self, engine: &str, service_id: &str) -> PathBuf {
        self.engine_state_dir(engine)
            .join(format!("{service_id}.json"))
    }

    /// Directory holding rocm-dash telemetry daemon state.
    /// (G3 rocm-cli maintainer sign-off pending — engineering implementation only.)
    pub fn telemetry_state_dir(&self) -> PathBuf {
        self.data_dir.join("telemetry")
    }

    /// Log file for the rocm-dash telemetry daemon, under the shared logs dir.
    ///
    /// Deliberately under the canonical `AppPaths` data root
    /// (`~/.rocm/logs/rocmdashd.log`), NOT the legacy standalone rocm-dash XDG
    /// state path (`~/.local/state/rocm-dash/`). D6 unifies the dual-dir split
    /// onto `~/.rocm`; do not "restore" the old XDG location.
    pub fn daemon_log_path(&self) -> PathBuf {
        self.data_dir.join("logs").join("rocmdashd.log")
    }

    /// Directory for client-side (CLI/TUI process) `tracing` log files.
    ///
    /// Siblings the daemon's `rocmdashd.log` under the same canonical
    /// `~/.rocm/logs` root; the client writer rotates files inside this
    /// directory itself (see `apps/rocm/src/logging.rs`), so only the
    /// directory — not a single fixed file name — is exposed here.
    pub fn client_log_dir(&self) -> PathBuf {
        self.data_dir.join("logs")
    }
}

fn configured_managed_root_from_config(paths: &AppPaths) -> Option<PathBuf> {
    let bytes = fs::read(paths.config_path()).ok()?;
    let value = serde_json::from_slice::<serde_json::Value>(&bytes).ok()?;
    value
        .get("setup")?
        .get("therock_venv")?
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

pub fn engine_plugin_dirs(paths: &AppPaths) -> Vec<PathBuf> {
    vec![
        paths.primary_engine_plugin_dir(),
        paths.data_dir.join("engines"),
    ]
}

fn env_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExamineSummary {
    pub os: String,
    pub arch: String,
    pub kernel: Option<String>,
    pub distro: Option<String>,
    pub cpu: Option<String>,
    pub system_ram_gib: Option<f64>,
    pub interactive_terminal: bool,
    pub default_engine: String,
    pub detected_gfx_target: Option<String>,
    #[serde(default)]
    pub compatible_therock_family: Option<String>,
    #[serde(default)]
    pub detected_therock_family: Option<String>,
    pub driver: DriverSummary,
    pub legacy_rocm: LegacyRocmSummary,
    #[serde(default)]
    pub wsl: Option<WslSummary>,
    pub managed_runtime_count: usize,
    pub managed_service_count: usize,
    pub model_cache_entries: usize,
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
    pub cache_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriverSummary {
    pub policy: String,
    pub status: String,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyRocmSummary {
    pub status: String,
    pub paths: Vec<PathBuf>,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WslSummary {
    pub is_wsl: bool,
    pub dxg_device: bool,
    pub dxcore: bool,
    pub librocdxg: bool,
    pub rocdxg_dids: bool,
    pub ldconfig_librocdxg: bool,
    pub rocminfo: bool,
    pub cargo: bool,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct HostGpuSummary {
    pub name: Option<String>,
    pub gfx_target: Option<String>,
    pub therock_family: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct WindowsExamineInventory {
    cpu_model: Option<String>,
    system_ram_gib: Option<f64>,
    displays: Vec<WindowsDisplayAdapter>,
}

#[derive(Debug, Clone)]
struct WindowsDisplayAdapter {
    name: String,
    driver_version: Option<String>,
    pnp_device_id: Option<String>,
}

impl WindowsExamineInventory {
    #[cfg(windows)]
    fn is_empty(&self) -> bool {
        self.cpu_model.is_none() && self.system_ram_gib.is_none() && self.displays.is_empty()
    }

    #[cfg(windows)]
    fn merge_missing_from(&mut self, mut other: WindowsExamineInventory) {
        if self.cpu_model.is_none() {
            self.cpu_model = other.cpu_model.take();
        }
        if self.system_ram_gib.is_none() {
            self.system_ram_gib = other.system_ram_gib.take();
        }
        for display in other.displays {
            let duplicate = self.displays.iter_mut().find(|existing| {
                match (
                    existing.pnp_device_id.as_deref(),
                    display.pnp_device_id.as_deref(),
                ) {
                    (Some(left), Some(right)) => left.eq_ignore_ascii_case(right),
                    _ => {
                        !existing.name.trim().is_empty()
                            && !display.name.trim().is_empty()
                            && existing.name.eq_ignore_ascii_case(&display.name)
                    }
                }
            });
            if let Some(existing) = duplicate {
                if existing.name.trim().is_empty() && !display.name.trim().is_empty() {
                    existing.name = display.name;
                }
                if existing.driver_version.is_none() {
                    existing.driver_version = display.driver_version;
                }
                if existing.pnp_device_id.is_none() {
                    existing.pnp_device_id = display.pnp_device_id;
                }
            } else {
                self.displays.push(display);
            }
        }
    }

    fn amd_display_driver_detail(&self) -> Option<String> {
        let display = self.preferred_amd_display()?;
        let name = display.name.trim();
        if name.is_empty() {
            return None;
        }
        let detail = format!(
            "{name} driver {}",
            display.driver_version.as_deref().unwrap_or("")
        );
        Some(detail.trim().to_owned())
    }

    fn amd_display_name(&self) -> Option<String> {
        self.preferred_amd_display()
            .map(|display| display.name.trim())
            .filter(|name| !name.is_empty())
            .map(str::to_owned)
    }

    fn preferred_amd_display(&self) -> Option<&WindowsDisplayAdapter> {
        self.displays
            .iter()
            .find(|display| {
                display
                    .pnp_device_id
                    .as_deref()
                    .and_then(amd_pci_device_id_from_pnp_id)
                    .and_then(|device_id| gfx_target_from_amd_pci_device_id(&device_id))
                    .is_some()
            })
            .or_else(|| {
                self.displays
                    .iter()
                    .find(|display| gfx_target_from_amd_marketing_name(&display.name).is_some())
            })
            .or_else(|| {
                self.displays
                    .iter()
                    .find(|display| !display.name.trim().is_empty())
            })
    }

    fn display_gfx_target(&self) -> Option<String> {
        parse_windows_display_gfx_target(&self.display_gfx_probe_text())
    }

    fn display_gfx_probe_text(&self) -> String {
        self.displays
            .iter()
            .map(|display| {
                format!(
                    "{}\t{}",
                    display.name,
                    display.pnp_device_id.as_deref().unwrap_or("")
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl ExamineSummary {
    pub fn gather() -> Result<Self> {
        let paths = AppPaths::discover()?;
        let windows_inventory = detect_windows_examine_inventory();
        let wsl = detect_wsl_summary();
        let detected_gfx_target = detect_examine_gfx_target_fast(windows_inventory.as_ref());
        let compatible_therock_family = detected_gfx_target
            .as_deref()
            .and_then(normalize_therock_family);
        let detected_therock_family = detect_managed_therock_family(&paths);
        Ok(Self {
            os: runtime_os_name().to_owned(),
            arch: std::env::consts::ARCH.to_owned(),
            kernel: detect_kernel_version(),
            distro: detect_distro_name(),
            cpu: detect_cpu_model_with_windows_inventory(windows_inventory.as_ref()),
            system_ram_gib: detect_system_ram_gib_with_windows_inventory(
                windows_inventory.as_ref(),
            ),
            interactive_terminal: interactive_terminal(),
            default_engine: default_engine_for_platform().to_owned(),
            detected_gfx_target,
            compatible_therock_family,
            detected_therock_family,
            driver: detect_driver_summary_with_windows_inventory(
                windows_inventory.as_ref(),
                wsl.as_ref(),
            ),
            legacy_rocm: detect_legacy_rocm_summary(),
            wsl,
            managed_runtime_count: count_json_files(
                &paths.data_dir.join("runtimes").join("registry"),
            ),
            managed_service_count: count_json_files(&paths.services_dir()),
            model_cache_entries: count_dir_entries(&paths.data_dir.join("models")),
            config_dir: paths.config_dir,
            data_dir: paths.data_dir,
            cache_dir: paths.cache_dir,
        })
    }

    pub fn render_text(&self) -> String {
        let legacy_paths = if self.legacy_rocm.paths.is_empty() {
            "<none>".to_owned()
        } else {
            self.legacy_rocm
                .paths
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let wsl = self.wsl.as_ref();
        format!(
            "rocm examine\n  os: {}\n  arch: {}\n  kernel: {}\n  distro: {}\n  cpu: {}\n  system_ram: {}\n  interactive_terminal: {}\n  default_engine: {}\n  detected_gfx_target: {}\n  compatible_therock_family: {}\n  detected_therock_family: {}\n  driver_policy: {}\n  driver_status: {}\n  driver_detail: {}\n  legacy_rocm_status: {}\n  legacy_rocm_paths: {}\n  legacy_rocm_detail: {}\n  legacy_rocm_guidance: {}\n  wsl: {}\n  wsl_dxg_device: {}\n  wsl_dxcore: {}\n  wsl_librocdxg: {}\n  wsl_rocdxg_dids: {}\n  wsl_ldconfig_librocdxg: {}\n  wsl_global_rocminfo: {}\n  wsl_cargo: {}\n  wsl_detail: {}\n  managed_runtimes: {}\n  managed_services: {}\n  model_cache_entries: {}\n  config_dir: {}\n  data_dir: {}\n  cache_dir: {}\n",
            self.os,
            self.arch,
            self.kernel.as_deref().unwrap_or("<unknown>"),
            self.distro.as_deref().unwrap_or("<unknown>"),
            self.cpu.as_deref().unwrap_or("<unknown>"),
            self.system_ram_gib
                .map_or_else(|| "<unknown>".to_owned(), format_gib_value),
            self.interactive_terminal,
            self.default_engine,
            self.detected_gfx_target.as_deref().unwrap_or("<unknown>"),
            self.compatible_therock_family
                .as_deref()
                .unwrap_or("<unknown>"),
            self.detected_therock_family
                .as_deref()
                .unwrap_or("<not detected>"),
            self.driver.policy,
            self.driver.status,
            self.driver.detail.as_deref().unwrap_or("<unknown>"),
            self.legacy_rocm.status,
            legacy_paths,
            self.legacy_rocm.detail.as_deref().unwrap_or("<unknown>"),
            self.legacy_rocm_guidance(),
            wsl.is_some_and(|summary| summary.is_wsl),
            wsl.is_some_and(|summary| summary.dxg_device),
            wsl.is_some_and(|summary| summary.dxcore),
            wsl.is_some_and(|summary| summary.librocdxg),
            wsl.is_some_and(|summary| summary.rocdxg_dids),
            wsl.is_some_and(|summary| summary.ldconfig_librocdxg),
            wsl.is_some_and(|summary| summary.rocminfo),
            wsl.is_some_and(|summary| summary.cargo),
            wsl.and_then(|summary| summary.detail.as_deref())
                .unwrap_or("<not WSL>"),
            self.managed_runtime_count,
            self.managed_service_count,
            self.model_cache_entries,
            self.config_dir.display(),
            self.data_dir.display(),
            self.cache_dir.display(),
        )
    }

    const fn legacy_rocm_guidance(&self) -> &'static str {
        if self.legacy_rocm.paths.is_empty() {
            return "none";
        }
        if self.managed_runtime_count == 0 {
            return "legacy ROCm detected; install a managed TheRock runtime with `rocm install sdk --channel release --format wheel` and keep legacy ROCm unmanaged";
        }
        "legacy ROCm detected; keep it side-by-side and use rocm-cli managed TheRock runtimes for local engines"
    }
}

pub fn interactive_terminal() -> bool {
    stdin().is_terminal() && stdout().is_terminal()
}

pub const fn default_engine_for_platform() -> &'static str {
    "lemonade"
}

const VLLM_PREFERRED_THEROCK_FAMILIES: &[&str] = &["gfx906", "gfx908", "gfx90a"];

pub fn preferred_serve_engine_for_host_gpu_summary(
    summary: &HostGpuSummary,
) -> Option<&'static str> {
    // The vLLM engine adapter bails out on native Windows builds, so never prefer it
    // there. WSL builds as a Linux target and therefore remains eligible.
    if cfg!(windows) {
        return None;
    }
    preferred_serve_engine_for_therock_family(
        summary
            .therock_family
            .as_deref()
            .or(summary.gfx_target.as_deref()),
    )
}

fn preferred_serve_engine_for_therock_family(family: Option<&str>) -> Option<&'static str> {
    let family = family?.trim();
    if family.is_empty() {
        return None;
    }

    let family = normalize_therock_family(family)
        .as_deref()
        .unwrap_or(family)
        .to_ascii_lowercase();
    if family.ends_with("-dcgpu")
        || VLLM_PREFERRED_THEROCK_FAMILIES
            .iter()
            .any(|candidate| *candidate == family)
    {
        Some("vllm")
    } else {
        None
    }
}

fn detect_kernel_version() -> Option<String> {
    if runtime_is_windows() {
        capture_optional_command("cmd", &["/C", "ver"])
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    } else {
        capture_optional_command("uname", &["-r"])
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    }
}

fn detect_distro_name() -> Option<String> {
    if runtime_is_windows() {
        return Some("Windows".to_owned());
    }

    if runtime_is_linux() {
        return parse_os_release_pretty_name(&fs::read_to_string("/etc/os-release").ok()?)
            .or_else(|| Some("Linux".to_owned()));
    }

    None
}

fn parse_os_release_pretty_name(text: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let value = line.strip_prefix("PRETTY_NAME=")?.trim();
        let value = value.trim_matches('"').trim_matches('\'').trim();
        (!value.is_empty()).then(|| value.to_owned())
    })
}

fn detect_cpu_model_with_windows_inventory(
    windows_inventory: Option<&WindowsExamineInventory>,
) -> Option<String> {
    if runtime_is_windows()
        && let Some(inventory) = windows_inventory
    {
        return inventory.cpu_model.clone();
    }

    detect_cpu_model()
}

fn detect_cpu_model() -> Option<String> {
    if runtime_is_windows() {
        let script =
            "Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name";
        return capture_optional_command_with_timeout(
            "powershell",
            &["-NoProfile", "-Command", script],
            OPTIONAL_COMMAND_TIMEOUT,
        )
        .map(|value| normalize_cpu_model(&value))
        .filter(|value| !value.is_empty());
    }

    if runtime_is_linux()
        && let Some(model) = fs::read_to_string("/proc/cpuinfo").ok().and_then(|text| {
            text.lines().find_map(|line| {
                let value = line
                    .strip_prefix("model name")
                    .and_then(|rest| rest.split_once(':').map(|(_, value)| value))
                    .or_else(|| {
                        line.strip_prefix("Hardware")
                            .and_then(|rest| rest.split_once(':').map(|(_, value)| value))
                    })?;
                let value = normalize_cpu_model(value);
                (!value.is_empty()).then_some(value)
            })
        })
    {
        return Some(model);
    }

    None
}

fn detect_system_ram_gib_with_windows_inventory(
    windows_inventory: Option<&WindowsExamineInventory>,
) -> Option<f64> {
    if runtime_is_windows()
        && let Some(inventory) = windows_inventory
    {
        return inventory.system_ram_gib;
    }

    detect_system_ram_gib()
}

pub fn detect_system_ram_gib() -> Option<f64> {
    if runtime_is_windows() {
        let script = "(Get-CimInstance -ClassName Win32_ComputerSystem -Property TotalPhysicalMemory).TotalPhysicalMemory";
        return capture_optional_command_with_timeout(
            "powershell",
            &["-NoProfile", "-Command", script],
            OPTIONAL_COMMAND_TIMEOUT,
        )
        .and_then(|value| bytes_text_to_gib(&value));
    }

    if runtime_is_linux()
        && let Some(kib) = fs::read_to_string("/proc/meminfo").ok().and_then(|text| {
            text.lines().find_map(|line| {
                let value = line.strip_prefix("MemTotal:")?.trim();
                let number = value.split_whitespace().next()?.parse::<f64>().ok()?;
                Some(number)
            })
        })
    {
        return Some(kib / 1024.0 / 1024.0);
    }

    if cfg!(target_os = "macos") {
        return capture_optional_command("sysctl", &["-n", "hw.memsize"])
            .and_then(|value| bytes_text_to_gib(&value));
    }

    None
}

fn bytes_text_to_gib(value: &str) -> Option<f64> {
    let bytes = value.trim().parse::<f64>().ok()?;
    (bytes > 0.0).then_some(bytes / 1024.0 / 1024.0 / 1024.0)
}

fn format_gib_value(value: f64) -> String {
    if value >= 10.0 {
        format!("{value:.0} GiB")
    } else {
        format!("{value:.1} GiB")
    }
}

fn normalize_cpu_model(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn detect_wsl_summary() -> Option<WslSummary> {
    if !runtime_is_linux() {
        return None;
    }

    let proc_version = fs::read_to_string("/proc/version").unwrap_or_default();
    let dxg_device = Path::new("/dev/dxg").exists();
    let is_wsl = dxg_device || proc_version.to_ascii_lowercase().contains("microsoft");
    if !is_wsl {
        return None;
    }

    let dxcore = Path::new("/usr/lib/wsl/lib/libdxcore.so").exists();
    let librocdxg = Path::new("/opt/rocm/lib/librocdxg.so").exists();
    let rocdxg_dids = Path::new("/opt/rocm/share/rocdxg/dids.conf").exists();
    let ldconfig_text = capture_optional_command("ldconfig", &["-p"]).unwrap_or_default();
    let ldconfig_librocdxg = ldconfig_text.contains("librocdxg.so");
    let rocminfo = tool_on_path("rocminfo");
    let cargo = tool_on_path("cargo");
    let mut missing = Vec::new();
    if !dxg_device {
        missing.push("/dev/dxg");
    }
    if !dxcore {
        missing.push("/usr/lib/wsl/lib/libdxcore.so");
    }
    if !librocdxg {
        missing.push("/opt/rocm/lib/librocdxg.so");
    }
    if !ldconfig_librocdxg {
        missing.push("ldconfig:librocdxg.so");
    }
    let detail = if missing.is_empty() {
        Some("WSL DXCore and ROCDXG plumbing detected".to_owned())
    } else {
        Some(format!("missing {}", missing.join(", ")))
    };

    Some(WslSummary {
        is_wsl,
        dxg_device,
        dxcore,
        librocdxg,
        rocdxg_dids,
        ldconfig_librocdxg,
        rocminfo,
        cargo,
        detail,
    })
}

fn detect_driver_summary_with_windows_inventory(
    windows_inventory: Option<&WindowsExamineInventory>,
    wsl: Option<&WslSummary>,
) -> DriverSummary {
    if runtime_is_windows() {
        let detail = windows_inventory
            .and_then(WindowsExamineInventory::amd_display_driver_detail)
            .or_else(|| {
                if windows_inventory.is_none() {
                    detect_windows_amd_display_driver()
                } else {
                    None
                }
            });
        return windows_driver_summary(detail);
    }

    if let Some(wsl) = wsl {
        return wsl_driver_summary(wsl);
    }

    detect_driver_summary()
}

fn detect_driver_summary() -> DriverSummary {
    if runtime_is_windows() {
        let detail = detect_windows_amd_display_driver();
        return windows_driver_summary(detail);
    }

    if runtime_is_linux() {
        let module_detected = Path::new("/sys/module/amdgpu").exists();
        return DriverSummary {
            policy: "linux_official_amd_dkms_wrapper".to_owned(),
            status: if module_detected {
                "amdgpu_available".to_owned()
            } else {
                "not_detected".to_owned()
            },
            detail: if Path::new("/dev/kfd").exists() {
                Some("/dev/kfd is present".to_owned())
            } else if module_detected {
                Some("amdgpu module metadata is present".to_owned())
            } else {
                None
            },
        };
    }

    DriverSummary {
        policy: "inspection_only".to_owned(),
        status: "unsupported_platform".to_owned(),
        detail: None,
    }
}

fn wsl_driver_summary(wsl: &WslSummary) -> DriverSummary {
    let ready = wsl.dxg_device && wsl.dxcore && wsl.librocdxg && wsl.ldconfig_librocdxg;
    let status = if ready {
        "wsl_rocdxg_ready"
    } else if wsl.dxg_device && wsl.dxcore {
        "wsl_rocdxg_missing"
    } else {
        "wsl_gpu_plumbing_missing"
    };
    DriverSummary {
        policy: "wsl_rocdxg".to_owned(),
        status: status.to_owned(),
        detail: wsl.detail.clone(),
    }
}

fn windows_driver_summary(detail: Option<String>) -> DriverSummary {
    DriverSummary {
        policy: "windows_validate_only".to_owned(),
        status: if detail.is_some() {
            "amd_display_driver_detected".to_owned()
        } else {
            "not_detected".to_owned()
        },
        detail,
    }
}

fn detect_legacy_rocm_summary() -> LegacyRocmSummary {
    let mut paths = Vec::new();
    let mut candidates = Vec::new();

    if let Some(path) = std::env::var_os("ROCM_PATH") {
        candidates.push(PathBuf::from(path));
    }

    if runtime_is_windows() {
        candidates.push(PathBuf::from(r"C:\Program Files\AMD\ROCm"));
        candidates.push(PathBuf::from(r"C:\Program Files\ROCm"));
    } else {
        candidates.push(PathBuf::from("/opt/rocm"));
        candidates.push(PathBuf::from("/usr/local/rocm"));
    }

    for candidate in candidates {
        if legacy_rocm_candidate_exists(&candidate) && !paths.iter().any(|path| path == &candidate)
        {
            paths.push(candidate);
        }
    }

    let status = if paths.is_empty() {
        "not_detected"
    } else {
        "detected_unmanaged"
    };
    let detail = if paths.is_empty() {
        None
    } else {
        Some("legacy ROCm installs are reported for compatibility only; rocm-cli manages TheRock runtimes separately".to_owned())
    };

    LegacyRocmSummary {
        status: status.to_owned(),
        paths,
        detail,
    }
}

#[allow(clippy::case_sensitive_file_extension_comparisons)] // ROCm installs the runtime DLL as lowercase `amdhip64.dll`
fn legacy_rocm_candidate_exists(candidate: &Path) -> bool {
    if !candidate.exists() {
        return false;
    }
    if [
        candidate.join("bin").join("rocminfo"),
        candidate.join("bin").join("rocminfo.exe"),
        candidate.join("bin").join("hipcc"),
        candidate.join("bin").join("hipcc.bat"),
        candidate.join("lib").join("libamdhip64.so"),
        candidate.join("lib").join("libhsa-runtime64.so"),
        candidate.join(".info").join("version"),
    ]
    .iter()
    .any(|marker| marker.exists())
    {
        return true;
    }

    fs::read_dir(candidate.join("bin"))
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .any(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with("amdhip64") && name.ends_with(".dll"))
        })
}

#[cfg(windows)]
fn detect_windows_amd_display_driver() -> Option<String> {
    if !runtime_is_windows() {
        return None;
    }
    let script = "$gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.AdapterCompatibility -match 'AMD|Advanced Micro Devices' -or $_.Name -match 'AMD|Radeon|Instinct' } | Select-Object -First 1 -Property Name,DriverVersion; if ($gpu) { \"$($gpu.Name) driver $($gpu.DriverVersion)\" }";
    capture_optional_command("powershell", &["-NoProfile", "-Command", script])
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

#[cfg(not(windows))]
const fn detect_windows_amd_display_driver() -> Option<String> {
    None
}

#[cfg(windows)]
fn detect_windows_examine_inventory() -> Option<WindowsExamineInventory> {
    if !runtime_is_windows() {
        return None;
    }
    let mut inventory = WindowsExamineInventory::default();
    if let Some(pnp_util) = detect_windows_examine_inventory_from_pnputil() {
        inventory.merge_missing_from(pnp_util);
    }
    if inventory.displays.is_empty()
        && let Some(video) = detect_windows_examine_inventory_from_video_controller()
    {
        inventory.merge_missing_from(video);
    }
    if inventory.displays.is_empty()
        && let Some(pnp) = detect_windows_examine_inventory_from_pnp_entity()
    {
        inventory.merge_missing_from(pnp);
    }
    if (inventory.cpu_model.is_none() || inventory.system_ram_gib.is_none())
        && let Some(system) = detect_windows_system_inventory_from_cim()
    {
        inventory.merge_missing_from(system);
    }

    (!inventory.is_empty()).then_some(inventory)
}

#[cfg(windows)]
fn detect_windows_examine_inventory_from_pnputil() -> Option<WindowsExamineInventory> {
    if !runtime_is_windows() {
        return None;
    }
    capture_optional_command_with_timeout(
        "pnputil",
        &["/enum-devices", "/class", "Display"],
        WINDOWS_INVENTORY_QUERY_TIMEOUT,
    )
    .map(|output| parse_windows_pnputil_display_inventory(&output))
}

#[cfg(windows)]
fn detect_windows_examine_inventory_from_video_controller() -> Option<WindowsExamineInventory> {
    if !runtime_is_windows() {
        return None;
    }
    capture_optional_command_with_timeout(
        "powershell",
        &[
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            WINDOWS_VIDEO_CONTROLLER_INVENTORY_SCRIPT,
        ],
        WINDOWS_INVENTORY_QUERY_TIMEOUT,
    )
    .map(|output| parse_windows_examine_inventory(&output))
}

#[cfg(windows)]
fn detect_windows_system_inventory_from_cim() -> Option<WindowsExamineInventory> {
    if !runtime_is_windows() {
        return None;
    }
    capture_optional_command_with_timeout(
        "powershell",
        &[
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            WINDOWS_SYSTEM_INVENTORY_SCRIPT,
        ],
        OPTIONAL_COMMAND_TIMEOUT,
    )
    .map(|output| parse_windows_examine_inventory(&output))
}

#[cfg(windows)]
fn detect_windows_examine_inventory_from_pnp_entity() -> Option<WindowsExamineInventory> {
    if !runtime_is_windows() {
        return None;
    }
    capture_optional_command_with_timeout(
        "powershell",
        &[
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            WINDOWS_PNP_ENTITY_INVENTORY_SCRIPT,
        ],
        WINDOWS_INVENTORY_QUERY_TIMEOUT,
    )
    .map(|output| parse_windows_examine_inventory(&output))
}

#[cfg(not(windows))]
const fn detect_windows_examine_inventory() -> Option<WindowsExamineInventory> {
    None
}

#[cfg(any(windows, test))]
fn clean_windows_display_name(value: &str) -> String {
    let value = value.trim();
    let value = value.rsplit_once(';').map_or(value, |(_, name)| name);
    value.trim().to_owned()
}

#[cfg_attr(not(windows), allow(dead_code))]
fn parse_windows_examine_inventory(text: &str) -> WindowsExamineInventory {
    let mut inventory = WindowsExamineInventory::default();

    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let mut fields = line.split('\t');
        match fields.next() {
            Some("CPU") => {
                let cpu_model = fields.collect::<Vec<_>>().join("\t");
                let cpu_model = normalize_cpu_model(&cpu_model);
                if !cpu_model.is_empty() {
                    inventory.cpu_model = Some(cpu_model);
                }
            }
            Some("RAM") => {
                let bytes = fields.next().unwrap_or("").trim();
                inventory.system_ram_gib = bytes_text_to_gib(bytes);
            }
            Some("GPU") => {
                let name = fields.next().unwrap_or("").trim().to_owned();
                let driver_version = fields
                    .next()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_owned);
                let pnp_device_id = fields
                    .next()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_owned);
                if !name.is_empty() || driver_version.is_some() || pnp_device_id.is_some() {
                    inventory.displays.push(WindowsDisplayAdapter {
                        name,
                        driver_version,
                        pnp_device_id,
                    });
                }
            }
            _ => {}
        }
    }

    inventory
}

#[cfg(any(windows, test))]
fn parse_windows_pnputil_display_inventory(text: &str) -> WindowsExamineInventory {
    let mut inventory = WindowsExamineInventory::default();
    let mut name: Option<String> = None;
    let mut instance_id: Option<String> = None;
    let mut driver_version: Option<String> = None;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            push_windows_pnputil_display(
                &mut inventory,
                &mut name,
                &mut instance_id,
                &mut driver_version,
            );
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim().to_ascii_lowercase();
        let value = value.trim();
        if value.is_empty() {
            continue;
        }
        match key.as_str() {
            "instance id" | "device instance id" => {
                instance_id = Some(value.to_owned());
            }
            "device description" | "friendly name" | "name" => {
                name = Some(clean_windows_display_name(value));
            }
            "driver version" => {
                driver_version = Some(value.to_owned());
            }
            _ => {}
        }
    }
    push_windows_pnputil_display(
        &mut inventory,
        &mut name,
        &mut instance_id,
        &mut driver_version,
    );

    inventory
}

#[cfg(any(windows, test))]
fn push_windows_pnputil_display(
    inventory: &mut WindowsExamineInventory,
    name: &mut Option<String>,
    instance_id: &mut Option<String>,
    driver_version: &mut Option<String>,
) {
    let pnp = instance_id.take();
    let display_name = name.take().unwrap_or_default();
    let driver = driver_version.take();
    let has_amd_id = pnp
        .as_deref()
        .is_some_and(|value| value.to_ascii_uppercase().contains("VEN_1002"));
    let has_amd_name = display_name
        .to_ascii_lowercase()
        .split_whitespace()
        .any(|token| matches!(token, "amd" | "radeon" | "instinct"));
    if !has_amd_id && !has_amd_name {
        return;
    }
    inventory.displays.push(WindowsDisplayAdapter {
        name: display_name,
        driver_version: driver,
        pnp_device_id: pnp,
    });
}

pub fn detect_host_gpu_diagnostics() -> String {
    use std::fmt::Write as _;
    let mut output = String::new();
    let _ = writeln!(output, "GPU detection diagnostics");
    let _ = writeln!(output, "  runtime_os: {}", runtime_os_name());
    let summary = detect_host_gpu_summary(None);
    let _ = writeln!(
        output,
        "  detected_name: {}",
        summary.name.as_deref().unwrap_or("<unknown>")
    );
    let _ = writeln!(
        output,
        "  detected_gfx_target: {}",
        summary.gfx_target.as_deref().unwrap_or("<unknown>")
    );
    let _ = writeln!(
        output,
        "  detected_therock_family: {}",
        summary.therock_family.as_deref().unwrap_or("<unknown>")
    );

    if runtime_is_windows() {
        append_windows_gpu_probe_diagnostics(&mut output);
    } else if runtime_is_linux() {
        let _ = writeln!(
            output,
            "  linux_sysfs_gfx_target: {}",
            detect_linux_sysfs_gfx_target()
                .as_deref()
                .unwrap_or("<not found>")
        );
        let _ = writeln!(
            output,
            "  linux_primary_gpu_name: {}",
            detect_linux_primary_gpu_name()
                .as_deref()
                .unwrap_or("<not found>")
        );
        if is_wsl_environment_fast() {
            let wsl_probe = detect_wsl_windows_display_probe_text().unwrap_or_default();
            let _ = writeln!(
                output,
                "  wsl_windows_display_probe_lines: {}",
                wsl_probe.lines().count()
            );
            for line in wsl_probe.lines().take(8) {
                let _ = writeln!(output, "    {line}");
            }
        }
    }

    output
}

#[cfg(windows)]
fn append_windows_gpu_probe_diagnostics(output: &mut String) {
    append_windows_probe_diagnostics(
        output,
        "pnputil display devices",
        "pnputil",
        &["/enum-devices", "/class", "Display"],
        WINDOWS_INVENTORY_QUERY_TIMEOUT,
        parse_windows_pnputil_display_inventory,
    );
    append_windows_probe_diagnostics(
        output,
        "Win32_VideoController",
        "powershell",
        &[
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            WINDOWS_VIDEO_CONTROLLER_INVENTORY_SCRIPT,
        ],
        WINDOWS_INVENTORY_QUERY_TIMEOUT,
        parse_windows_examine_inventory,
    );
    append_windows_probe_diagnostics(
        output,
        "Win32_PnPEntity",
        "powershell",
        &[
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            WINDOWS_PNP_ENTITY_INVENTORY_SCRIPT,
        ],
        WINDOWS_INVENTORY_QUERY_TIMEOUT,
        parse_windows_examine_inventory,
    );
}

#[cfg(not(windows))]
const fn append_windows_gpu_probe_diagnostics(_output: &mut String) {}

#[cfg(windows)]
fn append_windows_probe_diagnostics(
    output: &mut String,
    label: &str,
    program: &str,
    args: &[&str],
    timeout: Duration,
    parse: fn(&str) -> WindowsExamineInventory,
) {
    use std::fmt::Write as _;
    let result = capture_diagnostic_command(program, args, timeout);
    let _ = writeln!(output, "  probe: {label}");
    let _ = writeln!(
        output,
        "    command: {} {}",
        result
            .program
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| program.to_owned()),
        args.join(" ")
    );
    if let Some(error) = result.error.as_deref() {
        let _ = writeln!(output, "    error: {error}");
    }
    if result.timed_out {
        let _ = writeln!(output, "    error: timed out");
    }
    if let Some(status) = result.status.as_deref() {
        let _ = writeln!(output, "    status: {status}");
    }

    let inventory = parse(&result.stdout);
    let _ = writeln!(output, "    display_count: {}", inventory.displays.len());
    for display in inventory.displays.iter().take(8) {
        let gfx = display
            .pnp_device_id
            .as_deref()
            .and_then(amd_pci_device_id_from_pnp_id)
            .and_then(|device_id| gfx_target_from_amd_pci_device_id(&device_id).map(str::to_owned))
            .or_else(|| gfx_target_from_amd_marketing_name(&display.name).map(str::to_owned))
            .unwrap_or_else(|| "<unknown>".to_owned());
        let _ = writeln!(
            output,
            "      gpu: name={} pnp={} driver={} gfx={}",
            empty_as_unknown(&display.name),
            display.pnp_device_id.as_deref().unwrap_or("<unknown>"),
            display.driver_version.as_deref().unwrap_or("<unknown>"),
            gfx
        );
    }
    append_diagnostic_stream(output, "stdout", &result.stdout);
    append_diagnostic_stream(output, "stderr", &result.stderr);
}

#[cfg(windows)]
fn empty_as_unknown(value: &str) -> &str {
    let value = value.trim();
    if value.is_empty() { "<unknown>" } else { value }
}

#[derive(Debug)]
#[cfg(windows)]
struct DiagnosticCommandResult {
    program: Option<PathBuf>,
    status: Option<String>,
    stdout: String,
    stderr: String,
    error: Option<String>,
    timed_out: bool,
}

#[cfg(windows)]
fn capture_diagnostic_command(
    program: &str,
    args: &[&str],
    timeout: Duration,
) -> DiagnosticCommandResult {
    let candidates = tool_path_candidates(program);
    let mut last_error = None;
    for candidate in candidates {
        let path = PathBuf::from(&candidate);
        let mut child = match Command::new(&path)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(error) => {
                last_error = Some(format!("failed to launch {}: {error}", path.display()));
                continue;
            }
        };
        let stdout_reader = child.stdout.take().map(|mut stdout| {
            thread::spawn(move || {
                let mut bytes = Vec::new();
                let _ = stdout.read_to_end(&mut bytes);
                bytes
            })
        });
        let stderr_reader = child.stderr.take().map(|mut stderr| {
            thread::spawn(move || {
                let mut bytes = Vec::new();
                let _ = stderr.read_to_end(&mut bytes);
                bytes
            })
        });

        let start = Instant::now();
        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    let stdout = stdout_reader
                        .map(|reader| reader.join().unwrap_or_default())
                        .unwrap_or_default();
                    let stderr = stderr_reader
                        .map(|reader| reader.join().unwrap_or_default())
                        .unwrap_or_default();
                    return DiagnosticCommandResult {
                        program: Some(path),
                        status: Some(status.to_string()),
                        stdout: String::from_utf8_lossy(&stdout).into_owned(),
                        stderr: String::from_utf8_lossy(&stderr).into_owned(),
                        error: None,
                        timed_out: false,
                    };
                }
                Ok(None) if start.elapsed() < timeout => {
                    thread::sleep(Duration::from_millis(25));
                }
                Ok(None) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    let stdout = stdout_reader
                        .map(|reader| reader.join().unwrap_or_default())
                        .unwrap_or_default();
                    let stderr = stderr_reader
                        .map(|reader| reader.join().unwrap_or_default())
                        .unwrap_or_default();
                    return DiagnosticCommandResult {
                        program: Some(path),
                        status: None,
                        stdout: String::from_utf8_lossy(&stdout).into_owned(),
                        stderr: String::from_utf8_lossy(&stderr).into_owned(),
                        error: None,
                        timed_out: true,
                    };
                }
                Err(error) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return DiagnosticCommandResult {
                        program: Some(path),
                        status: None,
                        stdout: String::new(),
                        stderr: String::new(),
                        error: Some(format!("failed to wait: {error}")),
                        timed_out: false,
                    };
                }
            }
        }
    }

    DiagnosticCommandResult {
        program: None,
        status: None,
        stdout: String::new(),
        stderr: String::new(),
        error: last_error.or_else(|| Some(format!("{program} was not found"))),
        timed_out: false,
    }
}

#[cfg(windows)]
fn append_diagnostic_stream(output: &mut String, name: &str, text: &str) {
    use std::fmt::Write as _;
    let mut lines = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .peekable();
    if lines.peek().is_none() {
        return;
    }
    let _ = writeln!(output, "    {name}:");
    for line in lines.take(12) {
        let _ = writeln!(
            output,
            "      {}",
            truncate_diagnostic_line(line.trim(), 220)
        );
    }
}

#[cfg(windows)]
fn truncate_diagnostic_line(line: &str, max_chars: usize) -> String {
    let mut chars = line.chars();
    let truncated = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() {
        format!("{truncated}...")
    } else {
        truncated
    }
}

fn count_json_files(dir: &Path) -> usize {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .filter(|entry| entry.path().extension().and_then(|value| value.to_str()) == Some("json"))
        .count()
}

fn count_dir_entries(dir: &Path) -> usize {
    fs::read_dir(dir).map_or(0, |entries| entries.flatten().count())
}

pub fn require_nonempty(value: &str, field_name: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{field_name} must not be empty");
    }
    Ok(())
}

pub fn detect_host_therock_family() -> Option<String> {
    detect_host_gfx_target().and_then(|target| normalize_therock_family(&target))
}

pub fn detect_host_gpu_summary(paths: Option<&AppPaths>) -> HostGpuSummary {
    detect_host_gpu_summary_fast(paths)
}

#[cfg(windows)]
fn detect_host_gpu_summary_fast(_paths: Option<&AppPaths>) -> HostGpuSummary {
    let windows_inventory = detect_windows_examine_inventory();
    let gfx_target = detect_windows_display_gfx_target_with_inventory(windows_inventory.as_ref());
    let therock_family = gfx_target.as_deref().and_then(normalize_therock_family);
    let name = windows_inventory
        .as_ref()
        .and_then(WindowsExamineInventory::amd_display_name);
    HostGpuSummary {
        name,
        gfx_target,
        therock_family,
    }
}

#[cfg(target_os = "linux")]
fn detect_host_gpu_summary_fast(_paths: Option<&AppPaths>) -> HostGpuSummary {
    if runtime_is_windows() {
        let windows_inventory = detect_windows_examine_inventory();
        let gfx_target =
            detect_windows_display_gfx_target_with_inventory(windows_inventory.as_ref());
        let therock_family = gfx_target.as_deref().and_then(normalize_therock_family);
        let name = windows_inventory
            .as_ref()
            .and_then(WindowsExamineInventory::amd_display_name);
        return HostGpuSummary {
            name,
            gfx_target,
            therock_family,
        };
    }

    let linux_gfx_target = detect_linux_sysfs_gfx_target();
    let linux_name = detect_linux_primary_gpu_name();
    let wsl_display_probe = if linux_gfx_target.is_none() || linux_name.is_none() {
        detect_wsl_windows_display_probe_text()
    } else {
        None
    };
    let gfx_target = linux_gfx_target.or_else(|| {
        wsl_display_probe
            .as_deref()
            .and_then(parse_windows_display_gfx_target)
    });
    let therock_family = gfx_target.as_deref().and_then(normalize_therock_family);
    let name = linux_name.or_else(|| {
        wsl_display_probe
            .as_deref()
            .and_then(parse_windows_display_name)
    });
    HostGpuSummary {
        name,
        gfx_target,
        therock_family,
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
fn detect_host_gpu_summary_fast(_paths: Option<&AppPaths>) -> HostGpuSummary {
    HostGpuSummary::default()
}

#[allow(dead_code)]
fn detect_host_gpu_summary_full(paths: Option<&AppPaths>) -> HostGpuSummary {
    let windows_inventory = detect_windows_examine_inventory();
    let wsl = detect_wsl_summary();
    let gfx_target =
        detect_host_gfx_target_with_context(windows_inventory.as_ref(), wsl.as_ref(), paths);
    let therock_family = gfx_target.as_deref().and_then(normalize_therock_family);
    let name = detect_host_gpu_name_with_context(windows_inventory.as_ref(), wsl.as_ref());
    HostGpuSummary {
        name,
        gfx_target,
        therock_family,
    }
}

fn detect_host_gpu_name_with_context(
    windows_inventory: Option<&WindowsExamineInventory>,
    wsl: Option<&WslSummary>,
) -> Option<String> {
    windows_inventory
        .and_then(WindowsExamineInventory::amd_display_name)
        .or_else(detect_linux_primary_gpu_name)
        .or_else(|| detect_wsl_windows_display_name(wsl))
}

pub fn detect_managed_therock_family(paths: &AppPaths) -> Option<String> {
    newer_therock_family(
        newest_therock_family_in_manifest_dir(&paths.data_dir.join("runtimes").join("registry")),
        newest_therock_family_in_engine_manifests(paths),
    )
    .map(|(_, family)| family)
}

fn newest_therock_family_in_engine_manifests(paths: &AppPaths) -> Option<(u128, String)> {
    let engines_dir = paths.data_dir.join("engines");
    let entries = fs::read_dir(engines_dir).ok()?;
    let mut best = None;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        best = newer_therock_family(
            best,
            newest_therock_family_in_manifest_dir(&path.join("manifests")),
        );
    }
    best
}

fn newest_therock_family_in_manifest_dir(path: &Path) -> Option<(u128, String)> {
    let entries = fs::read_dir(path).ok()?;
    let mut best = None;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Some(record) = fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<TheRockFamilyManifest>(&bytes).ok())
        else {
            continue;
        };
        let Some(family) = record.therock_family() else {
            continue;
        };
        best = newer_therock_family(
            best,
            Some((record.installed_at_unix_ms.unwrap_or(0), family)),
        );
    }
    best
}

fn detect_managed_therock_sdk_gfx_target(paths: &AppPaths) -> Option<String> {
    managed_therock_sdk_probe_candidates(&paths.data_dir.join("runtimes").join("registry"))
        .into_iter()
        .find_map(|candidate| {
            let tool = managed_sdk_tool_path(&candidate.bin_path, "rocm_agent_enumerator")?;
            let mut envs = Vec::new();
            if let Some(ld_library_path) = managed_sdk_ld_library_path(&candidate) {
                envs.push(("LD_LIBRARY_PATH", ld_library_path));
            }
            capture_optional_path_command_with_env(&tool, &[], &envs, OPTIONAL_COMMAND_TIMEOUT)
                .and_then(|output| extract_first_gfx_token(&output))
        })
}

#[derive(Debug, Clone, Default)]
pub struct ManagedRuntimeEnvironment {
    pub rocm_root: Option<PathBuf>,
    pub path_entries: Vec<PathBuf>,
    pub library_entries: Vec<PathBuf>,
}

pub fn active_managed_therock_environment(
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<Option<ManagedRuntimeEnvironment>> {
    Ok(select_active_managed_therock_record(paths, config)
        .map(|record| managed_therock_environment_from_record(&record)))
}

/// Channel (`"release"`/`"nightly"`) of the active managed TheRock runtime.
///
/// Reflects the channel recorded at install time. Returns `None` when there is no managed
/// runtime (system or legacy ROCm) or the registry record predates channel recording.
pub fn active_managed_therock_channel(
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Result<Option<String>> {
    Ok(select_active_managed_therock_record(paths, config).and_then(|record| record.channel))
}

/// Pick the active managed TheRock runtime record: the one matching
/// `config.active_runtime_key`, falling back to the most recently installed.
fn select_active_managed_therock_record(
    paths: &AppPaths,
    config: &RocmCliConfig,
) -> Option<TheRockFamilyManifest> {
    let registry_dir = paths.data_dir.join("runtimes").join("registry");
    let mut records = managed_therock_environment_records(&registry_dir);
    if records.is_empty() {
        return None;
    }

    if let Some(active_key) = config.active_runtime_key.as_deref()
        && let Some((_, record)) = records.iter().find(|(_, record)| {
            record
                .runtime_key
                .as_deref()
                .is_some_and(|key| key.eq_ignore_ascii_case(active_key))
                || record
                    .runtime_id
                    .as_deref()
                    .is_some_and(|id| id.eq_ignore_ascii_case(active_key))
        })
    {
        return Some(record.clone());
    }

    records.sort_by_key(|(_, record)| std::cmp::Reverse(record.installed_at_unix_ms.unwrap_or(0)));
    records.into_iter().next().map(|(_, record)| record)
}

pub fn prepend_runtime_paths(
    entries: &[PathBuf],
    current: Option<OsString>,
) -> Result<Option<OsString>> {
    let mut parts = Vec::new();
    for entry in entries {
        push_existing_runtime_path(&mut parts, entry.clone());
    }
    if let Some(current) = current
        && !current.is_empty()
    {
        for entry in std::env::split_paths(&current) {
            push_existing_runtime_path(&mut parts, entry);
        }
    }
    if parts.is_empty() {
        Ok(None)
    } else {
        std::env::join_paths(parts)
            .map(Some)
            .context("failed to join runtime environment paths")
    }
}

fn managed_therock_environment_records(
    registry_dir: &Path,
) -> Vec<(PathBuf, TheRockFamilyManifest)> {
    let Ok(entries) = fs::read_dir(registry_dir) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                return None;
            }
            let record = fs::read(&path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<TheRockFamilyManifest>(&bytes).ok())?;
            (record.looks_like_therock()
                && record.rocm_sdk.as_ref().is_some_and(|sdk| sdk.import_ok))
            .then_some((path, record))
        })
        .collect()
}

fn managed_therock_environment_from_record(
    record: &TheRockFamilyManifest,
) -> ManagedRuntimeEnvironment {
    let mut env = ManagedRuntimeEnvironment::default();
    let sdk = record.rocm_sdk.as_ref();
    env.rocm_root = sdk
        .and_then(|sdk| sdk.root_path.clone())
        .or_else(|| record.install_root.clone());

    if let Some(sdk) = sdk {
        if let Some(bin_path) = sdk.bin_path.as_ref() {
            push_existing_runtime_path(&mut env.path_entries, bin_path.clone());
        }
        for path in &sdk.bin_paths {
            push_existing_runtime_path(&mut env.path_entries, path.clone());
        }
        for path in &sdk.library_paths {
            push_existing_runtime_path(&mut env.library_entries, path.clone());
        }
        if let Some(root_path) = sdk.root_path.as_ref() {
            collect_runtime_environment_paths(root_path, &mut env);
        }
        for root_path in &sdk.runtime_roots {
            collect_runtime_environment_paths(root_path, &mut env);
        }
    }
    if let Some(install_root) = record.install_root.as_ref() {
        collect_runtime_environment_paths(install_root, &mut env);
    }
    if runtime_is_linux() {
        push_existing_runtime_path(&mut env.library_entries, PathBuf::from("/usr/lib/wsl/lib"));
    }
    env
}

fn collect_runtime_environment_paths(root: &Path, env: &mut ManagedRuntimeEnvironment) {
    for path in [
        root.join("bin"),
        root.join("lib"),
        root.join("lib64"),
        root.join("lib").join("rocm_sysdeps").join("lib"),
    ] {
        if !path.is_dir() {
            continue;
        }
        if path.file_name().and_then(|value| value.to_str()) == Some("bin") {
            push_existing_runtime_path(&mut env.path_entries, path.clone());
        }
        push_existing_runtime_path(&mut env.library_entries, path);
    }
}

fn push_existing_runtime_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !path.exists() || paths.iter().any(|existing| existing == &path) {
        return;
    }
    paths.push(path);
}

fn managed_therock_sdk_probe_candidates(registry_dir: &Path) -> Vec<TheRockSdkProbeCandidate> {
    let Ok(entries) = fs::read_dir(registry_dir) else {
        return Vec::new();
    };
    let mut candidates = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Some(record) = fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<TheRockFamilyManifest>(&bytes).ok())
        else {
            continue;
        };
        if !record.looks_like_therock() {
            continue;
        }
        let Some(sdk) = record.rocm_sdk else {
            continue;
        };
        if !sdk.import_ok {
            continue;
        }
        let Some(root_path) = sdk.root_path else {
            continue;
        };
        let Some(bin_path) = sdk.bin_path else {
            continue;
        };
        candidates.push(TheRockSdkProbeCandidate {
            installed_at_unix_ms: record.installed_at_unix_ms.unwrap_or(0),
            site_packages: sdk.site_packages,
            root_path,
            bin_path,
        });
    }
    candidates.sort_by_key(|candidate| std::cmp::Reverse(candidate.installed_at_unix_ms));
    candidates
}

fn managed_sdk_tool_path(bin_path: &Path, tool: &str) -> Option<PathBuf> {
    let mut names = vec![tool.to_owned()];
    if runtime_is_windows() {
        names.push(format!("{tool}.exe"));
    }
    names.push(format!("{tool}.cmd"));
    names.push(format!("{tool}.bat"));
    names
        .into_iter()
        .map(|name| bin_path.join(name))
        .find(|path| path.is_file())
}

fn managed_sdk_ld_library_path(candidate: &TheRockSdkProbeCandidate) -> Option<OsString> {
    let mut paths = Vec::new();
    collect_sdk_library_paths(&candidate.root_path, &mut paths);
    if let Some(site_packages) = candidate.site_packages.as_deref()
        && let Ok(entries) = fs::read_dir(site_packages)
    {
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if name.starts_with("_rocm_sdk_") {
                collect_sdk_library_paths(&path, &mut paths);
            }
        }
    }
    let wsl_lib = PathBuf::from("/usr/lib/wsl/lib");
    if wsl_lib.is_dir() {
        paths.push(wsl_lib);
    }
    if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH")
        && !existing.is_empty()
    {
        paths.extend(std::env::split_paths(&existing));
    }
    if paths.is_empty() {
        None
    } else {
        std::env::join_paths(paths).ok()
    }
}

fn collect_sdk_library_paths(root: &Path, paths: &mut Vec<PathBuf>) {
    for path in [
        root.join("bin"),
        root.join("lib"),
        root.join("lib64"),
        root.join("lib").join("rocm_sysdeps").join("lib"),
    ] {
        if path.is_dir() {
            paths.push(path);
        }
    }
}

fn newer_therock_family(
    left: Option<(u128, String)>,
    right: Option<(u128, String)>,
) -> Option<(u128, String)> {
    match (left, right) {
        (Some(left), Some(right)) if left.0 > right.0 => Some(left),
        (Some(_) | None, Some(right)) => Some(right),
        (Some(left), None) => Some(left),
        (None, None) => None,
    }
}

#[derive(Debug, Clone, Deserialize)]
struct TheRockFamilyManifest {
    #[serde(default)]
    runtime_key: Option<String>,
    #[serde(default)]
    runtime_id: Option<String>,
    #[serde(default)]
    family: Option<String>,
    #[serde(default)]
    therock_family: Option<String>,
    #[serde(default)]
    channel: Option<String>,
    #[serde(default)]
    rocm_sdk: Option<TheRockSdkProbeManifest>,
    #[serde(default)]
    install_root: Option<PathBuf>,
    #[serde(default)]
    installed_at_unix_ms: Option<u128>,
}

impl TheRockFamilyManifest {
    fn therock_family(&self) -> Option<String> {
        if !self.looks_like_therock() {
            return None;
        }
        self.therock_family
            .as_deref()
            .or(self.family.as_deref())
            .and_then(normalize_therock_family)
    }

    fn looks_like_therock(&self) -> bool {
        self.therock_family.is_some()
            || self
                .runtime_id
                .as_deref()
                .is_some_and(|runtime_id| runtime_id.to_ascii_lowercase().starts_with("therock-"))
    }
}

#[derive(Debug, Clone, Deserialize)]
struct TheRockSdkProbeManifest {
    #[serde(default)]
    import_ok: bool,
    #[serde(default)]
    site_packages: Option<PathBuf>,
    #[serde(default)]
    root_path: Option<PathBuf>,
    #[serde(default)]
    bin_path: Option<PathBuf>,
    #[serde(default)]
    runtime_roots: Vec<PathBuf>,
    #[serde(default)]
    bin_paths: Vec<PathBuf>,
    #[serde(default)]
    library_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
struct TheRockSdkProbeCandidate {
    installed_at_unix_ms: u128,
    site_packages: Option<PathBuf>,
    root_path: PathBuf,
    bin_path: PathBuf,
}

pub fn detect_host_gfx_target() -> Option<String> {
    let paths = AppPaths::discover().ok();
    detect_host_gpu_summary_fast(paths.as_ref()).gfx_target
}

fn detect_examine_gfx_target_fast(
    windows_inventory: Option<&WindowsExamineInventory>,
) -> Option<String> {
    if runtime_is_windows() {
        return detect_windows_display_gfx_target_with_inventory(windows_inventory);
    }

    if runtime_is_linux() {
        return detect_linux_sysfs_gfx_target().or_else(detect_wsl_windows_display_gfx_target_fast);
    }

    None
}

#[allow(dead_code)]
fn detect_host_gfx_target_with_context(
    windows_inventory: Option<&WindowsExamineInventory>,
    wsl: Option<&WslSummary>,
    paths: Option<&AppPaths>,
) -> Option<String> {
    if runtime_is_windows() {
        return detect_windows_display_gfx_target_with_inventory(windows_inventory)
            .or_else(|| {
                capture_optional_command("rocm_agent_enumerator", &[])
                    .and_then(|output| extract_first_gfx_token(&output))
            })
            .or_else(|| {
                capture_optional_command("rocminfo", &[])
                    .and_then(|output| extract_first_gfx_token(&output))
            });
    }

    detect_linux_sysfs_gfx_target()
        .or_else(|| {
            capture_optional_command("rocm_agent_enumerator", &[])
                .and_then(|output| extract_first_gfx_token(&output))
        })
        .or_else(|| {
            capture_optional_command("rocminfo", &[])
                .and_then(|output| extract_first_gfx_token(&output))
        })
        .or_else(|| paths.and_then(detect_managed_therock_sdk_gfx_target))
        .or_else(|| detect_wsl_windows_display_gfx_target(wsl))
        .or_else(|| detect_windows_display_gfx_target_with_inventory(windows_inventory))
}

pub fn extract_first_gfx_token(text: &str) -> Option<String> {
    text.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '-' || ch == '_'))
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .find_map(|token| {
            let normalized = token.to_ascii_lowercase();
            if normalized.starts_with("gfx") {
                Some(normalized)
            } else {
                None
            }
        })
}

pub fn normalize_therock_family(value: &str) -> Option<String> {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }

    let target = extract_first_gfx_token(&normalized).unwrap_or(normalized);
    match target.as_str() {
        "gfx101x-dgpu" => Some("gfx101X-dgpu".to_owned()),
        "gfx103x-dgpu" => Some("gfx103X-dgpu".to_owned()),
        "gfx110x-all" => Some("gfx110X-all".to_owned()),
        "gfx120x-all" => Some("gfx120X-all".to_owned()),
        "gfx90x-dgpu" => Some("gfx90X-dgpu".to_owned()),
        "gfx94x-dcgpu" => Some("gfx94X-dcgpu".to_owned()),
        "gfx950-dcgpu" => Some("gfx950-dcgpu".to_owned()),
        value if value.starts_with("gfx101") => Some("gfx101X-dgpu".to_owned()),
        value if value.starts_with("gfx103") => Some("gfx103X-dgpu".to_owned()),
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103" => Some("gfx110X-all".to_owned()),
        value if value.starts_with("gfx1150") => Some("gfx1150".to_owned()),
        value if value.starts_with("gfx1151") => Some("gfx1151".to_owned()),
        value if value.starts_with("gfx1152") => Some("gfx1152".to_owned()),
        value if value.starts_with("gfx1153") => Some("gfx1153".to_owned()),
        "gfx1200" | "gfx1201" => Some("gfx120X-all".to_owned()),
        value if value.starts_with("gfx900") => Some("gfx900".to_owned()),
        value if value.starts_with("gfx906") => Some("gfx906".to_owned()),
        value if value.starts_with("gfx908") => Some("gfx908".to_owned()),
        value if value.starts_with("gfx90a") => Some("gfx90a".to_owned()),
        value if value.starts_with("gfx950") => Some("gfx950-dcgpu".to_owned()),
        value
            if value.starts_with("gfx942")
                || value.starts_with("gfx94")
                || value.starts_with("gfx9-4") =>
        {
            Some("gfx94X-dcgpu".to_owned())
        }
        value if value.starts_with("gfx90") => Some("gfx90X-dcgpu".to_owned()),
        _ => None,
    }
}

/// The TheRock package families the CLI recognizes.
///
/// This is the full set of values [`normalize_therock_family`] can produce. Used
/// to tell the user which `--family` values are valid when GPU auto-detection
/// cannot resolve an installable runtime. Whether a given family currently has
/// published wheels depends on the channel and release; recognition here does
/// not guarantee availability.
///
/// Kept in sync with [`normalize_therock_family`] by
/// `known_therock_families_all_round_trip` — every entry must normalize back to
/// itself.
pub const fn known_therock_families() -> &'static [&'static str] {
    &[
        "gfx90X-dgpu",
        "gfx90X-dcgpu",
        "gfx900",
        "gfx906",
        "gfx908",
        "gfx90a",
        "gfx94X-dcgpu",
        "gfx950-dcgpu",
        "gfx101X-dgpu",
        "gfx103X-dgpu",
        "gfx110X-all",
        "gfx1150",
        "gfx1151",
        "gfx1152",
        "gfx1153",
        "gfx120X-all",
    ]
}

fn capture_optional_command(program: &str, args: &[&str]) -> Option<String> {
    capture_optional_command_with_timeout(program, args, OPTIONAL_COMMAND_TIMEOUT)
}

fn capture_optional_command_with_timeout(
    program: &str,
    args: &[&str],
    timeout: Duration,
) -> Option<String> {
    for candidate in tool_path_candidates(program) {
        if let Some(output) =
            capture_optional_command_candidate_with_timeout(Path::new(&candidate), args, timeout)
        {
            return Some(output);
        }
    }
    None
}

fn capture_optional_command_candidate_with_timeout(
    program: &Path,
    args: &[&str],
    timeout: Duration,
) -> Option<String> {
    let mut child = match Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            debug_command_capture_failure(program, "spawn", &error.to_string());
            return None;
        }
    };
    let mut stdout_reader = child.stdout.take().map(|mut stdout| {
        thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = stdout.read_to_end(&mut bytes);
            bytes
        })
    });

    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let bytes = stdout_reader
                    .take()
                    .map(|reader| reader.join().unwrap_or_default())
                    .unwrap_or_default();
                if status.success() {
                    return String::from_utf8(bytes).ok();
                }
                debug_command_capture_failure(program, "exit", &format!("status {status}"));
                return None;
            }
            Ok(None) if start.elapsed() < timeout => {
                thread::sleep(Duration::from_millis(25));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                if let Some(reader) = stdout_reader.take() {
                    let _ = reader.join();
                }
                debug_command_capture_failure(program, "timeout", "timed out");
                return None;
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                if let Some(reader) = stdout_reader.take() {
                    let _ = reader.join();
                }
                debug_command_capture_failure(program, "wait", "failed to wait");
                return None;
            }
        }
    }
}

fn debug_command_capture_failure(program: &Path, stage: &str, detail: &str) {
    if !env_flag("ROCM_CLI_DEBUG_COMMAND_CAPTURE") {
        return;
    }
    eprintln!(
        "rocm debug: command capture {stage} failed for {}: {detail}",
        program.display()
    );
}

fn capture_optional_path_command_with_env(
    program: &Path,
    args: &[&str],
    envs: &[(&str, OsString)],
    timeout: Duration,
) -> Option<String> {
    let output_path = std::env::temp_dir().join(format!(
        "rocm-cli-command-{}-{}.out",
        std::process::id(),
        unix_time_millis()
    ));
    let output_file = fs::File::create(&output_path).ok()?;
    let mut command = Command::new(program);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(output_file))
        .stderr(Stdio::null());
    for (key, value) in envs {
        command.env(key, value);
    }
    let Ok(mut child) = command.spawn() else {
        let _ = fs::remove_file(&output_path);
        return None;
    };

    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let bytes = if status.success() {
                    fs::read(&output_path).ok()
                } else {
                    None
                };
                let _ = fs::remove_file(&output_path);
                return bytes.and_then(|bytes| String::from_utf8(bytes).ok());
            }
            Ok(None) if start.elapsed() < timeout => {
                thread::sleep(Duration::from_millis(25));
            }
            Ok(None) | Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = fs::remove_file(&output_path);
                return None;
            }
        }
    }
}

fn tool_on_path(program: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|path| {
        std::env::split_paths(&path).any(|dir| {
            tool_path_candidates(program)
                .into_iter()
                .any(|name| dir.join(name).is_file())
        })
    })
}

fn tool_path_candidates(program: &str) -> Vec<String> {
    let path = Path::new(program);
    if path.extension().is_some() || !runtime_is_windows() {
        return vec![program.to_owned()];
    }
    let mut names = Vec::new();
    names.push(program.to_owned());
    let pathext = std::env::var("PATHEXT").unwrap_or_else(|_| ".COM;.EXE;.BAT;.CMD".to_owned());
    for ext in pathext
        .split(';')
        .map(str::trim)
        .filter(|ext| !ext.is_empty())
    {
        names.push(format!("{program}{ext}"));
        names.push(format!("{program}{}", ext.to_ascii_lowercase()));
    }
    names.extend(windows_absolute_tool_candidates(program));
    names.sort();
    names.dedup();
    names
}

fn windows_absolute_tool_candidates(program: &str) -> Vec<String> {
    if !runtime_is_windows() {
        return Vec::new();
    }
    let program = program.trim().to_ascii_lowercase();
    let system_root = std::env::var("SystemRoot")
        .or_else(|_| std::env::var("WINDIR"))
        .unwrap_or_else(|_| r"C:\Windows".to_owned());
    match program.as_str() {
        "cmd" | "cmd.exe" => vec![format!(r"{system_root}\System32\cmd.exe")],
        "pnputil" | "pnputil.exe" => vec![format!(r"{system_root}\System32\pnputil.exe")],
        "powershell" | "powershell.exe" => vec![
            format!(r"{system_root}\System32\WindowsPowerShell\v1.0\powershell.exe"),
            "powershell.exe".to_owned(),
        ],
        "pwsh" | "pwsh.exe" => vec!["pwsh.exe".to_owned()],
        _ => Vec::new(),
    }
}

#[cfg(windows)]
fn detect_windows_display_gfx_target() -> Option<String> {
    if !runtime_is_windows() {
        return None;
    }
    capture_optional_command_with_timeout(
        "powershell",
        &[
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            WINDOWS_VIDEO_CONTROLLER_INVENTORY_SCRIPT,
        ],
        WINDOWS_INVENTORY_QUERY_TIMEOUT,
    )
    .map(|output| parse_windows_examine_inventory(&output).display_gfx_probe_text())
    .and_then(|output| parse_windows_display_gfx_target(&output))
}

#[cfg(not(windows))]
const fn detect_windows_display_gfx_target() -> Option<String> {
    None
}

fn detect_windows_display_gfx_target_with_inventory(
    windows_inventory: Option<&WindowsExamineInventory>,
) -> Option<String> {
    if runtime_is_windows() {
        return windows_inventory
            .and_then(WindowsExamineInventory::display_gfx_target)
            .or_else(|| {
                if windows_inventory.is_none() {
                    detect_windows_display_gfx_target()
                } else {
                    None
                }
            });
    }

    detect_windows_display_gfx_target()
}

fn detect_wsl_windows_display_gfx_target(wsl: Option<&WslSummary>) -> Option<String> {
    if !runtime_is_linux() || wsl.is_none() {
        return None;
    }

    detect_wsl_windows_display_gfx_target_fast()
}

fn detect_wsl_windows_display_gfx_target_fast() -> Option<String> {
    detect_wsl_windows_display_probe_text()
        .as_deref()
        .and_then(parse_windows_display_gfx_target)
}

fn detect_wsl_windows_display_name(wsl: Option<&WslSummary>) -> Option<String> {
    if !runtime_is_linux() || !wsl.is_some_and(|summary| summary.is_wsl) {
        return None;
    }

    detect_wsl_windows_display_name_fast()
}

fn detect_wsl_windows_display_name_fast() -> Option<String> {
    detect_wsl_windows_display_probe_text()
        .as_deref()
        .and_then(parse_windows_display_name)
}

fn detect_wsl_windows_display_probe_text() -> Option<String> {
    if !is_wsl_environment_fast() {
        return None;
    }

    capture_optional_command_with_timeout(
        "powershell.exe",
        &[
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            WINDOWS_VIDEO_CONTROLLER_INVENTORY_SCRIPT,
        ],
        WINDOWS_INVENTORY_QUERY_TIMEOUT,
    )
    .map(|output| {
        parse_windows_examine_inventory(&output)
            .display_gfx_probe_text()
            .trim()
            .to_owned()
    })
    .filter(|output| !output.is_empty())
}

fn is_wsl_environment_fast() -> bool {
    if !runtime_is_linux() {
        return false;
    }
    Path::new("/dev/dxg").exists()
        || fs::read_to_string("/proc/version")
            .is_ok_and(|text| text.to_ascii_lowercase().contains("microsoft"))
}

#[cfg(target_os = "linux")]
fn detect_linux_primary_gpu_name() -> Option<String> {
    if !runtime_is_linux() {
        return None;
    }

    let entries = fs::read_dir("/sys/class/drm").ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let device_dir = entry.path().join("device");
        if !is_amdgpu_device(&device_dir) {
            continue;
        }
        for file_name in ["product_name", "product", "model"] {
            let Some(value) = fs::read_to_string(device_dir.join(file_name)).ok() else {
                continue;
            };
            let value = value.trim();
            if !value.is_empty() {
                return Some(value.to_owned());
            }
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
const fn detect_linux_primary_gpu_name() -> Option<String> {
    None
}

fn parse_windows_display_gfx_target(text: &str) -> Option<String> {
    let mut name_fallback = None;
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let (name, pnp_id) = line.split_once('\t').unwrap_or((line, ""));
        if let Some(device_id) = amd_pci_device_id_from_pnp_id(pnp_id)
            && let Some(target) = gfx_target_from_amd_pci_device_id(&device_id)
        {
            return Some(target.to_owned());
        }
        if name_fallback.is_none() {
            name_fallback = gfx_target_from_amd_marketing_name(name).map(str::to_owned);
        }
    }
    name_fallback
}

fn parse_windows_display_name(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .find_map(|line| {
            let (name, _) = line.split_once('\t').unwrap_or((line, ""));
            let name = name.trim();
            (!name.is_empty()).then(|| name.to_owned())
        })
}

fn amd_pci_device_id_from_pnp_id(pnp_id: &str) -> Option<String> {
    let upper = pnp_id.to_ascii_uppercase();
    if !upper.contains("VEN_1002") {
        return None;
    }
    let start = upper.find("DEV_")? + "DEV_".len();
    let device_id = upper[start..]
        .chars()
        .take_while(char::is_ascii_hexdigit)
        .take(4)
        .collect::<String>();
    if device_id.len() == 4 {
        Some(device_id.to_ascii_lowercase())
    } else {
        None
    }
}

fn gfx_target_from_amd_pci_device_id(device_id: &str) -> Option<&'static str> {
    match device_id.to_ascii_lowercase().as_str() {
        // Navi 21 / 22 / 23 / 24: Radeon RX 6000 desktop and mobile ASICs.
        "73a0" | "73a1" | "73a2" | "73a3" | "73a5" | "73a8" | "73a9" | "73ab" | "73ac" | "73ad"
        | "73ae" | "73af" => Some("gfx1030"),
        "73c0" | "73c1" | "73c3" => Some("gfx1031"),
        "73e0" | "73e1" | "73e2" | "73e3" | "73e8" | "73e9" | "73ea" | "73eb" | "73ec" | "73ed"
        | "73ef" => Some("gfx1032"),
        "7420" | "7421" | "7422" | "7423" | "7424" | "743f" => Some("gfx1034"),
        // RDNA2 APUs.
        "163f" => Some("gfx1033"),
        "164d" | "1681" => Some("gfx1035"),
        "164e" => Some("gfx1036"),
        // RDNA3 APUs.
        "15bf" | "164f" | "1900" | "1901" => Some("gfx1103"),
        // RDNA3.5 APUs with public PCI IDs that map cleanly to one gfx target.
        "1114" => Some("gfx1152"),
        // Navi 48: Radeon RX 9070 / 9070 XT / 9070 GRE.
        "7550" => Some("gfx1201"),
        _ => None,
    }
}

fn gfx_target_from_amd_marketing_name(name: &str) -> Option<&'static str> {
    let lower = name.to_ascii_lowercase();
    let normalized = normalize_marketing_name_for_match(&lower);
    for entry in AMD_MARKETING_GFX_TARGETS {
        if marketing_name_contains(&normalized, entry.pattern) {
            return Some(entry.gfx_target);
        }
    }
    None
}

#[derive(Debug, Clone, Copy)]
struct AmdMarketingGfxTarget {
    pattern: &'static str,
    gfx_target: &'static str,
}

const AMD_MARKETING_GFX_TARGETS: &[AmdMarketingGfxTarget] = &[
    // RDNA4 discrete.
    AmdMarketingGfxTarget {
        pattern: "ai pro r9700",
        gfx_target: "gfx1201",
    },
    AmdMarketingGfxTarget {
        pattern: "ai pro r9600",
        gfx_target: "gfx1201",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 9070",
        gfx_target: "gfx1201",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 9060",
        gfx_target: "gfx1200",
    },
    // RDNA3 discrete.
    AmdMarketingGfxTarget {
        pattern: "pro w7900",
        gfx_target: "gfx1100",
    },
    AmdMarketingGfxTarget {
        pattern: "pro w7800",
        gfx_target: "gfx1100",
    },
    AmdMarketingGfxTarget {
        pattern: "pro w7700",
        gfx_target: "gfx1101",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 7900",
        gfx_target: "gfx1100",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 7800",
        gfx_target: "gfx1101",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 7700",
        gfx_target: "gfx1101",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 7600",
        gfx_target: "gfx1102",
    },
    // RDNA2 discrete. Mobile names that share number prefixes are listed before desktop.
    AmdMarketingGfxTarget {
        pattern: "pro w6800",
        gfx_target: "gfx1030",
    },
    AmdMarketingGfxTarget {
        pattern: "pro w6600",
        gfx_target: "gfx1032",
    },
    AmdMarketingGfxTarget {
        pattern: "pro v620",
        gfx_target: "gfx1030",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6850m",
        gfx_target: "gfx1031",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6800m",
        gfx_target: "gfx1031",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6700m",
        gfx_target: "gfx1031",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6700s",
        gfx_target: "gfx1032",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6650m",
        gfx_target: "gfx1032",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6600m",
        gfx_target: "gfx1032",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6600s",
        gfx_target: "gfx1032",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6500m",
        gfx_target: "gfx1034",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6400m",
        gfx_target: "gfx1034",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6300m",
        gfx_target: "gfx1034",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6950",
        gfx_target: "gfx1030",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6900",
        gfx_target: "gfx1030",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6800",
        gfx_target: "gfx1030",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6750",
        gfx_target: "gfx1031",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6700",
        gfx_target: "gfx1031",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6650",
        gfx_target: "gfx1032",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6600",
        gfx_target: "gfx1032",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6500",
        gfx_target: "gfx1034",
    },
    AmdMarketingGfxTarget {
        pattern: "rx 6400",
        gfx_target: "gfx1034",
    },
    // RDNA3.5 / Strix Halo APUs.
    AmdMarketingGfxTarget {
        pattern: "8060s",
        gfx_target: "gfx1151",
    },
    AmdMarketingGfxTarget {
        pattern: "8050s",
        gfx_target: "gfx1151",
    },
    AmdMarketingGfxTarget {
        pattern: "8040s",
        gfx_target: "gfx1151",
    },
    // RDNA3.5 APUs.
    AmdMarketingGfxTarget {
        pattern: "890m",
        gfx_target: "gfx1150",
    },
    AmdMarketingGfxTarget {
        pattern: "880m",
        gfx_target: "gfx1150",
    },
    AmdMarketingGfxTarget {
        pattern: "860m",
        gfx_target: "gfx1152",
    },
    AmdMarketingGfxTarget {
        pattern: "840m",
        gfx_target: "gfx1152",
    },
    AmdMarketingGfxTarget {
        pattern: "820m",
        gfx_target: "gfx1153",
    },
    // RDNA3 APUs.
    AmdMarketingGfxTarget {
        pattern: "780m",
        gfx_target: "gfx1103",
    },
    AmdMarketingGfxTarget {
        pattern: "760m",
        gfx_target: "gfx1103",
    },
    AmdMarketingGfxTarget {
        pattern: "740m",
        gfx_target: "gfx1103",
    },
    // RDNA2 APUs.
    AmdMarketingGfxTarget {
        pattern: "680m",
        gfx_target: "gfx1035",
    },
    AmdMarketingGfxTarget {
        pattern: "660m",
        gfx_target: "gfx1035",
    },
    AmdMarketingGfxTarget {
        pattern: "610m",
        gfx_target: "gfx1036",
    },
    AmdMarketingGfxTarget {
        pattern: "steam deck",
        gfx_target: "gfx1033",
    },
    AmdMarketingGfxTarget {
        pattern: "van gogh",
        gfx_target: "gfx1033",
    },
];

fn normalize_marketing_name_for_match(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            'a'..='z' | '0'..='9' => ch,
            'A'..='Z' => ch.to_ascii_lowercase(),
            _ => ' ',
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn marketing_name_contains(normalized_name: &str, pattern: &str) -> bool {
    normalized_name
        .split_whitespace()
        .collect::<Vec<_>>()
        .windows(pattern.split_whitespace().count())
        .any(|window| window.join(" ") == pattern)
}

#[cfg(target_os = "linux")]
fn detect_linux_sysfs_gfx_target() -> Option<String> {
    if !runtime_is_linux() {
        return None;
    }

    detect_linux_kfd_gfx_target().or_else(detect_linux_drm_ip_discovery_gfx_target)
}

#[cfg(not(target_os = "linux"))]
const fn detect_linux_sysfs_gfx_target() -> Option<String> {
    None
}

#[cfg(target_os = "linux")]
fn detect_linux_kfd_gfx_target() -> Option<String> {
    let nodes_dir = Path::new("/sys/class/kfd/kfd/topology/nodes");
    let entries = fs::read_dir(nodes_dir).ok()?;
    for entry in entries.flatten() {
        let Some(value) = fs::read_to_string(entry.path().join("gfx_target_version")).ok() else {
            continue;
        };
        let Some(token) = parse_linux_kfd_gfx_target(value.trim()) else {
            continue;
        };
        return Some(token);
    }
    None
}

/// AMD GPU device ordinals usable for a GPU-required launch on this host, after
/// applying the runtime visibility mask (`HIP_VISIBLE_DEVICES`, then
/// `ROCR_VISIBLE_DEVICES`).
///
/// `Some(indices)` is an authoritative answer: an empty vector means no GPU is
/// usable, so a GPU-required launch must fail rather than fall back to CPU or
/// assume device 0. `None` means availability could not be probed on this
/// platform (e.g. the KFD device exists but its topology is unreadable, or a
/// non-Linux target) and callers should not block a launch on this basis.
#[must_use]
pub fn usable_amd_gpu_indices() -> Option<Vec<u32>> {
    probe_usable_amd_gpu_indices()
}

/// Whether at least one AMD GPU is usable for a GPU-required launch.
///
/// `false` only when the probe authoritatively reports zero usable devices; an
/// unprobeable platform reports `true` so it does not block launches (see
/// [`usable_amd_gpu_indices`]).
#[must_use]
pub fn has_usable_amd_gpu() -> bool {
    usable_amd_gpu_indices().is_none_or(|indices| !indices.is_empty())
}

#[cfg(target_os = "linux")]
fn probe_usable_amd_gpu_indices() -> Option<Vec<u32>> {
    let present =
        combine_amd_gpu_counts(linux_kfd_gpu_node_count(), linux_drm_amdgpu_card_count())?;
    usable_amd_gpu_indices_from(present, visibility_mask_from_env())
}

#[cfg(not(target_os = "linux"))]
fn probe_usable_amd_gpu_indices() -> Option<Vec<u32>> {
    None
}

/// Combine the KFD-topology and DRM-card AMD GPU counts into one "GPUs present"
/// figure, taking the larger available count. Some hosts (e.g. Strix Halo APUs)
/// enumerate the GPU only via DRM ip-discovery and report zero KFD GPU nodes, so
/// relying on KFD alone would wrongly conclude there is no GPU and block serving.
/// `None` only when NEITHER surface could be read (availability truly unknown).
#[cfg(any(target_os = "linux", test))]
fn combine_amd_gpu_counts(kfd: Option<usize>, drm: Option<usize>) -> Option<usize> {
    match (kfd, drm) {
        (Some(k), Some(d)) => Some(k.max(d)),
        (Some(n), None) | (None, Some(n)) => Some(n),
        (None, None) => None,
    }
}

/// Count AMD (`amdgpu`) primary DRM cards under `/sys/class/drm` (`card0`,
/// `card1`, …), skipping connector sub-nodes like `card0-DP-1`. `None` when the
/// DRM class dir can't be read; `Some(0)` when it is readable with no AMD card.
#[cfg(target_os = "linux")]
fn linux_drm_amdgpu_card_count() -> Option<usize> {
    let entries = fs::read_dir(Path::new("/sys/class/drm")).ok()?;
    let count = entries
        .flatten()
        .filter(|entry| {
            let card_path = entry.path();
            let Some(name) = card_path.file_name().and_then(|value| value.to_str()) else {
                return false;
            };
            name.starts_with("card")
                && !name.contains('-')
                && is_amdgpu_device(&card_path.join("device"))
        })
        .count();
    Some(count)
}

/// Count AMD GPU nodes in the KFD topology. `Some(0)` is an authoritative "no
/// GPU" (topology readable with no GPU node, or no KFD device at all); `None`
/// means the topology could not be read even though `/dev/kfd` exists, so
/// availability is unknown and must not be treated as zero.
#[cfg(target_os = "linux")]
fn linux_kfd_gpu_node_count() -> Option<usize> {
    let nodes_dir = Path::new("/sys/class/kfd/kfd/topology/nodes");
    match fs::read_dir(nodes_dir) {
        Ok(entries) => Some(
            entries
                .flatten()
                .filter(|entry| kfd_node_is_gpu(&entry.path()))
                .count(),
        ),
        Err(_) if Path::new("/dev/kfd").exists() => None,
        Err(_) => Some(0),
    }
}

/// A KFD topology node is a GPU (not the CPU node) when its
/// `gfx_target_version` is a nonzero value; CPU nodes report `0`.
#[cfg(target_os = "linux")]
fn kfd_node_is_gpu(node_dir: &Path) -> bool {
    fs::read_to_string(node_dir.join("gfx_target_version"))
        .ok()
        .is_some_and(|value| kfd_gfx_target_version_is_gpu(value.trim()))
}

#[cfg(any(target_os = "linux", test))]
fn kfd_gfx_target_version_is_gpu(value: &str) -> bool {
    value
        .trim()
        .parse::<u64>()
        .is_ok_and(|version| version != 0)
}

/// The active GPU visibility mask, preferring `HIP_VISIBLE_DEVICES` then
/// `ROCR_VISIBLE_DEVICES`. `None` when neither is set; an explicitly empty value
/// is returned as `Some("")` so callers can distinguish "unset" (all visible)
/// from "set to nothing" (all masked out).
///
/// Linux-only: its sole caller is the Linux probe. (Not `+ test` — no test
/// references it directly, so compiling it into a non-Linux test build would be
/// dead code, which the workspace lints deny.)
#[cfg(target_os = "linux")]
fn visibility_mask_from_env() -> Option<String> {
    ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"]
        .into_iter()
        .find_map(std::env::var_os)
        .map(|value| value.to_string_lossy().into_owned())
}

/// Apply a `HIP_VISIBLE_DEVICES`-style `mask` to the present device ordinals
/// (`0..present`). `None` means no mask is set (every present device is visible).
/// An empty value hides every device. A nonempty mask containing UUIDs or invalid
/// ordinals returns `None`: the ordinal-only probe cannot interpret it
/// authoritatively, so callers must not mistake it for "no GPU".
#[cfg(any(target_os = "linux", test))]
fn usable_amd_gpu_indices_from(present: usize, mask: Option<String>) -> Option<Vec<u32>> {
    let Some(mask) = mask else {
        return Some((0..present as u32).collect());
    };
    if mask.is_empty() {
        return Some(Vec::new());
    }
    let mut visible = Vec::new();
    for token in mask.split(',') {
        let index = token.trim().parse::<u32>().ok()?;
        if (index as usize) >= present {
            return None;
        }
        if !visible.contains(&index) {
            visible.push(index);
        }
    }
    Some(visible)
}

#[cfg(any(target_os = "linux", test))]
fn parse_linux_kfd_gfx_target(value: &str) -> Option<String> {
    if let Some(token) = extract_first_gfx_token(value) {
        return Some(token);
    }
    let digits = value.trim();
    if digits.is_empty() || !digits.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    match digits.len() {
        3 | 4 => Some(format!("gfx{digits}")),
        5 | 6 => {
            let raw: u32 = digits.parse().ok()?;
            let major = raw / 10_000;
            let minor = (raw / 100) % 100;
            let revision = raw % 100;
            if let Some(token) = gfx_target_from_gc_version(major, minor, revision) {
                return Some(token);
            }
            Some(format!("gfx{digits}"))
        }
        _ => None,
    }
}

#[cfg(target_os = "linux")]
fn detect_linux_drm_ip_discovery_gfx_target() -> Option<String> {
    let drm_dir = Path::new("/sys/class/drm");
    let entries = fs::read_dir(drm_dir).ok()?;
    for entry in entries.flatten() {
        let card_path = entry.path();
        let Some(card_name) = card_path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if !card_name.starts_with("card") || card_name.contains('-') {
            continue;
        }
        let device_dir = card_path.join("device");
        if !is_amdgpu_device(&device_dir) {
            continue;
        }
        let gc_root = device_dir.join("ip_discovery");
        let token = detect_ip_discovery_gc_target(&gc_root);
        if token.is_some() {
            return token;
        }
    }
    None
}

#[cfg(any(target_os = "linux", test))]
fn is_amdgpu_device(device_dir: &Path) -> bool {
    if let Ok(vendor) = fs::read_to_string(device_dir.join("vendor"))
        && vendor.trim().eq_ignore_ascii_case("0x1002")
    {
        return true;
    }
    if let Ok(uevent) = fs::read_to_string(device_dir.join("uevent")) {
        return uevent.lines().any(|line| line.trim() == "DRIVER=amdgpu");
    }
    false
}

#[cfg(any(target_os = "linux", test))]
fn detect_ip_discovery_gc_target(ip_discovery_dir: &Path) -> Option<String> {
    let die_entries = fs::read_dir(ip_discovery_dir.join("die")).ok()?;
    for die in die_entries.flatten() {
        let Some(gc_entries) = fs::read_dir(die.path().join("GC")).ok() else {
            continue;
        };
        for gc in gc_entries.flatten() {
            let block = gc.path();
            let Some(major) = fs::read_to_string(block.join("major"))
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok())
            else {
                continue;
            };
            let Some(minor) = fs::read_to_string(block.join("minor"))
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok())
            else {
                continue;
            };
            let Some(revision) = fs::read_to_string(block.join("revision"))
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok())
            else {
                continue;
            };
            if let Some(token) = gfx_target_from_gc_version(major, minor, revision) {
                return Some(token);
            }
        }
    }
    None
}

#[cfg(any(target_os = "linux", test))]
fn gfx_target_from_gc_version(major: u32, minor: u32, revision: u32) -> Option<String> {
    if major == 0 {
        return None;
    }
    Some(format!("gfx{major}{minor}{revision}"))
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum WatcherMode {
    Observe,
    Propose,
    Contained,
}

impl WatcherMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Observe => "observe",
            Self::Propose => "propose",
            Self::Contained => "contained",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BuiltinWatcherSpec {
    pub id: &'static str,
    pub summary: &'static str,
    pub trigger: &'static str,
    pub default_mode: WatcherMode,
    pub actions: &'static [&'static str],
}

const BUILTIN_WATCHERS: &[BuiltinWatcherSpec] = &[
    BuiltinWatcherSpec {
        id: "therock-update",
        summary: "Emit scheduled TheRock update reminders and proposals.",
        trigger: "schedule: every 6h",
        default_mode: WatcherMode::Observe,
        actions: &["remind_update_check", "queue_update_proposal"],
    },
    BuiltinWatcherSpec {
        id: "server-recover",
        summary: "Observe or restart failed managed services when restart metadata exists.",
        trigger: "event: managed_service_failed",
        default_mode: WatcherMode::Contained,
        actions: &["collect_failure_snapshot", "restart_managed_service"],
    },
    BuiltinWatcherSpec {
        id: "gpu-metrics",
        summary: "Record read-only local amd-smi GPU telemetry availability; no proposals or mutations.",
        trigger: "event: gpu.metrics availability/unavailability",
        default_mode: WatcherMode::Observe,
        actions: &["record_gpu_metrics"],
    },
    BuiltinWatcherSpec {
        id: "cache-warm",
        summary: "Queue reviewed artifact prefetch proposals for registry model artifacts.",
        trigger: "event: cache.warm",
        default_mode: WatcherMode::Propose,
        actions: &["queue_prefetch_proposal"],
    },
    BuiltinWatcherSpec {
        id: "driver-upgrade",
        summary: "Queue reviewed read-only driver install plans when a local driver update signal is received.",
        trigger: "event: update.available component=driver",
        default_mode: WatcherMode::Propose,
        actions: &["prepare_driver_plan"],
    },
    BuiltinWatcherSpec {
        id: "gpu-thermal-protect",
        summary: "Queue reviewed stop-serving proposals when GPU temperature or memory pressure is high.",
        trigger: "event: gpu.thermal_pressure or gpu.memory_pressure",
        default_mode: WatcherMode::Propose,
        actions: &["queue_stop_server_proposal"],
    },
];

pub const fn builtin_watchers() -> &'static [BuiltinWatcherSpec] {
    BUILTIN_WATCHERS
}

pub fn builtin_watcher(id: &str) -> Option<&'static BuiltinWatcherSpec> {
    builtin_watchers().iter().find(|watcher| watcher.id == id)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngineUserConfig {
    #[serde(default)]
    pub preferred_runtime_id: Option<String>,
    #[serde(default)]
    pub preferred_env_id: Option<String>,
    #[serde(default)]
    pub last_installed_runtime_id: Option<String>,
    #[serde(default)]
    pub last_installed_env_id: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WatcherUserConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub mode: Option<WatcherMode>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AutomationsConfig {
    #[serde(default)]
    pub daemon_enabled: bool,
    #[serde(default)]
    pub watchers: BTreeMap<String, WatcherUserConfig>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderUserConfig {
    #[serde(default)]
    pub enabled: bool,
}

pub const TELEMETRY_MODE_LOCAL: &str = "local";
pub const TELEMETRY_MODE_OFF: &str = "off";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    #[serde(default = "default_telemetry_mode")]
    pub mode: String,
}

pub const PERMISSIONS_MODE_ASK: &str = "ask";
pub const PERMISSIONS_MODE_FULL_ACCESS: &str = "full_access";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionsConfig {
    #[serde(default = "default_permissions_mode")]
    pub mode: String,
}

impl Default for PermissionsConfig {
    fn default() -> Self {
        Self {
            mode: PERMISSIONS_MODE_ASK.to_owned(),
        }
    }
}

impl PermissionsConfig {
    pub fn mode_label(&self) -> &str {
        let mode = self.mode.trim();
        if mode.eq_ignore_ascii_case(PERMISSIONS_MODE_FULL_ACCESS) {
            PERMISSIONS_MODE_FULL_ACCESS
        } else {
            PERMISSIONS_MODE_ASK
        }
    }

    pub fn full_access_enabled(&self) -> bool {
        self.mode_label() == PERMISSIONS_MODE_FULL_ACCESS
    }
}

fn default_permissions_mode() -> String {
    PERMISSIONS_MODE_ASK.to_owned()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SetupConfig {
    #[serde(default)]
    pub completed: bool,
    #[serde(default)]
    pub therock_venv: Option<PathBuf>,
    #[serde(default)]
    pub cli_install_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ManagedToolConfig {
    #[serde(default)]
    pub path: Option<PathBuf>,
    #[serde(default)]
    pub managed: bool,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            mode: TELEMETRY_MODE_LOCAL.to_owned(),
        }
    }
}

impl TelemetryConfig {
    pub fn mode_label(&self) -> &str {
        let mode = self.mode.trim();
        if mode.is_empty() {
            TELEMETRY_MODE_LOCAL
        } else {
            mode
        }
    }

    pub fn local_inspection_enabled(&self) -> bool {
        self.mode_label().eq_ignore_ascii_case(TELEMETRY_MODE_LOCAL)
    }

    pub fn known_mode(&self) -> bool {
        self.mode_label().eq_ignore_ascii_case(TELEMETRY_MODE_LOCAL)
            || self.mode_label().eq_ignore_ascii_case(TELEMETRY_MODE_OFF)
    }
}

fn default_telemetry_mode() -> String {
    TELEMETRY_MODE_LOCAL.to_owned()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RocmCliConfig {
    #[serde(default)]
    pub default_engine: Option<String>,
    #[serde(default)]
    pub default_runtime_id: Option<String>,
    #[serde(default)]
    pub active_runtime_key: Option<String>,
    #[serde(default)]
    pub previous_runtime_key: Option<String>,
    #[serde(default)]
    pub planner_provider: Option<String>,
    #[serde(default)]
    pub onboarding_dismissed: bool,
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    #[serde(default)]
    pub permissions: PermissionsConfig,
    #[serde(default)]
    pub setup: SetupConfig,
    #[serde(default)]
    pub tools: BTreeMap<String, ManagedToolConfig>,
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderUserConfig>,
    #[serde(default)]
    pub engines: BTreeMap<String, EngineUserConfig>,
    #[serde(default)]
    pub automations: AutomationsConfig,
    /// rocm-dash telemetry/dashboard knobs. Nested as a sub-config
    /// so it never collides with the rocm-cli `telemetry` analytics policy on
    /// rebase. Every field defaults, so the section is fully optional.
    #[serde(default)]
    pub dashboard: DashboardConfig,
}

// ===== rocm-dash dashboard sub-config =====
//
// Additive nesting under the canonical `RocmCliConfig`. The rocm-cli
// `TelemetryConfig { mode }` is an analytics opt-in *policy*; this
// `DashboardConfig` is the operational *spec* (listen address + tick cadence +
// chat endpoint). They are deliberately separate axes and never share a field.
// Pure `with_*()` transforms are scoped to this sub-config only — rocm-cli's own
// config keeps its in-place `&mut` mutation convention untouched.

fn default_dashboard_socket() -> String {
    // Choose a socket location whose *parent* directory is always user-owned so
    // that run_unix can tighten it to 0o700 without EPERM. See
    // `dashboard_socket_path` for the precedence. This resolver is mirrored in
    // `rocm-dash-core` so the canonical `rocm` config and a standalone
    // `rocm-dash` config resolve to the same place; keep the two in sync.
    let path = dashboard_socket_path(
        std::env::var_os("XDG_RUNTIME_DIR"),
        std::env::var_os("HOME"),
        // An empty `USER` must fall through to `LOGNAME`, not short-circuit it.
        std::env::var("USER")
            .ok()
            .filter(|v| !v.is_empty())
            .or_else(|| std::env::var("LOGNAME").ok().filter(|v| !v.is_empty())),
        std::env::temp_dir(),
    );
    format!("unix:{}", path.display())
}

/// Pure core of [`default_dashboard_socket`]: resolve the socket path from
/// explicit env inputs so the precedence is testable without mutating
/// process-global env vars (unsafe and racy under parallel tests in edition
/// 2024). Precedence:
///
/// 1. `$XDG_RUNTIME_DIR` — already mode `0700` on systemd systems, ideal.
/// 2. `$HOME/.rocm/data/telemetry` — standard per-user data dir.
/// 3. `temp_dir()/rocm-<user>` — user-named subdir so the parent is something we
///    create and own, not `/tmp` itself.
fn dashboard_socket_path(
    xdg_runtime_dir: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
    user: Option<String>,
    temp_dir: std::path::PathBuf,
) -> std::path::PathBuf {
    use std::path::PathBuf;
    xdg_runtime_dir
        .filter(|v| !v.is_empty())
        .map(|d| PathBuf::from(d).join("rocmdashd.sock"))
        .or_else(|| {
            home.filter(|v| !v.is_empty()).map(|h| {
                PathBuf::from(h)
                    .join(".rocm")
                    .join("data")
                    .join("telemetry")
                    .join("rocmdashd.sock")
            })
        })
        .unwrap_or_else(|| {
            let raw = user.unwrap_or_else(|| "user".to_owned());
            // Sanitize: keep only alphanumeric, hyphen, and underscore so a path
            // separator or `..` in the env var cannot escape the subdirectory.
            let sanitized: String = raw
                .chars()
                .map(|c| {
                    if c.is_alphanumeric() || c == '-' || c == '_' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect();
            let sanitized = if sanitized.is_empty() {
                "user".to_owned()
            } else {
                sanitized
            };
            temp_dir
                .join(format!("rocm-{sanitized}"))
                .join("rocmdashd.sock")
        })
}

fn default_dashboard_listen() -> String {
    default_dashboard_socket()
}

fn default_dashboard_connect() -> String {
    default_dashboard_socket()
}

fn default_dashboard_theme() -> String {
    "default-dark".to_owned()
}

const fn default_gpu_tick_secs() -> f64 {
    1.0
}

const fn default_discovery_tick_secs() -> f64 {
    5.0
}

const fn default_instance_tick_secs() -> f64 {
    2.0
}

/// Telemetry daemon operational spec. Tick cadences are stored as f64 seconds in
/// the unified JSON config; use the `*_tick()` accessors for `Duration`s.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DashboardDaemonConfig {
    /// `unix:/path/to.sock` or `tcp:host:port`.
    #[serde(default = "default_dashboard_listen")]
    pub listen: String,
    /// Optional shared secret. Required for TCP, ignored for Unix sockets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    #[serde(default = "default_gpu_tick_secs")]
    pub gpu_tick_secs: f64,
    #[serde(default = "default_discovery_tick_secs")]
    pub discovery_tick_secs: f64,
    #[serde(default = "default_instance_tick_secs")]
    pub instance_tick_secs: f64,
    /// Watch this file for new normalized benchmark CSV rows. When unset, the
    /// daemon derives `<data_dir>/bench/results.csv` from the current `AppPaths`
    /// at startup so machine-specific paths are never persisted in config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bench_results_dir: Option<PathBuf>,
}

impl Default for DashboardDaemonConfig {
    fn default() -> Self {
        Self {
            listen: default_dashboard_listen(),
            token: None,
            gpu_tick_secs: default_gpu_tick_secs(),
            discovery_tick_secs: default_discovery_tick_secs(),
            instance_tick_secs: default_instance_tick_secs(),
            bench_results_dir: None,
        }
    }
}

impl DashboardDaemonConfig {
    fn secs_to_duration(s: f64, fallback: Duration) -> Duration {
        // try_from_secs_f64 rejects NaN, negative, inf, and values that
        // overflow Duration (extremely large finite f64).
        Duration::try_from_secs_f64(s).unwrap_or(fallback)
    }

    pub fn gpu_tick(&self) -> Duration {
        Self::secs_to_duration(self.gpu_tick_secs, Duration::from_secs(1))
    }

    pub fn discovery_tick(&self) -> Duration {
        Self::secs_to_duration(self.discovery_tick_secs, Duration::from_secs(5))
    }

    pub fn instance_tick(&self) -> Duration {
        Self::secs_to_duration(self.instance_tick_secs, Duration::from_secs(2))
    }
}

/// Dashboard TUI spec. The chat endpoint URL / model / auth-header *name* are
/// plain data; the auth-header *value* (API key) is always env-only and never
/// stored here (AMD gateway invariant).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DashboardTuiConfig {
    #[serde(default = "default_dashboard_connect")]
    pub connect: String,
    #[serde(default = "default_dashboard_theme")]
    pub theme: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_auth_header: Option<String>,
}

impl Default for DashboardTuiConfig {
    fn default() -> Self {
        Self {
            connect: default_dashboard_connect(),
            theme: default_dashboard_theme(),
            chat_url: None,
            chat_model: None,
            chat_auth_header: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct DashboardConfig {
    #[serde(default)]
    pub daemon: DashboardDaemonConfig,
    #[serde(default)]
    pub tui: DashboardTuiConfig,
}

impl DashboardConfig {
    /// Return a copy with the chat endpoint base URL + model set and the custom
    /// auth header cleared (mirrors the rocm-dash `config_with_chat` behavior).
    /// Immutable transform — scoped to the dashboard sub-config only.
    #[must_use]
    pub fn with_chat_endpoint(
        mut self,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        self.tui.chat_url = Some(base_url.into());
        self.tui.chat_model = Some(model.into());
        self.tui.chat_auth_header = None;
        self
    }

    /// Return a copy with the dashboard theme set.
    #[must_use]
    pub fn with_theme(mut self, theme: impl Into<String>) -> Self {
        self.tui.theme = theme.into();
        self
    }

    /// Return a copy with the telemetry daemon listen address set.
    #[must_use]
    pub fn with_daemon_listen(mut self, listen: impl Into<String>) -> Self {
        self.daemon.listen = listen.into();
        self
    }
}

/// Legacy rocm-dash TOML config shape (`~/.config/rocm-dash/config.toml`),
/// parsed for one-shot migration into the unified JSON config. Every field is
/// optional so partial/legacy files parse cleanly; only the carried-forward
/// fields are mirrored.
#[derive(Debug, Default, Deserialize)]
struct LegacyDashToml {
    #[serde(default)]
    default_engine: Option<String>,
    #[serde(default)]
    daemon: LegacyDashDaemon,
    #[serde(default)]
    tui: LegacyDashTui,
    #[serde(default)]
    engines: BTreeMap<String, EngineUserConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct LegacyDashDaemon {
    #[serde(default)]
    listen: Option<String>,
    #[serde(default)]
    token: Option<String>,
    #[serde(default)]
    gpu_tick: Option<f64>,
    #[serde(default)]
    discovery_tick: Option<f64>,
    #[serde(default)]
    instance_tick: Option<f64>,
    #[serde(default)]
    bench_results_dir: Option<PathBuf>,
}

#[derive(Debug, Default, Deserialize)]
struct LegacyDashTui {
    #[serde(default)]
    connect: Option<String>,
    #[serde(default)]
    theme: Option<String>,
    #[serde(default)]
    chat_url: Option<String>,
    #[serde(default)]
    chat_model: Option<String>,
    #[serde(default)]
    chat_auth_header: Option<String>,
}

impl RocmCliConfig {
    pub fn load(paths: &AppPaths) -> Result<Self> {
        let path = paths.config_path();
        if !path.is_file() {
            return Ok(Self::default());
        }

        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse {}", path.display()))
    }

    pub fn save(&self, paths: &AppPaths) -> Result<()> {
        let path = paths.config_path();
        fs::create_dir_all(&paths.config_dir)
            .with_context(|| format!("failed to create {}", paths.config_dir.display()))?;
        fs::write(
            &path,
            serde_json::to_vec_pretty(self).context("failed to serialize rocm-cli config")?,
        )
        .with_context(|| format!("failed to write {}", path.display()))?;
        Ok(())
    }

    pub fn engine_config(&self, engine: &str) -> Option<&EngineUserConfig> {
        self.engines.get(engine)
    }

    pub fn engine_config_mut(&mut self, engine: &str) -> &mut EngineUserConfig {
        self.engines.entry(engine.to_owned()).or_default()
    }

    pub fn provider_config(&self, provider: &str) -> Option<&ProviderUserConfig> {
        self.providers.get(provider)
    }

    pub fn provider_config_mut(&mut self, provider: &str) -> &mut ProviderUserConfig {
        self.providers.entry(provider.to_owned()).or_default()
    }

    pub fn provider_enabled(&self, provider: &str) -> bool {
        provider.eq_ignore_ascii_case("local")
            || self
                .provider_config(provider)
                .is_some_and(|cfg| cfg.enabled)
    }

    pub fn watcher_config(&self, watcher: &str) -> Option<&WatcherUserConfig> {
        self.automations.watchers.get(watcher)
    }

    pub fn watcher_config_mut(&mut self, watcher: &str) -> &mut WatcherUserConfig {
        self.automations
            .watchers
            .entry(watcher.to_owned())
            .or_default()
    }

    pub fn automation_daemon_enabled(&self) -> bool {
        self.automations.daemon_enabled || self.automations.watchers.values().any(|cfg| cfg.enabled)
    }

    pub fn watcher_enabled(&self, watcher: &BuiltinWatcherSpec) -> bool {
        self.watcher_config(watcher.id)
            .is_some_and(|cfg| cfg.enabled)
    }

    pub fn effective_watcher_mode(&self, watcher: &BuiltinWatcherSpec) -> WatcherMode {
        self.watcher_config(watcher.id)
            .and_then(|cfg| cfg.mode)
            .unwrap_or(watcher.default_mode)
    }

    /// Location of the legacy rocm-dash TOML config, honoring `XDG_CONFIG_HOME`
    /// (`~/.config/rocm-dash/config.toml` on Linux).
    fn legacy_dashboard_toml_path() -> Option<PathBuf> {
        directories::BaseDirs::new()
            .map(|dirs| dirs.config_dir().join("rocm-dash").join("config.toml"))
    }

    /// One-shot migration of a legacy rocm-dash `config.toml` into the unified
    /// JSON config. If no `config.json` exists yet **and** a legacy TOML is
    /// present, its knobs are mapped into `dashboard` (and the canonical
    /// `default_engine`/`engines`), `config.json` is written once, and the
    /// migrated legacy path is returned so the caller can print a notice. The
    /// TOML is left untouched. Returns `Ok(None)` when there is nothing to do
    /// (already on the unified config, or no legacy file) — never clobbers an
    /// existing `config.json`.
    pub fn migrate_legacy_dashboard_toml(paths: &AppPaths) -> Result<Option<PathBuf>> {
        let Some(legacy) = Self::legacy_dashboard_toml_path() else {
            return Ok(None);
        };
        Self::migrate_legacy_dashboard_toml_from(paths, &legacy)
    }

    /// Testable core of [`migrate_legacy_dashboard_toml`] with an explicit legacy
    /// path. Same one-shot, non-clobbering semantics.
    pub fn migrate_legacy_dashboard_toml_from(
        paths: &AppPaths,
        legacy: &Path,
    ) -> Result<Option<PathBuf>> {
        if paths.config_path().is_file() || !legacy.is_file() {
            return Ok(None);
        }

        let raw = fs::read_to_string(legacy)
            .with_context(|| format!("failed to read {}", legacy.display()))?;
        let parsed: LegacyDashToml = toml::from_str(&raw)
            .with_context(|| format!("failed to parse legacy config {}", legacy.display()))?;

        let mut config = Self::default();

        // Dashboard-specific knobs map into the new sub-config.
        let d = &parsed.daemon;
        if let Some(v) = &d.listen {
            config.dashboard.daemon.listen = v.clone();
        }
        config.dashboard.daemon.token = d.token.clone();
        if let Some(v) = d.gpu_tick {
            config.dashboard.daemon.gpu_tick_secs = v;
        }
        if let Some(v) = d.discovery_tick {
            config.dashboard.daemon.discovery_tick_secs = v;
        }
        if let Some(v) = d.instance_tick {
            config.dashboard.daemon.instance_tick_secs = v;
        }
        config.dashboard.daemon.bench_results_dir = d.bench_results_dir.clone();

        let t = &parsed.tui;
        if let Some(v) = &t.connect {
            config.dashboard.tui.connect = v.clone();
        }
        if let Some(v) = &t.theme {
            config.dashboard.tui.theme = v.clone();
        }
        config.dashboard.tui.chat_url = t.chat_url.clone();
        config.dashboard.tui.chat_model = t.chat_model.clone();
        config.dashboard.tui.chat_auth_header = t.chat_auth_header.clone();

        // `default_engine` / `engines` map onto the canonical rocm-cli fields
        // (identical shape) — not a second source of truth inside `dashboard`.
        config.default_engine = parsed.default_engine.clone();
        config.engines = parsed.engines.clone();

        config.save(paths)?;
        Ok(Some(legacy.to_path_buf()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatcherRuntimeSnapshot {
    pub id: String,
    pub enabled: bool,
    pub mode: WatcherMode,
    pub summary: String,
    #[serde(default)]
    pub last_event: Option<String>,
    #[serde(default)]
    pub last_event_unix_ms: Option<u128>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationRuntimeState {
    pub running: bool,
    pub automations_enabled: bool,
    pub daemon_pid: u32,
    pub started_at_unix_ms: u128,
    pub last_tick_unix_ms: u128,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_webhook_endpoint: Option<String>,
    pub active_watchers: Vec<WatcherRuntimeSnapshot>,
}

impl AutomationRuntimeState {
    pub fn load(paths: &AppPaths) -> Result<Option<Self>> {
        let path = paths.automation_state_path();
        if !path.is_file() {
            return Ok(None);
        }

        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        let state = serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        Ok(Some(state))
    }

    pub fn write(&self, paths: &AppPaths) -> Result<()> {
        paths.ensure()?;
        let path = paths.automation_state_path();
        fs::write(
            &path,
            serde_json::to_vec_pretty(self)
                .context("failed to serialize automation runtime state")?,
        )
        .with_context(|| format!("failed to write {}", path.display()))?;
        Ok(())
    }

    pub fn watcher_mut(&mut self, watcher_id: &str) -> Option<&mut WatcherRuntimeSnapshot> {
        self.active_watchers
            .iter_mut()
            .find(|watcher| watcher.id == watcher_id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationEventRecord {
    pub at_unix_ms: u128,
    pub watcher_id: String,
    pub level: String,
    pub action: String,
    pub message: String,
    #[serde(default)]
    pub service_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AutomationTriggerEvent {
    pub at_unix_ms: u128,
    pub kind: String,
    pub source: String,
    #[serde(default)]
    pub watcher_hint: Option<String>,
    #[serde(default)]
    pub service_id: Option<String>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomationProposalRecord {
    pub at_unix_ms: u128,
    #[serde(default)]
    pub proposal_id: String,
    pub watcher_id: String,
    pub action: String,
    pub title: String,
    pub message: String,
    pub status: String,
    #[serde(default)]
    pub service_id: Option<String>,
    #[serde(default)]
    pub tool: Option<String>,
    #[serde(default)]
    pub arguments: serde_json::Value,
    #[serde(default)]
    pub reviewed_at_unix_ms: Option<u128>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEventRecord {
    pub at_unix_ms: u128,
    pub source: String,
    pub category: String,
    pub actor: String,
    pub level: String,
    pub action: String,
    pub message: String,
    #[serde(default)]
    pub watcher_id: Option<String>,
    #[serde(default)]
    pub service_id: Option<String>,
}

pub fn append_automation_event(paths: &AppPaths, event: &AutomationEventRecord) -> Result<()> {
    paths.ensure()?;
    let path = paths.automation_events_path();
    let mut line =
        serde_json::to_string(event).context("failed to serialize automation event record")?;
    line.push('\n');
    let mut existing = if path.is_file() {
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?
    } else {
        String::new()
    };
    existing.push_str(&line);
    fs::write(&path, existing).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

pub fn append_automation_proposal(
    paths: &AppPaths,
    proposal: &AutomationProposalRecord,
) -> Result<()> {
    paths.ensure()?;
    let path = paths.automation_proposals_path();
    let mut proposal = proposal.clone();
    if proposal.proposal_id.is_empty() {
        proposal.proposal_id = generate_proposal_id(&proposal.watcher_id);
    }
    let mut line = serde_json::to_string(&proposal)
        .context("failed to serialize automation proposal record")?;
    line.push('\n');
    let mut existing = if path.is_file() {
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?
    } else {
        String::new()
    };
    existing.push_str(&line);
    fs::write(&path, existing).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

pub fn append_audit_event(paths: &AppPaths, event: &AuditEventRecord) -> Result<()> {
    paths.ensure()?;
    let path = paths.audit_events_path();
    let mut line =
        serde_json::to_string(event).context("failed to serialize audit event record")?;
    line.push('\n');
    let mut existing = if path.is_file() {
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?
    } else {
        String::new()
    };
    existing.push_str(&line);
    fs::write(&path, existing).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

pub fn load_automation_proposals(paths: &AppPaths) -> Result<Vec<AutomationProposalRecord>> {
    let path = paths.automation_proposals_path();
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let text =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut proposals = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let mut proposal = serde_json::from_str::<AutomationProposalRecord>(line)
            .with_context(|| format!("failed to parse proposal record in {}", path.display()))?;
        normalize_proposal_identity(&mut proposal, index);
        proposals.push(proposal);
    }
    Ok(proposals)
}

pub fn load_recent_automation_proposals(
    paths: &AppPaths,
    limit: usize,
) -> Result<Vec<AutomationProposalRecord>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let mut proposals = load_automation_proposals(paths)?;
    proposals.reverse();
    proposals.truncate(limit);
    Ok(proposals)
}

pub fn find_automation_proposal(
    paths: &AppPaths,
    proposal_id: &str,
) -> Result<AutomationProposalRecord> {
    load_automation_proposals(paths)?
        .into_iter()
        .find(|proposal| proposal.proposal_id == proposal_id)
        .with_context(|| format!("automation proposal `{proposal_id}` not found"))
}

pub fn replace_automation_proposal(
    paths: &AppPaths,
    updated: &AutomationProposalRecord,
) -> Result<AutomationProposalRecord> {
    require_nonempty(&updated.proposal_id, "proposal_id")?;
    let mut proposals = load_automation_proposals(paths)?;
    let Some(existing) = proposals
        .iter_mut()
        .find(|proposal| proposal.proposal_id == updated.proposal_id)
    else {
        bail!("automation proposal `{}` not found", updated.proposal_id);
    };
    *existing = updated.clone();
    write_automation_proposals(paths, &proposals)?;
    Ok(updated.clone())
}

pub fn update_automation_proposal_status(
    paths: &AppPaths,
    proposal_id: &str,
    status: &str,
) -> Result<AutomationProposalRecord> {
    require_nonempty(proposal_id, "proposal_id")?;
    require_nonempty(status, "status")?;
    let mut proposals = load_automation_proposals(paths)?;
    let Some(proposal) = proposals
        .iter_mut()
        .find(|proposal| proposal.proposal_id == proposal_id)
    else {
        bail!("automation proposal `{proposal_id}` not found");
    };
    status.clone_into(&mut proposal.status);
    if status != "pending" {
        proposal.reviewed_at_unix_ms = Some(unix_time_millis());
    }
    let updated = proposal.clone();
    write_automation_proposals(paths, &proposals)?;
    Ok(updated)
}

pub fn load_recent_audit_events(paths: &AppPaths, limit: usize) -> Result<Vec<AuditEventRecord>> {
    let path = paths.audit_events_path();
    if !path.is_file() || limit == 0 {
        return Ok(Vec::new());
    }

    let text =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut events = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let event = serde_json::from_str::<AuditEventRecord>(line)
            .with_context(|| format!("failed to parse audit event in {}", path.display()))?;
        events.push(event);
    }
    if events.len() > limit {
        events.drain(0..events.len() - limit);
    }
    Ok(events)
}

fn write_automation_proposals(
    paths: &AppPaths,
    proposals: &[AutomationProposalRecord],
) -> Result<()> {
    paths.ensure()?;
    let path = paths.automation_proposals_path();
    let mut text = String::new();
    for proposal in proposals {
        text.push_str(
            &serde_json::to_string(proposal)
                .context("failed to serialize automation proposal record")?,
        );
        text.push('\n');
    }
    fs::write(&path, text).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

fn normalize_proposal_identity(proposal: &mut AutomationProposalRecord, index: usize) {
    if proposal.proposal_id.is_empty() {
        proposal.proposal_id = format!("legacy-{}-{index}", proposal.at_unix_ms);
    }
}

pub fn generate_proposal_id(prefix: &str) -> String {
    let normalized_prefix = prefix
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_owned();
    let prefix = if normalized_prefix.is_empty() {
        "proposal"
    } else {
        normalized_prefix.as_str()
    };
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!("{prefix}-{nanos}")
}

pub fn load_recent_automation_events(
    paths: &AppPaths,
    limit: usize,
) -> Result<Vec<AutomationEventRecord>> {
    let path = paths.automation_events_path();
    if !path.is_file() {
        return Ok(Vec::new());
    }

    let bytes = fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let text =
        String::from_utf8(bytes).with_context(|| format!("failed to decode {}", path.display()))?;
    let mut events = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let event = serde_json::from_str::<AutomationEventRecord>(line)
            .with_context(|| format!("failed to parse event in {}", path.display()))?;
        events.push(event);
    }
    if events.len() > limit {
        events.drain(0..events.len() - limit);
    }
    Ok(events)
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelRecipeArtifactSourcePolicyRecord {
    pub policy: String,
    #[serde(default)]
    pub required_hosts: Vec<String>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelRecipeArtifactRecord {
    pub artifact_id: String,
    pub kind: String,
    pub uri: String,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub size_bytes: Option<u64>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub gated: Option<bool>,
    #[serde(default)]
    pub quantization: Option<String>,
    #[serde(default)]
    pub engines: Vec<String>,
    #[serde(default)]
    pub source_policy: Option<ModelRecipeArtifactSourcePolicyRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelRecipeEndpointRecord {
    pub endpoint_mode: String,
    #[serde(default)]
    pub settings: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelRecipeUnsupportedCombinationRecord {
    pub combination: String,
    pub reason: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelRecipeEngineRecord {
    pub engine: String,
    #[serde(default)]
    pub required_flags: Vec<String>,
    #[serde(default)]
    pub parser_settings: BTreeMap<String, String>,
    #[serde(default)]
    pub preferred_endpoint: Option<ModelRecipeEndpointRecord>,
    #[serde(default)]
    pub unsupported_combinations: Vec<ModelRecipeUnsupportedCombinationRecord>,
    #[serde(default)]
    pub notes: Vec<String>,
    /// Overrides the recipe `canonical_model_id` when this engine serves the model.
    /// Lets a single alias resolve to engine-specific artifacts (for example a GGUF
    /// id for Lemonade versus a Hugging Face repo id for vLLM).
    #[serde(default)]
    pub model_id_override: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelArtifactCacheStatus {
    pub artifact_id: String,
    pub status: String,
    pub marker_path: PathBuf,
    pub reason: String,
}

pub fn model_artifact_cache_marker_path(
    paths: &AppPaths,
    model_ref: &str,
    artifact_id: &str,
) -> PathBuf {
    let model_component = cache_marker_component("model", model_ref);
    let artifact_component = cache_marker_component("artifact", artifact_id);
    paths
        .data_dir
        .join("models")
        .join("artifacts")
        .join(&model_component)
        .join(format!("{artifact_component}.json"))
}

fn cache_marker_component(kind: &str, value: &str) -> String {
    let slug = sanitize_component(value)
        .trim_matches('-')
        .chars()
        .take(32)
        .collect::<String>();
    let slug = if slug.is_empty() {
        kind.to_owned()
    } else {
        slug
    };
    format!("{slug}--x{}", hex_encode_lower(value.as_bytes()))
}

fn hex_encode_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

pub fn model_artifact_cache_status(
    paths: &AppPaths,
    model_ref: &str,
    artifact: &ModelRecipeArtifactRecord,
) -> ModelArtifactCacheStatus {
    let marker_path = model_artifact_cache_marker_path(paths, model_ref, &artifact.artifact_id);
    if marker_path.is_file() {
        ModelArtifactCacheStatus {
            artifact_id: artifact.artifact_id.clone(),
            status: "metadata_present".to_owned(),
            marker_path,
            reason: "rocm-cli artifact cache marker exists; artifact bytes are still engine/source specific".to_owned(),
        }
    } else {
        ModelArtifactCacheStatus {
            artifact_id: artifact.artifact_id.clone(),
            status: "missing".to_owned(),
            marker_path,
            reason:
                "no rocm-cli artifact cache marker; prefetch requires an approved source policy"
                    .to_owned(),
        }
    }
}

pub fn resolve_model_recipe_artifact(
    artifact_ref: &str,
) -> Result<Option<(ModelRecipeRecord, ModelRecipeArtifactRecord)>> {
    require_nonempty(artifact_ref, "artifact_ref")?;
    let registry = load_model_recipe_registry()?;
    let artifact_ref = artifact_ref.trim();
    if let Some((model_ref, artifact_id)) = artifact_ref.split_once('#') {
        require_nonempty(model_ref, "artifact model_ref")?;
        require_nonempty(artifact_id, "artifact_id")?;
        let Some(recipe) = registry
            .recipes
            .into_iter()
            .find(|recipe| recipe.matches_ref(model_ref))
        else {
            return Ok(None);
        };
        return Ok(recipe
            .artifacts
            .iter()
            .position(|artifact| artifact.artifact_id == artifact_id)
            .map(|index| {
                let artifact = recipe.artifacts[index].clone();
                (recipe, artifact)
            }));
    }

    let mut matches = registry
        .recipes
        .into_iter()
        .filter_map(|recipe| {
            recipe
                .artifacts
                .iter()
                .position(|artifact| artifact.artifact_id == artifact_ref)
                .map(|index| {
                    let artifact = recipe.artifacts[index].clone();
                    (recipe, artifact)
                })
        })
        .collect::<Vec<_>>();
    match matches.len() {
        0 => Ok(None),
        1 => Ok(matches.pop()),
        _ => bail!("artifact_ref `{artifact_ref}` is ambiguous; use `<model-ref>#{artifact_ref}`"),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelRecipeRecord {
    pub canonical_model_id: String,
    pub aliases: Vec<String>,
    pub task: String,
    pub source: String,
    pub revision: String,
    pub loader: String,
    pub trust_remote_code: bool,
    pub dtype: String,
    pub device_policy: String,
    #[serde(default)]
    pub min_gpu_mem_gb: Option<u32>,
    #[serde(default)]
    pub recommended_system_ram_gb: Option<u32>,
    #[serde(default)]
    pub quantization: Option<String>,
    #[serde(default)]
    pub artifact_hint: Option<String>,
    #[serde(default)]
    pub artifacts: Vec<ModelRecipeArtifactRecord>,
    #[serde(default)]
    pub engine_recipes: Vec<ModelRecipeEngineRecord>,
    #[serde(default)]
    pub manual_alternatives: Vec<String>,
    #[serde(default)]
    pub featured: bool,
    pub chat_template_mode: String,
    pub preferred_engines: Vec<String>,
    pub warnings: Vec<String>,
}

impl ModelRecipeRecord {
    pub fn matches_ref(&self, model_ref: &str) -> bool {
        let normalized = normalize_model_ref(model_ref);
        normalize_model_ref(&self.canonical_model_id) == normalized
            || self
                .aliases
                .iter()
                .any(|alias| normalize_model_ref(alias) == normalized)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelCatalogPlatform {
    pub label: String,
    pub engines: Vec<String>,
    #[serde(default)]
    pub gfx_families: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelRecipeIndexDocument {
    pub schema_version: u32,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub generated_at_unix_ms: Option<u128>,
    #[serde(default)]
    pub platforms: Vec<ModelCatalogPlatform>,

    recipes: Vec<ModelRecipeRecord>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ModelRecipeRegistry {
    pub recipes: Vec<ModelRecipeRecord>,
    pub platforms: Vec<ModelCatalogPlatform>,
    pub source: ModelRecipeRegistrySource,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ModelRecipeRegistrySource {
    BuiltIn,
    SignedIndex {
        index_path: PathBuf,
        signature_path: PathBuf,
        public_key_path: PathBuf,
    },
}

impl ModelRecipeIndexDocument {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            bail!(
                "model recipe index schema_version {} is unsupported; expected 1",
                self.schema_version
            );
        }
        if self.recipes.is_empty() {
            bail!("model recipe index must contain at least one recipe");
        }

        let mut refs = BTreeMap::<String, String>::new();
        for recipe in &self.recipes {
            require_nonempty(&recipe.canonical_model_id, "canonical_model_id")?;
            require_nonempty(&recipe.task, "task")?;
            require_nonempty(&recipe.source, "source")?;
            require_nonempty(&recipe.revision, "revision")?;
            require_nonempty(&recipe.loader, "loader")?;
            require_nonempty(&recipe.dtype, "dtype")?;
            require_nonempty(&recipe.device_policy, "device_policy")?;
            require_nonempty(&recipe.chat_template_mode, "chat_template_mode")?;
            validate_model_device_policy(&recipe.device_policy)?;
            insert_unique_model_ref(
                &mut refs,
                &recipe.canonical_model_id,
                &recipe.canonical_model_id,
            )?;
            for alias in &recipe.aliases {
                require_nonempty(alias, "alias")?;
                insert_unique_model_ref(&mut refs, alias, &recipe.canonical_model_id)?;
            }
            for artifact in &recipe.artifacts {
                validate_model_recipe_artifact(artifact, &recipe.canonical_model_id)?;
            }
            let mut engines = BTreeMap::<String, String>::new();
            for engine_recipe in &recipe.engine_recipes {
                validate_model_recipe_engine_record(engine_recipe, &recipe.canonical_model_id)?;
                let normalized = normalize_model_ref(&engine_recipe.engine);
                if let Some(existing) = engines.insert(normalized, engine_recipe.engine.clone()) {
                    bail!(
                        "engine recipe for `{}` on `{}` is duplicated by `{existing}`",
                        engine_recipe.engine,
                        recipe.canonical_model_id
                    );
                }
            }
        }

        Ok(())
    }
}

pub fn builtin_model_recipe_registry() -> ModelRecipeRegistry {
    let doc = builtin_model_catalog_document();
    ModelRecipeRegistry {
        recipes: doc.recipes.clone(),
        platforms: doc.platforms.clone(),
        source: ModelRecipeRegistrySource::BuiltIn,
    }
}

pub fn load_model_recipe_registry() -> Result<ModelRecipeRegistry> {
    let configured_index = env_path_override("ROCM_CLI_MODEL_RECIPE_INDEX_PATH");
    if configured_index.is_none() && env_flag("ROCM_CLI_REQUIRE_MODEL_RECIPE_SIGNATURE") {
        bail!(
            "signed model recipe index is required but ROCM_CLI_MODEL_RECIPE_INDEX_PATH is not configured"
        );
    }
    let Some(index_path) = configured_index else {
        return Ok(builtin_model_recipe_registry());
    };

    let signature_path = env_path_override("ROCM_CLI_MODEL_RECIPE_INDEX_SIGNATURE_PATH")
        .unwrap_or_else(|| model_recipe_index_signature_path(&index_path));
    let public_key_path = env_path_override("ROCM_CLI_MODEL_RECIPE_INDEX_PUBLIC_KEY_PATH")
        .context(
            "signed model recipe index requires ROCM_CLI_MODEL_RECIPE_INDEX_PUBLIC_KEY_PATH",
        )?;
    let document = load_signed_model_recipe_index(&index_path, &signature_path, &public_key_path)?;
    let platforms = if document.platforms.is_empty() {
        builtin_model_catalog_document().platforms.clone()
    } else {
        document.platforms
    };
    Ok(ModelRecipeRegistry {
        recipes: document.recipes,
        platforms,
        source: ModelRecipeRegistrySource::SignedIndex {
            index_path,
            signature_path,
            public_key_path,
        },
    })
}

pub fn resolve_model_recipe(model_ref: &str) -> Result<Option<ModelRecipeRecord>> {
    Ok(load_model_recipe_registry()?
        .recipes
        .into_iter()
        .find(|recipe| recipe.matches_ref(model_ref)))
}

pub fn load_signed_model_recipe_index(
    index_path: &Path,
    signature_path: &Path,
    public_key_path: &Path,
) -> Result<ModelRecipeIndexDocument> {
    verify_model_recipe_index_signature(index_path, signature_path, public_key_path)?;
    let document = load_model_recipe_index(index_path)?;
    document.validate()?;
    Ok(document)
}

pub fn load_model_recipe_index(index_path: &Path) -> Result<ModelRecipeIndexDocument> {
    let bytes = fs::read(index_path)
        .with_context(|| format!("failed to read model recipe index {}", index_path.display()))?;
    let document =
        serde_json::from_slice::<ModelRecipeIndexDocument>(&bytes).with_context(|| {
            format!(
                "failed to parse model recipe index {}",
                index_path.display()
            )
        })?;
    document.validate()?;
    Ok(document)
}

pub fn model_recipe_index_signature_path(index_path: &Path) -> PathBuf {
    let mut signature = index_path.as_os_str().to_os_string();
    signature.push(".sig");
    PathBuf::from(signature)
}

/// Normalize a PEM document the way the OpenSSL CLI tolerated input, so keys
/// produced or copied through other tooling still parse with the strict RFC 7468
/// reader. Strips a leading UTF-8 BOM, accepts any line-ending style (CRLF, lone
/// CR, or LF), and drops trailing whitespace from each line — Windows tooling
/// (e.g. PowerShell `Set-Content`) can introduce CRLF or a stray trailing space
/// on the `-----BEGIN ...-----` boundary that the parser would otherwise reject.
fn normalize_pem(pem: &str) -> String {
    let without_bom = pem.strip_prefix('\u{feff}').unwrap_or(pem);
    let unified = without_bom.replace("\r\n", "\n").replace('\r', "\n");
    let mut normalized: String = unified
        .split('\n')
        .map(|line| line.trim_end_matches([' ', '\t']))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    normalized.push('\n');
    normalized
}

/// Verify an RSASSA-PKCS#1 v1.5 signature over SHA-256 using a pure-Rust
/// implementation, with no external `openssl` process.
///
/// `public_key_pem` is a SubjectPublicKeyInfo PEM (`-----BEGIN PUBLIC KEY-----`),
/// exactly what `openssl rsa -pubout` emits and what `openssl dgst -sha256 -verify`
/// consumes, so verification is byte-compatible with that command. `label` names the
/// artifact being checked (e.g. `"metadata"`); on a bad signature the error reads
/// `"<label> signature verification failed"` to preserve existing diagnostics.
pub fn verify_rsa_pkcs1_sha256_signature(
    public_key_pem: &str,
    payload: &[u8],
    signature: &[u8],
    label: &str,
) -> Result<()> {
    use rsa::RsaPublicKey;
    use rsa::pkcs1v15::{Signature, VerifyingKey};
    use rsa::pkcs8::DecodePublicKey;
    use rsa::signature::Verifier;
    use sha2::Sha256;

    let public_key = RsaPublicKey::from_public_key_pem(&normalize_pem(public_key_pem))
        .with_context(|| format!("{label} public key is not a valid RSA public key"))?;
    let signature = Signature::try_from(signature)
        .with_context(|| format!("{label} signature is malformed"))?;
    VerifyingKey::<Sha256>::new(public_key)
        .verify(payload, &signature)
        .map_err(|error| anyhow::anyhow!("{label} signature verification failed: {error}"))
}

/// Produce an RSASSA-PKCS#1 v1.5 signature over SHA-256 with a pure-Rust
/// implementation, with no external `openssl` process.
///
/// `private_key_pem` is a PKCS#8 private-key PEM (`-----BEGIN PRIVATE KEY-----`),
/// as emitted by `openssl genpkey`. The signature is deterministic and
/// byte-identical to `openssl dgst -sha256 -sign`, so artifacts signed here verify
/// with either implementation.
pub fn sign_rsa_pkcs1_sha256_signature(private_key_pem: &str, payload: &[u8]) -> Result<Vec<u8>> {
    use rsa::RsaPrivateKey;
    use rsa::pkcs1v15::SigningKey;
    use rsa::pkcs8::DecodePrivateKey;
    use rsa::signature::{SignatureEncoding, Signer};
    use sha2::Sha256;

    let private_key = RsaPrivateKey::from_pkcs8_pem(&normalize_pem(private_key_pem))
        .context("signing private key is not a valid PKCS#8 RSA private key")?;
    let signature = SigningKey::<Sha256>::new(private_key)
        .try_sign(payload)
        .context("failed to produce RSA signature")?;
    Ok(signature.to_bytes().into_vec())
}

/// Generate a fresh 2048-bit RSA signing keypair, returned as
/// `(PKCS#8 private-key PEM, SubjectPublicKeyInfo public-key PEM)` — the same
/// formats `openssl genpkey` / `openssl rsa -pubout` produce.
pub fn generate_rsa_signing_keypair() -> Result<(String, String)> {
    use rsa::RsaPrivateKey;
    use rsa::pkcs8::{EncodePrivateKey, EncodePublicKey, LineEnding};

    let mut rng = rand::thread_rng();
    let private_key =
        RsaPrivateKey::new(&mut rng, 2048).context("failed to generate RSA signing key")?;
    let private_pem = private_key
        .to_pkcs8_pem(LineEnding::LF)
        .context("failed to encode private key")?
        .to_string();
    let public_pem = rsa::RsaPublicKey::from(&private_key)
        .to_public_key_pem(LineEnding::LF)
        .context("failed to encode public key")?;
    Ok((private_pem, public_pem))
}

pub fn verify_model_recipe_index_signature(
    index_path: &Path,
    signature_path: &Path,
    public_key_path: &Path,
) -> Result<()> {
    if !signature_path.is_file() {
        bail!(
            "model recipe index signature is missing: {}",
            signature_path.display()
        );
    }
    if !public_key_path.is_file() {
        bail!(
            "model recipe index public key is missing: {}",
            public_key_path.display()
        );
    }
    let public_key_pem = fs::read_to_string(public_key_path).with_context(|| {
        format!(
            "failed to read model recipe index public key: {}",
            public_key_path.display()
        )
    })?;
    let signature = fs::read(signature_path).with_context(|| {
        format!(
            "failed to read model recipe index signature: {}",
            signature_path.display()
        )
    })?;
    let payload = fs::read(index_path).with_context(|| {
        format!(
            "failed to read model recipe index: {}",
            index_path.display()
        )
    })?;
    verify_rsa_pkcs1_sha256_signature(&public_key_pem, &payload, &signature, "model recipe index")
}

fn validate_model_device_policy(policy: &str) -> Result<()> {
    match policy {
        "gpu_required" | "gpu_preferred" | "cpu_only" => Ok(()),
        other => bail!(
            "model recipe device_policy `{other}` is unsupported; expected gpu_required, gpu_preferred, or cpu_only"
        ),
    }
}

fn insert_unique_model_ref(
    refs: &mut BTreeMap<String, String>,
    model_ref: &str,
    canonical_model_id: &str,
) -> Result<()> {
    let normalized = normalize_model_ref(model_ref);
    if let Some(existing) = refs.insert(normalized, canonical_model_id.to_owned()) {
        bail!(
            "model recipe ref `{model_ref}` is duplicated by `{existing}` and `{canonical_model_id}`"
        );
    }
    Ok(())
}

fn validate_model_recipe_artifact(
    artifact: &ModelRecipeArtifactRecord,
    canonical_model_id: &str,
) -> Result<()> {
    require_nonempty(&artifact.artifact_id, "artifact_id")?;
    require_nonempty(&artifact.kind, "artifact kind")?;
    require_nonempty(&artifact.uri, "artifact uri")?;
    if let Some(sha256) = artifact.sha256.as_deref()
        && (sha256.len() != 64 || !sha256.chars().all(|ch| ch.is_ascii_hexdigit()))
    {
        bail!(
            "artifact `{}` for `{canonical_model_id}` has invalid sha256",
            artifact.artifact_id
        );
    }
    if let Some(source_policy) = &artifact.source_policy {
        validate_model_recipe_artifact_source_policy(source_policy, artifact, canonical_model_id)?;
    }
    Ok(())
}

fn validate_model_recipe_artifact_source_policy(
    source_policy: &ModelRecipeArtifactSourcePolicyRecord,
    artifact: &ModelRecipeArtifactRecord,
    canonical_model_id: &str,
) -> Result<()> {
    require_nonempty(&source_policy.policy, "artifact source_policy policy")?;
    for host in &source_policy.required_hosts {
        require_nonempty(host, "artifact source_policy required_host")?;
        if host.contains('/') || host.contains('@') || host.contains(':') {
            bail!(
                "artifact `{}` for `{canonical_model_id}` has invalid source_policy required_host `{host}`",
                artifact.artifact_id
            );
        }
    }
    for note in &source_policy.notes {
        require_nonempty(note, "artifact source_policy note")?;
    }

    if !source_policy.required_hosts.is_empty() {
        let Some(host) = recipe_artifact_url_host(&artifact.uri) else {
            bail!(
                "artifact `{}` for `{canonical_model_id}` declares required source hosts but its uri is not HTTP(S)",
                artifact.artifact_id
            );
        };
        if !source_policy
            .required_hosts
            .iter()
            .any(|required| required.eq_ignore_ascii_case(&host))
        {
            bail!(
                "artifact `{}` for `{canonical_model_id}` uri host `{host}` is not allowed by source_policy",
                artifact.artifact_id
            );
        }
    }

    match source_policy.policy.as_str() {
        "direct_https_sha256" => {
            if !artifact.uri.starts_with("https://") {
                bail!(
                    "artifact `{}` for `{canonical_model_id}` source_policy direct_https_sha256 requires an HTTPS uri",
                    artifact.artifact_id
                );
            }
            validate_prefetch_integrity_metadata(artifact, canonical_model_id)?;
        }
        "huggingface_public" => {
            if artifact.gated.unwrap_or(false) {
                bail!(
                    "artifact `{}` for `{canonical_model_id}` source_policy huggingface_public cannot be used for a gated artifact",
                    artifact.artifact_id
                );
            }
            validate_huggingface_source_policy_uri(source_policy, artifact, canonical_model_id)?;
            validate_prefetch_integrity_metadata(artifact, canonical_model_id)?;
        }
        "huggingface_authenticated" => {
            validate_huggingface_source_policy_uri(source_policy, artifact, canonical_model_id)?;
            validate_prefetch_integrity_metadata(artifact, canonical_model_id)?;
        }
        "manual_only" => {}
        other => bail!(
            "artifact `{}` for `{canonical_model_id}` has unsupported source_policy `{other}`",
            artifact.artifact_id
        ),
    }
    Ok(())
}

fn validate_prefetch_integrity_metadata(
    artifact: &ModelRecipeArtifactRecord,
    canonical_model_id: &str,
) -> Result<()> {
    if artifact.sha256.is_none() {
        bail!(
            "artifact `{}` for `{canonical_model_id}` source_policy requires sha256 metadata",
            artifact.artifact_id
        );
    }
    if artifact.size_bytes.is_none() {
        bail!(
            "artifact `{}` for `{canonical_model_id}` source_policy requires size_bytes metadata",
            artifact.artifact_id
        );
    }
    Ok(())
}

fn validate_huggingface_source_policy_uri(
    source_policy: &ModelRecipeArtifactSourcePolicyRecord,
    artifact: &ModelRecipeArtifactRecord,
    canonical_model_id: &str,
) -> Result<()> {
    if !artifact.uri.starts_with("https://") {
        bail!(
            "artifact `{}` for `{canonical_model_id}` source_policy {} requires an HTTPS Hugging Face uri",
            artifact.artifact_id,
            source_policy.policy
        );
    }
    if !recipe_artifact_uri_is_huggingface(&artifact.uri) {
        bail!(
            "artifact `{}` for `{canonical_model_id}` source_policy {} requires a Hugging Face uri",
            artifact.artifact_id,
            source_policy.policy
        );
    }
    Ok(())
}

fn recipe_artifact_uri_is_huggingface(uri: &str) -> bool {
    recipe_artifact_url_host(uri).is_some_and(|host| {
        host == "huggingface.co"
            || host.ends_with(".huggingface.co")
            || host == "hf.co"
            || host.ends_with(".hf.co")
    })
}

fn recipe_artifact_url_host(uri: &str) -> Option<String> {
    let rest = uri
        .strip_prefix("https://")
        .or_else(|| uri.strip_prefix("http://"))?;
    let authority = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default()
        .trim();
    if authority.is_empty() || authority.contains('@') {
        return None;
    }
    let host = authority
        .strip_prefix('[')
        .and_then(|value| value.split_once(']').map(|(host, _)| host))
        .unwrap_or_else(|| authority.split(':').next().unwrap_or_default())
        .trim()
        .trim_end_matches('.')
        .to_ascii_lowercase();
    (!host.is_empty()).then_some(host)
}

fn validate_model_recipe_engine_record(
    engine_recipe: &ModelRecipeEngineRecord,
    canonical_model_id: &str,
) -> Result<()> {
    require_nonempty(&engine_recipe.engine, "engine recipe engine")?;
    for flag in &engine_recipe.required_flags {
        require_nonempty(flag, "engine required flag")?;
    }
    for (key, value) in &engine_recipe.parser_settings {
        require_nonempty(key, "engine parser setting key")?;
        require_nonempty(value, "engine parser setting value")?;
    }
    if let Some(endpoint) = engine_recipe.preferred_endpoint.as_ref() {
        require_nonempty(&endpoint.endpoint_mode, "engine preferred endpoint mode")?;
        for (key, value) in &endpoint.settings {
            require_nonempty(key, "engine endpoint setting key")?;
            require_nonempty(value, "engine endpoint setting value")?;
        }
    }
    for item in &engine_recipe.unsupported_combinations {
        require_nonempty(&item.combination, "engine unsupported combination")?;
        require_nonempty(&item.reason, "engine unsupported combination reason")?;
    }
    for note in &engine_recipe.notes {
        require_nonempty(note, "engine recipe note")?;
    }
    if let Some(model_id_override) = engine_recipe.model_id_override.as_deref() {
        require_nonempty(model_id_override, "engine model id override")?;
    }
    if engine_recipe.required_flags.is_empty()
        && engine_recipe.parser_settings.is_empty()
        && engine_recipe.preferred_endpoint.is_none()
        && engine_recipe.unsupported_combinations.is_empty()
        && engine_recipe.notes.is_empty()
        && engine_recipe.model_id_override.is_none()
    {
        bail!(
            "engine recipe for `{}` on `{canonical_model_id}` must not be empty",
            engine_recipe.engine
        );
    }
    Ok(())
}

pub fn builtin_model_recipes() -> Vec<ModelRecipeRecord> {
    builtin_model_catalog_document().recipes.clone()
}

/// The curated fallback catalog shipped inside the binary. It is authored as JSON
/// (`model_catalog.json`) using the same schema as an external signed recipe
/// index, so the offline default and hosted indexes share one format. Parsed once
/// and cached; a malformed catalog is a test-time bug guarded by a unit test.
fn builtin_model_catalog_document() -> &'static ModelRecipeIndexDocument {
    static CATALOG: std::sync::OnceLock<ModelRecipeIndexDocument> = std::sync::OnceLock::new();
    CATALOG.get_or_init(|| {
        let document =
            serde_json::from_str::<ModelRecipeIndexDocument>(include_str!("model_catalog.json"))
                .expect("built-in model catalog JSON must parse");
        document
            .validate()
            .expect("built-in model catalog must satisfy the recipe index schema");
        document
    })
}

pub fn resolve_builtin_model_recipe(model_ref: &str) -> Option<ModelRecipeRecord> {
    builtin_model_recipes()
        .into_iter()
        .find(|recipe| recipe.matches_ref(model_ref))
}

/// Returns the ordered platform definitions from the registry.
pub fn model_catalog_platforms(registry: &ModelRecipeRegistry) -> Vec<ModelCatalogPlatform> {
    registry.platforms.clone()
}

/// The label of the hardware platform a recipe targets, derived from its first
/// preferred engine matched against the catalog's platform definitions.
pub fn model_recipe_target_platform_label(
    recipe: &ModelRecipeRecord,
    platforms: &[ModelCatalogPlatform],
) -> String {
    let engine = recipe
        .preferred_engines
        .first()
        .map(|e| e.trim().to_ascii_lowercase())
        .unwrap_or_default();
    platforms
        .iter()
        .find(|p| p.engines.iter().any(|e| e.eq_ignore_ascii_case(&engine)))
        .map_or_else(|| engine.clone(), |p| p.label.clone())
}

/// Whether the given normalized TheRock family matches a platform's gfx targets.
pub fn platform_matches_gfx_family(platform: &ModelCatalogPlatform, gfx_family: &str) -> bool {
    platform
        .gfx_families
        .iter()
        .any(|f| f.eq_ignore_ascii_case(gfx_family))
}

/// Whether a recipe appears in the curated `rocm model` short list.
///
/// Driven by the `featured` field in the catalog JSON. Hidden recipes stay fully
/// resolvable for `rocm serve` and the crate's smoke tests; only the user-facing
/// `rocm model` list omits them.
pub const fn model_recipe_featured(recipe: &ModelRecipeRecord) -> bool {
    recipe.featured
}

pub fn normalize_model_ref(model_ref: &str) -> String {
    model_ref.trim().to_ascii_lowercase()
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct ManagedServiceRecord {
    pub service_id: String,
    pub engine: String,
    pub model_ref: String,
    pub canonical_model_id: String,
    pub host: String,
    pub port: u16,
    pub endpoint_url: String,
    pub mode: String,
    pub status: String,
    pub supervisor_pid: u32,
    pub engine_pid: Option<u32>,
    #[serde(default)]
    pub runtime_id: Option<String>,
    #[serde(default)]
    pub env_id: Option<String>,
    #[serde(default)]
    pub device_policy: Option<String>,
    #[serde(default)]
    pub gpu_indices: Vec<u32>,
    #[serde(default)]
    pub engine_recipe_json: Option<String>,
    #[serde(default)]
    pub restart_count: u32,
    #[serde(default)]
    pub last_restart_unix_ms: Option<u128>,
    /// Coarse startup stage (`downloading`/`loading`/`warmup`) parsed from the
    /// serve process's own log output while it is coming up. Set to `None` once
    /// the service reaches `ready`, and absent on older on-disk records.
    #[serde(default)]
    pub startup_phase: Option<String>,
    pub manifest_path: PathBuf,
    pub log_path: PathBuf,
    pub engine_state_path: PathBuf,
    pub created_at_unix_ms: u128,
}

impl ManagedServiceRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        paths: &AppPaths,
        service_id: impl Into<String>,
        engine: impl Into<String>,
        model_ref: impl Into<String>,
        canonical_model_id: impl Into<String>,
        host: impl Into<String>,
        port: u16,
        mode: impl Into<String>,
        supervisor_pid: u32,
        runtime_id: Option<String>,
        env_id: Option<String>,
        device_policy: Option<String>,
    ) -> Self {
        let service_id = service_id.into();
        let engine = engine.into();
        let host = host.into();
        let manifest_path = paths.service_manifest_path(&service_id);
        let log_path = paths.service_log_path(&service_id);
        let engine_state_path = paths.service_engine_state_path(&engine, &service_id);
        Self {
            endpoint_url: format!("{}/v1", format_http_base_url(&host, port)),
            service_id,
            engine,
            model_ref: model_ref.into(),
            canonical_model_id: canonical_model_id.into(),
            host,
            port,
            mode: mode.into(),
            status: "starting".to_owned(),
            supervisor_pid,
            engine_pid: None,
            runtime_id,
            env_id,
            device_policy,
            gpu_indices: Vec::new(),
            engine_recipe_json: None,
            restart_count: 0,
            last_restart_unix_ms: None,
            startup_phase: None,
            manifest_path,
            log_path,
            engine_state_path,
            created_at_unix_ms: unix_time_millis(),
        }
    }

    pub fn normalize_paths_for_host(&mut self) {
        self.manifest_path = normalize_runtime_path_for_host(&self.manifest_path);
        self.log_path = normalize_runtime_path_for_host(&self.log_path);
        self.engine_state_path = normalize_runtime_path_for_host(&self.engine_state_path);
    }

    pub fn refresh_from_engine_state(&mut self) -> Result<bool> {
        if !matches!(
            self.status.as_str(),
            "starting" | "running" | "recovering" | "ready"
        ) {
            return Ok(false);
        }
        self.normalize_paths_for_host();
        if !self.engine_state_path.is_file() {
            return Ok(false);
        }
        let bytes = fs::read(&self.engine_state_path)
            .with_context(|| format!("failed to read {}", self.engine_state_path.display()))?;
        let state = serde_json::from_slice::<serde_json::Value>(&bytes)
            .with_context(|| format!("failed to parse {}", self.engine_state_path.display()))?;
        let Some(status) = state
            .get("status")
            .and_then(serde_json::Value::as_str)
            .filter(|value| matches!(*value, "ready" | "running" | "starting" | "failed"))
        else {
            return Ok(false);
        };

        let previous = self.status.clone();
        status.clone_into(&mut self.status);
        if let Some(endpoint_url) = state
            .get("endpoint_url")
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.trim().is_empty())
        {
            endpoint_url.clone_into(&mut self.endpoint_url);
        }
        if let Some(runtime_id) = state
            .get("runtime_id")
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.trim().is_empty())
        {
            self.runtime_id = Some(runtime_id.to_owned());
        }
        if let Some(env_id) = state
            .get("env_id")
            .and_then(serde_json::Value::as_str)
            .filter(|value| !value.trim().is_empty())
        {
            self.env_id = Some(env_id.to_owned());
        }
        if let Some(pid) = state
            .get("server_pid")
            .or_else(|| state.get("pid"))
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
        {
            self.engine_pid = Some(pid);
        }
        Ok(self.status != previous)
    }

    fn with_storage_paths(&self) -> Self {
        let mut record = self.clone();
        record.manifest_path = normalize_runtime_path_for_storage(&record.manifest_path);
        record.log_path = normalize_runtime_path_for_storage(&record.log_path);
        record.engine_state_path = normalize_runtime_path_for_storage(&record.engine_state_path);
        record
    }

    pub fn write(&self) -> Result<()> {
        let mut host_record = self.clone();
        host_record.normalize_paths_for_host();
        let parent = host_record
            .manifest_path
            .parent()
            .context("service manifest path must have a parent directory")?;
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
        let storage_record = host_record.with_storage_paths();
        fs::write(
            &host_record.manifest_path,
            serde_json::to_vec_pretty(&storage_record)
                .context("failed to serialize service record")?,
        )
        .with_context(|| format!("failed to write {}", host_record.manifest_path.display()))?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexBridgeSnapshot {
    pub protocol: String,
    pub generated_at_unix_ms: u128,
    pub examine: ExamineSummary,
    pub gpu: CodexBridgeGpuSnapshot,
    pub config: RocmCliConfig,
    #[serde(default)]
    pub automation_runtime: Option<AutomationRuntimeState>,
    #[serde(default)]
    pub recent_automation_events: Vec<AutomationEventRecord>,
    #[serde(default)]
    pub engines: Vec<CodexBridgeEngine>,
    #[serde(default)]
    pub services: Vec<ManagedServiceRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexBridgeGpuSnapshot {
    pub amd_smi_available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub static_snapshot: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub monitor_snapshot: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexBridgeEngine {
    pub id: String,
    pub summary: String,
    pub default_for_platform: bool,
    pub installed_binary: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binary_path: Option<String>,
}

pub fn sibling_binary_path(binary_name: &str) -> Result<PathBuf> {
    require_nonempty(binary_name, "binary_name")?;
    let current_exe = current_executable_path()?;
    let candidates = sibling_binary_candidates(&current_exe, binary_name)?;
    for candidate in &candidates {
        if candidate.is_file() {
            return Ok(candidate.clone());
        }
    }
    let candidate_text = candidates
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    bail!(
        "unable to locate sibling binary {}; checked {} next to {}",
        platform_binary_name(binary_name),
        candidate_text,
        current_exe.display()
    )
}

pub fn sibling_binary_exists(binary_name: &str) -> bool {
    let Ok(current_exe) = current_executable_path() else {
        return false;
    };
    let Ok(candidates) = sibling_binary_candidates(&current_exe, binary_name) else {
        return false;
    };
    candidates.iter().any(|candidate| candidate.is_file())
}

fn sibling_binary_candidates(current_exe: &Path, binary_name: &str) -> Result<Vec<PathBuf>> {
    let Some(binary_dir) = current_exe.parent() else {
        bail!("current executable has no parent directory");
    };
    let binary = platform_binary_name(binary_name);
    let mut candidates = Vec::new();
    let mut push_candidate = |path: PathBuf| {
        if !candidates.iter().any(|candidate| candidate == &path) {
            candidates.push(path);
        }
    };
    push_candidate(binary_dir.join(&binary));
    if binary_dir.file_name().and_then(|name| name.to_str()) == Some("deps")
        && let Some(parent) = binary_dir.parent()
    {
        push_candidate(parent.join(&binary));
        if let Some(target_dir) = parent.parent() {
            for profile in ["release", "debug"] {
                push_candidate(target_dir.join(profile).join(&binary));
            }
        }
    }
    Ok(candidates)
}

pub fn engine_binary_path(engine: &str) -> Result<PathBuf> {
    sibling_binary_path(&format!("rocm-engine-{engine}"))
}

pub fn daemon_binary_path() -> Result<PathBuf> {
    let current_exe = current_executable_path()?;
    if current_exe
        .parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        == Some("deps")
        && let Ok(rocm) = sibling_binary_path("rocm")
    {
        return Ok(rocm);
    }
    Ok(current_exe)
}

pub fn resolve_amd_smi_binary() -> OsString {
    if let Some(path) = default_data_dir()
        .map(|data_dir| data_dir.join("runtimes").join("registry"))
        .and_then(|registry_dir| resolve_amd_smi_binary_in_registry(&registry_dir))
    {
        return path;
    }
    resolve_amd_smi_binary_in_home(runtime_home_dir().as_deref())
}

/// Locate `amd-smi` inside the bin directories of the newest managed ROCm SDK
/// runtime recorded in the registry. The binary ships with the TheRock wheel
/// (under the SDK `bin_path` and/or the venv `install_root/bin`) and is not on
/// `PATH`, so the home-directory fallbacks below never find it.
fn resolve_amd_smi_binary_in_registry(registry_dir: &Path) -> Option<OsString> {
    let mut records = managed_therock_environment_records(registry_dir);
    records.sort_by_key(|(_, record)| std::cmp::Reverse(record.installed_at_unix_ms.unwrap_or(0)));
    records.into_iter().find_map(|(_, record)| {
        amd_smi_bin_dirs_for_record(&record)
            .iter()
            .find_map(|bin_dir| managed_sdk_tool_path(bin_dir, "amd-smi"))
            .map(PathBuf::into_os_string)
    })
}

fn amd_smi_bin_dirs_for_record(record: &TheRockFamilyManifest) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(sdk) = record.rocm_sdk.as_ref() {
        if let Some(bin_path) = sdk.bin_path.as_ref() {
            dirs.push(bin_path.clone());
        }
        for bin_path in &sdk.bin_paths {
            if !dirs.contains(bin_path) {
                dirs.push(bin_path.clone());
            }
        }
    }
    if let Some(install_root) = record.install_root.as_ref() {
        let install_bin = install_root.join("bin");
        if !dirs.contains(&install_bin) {
            dirs.push(install_bin);
        }
    }
    dirs
}

fn resolve_amd_smi_binary_in_home(home_dir: Option<&Path>) -> OsString {
    if let Some(home_dir) = home_dir {
        let venv_bin = home_dir.join("rocm_venvs").join("default").join("bin");
        if let Some(path) = managed_sdk_tool_path(&venv_bin, "amd-smi") {
            return path.into_os_string();
        }

        let legacy_bin = home_dir.join(".rocm").join("bin");
        if let Some(path) = managed_sdk_tool_path(&legacy_bin, "amd-smi") {
            return path.into_os_string();
        }
    }

    "amd-smi".into()
}

pub fn generate_service_id(engine: &str, model_ref: &str) -> String {
    let model_slug = sanitize_component(model_ref)
        .trim_matches('-')
        .chars()
        .take(24)
        .collect::<String>();
    format!(
        "{}-{}-{}",
        sanitize_component(engine),
        model_slug,
        unix_time_millis()
    )
}

pub fn sanitize_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => ch.to_ascii_lowercase(),
            _ => '-',
        })
        .collect()
}

pub fn unix_time_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use std::path::Path;
    use std::path::PathBuf;

    // The socket-path precedence is mirrored in `rocm-dash-core`; these tests
    // mirror the ones there so a divergence in either crate is caught.

    #[test]
    fn dashboard_socket_path_prefers_xdg_runtime_dir() {
        // Tier 1: $XDG_RUNTIME_DIR is already mode 0700 on systemd systems, so it
        // must win over $HOME and the temp dir.
        let path = dashboard_socket_path(
            Some("/run/user/1000".into()),
            Some("/home/alice".into()),
            Some("alice".to_owned()),
            PathBuf::from("/tmp"),
        );
        assert_eq!(path, PathBuf::from("/run/user/1000/rocmdashd.sock"));
    }

    #[test]
    fn dashboard_socket_path_falls_back_to_home_then_temp() {
        // Tier 2: no XDG → per-user data dir under $HOME.
        let path = dashboard_socket_path(
            None,
            Some("/home/alice".into()),
            Some("alice".to_owned()),
            PathBuf::from("/tmp"),
        );
        assert_eq!(
            path,
            PathBuf::from("/home/alice/.rocm/data/telemetry/rocmdashd.sock")
        );

        // Tier 3: no XDG and no HOME → user-named subdir of the temp dir, never
        // the bare temp dir itself.
        let path =
            dashboard_socket_path(None, None, Some("alice".to_owned()), PathBuf::from("/tmp"));
        assert_eq!(path, PathBuf::from("/tmp/rocm-alice/rocmdashd.sock"));
    }

    #[test]
    fn dashboard_socket_path_sanitizes_user_and_skips_empty_env() {
        // An empty XDG/HOME value is treated as unset (falls through), and a user
        // name with path separators cannot escape the intended subdirectory.
        let path = dashboard_socket_path(
            Some("".into()),
            Some("".into()),
            Some("../../etc".to_owned()),
            PathBuf::from("/tmp"),
        );
        assert_eq!(path, PathBuf::from("/tmp/rocm-______etc/rocmdashd.sock"));

        // No user name at all still yields a valid per-user subdir.
        let path = dashboard_socket_path(None, None, None, PathBuf::from("/tmp"));
        assert_eq!(path, PathBuf::from("/tmp/rocm-user/rocmdashd.sock"));

        // A bare empty user name (as opposed to unset) also falls back to "user".
        let path = dashboard_socket_path(None, None, Some(String::new()), PathBuf::from("/tmp"));
        assert_eq!(path, PathBuf::from("/tmp/rocm-user/rocmdashd.sock"));
    }

    #[test]
    fn openai_models_endpoint_has_model_checks_loaded_model_ids() -> Result<()> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let server = std::thread::spawn(move || -> Result<String> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
            let mut request_bytes = Vec::new();
            let mut buffer = [0_u8; 512];
            loop {
                let read = stream.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                request_bytes.extend_from_slice(&buffer[..read]);
                if String::from_utf8_lossy(&request_bytes).contains("\r\n\r\n") {
                    break;
                }
            }
            let request = String::from_utf8_lossy(&request_bytes).into_owned();
            let body = r#"{"data":[{"id":"Qwen3-0.6B-GGUF"}]}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )?;
            Ok(request)
        });
        let endpoint = format!("http://127.0.0.1:{port}/v1");

        assert!(openai_models_endpoint_has_model(
            &endpoint,
            Some("qwen"),
            Duration::from_secs(2)
        )?);

        let request = server.join().expect("server thread should not panic")?;
        assert!(request.starts_with("GET /v1/models HTTP/1.1"));
        Ok(())
    }

    // EAI-7333: the service healthcheck marks a model "ready" purely from
    // `/v1/models` listing it (via `openai_models_endpoint_has_model`), without
    // verifying inference. This test pins that gap: a server that lists the model
    // on `/v1/models` but is NOT able to serve `/v1/chat/completions` still
    // reports the model as present. The readiness signal is therefore a false
    // positive for inference-readiness — a caller must additionally probe
    // inference before treating a service as usable. When EAI-7333 is fixed
    // (readiness gated on an inference probe, not just `/v1/models`), the
    // healthcheck path should no longer rely on this signal alone.
    #[test]
    fn models_endpoint_readiness_does_not_imply_inference_ready() -> Result<()> {
        // A server that answers `/v1/models` with the model listed, but would
        // hang/refuse an actual chat request (it only ever serves this one
        // response, then closes) — mirroring an engine whose model is still
        // loading while `/v1/models` already responds.
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let server = std::thread::spawn(move || -> Result<()> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
            let mut buffer = [0_u8; 512];
            let _ = stream.read(&mut buffer)?;
            let body = r#"{"data":[{"id":"Qwen/Qwen2.5-1.5B-Instruct"}]}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )?;
            Ok(())
        });
        let endpoint = format!("http://127.0.0.1:{port}/v1");

        // The `/v1/models` probe reports the model present — this is the exact
        // signal the healthcheck uses to declare "ready".
        let models_ready = openai_models_endpoint_has_model(
            &endpoint,
            Some("Qwen/Qwen2.5-1.5B-Instruct"),
            Duration::from_secs(2),
        )?;
        assert!(
            models_ready,
            "/v1/models lists the model, so the current healthcheck would report ready"
        );

        // But that says nothing about inference: the server served only the
        // models response and closed, so a chat request would not succeed.
        // Readiness based on this signal alone is a false positive (EAI-7333).
        server.join().expect("server thread should not panic")?;
        Ok(())
    }

    #[test]
    fn resolve_amd_smi_binary_prefers_home_rocm_venv_path() -> Result<()> {
        let temp_root =
            std::env::temp_dir().join(format!("rocm-cli-amd-smi-{}", unix_time_millis()));
        let bin_dir = temp_root.join("rocm_venvs").join("default").join("bin");
        fs::create_dir_all(&bin_dir)?;
        let amd_smi_path = bin_dir.join("amd-smi");
        fs::write(&amd_smi_path, b"#!/bin/sh\nexit 0\n")?;

        let resolved = resolve_amd_smi_binary_in_home(Some(Path::new(&temp_root)));

        let _ = fs::remove_file(&amd_smi_path);
        let _ = fs::remove_dir_all(&temp_root);

        assert_eq!(resolved, amd_smi_path.into_os_string());
        Ok(())
    }

    #[test]
    fn resolve_amd_smi_binary_in_registry_uses_newest_runtime_sdk_bin() -> Result<()> {
        let temp_root =
            std::env::temp_dir().join(format!("rocm-cli-amd-smi-registry-{}", unix_time_millis()));
        let registry_dir = temp_root.join("runtimes/registry");
        fs::create_dir_all(&registry_dir)?;

        // Older runtime: amd-smi only under the venv install_root/bin.
        let old_root = temp_root.join("release-wheel-gfx94x-dcgpu-7-13-0");
        let old_bin = old_root.join("bin");
        fs::create_dir_all(&old_bin)?;
        fs::write(old_bin.join("amd-smi"), b"#!/bin/sh\nexit 0\n")?;
        fs::write(
            registry_dir.join("old.json"),
            serde_json::to_vec(&serde_json::json!({
                "runtime_id": "therock-stable:gfx94X-dcgpu",
                "install_root": old_root,
                "installed_at_unix_ms": 1_000_u128,
                "rocm_sdk": { "import_ok": true },
            }))?,
        )?;

        // Newer runtime: amd-smi under the SDK devel bin_path.
        let new_bin =
            temp_root.join("release-wheel-gfx94x-dcgpu-7-14-0a20260611/_rocm_sdk_devel/bin");
        fs::create_dir_all(&new_bin)?;
        let new_amd_smi = new_bin.join("amd-smi");
        fs::write(&new_amd_smi, b"#!/bin/sh\nexit 0\n")?;
        fs::write(
            registry_dir.join("new.json"),
            serde_json::to_vec(&serde_json::json!({
                "runtime_id": "therock-stable:gfx94X-dcgpu",
                "installed_at_unix_ms": 2_000_u128,
                "rocm_sdk": { "import_ok": true, "bin_path": new_bin },
            }))?,
        )?;

        let resolved = resolve_amd_smi_binary_in_registry(&registry_dir);

        let _ = fs::remove_dir_all(&temp_root);

        assert_eq!(resolved, Some(new_amd_smi.into_os_string()));
        Ok(())
    }

    #[test]
    fn default_engine_is_always_usable_on_windows() {
        if cfg!(windows) {
            assert_eq!(default_engine_for_platform(), "lemonade");
        }
    }

    #[test]
    fn instinct_dcgpu_family_prefers_vllm() {
        // On Instinct data-center GPUs (TheRock `*-dcgpu` families, e.g. the
        // MI300X's gfx94X-dcgpu) the default serving engine is vLLM. This is the
        // GPU-family preference the serve engine selection honors before falling
        // back to a recipe's own preferred engine. vLLM is Linux-only, so the
        // preference does not apply on native Windows.
        let summary = HostGpuSummary {
            name: Some("AMD Instinct MI300X".to_owned()),
            gfx_target: Some("gfx942".to_owned()),
            therock_family: Some("gfx94X-dcgpu".to_owned()),
        };
        let preferred = preferred_serve_engine_for_host_gpu_summary(&summary);
        if cfg!(windows) {
            assert_eq!(preferred, None, "vLLM is not preferred on native Windows");
        } else {
            assert_eq!(preferred, Some("vllm"));
        }
    }

    #[test]
    fn consumer_gpu_family_has_no_vllm_preference() {
        // A non-dcgpu consumer family (e.g. gfx110X-all) has no GPU-level vLLM
        // preference, so serve selection falls through to the recipe/platform
        // default rather than forcing vLLM.
        let summary = HostGpuSummary {
            name: Some("AMD Radeon".to_owned()),
            gfx_target: Some("gfx1100".to_owned()),
            therock_family: Some("gfx110X-all".to_owned()),
        };
        assert_eq!(preferred_serve_engine_for_host_gpu_summary(&summary), None);
    }

    #[test]
    fn normalize_therock_family_maps_gfx1101_to_gfx110x_all() {
        assert_eq!(
            normalize_therock_family("gfx1101"),
            Some("gfx110X-all".to_owned())
        );
    }

    #[test]
    fn normalize_therock_family_maps_gfx1103_to_gfx110x_all() {
        assert_eq!(
            normalize_therock_family("gfx1103"),
            Some("gfx110X-all".to_owned())
        );
    }

    #[test]
    fn normalize_therock_family_maps_gfx1201_to_gfx120x_all() {
        assert_eq!(
            normalize_therock_family("gfx1201"),
            Some("gfx120X-all".to_owned())
        );
    }

    #[test]
    fn normalize_therock_family_accepts_canonical_family_labels() {
        assert_eq!(
            normalize_therock_family("gfx120X-all"),
            Some("gfx120X-all".to_owned())
        );
        assert_eq!(
            normalize_therock_family("gfx110X-all"),
            Some("gfx110X-all".to_owned())
        );
        assert_eq!(
            normalize_therock_family("gfx94X-dcgpu"),
            Some("gfx94X-dcgpu".to_owned())
        );
    }

    #[test]
    fn known_therock_families_all_round_trip() {
        for family in known_therock_families() {
            assert_eq!(
                normalize_therock_family(family).as_deref(),
                Some(*family),
                "known family `{family}` must normalize back to itself"
            );
        }
    }

    #[test]
    fn known_therock_families_is_not_empty() {
        assert!(!known_therock_families().is_empty());
    }

    #[test]
    fn preferred_serve_engine_uses_vllm_for_supported_therock_families() {
        assert_eq!(
            preferred_serve_engine_for_therock_family(Some("gfx90a")),
            Some("vllm")
        );
        assert_eq!(
            preferred_serve_engine_for_therock_family(Some("gfx950")),
            Some("vllm")
        );
        assert_eq!(
            preferred_serve_engine_for_therock_family(Some("gfx999-dcgpu")),
            Some("vllm")
        );
        assert_eq!(preferred_serve_engine_for_therock_family(None), None);
    }

    #[test]
    fn preferred_serve_engine_host_summary_respects_platform_and_fields() {
        // `gfx_target` is consulted as a fallback when `therock_family` is absent.
        let summary = HostGpuSummary {
            gfx_target: Some("gfx950".to_owned()),
            ..HostGpuSummary::default()
        };
        // The vLLM adapter is unsupported on native Windows, so the preference is
        // gated off there while remaining active on Linux/WSL builds.
        let expected = if cfg!(windows) { None } else { Some("vllm") };
        assert_eq!(
            preferred_serve_engine_for_host_gpu_summary(&summary),
            expected
        );

        // No GPU information never resolves to a vLLM preference on any platform.
        assert_eq!(
            preferred_serve_engine_for_host_gpu_summary(&HostGpuSummary::default()),
            None
        );
    }

    #[test]
    fn windows_display_parser_maps_rx_9070_xt_device_id_to_gfx1201() {
        let text = "ASPEED Graphics Family(WDDM)\tPCI\\VEN_1A03&DEV_2000\nAMD Radeon RX 9070 XT\tPCI\\VEN_1002&DEV_7550&SUBSYS_2435148C&REV_C0";
        assert_eq!(
            parse_windows_display_gfx_target(text),
            Some("gfx1201".to_owned())
        );
    }

    #[test]
    fn windows_display_parser_maps_known_amd_pci_ids() {
        for (device_id, expected) in [
            ("73A0", "gfx1030"),
            ("73C0", "gfx1031"),
            ("73E0", "gfx1032"),
            ("163F", "gfx1033"),
            ("743F", "gfx1034"),
            ("1681", "gfx1035"),
            ("164E", "gfx1036"),
            ("15BF", "gfx1103"),
            ("164F", "gfx1103"),
            ("1900", "gfx1103"),
            ("1114", "gfx1152"),
        ] {
            assert_eq!(
                parse_windows_display_gfx_target(&format!(
                    "AMD Display Adapter\tPCI\\VEN_1002&DEV_{device_id}"
                )),
                Some(expected.to_owned()),
                "{device_id}"
            );
        }
    }

    #[test]
    fn windows_display_parser_falls_back_to_name_when_pci_id_is_uncertain() {
        assert_eq!(
            parse_windows_display_gfx_target("AMD Radeon 820M\tPCI\\VEN_1002&DEV_1902"),
            Some("gfx1153".to_owned())
        );
    }

    #[test]
    fn windows_display_name_parser_uses_first_nonempty_adapter_name() {
        assert_eq!(
            parse_windows_display_name("\nAMD Radeon RX 9070 XT\tPCI\\VEN_1002&DEV_7550\n"),
            Some("AMD Radeon RX 9070 XT".to_owned())
        );
    }

    #[test]
    fn windows_display_name_cleaner_removes_inf_resource_prefix() {
        assert_eq!(
            clean_windows_display_name("@oem40.inf,%amd7550.23%;AMD Radeon RX 9070 XT"),
            "AMD Radeon RX 9070 XT"
        );
        assert_eq!(
            clean_windows_display_name("AMD Radeon RX 9070 XT"),
            "AMD Radeon RX 9070 XT"
        );
    }

    #[test]
    fn windows_display_parser_maps_known_marketing_names() {
        for (name, expected) in [
            ("AMD Radeon RX 9070 XT\t", "gfx1201"),
            ("AMD Radeon RX 9060 XT\t", "gfx1200"),
            ("AMD Radeon RX 7900 XTX\t", "gfx1100"),
            ("AMD Radeon RX 7800 XT\t", "gfx1101"),
            ("AMD Radeon RX 7600\t", "gfx1102"),
            ("AMD Radeon RX 6800 XT\t", "gfx1030"),
            ("AMD Radeon RX 6800M\t", "gfx1031"),
            ("AMD Radeon RX 6700 XT\t", "gfx1031"),
            ("AMD Radeon RX 6600\t", "gfx1032"),
            ("AMD Radeon RX 6500 XT\t", "gfx1034"),
            ("AMD Radeon 680M\t", "gfx1035"),
            ("AMD Radeon 660M\t", "gfx1035"),
            ("AMD Radeon 610M\t", "gfx1036"),
            ("AMD Radeon 780M\t", "gfx1103"),
            ("AMD Radeon 760M\t", "gfx1103"),
            ("AMD Radeon 740M\t", "gfx1103"),
            ("AMD Radeon 8060S\t", "gfx1151"),
            ("AMD Radeon 890M\t", "gfx1150"),
            ("AMD Radeon 860M\t", "gfx1152"),
            ("AMD Radeon 820M\t", "gfx1153"),
            ("Steam Deck\t", "gfx1033"),
        ] {
            assert_eq!(
                parse_windows_display_gfx_target(name),
                Some(expected.to_owned()),
                "{name}"
            );
        }
    }

    #[test]
    fn amd_pci_device_id_parser_requires_amd_vendor() {
        assert_eq!(
            amd_pci_device_id_from_pnp_id("PCI\\VEN_1002&DEV_7550&SUBSYS_2435148C"),
            Some("7550".to_owned())
        );
        assert_eq!(
            amd_pci_device_id_from_pnp_id("PCI\\VEN_1A03&DEV_2000"),
            None
        );
    }

    #[test]
    fn windows_examine_inventory_parser_feeds_cpu_driver_and_gfx_detection() {
        let inventory = parse_windows_examine_inventory(
            "CPU\t  AMD Ryzen 9 9950X  16-Core Processor  \nRAM\t68719476736\nGPU\tAMD Radeon RX 9070 XT\t32.0.13031.9001\tPCI\\VEN_1002&DEV_7550&SUBSYS_2435148C&REV_C0\n",
        );

        assert_eq!(
            inventory.cpu_model.as_deref(),
            Some("AMD Ryzen 9 9950X 16-Core Processor")
        );
        assert_eq!(inventory.system_ram_gib, Some(64.0));
        assert_eq!(
            inventory.amd_display_driver_detail().as_deref(),
            Some("AMD Radeon RX 9070 XT driver 32.0.13031.9001")
        );
        assert_eq!(inventory.display_gfx_target(), Some("gfx1201".to_owned()));
    }

    #[test]
    fn windows_pnputil_inventory_parser_detects_780m_device_id() {
        let inventory = parse_windows_pnputil_display_inventory(
            "\
Instance ID:                PCI\\VEN_1002&DEV_15BF&SUBSYS_15021025&REV_C1\\4&2F6D7E4A&0&0041
Device Description:        AMD Radeon 780M Graphics
Class Name:                Display
Class GUID:                {4d36e968-e325-11ce-bfc1-08002be10318}
Manufacturer Name:         Advanced Micro Devices, Inc.
Status:                    Started
Driver Name:               oem42.inf
",
        );

        assert_eq!(
            inventory.amd_display_name().as_deref(),
            Some("AMD Radeon 780M Graphics")
        );
        assert_eq!(inventory.display_gfx_target(), Some("gfx1103".to_owned()));
    }

    #[test]
    fn windows_pnputil_inventory_parser_ignores_non_amd_display() {
        let inventory = parse_windows_pnputil_display_inventory(
            "\
Instance ID:                PCI\\VEN_8086&DEV_9A49&SUBSYS_00000000
Device Description:        Intel UHD Graphics
Class Name:                Display
",
        );

        assert!(inventory.displays.is_empty());
    }

    #[test]
    fn windows_examine_inventory_prefers_real_gpu_over_noisy_amd_pnp_entries() {
        let inventory = parse_windows_examine_inventory(
            "GPU\tAMD Bluetooth Capture Audio Device\t\t{2101C4C0-2C15-4035-A0D0-EEC3C2277B11}\\CAPTURE&CP_111215637\nGPU\tAMD-OpenGL User Mode Driver\t\tSWD\\DRIVERENUM\\AMDOGL&5&BAA66E4&0\nGPU\tAMD Radeon 780M Graphics\t\tPCI\\VEN_1002&DEV_1900&SUBSYS_50EE17AA&REV_D0\\4&EB5E2B6&0&0041\n",
        );

        assert_eq!(
            inventory.amd_display_name().as_deref(),
            Some("AMD Radeon 780M Graphics")
        );
        assert_eq!(inventory.display_gfx_target(), Some("gfx1103".to_owned()));
    }

    #[test]
    fn windows_examine_gfx_detection_uses_inventory_without_rocm_tools() {
        if !cfg!(windows) {
            return;
        }
        let inventory = parse_windows_examine_inventory(
            "GPU\tAMD Radeon RX 9070 XT\t32.0.23033.1002\tPCI\\VEN_1002&DEV_7550",
        );

        assert_eq!(
            detect_host_gfx_target_with_context(Some(&inventory), None, None),
            Some("gfx1201".to_owned())
        );
    }

    #[test]
    fn gc_version_converts_to_gfx_target() {
        assert_eq!(
            gfx_target_from_gc_version(11, 0, 1),
            Some("gfx1101".to_owned())
        );
        assert_eq!(
            gfx_target_from_gc_version(11, 0, 3),
            Some("gfx1103".to_owned())
        );
    }

    #[test]
    fn linux_kfd_gfx_target_parser_accepts_numeric_and_direct_tokens() {
        assert_eq!(
            parse_linux_kfd_gfx_target("110003"),
            Some("gfx1103".to_owned())
        );
        assert_eq!(
            parse_linux_kfd_gfx_target("120001"),
            Some("gfx1201".to_owned())
        );
        assert_eq!(
            parse_linux_kfd_gfx_target("gfx1103"),
            Some("gfx1103".to_owned())
        );
        assert_eq!(parse_linux_kfd_gfx_target("not-a-target"), None);
    }

    #[test]
    fn linux_ip_discovery_gc_fixture_maps_to_gfx_target() -> Result<()> {
        let (root, _paths) = temp_app_paths("linux-ip-discovery");
        let gc = root
            .join("ip_discovery")
            .join("die")
            .join("0")
            .join("GC")
            .join("0");
        fs::create_dir_all(&gc)?;
        fs::write(gc.join("major"), "11")?;
        fs::write(gc.join("minor"), "0")?;
        fs::write(gc.join("revision"), "3")?;

        assert_eq!(
            detect_ip_discovery_gc_target(&root.join("ip_discovery")),
            Some("gfx1103".to_owned())
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn linux_amdgpu_device_fixture_accepts_vendor_or_uevent_driver() -> Result<()> {
        let (root, _paths) = temp_app_paths("linux-amdgpu-device");
        let vendor = root.join("vendor");
        fs::create_dir_all(&root)?;
        fs::write(&vendor, "0x1002\n")?;
        assert!(is_amdgpu_device(&root));
        fs::remove_file(&vendor)?;
        fs::write(root.join("uevent"), "DRIVER=amdgpu\n")?;
        assert!(is_amdgpu_device(&root));
        fs::write(root.join("uevent"), "DRIVER=i915\n")?;
        assert!(!is_amdgpu_device(&root));
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn audit_events_path_lives_under_data_audit() {
        let (_root, paths) = temp_app_paths("audit-path");
        assert_eq!(
            paths.audit_events_path(),
            paths.data_dir.join("audit").join("events.jsonl")
        );
        assert_eq!(
            paths.automation_proposals_path(),
            paths.data_dir.join("automations").join("proposals.jsonl")
        );
    }

    #[test]
    fn counts_json_files_and_model_cache_entries_for_examine() -> Result<()> {
        let (root, paths) = temp_app_paths("examine-counts");
        let registry = paths.data_dir.join("runtimes").join("registry");
        let models = paths.data_dir.join("models");
        fs::create_dir_all(&registry)?;
        fs::create_dir_all(&models)?;
        fs::write(registry.join("runtime-a.json"), "{}")?;
        fs::write(registry.join("runtime-b.json"), "{}")?;
        fs::write(registry.join("notes.txt"), "skip")?;
        fs::create_dir_all(models.join("hf"))?;
        fs::write(models.join("local.bin"), "model")?;

        assert_eq!(count_json_files(&registry), 2);
        assert_eq!(count_dir_entries(&models), 2);
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn managed_therock_family_uses_runtime_manifest_not_host_mapping() -> Result<()> {
        let (root, paths) = temp_app_paths("managed-therock-family");
        let registry = paths.data_dir.join("runtimes").join("registry");
        fs::create_dir_all(&registry)?;
        fs::write(
            registry.join("newest.json"),
            r#"{
                "runtime_id": "therock-release:gfx120X-all",
                "family": "gfx1201",
                "installed_at_unix_ms": 20
            }"#,
        )?;
        fs::write(
            registry.join("older.json"),
            r#"{
                "runtime_id": "therock-release:gfx110X-all",
                "family": "gfx1103",
                "installed_at_unix_ms": 10
            }"#,
        )?;
        fs::write(
            registry.join("not-therock.json"),
            r#"{
                "runtime_id": "other-runtime",
                "family": "gfx1030",
                "installed_at_unix_ms": 30
            }"#,
        )?;

        assert_eq!(
            detect_managed_therock_family(&paths),
            Some("gfx120X-all".to_owned())
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn managed_therock_family_falls_back_to_engine_env_manifest() -> Result<()> {
        let (root, paths) = temp_app_paths("engine-therock-family");
        let manifests = paths.engine_manifests_dir("vllm");
        fs::create_dir_all(&manifests)?;
        fs::write(
            manifests.join("env.json"),
            r#"{
                "runtime_id": "therock-release",
                "therock_family": "gfx1151",
                "installed_at_unix_ms": 15
            }"#,
        )?;

        assert_eq!(
            detect_managed_therock_family(&paths),
            Some("gfx1151".to_owned())
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn managed_therock_family_is_none_without_therock_manifest() -> Result<()> {
        let (root, paths) = temp_app_paths("no-therock-family");
        let registry = paths.data_dir.join("runtimes").join("registry");
        fs::create_dir_all(&registry)?;
        fs::write(
            registry.join("other.json"),
            r#"{
                "runtime_id": "other-runtime",
                "family": "gfx1201",
                "installed_at_unix_ms": 99
            }"#,
        )?;

        assert_eq!(detect_managed_therock_family(&paths), None);
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn managed_sdk_probe_detects_gfx_from_therock_tool() -> Result<()> {
        let (root, paths) = temp_app_paths("managed-sdk-gfx");
        let registry = paths.data_dir.join("runtimes").join("registry");
        let site_packages = root.join("site-packages");
        let sdk_root = site_packages.join("_rocm_sdk_devel");
        let sdk_bin = sdk_root.join("bin");
        fs::create_dir_all(&sdk_bin)?;
        write_fake_rocm_agent_enumerator(&sdk_bin, "gfx1201")?;
        fs::create_dir_all(&registry)?;
        fs::write(
            registry.join("runtime.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "runtime_id": "therock-release:gfx120X-all",
                "family": "gfx120X-all",
                "installed_at_unix_ms": 10,
                "rocm_sdk": {
                    "import_ok": true,
                    "site_packages": site_packages,
                    "root_path": sdk_root,
                    "bin_path": sdk_bin
                }
            }))?,
        )?;

        assert_eq!(
            detect_managed_therock_sdk_gfx_target(&paths),
            Some("gfx1201".to_owned())
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn managed_sdk_probe_skips_non_therock_manifests() -> Result<()> {
        let (root, paths) = temp_app_paths("managed-sdk-skip-non-therock");
        let registry = paths.data_dir.join("runtimes").join("registry");
        let site_packages = root.join("site-packages");
        let sdk_root = site_packages.join("_rocm_sdk_devel");
        let sdk_bin = sdk_root.join("bin");
        fs::create_dir_all(&sdk_bin)?;
        write_fake_rocm_agent_enumerator(&sdk_bin, "gfx9999")?;
        fs::create_dir_all(&registry)?;
        fs::write(
            registry.join("runtime.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "runtime_id": "external-runtime",
                "family": "gfx120X-all",
                "installed_at_unix_ms": 10,
                "rocm_sdk": {
                    "import_ok": true,
                    "site_packages": site_packages,
                    "root_path": sdk_root,
                    "bin_path": sdk_bin
                }
            }))?,
        )?;

        assert_eq!(detect_managed_therock_sdk_gfx_target(&paths), None);
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn active_managed_therock_channel_reads_recorded_channel() -> Result<()> {
        let (root, paths) = temp_app_paths("active-therock-channel");
        let registry = paths.data_dir.join("runtimes").join("registry");
        fs::create_dir_all(&registry)?;
        fs::write(
            registry.join("runtime.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "runtime_id": "therock-nightly:gfx120X-all",
                "family": "gfx120X-all",
                "channel": "nightly",
                "installed_at_unix_ms": 10,
                "rocm_sdk": { "import_ok": true }
            }))?,
        )?;

        let config = RocmCliConfig::default();
        assert_eq!(
            active_managed_therock_channel(&paths, &config)?,
            Some("nightly".to_owned())
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn active_managed_therock_channel_is_none_without_runtime() -> Result<()> {
        let (root, paths) = temp_app_paths("active-therock-channel-none");
        let config = RocmCliConfig::default();
        assert_eq!(active_managed_therock_channel(&paths, &config)?, None);
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn active_managed_therock_channel_falls_back_to_most_recent() -> Result<()> {
        let (root, paths) = temp_app_paths("active-therock-channel-recent");
        let registry = paths.data_dir.join("runtimes").join("registry");
        fs::create_dir_all(&registry)?;
        write_therock_channel_record(&registry, "older", "release", 10)?;
        write_therock_channel_record(&registry, "newer", "nightly", 20)?;

        // No active_runtime_key set: the most recently installed runtime wins.
        let config = RocmCliConfig::default();
        assert_eq!(
            active_managed_therock_channel(&paths, &config)?,
            Some("nightly".to_owned())
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn active_managed_therock_channel_prefers_active_runtime_key() -> Result<()> {
        let (root, paths) = temp_app_paths("active-therock-channel-active-key");
        let registry = paths.data_dir.join("runtimes").join("registry");
        fs::create_dir_all(&registry)?;
        write_therock_channel_record(&registry, "older", "release", 10)?;
        write_therock_channel_record(&registry, "newer", "nightly", 20)?;

        // The active key points at the older runtime, overriding recency.
        let config = RocmCliConfig {
            active_runtime_key: Some("therock-release:older".to_owned()),
            ..RocmCliConfig::default()
        };
        assert_eq!(
            active_managed_therock_channel(&paths, &config)?,
            Some("release".to_owned())
        );
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    fn write_therock_channel_record(
        registry: &Path,
        name: &str,
        channel: &str,
        installed_at_unix_ms: u64,
    ) -> Result<()> {
        fs::write(
            registry.join(format!("{name}.json")),
            serde_json::to_vec_pretty(&serde_json::json!({
                "runtime_id": format!("therock-{channel}:{name}"),
                "family": "gfx120X-all",
                "channel": channel,
                "installed_at_unix_ms": installed_at_unix_ms,
                "rocm_sdk": { "import_ok": true }
            }))?,
        )?;
        Ok(())
    }

    #[test]
    fn examine_render_includes_driver_and_state_counts() {
        let summary = ExamineSummary {
            os: "windows".to_owned(),
            arch: "x86_64".to_owned(),
            kernel: Some("10.0.26100".to_owned()),
            distro: Some("Windows".to_owned()),
            cpu: Some("AMD Ryzen".to_owned()),
            system_ram_gib: Some(64.0),
            interactive_terminal: false,
            default_engine: "vllm".to_owned(),
            detected_gfx_target: None,
            compatible_therock_family: Some("gfx120X-all".to_owned()),
            detected_therock_family: None,
            driver: DriverSummary {
                policy: "windows_validate_only".to_owned(),
                status: "amd_display_driver_detected".to_owned(),
                detail: Some("AMD Radeon driver 1.2.3".to_owned()),
            },
            legacy_rocm: LegacyRocmSummary {
                status: "detected_unmanaged".to_owned(),
                paths: vec![PathBuf::from("C:\\Program Files\\AMD\\ROCm")],
                detail: Some("legacy install".to_owned()),
            },
            wsl: None,
            managed_runtime_count: 2,
            managed_service_count: 1,
            model_cache_entries: 3,
            config_dir: PathBuf::from("config"),
            data_dir: PathBuf::from("data"),
            cache_dir: PathBuf::from("cache"),
        };

        let rendered = summary.render_text();
        assert!(rendered.contains("distro: Windows"));
        assert!(rendered.contains("cpu: AMD Ryzen"));
        assert!(rendered.contains("system_ram: 64 GiB"));
        assert!(rendered.contains("compatible_therock_family: gfx120X-all"));
        assert!(rendered.contains("detected_therock_family: <not detected>"));
        assert!(rendered.contains("driver_policy: windows_validate_only"));
        assert!(rendered.contains("driver_status: amd_display_driver_detected"));
        assert!(rendered.contains("legacy_rocm_status: detected_unmanaged"));
        assert!(rendered.contains("legacy_rocm_paths: C:\\Program Files\\AMD\\ROCm"));
        assert!(
            rendered.contains("legacy_rocm_guidance: legacy ROCm detected; keep it side-by-side")
        );
        assert!(rendered.contains("wsl: false"));
        assert!(rendered.contains("managed_runtimes: 2"));
        assert!(rendered.contains("managed_services: 1"));
        assert!(rendered.contains("model_cache_entries: 3"));
    }

    #[test]
    fn examine_render_guides_managed_runtime_install_when_only_legacy_rocm_exists() {
        let summary = ExamineSummary {
            os: "linux".to_owned(),
            arch: "x86_64".to_owned(),
            kernel: None,
            distro: None,
            cpu: None,
            system_ram_gib: None,
            interactive_terminal: false,
            default_engine: "vllm".to_owned(),
            detected_gfx_target: None,
            compatible_therock_family: None,
            detected_therock_family: None,
            driver: DriverSummary {
                policy: "linux_official_amd_dkms_wrapper".to_owned(),
                status: "amdgpu_available".to_owned(),
                detail: None,
            },
            legacy_rocm: LegacyRocmSummary {
                status: "detected_unmanaged".to_owned(),
                paths: vec![PathBuf::from("/opt/rocm")],
                detail: Some("legacy install".to_owned()),
            },
            wsl: None,
            managed_runtime_count: 0,
            managed_service_count: 0,
            model_cache_entries: 0,
            config_dir: PathBuf::from("config"),
            data_dir: PathBuf::from("data"),
            cache_dir: PathBuf::from("cache"),
        };

        let rendered = summary.render_text();

        assert!(rendered.contains(
            "legacy_rocm_guidance: legacy ROCm detected; install a managed TheRock runtime"
        ));
        assert!(rendered.contains("rocm install sdk --channel release --format wheel"));
    }

    #[test]
    fn wsl_driver_summary_reports_missing_rocdxg_without_amdgpu_fallback() {
        let summary = WslSummary {
            is_wsl: true,
            dxg_device: true,
            dxcore: true,
            librocdxg: false,
            rocdxg_dids: false,
            ldconfig_librocdxg: false,
            rocminfo: false,
            cargo: true,
            detail: Some("missing /opt/rocm/lib/librocdxg.so".to_owned()),
        };

        let driver = wsl_driver_summary(&summary);

        assert_eq!(driver.policy, "wsl_rocdxg");
        assert_eq!(driver.status, "wsl_rocdxg_missing");
        assert!(
            driver
                .detail
                .as_deref()
                .is_some_and(|detail| detail.contains("librocdxg"))
        );
    }

    #[test]
    fn linux_legacy_rocm_detection_ignores_rocdxg_only_directory() -> Result<()> {
        let (root, _) = temp_app_paths("rocdxg-only");
        let rocm = root.join("rocm");
        fs::create_dir_all(rocm.join("lib"))?;
        fs::write(rocm.join("lib").join("librocdxg.so"), "")?;

        assert!(!legacy_rocm_candidate_exists(&rocm));

        fs::create_dir_all(rocm.join("bin"))?;
        fs::write(rocm.join("bin").join("rocminfo"), "")?;
        assert!(legacy_rocm_candidate_exists(&rocm));

        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn parses_os_release_pretty_name() {
        assert_eq!(
            parse_os_release_pretty_name("NAME=Ubuntu\nPRETTY_NAME=\"Ubuntu 24.04.2 LTS\"\n"),
            Some("Ubuntu 24.04.2 LTS".to_owned())
        );
    }

    #[test]
    fn append_audit_event_writes_jsonl_record() -> Result<()> {
        let (root, paths) = temp_app_paths("append-audit");
        let event = AuditEventRecord {
            at_unix_ms: 123,
            source: "rocmd".to_owned(),
            category: "automation".to_owned(),
            actor: "watcher:server-recover".to_owned(),
            level: "info".to_owned(),
            action: "restart_managed_service".to_owned(),
            message: "restarted failed managed service svc-1".to_owned(),
            watcher_id: Some("server-recover".to_owned()),
            service_id: Some("svc-1".to_owned()),
        };

        append_audit_event(&paths, &event)?;

        let text = fs::read_to_string(paths.audit_events_path())?;
        let parsed = serde_json::from_str::<AuditEventRecord>(text.trim())?;
        fs::remove_dir_all(root).ok();
        assert_eq!(parsed.category, "automation");
        assert_eq!(parsed.watcher_id.as_deref(), Some("server-recover"));
        assert_eq!(parsed.service_id.as_deref(), Some("svc-1"));
        Ok(())
    }

    #[test]
    fn append_and_load_recent_automation_proposals() -> Result<()> {
        let (root, paths) = temp_app_paths("append-proposal");
        append_automation_proposal(
            &paths,
            &AutomationProposalRecord {
                at_unix_ms: 1,
                proposal_id: "proposal-1".to_owned(),
                watcher_id: "therock-update".to_owned(),
                action: "queue_update_proposal".to_owned(),
                title: "Check TheRock updates".to_owned(),
                message: "run rocm update".to_owned(),
                status: "pending".to_owned(),
                service_id: None,
                tool: Some("check_updates".to_owned()),
                arguments: serde_json::json!({}),
                reviewed_at_unix_ms: None,
            },
        )?;
        append_automation_proposal(
            &paths,
            &AutomationProposalRecord {
                at_unix_ms: 2,
                proposal_id: "proposal-2".to_owned(),
                watcher_id: "server-recover".to_owned(),
                action: "queue_restart_proposal".to_owned(),
                title: "Restart service".to_owned(),
                message: "restart svc-1".to_owned(),
                status: "pending".to_owned(),
                service_id: Some("svc-1".to_owned()),
                tool: Some("restart_server".to_owned()),
                arguments: serde_json::json!({ "service_id": "svc-1" }),
                reviewed_at_unix_ms: None,
            },
        )?;

        let proposals = load_recent_automation_proposals(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(proposals.len(), 1);
        assert_eq!(proposals[0].watcher_id, "server-recover");
        assert_eq!(proposals[0].proposal_id, "proposal-2");
        assert_eq!(proposals[0].service_id.as_deref(), Some("svc-1"));
        Ok(())
    }

    #[test]
    fn proposal_status_update_rewrites_record() -> Result<()> {
        let (root, paths) = temp_app_paths("proposal-status");
        append_automation_proposal(
            &paths,
            &AutomationProposalRecord {
                at_unix_ms: 1,
                proposal_id: "proposal-1".to_owned(),
                watcher_id: "server-recover".to_owned(),
                action: "queue_restart_proposal".to_owned(),
                title: "Restart service".to_owned(),
                message: "restart svc-1".to_owned(),
                status: "pending".to_owned(),
                service_id: Some("svc-1".to_owned()),
                tool: Some("restart_server".to_owned()),
                arguments: serde_json::json!({ "service_id": "svc-1" }),
                reviewed_at_unix_ms: None,
            },
        )?;

        let updated = update_automation_proposal_status(&paths, "proposal-1", "rejected")?;
        let loaded = find_automation_proposal(&paths, "proposal-1")?;
        fs::remove_dir_all(root).ok();

        assert_eq!(updated.status, "rejected");
        assert_eq!(loaded.status, "rejected");
        assert!(loaded.reviewed_at_unix_ms.is_some());
        Ok(())
    }

    #[test]
    fn load_recent_audit_events_returns_tail() -> Result<()> {
        let (root, paths) = temp_app_paths("audit-tail");
        append_audit_event(
            &paths,
            &AuditEventRecord {
                at_unix_ms: 1,
                source: "rocm".to_owned(),
                category: "proposal".to_owned(),
                actor: "tui".to_owned(),
                level: "info".to_owned(),
                action: "proposal_approved".to_owned(),
                message: "approved proposal-1".to_owned(),
                watcher_id: None,
                service_id: None,
            },
        )?;
        append_audit_event(
            &paths,
            &AuditEventRecord {
                at_unix_ms: 2,
                source: "rocm".to_owned(),
                category: "proposal".to_owned(),
                actor: "tui".to_owned(),
                level: "info".to_owned(),
                action: "proposal_rejected".to_owned(),
                message: "rejected proposal-2".to_owned(),
                watcher_id: None,
                service_id: None,
            },
        )?;

        let events = load_recent_audit_events(&paths, 1)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].action, "proposal_rejected");
        Ok(())
    }

    #[test]
    fn builtin_model_catalog_json_parses_and_validates() {
        // Guards the embedded catalog: malformed JSON, a bad device policy, or a
        // duplicate alias/canonical id would fail the shared index schema here
        // instead of panicking at runtime.
        let document =
            serde_json::from_str::<ModelRecipeIndexDocument>(include_str!("model_catalog.json"))
                .expect("catalog JSON parses");
        document
            .validate()
            .expect("catalog satisfies the index schema");
        assert!(
            document.recipes.len() >= 10,
            "curated catalog is non-trivial"
        );
        // The default Lemonade assistant must remain resolvable from the catalog.
        assert!(
            document
                .recipes
                .iter()
                .any(|recipe| recipe.canonical_model_id == "Qwen3-4B-Instruct-2507-GGUF"),
            "built-in assistant recipe present"
        );
    }

    #[test]
    fn builtin_catalog_authors_vllm_tool_call_parsers() {
        // vLLM does not auto-detect a tool-call parser; the correct value is sourced
        // from explicit per-model recipe metadata (never guessed at runtime). This
        // guards the authored parser for well-known chat families so a regression is
        // caught here rather than as an HTTP 400 in the TUI chat tab.
        let document =
            serde_json::from_str::<ModelRecipeIndexDocument>(include_str!("model_catalog.json"))
                .expect("catalog JSON parses");
        let tool_call_parser = |model_id: &str| -> Option<String> {
            document
                .recipes
                .iter()
                .find(|recipe| recipe.canonical_model_id == model_id)?
                .engine_recipes
                .iter()
                .find(|engine_recipe| engine_recipe.engine == "vllm")
                .and_then(|engine_recipe| {
                    let flags = &engine_recipe.required_flags;
                    assert!(
                        flags.iter().any(|flag| flag == "--enable-auto-tool-choice"),
                        "{model_id}: --tool-call-parser must be paired with --enable-auto-tool-choice"
                    );
                    let index = flags.iter().position(|flag| flag == "--tool-call-parser")?;
                    flags.get(index + 1).cloned()
                })
        };
        // Reported repro: a lemonade-preferred Qwen forced onto vLLM must still
        // carry the Qwen-family parser.
        assert_eq!(
            tool_call_parser("Qwen/Qwen2.5-1.5B-Instruct").as_deref(),
            Some("hermes")
        );
        assert_eq!(
            tool_call_parser("Qwen/Qwen3-32B-FP8").as_deref(),
            Some("hermes")
        );
        assert_eq!(
            tool_call_parser("meta-llama/Llama-3.2-3B-Instruct").as_deref(),
            Some("llama3_json")
        );
    }

    #[test]
    fn model_recipe_target_platform_groups_by_engine() {
        let registry = builtin_model_recipe_registry();
        let platforms = model_catalog_platforms(&registry);
        // The (hidden) built-in assistant is a Lemonade recipe → Ryzen AI (Strix Halo).
        let strix = resolve_builtin_model_recipe("qwen").expect("qwen assistant");
        assert_eq!(
            model_recipe_target_platform_label(&strix, &platforms),
            "AMD Ryzen AI — Strix Halo (Lemonade / llama.cpp)"
        );
        let mi300x = resolve_builtin_model_recipe("qwen3.6-27b").expect("qwen3.6-27b");
        assert_eq!(
            model_recipe_target_platform_label(&mi300x, &platforms),
            "AMD Instinct — MI300X, MI350X, MI355X (vLLM)"
        );
        // vLLM recipes land on the Instinct platform.
        let llama = resolve_builtin_model_recipe("llama-3.2-3b-instruct").expect("llama");
        assert_eq!(
            model_recipe_target_platform_label(&llama, &platforms),
            "AMD Instinct — MI300X, MI350X, MI355X (vLLM)"
        );
    }

    #[test]
    fn featured_catalog_is_curated_but_hidden_stay_resolvable() {
        // Current popular models are featured in the curated list — GGUF for
        // Strix Halo (served by their owner/repo:variant id) and BF16 for MI300X.
        for alias in ["qwen3.6", "gemma-4", "qwen3.6-27b", "gemma-4-31b"] {
            let recipe = resolve_builtin_model_recipe(alias).unwrap_or_else(|| panic!("{alias}"));
            assert!(model_recipe_featured(&recipe), "{alias} should be featured");
        }
        // The Strix Halo entries carry an explicit GGUF quant variant so they are
        // directly servable via Lemonade.
        assert_eq!(
            resolve_builtin_model_recipe("qwen3.6")
                .unwrap()
                .canonical_model_id,
            "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M"
        );
        // ...while the default assistant, smoke paths, and superseded models stay
        // resolvable for `rocm serve` but are hidden from the curated list.
        for alias in ["qwen", "tiny-gpt2", "qwen3.5", "glm-5"] {
            let recipe = resolve_builtin_model_recipe(alias).unwrap_or_else(|| panic!("{alias}"));
            assert!(!model_recipe_featured(&recipe), "{alias} should be hidden");
        }
    }

    #[test]
    fn builtin_recipe_resolves_alias_and_canonical_model() {
        let qwen = resolve_builtin_model_recipe("qwen").expect("qwen alias should resolve");
        assert_eq!(qwen.canonical_model_id, "Qwen3-4B-Instruct-2507-GGUF");
        assert_eq!(qwen.dtype, "gguf");
        assert_eq!(qwen.device_policy, "gpu_required");
        assert_eq!(qwen.preferred_engines, vec!["lemonade"]);

        let qwen35 = resolve_builtin_model_recipe("qwen3.5").expect("qwen3.5 alias should resolve");
        assert_eq!(qwen35.canonical_model_id, "Qwen/Qwen3.5-4B");
        assert_eq!(qwen35.preferred_engines, vec!["vllm"]);
        let lemonade_qwen =
            resolve_builtin_model_recipe("lemonade-qwen").expect("lemonade qwen alias");
        assert_eq!(
            lemonade_qwen.canonical_model_id,
            "Qwen3-4B-Instruct-2507-GGUF"
        );
        assert_eq!(lemonade_qwen.preferred_engines, vec!["lemonade"]);
        assert_eq!(lemonade_qwen.device_policy, "gpu_required");
        assert!(
            qwen35
                .warnings
                .iter()
                .any(|warning| warning.contains("qwen3_5"))
        );

        let tiny = resolve_builtin_model_recipe("sshleifer/tiny-gpt2")
            .expect("canonical tiny model should resolve");
        assert_eq!(tiny.canonical_model_id, "sshleifer/tiny-gpt2");
        assert_eq!(tiny.device_policy, "gpu_required");
        assert_eq!(tiny.min_gpu_mem_gb, Some(2));
    }

    #[test]
    fn builtin_recipe_records_remote_code_policy() {
        let glm = resolve_builtin_model_recipe("glm-5").expect("glm alias should resolve");
        assert!(glm.trust_remote_code);
        assert_eq!(glm.device_policy, "gpu_required");
        assert!(
            glm.warnings
                .iter()
                .any(|item| item.contains("trust_remote_code"))
        );
    }

    #[test]
    fn model_recipe_index_validates_artifact_metadata() -> Result<()> {
        let mut recipe = sample_recipe_with_artifact("Qwen/Test-1B", &["test-qwen"]);
        recipe.artifacts[0].uri =
            "https://huggingface.co/Qwen/Test-1B/resolve/main/model.safetensors".to_owned();
        recipe.artifacts[0].source_policy = Some(ModelRecipeArtifactSourcePolicyRecord {
            policy: "huggingface_public".to_owned(),
            required_hosts: vec!["huggingface.co".to_owned()],
            notes: vec!["test metadata only".to_owned()],
        });
        recipe.engine_recipes.push(ModelRecipeEngineRecord {
            engine: "vllm".to_owned(),
            required_flags: vec!["--enable-auto-tool-choice".to_owned()],
            parser_settings: BTreeMap::from([("reasoning_parser".to_owned(), "qwen3".to_owned())]),
            preferred_endpoint: Some(ModelRecipeEndpointRecord {
                endpoint_mode: "openai".to_owned(),
                settings: BTreeMap::from([("streaming".to_owned(), "true".to_owned())]),
            }),
            unsupported_combinations: vec![ModelRecipeUnsupportedCombinationRecord {
                combination: "native Windows GPU serving".to_owned(),
                reason: "vLLM ROCm serving is Linux/WSL only".to_owned(),
            }],
            notes: vec!["metadata only; adapter protocol does not consume this yet".to_owned()],
            model_id_override: None,
        });
        let index = ModelRecipeIndexDocument {
            schema_version: 1,
            source: Some("fixture".to_owned()),
            generated_at_unix_ms: Some(123),
            platforms: Vec::new(),
            recipes: vec![recipe],
        };

        index.validate()?;

        let artifact = &index.recipes[0].artifacts[0];
        assert_eq!(artifact.kind, "huggingface");
        let expected_sha = "a".repeat(64);
        assert_eq!(artifact.sha256.as_deref(), Some(expected_sha.as_str()));
        assert_eq!(artifact.engines, vec!["vllm"]);
        assert_eq!(
            artifact
                .source_policy
                .as_ref()
                .map(|policy| policy.policy.as_str()),
            Some("huggingface_public")
        );
        let settings = index.recipes[0]
            .engine_recipes
            .first()
            .expect("vllm settings should validate");
        assert_eq!(
            settings.parser_settings.get("reasoning_parser"),
            Some(&"qwen3".to_owned())
        );
        assert_eq!(
            settings
                .preferred_endpoint
                .as_ref()
                .map(|endpoint| endpoint.endpoint_mode.as_str()),
            Some("openai")
        );
        Ok(())
    }

    #[test]
    fn model_recipe_index_rejects_invalid_artifact_source_policy() {
        let mut recipe = sample_recipe_with_artifact("Qwen/Test-1B", &["test-qwen"]);
        recipe.artifacts[0].uri =
            "https://example.invalid/Qwen/Test-1B/model.safetensors".to_owned();
        recipe.artifacts[0].source_policy = Some(ModelRecipeArtifactSourcePolicyRecord {
            policy: "huggingface_authenticated".to_owned(),
            required_hosts: vec!["huggingface.co".to_owned()],
            notes: Vec::new(),
        });

        let error = ModelRecipeIndexDocument {
            schema_version: 1,
            source: Some("fixture".to_owned()),
            generated_at_unix_ms: Some(123),
            platforms: Vec::new(),
            recipes: vec![recipe],
        }
        .validate()
        .expect_err("source policy host mismatch should be rejected")
        .to_string();

        assert!(error.contains("source_policy"));
        assert!(error.contains("not allowed"));
    }

    #[test]
    fn model_recipe_index_source_policy_requires_integrity_metadata() {
        let mut recipe = sample_recipe_with_artifact("Qwen/Test-1B", &["test-qwen"]);
        recipe.artifacts[0].uri = "https://example.invalid/model.bin".to_owned();
        recipe.artifacts[0].sha256 = None;
        recipe.artifacts[0].source_policy = Some(ModelRecipeArtifactSourcePolicyRecord {
            policy: "direct_https_sha256".to_owned(),
            required_hosts: vec!["example.invalid".to_owned()],
            notes: Vec::new(),
        });

        let error = ModelRecipeIndexDocument {
            schema_version: 1,
            source: Some("fixture".to_owned()),
            generated_at_unix_ms: Some(123),
            platforms: Vec::new(),
            recipes: vec![recipe],
        }
        .validate()
        .expect_err("source policy should require sha256")
        .to_string();

        assert!(error.contains("requires sha256"));
    }

    #[test]
    fn model_recipe_index_rejects_empty_engine_recipe() {
        let mut recipe = sample_recipe_with_artifact("Qwen/Test-1B", &["test-qwen"]);
        recipe.engine_recipes.push(ModelRecipeEngineRecord {
            engine: "vllm".to_owned(),
            ..ModelRecipeEngineRecord::default()
        });

        let error = ModelRecipeIndexDocument {
            schema_version: 1,
            source: Some("fixture".to_owned()),
            generated_at_unix_ms: Some(123),
            platforms: Vec::new(),
            recipes: vec![recipe],
        }
        .validate()
        .expect_err("empty engine recipe should be rejected")
        .to_string();

        assert!(error.contains("engine recipe for `vllm`"));
        assert!(error.contains("must not be empty"));
    }

    #[test]
    fn model_recipe_index_rejects_duplicate_engine_recipes() {
        let mut recipe = sample_recipe_with_artifact("Qwen/Test-1B", &["test-qwen"]);
        recipe.engine_recipes = vec![
            ModelRecipeEngineRecord {
                engine: "vllm".to_owned(),
                notes: vec!["first".to_owned()],
                ..ModelRecipeEngineRecord::default()
            },
            ModelRecipeEngineRecord {
                engine: "VLLM".to_owned(),
                notes: vec!["second".to_owned()],
                ..ModelRecipeEngineRecord::default()
            },
        ];

        let error = ModelRecipeIndexDocument {
            schema_version: 1,
            source: Some("fixture".to_owned()),
            generated_at_unix_ms: Some(123),
            platforms: Vec::new(),
            recipes: vec![recipe],
        }
        .validate()
        .expect_err("duplicate engine recipes should be rejected")
        .to_string();

        assert!(error.contains("engine recipe for `VLLM`"));
        assert!(error.contains("duplicated"));
    }

    #[test]
    fn model_recipe_index_requires_unsupported_combination_reason() {
        let mut recipe = sample_recipe_with_artifact("Qwen/Test-1B", &["test-qwen"]);
        recipe.engine_recipes.push(ModelRecipeEngineRecord {
            engine: "vllm".to_owned(),
            unsupported_combinations: vec![ModelRecipeUnsupportedCombinationRecord {
                combination: "native Windows GPU serving".to_owned(),
                reason: String::new(),
            }],
            ..ModelRecipeEngineRecord::default()
        });

        let error = ModelRecipeIndexDocument {
            schema_version: 1,
            source: Some("fixture".to_owned()),
            generated_at_unix_ms: Some(123),
            platforms: Vec::new(),
            recipes: vec![recipe],
        }
        .validate()
        .expect_err("unsupported combinations need reasons")
        .to_string();

        assert!(error.contains("engine unsupported combination reason"));
    }

    #[test]
    fn model_artifact_cache_status_uses_deterministic_marker_without_creating_dirs() -> Result<()> {
        let (root, paths) = temp_app_paths("artifact-cache-status");
        let mut recipe = sample_recipe_with_artifact("Qwen/Test-1B", &["test-qwen"]);
        let artifact = recipe.artifacts.remove(0);

        let missing = model_artifact_cache_status(&paths, "Qwen/Test-1B", &artifact);
        assert_eq!(missing.status, "missing");
        assert!(
            missing
                .marker_path
                .to_string_lossy()
                .contains("hf-main--x68662d6d61696e.json")
        );
        assert!(
            missing
                .marker_path
                .to_string_lossy()
                .contains("qwen-test-1b")
        );
        assert!(!paths.data_dir.exists());

        let parent = missing.marker_path.parent().expect("marker has parent");
        fs::create_dir_all(parent)?;
        fs::write(&missing.marker_path, "{}")?;
        let present = model_artifact_cache_status(&paths, "Qwen/Test-1B", &artifact);

        fs::remove_dir_all(root).ok();
        assert_eq!(present.status, "metadata_present");
        Ok(())
    }

    #[test]
    fn model_artifact_cache_marker_path_includes_model_identity() {
        let (_root, paths) = temp_app_paths("artifact-cache-model-scope");

        let first = model_artifact_cache_marker_path(&paths, "Qwen/Test-1B", "hf-main");
        let second = model_artifact_cache_marker_path(&paths, "Qwen/Other-1B", "hf-main");

        assert_ne!(first, second);
        assert!(first.to_string_lossy().contains("qwen-test-1b"));
        assert!(second.to_string_lossy().contains("qwen-other-1b"));
    }

    #[test]
    fn model_artifact_cache_marker_path_is_collision_proof_for_similar_refs() {
        let (_root, paths) = temp_app_paths("artifact-cache-collision-proof");

        let dash = model_artifact_cache_marker_path(&paths, "Qwen/Test-1B", "hf-main");
        let underscore = model_artifact_cache_marker_path(&paths, "Qwen/Test_1B", "hf-main");
        let case_variant = model_artifact_cache_marker_path(&paths, "qwen/test-1b", "hf-main");

        assert_ne!(dash, underscore);
        assert_ne!(dash, case_variant);
        assert!(
            dash.to_string_lossy()
                .contains("--x5177656e2f546573742d3142")
        );
        assert!(
            underscore
                .to_string_lossy()
                .contains("--x5177656e2f546573745f3142")
        );
        assert!(
            case_variant
                .to_string_lossy()
                .contains("--x7177656e2f746573742d3162")
        );
    }

    #[test]
    fn model_recipe_index_rejects_duplicate_aliases() {
        let error = ModelRecipeIndexDocument {
            schema_version: 1,
            source: None,
            generated_at_unix_ms: None,
            platforms: Vec::new(),
            recipes: vec![
                sample_recipe_with_artifact("Qwen/Test-1B", &["shared-alias"]),
                sample_recipe_with_artifact("Qwen/Other-1B", &["shared-alias"]),
            ],
        }
        .validate()
        .expect_err("duplicate aliases should be rejected")
        .to_string();

        assert!(error.contains("duplicated"));
        assert!(error.contains("shared-alias"));
    }

    #[test]
    fn load_model_recipe_index_reads_local_fixture() -> Result<()> {
        let (root, _paths) = temp_app_paths("recipe-index-fixture");
        let index_path = root.join("recipes.json");
        let document = ModelRecipeIndexDocument {
            schema_version: 1,
            source: Some("fixture".to_owned()),
            generated_at_unix_ms: Some(123),
            platforms: Vec::new(),
            recipes: vec![sample_recipe_with_artifact("Qwen/Test-1B", &["test-qwen"])],
        };
        fs::create_dir_all(&root)?;
        fs::write(&index_path, serde_json::to_vec_pretty(&document)?)?;

        let loaded = load_model_recipe_index(&index_path)?;
        fs::remove_dir_all(root).ok();

        assert_eq!(loaded.source.as_deref(), Some("fixture"));
        assert_eq!(loaded.recipes[0].canonical_model_id, "Qwen/Test-1B");
        assert_eq!(loaded.recipes[0].artifacts.len(), 1);
        Ok(())
    }

    #[test]
    fn model_recipe_index_signature_path_is_detached_sidecar() {
        assert_eq!(
            model_recipe_index_signature_path(Path::new("recipes/index.json")),
            PathBuf::from("recipes/index.json.sig")
        );
    }

    #[test]
    fn model_recipe_index_signature_accepts_generated_key_and_rejects_tamper() -> Result<()> {
        let (root, _paths) = temp_app_paths("recipe-index-generated-signature");
        fs::create_dir_all(&root)?;
        let private_key = root.join("recipe-private.pem");
        let public_key = root.join("recipe-public.pem");
        let index_path = root.join("recipes.json");
        let signature_path = model_recipe_index_signature_path(&index_path);
        let document = ModelRecipeIndexDocument {
            schema_version: 1,
            source: Some("fixture".to_owned()),
            generated_at_unix_ms: Some(123),
            platforms: Vec::new(),
            recipes: vec![sample_recipe_with_artifact("Qwen/Test-1B", &["test-qwen"])],
        };

        generate_test_signing_key(&private_key, &public_key)?;
        fs::write(&index_path, serde_json::to_vec_pretty(&document)?)?;
        sign_test_payload(&private_key, &index_path, &signature_path)?;

        load_signed_model_recipe_index(&index_path, &signature_path, &public_key)?;

        let tampered = ModelRecipeIndexDocument {
            source: Some("tampered".to_owned()),
            ..document
        };
        fs::write(&index_path, serde_json::to_vec_pretty(&tampered)?)?;
        let error = load_signed_model_recipe_index(&index_path, &signature_path, &public_key)
            .unwrap_err()
            .to_string();

        assert!(error.contains("model recipe index signature verification failed"));
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    /// Produce a `(private-key PEM, SPKI public-key PEM, payload, signature)`
    /// tuple with the OpenSSL CLI, the same tool the installers and release
    /// signing use (`install.sh`, `scripts/package-linux-release.sh`). Returns
    /// `None` when openssl is unavailable or its spawn fails, so this interop
    /// guard never reintroduces the build-failing flake we removed: the pure-Rust
    /// sign/verify path is fully covered by the round-trip test, and this only
    /// adds cross-checking against real openssl output where openssl exists.
    fn openssl_signed_vector(dir: &Path) -> Option<(String, String, Vec<u8>, Vec<u8>)> {
        let private_key = dir.join("interop-private.pem");
        let public_key = dir.join("interop-public.pem");
        let payload_path = dir.join("interop-payload.bin");
        let signature_path = dir.join("interop.sig");
        fs::write(&payload_path, b"version = 1\n").ok()?;

        let run = |args: &[&str]| -> Option<()> {
            let output = Command::new("openssl").args(args).output().ok()?;
            output.status.success().then_some(())
        };
        run(&[
            "genpkey",
            "-algorithm",
            "RSA",
            "-pkeyopt",
            "rsa_keygen_bits:2048",
            "-out",
            private_key.to_string_lossy().as_ref(),
        ])?;
        run(&[
            "rsa",
            "-in",
            private_key.to_string_lossy().as_ref(),
            "-pubout",
            "-out",
            public_key.to_string_lossy().as_ref(),
        ])?;
        run(&[
            "dgst",
            "-sha256",
            "-sign",
            private_key.to_string_lossy().as_ref(),
            "-out",
            signature_path.to_string_lossy().as_ref(),
            payload_path.to_string_lossy().as_ref(),
        ])?;

        Some((
            fs::read_to_string(&private_key).ok()?,
            fs::read_to_string(&public_key).ok()?,
            fs::read(&payload_path).ok()?,
            fs::read(&signature_path).ok()?,
        ))
    }

    #[test]
    fn signing_tolerates_crlf_and_trailing_whitespace_pems() -> Result<()> {
        // Windows tooling (PowerShell Set-Content, editors) can rewrite a PEM with
        // CRLF endings, a stray trailing space on the boundary line, or a UTF-8 BOM.
        // The OpenSSL CLI accepted these, so the Rust path must too.
        let (private_pem, public_pem) = generate_rsa_signing_keypair()?;
        let payload = b"version = 1\n";

        let crlf_private = private_pem.replace('\n', "\r\n");
        let trailing_ws_private = private_pem
            .lines()
            .map(|line| format!("{line} "))
            .collect::<Vec<_>>()
            .join("\n");
        let bom_public = format!("\u{feff}{public_pem}");

        for variant in [crlf_private, trailing_ws_private] {
            let signature = sign_rsa_pkcs1_sha256_signature(&variant, payload)?;
            verify_rsa_pkcs1_sha256_signature(&public_pem, payload, &signature, "metadata")?;
        }
        let signature = sign_rsa_pkcs1_sha256_signature(&private_pem, payload)?;
        verify_rsa_pkcs1_sha256_signature(&bom_public, payload, &signature, "metadata")?;
        Ok(())
    }

    #[test]
    fn rsa_sign_verify_round_trips_in_pure_rust() -> Result<()> {
        let (private_pem, public_pem) = generate_rsa_signing_keypair()?;
        let payload = b"version = 1\n";

        let signature = sign_rsa_pkcs1_sha256_signature(&private_pem, payload)?;
        verify_rsa_pkcs1_sha256_signature(&public_pem, payload, &signature, "metadata")?;

        let mut tampered = payload.to_vec();
        tampered[0] ^= 0x01;
        let error =
            verify_rsa_pkcs1_sha256_signature(&public_pem, &tampered, &signature, "metadata")
                .expect_err("a tampered payload must be rejected")
                .to_string();
        assert!(error.contains("metadata signature verification failed"));
        Ok(())
    }

    #[test]
    fn rsa_verifier_is_byte_compatible_with_openssl_output() -> Result<()> {
        let (root, _paths) = temp_app_paths("openssl-interop-vector");
        fs::create_dir_all(&root)?;
        let Some((private_key_pem, public_key_pem, payload, openssl_signature)) =
            openssl_signed_vector(&root)
        else {
            eprintln!(
                "skipping openssl interop check: openssl CLI unavailable or failed to produce a signature"
            );
            fs::remove_dir_all(&root).ok();
            return Ok(());
        };

        // Our verifier accepts a signature produced by the openssl CLI.
        verify_rsa_pkcs1_sha256_signature(
            &public_key_pem,
            &payload,
            &openssl_signature,
            "metadata",
        )
        .expect("pure-Rust verifier must accept an openssl-produced signature");

        // Our signer is byte-identical to `openssl dgst -sha256 -sign`, so artifacts
        // signed by the Rust xtask verify with the openssl-based installers and vice-versa.
        let rust_signature = sign_rsa_pkcs1_sha256_signature(&private_key_pem, &payload)?;
        assert_eq!(
            rust_signature, openssl_signature,
            "Rust signature must match openssl byte-for-byte"
        );

        let mut tampered = payload;
        tampered[0] ^= 0x01;
        let error = verify_rsa_pkcs1_sha256_signature(
            &public_key_pem,
            &tampered,
            &openssl_signature,
            "metadata",
        )
        .expect_err("a tampered payload must be rejected")
        .to_string();
        assert!(error.contains("metadata signature verification failed"));
        fs::remove_dir_all(&root).ok();
        Ok(())
    }

    fn generate_test_signing_key(private_key: &Path, public_key: &Path) -> Result<()> {
        let (private_pem, public_pem) = generate_rsa_signing_keypair()?;
        fs::write(private_key, private_pem.as_bytes())?;
        fs::write(public_key, public_pem.as_bytes())?;
        Ok(())
    }

    fn sign_test_payload(private_key: &Path, payload: &Path, signature: &Path) -> Result<()> {
        let private_pem = fs::read_to_string(private_key)?;
        let payload_bytes = fs::read(payload)?;
        let produced = sign_rsa_pkcs1_sha256_signature(&private_pem, &payload_bytes)?;
        fs::write(signature, produced)?;
        Ok(())
    }

    fn sample_recipe_with_artifact(
        canonical_model_id: &str,
        aliases: &[&str],
    ) -> ModelRecipeRecord {
        ModelRecipeRecord {
            canonical_model_id: canonical_model_id.to_owned(),
            aliases: aliases.iter().map(|alias| (*alias).to_owned()).collect(),
            task: "chat".to_owned(),
            source: "signed_recipe_index".to_owned(),
            revision: "main".to_owned(),
            loader: "transformers".to_owned(),
            trust_remote_code: false,
            dtype: "bfloat16".to_owned(),
            device_policy: "gpu_required".to_owned(),
            min_gpu_mem_gb: Some(12),
            recommended_system_ram_gb: Some(16),
            quantization: Some("none".to_owned()),
            artifact_hint: None,
            artifacts: vec![ModelRecipeArtifactRecord {
                artifact_id: "hf-main".to_owned(),
                kind: "huggingface".to_owned(),
                uri: canonical_model_id.to_owned(),
                revision: Some("main".to_owned()),
                sha256: Some("a".repeat(64)),
                size_bytes: Some(1024),
                license: Some("apache-2.0".to_owned()),
                gated: Some(false),
                quantization: Some("none".to_owned()),
                engines: vec!["vllm".to_owned()],
                source_policy: None,
            }],
            engine_recipes: Vec::new(),
            manual_alternatives: Vec::new(),
            featured: false,
            chat_template_mode: "auto".to_owned(),
            preferred_engines: vec!["vllm".to_owned()],
            warnings: Vec::new(),
        }
    }

    #[test]
    fn config_defaults_to_local_telemetry_policy() {
        let config = RocmCliConfig::default();

        assert_eq!(config.telemetry.mode_label(), TELEMETRY_MODE_LOCAL);
        assert!(config.telemetry.local_inspection_enabled());
        assert!(config.telemetry.known_mode());
    }

    #[test]
    fn config_defaults_to_ask_permissions_and_incomplete_setup() {
        let config = RocmCliConfig::default();

        assert_eq!(config.permissions.mode_label(), PERMISSIONS_MODE_ASK);
        assert!(!config.permissions.full_access_enabled());
        assert!(!config.setup.completed);
        assert!(config.setup.therock_venv.is_none());
        assert!(config.planner_provider.is_none());
        assert!(config.tools.is_empty());
    }

    #[test]
    fn config_persists_setup_permissions_and_managed_tools() -> Result<()> {
        let (root, paths) = temp_app_paths("config-managed-state");
        let mut config = RocmCliConfig::default();
        let venv = paths.data_dir.join("runtimes").join("therock");
        let python = paths
            .data_dir
            .join("tools")
            .join("python")
            .join("python.exe");

        config.permissions.mode = PERMISSIONS_MODE_FULL_ACCESS.to_owned();
        config.planner_provider = Some("local".to_owned());
        config.setup.completed = true;
        config.setup.therock_venv = Some(venv.clone());
        config.tools.insert(
            "python".to_owned(),
            ManagedToolConfig {
                path: Some(python.clone()),
                managed: true,
            },
        );
        config.save(&paths)?;

        let loaded = RocmCliConfig::load(&paths)?;
        fs::remove_dir_all(root).ok();

        assert!(loaded.permissions.full_access_enabled());
        assert_eq!(loaded.planner_provider.as_deref(), Some("local"));
        assert!(loaded.setup.completed);
        assert_eq!(loaded.setup.therock_venv.as_deref(), Some(venv.as_path()));
        let tool = loaded.tools.get("python").expect("python tool should load");
        assert!(tool.managed);
        assert_eq!(tool.path.as_deref(), Some(python.as_path()));
        Ok(())
    }

    #[test]
    fn app_paths_discover_defaults_to_home_rocm_when_unoverridden() -> Result<()> {
        if env_path_override("ROCM_CLI_CONFIG_DIR").is_some()
            || env_path_override("ROCM_CLI_DATA_DIR").is_some()
            || env_path_override("ROCM_CLI_CACHE_DIR").is_some()
        {
            return Ok(());
        }
        let Some(home_rocm) = home_rocm_dir() else {
            return Ok(());
        };

        let paths = AppPaths::discover()?;

        assert_eq!(paths.config_dir, home_rocm);
        let home_paths = AppPaths {
            config_dir: home_rocm.clone(),
            data_dir: home_rocm.clone(),
            cache_dir: home_rocm.join("cache"),
        };
        if let Some(managed_root) = configured_managed_root_from_config(&home_paths) {
            let managed_root = normalize_runtime_path_for_host(&managed_root);
            assert_eq!(paths.data_dir, managed_root);
            assert_eq!(paths.cache_dir, managed_root.join("cache"));
        } else {
            assert_eq!(paths.data_dir, home_rocm);
            assert_eq!(paths.cache_dir, home_rocm.join("cache"));
        }
        assert_eq!(paths.config_path(), home_rocm.join("config.json"));
        Ok(())
    }

    #[test]
    #[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
    fn engine_envs_dir_honors_dedicated_root_override() {
        let (root, paths) = temp_app_paths("engine-envs-root-override");
        let override_root = root.join("runtime").join("engines");
        let previous = std::env::var_os("ROCM_CLI_ENGINE_ENVS_ROOT");
        unsafe {
            std::env::set_var("ROCM_CLI_ENGINE_ENVS_ROOT", &override_root);
        }

        assert_eq!(
            paths.engine_envs_dir("vllm"),
            normalize_runtime_path_for_host(&override_root)
                .join("vllm")
                .join("envs")
        );

        unsafe {
            match previous {
                Some(value) => std::env::set_var("ROCM_CLI_ENGINE_ENVS_ROOT", value),
                None => std::env::remove_var("ROCM_CLI_ENGINE_ENVS_ROOT"),
            }
        }
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn legacy_config_without_telemetry_uses_default_policy() -> Result<()> {
        let config = serde_json::from_value::<RocmCliConfig>(serde_json::json!({
            "default_engine": "vllm"
        }))?;

        assert_eq!(config.default_engine.as_deref(), Some("vllm"));
        assert_eq!(config.telemetry.mode_label(), TELEMETRY_MODE_LOCAL);
        Ok(())
    }

    #[test]
    fn provider_config_defaults_to_local_only() {
        let mut config = RocmCliConfig::default();

        assert!(config.provider_enabled("local"));
        assert!(!config.provider_enabled("openai"));
        assert!(!config.provider_enabled("anthropic"));

        config.provider_config_mut("openai").enabled = true;
        assert!(config.provider_enabled("openai"));
    }

    #[test]
    fn builtin_watchers_include_read_only_gpu_metrics() {
        let watcher = builtin_watcher("gpu-metrics").expect("gpu-metrics watcher should exist");

        assert_eq!(watcher.default_mode, WatcherMode::Observe);
        assert!(watcher.trigger.contains("gpu.metrics"));
        assert_eq!(watcher.actions, &["record_gpu_metrics"]);
    }

    #[test]
    fn builtin_watchers_include_reviewed_cache_warm() {
        let watcher = builtin_watcher("cache-warm").expect("cache-warm watcher should exist");

        assert_eq!(watcher.default_mode, WatcherMode::Propose);
        assert!(watcher.trigger.contains("cache.warm"));
        assert_eq!(watcher.actions, &["queue_prefetch_proposal"]);
    }

    #[test]
    fn builtin_watchers_include_reviewed_driver_upgrade() {
        let watcher =
            builtin_watcher("driver-upgrade").expect("driver-upgrade watcher should exist");

        assert_eq!(watcher.default_mode, WatcherMode::Propose);
        assert!(watcher.trigger.contains("update.available"));
        assert!(watcher.trigger.contains("component=driver"));
        assert_eq!(watcher.actions, &["prepare_driver_plan"]);
    }

    #[test]
    fn builtin_watchers_include_reviewed_gpu_thermal_protect() {
        let watcher = builtin_watcher("gpu-thermal-protect")
            .expect("gpu-thermal-protect watcher should exist");

        assert_eq!(watcher.default_mode, WatcherMode::Propose);
        assert!(watcher.trigger.contains("gpu.thermal_pressure"));
        assert!(watcher.trigger.contains("gpu.memory_pressure"));
        assert_eq!(watcher.actions, &["queue_stop_server_proposal"]);
    }

    #[test]
    fn engine_plugin_dirs_are_data_owned_and_ordered() {
        let (_root, paths) = temp_app_paths("engine-plugin-dirs");

        assert_eq!(
            engine_plugin_dirs(&paths),
            vec![
                paths.primary_engine_plugin_dir(),
                paths.data_dir.join("engines")
            ]
        );
    }

    #[test]
    fn http_host_formatting_brackets_ipv6_literals() {
        assert_eq!(format_host_port("127.0.0.1", 11435), "127.0.0.1:11435");
        assert_eq!(
            format_http_base_url("localhost", 11435),
            "http://localhost:11435"
        );
        assert_eq!(format_host_port("::1", 11435), "[::1]:11435");
        assert_eq!(format_http_base_url("::1", 11435), "http://[::1]:11435");
        assert_eq!(format_host_port("[::1]", 11435), "[::1]:11435");
    }

    fn temp_app_paths(name: &str) -> (PathBuf, AppPaths) {
        let root = workspace_test_artifact_dir().join(format!(
            "rocm-core-{name}-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        (root, paths)
    }

    fn workspace_test_artifact_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join(".rocm-work")
            .join("tests")
            .join("core")
    }

    fn write_fake_rocm_agent_enumerator(bin_dir: &Path, target: &str) -> Result<()> {
        if cfg!(windows) {
            let path = bin_dir.join("rocm_agent_enumerator.cmd");
            fs::write(path, format!("@echo off\r\necho {target}\r\n"))?;
        } else {
            let path = bin_dir.join("rocm_agent_enumerator");
            fs::write(&path, format!("#!/bin/sh\necho {target}\n"))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&path, fs::Permissions::from_mode(0o755))?;
            }
        }
        Ok(())
    }

    // ===== Dashboard sub-config + migration =====

    #[test]
    #[allow(clippy::float_cmp)]
    fn dashboard_config_defaults_and_json_round_trip() {
        let cfg = DashboardConfig::default();
        assert!(
            cfg.daemon.listen.starts_with("unix:") && cfg.daemon.listen.ends_with("rocmdashd.sock"),
            "default listen must be a unix socket path ending with rocmdashd.sock, got: {}",
            cfg.daemon.listen
        );
        assert_eq!(cfg.daemon.gpu_tick_secs, 1.0);
        assert_eq!(cfg.daemon.discovery_tick_secs, 5.0);
        assert_eq!(cfg.daemon.instance_tick_secs, 2.0);
        assert_eq!(cfg.tui.theme, "default-dark");
        assert_eq!(cfg.tui.chat_url, None);
        // daemon and tui defaults must agree so a default client finds the daemon.
        assert_eq!(cfg.daemon.listen, cfg.tui.connect);

        let json = serde_json::to_string(&cfg).unwrap();
        let back: DashboardConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back, cfg);
    }

    #[test]
    fn rocm_cli_config_dashboard_section_is_optional() {
        // A config.json with no `dashboard` key parses to the default sub-config.
        let json = r#"{"default_engine":"vllm"}"#;
        let cfg: RocmCliConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.default_engine.as_deref(), Some("vllm"));
        assert_eq!(cfg.dashboard, DashboardConfig::default());
    }

    #[test]
    fn dashboard_with_transforms_are_immutable_and_scoped() {
        let base = DashboardConfig::default();
        let chat = base
            .clone()
            .with_chat_endpoint("http://127.0.0.1:8000", "llama-3.1-8b");
        // Original is untouched (immutable transform).
        assert_eq!(base.tui.chat_url, None);
        assert_eq!(chat.tui.chat_url.as_deref(), Some("http://127.0.0.1:8000"));
        assert_eq!(chat.tui.chat_model.as_deref(), Some("llama-3.1-8b"));
        assert_eq!(chat.tui.chat_auth_header, None);

        let themed = base.clone().with_theme("nord");
        assert_eq!(base.tui.theme, "default-dark");
        assert_eq!(themed.tui.theme, "nord");

        let relisten = base.clone().with_daemon_listen("tcp:127.0.0.1:9000");
        assert!(
            base.daemon.listen.starts_with("unix:"),
            "default listen must use unix scheme"
        );
        assert_eq!(relisten.daemon.listen, "tcp:127.0.0.1:9000");
    }

    #[test]
    fn dashboard_daemon_tick_accessors_map_secs_to_duration() {
        let d = DashboardDaemonConfig {
            gpu_tick_secs: 0.5,
            discovery_tick_secs: 10.0,
            instance_tick_secs: 3.0,
            ..Default::default()
        };
        assert_eq!(d.gpu_tick(), Duration::from_secs_f64(0.5));
        assert_eq!(d.discovery_tick(), Duration::from_secs(10));
        assert_eq!(d.instance_tick(), Duration::from_secs(3));
    }

    #[test]
    fn dashboard_bench_results_path_is_derived_not_persisted() {
        let config = DashboardDaemonConfig::default();
        assert_eq!(config.bench_results_dir, None);

        let json = serde_json::to_value(config).unwrap();
        assert!(
            json.get("bench_results_dir").is_none(),
            "machine-specific derived path must not be serialized"
        );
    }

    #[test]
    fn app_paths_expose_telemetry_and_daemon_log_paths() -> Result<()> {
        let (root, paths) = temp_app_paths("telemetry-paths");
        assert_eq!(
            paths.telemetry_state_dir(),
            paths.data_dir.join("telemetry")
        );
        assert_eq!(
            paths.daemon_log_path(),
            paths.data_dir.join("logs").join("rocmdashd.log")
        );
        assert_eq!(paths.client_log_dir(), paths.data_dir.join("logs"));
        // ensure() creates the telemetry state dir alongside the others.
        paths.ensure()?;
        assert!(paths.telemetry_state_dir().is_dir());
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn migrate_legacy_dashboard_toml_maps_knobs_and_is_one_shot() -> Result<()> {
        let (root, paths) = temp_app_paths("migrate-dash");
        paths.ensure()?;
        let legacy = root.join("legacy-config.toml");
        fs::write(
            &legacy,
            r#"
default_engine = "vllm"

[daemon]
listen = "unix:/tmp/custom.sock"
token = "secret"
gpu_tick = 0.5
discovery_tick = 10
instance_tick = 3

[tui]
connect = "unix:/tmp/custom.sock"
theme = "nord"
chat_url = "http://127.0.0.1:8000"
chat_model = "llama-3.1-8b"

[engines.vllm]
preferred_env_id = "env-1"
last_installed_runtime_id = "therock-release"
"#,
        )?;

        // First migration writes config.json once and reports the legacy path.
        let migrated = RocmCliConfig::migrate_legacy_dashboard_toml_from(&paths, &legacy)?;
        assert_eq!(migrated.as_deref(), Some(legacy.as_path()));
        assert!(paths.config_path().is_file());
        // The legacy TOML is left untouched.
        assert!(legacy.is_file());

        // The written config maps every knob into the dashboard sub-config and
        // the canonical engine fields.
        let loaded = RocmCliConfig::load(&paths)?;
        assert_eq!(loaded.dashboard.daemon.listen, "unix:/tmp/custom.sock");
        assert_eq!(loaded.dashboard.daemon.token.as_deref(), Some("secret"));
        assert_eq!(loaded.dashboard.daemon.gpu_tick_secs, 0.5);
        assert_eq!(loaded.dashboard.daemon.discovery_tick_secs, 10.0);
        assert_eq!(loaded.dashboard.daemon.instance_tick_secs, 3.0);
        assert_eq!(loaded.dashboard.tui.connect, "unix:/tmp/custom.sock");
        assert_eq!(loaded.dashboard.tui.theme, "nord");
        assert_eq!(
            loaded.dashboard.tui.chat_url.as_deref(),
            Some("http://127.0.0.1:8000")
        );
        assert_eq!(
            loaded.dashboard.tui.chat_model.as_deref(),
            Some("llama-3.1-8b")
        );
        assert_eq!(loaded.default_engine.as_deref(), Some("vllm"));
        assert_eq!(
            loaded.engines["vllm"].preferred_env_id.as_deref(),
            Some("env-1")
        );

        // Second call is a no-op (config.json already exists — never clobbers).
        let again = RocmCliConfig::migrate_legacy_dashboard_toml_from(&paths, &legacy)?;
        assert_eq!(again, None);

        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn migrate_legacy_dashboard_toml_without_legacy_is_noop() -> Result<()> {
        let (root, paths) = temp_app_paths("migrate-dash-absent");
        paths.ensure()?;
        let legacy = root.join("does-not-exist.toml");
        let migrated = RocmCliConfig::migrate_legacy_dashboard_toml_from(&paths, &legacy)?;
        assert_eq!(migrated, None);
        assert!(!paths.config_path().is_file());
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn kfd_gfx_target_version_distinguishes_gpu_from_cpu_nodes() {
        // CPU topology nodes report a zero gfx target version; GPU nodes report a
        // nonzero one.
        assert!(!kfd_gfx_target_version_is_gpu("0"));
        assert!(!kfd_gfx_target_version_is_gpu(""));
        assert!(!kfd_gfx_target_version_is_gpu("not-a-number"));
        assert!(kfd_gfx_target_version_is_gpu("90402"));
        assert!(kfd_gfx_target_version_is_gpu("110000"));
    }

    #[test]
    fn combine_amd_gpu_counts_takes_larger_and_needs_one_readable_surface() {
        // Strix Halo shape: KFD reports 0 GPU nodes, DRM sees the iGPU → 1 present.
        assert_eq!(combine_amd_gpu_counts(Some(0), Some(1)), Some(1));
        // Discrete GPUs: both surfaces agree.
        assert_eq!(combine_amd_gpu_counts(Some(8), Some(8)), Some(8));
        // One surface unreadable → use the other.
        assert_eq!(combine_amd_gpu_counts(None, Some(1)), Some(1));
        assert_eq!(combine_amd_gpu_counts(Some(2), None), Some(2));
        // Neither readable → unknown (caller must not treat as zero).
        assert_eq!(combine_amd_gpu_counts(None, None), None);
        // Both agree on zero → authoritative no-GPU.
        assert_eq!(combine_amd_gpu_counts(Some(0), Some(0)), Some(0));
    }

    #[test]
    fn usable_gpu_indices_unset_mask_returns_all_present_devices() {
        assert_eq!(usable_amd_gpu_indices_from(0, None), Some(Vec::new()));
        assert_eq!(usable_amd_gpu_indices_from(1, None), Some(vec![0]));
        assert_eq!(usable_amd_gpu_indices_from(3, None), Some(vec![0, 1, 2]));
    }

    #[test]
    fn usable_gpu_indices_empty_mask_hides_every_device() {
        // The masked-device path: GPUs are present but fully masked out.
        assert_eq!(
            usable_amd_gpu_indices_from(2, Some(String::new())),
            Some(Vec::new())
        );
    }

    #[test]
    fn usable_gpu_indices_honors_valid_ordinal_masks() {
        assert_eq!(
            usable_amd_gpu_indices_from(4, Some("2,0".to_owned())),
            Some(vec![2, 0])
        );
        // Duplicates are collapsed.
        assert_eq!(
            usable_amd_gpu_indices_from(2, Some("1,1".to_owned())),
            Some(vec![1])
        );
    }

    #[test]
    fn usable_gpu_indices_treats_unsupported_masks_as_unprobeable() {
        assert_eq!(usable_amd_gpu_indices_from(2, Some("0,5".to_owned())), None);
        assert_eq!(
            usable_amd_gpu_indices_from(2, Some("GPU-deadbeef".to_owned())),
            None
        );
    }
}
