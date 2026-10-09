// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
#[cfg(windows)]
use std::ffi::OsStr;
use std::ffi::OsString;
use std::fs;
use std::io::{Read, Write};
use std::net::{IpAddr, TcpStream, ToSocketAddrs};
#[cfg(windows)]
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::Command;
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

pub mod browser;
pub mod diagnose;
pub mod disk_space;
pub mod examine;
pub mod fix;
pub mod host_gpu;
pub mod managed_runtime;
pub mod model_readiness;
pub mod openmpi;
pub mod proc_lifecycle;
pub mod report;
pub mod report_delivery;
pub mod rocm_install;
pub mod runtime;
pub mod terminal;

#[cfg(test)]
mod test_env;
#[cfg(test)]
mod test_support;
pub mod uv;
pub use diagnose::{
    DiagnoseReport, Diagnosis, Fix, Route, VLLM_OOM_CANONICAL_SYMPTOM, diagnose as run_diagnose,
    render_report_text as render_diagnose_text, vllm_oom_symptom_is_diagnosable,
};
pub use disk_space::{
    SpaceCheck, available_space_for_path, check_space_for_path, ensure_space_for,
    estimated_extracted_size, format_bytes, insufficient_space_message, map_write_error,
    mount_for_path, on_same_mount, warn_if_low_space, with_margin,
};
pub use examine::{
    Examination, FrameworkProbe, WSL_PLATFORM_NOTE, gfx_is_apu_family, probe_wsl_distro_from_host,
};
pub use fix::{
    CATALOG_CONTRACT_VERSION, CatalogManifest, FixOptions, ManifestEntry, ManifestPlatform,
    apply as apply_fix, catalog_manifest, catalog_manifest_json, exit as fix_exit,
    list_recipes as list_fix_recipes,
};
pub use host_gpu::{
    DriverSummary, ExamineSummary, HostGpuSummary, WslHostDriverProbe, WslSummary,
    default_engine_for_host, default_engine_for_platform, detect_gpu_driver_version,
    detect_host_gfx_target, detect_host_gpu_diagnostics, detect_host_gpu_summary,
    detect_system_ram_gib, extract_first_gfx_token, has_usable_amd_gpu, interactive_terminal,
    is_amdgpu_device, is_wsl_host, known_therock_families, normalize_therock_family,
    preferred_serve_engine_for_host_gpu_summary, usable_amd_gpu_indices,
};
#[cfg(any(target_os = "linux", test))]
use host_gpu::{KfdGpuNode, kfd_gpu_nodes_in};
use host_gpu::{
    collect_sdk_library_paths, detect_linux_sysfs_gfx_target, detect_local_windows_host_driver,
    detect_wsl_host_driver, detect_wsl_summary, is_wsl1_kernel, ldconfig_cache,
    ldconfig_lists_librocdxg, managed_therock_sdk_probe_candidates,
};
pub use managed_runtime::{
    FrameworkInterpreter, ManagedRuntimeEnvironment, active_managed_framework_interpreter,
    active_managed_therock_channel, active_managed_therock_environment,
    active_managed_therock_version, detect_managed_therock_family, prepend_runtime_paths,
};
use managed_runtime::{
    TheRockFamilyManifest, managed_sdk_tool_path, managed_therock_environment_records,
};
pub use proc_lifecycle::{
    IdentityState, KillScope, ProcessIdentity, TerminationOutcome, identity_state,
    process_start_ticks, terminate_verified,
};
pub use report::{
    APPROVED_ARCHITECTURES, APPROVED_ARCHITECTURES_SOURCE, REPORT_SCHEMA_VERSION, ReadOutcome,
    Refusal as ReportRefusal, Report, is_rocm_supported, prepare_report, read_report,
    refusal_envelope,
};
pub use rocm_install::{LegacyRocmSummary, detect_legacy_rocm_sdk, detect_legacy_rocm_summary};
use rocm_install::{RocmInstall, discover_rocm_installs, rocm_install_version};
use runtime::env_path_override;
pub use runtime::{
    RUNTIME_LIBRARY_PATH_ENV, RuntimeHost, RuntimePlatform, current_executable_path,
    default_cache_dir, default_config_dir, default_data_dir, default_interactive_shell_program,
    managed_logs_dir, managed_pip_cache_dir, managed_runtime_cache_dir, managed_runtime_data_root,
    managed_tools_dir, managed_uv_cache_dir, normalize_runtime_path_for_host,
    normalize_runtime_path_for_storage, normalize_runtime_path_text_for_host,
    normalize_runtime_path_text_for_platform, normalize_runtime_path_text_for_storage,
    platform_binary_name, prepend_runtime_path, resolve_path_through_symlinks,
    runtime_directory_label, runtime_drive_root_for_key, runtime_drive_roots, runtime_exe_suffix,
    runtime_home_dir, runtime_install_root_is_protected, runtime_is_linux, runtime_is_windows,
    runtime_os_name, runtime_path_for_child, runtime_path_for_windows_child,
    runtime_path_is_same_or_inside, runtime_path_list_join, runtime_path_list_split,
    runtime_path_sort_key, runtime_path_text_is_absolute_for_host,
    runtime_path_text_is_absolute_for_platform, runtime_paths_equivalent,
    runtime_python_activation_hint, runtime_python_activation_script, runtime_python_bin_dir_name,
    runtime_python_env_bin_dir, runtime_python_executable_in_env, runtime_python_executable_name,
    runtime_rocm_library_filename, shell_command_for_host, user_runtime_dir,
};
pub use uv::{
    DEFAULT_UV_TIMEOUT_SECS, DependencyViolation, UV_CACHE_DIR_ENV, UV_CACHE_DIR_OVERRIDE_ENV,
    UvCacheSource, ViolationSubject, check_dependencies, ensure_uv_binary, split_local_version,
    uv_binary_name, uv_cache_source, uv_command_env, uv_http_timeout_secs, uv_pip_check_args,
    uv_pip_freeze_args, uv_pip_install_base, uv_venv_args, violation_subject, violations_requiring,
};

pub const DEFAULT_LOCAL_PORT: u16 = 11_435;
pub const DEFAULT_LOCAL_HOST: &str = "127.0.0.1";

/// The variable that opts a machine out of rocm-cli choosing its runtime's torch.
pub const TORCH_ALIGNMENT_DISABLED_ENV: &str = "ROCM_CLI_DISABLE_TORCH_ALIGNMENT";

/// Whether the user has opted out of rocm-cli choosing this runtime's torch.
///
/// Presence is the signal, so any value — including the empty string — disables the
/// alignment; that keeps `ROCM_CLI_DISABLE_TORCH_ALIGNMENT=` from reading as "off"
/// to one side and "on" to the other.
///
/// The CLI and the vLLM engine both consult this: the engine cannot call into the
/// binary that owns the alignment, and a duplicated read is a contract that drifts.
/// If the two ever disagreed, a runtime the CLI deliberately left alone would be
/// rewritten by the engine on the very next `rocm engines install vllm` — the fight
/// the opt-out exists to end.
///
/// This suppresses the correction, not the diagnosis. The runtime is still asked
/// what it can do, the dependency check still runs, and a runtime that opens no
/// device or cannot run a kernel on one is still reported as such.
pub fn torch_alignment_disabled() -> bool {
    std::env::var_os(TORCH_ALIGNMENT_DISABLED_ENV).is_some()
}

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

/// Attempts a download makes before giving up. The first attempt is not a
/// retry, so this is one initial try plus two retries.
pub const DOWNLOAD_MAX_ATTEMPTS: u32 = 3;

/// Chunk size for the streaming copy. Fixed, so peak memory is independent of
/// the artifact size — the whole point of streaming a multi-gigabyte tarball.
const DOWNLOAD_CHUNK_BYTES: usize = 64 * 1024;

/// Exponential backoff between download attempts.
///
/// A copy of the shape used by the dashboard's reconnect loop rather than a
/// shared dependency: `rocm-core` sits below the dash crates, so it cannot
/// import theirs.
#[derive(Debug, Clone, Copy)]
pub struct Backoff {
    current: Duration,
    max: Duration,
    factor: u32,
}

impl Default for Backoff {
    fn default() -> Self {
        Self::new(Duration::from_millis(500), Duration::from_secs(8), 2)
    }
}

impl Backoff {
    pub const fn new(initial: Duration, max: Duration, factor: u32) -> Self {
        Self {
            current: initial,
            max,
            factor,
        }
    }

    /// The delay to wait before the next attempt, then grow toward `max`.
    pub fn next_delay(&mut self) -> Duration {
        let delay = self.current;
        self.current = self.current.saturating_mul(self.factor).min(self.max);
        delay
    }
}

/// A download of a single artifact to a single path.
#[derive(Debug, Clone, Copy)]
pub struct DownloadRequest<'a> {
    pub url: &'a str,
    pub destination: &'a Path,
    pub timeout: Duration,
    pub headers: &'a [(&'a str, &'a str)],
    /// Refuse a body larger than this, so an unexpected response cannot fill
    /// the disk. `None` accepts whatever the server sends.
    pub max_bytes: Option<u64>,
    /// Size the caller already knows from a manifest. Checked in addition to
    /// the server's own `Content-Length`.
    pub expected_len: Option<u64>,
    pub expected_sha256: Option<&'a str>,
}

impl<'a> DownloadRequest<'a> {
    pub const fn new(url: &'a str, destination: &'a Path, timeout: Duration) -> Self {
        Self {
            url,
            destination,
            timeout,
            headers: &[],
            max_bytes: None,
            expected_len: None,
            expected_sha256: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DownloadOutcome {
    pub bytes_written: u64,
    pub sha256: String,
}

/// Stream `request.url` to `request.destination`, retrying transient failures
/// and resuming where the server allows it.
///
/// Bytes go through a fixed [`DOWNLOAD_CHUNK_BYTES`] buffer into a sibling
/// `.part` file and are hashed on the way past, so peak memory does not scale
/// with the artifact — a multi-gigabyte SDK tarball costs the same as a small
/// one. The `.part` file is a sibling of the destination so the final rename
/// stays within one filesystem and is atomic: a caller that finds the
/// destination present knows it holds a complete download, and an interrupted
/// run never leaves a truncated file that a later run would treat as cached.
///
/// # Integrity
///
/// With `expected_sha256` the content is authenticated. Without it the only
/// integrity signal is the byte count, cross-checked against `Content-Length`
/// and `expected_len`. That catches truncation and interrupted transfers, which
/// is what this guards against, but `Content-Length` is unauthenticated and a
/// matching length proves nothing about the bytes — do not read a successful
/// return as "the artifact is genuine" unless a digest was supplied.
pub fn download_file_streaming(request: &DownloadRequest<'_>) -> Result<DownloadOutcome> {
    download_file_streaming_with_progress(request, &mut |_written, _total| {})
}

/// As [`download_file_streaming`], but reports progress via `on_progress`.
///
/// `on_progress` is called with the cumulative bytes written and, when
/// known, the total size — once before the transfer starts and once after
/// every chunk is written to disk. The byte count is monotonically
/// non-decreasing across the whole call, including across retries: an
/// attempt that restarts from scratch (the server ignored `Range`, or
/// resumed at the wrong offset and had its partial file discarded) counts
/// its own bytes from 0 internally, but the byte count `on_progress` sees
/// never drops below the highest value already reported by an earlier
/// attempt. The total is not clamped the same way and is passed through as
/// reported by the current attempt, so it can go from `None` to `Some` (or
/// back) mid-transfer if a retry's response differs on `Content-Length`.
/// Note also that for the whole duration of a from-scratch restart, the
/// byte count holds flat at the prior high-water mark until the new attempt
/// catches back up — a caller driving a static progress line should pair
/// this with an animated indicator (as the `rocm` CLI's spinner does) so a
/// long restart doesn't look hung. Callers that don't need progress should
/// use [`download_file_streaming`] instead.
pub fn download_file_streaming_with_progress(
    request: &DownloadRequest<'_>,
    on_progress: &mut dyn FnMut(u64, Option<u64>),
) -> Result<DownloadOutcome> {
    if let Some(parent) = request.destination.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let partial_path = partial_download_path(request.destination);
    // Resume is scoped to retries within this call. A `.part` file found at
    // entry is debris from an earlier run — process ids get reused, so its
    // contents cannot be attributed to this URL and appending to it would
    // silently produce a corrupt artifact.
    let _ = fs::remove_file(&partial_path);
    let mut backoff = Backoff::default();
    let mut attempt = 1;
    // `download_attempt` reports whatever it has on disk for *this* attempt,
    // which resets to 0 on a from-scratch restart even though earlier
    // attempts already progressed further. Clamp to a high-water mark here
    // so every caller — not just ones that happen to add their own UI-side
    // clamp — sees a byte count that never goes backwards.
    let mut high_water = 0_u64;
    let mut monotonic_progress = move |written: u64, total: Option<u64>| {
        high_water = high_water.max(written);
        on_progress(high_water, total);
    };
    let outcome = loop {
        match download_attempt(request, &partial_path, &mut monotonic_progress) {
            Ok(outcome) => break outcome,
            Err(error) => {
                let retryable = error.retryable && attempt < DOWNLOAD_MAX_ATTEMPTS;
                if !retryable {
                    let _ = fs::remove_file(&partial_path);
                    return Err(error.error);
                }
                thread::sleep(backoff.next_delay());
                attempt += 1;
            }
        }
    };
    fs::rename(&partial_path, request.destination).map_err(|error| {
        let _ = fs::remove_file(&partial_path);
        anyhow::Error::new(error).context(format!(
            "failed to move the completed download into {}",
            request.destination.display()
        ))
    })?;
    Ok(outcome)
}

/// Where the in-progress bytes live. Keyed by process id so two processes
/// downloading the same destination cannot append into each other's file.
fn partial_download_path(destination: &Path) -> PathBuf {
    let mut name = destination.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".part-{}", std::process::id()));
    destination.with_file_name(name)
}

/// A failed attempt, and whether trying again could plausibly help.
struct DownloadAttemptError {
    error: anyhow::Error,
    retryable: bool,
}

const fn transient(error: anyhow::Error) -> DownloadAttemptError {
    DownloadAttemptError {
        error,
        retryable: true,
    }
}

const fn permanent(error: anyhow::Error) -> DownloadAttemptError {
    DownloadAttemptError {
        error,
        retryable: false,
    }
}

/// Whether a status is worth another attempt. Server-side and rate-limit
/// responses can succeed later; the rest of `4xx` means the request itself is
/// wrong, and repeating it just wastes the user's time.
const fn status_is_retryable(status: u16) -> bool {
    status == 408 || status == 429 || status >= 500
}

/// A single attempt at the transfer. `written` — and so what this reports
/// through `on_progress` — reflects only what this attempt itself has put on
/// disk: a confirmed `206` continuation starts counting from the resumed
/// offset, but a restart (the server ignored `Range`, or resumed at the
/// wrong offset and had its partial file discarded) truncates the file and
/// starts counting from 0 again, even if a previous attempt already reported
/// further along. That's fine — [`download_file_streaming_with_progress`]
/// wraps `on_progress` with a high-water mark so callers never observe the
/// drop; this function does not need to care.
fn download_attempt(
    request: &DownloadRequest<'_>,
    partial_path: &Path,
    on_progress: &mut dyn FnMut(u64, Option<u64>),
) -> Result<DownloadOutcome, DownloadAttemptError> {
    // Resume from whatever a previous attempt already wrote. A missing file is
    // simply a fresh start.
    let resume_from = fs::metadata(partial_path).map_or(0, |meta| meta.len());
    let mut call = ureq::get(request.url).timeout(request.timeout);
    for (name, value) in request.headers {
        call = call.set(name, value);
    }
    if resume_from > 0 {
        call = call.set("Range", &format!("bytes={resume_from}-"));
    }
    let response = match call.call() {
        Ok(response) => response,
        Err(ureq::Error::Status(status, _)) => {
            let error = anyhow::anyhow!("HTTP {status} while downloading {url}", url = request.url);
            return Err(if status_is_retryable(status) {
                transient(error)
            } else {
                permanent(error)
            });
        }
        // Transport failures are exactly the interruptions worth retrying.
        Err(error) => {
            return Err(transient(
                anyhow::Error::new(error).context(format!("failed to download {}", request.url)),
            ));
        }
    };
    let status = response.status();
    if status != 200 && status != 206 {
        return Err(permanent(anyhow::anyhow!(
            "HTTP {status} while downloading {url}",
            url = request.url
        )));
    }
    // A `206` that does not confirm continuing from `resume_from` cannot be
    // treated as a fresh full download either: its `Content-Length` is the
    // length of whatever slice the server chose to send, not the whole
    // artifact, so accepting it as complete would silently rename a truncated
    // body into place. Discard the partial file and restart clean with a
    // plain `GET` instead of reinterpreting the mismatched slice.
    if status == 206 && resume_from > 0 && content_range_start(&response) != Some(resume_from) {
        let _ = fs::remove_file(partial_path);
        return Err(transient(anyhow::anyhow!(
            "{url} resumed from an unexpected byte offset; restarting the download",
            url = request.url
        )));
    }
    // Only append when the server confirmed it is continuing from exactly where
    // we stopped. A server that ignores `Range` answers 200 with the whole body,
    // which restarts cleanly too.
    let resuming = status == 206 && resume_from > 0;
    let remaining_len = header_u64(&response, "Content-Length");
    let total_len =
        remaining_len.map(|len| len.saturating_add(if resuming { resume_from } else { 0 }));
    // Fall back to the caller-supplied expected length when the server omits
    // `Content-Length`, so progress reporting doesn't lose a total that's
    // already known and already used for the preflight checks below.
    let reported_total = total_len.or(request.expected_len);
    if let Some(total) = reported_total {
        if let Some(max_bytes) = request.max_bytes
            && total > max_bytes
        {
            return Err(permanent(anyhow::anyhow!(
                "{url} is {total} bytes, over the approved limit of {max_bytes}",
                url = request.url
            )));
        }
        // Free-space preflight, before a single byte is read: an exact size
        // means we can refuse upfront instead of failing partway through. Only
        // the bytes still to come need room; a resumed prefix already has it.
        let still_needed = total.saturating_sub(if resuming { resume_from } else { 0 });
        if let Err(error) = disk_space::ensure_space_for(
            &format!("download {}", request.url),
            request.destination,
            disk_space::with_margin(still_needed),
        ) {
            return Err(permanent(error));
        }
    }

    let mut hasher = Sha256::new();
    let mut written = 0_u64;
    let mut file = if resuming {
        // Re-hash the prefix so the digest covers the whole artifact, not just
        // the bytes this attempt happened to fetch.
        let mut existing =
            fs::File::open(partial_path).map_err(|error| permanent(anyhow::Error::new(error)))?;
        written = std::io::copy(&mut existing, &mut hasher)
            .map_err(|error| permanent(anyhow::Error::new(error)))?;
        fs::OpenOptions::new()
            .append(true)
            .open(partial_path)
            .map_err(|error| permanent(anyhow::Error::new(error)))?
    } else {
        fs::File::create(partial_path).map_err(|error| {
            permanent(
                anyhow::Error::new(error)
                    .context(format!("failed to create {}", partial_path.display())),
            )
        })?
    };

    on_progress(written, reported_total);

    let mut reader = response.into_reader();
    let mut buffer = vec![0_u8; DOWNLOAD_CHUNK_BYTES];
    loop {
        let read = match reader.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => read,
            // A mid-body failure keeps the partial file: the next attempt
            // resumes from it. A server that drops the connection early and one
            // that closes cleanly after a short body are the same problem to
            // the user, so report the shortfall either way rather than a bare
            // transport error.
            Err(error) => {
                let reason = reported_total.map_or_else(
                    || format!("failed while downloading {}", request.url),
                    |expected| {
                        format!(
                            "incomplete download of {}: expected {expected} bytes, got {written}",
                            request.url
                        )
                    },
                );
                return Err(transient(anyhow::Error::new(error).context(reason)));
            }
        };
        written = written.saturating_add(read as u64);
        if let Some(max_bytes) = request.max_bytes
            && written > max_bytes
        {
            return Err(permanent(anyhow::anyhow!(
                "{url} exceeded the approved limit of {max_bytes} bytes",
                url = request.url
            )));
        }
        hasher.update(&buffer[..read]);
        if let Err(error) = file.write_all(&buffer[..read]) {
            return Err(permanent(disk_space::map_write_error(error, partial_path)));
        }
        on_progress(written, reported_total);
    }
    if let Err(error) = file.sync_all() {
        return Err(permanent(disk_space::map_write_error(error, partial_path)));
    }
    drop(file);

    // Two independent length contracts, checked separately because they fail
    // for different reasons and deserve different handling.
    //
    // The server's own `Content-Length`: a shortfall means the transfer was cut
    // short, so the bytes on disk are good as far as they go and the next
    // attempt resumes from them.
    if let Some(expected) = total_len
        && written != expected
    {
        return Err(transient(anyhow::anyhow!(
            "incomplete download of {url}: expected {expected} bytes, got {written}",
            url = request.url
        )));
    }
    // A size the caller knew in advance: the server delivered a complete body
    // that is not the artifact the manifest describes. Refetching returns the
    // same wrong thing, so do not spend the remaining attempts on it.
    if let Some(expected) = request.expected_len
        && written != expected
    {
        return Err(permanent(anyhow::anyhow!(
            "{url} is {written} bytes, but {expected} were expected",
            url = request.url
        )));
    }
    let sha256 = format!("{:x}", hasher.finalize());
    if let Some(expected) = request.expected_sha256
        && !sha256.eq_ignore_ascii_case(expected)
    {
        // The bytes are wrong, not merely incomplete; resuming would append to
        // a corrupt prefix, so discard it and fail.
        let _ = fs::remove_file(partial_path);
        return Err(permanent(anyhow::anyhow!(
            "SHA-256 mismatch for {url}: expected {expected}, got {sha256}",
            url = request.url
        )));
    }
    Ok(DownloadOutcome {
        bytes_written: written,
        sha256,
    })
}

fn header_u64(response: &ureq::Response, name: &str) -> Option<u64> {
    response
        .header(name)
        .and_then(|value| value.trim().parse::<u64>().ok())
}

/// First byte offset from a `Content-Range: bytes <start>-<end>/<total>` header.
fn content_range_start(response: &ureq::Response) -> Option<u64> {
    let value = response.header("Content-Range")?;
    let range = value.trim().strip_prefix("bytes")?.trim_start();
    let start = range.split('-').next()?.trim();
    start.parse::<u64>().ok()
}

pub fn download_file_to_path(url: &str, destination: &Path, timeout: Duration) -> Result<()> {
    download_file_streaming(&DownloadRequest::new(url, destination, timeout))?;
    Ok(())
}

/// As [`download_file_to_path`], but reports progress via `on_progress`. See
/// [`download_file_streaming_with_progress`] for callback semantics.
pub fn download_file_to_path_with_progress(
    url: &str,
    destination: &Path,
    timeout: Duration,
    on_progress: &mut dyn FnMut(u64, Option<u64>),
) -> Result<()> {
    download_file_streaming_with_progress(
        &DownloadRequest::new(url, destination, timeout),
        on_progress,
    )?;
    Ok(())
}

pub fn http_get_text(endpoint_url: &str, path: &str, timeout: Duration) -> Result<String> {
    http_get_text_with_auth(endpoint_url, path, None, timeout)
}

/// As [`http_get_text`], but authenticated.
///
/// Sends `Authorization: Bearer <key>` when `endpoint_api_key` is `Some`. Used to
/// probe endpoints that `rocm serve` has protected with an API key; `None`
/// preserves the unauthenticated behavior for loopback endpoints.
pub fn http_get_text_with_auth(
    endpoint_url: &str,
    path: &str,
    endpoint_api_key: Option<&str>,
    timeout: Duration,
) -> Result<String> {
    let deadline = Instant::now() + timeout;
    let (host, port) = parse_http_endpoint(endpoint_url)
        .with_context(|| format!("unsupported endpoint URL `{endpoint_url}`"))?;
    let mut stream = connect_tcp_stream(&host, port, timeout)?;
    let host_header = format_host_port(&host, port);
    let auth_header = match endpoint_api_key {
        Some(key) => format!("Authorization: Bearer {key}\r\n"),
        None => String::new(),
    };
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {host_header}\r\nAccept: application/json\r\n{auth_header}Connection: close\r\n\r\n"
    );
    write_all_tcp_stream(&mut stream, request.as_bytes())
        .with_context(|| format!("failed to write HTTP GET {path}"))?;
    let response = read_http_response_bounded(&mut stream, deadline)
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

/// POST a JSON body and return the response status line plus body.
///
/// The POST sibling of [`http_get_text_with_auth`]. Unlike the GET helper this
/// does not treat a non-200 as an error: callers that probe an endpoint need to
/// tell "the server answered, with a refusal" apart from "the server never
/// answered", and only the former proves the request path is alive.
pub fn http_post_json_with_auth(
    endpoint_url: &str,
    path: &str,
    body: &serde_json::Value,
    endpoint_api_key: Option<&str>,
    timeout: Duration,
) -> Result<HttpResponseParts> {
    let deadline = Instant::now() + timeout;
    let (host, port) = parse_http_endpoint(endpoint_url)
        .with_context(|| format!("unsupported endpoint URL `{endpoint_url}`"))?;
    let mut stream = connect_tcp_stream(&host, port, timeout)?;
    let host_header = format_host_port(&host, port);
    let auth_header = match endpoint_api_key {
        Some(key) => format!("Authorization: Bearer {key}\r\n"),
        None => String::new(),
    };
    let payload = serde_json::to_string(body).context("failed to serialize HTTP JSON body")?;
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {host_header}\r\nContent-Type: application/json\r\nAccept: application/json\r\nContent-Length: {}\r\n{auth_header}Connection: close\r\n\r\n{payload}",
        payload.len()
    );
    write_all_tcp_stream(&mut stream, request.as_bytes())
        .with_context(|| format!("failed to write HTTP POST {path}"))?;
    let response = read_http_response_bounded(&mut stream, deadline)
        .with_context(|| format!("failed to read HTTP POST {path}"))?;
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .context("HTTP response was missing a body")?;
    let status_line = headers.lines().next().unwrap_or_default();
    let status = http_status_code(status_line)
        .with_context(|| format!("unparsable HTTP status line `{status_line}`"))?;
    Ok(HttpResponseParts {
        status,
        body: body.to_owned(),
    })
}

/// Budget for the single-token chat request that proves a service can serve.
///
/// Generously longer than a model-listing timeout: the probe is a real inference
/// request, and a first request against a freshly loaded model pays for prompt
/// processing before it answers.
pub const INFERENCE_PROBE_TIMEOUT: Duration = Duration::from_secs(8);

/// Engine state-file key recording when inference was first confirmed.
///
/// Written by the engine healthchecks and adopted by
/// [`ManagedServiceRecord::refresh_from_engine_state`], so the CLI side does not
/// re-probe what an engine already verified.
pub const INFERENCE_VERIFIED_STATE_KEY: &str = "inference_verified_at_unix_ms";

/// Engine state-file key recording the last inference probe attempt.
pub const INFERENCE_PROBE_ATTEMPTED_STATE_KEY: &str = "inference_probe_attempted_at_unix_ms";

/// Minimum gap between inference probes against a service that is still loading.
///
/// Only a *successful* probe latches, so without this a warming model would be
/// re-probed by every readiness poll — and each attempt costs up to
/// [`INFERENCE_PROBE_TIMEOUT`], which is the whole poll's latency. The
/// supervisor ticks every few seconds and `services list` sits in front of a
/// user, so the unthrottled cost lands exactly where it is most visible. The
/// price of throttling is that readiness can be noticed up to this late.
pub const INFERENCE_PROBE_RETRY_INTERVAL: Duration = Duration::from_secs(15);

/// Merge `patch`'s top-level keys into the JSON object stored at `path`.
///
/// Creates the file (and its parent) when absent, and replaces a non-object
/// document rather than failing — the caller is recording a fact about a live
/// service, not validating an existing file.
pub fn merge_json_state_file(path: &Path, patch: &serde_json::Value) -> Result<()> {
    let mut value = fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    if !value.is_object() {
        value = serde_json::json!({});
    }
    let object = value.as_object_mut().expect("object checked above");
    if let Some(patch) = patch.as_object() {
        for (key, patch_value) in patch {
            object.insert(key.clone(), patch_value.clone());
        }
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(
        path,
        serde_json::to_vec_pretty(&value).context("failed to serialize service state")?,
    )
    .with_context(|| format!("failed to write {}", path.display()))
}

/// Whether inference has been confirmed for a service, from its engine state
/// file — probing at most once, and at most once per
/// [`INFERENCE_PROBE_RETRY_INTERVAL`] while it is still loading.
///
/// Shared by the engine adapters so the latch and backoff bookkeeping has one
/// implementation: the engines differ in how they decide a model is *listed*,
/// but not in what confirming inference means.
///
/// The attempt is recorded before the probe runs, so a caller killed mid-probe
/// still leaves the throttle in place instead of freeing the next poll to spend
/// another full timeout.
pub fn engine_state_inference_verified(
    state_path: &Path,
    state: Option<&serde_json::Value>,
    endpoint_url: &str,
    model_ref: &str,
    endpoint_api_key: Option<&str>,
) -> bool {
    let state_u64 = |key: &str| {
        state
            .and_then(|value| value.get(key))
            .and_then(serde_json::Value::as_u64)
    };
    if state_u64(INFERENCE_VERIFIED_STATE_KEY).is_some() {
        return true;
    }
    if model_ref.trim().is_empty() {
        return false;
    }
    let now = unix_time_millis() as u64;
    if let Some(attempted_at) = state_u64(INFERENCE_PROBE_ATTEMPTED_STATE_KEY)
        && now.saturating_sub(attempted_at) < INFERENCE_PROBE_RETRY_INTERVAL.as_millis() as u64
    {
        return false;
    }
    let _ = merge_json_state_file(
        state_path,
        &serde_json::json!({ INFERENCE_PROBE_ATTEMPTED_STATE_KEY: now }),
    );
    if !openai_chat_completion_probe(
        endpoint_url,
        model_ref,
        endpoint_api_key,
        INFERENCE_PROBE_TIMEOUT,
    )
    .unwrap_or(false)
    {
        return false;
    }
    let _ = merge_json_state_file(
        state_path,
        &serde_json::json!({ INFERENCE_VERIFIED_STATE_KEY: unix_time_millis() as u64 }),
    );
    true
}

/// The parts of an HTTP response a probe needs: the status code and the body.
#[derive(Debug, Clone)]
pub struct HttpResponseParts {
    pub status: u16,
    pub body: String,
}

fn http_status_code(status_line: &str) -> Option<u16> {
    status_line.split_whitespace().nth(1)?.parse().ok()
}

/// Ask the endpoint for a single token and report whether it answered.
///
/// This is the readiness signal that `/v1/models` cannot give: an engine lists a
/// model as soon as it accepts the name, which can be minutes before the weights
/// are resident and the first chat request stops hanging.
///
/// "Answered" means a complete HTTP response with a status below 500, not a
/// successful generation. A `4xx` still proves the inference path is up and the
/// model is loaded — the request was understood and refused on its merits — while
/// the failure this guards against is a hang, a dropped connection, or the `5xx`
/// an engine returns while it is still warming up. Insisting on `200` with
/// non-empty content would also wrongly fail a reasoning model, which can spend
/// its whole (tiny) token budget before emitting any content.
///
/// The rule does mean a `404` reads as serving. That is harmless for the engines
/// shipped today — both implement `/v1/chat/completions`, and a wrong key fails
/// the model listing that gates this call — but an engine that does not expose an
/// OpenAI-shaped chat route would need a different signal rather than this one.
pub fn openai_chat_completion_probe(
    endpoint_url: &str,
    model_ref: &str,
    endpoint_api_key: Option<&str>,
    timeout: Duration,
) -> Result<bool> {
    let status = openai_chat_completion_status(endpoint_url, model_ref, endpoint_api_key, timeout)?;
    Ok(status < 500)
}

/// Send the smallest possible chat request and return the HTTP status.
///
/// Callers pick their own bar. Readiness ([`openai_chat_completion_probe`]) only
/// needs to know the inference path answers at all, while a post-load smoke test
/// wants a real `200` — there, a `4xx` means the model that came up is not the
/// one that was asked for, which is a failure worth surfacing rather than
/// tolerating.
pub fn openai_chat_completion_status(
    endpoint_url: &str,
    model_ref: &str,
    endpoint_api_key: Option<&str>,
    timeout: Duration,
) -> Result<u16> {
    let body = serde_json::json!({
        "model": model_ref,
        "messages": [{"role": "user", "content": "Say ok."}],
        "max_tokens": 2,
        "stream": false,
    });
    let response = http_post_json_with_auth(
        endpoint_url,
        "/v1/chat/completions",
        &body,
        endpoint_api_key,
        timeout,
    )?;
    Ok(response.status)
}

pub fn openai_models_endpoint_has_model(
    endpoint_url: &str,
    expected_model: Option<&str>,
    endpoint_api_key: Option<&str>,
    timeout: Duration,
) -> Result<bool> {
    let body = http_get_text_with_auth(endpoint_url, "/v1/models", endpoint_api_key, timeout)?;
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
    endpoint_api_key: Option<&str>,
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
    openai_models_endpoint_has_model(&record.endpoint_url, expected, endpoint_api_key, timeout)
}

/// How far along a managed service's endpoint is.
///
/// The middle state is the one that matters: an engine lists a model within
/// seconds of accepting its name, while the weights can take minutes to become
/// usable. Callers must not treat `Listing` as ready — nor as dead, since the
/// service is coming up normally and restarting it would start the wait over.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum EndpointReadiness {
    /// Not answering at all: wrong port, process gone, or nothing bound yet.
    Unreachable,
    /// Answering and advertising the model, but inference has not come back.
    Listing,
    /// A real inference request has succeeded.
    Serving,
}

/// The result of a readiness check, plus whether it left the record dirty.
#[derive(Debug, Clone, Copy)]
pub struct EndpointReadinessOutcome {
    pub readiness: EndpointReadiness,
    /// The check updated the record's probe bookkeeping. Persist it with
    /// [`ManagedServiceRecord::write`] — the throttle in
    /// [`managed_service_endpoint_readiness`] only works if the attempt survives
    /// the process, since each CLI invocation starts fresh.
    pub record_changed: bool,
}

/// How far along the service's endpoint is, probing inference at most once.
///
/// Stronger than [`managed_service_endpoint_model_ready`], which only asks
/// whether the endpoint lists the model. A service reaches [`Serving`] once a
/// real inference request has come back, and that verdict is **latched** into
/// `record.inference_verified_at_unix_ms`: readiness is polled repeatedly (by
/// `services list`, the dash, and the supervisor), and re-probing on every poll
/// would queue a generation request behind the user's own traffic. The trade-off
/// is that a service which degrades after start still reports ready — the same
/// as before this check existed.
///
/// A *failed* probe cannot latch, so those are throttled instead: a still-loading
/// service is re-probed at most once per [`INFERENCE_PROBE_RETRY_INTERVAL`],
/// which keeps a warming model from costing every caller a full
/// `probe_timeout`.
///
/// Mutates `record` when it probes; persist it when `record_changed` is set.
///
/// [`Serving`]: EndpointReadiness::Serving
pub fn managed_service_endpoint_readiness(
    record: &mut ManagedServiceRecord,
    endpoint_api_key: Option<&str>,
    listing_timeout: Duration,
    probe_timeout: Duration,
) -> EndpointReadinessOutcome {
    let outcome = |readiness, record_changed| EndpointReadinessOutcome {
        readiness,
        record_changed,
    };
    let listed = managed_service_endpoint_model_ready(record, endpoint_api_key, listing_timeout)
        .unwrap_or(false);
    if !listed {
        return outcome(EndpointReadiness::Unreachable, false);
    }
    if record.inference_verified_at_unix_ms.is_some() {
        return outcome(EndpointReadiness::Serving, false);
    }
    let now = unix_time_millis() as u64;
    if let Some(attempted_at) = record.inference_probe_attempted_at_unix_ms
        && now.saturating_sub(attempted_at) < INFERENCE_PROBE_RETRY_INTERVAL.as_millis() as u64
    {
        return outcome(EndpointReadiness::Listing, false);
    }
    let model_ref = if record.canonical_model_id.trim().is_empty() {
        record.model_ref.as_str()
    } else {
        record.canonical_model_id.as_str()
    };
    record.inference_probe_attempted_at_unix_ms = Some(now);
    if !openai_chat_completion_probe(
        &record.endpoint_url,
        model_ref,
        endpoint_api_key,
        probe_timeout,
    )
    .unwrap_or(false)
    {
        return outcome(EndpointReadiness::Listing, true);
    }
    record.inference_verified_at_unix_ms = Some(unix_time_millis() as u64);
    outcome(EndpointReadiness::Serving, true)
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
    // Bound the connect as well as the reads: a probe against an engine that is
    // pinned solid must fail within the caller's timeout, not sit in the OS
    // default SYN retry window.
    let stream = TcpStream::connect_timeout(&addr, timeout)
        .with_context(|| format!("failed to connect to {host}:{port}"))?;
    stream.set_read_timeout(Some(timeout)).ok();
    stream.set_write_timeout(Some(timeout)).ok();
    Ok(stream)
}

pub fn write_all_tcp_stream(stream: &mut TcpStream, bytes: &[u8]) -> Result<()> {
    stream
        .write_all(bytes)
        .context("failed to write to TCP stream")
}

/// Read one HTTP response, bounded by a wall-clock deadline.
///
/// Two problems with reading to end-of-stream instead. A response is only
/// complete at EOF if the peer actually closes: `Connection: close` asks for
/// that, but nothing obliges a server or an intervening proxy to honor it, so a
/// service that writes a perfectly good response and holds the socket open would
/// stall until the read timeout and have its answer thrown away. And a socket
/// read timeout bounds each `read` call, not the sequence of them, so a
/// slow-drip responder could stretch the total wait to an arbitrary multiple of
/// what the caller asked for. This returns as soon as the response is complete by
/// its own framing, and never runs past `deadline` in total.
pub fn read_http_response_bounded(stream: &mut TcpStream, deadline: Instant) -> Result<String> {
    let mut response = Vec::new();
    let mut chunk = [0_u8; 4096];
    while !http_response_is_complete(&response) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            bail!("timed out reading HTTP response");
        }
        stream.set_read_timeout(Some(remaining)).ok();
        match stream.read(&mut chunk) {
            // Peer closed: whatever arrived is the whole response.
            Ok(0) => break,
            Ok(read) => response.extend_from_slice(&chunk[..read]),
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                bail!("timed out reading HTTP response");
            }
            // A signal delivered to this thread aborts the read with `EINTR`.
            // `SA_RESTART` does not save us: Linux never restarts a socket read
            // that has a receive timeout set, and this loop sets one on every
            // pass (see signal(7), "Interruption of system calls"). Any handler
            // in the process is enough — `crossterm`'s `SIGWINCH` hook is linked
            // into the CLI. Nothing is wrong with the connection, so read again;
            // `deadline` still bounds the total wait.
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error).context("failed to read TCP stream"),
        }
    }
    Ok(String::from_utf8_lossy(&response).into_owned())
}

/// Whether the bytes so far are a complete HTTP response by their own framing.
///
/// `false` for a response that declares neither a length nor chunked encoding —
/// those are delimited by the connection closing, so the caller must keep reading
/// until EOF.
fn http_response_is_complete(response: &[u8]) -> bool {
    // Headers are ASCII by the HTTP spec, so it is safe to lossy-decode just
    // that slice to parse them. The body length check below stays on raw
    // bytes: lossy-decoding a body that ends mid multi-byte UTF-8 sequence
    // replaces the truncated tail with a 3-byte U+FFFD, which can inflate a
    // partial body's *decoded* length past the declared Content-Length and
    // report completeness one read early.
    let Some(header_end) = response.windows(4).position(|w| w == b"\r\n\r\n") else {
        return false;
    };
    let headers = String::from_utf8_lossy(&response[..header_end]);
    let body = &response[header_end + 4..];
    let header_value = |name: &str| {
        headers.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.trim()
                .eq_ignore_ascii_case(name)
                .then(|| value.trim().to_owned())
        })
    };
    if let Some(length) =
        header_value("Content-Length").and_then(|value| value.parse::<usize>().ok())
    {
        return body.len() >= length;
    }
    if header_value("Transfer-Encoding")
        .is_some_and(|value| value.to_ascii_lowercase().contains("chunked"))
    {
        return body.ends_with(b"0\r\n\r\n");
    }
    false
}

/// Outcome of a detached Windows spawn that was briefly watched for an early exit.
///
/// The observation runs while the process handle returned by `CreateProcessW` is
/// still open, so a reported exit code is always the exit code of the process that
/// was just spawned. Re-opening the process by PID after the fact could not offer
/// that guarantee — Windows recycles PIDs, so by then the PID may name an
/// unrelated process.
#[cfg(windows)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetachedSpawn {
    /// PID of the spawned process.
    pub pid: u32,
    /// `Some(code)` if the process had already exited when the observation window
    /// elapsed, `None` if it was still running. `None` is also reported if the
    /// exit code could not be read, which degrades to the unwatched behaviour
    /// rather than inventing a failure.
    pub early_exit_code: Option<u32>,
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
        None,
    )
    .map(|spawn| spawn.pid)
}

/// As [`spawn_detached_no_inherit`], but watches the new process for up to
/// `settle` before returning, so the caller can tell "started" apart from
/// "started and died immediately".
///
/// Creation flags and handle inheritance are identical to
/// [`spawn_detached_no_inherit`]: the child stays fully detached and outlives this
/// process. The wait is bounded and only delays the caller by `settle`; it is not
/// a join.
///
/// This exists because the detached spawn primitives hand back a bare PID rather
/// than a `std::process::Child`, so a caller has no equivalent of `try_wait()` to
/// notice a child that failed during startup.
#[cfg(windows)]
pub fn spawn_detached_no_inherit_watching_startup(
    program: &Path,
    args: &[String],
    env_overrides: &[(&str, &Path)],
    settle: Duration,
) -> Result<DetachedSpawn> {
    spawn_windows_no_inherit(
        program,
        args,
        env_overrides,
        DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT,
        false,
        None,
        Some(settle),
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
        None,
    )
    .map(|spawn| spawn.pid)
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
        None,
    );
    unsafe {
        CloseHandle(stdout_handle);
        CloseHandle(stderr_handle);
    }
    result.map(|spawn| spawn.pid)
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

/// Send `signal` to `pid`, optionally extending to its transitive children.
///
/// Delivery to a process that has already exited (`ESRCH`) counts as success:
/// the goal — that process no longer running — is already met. Returns `false`
/// only when a signal could not be delivered for another reason (for example
/// `EPERM`). Used by the verified-termination logic in [`proc_lifecycle`].
#[cfg(not(windows))]
#[allow(unsafe_code)] // libc FFI
pub(crate) fn signal_process_scope(pid: u32, signal: i32, include_tree: bool) -> bool {
    let targets = if include_tree {
        collect_process_tree(pid)
    } else {
        vec![pid]
    };
    let mut delivered = true;
    for target in targets {
        let status = unsafe { libc::kill(target.cast_signed(), signal) };
        if status != 0 && std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH) {
            delivered = false;
        }
    }
    delivered
}

/// Snapshot `root` plus its transitive descendants as a flat PID list.
///
/// Used by [`proc_lifecycle`] to bind a tree termination to the exact processes
/// present when the stop began. On platforms without `/proc` only `root` is
/// returned.
#[cfg(not(windows))]
pub(crate) fn process_tree_pids(root: u32) -> Vec<u32> {
    collect_process_tree(root)
}

#[cfg(windows)]
pub(crate) fn process_tree_pids(root: u32) -> Vec<u32> {
    vec![root]
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

/// An advisory, cross-process exclusive lock backed by a lock file.
///
/// Wraps the standard-library file lock (`std::fs::File::lock`), so the exclusion
/// holds between *separate `rocm` processes*, not just threads: each caller opens
/// the same lock-file path and only one can hold the lock at a time. It exists to
/// serialize check-then-act sequences over shared on-disk state — the daemon
/// autostart decision and the managed-serve GPU select-then-claim — so two
/// concurrent invocations cannot both pass the same TOCTOU check.
///
/// The lock is released when the guard is dropped, and by the OS if the process
/// exits while holding it (so a crashed holder never wedges the next caller).
#[derive(Debug)]
pub struct FileLock {
    file: fs::File,
    path: PathBuf,
}

impl FileLock {
    /// Acquire an exclusive lock on `path`, creating the lock file and any
    /// missing parent directories first. Blocks until the lock is available.
    ///
    /// The lock file itself carries no data; it is a rendezvous point, so an
    /// existing file is reused (never truncated) and its contents are ignored.
    pub fn acquire(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create lock directory {}", parent.display()))?;
        }
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .with_context(|| format!("failed to open lock file {}", path.display()))?;
        file.lock()
            .with_context(|| format!("failed to acquire lock {}", path.display()))?;
        Ok(Self { file, path })
    }

    /// The lock file backing this guard.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        // Best-effort: an unlock failure only means the OS releases it slightly
        // later (at the latest when the file handle closes), never a lost lock.
        let _ = self.file.unlock();
    }
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
    settle: Option<Duration>,
) -> Result<DetachedSpawn> {
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
    // Observe the child *before* the handle is closed. Waiting on this handle is
    // race-free; re-opening the process later by PID would not be, since Windows
    // recycles PIDs and the PID could by then belong to something else.
    let early_exit_code =
        settle.and_then(|settle| unsafe { observe_early_exit(process_info.hProcess, settle) });
    unsafe {
        CloseHandle(process_info.hThread);
        CloseHandle(process_info.hProcess);
    }
    Ok(DetachedSpawn {
        pid: process_info.dwProcessId,
        early_exit_code,
    })
}

/// Wait up to `settle` for `process` to exit, reporting its exit code if it did.
///
/// Returns `None` both when the process is still running and when the exit code
/// could not be read, so a failed query degrades to "assume it is alive" rather
/// than reporting a startup failure that may not have happened.
///
/// # Safety
///
/// `process` must be a live process handle granting `SYNCHRONIZE` and
/// `PROCESS_QUERY_INFORMATION` access. The handle is not closed here.
#[cfg(windows)]
#[allow(unsafe_code)] // Win32 FFI
unsafe fn observe_early_exit(
    process: windows_sys::Win32::Foundation::HANDLE,
    settle: Duration,
) -> Option<u32> {
    use windows_sys::Win32::Foundation::WAIT_OBJECT_0;

    if unsafe { WaitForSingleObject(process, wait_timeout_millis(settle)) } != WAIT_OBJECT_0 {
        return None;
    }
    let mut exit_code: u32 = 0;
    if unsafe { GetExitCodeProcess(process, &raw mut exit_code) } == 0 {
        return None;
    }
    Some(exit_code)
}

/// Clamp a wait budget to the `u32` milliseconds `WaitForSingleObject` takes,
/// never yielding `INFINITE`.
///
/// `INFINITE` is `u32::MAX`, so a saturating conversion of a long duration would
/// silently turn a bounded startup check into one that blocks until the child
/// exits. Cap one millisecond below it instead.
#[cfg(any(windows, test))]
fn wait_timeout_millis(budget: Duration) -> u32 {
    const LONGEST_FINITE_WAIT_MS: u128 = (u32::MAX - 1) as u128;
    budget.as_millis().min(LONGEST_FINITE_WAIT_MS) as u32
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
        let paths = Self {
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
        Ok(Self::discover_from_paths(
            paths,
            data_dir_override.is_some(),
            cache_dir_override.is_some(),
        ))
    }

    fn discover_from_paths(
        mut paths: Self,
        data_dir_overridden: bool,
        cache_dir_overridden: bool,
    ) -> Self {
        if !data_dir_overridden
            && let Some(managed_root) = configured_managed_root_from_config(&paths)
        {
            paths = paths.with_managed_root(managed_root, cache_dir_overridden);
        }
        paths.normalize_for_host()
    }

    fn normalize_for_host(mut self) -> Self {
        self.config_dir = normalize_runtime_path_for_host(&self.config_dir);
        self.data_dir = normalize_runtime_path_for_host(&self.data_dir);
        self.cache_dir = normalize_runtime_path_for_host(&self.cache_dir);
        self
    }

    #[must_use]
    pub fn with_managed_root(mut self, root: impl Into<PathBuf>, keep_cache_dir: bool) -> Self {
        self.data_dir = managed_runtime_data_root(&root.into());
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

    /// Where engine virtualenvs live, honouring `ROCM_CLI_ENGINE_ENVS_ROOT`.
    ///
    /// Read-side of a write-only contract at present: `apps/rocm` exports this
    /// key into the children it spawns, and this is the only code that reads it
    /// back, so no in-tree production path reaches here today. That predates
    /// the seam below and is deliberate — the key exists for an engine process
    /// to honour. Being `pub` says nothing either way: `unreachable_pub` is
    /// switched off workspace-wide (see the root `Cargo.toml`), so nothing warns
    /// about an unused one. Delete this and the key together if the contract is
    /// dropped; `engine_envs_dir_reads_its_root_from_the_environment` is what
    /// keeps the lookup honest meanwhile.
    pub fn engine_envs_root(&self) -> PathBuf {
        self.engine_envs_root_from(env_path_override("ROCM_CLI_ENGINE_ENVS_ROOT").as_deref())
    }

    /// [`Self::engine_envs_root`] against a caller-supplied override.
    ///
    /// Lets a test drive the override without `std::env::set_var`, which is
    /// process-global and races other tests under a threaded runner. Same seam
    /// as [`discover_rocm_installs_in_layout`].
    fn engine_envs_root_from(&self, override_root: Option<&Path>) -> PathBuf {
        override_root.map_or_else(
            || self.data_dir.join("engines"),
            normalize_runtime_path_for_host,
        )
    }

    pub fn engine_envs_dir(&self, engine: &str) -> PathBuf {
        self.engine_envs_root().join(engine).join("envs")
    }

    /// [`Self::engine_envs_dir`] against a caller-supplied override.
    #[cfg(test)]
    fn engine_envs_dir_from(&self, engine: &str, override_root: Option<&Path>) -> PathBuf {
        self.engine_envs_root_from(override_root)
            .join(engine)
            .join("envs")
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

    /// Lock file serializing the managed-serve GPU select-then-claim sequence, so
    /// two concurrent `rocm serve` invocations cannot read the same free GPU and
    /// both launch on it. Held from auto-selection through the claiming service
    /// record write (see [`FileLock`]).
    pub fn managed_launch_lock_path(&self) -> PathBuf {
        self.services_dir().join("launch.lock")
    }

    /// Where `rocm remote` records the sessions it started on other machines.
    ///
    /// Kept beside [`Self::services_dir`] and following the same file-per-record
    /// shape, but deliberately separate: these describe work running on a
    /// *different* machine, and anything walking the local service registry
    /// (status rendering, the daemon's recovery supervisor) must not mistake a
    /// remote session for a local server it can supervise.
    pub fn remote_sessions_dir(&self) -> PathBuf {
        self.data_dir.join("remote-sessions")
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

    /// Lock file serializing the daemon autostart check-then-spawn, so two
    /// concurrent callers cannot both observe "not running" and each spawn a
    /// background automation daemon (see [`FileLock`]).
    pub fn automation_autostart_lock_path(&self) -> PathBuf {
        self.automations_dir().join("autostart.lock")
    }

    /// Short-lived claim written by the autostart holder right after it spawns
    /// the daemon, recording the child PID and spawn time. It bridges the gap
    /// between `spawn()` and the child publishing its runtime state: a concurrent
    /// caller that acquires the autostart lock in that window sees the claim and
    /// defers instead of spawning a duplicate daemon.
    pub fn automation_autostart_claim_path(&self) -> PathBuf {
        self.automations_dir().join("autostart.claim")
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

    /// The directive file that moves `rocm dash`'s telemetry daemon off wall time
    /// and onto a logical observation clock (see `docs/release-trust.md`).
    ///
    /// `rocm dash` reads it in every build, and nothing in rocm-cli creates it —
    /// only the E2E harness plants one, in its isolated data root. Both resolve it
    /// through this one method, so the path the harness plants and the path the
    /// dashboard reads cannot drift apart. If they did, the dashboard would
    /// silently stay on wall time, and the clock-driven scenarios would not fail
    /// outright: they would keep passing, but only by timing.
    pub fn dash_test_clock_file(&self) -> PathBuf {
        self.telemetry_state_dir().join("test-clock-offset")
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

pub fn require_nonempty(value: &str, field_name: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{field_name} must not be empty");
    }
    Ok(())
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
    /// Described in: `rocm setup status`/`reset` output, their `--help` doc
    /// comments, README.md's Setup section, docs/testing.md,
    /// docs/manual-testing.md, and the module doc comment in
    /// `crates/rocm-dash-tui/src/ui/onboarding.rs`. Nothing reads this to
    /// auto-open onboarding — grep this field, not a phrase, before touching
    /// any of those surfaces.
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
    /// Described in: `rocm setup status`/`reset` output, their `--help` doc
    /// comments, README.md's Setup section, docs/testing.md,
    /// docs/manual-testing.md, and the module doc comment in
    /// `crates/rocm-dash-tui/src/ui/onboarding.rs`. Nothing reads this to
    /// auto-open onboarding — grep this field, not a phrase, before touching
    /// any of those surfaces.
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
    // `runtime::user_runtime_dir` for the precedence. This resolver is mirrored
    // in `rocm-dash-core` so the canonical `rocm` config and a standalone
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
///
/// The tier chain itself lives in [`user_runtime_dir`], which the Lemonade
/// engine also uses to synthesize a runtime directory for its child process.
/// Only tier 2 needs the `telemetry` leaf: tiers 1 and 3 are already per-user
/// runtime directories, so the socket sits directly in them.
fn dashboard_socket_path(
    xdg_runtime_dir: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
    user: Option<String>,
    temp_dir: std::path::PathBuf,
) -> std::path::PathBuf {
    user_runtime_dir(xdg_runtime_dir, home, user, temp_dir, "telemetry", "").join("rocmdashd.sock")
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

fn deserialize_optional_chat_temperature<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<f32>::deserialize(deserializer)?;
    if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
        return Err(serde::de::Error::custom(
            "dashboard.tui.chat_temperature must be a finite value >= 0.0",
        ));
    }
    Ok(value)
}

fn deserialize_optional_chat_top_p<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<f32>::deserialize(deserializer)?;
    if value.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
        return Err(serde::de::Error::custom(
            "dashboard.tui.chat_top_p must be a finite value between 0.0 and 1.0",
        ));
    }
    Ok(value)
}

fn deserialize_optional_chat_max_tokens<'de, D>(deserializer: D) -> Result<Option<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<u32>::deserialize(deserializer)?;
    if value == Some(0) {
        return Err(serde::de::Error::custom(
            "dashboard.tui.chat_max_tokens must be greater than 0",
        ));
    }
    Ok(value)
}

/// Dashboard TUI spec. The chat endpoint URL / model / auth-header *name* are
/// plain data; the auth-header *value* (API key) is always env-only and never
/// stored here (AMD gateway invariant).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
    /// Sampling temperature applied to chat requests (parity with the
    /// `rocm chat` / `rocm serve` `--temperature` flag). `None` leaves the
    /// endpoint default untouched; a CLI flag overrides this.
    #[serde(
        default,
        deserialize_with = "deserialize_optional_chat_temperature",
        skip_serializing_if = "Option::is_none"
    )]
    pub chat_temperature: Option<f32>,
    /// Nucleus-sampling `top_p` applied to chat requests (parity with
    /// `--top-p`). `None` leaves the endpoint default untouched.
    #[serde(
        default,
        deserialize_with = "deserialize_optional_chat_top_p",
        skip_serializing_if = "Option::is_none"
    )]
    pub chat_top_p: Option<f32>,
    /// Upper bound on generated tokens for chat requests (parity with
    /// `--max-tokens`). `None` uses the built-in dashboard default.
    #[serde(
        default,
        deserialize_with = "deserialize_optional_chat_max_tokens",
        skip_serializing_if = "Option::is_none"
    )]
    pub chat_max_tokens: Option<u32>,
}

impl Default for DashboardTuiConfig {
    fn default() -> Self {
        Self {
            connect: default_dashboard_connect(),
            theme: default_dashboard_theme(),
            chat_url: None,
            chat_model: None,
            chat_auth_header: None,
            chat_temperature: None,
            chat_top_p: None,
            chat_max_tokens: None,
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

/// Number of random alphanumeric characters in a generated endpoint API key.
/// 48 chars from a 62-symbol alphabet is ~285 bits of entropy — far beyond what
/// a bearer token guarding a network endpoint needs, with no padding characters
/// that would complicate copy/paste into client configs or shell env vars.
const ENDPOINT_API_KEY_LEN: usize = 48;

/// Generate a fresh, cryptographically-random API key for a public endpoint.
///
/// The value is URL-safe alphanumeric (`[A-Za-z0-9]`) so it can be dropped
/// verbatim into an `Authorization: Bearer` header, a client config file, or an
/// environment variable without escaping.
///
/// Drawn from `rand::rng()`, a CSPRNG seeded from the operating system;
/// deliberately *not* derived from `generate_service_id` (a timestamp-based,
/// guessable identifier — unsuitable as a secret).
pub fn generate_endpoint_api_key() -> String {
    use rand::Rng;
    use rand::distr::Alphanumeric;

    rand::rng()
        .sample_iter(&Alphanumeric)
        .take(ENDPOINT_API_KEY_LEN)
        .map(char::from)
        .collect()
}

/// Return `true` if `key` contains a character that must never appear in an API
/// key used verbatim in an `Authorization: Bearer` header.
///
/// The CLI and engine adapters build those header lines by raw string
/// interpolation (`Authorization: Bearer {key}\r\n`), so an embedded CR or LF in
/// the key would inject additional header lines (HTTP header injection). A
/// control character has no legitimate place in a bearer token, so we reject the
/// whole class rather than only CR/LF. Callers apply this at input validation
/// (rejecting a supplied key) and defensively when reading the key file.
pub fn endpoint_api_key_has_forbidden_chars(key: &str) -> bool {
    key.chars().any(char::is_control)
}

/// Generate a fresh 2048-bit RSA signing keypair, returned as
/// `(PKCS#8 private-key PEM, SubjectPublicKeyInfo public-key PEM)` — the same
/// formats `openssl genpkey` / `openssl rsa -pubout` produce.
pub fn generate_rsa_signing_keypair() -> Result<(String, String)> {
    use rsa::RsaPrivateKey;
    use rsa::pkcs8::{EncodePrivateKey, EncodePublicKey, LineEnding};

    // `rsa` 0.9 is built against `rand_core` 0.6, while the workspace `rand` is
    // 0.9 (`rand_core` 0.9) — the two trait sets are not interchangeable, so an
    // rng from `rand::rng()` does not satisfy `RsaPrivateKey::new`. Use the
    // `rand_core` that `rsa` itself re-exports, which keeps the versions matched
    // no matter which one `rand` moves to. `OsRng` draws straight from the
    // operating system CSPRNG.
    let mut rng = rsa::rand_core::OsRng;
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
    /// Kernel start-time of the supervisor (launcher) process, captured at
    /// spawn. Paired with `supervisor_pid` it forms an identity that survives
    /// PID recycling, so a later stop never signals a reused PID. `None` for
    /// records written before this field existed.
    #[serde(default)]
    pub supervisor_start_ticks: Option<u64>,
    /// Kernel start-time of the engine server process, adopted from the engine
    /// state file whenever `engine_pid` is refreshed from it. The launcher and
    /// the engine server are distinct processes, so each PID carries its own
    /// identity token. `None` until the engine state records one.
    #[serde(default)]
    pub engine_start_ticks: Option<u64>,
    /// Whether this service must never come back up without an endpoint key.
    ///
    /// The bind address alone cannot answer that. A service bound to loopback is
    /// unreachable from elsewhere *until something republishes the port* — a
    /// tailnet publish, a proxy, a container port map — and the publish outlives
    /// the process. So the requirement has to be recorded next to the service and
    /// survive a restart, exactly as the key itself does.
    #[serde(default)]
    pub requires_api_key: bool,
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
    /// When a stop was requested but could not confirm that every recorded
    /// process died. It records *intent*: the operator asked for this service to
    /// go away, so once the processes are observed gone the endpoint key may be
    /// dropped. A service that merely crashed carries no such intent and keeps
    /// its key, so it stays restartable/recoverable. Cleared on a confirmed stop
    /// and on a successful respawn. `None` on records written before this field
    /// existed, which is the safe default (keep the key).
    #[serde(default)]
    pub stop_requested_unix_ms: Option<u128>,
    /// When the last inference probe was attempted. Throttles re-probing of a
    /// service that is listed but still loading — see
    /// [`INFERENCE_PROBE_RETRY_INTERVAL`]. Absent on records written before
    /// readiness was gated on inference.
    #[serde(default)]
    pub inference_probe_attempted_at_unix_ms: Option<u64>,
    /// Coarse startup stage (`downloading`/`loading`/`warmup`) parsed from the
    /// serve process's own log output while it is coming up. Set to `None` once
    /// the service reaches `ready`, and absent on older on-disk records.
    #[serde(default)]
    pub startup_phase: Option<String>,
    /// When a real inference request first succeeded against this service. Once
    /// set, readiness checks stop re-probing and fall back to the cheap endpoint
    /// query — see [`managed_service_endpoint_readiness`]. Adopted from the
    /// engine state file when present, and absent on records written before
    /// readiness was gated on inference.
    #[serde(default)]
    pub inference_verified_at_unix_ms: Option<u64>,
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
            supervisor_start_ticks: None,
            engine_start_ticks: None,
            // Off unless a caller says otherwise: local loopback serving stays
            // credential-free, which is the unchanged default.
            requires_api_key: false,
            runtime_id,
            env_id,
            device_policy,
            gpu_indices: Vec::new(),
            engine_recipe_json: None,
            restart_count: 0,
            last_restart_unix_ms: None,
            stop_requested_unix_ms: None,
            startup_phase: None,
            inference_verified_at_unix_ms: None,
            inference_probe_attempted_at_unix_ms: None,
            manifest_path,
            log_path,
            engine_state_path,
            created_at_unix_ms: unix_time_millis(),
        }
    }

    /// Drop the per-run state that a restart invalidates, and count the restart.
    ///
    /// A restart reuses this manifest but spawns a different server with an
    /// unloaded model, so anything describing the previous run has to go. Chiefly
    /// the inference verification: left set, it short-circuits readiness straight
    /// back to "ready" as soon as the new server lists the model, reinstating the
    /// false positive the probe exists to prevent. The engine's own state file is
    /// rewritten from scratch on restart, so only this copy needs clearing — and
    /// [`Self::refresh_from_engine_state`] only ever adopts a verification, never
    /// clears one, so a stale value here would survive indefinitely.
    pub fn reset_for_restart(&mut self) {
        self.inference_verified_at_unix_ms = None;
        self.inference_probe_attempted_at_unix_ms = None;
        self.restart_count = self.restart_count.saturating_add(1);
        self.last_restart_unix_ms = Some(unix_time_millis());
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
        // Adopt the engine server PID together with ITS OWN start-time token so
        // a later stop verifies the server process, not the launcher. The ticks
        // key must match the PID key: `server_pid`↔`server_start_ticks`,
        // `pid`↔`start_ticks`.
        let engine_pid = state
            .get("server_pid")
            .and_then(serde_json::Value::as_u64)
            .map(|pid| (pid, "server_start_ticks"))
            .or_else(|| {
                state
                    .get("pid")
                    .and_then(serde_json::Value::as_u64)
                    .map(|pid| (pid, "start_ticks"))
            });
        if let Some((pid, ticks_key)) = engine_pid
            && let Ok(pid) = u32::try_from(pid)
        {
            self.engine_pid = Some(pid);
            self.engine_start_ticks = state.get(ticks_key).and_then(serde_json::Value::as_u64);
        }
        // Adopt the engine's inference verification so the CLI side does not
        // re-probe a service the engine healthcheck already confirmed.
        if self.inference_verified_at_unix_ms.is_none()
            && let Some(verified_at) = state
                .get(INFERENCE_VERIFIED_STATE_KEY)
                .and_then(serde_json::Value::as_u64)
        {
            self.inference_verified_at_unix_ms = Some(verified_at);
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

/// The `--gpu-memory-utilization` workaround for a shared/busy GPU.
///
/// Shared by the `rocm` CLI (pre-launch low-VRAM note), the vLLM engine adapter
/// (post-failure OOM hint) and the `fix-16-vllm-oom` diagnosis summary, so those
/// surfaces never drift into different wording for the same fix. The
/// `rocm fix fix-16-vllm-oom` catalog rationale is deliberately NOT this text —
/// it frames the same fault for a different reader — but its worked value is
/// pinned to this one (see below). vLLM reserves a fixed fraction of each
/// GPU's *total* VRAM by default (~0.9), independent of the model size or how
/// much is currently free, so on a shared or busy card that reservation
/// collides with memory already in use and the engine OOMs even a tiny model.
///
/// The worked example must stay the value the `fix-16-vllm-oom` recipe and the
/// docs hand the user (`0.5`). A smaller budget such as `0.1` sits below the
/// weights of most models people actually serve, so it trades one startup
/// failure for another. Pinned by
/// `the_utilization_hint_example_matches_the_recipe_command`.
pub const VLLM_GPU_MEMORY_UTILIZATION_HINT: &str = "vLLM reserves ~90% of the GPU's total VRAM by default; on a shared or busy GPU this can \
     collide with memory already in use. Lower the reservation with `--gpu-memory-utilization \
     <0-1>` (e.g. 0.5 for a small model), or target a less-busy GPU with `--gpu <index>`.";

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

/// A validated managed-service identifier that is safe to use as a single
/// filesystem path component.
///
/// Every managed-service path (e.g. the endpoint-key sidecar) is built as
/// `services_dir().join(format!("{service_id}..."))`. A `ServiceId` can only be
/// constructed through [`ServiceId::new`], which rejects path separators, `..`
/// traversal, and control characters — so a value of this type can never make a
/// join escape its intended directory. Prefer threading a `ServiceId` (or
/// validating with it) over passing raw `&str` ids into path builders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceId(String);

impl ServiceId {
    /// Validate an untrusted string as a service id.
    ///
    /// Rejects empty/whitespace-only input, path separators (`/` and `\`), `..`
    /// traversal sequences, and control characters. Ids produced by
    /// [`generate_service_id`] are always accepted.
    ///
    /// # Errors
    /// Returns an error describing the first rule the input violates.
    pub fn new(value: &str) -> Result<Self> {
        if value.trim().is_empty() {
            bail!("service id must not be empty");
        }
        if value.contains('/') || value.contains('\\') {
            bail!("service id must not contain path separators");
        }
        if value.contains("..") {
            bail!("service id must not contain `..`");
        }
        if value.chars().any(char::is_control) {
            bail!("service id must not contain control characters");
        }
        Ok(Self(value.to_owned()))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::convert::TryFrom<&str> for ServiceId {
    type Error = anyhow::Error;
    fn try_from(value: &str) -> Result<Self> {
        Self::new(value)
    }
}

impl std::fmt::Display for ServiceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for ServiceId {
    fn as_ref(&self) -> &str {
        &self.0
    }
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
    use crate::test_support::{temp_app_paths, workspace_test_artifact_dir};

    use std::fs;
    use std::path::Path;
    use std::path::PathBuf;

    #[test]
    fn wait_timeout_millis_keeps_the_startup_wait_bounded() {
        assert_eq!(wait_timeout_millis(Duration::from_millis(200)), 200);
        assert_eq!(wait_timeout_millis(Duration::ZERO), 0);
        // Sub-millisecond budgets truncate to a poll rather than rounding up.
        assert_eq!(wait_timeout_millis(Duration::from_micros(900)), 0);
        // A budget past the `u32` millisecond range must never land on
        // `INFINITE` (`u32::MAX`), which would block until the child exits.
        let huge = wait_timeout_millis(Duration::from_secs(u64::from(u32::MAX)));
        assert_eq!(huge, u32::MAX - 1);
        assert_ne!(huge, u32::MAX);
    }

    /// Watching a detached spawn must surface a child that died during startup.
    /// Only meaningful on Windows: the detached spawn primitives return a bare
    /// PID there, so this wait is the only chance to notice the exit.
    #[cfg(windows)]
    #[test]
    fn watched_detached_spawn_reports_a_child_that_exits_immediately() {
        // Each token is a separate argument so the assembled command line needs no
        // quoting, keeping `cmd.exe`'s quote-stripping rules out of the test.
        let spawn = spawn_detached_no_inherit_watching_startup(
            &test_system32_tool("cmd.exe"),
            &["/C".to_owned(), "exit".to_owned(), "7".to_owned()],
            &[],
            Duration::from_secs(10),
        )
        .expect("spawn should succeed");
        assert_eq!(spawn.early_exit_code, Some(7));
        assert_ne!(spawn.pid, 0);
    }

    /// The converse: a child that is still running must not be reported as a
    /// startup failure, or every successful launch would be rejected.
    #[cfg(windows)]
    #[test]
    fn watched_detached_spawn_does_not_report_a_child_that_keeps_running() {
        // `ping` against localhost is the dependency-free Windows sleep, and ~9s
        // outlasts the 200 ms observation window by well over an order of
        // magnitude. Spawned directly rather than through `cmd /C` so the
        // returned PID is the process the test has to clean up: on Windows
        // `terminate_process_tree` is plain `terminate_process`, so an
        // intermediate shell would leave `ping` orphaned.
        let spawn = spawn_detached_no_inherit_watching_startup(
            &test_system32_tool("ping.exe"),
            &["-n".to_owned(), "10".to_owned(), "127.0.0.1".to_owned()],
            &[],
            Duration::from_millis(200),
        )
        .expect("spawn should succeed");
        // Clean up before asserting, so a failing assertion cannot leak the child.
        let early_exit_code = spawn.early_exit_code;
        let was_running = process_is_running(spawn.pid);
        let _ = terminate_process(spawn.pid);
        assert_eq!(early_exit_code, None);
        assert!(was_running);
    }

    /// `CreateProcessW` is called with an explicit application name, so it does
    /// no `PATH` search — the program has to be a full path.
    #[cfg(windows)]
    fn test_system32_tool(exe: &str) -> PathBuf {
        PathBuf::from(std::env::var_os("SystemRoot").unwrap_or_else(|| r"C:\Windows".into()))
            .join("System32")
            .join(exe)
    }

    #[test]
    fn file_lock_creates_missing_parent_dirs_and_lock_file() {
        let dir =
            std::env::temp_dir().join(format!("rocm-core-filelock-create-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let lock_path = dir.join("nested").join("child").join("guard.lock");
        assert!(!lock_path.exists(), "precondition: lock file absent");

        let guard = FileLock::acquire(&lock_path).expect("acquire creates parents");
        assert!(lock_path.is_file(), "lock file is created on acquire");
        assert_eq!(guard.path(), lock_path.as_path());
        drop(guard);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_lock_serializes_concurrent_holders() {
        use std::sync::mpsc;
        use std::time::Duration;

        let dir =
            std::env::temp_dir().join(format!("rocm-core-filelock-excl-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let lock_path = dir.join("guard.lock");

        // First holder takes the lock and keeps it until we explicitly release it.
        let held = FileLock::acquire(&lock_path).expect("first acquire");

        let (started_tx, started_rx) = mpsc::channel();
        let (acquired_tx, acquired_rx) = mpsc::channel();
        let thread_path = lock_path;
        let handle = std::thread::spawn(move || {
            started_tx.send(()).expect("signal about-to-acquire");
            // Blocks until the main thread drops `held`.
            let _guard = FileLock::acquire(&thread_path).expect("second acquire");
            acquired_tx.send(()).expect("signal acquired");
        });

        // Ensure the contender has reached its acquire call before we assert it
        // is blocked, so the negative check below is about the lock, not
        // scheduling latency.
        started_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("contender started");
        assert!(
            acquired_rx
                .recv_timeout(Duration::from_millis(300))
                .is_err(),
            "second acquire must block while the first lock is still held"
        );

        // Releasing the first lock lets the contender proceed promptly.
        drop(held);
        acquired_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("second acquire proceeds once the first lock is released");
        handle.join().expect("contender thread joins");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_lock_distinct_paths_do_not_contend() {
        let dir = std::env::temp_dir().join(format!(
            "rocm-core-filelock-distinct-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);

        // Two different lock files are independent; holding one must not block the
        // other in the same process.
        let a = FileLock::acquire(dir.join("a.lock")).expect("acquire a");
        let b = FileLock::acquire(dir.join("b.lock")).expect("acquire b");
        drop(a);
        drop(b);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn service_id_accepts_generated_and_plain_ids() {
        // A freshly generated id must always validate.
        let generated = generate_service_id("vllm", "Qwen/Qwen3.5");
        assert!(ServiceId::new(&generated).is_ok());
        // Plain alphanumeric-with-dashes ids validate and round-trip verbatim.
        let id = ServiceId::new("svc-vllm-qwen-1730000000000").expect("valid id");
        assert_eq!(id.as_str(), "svc-vllm-qwen-1730000000000");
        assert_eq!(id.to_string(), "svc-vllm-qwen-1730000000000");
    }

    #[test]
    fn service_id_rejects_traversal_and_separators() {
        // Anything that could make `services_dir().join(id)` escape the directory
        // (or otherwise not resolve to a single child component) is rejected.
        for bad in [
            "",
            "   ",
            "../../etc/passwd",
            "..",
            "a/b",
            "a\\b",
            "/abs",
            "svc\r\ninject",
            "svc\0nul",
        ] {
            assert!(
                ServiceId::new(bad).is_err(),
                "expected {bad:?} to be rejected"
            );
        }
    }

    #[test]
    fn generate_endpoint_api_key_is_random_and_alphanumeric() {
        let key = generate_endpoint_api_key();
        assert_eq!(key.len(), ENDPOINT_API_KEY_LEN);
        assert!(
            key.chars().all(|c| c.is_ascii_alphanumeric()),
            "key must be URL-safe alphanumeric, got {key:?}"
        );
        // Two draws must differ — a constant key would be catastrophic for auth.
        assert_ne!(generate_endpoint_api_key(), generate_endpoint_api_key());
        // A freshly generated key must itself pass the header-safety predicate.
        assert!(!endpoint_api_key_has_forbidden_chars(
            &generate_endpoint_api_key()
        ));
    }

    #[test]
    fn endpoint_api_key_has_forbidden_chars_flags_control_chars() {
        // Control characters (notably CR/LF) enable header injection when the key
        // is interpolated into a raw `Authorization: Bearer` line.
        for bad in ["key\r\ninject", "key\nother", "tab\there", "nul\0byte"] {
            assert!(
                endpoint_api_key_has_forbidden_chars(bad),
                "should reject {bad:?}"
            );
        }
        // Ordinary printable keys are accepted.
        for good in ["my-key", "AbC123._~-", "sk-proj-abcDEF0123456789"] {
            assert!(
                !endpoint_api_key_has_forbidden_chars(good),
                "should accept {good:?}"
            );
        }
    }

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
            None,
            Duration::from_secs(2)
        )?);

        let request = server.join().expect("server thread should not panic")?;
        assert!(request.starts_with("GET /v1/models HTTP/1.1"));
        Ok(())
    }

    #[test]
    fn openai_models_endpoint_sends_bearer_when_key_present() -> Result<()> {
        // A protected endpoint: 200 with the model list only when the request
        // carries `Authorization: Bearer test-key`, otherwise 401. Serves two
        // connections — the no-key probe, then the with-key probe.
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let server = std::thread::spawn(move || -> Result<()> {
            for _ in 0..2 {
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
                if request.contains("Authorization: Bearer test-key") {
                    let body = r#"{"data":[{"id":"qwen"}]}"#;
                    write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    )?;
                } else {
                    write!(
                        stream,
                        "HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    )?;
                }
            }
            Ok(())
        });
        let endpoint = format!("http://127.0.0.1:{port}/v1");

        // Without the key the protected endpoint 401s → treated as not ready.
        assert!(
            openai_models_endpoint_has_model(&endpoint, Some("qwen"), None, Duration::from_secs(2))
                .is_err()
        );
        // With the key the bearer header is sent → the model reads as ready.
        assert!(openai_models_endpoint_has_model(
            &endpoint,
            Some("qwen"),
            Some("test-key"),
            Duration::from_secs(2)
        )?);

        server.join().expect("server thread should not panic")?;
        Ok(())
    }

    // Why readiness is gated on an inference probe (EAI-7333): a server that
    // lists the model on `/v1/models` but cannot yet serve
    // `/v1/chat/completions` still reports the model as present. This test pins
    // that `openai_models_endpoint_has_model` alone is a false positive for
    // inference-readiness, which is why callers must additionally probe
    // inference — see `openai_chat_completion_probe` and
    // `managed_service_endpoint_readiness`.
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
            None,
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

    /// Serve `count` canned HTTP responses on a loopback port, returning the port
    /// and a handle yielding the requests that were received. Each response is
    /// `(status_line, body)`; a `None` response accepts the connection and never
    /// answers, standing in for an engine that hangs.
    fn spawn_canned_http_server(
        responses: Vec<Option<(&'static str, &'static str)>>,
    ) -> Result<(u16, std::thread::JoinHandle<Result<Vec<String>>>)> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let handle = std::thread::spawn(move || -> Result<Vec<String>> {
            let mut requests = Vec::new();
            for response in responses {
                let (mut stream, _) = listener.accept()?;
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut request_bytes = Vec::new();
                let mut buffer = [0_u8; 512];
                while let Ok(read) = stream.read(&mut buffer) {
                    if read == 0 {
                        break;
                    }
                    request_bytes.extend_from_slice(&buffer[..read]);
                    let text = String::from_utf8_lossy(&request_bytes);
                    // Requests with a body (POST) are complete once the declared
                    // content length has arrived after the header terminator.
                    if let Some((headers, body)) = text.split_once("\r\n\r\n") {
                        let declared = headers
                            .lines()
                            .find_map(|line| {
                                line.strip_prefix("Content-Length: ")?.trim().parse().ok()
                            })
                            .unwrap_or(0_usize);
                        if body.len() >= declared {
                            break;
                        }
                    }
                }
                requests.push(String::from_utf8_lossy(&request_bytes).into_owned());
                let Some((status_line, body)) = response else {
                    // Hang: hold the connection open without answering until the
                    // client's read timeout fires, then drop it.
                    std::thread::sleep(Duration::from_millis(1500));
                    continue;
                };
                write!(
                    stream,
                    "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )?;
            }
            Ok(requests)
        });
        Ok((port, handle))
    }

    /// How a scripted download server answers one request.
    #[derive(Clone, Copy)]
    enum DownloadReply {
        /// The whole body with a matching `Content-Length`.
        Complete,
        /// Declare the full length, send only `sent` bytes, then close —
        /// an interrupted transfer.
        Truncated {
            sent: usize,
        },
        /// Honour `Range` and serve the remainder as `206`.
        Resume,
        /// Answer `206` but from `start` regardless of what `Range` asked for,
        /// as a non-compliant server or a broken caching proxy might.
        ResumeAtWrongOffset {
            start: usize,
        },
        /// Ignore `Range` and answer `200` with the whole body, as a server
        /// without range support does.
        IgnoreRange,
        Status(&'static str),
    }

    impl DownloadReply {
        fn client_may_disconnect(self) -> bool {
            matches!(
                self,
                Self::ResumeAtWrongOffset { .. } | Self::Truncated { sent: 0 }
            )
        }
    }

    fn allow_expected_peer_disconnect(result: std::io::Result<()>) -> std::io::Result<()> {
        match result {
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::BrokenPipe | std::io::ErrorKind::ConnectionReset
                ) =>
            {
                Ok(())
            }
            result => result,
        }
    }

    /// Serve `body` over loopback, answering each request per `replies`.
    /// Returns the port and a handle yielding the raw requests received.
    fn spawn_download_server(
        body: Vec<u8>,
        replies: Vec<DownloadReply>,
    ) -> Result<(u16, std::thread::JoinHandle<Result<Vec<String>>>)> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let handle = std::thread::spawn(move || -> Result<Vec<String>> {
            let mut requests = Vec::new();
            for reply in replies {
                let (mut stream, _) = listener.accept()?;
                // An accepted socket inherits the listener's mode on Windows,
                // where a non-blocking read fails with `WSAEWOULDBLOCK` instead
                // of waiting. Be explicit rather than rely on the platform.
                stream.set_nonblocking(false)?;
                stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
                let mut request_bytes = Vec::new();
                let mut buffer = [0_u8; 512];
                while let Ok(read) = stream.read(&mut buffer) {
                    if read == 0 {
                        break;
                    }
                    request_bytes.extend_from_slice(&buffer[..read]);
                    if request_bytes.windows(4).any(|window| window == b"\r\n\r\n") {
                        break;
                    }
                }
                let request = String::from_utf8_lossy(&request_bytes).into_owned();
                let range_start = request
                    .lines()
                    .find_map(|line| line.strip_prefix("Range: bytes="))
                    .and_then(|value| value.trim().split('-').next()?.parse::<usize>().ok())
                    .unwrap_or(0);
                requests.push(request);
                let total = body.len();
                let client_may_disconnect = reply.client_may_disconnect();
                match reply {
                    DownloadReply::Complete | DownloadReply::IgnoreRange => {
                        write!(
                            stream,
                            "HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nContent-Length: {total}\r\nConnection: close\r\n\r\n"
                        )?;
                        stream.write_all(&body)?;
                    }
                    DownloadReply::Truncated { sent } => {
                        write!(
                            stream,
                            "HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nContent-Length: {total}\r\nConnection: close\r\n\r\n"
                        )?;
                        stream.write_all(&body[..sent.min(total)])?;
                    }
                    DownloadReply::Resume => {
                        let start = range_start.min(total);
                        write!(
                            stream,
                            "HTTP/1.1 206 Partial Content\r\nAccept-Ranges: bytes\r\nContent-Range: bytes {start}-{}/{total}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            total.saturating_sub(1),
                            total - start
                        )?;
                        stream.write_all(&body[start..])?;
                    }
                    DownloadReply::ResumeAtWrongOffset { start } => {
                        let start = start.min(total);
                        write!(
                            stream,
                            "HTTP/1.1 206 Partial Content\r\nAccept-Ranges: bytes\r\nContent-Range: bytes {start}-{}/{total}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            total.saturating_sub(1),
                            total - start
                        )?;
                        allow_expected_peer_disconnect(stream.write_all(&body[start..]))?;
                    }
                    DownloadReply::Status(status_line) => {
                        write!(
                            stream,
                            "{status_line}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        )?;
                    }
                }
                if client_may_disconnect {
                    allow_expected_peer_disconnect(stream.flush())?;
                } else {
                    stream.flush()?;
                }
            }
            Ok(requests)
        });
        Ok((port, handle))
    }

    /// A body large enough to span many `DOWNLOAD_CHUNK_BYTES` reads, so the
    /// streaming loop is genuinely exercised rather than fitting in one chunk.
    fn download_body() -> Vec<u8> {
        (0..DOWNLOAD_CHUNK_BYTES * 3 + 1234)
            .map(|index| (index % 251) as u8)
            .collect()
    }

    fn download_scratch(tag: &str) -> PathBuf {
        let dir = workspace_test_artifact_dir().join(format!(
            "download-{tag}-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("failed to create scratch dir");
        dir
    }

    #[test]
    fn download_fixture_tolerates_only_expected_peer_disconnects() {
        assert!(DownloadReply::ResumeAtWrongOffset { start: 1 }.client_may_disconnect());
        assert!(DownloadReply::Truncated { sent: 0 }.client_may_disconnect());
        assert!(!DownloadReply::Truncated { sent: 1 }.client_may_disconnect());
        assert!(!DownloadReply::Complete.client_may_disconnect());

        for kind in [
            std::io::ErrorKind::BrokenPipe,
            std::io::ErrorKind::ConnectionReset,
        ] {
            allow_expected_peer_disconnect(Err(std::io::Error::from(kind)))
                .expect("an intentionally rejected response may close its socket");
        }

        let error =
            allow_expected_peer_disconnect(Err(std::io::Error::from(std::io::ErrorKind::TimedOut)))
                .expect_err("unrelated fixture I/O failures must remain visible");
        assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
    }

    #[test]
    fn download_streams_a_large_body_and_reports_its_digest() -> Result<()> {
        let body = download_body();
        let (port, server) = spawn_download_server(body.clone(), vec![DownloadReply::Complete])?;
        let dir = download_scratch("complete");
        let destination = dir.join("artifact.bin");

        let outcome = download_file_streaming(&DownloadRequest::new(
            &format!("http://127.0.0.1:{port}/artifact.bin"),
            &destination,
            Duration::from_secs(10),
        ))?;

        let written = fs::read(&destination)?;
        let expected_digest = format!("{:x}", Sha256::digest(&body));
        server.join().expect("server thread")?;
        let leftovers: Vec<_> = fs::read_dir(&dir)?
            .filter_map(|entry| entry.ok().map(|entry| entry.file_name()))
            .filter(|name| name.to_string_lossy().contains(".part"))
            .collect();
        fs::remove_dir_all(&dir).ok();

        assert_eq!(written, body, "the saved file must match the served bytes");
        assert_eq!(outcome.bytes_written, body.len() as u64);
        assert_eq!(outcome.sha256, expected_digest);
        assert!(leftovers.is_empty(), "the .part file must not survive");
        Ok(())
    }

    #[test]
    fn download_with_progress_reports_cumulative_bytes_across_chunks() -> Result<()> {
        let body = download_body();
        let (port, server) = spawn_download_server(body.clone(), vec![DownloadReply::Complete])?;
        let dir = download_scratch("progress");
        let destination = dir.join("artifact.bin");

        let mut calls: Vec<(u64, Option<u64>)> = Vec::new();
        let outcome = download_file_streaming_with_progress(
            &DownloadRequest::new(
                &format!("http://127.0.0.1:{port}/artifact.bin"),
                &destination,
                Duration::from_secs(10),
            ),
            &mut |written, total| calls.push((written, total)),
        )?;

        server.join().expect("server thread")?;
        fs::remove_dir_all(&dir).ok();

        let total = Some(body.len() as u64);
        assert_eq!(
            calls.first(),
            Some(&(0, total)),
            "the first call must fire before any bytes are read, already knowing the total: {calls:?}"
        );
        assert_eq!(
            calls.last(),
            Some(&(body.len() as u64, total)),
            "the last call must report the complete byte count: {calls:?}"
        );
        assert!(
            calls.windows(2).all(|pair| pair[0].0 <= pair[1].0),
            "cumulative bytes must never go backwards: {calls:?}"
        );
        assert!(
            calls
                .iter()
                .all(|&(_, reported_total)| reported_total == total),
            "the total must stay constant across an attempt: {calls:?}"
        );
        assert_eq!(outcome.bytes_written, body.len() as u64);
        Ok(())
    }

    #[test]
    // This guards the resume path re-seeding `written` from the partial file
    // already on disk (see `written = std::io::copy(...)` above), not the
    // high-water-mark clamp itself — it would pass unchanged with the clamp
    // removed entirely. `download_with_progress_stays_monotonic_after_a_discarded_restart`
    // below is the one that actually exercises the clamp.
    fn download_with_progress_reports_the_resumed_offset_before_reading_more() -> Result<()> {
        let body = download_body();
        let (port, server) = spawn_download_server(
            body.clone(),
            vec![
                DownloadReply::Truncated { sent: 5000 },
                DownloadReply::Resume,
            ],
        )?;
        let dir = download_scratch("progress-resume");
        let destination = dir.join("artifact.bin");

        let mut calls: Vec<(u64, Option<u64>)> = Vec::new();
        let outcome = download_file_streaming_with_progress(
            &DownloadRequest::new(
                &format!("http://127.0.0.1:{port}/artifact.bin"),
                &destination,
                Duration::from_secs(10),
            ),
            &mut |written, total| calls.push((written, total)),
        )?;

        server.join().expect("server thread")?;
        fs::remove_dir_all(&dir).ok();

        let total = Some(body.len() as u64);
        assert!(
            calls.windows(2).all(|pair| pair[0].0 <= pair[1].0),
            "byte counts must never regress across a retried attempt, e.g. reset to 0: {calls:?}"
        );
        assert!(
            calls.iter().any(|&(written, _)| written == 5000),
            "the resumed attempt must report the byte count already on disk before reading more: {calls:?}"
        );
        assert!(
            calls
                .iter()
                .all(|&(_, reported_total)| reported_total == total),
            "the total size must stay stable across the retry: {calls:?}"
        );
        assert_eq!(
            calls.last(),
            Some(&(body.len() as u64, total)),
            "the last call must report the complete byte count: {calls:?}"
        );
        assert_eq!(outcome.bytes_written, body.len() as u64);
        Ok(())
    }

    #[test]
    fn download_with_progress_stays_monotonic_after_a_discarded_restart() -> Result<()> {
        // The second reply resumes at the wrong offset, so its partial file is
        // discarded and the third attempt restarts from scratch — internally
        // reporting 0 bytes written again even though the first attempt had
        // already reached 5000. The caller must never see that drop.
        let body = download_body();
        let (port, server) = spawn_download_server(
            body.clone(),
            vec![
                DownloadReply::Truncated { sent: 5000 },
                DownloadReply::ResumeAtWrongOffset { start: 8000 },
                DownloadReply::Complete,
            ],
        )?;
        let dir = download_scratch("progress-wrong-offset");
        let destination = dir.join("artifact.bin");

        let mut calls: Vec<(u64, Option<u64>)> = Vec::new();
        let outcome = download_file_streaming_with_progress(
            &DownloadRequest::new(
                &format!("http://127.0.0.1:{port}/artifact.bin"),
                &destination,
                Duration::from_secs(10),
            ),
            &mut |written, total| calls.push((written, total)),
        )?;

        let requests = server.join().expect("server thread")?;
        fs::remove_dir_all(&dir).ok();

        let total = Some(body.len() as u64);
        assert_eq!(
            requests.len(),
            3,
            "the wrong-offset reply must be discarded and retried, not accepted"
        );
        assert!(
            calls.windows(2).all(|pair| pair[0].0 <= pair[1].0),
            "cumulative bytes must never go backwards, even across a discarded \
             partial file and a from-scratch restart: {calls:?}"
        );
        assert!(
            calls.iter().any(|&(written, _)| written == 5000),
            "the truncated first attempt's progress must not be lost once the \
             restart reports 0 internally: {calls:?}"
        );
        assert_eq!(
            calls.last(),
            Some(&(body.len() as u64, total)),
            "the last call must report the complete byte count: {calls:?}"
        );
        assert_eq!(outcome.bytes_written, body.len() as u64);
        Ok(())
    }

    #[test]
    fn download_interrupted_beyond_recovery_leaves_no_destination_file() -> Result<()> {
        // Every attempt ends early, so the download never completes and the
        // user is left with the truncation as the reported reason.
        let body = download_body();
        let (port, server) = spawn_download_server(
            body,
            vec![DownloadReply::Truncated { sent: 4096 }; DOWNLOAD_MAX_ATTEMPTS as usize],
        )?;
        let dir = download_scratch("interrupted");
        let destination = dir.join("artifact.bin");

        let error = download_file_streaming(&DownloadRequest::new(
            &format!("http://127.0.0.1:{port}/artifact.bin"),
            &destination,
            Duration::from_secs(5),
        ))
        .expect_err("a download that never completes must fail");

        let requests = server.join().expect("server thread")?;
        let entries: Vec<_> = fs::read_dir(&dir)?
            .filter_map(|entry| entry.ok().map(|entry| entry.file_name()))
            .collect();
        fs::remove_dir_all(&dir).ok();

        assert_eq!(
            requests.len(),
            DOWNLOAD_MAX_ATTEMPTS as usize,
            "a truncated transfer is transient, so every attempt should be spent"
        );
        assert!(
            !destination.exists(),
            "a partial download must never appear at the destination, where a \
             later run would treat it as a complete cached artifact"
        );
        assert!(
            entries.is_empty(),
            "the .part file must be cleaned up, found {entries:?}"
        );
        assert!(
            error.to_string().contains("incomplete download"),
            "the user should be told the transfer was short: {error}"
        );
        Ok(())
    }

    #[test]
    fn download_resumes_from_where_the_transfer_stopped() -> Result<()> {
        let body = download_body();
        let (port, server) = spawn_download_server(
            body.clone(),
            vec![
                DownloadReply::Truncated { sent: 5000 },
                DownloadReply::Resume,
            ],
        )?;
        let dir = download_scratch("resume");
        let destination = dir.join("artifact.bin");

        let outcome = download_file_streaming(&DownloadRequest::new(
            &format!("http://127.0.0.1:{port}/artifact.bin"),
            &destination,
            Duration::from_secs(10),
        ))?;

        let written = fs::read(&destination)?;
        let requests = server.join().expect("server thread")?;
        fs::remove_dir_all(&dir).ok();

        assert_eq!(
            written, body,
            "a resumed download must reconstruct the artifact exactly"
        );
        assert_eq!(
            outcome.sha256,
            format!("{:x}", Sha256::digest(&body)),
            "the digest must cover the resumed prefix too, not just the second attempt"
        );
        assert_eq!(requests.len(), 2, "one retry, not a full restart");
        assert!(
            requests[1].contains("Range: bytes=5000-"),
            "the retry must ask to continue from byte 5000: {}",
            requests[1]
        );
        Ok(())
    }

    #[test]
    fn download_restarts_cleanly_when_resume_lands_at_the_wrong_offset() -> Result<()> {
        // The second reply answers `206` from byte 8000, not the byte 5000 we
        // asked to resume from. Accepting that slice's own `Content-Length` as
        // the whole artifact would silently rename a corrupt file into place;
        // instead the third attempt must be a plain `GET` that refetches the
        // whole thing.
        let body = download_body();
        let (port, server) = spawn_download_server(
            body.clone(),
            vec![
                DownloadReply::Truncated { sent: 5000 },
                DownloadReply::ResumeAtWrongOffset { start: 8000 },
                DownloadReply::Complete,
            ],
        )?;
        let dir = download_scratch("wrong-offset");
        let destination = dir.join("artifact.bin");

        let outcome = download_file_streaming(&DownloadRequest::new(
            &format!("http://127.0.0.1:{port}/artifact.bin"),
            &destination,
            Duration::from_secs(10),
        ))?;

        let written = fs::read(&destination)?;
        let requests = server.join().expect("server thread")?;
        let leftovers: Vec<_> = fs::read_dir(&dir)?
            .filter_map(|entry| entry.ok().map(|entry| entry.file_name()))
            .filter(|name| name.to_string_lossy().contains(".part"))
            .collect();
        fs::remove_dir_all(&dir).ok();

        assert_eq!(
            written, body,
            "a mismatched-offset resume must not be accepted as a complete artifact"
        );
        assert_eq!(
            outcome.sha256,
            format!("{:x}", Sha256::digest(&body)),
            "the digest must cover the whole artifact from the clean restart"
        );
        assert_eq!(
            requests.len(),
            3,
            "the wrong-offset reply must be discarded and retried, not accepted"
        );
        assert!(
            !requests[2].contains("Range:"),
            "the restart after a wrong-offset resume must be a plain GET: {}",
            requests[2]
        );
        assert!(leftovers.is_empty(), "the .part file must not survive");
        Ok(())
    }

    #[test]
    fn download_restarts_cleanly_when_the_server_ignores_range() -> Result<()> {
        // A server without range support answers 200 with the whole body.
        // Appending that onto the bytes already written would double the file.
        let body = download_body();
        let (port, server) = spawn_download_server(
            body.clone(),
            vec![
                DownloadReply::Truncated { sent: 3000 },
                DownloadReply::IgnoreRange,
            ],
        )?;
        let dir = download_scratch("ignore-range");
        let destination = dir.join("artifact.bin");

        download_file_streaming(&DownloadRequest::new(
            &format!("http://127.0.0.1:{port}/artifact.bin"),
            &destination,
            Duration::from_secs(10),
        ))?;

        let written = fs::read(&destination)?;
        server.join().expect("server thread")?;
        fs::remove_dir_all(&dir).ok();

        assert_eq!(written, body, "the restart must not append onto the prefix");
        Ok(())
    }

    #[test]
    fn download_rejects_a_digest_mismatch_without_retrying() -> Result<()> {
        let body = download_body();
        let (port, server) = spawn_download_server(body, vec![DownloadReply::Complete])?;
        let dir = download_scratch("digest");
        let destination = dir.join("artifact.bin");
        let url = format!("http://127.0.0.1:{port}/artifact.bin");
        let wrong_digest = "a".repeat(64);
        let mut request = DownloadRequest::new(&url, &destination, Duration::from_secs(10));
        request.expected_sha256 = Some(&wrong_digest);

        let error = download_file_streaming(&request).expect_err("wrong digest must fail");

        let requests = server.join().expect("server thread")?;
        let leftovers = fs::read_dir(&dir)?.count();
        fs::remove_dir_all(&dir).ok();

        assert!(error.to_string().contains("SHA-256 mismatch"), "{error}");
        assert!(!destination.exists());
        assert_eq!(
            requests.len(),
            1,
            "corrupt bytes are not a transient failure; retrying would re-fetch the same thing"
        );
        assert_eq!(leftovers, 0, "the corrupt prefix must be discarded");
        Ok(())
    }

    #[test]
    fn download_does_not_retry_a_client_error() -> Result<()> {
        let body = download_body();
        let (port, server) =
            spawn_download_server(body, vec![DownloadReply::Status("HTTP/1.1 404 Not Found")])?;
        let dir = download_scratch("not-found");
        let destination = dir.join("artifact.bin");

        let error = download_file_streaming(&DownloadRequest::new(
            &format!("http://127.0.0.1:{port}/artifact.bin"),
            &destination,
            Duration::from_secs(5),
        ))
        .expect_err("404 must fail");

        let requests = server.join().expect("server thread")?;
        fs::remove_dir_all(&dir).ok();

        assert!(error.to_string().contains("404"), "{error}");
        assert_eq!(requests.len(), 1, "a 404 will not become a 200 on retry");
        Ok(())
    }

    #[test]
    fn download_refuses_a_body_over_the_approved_limit() -> Result<()> {
        let body = download_body();
        let total = body.len() as u64;
        let (port, server) =
            spawn_download_server(body, vec![DownloadReply::Truncated { sent: 0 }])?;
        let dir = download_scratch("max-bytes");
        let destination = dir.join("artifact.bin");
        let url = format!("http://127.0.0.1:{port}/artifact.bin");
        let mut request = DownloadRequest::new(&url, &destination, Duration::from_secs(5));
        request.max_bytes = Some(total - 1);

        let error = download_file_streaming(&request).expect_err("over-limit must fail");

        server.join().expect("server thread")?;
        fs::remove_dir_all(&dir).ok();

        assert!(error.to_string().contains("approved limit"), "{error}");
        assert!(!destination.exists());
        Ok(())
    }

    #[test]
    fn download_enforces_a_caller_declared_size_the_server_agrees_with() -> Result<()> {
        // The server sends a complete, self-consistent body — it just is not
        // the artifact the manifest described. A digest would catch this too,
        // but the size contract is independent and must hold on its own.
        let body = download_body();
        let (port, server) = spawn_download_server(body.clone(), vec![DownloadReply::Complete])?;
        let dir = download_scratch("expected-len");
        let destination = dir.join("artifact.bin");
        let url = format!("http://127.0.0.1:{port}/artifact.bin");
        let mut request = DownloadRequest::new(&url, &destination, Duration::from_secs(10));
        request.expected_len = Some(body.len() as u64 + 1);

        let error = download_file_streaming(&request).expect_err("wrong size must fail");

        let requests = server.join().expect("server thread")?;
        fs::remove_dir_all(&dir).ok();

        assert!(error.to_string().contains("were expected"), "{error}");
        assert!(!destination.exists());
        assert_eq!(
            requests.len(),
            1,
            "a manifest disagreement will not resolve itself on retry"
        );
        Ok(())
    }

    #[test]
    fn download_backoff_grows_then_settles_at_its_ceiling() {
        let mut backoff = Backoff::new(Duration::from_millis(100), Duration::from_millis(800), 2);
        let delays: Vec<u128> = (0..5).map(|_| backoff.next_delay().as_millis()).collect();
        assert_eq!(delays, vec![100, 200, 400, 800, 800]);
    }

    const CHAT_OK_BODY: &str = r#"{"choices":[{"message":{"role":"assistant","content":"ok"}}]}"#;
    const MODELS_OK_BODY: &str = r#"{"data":[{"id":"Qwen/Qwen3-0.6B"}]}"#;

    #[test]
    fn chat_completion_probe_passes_when_the_endpoint_answers() -> Result<()> {
        let (port, server) =
            spawn_canned_http_server(vec![Some(("HTTP/1.1 200 OK", CHAT_OK_BODY))])?;
        let endpoint = format!("http://127.0.0.1:{port}/v1");

        assert!(openai_chat_completion_probe(
            &endpoint,
            "Qwen/Qwen3-0.6B",
            None,
            Duration::from_secs(2)
        )?);

        let requests = server.join().expect("server thread should not panic")?;
        let request = requests.first().expect("the probe sends one request");
        assert!(
            request.starts_with("POST /v1/chat/completions HTTP/1.1"),
            "probe must exercise the inference path, got: {request}"
        );
        assert!(
            request.contains("\"model\":\"Qwen/Qwen3-0.6B\"") && request.contains("\"max_tokens\""),
            "probe asks the served model for a token-capped completion, got: {request}"
        );
        Ok(())
    }

    #[test]
    fn chat_completion_probe_separates_a_refusal_from_a_warmup_failure() -> Result<()> {
        // A refusal proves the inference path is up and the model is resident:
        // the request was understood and rejected on its merits. A 5xx is what an
        // engine returns while it is still warming up, which is not ready.
        let (port, server) = spawn_canned_http_server(vec![
            Some(("HTTP/1.1 400 Bad Request", r#"{"error":"unsupported"}"#)),
            Some((
                "HTTP/1.1 503 Service Unavailable",
                r#"{"error":"loading model"}"#,
            )),
        ])?;
        let endpoint = format!("http://127.0.0.1:{port}/v1");

        assert!(openai_chat_completion_probe(
            &endpoint,
            "qwen",
            None,
            Duration::from_secs(2)
        )?);
        assert!(!openai_chat_completion_probe(
            &endpoint,
            "qwen",
            None,
            Duration::from_secs(2)
        )?);

        server.join().expect("server thread should not panic")?;
        Ok(())
    }

    #[test]
    fn chat_completion_probe_accepts_a_response_from_a_server_that_holds_the_socket() -> Result<()>
    {
        // `Connection: close` is a request, not a guarantee — a server or an
        // intervening proxy may answer in full and keep the socket open. Reading
        // to EOF would stall until the timeout and throw the answer away, leaving
        // a perfectly healthy service stuck reporting "not ready".
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let server = std::thread::spawn(move || -> Result<()> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
            let mut buffer = [0_u8; 1024];
            let _ = stream.read(&mut buffer);
            let body = CHAT_OK_BODY;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            )?;
            stream.flush()?;
            // Hold the connection open past the probe's timeout.
            std::thread::sleep(Duration::from_secs(3));
            Ok(())
        });
        let endpoint = format!("http://127.0.0.1:{port}/v1");

        let started = Instant::now();
        assert!(
            openai_chat_completion_probe(&endpoint, "qwen", None, Duration::from_secs(2))?,
            "a complete response counts even when the peer keeps the socket open"
        );
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "the framed response is complete, so the probe must not wait for EOF"
        );

        server.join().expect("server thread should not panic")?;
        Ok(())
    }

    #[test]
    fn http_read_is_bounded_across_reads_not_just_per_read() -> Result<()> {
        // A socket read timeout bounds each `read`, not the sequence of them. A
        // server that dribbles bytes forever, each within the per-read timeout,
        // must still hit the caller's overall budget.
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let server = std::thread::spawn(move || -> Result<()> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
            let mut buffer = [0_u8; 1024];
            let _ = stream.read(&mut buffer);
            // Never declares a length and never finishes: one byte at a time,
            // comfortably inside any per-read timeout.
            for _ in 0..200 {
                if stream.write_all(b"x").is_err() || stream.flush().is_err() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Ok(())
        });
        let endpoint = format!("http://127.0.0.1:{port}/v1");

        let started = Instant::now();
        assert!(
            openai_chat_completion_probe(&endpoint, "qwen", None, Duration::from_millis(500))
                .is_err(),
            "a response that never completes is not a passing probe"
        );
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "the call must honor its own budget, not a multiple of it: took {:?}",
            started.elapsed()
        );

        let _ = server.join();
        Ok(())
    }

    #[test]
    fn http_response_is_complete_does_not_miscount_a_split_multibyte_char() {
        // A body ending in a multi-byte UTF-8 character can arrive one byte
        // short of the declared Content-Length. Lossy-decoding the whole
        // buffer to check completeness turns that dangling partial sequence
        // into a 3-byte U+FFFD replacement, inflating the decoded length past
        // the declared one and reporting completeness a read early.
        let body = "hi \u{2603}"; // snowman is a 3-byte UTF-8 character
        let full = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .into_bytes();
        let truncated = &full[..full.len() - 1];

        assert!(!http_response_is_complete(truncated));
        assert!(http_response_is_complete(&full));
    }

    /// A signal arriving mid-response must not fail the request.
    ///
    /// A signal delivered while the client is parked in `read` aborts it with
    /// `EINTR`. `SA_RESTART` does not save us: Linux never restarts a socket read
    /// that has a receive timeout set, and this client sets one on every pass.
    /// Any handler in the process is enough to trigger it — `crossterm`'s
    /// `SIGWINCH` hook is linked into the CLI — so treating `EINTR` as a
    /// transport error turned an unrelated signal into a spurious "endpoint
    /// unreachable".
    ///
    /// Linux-only on purpose. The guarantee being exercised — a receive timeout
    /// defeats `SA_RESTART` — is documented for Linux; BSD-derived kernels may
    /// restart the read instead, which would leave this passing without ever
    /// reaching the retry. CI has no macOS runner to tell the difference.
    #[cfg(target_os = "linux")]
    #[test]
    fn http_read_survives_a_signal_arriving_mid_response() -> Result<()> {
        extern "C" fn noop_handler(_signal: libc::c_int) {}

        let body = r#"{"data":[{"id":"qwen"}]}"#;
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        let server = std::thread::spawn(move || -> Result<()> {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
            let mut buffer = [0_u8; 1024];
            let _ = stream.read(&mut buffer);
            // Answer late, so the client is blocked in `read` while the signals
            // land rather than racing them.
            std::thread::sleep(Duration::from_millis(400));
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            )?;
            stream.flush()?;
            Ok(())
        });

        // Install the handler with SA_RESTART set, to show that the flag is not
        // what protects this read.
        //
        // The handler is left installed rather than restored: the default
        // disposition for SIGUSR1 is to kill the process, so putting it back
        // would let a signal still in flight take the whole test binary down.
        // Leaving a no-op handler is inert — nothing else here raises SIGUSR1,
        // and the signals below are aimed at this thread alone, so no other
        // test sharing this process can observe either one.
        #[allow(unsafe_code)] // libc FFI
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = noop_handler as *const () as libc::sighandler_t;
            action.sa_flags = libc::SA_RESTART;
            libc::sigemptyset(&raw mut action.sa_mask);
            assert_eq!(
                libc::sigaction(libc::SIGUSR1, &raw const action, std::ptr::null_mut()),
                0,
                "failed to install the SIGUSR1 handler"
            );
        }

        // Target this thread specifically: a process-directed signal could land
        // on any thread and disturb an unrelated test sharing this process.
        #[allow(unsafe_code)] // libc FFI
        let reader = unsafe { libc::pthread_self() };
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let signaller = {
            let stop = std::sync::Arc::clone(&stop);
            std::thread::spawn(move || {
                // Let the connect and the request write finish first; only the
                // response read is under test.
                std::thread::sleep(Duration::from_millis(50));
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    #[allow(unsafe_code)] // libc FFI
                    unsafe {
                        libc::pthread_kill(reader, libc::SIGUSR1);
                    }
                    std::thread::sleep(Duration::from_millis(10));
                }
            })
        };

        let endpoint = format!("http://127.0.0.1:{port}/v1");
        let response =
            http_get_text_with_auth(&endpoint, "/v1/models", None, Duration::from_secs(5));

        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        signaller.join().expect("signaller thread should not panic");
        server.join().expect("server thread should not panic")?;

        assert_eq!(
            response?, body,
            "an interrupted read is retryable, not a failed request"
        );
        Ok(())
    }

    #[test]
    fn chat_completion_probe_fails_on_a_hung_endpoint() -> Result<()> {
        // The reported symptom: the endpoint accepts the connection and never
        // answers. The probe must give up within its timeout, not wait forever.
        let (port, server) = spawn_canned_http_server(vec![None])?;
        let endpoint = format!("http://127.0.0.1:{port}/v1");

        let started = Instant::now();
        assert!(
            openai_chat_completion_probe(&endpoint, "qwen", None, Duration::from_millis(300))
                .is_err(),
            "a hung endpoint is not inference-ready"
        );
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "the probe must be bounded by its timeout"
        );

        server.join().expect("server thread should not panic")?;
        Ok(())
    }

    fn probe_test_record(port: u16) -> ManagedServiceRecord {
        let root = PathBuf::from("/tmp/rocm-inference-probe-test");
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        ManagedServiceRecord::new(
            &paths,
            "svc-probe",
            "vllm",
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-0.6B",
            "127.0.0.1",
            port,
            "serve",
            4242,
            None,
            None,
            None,
        )
    }

    #[test]
    fn inference_readiness_latches_after_the_first_successful_probe() -> Result<()> {
        // First check: list the model, then probe inference. Second check: the
        // verdict is latched, so only the cheap listing is re-issued — repeated
        // `services list` polls must not queue generation work behind real
        // traffic.
        let (port, server) = spawn_canned_http_server(vec![
            Some(("HTTP/1.1 200 OK", MODELS_OK_BODY)),
            Some(("HTTP/1.1 200 OK", CHAT_OK_BODY)),
            Some(("HTTP/1.1 200 OK", MODELS_OK_BODY)),
        ])?;
        let mut record = probe_test_record(port);

        assert_eq!(
            managed_service_endpoint_readiness(
                &mut record,
                None,
                Duration::from_secs(2),
                Duration::from_secs(2)
            )
            .readiness,
            EndpointReadiness::Serving
        );
        assert!(
            record.inference_verified_at_unix_ms.is_some(),
            "a passing probe is recorded so later checks can skip it"
        );

        assert_eq!(
            managed_service_endpoint_readiness(
                &mut record,
                None,
                Duration::from_secs(2),
                Duration::from_secs(2)
            )
            .readiness,
            EndpointReadiness::Serving
        );

        let requests = server.join().expect("server thread should not panic")?;
        let paths: Vec<&str> = requests
            .iter()
            .filter_map(|request| request.lines().next())
            .collect();
        assert_eq!(
            paths,
            vec![
                "GET /v1/models HTTP/1.1",
                "POST /v1/chat/completions HTTP/1.1",
                "GET /v1/models HTTP/1.1",
            ],
            "the second readiness check must not re-probe inference"
        );
        Ok(())
    }

    #[test]
    fn a_warming_service_is_not_re_probed_on_every_poll() -> Result<()> {
        // Only a successful probe latches, so a model that is listed but still
        // loading would otherwise be re-probed by every poll — and each attempt
        // costs the full probe timeout, paid by `services list` and the dash in
        // front of a user. The second check must cost a listing and nothing more.
        let (port, server) = spawn_canned_http_server(vec![
            Some(("HTTP/1.1 200 OK", MODELS_OK_BODY)),
            Some((
                "HTTP/1.1 503 Service Unavailable",
                r#"{"error":"loading model"}"#,
            )),
            Some(("HTTP/1.1 200 OK", MODELS_OK_BODY)),
        ])?;
        let mut record = probe_test_record(port);

        let first = managed_service_endpoint_readiness(
            &mut record,
            None,
            Duration::from_secs(2),
            Duration::from_secs(2),
        );
        assert_eq!(first.readiness, EndpointReadiness::Listing);
        assert!(
            first.record_changed && record.inference_probe_attempted_at_unix_ms.is_some(),
            "the attempt must be recorded, and persisted by the caller — each CLI \
             run is a fresh process, so an unwritten attempt throttles nothing"
        );

        let second = managed_service_endpoint_readiness(
            &mut record,
            None,
            Duration::from_secs(2),
            Duration::from_secs(2),
        );
        assert_eq!(second.readiness, EndpointReadiness::Listing);
        assert!(!second.record_changed);

        let requests = server.join().expect("server thread should not panic")?;
        let paths: Vec<&str> = requests
            .iter()
            .filter_map(|request| request.lines().next())
            .collect();
        assert_eq!(
            paths,
            vec![
                "GET /v1/models HTTP/1.1",
                "POST /v1/chat/completions HTTP/1.1",
                "GET /v1/models HTTP/1.1",
            ],
            "the second check must not re-probe inside the retry interval"
        );
        Ok(())
    }

    #[test]
    fn restarting_drops_the_previous_runs_inference_verification() {
        // The restarted child is a different server with an unloaded model. If the
        // verification carried over, readiness would short-circuit to "ready" the
        // moment the new server listed the model — the original false positive,
        // reinstated. `refresh_from_engine_state` only ever adopts a verification,
        // so nothing downstream would clear it.
        let mut record = probe_test_record(11435);
        record.inference_verified_at_unix_ms = Some(1);
        record.inference_probe_attempted_at_unix_ms = Some(1);
        record.restart_count = 2;

        record.reset_for_restart();

        assert_eq!(record.inference_verified_at_unix_ms, None);
        assert_eq!(
            record.inference_probe_attempted_at_unix_ms, None,
            "the retry throttle is per-run too; the new child deserves an              immediate first probe"
        );
        assert_eq!(record.restart_count, 3);
        assert!(record.last_restart_unix_ms.is_some());
    }

    #[test]
    fn inference_readiness_is_withheld_while_the_model_only_lists() -> Result<()> {
        // The bug: `/v1/models` answers within seconds while the model loads for
        // minutes and inference returns nothing. That service is not ready.
        let (port, server) = spawn_canned_http_server(vec![
            Some(("HTTP/1.1 200 OK", MODELS_OK_BODY)),
            Some((
                "HTTP/1.1 503 Service Unavailable",
                r#"{"error":"loading model"}"#,
            )),
        ])?;
        let mut record = probe_test_record(port);

        assert_eq!(
            managed_service_endpoint_readiness(
                &mut record,
                None,
                Duration::from_secs(2),
                Duration::from_secs(2)
            )
            .readiness,
            EndpointReadiness::Listing,
            "a listed-but-unservable model is coming up, not dead"
        );
        assert!(
            record.inference_verified_at_unix_ms.is_none(),
            "nothing is latched until inference actually succeeds"
        );

        server.join().expect("server thread should not panic")?;
        Ok(())
    }

    #[test]
    fn inference_probe_sends_the_service_key_to_a_protected_endpoint() -> Result<()> {
        let (port, server) = spawn_canned_http_server(vec![
            Some(("HTTP/1.1 200 OK", MODELS_OK_BODY)),
            Some(("HTTP/1.1 200 OK", CHAT_OK_BODY)),
        ])?;
        let mut record = probe_test_record(port);

        assert_eq!(
            managed_service_endpoint_readiness(
                &mut record,
                Some("test-key"),
                Duration::from_secs(2),
                Duration::from_secs(2)
            )
            .readiness,
            EndpointReadiness::Serving
        );

        let requests = server.join().expect("server thread should not panic")?;
        assert!(
            requests
                .iter()
                .all(|request| request.contains("Authorization: Bearer test-key")),
            "a protected service must not read as unready for want of its own key"
        );
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
    /// tuple with the OpenSSL CLI, cross-checking the pure-Rust sign/verify path
    /// against the same RSASSA-PKCS#1 v1.5 over SHA-256 scheme the installer
    /// (`install.sh`) and release packaging (`cargo xtask package`) use. Returns
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
    fn with_managed_root_keeps_reprovisioning_flat_from_runtime_leaf() {
        let data_root = PathBuf::from("/tmp/rocm-cli-reprovision");
        let paths = AppPaths {
            config_dir: data_root.clone(),
            data_dir: data_root.clone(),
            cache_dir: data_root.join("cache"),
        };
        // A prior install persisted the runtime's own install_root as the managed
        // root; rebasing onto it must recover the canonical data root, not append
        // a second `runtimes/wheel` when the next runtime is provisioned.
        let leaf = data_root
            .join("runtimes")
            .join("wheel")
            .join("release-wheel-gfx942-7-0");
        let rebased = paths.with_managed_root(leaf, false);

        assert_eq!(rebased.data_dir, data_root);
        let next_root = rebased
            .data_dir
            .join("runtimes")
            .join("wheel")
            .join("nightly-wheel-gfx942-7-1");
        // Count path components, not a literal separator, so the assertion holds
        // on Windows too.
        let runtimes_segments = next_root
            .components()
            .filter(|component| component.as_os_str() == std::ffi::OsStr::new("runtimes"))
            .count();
        assert_eq!(runtimes_segments, 1);
    }

    #[test]
    fn app_paths_apply_configured_managed_root_when_unoverridden() -> Result<()> {
        let (root, paths) = temp_app_paths("configured-managed-root");
        let managed_root = root.join("managed");
        let persisted_runtime = managed_root
            .join("runtimes")
            .join("wheel")
            .join("release-wheel-gfx942-7-0");
        fs::create_dir_all(&paths.config_dir)?;
        fs::write(
            paths.config_path(),
            serde_json::to_vec_pretty(&serde_json::json!({
                "setup": { "therock_venv": persisted_runtime }
            }))?,
        )?;

        let discovered = AppPaths::discover_from_paths(paths.clone(), false, false);
        assert_eq!(discovered.config_dir, paths.config_dir);
        assert_eq!(discovered.data_dir, managed_root);
        assert_eq!(discovered.cache_dir, managed_root.join("cache"));

        let data_overridden = AppPaths::discover_from_paths(paths.clone(), true, false);
        assert_eq!(data_overridden.data_dir, paths.data_dir);
        assert_eq!(data_overridden.cache_dir, paths.cache_dir);

        let cache_overridden = AppPaths::discover_from_paths(paths.clone(), false, true);
        assert_eq!(cache_overridden.data_dir, managed_root);
        assert_eq!(cache_overridden.cache_dir, paths.cache_dir);

        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn engine_envs_dir_honors_dedicated_root_override() {
        let (root, paths) = temp_app_paths("engine-envs-root-override");
        let override_root = root.join("runtime").join("engines");

        // Passed in rather than set via `std::env::set_var`: the environment is
        // process-global, so mutating it here would race any concurrent test
        // that reads the same key under a threaded runner.
        //
        // What this assertion covers is the override being consulted and the
        // `<root>/<engine>/envs` shape built on it. It deliberately does NOT
        // claim to cover the host-normalisation step, and routing the expected
        // value through the same call the production code makes is why: on a
        // non-Windows target `normalize_runtime_path_for_host` returns its
        // input unchanged (see `normalize_runtime_path_text_for_platform`) for
        // any path these tests can produce, so dropping normalisation from the
        // override arm cannot fail this on a Linux lane — nor, with an
        // already-normal temp path, on a Windows one. Not the identity in the
        // strict sense: the helper round-trips through `Path::display()`, which
        // is lossy for a non-UTF-8 path. Temp paths here are UTF-8, so the
        // distinction does not reach this assertion, but it is not a no-op.
        // Making it fail would need the platform threaded through the seam, and
        // the normaliser itself is already covered on every host by
        // `runtime_path_normalization_accepts_windows_drive_forms` and its
        // neighbours, which pass the platform in explicitly.
        assert_eq!(
            paths.engine_envs_dir_from("vllm", Some(&override_root)),
            normalize_runtime_path_for_host(&override_root)
                .join("vllm")
                .join("envs")
        );

        fs::remove_dir_all(root).ok();
    }

    /// Serializes tests that replace a process-global env var while they run.
    ///
    /// Named `*_TEST_LOCK` so the env-mutation contract guard recognises the
    /// discipline by suffix rather than by a hardcoded list of lock names.
    static ENGINE_ENVS_ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// The two tests around this one drive the seam, which deliberately does
    /// not read the environment — so on their own the production lookup, and
    /// the key it names, could both be deleted without failing anything. This
    /// drives `engine_envs_dir` itself against a real variable.
    #[test]
    fn engine_envs_dir_reads_its_root_from_the_environment() {
        let _guard = ENGINE_ENVS_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (root, paths) = temp_app_paths("engine-envs-root-env");
        let override_root = root.join("runtime").join("engines");

        // Restored on drop rather than on the next line, so a panic inside
        // `engine_envs_dir` cannot leave this key pointing at the directory
        // removed below. `RestoredEnvVar` only restores; the lock above is what
        // serializes, and the contract guard requires it here because
        // `RestoredEnvVar::set(` is in its mutation list.
        let restore =
            crate::test_env::RestoredEnvVar::set("ROCM_CLI_ENGINE_ENVS_ROOT", &override_root);
        let resolved = paths.engine_envs_dir("vllm");
        drop(restore);

        // Same scope as the seam test above: this pins that `engine_envs_dir`
        // reaches the variable, not that the value is host-normalised on the
        // way through. See that test for why the normalisation step is not
        // observable here.
        assert_eq!(
            resolved,
            normalize_runtime_path_for_host(&override_root)
                .join("vllm")
                .join("envs"),
            "engine_envs_dir must reach $ROCM_CLI_ENGINE_ENVS_ROOT"
        );

        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn engine_envs_dir_falls_back_to_the_data_dir_without_an_override() {
        // The other half of the override contract: with nothing supplied the
        // root is the data dir, which is what the unset-environment production
        // path resolves to.
        let (root, paths) = temp_app_paths("engine-envs-root-default");

        assert_eq!(
            paths.engine_envs_dir_from("vllm", None),
            paths.data_dir.join("engines").join("vllm").join("envs")
        );

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
    #[allow(clippy::float_cmp)]
    fn dashboard_tui_inference_params_round_trip_and_default_absent() {
        // Defaults leave the sampling knobs unset and out of the serialized JSON
        // (skip_serializing_if), so a stock config carries no sampling override.
        let default_json = serde_json::to_value(DashboardTuiConfig::default()).unwrap();
        assert!(default_json.get("chat_temperature").is_none());
        assert!(default_json.get("chat_top_p").is_none());
        assert!(default_json.get("chat_max_tokens").is_none());

        // When set, all three round-trip through JSON unchanged.
        let cfg = DashboardTuiConfig {
            chat_temperature: Some(0.25),
            chat_top_p: Some(0.5),
            chat_max_tokens: Some(512),
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: DashboardTuiConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.chat_temperature, Some(0.25));
        assert_eq!(back.chat_top_p, Some(0.5));
        assert_eq!(back.chat_max_tokens, Some(512));
    }

    #[test]
    fn dashboard_tui_rejects_invalid_inference_params() {
        for (field, value, expected) in [
            ("chat_temperature", "-0.1", "chat_temperature"),
            ("chat_top_p", "1.1", "chat_top_p"),
            ("chat_max_tokens", "0", "chat_max_tokens"),
        ] {
            let json = format!(r#"{{"{field}":{value}}}"#);
            let error = serde_json::from_str::<DashboardTuiConfig>(&json)
                .expect_err("invalid inference parameter must be rejected")
                .to_string();
            assert!(
                error.contains(expected),
                "unexpected error for {field}: {error}"
            );
        }
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
}
