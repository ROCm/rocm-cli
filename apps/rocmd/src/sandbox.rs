// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use crate::ARTIFACT_PREFETCH_TIMEOUT;
use crate::cli::{SandboxToolArg, SandboxToolPolicy};
use crate::common::{self, CommandCapture};
use crate::persistence::load_managed_services;
use crate::service::stop_managed_service;
use crate::watchers::restart_managed_service;
use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, AuditEventRecord, ExamineSummary, ModelRecipeArtifactRecord, append_audit_event,
    model_artifact_cache_status, resolve_model_recipe_artifact, unix_time_millis,
};
use serde_json::Value;
use serde_json::json;
#[cfg(test)]
use sha2::Digest;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
#[cfg(target_os = "linux")]
use std::process::{Command as ProcessCommand, Stdio};
#[cfg(any(test, target_os = "linux"))]
use std::thread;
use std::time::Duration;

pub(crate) fn run_sandbox_runner(
    paths: &AppPaths,
    tool: SandboxToolArg,
    service_id: Option<String>,
    artifact_ref: Option<String>,
    message: Option<String>,
    allow_native_fallback: bool,
    policy: SandboxToolPolicy,
) -> Result<Value> {
    #[cfg(target_os = "linux")]
    {
        if command_available("bwrap") {
            return run_bubblewrap_sandbox(paths, tool, service_id, artifact_ref, message, policy);
        }
    }

    if allow_native_fallback {
        return run_native_restricted_sandbox(
            paths,
            tool,
            service_id,
            artifact_ref,
            message,
            policy,
        );
    }

    bail!(
        "isolated sandbox runner is unavailable on this host; pass --allow-native-fallback to run only the restricted internal tool API"
    )
}

#[cfg(target_os = "linux")]
fn run_bubblewrap_sandbox(
    paths: &AppPaths,
    tool: SandboxToolArg,
    service_id: Option<String>,
    artifact_ref: Option<String>,
    message: Option<String>,
    policy: SandboxToolPolicy,
) -> Result<Value> {
    paths.ensure()?;

    let current_exe = std::env::current_exe().context("failed to resolve rocmd executable")?;
    let exe_dir = current_exe
        .parent()
        .context("rocmd executable has no parent directory")?
        .to_path_buf();

    let mut command = ProcessCommand::new("bwrap");
    command
        .arg("--die-with-parent")
        .arg("--new-session")
        .arg("--unshare-ipc")
        .arg("--unshare-uts")
        .arg("--proc")
        .arg("/proc")
        .arg("--dev")
        .arg("/dev")
        .arg("--tmpfs")
        .arg("/tmp")
        .arg("--tmpfs")
        .arg("/run");
    if !(matches!(tool, SandboxToolArg::PrefetchArtifact) && policy.allow_artifact_download) {
        command.arg("--unshare-net");
    }

    for path in ["/usr", "/bin", "/lib", "/lib64", "/etc"] {
        let path = std::path::Path::new(path);
        if path.exists() {
            command.arg("--ro-bind").arg(path).arg(path);
        }
    }

    command.arg("--ro-bind").arg(&exe_dir).arg(&exe_dir);
    bind_app_path_for_sandbox(&mut command, &paths.config_dir, false)?;
    bind_app_path_for_sandbox(&mut command, &paths.data_dir, tool.writes_data())?;
    bind_app_path_for_sandbox(&mut command, &paths.cache_dir, tool.writes_cache())?;
    append_sandbox_tool_command_args(
        &mut command,
        &current_exe,
        tool,
        service_id.as_deref(),
        artifact_ref.as_deref(),
        message.as_deref(),
        policy,
    );

    let output = run_process_with_timeout(command, Duration::from_mins(1))?;
    let value = parse_sandbox_child_output(output, "bubblewrap sandbox")?;
    record_sandbox_audit(paths, tool, "bubblewrap", true, service_id.as_deref())?;
    Ok(sandbox_report(tool, "bubblewrap", value))
}

#[cfg(target_os = "linux")]
fn bind_app_path_for_sandbox(
    command: &mut ProcessCommand,
    path: &std::path::Path,
    writable: bool,
) -> Result<()> {
    fs::create_dir_all(path).with_context(|| format!("failed to create {}", path.display()))?;
    if writable {
        command.arg("--bind");
    } else {
        command.arg("--ro-bind");
    }
    command.arg(path).arg(path);
    Ok(())
}

fn run_native_restricted_sandbox(
    paths: &AppPaths,
    tool: SandboxToolArg,
    service_id: Option<String>,
    artifact_ref: Option<String>,
    message: Option<String>,
    policy: SandboxToolPolicy,
) -> Result<Value> {
    let value = run_sandbox_tool(
        paths,
        tool,
        service_id.clone(),
        artifact_ref,
        message,
        policy,
    )?;
    record_sandbox_audit(
        paths,
        tool,
        "native_restricted",
        true,
        service_id.as_deref(),
    )?;
    Ok(sandbox_report(tool, "native_restricted", value))
}

pub(crate) fn run_sandbox_tool(
    paths: &AppPaths,
    tool: SandboxToolArg,
    service_id: Option<String>,
    artifact_ref: Option<String>,
    message: Option<String>,
    policy: SandboxToolPolicy,
) -> Result<Value> {
    match tool {
        SandboxToolArg::CheckUpdates => {
            let output =
                common::run_rocm_capture_for_paths(paths, &["update"], Duration::from_mins(1))?;
            Ok(sandbox_check_updates_value(output))
        }
        SandboxToolArg::DriverPlan => {
            let output = common::run_rocm_capture_for_paths(
                paths,
                &["install", "driver", "--dkms", "--dry-run"],
                Duration::from_mins(1),
            )?;
            Ok(sandbox_driver_plan_value(output))
        }
        SandboxToolArg::ExamineSnapshot => {
            let examine = ExamineSummary::gather()?;
            Ok(json!({
                "tool": tool.as_cli_value(),
                "status": "captured",
                "mutating": false,
                "examine": examine,
            }))
        }
        SandboxToolArg::ListServers => {
            let services = load_managed_services(paths)?;
            Ok(json!({
                "tool": tool.as_cli_value(),
                "status": "listed",
                "mutating": false,
                "count": services.len(),
                "services": services,
            }))
        }
        SandboxToolArg::RestartServer => {
            let service_id = service_id.context("restart_server requires `--service-id`")?;
            let mut record = load_managed_services(paths)?
                .into_iter()
                .find(|record| record.service_id == service_id)
                .with_context(|| format!("managed service `{service_id}` not found"))?;
            restart_managed_service(paths, &mut record)?;
            Ok(json!({
                "tool": tool.as_cli_value(),
                "status": "restarted",
                "mutating": true,
                "service": record,
            }))
        }
        SandboxToolArg::StopServer => {
            let service_id = service_id.context("stop_server requires `--service-id`")?;
            let stopped = stop_managed_service(paths, &service_id)?;
            Ok(json!({
                "tool": tool.as_cli_value(),
                "status": "stopped",
                "mutating": true,
                "result": stopped,
            }))
        }
        SandboxToolArg::PrefetchArtifact => {
            let artifact_ref =
                artifact_ref.context("prefetch_artifact requires `--artifact-ref`")?;
            let resolved = resolve_model_recipe_artifact(&artifact_ref)?.with_context(|| {
                format!("artifact_ref `{artifact_ref}` was not found in the model recipe registry")
            })?;
            let (recipe, artifact) = resolved;
            prefetch_artifact_value_with_policy(
                paths,
                &artifact_ref,
                &recipe.canonical_model_id,
                artifact,
                policy,
            )
        }
        SandboxToolArg::NotifyUser => {
            let message = message.unwrap_or_else(|| "sandbox notification".to_owned());
            record_notification_audit(paths, "sandbox:notify_user", "notify_user", None, &message)?;
            Ok(json!({
                "tool": tool.as_cli_value(),
                "status": "notified",
                "mutating": false,
                "message": message,
                "notification_recorded": true,
            }))
        }
    }
}

pub(crate) fn record_notification_audit(
    paths: &AppPaths,
    actor: &str,
    action: &str,
    watcher_id: Option<&str>,
    message: &str,
) -> Result<()> {
    append_audit_event(
        paths,
        &AuditEventRecord {
            at_unix_ms: unix_time_millis(),
            source: "rocmd".to_owned(),
            category: "notification".to_owned(),
            actor: actor.to_owned(),
            level: "info".to_owned(),
            action: action.to_owned(),
            message: message.to_owned(),
            watcher_id: watcher_id.map(ToOwned::to_owned),
            service_id: None,
        },
    )
}

fn prefetch_artifact_value_with_policy(
    paths: &AppPaths,
    artifact_ref: &str,
    model: &str,
    artifact: ModelRecipeArtifactRecord,
    policy: SandboxToolPolicy,
) -> Result<Value> {
    let cache = model_artifact_cache_status(paths, model, &artifact);
    if !policy.allow_artifact_download {
        return Ok(json!({
            "tool": SandboxToolArg::PrefetchArtifact.as_cli_value(),
            "artifact_ref": artifact_ref,
            "model": model,
            "artifact": artifact,
            "cache": cache,
            "status": "source_policy_required",
            "mutating": false,
            "network_used": false,
            "message": "artifact prefetch requires an approved source policy before network access; no artifact bytes were downloaded",
        }));
    }

    if cache.marker_path.is_file() {
        return Ok(json!({
            "tool": SandboxToolArg::PrefetchArtifact.as_cli_value(),
            "artifact_ref": artifact_ref,
            "model": model,
            "artifact": artifact,
            "cache": cache,
            "status": "cached",
            "mutating": false,
            "network_used": false,
            "message": "artifact cache marker already exists; no network request was made",
        }));
    }

    if let Some(message) = declared_source_policy_block_message(&artifact) {
        return Ok(prefetch_blocked_value(
            artifact_ref,
            model,
            artifact,
            cache,
            &message,
        ));
    }
    let source_policy_requires_huggingface_auth =
        artifact_declares_huggingface_authenticated_policy(&artifact);
    let artifact_kind = artifact.kind.to_ascii_lowercase();
    let huggingface_artifact = artifact_is_huggingface(&artifact);
    if artifact.gated.unwrap_or(false) && !huggingface_artifact {
        return Ok(prefetch_blocked_value(
            artifact_ref,
            model,
            artifact,
            cache,
            "gated artifacts require a source-specific authentication policy before download",
        ));
    }
    if !matches!(artifact_kind.as_str(), "url" | "http" | "https" | "direct")
        && !huggingface_artifact
    {
        return Ok(prefetch_blocked_value(
            artifact_ref,
            model,
            artifact,
            cache,
            "approved live prefetch currently supports direct HTTP(S) artifacts only",
        ));
    }
    if !artifact.uri.starts_with("https://") && !artifact.uri.starts_with("http://") {
        return Ok(prefetch_blocked_value(
            artifact_ref,
            model,
            artifact,
            cache,
            if huggingface_artifact {
                "authenticated Hugging Face prefetch requires an HTTP(S) artifact URI in the signed recipe metadata"
            } else {
                "approved live prefetch requires an HTTP(S) artifact URI"
            },
        ));
    }
    let mut request_headers = Vec::<(&str, String)>::new();
    let source_policy = if huggingface_artifact
        && (artifact.gated.unwrap_or(false)
            || policy.allow_huggingface_download
            || source_policy_requires_huggingface_auth)
    {
        if !policy.allow_huggingface_download {
            return Ok(prefetch_blocked_value(
                artifact_ref,
                model,
                artifact,
                cache,
                "Hugging Face artifacts require --allow-huggingface-download before rocm-cli may use an authentication token",
            ));
        }
        if !artifact.uri.starts_with("https://") {
            return Ok(prefetch_blocked_value(
                artifact_ref,
                model,
                artifact,
                cache,
                "rocm-cli will not send a Hugging Face token over plain HTTP; use an HTTPS Hugging Face artifact URI",
            ));
        }
        if !is_huggingface_url(&artifact.uri) {
            return Ok(prefetch_blocked_value(
                artifact_ref,
                model,
                artifact,
                cache,
                "rocm-cli will not send a Hugging Face token to a non-Hugging Face URL",
            ));
        }
        let Some(token) = policy
            .huggingface_token
            .as_deref()
            .map(str::trim)
            .filter(|token| !token.is_empty())
        else {
            return Ok(prefetch_blocked_value(
                artifact_ref,
                model,
                artifact,
                cache,
                "Hugging Face artifact prefetch needs ROCM_CLI_HUGGINGFACE_TOKEN, HF_TOKEN, or HUGGING_FACE_HUB_TOKEN",
            ));
        };
        request_headers.push(("Authorization", format!("Bearer {token}")));
        "huggingface_authenticated"
    } else {
        "explicit_allow_artifact_download"
    };
    let Some(expected_sha256) = artifact
        .sha256
        .as_deref()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| value.len() == 64 && value.chars().all(|ch| ch.is_ascii_hexdigit()))
    else {
        return Ok(prefetch_blocked_value(
            artifact_ref,
            model,
            artifact,
            cache,
            "approved live prefetch requires a valid sha256 in the recipe artifact metadata",
        ));
    };
    let Some(size_bytes) = artifact.size_bytes else {
        return Ok(prefetch_blocked_value(
            artifact_ref,
            model,
            artifact,
            cache,
            "approved live prefetch requires size_bytes in the recipe artifact metadata",
        ));
    };
    let max_bytes = policy.artifact_max_bytes.unwrap_or(size_bytes);
    if size_bytes > max_bytes {
        return Ok(prefetch_blocked_value(
            artifact_ref,
            model,
            artifact,
            cache,
            "artifact size exceeds the approved download byte limit",
        ));
    }

    // Streamed straight to its final path: model artifacts run to gigabytes, so
    // buffering one to hash it would hold the whole file in memory. The size and
    // digest from the recipe are enforced inside the download, which also gives
    // this path a free-space preflight and resume on a dropped connection.
    let artifact_path = artifact_bytes_path_for_marker(&cache.marker_path);
    let mut headers: Vec<(&str, &str)> = vec![("User-Agent", "rocm-cli")];
    headers.extend(
        request_headers
            .iter()
            .map(|(name, value)| (*name, value.as_str())),
    );
    let outcome = rocm_core::download_file_streaming(&rocm_core::DownloadRequest {
        url: &artifact.uri,
        destination: &artifact_path,
        timeout: ARTIFACT_PREFETCH_TIMEOUT,
        headers: &headers,
        max_bytes: Some(max_bytes),
        expected_len: Some(size_bytes),
        expected_sha256: Some(&expected_sha256),
    })
    .with_context(|| format!("failed to prefetch artifact `{artifact_ref}`"))?;
    let actual_sha256 = outcome.sha256;
    write_file_atomically(
        &cache.marker_path,
        &serde_json::to_vec_pretty(&json!({
            "artifact_ref": artifact_ref,
            "model": model,
            "artifact": artifact,
            "bytes_path": artifact_path,
            "size_bytes": size_bytes,
            "sha256": actual_sha256,
            "prefetched_at_unix_ms": unix_time_millis(),
            "source_policy": source_policy,
        }))
        .context("failed to serialize artifact cache marker")?,
    )?;
    let cache = model_artifact_cache_status(paths, model, &artifact);
    Ok(json!({
        "tool": SandboxToolArg::PrefetchArtifact.as_cli_value(),
        "artifact_ref": artifact_ref,
        "model": model,
        "artifact": artifact,
        "cache": cache,
        "status": "prefetched",
        "mutating": true,
        "network_used": true,
        "bytes_path": artifact_path,
        "size_bytes": size_bytes,
        "sha256": actual_sha256,
        "source_policy": source_policy,
        "message": "artifact downloaded and verified with recipe sha256",
    }))
}

fn declared_source_policy_block_message(artifact: &ModelRecipeArtifactRecord) -> Option<String> {
    let source_policy = artifact.source_policy.as_ref()?;
    if !source_policy.required_hosts.is_empty() {
        let Some(host) = http_url_host(&artifact.uri) else {
            return Some(
                "recipe source policy declares required hosts but the artifact URI is not HTTP(S)"
                    .to_owned(),
            );
        };
        if !source_policy
            .required_hosts
            .iter()
            .any(|required| required.eq_ignore_ascii_case(&host))
        {
            return Some(format!(
                "artifact host `{host}` is not allowed by the recipe source policy"
            ));
        }
    }

    match source_policy.policy.as_str() {
        "direct_https_sha256" => {
            if artifact.uri.starts_with("https://") {
                None
            } else {
                Some(
                    "recipe source policy direct_https_sha256 requires an HTTPS artifact URI"
                        .to_owned(),
                )
            }
        }
        "huggingface_public" => {
            if artifact.gated.unwrap_or(false) {
                Some(
                    "recipe source policy marks this as public Hugging Face, but the artifact is gated"
                        .to_owned(),
                )
            } else {
                huggingface_policy_uri_block_message(artifact, "huggingface_public")
            }
        }
        "huggingface_authenticated" => {
            huggingface_policy_uri_block_message(artifact, "huggingface_authenticated")
        }
        "manual_only" => Some(
            "recipe source policy marks this artifact as manual-only; rocm-cli will not download it"
                .to_owned(),
        ),
        other => Some(format!(
            "recipe source policy `{other}` is not supported by this rocm-cli build"
        )),
    }
}

fn huggingface_policy_uri_block_message(
    artifact: &ModelRecipeArtifactRecord,
    policy: &str,
) -> Option<String> {
    if !artifact.uri.starts_with("https://") {
        return Some(format!(
            "recipe source policy {policy} requires an HTTPS Hugging Face artifact URI"
        ));
    }
    if !is_huggingface_url(&artifact.uri) {
        return Some(format!(
            "recipe source policy {policy} requires a Hugging Face artifact URI"
        ));
    }
    None
}

fn artifact_declares_huggingface_authenticated_policy(
    artifact: &ModelRecipeArtifactRecord,
) -> bool {
    artifact
        .source_policy
        .as_ref()
        .is_some_and(|policy| policy.policy == "huggingface_authenticated")
}

fn artifact_is_huggingface(artifact: &ModelRecipeArtifactRecord) -> bool {
    let kind = artifact.kind.to_ascii_lowercase();
    matches!(kind.as_str(), "huggingface" | "hf" | "hugging_face")
        || is_huggingface_url(&artifact.uri)
}

fn is_huggingface_url(url: &str) -> bool {
    http_url_host(url).is_some_and(|host| {
        host == "huggingface.co"
            || host.ends_with(".huggingface.co")
            || host == "hf.co"
            || host.ends_with(".hf.co")
    })
}

fn http_url_host(url: &str) -> Option<String> {
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
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

fn prefetch_blocked_value(
    artifact_ref: &str,
    model: &str,
    artifact: ModelRecipeArtifactRecord,
    cache: rocm_core::ModelArtifactCacheStatus,
    reason: &str,
) -> Value {
    json!({
        "tool": SandboxToolArg::PrefetchArtifact.as_cli_value(),
        "artifact_ref": artifact_ref,
        "model": model,
        "artifact": artifact,
        "cache": cache,
        "status": "blocked",
        "mutating": false,
        "network_used": false,
        "message": reason,
    })
}

fn artifact_bytes_path_for_marker(marker_path: &Path) -> PathBuf {
    marker_path.with_extension("bin")
}

/// Hex digest of a buffer. Production hashing happens incrementally inside the
/// streaming download; this exists only so tests can state an expected digest.
#[cfg(test)]
fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", sha2::Sha256::digest(bytes))
}

const ATOMIC_WRITE_TEMP_ATTEMPTS: u32 = 128;

fn temp_sibling_path(path: &Path, suffix: &OsStr) -> Result<PathBuf> {
    let parent = path.parent().context("file path has no parent directory")?;
    let mut file_name = path
        .file_name()
        .context("file path has no file name")?
        .to_os_string();
    file_name.push(".tmp-");
    file_name.push(suffix);
    Ok(parent.join(file_name))
}

/// Stage-and-publish a file here, sharing only the publish step with `rocm-core`.
///
/// Deliberately not [`rocm_core::write_file_atomically`]: only the
/// Windows-sensitive publish (`ReplaceFileW` and its fallbacks) is
/// single-sourced. The staging half stays local because it carries the
/// `suffix_for_attempt` and `before_publish` seams the tests drive. This path
/// does not `sync_all` before publishing, so it is atomic but carries no
/// crash-durability guarantee for the staged bytes.
fn write_file_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let temp_id = format!("{}-{}", std::process::id(), unix_time_millis());
    write_file_atomically_with(
        path,
        bytes,
        |attempt| OsString::from(format!("{temp_id}-{attempt}")),
        || {},
    )
}

fn write_file_atomically_with<S, P>(
    path: &Path,
    bytes: &[u8],
    suffix_for_attempt: S,
    before_publish: P,
) -> Result<()>
where
    S: FnMut(u32) -> OsString,
    P: FnOnce(),
{
    write_file_atomically_with_publish(
        path,
        bytes,
        suffix_for_attempt,
        before_publish,
        publish_temp_file,
    )
}

fn write_file_atomically_with_publish<S, P, F>(
    path: &Path,
    bytes: &[u8],
    mut suffix_for_attempt: S,
    before_publish: P,
    publish: F,
) -> Result<()>
where
    S: FnMut(u32) -> OsString,
    P: FnOnce(),
    F: FnOnce(&Path, &Path) -> io::Result<()>,
{
    let parent = path.parent().context("file path has no parent directory")?;
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;

    let mut reserved = None;
    for attempt in 0..ATOMIC_WRITE_TEMP_ATTEMPTS {
        let suffix = suffix_for_attempt(attempt);
        let tmp = temp_sibling_path(path, &suffix)?;
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
        {
            Ok(file) => {
                reserved = Some((tmp, file));
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(error).with_context(|| format!("failed to create {}", tmp.display()));
            }
        }
    }
    let Some((tmp, mut file)) = reserved else {
        bail!(
            "failed to reserve a temporary file next to {} after {} attempts",
            path.display(),
            ATOMIC_WRITE_TEMP_ATTEMPTS
        );
    };

    if let Err(error) = file.write_all(bytes) {
        drop(file);
        let _ = fs::remove_file(&tmp);
        return Err(error).with_context(|| format!("failed to write {}", tmp.display()));
    }
    drop(file);
    before_publish();

    publish(&tmp, path).inspect_err(|_| {
        let _ = fs::remove_file(&tmp);
    })?;
    Ok(())
}

/// The publish step lives in `rocm-core` so there is one implementation of the
/// Windows `ReplaceFileW` handling for the whole workspace.
fn publish_temp_file(tmp: &Path, path: &Path) -> io::Result<()> {
    rocm_core::publish_temp_file(tmp, path)
}

pub(crate) fn sandbox_check_updates_value(output: CommandCapture) -> Value {
    let status = update_check_status(&output);
    let update_available =
        output.exit_status == 0 && update_output_reports_update_available(&output.stdout);
    let message = common::update_check_message(status);
    json!({
        "tool": SandboxToolArg::CheckUpdates.as_cli_value(),
        "status": status,
        "update_available": update_available,
        "mutating": false,
        "message": message,
        "argv": output.argv,
        "exit_status": output.exit_status,
        "stdout": output.stdout,
        "stderr": output.stderr,
    })
}

fn update_check_status(output: &CommandCapture) -> &'static str {
    if output.exit_status != 0 {
        "error"
    // A newer version outranks a composition repair: reporting the repair while
    // some runtime is a whole release behind would understate the tree.
    } else if update_output_reports_status(&output.stdout, "update_available") {
        "update_available"
    } else if update_output_reports_status(&output.stdout, "repair_available") {
        "repair_available"
    } else {
        "checked"
    }
}

fn update_output_reports_status(stdout: &str, status: &str) -> bool {
    let expected = format!("status={status}");
    stdout.split_whitespace().any(|part| part == expected)
}

fn update_output_reports_update_available(stdout: &str) -> bool {
    update_output_reports_status(stdout, "update_available")
        || update_output_reports_status(stdout, "repair_available")
        || stdout
            .split_whitespace()
            .any(|part| part == "update_available=true")
}

pub(crate) fn sandbox_driver_plan_value(output: CommandCapture) -> Value {
    let status = if output.exit_status == 0 {
        "planned"
    } else {
        "error"
    };
    json!({
        "tool": SandboxToolArg::DriverPlan.as_cli_value(),
        "status": status,
        "mutating": false,
        "message": "ran read-only `rocm install driver --dkms --dry-run`; no driver commands were executed",
        "argv": output.argv,
        "exit_status": output.exit_status,
        "stdout": output.stdout,
        "stderr": output.stderr,
    })
}

fn sandbox_report(tool: SandboxToolArg, isolation: &str, output: Value) -> Value {
    json!({
        "protocol": "rocmd-sandbox-run-v0",
        "tool": tool.as_cli_value(),
        "isolation": isolation,
        "ok": true,
        "ok_meaning": "sandbox wrapper completed; inspect output.status and output.exit_status for the restricted tool result",
        "output": output,
    })
}

fn record_sandbox_audit(
    paths: &AppPaths,
    tool: SandboxToolArg,
    isolation: &str,
    ok: bool,
    service_id: Option<&str>,
) -> Result<()> {
    append_audit_event(
        paths,
        &AuditEventRecord {
            at_unix_ms: unix_time_millis(),
            source: "rocmd".to_owned(),
            category: "sandbox".to_owned(),
            actor: "sandbox-runner".to_owned(),
            level: if ok { "info" } else { "error" }.to_owned(),
            action: tool.as_cli_value().to_owned(),
            message: format!(
                "sandbox tool `{}` completed with isolation `{isolation}`",
                tool.as_cli_value()
            ),
            watcher_id: None,
            service_id: service_id.map(str::to_owned),
        },
    )
}

#[cfg(target_os = "linux")]
fn append_sandbox_tool_command_args(
    command: &mut ProcessCommand,
    rocmd_binary: &std::path::Path,
    tool: SandboxToolArg,
    service_id: Option<&str>,
    artifact_ref: Option<&str>,
    message: Option<&str>,
    policy: SandboxToolPolicy,
) {
    command
        .arg("--")
        .arg(rocmd_binary)
        .arg("sandbox-tool")
        .arg(tool.as_cli_value());
    if let Some(service_id) = service_id {
        command.arg("--service-id").arg(service_id);
    }
    if let Some(artifact_ref) = artifact_ref {
        command.arg("--artifact-ref").arg(artifact_ref);
    }
    if policy.allow_artifact_download {
        command.arg("--allow-artifact-download");
    }
    if policy.allow_huggingface_download {
        command.arg("--allow-huggingface-download");
    }
    if let Some(max_bytes) = policy.artifact_max_bytes {
        command
            .arg("--artifact-max-bytes")
            .arg(max_bytes.to_string());
    }
    if let Some(message) = message {
        command.arg("--message").arg(message);
    }
}

#[cfg(target_os = "linux")]
fn run_process_with_timeout(
    mut command: ProcessCommand,
    timeout: Duration,
) -> Result<std::process::Output> {
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to spawn sandbox process")?;
    let started = std::time::Instant::now();
    loop {
        if child
            .try_wait()
            .context("failed to poll sandbox process")?
            .is_some()
        {
            return child
                .wait_with_output()
                .context("failed to collect sandbox process output");
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let output = child
                .wait_with_output()
                .context("failed to collect timed-out sandbox process output")?;
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            bail!(
                "sandbox process exceeded {}s timeout: {}",
                timeout.as_secs(),
                if !stderr.is_empty() {
                    stderr
                } else if !stdout.is_empty() {
                    stdout
                } else {
                    "no output".to_owned()
                }
            );
        }
        thread::sleep(Duration::from_millis(50));
    }
}

#[cfg(target_os = "linux")]
fn parse_sandbox_child_output(output: std::process::Output, label: &str) -> Result<Value> {
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !output.status.success() {
        bail!(
            "{label} failed: {}",
            if !stderr.is_empty() {
                stderr
            } else if !stdout.is_empty() {
                stdout
            } else {
                format!("exit status {}", output.status)
            }
        );
    }
    serde_json::from_str(&stdout).with_context(|| format!("failed to parse {label} json output"))
}

#[cfg(target_os = "linux")]
fn command_available(name: &str) -> bool {
    ProcessCommand::new(name)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_app_paths;
    use std::io::Read;

    fn serve_one_http_response(bytes: Vec<u8>) -> Result<String> {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0))?;
        let addr = listener.local_addr()?;
        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut request = [0_u8; 1024];
                let _ = stream.read(&mut request);
                let header = format!(
                    "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    bytes.len()
                );
                let _ = stream.write_all(header.as_bytes());
                let _ = stream.write_all(&bytes);
            }
        });
        Ok(format!("http://{addr}/artifact.bin"))
    }

    /// Regression: a failed write must not leave a `.tmp-*` scratch file
    /// behind. The name is unique per attempt, so before this an orphan
    /// accumulated per retry — and when the failure is a full disk, those
    /// orphans are exactly what keeps it full.
    ///
    /// Provokes the failure by pointing the destination at a non-empty
    /// directory: the temp file is written, then neither the rename nor the
    /// replace fallback can succeed. Portable, unlike an out-of-space test.
    #[test]
    fn write_file_atomically_cleans_up_temp_when_the_rename_fails() {
        let root = std::env::temp_dir().join(format!(
            "rocmd-atomic-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        let occupied = root.join("manifest.json");
        fs::create_dir_all(occupied.join("nested")).unwrap();
        fs::write(occupied.join("nested").join("keep"), b"x").unwrap();

        write_file_atomically(&occupied, b"payload")
            .expect_err("renaming onto a non-empty directory should fail");

        let leftovers: Vec<String> = fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp-"))
            .collect();
        let _ = fs::remove_dir_all(&root);
        assert!(
            leftovers.is_empty(),
            "failed write left temp files behind: {leftovers:?}"
        );
    }

    /// The temp name keeps every extension, so a cleanup sweep over a cache
    /// directory can still tell what a leftover was going to be.
    #[test]
    fn write_file_atomically_temp_name_preserves_multi_dot_file_names() {
        let root = std::env::temp_dir().join(format!(
            "rocmd-atomic-name-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        let occupied = root.join("artifact.tar.gz");
        fs::create_dir_all(occupied.join("nested")).unwrap();
        fs::write(occupied.join("nested").join("keep"), b"x").unwrap();

        // Fails after the temp file exists, so the observed name is the real one.
        write_file_atomically(&occupied, b"payload").expect_err("rename should fail");

        let names: Vec<String> = fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        let _ = fs::remove_dir_all(&root);
        assert_eq!(
            names,
            vec!["artifact.tar.gz".to_owned()],
            "only the occupied destination should remain"
        );
    }

    #[test]
    fn concurrent_atomic_writes_do_not_remove_a_published_destination() {
        let root = std::env::temp_dir().join(format!(
            "rocmd-atomic-collision-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("manifest.json");
        let before_publish = std::sync::Arc::new(std::sync::Barrier::new(2));

        let writers: Vec<_> = [b"first".as_slice(), b"second".as_slice()]
            .into_iter()
            .enumerate()
            .map(|(writer, bytes)| {
                let destination = destination.clone();
                let before_publish = std::sync::Arc::clone(&before_publish);
                std::thread::spawn(move || {
                    write_file_atomically_with(
                        &destination,
                        bytes,
                        |attempt| {
                            if attempt == 0 {
                                OsString::from("same-millisecond")
                            } else {
                                OsString::from(format!("same-millisecond-{writer}-{attempt}"))
                            }
                        },
                        || {
                            before_publish.wait();
                        },
                    )
                })
            })
            .collect();

        for writer in writers {
            writer.join().unwrap().unwrap();
        }
        let published = fs::read(&destination).expect("a writer must remain published");
        let _ = fs::remove_dir_all(&root);
        assert!(published == b"first" || published == b"second");
    }

    #[test]
    fn failed_atomic_replace_preserves_destination_and_cleans_temp() {
        let root = std::env::temp_dir().join(format!(
            "rocmd-atomic-replace-failure-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("manifest.json");
        fs::write(&destination, b"published").unwrap();

        write_file_atomically_with_publish(
            &destination,
            b"replacement",
            |attempt| OsString::from(format!("replace-failure-{attempt}")),
            || {},
            |tmp, path| {
                assert_eq!(fs::read(tmp).unwrap(), b"replacement");
                assert_eq!(path, destination);
                Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "simulated atomic replacement failure",
                ))
            },
        )
        .expect_err("simulated replacement failure must be returned");

        assert_eq!(fs::read(&destination).unwrap(), b"published");
        let leftovers: Vec<_> = fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        let _ = fs::remove_dir_all(&root);
        assert_eq!(leftovers, vec![OsString::from("manifest.json")]);
    }

    #[cfg(unix)]
    #[test]
    fn atomic_temp_name_preserves_non_unicode_file_name_bytes() {
        use std::os::unix::ffi::{OsStrExt, OsStringExt};

        let file_name = OsString::from_vec(b"manifest-\xff.json".to_vec());
        let destination = Path::new("/tmp").join(&file_name);
        let temp = temp_sibling_path(&destination, std::ffi::OsStr::new("collision")).unwrap();

        let mut expected = file_name.into_vec();
        expected.extend_from_slice(b".tmp-collision");
        assert_eq!(temp.file_name().unwrap().as_bytes(), expected);
    }

    /// Mirrors the `/dev/shm` reproduction from the original report: a genuine
    /// ENOSPC, not a rename failure standing in for one.
    ///
    /// Ignored by default because it fills `/dev/shm`, which is shared with
    /// anything else on the host, so it is not safe to run concurrently.
    #[test]
    #[ignore = "fills /dev/shm to provoke ENOSPC; not safe to run concurrently"]
    fn write_file_atomically_cleans_up_temp_on_write_failure() {
        let shm = Path::new("/dev/shm");
        if !shm.is_dir() {
            eprintln!("skipping: /dev/shm unavailable");
            return;
        }
        let dir = shm.join(format!("rocmd-enospc-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("manifest.json");
        // Larger than the tmpfs, so the write is guaranteed to hit ENOSPC.
        let payload = vec![b'x'; 256 * 1024 * 1024];

        write_file_atomically(&dest, &payload)
            .expect_err("writing past the end of the filesystem should fail");
        let leftovers: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        let destination_exists = dest.exists();
        let _ = fs::remove_dir_all(&dir);

        assert!(
            leftovers.is_empty(),
            "failed write left files behind: {leftovers:?}"
        );
        assert!(!destination_exists);
    }

    #[test]
    fn sandbox_check_updates_value_is_read_only_and_preserves_output() {
        let value = sandbox_check_updates_value(CommandCapture {
            argv: vec!["rocm".to_owned(), "update".to_owned()],
            exit_status: 0,
            stdout: "update\n  runtime release status=up_to_date\n".to_owned(),
            stderr: String::new(),
        });

        assert_eq!(
            value.get("tool").and_then(Value::as_str),
            Some("check_updates")
        );
        assert_eq!(value.get("status").and_then(Value::as_str), Some("checked"));
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("no updates were applied"))
        );
        assert!(
            value
                .get("stdout")
                .and_then(Value::as_str)
                .is_some_and(|stdout| stdout.contains("status=up_to_date"))
        );
    }

    #[test]
    fn sandbox_check_updates_value_marks_runtime_update_available() {
        let value = sandbox_check_updates_value(CommandCapture {
            argv: vec!["rocm".to_owned(), "update".to_owned()],
            exit_status: 0,
            stdout: "update\n  runtime release-pip-gfx120x-all status=update_available installed=7.13.0 latest=7.14.0\n".to_owned(),
            stderr: String::new(),
        });

        assert_eq!(
            value.get("tool").and_then(Value::as_str),
            Some("check_updates")
        );
        assert_eq!(
            value.get("status").and_then(Value::as_str),
            Some("update_available")
        );
        assert_eq!(
            value.get("update_available").and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(
                    |message| message.contains("a ROCm runtime update is available")
                        && message.contains("no updates were applied")
                )
        );
        assert!(
            value
                .get("stdout")
                .and_then(Value::as_str)
                .is_some_and(|stdout| stdout.contains("status=update_available"))
        );
    }

    #[test]
    fn sandbox_check_updates_value_marks_runtime_repair_available() {
        let value = sandbox_check_updates_value(CommandCapture {
            argv: vec!["rocm".to_owned(), "update".to_owned()],
            exit_status: 0,
            stdout: "update\n  runtime release-wheel-multi-arch-7-14-0 status=repair_available installed=7.14.0 latest=7.14.0\n".to_owned(),
            stderr: String::new(),
        });

        assert_eq!(
            value.get("status").and_then(Value::as_str),
            Some("repair_available")
        );
        // The watcher's notify decision reads this flag, so a repair the user
        // has to apply by hand must not be reported as nothing to do.
        assert_eq!(
            value.get("update_available").and_then(Value::as_bool),
            Some(true)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("runtime repair is available")
                    && message.contains("no updates were applied"))
        );
    }

    #[test]
    fn a_newer_version_outranks_a_composition_repair_in_the_watcher_report() {
        let value = sandbox_check_updates_value(CommandCapture {
            argv: vec!["rocm".to_owned(), "update".to_owned()],
            exit_status: 0,
            stdout: "update\n  runtime old status=repair_available installed=7.14.0 latest=7.14.0\n  runtime stale status=update_available installed=7.13.0 latest=7.14.0\n".to_owned(),
            stderr: String::new(),
        });

        assert_eq!(
            value.get("status").and_then(Value::as_str),
            Some("update_available")
        );
    }

    #[test]
    fn sandbox_driver_plan_value_is_read_only_and_preserves_output() {
        let value = sandbox_driver_plan_value(CommandCapture {
            argv: vec![
                "rocm".to_owned(),
                "install".to_owned(),
                "driver".to_owned(),
                "--dkms".to_owned(),
                "--dry-run".to_owned(),
            ],
            exit_status: 0,
            stdout: "driver install plan\n  supported: true\n".to_owned(),
            stderr: String::new(),
        });

        assert_eq!(
            value.get("tool").and_then(Value::as_str),
            Some("driver_plan")
        );
        assert_eq!(value.get("status").and_then(Value::as_str), Some("planned"));
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("no driver commands were executed"))
        );
        assert!(
            value
                .get("stdout")
                .and_then(Value::as_str)
                .is_some_and(|stdout| stdout.contains("driver install plan"))
        );
    }

    #[test]
    fn sandbox_tool_cli_values_cover_restricted_plan_api() {
        let names = [
            SandboxToolArg::CheckUpdates,
            SandboxToolArg::ExamineSnapshot,
            SandboxToolArg::ListServers,
            SandboxToolArg::RestartServer,
            SandboxToolArg::StopServer,
            SandboxToolArg::PrefetchArtifact,
            SandboxToolArg::NotifyUser,
            SandboxToolArg::DriverPlan,
        ]
        .into_iter()
        .map(SandboxToolArg::as_cli_value)
        .collect::<Vec<_>>();

        for expected in [
            "check_updates",
            "examine_snapshot",
            "list_servers",
            "restart_server",
            "stop_server",
            "prefetch_artifact",
            "notify_user",
            "driver_plan",
        ] {
            assert!(names.contains(&expected), "missing sandbox tool {expected}");
        }
    }

    #[test]
    fn sandbox_tool_examine_snapshot_is_read_only() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-examine-snapshot");
        let value = run_sandbox_tool(
            &paths,
            SandboxToolArg::ExamineSnapshot,
            None,
            None,
            None,
            SandboxToolPolicy::default(),
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(
            value.get("tool").and_then(Value::as_str),
            Some("examine_snapshot")
        );
        assert_eq!(
            value.get("status").and_then(Value::as_str),
            Some("captured")
        );
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert!(value.get("examine").is_some());
        Ok(())
    }

    #[test]
    fn sandbox_tool_list_servers_returns_records() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-list-servers");
        paths.ensure()?;
        let record = rocm_core::ManagedServiceRecord::new(
            &paths,
            "svc-1",
            "vllm",
            "qwen",
            "Qwen/Qwen3.5",
            "127.0.0.1",
            11435,
            "managed",
            123,
            None,
            None,
            None,
        );
        record.write()?;

        let value = run_sandbox_tool(
            &paths,
            SandboxToolArg::ListServers,
            None,
            None,
            None,
            SandboxToolPolicy::default(),
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(
            value.get("tool").and_then(Value::as_str),
            Some("list_servers")
        );
        assert_eq!(value.get("status").and_then(Value::as_str), Some("listed"));
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert_eq!(value.get("count").and_then(Value::as_u64), Some(1));
        Ok(())
    }

    #[test]
    fn sandbox_tool_list_servers_first_run_returns_empty_list() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-list-servers-empty");
        let value = run_sandbox_tool(
            &paths,
            SandboxToolArg::ListServers,
            None,
            None,
            None,
            SandboxToolPolicy::default(),
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(
            value.get("tool").and_then(Value::as_str),
            Some("list_servers")
        );
        assert_eq!(value.get("count").and_then(Value::as_u64), Some(0));
        assert_eq!(
            value
                .get("services")
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(0)
        );
        Ok(())
    }

    #[test]
    fn sandbox_tool_requires_service_id_for_restart() {
        let (root, paths) = temp_app_paths("sandbox-restart-requires-service");
        let error = run_sandbox_tool(
            &paths,
            SandboxToolArg::RestartServer,
            None,
            None,
            None,
            SandboxToolPolicy::default(),
        )
        .unwrap_err();
        fs::remove_dir_all(root).ok();

        assert!(
            error.to_string().contains("restart_server requires"),
            "{error:#}"
        );
    }

    #[test]
    fn sandbox_tool_restart_server_reports_missing_service() {
        let (root, paths) = temp_app_paths("sandbox-restart-missing-service");
        let error = run_sandbox_tool(
            &paths,
            SandboxToolArg::RestartServer,
            Some("missing-service".to_owned()),
            None,
            None,
            SandboxToolPolicy::default(),
        )
        .unwrap_err();
        fs::remove_dir_all(root).ok();

        assert!(
            error
                .to_string()
                .contains("managed service `missing-service` not found"),
            "{error:#}"
        );
    }

    #[test]
    fn sandbox_tool_requires_service_id_for_stop() {
        let (root, paths) = temp_app_paths("sandbox-stop-requires-service");
        let error = run_sandbox_tool(
            &paths,
            SandboxToolArg::StopServer,
            None,
            None,
            None,
            SandboxToolPolicy::default(),
        )
        .unwrap_err();
        fs::remove_dir_all(root).ok();

        assert!(
            error.to_string().contains("stop_server requires"),
            "{error:#}"
        );
    }

    #[test]
    fn sandbox_tool_stop_server_reports_missing_service() {
        let (root, paths) = temp_app_paths("sandbox-stop-missing-service");
        let error = run_sandbox_tool(
            &paths,
            SandboxToolArg::StopServer,
            Some("missing-service".to_owned()),
            None,
            None,
            SandboxToolPolicy::default(),
        )
        .unwrap_err();
        fs::remove_dir_all(root).ok();

        assert!(
            error
                .to_string()
                .contains("managed service `missing-service` not found"),
            "{error:#}"
        );
    }

    #[test]
    fn sandbox_tool_notify_user_is_read_only() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-notify-user");
        let value = run_sandbox_tool(
            &paths,
            SandboxToolArg::NotifyUser,
            None,
            None,
            Some("ROCm setup is ready.".to_owned()),
            SandboxToolPolicy::default(),
        )?;
        let audit_text = fs::read_to_string(paths.audit_events_path())?;
        fs::remove_dir_all(root).ok();

        assert_eq!(
            value.get("tool").and_then(Value::as_str),
            Some("notify_user")
        );
        assert_eq!(
            value.get("status").and_then(Value::as_str),
            Some("notified")
        );
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert_eq!(
            value.get("message").and_then(Value::as_str),
            Some("ROCm setup is ready.")
        );
        assert!(audit_text.contains("\"category\":\"notification\""));
        assert!(audit_text.contains("ROCm setup is ready."));
        Ok(())
    }

    #[test]
    fn sandbox_runner_native_fallback_records_audit() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-native-audit");
        paths.ensure()?;

        let value = run_native_restricted_sandbox(
            &paths,
            SandboxToolArg::NotifyUser,
            None,
            None,
            Some("hello".to_owned()),
            SandboxToolPolicy::default(),
        )?;
        let audit_text = fs::read_to_string(paths.audit_events_path())?;
        fs::remove_dir_all(root).ok();

        assert_eq!(
            value.get("isolation").and_then(Value::as_str),
            Some("native_restricted")
        );
        assert!(audit_text.contains("\"category\":\"sandbox\""));
        assert!(audit_text.contains("\"action\":\"notify_user\""));
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn bubblewrap_command_separates_child_args_from_bwrap_options() {
        let mut command = ProcessCommand::new("bwrap");
        append_sandbox_tool_command_args(
            &mut command,
            Path::new("/tmp/rocmd"),
            SandboxToolArg::NotifyUser,
            None,
            None,
            Some("hello"),
            SandboxToolPolicy::default(),
        );
        let args = command
            .get_args()
            .map(|value| value.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        let separator = args
            .iter()
            .position(|arg| arg == "--")
            .expect("bubblewrap command should separate child command args");

        assert_eq!(
            args.get(separator + 1).map(String::as_str),
            Some("/tmp/rocmd")
        );
        assert_eq!(
            args.get(separator + 4).map(String::as_str),
            Some("--message")
        );
        assert_eq!(args.get(separator + 5).map(String::as_str), Some("hello"));
    }

    #[test]
    fn sandbox_prefetch_requires_artifact_ref() {
        let (root, paths) = temp_app_paths("sandbox-prefetch-requires-ref");
        let error = run_sandbox_tool(
            &paths,
            SandboxToolArg::PrefetchArtifact,
            None,
            None,
            None,
            SandboxToolPolicy::default(),
        )
        .unwrap_err();
        fs::remove_dir_all(root).ok();

        assert!(
            error.to_string().contains("prefetch_artifact requires"),
            "{error:#}"
        );
    }

    #[test]
    fn sandbox_prefetch_reports_policy_required_without_network() {
        let (root, paths) = temp_app_paths("sandbox-prefetch-policy");
        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#hf-main",
            "Qwen/Test-1B",
            ModelRecipeArtifactRecord {
                artifact_id: "hf-main".to_owned(),
                kind: "huggingface".to_owned(),
                uri: "Qwen/Test-1B".to_owned(),
                revision: Some("main".to_owned()),
                sha256: Some("a".repeat(64)),
                size_bytes: Some(1024),
                license: Some("apache-2.0".to_owned()),
                gated: Some(false),
                quantization: Some("bf16".to_owned()),
                engines: vec!["vllm".to_owned()],
                source_policy: None,
            },
            SandboxToolPolicy::default(),
        )
        .expect("policy value should render");
        fs::remove_dir_all(root).ok();

        assert_eq!(
            value.get("status").and_then(Value::as_str),
            Some("source_policy_required")
        );
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("cache")
                .and_then(|cache| cache.get("status"))
                .and_then(Value::as_str)
                .is_some_and(|status| status == "missing")
        );
    }

    #[test]
    fn sandbox_prefetch_unknown_artifact_ref_errors() {
        let (root, paths) = temp_app_paths("sandbox-prefetch-unknown-ref");
        let error = run_sandbox_tool(
            &paths,
            SandboxToolArg::PrefetchArtifact,
            None,
            Some("Qwen/Missing#hf-main".to_owned()),
            None,
            SandboxToolPolicy::default(),
        )
        .unwrap_err();
        fs::remove_dir_all(root).ok();

        assert!(error.to_string().contains("was not found"), "{error:#}");
    }

    #[test]
    fn sandbox_prefetch_cached_marker_skips_network() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-cached");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "direct-bin".to_owned(),
            kind: "url".to_owned(),
            uri: "https://example.invalid/should-not-be-fetched.bin".to_owned(),
            revision: None,
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(false),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };
        let cache = model_artifact_cache_status(&paths, "Qwen/Test-1B", &artifact);
        write_file_atomically(&cache.marker_path, br#"{"cached":true}"#)?;

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#direct-bin",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("cached"));
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(false));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_blocks_artifact_larger_than_approved_limit() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-byte-limit");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "direct-bin".to_owned(),
            kind: "url".to_owned(),
            uri: "https://example.invalid/too-large.bin".to_owned(),
            revision: None,
            sha256: Some("a".repeat(64)),
            size_bytes: Some(2048),
            license: Some("test-only".to_owned()),
            gated: Some(false),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#direct-bin",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("byte limit"))
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_blocks_non_direct_non_huggingface_source() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-non-direct");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "torrent-bin".to_owned(),
            kind: "torrent".to_owned(),
            uri: "https://example.invalid/artifact.torrent".to_owned(),
            revision: None,
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(false),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#torrent-bin",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("direct HTTP(S)"))
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_blocks_gated_huggingface_without_source_policy() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-hf-policy");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "hf-main".to_owned(),
            kind: "huggingface".to_owned(),
            uri: "https://huggingface.co/Qwen/Test-1B/resolve/main/model.safetensors".to_owned(),
            revision: Some("main".to_owned()),
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(true),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#hf-main",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("--allow-huggingface-download"))
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_blocks_huggingface_source_policy_without_token() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-hf-token");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "hf-main".to_owned(),
            kind: "huggingface".to_owned(),
            uri: "https://huggingface.co/Qwen/Test-1B/resolve/main/model.safetensors".to_owned(),
            revision: Some("main".to_owned()),
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(true),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#hf-main",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                allow_huggingface_download: true,
                huggingface_token: None,
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("ROCM_CLI_HUGGINGFACE_TOKEN"))
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_respects_manual_only_source_policy() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-manual-policy");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "manual-bin".to_owned(),
            kind: "url".to_owned(),
            uri: "https://example.invalid/manual.bin".to_owned(),
            revision: None,
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(false),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: Some(rocm_core::ModelRecipeArtifactSourcePolicyRecord {
                policy: "manual_only".to_owned(),
                required_hosts: Vec::new(),
                notes: vec!["requires manual license review".to_owned()],
            }),
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#manual-bin",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("manual-only"))
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_respects_declared_huggingface_auth_policy() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-hf-declared-auth");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "hf-main".to_owned(),
            kind: "huggingface".to_owned(),
            uri: "https://huggingface.co/Qwen/Test-1B/resolve/main/model.safetensors".to_owned(),
            revision: Some("main".to_owned()),
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(false),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: Some(rocm_core::ModelRecipeArtifactSourcePolicyRecord {
                policy: "huggingface_authenticated".to_owned(),
                required_hosts: vec!["huggingface.co".to_owned()],
                notes: Vec::new(),
            }),
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#hf-main",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("--allow-huggingface-download"))
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_never_sends_huggingface_token_to_non_huggingface_url() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-hf-host");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "hf-main".to_owned(),
            kind: "huggingface".to_owned(),
            uri: "https://example.invalid/Qwen/Test-1B/model.safetensors".to_owned(),
            revision: Some("main".to_owned()),
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(true),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#hf-main",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                allow_huggingface_download: true,
                huggingface_token: Some("hf_secret_should_not_leak".to_owned()),
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("non-Hugging Face URL"))
        );
        assert!(
            !serde_json::to_string(&value)?.contains("hf_secret_should_not_leak"),
            "prefetch report must not include authentication tokens"
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_never_sends_huggingface_token_over_plain_http() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-hf-https");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "hf-main".to_owned(),
            kind: "huggingface".to_owned(),
            uri: "http://huggingface.co/Qwen/Test-1B/model.safetensors".to_owned(),
            revision: Some("main".to_owned()),
            sha256: Some("a".repeat(64)),
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(true),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#hf-main",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                allow_huggingface_download: true,
                huggingface_token: Some("hf_secret_should_not_leak".to_owned()),
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("plain HTTP"))
        );
        assert!(
            !serde_json::to_string(&value)?.contains("hf_secret_should_not_leak"),
            "prefetch report must not include authentication tokens"
        );
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_downloads_direct_artifact_when_approved() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-download");
        let bytes = b"tiny verified artifact".to_vec();
        let uri = serve_one_http_response(bytes.clone())?;
        let sha256 = sha256_hex(&bytes);
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "direct-bin".to_owned(),
            kind: "url".to_owned(),
            uri,
            revision: None,
            sha256: Some(sha256.clone()),
            size_bytes: Some(bytes.len() as u64),
            license: Some("test-only".to_owned()),
            gated: Some(false),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#direct-bin",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        let bytes_path = value
            .get("bytes_path")
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .expect("bytes_path should be present");
        let marker_path = value
            .get("cache")
            .and_then(|cache| cache.get("marker_path"))
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .expect("marker path should be present");

        assert_eq!(
            value.get("status").and_then(Value::as_str),
            Some("prefetched")
        );
        assert_eq!(value.get("mutating").and_then(Value::as_bool), Some(true));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            value.get("sha256").and_then(Value::as_str),
            Some(sha256.as_str())
        );
        assert_eq!(
            value.get("source_policy").and_then(Value::as_str),
            Some("explicit_allow_artifact_download")
        );
        assert_eq!(fs::read(&bytes_path)?, bytes);
        assert!(marker_path.is_file());
        fs::remove_dir_all(root).ok();
        Ok(())
    }

    #[test]
    fn sandbox_prefetch_blocks_approved_artifact_without_sha256() -> Result<()> {
        let (root, paths) = temp_app_paths("sandbox-prefetch-no-sha");
        let artifact = ModelRecipeArtifactRecord {
            artifact_id: "direct-bin".to_owned(),
            kind: "url".to_owned(),
            uri: "https://example.invalid/artifact.bin".to_owned(),
            revision: None,
            sha256: None,
            size_bytes: Some(12),
            license: Some("test-only".to_owned()),
            gated: Some(false),
            quantization: None,
            engines: vec!["vllm".to_owned()],
            source_policy: None,
        };

        let value = prefetch_artifact_value_with_policy(
            &paths,
            "qwen#direct-bin",
            "Qwen/Test-1B",
            artifact,
            SandboxToolPolicy {
                allow_artifact_download: true,
                artifact_max_bytes: Some(1024),
                ..SandboxToolPolicy::default()
            },
        )?;
        fs::remove_dir_all(root).ok();

        assert_eq!(value.get("status").and_then(Value::as_str), Some("blocked"));
        assert_eq!(
            value.get("network_used").and_then(Value::as_bool),
            Some(false)
        );
        assert!(
            value
                .get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains("sha256"))
        );
        Ok(())
    }

    #[test]
    fn huggingface_url_detection_is_host_scoped() {
        assert!(is_huggingface_url(
            "https://huggingface.co/Qwen/Test-1B/resolve/main/model.safetensors"
        ));
        assert!(is_huggingface_url(
            "https://cdn-lfs.huggingface.co/repos/example"
        ));
        assert!(is_huggingface_url("https://hf.co/Qwen/Test-1B"));
        assert!(!is_huggingface_url(
            "https://huggingface.co.evil.example/Qwen/Test-1B"
        ));
        assert!(!is_huggingface_url(
            "https://huggingface.co@evil.example/Qwen/Test-1B"
        ));
    }
}
