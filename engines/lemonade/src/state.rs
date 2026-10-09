// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, format_http_base_url};
use rocm_engine_protocol::{
    DevicePolicy, ENGINE_RECIPE_CONTRACT_VERSION, EngineDeviceAvailability, EngineRecipeHint,
    GpuSelection,
};
use serde_json::{Value, json};
use std::collections::{VecDeque, hash_map::DefaultHasher};
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, Seek};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
#[cfg(not(windows))]
use std::process::{Command as ProcessCommand, Stdio};

use crate::backend_alignment::LemonadeRuntime;
use crate::install::LemonadeInstallManifest;
use crate::{DEFAULT_HOST, DEFAULT_MODEL, ENGINE_NAME, ServeHttpRequest, current_unix_millis};

/// Maximum bytes to read from the end of a log file when extracting tail lines.
/// Prevents reading entire gigabyte-sized logs on startup timeout.
const MAX_TAIL_READ: u64 = 4 * 1024 * 1024; // 4MB

#[derive(Debug, Clone)]
pub(crate) struct ServiceFiles {
    pub(crate) state_path: PathBuf,
    pub(crate) log_path: PathBuf,
}

/// Whether a real inference request has succeeded against this service.
///
/// Latch and backoff bookkeeping lives in `rocm-core` so both engines share one
/// implementation — what counts as *listed* differs per engine, what counts as
/// *serving* does not.
pub(crate) fn inference_verified(
    state_path: &Path,
    state: Option<&Value>,
    endpoint_url: &str,
    model_ref: &str,
) -> bool {
    let Some((host, port)) = parse_http_endpoint(endpoint_url) else {
        return false;
    };
    rocm_core::engine_state_inference_verified(
        state_path,
        state,
        &format_http_base_url(&host, port),
        model_ref,
        rocm_engine_protocol::resolve_endpoint_api_key().as_deref(),
    )
}

/// The device string reported once a model is loaded. Reflects the backend that
/// actually ran (`rocm`, `vulkan`, …) and the pinned GPU ordinal when known,
/// rather than a hardcoded value, so the report matches observed execution.
pub(crate) fn reported_device(state: Option<&Value>, backend: &str) -> String {
    match state.and_then(first_gpu_index_from_state) {
        Some(index) => format!("{backend} gpu {index}"),
        None => format!("{backend} gpu"),
    }
}

/// The first pinned GPU ordinal recorded in service state, if any.
fn first_gpu_index_from_state(state: &Value) -> Option<u32> {
    state
        .get("gpu_indices")
        .and_then(Value::as_array)
        .and_then(|indices| indices.first())
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

pub(crate) fn serve_http_command_args(request: &ServeHttpRequest) -> Vec<String> {
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
            serde_json::to_string(engine_recipe).expect("engine recipe serializes"),
        ]);
    }
    args
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

pub(crate) fn write_running_state(
    request: &ServeHttpRequest,
    runtime: &LemonadeRuntime,
    pid: u32,
    server_pid: Option<u32>,
    status: &str,
) -> Result<()> {
    write_state(
        &request.state_path,
        &json!({
            "service_id": request.service_id,
            "engine": ENGINE_NAME,
            "status": status,
            "pid": pid,
            "server_pid": server_pid,
            "model_ref": request.model_ref,
            "canonical_model_id": request.model_ref,
            "host": request.host,
            "port": request.port,
            "endpoint_url": endpoint_url(&request.host, request.port),
            "device_policy": device_policy_name(&request.device_policy),
            "gpu_indices": request.gpu_indices,
            "runtime_id": request.runtime_id.as_deref().unwrap_or(runtime.manifest.env_id.as_str()),
            "env_id": request.env_id.as_deref().unwrap_or(runtime.manifest.env_id.as_str()),
            "runtime_kind": "lemonade_embeddable",
            "runtime_executable": runtime.manifest.lemond,
            "log_path": request.log_path.as_ref().map(|path| path.display().to_string()),
            "engine_recipe": request.engine_recipe,
            "started_at_unix_ms": current_unix_millis(),
            // Kernel start-times captured while each PID is alive. Paired with the
            // PIDs, they identify these exact processes across PID recycling so a
            // later stop never signals a reused PID.
            "start_ticks": rocm_core::process_start_ticks(pid),
            "server_start_ticks": server_pid.and_then(rocm_core::process_start_ticks)
        }),
    )
}

pub(crate) fn service_files(service_id: &str) -> Result<ServiceFiles> {
    let paths = AppPaths::discover()?;
    Ok(ServiceFiles {
        state_path: paths
            .engine_state_dir(ENGINE_NAME)
            .join(format!("{service_id}.json")),
        log_path: paths
            .engine_logs_dir(ENGINE_NAME)
            .join(format!("{service_id}.log")),
    })
}

pub(crate) fn endpoint_url(host: &str, port: u16) -> String {
    format!("{}/v1", format_http_base_url(host, port))
}

pub(crate) fn endpoint_url_from_state(state: &Value) -> Option<String> {
    value_string(state, "endpoint_url").or_else(|| {
        let host = value_string(state, "host")?;
        let port = value_u32(state, "port")?;
        let port = u16::try_from(port).ok()?;
        Some(endpoint_url(&host, port))
    })
}

pub(crate) fn read_service_state(path: &Path) -> Result<Value> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
}

fn write_state(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

pub(crate) fn merge_json_state(path: &Path, patch: &Value) -> Result<()> {
    let mut value = read_service_state(path).unwrap_or_else(|_| json!({}));
    if !value.is_object() {
        value = json!({});
    }
    let object = value.as_object_mut().expect("object checked above");
    if let Some(patch) = patch.as_object() {
        for (key, value) in patch {
            object.insert(key.clone(), value.clone());
        }
    }
    write_state(path, &value)
}

pub(crate) fn mark_json_status(path: &Path, status: &str) -> Result<()> {
    merge_json_state(
        path,
        &json!({
            "engine": ENGINE_NAME,
            "status": status,
            "stopped_at_unix_ms": current_unix_millis(),
        }),
    )
}

pub(crate) fn value_string(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(ToOwned::to_owned)
}

fn value_u32(value: &Value, key: &str) -> Option<u32> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

/// Reconstruct the identity (PID + kernel start-time) of the process a stop
/// should terminate. The server PID is preferred, falling back to the launcher
/// PID, each paired with its recorded start-time. `start_ticks` is absent in
/// pre-existing state files, in which case verification degrades to best-effort.
pub(crate) fn identity_from_state(state: &Value) -> Option<rocm_core::ProcessIdentity> {
    if let Some(server_pid) = value_u32(state, "server_pid") {
        let start_ticks = state.get("server_start_ticks").and_then(Value::as_u64);
        return Some(rocm_core::ProcessIdentity::new(server_pid, start_ticks));
    }
    let pid = value_u32(state, "pid")?;
    let start_ticks = state.get("start_ticks").and_then(Value::as_u64);
    Some(rocm_core::ProcessIdentity::new(pid, start_ticks))
}

pub(crate) fn tail_lines(path: &Path, limit: usize) -> Result<Vec<String>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to stat {}", path.display()))?;
    let file_size = metadata.len();

    // For small files, read normally to preserve exact line count.
    if file_size <= MAX_TAIL_READ {
        let reader = std::io::BufReader::new(file);
        let mut lines = VecDeque::with_capacity(limit);
        for line in reader.lines() {
            let line = line.with_context(|| format!("failed to read {}", path.display()))?;
            if lines.len() == limit {
                lines.pop_front();
            }
            lines.push_back(line);
        }
        return Ok(lines.into_iter().collect());
    }

    // For large files, seek near the end and read only the final chunk.
    // This prevents reading multi-gigabyte logs during timeout errors.
    let mut file = file;
    let seek_pos = file_size.saturating_sub(MAX_TAIL_READ);
    file.seek(std::io::SeekFrom::Start(seek_pos))
        .with_context(|| format!("failed to seek in {}", path.display()))?;

    let reader = std::io::BufReader::new(file);
    let mut lines = VecDeque::with_capacity(limit);
    let mut skipped_first = seek_pos == 0;
    for line in reader.lines() {
        let line = line.with_context(|| format!("failed to read {}", path.display()))?;
        // Skip the first line after seeking, as it may be partial.
        if !skipped_first {
            skipped_first = true;
            continue;
        }
        if lines.len() == limit {
            lines.pop_front();
        }
        lines.push_back(line);
    }
    Ok(lines.into_iter().collect())
}

pub(crate) fn terminate_pid(pid: u32, _force: bool) -> bool {
    rocm_core::terminate_process(pid).is_ok()
}

pub(crate) fn free_local_port() -> Result<u16> {
    let listener = TcpListener::bind((DEFAULT_HOST, 0)).context("failed to reserve local port")?;
    Ok(listener.local_addr()?.port())
}

pub(crate) fn parse_http_endpoint(endpoint_url: &str) -> Option<(String, u16)> {
    let without_scheme = endpoint_url.trim().strip_prefix("http://")?;
    let authority = without_scheme.split('/').next()?.trim();
    if let Some(rest) = authority.strip_prefix('[') {
        let end = rest.find(']')?;
        let host = rest[..end].to_owned();
        let port = rest[end + 1..].strip_prefix(':')?.parse().ok()?;
        return Some((host, port));
    }
    let (host, port) = authority.rsplit_once(':')?;
    Some((host.to_owned(), port.parse().ok()?))
}

pub(crate) fn normalize_device_policy(policy: Option<DevicePolicy>) -> Result<DevicePolicy> {
    match policy.unwrap_or(DevicePolicy::GpuRequired) {
        DevicePolicy::GpuRequired | DevicePolicy::GpuPreferred => Ok(DevicePolicy::GpuRequired),
        DevicePolicy::CpuOnly => {
            bail!("Lemonade adapter requires ROCm GPU execution; no CPU fallback is used")
        }
    }
}

pub(crate) fn require_gpu_required(policy: &DevicePolicy) -> Result<()> {
    match policy {
        DevicePolicy::GpuRequired | DevicePolicy::GpuPreferred => Ok(()),
        DevicePolicy::CpuOnly => {
            bail!("Lemonade adapter requires ROCm GPU execution; no CPU fallback is used")
        }
    }
}

/// Probe real GPU availability and resolve the device ordinal(s) to pin before a
/// GPU-required launch. Returns the resolved indices, or an error that stops
/// serving before any process is spawned:
/// - no usable device → fail with actionable guidance (no CPU/GPU-0 fallback);
/// - `auto` (empty `requested`) → the first present, visible device;
/// - explicit index not among the usable devices → fail.
///
/// When availability cannot be probed on this platform, the requested selection
/// is passed through unchanged so serving is not blocked on that basis.
pub(crate) fn resolve_serve_gpu_indices(requested: &[u32]) -> Result<Vec<u32>> {
    resolve_gpu_indices_against(requested, rocm_core::usable_amd_gpu_indices())
}

/// Pure resolution used by [`resolve_serve_gpu_indices`], split out so the
/// auto/explicit/no-device policy can be unit-tested without real hardware.
/// `usable` is the probe result: `None` (unprobeable) passes the request
/// through; `Some(_)` is authoritative.
fn resolve_gpu_indices_against(requested: &[u32], usable: Option<Vec<u32>>) -> Result<Vec<u32>> {
    let Some(usable) = usable else {
        return Ok(requested.to_vec());
    };
    if usable.is_empty() {
        bail!(
            "no usable AMD GPU detected; `rocm serve` requires a GPU under the default \
             GPU-required policy and does not fall back to CPU. Check the driver with \
             `rocm examine`, confirm /dev/kfd is present, and ensure HIP_VISIBLE_DEVICES / \
             ROCR_VISIBLE_DEVICES are not masking every device."
        );
    }
    if requested.is_empty() {
        return Ok(vec![usable[0]]);
    }
    for index in requested {
        if !usable.contains(index) {
            let visible = usable
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            bail!(
                "requested GPU {index} is not available on this host; usable GPU indices: [{visible}]"
            );
        }
    }
    Ok(requested.to_vec())
}

/// GPU device availability for `detect`, sourced from the real device probe. When
/// the probe is authoritative it reports true/false with a reason; when it cannot
/// determine availability on this platform, it falls back to whether the runtime
/// is installed so `detect` stays informative.
pub(crate) fn gpu_availability_device(runtime_installed: bool) -> EngineDeviceAvailability {
    match rocm_core::usable_amd_gpu_indices() {
        Some(indices) if !indices.is_empty() => EngineDeviceAvailability {
            kind: "rocm_gpu".to_owned(),
            available: true,
            reason: None,
        },
        Some(_) => EngineDeviceAvailability {
            kind: "rocm_gpu".to_owned(),
            available: false,
            reason: Some(
                "no AMD GPU detected, or every device is masked by \
                 HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES"
                    .to_owned(),
            ),
        },
        None => EngineDeviceAvailability {
            kind: "rocm_gpu".to_owned(),
            available: runtime_installed,
            reason: if runtime_installed {
                None
            } else {
                Some("Lemonade embeddable runtime is not installed".to_owned())
            },
        },
    }
}

pub(crate) fn parse_device_policy_arg(value: Option<&str>) -> Result<DevicePolicy> {
    match value.unwrap_or("gpu_required") {
        "gpu" | "gpu_required" | "gpu_preferred" => Ok(DevicePolicy::GpuRequired),
        "cpu" | "cpu_only" => Ok(DevicePolicy::CpuOnly),
        other => bail!("unknown device policy `{other}`"),
    }
}

/// Parse a `--gpu` CLI value into an optional `GpuSelection` for `LaunchRequest`.
pub(crate) fn parse_gpu_selection_arg(value: Option<&str>) -> Result<Option<GpuSelection>> {
    value
        .map(|raw| GpuSelection::parse_cli_value(raw).map_err(anyhow::Error::msg))
        .transpose()
}

/// Parse a `--gpu` CLI value into explicit device ordinals (empty for `auto`).
pub(crate) fn parse_gpu_indices_arg(value: Option<&str>) -> Result<Vec<u32>> {
    Ok(rocm_engine_protocol::launch_gpu_indices(
        parse_gpu_selection_arg(value)?.as_ref(),
    ))
}

const fn device_policy_name(policy: &DevicePolicy) -> &'static str {
    match policy {
        DevicePolicy::GpuRequired => "gpu_required",
        DevicePolicy::GpuPreferred => "gpu_preferred",
        DevicePolicy::CpuOnly => "cpu_only",
    }
}

pub(crate) fn accepted_engine_recipe(
    engine_recipe: Option<EngineRecipeHint>,
) -> Result<Option<EngineRecipeHint>> {
    if let Some(hint) = &engine_recipe {
        if hint.engine != ENGINE_NAME {
            bail!(
                "engine_recipe target `{}` does not match adapter `{}`",
                hint.engine,
                ENGINE_NAME
            );
        }
        if hint.contract_version != ENGINE_RECIPE_CONTRACT_VERSION {
            bail!(
                "engine_recipe contract `{}` is unsupported; expected `{}`",
                hint.contract_version,
                ENGINE_RECIPE_CONTRACT_VERSION
            );
        }
    }
    Ok(engine_recipe)
}

pub(crate) fn parse_engine_recipe_json(value: Option<String>) -> Result<Option<EngineRecipeHint>> {
    value
        .map(|text| {
            serde_json::from_str::<EngineRecipeHint>(&text)
                .context("failed to parse engine recipe JSON")
        })
        .transpose()
        .and_then(accepted_engine_recipe)
}

pub(crate) fn resolve_lemonade_model_ref(model_ref: &str) -> String {
    let trimmed = model_ref.trim();
    let lower = trimmed.to_ascii_lowercase();
    // Only exact, recognized shorthand aliases map to the bundled assistant GGUF.
    // A syntactically valid `owner/repo` Hugging Face checkpoint must never be
    // silently rewritten here — e.g. `Qwen/Qwen2.5-1.5B-Instruct` is a real
    // checkpoint id, not a shorthand, and is passed through unchanged so the
    // serve path can honour (or explicitly reject) it rather than substituting
    // an unrelated model behind the user's back.
    if trimmed.is_empty()
        || matches!(
            lower.as_str(),
            "qwen"
                | "assistant"
                | "default"
                | "small"
                | "lemonade-qwen"
                | "qwen-gguf"
                | "qwen3-4b"
                | "qwen3-4b-instruct"
                | "qwen3-4b-instruct-2507-gguf"
        )
    {
        DEFAULT_MODEL.to_owned()
    } else if matches!(
        lower.as_str(),
        "tiny" | "qwen-smoke" | "lemonade-tiny" | "qwen3-0.6b-gguf"
    ) {
        "Qwen3-0.6B-GGUF".to_owned()
    } else {
        trimmed.to_owned()
    }
}

pub(crate) fn manifest_lock_hash(manifest: &LemonadeInstallManifest) -> String {
    let mut hasher = DefaultHasher::new();
    manifest.env_id.hash(&mut hasher);
    manifest.version.hash(&mut hasher);
    manifest.runtime_dir.hash(&mut hasher);
    manifest.lemond.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::scratch_dir;
    use std::io::{Read, Write};
    use std::time::Duration;

    #[test]
    fn identity_from_state_prefers_server_pid_and_its_start_ticks() {
        let state = json!({
            "pid": 100,
            "start_ticks": 111_u64,
            "server_pid": 200,
            "server_start_ticks": 222_u64,
        });
        let identity = identity_from_state(&state).expect("identity");
        assert_eq!(identity.pid, 200);
        assert_eq!(identity.start_ticks, Some(222));
    }

    #[test]
    fn identity_from_state_falls_back_to_launcher_pid() {
        let state = json!({ "pid": 100, "start_ticks": 111_u64 });
        let identity = identity_from_state(&state).expect("identity");
        assert_eq!(identity.pid, 100);
        assert_eq!(identity.start_ticks, Some(111));
    }

    #[test]
    fn identity_from_legacy_state_has_no_start_ticks() {
        // Pre-existing state files carry only PIDs; verification must degrade.
        let state = json!({ "server_pid": 200 });
        let identity = identity_from_state(&state).expect("identity");
        assert_eq!(identity.pid, 200);
        assert_eq!(identity.start_ticks, None);
    }

    #[test]
    fn identity_from_state_without_any_pid_is_none() {
        assert!(identity_from_state(&json!({ "status": "running" })).is_none());
    }

    #[test]
    fn qwen_alias_resolves_to_validated_assistant_gguf_model() {
        assert_eq!(resolve_lemonade_model_ref("qwen"), DEFAULT_MODEL);
        assert_eq!(resolve_lemonade_model_ref("qwen-smoke"), "Qwen3-0.6B-GGUF");
    }

    #[test]
    fn canonical_hugging_face_name_passes_through_unchanged() {
        assert_eq!(
            resolve_lemonade_model_ref("LiquidAI/LFM2.5-230M-GGUF:Q4_0"),
            "LiquidAI/LFM2.5-230M-GGUF:Q4_0"
        );
        // Regression (EAI-7370): a fully-qualified checkpoint id that merely
        // contains a known-alias substring must not be swapped for the bundled
        // default. `Qwen/Qwen2.5-1.5B-Instruct` was silently served as
        // `Qwen3-4B-Instruct-2507-GGUF`; it must now pass through verbatim.
        assert_eq!(
            resolve_lemonade_model_ref("Qwen/Qwen2.5-1.5B-Instruct"),
            "Qwen/Qwen2.5-1.5B-Instruct"
        );
    }

    /// Answer `count` chat requests on a loopback port with the given status,
    /// recording how many arrived.
    fn spawn_chat_endpoint(
        status_line: &'static str,
        count: usize,
    ) -> (u16, std::thread::JoinHandle<usize>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let port = listener.local_addr().expect("local addr").port();
        let handle = std::thread::spawn(move || {
            let mut served = 0;
            for _ in 0..count {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut buffer = [0_u8; 1024];
                let _ = stream.read(&mut buffer);
                let body = r#"{"choices":[{"message":{"content":"ok"}}]}"#;
                let _ = write!(
                    stream,
                    "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                served += 1;
            }
            served
        });
        (port, handle)
    }

    #[test]
    fn inference_verification_latches_into_the_state_file() {
        // First check probes and records the verdict; the second reads the latch
        // and leaves the model alone.
        let (port, server) = spawn_chat_endpoint("HTTP/1.1 200 OK", 1);
        let dir = scratch_dir("inference-latch");
        let state_path = dir.join("state.json");
        fs::write(&state_path, json!({"status": "ready"}).to_string()).expect("seed state");
        let endpoint = format_http_base_url("127.0.0.1", port);

        let state = read_service_state(&state_path).ok();
        assert!(inference_verified(
            &state_path,
            state.as_ref(),
            &endpoint,
            DEFAULT_MODEL
        ));

        let state = read_service_state(&state_path).expect("state readable");
        assert!(
            state
                .get(rocm_core::INFERENCE_VERIFIED_STATE_KEY)
                .and_then(Value::as_u64)
                .is_some(),
            "a passing probe is latched so later healthchecks skip it"
        );
        assert!(inference_verified(
            &state_path,
            Some(&state),
            &endpoint,
            DEFAULT_MODEL
        ));

        assert_eq!(
            server.join().expect("server thread"),
            1,
            "the latched check must not send a second inference request"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn inference_verification_withheld_while_the_endpoint_cannot_serve() {
        // The reported failure: the model is listed but inference still 5xxs.
        let (port, server) = spawn_chat_endpoint("HTTP/1.1 503 Service Unavailable", 1);
        let dir = scratch_dir("inference-unverified");
        let state_path = dir.join("state.json");
        fs::write(&state_path, json!({"status": "ready"}).to_string()).expect("seed state");
        let endpoint = format_http_base_url("127.0.0.1", port);

        let state = read_service_state(&state_path).ok();
        assert!(!inference_verified(
            &state_path,
            state.as_ref(),
            &endpoint,
            DEFAULT_MODEL
        ));

        let state = read_service_state(&state_path).expect("state readable");
        assert!(
            state.get(rocm_core::INFERENCE_VERIFIED_STATE_KEY).is_none(),
            "nothing is latched until inference actually answers"
        );

        server.join().expect("server thread");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn serve_gpu_resolution_fails_fast_when_no_device_usable() {
        // No-device / masked-device path: the probe authoritatively reports zero
        // usable GPUs, so serving is refused before any process is spawned.
        let error = resolve_gpu_indices_against(&[], Some(vec![]))
            .expect_err("zero usable devices must be rejected");
        assert!(error.to_string().contains("no usable AMD GPU"));
    }

    #[test]
    fn serve_gpu_resolution_auto_pins_first_present_device() {
        // `auto` (empty request) resolves to the first present device — never an
        // assumed device 0 when device 0 is not present.
        assert_eq!(
            resolve_gpu_indices_against(&[], Some(vec![1, 2])).unwrap(),
            vec![1]
        );
    }

    #[test]
    fn serve_gpu_resolution_validates_explicit_index() {
        // An explicitly requested present device is honored.
        assert_eq!(
            resolve_gpu_indices_against(&[2], Some(vec![0, 1, 2])).unwrap(),
            vec![2]
        );
        // A requested device that is not present is rejected rather than silently
        // remapped to another GPU.
        let error = resolve_gpu_indices_against(&[3], Some(vec![0, 1]))
            .expect_err("absent device must be rejected");
        assert!(
            error
                .to_string()
                .contains("requested GPU 3 is not available")
        );
    }

    #[test]
    fn serve_gpu_resolution_passes_through_when_unprobeable() {
        // When availability cannot be determined, the request is passed through
        // unchanged so serving is not blocked on that basis.
        assert_eq!(
            resolve_gpu_indices_against(&[], None).unwrap(),
            Vec::<u32>::new()
        );
        assert_eq!(resolve_gpu_indices_against(&[1], None).unwrap(), vec![1]);
    }

    #[test]
    fn reported_device_reflects_backend_and_pinned_ordinal() {
        let state = json!({ "gpu_indices": [1] });
        assert_eq!(reported_device(Some(&state), "vulkan"), "vulkan gpu 1");
        // Without a recorded ordinal the backend is still reported truthfully.
        assert_eq!(reported_device(None, "rocm"), "rocm gpu");
    }

    #[test]
    fn device_policy_rejects_cpu_without_fallback() {
        let error = normalize_device_policy(Some(DevicePolicy::CpuOnly))
            .expect_err("cpu should be rejected")
            .to_string();
        assert!(error.contains("no CPU fallback"));
    }

    #[test]
    fn endpoint_parser_supports_ipv6_loopback() {
        assert_eq!(
            parse_http_endpoint("http://[::1]:11435/v1"),
            Some(("::1".to_owned(), 11435))
        );
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
        let args = serve_http_command_args(&request);
        assert!(args.contains(&"--runtime-id".to_owned()));
        assert!(args.contains(&"runtime".to_owned()));
        assert!(args.contains(&"--env-id".to_owned()));
        assert!(args.contains(&"env".to_owned()));
        assert!(!args.iter().any(|arg| arg == "cpu"));
    }
}
