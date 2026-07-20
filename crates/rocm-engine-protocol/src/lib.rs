// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub const ENGINE_PROTOCOL_VERSION: &str = "0.1.0";
pub const ENGINE_RECIPE_CONTRACT_VERSION: &str = "0.1.0";
pub const ENGINE_PLUGIN_BINARY_PREFIX: &str = "rocm-engine-";
pub const DEFAULT_LOG_TAIL_LINES: usize = 80;

/// Environment variable holding a *path* to a 0600 file whose contents are the
/// endpoint API key.
///
/// `rocm serve` sets this explicitly on the internal `__engine-serve-http` child
/// (never argv), and each engine adapter reads it to decide whether to enforce
/// authentication. A path — not the secret value — is what the detached-spawn
/// primitives accept as an env override, and a key file keeps the secret off both
/// the argv and the environment block.
///
/// The adapter reads *only* this explicitly-set carrier, never a value-bearing
/// variable: the engine child inherits the launching shell's environment, so
/// reading a variable like `ROCM_SERVE_API_KEY` would let a stray value in the
/// user's shell silently authenticate an endpoint we intended to leave open
/// (e.g. a loopback bind).
pub const ENDPOINT_API_KEY_FILE_ENV: &str = "ROCM_SERVE_API_KEY_FILE";

/// Resolve the endpoint API key an engine adapter must enforce, if any.
///
/// Reads the key file named by [`ENDPOINT_API_KEY_FILE_ENV`]. Returns `None` when
/// the variable is unset or the file is empty/missing — i.e. an unauthenticated
/// (loopback) endpoint. The value is trimmed of surrounding whitespace/newlines.
pub fn resolve_endpoint_api_key() -> Option<String> {
    let path = std::env::var(ENDPOINT_API_KEY_FILE_ENV).ok()?;
    endpoint_api_key_from_file(std::path::Path::new(&path))
}

/// Resolve the *path* to the endpoint API key file, for engines that authenticate
/// from a file rather than an env var (e.g. llama-server's `--api-key-file`).
///
/// Returns the path named by [`ENDPOINT_API_KEY_FILE_ENV`] when it exists and holds
/// a non-empty key — the same condition under which [`resolve_endpoint_api_key`]
/// returns `Some`. Handing the child the CLI-managed key file directly (rather than
/// copying the secret into a second file) keeps the key's lifecycle owned by
/// `rocm serve`, which creates the file before spawn and deletes it on stop, so no
/// stale plaintext copy is ever left behind.
pub fn resolve_endpoint_api_key_file() -> Option<PathBuf> {
    endpoint_api_key_file_if_valid(&PathBuf::from(
        std::env::var(ENDPOINT_API_KEY_FILE_ENV).ok()?,
    ))
}

/// Return `path` when it holds a non-empty endpoint key, otherwise `None`.
///
/// Uses [`endpoint_api_key_from_file`] for the key check. Split out from
/// [`resolve_endpoint_api_key_file`] so the key-presence gate is testable without
/// mutating the process environment.
pub fn endpoint_api_key_file_if_valid(path: &std::path::Path) -> Option<PathBuf> {
    endpoint_api_key_from_file(path).map(|_| path.to_path_buf())
}

/// Read and trim an endpoint API key from `path`.
///
/// Returns `None` for a missing file or empty/whitespace-only contents. Split out
/// from [`resolve_endpoint_api_key`] so it is testable without mutating the
/// process environment.
pub fn endpoint_api_key_from_file(path: &std::path::Path) -> Option<String> {
    let contents = std::fs::read_to_string(path).ok()?;
    let trimmed = contents.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EnginePluginDescriptor {
    pub id: String,
    pub executable_path: PathBuf,
}

pub fn platform_engine_plugin_binary_name(engine_id: &str) -> String {
    let binary_name = format!("{ENGINE_PLUGIN_BINARY_PREFIX}{engine_id}");
    rocm_core::platform_binary_name(&binary_name)
}

pub fn engine_id_from_plugin_binary_name(name: &str) -> Option<String> {
    if name.contains('/') || name.contains('\\') {
        return None;
    }
    let name = strip_optional_exe_suffix(name);
    let engine_id = name.strip_prefix(ENGINE_PLUGIN_BINARY_PREFIX)?;
    is_valid_engine_id(engine_id).then(|| engine_id.to_owned())
}

pub fn discover_engine_plugins<I, P>(plugin_dirs: I) -> std::io::Result<Vec<EnginePluginDescriptor>>
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path>,
{
    let mut plugins = Vec::new();
    for plugin_dir in plugin_dirs {
        let plugin_dir = plugin_dir.as_ref();
        if !plugin_dir.is_dir() {
            continue;
        }

        for entry in fs::read_dir(plugin_dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            let Some(id) = engine_id_from_plugin_binary_name(name) else {
                continue;
            };
            if plugins
                .iter()
                .any(|plugin: &EnginePluginDescriptor| plugin.id == id)
            {
                continue;
            }
            plugins.push(EnginePluginDescriptor {
                id,
                executable_path: path,
            });
        }
    }
    plugins.sort_by(|left, right| {
        left.id
            .cmp(&right.id)
            .then_with(|| left.executable_path.cmp(&right.executable_path))
    });
    Ok(plugins)
}

fn strip_optional_exe_suffix(name: &str) -> &str {
    if name
        .get(name.len().saturating_sub(4)..)
        .is_some_and(|suffix| suffix.eq_ignore_ascii_case(".exe"))
    {
        &name[..name.len() - 4]
    } else {
        name
    }
}

fn is_valid_engine_id(engine_id: &str) -> bool {
    !engine_id.is_empty()
        && engine_id != "."
        && engine_id != ".."
        && engine_id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-'))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EngineMethod {
    Detect,
    Install,
    Capabilities,
    ResolveModel,
    Launch,
    Healthcheck,
    Endpoint,
    Stop,
    Logs,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DevicePolicy {
    GpuRequired,
    GpuPreferred,
    CpuOnly,
}

/// Which GPU device a server should run on. `Auto` lets the CLI pick the first
/// free GPU; `Index` pins one explicit GPU ordinal. Running a single model
/// across multiple GPUs is not supported.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GpuSelection {
    Auto,
    Index(u32),
}

impl GpuSelection {
    /// Parse a CLI value (`auto` or a single GPU index like `1`) into a
    /// selection. Returns an error for non-numeric input or for a comma list,
    /// since serving a model across multiple GPUs is not supported.
    pub fn parse_cli_value(value: &str) -> Result<Self, String> {
        let trimmed = value.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        if trimmed.contains(',') {
            return Err(format!(
                "invalid --gpu value `{value}`: serving across multiple GPUs is not supported; pass a single GPU index or `auto`"
            ));
        }
        let index: u32 = trimmed.parse().map_err(|_| {
            format!("invalid --gpu value `{value}`: `{trimmed}` is not a GPU index")
        })?;
        Ok(Self::Index(index))
    }
}

/// Render an explicit GPU ordinal as a comma-separated `HIP_VISIBLE_DEVICES`
/// value. Returns `None` when the slice is empty (no pinning requested).
#[must_use]
pub fn gpu_indices_to_csv(indices: &[u32]) -> Option<String> {
    if indices.is_empty() {
        return None;
    }
    Some(
        indices
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(","),
    )
}

/// Pin the resolved GPU ordinals onto a child process by exporting
/// `HIP_VISIBLE_DEVICES`.
///
/// This is the single contract point engines use to make `--gpu`/`auto`
/// selection visible to the spawned server; an empty slice is a no-op (the
/// engine keeps its default device visibility, never a CPU fallback).
pub fn apply_gpu_visibility(command: &mut std::process::Command, gpu_indices: &[u32]) {
    if let Some(csv) = gpu_indices_to_csv(gpu_indices) {
        command.env("HIP_VISIBLE_DEVICES", csv);
    }
}

/// The explicit GPU ordinal carried by an optional `GpuSelection`, as a list.
///
/// Used by engine launch paths. `Auto` and `None` yield an empty list (engine
/// default visibility), so engines only pin when the CLI resolved a concrete
/// index.
#[must_use]
pub fn launch_gpu_indices(selection: Option<&GpuSelection>) -> Vec<u32> {
    match selection {
        Some(GpuSelection::Index(index)) => vec![*index],
        _ => Vec::new(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineRequestEnvelope {
    pub method: EngineMethod,
    #[serde(default)]
    pub payload: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineErrorDetail {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineResponseEnvelope {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<EngineErrorDetail>,
}

impl EngineResponseEnvelope {
    pub fn success<T>(data: T) -> Self
    where
        T: Serialize,
    {
        Self {
            ok: true,
            data: Some(serde_json::to_value(data).expect("serializable response")),
            error: None,
        }
    }

    pub fn failure(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            ok: false,
            data: None,
            error: Some(EngineErrorDetail {
                code: code.into(),
                message: message.into(),
            }),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DetectRequest {
    pub runtime_id: Option<String>,
    pub device_filter: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallRequest {
    pub runtime_id: String,
    pub python_version: Option<String>,
    #[serde(default)]
    pub reinstall: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env_root: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolveModelRequest {
    pub model_ref: String,
    pub runtime_id: Option<String>,
    pub device_policy: Option<DevicePolicy>,
    pub recipe_override: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_recipe: Option<EngineRecipeHint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaunchRequest {
    pub service_id: String,
    pub env_id: Option<String>,
    pub runtime_id: Option<String>,
    pub model_ref: String,
    pub host: String,
    pub port: u16,
    pub device_policy: Option<DevicePolicy>,
    pub endpoint_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_recipe: Option<EngineRecipeHint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_selection: Option<GpuSelection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthcheckRequest {
    pub service_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointRequest {
    pub service_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopRequest {
    pub service_id: String,
    #[serde(default)]
    pub force: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogsRequest {
    pub service_id: String,
    pub tail_lines: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineDeviceAvailability {
    pub kind: String,
    pub available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCapabilities {
    pub cpu: bool,
    pub rocm_gpu: bool,
    pub openai_compatible: bool,
    pub tool_calling: bool,
    pub quantized_models: String,
    pub reasoning_parser: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectResponse {
    pub installed: bool,
    pub env_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_executable: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub managed_env: Option<bool>,
    pub python_version: Option<String>,
    pub torch_version: Option<String>,
    pub transformers_version: Option<String>,
    pub available_devices: Vec<EngineDeviceAvailability>,
    pub capabilities: EngineCapabilities,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallResponse {
    pub env_id: String,
    pub env_path: String,
    pub python_executable: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_executable: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub managed_env: Option<bool>,
    pub installed_packages: Vec<String>,
    pub capabilities: EngineCapabilities,
    pub lock_hash: String,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolveModelResponse {
    pub canonical_model_id: String,
    pub task: String,
    pub source: String,
    pub revision: String,
    pub loader: String,
    pub trust_remote_code: bool,
    pub chat_template_mode: String,
    pub dtype: String,
    pub device_policy: DevicePolicy,
    pub estimated_memory: String,
    pub launch_defaults: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_recipe: Option<EngineRecipeHint>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct EngineRecipeHint {
    pub contract_version: String,
    pub engine: String,
    #[serde(default)]
    pub required_flags: Vec<String>,
    #[serde(default)]
    pub parser_settings: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preferred_endpoint: Option<EngineRecipeEndpointHint>,
    #[serde(default)]
    pub unsupported_combinations: Vec<EngineRecipeUnsupportedCombinationHint>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct EngineRecipeEndpointHint {
    pub endpoint_mode: String,
    #[serde(default)]
    pub settings: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct EngineRecipeUnsupportedCombinationHint {
    pub combination: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaunchResponse {
    pub service_id: String,
    pub pid: u32,
    pub endpoint_url: String,
    pub log_path: String,
    pub state_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthcheckResponse {
    pub status: String,
    pub model_loaded: bool,
    pub device: String,
    pub uptime_sec: u64,
    pub queue_depth: u32,
    pub last_error: Option<String>,
    pub tokens_per_sec: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointResponse {
    pub endpoint_url: String,
    pub api_style: String,
    pub supported_routes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopResponse {
    pub stopped: bool,
    pub graceful: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogsResponse {
    pub log_path: String,
    pub recent_lines: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_api_key_from_file_reads_and_trims() {
        let dir = unique_temp_dir("endpoint-key");
        std::fs::create_dir_all(&dir).unwrap();
        let key_file = dir.join("endpoint-api-key");
        std::fs::write(&key_file, "  secret-key-value\n").unwrap();
        assert_eq!(
            endpoint_api_key_from_file(&key_file),
            Some("secret-key-value".to_owned())
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn endpoint_api_key_from_file_none_for_missing_or_empty() {
        let dir = unique_temp_dir("endpoint-key-empty");
        std::fs::create_dir_all(&dir).unwrap();
        let missing = dir.join("does-not-exist");
        assert_eq!(endpoint_api_key_from_file(&missing), None);
        let empty = dir.join("empty");
        std::fs::write(&empty, "   \n").unwrap();
        assert_eq!(endpoint_api_key_from_file(&empty), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn endpoint_api_key_file_if_valid_returns_path_only_when_key_present() {
        let dir = unique_temp_dir("endpoint-key-file-if-valid");
        std::fs::create_dir_all(&dir).unwrap();
        // A file holding a key resolves to that same path — the packaged
        // llama-server fallback hands this to `--api-key-file` instead of writing a
        // copy, so cleanup stays owned by the caller's key file.
        let key_file = dir.join("endpoint-key");
        std::fs::write(&key_file, "secret-key-value\n").unwrap();
        assert_eq!(
            endpoint_api_key_file_if_valid(&key_file),
            Some(key_file.clone())
        );
        // A missing or empty file is an unauthenticated (loopback) endpoint: no
        // `--api-key-file` should be passed.
        assert_eq!(endpoint_api_key_file_if_valid(&dir.join("missing")), None);
        let empty = dir.join("empty");
        std::fs::write(&empty, "\n").unwrap();
        assert_eq!(endpoint_api_key_file_if_valid(&empty), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn request_roundtrip_preserves_method() {
        let envelope = EngineRequestEnvelope {
            method: EngineMethod::Detect,
            payload: serde_json::json!({ "runtime_id": "therock-release" }),
        };

        let serialized = serde_json::to_string(&envelope).unwrap();
        let parsed: EngineRequestEnvelope = serde_json::from_str(&serialized).unwrap();
        assert_eq!(parsed.method, EngineMethod::Detect);
    }

    #[test]
    fn gpu_selection_parses_auto_variants() {
        assert_eq!(
            GpuSelection::parse_cli_value("auto").unwrap(),
            GpuSelection::Auto
        );
        assert_eq!(
            GpuSelection::parse_cli_value("AUTO").unwrap(),
            GpuSelection::Auto
        );
        assert_eq!(
            GpuSelection::parse_cli_value("  ").unwrap(),
            GpuSelection::Auto
        );
    }

    #[test]
    fn gpu_selection_parses_single_index() {
        assert_eq!(
            GpuSelection::parse_cli_value("1").unwrap(),
            GpuSelection::Index(1)
        );
        assert_eq!(
            GpuSelection::parse_cli_value("  2 ").unwrap(),
            GpuSelection::Index(2)
        );
    }

    #[test]
    fn gpu_selection_rejects_invalid_values() {
        assert!(GpuSelection::parse_cli_value("gpu0").is_err());
        assert!(GpuSelection::parse_cli_value("-1").is_err());
        // Multiple GPUs are no longer supported.
        assert!(GpuSelection::parse_cli_value("0,1").is_err());
        assert!(GpuSelection::parse_cli_value("0,2,3").is_err());
    }

    #[test]
    fn gpu_indices_helpers_handle_auto_and_pinned() {
        assert_eq!(gpu_indices_to_csv(&[]), None);
        assert_eq!(gpu_indices_to_csv(&[1]), Some("1".to_owned()));
        assert!(launch_gpu_indices(None).is_empty());
        assert!(launch_gpu_indices(Some(&GpuSelection::Auto)).is_empty());
        assert_eq!(launch_gpu_indices(Some(&GpuSelection::Index(2))), vec![2]);
    }

    #[test]
    fn apply_gpu_visibility_sets_hip_visible_devices_on_command() {
        let mut command = std::process::Command::new("true");
        apply_gpu_visibility(&mut command, &[2]);
        let hip = command
            .get_envs()
            .find(|(key, _)| *key == std::ffi::OsStr::new("HIP_VISIBLE_DEVICES"))
            .and_then(|(_, value)| value)
            .map(|value| value.to_string_lossy().into_owned());
        assert_eq!(hip.as_deref(), Some("2"));
    }

    #[test]
    fn apply_gpu_visibility_is_noop_without_pinned_indices() {
        let mut command = std::process::Command::new("true");
        apply_gpu_visibility(&mut command, &[]);
        assert!(
            command
                .get_envs()
                .all(|(key, _)| key != std::ffi::OsStr::new("HIP_VISIBLE_DEVICES"))
        );
    }

    #[test]
    fn engine_recipe_hint_roundtrips_through_resolve_request() {
        let request = ResolveModelRequest {
            model_ref: "Qwen/Test".to_owned(),
            runtime_id: None,
            device_policy: Some(DevicePolicy::GpuRequired),
            recipe_override: None,
            engine_recipe: Some(EngineRecipeHint {
                contract_version: ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
                engine: "vllm".to_owned(),
                required_flags: vec!["--enable-auto-tool-choice".to_owned()],
                parser_settings: BTreeMap::from([(
                    "reasoning_parser".to_owned(),
                    "qwen3".to_owned(),
                )]),
                preferred_endpoint: Some(EngineRecipeEndpointHint {
                    endpoint_mode: "openai".to_owned(),
                    settings: BTreeMap::from([("streaming".to_owned(), "true".to_owned())]),
                }),
                unsupported_combinations: vec![EngineRecipeUnsupportedCombinationHint {
                    combination: "native Windows GPU serving".to_owned(),
                    reason: "vLLM ROCm serving is Linux/WSL only".to_owned(),
                }],
                notes: vec!["signed recipe metadata".to_owned()],
            }),
        };

        let serialized = serde_json::to_string(&request).unwrap();
        let parsed: ResolveModelRequest = serde_json::from_str(&serialized).unwrap();

        let hint = parsed.engine_recipe.expect("hint should roundtrip");
        assert_eq!(hint.contract_version, ENGINE_RECIPE_CONTRACT_VERSION);
        assert_eq!(hint.engine, "vllm");
        assert_eq!(hint.required_flags, vec!["--enable-auto-tool-choice"]);
        assert_eq!(
            hint.parser_settings
                .get("reasoning_parser")
                .map(String::as_str),
            Some("qwen3")
        );
        assert_eq!(
            hint.preferred_endpoint
                .as_ref()
                .map(|endpoint| endpoint.endpoint_mode.as_str()),
            Some("openai")
        );
        assert_eq!(hint.unsupported_combinations.len(), 1);
        assert_eq!(hint.notes, vec!["signed recipe metadata"]);
    }

    #[test]
    fn engine_recipe_hint_roundtrips_through_launch_request() {
        let request = LaunchRequest {
            service_id: "svc".to_owned(),
            env_id: None,
            runtime_id: Some("runtime".to_owned()),
            model_ref: "Qwen/Test".to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 11435,
            device_policy: Some(DevicePolicy::GpuRequired),
            endpoint_mode: Some("openai".to_owned()),
            engine_recipe: Some(EngineRecipeHint {
                contract_version: ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
                engine: "vllm".to_owned(),
                required_flags: vec!["--enable-auto-tool-choice".to_owned()],
                parser_settings: BTreeMap::new(),
                preferred_endpoint: None,
                unsupported_combinations: Vec::new(),
                notes: Vec::new(),
            }),
            gpu_selection: None,
        };

        let serialized = serde_json::to_string(&request).unwrap();
        let parsed: LaunchRequest = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            parsed
                .engine_recipe
                .expect("hint should roundtrip")
                .required_flags,
            vec!["--enable-auto-tool-choice"]
        );
    }

    #[test]
    fn plugin_binary_names_normalize_to_engine_ids() {
        assert_eq!(
            engine_id_from_plugin_binary_name("rocm-engine-lemonade"),
            Some("lemonade".to_owned())
        );
        assert_eq!(
            engine_id_from_plugin_binary_name("rocm-engine-vllm.EXE"),
            Some("vllm".to_owned())
        );
        assert_eq!(engine_id_from_plugin_binary_name("rocm-engine-"), None);
        assert_eq!(engine_id_from_plugin_binary_name("rocm"), None);
        assert_eq!(
            engine_id_from_plugin_binary_name("rocm-engine-bad id"),
            None
        );
        assert_eq!(
            engine_id_from_plugin_binary_name("nested/rocm-engine-lemonade"),
            None
        );
    }

    #[test]
    fn discover_engine_plugins_returns_sorted_unique_plugin_binaries() {
        let root = unique_temp_dir("discover");
        let first_dir = root.join("first");
        let second_dir = root.join("second");
        fs::create_dir_all(&first_dir).unwrap();
        fs::create_dir_all(&second_dir).unwrap();

        // The `.exe` suffix is optional in `engine_id_from_plugin_binary_name`, so
        // the bare `rocm-engine-<id>` names are discovered on every platform.
        let first_lemonade = first_dir.join("rocm-engine-lemonade");
        let first_vllm = first_dir.join("rocm-engine-vllm");
        let second_lemonade = second_dir.join("rocm-engine-lemonade");
        fs::write(&first_lemonade, b"lemonade").unwrap();
        fs::write(&first_vllm, b"vllm").unwrap();
        fs::write(&second_lemonade, b"duplicate lemonade").unwrap();
        fs::write(first_dir.join("not-an-engine"), b"ignore").unwrap();
        // A directory matching the plugin naming must be ignored (only files count).
        fs::create_dir_all(second_dir.join("rocm-engine-ghost")).unwrap();

        let plugins =
            discover_engine_plugins([root.join("missing"), first_dir, second_dir]).unwrap();
        let ids = plugins
            .iter()
            .map(|plugin| plugin.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["lemonade", "vllm"]);
        assert_eq!(
            plugins
                .iter()
                .find(|plugin| plugin.id == "lemonade")
                .unwrap()
                .executable_path,
            first_lemonade
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn logs_payload_roundtrip_preserves_tail_request_and_lines() {
        let request = LogsRequest {
            service_id: "svc_qwen35_primary".to_owned(),
            tail_lines: Some(DEFAULT_LOG_TAIL_LINES),
        };

        let serialized = serde_json::to_string(&request).unwrap();
        let parsed: LogsRequest = serde_json::from_str(&serialized).unwrap();
        assert_eq!(parsed.service_id, "svc_qwen35_primary");
        assert_eq!(parsed.tail_lines, Some(DEFAULT_LOG_TAIL_LINES));

        let response = LogsResponse {
            log_path: "/tmp/service.log".to_owned(),
            recent_lines: vec!["ready".to_owned(), "listening".to_owned()],
        };
        let serialized = serde_json::to_string(&response).unwrap();
        let parsed: LogsResponse = serde_json::from_str(&serialized).unwrap();
        assert_eq!(parsed.log_path, "/tmp/service.log");
        assert_eq!(
            parsed.recent_lines,
            vec!["ready".to_owned(), "listening".to_owned()]
        );
    }

    fn unique_temp_dir(label: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "rocm-engine-protocol-{label}-{}-{unique}",
            std::process::id()
        ))
    }
}
