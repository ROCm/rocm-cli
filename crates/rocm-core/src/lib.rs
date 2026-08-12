// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::fs;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(all(test, unix))]
mod archive_props;
pub mod automation;
pub mod binary_paths;
pub mod browser;
pub mod config;
pub mod crypto;
pub mod diagnose;
pub mod disk_space;
pub mod examine;
pub mod fake_executable;
pub mod fix;
pub mod hardware_root;
pub mod host_gpu;
pub mod log_tail;
pub mod managed_runtime;
pub mod model_catalog;
pub mod model_readiness;
pub mod net;
pub mod openmpi;
pub mod os_release;
pub mod paths;
pub mod proc_lifecycle;
pub mod process;
pub mod report;
pub mod report_delivery;
pub mod rocm_install;
pub mod runtime;
pub mod service_record;
pub mod terminal;
pub mod user_path;

#[cfg(test)]
mod test_env;
#[cfg(test)]
mod test_support;
pub mod uv;
pub use automation::{
    AuditEventRecord, AutomationEventRecord, AutomationProposalRecord, AutomationRuntimeState,
    AutomationTriggerEvent, BuiltinWatcherSpec, WatcherRuntimeSnapshot, append_audit_event,
    append_automation_event, append_automation_proposal, builtin_watcher, builtin_watchers,
    find_automation_proposal, generate_proposal_id, load_automation_proposals,
    load_recent_audit_events, load_recent_automation_events, load_recent_automation_proposals,
    update_automation_proposal_status,
};
pub use binary_paths::{daemon_binary_path, engine_binary_path, sibling_binary_path};
pub use config::{
    AutomationsConfig, DashboardConfig, DashboardDaemonConfig, DashboardTuiConfig,
    EngineUserConfig, ManagedToolConfig, PERMISSIONS_MODE_ASK, PERMISSIONS_MODE_FULL_ACCESS,
    PermissionsConfig, ProviderUserConfig, RocmCliConfig, SetupConfig, TELEMETRY_MODE_LOCAL,
    TELEMETRY_MODE_OFF, TelemetryConfig, WatcherMode, WatcherUserConfig,
};
pub use crypto::{
    generate_rsa_signing_keypair, sign_rsa_pkcs1_sha256_signature,
    verify_rsa_pkcs1_sha256_signature,
};
pub use diagnose::{
    DiagnoseReport, Diagnosis, Fix, Route, VLLM_IMPORT_CANONICAL_SYMPTOM,
    VLLM_OOM_CANONICAL_SYMPTOM, diagnose as run_diagnose,
    render_report_text as render_diagnose_text, vllm_import_symptom_is_diagnosable,
    vllm_oom_symptom_is_diagnosable,
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
pub use hardware_root::host_path;
pub use host_gpu::{
    DriverSummary, ExamineSummary, HostGpuSummary, WslHostDriverProbe, WslSummary,
    default_engine_for_host, default_engine_for_platform, detect_gpu_driver_version,
    detect_host_gfx_target, detect_host_gpu_diagnostics, detect_host_gpu_summary,
    detect_system_ram_gib, extract_first_gfx_token, gfx_target_has_no_rocm_build,
    has_usable_amd_gpu, interactive_terminal, is_amdgpu_device, is_wsl_host,
    known_therock_families, normalize_therock_family, preferred_serve_engine_for_host_gpu_summary,
    usable_amd_gpu_indices,
};
#[cfg(any(target_os = "linux", test))]
use host_gpu::{KfdGpuNode, kfd_gpu_nodes_in};
use host_gpu::{
    collect_sdk_library_paths, detect_linux_sysfs_gfx_target, detect_local_windows_host_driver,
    detect_wsl_host_driver, detect_wsl_summary, is_wsl1_kernel, ldconfig_cache,
    ldconfig_lists_librocdxg, managed_therock_sdk_probe_candidates,
};
pub use log_tail::tail_lines;
pub use managed_runtime::{
    FrameworkInterpreter, ManagedRuntimeEnvironment, active_managed_framework_interpreter,
    active_managed_therock_channel, active_managed_therock_environment,
    active_managed_therock_version, detect_managed_therock_family, prepend_runtime_paths,
};
use managed_runtime::{
    TheRockFamilyManifest, managed_sdk_tool_path, managed_therock_environment_records,
    sort_records_newest_install_first,
};
pub use model_catalog::{
    ModelArtifactCacheStatus, ModelCatalogPlatform, ModelRecipeArtifactRecord,
    ModelRecipeArtifactSourcePolicyRecord, ModelRecipeEndpointRecord, ModelRecipeEngineRecord,
    ModelRecipeIndexDocument, ModelRecipeRecord, ModelRecipeRegistry, ModelRecipeRegistrySource,
    ModelRecipeUnsupportedCombinationRecord, builtin_model_recipe_registry, builtin_model_recipes,
    load_model_recipe_index, load_model_recipe_registry, load_signed_model_recipe_index,
    model_artifact_cache_marker_path, model_artifact_cache_status, model_catalog_platforms,
    model_recipe_featured, model_recipe_index_signature_path, model_recipe_target_platform_label,
    normalize_model_ref, platform_matches_gfx_family, resolve_builtin_model_recipe,
    resolve_model_recipe, resolve_model_recipe_artifact, verify_model_recipe_index_signature,
};
pub use net::{
    Backoff, DEFAULT_LOCAL_HOST, DEFAULT_LOCAL_PORT, DOWNLOAD_MAX_ATTEMPTS, DownloadOutcome,
    DownloadRequest, EndpointReadiness, EndpointReadinessOutcome, HttpResponseParts,
    INFERENCE_PROBE_ATTEMPTED_STATE_KEY, INFERENCE_PROBE_RETRY_INTERVAL, INFERENCE_PROBE_TIMEOUT,
    INFERENCE_VERIFIED_STATE_KEY, connect_tcp_stream, download_file_streaming,
    download_file_streaming_with_progress, download_file_to_path,
    download_file_to_path_with_progress, engine_state_inference_verified, format_host_for_url,
    format_host_port, format_http_base_url, http_get_text, http_get_text_with_auth,
    http_post_json_with_auth, managed_service_endpoint_model_ready,
    managed_service_endpoint_readiness, merge_json_state_file, openai_chat_completion_probe,
    openai_chat_completion_status, openai_models_endpoint_has_model, parse_http_endpoint,
    read_http_response_bounded, write_all_tcp_stream,
};
pub use paths::{AppPaths, engine_plugin_dirs};
pub use proc_lifecycle::{
    IdentityState, KillScope, ProcessIdentity, TerminationOutcome, identity_state,
    process_start_ticks, terminate_verified,
};
pub use process::{
    FileLock, detach_command_session, process_is_running, terminate_process, terminate_process_tree,
};
#[cfg(windows)]
pub use process::{
    spawn_detached_no_inherit, spawn_hidden_console_no_inherit, spawn_hidden_console_with_log,
    wait_for_process_exit,
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
pub use service_record::ManagedServiceRecord;
pub use uv::{
    DEFAULT_UV_TIMEOUT_SECS, DependencyViolation, UV_CACHE_DIR_ENV, UV_CACHE_DIR_OVERRIDE_ENV,
    UvCacheSource, ViolationSubject, check_dependencies, ensure_uv_binary, split_local_version,
    uv_binary_name, uv_cache_source, uv_command_env, uv_http_timeout_secs, uv_pip_check_args,
    uv_pip_freeze_args, uv_pip_install_base, uv_venv_args, violation_subject, violations_requiring,
};

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

pub fn require_nonempty(value: &str, field_name: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{field_name} must not be empty");
    }
    Ok(())
}

/// Picks the best available detail string from a failed subprocess's
/// captured output: stderr if non-blank after trimming, else stdout, else a
/// literal "no output"; the result is trimmed.
pub fn command_failure_detail(output: &std::process::Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !stderr.is_empty() {
        return stderr;
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if !stdout.is_empty() {
        return stdout;
    }
    "no output".to_owned()
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
    sort_records_newest_install_first(&mut records);
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

/// Replace `path`'s content by writing a sibling temp file and renaming it
/// over the destination: an interrupted write (full disk, crash, killed
/// process) can never leave a truncated file in place, and a reader sees
/// either the old content or the new content. The destination is NOT removed
/// first: `fs::rename` replaces an existing file on both supported platforms
/// (it is `MoveFileEx` with `MOVEFILE_REPLACE_EXISTING` on Windows), and
/// removing it would open a window with no file at all, which a concurrent
/// reader could see as "file doesn't exist" instead of the old or new
/// content. The temp file is removed if the write or the rename fails.
///
/// `tmp_ext` names the temp file's extension (e.g. `"json"`, `"jsonl"`).
/// Both halves of the generated suffix matter: the timestamp keeps
/// successive writes to the same path apart, and the pid keeps two
/// processes writing in the same millisecond from clobbering each other's
/// temp file and losing one of the two updates.
///
/// Not guaranteed:
/// - Durability: the bytes are not fsynced before the rename, so a power
///   loss can still surface the new name with unflushed contents.
/// - Replacing a destination another process holds open without delete
///   sharing: on Windows the rename then fails with a sharing violation,
///   this returns that error, and the old destination stays in place.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8], tmp_ext: &str) -> Result<()> {
    let tmp_path = path.with_extension(format!(
        "{tmp_ext}.tmp-{}-{}",
        unix_time_millis(),
        std::process::id()
    ));
    fs::write(&tmp_path, bytes)
        .with_context(|| format!("failed to write {}", tmp_path.display()))
        .inspect_err(|_| {
            let _ = fs::remove_file(&tmp_path);
        })?;
    fs::rename(&tmp_path, path)
        .with_context(|| format!("failed to replace {}", path.display()))
        .inspect_err(|_| {
            let _ = fs::remove_file(&tmp_path);
        })?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Managed-service port leasing and the shared allocation lock.
//
// Both `rocm serve` and `rocmd` recovery contend for the same local TCP ports
// and the same service records, so the primitives that arbitrate them live here
// rather than in either binary.
// ---------------------------------------------------------------------------

/// True only for the exact canonical default loopback host.
///
/// Proactive OS collision detection (leasing a real socket before launch) is
/// claimed only for this one host. `localhost`, `::1`, `0.0.0.0`, and other
/// IPv4 addresses are "custom hosts": they require an explicit port and keep the
/// existing engine-owned bind/error behavior, because an IPv4-loopback probe
/// would not prove anything about the address the engine will actually bind.
#[must_use]
pub fn is_canonical_loopback_host(host: &str) -> bool {
    host.trim() == DEFAULT_LOCAL_HOST
}

/// A real, held `127.0.0.1:<port>` listener proving the port is bindable now.
///
/// The lease is the collision check: it is acquired before any manifest, log, or
/// child process exists, and [`LoopbackPortLease::release`] drops the socket
/// immediately before the engine child is spawned so the child can bind it. The
/// window between release and the child's bind is unavoidable without engine
/// socket-passing; it is far narrower than the previous "bind and hope" launch.
#[derive(Debug)]
pub struct LoopbackPortLease {
    port: u16,
    listener: Option<TcpListener>,
}

impl LoopbackPortLease {
    #[must_use]
    pub const fn port(&self) -> u16 {
        self.port
    }

    /// True while this lease still holds the socket open.
    #[must_use]
    pub const fn is_held(&self) -> bool {
        self.listener.is_some()
    }

    /// Close the socket and hand the port to the caller. Call this immediately
    /// before spawning the process that will bind it.
    pub fn release(mut self) -> u16 {
        self.listener = None;
        self.port
    }

    /// A lease over a port this process never probed, for custom hosts where the
    /// engine — not the CLI — owns bind success or failure.
    #[must_use]
    pub const fn engine_owned(port: u16) -> Self {
        Self {
            port,
            listener: None,
        }
    }
}

/// Try to take an exclusive IPv4-loopback lease on `port`.
///
/// Errors are returned verbatim so callers can distinguish "occupied"
/// ([`std::io::ErrorKind::AddrInUse`], the only condition that should advance an
/// automatic scan) from a privileged or unavailable address, which must fail
/// immediately rather than silently choosing a different port.
pub fn lease_loopback_port(port: u16) -> std::io::Result<LoopbackPortLease> {
    let listener = TcpListener::bind(std::net::SocketAddrV4::new(
        std::net::Ipv4Addr::LOCALHOST,
        port,
    ))?;
    Ok(LoopbackPortLease {
        port,
        listener: Some(listener),
    })
}

/// File name of the shared allocation lock, inside [`AppPaths::services_dir`].
///
/// Not a `.json` file, so `load_managed_services` never mistakes it for a
/// service manifest.
pub const SERVICE_ALLOCATION_LOCK_FILE: &str = "allocation.lock";
/// Total time a launch or recovery waits for the allocation lock before failing
/// with an actionable error instead of hanging.
pub const SERVICE_ALLOCATION_LOCK_TIMEOUT: Duration = Duration::from_secs(5);
const SERVICE_ALLOCATION_LOCK_POLL: Duration = Duration::from_millis(25);

#[must_use]
pub fn service_allocation_lock_path(paths: &AppPaths) -> PathBuf {
    paths.services_dir().join(SERVICE_ALLOCATION_LOCK_FILE)
}

/// Cross-process guard serializing the managed-service allocation transaction:
/// duplicate detection, live-reservation refresh, port leasing, child start
/// confirmation, and record publication.
///
/// The lock is an OS advisory file lock, so it is released both by dropping this
/// guard and by process death — a crashed launcher never wedges the next one.
/// Full HTTP/model readiness deliberately runs *outside* the guard: it takes up
/// to 45 seconds and holds no allocation decision.
#[derive(Debug)]
pub struct ServiceAllocationLock {
    file: Option<fs::File>,
    path: PathBuf,
}

impl ServiceAllocationLock {
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ServiceAllocationLock {
    fn drop(&mut self) {
        if let Some(file) = self.file.take() {
            let _ = file.unlock();
        }
    }
}

/// Acquire the shared allocation lock for `paths`, waiting at most
/// [`SERVICE_ALLOCATION_LOCK_TIMEOUT`].
///
/// # Errors
/// Fails when the lock file cannot be created/locked, or when another process
/// held it for the whole timeout.
pub fn lock_service_allocation(paths: &AppPaths) -> Result<ServiceAllocationLock> {
    lock_service_allocation_at(
        &service_allocation_lock_path(paths),
        SERVICE_ALLOCATION_LOCK_TIMEOUT,
    )
}

/// [`lock_service_allocation`] against an explicit path and timeout. Exposed for
/// tests that need a private lock file or a short deadline.
///
/// # Errors
/// See [`lock_service_allocation`].
pub fn lock_service_allocation_at(path: &Path, timeout: Duration) -> Result<ServiceAllocationLock> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let file = fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    let deadline = Instant::now() + timeout;
    loop {
        match file.try_lock() {
            Ok(()) => {
                return Ok(ServiceAllocationLock {
                    file: Some(file),
                    path: path.to_path_buf(),
                });
            }
            Err(std::fs::TryLockError::WouldBlock) => {}
            Err(std::fs::TryLockError::Error(error)) => {
                return Err(
                    anyhow::Error::new(error).context(format!("failed to lock {}", path.display()))
                );
            }
        }
        let now = Instant::now();
        if now >= deadline {
            bail!(
                "timed out after {timeout:?} waiting for the managed-service allocation lock at \
                 {path}; another `rocm serve` launch or `rocmd` recovery is mid-transaction. \
                 Retry in a moment, and check `rocm services` for a stuck launch.",
                path = path.display()
            );
        }
        thread::sleep(SERVICE_ALLOCATION_LOCK_POLL.min(deadline - now));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::workspace_test_artifact_dir;

    use std::fs;
    use std::path::Path;
    use std::process::{Command, Stdio};

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
    fn resolve_amd_smi_binary_in_registry_breaks_install_time_ties_on_the_runtime_key() -> Result<()>
    {
        // Same shape as the active-runtime fallback test: both pairings of file
        // name and runtime key, so a read-order pick is wrong in one of them.
        for (first_file_key, second_file_key) in [("aaa", "bbb"), ("bbb", "aaa")] {
            let temp_root = std::env::temp_dir().join(format!(
                "rocm-cli-amd-smi-tie-{first_file_key}-{}-{}",
                std::process::id(),
                unix_time_millis()
            ));
            let registry_dir = temp_root.join("runtimes/registry");
            fs::create_dir_all(&registry_dir)?;
            for (file, key) in [("first", first_file_key), ("second", second_file_key)] {
                let bin = temp_root.join(key).join("bin");
                fs::create_dir_all(&bin)?;
                fs::write(bin.join("amd-smi"), b"#!/bin/sh\nexit 0\n")?;
                fs::write(
                    registry_dir.join(format!("{file}.json")),
                    serde_json::to_vec(&serde_json::json!({
                        "runtime_key": key,
                        "runtime_id": "therock-stable:gfx94X-dcgpu",
                        "installed_at_unix_ms": 2_000_u128,
                        "rocm_sdk": { "import_ok": true, "bin_path": bin },
                    }))?,
                )?;
            }

            let resolved = resolve_amd_smi_binary_in_registry(&registry_dir);
            let expected = temp_root.join("aaa").join("bin").join("amd-smi");
            let _ = fs::remove_dir_all(&temp_root);

            assert_eq!(
                resolved,
                Some(expected.into_os_string()),
                "files first={first_file_key}, second={second_file_key}"
            );
        }
        Ok(())
    }

    fn test_exit_status() -> std::process::ExitStatus {
        std::process::ExitStatus::default()
    }

    fn test_process_output(stdout: &str, stderr: &str) -> std::process::Output {
        std::process::Output {
            status: test_exit_status(),
            stdout: stdout.as_bytes().to_vec(),
            stderr: stderr.as_bytes().to_vec(),
        }
    }

    #[test]
    fn command_failure_detail_prefers_nonempty_stderr() {
        let output = test_process_output("  stdout text\n", "  stderr text\n");
        assert_eq!(command_failure_detail(&output), "stderr text");
    }

    #[test]
    fn command_failure_detail_falls_back_to_stdout_when_stderr_is_whitespace_only() {
        let output = test_process_output("  stdout text\n", "   \n\t  ");
        assert_eq!(command_failure_detail(&output), "stdout text");
    }

    #[test]
    fn command_failure_detail_reports_no_output_when_both_are_empty_or_whitespace() {
        let output = test_process_output("  \n", "   ");
        assert_eq!(command_failure_detail(&output), "no output");
    }

    #[test]
    fn write_atomic_writes_bytes_and_leaves_no_temp_file() -> Result<()> {
        let temp_root =
            std::env::temp_dir().join(format!("rocm-cli-write-atomic-ok-{}", unix_time_millis()));
        fs::create_dir_all(&temp_root)?;
        let path = temp_root.join("store.jsonl");

        write_atomic(&path, b"hello\n", "jsonl")?;

        let written = fs::read(&path)?;
        let leftovers: Vec<_> = fs::read_dir(&temp_root)?
            .filter_map(Result::ok)
            .map(|entry| entry.file_name())
            .collect();
        let _ = fs::remove_dir_all(&temp_root);

        assert_eq!(written, b"hello\n");
        assert_eq!(
            leftovers,
            vec![OsString::from("store.jsonl")],
            "a successful write must leave no `*.tmp-*` sibling behind"
        );
        Ok(())
    }

    #[test]
    fn write_atomic_rename_failure_leaves_destination_and_dir_untouched() -> Result<()> {
        let temp_root =
            std::env::temp_dir().join(format!("rocm-cli-write-atomic-fail-{}", unix_time_millis()));
        fs::create_dir_all(&temp_root)?;
        let path = temp_root.join("store.jsonl");
        // A file can't be renamed over an existing directory, so making the
        // destination a directory forces `write_atomic`'s rename step to fail
        // after the temp file has already been written.
        fs::create_dir_all(&path)?;

        let result = write_atomic(&path, b"hello\n", "jsonl");
        let destination_is_still_a_dir = path.is_dir();
        let leftovers: Vec<_> = fs::read_dir(&temp_root)?
            .filter_map(Result::ok)
            .map(|entry| entry.file_name())
            .collect();
        let _ = fs::remove_dir_all(&temp_root);

        assert!(
            result.is_err(),
            "renaming a file onto an existing directory must fail"
        );
        assert!(
            destination_is_still_a_dir,
            "a failed rename must leave the destination untouched"
        );
        assert_eq!(
            leftovers,
            vec![OsString::from("store.jsonl")],
            "the temp file must be removed when the rename fails"
        );
        Ok(())
    }

    // -- managed-service allocation lock and loopback leasing ---------------

    const LOCK_CHILD_PATH_ENV: &str = "ROCM_CORE_TEST_LOCK_PATH";
    const LOCK_CHILD_READY_ENV: &str = "ROCM_CORE_TEST_LOCK_READY";
    const HANDOFF_PORT_ENV: &str = "ROCM_CORE_TEST_HANDOFF_PORT";

    fn lock_test_dir(label: &str) -> PathBuf {
        let dir = workspace_test_artifact_dir().join(format!(
            "{label}-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        fs::create_dir_all(&dir).expect("create lock test dir");
        dir
    }

    /// Re-run this test binary, executing exactly one `#[ignore]`d helper test in
    /// a genuinely separate process. Cross-process locking and socket handoff
    /// cannot be proven inside one process.
    fn spawn_helper_test(name: &str, envs: &[(&str, String)]) -> std::process::Child {
        let exe = std::env::current_exe().expect("test binary path");
        let mut command = Command::new(exe);
        command
            .arg(name)
            .arg("--exact")
            .arg("--ignored")
            .arg("--test-threads")
            .arg("1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        for (key, value) in envs {
            command.env(key, value);
        }
        command.spawn().expect("spawn helper test process")
    }

    #[test]
    fn canonical_loopback_host_is_only_the_exact_default_address() {
        assert!(is_canonical_loopback_host(DEFAULT_LOCAL_HOST));
        assert!(is_canonical_loopback_host("  127.0.0.1  "));
        // Everything else is a "custom host": explicit port, engine-owned bind.
        for host in [
            "localhost",
            "::1",
            "0.0.0.0",
            "127.0.0.2",
            "192.168.1.4",
            "",
        ] {
            assert!(
                !is_canonical_loopback_host(host),
                "{host} must not be treated as the canonical loopback host"
            );
        }
    }

    #[test]
    fn allocation_lock_path_lives_in_services_dir_and_is_not_a_manifest() {
        let paths = AppPaths {
            config_dir: PathBuf::from("/tmp/cfg"),
            data_dir: PathBuf::from("/tmp/data"),
            cache_dir: PathBuf::from("/tmp/cache"),
        };
        let path = service_allocation_lock_path(&paths);
        assert_eq!(path.parent(), Some(paths.services_dir().as_path()));
        // `load_managed_services` only reads `*.json`, so the lock must not be one.
        assert_ne!(
            path.extension().and_then(|value| value.to_str()),
            Some("json")
        );
    }

    #[test]
    fn allocation_lock_excludes_independent_handles_and_fails_closed_on_timeout() {
        let path = lock_test_dir("core-alloc-lock-exclusive").join("allocation.lock");
        let held = lock_service_allocation_at(&path, Duration::from_secs(5)).expect("first holder");
        assert_eq!(held.path(), path.as_path());

        // A second, independent file handle must contend on the OS lock rather
        // than sail through, and must give up with an actionable error.
        let started = Instant::now();
        let error = lock_service_allocation_at(&path, Duration::from_millis(200))
            .expect_err("a held lock must not be acquired twice");
        assert!(
            started.elapsed() >= Duration::from_millis(200),
            "waiter returned before its deadline"
        );
        let message = format!("{error:#}");
        assert!(message.contains("timed out"), "{message}");
        assert!(message.contains("allocation lock"), "{message}");

        drop(held);
        lock_service_allocation_at(&path, Duration::from_secs(5))
            .expect("lock is free once the guard is dropped");
    }

    #[test]
    #[ignore = "spawned as a child process by allocation_lock_is_released_when_the_holder_process_exits"]
    fn allocation_lock_child_holder() {
        let Ok(path) = std::env::var(LOCK_CHILD_PATH_ENV) else {
            return;
        };
        let ready = std::env::var(LOCK_CHILD_READY_ENV).expect("ready-file path");
        let guard = lock_service_allocation_at(Path::new(&path), Duration::from_secs(10))
            .expect("child acquires the shared lock");
        fs::write(&ready, "held").expect("signal that the child holds the lock");
        thread::sleep(Duration::from_millis(300));
        // Leak the guard so the lock is released by process exit alone — that is
        // the property a crashed launcher depends on.
        std::mem::forget(guard);
    }

    #[test]
    fn allocation_lock_is_released_when_the_holder_process_exits() {
        let dir = lock_test_dir("core-alloc-lock-crossproc");
        let path = dir.join("allocation.lock");
        let ready = dir.join("child-ready");
        let mut child = spawn_helper_test(
            "tests::allocation_lock_child_holder",
            &[
                (LOCK_CHILD_PATH_ENV, path.display().to_string()),
                (LOCK_CHILD_READY_ENV, ready.display().to_string()),
            ],
        );

        let deadline = Instant::now() + Duration::from_mins(1);
        while !ready.exists() {
            assert!(
                Instant::now() < deadline,
                "child never reported holding the lock"
            );
            thread::sleep(Duration::from_millis(25));
        }

        let busy = lock_service_allocation_at(&path, Duration::from_millis(50))
            .expect_err("the lock must be exclusive across processes");
        assert!(format!("{busy:#}").contains("allocation lock"));

        let status = child.wait().expect("child holder exits");
        assert!(status.success(), "child holder failed: {status}");
        lock_service_allocation_at(&path, Duration::from_secs(10))
            .expect("the OS releases the lock when the holding process exits");
    }

    #[test]
    fn loopback_lease_blocks_a_second_bind_until_released() {
        let ephemeral = TcpListener::bind(("127.0.0.1", 0)).expect("ephemeral bind");
        let port = ephemeral.local_addr().expect("local addr").port();
        drop(ephemeral);

        let lease = lease_loopback_port(port).expect("free port leases");
        assert_eq!(lease.port(), port);
        assert!(lease.is_held());
        let conflict = lease_loopback_port(port).expect_err("a held port must not lease twice");
        assert_eq!(conflict.kind(), std::io::ErrorKind::AddrInUse);

        assert_eq!(lease.release(), port);
        lease_loopback_port(port).expect("the port is bindable again after release");
    }

    #[test]
    fn engine_owned_lease_holds_no_socket() {
        // Custom hosts get a port carrier, never a loopback probe.
        let lease = LoopbackPortLease::engine_owned(8000);
        assert_eq!(lease.port(), 8000);
        assert!(!lease.is_held());
    }

    #[test]
    #[ignore = "spawned as a child process by released_loopback_lease_is_immediately_bindable_by_another_process"]
    fn loopback_handoff_child() {
        let Ok(port) = std::env::var(HANDOFF_PORT_ENV) else {
            return;
        };
        let port: u16 = port.parse().expect("handoff port");
        let lease = lease_loopback_port(port).expect("child binds the released port");
        assert_eq!(lease.port(), port);
    }

    #[test]
    fn released_loopback_lease_is_immediately_bindable_by_another_process() {
        let ephemeral = TcpListener::bind(("127.0.0.1", 0)).expect("ephemeral bind");
        let port = ephemeral.local_addr().expect("local addr").port();
        drop(ephemeral);
        let lease = lease_loopback_port(port).expect("parent leases the port");
        assert_eq!(lease.release(), port);

        let status = spawn_helper_test(
            "tests::loopback_handoff_child",
            &[(HANDOFF_PORT_ENV, port.to_string())],
        )
        .wait()
        .expect("handoff child exits");
        assert!(
            status.success(),
            "a separate process must bind the port immediately after the lease is released"
        );
    }
}
