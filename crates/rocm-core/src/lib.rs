// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(all(test, unix))]
mod archive_props;
pub mod browser;
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
pub mod terminal;
pub mod user_path;

#[cfg(test)]
mod test_env;
#[cfg(test)]
mod test_support;
pub mod uv;
pub use crypto::{
    generate_rsa_signing_keypair, sign_rsa_pkcs1_sha256_signature,
    verify_rsa_pkcs1_sha256_signature,
};
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
pub use terminal::{quotable_in_single_quotes, strip_terminal_control_sequences};
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

    /// Persist the config by replacing it: write a sibling temp file, then
    /// rename it over `config.json`.
    ///
    /// Guarantees: an interrupted save (full disk, crash, killed process) can
    /// never leave a truncated or empty `config.json`, which `load` refuses to
    /// parse, and a reader sees either the old file or the new one. The
    /// destination is never removed first, because `load` reads a missing
    /// `config.json` as "no config" and answers with defaults.
    ///
    /// The temp file is `config.json.tmp-<millis>-<pid>`; the pid keeps two
    /// `rocm` processes saving in the same millisecond apart. It is removed if
    /// the write or the rename fails.
    ///
    /// Not guaranteed:
    /// - Durability. The bytes are not fsynced before the rename, so a power
    ///   loss can still surface the new name with unflushed contents.
    /// - Replacing a `config.json` that another process holds open without
    ///   delete sharing. On Windows the rename then fails with a sharing
    ///   violation, the save returns that error, and the old config stays in
    ///   place.
    pub fn save(&self, paths: &AppPaths) -> Result<()> {
        let path = paths.config_path();
        fs::create_dir_all(&paths.config_dir)
            .with_context(|| format!("failed to create {}", paths.config_dir.display()))?;
        let bytes =
            serde_json::to_vec_pretty(self).context("failed to serialize rocm-cli config")?;
        // Both halves of the suffix matter: the timestamp keeps successive saves
        // apart, and the pid keeps two `rocm` processes saving in the same
        // millisecond from writing each other's temp file and losing one of the
        // two updates.
        let tmp_path = path.with_extension(format!(
            "json.tmp-{}-{}",
            unix_time_millis(),
            std::process::id()
        ));
        // Cleaned up on failure, like the rename below: a full disk otherwise
        // strands a partial `config.json.tmp-<ts>-<pid>` next to the real config
        // on every attempt.
        fs::write(&tmp_path, bytes)
            .with_context(|| format!("failed to write {}", tmp_path.display()))
            .inspect_err(|_| {
                let _ = fs::remove_file(&tmp_path);
            })?;
        // The destination is NOT removed first: `fs::rename` replaces an
        // existing file on both supported platforms (it is `MoveFileEx` with
        // `MOVEFILE_REPLACE_EXISTING` on Windows), and removing it would open a
        // window with no `config.json` at all — which `load` reads as "no
        // config" and silently answers with defaults, losing every setting
        // without so much as an error.
        fs::rename(&tmp_path, &path)
            .with_context(|| {
                format!(
                    "failed to move rocm-cli config {} into {}",
                    tmp_path.display(),
                    path.display()
                )
            })
            .inspect_err(|_| {
                let _ = fs::remove_file(&tmp_path);
            })?;
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

    /// Record `pid` as the supervisor, together with its start-time token.
    ///
    /// Call it while the process is known to be alive — just after spawning
    /// it. A stop verifies a recorded PID against its token before signalling
    /// it, and [`crate::identity_state`] treats a PID recorded *without* one as
    /// a match, so a PID written on its own is signalled on trust alone once it
    /// has been recycled. Writing the two together is what keeps that from
    /// happening at whichever site records a PID next.
    pub fn record_supervisor_identity(&mut self, pid: u32) {
        self.supervisor_pid = pid;
        self.supervisor_start_ticks = crate::process_start_ticks(pid);
    }

    /// Whether `self` and `other` — two snapshots of one service's record —
    /// name the same processes, token for token.
    ///
    /// A stop reads the record, signals what it names, and only then writes its
    /// verdict. If the record names different processes by then — a restart
    /// recorded its new supervisor meanwhile — the verdict does not cover them,
    /// and both `rocm services stop` and `rocmd`'s stop gate their "stopped"
    /// claim on this.
    pub fn names_same_processes(&self, other: &Self) -> bool {
        self.supervisor_pid == other.supervisor_pid
            && self.supervisor_start_ticks == other.supervisor_start_ticks
            && self.engine_pid == other.engine_pid
            && self.engine_start_ticks == other.engine_start_ticks
    }

    /// Record `pid` as the engine, together with its start-time token. See
    /// [`Self::record_supervisor_identity`].
    pub fn record_engine_identity(&mut self, pid: u32) {
        self.engine_pid = Some(pid);
        self.engine_start_ticks = crate::process_start_ticks(pid);
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
        // `requested_runtime_id` is the selector the launch asked for: an exact
        // runtime key or a family id, whichever was requested. `runtime_id` is
        // only what the engine resolved that to, which for a TheRock runtime is
        // the manifest's family id (`therock-release:gfx120X-all`), shared by
        // every installed version of that family. When the launch pinned an
        // exact key, adopting the resolved form would throw away the version
        // the service is really on, and the record is what a restart re-pins
        // and what runtime activation compares against. Prefer the requested
        // value, falling back to the resolved one for engines that record only
        // that. A blank value in either field is ignored.
        if let Some(runtime_id) = ["requested_runtime_id", "runtime_id"]
            .into_iter()
            .filter_map(|key| state.get(key).and_then(serde_json::Value::as_str))
            .find(|value| !value.trim().is_empty())
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
            && let Ok(engine_pid) = u32::try_from(pid)
        {
            self.engine_pid = Some(engine_pid);
            // When `server_pid` names the same process as `pid`, `start_ticks`
            // is that process's token too. vLLM wrote its state that way —
            // `server_pid` equal to `pid`, and only `start_ticks` — before it
            // wrote `server_start_ticks`, so without this the PID of every such
            // service was adopted with no token, and a stop could only signal
            // it blind once the PID had been recycled.
            self.engine_start_ticks = state
                .get(ticks_key)
                .and_then(serde_json::Value::as_u64)
                .or_else(|| {
                    (state.get("pid").and_then(serde_json::Value::as_u64) == Some(pid))
                        .then(|| state.get("start_ticks").and_then(serde_json::Value::as_u64))
                        .flatten()
                });
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
///
/// The stated bound must be the one `rocm serve` actually accepts:
/// `parse_gpu_memory_utilization` rejects `<= 0` and accepts `1`, so the domain
/// is `(0, 1]` and the older `<0-1>` wording wrongly advertised `0`. Pinned by
/// `the_utilization_hint_states_the_bound_the_parser_accepts`, because this
/// wording has already been reverted once by a merge that resolved the line from
/// a pre-fix tree.
pub const VLLM_GPU_MEMORY_UTILIZATION_HINT: &str = "vLLM reserves ~90% of the GPU's total VRAM by default; on a shared or busy GPU this can \
     collide with memory already in use. Lower the reservation with `--gpu-memory-utilization \
     <fraction greater than 0 and at most 1>` (e.g. 0.5 for a small model), or target a less-busy \
     GPU with `--gpu <index>`.";

/// Lead-in of the vLLM engine's own post-failure out-of-memory hint.
///
/// Public because two crates have to agree on it byte for byte:
/// `rocm-engine-vllm` renders its hint with this prefix, and
/// [`vllm_oom_diagnose_symptom`] recognises it in order *not* to quote
/// rocm-cli's own sentence back at the user as though the engine had printed
/// it. See [`is_rocm_authored_log_line`].
pub const VLLM_OOM_DETECTED_PREFIX: &str = "Detected an out-of-memory failure (";

/// The `rocm diagnose` invocation the OOM surfaces print, as it appears in text.
///
/// Shared for the same reason as [`VLLM_OOM_DETECTED_PREFIX`]: it is how a line
/// rocm-cli authored is told apart from a line the engine printed.
pub const VLLM_OOM_DIAGNOSE_COMMAND: &str = "rocm diagnose --symptom";

/// Whether this log segment is text *rocm-cli itself* wrote into the engine log,
/// rather than output the engine produced.
///
/// A managed launch shares one log between the supervisor and the engine: when
/// vLLM OOMs, the engine bails with its startup-log context — which already
/// carries this crate's own OOM hint — and the CLI then writes that whole error
/// back into the same file. The hint quotes the user's failing line and names
/// the allocator keywords, so it scores as an OOM just as strongly as the real
/// line does, and it lands *after* it. A newest-first scan would therefore
/// always select rocm-cli's own sentence, which contains `'` and backticks, so
/// [`quotable_in_single_quotes`] would reject it and the printed command would
/// fall back to [`VLLM_OOM_CANONICAL_SYMPTOM`] every single time in real use —
/// exactly the "quote the user's real line" promise the note is built on.
///
/// Matching on rocm-cli's own wording is deliberate, rather than truncating the
/// tail at the engine's `Error:` block: the error block also *echoes* the log
/// tail verbatim, and those echoed lines are the user's real output and remain
/// legitimate choices. Only the sentences this crate composed are excluded.
/// vLLM never prints either marker.
///
/// Pinned by `the_engine_hint_is_never_quoted_back_as_the_users_own_line`.
fn is_rocm_authored_log_line(segment: &str) -> bool {
    segment.contains(VLLM_OOM_DETECTED_PREFIX) || segment.contains(VLLM_OOM_DIAGNOSE_COMMAND)
}

/// Whether one log segment carries a genuine out-of-memory failure, scored with
/// the *same* rule as the `rocm diagnose` vLLM-OOM checker (`check_16_vllm_oom`,
/// reached through [`vllm_oom_symptom_is_diagnosable`]) so the engine hint, the
/// post-failure serve summary and `rocm diagnose` never disagree on what counts
/// as a vLLM OOM. The segment is vLLM's own process output, so the checker's
/// required vLLM anchor is supplied by context: it is scored as
/// `vllm: <segment>` and clears the checker's `MIN_SCORE_FOR_MATCH` or not.
///
/// This deliberately rejects a bare `out of memory` with no allocator-shaped
/// signature — a kernel OOM-killer line, a dependency's log, or an unrelated
/// subprocess — the same false positive the checker's threshold guards against,
/// rather than driving the user-facing memory note off any stray OOM substring
/// in the tail. A real vLLM/HIP allocation failure (`HIP out of memory`,
/// `CUDA out of memory`, `hipErrorOutOfMemory`, or `torch.OutOfMemoryError`
/// corroborated by an allocator message) still clears it.
///
/// Scoring each [`terminal::rendered_lines`] row of the segment, rather than the
/// segment whole, is what actually keeps this in step with the checker.
/// [`vllm_anchored_lines`] re-splits the string it is handed with
/// `rendered_lines` and keeps only the rows carrying the `vllm` anchor — and
/// `rendered_lines` ends a row on far more than the `\n`/`\r` that
/// [`vllm_log_segments`] splits on: every non-`SGR` CSI (`ESC[2K`, `ESC[1A`),
/// `ESC E`/`ESC D`, `VT`, `FF`, `U+0085` and `U+2028`/`U+2029`. So a real
/// repaint such as `Loading weights: 90%\r\x1b[2Ktorch.OutOfMemoryError: ...`
/// becomes the segment `\x1b[2Ktorch.OutOfMemoryError: ...`, which the checker
/// renders as `["vllm: ", "torch.OutOfMemoryError: ..."]`. Only the *empty*
/// anchored row would be scored, the tail would read as "no OOM", and both the
/// engine hint and the serve-summary note would silently vanish for the shape of
/// failure they exist to explain. Anchoring each rendered row instead closes
/// that gap for the whole break set at once, rather than for `\r` alone.
///
/// This costs nothing on the quoting side, and provably so: a segment that
/// `rendered_lines` splits into more than one row must contain a control
/// character or an `ESC`, and [`quotable_in_single_quotes`] rejects every one of
/// those. So a multi-row segment can only ever reach the printed command as the
/// canonical fallback — it can never be stripped into a runnable lookalike of
/// the user's error. The raw segment, control bytes intact, is still what
/// [`vllm_oom_diagnose_symptom`] returns, so that guard keeps working.
fn segment_shows_oom(segment: &str) -> bool {
    terminal::rendered_lines(segment)
        .iter()
        .any(|row| diagnose::vllm_oom_symptom_is_diagnosable(&format!("vllm: {row}")))
}

/// The log tail split into candidate symptom lines.
///
/// Splits on `\r` as well as `\n` because vLLM draws its progress bar with bare
/// carriage returns, so a real failure arrives as
/// `Loading weights: 90%\rtorch.OutOfMemoryError: ...` on a single `\n` line and
/// the exception has to become a candidate in its own right.
///
/// Deliberately a raw split and *not* [`terminal::rendered_lines`], even though
/// that is what the diagnostic re-splits with. `rendered_lines` also discards
/// escape sequences and other control bytes, and the segment returned here
/// becomes the `--symptom` argument: sanitising it means
/// [`quotable_in_single_quotes`] only ever sees clean text, never rejects it,
/// and the canonical-symptom fallback stops firing — so a colourised OOM line
/// would be stripped into a lookalike and printed rather than refused. Keeping
/// the raw bytes here and doing the *scoring* on rendered rows inside
/// [`segment_shows_oom`] is what lets both properties hold at once.
///
/// The bound on that guarantee is worth stating, because the obvious reading of
/// it is wrong. It is a property of *this read of this text*: whatever text a
/// caller hands in, a colourised OOM line inside it is refused and the canonical
/// symptom printed. It is **not** an end-to-end property of a managed launch,
/// and the reason is the engine's own bail. `startup_log_context` echoes the log
/// tail as [`terminal::rendered_lines`] rows — it has to, since that text is
/// printed to a terminal — and the supervisor appends that whole bail to the
/// same shared log. The echo is then the *only* copy of the failing line inside
/// the CLI's tail window: measured on a 200-line log, the engine's raw
/// colourised line ends up 84 lines from the end and the window is
/// `DEFAULT_LOG_TAIL_LINES` = 80. So the serve summary, which reads after the
/// bail, selects the rendered row and quotes it instead of falling back.
///
/// That outcome is sound — the row is text the terminal really drew, it carries
/// no control bytes or quotes, and quotable still implies diagnosable — but it
/// does mean the engine and the CLI can print *different* `--symptom` arguments
/// for the same colourised failure: the engine the canonical symptom, the CLI
/// the user's rendered line. Filtering the echo out to force agreement is not an
/// option: measured, it leaves no OOM line in the window at all and the serve
/// note disappears for every real managed launch. Pinned, both halves, by
/// `the_colourised_round_trip_quotes_the_row_the_terminal_drew`.
fn vllm_log_segments(log: &str) -> impl Iterator<Item = String> + '_ {
    log.lines()
        .flat_map(|line| line.split('\r'))
        .map(|segment| segment.trim().to_owned())
        .filter(|segment| !segment.is_empty())
}

/// The `rocm diagnose --symptom` string to route a vLLM OOM log tail to, or
/// `None` when the tail carries no OOM line to route.
///
/// Returns the user's actual failing *raw* line (so `diagnose` echoes their real
/// error), selected with [`segment_shows_oom`], which is the checker's own rule.
/// Shared by the vLLM engine's post-failure hint and the `rocm` CLI serve summary
/// so both surfaces apply one selection rule to the text they are given, instead
/// of hand-rolling it twice. It is also the OOM *gate* for both: `None` is the
/// "this tail is not an OOM" answer, so no caller needs a separate predicate
/// first.
///
/// Sharing the rule is not the same as the two surfaces always printing the same
/// command, and the difference is real rather than theoretical: they read the
/// shared log at different points, and the engine's bail puts a sanitized echo of
/// the tail into it in between. See [`vllm_log_segments`] for the measurement and
/// for why closing the gap would cost the serve note entirely.
///
/// The exact guarantee is narrower than "whatever comes back is diagnosable",
/// and the difference is load-bearing. What holds is:
///
/// > if [`quotable_in_single_quotes`] accepts the returned value, then
/// > [`vllm_oom_symptom_is_diagnosable`] accepts it too.
///
/// That is what the callers need, because quotability is what they gate the
/// printed command on, and it is exactly enough: a returned line that is *not*
/// quotable is replaced by [`VLLM_OOM_CANONICAL_SYMPTOM`], which is diagnosable,
/// so the printed command always reports a cause either way.
///
/// The stronger claim is false, and necessarily so. A segment carrying an
/// escape — `\x1b[2Ktorch.OutOfMemoryError: ...` from a progress-bar repaint —
/// is scored by [`segment_shows_oom`] on its rendered rows and is a genuine OOM,
/// but handed to the checker *whole* it splits into `["vllm: ", "torch..."]`,
/// leaving the anchor on the empty row and scoring 0. Returning the rendered row
/// instead would make the claim true and reintroduce the hole it exists to
/// close: the stripped row is quotable, so a colourised line would be printed as
/// a runnable lookalike of the user's error rather than refused. The raw bytes
/// are kept, the quoting guard does its job, and the fallback covers the rest.
/// Pinned by `every_rendered_line_break_keeps_the_gate_in_step_with_the_scorer`.
///
/// Lines rocm-cli wrote into the log itself are skipped — see
/// [`is_rocm_authored_log_line`] for why the newest-first scan would otherwise
/// always land on this crate's own hint instead of the user's error.
///
/// Scanning with `.rev()` is deliberate, not incidental: when several lines of
/// the tail mention running out of memory, the *last* one is the proximate
/// failure — the point where the process actually gave up — and the earlier ones
/// are usually the allocator's own retry chatter leading up to it. The cost is
/// real and worth stating: on a tail like "torch.OutOfMemoryError: ... Tried to
/// allocate 7.21 GiB. GPU 0 has a total capacity of 24.00 GiB." followed by a
/// bare "RuntimeError: ... killed: out of memory", the vaguer line wins even
/// though the first carries the allocation size and the capacity. Both are
/// diagnosable, so this picks which line is quoted, never whether one is. The
/// user still sees every line: callers print the whole log tail beside the hint.
///
/// The return value is *log text*, not a shell-safe token: some callers only
/// display it. A caller that renders it inside a quoted command must gate it on
/// [`quotable_in_single_quotes`] first and fall back to
/// [`VLLM_OOM_CANONICAL_SYMPTOM`] otherwise. Guaranteeing quotability here
/// instead would impose the command-builder's constraint on the display-only
/// callers, silently withholding the user's real error from text that is never
/// pasted anywhere.
#[must_use]
pub fn vllm_oom_diagnose_symptom(log_tail: &str) -> Option<String> {
    vllm_log_segments(log_tail)
        .filter(|segment| !is_rocm_authored_log_line(segment))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .find(|segment| segment_shows_oom(segment))
        .map(|segment| format!("vllm: {segment}"))
}

/// The text to *show* the user for a segment [`vllm_oom_diagnose_symptom`]
/// selected, with the `vllm: ` anchor already removed by the caller.
///
/// Not the same operation as stripping the segment's control bytes, and the
/// difference is the whole point. [`strip_terminal_control_sequences`] deletes
/// every break token — `\r`, `ESC[1A`, `NEL`, `VT`, `U+2028`… — *without a
/// separator*, so a progress-bar repaint such as
/// `Loading weights: 90%\x1b[1Atorch.OutOfMemoryError: HIP out of memory` comes
/// back as the single run-together line
/// `Loading weights: 90%torch.OutOfMemoryError: HIP out of memory`. That string
/// is text vLLM never printed: two rows the terminal drew separately, welded
/// into one. Displaying it beside "Detected an out-of-memory failure" presents a
/// fabricated line as the user's own output, and — once rocm-cli writes its bail
/// back into the shared log — a *later* read finds that welded line, scores it
/// as an OOM, finds it quotable (the control bytes are gone) and quotes it into
/// the `rocm diagnose --symptom` command the user is invited to paste.
///
/// So this returns the [`terminal::rendered_lines`] row that actually scored —
/// the same per-row rule [`segment_shows_oom`] gates on — which is one row the
/// terminal really drew. When no row scores (the caller passed a segment that is
/// not an OOM at all) it falls back to stripping, which is lossless for the
/// single-row case that is by far the common one.
///
/// When more than one row scores, the **last** one is taken, for the same reason
/// [`vllm_oom_diagnose_symptom`] scans its segments with `.rev()`: the newest
/// line is the proximate failure and the earlier ones are the allocator's retry
/// chatter leading up to it. Picking the first instead would make the display
/// disagree with the selector about which line "the failure" is, inside the one
/// message that prints both.
///
/// Pinned by `the_displayed_oom_line_is_a_row_the_terminal_actually_drew`.
#[must_use]
pub fn vllm_oom_display_line(segment: &str) -> String {
    terminal::rendered_lines(segment)
        .into_iter()
        .rev()
        .find(|row| diagnose::vllm_oom_symptom_is_diagnosable(&format!("vllm: {row}")))
        .unwrap_or_else(|| terminal::strip_terminal_control_sequences(segment))
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
    use crate::test_support::{probe_test_record, temp_app_paths};

    use std::fs;
    use std::path::Path;
    use std::path::PathBuf;

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

    /// Writes `state` as the record's engine state file and runs the refresh.
    fn refresh_from_state(name: &str, state: &serde_json::Value) -> ManagedServiceRecord {
        let dir = std::env::temp_dir().join(format!(
            "rocm-core-engine-state-{name}-{}-{}",
            std::process::id(),
            crate::unix_time_millis()
        ));
        fs::create_dir_all(&dir).expect("create state dir");
        let mut record = probe_test_record(11437);
        record.status = "running".to_owned();
        record.engine_state_path = dir.join("state.json");
        fs::write(
            &record.engine_state_path,
            serde_json::to_vec(state).expect("serialize state"),
        )
        .expect("write state");
        record.refresh_from_engine_state().expect("refresh");
        let _ = fs::remove_dir_all(dir);
        record
    }

    /// vLLM records one process as both `pid` and `server_pid` with only
    /// `start_ticks`. The refresh prefers `server_pid`, so it has to find that
    /// process's token under `start_ticks`, or the PID is adopted with none.
    #[test]
    fn engine_refresh_takes_start_ticks_when_server_pid_is_the_same_process() {
        let record = refresh_from_state(
            "same-pid",
            &serde_json::json!({
                "status": "running",
                "pid": 4321,
                "server_pid": 4321,
                "start_ticks": 987_u64,
            }),
        );
        assert_eq!(record.engine_pid, Some(4321));
        assert_eq!(record.engine_start_ticks, Some(987));
    }

    /// The fallback is only for the same process: a distinct server PID never
    /// borrows the launcher's token, which would make the stop's identity check
    /// refute the server and leave it running.
    #[test]
    fn engine_refresh_never_gives_a_server_pid_the_launchers_token() {
        let record = refresh_from_state(
            "distinct-pid",
            &serde_json::json!({
                "status": "running",
                "pid": 100,
                "server_pid": 200,
                "start_ticks": 111_u64,
            }),
        );
        assert_eq!(record.engine_pid, Some(200));
        assert_eq!(record.engine_start_ticks, None);

        let record = refresh_from_state(
            "distinct-pid-own-token",
            &serde_json::json!({
                "status": "running",
                "pid": 100,
                "server_pid": 200,
                "start_ticks": 111_u64,
                "server_start_ticks": 222_u64,
            }),
        );
        assert_eq!(record.engine_start_ticks, Some(222));
    }

    /// Each role is written with its own PID's token, so a stop can verify it.
    /// A role recorded without one is signalled on trust alone once its PID has
    /// been recycled. One live process stands in for both roles, as it does for
    /// a `rocm serve --background` launch.
    ///
    /// Linux-only: `process_start_ticks` reads `/proc`, and is `None` elsewhere.
    #[cfg(target_os = "linux")]
    #[test]
    fn recording_a_process_identity_writes_the_pid_with_its_own_token() {
        let mut child = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .expect("spawn child");
        let pid = child.id();
        let expected = crate::process_start_ticks(pid);
        let mut record = probe_test_record(11436);

        record.record_supervisor_identity(pid);
        record.record_engine_identity(pid);
        let _ = child.kill();
        let _ = child.wait();

        assert!(expected.is_some(), "precondition: a live child has a token");
        assert_eq!(record.supervisor_pid, pid);
        assert_eq!(record.supervisor_start_ticks, expected);
        assert_eq!(record.engine_pid, Some(pid));
        assert_eq!(record.engine_start_ticks, expected);
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

    /// The OOM gate both production surfaces actually use. There is no separate
    /// classifier predicate: `vllm_oom_diagnose_symptom(..).is_some()` *is* the
    /// gate, so these threshold assertions are made against the function that
    /// ships rather than against a parallel one no caller reaches.
    fn log_shows_oom(log: &str) -> bool {
        vllm_oom_diagnose_symptom(log).is_some()
    }

    #[test]
    fn vllm_oom_gate_matches_allocator_signatures_but_not_generic_failures() {
        // A real allocator message (the `MIN_SCORE_FOR_MATCH`-clearing shape the
        // `rocm diagnose` vLLM-OOM checker recognises) is a genuine OOM, whichever
        // module path the exception carries.
        assert!(log_shows_oom(
            "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB."
        ));
        assert!(log_shows_oom(
            "torch.cuda.OutOfMemoryError: CUDA out of memory"
        ));
        assert!(log_shows_oom("RuntimeError: hipErrorOutOfMemory"));
        // Case-insensitive.
        assert!(log_shows_oom("HIP OUT OF MEMORY"));

        // Sub-threshold on their own — the same lines the diagnose checker keeps
        // below `MIN_SCORE_FOR_MATCH` so a stray OOM phrase does not carry the
        // verdict. Detecting these here would contradict that checker (they are
        // the false positives the shared threshold exists to reject).
        //
        // The bare exception class every PyTorch OOM raises, with no allocator
        // message to corroborate it.
        assert!(!log_shows_oom("TORCH.OUTOFMEMORYERROR"));
        // A spaced "out of memory" with no allocator anchor phrase.
        assert!(!log_shows_oom("HIP error: out of memory"));
        // A kernel OOM-killer line from an unrelated subprocess must NOT drive
        // the vLLM memory note.
        assert!(!log_shows_oom(
            "Out of memory: Killed process 4242 (python)"
        ));
        // The generic EngineCore wrapper is NOT an OOM signature.
        assert!(!log_shows_oom(
            "EngineCore failed: engine core initialization failed"
        ));
        assert!(!log_shows_oom("OSError: model weights not found"));
        assert!(!log_shows_oom(""));
    }

    #[test]
    fn every_rendered_line_break_keeps_the_gate_in_step_with_the_scorer() {
        // The scorer (`vllm_anchored_lines`) re-splits with `rendered_lines`,
        // which ends a row on far more than the `\n`/`\r` the segmenter splits
        // on. Splitting the log on `\r` alone therefore left the anchor on an
        // empty leading piece for every *other* break, the tail scored as "no
        // OOM", and both the engine hint and the serve-summary note silently
        // vanished. Each prefix below is a separate `rendered_lines` boundary.
        let exception = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";
        for (label, break_seq) in [
            ("bare CR", "\r"),
            ("erase-line CSI after a CR repaint", "\r\u{1b}[2K"),
            ("cursor-up CSI", "\u{1b}[1A"),
            ("NEL escape", "\u{1b}E"),
            ("IND escape", "\u{1b}D"),
            ("vertical tab", "\u{b}"),
            ("form feed", "\u{c}"),
            ("C1 NEL", "\u{85}"),
            ("line separator", "\u{2028}"),
            ("paragraph separator", "\u{2029}"),
        ] {
            let tail = format!("Loading weights: 90%{break_seq}{exception}\n");
            assert!(
                log_shows_oom(&tail),
                "{label}: an allocator exception after this break must still read as an OOM"
            );
            let symptom = vllm_oom_diagnose_symptom(&tail)
                .unwrap_or_else(|| panic!("{label}: an OOM tail must select a line"));

            // The contract the callers actually rely on: *if* the line is
            // quotable it must be diagnosable, because quotability is what they
            // gate the printed command on. When it is not quotable the canonical
            // symptom stands in, and that is diagnosable by definition — so the
            // printed command reports a cause on every branch.
            //
            // The stronger "the returned line is always diagnosable" does NOT
            // hold here and must not be asserted: a segment carrying an escape
            // scores on its rendered rows but splits into an empty anchored row
            // when the checker is handed it whole. Making that claim true would
            // mean returning the stripped row, which is quotable, and a
            // colourised line would then be printed as a runnable lookalike of
            // the user's error instead of being refused.
            if terminal::quotable_in_single_quotes(&symptom) {
                assert!(
                    diagnose::vllm_oom_symptom_is_diagnosable(&symptom),
                    "{label}: a quotable selected line must be diagnosable"
                );
            } else {
                assert!(
                    diagnose::vllm_oom_symptom_is_diagnosable(VLLM_OOM_CANONICAL_SYMPTOM),
                    "{label}: the fallback the callers substitute must report a cause"
                );
            }

            // Either way the user still sees their real error: the raw bytes
            // survive selection, so rendering for display recovers the
            // exception rather than the progress-bar text that preceded it.
            //
            // Asserted as exact equality, not `contains`. `contains` passed on
            // the welded `Loading weights: 90%torch.OutOfMemoryError: ...` that
            // plain stripping produced, so it could not fail on the very bug
            // this line is here to catch.
            let displayed =
                vllm_oom_display_line(symptom.strip_prefix("vllm: ").unwrap_or(&symptom));
            assert_eq!(
                displayed, exception,
                "{label}: the displayed line must be the exception row on its own, with no \
                 progress-bar text welded onto its front"
            );

            // An escape-bearing segment must never pass the quoting guard, or it
            // would reach the printed command stripped into a lookalike.
            if break_seq.contains('\u{1b}') {
                assert!(
                    !terminal::quotable_in_single_quotes(&symptom),
                    "{label}: an escape-bearing symptom must not pass the quoting guard"
                );
            }
        }
    }

    #[test]
    fn the_displayed_oom_line_is_a_row_the_terminal_actually_drew() {
        let exception = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";

        // The bug: `strip_terminal_control_sequences` deletes the break token
        // with nothing in its place, welding the progress bar onto the front of
        // the exception. The welded string is text vLLM never emitted, and it is
        // quotable (no control bytes left), so it would reach the printed
        // `rocm diagnose --symptom` command as a runnable lookalike.
        let repaint = format!("Loading weights: 90%\u{1b}[1A{exception}");
        let welded = terminal::strip_terminal_control_sequences(&repaint);
        assert_eq!(
            welded,
            format!("Loading weights: 90%{exception}"),
            "this pins the shape of the bug, so the assertion below cannot pass vacuously"
        );
        assert!(
            terminal::quotable_in_single_quotes(&welded),
            "the welded line is quotable, which is exactly why it must never be displayed \
             or written back into the shared log"
        );

        assert_eq!(
            vllm_oom_display_line(&repaint),
            exception,
            "the displayed line must be the rendered row that scored, not the whole segment \
             with its break tokens deleted"
        );

        // A plain single-row line is unchanged, so the common case is lossless.
        assert_eq!(vllm_oom_display_line(exception), exception);
        // A line that is not an OOM at all has no scoring row to pick, so the
        // helper still returns something printable rather than an empty string.
        assert_eq!(
            vllm_oom_display_line("\u{1b}[31mOSError: model weights not found\u{1b}[0m"),
            "OSError: model weights not found"
        );

        // When several rows of one segment score, the newest wins — the same
        // choice `vllm_oom_diagnose_symptom` makes with `.rev()` across
        // segments. The two run inside one message, so picking opposite ends
        // would have the hint display one line while the `--symptom` argument
        // names another.
        let retry = "RuntimeError: hipErrorOutOfMemory";
        let proximate = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";
        assert!(
            diagnose::vllm_oom_symptom_is_diagnosable(&format!("vllm: {retry}"))
                && diagnose::vllm_oom_symptom_is_diagnosable(&format!("vllm: {proximate}")),
            "both rows must score, or this fixture cannot discriminate which end is taken"
        );
        assert_eq!(
            vllm_oom_display_line(&format!("{retry}\u{1b}[1A{proximate}")),
            proximate,
            "the displayed row must be the last one that scored, not the first"
        );
    }

    #[test]
    fn the_engine_hint_is_never_quoted_back_as_the_users_own_line() {
        // A managed launch shares one log between the supervisor and the engine,
        // and the engine's bail (which embeds this crate's own OOM hint) is
        // written back into it *after* the real failure. That hint quotes the
        // user's line and names the allocator keywords, so it scores as an OOM
        // too — and being last, a newest-first scan always selected it. Since it
        // contains `'`, the quoting guard then rejected it and the printed
        // command fell back to the canonical symptom in every real run, which is
        // precisely the promise the note is built on.
        let real_line = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";
        let tail = format!(
            "{real_line}\n\
             Error: vLLM server process exited before becoming ready (status: exit status: 1)\n\
             \n\
             Startup log, last 50 lines as the terminal drew them:\n\
             {real_line}\n\
             \n\
             {VLLM_OOM_DETECTED_PREFIX}{real_line}). {VLLM_GPU_MEMORY_UTILIZATION_HINT}\n\
             For conditional remediation, run `{VLLM_OOM_DIAGNOSE_COMMAND} \
             'vllm: {real_line}'`.\n"
        );
        let symptom = vllm_oom_diagnose_symptom(&tail).expect("the real OOM line is still found");
        assert_eq!(
            symptom,
            format!("vllm: {real_line}"),
            "the user's own failing line must win over rocm-cli's own hint"
        );
        // ...and because it is the user's real line, it is quotable, so the
        // command the user is told to run actually carries their error.
        assert!(
            terminal::quotable_in_single_quotes(&symptom),
            "the selected line must survive the quoting guard, or the note falls \
             back to the canonical symptom exactly as it did before this fix"
        );
    }

    #[test]
    fn vllm_oom_diagnose_symptom_selects_the_failing_line_or_reports_no_oom() {
        // The selector *is* the gate: a tail with an OOM always yields a line,
        // and a tail without one always yields `None`, which is what lets
        // callers use this directly instead of asking the same question twice.
        let tail = concat!(
            "  File \"/opt/vllm/worker.py\", line 212, in load_model\n",
            "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.\n",
            "INFO shutting down worker\n"
        );
        assert_eq!(
            vllm_oom_diagnose_symptom(tail).as_deref(),
            Some("vllm: torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB."),
            "the user's own failing line must be routed into `rocm diagnose --symptom`"
        );
        // Whatever comes back must be diagnosable, or the printed command would
        // report no known cause.
        let symptom = vllm_oom_diagnose_symptom(tail).expect("an OOM tail selects a line");
        assert!(diagnose::vllm_oom_symptom_is_diagnosable(&symptom));
        // The *last* OOM line wins: a tail ends with the failure that killed the
        // launch, and an earlier retry's line would misreport it.
        assert_eq!(
            vllm_oom_diagnose_symptom("RuntimeError: hipErrorOutOfMemory\nHIP OUT OF MEMORY\n")
                .as_deref(),
            Some("vllm: HIP OUT OF MEMORY")
        );
        // A progress bar repaints with a bare `\r`, so vLLM's own output puts
        // the repaint and the exception that follows it on ONE `\n` line. The
        // diagnostic re-splits on `\r` and keeps only anchored segments, so
        // anchoring the physical line once left the exception unanchored and
        // discarded: the tail read as "no OOM" and both the engine hint and the
        // serve-summary note vanished for the commonest shape of this failure.
        let repainted = "Loading weights: 90%\rtorch.OutOfMemoryError: HIP out of memory. \
                         Tried to allocate 7.21 GiB.\n";
        assert!(
            log_shows_oom(repainted),
            "an allocator exception after a progress-bar carriage return is still an OOM"
        );
        let repainted_symptom =
            vllm_oom_diagnose_symptom(repainted).expect("the CR-split tail still selects a line");
        assert_eq!(
            repainted_symptom,
            "vllm: torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.",
            "the exception segment must be quoted, not the progress-bar segment that \
             happened to carry the anchor"
        );
        // ...and what comes back is still diagnosable, so the printed command
        // reports a cause rather than reading as "checked, found nothing".
        assert!(diagnose::vllm_oom_symptom_is_diagnosable(
            &repainted_symptom
        ));
        // No OOM line, no symptom — including the sub-threshold shapes the
        // shared classifier rejects, which must not be reported as a cause.
        assert_eq!(
            vllm_oom_diagnose_symptom("OSError: model weights not found"),
            None
        );
        assert_eq!(
            vllm_oom_diagnose_symptom("Out of memory: Killed process 4242 (python)"),
            None
        );
        assert_eq!(vllm_oom_diagnose_symptom(""), None);
    }

    /// The `inspect_err` cleanup on the rename, which the happy-path test above
    /// cannot reach: it only ever asserts that a SUCCESSFUL save leaves nothing
    /// behind, so deleting the cleanup would go unnoticed until a user with a
    /// failing disk accumulated one `config.json.tmp-<ts>-<pid>` per attempt.
    ///
    /// A directory at the destination is what makes the rename fail on both
    /// supported platforms without needing permissions a root CI lane ignores —
    /// the same device `write_file_atomically_cleans_up_temp_when_the_rename_fails`
    /// uses in `apps/rocm`.
    #[test]
    fn a_config_save_that_cannot_publish_cleans_up_its_temp_file() -> Result<()> {
        let (root, paths) = temp_app_paths("config-save-rename-fails");
        fs::create_dir_all(&paths.config_dir)?;
        let occupied = paths.config_path();
        fs::create_dir_all(occupied.join("nested"))?;
        fs::write(occupied.join("nested").join("keep"), b"x")?;

        RocmCliConfig::default()
            .save(&paths)
            .expect_err("renaming onto a non-empty directory must fail");

        let leftovers: Vec<String> = fs::read_dir(&paths.config_dir)?
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp-"))
            .collect();
        let _ = fs::remove_dir_all(&root);

        assert!(
            leftovers.is_empty(),
            "a save that could not publish must not strand its temp file: {leftovers:?}"
        );
        Ok(())
    }

    #[test]
    fn saving_the_config_replaces_it_without_leaving_a_temp_file_behind() -> Result<()> {
        let (root, paths) = temp_app_paths("config-save-atomic");
        let mut config = RocmCliConfig {
            default_engine: Some("vllm".to_owned()),
            ..RocmCliConfig::default()
        };
        config.save(&paths)?;
        // Overwriting is the interesting case: the save goes through a temp file
        // and a rename rather than truncating the live config, so an interrupted
        // save can never leave a half-written file that `load` refuses to parse.
        config.default_engine = Some("lemonade".to_owned());
        config.save(&paths)?;

        let loaded = RocmCliConfig::load(&paths)?;
        let leftovers = fs::read_dir(&paths.config_dir)?
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp-"))
            .collect::<Vec<_>>();
        let _ = fs::remove_dir_all(&root);

        assert_eq!(loaded.default_engine.as_deref(), Some("lemonade"));
        assert!(
            leftovers.is_empty(),
            "a completed save must leave no temp file in the config folder: {leftovers:?}"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn saving_the_config_replaces_the_file_rather_than_rewriting_it_in_place() -> Result<()> {
        use std::os::unix::fs::MetadataExt;

        // The inode is the observable difference between a rewrite in place and
        // a replacement, and the reason the doc comment can promise what it
        // promises. A `fs::write` over the live path truncates and refills the
        // SAME file, so a reader holding it open — or one that opens it mid-save
        // — can see an empty or half-written config, which `load` hard-errors
        // on. A rename publishes a DIFFERENT file over the name, so a reader
        // sees either the old config or the new one and never the gap between.
        //
        // The sibling test above only checks that no temp file is left behind,
        // which holds just as well for an implementation with no temp file at
        // all, so it cannot tell the two apart.
        //
        // The `#[cfg(windows)]` function of the same name below asserts the same
        // guarantee through `GetFileInformationByHandle`.
        let (root, paths) = temp_app_paths("config-save-replaces");
        let mut config = RocmCliConfig {
            default_engine: Some("vllm".to_owned()),
            ..RocmCliConfig::default()
        };
        config.save(&paths)?;
        let before = fs::metadata(paths.config_path())?.ino();

        config.default_engine = Some("lemonade".to_owned());
        config.save(&paths)?;
        let after = fs::metadata(paths.config_path())?.ino();

        let loaded = RocmCliConfig::load(&paths)?;
        let _ = fs::remove_dir_all(&root);

        assert_ne!(
            before, after,
            "an overwrite that keeps the same inode rewrote the live config in \
             place, so an interrupted save can leave it truncated"
        );
        assert_eq!(loaded.default_engine.as_deref(), Some("lemonade"));
        Ok(())
    }

    /// The volume serial number paired with the 64-bit file index: this
    /// platform's file identity, the way an inode is on Unix.
    ///
    /// Read through `GetFileInformationByHandle` rather than
    /// `std::os::windows::fs::MetadataExt`, whose `volume_serial_number` and
    /// `file_index` accessors are unstable behind the `windows_by_handle`
    /// feature gate and so cannot be called on the pinned stable toolchain.
    #[cfg(windows)]
    #[allow(unsafe_code)] // Win32 FFI
    fn windows_file_identity(path: &Path) -> Result<(u32, u64)> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::HANDLE;
        use windows_sys::Win32::Storage::FileSystem::{
            BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
        };

        let file =
            fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        // All-POD `repr(C)` record, fully overwritten by the call below on
        // success and never read on failure.
        let mut info: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
        let handle = file.as_raw_handle() as HANDLE;
        if unsafe { GetFileInformationByHandle(handle, &raw mut info) } == 0 {
            bail!(
                "failed to read the file identity of {}: {}",
                path.display(),
                std::io::Error::last_os_error()
            );
        }
        Ok((
            info.dwVolumeSerialNumber,
            (u64::from(info.nFileIndexHigh) << 32) | u64::from(info.nFileIndexLow),
        ))
    }

    #[cfg(windows)]
    #[test]
    fn saving_the_config_replaces_the_file_rather_than_rewriting_it_in_place() -> Result<()> {
        // The Windows counterpart of the Unix test above. The file index
        // together with the volume serial number is this platform's file
        // identity, the way an inode is on Unix: a `fs::write` over the live
        // path refills the SAME file and keeps the pair, while a
        // publish-by-rename puts a DIFFERENT file behind the name. This keeps a
        // Windows-only regression to a plain `fs::write` from passing every
        // gate.
        //
        // The identity is not optional: `GetFileInformationByHandle` either
        // fills both fields or fails, and a failure is a test error rather than
        // a silently skipped assertion.
        let (root, paths) = temp_app_paths("config-save-replaces");
        let mut config = RocmCliConfig {
            default_engine: Some("vllm".to_owned()),
            ..RocmCliConfig::default()
        };
        config.save(&paths)?;
        let before = windows_file_identity(&paths.config_path())?;

        config.default_engine = Some("lemonade".to_owned());
        config.save(&paths)?;
        let after = windows_file_identity(&paths.config_path())?;

        let loaded = RocmCliConfig::load(&paths)?;
        let _ = fs::remove_dir_all(&root);

        assert_ne!(
            before, after,
            "an overwrite that keeps the same file identity rewrote the live \
             config in place, so an interrupted save can leave it truncated"
        );
        assert_eq!(loaded.default_engine.as_deref(), Some("lemonade"));
        Ok(())
    }

    #[test]
    fn engine_state_refresh_keeps_the_runtime_selector_the_launch_requested() -> Result<()> {
        let (root, paths) = temp_app_paths("record-runtime-pin");
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-runtime-pin",
            "vllm",
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-0.6B",
            "127.0.0.1",
            9,
            "managed",
            std::process::id(),
            Some("release-pip-gfx120x-all-7-13-0".to_owned()),
            None,
            Some("gpu_required".to_owned()),
        );
        fs::create_dir_all(
            record
                .engine_state_path
                .parent()
                .expect("engine state path has a parent"),
        )?;
        // What a vLLM server writes: `runtime_id` is the manifest's family id,
        // shared by every installed version of that family, while
        // `requested_runtime_id` is the selector the launch asked for, here an
        // exact runtime key.
        fs::write(
            &record.engine_state_path,
            serde_json::to_vec(&serde_json::json!({
                "status": "running",
                "runtime_id": "therock-release:gfx120X-all",
                "requested_runtime_id": "release-pip-gfx120x-all-7-13-0",
            }))?,
        )?;

        let refreshed = record.refresh_from_engine_state()?;
        let _ = fs::remove_dir_all(&root);

        assert!(refreshed);
        assert_eq!(
            record.runtime_id.as_deref(),
            Some("release-pip-gfx120x-all-7-13-0"),
            "adopting the resolved family id would throw away which installed \
             version the service is actually on — the record is what a restart \
             re-pins and what runtime activation compares against"
        );
        Ok(())
    }

    #[test]
    fn engine_state_refresh_falls_back_to_the_resolved_runtime_id() -> Result<()> {
        let (root, paths) = temp_app_paths("record-runtime-fallback");
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-runtime-fallback",
            "lemonade",
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-0.6B",
            "127.0.0.1",
            9,
            "managed",
            std::process::id(),
            None,
            None,
            Some("gpu_required".to_owned()),
        );
        fs::create_dir_all(
            record
                .engine_state_path
                .parent()
                .expect("engine state path has a parent"),
        )?;
        // An engine that records only the resolved runtime must still be read:
        // preferring `requested_runtime_id` must not mean ignoring `runtime_id`.
        fs::write(
            &record.engine_state_path,
            serde_json::to_vec(&serde_json::json!({
                "status": "running",
                "runtime_id": "lemonade-embeddable-8.1.11",
                "requested_runtime_id": serde_json::Value::Null,
            }))?,
        )?;

        record.refresh_from_engine_state()?;
        let _ = fs::remove_dir_all(&root);

        assert_eq!(
            record.runtime_id.as_deref(),
            Some("lemonade-embeddable-8.1.11")
        );
        Ok(())
    }

    #[test]
    fn engine_state_refresh_ignores_a_blank_requested_runtime_id() -> Result<()> {
        // A blank selector names no runtime, so adopting it would leave the
        // record with an empty runtime that matches nothing and that a restart
        // would re-pin. Both an empty string and whitespace must fall through
        // to the resolved `runtime_id`.
        for (blank, name) in [("", "empty"), ("   ", "whitespace")] {
            let (root, paths) = temp_app_paths(&format!("record-runtime-blank-{name}"));
            let mut record = ManagedServiceRecord::new(
                &paths,
                "svc-runtime-blank",
                "lemonade",
                "Qwen/Qwen3-0.6B",
                "Qwen/Qwen3-0.6B",
                "127.0.0.1",
                9,
                "managed",
                std::process::id(),
                None,
                None,
                Some("gpu_required".to_owned()),
            );
            fs::create_dir_all(
                record
                    .engine_state_path
                    .parent()
                    .expect("engine state path has a parent"),
            )?;
            fs::write(
                &record.engine_state_path,
                serde_json::to_vec(&serde_json::json!({
                    "status": "running",
                    "runtime_id": "lemonade-embeddable-8.1.11",
                    "requested_runtime_id": blank,
                }))?,
            )?;

            let refreshed = record.refresh_from_engine_state()?;
            let _ = fs::remove_dir_all(&root);

            assert!(refreshed, "{name}: the engine state must still be adopted");
            assert_eq!(
                record.runtime_id.as_deref(),
                Some("lemonade-embeddable-8.1.11"),
                "{name}: a blank requested_runtime_id must fall through to runtime_id"
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
}
