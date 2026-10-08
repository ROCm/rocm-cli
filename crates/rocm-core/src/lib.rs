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
#[cfg(windows)]
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub mod browser;
pub mod diagnose;
pub mod disk_space;
pub mod examine;
pub mod fix;
pub mod host_gpu;
pub mod managed_runtime;
pub mod model_readiness;
pub mod net;
pub mod openmpi;
pub mod proc_lifecycle;
pub mod process;
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
    use crate::test_support::temp_app_paths;

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
