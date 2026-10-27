// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
// `Command` is this module's own — [`amd_smi_json_with`] builds one — but
// nothing here runs it: the wait, the kill and both pipe drains belong to
// [`crate::process::run_with_timeout`], which is why no production `Stdio`
// import comes back with it (the helper sets the child's stdio itself, and its
// docs say callers should not). `Stdio` and `thread` are left to the KFD open
// probe and its Linux-only test, hence the `cfg`s: an unused import is a hard
// error on the Windows lane's `-D warnings` build.
use std::process::Command;
#[cfg(all(test, target_os = "linux"))]
use std::process::Stdio;
#[cfg(target_os = "linux")]
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
    command_failure_detail_or_else(output, || "no output".to_owned())
}

/// The stderr-then-stdout precedence behind [`command_failure_detail`], with the
/// both-blank fallback left to the caller.
///
/// Exists so the precedence rule lives in exactly one place. [`amd_smi_json`]
/// needs the same ordering but a different last resort — `exit status <code>`
/// rather than `no output`, because a silent amd-smi's exit code is the only
/// diagnostic left — and inlining the two `is_empty` checks there to get it put
/// the same rule in two places. `fallback` is a closure because formatting an
/// `ExitStatus` is wasted work on the overwhelmingly common non-blank path.
pub fn command_failure_detail_or_else(
    output: &std::process::Output,
    fallback: impl FnOnce() -> String,
) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !stderr.is_empty() {
        return stderr;
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if !stdout.is_empty() {
        return stdout;
    }
    fallback()
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

/// Device node of the AMD kernel fusion driver. Bare-metal Linux only: no other
/// platform exposes one, and WSL — which *is* `target_os = "linux"` — is not an
/// exception to hand-wave past. It reaches the GPU through `/dev/dxg` and the
/// Windows host driver and has no `/dev/kfd` at all, which is why the WSL
/// diagnosis catalog skips every bare-metal driver check (see `docs/wsl.md` and
/// [`crate::examine::WslFacts`]).
const KFD_DEVICE: &str = "/dev/kfd";

/// Name of the E2E-only seam that moves the pre-flight's device node, so the
/// gate's two branches are both reachable from a Gherkin scenario.
///
/// Honoured only by a build carrying the `e2e-test-hooks` feature, which no
/// released artefact does; see [`preflight_kfd_device`].
#[cfg(feature = "e2e-test-hooks")]
pub(crate) const TEST_KFD_DEVICE_ENV: &str = "ROCM_E2E_KFD_DEVICE";

/// The device node [`amd_smi_json`] runs its pre-flight against.
///
/// [`KFD_DEVICE`] in every shipped build. An `e2e-test-hooks` build lets
/// [`TEST_KFD_DEVICE_ENV`] name a different path, which is what makes the gate
/// testable from the outside at all.
///
/// Why a seam rather than the existing [`crate::hardware_root`] re-rooting:
/// [`kfd_readable`] deliberately keeps the *real* node's veto, so a simulated
/// root can only ever make the gate decline, never allow. That asymmetry is the
/// invariant that stops a simulated root launching `amd-smi` on a machine with
/// no GPU, and it is correct — but it also means no simulated root can exercise
/// the *allowed* branch on a GPU-less runner, which is where the per-PR lane
/// runs. A path override sidesteps that without touching the invariant: the
/// production build has no override, so the veto and the real device are all
/// there is.
///
/// Safe to be an override rather than a bypass. It relocates the probe; it does
/// not skip it. A build with the feature on and the variable unset — every unit
/// test in this crate, and every E2E scenario that does not set it — gates on
/// `/dev/kfd` exactly as production does.
#[cfg(feature = "e2e-test-hooks")]
fn preflight_kfd_device() -> std::path::PathBuf {
    std::env::var_os(TEST_KFD_DEVICE_ENV)
        .map_or_else(|| std::path::PathBuf::from(KFD_DEVICE), Into::into)
}

/// Production counterpart of [`preflight_kfd_device`]: the constant, with no
/// environment consulted. See that function for why the seam exists.
#[cfg(not(feature = "e2e-test-hooks"))]
fn preflight_kfd_device() -> std::path::PathBuf {
    std::path::PathBuf::from(KFD_DEVICE)
}

/// Whether `/dev/kfd` can be opened for reading — the pre-flight every
/// `amd-smi` launch goes through, and it is **mandatory** before every launch.
///
/// What it actually detects is **inaccessible**, not **unhealthy**. `kfd_open`
/// refuses when the node is absent, when permission is denied, when the driver
/// exposes no KFD GPU node, and while a reset is in progress; a driver that is
/// merely version-mismatched or wedged after a fault still opens. Those are the
/// states in which `amd-smi` blocks in uninterruptible kernel sleep — no signal
/// escapes, so the process survives both Ctrl-C and `SIGKILL`, and no timeout
/// rescues it, because killing a D-state child is a no-op. This gate removes
/// the reachable half of that hazard; it does not claim to see the rest.
///
/// The probe does not assume `kfd_open` returns promptly. An `open` that blocks
/// — whatever the reason, including a state this gate has not anticipated —
/// is bounded by [`KFD_OPEN_TIMEOUT`] and read as "not readable", so the gate
/// declines rather than inheriting the stall it exists to prevent. See
/// [`device_readable_within`].
///
/// Off Linux there is no such device and no such kernel state, so the check is
/// `true` without probing anything. That wave-through is load-bearing rather
/// than pedantic: `amd-smi.exe` is a supported Windows binary (see
/// [`managed_sdk_tool_path`]), so a check that simply failed to find `/dev/kfd`
/// there would silently disable GPU detection on every Windows host.
///
/// WSL needs a wave-through for the same reason and cannot get it from `cfg`:
/// it compiles as Linux yet has no `/dev/kfd` (see [`KFD_DEVICE`]), so the real
/// open fails on every WSL host. But WSL identity alone is not the condition.
/// `b4f98bc2` replaced exactly such a bare identity check in the dashboard
/// pre-flight with [`has_usable_amd_gpu`], because `/dev/dxg` is vendor-agnostic
/// and a WSL host whose ROCDXG plumbing is not ready is one where `amd-smi`
/// still hangs — a different hazard from the KFD one, which an "is this WSL"
/// test cannot tell apart from a healthy host. So the substitution is
/// `is_wsl_host() && has_usable_amd_gpu()`, the same conjunction
/// `apps/rocm/src/dash.rs::gpu_reachable_for_preflight` feeds the dashboard, and
/// `serve`/`rocmd`/the dashboard keep the single cross-surface verdict that
/// commit established. On bare metal `is_wsl_host()` short-circuits, so
/// [`has_usable_amd_gpu`] is never consulted and behaviour is a plain
/// `/dev/kfd` open.
///
/// Hardware-probe routing (see [`crate::hardware_root`]): like the dashboard
/// pre-flight, this reads `host_path("/dev/kfd")` *and* keeps the real node's
/// veto. A simulated E2E root can therefore make the gate decline on a machine
/// that has a GPU, but can never make it allow a launch on a machine that does
/// not.
#[cfg(target_os = "linux")]
fn kfd_readable(device: &Path) -> bool {
    kfd_readable_with(
        // `&&` short-circuits: `has_usable_amd_gpu` runs on WSL only.
        is_wsl_host() && has_usable_amd_gpu(),
        device,
        &host_path(device),
    )
}

/// [`kfd_readable`] with its two non-syscall inputs injected, so the
/// wave-through and the simulated-root routing can both be pinned from a test
/// on an ordinary Linux host rather than only on WSL or under an E2E root.
///
/// `wsl_gpu_reachable` may only ever be the WSL conjunction described on
/// [`kfd_readable`]; it substitutes for the device open entirely, so widening it
/// to any fail-open verdict would disable the gate on bare metal.
#[cfg(target_os = "linux")]
fn kfd_readable_with(wsl_gpu_reachable: bool, real_device: &Path, host_device: &Path) -> bool {
    wsl_gpu_reachable
        || (device_readable(real_device)
            && (host_device == real_device || device_readable(host_device)))
}

/// How long the pre-flight waits for the device `open` itself before giving up
/// on it and declining the launch.
///
/// The gate exists to avoid spawning `amd-smi` against a device that can park a
/// process in uninterruptible sleep. Performing the probe with a plain blocking
/// `open` would make the gate's own safety rest on an *assumption* — that
/// `kfd_open` always returns promptly, and that only amd-smi's later ioctls can
/// enter D-state — and that assumption cannot be verified against a
/// reset-in-progress device from any test. If it were ever wrong, the gate
/// would have relocated the hazard into a synchronous call with *less*
/// protection than the spawn it replaced, since there is no child to kill.
///
/// Bounding the open removes the need to be right about that: a `/dev/kfd` open
/// that has not returned in this long is treated as "not readable", which is the
/// declining answer, so the unverifiable case fails in the safe direction. Short
/// because a healthy `kfd_open` is a handful of microseconds and the only thing
/// a longer wait buys is a longer stall on a host that is already unhealthy.
#[cfg(target_os = "linux")]
const KFD_OPEN_TIMEOUT: Duration = Duration::from_secs(2);

/// Whether `device` can be opened for reading, giving up after
/// [`KFD_OPEN_TIMEOUT`].
#[cfg(target_os = "linux")]
fn device_readable(device: &Path) -> bool {
    device_readable_within(device, KFD_OPEN_TIMEOUT)
}

/// [`device_readable`] with the bound injected, so a test can pin the timeout
/// path against a device that really does block in `open` (a FIFO with no
/// writer) without stalling the suite for the production bound.
///
/// The `open` runs on its own thread and the caller waits on a channel, because
/// there is no portable way to interrupt a blocking `open` once it has entered
/// the kernel. A probe that overruns leaves that thread parked — the same
/// trade-off [`crate::process::run_with_timeout`] makes when it abandons rather
/// than joins its readers, and for the same reason: one
/// leaked thread is strictly cheaper than a blocked caller, and a device that
/// can park a thread forever is exactly the device this whole gate exists to
/// refuse. The opened handle is dropped on the probe thread, so a late success
/// closes its own descriptor rather than leaking one.
///
/// If no thread can be spawned the probe falls back to a direct blocking open:
/// that is the pre-existing behaviour, and refusing every launch because the
/// process is out of threads would be a worse failure than the one being
/// guarded against.
#[cfg(target_os = "linux")]
fn device_readable_within(device: &Path, timeout: Duration) -> bool {
    let (sender, receiver) = std::sync::mpsc::channel();
    let probe_path = device.to_path_buf();
    let spawned = thread::Builder::new()
        .name("rocm-kfd-open-probe".to_owned())
        .spawn(move || {
            let readable = fs::OpenOptions::new().read(true).open(&probe_path).is_ok();
            // The receiver is gone whenever the probe overran; dropping the
            // send result is the whole handling.
            let _ = sender.send(readable);
        })
        .is_ok();
    if !spawned {
        return fs::OpenOptions::new().read(true).open(device).is_ok();
    }
    // Both error arms — timeout and a probe thread that died without sending —
    // mean "no readable answer", which the gate must read as not readable.
    receiver.recv_timeout(timeout).unwrap_or(false)
}

/// Off-Linux counterpart of [`kfd_readable`]: see that function's docs for why
/// the wave-through is load-bearing rather than a stub.
///
/// `const` because it has to be. The Windows lane builds with `-D warnings` and
/// runs clippy, so `clippy::missing_const_for_fn` makes this a hard compile
/// error there while the Linux lane — where the real body opens a device and
/// cannot be `const` — never sees the lint. A `cfg`-selected pair of functions
/// is linted per branch, not per name.
#[cfg(not(target_os = "linux"))]
const fn kfd_readable(_device: &Path) -> bool {
    true
}

/// How much of a failed `amd-smi`'s own output the error message quotes.
///
/// The bound is not cosmetic. `rocmd` renders this whole error chain into
/// `CodexBridgeGpuSnapshot.note` and the GPU-metrics watcher copies that note
/// into the automation-event and audit JSONL files on every poll — files whose
/// append helpers rewrite the whole file each time. Unbounded, one `amd-smi`
/// that prints a payload and then fails grows both logs by that much per tick,
/// and every subsequent append costs more than the last. A 151K-character JSONL
/// line was reproduced from exactly that poller.
///
/// Deliberately the same name and value as the `MAX_QUOTED_OUTPUT_CHARS` in
/// `fix/rocmai-448-drain-child-pipes` (PR #526), which bounds the same quote for
/// the same reason one layer down, in its `run_with_timeout`. The two branches
/// are building overlapping subprocess-timeout helpers; keeping the bound
/// identical means converging on one of them is a deletion rather than a
/// behaviour change. Sized to still hold a real stderr backtrace: the point is
/// to stop a payload-sized quote, not to shorten diagnostics.
const MAX_QUOTED_OUTPUT_CHARS: usize = 2000;

/// Run `amd-smi <args>` and parse its stdout as JSON, declining to launch at all
/// unless the `/dev/kfd` pre-flight in [`kfd_readable`] passes.
///
/// This is the supported way to launch `amd-smi` from a synchronous caller. The
/// pre-flight has to happen at every launch site or it protects nothing, and
/// routing the launch through one function is what keeps that true as sites are
/// added.
///
/// `timeout` is defence in depth, not the mitigation: see
/// [`crate::process::run_with_timeout`], which kills the timed-out child and
/// waits on it — neither of which dislodges a process in uninterruptible sleep,
/// and the `wait` of an undislodgeable one does not return either. It bounds
/// the ordinary failures — a slow or wedged-but-signalable `amd-smi` — and the
/// pre-flight above, not the timeout, is what keeps the D-state case away.
pub fn amd_smi_json(args: &[&str], timeout: Duration) -> Result<serde_json::Value> {
    amd_smi_json_with(
        &resolve_amd_smi_binary(),
        // [`preflight_kfd_device`], not the constant: identical in every shipped
        // build, and the one seam an E2E scenario can move so that both sides of
        // the gate are reachable from the lane that runs per PR.
        &preflight_kfd_device(),
        args,
        timeout,
    )
}

fn amd_smi_json_with(
    binary: &OsStr,
    kfd_device: &Path,
    args: &[&str],
    timeout: Duration,
) -> Result<serde_json::Value> {
    if !kfd_readable(kfd_device) {
        bail!(
            "declined to launch `amd-smi {}`: {} could not be opened for reading (absent, \
             permission denied, no KFD GPU node, or a reset in progress), and amd-smi can \
             block unkillably in uninterruptible sleep against a device in that state",
            args.join(" "),
            kfd_device.display()
        );
    }

    let mut command = Command::new(binary);
    // No stdio here: `run_with_timeout` sets null stdin and piped stdout/stderr
    // itself, since capturing both is its contract, and its docs say callers
    // should not set them.
    command.args(args);
    let output = crate::process::run_with_timeout(command, timeout, "amd-smi")
        .with_context(|| format!("failed to launch amd-smi {}", args.join(" ")))?;

    if !output.status.success() {
        // Truncated, not dropped. This message reaches `rocmd`'s snapshot note
        // and from there the JSONL logs, so the quote has to be bounded at the
        // point that knows these characters are raw child output — see
        // [`MAX_QUOTED_OUTPUT_CHARS`]. The head is where a failing `amd-smi`
        // says what went wrong, and the command is named before it either way.
        //
        // The stderr-then-stdout precedence is `command_failure_detail`'s, taken
        // from the shared helper rather than re-spelled here; only the
        // both-blank fallback differs, naming the exit status instead of the
        // generic "no output".
        bail!(
            "amd-smi {} failed: {}",
            args.join(" "),
            crate::examine::truncate_to_chars(
                command_failure_detail_or_else(&output, || format!(
                    "exit status {}",
                    output.status
                )),
                MAX_QUOTED_OUTPUT_CHARS,
            )
        );
    }

    serde_json::from_slice(&output.stdout)
        .with_context(|| format!("failed to parse amd-smi {} json", args.join(" ")))
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use std::path::Path;

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

    /// Serialise a test that reads a device node through [`kfd_readable`].
    ///
    /// The gate resolves its device through `host_path`, so in an
    /// `e2e-test-hooks` build another test setting `ROCM_CLI_TEST_HOST_ROOT`
    /// process-wide would re-root this test's own tempdir device out from under
    /// it, and the gate would decline a device the test had just created. This
    /// is the same lock `fix.rs` holds for the same reason; without the feature
    /// it is uncontended and costs nothing.
    #[cfg(target_os = "linux")]
    fn host_root_guard() -> std::sync::MutexGuard<'static, ()> {
        crate::hardware_root::HOST_ROOT_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The gate has to stop the *spawn*, not merely report a failure after the
    /// fact: the launch that already happened is the launch that hangs. The
    /// fake `amd-smi` leaves a marker file, so its absence is direct evidence
    /// nothing ran — and running the same fixture against a readable device
    /// proves the fake would otherwise have executed, which is what stops the
    /// blocked case from passing vacuously.
    ///
    /// Linux-only because `kfd_readable` is deliberately a no-op elsewhere;
    /// off Linux there is no gate to exercise.
    #[test]
    #[cfg(target_os = "linux")]
    fn the_kfd_gate_stops_amd_smi_before_it_is_spawned() -> Result<()> {
        // WSL compiles as Linux but is waved through deliberately (see
        // `the_kfd_preflight_waves_through_wsl_hosts`), so there is no gate to
        // exercise on such a host and the blocked case could never hold.
        if is_wsl_host() {
            return Ok(());
        }
        let _host_root = host_root_guard();

        let temp_root =
            std::env::temp_dir().join(format!("rocm-cli-kfd-gate-{}", unix_time_millis()));
        fs::create_dir_all(&temp_root)?;
        let marker = temp_root.join("amd-smi-ran");
        let fake_amd_smi = temp_root.join("amd-smi");
        // Not `fs::write` + `set_permissions`: this process is about to execute
        // the file, and a write fd leaked into a parallel test's fork would make
        // the first run fail with `ETXTBSY`. See `fake_executable`.
        crate::fake_executable::write_fake_executable(
            &fake_amd_smi,
            format!("#!/bin/sh\ntouch '{}'\necho '{{}}'\n", marker.display()),
        )?;
        let readable_device = temp_root.join("stand-in-kfd");
        fs::write(&readable_device, b"")?;

        let timeout = Duration::from_secs(10);
        let args = ["list", "--json"];

        let blocked = amd_smi_json_with(
            fake_amd_smi.as_os_str(),
            &temp_root.join("absent-kfd"),
            &args,
            timeout,
        );
        let ran_while_blocked = marker.exists();

        let allowed = amd_smi_json_with(fake_amd_smi.as_os_str(), &readable_device, &args, timeout);
        let ran_while_allowed = marker.exists();

        let _ = fs::remove_dir_all(&temp_root);

        assert!(
            blocked.is_err(),
            "an unreadable KFD device must stop the launch"
        );
        assert!(
            !ran_while_blocked,
            "amd-smi was spawned even though the pre-flight failed"
        );
        assert!(
            allowed.is_ok(),
            "a readable device must let the launch through: {allowed:?}"
        );
        assert!(
            ran_while_allowed,
            "the fake amd-smi never ran at all, so the blocked case proves nothing"
        );
        Ok(())
    }

    /// No host off Linux has a `/dev/kfd`, so the pre-flight must wave the
    /// launch through instead of reading the absent node as "unsafe".
    /// `amd-smi.exe` is a supported Windows binary, so treating its absence as
    /// a failure would disable GPU detection on every Windows host.
    #[test]
    #[cfg(not(target_os = "linux"))]
    fn the_kfd_preflight_waves_through_hosts_that_have_no_kfd() {
        assert!(kfd_readable(Path::new("/definitely/not/a/device/node")));
    }

    /// WSL has no `/dev/kfd` either, but unlike Windows it cannot be waved
    /// through by `cfg`: it compiles as Linux, so the real open runs, always
    /// fails, and declines every routed `amd-smi` launch on a supported
    /// platform — silently, since both `serve` probes drop the error with
    /// `.ok()?`. Same wave-through as the off-Linux case above, decided at
    /// runtime instead.
    ///
    /// The *conjunction* is what is pinned, not WSL identity: a WSL host whose
    /// ROCDXG plumbing is not ready is one where `amd-smi` still hangs, which is
    /// why `b4f98bc2` replaced a bare identity check with `has_usable_amd_gpu()`
    /// in the dashboard pre-flight. The unready-WSL case below is the one that
    /// distinguishes this from that mistake; the bare-metal case is the control
    /// that stops the whole test passing with the gate deleted.
    #[test]
    #[cfg(target_os = "linux")]
    fn the_kfd_preflight_waves_through_wsl_hosts_only_when_the_gpu_is_reachable() {
        let absent = Path::new("/definitely/not/a/device/node");
        assert!(
            kfd_readable_with(true, absent, absent),
            "a ready WSL host has no /dev/kfd, so its absence must not decline the launch"
        );
        assert!(
            !kfd_readable_with(false, absent, absent),
            "an unready WSL host — or a bare-metal one with no readable KFD — must be declined"
        );
    }

    /// The simulated-root routing, which `kfd_readable` applies to the real
    /// `/dev/kfd` on the production path.
    ///
    /// Both halves are load-bearing and they fail in opposite directions. A
    /// simulated root that hides the device must decline (so an E2E lane can
    /// manufacture the declined branch on a machine that does have a GPU), and
    /// a simulated root that *invents* a device must not allow (so no test hook
    /// can ever start a real `amd-smi` on a machine that has no GPU — the veto
    /// `hardware_root`'s module contract requires of this probe).
    #[test]
    #[cfg(target_os = "linux")]
    fn the_kfd_gate_honours_a_simulated_root_but_keeps_the_real_device_veto() -> Result<()> {
        let _host_root = host_root_guard();
        let temp_root =
            std::env::temp_dir().join(format!("rocm-cli-kfd-root-{}", unix_time_millis()));
        fs::create_dir_all(&temp_root)?;
        let present = temp_root.join("present-kfd");
        fs::write(&present, b"")?;
        let absent = temp_root.join("absent-kfd");

        let real_present_root_absent = kfd_readable_with(false, &present, &absent);
        let real_absent_root_present = kfd_readable_with(false, &absent, &present);
        let both_present = kfd_readable_with(false, &present, &present);
        let _ = fs::remove_dir_all(&temp_root);

        assert!(
            !real_present_root_absent,
            "a simulated root that hides the device must decline the launch"
        );
        assert!(
            !real_absent_root_present,
            "a simulated root must never be able to start amd-smi on a host with no real device"
        );
        assert!(
            both_present,
            "the ordinary case must still pass, or the two above prove nothing"
        );
        Ok(())
    }

    /// The gate's own probe must not be able to hang.
    ///
    /// `kfd_readable` was reviewed as resting on an unverified assumption: that
    /// `open("/dev/kfd")` always returns promptly, so that only amd-smi's later
    /// ioctls can enter uninterruptible sleep. If that were ever wrong the gate
    /// would have moved the hazard into a synchronous call with *less*
    /// protection than the spawn it replaced — no child to kill, no timeout.
    /// Rather than assert the assumption, the probe is bounded; this pins that.
    ///
    /// A FIFO is the fixture that makes the blocking case real rather than
    /// simulated: `open(O_RDONLY)` on a FIFO with no writer blocks in the kernel
    /// indefinitely, which is the observable shape of the hazard. The deadline
    /// assertion is what distinguishes a bounded probe from an unbounded one —
    /// without it, deleting the timeout would hang the suite rather than fail
    /// this test, and a hang is not a readable failure. The readable control in
    /// the same test stops the whole thing passing vacuously, e.g. if
    /// `device_readable_within` were mutated to return `false` unconditionally.
    ///
    /// The FIFO is created by shelling out to `mkfifo(1)` rather than calling
    /// `libc::mkfifo`: the workspace lints deny `unsafe_code`, and one test
    /// fixture does not justify an exception. The tool is coreutils, so it is
    /// present wherever this test runs; a host without it skips rather than
    /// fails, since a missing fixture is not evidence about the probe.
    #[test]
    #[cfg(target_os = "linux")]
    fn the_kfd_probe_declines_a_device_whose_open_blocks_instead_of_hanging() -> Result<()> {
        let temp_root =
            std::env::temp_dir().join(format!("rocm-cli-kfd-open-{}", unix_time_millis()));
        fs::create_dir_all(&temp_root)?;
        let fifo = temp_root.join("blocking-kfd");
        let made = Command::new("mkfifo")
            .arg(&fifo)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        if !matches!(&made, Ok(status) if status.success()) {
            let _ = fs::remove_dir_all(&temp_root);
            return Ok(());
        }

        let probe_timeout = Duration::from_millis(300);
        // The probe runs on a thread the test watches with its own, much looser
        // deadline, rather than being called inline. A mutant that drops the
        // bound from `device_readable_within` would block forever on this FIFO;
        // called inline that would hang the whole test binary, and a hang is a
        // CI timeout rather than a readable failure. Watched, it fails here with
        // a message naming the cause.
        let (verdict_tx, verdict_rx) = std::sync::mpsc::channel();
        thread::Builder::new()
            .name("kfd-open-probe-test".to_owned())
            .spawn(move || {
                let _ = verdict_tx.send(device_readable_within(&fifo, probe_timeout));
            })
            .context("could not start the watched probe thread")?;
        let watchdog = probe_timeout * 20;
        let blocked = verdict_rx.recv_timeout(watchdog);

        let readable = temp_root.join("ordinary-kfd");
        fs::write(&readable, b"")?;
        let allowed = device_readable_within(&readable, probe_timeout);
        // The overrun probe's thread is still parked in `open` and stays that
        // way until this test binary exits — deliberately, and the same leak
        // `device_readable_within` documents. Unlinking the FIFO does not
        // release it, and nothing here needs it released.
        let _ = fs::remove_dir_all(&temp_root);

        assert_eq!(
            blocked,
            Ok(false),
            "a device whose open never returns must read as not readable within its \
             {probe_timeout:?} bound, so the gate declines; `Err` here means the probe had still \
             not returned after {watchdog:?} — an unbounded open relocates the D-state hang into \
             the gate itself"
        );
        assert!(
            allowed,
            "an ordinary readable device must still pass, or the blocked case proves nothing"
        );
        Ok(())
    }

    /// Pins the production wiring the injected helpers cannot: that
    /// [`amd_smi_json`] gates on [`KFD_DEVICE`] itself.
    ///
    /// Every other gate test reaches `amd_smi_json_with` with a device of its
    /// own choosing, so swapping the constant out of the public wrapper — which
    /// disables the gate for every production caller — leaves them all green.
    /// This compares the wrapper's observable verdict against the gate read
    /// directly, so the two can only agree while the wrapper passes the real
    /// device through.
    ///
    /// Non-vacuous on *every* host in an `e2e-test-hooks` build: the empty
    /// simulated root below has no `dev/kfd`, so the gate declines whatever the
    /// machine has, while a wrapper passing any other path — `/` being the
    /// obvious mutant — re-roots to the root directory itself, which opens.
    /// Without the feature it is non-vacuous on any host with no readable
    /// `/dev/kfd` (every non-GPU CI runner) and asserts only the absence of a
    /// spurious decline on a GPU host.
    #[test]
    #[cfg(target_os = "linux")]
    fn the_public_amd_smi_entry_point_gates_on_the_real_kfd_device() {
        // Acquired by name rather than through `host_root_guard()`: this is the
        // one test here that also calls `RestoredEnvVar::set`, and the
        // env-mutation contract guard matches TEXT -- it recognises a
        // serializer by the `_TEST_LOCK` suffix, which the helper's name hides.
        // Same lock, spelled so the guard can see it.
        let _host_root = crate::hardware_root::HOST_ROOT_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        #[cfg(feature = "e2e-test-hooks")]
        let _simulated_host = {
            let root = tempfile::tempdir().expect("a simulated host root");
            let variable = crate::test_env::RestoredEnvVar::set(
                crate::hardware_root::TEST_HOST_ROOT_ENV,
                root.path(),
            );
            (variable, root)
        };

        let gate_allows = kfd_readable(Path::new(KFD_DEVICE));

        // A deliberately bogus subcommand: if the gate allows the launch this
        // still fails, but with a *launch* error rather than a decline, and it
        // is the decline that is under test. The timeout is short because the
        // pre-flight has already passed by the time anything is spawned.
        let error = amd_smi_json(&["--rocm-cli-gate-probe"], Duration::from_secs(5)).err();
        let declined = error.is_some_and(|error| {
            let message = format!("{error:#}");
            message.contains("declined to launch") && message.contains(KFD_DEVICE)
        });

        assert_eq!(
            declined, !gate_allows,
            "amd_smi_json must decline exactly when the gate on {KFD_DEVICE} does; a wrapper \
             that passes some other path declines on neither host"
        );
    }

    /// The E2E path seam is an override, not a bypass: with the variable unset
    /// the pre-flight device is `/dev/kfd` on every build, and with it set the
    /// gate still *runs* — against the named path — rather than being skipped.
    ///
    /// Worth pinning because the seam's whole justification is that it cannot
    /// weaken a shipped build. Both halves are load-bearing: a `preflight_kfd_device`
    /// that read the variable unconditionally would fail the unset half here and
    /// hand any process on the host a way to disable the gate, and one that
    /// returned a readable path outright — the bypass this deliberately is not —
    /// would fail the declining half.
    #[test]
    #[cfg(target_os = "linux")]
    fn the_e2e_kfd_path_seam_moves_the_gate_without_disabling_it() {
        // The same serializer the simulated-root tests take, rather than a
        // second one: both mutate the environment the pre-flight reads, and one
        // lock for the pair cannot deadlock against itself.
        let _env = crate::hardware_root::HOST_ROOT_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        assert_eq!(
            preflight_kfd_device(),
            Path::new(KFD_DEVICE),
            "with no seam set, every build must gate on the real device"
        );

        #[cfg(feature = "e2e-test-hooks")]
        {
            let scratch = tempfile::tempdir().expect("a scratch dir");
            let absent = scratch.path().join("no-such-kfd");
            let _seam = crate::test_env::RestoredEnvVar::set(TEST_KFD_DEVICE_ENV, &absent);

            assert_eq!(
                preflight_kfd_device(),
                absent,
                "an e2e-test-hooks build must honour the seam"
            );

            let error = amd_smi_json(&["--rocm-cli-gate-probe"], Duration::from_secs(5))
                .expect_err("an unopenable seam path must decline the launch");
            let message = format!("{error:#}");
            assert!(
                message.contains("declined to launch")
                    && message.contains(&absent.display().to_string()),
                "the seam must move the gate, not remove it; got: {message}"
            );
        }
    }

    /// A failing `amd-smi` that prints a payload must not paste the payload into
    /// the error. `rocmd` puts this message in its snapshot note and the
    /// GPU-metrics watcher copies that note into rewrite-the-whole-file JSONL
    /// logs on every poll, which is how a 151K-character log line was produced.
    #[test]
    #[cfg(target_os = "linux")]
    fn a_failing_amd_smi_cannot_paste_its_whole_output_into_the_error() -> Result<()> {
        const PAYLOAD_CHARS: usize = 200_000;

        if is_wsl_host() {
            return Ok(());
        }

        let _host_root = host_root_guard();

        let temp_root =
            std::env::temp_dir().join(format!("rocm-cli-amd-smi-quote-{}", unix_time_millis()));
        fs::create_dir_all(&temp_root)?;
        let fake_amd_smi = temp_root.join("amd-smi");
        // See the sibling gate test: planting an executable this process then
        // runs has to go through `fake_executable` or it races `ETXTBSY`.
        crate::fake_executable::write_fake_executable(
            &fake_amd_smi,
            format!("#!/bin/sh\nyes x | head -c {PAYLOAD_CHARS} | tr -d '\\n' >&2\nexit 1\n"),
        )?;
        let readable_device = temp_root.join("stand-in-kfd");
        fs::write(&readable_device, b"")?;

        let error = amd_smi_json_with(
            fake_amd_smi.as_os_str(),
            &readable_device,
            &["list", "--json"],
            Duration::from_secs(30),
        )
        .expect_err("a child that exits 1 must fail the call");
        let message = format!("{error:#}");
        let _ = fs::remove_dir_all(&temp_root);

        assert!(
            message.chars().count() < PAYLOAD_CHARS / 10,
            "the failure message quoted {} characters of child output unbounded",
            message.chars().count()
        );
        assert!(
            message.contains("amd-smi list --json failed"),
            "the bound must not cost the message its subject: {message}"
        );
        assert!(
            message.contains("[truncated]"),
            "an over-long quote must say it was cut, not look like the whole story: {message}"
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
}
