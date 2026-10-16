// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

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
pub use service_record::ManagedServiceRecord;
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
