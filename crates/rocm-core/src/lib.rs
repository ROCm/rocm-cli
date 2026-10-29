// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(all(test, unix))]
mod archive_props;
pub mod automation;
pub mod binary_paths;
pub mod browser;
pub mod codex_bridge;
pub mod command_capture;
pub mod config;
pub mod confirm;
pub mod crypto;
pub mod diagnose;
pub mod disk_space;
pub mod endpoint_key;
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
pub mod placeholder;
pub mod proc_lifecycle;
pub mod process;
pub mod quoting;
pub mod report;
pub mod report_delivery;
pub mod report_rollup;
pub mod rocm_install;
pub mod runtime;
pub mod service_identity;
pub mod service_record;
pub mod terminal;
pub mod user_path;

#[cfg(test)]
mod test_env;
#[cfg(test)]
mod test_support;
pub mod uv;
pub mod windows_driver;
pub use automation::{
    AuditEventRecord, AutomationEventRecord, AutomationProposalRecord, AutomationRuntimeState,
    AutomationTriggerEvent, BuiltinWatcherSpec, EVENT_LOG_ROTATE_BYTES,
    EVENT_LOG_ROTATED_GENERATIONS, REVIEWED_PROPOSALS_RETAINED, WatcherRuntimeSnapshot,
    append_audit_event, append_automation_event, append_automation_proposal, builtin_watcher,
    builtin_watchers, find_automation_proposal, generate_proposal_id, load_automation_proposals,
    load_recent_audit_events, load_recent_automation_events, load_recent_automation_proposals,
    update_automation_proposal_status,
};
pub use binary_paths::{daemon_binary_path, engine_binary_path, sibling_binary_path};
pub use codex_bridge::{CodexBridgeEngine, CodexBridgeGpuSnapshot, CodexBridgeSnapshot};
pub use command_capture::{CommandCapture, command_capture_text, run_rocm_capture_with_env};
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
pub use endpoint_key::{endpoint_api_key_has_forbidden_chars, generate_endpoint_api_key};
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
    MODEL_RECIPE_STALE_AFTER_MS, ModelArtifactCacheStatus, ModelCatalogPlatform,
    ModelRecipeArtifactRecord, ModelRecipeArtifactSourcePolicyRecord, ModelRecipeEndpointRecord,
    ModelRecipeEngineRecord, ModelRecipeIndexDocument, ModelRecipeRecord, ModelRecipeRegistry,
    ModelRecipeRegistrySource, ModelRecipeUnsupportedCombinationRecord,
    builtin_model_recipe_registry, builtin_model_recipes, load_model_recipe_index,
    load_model_recipe_registry, load_signed_model_recipe_index, model_artifact_cache_marker_path,
    model_artifact_cache_status, model_catalog_platforms, model_recipe_featured,
    model_recipe_index_signature_path, model_recipe_staleness_warning,
    model_recipe_target_platform_label, normalize_model_ref, platform_matches_gfx_family,
    resolve_builtin_model_recipe, resolve_model_recipe, resolve_model_recipe_artifact,
    verify_model_recipe_index_signature,
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
pub use quoting::{posix_shell_quote, powershell_quote};
pub use report::{
    APPROVED_ARCHITECTURES, APPROVED_ARCHITECTURES_SOURCE, GROUP_AXES, REPORT_SCHEMA_VERSION,
    ReadOutcome, Refusal as ReportRefusal, Report, is_rocm_supported, prepare_report, read_report,
    refusal_envelope,
};
pub use report_rollup::{Group as ReportGroup, Rollup as ReportRollup};
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
pub use service_identity::{ServiceId, generate_service_id};
pub use service_record::{ManagedServiceRecord, engine_stop_decides_confirmation};
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

/// Replace `path`'s content with `bytes`, atomically.
///
/// Writes a sibling temp file and renames it over the destination: an
/// interrupted write (full disk, crash, killed process) can never leave a
/// truncated file in place, and a reader sees
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
/// - Serializing writers: this takes no lock, so two read-modify-write
///   cycles that overlap still lose one update (last rename wins). A caller
///   whose new content depends on the old must hold a lock across the whole
///   cycle, as the JSONL proposal store does (`automation::jsonl_lock_path`).
///   Two callers in the *same process* within the same millisecond are a
///   sharper case of this, not just a milder one: they generate the exact
///   same temp path below, so concurrent `fs::write`s truncating that one
///   path can publish mixed bytes, and the losing rename then fails outright
///   (its source file already gone) rather than merely losing an update —
///   this is the same-process collision
///   `automation::tests::concurrent_appends_from_many_threads_lose_no_records`'s
///   own comment describes ("two threads of one process even shared a
///   temp-file name") — fixed for its own callers by the per-store lock
///   `automation.rs` takes before ever calling into this function.
/// - Durability: the bytes are not fsynced before the rename, so a power
///   loss can still surface the new name with unflushed contents.
/// - Replacing a destination another process holds open without delete
///   sharing: on Windows the rename then fails with a sharing violation,
///   this returns that error, and the old destination stays in place.
pub fn write_atomic(path: &Path, bytes: &[u8], tmp_ext: &str) -> Result<()> {
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
