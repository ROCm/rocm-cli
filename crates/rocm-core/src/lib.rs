// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(all(test, unix))]
mod archive_props;
pub mod automation;
pub mod binary_paths;
pub mod browser;
pub mod codex_bridge;
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
    AutomationTriggerEvent, BuiltinWatcherSpec, WatcherRuntimeSnapshot, append_audit_event,
    append_automation_event, append_automation_proposal, builtin_watcher, builtin_watchers,
    find_automation_proposal, generate_proposal_id, load_automation_proposals,
    load_recent_audit_events, load_recent_automation_events, load_recent_automation_proposals,
    update_automation_proposal_status,
};
pub use binary_paths::{daemon_binary_path, engine_binary_path, sibling_binary_path};
pub use codex_bridge::{CodexBridgeEngine, CodexBridgeGpuSnapshot, CodexBridgeSnapshot};
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
    DownloadRequest, EndpointIdentity, EndpointReadiness, EndpointReadinessOutcome,
    HttpResponseParts, INFERENCE_PROBE_ATTEMPTED_STATE_KEY, INFERENCE_PROBE_RETRY_INTERVAL,
    INFERENCE_PROBE_TIMEOUT, INFERENCE_VERIFIED_STATE_KEY, connect_tcp_stream,
    download_file_streaming, download_file_streaming_with_progress, download_file_to_path,
    download_file_to_path_with_progress, engine_state_inference_verified, format_host_for_url,
    format_host_port, format_http_base_url, http_get_text, http_get_text_with_auth,
    http_get_with_auth, http_post_json_with_auth, managed_service_endpoint_identity,
    managed_service_endpoint_model_ready, managed_service_endpoint_readiness,
    merge_json_state_file, openai_chat_completion_probe, openai_chat_completion_status,
    openai_models_endpoint_has_model, openai_models_endpoint_identity, parse_http_endpoint,
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

/// How many temp names to try before giving up on staging an atomic write.
const ATOMIC_WRITE_TEMP_ATTEMPTS: u32 = 128;

/// A unique temp path next to `path`, preserving the full file name so a
/// multi-extension artifact keeps its extensions (`sdk.tar.gz` becomes
/// `sdk.tar.gz.tmp-<id>`, where `with_extension` would drop `.gz`).
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

/// Move a staged temp file onto `path`, replacing whatever is there.
///
/// The single implementation of the publish step for the whole workspace. It
/// lives here rather than in each caller because the Windows half is not
/// something to re-derive: a plain `rename` over an existing file is not
/// reliable while another process holds the destination open (an antivirus or
/// indexer scanning a file this CLI just wrote is enough), so an existing
/// destination goes through `ReplaceFileW`. Callers that got this wrong would
/// fail in the direction that matters — `ManagedServiceRecord::write` is on the
/// uninstall stop path, where a failed publish is reported as a service that
/// could not be stopped and aborts the whole uninstall.
#[cfg(not(windows))]
pub fn publish_temp_file(tmp: &Path, path: &Path) -> std::io::Result<()> {
    fs::rename(tmp, path)
}

/// Move a staged temp file onto `path`, replacing whatever is there.
///
/// See the non-Windows twin for why this is centralized.
#[cfg(windows)]
pub fn publish_temp_file(tmp: &Path, path: &Path) -> std::io::Result<()> {
    if path.try_exists()? {
        return replace_file_windows(path, tmp);
    }

    match fs::rename(tmp, path) {
        Ok(()) => Ok(()),
        Err(rename_error) => {
            // The destination can appear between the check and the rename.
            if path.try_exists()? {
                replace_file_windows(path, tmp)
            } else {
                Err(rename_error)
            }
        }
    }
}

#[cfg(windows)]
#[allow(unsafe_code)] // Win32 FFI
fn replace_file_windows(path: &Path, replacement: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::ReplaceFileW;

    let path_wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let replacement_wide: Vec<u16> = replacement
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();

    // SAFETY: both path buffers are valid, NUL-terminated UTF-16 strings and
    // remain alive for the duration of the synchronous Windows API call. The
    // optional backup, exclude, and reserved pointers are intentionally null.
    let replaced = unsafe {
        ReplaceFileW(
            path_wide.as_ptr(),
            replacement_wide.as_ptr(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    if replaced == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// Write `bytes` to `path` so a concurrent reader sees either the old contents
/// or the new ones, never a half-written file.
///
/// Staged under a unique sibling name reserved with `create_new` — so two
/// writers cannot pick the same scratch file — and published with
/// [`publish_temp_file`]. A failed publish takes the scratch file with it.
///
/// Four limits worth knowing before reaching for this:
///
/// * **Atomic, not durable.** The staged bytes are `sync_all`ed before the
///   publish, so a crash cannot leave a half-written or zero-length file. The
///   containing *directory* is never fsynced, so the rename itself can still be
///   lost by a power failure — the guarantee is that readers only ever see one
///   complete version, not that the newest one survives a crash. That flush is
///   held by construction, not by test: deleting it leaves every test green,
///   because reproducing what it prevents needs a real crash between the write
///   and the rename. Do not read a green suite as evidence it is still there.
/// * **Permissions are not carried over.** This replaces the target inode with
///   a freshly created file, so an existing file's mode does not survive (a
///   difference from the `fs::write` it usually replaces), and the new file gets
///   whatever the umask allows. The containing directory does not make up for
///   it: `AppPaths::ensure` creates it with plain `create_dir_all` and no mode of
///   its own. A caller with something sensitive to write must set the mode
///   explicitly here; it cannot lean on the directory. `RocmCliConfig::save` is
///   such a caller today (it persists the dashboard token) and does not yet.
/// * **No lock.** Two read-modify-write cycles that overlap still lose one
///   update (last rename wins), so a caller whose new content depends on the old
///   must hold a lock across the whole cycle, as the JSONL stores in
///   `automation` do.
/// * **A symlink at `path` is replaced, not written through.** The publish is a
///   rename onto `path` itself, so a symlink there is what gets replaced — where
///   `fs::write` would have followed it and overwritten the target. That is the
///   safer direction for state files (it cannot be aimed somewhere else by
///   planting a link) but it is a behaviour change, so a caller that means to
///   write through a link must resolve it first.
pub fn write_file_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let temp_id = format!("{}-{}", std::process::id(), unix_time_millis());
    write_file_atomically_with_temp_id(path, bytes, &temp_id)
}

/// [`write_file_atomically`] with the scratch-file id supplied, so a test can
/// plant files under the names the attempts will try and drive the
/// already-exists retry and the exhausted-attempts failure deterministically.
fn write_file_atomically_with_temp_id(path: &Path, bytes: &[u8], temp_id: &str) -> Result<()> {
    let parent = path.parent().context("file path has no parent directory")?;
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;

    let mut reserved = None;
    for attempt in 0..ATOMIC_WRITE_TEMP_ATTEMPTS {
        let tmp = temp_sibling_path(path, &OsString::from(format!("{temp_id}-{attempt}")))?;
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
        {
            Ok(file) => {
                reserved = Some((tmp, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(error).with_context(|| format!("failed to create {}", tmp.display()));
            }
        }
    }
    let Some((tmp, mut file)) = reserved else {
        bail!(
            "failed to reserve a temporary file next to {} after {ATOMIC_WRITE_TEMP_ATTEMPTS} \
             attempts",
            path.display()
        );
    };

    if let Err(error) = file.write_all(bytes) {
        drop(file);
        let _ = fs::remove_file(&tmp);
        return Err(disk_space::map_write_error(error, &tmp));
    }
    // Flush before publishing. Without this the rename can reach the disk while
    // the bytes have not, leaving a zero-length file after a crash — which for a
    // service record is not a lost update but an unparseable manifest, and the
    // uninstall gate refuses to remove anything while one of those exists.
    if let Err(error) = file.sync_all() {
        drop(file);
        let _ = fs::remove_file(&tmp);
        return Err(disk_space::map_write_error(error, &tmp));
    }
    drop(file);

    publish_temp_file(&tmp, path)
        .inspect_err(|_| {
            let _ = fs::remove_file(&tmp);
        })
        .with_context(|| format!("failed to publish {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use std::path::Path;

    /// A private root for one test, removed on the way out.
    fn atomic_write_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "rocm-core-atomic-write-{name}-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("failed to create the test root");
        root
    }

    /// The scratch files `write_file_atomically` stages under, if any survived.
    ///
    /// `temp_sibling_path` names them `<file>.tmp-<pid>-<millis>-<attempt>`, so
    /// a leftover is visible as a sibling containing `.tmp-`.
    fn leftover_scratch_files(dir: &Path) -> Vec<String> {
        fs::read_dir(dir)
            .expect("failed to read the test root")
            .filter_map(|entry| {
                let name = entry.ok()?.file_name().to_string_lossy().into_owned();
                name.contains(".tmp-").then_some(name)
            })
            .collect()
    }

    #[test]
    fn a_taken_scratch_name_is_skipped_and_the_write_still_lands() {
        // Another writer already holds attempt 0's name: the write must move on to
        // the next attempt rather than fail or clobber the file it does not own.
        let root = atomic_write_root("scratch-name-taken");
        let target = root.join("record.json");
        let taken = root.join("record.json.tmp-fixed-0");
        fs::write(&taken, b"someone else's scratch").expect("plant the taken name");

        write_file_atomically_with_temp_id(&target, b"mine", "fixed").expect("write");

        assert_eq!(fs::read(&target).expect("read"), b"mine");
        assert_eq!(
            fs::read(&taken).expect("read"),
            b"someone else's scratch",
            "a scratch file that is not ours is left untouched"
        );
        assert_eq!(
            leftover_scratch_files(&root),
            vec!["record.json.tmp-fixed-0".to_owned()],
            "our own scratch file is gone once published"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn every_scratch_name_taken_is_an_error_not_a_clobber() {
        let root = atomic_write_root("scratch-names-exhausted");
        let target = root.join("record.json");
        fs::write(&target, b"original").expect("seed the target");
        for attempt in 0..ATOMIC_WRITE_TEMP_ATTEMPTS {
            fs::write(root.join(format!("record.json.tmp-fixed-{attempt}")), b"x")
                .expect("plant a taken name");
        }

        let error = write_file_atomically_with_temp_id(&target, b"new", "fixed")
            .expect_err("no scratch name is free");

        assert!(
            format!("{error:#}").contains("failed to reserve a temporary file"),
            "{error:#}"
        );
        assert_eq!(
            fs::read(&target).expect("read"),
            b"original",
            "a failed reservation leaves the target as it was"
        );
        let _ = fs::remove_dir_all(root);
    }

    /// Linux/unix: relies on a rename replacing the directory entry while an
    /// already-open descriptor keeps reading the previous inode. Windows
    /// publishes through `ReplaceFileW` and does not offer the same observation.
    #[cfg(unix)]
    #[test]
    fn a_service_record_write_never_shows_a_reader_a_truncated_manifest() {
        // `ManagedServiceRecord::write` runs on the uninstall stop path, where a
        // torn manifest is not a lost update but an unparseable record that makes
        // the gate refuse to remove anything.
        //
        // Asserting "the bytes landed" would not pin that: a plain `fs::write`
        // also leaves correct final bytes and creates no scratch file, so such a
        // test passes either way. The property only the atomic path has is what a
        // *concurrent reader* sees — `fs::write` truncates the existing inode in
        // place, so a reader holding it open watches the record disappear and
        // come back, while the publish swaps in a new inode and leaves the old
        // one whole. Hold the manifest open across a rewrite and require the old
        // view to still be a complete, parseable record.
        use std::io::{Read, Seek, SeekFrom};

        let root = atomic_write_root("service-record-reader-safety");
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        paths.ensure().expect("create the app directories");
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-reader-safety",
            "vllm",
            "amd/model",
            "amd/model",
            "127.0.0.1",
            8000,
            "managed",
            0,
            None,
            None,
            None,
        );
        record.status = "ready".to_owned();
        record.write().expect("seed the manifest");

        // A reader that opened the manifest before the rewrite began.
        let mut reader = fs::File::open(&record.manifest_path).expect("open the manifest");

        record.status = "stopped".to_owned();
        record.write().expect("rewrite the manifest");

        reader.seek(SeekFrom::Start(0)).expect("rewind");
        let mut seen = Vec::new();
        reader
            .read_to_end(&mut seen)
            .expect("read the open manifest");
        let parsed: ManagedServiceRecord = serde_json::from_slice(&seen)
            .expect("a reader holding the manifest open must never see a truncated document");
        assert_eq!(
            parsed.status, "ready",
            "the open descriptor must still see the pre-write record, not a rewritten inode"
        );

        // And the published file is the new one.
        let published: ManagedServiceRecord =
            serde_json::from_slice(&fs::read(&record.manifest_path).expect("read back"))
                .expect("the published manifest must parse");
        assert_eq!(published.status, "stopped", "the rewrite must have landed");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn a_runtime_state_write_never_shows_a_reader_a_truncated_file() {
        // The uninstall gate reads this file to decide whether a daemon is live; a
        // torn write makes it unparseable and aborts the run. Same property as
        // the service record: a reader holding the old inode must keep seeing a
        // whole document across a rewrite.
        use std::io::{Read, Seek, SeekFrom};

        let root = atomic_write_root("runtime-state-reader-safety");
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        paths.ensure().expect("create the app directories");
        let mut state = AutomationRuntimeState {
            running: true,
            automations_enabled: true,
            daemon_pid: 1,
            daemon_start_ticks: None,
            started_at_unix_ms: 1,
            last_tick_unix_ms: 1,
            local_webhook_endpoint: None,
            active_watchers: Vec::new(),
        };
        state.write(&paths).expect("seed the state file");
        let mut reader =
            fs::File::open(paths.automation_state_path()).expect("open the state file");

        state.running = false;
        state.write(&paths).expect("rewrite the state file");

        reader.seek(SeekFrom::Start(0)).expect("rewind");
        let mut seen = Vec::new();
        reader.read_to_end(&mut seen).expect("read the open file");
        let parsed: AutomationRuntimeState = serde_json::from_slice(&seen)
            .expect("a reader holding the file open must never see a truncated document");
        assert!(
            parsed.running,
            "the open descriptor must see the pre-write state"
        );
        let published = AutomationRuntimeState::load(&paths)
            .expect("load")
            .expect("present");
        assert!(!published.running, "the rewrite must have landed");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn a_service_record_write_leaves_no_scratch_sibling() {
        // This pins scratch-file hygiene, NOT atomic publishing: a plain
        // `fs::write` creates no scratch file at all and would pass too. The
        // atomicity property is pinned next door, by
        // `a_service_record_write_never_shows_a_reader_a_truncated_manifest`,
        // which fails against an in-place write.
        //
        // It still earns its place, but for a smaller reason than uninstall:
        // scratch siblings are named `<record>.json.tmp-<pid>-<millis>-<attempt>`,
        // whose extension is `tmp-…`, and both `unreadable_service_manifests`
        // and the record loader filter on `extension == "json"` — so a leaked
        // one is skipped, not mistaken for a corrupt record, and cannot abort
        // uninstall. What this guards is directory hygiene: a publish that
        // aborts between staging and rename leaves litter that accumulates
        // silently, one file per failed write, in a directory an operator reads
        // by hand.
        let root = atomic_write_root("service-record-publish");
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        paths.ensure().expect("create the app directories");
        let mut record = ManagedServiceRecord::new(
            &paths,
            "svc-atomic-publish",
            "vllm",
            "amd/model",
            "amd/model",
            "127.0.0.1",
            8000,
            "managed",
            0,
            None,
            None,
            None,
        );
        record.write().expect("first publish");
        record.status = "stopped".to_owned();
        record
            .write()
            .expect("republish over the existing manifest");

        let services_dir = paths.services_dir();
        let stray = fs::read_dir(&services_dir)
            .expect("read the services directory")
            .filter_map(|entry| {
                let name = entry.ok()?.file_name().to_string_lossy().into_owned();
                name.contains(".tmp-").then_some(name)
            })
            .collect::<Vec<_>>();
        assert!(
            stray.is_empty(),
            "publishing a record must leave no scratch manifest: {stray:?}"
        );
        let written = fs::read(&record.manifest_path).expect("read the manifest back");
        let parsed: ManagedServiceRecord =
            serde_json::from_slice(&written).expect("the published manifest must always parse");
        assert_eq!(parsed.status, "stopped", "the republish must have landed");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn write_file_atomically_replaces_contents_and_leaves_no_scratch_file() {
        // The publish must land the new bytes and take its staging file with it;
        // a leaked scratch sibling in `services_dir` would be a file the uninstall
        // gate has to reason about.
        let root = atomic_write_root("replaces");
        let target = root.join("record.json");
        fs::write(&target, b"old contents").expect("seed the file");

        write_file_atomically(&target, b"new contents").expect("the write must succeed");

        assert_eq!(
            fs::read(&target).expect("read back"),
            b"new contents",
            "the published file must hold the new bytes"
        );
        assert!(
            leftover_scratch_files(&root).is_empty(),
            "a successful publish must leave no scratch file: {:?}",
            leftover_scratch_files(&root)
        );
        let _ = fs::remove_dir_all(root);
    }

    /// Unix: needs a real symlink.
    #[cfg(unix)]
    #[test]
    fn write_file_atomically_replaces_a_symlink_instead_of_writing_through_it() {
        // Documented behaviour, and a change from the `fs::write` this replaced:
        // that followed a symlink and overwrote whatever it pointed at, while
        // the publish here renames onto the path itself. Security-relevant in
        // the safe direction — planting a link next to a state file can no
        // longer redirect the write somewhere else — but a change, so it is
        // pinned rather than left as prose.
        let root = atomic_write_root("symlink-target");
        let elsewhere = root.join("elsewhere");
        fs::write(&elsewhere, b"must not be touched").expect("seed the link target");
        let target = root.join("state.json");
        std::os::unix::fs::symlink(&elsewhere, &target).expect("plant a symlink at the target");

        write_file_atomically(&target, b"published").expect("the write must succeed");

        assert_eq!(
            fs::read(&elsewhere).expect("read the link target"),
            b"must not be touched",
            "the write must not follow the symlink to its target"
        );
        assert_eq!(
            fs::read(&target).expect("read the published file"),
            b"published",
            "the published bytes must land at the path itself"
        );
        assert!(
            !fs::symlink_metadata(&target)
                .expect("stat the published path")
                .file_type()
                .is_symlink(),
            "the symlink must have been replaced by a regular file"
        );
        assert!(
            leftover_scratch_files(&root).is_empty(),
            "no scratch file may survive a successful publish"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn write_file_atomically_creates_the_parent_directory() {
        // `ManagedServiceRecord::write` targets a services dir that may not exist
        // yet on a first write.
        let root = atomic_write_root("creates-parent");
        let target = root.join("nested").join("deeper").join("record.json");

        write_file_atomically(&target, b"contents").expect("the write must succeed");

        assert_eq!(fs::read(&target).expect("read back"), b"contents");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn write_file_atomically_cleans_up_when_publishing_fails() {
        // A failed publish must not strand its staging file. The destination is a
        // non-empty directory, which no platform will let a file replace, so the
        // failure happens at the publish step with the scratch file already
        // written — exactly the path that has to clean up after itself.
        let root = atomic_write_root("publish-fails");
        let target = root.join("record.json");
        fs::create_dir_all(&target).expect("seed a directory where the file goes");
        fs::write(target.join("occupant"), b"x").expect("make it non-empty");

        let error = write_file_atomically(&target, b"contents")
            .expect_err("replacing a non-empty directory must fail");

        assert!(
            leftover_scratch_files(&root).is_empty(),
            "a failed publish must remove its scratch file, got {:?} ({error:#})",
            leftover_scratch_files(&root)
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn write_file_atomically_refuses_a_path_with_no_parent_or_an_unusable_parent() {
        let error = write_file_atomically(Path::new("/"), b"x")
            .expect_err("a root path has no directory to stage in");
        assert!(format!("{error:#}").contains("no parent"), "{error:#}");

        // A regular file where the parent directory should be: the directory
        // cannot be created, and nothing may be left behind.
        let root = atomic_write_root("parent-is-a-file");
        let blocker = root.join("blocker");
        fs::write(&blocker, b"x").expect("seed the blocking file");
        let error = write_file_atomically(&blocker.join("record.json"), b"x")
            .expect_err("a file cannot be a parent directory");
        assert!(
            format!("{error:#}").contains("failed to create"),
            "{error:#}"
        );
        assert!(leftover_scratch_files(&root).is_empty());
        let _ = fs::remove_dir_all(root);
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
    fn write_file_atomically_writes_bytes_and_leaves_no_temp_file() -> Result<()> {
        let temp_root =
            std::env::temp_dir().join(format!("rocm-cli-write-atomic-ok-{}", unix_time_millis()));
        fs::create_dir_all(&temp_root)?;
        let path = temp_root.join("store.jsonl");

        write_file_atomically(&path, b"hello\n")?;

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
    fn write_file_atomically_rename_failure_leaves_destination_and_dir_untouched() -> Result<()> {
        let temp_root =
            std::env::temp_dir().join(format!("rocm-cli-write-atomic-fail-{}", unix_time_millis()));
        fs::create_dir_all(&temp_root)?;
        let path = temp_root.join("store.jsonl");
        // A file can't be renamed over an existing directory, so making the
        // destination a directory forces `write_file_atomically`'s rename step to fail
        // after the temp file has already been written.
        fs::create_dir_all(&path)?;

        let result = write_file_atomically(&path, b"hello\n");
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
