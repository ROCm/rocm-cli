// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand};
use rocm_core::{
    AppPaths, DEFAULT_LOCAL_PORT, DependencyViolation, check_dependencies, ensure_uv_binary,
    format_http_base_url, openai_models_endpoint_has_model, require_nonempty, split_local_version,
    uv_command_env, uv_pip_install_base, violation_subject, violations_requiring,
};
use rocm_engine_protocol::{
    DEFAULT_LOG_TAIL_LINES, DetectRequest, DetectResponse, DevicePolicy,
    ENGINE_RECIPE_CONTRACT_VERSION, EndpointRequest, EndpointResponse, EngineCapabilities,
    EngineDeviceAvailability, EngineMethod, EngineRecipeHint, EngineRequestEnvelope,
    EngineResponseEnvelope, GpuSelection, HealthcheckRequest, HealthcheckResponse, InstallRequest,
    InstallResponse, LaunchRequest, LaunchResponse, LogsRequest, LogsResponse, ResolveModelRequest,
    ResolveModelResponse, StopRequest, StopResponse,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::hash_map::DefaultHasher;
use std::ffi::OsString;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const ENGINE_NAME: &str = "vllm";
const DEFAULT_HOST: &str = "127.0.0.1";
const HEALTHCHECK_TIMEOUT_MS: u64 = 700;
const STARTUP_FAILURE_LOG_TAIL_LINES: usize = 80;
const MAX_TAIL_READ: u64 = 4 * 1024 * 1024;
/// How long a stop waits for the server to actually exit after each signal
/// before reporting a timeout (or, under `force`, escalating to `SIGKILL`).
const STOP_GRACE: Duration = Duration::from_secs(10);
/// vLLM launches worker descendants that retain GPU allocations.
const STOP_SCOPE: rocm_core::KillScope = rocm_core::KillScope::Tree;
/// Default ROCm wheel index for `uv pip install vllm`.
///
/// This pins both the vLLM release and the ROCm ABI tag, so it can drift from
/// the resolved runtime. Override it with `ROCM_CLI_VLLM_ROCM_INDEX_URL` to
/// match a different vLLM/ROCm combination without rebuilding; see
/// [`resolve_vllm_install_target`] for how the requirement follows the URL.
///
/// Keep the release and ABI tag here in sync with [`VLLM_PINNED_SPEC`] below;
/// `vllm_pinned_spec_matches_extra_index_url` fails if the two drift apart.
const VLLM_ROCM_EXTRA_INDEX_URL: &str = "https://wheels.vllm.ai/rocm/0.26.0/rocm723";
/// Exact requirement handed to `uv pip install` for [`VLLM_ROCM_EXTRA_INDEX_URL`].
///
/// The index URL alone is not a pin: it only constrains where wheels are
/// fetched from, so a bare `vllm` requirement would install whatever version
/// that path happens to serve (or fall back to PyPI when no wheel on the index
/// matches the interpreter). Spelling the version and local ABI tag out here
/// makes the pin real.
const VLLM_PINNED_SPEC: &str = "vllm==0.26.0+rocm723";
/// Prefix shared by every published vLLM ROCm wheel index.
///
/// Release indexes are `{VLLM_ROCM_INDEX_PREFIX}/<version>/<abi>`, which is
/// what lets [`vllm_rocm_build_from_index_url`] recover the build a custom
/// index serves and keep the requirement pinned to it.
const VLLM_ROCM_INDEX_PREFIX: &str = "https://wheels.vllm.ai/rocm";
/// Default time to wait for vLLM to report readiness before giving up.
const DEFAULT_VLLM_READY_TIMEOUT: Duration = Duration::from_mins(5);

#[derive(Parser, Debug)]
#[command(name = "rocm-engine-vllm", about = "rocm-cli vLLM engine adapter")]
struct Cli {
    #[command(subcommand)]
    command: CommandKind,
}

#[derive(Subcommand, Debug)]
enum CommandKind {
    Detect,
    Capabilities,
    Install {
        #[arg(long, default_value = "external-vllm")]
        runtime_id: String,
        #[arg(long)]
        reinstall: bool,
    },
    ResolveModel {
        model_ref: String,
        #[arg(long)]
        device_policy: Option<String>,
    },
    Launch {
        service_id: String,
        model_ref: String,
        #[arg(long, default_value = DEFAULT_HOST)]
        host: String,
        #[arg(long, default_value_t = DEFAULT_LOCAL_PORT)]
        port: u16,
        #[arg(long)]
        device_policy: Option<String>,
        #[arg(long)]
        gpu: Option<String>,
        #[arg(long)]
        runtime_id: Option<String>,
        #[arg(long)]
        env_id: Option<String>,
    },
    Stdio,
    ServeHttp {
        service_id: String,
        model_ref: String,
        #[arg(long, default_value = DEFAULT_HOST)]
        host: String,
        #[arg(long, default_value_t = DEFAULT_LOCAL_PORT)]
        port: u16,
        #[arg(long, default_value = "gpu_required")]
        device_policy: String,
        #[arg(long)]
        gpu: Option<String>,
        #[arg(long)]
        runtime_id: Option<String>,
        #[arg(long)]
        env_id: Option<String>,
        #[arg(long)]
        state_path: PathBuf,
        #[arg(long)]
        engine_recipe_json: Option<String>,
    },
}

#[derive(Debug, Clone)]
struct VllmRuntime {
    runtime_id: String,
    env_id: String,
    command: PathBuf,
    python_executable: Option<PathBuf>,
    version: Option<String>,
    source: String,
    sdk_root: Option<PathBuf>,
    sdk_bin: Option<PathBuf>,
    sdk_bin_paths: Vec<PathBuf>,
    sdk_library_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TheRockRuntimeManifest {
    #[serde(default)]
    runtime_key: Option<String>,
    #[serde(default)]
    runtime_id: Option<String>,
    #[serde(default)]
    python_executable: Option<PathBuf>,
    #[serde(default)]
    rocm_sdk: Option<RocmSdkRuntimeProbe>,
    /// The SDK's own version, used only to reconstruct a build identifier for
    /// manifests written before `sdk_torch` was recorded.
    #[serde(default)]
    version: Option<String>,
    /// The torch the SDK install wrote, e.g. `2.11.0+rocm7.13.0`.
    ///
    /// The CLI records it so a later engine install can be told apart from the SDK's
    /// own work. Read here for the same reason in reverse: it is the only way this
    /// engine can recognise a torch the CLI deliberately put back, as opposed to one
    /// some other installer left behind.
    #[serde(default)]
    sdk_torch: Option<String>,
    #[serde(default)]
    installed_at_unix_ms: Option<u128>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RocmSdkRuntimeProbe {
    #[serde(default)]
    import_ok: bool,
    #[serde(default)]
    root_path: Option<PathBuf>,
    #[serde(default)]
    bin_path: Option<PathBuf>,
    #[serde(default)]
    bin_paths: Vec<PathBuf>,
    #[serde(default)]
    library_paths: Vec<PathBuf>,
    #[serde(default)]
    rocm_sdk_version: Option<String>,
}

#[derive(Debug, Clone)]
struct ServiceFiles {
    state_path: PathBuf,
    log_path: PathBuf,
}

#[derive(Debug, Clone)]
struct ManagedRuntimePython {
    runtime_id: String,
    python_executable: PathBuf,
}

pub fn run_cli() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        CommandKind::Detect => print_json(&detect_response())?,
        CommandKind::Capabilities => print_json(&capabilities())?,
        CommandKind::Install {
            runtime_id,
            reinstall,
        } => print_json(&install_response(InstallRequest {
            runtime_id,
            python_version: None,
            env_root: None,
            reinstall,
        })?)?,
        CommandKind::ResolveModel {
            model_ref,
            device_policy,
        } => print_json(&resolve_model_response(ResolveModelRequest {
            model_ref,
            runtime_id: None,
            device_policy: device_policy
                .as_deref()
                .map(|value| parse_device_policy_arg(Some(value)))
                .transpose()?,
            recipe_override: None,
            engine_recipe: None,
        })?)?,
        CommandKind::Launch {
            service_id,
            model_ref,
            host,
            port,
            device_policy,
            gpu,
            runtime_id,
            env_id,
        } => print_json(&launch_service(LaunchRequest {
            service_id,
            env_id,
            runtime_id,
            model_ref,
            host,
            port,
            device_policy: Some(parse_device_policy_arg(device_policy.as_deref())?),
            endpoint_mode: Some("openai".to_owned()),
            engine_recipe: None,
            gpu_selection: parse_gpu_selection_arg(gpu.as_deref())?,
        })?)?,
        CommandKind::Stdio => {
            let envelope = read_request()?;
            print_json(&handle_envelope(envelope))?;
        }
        CommandKind::ServeHttp {
            service_id,
            model_ref,
            host,
            port,
            device_policy,
            gpu,
            runtime_id,
            env_id,
            state_path,
            engine_recipe_json,
        } => {
            let log_path = AppPaths::discover()?
                .engine_logs_dir(ENGINE_NAME)
                .join(format!("{service_id}.log"));
            serve_http(ServeHttpRequest {
                service_id,
                model_ref,
                host,
                port,
                device_policy: parse_device_policy_arg(Some(&device_policy))?,
                gpu_indices: parse_gpu_indices_arg(gpu.as_deref())?,
                runtime_id,
                env_id,
                state_path,
                log_path: Some(log_path),
                engine_recipe: parse_engine_recipe_json(engine_recipe_json)?,
            })?;
        }
    }
    Ok(())
}

pub fn builtin_handle_envelope(envelope: EngineRequestEnvelope) -> EngineResponseEnvelope {
    handle_envelope(envelope)
}

#[allow(clippy::too_many_arguments)]
pub fn builtin_serve_http(
    service_id: String,
    model_ref: String,
    host: String,
    port: u16,
    device_policy: DevicePolicy,
    gpu_indices: Vec<u32>,
    runtime_id: Option<String>,
    env_id: Option<String>,
    state_path: PathBuf,
    log_path: Option<PathBuf>,
    engine_recipe: Option<EngineRecipeHint>,
) -> Result<()> {
    serve_http(ServeHttpRequest {
        service_id,
        model_ref,
        host,
        port,
        device_policy,
        gpu_indices,
        runtime_id,
        env_id,
        state_path,
        log_path,
        engine_recipe,
    })
}

fn handle_envelope(envelope: EngineRequestEnvelope) -> EngineResponseEnvelope {
    match envelope.method {
        EngineMethod::Detect => {
            deserialize_and_respond::<DetectRequest, _, _>(envelope.payload, |_| {
                Ok(detect_response())
            })
        }
        EngineMethod::Capabilities => EngineResponseEnvelope::success(capabilities()),
        EngineMethod::Install => {
            deserialize_and_respond::<InstallRequest, _, _>(envelope.payload, install_response)
        }
        EngineMethod::ResolveModel => deserialize_and_respond::<ResolveModelRequest, _, _>(
            envelope.payload,
            resolve_model_response,
        ),
        EngineMethod::Launch => {
            deserialize_and_respond::<LaunchRequest, _, _>(envelope.payload, launch_service)
        }
        EngineMethod::Healthcheck => deserialize_and_respond::<HealthcheckRequest, _, _>(
            envelope.payload,
            healthcheck_service,
        ),
        EngineMethod::Endpoint => {
            deserialize_and_respond::<EndpointRequest, _, _>(envelope.payload, endpoint_response)
        }
        EngineMethod::Stop => {
            deserialize_and_respond::<StopRequest, _, _>(envelope.payload, stop_service)
        }
        EngineMethod::Logs => {
            deserialize_and_respond::<LogsRequest, _, _>(envelope.payload, logs_response)
        }
    }
}

fn deserialize_and_respond<T, F, U>(payload: Value, handler: F) -> EngineResponseEnvelope
where
    T: DeserializeOwned,
    F: FnOnce(T) -> Result<U>,
    U: Serialize,
{
    match serde_json::from_value::<T>(payload) {
        Ok(request) => match handler(request) {
            Ok(response) => EngineResponseEnvelope::success(response),
            Err(error) => EngineResponseEnvelope::failure("request_failed", error.to_string()),
        },
        Err(error) => EngineResponseEnvelope::failure("invalid_payload", error.to_string()),
    }
}

fn detect_response() -> DetectResponse {
    let runtime = resolve_vllm_runtime(None);
    let installed = runtime.is_ok();
    let mut notes = Vec::new();
    if cfg!(windows) {
        notes.push(windows_unsupported_message().to_owned());
    } else if let Err(error) = runtime.as_ref() {
        notes.push(error.to_string());
    }
    let runtime = runtime.ok();
    if let Some(runtime) = runtime.as_ref() {
        notes.push(format!(
            "vLLM command resolved from {}; no CPU fallback is used",
            runtime.source
        ));
    }

    DetectResponse {
        installed,
        env_id: runtime.as_ref().map(|runtime| runtime.env_id.clone()),
        runtime_kind: Some("external_vllm".to_owned()),
        runtime_executable: runtime
            .as_ref()
            .map(|runtime| runtime.command.display().to_string()),
        managed_env: Some(runtime.as_ref().is_some_and(runtime_is_managed)),
        python_version: runtime
            .as_ref()
            .and_then(|runtime| runtime.version.as_ref())
            .map(|version| format!("vllm {version}")),
        torch_version: None,
        transformers_version: None,
        available_devices: vec![EngineDeviceAvailability {
            kind: "rocm_gpu".to_owned(),
            available: installed && !cfg!(windows),
            reason: if cfg!(windows) {
                Some(windows_unsupported_message().to_owned())
            } else if installed {
                None
            } else {
                Some("vLLM is not installed in a Linux/WSL ROCm Python environment".to_owned())
            },
        }],
        capabilities: capabilities(),
        notes,
    }
}

fn capabilities() -> EngineCapabilities {
    EngineCapabilities {
        cpu: false,
        rocm_gpu: !cfg!(windows),
        openai_compatible: true,
        tool_calling: false,
        quantized_models: "vllm-supported".to_owned(),
        reasoning_parser: false,
    }
}

fn install_response(request: InstallRequest) -> Result<InstallResponse> {
    // Resolve regardless of `reinstall`. Resolution answers *which* interpreter holds
    // vLLM, and a forced reinstall needs that answer just as much as a repair does —
    // gating it on `reinstall` left `--reinstall` with no assessed environment, so it
    // fell back to `resolve_managed_runtime_python` (first prefix-matching candidate)
    // and could reinstall a healthy environment while leaving the broken one broken.
    // Only the short-circuit below is gated on `reinstall`.
    let resolved = resolve_vllm_runtime(Some(&request.runtime_id)).ok();
    // A resolvable vLLM is not necessarily a usable one. `rocm install sdk` writes the
    // TheRock torch stack into the same environment vLLM lives in, so a second run
    // replaces the torch build vLLM pins without touching vLLM itself.
    // Short-circuiting on "vllm resolves" left that environment unrepaired; short-circuit
    // on "vllm's own requirements are met" instead, so the install below restores them.
    let (already_installed, repair, assessed) = match resolved {
        Some(runtime) => {
            let repair = assess_runtime_repair(&runtime);
            if request.reinstall || repair.needed {
                (None, repair, assessed_python_for_repair(&runtime))
            } else {
                (Some(runtime), repair, None)
            }
        }
        None => (None, RepairAssessment::default(), None),
    };
    let runtime = if let Some(runtime) = already_installed {
        runtime
    } else {
        // A repair installs into the environment that was *assessed*. The two resolvers
        // do not agree: `resolve_vllm_runtime` walks the candidates until one actually
        // has vLLM beside it, while `resolve_managed_runtime_python` takes the first
        // candidate unconditionally — and `runtime_id` matches by prefix, so several
        // candidates routinely qualify. Installing into a different interpreter than the
        // one found broken would leave the broken one broken and report it fixed.
        let managed = match assessed {
            Some(assessed) => assessed,
            None => resolve_managed_runtime_python(Some(&request.runtime_id))?.with_context(
                || {
                    let base = format!(
                        "runtime `{}` did not resolve to a managed TheRock Python environment for automatic vLLM install",
                        request.runtime_id
                    );
                    match describe_skipped_managed_runtimes(Some(&request.runtime_id)) {
                        Some(skipped) => format!("{base}\n\n{skipped}"),
                        None => base,
                    }
                },
            )?,
        };
        install_vllm_with_uv(&managed.python_executable, request.reinstall)?;
        resolve_vllm_runtime(Some(&managed.runtime_id)).with_context(|| {
            format!(
                "vLLM install completed in {}, but runtime `{}` still could not be resolved",
                managed.python_executable.display(),
                managed.runtime_id
            )
        })?
    };
    let env_path = runtime
        .command
        .parent()
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    Ok(InstallResponse {
        env_id: runtime.env_id.clone(),
        env_path: env_path.display().to_string(),
        python_executable: runtime
            .python_executable
            .as_ref()
            .unwrap_or(&runtime.command)
            .display()
            .to_string(),
        runtime_kind: Some("external_vllm".to_owned()),
        runtime_executable: Some(runtime.command.display().to_string()),
        managed_env: Some(runtime_is_managed(&runtime)),
        installed_packages: vec![format!(
            "vllm{}",
            runtime
                .version
                .as_deref()
                .map(|version| format!("=={version}"))
                .unwrap_or_default()
        )],
        capabilities: capabilities(),
        lock_hash: runtime_lock_hash(&runtime),
        warnings: repair
            .notes
            .into_iter()
            .chain(vllm_runtime_warnings(&runtime))
            .collect(),
    })
}

/// Whether a resolvable vLLM environment still needs an install pass, and what to tell
/// the user about why.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct RepairAssessment {
    /// The environment holds vLLM but not the dependencies vLLM declares, so the
    /// install must run even though `reinstall` was not requested.
    needed: bool,
    /// Findings to surface with the install response.
    notes: Vec<String>,
}

/// The environment a repair must target: the one that was assessed, named by its own
/// runtime id rather than by the (possibly prefix-matching) id the caller requested.
fn assessed_python_for_repair(runtime: &VllmRuntime) -> Option<ManagedRuntimePython> {
    Some(ManagedRuntimePython {
        runtime_id: runtime.runtime_id.clone(),
        python_executable: runtime.python_executable.clone()?,
    })
}

/// Decide whether an already-resolvable vLLM environment must be reinstalled.
///
/// Only managed environments are assessed. An external environment belongs to the user:
/// rocm-cli reports on it but does not rewrite its packages, which is the very failure
/// mode this check exists to catch.
fn assess_runtime_repair(runtime: &VllmRuntime) -> RepairAssessment {
    if !runtime_is_managed(runtime) {
        return RepairAssessment::default();
    }
    let Some(python) = runtime.python_executable.as_ref() else {
        return RepairAssessment::default();
    };
    let paths = match AppPaths::discover() {
        Ok(paths) => paths,
        Err(error) => return unverified_repair(&error.to_string()),
    };
    match check_dependencies(&paths, python) {
        Ok(violations) => repair_from_violations(
            &violations,
            recorded_sdk_torch_build(&runtime.runtime_id).as_deref(),
            torch_alignment_disabled(),
        ),
        // An unusable `uv` or an offline host must not block an install that would
        // otherwise succeed; report that the check did not run and carry on as before.
        Err(error) => unverified_repair(&error.to_string()),
    }
}

/// The package whose build the SDK and the engine both have an opinion about.
const TORCH_PACKAGE: &str = "torch";

/// Whether the user has opted out of rocm-cli choosing this runtime's torch.
///
/// The CLI's own opt-out is the same call, not a matching one: the engine cannot
/// call into the binary that owns the alignment, and a duplicated read is a
/// contract that drifts. [`rocm_core::torch_alignment_disabled`] carries the rest.
fn torch_alignment_disabled() -> bool {
    rocm_core::torch_alignment_disabled()
}

/// Whether this violation is the torch divergence rocm-cli deliberately leaves behind.
///
/// After an engine install, rocm-cli puts back the SDK's *build* of the torch release
/// the engine pins, because the engine's build cannot open a device against the
/// installed SDK libraries. The engine's metadata pins an exact version and cannot
/// express "same release, the SDK's build", so `uv pip check` reports the result as
/// unsatisfied forever. Treating that as a defect makes the two mechanisms fight: the
/// engine reinstalls torch to its own build, rocm-cli puts the SDK's back, and the
/// next invocation starts over — two full torch-stack flips a run, and a warning
/// claiming a repair that undid the intended state.
///
/// With the alignment disabled the rule is simply "any torch pin": see below.
///
/// Otherwise three conditions, all required, and they are exactly the rule rocm-cli
/// applies: take the *release* from the engine's pin and the *build* from the SDK.
///
/// The violation must be about torch — any other unmet requirement is real. The
/// installed release must be the one the engine pins; an SDK torch of a *different*
/// release is the separate bug where the engine cannot accept what the SDK installed,
/// and a reinstall is the right answer there. And the installed build must be the one
/// the runtime's manifest records for the SDK; a torch from neither side is the
/// breakage this check exists to catch. Miss any one and the engine either fights the
/// alignment or silently accepts a runtime that cannot serve.
fn is_intended_torch_divergence(
    detail: &str,
    sdk_torch_build: Option<&str>,
    torch_alignment_disabled: bool,
) -> bool {
    let Some(subject) = violation_subject(detail) else {
        return false;
    };
    if !subject.package.eq_ignore_ascii_case(TORCH_PACKAGE) {
        return false;
    }
    // Opted out, so rocm-cli does not choose this runtime's torch and no build it
    // holds can be wrong *here*: the pin is unmet because the user meant it to be.
    // The build and release tests below are the aligned-case rule — asking a
    // hand-installed torch to match the SDK's build would fail every time, and the
    // reinstall that followed would install the engine's build over exactly the torch
    // the opt-out exists to keep. Only torch is spared: the check above already
    // rejected every other package, so an unrelated vLLM-owned defect still repairs.
    if torch_alignment_disabled {
        return true;
    }
    let Some(sdk_torch_build) = sdk_torch_build else {
        return false;
    };
    let (Some(required), Some(installed)) =
        (subject.required.as_deref(), subject.installed.as_deref())
    else {
        return false;
    };
    let (installed_release, Some(installed_build)) = split_local_version(installed) else {
        return false;
    };
    installed_build == sdk_torch_build && installed_release == split_local_version(required).0
}

/// The repair decision for a set of violations found in the environment.
///
/// `sdk_torch_build` is the build identifier the runtime's manifest records for the
/// SDK's torch, or `None` when it cannot be determined — in which case nothing is
/// treated as intended and the previous behaviour stands.
///
/// `torch_alignment_disabled` is the user's opt-out. It changes which violations count
/// as defects, never whether defects are acted on: a torch pin stops being one, and
/// everything else vLLM requires is assessed exactly as before. Returning early on the
/// opt-out instead would hide a broken torchvision behind an unrelated preference.
fn repair_from_violations(
    violations: &[DependencyViolation],
    sdk_torch_build: Option<&str>,
    torch_alignment_disabled: bool,
) -> RepairAssessment {
    let owned = violations_requiring(violations, ENGINE_NAME);
    if owned.is_empty() {
        return RepairAssessment::default();
    }
    let (intended, defects): (Vec<&DependencyViolation>, Vec<&DependencyViolation>) =
        owned.into_iter().partition(|violation| {
            is_intended_torch_divergence(
                &violation.detail,
                sdk_torch_build,
                torch_alignment_disabled,
            )
        });

    if defects.is_empty() {
        // Nothing to repair. Reinstalling here would replace that torch with the
        // engine's build and hand back a runtime nobody asked for.
        //
        // Which sentence is true depends on whose torch this is. Under the alignment it
        // is the SDK's and rocm-cli put it there; under the opt-out it is the user's and
        // rocm-cli never touched it. Reusing the first line for the second case would
        // tell a user who hand-installed torch that the CLI had installed it for them.
        let headline = if torch_alignment_disabled {
            "torch alignment is disabled by ROCM_CLI_DISABLE_TORCH_ALIGNMENT; the torch this runtime holds is the user's and a reinstall would replace it"
        } else {
            "the runtime holds the SDK's build of the torch vLLM pins; that divergence is intended and a reinstall would undo it"
        };
        let mut notes = vec![headline.to_owned()];
        notes.extend(
            intended
                .iter()
                .map(|violation| format!("expected divergence: {}", violation.detail)),
        );
        return RepairAssessment {
            needed: false,
            notes,
        };
    }

    let mut notes = vec![
        "the runtime environment did not satisfy vLLM's pinned dependencies; vLLM was reinstalled to restore them".to_owned(),
    ];
    // One note per violation rather than one joined line. The real failure is the whole
    // torch stack — torch, torchvision and torchaudio move together when the SDK writes
    // over the engine's pins — so joining them produced a single ~380-character line that
    // is unreadable in a terminal. Mirrors the per-finding `violation:` lines the
    // CLI-side renderer already emits.
    notes.extend(
        defects
            .iter()
            .map(|violation| format!("violation: {}", violation.detail)),
    );
    // An intended divergence alongside a real one is still worth naming, so the reader
    // is not left thinking the reinstall was about torch when it was not.
    notes.extend(
        intended
            .iter()
            .map(|violation| format!("expected divergence: {}", violation.detail)),
    );
    // The hint blames `rocm install sdk` for writing the SDK torch stack over vLLM's
    // pins, which stops being a live theory once the manifest names the SDK's build:
    // the alignment then identifies that stack and settles it, so a defect surviving
    // to here is something else. It stays on under the opt-out, which spares only the
    // package named `torch` — a `torchvision` or `torchaudio` defect is still the SDK
    // stack written over vLLM's pins, and the opt-out has turned off the step that
    // would have corrected it, so the hint is more use there rather than less.
    if sdk_torch_build.is_none() {
        notes.push(
            "if this recurs after `rocm install sdk`, the SDK torch stack is being written over vLLM's pinned torch".to_owned(),
        );
    }
    RepairAssessment {
        needed: true,
        notes,
    }
}

fn unverified_repair(reason: &str) -> RepairAssessment {
    RepairAssessment {
        needed: false,
        notes: vec![format!(
            "vLLM's pinned dependencies could not be verified in this environment: {reason}"
        )],
    }
}

fn runtime_is_managed(runtime: &VllmRuntime) -> bool {
    runtime.source.starts_with("managed_runtime_manifest")
}

fn vllm_runtime_warnings(runtime: &VllmRuntime) -> Vec<String> {
    let runtime_scope = if runtime_is_managed(runtime) {
        "rocm-cli records this vLLM command from a managed TheRock runtime; `rocm engines install vllm` can install vLLM into that runtime"
    } else {
        "rocm-cli records this as an external vLLM runtime; install/upgrade vLLM in that environment manually"
    };
    let mut warnings = vec![
        runtime_scope.to_owned(),
        "vLLM serving remains ROCm GPU required; no CPU fallback is used".to_owned(),
    ];
    if !cfg!(windows) && !rocm_core::openmpi::detect_openmpi().present {
        warnings.push(format!(
            "OpenMPI runtime (libmpi.so / libmpi_cxx.so / mpirun) was not found; vLLM requires it. {}, or rerun `rocm engines install vllm --yes`.",
            rocm_core::openmpi::install_hint()
        ));
    }
    if !cfg!(windows) && !rocm_core::openmpi::libatomic_present() {
        warnings.push(format!(
            "libatomic runtime (libatomic.so.1) was not found; vLLM's torch wheel requires it. {}, or rerun `rocm engines install vllm --yes`.",
            rocm_core::openmpi::libatomic_install_hint()
        ));
    }
    if !cfg!(windows) && !rocm_core::openmpi::libnuma_present() {
        warnings.push(format!(
            "system numactl runtime (libnuma.so.1 with libnuma_1.2) was not found; vLLM's torch wheel requires it and the ROCm SDK's bundled numa cannot satisfy it. {}, or rerun `rocm engines install vllm --yes`.",
            rocm_core::openmpi::libnuma_install_hint()
        ));
    }
    warnings
}

fn resolve_model_response(request: ResolveModelRequest) -> Result<ResolveModelResponse> {
    let device_policy = normalize_vllm_device_policy(request.device_policy)?;
    let engine_recipe = accepted_engine_recipe(request.engine_recipe)?;
    Ok(ResolveModelResponse {
        canonical_model_id: request.model_ref,
        task: "text-generation".to_owned(),
        source: "huggingface_or_local".to_owned(),
        revision: "main".to_owned(),
        loader: "vllm".to_owned(),
        trust_remote_code: false,
        chat_template_mode: "engine_default".to_owned(),
        dtype: "auto".to_owned(),
        device_policy,
        estimated_memory: "engine-reported".to_owned(),
        launch_defaults: json!({
            "endpoint_mode": "openai",
            "host": DEFAULT_HOST,
            "port": DEFAULT_LOCAL_PORT
        }),
        engine_recipe,
        warnings: vec![
            "vLLM is treated as a ROCm GPU engine in rocm-cli; select another engine explicitly for CPU serving".to_owned(),
        ],
    })
}

fn accepted_engine_recipe(
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

fn parse_engine_recipe_json(value: Option<String>) -> Result<Option<EngineRecipeHint>> {
    value
        .map(|text| {
            serde_json::from_str::<EngineRecipeHint>(&text)
                .context("failed to parse engine recipe JSON")
        })
        .transpose()
        .and_then(accepted_engine_recipe)
}

fn launch_service(request: LaunchRequest) -> Result<LaunchResponse> {
    let device_policy = normalize_vllm_device_policy(request.device_policy)?;
    let engine_recipe = accepted_engine_recipe(request.engine_recipe)?;
    let requested_gpu_indices =
        rocm_engine_protocol::launch_gpu_indices(request.gpu_selection.as_ref());
    let gpu_indices = resolve_serve_gpu_indices(&requested_gpu_indices)?;
    let runtime = resolve_vllm_runtime(request.runtime_id.as_deref())?;
    let state_path = AppPaths::discover()?
        .engine_state_dir(ENGINE_NAME)
        .join(format!("{}.json", request.service_id));
    let log_path = AppPaths::discover()?
        .engine_logs_dir(ENGINE_NAME)
        .join(format!("{}.log", request.service_id));
    let serve_request = ServeHttpRequest {
        service_id: request.service_id.clone(),
        model_ref: request.model_ref.clone(),
        host: request.host.clone(),
        port: request.port,
        device_policy,
        gpu_indices,
        runtime_id: request.runtime_id.clone(),
        env_id: request.env_id.clone(),
        state_path: state_path.clone(),
        log_path: Some(log_path.clone()),
        engine_recipe,
    };
    let child = spawn_vllm_server(&serve_request, &runtime, Some(&log_path))?;
    let pid = child.id();
    write_running_state(&serve_request, &runtime, pid)?;
    Ok(LaunchResponse {
        service_id: request.service_id,
        pid,
        endpoint_url: endpoint_url(&request.host, request.port),
        log_path: log_path.display().to_string(),
        state_path: state_path.display().to_string(),
    })
}

#[derive(Debug, Clone)]
struct ServeHttpRequest {
    service_id: String,
    model_ref: String,
    host: String,
    port: u16,
    device_policy: DevicePolicy,
    gpu_indices: Vec<u32>,
    runtime_id: Option<String>,
    env_id: Option<String>,
    state_path: PathBuf,
    log_path: Option<PathBuf>,
    engine_recipe: Option<EngineRecipeHint>,
}

fn serve_http(mut request: ServeHttpRequest) -> Result<()> {
    request.gpu_indices = resolve_serve_gpu_indices(&request.gpu_indices)?;
    let runtime = resolve_vllm_runtime(request.runtime_id.as_deref())?;
    let mut child = spawn_vllm_server(&request, &runtime, request.log_path.as_deref())?;
    write_running_state(&request, &runtime, child.id())?;

    // Wait for the server to become ready, with comprehensive error logging
    if let Err(e) = wait_for_vllm_ready(
        &mut child,
        &request.host,
        request.port,
        &request.model_ref,
        vllm_ready_timeout(),
        request.log_path.as_deref(),
    ) {
        // Terminate the whole vLLM process tree so the EngineCore worker (which
        // holds the GPU allocation) does not survive and leak device memory.
        let _ = rocm_core::terminate_process_tree(child.id());
        let _ = child.wait();
        write_terminal_state(&request.state_path, "failed")?;
        return Err(e);
    }

    let status = child.wait().context("failed waiting for vLLM server")?;
    write_terminal_state(
        &request.state_path,
        if status.success() {
            "stopped"
        } else {
            "failed"
        },
    )?;
    if status.success() {
        Ok(())
    } else {
        std::process::exit(status.code().unwrap_or(1));
    }
}

fn spawn_vllm_server(
    request: &ServeHttpRequest,
    runtime: &VllmRuntime,
    log_path: Option<&Path>,
) -> Result<std::process::Child> {
    require_nonempty(&request.service_id, "service_id")?;
    require_nonempty(&request.model_ref, "model_ref")?;
    if !matches!(request.device_policy, DevicePolicy::GpuRequired) {
        bail!("vLLM launch requires ROCm GPU execution; no CPU fallback is used");
    }

    // Fail fast when the OpenMPI runtime is missing. vLLM's ROCm torch wheel
    // dlopen()s the OpenMPI libraries during `import torch`; without them the
    // process dies with the cryptic `libmpi_cxx.so.40: cannot open shared object
    // file` error from deep inside torch. Surface a clear, actionable message
    // here instead so the user knows exactly what to install.
    if !cfg!(windows) && !rocm_core::openmpi::detect_openmpi().present {
        bail!(
            "vLLM requires the OpenMPI runtime (libmpi.so / libmpi_cxx.so and mpirun), which was not found; \
without it `import torch` fails with `libmpi_cxx.so.40: cannot open shared object file`. \
{}, or run `rocm engines install vllm --yes` to install it automatically, then retry.",
            rocm_core::openmpi::install_hint()
        );
    }

    // Fail fast when the libatomic runtime is missing. PyTorch's ROCm wheel links
    // `libatomic.so.1`, so `import torch` dies with `libatomic.so.1: cannot open
    // shared object file` on minimal hosts (notably RHEL UBI, which ships only
    // GCC's libatomic.so linker script). Unlike libnuma it is not bundled by the
    // ROCm SDK, so it must be installed from the system package manager.
    if !cfg!(windows) && !rocm_core::openmpi::libatomic_present() {
        bail!(
            "vLLM requires the libatomic runtime (libatomic.so.1), which was not found; \
without it `import torch` fails with `libatomic.so.1: cannot open shared object file`. \
{}, or run `rocm engines install vllm --yes` to install it automatically, then retry.",
            rocm_core::openmpi::libatomic_install_hint()
        );
    }

    // Fail fast when the real numactl runtime is missing. PyTorch's `libc10.so`
    // binds `libnuma.so.1`'s `libnuma_1.2` symbol version. The ROCm SDK bundles
    // numa only under a renamed soname with rewritten symbol versions, so it
    // cannot satisfy that binding and `import torch` dies with
    // `libnuma.so.1: ... version 'libnuma_1.2' not found`. The upstream numactl
    // runtime must be installed from the system package manager.
    if !cfg!(windows) && !rocm_core::openmpi::libnuma_present() {
        bail!(
            "vLLM requires the system numactl runtime (libnuma.so.1 with the libnuma_1.2 symbols), which was not found; \
the ROCm SDK's bundled numa uses renamed symbol versions and cannot satisfy it, so `import torch` fails with \
`libnuma.so.1: version 'libnuma_1.2' not found`. \
{}, or run `rocm engines install vllm --yes` to install it automatically, then retry.",
            rocm_core::openmpi::libnuma_install_hint()
        );
    }

    if let Some(parent) = request.state_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    if let Some(log_path) = log_path
        && let Some(parent) = log_path.parent()
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let mut command = ProcessCommand::new(&runtime.command);
    command
        .args(vllm_serve_args(
            &request.model_ref,
            &request.host,
            request.port,
            vllm_enforce_eager_enabled(),
            request.engine_recipe.as_ref(),
        ))
        .stdin(Stdio::null());
    // When `rocm serve` protects a public endpoint, the key arrives via the
    // environment / key file (never argv). Hand it to vLLM as `VLLM_API_KEY` so
    // its OpenAI server rejects unauthenticated requests; passing it through the
    // env keeps it out of the process table.
    if let Some(api_key) = rocm_engine_protocol::resolve_endpoint_api_key() {
        command.env("VLLM_API_KEY", api_key);
    }
    apply_therock_env(&mut command, runtime)?;
    rocm_engine_protocol::apply_gpu_visibility(&mut command, &request.gpu_indices);
    if let Some(log_path) = log_path {
        let log = fs::File::create(log_path)
            .with_context(|| format!("failed to create {}", log_path.display()))?;
        command.stdout(Stdio::from(
            log.try_clone().context("failed to clone log handle")?,
        ));
        command.stderr(Stdio::from(log));
    }

    command.spawn().map_err(|error| {
        let base = anyhow::Error::new(error).context(format!(
            "failed to spawn vLLM command {}",
            runtime.command.display()
        ));
        match stale_interpreter_hint(&runtime.command) {
            Some(hint) => base.context(hint),
            None => base,
        }
    })
}

/// Explain a spawn that failed on a script whose `#!` interpreter is gone.
///
/// The kernel reports the missing *interpreter* as ENOENT against the *script*,
/// so the raw error names a file that is plainly there. A virtualenv whose folder
/// was recorded under one path and now lives at another produces exactly this: the
/// entry points still exist, and not one of them can start.
///
/// Returns `None` whenever the ordinary reading is the right one — a genuinely
/// absent file, a binary, or a shebang whose interpreter is present — so this only
/// ever speaks up when it has something to add.
fn stale_interpreter_hint(command: &Path) -> Option<String> {
    if !command.is_file() {
        return None;
    }
    let head = fs::read(command).ok()?;
    let first_line = head.split(|byte| *byte == b'\n').next()?;
    let text = String::from_utf8_lossy(first_line);
    let shebang = text.strip_prefix("#!")?.trim();
    // `#!/usr/bin/env python` names the launcher, not the interpreter; the path
    // that goes stale is the direct one a venv writes.
    let interpreter = shebang.split_whitespace().next()?;
    if Path::new(interpreter).is_file() {
        return None;
    }
    Some(format!(
        "`{}` is present, but the interpreter on its `#!` line is not: {interpreter}. \
         Its environment was recorded at a folder that is no longer there, so none of \
         its entry points can start. Reinstall it with `rocm install sdk`.",
        command.display()
    ))
}

fn healthcheck_service(request: HealthcheckRequest) -> Result<HealthcheckResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let files = service_files(&request.service_id)?;
    let state = read_service_state(&files.state_path).ok();
    let endpoint_url = state.as_ref().and_then(endpoint_url_from_state);
    let model_ref = state
        .as_ref()
        .and_then(|value| value_string(value, "model_ref"));
    let listed = endpoint_url
        .as_deref()
        .map(|endpoint| query_loaded_model_endpoint(endpoint, model_ref.as_deref()))
        .transpose()
        .unwrap_or(None)
        .unwrap_or(false);
    // `/v1/models` lists a model as soon as the server accepts its name, which can
    // be minutes before the weights are resident. Confirm inference once before
    // reporting ready.
    let ready = listed
        && endpoint_url.as_deref().is_some_and(|endpoint| {
            inference_verified(
                &files.state_path,
                state.as_ref(),
                endpoint,
                model_ref.as_deref().unwrap_or_default(),
            )
        });
    let state_status = state
        .as_ref()
        .and_then(|value| value_string(value, "status"))
        .unwrap_or_else(|| "unknown".to_owned());
    let device = if state.is_some() {
        "rocm_gpu"
    } else {
        "unknown"
    };
    Ok(HealthcheckResponse::for_readiness(
        listed,
        ready,
        &state_status,
        device,
    ))
}

fn endpoint_response(request: EndpointRequest) -> Result<EndpointResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let files = service_files(&request.service_id)?;
    let state = read_service_state(&files.state_path)
        .with_context(|| format!("service state not found for `{}`", request.service_id))?;
    let endpoint_url = endpoint_url_from_state(&state)
        .with_context(|| format!("service `{}` has no endpoint URL", request.service_id))?;
    Ok(EndpointResponse {
        endpoint_url,
        api_style: "openai".to_owned(),
        supported_routes: vec![
            "/health".to_owned(),
            "/v1/models".to_owned(),
            "/v1/chat/completions".to_owned(),
            "/v1/completions".to_owned(),
        ],
    })
}

fn logs_response(request: LogsRequest) -> Result<LogsResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let files = service_files(&request.service_id)?;
    let limit = request.tail_lines.unwrap_or(DEFAULT_LOG_TAIL_LINES);
    Ok(LogsResponse {
        log_path: files.log_path.display().to_string(),
        recent_lines: if files.log_path.is_file() {
            tail_lines(&files.log_path, limit)?
        } else {
            Vec::new()
        },
    })
}

fn stop_service(request: StopRequest) -> Result<StopResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let files = service_files(&request.service_id)?;
    let state = read_service_state(&files.state_path).ok();
    let outcome = state
        .as_ref()
        .and_then(identity_from_state)
        // vLLM spawns an `EngineCore` worker that pins the GPU allocation, so
        // the whole process tree must be signalled — but only after the recorded
        // PID is confirmed to still be our server, never a recycled stranger.
        .map(|identity| {
            rocm_core::terminate_verified(&identity, STOP_SCOPE, STOP_GRACE, request.force)
        });
    let (stopped, graceful) = match outcome {
        Some(outcome) => (outcome.stopped(), outcome.graceful()),
        // No PID recorded: nothing we can confirm stopping.
        None => (false, false),
    };
    if stopped {
        write_terminal_state(&files.state_path, "stopped")?;
    }
    Ok(StopResponse { stopped, graceful })
}

fn resolve_vllm_runtime(runtime_id: Option<&str>) -> Result<VllmRuntime> {
    if cfg!(windows) {
        bail!("{}", windows_unsupported_message());
    }

    if let Some(command) = std::env::var_os("ROCM_CLI_VLLM_COMMAND")
        .or_else(|| std::env::var_os("VLLM_COMMAND"))
        .map(PathBuf::from)
    {
        let command = resolve_command_path(&command)?;
        return Ok(VllmRuntime {
            runtime_id: runtime_id.unwrap_or("external-vllm").to_owned(),
            env_id: "external-vllm-command".to_owned(),
            command,
            python_executable: None,
            version: None,
            source: "environment command".to_owned(),
            sdk_root: None,
            sdk_bin: None,
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
        });
    }

    if let Some(python) = std::env::var_os("ROCM_CLI_VLLM_PYTHON")
        .or_else(|| std::env::var_os("VLLM_PYTHON"))
        .map(PathBuf::from)
        .filter(|path| path.is_file())
    {
        return runtime_from_python(
            python,
            runtime_id.unwrap_or("external-vllm-python"),
            "environment python",
            None,
            None,
            Vec::new(),
            Vec::new(),
        );
    }

    if let Some(runtime) = resolve_managed_runtime(runtime_id)? {
        return Ok(runtime);
    }

    if let Some(command) = find_command_on_path("vllm") {
        return Ok(VllmRuntime {
            runtime_id: runtime_id.unwrap_or("external-vllm-path").to_owned(),
            env_id: "external-vllm-path".to_owned(),
            command,
            python_executable: None,
            version: None,
            source: "PATH".to_owned(),
            sdk_root: None,
            sdk_bin: None,
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
        });
    }

    let base = "vLLM is not installed in a Linux/WSL ROCm Python environment. Install/build vLLM against a ROCm-capable Python environment, then set ROCM_CLI_VLLM_COMMAND, set ROCM_CLI_VLLM_PYTHON, or install it into the active rocm-cli TheRock runtime. Native Windows is skipped; no CPU fallback is used.";
    match describe_skipped_managed_runtimes(runtime_id) {
        Some(skipped) => bail!("{base}\n\n{skipped}"),
        None => bail!("{base}"),
    }
}

fn runtime_from_python(
    python: PathBuf,
    runtime_id: &str,
    source: &str,
    sdk_root: Option<PathBuf>,
    sdk_bin: Option<PathBuf>,
    sdk_bin_paths: Vec<PathBuf>,
    sdk_library_paths: Vec<PathBuf>,
) -> Result<VllmRuntime> {
    let command = vllm_command_from_python(&python)
        .with_context(|| format!("vLLM command not found beside {}", python.display()))?;
    let version = probe_vllm_version(&python).ok().flatten();
    Ok(VllmRuntime {
        runtime_id: runtime_id.to_owned(),
        env_id: format!("external-vllm-{}", stable_id_component(runtime_id)),
        command,
        python_executable: Some(python),
        version,
        source: source.to_owned(),
        sdk_root,
        sdk_bin,
        sdk_bin_paths,
        sdk_library_paths,
    })
}

fn install_vllm_with_uv(python: &Path, reinstall: bool) -> Result<()> {
    let paths = AppPaths::discover()?;
    let uv = ensure_uv_binary(&paths).context("failed to acquire uv binary for vLLM install")?;
    let VllmInstallTarget {
        index_url,
        requirement,
    } = vllm_install_target()?;
    let mut args = uv_pip_install_base(python);
    // Without `--reinstall`, `uv pip install vllm` is a no-op when the wheel is
    // already present, which would silently turn a requested reinstall into a
    // no-op. Force the reinstall so the caller's intent is honored.
    if reinstall {
        args.push("--reinstall".to_owned());
    }
    args.push(requirement.clone());
    args.push("--extra-index-url".to_owned());
    args.push(index_url.clone());
    let output = ProcessCommand::new(&uv)
        .args(args)
        .envs(uv_command_env(&paths))
        .output()
        .context("failed to launch uv pip install for vLLM")?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let detail = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        "no output".to_owned()
    };
    bail!(
        "`uv pip install {} --extra-index-url {}` failed for {}: {}",
        requirement,
        index_url,
        python.display(),
        detail
    )
}

fn resolve_managed_runtime(runtime_id: Option<&str>) -> Result<Option<VllmRuntime>> {
    let candidates = collect_managed_runtime_candidates(runtime_id)?;
    for candidate in candidates {
        if let Ok(runtime) = runtime_from_python(
            candidate.python_executable,
            &candidate.runtime_id,
            &candidate.source,
            candidate.sdk_root,
            candidate.sdk_bin,
            candidate.sdk_bin_paths,
            candidate.sdk_library_paths,
        ) {
            return Ok(Some(runtime));
        }
    }
    Ok(None)
}

fn resolve_managed_runtime_python(
    runtime_id: Option<&str>,
) -> Result<Option<ManagedRuntimePython>> {
    let candidates = collect_managed_runtime_candidates(runtime_id)?;
    let Some(candidate) = candidates.into_iter().next() else {
        return Ok(None);
    };
    Ok(Some(ManagedRuntimePython {
        runtime_id: candidate.runtime_id,
        python_executable: candidate.python_executable,
    }))
}

#[derive(Debug, Clone)]
struct ManagedRuntimeCandidate {
    runtime_id: String,
    source: String,
    python_executable: PathBuf,
    sdk_root: Option<PathBuf>,
    sdk_bin: Option<PathBuf>,
    sdk_bin_paths: Vec<PathBuf>,
    sdk_library_paths: Vec<PathBuf>,
}

/// The runtime manifests matching `runtime_id`, most recently installed first.
fn load_runtime_manifests(runtime_id: Option<&str>) -> Result<Vec<TheRockRuntimeManifest>> {
    let paths = AppPaths::discover()?;
    let registry = paths.data_dir.join("runtimes").join("registry");
    if !registry.is_dir() {
        return Ok(Vec::new());
    }
    let mut manifests = Vec::new();
    for entry in
        fs::read_dir(&registry).with_context(|| format!("failed to read {}", registry.display()))?
    {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        let Ok(manifest) = serde_json::from_slice::<TheRockRuntimeManifest>(&bytes) else {
            continue;
        };
        if !runtime_matches(&manifest, runtime_id) {
            continue;
        }
        manifests.push((manifest.installed_at_unix_ms.unwrap_or(0), manifest));
    }
    manifests.sort_by_key(|(installed_at, _)| std::cmp::Reverse(*installed_at));
    Ok(manifests
        .into_iter()
        .map(|(_, manifest)| manifest)
        .collect())
}

/// The torch build the SDK installed into this runtime, as its manifest records it.
///
/// Read from the manifest, never from the environment. By the time this runs the
/// environment may already hold some other installer's build, and taking that for
/// the SDK's would conclude the runtime is correct and leave it wrong for good.
fn recorded_sdk_torch_build(runtime_id: &str) -> Option<String> {
    let manifest = load_runtime_manifests(Some(runtime_id))
        .ok()?
        .into_iter()
        .next()?;
    sdk_torch_build_from_manifest(&manifest)
}

/// The SDK's torch build identifier, with the fallback for older manifests.
///
/// Manifests written before `sdk_torch` was recorded still name the SDK version, and
/// TheRock builds that into the local segment as `rocm<version>`. Mirrors the CLI's
/// `sdk_torch_build_for_key` so both sides agree on what "the SDK's build" means.
fn sdk_torch_build_from_manifest(manifest: &TheRockRuntimeManifest) -> Option<String> {
    if let Some(recorded) = manifest.sdk_torch.as_deref()
        && let Some(build) = split_local_version(recorded).1
    {
        return Some(build.to_owned());
    }
    let version = manifest
        .rocm_sdk
        .as_ref()
        .and_then(|probe| probe.rocm_sdk_version.clone())
        .or_else(|| manifest.version.clone())?;
    (!version.trim().is_empty()).then(|| format!("rocm{version}"))
}

/// Registered runtimes that matched the request but were passed over because the
/// interpreter they record is not there, phrased for the end of an error message.
///
/// [`collect_managed_runtime_candidates`] drops those silently, which is right for
/// resolution — a runtime that cannot run is not a candidate — but leaves the
/// failure describing a registry that looks empty while `rocm runtimes list`
/// happily prints the entry. Naming the manifest and the interpreter turns
/// "nothing resolved" into something actionable.
///
/// Best-effort: this only ever decorates an error that is already being returned,
/// so any problem reading the registry yields no note rather than replacing the
/// original failure.
fn describe_skipped_managed_runtimes(runtime_id: Option<&str>) -> Option<String> {
    let paths = AppPaths::discover().ok()?;
    let registry = paths.data_dir.join("runtimes").join("registry");
    let mut skipped = Vec::new();
    for entry in fs::read_dir(&registry).ok()? {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Ok(bytes) = fs::read(&path) else { continue };
        let Ok(manifest) = serde_json::from_slice::<TheRockRuntimeManifest>(&bytes) else {
            continue;
        };
        if !runtime_matches(&manifest, runtime_id) {
            continue;
        }
        let Some(python) = manifest.python_executable.as_ref() else {
            continue;
        };
        if python.is_file() {
            continue;
        }
        let key = manifest.runtime_key.as_deref().unwrap_or("<unnamed>");
        skipped.push(format!("  {key}: {} is missing", python.display()));
    }
    if skipped.is_empty() {
        return None;
    }
    skipped.sort();
    Some(format!(
        "These registered runtimes were skipped because the Python interpreter they \
         record is not there:\n{}\nReinstall one with `rocm install sdk`, or drop it with \
         `rocm runtimes uninstall <runtime_key>`.",
        skipped.join("\n")
    ))
}

fn collect_managed_runtime_candidates(
    runtime_id: Option<&str>,
) -> Result<Vec<ManagedRuntimeCandidate>> {
    let mut candidates = Vec::new();
    for manifest in load_runtime_manifests(runtime_id)? {
        let Some(python) = manifest
            .python_executable
            .clone()
            .filter(|path| path.is_file())
        else {
            continue;
        };
        let runtime_id = manifest
            .runtime_id
            .as_deref()
            .unwrap_or("therock-vllm-runtime")
            .to_owned();
        let source = manifest.runtime_key.as_deref().map_or_else(
            || "managed_runtime_manifest".to_owned(),
            |key| format!("managed_runtime_manifest:{key}"),
        );
        let (sdk_root, sdk_bin, sdk_bin_paths, sdk_library_paths) = manifest
            .rocm_sdk
            .as_ref()
            .filter(|probe| probe.import_ok)
            .map_or((None, None, Vec::new(), Vec::new()), |probe| {
                (
                    probe.root_path.clone(),
                    probe.bin_path.clone(),
                    probe.bin_paths.clone(),
                    probe.library_paths.clone(),
                )
            });
        candidates.push(ManagedRuntimeCandidate {
            runtime_id,
            source,
            python_executable: python,
            sdk_root,
            sdk_bin,
            sdk_bin_paths,
            sdk_library_paths,
        });
    }
    Ok(candidates)
}

fn runtime_matches(manifest: &TheRockRuntimeManifest, requested: Option<&str>) -> bool {
    let Some(requested) = requested.map(str::trim).filter(|value| !value.is_empty()) else {
        return true;
    };
    let requested = requested.to_ascii_lowercase();
    if requested == "external" || requested == "external-vllm" {
        return false;
    }
    for candidate in [
        manifest.runtime_id.as_deref(),
        manifest.runtime_key.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        let candidate = candidate.to_ascii_lowercase();
        if candidate == requested || candidate.starts_with(&requested) {
            return true;
        }
    }
    false
}

/// Resolve and validate the GPU ordinal before spawning vLLM. An authoritative
/// empty probe result fails under the adapter's GPU-only policy; `auto` selects
/// the first visible device. Unknown availability remains permissive so WSL and
/// unsupported probe surfaces are not blocked.
fn resolve_serve_gpu_indices(requested: &[u32]) -> Result<Vec<u32>> {
    resolve_gpu_indices_against(requested, rocm_core::usable_amd_gpu_indices())
}

fn resolve_gpu_indices_against(requested: &[u32], usable: Option<Vec<u32>>) -> Result<Vec<u32>> {
    let Some(usable) = usable else {
        return Ok(requested.to_vec());
    };
    if usable.is_empty() {
        bail!(
            "no usable AMD GPU detected; vLLM requires ROCm GPU execution and does not fall back \
             to CPU. Check the driver with `rocm examine` and ensure HIP_VISIBLE_DEVICES / \
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

fn normalize_vllm_device_policy(policy: Option<DevicePolicy>) -> Result<DevicePolicy> {
    match policy.unwrap_or(DevicePolicy::GpuRequired) {
        DevicePolicy::GpuRequired => Ok(DevicePolicy::GpuRequired),
        DevicePolicy::GpuPreferred => Ok(DevicePolicy::GpuRequired),
        DevicePolicy::CpuOnly => {
            bail!("vLLM adapter is ROCm GPU-only in rocm-cli; no CPU fallback is used")
        }
    }
}

fn parse_device_policy_arg(policy: Option<&str>) -> Result<DevicePolicy> {
    match policy.unwrap_or("gpu_required") {
        "gpu" | "gpu_required" => Ok(DevicePolicy::GpuRequired),
        "gpu_preferred" => Ok(DevicePolicy::GpuPreferred),
        "cpu" | "cpu_only" => Ok(DevicePolicy::CpuOnly),
        other => bail!("unsupported device policy: {other}"),
    }
}

/// Parse a `--gpu` CLI value into an optional `GpuSelection` for `LaunchRequest`.
fn parse_gpu_selection_arg(value: Option<&str>) -> Result<Option<GpuSelection>> {
    value
        .map(|raw| GpuSelection::parse_cli_value(raw).map_err(anyhow::Error::msg))
        .transpose()
}

/// Parse a `--gpu` CLI value into explicit device ordinals (empty for `auto`).
fn parse_gpu_indices_arg(value: Option<&str>) -> Result<Vec<u32>> {
    Ok(rocm_engine_protocol::launch_gpu_indices(
        parse_gpu_selection_arg(value)?.as_ref(),
    ))
}

fn vllm_command_from_python(python: &Path) -> Option<PathBuf> {
    let dir = python.parent()?;
    candidate_command_names("vllm")
        .into_iter()
        .map(|name| dir.join(name))
        .find(|path| path.is_file())
}

fn find_command_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for candidate in candidate_command_names(name) {
            let path = dir.join(candidate);
            if path.is_file() {
                return Some(path);
            }
        }
    }
    None
}

fn resolve_command_path(command: &Path) -> Result<PathBuf> {
    if command.components().count() > 1 || command.is_absolute() {
        if command.is_file() {
            return Ok(command.to_path_buf());
        }
        bail!(
            "configured vLLM command is not a file: {}",
            command.display()
        );
    }
    find_command_on_path(&command.display().to_string()).with_context(|| {
        format!(
            "configured vLLM command `{}` was not found on PATH",
            command.display()
        )
    })
}

fn candidate_command_names(name: &str) -> Vec<String> {
    if cfg!(windows) {
        vec![
            format!("{name}.exe"),
            format!("{name}.cmd"),
            name.to_owned(),
        ]
    } else {
        vec![name.to_owned()]
    }
}

fn probe_vllm_version(python: &Path) -> Result<Option<String>> {
    let script = r#"import importlib.metadata, importlib.util, json
spec = importlib.util.find_spec("vllm")
version = None
if spec is not None:
    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
print(json.dumps({"present": spec is not None, "version": version}))
"#;
    let output = ProcessCommand::new(python)
        .arg("-c")
        .arg(script)
        .output()
        .with_context(|| format!("failed to probe vLLM with {}", python.display()))?;
    if !output.status.success() {
        bail!(
            "vLLM probe failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let value: Value = serde_json::from_slice(&output.stdout).context("invalid vLLM probe JSON")?;
    if value
        .get("present")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        Ok(value
            .get("version")
            .and_then(Value::as_str)
            .map(str::to_owned))
    } else {
        bail!("Python environment does not contain the vLLM package")
    }
}

fn apply_therock_env(command: &mut ProcessCommand, runtime: &VllmRuntime) -> Result<()> {
    command.env("VLLM_TARGET_DEVICE", "rocm");
    let Some(root) = runtime.sdk_root.as_ref() else {
        return Ok(());
    };
    let bin = runtime.sdk_bin.as_ref();
    command
        .env("ROCM_SDK_ROOT", root)
        .env("ROCM_PATH", root)
        .env("ROCM_HOME", root)
        .env("HIP_PATH", root)
        .env("ROCM_CLI_THEROCK_RUNTIME_ID", &runtime.runtime_id);
    if let Some(bin) = bin {
        command.env("ROCM_CLI_THEROCK_SDK_BIN", bin).env(
            "PATH",
            prepend_path_entries(&runtime_bin_paths(runtime), std::env::var_os("PATH"))?,
        );
    } else if !runtime.sdk_bin_paths.is_empty() {
        command.env(
            "PATH",
            prepend_path_entries(&runtime_bin_paths(runtime), std::env::var_os("PATH"))?,
        );
    }
    if !cfg!(windows) {
        command.env(
            "LD_LIBRARY_PATH",
            prepend_path_entries(
                &therock_library_path_entries(runtime),
                std::env::var_os("LD_LIBRARY_PATH"),
            )?,
        );
    }
    Ok(())
}

/// Full argument vector for `vllm serve`, as spawned by [`spawn_vllm_server`].
///
/// Kept as a pure function so the exact argv can be asserted in tests: memory
/// and graph-mode flags materially change how much VRAM vLLM claims, and a
/// wiring mistake there is invisible until a live GPU launch.
///
/// Notably absent: `--gpu-memory-utilization`. rocm-cli deliberately does not
/// supply a default — vLLM's own default applies unless the user asks for a
/// specific fraction via `rocm serve --gpu-memory-utilization`, which arrives
/// here through the engine recipe's `required_flags`.
fn vllm_serve_args(
    model_ref: &str,
    host: &str,
    port: u16,
    enforce_eager: bool,
    engine_recipe: Option<&EngineRecipeHint>,
) -> Vec<String> {
    let mut args = vec![
        "serve".to_owned(),
        model_ref.to_owned(),
        "--host".to_owned(),
        host.to_owned(),
        "--port".to_owned(),
        port.to_string(),
    ];
    // vLLM's FULL CUDA-graph replay hangs ROCm gfx94x GPUs on the first decode
    // (surfaces as `HW Exception ... reason :GPU Hang`, which kills the engine and
    // drops every inference request). Eager mode disables CUDA graphs and keeps
    // inference stable. Allow opting back in via env once a runtime ships a fix.
    if enforce_eager {
        args.push("--enforce-eager".to_owned());
    }
    args.extend(engine_recipe_launch_args(engine_recipe));
    args
}

fn engine_recipe_launch_args(engine_recipe: Option<&EngineRecipeHint>) -> Vec<String> {
    engine_recipe
        .map(|hint| hint.required_flags.clone())
        .unwrap_or_default()
}

/// Whether to launch vLLM with `--enforce-eager` (CUDA graphs disabled).
///
/// Defaults to enabled because FULL CUDA-graph replay hangs ROCm gfx94x GPUs
/// during decode. Set `ROCM_CLI_VLLM_ENFORCE_EAGER` to `0`/`false`/`no`/`off`
/// to re-enable CUDA graphs on runtimes where the hang is fixed.
fn vllm_enforce_eager_enabled() -> bool {
    std::env::var("ROCM_CLI_VLLM_ENFORCE_EAGER")
        .ok()
        .is_none_or(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
}

/// Wheel index and exact requirement for one `uv pip install vllm`.
///
/// The two are always derived from the same `<version>+<abi>` build, so the
/// install can never be pinned to something the index does not serve — and can
/// never be left unpinned, which would let the resolver fall through to PyPI's
/// non-ROCm build.
#[derive(Debug, Clone, PartialEq, Eq)]
struct VllmInstallTarget {
    index_url: String,
    requirement: String,
}

/// Pure resolution of the ROCm wheel index from an optional override value.
///
/// Falls back to [`VLLM_ROCM_EXTRA_INDEX_URL`] when the override is absent or
/// blank; a usable one is trimmed. The caller supplies the value (see
/// [`vllm_install_target`] for the environment read) so this stays testable.
fn resolve_vllm_rocm_extra_index_url(override_value: Option<String>) -> String {
    override_value
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| VLLM_ROCM_EXTRA_INDEX_URL.to_owned())
}

/// Index URL and requirement for the current environment.
fn vllm_install_target() -> Result<VllmInstallTarget> {
    resolve_vllm_install_target(std::env::var("ROCM_CLI_VLLM_ROCM_INDEX_URL").ok())
}

/// Resolve the wheel index and the exact requirement to install from it.
///
/// Without an override, the built-in [`VLLM_ROCM_EXTRA_INDEX_URL`] and
/// [`VLLM_PINNED_SPEC`] are used. With `ROCM_CLI_VLLM_ROCM_INDEX_URL` set, the
/// build is recovered from the URL when it has the published
/// `{VLLM_ROCM_INDEX_PREFIX}/<version>/<abi>` shape — so pointing at another
/// release of the same index works *and* stays pinned, with no extra
/// configuration.
///
/// A URL of any other shape cannot be pinned automatically, and dropping the
/// pin is not an option: the install passes `--extra-index-url`, so PyPI stays
/// in play and a bare `vllm` can silently resolve to the non-ROCm build. Such a
/// URL is therefore rejected rather than installed unpinned.
fn resolve_vllm_install_target(index_override: Option<String>) -> Result<VllmInstallTarget> {
    let index_url = resolve_vllm_rocm_extra_index_url(index_override);
    if index_url == VLLM_ROCM_EXTRA_INDEX_URL {
        return Ok(VllmInstallTarget {
            index_url,
            requirement: VLLM_PINNED_SPEC.to_owned(),
        });
    }

    let (version, abi) = vllm_rocm_build_from_index_url(&index_url).ok_or_else(|| {
        anyhow!(
            "cannot determine which vLLM build `{index_url}` serves: it is not a published \
             release index of the form `{VLLM_ROCM_INDEX_PREFIX}/<version>/<abi>`. Installing an \
             unpinned `vllm` instead would let the resolver fall back to PyPI's non-ROCm build, \
             so point ROCM_CLI_VLLM_ROCM_INDEX_URL at a release index, or unset it to use the \
             built-in one."
        )
    })?;

    Ok(VllmInstallTarget {
        index_url,
        requirement: format!("vllm=={version}+{abi}"),
    })
}

/// Recover the `<version>+<abi>` build a published release index serves.
///
/// Returns `None` for anything that is not
/// `{VLLM_ROCM_INDEX_PREFIX}/<version>/<abi>`, including the rolling top-level
/// index, which is latest-only and therefore not a pin.
fn vllm_rocm_build_from_index_url(index_url: &str) -> Option<(String, String)> {
    let valid = |segment: &str| {
        !segment.is_empty()
            && segment
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
    };
    let rest = index_url
        .trim_end_matches('/')
        .strip_prefix(VLLM_ROCM_INDEX_PREFIX)?
        .strip_prefix('/')?;
    let (version, abi) = rest.split_once('/')?;
    (valid(version) && valid(abi)).then(|| (version.to_owned(), abi.to_owned()))
}

/// Time to wait for vLLM to become ready before terminating the process tree.
///
/// Defaults to [`DEFAULT_VLLM_READY_TIMEOUT`]. A valid-but-slow cold start
/// (large weight download, first-decode compile) can exceed the default, so the
/// timeout is configurable via `ROCM_CLI_VLLM_READY_TIMEOUT_SECS` (a positive
/// integer number of seconds).
fn vllm_ready_timeout() -> Duration {
    resolve_vllm_ready_timeout(std::env::var("ROCM_CLI_VLLM_READY_TIMEOUT_SECS").ok())
}

fn resolve_vllm_ready_timeout(override_value: Option<String>) -> Duration {
    override_value
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|secs| *secs > 0)
        .map_or(DEFAULT_VLLM_READY_TIMEOUT, Duration::from_secs)
}

fn runtime_bin_paths(runtime: &VllmRuntime) -> Vec<PathBuf> {
    let mut entries = Vec::new();
    if let Some(bin) = runtime.sdk_bin.as_ref() {
        entries.push(bin.clone());
    }
    entries.extend(runtime.sdk_bin_paths.iter().cloned());
    dedupe_paths(entries)
}

fn therock_library_path_entries(runtime: &VllmRuntime) -> Vec<PathBuf> {
    let Some(root) = runtime.sdk_root.as_ref() else {
        return dedupe_paths(runtime.sdk_library_paths.clone());
    };
    let mut entries = runtime.sdk_library_paths.clone();
    entries.extend([
        root.join("lib"),
        root.join("lib64"),
        root.join("lib").join("rocm_sysdeps").join("lib"),
    ]);
    if cfg!(target_os = "linux") {
        let wsl_dxcore_lib = PathBuf::from("/usr/lib/wsl/lib");
        if wsl_dxcore_lib.is_dir() {
            entries.push(wsl_dxcore_lib);
        }
        // OpenMPI is installed outside the default loader path on some distros
        // (notably RHEL-family under /usr/lib64/openmpi/lib); make sure vLLM can
        // load libmpi.so at launch when it lives there.
        entries.extend(rocm_core::openmpi::openmpi_library_dirs());

        if let Some(compat_dir) = runtime_compat_dir(runtime) {
            // PyTorch's `libtorch_global_deps.so` lists `libmpi_cxx.so.40` as a
            // NEEDED dependency, but OpenMPI 5.x removed the legacy C++ bindings,
            // so `import torch` aborts with `libmpi_cxx.so.40: cannot open shared
            // object file`. When no real `libmpi_cxx.so*` exists, materialize an
            // embedded `libmpi_cxx.so.40` stub (built at compile time, see
            // rocm-core's build.rs) into a runtime-owned directory and add it to
            // the loader path. The stub only *defines* the legacy C++ binding
            // symbols torch needs; they are never called in single-node serving,
            // so this is safe.
            if let Some(dir) = rocm_core::openmpi::ensure_mpi_cxx_compat(&compat_dir) {
                entries.push(dir);
            }
            // PyTorch's `libc10.so` NEEDS the standard `libnuma.so.1` soname with
            // the upstream `libnuma_1.2` symbol version. TheRock bundles numa only
            // under the renamed soname `librocm_sysdeps_numa.so.1` whose versions
            // are rewritten to `AMDROCM_SYSDEPS_1.0_libnuma_*`, which cannot
            // satisfy that binding. An older rocm-cli release symlinked
            // `libnuma.so.1` to that bundled library; on the loader path it
            // shadowed any real system libnuma and broke `import torch` with
            // `version 'libnuma_1.2' not found`. Remove that stale shim here so
            // the real numactl runtime (installed via the package manager) wins;
            // `libnuma_present()`/`spawn_vllm_server` handle the install/preflight.
            remove_stale_numa_shim(&compat_dir);
        }
    }
    dedupe_paths(entries)
}

/// Remove a stale `libnuma.so.1` compatibility symlink left by older rocm-cli
/// versions in `compat_dir`. That shim pointed at the ROCm SDK's bundled
/// `librocm_sysdeps_numa.so.1`, whose renamed symbol versions cannot satisfy the
/// `libnuma_1.2` symbol PyTorch's `libc10.so` binds; leaving it on the loader
/// path would shadow a correctly installed system libnuma. No-op when absent or
/// when the entry is not a symlink.
fn remove_stale_numa_shim(compat_dir: &Path) {
    let link = compat_dir.join("libnuma.so.1");
    if let Ok(meta) = link.symlink_metadata()
        && meta.file_type().is_symlink()
    {
        let _ = fs::remove_file(&link);
    }
}

/// A writable, runtime-owned directory for managed-runtime library compatibility
/// shims (see [`therock_library_path_entries`]). Prefers the managed Python
/// environment root (`<env>/bin/python` -> `<env>`); falls back to the SDK root
/// when no Python launcher is recorded.
fn runtime_compat_dir(runtime: &VllmRuntime) -> Option<PathBuf> {
    const COMPAT_DIR_NAME: &str = "rocm-cli-lib-compat";
    if let Some(env_root) = runtime
        .python_executable
        .as_ref()
        .and_then(|python| python.parent())
        .and_then(|bin| bin.parent())
    {
        return Some(env_root.join(COMPAT_DIR_NAME));
    }
    runtime
        .sdk_root
        .as_ref()
        .map(|root| root.join(COMPAT_DIR_NAME))
}

fn dedupe_paths(entries: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut deduped = Vec::new();
    for entry in entries {
        if !entry.as_os_str().is_empty() && !deduped.iter().any(|seen| seen == &entry) {
            deduped.push(entry);
        }
    }
    deduped
}

fn prepend_path_entries(entries: &[PathBuf], current: Option<OsString>) -> Result<OsString> {
    let mut parts = Vec::new();
    for entry in entries {
        if !entry.as_os_str().is_empty() && !parts.iter().any(|part: &PathBuf| part == entry) {
            parts.push(entry.clone());
        }
    }
    if let Some(current) = current {
        for entry in std::env::split_paths(&current) {
            if !entry.as_os_str().is_empty() && !parts.iter().any(|part| part == &entry) {
                parts.push(entry);
            }
        }
    }
    std::env::join_paths(parts).context("failed to compose runtime path")
}

fn service_files(service_id: &str) -> Result<ServiceFiles> {
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

fn write_running_state(request: &ServeHttpRequest, runtime: &VllmRuntime, pid: u32) -> Result<()> {
    write_state(
        &request.state_path,
        &json!({
            "service_id": request.service_id,
            "engine": ENGINE_NAME,
            "status": "running",
            "pid": pid,
            "model_ref": request.model_ref,
            "host": request.host,
            "port": request.port,
            "endpoint_url": endpoint_url(&request.host, request.port),
            "device_policy": "gpu_required",
            "runtime_id": runtime.runtime_id,
            "requested_runtime_id": request.runtime_id,
            "env_id": request.env_id.as_deref().unwrap_or(runtime.env_id.as_str()),
            "runtime_executable": runtime.command,
            "server_pid": pid,
            "engine_recipe": request.engine_recipe,
            "engine_recipe_required_flags": engine_recipe_launch_args(request.engine_recipe.as_ref()),
            "therock_runtime_env": therock_runtime_env_state(runtime),
            "started_at_unix_ms": current_unix_millis(),
            // Kernel start-time of the launcher PID, captured while it is alive.
            // Paired with `pid`, it identifies this exact process across PID
            // recycling so a later stop never signals a reused PID.
            "start_ticks": rocm_core::process_start_ticks(pid)
        }),
    )
}

fn therock_runtime_env_state(runtime: &VllmRuntime) -> Option<Value> {
    let root = runtime.sdk_root.as_ref()?;
    Some(json!({
        "runtime_id": runtime.runtime_id,
        "env_id": runtime.env_id,
        "root": root.display().to_string(),
        "bin": runtime.sdk_bin.as_ref().map(|path| path.display().to_string()),
        "bin_paths": runtime_bin_paths(runtime)
            .into_iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>(),
        "library_paths": therock_library_path_entries(runtime)
            .into_iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>(),
        "source": runtime.source,
    }))
}

fn write_terminal_state(state_path: &Path, status: &str) -> Result<()> {
    let mut state = read_service_state(state_path).unwrap_or_else(|_| json!({}));
    if let Some(object) = state.as_object_mut() {
        object.insert("status".to_owned(), Value::String(status.to_owned()));
        object.insert(
            "stopped_at_unix_ms".to_owned(),
            Value::from(current_unix_millis() as u64),
        );
    }
    write_state(state_path, &state)
}

fn read_service_state(path: &Path) -> Result<Value> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
}

fn write_state(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(
        path,
        serde_json::to_vec_pretty(value).context("failed to serialize vLLM state")?,
    )
    .with_context(|| format!("failed to write {}", path.display()))
}

fn endpoint_url(host: &str, port: u16) -> String {
    format!("{}/v1", format_http_base_url(host, port))
}

fn endpoint_url_from_state(state: &Value) -> Option<String> {
    value_string(state, "endpoint_url").or_else(|| {
        let host = value_string(state, "host")?;
        let port = state.get("port")?.as_u64()?;
        let port = u16::try_from(port).ok()?;
        Some(endpoint_url(&host, port))
    })
}

fn query_loaded_model_endpoint(endpoint_url: &str, model_ref: Option<&str>) -> Result<bool> {
    // Send the endpoint key when the server is protected so the healthcheck does
    // not read a 401 as "not ready" and kill a correctly-authenticated server.
    let endpoint_api_key = rocm_engine_protocol::resolve_endpoint_api_key();
    openai_models_endpoint_has_model(
        endpoint_url,
        model_ref,
        endpoint_api_key.as_deref(),
        Duration::from_millis(HEALTHCHECK_TIMEOUT_MS),
    )
}

/// Whether the endpoint can actually complete a chat request, as opposed to
/// merely listing the model.
fn query_inference_probe_endpoint(endpoint_url: &str, model_ref: &str) -> Result<bool> {
    if model_ref.trim().is_empty() {
        return Ok(false);
    }
    // Send the endpoint key for the same reason the models query does: a 401 from
    // a correctly-protected server must not read as "cannot serve".
    let endpoint_api_key = rocm_engine_protocol::resolve_endpoint_api_key();
    rocm_core::openai_chat_completion_probe(
        endpoint_url,
        model_ref,
        endpoint_api_key.as_deref(),
        rocm_core::INFERENCE_PROBE_TIMEOUT,
    )
}

/// Whether a real inference request has succeeded against this service.
///
/// Latch and backoff bookkeeping lives in `rocm-core` so both engines share one
/// implementation — what counts as *listed* differs per engine, what counts as
/// *serving* does not.
fn inference_verified(
    state_path: &Path,
    state: Option<&Value>,
    endpoint_url: &str,
    model_ref: &str,
) -> bool {
    rocm_core::engine_state_inference_verified(
        state_path,
        state,
        endpoint_url,
        model_ref,
        rocm_engine_protocol::resolve_endpoint_api_key().as_deref(),
    )
}

fn pid_from_state(state: &Value) -> Option<u32> {
    state
        .get("pid")?
        .as_u64()
        .and_then(|pid| pid.try_into().ok())
}

/// Reconstruct the recorded process identity (PID + kernel start-time) from a
/// service state file. `start_ticks` is absent in state files written before
/// this field existed, in which case verification degrades to best-effort.
fn identity_from_state(state: &Value) -> Option<rocm_core::ProcessIdentity> {
    let pid = pid_from_state(state)?;
    let start_ticks = state.get("start_ticks").and_then(Value::as_u64);
    Some(rocm_core::ProcessIdentity::new(pid, start_ticks))
}

fn value_string(value: &Value, key: &str) -> Option<String> {
    value.get(key)?.as_str().map(str::to_owned)
}

fn stable_id_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

fn runtime_lock_hash(runtime: &VllmRuntime) -> String {
    let mut hasher = DefaultHasher::new();
    runtime.runtime_id.hash(&mut hasher);
    runtime.command.hash(&mut hasher);
    runtime.version.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn current_unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

const fn windows_unsupported_message() -> &'static str {
    "vLLM ROCm serving is supported by rocm-cli only on Linux/WSL; native Windows vLLM is skipped. No CPU fallback is used."
}

fn read_request() -> Result<EngineRequestEnvelope> {
    let mut buffer = String::new();
    std::io::stdin()
        .read_to_string(&mut buffer)
        .context("failed to read stdin for engine request")?;
    serde_json::from_str(&buffer).context("failed to parse engine request envelope")
}

fn print_json<T: Serialize>(value: &T) -> Result<()> {
    let stdout = std::io::stdout();
    let mut handle = stdout.lock();
    serde_json::to_writer_pretty(&mut handle, value)?;
    writeln!(&mut handle)?;
    Ok(())
}

/// Builds a human-readable summary of the tail of the startup log, if available.
/// Returns an empty string when no log is present or it cannot be read.
fn startup_log_context(log_path: Option<&Path>) -> String {
    let summary = log_path
        .and_then(|p| summarize_startup_log_tail(p, STARTUP_FAILURE_LOG_TAIL_LINES).ok())
        .unwrap_or_default();
    if summary.is_empty() {
        return String::new();
    }
    let hint = oom_utilization_hint(&summary);
    format!("\n\nLast {STARTUP_FAILURE_LOG_TAIL_LINES} lines of startup log:\n{summary}{hint}")
}

/// Appends [`rocm_core::VLLM_GPU_MEMORY_UTILIZATION_HINT`] when the startup-log
/// tail shows an out-of-memory failure, and nothing otherwise so healthy
/// failures are not cluttered with an irrelevant memory hint. Shares its wording
/// with the `rocm` CLI's pre-launch low-VRAM note so both surfaces point the
/// user at the same fix rather than drifting into different phrasing.
fn oom_utilization_hint(log_tail: &str) -> String {
    if !log_tail_shows_oom(log_tail) {
        return String::new();
    }
    // Route the user's *actual* failing line into the `--symptom` example
    // rather than a canned string, prefixed with a `vllm:` anchor so the
    // diagnose checker attributes it to this failure mode.
    let symptom_line = log_tail
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty() && log_tail_shows_oom(line))
        .unwrap_or("out of memory");
    format!(
        "\n\nDetected an out-of-memory failure. {}\n\
         For conditional remediation, run `rocm diagnose --symptom 'vllm: {symptom_line}'`.",
        rocm_core::VLLM_GPU_MEMORY_UTILIZATION_HINT
    )
}

/// Case-insensitive scan for the out-of-memory signatures vLLM/PyTorch emit on a
/// HIP allocation failure (e.g. `torch.OutOfMemoryError: HIP out of memory`).
fn log_tail_shows_oom(log_tail: &str) -> bool {
    let lower = log_tail.to_ascii_lowercase();
    lower.contains("out of memory") || lower.contains("outofmemory")
}

/// Polls the vLLM endpoint until it reports the model is loaded, or times out.
/// Uses a monotonic clock (`Instant`) so wall-clock adjustments cannot corrupt the
/// timeout, and surfaces an early process exit immediately instead of waiting out
/// the full readiness window.
fn wait_for_vllm_ready(
    child: &mut std::process::Child,
    host: &str,
    port: u16,
    model_ref: &str,
    timeout: Duration,
    log_path: Option<&Path>,
) -> Result<()> {
    let start = Instant::now();
    let endpoint = format!("http://{host}:{port}");
    let poll_interval = Duration::from_millis(500);

    loop {
        // Surface an early process exit (bad model ref, missing deps, etc.)
        // immediately instead of waiting out the full readiness timeout.
        if let Some(status) = child
            .try_wait()
            .context("failed to poll vLLM server process status")?
        {
            let log_context = startup_log_context(log_path);
            bail!(
                "vLLM server process exited before becoming ready (status: {status}){log_context}"
            );
        }

        if start.elapsed() > timeout {
            let log_context = startup_log_context(log_path);
            bail!(
                "vLLM server at {host}:{port} failed to become ready within {} seconds{log_context}",
                timeout.as_secs()
            );
        }

        // Listing the model is not enough to hand the endpoint to a caller: vLLM
        // advertises it well before the first request can be served. Only return
        // once inference has actually answered.
        match query_loaded_model_endpoint(&endpoint, Some(model_ref)) {
            Ok(true) if query_inference_probe_endpoint(&endpoint, model_ref).unwrap_or(false) => {
                return Ok(());
            }
            _ => std::thread::sleep(poll_interval),
        }
    }
}

/// Reads the last N lines from a log file and returns them as a formatted string.
/// Handles large files by seeking to near the end and reading backwards.
fn summarize_startup_log_tail(log_path: &Path, limit: usize) -> Result<String> {
    let lines = tail_lines(log_path, limit)?;
    if lines.is_empty() {
        return Ok(String::new());
    }
    Ok(lines.join("\n"))
}

/// Reads the last N lines from a file efficiently by seeking.
/// For files larger than MAX_TAIL_READ, seeks to MAX_TAIL_READ from the end.
fn tail_lines(path: &Path, limit: usize) -> Result<Vec<String>> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("failed to open log file {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to read metadata for {}", path.display()))?;
    let file_size = metadata.len();

    // For large files, seek to MAX_TAIL_READ from the end. When the seek lands in
    // the middle of a line, the first line read back is a partial fragment that
    // must be dropped. When it lands exactly on a line boundary (the byte before
    // `seek_pos` is a newline) the first line is complete and must be kept.
    let mut first_line_is_partial = false;
    if file_size > MAX_TAIL_READ {
        let seek_pos = file_size - MAX_TAIL_READ;
        // Probe the byte preceding `seek_pos` to classify the first line, then
        // leave the cursor at `seek_pos` for the buffered read below.
        file.seek(SeekFrom::Start(seek_pos - 1))
            .with_context(|| format!("failed to seek in log file {}", path.display()))?;
        let mut probe = [0u8; 1];
        file.read_exact(&mut probe)
            .with_context(|| format!("failed to read from log file {}", path.display()))?;
        first_line_is_partial = probe[0] != b'\n';
    }

    let buffered = BufReader::new(file);
    let mut lines: Vec<String> = buffered
        .lines()
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| format!("failed to read lines from {}", path.display()))?;

    // Drop the leading partial line produced by seeking into the middle of a line.
    if first_line_is_partial && !lines.is_empty() {
        lines.remove(0);
    }

    // Return only the last `limit` lines
    let start_idx = if lines.len() > limit {
        lines.len() - limit
    } else {
        0
    };
    Ok(lines.into_iter().skip(start_idx).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A throwaway directory for the shebang cases. Scoped per test and per
    /// thread so the suite can keep running these concurrently.
    fn shebang_scratch(name: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!(
            "vllm-shebang-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::remove_dir_all(&root).ok();
        fs::create_dir_all(&root).expect("create scratch dir");
        root
    }

    #[test]
    fn a_dead_shebang_interpreter_is_named() {
        // The reported confusion: ENOENT against a script that is plainly there,
        // because the kernel reports the missing interpreter against the script.
        let root = shebang_scratch("dead");
        let script = root.join("vllm");
        fs::write(&script, "#!/gone/bin/python\nprint()\n").unwrap();

        let hint = stale_interpreter_hint(&script).expect("a dead interpreter must be named");

        assert!(hint.contains("/gone/bin/python"), "{hint}");
        assert!(hint.contains("is present"), "{hint}");
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_live_shebang_interpreter_gets_no_hint() {
        // Nothing to add: the ordinary reading of the error is the right one.
        let root = shebang_scratch("live");
        let interpreter = root.join("python");
        fs::write(&interpreter, "").unwrap();
        let script = root.join("vllm");
        fs::write(&script, format!("#!{}\n", interpreter.display())).unwrap();

        assert!(stale_interpreter_hint(&script).is_none());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_genuinely_missing_command_gets_no_hint() {
        let root = shebang_scratch("absent");
        assert!(stale_interpreter_hint(&root.join("not-there")).is_none());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_command_with_no_shebang_gets_no_hint() {
        // A real binary. Reading its first bytes must not produce a hint.
        let root = shebang_scratch("binary");
        let binary = root.join("vllm");
        fs::write(&binary, [0x7f, b'E', b'L', b'F', 0x02, 0x01, 0x01, 0x00]).unwrap();

        assert!(stale_interpreter_hint(&binary).is_none());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn shebang_arguments_do_not_hide_the_interpreter() {
        // `#!/path/python -X foo` names the interpreter first; the flags are not
        // part of the path being checked.
        let root = shebang_scratch("args");
        let script = root.join("vllm");
        fs::write(&script, "#!/gone/bin/python -X utf8\n").unwrap();

        let hint = stale_interpreter_hint(&script).expect("interpreter must still be found");

        assert!(hint.contains("/gone/bin/python"), "{hint}");
        assert!(
            !hint.contains("-X"),
            "the flags are not part of the path: {hint}"
        );
        fs::remove_dir_all(&root).ok();
    }

    /// Answer `count` chat requests on a loopback port with the given status,
    /// reporting how many arrived.
    fn spawn_chat_endpoint(
        status_line: &'static str,
        count: usize,
    ) -> (u16, std::thread::JoinHandle<usize>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind loopback");
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

    fn probe_state_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "rocm-vllm-probe-{tag}-{}-{}.json",
            std::process::id(),
            current_unix_millis()
        ))
    }

    #[test]
    fn inference_verification_latches_into_the_state_file() -> Result<()> {
        // First check probes and records the verdict; the second reads the latch
        // and leaves the model alone.
        let (port, server) = spawn_chat_endpoint("HTTP/1.1 200 OK", 1);
        let state_path = probe_state_path("latch");
        write_state(&state_path, &json!({"status": "running"}))?;
        let endpoint = endpoint_url("127.0.0.1", port);

        let state = read_service_state(&state_path).ok();
        assert!(inference_verified(
            &state_path,
            state.as_ref(),
            &endpoint,
            "facebook/opt-125m"
        ));

        let state = read_service_state(&state_path)?;
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
            "facebook/opt-125m"
        ));

        assert_eq!(
            server.join().expect("server thread"),
            1,
            "the latched check must not send a second inference request"
        );
        fs::remove_file(&state_path).ok();
        Ok(())
    }

    #[test]
    fn inference_verification_withheld_while_the_model_is_still_loading() -> Result<()> {
        // The reported failure: `/v1/models` answers but inference does not.
        let (port, server) = spawn_chat_endpoint("HTTP/1.1 503 Service Unavailable", 1);
        let state_path = probe_state_path("loading");
        write_state(&state_path, &json!({"status": "running"}))?;
        let endpoint = endpoint_url("127.0.0.1", port);

        let state = read_service_state(&state_path).ok();
        assert!(!inference_verified(
            &state_path,
            state.as_ref(),
            &endpoint,
            "facebook/opt-125m"
        ));
        assert!(
            read_service_state(&state_path)?
                .get(rocm_core::INFERENCE_VERIFIED_STATE_KEY)
                .is_none(),
            "nothing is latched until inference actually answers"
        );

        server.join().expect("server thread");
        fs::remove_file(&state_path).ok();
        Ok(())
    }

    fn test_engine_recipe(engine: &str, contract_version: &str) -> EngineRecipeHint {
        EngineRecipeHint {
            contract_version: contract_version.to_owned(),
            engine: engine.to_owned(),
            required_flags: vec!["--enable-auto-tool-choice".to_owned()],
            parser_settings: std::collections::BTreeMap::default(),
            preferred_endpoint: None,
            unsupported_combinations: Vec::new(),
            notes: vec!["test recipe".to_owned()],
        }
    }

    #[test]
    fn parse_gpu_args_map_to_indices() {
        assert_eq!(parse_gpu_indices_arg(None).unwrap(), Vec::<u32>::new());
        assert_eq!(
            parse_gpu_indices_arg(Some("auto")).unwrap(),
            Vec::<u32>::new()
        );
        assert_eq!(parse_gpu_indices_arg(Some("2")).unwrap(), vec![2]);
        assert!(parse_gpu_selection_arg(Some("nope")).is_err());
        assert!(parse_gpu_selection_arg(Some("0,1")).is_err());
    }

    #[test]
    fn gpu_required_launch_rejects_no_usable_device() {
        let error = resolve_gpu_indices_against(&[], Some(Vec::new()))
            .expect_err("zero usable devices must be rejected");
        assert!(error.to_string().contains("no usable AMD GPU"));
    }

    #[test]
    fn gpu_required_launch_selects_and_validates_visible_devices() {
        assert_eq!(
            resolve_gpu_indices_against(&[], Some(vec![1, 2])).unwrap(),
            vec![1]
        );
        assert_eq!(
            resolve_gpu_indices_against(&[2], Some(vec![1, 2])).unwrap(),
            vec![2]
        );
        let error = resolve_gpu_indices_against(&[0], Some(vec![1, 2]))
            .expect_err("masked device must be rejected");
        assert!(error.to_string().contains("not available"));
    }

    #[test]
    fn gpu_required_launch_allows_unprobeable_hosts() {
        assert_eq!(
            resolve_gpu_indices_against(&[], None).unwrap(),
            Vec::<u32>::new()
        );
        assert_eq!(resolve_gpu_indices_against(&[3], None).unwrap(), vec![3]);
    }

    #[test]
    fn cpu_policy_is_rejected_without_fallback() {
        let error = normalize_vllm_device_policy(Some(DevicePolicy::CpuOnly))
            .expect_err("vLLM CPU policy must fail");
        assert!(error.to_string().contains("no CPU fallback is used"));
    }

    #[test]
    fn gpu_preferred_resolves_to_gpu_required() -> Result<()> {
        assert_eq!(
            normalize_vllm_device_policy(Some(DevicePolicy::GpuPreferred))?,
            DevicePolicy::GpuRequired
        );
        Ok(())
    }

    #[test]
    fn engine_recipe_launch_args_forward_required_flags() {
        let hint = test_engine_recipe(ENGINE_NAME, ENGINE_RECIPE_CONTRACT_VERSION);

        assert_eq!(
            engine_recipe_launch_args(Some(&hint)),
            vec!["--enable-auto-tool-choice".to_owned()]
        );
    }

    #[test]
    fn managed_env_reflects_managed_runtime_manifest_source() {
        let mut runtime = VllmRuntime {
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            env_id: "external-vllm-therock".to_owned(),
            command: PathBuf::from(if cfg!(windows) {
                r"C:\venv\Scripts\vllm.exe"
            } else {
                "/home/user/.venv/bin/vllm"
            }),
            python_executable: None,
            version: Some("test".to_owned()),
            source: "managed_runtime_manifest:vllm-source-pip-gfx120x-all".to_owned(),
            sdk_root: None,
            sdk_bin: None,
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
        };

        assert!(runtime_is_managed(&runtime));
        assert!(
            vllm_runtime_warnings(&runtime)
                .iter()
                .any(|warning| warning.contains("managed TheRock runtime"))
        );

        runtime.source = "environment command".to_owned();
        assert!(!runtime_is_managed(&runtime));
        assert!(
            vllm_runtime_warnings(&runtime)
                .iter()
                .any(|warning| warning.contains("external vLLM runtime"))
        );
    }

    #[test]
    fn endpoint_response_errors_without_service_state() {
        let error = endpoint_response(EndpointRequest {
            service_id: format!("missing-{}", current_unix_millis()),
        })
        .expect_err("missing service state should not produce a default endpoint");

        assert!(error.to_string().contains("service state not found"));
    }

    #[test]
    fn endpoint_url_falls_back_to_host_and_port() {
        let state = json!({
            "host": "127.0.0.1",
            "port": 12345
        });
        assert_eq!(
            endpoint_url_from_state(&state),
            Some("http://127.0.0.1:12345/v1".to_owned())
        );
        let ipv6_state = json!({
            "host": "::1",
            "port": 12345
        });
        assert_eq!(
            endpoint_url_from_state(&ipv6_state),
            Some("http://[::1]:12345/v1".to_owned())
        );
    }

    #[test]
    fn resolve_model_omits_gpu_memory_utilization_default() -> Result<()> {
        let response = resolve_model_response(ResolveModelRequest {
            model_ref: "facebook/opt-125m".to_owned(),
            runtime_id: None,
            device_policy: Some(DevicePolicy::GpuRequired),
            recipe_override: None,
            engine_recipe: None,
        })?;

        assert!(
            response
                .launch_defaults
                .get("gpu_memory_utilization")
                .is_none(),
            "rocm-cli defers to vLLM's own default, so it must not advertise one: {}",
            response.launch_defaults
        );
        Ok(())
    }

    #[test]
    fn oom_utilization_hint_fires_on_out_of_memory_log_tails() {
        // The exact PyTorch/HIP signature from the field report.
        let torch = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";
        assert!(log_tail_shows_oom(torch));
        let hint = oom_utilization_hint(torch);
        assert!(
            hint.contains("--gpu-memory-utilization"),
            "an OOM tail must surface the utilization workaround: {hint}"
        );
        assert!(hint.contains("--gpu <index>"));
        assert!(
            hint.contains("rocm diagnose --symptom"),
            "an OOM tail must route the user to the conditional catalog entry: {hint}"
        );
        assert!(
            hint.contains(
                "vllm: torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB."
            ),
            "the user's real failing line must be routed into --symptom, not a canned string: {hint}"
        );

        // Detection is case-insensitive and also matches the spaced phrasing.
        assert!(log_tail_shows_oom("HIP OUT OF MEMORY"));
        assert!(log_tail_shows_oom("RuntimeError: CUDA out of memory"));
    }

    #[test]
    fn oom_utilization_hint_stays_quiet_for_unrelated_failures() {
        let unrelated = "ValueError: model architecture 'FooForCausalLM' is not supported";
        assert!(!log_tail_shows_oom(unrelated));
        assert!(
            oom_utilization_hint(unrelated).is_empty(),
            "non-OOM failures must not carry a memory hint"
        );
    }

    #[test]
    fn vllm_serve_args_omit_gpu_memory_utilization_without_override() {
        let args = vllm_serve_args("facebook/opt-125m", "127.0.0.1", 8000, false, None);

        assert_eq!(
            args,
            vec![
                "serve",
                "facebook/opt-125m",
                "--host",
                "127.0.0.1",
                "--port",
                "8000"
            ]
        );
        assert!(
            !args.iter().any(|arg| arg == "--gpu-memory-utilization"),
            "vLLM must apply its own default when the user asked for nothing: {args:?}"
        );
    }

    #[test]
    fn vllm_serve_args_forward_recipe_gpu_memory_utilization() {
        let hint = EngineRecipeHint {
            contract_version: ENGINE_RECIPE_CONTRACT_VERSION.to_owned(),
            engine: ENGINE_NAME.to_owned(),
            required_flags: vec!["--gpu-memory-utilization".to_owned(), "0.35".to_owned()],
            ..EngineRecipeHint::default()
        };

        let args = vllm_serve_args("facebook/opt-125m", "127.0.0.1", 8000, true, Some(&hint));

        let position = args
            .iter()
            .position(|arg| arg == "--gpu-memory-utilization")
            .expect("explicit utilization must reach the spawned argv");
        assert_eq!(args.get(position + 1).map(String::as_str), Some("0.35"));
        assert!(args.contains(&"--enforce-eager".to_owned()));
    }

    #[test]
    fn resolve_model_echoes_matching_engine_recipe() -> Result<()> {
        let hint = test_engine_recipe(ENGINE_NAME, ENGINE_RECIPE_CONTRACT_VERSION);
        let response = resolve_model_response(ResolveModelRequest {
            model_ref: "facebook/opt-125m".to_owned(),
            runtime_id: None,
            device_policy: Some(DevicePolicy::GpuRequired),
            recipe_override: None,
            engine_recipe: Some(hint.clone()),
        })?;

        assert_eq!(response.engine_recipe, Some(hint));
        Ok(())
    }

    #[test]
    fn resolve_model_rejects_mismatched_engine_recipe() {
        let error = resolve_model_response(ResolveModelRequest {
            model_ref: "facebook/opt-125m".to_owned(),
            runtime_id: None,
            device_policy: Some(DevicePolicy::GpuRequired),
            recipe_override: None,
            engine_recipe: Some(test_engine_recipe(
                "lemonade",
                ENGINE_RECIPE_CONTRACT_VERSION,
            )),
        })
        .expect_err("mismatched engine recipe should fail");

        assert!(error.to_string().contains("does not match adapter"));
    }

    #[test]
    fn resolve_model_rejects_unsupported_engine_recipe_contract() {
        let error = resolve_model_response(ResolveModelRequest {
            model_ref: "facebook/opt-125m".to_owned(),
            runtime_id: None,
            device_policy: Some(DevicePolicy::GpuRequired),
            recipe_override: None,
            engine_recipe: Some(test_engine_recipe(ENGINE_NAME, "999.0.0")),
        })
        .expect_err("unsupported recipe contract should fail");

        assert!(error.to_string().contains("unsupported"));
    }

    #[test]
    fn tail_lines_returns_suffix() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "rocm-vllm-tail-{}-{}.log",
            std::process::id(),
            current_unix_millis()
        ));
        fs::write(&path, "a\nb\nc\n")?;
        let lines = tail_lines(&path, 2)?;
        fs::remove_file(path).ok();
        assert_eq!(lines, vec!["b".to_owned(), "c".to_owned()]);
        Ok(())
    }

    #[test]
    fn tail_lines_keeps_first_line_when_seek_lands_on_boundary() -> Result<()> {
        // Build a file where the MAX_TAIL_READ window starts exactly on a line
        // boundary: a prefix ending in '\n', followed by exactly MAX_TAIL_READ
        // bytes of complete lines. The first windowed line must NOT be dropped.
        let prefix = format!("{}\n", "p".repeat(63));
        let mut tail = String::from("FIRSTLINE\n");
        while tail.len() + 2 <= MAX_TAIL_READ as usize {
            tail.push_str("y\n");
        }
        while tail.len() < MAX_TAIL_READ as usize {
            tail.push('z');
        }
        assert_eq!(tail.len(), MAX_TAIL_READ as usize);

        let path = std::env::temp_dir().join(format!(
            "rocm-vllm-tail-boundary-{}-{}.log",
            std::process::id(),
            current_unix_millis()
        ));
        fs::write(&path, format!("{prefix}{tail}"))?;
        let lines = tail_lines(&path, usize::MAX)?;
        fs::remove_file(&path).ok();

        assert_eq!(
            lines.first().map(String::as_str),
            Some("FIRSTLINE"),
            "complete first line must be preserved when the seek lands on a newline boundary"
        );
        assert!(
            !lines.iter().any(|line| line.contains('p')),
            "bytes before the tail window must not appear"
        );
        Ok(())
    }

    #[test]
    fn tail_lines_drops_partial_first_line_when_seek_lands_midline() -> Result<()> {
        // The window starts in the middle of a line, so the leading fragment is
        // partial and must be dropped.
        let prefix = "p".repeat(64);
        let mut tail = String::from("PARTIALFRAGMENT");
        tail.push('\n');
        tail.push_str("SECONDLINE\n");
        while tail.len() < MAX_TAIL_READ as usize {
            tail.push_str("y\n");
        }
        // Trim back to exactly MAX_TAIL_READ bytes so the window starts mid-line.
        tail.truncate(MAX_TAIL_READ as usize);

        let path = std::env::temp_dir().join(format!(
            "rocm-vllm-tail-midline-{}-{}.log",
            std::process::id(),
            current_unix_millis()
        ));
        fs::write(&path, format!("{prefix}{tail}"))?;
        let lines = tail_lines(&path, usize::MAX)?;
        fs::remove_file(&path).ok();

        assert_eq!(
            lines.first().map(String::as_str),
            Some("SECONDLINE"),
            "partial leading fragment must be dropped when the seek lands mid-line"
        );
        Ok(())
    }

    #[test]
    fn vllm_ready_timeout_uses_default_without_override() {
        assert_eq!(resolve_vllm_ready_timeout(None), DEFAULT_VLLM_READY_TIMEOUT);
    }

    #[test]
    fn vllm_ready_timeout_honors_positive_override() {
        assert_eq!(
            resolve_vllm_ready_timeout(Some(" 900 ".to_owned())),
            Duration::from_mins(15)
        );
    }

    #[test]
    fn vllm_ready_timeout_ignores_invalid_or_zero_override() {
        assert_eq!(
            resolve_vllm_ready_timeout(Some("0".to_owned())),
            DEFAULT_VLLM_READY_TIMEOUT
        );
        assert_eq!(
            resolve_vllm_ready_timeout(Some("not-a-number".to_owned())),
            DEFAULT_VLLM_READY_TIMEOUT
        );
    }

    fn install_target(index: Option<&str>) -> Result<VllmInstallTarget> {
        resolve_vllm_install_target(index.map(ToOwned::to_owned))
    }

    fn violation(requiring: &str, detail: &str) -> DependencyViolation {
        DependencyViolation {
            requiring: requiring.to_owned(),
            detail: detail.to_owned(),
        }
    }

    #[test]
    fn a_repair_targets_the_environment_that_was_assessed() {
        // `resolve_managed_runtime_python` returns the first candidate unconditionally,
        // while the assessment walked on to the candidate that actually has vLLM. Both
        // match when `runtime_id` matching is by prefix, so the repair must follow the
        // assessed environment or it fixes an interpreter nobody found broken.
        let assessed = VllmRuntime {
            runtime_id: "nightly-wheel-gfx94x-dcgpu-7-14-0a20260611".to_owned(),
            env_id: "external-vllm-therock".to_owned(),
            command: PathBuf::from("/rocm/runtimes/wheel/nightly-gfx94x/bin/vllm"),
            python_executable: Some(PathBuf::from(
                "/rocm/runtimes/wheel/nightly-gfx94x/bin/python",
            )),
            version: Some("0.26.0".to_owned()),
            source: "managed_runtime_manifest:nightly-wheel-gfx94x-dcgpu".to_owned(),
            sdk_root: None,
            sdk_bin: None,
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
        };

        let target = assessed_python_for_repair(&assessed).expect("a managed runtime has a python");

        assert_eq!(
            target.python_executable,
            PathBuf::from("/rocm/runtimes/wheel/nightly-gfx94x/bin/python")
        );
        assert_eq!(
            target.runtime_id, "nightly-wheel-gfx94x-dcgpu-7-14-0a20260611",
            "the assessed runtime's own id, not the prefix the caller asked for"
        );
    }

    /// The build identifier the SDK recorded in the runtimes used by these tests.
    const SDK_BUILD: &str = "rocm7.13.0";

    /// `repair_from_violations`'s opt-out argument, named at the call sites so a bare
    /// `false`/`true` does not have to be decoded against the signature.
    const ALIGNED: bool = false;
    const OPTED_OUT: bool = true;

    #[test]
    fn a_consistent_environment_is_not_reinstalled() {
        // The other settled state: the SDK published no build of the release vLLM
        // pins, so the runtime kept the engine's own build and the exact pin is
        // satisfied. `uv pip check` reports nothing at all, and the recorded SDK
        // build must not manufacture a finding out of that silence.
        assert_eq!(
            repair_from_violations(&[], Some(SDK_BUILD), ALIGNED),
            RepairAssessment::default()
        );
    }

    #[test]
    fn a_torch_of_the_wrong_release_still_forces_a_reinstall() {
        // The other direction of the same problem: the SDK installed a torch
        // *release* the engine does not accept. The build is the SDK's, but the
        // release is not the engine's, so this is not the intended divergence and
        // the reinstall that restores the engine's release must still happen.
        let assessment = repair_from_violations(
            &[violation(
                "vllm",
                "The package `vllm` requires `torch==2.10.0+git8514f05`, but `2.9.1+rocm7.14.0a20260611` is installed",
            )],
            Some("rocm7.14.0a20260611"),
            ALIGNED,
        );

        assert!(assessment.needed);
        assert!(
            assessment
                .notes
                .iter()
                .any(|note| note.contains("2.10.0+git8514f05")),
            "the reported note names the pin that was violated: {:?}",
            assessment.notes
        );
    }

    #[test]
    fn the_intended_torch_divergence_alone_does_not_force_a_reinstall() {
        // The steady state this change exists to stop churning. rocm-cli put the
        // SDK's build of the release vLLM pins back after the engine install; the
        // engine's exact pin cannot express that, so `uv pip check` reports it
        // forever. Reinstalling would replace it with the build that opens no
        // device, and the next invocation would do the whole thing again.
        let assessment = repair_from_violations(
            &[violation(
                "vllm",
                "The package `vllm` requires `torch==2.11.0+gitd0c8b1f`, but `2.11.0+rocm7.13.0` is installed",
            )],
            Some(SDK_BUILD),
            ALIGNED,
        );

        assert!(
            !assessment.needed,
            "the intended divergence must not trigger a reinstall: {:?}",
            assessment.notes
        );
        assert!(
            assessment
                .notes
                .iter()
                .all(|note| !note.contains("was reinstalled")),
            "no note may claim a repair that did not happen: {:?}",
            assessment.notes
        );
    }

    #[test]
    fn a_torch_from_neither_side_still_forces_a_reinstall() {
        // Same release the engine pins, but a build belonging to neither the SDK nor
        // the engine — someone installed a torch by hand, or a resolver picked one
        // off PyPI. Nothing about that is intended.
        let assessment = repair_from_violations(
            &[violation(
                "vllm",
                "The package `vllm` requires `torch==2.11.0+gitd0c8b1f`, but `2.11.0+cpu` is installed",
            )],
            Some(SDK_BUILD),
            ALIGNED,
        );

        assert!(assessment.needed);
    }

    #[test]
    fn an_unidentified_sdk_build_keeps_the_previous_behaviour() {
        // Without a recorded build there is no way to tell the intended divergence
        // from a defect, and guessing in the permissive direction would leave a
        // genuinely broken runtime alone. Fall back to repairing.
        let assessment = repair_from_violations(
            &[violation(
                "vllm",
                "The package `vllm` requires `torch==2.11.0+gitd0c8b1f`, but `2.11.0+rocm7.13.0` is installed",
            )],
            None,
            ALIGNED,
        );

        assert!(assessment.needed);
        assert!(
            assessment
                .notes
                .iter()
                .any(|note| note.contains("rocm install sdk")),
            "the SDK-overwrite hint belongs to exactly this un-identifiable case: {:?}",
            assessment.notes
        );
    }

    #[test]
    fn the_whole_replaced_torch_stack_is_reported_one_finding_per_line() {
        // What the failure looks like on hardware right after `rocm install sdk`:
        // the SDK moves torch, torchvision and torchaudio together, so all three
        // pins are violated at once. Joining them into a single note produced one
        // ~380-character line; each finding gets its own so a terminal can show it.
        //
        // Only torch is realigned, so only torch's divergence is intended. The other
        // two are genuine and still drive the reinstall — which is what restores all
        // three to the engine's builds before rocm-cli puts torch back.
        let assessment = repair_from_violations(
            &[
                violation(
                    "vllm",
                    "The package `vllm` requires `torch==2.11.0+gitd0c8b1f`, but `2.11.0+rocm7.13.0` is installed",
                ),
                violation(
                    "vllm",
                    "The package `vllm` requires `torchvision==0.24.1+d801a34`, but `0.26.0+rocm7.13.0` is installed",
                ),
                violation(
                    "vllm",
                    "The package `vllm` requires `torchaudio==2.9.0+eaa9e4e`, but `2.11.0+rocm7.13.0` is installed",
                ),
            ],
            Some(SDK_BUILD),
            ALIGNED,
        );

        assert!(assessment.needed);
        let violation_notes: Vec<&String> = assessment
            .notes
            .iter()
            .filter(|note| note.starts_with("violation: "))
            .collect();
        assert_eq!(
            violation_notes.len(),
            2,
            "every genuinely violated pin gets its own note: {:?}",
            assessment.notes
        );
        for package in ["torchvision==", "torchaudio=="] {
            assert!(
                violation_notes.iter().any(|note| note.contains(package)),
                "{package} is missing from the reported notes: {:?}",
                assessment.notes
            );
        }
        assert!(
            violation_notes
                .iter()
                .all(|note| !note.contains("torch==2.11.0")),
            "the realigned torch is a divergence, not a violation: {:?}",
            assessment.notes
        );
        assert!(
            assessment
                .notes
                .iter()
                .any(|note| note.starts_with("expected divergence: ")
                    && note.contains("torch==2.11.0+gitd0c8b1f")),
            "the intended divergence is still named, so the reader is not left \
             thinking the reinstall was about torch: {:?}",
            assessment.notes
        );
        assert!(
            assessment.notes.iter().all(|note| note.len() < 200),
            "no note should be a wall of joined findings: {:?}",
            assessment.notes
        );
    }

    #[test]
    fn an_sdk_built_torchvision_is_still_a_violation() {
        // The SDK writes the whole torch stack, so torchvision can carry the same
        // build as torch and — when the releases happen to line up — look exactly
        // like the intended divergence. rocm-cli realigns torch and nothing else, so
        // this is a real violation the engine must repair. The stack test above
        // cannot catch a regression here: its torchvision release differs too, so
        // the release check alone would still reject it.
        let assessment = repair_from_violations(
            &[violation(
                "vllm",
                "The package `vllm` requires `torchvision==0.24.1+d801a34`, but `0.24.1+rocm7.13.0` is installed",
            )],
            Some(SDK_BUILD),
            ALIGNED,
        );

        assert!(
            assessment.needed,
            "only torch is realigned; another package at the SDK's build is a genuine violation: {:?}",
            assessment.notes
        );
    }

    #[test]
    fn unrelated_upstream_conflicts_do_not_force_a_reinstall() {
        // These environments routinely carry conflicts between third-party packages.
        // Reinstalling vLLM would not resolve them, so they must not trigger one.
        let assessment = repair_from_violations(
            &[
                violation(
                    "tilelang",
                    "The package `tilelang` requires `cloudpickle>=3.0`, but `2.2.1` is installed",
                ),
                violation(
                    "torch",
                    "The package `torch` requires `sympy>=1.13`, but `1.12` is installed",
                ),
            ],
            Some(SDK_BUILD),
            ALIGNED,
        );

        assert_eq!(assessment, RepairAssessment::default());
    }

    #[test]
    fn an_opted_out_custom_torch_alone_does_not_force_a_reinstall() {
        // The runtime the opt-out exists to produce: the user set
        // ROCM_CLI_DISABLE_TORCH_ALIGNMENT, rocm-cli left their torch alone, and the
        // engine's exact pin is therefore unmet. The build belongs to neither the SDK
        // nor the engine — it is whatever the user chose — so the aligned-case rule
        // would call it a defect and reinstall vLLM, which installs the engine's torch
        // over the one the opt-out was set to keep. That is the CLI-side fight moved
        // into the engine, and it would make the opt-out worthless on any managed
        // runtime.
        let assessment = repair_from_violations(
            &[violation(
                "vllm",
                "The package `vllm` requires `torch==2.11.0+gitd0c8b1f`, but `2.9.1+cu128` is installed",
            )],
            Some(SDK_BUILD),
            OPTED_OUT,
        );

        assert!(
            !assessment.needed,
            "the opt-out must spare a hand-installed torch: {:?}",
            assessment.notes
        );
        assert!(
            assessment
                .notes
                .iter()
                .all(|note| !note.contains("was reinstalled")),
            "no note may claim a repair that did not happen: {:?}",
            assessment.notes
        );
        assert!(
            assessment
                .notes
                .iter()
                .any(|note| note.contains("ROCM_CLI_DISABLE_TORCH_ALIGNMENT")),
            "the reason given must be the opt-out, not a divergence rocm-cli produced: {:?}",
            assessment.notes
        );
        assert!(
            assessment
                .notes
                .iter()
                .all(|note| !note.contains("the runtime holds the SDK's build")),
            "rocm-cli did not install this torch and must not say it did: {:?}",
            assessment.notes
        );
    }

    #[test]
    fn an_opted_out_custom_torch_still_repairs_an_unrelated_defect() {
        // The opt-out is about torch, not about the environment. A vLLM-owned pin that
        // has nothing to do with torch is broken the same way it was before, and
        // reinstalling vLLM is still what fixes it. Returning early on the opt-out
        // would hide this defect behind a preference about a different package, and the
        // runtime would stay unable to serve with nothing said about why.
        let assessment = repair_from_violations(
            &[
                violation(
                    "vllm",
                    "The package `vllm` requires `torch==2.11.0+gitd0c8b1f`, but `2.9.1+cu128` is installed",
                ),
                violation(
                    "vllm",
                    "The package `vllm` requires `torchvision==0.24.1+d801a34`, but `0.20.0+cu128` is installed",
                ),
            ],
            Some(SDK_BUILD),
            OPTED_OUT,
        );

        assert!(
            assessment.needed,
            "an unrelated vLLM pin is still a defect under the opt-out: {:?}",
            assessment.notes
        );
        let violation_notes: Vec<&String> = assessment
            .notes
            .iter()
            .filter(|note| note.starts_with("violation: "))
            .collect();
        assert_eq!(
            violation_notes.len(),
            1,
            "only the unrelated pin is a violation: {:?}",
            assessment.notes
        );
        assert!(
            violation_notes[0].contains("torchvision=="),
            "the defect named must be the unrelated one: {:?}",
            assessment.notes
        );
        assert!(
            assessment
                .notes
                .iter()
                .any(|note| note.starts_with("expected divergence: ")
                    && note.contains("torch==2.11.0+gitd0c8b1f")),
            "the spared torch is still named, so the reader is not left thinking the \
             reinstall was about torch: {:?}",
            assessment.notes
        );
    }

    #[test]
    fn a_recorded_sdk_torch_names_the_build() {
        let manifest = TheRockRuntimeManifest {
            sdk_torch: Some("2.11.0+rocm7.13.0".to_owned()),
            ..TheRockRuntimeManifest::default()
        };

        assert_eq!(
            sdk_torch_build_from_manifest(&manifest).as_deref(),
            Some("rocm7.13.0")
        );
    }

    #[test]
    fn a_manifest_without_sdk_torch_reconstructs_the_build_from_the_sdk_version() {
        // Written before `sdk_torch` was recorded. These are the runtimes already on
        // real machines, so the fallback is what repairs them rather than a nicety.
        let manifest = TheRockRuntimeManifest {
            rocm_sdk: Some(RocmSdkRuntimeProbe {
                rocm_sdk_version: Some("7.13.0".to_owned()),
                ..RocmSdkRuntimeProbe::default()
            }),
            ..TheRockRuntimeManifest::default()
        };

        assert_eq!(
            sdk_torch_build_from_manifest(&manifest).as_deref(),
            Some("rocm7.13.0")
        );
    }

    #[test]
    fn a_manifest_that_identifies_no_sdk_build_says_so() {
        assert_eq!(
            sdk_torch_build_from_manifest(&TheRockRuntimeManifest::default()),
            None
        );
    }

    #[test]
    fn a_settled_runtime_converges_on_the_build_its_own_manifest_records() {
        // The convergence proof the tests above cannot give on their own. They hand
        // the classification a build literal, so a change to what
        // `sdk_torch_build_from_manifest` yields — `7.13.0` where the local segment
        // reads `rocm7.13.0`, say — would leave every one of them passing while the
        // real pipeline churned forever: the engine would call the realigned torch a
        // defect, reinstall its own build, rocm-cli would put the SDK's back, and the
        // next invocation would start over. Feeding the classification the value the
        // manifest actually produces is what ties the two halves together.
        //
        // The SDK's own torch release is deliberately not the one vLLM pins, because
        // that is the case realignment exists for: the release comes from the engine,
        // only the build comes from the SDK.
        let settled = violation(
            "vllm",
            "The package `vllm` requires `torch==2.11.0+gitd0c8b1f`, but `2.11.0+rocm7.13.0` is installed",
        );
        let recorded = TheRockRuntimeManifest {
            sdk_torch: Some("2.9.1+rocm7.13.0".to_owned()),
            ..TheRockRuntimeManifest::default()
        };
        // Written before `sdk_torch` was recorded. These runtimes are already on real
        // machines, so they have to settle too rather than churn forever.
        let reconstructed = TheRockRuntimeManifest {
            rocm_sdk: Some(RocmSdkRuntimeProbe {
                rocm_sdk_version: Some("7.13.0".to_owned()),
                ..RocmSdkRuntimeProbe::default()
            }),
            ..TheRockRuntimeManifest::default()
        };

        for manifest in [recorded, reconstructed] {
            let build = sdk_torch_build_from_manifest(&manifest)
                .expect("both manifest generations identify the SDK's build");
            let assessment =
                repair_from_violations(std::slice::from_ref(&settled), Some(&build), ALIGNED);

            assert!(
                !assessment.needed,
                "the state rocm-cli settles on must survive the engine's own check: {:?}",
                assessment.notes
            );
        }
    }

    #[test]
    fn an_unrunnable_check_reports_itself_without_forcing_a_reinstall() {
        let assessment = unverified_repair("uv binary is unavailable");

        assert!(!assessment.needed);
        assert_eq!(assessment.notes.len(), 1);
        assert!(
            assessment.notes[0].contains("could not be verified"),
            "{:?}",
            assessment.notes
        );
    }

    #[test]
    fn vllm_extra_index_url_defaults_to_const() {
        assert_eq!(
            resolve_vllm_rocm_extra_index_url(None),
            VLLM_ROCM_EXTRA_INDEX_URL
        );
        assert_eq!(
            resolve_vllm_rocm_extra_index_url(Some("   ".to_owned())),
            VLLM_ROCM_EXTRA_INDEX_URL
        );
    }

    #[test]
    fn vllm_extra_index_url_honors_override() {
        assert_eq!(
            resolve_vllm_rocm_extra_index_url(Some(
                "  https://example.test/rocm/wheels  ".to_owned()
            )),
            "https://example.test/rocm/wheels"
        );
    }

    #[test]
    fn vllm_install_target_defaults_to_the_built_in_pin() {
        let target = install_target(None).expect("built-in target resolves");
        assert_eq!(target.index_url, VLLM_ROCM_EXTRA_INDEX_URL);
        assert_eq!(target.requirement, VLLM_PINNED_SPEC);

        let blank = install_target(Some("  ")).expect("a blank override is ignored");
        assert_eq!(blank, target);
    }

    #[test]
    fn vllm_install_target_stays_pinned_for_a_same_shape_index_override() {
        for (index, expected_url) in [
            (
                " https://wheels.vllm.ai/rocm/0.27.0/rocm730 ",
                "https://wheels.vllm.ai/rocm/0.27.0/rocm730",
            ),
            (
                "https://wheels.vllm.ai/rocm/0.27.0/rocm730/",
                "https://wheels.vllm.ai/rocm/0.27.0/rocm730/",
            ),
        ] {
            let target = install_target(Some(index)).expect("published index shape resolves");
            assert_eq!(target.index_url, expected_url);
            assert_eq!(target.requirement, "vllm==0.27.0+rocm730");
        }
    }

    #[test]
    fn vllm_install_target_fails_rather_than_unpinning_an_unknown_index() {
        for index in [
            "https://example.test/rocm/wheels",
            // Rolling latest-only index: no version/ABI to pin to.
            "https://wheels.vllm.ai/rocm/",
            "https://wheels.vllm.ai/rocm/0.27.0",
            "https://wheels.vllm.ai/rocm//rocm730",
            "https://wheels.vllm.ai/rocm/0.27.0/rocm 730",
        ] {
            let error = install_target(Some(index))
                .expect_err("an unpinnable index must fail")
                .to_string();
            assert!(
                error.contains(index.trim()),
                "error for {index} should quote the index URL: {error}"
            );
            assert!(
                error.contains(VLLM_ROCM_INDEX_PREFIX),
                "error for {index} should show the expected index shape: {error}"
            );
        }
    }

    #[test]
    fn vllm_pinned_spec_matches_extra_index_url() {
        let (version, abi) = VLLM_PINNED_SPEC
            .strip_prefix("vllm==")
            .and_then(|rest| rest.split_once('+'))
            .expect("pinned spec is spelled `vllm==<version>+<abi>`");
        let expected_suffix = format!("/{version}/{abi}");
        assert!(
            VLLM_ROCM_EXTRA_INDEX_URL.ends_with(&expected_suffix),
            "pinned spec {VLLM_PINNED_SPEC} does not match index {VLLM_ROCM_EXTRA_INDEX_URL}: \
             expected the index to end with {expected_suffix}"
        );
    }

    #[test]
    fn vllm_default_index_has_the_published_release_shape() {
        assert_eq!(
            vllm_rocm_build_from_index_url(VLLM_ROCM_EXTRA_INDEX_URL),
            Some(("0.26.0".to_owned(), "rocm723".to_owned())),
            "the built-in index must parse back to its build, or overrides of the same shape \
             cannot be pinned"
        );
    }

    #[test]
    fn stdio_protocol_routes_all_methods_without_side_effects() {
        let service_id = format!(
            "missing-protocol-{}-{}",
            std::process::id(),
            current_unix_millis()
        );
        let success_cases = [
            (EngineMethod::Detect, json!({})),
            (EngineMethod::Capabilities, json!({})),
            (
                EngineMethod::ResolveModel,
                json!({
                    "model_ref": "Qwen/Qwen3.5-4B",
                    "device_policy": "gpu_required"
                }),
            ),
            (
                EngineMethod::Healthcheck,
                json!({
                    "service_id": service_id.as_str()
                }),
            ),
            (
                EngineMethod::Logs,
                json!({
                    "service_id": service_id.as_str(),
                    "tail_lines": 4
                }),
            ),
            (
                EngineMethod::Stop,
                json!({
                    "service_id": service_id.as_str(),
                    "force": false
                }),
            ),
        ];

        for (method, payload) in success_cases {
            let response = handle_envelope(EngineRequestEnvelope { method, payload });
            assert!(
                response.ok,
                "expected protocol method to return a typed success envelope: {:?}",
                response.error
            );
        }

        let endpoint = handle_envelope(EngineRequestEnvelope {
            method: EngineMethod::Endpoint,
            payload: json!({
                "service_id": service_id.as_str()
            }),
        });
        assert!(!endpoint.ok);
        assert_eq!(
            endpoint.error.as_ref().map(|error| error.code.as_str()),
            Some("request_failed")
        );

        for method in [EngineMethod::Install, EngineMethod::Launch] {
            let response = handle_envelope(EngineRequestEnvelope {
                method,
                payload: json!({}),
            });
            assert!(!response.ok);
            assert_eq!(
                response.error.as_ref().map(|error| error.code.as_str()),
                Some("invalid_payload")
            );
        }
    }

    #[test]
    fn therock_library_path_entries_include_sysdeps_for_hip_apps() {
        let root = PathBuf::from(if cfg!(windows) {
            r"C:\rocm-sdk"
        } else {
            "/tmp/rocm-sdk"
        });
        let runtime = VllmRuntime {
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            env_id: "external-vllm-therock".to_owned(),
            command: PathBuf::from("vllm"),
            python_executable: None,
            version: None,
            source: "managed_runtime_manifest:test".to_owned(),
            sdk_root: Some(root.clone()),
            sdk_bin: Some(root.join("bin")),
            sdk_bin_paths: vec![root.join("runtime").join("bin")],
            sdk_library_paths: vec![root.join("runtime").join("lib")],
        };
        let entries = therock_library_path_entries(&runtime);
        assert!(entries.contains(&root.join("runtime").join("lib")));
        assert!(entries.contains(&root.join("lib")));
        assert!(
            entries
                .iter()
                .any(|entry| entry.ends_with(Path::new("lib").join("rocm_sysdeps").join("lib")))
        );
    }

    #[test]
    fn launch_env_sets_vllm_rocm_target_device() -> Result<()> {
        let runtime = VllmRuntime {
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            env_id: "external-vllm-therock".to_owned(),
            command: PathBuf::from("vllm"),
            python_executable: None,
            version: None,
            source: "managed_runtime_manifest:test".to_owned(),
            sdk_root: None,
            sdk_bin: None,
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
        };
        let mut command = ProcessCommand::new("vllm");

        apply_therock_env(&mut command, &runtime)?;

        let target_device = command
            .get_envs()
            .find_map(|(key, value)| (key == "VLLM_TARGET_DEVICE").then_some(value))
            .flatten();
        assert_eq!(target_device, Some(std::ffi::OsStr::new("rocm")));
        Ok(())
    }

    #[test]
    fn running_state_records_managed_therock_env_for_gpu_verification() -> Result<()> {
        let state_path = std::env::temp_dir().join(format!(
            "rocm-vllm-state-{}-{}.json",
            std::process::id(),
            current_unix_millis()
        ));
        let request = ServeHttpRequest {
            service_id: "svc-vllm".to_owned(),
            model_ref: "facebook/opt-125m".to_owned(),
            host: "127.0.0.1".to_owned(),
            port: 11439,
            device_policy: DevicePolicy::GpuRequired,
            gpu_indices: Vec::new(),
            runtime_id: Some("runtime-key-gfx120x".to_owned()),
            env_id: None,
            state_path: state_path.clone(),
            log_path: None,
            engine_recipe: None,
        };
        let runtime = VllmRuntime {
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            env_id: "external-vllm-therock".to_owned(),
            command: PathBuf::from(if cfg!(windows) {
                r"C:\venv\Scripts\vllm.exe"
            } else {
                "/home/user/.venv/bin/vllm"
            }),
            python_executable: None,
            version: Some("test".to_owned()),
            source: "managed_runtime_manifest:test".to_owned(),
            sdk_root: Some(PathBuf::from(if cfg!(windows) {
                r"C:\rocm-sdk"
            } else {
                "/home/user/.venv/lib/python/site-packages/rocm_sdk"
            })),
            sdk_bin: Some(PathBuf::from(if cfg!(windows) {
                r"C:\rocm-sdk\bin"
            } else {
                "/home/user/.venv/lib/python/site-packages/rocm_sdk/bin"
            })),
            sdk_bin_paths: vec![PathBuf::from(if cfg!(windows) {
                r"C:\rocm-sdk\extra-bin"
            } else {
                "/home/user/.venv/lib/python/site-packages/_rocm_sdk_libraries/bin"
            })],
            sdk_library_paths: vec![PathBuf::from(if cfg!(windows) {
                r"C:\rocm-sdk\extra-lib"
            } else {
                "/home/user/.venv/lib/python/site-packages/_rocm_sdk_libraries/lib"
            })],
        };

        write_running_state(&request, &runtime, 12345)?;
        let state = read_service_state(&state_path)?;
        fs::remove_file(&state_path).ok();

        assert_eq!(state.get("server_pid").and_then(Value::as_u64), Some(12345));
        assert_eq!(
            state.get("runtime_id").and_then(Value::as_str),
            Some("therock-release:gfx120X-all")
        );
        assert_eq!(
            state.get("requested_runtime_id").and_then(Value::as_str),
            Some("runtime-key-gfx120x")
        );
        let runtime_env = state
            .get("therock_runtime_env")
            .expect("runtime env should be recorded");
        assert_eq!(
            runtime_env.get("runtime_id").and_then(Value::as_str),
            Some("therock-release:gfx120X-all")
        );
        assert!(
            runtime_env
                .get("root")
                .and_then(Value::as_str)
                .is_some_and(|root| root.contains("rocm"))
        );
        assert!(
            runtime_env
                .get("bin_paths")
                .and_then(Value::as_array)
                .is_some_and(|paths| paths.len() >= 2)
        );
        assert!(
            runtime_env
                .get("library_paths")
                .and_then(Value::as_array)
                .is_some_and(|paths| !paths.is_empty())
        );
        // The identity token must be recorded so a later stop can verify it.
        assert!(state.get("start_ticks").is_some());
        Ok(())
    }

    #[test]
    fn stop_scope_includes_vllm_workers() {
        assert_eq!(STOP_SCOPE, rocm_core::KillScope::Tree);
    }

    #[test]
    fn identity_from_state_carries_pid_and_start_ticks() {
        let state = json!({ "pid": 4321, "start_ticks": 987_654_u64 });
        let identity = identity_from_state(&state).expect("identity");
        assert_eq!(identity.pid, 4321);
        assert_eq!(identity.start_ticks, Some(987_654));
    }

    #[test]
    fn identity_from_legacy_state_has_no_start_ticks() {
        // State files written before this change carry only `pid`; verification
        // must degrade gracefully rather than fail to parse.
        let state = json!({ "pid": 4321 });
        let identity = identity_from_state(&state).expect("identity");
        assert_eq!(identity.pid, 4321);
        assert_eq!(identity.start_ticks, None);
    }

    #[test]
    fn identity_from_state_without_pid_is_none() {
        assert!(identity_from_state(&json!({ "status": "running" })).is_none());
    }
}
