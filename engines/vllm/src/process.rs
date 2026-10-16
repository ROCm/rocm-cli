// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, anyhow, bail};
use rocm_core::{AppPaths, DEFAULT_LOCAL_PORT, require_nonempty};
use rocm_engine_protocol::{
    DEFAULT_LOG_TAIL_LINES, DevicePolicy, ENGINE_RECIPE_CONTRACT_VERSION, EngineRecipeHint,
    GpuSelection, LaunchRequest, LaunchResponse, ResolveModelRequest, ResolveModelResponse,
    StopRequest, StopResponse,
};
use serde_json::json;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::time::{Duration, Instant};

use crate::runtime::{
    VllmRuntime, ensure_therock_compat_shims, runtime_bin_paths, runtime_has_discover_build,
    therock_library_path_entries_for_launch,
};
use crate::{ServeHttpRequest, engine_recipe_launch_args};

/// Tail budget for the startup-failure summary (and its OOM hint).
///
/// Derived from [`DEFAULT_LOG_TAIL_LINES`] rather than spelled as its own
/// literal so this surface and the CLI's serve summary — which reads the same
/// protocol constant in `append_oom_serve_note` — cannot drift on what counts as
/// "the tail" of a failed launch. A change to the protocol budget moves both.
const STARTUP_FAILURE_LOG_TAIL_LINES: usize = DEFAULT_LOG_TAIL_LINES;

/// How long a stop waits for the server to actually exit after each signal
/// before reporting a timeout (or, under `force`, escalating to `SIGKILL`).
const STOP_GRACE: Duration = Duration::from_secs(10);
/// vLLM launches worker descendants that retain GPU allocations.
const STOP_SCOPE: rocm_core::KillScope = rocm_core::KillScope::Tree;
/// Default time to wait for vLLM to report readiness before giving up.
const DEFAULT_VLLM_READY_TIMEOUT: Duration = Duration::from_mins(5);

pub(crate) fn resolve_model_response(request: ResolveModelRequest) -> Result<ResolveModelResponse> {
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
            "host": crate::DEFAULT_HOST,
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
        if hint.engine != crate::ENGINE_NAME {
            bail!(
                "engine_recipe target `{}` does not match adapter `{}`",
                hint.engine,
                crate::ENGINE_NAME
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

pub(crate) fn launch_service(request: LaunchRequest) -> Result<LaunchResponse> {
    let device_policy = normalize_vllm_device_policy(request.device_policy)?;
    let engine_recipe = accepted_engine_recipe(request.engine_recipe)?;
    let requested_gpu_indices =
        rocm_engine_protocol::launch_gpu_indices(request.gpu_selection.as_ref());
    let gpu_indices = resolve_serve_gpu_indices(&requested_gpu_indices)?;
    let runtime = crate::runtime::resolve_vllm_runtime(request.runtime_id.as_deref())?;
    let state_path = AppPaths::discover()?
        .engine_state_dir(crate::ENGINE_NAME)
        .join(format!("{}.json", request.service_id));
    let log_path = AppPaths::discover()?
        .engine_logs_dir(crate::ENGINE_NAME)
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
    crate::state::write_running_state(&serve_request, &runtime, pid)?;
    Ok(LaunchResponse {
        service_id: request.service_id,
        pid,
        endpoint_url: crate::state::endpoint_url(&request.host, request.port),
        log_path: log_path.display().to_string(),
        state_path: state_path.display().to_string(),
    })
}

pub(crate) fn serve_http(mut request: ServeHttpRequest) -> Result<()> {
    request.gpu_indices = resolve_serve_gpu_indices(&request.gpu_indices)?;
    let runtime = crate::runtime::resolve_vllm_runtime(request.runtime_id.as_deref())?;
    let mut child = spawn_vllm_server(&request, &runtime, request.log_path.as_deref())?;
    crate::state::write_running_state(&request, &runtime, child.id())?;

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
        crate::state::write_terminal_state(&request.state_path, "failed")?;
        return Err(e);
    }

    let status = child.wait().context("failed waiting for vLLM server")?;
    crate::state::write_terminal_state(
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
{}, then retry.",
            rocm_core::openmpi::install_hint()
                .or("run `rocm engines install vllm --yes` to install it automatically")
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
{}, then retry.",
            rocm_core::openmpi::libatomic_install_hint()
                .or("run `rocm engines install vllm --yes` to install it automatically")
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
{}, then retry.",
            rocm_core::openmpi::libnuma_install_hint()
                .or("run `rocm engines install vllm --yes` to install it automatically")
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

pub(crate) fn stop_service(request: StopRequest) -> Result<StopResponse> {
    require_nonempty(&request.service_id, "service_id")?;
    let files = crate::state::service_files(&request.service_id)?;
    let state = crate::state::read_service_state(&files.state_path).ok();
    let outcome = state
        .as_ref()
        .and_then(crate::state::identity_from_state)
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
        crate::state::write_terminal_state(&files.state_path, "stopped")?;
    }
    Ok(StopResponse { stopped, graceful })
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

pub(crate) fn parse_device_policy_arg(policy: Option<&str>) -> Result<DevicePolicy> {
    match policy.unwrap_or("gpu_required") {
        "gpu" | "gpu_required" => Ok(DevicePolicy::GpuRequired),
        "gpu_preferred" => Ok(DevicePolicy::GpuPreferred),
        "cpu" | "cpu_only" => Ok(DevicePolicy::CpuOnly),
        other => bail!("unsupported device policy: {other}"),
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

pub(crate) fn apply_therock_env(command: &mut ProcessCommand, runtime: &VllmRuntime) -> Result<()> {
    apply_therock_env_with(command, runtime, rocm_core::openmpi::ensure_mpi_cxx_compat)
}

/// Same as [`apply_therock_env`], with the `libmpi_cxx` probe
/// [`ensure_therock_compat_shims`] takes injectable, so a test can exercise
/// the whole "no shim needed" hand-off into `LD_LIBRARY_PATH` without
/// depending on this host's own OpenMPI/compiler state. Production code
/// always goes through [`apply_therock_env`].
fn apply_therock_env_with(
    command: &mut ProcessCommand,
    runtime: &VllmRuntime,
    ensure_mpi_cxx_compat: impl FnOnce(&Path) -> Option<PathBuf>,
) -> Result<()> {
    command.env("VLLM_TARGET_DEVICE", "rocm");
    if runtime_has_discover_build(runtime) {
        apply_vllm_rocm10_discover_env(command, runtime)?;
    }
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
        let compat_shim = ensure_therock_compat_shims(runtime, ensure_mpi_cxx_compat);
        command.env(
            "LD_LIBRARY_PATH",
            prepend_path_entries(
                &therock_library_path_entries_for_launch(runtime, compat_shim),
                std::env::var_os("LD_LIBRARY_PATH"),
            )?,
        );
    }
    Ok(())
}
/// The venv root a runtime's vLLM lives in, derived from its own python
/// executable (`<venv>/bin/python` → `<venv>`) rather than the TheRock SDK
/// root: the ROCm 10.x discovery install lands packages in the venv's own
/// site-packages, not under the SDK tree.
fn vllm_venv_root(runtime: &VllmRuntime) -> Result<PathBuf> {
    let python = runtime.python_executable.as_ref().ok_or_else(|| {
        anyhow!(
            "runtime `{}` has no python executable to derive a venv root from",
            runtime.runtime_id
        )
    })?;
    python
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| {
            anyhow!(
                "python executable `{}` is not inside a venv (bin/Scripts) layout",
                python.display()
            )
        })
}
/// Finds `<venv>/lib/python3.*/site-packages`, the one location a venv keeps
/// its packages under. Bails on anything other than exactly one `python3.*`
/// directory: zero means `venv` isn't a real venv, and more than one means
/// the layout is ambiguous and picking one would be a guess.
fn venv_site_packages_dir(venv: &Path) -> Result<PathBuf> {
    let lib_dir = venv.join("lib");
    let mut python_dirs = fs::read_dir(&lib_dir)
        .with_context(|| format!("failed to read {}", lib_dir.display()))?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("python3."))
        })
        .collect::<Vec<_>>();
    match python_dirs.len() {
        1 => Ok(python_dirs.remove(0).join("site-packages")),
        0 => bail!("no python3.* directory found under {}", lib_dir.display()),
        n => bail!(
            "found {n} python3.* directories under {}, expected exactly one",
            lib_dir.display()
        ),
    }
}
/// Environment vLLM needs on ROCm 10.x discovery installs, beyond the shared
/// TheRock env `apply_therock_env` already sets: `amd_smi` lives under the
/// venv's own site-packages rather than the SDK tree, and Triton's ROCm
/// flash-attention backend needs to be opted into explicitly.
fn apply_vllm_rocm10_discover_env(
    command: &mut ProcessCommand,
    runtime: &VllmRuntime,
) -> Result<()> {
    let venv = vllm_venv_root(runtime)?;
    let site_packages = venv_site_packages_dir(&venv)?;
    command.env(
        "PYTHONPATH",
        prepend_path_entries(
            &[site_packages
                .join("_rocm_sdk_core")
                .join("share")
                .join("amd_smi")],
            std::env::var_os("PYTHONPATH"),
        )?,
    );
    command.env("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE");
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
/// Builds a human-readable summary of the tail of the startup log, if available.
/// Returns an empty string when no log is present or it cannot be read.
///
/// The tail is sanitized before it is embedded. It is untrusted subprocess
/// output and this whole string is `bail!`-ed to the user's terminal, so
/// stripping control bytes out of the one echoed line further down while
/// printing the raw tail a few lines above would have left the escapes a clear
/// path to the terminal inside the same message. `oom_utilization_hint` still
/// reads the *raw* tail, because what it selects from it is quoted into a
/// command and has its own, stricter admissibility test.
///
/// Sanitizing with [`rocm_core::terminal::rendered_lines`] rather than
/// `strip_terminal_control_sequences` is load-bearing, not a stylistic
/// preference. Stripping deletes every line-break token (`\r`, `ESC[1A`, `NEL`,
/// `VT`, `U+2028`…) *with no separator in its place*, so a progress-bar repaint
/// — `Loading weights: 90%\rtorch.OutOfMemoryError: HIP out of memory.`, which
/// vLLM emits as one `\n` line — was echoed as the single welded line
/// `Loading weights: 90%torch.OutOfMemoryError: HIP out of memory.`. That is
/// text the engine never printed, and it is worse than cosmetic: a managed
/// launch shares one log between the supervisor and the engine, so the CLI
/// writes this whole bail back into the same file, and the next read finds the
/// welded line, scores it as a genuine OOM, finds it quotable (the control bytes
/// are gone) and prints it inside the `rocm diagnose --symptom '…'` command the
/// serve summary invites the user to paste. Expanding each physical line into
/// the rows a terminal would have drawn keeps the echo faithful, keeps the
/// tail's shape, and leaves nothing weldable behind for the second read.
///
/// Pinned by `the_echoed_tail_never_welds_two_rendered_rows_into_one_line`.
fn startup_log_context(log_path: Option<&Path>) -> String {
    let summary = log_path
        .and_then(|p| summarize_startup_log_tail(p, STARTUP_FAILURE_LOG_TAIL_LINES).ok())
        .unwrap_or_default();
    if summary.is_empty() {
        return String::new();
    }
    let hint = oom_utilization_hint(&summary);
    let summary = summary
        .lines()
        .flat_map(rocm_core::terminal::rendered_lines)
        .collect::<Vec<_>>()
        .join("\n");
    // The count is the *read* budget — how many physical lines of the log were
    // taken — not how many rows are printed below. Those differ, and can differ
    // a lot: each `\r` repaint of a progress bar expands into its own rendered
    // row, so three physical lines carrying 300 repaints are echoed as 300-odd
    // rows. Saying "last N lines" and then printing more of them is the kind of
    // small lie that costs a reader their trust in the rest of the message, so
    // the sentence says which N it means.
    format!(
        "\n\nStartup log, last {STARTUP_FAILURE_LOG_TAIL_LINES} lines as the terminal drew \
         them:\n{summary}{hint}"
    )
}

/// Appends [`rocm_core::VLLM_GPU_MEMORY_UTILIZATION_HINT`] when the startup-log
/// tail shows an out-of-memory failure, and nothing otherwise so healthy
/// failures are not cluttered with an irrelevant memory hint. Shares its wording
/// with the `rocm` CLI's pre-launch low-VRAM note so both surfaces point the
/// user at the same fix rather than drifting into different phrasing.
fn oom_utilization_hint(log_tail: &str) -> String {
    // Route the user's *actual* failing line into the `--symptom` example when
    // it can be rendered as one intact single-quoted argument; otherwise fall
    // back to the canonical symptom so the printed command always reports a
    // cause rather than reading as "the tool checked and there's no known
    // cause". Selecting the line is the shared helper's job, so this surface and
    // the `rocm` CLI serve summary apply one selection *rule* to `rocm diagnose`
    // instead of hand-rolling it twice; it classifies each line with the *same*
    // rule the diagnose checker uses. One rule is not one answer: the two read
    // the shared log at different points, with this bail's own sanitized echo
    // written into it in between, so for a colourised line they can print
    // different (both sound) `--symptom` arguments -- this one the canonical
    // symptom, the CLI the rendered row. See `rocm_core::vllm_log_segments` for
    // the measurement. Its guarantee is specifically that
    // a line passing `quotable_in_single_quotes` is diagnosable -- not that
    // every returned line is -- which is exactly what the branch below needs,
    // since the non-quotable branch substitutes the canonical symptom anyway.
    //
    // No line means the tail carries no OOM, so the helper's `None` is the OOM
    // gate too; there is no separate classifier predicate to ask first.
    //
    // The helper also skips lines rocm-cli wrote into the log itself, which
    // matters precisely here: this hint is rendered into the engine's bail, the
    // CLI writes that whole error back into the same shared log, and the hint
    // quotes the user's line and names the allocator keywords, so it scores as
    // an OOM just as strongly as the real line does. Being last, it would win
    // the newest-first scan on the next read — and it contains `'`, so the
    // printed command would fall back to the canonical symptom in every real
    // run. That is why the two markers below come from shared constants rather
    // than being spelled out here.
    let Some(symptom) = rocm_core::vllm_oom_diagnose_symptom(log_tail) else {
        return String::new();
    };
    // The line is subprocess output, so it is echoed only after the terminal
    // control bytes vLLM's colourised logger emits are removed -- but removing
    // them is not enough on its own. A break token deleted with nothing in its
    // place welds the two rows either side of it into a line the engine never
    // printed (`Loading weights: 90%torch.OutOfMemoryError: ...`), which would
    // then be presented here as the user's own output. The shared helper returns
    // the rendered row that actually scored, which is one row the terminal
    // really drew.
    let symptom_line =
        rocm_core::vllm_oom_display_line(symptom.strip_prefix("vllm: ").unwrap_or(&symptom));
    let symptom = if rocm_core::quotable_in_single_quotes(&symptom) {
        symptom
    } else {
        rocm_core::VLLM_OOM_CANONICAL_SYMPTOM.to_owned()
    };
    format!(
        "\n\n{}{symptom_line}). {}\n\
         For conditional remediation, run `{} '{symptom}'`.",
        rocm_core::VLLM_OOM_DETECTED_PREFIX,
        rocm_core::VLLM_GPU_MEMORY_UTILIZATION_HINT,
        rocm_core::VLLM_OOM_DIAGNOSE_COMMAND,
    )
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
        match crate::state::query_loaded_model_endpoint(&endpoint, Some(model_ref)) {
            Ok(true)
                if crate::state::query_inference_probe_endpoint(&endpoint, model_ref)
                    .unwrap_or(false) =>
            {
                return Ok(());
            }
            _ => std::thread::sleep(poll_interval),
        }
    }
}
/// Reads the last N lines from a log file and returns them as a formatted string.
/// Delegates the seek/tail logic to `rocm_core::tail_lines`.
fn summarize_startup_log_tail(log_path: &Path, limit: usize) -> Result<String> {
    let lines = rocm_core::tail_lines(log_path, limit, crate::state::MAX_TAIL_READ)?;
    if lines.is_empty() {
        return Ok(String::new());
    }
    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ENGINE_NAME;
    use crate::runtime::VllmRuntime;
    use crate::state::current_unix_millis;
    use std::fmt::Write as _;

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
    /// The OOM gate [`oom_utilization_hint`] itself uses. There is no separate
    /// classifier predicate in `rocm-core`: the selector returning `Some` *is*
    /// the gate, so these tests assert against the function that ships.
    fn detects_oom(log: &str) -> bool {
        rocm_core::vllm_oom_diagnose_symptom(log).is_some()
    }

    #[test]
    fn the_hint_never_quotes_itself_back_on_a_second_read_of_the_shared_log() {
        // A managed launch shares one log between the supervisor and the engine.
        // The engine bails with `startup_log_context`, which embeds *this* hint,
        // and the CLI then writes that whole error back into the same file. So
        // on the read that drives the serve summary the hint is in the log, it
        // is last, and it scores as an OOM just as strongly as the real line.
        //
        // That is the end-to-end shape of the bug: the newest-first scan picked
        // this sentence, which contains `'`, the quoting guard rejected it, and
        // the note fell back to the canonical symptom in every real run — so the
        // "we quote your own failing line" promise never once held in the field.
        // A unit test on the hint alone cannot see it; only feeding the hint
        // back through the selector can.
        let real_line = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";
        let hint = oom_utilization_hint(real_line);
        assert!(!hint.is_empty(), "the real line must produce a hint");

        let second_read = format!("{real_line}\nError: launch failed{hint}\n");
        let symptom = rocm_core::vllm_oom_diagnose_symptom(&second_read)
            .expect("the real OOM line must still be found under the hint");
        assert_eq!(
            symptom,
            format!("vllm: {real_line}"),
            "the user's own line must win over the hint rocm-cli wrote itself"
        );
        assert!(
            rocm_core::quotable_in_single_quotes(&symptom),
            "and it must survive the quoting guard, or the printed command falls \
             back to the canonical symptom exactly as it did before this fix"
        );
    }

    #[test]
    fn oom_utilization_hint_fires_on_out_of_memory_log_tails() {
        // The exact PyTorch/HIP signature from the field report.
        let torch = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";
        assert!(detects_oom(torch));
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
        assert!(detects_oom("HIP OUT OF MEMORY"));
        assert!(detects_oom("RuntimeError: CUDA out of memory"));
    }

    #[test]
    fn every_emitted_oom_symptom_is_diagnosable_and_names_the_branch_that_produced_it() {
        // Closes the loop between the two layers: the engine prints
        // `rocm diagnose --symptom '<symptom>'`, so whatever it emits must
        // actually score for the diagnose checker. What makes that true is the
        // pair of branches, not the selector alone: a *quotable* selected line
        // is diagnosable because the selector scores with the checker's own
        // rule, and a non-quotable one is replaced by the canonical symptom,
        // which is diagnosable by definition. This guards both branches.
        //
        // Each case pins the *exact* symptom, not just that one is diagnosable.
        // Diagnosability alone cannot fail for the defect this test is named
        // for: the canonical symptom is diagnosable by construction, so a
        // mutant that deletes the "route the user's real failing line" feature
        // entirely and always returns VLLM_OOM_CANONICAL_SYMPTOM satisfies
        // every such assertion. Naming the expected symptom per line is what
        // makes the routing branch and the fallback branch separately
        // falsifiable -- and it is why the table below deliberately mixes the
        // two.
        //
        // Since the detector was reconciled with the checker, the fallback is no
        // longer reached by sub-threshold scoring (those lines now produce no
        // hint at all -- see `sub_threshold_lines_carry_no_hint_at_all`); it is
        // reached by a line that scores but cannot be rendered as one intact
        // single-quoted argument. So the fallback rows here are the
        // *unquotable* ones, which is also what keeps the admissibility guard
        // falsifiable through the surface that actually prints the command.
        let accepted_lines = [
            // Score and are quotable: the user's own line must be quoted verbatim.
            (
                "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.",
                "vllm: torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.",
            ),
            (
                "RuntimeError: hipErrorOutOfMemory",
                "vllm: RuntimeError: hipErrorOutOfMemory",
            ),
            (
                "torch.cuda.OutOfMemoryError: CUDA out of memory",
                "vllm: torch.cuda.OutOfMemoryError: CUDA out of memory",
            ),
            ("CUDA out of memory", "vllm: CUDA out of memory"),
            // Score but are inadmissible in a single-quoted word, so they must
            // reach the canonical fallback: an apostrophe (which would close the
            // quote), a colourised line (control bytes), and a bidi override
            // (a `Cf` format character, which `char::is_control` does not catch
            // and which reorders how the printed command renders).
            (
                "torch.OutOfMemoryError: HIP out of memory. GPU 0 can't allocate the model's weights.",
                rocm_core::VLLM_OOM_CANONICAL_SYMPTOM,
            ),
            (
                "\u{1b}[31mtorch.OutOfMemoryError: HIP out of memory\u{1b}[0m",
                rocm_core::VLLM_OOM_CANONICAL_SYMPTOM,
            ),
            (
                "torch.OutOfMemoryError: \u{202e}HIP out of memory. Tried to allocate 7.21 GiB.",
                rocm_core::VLLM_OOM_CANONICAL_SYMPTOM,
            ),
        ];
        let mut routed_verbatim = 0;
        let mut fell_back = 0;
        for (line, expected) in accepted_lines {
            assert!(detects_oom(line), "detector must accept: {line:?}");
            let hint = oom_utilization_hint(line);
            let symptom = quoted_symptom_argument(&hint);
            assert_eq!(
                symptom, expected,
                "wrong branch for {line:?}: the engine must quote the user's own line when \
                 it can be rendered as one intact single-quoted argument and fall back to \
                 the canonical symptom only when it cannot"
            );
            assert!(
                rocm_core::vllm_oom_symptom_is_diagnosable(symptom),
                "emitted symptom must be diagnosable, got {symptom:?} for line {line:?}"
            );
            if symptom == rocm_core::VLLM_OOM_CANONICAL_SYMPTOM {
                fell_back += 1;
            } else {
                routed_verbatim += 1;
            }
        }
        // Both counters, not just the verbatim one: a table that drifted until
        // every row took the same branch would still satisfy every per-row
        // assertion above, which is the vacuity this test was written to close.
        assert_eq!(
            (routed_verbatim, fell_back),
            (4, 3),
            "the table must keep exercising both branches; if an admissibility change \
             moved a line across the quotability boundary, re-pick the fixture rather than \
             relaxing the expectation. Scoring is not what separates these rows — every row \
             here is comfortably accepted by the detector, and the three fallbacks are \
             fallbacks purely because they cannot be rendered as one single-quoted word"
        );
    }

    #[test]
    fn sub_threshold_lines_carry_no_hint_at_all() {
        // The reconciled detector rejects sub-threshold lines the loose scan used
        // to accept -- exactly the false positives the diagnose checker's
        // threshold guards against. Rejecting them here keeps the two layers from
        // disagreeing (the engine must not print a `--symptom` the checker would
        // then score as "no known cause").
        //
        // This is a different property from the one above: there the question is
        // *which* symptom an emitted hint carries, here it is whether a hint is
        // emitted at all. Before the detector was reconciled these lines reached
        // the canonical fallback instead; now they produce nothing, and printing
        // a memory hint for them would be a regression the symptom table cannot
        // see.
        let rejected_lines = [
            // The bare exception class every PyTorch OOM raises, uncorroborated.
            "torch.cuda.OutOfMemoryError",
            // A spaced "out of memory" with no allocator anchor phrase.
            "HIP error: out of memory",
            // A kernel OOM-killer line from an unrelated subprocess.
            "the process was killed: out of memory",
        ];
        for line in rejected_lines {
            assert!(
                !detects_oom(line),
                "detector must reject sub-threshold line: {line}"
            );
            assert!(
                oom_utilization_hint(line).is_empty(),
                "a sub-threshold line must not carry a memory hint: {line}"
            );
        }
    }

    /// The `--symptom` value the hint actually hands the user, read back out of
    /// the rendered text exactly the way a shell would: everything between the
    /// opening quote that follows `--symptom ` and the next `'`.
    fn quoted_symptom_argument(hint: &str) -> &str {
        hint.split("--symptom '")
            .nth(1)
            .and_then(|rest| rest.split('\'').next())
            .expect("hint must carry a --symptom value")
    }

    #[test]
    fn an_apostrophe_in_the_failing_line_cannot_break_out_of_the_printed_command() {
        // A realistic vLLM/PyTorch failing line: apostrophes are routine in
        // Python error text, and this one scores well above MIN_SCORE_FOR_MATCH
        // (torch.OutOfMemoryError + HIP out of memory), so the "route the user's
        // real line" branch selects it.
        let log_tail = concat!(
            "ERROR 09-14 12:00:01 engine.py:389] torch.OutOfMemoryError: HIP out of memory. ",
            "Tried to allocate 7.21 GiB. GPU 0 can't allocate the model's weights."
        );
        assert!(detects_oom(log_tail));
        let hint = oom_utilization_hint(log_tail);

        // The rendered command must be one intact single-quoted argument: no
        // user-controlled byte may close the quote, so nothing the subprocess
        // printed can land outside it in a command the user is told to paste.
        let command = hint
            .split("run `")
            .nth(1)
            .and_then(|rest| rest.split('`').next())
            .expect("hint must print a runnable command");
        assert_eq!(
            command.matches('\'').count(),
            2,
            "the --symptom argument must stay a single balanced quoted word: {command}"
        );
        let symptom = quoted_symptom_argument(&hint);
        // Pin which branch ran, not just that no apostrophe survived: "contains
        // no `'`" also holds if the apostrophes were silently deleted from the
        // user's line, which is the escaping-style behaviour the fallback exists
        // to avoid. Rejection means the canonical symptom, exactly.
        assert_eq!(
            symptom,
            rocm_core::VLLM_OOM_CANONICAL_SYMPTOM,
            "a quote-bearing line must be rejected in favour of the canonical symptom, \
             not silently rewritten: {symptom:?}"
        );
        // ...and the command it does print must still report a cause.
        assert!(
            rocm_core::vllm_oom_symptom_is_diagnosable(symptom),
            "the fallback symptom must still be diagnosable: {symptom:?}"
        );
    }

    #[test]
    fn control_bytes_from_the_log_are_stripped_from_the_echoed_line() {
        // vLLM's logger colourises; an ANSI-coloured OOM line must not repaint
        // the user's terminal from inside rocm-cli's own error message.
        let log_tail = "\u{1b}[31mRuntimeError: HIP out of memory\u{1b}[0m\u{7}";
        let hint = oom_utilization_hint(log_tail);
        assert!(
            !hint
                .chars()
                .any(|c| c.is_control() && c != '\n' && c != '\t'),
            "no cursor-moving control byte may survive into the printed hint — `\\t` is \
             exempt because `rendered_lines` keeps it deliberately, as intra-line \
             whitespace that moves along the row it is already on: {hint:?}"
        );
        // Pin the exact rendering rather than the absence of a few fragments.
        // Absence checks cannot fail for the defect they name: a stripper that
        // drops the escape byte and the `[` it introduces and nothing further
        // leaves `31m`/`0m` behind, which contains neither `[31m` nor `[0m` and
        // carries no control byte, so every absence check still passes.
        assert_eq!(
            rocm_core::strip_terminal_control_sequences(log_tail),
            "RuntimeError: HIP out of memory",
            "the ANSI sequence must be removed whole, not just its escape byte"
        );
        // The colourised line is not quotable (control bytes), so the command
        // falls back to the canonical symptom while the echoed sentence keeps
        // the user's own line, stripped.
        assert_eq!(
            hint,
            format!(
                "\n\nDetected an out-of-memory failure (RuntimeError: HIP out of memory). {}\n\
                 For conditional remediation, run `rocm diagnose --symptom '{}'`.",
                rocm_core::VLLM_GPU_MEMORY_UTILIZATION_HINT,
                rocm_core::VLLM_OOM_CANONICAL_SYMPTOM
            ),
            "the echoed line must render exactly, with the whole escape sequence gone"
        );
    }

    #[test]
    fn the_printed_log_tail_is_sanitized_not_just_the_echoed_line() -> Result<()> {
        // The stripper's contract is that nothing in this message can repaint
        // the user's terminal. The log tail is printed a few lines above the
        // echoed line in the same `bail!`-ed string, and it used to go out raw,
        // so the escapes removed from one had a clear path to the terminal via
        // the other.
        let path = std::env::temp_dir().join(format!(
            "rocm-vllm-sanitize-{}-{}.log",
            std::process::id(),
            current_unix_millis()
        ));
        fs::write(
            &path,
            "\u{1b}[31mINFO starting\u{1b}[0m\n\u{1b}]0;pwned\u{7}RuntimeError: HIP out of memory\n",
        )?;
        let context = startup_log_context(Some(path.as_path()));
        fs::remove_file(&path).ok();

        assert!(
            !context
                .chars()
                .any(|c| c.is_control() && c != '\n' && c != '\t'),
            "no cursor-moving control byte may survive into the printed failure message — \
             `\\t` is exempt, see the hint assertion above: {context:?}"
        );
        assert!(
            !context.contains("pwned"),
            "an OSC body in the tail must not be emitted as text: {context:?}"
        );
        assert_eq!(
            context
                .split("as the terminal drew them:\n")
                .nth(1)
                .and_then(|rest| rest.split("\n\nDetected").next()),
            Some("INFO starting\nRuntimeError: HIP out of memory"),
            "the tail must keep its text and its line structure, minus the escapes: {context:?}"
        );
        Ok(())
    }

    #[test]
    fn the_echoed_tail_never_welds_two_rendered_rows_into_one_line() -> Result<()> {
        // vLLM draws its progress bar with bare carriage returns, so a real OOM
        // arrives as ONE `\n` line carrying two rows the terminal drew. The echo
        // used to sanitize with `strip_terminal_control_sequences`, which drops
        // the break with nothing in its place and welds them together.
        //
        // That is not cosmetic. A managed launch shares one log between the
        // supervisor and the engine: the CLI writes this whole bail back into
        // the same file, so the *second* read sees the welded line, scores it as
        // a genuine OOM, finds it quotable (the control bytes are gone) and
        // quotes it into the `rocm diagnose --symptom '...'` command the serve
        // summary invites the user to paste. The user would be told to paste a
        // line vLLM never printed.
        let exception = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";
        let welded = format!("Loading weights: 90%{exception}");
        let path = std::env::temp_dir().join(format!(
            "rocm-vllm-weld-{}-{}.log",
            std::process::id(),
            current_unix_millis()
        ));
        fs::write(&path, format!("Loading weights: 90%\r{exception}\n"))?;
        let context = startup_log_context(Some(path.as_path()));
        fs::remove_file(&path).ok();

        assert!(
            !context.contains(&welded),
            "the echoed tail must not weld the progress bar onto the exception: {context:?}"
        );
        assert!(
            context.contains(&format!("Loading weights: 90%\n{exception}")),
            "each row the terminal drew must be echoed on its own line: {context:?}"
        );
        // The displayed line beside the hint is the scoring row, not the segment
        // with its breaks deleted.
        assert!(
            context.contains(&format!(
                "{}{exception})",
                rocm_core::VLLM_OOM_DETECTED_PREFIX
            )),
            "the hint must name the exception row alone: {context:?}"
        );
        assert!(
            !context
                .chars()
                .any(|c| c.is_control() && c != '\n' && c != '\t'),
            "no cursor-moving control byte may survive into the printed failure message — \
             `\\t` is exempt, see the hint assertion above: {context:?}"
        );

        // Now the second read: rebuild the shared log exactly as the supervisor
        // leaves it (the raw engine output, then the CLI's `Error:` echo of this
        // bail) and confirm the selector still picks the engine's real row.
        let second_read = format!(
            "Loading weights: 90%\r{exception}\n\
             Error: vLLM server process exited before becoming ready (status: exit status: 1)\
             {context}\n"
        );
        assert_eq!(
            rocm_core::vllm_oom_diagnose_symptom(&second_read).as_deref(),
            Some(format!("vllm: {exception}").as_str()),
            "the welded echo must never become the symptom the user is told to paste"
        );
        Ok(())
    }

    /// Two branches `the_echoed_tail_never_welds…` cannot reach, because its
    /// fixture breaks on a bare `\r` and so its *selected segment* is already
    /// one row by the time either branch runs. Both survived a mutation run
    /// against the whole `process.rs` suite before this test existed:
    ///
    ///   - computing `oom_utilization_hint` from the sanitized summary instead
    ///     of the raw tail (103 passed, exit 0), and
    ///   - swapping `vllm_oom_display_line` for
    ///     `strip_terminal_control_sequences` (103 passed, exit 0).
    ///
    /// `ESC[2K` is the discriminator. [`rocm_core::terminal::rendered_lines`]
    /// ends a row on it, but `vllm_log_segments` does not split on it at all, so
    /// the segment reaching both branches really does carry two rows and an
    /// escape — which is what vLLM's colourised progress bar actually emits.
    #[test]
    fn the_hint_reads_the_raw_tail_and_displays_the_row_that_scored() -> Result<()> {
        let exception = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";
        let repaint = format!("Loading weights: 90%\u{1b}[2K{exception}");
        let path = std::env::temp_dir().join(format!(
            "rocm-vllm-repaint-{}-{}.log",
            std::process::id(),
            current_unix_millis()
        ));
        fs::write(&path, format!("{repaint}\n"))?;
        let context = startup_log_context(Some(path.as_path()));
        fs::remove_file(&path).ok();

        // Branch 1 — the hint is computed from the *raw* tail. There the segment
        // still carries the `ESC[2K`, so the quoting guard rejects it and the
        // printed command names the canonical symptom. Computed from the
        // sanitized summary instead, the escape is gone, the guard accepts, and
        // the exception would be quoted verbatim.
        assert!(
            context.contains(&format!(
                "`{} '{}'`",
                rocm_core::VLLM_OOM_DIAGNOSE_COMMAND,
                rocm_core::VLLM_OOM_CANONICAL_SYMPTOM
            )),
            "an escape-bearing segment must fall back to the canonical symptom: {context:?}"
        );
        assert!(
            !context.contains(&format!("'vllm: {exception}'")),
            "the raw segment is not quotable, so its rendered row must not be quoted: {context:?}"
        );

        // Branch 2 — the displayed line is the row that scored. Stripping the
        // segment instead welds the progress bar onto the exception, producing a
        // line vLLM never printed.
        assert!(
            context.contains(&format!(
                "{}{exception})",
                rocm_core::VLLM_OOM_DETECTED_PREFIX
            )),
            "the hint must display the scoring row alone: {context:?}"
        );
        assert!(
            !context.contains(&format!("Loading weights: 90%{exception}")),
            "the displayed line must never weld the progress bar onto the exception: {context:?}"
        );
        Ok(())
    }

    /// The end-to-end bound on the raw-split guarantee, measured rather than
    /// assumed — and the reason the obvious tightening is not available.
    ///
    /// `vllm_log_segments` keeps raw bytes so a colourised OOM line is refused
    /// by the quoting guard rather than printed as a stripped lookalike. That
    /// holds for each surface on the text it reads. It does not survive the
    /// engine's own bail: the echo here is necessarily sanitized (it goes to a
    /// terminal), the supervisor appends the bail to the same shared log, and
    /// the raw line then falls out of the CLI's `DEFAULT_LOG_TAIL_LINES` window.
    /// So the two surfaces can print different `--symptom` arguments for one
    /// failure. This pins both halves: that the difference exists and is
    /// harmless, and that filtering the echo out would cost the note entirely.
    #[test]
    fn the_colourised_round_trip_quotes_the_row_the_terminal_drew() -> Result<()> {
        let exception = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";
        let coloured = format!("\u{1b}[31m{exception}\u{1b}[0m");
        let mut log = String::new();
        for shard in 0..200 {
            writeln!(log, "INFO vllm.engine loading shard {shard}").expect("write to String");
        }
        writeln!(log, "{coloured}").expect("write to String");
        let path = std::env::temp_dir().join(format!(
            "rocm-vllm-roundtrip-{}-{}.log",
            std::process::id(),
            current_unix_millis()
        ));
        fs::write(&path, &log)?;

        // Read 1 — the engine. The raw segment carries escapes, so the guard
        // rejects it and the canonical symptom is printed.
        let context = startup_log_context(Some(path.as_path()));
        assert!(
            context.contains(&format!(
                "`{} '{}'`",
                rocm_core::VLLM_OOM_DIAGNOSE_COMMAND,
                rocm_core::VLLM_OOM_CANONICAL_SYMPTOM
            )),
            "the engine must refuse the colourised line and fall back: {context:?}"
        );

        // The supervisor appends the whole bail to the same log.
        fs::write(
            &path,
            format!(
                "{log}Error: vLLM server process exited before becoming ready \
                 (status: exit status: 1){context}\n"
            ),
        )?;
        let tail =
            rocm_core::tail_lines(&path, STARTUP_FAILURE_LOG_TAIL_LINES, MAX_TAIL_READ)?.join("\n");
        fs::remove_file(&path).ok();

        // The measurement the bound rests on: the engine's raw line is no longer
        // in the window the CLI reads. Only the sanitized echo of it is.
        assert!(
            !tail.contains(&coloured),
            "the raw colourised line must be out of the window, or this test proves nothing"
        );

        // Read 2 — the CLI. It quotes the rendered row, not the canonical
        // symptom. Different from read 1, and sound: the row is text the
        // terminal drew, it is quotable, and quotable implies diagnosable.
        let symptom = rocm_core::vllm_oom_diagnose_symptom(&tail)
            .expect("the echoed tail must still carry a detectable OOM");
        assert_eq!(symptom, format!("vllm: {exception}"));
        assert!(rocm_core::quotable_in_single_quotes(&symptom));
        assert!(rocm_core::vllm_oom_symptom_is_diagnosable(&symptom));

        // And the half that forecloses the tightening. The echo is a full tail
        // window plus the hint, so by the time the CLI reads, the window has
        // scrolled past the `Startup log, last N lines` header entirely: it holds
        // echoed rows and rocm-cli's own hint, and nothing the engine wrote
        // directly. A filter anchored on that header therefore never fires.
        let header = format!(
            "Startup log, last {STARTUP_FAILURE_LOG_TAIL_LINES} lines as the terminal drew them:"
        );
        assert!(
            !tail.contains(header.as_str()),
            "the echo header must have scrolled out, or the window still holds raw engine output"
        );
        // Anchored on the hint instead and scanning newest-first — the only
        // other marker available — suppression swallows the whole window and
        // the serve summary's note disappears for every real managed launch.
        let mut inside_echo = false;
        let still_selectable: Vec<&str> = tail
            .lines()
            .flat_map(|line| line.split('\r'))
            .map(str::trim)
            .filter(|segment| !segment.is_empty())
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .filter(|segment| {
                if segment.contains(rocm_core::VLLM_OOM_DIAGNOSE_COMMAND) {
                    inside_echo = true;
                }
                !inside_echo
            })
            .collect();
        assert!(
            still_selectable
                .iter()
                .all(|segment| rocm_core::vllm_oom_diagnose_symptom(segment).is_none()),
            "keeping the echo out of the CLI's selection leaves no OOM line at all: \
             {still_selectable:?}"
        );
        Ok(())
    }

    /// Pins a **known gap**, not desired behaviour: when the OOM line sits near
    /// the start of the window this engine scans, the `rocm serve` summary's
    /// note disappears even though this engine diagnosed the OOM.
    ///
    /// Both surfaces use the same tail *budget*
    /// (`rocm_engine_protocol::DEFAULT_LOG_TAIL_LINES`), but not the same
    /// window. This engine scans the last N raw lines and bails; the supervisor
    /// appends that whole bail — a header, the tail echoed as rendered rows, and
    /// the hint — to the same shared log; the CLI then reads N lines of the
    /// *longer* file. So the CLI's window always reaches less far back into the
    /// original log than this one did, and when the bail is itself longer than N
    /// rows it holds nothing the engine wrote directly. With the OOM line
    /// followed by 77 further lines it is diagnosed here and invisible there.
    ///
    /// Closing it is a design decision rather than a tweak, which is why it is
    /// pinned instead of fixed. Re-windowing the CLI to the N lines *before* the
    /// bail would restore agreement, but the raw colourised line comes back into
    /// view with it and the note stops quoting the user's own rendered row —
    /// trading this gap for the regression
    /// `the_colourised_round_trip_quotes_the_row_the_terminal_drew` exists to
    /// prevent. Recording the engine's selected symptom as structured data
    /// alongside the log would remove the class. **Update this test when the gap
    /// closes; do not delete it.**
    #[test]
    fn the_cli_window_can_miss_an_oom_the_engine_saw() -> Result<()> {
        let exception = "torch.OutOfMemoryError: HIP out of memory. Tried to allocate 7.21 GiB.";
        let mut log = String::new();
        for shard in 0..200 {
            writeln!(log, "INFO vllm.engine loading shard {shard}").expect("write to String");
        }
        writeln!(log, "{exception}").expect("write to String");
        // Enough trailing lines that the OOM still sits inside this engine's
        // window (78 lines from the end, budget 80) but far enough from the end
        // that the appended bail pushes its echo out of the CLI's.
        for worker in 0..77 {
            writeln!(log, "INFO vllm.engine shutting down worker {worker}")
                .expect("write to String");
        }
        let path = std::env::temp_dir().join(format!(
            "rocm-vllm-window-{}-{}.log",
            std::process::id(),
            current_unix_millis()
        ));
        fs::write(&path, &log)?;

        // Read 1 — the engine. It sees the OOM and says so.
        let context = startup_log_context(Some(path.as_path()));
        assert!(
            context.contains(rocm_core::VLLM_OOM_DETECTED_PREFIX),
            "the engine must diagnose this OOM, or the test proves nothing: {context:?}"
        );

        // The supervisor appends the whole bail to the same log.
        fs::write(
            &path,
            format!(
                "{log}Error: vLLM server process exited before becoming ready \
                 (status: exit status: 1){context}\n"
            ),
        )?;
        let tail =
            rocm_core::tail_lines(&path, STARTUP_FAILURE_LOG_TAIL_LINES, MAX_TAIL_READ)?.join("\n");
        fs::remove_file(&path).ok();

        // Read 2 — the CLI, same budget, later file. The gap.
        assert!(
            rocm_core::vllm_oom_diagnose_symptom(&tail).is_none(),
            "this test pins the known window gap; if the CLI now finds the OOM the gap is \
             closed — update the test, `append_oom_serve_note`'s comment and the README's \
             \"absence is not evidence\" list together: {tail:?}"
        );
        Ok(())
    }

    #[test]
    fn oom_utilization_hint_stays_quiet_for_unrelated_failures() {
        let unrelated = "ValueError: model architecture 'FooForCausalLM' is not supported";
        assert!(!detects_oom(unrelated));
        assert!(
            oom_utilization_hint(unrelated).is_empty(),
            "non-OOM failures must not carry a memory hint"
        );

        // vLLM's generic EngineCore wrapper is the terminal line for *any*
        // startup crash (unsupported arch, shm size, TP misconfig, missing
        // weights, OOM, ...); treating it as an OOM signature would misreport
        // those unrelated failures as memory exhaustion.
        let wrapper_only = "ERROR Engine core initialization failed";
        assert!(!detects_oom(wrapper_only));
        assert!(
            oom_utilization_hint(wrapper_only).is_empty(),
            "the generic EngineCore wrapper alone must not be treated as OOM"
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
            rocm_sdk_version: None,
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
    fn stop_scope_includes_vllm_workers() {
        assert_eq!(STOP_SCOPE, rocm_core::KillScope::Tree);
    }
    #[test]
    #[cfg(target_os = "linux")]
    fn apply_therock_env_actually_runs_the_compat_shim_step() -> Result<()> {
        // The stale-numa-shim removal `ensure_therock_compat_shims` performs
        // does not depend on host state — it always removes a `libnuma.so.1`
        // symlink it finds — so assert on that as proof the call actually
        // happened. See `apply_therock_env_writes_libmpi_cxx_stub_when_needed`
        // below for the other half of that call, the mpi_cxx stub write,
        // which does depend on host state and needs its own oracle.
        let env = tempfile::tempdir()?;
        let compat_dir = env.path().join("rocm-cli-lib-compat");
        fs::create_dir_all(&compat_dir)?;
        let stale_shim = compat_dir.join("libnuma.so.1");
        std::os::unix::fs::symlink("/does/not/exist", &stale_shim)?;

        let runtime = VllmRuntime {
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            env_id: "external-vllm-therock".to_owned(),
            command: PathBuf::from("vllm"),
            python_executable: Some(env.path().join("bin").join("python")),
            version: None,
            source: "managed_runtime_manifest:test".to_owned(),
            sdk_root: Some(env.path().join("sdk")),
            sdk_bin: None,
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
            rocm_sdk_version: None,
        };
        let mut command = ProcessCommand::new("vllm");

        apply_therock_env(&mut command, &runtime)?;

        assert!(
            stale_shim.symlink_metadata().is_err(),
            "ensure_therock_compat_shims must have removed the stale numa shim"
        );
        Ok(())
    }
    #[test]
    #[cfg(target_os = "linux")]
    fn apply_therock_env_writes_libmpi_cxx_stub_when_needed() -> Result<()> {
        // Whether this host needs the `libmpi_cxx.so.40` stub depends on host
        // state this test does not control: a real `libmpi_cxx.so*` already
        // present, or no stub embedded at compile time (no C compiler on the
        // build host). Use `ensure_mpi_cxx_compat` itself as the oracle,
        // against a throwaway directory so probing does not touch the
        // directory this test actually asserts on. When it answers `None`
        // here, `apply_therock_env`'s own call cannot write a stub either, and
        // there is nothing this test can check on this host.
        let probe = tempfile::tempdir()?;
        if rocm_core::openmpi::ensure_mpi_cxx_compat(probe.path()).is_none() {
            return Ok(());
        }

        let env = tempfile::tempdir()?;
        let runtime = VllmRuntime {
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            env_id: "external-vllm-therock".to_owned(),
            command: PathBuf::from("vllm"),
            python_executable: Some(env.path().join("bin").join("python")),
            version: None,
            source: "managed_runtime_manifest:test".to_owned(),
            sdk_root: Some(env.path().join("sdk")),
            sdk_bin: None,
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
            rocm_sdk_version: None,
        };
        let mut command = ProcessCommand::new("vllm");

        apply_therock_env(&mut command, &runtime)?;

        let compat_dir = env.path().join("rocm-cli-lib-compat");
        let stub = compat_dir.join("libmpi_cxx.so.40");
        assert!(
            stub.is_file(),
            "ensure_therock_compat_shims must have written the libmpi_cxx stub"
        );

        let ld_library_path = command
            .get_envs()
            .find_map(|(key, value)| (key == "LD_LIBRARY_PATH").then_some(value))
            .flatten()
            .expect("LD_LIBRARY_PATH set");
        assert!(
            std::env::split_paths(ld_library_path).any(|entry| entry == compat_dir),
            "ensure_therock_compat_shims's answer must reach LD_LIBRARY_PATH: \
             {ld_library_path:?} should contain {compat_dir:?}"
        );
        Ok(())
    }
    #[test]
    #[cfg(target_os = "linux")]
    fn apply_therock_env_excludes_a_leftover_stub_this_launch_does_not_need() -> Result<()> {
        // The opposite host state from
        // `apply_therock_env_writes_libmpi_cxx_stub_when_needed`, made
        // deterministic (rather than dependent on this host's own OpenMPI
        // install or compiler) by injecting the `ensure_mpi_cxx_compat`
        // answer through `apply_therock_env_with`: when this launch's probe
        // says no shim is needed, a stub some earlier run left behind in
        // `compat_dir` must not keep that directory on the loader path.
        let env = tempfile::tempdir()?;
        let runtime = VllmRuntime {
            runtime_id: "therock-release:gfx120X-all".to_owned(),
            env_id: "external-vllm-therock".to_owned(),
            command: PathBuf::from("vllm"),
            python_executable: Some(env.path().join("bin").join("python")),
            version: None,
            source: "managed_runtime_manifest:test".to_owned(),
            sdk_root: Some(env.path().join("sdk")),
            sdk_bin: None,
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
            rocm_sdk_version: None,
        };

        // Plant the leftover directly, bypassing `ensure_mpi_cxx_compat`:
        // this is what a prior launch leaves behind on a host that has since
        // gained a real `libmpi_cxx.so*`.
        let compat_dir = env.path().join("rocm-cli-lib-compat");
        fs::create_dir_all(&compat_dir)?;
        fs::write(compat_dir.join("libmpi_cxx.so.40"), b"stale stub")?;

        let mut command = ProcessCommand::new("vllm");

        apply_therock_env_with(&mut command, &runtime, |_compat_dir| None)?;

        let ld_library_path = command
            .get_envs()
            .find_map(|(key, value)| (key == "LD_LIBRARY_PATH").then_some(value))
            .flatten()
            .expect("LD_LIBRARY_PATH set");
        assert!(
            !std::env::split_paths(ld_library_path).any(|entry| entry == compat_dir),
            "a leftover stub must not keep the compat dir on the loader path \
             when this launch does not need one: {ld_library_path:?} should \
             not contain {compat_dir:?}"
        );
        Ok(())
    }
    fn test_vllm_runtime(
        python_executable: Option<PathBuf>,
        rocm_sdk_version: Option<String>,
    ) -> VllmRuntime {
        VllmRuntime {
            runtime_id: "test".to_owned(),
            env_id: "test".to_owned(),
            command: PathBuf::from("vllm"),
            python_executable,
            version: None,
            source: "test".to_owned(),
            sdk_root: None,
            sdk_bin: None,
            sdk_bin_paths: Vec::new(),
            sdk_library_paths: Vec::new(),
            rocm_sdk_version,
        }
    }
    #[test]
    fn vllm_venv_root_fails_without_a_python_executable() {
        let runtime = test_vllm_runtime(None, None);
        let error = vllm_venv_root(&runtime)
            .expect_err("no python executable")
            .to_string();
        assert!(error.contains("no python executable"), "{error}");
    }
    #[test]
    fn vllm_venv_root_derives_the_venv_from_bin_python() {
        let runtime = test_vllm_runtime(Some(PathBuf::from("/opt/venv/bin/python")), None);
        assert_eq!(
            vllm_venv_root(&runtime).unwrap(),
            PathBuf::from("/opt/venv")
        );
    }
    #[test]
    fn venv_site_packages_dir_finds_the_single_python3_dir() -> Result<()> {
        let venv = tempfile::tempdir()?;
        fs::create_dir_all(
            venv.path()
                .join("lib")
                .join("python3.12")
                .join("site-packages"),
        )?;
        let site_packages = venv_site_packages_dir(venv.path())?;
        assert_eq!(
            site_packages,
            venv.path()
                .join("lib")
                .join("python3.12")
                .join("site-packages")
        );
        Ok(())
    }
    #[test]
    fn venv_site_packages_dir_fails_with_zero_or_multiple_python3_dirs() -> Result<()> {
        let venv = tempfile::tempdir()?;
        fs::create_dir_all(venv.path().join("lib"))?;
        let error = venv_site_packages_dir(venv.path())
            .expect_err("no python3.* dir")
            .to_string();
        assert!(error.contains("no python3.*"), "{error}");

        fs::create_dir_all(venv.path().join("lib").join("python3.11"))?;
        fs::create_dir_all(venv.path().join("lib").join("python3.12"))?;
        let error = venv_site_packages_dir(venv.path())
            .expect_err("ambiguous python3.* dirs")
            .to_string();
        assert!(error.contains("found 2 python3.*"), "{error}");
        Ok(())
    }
    #[test]
    fn apply_vllm_rocm10_discover_env_sets_pythonpath_and_flash_attention_flag() -> Result<()> {
        let venv = tempfile::tempdir()?;
        fs::create_dir_all(
            venv.path()
                .join("lib")
                .join("python3.12")
                .join("site-packages"),
        )?;
        let runtime = test_vllm_runtime(
            Some(venv.path().join("bin").join("python")),
            Some("10.0.0".to_owned()),
        );
        let mut command = ProcessCommand::new("vllm");
        apply_vllm_rocm10_discover_env(&mut command, &runtime)?;

        let pythonpath = command
            .get_envs()
            .find_map(|(key, value)| (key == "PYTHONPATH").then_some(value))
            .flatten()
            .expect("PYTHONPATH set");
        let expected_entry = venv
            .path()
            .join("lib")
            .join("python3.12")
            .join("site-packages")
            .join("_rocm_sdk_core")
            .join("share")
            .join("amd_smi");
        assert!(
            std::env::split_paths(pythonpath).any(|entry| entry == expected_entry),
            "{pythonpath:?} should contain {expected_entry:?}"
        );

        let flag = command
            .get_envs()
            .find_map(|(key, value)| (key == "FLASH_ATTENTION_TRITON_AMD_ENABLE").then_some(value))
            .flatten();
        assert_eq!(flag, Some(std::ffi::OsStr::new("TRUE")));
        Ok(())
    }
    #[test]
    fn apply_therock_env_dispatches_to_discover_env_for_a_discover_rocm_sdk_version() -> Result<()>
    {
        let venv = tempfile::tempdir()?;
        fs::create_dir_all(
            venv.path()
                .join("lib")
                .join("python3.12")
                .join("site-packages"),
        )?;
        let runtime = test_vllm_runtime(
            Some(venv.path().join("bin").join("python")),
            Some("10.0.0".to_owned()),
        );
        let mut command = ProcessCommand::new("vllm");
        apply_therock_env(&mut command, &runtime)?;

        let flag = command
            .get_envs()
            .find_map(|(key, value)| (key == "FLASH_ATTENTION_TRITON_AMD_ENABLE").then_some(value))
            .flatten();
        assert_eq!(flag, Some(std::ffi::OsStr::new("TRUE")));
        Ok(())
    }
}
