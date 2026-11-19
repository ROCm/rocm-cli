// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, anyhow, bail};
use rocm_core::{
    AppPaths, TENSORIZER_PYTHON314_FIX, command_failure_detail, ensure_uv_binary,
    split_local_version, tensorizer_override_requirement, uv_command_env, uv_pip_freeze_args,
    uv_pip_install_base,
};
use rocm_engine_protocol::{InstallRequest, InstallResponse};
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

use crate::discover_builds::{
    DiscoverVersion, TorchStack, VllmRocmDiscoverBuild, rocm_sdk_series_matches, rocm_series,
    vllm_rocm_discover_build,
};
use crate::repair::{RepairAssessment, assess_runtime_repair};
use crate::runtime::{
    VllmRuntime, assessed_python_for_repair, describe_skipped_managed_runtimes,
    recorded_sdk_index_url, resolve_managed_runtime_python, resolve_vllm_runtime,
    runtime_is_managed, vllm_runtime_warnings,
};
use crate::state::runtime_lock_hash;
use crate::wheel_builds::{
    VllmInstallRoute, VllmInstallTarget, vllm_install_route, vllm_install_target,
};

pub(crate) fn install_response(request: InstallRequest) -> Result<InstallResponse> {
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
    let mut discover_pins: Vec<String> = Vec::new();
    let mut discover_notes: Vec<String> = Vec::new();
    // True whenever this run installed vLLM: a first install, `--reinstall` or a
    // repair pass. Only a runtime that was resolved as already installed skips it.
    let installed_vllm_now = already_installed.is_none();
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
        let outcome = install_vllm_with_uv(
            &managed.python_executable,
            request.reinstall,
            managed.rocm_sdk_version.as_deref(),
        )?;
        discover_pins = outcome.pins;
        discover_notes = outcome.notes;
        resolve_vllm_runtime(Some(&managed.runtime_id)).with_context(|| {
            format!(
                "vLLM install completed in {}, but runtime `{}` still could not be resolved",
                managed.python_executable.display(),
                managed.runtime_id
            )
        })?
    };
    discover_notes.extend(ensure_runtime_tensorizer(&runtime, installed_vllm_now)?);
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
        capabilities: crate::capabilities(),
        lock_hash: runtime_lock_hash(&runtime),
        warnings: repair
            .notes
            .into_iter()
            .chain(vllm_runtime_warnings(&runtime))
            .chain(discover_notes)
            .chain((!discover_pins.is_empty()).then(|| {
                format!(
                    "vLLM ROCm 10.x discovery pinned: {}",
                    discover_pins.join(", ")
                )
            }))
            .collect(),
    })
}
/// What the tensorizer import probe found.
#[derive(Debug, PartialEq, Eq)]
enum TensorizerProbe {
    /// `tensorizer` imports, is not installed (vLLM guards that case itself), or
    /// the import failed for a reason that is not tensorizer's (a torch or
    /// shared-library failure, which the runtime device check reports).
    Usable,
    /// `tensorizer` is installed and raises on import; carries the exception line.
    Broken(String),
}

/// How long the import probe and the upgrade may run before they are abandoned.
const TENSORIZER_STEP_TIMEOUT: std::time::Duration = std::time::Duration::from_mins(5);

/// The path of the last `File "..."` frame in a Python traceback.
fn last_traceback_frame(stderr: &str) -> Option<&str> {
    stderr.lines().rev().find_map(|line| {
        let rest = line.trim_start().strip_prefix("File \"")?;
        rest.split_once('"').map(|(path, _)| path)
    })
}

/// Whether `path` lies inside a `tensorizer` package directory, with either
/// path separator.
fn path_is_in_tensorizer_package(path: &str) -> bool {
    path.contains("/tensorizer/") || path.contains("\\tensorizer\\")
}

/// Judges the stderr of a failed `import tensorizer.serialization`.
///
/// The import pulls in torch, so a torch or shared-library failure has tensorizer
/// frames in its traceback too. A failure is tensorizer's own when the last frame
/// before the exception is inside the tensorizer package and the exception is not
/// a library-load or missing-module error (`ImportError`, `OSError`): that is the
/// shape of the Python 3.14 `KeyError` and of any other tensorizer code that does
/// not run on this interpreter. Everything else is left to the device check.
fn classify_tensorizer_failure(stderr: &str) -> TensorizerProbe {
    // Python 3.13+ colourises tracebacks under `FORCE_COLOR` even when piped, which
    // wraps the quoted path in `File "..."` in escapes and defeats the frame
    // parser. The probe asks for no colour; this is the second line of defence.
    let stderr = stderr
        .lines()
        .map(rocm_core::strip_terminal_control_sequences)
        .collect::<Vec<_>>()
        .join("\n");
    let stderr = stderr.as_str();
    let Some(exception) = stderr.lines().rev().find(|line| !line.trim().is_empty()) else {
        return TensorizerProbe::Usable;
    };
    let exception = exception.trim();
    let kind = exception.split([':', ' ']).next().unwrap_or_default();
    let is_load_failure = kind.ends_with("ImportError")
        || kind.ends_with("OSError")
        || kind == "ModuleNotFoundError"
        || exception.contains("shared object file")
        || exception.contains("DLL load failed");
    let in_tensorizer = last_traceback_frame(stderr).is_some_and(path_is_in_tensorizer_package);
    if in_tensorizer && !is_load_failure {
        TensorizerProbe::Broken(exception.to_owned())
    } else {
        TensorizerProbe::Usable
    }
}

/// The command that imports what vLLM imports from `tensorizer` in `python`.
///
/// Colour is switched off (a forced-colour environment makes the interpreter wrap
/// the traceback's quoted paths in escape codes), and with `runtime` the command
/// carries the environment the server is launched with (SDK library path,
/// `PYTHONPATH` for a ROCm 10.x discover build, and the rest), so a runtime that
/// needs it to import torch is judged as it will run.
fn tensorizer_probe_command(
    python: &Path,
    runtime: Option<&VllmRuntime>,
) -> Result<ProcessCommand> {
    let mut command = ProcessCommand::new(python);
    // Not `-I`: that implies `-E` and would drop the `PYTHONPATH` (and the rest of
    // the `PYTHON*` environment) the launch sets for a ROCm 10.x discover build,
    // so a module the server finds would read as missing. `-s` keeps the user's
    // site directory out, and `PYTHONSAFEPATH` keeps the working directory off
    // `sys.path` (Python 3.11+; older interpreters ignore it).
    command
        .args(["-s", "-c", "import tensorizer.serialization"])
        .env("PYTHONSAFEPATH", "1");
    if let Some(runtime) = runtime {
        // A launch environment that cannot be built is a failure of the check, not
        // a runtime that imports: probing without it could read as `Usable` a
        // failure the server would not have.
        crate::process::apply_therock_env(&mut command, runtime)
            .context("could not build the environment the server is launched with")?;
    }
    command
        .env_remove("FORCE_COLOR")
        .env("NO_COLOR", "1")
        .env("PYTHON_COLORS", "0");
    Ok(command)
}

/// Imports what vLLM imports from `tensorizer` in `python`.
///
/// vLLM's model loader imports `tensorizer.serialization` at startup and only
/// guards `ImportError`, so any other exception raised inside tensorizer ends
/// the engine core. With `runtime`, the probe runs under the environment the
/// server is launched with (SDK library path and the rest), so a runtime that
/// needs it to import torch is judged as it will run.
fn probe_tensorizer_import(
    python: &Path,
    runtime: Option<&VllmRuntime>,
) -> Result<TensorizerProbe> {
    probe_tensorizer_import_within(python, runtime, TENSORIZER_STEP_TIMEOUT)
}

/// [`probe_tensorizer_import`] with the deadline as a parameter, so a test can
/// exercise the timeout without waiting minutes.
fn probe_tensorizer_import_within(
    python: &Path,
    runtime: Option<&VllmRuntime>,
    limit: std::time::Duration,
) -> Result<TensorizerProbe> {
    let command = tensorizer_probe_command(python, runtime)?;
    let output = rocm_core::process::run_with_timeout(command, limit, "tensorizer import probe")
        .with_context(|| format!("failed to import tensorizer with {}", python.display()))?;
    if output.status.success() {
        return Ok(TensorizerProbe::Usable);
    }
    Ok(classify_tensorizer_failure(&String::from_utf8_lossy(
        &output.stderr,
    )))
}

/// Installs the fixed tensorizer over vLLM's pin, leaving everything else as it is.
fn upgrade_tensorizer(uv: &Path, paths: &AppPaths, python: &Path) -> Result<()> {
    upgrade_tensorizer_within(uv, paths, python, TENSORIZER_STEP_TIMEOUT)
}

/// [`upgrade_tensorizer`] with the deadline as a parameter, so a test can
/// exercise the timeout without waiting minutes.
fn upgrade_tensorizer_within(
    uv: &Path,
    paths: &AppPaths,
    python: &Path,
    limit: std::time::Duration,
) -> Result<()> {
    let mut args = uv_pip_install_base(python);
    args.push("--no-deps".to_owned());
    args.push(tensorizer_override_requirement());
    let mut command = ProcessCommand::new(uv);
    command.args(&args).envs(uv_command_env(paths));
    let output =
        rocm_core::process::run_with_timeout(command, limit, "uv pip install for tensorizer")
            .context("failed to run uv pip install for tensorizer")?;
    if output.status.success() {
        return Ok(());
    }
    bail!(
        "`uv {}` failed: {}",
        args.join(" "),
        command_failure_detail(&output)
    )
}

/// The decision behind [`ensure_runtime_tensorizer`], with the probe and the
/// upgrade injected.
///
/// Only a managed runtime is rocm-cli's to change, so an external one is not
/// probed at all. The probe runs first so an interpreter the pinned release works
/// on keeps exactly what vLLM asked for.
///
/// Three things can go wrong after a broken tensorizer is found: the upgrade
/// fails, the probe cannot run, or the upgrade leaves tensorizer unimportable.
/// Each is an error when `fatal` (this run installed vLLM, whether a first
/// install, `--reinstall` or a repair, so the runtime would be left unable to
/// serve or unchecked) and otherwise a note carrying the same facts, so a no-op
/// install on a runtime this run did not touch (an offline host, say) stays one.
/// Returns the note describing what was done, if anything.
fn tensorizer_step(
    managed: bool,
    fatal: bool,
    mut probe: impl FnMut() -> Result<TensorizerProbe>,
    upgrade: impl FnOnce() -> Result<()>,
) -> Result<Option<String>> {
    if !managed {
        return Ok(None);
    }
    // A probe that cannot run (spawn failure, timeout) is a failure of the check,
    // not a finding. It fails when this run installed vLLM, which must not report
    // a runtime it could not look at. On a runtime it did not touch it is a note,
    // so a host where the check cannot run keeps the no-op install it had before.
    let unchecked = |error: anyhow::Error| -> Result<Option<String>> {
        if fatal {
            Err(error)
        } else {
            Ok(Some(format!(
                "could not check whether vLLM's tensorizer imports: {error:#}"
            )))
        }
    };
    let reason = match probe() {
        Ok(TensorizerProbe::Broken(reason)) => reason,
        Ok(TensorizerProbe::Usable) => return Ok(None),
        Err(error) => return unchecked(error),
    };
    let requirement = tensorizer_override_requirement();
    if let Err(error) = upgrade() {
        let message = format!(
            "vLLM's pinned tensorizer cannot be imported ({reason}) and upgrading it to \
             {requirement} failed: {error:#}"
        );
        return if fatal {
            Err(anyhow::anyhow!("vLLM was installed, but {message}"))
        } else {
            Ok(Some(format!("{message}; vLLM would fail at startup")))
        };
    }
    let after = match probe() {
        Ok(after) => after,
        Err(error) => return unchecked(error),
    };
    match after {
        TensorizerProbe::Usable => Ok(Some(format!(
            "vLLM pins a tensorizer that cannot be imported under this Python ({reason}); \
             upgraded it to {TENSORIZER_PYTHON314_FIX}, so `uv pip check` reports vLLM's pin as \
             unmet on purpose"
        ))),
        TensorizerProbe::Broken(after) => {
            // An upgrade can fail to help (the exception is not the one 2.12.1
            // fixes). When this run installed vLLM that leaves a runtime that
            // cannot serve, so it fails; on a runtime this run did not touch the
            // install used to be a no-op and stays one, with the same facts as a note.
            let message = format!(
                "tensorizer still cannot be imported after upgrading it to {requirement}: \
                 {after}. vLLM would fail at startup, so this runtime cannot serve with vLLM."
            );
            if fatal {
                bail!("vLLM was installed, but {message}")
            }
            Ok(Some(message))
        }
    }
}

/// Makes sure vLLM's `tensorizer` dependency imports in `runtime`, upgrading it
/// when it does not.
///
/// vLLM's wheel pins `tensorizer==2.10.1`, which raises
/// `KeyError: 'crypto_stream_salsa20_NONCEBYTES'` at import under Python 3.14
/// (its constants metaclass reads annotations from the class namespace, where
/// 3.14 no longer puts them; fixed in tensorizer 2.12.1). The ROCm 10.1.0 vLLM
/// build exists for Python 3.14 only, so the pin cannot be avoided by choosing
/// another Python, and an install that reports success would end in an engine
/// core crash at the first `rocm serve`. Runs after an install pass (`installed_now`)
/// and on an existing runtime, so one installed before this check existed is
/// repaired by the next `rocm engines install vllm`.
fn ensure_runtime_tensorizer(runtime: &VllmRuntime, installed_now: bool) -> Result<Option<String>> {
    ensure_runtime_tensorizer_with(runtime, installed_now, |python| {
        let paths = AppPaths::discover()?;
        let uv =
            ensure_uv_binary(&paths).context("failed to acquire uv binary for vLLM install")?;
        upgrade_tensorizer(&uv, &paths, python)
    })
}

/// [`ensure_runtime_tensorizer`] with the upgrade injected, so a test can drive
/// the real probe (and its launch environment) against a stand-in interpreter.
fn ensure_runtime_tensorizer_with(
    runtime: &VllmRuntime,
    installed_now: bool,
    upgrade: impl FnOnce(&Path) -> Result<()>,
) -> Result<Option<String>> {
    // The managed check lives in `tensorizer_step`, once, so the rule has one home.
    let Some(python) = runtime.python_executable.as_deref() else {
        return Ok(None);
    };
    tensorizer_step(
        runtime_is_managed(runtime),
        installed_now,
        || probe_tensorizer_import(python, Some(runtime)),
        || upgrade(python),
    )
}

/// What a vLLM install route resolved: the pins worth surfacing as an install
/// warning, and a note for each place the route fell back to a weaker choice
/// than the one it prefers. Both are empty for the static-pin route.
#[derive(Debug, Default)]
struct VllmInstallOutcome {
    pins: Vec<String>,
    notes: Vec<String>,
}
/// Installs vLLM into `python`, returning the pins and fallback notes worth
/// surfacing to the caller as install warnings.
fn install_vllm_with_uv(
    python: &Path,
    reinstall: bool,
    rocm_sdk_version: Option<&str>,
) -> Result<VllmInstallOutcome> {
    let index_override = std::env::var("ROCM_CLI_VLLM_ROCM_INDEX_URL").ok();
    let index_override = index_override
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());

    match vllm_install_route(index_override, rocm_sdk_version) {
        VllmInstallRoute::RocmDiscover => {
            let paths = AppPaths::discover()?;
            let uv =
                ensure_uv_binary(&paths).context("failed to acquire uv binary for vLLM install")?;
            // Unwrap is safe: `vllm_install_route` only returns `RocmDiscover`
            // when `rocm_sdk_version` looks up a build in the table.
            let build = rocm_sdk_version.and_then(vllm_rocm_discover_build).expect(
                "vllm_install_route returned RocmDiscover without a matching discover build",
            );
            install_vllm_rocm10_discover(&uv, &paths, python, reinstall, build)
        }
        VllmInstallRoute::Static => {
            let paths = AppPaths::discover()?;
            let uv =
                ensure_uv_binary(&paths).context("failed to acquire uv binary for vLLM install")?;
            let VllmInstallTarget {
                index_url,
                requirement,
            } = vllm_install_target(rocm_sdk_version)?;
            let mut args = uv_pip_install_base(python);
            // Without `--reinstall`, `uv pip install vllm` is a no-op when the
            // wheel is already present, which would silently turn a requested
            // reinstall into a no-op. Force the reinstall so the caller's
            // intent is honored.
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
                return Ok(VllmInstallOutcome::default());
            }
            let detail = command_failure_detail(&output);
            bail!(
                "`uv pip install {} --extra-index-url {}` failed for {}: {}",
                requirement,
                index_url,
                python.display(),
                detail
            )
        }
    }
}
/// Resolves the exact requirement `uv` would install for `pkg` constrained
/// by `version` (see [`DiscoverVersion::requirement`]) from `index_url`,
/// without installing anything.
///
/// `uv` has no `pip download` command (and never has — it's a declined
/// upstream feature request, astral-sh/uv#3163), so this uses `uv pip
/// install --dry-run` instead: it runs the real resolver against `python`'s
/// platform/interpreter tags and reports the version it would install on a
/// ` + {pkg}==<version>` line, which is parsed back out by
/// [`dry_run_resolved_pin`]. A version with no compatible build published
/// surfaces as a resolver failure (never fall back to unpinned PyPI).
///
/// `--reinstall` is always passed here (independent of the caller's own
/// `reinstall` request, which governs the *real* install below): without it,
/// a dry-run against a package already present in `python` prints `Would
/// make no changes` with no ` + {pkg}==<version>` line at all, so a
/// discovery pin could never be resolved for an environment being repaired.
fn discover_pinned_requirement(
    uv: &Path,
    paths: &AppPaths,
    python: &Path,
    index_url: &str,
    pkg: &str,
    version: DiscoverVersion,
) -> Result<String> {
    discover_pinned_requirement_for(uv, paths, python, index_url, pkg, &version.requirement(pkg))
}

/// [`discover_pinned_requirement`] for a requirement already spelled out, so a
/// version learned at run time (not only a [`DiscoverVersion`] baked into a
/// build row) can be confirmed against `index_url` the same way.
fn discover_pinned_requirement_for(
    uv: &Path,
    paths: &AppPaths,
    python: &Path,
    index_url: &str,
    pkg: &str,
    requirement: &str,
) -> Result<String> {
    let output = ProcessCommand::new(uv)
        .args([
            "pip",
            "install",
            "--dry-run",
            "--reinstall",
            "--no-deps",
            "--index-url",
        ])
        .arg(index_url)
        .args(["--prerelease", "allow", "--python"])
        .arg(python)
        .arg(requirement)
        .envs(uv_command_env(paths))
        .output()
        .with_context(|| format!("failed to launch uv pip install --dry-run for {pkg}"))?;
    // `uv pip install --dry-run` writes its whole human-readable resolution
    // report — including the ` + {pkg}==<version>` line this parses — to
    // stderr; stdout is empty on both success and failure. Check both so a
    // future `uv` that moves the report back to stdout keeps working too.
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    if !output.status.success() {
        let detail = command_failure_detail(&output);
        bail!("`uv pip install --dry-run {requirement}` from {index_url} failed: {detail}");
    }
    dry_run_resolved_pin(&stderr, pkg)
        .or_else(|| dry_run_resolved_pin(&stdout, pkg))
        .ok_or_else(|| {
            let reported = if stderr.trim().is_empty() {
                stdout.trim()
            } else {
                stderr.trim()
            };
            anyhow!(
                "`uv pip install --dry-run {requirement}` from {index_url} did not report a \
                 resolved version for {pkg}: {reported}"
            )
        })
}
/// The row's torch, torchvision and torchaudio pins, in that order.
///
/// A [`TorchStack::Discover`] row resolves them from its index. A
/// [`TorchStack::FromInstalledSdk`] row takes them from the runtime the SDK
/// install already resolved, because the index it realigns from serves
/// several ROCm lines at once and a series such as `torch==2.14.*` would pick
/// the newest line's build (see [`TorchStack`]).
fn torch_stack_pins(
    uv: &Path,
    paths: &AppPaths,
    python: &Path,
    build: &VllmRocmDiscoverBuild,
) -> Result<Vec<String>> {
    match build.torch_stack {
        TorchStack::Discover => ["torch", "torchvision", "torchaudio"]
            .into_iter()
            .map(|pkg| {
                let version = match pkg {
                    "torch" => DiscoverVersion::Series(build.torch_version_prefix),
                    "torchvision" => DiscoverVersion::Series(build.torchvision_version_prefix),
                    _ => build.torchaudio_version,
                };
                discover_pinned_requirement(uv, paths, python, build.torch_index_url, pkg, version)
            })
            .collect(),
        TorchStack::FromInstalledSdk => {
            let output = ProcessCommand::new(uv)
                .args(uv_pip_freeze_args(python))
                .envs(uv_command_env(paths))
                .output()
                .context("failed to launch `uv pip freeze` to read the SDK's torch stack")?;
            if !output.status.success() {
                bail!(
                    "`uv pip freeze` failed for {}: {}",
                    python.display(),
                    String::from_utf8_lossy(&output.stderr).trim()
                );
            }
            installed_torch_stack_pins(&String::from_utf8_lossy(&output.stdout), build)
        }
    }
}

/// Whether an installed `release` (a `uv pip freeze` version, with any `+local` part)
/// satisfies `wanted`: a series takes any release inside it, an exact pin takes that
/// release and no longer one, with the same meaning as PEP 440 `==release`.
fn stack_release_matches(release: &str, wanted: DiscoverVersion) -> bool {
    match wanted {
        DiscoverVersion::Series(series) => release.starts_with(&format!("{series}.")),
        DiscoverVersion::Exact(exact) => split_local_version(release).0 == exact,
    }
}
/// [`torch_stack_pins`] for a [`TorchStack::FromInstalledSdk`] row, from the
/// `uv pip freeze` output of the SDK runtime.
///
/// Each pin must belong to the row's torch, torchvision and torchaudio series
/// and be built for the row's ROCm line, so a runtime whose torch is not the
/// one the vLLM wheel was built against is refused instead of silently kept.
fn installed_torch_stack_pins(freeze: &str, build: &VllmRocmDiscoverBuild) -> Result<Vec<String>> {
    let wanted = [
        ("torch", DiscoverVersion::Series(build.torch_version_prefix)),
        (
            "torchvision",
            DiscoverVersion::Series(build.torchvision_version_prefix),
        ),
        ("torchaudio", build.torchaudio_version),
    ];
    wanted
        .into_iter()
        .map(|(pkg, wanted_release)| {
            let prefix = format!("{pkg}==");
            let pin = freeze
                .lines()
                .map(str::trim)
                .find(|line| line.starts_with(&prefix))
                .with_context(|| {
                    format!(
                        "the ROCm SDK runtime has no `{pkg}` installed; vLLM for ROCm {} reuses \
                         the SDK's own torch stack, so reinstall the SDK with `rocm install sdk`",
                        build.rocm_sdk_version
                    )
                })?;
            let release = &pin[prefix.len()..];
            if !stack_release_matches(release, wanted_release) {
                bail!(
                    "the ROCm SDK runtime has `{pin}`, but vLLM for ROCm {} needs `{pkg}` {}",
                    build.rocm_sdk_version,
                    match wanted_release {
                        DiscoverVersion::Series(series) => format!("{series}.x"),
                        DiscoverVersion::Exact(release) => release.to_owned(),
                    }
                );
            }
            ensure_rocm_local_version_matches(pin, build.rocm_sdk_version)?;
            Ok(pin.to_owned())
        })
        .collect()
}
/// Runs the dependency-resolving `uv pip install --dry-run` for the discovered
/// vLLM pin and returns uv's report, without installing anything.
///
/// vLLM's wheel declares exact versions of the sibling wheels it was built
/// against (`amd-aiter==<dev build>`), and the index keeps publishing newer
/// siblings that those vLLM builds do not accept. Picking each package's newest
/// release independently, as the series match does, can therefore name a
/// combination the vLLM wheel itself rules out; the resolver's own answer for
/// vLLM is the source of truth for what pairs with it. The resolution uses the
/// same default index and the same 403-tolerant config file as the real install
/// that follows, but not the same flags (`--reinstall` for the whole set, and no
/// torch pins), so it is a close approximation of that install, not a guarantee
/// about it. Only the companion lines of its output are consumed.
fn resolve_vllm_dependencies(
    uv: &Path,
    paths: &AppPaths,
    python: &Path,
    vllm_pin: &str,
    config_file: &Path,
) -> Result<String> {
    let mut args = uv_pip_install_base(python);
    args.extend(
        [
            "--dry-run",
            "--reinstall",
            "--prerelease",
            "allow",
            "--config-file",
        ]
        .map(str::to_owned),
    );
    args.push(config_file.display().to_string());
    args.push(vllm_pin.to_owned());
    let output = ProcessCommand::new(uv)
        .args(&args)
        .envs(uv_command_env(paths))
        .output()
        .context("failed to launch uv pip install --dry-run to resolve vLLM's dependencies")?;
    if !output.status.success() {
        bail!(
            "`uv {}` failed: {}",
            args.join(" "),
            command_failure_detail(&output)
        );
    }
    Ok(format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    ))
}

/// The pin for a package that ships alongside vLLM (`flash-attn`,
/// `amd-aiter`): the version vLLM's own dependencies resolve to when the vLLM
/// index serves that exact build, otherwise the newest release of `series`.
///
/// The resolver's answer is only taken after the vLLM index is asked for that
/// exact version on its own (`--index-url`, no PyPI): an unpinned dependency
/// can resolve to a public PyPI release, which is the non-ROCm build this
/// whole route exists to avoid, and such an answer falls through to the series
/// match as before.
///
/// `vllm_resolution` is `None` when the resolve probe itself failed. A
/// resolved pin the vLLM index does not confirm is not fatal either, but the
/// reason comes back with the pin: a transient uv or network failure looks the
/// same as "the index does not serve it", and both end in the series pin. The
/// confirm call is tried twice, because on ROCm 10.1 the series pin is the pair
/// that cannot be installed together, so a single transient failure should not
/// be enough to land on it.
fn discover_vllm_companion_pin(
    uv: &Path,
    paths: &AppPaths,
    python: &Path,
    build: &VllmRocmDiscoverBuild,
    vllm_resolution: Option<&str>,
    pkg: &str,
    series: DiscoverVersion,
) -> Result<(String, Option<String>)> {
    let mut fallback_note = None;
    if let Some(requirement) =
        vllm_resolution.and_then(|resolved| dry_run_resolved_pin(resolved, pkg))
    {
        let confirm = || {
            discover_pinned_requirement_for(
                uv,
                paths,
                python,
                build.vllm_index_url,
                pkg,
                &requirement,
            )
        };
        match confirm().or_else(|_| confirm()) {
            Ok(pin) => return Ok((pin, None)),
            Err(err) => {
                fallback_note = Some(format!(
                    "vLLM's dependencies resolve {pkg} to `{requirement}`, but the vLLM index \
                     did not confirm that build ({err:#}); using the newest {pkg} of its series \
                     instead"
                ));
            }
        }
    }
    let pin = discover_pinned_requirement(uv, paths, python, build.vllm_index_url, pkg, series)?;
    Ok((pin, fallback_note))
}

/// Parses the ` + {pkg}==<version>` line `uv pip install --dry-run` prints
/// for each package it would install, returning it as a `{pkg}==<version>`
/// requirement.
fn dry_run_resolved_pin(stdout: &str, pkg: &str) -> Option<String> {
    let prefix = format!("+ {pkg}==");
    stdout.lines().find_map(|line| {
        let version = line.trim_start().strip_prefix(&prefix)?;
        (!version.is_empty()).then(|| format!("{pkg}=={version}"))
    })
}
/// `err` with the fallbacks this install used appended, so a failure on a fallback
/// names it. The cause comes first, so plain `{}` display leads with the uv failure and
/// not with a note about how the pins or the index were chosen.
fn with_fallback_notes(err: anyhow::Error, notes: &[String]) -> anyhow::Error {
    if notes.is_empty() {
        err
    } else {
        anyhow!(
            "{err:#}\n\nFallbacks used during this install: {}",
            notes.join(" ")
        )
    }
}
/// The index the torch stack is fetched from for `build`.
///
/// A [`TorchStack::FromInstalledSdk`] row reuses the SDK's own builds, so it asks for
/// them from the index the SDK came from (`recorded`, from the runtime manifest) rather
/// than the row's fixed one: the row serves nightly and release SDKs alike, and each
/// one's torch lives only on its own index. A manifest that recorded none, and every
/// row that discovers its stack, keeps the row's `torch_index_url`.
fn torch_index_url_for(build: &VllmRocmDiscoverBuild, recorded: Option<&str>) -> String {
    match (build.torch_stack, recorded) {
        (TorchStack::FromInstalledSdk, Some(url)) => url.to_owned(),
        _ => build.torch_index_url.to_owned(),
    }
}
/// Builds the `uv pip install` argv for pinned torch alone, scoped to only
/// `torch_index_url`.
///
/// Mixing `vllm_index_url` into the same call (as a prior version of this
/// code did) makes `uv` probe torch's package name under the wrong index
/// too; AMD's CloudFront/S3-backed discovery hosts answer a nonexistent
/// sub-path with a fatal 403 rather than a fall-through-safe 404, which
/// aborts the whole resolution before `uv` ever reaches the index that
/// actually has torch.
///
/// Uses `--index-url`, not `--extra-index-url`: the latter is kept
/// pip-compatible by `uv`, meaning it is checked *after* the implicit
/// default PyPI index, not before it. This call installs a single exact
/// pin with `--no-deps`, so there is nothing legitimate for PyPI to supply;
/// leaving it reachable only risks `uv` silently satisfying the pin from a
/// same-numbered public PyPI release instead of AMD's build.
///
/// `--no-deps` matters just as much here as it does in
/// [`discover_pinned_requirement`]: this torch wheel's own metadata pins an
/// exact `rocm[libraries]==<version>` dependency, naming the ROCm release
/// its ROCm libs were built against. That release is managed separately by
/// `rocm install sdk`, which can legitimately be a different version track
/// (e.g. a nightly build) than this discovery row's pin, so the `rocm`
/// project is never published under this index at that exact version;
/// resolving it here fails the whole install instead of leaving the
/// already-installed SDK alone.
fn vllm_rocm10_discover_torch_install_args(
    python: &Path,
    reinstall: bool,
    torch_index_url: &str,
    torch_pin: &str,
) -> Vec<String> {
    let mut args = uv_pip_install_base(python);
    if reinstall {
        args.push("--reinstall-package".to_owned());
        args.push("torch".to_owned());
    }
    args.push(torch_pin.to_owned());
    args.push("--no-deps".to_owned());
    args.push("--prerelease".to_owned());
    args.push("allow".to_owned());
    args.push("--index-url".to_owned());
    args.push(torch_index_url.to_owned());
    args
}
/// Builds the `uv pip install` argv that realigns torch, torchvision and
/// torchaudio back to `pins` after the full-dependency install, all
/// `--no-deps` and scoped to `torch_index_url`.
///
/// `pins` is always the torch/torchvision/torchaudio trio discovered fresh
/// from this row's `torch_index_url`, so they are guaranteed coherent with
/// each other even when they weren't with whatever the SDK install resolved
/// first. A single call realigns the whole stack together rather than one
/// `uv` invocation per package.
///
/// Uses `--index-url`, for the same reason
/// [`vllm_rocm10_discover_torch_install_args`] does: torch's pin always
/// carries a `+rocmX.Y` local version with no PyPI equivalent, but
/// torchvision/torchaudio do not, so `--extra-index-url`'s lower-than-default
/// priority would let a same-numbered public PyPI wheel silently win over
/// AMD's build here, reproducing the exact ABI mismatch this realign step
/// exists to prevent.
fn vllm_rocm10_discover_realign_install_args(
    python: &Path,
    torch_index_url: &str,
    pins: &[String],
) -> Vec<String> {
    let mut args = uv_pip_install_base(python);
    for pin in pins {
        if let Some((name, _)) = pin.split_once("==") {
            args.push("--reinstall-package".to_owned());
            args.push(name.to_owned());
        }
    }
    args.extend(pins.iter().cloned());
    args.push("--no-deps".to_owned());
    args.push("--prerelease".to_owned());
    args.push("allow".to_owned());
    args.push("--index-url".to_owned());
    args.push(torch_index_url.to_owned());
    args
}
/// Builds the `uv pip install` argv for pinned vllm/flash-attn/amd-aiter,
/// scoped to only `vllm_index_url`. tensorizer is not pinned here: vllm's own
/// wheel metadata already declares an exact tensorizer dependency, and a
/// second, independently chosen pin for the same package risks conflicting
/// with it.
///
/// See [`vllm_rocm10_discover_torch_install_args`] for why torch is never
/// mixed into this same call. This call resolves full dependencies (unlike
/// torch's `--no-deps` call), which can pull in an unconstrained `torch` from
/// PyPI and silently undo the exact pin just installed; callers must re-run
/// [`vllm_rocm10_discover_torch_install_args`] afterwards to realign it.
///
/// `vllm_index_url` is supplied via `--config-file`, not `--extra-index-url`,
/// here: `uv` does not merge a CLI `--extra-index-url` with a config-file
/// `[[index]]` entry for the same URL, so a CLI copy would keep 403ing fatally
/// even with the config file's `ignore-error-codes` also active.
fn vllm_rocm10_discover_remaining_install_args(
    python: &Path,
    reinstall: bool,
    pins: &[String],
) -> Vec<String> {
    let mut args = uv_pip_install_base(python);
    if reinstall {
        for pin in pins {
            if let Some((name, _)) = pin.split_once("==") {
                args.push("--reinstall-package".to_owned());
                args.push(name.to_owned());
            }
        }
    }
    args.extend(pins.iter().cloned());
    args.push("--prerelease".to_owned());
    args.push("allow".to_owned());
    args
}
/// Writes a scratch `uv.toml` that keeps `uv` searching other indexes when
/// `vllm_index_url` 403s on a package it doesn't host, instead of aborting
/// resolution outright.
///
/// `uv`'s default index-strategy treats a 403 from any configured index as
/// fatal, not as "not found here, try the next index" — the docs note this is
/// normally only special-cased for the pytorch index. AMD's staging index is a
/// CDN-backed static index with the same 403-for-missing-package behavior, so
/// vllm's own plain-PyPI transitive dependencies (e.g. `lm-format-enforcer`)
/// need the same `ignore-error-codes` treatment here. There is no CLI flag or
/// env var for it, only this config file.
fn write_vllm_index_uv_config(paths: &AppPaths, build: &VllmRocmDiscoverBuild) -> Result<PathBuf> {
    std::fs::create_dir_all(&paths.cache_dir)
        .with_context(|| format!("failed to create {}", paths.cache_dir.display()))?;
    let path = paths.cache_dir.join("vllm-rocm10-uv.toml");
    let contents = format!(
        "[[index]]\nurl = \"{}\"\nignore-error-codes = [403]\n",
        build.vllm_index_url
    );
    std::fs::write(&path, contents)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(path)
}
/// Runs a `uv pip install` invocation, surfacing a failure with the full
/// command and `uv`'s own output.
fn run_uv_pip_install(uv: &Path, paths: &AppPaths, python: &Path, args: Vec<String>) -> Result<()> {
    let output = ProcessCommand::new(uv)
        .args(&args)
        .envs(uv_command_env(paths))
        .output()
        .context("failed to launch uv pip install for vLLM (ROCm 10.x discovery)")?;
    if output.status.success() {
        return Ok(());
    }
    let detail = command_failure_detail(&output);
    bail!(
        "`uv pip install {}` failed for {}: {}",
        args.join(" "),
        python.display(),
        detail
    )
}
/// Rejects a discovered pin whose `+rocm<version>` local version names a
/// different ROCm line than the SDK being installed for.
///
/// Discovery constrains the *release* (`vllm==0.29.*`) but cannot constrain the
/// local version, because PEP 440 has no local-version wildcard. Both discovery
/// indexes serve more than one ROCm line at once (staging carries 10.0 and 10.1
/// vLLM side by side), so without this a mis-stocked or re-pointed index could
/// resolve a 10.0 wheel onto a 10.1 SDK and only fail at import time.
/// `flash-attn` and `amd-aiter` publish no rocm local version at all, so a pin
/// without one passes.
fn ensure_rocm_local_version_matches(pin: &str, rocm_sdk_version: &str) -> Result<()> {
    let Some((_, version)) = pin.split_once("==") else {
        return Ok(());
    };
    let Some(local) = split_local_version(version).1 else {
        return Ok(());
    };
    let Some(matches) = local_rocm_matches(local, rocm_sdk_version) else {
        return Ok(());
    };
    if matches {
        return Ok(());
    }
    bail!(
        "discovered `{pin}` for ROCm SDK {rocm_sdk_version}, but its `+{local}` build targets a \
         different ROCm release; the index for this ROCm line is serving another SDK's wheels"
    )
}
/// Whether the ROCm line a local version names is `rocm_sdk_version`'s, or `None` when
/// it has no `rocm` segment at all. A dotted `rocm10.2.0` is read from its first `rocm` segment to the end
/// of the local version, wherever it sits; a compact `rocm102` is the digits of that
/// segment alone, so `rocm102` as the whole local version and `d20261110.rocm102` after
/// other segments are both read.
fn local_rocm_matches(local: &str, rocm_sdk_version: &str) -> Option<bool> {
    let segments: Vec<&str> = local.split('.').collect();
    let at = segments
        .iter()
        .position(|segment| segment.starts_with("rocm"))?;
    let dotted = segments[at..].join(".");
    if let Some(recorded) = dotted.strip_prefix("rocm")
        && rocm_series(recorded).is_some()
    {
        return Some(rocm_sdk_series_matches(recorded, rocm_sdk_version));
    }
    // `.rocm102` names 10.2 with no dot between major and minor. A `rocm` segment
    // that is neither spelling (`+rocm`, `+rocmabc`) names no line this SDK can be
    // matched to, so it is a mismatch, not a pass.
    Some(compact_rocm_local_segment(local).is_some_and(|compact| {
        rocm_series(rocm_sdk_version)
            .is_some_and(|(major, minor)| compact == format!("{major}{minor}"))
    }))
}
/// The digits of a `rocm<digits>` segment in a dotted local version, e.g.
/// `102` for `g5df63a720.d20261110.rocm102`: the spelling the 10.2 vLLM wheels
/// use, with no dot between the ROCm major and minor.
fn compact_rocm_local_segment(local: &str) -> Option<&str> {
    local
        .split('.')
        .filter_map(|segment| segment.strip_prefix("rocm"))
        .find(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
}
/// The PEP 425 interpreter tag of `python`, e.g. `cp314`.
fn python_interpreter_tag(python: &Path) -> Result<String> {
    let output = ProcessCommand::new(python)
        .args([
            "-c",
            "import sys; print(f'cp{sys.version_info[0]}{sys.version_info[1]}')",
        ])
        .output()
        .with_context(|| format!("failed to launch {} to read its version", python.display()))?;
    if !output.status.success() {
        bail!(
            "{} could not report its version: {}",
            python.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}
/// Fails before discovery when `python` is not the interpreter this row's
/// wheels are built for.
///
/// Checked up front rather than left to `uv`: every package in the row is
/// published for one tag only, so a mismatched interpreter surfaces as three
/// separate "no compatible version found" resolver dumps that name the
/// requirement but never the interpreter, which is the thing to fix.
fn ensure_discover_python_tag(python: &Path, build: &VllmRocmDiscoverBuild) -> Result<()> {
    let tag = python_interpreter_tag(python)?;
    if tag == build.python_tag {
        return Ok(());
    }
    bail!(
        "vLLM for ROCm {} is published for {} only, but {} is {}.\n\
         Re-create this runtime's Python environment on {}, for example by re-running \
         `rocm install sdk --version {}` so ROCm CLI provisions a matching interpreter.",
        build.rocm_sdk_version,
        build.python_tag,
        python.display(),
        tag,
        build.python_tag,
        build.rocm_sdk_version
    )
}
/// Lists every package `uv pip freeze` reports as installed in `python`'s
/// environment, as bare names (no version).
fn installed_package_names(uv: &Path, paths: &AppPaths, python: &Path) -> Result<Vec<String>> {
    let output = ProcessCommand::new(uv)
        .args(uv_pip_freeze_args(python))
        .envs(uv_command_env(paths))
        .output()
        .context("failed to launch `uv pip freeze` to find installed packages")?;
    if !output.status.success() {
        bail!(
            "`uv pip freeze` failed for {}: {}",
            python.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.split_once("==").map(|(name, _)| name.to_owned()))
        .collect())
}
/// Whether `name` is one of AMD's per-GPU-architecture device-kernel plugin
/// packages for `base_pkg` (e.g. `amd-torch-device-gfx1151` or
/// `amd-torchvision-device-gfx115x` for `base_pkg` `torch`/`torchvision`).
/// Each is versioned in lockstep with its own `base_pkg`, not with torch.
fn is_amd_device_package_for(name: &str, base_pkg: &str) -> bool {
    name.to_ascii_lowercase()
        .replace('_', "-")
        .starts_with(&format!("amd-{base_pkg}-device-gfx"))
}
/// Discovers and installs the current vLLM/flash-attn/amd-aiter wheels for a
/// [`VllmRocmDiscoverBuild`] row, pinning each to the exact version `uv pip
/// install --dry-run` resolved so the real install can never silently drift
/// to a different (or non-ROCm) build. Returns the pins actually installed,
/// and a note for each companion pin that fell back to the series match.
fn install_vllm_rocm10_discover(
    uv: &Path,
    paths: &AppPaths,
    python: &Path,
    reinstall: bool,
    build: &VllmRocmDiscoverBuild,
) -> Result<VllmInstallOutcome> {
    let mut notes: Vec<String> = Vec::new();
    // Every failure after a fallback was taken names it, wherever it happens: a note
    // kept only for the success path would be lost exactly when the fallback it
    // describes is what failed.
    match install_vllm_rocm10_discover_collecting(uv, paths, python, reinstall, build, &mut notes) {
        Ok(pins) => Ok(VllmInstallOutcome { pins, notes }),
        Err(err) => Err(with_fallback_notes(err, &notes)),
    }
}
/// [`install_vllm_rocm10_discover`]'s sequence, recording each fallback it takes in
/// `notes` as it goes and returning the pins installed.
fn install_vllm_rocm10_discover_collecting(
    uv: &Path,
    paths: &AppPaths,
    python: &Path,
    reinstall: bool,
    build: &VllmRocmDiscoverBuild,
    notes: &mut Vec<String>,
) -> Result<Vec<String>> {
    ensure_discover_python_tag(python, build)?;
    // Discovered (or read from the SDK runtime) together, as one coherent set,
    // rather than snapshotted from whatever the SDK install resolved earlier
    // for a row that discovers: that snapshot can pair with a *different*
    // torch version than the one this row pins, producing an ABI-incompatible
    // triplet at realign time.
    let mut torch_stack = torch_stack_pins(uv, paths, python, build)?.into_iter();
    let (torch, torchvision, torchaudio) = (
        torch_stack.next().context("torch pin")?,
        torch_stack.next().context("torchvision pin")?,
        torch_stack.next().context("torchaudio pin")?,
    );
    let vllm = discover_pinned_requirement(
        uv,
        paths,
        python,
        build.vllm_index_url,
        "vllm",
        DiscoverVersion::Series(build.vllm_version_prefix),
    )?;
    let config_file = write_vllm_index_uv_config(paths, build)?;
    // The probe only improves on the series match, so a probe that fails does
    // not fail the install: the real install below is the authority, and it
    // reports its own error. The reason is kept, because the series match is
    // exactly the choice this probe exists to improve on.
    let vllm_resolution = match resolve_vllm_dependencies(uv, paths, python, &vllm, &config_file) {
        Ok(resolution) => Some(resolution),
        Err(err) => {
            notes.push(format!(
                "could not resolve the dependencies of {vllm} ({err:#}); flash-attn and \
                 amd-aiter use the newest release of their series, which may not be the \
                 build this vLLM was made against"
            ));
            None
        }
    };
    let (flash_attn, flash_attn_note) = discover_vllm_companion_pin(
        uv,
        paths,
        python,
        build,
        vllm_resolution.as_deref(),
        "flash-attn",
        DiscoverVersion::Series(build.flash_attn_version_prefix),
    )?;
    let (amd_aiter, amd_aiter_note) = discover_vllm_companion_pin(
        uv,
        paths,
        python,
        build,
        vllm_resolution.as_deref(),
        "amd-aiter",
        DiscoverVersion::Series(build.amd_aiter_version_prefix),
    )?;
    notes.extend(flash_attn_note);
    notes.extend(amd_aiter_note);

    let pins = vec![torch, torchvision, torchaudio, vllm, flash_attn, amd_aiter];
    for pin in &pins {
        ensure_rocm_local_version_matches(pin, build.rocm_sdk_version)?;
    }

    // Two separate `uv pip install` calls, each scoped to exactly one index:
    // see `vllm_rocm10_discover_torch_install_args` for why combining both
    // indexes into one call is unsafe.
    let torch_index_url = if build.torch_stack == TorchStack::FromInstalledSdk {
        let recorded_index = recorded_sdk_index_url(paths, python);
        match &recorded_index {
            Ok(Some(_)) => {}
            Ok(None) => notes.push(format!(
                "no runtime manifest for this interpreter records a wheel index; the torch stack is \
                 fetched from {}, which may not be the index the SDK was installed from",
                build.torch_index_url
            )),
            Err(err) => notes.push(format!(
                "could not read the runtime registry ({err:#}); the torch stack is fetched \
                 from {}, which may not be the index the SDK was installed from",
                build.torch_index_url
            )),
        }
        torch_index_url_for(build, recorded_index.ok().flatten().as_deref())
    } else {
        build.torch_index_url.to_owned()
    };
    let torch_args =
        vllm_rocm10_discover_torch_install_args(python, reinstall, &torch_index_url, &pins[0]);
    run_uv_pip_install(uv, paths, python, torch_args)?;

    let mut remaining_args =
        vllm_rocm10_discover_remaining_install_args(python, reinstall, &pins[3..]);
    remaining_args.push("--config-file".to_owned());
    remaining_args.push(config_file.display().to_string());
    run_uv_pip_install(uv, paths, python, remaining_args)?;

    // The full-dependency resolve above can replace torch (and, pulling it in
    // transitively, torchvision/torchaudio) with unconstrained PyPI builds;
    // force the whole stack back to the coherent trio discovered above. This
    // must run unconditionally, before the device-plugin discovery below: a
    // `uv pip freeze` failure there must not skip restoring the trio too.
    let trio_realign_args =
        vllm_rocm10_discover_realign_install_args(python, &torch_index_url, &pins[..3]);
    run_uv_pip_install(uv, paths, python, trio_realign_args)?;

    // The full-dependency resolve can also pull in an AMD device-kernel
    // plugin for any of the three (`amd-torch-device-gfx*`,
    // `amd-torchvision-device-gfx*`, ...) pinned to whatever the index's
    // newest release is, unconstrained by -- and so routinely ahead of --
    // the base package version just realigned above; the mismatch doesn't
    // fail here, it fails at serve time as a HIP "Cannot find Symbol" crash.
    // Realign any such installed device package to the exact version of its
    // own base package: a device package always shares its base package's
    // version number, not any other package's.
    let installed = installed_package_names(uv, paths, python)?;
    let mut device_pins: Vec<String> = Vec::new();
    for pin in &pins[..3] {
        // Fail closed rather than emit a `name==` with an empty release: every
        // other surprise on this install path bails too.
        let (base_pkg, release) = pin.split_once("==").with_context(|| {
            format!("discovered pin `{pin}` is not a `name==version` requirement")
        })?;
        device_pins.extend(
            installed
                .iter()
                .filter(|name| is_amd_device_package_for(name, base_pkg))
                .map(|name| format!("{name}=={release}")),
        );
    }
    if !device_pins.is_empty() {
        let device_realign_args =
            vllm_rocm10_discover_realign_install_args(python, &torch_index_url, &device_pins);
        run_uv_pip_install(uv, paths, python, device_realign_args)?;
    }

    let mut pins = pins;
    pins.extend(device_pins);
    Ok(pins)
}
#[cfg(test)]
mod tests {
    use super::*;
    const KEY_ERROR_UNIX: &str = "Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\n  File \"/venv/lib/python3.14/site-packages/tensorizer/serialization.py\", line 40, in <module>\n    from . import _crypt\n  File \"/venv/lib/python3.14/site-packages/tensorizer/_crypt/_encryption.py\", line 367, in __new__\n    type, NamedTuple(name, **{k: annotations[k] for k in entries})\nKeyError: 'crypto_stream_salsa20_NONCEBYTES'\n";

    #[test]
    fn the_python_314_key_error_is_a_broken_tensorizer() {
        assert_eq!(
            classify_tensorizer_failure(KEY_ERROR_UNIX),
            TensorizerProbe::Broken("KeyError: 'crypto_stream_salsa20_NONCEBYTES'".to_owned())
        );
    }

    #[test]
    fn a_windows_style_tensorizer_path_is_recognised() {
        let stderr = "Traceback (most recent call last):\n  File \"C:\\venv\\Lib\\site-packages\\tensorizer\\_crypt\\_encryption.py\", line 367, in __new__\nKeyError: 'x'\n";
        assert!(matches!(
            classify_tensorizer_failure(stderr),
            TensorizerProbe::Broken(_)
        ));
    }

    #[test]
    fn a_torch_failure_with_tensorizer_frames_is_not_a_broken_tensorizer() {
        // `import tensorizer.serialization` imports torch, so a missing HIP library
        // has tensorizer frames above the torch ones.
        let stderr = "Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\n  File \"/venv/lib/python3.14/site-packages/tensorizer/serialization.py\", line 12, in <module>\n    import torch\n  File \"/venv/lib/python3.14/site-packages/torch/__init__.py\", line 300, in <module>\n    _load_global_deps()\nOSError: libamdhip64.so.7: cannot open shared object file: No such file or directory\n";
        assert_eq!(classify_tensorizer_failure(stderr), TensorizerProbe::Usable);
    }

    #[test]
    fn a_non_load_exception_raised_below_tensorizer_is_not_a_broken_tensorizer() {
        // Tensorizer frames above, but the exception comes from torch and is not a
        // library-load error, so only the last-frame rule can acquit tensorizer.
        let stderr = "Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\n  File \"/venv/lib/python3.14/site-packages/tensorizer/serialization.py\", line 12, in <module>\n    import torch\n  File \"/venv/lib/python3.14/site-packages/torch/_utils.py\", line 80, in helper\n    return table[key]\nKeyError: 'gfx1151'\n";
        assert_eq!(classify_tensorizer_failure(stderr), TensorizerProbe::Usable);
    }

    /// Each case raises its exception with tensorizer as the LAST frame, so only
    /// the library-load rule can acquit it, and each carries exactly one of the
    /// rule's markers: the exception type or the message.
    #[test]
    fn each_library_load_marker_alone_acquits_a_tensorizer_frame() {
        let cases = [
            (
                "OSError type alone",
                "OSError: [Errno 2] No such file or directory: 'libsodium.so'",
            ),
            (
                "ImportError type alone",
                "ImportError: cannot import name 'x' from 'y'",
            ),
            (
                "ModuleNotFoundError type alone",
                "ModuleNotFoundError: No module named 'redis'",
            ),
            (
                "'shared object file' alone, wrapped in another type",
                "RuntimeError: libamdhip64.so.7: cannot open shared object file",
            ),
            (
                "'DLL load failed' alone, wrapped in another type",
                "RuntimeError: DLL load failed while importing _tensorizer: The specified module could not be found.",
            ),
        ];
        for (what, exception) in cases {
            let stderr = format!(
                "Traceback (most recent call last):\n  File \"/venv/site-packages/tensorizer/serialization.py\", line 3, in <module>\n{exception}\n"
            );
            assert_eq!(
                classify_tensorizer_failure(&stderr),
                TensorizerProbe::Usable,
                "{what}"
            );
        }
        // The same frame with a non-load exception is tensorizer's own failure.
        let broken = "Traceback (most recent call last):\n  File \"/venv/site-packages/tensorizer/serialization.py\", line 3, in <module>\nKeyError: 'x'\n";
        assert!(matches!(
            classify_tensorizer_failure(broken),
            TensorizerProbe::Broken(_)
        ));
    }

    #[test]
    fn an_absent_tensorizer_or_empty_stderr_is_not_a_finding() {
        let absent = "Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\nModuleNotFoundError: No module named 'tensorizer'\n";
        assert_eq!(classify_tensorizer_failure(absent), TensorizerProbe::Usable);
        assert_eq!(classify_tensorizer_failure(""), TensorizerProbe::Usable);
    }

    fn broken() -> TensorizerProbe {
        TensorizerProbe::Broken("KeyError".to_owned())
    }

    #[test]
    fn an_unmanaged_runtime_is_never_probed_or_changed() -> Result<()> {
        let note = tensorizer_step(
            false,
            true,
            || panic!("an environment rocm-cli does not own must not be probed"),
            || panic!("an environment rocm-cli does not own must not be changed"),
        )?;
        assert_eq!(note, None);
        let runtime = VllmRuntime {
            source: "environment python".to_owned(),
            python_executable: Some(PathBuf::from("/nonexistent/python")),
            ..VllmRuntime::for_test()
        };
        assert!(!runtime_is_managed(&runtime));
        assert_eq!(ensure_runtime_tensorizer(&runtime, true)?, None);
        Ok(())
    }

    #[test]
    fn a_usable_tensorizer_is_left_alone() -> Result<()> {
        let note = tensorizer_step(
            true,
            true,
            || Ok(TensorizerProbe::Usable),
            || panic!("a usable tensorizer must not be upgraded"),
        )?;
        assert_eq!(note, None);
        Ok(())
    }

    #[test]
    fn a_broken_tensorizer_is_upgraded_once_and_probed_again() -> Result<()> {
        let mut probes = vec![TensorizerProbe::Usable, broken()];
        let mut upgrades = 0;
        let note = tensorizer_step(
            true,
            true,
            || Ok(probes.pop().expect("probed more often than expected")),
            || {
                upgrades += 1;
                Ok(())
            },
        )?
        .expect("an override is reported");
        assert_eq!(upgrades, 1);
        assert!(probes.is_empty(), "the upgrade must be probed again");
        assert!(
            note.contains("2.12.1") && note.contains("on purpose"),
            "{note}"
        );
        Ok(())
    }

    #[test]
    fn an_upgrade_that_does_not_help_fails_an_install_this_run_made_and_only_warns_otherwise()
    -> Result<()> {
        let note = tensorizer_step(true, false, || Ok(broken()), || Ok(()))?
            .expect("a runtime this run did not install into gets a note");
        assert!(note.contains("still cannot be imported"), "{note}");
        assert!(
            !note.contains("was installed"),
            "no install ran on that runtime: {note}"
        );
        let error = tensorizer_step(true, true, || Ok(broken()), || Ok(()))
            .expect_err("an install whose tensorizer still cannot import must not succeed");
        assert!(
            format!("{error:#}").contains("vLLM was installed, but tensorizer still cannot"),
            "{error:#}"
        );
        Ok(())
    }

    #[test]
    fn a_failed_upgrade_fails_an_install_but_only_warns_on_an_existing_runtime() -> Result<()> {
        let fail = || -> Result<()> { bail!("network unreachable") };
        let error = tensorizer_step(true, true, || Ok(broken()), fail)
            .expect_err("an install must not leave an unusable runtime");
        // The message already carries the cause once; chaining the error as a
        // source as well would print it twice under `{:#}`.
        assert_eq!(
            format!("{error:#}").matches("network unreachable").count(),
            1,
            "{error:#}"
        );
        let note = tensorizer_step(true, false, || Ok(broken()), fail)?
            .expect("the failure is reported as a note");
        assert!(
            note.contains("network unreachable") && note.contains("would fail at startup"),
            "{note}"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn the_probe_reads_the_interpreters_traceback() -> Result<()> {
        use rocm_core::fake_executable::write_fake_executable;
        let root = tempfile::tempdir()?;
        let python = root.path().join("python");
        write_fake_executable(
            &python,
            format!("#!/bin/sh\ncat >&2 <<'EOF'\n{KEY_ERROR_UNIX}EOF\nexit 1\n"),
        )?;
        assert!(matches!(
            probe_tensorizer_import(&python, None)?,
            TensorizerProbe::Broken(_)
        ));
        Ok(())
    }

    /// A ROCm 10.x discover runtime whose stand-in interpreter, like a real one,
    /// does not see `PYTHONPATH` under `-I` (which implies `-E`), and without it
    /// stops on a missing module before it ever reaches tensorizer.
    #[cfg(unix)]
    #[test]
    fn the_probe_keeps_the_pythonpath_the_server_launches_with() -> Result<()> {
        use rocm_core::fake_executable::write_fake_executable;
        let venv = tempfile::tempdir()?;
        std::fs::create_dir_all(venv.path().join("bin"))?;
        std::fs::create_dir_all(venv.path().join("lib/python3.14/site-packages"))?;
        let python = venv.path().join("bin/python");
        write_fake_executable(
            &python,
            format!(
                "#!/bin/sh\ncase \" $* \" in *\" -I \"*) unset PYTHONPATH ;; esac\nif [ -z \"$PYTHONPATH\" ]; then\n  echo \"ModuleNotFoundError: No module named 'amdsmi'\" >&2\n  exit 1\nfi\ncat >&2 <<'EOF'\n{KEY_ERROR_UNIX}EOF\nexit 1\n"
            ),
        )?;
        let runtime = VllmRuntime {
            python_executable: Some(python),
            rocm_sdk_version: Some("10.1.0".to_owned()),
            ..VllmRuntime::for_test()
        };
        assert!(
            matches!(
                probe_tensorizer_import(
                    runtime.python_executable.as_deref().unwrap(),
                    Some(&runtime)
                )?,
                TensorizerProbe::Broken(_)
            ),
            "a missing module the server's PYTHONPATH provides must not hide a broken tensorizer"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn the_upgrade_is_a_no_deps_install_of_the_fixed_release() -> Result<()> {
        use rocm_core::fake_executable::write_fake_executable;
        let root = tempfile::tempdir()?;
        let paths = AppPaths {
            config_dir: root.path().join("config"),
            data_dir: root.path().join("data"),
            cache_dir: root.path().join("cache"),
        };
        let log = root.path().join("uv.log");
        let uv = root.path().join("uv");
        write_fake_executable(
            &uv,
            format!("#!/bin/sh\necho \"$@\" >> \"{}\"\n", log.display()),
        )?;
        upgrade_tensorizer(&uv, &paths, Path::new("/venv/bin/python"))?;
        let calls = std::fs::read_to_string(&log)?;
        assert!(
            calls.contains("--no-deps") && calls.contains("tensorizer==2.12.1"),
            "{calls}"
        );
        Ok(())
    }

    #[test]
    fn a_coloured_traceback_is_still_a_broken_tensorizer() {
        // What Python 3.13+ writes under FORCE_COLOR: escapes wrap the quoted path.
        let coloured = "\u{1b}[1;35mTraceback (most recent call last):\u{1b}[0m\n  File \u{1b}[35m\"/venv/lib/python3.14/site-packages/tensorizer/_crypt/_encryption.py\u{1b}[0m\", line \u{1b}[35m367\u{1b}[0m, in \u{1b}[35m__new__\u{1b}[0m\n\u{1b}[1;35mKeyError\u{1b}[0m: \u{1b}[35m'crypto_stream_salsa20_NONCEBYTES'\u{1b}[0m\n";
        assert_eq!(
            classify_tensorizer_failure(coloured),
            TensorizerProbe::Broken("KeyError: 'crypto_stream_salsa20_NONCEBYTES'".to_owned())
        );
    }

    #[test]
    fn the_probe_asks_for_no_colour_and_runs_in_the_launch_environment() {
        let runtime = VllmRuntime::for_test();
        let command = tensorizer_probe_command(Path::new("python"), Some(&runtime))
            .expect("a bare runtime has a launch environment");
        let env = |key: &str| {
            command
                .get_envs()
                .find(|(k, _)| *k == std::ffi::OsStr::new(key))
                .map(|(_, v)| v.map(std::ffi::OsStr::to_owned))
        };
        assert_eq!(
            env("FORCE_COLOR"),
            Some(None),
            "FORCE_COLOR must be removed"
        );
        assert_eq!(env("NO_COLOR"), Some(Some("1".into())));
        let args: Vec<_> = command.get_args().collect();
        assert!(
            !args.contains(&std::ffi::OsStr::new("-I"))
                && !args.contains(&std::ffi::OsStr::new("-E")),
            "-I and -E drop the PYTHON* environment the server launches with: {args:?}"
        );
        assert!(
            args.contains(&std::ffi::OsStr::new("-s")),
            "the user's site directory must stay out of the probe: {args:?}"
        );
        assert_eq!(
            env("PYTHONSAFEPATH"),
            Some(Some("1".into())),
            "the working directory must stay off sys.path"
        );
        assert_eq!(env("PYTHON_COLORS"), Some(Some("0".into())));
        assert_eq!(
            env("VLLM_TARGET_DEVICE"),
            Some(Some("rocm".into())),
            "the probe must carry the environment the server launches with"
        );
        let bare =
            tensorizer_probe_command(Path::new("python"), None).expect("no runtime to build");
        assert!(
            bare.get_envs()
                .all(|(k, _)| k != std::ffi::OsStr::new("VLLM_TARGET_DEVICE"))
        );
    }

    #[test]
    fn a_probe_that_cannot_run_fails_an_install_but_only_warns_on_an_existing_runtime() -> Result<()>
    {
        let cannot = || -> Result<TensorizerProbe> { bail!("timed out after 300s") };
        let error = tensorizer_step(true, true, cannot, || Ok(()))
            .expect_err("an install must not report a runtime it could not check");
        assert!(format!("{error:#}").contains("timed out"));
        let note = tensorizer_step(true, false, cannot, || Ok(()))?
            .expect("an existing runtime gets a note");
        assert!(
            note.contains("could not check") && note.contains("timed out"),
            "{note}"
        );

        // The second probe, after a successful upgrade, is held to the same rule.
        let mut calls = 0;
        let second_fails = || {
            calls += 1;
            if calls == 1 {
                Ok(broken())
            } else {
                bail!("spawn failed")
            }
        };
        let note = tensorizer_step(true, false, second_fails, || Ok(()))?
            .expect("an existing runtime gets a note");
        assert!(note.contains("could not check"), "{note}");
        Ok(())
    }

    /// A managed runtime whose interpreter is a stand-in that raises the Python
    /// 3.14 `KeyError` only when launched with the server's environment, and a
    /// torch-style `OSError` otherwise.
    #[cfg(unix)]
    fn runtime_needing_launch_env(root: &Path) -> Result<VllmRuntime> {
        use rocm_core::fake_executable::write_fake_executable;
        let python = root.join("python");
        write_fake_executable(
            &python,
            format!(
                "#!/bin/sh\nif [ \"$VLLM_TARGET_DEVICE\" = rocm ]; then\ncat >&2 <<'EOF'\n{KEY_ERROR_UNIX}EOF\nelse\necho 'OSError: libamdhip64.so.7: cannot open shared object file' >&2\nfi\nexit 1\n"
            ),
        )?;
        Ok(VllmRuntime {
            python_executable: Some(python),
            ..VllmRuntime::for_test()
        })
    }

    #[cfg(unix)]
    #[test]
    fn the_probe_sees_the_launch_environment_and_the_upgrade_follows() -> Result<()> {
        let root = tempfile::tempdir()?;
        let runtime = runtime_needing_launch_env(root.path())?;
        let mut upgraded = false;
        // Still broken after the "upgrade": the stand-in never changes, so the
        // refusal proves the probe ran with the launch environment both times.
        let error = ensure_runtime_tensorizer_with(&runtime, true, |_| {
            upgraded = true;
            Ok(())
        })
        .expect_err("the stand-in stays broken");
        assert!(
            upgraded,
            "the Python 3.14 failure was only visible with the launch environment"
        );
        assert!(format!("{error:#}").contains("still cannot be imported"));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_failed_upgrade_fails_an_install_this_run_made_and_only_warns_on_an_existing_runtime()
    -> Result<()> {
        let root = tempfile::tempdir()?;
        let runtime = runtime_needing_launch_env(root.path())?;
        let fail = |_: &Path| -> Result<()> { bail!("network unreachable") };
        let error = ensure_runtime_tensorizer_with(&runtime, true, fail)
            .expect_err("an install this run made must fail");
        assert!(format!("{error:#}").contains("network unreachable"));
        let note = ensure_runtime_tensorizer_with(&runtime, false, fail)?
            .expect("an existing runtime only gets a note");
        assert!(note.contains("network unreachable"), "{note}");
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_hung_probe_is_abandoned_promptly_and_its_child_killed() -> Result<()> {
        use rocm_core::fake_executable::write_fake_executable;
        let root = tempfile::tempdir()?;
        let pid_file = root.path().join("pid");
        let python = root.path().join("python");
        write_fake_executable(
            &python,
            format!(
                "#!/bin/sh\necho $$ > \"{}\"\nexec sleep 30\n",
                pid_file.display()
            ),
        )?;
        let started = std::time::Instant::now();
        let error =
            probe_tensorizer_import_within(&python, None, std::time::Duration::from_millis(300))
                .expect_err("a hung probe must be abandoned");
        assert!(format!("{error:#}").contains("exceeded"), "{error:#}");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(10),
            "the probe waited out the child: {:?}",
            started.elapsed()
        );
        let pid = std::fs::read_to_string(&pid_file)?;
        assert!(
            !Path::new(&format!("/proc/{}", pid.trim())).exists(),
            "the hung child was left running"
        );
        Ok(())
    }

    /// 5 MB on stderr before the real traceback: more than a pipe holds, so an
    /// undrained pipe would stall the child until the deadline.
    #[cfg(unix)]
    #[test]
    fn a_probe_that_prints_more_than_a_pipe_holds_is_still_classified() -> Result<()> {
        use rocm_core::fake_executable::write_fake_executable;
        let root = tempfile::tempdir()?;
        let python = root.path().join("python");
        write_fake_executable(
            &python,
            format!(
                "#!/bin/sh\nhead -c 5000000 /dev/zero | tr '\\0' 'x' >&2\necho >&2\ncat >&2 <<'EOF'\n{KEY_ERROR_UNIX}EOF\nexit 1\n"
            ),
        )?;
        let probe =
            probe_tensorizer_import_within(&python, None, std::time::Duration::from_secs(20))?;
        assert!(matches!(probe, TensorizerProbe::Broken(_)), "{probe:?}");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn an_upgrade_that_prints_more_than_a_pipe_holds_still_finishes() -> Result<()> {
        use rocm_core::fake_executable::write_fake_executable;
        let root = tempfile::tempdir()?;
        let paths = AppPaths {
            config_dir: root.path().join("config"),
            data_dir: root.path().join("data"),
            cache_dir: root.path().join("cache"),
        };
        let uv = root.path().join("uv");
        write_fake_executable(&uv, "#!/bin/sh\nhead -c 5000000 /dev/zero >&2\nexit 0\n")?;
        upgrade_tensorizer_within(
            &uv,
            &paths,
            Path::new("/venv/bin/python"),
            std::time::Duration::from_secs(20),
        )
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_hung_upgrade_is_abandoned_promptly_and_its_child_killed() -> Result<()> {
        use rocm_core::fake_executable::write_fake_executable;
        let root = tempfile::tempdir()?;
        let paths = AppPaths {
            config_dir: root.path().join("config"),
            data_dir: root.path().join("data"),
            cache_dir: root.path().join("cache"),
        };
        let pid_file = root.path().join("pid");
        let uv = root.path().join("uv");
        write_fake_executable(
            &uv,
            format!(
                "#!/bin/sh\necho $$ > \"{}\"\nexec sleep 30\n",
                pid_file.display()
            ),
        )?;
        let started = std::time::Instant::now();
        let error = upgrade_tensorizer_within(
            &uv,
            &paths,
            Path::new("/venv/bin/python"),
            std::time::Duration::from_millis(300),
        )
        .expect_err("a hung upgrade must be abandoned");
        assert!(format!("{error:#}").contains("exceeded"), "{error:#}");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(10),
            "the upgrade waited out the child: {:?}",
            started.elapsed()
        );
        let pid = std::fs::read_to_string(&pid_file)?;
        assert!(
            !Path::new(&format!("/proc/{}", pid.trim())).exists(),
            "the hung child was left running"
        );
        Ok(())
    }

    #[test]
    fn dry_run_resolved_pin_parses_a_rotated_dev_tag() {
        let stdout = "Resolved 1 package in 601ms\nWould download 1 package\nWould install 1 package\n + vllm==0.27.1.dev5+rocm10.0.0.gf46a9dfe2.d20260826\n";
        assert_eq!(
            dry_run_resolved_pin(stdout, "vllm"),
            Some("vllm==0.27.1.dev5+rocm10.0.0.gf46a9dfe2.d20260826".to_owned())
        );
    }
    #[test]
    fn dry_run_resolved_pin_parses_a_simple_version() {
        let stdout = "Resolved 1 package in 553ms\n + flash-attn==2.8.3\n";
        assert_eq!(
            dry_run_resolved_pin(stdout, "flash-attn"),
            Some("flash-attn==2.8.3".to_owned())
        );
    }
    #[test]
    fn dry_run_resolved_pin_ignores_other_packages_and_missing_lines() {
        let stdout = "Resolved 1 package in 553ms\n + amd-aiter==0.1.20.post1\n";
        assert_eq!(dry_run_resolved_pin(stdout, "vllm"), None);
        assert_eq!(dry_run_resolved_pin("no solution found", "vllm"), None);
    }
    #[test]
    fn vllm_rocm10_discover_torch_install_args_use_only_the_torch_index() {
        let python = PathBuf::from("/opt/venv/bin/python");
        let build = vllm_rocm_discover_build("10.0.0").expect("10.0.0 has a discover row");
        let torch_pin = "torch==2.12.0+rocm10.0.0";

        let args = vllm_rocm10_discover_torch_install_args(
            &python,
            false,
            build.torch_index_url,
            torch_pin,
        );
        assert!(!args.contains(&"--reinstall".to_owned()));
        assert!(!args.contains(&"--reinstall-package".to_owned()));
        assert!(args.contains(&torch_pin.to_owned()));
        assert!(args.contains(&"--no-deps".to_owned()));
        assert!(args.contains(&"--prerelease".to_owned()));
        assert!(args.contains(&"allow".to_owned()));
        assert!(args.contains(&build.torch_index_url.to_owned()));
        assert!(
            args.contains(&"--index-url".to_owned()),
            "{args:?} must use --index-url, not --extra-index-url, or a same-numbered public \
             PyPI wheel can silently outrank AMD's build"
        );
        assert!(!args.contains(&"--extra-index-url".to_owned()));
        assert!(
            !args.contains(&build.vllm_index_url.to_owned()),
            "{args:?} must never reference vllm_index_url, or `uv` may probe torch under it \
             and hit a fatal 403 from a nonexistent sub-path"
        );

        let args = vllm_rocm10_discover_torch_install_args(
            &python,
            true,
            build.torch_index_url,
            torch_pin,
        );
        assert!(args.contains(&"--reinstall-package".to_owned()));
        assert!(args.contains(&"torch".to_owned()));
        assert!(!args.contains(&"--reinstall".to_owned()));
    }
    #[test]
    fn an_installed_sdk_row_fetches_its_torch_stack_from_the_index_the_sdk_came_from() {
        let python = PathBuf::from("/opt/venv/bin/python");
        let release = "https://stable.repo.amd.com/rocm/whl-next";
        let ten_two = vllm_rocm_discover_build("10.2.0").expect("10.2 has a discover row");
        assert_eq!(ten_two.torch_stack, TorchStack::FromInstalledSdk);

        // A release SDK's torch is not on the row's nightly index, so the row's own
        // index must not be the one handed to uv.
        let index = torch_index_url_for(ten_two, Some(release));
        assert_eq!(index, release);
        assert_ne!(index, ten_two.torch_index_url);
        let pins = vec!["torch==2.14.0+rocm10.2.0".to_owned()];
        let install = vllm_rocm10_discover_torch_install_args(&python, false, &index, &pins[0]);
        let realign = vllm_rocm10_discover_realign_install_args(&python, &index, &pins);
        for args in [&install, &realign] {
            assert!(args.contains(&release.to_owned()), "{args:?}");
            assert!(
                !args.contains(&ten_two.torch_index_url.to_owned()),
                "{args:?}"
            );
        }

        // A manifest that recorded no index keeps the row's.
        assert_eq!(torch_index_url_for(ten_two, None), ten_two.torch_index_url);
        // A row that discovers its stack never takes the recorded index.
        let ten_one = vllm_rocm_discover_build("10.1.0").expect("10.1 has a discover row");
        assert_eq!(ten_one.torch_stack, TorchStack::Discover);
        assert_eq!(
            torch_index_url_for(ten_one, Some(release)),
            ten_one.torch_index_url
        );
    }
    #[test]
    fn vllm_rocm10_discover_remaining_install_args_use_only_the_vllm_index() {
        let pins = vec![
            "vllm==0.27.1.dev5+rocm10.0.0".to_owned(),
            "flash-attn==2.8.3".to_owned(),
            "amd-aiter==0.1.4".to_owned(),
            "tensorizer==2.12.1".to_owned(),
        ];
        let python = PathBuf::from("/opt/venv/bin/python");
        let build = vllm_rocm_discover_build("10.0.0").expect("10.0.0 has a discover row");

        let args = vllm_rocm10_discover_remaining_install_args(&python, false, &pins);
        assert!(!args.contains(&"--reinstall-package".to_owned()));
        for pin in &pins {
            assert!(args.contains(pin), "{args:?} should contain {pin}");
        }
        assert!(args.contains(&"--prerelease".to_owned()));
        assert!(args.contains(&"allow".to_owned()));
        assert!(
            !args.contains(&build.vllm_index_url.to_owned()),
            "{args:?} must supply vllm_index_url via --config-file, not --extra-index-url, or \
             `uv` 403s fatally on a package the index doesn't host despite ignore-error-codes"
        );
        assert!(
            !args.contains(&build.torch_index_url.to_owned()),
            "{args:?} must never reference torch_index_url, or `uv` may probe flash-attn/\
             amd-aiter under it and hit a fatal 403 from a nonexistent sub-path"
        );

        let args = vllm_rocm10_discover_remaining_install_args(&python, true, &pins);
        for pin in &pins {
            let name = pin.split_once("==").unwrap().0;
            assert!(args.contains(&"--reinstall-package".to_owned()));
            assert!(
                args.contains(&name.to_owned()),
                "{args:?} should reinstall {name}"
            );
        }
        assert!(!args.contains(&"--reinstall".to_owned()));
    }
    /// A stand-in `uv` that answers the two kinds of dry-run the discovery makes the
    /// way AMD's staging index did when ROCm 10.1's vLLM stopped installing: the
    /// index serves a newer `amd-aiter` (0.1.24.post1) than the one the newest
    /// vLLM 0.29 wheel declares (a 0.1.22 dev build), and the `--no-deps`
    /// single-index query serves only builds that index really holds.
    #[cfg(unix)]
    fn fake_uv_for_staging_index(dir: &Path) -> PathBuf {
        let script = r#"#!/bin/sh
for last; do :; done
case "$*" in
  *--no-deps*)
    case "$last" in
      amd-aiter==0.1.22.post2.dev0+gb4d9154d1.d20261002) echo " + $last" >&2 ;;
      amd-aiter==0.1.\*) echo " + amd-aiter==0.1.24.post1" >&2 ;;
      flash-attn==2.8.3|flash-attn==2.8.\*) echo " + flash-attn==2.8.3" >&2 ;;
      *) echo "error: no version of ${last} on this index" >&2; exit 1 ;;
    esac ;;
  *) exit 64 ;;
esac
"#;
        let path = dir.join("uv");
        rocm_core::fake_executable::write_fake_executable(&path, script).unwrap();
        path
    }

    #[cfg(unix)]
    #[test]
    fn companion_pins_follow_what_the_discovered_vllm_depends_on() -> Result<()> {
        // The ROCm 10.1 failure: installing `amd-aiter==0.1.24.post1` (the newest
        // release in the series) beside a vLLM wheel that depends on the 0.1.22 dev
        // build made `uv` report the pair unsatisfiable, so the reinstall failed.
        let root = tempfile::tempdir()?;
        let paths = AppPaths {
            config_dir: root.path().join("config"),
            data_dir: root.path().join("data"),
            cache_dir: root.path().join("cache"),
        };
        let uv = fake_uv_for_staging_index(root.path());
        let python = root.path().join("python");
        let build = vllm_rocm_discover_build("10.1.0").expect("10.1.0 has a discover row");
        let resolution = "Resolved 232 packages in 5.33s\n + amd-aiter==0.1.22.post2.dev0+gb4d9154d1.d20261002\n + flash-attn==2.8.3\n + torch==2.14.1\n";

        let (amd_aiter, note) = discover_vllm_companion_pin(
            &uv,
            &paths,
            &python,
            build,
            Some(resolution),
            "amd-aiter",
            DiscoverVersion::Series(build.amd_aiter_version_prefix),
        )?;
        assert_eq!(
            amd_aiter,
            "amd-aiter==0.1.22.post2.dev0+gb4d9154d1.d20261002"
        );
        assert_eq!(note, None, "a confirmed pin needs no fallback note");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn companion_pin_the_vllm_index_does_not_serve_falls_back_to_the_series() -> Result<()> {
        // A dependency vLLM leaves unpinned can resolve to a public PyPI release.
        // The vLLM index does not hold it, so it must not become the pin (that is
        // the non-ROCm build the discovery route exists to avoid): the series
        // match decides, as it did before vLLM's own resolution was consulted.
        let root = tempfile::tempdir()?;
        let paths = AppPaths {
            config_dir: root.path().join("config"),
            data_dir: root.path().join("data"),
            cache_dir: root.path().join("cache"),
        };
        let uv = fake_uv_for_staging_index(root.path());
        let python = root.path().join("python");
        let build = vllm_rocm_discover_build("10.1.0").expect("10.1.0 has a discover row");

        let (flash_attn, note) = discover_vllm_companion_pin(
            &uv,
            &paths,
            &python,
            build,
            Some("Resolved 232 packages\n + flash-attn==9.9.9\n"),
            "flash-attn",
            DiscoverVersion::Series(build.flash_attn_version_prefix),
        )?;
        assert_eq!(flash_attn, "flash-attn==2.8.3");
        // The fallback is not silent: the resolved build and the index's answer
        // are both in the note.
        let note = note.expect("an unconfirmed pin must leave a note");
        assert!(
            note.contains("flash-attn==9.9.9") && note.contains("did not confirm"),
            "{note}"
        );

        // And a vLLM that depends on neither sibling leaves the series in charge.
        let (amd_aiter, note) = discover_vllm_companion_pin(
            &uv,
            &paths,
            &python,
            build,
            Some("Resolved 3 packages\n + vllm==0.29.1\n"),
            "amd-aiter",
            DiscoverVersion::Series(build.amd_aiter_version_prefix),
        )?;
        assert_eq!(amd_aiter, "amd-aiter==0.1.24.post1");
        assert_eq!(note, None, "nothing was resolved, so nothing was dropped");

        // A probe that failed leaves no resolution at all: the series decides.
        let (amd_aiter, note) = discover_vllm_companion_pin(
            &uv,
            &paths,
            &python,
            build,
            None,
            "amd-aiter",
            DiscoverVersion::Series(build.amd_aiter_version_prefix),
        )?;
        assert_eq!(amd_aiter, "amd-aiter==0.1.24.post1");
        assert_eq!(note, None, "the probe failure is reported by the caller");
        Ok(())
    }

    /// Runs `install_vllm_rocm10_discover` against a fake `uv` that answers the
    /// probe, the exact-version confirm and the real installs, and returns what
    /// it returned together with every `uv` argv it saw.
    ///
    /// Unlike the helper-level tests above this drives the real resolve output
    /// through the whole discovery sequence.
    #[cfg(unix)]
    fn run_discover_with_fake_uv(
        probe_ok: bool,
        confirm_ok: bool,
        install_ok: bool,
    ) -> Result<(Result<VllmInstallOutcome>, Vec<String>)> {
        use rocm_core::fake_executable::write_fake_executable;
        let root = tempfile::tempdir()?;
        let paths = AppPaths {
            config_dir: root.path().join("config"),
            data_dir: root.path().join("data"),
            cache_dir: root.path().join("cache"),
        };
        let python = root.path().join("python");
        write_fake_executable(&python, "#!/bin/sh\necho cp314\n")?;
        let log = root.path().join("uv.log");
        let uv = root.path().join("uv");
        let probe = if probe_ok {
            r#"echo "Resolved 232 packages" >&2
    echo " + amd-aiter==0.1.22.post2.dev0+gb4d9154d1.d20261002" >&2
    echo " + flash-attn==2.8.3" >&2
    echo " + vllm==9.9.9" >&2"#
        } else {
            r#"echo "error: No solution found when resolving dependencies" >&2
    exit 2"#
        };
        let confirm = if confirm_ok {
            r#"echo " + ${last}" >&2"#
        } else {
            r#"echo "error: connection reset" >&2
        exit 1"#
        };
        let install = if install_ok {
            "exit 0"
        } else {
            "echo 'error: unsatisfiable' >&2; exit 1"
        };
        write_fake_executable(
            &uv,
            format!(
                r#"#!/bin/sh
echo "$@" >> "{log}"
last=""
for a in "$@"; do last="$a"; done
case "$*" in
  *--dry-run*--config-file*)
    {probe}
    ;;
  *--dry-run*)
    pkg=${{last%%==*}}
    case "$last" in
      *'*'*) echo " + ${{pkg}}==9.9.9" >&2 ;;
      amd-aiter==*|flash-attn==*)
        {confirm}
        ;;
      *) echo " + ${{last}}" >&2 ;;
    esac
    ;;
  *--config-file*)
    {install}
    ;;
  *"pip freeze"*)
    true
    ;;
esac
exit 0
"#,
                log = log.display()
            ),
        )?;
        let build = vllm_rocm_discover_build("10.1.0").expect("10.1.0 has a discover row");
        let outcome = install_vllm_rocm10_discover(&uv, &paths, &python, true, build);
        let calls = std::fs::read_to_string(&log)?
            .lines()
            .map(str::to_owned)
            .collect();
        Ok((outcome, calls))
    }

    /// Runs `install_vllm_rocm10_discover` on the 10.2 row against a fake `uv`
    /// whose `pip freeze` reports a release SDK's torch stack, with `prepare`
    /// free to set up the runtime registry first. Returns the outcome and every
    /// `uv` argv.
    #[cfg(unix)]
    fn run_ten_two_discover_with_fake_uv(
        prepare: impl FnOnce(&AppPaths, &Path) -> Result<()>,
    ) -> Result<(Result<VllmInstallOutcome>, Vec<String>)> {
        run_discover_row_with_fake_uv("10.2.0", None, prepare)
    }
    /// [`run_ten_two_discover_with_fake_uv`] for any row, optionally failing the
    /// `fail_install`th real `uv pip install` (1 torch, 2 remaining deps, 3 trio
    /// realign, 4 device realign; the dry-runs and `pip freeze` are not counted).
    #[cfg(unix)]
    fn run_discover_row_with_fake_uv(
        row: &str,
        fail_install: Option<u32>,
        prepare: impl FnOnce(&AppPaths, &Path) -> Result<()>,
    ) -> Result<(Result<VllmInstallOutcome>, Vec<String>)> {
        use rocm_core::fake_executable::write_fake_executable;
        let root = tempfile::tempdir()?;
        let paths = AppPaths {
            config_dir: root.path().join("config"),
            data_dir: root.path().join("data"),
            cache_dir: root.path().join("cache"),
        };
        let python = root.path().join("python");
        write_fake_executable(&python, "#!/bin/sh\necho cp314\n")?;
        prepare(&paths, &python)?;
        let log = root.path().join("uv.log");
        let uv = root.path().join("uv");
        write_fake_executable(
            &uv,
            format!(
                r#"#!/bin/sh
echo "$@" >> "{log}"
last=""
for a in "$@"; do last="$a"; done
case "$*" in
  *--dry-run*|*"pip freeze"*) ;;
  *)
    n=$(cat "{count}" 2>/dev/null || echo 0)
    n=$((n + 1))
    echo "$n" > "{count}"
    if [ "$n" = "{fail_at}" ]; then echo 'error: boom' >&2; exit 1; fi
    ;;
esac
case "$*" in
  *--dry-run*--config-file*)
    echo "Resolved 3 packages" >&2
    ;;
  *--dry-run*)
    pkg=${{last%%==*}}
    case "$last" in
      *'*'*) echo " + ${{pkg}}==9.9.9" >&2 ;;
      *) echo " + ${{last}}" >&2 ;;
    esac
    ;;
  *"pip freeze"*)
    echo "torch==2.14.0+rocm10.2.0"
    echo "torchvision==0.29.2+rocm10.2.0"
    echo "torchaudio==2.11.1+rocm10.2.0"
    echo "amd-torch-device-gfx1151==2.14.0+rocm10.2.0"
    ;;
esac
exit 0
"#,
                log = log.display(),
                count = root.path().join("uv.count").display(),
                fail_at = fail_install.unwrap_or(0)
            ),
        )?;
        let build = vllm_rocm_discover_build(row).expect("the row has a discover build");
        let outcome = install_vllm_rocm10_discover(&uv, &paths, &python, true, build);
        let calls = std::fs::read_to_string(&log)?
            .lines()
            .map(str::to_owned)
            .collect();
        Ok((outcome, calls))
    }

    #[cfg(unix)]
    #[test]
    fn a_release_sdk_on_the_ten_two_row_gets_its_torch_stack_from_its_own_index() -> Result<()> {
        let release = "https://stable.example/rocm/whl-next";
        let (outcome, calls) = run_ten_two_discover_with_fake_uv(|paths, python| {
            let registry = paths.data_dir.join("runtimes").join("registry");
            std::fs::create_dir_all(&registry)?;
            std::fs::write(
                registry.join("release.json"),
                format!(
                    r#"{{"python_executable":"{}","index_url":"{release}"}}"#,
                    python.display()
                ),
            )?;
            Ok(())
        })?;
        outcome?;
        let build = vllm_rocm_discover_build("10.2.0").expect("10.2.0 has a discover row");
        // The torch install, the trio realign and the device realign are the real
        // installs that name a torch-stack package; each must ask the SDK's index.
        let torch_stack_installs: Vec<&String> = calls
            .iter()
            .filter(|call| !call.contains("--dry-run") && call.contains("--no-deps"))
            .filter(|call| call.contains("torch"))
            .collect();
        assert_eq!(
            torch_stack_installs.len(),
            3,
            "expected torch install, trio realign and device realign: {calls:?}"
        );
        for call in torch_stack_installs {
            assert!(call.contains(&format!("--index-url {release}")), "{call}");
            assert!(!call.contains(build.torch_index_url), "{call}");
        }
        Ok(())
    }

    /// Plants a runtime manifest for the interpreter, with `extra` JSON fields.
    #[cfg(unix)]
    fn plant_manifest(paths: &AppPaths, python: &Path, extra: &str) -> Result<()> {
        let registry = paths.data_dir.join("runtimes").join("registry");
        std::fs::create_dir_all(&registry)?;
        std::fs::write(
            registry.join("runtime.json"),
            format!(r#"{{"python_executable":"{}"{extra}}}"#, python.display()),
        )?;
        Ok(())
    }
    /// Makes the registry unreadable: a directory where a manifest file belongs
    /// cannot be read as a file by anyone, root included, unlike a mode-000 registry.
    #[cfg(unix)]
    fn make_registry_unreadable(paths: &AppPaths) -> Result<()> {
        let broken = paths
            .data_dir
            .join("runtimes")
            .join("registry")
            .join("broken.json");
        std::fs::create_dir_all(broken)?;
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn the_ten_two_row_says_when_it_falls_back_to_its_own_index() -> Result<()> {
        let build = vllm_rocm_discover_build("10.2.0").expect("10.2.0 has a discover row");
        let on_row_index = |calls: &[String]| {
            calls
                .iter()
                .any(|call| call.contains(&format!("--index-url {}", build.torch_index_url)))
        };

        // A manifest that records no index (an older one, or an adopted runtime):
        // the row's index, and the install says so.
        let (outcome, calls) =
            run_ten_two_discover_with_fake_uv(|paths, python| plant_manifest(paths, python, ""))?;
        let notes = outcome?.notes;
        assert!(
            notes
                .iter()
                .any(|note| note.contains("records a wheel index")),
            "{notes:?}"
        );
        assert!(on_row_index(&calls));

        // A registry that cannot be read: the same index, and the install says so.
        let (outcome, calls) =
            run_ten_two_discover_with_fake_uv(|paths, _| make_registry_unreadable(paths))?;
        let notes = outcome?.notes;
        assert!(
            notes
                .iter()
                .any(|note| note.contains("could not read the runtime registry")),
            "{notes:?}"
        );
        assert!(on_row_index(&calls));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_row_that_discovers_its_stack_never_looks_at_the_registry_or_says_anything_about_it()
    -> Result<()> {
        // 10.1's index is fixed by design, so an unreadable registry or a manifest
        // with no index is not something its install has a reason to mention.
        let (outcome, _) = run_discover_row_with_fake_uv("10.1.0", None, |paths, _| {
            make_registry_unreadable(paths)
        })?;
        let notes = outcome?.notes;
        assert!(
            notes
                .iter()
                .all(|note| !note.contains("registry") && !note.contains("wheel index")),
            "{notes:?}"
        );
        let (outcome, _) = run_discover_row_with_fake_uv("10.1.0", None, |paths, python| {
            plant_manifest(paths, python, "")
        })?;
        assert!(
            outcome?
                .notes
                .iter()
                .all(|note| !note.contains("wheel index"))
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_failed_install_step_on_the_fallback_index_names_the_fallback() -> Result<()> {
        // Each real install in turn: the torch install, the remaining deps, the trio
        // realign and the device realign all run after the index fell back.
        for step in 1..=4 {
            let (outcome, _) =
                run_discover_row_with_fake_uv("10.2.0", Some(step), |paths, python| {
                    plant_manifest(paths, python, "")
                })?;
            let err = format!("{:#}", outcome.expect_err("an install step fails"));
            assert!(err.contains("boom"), "step {step}: {err}");
            assert!(
                err.contains("records a wheel index") && err.contains("Fallbacks used"),
                "step {step}: the uv failure must carry the note about the index it ran \
                 against: {err}"
            );
        }

        // With the SDK's own index recorded there is nothing to add to the failure.
        let (outcome, _) = run_discover_row_with_fake_uv("10.2.0", Some(1), |paths, python| {
            plant_manifest(
                paths,
                python,
                r#","index_url":"https://stable.example/whl""#,
            )
        })?;
        let err = format!("{:#}", outcome.expect_err("the torch install fails"));
        assert!(
            err.contains("boom") && !err.contains("Fallbacks used"),
            "{err}"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn discovery_installs_the_companions_the_discovered_vllm_depends_on() -> Result<()> {
        // The real resolve output flows through the whole sequence: the probe
        // names the 0.1.22 dev build, the index confirms it, and that build (not
        // the series' newest) is what the real install pins.
        let (outcome, calls) = run_discover_with_fake_uv(true, true, true)?;
        let outcome = outcome?;
        assert!(
            outcome
                .pins
                .contains(&"amd-aiter==0.1.22.post2.dev0+gb4d9154d1.d20261002".to_owned()),
            "{:?}",
            outcome.pins
        );
        assert!(outcome.pins.contains(&"flash-attn==2.8.3".to_owned()));
        assert!(outcome.notes.is_empty(), "{:?}", outcome.notes);
        let real_install = calls
            .iter()
            .find(|call| call.contains("--reinstall-package vllm"))
            .expect("the real vllm install ran");
        assert!(
            real_install.contains("amd-aiter==0.1.22.post2.dev0+gb4d9154d1.d20261002"),
            "{real_install}"
        );
        assert!(!real_install.contains("amd-aiter==9.9.9"), "{real_install}");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_transient_confirm_failure_is_retried_once_before_falling_back() -> Result<()> {
        use rocm_core::fake_executable::write_fake_executable;
        let root = tempfile::tempdir()?;
        let paths = AppPaths {
            config_dir: root.path().join("config"),
            data_dir: root.path().join("data"),
            cache_dir: root.path().join("cache"),
        };
        let python = root.path().join("python");
        let marker = root.path().join("first-call-seen");
        let uv = root.path().join("uv");
        // The exact-build confirm fails the first time it is asked and succeeds
        // the second, like a dropped connection.
        write_fake_executable(
            &uv,
            format!(
                r#"#!/bin/sh
for last; do :; done
if [ -e "{marker}" ]; then
  echo " + $last" >&2
else
  touch "{marker}"
  echo "error: connection reset" >&2
  exit 1
fi
"#,
                marker = marker.display()
            ),
        )?;
        let build = vllm_rocm_discover_build("10.1.0").expect("10.1.0 has a discover row");
        let resolution =
            "Resolved 3 packages\n + amd-aiter==0.1.22.post2.dev0+gb4d9154d1.d20261002\n";

        let (pin, note) = discover_vllm_companion_pin(
            &uv,
            &paths,
            &python,
            build,
            Some(resolution),
            "amd-aiter",
            DiscoverVersion::Series(build.amd_aiter_version_prefix),
        )?;
        assert_eq!(pin, "amd-aiter==0.1.22.post2.dev0+gb4d9154d1.d20261002");
        assert_eq!(note, None, "a retry that succeeds needs no note");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_failed_resolve_probe_falls_back_to_the_series_and_says_why() -> Result<()> {
        // The probe only improves on the series match; it must not make an
        // install that worked before fail. The reason is reported, not dropped.
        let (outcome, _) = run_discover_with_fake_uv(false, true, true)?;
        let outcome = outcome?;
        assert!(outcome.pins.contains(&"amd-aiter==9.9.9".to_owned()));
        assert!(outcome.pins.contains(&"flash-attn==9.9.9".to_owned()));
        assert_eq!(outcome.notes.len(), 1, "{:?}", outcome.notes);
        assert!(
            outcome.notes[0].contains("could not resolve the dependencies")
                && outcome.notes[0].contains("No solution found"),
            "{:?}",
            outcome.notes
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn an_unconfirmed_companion_pin_is_reported_and_follows_a_failed_install() -> Result<()> {
        // A transient index failure on the confirm call ends in the series pin,
        // which is the pair that can be unsatisfiable. Both the success path
        // and the failure path must carry the reason.
        let (outcome, _) = run_discover_with_fake_uv(true, false, true)?;
        let outcome = outcome?;
        assert_eq!(outcome.notes.len(), 2, "{:?}", outcome.notes);
        assert!(
            outcome
                .notes
                .iter()
                .all(|note| note.contains("connection reset")),
            "{:?}",
            outcome.notes
        );

        let (outcome, _) = run_discover_with_fake_uv(true, false, false)?;
        let err = outcome.expect_err("the real install was made to fail");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("unsatisfiable") && rendered.contains("connection reset"),
            "the failure must name both the install error and the dropped pin: {rendered}"
        );
        // Plain display is what a user sees: the uv failure comes first and the
        // note about the fallback follows it.
        let plain = err.to_string();
        let cause_at = plain.find("unsatisfiable").expect("the cause is shown");
        let note_at = plain.find("connection reset").expect("the note is shown");
        assert!(cause_at < note_at, "the cause must lead: {plain}");

        // With nothing dropped, the install error is left as it was.
        let (outcome, _) = run_discover_with_fake_uv(true, true, false)?;
        let rendered = format!(
            "{:#}",
            outcome.expect_err("the real install was made to fail")
        );
        assert!(!rendered.contains("did not confirm"), "{rendered}");
        Ok(())
    }

    #[test]
    fn write_vllm_index_uv_config_writes_an_ignore_error_codes_entry() -> Result<()> {
        let root = tempfile::tempdir()?;
        let paths = AppPaths {
            config_dir: root.path().join("config"),
            data_dir: root.path().join("data"),
            cache_dir: root.path().join("cache"),
        };
        let build = vllm_rocm_discover_build("10.0.0").expect("10.0.0 has a discover row");

        let config_file = write_vllm_index_uv_config(&paths, build)?;
        let contents = std::fs::read_to_string(&config_file)?;
        assert!(contents.contains(build.vllm_index_url), "{contents}");
        assert!(
            contents.contains("ignore-error-codes = [403]"),
            "{contents}"
        );
        Ok(())
    }
    /// The prefix boundary (`-device-gfx`, not a bare `starts_with` of
    /// `amd-{base_pkg}`) must stop `torch`'s match from also catching
    /// `torchvision`'s device package: `amd-torchvision-...` does start with
    /// `amd-torch`, just not with `amd-torch-device-gfx`.
    #[test]
    fn is_amd_device_package_for_does_not_cross_match_prefixed_base_packages() {
        assert!(is_amd_device_package_for(
            "amd-torch-device-gfx1151",
            "torch"
        ));
        assert!(!is_amd_device_package_for(
            "amd-torchvision-device-gfx1151",
            "torch"
        ));
        assert!(is_amd_device_package_for(
            "amd-torchvision-device-gfx1151",
            "torchvision"
        ));
        // Real `uv pip freeze` output is PEP 503 canonical (hyphenated), but
        // the check normalizes underscores too.
        assert!(is_amd_device_package_for(
            "amd_torch_device_gfx1151",
            "torch"
        ));
    }
    #[test]
    fn rocm_local_version_guard_rejects_another_lines_wheel() {
        // No local version at all (flash-attn, amd-aiter) is fine.
        assert!(ensure_rocm_local_version_matches("flash-attn==2.8.3", "10.1.0").is_ok());
        // Same line, different patch and rc suffix, is fine.
        assert!(ensure_rocm_local_version_matches("torch==2.12.0+rocm10.1.0rc3", "10.1.0").is_ok());
        // A 10.0 wheel resolved for a 10.1 SDK is not.
        let err = ensure_rocm_local_version_matches(
            "vllm==0.27.1.dev5+rocm10.0.0.gf46a9dfe2.d20260826",
            "10.1.0",
        )
        .expect_err("a 10.0 wheel must not install onto a 10.1 SDK");
        let rendered = format!("{err:#}");
        assert!(rendered.contains("10.1.0"), "{rendered}");
        assert!(rendered.contains("rocm10.0.0"), "{rendered}");
    }
    /// The 10.2 vLLM wheels spell their ROCm line `.rocm102`, not
    /// `+rocm10.2.0`; the guard must still compare it.
    #[test]
    fn rocm_local_version_guard_compares_the_compact_rocm_spelling() {
        let ten_two = "vllm==0.32.1.dev0+g5df63a720.d20261110.rocm102";
        assert!(ensure_rocm_local_version_matches(ten_two, "10.2.0").is_ok());
        assert!(ensure_rocm_local_version_matches(ten_two, "10.2.0a20261022").is_ok());
        let err = ensure_rocm_local_version_matches(ten_two, "10.1.0")
            .expect_err("a 10.2 wheel must not install onto a 10.1 SDK");
        assert!(format!("{err:#}").contains("rocm102"), "{err:#}");
        // A tagged release carries the compact spelling as the whole local version.
        let bare = "vllm==0.32.1+rocm102";
        assert!(ensure_rocm_local_version_matches(bare, "10.2.0").is_ok());
        assert!(ensure_rocm_local_version_matches(bare, "10.1.0").is_err());
        // A `rocm` segment that is neither spelling names no line, so it is no match.
        for unreadable in [
            "vllm==0.32.1+rocm",
            "vllm==0.32.1+rocmabc",
            "vllm==0.32.1+g5d.rocm.x",
        ] {
            assert!(
                ensure_rocm_local_version_matches(unreadable, "10.2.0").is_err(),
                "{unreadable}"
            );
        }
        // And the dotted spelling is read as a series wherever its segment sits.
        let late = "vllm==0.32.1+g5df63a720.rocm10.2.0";
        assert!(ensure_rocm_local_version_matches(late, "10.2.0").is_ok());
        assert!(ensure_rocm_local_version_matches(late, "10.3.0").is_err());
        // And a 10.1 wheel must not slip onto a 10.2 SDK through the dotted spelling.
        assert!(
            ensure_rocm_local_version_matches("vllm==0.29.1.dev0+rocm10.1.0.gabc", "10.2.0")
                .is_err()
        );
    }
    /// A 10.2 runtime keeps the torch stack its SDK install resolved; a stack
    /// from another series or another ROCm line is refused, not kept.
    #[test]
    fn installed_torch_stack_pins_reads_and_checks_the_sdk_stack() {
        let ten_two = vllm_rocm_discover_build("10.2.0").expect("10.2 row");
        let freeze = "flash-attn==2.8.3\ntorch==2.14.0+rocm10.2.0a20261022\n\
                      torchaudio==2.11.1+rocm10.2.0a20261022\n\
                      torchvision==0.29.2+rocm10.2.0a20261022\n";
        assert_eq!(
            installed_torch_stack_pins(freeze, ten_two).expect("coherent 10.2 stack"),
            [
                "torch==2.14.0+rocm10.2.0a20261022",
                "torchvision==0.29.2+rocm10.2.0a20261022",
                "torchaudio==2.11.1+rocm10.2.0a20261022",
            ]
        );
        // The 10.1 torch an unpinned SDK install can leave behind is not 2.14's pair.
        let old = freeze.replace("2.14.0+rocm10.2.0a20261022", "2.13.0+rocm10.2.0a20261022");
        let err = installed_torch_stack_pins(&old, ten_two).expect_err("torch 2.13");
        assert!(
            format!("{err:#}").contains("needs `torch` 2.14.x"),
            "{err:#}"
        );
        // Another ROCm line is refused by the same guard as every other pin.
        let other_line = freeze.replace(
            "torch==2.14.0+rocm10.2.0a20261022",
            "torch==2.14.0+rocm10.3.0a20261110",
        );
        assert!(installed_torch_stack_pins(&other_line, ten_two).is_err());
        // A runtime with no torch says what to do.
        let err = installed_torch_stack_pins("flash-attn==2.8.3\n", ten_two).expect_err("no torch");
        assert!(format!("{err:#}").contains("rocm install sdk"), "{err:#}");
    }
    /// An `Exact` torchaudio pin on a row that reuses the SDK's stack means that
    /// release and no other (PEP 440 `==release`): the `+local` part is not
    /// compared, and a longer release is not a match.
    #[test]
    fn installed_torch_stack_pins_treats_an_exact_pin_as_that_release() {
        let ten_two = vllm_rocm_discover_build("10.2.0").expect("10.2 row");
        let exact = |release: &'static str| VllmRocmDiscoverBuild {
            torchaudio_version: DiscoverVersion::Exact(release),
            ..*ten_two
        };
        let freeze = |torchaudio: &str| {
            format!(
                "torch==2.14.0+rocm10.2.0a20261022\ntorchvision==0.29.2+rocm10.2.0a20261022\n\
                 torchaudio=={torchaudio}+rocm10.2.0a20261022\n"
            )
        };
        let accepted = installed_torch_stack_pins(&freeze("2.11.0.2"), &exact("2.11.0.2"))
            .expect("the exact release with its local version");
        assert_eq!(accepted[2], "torchaudio==2.11.0.2+rocm10.2.0a20261022");
        for installed in ["2.11.0.2.9", "2.11.0.3", "2.11.1"] {
            assert!(
                installed_torch_stack_pins(&freeze(installed), &exact("2.11.0.2")).is_err(),
                "{installed} is not the release 2.11.0.2"
            );
        }
        // A shorter pin does not accept the longer rebuild it is a prefix of.
        assert!(installed_torch_stack_pins(&freeze("2.11.0.2"), &exact("2.11.0")).is_err());
    }
    /// Pins `install_vllm_rocm10_discover`'s own call sequencing, not just its
    /// arg-builder helpers in isolation: the remaining-deps install must use
    /// `--config-file` (never `--extra-index-url`, which cannot tolerate the
    /// 403 `lm-format-enforcer` needs), and a torch realign call must follow
    /// it so that install's full-dependency resolve cannot leave an
    /// unconstrained torch in place. Runs offline against fake `uv`/`python`
    /// shims, same pattern as `crates/rocm-core/src/examine/framework.rs`.
    #[test]
    #[cfg(unix)]
    fn install_vllm_rocm10_discover_realigns_torch_after_the_config_file_install() -> Result<()> {
        use rocm_core::fake_executable::write_fake_executable;

        let root = tempfile::tempdir()?;
        let paths = AppPaths {
            config_dir: root.path().join("config"),
            data_dir: root.path().join("data"),
            cache_dir: root.path().join("cache"),
        };
        let log = root.path().join("uv-calls.log");

        let python = root.path().join("python");
        write_fake_executable(&python, "#!/bin/sh\necho cp314\n")?;

        // Answers every `--dry-run` discovery call with a resolved pin for
        // whatever package the requirement names, and otherwise just records
        // the call (the real installs) for the assertions below.
        let uv = root.path().join("uv");
        write_fake_executable(
            &uv,
            format!(
                r#"#!/bin/sh
echo "$@" >> "{log}"
last=""
for a in "$@"; do last="$a"; done
case "$*" in
  *--dry-run*)
    pkg=${{last%%==*}}
    echo " + ${{pkg}}==9.9.9" >&2
    ;;
  *"pip freeze"*)
    true
    ;;
esac
exit 0
"#,
                log = log.display()
            ),
        )?;

        let build = vllm_rocm_discover_build("10.1.0").expect("10.1.0 has a discover row");
        let pins = install_vllm_rocm10_discover(&uv, &paths, &python, true, build)?.pins;
        assert_eq!(pins.len(), 6);

        let calls: Vec<Vec<String>> = std::fs::read_to_string(&log)?
            .lines()
            .map(|line| line.split_whitespace().map(ToOwned::to_owned).collect())
            .collect();
        let discovered: Vec<&String> = calls
            .iter()
            .filter(|args| args.contains(&"--dry-run".to_owned()))
            .filter_map(|args| args.last())
            .collect();
        assert_eq!(
            discovered,
            [
                "torch==2.13.*",
                "torchvision==0.28.*",
                "torchaudio==2.11.0.2",
                "vllm==0.29.*",
                // The resolve probe for the vLLM just discovered, which is
                // where flash-attn and amd-aiter get their pins when the vLLM
                // index serves that exact version. The fake uv answers it with
                // no dependencies, so the series pins below still apply.
                "vllm==9.9.9",
                "flash-attn==2.8.*",
                "amd-aiter==0.1.*",
            ],
            "discovery must ask uv for exactly torchaudio 2.11.0.2 on 10.1: {calls:?}"
        );
        let real_installs: Vec<&Vec<String>> = calls
            .iter()
            .filter(|args| !args.contains(&"--dry-run".to_owned()))
            .filter(|args| args.get(1).map(String::as_str) != Some("freeze"))
            .collect();
        assert_eq!(
            real_installs.len(),
            3,
            "expected torch install, remaining-deps install, stack realign: {calls:?}"
        );

        let torch_install = real_installs[0];
        assert!(torch_install.contains(&"--index-url".to_owned()));
        assert!(!torch_install.contains(&"--extra-index-url".to_owned()));
        assert!(!torch_install.contains(&"--config-file".to_owned()));

        let remaining_install = real_installs[1];
        assert!(
            remaining_install.contains(&"--config-file".to_owned()),
            "the remaining-deps install must use --config-file, not --extra-index-url, \
             for the 403-tolerant index: {remaining_install:?}"
        );
        assert!(!remaining_install.contains(&"--extra-index-url".to_owned()));

        let stack_realign = real_installs[2];
        assert!(
            stack_realign.contains(&"--reinstall-package".to_owned())
                && stack_realign.contains(&"torch".to_owned()),
            "the full-dependency install can silently replace torch with an unconstrained \
             PyPI build; a forced reinstall must follow it: {stack_realign:?}"
        );
        assert!(
            stack_realign.contains(&"torchvision==9.9.9".to_owned())
                && stack_realign.contains(&"torchaudio==9.9.9".to_owned()),
            "torchvision/torchaudio must be discovered fresh and restored alongside torch, \
             or the full-dependency resolve can leave an ABI-incompatible build in place: \
             {stack_realign:?}"
        );
        assert!(stack_realign.contains(&"--index-url".to_owned()));
        assert!(!stack_realign.contains(&"--extra-index-url".to_owned()));
        assert!(!stack_realign.contains(&"--config-file".to_owned()));

        Ok(())
    }
    /// The full-dependency install can leave an AMD torch device-kernel
    /// plugin pinned to whatever the index's newest release is, ahead of the
    /// torch version realigned alongside it -- a mismatch that doesn't fail
    /// install, only serving. Whichever such package `uv pip freeze` reports
    /// installed must be realigned to torch's exact release too.
    #[test]
    #[cfg(unix)]
    fn install_vllm_rocm10_discover_realigns_stale_device_packages_per_base_package() -> Result<()>
    {
        use rocm_core::fake_executable::write_fake_executable;

        let root = tempfile::tempdir()?;
        let paths = AppPaths {
            config_dir: root.path().join("config"),
            data_dir: root.path().join("data"),
            cache_dir: root.path().join("cache"),
        };
        let log = root.path().join("uv-calls.log");

        let python = root.path().join("python");
        write_fake_executable(&python, "#!/bin/sh\necho cp314\n")?;

        let uv = root.path().join("uv");
        write_fake_executable(
            &uv,
            format!(
                r#"#!/bin/sh
echo "$@" >> "{log}"
last=""
for a in "$@"; do last="$a"; done
case "$*" in
  *--dry-run*)
    pkg=${{last%%==*}}
    case "$pkg" in
      torchvision) echo " + ${{pkg}}==8.8.8" >&2 ;;
      *) echo " + ${{pkg}}==9.9.9" >&2 ;;
    esac
    ;;
  *"pip freeze"*)
    echo "amd_torch_device_gfx1151==2.14.0+rocm10.1.0"
    echo "amd_torchvision_device_gfx1151==1.1.1+rocm10.1.0"
    ;;
esac
exit 0
"#,
                log = log.display()
            ),
        )?;

        let build = vllm_rocm_discover_build("10.1.0").expect("10.1.0 has a discover row");
        let pins = install_vllm_rocm10_discover(&uv, &paths, &python, true, build)?.pins;
        assert!(
            pins.contains(&"amd_torch_device_gfx1151==9.9.9".to_owned()),
            "the stale torch device package must be realigned to torch's own resolved release: {pins:?}"
        );
        assert!(
            pins.contains(&"amd_torchvision_device_gfx1151==8.8.8".to_owned()),
            "the stale torchvision device package must be realigned to torchvision's own \
             resolved release, not torch's: {pins:?}"
        );
        // Guards the prefix-boundary check in `is_amd_device_package_for`: a
        // weakened boundary (e.g. matching on a bare `starts_with` of
        // `amd-torch`, which `amd-torchvision-device-gfx1151` also satisfies)
        // would add a second, wrongly-cross-matched pin alongside the
        // correct one above, invisible to a `.contains()`-only check.
        assert!(
            !pins.contains(&"amd_torchvision_device_gfx1151==9.9.9".to_owned()),
            "torchvision's device package must not be cross-matched under torch's base \
             package and pinned to torch's release: {pins:?}"
        );
        assert_eq!(
            pins.len(),
            8,
            "expected the 6 discovered pins plus exactly one device pin per base package: {pins:?}"
        );

        let calls: Vec<Vec<String>> = std::fs::read_to_string(&log)?
            .lines()
            .map(|line| line.split_whitespace().map(ToOwned::to_owned).collect())
            .collect();
        let stack_realign = calls
            .iter()
            .rfind(|args| {
                !args.contains(&"--dry-run".to_owned())
                    && args.get(1).map(String::as_str) != Some("freeze")
            })
            .expect("a stack realign call must have run");
        assert!(
            stack_realign.contains(&"--reinstall-package".to_owned())
                && stack_realign.contains(&"amd_torch_device_gfx1151".to_owned())
                && stack_realign.contains(&"amd_torch_device_gfx1151==9.9.9".to_owned()),
            "the realign call must reinstall the stale torch device package pinned to torch's \
             resolved release: {stack_realign:?}"
        );
        assert!(
            stack_realign.contains(&"amd_torchvision_device_gfx1151".to_owned())
                && stack_realign.contains(&"amd_torchvision_device_gfx1151==8.8.8".to_owned()),
            "the realign call must reinstall the stale torchvision device package pinned to \
             torchvision's resolved release: {stack_realign:?}"
        );

        Ok(())
    }
    /// Pins the ordering fix directly: the trio realign (torch/torchvision/
    /// torchaudio `--no-deps`) must run before `uv pip freeze` is even
    /// invoked, so a freeze failure aborts only the device-plugin lookup that
    /// follows it, never the realign itself. Reverting the ordering (freeze
    /// before realign) would leave only 2 real installs here instead of 3.
    #[test]
    #[cfg(unix)]
    fn install_vllm_rocm10_discover_realigns_the_trio_even_if_pip_freeze_fails() -> Result<()> {
        use rocm_core::fake_executable::write_fake_executable;

        let root = tempfile::tempdir()?;
        let paths = AppPaths {
            config_dir: root.path().join("config"),
            data_dir: root.path().join("data"),
            cache_dir: root.path().join("cache"),
        };
        let log = root.path().join("uv-calls.log");

        let python = root.path().join("python");
        write_fake_executable(&python, "#!/bin/sh\necho cp314\n")?;

        let uv = root.path().join("uv");
        write_fake_executable(
            &uv,
            format!(
                r#"#!/bin/sh
echo "$@" >> "{log}"
last=""
for a in "$@"; do last="$a"; done
case "$*" in
  *--dry-run*)
    pkg=${{last%%==*}}
    echo " + ${{pkg}}==9.9.9" >&2
    ;;
  *"pip freeze"*)
    echo "simulated uv pip freeze failure" >&2
    exit 1
    ;;
esac
exit 0
"#,
                log = log.display()
            ),
        )?;

        let build = vllm_rocm_discover_build("10.1.0").expect("10.1.0 has a discover row");
        let err = install_vllm_rocm10_discover(&uv, &paths, &python, true, build)
            .expect_err("a pip freeze failure must surface, not silently succeed");
        assert!(
            format!("{err:#}").contains("pip freeze"),
            "error should name the step that failed: {err:#}"
        );

        let calls: Vec<Vec<String>> = std::fs::read_to_string(&log)?
            .lines()
            .map(|line| line.split_whitespace().map(ToOwned::to_owned).collect())
            .collect();
        let real_installs: Vec<&Vec<String>> = calls
            .iter()
            .filter(|args| !args.contains(&"--dry-run".to_owned()))
            .filter(|args| args.get(1).map(String::as_str) != Some("freeze"))
            .collect();
        assert_eq!(
            real_installs.len(),
            3,
            "torch install, remaining-deps install, and the trio realign must all run before \
             `uv pip freeze` is even invoked, so its failure cannot skip the realign: {calls:?}"
        );
        let stack_realign = real_installs[2];
        assert!(
            stack_realign.contains(&"--no-deps".to_owned())
                && stack_realign.contains(&"--reinstall-package".to_owned())
                && stack_realign.contains(&"torch".to_owned())
                && stack_realign.contains(&"torchvision==9.9.9".to_owned())
                && stack_realign.contains(&"torchaudio==9.9.9".to_owned()),
            "the trio realign must have already run before the freeze failure: {stack_realign:?}"
        );

        Ok(())
    }
}
