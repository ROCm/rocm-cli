// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, anyhow, bail};
use rocm_core::{
    AppPaths, command_failure_detail, ensure_uv_binary, split_local_version, uv_command_env,
    uv_pip_freeze_args, uv_pip_install_base,
};
use rocm_engine_protocol::{InstallRequest, InstallResponse};
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

use crate::repair::{RepairAssessment, assess_runtime_repair};
use crate::runtime::{
    assessed_python_for_repair, describe_skipped_managed_runtimes, resolve_managed_runtime_python,
    resolve_vllm_runtime, rocm_sdk_series_matches, runtime_is_managed, vllm_runtime_warnings,
};
use crate::state::runtime_lock_hash;
use crate::wheel_builds::{
    DiscoverVersion, VllmInstallRoute, VllmInstallTarget, VllmRocmDiscoverBuild,
    vllm_install_route, vllm_install_target, vllm_rocm_discover_build,
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
    build: &VllmRocmDiscoverBuild,
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
    args.push(build.torch_index_url.to_owned());
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
    build: &VllmRocmDiscoverBuild,
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
    args.push(build.torch_index_url.to_owned());
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
    let Some(local_rocm) = local.strip_prefix("rocm") else {
        return Ok(());
    };
    if rocm_sdk_series_matches(local_rocm, rocm_sdk_version) {
        return Ok(());
    }
    bail!(
        "discovered `{pin}` for ROCm SDK {rocm_sdk_version}, but its `+{local}` build targets a \
         different ROCm release; the index for this ROCm line is serving another SDK's wheels"
    )
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
    ensure_discover_python_tag(python, build)?;
    let torch = discover_pinned_requirement(
        uv,
        paths,
        python,
        build.torch_index_url,
        "torch",
        DiscoverVersion::Series(build.torch_version_prefix),
    )?;
    // Discovered fresh from the same index and row as torch, rather than
    // snapshotted from whatever the SDK install resolved earlier: that
    // snapshot can pair with a *different* torch version than the one this
    // row pins (the SDK's own install is unpinned and may have picked the
    // newest torch/torchvision/torchaudio available, independently of each
    // other), producing an ABI-incompatible triplet at realign time.
    let torchvision = discover_pinned_requirement(
        uv,
        paths,
        python,
        build.torch_index_url,
        "torchvision",
        DiscoverVersion::Series(build.torchvision_version_prefix),
    )?;
    let torchaudio = discover_pinned_requirement(
        uv,
        paths,
        python,
        build.torch_index_url,
        "torchaudio",
        build.torchaudio_version,
    )?;
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
    let mut notes: Vec<String> = Vec::new();
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
    let torch_args = vllm_rocm10_discover_torch_install_args(python, reinstall, build, &pins[0]);
    run_uv_pip_install(uv, paths, python, torch_args)?;

    let mut remaining_args =
        vllm_rocm10_discover_remaining_install_args(python, reinstall, &pins[3..]);
    remaining_args.push("--config-file".to_owned());
    remaining_args.push(config_file.display().to_string());
    if let Err(err) = run_uv_pip_install(uv, paths, python, remaining_args) {
        // A note kept only for the success path would be lost exactly when the
        // fallback pins are what failed.
        return Err(if notes.is_empty() {
            err
        } else {
            // The cause first, so plain `{}` display leads with the uv failure
            // and not with a note about how the pins were chosen.
            anyhow!(
                "{err:#}\n\nThe fallback pins were used because: {}",
                notes.join(" ")
            )
        });
    }

    // The full-dependency resolve above can replace torch (and, pulling it in
    // transitively, torchvision/torchaudio) with unconstrained PyPI builds;
    // force the whole stack back to the coherent trio discovered above. This
    // must run unconditionally, before the device-plugin discovery below: a
    // `uv pip freeze` failure there must not skip restoring the trio too.
    let trio_realign_args = vllm_rocm10_discover_realign_install_args(python, build, &pins[..3]);
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
            vllm_rocm10_discover_realign_install_args(python, build, &device_pins);
        run_uv_pip_install(uv, paths, python, device_realign_args)?;
    }

    let mut pins = pins;
    pins.extend(device_pins);
    Ok(VllmInstallOutcome { pins, notes })
}
#[cfg(test)]
mod tests {
    use super::*;
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

        let args = vllm_rocm10_discover_torch_install_args(&python, false, build, torch_pin);
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

        let args = vllm_rocm10_discover_torch_install_args(&python, true, build, torch_pin);
        assert!(args.contains(&"--reinstall-package".to_owned()));
        assert!(args.contains(&"torch".to_owned()));
        assert!(!args.contains(&"--reinstall".to_owned()));
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
        use std::os::unix::fs::PermissionsExt as _;
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
        std::fs::write(&path, script).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
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
    /// Pins `install_vllm_rocm10_discover`'s own call sequencing, not just its
    /// arg-builder helpers in isolation: the remaining-deps install must use
    /// `--config-file` (never `--extra-index-url`, which cannot tolerate the
    /// 403 `lm-format-enforcer` needs), and a torch realign call must follow
    /// it so that install's full-dependency resolve cannot leave an
    /// unconstrained torch in place. Runs offline against fake `uv`/`python`
    /// shims, same pattern as `crates/rocm-core/src/examine.rs`.
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
