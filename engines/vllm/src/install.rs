// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, anyhow, bail};
use rocm_core::{
    AppPaths, DependencyViolation, check_dependencies, ensure_uv_binary, split_local_version,
    uv_command_env, uv_pip_install_base, violation_subject, violations_requiring,
};
use rocm_engine_protocol::{InstallRequest, InstallResponse};
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

use crate::ENGINE_NAME;
use crate::runtime::VllmRuntime;

/// Known-good `(ROCm SDK version, vLLM version, ABI tag)` combinations for
/// `uv pip install vllm`, keyed by the ROCm SDK version recorded in the
/// runtime manifest (see `rocm_sdk_version_from_manifest`).
///
/// Add a row only once wheels.vllm.ai actually publishes a build for that
/// ROCm SDK version — see `docs/vllm.md` for AMD's current ROCm X guidance
/// when this table has no matching row.
struct VllmRocmBuild {
    rocm_sdk_version: &'static str,
    vllm_version: &'static str,
    abi: &'static str,
}

const VLLM_ROCM_BUILD_TABLE: &[VllmRocmBuild] = &[VllmRocmBuild {
    rocm_sdk_version: "7.2.3",
    vllm_version: "0.26.0",
    abi: "rocm723",
}];
/// Prefix shared by every published vLLM ROCm wheel index.
///
/// Release indexes are `{VLLM_ROCM_INDEX_PREFIX}/<version>/<abi>`, which is
/// what lets [`vllm_rocm_build_from_index_url`] recover the build a custom
/// index serves and keep the requirement pinned to it.
const VLLM_ROCM_INDEX_PREFIX: &str = "https://wheels.vllm.ai/rocm";
/// A ROCm SDK version whose vLLM/flash-attn/amd-aiter wheels aren't published
/// under a fixed filename (AMD rotates the dev-tag suffix constantly), so the
/// exact wheel must be discovered from the index at install time instead of
/// pinned in [`VLLM_ROCM_BUILD_TABLE`].
pub(crate) struct VllmRocmDiscoverBuild {
    rocm_sdk_version: &'static str,
    vllm_version_prefix: &'static str,
    flash_attn_version_prefix: &'static str,
    amd_aiter_version_prefix: &'static str,
    torch_requirement: &'static str,
    tensorizer_requirement: &'static str,
}

const VLLM_ROCM_DISCOVER_BUILD_TABLE: &[VllmRocmDiscoverBuild] = &[VllmRocmDiscoverBuild {
    rocm_sdk_version: "10.0.0",
    vllm_version_prefix: "0.27",
    flash_attn_version_prefix: "2.8",
    amd_aiter_version_prefix: "0.1",
    torch_requirement: "torch==2.12.0+rocm10.0.0",
    tensorizer_requirement: "tensorizer==2.12.1",
}];
/// Index that publishes the rotating-dev-tag vLLM/flash-attn/amd-aiter
/// wheels for [`VLLM_ROCM_DISCOVER_BUILD_TABLE`] rows.
const VLLM_ROCM_DISCOVER_INDEX_URL: &str = "https://rocm.frameworks.amd.com/whl-multi-arch/vllm/";
/// Index that publishes the pinned torch build for
/// [`VLLM_ROCM_DISCOVER_BUILD_TABLE`] rows.
const VLLM_ROCM_DISCOVER_TORCH_INDEX_URL: &str = "https://stable.repo.amd.com/rocm/whl-next/";
/// Looks up the discovery build recipe for a ROCm SDK version, if any.
///
/// Matched on major version only: unlike [`VLLM_ROCM_BUILD_TABLE`], where a
/// row pins one exact release's wheel filename, a discover row is a live
/// resolver recipe AMD's index applies uniformly across an entire ROCm major
/// line. AMD's preview wheels are tagged with the real target release
/// (`whl-multi-arch/torch/` carries `+rocm7.13.0`, `+rocm7.14.0`, and
/// `+rocm7.14.1` as genuinely distinct, coexisting builds), so once ROCm
/// 10.x's target moves past `10.0.0` the same rotation will happen here; a
/// row keyed to an exact string would then silently stop matching. `10.0.0`
/// and `10.1.0` should both discover through the same `"10.0.0"` row. This
/// intentionally differs from `apps/rocm/src/therock.rs`'s SDK layout
/// selection, which avoids major-only gating for unrelated reasons (on-disk
/// layout, not wheel availability).
pub(crate) fn vllm_rocm_discover_build(
    rocm_sdk_version: &str,
) -> Option<&'static VllmRocmDiscoverBuild> {
    VLLM_ROCM_DISCOVER_BUILD_TABLE
        .iter()
        .find(|build| rocm_sdk_major_matches(rocm_sdk_version, build.rocm_sdk_version))
}
/// Whether `recorded` (a runtime manifest's live `rocm_sdk.__version__` probe)
/// and `table_key` (a literal key in [`VLLM_ROCM_DISCOVER_BUILD_TABLE`]) share
/// a ROCm SDK major version, ignoring minor, patch, and any dev/pre-release
/// suffix. See [`vllm_rocm_discover_build`] for why major alone is enough here.
fn rocm_sdk_major_matches(recorded: &str, table_key: &str) -> bool {
    fn major(version: &str) -> Option<u64> {
        version.trim().split('.').next()?.parse().ok()
    }
    major(recorded).is_some() && major(recorded) == major(table_key)
}
/// Whether `recorded` (a runtime manifest's live `rocm_sdk.__version__` probe)
/// is the same release as `table_key` (a literal key in [`VLLM_ROCM_BUILD_TABLE`]),
/// ignoring any dev/pre-release suffix.
///
/// Real probes routinely carry a suffix the table keys never do (e.g.
/// `7.13.0a20260423`), and that suffix is not valid semver pre-release syntax
/// (no leading `-`), so comparing by exact string, or even by a semver parse,
/// would reject a matching release. Comparing by leading `major.minor.patch`
/// instead matches the release the table row actually covers.
fn rocm_sdk_version_matches(recorded: &str, table_key: &str) -> bool {
    fn release_triple(version: &str) -> Option<(u64, u64, u64)> {
        let mut parts = version.trim().split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        let patch: String = parts
            .next()?
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        Some((major, minor, patch.parse().ok()?))
    }
    release_triple(recorded).is_some_and(|version| Some(version) == release_triple(table_key))
}
/// How [`install_vllm_with_uv`] should install vLLM for a given target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VllmInstallRoute {
    /// Pin from [`VLLM_ROCM_BUILD_TABLE`] (or a caller-supplied index override).
    Static,
    /// Discover the current wheels for a [`VLLM_ROCM_DISCOVER_BUILD_TABLE`] row.
    RocmDiscover,
}
/// Picks the install route for a vLLM install. A caller-supplied index
/// override always wins (it means the caller already knows exactly which
/// wheels to use), otherwise a known ROCm SDK version routes through
/// discovery, and everything else falls back to the static pin table.
fn vllm_install_route(
    index_override: Option<&str>,
    rocm_sdk_version: Option<&str>,
) -> VllmInstallRoute {
    if index_override.is_some() {
        return VllmInstallRoute::Static;
    }
    match rocm_sdk_version.and_then(vllm_rocm_discover_build) {
        Some(_) => VllmInstallRoute::RocmDiscover,
        None => VllmInstallRoute::Static,
    }
}

pub(crate) fn install_response(request: InstallRequest) -> Result<InstallResponse> {
    // Resolve regardless of `reinstall`. Resolution answers *which* interpreter holds
    // vLLM, and a forced reinstall needs that answer just as much as a repair does —
    // gating it on `reinstall` left `--reinstall` with no assessed environment, so it
    // fell back to `crate::runtime::resolve_managed_runtime_python` (first prefix-matching candidate)
    // and could reinstall a healthy environment while leaving the broken one broken.
    // Only the short-circuit below is gated on `reinstall`.
    let resolved = crate::runtime::resolve_vllm_runtime(Some(&request.runtime_id)).ok();
    // A resolvable vLLM is not necessarily a usable one. `rocm install sdk` writes the
    // TheRock torch stack into the same environment vLLM lives in, so a second run
    // replaces the torch build vLLM pins without touching vLLM itself.
    // Short-circuiting on "vllm resolves" left that environment unrepaired; short-circuit
    // on "vllm's own requirements are met" instead, so the install below restores them.
    let (already_installed, repair, assessed) = match resolved {
        Some(runtime) => {
            let repair = assess_runtime_repair(&runtime);
            if request.reinstall || repair.needed {
                (
                    None,
                    repair,
                    crate::runtime::assessed_python_for_repair(&runtime),
                )
            } else {
                (Some(runtime), repair, None)
            }
        }
        None => (None, RepairAssessment::default(), None),
    };
    let mut discover_pins: Vec<String> = Vec::new();
    let runtime = if let Some(runtime) = already_installed {
        runtime
    } else {
        // A repair installs into the environment that was *assessed*. The two resolvers
        // do not agree: `crate::runtime::resolve_vllm_runtime` walks the candidates until one actually
        // has vLLM beside it, while `crate::runtime::resolve_managed_runtime_python` takes the first
        // candidate unconditionally — and `runtime_id` matches by prefix, so several
        // candidates routinely qualify. Installing into a different interpreter than the
        // one found broken would leave the broken one broken and report it fixed.
        let managed = match assessed {
            Some(assessed) => assessed,
            None => crate::runtime::resolve_managed_runtime_python(Some(&request.runtime_id))?.with_context(
                || {
                    let base = format!(
                        "runtime `{}` did not resolve to a managed TheRock Python environment for automatic vLLM install",
                        request.runtime_id
                    );
                    match crate::runtime::describe_skipped_managed_runtimes(Some(&request.runtime_id)) {
                        Some(skipped) => format!("{base}\n\n{skipped}"),
                        None => base,
                    }
                },
            )?,
        };
        discover_pins = install_vllm_with_uv(
            &managed.python_executable,
            request.reinstall,
            managed.rocm_sdk_version.as_deref(),
        )?;
        crate::runtime::resolve_vllm_runtime(Some(&managed.runtime_id)).with_context(|| {
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
        managed_env: Some(crate::runtime::runtime_is_managed(&runtime)),
        installed_packages: vec![format!(
            "vllm{}",
            runtime
                .version
                .as_deref()
                .map(|version| format!("=={version}"))
                .unwrap_or_default()
        )],
        capabilities: crate::capabilities(),
        lock_hash: crate::state::runtime_lock_hash(&runtime),
        warnings: repair
            .notes
            .into_iter()
            .chain(crate::runtime::vllm_runtime_warnings(&runtime))
            .chain((!discover_pins.is_empty()).then(|| {
                format!(
                    "vLLM ROCm 10.x discovery pinned: {}",
                    discover_pins.join(", ")
                )
            }))
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
/// Decide whether an already-resolvable vLLM environment must be reinstalled.
///
/// Only managed environments are assessed. An external environment belongs to the user:
/// rocm-cli reports on it but does not rewrite its packages, which is the very failure
/// mode this check exists to catch.
fn assess_runtime_repair(runtime: &VllmRuntime) -> RepairAssessment {
    if !crate::runtime::runtime_is_managed(runtime) {
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
            crate::runtime::recorded_sdk_torch_build(&runtime.runtime_id).as_deref(),
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
/// Installs vLLM into `python`, returning any resolved pins worth surfacing
/// to the caller as install warnings (empty for the static-pin route).
fn install_vllm_with_uv(
    python: &Path,
    reinstall: bool,
    rocm_sdk_version: Option<&str>,
) -> Result<Vec<String>> {
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
                return Ok(Vec::new());
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
    }
}
/// Resolves the exact requirement `uv` would install for
/// `{pkg}=={version_prefix}.*` from `index_url`, without installing anything.
///
/// `uv` has no `pip download` command (and never has — it's a declined
/// upstream feature request, astral-sh/uv#3163), so this uses `uv pip
/// install --dry-run` instead: it runs the real resolver against `python`'s
/// platform/interpreter tags and reports the version it would install on a
/// ` + {pkg}==<version>` line, which is parsed back out by
/// [`dry_run_resolved_pin`]. A prefix with no compatible build published
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
    version_prefix: &str,
) -> Result<String> {
    let requirement_prefix = format!("{pkg}=={version_prefix}.*");
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
        .arg(&requirement_prefix)
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
        let stderr_trimmed = stderr.trim();
        let detail = if !stderr_trimmed.is_empty() {
            stderr_trimmed.to_owned()
        } else if !stdout.trim().is_empty() {
            stdout.trim().to_owned()
        } else {
            "no output".to_owned()
        };
        bail!("`uv pip install --dry-run {requirement_prefix}` from {index_url} failed: {detail}");
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
                "`uv pip install --dry-run {requirement_prefix}` from {index_url} did not report a \
                 resolved version for {pkg}: {reported}"
            )
        })
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
/// Builds the `uv pip install` argv for a ROCm 10.x discovery install: the
/// resolved `pins` plus `--prerelease allow` and both discovery indexes.
fn vllm_rocm10_discover_install_args(
    python: &Path,
    reinstall: bool,
    pins: &[String],
) -> Vec<String> {
    let mut args = uv_pip_install_base(python);
    if reinstall {
        args.push("--reinstall".to_owned());
    }
    args.extend(pins.iter().cloned());
    args.push("--prerelease".to_owned());
    args.push("allow".to_owned());
    args.push("--extra-index-url".to_owned());
    args.push(VLLM_ROCM_DISCOVER_INDEX_URL.to_owned());
    args.push("--extra-index-url".to_owned());
    args.push(VLLM_ROCM_DISCOVER_TORCH_INDEX_URL.to_owned());
    args
}
/// Discovers and installs the current vLLM/flash-attn/amd-aiter wheels for a
/// [`VllmRocmDiscoverBuild`] row, pinning each to the exact version `uv pip
/// install --dry-run` resolved so the real install can never silently drift
/// to a different (or non-ROCm) build. Returns the five pins actually installed.
fn install_vllm_rocm10_discover(
    uv: &Path,
    paths: &AppPaths,
    python: &Path,
    reinstall: bool,
    build: &VllmRocmDiscoverBuild,
) -> Result<Vec<String>> {
    let vllm = discover_pinned_requirement(
        uv,
        paths,
        python,
        VLLM_ROCM_DISCOVER_INDEX_URL,
        "vllm",
        build.vllm_version_prefix,
    )?;
    let flash_attn = discover_pinned_requirement(
        uv,
        paths,
        python,
        VLLM_ROCM_DISCOVER_INDEX_URL,
        "flash-attn",
        build.flash_attn_version_prefix,
    )?;
    let amd_aiter = discover_pinned_requirement(
        uv,
        paths,
        python,
        VLLM_ROCM_DISCOVER_INDEX_URL,
        "amd-aiter",
        build.amd_aiter_version_prefix,
    )?;

    let pins = vec![
        build.torch_requirement.to_owned(),
        vllm,
        flash_attn,
        amd_aiter,
        build.tensorizer_requirement.to_owned(),
    ];

    let args = vllm_rocm10_discover_install_args(python, reinstall, &pins);
    let output = ProcessCommand::new(uv)
        .args(args)
        .envs(uv_command_env(paths))
        .output()
        .context("failed to launch uv pip install for vLLM (ROCm 10.x discovery)")?;
    if output.status.success() {
        return Ok(pins);
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
        "`uv pip install {}` failed for {}: {}",
        pins.join(" "),
        python.display(),
        detail
    )
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
/// Index URL and requirement for the given runtime's recorded ROCm SDK version.
fn vllm_install_target(rocm_sdk_version: Option<&str>) -> Result<VllmInstallTarget> {
    resolve_vllm_install_target(
        std::env::var("ROCM_CLI_VLLM_ROCM_INDEX_URL").ok(),
        rocm_sdk_version,
    )
}
/// Resolve the wheel index and the exact requirement to install from it.
///
/// With `ROCM_CLI_VLLM_ROCM_INDEX_URL` set to a non-blank value, the build is
/// recovered from the URL when it has the published
/// `{VLLM_ROCM_INDEX_PREFIX}/<version>/<abi>` shape — so pointing at another
/// release of the same index works *and* stays pinned, with no extra
/// configuration. A URL of any other shape cannot be pinned automatically, and
/// dropping the pin is not an option: the install passes `--extra-index-url`,
/// so PyPI stays in play and a bare `vllm` can silently resolve to the
/// non-ROCm build. Such a URL is therefore rejected rather than installed
/// unpinned.
///
/// Without an override, `rocm_sdk_version` (the ROCm SDK version recorded in
/// the target runtime's manifest) is looked up in [`VLLM_ROCM_BUILD_TABLE`].
/// A version with no row falls back to the table's first (default) row,
/// matching the single unconditional pin `rocm-cli` used before per-version
/// rows existed, rather than leaving an unrecognized version uninstallable,
/// *unless* the version's major release matches a
/// [`VLLM_ROCM_DISCOVER_BUILD_TABLE`] row (see [`vllm_rocm_discover_build`]),
/// in which case guessing the default row's wheel would very likely install
/// an ABI-incompatible build, so that case fails closed instead. In practice
/// callers route such a version through discovery before ever reaching this
/// function (see `vllm_install_route`); this is a safety net for the case
/// where they don't.
fn resolve_vllm_install_target(
    index_override: Option<String>,
    rocm_sdk_version: Option<&str>,
) -> Result<VllmInstallTarget> {
    let index_override = index_override
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());

    if let Some(index_url) = index_override {
        let (version, abi) = vllm_rocm_build_from_index_url(&index_url).ok_or_else(|| {
            anyhow!(
                "cannot determine which vLLM build `{index_url}` serves: it is not a published \
                 release index of the form `{VLLM_ROCM_INDEX_PREFIX}/<version>/<abi>`. Installing \
                 an unpinned `vllm` instead would let the resolver fall back to PyPI's non-ROCm \
                 build, so point ROCM_CLI_VLLM_ROCM_INDEX_URL at a release index, or unset it to \
                 use the built-in one."
            )
        })?;

        return Ok(VllmInstallTarget {
            index_url,
            requirement: format!("vllm=={version}+{abi}"),
        });
    }

    let rocm_sdk_version = rocm_sdk_version.ok_or_else(|| {
        anyhow!(
            "cannot install vLLM: the target runtime's ROCm SDK version could not be determined \
             from its manifest, so no compatible vLLM build can be selected. Set \
             ROCM_CLI_VLLM_ROCM_INDEX_URL to a published release index to install anyway."
        )
    })?;

    let default_build = VLLM_ROCM_BUILD_TABLE
        .first()
        .ok_or_else(|| anyhow!("VLLM_ROCM_BUILD_TABLE has no default row"))?;
    let build = match VLLM_ROCM_BUILD_TABLE
        .iter()
        .find(|build| rocm_sdk_version_matches(rocm_sdk_version, build.rocm_sdk_version))
    {
        Some(build) => build,
        None if VLLM_ROCM_DISCOVER_BUILD_TABLE
            .iter()
            .any(|build| rocm_sdk_major_matches(rocm_sdk_version, build.rocm_sdk_version)) =>
        {
            bail!(
                "cannot install vLLM: ROCm SDK version `{rocm_sdk_version}` has no static \
                 build pin, and its major release is only known to `rocm-cli` via live wheel \
                 discovery, not a static pin; guessing the default \
                 vllm=={}+{} build would very likely install an incompatible wheel. Set \
                 ROCM_CLI_VLLM_ROCM_INDEX_URL to a published release index to install anyway.",
                default_build.vllm_version,
                default_build.abi,
            );
        }
        None => default_build,
    };

    Ok(VllmInstallTarget {
        index_url: format!(
            "{VLLM_ROCM_INDEX_PREFIX}/{}/{}",
            build.vllm_version, build.abi
        ),
        requirement: format!("vllm=={}+{}", build.vllm_version, build.abi),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{RocmSdkRuntimeProbe, TheRockRuntimeManifest};

    fn install_target(
        index: Option<&str>,
        rocm_sdk_version: Option<&str>,
    ) -> Result<VllmInstallTarget> {
        resolve_vllm_install_target(index.map(ToOwned::to_owned), rocm_sdk_version)
    }
    fn violation(requiring: &str, detail: &str) -> DependencyViolation {
        DependencyViolation {
            requiring: requiring.to_owned(),
            detail: detail.to_owned(),
        }
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
    fn a_settled_runtime_converges_on_the_build_its_own_manifest_records() {
        // The convergence proof the tests above cannot give on their own. They hand
        // the classification a build literal, so a change to what
        // `crate::runtime::sdk_torch_build_from_manifest` yields — `7.13.0` where the local segment
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
            let build = crate::runtime::sdk_torch_build_from_manifest(&manifest)
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
    fn vllm_install_target_resolves_a_known_rocm_sdk_version() {
        let build = VLLM_ROCM_BUILD_TABLE
            .first()
            .expect("build table has at least one row for this test to check");
        let target = install_target(None, Some(build.rocm_sdk_version))
            .expect("a table row resolves to a pinned target");
        assert_eq!(
            target.index_url,
            format!(
                "{VLLM_ROCM_INDEX_PREFIX}/{}/{}",
                build.vllm_version, build.abi
            )
        );
        assert_eq!(
            target.requirement,
            format!("vllm=={}+{}", build.vllm_version, build.abi)
        );

        let blank = install_target(Some("  "), Some(build.rocm_sdk_version))
            .expect("a blank override is ignored");
        assert_eq!(blank, target);
    }
    #[test]
    fn vllm_install_target_falls_back_to_the_default_row_for_an_unknown_rocm_sdk_version() {
        let default_build = VLLM_ROCM_BUILD_TABLE
            .first()
            .expect("build table has at least one row for this test to check");

        for unknown_version in ["999.0.0", "7.13.0a20260326", "not-a-version"] {
            let target = install_target(None, Some(unknown_version))
                .expect("an unrecognized version still resolves to the default pin");
            assert_eq!(
                target.requirement,
                format!("vllm=={}+{}", default_build.vllm_version, default_build.abi)
            );
        }
    }
    #[test]
    fn vllm_install_target_refuses_to_guess_a_static_pin_for_an_unmatched_discover_major() {
        // `10.1.0a20260822` shares a major with the `VLLM_ROCM_DISCOVER_BUILD_TABLE`
        // row but does not match any `VLLM_ROCM_BUILD_TABLE` row, so guessing the
        // ROCm 7.2.3 default here would install an incompatible wheel. Callers
        // route this version through discovery before reaching this function (see
        // `vllm_install_route`); this checks the fallback itself fails closed.
        let error = install_target(None, Some("10.1.0a20260822"))
            .expect_err("an unmatched version in a known discovery major must not guess")
            .to_string();
        assert!(error.contains("10.1.0a20260822"), "{error}");
        assert!(error.contains("ROCM_CLI_VLLM_ROCM_INDEX_URL"), "{error}");
    }
    #[test]
    fn vllm_install_target_fails_when_no_rocm_sdk_version_is_known() {
        let error = install_target(None, None)
            .expect_err("with no override and no version, nothing can be pinned")
            .to_string();
        assert!(error.contains("could not be determined"), "{error}");
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
            let target = install_target(Some(index), None).expect("published index shape resolves");
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
            let error = install_target(Some(index), None)
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
    fn every_build_table_row_has_a_self_consistent_index_and_requirement() {
        for build in VLLM_ROCM_BUILD_TABLE {
            let index_url = format!(
                "{VLLM_ROCM_INDEX_PREFIX}/{}/{}",
                build.vllm_version, build.abi
            );
            assert_eq!(
                vllm_rocm_build_from_index_url(&index_url),
                Some((build.vllm_version.to_owned(), build.abi.to_owned())),
                "row for ROCm SDK {} must parse back to its own build",
                build.rocm_sdk_version
            );
        }
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
    fn vllm_rocm_discover_build_looks_up_known_and_unknown_versions() {
        assert!(vllm_rocm_discover_build("10.0.0").is_some());
        // AMD tags preview wheels with the real target release (verified via
        // whl-multi-arch/torch/'s coexisting +rocm7.13.0/7.14.0/7.14.1
        // builds), so ROCm 10.x's tag will move past 10.0.0 the same way;
        // the discovery recipe must keep firing across the whole major line,
        // not just the exact version it happened to be added for.
        assert!(vllm_rocm_discover_build("10.1.0a20260822").is_some());
        assert!(vllm_rocm_discover_build("999.0.0").is_none());
    }
    #[test]
    fn rocm_sdk_version_matches_ignores_dev_suffixes_and_rejects_other_releases() {
        assert!(rocm_sdk_version_matches("10.0.0", "10.0.0"));
        assert!(rocm_sdk_version_matches("7.13.0a20260423", "7.13.0"));
        assert!(rocm_sdk_version_matches("7.2.3.dev0+abc", "7.2.3"));
        assert!(!rocm_sdk_version_matches("7.14.1", "7.2.3"));
        assert!(!rocm_sdk_version_matches("garbage", "7.2.3"));
    }
    #[test]
    fn rocm_sdk_major_matches_ignores_minor_patch_and_dev_suffixes() {
        assert!(rocm_sdk_major_matches("10.0.0", "10.0.0"));
        assert!(rocm_sdk_major_matches("10.1.0a20260822", "10.0.0"));
        assert!(rocm_sdk_major_matches("10.99.7.dev0+abc", "10.0.0"));
        assert!(!rocm_sdk_major_matches("7.13.0", "10.0.0"));
        assert!(!rocm_sdk_major_matches("garbage", "10.0.0"));
    }
    #[test]
    fn vllm_install_route_prefers_an_index_override_even_for_a_discover_version() {
        assert_eq!(
            vllm_install_route(Some("https://example.test/rocm"), Some("10.0.0")),
            VllmInstallRoute::Static
        );
    }
    #[test]
    fn vllm_install_route_discovers_for_a_known_discover_version() {
        assert_eq!(
            vllm_install_route(None, Some("10.0.0")),
            VllmInstallRoute::RocmDiscover
        );
        // A same-major, different-minor/patch nightly must still route
        // through discovery rather than falling back to the static table.
        assert_eq!(
            vllm_install_route(None, Some("10.1.0a20260822")),
            VllmInstallRoute::RocmDiscover
        );
    }
    #[test]
    fn vllm_install_route_falls_back_to_static_for_unknown_or_missing_versions() {
        assert_eq!(
            vllm_install_route(None, Some("7.2.3")),
            VllmInstallRoute::Static
        );
        assert_eq!(vllm_install_route(None, None), VllmInstallRoute::Static);
    }
    #[test]
    fn vllm_rocm10_discover_install_args_includes_pins_and_both_indexes() {
        let pins = vec![
            "torch==2.12.0+rocm10.0.0".to_owned(),
            "vllm==0.27.1.dev5+rocm10.0.0".to_owned(),
            "flash-attn==2.8.3".to_owned(),
            "amd-aiter==0.1.4".to_owned(),
            "tensorizer==2.12.1".to_owned(),
        ];
        let python = PathBuf::from("/opt/venv/bin/python");

        let args = vllm_rocm10_discover_install_args(&python, false, &pins);
        assert!(!args.contains(&"--reinstall".to_owned()));
        for pin in &pins {
            assert!(args.contains(pin), "{args:?} should contain {pin}");
        }
        assert!(args.contains(&"--prerelease".to_owned()));
        assert!(args.contains(&"allow".to_owned()));
        assert!(args.contains(&VLLM_ROCM_DISCOVER_INDEX_URL.to_owned()));
        assert!(args.contains(&VLLM_ROCM_DISCOVER_TORCH_INDEX_URL.to_owned()));

        let args = vllm_rocm10_discover_install_args(&python, true, &pins);
        assert!(args.contains(&"--reinstall".to_owned()));
    }
}
