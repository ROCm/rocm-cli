// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, anyhow, bail};
use rocm_core::{
    AppPaths, DependencyViolation, check_dependencies, ensure_uv_binary, split_local_version,
    uv_command_env, uv_pip_freeze_args, uv_pip_install_base, violation_subject,
    violations_requiring,
};
use rocm_engine_protocol::{InstallRequest, InstallResponse};
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

use crate::ENGINE_NAME;
use crate::runtime::{
    VllmRuntime, assessed_python_for_repair, describe_skipped_managed_runtimes,
    recorded_sdk_torch_build, resolve_managed_runtime_python, resolve_vllm_runtime,
    runtime_is_managed, vllm_runtime_warnings,
};
use crate::state::runtime_lock_hash;

/// Known-good `(ROCm SDK version, vLLM version, ABI tag)` combinations for
/// `uv pip install vllm`, keyed by the ROCm SDK version recorded in the
/// runtime manifest (see [`crate::runtime::rocm_sdk_version_from_manifest`]).
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
/// How a [`VllmRocmDiscoverBuild`] row constrains the release discovery may
/// resolve for one package.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DiscoverVersion {
    /// `{pkg}=={series}.*`: the newest release in the series.
    Series(&'static str),
    /// `{pkg}=={release}`: exactly this release. Under PEP 440 a specifier
    /// without a local version still matches any `+rocm…` local version of
    /// it, but not a longer release such as `{release}.2`. Needed where AMD
    /// publishes rebuilds as extra release segments (torchaudio
    /// `2.11.0`/`2.11.0.2`/`2.11.0.3`, each paired with a different torch), so
    /// a series match would pick the rebuild for the wrong torch.
    Exact(&'static str),
}
impl DiscoverVersion {
    /// The requirement handed to `uv` for `pkg`.
    fn requirement(self, pkg: &str) -> String {
        match self {
            Self::Series(series) => format!("{pkg}=={series}.*"),
            Self::Exact(release) => format!("{pkg}=={release}"),
        }
    }
}
/// A ROCm SDK version whose vLLM/flash-attn/amd-aiter wheels aren't published
/// under a fixed filename (AMD rotates the dev-tag suffix constantly), so the
/// exact wheel must be discovered from the index at install time instead of
/// pinned in [`VLLM_ROCM_BUILD_TABLE`].
pub(crate) struct VllmRocmDiscoverBuild {
    rocm_sdk_version: &'static str,
    /// PEP 425 interpreter tag this row's wheels are published for. AMD builds
    /// ROCm 10.x `vllm`, `flash-attn` and `amd-aiter` for exactly one CPython
    /// version, so a venv on any other interpreter resolves nothing at all.
    python_tag: &'static str,
    /// Index publishing this row's rotating-dev-tag vLLM/flash-attn/amd-aiter
    /// wheels. Per-row because ROCm 10.1 frameworks are staged on a different
    /// host than 10.0's.
    vllm_index_url: &'static str,
    /// Index publishing this row's torch build.
    torch_index_url: &'static str,
    vllm_version_prefix: &'static str,
    flash_attn_version_prefix: &'static str,
    amd_aiter_version_prefix: &'static str,
    torch_version_prefix: &'static str,
    /// torchvision's version prefix is always the ROCm "Next" torch minor
    /// plus 15 (torch 2.12 pairs with torchvision 0.27); torchaudio has no
    /// such rule and is hardcoded per row to whatever AMD has actually
    /// published for it. Both are discovered fresh alongside torch rather
    /// than trusted from a pre-install snapshot, so the triplet installed
    /// here is always coherent with the torch version this row actually pins
    /// (see `install_vllm_rocm10_discover`).
    torchvision_version_prefix: &'static str,
    /// [`DiscoverVersion::Exact`] wherever the row's index publishes more than
    /// one rebuild of a torchaudio series, which `whl-next` does (`2.11.0`,
    /// `2.11.0.2`, `2.11.0.3`, each paired with a different torch): a series
    /// match takes the newest, i.e. the rebuild for some *other* row's torch.
    /// [`DiscoverVersion::Series`] is only safe on an index that serves one
    /// build per series.
    torchaudio_version: DiscoverVersion,
}

const VLLM_ROCM_DISCOVER_BUILD_TABLE: &[VllmRocmDiscoverBuild] = &[
    VllmRocmDiscoverBuild {
        rocm_sdk_version: "10.0.0",
        python_tag: "cp314",
        vllm_index_url: "https://rocm.frameworks.amd.com/whl-multi-arch/vllm/",
        torch_index_url: "https://stable.repo.amd.com/rocm/whl-next/",
        vllm_version_prefix: "0.27",
        flash_attn_version_prefix: "2.8",
        amd_aiter_version_prefix: "0.1",
        torch_version_prefix: "2.12",
        torchvision_version_prefix: "0.27",
        // Still a series match, unlike the 10.1 row below, and unverified
        // against `whl-next` now that both rows read from it: that index is
        // flat across ROCm lines, so a series match here can resolve a
        // `+rocm10.1.0` wheel, which `ensure_rocm_local_version_matches`
        // rejects. Left as-is rather than pinned blind -- which torchaudio
        // rebuild pairs with 10.0's torch 2.12 needs a live 10.0 box to
        // establish, and this row has no coverage on the self-hosted lanes.
        torchaudio_version: DiscoverVersion::Series("2.11"),
    },
    VllmRocmDiscoverBuild {
        rocm_sdk_version: "10.1.0",
        python_tag: "cp314",
        vllm_index_url: "https://rocm.frameworks-prereleases.amd.com/whl-multi-arch-staging/vllm/",
        // Torch stack from `whl-next`, not staging: it is the index
        // `rocm install sdk` resolves the 10.1 SDK's own `+rocm10.1.0` torch
        // stack from, so vLLM realigns onto the same builds rather than
        // staging's `rc` ones.
        torch_index_url: "https://stable.repo.amd.com/rocm/whl-next/",
        vllm_version_prefix: "0.29",
        flash_attn_version_prefix: "2.8",
        amd_aiter_version_prefix: "0.1",
        // 2.13, not 2.12: AMD's own `rocm/vllm:rocm10.1.0_..._vllm-0.29.0`
        // image (whose Dockerfile build args pin `TORCH_VERSION=2.13.0`,
        // `TORCHVISION_VERSION=0.28.0`) builds this exact vllm release
        // against torch 2.13, not 2.12. `vllm` 0.29's own `_rocm_C.abi3.so`
        // needs a `c10::NotImplementedError` constructor symbol torch
        // 2.12.0+rocm10.1.0 does not export; 2.13 is published on this same
        // `whl-next` index and matches what vllm was actually built against.
        torch_version_prefix: "2.13",
        torchvision_version_prefix: "0.28",
        // Exact, not `2.11.*`: `whl-next` publishes three `+rocm10.1.0`
        // torchaudio builds, `2.11.0`, `2.11.0.2` and `2.11.0.3`. `2.11.0` is
        // a CPU-only rebuild (no `libamdhip64`/`libc10_hip`/`libtorch_hip`
        // in its `.so`s) that predates torch's stable-ABI split and does not
        // reference the `torch_exception_get_what` / `torch_new_stable_ivalue`
        // symbols torch 2.13 exports. `2.11.0.2` and `2.11.0.3` are both HIP
        // builds against that same stable ABI (confirmed on a live Strix Halo
        // box: torch 2.13.0+rocm10.1.0 exports both symbols as `T`, and
        // torchaudio 2.11.0.2 imports them as `U`); a series match would
        // resolve the newest, `2.11.0.3`, so pin the lower of the two
        // confirmed-working builds instead.
        torchaudio_version: DiscoverVersion::Exact("2.11.0.2"),
    },
];
/// Looks up the discovery build recipe for a ROCm SDK version, if any.
///
/// Matched on `major.minor`, ignoring patch and any dev/pre-release suffix. A
/// discover row is a live resolver recipe rather than one pinned filename, so
/// it does span a release line, but only a `major.minor` one: 10.0 and 10.1
/// publish genuinely different wheels (vLLM `0.27` from
/// `rocm.frameworks.amd.com` versus `0.29` from the staging host), so a
/// major-only row would serve one SDK's wheels to the other. Patch is still
/// ignored, because AMD does rotate the patch and dev tag within a line
/// (`whl-multi-arch/torch/` carries `+rocm7.13.0`, `+rocm7.14.0` and
/// `+rocm7.14.1` as coexisting builds) and a row keyed to an exact string
/// would silently stop matching. An unknown line matches nothing and fails
/// closed in [`resolve_vllm_install_target`] rather than falling back to a
/// stale static pin.
pub(crate) fn vllm_rocm_discover_build(
    rocm_sdk_version: &str,
) -> Option<&'static VllmRocmDiscoverBuild> {
    VLLM_ROCM_DISCOVER_BUILD_TABLE
        .iter()
        .find(|build| rocm_sdk_series_matches(rocm_sdk_version, build.rocm_sdk_version))
}
/// Whether `recorded` and `table_key` name the same ROCm `major.minor` line,
/// ignoring patch and any dev/pre-release suffix. See
/// [`vllm_rocm_discover_build`] for why the line, not the exact release, is
/// what a discover row covers.
fn rocm_sdk_series_matches(recorded: &str, table_key: &str) -> bool {
    fn series(version: &str) -> Option<(u64, u64)> {
        let mut parts = version.trim().split('.');
        let major = parts.next()?.parse().ok()?;
        let minor: String = parts
            .next()?
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        Some((major, minor.parse().ok()?))
    }
    series(recorded).is_some_and(|version| Some(version) == series(table_key))
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
pub(crate) enum VllmInstallRoute {
    /// Pin from [`VLLM_ROCM_BUILD_TABLE`] (or a caller-supplied index override).
    Static,
    /// Discover the current wheels for a [`VLLM_ROCM_DISCOVER_BUILD_TABLE`] row.
    RocmDiscover,
}
/// Picks the install route for a vLLM install. A caller-supplied index
/// override always wins (it means the caller already knows exactly which
/// wheels to use), otherwise a known ROCm SDK version routes through
/// discovery, and everything else falls back to the static pin table.
pub(crate) fn vllm_install_route(
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
        discover_pins = install_vllm_with_uv(
            &managed.python_executable,
            request.reinstall,
            managed.rocm_sdk_version.as_deref(),
        )?;
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
    let requirement = version.requirement(pkg);
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
        .arg(&requirement)
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
/// to a different (or non-ROCm) build. Returns the pins actually installed.
fn install_vllm_rocm10_discover(
    uv: &Path,
    paths: &AppPaths,
    python: &Path,
    reinstall: bool,
    build: &VllmRocmDiscoverBuild,
) -> Result<Vec<String>> {
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
    let flash_attn = discover_pinned_requirement(
        uv,
        paths,
        python,
        build.vllm_index_url,
        "flash-attn",
        DiscoverVersion::Series(build.flash_attn_version_prefix),
    )?;
    let amd_aiter = discover_pinned_requirement(
        uv,
        paths,
        python,
        build.vllm_index_url,
        "amd-aiter",
        DiscoverVersion::Series(build.amd_aiter_version_prefix),
    )?;

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
    let config_file = write_vllm_index_uv_config(paths, build)?;
    remaining_args.push("--config-file".to_owned());
    remaining_args.push(config_file.display().to_string());
    run_uv_pip_install(uv, paths, python, remaining_args)?;

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
    Ok(pins)
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
/// function (see [`vllm_install_route`]); this is a safety net for the case
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
    use crate::runtime::{
        RocmSdkRuntimeProbe, TheRockRuntimeManifest, sdk_torch_build_from_manifest,
    };

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
        // builds), so a pre-release suffix on a known major.minor must still
        // match its row (here the dedicated 10.1.0 one), not just an exact
        // version string.
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
    /// A series renders with the `.*` that lets `uv` take the newest release
    /// in it; an exact release renders bare, so it cannot.
    #[test]
    fn discover_version_renders_series_and_exact_requirements() {
        assert_eq!(
            DiscoverVersion::Series("2.11").requirement("torchaudio"),
            "torchaudio==2.11.*"
        );
        // No trailing `.*`: that would also admit the `2.11.0.2`/`2.11.0.3`
        // rebuilds an exact pin exists to exclude.
        assert_eq!(
            DiscoverVersion::Exact("2.11.0").requirement("torchaudio"),
            "torchaudio==2.11.0"
        );
    }
    /// The 10.1 row must reach the staging host for vLLM, which is only
    /// published there, but takes its torch stack from `whl-next` just like
    /// 10.0, the same index `rocm install sdk` resolves its own
    /// `+rocm10.1.0` torch stack from (though the SDK's resolve and this
    /// row's discovery are independent: this row pins its own version
    /// prefixes rather than matching whatever the SDK happened to install).
    /// Its torchaudio is pinned exactly: a `2.11.*` match resolves the
    /// newest rebuild, `2.11.0.3`; `2.11.0.2` is the lower of the two builds
    /// confirmed (on a live Strix Halo box) to load against torch 2.13
    /// (therock-next-09). 10.0 keeps the production frameworks index for
    /// vLLM. The live half of this is therock-next-09.
    #[test]
    fn discover_rows_select_their_own_indexes() {
        let ten_zero = vllm_rocm_discover_build("10.0.0").expect("10.0.0 has a discover row");
        assert_eq!(
            ten_zero.vllm_index_url,
            "https://rocm.frameworks.amd.com/whl-multi-arch/vllm/"
        );
        assert_eq!(
            ten_zero.torch_index_url,
            "https://stable.repo.amd.com/rocm/whl-next/"
        );
        assert_eq!(ten_zero.vllm_version_prefix, "0.27");

        // A real probe carries a patch and dev suffix the table key never does.
        let ten_one = vllm_rocm_discover_build("10.1.0a20260928").expect("10.1 has a discover row");
        assert_eq!(
            ten_one.vllm_index_url,
            "https://rocm.frameworks-prereleases.amd.com/whl-multi-arch-staging/vllm/"
        );
        assert_eq!(
            ten_one.torch_index_url,
            "https://stable.repo.amd.com/rocm/whl-next/"
        );
        assert_eq!(ten_one.vllm_version_prefix, "0.29");
        // 2.13/0.28, not 2.12/0.27: the pair AMD's own
        // `rocm/vllm:...vllm-0.29.0` image builds this vllm release against.
        assert_eq!(ten_one.torch_version_prefix, "2.13");
        assert_eq!(ten_one.torchvision_version_prefix, "0.28");
        // `whl-next` carries `2.11.0`, `2.11.0.2` and `2.11.0.3` for
        // `+rocm10.1.0`; `2.11.0` is CPU-only, and a series match would
        // resolve the newest, `2.11.0.3`, not the confirmed-working
        // `2.11.0.2`, so this must stay exact and never a series.
        assert_eq!(
            ten_one.torchaudio_version,
            DiscoverVersion::Exact("2.11.0.2")
        );

        // An unpublished line must not borrow another line's wheels.
        assert!(vllm_rocm_discover_build("10.2.0").is_none());
        assert!(vllm_rocm_discover_build("7.2.3").is_none());
    }
    /// An unknown 10.x line fails closed rather than silently resolving the
    /// static table's 7.2.3 pin, which is the whole point of keeping
    /// [`rocm_sdk_major_matches`] alongside the narrower row lookup.
    #[test]
    fn unknown_rocm10_line_still_fails_closed() {
        let err = install_target(None, Some("10.2.0")).expect_err("10.2.0 has no vLLM build");
        let rendered = format!("{err:#}");
        assert!(
            !rendered.contains("7.2.3"),
            "should not fall back to the static pin: {rendered}"
        );
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
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir()?;
        let paths = AppPaths {
            config_dir: root.path().join("config"),
            data_dir: root.path().join("data"),
            cache_dir: root.path().join("cache"),
        };
        let log = root.path().join("uv-calls.log");

        let python = root.path().join("python");
        std::fs::write(&python, "#!/bin/sh\necho cp314\n")?;
        std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755))?;

        // Answers every `--dry-run` discovery call with a resolved pin for
        // whatever package the requirement names, and otherwise just records
        // the call (the real installs) for the assertions below.
        let uv = root.path().join("uv");
        std::fs::write(
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
        std::fs::set_permissions(&uv, std::fs::Permissions::from_mode(0o755))?;

        let build = vllm_rocm_discover_build("10.1.0").expect("10.1.0 has a discover row");
        let pins = install_vllm_rocm10_discover(&uv, &paths, &python, true, build)?;
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
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir()?;
        let paths = AppPaths {
            config_dir: root.path().join("config"),
            data_dir: root.path().join("data"),
            cache_dir: root.path().join("cache"),
        };
        let log = root.path().join("uv-calls.log");

        let python = root.path().join("python");
        std::fs::write(&python, "#!/bin/sh\necho cp314\n")?;
        std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755))?;

        let uv = root.path().join("uv");
        std::fs::write(
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
        std::fs::set_permissions(&uv, std::fs::Permissions::from_mode(0o755))?;

        let build = vllm_rocm_discover_build("10.1.0").expect("10.1.0 has a discover row");
        let pins = install_vllm_rocm10_discover(&uv, &paths, &python, true, build)?;
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
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir()?;
        let paths = AppPaths {
            config_dir: root.path().join("config"),
            data_dir: root.path().join("data"),
            cache_dir: root.path().join("cache"),
        };
        let log = root.path().join("uv-calls.log");

        let python = root.path().join("python");
        std::fs::write(&python, "#!/bin/sh\necho cp314\n")?;
        std::fs::set_permissions(&python, std::fs::Permissions::from_mode(0o755))?;

        let uv = root.path().join("uv");
        std::fs::write(
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
        std::fs::set_permissions(&uv, std::fs::Permissions::from_mode(0o755))?;

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
