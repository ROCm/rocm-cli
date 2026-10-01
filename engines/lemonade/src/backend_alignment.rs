// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, RocmCliConfig, active_managed_therock_version, runtime_is_linux};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::time::Duration;

use crate::direct_llama::find_llama_server_binary_for_backend;
use crate::install::{LemonadeInstallManifest, read_manifest};
use crate::process::{
    LemonadeProcessEnvironment, apply_lemonade_process_environment, hide_child_console_window,
    lemonade_process_environment, run_lemonade_backend_install, spawn_lemond,
    wait_for_lemonade_cli_status,
};
use crate::state::{free_local_port, terminate_pid};
use crate::{DEFAULT_HOST, LLAMACPP_RECIPE, ROCM_BACKEND_NAME};

/// Resource file inside the extracted Lemonade embeddable pinning which
/// llama.cpp build (and paired ROCm version) `llamacpp:rocm` downloads. Its own
/// header comment invites hand-editing: "You can modify these values to pin
/// specific versions without rebuilding the application."
const BACKEND_VERSIONS_RESOURCE: &str = "resources/backend_versions.json";

/// Preferred llama.cpp backends, best first. Lemonade reports per-GPU support;
/// we pick the highest-priority backend it considers supported on this host.
/// GPU backends only — `cpu` is intentionally excluded so the router path never
/// selects CPU under the GPU-required policy (AGENTS.md §6, matching
/// `DIRECT_LLAMA_SERVER_BACKENDS`). When only CPU is usable, selection returns
/// `None` and the caller fails with actionable guidance.
const LLAMACPP_BACKEND_PRIORITY: [&str; 2] = ["rocm", "vulkan"];

#[derive(Debug, Clone)]
pub(crate) struct LemonadeRuntime {
    pub(crate) manifest: LemonadeInstallManifest,
}

/// The variable that opts a machine out of rocm-cli aligning Lemonade's
/// `llamacpp:rocm` backend to the active ROCm SDK.
///
/// `resources/backend_versions.json` is a documented, first-party
/// customization point — its own header comment invites users to hand-edit it
/// to pin specific versions without rebuilding — so alignment running
/// unconditionally on every plain install would silently overwrite a user's
/// manual pin. Mirrors vLLM's `ROCM_CLI_DISABLE_TORCH_ALIGNMENT`
/// (`rocm_core::TORCH_ALIGNMENT_DISABLED_ENV`): presence is the signal, so any
/// value — including the empty string — disables the alignment.
pub const LEMONADE_BACKEND_ALIGNMENT_DISABLED_ENV: &str =
    "ROCM_CLI_DISABLE_LEMONADE_BACKEND_ALIGNMENT";

/// Whether the user has opted out of rocm-cli aligning Lemonade's backend.
fn lemonade_backend_alignment_disabled() -> bool {
    std::env::var_os(LEMONADE_BACKEND_ALIGNMENT_DISABLED_ENV).is_some()
}

/// Point Lemonade's `llamacpp:rocm` backend at the ROCm version rocm-cli already has
/// installed and active, instead of Lemonade's hardcoded pin (which does not track
/// whatever the user separately installed via `rocm install sdk`).
///
/// Best-effort and defensive at every step: if there is no managed rocm-cli SDK, the
/// pinned version already matches, or an aligned attempt cannot be verified to actually
/// resolve its GPU library, this falls back to (or reverts to) today's unmodified
/// pinned-version install — never a regression, and never a silent CPU fallback
/// (AGENTS.md §6): an aligned backend is only kept once [`rocm_backend_resolves`]
/// confirms it.
///
/// Returns the aligned version string when one was applied and verified, so the caller
/// can surface it in the install response.
pub(crate) fn prepare_llamacpp_backend_for_active_rocm(
    paths: &AppPaths,
    manifest: &mut LemonadeInstallManifest,
) -> Result<Option<String>> {
    prepare_llamacpp_backend_for_active_rocm_impl(
        manifest,
        lemonade_backend_alignment_disabled(),
        || active_rocm_sdk_version_for_alignment(paths),
        read_backend_versions_therock_version,
        best_llamacpp_backend_for_host,
        install_best_llamacpp_backend,
        |manifest, backend_versions_path, target_version, pinned_version, disabled| {
            align_llamacpp_backend_to_version(
                manifest,
                backend_versions_path,
                target_version,
                pinned_version,
                disabled,
                try_llamacpp_backend_alignment,
                install_best_llamacpp_backend,
                latest_llamacpp_rocm_stable_tag,
            )
        },
    )
}

/// The gate and lookup chain behind [`prepare_llamacpp_backend_for_active_rocm`],
/// isolated from its real dependencies (the active-SDK/config lookup, the packaged
/// pin file, and the `lemond`-spawning backend probe) so this crate's own unit tests
/// can drive the disabled early return and the probe-error fallback -- the two
/// branches that decide whether any alignment is even attempted, and previously had
/// no coverage outside a `@nightly @requires-gpu` e2e lane. `disabled` is
/// [`lemonade_backend_alignment_disabled`]'s resolved value, passed in for the same
/// reason `align` (below) takes it as a parameter: a test can drive both branches
/// without touching process environment.
///
/// This function, not `align`, is the real production enforcement point for the
/// opt-out: the single production call site passes `disabled` straight through to
/// `align`, but every one of *this* function's own skip branches returns before
/// `align` is ever reached, so `align`'s production call always sees `disabled ==
/// false` here as well as there. `align`'s own `if disabled` branch exists only so
/// its unit tests can pin the state machine's half of the contract (the pin and
/// llama.cpp tag staying untouched) without duplicating this whole call chain.
#[allow(clippy::too_many_arguments)]
fn prepare_llamacpp_backend_for_active_rocm_impl(
    manifest: &mut LemonadeInstallManifest,
    disabled: bool,
    mut active_rocm_sdk_version: impl FnMut() -> Result<Option<String>>,
    mut pinned_version_lookup: impl FnMut(&Path) -> Option<String>,
    mut selected_backend_lookup: impl FnMut(&LemonadeInstallManifest) -> Result<Option<String>>,
    mut fallback_install: impl FnMut(&mut LemonadeInstallManifest, bool) -> Result<()>,
    mut align: impl FnMut(
        &mut LemonadeInstallManifest,
        &Path,
        &str,
        &str,
        bool,
    ) -> Result<Option<String>>,
) -> Result<Option<String>> {
    // Announced here, unconditionally, rather than only where `align` happens to be
    // reached: every one of the skip branches below -- not Linux, no active SDK
    // version, unreadable pin, pin already matches, the vulkan-fallback probe --
    // would otherwise leave a user who set the variable unable to tell it took
    // effect, exactly the silence this message exists to avoid.
    if disabled {
        eprintln!(
            "Lemonade backend alignment is disabled by {LEMONADE_BACKEND_ALIGNMENT_DISABLED_ENV}; \
             using whatever backend_versions.json already pins."
        );
        // Checked earliest, before even the cheap version/pin lookups below, let alone
        // the vulkan-gate probe (a real `lemond` spawn) -- none of that work is worth
        // doing when the opt-out means its result can't change what happens next.
        fallback_install(manifest, false)?;
        return Ok(None);
    }
    // Alignment is only ever verifiable on Linux ([`rocm_backend_resolves`] always
    // reports unresolved elsewhere), so attempting it on Windows can only burn up to
    // three multi-GB backend installs and a network round-trip for a guaranteed-futile
    // outcome. Skip straight to the ordinary pinned-version install.
    if !runtime_is_linux() {
        fallback_install(manifest, false)?;
        return Ok(None);
    }

    let target_version = active_rocm_sdk_version().unwrap_or_else(|error| {
        eprintln!(
            "Warning: could not determine the active ROCm SDK version to align Lemonade's \
             backend with: {error:#}"
        );
        None
    });
    let Some(target_version) = target_version else {
        fallback_install(manifest, false)?;
        return Ok(None);
    };

    let backend_versions_path = manifest.runtime_dir.join(BACKEND_VERSIONS_RESOURCE);
    let Some(pinned_version) = pinned_version_lookup(&backend_versions_path) else {
        fallback_install(manifest, false)?;
        return Ok(None);
    };
    if pinned_version == target_version {
        fallback_install(manifest, false)?;
        return Ok(None);
    }

    // Verification below requires the backend Lemonade selects to be `rocm`
    // specifically ([`try_llamacpp_backend_alignment`]'s `has_usable_binary` gate) --
    // it falls back to `vulkan` on hosts where Lemonade's ROCm build is unsupported
    // (WSL2 being the documented case), and no tier can ever pass there. Check which
    // backend this host actually gets *before* patching `backend_versions.json`, the
    // same guaranteed-futile case already short-circuited for Windows above: otherwise
    // every install on such a host burns two forced reinstalls plus a GitHub round-trip
    // discovering what this cheap query already knows.
    let selected_backend = match selected_backend_lookup(manifest) {
        Ok(selected_backend) => selected_backend,
        Err(error) => {
            eprintln!(
                "Warning: could not determine which llama.cpp backend this host would select; \
                 skipping ROCm backend alignment: {error:#}"
            );
            fallback_install(manifest, false)?;
            return Ok(None);
        }
    };
    if !should_attempt_llamacpp_backend_alignment(selected_backend.as_deref()) {
        fallback_install(manifest, false)?;
        return Ok(None);
    }

    align(
        manifest,
        &backend_versions_path,
        &target_version,
        &pinned_version,
        disabled,
    )
}

/// The Tier 1 / Tier 2 / revert state machine, isolated from the config and active-SDK
/// lookups above so it can be exercised in tests against a temp `backend_versions.json`
/// with the install/align/latest-tag steps injected, instead of spawning a real
/// `lemond` and reaching GitHub.
///
/// `disabled` is the resolved value of [`lemonade_backend_alignment_disabled`], passed
/// in rather than read here so a test can drive both branches without touching process
/// environment (mirrors vLLM's `torch_alignment_disabled` parameter). The real
/// production enforcement of the opt-out lives one level up, in
/// [`prepare_llamacpp_backend_for_active_rocm_impl`]: its own early return skips
/// every lookup this function depends on, so its single production call site always
/// passes `disabled == false` here. This function's own `if disabled` branch exists
/// only so its unit tests can pin this half of the contract (the pin and llama.cpp
/// tag staying untouched when the flag is set) without spinning up the whole call
/// chain above it. The user-facing announcement lives in
/// [`prepare_llamacpp_backend_for_active_rocm_impl`] as well, so it fires on every
/// skip path the opt-out affects, not only the one this function's own gate reaches.
#[allow(clippy::too_many_arguments)]
fn align_llamacpp_backend_to_version(
    manifest: &mut LemonadeInstallManifest,
    backend_versions_path: &Path,
    target_version: &str,
    pinned_version: &str,
    disabled: bool,
    mut align: impl FnMut(&mut LemonadeInstallManifest, &str, bool, &str) -> bool,
    mut fallback_install: impl FnMut(&mut LemonadeInstallManifest, bool) -> Result<()>,
    mut latest_tag: impl FnMut() -> Result<String>,
) -> Result<Option<String>> {
    if disabled {
        fallback_install(manifest, false)?;
        return Ok(None);
    }
    if let Err(error) =
        write_backend_versions_therock_version(backend_versions_path, target_version)
    {
        eprintln!("Warning: could not pin Lemonade's backend to ROCm {target_version}: {error:#}");
        fallback_install(manifest, false)?;
        return Ok(None);
    }

    // Tier 1: keep Lemonade's own pinned llama.cpp build, just point it at the active
    // ROCm version. Works when that specific build's release actually shipped a
    // matching ROCm-version asset. Always forces a reinstall: this whole function
    // only runs when the pin actually changed (the `pinned_version == target_version`
    // check above), so a backend already on disk was installed against the OLD
    // pin and must not be trusted just because it happens to still resolve.
    if align(manifest, target_version, true, "its pinned llama.cpp build") {
        return Ok(Some(target_version.to_owned()));
    }

    // Tier 2: the pinned build may simply be too old to ever have shipped a
    // matching ROCm-version asset — Lemonade only started publishing per-ROCm-
    // version builds partway through its release history, and the embeddable's
    // pin (frozen at whatever build was current when this Lemonade version was
    // cut) does not track that. Point at the current newest build instead, still
    // paired with the active ROCm version, forcing a real reinstall attempt
    // regardless of whatever Tier 1 left on disk.
    let pinned_tag = read_backend_versions_llamacpp_tag(backend_versions_path);
    let latest_tag_applied = match latest_tag() {
        Ok(latest_tag) if pinned_tag.as_deref() != Some(latest_tag.as_str()) => {
            match write_backend_versions_llamacpp_tag(backend_versions_path, &latest_tag) {
                Ok(()) => true,
                Err(error) => {
                    eprintln!(
                        "Warning: could not pin Lemonade's backend to llama.cpp {latest_tag}: \
                         {error:#}"
                    );
                    false
                }
            }
        }
        // The pinned tag already is the latest — Tier 1 already tried it.
        Ok(_) => false,
        Err(error) => {
            eprintln!("Warning: could not determine Lemonade's latest llama.cpp build: {error:#}");
            false
        }
    };
    if latest_tag_applied && align(manifest, target_version, true, "the latest llama.cpp build") {
        return Ok(Some(target_version.to_owned()));
    }

    // Neither tier produced a verified GPU backend. Revert everything — including
    // the llama.cpp build tag, so the fallback install below isn't itself
    // misdirected — and retry once more with Lemonade's original pinned defaults.
    // Every step below is best-effort, matching its sibling warnings: a failure to
    // restore the pin still lets the fallback install run rather than hard-failing
    // the whole install, though it does leave the pin holding the unverified
    // target version until a later run corrects it.
    eprintln!(
        "Warning: could not align Lemonade's ROCm backend to {target_version}; reverting to the \
         default pinned version."
    );
    if let Err(error) =
        write_backend_versions_therock_version(backend_versions_path, pinned_version)
    {
        eprintln!(
            "Warning: could not restore Lemonade's default pinned backend version \
             ({pinned_version}): {error:#}"
        );
    }
    if latest_tag_applied {
        // Best-effort, matching the therock.version restore above: a failure here
        // must not skip the fallback reinstall below. Restore the original tag when
        // there was one; otherwise Tier 2 pinned a key that did not exist before, so
        // removing it (not writing some placeholder) is what "reverted" means.
        let restore_result = match pinned_tag.as_deref() {
            Some(tag) => write_backend_versions_llamacpp_tag(backend_versions_path, tag),
            None => remove_backend_versions_llamacpp_tag(backend_versions_path),
        };
        if let Err(error) = restore_result {
            eprintln!(
                "Warning: could not restore Lemonade's default pinned llama.cpp build tag: \
                 {error:#}"
            );
        }
    }
    let aside_dirs = set_aside_rocm_llamacpp_backend_dirs(manifest);
    match fallback_install(manifest, true) {
        Ok(()) => {
            resolve_rocm_llamacpp_backend_dirs_aside(manifest, &aside_dirs, true);
            Ok(None)
        }
        Err(error) => {
            // Restore the pre-alignment backend before propagating: a failed
            // fallback install must not leave the host with no ROCm backend at all.
            resolve_rocm_llamacpp_backend_dirs_aside(manifest, &aside_dirs, false);
            Err(error)
        }
    }
}

/// Attempt one llama.cpp backend install aligned to `target_version`'s ROCm pairing
/// (already written into `resources/backend_versions.json`), then verify — never
/// simply trust Lemonade's own reported success — that the resulting GPU backend
/// actually resolves against rocm-cli's active ROCm SDK (AGENTS.md §6: no silent CPU
/// fallback). `force_reinstall` must be `true` for every caller here: alignment is
/// only ever attempted when the pin actually changed, and a skipped "already
/// installed" reinstall would leave a stale, pre-alignment binary that can still
/// pass the `ldd`-resolves check below, reporting success for a version that was
/// never actually installed. Returns whether a verified backend is now in place; on
/// success, `manifest.backend_name` is updated to match it.
fn try_llamacpp_backend_alignment(
    manifest: &mut LemonadeInstallManifest,
    target_version: &str,
    force_reinstall: bool,
    attempt_label: &str,
) -> bool {
    debug_assert!(
        force_reinstall,
        "every alignment attempt must force a reinstall; see the doc comment above"
    );
    let aside_dirs = set_aside_rocm_llamacpp_backend_dirs(manifest);
    let install_result = install_best_llamacpp_backend(manifest, force_reinstall);
    // `ensure_best_llamacpp_backend` records the backend it attempted into
    // `manifest.backend_name` before the fallible install step runs, so this is accurate
    // even when `install_result` is `Err` below. Scoping to it — rather than scanning
    // every backend directory — means a stale `rocm-*` directory left by an earlier
    // attempt can never be mistaken for this round's verification, including when this
    // round actually picked `vulkan`.
    let backend_name = manifest.backend_name.clone();
    let has_usable_binary = backend_name == ROCM_BACKEND_NAME
        && lemonade_process_environment().is_ok_and(|process_env| {
            find_llama_server_binary_for_backend(manifest, &backend_name)
                .is_some_and(|binary| rocm_backend_resolves(&binary, &process_env))
        });
    let succeeded = llamacpp_backend_alignment_succeeded(&install_result, has_usable_binary);
    resolve_rocm_llamacpp_backend_dirs_aside(manifest, &aside_dirs, succeeded);
    if !succeeded {
        match install_result {
            Ok(()) => eprintln!(
                "Warning: {attempt_label} installed for ROCm {target_version}, but its GPU \
                 backend does not resolve against rocm-cli's active ROCm SDK; not using it."
            ),
            Err(error) if has_usable_binary => eprintln!(
                "Warning: could not install {attempt_label} for ROCm {target_version} \
                 ({error:#}); a backend binary is present but may predate this alignment \
                 attempt, so it is not trusted without a completed install."
            ),
            Err(error) => eprintln!(
                "Warning: could not install {attempt_label} for ROCm {target_version}: {error:#}"
            ),
        }
        return false;
    }
    eprintln!("Aligned Lemonade's ROCm backend to {target_version} using {attempt_label}.");
    true
}

/// Whether an alignment attempt counts as a verified success: the install must
/// have actually completed, not merely have left behind some binary that
/// happens to resolve. `ldd`-based verification cannot tell a stale build
/// (paired with a different ROCm version, possibly left over from before this
/// attempt) from a version-matched one, so a failed install must never be
/// forgiven by a leftover binary passing that check.
const fn llamacpp_backend_alignment_succeeded(
    install_result: &Result<()>,
    has_usable_binary: bool,
) -> bool {
    install_result.is_ok() && has_usable_binary
}

/// GitHub repo backing Lemonade's `llamacpp:rocm-stable` builds. Bumping
/// `resources/backend_versions.json`'s `therock.version` alone doesn't get a
/// ROCm-version-matched build if the pinned tag predates that ROCm version's
/// support: Lemonade only started publishing per-ROCm-version assets partway
/// through this repo's release history, and the embeddable's own pin can be
/// (and, at the time of writing, is) older than that.
const LLAMACPP_ROCM_STABLE_REPO: &str = "lemonade-sdk/llama.cpp";

/// The newest published release tag for [`LLAMACPP_ROCM_STABLE_REPO`] (e.g.
/// `b10952`), queried directly rather than assumed — the whole reason Tier 2
/// exists is that the embeddable's own pinned tag cannot be trusted to have a
/// matching-ROCm-version asset.
fn latest_llamacpp_rocm_stable_tag() -> Result<String> {
    let url = format!("https://api.github.com/repos/{LLAMACPP_ROCM_STABLE_REPO}/releases/latest");
    let timeout = Duration::from_secs(15);
    // `timeout_connect` takes precedence over `timeout` and defaults to 30s, so without
    // it a host that blackholes rather than refuses would stall well past the intended
    // ceiling.
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(timeout)
        .timeout(timeout)
        .build();
    let response = agent
        .get(&url)
        .set("User-Agent", "rocm-cli")
        .call()
        .with_context(|| format!("failed to query {url}"))?;
    let body: Value = response
        .into_json()
        .with_context(|| format!("failed to parse response from {url}"))?;
    body.get("tag_name")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .with_context(|| format!("{url} response is missing 'tag_name'"))
}

/// Read `resources/backend_versions.json`'s pinned `llamacpp.rocm-stable` build tag.
fn read_backend_versions_llamacpp_tag(path: &Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    let value: Value = serde_json::from_slice(&bytes).ok()?;
    value
        .get("llamacpp")?
        .get("rocm-stable")?
        .as_str()
        .map(str::to_owned)
}

/// Overwrite `resources/backend_versions.json`'s `llamacpp.rocm-stable` build tag in
/// place, preserving every other key.
fn write_backend_versions_llamacpp_tag(path: &Path, tag: &str) -> Result<()> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    value
        .get_mut("llamacpp")
        .and_then(Value::as_object_mut)
        .with_context(|| format!("{} has no 'llamacpp' object to patch", path.display()))?
        .insert("rocm-stable".to_owned(), Value::String(tag.to_owned()));
    fs::write(path, serde_json::to_vec_pretty(&value)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

/// Remove `resources/backend_versions.json`'s `llamacpp.rocm-stable` key entirely,
/// restoring the "never pinned" state — the counterpart to
/// [`write_backend_versions_llamacpp_tag`] for reverting Tier 2's pin when there was
/// no prior tag to restore it to.
fn remove_backend_versions_llamacpp_tag(path: &Path) -> Result<()> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    value
        .get_mut("llamacpp")
        .and_then(Value::as_object_mut)
        .with_context(|| format!("{} has no 'llamacpp' object to patch", path.display()))?
        .remove("rocm-stable");
    fs::write(path, serde_json::to_vec_pretty(&value)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

/// The active rocm-cli-managed ROCm SDK version to align Lemonade's backend to,
/// restricted to plain `X.Y.Z` release versions. Nightly builds carry a date suffix
/// (e.g. `7.14.0a20260601`) that does not correspond to any llama.cpp build tag
/// Lemonade publishes, so those are left alone rather than guessed at.
///
/// Errors (rather than defaulting) when the config can't be loaded: an empty default
/// config would discard the active runtime key and let version selection guess the
/// most-recently-installed runtime, then mutate Lemonade based on that guess.
fn active_rocm_sdk_version_for_alignment(paths: &AppPaths) -> Result<Option<String>> {
    let config = RocmCliConfig::load(paths)?;
    let Some(version) = active_managed_therock_version(paths, &config)? else {
        return Ok(None);
    };
    Ok(looks_like_plain_semver(&version).then_some(version))
}

/// Whether `version` is a plain `major.minor.patch` string with no build/date suffix
/// (e.g. `10.0.0`, not `7.14.0a20260601`).
fn looks_like_plain_semver(version: &str) -> bool {
    let parts: Vec<&str> = version.split('.').collect();
    parts.len() == 3
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()))
}

/// Read `resources/backend_versions.json`'s `therock.version` — Lemonade's pin for
/// which private ROCm runtime its `llamacpp:rocm` backend downloads.
fn read_backend_versions_therock_version(path: &Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    let value: Value = serde_json::from_slice(&bytes).ok()?;
    value
        .get("therock")?
        .get("version")?
        .as_str()
        .map(str::to_owned)
}

/// Overwrite `resources/backend_versions.json`'s `therock.version` in place,
/// preserving every other key. Lemonade's own comment in that file invites exactly
/// this: "You can modify these values to pin specific versions without rebuilding the
/// application."
fn write_backend_versions_therock_version(path: &Path, version: &str) -> Result<()> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut value: Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    value
        .get_mut("therock")
        .and_then(Value::as_object_mut)
        .with_context(|| format!("{} has no 'therock' object to patch", path.display()))?
        .insert("version".to_owned(), Value::String(version.to_owned()));
    fs::write(path, serde_json::to_vec_pretty(&value)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

/// ROCm shared-library sonames the `llamacpp:rocm` backend's `libggml-hip.so` links
/// against. If any fail to resolve, the backend would silently fall back to CPU
/// inference at runtime instead of failing loudly — unacceptable under the
/// GPU-required policy (AGENTS.md §6), so this must be verified, not assumed from
/// version numbers matching.
///
/// Linux-only, like the `ldd`-based verification it exists for: the non-Linux
/// `rocm_backend_resolves` stub below never references it, so leaving it
/// ungated makes it dead code — and a build failure under `-D warnings` — on
/// every other target.
#[cfg(target_os = "linux")]
const ROCM_BACKEND_REQUIRED_SONAMES: [&str; 4] = [
    "libhipblas.so",
    "librocblas.so",
    "libamdhip64.so",
    "libhsa-runtime64.so",
];

/// Whether the ROCm GPU backend beside `llama_server_binary` actually resolves its
/// required ROCm libraries under `process_env` (rocm-cli's injected ROCm environment),
/// checked with `ldd` rather than assumed. `false` on any I/O/parse failure.
#[cfg(target_os = "linux")]
fn rocm_backend_resolves(
    llama_server_binary: &Path,
    process_env: &LemonadeProcessEnvironment,
) -> bool {
    let Some(backend_dir) = llama_server_binary.parent() else {
        return false;
    };
    let hip_backend = backend_dir.join("libggml-hip.so");
    if !hip_backend.is_file() {
        return false;
    }
    let mut command = ProcessCommand::new("ldd");
    command
        .arg(&hip_backend)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if apply_lemonade_process_environment(&mut command, process_env).is_err() {
        return false;
    }
    let Ok(output) = command.output() else {
        return false;
    };
    output.status.success()
        && ldd_output_resolves_all(
            &String::from_utf8_lossy(&output.stdout),
            &ROCM_BACKEND_REQUIRED_SONAMES,
        )
}

/// Off Linux, `ldd`-based verification is not exercised: a version-aligned backend is
/// never accepted without it, so this always reports unresolved.
#[cfg(not(target_os = "linux"))]
const fn rocm_backend_resolves(
    _llama_server_binary: &Path,
    _process_env: &LemonadeProcessEnvironment,
) -> bool {
    false
}

/// Whether every `required` soname prefix appears in `ldd` output with a resolved
/// path (i.e. not `=> not found`). Linux-only: see [`ROCM_BACKEND_REQUIRED_SONAMES`].
#[cfg(target_os = "linux")]
fn ldd_output_resolves_all(output: &str, required: &[&str]) -> bool {
    required.iter().all(|soname| {
        output.lines().any(|line| {
            let line = line.trim();
            line.starts_with(soname) && !line.contains("not found")
        })
    })
}

fn install_best_llamacpp_backend(
    manifest: &mut LemonadeInstallManifest,
    force_reinstall: bool,
) -> Result<()> {
    let port = free_local_port()?;
    let log_path_buf = manifest.runtime_dir.join("install-lemond.log");
    let log_path = Some(log_path_buf.as_path());
    let process_env = lemonade_process_environment()?;
    let mut child = spawn_lemond(manifest, DEFAULT_HOST, port, log_path, &process_env)?;
    let result = (|| -> Result<String> {
        wait_for_lemonade_cli_status(
            manifest,
            DEFAULT_HOST,
            port,
            Duration::from_secs(30),
            log_path,
            &process_env,
        )?;
        ensure_best_llamacpp_backend(manifest, DEFAULT_HOST, port, &process_env, force_reinstall)
    })();
    let _ = terminate_pid(child.id(), true);
    let _ = child.wait();
    manifest.backend_name = result?;
    Ok(())
}

/// Whether the vulkan-gate probe's result means alignment could possibly succeed --
/// only when the host would select Lemonade's `rocm` backend by itself. Any other
/// selection (including the probe itself failing to determine one) means every
/// alignment tier is guaranteed to fail ([`try_llamacpp_backend_alignment`]'s
/// `has_usable_binary` gate requires the `rocm` backend specifically), so callers
/// should skip straight to an ordinary install instead of patching
/// `backend_versions.json` and burning two forced reinstalls plus a GitHub round-trip
/// discovering what this decision already knows. A pure function so this gate is
/// unit-testable without spawning the real `lemond` process
/// [`best_llamacpp_backend_for_host`] needs.
fn should_attempt_llamacpp_backend_alignment(selected_backend: Option<&str>) -> bool {
    selected_backend == Some(ROCM_BACKEND_NAME)
}

/// Which llama.cpp backend Lemonade would select on this host
/// (`LLAMACPP_BACKEND_PRIORITY`), without installing anything -- mirrors
/// [`install_best_llamacpp_backend`]'s spawn/query steps but stops short of the
/// install, so the ROCm backend alignment dance can check whether it is even
/// reachable before patching `backend_versions.json` or forcing a single reinstall.
fn best_llamacpp_backend_for_host(manifest: &LemonadeInstallManifest) -> Result<Option<String>> {
    let port = free_local_port()?;
    let log_path_buf = manifest.runtime_dir.join("install-lemond.log");
    let log_path = Some(log_path_buf.as_path());
    let process_env = lemonade_process_environment()?;
    let mut child = spawn_lemond(manifest, DEFAULT_HOST, port, log_path, &process_env)?;
    let result = (|| -> Result<Option<String>> {
        wait_for_lemonade_cli_status(
            manifest,
            DEFAULT_HOST,
            port,
            Duration::from_secs(30),
            log_path,
            &process_env,
        )?;
        let listing = run_lemonade_backends_list(manifest, &process_env)?;
        let backends = parse_llamacpp_backend_statuses(&listing);
        Ok(select_best_llamacpp_backend(&backends).map(|(name, _)| name))
    })();
    let _ = terminate_pid(child.id(), true);
    let _ = child.wait();
    result
}

/// Ask Lemonade which llama.cpp backends it supports on this GPU, choose the best
/// one (`LLAMACPP_BACKEND_PRIORITY`), install it if necessary, and return its name.
/// Retry the backend install itself once, rather than the whole Lemonade runtime
/// preparation. The embeddable download already has bounded transport retries;
/// repeating that outer operation would redo deterministic failures and may
/// re-extract a healthy runtime. A backend subprocess can instead fail after a
/// completed download when its connection to lemond is interrupted, and a second
/// call can reuse the backend cache immediately without a delay.
pub(crate) fn install_llamacpp_backend_with_retry(
    mut install: impl FnMut() -> Result<()>,
) -> Result<()> {
    match install() {
        Ok(()) => Ok(()),
        Err(first_error) => {
            eprintln!("Lemonade backend installation failed; retrying once: {first_error:#}");
            install().with_context(|| {
                "Lemonade backend installation failed again; run `rocm engines install \
                 lemonade --reinstall` and then retry `rocm serve`"
            })
        }
    }
}

pub(crate) fn ensure_best_llamacpp_backend(
    manifest: &mut LemonadeInstallManifest,
    host: &str,
    port: u16,
    process_env: &LemonadeProcessEnvironment,
    force_reinstall: bool,
) -> Result<String> {
    let listing = run_lemonade_backends_list(manifest, process_env)?;
    let backends = parse_llamacpp_backend_statuses(&listing);
    let Some((backend, already_installed)) = select_best_llamacpp_backend(&backends) else {
        bail!(
            "Lemonade reports no supported GPU llama.cpp backend for this host (status: {}). \
             The GPU-required policy does not fall back to CPU; install a ROCm or Vulkan \
             backend (e.g. `rocm engines install lemonade`) or verify the GPU driver with \
             `rocm examine`.",
            describe_llamacpp_backends(&backends)
        );
    };
    // Record the attempted backend before the fallible install step below, so a caller
    // verifying the result can scope its check to this backend even when the install
    // itself errors out.
    manifest.backend_name = backend.clone();
    if already_installed && !force_reinstall {
        eprintln!("Using installed Lemonade {LLAMACPP_RECIPE}:{backend} backend.");
    } else {
        eprintln!("Installing Lemonade {LLAMACPP_RECIPE}:{backend} backend...");
        install_llamacpp_backend_with_retry(|| {
            run_lemonade_backend_install(manifest, host, port, &backend, process_env)
        })?;
    }
    Ok(backend)
}

/// Parse `lemonade backends` table output into `(backend, status)` pairs for the
/// `llamacpp` recipe only. Status is one of `installed`/`installable`/`unsupported`.
fn parse_llamacpp_backend_statuses(output: &str) -> Vec<(String, String)> {
    const RECIPES: [&str; 8] = [
        "flm",
        "kokoro",
        "llamacpp",
        "ryzenai-llm",
        "sd-cpp",
        "vllm",
        "whispercpp",
        "embeddings",
    ];
    const STATUSES: [&str; 3] = ["installed", "installable", "unsupported"];
    let mut current_recipe = "";
    let mut result = Vec::new();
    for line in output.lines() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some(&first) = tokens.first() else {
            continue;
        };
        if first == "Recipe" || first.starts_with("---") {
            continue;
        }
        // A row either starts with a recipe name (recipe + its first backend) or,
        // for grouped recipes, with a backend name continuing the previous recipe.
        let (backend, rest) = if RECIPES.contains(&first) {
            current_recipe = match RECIPES.iter().find(|r| **r == first) {
                Some(r) => r,
                None => current_recipe,
            };
            match tokens.get(1) {
                Some(backend) => (*backend, &tokens[2..]),
                None => continue,
            }
        } else {
            (first, &tokens[1..])
        };
        if current_recipe != LLAMACPP_RECIPE {
            continue;
        }
        if let Some(status) = rest.iter().find(|t| STATUSES.contains(t)) {
            result.push((backend.to_owned(), (*status).to_owned()));
        }
    }
    result
}

/// Choose the highest-priority backend Lemonade considers usable (installed or
/// installable). Returns `(backend, already_installed)`.
fn select_best_llamacpp_backend(backends: &[(String, String)]) -> Option<(String, bool)> {
    for candidate in LLAMACPP_BACKEND_PRIORITY {
        if let Some((name, status)) = backends
            .iter()
            .find(|(b, s)| b == candidate && (s == "installed" || s == "installable"))
        {
            return Some((name.clone(), status == "installed"));
        }
    }
    None
}

fn describe_llamacpp_backends(backends: &[(String, String)]) -> String {
    if backends.is_empty() {
        return "none reported".to_owned();
    }
    backends
        .iter()
        .map(|(b, s)| format!("{b}={s}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn run_lemonade_backends_list(
    manifest: &LemonadeInstallManifest,
    process_env: &LemonadeProcessEnvironment,
) -> Result<String> {
    let mut command = ProcessCommand::new(&manifest.lemonade);
    command
        .arg("backends")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    apply_lemonade_process_environment(&mut command, process_env)?;
    hide_child_console_window(&mut command);
    let output = command
        .output()
        .with_context(|| format!("failed to run {}", manifest.lemonade.display()))?;
    if !output.status.success() {
        bail!(
            "Lemonade backends query failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

pub(crate) fn resolve_runtime() -> Result<LemonadeRuntime> {
    let paths = AppPaths::discover()?;
    let manifest = read_manifest(&paths)?;
    if !manifest.lemond.is_file() {
        bail!(
            "Lemonade runtime is missing {}; run `rocm engines install lemonade`",
            manifest.lemond.display()
        );
    }
    Ok(LemonadeRuntime { manifest })
}

/// The ROCm-family llama.cpp backend directory names under `bin/llamacpp/`. Lemonade's
/// installer extracts each build into its own build-numbered subdirectory rather than
/// replacing one in place, so all three names can accumulate across attempts.
pub(crate) const ROCM_LLAMACPP_BACKEND_DIRS: [&str; 3] = ["rocm-stable", "rocm-nightly", "rocm"];

/// Suffix appended to a ROCm-family backend directory moved aside by
/// [`set_aside_rocm_llamacpp_backend_dirs`].
const BACKEND_DIR_ASIDE_SUFFIX: &str = ".pre-alignment";

/// Move every existing ROCm-family llama.cpp backend directory aside (rather than
/// deleting it outright) before a forced alignment reinstall, so a failed install can
/// be recovered from via [`resolve_rocm_llamacpp_backend_dirs_aside`] instead of
/// leaving the host with no ROCm backend at all. Lemonade's installer extracts each
/// build into its own build-numbered subdirectory rather than replacing one in place,
/// so a tier that installs a different `therock.version`/llama.cpp tag than a prior
/// attempt (or the pinned default) can otherwise leave two builds side by side --
/// after which [`crate::direct_llama::find_binary_in`]'s directory-order fallback, used by both alignment
/// verification and `rocm serve`, may resolve to whichever build a rejected tier
/// installed instead of the one just verified.
///
/// Returns the `(original, aside)` pairs actually moved; only those need resolving.
fn set_aside_rocm_llamacpp_backend_dirs(
    manifest: &LemonadeInstallManifest,
) -> Vec<(PathBuf, PathBuf)> {
    let llamacpp_dir = manifest.runtime_dir.join("bin").join("llamacpp");
    ROCM_LLAMACPP_BACKEND_DIRS
        .iter()
        .filter_map(|backend| {
            let original = llamacpp_dir.join(backend);
            if !original.exists() {
                return None;
            }
            let aside = llamacpp_dir.join(format!("{backend}{BACKEND_DIR_ASIDE_SUFFIX}"));
            // A leftover aside path from an earlier, interrupted attempt must not
            // block this rename.
            let _ = fs::remove_dir_all(&aside);
            match fs::rename(&original, &aside) {
                Ok(()) => Some((original, aside)),
                Err(error) => {
                    eprintln!(
                        "Warning: could not move {} aside before backend alignment: {error:#}",
                        original.display()
                    );
                    None
                }
            }
        })
        .collect()
}

/// Resolve the aside directories from [`set_aside_rocm_llamacpp_backend_dirs`]: on
/// success, discard them -- the fresh install replaced what they held. On failure,
/// clear every [`ROCM_LLAMACPP_BACKEND_DIRS`] name (not just the ones that were
/// actually asided) before moving each `aside` back into place: a failed install can
/// land its output under a *different* ROCm-family name than the one that was
/// asided (Lemonade extracts each release into its own build-numbered directory
/// rather than replacing one in place), and [`crate::direct_llama::find_binary_in`]'s priority-ordered
/// lookup would otherwise let that stray, unverified directory shadow the good
/// build this function just restored. Best-effort: a failure here just risks the
/// same stale-directory ambiguity the aside exists to prevent, not the reinstall
/// itself.
fn resolve_rocm_llamacpp_backend_dirs_aside(
    manifest: &LemonadeInstallManifest,
    aside_dirs: &[(PathBuf, PathBuf)],
    success: bool,
) {
    if success {
        for (_, aside) in aside_dirs {
            let _ = fs::remove_dir_all(aside);
        }
        return;
    }
    let llamacpp_dir = manifest.runtime_dir.join("bin").join("llamacpp");
    for backend in ROCM_LLAMACPP_BACKEND_DIRS {
        let _ = fs::remove_dir_all(llamacpp_dir.join(backend));
    }
    for (original, aside) in aside_dirs {
        let _ = fs::rename(aside, original);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::install::platform_binary_name;
    use anyhow::anyhow;
    use serde_json::json;
    use std::fs;

    /// A fresh scratch directory under the crate's `target/`. The base is
    /// `CARGO_MANIFEST_DIR`, a compile-time constant, so the path never derives from a
    /// runtime environment read.
    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("lemonade-fs-test-{tag}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn backend_install_succeeds_without_retry() {
        let mut attempts = 0;

        install_llamacpp_backend_with_retry(|| {
            attempts += 1;
            Ok(())
        })
        .unwrap();

        assert_eq!(attempts, 1);
    }

    #[test]
    fn backend_install_recovers_on_the_second_attempt() {
        let mut attempts = 0;

        install_llamacpp_backend_with_retry(|| {
            attempts += 1;
            if attempts == 1 {
                bail!("first backend connection was interrupted");
            }
            Ok(())
        })
        .unwrap();

        assert_eq!(attempts, 2);
    }

    #[test]
    fn backend_install_stops_after_one_retry_with_reinstall_guidance() {
        let mut attempts = 0;

        let error = install_llamacpp_backend_with_retry(|| {
            attempts += 1;
            if attempts == 1 {
                bail!("first backend connection was interrupted");
            }
            bail!("second backend connection was interrupted");
        })
        .unwrap_err();
        let rendered = format!("{error:#}");

        assert_eq!(attempts, 2);
        assert!(
            rendered.contains("second backend connection was interrupted"),
            "{rendered}"
        );
        assert!(
            !rendered.contains("first backend connection was interrupted"),
            "the terminal error must be the retry's failure: {rendered}"
        );
        assert!(
            rendered.contains("rocm engines install lemonade --reinstall"),
            "{rendered}"
        );
        assert!(rendered.contains("retry `rocm serve`"), "{rendered}");
    }

    fn test_manifest(runtime_dir: PathBuf) -> LemonadeInstallManifest {
        LemonadeInstallManifest {
            env_id: "test".to_owned(),
            version: rocm_deps::LEMONADE_VERSION.to_owned(),
            runtime_dir,
            lemond: PathBuf::from("lemond"),
            lemonade: PathBuf::from("lemonade"),
            backend_recipe: LLAMACPP_RECIPE.to_owned(),
            backend_name: ROCM_BACKEND_NAME.to_owned(),
            installed_at_unix_ms: 0,
        }
    }

    #[test]
    fn plain_semver_accepts_release_versions_only() {
        assert!(looks_like_plain_semver("10.0.0"));
        assert!(looks_like_plain_semver("7.13.0"));
        assert!(!looks_like_plain_semver("7.14.0a20260601"));
        assert!(!looks_like_plain_semver("10.0"));
        assert!(!looks_like_plain_semver("10.0.0.1"));
        assert!(!looks_like_plain_semver(""));
    }

    #[test]
    fn backend_versions_therock_version_round_trips() {
        let dir = scratch_dir("backend-versions-round-trip");
        let path = dir.join("backend_versions.json");
        fs::write(
            &path,
            serde_json::to_vec_pretty(&json!({
                "llamacpp": { "rocm-stable": "b9752" },
                "therock": { "version": "7.13.0", "architectures": ["gfx1151"] },
            }))
            .unwrap(),
        )
        .unwrap();

        assert_eq!(
            read_backend_versions_therock_version(&path),
            Some("7.13.0".to_owned())
        );

        write_backend_versions_therock_version(&path, "10.0.0").unwrap();
        assert_eq!(
            read_backend_versions_therock_version(&path),
            Some("10.0.0".to_owned())
        );

        // Every other key, including sibling keys under `therock`, is preserved.
        let value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(value["llamacpp"]["rocm-stable"], "b9752");
        assert_eq!(value["therock"]["architectures"][0], "gfx1151");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn backend_versions_therock_version_missing_object_is_an_error() {
        let dir = scratch_dir("backend-versions-missing-therock");
        let path = dir.join("backend_versions.json");
        fs::write(
            &path,
            serde_json::to_vec_pretty(&json!({"llamacpp": {}})).unwrap(),
        )
        .unwrap();

        assert_eq!(read_backend_versions_therock_version(&path), None);
        assert!(write_backend_versions_therock_version(&path, "10.0.0").is_err());

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn backend_versions_llamacpp_tag_round_trips() {
        let dir = scratch_dir("backend-versions-tag-round-trip");
        let path = dir.join("backend_versions.json");
        fs::write(
            &path,
            serde_json::to_vec_pretty(&json!({
                "llamacpp": { "rocm-stable": "b9752", "vulkan": "b9747" },
                "therock": { "version": "7.13.0" },
            }))
            .unwrap(),
        )
        .unwrap();

        assert_eq!(
            read_backend_versions_llamacpp_tag(&path),
            Some("b9752".to_owned())
        );

        write_backend_versions_llamacpp_tag(&path, "b10952").unwrap();
        assert_eq!(
            read_backend_versions_llamacpp_tag(&path),
            Some("b10952".to_owned())
        );

        // Sibling keys, including the unrelated `llamacpp.vulkan` tag and the
        // whole `therock` object, are preserved.
        let value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(value["llamacpp"]["vulkan"], "b9747");
        assert_eq!(value["therock"]["version"], "7.13.0");

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn backend_versions_llamacpp_tag_missing_object_is_an_error() {
        let dir = scratch_dir("backend-versions-missing-llamacpp");
        let path = dir.join("backend_versions.json");
        fs::write(
            &path,
            serde_json::to_vec_pretty(&json!({"therock": {}})).unwrap(),
        )
        .unwrap();

        assert_eq!(read_backend_versions_llamacpp_tag(&path), None);
        assert!(write_backend_versions_llamacpp_tag(&path, "b10952").is_err());

        fs::remove_dir_all(&dir).ok();
    }

    fn write_backend_versions_fixture(path: &Path, therock_version: &str, llamacpp_tag: &str) {
        fs::write(
            path,
            serde_json::to_vec_pretty(&json!({
                "llamacpp": { "rocm-stable": llamacpp_tag },
                "therock": { "version": therock_version },
            }))
            .unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn set_aside_then_discard_removes_every_rocm_family_dir_but_not_vulkan() {
        // The bug this guards: Tier 2 installing a different llama.cpp tag than a
        // prior attempt left two build directories side by side, and the
        // directory-order lookup in `find_binary_in` could resolve to whichever one
        // a rejected tier installed instead of the one alignment just verified.
        let dir = scratch_dir("aside-then-discard-rocm-backend-dirs");
        let runtime_dir = dir.join("runtime");
        let llamacpp = runtime_dir.join("bin").join("llamacpp");
        let server = platform_binary_name("llama-server");
        for backend in ["rocm-stable", "rocm-nightly", "rocm", "vulkan"] {
            let backend_dir = llamacpp.join(backend);
            fs::create_dir_all(&backend_dir).unwrap();
            fs::write(backend_dir.join(&server), b"x").unwrap();
        }
        let manifest = test_manifest(runtime_dir);

        let aside_dirs = set_aside_rocm_llamacpp_backend_dirs(&manifest);
        for backend in ["rocm-stable", "rocm-nightly", "rocm"] {
            assert!(
                !llamacpp.join(backend).exists(),
                "{backend} should have been moved aside"
            );
        }
        resolve_rocm_llamacpp_backend_dirs_aside(&manifest, &aside_dirs, true);

        for backend in ["rocm-stable", "rocm-nightly", "rocm"] {
            assert!(
                !llamacpp.join(backend).exists(),
                "{backend} should still be gone after a discard"
            );
            assert!(
                !llamacpp
                    .join(format!("{backend}{BACKEND_DIR_ASIDE_SUFFIX}"))
                    .exists(),
                "{backend}'s aside copy should have been discarded on success"
            );
        }
        assert!(
            llamacpp.join("vulkan").join(&server).is_file(),
            "vulkan is untouched by ROCm alignment and must survive"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resolve_rocm_llamacpp_backend_dirs_aside_restores_on_failure() {
        // The bug this guards: a failed final install (e.g. the revert branch's
        // fallback_install erroring out) must never leave the host with zero ROCm
        // backend directories -- the whole reason to set builds aside instead of
        // deleting them outright.
        let dir = scratch_dir("aside-restores-on-failure");
        let runtime_dir = dir.join("runtime");
        let llamacpp = runtime_dir.join("bin").join("llamacpp");
        let server = platform_binary_name("llama-server");
        let rocm_dir = llamacpp.join("rocm");
        fs::create_dir_all(&rocm_dir).unwrap();
        fs::write(rocm_dir.join(&server), b"original").unwrap();
        let manifest = test_manifest(runtime_dir);

        let aside_dirs = set_aside_rocm_llamacpp_backend_dirs(&manifest);
        assert!(
            !rocm_dir.exists(),
            "the original must be moved aside before the install is attempted"
        );
        // Simulate a failed install leaving nothing behind at the original path.
        resolve_rocm_llamacpp_backend_dirs_aside(&manifest, &aside_dirs, false);

        assert_eq!(
            fs::read(rocm_dir.join(&server)).unwrap(),
            b"original",
            "the pre-existing backend must be restored after a failed install"
        );
        assert!(
            !llamacpp
                .join(format!("rocm{BACKEND_DIR_ASIDE_SUFFIX}"))
                .exists(),
            "the aside copy must be moved back, not left behind"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resolve_rocm_llamacpp_backend_dirs_aside_removes_a_failed_installs_stray_residue() {
        // The bug this guards: a failed install can land its (broken) output under a
        // DIFFERENT ROCm-family name than the one that was asided -- Lemonade
        // extracts each release into its own build-numbered directory rather than
        // replacing one in place. Only cleaning up the asided name left that stray
        // directory on disk, where `find_binary_in`'s priority order
        // (rocm-stable > rocm-nightly > rocm) could shadow the just-restored good
        // build with it.
        let dir = scratch_dir("aside-removes-stray-residue-in-different-dir");
        let runtime_dir = dir.join("runtime");
        let llamacpp = runtime_dir.join("bin").join("llamacpp");
        let server = platform_binary_name("llama-server");
        let rocm_dir = llamacpp.join("rocm");
        fs::create_dir_all(&rocm_dir).unwrap();
        fs::write(rocm_dir.join(&server), b"original").unwrap();
        let manifest = test_manifest(runtime_dir);

        let aside_dirs = set_aside_rocm_llamacpp_backend_dirs(&manifest);
        // Simulate a failed install that landed its (unverified) output in
        // `rocm-stable` instead of `rocm`, the name that was actually asided.
        let stray_dir = llamacpp.join("rocm-stable");
        fs::create_dir_all(&stray_dir).unwrap();
        fs::write(stray_dir.join(&server), b"stray").unwrap();

        resolve_rocm_llamacpp_backend_dirs_aside(&manifest, &aside_dirs, false);

        assert!(
            !stray_dir.exists(),
            "the failed install's residue in a different backend dir must be removed"
        );
        assert_eq!(
            fs::read(rocm_dir.join(&server)).unwrap(),
            b"original",
            "the pre-existing backend must still be restored"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn llamacpp_backend_alignment_requires_both_a_successful_install_and_a_usable_binary() {
        // The bug this guards: a failed install (e.g. Tier 1's pinned build 404ing
        // for a ROCm version it predates) must never be forgiven by a leftover
        // binary from a PREVIOUS, different-version install that happens to still
        // pass the ldd-resolves check -- `ldd` cannot tell the two apart.
        assert!(llamacpp_backend_alignment_succeeded(&Ok(()), true));
        assert!(!llamacpp_backend_alignment_succeeded(&Ok(()), false));
        assert!(!llamacpp_backend_alignment_succeeded(
            &Err(anyhow!("boom")),
            true
        ));
        assert!(!llamacpp_backend_alignment_succeeded(
            &Err(anyhow!("boom")),
            false
        ));
    }

    #[test]
    fn should_attempt_llamacpp_backend_alignment_requires_the_rocm_backend_specifically() {
        // The bug this guards: alignment is guaranteed to fail on a host that falls
        // back to vulkan (WSL2 being the documented case), so it must be skipped for
        // any selection other than exactly `rocm` -- including the probe itself
        // failing to determine one at all.
        assert!(should_attempt_llamacpp_backend_alignment(Some(
            ROCM_BACKEND_NAME
        )));
        assert!(!should_attempt_llamacpp_backend_alignment(Some("vulkan")));
        assert!(!should_attempt_llamacpp_backend_alignment(None));
    }

    #[test]
    fn align_honors_the_disabled_flag_without_touching_the_pin_or_the_injected_steps() {
        // This pins `align`'s own half of the opt-out contract in isolation: the pin
        // and llama.cpp tag stay byte-identical and `align`/`latest_tag` are never
        // called. The real production gate -- whether this function is even reached
        // with `disabled == true` in the first place -- lives one level up, in
        // `prepare_llamacpp_backend_for_active_rocm_impl`; see
        // `prepare_disabled_early_return_skips_every_lookup_and_installs_unforced`
        // below for that half.
        let dir = scratch_dir("align-disabled");
        let path = dir.join("backend_versions.json");
        write_backend_versions_fixture(&path, "7.13.0", "b9752");
        let mut manifest = test_manifest(dir.clone());

        let result = align_llamacpp_backend_to_version(
            &mut manifest,
            &path,
            "10.0.0",
            "7.13.0",
            true,
            |_manifest, _target, _force_reinstall, _label| {
                panic!("disabled means no alignment attempt of any kind")
            },
            |_manifest, force_reinstall| {
                assert!(
                    !force_reinstall,
                    "the disabled path installs whatever is already pinned, not a fresh reinstall"
                );
                Ok(())
            },
            || panic!("disabled means tier 2's tag lookup must not run either"),
        );

        assert_eq!(result.unwrap(), None);
        assert_eq!(
            read_backend_versions_therock_version(&path),
            Some("7.13.0".to_owned()),
            "the packaged pin must survive untouched when alignment is disabled"
        );
        assert_eq!(
            read_backend_versions_llamacpp_tag(&path),
            Some("b9752".to_owned()),
            "the packaged llama.cpp tag must survive untouched when alignment is disabled"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn prepare_disabled_early_return_skips_every_lookup_and_installs_unforced() {
        // The real production gate for the opt-out: previously this was covered only
        // by a @nightly @requires-gpu e2e scenario, which every per-PR and
        // merge-queue lane skips -- so a regression that silently re-enabled
        // alignment here shipped green everywhere that actually gates a merge. Every
        // lookup the disabled path is supposed to skip panics if reached; the
        // unforced install is the falsifiable positive.
        let mut manifest = test_manifest(PathBuf::from("unused-runtime-dir"));

        let result = prepare_llamacpp_backend_for_active_rocm_impl(
            &mut manifest,
            true,
            || panic!("disabled means the active SDK version must never be probed"),
            |_path| panic!("disabled means the packaged pin must never be read"),
            |_manifest| panic!("disabled means the vulkan-fallback probe must never run"),
            |_manifest, force_reinstall| {
                assert!(
                    !force_reinstall,
                    "the disabled path installs whatever is already pinned, not a fresh reinstall"
                );
                Ok(())
            },
            |_manifest, _path, _target, _pinned, _disabled| {
                panic!("disabled means align must never be reached")
            },
        );

        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn prepare_probe_error_falls_back_to_an_unforced_install_without_reaching_align() {
        // The bug this guards: a host where the vulkan-fallback probe itself errors
        // (e.g. `lemond` failing to start) must degrade to an ordinary install
        // rather than propagating the error or attempting alignment blind -- the
        // same "skip, don't fail" contract every other probe/lookup failure in this
        // gate already gets.
        let mut manifest = test_manifest(PathBuf::from("unused-runtime-dir"));

        let result = prepare_llamacpp_backend_for_active_rocm_impl(
            &mut manifest,
            false,
            || Ok(Some("10.0.0".to_owned())),
            |_path| Some("7.13.0".to_owned()),
            |_manifest| Err(anyhow!("lemond failed to start")),
            |_manifest, force_reinstall| {
                assert!(
                    !force_reinstall,
                    "a probe error falls back to an unforced install, not a reinstall"
                );
                Ok(())
            },
            |_manifest, _path, _target, _pinned, _disabled| {
                panic!("a probe error must skip align entirely")
            },
        );

        assert_eq!(result.unwrap(), None);
    }

    #[test]
    fn align_tier1_success_keeps_target_version_pinned() {
        let dir = scratch_dir("align-tier1-success");
        let path = dir.join("backend_versions.json");
        write_backend_versions_fixture(&path, "7.13.0", "b9752");
        let mut manifest = test_manifest(dir.clone());

        let result = align_llamacpp_backend_to_version(
            &mut manifest,
            &path,
            "10.0.0",
            "7.13.0",
            false,
            |_manifest, _target, force_reinstall, _label| {
                // A stale backend installed against the OLD pin must not be able to
                // fake success just because it happens to still resolve -- Tier 1
                // must force a real reinstall attempt every time.
                assert!(force_reinstall, "tier 1 must force a reinstall");
                true
            },
            |_manifest, _force_reinstall| panic!("tier 1 succeeded; no fallback install"),
            || panic!("tier 1 succeeded; tier 2's tag lookup must not run"),
        );

        assert_eq!(result.unwrap(), Some("10.0.0".to_owned()));
        assert_eq!(
            read_backend_versions_therock_version(&path),
            Some("10.0.0".to_owned())
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn align_falls_through_to_tier2_and_succeeds() {
        let dir = scratch_dir("align-tier2-success");
        let path = dir.join("backend_versions.json");
        write_backend_versions_fixture(&path, "7.13.0", "b9752");
        let mut manifest = test_manifest(dir.clone());
        let mut align_calls = 0;

        let result = align_llamacpp_backend_to_version(
            &mut manifest,
            &path,
            "10.0.0",
            "7.13.0",
            false,
            |_manifest, _target, force_reinstall, _label| {
                // Both tiers force a reinstall now, so the mock can no longer use
                // force_reinstall itself to distinguish tier 1 from tier 2 -- use
                // call order instead. Tier 1 fails; Tier 2 succeeds.
                assert!(
                    force_reinstall,
                    "every alignment attempt forces a reinstall"
                );
                align_calls += 1;
                align_calls > 1
            },
            |_manifest, _force_reinstall| panic!("tier 2 succeeded; no fallback install"),
            || Ok("b10952".to_owned()),
        );

        assert_eq!(result.unwrap(), Some("10.0.0".to_owned()));
        assert_eq!(align_calls, 2, "both tiers were attempted");
        assert_eq!(
            read_backend_versions_therock_version(&path),
            Some("10.0.0".to_owned())
        );
        assert_eq!(
            read_backend_versions_llamacpp_tag(&path),
            Some("b10952".to_owned()),
            "tier 2 pins the newer llama.cpp tag"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn align_reverts_the_llamacpp_tag_when_tier2_pinned_a_newer_one_and_still_fails() {
        // Regression test: the tag restore used to be silently skipped whenever
        // deleted, and nothing caught it -- both tiers must fail here so the
        // revert path actually runs, and the tag must come back to its original
        // value rather than staying on Tier 2's newer pin.
        let dir = scratch_dir("align-revert-restores-tag");
        let path = dir.join("backend_versions.json");
        write_backend_versions_fixture(&path, "7.13.0", "b9752");
        let mut manifest = test_manifest(dir.clone());

        let result = align_llamacpp_backend_to_version(
            &mut manifest,
            &path,
            "10.0.0",
            "7.13.0",
            false,
            |_manifest, _target, _force_reinstall, _label| false,
            |_manifest, force_reinstall| {
                assert!(force_reinstall);
                Ok(())
            },
            || Ok("b10952".to_owned()),
        );

        assert_eq!(result.unwrap(), None);
        assert_eq!(
            read_backend_versions_therock_version(&path),
            Some("7.13.0".to_owned())
        );
        assert_eq!(
            read_backend_versions_llamacpp_tag(&path),
            Some("b9752".to_owned()),
            "the llama.cpp tag must be restored to its original value, not left on Tier 2's pin"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn align_removes_the_llamacpp_tag_on_revert_when_none_was_pinned_before() {
        // The resource file may have no `llamacpp.rocm-stable` key at all (an
        // embeddable that never shipped a pin). Tier 2 still writes one; reverting
        // must remove it again rather than leaving it in place with no original
        // value to restore it to.
        let dir = scratch_dir("align-revert-removes-tag");
        let path = dir.join("backend_versions.json");
        fs::write(
            &path,
            serde_json::to_vec_pretty(&json!({
                "llamacpp": {},
                "therock": { "version": "7.13.0" },
            }))
            .unwrap(),
        )
        .unwrap();
        let mut manifest = test_manifest(dir.clone());

        let result = align_llamacpp_backend_to_version(
            &mut manifest,
            &path,
            "10.0.0",
            "7.13.0",
            false,
            |_manifest, _target, _force_reinstall, _label| false,
            |_manifest, force_reinstall| {
                assert!(force_reinstall);
                Ok(())
            },
            || Ok("b10952".to_owned()),
        );

        assert_eq!(result.unwrap(), None);
        assert_eq!(
            read_backend_versions_llamacpp_tag(&path),
            None,
            "no tag was pinned before Tier 2; reverting must remove it, not invent a value"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn align_skips_tier2_reinstall_when_pinned_tag_is_already_latest() {
        let dir = scratch_dir("align-tier2-tag-unchanged");
        let path = dir.join("backend_versions.json");
        write_backend_versions_fixture(&path, "7.13.0", "b9752");
        let mut manifest = test_manifest(dir.clone());
        let mut align_calls = 0;

        let result = align_llamacpp_backend_to_version(
            &mut manifest,
            &path,
            "10.0.0",
            "7.13.0",
            false,
            |_manifest, _target, _force_reinstall, _label| {
                align_calls += 1;
                false
            },
            |_manifest, force_reinstall| {
                assert!(
                    force_reinstall,
                    "the revert fallback always forces a reinstall"
                );
                Ok(())
            },
            // Already the latest tag: Tier 2's own reinstall must not be attempted.
            || Ok("b9752".to_owned()),
        );

        assert_eq!(result.unwrap(), None);
        assert_eq!(align_calls, 1, "only tier 1 was attempted");
        assert_eq!(
            read_backend_versions_therock_version(&path),
            Some("7.13.0".to_owned()),
            "reverted to the original pinned version"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn align_reverts_and_still_runs_fallback_when_the_restore_write_fails() {
        // Regression test: a failure while restoring the original pinned version must
        // not abort the install — the fallback reinstall below it is the recovery
        // path, and every sibling warning in this function is best-effort.
        let dir = scratch_dir("align-revert-write-fails");
        let path = dir.join("backend_versions.json");
        write_backend_versions_fixture(&path, "7.13.0", "b9752");
        let mut manifest = test_manifest(dir.clone());
        let mut fallback_called = false;

        // Corrupt the file's `therock` object between tier attempts and the revert, so
        // the revert's own write fails.
        let result = align_llamacpp_backend_to_version(
            &mut manifest,
            &path,
            "10.0.0",
            "7.13.0",
            false,
            |_manifest, _target, _force_reinstall, _label| {
                fs::write(
                    &path,
                    serde_json::to_vec_pretty(&json!({"llamacpp": {}})).unwrap(),
                )
                .unwrap();
                false
            },
            |_manifest, force_reinstall| {
                fallback_called = true;
                assert!(force_reinstall);
                Ok(())
            },
            || bail!("network unavailable"),
        );

        assert_eq!(
            result.unwrap(),
            None,
            "never propagates the restore-write failure"
        );
        assert!(fallback_called, "the fallback reinstall still ran");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn align_revert_restores_the_backend_dir_when_the_fallback_install_fails() {
        // The bug this guards: the revert branch used to delete the ROCm backend
        // directories before the fallback install, with no recovery if that install
        // then failed -- leaving the host with no ROCm backend at all. Both tiers
        // fail here (forcing the revert), and the fallback install also fails, so
        // the pre-existing backend must come back rather than staying deleted.
        let dir = scratch_dir("align-revert-fallback-install-fails");
        let path = dir.join("backend_versions.json");
        write_backend_versions_fixture(&path, "7.13.0", "b9752");
        let mut manifest = test_manifest(dir.clone());
        let llamacpp = dir.join("bin").join("llamacpp");
        let server = platform_binary_name("llama-server");
        let rocm_dir = llamacpp.join("rocm");
        fs::create_dir_all(&rocm_dir).unwrap();
        fs::write(rocm_dir.join(&server), b"pre-existing").unwrap();

        let result = align_llamacpp_backend_to_version(
            &mut manifest,
            &path,
            "10.0.0",
            "7.13.0",
            false,
            |_manifest, _target, _force_reinstall, _label| false,
            |_manifest, force_reinstall| {
                assert!(force_reinstall);
                bail!("network unavailable")
            },
            || Ok("b10952".to_owned()),
        );

        assert!(
            result.is_err(),
            "the fallback install's failure must be propagated"
        );
        assert_eq!(
            fs::read(rocm_dir.join(&server)).unwrap(),
            b"pre-existing",
            "the pre-existing backend must be restored, not left deleted"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn ldd_output_resolves_all_requires_every_soname_resolved() {
        let resolved = "\
\tlibggml-base.so.0 => /opt/rocm/lib/libggml-base.so.0 (0x00007f0)\n\
\tlibhipblas.so.3 => /opt/rocm/lib/libhipblas.so.3 (0x00007f1)\n\
\tlibrocblas.so.5 => /opt/rocm/lib/librocblas.so.5 (0x00007f2)\n\
\tlibamdhip64.so.7 => /opt/rocm/lib/libamdhip64.so.7 (0x00007f3)\n\
\tlibhsa-runtime64.so.1 => /opt/rocm/lib/libhsa-runtime64.so.1 (0x00007f4)\n";
        assert!(ldd_output_resolves_all(
            resolved,
            &ROCM_BACKEND_REQUIRED_SONAMES
        ));

        let missing_one = "\
\tlibhipblas.so.3 => not found\n\
\tlibrocblas.so.5 => /opt/rocm/lib/librocblas.so.5 (0x00007f2)\n\
\tlibamdhip64.so.7 => /opt/rocm/lib/libamdhip64.so.7 (0x00007f3)\n\
\tlibhsa-runtime64.so.1 => /opt/rocm/lib/libhsa-runtime64.so.1 (0x00007f4)\n";
        assert!(!ldd_output_resolves_all(
            missing_one,
            &ROCM_BACKEND_REQUIRED_SONAMES
        ));
    }

    #[test]
    fn parses_llamacpp_backends_from_table() {
        let output = "\
Recipe              Backend     Status          Message/Version                               Action
----------------------------------------------------------------------------------------------------
kokoro              cpu         installable     Backend is supported but not installed.       lemonade backends install kokoro:cpu
                    metal       unsupported     Requires macOS                                -
llamacpp            cpu         installable     Backend is supported but not installed.       lemonade backends install llamacpp:cpu
                    metal       unsupported     Requires macOS                                -
                    rocm        unsupported     Unsupported GPU                               -
                    system      unsupported     Requires Linux                               -
                    vulkan      installable     Backend is supported but not installed.       lemonade backends install llamacpp:vulkan
vllm                rocm        unsupported     Requires Linux                               -
";
        let backends = parse_llamacpp_backend_statuses(output);
        assert_eq!(
            backends,
            vec![
                ("cpu".to_owned(), "installable".to_owned()),
                ("metal".to_owned(), "unsupported".to_owned()),
                ("rocm".to_owned(), "unsupported".to_owned()),
                ("system".to_owned(), "unsupported".to_owned()),
                ("vulkan".to_owned(), "installable".to_owned()),
            ]
        );
    }

    #[test]
    fn selects_vulkan_when_rocm_unsupported() {
        let backends = vec![
            ("cpu".to_owned(), "installable".to_owned()),
            ("rocm".to_owned(), "unsupported".to_owned()),
            ("vulkan".to_owned(), "installable".to_owned()),
        ];
        assert_eq!(
            select_best_llamacpp_backend(&backends),
            Some(("vulkan".to_owned(), false))
        );
    }

    #[test]
    fn prefers_installed_rocm_when_supported() {
        let backends = vec![
            ("rocm".to_owned(), "installed".to_owned()),
            ("vulkan".to_owned(), "installable".to_owned()),
        ];
        assert_eq!(
            select_best_llamacpp_backend(&backends),
            Some(("rocm".to_owned(), true))
        );
    }

    #[test]
    fn never_selects_cpu_when_no_gpu_backend() {
        // Under the GPU-required policy there is no silent CPU fallback: when only
        // a CPU backend is usable, selection returns nothing and the caller fails.
        let backends = vec![
            ("rocm".to_owned(), "unsupported".to_owned()),
            ("cpu".to_owned(), "installable".to_owned()),
        ];
        assert_eq!(select_best_llamacpp_backend(&backends), None);
    }

    #[test]
    fn selects_nothing_when_all_unsupported() {
        let backends = vec![("rocm".to_owned(), "unsupported".to_owned())];
        assert_eq!(select_best_llamacpp_backend(&backends), None);
    }
}
