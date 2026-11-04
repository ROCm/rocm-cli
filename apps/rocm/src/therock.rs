// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

/// Property-based tests for version ordering and update-source selection.
/// A child module so it can reach this module's private items, and the
/// `pub(super)` comparators in `version`, directly.
#[cfg(test)]
#[path = "therock_version_proptests.rs"]
mod version_proptests;

mod consent;
mod family;
mod fetch;
mod io;
mod probe;
mod process;
mod python;
mod resolve;
mod source;
#[cfg(test)]
mod test_support;
mod update;
mod version;

use self::consent::{
    SdkInstallApproval, active_default_runtime_relation, confirm_overwrite_existing_sdk,
    fresh_install_line, refuse_non_interactive_message, sdk_install_approval,
};
pub(crate) use self::consent::{
    SdkInstallApprovalSource, SdkInstallConsent, preapproved_install_line,
};
use self::family::{family_override_or_recovered_arch, resolve_family};
pub(crate) use self::fetch::truthy_env;
use self::fetch::{download_file, preflight_tarball_space};
pub(crate) use self::io::write_file_atomically;
use self::probe::probe_rocm_sdk_runtime_for_target;
pub(crate) use self::probe::{
    RuntimeDeviceProbe, TorchAlignmentProbe, install_pinned_package, probe_rocm_sdk_runtime,
    probe_runtime_devices, probe_torch_alignment, requirement_pinned_version, split_local_version,
};
use self::process::{extract_tarball_and_discard_archive, progress_line, run_uv_progress_command};
use self::python::{
    ensure_uv_venv, managed_python_version, python_requirement, resolve_python_launcher,
    wheel_compatibility_for_python,
};
use self::resolve::{
    AggregateDeviceTarget, canonical_auto_select_needs_next_python, resolve_pip_runtime,
    resolve_tarball_artifact, unverified_torchaudio_note, wheel_runtime_composition,
};
pub(crate) use self::source::RuntimeVersionSelector;
#[cfg(test)]
use self::source::resolve_source;
use self::source::{
    SourceLayout, THEROCK_NEXT_MIN_MAJOR, TheRockChannel, next_layout_requested,
    requested_source_layout, select_source_layout,
};
pub(crate) use self::update::{
    maybe_refresh_startup_update_check, render_update_json, render_update_report,
    runtime_update_plan,
};
pub(crate) use self::version::runtime_version_display;
use self::version::{
    parse_host_version, parse_version_for_ordering, runtime_version_build_date, version_relation,
};
#[cfg(test)]
pub(crate) use crate::runtime_registry::RocmSdkLibraryProbe;
#[cfg(test)]
use crate::runtime_registry::runtime_registry_dir;
pub(crate) use crate::runtime_registry::{
    InstalledRuntimeManifest, RocmSdkPythonProbe, WheelRuntimeComposition, load_runtime_manifests,
    validate_rocm_sdk_runtime_probe,
};
use crate::runtime_registry::{
    runtime_manifest_path, save_runtime_manifest, wheel_composition_includes_devel,
};
use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, detect_host_gpu_diagnostics, detect_legacy_rocm_summary, ensure_uv_binary,
    interactive_terminal, runtime_is_windows, runtime_python_executable_in_env, unix_time_millis,
    uv_pip_install_base, uv_venv_args,
};
#[cfg(test)]
use rocm_core::{disk_space, managed_uv_cache_dir, normalize_therock_family};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

fn render_canonical_provenance(
    output: &mut String,
    channel: TheRockChannel,
    source_url: &str,
    layout_generation: &str,
    version: &str,
) {
    let build_date = runtime_version_build_date(version)
        .unwrap_or_else(|| "not encoded in stable version".to_owned());
    let _ = writeln!(output, "  channel: {}", channel.as_str());
    let _ = writeln!(output, "  canonical_source: {source_url}");
    let _ = writeln!(output, "  selected_rocm_version: {version}");
    let _ = writeln!(output, "  build_date: {build_date}");
    let _ = writeln!(output, "  source_layout_generation: {layout_generation}");
}

/// Outcome of an `install sdk` request.
///
/// `mutated` is `false` for dry-run plans and for installs the user declined at
/// the confirmation prompt, so the caller can skip the post-install activation
/// and success reporting that only make sense after a real install.
#[derive(Debug)]
pub(crate) struct SdkInstallResult {
    pub output: String,
    pub mutated: bool,
}

impl SdkInstallResult {
    const fn plan(output: String) -> Self {
        Self {
            output,
            mutated: false,
        }
    }

    const fn installed(output: String) -> Self {
        Self {
            output,
            mutated: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct InstallSourceOverride<'a> {
    family: Option<&'a str>,
    device_target: Option<&'a str>,
    layout: Option<SourceLayout>,
}

/// What to install, as resolved from the `rocm install sdk` arguments.
///
/// A struct rather than a parameter list: with `include_devel` added this is
/// past the point where positional `bool`s at a call site say anything about
/// what they mean.
#[derive(Debug)]
pub(crate) struct SdkInstallRequest<'a> {
    pub channel: &'a str,
    pub format: &'a str,
    pub prefix: Option<PathBuf>,
    pub version_selector: Option<RuntimeVersionSelector>,
    pub family_override: Option<&'a str>,
    pub dry_run: bool,
    /// Install the compiler and headers alongside the runtime libraries.
    pub include_devel: bool,
    /// The *source* of any up-front approval rather than a bare bool, because
    /// the progress line names it: `--yes` and
    /// `--approve-replacing-active-default` both clear this gate, but only the
    /// former also approves a `sudo` system-package install, so collapsing them
    /// would print "Approved by --yes" on every install ROCm CLI's own
    /// terminal-less surfaces make.
    pub consent: SdkInstallConsent,
}

impl Default for SdkInstallRequest<'_> {
    fn default() -> Self {
        Self {
            channel: "",
            format: "",
            prefix: None,
            version_selector: None,
            family_override: None,
            dry_run: false,
            include_devel: false,
            // Withholding consent is the safe default: a caller that leaves it
            // unset has granted nothing, so the gate prompts or refuses rather
            // than silently displacing the active default runtime.
            consent: SdkInstallConsent::Ask,
        }
    }
}

/// Install a TheRock SDK runtime.
pub(crate) fn install_sdk(
    paths: &AppPaths,
    request: SdkInstallRequest<'_>,
) -> Result<SdkInstallResult> {
    let SdkInstallRequest {
        channel,
        format,
        prefix,
        version_selector,
        family_override,
        dry_run,
        include_devel,
        consent,
    } = request;
    let channel = TheRockChannel::parse(channel)?;
    ensure_install_format_supported(format)?;
    match format {
        "wheel" => install_wheel_runtime(
            paths,
            channel,
            prefix,
            InstallSourceOverride {
                family: family_override,
                ..InstallSourceOverride::default()
            },
            version_selector.as_ref(),
            dry_run,
            include_devel,
            consent,
        ),
        "tarball" => {
            // A tarball catalog lists whole archives, not a resolvable package
            // graph, so there is nothing to pin a version against — except the
            // ROCm 10 catalog, which is reachable *only* by naming a version,
            // because nothing else distinguishes it from the canonical one.
            if let Some(selector) = version_selector.as_ref()
                && !next_layout_requested(selector)
            {
                bail!(
                    "specific TheRock version selection for tarball installs is only supported for a stable ROCm {THEROCK_NEXT_MIN_MAJOR}+ pin (e.g. `--version 10.0.0`); for any other version, use `--format wheel`"
                )
            }
            install_tarball_runtime(
                paths,
                channel,
                prefix,
                family_override,
                version_selector.as_ref(),
                None,
                dry_run,
                consent,
            )
        }
        other => bail!("unsupported install format: {other}"),
    }
}

/// Apply an update using the exact family, device payload, and source layout
/// resolved by its plan.
///
/// Consent is preapproved rather than asked for: the update targets the runtime
/// the user selected (or the active default), so installing over it is the
/// operation requested, and `rocm update` has no terminal contract — reaching a
/// prompt here would only fail the command. `rocm update` does accept a `--yes`
/// flag for consistency with other mutating commands, but it is inert and never
/// reaches this function, so it grants nothing here.
///
/// `activate_after_install` mirrors `rocm update --apply --activate` so the
/// approval line can state what will actually happen. Without it,
/// `apply_runtime_update` installs beside the active default and leaves the
/// default untouched, so the line must not claim an activation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn install_sdk_for_update(
    paths: &AppPaths,
    channel: &str,
    format: &str,
    family: &str,
    device_target: Option<&str>,
    source_layout_generation: Option<&str>,
    include_devel: bool,
    dry_run: bool,
    activate_after_install: bool,
) -> Result<SdkInstallResult> {
    let channel = TheRockChannel::parse(channel)?;
    ensure_install_format_supported(format)?;
    let consent = SdkInstallConsent::Preapproved(SdkInstallApprovalSource::UpdateApply {
        activates: activate_after_install,
    });
    let layout = Some(SourceLayout::from_generation(source_layout_generation)?);
    match format {
        "wheel" => install_wheel_runtime(
            paths,
            channel,
            None,
            InstallSourceOverride {
                family: Some(family),
                device_target,
                layout,
            },
            None,
            dry_run,
            include_devel,
            consent,
        ),
        "tarball" => install_tarball_runtime(
            paths,
            channel,
            None,
            Some(family),
            None,
            layout,
            dry_run,
            consent,
        ),
        other => bail!("unsupported install format: {other}"),
    }
}

fn ensure_install_format_supported(format: &str) -> Result<()> {
    ensure_install_format_supported_for_platform(format, runtime_is_windows())
}

fn ensure_install_format_supported_for_platform(format: &str, windows: bool) -> Result<()> {
    if windows && format == "tarball" {
        bail!(
            "TheRock tarball installs are not supported on Windows; use `rocm install sdk --format wheel` for a managed wheel virtual environment"
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn install_wheel_runtime(
    paths: &AppPaths,
    channel: TheRockChannel,
    prefix: Option<PathBuf>,
    source_override: InstallSourceOverride<'_>,
    version_selector: Option<&RuntimeVersionSelector>,
    dry_run: bool,
    include_devel: bool,
    consent: SdkInstallConsent,
) -> Result<SdkInstallResult> {
    let InstallSourceOverride {
        family: family_override,
        device_target: device_target_override,
        layout: layout_override,
    } = source_override;
    // `device_target_override` (an update apply's exact recorded arch) is only
    // otherwise consulted after `resolve_pip_runtime` returns — too late for the
    // channel/arch check below, which needs an exact arch before it can run at
    // all. Recover it into the family override up front, same as `resolve_pip_runtime`
    // does internally.
    let recovered_family_override = family_override
        .map(|family| family_override_or_recovered_arch(family, device_target_override));
    // Fail fast on a request `resolve_pip_runtime` could only ever reject (wrong
    // channel, or a grouped family with no exact arch) before paying for Python
    // provisioning below, which can mean a real download for a layout this host
    // or channel can never actually use.
    select_source_layout(
        channel,
        &resolve_family(paths, recovered_family_override.as_deref())?,
        version_selector,
    )?;
    // The interpreter has to be picked before `resolve_pip_runtime`, which
    // selects wheels using that interpreter's own tags. `requested_source_layout`
    // only answers from the *request*: an explicit stable ROCm 10+ pin already
    // routes to `Next` and gets `cp314` that way, but a plain auto-select or a
    // nightly build-date/prerelease pin always reports `Canonical`, even once
    // the canonical/nightly feed's own "latest" has rotated past that major on
    // its own, with no `Next` routing involved. `canonical_auto_select_needs_next_python`
    // peeks the channel's own `rocm` listing to catch that case.
    let python_requirement = match requested_source_layout(layout_override, version_selector) {
        SourceLayout::Next => python_requirement(SourceLayout::Next),
        SourceLayout::Canonical => {
            if canonical_auto_select_needs_next_python(paths, channel, version_selector)? {
                python_requirement(SourceLayout::Next)
            } else {
                python_requirement(SourceLayout::Canonical)
            }
        }
    };
    progress_line(format!(
        "Checking Python for the ROCm install; if needed, ROCm CLI will prepare Python {}.",
        managed_python_version(python_requirement)
    ));
    let python_launcher = resolve_python_launcher(paths, python_requirement)?;
    progress_line(match python_launcher.source {
        "path" => format!(
            "Using Python from PATH: {}.",
            python_launcher.executable.display()
        ),
        "env" => format!(
            "Using Python from ROCM_CLI_PYTHON: {}.",
            python_launcher.executable.display()
        ),
        "managed" => format!(
            "Using ROCm CLI's portable Python: {}.",
            python_launcher.executable.display()
        ),
        _ => format!(
            "Using Python from {}.",
            python_launcher.executable.display()
        ),
    });
    let wheel_compatibility = wheel_compatibility_for_python(&python_launcher.executable)?;
    progress_line(format!(
        "Checking TheRock {} packages for this AMD GPU...",
        channel.as_str()
    ));
    let resolution = resolve_pip_runtime(
        paths,
        channel,
        recovered_family_override.as_deref(),
        &wheel_compatibility,
        version_selector,
        layout_override,
        include_devel,
    )?;
    let device_target = device_target_override.map_or_else(
        || resolution.device_target.clone(),
        |target| {
            AggregateDeviceTarget::resolve(
                Some(target),
                &resolution.family,
                &resolution.published_device_targets,
            )
        },
    );
    // The exact device payload, not the version alone, decides what this runtime
    // can run, so it is what identifies the runtime. A preview on a host with no
    // usable target still composes a key here — from the `<undetermined>` extras
    // — which no real install can ever produce, and the refusal below stops it
    // from reaching a manifest.
    let wheel_composition = wheel_runtime_composition(&resolution, &device_target, include_devel);
    progress_line(format!(
        "Found canonical TheRock aggregate version {} with a matching PyTorch stack for target family {}.",
        resolution.latest_version, resolution.family
    ));
    let runtime_key = wheel_runtime_key(channel, &resolution.latest_version, &wheel_composition);
    let install_root = resolved_install_root(paths, "wheel", &runtime_key, prefix);
    let manifest_path = runtime_manifest_path(paths, &runtime_key);

    let mut output = String::new();
    let _ = writeln!(output, "sdk install");
    let _ = writeln!(
        output,
        "  summary: rocm-cli will install the ROCm SDK and matching PyTorch packages for this Python and operating system"
    );
    render_canonical_provenance(
        &mut output,
        channel,
        &resolution.index_url,
        resolution.layout.generation(),
        &resolution.latest_version,
    );
    let _ = writeln!(output, "  format: wheel");
    if let Some(selector) = version_selector {
        let _ = writeln!(output, "  requested: {}", selector.describe());
    }
    let _ = writeln!(output, "  target_family: {}", resolution.family);
    let _ = writeln!(
        output,
        "  target_family_source: {}",
        resolution.family_source
    );
    let _ = writeln!(output, "  device_target: {}", device_target.as_str());
    if let Some(reason) = device_target.reason() {
        let _ = writeln!(output, "  device_target_reason: {reason}");
    }
    let _ = writeln!(output, "  index_url: {}", resolution.index_url);
    let _ = writeln!(
        output,
        "  latest_compatible_version: {}",
        runtime_version_display(&resolution.latest_version)
    );
    // Probed once and reused by the progress line below: each call is a full
    // filesystem scan for a legacy ROCm, and the summary block and the visible
    // note report the same answer about the same resolved version.
    let host_version_newer = host_rocm_version_newer_than(&resolution.latest_version);
    if let Some(host_version) = host_version_newer.as_deref() {
        let _ = writeln!(
            output,
            "  version_note: {}",
            wheel_host_version_note(
                host_version,
                &runtime_version_display(&resolution.latest_version)
            )
        );
    }
    let _ = writeln!(
        output,
        "  compatibility_key: {}",
        runtime_version_display(&resolution.package_versions.compatibility_key)
    );
    let _ = writeln!(output, "  target: {}", install_root.display());
    let _ = writeln!(output, "  runtime_key: {runtime_key}");
    let _ = writeln!(
        output,
        "  python_launcher: {}",
        python_launcher.executable.display()
    );
    let _ = writeln!(output, "  python_source: {}", python_launcher.source);
    let _ = writeln!(
        output,
        "  python_wheel_tag: {}",
        wheel_compatibility.python_tag
    );
    let _ = writeln!(
        output,
        "  platform_wheel_tags: {}",
        wheel_compatibility.platform_tags.join(",")
    );
    // What resolution asked for, printed beside what the install will ask for.
    // The two are produced by different code paths from the same `include_devel`
    // and must name the same extras; showing only the second would hide a
    // resolve that constrained the version choice by a toolchain the user
    // declined. Absent on the canonical layout, which resolves by scraping each
    // package's index instead of compiling a requirement set.
    if let Some(specs) = resolution.version_resolution_specs.as_ref() {
        let _ = writeln!(output, "  version_resolution_specs: {}", specs.join(" "));
    }
    // Printed from the same value that chose the pin, so the line cannot claim
    // a verified build the requirements above do not name.
    if let Some(pin) = resolution.torchaudio_pin.as_ref() {
        let _ = writeln!(output, "  torchaudio_selection: {}", pin.describe());
    }
    let _ = writeln!(
        output,
        "  package_specs: {}",
        wheel_composition.package_specs.join(" ")
    );
    let _ = writeln!(
        output,
        "  package_policy: resolve the pinned target-complete rocm, torch, torchvision, and torchaudio plan from published package metadata, then install it in one uv transaction"
    );
    let no_wheel_warning = repo_version_without_wheels(
        resolution.newest_repo_version.as_deref(),
        &resolution.latest_version,
    )
    .map(|newest| {
        no_wheel_warning_message(
            &newest,
            &runtime_version_display(&resolution.latest_version),
        )
    });
    if let Some(warning) = no_wheel_warning.as_deref() {
        let _ = writeln!(output, "  warning: {warning}");
    }
    if dry_run {
        let env_python = venv_python_path(&install_root);
        let mut install_args = uv_pip_install_base(&env_python);
        install_args.extend(["--index-url".to_owned(), resolution.index_url.clone()]);
        if matches!(channel, TheRockChannel::Nightly) {
            install_args.extend(["--prerelease".to_owned(), "allow".to_owned()]);
        }
        install_args.extend(wheel_composition.package_specs.iter().cloned());
        let venv_args = uv_venv_args(&python_launcher.executable, &install_root);
        let venv_args_display = venv_args
            .iter()
            .map(|arg| quote_display_arg(arg))
            .collect::<Vec<_>>()
            .join(" ");
        let install_args_display = install_args
            .iter()
            .map(|arg| quote_display_arg(arg))
            .collect::<Vec<_>>()
            .join(" ");
        let _ = writeln!(output, "  mode: dry-run");
        let _ = writeln!(
            output,
            "  command: uv {venv_args_display} && uv {install_args_display}"
        );
        let _ = writeln!(
            output,
            "  activation: use the managed venv Python; TheRock libraries are resolved from that venv by rocm_sdk.initialize_process"
        );
        let _ = writeln!(output, "  manifest: {}", manifest_path.display());
        return Ok(SdkInstallResult::plan(output));
    }

    // Requirement: surface the "newest version has no wheels" case as a warning
    // on the real install path too, not only in the dry-run plan.
    if let Some(warning) = no_wheel_warning.as_deref() {
        progress_line(format!("Warning: {warning}"));
    }
    if let Some(note) = resolution
        .torchaudio_pin
        .as_ref()
        .and_then(unverified_torchaudio_note)
    {
        progress_line(note);
    }

    // Explain, in the visible install log, why an older TheRock ROCm is chosen
    // when this host reports a newer legacy ROCm (the ticket's 7.14 case). The
    // same note is recorded in the summary block above; surfacing it here keeps
    // it from being buried at the end of a long key/value dump.
    if let Some(host_version) = host_version_newer.as_deref() {
        progress_line(format!(
            "Note: this host reports ROCm {host_version}, but ROCm {resolved} is the newest TheRock ROCm with a matching PyTorch stack for this repository; installing {resolved} (pass `--version <VERSION>` to override).",
            resolved = runtime_version_display(&resolution.latest_version)
        ));
    }

    // Past the preview, the plan has to be installable. A runtime composed
    // without its exact device payload loads and then faults on the first
    // kernel, so an undetermined target is refused here rather than papered
    // over with every published payload.
    //
    // Checked *before* the consent gate: an install that cannot work should say
    // so, not first demand a consent flag for it. With the order reversed, a
    // non-interactive host with an unresolvable target reports only "re-run with
    // --approve-replacing-active-default", and supplying it just surfaces this
    // error instead.
    if let Some(reason) = device_target.reason() {
        bail!(
            "cannot compose a canonical TheRock {} runtime: {reason}.\n\
             The aggregate `rocm` distribution ships no GPU backend unless an exact `device-<target>` extra requests one, so this install would produce a runtime that cannot run a kernel.\n\
             Re-run `rocm install sdk` on the target host, or preview the plan with `--dry-run`.\n\n{}",
            channel.as_str(),
            detect_host_gpu_diagnostics()
        );
    }

    // Installs with no active default runtime proceed with just an informational
    // line. Only an install that would displace the current active default asks
    // for confirmation, and it asks regardless of family or channel because
    // activation is global. With no terminal to answer the prompt it refuses
    // instead, naming `--approve-replacing-active-default` as the
    // non-interactive approval — see `refuse_non_interactive_message` for why
    // that flag and not `--yes`.
    let existing = active_default_runtime_relation(
        paths,
        channel,
        &resolution.family,
        &resolution.latest_version,
    )?;
    match sdk_install_approval(existing.is_some(), consent, interactive_terminal()) {
        SdkInstallApproval::ProceedFresh => {
            progress_line(fresh_install_line(
                &runtime_version_display(&resolution.latest_version),
                &resolution.family,
            ));
        }
        SdkInstallApproval::ProceedApproved(source) => {
            progress_line(preapproved_install_line(
                source,
                existing.as_deref().unwrap_or_default(),
                &runtime_version_display(&resolution.latest_version),
            ));
        }
        SdkInstallApproval::PromptOverwrite => {
            if !confirm_overwrite_existing_sdk(
                channel,
                &resolution.family,
                &resolution.latest_version,
                existing.as_deref().unwrap_or_default(),
            )? {
                let _ = writeln!(
                    output,
                    "  status: cancelled by user; the existing ROCm SDK was left unchanged"
                );
                return Ok(SdkInstallResult::plan(output));
            }
        }
        SdkInstallApproval::RefuseNonInteractive => {
            bail!(refuse_non_interactive_message(
                existing.as_deref().unwrap_or_default()
            ));
        }
    }

    let uv = ensure_uv_binary(paths)?;
    fs::create_dir_all(
        install_root
            .parent()
            .context("runtime install root has no parent directory")?,
    )?;
    progress_line(format!(
        "Creating Python environment at {}.",
        install_root.display()
    ));
    ensure_uv_venv(paths, &uv, &python_launcher.executable, &install_root)?;
    let env_python = venv_python_path(&install_root);

    progress_line(format!(
        "Installing {} from {}",
        wheel_composition.package_specs.join(" "),
        resolution.index_url
    ));
    let mut install_args = uv_pip_install_base(&env_python);
    install_args.extend(["--index-url".to_owned(), resolution.index_url.clone()]);
    if matches!(channel, TheRockChannel::Nightly) {
        install_args.extend(["--prerelease".to_owned(), "allow".to_owned()]);
    }
    install_args.extend(wheel_composition.package_specs.iter().cloned());
    run_uv_progress_command(
        paths,
        &uv,
        install_args
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice(),
        // Read back out of the specs `uv` is being handed on this very call
        // rather than off `include_devel` a second time, so the line a user
        // watches cannot name a toolchain the install is not requesting.
        if wheel_composition_includes_devel(Some(&wheel_composition)).unwrap_or(include_devel) {
            "install TheRock SDK with the compiler toolchain, torch stack, and resolved dependencies"
        } else {
            "install TheRock SDK, torch stack, and resolved dependencies"
        },
    )?;

    progress_line("Checking the installed ROCm SDK...");
    let rocm_sdk_probe =
        probe_rocm_sdk_runtime_for_target(&env_python, Some(device_target.as_str()))
            .context("TheRock packages did not expose a usable rocm_sdk runtime")?;
    validate_rocm_sdk_runtime_probe(&rocm_sdk_probe)?;
    let installed_version = rocm_sdk_probe
        .rocm_sdk_version
        .clone()
        .unwrap_or_else(|| resolution.latest_version.clone());
    let manifest = InstalledRuntimeManifest {
        runtime_key: runtime_key.clone(),
        runtime_id: format!("therock-{}:{}", channel.as_str(), resolution.family),
        channel: channel.as_str().to_owned(),
        format: "wheel".to_owned(),
        family: resolution.family.clone(),
        family_source: resolution.family_source.clone(),
        version: installed_version.clone(),
        install_root: install_root.clone(),
        selected_artifact_url: resolution.index_url.clone(),
        source_layout_generation: Some(resolution.layout.generation().to_owned()),
        index_url: Some(resolution.index_url.clone()),
        tarball_file_name: None,
        python_launcher: Some(python_launcher.executable.display().to_string()),
        python_executable: Some(env_python.display().to_string()),
        pip_cache_dir: None,
        rocm_sdk: Some(rocm_sdk_probe.clone()),
        sdk_torch: Some(resolution.package_versions.torch.clone()),
        wheel_composition: Some(wheel_composition),
        read_only: false,
        imported_from: None,
        devel: include_devel,
        installed_at_unix_ms: unix_time_millis(),
    };
    save_runtime_manifest(paths, &manifest)?;

    let _ = writeln!(
        output,
        "  installed_version: {}",
        runtime_version_display(&installed_version)
    );
    let _ = writeln!(output, "  python_executable: {}", env_python.display());
    if let Some(site_packages) = rocm_sdk_probe.site_packages.as_ref() {
        let _ = writeln!(output, "  site_packages: {}", site_packages.display());
    }
    if let Some(root_path) = rocm_sdk_probe.root_path.as_ref() {
        let _ = writeln!(output, "  rocm_sdk_root: {}", root_path.display());
    }
    if let Some(bin_path) = rocm_sdk_probe.bin_path.as_ref() {
        let _ = writeln!(output, "  rocm_sdk_bin: {}", bin_path.display());
    }
    if let Some(version) = rocm_sdk_probe.rocm_sdk_version.as_deref() {
        let _ = writeln!(
            output,
            "  rocm_sdk_version: {}",
            runtime_version_display(version)
        );
    }
    if let Some(target_family) = rocm_sdk_probe.resolved_target_family.as_deref() {
        let _ = writeln!(output, "  rocm_sdk_target_family: {target_family}");
    }
    let _ = writeln!(output, "  manifest: {}", manifest_path.display());
    Ok(SdkInstallResult::installed(output))
}

/// Quotes `value` for the dry-run preview line only -- it is never fed to a
/// real subprocess. Deliberately kept separate from `rocm_core::quoting`
/// (ROCMAI-727); see that module's doc comment for why.
fn quote_display_arg(value: &str) -> String {
    if value.is_empty()
        || value
            .chars()
            .any(|ch| ch.is_whitespace() || matches!(ch, '[' | ']' | '(' | ')' | '&' | ';' | '|'))
    {
        format!("\"{}\"", value.replace('"', "\\\""))
    } else {
        value.to_owned()
    }
}

/// The host's legacy/system ROCm version when it is strictly newer than the
/// version rocm-cli is about to install, otherwise `None`. Used to explain why a
/// seemingly older wheel version is selected over the host's ROCm.
fn host_rocm_version_newer_than(resolved_version: &str) -> Option<String> {
    host_version_newer_than(detect_legacy_rocm_summary().version, resolved_version)
}

/// Pure core of [`host_rocm_version_newer_than`]: given the host's detected ROCm
/// version (if any) and the version about to be installed, return the host
/// version only when it is strictly newer. Split out from the filesystem probe so
/// the newer-than decision is unit-testable without a real legacy ROCm on disk.
fn host_version_newer_than(host_version: Option<String>, resolved_version: &str) -> Option<String> {
    let host_version = host_version?;
    // Only claim the host is newer when BOTH versions parse and the host is
    // strictly greater. Only a string that fails to parse is "can't tell", and
    // "can't tell" is never "newer". Falling back to a lexicographic compare
    // here wrongly ranks e.g. `7.2.4-98` above `7.13.0` (because '2' > '1' at
    // the third char), inventing a host-newer note that misleads the user.
    //
    // Each side is read with the parser for where it came from.
    // `parse_host_version` additionally drops the packaging build number a
    // legacy ROCm reports (`7.2.4-98` -> 7.2.4); a two-component report
    // (`7.4` -> 7.4.0) and a `-rc1` pre-release are read the same way by both.
    // `resolved_version` is not a host string: it is the version resolved from
    // the index or the tarball catalogue, so it is read exactly as the
    // comparator that picked it reads it.
    let host_parsed = parse_host_version(&host_version)?;
    let resolved_parsed = parse_version_for_ordering(resolved_version)?;
    (host_parsed > resolved_parsed).then_some(host_version)
}

/// The wheel-path `version_note` body explaining why a host-newer legacy ROCm is
/// passed over for the newest TheRock version that still has a matching PyTorch
/// stack. Pure so the exact user-facing wording is unit-testable.
fn wheel_host_version_note(host_version: &str, resolved_display: &str) -> String {
    format!(
        "this host reports ROCm {host_version}, but {resolved_display} is the newest TheRock ROCm with a matching PyTorch stack, so it is selected; pass `--version <VERSION>` to override"
    )
}

/// The tarball-path `version_note` body explaining why a host-newer legacy ROCm is
/// passed over for the newest TheRock tarball for this GPU family.
fn tarball_host_version_note(host_version: &str, resolved_display: &str) -> String {
    format!(
        "this host reports ROCm {host_version}, but {resolved_display} is the newest TheRock ROCm tarball for this GPU family, so it is selected"
    )
}

/// The `warning` body surfaced when the repository's newest version has no
/// installable PyTorch wheels for this Python/platform, so an older one is used.
fn no_wheel_warning_message(newest: &str, resolved_display: &str) -> String {
    format!(
        "ROCm {newest} is the newest version in this repository but has no installable PyTorch wheels for this Python and platform; installing ROCm {resolved_display} instead"
    )
}

/// The repo's newest version when it is strictly newer than the version we are
/// about to install — i.e. the newest release exists in the index but has no
/// installable PyTorch wheels for this Python/platform, so an older one is used.
/// Returns `None` when the newest version is the one being installed (or is
/// unknown), so the caller only warns when there is a real gap.
fn repo_version_without_wheels(
    newest_repo: Option<&str>,
    resolved_version: &str,
) -> Option<String> {
    let newest = newest_repo?;
    // "Is there a genuinely newer version?", not "which string sorts higher?".
    // Two spellings of one version would otherwise produce a warning naming the
    // same version on both sides of "installing X instead", and a pair this
    // tool cannot identify would produce one it cannot justify.
    matches!(
        version_relation(newest, resolved_version),
        Some(Ordering::Greater)
    )
    .then(|| newest.to_owned())
}

#[allow(clippy::too_many_arguments)]
fn install_tarball_runtime(
    paths: &AppPaths,
    channel: TheRockChannel,
    prefix: Option<PathBuf>,
    family_override: Option<&str>,
    version_selector: Option<&RuntimeVersionSelector>,
    layout_override: Option<SourceLayout>,
    dry_run: bool,
    consent: SdkInstallConsent,
) -> Result<SdkInstallResult> {
    let artifact = resolve_tarball_artifact(
        paths,
        channel,
        family_override,
        version_selector,
        layout_override,
    )?;
    let runtime_key = runtime_key(
        channel,
        "tarball",
        &artifact.family,
        Some(&artifact.version),
    );
    let install_root = resolved_install_root(paths, "tarball", &runtime_key, prefix);
    let manifest_path = runtime_manifest_path(paths, &runtime_key);
    let cache_path = paths.cache_dir.join("therock").join(&artifact.file_name);

    let mut output = String::new();
    let _ = writeln!(output, "sdk install");
    render_canonical_provenance(
        &mut output,
        channel,
        &artifact.catalog_url,
        artifact.layout.generation(),
        &artifact.version,
    );
    let _ = writeln!(output, "  format: tarball");
    let _ = writeln!(output, "  family: {}", artifact.family);
    let _ = writeln!(output, "  family_source: {}", artifact.family_source);
    let _ = writeln!(output, "  tarball: {}", artifact.file_name);
    let _ = writeln!(output, "  tarball_url: {}", artifact.url);
    let _ = writeln!(
        output,
        "  latest_version: {}",
        runtime_version_display(&artifact.version)
    );
    // Probed once and reused by the progress line below; see the wheel path.
    let host_version_newer = host_rocm_version_newer_than(&artifact.version);
    if let Some(host_version) = host_version_newer.as_deref() {
        let _ = writeln!(
            output,
            "  version_note: {}",
            tarball_host_version_note(host_version, &runtime_version_display(&artifact.version))
        );
    }
    let _ = writeln!(output, "  target: {}", install_root.display());
    let _ = writeln!(output, "  cache_path: {}", cache_path.display());
    let _ = writeln!(output, "  runtime_key: {runtime_key}");
    if dry_run {
        let _ = writeln!(output, "  mode: dry-run");
        let _ = writeln!(output, "  manifest: {}", manifest_path.display());
        return Ok(SdkInstallResult::plan(output));
    }

    // Mirror the wheel path: surface the host-newer ROCm explanation as a visible
    // line so "why this version and not the host's newer ROCm" is in the install
    // log rather than only in the trailing summary block.
    if let Some(host_version) = host_version_newer.as_deref() {
        progress_line(format!(
            "Note: this host reports ROCm {host_version}, but ROCm {resolved} is the newest TheRock ROCm tarball for this GPU family; installing {resolved}.",
            resolved = runtime_version_display(&artifact.version)
        ));
    }

    // Same gate as the wheel path: only an install that would displace the
    // current active default runtime asks for confirmation, and it asks
    // regardless of family or channel because activation is global.
    let existing =
        active_default_runtime_relation(paths, channel, &artifact.family, &artifact.version)?;
    match sdk_install_approval(existing.is_some(), consent, interactive_terminal()) {
        SdkInstallApproval::ProceedFresh => {
            progress_line(fresh_install_line(
                &runtime_version_display(&artifact.version),
                &artifact.family,
            ));
        }
        SdkInstallApproval::ProceedApproved(source) => {
            progress_line(preapproved_install_line(
                source,
                existing.as_deref().unwrap_or_default(),
                &runtime_version_display(&artifact.version),
            ));
        }
        SdkInstallApproval::PromptOverwrite => {
            if !confirm_overwrite_existing_sdk(
                channel,
                &artifact.family,
                &artifact.version,
                existing.as_deref().unwrap_or_default(),
            )? {
                let _ = writeln!(
                    output,
                    "  status: cancelled by user; the existing ROCm SDK was left unchanged"
                );
                return Ok(SdkInstallResult::plan(output));
            }
        }
        SdkInstallApproval::RefuseNonInteractive => {
            bail!(refuse_non_interactive_message(
                existing.as_deref().unwrap_or_default()
            ));
        }
    }

    fs::create_dir_all(paths.cache_dir.join("therock"))?;
    fs::create_dir_all(&install_root)?;
    if has_nontrivial_directory_contents(&install_root)? {
        bail!(
            "tarball install target {} is not empty; choose a clean prefix or remove the old extraction first",
            install_root.display()
        );
    }

    if let Some(warning) = preflight_tarball_space(&artifact.url, &cache_path, &install_root)? {
        let _ = writeln!(output, "  {warning}");
    }

    let download_label = format!("Downloading {}…", artifact.file_name);
    let spinner = crate::cli_progress::AnimatedSpinner::start(download_label.clone());
    let download_result = download_file(&artifact.url, &cache_path, &mut |bytes, total| {
        spinner.set_progress(&download_label, bytes, total);
    });
    drop(spinner);
    download_result?;

    let extract_spinner =
        crate::cli_progress::AnimatedSpinner::start(format!("Extracting {}…", artifact.file_name));
    let extract_result = extract_tarball_and_discard_archive(&cache_path, &install_root);
    drop(extract_spinner);
    if let Some(cleanup_warning) = extract_result? {
        progress_line(cleanup_warning);
    }

    let manifest = InstalledRuntimeManifest {
        runtime_key: runtime_key.clone(),
        runtime_id: format!("therock-{}:{}", channel.as_str(), artifact.family),
        channel: channel.as_str().to_owned(),
        format: "tarball".to_owned(),
        family: artifact.family.clone(),
        family_source: artifact.family_source.clone(),
        version: artifact.version.clone(),
        install_root: install_root.clone(),
        selected_artifact_url: artifact.url.clone(),
        source_layout_generation: Some(artifact.layout.generation().to_owned()),
        index_url: None,
        tarball_file_name: Some(artifact.file_name.clone()),
        python_launcher: None,
        python_executable: None,
        pip_cache_dir: None,
        rocm_sdk: None,
        sdk_torch: None,
        wheel_composition: None,
        read_only: false,
        imported_from: None,
        // Tarball artifacts ship the whole SDK; the extra is a wheel concept.
        devel: true,
        installed_at_unix_ms: unix_time_millis(),
    };
    save_runtime_manifest(paths, &manifest)?;

    let _ = writeln!(output, "  extracted: {}", install_root.display());
    let _ = writeln!(output, "  manifest: {}", manifest_path.display());
    Ok(SdkInstallResult::installed(output))
}

/// Identify a wheel runtime by channel, version, AND the exact composition it
/// was installed from.
///
/// Two installs of the same version that request different device payloads are
/// different runtimes: one can run this host's kernels and the other cannot. A
/// version-only key gave them the same name, so a corrected composition
/// overwrote the old tree in place — the one thing side-by-side installs exist
/// to avoid — and left no way to tell the two apart afterwards.
///
/// The fingerprint is a truncated SHA-256 over the generation and the specs,
/// length-delimited so no regrouping of the same characters collides. Truncation
/// is safe here: this names sibling directories, it does not authenticate them.
fn wheel_runtime_key(
    channel: TheRockChannel,
    version: &str,
    composition: &WheelRuntimeComposition,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(composition.source_layout_generation.as_bytes());
    for package_spec in &composition.package_specs {
        hasher.update([0]);
        hasher.update(package_spec.as_bytes());
    }
    hasher.update([0]);
    hasher.update(b"rocm-sdk-target");
    if let Some(target) = &composition.rocm_sdk_target {
        hasher.update([0]);
        hasher.update(target.as_bytes());
    }
    let fingerprint = format!("{:x}", hasher.finalize());
    slugify(&format!(
        "{}-wheel-multi-arch-{version}-{}",
        channel.as_str(),
        &fingerprint[..16]
    ))
}

const fn platform_tarball_token() -> &'static str {
    if runtime_is_windows() {
        "windows"
    } else {
        "linux"
    }
}

fn runtime_key(
    channel: TheRockChannel,
    format: &str,
    family: &str,
    version: Option<&str>,
) -> String {
    match version {
        Some(version) if !version.trim().is_empty() => {
            slugify(&format!("{}-{format}-{family}-{version}", channel.as_str()))
        }
        _ => slugify(&format!("{}-{format}-{family}", channel.as_str())),
    }
}

fn managed_runtime_root(paths: &AppPaths, format: &str, runtime_key: &str) -> PathBuf {
    paths
        .data_dir
        .join("runtimes")
        .join(format)
        .join(runtime_key)
}

/// Where to build a runtime, resolved to its real location on disk first.
///
/// `install sdk` writes this path into three places that all outlive the command:
/// the registry manifest, the sidecar beside the runtime, and — via `uv` — the
/// `#!` line of every console script in the venv. So the path handed to the
/// installer has to name where the files land, not the route taken to get there.
/// Reaching `data/runtimes` through a symlink is enough to make those differ, and
/// once the link goes the runtime reports itself installed at a folder that is not
/// there while the files sit untouched next door.
///
/// Applies to `--prefix` too, which has the identical failure mode.
///
/// Only the root is resolved. `python_executable` is derived from it and must
/// keep the venv's own `bin/python`, which is itself a symlink to the base
/// interpreter — resolving that would record the system Python and break venv
/// semantics. The adopt path already draws the line in the same place: see
/// `main::adopt_runtime_from_probe`, which canonicalizes the install root next to
/// `absolute_existing_file_path_preserving_symlink` for the interpreter.
fn resolved_install_root(
    paths: &AppPaths,
    format: &str,
    runtime_key: &str,
    prefix: Option<PathBuf>,
) -> PathBuf {
    let requested = prefix.unwrap_or_else(|| managed_runtime_root(paths, format, runtime_key));
    rocm_core::resolve_path_through_symlinks(&requested)
}

fn has_nontrivial_directory_contents(path: &Path) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let entries =
        fs::read_dir(path).with_context(|| format!("failed to read {}", path.display()))?;
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        if name.to_string_lossy().starts_with('.') {
            continue;
        }
        return Ok(true);
    }
    Ok(false)
}

fn venv_python_path(install_root: &Path) -> PathBuf {
    runtime_python_executable_in_env(install_root)
}

fn slugify(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => ch.to_ascii_lowercase(),
            _ => '-',
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[cfg(windows)]
    use super::test_support::workspace_test_artifact_dir;
    use super::test_support::{test_paths, test_runtime_manifest};
    use super::*;

    #[test]
    fn download_space_requirement_includes_the_safety_margin() {
        let archive = 2 * 1024 * 1024 * 1024;
        assert_eq!(
            disk_space::with_margin(archive),
            archive + archive / disk_space::SPACE_MARGIN_DIVISOR
        );
        assert!(disk_space::with_margin(archive) > archive);
    }

    #[test]
    fn extraction_estimate_exceeds_the_compressed_archive() {
        let archive = 3 * 1024 * 1024 * 1024;
        let estimate = disk_space::estimated_extracted_size(archive);
        assert!(
            estimate > archive,
            "extraction must reserve headroom beyond the archive: {estimate} vs {archive}"
        );
        assert_eq!(estimate, archive * disk_space::EXTRACTED_SIZE_MULTIPLIER);
    }

    #[test]
    fn write_file_atomically_reports_a_full_disk_clearly() {
        // Exercise the mapping the write path uses, without filling a disk.
        let error = disk_space::map_write_error(
            std::io::Error::from(std::io::ErrorKind::StorageFull),
            Path::new("/cache/rocm.tar.gz.tmp"),
        );
        let text = format!("{error:#}");
        assert!(text.contains("ran out of disk space"), "{text}");
    }

    #[test]
    fn normalize_therock_family_maps_gfx1103_to_gfx110x_all() {
        assert_eq!(
            normalize_therock_family("gfx1103"),
            Some("gfx110X-all".to_owned())
        );
    }

    #[test]
    fn normalize_therock_family_maps_gfx1101_to_gfx110x_all() {
        assert_eq!(
            normalize_therock_family("gfx1101"),
            Some("gfx110X-all".to_owned())
        );
    }

    #[test]
    fn gfx90c_resolves_to_no_family_instead_of_the_datacenter_one() {
        // gfx90c is the Vega iGPU in Renoir/Cezanne APUs. It used to normalize to
        // `gfx90X-dcgpu`, so a plain `rocm install sdk` targeted data-center
        // packages and the error listed that family as a `--family` choice.
        assert_eq!(normalize_therock_family("gfx90c"), None);
    }

    #[test]
    fn canonical_channels_use_only_their_aggregate_streams() {
        let release = resolve_source(TheRockChannel::Release, SourceLayout::Canonical);
        assert_eq!(
            release.wheel_index,
            "https://repo.amd.com/rocm/whl-multi-arch"
        );
        assert_eq!(
            release.tarball_catalog,
            "https://repo.amd.com/rocm/tarball/"
        );
        assert_eq!(release.layout.generation(), "multi-arch-v2");

        let nightly = resolve_source(TheRockChannel::Nightly, SourceLayout::Canonical);
        assert_eq!(
            nightly.wheel_index,
            "https://rocm.nightlies.amd.com/whl-multi-arch"
        );
        assert_eq!(
            nightly.tarball_catalog,
            "https://rocm.nightlies.amd.com/tarball/"
        );
        assert_eq!(nightly.layout.generation(), "multi-arch-v2");
    }

    #[test]
    fn canonical_provenance_reports_required_dry_run_fields() {
        let source = resolve_source(TheRockChannel::Nightly, SourceLayout::Canonical);
        let mut output = String::new();
        render_canonical_provenance(
            &mut output,
            TheRockChannel::Nightly,
            &source.wheel_index,
            source.layout.generation(),
            "10.1.0a20260822",
        );

        assert!(output.contains("channel: nightly"));
        assert!(output.contains("canonical_source: https://rocm.nightlies.amd.com/whl-multi-arch"));
        assert!(output.contains("selected_rocm_version: 10.1.0a20260822"));
        assert!(output.contains("build_date: 2026-08-22"));
        assert!(output.contains("source_layout_generation: multi-arch-v2"));
    }

    /// The registry load itself breaks install-time ties on the runtime key,
    /// not only the sort helper it calls: the startup update check and
    /// `runtimes_cmd::newest_installed_runtime_manifest` both take the first
    /// entry `load_runtime_manifests` returns, and neither re-sorts.
    ///
    /// Both pairings of file name and runtime key are written, so whichever
    /// file `read_dir` returns first holds `bbb` in one of them: a load that
    /// kept the read order for a tie picks the wrong runtime there.
    #[test]
    fn registry_load_breaks_install_time_ties_on_the_runtime_key() -> Result<()> {
        for (first_file_key, second_file_key) in [("aaa", "bbb"), ("bbb", "aaa")] {
            let (root, paths) = test_paths("registry-load-tie");
            let registry = runtime_registry_dir(&paths);
            fs::create_dir_all(&registry)?;
            for (file, key) in [("first", first_file_key), ("second", second_file_key)] {
                fs::write(
                    registry.join(format!("{file}.json")),
                    serde_json::to_vec_pretty(&test_runtime_manifest(
                        key,
                        "therock-release:gfx120X-all",
                        2_000,
                    ))?,
                )?;
            }
            let pairing = format!("files first={first_file_key}, second={second_file_key}");

            let keys = load_runtime_manifests(&paths)?
                .into_iter()
                .map(|manifest| manifest.runtime_key)
                .collect::<Vec<_>>();
            assert_eq!(keys, ["aaa", "bbb"], "{pairing}");
            assert_eq!(
                crate::runtimes_cmd::newest_installed_runtime_manifest(&paths)?
                    .map(|manifest| manifest.runtime_key)
                    .as_deref(),
                Some("aaa"),
                "{pairing}"
            );

            let _ = fs::remove_dir_all(&root);
        }
        Ok(())
    }

    /// A data dir whose `runtimes` folder is a symlink to somewhere else, plus the
    /// real folder it points at — both canonical, so an assertion cannot pass or
    /// fail on whether the machine's temp dir is itself behind a link.
    ///
    /// Deliberately not `test_paths`: that builds paths by plain `join` and never
    /// canonicalizes, which is exactly the property under test here.
    #[cfg(unix)]
    fn linked_runtimes_paths(name: &str) -> (PathBuf, AppPaths, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "rocm-cli-linked-runtimes-{name}-{}",
            crate::test_support::unique_suffix()
        ));
        fs::create_dir_all(root.join("data")).unwrap();
        let root = root.canonicalize().unwrap();
        let real = root.join("data").join("real-runtimes");
        fs::create_dir_all(&real).unwrap();
        std::os::unix::fs::symlink(&real, root.join("data").join("runtimes")).unwrap();
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        (root, paths, real)
    }

    #[cfg(unix)]
    #[test]
    fn linked_runtimes_paths_gives_each_call_its_own_root_even_for_one_label() {
        crate::test_support::assert_each_call_gets_its_own_root(|| {
            linked_runtimes_paths("same-label").0
        });
    }

    #[test]
    #[cfg(unix)]
    fn install_root_resolves_a_symlinked_runtimes_folder() {
        // The e2e harness points a scenario's `data/runtimes` at a shared tree this
        // way. Recording the link's path made the runtime name a folder that
        // vanished with the scenario, while the files stayed where they were
        // written (rocm-cli#315).
        let (root, paths, real) = linked_runtimes_paths("resolves");
        let runtime_key = "release-wheel-gfx120x-all-7-14-0";

        let resolved = resolved_install_root(&paths, "wheel", runtime_key, None);

        assert_eq!(resolved, real.join("wheel").join(runtime_key));
        assert!(
            !resolved.starts_with(paths.data_dir.join("runtimes")),
            "the recorded root must not be expressed through the link: {}",
            resolved.display()
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    #[cfg(unix)]
    fn install_root_resolves_a_symlinked_prefix() {
        // `--prefix` has the identical failure mode, so it gets the identical fix.
        let (root, paths, real) = linked_runtimes_paths("prefix");
        let prefix = paths.data_dir.join("runtimes").join("chosen-env");

        let resolved = resolved_install_root(&paths, "wheel", "unused-key", Some(prefix));

        assert_eq!(resolved, real.join("chosen-env"));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    #[cfg(unix)]
    fn install_root_resolves_the_tarball_format_too() {
        let (root, paths, real) = linked_runtimes_paths("tarball");
        let runtime_key = "release-tarball-gfx120x-all-7-14-0";

        let resolved = resolved_install_root(&paths, "tarball", runtime_key, None);

        assert_eq!(resolved, real.join("tarball").join(runtime_key));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn install_root_is_unchanged_when_nothing_is_linked() {
        // Guards against gratuitous rewriting: on an ordinary tree the recorded
        // root must still be the plain layout path, so existing installs and the
        // paths compared against them do not move.
        let root = std::env::temp_dir().join(format!(
            "rocm-cli-plain-runtimes-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        fs::create_dir_all(root.join("data")).unwrap();
        // Resolved rather than canonicalized: on Windows `canonicalize` returns a
        // verbatim `\\?\C:\…` path, so the expectation would carry a prefix the
        // resolver deliberately strips and the test would fail on that rather than
        // on whether the layout path moved.
        let root = rocm_core::resolve_path_through_symlinks(&root);
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        let runtime_key = "release-wheel-gfx120x-all-7-14-0";

        assert_eq!(
            resolved_install_root(&paths, "wheel", runtime_key, None),
            managed_runtime_root(&paths, "wheel", runtime_key)
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn managed_uv_cache_sits_under_the_data_dir_for_generated_runtime_folders() {
        let (_root, paths) = test_paths("managed-uv-cache");
        let runtime_key = "release-wheel-multi-arch-7-14-0";
        let install_root = managed_runtime_root(&paths, "wheel", runtime_key);
        assert!(install_root.starts_with(&paths.data_dir));
        // Without --prefix the generated runtime folder is itself under the data dir, so
        // the uv cache is reachable from the environment it populates without crossing a
        // mount point.
        assert!(managed_uv_cache_dir(&paths.data_dir).starts_with(&paths.data_dir));
    }

    #[test]
    fn uv_cache_does_not_follow_a_prefix_install_root() {
        // Documents a known gap rather than an intended behavior: `--prefix` relocates
        // install_root only, while the uv cache stays keyed off the data dir. When reaching
        // one from the other crosses a mount point uv falls back to copying — it is the
        // mount, not the filesystem, so a bind mount is enough. Tracked separately; see
        // the `--prefix` non-goal on the PR that introduced the colocation.
        let (_root, paths) = test_paths("prefix-uv-cache");
        let prefix_root = PathBuf::from("/mnt/elsewhere/envs/my-env");
        let cache = managed_uv_cache_dir(&paths.data_dir);

        assert!(
            !cache.starts_with(&prefix_root),
            "cache {} unexpectedly followed the --prefix root",
            cache.display()
        );
        assert!(cache.starts_with(&paths.data_dir));
    }

    #[test]
    fn display_command_quotes_package_extras() {
        assert_eq!(quote_display_arg("package[extra]"), "\"package[extra]\"");
        assert_eq!(
            quote_display_arg("C:\\Program Files\\Python\\python.exe"),
            "\"C:\\Program Files\\Python\\python.exe\""
        );
    }

    #[test]
    fn aggregate_wheel_runtime_key_separates_device_payloads_at_one_version() {
        let base = WheelRuntimeComposition {
            source_layout_generation: "multi-arch-v2".to_owned(),
            package_specs: vec!["rocm[libraries,devel,device-gfx942]==7.14.0".to_owned()],
            rocm_sdk_target: Some("gfx942".to_owned()),
        };
        let other_payload = WheelRuntimeComposition {
            source_layout_generation: "multi-arch-v2".to_owned(),
            package_specs: vec!["rocm[libraries,devel,device-gfx950]==7.14.0".to_owned()],
            rocm_sdk_target: Some("gfx950".to_owned()),
        };
        let other_generation = WheelRuntimeComposition {
            source_layout_generation: "multi-arch-v3".to_owned(),
            package_specs: base.package_specs.clone(),
            rocm_sdk_target: base.rocm_sdk_target.clone(),
        };
        let mut legacy_probe_default = base.clone();
        legacy_probe_default.rocm_sdk_target = None;

        let base_key = wheel_runtime_key(TheRockChannel::Release, "7.14.0", &base);

        // Still names its channel and version: the retention policy and every
        // human reading `runtimes list` group on that prefix.
        assert!(
            base_key.starts_with("release-wheel-multi-arch-7-14-0-"),
            "{base_key}"
        );
        assert_eq!(
            base_key,
            wheel_runtime_key(TheRockChannel::Release, "7.14.0", &base),
            "the same composition must always name the same runtime"
        );
        assert_ne!(
            base_key,
            wheel_runtime_key(TheRockChannel::Release, "7.14.0", &other_payload),
            "a different device payload is a different runtime, not an overwrite"
        );
        assert_ne!(
            base_key,
            wheel_runtime_key(TheRockChannel::Release, "7.14.0", &legacy_probe_default,),
            "a runtime probed without the exact SDK target must be repaired side by side"
        );
        assert_ne!(
            base_key,
            wheel_runtime_key(TheRockChannel::Release, "7.14.0", &other_generation),
            "a source-layout generation change must not reuse the old tree"
        );
    }

    #[test]
    fn stable_provenance_uses_neutral_build_date_wording() {
        let source = resolve_source(TheRockChannel::Release, SourceLayout::Canonical);
        let mut output = String::new();
        render_canonical_provenance(
            &mut output,
            TheRockChannel::Release,
            &source.wheel_index,
            source.layout.generation(),
            "7.14.0",
        );
        assert!(output.contains("build_date: not encoded in stable version"));
        assert!(!output.contains("not published"));
    }

    #[test]
    fn windows_v1_rejects_tarball_runtime_format() {
        let error = ensure_install_format_supported_for_platform("tarball", true)
            .unwrap_err()
            .to_string();

        assert!(error.contains("tarball installs are not supported on Windows"));
        assert!(error.contains("rocm install sdk --format wheel"));
        assert!(error.contains("managed wheel virtual environment"));
    }

    #[test]
    fn linux_allows_tarball_runtime_format() {
        ensure_install_format_supported_for_platform("tarball", false).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn install_sdk_rejects_tarball_on_windows_before_resolution() {
        let root = workspace_test_artifact_dir().join(format!(
            "rocm-cli-therock-test-{}",
            crate::test_support::unique_suffix()
        ));
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };

        let error = install_sdk(
            &paths,
            SdkInstallRequest {
                channel: "release",
                format: "tarball",
                dry_run: true,
                consent: SdkInstallConsent::Preapproved(SdkInstallApprovalSource::AssumeYes),
                ..SdkInstallRequest::default()
            },
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("tarball installs are not supported on Windows"));
        assert!(error.contains("rocm install sdk --format wheel"));
    }

    #[test]
    fn repo_version_without_wheels_warns_only_when_newest_is_newer() {
        // Newest repo version has no wheels (newer than the installable one) -> warn.
        assert_eq!(
            repo_version_without_wheels(Some("7.14.0"), "7.13.0").as_deref(),
            Some("7.14.0")
        );
        // Newest repo version is the one being installed -> no warning.
        assert!(repo_version_without_wheels(Some("7.13.0"), "7.13.0").is_none());
        // A specific version was requested (no "newest" known) -> no warning.
        assert!(repo_version_without_wheels(None, "7.13.0").is_none());
        // Defensive: an older "newest" (should not happen) never warns.
        assert!(repo_version_without_wheels(Some("7.12.0"), "7.13.0").is_none());
    }

    #[test]
    fn host_version_newer_than_reports_only_a_strictly_newer_host() {
        // Host ROCm is newer than the version being installed -> surface it.
        assert_eq!(
            host_version_newer_than(Some("7.14.0".to_owned()), "7.13.0").as_deref(),
            Some("7.14.0")
        );
        // Host ROCm matches the installed version -> nothing to explain.
        assert!(host_version_newer_than(Some("7.13.0".to_owned()), "7.13.0").is_none());
        // Host ROCm is older than the installed version -> nothing to explain.
        assert!(host_version_newer_than(Some("7.12.0".to_owned()), "7.13.0").is_none());
        // No legacy ROCm detected on the host -> nothing to explain.
        assert!(host_version_newer_than(None, "7.13.0").is_none());

        // A build-suffixed host version must not be lexicographically ranked
        // above the resolved version: `7.2.4-98` is numerically OLDER than
        // `7.13.0`, so no host-newer note. (This is the reported regression:
        // char-compare put `7.2…` above `7.13…`.)
        assert!(host_version_newer_than(Some("7.2.4-98".to_owned()), "7.13.0").is_none());
        // Two-component host reports are parsed as `.0`; still older here.
        assert!(host_version_newer_than(Some("7.4".to_owned()), "7.13.0").is_none());
        assert!(host_version_newer_than(Some("7.9".to_owned()), "7.13.0").is_none());
        // A build suffix on an equal version is not "newer".
        assert!(host_version_newer_than(Some("7.13.0-56".to_owned()), "7.13.0").is_none());
        // A genuinely newer build-suffixed host is surfaced, keeping the
        // original reported string (suffix included) for the user-facing note.
        assert_eq!(
            host_version_newer_than(Some("7.20.1-33".to_owned()), "7.13.0").as_deref(),
            Some("7.20.1-33")
        );
        // A host string we cannot parse is "can't tell", never "newer".
        assert!(host_version_newer_than(Some("unknown".to_owned()), "7.13.0").is_none());
    }

    #[test]
    fn parse_host_version_drops_only_an_all_digit_build_suffix() {
        // An all-digit tail is a packaging build number and is dropped.
        assert_eq!(
            parse_host_version("7.2.4-98"),
            parse_version_for_ordering("7.2.4")
        );
        assert!(parse_host_version("7.2.4-98").is_some());
        // A PEP 440 pre-release spelled with `-` is kept and read as one.
        assert_eq!(
            parse_host_version("7.0.0-rc1"),
            parse_version_for_ordering("7.0.0rc1")
        );
        // A tail that starts with a digit but is not all digits is neither a
        // build number nor a PEP 440 stage. The whole string goes to the
        // ordering parser, which cannot read it, so the host version is
        // unreadable rather than silently read as `7.2.4`.
        assert_eq!(parse_host_version("7.2.4-98rc1"), None);
        assert_eq!(parse_version_for_ordering("7.2.4-98rc1"), None);
        // What the user sees: an unreadable host version is "can't tell", so
        // no host-newer note, even where the dropped-suffix reading (`7.20.1`)
        // would have been newer than the version being installed.
        assert!(host_version_newer_than(Some("7.20.1-33rc1".to_owned()), "7.13.0").is_none());
        // An empty tail is not a build number: `.all()` holds vacuously on it,
        // so without the emptiness guard `7.2.4-` would be read as `7.2.4`.
        assert_eq!(parse_host_version("7.2.4-"), None);
        // The rule is deliberately strict: only an all-digit tail is a build
        // number, so distro-style packaging tails are unreadable too. The cost
        // is a missing host-newer note for such a host, never a false one.
        for distro_tail in [
            "7.2.4-98.el9",
            "6.4.1-120~22.04",
            "7.2.4-98-1",
            "7.2.4-98+local",
        ] {
            assert_eq!(parse_host_version(distro_tail), None, "{distro_tail}");
        }
    }

    #[test]
    fn host_version_notes_and_warning_render_the_expected_text() {
        // Wheel path: the note names both versions and offers the --version override.
        let wheel = wheel_host_version_note("7.14.0", "7.13.0");
        assert_eq!(
            wheel,
            "this host reports ROCm 7.14.0, but 7.13.0 is the newest TheRock ROCm with a matching PyTorch stack, so it is selected; pass `--version <VERSION>` to override"
        );

        // Tarball path: same explanation, phrased for the GPU-family tarball.
        let tarball = tarball_host_version_note("7.14.0", "7.13.0");
        assert_eq!(
            tarball,
            "this host reports ROCm 7.14.0, but 7.13.0 is the newest TheRock ROCm tarball for this GPU family, so it is selected"
        );

        // The no-wheels warning names the repo-newest and the fallback it installs.
        let warning = no_wheel_warning_message("7.14.0", "7.13.0");
        assert_eq!(
            warning,
            "ROCm 7.14.0 is the newest version in this repository but has no installable PyTorch wheels for this Python and platform; installing ROCm 7.13.0 instead"
        );
    }
}
