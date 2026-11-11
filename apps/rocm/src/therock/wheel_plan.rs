// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The `sdk install` plan a wheel install prints, key by key.
//!
//! A preview prints it and stops; a real install prints it as the head of its
//! own output. [`write_wheel_plan`] writes the lines both share, and
//! [`write_dry_run_tail`] the lines only a preview adds.
//!
//! One writer serves a resolved plan and one that could not be resolved, so the
//! two cannot drift apart: a line the plan gains appears in both. They differ
//! only in the lines made from the package versions `uv pip compile` chooses on
//! the ROCm 10 layout (`package_specs`, `runtime_key`, `target`, `manifest` and
//! `command`). [`PlanStack::NeedsUv`] says those are not resolved, and why,
//! where [`PlanStack::Resolved`] carries their values.

use super::kept_torch::{self, KeptTorch};
use super::python::{PlannedPython, WheelCompatibility};
use super::resolve::{AggregateDeviceTarget, TorchaudioPin, UnresolvedPipRuntime};
use super::source::{RuntimeVersionSelector, TheRockChannel};
use super::{
    no_wheel_warning_for, render_canonical_provenance, runtime_version_display, venv_python_path,
    wheel_host_version_note,
};
use crate::shell_quote::shell_quote;
use rocm_core::{uv_pip_install_base, uv_venv_args};
use std::fmt::Write as _;
use std::path::Path;

/// The value printed for a line that depends on the package versions `uv` has not
/// chosen.
pub(super) const NOT_RESOLVED: &str = "not resolved by this preview";

/// The facts the plan prints, from a resolution and the Python decision.
pub(super) struct WheelPlan<'a> {
    pub(super) channel: TheRockChannel,
    pub(super) version_selector: Option<&'a RuntimeVersionSelector>,
    pub(super) index_url: &'a str,
    pub(super) layout_generation: &'a str,
    /// The `rocm` version the plan is for.
    pub(super) latest_version: &'a str,
    pub(super) family: &'a str,
    pub(super) family_source: &'a str,
    pub(super) device_target: &'a AggregateDeviceTarget,
    /// The host's ROCm when it is newer than `latest_version`.
    pub(super) host_version_newer: Option<&'a str>,
    pub(super) python: &'a PlannedPython,
    pub(super) wheel_compatibility: &'a WheelCompatibility,
    /// The requirement lines version resolution asked `uv` about. `None` on the
    /// canonical layout, which resolves without `uv`.
    pub(super) version_resolution_specs: Option<&'a [String]>,
    pub(super) torchaudio_pin: Option<&'a TorchaudioPin>,
    pub(super) no_wheel_warning: Option<&'a str>,
    pub(super) stack: PlanStack<'a>,
}

/// Everything in the plan that is made from the package versions.
pub(super) enum PlanStack<'a> {
    /// The versions are chosen; these are what they produce.
    Resolved(ResolvedStack<'a>),
    /// A preview of the ROCm 10 layout with no `uv`: the versions were not chosen,
    /// so nothing made from them can be printed.
    NeedsUv {
        /// The install folder when the user named one with `--prefix`: it does not
        /// depend on the versions. Otherwise the runtime key names it.
        explicit_install_root: Option<&'a Path>,
    },
}

pub(super) struct ResolvedStack<'a> {
    pub(super) compatibility_key: &'a str,
    pub(super) install_root: &'a Path,
    pub(super) runtime_key: &'a str,
    pub(super) package_specs: &'a [String],
    /// The torch stack the install leaves alone, when the opt-out asked for that and
    /// the environment has one. The specs above stay the runtime's composition, as
    /// recorded in its manifest; what `uv` is handed is [`kept_torch::install_specs`]
    /// of them.
    pub(super) kept_torch: Option<&'a KeptTorch>,
    pub(super) manifest_path: &'a Path,
}

/// The `compatibility_key` the install is recorded under is the `rocm` build every
/// package in the stack shares, so a plan that has not chosen the stack still
/// knows it: the `rocm` version.
const fn compatibility_key<'a>(plan: &'a WheelPlan<'a>) -> &'a str {
    match &plan.stack {
        PlanStack::Resolved(stack) => stack.compatibility_key,
        PlanStack::NeedsUv { .. } => plan.latest_version,
    }
}

/// Writes the plan from `sdk install` to `package_policy` and the repository
/// warning. A real install carries on from here; a preview adds
/// [`write_dry_run_tail`].
pub(super) fn write_wheel_plan(output: &mut String, plan: &WheelPlan<'_>) {
    let _ = writeln!(output, "sdk install");
    let _ = writeln!(
        output,
        "  summary: rocm-cli will install the ROCm SDK and matching PyTorch packages for this Python and operating system"
    );
    render_canonical_provenance(
        output,
        plan.channel,
        plan.index_url,
        plan.layout_generation,
        plan.latest_version,
    );
    let _ = writeln!(output, "  format: wheel");
    if let Some(selector) = plan.version_selector {
        let _ = writeln!(output, "  requested: {}", selector.describe());
    }
    let _ = writeln!(output, "  target_family: {}", plan.family);
    let _ = writeln!(output, "  target_family_source: {}", plan.family_source);
    let _ = writeln!(output, "  device_target: {}", plan.device_target.as_str());
    if let Some(reason) = plan.device_target.reason() {
        let _ = writeln!(output, "  device_target_reason: {reason}");
    }
    let _ = writeln!(output, "  index_url: {}", plan.index_url);
    let _ = writeln!(
        output,
        "  latest_compatible_version: {}",
        runtime_version_display(plan.latest_version)
    );
    if let Some(host_version) = plan.host_version_newer {
        let _ = writeln!(
            output,
            "  version_note: {}",
            wheel_host_version_note(host_version, &runtime_version_display(plan.latest_version))
        );
    }
    let _ = writeln!(
        output,
        "  compatibility_key: {}",
        runtime_version_display(compatibility_key(plan))
    );
    match &plan.stack {
        PlanStack::Resolved(stack) => {
            let _ = writeln!(output, "  target: {}", stack.install_root.display());
            let _ = writeln!(output, "  runtime_key: {}", stack.runtime_key);
        }
        PlanStack::NeedsUv {
            explicit_install_root,
        } => {
            match explicit_install_root {
                Some(root) => {
                    let _ = writeln!(output, "  target: {}", root.display());
                }
                None => {
                    let _ = writeln!(output, "  target: {NOT_RESOLVED}");
                }
            }
            let _ = writeln!(output, "  runtime_key: {NOT_RESOLVED}");
        }
    }
    let _ = writeln!(
        output,
        "  python_launcher: {}",
        plan.python.launcher_display()
    );
    let _ = writeln!(output, "  python_source: {}", plan.python.source().as_str());
    if let Some(note) = plan.python.note() {
        let _ = writeln!(output, "  python_note: {note}");
    }
    if matches!(plan.stack, PlanStack::NeedsUv { .. }) {
        let _ = writeln!(output, "  resolution_note: {UNRESOLVED_NOTE}");
    }
    let _ = writeln!(
        output,
        "  python_wheel_tag: {}",
        plan.wheel_compatibility.python_tag
    );
    let _ = writeln!(
        output,
        "  platform_wheel_tags: {}",
        plan.wheel_compatibility.platform_tags.join(",")
    );
    // What resolution asked for, printed beside what the install will ask for.
    // The two are produced by different code paths from the same `include_devel`
    // and must name the same extras; showing only the second would hide a
    // resolve that constrained the version choice by a toolchain the user
    // declined. Absent on the canonical layout, which resolves by scraping each
    // package's index instead of compiling a requirement set.
    if let Some(specs) = plan.version_resolution_specs {
        let _ = writeln!(output, "  version_resolution_specs: {}", specs.join(" "));
    }
    // Printed from the same value that chose the pin, so the line cannot claim
    // a verified build the requirements above do not name.
    if let Some(pin) = plan.torchaudio_pin {
        let _ = writeln!(output, "  torchaudio_selection: {}", pin.describe());
    }
    match &plan.stack {
        PlanStack::Resolved(stack) => {
            let _ = writeln!(output, "  package_specs: {}", stack.package_specs.join(" "));
        }
        PlanStack::NeedsUv { .. } => {
            let _ = writeln!(output, "  package_specs: {NOT_RESOLVED}");
        }
    }
    let _ = writeln!(
        output,
        "  package_policy: resolve the pinned target-complete rocm, torch, torchvision, and torchaudio plan from published package metadata, then install it in one uv transaction"
    );
    if let PlanStack::Resolved(ResolvedStack {
        kept_torch: Some(kept),
        ..
    }) = &plan.stack
    {
        let _ = writeln!(output, "{}", kept_torch::plan_line(kept));
    }
    if let Some(warning) = plan.no_wheel_warning {
        let _ = writeln!(output, "  warning: {warning}");
    }
}

/// What a preview says about the versions it did not choose. Written beside
/// `python_note`, in the same voice: what a real install would do, and that this
/// preview did not.
pub(super) const UNRESOLVED_NOTE: &str = "no uv is installed, so a real install would fetch it and run `uv pip compile` to choose the torch and torchvision versions and to check that the published packages install together, which this preview did not do; the package specs, runtime key, install folder, manifest and command depend on those versions and are not resolved here";

/// The progress line for the same decision as [`UNRESOLVED_NOTE`], in the voice of
/// the Python line (`... a real install would prepare Python X. This preview does
/// not.`).
pub(super) const UNRESOLVED_PROGRESS: &str = "No uv is installed; a real install would fetch it and resolve the torch and torchvision versions with `uv pip compile`. This preview does not.";

/// The lines a preview adds after [`write_wheel_plan`]: the commands a real install
/// would run, and where it would record the runtime.
pub(super) fn write_dry_run_tail(output: &mut String, plan: &WheelPlan<'_>) {
    let _ = writeln!(output, "  mode: dry-run");
    match &plan.stack {
        PlanStack::Resolved(stack) => {
            let env_python = venv_python_path(stack.install_root);
            let mut install_args = uv_pip_install_base(&env_python);
            install_args.extend(["--index-url".to_owned(), plan.index_url.to_owned()]);
            if matches!(plan.channel, TheRockChannel::Nightly) {
                install_args.extend(["--prerelease".to_owned(), "allow".to_owned()]);
            }
            install_args.extend(kept_torch::install_specs(
                stack.package_specs,
                stack.kept_torch,
            ));
            let venv_args = uv_venv_args(&plan.python.venv_python_arg(), stack.install_root);
            // This line is offered as the command to run, so it has to be one a
            // shell reads back as the argv it previews. The install root in it
            // comes from a `--prefix` the local assistant can choose
            // (`install_sdk_dry_run` is a model-callable tool), so a bare or
            // double-quoted value here would be a model-chosen value in a line a
            // user is invited to paste.
            let venv_args_display = venv_args
                .iter()
                .map(|arg| shell_quote(arg))
                .collect::<Vec<_>>()
                .join(" ");
            let install_args_display = install_args
                .iter()
                .map(|arg| shell_quote(arg))
                .collect::<Vec<_>>()
                .join(" ");
            let _ = writeln!(
                output,
                "  command: uv {venv_args_display} && uv {install_args_display}"
            );
        }
        PlanStack::NeedsUv { .. } => {
            let _ = writeln!(output, "  command: {NOT_RESOLVED}");
        }
    }
    let _ = writeln!(
        output,
        "  activation: use the managed venv Python; TheRock libraries are resolved from that venv by rocm_sdk.initialize_process"
    );
    match &plan.stack {
        PlanStack::Resolved(stack) => {
            let _ = writeln!(output, "  manifest: {}", stack.manifest_path.display());
        }
        PlanStack::NeedsUv { .. } => {
            let _ = writeln!(output, "  manifest: {NOT_RESOLVED}");
        }
    }
}

/// What [`unresolved_wheel_preview`] is made from.
pub(super) struct UnresolvedPreview<'a> {
    pub(super) channel: TheRockChannel,
    pub(super) version_selector: Option<&'a RuntimeVersionSelector>,
    pub(super) unresolved: &'a UnresolvedPipRuntime,
    /// An update apply's recorded exact arch, resolved against the targets the
    /// source publishes exactly as a resolved plan does.
    pub(super) device_target_override: Option<&'a str>,
    /// The `--prefix` the user named. It does not depend on the versions, so it is
    /// the plan's `target`; without one the runtime key names the install folder
    /// and the `target` is not resolved.
    pub(super) prefix: Option<&'a Path>,
    /// The host's ROCm when it is newer than the version the plan is for.
    pub(super) host_version_newer: Option<&'a str>,
    pub(super) python: &'a PlannedPython,
    pub(super) wheel_compatibility: &'a WheelCompatibility,
}

/// The whole plan for a preview of the ROCm 10 layout that stopped before
/// `uv pip compile`: what was chosen without `uv`, written by the same writer as
/// every other plan, with the lines made from the unchosen versions marked.
pub(super) fn unresolved_wheel_preview(preview: &UnresolvedPreview<'_>) -> String {
    let UnresolvedPreview {
        channel,
        version_selector,
        unresolved,
        device_target_override,
        prefix,
        host_version_newer,
        python,
        wheel_compatibility,
    } = *preview;
    let device_target = AggregateDeviceTarget::prefer_recorded(
        &unresolved.device_target,
        device_target_override,
        &unresolved.family,
        &unresolved.published_device_targets,
    );
    let explicit_install_root = prefix.map(rocm_core::resolve_path_through_symlinks);
    let no_wheel_warning = no_wheel_warning_for(
        unresolved.newest_repo_version.as_deref(),
        &unresolved.rocm_version,
    );
    let plan = WheelPlan {
        channel,
        version_selector,
        index_url: &unresolved.index_url,
        layout_generation: unresolved.layout.generation(),
        latest_version: &unresolved.rocm_version,
        family: &unresolved.family,
        family_source: &unresolved.family_source,
        device_target: &device_target,
        host_version_newer,
        python,
        wheel_compatibility,
        version_resolution_specs: Some(&unresolved.version_resolution_specs),
        torchaudio_pin: Some(&unresolved.torchaudio_pin),
        no_wheel_warning: no_wheel_warning.as_deref(),
        stack: PlanStack::NeedsUv {
            explicit_install_root: explicit_install_root.as_deref(),
        },
    };
    let mut output = String::new();
    write_wheel_plan(&mut output, &plan);
    write_dry_run_tail(&mut output, &plan);
    output
}
