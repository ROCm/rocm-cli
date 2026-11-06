// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The `sdk install` plan a wheel install prints, key by key.
//!
//! A preview prints it and stops; a real install prints it as the head of its
//! own output. [`write_wheel_plan`] writes the lines both share, and
//! [`write_dry_run_tail`] the lines only a preview adds. Moved out of
//! `install_wheel_runtime` so that one writer serves every plan it prints.

use super::python::{PlannedPython, WheelCompatibility};
use super::resolve::{AggregateDeviceTarget, TorchaudioPin};
use super::source::{RuntimeVersionSelector, TheRockChannel};
use super::{
    render_canonical_provenance, runtime_version_display, venv_python_path, wheel_host_version_note,
};
use crate::shell_quote::shell_quote;
use rocm_core::{uv_pip_install_base, uv_venv_args};
use std::fmt::Write as _;
use std::path::Path;

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
    pub(super) compatibility_key: &'a str,
    pub(super) install_root: &'a Path,
    pub(super) runtime_key: &'a str,
    pub(super) python: &'a PlannedPython,
    pub(super) wheel_compatibility: &'a WheelCompatibility,
    /// The requirement lines version resolution asked `uv` about. `None` on the
    /// canonical layout, which resolves without `uv`.
    pub(super) version_resolution_specs: Option<&'a [String]>,
    pub(super) torchaudio_pin: Option<&'a TorchaudioPin>,
    pub(super) package_specs: &'a [String],
    pub(super) no_wheel_warning: Option<&'a str>,
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
        runtime_version_display(plan.compatibility_key)
    );
    let _ = writeln!(output, "  target: {}", plan.install_root.display());
    let _ = writeln!(output, "  runtime_key: {}", plan.runtime_key);
    let _ = writeln!(
        output,
        "  python_launcher: {}",
        plan.python.launcher_display()
    );
    let _ = writeln!(output, "  python_source: {}", plan.python.source().as_str());
    if let Some(note) = plan.python.note() {
        let _ = writeln!(output, "  python_note: {note}");
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
    let _ = writeln!(output, "  package_specs: {}", plan.package_specs.join(" "));
    let _ = writeln!(
        output,
        "  package_policy: resolve the pinned target-complete rocm, torch, torchvision, and torchaudio plan from published package metadata, then install it in one uv transaction"
    );
    if let Some(warning) = plan.no_wheel_warning {
        let _ = writeln!(output, "  warning: {warning}");
    }
}

/// The lines a preview adds after [`write_wheel_plan`]: the commands a real install
/// would run, and where it would record the runtime.
pub(super) fn write_dry_run_tail(output: &mut String, plan: &WheelPlan<'_>, manifest_path: &Path) {
    let env_python = venv_python_path(plan.install_root);
    let mut install_args = uv_pip_install_base(&env_python);
    install_args.extend(["--index-url".to_owned(), plan.index_url.to_owned()]);
    if matches!(plan.channel, TheRockChannel::Nightly) {
        install_args.extend(["--prerelease".to_owned(), "allow".to_owned()]);
    }
    install_args.extend(plan.package_specs.iter().cloned());
    let venv_args = uv_venv_args(&plan.python.venv_python_arg(), plan.install_root);
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
}
