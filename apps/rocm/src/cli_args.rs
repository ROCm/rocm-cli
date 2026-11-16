// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The clap value enums behind the `--framework`, `--format`, `--provider`,
//! `--mode`, `--device`, `--telemetry` and `--permissions` style arguments, and
//! the conversions from them to the library types they stand for.
//!
//! Mechanically relocated from `cli.rs` with no behavior change: attributes,
//! help text, aliases and declaration order are unchanged. `cli.rs` re-exports
//! each type, so every module keeps reaching them as `crate::cli::Provider` and
//! so on.

use clap::ValueEnum;
use rocm_core::{
    PERMISSIONS_MODE_ASK, PERMISSIONS_MODE_FULL_ACCESS, TELEMETRY_MODE_LOCAL, TELEMETRY_MODE_OFF,
    WatcherMode,
};

/// Which framework `rocm examine` should probe.
///
/// The CLI-facing mirror of [`rocm_core::FrameworkProbe`], which cannot derive
/// `ValueEnum` itself without pulling clap into `rocm-core`. Every variant of
/// that enum existed and was reachable in the library, but no caller ever passed
/// anything but the default, so the choice could not be made from the command
/// line at all.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum FrameworkArg {
    /// Try PyTorch first, then llama.cpp (the default).
    Auto,
    /// Probe PyTorch only.
    Pytorch,
    /// Probe llama.cpp only.
    #[value(name = "llama-cpp", alias = "llamacpp")]
    LlamaCpp,
    /// Probe no framework at all - fastest, and enough for GPU and driver questions.
    Skip,
}

impl From<FrameworkArg> for rocm_core::FrameworkProbe {
    fn from(value: FrameworkArg) -> Self {
        match value {
            FrameworkArg::Auto => Self::Auto,
            FrameworkArg::Pytorch => Self::PyTorch,
            FrameworkArg::LlamaCpp => Self::LlamaCpp,
            FrameworkArg::Skip => Self::Skip,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum InstallFormat {
    /// Python wheel packages (recommended; smaller, pip-installable).
    Wheel,
    /// Self-contained tarball archive of the ROCm SDK.
    Tarball,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum Provider {
    /// A local OpenAI-compatible server started via `rocm serve` (no API key).
    Local,
    /// Anthropic Claude API (requires an API key).
    Anthropic,
    /// OpenAI API or OpenAI-compatible endpoint (requires an API key).
    Openai,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum WatcherModeArg {
    /// Only report findings; never take action.
    Observe,
    /// Report findings and propose changes for you to approve.
    Propose,
    /// Apply changes automatically within a contained, reversible scope.
    Contained,
}

impl From<WatcherModeArg> for WatcherMode {
    fn from(value: WatcherModeArg) -> Self {
        match value {
            WatcherModeArg::Observe => Self::Observe,
            WatcherModeArg::Propose => Self::Propose,
            WatcherModeArg::Contained => Self::Contained,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum TelemetryModeArg {
    /// Collect GPU telemetry on this machine only; nothing leaves the device.
    Local,
    /// Disable telemetry collection entirely.
    Off,
}

/// User-facing `rocm serve --device` policy.
///
/// Modeled as a clap `ValueEnum` so invalid input is rejected by clap's usage
/// validation (exit code 2, with the valid choices listed) rather than a generic
/// application error (exit code 1) — matching how every other enum-style CLI
/// argument behaves. The historical aliases (`auto`/`gpu` for `gpu_required`,
/// `cpu` for `cpu_only`) stay accepted for backward compatibility but are hidden
/// from the advertised list.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "snake_case")]
pub(crate) enum DevicePolicyArg {
    /// Require a ROCm GPU; fail if none is usable (the default).
    #[value(alias = "auto", alias = "gpu")]
    GpuRequired,
    /// Accepted for compatibility; behaves identically to `gpu_required`
    /// (rocm serve has no CPU fallback path).
    GpuPreferred,
    /// Accepted but always rejected: rocm serve has no CPU fallback path, so
    /// this exits with an error. Hidden from `--help`, shell completions and the
    /// invalid-value suggestion list so it is never advertised as usable, while
    /// still parsing so the deliberate rejection message is preserved.
    #[value(alias = "cpu", hide = true)]
    CpuOnly,
}

impl DevicePolicyArg {
    /// Canonical policy string understood by [`crate::gpu_policy::parse_device_policy`]. Routing the
    /// parsed variant back through the string keeps `parse_device_policy` the
    /// single choke point that enforces the `cpu_only` rejection across every
    /// caller, rather than adding a second conversion path that could bypass it.
    pub(crate) const fn as_policy_str(self) -> &'static str {
        match self {
            Self::GpuRequired => "gpu_required",
            Self::GpuPreferred => "gpu_preferred",
            Self::CpuOnly => "cpu_only",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "snake_case")]
pub(crate) enum SandboxToolArg {
    ListServers,
    RestartServer,
    StopServer,
}

impl SandboxToolArg {
    pub(crate) const fn as_cli_value(self) -> &'static str {
        match self {
            Self::ListServers => "list_servers",
            Self::RestartServer => "restart_server",
            Self::StopServer => "stop_server",
        }
    }
}

impl TelemetryModeArg {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Local => TELEMETRY_MODE_LOCAL,
            Self::Off => TELEMETRY_MODE_OFF,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum PermissionsModeArg {
    FullAccess,
    Ask,
}

impl PermissionsModeArg {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::FullAccess => PERMISSIONS_MODE_FULL_ACCESS,
            Self::Ask => PERMISSIONS_MODE_ASK,
        }
    }
}
