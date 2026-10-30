// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! SDK-install finalization: settling an engine onto a runtime after an SDK
//! install, torch alignment, engine-dependency checks, and device checks.
//!
//! Mechanically relocated from `main.rs` with no behavior change. Called from
//! `main.rs`'s `install()` (the SDK-install path), from `engines_cmd.rs`
//! (engine install) and, for `real_os_release`, from `driver_install.rs`. It calls into the sibling `runtimes_cmd.rs` for runtime
//! activation; `runtimes_cmd.rs` reaches back only from its own test module.
//! `SdkInstallConsents` stays
//! `pub(crate)` and is read from `install()`; several torch/dependency/device
//! helpers (`ensure_openmpi_for_vllm`, `ensure_libatomic_for_torch`,
//! `ensure_libnuma_for_torch`, `settle_engine_install`) are also `pub(crate)`
//! and read from `engines_cmd.rs`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};

use anyhow::{Context, Result, bail};
use rocm_core::{
    AppPaths, ExamineSummary, RocmCliConfig, preferred_serve_engine_for_host_gpu_summary,
};
use rocm_engine_protocol::{EngineMethod, InstallRequest, InstallResponse};

use crate::engine_ipc::engine_request_with_env_root;
use crate::engines_cmd::{
    engine_manages_own_runtime, env_root_for_engine_install, runtime_key_for_python,
    runtime_manifest_for_selector,
};
use crate::runtimes_cmd::{
    activate_runtime, newest_installed_runtime_manifest, recover_setup_runtime_registration,
};
use crate::therock;
use crate::{ROLLBACK_RECOVERY_HINT, record_cli_audit_event, runtime_services, sanitize_log_value};

#[derive(Debug, Clone)]
pub(crate) struct SdkInstallFinalization {
    pub(crate) runtime_key: String,
    install_root: PathBuf,
    family: String,
    pub(crate) previous_runtime_key: Option<String>,
    services: runtime_services::RuntimeServiceReconciliation,
}

pub(crate) fn print_sdk_install_success(finalized: &SdkInstallFinalization) {
    print!("{}", render_sdk_install_success(finalized));
}

fn preferred_engine_for_sdk_family(family: &str) -> Option<&'static str> {
    let summary = rocm_core::HostGpuSummary {
        therock_family: Some(family.to_owned()),
        ..rocm_core::HostGpuSummary::default()
    };
    preferred_serve_engine_for_host_gpu_summary(&summary)
}

/// The two unrelated consents `rocm install sdk` can be given.
///
/// They are separate because they authorize different things and are answerable
/// in different places. Replacing the active default runtime is a decision, and
/// an argv can express it fully. Approving a system-package install means
/// approving `sudo`, which — unless the host is root or has passwordless sudo —
/// needs a human at a terminal to type a password.
///
/// `--yes` grants both, which is what a user typing it at a terminal means.
/// ROCm CLI's own non-interactive surfaces need only the first: they spawn
/// `rocm` with null stdin, so a sudo password prompt there can never be
/// answered, and treating their approval as covering it would run sudo they
/// cannot complete — and, for the vLLM/OpenMPI plan, abort the engine
/// auto-install that used to warn and continue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SdkInstallConsents {
    /// Approve replacing whatever runtime is currently the active default, and
    /// which flag granted it. The source is carried rather than flattened to a
    /// bool because the install log names it: crediting `--yes` on a surface
    /// that only ever passed the narrow flag would tell the reader that consent
    /// to run `sudo` had been given when it had not.
    pub(crate) replace_active_default: therock::SdkInstallConsent,
    /// Approve installing required system packages (OpenMPI, libatomic,
    /// libnuma) through the system package manager, which means `sudo`.
    pub(crate) system_packages: bool,
}

impl SdkInstallConsents {
    pub(crate) const fn resolve(yes: bool, approve_replacing_active_default: bool) -> Self {
        let replace_active_default = if yes {
            therock::SdkInstallConsent::Preapproved(therock::SdkInstallApprovalSource::AssumeYes)
        } else if approve_replacing_active_default {
            therock::SdkInstallConsent::Preapproved(
                therock::SdkInstallApprovalSource::ApproveReplacingActiveDefault,
            )
        } else {
            therock::SdkInstallConsent::Ask
        };
        Self {
            replace_active_default,
            system_packages: yes,
        }
    }
}

/// What to do with a distro-aware system-package install plan, given the caller's
/// approval and what this host lets us do without a password.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SystemPackageInstallAction {
    /// Print the commands (and the preflight checks) and continue without them.
    /// Nothing privileged is run, so nothing can block on a password prompt.
    PrintManualCommands,
    /// Run the plan. `run_system_package_install_plan` inherits stdin so an
    /// interactive `sudo` password prompt can be answered.
    RunPlan {
        /// The `approval:` line explaining why this was allowed to run.
        approval: &'static str,
        /// Whether a failed command is an error rather than a warning. Only an
        /// explicit approval escalates: the automatic (root/passwordless) path
        /// must never let a missing system package fail an unattended install.
        escalate_failure: bool,
    },
}

/// Decide between the two, given whether the *system-package* consent was granted
/// and whether the host can install without prompting.
///
/// Split out from the two callers below so the decision is testable without a
/// package manager, and so the "approved but no way to answer a password prompt"
/// case has one place to be reasoned about.
const fn system_package_install_action(
    approved: bool,
    can_autoinstall: bool,
) -> SystemPackageInstallAction {
    if !approved && !can_autoinstall {
        return SystemPackageInstallAction::PrintManualCommands;
    }
    SystemPackageInstallAction::RunPlan {
        approval: if approved {
            "granted by --yes"
        } else {
            "auto (root or passwordless sudo available)"
        },
        escalate_failure: approved,
    }
}

/// The real host's `/etc/os-release`, falling back to `/usr/lib/os-release`
/// when the former is absent (per `os-release(5)`).
///
/// Deliberately not routed through `rocm_core::host_path`: every caller builds
/// a package-manager plan (`apt`, `dnf`, `zypper`) that then runs against the
/// real machine, so a simulated host root must not choose its distro. The descriptive distro name `examine` reports comes from
/// `ExamineSummary::gather`'s `distro` field, which does follow a simulated
/// root.
pub(crate) fn real_os_release() -> rocm_core::hardware_root::OsRelease {
    rocm_core::hardware_root::read_os_release(rocm_core::hardware_root::identity_path_for_str)
}

/// Run a command given as an argv vector directly, without going through a shell.
///
/// Used for [`run_system_package_install_plan`], whose commands are modeled as
/// argv vectors so no shell quoting or `sudo`-prefix string handling is needed.
fn run_argv_with_stdin(argv: &[String], stdin: Stdio) -> Result<()> {
    let (program, args) = argv
        .split_first()
        .context("install command has no program to run")?;
    let status = ProcessCommand::new(program)
        .args(args)
        .stdin(stdin)
        .status()
        .with_context(|| format!("failed to launch `{}`", argv.join(" ")))?;
    if !status.success() {
        bail!("`{}` exited with {status}", argv.join(" "));
    }
    Ok(())
}

/// Ensure the OpenMPI runtime that vLLM requires is present before the vLLM wheel
/// is installed. On Linux/WSL, when OpenMPI is missing, this installs it through
/// the system package manager.
///
/// The privileged install runs automatically when it can be performed without an
/// interactive prompt (the process is root, or passwordless `sudo` is available),
/// or when `approved` (the `--yes` flag) is set. Otherwise the distro-aware plan
/// is printed and the caller continues without it.
///
/// Returns `Ok(())` when OpenMPI is present, was installed, or could not be
/// installed automatically without explicit approval (warn-and-continue). When
/// the user explicitly approved the install with `--yes` and it still fails, the
/// error is propagated so the caller does not silently proceed past a failure the
/// user asked to perform.
pub(crate) fn ensure_openmpi_for_vllm(approved: bool) -> Result<()> {
    if cfg!(windows) {
        return Ok(());
    }
    let status = rocm_core::openmpi::detect_openmpi();
    if status.present {
        return Ok(());
    }

    let plan = rocm_core::openmpi::plan_for_os_release_read(
        &real_os_release(),
        rocm_core::openmpi::build_openmpi_install_plan,
    );

    println!("openmpi setup");
    println!(
        "  reason: vLLM requires the OpenMPI runtime (libmpi.so / mpirun), which was not found"
    );
    if let Some(manager) = plan.package_manager.as_deref() {
        println!("  package_manager: {manager}");
    }
    println!("  detail: {}", plan.reason);

    if !plan.supported {
        eprintln!(
            "warning: could not determine how to install OpenMPI automatically; install it manually so vLLM can load libmpi.so"
        );
        return Ok(());
    }

    println!("  commands:");
    for command in &plan.commands {
        println!("    {command}");
    }

    let can_autoinstall = rocm_core::openmpi::can_autoinstall();
    let SystemPackageInstallAction::RunPlan {
        approval,
        escalate_failure,
    } = system_package_install_action(approved, can_autoinstall)
    else {
        for check in &plan.preflight_checks {
            println!("  preflight: {check}");
        }
        eprintln!("warning: OpenMPI is required by vLLM but was not installed automatically");
        eprintln!(
            "warning: passwordless sudo is unavailable; run the commands above manually, or rerun with --yes to approve an interactive sudo prompt"
        );
        return Ok(());
    };

    println!("  approval: {approval}");
    match run_system_package_install_plan(&plan) {
        Ok(()) => {
            if rocm_core::openmpi::detect_openmpi().present {
                println!("  status: installed");
            } else {
                eprintln!(
                    "warning: OpenMPI install commands completed but the runtime (libmpi.so / mpirun) was still not found; verify the package manager output above"
                );
            }
            Ok(())
        }
        Err(error) => {
            // An explicit `--yes` is a deliberate request to perform the install;
            // surface the failure so the vLLM install does not silently proceed
            // past something the user asked for. The auto (unapproved) path keeps
            // the warn-and-continue behavior so a missing OpenMPI never blocks an
            // otherwise-unattended install.
            //
            // "Surface" is the exact claim, and it is not the same as failing the
            // command: this error propagates out of the engine auto-install, and
            // `finish_sdk_install` routes it through
            // `engine_auto_install_failure_is_fatal`, which matches only
            // `UnusableRuntimeAfterInstall`. So `rocm install sdk` still prints
            // the failure and exits 0. Said here because the downgrade happens
            // far away and reads as a non-zero exit from this site alone.
            if escalate_failure {
                return Err(error.context(
                    "OpenMPI install approved with --yes failed; rerun the commands above manually or retry without --yes to continue without OpenMPI",
                ));
            }
            eprintln!("warning: OpenMPI install failed: {error}");
            eprintln!(
                "warning: continuing vLLM install; run the commands above manually so vLLM can load libmpi.so"
            );
            Ok(())
        }
    }
}

/// Static description of a PyTorch runtime library dependency that may need to
/// be installed from the system package manager. Drives [`ensure_torch_runtime_dep`]
/// so libatomic/libnuma (and any future additions) share one control flow.
struct TorchRuntimeDep {
    /// Short label used in the setup header and warnings (e.g. `"libatomic"`).
    name: &'static str,
    /// Runtime soname referenced in status messages (e.g. `"libatomic.so.1"`).
    soname: &'static str,
    /// `reason:` line explaining why the dependency is required.
    reason: &'static str,
    /// Detection probe: returns whether the dependency is already loadable.
    present: fn() -> bool,
    /// Builds the distro-aware install plan from the parsed os-release fields.
    build_plan: rocm_core::openmpi::InstallPlanBuilder,
}

/// Ensure the `libatomic` runtime that PyTorch's ROCm wheel links against is
/// present. The SDK runtime wheel bundles PyTorch, and vLLM uses it too, so this
/// is invoked both after `rocm install sdk` and during `rocm engines install
/// vllm`. On Linux/WSL, when `libatomic.so.1` is missing it installs it through
/// the system package manager (automatically when no interactive prompt is
/// needed or when `approved`, otherwise it prints the distro-aware plan). Never
/// blocks or fails the install (warn-and-continue). It is a no-op when
/// libatomic is already present.
pub(crate) fn ensure_libatomic_for_torch(approved: bool) {
    ensure_torch_runtime_dep(
        approved,
        &TorchRuntimeDep {
            name: "libatomic",
            soname: "libatomic.so.1",
            reason: "PyTorch's ROCm wheel requires the libatomic runtime (libatomic.so.1), which was not found",
            present: rocm_core::openmpi::libatomic_present,
            build_plan: rocm_core::openmpi::build_libatomic_install_plan,
        },
    );
}

/// Ensure the real `libnuma` (numactl) runtime that PyTorch's ROCm wheel binds
/// is present. Like [`ensure_libatomic_for_torch`], this is invoked after
/// `rocm install sdk` and during `rocm engines install vllm`. PyTorch's
/// `libc10.so` binds `libnuma.so.1`'s `libnuma_1.2` symbols; the ROCm SDK only
/// bundles numa under a renamed soname with rewritten versions that cannot
/// satisfy it, so the upstream numactl runtime must be installed from the system
/// package manager. On Linux/WSL, when `libnuma.so.1` is missing it installs it
/// automatically when no interactive prompt is needed or when `approved`,
/// otherwise it prints the distro-aware plan. Never blocks or fails the install
/// (warn-and-continue). No-op when libnuma is already present.
pub(crate) fn ensure_libnuma_for_torch(approved: bool) {
    ensure_torch_runtime_dep(
        approved,
        &TorchRuntimeDep {
            name: "libnuma",
            soname: "libnuma.so.1",
            reason: "PyTorch's ROCm wheel requires the system numactl runtime (libnuma.so.1 with libnuma_1.2), which was not found",
            present: rocm_core::openmpi::libnuma_present,
            build_plan: rocm_core::openmpi::build_libnuma_install_plan,
        },
    );
}

/// Whether an E2E scenario has asked to skip the torch runtime dependency
/// checks entirely. These checks run a real system package-manager install
/// (`apt-get` or equivalent) whenever a dependency happens to be missing on
/// the host, which is slow, network-dependent, and mutates host state — none
/// of which a PTY scenario testing an unrelated concern (e.g. the download
/// spinner) should depend on. Only active under `e2e-test-hooks`; production
/// builds always run the real check.
#[cfg(feature = "e2e-test-hooks")]
fn torch_runtime_dep_checks_disabled() -> bool {
    std::env::var_os("ROCM_CLI_DISABLE_TORCH_RUNTIME_DEP_CHECKS").is_some()
}

#[cfg(not(feature = "e2e-test-hooks"))]
const fn torch_runtime_dep_checks_disabled() -> bool {
    false
}

/// Shared control flow behind [`ensure_libatomic_for_torch`] and
/// [`ensure_libnuma_for_torch`]: detect the dependency, print the distro-aware
/// plan, and (when approved or auto-installable) run it via
/// [`run_system_package_install_plan`]. Always warn-and-continue; never fails the
/// caller. No-op on Windows or when the dependency is already present.
fn ensure_torch_runtime_dep(approved: bool, dep: &TorchRuntimeDep) {
    if cfg!(windows) {
        return;
    }
    if torch_runtime_dep_checks_disabled() {
        return;
    }
    if (dep.present)() {
        return;
    }

    let plan = rocm_core::openmpi::plan_for_os_release_read(&real_os_release(), dep.build_plan);

    println!("{} setup", dep.name);
    println!("  reason: {}", dep.reason);
    if let Some(manager) = plan.package_manager.as_deref() {
        println!("  package_manager: {manager}");
    }
    println!("  detail: {}", plan.reason);

    if !plan.supported {
        // The distro is unknown, or its release file is unreadable (the detail
        // above names the line), or the dependency ships with the base
        // toolchain; nothing actionable to auto-install.
        return;
    }

    println!("  commands:");
    for command in &plan.commands {
        println!("    {command}");
    }

    let can_autoinstall = rocm_core::openmpi::can_autoinstall();
    // `escalate_failure` is deliberately ignored here: a missing libatomic/libnuma
    // only warns, whatever approved the attempt.
    let SystemPackageInstallAction::RunPlan { approval, .. } =
        system_package_install_action(approved, can_autoinstall)
    else {
        for check in &plan.preflight_checks {
            println!("  preflight: {check}");
        }
        eprintln!(
            "warning: {} is required by PyTorch but was not installed automatically",
            dep.name
        );
        eprintln!(
            "warning: passwordless sudo is unavailable; run the commands above manually, or rerun with --yes to approve an interactive sudo prompt"
        );
        return;
    };

    println!("  approval: {approval}");
    match run_system_package_install_plan(&plan) {
        Ok(()) => {
            if (dep.present)() {
                println!("  status: installed");
            } else {
                eprintln!(
                    "warning: {} install commands completed but {} was still not found; verify the package manager output above",
                    dep.name, dep.soname
                );
            }
        }
        Err(error) => {
            eprintln!("warning: {} install failed: {error}", dep.name);
            eprintln!(
                "warning: continuing; run the commands above manually so PyTorch can load {}",
                dep.soname
            );
        }
    }
}

/// Refuse an `apt-get install` step that would remove ROCm/AMDGPU packages.
///
/// `apt-get install -y` assumes yes for *removals* as well as installs, so a
/// dependency solution that evicts the ROCm stack would otherwise be applied
/// unattended (this runs automatically under `rocm install sdk` whenever root or
/// passwordless sudo is available). Simulating first turns that silent breakage
/// into an actionable error naming every package at risk.
///
/// Only apt is gated: the dnf/zypper/pacman plans install additive runtime
/// packages and have no equivalent assume-yes removal path. A simulation that
/// cannot be run (apt missing, transient failure) is not treated as a refusal —
/// the real command reports the failure with better context.
fn ensure_apt_install_preserves_rocm(argv: &[String]) -> Result<()> {
    let Some(simulate_argv) = rocm_core::openmpi::apt_simulate_argv(argv) else {
        return Ok(());
    };
    let (program, args) = simulate_argv
        .split_first()
        .context("apt simulation has no program to run")?;
    let Ok(output) = ProcessCommand::new(program)
        .args(args)
        .stdin(Stdio::null())
        .output()
    else {
        return Ok(());
    };
    if !output.status.success() {
        return Ok(());
    }

    let removals =
        rocm_core::openmpi::parse_apt_simulate_removals(&String::from_utf8_lossy(&output.stdout));
    let protected: Vec<&String> = removals
        .iter()
        .filter(|package| rocm_core::openmpi::is_protected_rocm_package(package))
        .collect();
    if protected.is_empty() {
        return Ok(());
    }

    let rendered = |packages: &[&String]| {
        packages
            .iter()
            .map(|package| package.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let others: Vec<&String> = removals
        .iter()
        .filter(|package| !rocm_core::openmpi::is_protected_rocm_package(package))
        .collect();
    let other_detail = if others.is_empty() {
        String::new()
    } else {
        format!("\n  other packages it would remove: {}", rendered(&others))
    };
    bail!(
        "refusing to run `{}`: it would remove ROCm packages and break this installation\n  ROCm packages it would remove: {}{}\n  action: install the dependency manually after resolving the conflict (for example with a package version that does not evict ROCm), then rerun",
        argv.join(" "),
        rendered(&protected),
        other_detail
    );
}

fn run_system_package_install_plan(
    plan: &rocm_core::openmpi::SystemPackageInstallPlan,
) -> Result<()> {
    let root = rocm_core::openmpi::running_as_root();
    for command in &plan.commands {
        // `resolved_argv` prepends `sudo` only when the command needs root and we
        // are not already root (where `sudo` may be absent); the argv runs
        // directly without a shell.
        let argv = command.resolved_argv(root);
        // Never let an automatic dependency install evict the ROCm stack.
        ensure_apt_install_preserves_rocm(&argv)?;
        // Inherit stdin so an interactive `sudo` password prompt (the case the
        // `--yes` approval exists for) can be answered. When already root or
        // passwordless sudo is configured, sudo does not prompt and the inherited
        // stdin is simply unused.
        run_argv_with_stdin(&argv, Stdio::inherit())
            .with_context(|| format!("system package install command failed: {command}"))?;
    }
    Ok(())
}

pub(crate) fn maybe_auto_install_sdk_preferred_engine(
    paths: &AppPaths,
    finalized: &SdkInstallFinalization,
    approved: bool,
) -> Result<()> {
    let Some(engine) = preferred_engine_for_sdk_family(&finalized.family) else {
        return Ok(());
    };

    println!("engine auto-install");
    println!(
        "  reason: detected ROCm GPU family prefers {engine} ({})",
        finalized.family
    );
    println!("  engine: {engine}");
    println!("  runtime_id: {}", finalized.runtime_key);

    ensure_openmpi_for_vllm(approved)?;

    let mut config = RocmCliConfig::load(paths)?;
    let env_root = env_root_for_engine_install(paths, &config, engine, &finalized.runtime_key)?;
    let response = match engine_request_with_env_root::<_, InstallResponse>(
        Some(paths),
        engine,
        EngineMethod::Install,
        &InstallRequest {
            runtime_id: finalized.runtime_key.clone(),
            python_version: None,
            reinstall: false,
            env_root: env_root.clone(),
        },
        env_root.as_deref(),
    ) {
        Ok(response) => response,
        Err(error) => {
            // A failed engine install is the case most likely to leave the SDK's torch
            // sitting over the engine's pin, because the install is what would have
            // restored it. Reporting the check only on success would print nothing here
            // — and `install sdk` still exits 0 — which is exactly the silent broken
            // runtime this block exists to prevent.
            report_engine_dependency_check(
                paths,
                engine,
                runtime_python_for_key(paths, &finalized.runtime_key).as_deref(),
                &finalized.runtime_key,
                // The engine install failed, so no alignment ran and any
                // divergence found here is a real one.
                None,
            );
            return Err(error);
        }
    };
    println!("  reinstall: false");
    println!("  env_id: {}", response.env_id);
    println!("  env_path: {}", response.env_path);
    for warning in &response.warnings {
        println!("  warning: {warning}");
    }

    if response.managed_env == Some(false) {
        println!("  note: external runtime");
    } else {
        let engine_config = config.engine_config_mut(engine);
        engine_config.last_installed_runtime_id = Some(finalized.runtime_key.clone());
        engine_config.last_installed_env_id = Some(response.env_id.clone());
        if engine_config.preferred_runtime_id.is_none() && engine_config.preferred_env_id.is_none()
        {
            engine_config.preferred_env_id = Some(response.env_id.clone());
        }
        config.save(paths)?;

        settle_engine_install(paths, engine, &finalized.runtime_key, &response)?;
    }

    record_cli_audit_event(
        paths,
        "engine",
        "engine_auto_install",
        "info",
        format!(
            "auto-installed engine={} runtime_id={} env_id={} family={}",
            engine, finalized.runtime_key, response.env_id, finalized.family
        ),
        None,
    );
    Ok(())
}

/// Whether an installed engine's declared requirements are met in the environment it
/// shares with the ROCm SDK.
#[derive(Debug, Clone, PartialEq, Eq)]
enum EngineDependencyCheck {
    /// Every requirement the engine declares is satisfied.
    Satisfied,
    /// The engine declares requirements the environment does not meet, one line each,
    /// as the resolver reported them.
    ///
    /// `expected` carries any divergence the install made on purpose that shares the
    /// run with a genuine violation. It is reported separately rather than folded in
    /// because the remedy differs: reinstalling the engine repairs `violations` and
    /// destroys `expected`, so naming that remedy is only safe while `expected` is
    /// empty.
    Violated {
        violations: Vec<String>,
        expected: Vec<String>,
    },
    /// The only unmet requirements are ones the install deliberately diverged from.
    ///
    /// Torch alignment leaves the runtime holding the SDK's build of the release
    /// the engine pins, which the engine's exact-pin metadata cannot express. The
    /// distinction matters because the remedy for a real violation — reinstalling
    /// the engine — is precisely what would undo the alignment and restore a
    /// runtime that cannot open a device.
    ExpectedDivergence(Vec<String>),
    /// The check itself could not run (no usable `uv`, unreadable environment).
    NotVerified(String),
}

/// Print the dependency check for `engine` and record it in the CLI audit log.
///
/// A surviving violation does not fail the install: the SDK itself installed correctly,
/// and abandoning a multi-gigabyte install would cost the user more than the warning and
/// the one-line remedy it names.
fn report_engine_dependency_check(
    paths: &AppPaths,
    engine: &str,
    python: Option<&Path>,
    runtime_key: &str,
    realigned_package: Option<&str>,
) {
    let outcome = engine_dependency_check(paths, engine, python, realigned_package);
    print!("{}", render_engine_dependency_check(engine, &outcome));
    let (level, message) = match &outcome {
        EngineDependencyCheck::Satisfied => (
            "info",
            format!("engine={engine} runtime_id={runtime_key} dependency_check=satisfied"),
        ),
        EngineDependencyCheck::Violated {
            violations,
            expected,
        } => (
            "error",
            format!(
                "engine={engine} runtime_id={runtime_key} dependency_check=violated: {}{}",
                violations.join("; "),
                if expected.is_empty() {
                    String::new()
                } else {
                    format!(" (expected_divergence: {})", expected.join("; "))
                }
            ),
        ),
        EngineDependencyCheck::ExpectedDivergence(details) => (
            "info",
            format!(
                "engine={engine} runtime_id={runtime_key} dependency_check=expected_divergence: {}",
                details.join("; ")
            ),
        ),
        EngineDependencyCheck::NotVerified(reason) => (
            "info",
            format!(
                "engine={engine} runtime_id={runtime_key} dependency_check=not_verified: {reason}"
            ),
        ),
    };
    record_cli_audit_event(
        paths,
        "engine",
        "engine_dependency_check",
        level,
        message,
        None,
    );
}

/// The Python interpreter of the managed runtime `runtime_key`, when its manifest names
/// one. Used to check an environment whose engine install did not get far enough to
/// report its own interpreter.
fn runtime_python_for_key(paths: &AppPaths, runtime_key: &str) -> Option<PathBuf> {
    let manifests = therock::load_runtime_manifests(paths).ok()?;
    let manifest = runtime_manifest_for_selector(&manifests, runtime_key)?;
    manifest.python_executable.as_deref().map(PathBuf::from)
}

/// The runtime's recorded ROCm library directories.
///
/// These are what a served process gets on its loader path, so a probe must use
/// them too or it will judge a healthy runtime unusable.
fn runtime_library_paths_for_key(paths: &AppPaths, runtime_key: &str) -> Vec<PathBuf> {
    let Ok(manifests) = therock::load_runtime_manifests(paths) else {
        return Vec::new();
    };
    runtime_manifest_for_selector(&manifests, runtime_key)
        .and_then(|manifest| manifest.rocm_sdk.as_ref())
        .map(|probe| probe.library_paths.clone())
        .unwrap_or_default()
}

/// The wheel index the runtime was installed from.
fn runtime_index_url_for_key(paths: &AppPaths, runtime_key: &str) -> Option<String> {
    let manifests = therock::load_runtime_manifests(paths).ok()?;
    let manifest = runtime_manifest_for_selector(&manifests, runtime_key)?;
    manifest.index_url.clone()
}

/// Which torch a runtime should hold once an engine has been installed into it.
///
/// Two installers write torch into the same environment. The SDK install writes
/// TheRock's build; the engine then writes the build from its own index, pinned
/// to an exact version. Letting either side win unconditionally is wrong, and
/// both failures have been observed in the field:
///
/// * If the engine's build always wins, the runtime can end up with a torch that
///   loads against the installed SDK and then enumerates no devices, so serving
///   fails with an unhelpful error long after the install reported success.
/// * If the SDK's build always wins, the runtime can end up on a torch that
///   enumerates a device and still has no kernel image for this exact target,
///   failing on the first tensor operation instead.
///
/// So compatibility is tested rather than assumed. A torch already executing a
/// GPU kernel with this SDK is kept exactly as it is — that decision is
/// `TorchRetention`, and it runs first. This alignment is the repair for
/// everything else, and what it installs is the SDK's *build* of the *release*
/// the engine pins: the release comes from the engine, which is built against it,
/// and the build comes from the SDK, which the libraries belong to.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TorchAlignment {
    /// The runtime already holds the SDK build of the engine's torch release.
    AlreadyAligned { version: String },
    /// The engine's build was replaced with the SDK build of the same release.
    Realigned { from: String, to: String },
    /// The SDK publishes no build of that release; the engine's own is kept.
    ///
    /// Claimed only when the resolver actually said so. Any other failure is an
    /// `InstallFailed`, because asserting "not published" over a network or disk
    /// error sends the reader hunting for a missing wheel that exists.
    Unavailable { wanted: String, kept: String },
    /// The realignment install failed for some other reason, carried verbatim.
    InstallFailed {
        wanted: String,
        kept: String,
        error: String,
    },
    /// The user opted out, and a replacement was due: their torch is kept.
    ///
    /// Distinct from every other outcome because nothing was attempted. Reading
    /// it as `NotApplicable` would say the rule had no opinion, when it had one
    /// and was told not to act on it, and that difference is what the reader
    /// needs: `wanted` is the build that was not installed, `kept` the one still
    /// there. Only reachable when those two actually differ, so a runtime the
    /// opt-out changed nothing about is never described as one it spared.
    Disabled { wanted: String, kept: String },
    /// Nothing to decide — no exact pin, no engine metadata, or no SDK torch.
    NotApplicable(String),
}

/// What the alignment rule concludes, before any of it is acted on.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TorchAlignmentPlan {
    AlreadyAligned { version: String },
    Install { wanted: String, from: String },
    NotApplicable(String),
}

/// The rule itself, kept free of I/O so both field failures can be tested.
///
/// `wanted` is the engine's pinned *release* carrying the SDK's *build*.
fn plan_torch_alignment(
    sdk_build: Option<&str>,
    installed_torch: Option<&str>,
    engine_requirement: Option<&str>,
    engine: &str,
) -> TorchAlignmentPlan {
    let Some(sdk_build) = sdk_build else {
        return TorchAlignmentPlan::NotApplicable(
            "the runtime manifest does not identify the SDK's torch build".to_owned(),
        );
    };
    let Some(requirement) = engine_requirement else {
        return TorchAlignmentPlan::NotApplicable(format!("{engine} does not pin torch"));
    };
    let Some(pinned) = therock::requirement_pinned_version(requirement) else {
        return TorchAlignmentPlan::NotApplicable(format!(
            "{engine} does not pin torch to an exact version ({requirement})"
        ));
    };
    let wanted = format!("{}+{sdk_build}", therock::split_local_version(pinned).0);
    let installed = installed_torch.unwrap_or_default().to_owned();
    if installed == wanted {
        TorchAlignmentPlan::AlreadyAligned { version: wanted }
    } else {
        TorchAlignmentPlan::Install {
            wanted,
            from: installed,
        }
    }
}

/// Whether a failed install means the resolver could not find that version.
///
/// Deliberately narrow. Anything unmatched is reported as a plain failure with
/// its error, so an unrecognised message degrades to the honest answer rather
/// than to a confident wrong one.
fn install_error_reports_version_unavailable(error: &str) -> bool {
    let error = error.to_ascii_lowercase();
    error.contains("no solution found")
        || error.contains("were found for")
        || error.contains("not found in the package registry")
        || error.contains("has no version")
}

/// Whether the user has opted out of realigning torch.
///
/// The alignment rewrites a package the user may have installed deliberately, and
/// it runs on every path that installs an engine, so a hand-installed torch is
/// otherwise replaced again by the next `engines install`, `install sdk`, or
/// `engines shell`. The stack this resolves to — the SDK's build of the release
/// the engine pins — is not validated against the supported matrix, so "the SDK's
/// build does not work on this machine" is a case that can happen rather than a
/// hypothetical one, and it needs an exit that is not "stop using the CLI".
///
/// The engine reads the same variable through the same helper, so the two sides
/// cannot drift; [`rocm_core::torch_alignment_disabled`] carries why that matters.
/// Suppressing the correction does not suppress the diagnosis: a runtime that
/// opens no device or cannot run a kernel on one is still reported as such — and
/// still fails the install on a host where a GPU was found.
fn torch_alignment_disabled() -> bool {
    rocm_core::torch_alignment_disabled()
}

/// Install the SDK's build of the release the engine pins, when that is needed.
///
/// `torch` is the metadata probe the caller already took: the retention decision
/// that runs first needs the engine's pin too, and asking the runtime the same
/// question twice would be a second interpreter launch for an answer in hand.
///
/// The opt-out is read where the install would run, not at the top. That is the
/// only place it changes anything, and reading it there is what lets the result
/// name the replacement it declined to make: a runtime with no replacement due
/// reports what it is, rather than reporting an opt-out that skipped nothing.
fn align_runtime_torch(
    paths: &AppPaths,
    python: &Path,
    index_url: Option<&str>,
    sdk_build: Option<&str>,
    engine: &str,
    torch: Result<&therock::TorchAlignmentProbe, &anyhow::Error>,
) -> TorchAlignment {
    let probe = match torch {
        Ok(probe) => probe,
        Err(error) => return TorchAlignment::NotApplicable(error.to_string()),
    };
    let plan = plan_torch_alignment(
        sdk_build,
        probe.installed_torch.as_deref(),
        probe.engine_requires_torch.as_deref(),
        engine,
    );
    let (wanted, from) = match plan {
        TorchAlignmentPlan::AlreadyAligned { version } => {
            return TorchAlignment::AlreadyAligned { version };
        }
        TorchAlignmentPlan::NotApplicable(reason) => {
            return TorchAlignment::NotApplicable(reason);
        }
        TorchAlignmentPlan::Install { wanted, from } => (wanted, from),
    };
    // Read only now that a replacement is actually due, so the opt-out reports
    // the install it stopped rather than standing in for a runtime that needed
    // nothing. Everything after this point is the rewrite itself.
    if torch_alignment_disabled() {
        return TorchAlignment::Disabled { wanted, kept: from };
    }
    let Some(index_url) = index_url else {
        return TorchAlignment::NotApplicable(
            "the runtime manifest records no wheel index to install from".to_owned(),
        );
    };
    match therock::install_pinned_package(
        paths,
        python,
        index_url,
        "torch",
        &format!("torch=={wanted}"),
    ) {
        Ok(()) => TorchAlignment::Realigned { from, to: wanted },
        // The SDK index may not publish this release at all. That is a real
        // possibility, not an error to abort on: the engine's own build is left
        // in place and the device check that follows reports whether it works.
        Err(error) => {
            let error = format!("{error:#}");
            if install_error_reports_version_unavailable(&error) {
                TorchAlignment::Unavailable { wanted, kept: from }
            } else {
                TorchAlignment::InstallFailed {
                    wanted,
                    kept: from,
                    error,
                }
            }
        }
    }
}

/// Render one alignment outcome as the `torch_alignment:` check block.
///
/// `engine` names the engine whose pin decided the release, so the realigned line
/// reads `... the release vllm pins` rather than an anonymous "the engine". It
/// arrives from the engine selection and is sanitized like every other
/// interpolated value.
///
/// `before` is the verdict the runtime gave *prior* to the realignment. Only the
/// realigned outcome replaced anything, so only that arm prints it — and it must,
/// because the `device_check:` block further down reports the runtime as it is
/// now. Without the before verdict beside it, a realignment that ends in a
/// failing runtime does not say whether the alignment broke something that
/// worked or found something already broken, and answering that afterwards means
/// going to the machine to look. The value is already in hand at the call site;
/// dropping it only moves the cost onto whoever reads the output.
fn render_torch_alignment(
    outcome: &TorchAlignment,
    engine: &str,
    before: Option<&RuntimeDeviceCheck>,
) -> String {
    let mut output = String::new();
    match outcome {
        TorchAlignment::AlreadyAligned { version } => {
            let _ = writeln!(
                output,
                "  torch_alignment: already_aligned ({})",
                sanitize_log_value(version)
            );
        }
        TorchAlignment::Realigned { from, to } => {
            let _ = writeln!(output, "  torch_alignment: realigned");
            let _ = writeln!(
                output,
                "    {} -> {} (the SDK's build of the release {} pins)",
                sanitize_log_value(from),
                sanitize_log_value(to),
                sanitize_log_value(engine)
            );
            if let Some(before) = before {
                let _ = writeln!(
                    output,
                    "    before this replacement: {} (device_check below is after it)",
                    sanitize_log_value(&device_check_verdict(before))
                );
            }
        }
        TorchAlignment::Unavailable { wanted, kept } => {
            let _ = writeln!(output, "  torch_alignment: unavailable");
            let _ = writeln!(
                output,
                "    the SDK index publishes no {}; keeping {}",
                sanitize_log_value(wanted),
                sanitize_log_value(kept)
            );
        }
        TorchAlignment::InstallFailed {
            wanted,
            kept,
            error,
        } => {
            let _ = writeln!(output, "  torch_alignment: install_failed");
            let _ = writeln!(
                output,
                "    could not install {}: {}",
                sanitize_log_value(wanted),
                sanitize_log_value(error)
            );
            let _ = writeln!(output, "    keeping {}", sanitize_log_value(kept));
        }
        TorchAlignment::Disabled { wanted, kept } => {
            let _ = writeln!(output, "  torch_alignment: disabled");
            let _ = writeln!(
                output,
                "    ROCM_CLI_DISABLE_TORCH_ALIGNMENT is set; keeping {} rather than installing {}",
                sanitize_log_value(kept),
                sanitize_log_value(wanted)
            );
        }
        TorchAlignment::NotApplicable(reason) => {
            let _ = writeln!(
                output,
                "  torch_alignment: not_applicable ({})",
                sanitize_log_value(reason)
            );
        }
    }
    output
}

/// Repair the runtime's torch, print the `torch_alignment:` block, and record it.
///
/// Reached only when no torch in the runtime has proven it can run a GPU kernel.
/// The outcome is returned so the dependency check that follows can be read
/// against it: `deliberately_diverged_package` turns it into the one package
/// whose divergence is intended.
fn report_torch_alignment(
    paths: &AppPaths,
    engine: &str,
    python: &Path,
    runtime_key: &str,
    sdk_build: Option<&str>,
    torch: Result<&therock::TorchAlignmentProbe, &anyhow::Error>,
    before: &RuntimeDeviceCheck,
) -> TorchAlignment {
    let index_url = runtime_index_url_for_key(paths, runtime_key);
    let outcome = align_runtime_torch(
        paths,
        python,
        index_url.as_deref(),
        sdk_build,
        engine,
        torch,
    );
    print!("{}", render_torch_alignment(&outcome, engine, Some(before)));
    let (level, message) = match &outcome {
        TorchAlignment::AlreadyAligned { version } => (
            "info",
            format!(
                "engine={engine} runtime_id={runtime_key} torch_alignment=already_aligned version={version}"
            ),
        ),
        // The before verdict rides along in the audit line too: the log is read
        // long after the install, when the runtime on disk can no longer answer
        // what it was like beforehand.
        TorchAlignment::Realigned { from, to } => (
            "info",
            format!(
                "engine={engine} runtime_id={runtime_key} torch_alignment=realigned from={from} to={to} before={}",
                device_check_verdict(before)
            ),
        ),
        TorchAlignment::Unavailable { wanted, kept } => (
            "error",
            format!(
                "engine={engine} runtime_id={runtime_key} torch_alignment=unavailable wanted={wanted} kept={kept}"
            ),
        ),
        TorchAlignment::InstallFailed {
            wanted,
            kept,
            error,
        } => (
            "error",
            format!(
                "engine={engine} runtime_id={runtime_key} torch_alignment=install_failed wanted={wanted} kept={kept}: {error}"
            ),
        ),
        // The user's own decision, carried out as asked, so it is not an error.
        // The device check that follows is what says whether the torch they kept
        // works, and that verdict is recorded — and enforced — on its own.
        TorchAlignment::Disabled { wanted, kept } => (
            "info",
            format!(
                "engine={engine} runtime_id={runtime_key} torch_alignment=disabled kept={kept} wanted={wanted}"
            ),
        ),
        TorchAlignment::NotApplicable(reason) => (
            "info",
            format!(
                "engine={engine} runtime_id={runtime_key} torch_alignment=not_applicable: {reason}"
            ),
        ),
    };
    record_cli_audit_event(paths, "engine", "torch_alignment", level, message, None);
    outcome
}

/// The package the runtime now deliberately diverges on, if any, so the
/// dependency check can tell that divergence apart from a real violation.
///
/// Both `Realigned` and `AlreadyAligned` diverge from the engine's exact pin —
/// the second is a rerun over a runtime already put right, which is the normal
/// state on every refresh after the first.
///
/// `Disabled` counts for the same reason, from the other direction: the torch
/// that does not satisfy the pin is the one the user told us to leave alone, so
/// reporting it as a violation would answer their instruction with an error and
/// a `--reinstall` remedy that would undo it. It is only ever constructed over a
/// real mismatch, so this never suppresses a divergence that is not there.
///
/// `Unavailable` and `InstallFailed` stay excluded. Nothing was replaced there
/// either, but nothing was intended either — the repair was attempted and did
/// not happen — so whatever the dependency check finds is a real finding.
const fn deliberately_diverged_package(outcome: &TorchAlignment) -> Option<&'static str> {
    match outcome {
        TorchAlignment::Realigned { .. }
        | TorchAlignment::AlreadyAligned { .. }
        | TorchAlignment::Disabled { .. } => Some("torch"),
        TorchAlignment::Unavailable { .. }
        | TorchAlignment::InstallFailed { .. }
        | TorchAlignment::NotApplicable(_) => None,
    }
}

/// Whether the installed runtime can actually run work on a GPU.
///
/// The dependency check answers whether the engine's declared requirements are
/// satisfied. That is a question about metadata, and it is not the same question
/// as whether the environment works. Two distinct answers matter here, because a
/// torch can fail at either step: one built against a different ROCm version than
/// the installed SDK loads cleanly and then reports no devices, which vLLM turns
/// into `Failed to infer device type` at first serve; one built without a kernel
/// image for this target reports a device and then dies on the first tensor
/// operation. Only a torch that gets past both has been shown to work.
#[derive(Debug, Clone, PartialEq, Eq)]
enum RuntimeDeviceCheck {
    /// torch imported and reported at least one device.
    Usable {
        device_count: u32,
        torch_version: String,
    },
    /// torch imported but reported no devices — a torch built for another SDK.
    NoDevices {
        torch_version: String,
        hip_version: String,
    },
    /// torch found a device and then could not execute a kernel on it.
    ///
    /// Distinct from `NoDevices` because the remedy is different: the runtime is
    /// not looking at the wrong SDK, it is holding a build with no code for this
    /// GPU.
    KernelFailed {
        torch_version: String,
        error: String,
    },
    /// The question could not be answered — including when torch does not import.
    ///
    /// Never a reason to assume healthy, but on its own never fatal either: see
    /// `install_left_runtime_unusable`, which acts only on the verdicts where the
    /// runtime was asked and answered badly. This variant covers benign causes as
    /// well as real ones — a runtime whose Python could not be located, or a probe
    /// that could not launch — and failing a multi-gigabyte install because a
    /// probe did not run is worse than reporting what was and was not seen. The
    /// cost is that a runtime whose torch is present but unimportable is reported
    /// rather than failed.
    NotVerified(String),
}

fn runtime_device_check(python: Option<&Path>, library_paths: &[PathBuf]) -> RuntimeDeviceCheck {
    let Some(python) = python else {
        return RuntimeDeviceCheck::NotVerified(
            "the runtime's Python environment could not be located".to_owned(),
        );
    };
    match therock::probe_runtime_devices(python, library_paths) {
        Ok(probe) => classify_runtime_device_probe(probe),
        Err(error) => RuntimeDeviceCheck::NotVerified(error.to_string()),
    }
}

/// Read one probe as a verdict, kept free of I/O so every state can be tested.
fn classify_runtime_device_probe(probe: therock::RuntimeDeviceProbe) -> RuntimeDeviceCheck {
    if !probe.import_ok {
        return RuntimeDeviceCheck::NotVerified(
            probe
                .error
                .unwrap_or_else(|| "torch did not import".to_owned()),
        );
    }
    let torch_version = probe.torch_version.unwrap_or_else(|| "unknown".to_owned());
    // A kernel that would not launch is definitive, so it is read before the
    // count and never dropped: the probe only reaches that step after it has
    // already enumerated a device, and the count alone would read as healthy.
    if let Some(error) = probe.kernel_error {
        return RuntimeDeviceCheck::KernelFailed {
            torch_version,
            error,
        };
    }
    match probe.device_count {
        Some(0) => RuntimeDeviceCheck::NoDevices {
            torch_version,
            hip_version: probe.hip_version.unwrap_or_else(|| "unknown".to_owned()),
        },
        Some(device_count) => RuntimeDeviceCheck::Usable {
            device_count,
            torch_version,
        },
        // A `device_count()` that raises — HIP init failures are the live example —
        // reports no count and puts the real exception in `error`. That exception is
        // precisely the diagnostic this probe exists to capture, so prefer it over
        // the generic line, which is only right when the probe returned nothing at
        // all to explain itself.
        None => RuntimeDeviceCheck::NotVerified(
            probe
                .error
                .unwrap_or_else(|| "torch imported but did not report a device count".to_owned()),
        ),
    }
}

/// One device-check verdict on one line, for quoting inside another block.
///
/// The full block explains the verdict and names a consequence, which is right
/// where it is the answer and wrong where it is context for something else. This
/// keeps the part that identifies the verdict — the name and the torch it was
/// asked about — so a before/after pair reads as a pair rather than as two
/// competing diagnoses.
fn device_check_verdict(outcome: &RuntimeDeviceCheck) -> String {
    match outcome {
        RuntimeDeviceCheck::Usable {
            device_count,
            torch_version,
        } => format!("usable ({device_count} device(s), torch {torch_version})"),
        RuntimeDeviceCheck::NoDevices { torch_version, .. } => {
            format!("no_devices (torch {torch_version})")
        }
        RuntimeDeviceCheck::KernelFailed { torch_version, .. } => {
            format!("kernel_failed (torch {torch_version})")
        }
        RuntimeDeviceCheck::NotVerified(reason) => format!("not_verified ({reason})"),
    }
}

fn render_runtime_device_check(outcome: &RuntimeDeviceCheck) -> String {
    let mut output = String::new();
    match outcome {
        RuntimeDeviceCheck::Usable {
            device_count,
            torch_version,
        } => {
            let _ = writeln!(
                output,
                "  device_check: usable ({device_count} device(s), torch {})",
                sanitize_log_value(torch_version)
            );
        }
        RuntimeDeviceCheck::NotVerified(reason) => {
            let _ = writeln!(
                output,
                "  device_check: not_verified ({})",
                sanitize_log_value(reason)
            );
        }
        RuntimeDeviceCheck::NoDevices {
            torch_version,
            hip_version,
        } => {
            let _ = writeln!(output, "  device_check: no_devices");
            let _ = writeln!(
                output,
                "    torch {} (hip {}) imported but reports 0 devices",
                sanitize_log_value(torch_version),
                sanitize_log_value(hip_version)
            );
            // Deliberately no remedy: the reinstall that would satisfy the
            // engine's pin is what produces this state, so naming it here would
            // send people in a circle. Say what is wrong and let them choose.
            let _ = writeln!(
                output,
                "    serving will fail with `Failed to infer device type`; this torch is built \
                 for a different ROCm version than the installed SDK"
            );
        }
        RuntimeDeviceCheck::KernelFailed {
            torch_version,
            error,
        } => {
            let _ = writeln!(output, "  device_check: kernel_failed");
            let _ = writeln!(
                output,
                "    torch {} found a GPU but could not run a kernel on it: {}",
                sanitize_log_value(torch_version),
                sanitize_log_value(error)
            );
            let _ = writeln!(
                output,
                "    serving will fail on the first tensor operation; this torch has no kernel \
                 image for this GPU"
            );
        }
    }
    output
}

/// Print one device-check verdict and record it.
///
/// The verdict is taken as an argument because the settle path probes the runtime
/// before it decides what to do with it, and the block is printed in its usual
/// place afterwards rather than where the probe happened to run.
fn report_runtime_device_check(
    paths: &AppPaths,
    engine: &str,
    runtime_key: &str,
    outcome: RuntimeDeviceCheck,
) -> RuntimeDeviceCheck {
    print!("{}", render_runtime_device_check(&outcome));
    let (level, message) = match &outcome {
        RuntimeDeviceCheck::Usable { device_count, .. } => (
            "info",
            format!(
                "engine={engine} runtime_id={runtime_key} device_check=usable devices={device_count}"
            ),
        ),
        RuntimeDeviceCheck::NoDevices {
            torch_version,
            hip_version,
        } => (
            "error",
            format!(
                "engine={engine} runtime_id={runtime_key} device_check=no_devices torch={torch_version} hip={hip_version}"
            ),
        ),
        RuntimeDeviceCheck::KernelFailed {
            torch_version,
            error,
        } => (
            "error",
            format!(
                "engine={engine} runtime_id={runtime_key} device_check=kernel_failed torch={torch_version}: {error}"
            ),
        ),
        RuntimeDeviceCheck::NotVerified(reason) => (
            "info",
            format!("engine={engine} runtime_id={runtime_key} device_check=not_verified: {reason}"),
        ),
    };
    record_cli_audit_event(
        paths,
        "engine",
        "runtime_device_check",
        level,
        message,
        None,
    );
    outcome
}

/// The build identifier of the torch that belongs to this runtime's SDK.
///
/// Taken from the manifest, never from whatever torch happens to be installed.
/// The environment tells you what is there now, which after an engine install is
/// the engine's build — trusting that would let a runtime already holding the
/// wrong torch declare itself correct and never recover.
///
/// Older manifests predate the recorded value, so fall back to deriving it from
/// the SDK version, which is how TheRock names these builds.
fn sdk_torch_build_for_key(paths: &AppPaths, runtime_key: &str) -> Option<String> {
    let manifests = therock::load_runtime_manifests(paths).ok()?;
    let manifest = runtime_manifest_for_selector(&manifests, runtime_key)?;
    sdk_torch_build_from_manifest(manifest)
}

/// The build identifier one manifest names, with the fallback for older manifests.
///
/// Split out from the lookup so the decision can be tested without a registry on
/// disk: the lookup is a directory scan, but this is the part that has to be right
/// for a runtime already stuck on the engine's build to recover.
fn sdk_torch_build_from_manifest(manifest: &therock::InstalledRuntimeManifest) -> Option<String> {
    if let Some(recorded) = manifest.sdk_torch.as_deref()
        && let Some(build) = therock::split_local_version(recorded).1
    {
        return Some(build.to_owned());
    }
    let version = manifest
        .rocm_sdk
        .as_ref()
        .and_then(|probe| probe.rocm_sdk_version.clone())
        .unwrap_or_else(|| manifest.version.clone());
    (!version.trim().is_empty()).then(|| format!("rocm{version}"))
}

/// Which torch a runtime keeps once one of them has run a GPU kernel.
///
/// Exactly two builds are allowed to stand: the engine's own exact pin, and the
/// SDK's build of the release it pins. Both are states this tool produces on
/// purpose, and each is a fixed point — a rerun over either one keeps it and
/// installs nothing, which is what stops `--reinstall` from oscillating between
/// the two package sources. Anything else is repaired by `TorchAlignment`.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TorchRetention {
    /// The engine's exact pin ran a kernel. It satisfies the pin, so it diverges
    /// from nothing and the dependency check is read straight.
    EngineBuild { version: String },
    /// The SDK's build of the pinned release ran. The engine's pin stays
    /// deliberately unsatisfied, which the dependency check is told to expect.
    SdkBuild { version: String },
    /// Nothing here has been shown to work: align, then look again.
    Realign,
}

/// Decide retention from the torch that actually executed a kernel.
///
/// The version compared is the one the running interpreter reported, not the one
/// distribution metadata claims: the build that ran is the build being kept. The
/// engine's pin is checked first, so a pin that already names the SDK's build is
/// reported as satisfied rather than as an intended divergence from itself.
fn classify_retained_torch(
    sdk_build: Option<&str>,
    devices: &RuntimeDeviceCheck,
    engine_requirement: Option<&str>,
) -> TorchRetention {
    // Only a kernel that ran earns retention. Every other verdict — no devices,
    // a failed launch, or no answer at all — goes to the repair path, which is
    // also what this tool did before it could tell those apart.
    let RuntimeDeviceCheck::Usable { torch_version, .. } = devices else {
        return TorchRetention::Realign;
    };
    let Some(pinned) = engine_requirement.and_then(therock::requirement_pinned_version) else {
        return TorchRetention::Realign;
    };
    if torch_version.as_str() == pinned {
        return TorchRetention::EngineBuild {
            version: torch_version.clone(),
        };
    }
    let Some(sdk_build) = sdk_build else {
        return TorchRetention::Realign;
    };
    // The same version `plan_torch_alignment` would install, so the state that
    // repair produces is the state recognised here on the next run.
    let sdk_torch = format!("{}+{sdk_build}", therock::split_local_version(pinned).0);
    if torch_version.as_str() == sdk_torch {
        return TorchRetention::SdkBuild {
            version: torch_version.clone(),
        };
    }
    TorchRetention::Realign
}

/// The package a retained torch deliberately diverges on, if any.
///
/// The SDK's build does not satisfy the engine's exact pin and is kept anyway;
/// the engine's own build satisfies it, so claiming a divergence there would
/// suppress a violation that has not happened.
const fn retained_diverged_package(retention: &TorchRetention) -> Option<&'static str> {
    match retention {
        TorchRetention::SdkBuild { .. } => Some("torch"),
        TorchRetention::EngineBuild { .. } | TorchRetention::Realign => None,
    }
}

/// Render a retention as the `torch_alignment:` block, in place of an alignment.
///
/// `Realign` renders nothing: nothing was retained, and `render_torch_alignment`
/// prints the block for the repair that runs instead.
fn render_torch_retention(retention: &TorchRetention, engine: &str) -> String {
    let mut output = String::new();
    match retention {
        TorchRetention::EngineBuild { version } => {
            let _ = writeln!(
                output,
                "  torch_alignment: retained_engine_build ({})",
                sanitize_log_value(version)
            );
            let _ = writeln!(
                output,
                "    the torch {} pins ran a GPU kernel with this SDK",
                sanitize_log_value(engine)
            );
        }
        TorchRetention::SdkBuild { version } => {
            let _ = writeln!(
                output,
                "  torch_alignment: retained_sdk_build ({})",
                sanitize_log_value(version)
            );
            let _ = writeln!(
                output,
                "    the SDK's build of the release {} pins ran a GPU kernel",
                sanitize_log_value(engine)
            );
        }
        TorchRetention::Realign => {}
    }
    output
}

/// Print the retention block and record it. `Realign` records nothing.
fn report_torch_retention(
    paths: &AppPaths,
    engine: &str,
    runtime_key: &str,
    retention: &TorchRetention,
) {
    print!("{}", render_torch_retention(retention, engine));
    let (state, version) = match retention {
        TorchRetention::EngineBuild { version } => ("retained_engine_build", version),
        TorchRetention::SdkBuild { version } => ("retained_sdk_build", version),
        TorchRetention::Realign => return,
    };
    record_cli_audit_event(
        paths,
        "engine",
        "torch_alignment",
        "info",
        format!(
            "engine={engine} runtime_id={runtime_key} torch_alignment={state} version={version}"
        ),
        None,
    );
}

/// Whether this host has a GPU at all, answered by inspecting the host itself.
///
/// Deliberately independent of the runtime under test: the runtime reporting no
/// devices is the symptom being judged, so it cannot also be the evidence. The
/// third state is the one that matters — a host that could not be examined is not
/// a host without a GPU, and failing a multi-gigabyte install on that guess would
/// be worse than saying what was seen.
#[derive(Debug, Clone, PartialEq, Eq)]
enum HostGpu {
    Detected,
    Absent,
    NotVerified(String),
}

fn detect_host_gpu() -> HostGpu {
    // Reads only the GPU target, so it skips the driver version and date query.
    match ExamineSummary::gather_without_driver_facts() {
        Ok(summary) if summary.detected_gfx_target.is_some() => HostGpu::Detected,
        Ok(_) => HostGpu::Absent,
        Err(error) => HostGpu::NotVerified(format!("{error:#}")),
    }
}

/// Say why a runtime that cannot serve did not fail the install.
fn report_unverified_host_gpu(paths: &AppPaths, engine: &str, runtime_key: &str, reason: &str) {
    println!("  host_gpu: not_verified ({})", sanitize_log_value(reason));
    println!("    the install is not failed on a host whose GPUs could not be inspected");
    record_cli_audit_event(
        paths,
        "engine",
        "host_gpu",
        "info",
        format!("engine={engine} runtime_id={runtime_key} host_gpu=not_verified: {reason}"),
        None,
    );
}

/// Whether the runtime this install produced is one that cannot serve.
///
/// Both bad verdicts count: a runtime that opens no device and one that opens a
/// device it cannot run a kernel on both fail at first serve. Neither says
/// anything about the alignment that preceded it — a repair that could not run
/// may still leave a working environment, and one that ran may still not have
/// helped, so what the runtime does now is the only thing read.
const fn runtime_cannot_serve(devices: &RuntimeDeviceCheck) -> bool {
    matches!(
        devices,
        RuntimeDeviceCheck::NoDevices { .. } | RuntimeDeviceCheck::KernelFailed { .. }
    )
}

/// Whether an install finished having produced a runtime that cannot serve.
///
/// Gated on the host, because the same verdict means different things on
/// different machines: no device is the correct answer on a machine with no GPU,
/// and a host that could not be examined has not told us which machine this is.
/// Only a GPU found independently of the runtime turns a runtime that cannot
/// serve into a failed install.
const fn install_left_runtime_unusable(devices: &RuntimeDeviceCheck, host_gpu: &HostGpu) -> bool {
    matches!(host_gpu, HostGpu::Detected) && runtime_cannot_serve(devices)
}

/// An engine install finished having left a runtime that cannot run GPU work.
///
/// A distinct type rather than a plain message so `install sdk` can tell this
/// apart from an ordinary engine-install failure. It deliberately tolerates the
/// latter — a failed engine install still leaves a good multi-gigabyte SDK
/// behind, and discarding that would be worse — but a runtime it has just left
/// unable to serve is the exact silent success this change exists to end, so
/// that one has to fail the command.
#[derive(Debug)]
struct UnusableRuntimeAfterInstall(String);

impl std::fmt::Display for UnusableRuntimeAfterInstall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UnusableRuntimeAfterInstall {}

fn unusable_runtime_error(engine: &str, runtime_key: &str) -> anyhow::Error {
    anyhow::Error::new(UnusableRuntimeAfterInstall(format!(
        "the {engine} install left runtime `{runtime_key}` on a torch that cannot run GPU work \
         with this SDK, on a host where a GPU was found. The ROCm SDK itself is installed; read \
         the device check above and install a torch build that suits both this SDK and this GPU."
    )))
}

/// Whether a failed engine auto-install should fail `rocm install sdk` itself.
///
/// Only the unusable-runtime case does. Everything else — an unreachable engine
/// index, a resolver error, a missing build tool — leaves the SDK installed and
/// usable, so it warns and keeps the successful exit rather than throwing away
/// the install that did succeed.
fn engine_auto_install_failure_is_fatal(error: &anyhow::Error) -> bool {
    error.is::<UnusableRuntimeAfterInstall>()
}

/// Everything `install sdk` does once the SDK itself is on disk: auto-install the
/// family's engine, record the install, and decide the exit code.
///
/// The engine auto-install is a parameter so that decision is reachable from a
/// test without a multi-gigabyte install behind it. It is the part worth pinning:
/// a failed engine install must keep the successful exit, because the SDK is
/// installed and usable and only a separately retryable step is missing, while a
/// runtime this install left unable to open a device must not — reporting success
/// for that is the defect being fixed. The SDK's own audit event is recorded
/// either way, before the command fails, because the SDK install did complete.
pub(crate) fn finish_sdk_install(
    paths: &AppPaths,
    finalized: Option<&SdkInstallFinalization>,
    audit_action: &str,
    audit_detail: String,
    auto_install: impl FnOnce(&AppPaths, &SdkInstallFinalization) -> Result<()>,
) -> Result<()> {
    let mut unusable_runtime = None;
    if let Some(finalized) = finalized
        && let Err(error) = auto_install(paths, finalized)
    {
        record_cli_audit_event(
            paths,
            "engine",
            "engine_auto_install",
            "error",
            format!(
                "auto-install failed engine=vllm runtime_id={} family={}: {error}",
                finalized.runtime_key, finalized.family
            ),
            None,
        );
        if engine_auto_install_failure_is_fatal(&error) {
            unusable_runtime = Some(error);
        } else {
            eprintln!("warning: automatic vLLM install failed: {error}");
            eprintln!(
                "warning: SDK install completed; you can run `rocm engines install vllm --runtime-id {}` after vLLM is available in that runtime",
                finalized.runtime_key
            );
        }
    }
    record_cli_audit_event(paths, "runtime", audit_action, "info", audit_detail, None);
    if let Some(error) = unusable_runtime {
        return Err(error);
    }
    Ok(())
}

/// Whether rocm-cli owns the torch in the runtime this install just produced.
///
/// Two kinds of environment are left alone, because there is no torch here that
/// rocm-cli put in place. External environments are the obvious one. The subtler
/// one is an engine that manages its own runtime: those report
/// `managed_env: Some(true)` — rocm-cli did create the runtime — but their
/// `python_executable` is the engine's native binary, not an interpreter, and no
/// torch ever lived in there. Probing one anyway spawns that binary with a
/// generated `.py` path as `argv[1]` and prints check blocks about a package the
/// runtime never had.
fn settles_runtime_torch(engine: &str, managed_env: Option<bool>) -> bool {
    managed_env != Some(false) && !engine_manages_own_runtime(engine)
}

/// Settle which torch a managed runtime keeps after an engine install, then report.
///
/// Every path that installs an engine into a managed runtime must call this.
/// Skipping it anywhere lets the engine's own torch win silently and can leave a
/// runtime that cannot serve — including after an explicit
/// `rocm engines install <engine> --reinstall`, which is exactly what someone
/// reaches for when a runtime already looks wrong.
///
/// The runtime is asked what it can do before anything is written to it, and a
/// torch that already runs a GPU kernel is kept, whichever of the two intended
/// builds it is. Only when nothing has proven itself does the alignment install
/// run, and then the runtime is asked again, because that answer is now about a
/// different torch.
///
/// See `settles_runtime_torch` for the environments this deliberately skips.
pub(crate) fn settle_engine_install(
    paths: &AppPaths,
    engine: &str,
    selector: &str,
    response: &InstallResponse,
) -> Result<()> {
    if !settles_runtime_torch(engine, response.managed_env) {
        return Ok(());
    }
    let python = Path::new(&response.python_executable);
    // Which runtime is being settled has to be asked of the environment, not of
    // the caller. Two of the three callers pass a `runtime_id`, and side-by-side
    // installs of one channel and family share it, so the selector can name two
    // runtimes at once; `runtime_manifest_for_selector` then correctly declines
    // to guess and every lookup below silently comes back empty. It can also
    // name the *active* runtime while the engine was installed into an older
    // one, which is worse than empty — it settles the wrong tree's torch.
    // The interpreter is unambiguous, so let it name its own runtime.
    let owned = runtime_key_for_python(paths, python);
    let runtime_key = owned.as_deref().unwrap_or(selector);
    let library_paths = runtime_library_paths_for_key(paths, runtime_key);
    let sdk_build = sdk_torch_build_for_key(paths, runtime_key);
    let torch = therock::probe_torch_alignment(python, engine);
    let probed = runtime_device_check(Some(python), &library_paths);

    let retention = classify_retained_torch(
        sdk_build.as_deref(),
        &probed,
        torch
            .as_ref()
            .ok()
            .and_then(|probe| probe.engine_requires_torch.as_deref()),
    );
    let (diverged, devices) = match retention {
        TorchRetention::Realign => {
            let alignment = report_torch_alignment(
                paths,
                engine,
                python,
                runtime_key,
                sdk_build.as_deref(),
                torch.as_ref(),
                &probed,
            );
            // Only a realignment replaced torch. After every other outcome the
            // environment is the one already probed, and asking it again would
            // spend a second interpreter launch to be told the same thing.
            let devices = match alignment {
                TorchAlignment::Realigned { .. } => {
                    runtime_device_check(Some(python), &library_paths)
                }
                _ => probed,
            };
            (deliberately_diverged_package(&alignment), devices)
        }
        retention => {
            report_torch_retention(paths, engine, runtime_key, &retention);
            (retained_diverged_package(&retention), probed)
        }
    };
    report_engine_dependency_check(paths, engine, Some(python), runtime_key, diverged);
    let devices = report_runtime_device_check(paths, engine, runtime_key, devices);

    // Only a runtime that cannot serve raises the question the host answers, so
    // the host is inspected only then. Both readers below already require this,
    // so asking earlier would spend a full host scan on every healthy install to
    // reach an answer nothing goes on to read.
    if !runtime_cannot_serve(&devices) {
        return Ok(());
    }

    // A runtime that cannot serve is only this install's failure where the host
    // has a GPU to serve with. Where the host could not be examined, say so
    // rather than failing a multi-gigabyte install on a guess.
    let host_gpu = detect_host_gpu();
    if let HostGpu::NotVerified(reason) = &host_gpu {
        report_unverified_host_gpu(paths, engine, runtime_key, reason);
    }
    if install_left_runtime_unusable(&devices, &host_gpu) {
        return Err(unusable_runtime_error(engine, runtime_key));
    }
    Ok(())
}

fn engine_dependency_check(
    paths: &AppPaths,
    engine: &str,
    python: Option<&Path>,
    realigned_package: Option<&str>,
) -> EngineDependencyCheck {
    let Some(python) = python else {
        return EngineDependencyCheck::NotVerified(
            "the runtime's Python environment could not be located".to_owned(),
        );
    };
    match rocm_core::check_dependencies(paths, python) {
        Ok(violations) => {
            let owned = rocm_core::violations_requiring(&violations, engine);
            let details: Vec<String> = owned
                .iter()
                .map(|violation| violation.detail.clone())
                .collect();
            classify_dependency_details(details, realigned_package)
        }
        Err(error) => EngineDependencyCheck::NotVerified(error.to_string()),
    }
}

/// Separate a deliberate divergence from a genuine violation.
///
/// Each line is judged on its own subject: a genuine violation elsewhere in the
/// same run says nothing about the package the install deliberately diverged on,
/// and must not drag it along. Lumping the two together loses the distinction
/// exactly where it matters most — the remedy for the genuine violation would
/// reinstall the engine and undo the alignment.
fn classify_dependency_details(
    details: Vec<String>,
    realigned_package: Option<&str>,
) -> EngineDependencyCheck {
    if details.is_empty() {
        return EngineDependencyCheck::Satisfied;
    }
    let Some(package) = realigned_package else {
        return EngineDependencyCheck::Violated {
            violations: details,
            expected: Vec::new(),
        };
    };
    // A line whose subject cannot be parsed is not evidence of a divergence, so it
    // stays a violation: the conservative side keeps saying something is wrong.
    let (expected, violations): (Vec<String>, Vec<String>) =
        details.into_iter().partition(|detail| {
            rocm_core::violation_subject(detail)
                .is_some_and(|subject| subject.package.eq_ignore_ascii_case(package))
        });
    if violations.is_empty() {
        EngineDependencyCheck::ExpectedDivergence(expected)
    } else {
        EngineDependencyCheck::Violated {
            violations,
            expected,
        }
    }
}

fn render_engine_dependency_check(engine: &str, outcome: &EngineDependencyCheck) -> String {
    let mut output = String::new();
    match outcome {
        EngineDependencyCheck::Satisfied => {
            let _ = writeln!(output, "  dependency_check: satisfied");
        }
        EngineDependencyCheck::NotVerified(reason) => {
            let _ = writeln!(
                output,
                "  dependency_check: not_verified ({})",
                sanitize_log_value(reason)
            );
        }
        EngineDependencyCheck::Violated {
            violations,
            expected,
        } => {
            let _ = writeln!(output, "  dependency_check: violated");
            for detail in violations {
                let _ = writeln!(output, "  violation: {}", sanitize_log_value(detail));
            }
            for detail in expected {
                let _ = writeln!(output, "  divergence: {}", sanitize_log_value(detail));
            }
            if expected.is_empty() {
                let _ = writeln!(
                    output,
                    "  action: rocm engines install {engine} --reinstall"
                );
            } else {
                // The reinstall would repair the violations above and reinstate the
                // engine's own build of the diverged package, restoring a runtime
                // that cannot open a device. Naming a remedy that trades one defect
                // for a worse one is not worth the convenience.
                let _ = writeln!(
                    output,
                    "  action: repair the violations above without reinstalling {engine}; a reinstall would undo the divergence kept on purpose (see torch_alignment above)"
                );
            }
        }
        EngineDependencyCheck::ExpectedDivergence(details) => {
            let _ = writeln!(output, "  dependency_check: expected_divergence");
            for detail in details {
                let _ = writeln!(output, "  divergence: {}", sanitize_log_value(detail));
            }
            // Deliberately not the reinstall remedy. It does not apply to
            // either state that reaches here: a reinstall realigns torch again
            // and reproduces the SDK build, and where the user has opted out of
            // that realignment it would undo the decision they made. Which of
            // the two this is, the block above has already said.
            let _ = writeln!(
                output,
                "  action: none; this torch is kept on purpose (see torch_alignment above)"
            );
        }
    }
    output
}

pub(crate) fn render_sdk_install_success(finalized: &SdkInstallFinalization) -> String {
    let mut output = format!(
        "ROCm SDK installed successfully.\n  install folder: {}\n  active runtime: {}\n  next step: run `rocm help` to see how to use rocm-cli.\n",
        finalized.install_root.display(),
        finalized.runtime_key
    );
    let _ = write!(
        output,
        "{}",
        runtime_services::render_runtime_service_reconciliation(
            &finalized.services,
            &finalized.runtime_key
        )
    );
    if finalized.previous_runtime_key.is_some() {
        let _ = writeln!(output, "{ROLLBACK_RECOVERY_HINT}");
    }
    output
}

pub(crate) fn finalize_successful_sdk_install(
    paths: &AppPaths,
) -> Result<Option<SdkInstallFinalization>> {
    let Some(manifest) = newest_installed_runtime_manifest(paths)? else {
        return Ok(None);
    };
    let mut config = RocmCliConfig::load(paths)?;
    config.setup.completed = true;
    config.setup.therock_venv = Some(manifest.install_root.clone());
    config.save(paths)?;

    let activation_paths = paths
        .clone()
        .with_managed_root(manifest.install_root.clone(), false);
    if paths.config_dir != activation_paths.config_dir
        || paths.data_dir != activation_paths.data_dir
    {
        recover_setup_runtime_registration(paths, &config)?;
        let mut current_config = RocmCliConfig::load(paths)?;
        current_config.setup.completed = true;
        current_config.setup.therock_venv = Some(manifest.install_root.clone());
        // This activation re-activates in the caller's original root. The
        // user-facing reconciliation report comes from the activation against
        // `activation_paths` below, so the result here is intentionally discarded.
        let _ = activate_runtime(paths, &mut current_config, &manifest.runtime_key)?;
    }

    recover_setup_runtime_registration(&activation_paths, &config)?;

    let mut config = RocmCliConfig::load(&activation_paths)?;
    config.setup.completed = true;
    config.setup.therock_venv = Some(manifest.install_root.clone());
    let activation = activate_runtime(&activation_paths, &mut config, &manifest.runtime_key)?;

    Ok(Some(SdkInstallFinalization {
        runtime_key: activation.runtime_key,
        install_root: manifest.install_root,
        family: manifest.family,
        previous_runtime_key: activation.previous_runtime_key,
        services: activation.services,
    }))
}

#[cfg(test)]
pub(crate) mod tests {
    use std::fs;

    use super::*;
    use crate::test_support::write_test_pip_runtime;
    use crate::tests::{
        NEW_RUNTIME_KEY, OLD_RUNTIME_KEY, ScopedTestEnv, plant_service_record_on_runtime,
        test_app_paths, test_paths, test_runtime_manifest_for_update,
    };
    use clap::Parser as _;
    use rocm_core::load_recent_audit_events;

    use crate::chat::rocm_chat_tool_requested_args;
    use crate::{Cli, Command, InstallTarget, providers};

    /// Resolve the argv a non-interactive surface produces the way `install()`
    /// does, so a test can assert what that argv actually consents to rather
    /// than which flag string it happens to contain.
    pub(crate) fn consents_for_install_sdk_argv(args: &[String]) -> SdkInstallConsents {
        let cli =
            Cli::try_parse_from(std::iter::once("rocm".to_owned()).chain(args.iter().cloned()))
                .unwrap_or_else(|error| {
                    panic!("{args:?} must parse as a `rocm` invocation: {error}")
                });
        let Some(Command::Install {
            target:
                InstallTarget::Sdk {
                    yes,
                    approve_replacing_active_default,
                    ..
                },
        }) = cli.command
        else {
            panic!("{args:?} is not an `install sdk` invocation");
        };
        SdkInstallConsents::resolve(yes, approve_replacing_active_default)
    }

    #[test]
    fn an_injected_consent_does_not_approve_privileged_package_installs() {
        // The worked regression: Linux, a vLLM-preferred GPU family, OpenMPI
        // absent, and neither root nor passwordless sudo. The chat, MCP and
        // daemon surfaces spawn `rocm` with null stdin, so there is no terminal
        // on which a `sudo` password prompt could ever be answered. Injecting
        // `--yes` to clear the runtime-replacement prompt used to grant the
        // second, unrelated consent that flag carries, which made
        // `ensure_openmpi_for_vllm` run a sudo it cannot complete and then
        // escalate the failure — aborting `maybe_auto_install_sdk_preferred_engine`
        // before the vLLM engine install that previously warned and continued.
        let call = providers::ChatToolCall {
            id: None,
            name: "install_sdk".to_owned(),
            arguments: serde_json::json!({ "channel": "release", "format": "wheel" }),
        };
        let injected = consents_for_install_sdk_argv(
            &rocm_chat_tool_requested_args(&call).expect("install_sdk tool builds args"),
        );
        assert_eq!(
            injected.replace_active_default,
            therock::SdkInstallConsent::Preapproved(
                therock::SdkInstallApprovalSource::ApproveReplacingActiveDefault
            )
        );
        assert!(
            !injected.system_packages,
            "an injected consent must not approve a privileged package install"
        );
        assert_eq!(
            system_package_install_action(injected.system_packages, false),
            SystemPackageInstallAction::PrintManualCommands,
            "on a host without passwordless sudo the spawn must print the commands, not run sudo"
        );

        // And the other half of the property: a `--yes` the user actually typed
        // still approves both, and a failure of the install it asked for is
        // still an error rather than a warning.
        let typed = consents_for_install_sdk_argv(&[
            "install".to_owned(),
            "sdk".to_owned(),
            "--yes".to_owned(),
        ]);
        assert_eq!(
            typed.replace_active_default,
            therock::SdkInstallConsent::Preapproved(therock::SdkInstallApprovalSource::AssumeYes),
            "a --yes the user typed must still be credited to --yes"
        );
        assert!(typed.system_packages);
        assert_eq!(
            system_package_install_action(typed.system_packages, false),
            SystemPackageInstallAction::RunPlan {
                approval: "granted by --yes",
                escalate_failure: true,
            },
        );

        // Root or passwordless sudo installs without any approval, as before —
        // the consent split must not have made the automatic path conditional.
        assert_eq!(
            system_package_install_action(injected.system_packages, true),
            SystemPackageInstallAction::RunPlan {
                approval: "auto (root or passwordless sudo available)",
                escalate_failure: false,
            },
        );
    }

    #[test]
    fn sdk_install_auto_engine_selection_prefers_vllm_for_supported_families() {
        // vLLM is unsupported on native Windows, so the SDK family preference is gated
        // off there and resolves to None.
        let expected = if cfg!(windows) { None } else { Some("vllm") };
        assert_eq!(preferred_engine_for_sdk_family("gfx90a"), expected);
        assert_eq!(preferred_engine_for_sdk_family("gfx94X-dcgpu"), expected);
        assert_eq!(preferred_engine_for_sdk_family("gfx120X-all"), None);
    }

    /// The real candidate path/text `real_os_release_ignores_a_simulated_host_root`
    /// expects, read independently rather than by calling
    /// `hardware_root::read_os_release` a second time. `Absent` and
    /// `Unreadable` are not interchangeable: a non-`NotFound` error must
    /// always report `Unreadable`, never silently fall through to `Absent`.
    #[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
    enum ExpectedOsRelease {
        Found(&'static str, String),
        Absent,
        Unreadable(&'static str),
    }

    /// The package-manager plans run against the real machine, so a simulated
    /// host root must not change the distro they are built for: under a root
    /// whose `/etc/os-release` names a different distro, `real_os_release`
    /// still returns the real file.
    #[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
    #[test]
    fn real_os_release_ignores_a_simulated_host_root() {
        let root = std::env::temp_dir().join(format!(
            "rocm-os-release-host-root-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        fs::create_dir_all(root.join("etc")).expect("create planted /etc");
        fs::write(
            root.join("etc/os-release"),
            "ID=planted-distro\nVERSION_ID=\"99.9\"\n",
        )
        .expect("write planted os-release");

        let read = {
            let mut env = ScopedTestEnv::new();
            env.set(
                rocm_core::hardware_root::TEST_HOST_ROOT_ENV,
                &root.to_string_lossy(),
            );
            real_os_release()
        };
        fs::remove_dir_all(&root).ok();

        assert!(
            !read.text().contains("planted-distro"),
            "read the simulated root's os-release: {:?}",
            read.text()
        );
        // Independent of `hardware_root::read_os_release`: rather than call
        // the exact same helper with the same identity resolver `real_os_release`
        // wraps one line away (which would pass even if its `.context(...)`
        // call or resolver closure broke), read both candidate paths directly
        // and apply the same existence-only fallback rule inline. This mirrors
        // which variant is correct for each case, rather than accepting
        // `Absent` or `Unreadable` interchangeably — see `ExpectedOsRelease`'s
        // own doc comment for why that distinction matters.
        let expected = match fs::read_to_string("/etc/os-release") {
            Ok(text) => ExpectedOsRelease::Found("/etc/os-release", text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                match fs::read_to_string("/usr/lib/os-release") {
                    Ok(text) => ExpectedOsRelease::Found("/usr/lib/os-release", text),
                    Err(e2) if e2.kind() == std::io::ErrorKind::NotFound => {
                        ExpectedOsRelease::Absent
                    }
                    Err(_) => ExpectedOsRelease::Unreadable("/usr/lib/os-release"),
                }
            }
            Err(_) => ExpectedOsRelease::Unreadable("/etc/os-release"),
        };
        match expected {
            ExpectedOsRelease::Found(path, text) => {
                assert_eq!((read.path(), read.text()), (path, text.as_str()));
            }
            ExpectedOsRelease::Absent => assert!(
                matches!(&read, rocm_core::hardware_root::OsRelease::Absent),
                "expected Absent, got {read:?}"
            ),
            ExpectedOsRelease::Unreadable(path) => assert!(
                matches!(
                    &read,
                    rocm_core::hardware_root::OsRelease::Unreadable { path: p, .. } if *p == path
                ),
                "expected Unreadable at {path}, got {read:?}"
            ),
        }
    }

    // The install report is only as honest as the reconciliation that reaches
    // it: a server still on the runtime the install just replaced has to survive
    // `finalize_successful_sdk_install` into the finalization the report renders.
    // The service is planted under the same rebased paths the production
    // activation reads its service records from.
    // Mutation: replacing `services: activation.services` with a default
    // reconciliation in `finalize_successful_sdk_install` fails the `stale`
    // assertion and the rendered-count assertion.
    #[test]
    fn sdk_install_finalization_carries_the_servers_left_on_the_replaced_runtime() -> Result<()> {
        let (root, paths) = test_paths("sdk-install-finalization-services");
        write_test_pip_runtime(
            &paths,
            OLD_RUNTIME_KEY,
            "therock-release:gfx120X-all",
            "7.12.0",
            10,
        )?;
        let newest = write_test_pip_runtime(
            &paths,
            NEW_RUNTIME_KEY,
            "therock-release:gfx120X-all",
            "7.13.0",
            20,
        )?;
        let activation_paths = paths.clone().with_managed_root(newest.install_root, false);
        activation_paths.ensure()?;
        plant_service_record_on_runtime(
            &activation_paths,
            "svc-sdk-stale",
            "vllm",
            "starting",
            std::process::id(),
            Some(OLD_RUNTIME_KEY),
            None,
        )?;

        let finalized = finalize_successful_sdk_install(&paths)?
            .context("sdk install finalization should select the newest runtime")?;
        let rendered = render_sdk_install_success(&finalized);
        let _ = fs::remove_dir_all(&root);

        assert_eq!(finalized.runtime_key, NEW_RUNTIME_KEY);
        assert_eq!(
            finalized.services.stale,
            vec![runtime_services::RuntimeServiceEntry {
                service_id: "svc-sdk-stale".to_owned(),
                engine: "vllm".to_owned(),
                recorded_runtime: Some(OLD_RUNTIME_KEY.to_owned()),
            }],
            "the server still on the replaced runtime must be carried into the \
             finalization: {:?}",
            finalized.services
        );
        assert!(
            rendered.contains("services_on_previous_runtime: 1")
                && rendered.contains(&format!(
                    "- svc-sdk-stale engine=vllm recorded_runtime={OLD_RUNTIME_KEY}"
                )),
            "the install report names the server the install left behind:\n{rendered}"
        );
        Ok(())
    }

    #[test]
    fn a_satisfied_dependency_check_is_stated_plainly() {
        let rendered = render_engine_dependency_check("vllm", &EngineDependencyCheck::Satisfied);

        assert_eq!(rendered, "  dependency_check: satisfied\n");
    }

    #[test]
    fn a_violated_dependency_check_names_the_pin_and_the_remedy() {
        // The SDK torch stack replaced the build vLLM pins. The install
        // succeeded, so the only signal the user gets is this block.
        let rendered = render_engine_dependency_check(
            "vllm",
            &EngineDependencyCheck::Violated {
                violations: vec![
                    "The package `vllm` requires `torch==2.10.0+git8514f05`, but `2.9.1+rocm7.14.0a20260611` is installed".to_owned(),
                ],
                expected: Vec::new(),
            },
        );

        assert!(rendered.contains("  dependency_check: violated\n"));
        assert!(rendered.contains("torch==2.10.0+git8514f05"));
        assert!(rendered.contains("2.9.1+rocm7.14.0a20260611"));
        assert!(rendered.contains("  action: rocm engines install vllm --reinstall\n"));
    }

    #[test]
    fn a_divergence_on_the_realigned_package_does_not_advise_undoing_it() {
        // The reinstall remedy is correct for a real violation and catastrophic
        // here: it reinstates the engine's own build and restores a runtime that
        // cannot open a device.
        let outcome = classify_dependency_details(
            vec![
                "The package `vllm` requires `torch==2.11.0+gitd0c8b1f`, but `2.11.0+rocm7.13.0` is installed".to_owned(),
            ],
            Some("torch"),
        );

        assert!(matches!(
            outcome,
            EngineDependencyCheck::ExpectedDivergence(_)
        ));
        let rendered = render_engine_dependency_check("vllm", &outcome);
        assert!(rendered.contains("  dependency_check: expected_divergence\n"));
        assert!(
            !rendered.contains("action: rocm engines install vllm --reinstall"),
            "the reinstall remedy does not apply to a deliberate divergence: {rendered}"
        );
    }

    #[test]
    fn a_divergence_on_any_other_package_is_still_a_violation() {
        // The unrelated violation is real and must keep saying so, but it does not
        // make the deliberate torch divergence one too — and the reinstall that
        // would repair numpy is exactly what must not be advised while the
        // alignment stands.
        let outcome = classify_dependency_details(
            vec![
                "The package `vllm` requires `torch==2.11.0+gitd0c8b1f`, but `2.11.0+rocm7.13.0` is installed".to_owned(),
                "The package `vllm` requires `numpy==1.26.4`, but `2.0.0` is installed".to_owned(),
            ],
            Some("torch"),
        );

        let EngineDependencyCheck::Violated {
            violations,
            expected,
        } = &outcome
        else {
            panic!("an unrelated violation is still a violation: {outcome:?}");
        };
        assert_eq!(violations.len(), 1, "only numpy violates: {violations:?}");
        assert!(violations[0].contains("numpy"));
        assert_eq!(expected.len(), 1, "torch diverged on purpose: {expected:?}");
        assert!(expected[0].contains("torch"));

        let rendered = render_engine_dependency_check("vllm", &outcome);
        assert!(rendered.contains("  dependency_check: violated\n"));
        assert!(rendered.contains("  violation: The package `vllm` requires `numpy"));
        assert!(rendered.contains("  divergence: The package `vllm` requires `torch"));
        assert!(
            !rendered.contains("action: rocm engines install vllm --reinstall"),
            "the reinstall would undo the alignment while repairing numpy: {rendered}"
        );
    }

    #[test]
    fn a_line_whose_subject_cannot_be_parsed_stays_a_violation() {
        // Conservative by construction: an unrecognised shape is not evidence that
        // the divergence was intended, so it must not be quietly excused.
        let outcome = classify_dependency_details(
            vec!["something uv said that this parser does not recognise".to_owned()],
            Some("torch"),
        );

        assert!(matches!(outcome, EngineDependencyCheck::Violated { .. }));
    }

    /// The engine's build enumerates no devices against the installed SDK.
    ///
    /// Observed on MI300X: the engine pins a torch built for a different ROCm
    /// version, installs it over the SDK's, and the runtime then reports zero
    /// devices. The release is right; only the build is wrong.
    #[test]
    fn the_engines_build_is_replaced_by_the_sdk_build_of_the_same_release() {
        let plan = plan_torch_alignment(
            Some("rocm7.13.0"),
            Some("2.11.0+gitd0c8b1f"),
            Some("torch==2.11.0+gitd0c8b1f"),
            "vllm",
        );

        assert_eq!(
            plan,
            TorchAlignmentPlan::Install {
                wanted: "2.11.0+rocm7.13.0".to_owned(),
                from: "2.11.0+gitd0c8b1f".to_owned(),
            }
        );
    }

    /// The SDK picked a torch *release* the engine does not accept.
    ///
    /// The mirror-image failure: here the SDK's own choice is the wrong one, and
    /// simply keeping it would break the engine. The rule takes the release the
    /// engine pins while still keeping the SDK's build of it — so neither the
    /// engine's build nor the SDK's release selection wins outright.
    #[test]
    fn the_sdks_release_is_corrected_to_the_one_the_engine_pins() {
        // The installed torch already carries the SDK's build, so the build is not
        // what is wrong here — the release is. Pinning it needs the two sides to
        // disagree on the release and agree on the build, which is the opposite of
        // the case above; identical arguments to it would only restate that one.
        let plan = plan_torch_alignment(
            Some("rocm7.14.0a20260611"),
            Some("2.11.0+rocm7.14.0a20260611"),
            Some("torch==2.10.0+git8514f05"),
            "vllm",
        );

        assert_eq!(
            plan,
            TorchAlignmentPlan::Install {
                wanted: "2.10.0+rocm7.14.0a20260611".to_owned(),
                from: "2.11.0+rocm7.14.0a20260611".to_owned(),
            },
            "the engine's release must win while the SDK's build is kept"
        );
    }

    #[test]
    fn an_install_is_failed_only_when_it_left_a_runtime_that_cannot_serve() {
        let no_devices = RuntimeDeviceCheck::NoDevices {
            torch_version: "2.11.0+gitd0c8b1f".to_owned(),
            hip_version: "7.2.53211".to_owned(),
        };
        let kernel_failed = RuntimeDeviceCheck::KernelFailed {
            torch_version: "2.11.0+rocm7.13.0".to_owned(),
            error: "AcceleratorError: device kernel image is invalid".to_owned(),
        };
        let usable = RuntimeDeviceCheck::Usable {
            device_count: 8,
            torch_version: "2.11.0+rocm7.13.0".to_owned(),
        };

        // A machine with a GPU that this runtime cannot use, either way round.
        assert!(install_left_runtime_unusable(
            &no_devices,
            &HostGpu::Detected
        ));
        assert!(install_left_runtime_unusable(
            &kernel_failed,
            &HostGpu::Detected
        ));
        // A machine with a GPU and a runtime that can use it.
        assert!(!install_left_runtime_unusable(&usable, &HostGpu::Detected));
        // On a host with no GPU both verdicts are the expected answer, not a
        // reason to throw away a multi-gigabyte install.
        assert!(!install_left_runtime_unusable(
            &no_devices,
            &HostGpu::Absent
        ));
        assert!(!install_left_runtime_unusable(
            &kernel_failed,
            &HostGpu::Absent
        ));
        // A host we could not examine is not a host without a GPU, but it is not
        // evidence of one either, so it never fails the install on its own.
        let unknown = HostGpu::NotVerified("lspci is not installed".to_owned());
        assert!(!install_left_runtime_unusable(&no_devices, &unknown));
        assert!(!install_left_runtime_unusable(&kernel_failed, &unknown));
    }

    #[test]
    fn a_kernel_launch_failure_is_not_reported_as_a_usable_device() {
        // The probe enumerated a device and then failed to run anything on it.
        // Reading only the count calls that healthy, which is the whole defect:
        // the install passes and the first serve dies.
        let outcome = classify_runtime_device_probe(therock::RuntimeDeviceProbe {
            import_ok: true,
            torch_version: Some("2.11.0+rocm7.14.0".to_owned()),
            hip_version: Some("7.14.0".to_owned()),
            device_count: Some(1),
            error: None,
            kernel_error: Some("AcceleratorError: device kernel image is invalid".to_owned()),
        });

        assert_eq!(
            outcome,
            RuntimeDeviceCheck::KernelFailed {
                torch_version: "2.11.0+rocm7.14.0".to_owned(),
                error: "AcceleratorError: device kernel image is invalid".to_owned(),
            }
        );
    }

    #[test]
    fn an_enumeration_failure_stays_a_verdict_of_its_own() {
        // `error` still carries the failures that happen before any kernel runs.
        // Reporting one of those as a kernel failure would name the wrong remedy.
        let import_failed = classify_runtime_device_probe(therock::RuntimeDeviceProbe {
            import_ok: false,
            torch_version: None,
            hip_version: None,
            device_count: None,
            error: Some("ImportError: libamdhip64.so.7".to_owned()),
            kernel_error: None,
        });
        let count_raised = classify_runtime_device_probe(therock::RuntimeDeviceProbe {
            import_ok: true,
            torch_version: Some("2.11.0+rocm7.13.0".to_owned()),
            hip_version: Some("7.13.0".to_owned()),
            device_count: None,
            error: Some("RuntimeError: HIP failed to initialize".to_owned()),
            kernel_error: None,
        });
        let no_devices = classify_runtime_device_probe(therock::RuntimeDeviceProbe {
            import_ok: true,
            torch_version: Some("2.11.0+gitd0c8b1f".to_owned()),
            hip_version: Some("7.13.0".to_owned()),
            device_count: Some(0),
            error: None,
            kernel_error: None,
        });

        assert_eq!(
            import_failed,
            RuntimeDeviceCheck::NotVerified("ImportError: libamdhip64.so.7".to_owned())
        );
        assert_eq!(
            count_raised,
            RuntimeDeviceCheck::NotVerified("RuntimeError: HIP failed to initialize".to_owned())
        );
        assert_eq!(
            no_devices,
            RuntimeDeviceCheck::NoDevices {
                torch_version: "2.11.0+gitd0c8b1f".to_owned(),
                hip_version: "7.13.0".to_owned(),
            }
        );
    }

    #[test]
    fn a_kernel_failure_is_reported_as_a_failure_and_says_what_broke() {
        let rendered = render_runtime_device_check(&RuntimeDeviceCheck::KernelFailed {
            torch_version: "2.11.0+rocm7.14.0".to_owned(),
            error: "AcceleratorError: device kernel image is invalid".to_owned(),
        });

        assert!(rendered.contains("  device_check: kernel_failed\n"));
        assert!(rendered.contains("2.11.0+rocm7.14.0"));
        assert!(rendered.contains("device kernel image is invalid"));
        // The remedy differs from `no_devices`, so the text must not borrow its
        // explanation about being built for a different ROCm version.
        assert!(!rendered.contains("Failed to infer device type"));
        assert!(rendered.contains("no kernel image for this GPU"));
    }

    /// The engine's own pin runs: keep it, and claim no divergence from it.
    #[test]
    fn a_working_engine_torch_is_kept_instead_of_realigned() {
        let retention = classify_retained_torch(
            Some("rocm7.13.0"),
            &RuntimeDeviceCheck::Usable {
                device_count: 1,
                torch_version: "2.11.0+gitd0c8b1f".to_owned(),
            },
            Some("torch==2.11.0+gitd0c8b1f"),
        );

        assert_eq!(
            retention,
            TorchRetention::EngineBuild {
                version: "2.11.0+gitd0c8b1f".to_owned(),
            }
        );
        assert_eq!(retained_diverged_package(&retention), None);
        let rendered = render_torch_retention(&retention, "vllm");
        assert!(
            rendered.contains("  torch_alignment: retained_engine_build (2.11.0+gitd0c8b1f)\n")
        );
        assert!(rendered.contains("ran a GPU kernel with this SDK"));
    }

    /// The SDK's build of the pinned release runs: keep it, and expect the pin to
    /// stay unsatisfied so the dependency check does not report it as a violation.
    #[test]
    fn a_working_sdk_torch_is_kept_as_an_intended_divergence() {
        let retention = classify_retained_torch(
            Some("rocm7.13.0"),
            &RuntimeDeviceCheck::Usable {
                device_count: 8,
                torch_version: "2.11.0+rocm7.13.0".to_owned(),
            },
            Some("torch==2.11.0+gitd0c8b1f"),
        );

        assert_eq!(
            retention,
            TorchRetention::SdkBuild {
                version: "2.11.0+rocm7.13.0".to_owned(),
            }
        );
        assert_eq!(retained_diverged_package(&retention), Some("torch"));
        let rendered = render_torch_retention(&retention, "vllm");
        assert!(rendered.contains("  torch_alignment: retained_sdk_build (2.11.0+rocm7.13.0)\n"));
        assert!(rendered.contains("the SDK's build of the release vllm pins ran a GPU kernel"));
    }

    /// Both intended builds are fixed points, so a repeated install settles rather
    /// than swapping torch back and forth between the two package sources.
    #[test]
    fn repairing_a_runtime_converges_on_one_of_the_two_intended_builds() {
        let after_engine_install = classify_retained_torch(
            Some("rocm7.13.0"),
            &RuntimeDeviceCheck::Usable {
                device_count: 1,
                torch_version: "2.11.0+gitd0c8b1f".to_owned(),
            },
            Some("torch==2.11.0+gitd0c8b1f"),
        );
        // What the realignment installs, seen on the next run over the same runtime.
        let after_realignment = classify_retained_torch(
            Some("rocm7.13.0"),
            &RuntimeDeviceCheck::Usable {
                device_count: 1,
                torch_version: "2.11.0+rocm7.13.0".to_owned(),
            },
            Some("torch==2.11.0+gitd0c8b1f"),
        );

        assert!(!matches!(after_engine_install, TorchRetention::Realign));
        assert!(!matches!(after_realignment, TorchRetention::Realign));
    }

    /// A pin that already names the SDK's build satisfies itself. Calling that an
    /// intended divergence would suppress a violation that has not happened.
    #[test]
    fn a_pin_that_already_names_the_sdk_build_diverges_from_nothing() {
        let retention = classify_retained_torch(
            Some("rocm7.13.0"),
            &RuntimeDeviceCheck::Usable {
                device_count: 1,
                torch_version: "2.11.0+rocm7.13.0".to_owned(),
            },
            Some("torch==2.11.0+rocm7.13.0"),
        );

        assert_eq!(
            retention,
            TorchRetention::EngineBuild {
                version: "2.11.0+rocm7.13.0".to_owned(),
            }
        );
        assert_eq!(retained_diverged_package(&retention), None);
    }

    /// Everything that has not been shown to work goes to the repair path — which
    /// is what this tool did unconditionally before it could test a kernel.
    #[test]
    fn a_torch_that_has_not_been_shown_to_work_is_realigned() {
        let usable = |version: &str| RuntimeDeviceCheck::Usable {
            device_count: 1,
            torch_version: version.to_owned(),
        };
        let pin = Some("torch==2.11.0+gitd0c8b1f");

        // A third build nobody here installed on purpose.
        assert_eq!(
            classify_retained_torch(Some("rocm7.13.0"), &usable("2.9.0+cpu"), pin),
            TorchRetention::Realign
        );
        // No devices at all, whatever build it is.
        assert_eq!(
            classify_retained_torch(
                Some("rocm7.13.0"),
                &RuntimeDeviceCheck::NoDevices {
                    torch_version: "2.11.0+gitd0c8b1f".to_owned(),
                    hip_version: "7.13.0".to_owned(),
                },
                pin
            ),
            TorchRetention::Realign
        );
        // A device that cannot run a kernel is never retained, not even when it
        // holds exactly the build the engine pins.
        assert_eq!(
            classify_retained_torch(
                Some("rocm7.13.0"),
                &RuntimeDeviceCheck::KernelFailed {
                    torch_version: "2.11.0+gitd0c8b1f".to_owned(),
                    error: "device kernel image is invalid".to_owned(),
                },
                pin
            ),
            TorchRetention::Realign
        );
        // No answer is not an answer.
        assert_eq!(
            classify_retained_torch(
                Some("rocm7.13.0"),
                &RuntimeDeviceCheck::NotVerified("torch did not import".to_owned()),
                pin
            ),
            TorchRetention::Realign
        );
        // Nothing pins torch, so there is no release to hold either build of.
        assert_eq!(
            classify_retained_torch(Some("rocm7.13.0"), &usable("2.11.0+gitd0c8b1f"), None),
            TorchRetention::Realign
        );
        // A range is not an exact pin, so it cannot identify the engine's build.
        assert_eq!(
            classify_retained_torch(
                Some("rocm7.13.0"),
                &usable("2.11.0+gitd0c8b1f"),
                Some("torch>=2.10")
            ),
            TorchRetention::Realign
        );
        // The manifest does not say what the SDK's build is, so a torch that is
        // not the engine's pin cannot be recognised as the other intended one.
        assert_eq!(
            classify_retained_torch(None, &usable("2.11.0+rocm7.13.0"), pin),
            TorchRetention::Realign
        );
        // Nothing is retained, so nothing is reported in place of the repair.
        assert!(render_torch_retention(&TorchRetention::Realign, "vllm").is_empty());
        assert_eq!(retained_diverged_package(&TorchRetention::Realign), None);
    }

    fn sdk_install_finalization() -> SdkInstallFinalization {
        SdkInstallFinalization {
            runtime_key: "wheel-gfx942-7.13.0".to_owned(),
            install_root: PathBuf::from("/tmp/does-not-need-to-exist"),
            family: "gfx94X-dcgpu".to_owned(),
            previous_runtime_key: None,
            services: runtime_services::RuntimeServiceReconciliation::default(),
        }
    }

    /// Run the real `install sdk` completion path against a stubbed auto-install.
    pub(crate) fn finish_sdk_install_with(
        paths: &AppPaths,
        outcome: Result<()>,
    ) -> (Result<()>, SdkInstallFinalization) {
        let finalized = sdk_install_finalization();
        let result = finish_sdk_install(
            paths,
            Some(&finalized),
            "install_sdk",
            "sdk install completed".to_owned(),
            |_, _| outcome,
        );
        (result, finalized)
    }

    #[test]
    fn render_sdk_install_success_includes_service_reconciliation_section() {
        // Verify that the reconciliation section is printed even when no services
        // are stale, so callers can distinguish "looked and found nothing" from
        // "never looked" — the same guarantee `runtimes activate` provides.
        // A previous runtime is recorded, so the rollback hint is printed too and
        // the ordering below has two things to order.
        let finalized = SdkInstallFinalization {
            previous_runtime_key: Some("wheel-gfx942-7.12.0".to_owned()),
            ..sdk_install_finalization()
        };
        let rendered = render_sdk_install_success(&finalized);
        assert!(
            rendered.contains("services_on_previous_runtime:"),
            "`render_sdk_install_success` must include the reconciliation section \
             so the install report is consistent with `runtimes activate`:\n{rendered}"
        );
        assert!(
            rendered.contains("services_on_previous_runtime: 0"),
            "with no live services the count must be 0:\n{rendered}"
        );
        // The reconciliation must appear before any rollback hint, matching the
        // ordering the `runtimes activate` handler uses.
        // Mutation: printing the hint before the reconciliation section fails the
        // position comparison; dropping the hint fails the `expect`.
        let recon_pos = rendered
            .find("services_on_previous_runtime:")
            .expect("the reconciliation section is printed");
        let rollback_pos = rendered
            .find("rocm runtimes rollback")
            .expect("a recorded previous runtime prints the rollback hint");
        assert!(
            recon_pos < rollback_pos,
            "reconciliation section must precede the rollback hint:\n{rendered}"
        );
    }

    #[test]
    fn install_sdk_fails_when_the_install_left_a_runtime_that_cannot_open_a_device() -> Result<()> {
        let (_root, paths) = test_paths("install-sdk-unusable");

        // Exercises the command's own completion path, not just the predicate: if the
        // catch goes back to warning and falling through, this fails.
        let (result, _) = finish_sdk_install_with(
            &paths,
            Err(unusable_runtime_error("vllm", "wheel-gfx942-7.13.0")),
        );
        let error = result.expect_err("an unusable runtime must fail `install sdk`");
        // And it must still say the SDK survived, so a transient index failure does
        // not read as a ruined install that has to be started from scratch.
        assert!(
            error
                .to_string()
                .contains("The ROCm SDK itself is installed"),
            "the failure must say the SDK survived:\n{error}"
        );

        // The SDK install did complete, so its record is written before the command
        // fails — otherwise the audit trail would show an install that never happened.
        let actions: Vec<String> = load_recent_audit_events(&paths, 10)?
            .into_iter()
            .map(|event| event.action)
            .collect();
        assert!(
            actions.iter().any(|action| action == "install_sdk"),
            "the successful SDK install was not recorded before failing: {actions:?}"
        );
        Ok(())
    }

    #[test]
    fn the_unusable_runtime_failure_survives_being_wrapped_in_context() {
        // The error reaches the catch through several `?` hops. Flattening it into a
        // plain message is the realistic way a later change silently restores the
        // exit-0 bug, since every failure would then look recoverable.
        let propagated: Result<()> = Err(unusable_runtime_error("vllm", "wheel-gfx942-7.13.0"))
            .context("automatic vLLM install failed");
        let propagated = propagated.expect_err("expected the unusable-runtime error");
        assert!(engine_auto_install_failure_is_fatal(&propagated));
        assert!(!engine_auto_install_failure_is_fatal(&anyhow::anyhow!(
            "uv exited with status 1"
        )));
    }

    #[test]
    fn an_install_failure_is_not_reported_as_a_missing_wheel() {
        // Saying "the index publishes no such build" over a network or disk error
        // sends the reader hunting for a wheel that exists. Anything the resolver
        // did not actually attribute to a missing version degrades to the honest
        // answer: the failure, verbatim.
        assert!(!install_error_reports_version_unavailable(
            "failed to launch uv: Permission denied"
        ));
        assert!(!install_error_reports_version_unavailable(
            "error sending request: dns error: failed to lookup address"
        ));
        assert!(install_error_reports_version_unavailable(
            "No solution found when resolving: torch==2.10.0+rocm7.14.0a20260611"
        ));
    }

    #[test]
    fn an_unresolvable_version_still_reads_as_unavailable() {
        let rendered = render_torch_alignment(
            &TorchAlignment::Unavailable {
                wanted: "2.10.0+rocm7.14.0a20260611".to_owned(),
                kept: "2.10.0+git8514f05".to_owned(),
            },
            "vllm",
            None,
        );

        assert!(rendered.contains("  torch_alignment: unavailable\n"));
        assert!(rendered.contains("publishes no 2.10.0+rocm7.14.0a20260611"));
    }

    #[test]
    fn a_failed_realignment_reports_the_error_it_actually_hit() {
        let rendered = render_torch_alignment(
            &TorchAlignment::InstallFailed {
                wanted: "2.11.0+rocm7.13.0".to_owned(),
                kept: "2.11.0+gitd0c8b1f".to_owned(),
                error: "failed to launch uv: Permission denied".to_owned(),
            },
            "vllm",
            None,
        );

        assert!(rendered.contains("  torch_alignment: install_failed\n"));
        assert!(rendered.contains("Permission denied"));
        assert!(
            !rendered.contains("publishes no"),
            "an install failure must not be described as a missing wheel: {rendered}"
        );
    }

    #[test]
    fn a_realigned_runtime_names_both_builds_and_the_engine_that_decided_the_release() {
        // The line a successful run actually prints, and the one a user reads when
        // deciding whether the divergence reported just below it is expected.
        let rendered = render_torch_alignment(
            &TorchAlignment::Realigned {
                from: "2.11.0+gitd0c8b1f".to_owned(),
                to: "2.11.0+rocm7.13.0".to_owned(),
            },
            "vllm",
            None,
        );

        assert!(rendered.contains("  torch_alignment: realigned\n"));
        assert!(rendered.contains("2.11.0+gitd0c8b1f -> 2.11.0+rocm7.13.0"));
        assert!(
            rendered.contains("the release vllm pins"),
            "the engine that pinned the release is named rather than left anonymous: {rendered}"
        );
    }

    #[test]
    fn a_realignment_reports_what_the_runtime_could_do_before_it() {
        // The whole point of the line: the device_check printed afterwards is the
        // after state, so without this one a realignment that ends badly does not
        // say whether it broke a working runtime or repaired a broken one. Here it
        // repaired one, and the output says so without anyone visiting the machine.
        let rendered = render_torch_alignment(
            &TorchAlignment::Realigned {
                from: "2.11.0+rocm7.14.0".to_owned(),
                to: "2.11.0+rocm7.13.0".to_owned(),
            },
            "vllm",
            Some(&RuntimeDeviceCheck::KernelFailed {
                torch_version: "2.11.0+rocm7.14.0".to_owned(),
                error: "HIP error: hipErrorInvalidImage".to_owned(),
            }),
        );

        assert!(
            rendered.contains("before this replacement: kernel_failed (torch 2.11.0+rocm7.14.0)"),
            "the before verdict names itself and the torch it judged: {rendered}"
        );
        assert!(
            rendered.contains("device_check below is after it"),
            "the reader is told which of the two blocks is which: {rendered}"
        );
    }

    #[test]
    fn only_a_realignment_reports_a_before_verdict() {
        // Every other outcome left the runtime as it was, so the device_check
        // below is already about the same torch this block names. A before/after
        // pair there would invite reading a change into an outcome that made none.
        let before = RuntimeDeviceCheck::Usable {
            device_count: 1,
            torch_version: "2.11.0+rocm7.13.0".to_owned(),
        };

        for outcome in [
            TorchAlignment::AlreadyAligned {
                version: "2.11.0+rocm7.13.0".to_owned(),
            },
            TorchAlignment::Disabled {
                wanted: "2.11.0+rocm7.13.0".to_owned(),
                kept: "2.9.0+cpu".to_owned(),
            },
            TorchAlignment::Unavailable {
                wanted: "2.11.0+rocm7.13.0".to_owned(),
                kept: "2.11.0+gitd0c8b1f".to_owned(),
            },
        ] {
            let rendered = render_torch_alignment(&outcome, "vllm", Some(&before));
            assert!(
                !rendered.contains("before this replacement"),
                "{outcome:?} replaced nothing: {rendered}"
            );
        }
    }

    #[test]
    fn an_already_aligned_runtime_reports_the_version_it_kept() {
        // Every refresh after the first lands here, so this is the most frequently
        // printed of the five outcomes.
        let rendered = render_torch_alignment(
            &TorchAlignment::AlreadyAligned {
                version: "2.11.0+rocm7.13.0".to_owned(),
            },
            "vllm",
            None,
        );

        assert!(rendered.contains("  torch_alignment: already_aligned (2.11.0+rocm7.13.0)\n"));
    }

    #[test]
    fn a_managed_python_runtime_is_settled() {
        assert!(settles_runtime_torch("vllm", Some(true)));
        assert!(
            settles_runtime_torch("vllm", None),
            "an engine that does not report the field is still assumed managed, as before"
        );
    }

    #[test]
    fn an_external_runtime_is_left_alone() {
        assert!(!settles_runtime_torch("vllm", Some(false)));
    }

    #[test]
    fn an_engine_that_manages_its_own_runtime_is_left_alone() {
        // Lemonade reports `managed_env: Some(true)` — rocm-cli did create the
        // runtime — but its `python_executable` is the Lemonade binary, not an
        // interpreter, and no torch ever lived there. Settling it would spawn that
        // binary twice with a generated `.py` path as `argv[1]` and print torch and
        // device check blocks for a runtime that has neither, on the serve path.
        assert!(!settles_runtime_torch("lemonade", Some(true)));
    }

    #[test]
    fn an_already_correct_runtime_is_left_alone() {
        let plan = plan_torch_alignment(
            Some("rocm7.13.0"),
            Some("2.11.0+rocm7.13.0"),
            Some("torch==2.11.0+gitd0c8b1f"),
            "vllm",
        );

        assert_eq!(
            plan,
            TorchAlignmentPlan::AlreadyAligned {
                version: "2.11.0+rocm7.13.0".to_owned()
            }
        );
    }

    #[test]
    fn a_loose_torch_requirement_is_not_second_guessed() {
        // Without an exact pin there is no release to carry over, so the engine's
        // resolution stands rather than being overridden on a guess.
        let plan = plan_torch_alignment(
            Some("rocm7.13.0"),
            Some("2.11.0+gitd0c8b1f"),
            Some("torch>=2.10"),
            "vllm",
        );

        assert!(
            matches!(plan, TorchAlignmentPlan::NotApplicable(reason) if reason.contains("exact")),
            "a non-pinned requirement must be left alone"
        );
    }

    #[test]
    fn an_unidentified_sdk_build_is_not_acted_on() {
        // Nothing to carry over, so guessing a build would be worse than doing
        // nothing and letting the device check report the result.
        let plan = plan_torch_alignment(
            None,
            Some("2.11.0+gitd0c8b1f"),
            Some("torch==2.11.0+gitd0c8b1f"),
            "vllm",
        );

        assert!(matches!(plan, TorchAlignmentPlan::NotApplicable(_)));
    }

    /// The build comes from the manifest, not from whatever torch is installed.
    ///
    /// This is what makes a runtime already overwritten by the engine recoverable:
    /// reading the environment would call the engine's build the SDK's and leave the
    /// runtime broken for good. `plan_torch_alignment` takes the build as an opaque
    /// argument, so the property lives here, in the lookup that produces it.
    #[test]
    fn the_sdk_build_is_read_from_the_manifest_not_the_environment() {
        let mut manifest =
            test_runtime_manifest_for_update("wheel-gfx94x", "gfx94x", "gfx94x-dcgpu", "7.13.0");
        manifest.sdk_torch = Some("2.11.0+rocm7.13.0".to_owned());

        assert_eq!(
            sdk_torch_build_from_manifest(&manifest).as_deref(),
            Some("rocm7.13.0"),
            "the recorded SDK torch names the build"
        );
    }

    /// Manifests written before `sdk_torch` existed still have to yield a build.
    ///
    /// These are the runtimes already on real machines, so this fallback is the
    /// repair path rather than a nicety. TheRock names the build `rocm<version>`.
    #[test]
    fn a_manifest_without_a_recorded_torch_derives_the_build_from_the_sdk_version() {
        let manifest =
            test_runtime_manifest_for_update("wheel-gfx94x", "gfx94x", "gfx94x-dcgpu", "7.13.0");
        assert!(
            manifest.sdk_torch.is_none(),
            "this test is about the pre-change manifest shape"
        );

        assert_eq!(
            sdk_torch_build_from_manifest(&manifest).as_deref(),
            Some("rocm7.13.0")
        );
    }

    /// A dated alpha carries its date into the build, and must not be trimmed.
    ///
    /// Nightly SDKs version as `7.14.0a20260812`, and TheRock's torch for one is
    /// `+rocm7.14.0a20260812` — the date is part of the build, not decoration on
    /// the version. Shortening it to the `7.14.0` release would name a build that
    /// exists for a different SDK, so the alignment would install a torch built
    /// against libraries the runtime does not have. The fallback is a whole-string
    /// interpolation today; this pins that, because the tempting "clean up the
    /// version first" refactor is the one that breaks every nightly runtime.
    #[test]
    fn a_dated_alpha_sdk_keeps_its_date_in_the_derived_build() {
        let manifest = test_runtime_manifest_for_update(
            "wheel-gfx94x",
            "gfx94x",
            "gfx94x-dcgpu",
            "7.14.0a20260812",
        );
        assert!(
            manifest.sdk_torch.is_none(),
            "this test is about the pre-change manifest shape"
        );

        assert_eq!(
            sdk_torch_build_from_manifest(&manifest).as_deref(),
            Some("rocm7.14.0a20260812")
        );
    }

    /// A manifest that names no version at all must not invent a build.
    #[test]
    fn a_manifest_with_no_version_identifies_no_build() {
        let mut manifest =
            test_runtime_manifest_for_update("wheel-gfx94x", "gfx94x", "gfx94x-dcgpu", "  ");
        manifest.sdk_torch = None;

        assert_eq!(sdk_torch_build_from_manifest(&manifest), None);
    }

    /// A torch the user installed themselves is kept, and named as kept.
    ///
    /// The Python does not exist and an index is supplied, so an alignment that
    /// still ran would reach the install and come back `InstallFailed`. Getting
    /// the divergence described instead — the build that was not installed, and
    /// the one still there — is what shows the rewrite was skipped rather than
    /// attempted and then reported differently.
    #[test]
    fn disabling_alignment_keeps_the_installed_torch_and_says_what_it_declined() {
        let mut env = ScopedTestEnv::new();
        env.set("ROCM_CLI_DISABLE_TORCH_ALIGNMENT", "1");
        let probe = therock::TorchAlignmentProbe {
            installed_torch: Some("2.9.0+cpu".to_owned()),
            engine_requires_torch: Some("torch==2.11.0+gitd0c8b1f".to_owned()),
        };

        let outcome = align_runtime_torch(
            &test_app_paths(),
            Path::new("/nonexistent/python"),
            Some("https://example.invalid/simple"),
            Some("rocm7.13.0"),
            "vllm",
            Ok(&probe),
        );

        assert_eq!(
            outcome,
            TorchAlignment::Disabled {
                wanted: "2.11.0+rocm7.13.0".to_owned(),
                kept: "2.9.0+cpu".to_owned(),
            }
        );
    }

    /// Unset, the rewrite is attempted as usual.
    ///
    /// Paired with the test above so the opt-out cannot appear to work for an
    /// unrelated reason. The index is withheld here on purpose: it is the first
    /// thing read after the gate, so the call stops there rather than reaching a
    /// real install, and the outcome still tells the two paths apart.
    #[test]
    fn alignment_runs_unless_the_variable_is_set() {
        let mut env = ScopedTestEnv::new();
        env.clear("ROCM_CLI_DISABLE_TORCH_ALIGNMENT");
        let probe = therock::TorchAlignmentProbe {
            installed_torch: Some("2.9.0+cpu".to_owned()),
            engine_requires_torch: Some("torch==2.11.0+gitd0c8b1f".to_owned()),
        };

        let outcome = align_runtime_torch(
            &test_app_paths(),
            Path::new("/nonexistent/python"),
            None,
            Some("rocm7.13.0"),
            "vllm",
            Ok(&probe),
        );

        assert_eq!(
            outcome,
            TorchAlignment::NotApplicable(
                "the runtime manifest records no wheel index to install from".to_owned()
            ),
            "the opt-out must not fire when the variable is unset"
        );
    }

    /// Opting out of a rewrite that was never due reports the runtime as it is.
    ///
    /// `Disabled` says a replacement was declined. On a runtime already holding
    /// the SDK's build there was none to decline, and saying otherwise would send
    /// the reader looking for a torch that was spared when nothing was.
    #[test]
    fn disabling_alignment_over_an_aligned_runtime_still_reports_it_aligned() {
        let mut env = ScopedTestEnv::new();
        env.set("ROCM_CLI_DISABLE_TORCH_ALIGNMENT", "1");
        let probe = therock::TorchAlignmentProbe {
            installed_torch: Some("2.11.0+rocm7.13.0".to_owned()),
            engine_requires_torch: Some("torch==2.11.0+gitd0c8b1f".to_owned()),
        };

        let outcome = align_runtime_torch(
            &test_app_paths(),
            Path::new("/nonexistent/python"),
            Some("https://example.invalid/simple"),
            Some("rocm7.13.0"),
            "vllm",
            Ok(&probe),
        );

        assert_eq!(
            outcome,
            TorchAlignment::AlreadyAligned {
                version: "2.11.0+rocm7.13.0".to_owned(),
            }
        );
    }

    /// The kept torch is a deliberate divergence, and the remedy is not reinstall.
    ///
    /// The engine's pin is unsatisfied on purpose here, so reporting a violation
    /// would answer the user's instruction with an error — and the remedy that
    /// comes with it, `--reinstall`, is the one action that would undo the very
    /// decision they made.
    #[test]
    fn a_disabled_alignment_diverges_on_torch_without_offering_a_reinstall() {
        let outcome = TorchAlignment::Disabled {
            wanted: "2.11.0+rocm7.13.0".to_owned(),
            kept: "2.9.0+cpu".to_owned(),
        };
        assert_eq!(deliberately_diverged_package(&outcome), Some("torch"));

        let rendered = render_torch_alignment(&outcome, "vllm", None);
        assert!(
            rendered.contains("  torch_alignment: disabled\n"),
            "the block has to name the state, got {rendered:?}"
        );
        assert!(
            rendered.contains(
                "ROCM_CLI_DISABLE_TORCH_ALIGNMENT is set; keeping 2.9.0+cpu rather than installing 2.11.0+rocm7.13.0"
            ),
            "the block has to name the variable and both builds, got {rendered:?}"
        );

        let dependencies = classify_dependency_details(
            vec![
                "The package `vllm` requires `torch==2.11.0+gitd0c8b1f`, but `2.9.0+cpu` is installed".to_owned(),
            ],
            deliberately_diverged_package(&outcome),
        );
        assert!(matches!(
            dependencies,
            EngineDependencyCheck::ExpectedDivergence(_)
        ));
        let rendered = render_engine_dependency_check("vllm", &dependencies);
        assert!(
            !rendered.contains("--reinstall"),
            "the remedy would undo the opt-out, got {rendered:?}"
        );
    }

    /// Opting out suppresses the correction, not the diagnosis.
    ///
    /// The escape hatch exists because the SDK's build does not always work, so it
    /// cannot also become a way to make a runtime that does not work pass. The
    /// device check is taken over the torch the user kept and read exactly as it
    /// would be otherwise: a host with a GPU the runtime cannot open, or can open
    /// and then not run a kernel on, still fails the install.
    #[test]
    fn opting_out_still_fails_an_install_that_left_a_gpu_host_unable_to_serve() {
        let no_devices = RuntimeDeviceCheck::NoDevices {
            torch_version: "2.9.0+cpu".to_owned(),
            hip_version: "7.2.53211".to_owned(),
        };
        let kernel_failed = RuntimeDeviceCheck::KernelFailed {
            torch_version: "2.9.0+cpu".to_owned(),
            error: "AcceleratorError: device kernel image is invalid".to_owned(),
        };

        assert!(install_left_runtime_unusable(
            &no_devices,
            &HostGpu::Detected
        ));
        assert!(install_left_runtime_unusable(
            &kernel_failed,
            &HostGpu::Detected
        ));
    }

    /// On a machine with no GPU the same verdict is the correct answer.
    ///
    /// Someone running a CPU torch deliberately is the person this opt-out is for,
    /// and a runtime reporting no device there has not failed at anything.
    #[test]
    fn opting_out_on_a_host_with_no_gpu_leaves_the_install_successful() {
        let no_devices = RuntimeDeviceCheck::NoDevices {
            torch_version: "2.9.0+cpu".to_owned(),
            hip_version: "7.2.53211".to_owned(),
        };

        assert!(!install_left_runtime_unusable(
            &no_devices,
            &HostGpu::Absent
        ));
    }

    #[test]
    fn a_runtime_that_sees_no_devices_names_the_torch_that_cannot_use_the_sdk() {
        // The install succeeded and the engine's requirements are satisfied, so
        // every other surface reports this runtime healthy. Measured on MI300X:
        // this torch loads against a ROCm 7.13 SDK and enumerates nothing, and
        // the first symptom would otherwise be a serve failure.
        let rendered = render_runtime_device_check(&RuntimeDeviceCheck::NoDevices {
            torch_version: "2.11.0+gitd0c8b1f".to_owned(),
            hip_version: "7.2.53211".to_owned(),
        });

        assert!(rendered.contains("  device_check: no_devices\n"));
        assert!(rendered.contains("2.11.0+gitd0c8b1f"));
        assert!(rendered.contains("7.2.53211"));
        assert!(rendered.contains("Failed to infer device type"));
        // No remedy is offered on purpose: reinstalling to satisfy the engine's
        // pin is what produces this state.
        assert!(
            !rendered.contains("action:"),
            "a remedy that recreates the fault must not be suggested: {rendered}"
        );
    }

    #[test]
    fn a_usable_runtime_states_how_many_devices_it_found() {
        let rendered = render_runtime_device_check(&RuntimeDeviceCheck::Usable {
            device_count: 8,
            torch_version: "2.11.0+rocm7.13.0".to_owned(),
        });

        assert_eq!(
            rendered,
            "  device_check: usable (8 device(s), torch 2.11.0+rocm7.13.0)\n"
        );
    }

    #[test]
    fn a_device_check_without_an_interpreter_is_not_verified_not_healthy() {
        // Same rule as the dependency check: absence of an answer is never
        // allowed to read as a passing one.
        let outcome = runtime_device_check(None, &[]);

        assert_eq!(
            outcome,
            RuntimeDeviceCheck::NotVerified(
                "the runtime's Python environment could not be located".to_owned()
            )
        );
        assert!(render_runtime_device_check(&outcome).contains("not_verified"));
    }

    #[test]
    fn an_unlocatable_environment_is_reported_not_assumed_healthy() {
        // The engine install can fail before it reports its own interpreter. That path
        // still prints a block, and with no interpreter to check it must say so rather
        // than fall through to `satisfied`.
        let (root, paths) = test_paths("engine-dependency-no-python");

        let outcome = engine_dependency_check(&paths, "vllm", None, None);
        let _ = fs::remove_dir_all(root);

        assert_eq!(
            outcome,
            EngineDependencyCheck::NotVerified(
                "the runtime's Python environment could not be located".to_owned()
            )
        );
        assert!(
            render_engine_dependency_check("vllm", &outcome).contains("not_verified"),
            "the failure path still prints a dependency_check block"
        );
    }

    #[test]
    fn a_dependency_check_that_could_not_run_says_so_instead_of_claiming_health() {
        let rendered = render_engine_dependency_check(
            "vllm",
            &EngineDependencyCheck::NotVerified("uv binary is unavailable\nofflinehost".to_owned()),
        );

        assert!(rendered.contains("not_verified"));
        assert!(!rendered.contains("satisfied"));
        // A multi-line failure must stay on one reported line.
        assert_eq!(rendered.lines().count(), 1, "{rendered}");
    }
}
