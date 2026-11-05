// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! SDK-install finalization: settling an engine onto a runtime after an SDK
//! install, torch alignment, engine-dependency checks, and device checks.
//!
//! Mechanically relocated from `main.rs` with no behavior change.
//!
//! Called from `install_cmd.rs`'s `install()` (the SDK-install path) and from
//! `engines_cmd.rs` (engine install).
//!
//! It calls into the sibling `runtimes_cmd.rs` for runtime activation;
//! `runtimes_cmd.rs` reaches back only from its own test module.
//!
//! The host system-package installation (`ensure_openmpi_for_vllm`,
//! `ensure_libatomic_for_torch`, `ensure_libnuma_for_torch`) lives in
//! `system_packages.rs`; this module re-exports those three names for
//! `install_cmd.rs` and `engines_cmd.rs`.
//!
//! `SdkInstallConsents` stays `pub(crate)` and is read from `install()`;
//! `settle_engine_install` is also `pub(crate)` and read from `engines_cmd.rs`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Result;
use rocm_core::{
    AppPaths, ExamineSummary, RocmCliConfig, preferred_serve_engine_for_host_gpu_summary,
};
use rocm_engine_protocol::{EngineMethod, InstallRequest, InstallResponse};

use crate::engine_install_env::{
    engine_manages_own_runtime, env_root_for_engine_install, runtime_key_for_python,
};
use crate::engine_ipc::engine_request_with_env_root;
use crate::runtimes_cmd::{
    activate_runtime, newest_installed_runtime_manifest, recover_setup_runtime_registration,
};
pub(crate) use crate::system_packages::{
    ensure_libatomic_for_torch, ensure_libnuma_for_torch, ensure_openmpi_for_vllm,
};
use crate::therock;
use crate::{ROLLBACK_RECOVERY_HINT, record_cli_audit_event, runtime_services, sanitize_log_value};

mod torch_alignment;

use torch_alignment::{
    TorchAlignment, TorchRetention, classify_retained_torch, deliberately_diverged_package,
    report_torch_alignment, report_torch_retention, retained_diverged_package,
    runtime_library_paths_for_key, runtime_python_for_key, sdk_torch_build_for_key,
};

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
        NEW_RUNTIME_KEY, OLD_RUNTIME_KEY, plant_service_record_on_runtime, test_paths,
    };
    use anyhow::Context as _;
    use clap::Parser as _;
    use rocm_core::load_recent_audit_events;

    use crate::{Cli, Command, InstallTarget};

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
    fn sdk_install_auto_engine_selection_prefers_vllm_for_supported_families() {
        // vLLM is unsupported on native Windows, so the SDK family preference is gated
        // off there and resolves to None.
        let expected = if cfg!(windows) { None } else { Some("vllm") };
        assert_eq!(preferred_engine_for_sdk_family("gfx90a"), expected);
        assert_eq!(preferred_engine_for_sdk_family("gfx94X-dcgpu"), expected);
        assert_eq!(preferred_engine_for_sdk_family("gfx120X-all"), None);
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
