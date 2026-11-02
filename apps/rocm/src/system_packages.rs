// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Host system-package installation for the SDK and engine installs: the
//! distro-aware OpenMPI, libatomic and libnuma plans, the decision whether to run
//! one given the caller's consent, and running its commands (`sudo` apt, dnf or
//! zypper) without a shell.
//!
//! Relocated from `runtime_install_finalize.rs` with no behavior change.
//!
//! A leaf: it uses only `std`, `anyhow` and `rocm_core`. The consent bundle for
//! `rocm install sdk` (`SdkInstallConsents`) stays in `runtime_install_finalize.rs`,
//! which passes the plain `system_packages` bool in. That module re-exports
//! `ensure_openmpi_for_vllm`, `ensure_libatomic_for_torch` and
//! `ensure_libnuma_for_torch` for `main.rs` and `engines_cmd.rs`, and
//! `driver_install.rs` reads `real_os_release` from here directly.

use std::process::{Command as ProcessCommand, Stdio};

use anyhow::{Context, Result, bail};

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

#[cfg(test)]
mod tests {
    #[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
    use std::fs;

    use super::*;
    use crate::chat::rocm_chat_tool_requested_args;
    use crate::providers;
    use crate::runtime_install_finalize::tests::consents_for_install_sdk_argv;
    #[cfg(all(feature = "e2e-test-hooks", target_os = "linux"))]
    use crate::tests::ScopedTestEnv;
    use crate::therock;

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
}
