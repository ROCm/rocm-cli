// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Torch alignment and retention for an engine install on a managed runtime:
//! deciding whether the SDK's torch must be realigned to the engine's pin or
//! kept as an intended divergence, running and reporting the realignment, and
//! reading the runtime's interpreter, library paths, index URL and torch build
//! from its manifest.
//!
//! Relocated from `runtime_install_finalize.rs` with no behavior change.
//!
//! A child of `runtime_install_finalize.rs`: it uses the parent's device check
//! (`RuntimeDeviceCheck`, `device_check_verdict`), which stays there for good.
//! As a crate-level sibling it would form a cycle with the parent, which calls
//! into it from `settle_engine_install` and `maybe_auto_install_sdk_preferred_engine`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Result;
use rocm_core::AppPaths;

use super::{RuntimeDeviceCheck, device_check_verdict};
use crate::engines_cmd::runtime_manifest_for_selector;
use crate::therock;
use crate::{record_cli_audit_event, sanitize_log_value};

/// The Python interpreter of the managed runtime `runtime_key`, when its manifest names
/// one. Used to check an environment whose engine install did not get far enough to
/// report its own interpreter.
pub(super) fn runtime_python_for_key(paths: &AppPaths, runtime_key: &str) -> Option<PathBuf> {
    let manifests = therock::load_runtime_manifests(paths).ok()?;
    let manifest = runtime_manifest_for_selector(&manifests, runtime_key)?;
    manifest.python_executable.as_deref().map(PathBuf::from)
}

/// The runtime's recorded ROCm library directories.
///
/// These are what a served process gets on its loader path, so a probe must use
/// them too or it will judge a healthy runtime unusable.
pub(super) fn runtime_library_paths_for_key(paths: &AppPaths, runtime_key: &str) -> Vec<PathBuf> {
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
pub(super) enum TorchAlignment {
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
pub(super) fn report_torch_alignment(
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
pub(super) const fn deliberately_diverged_package(
    outcome: &TorchAlignment,
) -> Option<&'static str> {
    match outcome {
        TorchAlignment::Realigned { .. }
        | TorchAlignment::AlreadyAligned { .. }
        | TorchAlignment::Disabled { .. } => Some("torch"),
        TorchAlignment::Unavailable { .. }
        | TorchAlignment::InstallFailed { .. }
        | TorchAlignment::NotApplicable(_) => None,
    }
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
pub(super) fn sdk_torch_build_for_key(paths: &AppPaths, runtime_key: &str) -> Option<String> {
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
pub(super) enum TorchRetention {
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
pub(super) fn classify_retained_torch(
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
pub(super) const fn retained_diverged_package(retention: &TorchRetention) -> Option<&'static str> {
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
pub(super) fn report_torch_retention(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_install_finalize::{
        EngineDependencyCheck, classify_dependency_details, render_engine_dependency_check,
    };
    use crate::tests::{ScopedTestEnv, test_app_paths, test_runtime_manifest_for_update};

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
}
