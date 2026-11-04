// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm install`: the command handler (`install`, which handles the `sdk` and
//! `driver` targets), the `--version`/`--build-date` selector parsing and
//! display it uses (`therock_install_version_selector`,
//! `therock_install_version_selector_display`), and `sdk_install_request`,
//! which maps the parsed `install sdk` arguments onto a
//! `therock::SdkInstallRequest`.
//!
//! Mechanically relocated from `main.rs` (and, for `sdk_install_request`, from
//! `chat.rs`) with no behavior change; see `docs/architecture.md`.
//! `InstallTarget` and `InstallFormat`, the clap types `install` matches on,
//! stay in `main.rs` with the rest of the `Command`/`Cli` definition.
//! `dispatch()` calls `install` through the re-import `use
//! crate::install_cmd::{...};`. `therock_install_version_selector` is also
//! `pub(crate)`: `chat.rs`'s `render_install_sdk_dry_run_for_args` calls it as
//! `crate::therock_install_version_selector` through that re-import, which is
//! why `sdk_install_request` moved here rather than staying in `chat.rs` —
//! this module does not depend on `chat.rs`. `sdk_install_request` keeps the
//! `pub(crate)` it had in `chat.rs`, although only `install` and this module's
//! tests call it now; `therock_install_version_selector_display` is private.
//!
//! The two `sdk_install_request` regression tests moved with it out of
//! `chat.rs`'s `mod tests`.

use std::path::PathBuf;

use anyhow::{Result, bail};
use rocm_core::AppPaths;

use crate::cli_audit::record_cli_audit_event;
use crate::driver_install::{install_driver, reconcile_driver_install};
use crate::runtime_install_finalize::{
    SdkInstallConsents, ensure_libatomic_for_torch, ensure_libnuma_for_torch,
    finalize_successful_sdk_install, finish_sdk_install, maybe_auto_install_sdk_preferred_engine,
    print_sdk_install_success,
};
use crate::{InstallFormat, InstallTarget, therock};

pub(crate) fn therock_install_version_selector(
    version: Option<String>,
    build_date: Option<String>,
) -> Result<Option<therock::RuntimeVersionSelector>> {
    match (version, build_date) {
        (Some(_), Some(_)) => bail!("use either --version or --build-date, not both"),
        (Some(version), None) => Ok(Some(therock::RuntimeVersionSelector::version(version)?)),
        (None, Some(build_date)) => Ok(Some(therock::RuntimeVersionSelector::build_date(
            build_date,
        )?)),
        (None, None) => Ok(None),
    }
}

fn therock_install_version_selector_display(selector: &therock::RuntimeVersionSelector) -> String {
    match selector {
        therock::RuntimeVersionSelector::Version(version) => format!("version:{version}"),
        therock::RuntimeVersionSelector::BuildDate(date) => format!("build-date:{date}"),
    }
}

/// Map the parsed `install sdk` arguments onto the install request.
///
/// Extracted from the command arm so this mapping has a seam. `include_devel`
/// is the field that most needs one: nothing downstream re-derives it, so if
/// the flag stopped being forwarded here the install would silently go back to
/// pulling the compiler toolchain and every other assertion would still pass.
#[allow(clippy::too_many_arguments)]
pub(crate) const fn sdk_install_request<'a>(
    channel: &'a str,
    format: &'a str,
    prefix: Option<PathBuf>,
    version_selector: Option<therock::RuntimeVersionSelector>,
    family_override: Option<&'a str>,
    dry_run: bool,
    devel: bool,
    consent: therock::SdkInstallConsent,
) -> therock::SdkInstallRequest<'a> {
    therock::SdkInstallRequest {
        channel,
        format,
        prefix,
        version_selector,
        family_override,
        dry_run,
        include_devel: devel,
        consent,
    }
}

pub(crate) fn install(target: InstallTarget) -> Result<()> {
    let paths = AppPaths::discover()?;
    match target {
        InstallTarget::Sdk {
            channel,
            format,
            prefix,
            devel,
            version,
            build_date,
            family,
            dry_run,
            yes,
            approve_replacing_active_default,
        } => {
            let consents = SdkInstallConsents::resolve(yes, approve_replacing_active_default);
            let format_name = match format {
                InstallFormat::Wheel => "wheel",
                InstallFormat::Tarball => "tarball",
            };
            let version_selector = therock_install_version_selector(version, build_date)?;
            let version_selector_display = version_selector.as_ref().map_or_else(
                || "latest-compatible".to_owned(),
                therock_install_version_selector_display,
            );
            let prefix_display = prefix
                .as_ref()
                .map_or_else(|| "<managed>".to_owned(), |path| path.display().to_string());
            match therock::install_sdk(
                &paths,
                sdk_install_request(
                    &channel,
                    format_name,
                    prefix,
                    version_selector,
                    family.as_deref(),
                    dry_run,
                    devel,
                    consents.replace_active_default,
                ),
            ) {
                Ok(result) => {
                    let therock::SdkInstallResult { output, mutated } = result;
                    let finalized = if mutated {
                        finalize_successful_sdk_install(&paths)?
                    } else {
                        None
                    };
                    print!("{output}");
                    if let Some(finalized) = &finalized {
                        print_sdk_install_success(finalized);
                        // The SDK runtime wheel bundles PyTorch, whose ROCm build
                        // links against libatomic.so.1 and the system numactl
                        // runtime (libnuma.so.1 / libnuma_1.2). Ensure both are
                        // present for every SDK install, independent of which
                        // engine (if any) is auto-installed below.
                        ensure_libatomic_for_torch(consents.system_packages);
                        ensure_libnuma_for_torch(consents.system_packages);
                    }
                    finish_sdk_install(
                        &paths,
                        finalized.as_ref(),
                        if dry_run {
                            "install_sdk_dry_run"
                        } else {
                            "install_sdk"
                        },
                        {
                            // A real install that did not mutate the system was
                            // declined at the approval prompt; recording it as
                            // "completed" would lie in the audit trail. Dry-run
                            // never mutates but legitimately completes a preview.
                            let status = if dry_run || mutated {
                                "completed"
                            } else {
                                "cancelled"
                            };
                            format!(
                                "sdk install {status} channel={channel} format={format_name} prefix={prefix_display} version_selector={version_selector_display} dry_run={dry_run}"
                            )
                        },
                        |paths, finalized| {
                            maybe_auto_install_sdk_preferred_engine(
                                paths,
                                finalized,
                                consents.system_packages,
                            )
                        },
                    )?;
                }
                Err(error) => {
                    record_cli_audit_event(
                        &paths,
                        "runtime",
                        if dry_run {
                            "install_sdk_dry_run"
                        } else {
                            "install_sdk"
                        },
                        "error",
                        format!(
                            "sdk install failed channel={channel} format={format_name} prefix={prefix_display} version_selector={version_selector_display} dry_run={dry_run}: {error}"
                        ),
                        None,
                    );
                    return Err(error);
                }
            }
        }
        InstallTarget::Driver {
            dkms,
            yes,
            dry_run,
            reconcile,
        } => {
            if reconcile {
                if dkms || yes || dry_run {
                    bail!(
                        "`rocm install driver --reconcile` cannot be combined with --dkms, --yes, or --dry-run"
                    );
                }
                match reconcile_driver_install(&paths) {
                    Ok(output) => {
                        print!("{output}");
                        record_cli_audit_event(
                            &paths,
                            "driver",
                            "install_driver_reconcile",
                            "info",
                            "driver install state reconciled",
                            None,
                        );
                    }
                    Err(error) => {
                        record_cli_audit_event(
                            &paths,
                            "driver",
                            "install_driver_reconcile",
                            "error",
                            format!("driver install reconciliation failed: {error}"),
                            None,
                        );
                        return Err(error);
                    }
                }
                return Ok(());
            }
            match install_driver(&paths, dkms, yes, dry_run) {
                Ok(result) => {
                    print!("{}", result.output);
                    record_cli_audit_event(
                        &paths,
                        "driver",
                        if result.executed {
                            "install_driver_execute"
                        } else {
                            "install_driver_plan"
                        },
                        "info",
                        format!("driver install handled dkms={dkms} yes={yes} dry_run={dry_run}"),
                        None,
                    );
                }
                Err(error) => {
                    let executed = error.executed;
                    let error = error.source;
                    record_cli_audit_event(
                        &paths,
                        "driver",
                        if executed {
                            "install_driver_execute"
                        } else {
                            "install_driver_plan"
                        },
                        "error",
                        format!(
                            "driver install failed dkms={dkms} yes={yes} dry_run={dry_run}: {error}"
                        ),
                        None,
                    );
                    return Err(error);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;
    use crate::{Cli, Command};

    /// The half `main.rs`'s
    /// `install_sdk_devel_flag_defaults_off_and_wires_through_when_passed` cannot reach: that the parsed flag is what
    /// the install request carries.
    ///
    /// `install()` is not callable here — it needs `uv`, a live index and a real
    /// probe — so the mapping is extracted into `sdk_install_request` and pinned
    /// directly. Hardcoding `include_devel` at that mapping is the change this
    /// catches and the clap test does not.
    #[test]
    fn install_sdk_request_forwards_the_parsed_devel_flag() {
        for devel in [false, true] {
            let request = sdk_install_request(
                "release",
                "wheel",
                None,
                None,
                None,
                false,
                devel,
                therock::SdkInstallConsent::Ask,
            );
            assert_eq!(
                request.include_devel, devel,
                "the parsed --devel flag must reach the install request"
            );
        }

        // And the flag must not be confused with the neighbouring bool.
        let dry_run_only = sdk_install_request(
            "release",
            "wheel",
            None,
            None,
            None,
            true,
            false,
            therock::SdkInstallConsent::Ask,
        );
        assert!(dry_run_only.dry_run);
        assert!(!dry_run_only.include_devel);
    }

    /// End to end across the two seams a `rocm install sdk --devel` traverses:
    /// clap parse, then the request mapping. Neither alone proves the flag
    /// survives the trip.
    #[test]
    fn parsed_install_sdk_arguments_reach_the_request_with_devel_intact() {
        for (args, expected) in [
            (vec!["rocm", "install", "sdk"], false),
            (vec!["rocm", "install", "sdk", "--devel"], true),
        ] {
            let cli = Cli::try_parse_from(&args).expect("install sdk should parse");
            let Some(Command::Install {
                target: InstallTarget::Sdk { devel, .. },
            }) = cli.command
            else {
                panic!("expected an install sdk target for {args:?}");
            };
            let request = sdk_install_request(
                "release",
                "wheel",
                None,
                None,
                None,
                false,
                devel,
                therock::SdkInstallConsent::Ask,
            );
            assert_eq!(
                request.include_devel, expected,
                "--devel did not survive parse -> request for {args:?}"
            );
        }
    }
}
