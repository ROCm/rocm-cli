// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Putting the ROCm CLI on a machine that does not have it.
//!
//! The obvious approach — copy the binary we are running — is wrong, and
//! quietly so. It only works when both machines share an OS and CPU
//! architecture, and when they do not the copy still lands, still runs as a
//! file, and fails with something unhelpful at the first invocation.
//!
//! So provisioning never copies this machine's binary. It asks the remote to
//! fetch its own build, using the project's own installer, which already knows
//! how to detect a platform and verify what it downloads — and does all of that
//! *on the remote*, for the remote. Only if the remote cannot reach the release
//! host does this machine fetch on its behalf, and then it fetches an artifact
//! built for the remote's platform, not for ours.
//!
//! Which of those two applies is not guessed in advance. "Does this machine
//! have internet" has no reliable signal from the outside, so the remote install
//! is simply attempted, and only *that command* failing selects the fallback.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use super::bootstrap::{REMOTE_CLI_PATH, RemotePlatform};
use super::transport::Transport;

/// The installer, carried inside the binary rather than fetched.
///
/// The fallback path needs an installer on a machine that by definition cannot
/// download one, and pushing the copy we were built with also guarantees the
/// installer and the CLI driving it agree about artifact naming and verification.
const INSTALLER: &str = include_str!("../../../../install.sh");

/// Public location of the same installer, for the common path where the remote
/// fetches it itself.
const INSTALLER_URL: &str = "https://raw.githubusercontent.com/ROCm/rocm-cli/main/install.sh";

/// Where pushed files land on the remote. A dedicated directory so a failed run
/// leaves something obvious to clean up rather than litter in /tmp.
const REMOTE_STAGING: &str = "$HOME/.rocm/provision";

/// How the CLI got onto the remote, for reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Provisioned {
    /// The remote downloaded and installed its own build.
    RemoteInstaller,
    /// This machine fetched a build for the remote's platform and pushed it.
    PushedMatchedArtifact { asset: String },
}

/// Install the CLI on the remote and return how to invoke it.
pub(crate) fn install_cli(
    transport: &dyn Transport,
    target: &str,
    platform: Option<&RemotePlatform>,
    channel: &str,
) -> Result<(String, Provisioned)> {
    println!("  remote CLI: not found — installing it on {target} ...");

    match run_remote_installer(transport, channel) {
        Ok(()) => {
            verify_remote_cli(transport, target)?;
            println!("  remote CLI: installed by the remote itself");
            Ok((REMOTE_CLI_PATH.to_owned(), Provisioned::RemoteInstaller))
        }
        Err(remote_error) => {
            // Not a guess about connectivity: this specific command failed, so
            // fall back to fetching on the remote's behalf.
            println!("  remote CLI: the machine could not install it itself, fetching for it ...");
            let platform = platform.context(
                "cannot fetch a build for the remote because its OS and CPU architecture \
                 could not be determined, and installing a mismatched build would fail \
                 in a way that is hard to diagnose",
            )?;
            let asset = push_matched_artifact(transport, platform, channel).with_context(|| {
                format!(
                    "{target} could not install the CLI itself ({remote_error}), and \
                         fetching a matching build for it failed too"
                )
            })?;
            verify_remote_cli(transport, target)?;
            println!(
                "  remote CLI: installed from a build matching {}-{}",
                platform.os, platform.arch
            );
            Ok((
                REMOTE_CLI_PATH.to_owned(),
                Provisioned::PushedMatchedArtifact { asset },
            ))
        }
    }
}

/// Ask the remote to install its own build.
fn run_remote_installer(transport: &dyn Transport, channel: &str) -> Result<()> {
    let outcome = transport.exec(&remote_installer_command(channel))?;
    if !outcome.success {
        bail!("{}", outcome.stderr.trim());
    }
    Ok(())
}

fn remote_installer_command(channel: &str) -> String {
    // Piped to `sh` on the remote so the remote's own platform detection,
    // checksum and signature verification all run there, for it.
    format!(
        "curl -fsSL {INSTALLER_URL} | sh -s -- {}",
        super::shell_quote(channel)
    )
}

/// Fetch a build for the remote's platform on this machine, then push it.
fn push_matched_artifact(
    transport: &dyn Transport,
    platform: &RemotePlatform,
    channel: &str,
) -> Result<String> {
    let staging = tempdir_for_download()?;
    let asset = download_for(platform, channel, &staging)?;

    let remote_dir = REMOTE_STAGING;
    transport
        .run(&format!("mkdir -p {remote_dir}"))
        .context("failed to make a staging directory on the remote")?;

    // The archive travels with its checksum and, when present, its signature, so
    // the remote can repeat every check this machine made. Splitting the trust
    // chain across two machines must not shorten it.
    for (local, remote_name) in [
        (staging.join(&asset), asset.clone()),
        (
            staging.join(format!("{asset}.sha256")),
            format!("{asset}.sha256"),
        ),
        (staging.join(format!("{asset}.sig")), format!("{asset}.sig")),
    ] {
        if !local.exists() {
            continue;
        }
        transport
            .push_file(&local, &format!("{remote_dir}/{remote_name}"))
            .with_context(|| format!("failed to copy {remote_name} to the remote"))?;
    }

    let installer_path = staging.join("install.sh");
    std::fs::write(&installer_path, INSTALLER)
        .context("failed to stage the installer for copying")?;
    transport
        .push_file(&installer_path, &format!("{remote_dir}/install.sh"))
        .context("failed to copy the installer to the remote")?;

    let outcome = transport.exec(&format!(
        "ROCM_CLI_ARCHIVE={remote_dir}/{asset} sh {remote_dir}/install.sh {}",
        super::shell_quote(channel)
    ))?;
    if !outcome.success {
        bail!(
            "the remote rejected the build we fetched for it: {}",
            outcome.stderr.trim()
        );
    }
    let _ = std::fs::remove_dir_all(&staging);
    Ok(asset)
}

/// Run the installer here in download-only mode, targeting the remote's
/// platform, and return the artifact's file name.
fn download_for(
    platform: &RemotePlatform,
    channel: &str,
    into: &std::path::Path,
) -> Result<String> {
    let installer = into.join("install.sh");
    std::fs::create_dir_all(into)
        .with_context(|| format!("failed to create {}", into.display()))?;
    std::fs::write(&installer, INSTALLER).context("failed to stage the installer")?;

    let output = std::process::Command::new("sh")
        .arg(&installer)
        .arg(channel)
        .env("ROCM_CLI_DOWNLOAD_ONLY", "1")
        .env("ROCM_CLI_DOWNLOAD_DIR", into)
        .env("ROCM_CLI_TARGET_OS", &platform.os)
        .env("ROCM_CLI_TARGET_ARCH", &platform.arch)
        .output()
        .context("failed to run the installer to fetch a build for the remote")?;

    if !output.status.success() {
        bail!(
            "could not fetch a {}-{} build: {}",
            platform.os,
            platform.arch,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    parse_downloaded_asset(&String::from_utf8_lossy(&output.stdout))
        .context("the installer reported success but did not say which file it produced")
}

/// Pull the artifact name out of the installer's `downloaded:` line.
fn parse_downloaded_asset(stdout: &str) -> Option<String> {
    stdout
        .lines()
        .rev()
        .find_map(|line| line.trim().strip_prefix("downloaded:"))
        .map(str::trim)
        .and_then(|path| path.rsplit('/').next())
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
}

fn tempdir_for_download() -> Result<PathBuf> {
    let directory =
        std::env::temp_dir().join(format!("rocm-remote-provision-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory)
        .with_context(|| format!("failed to create {}", directory.display()))?;
    Ok(directory)
}

/// Confirm the freshly-installed CLI actually runs there.
///
/// The check that catches a build which landed but cannot execute — the failure
/// mode copying our own binary produced silently, and the reason this module
/// exists.
fn verify_remote_cli(transport: &dyn Transport, target: &str) -> Result<()> {
    let outcome = transport.exec(&format!("{REMOTE_CLI_PATH} --version"))?;
    if !outcome.success {
        bail!(
            "the CLI was installed on {target} but does not run there: {}\n\
             This usually means the build does not match the machine's OS or CPU.",
            outcome.stderr.trim()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::transport::{ScriptedStep, ScriptedTransport};

    #[test]
    fn the_remote_installs_its_own_build_when_it_can() {
        // The common path: the remote's own platform detection and verification
        // run on the remote, so nothing here has to reason about its hardware.
        let transport = ScriptedTransport::new(vec![
            ScriptedStep::ok("install.sh | sh", ""),
            ScriptedStep::ok(".local/bin/rocm --version", "rocm 1.2.3"),
        ]);
        let (cli, how) = install_cli(&transport, "gpu-box", None, "release").expect("provisioned");
        assert_eq!(cli, REMOTE_CLI_PATH);
        assert_eq!(how, Provisioned::RemoteInstaller);
    }

    #[test]
    fn a_build_that_lands_but_cannot_run_is_caught_and_explained() {
        // The exact failure that copying our own binary produced silently.
        let transport = ScriptedTransport::new(vec![
            ScriptedStep::ok("install.sh | sh", ""),
            ScriptedStep::fails(".local/bin/rocm --version", 126, "Exec format error"),
        ]);
        let error = install_cli(&transport, "gpu-box", None, "release")
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not run there"), "{error}");
        assert!(error.contains("OS or CPU"), "{error}");
    }

    #[test]
    fn an_unknown_remote_platform_refuses_rather_than_pushing_our_own_build() {
        // Without knowing the target's platform the only thing left to send is
        // this machine's binary, which is the mistake this module exists to
        // avoid. Refuse instead.
        let transport = ScriptedTransport::new(vec![ScriptedStep::fails(
            "install.sh | sh",
            1,
            "could not resolve host",
        )]);
        let error = install_cli(&transport, "gpu-box", None, "release")
            .unwrap_err()
            .to_string();
        assert!(error.contains("could not be determined"), "{error}");
    }

    #[test]
    fn the_channel_reaches_the_remote_installer_quoted() {
        assert!(remote_installer_command("nightly").contains("sh -s -- nightly"));
        // A channel is user input landing in a remote shell like any other.
        assert!(remote_installer_command("a; rm -rf /").contains(r"'a; rm -rf /'"));
    }

    #[test]
    fn the_downloaded_artifact_name_is_read_from_the_installers_own_report() {
        let stdout = "rocm-cli installer\n  channel: release\n\
                      downloaded: /tmp/x/rocm-cli-linux-amd64.tar.gz\n";
        assert_eq!(
            parse_downloaded_asset(stdout).as_deref(),
            Some("rocm-cli-linux-amd64.tar.gz")
        );
        assert_eq!(parse_downloaded_asset("no such line"), None);
    }

    #[test]
    fn the_installer_is_carried_in_the_binary_so_an_offline_remote_can_still_get_one() {
        // A machine that cannot download an artifact cannot download an
        // installer either, and shipping the one we were built with keeps the
        // installer and this code agreeing about artifact names and checks.
        assert!(INSTALLER.contains("rocm-cli installer"));
        assert!(
            INSTALLER.contains("ROCM_CLI_DOWNLOAD_ONLY"),
            "the embedded installer must be the one supporting download-only mode"
        );
        assert!(INSTALLER.contains("ROCM_CLI_ARCHIVE"));
    }
}
