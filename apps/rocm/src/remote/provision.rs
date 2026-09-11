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

    // An alternate signing key set in this machine's environment is an escape
    // hatch for private mirrors (see install.sh's resolve_public_keys). Forward
    // it to the remote's own install.sh, or the remote falls back to the pinned
    // production keys and rejects an archive this machine already trusted —
    // shortening the trust chain the comment above insists on not shortening.
    let signing_env = signing_env_fragment()?.unwrap_or_default();

    let outcome = transport.exec(&format!(
        "{signing_env}ROCM_CLI_ARCHIVE={} sh {remote_dir}/install.sh {}",
        super::shell_quote(&format!("{remote_dir}/{asset}")),
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

/// Build the `NAME=value ` fragment (quoted, trailing space) that forwards an
/// alternate signing key to the remote's install.sh, or `None` if this
/// process's environment sets neither variable.
///
/// Only `_PEM` ever carries a key across the wire. `_PATH` names a file on
/// *this* machine, and forwarding that path verbatim would tell the remote's
/// shell to open a file that is not there — the variable would be set but
/// useless. So a `_PATH` is read here and its *contents* are sent as `_PEM`.
///
/// When a key *is* forwarded, `_PATH` goes with it — deliberately empty — so
/// that the forwarded key wins. `resolve_public_keys` consults `_PATH` first,
/// so a value exported on the far side (through `/etc/environment` and pam_env,
/// which apply to non-interactive sshd sessions) would otherwise beat the key
/// we just sent, and the two machines would verify against different trust
/// roots. An explicitly-empty export reads as unset to install.sh's
/// `[ -n ... ]`, which is what makes one token enough to close that.
///
/// Note the scope: this guarantees *the key we send wins*, not that the
/// remote's own is always off. Forward nothing — the default, with neither
/// variable set here — and the fragment is empty, so a `_PATH` the remote
/// exports for itself still stands while this machine verifies against the
/// pinned keys. Blanking it unconditionally would close that too, at the cost
/// of overriding a remote operator's deliberate mirror-key config in the case
/// where we have no opinion at all. The divergence fails loud ("the remote
/// rejected the build we fetched for it") rather than silently trusting the
/// wrong root, so it is left as a decision rather than assumed.
///
/// `_PATH` is therefore checked first, because that is the order install.sh's
/// own `resolve_public_keys` uses. These two must not disagree: what they are
/// choosing between is the trust root a signature is verified against, so if
/// they picked differently, a remote provision would accept a build that a
/// local install would reject, and the mismatch would surface as "the remote
/// rejected the build we fetched for it" — pointing at the artifact rather
/// than at the key.
///
/// An empty value counts as unset, again matching install.sh, which tests
/// these with `[ -n ... ]`. Rust's `env::var` does not make that distinction
/// on its own: `FOO=` yields `Some("")`, which would otherwise forward an
/// empty key and silently discard the operator's real one.
fn signing_env_fragment() -> Result<Option<String>> {
    fn set_and_non_empty(name: &str) -> Option<String> {
        std::env::var(name).ok().filter(|value| !value.is_empty())
    }

    signing_env_fragment_from(
        set_and_non_empty("ROCM_CLI_SIGNING_PUBLIC_KEY_PEM"),
        set_and_non_empty("ROCM_CLI_SIGNING_PUBLIC_KEY_PATH"),
        |path: &str| std::fs::read_to_string(path),
    )
}

fn signing_env_fragment_from(
    pem_env: Option<String>,
    path_env: Option<String>,
    read_to_string: impl Fn(&str) -> std::io::Result<String>,
) -> Result<Option<String>> {
    // Defence in depth for callers that build these by hand rather than from
    // the environment: the empty-is-unset rule belongs to the resolution, not
    // to the one caller that happens to read env vars.
    let pem_env = pem_env.filter(|value| !value.is_empty());
    let path_env = path_env.filter(|value| !value.is_empty());

    let pem = match path_env {
        Some(path) => Some(read_to_string(&path).with_context(|| {
            format!(
                "failed to read the signing key at {path} \
                 (from ROCM_CLI_SIGNING_PUBLIC_KEY_PATH)"
            )
        })?),
        None => pem_env,
    };
    Ok(pem.map(|pem| {
        format!(
            "ROCM_CLI_SIGNING_PUBLIC_KEY_PEM={} ROCM_CLI_SIGNING_PUBLIC_KEY_PATH= ",
            super::shell_quote(&pem)
        )
    }))
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
    // A PID-keyed path under the shared, world-writable system temp directory
    // is predictable and PIDs get reused, so another local user could pre-stage
    // (or symlink) that exact path ahead of us; the old code then either wrote
    // the archive and signing material into whatever was already there, or had
    // its `remove_dir_all` above follow a planted symlink somewhere unintended.
    // Mixing in a nanosecond nonce makes the path unguessable, `create_dir`
    // (not `_all`) refuses to silently adopt an existing entry, and 0700 keeps
    // the contents unreadable to anyone else even if the name did leak.
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let directory = std::env::temp_dir().join(format!(
        "rocm-remote-provision-{}-{nonce}",
        std::process::id()
    ));
    create_restricted_dir(&directory)
        .with_context(|| format!("failed to create {}", directory.display()))?;
    Ok(directory)
}

/// Create `path` accessible only by its owner, with the mode applied *at*
/// creation.
///
/// What lands here is the release archive, its checksum and its signature — all
/// public material, so the point is not confidentiality. It is integrity, and
/// the window is narrower than it looks: `download_for` runs install.sh in
/// download-only mode, which fetches and verifies inside its *own* `mktemp -d`
/// and only then copies the proven artifacts out to here. So nothing in this
/// directory is waiting to be checked.
///
/// What it is waiting for is the push. Between landing and the `scp` above,
/// another local user able to write here could swap the archive together with a
/// matching `.sha256`, and the remote would re-verify the pair it was given. A
/// signature stops that — but install.sh only requires one on the release
/// channel with pinned keys present, or when a key is named explicitly
/// (`install.sh:418-431`), so on any other channel the checksum is the only
/// thing standing there. 0700 is what keeps a second local user out of the gap.
///
/// (The signing key itself never lands here. It travels in the command prefix,
/// and install.sh writes it into its own `mktemp -d` on the far side.)
///
/// Creating first and tightening afterwards leaves that window open for the
/// width of the umask. `DirBuilder::mode` closes it; this repo already uses the
/// same pattern in `dash.rs`'s `create_private_dir`, which documents why.
///
/// Deliberately not recursive: the name carries a nonce, so an existing
/// directory means someone else got there first and must be an error rather
/// than something to adopt.
#[cfg(unix)]
fn create_restricted_dir(path: &std::path::Path) -> Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new().mode(0o700).create(path)?;
    Ok(())
}

#[cfg(not(unix))]
fn create_restricted_dir(path: &std::path::Path) -> Result<()> {
    std::fs::create_dir(path)?;
    Ok(())
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

    #[test]
    fn neither_signing_var_set_forwards_nothing() {
        let fragment =
            signing_env_fragment_from(None, None, |path: &str| std::fs::read_to_string(path))
                .expect("no file read is attempted");
        assert_eq!(fragment, None);
    }

    #[test]
    fn an_explicit_pem_is_forwarded_as_is_when_it_is_the_only_one_set() {
        // The trailing empty `_PATH` is load-bearing, so it is asserted as part
        // of the whole fragment rather than trusted. Without it a key could be
        // forwarded correctly and still lose: resolve_public_keys reads `_PATH`
        // first, so a value the remote exports for itself — /etc/environment
        // via pam_env reaches non-interactive sshd sessions — would beat the
        // key we just sent. An explicitly-empty export is what install.sh's
        // `[ -n ... ]` reads as unset.
        let fragment = signing_env_fragment_from(Some("pem-content".to_owned()), None, |path| {
            panic!("must not read {path}: no _PATH was set")
        })
        .expect("no file read is attempted");
        assert_eq!(
            fragment.as_deref(),
            Some("ROCM_CLI_SIGNING_PUBLIC_KEY_PEM=pem-content ROCM_CLI_SIGNING_PUBLIC_KEY_PATH= ")
        );
    }

    #[test]
    fn a_path_wins_over_a_pem_because_that_is_what_install_sh_does() {
        // install.sh's resolve_public_keys returns _PATH first and only falls
        // through to _PEM. What the two are choosing between is the trust root
        // a signature is checked against, so disagreeing here would let a
        // remote provision verify against a different key than a local install
        // — and the operator would see it as a rejected artifact, not a key
        // mismatch. This test is the one that keeps the orders together.
        let fragment = signing_env_fragment_from(
            Some("pem-content".to_owned()),
            Some("/etc/rocm-signing.pem".to_owned()),
            |path| {
                assert_eq!(path, "/etc/rocm-signing.pem");
                Ok("path-content".to_owned())
            },
        )
        .expect("the fake reader succeeds");
        assert_eq!(
            fragment.as_deref(),
            Some("ROCM_CLI_SIGNING_PUBLIC_KEY_PEM=path-content ROCM_CLI_SIGNING_PUBLIC_KEY_PATH= ")
        );
    }

    #[test]
    fn an_empty_value_counts_as_unset_the_way_install_sh_reads_it() {
        // install.sh tests both with `[ -n ... ]`, so `FOO=` is unset to it.
        // Rust's env::var disagrees — it yields Some(""). Without this, an
        // empty _PEM alongside a real _PATH forwarded the empty one, throwing
        // away the operator's chosen trust root with no diagnostic and quietly
        // falling back to the pinned production keys.
        let fragment = signing_env_fragment_from(
            Some(String::new()),
            Some("/etc/rocm-signing.pem".to_owned()),
            |_| Ok("path-content".to_owned()),
        )
        .expect("the fake reader succeeds");
        assert_eq!(
            fragment.as_deref(),
            Some("ROCM_CLI_SIGNING_PUBLIC_KEY_PEM=path-content ROCM_CLI_SIGNING_PUBLIC_KEY_PATH= ")
        );

        // The mirror case: an empty _PATH must not be opened, and must not
        // suppress a real _PEM.
        let fragment = signing_env_fragment_from(
            Some("pem-content".to_owned()),
            Some(String::new()),
            |path| panic!("must not read {path:?}: an empty _PATH is unset"),
        )
        .expect("no file read is attempted");
        assert_eq!(
            fragment.as_deref(),
            Some("ROCM_CLI_SIGNING_PUBLIC_KEY_PEM=pem-content ROCM_CLI_SIGNING_PUBLIC_KEY_PATH= ")
        );

        // Both empty is the same as neither set.
        let fragment =
            signing_env_fragment_from(Some(String::new()), Some(String::new()), |path| {
                panic!("must not read {path:?}: an empty _PATH is unset")
            })
            .expect("no file read is attempted");
        assert_eq!(fragment, None);
    }

    #[test]
    fn a_path_is_read_locally_and_its_content_is_forwarded_not_the_path() {
        // The bug this guards against: forwarding _PATH verbatim names a file
        // on this machine, which is meaningless to the remote shell that runs
        // install.sh. Only file *content*, sent as _PEM, may cross the wire.
        let fragment =
            signing_env_fragment_from(None, Some("/etc/rocm-signing.pem".to_owned()), |path| {
                assert_eq!(path, "/etc/rocm-signing.pem");
                Ok("-----BEGIN PUBLIC KEY-----\nabc\n-----END PUBLIC KEY-----\n".to_owned())
            })
            .expect("the fake reader succeeds");
        let fragment = fragment.expect("a _PATH was set");
        assert!(fragment.starts_with("ROCM_CLI_SIGNING_PUBLIC_KEY_PEM="));
        assert!(!fragment.contains("/etc/rocm-signing.pem"));
        assert!(fragment.contains("BEGIN PUBLIC KEY"));
    }

    #[test]
    fn a_path_that_cannot_be_read_is_reported_rather_than_silently_dropped() {
        let error = signing_env_fragment_from(None, Some("/no/such/file".to_owned()), |_| {
            Err(std::io::Error::other("boom"))
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("/no/such/file"), "{error}");
    }
}
