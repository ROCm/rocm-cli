// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The standalone mise that provisions agent toolchains (PROTOTYPE).
//!
//! A host mise cannot be reused: a snap-packaged one (`/snap/bin/mise`) does
//! not run inside a sandbox, and any host mise brings the user's own config
//! and plugins. `rocm` downloads one pinned mise release into its data root
//! instead and checks it against a SHA-256 compiled into `rocm`, so neither
//! the release host nor a network attacker can substitute another binary.

use std::io::{Read, Write as _};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

/// The pinned mise release. Bump together with [`ASSETS`] and, after checking
/// `mise backends ls` of the new release, [`super::mise_snapshot::DISABLED_BACKENDS`].
pub(crate) const MISE_VERSION: &str = "2026.11.8";

/// `(target_arch, mise platform, SHA-256 of the release tarball)`, taken from
/// the release's `SHASUMS256.txt` and cross-checked against the asset digests
/// GitHub reports for it.
const ASSETS: &[(&str, &str, &str)] = &[
    (
        "x86_64",
        "linux-x64",
        "d2d5a46f507bba772686e2c8f2316b3a4439dac49a37a30db5c0e7bb077c5431",
    ),
    (
        "aarch64",
        "linux-arm64",
        "b36da3c4892380325d44dee3678ba618de191254d8a4aed8011ee66b28a6cbd7",
    ),
];
/// The binary's path inside the release tarball.
const ARCHIVE_ENTRY: &str = "mise/bin/mise";
/// A release tarball is ~60 MB; refuse to stream more than this.
const MAX_ARCHIVE_BYTES: u64 = 256 * 1024 * 1024;

/// Where the pinned mise lives under `tools_dir`, whether or not it exists yet.
pub(crate) fn path(tools_dir: &Path) -> PathBuf {
    tools_dir.join(format!("mise-{MISE_VERSION}")).join("mise")
}

/// Returns the pinned mise, downloading and verifying it first if needed.
pub(crate) fn ensure(tools_dir: &Path) -> Result<PathBuf> {
    let dest = path(tools_dir);
    if dest.is_file() {
        return Ok(dest);
    }
    if !cfg!(target_os = "linux") {
        bail!("toolchain provisioning is only supported on Linux");
    }
    let (_, platform, sha256) = ASSETS
        .iter()
        .find(|(arch, _, _)| *arch == std::env::consts::ARCH)
        .with_context(|| format!("no pinned mise release for {}", std::env::consts::ARCH))?;
    std::fs::create_dir_all(dest.parent().expect("has parent"))?;
    let _lock = rocm_core::FileLock::acquire(tools_dir.join(format!(".mise-{MISE_VERSION}.lock")))?;
    if dest.is_file() {
        return Ok(dest);
    }
    let url = format!(
        "https://github.com/jdx/mise/releases/download/v{MISE_VERSION}/mise-v{MISE_VERSION}-{platform}.tar.gz"
    );
    eprintln!("rocm: downloading mise {MISE_VERSION} from {url}");
    let spinner = crate::cli_progress::AnimatedSpinner::start(format!("mise {MISE_VERSION}"));
    let response = ureq::get(&url)
        .timeout(std::time::Duration::from_mins(10))
        .call()
        .with_context(|| format!("failed to download {url}"))?;
    let result = install_verified(response.into_reader(), sha256, &dest);
    drop(spinner);
    result?;
    eprintln!(
        "rocm: mise {MISE_VERSION} verified (sha256 {}…) and installed at {}",
        &sha256[..12],
        dest.display()
    );
    Ok(dest)
}

/// Streams `archive` to a `.part` file next to `dest` while hashing it,
/// refuses it unless the SHA-256 equals `expected_sha256`, and only then
/// extracts the mise binary to `dest`. Nothing is left at `dest` on failure.
fn install_verified(archive: impl Read, expected_sha256: &str, dest: &Path) -> Result<()> {
    let download = dest.with_extension("download.part");
    let binary = dest.with_extension("part");
    let result = verify_then_extract(archive, expected_sha256, &download, &binary).and_then(|()| {
        std::fs::rename(&binary, dest).with_context(|| format!("cannot install {}", dest.display()))
    });
    let _ = std::fs::remove_file(&download);
    if result.is_err() {
        let _ = std::fs::remove_file(&binary);
    }
    result
}

fn verify_then_extract(
    archive: impl Read,
    expected_sha256: &str,
    download: &Path,
    binary: &Path,
) -> Result<()> {
    let mut file = std::fs::File::create(download)?;
    let mut hasher = Sha256::new();
    let mut limited = archive.take(MAX_ARCHIVE_BYTES + 1);
    let mut buf = vec![0u8; 64 * 1024];
    let mut total = 0u64;
    loop {
        let n = limited.read(&mut buf)?;
        if n == 0 {
            break;
        }
        total += n as u64;
        if total > MAX_ARCHIVE_BYTES {
            bail!("mise archive is larger than {MAX_ARCHIVE_BYTES} bytes");
        }
        hasher.update(&buf[..n]);
        file.write_all(&buf[..n])?;
    }
    file.flush()?;
    drop(file);
    let actual = format!("{:x}", hasher.finalize());
    if actual != expected_sha256 {
        bail!(
            "mise archive checksum mismatch: expected sha256 {expected_sha256}, got {actual}; nothing was installed"
        );
    }
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(std::fs::File::open(download)?));
    for entry in tar.entries()? {
        let mut entry = entry?;
        if entry.path()?.as_ref() == Path::new(ARCHIVE_ENTRY)
            && entry.header().entry_type().is_file()
        {
            let mut out = std::fs::File::create(binary)?;
            std::io::copy(&mut entry, &mut out)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt as _;
                out.set_permissions(std::fs::Permissions::from_mode(0o755))?;
            }
            return Ok(());
        }
    }
    bail!("mise archive has no {ARCHIVE_ENTRY}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tarball(entry: &str, body: &[u8]) -> Vec<u8> {
        let mut builder = tar::Builder::new(flate2::write::GzEncoder::new(
            Vec::new(),
            flate2::Compression::fast(),
        ));
        let mut header = tar::Header::new_gnu();
        header.set_size(body.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder.append_data(&mut header, entry, body).unwrap();
        builder.into_inner().unwrap().finish().unwrap()
    }

    fn sha(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rocm-mise-bin-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_matching_archive_installs_the_binary() {
        let dir = scratch("ok");
        let archive = tarball(ARCHIVE_ENTRY, b"#!fake mise");
        let dest = dir.join("mise");
        install_verified(archive.as_slice(), &sha(&archive), &dest).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"#!fake mise");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_tampered_archive_is_refused_and_nothing_is_installed() {
        let dir = scratch("bad");
        let archive = tarball(ARCHIVE_ENTRY, b"#!fake mise");
        let pinned = sha(&archive);
        let tampered = tarball(ARCHIVE_ENTRY, b"#!evil mise");
        let dest = dir.join("mise");
        let e = format!(
            "{:#}",
            install_verified(tampered.as_slice(), &pinned, &dest).unwrap_err()
        );
        assert!(e.contains("checksum mismatch"), "{e}");
        assert!(!dest.exists());
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            0,
            "temp files left"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn pinned_assets_are_well_formed() {
        for (_, platform, sha256) in ASSETS {
            assert!(platform.starts_with("linux-"));
            assert_eq!(sha256.len(), 64);
            assert!(sha256.chars().all(|c| c.is_ascii_hexdigit()));
        }
    }
}
