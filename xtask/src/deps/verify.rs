// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `cargo xtask deps verify-downloads`: re-hash every binary
//! `runtimes/mise.lock` records.
//!
//! Lemonade publishes no build attestation, so its digests come from mise's
//! own version index (`checksum_source = "mise-versions"`), and that digest
//! is the only check the Lemonade adapter applies to a download. mise locks
//! uv with `provenance = "github-attestations"`, but that string is written
//! by the weekly update's job that runs upstream code, so nothing here takes
//! it on trust. This command downloads each asset from its locked URL and
//! compares its SHA-256 with the lock, so a digest an update changes is
//! checked against the real release asset.

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use sha2::{Digest as _, Sha256};

use super::mise_lock::{self, LockedAsset, MISE_LOCK};

/// Fetches a URL to a file.
pub(crate) trait Fetch {
    fn fetch(&self, url: &str, destination: &Path) -> Result<()>;
}

struct Http;

impl Fetch for Http {
    fn fetch(&self, url: &str, destination: &Path) -> Result<()> {
        rocm_core::download_file_to_path(url, destination, Duration::from_mins(10))
            .with_context(|| format!("failed to download {url}"))
    }
}

pub(crate) fn run() -> Result<()> {
    let root = crate::paths::workspace_root()?;
    let work = crate::paths::target_dir(&root).join("deps-verify");
    let problems = verify(&root, &work, &Http)?;
    if problems.is_empty() {
        return Ok(());
    }
    for problem in &problems {
        eprintln!("  - {problem}");
    }
    bail!(
        "{} locked download(s) do not match {MISE_LOCK}; do not commit this lock",
        problems.len()
    )
}

/// Every locked asset, re-hashed; one message per mismatch. A download that
/// fails is an error, never a pass.
fn verify(root: &Path, work: &Path, fetch: &dyn Fetch) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(root.join(MISE_LOCK))
        .with_context(|| format!("failed to read {MISE_LOCK}"))?;
    let tools = mise_lock::parse_lock(&text)?;
    std::fs::create_dir_all(work)?;
    let mut problems = Vec::new();
    for (name, tool) in &tools {
        for (platform, asset) in &tool.platforms {
            let file = work.join(format!("{}-{platform}", tool.version));
            let actual = fetched_sha256(fetch, asset, &file)?;
            if actual == asset.sha256 {
                eprintln!(
                    "{name}@{} {platform}: digest matches the download",
                    tool.version
                );
            } else {
                problems.push(format!(
                    "{name}@{} {platform}: {} has sha256 {actual}, but the lock records {}",
                    tool.version, asset.url, asset.sha256
                ));
            }
        }
    }
    Ok(problems)
}

fn fetched_sha256(fetch: &dyn Fetch, asset: &LockedAsset, file: &Path) -> Result<String> {
    let _ = std::fs::remove_file(file);
    fetch.fetch(&asset.url, file)?;
    let bytes =
        std::fs::read(file).with_context(|| format!("failed to read {}", file.display()))?;
    let _ = std::fs::remove_file(file);
    Ok(sha256_hex(&bytes))
}

/// Lower-case hex SHA-256, as `mise.lock` records it.
fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .fold(String::new(), |mut hex, byte| {
            use std::fmt::Write as _;
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    use super::*;

    /// Serves fixed bytes per URL and records what was fetched.
    struct Fake {
        bodies: BTreeMap<String, Vec<u8>>,
        fetched: RefCell<Vec<String>>,
    }

    impl Fetch for Fake {
        fn fetch(&self, url: &str, destination: &Path) -> Result<()> {
            self.fetched.borrow_mut().push(url.to_owned());
            let body = self
                .bodies
                .get(url)
                .with_context(|| format!("no such url {url}"))?;
            std::fs::write(destination, body)?;
            Ok(())
        }
    }

    fn hex(bytes: &[u8]) -> String {
        sha256_hex(bytes)
    }

    const UV: &str =
        "https://github.com/astral-sh/uv/releases/download/0.1/uv-x86_64-unknown-linux-gnu.tar.gz";
    const LINUX: &str = "https://github.com/o/tool/releases/download/v1/tool-1-ubuntu-x64.tar.gz";

    /// uv locked with its attestation string and the digest `uv_sum`,
    /// and a Lemonade-like tool locked with `lemonade_sum`.
    fn workspace_with(uv_sum: &str, lemonade_sum: &str) -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("runtimes")).unwrap();
        let lock = format!(
            "lockfile_version = 3\n\n\
             [[tools.\"aqua:astral-sh/uv\"]]\nversion = \"0.1\"\nbackend = \"aqua:astral-sh/uv\"\n\n\
             [tools.\"aqua:astral-sh/uv\".\"platforms.linux-x64\"]\n\
             checksum = \"sha256:{uv_sum}\"\nurl = \"{UV}\"\nprovenance = \"github-attestations\"\n\n\
             [[tools.\"github:o/tool\"]]\nversion = \"1\"\nbackend = \"github:o/tool\"\n\n\
             [tools.\"github:o/tool\".\"platforms.linux-x64\"]\n\
             checksum = \"sha256:{lemonade_sum}\"\nchecksum_source = \"mise-versions\"\nurl = \"{LINUX}\"\n"
        );
        std::fs::write(root.path().join(MISE_LOCK), lock).unwrap();
        root
    }

    fn workspace(lemonade_sum: &str) -> tempfile::TempDir {
        workspace_with(&hex(b"the uv archive"), lemonade_sum)
    }

    fn fake() -> Fake {
        Fake {
            bodies: BTreeMap::from([
                (UV.to_owned(), b"the uv archive".to_vec()),
                (LINUX.to_owned(), b"the real archive".to_vec()),
            ]),
            fetched: RefCell::default(),
        }
    }

    #[test]
    fn every_locked_download_is_rehashed_and_matching_digests_pass() {
        let root = workspace(&hex(b"the real archive"));
        let fake = fake();
        let found = verify(root.path(), &root.path().join("work"), &fake).unwrap();
        assert_eq!(found, Vec::<String>::new());
        assert_eq!(*fake.fetched.borrow(), [UV, LINUX]);
    }

    #[test]
    fn a_digest_that_does_not_match_the_download_is_reported() {
        let recorded = "b".repeat(64);
        let root = workspace(&recorded);
        let found = verify(root.path(), &root.path().join("work"), &fake()).unwrap();
        assert_eq!(
            found,
            [format!(
                "github:o/tool@1 linux-x64: {LINUX} has sha256 {}, but the lock records {recorded}",
                hex(b"the real archive")
            )]
        );
    }

    /// The `provenance` string is written by the job that ran upstream code,
    /// so it excuses nothing: uv locked "with its attestation" but with a
    /// swapped digest is reported like any other mismatch.
    #[test]
    fn a_provenance_string_never_skips_the_rehash() {
        let swapped = "c".repeat(64);
        let root = workspace_with(&swapped, &hex(b"the real archive"));
        let found = verify(root.path(), &root.path().join("work"), &fake()).unwrap();
        assert_eq!(
            found,
            [format!(
                "aqua:astral-sh/uv@0.1 linux-x64: {UV} has sha256 {}, but the lock records {swapped}",
                hex(b"the uv archive")
            )]
        );
    }

    #[test]
    fn a_failed_download_is_an_error_not_a_pass() {
        let root = workspace(&hex(b"the real archive"));
        let nothing = Fake {
            bodies: BTreeMap::new(),
            fetched: RefCell::default(),
        };
        assert!(verify(root.path(), &root.path().join("work"), &nothing).is_err());
    }
}
