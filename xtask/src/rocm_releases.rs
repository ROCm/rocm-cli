// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Resolves the self-hosted E2E `sdk_version` matrix labels
//! (`current` / `n-1` / `n-2`) to concrete TheRock package versions by reading
//! the release channel's own package indexes at run time (ROCMAI-430).
//!
//! The release channel is published in two layouts: ROCm 7.x only on the
//! canonical index, ROCm 10 and later only on the next layout's index (see
//! `THEROCK_NEXT_MIN_MAJOR` in `apps/rocm/src/therock/source.rs`). The three
//! newest final releases are therefore taken from the union of both listings,
//! so `n-1` can sit on one host and `n-2` on the other. The pins come from the
//! same indexes the pre-warm installs from, so nothing here needs a hand edit
//! when a release ships. The listing is read as plain text with no check of
//! wheel platform or Python tag, so a version published for only some platforms
//! would still be pinned.
//!
//! A failed fetch, or fewer than three final releases, is an error. There is
//! no fallback to a remembered list: a stale pin is the bug this replaced.

use anyhow::{Context, Result, bail};
use regex::Regex;
use semver::Version;
use std::process::Command;
use std::sync::OnceLock;

/// The `rocm` package's file listing on each layout's release index.
const RELEASE_INDEX_URLS: [&str; 2] = [
    "https://repo.amd.com/rocm/whl-multi-arch/rocm/",
    "https://stable.repo.amd.com/rocm/whl-next/rocm/",
];

/// Seconds `curl` may spend on one listing before the leg fails.
const FETCH_TIMEOUT_SECS: &str = "60";

/// One of the three legs of the self-hosted E2E `sdk_version` matrix.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum ReleaseLabel {
    Current,
    #[value(name = "n-1")]
    NMinus1,
    #[value(name = "n-2")]
    NMinus2,
}

impl ReleaseLabel {
    /// How many releases back from the newest this label pins.
    const fn offset(self) -> usize {
        match self {
            Self::Current => 0,
            Self::NMinus1 => 1,
            Self::NMinus2 => 2,
        }
    }

    /// The TheRock package version to pin `e2e-prewarm --version` to, read from
    /// the live release indexes.
    ///
    /// `Current` resolves to `None` without touching the network: it means
    /// "track the channel's latest", the pre-warm's default when no pin is
    /// given, so the common (non-matrix) path never depends on this lookup.
    pub fn pin(self) -> Result<Option<String>> {
        self.pin_with(curl_listing)
    }

    /// [`Self::pin`] with the listing fetch injected, so tests need no network.
    fn pin_with(self, fetch: impl Fn(&str) -> Result<String>) -> Result<Option<String>> {
        if matches!(self, Self::Current) {
            return Ok(None);
        }
        let mut listings = Vec::with_capacity(RELEASE_INDEX_URLS.len());
        for url in RELEASE_INDEX_URLS {
            listings.push(fetch(url).with_context(|| format!("failed to read {url}"))?);
        }
        let releases = newest_final_releases(&listings);
        let Some(version) = releases.get(self.offset()) else {
            bail!(
                "{self:?} needs the {} newest final ROCm releases, but the release indexes list only {}: {}",
                self.offset() + 1,
                releases.len(),
                releases.join(", ")
            );
        };
        Ok(Some(version.clone()))
    }
}

/// Every distinct final release named in the listings, newest first.
fn newest_final_releases(listings: &[String]) -> Vec<String> {
    static PACKAGE_FILE: OnceLock<Regex> = OnceLock::new();
    // `rocm-<version>` as a whole file name. The leading class rejects names
    // that merely end in `rocm` (`foo-rocm-9.9.9`), and a digit must follow the
    // dash, which rejects `rocm-sdk-core-…`. An optional percent-encoded local
    // suffix (`%2Bgfx942`) is allowed after the version. The trailing
    // alternation keeps only plain `X.Y.Z`: pre-releases (`7.13.0a20260605`) and
    // post-releases (`7.14.0.post1`) leave no `-`/`.tar`/`.whl` after the
    // digits. Post-releases are skipped on purpose: they patch a release the
    // list already names, and counting one would make n-2 the same release as n-1.
    let package_file = PACKAGE_FILE.get_or_init(|| {
        Regex::new(r"(?:^|[^A-Za-z0-9_-])rocm-(\d+\.\d+\.\d+)(?:%2[Bb][^\s/\x22<>-]*)?(?:-|\.tar\.gz|\.whl)")
            .expect("static regex is valid")
    });
    let mut versions: Vec<Version> = listings
        .iter()
        .flat_map(|listing| package_file.captures_iter(listing))
        .filter_map(|caps| Version::parse(&caps[1]).ok())
        .collect();
    versions.sort_unstable_by(|a, b| b.cmp(a));
    versions.dedup();
    versions.iter().map(Version::to_string).collect()
}

/// Fetches one listing with `curl`, present on every CI runner, which keeps a
/// TLS stack out of `xtask`. `-f` makes an HTTP error status a failure instead
/// of an error page that parses as an empty listing.
fn curl_listing(url: &str) -> Result<String> {
    let output = Command::new("curl")
        .args(["--fail", "--silent", "--show-error", "--location"])
        .args(["--max-time", FETCH_TIMEOUT_SECS, url])
        .output()
        .context("failed to run curl")?;
    if !output.status.success() {
        bail!(
            "curl exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    String::from_utf8(output.stdout).context("listing is not UTF-8")
}

#[cfg(test)]
mod tests {
    use super::{RELEASE_INDEX_URLS, ReleaseLabel, newest_final_releases};
    use anyhow::{Result, anyhow};

    const CANONICAL: &str = r#"
<a href="rocm-7.13.0-py3-none-linux_x86_64.whl">rocm-7.13.0-py3-none-linux_x86_64.whl</a><br/>
<a href="rocm-7.14.0-py3-none-linux_x86_64.whl">rocm-7.14.0-py3-none-linux_x86_64.whl</a><br/>
<a href="rocm-7.14.1.tar.gz">rocm-7.14.1.tar.gz</a><br/>
<a href="rocm-7.15.0a20260901.tar.gz">rocm-7.15.0a20260901.tar.gz</a><br/>
<a href="rocm-7.14.0.post1.tar.gz">rocm-7.14.0.post1.tar.gz</a><br/>
<a href="rocm-7.12.0%2Bgfx942-py3-none-any.whl">rocm-7.12.0%2Bgfx942-py3-none-any.whl</a><br/>
<a href="rocm-sdk-core-9.9.9-py3-none-any.whl">rocm-sdk-core-9.9.9-py3-none-any.whl</a><br/>
<a href="foo-rocm-9.9.9-py3-none-any.whl">foo-rocm-9.9.9-py3-none-any.whl</a><br/>
<a href="torch-9.8.7%2Brocm7.14.1-cp312-linux_x86_64.whl">torch</a><br/>
"#;
    const NEXT: &str = r#"
<a href="rocm-10.0.0-py3-none-any.whl">rocm-10.0.0-py3-none-any.whl</a><br/>
<a href="rocm-10.1.0-py3-none-any.whl">rocm-10.1.0-py3-none-any.whl</a><br/>
<a href="rocm-10.2.0rc1-py3-none-any.whl">rocm-10.2.0rc1-py3-none-any.whl</a><br/>
"#;

    fn serve(canonical: &'static str, next: &'static str) -> impl Fn(&str) -> Result<String> {
        move |url| match url {
            u if u == RELEASE_INDEX_URLS[0] => Ok(canonical.to_owned()),
            u if u == RELEASE_INDEX_URLS[1] => Ok(next.to_owned()),
            other => Err(anyhow!("unexpected url {other}")),
        }
    }

    #[test]
    fn the_three_newest_final_releases_span_both_layouts() {
        let listings = [CANONICAL.to_owned(), NEXT.to_owned()];
        assert_eq!(
            newest_final_releases(&listings),
            ["10.1.0", "10.0.0", "7.14.1", "7.14.0", "7.13.0", "7.12.0"]
        );
    }

    #[test]
    fn labels_pin_the_releases_behind_the_newest() {
        let pin = |label: ReleaseLabel| label.pin_with(serve(CANONICAL, NEXT)).unwrap();
        assert_eq!(pin(ReleaseLabel::Current), None);
        assert_eq!(pin(ReleaseLabel::NMinus1).as_deref(), Some("10.0.0"));
        assert_eq!(pin(ReleaseLabel::NMinus2).as_deref(), Some("7.14.1"));
    }

    #[test]
    fn a_new_release_moves_the_pins_without_an_edit() {
        let newer = r#"<a href="rocm-10.1.0-py3-none-any.whl">x</a><a href="rocm-10.2.0-py3-none-any.whl">x</a>"#;
        let pin = |label: ReleaseLabel| label.pin_with(serve(CANONICAL, newer)).unwrap();
        assert_eq!(pin(ReleaseLabel::NMinus1).as_deref(), Some("10.1.0"));
        assert_eq!(pin(ReleaseLabel::NMinus2).as_deref(), Some("7.14.1"));
    }

    #[test]
    fn current_needs_no_network() {
        let pin = ReleaseLabel::Current.pin_with(|_| Err(anyhow!("offline")));
        assert_eq!(pin.unwrap(), None);
    }

    #[test]
    fn an_unreachable_index_fails_the_leg() {
        let err = ReleaseLabel::NMinus1
            .pin_with(|url| {
                if url == RELEASE_INDEX_URLS[1] {
                    Err(anyhow!("offline"))
                } else {
                    Ok(CANONICAL.to_owned())
                }
            })
            .unwrap_err();
        assert!(
            format!("{err:#}").contains(RELEASE_INDEX_URLS[1]),
            "{err:#}"
        );
    }

    #[test]
    fn too_few_releases_fails_the_leg_and_names_what_was_found() {
        let one = r#"<a href="rocm-10.0.0-py3-none-any.whl">x</a>"#;
        let err = ReleaseLabel::NMinus2.pin_with(serve("", one)).unwrap_err();
        let text = err.to_string();
        assert!(
            text.contains("3 newest") && text.contains("only 1: 10.0.0"),
            "{text}"
        );
    }
}
