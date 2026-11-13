// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Pinned versions of the third-party runtimes rocm-cli manages, plus the
//! helpers that derive artifact names and URLs from them.
//!
//! Today that is Lemonade: its pin lives in `runtimes/mise.toml`, is locked
//! into `runtimes/mise.lock` by mise, and is turned into the constants below by
//! `build.rs`. The Lemonade adapter checks downloads against the locked
//! digests and derives its download URLs from the version with
//! [`lemonade_download_url`] (a test below ties those to the locked URLs); it
//! and `rocm engines` take the version from here, so no non-test source holds
//! a second Lemonade version.
//! Tooling pins in `mise.toml` (uv) get no constants; see `TOOLING_ONLY`.

// `pub const LEMONADE_VERSION: &str = "..."`, plus per-platform URL and digest, one set per runtime tool in `runtimes/mise.lock`.
include!(concat!(env!("OUT_DIR"), "/pins.rs"));

// The generator `build.rs` runs, compiled here only so its tests run.
#[cfg(test)]
mod pins_gen;

/// GitHub repository publishing the Lemonade embeddable release archives.
pub const LEMONADE_GITHUB_REPO: &str = "lemonade-sdk/lemonade";

/// Strip a leading `v` from a release tag (`v11.5.1` -> `11.5.1`).
#[must_use]
pub fn strip_v(tag: &str) -> &str {
    tag.strip_prefix('v').unwrap_or(tag)
}

/// Name of the Lemonade embeddable archive for a host token and extension,
/// e.g. `lemonade-embeddable-11.5.1-ubuntu-x64.tar.gz`.
#[must_use]
pub fn lemonade_archive_name(version: &str, os_arch: &str, extension: &str) -> String {
    format!(
        "lemonade-embeddable-{}-{os_arch}.{extension}",
        strip_v(version)
    )
}

/// Download URL of a Lemonade embeddable archive published under `v<version>`.
#[must_use]
pub fn lemonade_download_url(version: &str, archive_name: &str) -> String {
    format!(
        "https://github.com/{LEMONADE_GITHUB_REPO}/releases/download/v{}/{archive_name}",
        strip_v(version)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_version_is_a_bare_release_number() {
        assert!(
            !LEMONADE_VERSION.is_empty() && !LEMONADE_VERSION.starts_with('v'),
            "pin must be a bare version, got {LEMONADE_VERSION:?}"
        );
    }

    #[test]
    fn archive_names_and_urls_match_the_published_shape() {
        let version = LEMONADE_VERSION;
        let linux = lemonade_archive_name(version, "ubuntu-x64", "tar.gz");
        let windows = lemonade_archive_name(version, "windows-x64", "zip");
        assert_eq!(
            linux,
            format!("lemonade-embeddable-{version}-ubuntu-x64.tar.gz")
        );
        assert_eq!(
            windows,
            format!("lemonade-embeddable-{version}-windows-x64.zip")
        );
        assert_eq!(
            lemonade_download_url(version, &linux),
            format!(
                "https://github.com/lemonade-sdk/lemonade/releases/download/v{version}/{linux}"
            )
        );
    }

    /// The Lemonade adapter derives its download URL from the version and
    /// checks the download against the digest mise locked. The two only belong
    /// together if mise locked the very archive the adapter downloads — the
    /// embeddable build, not one of the installers the release also carries.
    #[test]
    fn the_locked_assets_are_the_archives_the_adapter_downloads() {
        for (os_arch, extension, locked_url) in [
            ("ubuntu-x64", "tar.gz", LEMONADE_LINUX_X64_URL),
            ("windows-x64", "zip", LEMONADE_WINDOWS_X64_URL),
        ] {
            let derived = lemonade_download_url(
                LEMONADE_VERSION,
                &lemonade_archive_name(LEMONADE_VERSION, os_arch, extension),
            );
            assert_eq!(
                locked_url, derived,
                "runtimes/mise.lock locked another asset for {os_arch}"
            );
        }
    }

    #[test]
    fn a_tagged_version_is_accepted_anywhere_a_version_is() {
        assert_eq!(
            lemonade_archive_name("v11.6.0", "ubuntu-x64", "tar.gz"),
            "lemonade-embeddable-11.6.0-ubuntu-x64.tar.gz"
        );
        assert_eq!(
            lemonade_download_url("v11.6.0", "archive.tar.gz"),
            "https://github.com/lemonade-sdk/lemonade/releases/download/v11.6.0/archive.tar.gz"
        );
    }
}
