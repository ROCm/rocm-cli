// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Pinned versions of the third-party runtimes rocm-cli manages, plus the
//! helpers that derive artifact names and URLs from them.
//!
//! The pins live in the workspace-root `runtime-deps.toml` and are turned into
//! the constants below by `build.rs`. Every consumer — the `rocm` binary and
//! the Lemonade engine adapter — derives what it needs from those constants, so
//! a runtime version is spelled exactly once in the repository and two crates
//! cannot disagree about it.

// `pub const LEMONADE_VERSION: &str = "..."`, one per `runtime-deps.toml` field.
include!(concat!(env!("OUT_DIR"), "/pins.rs"));

/// GitHub repository publishing the Lemonade embeddable release archives.
pub const LEMONADE_GITHUB_REPO: &str = "lemonade-sdk/lemonade";

/// Strip a leading `v` from a release tag (`v2026.44.1` -> `2026.44.1`).
#[must_use]
pub fn strip_v(tag: &str) -> &str {
    tag.strip_prefix('v').unwrap_or(tag)
}

/// Name of the Lemonade embeddable archive for a host token and extension,
/// e.g. `lemonade-embeddable-2026.44.1-ubuntu-x64.tar.gz`.
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

/// The `env_id` a Lemonade embeddable runtime of `version` is recorded under.
///
/// `version` is the bare release number, as in `LEMONADE_VERSION`, not a `v`
/// tag: the id is compared by exact string, so it is not normalised here.
///
/// For example `lemonade-embeddable-11.5.1`. The engine writes it into the install
/// manifest it creates, and `rocm` compares it with what the engine's `Detect`
/// reports to decide whether its own install is already done, so the two must
/// agree. The e2e suite repeats the shape when it plants a runtime, because it
/// imports nothing from this workspace; its serve-18 scenario fails if the
/// planted id stops matching.
#[must_use]
pub fn lemonade_env_id(version: &str) -> String {
    format!("lemonade-embeddable-{version}")
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

    #[test]
    fn the_lemonade_env_id_names_the_embeddable_and_its_version() {
        assert_eq!(lemonade_env_id("11.5.1"), "lemonade-embeddable-11.5.1");
        assert_eq!(
            lemonade_env_id(LEMONADE_VERSION),
            format!("lemonade-embeddable-{LEMONADE_VERSION}")
        );
    }
}
