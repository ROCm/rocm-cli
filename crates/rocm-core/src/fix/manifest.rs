// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The catalog manifest: the JSON contract that tools reading `rocm fix
//! --json` (the `rocm-doctor` skill, the published `doctor-catalog.json`)
//! agree on with this binary. It is derived from `RECIPES` on every call and
//! never stored beside it.

use super::catalog::RECIPES;
use super::options::{FixClass, exit};
use serde::{Deserialize, Serialize};

/// The version of the agreement between this binary and the tools that read its
/// catalog.
///
/// Rules, following `ENGINE_RECIPE_CONTRACT_VERSION` in the engine protocol: an
/// **addition** keeps the version — a new entry, a new field, a doc URL filled
/// in. A **removal or a type change** raises it. A reader declares the minimum
/// it understands and stops below it rather than guessing.
pub const CATALOG_CONTRACT_VERSION: u32 = 1;

/// What the CLI does with one entry on one platform.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestPlatform {
    pub os: String,
    pub class: FixClass,
    /// The argument [`FixClass::NeedsArgument`] is waiting for; absent otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub needs: Option<String>,
}

/// One catalog entry, as published.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestEntry {
    pub id: String,
    pub title: String,
    /// Every platform this applies on, and what it does there.
    ///
    /// A list rather than one class for the entry: an entry can apply on Linux
    /// as print-only and on Windows as auto, and flattening that is precisely
    /// the defect the class model was introduced to remove.
    pub platforms: Vec<ManifestPlatform>,
    pub needs_sudo: bool,
    pub needs_reboot: bool,
    pub needs_relogin: bool,
    /// Where to read more. Absent for most entries today; filling one in later
    /// is an addition and does not raise the contract version.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_url: Option<String>,
}

/// The published description of the catalog.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CatalogManifest {
    pub contract_version: u32,
    pub entries: Vec<ManifestEntry>,
    /// What each exit code from `rocm fix` means.
    pub exit_codes: std::collections::BTreeMap<String, i32>,
}

/// Describe the catalog for a machine to read.
///
/// Derived from `RECIPES` on every call, never stored alongside it. The whole
/// point is that there is one catalog and one description of it; a second copy
/// maintained by hand is the drift this is meant to end.
#[must_use]
pub fn catalog_manifest() -> CatalogManifest {
    CatalogManifest {
        contract_version: CATALOG_CONTRACT_VERSION,
        entries: RECIPES
            .iter()
            .map(|r| ManifestEntry {
                id: r.fix_id.to_owned(),
                title: r.title.to_owned(),
                platforms: r
                    .applies_on
                    .iter()
                    .map(|p| ManifestPlatform {
                        os: p.os.to_owned(),
                        class: p.class,
                        needs: p.needs.map(ToOwned::to_owned),
                    })
                    .collect(),
                needs_sudo: r.needs_sudo,
                needs_reboot: r.needs_reboot,
                needs_relogin: r.needs_relogin,
                doc_url: r.doc_url.map(ToOwned::to_owned),
            })
            .collect(),
        exit_codes: [
            ("ok", exit::OK),
            ("internal", exit::INTERNAL),
            ("unknown_id", exit::UNKNOWN_ID),
            ("not_applicable", exit::NOT_APPLICABLE),
            ("failed", exit::FAILED),
            ("declined", exit::DECLINED),
        ]
        .into_iter()
        .map(|(name, code)| (name.to_owned(), code))
        .collect(),
    }
}

/// The manifest as the published file holds it: pretty-printed, newline
/// terminated, so a catalog change reads as a reviewable diff rather than one
/// very long line.
///
/// # Errors
/// When the manifest cannot be serialized, which would mean a type in it stopped
/// being serializable.
pub fn catalog_manifest_json() -> Result<String, serde_json::Error> {
    Ok(serde_json::to_string_pretty(&catalog_manifest())? + "\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fix::catalog::find_recipe;

    /// Pinned exactly, per platform. Promoting an entry to `AUTO` starts
    /// changing machines that callers were told it only ever advised on, and
    /// the two corrections below were both silent precisely because a flat
    /// `bool` could not record the platform the claim was true on.
    #[test]
    fn the_manifest_describes_every_entry_exactly_once() {
        let manifest = catalog_manifest();
        let published: Vec<&str> = manifest.entries.iter().map(|e| e.id.as_str()).collect();
        let catalog: Vec<&str> = RECIPES.iter().map(|r| r.fix_id).collect();
        // Order too, not just membership: the published file is read as a diff,
        // and a reordering that changed nothing would still churn review.
        assert_eq!(
            published, catalog,
            "the manifest and the catalog must list the same entries in the same order"
        );
    }

    #[test]
    fn the_manifest_repeats_the_catalog_rather_than_reinterpreting_it() {
        // The manifest exists so a reader does not have to parse prose. If it
        // said anything the catalog does not, it would be a third copy to keep
        // correct -- which is the defect, not the fix.
        for entry in catalog_manifest().entries {
            let recipe = find_recipe(&entry.id).expect("every published entry is in the catalog");
            let published: Vec<(&str, FixClass)> = entry
                .platforms
                .iter()
                .map(|p| (p.os.as_str(), p.class))
                .collect();
            let actual: Vec<(&str, FixClass)> =
                recipe.applies_on.iter().map(|p| (p.os, p.class)).collect();
            assert_eq!(published, actual, "{}: platform classes drifted", entry.id);
            assert_eq!(entry.needs_sudo, recipe.needs_sudo, "{}", entry.id);
            assert_eq!(entry.needs_reboot, recipe.needs_reboot, "{}", entry.id);
            assert_eq!(entry.needs_relogin, recipe.needs_relogin, "{}", entry.id);
        }
    }

    #[test]
    fn a_published_doc_url_is_a_url() {
        // Most entries carry none, which is the honest answer until someone
        // sources one. What must not happen is a value that looks like a link
        // and is not, since that is what sends a reader somewhere wrong.
        for entry in catalog_manifest().entries {
            if let Some(url) = entry.doc_url {
                assert!(
                    url.starts_with("https://"),
                    "{}: doc_url is not a URL: {url}",
                    entry.id
                );
            }
        }
    }
}
