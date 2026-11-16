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
    /// The number behind each exit-code name from `rocm fix`.
    pub exit_codes: std::collections::BTreeMap<String, i32>,
    /// What each exit-code name in [`Self::exit_codes`] means, keyed by the same
    /// names. A sibling map rather than a richer value in `exit_codes`, so a
    /// reader written against name -> number keeps working: an added field,
    /// which does not raise the contract version.
    ///
    /// `unknown_id` (2) says that a malformed command line exits 2 as well, and
    /// how to tell the two refusals apart.
    #[serde(default)]
    pub exit_code_descriptions: std::collections::BTreeMap<String, String>,
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
        exit_codes: exit::MEANINGS
            .iter()
            .map(|m| (m.name.to_owned(), m.code))
            .collect(),
        exit_code_descriptions: exit::MEANINGS
            .iter()
            .map(|m| (m.name.to_owned(), m.description.to_owned()))
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

    /// Every published code carries a description, under the same name, and
    /// the table they come from names each code once. A name with a number and
    /// no meaning is the gap ROCMAI-955 reported; two names for one number would
    /// leave a reader unable to say which it got.
    #[test]
    fn every_published_exit_code_is_described_once() {
        let manifest = catalog_manifest();
        let named: Vec<&String> = manifest.exit_codes.keys().collect();
        let described: Vec<&String> = manifest.exit_code_descriptions.keys().collect();
        assert_eq!(named, described, "every exit code needs a description");
        for (name, text) in &manifest.exit_code_descriptions {
            assert!(!text.trim().is_empty(), "`{name}` is described by nothing");
        }
        let mut codes: Vec<i32> = exit::MEANINGS.iter().map(|m| m.code).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(
            codes.len(),
            exit::MEANINGS.len(),
            "two names publish the same exit code"
        );
        assert_eq!(
            manifest.exit_codes.len(),
            exit::MEANINGS.len(),
            "two entries share a name"
        );
    }

    /// Exit 2 means two things, and the description has to say both and how to
    /// tell them apart, or a caller that reads it still cannot. The message it
    /// names is the one `apply` prints, held to the same constant; that the
    /// argument parser really exits 2 is checked in `apps/rocm`, which owns it
    /// (`a_usage_error_exits_with_the_code_published_for_an_unknown_fix_id`).
    #[test]
    fn the_unknown_id_description_names_the_usage_error_it_shares_a_code_with() {
        let manifest = catalog_manifest();
        assert_eq!(manifest.exit_codes["unknown_id"], exit::UNKNOWN_ID);
        let text = &manifest.exit_code_descriptions["unknown_id"];
        for needed in [
            exit::UNKNOWN_ID_PREFIX,
            "argument parser",
            "malformed command line",
            "`error:`",
            // A one-time `rocm:` notice can precede either refusal, so the
            // promise is about a line, not about where stderr starts.
            "stderr line",
        ] {
            assert!(
                text.contains(needed),
                "the unknown_id description must mention `{needed}`: {text}"
            );
        }
    }

    /// `MEANINGS` claims to list every code `rocm fix` returns. The runners
    /// return the `exit::` constants (held by
    /// `fix_runners_return_named_exit_constants_not_integers`), so the set of
    /// constants is the set of codes, and this holds the table to it: a new
    /// constant fails here until it is described and published.
    #[test]
    fn every_exit_constant_is_published_and_nothing_else_is() {
        let file = syn::parse_file(include_str!("options.rs")).expect("options.rs parses");
        let exit_mod = file
            .items
            .iter()
            .find_map(|item| match item {
                syn::Item::Mod(m) if m.ident == "exit" => m.content.as_ref(),
                _ => None,
            })
            .expect("options.rs has an inline `mod exit`");
        let mut declared: Vec<(String, i32)> = exit_mod
            .1
            .iter()
            .filter_map(|item| match item {
                syn::Item::Const(c) if matches!(&*c.ty, syn::Type::Path(p) if p.path.is_ident("i32")) => {
                    let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Int(value),
                        ..
                    }) = &*c.expr
                    else {
                        panic!("exit::{} is not an integer literal", c.ident);
                    };
                    Some((
                        c.ident.to_string(),
                        value.base10_parse().expect("an i32 literal"),
                    ))
                }
                _ => None,
            })
            .collect();
        declared.sort_by_key(|(_, code)| *code);
        let mut published: Vec<i32> = exit::MEANINGS.iter().map(|m| m.code).collect();
        published.sort_unstable();
        let declared_codes: Vec<i32> = declared.iter().map(|(_, code)| *code).collect();
        assert_eq!(
            declared_codes, published,
            "every exit:: constant must have exactly one MEANINGS entry, and no entry may \
             publish a code no constant names; constants: {declared:?}"
        );
        // Name by name too: the published name is the constant's, lower-cased,
        // so a reader of the manifest and a reader of the code use one word.
        for (constant, code) in &declared {
            let meaning = exit::MEANINGS
                .iter()
                .find(|m| m.code == *code)
                .expect("checked above");
            assert_eq!(
                meaning.name,
                constant.to_lowercase(),
                "exit::{constant} is published under another name"
            );
        }
    }

    /// A manifest written before the descriptions existed still reads: the
    /// field was added, so the contract version did not move, and that is only
    /// true if an older document parses.
    #[test]
    fn a_manifest_without_descriptions_still_parses() {
        let mut value = serde_json::to_value(catalog_manifest()).expect("serializes");
        value
            .as_object_mut()
            .expect("an object")
            .remove("exit_code_descriptions");
        let older: CatalogManifest = serde_json::from_value(value).expect("an older manifest");
        assert!(older.exit_code_descriptions.is_empty());
        assert_eq!(older.contract_version, CATALOG_CONTRACT_VERSION);
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
