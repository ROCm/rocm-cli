// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Regenerate (or verify) the published Doctor catalog manifest.
//!
//! The manifest is what a tool reads instead of parsing the CLI's prose: every
//! entry, what the CLI does with each on each platform, and the meaning of every
//! exit code. It is rendered from the compiled catalog by
//! [`rocm_core::catalog_manifest_json`], so this never restates it -- the point
//! of the whole contract is that the catalog is described in exactly one place.
//!
//! The checked-in copy is a guard rather than the contract. Its value is that a
//! catalog change shows up in the pull request diff, in the form a consumer
//! reads, which is what makes deciding whether to raise `contract_version` a
//! deliberate act rather than something noticed afterwards.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::paths;

/// Published at the repository root, beside the other generated contract
/// artifacts (`MANIFEST.md`, `THIRD_PARTY_NOTICES.txt`).
const CATALOG_FILE: &str = "doctor-catalog.json";

fn catalog_path(root: &Path) -> PathBuf {
    root.join(CATALOG_FILE)
}

/// Regenerate (or, with `check`, verify) the published catalog manifest.
///
/// # Errors
/// When the workspace root cannot be found, the file cannot be read or written,
/// or `check` is set and the file is out of date.
pub fn run(check: bool) -> Result<()> {
    let root = paths::workspace_root()?;
    let path = catalog_path(&root);
    let rendered =
        rocm_core::catalog_manifest_json().context("failed to render the catalog manifest")?;

    // Missing counts as out of date rather than as an error, so the first run
    // and every later one behave the same way.
    let current = fs::read_to_string(&path).unwrap_or_default();

    if check {
        if current != rendered {
            bail!(
                "{CATALOG_FILE} is out of date; run `cargo xtask catalog` to update it.\n\
                 The catalog changed, so decide whether this is an addition (which keeps \
                 contract_version) or a removal or type change (which raises it)."
            );
        }
        return Ok(());
    }

    if current != rendered {
        fs::write(&path, rendered)
            .with_context(|| format!("failed to write {}", path.display()))?;
    }
    Ok(())
}
