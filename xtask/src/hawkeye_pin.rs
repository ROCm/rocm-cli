// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Guards against the hawkeye version documented in CONTRIBUTING.md drifting
//! from the version `ci.yml`'s `license-headers` job installs.
//!
//! CONTRIBUTING.md tells contributors to `cargo install hawkeye@<version>
//! --locked` so the local pre-commit hook produces the same verdict as CI's
//! pinned prebuilt binary. Nothing else ties those two strings together, so a
//! version bump in one and not the other would silently let contributors pass
//! locally on a hawkeye build that can still fail in CI (or vice versa). This
//! checks only that the two version strings agree — it does not check the
//! SHA256 pin, the `licenserc.toml` rules, or which file types the local hook
//! scans (see `.pre-commit-config.yaml`'s `license-headers` hook for that).

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    /// hawkeye version CONTRIBUTING.md tells contributors to install. Kept in
    /// sync with `ci.yml`'s `HAWKEYE_VERSION` by `pin_matches_ci_workflow`
    /// below.
    const HAWKEYE_VERSION: &str = "7.0.0";

    fn repo_root() -> PathBuf {
        // CARGO_MANIFEST_DIR is the xtask/ crate dir; its parent is the repo
        // root (same idiom as verify_pinned_keys::repo_root).
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask crate has a parent directory")
            .to_path_buf()
    }

    /// The documented pin and the version CI installs must never drift: if
    /// they do, a contributor's local hook can bless a header hawkeye's CI
    /// build would reject, or vice versa.
    #[test]
    fn pin_matches_ci_workflow() {
        let root = repo_root();

        let contributing = root.join("CONTRIBUTING.md");
        let contributing_text = std::fs::read_to_string(&contributing)
            .unwrap_or_else(|e| panic!("reading {}: {e}", contributing.display()));
        let documented = format!("cargo install hawkeye@{HAWKEYE_VERSION} --locked");
        assert!(
            contributing_text.contains(&documented),
            "CONTRIBUTING.md must instruct `{documented}` to match HAWKEYE_VERSION in \
             hawkeye_pin.rs"
        );

        let ci = root.join(".github/workflows/ci.yml");
        let ci_text = std::fs::read_to_string(&ci)
            .unwrap_or_else(|e| panic!("reading {}: {e}", ci.display()));
        // ci.yml's env var is `v`-prefixed (`HAWKEYE_VERSION: v7.0.0`); CONTRIBUTING.md's
        // prose is not (`hawkeye@7.0.0`). Both are formatted from the same bare
        // HAWKEYE_VERSION constant above, with the `v` added only on this side.
        let installed = format!("HAWKEYE_VERSION: v{HAWKEYE_VERSION}");
        assert!(
            ci_text.contains(&installed),
            "ci.yml must install `{installed}` to match HAWKEYE_VERSION in hawkeye_pin.rs"
        );
    }
}
