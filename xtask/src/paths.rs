// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Shared workspace-path resolution for the repository tasks.
//!
//! Several subcommands need to answer the same questions — where is the
//! workspace root, which `target/` directory is active, where do the built
//! release binaries land, and what is a binary's platform file name. Those
//! answers were previously duplicated (and could drift) across `e2e.rs`,
//! `demos.rs`, and the packaging scripts. Centralizing them here keeps a single
//! definition that honors `CARGO_TARGET_DIR`, an optional cross-compilation
//! target triple, and the platform executable suffix.
//!
//! Each function that reads an environment variable is a thin wrapper over a
//! pure `*_from` core that takes the resolved override as a parameter, so the
//! resolution logic is unit-testable without mutating process-global state
//! (which is `unsafe` under edition 2024 and denied workspace-wide).

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

/// Locate the workspace root by asking cargo for the workspace manifest and
/// returning its parent directory.
pub fn workspace_root() -> Result<PathBuf> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(cargo)
        .args(["locate-project", "--workspace", "--message-format", "plain"])
        .output()
        .context("failed to run `cargo locate-project`")?;
    if !output.status.success() {
        bail!(
            "`cargo locate-project` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let manifest = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let manifest = Path::new(&manifest);
    manifest.parent().map(Path::to_path_buf).with_context(|| {
        format!(
            "could not derive workspace root from {}",
            manifest.display()
        )
    })
}

/// Active cargo target directory: `CARGO_TARGET_DIR` if set, otherwise
/// `<root>/target`. A relative override is resolved against `root`.
pub fn target_dir(root: &Path) -> PathBuf {
    target_dir_from(root, std::env::var_os("CARGO_TARGET_DIR"))
}

/// Pure core of [`target_dir`]: resolve against an explicit override value.
fn target_dir_from(root: &Path, cargo_target_dir: Option<OsString>) -> PathBuf {
    match cargo_target_dir {
        Some(dir) => {
            let dir = PathBuf::from(dir);
            if dir.is_absolute() {
                dir
            } else {
                root.join(dir)
            }
        }
        None => root.join("target"),
    }
}

/// Directory holding the built release binaries.
///
/// Honors an explicit `ROCM_BIN_DIR` override first (absolute, or relative to
/// `root`). Otherwise it is `<target-dir>/release`, and when `target_triple` is
/// supplied — as cross-platform release/nightly builds do — the triple segment
/// is inserted (`<target-dir>/<triple>/release`) to match cargo's layout.
pub fn release_binary_dir(root: &Path, target_triple: Option<&str>) -> PathBuf {
    release_binary_dir_from(
        root,
        target_triple,
        std::env::var_os("ROCM_BIN_DIR"),
        std::env::var_os("CARGO_TARGET_DIR"),
    )
}

/// Pure core of [`release_binary_dir`]: resolve against explicit override values.
fn release_binary_dir_from(
    root: &Path,
    target_triple: Option<&str>,
    rocm_bin_dir: Option<OsString>,
    cargo_target_dir: Option<OsString>,
) -> PathBuf {
    if let Some(dir) = rocm_bin_dir {
        let dir = PathBuf::from(dir);
        return if dir.is_absolute() {
            dir
        } else {
            root.join(dir)
        };
    }
    let base = target_dir_from(root, cargo_target_dir);
    match target_triple {
        Some(triple) if !triple.is_empty() => base.join(triple).join("release"),
        _ => base.join("release"),
    }
}

/// A binary's platform file name (`rocm` on Unix, `rocm.exe` on Windows).
pub fn binary_name(name: &str) -> String {
    format!("{name}{}", std::env::consts::EXE_SUFFIX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_dir_defaults_to_root_target() {
        let root = Path::new("/ws");
        assert_eq!(target_dir_from(root, None), PathBuf::from("/ws/target"));
    }

    #[test]
    fn target_dir_honors_absolute_override() {
        let root = Path::new("/ws");
        assert_eq!(
            target_dir_from(root, Some(OsString::from("/elsewhere/out"))),
            PathBuf::from("/elsewhere/out")
        );
    }

    #[test]
    fn target_dir_resolves_relative_override_against_root() {
        let root = Path::new("/ws");
        assert_eq!(
            target_dir_from(root, Some(OsString::from("out"))),
            PathBuf::from("/ws/out")
        );
    }

    #[test]
    fn release_binary_dir_without_triple() {
        let root = Path::new("/ws");
        assert_eq!(
            release_binary_dir_from(root, None, None, None),
            PathBuf::from("/ws/target/release")
        );
    }

    #[test]
    fn release_binary_dir_inserts_triple_segment() {
        let root = Path::new("/ws");
        assert_eq!(
            release_binary_dir_from(root, Some("x86_64-unknown-linux-gnu"), None, None),
            PathBuf::from("/ws/target/x86_64-unknown-linux-gnu/release")
        );
    }

    #[test]
    fn release_binary_dir_honors_target_dir_override_with_triple() {
        let root = Path::new("/ws");
        assert_eq!(
            release_binary_dir_from(
                root,
                Some("x86_64-pc-windows-msvc"),
                None,
                Some(OsString::from("/build")),
            ),
            PathBuf::from("/build/x86_64-pc-windows-msvc/release")
        );
    }

    #[test]
    fn release_binary_dir_empty_triple_is_ignored() {
        let root = Path::new("/ws");
        assert_eq!(
            release_binary_dir_from(root, Some(""), None, None),
            PathBuf::from("/ws/target/release")
        );
    }

    #[test]
    fn release_binary_dir_honors_absolute_bin_override_over_triple() {
        let root = Path::new("/ws");
        assert_eq!(
            release_binary_dir_from(
                root,
                Some("aarch64-apple-darwin"),
                Some(OsString::from("/custom/bin")),
                None,
            ),
            PathBuf::from("/custom/bin")
        );
    }

    #[test]
    fn release_binary_dir_resolves_relative_bin_override_against_root() {
        let root = Path::new("/ws");
        assert_eq!(
            release_binary_dir_from(root, None, Some(OsString::from("out/bin")), None),
            PathBuf::from("/ws/out/bin")
        );
    }

    #[test]
    fn binary_name_uses_platform_suffix() {
        let expected = format!("rocm{}", std::env::consts::EXE_SUFFIX);
        assert_eq!(binary_name("rocm"), expected);
    }
}
