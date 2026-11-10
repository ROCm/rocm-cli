// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Locating `rocm-engine-<name>` siblings, and the CLI binary itself (a
//! sibling `rocm` only when running from a cargo `deps/` directory).

use anyhow::{Result, bail};
use std::path::{Path, PathBuf};

use crate::require_nonempty;
use crate::{current_executable_path, platform_binary_name};

pub fn sibling_binary_path(binary_name: &str) -> Result<PathBuf> {
    sibling_binary_path_from(&current_executable_path()?, binary_name)
}

/// [`sibling_binary_path`] for a caller that already holds the executable path.
///
/// A caller that decides something from the current executable and then looks
/// for a sibling passes the one value to both, so the two cannot disagree: the
/// lookup falls back to `argv[0]` where `std::env::current_exe` fails.
pub fn sibling_binary_path_from(current_exe: &Path, binary_name: &str) -> Result<PathBuf> {
    require_nonempty(binary_name, "binary_name")?;
    let candidates = sibling_binary_candidates(current_exe, binary_name)?;
    for candidate in &candidates {
        if candidate.is_file() {
            return Ok(candidate.clone());
        }
    }
    let candidate_text = candidates
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    bail!(
        "unable to locate sibling binary {}; checked {} next to {}",
        platform_binary_name(binary_name),
        candidate_text,
        current_exe.display()
    )
}

fn sibling_binary_candidates(current_exe: &Path, binary_name: &str) -> Result<Vec<PathBuf>> {
    let Some(binary_dir) = current_exe.parent() else {
        bail!("current executable has no parent directory");
    };
    let binary = platform_binary_name(binary_name);
    let mut candidates = Vec::new();
    let mut push_candidate = |path: PathBuf| {
        if !candidates.iter().any(|candidate| candidate == &path) {
            candidates.push(path);
        }
    };
    push_candidate(binary_dir.join(&binary));
    if binary_dir.file_name().and_then(|name| name.to_str()) == Some("deps")
        && let Some(parent) = binary_dir.parent()
    {
        push_candidate(parent.join(&binary));
        if let Some(target_dir) = parent.parent() {
            for profile in ["release", "debug"] {
                push_candidate(target_dir.join(profile).join(&binary));
            }
        }
    }
    Ok(candidates)
}

pub fn engine_binary_path(engine: &str) -> Result<PathBuf> {
    sibling_binary_path(&format!("rocm-engine-{engine}"))
}

pub fn daemon_binary_path() -> Result<PathBuf> {
    let current_exe = current_executable_path()?;
    if current_exe
        .parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        == Some("deps")
        && let Ok(candidates) = sibling_binary_candidates(&current_exe, "rocm")
        && let Some(rocm) = candidates.into_iter().find(|candidate| candidate.is_file())
    {
        return Ok(rocm);
    }
    Ok(current_exe)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lookup starts from the executable it is given, not from the
    /// process's own: a caller that chose something from one path must find the
    /// sibling beside that same path.
    #[test]
    fn sibling_binary_path_from_looks_beside_the_given_executable() {
        let dir = tempfile::tempdir().expect("tempdir");
        let exe = dir.path().join(platform_binary_name("rocm"));
        let sibling = dir.path().join(platform_binary_name("rocmd"));
        std::fs::write(&sibling, b"").expect("write sibling");

        assert_eq!(sibling_binary_path_from(&exe, "rocmd").unwrap(), sibling);

        let error = sibling_binary_path_from(&exe, "rocm-engine-none").unwrap_err();
        let text = error.to_string();
        assert!(text.contains("unable to locate sibling binary"), "{text}");
        assert!(
            text.contains(&dir.path().display().to_string()),
            "the error names where it looked: {text}"
        );
    }

    #[test]
    fn sibling_binary_path_from_rejects_an_empty_name() {
        assert!(sibling_binary_path_from(Path::new("/opt/bin/rocm"), "").is_err());
    }
}
