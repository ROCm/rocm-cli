// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Where an installed Lemonade runtime keeps the `llama-server` of a backend.
//!
//! The layout mirrors what the Lemonade engine searches when it serves a model
//! directly: `bin/llamacpp/<backend dir>/llama-server` under the runtime
//! directory, either flat or one build-numbered subdirectory down. The `rocm`
//! backend is a family of directory names; Vulkan never crosses over to ROCm.

use std::path::{Path, PathBuf};

/// Directory names that hold the `rocm` backend, in lookup order.
const ROCM_DIRS: [&str; 3] = ["rocm-stable", "rocm-nightly", "rocm"];

/// The outcome of looking for a backend's `llama-server`.
#[derive(Debug, PartialEq, Eq)]
pub struct ServerLookup {
    /// The binary, when one was found.
    pub found: Option<PathBuf>,
    /// Every directory that was searched, in order.
    pub searched: Vec<PathBuf>,
}

/// The platform's file name for the `llama-server` binary.
const fn server_binary_name() -> &'static str {
    if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    }
}

/// The directory names a backend may extract into, or `None` for a backend that
/// has no GPU `llama-server` (CPU and unknown names).
fn backend_dirs(backend_name: &str) -> Option<&'static [&'static str]> {
    match backend_name {
        "rocm" => Some(&ROCM_DIRS),
        "vulkan" => Some(&["vulkan"]),
        _ => None,
    }
}

/// Look for the selected backend's `llama-server` under `runtime_dir`.
///
/// Returns `Err` for a backend that has no GPU `llama-server`, so a recorded CPU
/// or unknown backend cannot pass as "nothing to find".
pub fn find_llama_server(runtime_dir: &Path, backend_name: &str) -> Result<ServerLookup, String> {
    let dirs = backend_dirs(backend_name).ok_or_else(|| {
        format!("backend `{backend_name}` has no GPU llama-server directory to look in")
    })?;
    let llamacpp = runtime_dir.join("bin").join("llamacpp");
    let binary = server_binary_name();
    let mut searched = Vec::new();
    for dir in dirs {
        let dir = llamacpp.join(dir);
        searched.push(dir.clone());
        if let Some(found) = binary_in(&dir, binary) {
            return Ok(ServerLookup {
                found: Some(found),
                searched,
            });
        }
    }
    Ok(ServerLookup {
        found: None,
        searched,
    })
}

/// `binary` directly in `dir`, or one level down in a subdirectory.
fn binary_in(dir: &Path, binary: &str) -> Option<PathBuf> {
    let direct = dir.join(binary);
    if direct.is_file() {
        return Some(direct);
    }
    let mut subdirs: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    subdirs.sort();
    subdirs
        .into_iter()
        .map(|subdir| subdir.join(binary))
        .find(|candidate| candidate.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn install(runtime: &Path, relative: &str) -> PathBuf {
        let dir = runtime.join("bin").join("llamacpp").join(relative);
        std::fs::create_dir_all(&dir).expect("create backend dir");
        let binary = dir.join(server_binary_name());
        std::fs::write(&binary, b"x").expect("write binary");
        binary
    }

    #[test]
    fn finds_a_flat_binary_for_the_selected_backend() {
        let root = tempfile::tempdir().expect("tempdir");
        let binary = install(root.path(), "vulkan");
        let lookup = find_llama_server(root.path(), "vulkan").expect("lookup");
        assert_eq!(lookup.found, Some(binary));
    }

    #[test]
    fn finds_a_build_numbered_binary_in_the_rocm_family() {
        let root = tempfile::tempdir().expect("tempdir");
        let binary = install(root.path(), "rocm-stable/llama-b9752");
        let lookup = find_llama_server(root.path(), "rocm").expect("lookup");
        assert_eq!(lookup.found, Some(binary));
    }

    #[test]
    fn rocm_lookup_follows_family_order() {
        let root = tempfile::tempdir().expect("tempdir");
        install(root.path(), "rocm");
        let stable = install(root.path(), "rocm-stable");
        let lookup = find_llama_server(root.path(), "rocm").expect("lookup");
        assert_eq!(lookup.found, Some(stable));
    }

    #[test]
    fn a_different_backend_does_not_stand_in() {
        let root = tempfile::tempdir().expect("tempdir");
        install(root.path(), "vulkan");
        install(root.path(), "cpu");
        let lookup = find_llama_server(root.path(), "rocm").expect("lookup");
        assert_eq!(lookup.found, None);
        let llamacpp = root.path().join("bin").join("llamacpp");
        let expected: Vec<PathBuf> = ROCM_DIRS.iter().map(|dir| llamacpp.join(dir)).collect();
        assert_eq!(lookup.searched, expected);
    }

    #[test]
    fn a_directory_without_the_binary_is_not_a_hit() {
        let root = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(root.path().join("bin/llamacpp/vulkan/llama-b1"))
            .expect("create dir");
        let lookup = find_llama_server(root.path(), "vulkan").expect("lookup");
        assert_eq!(lookup.found, None);
        assert_eq!(lookup.searched.len(), 1);
    }

    #[test]
    fn cpu_and_unknown_backends_are_rejected() {
        let root = tempfile::tempdir().expect("tempdir");
        install(root.path(), "cpu");
        for backend in ["cpu", "unknown", ""] {
            assert!(
                find_llama_server(root.path(), backend).is_err(),
                "{backend}"
            );
        }
    }
}
