// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Shared test-only fixtures for `rocmd`'s module test suites.
//!
//! Every `#[cfg(test)] mod tests` in this crate that exercises `AppPaths`
//! needs the same workspace-local scratch directory under
//! `.rocm-work/tests/rocmd`. Keeping one copy here (rather than one per
//! module) avoids re-diverging these helpers as `lib.rs` continues to be
//! split into focused modules (see `docs/architecture.md`).

use rocm_core::{AppPaths, unix_time_millis};
use std::fs;
use std::path::PathBuf;

pub(crate) fn temp_app_paths(name: &str) -> (PathBuf, AppPaths) {
    let root = unique_test_root(&format!(
        "rocmd-{name}-{}-{}",
        std::process::id(),
        unix_time_millis()
    ));
    let paths = AppPaths {
        config_dir: root.join("config"),
        data_dir: root.join("data"),
        cache_dir: root.join("cache"),
    };
    (root, paths)
}

pub(crate) fn unique_test_root(label: &str) -> PathBuf {
    let root = workspace_test_artifact_dir().join(label);
    fs::create_dir_all(&root).expect("create workspace-local test root");
    root
}

pub(crate) fn workspace_test_artifact_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(".rocm-work")
        .join("tests")
        .join("rocmd")
}
