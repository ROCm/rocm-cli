// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Shared fixtures for building a throwaway [`AppPaths`] under a per-test,
//! per-process directory.
//!
//! Used from every module split out of this crate's former `lib.rs` god file
//! (`host_gpu`, `rocm_install`, `managed_runtime`, and `lib.rs` itself) so the
//! naming/uniqueness scheme for test artifact directories stays in one place
//! rather than drifting across copies.

use crate::{AppPaths, unix_time_millis};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) fn temp_app_paths(name: &str) -> (PathBuf, AppPaths) {
    let root = workspace_test_artifact_dir().join(format!(
        "rocm-core-{name}-{}-{}",
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

/// Process id, wall-clock millis and a per-process counter, so two names
/// minted in the same millisecond by the same process still differ.
pub(crate) fn unique_suffix() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}-{}-{}",
        std::process::id(),
        unix_time_millis(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

pub(crate) fn workspace_test_artifact_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(".rocm-work")
        .join("tests")
        .join("core")
}
