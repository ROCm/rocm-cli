// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Unique scratch directories behind TheRock installs. The atomic file writes
//! that used to live here are `rocm_core::atomic_write`, shared with `rocmd`.
//!
//! A leaf of the `therock` module: nothing here depends on the rest of it.

use anyhow::{Context, Result, bail};
use rocm_core::unix_time_millis;
use std::fs;
use std::path::PathBuf;

pub(super) fn linux_temp_dir(prefix: &str) -> Result<PathBuf> {
    let root = std::env::temp_dir();
    let base = format!("{prefix}-{}-{}", std::process::id(), unix_time_millis());
    for attempt in 0..100 {
        let dir = root.join(format!("{base}-{attempt}"));
        match fs::create_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(error).with_context(|| format!("failed to create {}", dir.display()));
            }
        }
    }
    bail!(
        "failed to create a unique temporary directory under {}",
        root.display()
    )
}

pub(super) fn windows_temp_dir(prefix: &str) -> Result<PathBuf> {
    let root = windows_runtime_temp_root().unwrap_or_else(std::env::temp_dir);
    let base = format!("{prefix}-{}-{}", std::process::id(), unix_time_millis());
    for attempt in 0..100 {
        let dir = root.join(format!("{base}-{attempt}"));
        match fs::create_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(error).with_context(|| format!("failed to create {}", dir.display()));
            }
        }
    }
    bail!(
        "failed to create a unique temporary directory under {}",
        root.display()
    )
}

fn windows_runtime_temp_root() -> Option<PathBuf> {
    for name in ["TEMP", "TMP", "LOCALAPPDATA"] {
        if let Some(value) = std::env::var_os(name).filter(|value| !value.is_empty()) {
            let path = PathBuf::from(value);
            return Some(if name == "LOCALAPPDATA" {
                path.join("Temp")
            } else {
                path
            });
        }
    }
    None
}
