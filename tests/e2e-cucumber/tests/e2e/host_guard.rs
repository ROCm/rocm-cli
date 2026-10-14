// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The host check the Windows driver-install steps run before any CLI call.
//!
//! Kept free of the cucumber world so `driver_host_guard.rs` can include it and
//! call it with an OS name. The `e2e` target has `harness = false`, so a test
//! inside it would never run.

/// Panic unless `os` names native Windows. WSL reports `linux`.
pub fn require_os_is_windows(os: &str) {
    assert_eq!(
        os, "windows",
        "this driver-install step requires native Windows"
    );
}
