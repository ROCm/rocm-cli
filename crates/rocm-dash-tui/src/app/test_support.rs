// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Test-only helpers shared across `app`'s submodule test suites, and
//! (`with_registry_probe`) `crate::ui`'s own test module too — the module
//! itself and that one fn are `pub(crate)` rather than `pub(super)` for
//! exactly that cross-module reach.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::AppState;

pub(super) fn press(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

pub(super) fn st() -> AppState {
    AppState::new("t".into(), "default-dark".into())
}

/// Install a static live-registry probe on `state` (the test-only stand-in
/// for the bin-injected seam — see `registry_probe::StaticManagedRegistrySnapshot`),
/// so a test can exercise the live-read gates without a real registry on
/// disk. Shared by `app`'s own test module and `ui`'s, rather than each
/// carrying its own byte-identical copy.
pub(crate) fn with_registry_probe(
    state: &mut AppState,
    has_live_service: bool,
    managed_ids: &[&str],
) {
    use crate::registry_probe::{ManagedRegistrySnapshot, StaticManagedRegistrySnapshot};
    state.managed_registry_probe = Some(std::sync::Arc::new(StaticManagedRegistrySnapshot(
        ManagedRegistrySnapshot {
            has_live_service,
            managed_ids: managed_ids.iter().map(|s| (*s).to_string()).collect(),
        },
    )));
}
