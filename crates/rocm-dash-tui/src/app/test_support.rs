// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Test-only helpers shared across `app`'s submodule test suites.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::AppState;

pub(super) fn press(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

pub(super) fn st() -> AppState {
    AppState::new("t".into(), "default-dark".into())
}
