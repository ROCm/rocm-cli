// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Test-only helpers shared across `app`'s submodule test suites, and
//! (`press`, `st`, `with_registry_probe`) `crate::ui`'s test modules too — the
//! module itself and those fns are `pub(crate)` rather than `pub(super)` for
//! exactly that cross-module reach.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::{ActiveTab, AppState, ResolvedArgs};

/// A plain (no-modifier) key press. `crate::ui`'s screen test modules import
/// this — aliased to their local call-site name — rather than each carrying
/// its own byte-identical copy.
pub(crate) fn press(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

pub(crate) fn st() -> AppState {
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

/// Minimal `ResolvedArgs` for the `build_chat_agent` factory tests. Keys are
/// passed via the struct (the in-process seam) — never argv.
pub(super) fn args_with_anthropic_key(key: Option<&str>) -> ResolvedArgs {
    ResolvedArgs {
        connect: "test".into(),
        token: None,
        theme: "default-dark".into(),
        replay: None,
        initial_tab: ActiveTab::Chat,
        focus: None,
        chat_url: None,
        chat_model: None,
        chat_auth_header: None,
        chat_temperature: None,
        chat_top_p: None,
        chat_max_tokens: None,
        chat_env_url: None,
        chat_api_key: None,
        anthropic_api_key: key.map(str::to_string),
        chat_auto_consent: false,
        chat_mock: false,
        model_recipes: Vec::new(),
        runtimes: Vec::new(),
        automations: Vec::new(),
        chat_system_prompt: None,
        tool_executor: None,
        bench_results_dir: None,
        services_past_attempts: 0,
        managed_registry_probe: None,
        launcher_hosted: false,
    }
}
