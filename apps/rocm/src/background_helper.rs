// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! What the CLI can tell about the background automation helper (`rocm daemon`):
//! whether it is running, and the note shown to the user about it after a
//! service stop or when planning an uninstall.
//!
//! Relocated verbatim from `service_launch.rs` with no behavior change, so that
//! `services_text.rs` and `uninstall_plan.rs` can read the note
//! without importing the module that spawns services. Spawning the helper
//! (`ensure_background_helper_running` and its autostart lock) stays in
//! `service_launch.rs`, which reads `background_helper_already_running` from here.
//! This module imports nothing from the service modules.

use anyhow::Result;
use rocm_core::{AppPaths, AutomationRuntimeState};

/// The "should we spawn?" decision for [`ensure_background_helper_running`],
/// factored out so it is testable hermetically (no spawn side effect). Returns
/// `true` when the file-based runtime state says the daemon is `running` AND its
/// recorded `daemon_pid` is a live process — i.e. a second spawn must be guarded.
/// A missing state file, `running=false`, or a dead/zero pid returns `false`.
pub(crate) fn background_helper_already_running(paths: &AppPaths) -> Result<bool> {
    Ok(AutomationRuntimeState::load(paths)?
        .is_some_and(|state| state.running && rocm_core::process_is_running(state.daemon_pid)))
}

/// What the user should be told about the background automation helper
/// (`rocm daemon`) after stopping a service or when planning an uninstall.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HelperNote {
    /// The sentence to print.
    pub(crate) text: String,
    /// The helper is running right now.
    pub(crate) running: bool,
    /// An automation watcher is enabled, so the helper stays up on its own.
    pub(crate) outlives: bool,
}

/// The note for one combination of the two facts the helper itself acts on:
/// whether a watcher is enabled
/// ([`rocm_core::RocmCliConfig::automation_daemon_enabled`] gates both the spawn
/// and the helper's exit) and whether it is running. `None` for neither.
fn helper_note_for(enabled: &[&str], running: bool) -> Option<HelperNote> {
    let text = match (enabled.is_empty(), running) {
        (true, false) => return None,
        (true, true) => "the background automation helper (`rocm daemon`) is still running; no automation is enabled, so it exits on its own within a few seconds".to_owned(),
        (false, true) => {
            let disable = enabled
                .iter()
                .map(|id| format!("`rocm automations disable {id}`"))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "the background automation helper (`rocm daemon`) keeps running while an automation is enabled ({}); to stop it, run {disable}, after which it exits on its own within a few seconds",
                enabled.join(", ")
            )
        }
        (false, false) => format!(
            "an automation is enabled ({}) but the background automation helper (`rocm daemon`) is not running; `rocm serve` or `rocm automations enable` starts it",
            enabled.join(", ")
        ),
    };
    Some(HelperNote {
        text,
        running,
        outlives: !enabled.is_empty() && running,
    })
}

/// [`helper_note_for`] over the facts on disk. An unreadable config counts as
/// "no watcher enabled".
pub(crate) fn background_helper_note(paths: &AppPaths) -> Option<HelperNote> {
    let config = rocm_core::RocmCliConfig::load(paths).unwrap_or_default();
    let enabled: Vec<&str> = config
        .automations
        .watchers
        .iter()
        .filter(|(_, watcher)| watcher.enabled)
        .map(|(id, _)| id.as_str())
        .collect();
    let running = background_helper_already_running(paths).unwrap_or(false);
    helper_note_for(&enabled, running)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::fs;

    use super::*;
    use crate::tests::test_paths;

    /// Build an `AutomationRuntimeState` for the no-double-spawn guard tests.
    pub(crate) fn runtime_state(running: bool, daemon_pid: u32) -> AutomationRuntimeState {
        AutomationRuntimeState {
            running,
            automations_enabled: true,
            daemon_pid,
            started_at_unix_ms: 1,
            last_tick_unix_ms: 1,
            local_webhook_endpoint: None,
            active_watchers: Vec::new(),
        }
    }

    #[test]
    fn background_helper_already_running_true_for_live_pid() {
        // Phase-10 daemon no-double-spawn: a runtime-state.json with running=true
        // and a LIVE daemon_pid (this very test process) means the helper is
        // already up — the "should spawn?" decision must say NO (true ⇒ skip).
        // Hermetic + offline: no spawn, just the file-based liveness check.
        let (root, paths) = test_paths("helper-live-pid");
        runtime_state(true, std::process::id())
            .write(&paths)
            .expect("write runtime state");
        assert!(
            background_helper_already_running(&paths).expect("liveness check ok"),
            "live recorded pid + running=true ⇒ do not spawn a second daemon"
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn background_helper_already_running_false_for_dead_or_missing() {
        // The inverse guard cases — each must report NOT running (false ⇒ spawn):
        // (1) no state file at all, (2) running=true but a dead/zero pid,
        // (3) a live pid but running=false. None must spawn from this decision.
        let (root, paths) = test_paths("helper-dead-or-missing");
        // (1) No state file yet.
        assert!(
            !background_helper_already_running(&paths).expect("missing state ⇒ ok"),
            "no runtime state ⇒ not running"
        );
        // (2) running=true but pid 0 is never a live process.
        runtime_state(true, 0).write(&paths).expect("write state");
        assert!(
            !background_helper_already_running(&paths).expect("dead pid ⇒ ok"),
            "running=true + dead pid ⇒ not running (spawn)"
        );
        // (3) live pid but running flag is false.
        runtime_state(false, std::process::id())
            .write(&paths)
            .expect("write state");
        assert!(
            !background_helper_already_running(&paths).expect("not-running flag ⇒ ok"),
            "running=false ⇒ not running even with a live pid"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// Put the on-disk facts in one of the four combinations: whether a watcher
    /// is enabled, and whether the helper is running (this process stands in for
    /// a live daemon).
    pub(crate) fn set_helper_facts(paths: &AppPaths, watcher_enabled: bool, helper_running: bool) {
        let mut config = rocm_core::RocmCliConfig::default();
        config.watcher_config_mut("server-recover").enabled = watcher_enabled;
        config.save(paths).expect("save config");
        runtime_state(helper_running, std::process::id())
            .write(paths)
            .expect("write runtime state");
    }

    /// ROCMAI-827: the helper note is derived from both facts, so it cannot
    /// disagree with the state it describes. Each combination: the exact wording,
    /// and the running/outlives flags the uninstall plan acts on.
    #[test]
    fn helper_note_follows_watcher_enabled_and_helper_running() {
        let (root, paths) = test_paths("helper-note-combos");
        paths.ensure().expect("create app dirs");

        set_helper_facts(&paths, false, false);
        let neither = background_helper_note(&paths);
        set_helper_facts(&paths, false, true);
        let idle_running = background_helper_note(&paths).expect("note");
        set_helper_facts(&paths, true, false);
        let enabled_dead = background_helper_note(&paths).expect("note");
        set_helper_facts(&paths, true, true);
        let enabled_running = background_helper_note(&paths).expect("note");
        let _ = fs::remove_dir_all(&root);

        assert_eq!(neither, None);
        assert_eq!(
            idle_running.text,
            "the background automation helper (`rocm daemon`) is still running; no automation is enabled, so it exits on its own within a few seconds"
        );
        assert!(idle_running.running && !idle_running.outlives);
        assert_eq!(
            enabled_dead.text,
            "an automation is enabled (server-recover) but the background automation helper (`rocm daemon`) is not running; `rocm serve` or `rocm automations enable` starts it"
        );
        assert!(!enabled_dead.running && !enabled_dead.outlives);
        assert_eq!(
            enabled_running.text,
            "the background automation helper (`rocm daemon`) keeps running while an automation is enabled (server-recover); to stop it, run `rocm automations disable server-recover`, after which it exits on its own within a few seconds"
        );
        assert!(enabled_running.running && enabled_running.outlives);
    }

    /// The command the note names must exist and accept exactly those arguments,
    /// and its effect (the helper exiting once the last watcher is off) is
    /// `automations-11` in `features/automations.feature` plus the rocmd
    /// `automation_loop_` tests.
    #[test]
    fn the_command_named_by_the_helper_note_parses() {
        use clap::Parser as _;

        let (root, paths) = test_paths("helper-note-command");
        paths.ensure().expect("create app dirs");
        set_helper_facts(&paths, true, true);
        let note = background_helper_note(&paths).expect("a note");
        let _ = fs::remove_dir_all(&root);

        let command = note
            .text
            .split('`')
            .find(|part| part.starts_with("rocm automations disable"))
            .expect("the note names the disable command");
        let argv: Vec<&str> = command.split_whitespace().collect();
        assert!(crate::Cli::try_parse_from(argv).is_ok(), "{command}");
    }
}
