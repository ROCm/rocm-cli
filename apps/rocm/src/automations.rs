//! `rocm automations` command handler (EAI-6871 D5).
//!
//! Mechanically relocated from `main.rs` with no behavior change — the
//! `dispatch()` call site stays `automations(command)` (re-imported via
//! `use crate::automations::automations;`). Render/policy helpers remain in the
//! crate root and are reached through `crate::` (root items are visible to this
//! descendant module).

use anyhow::{Result, bail};
use rocm_core::{AppPaths, RocmCliConfig, builtin_watcher};

use crate::{AutomationsCommand, render_automations_text, watcher_policy_note};

pub(crate) fn automations(command: Option<AutomationsCommand>) -> Result<()> {
    let paths = AppPaths::discover()?;
    let mut config = RocmCliConfig::load(&paths)?;
    match command.unwrap_or(AutomationsCommand::List) {
        AutomationsCommand::List => {
            print!("{}", render_automations_text(&paths, &config)?);
        }
        AutomationsCommand::Enable { watcher, mode } => {
            let Some(spec) = builtin_watcher(&watcher) else {
                bail!("unknown watcher: {watcher}");
            };
            let entry = config.watcher_config_mut(spec.id);
            entry.enabled = true;
            if let Some(mode) = mode {
                entry.mode = Some(mode.into());
            }
            config.automations.daemon_enabled = true;
            config.save(&paths)?;
            println!("automation watcher enabled");
            println!("  watcher: {}", spec.id);
            println!("  mode: {}", config.effective_watcher_mode(spec).as_str());
            println!("  trigger: {}", spec.trigger);
            if let Some(note) = watcher_policy_note(spec.id) {
                println!("  policy: {note}");
            }
            println!("  config: {}", paths.config_path().display());
            println!(
                "  next step: run `rocmd run --automations-enabled` to start the persistent watcher loop"
            );
        }
        AutomationsCommand::Disable { watcher } => {
            let Some(spec) = builtin_watcher(&watcher) else {
                bail!("unknown watcher: {watcher}");
            };
            let entry = config.watcher_config_mut(spec.id);
            entry.enabled = false;
            if !config
                .automations
                .watchers
                .values()
                .any(|watcher| watcher.enabled)
            {
                config.automations.daemon_enabled = false;
            }
            config.save(&paths)?;
            println!("automation watcher disabled");
            println!("  watcher: {}", spec.id);
            println!("  config: {}", paths.config_path().display());
        }
    }
    Ok(())
}
