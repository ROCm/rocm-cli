// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm setup`: the command handler (`setup`, which runs `setup status` by
//! default and `setup reset`), the text `rocm setup status` prints
//! (`render_setup_status_text`) and the reset that clears the saved first-time
//! prompt state (`reset_setup_prompt_state`).
//!
//! Mechanically relocated from `main.rs` with no behavior change; see
//! `docs/architecture.md`. `SetupCommand`, the clap enum `setup` matches on,
//! lives in `cli.rs` with the rest of the `Command`/`Cli` definition.
//! `dispatch()` calls `setup` through the re-import `use
//! crate::setup_cmd::{...};`, and `chat.rs`'s read-only handling of `rocm setup`
//! and `rocm setup status` reaches `render_setup_status_text` as
//! `crate::render_setup_status_text` through the same re-import; both are
//! `pub(crate)`. `reset_setup_prompt_state` has no caller outside this module
//! and stays private.
//!
//! The three regression tests for this cluster live in this module's own
//! `mod tests`, reaching the shared `test_paths` fixture through
//! `crate::tests::` and `write_test_pip_runtime` through `crate::test_support::`.
//! `setup_reset_cli_output_is_plain_and_persists_first_time_prompt` pins the
//! wording of the reset message; the behavior it describes is proven by the
//! dashboard test its comment names.

use std::fmt::Write as _;

use anyhow::Result;
use rocm_core::{AppPaths, RocmCliConfig};

use crate::SetupCommand;
use crate::runtimes_cmd::validate_runtime_manifest_for_activation;
use crate::therock;

pub(crate) fn setup(command: Option<SetupCommand>) -> Result<()> {
    let paths = AppPaths::discover()?;
    let mut config = RocmCliConfig::load(&paths)?;
    match command.unwrap_or(SetupCommand::Status) {
        SetupCommand::Status => {
            print!("{}", render_setup_status_text(&paths, &config)?);
        }
        SetupCommand::Reset { yes: _ } => {
            print!("{}", reset_setup_prompt_state(&paths, &mut config)?);
        }
    }
    Ok(())
}

pub(crate) fn render_setup_status_text(paths: &AppPaths, config: &RocmCliConfig) -> Result<String> {
    let manifests = therock::load_runtime_manifests(paths)?;
    let active_manifest = config
        .active_runtime_key
        .as_deref()
        .and_then(|runtime_key| {
            manifests
                .iter()
                .find(|manifest| manifest.runtime_key.eq_ignore_ascii_case(runtime_key))
        });
    let active_ready = active_manifest
        .is_some_and(|manifest| validate_runtime_manifest_for_activation(manifest).is_ok());
    let state = if config.setup.completed && active_ready {
        "completed"
    } else if config.setup.completed {
        "completed; active runtime needs attention"
    } else if active_ready {
        "runtime ready; setup not completed"
    } else if config.onboarding_dismissed {
        "setup dismissed"
    } else {
        "first-time setup available - open manually via `rocm dash`'s Observe tab with `n`"
    };

    let mut output = String::new();
    let _ = writeln!(output, "ROCm setup");
    let _ = writeln!(output, "  status: {state}");
    if let Some(root) = config.setup.therock_venv.as_ref() {
        let _ = writeln!(output, "  install folder: {}", root.display());
    }
    if let Some(runtime_key) = config.active_runtime_key.as_deref() {
        let _ = writeln!(output, "  active_runtime_key: {runtime_key}");
    }
    match active_manifest {
        Some(manifest) => {
            let _ = writeln!(output, "  active_runtime_id: {}", manifest.runtime_id);
            let status = if active_ready { "ready" } else { "not_ready" };
            let _ = writeln!(output, "  active_runtime_status: {status}");
        }
        None if config.active_runtime_key.is_some() => {
            let _ = writeln!(output, "  active_runtime_status: missing_manifest");
        }
        None => {
            let _ = writeln!(output, "  active_runtime_status: <unset>");
        }
    }
    let _ = writeln!(output, "  help: run `rocm help` to see how to use rocm-cli");
    Ok(output)
}

fn reset_setup_prompt_state(paths: &AppPaths, config: &mut RocmCliConfig) -> Result<String> {
    config.onboarding_dismissed = false;
    config.setup.completed = false;
    config.save(paths)?;
    Ok([
        "Onboarding will not reopen automatically - open it from `rocm dash`'s Observe tab with `n`.",
        "ROCm installs were not deleted.",
        "Installed ROCm folders, API keys, and provider settings were kept.",
        "",
    ]
    .join("\n"))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::test_support::write_test_pip_runtime;
    use crate::tests::test_paths;

    #[test]
    fn setup_reset_cli_output_is_plain_and_persists_first_time_prompt() -> Result<()> {
        let (_root, paths) = test_paths("setup-reset-cli");
        let mut config = RocmCliConfig {
            onboarding_dismissed: true,
            setup: rocm_core::SetupConfig {
                completed: true,
                therock_venv: Some(paths.data_dir.join("envs").join("default")),
                cli_install_dir: None,
            },
            ..Default::default()
        };
        config.provider_config_mut("openai").enabled = true;
        config.save(&paths)?;

        let rendered = reset_setup_prompt_state(&paths, &mut config)?;

        // The claim itself (onboarding only opens via an explicit `n` on the
        // Observe tab, never automatically) is proven by
        // `crates/rocm-dash-tui/src/app/focus.rs`'s
        // `startup_focus_gate_only_opens_onboarding_for_explicit_setup_focus`
        // test and the `onboarding.rs` module doc — this assertion only
        // guards the string, not the behavior.
        assert!(rendered.contains("Onboarding will not reopen automatically"));
        assert!(rendered.contains("ROCm installs were not deleted"));
        assert!(rendered.contains("API keys"));
        // ROCMAI-543: the assertions above stop before the ` - open it from
        // `rocm dash`'s Observe tab` clause, so they stay green with the em
        // dash put back. This covers the whole block that `rocm setup --reset`
        // prints, including any line added to it later.
        assert!(
            rendered.is_ascii(),
            "the setup-reset prompt must be ASCII-only; non-ASCII punctuation renders as \
             mojibake on a Windows console using the OEM code page (ROCMAI-543): {rendered}"
        );
        assert!(!rendered.contains("request plan"));
        assert!(!rendered.contains("planner:"));
        assert!(!rendered.contains("tool_schema"));

        let saved = RocmCliConfig::load(&paths)?;
        assert!(!saved.onboarding_dismissed);
        assert!(!saved.setup.completed);
        assert!(saved.setup.therock_venv.is_some());
        assert!(saved.provider_enabled("openai"));
        Ok(())
    }

    #[test]
    fn setup_status_reports_completed_active_runtime() -> Result<()> {
        let (root, paths) = test_paths("setup-status-completed-runtime");
        let manifest = write_test_pip_runtime(
            &paths,
            "release-pip-gfx120x-all-status",
            "therock-release:gfx120X-all",
            "7.13.0",
            1,
        )?;
        let config = RocmCliConfig {
            default_runtime_id: Some(manifest.runtime_id.clone()),
            active_runtime_key: Some(manifest.runtime_key.clone()),
            setup: rocm_core::SetupConfig {
                completed: true,
                therock_venv: Some(manifest.install_root.clone()),
                cli_install_dir: None,
            },
            ..Default::default()
        };

        let rendered = render_setup_status_text(&paths, &config)?;

        assert!(rendered.contains("status: completed"), "{rendered}");
        assert!(
            rendered.contains(&format!(
                "install folder: {}",
                manifest.install_root.display()
            )),
            "{rendered}"
        );
        assert!(
            rendered.contains("active_runtime_key: release-pip-gfx120x-all-status"),
            "{rendered}"
        );
        assert!(
            rendered.contains("active_runtime_id: therock-release:gfx120X-all"),
            "{rendered}"
        );
        assert!(
            rendered.contains("active_runtime_status: ready"),
            "{rendered}"
        );
        assert!(rendered.contains("rocm help"), "{rendered}");

        let _ = fs::remove_dir_all(root);
        Ok(())
    }

    #[test]
    fn setup_status_reports_first_time_when_not_completed() -> Result<()> {
        let (_root, paths) = test_paths("setup-status-first-time");
        let config = RocmCliConfig::default();

        let rendered = render_setup_status_text(&paths, &config)?;

        // See the pointer comment in
        // `setup_reset_cli_output_is_plain_and_persists_first_time_prompt`
        // above: this only guards the string, not the underlying behavior.
        assert!(rendered.contains("status: first-time setup available - open manually"));
        assert!(rendered.contains("active_runtime_status: <unset>"));
        Ok(())
    }
}
