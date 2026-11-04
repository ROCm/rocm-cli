// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm config`: the command handler (`config`), which applies each
//! `ConfigCommand` subcommand to the saved configuration, the prompt it uses to
//! read a provider API key (`read_provider_key_from_user`), and the text
//! `rocm config show` prints (`render_config_text`, with
//! `telemetry_policy_summary` for its telemetry line).
//!
//! Mechanically relocated from `main.rs` with no behavior change; see
//! `docs/architecture.md`. `ConfigCommand`, the clap enum the handler matches
//! on, stays in `main.rs` with the rest of the `Command`/`Cli` definition, and
//! `dispatch()` calls `config` through the re-import `use
//! crate::config_cmd::{...};`. `render_config_text` is also `pub(crate)`:
//! `chat.rs`'s read-only handling of `rocm config show` reaches it as
//! `crate::render_config_text` through that re-import. `read_provider_key_from_user`
//! and `telemetry_policy_summary` have no caller outside this module and stay
//! private.
//!
//! The two `render_config_text` regression tests live in this module's own
//! `mod tests`, reaching the shared `test_paths` fixture through
//! `crate::tests::`.

use std::fmt::Write as _;
use std::fs;
use std::io::{self, Read};

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, RocmCliConfig, interactive_terminal};

use crate::cli_audit::record_cli_audit_event;
use crate::runtimes_cmd::active_runtime_marker_path;
use crate::{ConfigCommand, provider_keys, provider_name, providers};

pub(crate) fn config(command: ConfigCommand) -> Result<()> {
    let paths = AppPaths::discover()?;
    let mut config = RocmCliConfig::load(&paths)?;

    match command {
        ConfigCommand::Show => {
            print!("{}", render_config_text(&paths, &config));
        }
        ConfigCommand::SetEngine {
            engine,
            runtime_id,
            env_id,
            clear,
        } => {
            let entry = config.engine_config_mut(&engine);
            if clear {
                entry.preferred_runtime_id = None;
                entry.preferred_env_id = None;
            } else if let Some(runtime_id) = runtime_id {
                entry.preferred_runtime_id = Some(runtime_id);
                entry.preferred_env_id = None;
            } else if let Some(env_id) = env_id {
                entry.preferred_env_id = Some(env_id);
                entry.preferred_runtime_id = None;
            } else {
                bail!("set-engine requires --runtime-id, --env-id, or --clear");
            }
            config.save(&paths)?;
            println!("updated engine config for {engine}");
        }
        ConfigCommand::SetDefaultEngine { engine } => {
            config.default_engine = Some(engine.clone());
            config.save(&paths)?;
            println!("default engine set to {engine}");
        }
        ConfigCommand::ClearDefaultEngine => {
            config.default_engine = None;
            config.save(&paths)?;
            println!("default engine cleared");
        }
        ConfigCommand::SetDefaultRuntime { runtime_id } => {
            config.default_runtime_id = Some(runtime_id.clone());
            config.active_runtime_key = None;
            config.previous_runtime_key = None;
            config.save(&paths)?;
            let _ = fs::remove_file(active_runtime_marker_path(&paths));
            println!("default runtime set to {runtime_id}");
        }
        ConfigCommand::ClearDefaultRuntime => {
            config.default_runtime_id = None;
            config.active_runtime_key = None;
            config.previous_runtime_key = None;
            config.save(&paths)?;
            let _ = fs::remove_file(active_runtime_marker_path(&paths));
            println!("default runtime cleared");
        }
        ConfigCommand::SetTelemetry { mode } => {
            config.telemetry.mode = mode.as_str().to_owned();
            config.save(&paths)?;
            println!("telemetry mode set to {}", mode.as_str());
            println!("  policy: {}", telemetry_policy_summary(&config.telemetry));
        }
        ConfigCommand::SetPermissions { mode } => {
            config.permissions.mode = mode.as_str().to_owned();
            config.save(&paths)?;
            record_cli_audit_event(
                &paths,
                "permissions",
                "set_mode",
                "info",
                format!("permissions mode set to {}", mode.as_str()),
                None,
            );
            println!("permissions mode set to {}", mode.as_str());
        }
        ConfigCommand::SetPlannerProvider { provider } => {
            let provider = provider_name(provider);
            config.planner_provider = Some(provider.to_owned());
            config.save(&paths)?;
            println!("planner provider set to {provider}");
            if provider != "local" && !config.provider_enabled(provider) {
                println!(
                    "  next step: rocm config enable-provider {provider} before provider-assisted planning can send prompts"
                );
            }
        }
        ConfigCommand::ClearPlannerProvider => {
            config.planner_provider = None;
            config.save(&paths)?;
            println!("planner provider cleared");
        }
        ConfigCommand::EnableProvider { provider } => {
            let provider = provider_name(provider);
            if provider == "local" {
                bail!("local provider is always enabled and does not send prompts to a cloud API");
            }
            config.provider_config_mut(provider).enabled = true;
            config.save(&paths)?;
            println!("provider {provider} enabled for prompt sending");
            match providers::provider_key_status_text(provider) {
                Ok(status) if status.starts_with("no key saved") => {
                    println!("  key: {status}");
                    println!("  next step: rocm config set-provider-key {provider}");
                }
                Ok(status) => println!("  key: {status}"),
                Err(error) => println!("  key: unavailable ({error})"),
            }
        }
        ConfigCommand::DisableProvider { provider } => {
            let provider = provider_name(provider);
            if provider == "local" {
                bail!("local provider is always enabled and does not send prompts to a cloud API");
            }
            config.provider_config_mut(provider).enabled = false;
            config.save(&paths)?;
            println!("provider {provider} disabled for prompt sending");
        }
        ConfigCommand::SetProviderKey { provider } => {
            let provider = provider_name(provider);
            if provider == "local" {
                bail!("local provider does not use a cloud API key");
            }
            let key = read_provider_key_from_user(provider)?;
            let status = provider_keys::store_provider_credential(provider, &key)?;
            println!("{provider} API key saved");
            println!(
                "  key: {}",
                provider_keys::provider_key_status_label(&status)
            );
            println!(
                "  prompt sending: {}",
                if config.provider_enabled(provider) {
                    "enabled"
                } else {
                    "disabled"
                }
            );
            if !config.provider_enabled(provider) {
                println!("  next step: rocm config enable-provider {provider}");
            }
        }
        ConfigCommand::ClearProviderKey { provider } => {
            let provider = provider_name(provider);
            if provider == "local" {
                bail!("local provider does not use a cloud API key");
            }
            let status = provider_keys::remove_provider_credential(provider)?;
            println!("{provider} API key cleared");
            println!(
                "  key: {}",
                provider_keys::provider_key_status_label(&status)
            );
            println!(
                "  prompt sending: {}",
                if config.provider_enabled(provider) {
                    "enabled"
                } else {
                    "disabled"
                }
            );
        }
    }

    Ok(())
}

fn read_provider_key_from_user(provider: &str) -> Result<String> {
    let key = if interactive_terminal() {
        rpassword::prompt_password(format!("Paste {provider} API key: "))
            .context("failed to read provider API key")?
    } else {
        let mut input = String::new();
        io::stdin()
            .read_to_string(&mut input)
            .context("failed to read provider API key from stdin")?;
        input
    };
    let key = key.trim().to_owned();
    if key.is_empty() {
        bail!("{provider} API key was empty; nothing was saved");
    }
    Ok(key)
}

pub(crate) fn render_config_text(paths: &AppPaths, config: &RocmCliConfig) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "rocm config");
    let _ = writeln!(output, "  file: {}", paths.config_path().display());
    let _ = writeln!(
        output,
        "  default_engine: {}",
        config
            .default_engine
            .as_deref()
            .unwrap_or("<platform default>")
    );
    let _ = writeln!(
        output,
        "  default_runtime_id: {}",
        config.default_runtime_id.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(
        output,
        "  active_runtime_key: {}",
        config.active_runtime_key.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(
        output,
        "  previous_runtime_key: {}",
        config.previous_runtime_key.as_deref().unwrap_or("<unset>")
    );
    let _ = writeln!(
        output,
        "  onboarding_dismissed: {}",
        config.onboarding_dismissed
    );
    let _ = writeln!(
        output,
        "  telemetry_mode: {}",
        config.telemetry.mode_label()
    );
    let _ = writeln!(
        output,
        "  telemetry_policy: {}",
        telemetry_policy_summary(&config.telemetry)
    );
    let _ = writeln!(
        output,
        "  planner_provider: {}",
        config.planner_provider.as_deref().unwrap_or("<off>")
    );
    let _ = writeln!(output, "  providers:");
    for provider in ["local", "openai", "anthropic"] {
        let key_status = providers::provider_key_status_text(provider)
            .unwrap_or_else(|error| format!("key status unavailable: {error}"));
        let _ = writeln!(
            output,
            "    {provider}: {}",
            if config.provider_enabled(provider) {
                "enabled"
            } else {
                "disabled"
            }
        );
        let _ = writeln!(output, "      key: {key_status}");
    }
    if config.engines.is_empty() {
        let _ = writeln!(output, "  engines: none");
        return output;
    }
    for (engine, entry) in &config.engines {
        let _ = writeln!(output, "  engine: {engine}");
        let _ = writeln!(
            output,
            "    preferred_runtime_id: {}",
            entry.preferred_runtime_id.as_deref().unwrap_or("<unset>")
        );
        let _ = writeln!(
            output,
            "    preferred_env_id: {}",
            entry.preferred_env_id.as_deref().unwrap_or("<unset>")
        );
        let _ = writeln!(
            output,
            "    last_installed_runtime_id: {}",
            entry
                .last_installed_runtime_id
                .as_deref()
                .unwrap_or("<unset>")
        );
        let _ = writeln!(
            output,
            "    last_installed_env_id: {}",
            entry.last_installed_env_id.as_deref().unwrap_or("<unset>")
        );
    }
    output
}

fn telemetry_policy_summary(telemetry: &rocm_core::TelemetryConfig) -> &'static str {
    if telemetry.local_inspection_enabled() {
        "local amd-smi inspection only; no external reporting is implemented"
    } else if telemetry.known_mode() {
        "disabled; no local polling or external reporting is implemented"
    } else {
        "unknown mode treated as disabled; set `rocm config set-telemetry local|off`"
    }
}

#[cfg(test)]
mod tests {
    use rocm_core::TELEMETRY_MODE_OFF;

    use super::*;
    use crate::tests::test_paths;

    #[test]
    fn render_config_text_includes_default_runtime() {
        let (_root, paths) = test_paths("config-default-runtime");
        let config = RocmCliConfig {
            default_runtime_id: Some("therock-release:gfx120X-all".to_owned()),
            ..RocmCliConfig::default()
        };

        let rendered = render_config_text(&paths, &config);

        assert!(rendered.contains("default_runtime_id: therock-release:gfx120X-all"));
        assert!(rendered.contains("active_runtime_key: <unset>"));
    }

    #[test]
    fn render_config_text_includes_telemetry_policy() {
        let (_root, paths) = test_paths("config-telemetry-policy");
        let mut config = RocmCliConfig::default();

        let local = render_config_text(&paths, &config);
        assert!(local.contains("telemetry_mode: local"));
        assert!(local.contains("telemetry_policy: local amd-smi inspection only"));
        assert!(local.contains("no external reporting is implemented"));
        assert!(local.contains("  providers:"));
        assert!(local.contains("    local: enabled"));
        assert!(local.contains("    openai: disabled"));
        assert!(local.contains("    anthropic: disabled"));

        config.telemetry.mode = TELEMETRY_MODE_OFF.to_owned();
        config.provider_config_mut("openai").enabled = true;
        let off = render_config_text(&paths, &config);
        assert!(off.contains("telemetry_mode: off"));
        assert!(off.contains("telemetry_policy: disabled"));
        assert!(off.contains("no local polling"));
        assert!(off.contains("    openai: enabled"));
    }
}
