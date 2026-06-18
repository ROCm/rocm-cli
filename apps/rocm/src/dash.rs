//! `rocm dash` — launch the unified telemetry dashboard.
//!
//! Folds the rocm-dash launch verb into the `rocm` binary. It builds the
//! telemetry daemon's [`RunnerOptions`] — wiring `services_dir =
//! AppPaths::services_dir()` so the managed services that `rocm serve --managed`
//! writes there surface live in the dashboard (the D7 registry→scrape→`gen_tps`
//! seam) — auto-starts an embedded daemon when none is already listening, and
//! runs the ratatui dashboard TUI.
//!
//! The rest of `rocm` is synchronous; the async daemon/TUI run on a tokio
//! runtime built here. The two ratatui majors (0.29 in `tui.rs`, 0.30 in
//! `rocm-dash-tui`) coexist, each confined to its crate.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use rocm_core::{AppPaths, RocmCliConfig, builtin_model_recipes, builtin_watchers};
use rocm_dash_daemon::runner::RunnerOptions;
use rocm_dash_tui::app::{ActiveTab, ResolvedArgs};
use rocm_dash_tui::ui::automations_manager::AutomationSummary;
use rocm_dash_tui::ui::model_picker::ModelRecipeSummary;
use rocm_dash_tui::ui::runtime_manager::RuntimeSummary;

use crate::therock;

/// Build the telemetry-daemon options from the unified dashboard config.
///
/// `services_dir` is the load-bearing wire: pointing it at
/// [`AppPaths::services_dir`] makes the daemon discover managed services written
/// by `rocm serve --managed` and surface their `gen_tps` in the dashboard.
// Used by the #[cfg(unix)] embedded-daemon path and by tests; on Windows the
// non-test build never calls it (the daemon is unix-only), so allow dead_code there.
#[cfg_attr(windows, allow(dead_code))]
pub fn runner_options(
    config: &RocmCliConfig,
    paths: &AppPaths,
    enable_docker: bool,
) -> RunnerOptions {
    let d = &config.dashboard.daemon;
    RunnerOptions {
        bench_csv: d.bench_results_dir.clone(),
        enable_docker,
        image_patterns: None,
        gpu_tick: Duration::from_secs_f64(d.gpu_tick_secs),
        discovery_tick: Duration::from_secs_f64(d.discovery_tick_secs),
        instance_tick: Duration::from_secs_f64(d.instance_tick_secs),
        disable_vllm_metrics: false,
        vllm_metrics_host: "127.0.0.1".into(),
        // Lemonade discovery stays opt-in (mirrors a no-flag embedded daemon).
        enable_lemonade: false,
        lemonade_host: "127.0.0.1".into(),
        lemonade_port: 13305,
        persist_dir: Some(paths.telemetry_state_dir()),
        // D7 seam consumer: managed services from `rocm serve --managed`.
        services_dir: Some(paths.services_dir()),
        // amd-smi ships inside the managed runtime wheel's bin dir, not on PATH;
        // resolve the path so the GPU collector can find it.
        amd_smi_binary: Some(rocm_core::resolve_amd_smi_binary()),
    }
}

/// API key precedence — sourced from the environment ONLY (never TOML/CLI/source/
/// logs); see the chat invariant.
fn chat_api_key_from_env() -> Option<String> {
    ["ROCMDASH_CHAT_API_KEY", "OPENAI_API_KEY"]
        .into_iter()
        .find_map(|k| std::env::var(k).ok().filter(|v| !v.is_empty()))
}

/// Adapt the built-in `rocm-core` model recipes into the TUI-local summaries the
/// serve-wizard picker consumes (the bin owns the `rocm-core` dependency so the
/// dash crates stay free of it).
fn model_recipe_summaries() -> Vec<ModelRecipeSummary> {
    builtin_model_recipes()
        .into_iter()
        .map(|r| ModelRecipeSummary {
            id: r.canonical_model_id,
            aliases: r.aliases,
            task: r.task,
            preferred_engine: r.preferred_engines.into_iter().next(),
        })
        .collect()
}

/// Adapt the registered ROCm runtimes into the TUI-local summaries the runtime
/// manager consumes (the bin owns `rocm-core` / `therock`, so the dash crates
/// stay free of them). Tolerant: a load failure yields an empty list rather
/// than blocking the dashboard launch — the in-TUI refresh re-reads live.
fn runtime_summaries(paths: &AppPaths, config: &RocmCliConfig) -> Vec<RuntimeSummary> {
    let Ok(manifests) = therock::load_runtime_manifests(paths) else {
        return Vec::new();
    };
    let active_key = config.active_runtime_key.as_deref();
    let prev_key = config.previous_runtime_key.as_deref();
    let default_id = config.default_runtime_id.as_deref();
    // Mirror `render_runtimes_text`: a runtime is active by an explicit
    // active_runtime_key, or — absent one — by being the single manifest whose
    // runtime_id matches the configured default_runtime_id.
    let default_matches: Vec<&str> = manifests
        .iter()
        .filter(|m| Some(m.runtime_id.as_str()) == default_id)
        .map(|m| m.runtime_key.as_str())
        .collect();
    let single_default_key = if active_key.is_none() && default_matches.len() == 1 {
        Some(default_matches[0].to_string())
    } else {
        None
    };
    manifests
        .iter()
        .map(|m| {
            let active = active_key == Some(m.runtime_key.as_str())
                || single_default_key.as_deref() == Some(m.runtime_key.as_str());
            let rollback = prev_key == Some(m.runtime_key.as_str());
            RuntimeSummary {
                key: m.runtime_key.clone(),
                id: m.runtime_id.clone(),
                channel: m.channel.clone(),
                version: m.version.clone(),
                root: m.install_root.display().to_string(),
                active,
                rollback,
            }
        })
        .collect()
}

/// Adapt the built-in background checks into the TUI-local summaries the
/// automations manager consumes (enabled-state + effective mode come from the
/// unified config; the bin owns `rocm-core`).
fn automation_summaries(config: &RocmCliConfig) -> Vec<AutomationSummary> {
    builtin_watchers()
        .iter()
        .map(|w| AutomationSummary {
            id: w.id.to_string(),
            summary: w.summary.to_string(),
            enabled: config.watcher_enabled(w),
            mode: config.effective_watcher_mode(w).as_str().to_string(),
        })
        .collect()
}

/// Resolve the TUI args from the unified config + environment.
pub fn resolved_args(
    config: &RocmCliConfig,
    paths: &AppPaths,
    initial_tab: ActiveTab,
) -> ResolvedArgs {
    let t = &config.dashboard.tui;
    ResolvedArgs {
        connect: t.connect.clone(),
        token: config.dashboard.daemon.token.clone(),
        theme: t.theme.clone(),
        replay: None,
        initial_tab,
        chat_url: t.chat_url.clone(),
        chat_model: t.chat_model.clone(),
        chat_auth_header: t.chat_auth_header.clone(),
        chat_env_url: std::env::var("OPENAI_BASE_URL")
            .ok()
            .filter(|v| !v.is_empty()),
        chat_api_key: chat_api_key_from_env(),
        chat_auto_consent: false,
        chat_mock: false,
        model_recipes: model_recipe_summaries(),
        runtimes: runtime_summaries(paths, config),
        automations: automation_summaries(config),
    }
}

/// Entry point for `rocm dash`. Builds a tokio runtime and runs the dashboard.
pub fn run(replay: Option<PathBuf>, demo: bool, chat_mock: bool) -> Result<()> {
    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths)?;
    // `--demo` writes a deterministic synthetic session and replays it, so the
    // dashboard shows populated data with no GPU and no daemon.
    let replay = if demo {
        let path = std::env::temp_dir().join("rocm-dash-demo.ndjson");
        rocm_dash_daemon::demo::generate_file(
            &rocm_dash_daemon::demo::DemoOptions::default(),
            &path,
        )
        .context("generating the demo session")?;
        Some(path)
    } else {
        replay
    };
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building tokio runtime for the dashboard")?;
    rt.block_on(run_async(config, paths, replay, chat_mock))
}

async fn run_async(
    config: RocmCliConfig,
    paths: AppPaths,
    replay: Option<PathBuf>,
    chat_mock: bool,
) -> Result<()> {
    let mut args = resolved_args(&config, &paths, ActiveTab::Overview);
    args.replay = replay.clone();
    args.chat_mock = chat_mock;
    // A live daemon is only needed when connecting; replay/demo feeds events
    // straight into the TUI, so skip the embedded daemon in that mode.
    let embedded = if replay.is_none() {
        maybe_spawn_embedded_daemon(&args.connect, &config, &paths).await
    } else {
        None
    };

    let result = rocm_dash_tui::app::run(args)
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()));

    // Tidy up the embedded daemon on exit (best-effort).
    if let Some((handle, socket)) = embedded {
        handle.abort();
        if let Some(path) = socket {
            let _ = std::fs::remove_file(path);
        }
    }
    result
}

/// Auto-start an embedded telemetry daemon when no local one is already
/// listening, so `rocm dash` works without a separate `rocm daemon` terminal.
/// Returns the task handle + socket to clean up on exit, or `None` when an
/// existing daemon was found (we connect to it instead).
async fn maybe_spawn_embedded_daemon(
    connect: &str,
    config: &RocmCliConfig,
    paths: &AppPaths,
) -> Option<(tokio::task::JoinHandle<()>, Option<PathBuf>)> {
    #[cfg(unix)]
    {
        // Only auto-manage a LOCAL unix-socket daemon.
        let target = connect.strip_prefix("unix:")?;
        if tokio::net::UnixStream::connect(target).await.is_ok() {
            return None; // a daemon already answers here
        }

        let opts = runner_options(config, paths, false);
        let listen = connect.to_string();
        let socket = Some(PathBuf::from(target));

        let handle = tokio::spawn(async move {
            if let Err(e) = rocm_dash_daemon::server::run(&listen, opts).await {
                eprintln!("rocm: embedded telemetry daemon exited: {e:#}");
            }
        });
        // Poll until the daemon has bound and is accepting connections.
        // A fixed sleep is race-prone on slow or loaded systems.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
        loop {
            if tokio::net::UnixStream::connect(target).await.is_ok() {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                break; // Proceed anyway; the TUI client will retry with backoff.
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        Some((handle, socket))
    }
    #[cfg(windows)]
    {
        let _ = (connect, config, paths);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> RocmCliConfig {
        RocmCliConfig::default()
    }

    fn paths() -> AppPaths {
        AppPaths {
            config_dir: PathBuf::from("/tmp/rocm-cfg"),
            data_dir: PathBuf::from("/tmp/rocm-data"),
            cache_dir: PathBuf::from("/tmp/rocm-cache"),
        }
    }

    #[test]
    fn runner_options_wires_services_dir_to_registry() {
        let p = paths();
        let opts = runner_options(&cfg(), &p, false);
        // The serve→dashboard wire: daemon reads the managed-service registry.
        assert_eq!(opts.services_dir, Some(p.services_dir()));
        assert_eq!(opts.persist_dir, Some(p.telemetry_state_dir()));
        assert!(!opts.enable_docker);
    }

    #[test]
    fn resolved_args_take_connect_and_theme_from_config() {
        let c = cfg();
        let args = resolved_args(&c, &paths(), ActiveTab::Overview);
        assert_eq!(args.connect, c.dashboard.tui.connect);
        assert_eq!(args.theme, c.dashboard.tui.theme);
        assert!(!args.chat_mock);
        assert!(args.replay.is_none());
        // The serve-wizard recipe picker is fed from the built-in recipes.
        assert!(
            !args.model_recipes.is_empty(),
            "built-in model recipes flow through to the wizard"
        );
    }

    #[test]
    fn model_recipe_summaries_carry_id_and_engine() {
        let records = builtin_model_recipes();
        let summaries = model_recipe_summaries();
        assert!(!summaries.is_empty());
        assert_eq!(summaries.len(), records.len(), "no recipes dropped");
        // Every summary has a non-empty canonical id (the serve argv target).
        assert!(summaries.iter().all(|s| !s.id.is_empty()));
        // The preferred engine is actually plumbed (not zeroed) — at least one
        // recipe declares one, and the first summary mirrors its record.
        assert!(
            summaries.iter().any(|s| s.preferred_engine.is_some()),
            "preferred_engine forwarded"
        );
        let first = &records[0];
        let first_summary = &summaries[0];
        assert_eq!(first_summary.id, first.canonical_model_id);
        assert_eq!(first_summary.aliases, first.aliases);
        assert_eq!(first_summary.task, first.task);
        assert_eq!(
            first_summary.preferred_engine.as_ref(),
            first.preferred_engines.first()
        );
    }
}
