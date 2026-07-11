// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

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
//! runtime built here. The TUI lives entirely in the `rocm-dash-tui` crate.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use rocm_core::{AppPaths, RocmCliConfig, builtin_model_recipes, builtin_watchers};
use rocm_dash_daemon::runner::RunnerOptions;
use rocm_dash_tui::app::{ActiveTab, Focus, ResolvedArgs};
use rocm_dash_tui::ui::automations_manager::AutomationSummary;
use rocm_dash_tui::ui::launcher::LauncherChoice;
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
///
/// Key-sourcing asymmetry (intentional): the chat/OpenAI-compatible key is
/// env-only (`ROCMDASH_CHAT_API_KEY`, `OPENAI_API_KEY`) — this preserves the
/// long-standing chat invariant and is deliberately NOT extended to the secure
/// store. The Anthropic key (see [`anthropic_api_key_for_dash`]) additionally
/// consults the OS secure store via `provider_keys`, because the Anthropic
/// provider was added later with secure-store onboarding. Do not "harmonize"
/// these by adding secure-store lookup here without revisiting the invariant.
fn chat_api_key_from_env() -> Option<String> {
    ["ROCMDASH_CHAT_API_KEY", "OPENAI_API_KEY"]
        .into_iter()
        .find_map(|k| std::env::var(k).ok().filter(|v| !v.is_empty()))
}

/// Anthropic API key for the dash chat seam — sourced env-first (`ANTHROPIC_API_KEY`)
/// then the OS secure store, via the shared `provider_keys` resolver. The key
/// rides in-process through `ResolvedArgs` (NEVER argv). A missing key or an
/// unavailable store yields `None` (the dash still launches; switching to the
/// Anthropic provider then surfaces an actionable error turn).
fn anthropic_api_key_for_dash() -> Option<String> {
    crate::provider_keys::resolve_provider_api_key("anthropic", "ANTHROPIC_API_KEY")
        .ok()
        .map(|k| k.value)
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
///
/// MUST be called on a synchronous thread *before* any tokio runtime is entered:
/// the Anthropic-key secure-store fallback ([`anthropic_api_key_for_dash`]) uses
/// a blocking zbus client that spins its own runtime, which panics ("cannot
/// start a runtime from within a runtime") if invoked from inside `run_async`.
/// The sync entry points `run`/`run_chat` call this and pass the result in.
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
        // Default: not a focused host. `run_focused` sets this per launcher flow.
        focus: None,
        chat_url: t.chat_url.clone(),
        chat_model: t.chat_model.clone(),
        chat_auth_header: t.chat_auth_header.clone(),
        chat_env_url: std::env::var("OPENAI_BASE_URL")
            .ok()
            .filter(|v| !v.is_empty()),
        chat_api_key: chat_api_key_from_env(),
        anthropic_api_key: anthropic_api_key_for_dash(),
        chat_auto_consent: false,
        chat_mock: false,
        model_recipes: model_recipe_summaries(),
        runtimes: runtime_summaries(paths, config),
        automations: automation_summaries(config),
        // The real executor is injected in `run_async` for a live dash; None
        // here keeps demo/replay/mock behaving exactly as today.
        tool_executor: None,
    }
}

/// Build the multi-thread tokio runtime the async daemon/TUI run on. Shared by
/// the synchronous [`run`] and [`run_chat`] entry points (the rest of `rocm` is
/// synchronous; only the dashboard needs an async reactor).
fn build_dashboard_runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building tokio runtime for the dashboard")
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
    // Resolve TUI args — including the OS secure-store (keyring) lookup for the
    // Anthropic key — on this plain synchronous thread, BEFORE entering the tokio
    // runtime. The secure-store path (`provider_keys` → secret-service) uses
    // `zbus::blocking`, which builds its own runtime and `block_on`s internally;
    // doing that on a dash runtime worker thread panics with "Cannot start a
    // runtime from within a runtime". See `run_async`.
    let args = resolved_args(&config, &paths, ActiveTab::Home);
    let rt = build_dashboard_runtime()?;
    rt.block_on(run_async(config, paths, args, replay, chat_mock))
}

/// Where a launcher choice leads. Pure mapping so the hub-loop body stays
/// trivial and the routing is unit-testable without a terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LauncherRoute {
    /// Run in place via the focused host (Set up / Serve / Diagnose).
    Focused(Focus),
    /// Escalate into the full dashboard with the Chat tab focused.
    Chat,
    /// Escalate into the full dashboard (Home).
    Dashboard,
}

/// Map a launcher row to its destination. Set up / Serve / Diagnose run in place
/// (focused host); Chat and Open-dashboard are the only escalations into the
/// full Dash.
const fn launcher_route(choice: LauncherChoice) -> LauncherRoute {
    match choice {
        LauncherChoice::SetUp => LauncherRoute::Focused(Focus::Setup),
        LauncherChoice::Serve => LauncherRoute::Focused(Focus::Serve),
        LauncherChoice::Diagnose => LauncherRoute::Focused(Focus::Examine),
        LauncherChoice::Chat => LauncherRoute::Chat,
        LauncherChoice::OpenDashboard => LauncherRoute::Dashboard,
    }
}

/// Entry point for bare `rocm`: the launcher as a persistent hub.
///
/// Draws the minimal front door; runs the chosen flow; then redraws the menu so
/// the user can run several flows without relaunching. Set up / Serve / Diagnose
/// run in place via the focused host ([`run_focused`]); Chat and Open-dashboard
/// escalate into the full Dash. `q`/`Esc` (the `None` choice) leaves. A flow
/// error breaks the loop and propagates.
pub fn run_launcher(chat_mock: bool) -> Result<()> {
    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths).unwrap_or_default();
    let theme = config.dashboard.tui.theme;
    loop {
        match rocm_dash_tui::ui::launcher::run_launcher(&theme)? {
            None => return Ok(()),
            Some(choice) => match launcher_route(choice) {
                LauncherRoute::Focused(focus) => run_focused(focus)?,
                LauncherRoute::Chat => run_chat(chat_mock)?,
                LauncherRoute::Dashboard => run(None, false, chat_mock)?,
            },
        }
    }
}

/// Entry point for a focused launcher flow (Set up / Serve / Diagnose).
///
/// Opens the dashboard runtime hosting exactly the one overlay for `focus` — no
/// embedded daemon (see [`should_spawn_daemon`]), no tab shell — and returns to
/// the launcher when that overlay is closed at its root. The keyring lookup in
/// [`resolved_args`] runs here on the synchronous thread, before the runtime
/// (nested-runtime invariant).
pub fn run_focused(focus: Focus) -> Result<()> {
    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths)?;
    let mut args = resolved_args(&config, &paths, ActiveTab::Home);
    args.focus = Some(focus);
    let rt = build_dashboard_runtime()?;
    rt.block_on(run_async(config, paths, args, None, false))
}

/// Entry point for interactive `rocm chat`. Opens the unified dashboard with the
/// Chat tab focused. Thin wrapper over the same runtime/`run_async` path as
/// [`run`]; no replay/demo, embedded daemon as usual.
pub fn run_chat(chat_mock: bool) -> Result<()> {
    let paths = AppPaths::discover()?;
    let config = RocmCliConfig::load(&paths)?;
    // See `run`: resolve args (incl. the keyring lookup) before the runtime so the
    // secure-store `zbus::blocking` path never runs on a runtime worker thread.
    let args = resolved_args(&config, &paths, ActiveTab::Chat);
    let rt = build_dashboard_runtime()?;
    rt.block_on(run_async(config, paths, args, None, chat_mock))
}

/// Entry point for `rocm bootstrap setup`. Routes to the same focused Setup host
/// as the launcher's "Set up this system" row — the first-run onboarding wizard
/// (install ROCm SDK / adopt an existing folder), with no daemon or tab shell.
pub fn run_bootstrap() -> Result<()> {
    run_focused(bootstrap_focus())
}

/// The focused flow `rocm bootstrap setup` routes to — the onboarding host,
/// identical to the launcher's "Set up this system".
const fn bootstrap_focus() -> Focus {
    Focus::Setup
}

async fn run_async(
    config: RocmCliConfig,
    paths: AppPaths,
    mut args: ResolvedArgs,
    replay: Option<PathBuf>,
    chat_mock: bool,
) -> Result<()> {
    // `args` is built by the synchronous caller (`run`/`run_chat`) so the keyring
    // lookup inside `resolved_args` never runs on a runtime worker thread (it uses
    // `zbus::blocking`, which would otherwise panic: runtime-within-a-runtime).
    args.replay = replay.clone();
    args.chat_mock = chat_mock;
    // Inject the bin-side tool-execution seam for a live dash only. Demo/replay
    // and the offline chat mock keep `tool_executor = None` and behave as today.
    if !chat_mock && replay.is_none() {
        let executor: rocm_dash_tui::tool_exec::SharedRocmToolExecutor =
            std::sync::Arc::new(crate::dash_seam::BinToolExecutor::new(paths.clone()));
        args.tool_executor = Some(executor);
    }
    // A live daemon is only needed for a connected full dashboard; replay/demo
    // feeds events straight into the TUI, and a focused host draws no telemetry
    // and streams its own job via the job-bridge — both skip the embedded daemon.
    let embedded = if should_spawn_daemon(&args) {
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

/// Whether `run_async` should auto-start the embedded telemetry daemon: only for
/// a live, non-replay FULL dashboard. A focused host (`args.focus.is_some()`)
/// draws no telemetry and streams its own job via the job-bridge, so it needs no
/// daemon — and must not re-surface the socket-crash class for a flow that never
/// uses it. Replay/demo (`args.replay.is_some()`) feeds events straight in.
/// Pure predicate → unit-testable. Call after `args.replay` is set in `run_async`.
const fn should_spawn_daemon(args: &ResolvedArgs) -> bool {
    args.replay.is_none() && args.focus.is_none()
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

    /// EAI-7359 regression: the embedded daemon (`maybe_spawn_embedded_daemon`)
    /// always calls `runner_options(.., enable_docker=false)`, so the vLLM
    /// Prometheus scraper must NOT be gated on `enable_docker` — otherwise it
    /// is permanently dead in the common no-Docker / managed-vLLM case even
    /// though `vllm_prom.rs` has zero Docker dependency (plain HTTP GET).
    /// The scrape stays on by default; `disable_vllm_metrics` is the internal
    /// gate that would turn it off, but it is not currently wired to any CLI
    /// flag or config field, so today it is always `false`.
    #[test]
    fn runner_options_keeps_vllm_metrics_enabled_without_docker() {
        let p = paths();
        let opts = runner_options(&cfg(), &p, false);
        assert!(!opts.enable_docker);
        assert!(
            !opts.disable_vllm_metrics,
            "vLLM metrics must stay on by default even when Docker discovery is off"
        );
    }

    #[test]
    fn bootstrap_routes_to_focused_setup() {
        // `rocm bootstrap setup` must route to the same focused Setup host as the
        // launcher's "Set up this system" row (onboarding, no daemon/tab shell).
        assert_eq!(bootstrap_focus(), Focus::Setup);
    }

    #[test]
    fn launcher_route_maps_every_choice() {
        // Set up / Serve / Diagnose run in place; Chat / Open-dashboard escalate.
        assert_eq!(
            launcher_route(LauncherChoice::SetUp),
            LauncherRoute::Focused(Focus::Setup)
        );
        assert_eq!(
            launcher_route(LauncherChoice::Serve),
            LauncherRoute::Focused(Focus::Serve)
        );
        assert_eq!(
            launcher_route(LauncherChoice::Diagnose),
            LauncherRoute::Focused(Focus::Examine),
        );
        assert_eq!(launcher_route(LauncherChoice::Chat), LauncherRoute::Chat);
        assert_eq!(
            launcher_route(LauncherChoice::OpenDashboard),
            LauncherRoute::Dashboard
        );
    }

    #[test]
    fn resolved_args_default_has_no_focus() {
        // Normal `rocm dash` / `rocm chat` are NOT focused hosts.
        let args = resolved_args(&cfg(), &paths(), ActiveTab::Home);
        assert!(args.focus.is_none());
    }

    #[test]
    fn focused_and_replay_suppress_the_embedded_daemon() {
        // Full live dash → spawn the embedded daemon.
        let mut args = resolved_args(&cfg(), &paths(), ActiveTab::Home);
        assert!(should_spawn_daemon(&args), "full dash spawns the daemon");
        // Focused host → never spawn a daemon (avoids the socket-crash class).
        args.focus = Some(Focus::Examine);
        assert!(!should_spawn_daemon(&args), "focused host: no daemon");
        // Replay also suppresses it.
        args.focus = None;
        args.replay = Some(PathBuf::from("/tmp/x.ndjson"));
        assert!(!should_spawn_daemon(&args), "replay: no daemon");
    }

    #[test]
    fn resolved_args_take_connect_and_theme_from_config() {
        let c = cfg();
        let args = resolved_args(&c, &paths(), ActiveTab::Home);
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
