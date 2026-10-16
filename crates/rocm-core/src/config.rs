// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The unified `rocm-cli` config and its nested sections.
//!
//! `RocmCliConfig`: engine/provider/watcher/telemetry/permissions/setup
//! knobs (including `WatcherMode`, the persisted watcher-review-mode enum;
//! the built-in watcher catalog itself lives in `automation.rs`), and the
//! rocm-dash dashboard sub-config (daemon + TUI), including its one-shot
//! legacy-TOML migration.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::AppPaths;
use crate::BuiltinWatcherSpec;
use crate::user_runtime_dir;

/// Automation watcher review mode: how much autonomy a watcher has to act on
/// its own trigger without a human approving first.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum WatcherMode {
    Observe,
    Propose,
    Contained,
}

impl WatcherMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Observe => "observe",
            Self::Propose => "propose",
            Self::Contained => "contained",
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngineUserConfig {
    #[serde(default)]
    pub preferred_runtime_id: Option<String>,
    #[serde(default)]
    pub preferred_env_id: Option<String>,
    #[serde(default)]
    pub last_installed_runtime_id: Option<String>,
    #[serde(default)]
    pub last_installed_env_id: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WatcherUserConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub mode: Option<WatcherMode>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AutomationsConfig {
    #[serde(default)]
    pub daemon_enabled: bool,
    #[serde(default)]
    pub watchers: BTreeMap<String, WatcherUserConfig>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProviderUserConfig {
    #[serde(default)]
    pub enabled: bool,
}

pub const TELEMETRY_MODE_LOCAL: &str = "local";
pub const TELEMETRY_MODE_OFF: &str = "off";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    #[serde(default = "default_telemetry_mode")]
    pub mode: String,
}

pub const PERMISSIONS_MODE_ASK: &str = "ask";
pub const PERMISSIONS_MODE_FULL_ACCESS: &str = "full_access";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionsConfig {
    #[serde(default = "default_permissions_mode")]
    pub mode: String,
}

impl Default for PermissionsConfig {
    fn default() -> Self {
        Self {
            mode: PERMISSIONS_MODE_ASK.to_owned(),
        }
    }
}

impl PermissionsConfig {
    pub fn mode_label(&self) -> &str {
        let mode = self.mode.trim();
        if mode.eq_ignore_ascii_case(PERMISSIONS_MODE_FULL_ACCESS) {
            PERMISSIONS_MODE_FULL_ACCESS
        } else {
            PERMISSIONS_MODE_ASK
        }
    }

    pub fn full_access_enabled(&self) -> bool {
        self.mode_label() == PERMISSIONS_MODE_FULL_ACCESS
    }
}

fn default_permissions_mode() -> String {
    PERMISSIONS_MODE_ASK.to_owned()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SetupConfig {
    /// Described in: `rocm setup status`/`reset` output, their `--help` doc
    /// comments, README.md's Setup section, docs/testing.md,
    /// docs/manual-testing.md, and the module doc comment in
    /// `crates/rocm-dash-tui/src/ui/onboarding.rs`. Nothing reads this to
    /// auto-open onboarding — grep this field, not a phrase, before touching
    /// any of those surfaces.
    #[serde(default)]
    pub completed: bool,
    #[serde(default)]
    pub therock_venv: Option<PathBuf>,
    #[serde(default)]
    pub cli_install_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ManagedToolConfig {
    #[serde(default)]
    pub path: Option<PathBuf>,
    #[serde(default)]
    pub managed: bool,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            mode: TELEMETRY_MODE_LOCAL.to_owned(),
        }
    }
}

impl TelemetryConfig {
    pub fn mode_label(&self) -> &str {
        let mode = self.mode.trim();
        if mode.is_empty() {
            TELEMETRY_MODE_LOCAL
        } else {
            mode
        }
    }

    pub fn local_inspection_enabled(&self) -> bool {
        self.mode_label().eq_ignore_ascii_case(TELEMETRY_MODE_LOCAL)
    }

    pub fn known_mode(&self) -> bool {
        self.mode_label().eq_ignore_ascii_case(TELEMETRY_MODE_LOCAL)
            || self.mode_label().eq_ignore_ascii_case(TELEMETRY_MODE_OFF)
    }
}

fn default_telemetry_mode() -> String {
    TELEMETRY_MODE_LOCAL.to_owned()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RocmCliConfig {
    #[serde(default)]
    pub default_engine: Option<String>,
    #[serde(default)]
    pub default_runtime_id: Option<String>,
    #[serde(default)]
    pub active_runtime_key: Option<String>,
    #[serde(default)]
    pub previous_runtime_key: Option<String>,
    #[serde(default)]
    pub planner_provider: Option<String>,
    /// Described in: `rocm setup status`/`reset` output, their `--help` doc
    /// comments, README.md's Setup section, docs/testing.md,
    /// docs/manual-testing.md, and the module doc comment in
    /// `crates/rocm-dash-tui/src/ui/onboarding.rs`. Nothing reads this to
    /// auto-open onboarding — grep this field, not a phrase, before touching
    /// any of those surfaces.
    #[serde(default)]
    pub onboarding_dismissed: bool,
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    #[serde(default)]
    pub permissions: PermissionsConfig,
    #[serde(default)]
    pub setup: SetupConfig,
    #[serde(default)]
    pub tools: BTreeMap<String, ManagedToolConfig>,
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderUserConfig>,
    #[serde(default)]
    pub engines: BTreeMap<String, EngineUserConfig>,
    #[serde(default)]
    pub automations: AutomationsConfig,
    /// rocm-dash telemetry/dashboard knobs. Nested as a sub-config
    /// so it never collides with the rocm-cli `telemetry` analytics policy on
    /// rebase. Every field defaults, so the section is fully optional.
    #[serde(default)]
    pub dashboard: DashboardConfig,
}

// ===== rocm-dash dashboard sub-config =====
//
// Additive nesting under the canonical `RocmCliConfig`. The rocm-cli
// `TelemetryConfig { mode }` is an analytics opt-in *policy*; this
// `DashboardConfig` is the operational *spec* (listen address + tick cadence +
// chat endpoint). They are deliberately separate axes and never share a field.
// Pure `with_*()` transforms are scoped to this sub-config only — rocm-cli's own
// config keeps its in-place `&mut` mutation convention untouched.

fn default_dashboard_socket() -> String {
    // Choose a socket location whose *parent* directory is always user-owned so
    // that run_unix can tighten it to 0o700 without EPERM. See
    // `runtime::user_runtime_dir` for the precedence. This resolver is mirrored
    // in `rocm-dash-core` so the canonical `rocm` config and a standalone
    // `rocm-dash` config resolve to the same place; keep the two in sync.
    let path = dashboard_socket_path(
        std::env::var_os("XDG_RUNTIME_DIR"),
        std::env::var_os("HOME"),
        // An empty `USER` must fall through to `LOGNAME`, not short-circuit it.
        std::env::var("USER")
            .ok()
            .filter(|v| !v.is_empty())
            .or_else(|| std::env::var("LOGNAME").ok().filter(|v| !v.is_empty())),
        std::env::temp_dir(),
    );
    format!("unix:{}", path.display())
}

/// Pure core of [`default_dashboard_socket`]: resolve the socket path from
/// explicit env inputs so the precedence is testable without mutating
/// process-global env vars (unsafe and racy under parallel tests in edition
/// 2024). Precedence:
///
/// 1. `$XDG_RUNTIME_DIR` — already mode `0700` on systemd systems, ideal.
/// 2. `$HOME/.rocm/data/telemetry` — standard per-user data dir.
/// 3. `temp_dir()/rocm-<user>` — user-named subdir so the parent is something we
///    create and own, not `/tmp` itself.
///
/// The tier chain itself lives in [`crate::user_runtime_dir`], which the Lemonade
/// engine also uses to synthesize a runtime directory for its child process.
/// Only tier 2 needs the `telemetry` leaf: tiers 1 and 3 are already per-user
/// runtime directories, so the socket sits directly in them.
fn dashboard_socket_path(
    xdg_runtime_dir: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
    user: Option<String>,
    temp_dir: std::path::PathBuf,
) -> std::path::PathBuf {
    user_runtime_dir(xdg_runtime_dir, home, user, temp_dir, "telemetry", "").join("rocmdashd.sock")
}

fn default_dashboard_listen() -> String {
    default_dashboard_socket()
}

fn default_dashboard_connect() -> String {
    default_dashboard_socket()
}

fn default_dashboard_theme() -> String {
    "default-dark".to_owned()
}

const fn default_gpu_tick_secs() -> f64 {
    1.0
}

const fn default_discovery_tick_secs() -> f64 {
    5.0
}

const fn default_instance_tick_secs() -> f64 {
    2.0
}

/// Telemetry daemon operational spec. Tick cadences are stored as f64 seconds in
/// the unified JSON config; use the `*_tick()` accessors for `Duration`s.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DashboardDaemonConfig {
    /// `unix:/path/to.sock` or `tcp:host:port`.
    #[serde(default = "default_dashboard_listen")]
    pub listen: String,
    /// Optional shared secret. Required for TCP, ignored for Unix sockets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    #[serde(default = "default_gpu_tick_secs")]
    pub gpu_tick_secs: f64,
    #[serde(default = "default_discovery_tick_secs")]
    pub discovery_tick_secs: f64,
    #[serde(default = "default_instance_tick_secs")]
    pub instance_tick_secs: f64,
    /// Watch this file for new normalized benchmark CSV rows. When unset, the
    /// daemon derives `<data_dir>/bench/results.csv` from the current `AppPaths`
    /// at startup so machine-specific paths are never persisted in config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bench_results_dir: Option<PathBuf>,
}

impl Default for DashboardDaemonConfig {
    fn default() -> Self {
        Self {
            listen: default_dashboard_listen(),
            token: None,
            gpu_tick_secs: default_gpu_tick_secs(),
            discovery_tick_secs: default_discovery_tick_secs(),
            instance_tick_secs: default_instance_tick_secs(),
            bench_results_dir: None,
        }
    }
}

impl DashboardDaemonConfig {
    fn secs_to_duration(s: f64, fallback: Duration) -> Duration {
        // try_from_secs_f64 rejects NaN, negative, inf, and values that
        // overflow Duration (extremely large finite f64).
        Duration::try_from_secs_f64(s).unwrap_or(fallback)
    }

    pub fn gpu_tick(&self) -> Duration {
        Self::secs_to_duration(self.gpu_tick_secs, Duration::from_secs(1))
    }

    pub fn discovery_tick(&self) -> Duration {
        Self::secs_to_duration(self.discovery_tick_secs, Duration::from_secs(5))
    }

    pub fn instance_tick(&self) -> Duration {
        Self::secs_to_duration(self.instance_tick_secs, Duration::from_secs(2))
    }
}

fn deserialize_optional_chat_temperature<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<f32>::deserialize(deserializer)?;
    if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
        return Err(serde::de::Error::custom(
            "dashboard.tui.chat_temperature must be a finite value >= 0.0",
        ));
    }
    Ok(value)
}

fn deserialize_optional_chat_top_p<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<f32>::deserialize(deserializer)?;
    if value.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
        return Err(serde::de::Error::custom(
            "dashboard.tui.chat_top_p must be a finite value between 0.0 and 1.0",
        ));
    }
    Ok(value)
}

fn deserialize_optional_chat_max_tokens<'de, D>(deserializer: D) -> Result<Option<u32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<u32>::deserialize(deserializer)?;
    if value == Some(0) {
        return Err(serde::de::Error::custom(
            "dashboard.tui.chat_max_tokens must be greater than 0",
        ));
    }
    Ok(value)
}

/// Dashboard TUI spec. The chat endpoint URL / model / auth-header *name* are
/// plain data; the auth-header *value* (API key) is always env-only and never
/// stored here (AMD gateway invariant).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DashboardTuiConfig {
    #[serde(default = "default_dashboard_connect")]
    pub connect: String,
    #[serde(default = "default_dashboard_theme")]
    pub theme: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_auth_header: Option<String>,
    /// Sampling temperature applied to chat requests (parity with the
    /// `rocm chat` / `rocm serve` `--temperature` flag). `None` leaves the
    /// endpoint default untouched; a CLI flag overrides this.
    #[serde(
        default,
        deserialize_with = "deserialize_optional_chat_temperature",
        skip_serializing_if = "Option::is_none"
    )]
    pub chat_temperature: Option<f32>,
    /// Nucleus-sampling `top_p` applied to chat requests (parity with
    /// `--top-p`). `None` leaves the endpoint default untouched.
    #[serde(
        default,
        deserialize_with = "deserialize_optional_chat_top_p",
        skip_serializing_if = "Option::is_none"
    )]
    pub chat_top_p: Option<f32>,
    /// Upper bound on generated tokens for chat requests (parity with
    /// `--max-tokens`). `None` uses the built-in dashboard default.
    #[serde(
        default,
        deserialize_with = "deserialize_optional_chat_max_tokens",
        skip_serializing_if = "Option::is_none"
    )]
    pub chat_max_tokens: Option<u32>,
}

impl Default for DashboardTuiConfig {
    fn default() -> Self {
        Self {
            connect: default_dashboard_connect(),
            theme: default_dashboard_theme(),
            chat_url: None,
            chat_model: None,
            chat_auth_header: None,
            chat_temperature: None,
            chat_top_p: None,
            chat_max_tokens: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct DashboardConfig {
    #[serde(default)]
    pub daemon: DashboardDaemonConfig,
    #[serde(default)]
    pub tui: DashboardTuiConfig,
}

impl DashboardConfig {
    /// Return a copy with the chat endpoint base URL + model set and the custom
    /// auth header cleared (mirrors the rocm-dash `config_with_chat` behavior).
    /// Immutable transform — scoped to the dashboard sub-config only.
    #[must_use]
    pub fn with_chat_endpoint(
        mut self,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        self.tui.chat_url = Some(base_url.into());
        self.tui.chat_model = Some(model.into());
        self.tui.chat_auth_header = None;
        self
    }

    /// Return a copy with the dashboard theme set.
    #[must_use]
    pub fn with_theme(mut self, theme: impl Into<String>) -> Self {
        self.tui.theme = theme.into();
        self
    }

    /// Return a copy with the telemetry daemon listen address set.
    #[must_use]
    pub fn with_daemon_listen(mut self, listen: impl Into<String>) -> Self {
        self.daemon.listen = listen.into();
        self
    }
}

/// Legacy rocm-dash TOML config shape (`~/.config/rocm-dash/config.toml`),
/// parsed for one-shot migration into the unified JSON config. Every field is
/// optional so partial/legacy files parse cleanly; only the carried-forward
/// fields are mirrored.
#[derive(Debug, Default, Deserialize)]
struct LegacyDashToml {
    #[serde(default)]
    default_engine: Option<String>,
    #[serde(default)]
    daemon: LegacyDashDaemon,
    #[serde(default)]
    tui: LegacyDashTui,
    #[serde(default)]
    engines: BTreeMap<String, EngineUserConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct LegacyDashDaemon {
    #[serde(default)]
    listen: Option<String>,
    #[serde(default)]
    token: Option<String>,
    #[serde(default)]
    gpu_tick: Option<f64>,
    #[serde(default)]
    discovery_tick: Option<f64>,
    #[serde(default)]
    instance_tick: Option<f64>,
    #[serde(default)]
    bench_results_dir: Option<PathBuf>,
}

#[derive(Debug, Default, Deserialize)]
struct LegacyDashTui {
    #[serde(default)]
    connect: Option<String>,
    #[serde(default)]
    theme: Option<String>,
    #[serde(default)]
    chat_url: Option<String>,
    #[serde(default)]
    chat_model: Option<String>,
    #[serde(default)]
    chat_auth_header: Option<String>,
}

impl RocmCliConfig {
    pub fn load(paths: &AppPaths) -> Result<Self> {
        let path = paths.config_path();
        if !path.is_file() {
            return Ok(Self::default());
        }

        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        serde_json::from_slice(&bytes)
            .with_context(|| format!("failed to parse {}", path.display()))
    }

    /// Persist the config by replacing it via the shared `write_atomic`
    /// helper (write a sibling temp file, then rename it over `config.json`;
    /// see that helper's own doc for the write/rename/cleanup guarantees).
    ///
    /// The destination is never removed first, because `load` reads a
    /// missing `config.json` as "no config" and answers with defaults — a
    /// removed-then-not-yet-replaced window would lose every setting without
    /// so much as an error.
    pub fn save(&self, paths: &AppPaths) -> Result<()> {
        let path = paths.config_path();
        fs::create_dir_all(&paths.config_dir)
            .with_context(|| format!("failed to create {}", paths.config_dir.display()))?;
        let bytes =
            serde_json::to_vec_pretty(self).context("failed to serialize rocm-cli config")?;
        crate::write_atomic(&path, &bytes, "json")
            .with_context(|| format!("failed to save rocm-cli config to {}", path.display()))
    }

    pub fn engine_config(&self, engine: &str) -> Option<&EngineUserConfig> {
        self.engines.get(engine)
    }

    pub fn engine_config_mut(&mut self, engine: &str) -> &mut EngineUserConfig {
        self.engines.entry(engine.to_owned()).or_default()
    }

    pub fn provider_config(&self, provider: &str) -> Option<&ProviderUserConfig> {
        self.providers.get(provider)
    }

    pub fn provider_config_mut(&mut self, provider: &str) -> &mut ProviderUserConfig {
        self.providers.entry(provider.to_owned()).or_default()
    }

    pub fn provider_enabled(&self, provider: &str) -> bool {
        provider.eq_ignore_ascii_case("local")
            || self
                .provider_config(provider)
                .is_some_and(|cfg| cfg.enabled)
    }

    pub fn watcher_config(&self, watcher: &str) -> Option<&WatcherUserConfig> {
        self.automations.watchers.get(watcher)
    }

    pub fn watcher_config_mut(&mut self, watcher: &str) -> &mut WatcherUserConfig {
        self.automations
            .watchers
            .entry(watcher.to_owned())
            .or_default()
    }

    pub fn automation_daemon_enabled(&self) -> bool {
        self.automations.daemon_enabled || self.automations.watchers.values().any(|cfg| cfg.enabled)
    }

    pub fn watcher_enabled(&self, watcher: &BuiltinWatcherSpec) -> bool {
        self.watcher_config(watcher.id)
            .is_some_and(|cfg| cfg.enabled)
    }

    pub fn effective_watcher_mode(&self, watcher: &BuiltinWatcherSpec) -> WatcherMode {
        self.watcher_config(watcher.id)
            .and_then(|cfg| cfg.mode)
            .unwrap_or(watcher.default_mode)
    }

    /// Location of the legacy rocm-dash TOML config, honoring `XDG_CONFIG_HOME`
    /// (`~/.config/rocm-dash/config.toml` on Linux).
    fn legacy_dashboard_toml_path() -> Option<PathBuf> {
        directories::BaseDirs::new()
            .map(|dirs| dirs.config_dir().join("rocm-dash").join("config.toml"))
    }

    /// One-shot migration of a legacy rocm-dash `config.toml` into the unified
    /// JSON config. If no `config.json` exists yet **and** a legacy TOML is
    /// present, its knobs are mapped into `dashboard` (and the canonical
    /// `default_engine`/`engines`), `config.json` is written once, and the
    /// migrated legacy path is returned so the caller can print a notice. The
    /// TOML is left untouched. Returns `Ok(None)` when there is nothing to do
    /// (already on the unified config, or no legacy file) — never clobbers an
    /// existing `config.json`.
    pub fn migrate_legacy_dashboard_toml(paths: &AppPaths) -> Result<Option<PathBuf>> {
        let Some(legacy) = Self::legacy_dashboard_toml_path() else {
            return Ok(None);
        };
        Self::migrate_legacy_dashboard_toml_from(paths, &legacy)
    }

    /// Testable core of [`migrate_legacy_dashboard_toml`] with an explicit legacy
    /// path. Same one-shot, non-clobbering semantics.
    pub fn migrate_legacy_dashboard_toml_from(
        paths: &AppPaths,
        legacy: &Path,
    ) -> Result<Option<PathBuf>> {
        if paths.config_path().is_file() || !legacy.is_file() {
            return Ok(None);
        }

        let raw = fs::read_to_string(legacy)
            .with_context(|| format!("failed to read {}", legacy.display()))?;
        let parsed: LegacyDashToml = toml::from_str(&raw)
            .with_context(|| format!("failed to parse legacy config {}", legacy.display()))?;

        let mut config = Self::default();

        // Dashboard-specific knobs map into the new sub-config.
        let d = &parsed.daemon;
        if let Some(v) = &d.listen {
            config.dashboard.daemon.listen = v.clone();
        }
        config.dashboard.daemon.token = d.token.clone();
        if let Some(v) = d.gpu_tick {
            config.dashboard.daemon.gpu_tick_secs = v;
        }
        if let Some(v) = d.discovery_tick {
            config.dashboard.daemon.discovery_tick_secs = v;
        }
        if let Some(v) = d.instance_tick {
            config.dashboard.daemon.instance_tick_secs = v;
        }
        config.dashboard.daemon.bench_results_dir = d.bench_results_dir.clone();

        let t = &parsed.tui;
        if let Some(v) = &t.connect {
            config.dashboard.tui.connect = v.clone();
        }
        if let Some(v) = &t.theme {
            config.dashboard.tui.theme = v.clone();
        }
        config.dashboard.tui.chat_url = t.chat_url.clone();
        config.dashboard.tui.chat_model = t.chat_model.clone();
        config.dashboard.tui.chat_auth_header = t.chat_auth_header.clone();

        // `default_engine` / `engines` map onto the canonical rocm-cli fields
        // (identical shape) — not a second source of truth inside `dashboard`.
        config.default_engine = parsed.default_engine.clone();
        config.engines = parsed.engines.clone();

        config.save(paths)?;
        Ok(Some(legacy.to_path_buf()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_app_paths;

    // The socket-path precedence is mirrored in `rocm-dash-core`; these tests
    // mirror the ones there so a divergence in either crate is caught.

    #[test]
    fn dashboard_socket_path_prefers_xdg_runtime_dir() {
        // Tier 1: $XDG_RUNTIME_DIR is already mode 0700 on systemd systems, so it
        // must win over $HOME and the temp dir.
        let path = dashboard_socket_path(
            Some("/run/user/1000".into()),
            Some("/home/alice".into()),
            Some("alice".to_owned()),
            PathBuf::from("/tmp"),
        );
        assert_eq!(path, PathBuf::from("/run/user/1000/rocmdashd.sock"));
    }

    #[test]
    fn dashboard_socket_path_falls_back_to_home_then_temp() {
        // Tier 2: no XDG → per-user data dir under $HOME.
        let path = dashboard_socket_path(
            None,
            Some("/home/alice".into()),
            Some("alice".to_owned()),
            PathBuf::from("/tmp"),
        );
        assert_eq!(
            path,
            PathBuf::from("/home/alice/.rocm/data/telemetry/rocmdashd.sock")
        );

        // Tier 3: no XDG and no HOME → user-named subdir of the temp dir, never
        // the bare temp dir itself.
        let path =
            dashboard_socket_path(None, None, Some("alice".to_owned()), PathBuf::from("/tmp"));
        assert_eq!(path, PathBuf::from("/tmp/rocm-alice/rocmdashd.sock"));
    }

    #[test]
    fn dashboard_socket_path_sanitizes_user_and_skips_empty_env() {
        // An empty XDG/HOME value is treated as unset (falls through), and a user
        // name with path separators cannot escape the intended subdirectory.
        let path = dashboard_socket_path(
            Some("".into()),
            Some("".into()),
            Some("../../etc".to_owned()),
            PathBuf::from("/tmp"),
        );
        assert_eq!(path, PathBuf::from("/tmp/rocm-______etc/rocmdashd.sock"));

        // No user name at all still yields a valid per-user subdir.
        let path = dashboard_socket_path(None, None, None, PathBuf::from("/tmp"));
        assert_eq!(path, PathBuf::from("/tmp/rocm-user/rocmdashd.sock"));

        // A bare empty user name (as opposed to unset) also falls back to "user".
        let path = dashboard_socket_path(None, None, Some(String::new()), PathBuf::from("/tmp"));
        assert_eq!(path, PathBuf::from("/tmp/rocm-user/rocmdashd.sock"));
    }

    #[test]
    fn config_defaults_to_local_telemetry_policy() {
        let config = RocmCliConfig::default();

        assert_eq!(config.telemetry.mode_label(), TELEMETRY_MODE_LOCAL);
        assert!(config.telemetry.local_inspection_enabled());
        assert!(config.telemetry.known_mode());
    }

    #[test]
    fn config_defaults_to_ask_permissions_and_incomplete_setup() {
        let config = RocmCliConfig::default();

        assert_eq!(config.permissions.mode_label(), PERMISSIONS_MODE_ASK);
        assert!(!config.permissions.full_access_enabled());
        assert!(!config.setup.completed);
        assert!(config.setup.therock_venv.is_none());
        assert!(config.planner_provider.is_none());
        assert!(config.tools.is_empty());
    }

    #[test]
    fn config_persists_setup_permissions_and_managed_tools() -> Result<()> {
        let (root, paths) = temp_app_paths("config-managed-state");
        let mut config = RocmCliConfig::default();
        let venv = paths.data_dir.join("runtimes").join("therock");
        let python = paths
            .data_dir
            .join("tools")
            .join("python")
            .join("python.exe");

        config.permissions.mode = PERMISSIONS_MODE_FULL_ACCESS.to_owned();
        config.planner_provider = Some("local".to_owned());
        config.setup.completed = true;
        config.setup.therock_venv = Some(venv.clone());
        config.tools.insert(
            "python".to_owned(),
            ManagedToolConfig {
                path: Some(python.clone()),
                managed: true,
            },
        );
        config.save(&paths)?;

        let loaded = RocmCliConfig::load(&paths)?;
        fs::remove_dir_all(root).ok();

        assert!(loaded.permissions.full_access_enabled());
        assert_eq!(loaded.planner_provider.as_deref(), Some("local"));
        assert!(loaded.setup.completed);
        assert_eq!(loaded.setup.therock_venv.as_deref(), Some(venv.as_path()));
        let tool = loaded.tools.get("python").expect("python tool should load");
        assert!(tool.managed);
        assert_eq!(tool.path.as_deref(), Some(python.as_path()));
        Ok(())
    }

    #[test]
    fn legacy_config_without_telemetry_uses_default_policy() -> Result<()> {
        let config = serde_json::from_value::<RocmCliConfig>(serde_json::json!({
            "default_engine": "vllm"
        }))?;

        assert_eq!(config.default_engine.as_deref(), Some("vllm"));
        assert_eq!(config.telemetry.mode_label(), TELEMETRY_MODE_LOCAL);
        Ok(())
    }

    #[test]
    fn provider_config_defaults_to_local_only() {
        let mut config = RocmCliConfig::default();

        assert!(config.provider_enabled("local"));
        assert!(!config.provider_enabled("openai"));
        assert!(!config.provider_enabled("anthropic"));

        config.provider_config_mut("openai").enabled = true;
        assert!(config.provider_enabled("openai"));
    }

    // ===== Dashboard sub-config + migration =====

    #[test]
    #[allow(clippy::float_cmp)]
    fn dashboard_config_defaults_and_json_round_trip() {
        let cfg = DashboardConfig::default();
        assert!(
            cfg.daemon.listen.starts_with("unix:") && cfg.daemon.listen.ends_with("rocmdashd.sock"),
            "default listen must be a unix socket path ending with rocmdashd.sock, got: {}",
            cfg.daemon.listen
        );
        assert_eq!(cfg.daemon.gpu_tick_secs, 1.0);
        assert_eq!(cfg.daemon.discovery_tick_secs, 5.0);
        assert_eq!(cfg.daemon.instance_tick_secs, 2.0);
        assert_eq!(cfg.tui.theme, "default-dark");
        assert_eq!(cfg.tui.chat_url, None);
        // daemon and tui defaults must agree so a default client finds the daemon.
        assert_eq!(cfg.daemon.listen, cfg.tui.connect);

        let json = serde_json::to_string(&cfg).unwrap();
        let back: DashboardConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back, cfg);
    }

    #[test]
    fn rocm_cli_config_dashboard_section_is_optional() {
        // A config.json with no `dashboard` key parses to the default sub-config.
        let json = r#"{"default_engine":"vllm"}"#;
        let cfg: RocmCliConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.default_engine.as_deref(), Some("vllm"));
        assert_eq!(cfg.dashboard, DashboardConfig::default());
    }

    #[test]
    fn dashboard_with_transforms_are_immutable_and_scoped() {
        let base = DashboardConfig::default();
        let chat = base
            .clone()
            .with_chat_endpoint("http://127.0.0.1:8000", "llama-3.1-8b");
        // Original is untouched (immutable transform).
        assert_eq!(base.tui.chat_url, None);
        assert_eq!(chat.tui.chat_url.as_deref(), Some("http://127.0.0.1:8000"));
        assert_eq!(chat.tui.chat_model.as_deref(), Some("llama-3.1-8b"));
        assert_eq!(chat.tui.chat_auth_header, None);

        let themed = base.clone().with_theme("nord");
        assert_eq!(base.tui.theme, "default-dark");
        assert_eq!(themed.tui.theme, "nord");

        let relisten = base.clone().with_daemon_listen("tcp:127.0.0.1:9000");
        assert!(
            base.daemon.listen.starts_with("unix:"),
            "default listen must use unix scheme"
        );
        assert_eq!(relisten.daemon.listen, "tcp:127.0.0.1:9000");
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn dashboard_tui_inference_params_round_trip_and_default_absent() {
        // Defaults leave the sampling knobs unset and out of the serialized JSON
        // (skip_serializing_if), so a stock config carries no sampling override.
        let default_json = serde_json::to_value(DashboardTuiConfig::default()).unwrap();
        assert!(default_json.get("chat_temperature").is_none());
        assert!(default_json.get("chat_top_p").is_none());
        assert!(default_json.get("chat_max_tokens").is_none());

        // When set, all three round-trip through JSON unchanged.
        let cfg = DashboardTuiConfig {
            chat_temperature: Some(0.25),
            chat_top_p: Some(0.5),
            chat_max_tokens: Some(512),
            ..Default::default()
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: DashboardTuiConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.chat_temperature, Some(0.25));
        assert_eq!(back.chat_top_p, Some(0.5));
        assert_eq!(back.chat_max_tokens, Some(512));
    }

    #[test]
    fn dashboard_tui_rejects_invalid_inference_params() {
        for (field, value, expected) in [
            ("chat_temperature", "-0.1", "chat_temperature"),
            ("chat_top_p", "1.1", "chat_top_p"),
            ("chat_max_tokens", "0", "chat_max_tokens"),
        ] {
            let json = format!(r#"{{"{field}":{value}}}"#);
            let error = serde_json::from_str::<DashboardTuiConfig>(&json)
                .expect_err("invalid inference parameter must be rejected")
                .to_string();
            assert!(
                error.contains(expected),
                "unexpected error for {field}: {error}"
            );
        }
    }

    #[test]
    fn dashboard_daemon_tick_accessors_map_secs_to_duration() {
        let d = DashboardDaemonConfig {
            gpu_tick_secs: 0.5,
            discovery_tick_secs: 10.0,
            instance_tick_secs: 3.0,
            ..Default::default()
        };
        assert_eq!(d.gpu_tick(), Duration::from_secs_f64(0.5));
        assert_eq!(d.discovery_tick(), Duration::from_secs(10));
        assert_eq!(d.instance_tick(), Duration::from_secs(3));
    }

    #[test]
    fn dashboard_bench_results_path_is_derived_not_persisted() {
        let config = DashboardDaemonConfig::default();
        assert_eq!(config.bench_results_dir, None);

        let json = serde_json::to_value(config).unwrap();
        assert!(
            json.get("bench_results_dir").is_none(),
            "machine-specific derived path must not be serialized"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn migrate_legacy_dashboard_toml_maps_knobs_and_is_one_shot() -> Result<()> {
        let (root, paths) = temp_app_paths("migrate-dash");
        paths.ensure()?;
        let legacy = root.join("legacy-config.toml");
        fs::write(
            &legacy,
            r#"
default_engine = "vllm"

[daemon]
listen = "unix:/tmp/custom.sock"
token = "secret"
gpu_tick = 0.5
discovery_tick = 10
instance_tick = 3

[tui]
connect = "unix:/tmp/custom.sock"
theme = "nord"
chat_url = "http://127.0.0.1:8000"
chat_model = "llama-3.1-8b"

[engines.vllm]
preferred_env_id = "env-1"
last_installed_runtime_id = "therock-release"
"#,
        )?;

        // First migration writes config.json once and reports the legacy path.
        let migrated = RocmCliConfig::migrate_legacy_dashboard_toml_from(&paths, &legacy)?;
        assert_eq!(migrated.as_deref(), Some(legacy.as_path()));
        assert!(paths.config_path().is_file());
        // The legacy TOML is left untouched.
        assert!(legacy.is_file());

        // The written config maps every knob into the dashboard sub-config and
        // the canonical engine fields.
        let loaded = RocmCliConfig::load(&paths)?;
        assert_eq!(loaded.dashboard.daemon.listen, "unix:/tmp/custom.sock");
        assert_eq!(loaded.dashboard.daemon.token.as_deref(), Some("secret"));
        assert_eq!(loaded.dashboard.daemon.gpu_tick_secs, 0.5);
        assert_eq!(loaded.dashboard.daemon.discovery_tick_secs, 10.0);
        assert_eq!(loaded.dashboard.daemon.instance_tick_secs, 3.0);
        assert_eq!(loaded.dashboard.tui.connect, "unix:/tmp/custom.sock");
        assert_eq!(loaded.dashboard.tui.theme, "nord");
        assert_eq!(
            loaded.dashboard.tui.chat_url.as_deref(),
            Some("http://127.0.0.1:8000")
        );
        assert_eq!(
            loaded.dashboard.tui.chat_model.as_deref(),
            Some("llama-3.1-8b")
        );
        assert_eq!(loaded.default_engine.as_deref(), Some("vllm"));
        assert_eq!(
            loaded.engines["vllm"].preferred_env_id.as_deref(),
            Some("env-1")
        );

        // Second call is a no-op (config.json already exists — never clobbers).
        let again = RocmCliConfig::migrate_legacy_dashboard_toml_from(&paths, &legacy)?;
        assert_eq!(again, None);

        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn migrate_legacy_dashboard_toml_without_legacy_is_noop() -> Result<()> {
        let (root, paths) = temp_app_paths("migrate-dash-absent");
        paths.ensure()?;
        let legacy = root.join("does-not-exist.toml");
        let migrated = RocmCliConfig::migrate_legacy_dashboard_toml_from(&paths, &legacy)?;
        assert_eq!(migrated, None);
        assert!(!paths.config_path().is_file());
        let _ = fs::remove_dir_all(&root);
        Ok(())
    }

    /// The `inspect_err` cleanup on the rename, which the happy-path test above
    /// cannot reach: it only ever asserts that a SUCCESSFUL save leaves nothing
    /// behind, so deleting the cleanup would go unnoticed until a user with a
    /// failing disk accumulated one `config.json.tmp-<ts>-<pid>` per attempt.
    ///
    /// A directory at the destination is what makes the rename fail on both
    /// supported platforms without needing permissions a root CI lane ignores —
    /// the same device `write_file_atomically_cleans_up_temp_when_the_rename_fails`
    /// uses in `apps/rocm`.
    #[test]
    fn a_config_save_that_cannot_publish_cleans_up_its_temp_file() -> Result<()> {
        let (root, paths) = temp_app_paths("config-save-rename-fails");
        fs::create_dir_all(&paths.config_dir)?;
        let occupied = paths.config_path();
        fs::create_dir_all(occupied.join("nested"))?;
        fs::write(occupied.join("nested").join("keep"), b"x")?;

        RocmCliConfig::default()
            .save(&paths)
            .expect_err("renaming onto a non-empty directory must fail");

        let leftovers: Vec<String> = fs::read_dir(&paths.config_dir)?
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp-"))
            .collect();
        let _ = fs::remove_dir_all(&root);

        assert!(
            leftovers.is_empty(),
            "a save that could not publish must not strand its temp file: {leftovers:?}"
        );
        Ok(())
    }

    #[test]
    fn saving_the_config_replaces_it_without_leaving_a_temp_file_behind() -> Result<()> {
        let (root, paths) = temp_app_paths("config-save-atomic");
        let mut config = RocmCliConfig {
            default_engine: Some("vllm".to_owned()),
            ..RocmCliConfig::default()
        };
        config.save(&paths)?;
        // Overwriting is the interesting case: the save goes through a temp file
        // and a rename rather than truncating the live config, so an interrupted
        // save can never leave a half-written file that `load` refuses to parse.
        config.default_engine = Some("lemonade".to_owned());
        config.save(&paths)?;

        let loaded = RocmCliConfig::load(&paths)?;
        let leftovers = fs::read_dir(&paths.config_dir)?
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp-"))
            .collect::<Vec<_>>();
        let _ = fs::remove_dir_all(&root);

        assert_eq!(loaded.default_engine.as_deref(), Some("lemonade"));
        assert!(
            leftovers.is_empty(),
            "a completed save must leave no temp file in the config folder: {leftovers:?}"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn saving_the_config_replaces_the_file_rather_than_rewriting_it_in_place() -> Result<()> {
        use std::os::unix::fs::MetadataExt;

        // The inode is the observable difference between a rewrite in place and
        // a replacement, and the reason the doc comment can promise what it
        // promises. A `fs::write` over the live path truncates and refills the
        // SAME file, so a reader holding it open — or one that opens it mid-save
        // — can see an empty or half-written config, which `load` hard-errors
        // on. A rename publishes a DIFFERENT file over the name, so a reader
        // sees either the old config or the new one and never the gap between.
        //
        // The sibling test above only checks that no temp file is left behind,
        // which holds just as well for an implementation with no temp file at
        // all, so it cannot tell the two apart.
        //
        // The `#[cfg(windows)]` function of the same name below asserts the same
        // guarantee through `GetFileInformationByHandle`.
        let (root, paths) = temp_app_paths("config-save-replaces");
        let mut config = RocmCliConfig {
            default_engine: Some("vllm".to_owned()),
            ..RocmCliConfig::default()
        };
        config.save(&paths)?;
        let before = fs::metadata(paths.config_path())?.ino();

        config.default_engine = Some("lemonade".to_owned());
        config.save(&paths)?;
        let after = fs::metadata(paths.config_path())?.ino();

        let loaded = RocmCliConfig::load(&paths)?;
        let _ = fs::remove_dir_all(&root);

        assert_ne!(
            before, after,
            "an overwrite that keeps the same inode rewrote the live config in \
             place, so an interrupted save can leave it truncated"
        );
        assert_eq!(loaded.default_engine.as_deref(), Some("lemonade"));
        Ok(())
    }

    /// The volume serial number paired with the 64-bit file index: this
    /// platform's file identity, the way an inode is on Unix.
    ///
    /// Read through `GetFileInformationByHandle` rather than
    /// `std::os::windows::fs::MetadataExt`, whose `volume_serial_number` and
    /// `file_index` accessors are unstable behind the `windows_by_handle`
    /// feature gate and so cannot be called on the pinned stable toolchain.
    #[cfg(windows)]
    #[allow(unsafe_code)] // Win32 FFI
    fn windows_file_identity(path: &Path) -> Result<(u32, u64)> {
        use anyhow::bail;
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::HANDLE;
        use windows_sys::Win32::Storage::FileSystem::{
            BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
        };

        let file =
            fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        // All-POD `repr(C)` record, fully overwritten by the call below on
        // success and never read on failure.
        let mut info: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
        let handle = file.as_raw_handle() as HANDLE;
        if unsafe { GetFileInformationByHandle(handle, &raw mut info) } == 0 {
            bail!(
                "failed to read the file identity of {}: {}",
                path.display(),
                std::io::Error::last_os_error()
            );
        }
        Ok((
            info.dwVolumeSerialNumber,
            (u64::from(info.nFileIndexHigh) << 32) | u64::from(info.nFileIndexLow),
        ))
    }

    #[cfg(windows)]
    #[test]
    fn saving_the_config_replaces_the_file_rather_than_rewriting_it_in_place() -> Result<()> {
        // The Windows counterpart of the Unix test above. The file index
        // together with the volume serial number is this platform's file
        // identity, the way an inode is on Unix: a `fs::write` over the live
        // path refills the SAME file and keeps the pair, while a
        // publish-by-rename puts a DIFFERENT file behind the name. This keeps a
        // Windows-only regression to a plain `fs::write` from passing every
        // gate.
        //
        // The identity is not optional: `GetFileInformationByHandle` either
        // fills both fields or fails, and a failure is a test error rather than
        // a silently skipped assertion.
        let (root, paths) = temp_app_paths("config-save-replaces");
        let mut config = RocmCliConfig {
            default_engine: Some("vllm".to_owned()),
            ..RocmCliConfig::default()
        };
        config.save(&paths)?;
        let before = windows_file_identity(&paths.config_path())?;

        config.default_engine = Some("lemonade".to_owned());
        config.save(&paths)?;
        let after = windows_file_identity(&paths.config_path())?;

        let loaded = RocmCliConfig::load(&paths)?;
        let _ = fs::remove_dir_all(&root);

        assert_ne!(
            before, after,
            "an overwrite that keeps the same file identity rewrote the live \
             config in place, so an interrupted save can leave it truncated"
        );
        assert_eq!(loaded.default_engine.as_deref(), Some("lemonade"));
        Ok(())
    }
}
