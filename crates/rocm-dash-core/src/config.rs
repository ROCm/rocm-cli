// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! TOML config loaded from `~/.config/rocm-dash/config.toml`. Missing = defaults.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tracing::{debug, warn};

/// Default config location: `$XDG_CONFIG_HOME/rocm-dash/config.toml`
/// (or `~/.config/rocm-dash/config.toml` on Linux).
pub fn default_config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|p| p.join("rocm-dash").join("config.toml"))
}

impl Config {
    /// Load config from the given path. Returns `Ok(Config::default())` if the
    /// file does not exist; `Err` only on read/parse failure.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        if !path.exists() {
            debug!(path = %path.display(), "config file not found, using defaults");
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(path).map_err(ConfigError::Io)?;
        let cfg: Self = toml::from_str(&raw).map_err(|e| ConfigError::Parse(e.to_string()))?;
        Ok(cfg)
    }

    /// Load from the default path. Missing file → defaults; bad file → warn + defaults.
    pub fn load_default() -> Self {
        let Some(path) = default_config_path() else {
            return Self::default();
        };
        match Self::load(&path) {
            Ok(c) => c,
            Err(e) => {
                warn!(path = %path.display(), error = %e, "failed to load config; using defaults");
                Self::default()
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse: {0}")]
    Parse(String),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    /// Default serving engine (rocm-cli parity). `None` → platform default.
    /// Plain data only — `serve`/`engines` read this; no behavior lives in core.
    /// Declared first so it serializes as a top-level scalar before the tables.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_engine: Option<String>,
    #[serde(default)]
    pub daemon: DaemonConfig,
    #[serde(default)]
    pub tui: TuiConfig,
    /// Per-engine user preferences (runtime/env ids), keyed by engine name.
    /// Plain data mirrored from rocm-cli's `EngineUserConfig`. Serialized last
    /// (a map of tables). Empty map is omitted.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub engines: BTreeMap<String, EngineConfig>,
}

/// Per-engine user preferences. Plain data only (mirrors rocm-cli's `EngineUserConfig`).
///
/// No I/O or behavior lives in core. Immutable config transforms + persistence
/// live in the `rocm` binary, off the core boundary.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preferred_runtime_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preferred_env_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_installed_runtime_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_installed_env_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// `unix:/path/to.sock` or `tcp:host:port`.
    pub listen: String,
    /// Optional shared secret. Required for TCP, ignored for Unix sockets.
    pub token: Option<String>,
    #[serde(with = "duration_secs")]
    pub gpu_tick: Duration,
    #[serde(with = "duration_secs")]
    pub discovery_tick: Duration,
    #[serde(with = "duration_secs")]
    pub instance_tick: Duration,
    /// Watch this directory for new normalized CSVs.
    pub bench_results_dir: Option<PathBuf>,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            listen: "unix:/tmp/rocmdashd.sock".into(),
            token: None,
            gpu_tick: Duration::from_secs(1),
            discovery_tick: Duration::from_secs(5),
            instance_tick: Duration::from_secs(2),
            bench_results_dir: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuiConfig {
    pub connect: String,
    pub theme: String,
    /// Base URL of an OpenAI-compatible endpoint for the chat surface.
    /// Plain data only — the actual HTTP/async client lives in the TUI crate
    /// (core stays render/async-free; no tokio/ratatui at this boundary).
    #[serde(default)]
    pub chat_url: Option<String>,
    /// Model name to request on the chat endpoint. Plain data only.
    #[serde(default)]
    pub chat_model: Option<String>,
    /// Optional custom auth header NAME for gateways that don't use
    /// `Authorization: Bearer` (e.g. `Ocp-Apim-Subscription-Key` for Azure APIM).
    /// The header NAME is plain data; the secret VALUE is still env-only.
    #[serde(default)]
    pub chat_auth_header: Option<String>,
}

impl Default for TuiConfig {
    fn default() -> Self {
        Self {
            connect: "unix:/tmp/rocmdashd.sock".into(),
            theme: "default-dark".into(),
            chat_url: None,
            chat_model: None,
            chat_auth_header: None,
        }
    }
}

mod duration_secs {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        d.as_secs_f64().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        let secs = f64::deserialize(d)?;
        // try_from_secs_f64 rejects NaN, negative, inf, and overflow.
        Duration::try_from_secs_f64(secs)
            .map_err(|e| serde::de::Error::custom(format!("invalid duration: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn daemon_toml_with_gpu_tick(gpu_tick: &str) -> String {
        format!(
            "[daemon]\nlisten = \"unix:/tmp/test.sock\"\n\
             discovery_tick = 5.0\ninstance_tick = 2.0\ngpu_tick = {gpu_tick}\n"
        )
    }

    #[test]
    fn duration_rejects_negative_gpu_tick() {
        assert!(
            toml::from_str::<Config>(&daemon_toml_with_gpu_tick("-1.0")).is_err(),
            "negative gpu_tick must be rejected"
        );
    }

    #[test]
    fn duration_rejects_negative_discovery_tick() {
        let toml = "[daemon]\nlisten = \"unix:/tmp/test.sock\"\n\
                    gpu_tick = 1.0\ninstance_tick = 2.0\ndiscovery_tick = -0.001\n";
        assert!(
            toml::from_str::<Config>(toml).is_err(),
            "negative discovery_tick must be rejected"
        );
    }

    #[test]
    fn duration_accepts_valid_positive() {
        let c: Config = toml::from_str(&daemon_toml_with_gpu_tick("0.5")).unwrap();
        assert_eq!(c.daemon.gpu_tick, std::time::Duration::from_millis(500));
    }

    #[test]
    fn defaults_serialize_and_round_trip() {
        let c = Config::default();
        let s = toml::to_string(&c).unwrap();
        let back: Config = toml::from_str(&s).unwrap();
        assert_eq!(back.daemon.listen, c.daemon.listen);
        assert_eq!(back.tui.theme, c.tui.theme);
        // Chat fields default to None and survive a round-trip.
        assert_eq!(back.tui.chat_url, None);
        assert_eq!(back.tui.chat_model, None);
    }

    #[test]
    fn chat_fields_round_trip_when_set_and_default_when_absent() {
        // Explicit values survive a TOML round-trip.
        let mut c = Config::default();
        c.tui.chat_url = Some("http://127.0.0.1:8000".into());
        c.tui.chat_model = Some("llama-3.1-8b".into());
        c.tui.chat_auth_header = Some("Ocp-Apim-Subscription-Key".into());
        let s = toml::to_string(&c).unwrap();
        let back: Config = toml::from_str(&s).unwrap();
        assert_eq!(back.tui.chat_url.as_deref(), Some("http://127.0.0.1:8000"));
        assert_eq!(back.tui.chat_model.as_deref(), Some("llama-3.1-8b"));
        assert_eq!(
            back.tui.chat_auth_header.as_deref(),
            Some("Ocp-Apim-Subscription-Key")
        );

        // A [tui] table omitting the chat keys still parses (serde default).
        let partial = "[tui]\nconnect = \"unix:/tmp/x.sock\"\ntheme = \"nord\"\n";
        let parsed: Config = toml::from_str(partial).expect("partial tui parses");
        assert_eq!(parsed.tui.chat_url, None);
        assert_eq!(parsed.tui.chat_model, None);
    }

    #[test]
    fn engine_fields_round_trip_and_default_when_absent() {
        // default_engine + per-engine prefs survive a TOML round-trip.
        let mut c = Config {
            default_engine: Some("vllm".into()),
            ..Default::default()
        };
        c.engines.insert(
            "vllm".into(),
            EngineConfig {
                preferred_env_id: Some("env-1".into()),
                last_installed_runtime_id: Some("therock-release".into()),
                ..Default::default()
            },
        );
        let s = toml::to_string(&c).unwrap();
        let back: Config = toml::from_str(&s).unwrap();
        assert_eq!(back.default_engine.as_deref(), Some("vllm"));
        assert_eq!(
            back.engines["vllm"].preferred_env_id.as_deref(),
            Some("env-1")
        );
        assert_eq!(
            back.engines["vllm"].last_installed_runtime_id.as_deref(),
            Some("therock-release")
        );
        // The chat fields shipped earlier still round-trip alongside the new ones.
        assert_eq!(back.tui.theme, c.tui.theme);

        // A config omitting the engine keys parses to defaults (no engine config).
        let parsed: Config =
            toml::from_str("[tui]\nconnect = \"unix:/tmp/x.sock\"\ntheme = \"nord\"\n")
                .expect("partial config parses");
        assert_eq!(parsed.default_engine, None);
        assert!(parsed.engines.is_empty());
    }

    #[test]
    fn load_missing_file_returns_defaults() {
        let mut p = std::env::temp_dir();
        p.push(format!("rocm-dash-no-such-{}.toml", std::process::id()));
        let c = Config::load(&p).expect("missing file is not an error");
        assert_eq!(c.daemon.listen, Config::default().daemon.listen);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn load_overrides_only_specified_fields() {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "rocm-dash-partial-{}-{}.toml",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(
            &p,
            r#"
[daemon]
listen = "unix:/tmp/custom.sock"
token = "secret"
gpu_tick = 0.5
discovery_tick = 10
instance_tick = 3

[tui]
connect = "unix:/tmp/custom.sock"
theme = "default-dark"
"#,
        )
        .unwrap();
        let c = Config::load(&p).expect("load");
        assert_eq!(c.daemon.listen, "unix:/tmp/custom.sock");
        assert_eq!(c.daemon.token.as_deref(), Some("secret"));
        assert_eq!(c.daemon.gpu_tick.as_secs_f64(), 0.5);
        assert_eq!(c.daemon.discovery_tick.as_secs(), 10);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn load_bad_toml_is_error() {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "rocm-dash-bad-{}-{}.toml",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&p, "this is = not = valid = toml").unwrap();
        assert!(Config::load(&p).is_err());
        let _ = std::fs::remove_file(&p);
    }
}
