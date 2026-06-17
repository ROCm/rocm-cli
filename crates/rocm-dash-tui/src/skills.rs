//! Skills — declarative manifest + plan model for rocm-dash's auto-config /
//! auto-install mechanism.
//!
//! A **skill** is a TOML manifest (the external-team contribution contract):
//! `name`, `description`, optional `requires`, optional `detect` (how to tell if
//! it's already satisfied), and an ordered list of typed `steps`. This module is
//! the pure model: parse + validate + build an ordered plan + the
//! detect→config mapping for `auto-config-endpoint`. It has **no rig/reqwest**
//! dependency (the agent-tool wrappers live in `agent.rs`); execution + I/O live
//! in the `rocm` binary's `skills_cmd`.

use serde::{Deserialize, Serialize};

/// Lemonade's default OpenAI-compatible endpoint + port.
pub const LEMONADE_PORT: u16 = 13305;
pub const LEMONADE_ENDPOINT: &str = "http://localhost:13305/v1";
/// vLLM's conventional OpenAI-compatible endpoint + port.
pub const VLLM_PORT: u16 = 8000;
pub const VLLM_ENDPOINT: &str = "http://localhost:8000/v1";

/// One typed step in a skill manifest. Tagged by `type` in TOML.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Step {
    /// A read-only idempotency/health check command.
    Check { command: String },
    /// Run a command with args (the side-effecting step).
    Run {
        cmd: String,
        #[serde(default)]
        args: Vec<String>,
    },
    /// Download a file to a destination.
    Download { url: String, dest: String },
    /// Write a key into the rocm-dash config.
    WriteConfig { key: String, value: String },
}

impl Step {
    /// One-line, human-readable rendering for a plan.
    pub fn describe(&self) -> String {
        match self {
            Step::Check { command } => format!("check: {command}"),
            Step::Run { cmd, args } => format!("run: {cmd} {}", args.join(" "))
                .trim_end()
                .to_string(),
            Step::Download { url, dest } => format!("download: {url} -> {dest}"),
            Step::WriteConfig { key, value } => format!("write-config: {key} = {value}"),
        }
    }
}

/// How to detect whether a skill is already satisfied.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetectSpec {
    pub kind: DetectKind,
    pub target: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectKind {
    /// `target` is a binary expected on PATH.
    Binary,
    /// `target` is an endpoint expected to be reachable.
    Endpoint,
}

/// A parsed, validated skill manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillManifest {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub requires: Vec<String>,
    #[serde(default)]
    pub detect: Option<DetectSpec>,
    pub steps: Vec<Step>,
}

/// Skill loading/validation errors. String-only at the boundary; never panics.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SkillError {
    #[error("skill manifest parse error: {0}")]
    Parse(String),
    #[error("skill manifest invalid: {0}")]
    Invalid(String),
    #[error("unknown skill: {0}")]
    Unknown(String),
}

/// Parse + validate a TOML skill manifest. Clear errors, never panics.
pub fn parse_manifest(toml_str: &str) -> Result<SkillManifest, SkillError> {
    let manifest: SkillManifest =
        toml::from_str(toml_str).map_err(|e| SkillError::Parse(e.to_string()))?;
    validate(&manifest)?;
    Ok(manifest)
}

fn validate(m: &SkillManifest) -> Result<(), SkillError> {
    if m.name.trim().is_empty() {
        return Err(SkillError::Invalid("name must not be empty".to_string()));
    }
    if m.description.trim().is_empty() {
        return Err(SkillError::Invalid(
            "description must not be empty".to_string(),
        ));
    }
    if m.steps.is_empty() {
        return Err(SkillError::Invalid("steps must not be empty".to_string()));
    }
    Ok(())
}

/// Build an ordered, human-readable plan (the dry-run output). Pure.
pub fn build_plan(m: &SkillManifest) -> Vec<String> {
    m.steps
        .iter()
        .enumerate()
        .map(|(i, s)| format!("{}. {}", i + 1, s.describe()))
        .collect()
}

// ---------------------------------------------------------------------------
// Built-in skills (compiled in). External teams can drop additional `.toml`
// manifests in the user skills dir; discovery merges them in (in the binary).
// ---------------------------------------------------------------------------

/// install-lemonade: install AMD's Lemonade local LLM server from the official
/// **embeddable SDK** release (the `lemond` server + `lemonade` CLI), not pip.
/// The real apply path is special-cased + `--apply`-gated in the binary
/// (`apply_install_lemonade`): it queries the latest GitHub release, picks the
/// per-OS embeddable archive, downloads + extracts it into a rocm-owned dir, and
/// forces port 13305. `--dry-run` prints this plan only. The steps below are the
/// human-readable preview of that flow (the executor resolves the concrete OS/
/// version/URL at apply time). No Python, no pip.
pub const INSTALL_LEMONADE_TOML: &str = r#"
name = "install-lemonade"
description = "Install Lemonade — AMD's local OpenAI-compatible LLM server — from the official embeddable SDK release (lemond server, no pip)."

[detect]
kind = "endpoint"
target = "http://localhost:13305/v1"

[[steps]]
type = "check"
command = "probe http://localhost:13305/v1 (skip if Lemonade already running)"

[[steps]]
type = "run"
cmd = "curl"
args = ["-fsSL", "https://api.github.com/repos/lemonade-sdk/lemonade/releases/latest"]

[[steps]]
type = "run"
cmd = "tar"
args = ["-xf", "lemonade-embeddable-<version>-<os>.{tar.gz|zip}", "-C", "~/.local/share/rocm-dash/lemonade/<version>/"]

[[steps]]
type = "write_config"
key = "default_engine"
value = "lemonade"
"#;

/// auto-config-endpoint: detect a running local LLM endpoint (Lemonade/vLLM) and
/// set `default_engine` for `serve`. Chat keeps the user's configured backend
/// (e.g. the LLM gateway); `tui.chat_url` is filled only when it is unset.
pub const AUTO_CONFIG_ENDPOINT_TOML: &str = r#"
name = "auto-config-endpoint"
description = "Detect a running local LLM endpoint (Lemonade/vLLM) and set default_engine for serve. Chat keeps your configured backend; chat_url is set only when unset."

[detect]
kind = "endpoint"
target = "http://localhost:13305/v1"

[[steps]]
type = "check"
command = "probe http://localhost:13305/v1 (Lemonade) then http://localhost:8000/v1 (vLLM)"

[[steps]]
type = "write_config"
key = "default_engine"
value = "<detected engine>"

[[steps]]
type = "write_config"
key = "tui.chat_url"
value = "<detected endpoint> (ONLY if tui.chat_url is unset; an existing chat backend is preserved)"
"#;

/// The compiled-in skills. Malformed builtins are skipped (never panic) — they
/// are covered by tests so this stays exhaustive in practice.
pub fn builtin_skills() -> Vec<SkillManifest> {
    [INSTALL_LEMONADE_TOML, AUTO_CONFIG_ENDPOINT_TOML]
        .iter()
        .filter_map(|t| parse_manifest(t).ok())
        .collect()
}

/// Look up a built-in skill by name.
pub fn builtin_skill(name: &str) -> Option<SkillManifest> {
    builtin_skills().into_iter().find(|s| s.name == name)
}

// ---------------------------------------------------------------------------
// auto-config-endpoint: pure detect → config mapping.
// ---------------------------------------------------------------------------

/// The config change `auto-config-endpoint` would apply for a detected endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigChange {
    pub chat_url: String,
    pub default_engine: String,
}

/// Map a detected endpoint URL to the exact config change. Pure — no I/O.
/// `None` when nothing was detected. The engine is inferred from the port
/// (Lemonade 13305, vLLM 8000), defaulting to `local`.
pub fn auto_config_change(detected_endpoint: Option<&str>) -> Option<ConfigChange> {
    let endpoint = detected_endpoint?;
    let default_engine = if endpoint.contains(&LEMONADE_PORT.to_string()) {
        "lemonade"
    } else if endpoint.contains(&VLLM_PORT.to_string()) {
        "vllm"
    } else {
        "local"
    };
    Some(ConfigChange {
        chat_url: endpoint.to_string(),
        default_engine: default_engine.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Embeddable Lemonade SDK — pure asset selection.
//
// Lemonade is installed from the official **embeddable** release archives
// (`lemonade-embeddable-<ver>-<os-arch>.{tar.gz|zip}`) published on GitHub, not
// from pip. These functions are PURE (no network/disk): the executor in the
// `rocm` binary does the curl/tar I/O and calls these to decide *what* to fetch.
// ---------------------------------------------------------------------------

/// GitHub repo that publishes the Lemonade embeddable archives.
pub const LEMONADE_GITHUB_REPO: &str = "lemonade-sdk/lemonade";
/// Pinned embeddable version used as the offline fallback when the GitHub
/// releases API is unreachable. Bump deliberately. (Latest at authoring time.)
pub const LEMONADE_EMBEDDABLE_FALLBACK_VERSION: &str = "10.6.0";

/// A selected embeddable archive for a host triple — enough to download, extract,
/// and locate the server binary. Pure data; no I/O.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddableArtifact {
    /// Version without a leading `v`, e.g. `10.6.0`.
    pub version: String,
    /// The archive's `browser_download_url`.
    pub url: String,
    /// The archive file name, e.g. `lemonade-embeddable-10.6.0-ubuntu-x64.tar.gz`.
    pub archive_name: String,
    /// The unpacked server executable name (`lemond` / `lemond.exe`).
    pub server_bin: String,
}

/// Map a host (os, arch) — `std::env::consts::{OS, ARCH}` values — to the
/// embeddable asset's `<os-arch>` token + archive extension. `None` for an
/// unsupported triple (the release ships only these three).
fn embeddable_os_arch(os: &str, arch: &str) -> Option<(&'static str, &'static str)> {
    match (os, arch) {
        ("linux", "x86_64") => Some(("ubuntu-x64", "tar.gz")),
        ("windows", "x86_64") => Some(("windows-x64", "zip")),
        ("macos", "aarch64") => Some(("macos-arm64", "tar.gz")),
        _ => None,
    }
}

/// The unpacked server executable name for an OS.
fn server_bin_for(os: &str) -> &'static str {
    if os == "windows" {
        "lemond.exe"
    } else {
        "lemond"
    }
}

/// Strip a leading `v` from a release tag (`v10.6.0` → `10.6.0`).
fn strip_v(tag: &str) -> &str {
    tag.strip_prefix('v').unwrap_or(tag)
}

/// PURE: build the canonical embeddable artifact for `(os, arch, version)` with no
/// network — the offline-fallback path. `None` for an unsupported triple.
pub fn embeddable_artifact(os: &str, arch: &str, version: &str) -> Option<EmbeddableArtifact> {
    let (os_arch, ext) = embeddable_os_arch(os, arch)?;
    let ver = strip_v(version);
    let archive_name = format!("lemonade-embeddable-{ver}-{os_arch}.{ext}");
    let url = format!(
        "https://github.com/{LEMONADE_GITHUB_REPO}/releases/download/v{ver}/{archive_name}"
    );
    Some(EmbeddableArtifact {
        version: ver.to_string(),
        url,
        archive_name,
        server_bin: server_bin_for(os).to_string(),
    })
}

/// PURE: pick the embeddable asset from a GitHub `releases/latest` JSON body for
/// the host `(os, arch)`. Reads `tag_name` + `assets[].{name, browser_download_url}`
/// and matches the `lemonade-embeddable-*-<os-arch>.<ext>` asset, skipping the
/// `.pkg`/`.rpm`/`.msi` decoys. Returns `None` for an unsupported triple, malformed
/// JSON, or a missing asset/url. Never panics. The release publishes no checksum
/// asset, so integrity rests on TLS + the official-org URL (see recon note).
pub fn pick_embeddable_asset(
    release_json: &str,
    os: &str,
    arch: &str,
) -> Option<EmbeddableArtifact> {
    let (os_arch, ext) = embeddable_os_arch(os, arch)?;
    let value: serde_json::Value = serde_json::from_str(release_json).ok()?;
    let obj = value.as_object()?;
    let tag = obj.get("tag_name")?.as_str()?;
    let version = strip_v(tag).to_string();
    let suffix = format!("-{os_arch}.{ext}");
    let assets = obj.get("assets")?.as_array()?;
    for asset in assets {
        let Some(name) = asset.get("name").and_then(|n| n.as_str()) else {
            continue;
        };
        if name.starts_with("lemonade-embeddable-") && name.ends_with(&suffix) {
            let url = asset
                .get("browser_download_url")
                .and_then(|u| u.as_str())
                .unwrap_or_default();
            if url.is_empty() {
                continue;
            }
            return Some(EmbeddableArtifact {
                version,
                url: url.to_string(),
                archive_name: name.to_string(),
                server_bin: server_bin_for(os).to_string(),
            });
        }
    }
    None
}

/// The config `auto-config-endpoint` would actually apply, after the
/// chat-precedence policy is taken into account. Either field being `None` means
/// "leave it unchanged".
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AutoConfigPlan {
    /// New `tui.chat_url`, or `None` to leave the existing one untouched.
    pub set_chat_url: Option<String>,
    /// New `default_engine`, or `None` when nothing was detected.
    pub set_default_engine: Option<String>,
    /// A local endpoint was detected but `chat_url` was deliberately left alone
    /// because a chat backend is already configured (drives user messaging).
    pub kept_existing_chat: bool,
}

impl AutoConfigPlan {
    /// True when applying this plan would write nothing.
    pub fn is_noop(&self) -> bool {
        self.set_chat_url.is_none() && self.set_default_engine.is_none()
    }
}

/// Decide the config change for `auto-config-endpoint`, honoring chat precedence.
///
/// Policy (see the chat-endpoint design notes): the chat backend defaults to the
/// user's configured endpoint — typically the LLM gateway — and is **never**
/// overwritten by local detection. Running a local engine is a `serve` concern,
/// not a reason to hijack the chatbot. Therefore:
///   - `default_engine` always follows the detected local engine (for `serve`).
///   - `chat_url` is filled **only when no chat backend is configured yet**; an
///     existing `chat_url` is preserved (`kept_existing_chat = true`).
///
/// To point chat at a local engine, the user sets `tui.chat_url` explicitly —
/// detection will not do it for them.
pub fn auto_config_plan(
    detected_endpoint: Option<&str>,
    chat_url_configured: bool,
) -> AutoConfigPlan {
    match auto_config_change(detected_endpoint) {
        None => AutoConfigPlan::default(),
        Some(change) => AutoConfigPlan {
            set_chat_url: (!chat_url_configured).then_some(change.chat_url),
            set_default_engine: Some(change.default_engine),
            kept_existing_chat: chat_url_configured,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_manifest_builds_ordered_plan() {
        let m = parse_manifest(INSTALL_LEMONADE_TOML).expect("valid manifest");
        assert_eq!(m.name, "install-lemonade");
        // Embeddable install needs no Python/pip toolchain.
        assert!(m.requires.is_empty(), "no python3/pip requires");
        // Detection is now endpoint-based (the unpacked server answering on :13305),
        // not a `lemonade` binary on PATH.
        assert!(matches!(
            m.detect,
            Some(DetectSpec {
                kind: DetectKind::Endpoint,
                ..
            })
        ));
        let plan = build_plan(&m);
        assert!(plan[0].starts_with("1. check:"));
        // The plan describes the embeddable flow, never pip.
        assert!(plan.iter().any(|l| l.contains("releases/latest")));
        assert!(plan.iter().any(|l| l.contains("lemonade-embeddable-")));
        // Guard the pivot: ban the deprecated `pip install lemonade-sdk` command.
        // NB: the GitHub *org* is literally `lemonade-sdk/lemonade` (the URL we now
        // download from), so a bare `lemonade-sdk` substring is legitimate — only
        // the `pip install` invocation is forbidden.
        assert!(
            !INSTALL_LEMONADE_TOML.contains("pip install")
                && !INSTALL_LEMONADE_TOML.contains(r#"cmd = "pip""#),
            "install-lemonade TOML must not invoke pip"
        );
    }

    // The captured real `releases/latest` body (Phase 1 fixture).
    const RELEASE_FIXTURE: &str = include_str!("../tests/fixtures/lemonade_release_latest.json");

    #[test]
    fn pick_embeddable_asset_selects_per_triple() {
        let linux = pick_embeddable_asset(RELEASE_FIXTURE, "linux", "x86_64").expect("linux");
        assert_eq!(
            linux.archive_name,
            "lemonade-embeddable-10.6.0-ubuntu-x64.tar.gz"
        );
        assert_eq!(linux.version, "10.6.0");
        assert_eq!(linux.server_bin, "lemond");
        assert!(
            linux
                .url
                .starts_with("https://github.com/lemonade-sdk/lemonade/releases/download/v10.6.0/")
        );

        let win = pick_embeddable_asset(RELEASE_FIXTURE, "windows", "x86_64").expect("windows");
        assert_eq!(
            win.archive_name,
            "lemonade-embeddable-10.6.0-windows-x64.zip"
        );
        assert_eq!(win.server_bin, "lemond.exe");

        let mac = pick_embeddable_asset(RELEASE_FIXTURE, "macos", "aarch64").expect("macos");
        assert_eq!(
            mac.archive_name,
            "lemonade-embeddable-10.6.0-macos-arm64.tar.gz"
        );
        assert_eq!(mac.server_bin, "lemond");
    }

    #[test]
    fn pick_embeddable_asset_none_for_unsupported_triple() {
        // Release ships only ubuntu-x64 / windows-x64 / macos-arm64.
        assert!(pick_embeddable_asset(RELEASE_FIXTURE, "linux", "aarch64").is_none());
        assert!(pick_embeddable_asset(RELEASE_FIXTURE, "freebsd", "x86_64").is_none());
    }

    #[test]
    fn pick_embeddable_asset_malformed_json_is_none_not_panic() {
        for body in ["", "not json", "{}", "[1,2,3]", r#"{"tag_name":"v10.6.0"}"#] {
            assert!(
                pick_embeddable_asset(body, "linux", "x86_64").is_none(),
                "body {body:?} must yield None"
            );
        }
    }

    #[test]
    fn embeddable_artifact_builds_canonical_urls() {
        let a = embeddable_artifact("linux", "x86_64", "10.6.0").expect("linux");
        assert_eq!(
            a.url,
            "https://github.com/lemonade-sdk/lemonade/releases/download/v10.6.0/lemonade-embeddable-10.6.0-ubuntu-x64.tar.gz"
        );
        // Tolerates a leading `v` on the version input.
        let b = embeddable_artifact("windows", "x86_64", "v10.6.0").expect("windows");
        assert_eq!(
            b.url,
            "https://github.com/lemonade-sdk/lemonade/releases/download/v10.6.0/lemonade-embeddable-10.6.0-windows-x64.zip"
        );
        assert_eq!(b.server_bin, "lemond.exe");
        // Unsupported triple → None.
        assert!(embeddable_artifact("linux", "aarch64", "10.6.0").is_none());
        // The fallback const resolves for the common host.
        assert!(
            embeddable_artifact("linux", "x86_64", LEMONADE_EMBEDDABLE_FALLBACK_VERSION).is_some()
        );
    }

    #[test]
    fn malformed_manifest_is_clear_error_not_panic() {
        // Not valid TOML.
        assert!(matches!(
            parse_manifest("name = "),
            Err(SkillError::Parse(_))
        ));
        // Valid TOML but missing required fields → Invalid.
        let err = parse_manifest("name = \"x\"\ndescription = \"y\"\nsteps = []\n").unwrap_err();
        assert_eq!(
            err,
            SkillError::Invalid("steps must not be empty".to_string())
        );
        // Empty name → Invalid.
        assert!(matches!(
            parse_manifest(
                "name = \"\"\ndescription = \"d\"\n[[steps]]\ntype=\"check\"\ncommand=\"x\"\n"
            ),
            Err(SkillError::Invalid(_))
        ));
    }

    #[test]
    fn builtins_include_both_demo_skills() {
        let names: Vec<String> = builtin_skills().into_iter().map(|s| s.name).collect();
        assert!(names.contains(&"install-lemonade".to_string()));
        assert!(names.contains(&"auto-config-endpoint".to_string()));
        assert!(builtin_skill("install-lemonade").is_some());
        assert!(builtin_skill("nope").is_none());
    }

    #[test]
    fn auto_config_change_maps_endpoint_to_config_immutably() {
        // Lemonade port → lemonade engine.
        let c = auto_config_change(Some(LEMONADE_ENDPOINT)).expect("change");
        assert_eq!(c.chat_url, LEMONADE_ENDPOINT);
        assert_eq!(c.default_engine, "lemonade");
        // vLLM port → vllm engine.
        let c = auto_config_change(Some(VLLM_ENDPOINT)).expect("change");
        assert_eq!(c.default_engine, "vllm");
        // Unknown port → local.
        let c = auto_config_change(Some("http://localhost:9999/v1")).expect("change");
        assert_eq!(c.default_engine, "local");
        // Nothing detected → no change.
        assert_eq!(auto_config_change(None), None);
    }

    #[test]
    fn auto_config_plan_never_clobbers_configured_chat() {
        // A chat backend is already configured (e.g. the gateway): default_engine
        // updates for serve, but chat_url is preserved.
        let plan = auto_config_plan(Some(LEMONADE_ENDPOINT), true);
        assert_eq!(
            plan.set_chat_url, None,
            "must not overwrite configured chat"
        );
        assert_eq!(plan.set_default_engine.as_deref(), Some("lemonade"));
        assert!(plan.kept_existing_chat);
        assert!(!plan.is_noop());
    }

    #[test]
    fn auto_config_plan_fills_chat_only_when_unset() {
        // No chat backend configured → fill chat_url with the detected endpoint.
        let plan = auto_config_plan(Some(VLLM_ENDPOINT), false);
        assert_eq!(plan.set_chat_url.as_deref(), Some(VLLM_ENDPOINT));
        assert_eq!(plan.set_default_engine.as_deref(), Some("vllm"));
        assert!(!plan.kept_existing_chat);
    }

    #[test]
    fn auto_config_plan_noop_when_nothing_detected() {
        // Nothing running → write nothing at all, regardless of chat config.
        assert!(auto_config_plan(None, true).is_noop());
        assert!(auto_config_plan(None, false).is_noop());
    }

    #[test]
    fn auto_config_endpoint_manifest_parses() {
        let m = parse_manifest(AUTO_CONFIG_ENDPOINT_TOML).expect("valid");
        assert_eq!(m.name, "auto-config-endpoint");
        assert!(matches!(
            m.detect,
            Some(DetectSpec {
                kind: DetectKind::Endpoint,
                ..
            })
        ));
        // It writes both config keys.
        let plan = build_plan(&m);
        assert!(plan.iter().any(|l| l.contains("tui.chat_url")));
        assert!(plan.iter().any(|l| l.contains("default_engine")));
    }
}
