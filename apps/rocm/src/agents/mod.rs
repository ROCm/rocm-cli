// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm agents` — PROTOTYPE: run third-party coding agents inside an AXIS
//! sandbox, several side by side, without the agent ever holding the real
//! provider key.
//!
//! Each run gets its own instance directory (agent home, temp dir, generated
//! policy, record) under the data root, so two agents never share state and
//! the user's real home is never touched. The agent talks to a loopback
//! credential proxy owned by this process; it sees only a random placeholder,
//! which the proxy swaps for the real key on the way to the one upstream the
//! agent kind is bound to.

mod cred_proxy;
mod local;
mod policy;
mod registry;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};
use clap::Subcommand;
use rocm_core::AppPaths;

use registry::{AgentRecord, AgentStatus};

#[derive(Debug, Subcommand)]
pub(crate) enum AgentsCommand {
    /// Run an agent inside an AXIS sandbox in the current (or given) workspace.
    Run {
        /// Agent kind. Supported: claude-code.
        agent: String,
        /// Workspace the agent may read and write (default: current directory).
        #[arg(long)]
        workspace: Option<PathBuf>,
        /// Use the model served by a ready `rocm serve --managed` server
        /// instead of a cloud provider. Picks the server when exactly one is
        /// ready; see --service to choose.
        #[arg(long, conflicts_with = "service")]
        local: bool,
        /// Like --local, for the named server (an id from `rocm services list`).
        #[arg(long, value_name = "ID")]
        service: Option<String>,
        /// Arguments passed to the agent.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// List agents started by `rocm agents run`.
    List {
        /// Print JSON instead of a table.
        #[arg(long)]
        json: bool,
    },
}

/// What `rocm` needs to know to sandbox one agent kind.
pub(crate) struct AgentKind {
    pub name: &'static str,
    /// Executable looked up on PATH.
    pub binary: &'static str,
    /// Env var holding the real key on the host, and the placeholder inside.
    pub key_env: &'static str,
    /// Env var pointing the agent at the proxy.
    pub base_url_env: &'static str,
    /// Upstream used when the host does not set `base_url_env` itself (e.g.
    /// to an LLM gateway). Either way the proxy forwards only to this one.
    pub default_upstream: &'static str,
    /// Host env var with extra `Name: Value` request headers (newline
    /// separated) that carry gateway credentials. Injected by the proxy on
    /// the host; never set inside the sandbox.
    pub custom_headers_env: Option<&'static str>,
    /// The agent's own settings file (relative to the host `HOME`) whose
    /// `env` object the agent would apply to itself; consulted after the
    /// process environment, since inside the sandbox the agent cannot see it.
    pub settings_env_file: Option<&'static str>,
    /// Non-secret host env vars copied into the sandbox (model names).
    pub passthrough_env: &'static [&'static str],
    /// Extra env set inside the sandbox.
    pub extra_env: &'static [(&'static str, &'static str)],
    /// Extra env set inside the sandbox only when the backend is a model
    /// served by `rocm serve`: features its Anthropic endpoint refuses.
    pub local_env: &'static [(&'static str, &'static str)],
}

const KINDS: &[AgentKind] = &[
    AgentKind {
        name: "claude-code",
        binary: "claude",
        key_env: "ANTHROPIC_API_KEY",
        base_url_env: "ANTHROPIC_BASE_URL",
        default_upstream: "https://api.anthropic.com",
        custom_headers_env: Some("ANTHROPIC_CUSTOM_HEADERS"),
        settings_env_file: Some(".claude/settings.json"),
        passthrough_env: &[
            "ANTHROPIC_MODEL",
            "ANTHROPIC_SMALL_FAST_MODEL",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        ],
        extra_env: &[("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")],
        // rocm's Anthropic-compatible endpoint answers 400 to a request that
        // asks for extended thinking, which Claude Code does by default.
        local_env: &[
            ("CLAUDE_CODE_DISABLE_THINKING", "1"),
            ("CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING", "1"),
            ("MAX_THINKING_TOKENS", "0"),
        ],
    },
    AgentKind {
        // Demo/probe kind: a plain shell with the same sandbox and the same
        // Anthropic-bound credential proxy, to show what an agent can and cannot do.
        name: "shell",
        binary: "sh",
        key_env: "ANTHROPIC_API_KEY",
        base_url_env: "ANTHROPIC_BASE_URL",
        default_upstream: "https://api.anthropic.com",
        custom_headers_env: Some("ANTHROPIC_CUSTOM_HEADERS"),
        settings_env_file: Some(".claude/settings.json"),
        passthrough_env: &[
            "ANTHROPIC_MODEL",
            "ANTHROPIC_SMALL_FAST_MODEL",
            "ANTHROPIC_DEFAULT_OPUS_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        ],
        extra_env: &[],
        local_env: &[],
    },
];

pub(crate) fn run(command: AgentsCommand) -> Result<()> {
    let paths = AppPaths::discover()?;
    let root = paths.data_dir.join("agents");
    match command {
        AgentsCommand::Run {
            agent,
            workspace,
            local,
            service,
            args,
        } => {
            let local = if local || service.is_some() {
                Some(local::resolve(&paths, service.as_deref())?)
            } else {
                None
            };
            run_agent(&root, &agent, workspace, local, &args)
        }
        AgentsCommand::List { json } => list(&root, json),
    }
}

/// Where an agent's model requests end up, and what the sandbox is told.
struct Backend {
    /// The one upstream the credential proxy forwards to.
    upstream: String,
    /// Credential the proxy sets in the header the agent's placeholder came in;
    /// `None` for a local server that needs no key.
    real_key: Option<String>,
    /// Extra credential headers set by the proxy (gateway subscription key).
    inject_headers: Vec<(String, String)>,
    /// Model-name variables set inside the sandbox.
    model_env: Vec<(String, String)>,
    /// What `rocm agents list` shows in its UPSTREAM column.
    label: String,
    /// The line printed at start, saying where the model comes from.
    banner: String,
}

/// A cloud provider or an LLM gateway, configured as the agent itself would be:
/// from the host environment, then the agent's own settings file.
fn cloud_backend(kind: &AgentKind) -> Result<Backend> {
    let host = HostEnv::load(kind)?;
    let real_key = host.get(kind.key_env).with_context(|| {
        format!(
            "{} is not set in the environment{}",
            kind.key_env,
            kind.settings_env_file
                .map(|f| format!(" or in ~/{f}"))
                .unwrap_or_default()
        )
    })?;
    let upstream = host
        .get(kind.base_url_env)
        .unwrap_or_else(|| kind.default_upstream.to_owned())
        .trim_end_matches('/')
        .to_owned();
    let inject_headers = match kind.custom_headers_env {
        Some(var) => cred_proxy::parse_custom_headers(&host.get(var).unwrap_or_default())?,
        None => Vec::new(),
    };
    Ok(Backend {
        label: upstream_origin(&upstream),
        upstream,
        real_key: Some(real_key),
        inject_headers,
        model_env: kind
            .passthrough_env
            .iter()
            .filter_map(|k| host.get(k).map(|v| ((*k).to_owned(), v)))
            .collect(),
        banner: format!(
            "{} inside the sandbox is a placeholder; the real key stays in this process",
            kind.key_env
        ),
    })
}

/// A model served by `rocm serve --managed`. The host environment and the
/// agent's settings file are deliberately not consulted: a cloud key, gateway
/// header or model override there must not reach a local server, and the agent
/// must not be sent to a cloud model by a stray setting.
fn local_backend(kind: &AgentKind, target: &local::LocalTarget) -> Backend {
    Backend {
        upstream: target.anthropic_url.trim_end_matches('/').to_owned(),
        real_key: target.endpoint_key.clone(),
        inject_headers: Vec::new(),
        model_env: kind
            .passthrough_env
            .iter()
            .map(|k| ((*k).to_owned(), target.model.clone()))
            .chain(
                kind.local_env
                    .iter()
                    .map(|(k, v)| ((*k).to_owned(), (*v).to_owned())),
            )
            .collect(),
        label: format!("local:{}", target.service_id),
        banner: format!(
            "model {} is served locally by `{}`; no cloud key is used ({} inside the sandbox is a placeholder)",
            target.model, target.service_id, kind.key_env
        ),
    }
}

fn run_agent(
    root: &Path,
    agent: &str,
    workspace: Option<PathBuf>,
    local: Option<local::LocalTarget>,
    args: &[String],
) -> Result<()> {
    let kind = KINDS.iter().find(|k| k.name == agent).with_context(|| {
        let names: Vec<_> = KINDS.iter().map(|k| k.name).collect();
        format!("unknown agent '{agent}'; supported: {}", names.join(", "))
    })?;
    let axis = which(&std::env::var("ROCM_AXIS_BIN").unwrap_or_else(|_| "axis".into()))
        .context("AXIS not found; install it or set ROCM_AXIS_BIN")?;
    let agent_bin = which(kind.binary)
        .with_context(|| format!("'{}' not found on PATH", kind.binary))?
        .canonicalize()?;
    let backend = match &local {
        Some(target) => local_backend(kind, target),
        None => cloud_backend(kind)?,
    };
    let workspace = workspace
        .map_or_else(std::env::current_dir, Ok)?
        .canonicalize()
        .context("workspace does not exist")?;

    let id = format!("{}-{}", kind.name, &random_hex(4)?);
    let dir = root.join(&id);
    let home = dir.join("home");
    let tmp = dir.join("tmp");
    std::fs::create_dir_all(&home)?;
    std::fs::create_dir_all(&tmp)?;

    let placeholder = format!("rocm-placeholder-{}", random_hex(16)?);
    let record = Arc::new(Mutex::new(AgentRecord {
        id: id.clone(),
        agent: kind.name.into(),
        workspace: workspace.clone(),
        supervisor_pid: std::process::id(),
        started_unix: registry::now_unix(),
        ended_unix: None,
        proxy: String::new(),
        upstream: backend.label.clone(),
        requests: 0,
        rejected: 0,
        status: AgentStatus::Running,
        exit_code: None,
    }));
    let proxy = cred_proxy::start(cred_proxy::ProxyConfig {
        upstream: backend.upstream,
        inject_headers: backend.inject_headers,
        placeholder: placeholder.clone(),
        real_key: backend.real_key,
        audit_path: dir.join("audit.jsonl"),
        record: record.clone(),
        record_path: dir.join("record.json"),
    })?;
    let base_url = format!("http://{proxy}");
    record.lock().unwrap().proxy = base_url.clone();

    seed_agent_state(kind, &home, &placeholder)?;

    let policy_path = dir.join("policy.yaml");
    std::fs::write(
        &policy_path,
        policy::render(&policy::PolicyInputs {
            name: &id,
            agent_bin: &agent_bin,
            home: &home,
            tmp: &tmp,
        }),
    )?;
    registry::save(&dir.join("record.json"), &record.lock().unwrap())?;

    eprintln!("rocm: agent {id} in {}", workspace.display());
    eprintln!("rocm: sandbox policy {}", policy_path.display());
    eprintln!("rocm: {}", backend.banner);

    let mut sandbox_env: Vec<(String, String)> = vec![
        ("HOME".into(), path_str(&home)?),
        ("TMPDIR".into(), path_str(&tmp)?),
        ("CLAUDE_CODE_TMPDIR".into(), path_str(&tmp)?),
        ("PATH".into(), "/usr/local/bin:/usr/bin:/bin".into()),
        (
            "TERM".into(),
            std::env::var("TERM").unwrap_or_else(|_| "xterm-256color".into()),
        ),
        (kind.base_url_env.into(), base_url),
        (kind.key_env.into(), placeholder),
    ];
    sandbox_env.extend(backend.model_env);
    sandbox_env.extend(
        kind.extra_env
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string())),
    );

    let mut axis_cmd = Command::new(&axis);
    // New session, no controlling terminal. AXIS puts the agent in its own
    // process group but never makes it the terminal's foreground group, so an
    // interactive agent that reads the terminal (Claude Code's TUI) would be
    // stopped by SIGTTIN and the run would look hung. Job control applies
    // only to a *controlling* terminal; in a new session the inherited tty is
    // an ordinary device. Cost: terminal resize (SIGWINCH) and Ctrl+C-as-a-
    // signal reach `rocm`, not the agent; a raw-mode TUI still gets Ctrl+C as
    // a byte. The proper fix is a foreground hand-off in AXIS.
    rocm_core::detach_command_session(&mut axis_cmd);
    if let Some(var) = kind.custom_headers_env {
        axis_cmd.env_remove(var);
    }
    let status = axis_cmd
        .args(["run", "--policy"])
        .arg(&policy_path)
        .arg("--")
        .arg("env")
        .arg("-i")
        .args(sandbox_env.iter().map(|(k, v)| format!("{k}={v}")))
        .arg(&agent_bin)
        .args(args)
        .current_dir(&workspace)
        // The real key must never reach the AXIS process tree.
        .env_remove(kind.key_env)
        .status()
        .with_context(|| format!("failed to start {}", axis.display()))?;

    {
        let mut rec = record.lock().unwrap();
        rec.status = AgentStatus::Exited;
        rec.exit_code = status.code();
        rec.ended_unix = Some(registry::now_unix());
        registry::save(&dir.join("record.json"), &rec)?;
    }
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}

fn list(root: &Path, json: bool) -> Result<()> {
    let records = registry::load_all(root)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&records)?);
        return Ok(());
    }
    if records.is_empty() {
        println!("No agents. Start one with `rocm agents run claude-code`.");
        return Ok(());
    }
    println!(
        "{:<24} {:<8} {:>8} {:>8} {:>9}  {:<28} WORKSPACE",
        "AGENT", "STATUS", "UPTIME", "REQUESTS", "REJECTED", "UPSTREAM"
    );
    let now = registry::now_unix();
    for r in &records {
        println!(
            "{:<24} {:<8} {:>8} {:>8} {:>9}  {:<28} {}",
            r.id,
            r.status.label(),
            fmt_duration(r.ended_unix.unwrap_or(now).saturating_sub(r.started_unix)),
            r.requests,
            r.rejected,
            r.upstream,
            r.workspace.display()
        );
    }
    Ok(())
}

/// A fresh instance home would put Claude Code through first-run onboarding,
/// whose connectivity check goes straight to api.anthropic.com (not the
/// gateway) and fails inside the sandbox. Mark onboarding done and approve the
/// per-instance placeholder key (Claude Code records the last 20 characters).
/// The workspace-trust prompt is deliberately not pre-answered.
fn seed_agent_state(kind: &AgentKind, home: &Path, placeholder: &str) -> Result<()> {
    if kind.name != "claude-code" {
        return Ok(());
    }
    let tail: String = placeholder
        .chars()
        .rev()
        .take(20)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let state = serde_json::json!({
        "hasCompletedOnboarding": true,
        "customApiKeyResponses": { "approved": [tail], "rejected": [] },
    });
    std::fs::write(
        home.join(".claude.json"),
        serde_json::to_vec_pretty(&state)?,
    )?;
    Ok(())
}

/// Host-side configuration lookup: the process environment, then the `env`
/// object of the agent's own settings file.
struct HostEnv {
    settings: serde_json::Map<String, serde_json::Value>,
}

impl HostEnv {
    fn load(kind: &AgentKind) -> Result<Self> {
        let settings = match (kind.settings_env_file, std::env::var_os("HOME")) {
            (Some(rel), Some(home)) => {
                let path = Path::new(&home).join(rel);
                match std::fs::read(&path) {
                    Ok(bytes) => Self::env_object(&bytes)
                        .with_context(|| format!("cannot read env from {}", path.display()))?,
                    Err(_) => serde_json::Map::new(),
                }
            }
            _ => serde_json::Map::new(),
        };
        Ok(Self { settings })
    }

    fn env_object(bytes: &[u8]) -> Result<serde_json::Map<String, serde_json::Value>> {
        let doc: serde_json::Value = serde_json::from_slice(bytes)?;
        Ok(doc
            .get("env")
            .and_then(serde_json::Value::as_object)
            .cloned()
            .unwrap_or_default())
    }

    fn get(&self, name: &str) -> Option<String> {
        std::env::var(name)
            .ok()
            .or_else(|| self.settings.get(name)?.as_str().map(str::to_owned))
            .filter(|v| !v.is_empty())
    }
}

/// Scheme and host only, for display: a gateway path can be long and is not
/// what a user needs to recognise the destination.
fn upstream_origin(url: &str) -> String {
    let after_scheme = url.find("://").map_or(0, |i| i + 3);
    match url[after_scheme..].find('/') {
        Some(i) => url[..after_scheme + i].to_owned(),
        None => url.to_owned(),
    }
}

fn fmt_duration(secs: u64) -> String {
    match secs {
        s if s < 60 => format!("{s}s"),
        s if s < 3600 => format!("{}m{:02}s", s / 60, s % 60),
        s => format!("{}h{:02}m", s / 3600, (s % 3600) / 60),
    }
}

fn which(name: &str) -> Result<PathBuf> {
    let candidate = Path::new(name);
    if candidate.components().count() > 1 {
        return Ok(candidate.to_path_buf());
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|p| p.is_file())
        .with_context(|| format!("{name} not on PATH"))
}

fn path_str(path: &Path) -> Result<String> {
    match path.to_str() {
        Some(s) => Ok(s.to_owned()),
        None => bail!("non-UTF-8 path: {}", path.display()),
    }
}

fn random_hex(bytes: usize) -> Result<String> {
    use std::io::Read;
    let mut buf = vec![0u8; bytes];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut buf)?;
    Ok(buf.iter().fold(String::new(), |mut out, b| {
        use std::fmt::Write as _;
        let _ = write!(out, "{b:02x}");
        out
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_env_object_is_read_and_non_strings_ignored() {
        let env =
            HostEnv::env_object(br#"{"env":{"ROCM_TEST_ONLY_IN_SETTINGS":"from-settings","N":1}}"#)
                .unwrap();
        let host = HostEnv { settings: env };
        assert_eq!(
            host.get("ROCM_TEST_ONLY_IN_SETTINGS").as_deref(),
            Some("from-settings")
        );
        assert_eq!(host.get("N"), None);
        assert_eq!(host.get("ROCM_TEST_NOWHERE"), None);
    }

    #[test]
    fn upstream_origin_drops_path() {
        assert_eq!(
            upstream_origin("https://gw.example/Anthropic"),
            "https://gw.example"
        );
        assert_eq!(upstream_origin("http://127.0.0.1:1"), "http://127.0.0.1:1");
    }
}
