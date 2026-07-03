// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! `rocm remote` — bring up a managed model server on a remote GPU host and make
//! it callable from this machine.
//!
//! `serve` connects over SSH, ensures the CLI + ROCm are present (pushing the CLI
//! if missing), starts a *managed* `rocm serve` on the remote (reusing all of the
//! remote's engine/runtime/device selection), opens a **detached** loopback
//! tunnel that survives this command, records the session locally, and returns
//! immediately printing the OpenAI base URL and a `Pending` status.
//!
//! The `Pending → Healthy` progression comes for free from the remote's own
//! managed-service lifecycle (it loads the model in the background). There is no
//! local reconciler daemon: `status` polls the remote over SSH on demand and
//! checks the local tunnel PID for liveness, surfacing the two lifecycles
//! independently. `attach` rebuilds a dropped tunnel; `stop` tears both down.

mod bootstrap;
mod transport;

use std::fmt::Write as _;
use std::io::Write as _;
use std::thread::sleep;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Subcommand;
use rocm_core::{AppPaths, ManagedServiceRecord, RemoteSessionRecord, load_remote_sessions};

use transport::{SshTransport, Transport};

/// How often `status --watch` refreshes.
const WATCH_INTERVAL: Duration = Duration::from_secs(2);

#[derive(Subcommand, Debug)]
pub(crate) enum RemoteCommand {
    /// Start a managed model server on a remote GPU host and forward it locally.
    #[command(after_help = "EXAMPLES:\n  \
rocm remote serve gpu-box qwen2.5-7b-instruct\n  \
rocm remote serve user@10.0.0.5 qwen2.5-7b-instruct --engine llama.cpp\n  \
rocm remote serve gpu-box qwen2.5-7b-instruct --local-port 8000")]
    Serve {
        /// SSH destination: `user@host` or a `~/.ssh/config` host alias.
        host: String,
        /// Model name, alias, or model file path as seen on the remote host.
        model: String,
        /// SSH port on the remote host (defaults to the ssh/config default).
        #[arg(long)]
        ssh_port: Option<u16>,
        /// Port the model server listens on, on the remote side.
        #[arg(long, default_value_t = rocm_core::DEFAULT_LOCAL_PORT)]
        remote_port: u16,
        /// Local loopback port to forward to (defaults to an auto-picked free port).
        #[arg(long)]
        local_port: Option<u16>,
        /// Engine to use on the remote [e.g. lemonade, llama.cpp, vllm].
        #[arg(long)]
        engine: Option<String>,
        /// Device policy [possible values: gpu_required, gpu_preferred, cpu_only].
        #[arg(long)]
        device: Option<String>,
        /// GPU device to serve on: `auto` or a single index like `1`.
        #[arg(long, value_name = "INDEX|auto")]
        gpu: Option<String>,
    },
    /// Show remote serving sessions and their health.
    ///
    /// With no id, lists every session; with an id (or an unambiguous host),
    /// shows that one. `--watch` keeps the view live. (`rocm remote list` is
    /// reserved for listing configured hosts, distinct from active sessions.)
    Status {
        /// Session id, or a host that resolves to exactly one session.
        session: Option<String>,
        /// Refresh the view continuously until interrupted.
        #[arg(long)]
        watch: bool,
    },
    /// Rebuild a dropped local tunnel for a session without redeploying.
    Attach {
        /// Session id, or a host that resolves to exactly one session.
        session: String,
    },
    /// Stop a session: close the local tunnel and stop the remote server.
    Stop {
        /// Session id, or a host that resolves to exactly one session.
        session: String,
    },
}

pub(crate) fn remote(command: RemoteCommand) -> Result<()> {
    let paths = AppPaths::discover()?;
    match command {
        RemoteCommand::Serve {
            host,
            model,
            ssh_port,
            remote_port,
            local_port,
            engine,
            device,
            gpu,
        } => serve(
            &paths,
            ServeArgs {
                host,
                model,
                ssh_port,
                remote_port,
                local_port,
                engine,
                device,
                gpu,
            },
        ),
        RemoteCommand::Status { session, watch } => status(&paths, session.as_deref(), watch),
        RemoteCommand::Attach { session } => attach(&paths, &session),
        RemoteCommand::Stop { session } => stop(&paths, &session),
    }
}

struct ServeArgs {
    host: String,
    model: String,
    ssh_port: Option<u16>,
    remote_port: u16,
    local_port: Option<u16>,
    engine: Option<String>,
    device: Option<String>,
    gpu: Option<String>,
}

fn serve(paths: &AppPaths, args: ServeArgs) -> Result<()> {
    let transport = SshTransport::new(args.host.clone(), args.ssh_port);

    // 1. Reachability — fail fast with a clear message before doing anything else.
    println!("Connecting to {} over SSH...", args.host);
    transport
        .run("true")
        .with_context(|| format!("cannot reach {} over SSH", args.host))?;

    // 2. Ensure the remote is ready (push the CLI if missing; require ROCm).
    let cli = bootstrap::ensure_ready(&transport)?;
    let rocm = cli.invocation();

    // 3. Start a managed server on the remote. This returns quickly: the remote
    //    daemon loads the model in the background and tracks starting -> ready.
    let serve_cmd = build_remote_serve_command(rocm, &args);
    transport
        .run(&serve_cmd)
        .context("failed to start the managed server on the remote host")?;

    // 4. Discover the server we just started (match on the port we asked it to
    //    bind) so we can record its remote service id and current status.
    let record = discover_started_service(&transport, rocm, args.remote_port)?;

    // 5. Open a detached loopback tunnel that survives this command exiting.
    let local_port = match args.local_port {
        Some(port) => port,
        None => pick_free_local_port()?,
    };
    let tunnel_pid = transport
        .open_detached_forward(local_port, "127.0.0.1", args.remote_port)
        .context("failed to open the SSH port-forward")?;

    // 6. Register the session locally so status/attach/stop can find it later.
    let session_id = session_id_for(&args.host, args.remote_port);
    let session = RemoteSessionRecord::new(
        session_id,
        &args.host,
        args.ssh_port,
        &args.model,
        &record.service_id,
        rocm,
        args.remote_port,
        local_port,
        tunnel_pid,
    );
    session.write(paths)?;

    // 7. Return immediately with a pending status and a ready-to-use endpoint.
    println!("Deploying {} on remote host {}...", args.model, args.host);
    println!("Status: {}", remote_status_label(&record.status));
    println!("Local endpoint: {}", session.base_url);
    println!(
        "Track progress with `rocm remote status {}` (add --watch to keep it live).",
        session.session_id
    );
    Ok(())
}

/// Show one or all sessions; with `watch`, refresh until interrupted.
fn status(paths: &AppPaths, session: Option<&str>, watch: bool) -> Result<()> {
    if !watch {
        print!("{}", render_status_table(paths, session)?);
        return Ok(());
    }
    loop {
        // Clear screen + home cursor, then redraw.
        print!("\x1b[2J\x1b[H");
        print!("{}", render_status_table(paths, session)?);
        println!(
            "\n(watching every {}s — Ctrl-C to exit)",
            WATCH_INTERVAL.as_secs()
        );
        let _ = std::io::stdout().flush();
        sleep(WATCH_INTERVAL);
    }
}

/// Rebuild a dropped tunnel for a session, leaving the remote server untouched.
fn attach(paths: &AppPaths, needle: &str) -> Result<()> {
    let mut session = resolve_session(paths, needle)?;
    if rocm_core::process_is_running(session.tunnel_pid) {
        println!(
            "Tunnel for {} is already up (pid {}).",
            session.session_id, session.tunnel_pid
        );
        return Ok(());
    }
    let transport = SshTransport::new(session.host.clone(), session.ssh_port);
    let tunnel_pid = transport
        .open_detached_forward(session.local_port, "127.0.0.1", session.remote_port)
        .context("failed to reopen the SSH port-forward")?;
    session.tunnel_pid = tunnel_pid;
    session.write(paths)?;
    println!(
        "Reattached {}: {} (tunnel pid {}).",
        session.session_id, session.base_url, tunnel_pid
    );
    Ok(())
}

/// Stop a session: close the local tunnel and stop the remote server.
fn stop(paths: &AppPaths, needle: &str) -> Result<()> {
    let session = resolve_session(paths, needle)?;

    // Close the local tunnel (best effort — it may already be gone).
    if rocm_core::process_is_running(session.tunnel_pid) {
        let _ = rocm_core::terminate_process(session.tunnel_pid);
    }

    // Stop the remote server (best effort — surface but don't abort on failure).
    let transport = SshTransport::new(session.host.clone(), session.ssh_port);
    let stop_cmd = format!(
        "{} services stop {} --yes",
        session.remote_cli, session.remote_service_id
    );
    if let Err(err) = transport.run(&stop_cmd) {
        eprintln!(
            "warning: could not stop the remote server for {}: {err:#}",
            session.session_id
        );
    }

    RemoteSessionRecord::remove(paths, &session.session_id)?;
    println!("Stopped {} and removed the session.", session.session_id);
    Ok(())
}

/// Build the remote `rocm serve … --managed` invocation, quoting user-supplied
/// values so they survive the remote shell.
fn build_remote_serve_command(rocm: &str, args: &ServeArgs) -> String {
    let mut cmd = format!(
        "{rocm} serve {} --managed --host 127.0.0.1 --port {}",
        shell_quote(&args.model),
        args.remote_port,
    );
    if let Some(engine) = args.engine.as_deref() {
        let _ = write!(cmd, " --engine {}", shell_quote(engine));
    }
    if let Some(device) = args.device.as_deref() {
        let _ = write!(cmd, " --device {}", shell_quote(device));
    }
    if let Some(gpu) = args.gpu.as_deref() {
        let _ = write!(cmd, " --gpu {}", shell_quote(gpu));
    }
    cmd
}

/// Read `rocm services list --json` on the remote and pick the record bound to
/// `remote_port` (the one we just started). If several match, the most recently
/// created wins.
fn discover_started_service(
    transport: &dyn Transport,
    rocm: &str,
    remote_port: u16,
) -> Result<ManagedServiceRecord> {
    let listing = transport
        .run(&format!("{rocm} services list --json"))
        .context("failed to list managed services on the remote host")?;
    let records: Vec<ManagedServiceRecord> = serde_json::from_str(listing.trim()).context(
        "could not parse `rocm services list --json` from the remote — the remote CLI may be too \
         old to support --json",
    )?;
    records
        .into_iter()
        .filter(|r| r.port == remote_port)
        .max_by_key(|r| r.created_at_unix_ms)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "the managed server did not appear on the remote (no service bound to port {remote_port})"
            )
        })
}

/// Pick a free loopback port by binding to `:0` and reading the assigned port.
/// (Small TOCTOU window before `ssh` binds it; acceptable for v1.)
fn pick_free_local_port() -> Result<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")
        .context("failed to find a free local port for the tunnel")?;
    let port = listener
        .local_addr()
        .context("failed to read the assigned local port")?
        .port();
    Ok(port)
}

/// Deterministic, filename-safe session id from host + remote port. Re-serving
/// the same host+port intentionally reuses the same id (overwrites the record).
fn session_id_for(host: &str, remote_port: u16) -> String {
    let slug: String = host
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    format!("{}-{remote_port}", slug.trim_matches('-'))
}

/// Resolve a user handle — a session id, or a host that maps to exactly one
/// session — to its record.
fn resolve_session(paths: &AppPaths, needle: &str) -> Result<RemoteSessionRecord> {
    let sessions = load_remote_sessions(paths)?;
    if let Some(exact) = sessions.iter().find(|s| s.session_id == needle) {
        return Ok(exact.clone());
    }
    let by_host: Vec<&RemoteSessionRecord> = sessions.iter().filter(|s| s.host == needle).collect();
    match by_host.as_slice() {
        [] => bail!(
            "no remote session matches '{needle}'. Run `rocm remote status` to list sessions."
        ),
        [only] => Ok((*only).clone()),
        many => {
            let ids = many
                .iter()
                .map(|s| s.session_id.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            bail!("'{needle}' matches multiple sessions ({ids}); specify a session id.")
        }
    }
}

/// Map a remote server status word to a display label.
fn remote_status_label(raw: &str) -> String {
    match raw {
        "ready" | "running" => "Healthy".to_owned(),
        "starting" | "recovering" => "Pending".to_owned(),
        "failed" | "stopped" => "Failed".to_owned(),
        other => format!("Unknown ({other})"),
    }
}

/// Query the remote over SSH for this session's server status.
/// `Unreachable` = SSH failed; `Error` = the CLI ran but errored; `Gone` = the
/// service is no longer present on the remote.
fn query_remote_status(session: &RemoteSessionRecord) -> String {
    let transport = SshTransport::new(session.host.clone(), session.ssh_port);
    let listing = match transport.exec(&format!("{} services list --json", session.remote_cli)) {
        Ok(outcome) if outcome.success => outcome.stdout,
        Ok(_) => return "Error".to_owned(),
        Err(_) => return "Unreachable".to_owned(),
    };
    let records: Vec<ManagedServiceRecord> = match serde_json::from_str(listing.trim()) {
        Ok(records) => records,
        Err(_) => return "Error".to_owned(),
    };
    records
        .iter()
        .find(|r| r.service_id == session.remote_service_id)
        .map_or_else(|| "Gone".to_owned(), |r| remote_status_label(&r.status))
}

/// Render the status table for all sessions, or one when `filter` is given.
/// Wires the live remote-status probe; formatting lives in `format_sessions`.
fn render_status_table(paths: &AppPaths, filter: Option<&str>) -> Result<String> {
    let sessions = match filter {
        Some(needle) => vec![resolve_session(paths, needle)?],
        None => load_remote_sessions(paths)?,
    };
    Ok(format_sessions(
        &sessions,
        filter.is_some(),
        query_remote_status,
    ))
}

/// Pure formatter for the status table. `remote_status` supplies each session's
/// remote label (injected so tests don't hit the network); tunnel liveness is a
/// local PID check. `show_detail` adds the attach/stop footer for single-session
/// views.
fn format_sessions<F: Fn(&RemoteSessionRecord) -> String>(
    sessions: &[RemoteSessionRecord],
    show_detail: bool,
    remote_status: F,
) -> String {
    if sessions.is_empty() {
        return "No remote sessions. Start one with `rocm remote serve <host> <model>`.\n"
            .to_owned();
    }

    let headers = ["Host", "Model", "Local endpoint", "Remote", "Tunnel"];
    let mut rows: Vec<[String; 5]> = Vec::new();
    for session in sessions {
        let tunnel = if rocm_core::process_is_running(session.tunnel_pid) {
            format!("Up (pid {})", session.tunnel_pid)
        } else {
            "Down".to_owned()
        };
        rows.push([
            session.host.clone(),
            session.model.clone(),
            session.base_url.clone(),
            remote_status(session),
            tunnel,
        ]);
    }

    // Column widths = max of header and cells.
    let mut widths = headers.map(str::len);
    for row in &rows {
        for (i, cell) in row.iter().enumerate() {
            widths[i] = widths[i].max(cell.len());
        }
    }
    let mut out = String::new();
    let render_row = |out: &mut String, cells: &[String; 5]| {
        for (i, cell) in cells.iter().enumerate() {
            if i > 0 {
                out.push_str("  ");
            }
            let _ = write!(out, "{cell:<width$}", width = widths[i]);
        }
        out.push('\n');
    };
    render_row(&mut out, &headers.map(str::to_owned));
    for row in &rows {
        render_row(&mut out, row);
    }

    // Detail footer with the actionable handle when viewing a single session.
    if show_detail && let Some(session) = sessions.first() {
        let _ = write!(
            out,
            "\nsession: {id}\n  attach: rocm remote attach {id}\n  stop:   rocm remote stop {id}\n",
            id = session.session_id
        );
    }
    out
}

/// Minimal POSIX single-quote quoting for values interpolated into a remote
/// shell command: wrap in single quotes and escape embedded single quotes.
fn shell_quote(value: &str) -> String {
    if !value.is_empty()
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b'/' | b':'))
    {
        return value.to_owned();
    }
    let escaped = value.replace('\'', "'\\''");
    format!("'{escaped}'")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_args() -> ServeArgs {
        ServeArgs {
            host: "gpu-box".to_owned(),
            model: "qwen2.5-7b-instruct".to_owned(),
            ssh_port: None,
            remote_port: 11435,
            local_port: None,
            engine: None,
            device: None,
            gpu: None,
        }
    }

    #[test]
    fn remote_serve_command_is_managed_and_loopback() {
        let cmd = build_remote_serve_command("rocm", &base_args());
        assert_eq!(
            cmd,
            "rocm serve qwen2.5-7b-instruct --managed --host 127.0.0.1 --port 11435"
        );
    }

    #[test]
    fn remote_serve_command_threads_passthrough_flags() {
        let mut args = base_args();
        args.engine = Some("llama.cpp".to_owned());
        args.device = Some("gpu_required".to_owned());
        args.gpu = Some("auto".to_owned());
        let cmd = build_remote_serve_command("$HOME/.local/bin/rocm", &args);
        assert!(cmd.starts_with("$HOME/.local/bin/rocm serve qwen2.5-7b-instruct --managed"));
        assert!(cmd.contains(" --engine llama.cpp"));
        assert!(cmd.contains(" --device gpu_required"));
        assert!(cmd.contains(" --gpu auto"));
    }

    #[test]
    fn shell_quote_leaves_simple_values_bare() {
        assert_eq!(shell_quote("qwen2.5-7b-instruct"), "qwen2.5-7b-instruct");
        assert_eq!(shell_quote("./models/m.gguf"), "./models/m.gguf");
    }

    #[test]
    fn shell_quote_wraps_dangerous_values() {
        assert_eq!(shell_quote("a b"), "'a b'");
        assert_eq!(shell_quote("x; rm -rf /"), "'x; rm -rf /'");
        assert_eq!(shell_quote("it's"), "'it'\\''s'");
    }

    #[test]
    fn session_id_is_filename_safe_and_deterministic() {
        assert_eq!(session_id_for("gpu-box", 11435), "gpu-box-11435");
        assert_eq!(session_id_for("user@10.0.0.5", 8001), "user-10-0-0-5-8001");
        // Same host+port always yields the same id (record reuse on re-serve).
        assert_eq!(
            session_id_for("gpu-box", 11435),
            session_id_for("gpu-box", 11435)
        );
    }

    #[test]
    fn remote_status_label_maps_lifecycle_words() {
        assert_eq!(remote_status_label("ready"), "Healthy");
        assert_eq!(remote_status_label("starting"), "Pending");
        assert_eq!(remote_status_label("failed"), "Failed");
        assert_eq!(remote_status_label("weird"), "Unknown (weird)");
    }

    // Isolated AppPaths rooted at a temp dir, so registry tests don't touch the
    // real data dir.
    fn temp_paths(tag: &str) -> (std::path::PathBuf, AppPaths) {
        let root =
            std::env::temp_dir().join(format!("rocm-remote-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let paths = AppPaths {
            config_dir: root.join("config"),
            data_dir: root.join("data"),
            cache_dir: root.join("cache"),
        };
        (root, paths)
    }

    fn sample_session(host: &str, remote_port: u16) -> RemoteSessionRecord {
        RemoteSessionRecord::new(
            session_id_for(host, remote_port),
            host,
            None,
            "qwen2.5-7b",
            "lemonade-qwen-abc",
            "rocm",
            remote_port,
            8001,
            999_999_999, // pid that is not running
        )
    }

    #[test]
    fn resolve_session_by_id_and_by_unique_host() {
        let (root, paths) = temp_paths("resolve");
        sample_session("gpu-box", 11435).write(&paths).unwrap();

        // By exact session id.
        let by_id = resolve_session(&paths, "gpu-box-11435").unwrap();
        assert_eq!(by_id.host, "gpu-box");
        // By host (unique).
        let by_host = resolve_session(&paths, "gpu-box").unwrap();
        assert_eq!(by_host.session_id, "gpu-box-11435");
        // Unknown handle errors.
        assert!(resolve_session(&paths, "nope").is_err());

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn resolve_session_ambiguous_host_errors_with_ids() {
        let (root, paths) = temp_paths("ambiguous");
        sample_session("gpu-box", 11435).write(&paths).unwrap();
        sample_session("gpu-box", 11436).write(&paths).unwrap();

        let err = resolve_session(&paths, "gpu-box").unwrap_err().to_string();
        assert!(err.contains("multiple sessions"), "got: {err}");
        assert!(err.contains("gpu-box-11435"));
        assert!(err.contains("gpu-box-11436"));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn status_table_shows_both_lifecycles() {
        // Dead pid -> Tunnel: Down; remote status injected as Healthy. This is
        // the degraded case (remote up, tunnel down) surfaced as two columns.
        let session = sample_session("gpu-box", 11435);
        let table = format_sessions(std::slice::from_ref(&session), true, |_| {
            "Healthy".to_owned()
        });
        assert!(table.contains("Host"));
        assert!(table.contains("Local endpoint"));
        assert!(table.contains("Healthy"));
        assert!(table.contains("Down"));
        assert!(table.contains("http://127.0.0.1:8001/v1"));
        // Single-session view exposes the actionable handle.
        assert!(table.contains("attach: rocm remote attach gpu-box-11435"));
        assert!(table.contains("stop:   rocm remote stop gpu-box-11435"));
    }

    #[test]
    fn status_table_empty_when_no_sessions() {
        let table = format_sessions(&[], false, |_| "Healthy".to_owned());
        assert!(table.contains("No remote sessions"));
    }
}
