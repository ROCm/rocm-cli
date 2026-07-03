//! `rocm remote` — bring up a managed model server on a remote GPU host and make
//! it callable from this machine.
//!
//! Phase 1 (this module) implements the foreground happy path: connect over SSH,
//! ensure the CLI + ROCm are present (pushing the CLI if missing), start a
//! *managed* `rocm serve` on the remote (reusing all of the remote's engine /
//! runtime / device selection), open a loopback port-forward, and print an
//! OpenAI-compatible base URL. The command holds the tunnel open until
//! interrupted; the remote server is managed and persists.
//!
//! Phase 2 will add a local session registry and `rocm remote list/status/stop`
//! that reconcile against the remote's `rocm services list --json`.

mod bootstrap;
mod transport;

use std::fmt::Write as _;

use anyhow::{Context, Result};
use clap::Subcommand;
use rocm_core::ManagedServiceRecord;

use transport::{SshTransport, Transport};

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
        /// Local loopback port to forward to (defaults to the remote port).
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
}

pub(crate) fn remote(command: RemoteCommand) -> Result<()> {
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
        } => serve(ServeArgs {
            host,
            model,
            ssh_port,
            remote_port,
            local_port,
            engine,
            device,
            gpu,
        }),
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

fn serve(args: ServeArgs) -> Result<()> {
    let transport = SshTransport::new(args.host.clone(), args.ssh_port);
    let local_port = args.local_port.unwrap_or(args.remote_port);

    // 1. Reachability — fail fast with a clear message before doing anything else.
    println!("remote serve");
    println!("  host: {}", args.host);
    transport
        .run("true")
        .with_context(|| format!("cannot reach {} over SSH", args.host))?;

    // 2. Ensure the remote is ready (push the CLI if missing; require ROCm).
    let cli = bootstrap::ensure_ready(&transport)?;
    let rocm = cli.invocation();

    // 3. Start a managed server on the remote, bound to remote loopback.
    let serve_cmd = build_remote_serve_command(rocm, &args);
    println!(
        "  starting managed server (remote port {}) ...",
        args.remote_port
    );
    transport
        .run(&serve_cmd)
        .context("failed to start the managed server on the remote host")?;

    // 4. Discover the server we just started via the machine-readable listing,
    //    matching on the port we asked it to bind.
    let record = discover_started_service(&transport, rocm, args.remote_port)?;
    println!("  remote model: {}", record.canonical_model_id);
    println!("  remote status: {}", record.status);

    // 5. Open the loopback port-forward.
    let mut guard = transport
        .forward(local_port, "127.0.0.1", args.remote_port)
        .context("failed to open the SSH port-forward")?;

    // 6. Tell the user how to call it.
    println!();
    println!("Ready. OpenAI-compatible base URL:");
    println!("  http://127.0.0.1:{local_port}/v1");
    println!();
    println!(
        "  try: rocm chat --provider local --model {} --prompt \"hello\"",
        record.canonical_model_id
    );
    println!("  (set OPENAI_BASE_URL=http://127.0.0.1:{local_port}/v1 for other OpenAI clients)");
    println!();
    println!("The remote server is managed and will keep running after you disconnect.");
    println!("Press Ctrl-C to close the local forward.");

    // 7. Hold the tunnel open until interrupted; the guard tears it down on drop.
    guard.wait()?;
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
}
