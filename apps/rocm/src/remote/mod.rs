// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Driving a remote GPU host from this machine.
//!
//! The split this module is built around: **SSH is the control channel, not the
//! data path.** Everything that inspects or changes the remote — probing it,
//! starting a managed `rocm serve`, reading the service registry, tearing a
//! session down — goes over SSH via [`transport`]. The inference traffic does
//! not: the remote publishes its own loopback-bound service onto the tailnet
//! (see [`publish`]), so there is no local tunnel process and no local port to
//! keep alive. An endpoint therefore outlives the command that created it and
//! answers from any of the user's machines, not only this one.
//!
//! Two lifecycles, both on the remote and both able to fail alone: the model
//! server, and the publish pointing at it. `status` reports them as separate
//! facts rather than one health value, because the repair differs — a withdrawn
//! publish is re-declared, a dead server has to be started again.
//!
//! Keeping the control channel narrow is what makes this testable: it is one
//! trait with a scripted stand-in, and [`tailnet`]/[`publish`] parsing is pure,
//! so the flows below are unit tests with no network, no SSH server, and no
//! tailnet.

use std::fmt::Write as _;

use anyhow::{Context, Result, bail};
use clap::Subcommand;
use rocm_core::{AppPaths, ManagedServiceRecord};

use session::RemoteSessionRecord;
use transport::{SshTransport, Transport};

pub(crate) mod bootstrap;
pub(crate) mod doctor;
pub(crate) mod install;
pub(crate) mod provision;
pub(crate) mod publish;
pub(crate) mod session;
pub(crate) mod tailnet;
pub(crate) mod transport;

/// Default loopback port the model server binds on the remote.
const DEFAULT_REMOTE_PORT: u16 = 11434;
/// Default port the remote publishes to the tailnet.
const DEFAULT_TAILNET_PORT: u16 = 8000;
/// Release channel a remote installs from when nothing else is asked for.
const DEFAULT_CHANNEL: &str = "release";

#[derive(Subcommand, Debug)]
pub(crate) enum RemoteCommand {
    /// List the machines on your tailnet that could host a model.
    #[command(after_help = "EXAMPLES:\n  \
rocm remote targets\n  \
rocm remote targets --tag gpu")]
    Targets {
        /// Only show machines carrying this tailnet tag, such as `gpu`.
        #[arg(long, value_name = "TAG")]
        tag: Option<String>,
    },
    /// Serve a model on a remote machine and publish it to your tailnet.
    #[command(after_help = "EXAMPLES:\n  \
rocm remote serve gpu-box qwen2.5-7b-instruct\n  \
rocm remote serve gpu-box qwen2.5-7b-instruct --tailnet-port 8080")]
    Serve {
        /// Machine to serve on: a tailnet name, or an SSH destination on it.
        target: String,
        /// Model name, alias, or a path on the remote machine.
        model: String,
        /// Engine to use on the remote.
        #[arg(long)]
        engine: Option<String>,
        /// GPU to serve on, as the remote sees it.
        #[arg(long, value_name = "INDEX|auto")]
        gpu: Option<String>,
        /// SSH port for the control channel.
        #[arg(long, value_name = "PORT")]
        ssh_port: Option<u16>,
        /// Loopback port the model server binds on the remote.
        #[arg(long, value_name = "PORT", default_value_t = DEFAULT_REMOTE_PORT)]
        remote_port: u16,
        /// Port the remote publishes to your tailnet.
        #[arg(long, value_name = "PORT", default_value_t = DEFAULT_TAILNET_PORT)]
        tailnet_port: u16,
        /// Release channel to install from, if the remote needs the CLI.
        ///
        /// This CLI carries no record of the channel it was built from, so it
        /// cannot match yours automatically. Name it if your machines should
        /// track something other than release.
        #[arg(long, default_value = DEFAULT_CHANNEL)]
        channel: String,
        /// Install ROCm on the machine if it does not have it.
        ///
        /// Off by default. Installing a GPU stack can run for minutes and may
        /// need a reboot, which is a lot to start on a machine you are not
        /// sitting at without saying so. Machines the failure catalog says need
        /// a person are refused even with this set.
        #[arg(long)]
        install_rocm: bool,
    },
    /// Check a remote machine's GPU and ROCm health.
    #[command(after_help = "EXAMPLES:\n  \
rocm remote doctor gpu-box\n  \
rocm remote doctor gpu-box --symptom \"hip error 101\"")]
    Doctor {
        /// Machine to check: a tailnet name, or an SSH destination on it.
        target: String,
        /// Error text you saw, to sharpen the match.
        #[arg(long)]
        symptom: Option<String>,
        /// Show at most this many findings.
        #[arg(long, default_value_t = 5)]
        top: usize,
        /// SSH port for the control channel.
        #[arg(long, value_name = "PORT")]
        ssh_port: Option<u16>,
        /// Release channel to install from, if the remote needs the CLI.
        #[arg(long, default_value = DEFAULT_CHANNEL)]
        channel: String,
    },
    /// Show the remote sessions started from this machine.
    Status {
        /// Session id, or part of a machine name. Omit for all sessions.
        session: Option<String>,
    },
    /// Re-publish a session's endpoint without restarting the model.
    Attach {
        /// Session id, or part of a machine name.
        session: String,
    },
    /// Stop a remote session: withdraw the endpoint and stop the model.
    Stop {
        /// Session id, or part of a machine name.
        session: String,
        /// Forget the session locally even if the machine cannot confirm it
        /// stopped.
        ///
        /// For a machine that is gone for good. Everything the command could
        /// not finish is listed, because the risk of forgetting a session is
        /// that a live endpoint stops being anyone's problem.
        #[arg(long)]
        force: bool,
    },
}

pub(crate) fn run(command: RemoteCommand) -> Result<()> {
    match command {
        RemoteCommand::Targets { tag } => targets(tag.as_deref()),
        RemoteCommand::Serve {
            target,
            model,
            engine,
            gpu,
            ssh_port,
            remote_port,
            tailnet_port,
            channel,
            install_rocm,
        } => serve(&ServeRequest {
            target,
            model,
            engine,
            gpu,
            ssh_port,
            remote_port,
            tailnet_port,
            channel,
            install_rocm,
        }),
        RemoteCommand::Doctor {
            target,
            symptom,
            top,
            ssh_port,
            channel,
        } => remote_doctor(&target, symptom.as_deref(), top, ssh_port, &channel),
        RemoteCommand::Status { session } => status(session.as_deref()),
        RemoteCommand::Attach { session } => attach(&session),
        RemoteCommand::Stop { session, force } => stop(&session, force),
    }
}

/// Show candidate machines, or explain why we cannot see any.
///
/// Discovery never fails the command for a missing or idle Tailscale. A user
/// asking "what can I reach" deserves an answer about their setup, not an error
/// exit — and `rocm remote targets` is precisely the command someone runs while
/// still setting Tailscale up.
fn targets(tag: Option<&str>) -> Result<()> {
    match tailnet::local_status()? {
        tailnet::TailnetAvailability::NotInstalled => {
            println!(
                "Tailscale is not installed on this machine, so there are no targets to list."
            );
            println!();
            println!(
                "`rocm remote` reaches GPU machines over a tailnet. Install Tailscale and run"
            );
            println!("`tailscale up` on this machine and on the GPU machine, then try again.");
        }
        tailnet::TailnetAvailability::NotRunning { backend_state } => {
            println!("Tailscale is installed but not connected (state: {backend_state}).");
            println!();
            println!("Run `tailscale up` on this machine, then try again.");
        }
        tailnet::TailnetAvailability::Running(status) => {
            print!("{}", tailnet::render_targets(&status, tag));
        }
    }
    Ok(())
}

/// Check a machine's health without starting anything on it.
fn remote_doctor(
    target: &str,
    symptom: Option<&str>,
    top: usize,
    ssh_port: Option<u16>,
    channel: &str,
) -> Result<()> {
    // Resolved the same way `serve` resolves it, so a name that serves is a name
    // that can be checked first — which is the order these are meant to be used
    // in.
    resolve_target(target)?;
    let transport = SshTransport::new(target, ssh_port)?;
    let remote_cli = bootstrap::ensure_ready(&transport, target, channel)?;
    let (_, report) = doctor::examine_remote(&transport, &remote_cli, symptom)?;
    print!("{}", doctor::render_report(target, &report, top));
    Ok(())
}

pub(crate) struct ServeRequest {
    pub(crate) target: String,
    pub(crate) model: String,
    pub(crate) engine: Option<String>,
    pub(crate) gpu: Option<String>,
    pub(crate) ssh_port: Option<u16>,
    pub(crate) remote_port: u16,
    pub(crate) tailnet_port: u16,
    pub(crate) channel: String,
    pub(crate) install_rocm: bool,
}

fn serve(request: &ServeRequest) -> Result<()> {
    let paths = AppPaths::discover()?;
    let peer_host = resolve_target(&request.target)?;
    let transport = SshTransport::new(&request.target, request.ssh_port)?;

    println!("Preparing {} ...", request.target);
    let remote_cli = bootstrap::ensure_ready_with(
        &transport,
        &request.target,
        &request.channel,
        request.install_rocm,
    )?;

    // Mint the credential before starting anything. A model that comes up
    // unauthenticated and is then published is exposed for the window between
    // the two, and the whole point of publishing is that the window is visible
    // to every machine on the tailnet.
    let session_id = RemoteSessionRecord::id_for(&peer_host, request.remote_port);
    let api_key = rocm_core::generate_endpoint_api_key();
    session::store_key(&paths, &session_id, &api_key).context(
        "refusing to publish a model endpoint whose API key could not be saved locally: \
         without it you would have no way to call the endpoint you are about to expose",
    )?;

    println!("Starting {} on {} ...", request.model, request.target);
    let start = match transport
        .exec_with_stdin(&remote_serve_command(&remote_cli, request), Some(&api_key))
    {
        Ok(start) => start,
        Err(error) => {
            // The command may already have reached the remote — this fails on a
            // broken pipe while sending the key, or on losing the connection
            // while waiting — so the model's state is genuinely unknown from
            // here. Drop the key that now guards nothing, and say so rather
            // than leaving the user to assume nothing happened.
            session::clear_key(&paths, &session_id);
            return Err(error.context(format!(
                "lost contact with {} while starting the model, so it may or may not be \
                 running.\n\
                 Check with: ssh {} -- {remote_cli} services list",
                request.target, request.target
            )));
        }
    };
    if !start.success {
        session::clear_key(&paths, &session_id);
        bail!(
            "failed to start the model on {}: {}",
            request.target,
            start.stderr.trim()
        );
    }

    // From here the model is running on someone's GPU. Every remaining failure
    // has to leave the machine in a state the user can find and act on, so each
    // one unwinds what has been done rather than returning and forgetting.
    let remote_service_id =
        match discover_started_service(&transport, &remote_cli, request.remote_port) {
            Ok(service_id) => service_id,
            Err(error) => {
                session::clear_key(&paths, &session_id);
                // Nothing can be stopped by name when the name is what could not be
                // read, so the honest move is to say precisely where to look rather
                // than imply it was cleaned up.
                return Err(error.context(format!(
                    "a model may now be running on {} port {} with nothing tracking it.\n\
                 Check with: ssh {} -- {remote_cli} services list",
                    request.target, request.remote_port, request.target
                )));
            }
        };

    println!("Publishing to the tailnet ...");
    if let Err(error) = publish::publish(&transport, request.tailnet_port, request.remote_port) {
        // The model is up but unreachable. Stop it rather than leaving a GPU
        // occupied by something nobody can call and nothing records.
        let leftovers = unwind_partial_serve(
            &transport,
            &paths,
            &session_id,
            &remote_cli,
            Some(&remote_service_id),
            None,
        );
        return Err(describe_leftovers(error, &request.target, &leftovers));
    }

    let base_url = base_url_for(&peer_host, request.tailnet_port);
    let record = RemoteSessionRecord {
        session_id: session_id.clone(),
        target: request.target.clone(),
        peer_host,
        ssh_port: request.ssh_port,
        model: request.model.clone(),
        remote_service_id: remote_service_id.clone(),
        remote_cli: remote_cli.clone(),
        remote_port: request.remote_port,
        tailnet_port: request.tailnet_port,
        base_url,
        created_at_unix_ms: RemoteSessionRecord::now(),
    };
    if let Err(error) = record.write(&paths) {
        // The endpoint is live and published at this point. Without a record
        // nothing on this machine knows it exists, so leaving it up would be
        // exactly the untracked exposure the whole design tries to avoid.
        let leftovers = unwind_partial_serve(
            &transport,
            &paths,
            &session_id,
            &remote_cli,
            Some(&remote_service_id),
            Some((request.tailnet_port, request.remote_port)),
        );
        return Err(describe_leftovers(
            error.context("could not record the session on this machine"),
            &request.target,
            &leftovers,
        ));
    }

    println!();
    println!("{}", render_started(&paths, &record, &api_key));
    Ok(())
}

/// Undo as much of a half-finished `serve` as possible, returning whatever could
/// not be undone.
///
/// Withdraw before stopping, for the same reason teardown does: an endpoint
/// still answering is worse than a process still running. A stopped model behind
/// a live publish refuses connections; a live model behind a forgotten publish is
/// a GPU endpoint on the tailnet that nothing is tracking.
///
/// Every step's failure is collected rather than discarded. The caller needs to
/// tell the user what is still out there — silently swallowing a failed stop is
/// how a machine ends up with a model nobody remembers starting.
fn unwind_partial_serve(
    transport: &dyn Transport,
    paths: &AppPaths,
    session_id: &str,
    remote_cli: &str,
    service_id: Option<&str>,
    published: Option<(u16, u16)>,
) -> Vec<String> {
    let mut leftovers = Vec::new();

    if let Some((tailnet_port, remote_port)) = published
        && let Err(error) = publish::withdraw(transport, tailnet_port, remote_port)
    {
        leftovers.push(format!(
            "the endpoint on port {tailnet_port} may still be published ({error})"
        ));
    }

    if let Some(service_id) = service_id {
        match transport.exec(&format!(
            "{remote_cli} services stop {} --yes",
            shell_quote(service_id)
        )) {
            Ok(outcome) if outcome.success => {}
            Ok(outcome) => leftovers.push(format!(
                "the model ({service_id}) may still be running: {}",
                outcome.stderr.trim()
            )),
            Err(error) => leftovers.push(format!(
                "the model ({service_id}) may still be running: {error}"
            )),
        }
    }

    session::clear_key(paths, session_id);
    leftovers
}

/// Attach what could not be cleaned up to the error that caused it.
fn describe_leftovers(error: anyhow::Error, target: &str, leftovers: &[String]) -> anyhow::Error {
    if leftovers.is_empty() {
        return error;
    }
    error.context(format!(
        "{target} was left with things this command could not undo:\n{}\n\
         Check it with: rocm remote doctor {target}",
        leftovers
            .iter()
            .map(|leftover| format!("  - {leftover}"))
            .collect::<Vec<_>>()
            .join("\n")
    ))
}

/// Resolve a user-supplied target to the tailnet name its endpoint is built on.
fn resolve_target(target: &str) -> Result<String> {
    match tailnet::local_status()? {
        tailnet::TailnetAvailability::NotInstalled => bail!(
            "Tailscale is not installed on this machine.\n\
             `rocm remote serve` publishes the model onto your tailnet, so both machines \
             need it. Install Tailscale and run `tailscale up`, then try again."
        ),
        tailnet::TailnetAvailability::NotRunning { backend_state } => bail!(
            "Tailscale is installed but not connected (state: {backend_state}).\n\
             Run `tailscale up` on this machine, then try again."
        ),
        tailnet::TailnetAvailability::Running(status) => {
            let Some(peer) = tailnet::resolve_peer(&status, target)? else {
                bail!(
                    "`{target}` is not a machine on this tailnet.\n\
                     Run `rocm remote targets` to see what is."
                );
            };
            if !peer.online {
                // Cheaper and far clearer than letting SSH time out.
                bail!(
                    "`{target}` is on this tailnet but currently offline.\n\
                     Start it, or run `rocm remote targets` to pick another machine."
                );
            }
            Ok(peer.endpoint_host().to_owned())
        }
    }
}

/// The remote command that starts the model.
///
/// The API key arrives on stdin rather than in the command line: both the local
/// `ssh` invocation and the remote shell expose their arguments in the process
/// table, so an interpolated key would be readable by any other user on either
/// machine. `--require-api-key` is what makes the loopback bind authenticated
/// anyway, since the publish widens who can reach it.
fn remote_serve_command(remote_cli: &str, request: &ServeRequest) -> String {
    let mut command = format!(
        "IFS= read -r ROCM_SERVE_API_KEY; export ROCM_SERVE_API_KEY; \
         {remote_cli} serve {} --managed --require-api-key --host {} --port {}",
        shell_quote(&request.model),
        publish::LOOPBACK,
        request.remote_port
    );
    if let Some(engine) = &request.engine {
        let _ = write!(command, " --engine {}", shell_quote(engine));
    }
    if let Some(gpu) = &request.gpu {
        let _ = write!(command, " --gpu {}", shell_quote(gpu));
    }
    command
}

/// Find the service the remote just started, by the port we asked it to bind.
fn discover_started_service(
    transport: &dyn Transport,
    remote_cli: &str,
    remote_port: u16,
) -> Result<String> {
    let listing = transport
        .run(&format!("{remote_cli} services list --json --all"))
        .context("could not read the remote's service registry after starting the model")?;
    let records: Vec<ManagedServiceRecord> = serde_json::from_str(&listing).context(
        "could not understand the remote's service registry; the remote CLI may be a \
         different version than this one",
    )?;

    records
        .into_iter()
        .filter(|record| record.port == remote_port)
        // Several records can share a port over a machine's lifetime; the newest
        // is the one we just started.
        .max_by_key(|record| record.created_at_unix_ms)
        .map(|record| record.service_id)
        .with_context(|| {
            format!("the remote started no service on port {remote_port}; nothing to publish")
        })
}

fn base_url_for(peer_host: &str, tailnet_port: u16) -> String {
    // `/v1` to match what the local serve path records, so a URL from either
    // side can be pasted into the same client.
    format!("http://{peer_host}:{tailnet_port}/v1")
}

fn render_started(paths: &AppPaths, record: &RemoteSessionRecord, api_key: &str) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "Model serving on {}", record.target);
    let _ = writeln!(output);
    let _ = writeln!(output, "  endpoint: {}", record.base_url);
    let _ = writeln!(output, "  api key:  {api_key}");
    let _ = writeln!(output, "  session:  {}", record.session_id);
    // Say where the key was kept. It is shown once here, and without this the
    // only copy the user has is whatever their terminal still holds.
    let _ = writeln!(
        output,
        "  key file: {}",
        session::key_path(paths, &record.session_id).display()
    );
    let _ = writeln!(output);
    // Say who can reach this. The loopback-only mental model from local serving
    // does not carry over, and a user who assumes it does will not think to ask
    // whether their tailnet ACLs are right.
    let _ = writeln!(
        output,
        "This endpoint is reachable by every machine on your tailnet that your tailnet's"
    );
    let _ = writeln!(
        output,
        "access rules allow. The API key above is what stops anyone else calling it."
    );
    let _ = writeln!(output);
    let _ = writeln!(output, "  check:  rocm remote status {}", record.session_id);
    let _ = writeln!(output, "  stop:   rocm remote stop {}", record.session_id);
    output
}

/// How a session's model server is doing, as the remote reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ServerHealth {
    Healthy,
    Pending,
    Failed,
    /// The remote could not be reached at all.
    Unreachable,
    /// The remote answered, but not in a way we could read.
    Error,
    /// The remote reported a lifecycle state this CLI does not know.
    ///
    /// Distinct from `Failed`, which is a claim about the model. A word we do
    /// not recognise usually means the remote runs a different version, and
    /// calling that "failed" sends the user to restart something that may be
    /// working perfectly.
    Unrecognised {
        raw: String,
    },
    /// The remote has no record of this service any more.
    Gone,
}

impl ServerHealth {
    const fn label(&self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Pending => "starting",
            Self::Failed => "failed",
            Self::Unreachable => "unreachable",
            Self::Error => "error",
            Self::Gone => "gone",
            Self::Unrecognised { .. } => "unrecognised",
        }
    }

    /// The label plus, where it helps, why we cannot say more.
    fn describe(&self) -> String {
        match self {
            Self::Unrecognised { raw } => {
                format!("unrecognised state `{raw}` (the machine may run a different CLI version)")
            }
            other => other.label().to_owned(),
        }
    }
}

/// Map the remote registry's own lifecycle words onto the states we report.
fn health_from_status(raw: &str) -> ServerHealth {
    match raw {
        "ready" | "running" => ServerHealth::Healthy,
        "starting" | "recovering" => ServerHealth::Pending,
        "failed" | "stopped" => ServerHealth::Failed,
        // Not folded into `Failed`. An unknown word almost always means version
        // skew, and reporting it as a failure sends the user to restart a model
        // that may be serving fine.
        other => ServerHealth::Unrecognised {
            raw: other.to_owned(),
        },
    }
}

/// Both halves of one session, as observed right now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SessionObservation {
    pub(crate) server: ServerHealth,
    /// `None` when the remote could not be asked.
    pub(crate) publish: Option<publish::PublishState>,
}

/// Probe one session over the control channel.
fn observe(transport: &dyn Transport, record: &RemoteSessionRecord) -> SessionObservation {
    let listing = match transport.exec(&format!("{} services list --json --all", record.remote_cli))
    {
        Ok(outcome) if outcome.success => outcome.stdout,
        // Reached the machine but the command failed, versus could not reach it
        // at all. Different problems, different fixes, so different words.
        Ok(_) => {
            return SessionObservation {
                server: ServerHealth::Error,
                publish: None,
            };
        }
        Err(_) => {
            return SessionObservation {
                server: ServerHealth::Unreachable,
                publish: None,
            };
        }
    };

    let server = match serde_json::from_str::<Vec<ManagedServiceRecord>>(&listing) {
        Ok(records) => records
            .iter()
            .find(|candidate| candidate.service_id == record.remote_service_id)
            .map_or(ServerHealth::Gone, |found| {
                health_from_status(&found.status)
            }),
        Err(_) => ServerHealth::Error,
    };

    SessionObservation {
        server,
        publish: publish::publish_state(transport, record.tailnet_port, record.remote_port).ok(),
    }
}

fn status(session: Option<&str>) -> Result<()> {
    let paths = AppPaths::discover()?;
    let sessions = match session {
        Some(needle) => vec![session::resolve(&paths, needle)?],
        None => session::load_all(&paths)?,
    };

    if sessions.is_empty() {
        println!("No remote sessions have been started from this machine.");
        println!();
        println!("Start one with `rocm remote serve <machine> <model>`.");
        return Ok(());
    }

    let observations = sessions
        .iter()
        .map(|record| {
            let observed = SshTransport::new(&record.target, record.ssh_port).map_or(
                // A record naming a machine ssh cannot address is not a reason
                // to hide every other session from the listing.
                SessionObservation {
                    server: ServerHealth::Error,
                    publish: None,
                },
                |transport| observe(&transport, record),
            );
            (record.clone(), observed)
        })
        .collect::<Vec<_>>();
    print!("{}", render_status(&paths, &observations));
    Ok(())
}

/// Render sessions with their two lifecycles kept apart.
fn render_status(
    paths: &AppPaths,
    observations: &[(RemoteSessionRecord, SessionObservation)],
) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "Remote Sessions");
    let _ = writeln!(output);

    for (record, observed) in observations {
        let _ = writeln!(output, "- {}", record.session_id);
        let _ = writeln!(output, "  machine:  {}", record.target);
        let _ = writeln!(output, "  model:    {}", record.model);
        let _ = writeln!(output, "  endpoint: {}", record.base_url);
        // The path, not the credential: a listing lands in scrollback and CI
        // logs, and the file beside it is owner-only for a reason.
        let _ = writeln!(
            output,
            "  key file: {}",
            session::key_path(paths, &record.session_id).display()
        );
        // Two facts, never collapsed into one: which of them is wrong decides
        // whether the fix is `attach` or starting the model again.
        let _ = writeln!(output, "  model server: {}", observed.server.describe());
        let _ = writeln!(
            output,
            "  endpoint published: {}",
            match &observed.publish {
                Some(publish::PublishState::Published) => "yes".to_owned(),
                Some(publish::PublishState::Absent) => "no".to_owned(),
                Some(publish::PublishState::Foreign { forwards_to }) =>
                    format!("no — that port now forwards to {forwards_to}"),
                // Both mean "could not tell", and neither may be read as "no":
                // an endpoint that is still up must never render as one that is
                // down, or the user stops looking for it.
                Some(publish::PublishState::Unreadable) =>
                    "unknown — the machine's reply could not be read".to_owned(),
                None => "unknown — the machine could not be asked".to_owned(),
            }
        );

        if let Some(hint) = repair_hint(record, observed) {
            let _ = writeln!(output, "  {hint}");
        }
    }

    // A publish outlives the machine's reboots, so a stale record is not merely
    // untidy — it may be an endpoint still answering with nothing tracking it.
    if observations
        .iter()
        .any(|(_, observed)| matches!(observed.server, ServerHealth::Gone))
    {
        let _ = writeln!(output);
        let _ = writeln!(
            output,
            "A session whose model server is gone may still be publishing its port."
        );
        let _ = writeln!(output, "Run `rocm remote stop <session>` to clear it.");
    }
    output
}

/// The one command that fixes what is wrong, when exactly one thing is.
fn repair_hint(record: &RemoteSessionRecord, observed: &SessionObservation) -> Option<String> {
    match (&observed.server, &observed.publish) {
        (ServerHealth::Healthy, Some(publish::PublishState::Absent)) => Some(format!(
            "fix: rocm remote attach {} (the model is fine; only the endpoint is missing)",
            record.session_id
        )),
        (ServerHealth::Failed | ServerHealth::Gone, _) => Some(format!(
            "fix: rocm remote stop {} then serve again",
            record.session_id
        )),
        _ => None,
    }
}

fn attach(needle: &str) -> Result<()> {
    let paths = AppPaths::discover()?;
    let record = session::resolve(&paths, needle)?;
    let transport = SshTransport::new(&record.target, record.ssh_port)?;

    // Re-declaring a publish is cheap, but doing it over a dead model server
    // would produce an endpoint that answers with connection refused — worse
    // than one that is honestly absent.
    let observed = observe(&transport, &record);
    match observed.server {
        ServerHealth::Healthy | ServerHealth::Pending => {}
        other => bail!(
            "the model server for {} is {} on {}, so re-publishing would give you an \
             endpoint with nothing behind it.\n\
             Run `rocm remote stop {}` and serve again.",
            record.session_id,
            other.label(),
            record.target,
            record.session_id
        ),
    }

    publish::publish(&transport, record.tailnet_port, record.remote_port)?;
    println!("Endpoint re-published: {}", record.base_url);
    println!("The model was not restarted.");
    Ok(())
}

fn stop(needle: &str, force: bool) -> Result<()> {
    let paths = AppPaths::discover()?;
    let record = session::resolve(&paths, needle)?;
    let transport = SshTransport::new(&record.target, record.ssh_port)?;

    // Withdraw before stopping the model. If only one of the two can be done,
    // the endpoint being gone is the one that matters: a stopped model behind a
    // live publish is a refused connection, but a live model behind a forgotten
    // publish is an open GPU endpoint nobody is tracking.
    let withdrawn = publish::withdraw(&transport, record.tailnet_port, record.remote_port);
    let stopped = transport.exec(&format!(
        "{} services stop {} --yes",
        record.remote_cli,
        shell_quote(&record.remote_service_id)
    ));
    let model_stopped = matches!(&stopped, Ok(outcome) if outcome.success);

    if !force {
        if let Err(error) = withdrawn {
            // Keep the record. Deleting it here would leave a published endpoint
            // with nothing on this machine that can find it again.
            bail!(
                "could not confirm the endpoint for {} was withdrawn: {error}\n\
                 The session is still listed so you can retry with `rocm remote stop {}`.\n\
                 To clear it by hand: ssh {} -- tailscale serve --tcp={} off\n\
                 To forget it locally anyway: rocm remote stop {} --force",
                record.session_id,
                record.session_id,
                record.target,
                record.tailnet_port,
                record.session_id
            );
        }

        if !model_stopped {
            // Same reasoning one step further in. The record is the only thing on
            // this machine that knows the model's id and where it runs; dropping
            // it while the model is still up leaves a GPU occupied by something
            // the user can no longer name.
            bail!(
                "the endpoint for {} was withdrawn, but its model could not be stopped.\n\
                 The session is still listed so you can retry with `rocm remote stop {}`.\n\
                 To check the machine: ssh {} -- {} services list\n\
                 To forget it locally anyway: rocm remote stop {} --force",
                record.session_id,
                record.session_id,
                record.target,
                record.remote_cli,
                record.session_id
            );
        }
    }

    session::clear_key(&paths, &record.session_id);
    record.remove(&paths);

    print!(
        "{}",
        render_stopped(&record, withdrawn.is_ok(), model_stopped, force)
    );
    Ok(())
}

/// Report a teardown, naming anything it could not finish.
///
/// `--force` exists for a machine that is gone for good, and its whole risk is
/// that the user stops thinking about a session that may still be live. So a
/// forced stop is louder than a clean one, not quieter: it lists exactly what
/// may remain and the commands to deal with it once the machine is reachable.
fn render_stopped(
    record: &RemoteSessionRecord,
    withdrawn: bool,
    model_stopped: bool,
    forced: bool,
) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "Stopped {}.", record.session_id);
    let _ = writeln!(
        output,
        "  endpoint withdrawn: {}",
        if withdrawn { "yes" } else { "NOT CONFIRMED" }
    );
    let _ = writeln!(
        output,
        "  model server stopped: {}",
        if model_stopped {
            "yes"
        } else {
            "NOT CONFIRMED"
        }
    );

    if forced && !(withdrawn && model_stopped) {
        let _ = writeln!(output);
        let _ = writeln!(
            output,
            "The session was forgotten on this machine, but {} may still be running it.",
            record.target
        );
        if !withdrawn {
            let _ = writeln!(
                output,
                "  endpoint still reachable on the tailnet — clear it with:\n    \
                 ssh {} -- tailscale serve --tcp={} off",
                record.target, record.tailnet_port
            );
        }
        if !model_stopped {
            let _ = writeln!(
                output,
                "  model may still hold the GPU — check with:\n    \
                 ssh {} -- {} services list",
                record.target, record.remote_cli
            );
        }
    }
    output
}

/// Quote a value for a POSIX remote shell.
///
/// Model names and engine flags are user-supplied values being placed into a
/// command line that a shell on another machine will interpret. Anything not
/// obviously inert gets single-quoted, with embedded single quotes closed and
/// re-opened, so no input can end the quoting and start a new command.
fn shell_quote(value: &str) -> String {
    let inert = |character: char| {
        character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.' | '/' | ':' | '=')
    };
    if !value.is_empty() && value.chars().all(inert) {
        return value.to_owned();
    }
    format!("'{}'", value.replace('\'', r"'\''"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use transport::{ScriptedStep, ScriptedTransport, TransportCall};

    fn request() -> ServeRequest {
        ServeRequest {
            target: "gpu-box".to_owned(),
            model: "qwen2.5-7b-instruct".to_owned(),
            engine: None,
            gpu: None,
            ssh_port: None,
            remote_port: 11434,
            tailnet_port: 8000,
            channel: DEFAULT_CHANNEL.to_owned(),
            install_rocm: false,
        }
    }

    /// A serve-status document showing our own forward, for unwind tests.
    const PUBLISHED_FIXTURE: &str = r#"{"TCP": {"8000": {"TCPForward": "127.0.0.1:11434"}}}"#;

    /// A rendering-only paths root: `render_status` needs one to name each
    /// session's key file, and nothing here touches disk.
    fn render_paths() -> AppPaths {
        AppPaths {
            config_dir: std::path::PathBuf::from("/tmp/rocm-render/config"),
            data_dir: std::path::PathBuf::from("/tmp/rocm-render/data"),
            cache_dir: std::path::PathBuf::from("/tmp/rocm-render/cache"),
        }
    }

    /// An isolated config/data root, so an unwind test never clears a real
    /// endpoint key.
    fn temp_paths(tag: &str) -> (std::path::PathBuf, AppPaths) {
        let root = std::env::temp_dir().join(format!(
            "rocm-remote-unwind-{tag}-{}-{}",
            std::process::id(),
            rocm_core::unix_time_millis()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        (
            root.clone(),
            AppPaths {
                config_dir: root.join("config"),
                data_dir: root.join("data"),
                cache_dir: root.join("cache"),
            },
        )
    }

    fn sample_record() -> RemoteSessionRecord {
        RemoteSessionRecord {
            session_id: "remote-gpu-box-11434".to_owned(),
            target: "gpu-box".to_owned(),
            peer_host: "gpu-box.example-tailnet.ts.net".to_owned(),
            ssh_port: None,
            model: "qwen".to_owned(),
            remote_service_id: "svc-1".to_owned(),
            remote_cli: "rocm".to_owned(),
            remote_port: 11434,
            tailnet_port: 8000,
            base_url: "http://gpu-box.example-tailnet.ts.net:8000/v1".to_owned(),
            created_at_unix_ms: 1,
        }
    }

    #[test]
    fn the_remote_server_binds_loopback_but_demands_a_key() {
        // The bind stays loopback — the publish is what widens reach — so the
        // server would be credential-free without an explicit demand for a key.
        let command = remote_serve_command("rocm", &request());
        assert!(command.contains("--host 127.0.0.1"), "{command}");
        assert!(command.contains("--require-api-key"), "{command}");
        assert!(command.contains("--managed"), "{command}");
    }

    #[test]
    fn the_api_key_is_read_from_stdin_never_written_into_the_command() {
        // Both machines expose command arguments in their process tables, so an
        // interpolated key would be readable by any other user on either.
        let command = remote_serve_command("rocm", &request());
        assert!(
            command.starts_with("IFS= read -r ROCM_SERVE_API_KEY;"),
            "{command}"
        );
        assert!(command.contains("export ROCM_SERVE_API_KEY"), "{command}");
    }

    #[test]
    fn optional_engine_and_gpu_are_threaded_through() {
        let command = remote_serve_command(
            "rocm",
            &ServeRequest {
                engine: Some("vllm".to_owned()),
                gpu: Some("1".to_owned()),
                ..request()
            },
        );
        assert!(command.contains("--engine vllm"), "{command}");
        assert!(command.contains("--gpu 1"), "{command}");
    }

    #[cfg(unix)]
    #[test]
    fn hostile_values_survive_a_real_shell_as_one_literal_argument() {
        // The property that matters is not the shape of the quoting but what a
        // shell does with it. Ask one: each value must come back byte-identical,
        // proving it was neither expanded nor split nor able to start a second
        // command.
        for value in [
            "x'; rm -rf ~; echo '",
            "$(id)",
            "`id`",
            "a b",
            "it's",
            "*",
            "--not-a-flag",
            "qwen2.5-7b-instruct",
        ] {
            let output = std::process::Command::new("sh")
                .arg("-c")
                .arg(format!("printf %s {}", shell_quote(value)))
                .output()
                .expect("sh should run");
            assert_eq!(
                String::from_utf8_lossy(&output.stdout),
                value,
                "shell mangled {value:?}"
            );
        }
    }

    #[test]
    fn a_hostile_model_name_is_quoted_into_the_remote_command() {
        let command = remote_serve_command(
            "rocm",
            &ServeRequest {
                model: "x'; rm -rf ~; echo '".to_owned(),
                ..request()
            },
        );
        // Every embedded quote is closed and re-opened, so the payload cannot
        // end the quoting and start a statement of its own.
        assert!(command.contains(r"'\''"), "{command}");
        // And the flags we control still follow it as real flags.
        assert!(command.contains("--require-api-key"), "{command}");
    }

    #[test]
    fn shell_quoting_leaves_ordinary_values_alone_and_wraps_the_rest() {
        for inert in ["qwen2.5-7b-instruct", "vllm", "/models/a.gguf", "auto"] {
            assert_eq!(shell_quote(inert), inert);
        }
        assert_eq!(shell_quote("a b"), "'a b'");
        assert_eq!(shell_quote(""), "''");
        assert_eq!(shell_quote("it's"), r"'it'\''s'");
    }

    #[test]
    fn the_newest_service_on_the_port_is_the_one_just_started() {
        // A machine accumulates records on a port over its lifetime; picking an
        // older one would publish a port pointing at a dead server.
        let listing = r#"[
          {"service_id":"old","engine":"vllm","model_ref":"m","canonical_model_id":"m",
           "host":"127.0.0.1","port":11434,"endpoint_url":"http://127.0.0.1:11434/v1",
           "mode":"managed","status":"stopped","supervisor_pid":1,
           "manifest_path":"/a","log_path":"/b","engine_state_path":"/c",
           "created_at_unix_ms":100},
          {"service_id":"new","engine":"vllm","model_ref":"m","canonical_model_id":"m",
           "host":"127.0.0.1","port":11434,"endpoint_url":"http://127.0.0.1:11434/v1",
           "mode":"managed","status":"starting","supervisor_pid":2,
           "manifest_path":"/a","log_path":"/b","engine_state_path":"/c",
           "created_at_unix_ms":200},
          {"service_id":"other-port","engine":"vllm","model_ref":"m","canonical_model_id":"m",
           "host":"127.0.0.1","port":9999,"endpoint_url":"http://127.0.0.1:9999/v1",
           "mode":"managed","status":"ready","supervisor_pid":3,
           "manifest_path":"/a","log_path":"/b","engine_state_path":"/c",
           "created_at_unix_ms":300}
        ]"#;
        let transport =
            ScriptedTransport::new(vec![ScriptedStep::ok("services list --json", listing)]);
        assert_eq!(
            discover_started_service(&transport, "rocm", 11434).unwrap(),
            "new"
        );
    }

    #[test]
    fn a_registry_we_cannot_read_names_version_skew_as_the_likely_cause() {
        let transport = ScriptedTransport::new(vec![ScriptedStep::ok(
            "services list --json",
            "not json at all",
        )]);
        let error = discover_started_service(&transport, "rocm", 11434)
            .unwrap_err()
            .to_string();
        assert!(error.contains("different version"), "{error}");
    }

    #[test]
    fn the_two_lifecycles_are_reported_separately() {
        // The whole reason for two columns: which one is broken decides whether
        // the fix re-publishes or restarts.
        let healthy_but_unpublished = vec![(
            sample_record(),
            SessionObservation {
                server: ServerHealth::Healthy,
                publish: Some(publish::PublishState::Absent),
            },
        )];
        let rendered = render_status(&render_paths(), &healthy_but_unpublished);
        assert!(rendered.contains("model server: healthy"), "{rendered}");
        assert!(rendered.contains("endpoint published: no"), "{rendered}");
        assert!(
            rendered.contains("rocm remote attach"),
            "a live model with no endpoint should point at attach, not a restart: {rendered}"
        );
    }

    #[test]
    fn a_dead_server_is_not_offered_a_republish() {
        let rendered = render_status(
            &render_paths(),
            &[(
                sample_record(),
                SessionObservation {
                    server: ServerHealth::Failed,
                    publish: Some(publish::PublishState::Published),
                },
            )],
        );
        assert!(rendered.contains("model server: failed"), "{rendered}");
        assert!(!rendered.contains("attach"), "{rendered}");
        assert!(rendered.contains("stop"), "{rendered}");
    }

    #[test]
    fn a_session_the_remote_has_forgotten_warns_about_a_stray_endpoint() {
        // A publish survives reboots. A forgotten one is a GPU endpoint on the
        // tailnet with nothing tracking it.
        let rendered = render_status(
            &render_paths(),
            &[(
                sample_record(),
                SessionObservation {
                    server: ServerHealth::Gone,
                    publish: Some(publish::PublishState::Published),
                },
            )],
        );
        assert!(rendered.contains("may still be publishing"), "{rendered}");
    }

    #[test]
    fn an_unreachable_machine_reads_differently_from_a_broken_command() {
        let record = sample_record();

        let unreachable = ScriptedTransport::new(vec![]);
        assert_eq!(
            observe(&unreachable, &record).server,
            ServerHealth::Unreachable
        );

        let answered_badly =
            ScriptedTransport::new(vec![ScriptedStep::fails("services list --json", 1, "boom")]);
        assert_eq!(
            observe(&answered_badly, &record).server,
            ServerHealth::Error
        );
    }

    #[test]
    fn a_service_missing_from_the_remote_registry_is_gone_not_failed() {
        let record = sample_record();
        let transport = ScriptedTransport::new(vec![
            ScriptedStep::ok("services list --json", "[]"),
            ScriptedStep::ok("tailscale serve status --json", "{}"),
        ]);
        assert_eq!(observe(&transport, &record).server, ServerHealth::Gone);
    }

    #[test]
    fn remote_lifecycle_words_map_onto_reported_health() {
        for (raw, expected) in [
            ("ready", ServerHealth::Healthy),
            ("running", ServerHealth::Healthy),
            ("starting", ServerHealth::Pending),
            ("recovering", ServerHealth::Pending),
            ("failed", ServerHealth::Failed),
            ("stopped", ServerHealth::Failed),
        ] {
            assert_eq!(health_from_status(raw), expected, "{raw}");
        }
    }

    #[test]
    fn a_state_this_version_does_not_know_is_not_called_a_failure() {
        // Version skew is normal between the machine driving and the machine
        // driven. Reporting an unknown word as "failed" sends the user to
        // restart a model that may be serving perfectly.
        let health = health_from_status("quiescing");
        assert_eq!(
            health,
            ServerHealth::Unrecognised {
                raw: "quiescing".to_owned()
            }
        );
        assert!(
            health.describe().contains("different CLI version"),
            "{}",
            health.describe()
        );

        let rendered = render_status(
            &render_paths(),
            &[(
                sample_record(),
                SessionObservation {
                    server: health,
                    publish: Some(publish::PublishState::Published),
                },
            )],
        );
        assert!(
            rendered.contains("unrecognised state `quiescing`"),
            "{rendered}"
        );
        // And it must not be offered the dead-server repair.
        assert!(!rendered.contains("then serve again"), "{rendered}");
    }

    #[test]
    fn the_credential_is_recoverable_rather_than_shown_once_and_lost() {
        // The key is printed once when serving. Storing it without ever naming
        // where left the user's terminal scrollback as the only copy.
        let started = render_started(&render_paths(), &sample_record(), "the-key");
        assert!(started.contains("key file:"), "{started}");

        // Listings land in scrollback and CI logs, so they name the file rather
        // than echoing what is in it.
        let listed = render_status(
            &render_paths(),
            &[(
                sample_record(),
                SessionObservation {
                    server: ServerHealth::Healthy,
                    publish: Some(publish::PublishState::Published),
                },
            )],
        );
        assert!(listed.contains("key file:"), "{listed}");
        assert!(
            !listed.contains("the-key"),
            "a listing must not echo the credential:\n{listed}"
        );
        // And it lives with the session, not in the local service registry.
        assert!(listed.contains("remote-sessions"), "{listed}");
    }

    #[test]
    fn the_started_message_states_who_can_reach_the_endpoint() {
        // The loopback mental model from local serving does not carry over. A
        // user who assumes it does will never check their tailnet access rules.
        let rendered = render_started(&render_paths(), &sample_record(), "the-key");
        assert!(
            rendered.contains("every machine on your tailnet"),
            "{rendered}"
        );
        assert!(rendered.contains("the-key"), "{rendered}");
        assert!(
            rendered.contains("http://gpu-box.example-tailnet.ts.net:8000/v1"),
            "{rendered}"
        );
    }

    #[test]
    fn the_endpoint_url_matches_what_local_serving_records() {
        // Both sides record the OpenAI base including /v1, so a URL from either
        // pastes into the same client unchanged.
        assert_eq!(
            base_url_for("gpu-box.example-tailnet.ts.net", 8000),
            "http://gpu-box.example-tailnet.ts.net:8000/v1"
        );
    }

    #[test]
    fn a_teardown_that_leaves_the_model_running_keeps_the_record() {
        // The record is the only thing on this machine holding the model's id
        // and where it runs. Dropping it while the model is up leaves a GPU
        // occupied by something the user can no longer name, let alone stop.
        let rendered = render_stopped(&sample_record(), true, false, false);
        assert!(
            rendered.contains("model server stopped: NOT CONFIRMED"),
            "{rendered}"
        );
    }

    #[test]
    fn a_forced_teardown_is_louder_than_a_clean_one() {
        // Forgetting a session is only safe if the user is told exactly what may
        // outlive it. A quiet --force is how a live endpoint stops being
        // anyone's problem.
        let forced = render_stopped(&sample_record(), false, false, true);
        assert!(forced.contains("may still be running it"), "{forced}");
        assert!(
            forced.contains("tailscale serve --tcp=8000 off"),
            "a forced stop must say how to clear the endpoint: {forced}"
        );
        assert!(
            forced.contains("services list"),
            "and how to find the model: {forced}"
        );

        // A clean stop stays quiet: there is nothing left to warn about.
        let clean = render_stopped(&sample_record(), true, true, false);
        assert!(!clean.contains("may still be running it"), "{clean}");
        assert!(clean.contains("endpoint withdrawn: yes"), "{clean}");
    }

    #[test]
    fn a_failed_publish_stops_the_model_it_started() {
        // Otherwise a GPU is held by something nobody can call and nothing
        // records.
        let (root, paths) = temp_paths("unwind-publish");
        let transport = ScriptedTransport::new(vec![ScriptedStep::ok("services stop", "")]);

        let leftovers =
            unwind_partial_serve(&transport, &paths, "sess", "rocm", Some("svc-1"), None);

        assert!(leftovers.is_empty(), "{leftovers:?}");
        assert!(
            transport.calls().iter().any(|call| matches!(
                call,
                TransportCall::Exec { command, .. } if command.contains("services stop svc-1")
            )),
            "the started model should have been stopped: {:?}",
            transport.calls()
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn what_the_unwind_could_not_undo_is_reported_not_swallowed() {
        // A discarded cleanup failure is how a machine ends up running a model
        // nobody remembers starting.
        let (root, paths) = temp_paths("unwind-failed");
        let transport = ScriptedTransport::new(vec![
            ScriptedStep::ok("tailscale serve status --json", PUBLISHED_FIXTURE),
            ScriptedStep::fails("tailscale serve --tcp=8000 off", 1, "daemon busy"),
            ScriptedStep::fails("services stop", 1, "no such service"),
        ]);

        let leftovers = unwind_partial_serve(
            &transport,
            &paths,
            "sess",
            "rocm",
            Some("svc-1"),
            Some((8000, 11434)),
        );

        assert_eq!(leftovers.len(), 2, "{leftovers:?}");
        assert!(leftovers[0].contains("still be published"), "{leftovers:?}");
        assert!(leftovers[1].contains("still be running"), "{leftovers:?}");

        let described =
            describe_leftovers(anyhow::anyhow!("publish failed"), "gpu-box", &leftovers)
                .to_string();
        assert!(described.contains("could not undo"), "{described}");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn serve_sends_the_key_over_stdin_when_it_starts_the_model() {
        // Guards the pairing: the command reads stdin, and the caller actually
        // supplies it. Either alone leaves the server without a credential.
        let transport = ScriptedTransport::new(vec![ScriptedStep::ok("read -r", "")]);
        transport
            .exec_with_stdin(&remote_serve_command("rocm", &request()), Some("k"))
            .expect("scripted");
        assert!(matches!(
            transport.calls().first(),
            Some(TransportCall::Exec { stdin: Some(key), .. }) if key == "k"
        ));
    }
}
