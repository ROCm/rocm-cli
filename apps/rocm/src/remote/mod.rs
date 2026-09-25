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
        } => remote_doctor(&target, symptom.as_deref(), top, ssh_port),
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
) -> Result<()> {
    // Resolved the same way `serve` resolves it, so a name that serves is a name
    // that can be checked first — which is the order these are meant to be used
    // in.
    resolve_target(target)?;
    let transport = SshTransport::new(target, ssh_port)?;
    // Deliberately not `ensure_ready`: that provisions a missing CLI, and a
    // health check has no business installing software on a machine it was only
    // asked to look at. The daemon's read-only allowlist
    // (`ensure_rocm_command_is_read_only` in `apps/rocmd/src/lib.rs`) admits
    // `remote doctor` without the approval flow, which is only true while this
    // stays a pure read.
    let remote_cli = bootstrap::locate_cli(&transport, target)?;
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
    serve_with_transport(&transport, &paths, &peer_host, request)
}

/// The body of [`serve`], taking its transport and paths rather than building
/// them.
///
/// This is the seam that makes the credential handoff below testable end to
/// end: a test can drive this with a [`transport::ScriptedTransport`] and an
/// isolated [`AppPaths`], and see the same stdin write the real command path
/// produces, instead of only checking [`remote_serve_command`] and
/// [`Transport::exec_with_stdin`] in isolation with nothing pairing them.
fn serve_with_transport(
    transport: &dyn Transport,
    paths: &AppPaths,
    peer_host: &str,
    request: &ServeRequest,
) -> Result<()> {
    let session_id = RemoteSessionRecord::id_for(peer_host, request.remote_port);
    // Refuse a name something already sits under, exactly as `publish` refuses a
    // port that already forwards somewhere else.
    //
    // `id_for` is the machine and the port, with no nonce, so a second `serve`
    // against the same box computes the same id. Without this the second run
    // overwrites the first session's key when it mints its own, and — when it
    // then fails to start, because the first session still holds the port —
    // deletes the shared key outright. The user reads "failed to start the model" as "nothing
    // happened", while the first session is still serving on a published tailnet
    // endpoint that can no longer be called.
    //
    // Checked before the readiness probe, so a refusal costs no round trip and
    // cannot provision a machine this command then declines to use. That is also
    // the limit of what this check can do: it is not the claim. Provisioning sits
    // between here and the write, so a run starting inside that window sees the
    // same free name — `session::store_key` below is what actually takes it, in
    // one indivisible step. This check exists because it is cheap and because it
    // can see *what* is on disk, and a record and a stray credential need
    // different remedies.
    if session::exists(paths, &session_id) {
        // Which remedy applies depends on what is actually on disk. A key with no
        // record is not a session `stop` can reach: `load_all` enumerates `*.json`
        // only, so `resolve` cannot see it and would answer "no remote sessions are
        // recorded on this machine". Telling the user to stop it would be advice
        // that provably fails, on the one state this guard exists to detect.
        //
        // The key-only case is reachable: `serve` mints the credential before it
        // can know the remote service id, so any failure between the two — a
        // registry it cannot parse, a publish that will not confirm — leaves the
        // key behind with no record beside it.
        if RemoteSessionRecord::path_in(paths, &session_id).exists() {
            bail!(
                "a session is already recorded for {} on port {}.\n\
                 Serving again here would take over its credential and could leave it \
                 running with no way to call it.\n\
                 Stop it first: rocm remote stop {}\n\
                 Or serve on another port with `--remote-port`.",
                request.target,
                request.remote_port,
                session_id
            );
        }
        bail!(
            "a credential from an earlier `rocm remote serve` on {} port {} is still on \
             this machine, with no session recorded beside it.\n\
             An earlier attempt got far enough to mint a key and not far enough to record \
             what it started, so a model may be running there untracked.\n\
             Check the machine: ssh {} -- rocm services list\n\
             Once you are sure nothing is using it, delete: {}\n\
             Or serve on another port with `--remote-port`.",
            request.target,
            request.remote_port,
            request.target,
            session::key_path(paths, &session_id).display()
        );
    }

    println!("Preparing {} ...", request.target);
    let remote_cli = bootstrap::ensure_ready_with(
        transport,
        &request.target,
        &request.channel,
        request.install_rocm,
    )?;

    // Mint the credential before starting anything. A model that comes up
    // unauthenticated and is then published is exposed for the window between
    // the two, and the whole point of publishing is that the window is visible
    // to every machine on the tailnet.
    //
    // Writing it is also what *claims* the session name, and that is the check
    // that actually decides. The `session::exists` call above runs before
    // `ensure_ready_with` so a refusal costs no round trip and cannot provision
    // a machine this command then declines to use — but that placement is
    // precisely what makes it unable to decide: provisioning takes minutes, and
    // a second run started inside that window would see the same free name.
    // `store_key` creates the file exclusively, so only one run can be here.
    let api_key = rocm_core::generate_endpoint_api_key();
    if let Err(error) = session::store_key(paths, &session_id, &api_key) {
        if error.downcast_ref::<session::NameAlreadyHeld>().is_some() {
            // Nothing of ours is on disk — the create is what failed — so there
            // is nothing to unwind, and in particular nothing to clear: the
            // credential under this name belongs to the run that won, and the
            // model it guards may already be published.
            bail!(
                "another `rocm remote serve` claimed {} on port {} while this one was \
                 preparing the machine.\n\
                 Both runs name the session after the machine and the port, so carrying on \
                 would take over its credential and could leave it running with no way to \
                 call it.\n\
                 See what is there: rocm remote status\n\
                 Or serve on another port with `--remote-port`.",
                request.target,
                request.remote_port
            );
        }
        return Err(error.context(
            "refusing to publish a model endpoint whose API key could not be saved locally: \
             without it you would have no way to call the endpoint you are about to expose",
        ));
    }

    println!("Starting {} on {} ...", request.model, request.target);
    // The trailing newline is what makes the `&&` chain in
    // `remote_serve_command` work, not cosmetics: `IFS= read -r` returns
    // non-zero when it hits EOF without one, even though it did assign the
    // variable. Send the bare key and the remote reads it, reports failure, and
    // the `&&` stops the model from ever starting. It is also what makes the
    // short circuit *mean* something — a write truncated part-way delivers no
    // newline, so the read fails and nothing serves with half a key.
    let key_payload = format!("{api_key}\n");
    let start = match transport.exec_with_stdin(
        &remote_serve_command(&remote_cli, request),
        Some(&key_payload),
    ) {
        Ok(start) => start,
        Err(error) => {
            // The command may already have reached the remote — contact can be
            // lost after the remote has begun starting the model — so its state
            // is not knowable from here. Drop the key that now guards nothing,
            // and say so rather than leaving the user to assume nothing
            // happened.
            //
            // Note this also catches the case where ssh never reached the host
            // at all, where nothing was started and the uncertainty is
            // overstated. The inner error says "could not reach" plainly, so
            // the user is not misled, but telling the two apart here would need
            // the transport to report unreachability as something richer than a
            // message. Left as is rather than grown a typed error for it.
            session::clear_key(paths, &session_id);
            return Err(error.context(format!(
                "lost contact with {} while starting the model, so it may or may not be \
                 running.\n\
                 Check with: ssh {} -- {remote_cli} services list",
                request.target, request.target
            )));
        }
    };
    if !start.success {
        session::clear_key(paths, &session_id);
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
        match discover_started_service(transport, &remote_cli, request.remote_port) {
            Ok(service_id) => service_id,
            Err(error) => {
                // The key stays. The model is very likely running and it was
                // handed this credential, so deleting our only copy would leave
                // a service the user can find but cannot call — or stop through
                // its own API. Nothing can be stopped by name when the name is
                // what could not be read, so say where to look and hand back the
                // credential rather than implying it was all cleaned up.
                return Err(error.context(format!(
                    "a model may now be running on {} port {} with nothing tracking it.\n\
                     Check with: ssh {} -- {remote_cli} services list\n\
                     Its API key was kept at {} — it is the only copy.",
                    request.target,
                    request.remote_port,
                    request.target,
                    session::key_path(paths, &session_id).display()
                )));
            }
        };

    println!("Publishing to the tailnet ...");
    if let Err(error) = publish::publish(transport, request.tailnet_port, request.remote_port) {
        // The model is up but unreachable. Stop it rather than leaving a GPU
        // occupied by something nobody can call and nothing records.
        let leftovers = unwind_partial_serve(
            transport,
            paths,
            &session_id,
            &remote_cli,
            Some(&remote_service_id),
            None,
        );
        return Err(describe_leftovers(error, &request.target, &leftovers));
    }

    let base_url = base_url_for(peer_host, request.tailnet_port);
    let record = RemoteSessionRecord {
        session_id: session_id.clone(),
        target: request.target.clone(),
        peer_host: peer_host.to_owned(),
        ssh_port: request.ssh_port,
        model: request.model.clone(),
        remote_service_id: remote_service_id.clone(),
        remote_cli: remote_cli.clone(),
        remote_port: request.remote_port,
        tailnet_port: request.tailnet_port,
        base_url,
        created_at_unix_ms: RemoteSessionRecord::now(),
    };
    if let Err(error) = record.write(paths) {
        // The endpoint is live and published at this point. Without a record
        // nothing on this machine knows it exists, so leaving it up would be
        // exactly the untracked exposure the whole design tries to avoid.
        let leftovers = unwind_partial_serve(
            transport,
            paths,
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
    println!("{}", render_started(paths, &record, &api_key));
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

    // Only drop the credential when there is nothing left for it to guard. If any
    // step above failed, the model may still be running and its endpoint may still
    // be published, and the key is the only way to call it — deleting it here
    // would leave a live tailnet endpoint that nobody can use and nothing tracks.
    //
    // This matches what the rest of the module already does: the
    // `discover_started_service` failure path keeps the key and says where it is,
    // and `stop_with_transport` keeps it whenever it cannot confirm both halves.
    if leftovers.is_empty() {
        session::clear_key(paths, session_id);
    } else {
        leftovers.push(format!(
            "its API key was kept at {} — it is the only copy",
            session::key_path(paths, session_id).display()
        ));
    }
    leftovers
}

/// Attach what could not be cleaned up to the error that caused it.
fn describe_leftovers(error: anyhow::Error, target: &str, leftovers: &[String]) -> anyhow::Error {
    if leftovers.is_empty() {
        return error;
    }
    error.context(format!(
        "{target} was left with things this command could not undo:\n{}\n\
         Check it with: ssh {target} -- rocm services list",
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
///
/// Joined with `&&`, not `;`, and that is load-bearing rather than stylistic.
/// [`transport::run_with_piped_io`] treats a broken pipe on the stdin writer as a
/// mere symptom whenever the command itself failed, which is only sound if a
/// failed `read` cannot be followed by a successful `serve`. `&&` is what makes
/// the shell enforce that; under `;` the compound's status is whatever `serve`
/// returned, so a truncated key could report success on the credential path.
fn remote_serve_command(remote_cli: &str, request: &ServeRequest) -> String {
    let mut command = format!(
        "IFS= read -r ROCM_SERVE_API_KEY && export ROCM_SERVE_API_KEY && \
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
    /// The publishing state, or why it could not be read. Not an `Option`:
    /// "could not read it" has two causes worth telling apart, and an `Option`
    /// can only say that one of them happened.
    pub(crate) publish: publish::PublishObservation,
}

/// Probe one session over the control channel.
fn observe(transport: &dyn Transport, record: &RemoteSessionRecord) -> SessionObservation {
    let listing = match transport.exec(&format!("{} services list --json --all", record.remote_cli))
    {
        Ok(outcome) if outcome.success => outcome.stdout,
        // Reached the machine but the command failed, versus could not reach it
        // at all. Different problems, different fixes, so different words.
        // The machine is reachable — only this one command failed — so the
        // publishing state is still worth asking about, and the answer is a real
        // observation rather than a guess. Reporting `Unreachable` here rendered
        // as "the machine could not be asked", which is the wrong failure: it
        // sends the user to check the network for a machine that just answered.
        Ok(_) => {
            return SessionObservation {
                server: ServerHealth::Error,
                publish: publish::observe(transport, record.tailnet_port, record.remote_port),
            };
        }
        Err(_) => {
            return SessionObservation {
                server: ServerHealth::Unreachable,
                publish: publish::PublishObservation::Unreachable(
                    "the machine could not be reached over ssh".to_owned(),
                ),
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
        // `publish::observe`, not `publish_state(..).ok()`: the same two cases
        // separated above for `services list` — reached-but-failed versus never
        // reached — must stay separated here. `.ok()` collapsed them, so a
        // remote that answered with a concrete reason was reported as one that
        // was never asked.
        publish: publish::observe(transport, record.tailnet_port, record.remote_port),
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
            let observed = SshTransport::new(&record.target, record.ssh_port).map_or_else(
                // A record naming a machine ssh cannot address is not a reason
                // to hide every other session from the listing.
                //
                // `Unreachable`, not `Error`: nothing was ever sent, so this is
                // the never-reached side of the split, not "answered in a way we
                // could not read". And the constructor's own reason is carried
                // rather than replaced with a generic line — it names what is
                // wrong with the destination, which is the one thing that makes
                // this fixable.
                |error| SessionObservation {
                    server: ServerHealth::Unreachable,
                    publish: publish::PublishObservation::Unreachable(format!("{error:#}")),
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
    use publish::PublishObservation::{Failed, Known, Unreachable};

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
                Known(publish::PublishState::Published) => "yes".to_owned(),
                Known(publish::PublishState::Absent) => "no".to_owned(),
                Known(publish::PublishState::Foreign { forwards_to }) =>
                    format!("no — that port now forwards to {forwards_to}"),
                // Not "no": Funnel is classified before anything is asked about
                // our own forward and short-circuits, so this state says
                // nothing either way about whether the endpoint is published.
                // Leading with "no" answered a question it had not looked at,
                // and buried the one thing here that needs acting on.
                Known(publish::PublishState::FunnelAllowed) => format!(
                    "exposed — Tailscale Funnel is allowed on that port, which puts it on the \
                     public internet; run `tailscale funnel --tcp={} off` on the remote, then \
                     check again",
                    record.tailnet_port
                ),
                // The three below all mean "could not tell", and none may be
                // read as "no": an endpoint that is still up must never render
                // as one that is down, or the user stops looking for it. They
                // stay distinct because the next step differs — fix the reply,
                // fix the remote's tailscale, or fix the connection.
                Known(publish::PublishState::Unreadable) =>
                    "unknown — the machine's reply could not be read".to_owned(),
                // Reached, and it told us why. Printing its own words beats
                // "could not be asked", which describes a different failure and
                // sends the user to check the network instead of the remote.
                Failed(why) => format!("unknown — the machine answered: {why}"),
                Unreachable(why) => format!("unknown — the machine could not be asked: {why}"),
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
        (
            ServerHealth::Healthy,
            publish::PublishObservation::Known(publish::PublishState::Absent),
        ) => Some(format!(
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
    attach_with_transport(&transport, &record)
}

/// The body of [`attach`], taking its transport rather than building one.
///
/// The same seam [`serve_with_transport`] exists for, and for the same reason:
/// without it the only testable part of `attach` is [`render_status`], and the
/// refusal below — the one thing standing between a dead model and an endpoint
/// that answers with connection refused — is reachable by no test at all.
fn attach_with_transport(transport: &dyn Transport, record: &RemoteSessionRecord) -> Result<()> {
    // Re-declaring a publish is cheap, but doing it over a dead model server
    // would produce an endpoint that answers with connection refused — worse
    // than one that is honestly absent.
    let observed = observe(transport, record);
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

    publish::publish(transport, record.tailnet_port, record.remote_port)?;
    println!("Endpoint re-published: {}", record.base_url);
    println!("The model was not restarted.");
    Ok(())
}

fn stop(needle: &str, force: bool) -> Result<()> {
    let paths = AppPaths::discover()?;
    let record = session::resolve(&paths, needle)?;
    let transport = SshTransport::new(&record.target, record.ssh_port)?;
    stop_with_transport(&transport, &paths, &record, force)
}

/// The body of [`stop`], taking its transport and paths rather than building
/// them.
///
/// Teardown is the path where getting it wrong is worst — forgetting a session
/// whose endpoint is still published leaves a GPU endpoint on the tailnet that
/// nothing tracks — and it was the one path no test could drive.
fn stop_with_transport(
    transport: &dyn Transport,
    paths: &AppPaths,
    record: &RemoteSessionRecord,
    force: bool,
) -> Result<()> {
    // Withdraw before stopping the model. If only one of the two can be done,
    // the endpoint being gone is the one that matters: a stopped model behind a
    // live publish is a refused connection, but a live model behind a forgotten
    // publish is an open GPU endpoint nobody is tracking.
    let withdrawn = publish::withdraw(transport, record.tailnet_port, record.remote_port);
    let stopped = transport.exec(&format!(
        "{} services stop {} --yes",
        record.remote_cli,
        shell_quote(&record.remote_service_id)
    ));
    let stop_failure = describe_stop_failure(&stopped);
    let model_stopped = stop_failure.is_none();

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

        if let Some(why) = &stop_failure {
            // Same reasoning one step further in. The record is the only thing on
            // this machine that knows the model's id and where it runs; dropping
            // it while the model is still up leaves a GPU occupied by something
            // the user can no longer name.
            //
            // The remote's own words are carried, as the withdraw branch above
            // already does. Without them this arm reports only *that* the stop
            // failed, which reads the same whether the machine refused, the
            // service was already gone, or ssh never got there — three different
            // next steps behind one sentence.
            bail!(
                "the endpoint for {} was withdrawn, but its model could not be stopped: {why}\n\
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

    session::clear_key(paths, &record.session_id);
    record.remove(paths);

    print!(
        "{}",
        render_stopped(record, withdrawn.is_ok(), model_stopped, force)
    );
    Ok(())
}

/// Why the remote could not stop the model, or `None` if it did.
///
/// Keeps the two failure shapes apart the way the rest of this module does:
/// a machine that answered and refused carries its own stderr and exit status,
/// and one that was never reached carries the transport's reason. Collapsing
/// them into a bare bool is what left the teardown error with nothing to say.
fn describe_stop_failure(stopped: &Result<transport::RemoteOutcome>) -> Option<String> {
    match stopped {
        Ok(outcome) if outcome.success => None,
        Ok(outcome) => {
            let stderr = outcome.stderr.trim();
            let code = outcome
                .code
                .map_or_else(|| "signal".to_owned(), |code| code.to_string());
            Some(if stderr.is_empty() {
                format!("the machine answered but the command failed (exit {code})")
            } else {
                format!("the machine answered (exit {code}): {stderr}")
            })
        }
        Err(error) => Some(format!("{error:#}")),
    }
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
            command.starts_with("IFS= read -r ROCM_SERVE_API_KEY &&"),
            "{command}"
        );
        assert!(command.contains("export ROCM_SERVE_API_KEY"), "{command}");
    }

    /// Run the generated serve command under a real shell with `stdin_payload`
    /// on its stdin, and report whether it succeeded and whether it reached the
    /// serve step.
    ///
    /// `echo REACHED_SERVE` stands in for the remote CLI: reaching it means the
    /// `&&` chain did not short-circuit, and the marker says so in the failure.
    fn serve_command_under_a_shell(stdin_payload: &str) -> (bool, bool) {
        use std::io::Write as _;

        let command = remote_serve_command("echo REACHED_SERVE", &request());
        let mut child = std::process::Command::new("sh")
            .arg("-c")
            .arg(&command)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("failed to run the generated command under a shell");
        child
            .stdin
            .take()
            .expect("stdin was piped")
            .write_all(stdin_payload.as_bytes())
            .expect("the payload is far smaller than a pipe buffer");

        let output = child.wait_with_output().expect("the shell should finish");
        (
            output.status.success(),
            String::from_utf8_lossy(&output.stdout).contains("REACHED_SERVE"),
        )
    }

    #[test]
    fn only_a_complete_key_lets_the_command_reach_serve() {
        // The invariant `transport::run_with_piped_io` leans on: it demotes a
        // broken-pipe write error to a symptom whenever the command also failed,
        // which is only sound if a failed `read` cannot be followed by a
        // successful `serve`. Under `;` it could be — the compound's status
        // would be whatever `serve` returned — and a truncated API key would
        // then classify as success on the one path guarding the model.
        //
        // Asked of a real shell, and asked for the *exit status*: the whole
        // property is what `&&` does to a compound command, which no assertion
        // about the string can see. The same reasoning as
        // `the_staging_path_still_expands_on_the_remote_shell` in `provision`.

        // Nothing at all — the far side of a pipe that broke before any byte.
        assert_eq!(
            serve_command_under_a_shell(""),
            (false, false),
            "no key arrived, yet the command served or reported success"
        );

        // A key with no terminating newline. This is the case that matters and
        // the one a closed-stdin test cannot see: `read` assigns the variable
        // but still returns non-zero at EOF, so this is indistinguishable from
        // a truncated write — and must not serve. It is also exactly what the
        // caller used to send, which made the `&&` chain refuse every real
        // start until the caller began terminating the payload.
        assert_eq!(
            serve_command_under_a_shell("abc123"),
            (false, false),
            "an unterminated key is a truncated key; it must not reach serve"
        );

        // A complete line: the shape `serve_with_transport` actually sends.
        assert_eq!(
            serve_command_under_a_shell("abc123\n"),
            (true, true),
            "a complete key must reach serve, or no remote model ever starts"
        );
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
                publish: publish::PublishObservation::Known(publish::PublishState::Absent),
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
    fn a_funnel_exposed_port_is_reported_as_exposure_not_as_a_publish_answer() {
        // Funnel is classified before anything is asked about our own forward
        // and short-circuits, so this state says nothing either way about
        // whether the endpoint is published. Leading the line with "no"
        // answered a question it had not looked at, and buried the one thing
        // on it that needs acting on.
        // Port 443, not the 8000 the other fixtures use: Funnel only serves
        // 443, 8443 and 10000, so a session that can reach this state at all
        // is one started with `--tailnet-port`. Rendering the remedy for a
        // port Funnel cannot listen on would pin a line no user could ever
        // see.
        let record = RemoteSessionRecord {
            tailnet_port: 443,
            ..sample_record()
        };
        let rendered = render_status(
            &render_paths(),
            &[(
                record.clone(),
                SessionObservation {
                    server: ServerHealth::Healthy,
                    publish: publish::PublishObservation::Known(
                        publish::PublishState::FunnelAllowed,
                    ),
                },
            )],
        );
        assert!(
            !rendered.contains("endpoint published: no"),
            "Funnel exposure must not be rendered as an answer about publishing: {rendered}"
        );
        assert!(rendered.contains("exposed"), "{rendered}");
        assert!(
            rendered.contains("public internet"),
            "the reason it matters must be on the line: {rendered}"
        );
        // The remedy has to be copy-pasteable, so the real port belongs here
        // rather than a `<port>` placeholder the user has to substitute.
        assert!(
            rendered.contains(&format!(
                "tailscale funnel --tcp={} off",
                record.tailnet_port
            )),
            "{rendered}"
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
                    publish: publish::PublishObservation::Known(publish::PublishState::Published),
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
                    publish: publish::PublishObservation::Known(publish::PublishState::Published),
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
    fn a_remote_that_answers_about_publishing_is_not_reported_as_one_that_was_never_asked() {
        // The same distinction the test above pins for the model server, for
        // the publish half. It used to be lost: `publish_state(..).ok()`
        // mapped both causes to `None`, and the status line said "the machine
        // could not be asked" for a machine that had answered with a concrete,
        // actionable reason — sending the user to check the network instead of
        // the remote's tailscale.
        let record = sample_record();

        // Reached: `services list` succeeds, then `serve status` fails with
        // the remote's own words.
        let answered = ScriptedTransport::new(vec![
            ScriptedStep::ok("services list --json", "[]"),
            ScriptedStep::fails(
                "tailscale serve status --json",
                127,
                "tailscale: command not found",
            ),
        ]);
        let observed = observe(&answered, &record);
        let publish::PublishObservation::Failed(why) = &observed.publish else {
            panic!("a remote that answered must not be reported as unreachable: {observed:?}");
        };
        assert!(why.contains("tailscale: command not found"), "{why}");

        // Never reached at all: nothing scripted, so the transport errors.
        let unreachable = ScriptedTransport::new(vec![]);
        assert!(
            matches!(
                observe(&unreachable, &record).publish,
                publish::PublishObservation::Unreachable(_)
            ),
            "a host that was never reached must say so"
        );

        // And the two must not render the same, which is the whole point.
        let rendered_failed = render_status(&render_paths(), &[(record.clone(), observed)]);
        let rendered_unreachable = render_status(
            &render_paths(),
            &[(record.clone(), observe(&unreachable, &record))],
        );
        assert!(
            rendered_failed.contains("tailscale: command not found"),
            "the remote's own reason belongs on the status line: {rendered_failed}"
        );
        assert_ne!(
            rendered_failed, rendered_unreachable,
            "answered-and-failed must not read identically to never-answered"
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
                    publish: publish::PublishObservation::Known(publish::PublishState::Published),
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
                    publish: publish::PublishObservation::Known(publish::PublishState::Published),
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
    fn a_clean_unwind_still_drops_the_key_it_minted() {
        // The gate cuts both ways: when cleanup fully succeeds there is nothing
        // left for the credential to guard, so leaving it behind would be the
        // orphaned-secret half of the same mistake.
        let (root, paths) = temp_paths("unwind-clean");
        session::store_key(&paths, "sess", "the-key").unwrap();
        let transport = ScriptedTransport::new(vec![
            ScriptedStep::ok("tailscale serve status --json", PUBLISHED_FIXTURE),
            ScriptedStep::ok("tailscale serve --tcp=8000 off", ""),
            ScriptedStep::ok("services stop", ""),
        ]);

        let leftovers =
            unwind_partial_serve(&transport, &paths, "sess", "rocm", Some("svc-1"), None);

        assert!(leftovers.is_empty(), "{leftovers:?}");
        assert!(
            !session::key_path(&paths, "sess").exists(),
            "a clean unwind must not leave the credential behind"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn serving_twice_on_one_machine_and_port_is_refused_rather_than_overwriting() {
        // `id_for` is host + port with no nonce, so a second serve computes the
        // same id. Without this refusal it overwrites the first session's key,
        // then deletes it when the start fails on the port the first session
        // still holds — leaving a live, published endpoint nobody can call.
        let (root, paths) = temp_paths("serve-collision");
        let request = request();
        let peer_host = "gpu-box.example-tailnet.ts.net";
        let session_id = RemoteSessionRecord::id_for(peer_host, request.remote_port);
        session::store_key(&paths, &session_id, "the-first-sessions-key").unwrap();

        // No scripted steps: the refusal must land before anything is sent.
        let transport = ScriptedTransport::new(Vec::new());
        let error = serve_with_transport(&transport, &paths, peer_host, &request)
            .expect_err("a second serve on the same machine and port must be refused");
        let rendered = format!("{error:#}");

        // A key with no record beside it is NOT a session `stop` can reach:
        // `load_all` enumerates `*.json` only, so `resolve` would answer "no
        // remote sessions are recorded on this machine". Pointing the user at
        // `stop` here would be advice that provably fails.
        assert!(
            !rendered.contains("rocm remote stop"),
            "a key-only leftover cannot be stopped, so the refusal must not say to: {rendered}"
        );
        assert!(
            rendered.contains(&session::key_path(&paths, &session_id).display().to_string()),
            "the refusal must name the file the user has to deal with: {rendered}"
        );

        // The first session's credential is untouched.
        assert_eq!(
            std::fs::read_to_string(session::key_path(&paths, &session_id)).unwrap(),
            "the-first-sessions-key"
        );
        assert!(
            transport.calls().is_empty(),
            "nothing may be sent to the machine before the refusal: {:?}",
            transport.calls()
        );

        // With a real record beside the key, `stop` *is* the remedy, and the
        // refusal says so.
        let mut record = sample_record();
        record.session_id = session_id.clone();
        record.peer_host = peer_host.to_owned();
        record.write(&paths).unwrap();
        let error = serve_with_transport(&transport, &paths, peer_host, &request)
            .expect_err("a recorded session must still be refused");
        let rendered = format!("{error:#}");
        assert!(rendered.contains("already recorded"), "{rendered}");
        assert!(
            rendered.contains(&format!("rocm remote stop {session_id}")),
            "{rendered}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn serves_racing_on_one_machine_and_port_leave_exactly_one_session() {
        // The sequential refusal above is the outcome; this is the mechanism.
        // `id_for` is host + port with no nonce, so every run here computes the
        // same name, and the check that the name is free is separated from the
        // write that takes it by the whole readiness probe — minutes, when that
        // probe provisions a CLI. A check-then-write pair cannot survive two runs
        // in that window: both see a free name, both mint a key under it, and the
        // one that loses the port then deletes the other's credential on its way
        // out, leaving a published endpoint nobody can call.
        //
        // Asserting on the end state would not see it. "One record, one key
        // file" is true when both runs succeed too — they share a name. What is
        // only true when the claim is atomic is that exactly one run *returns*
        // having claimed it.
        //
        // The barrier is what makes the window real rather than hoped for: every
        // thread reaches the free-name check before any of them has had time to
        // write.
        const RACERS: usize = 16;
        let (root, paths) = temp_paths("serve-race");
        let peer_host = "gpu-box.example-tailnet.ts.net";
        let barrier = std::sync::Barrier::new(RACERS);

        let claimed = std::thread::scope(|scope| {
            let racers = (0..RACERS)
                .map(|_| {
                    scope.spawn(|| {
                        // One transport per thread: the scripted double records
                        // calls in a `RefCell` and is not shareable, and a real
                        // second `serve` would open its own connection anyway.
                        let transport = ScriptedTransport::new(full_serve_steps());
                        barrier.wait();
                        serve_with_transport(&transport, &paths, peer_host, &request()).is_ok()
                    })
                })
                .collect::<Vec<_>>();
            racers
                .into_iter()
                .map(|racer| racer.join().expect("no racer may panic"))
                .filter(|claimed| *claimed)
                .count()
        });

        assert_eq!(
            claimed, 1,
            "exactly one concurrent serve may claim a machine and port; {claimed} did"
        );

        let session_id = RemoteSessionRecord::id_for(peer_host, request().remote_port);
        let key = session::key_path(&paths, &session_id);
        assert!(
            key.exists(),
            "the run that won must still hold its credential: a loser's cleanup may not \
             delete a key it did not mint"
        );
        assert!(
            !std::fs::read_to_string(&key).expect("read").is_empty(),
            "the surviving credential must be the winner's, not an emptied file"
        );
        assert_eq!(
            session::load_all(&paths).expect("load").len(),
            1,
            "one session, not one per racer"
        );

        let _ = std::fs::remove_dir_all(root);
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
        session::store_key(&paths, "sess", "the-key").unwrap();
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

        assert_eq!(leftovers.len(), 3, "{leftovers:?}");
        assert!(leftovers[0].contains("still be published"), "{leftovers:?}");
        assert!(leftovers[1].contains("still be running"), "{leftovers:?}");

        // The credential outlives a cleanup that could not finish. Both steps
        // above failed, so the model may still be serving on a published
        // endpoint, and this key is the only way to call it.
        assert!(leftovers[2].contains("only copy"), "{leftovers:?}");
        assert!(
            session::key_path(&paths, "sess").exists(),
            "a failed unwind must not delete the key to a service that may still be running"
        );

        let described =
            describe_leftovers(anyhow::anyhow!("publish failed"), "gpu-box", &leftovers)
                .to_string();
        assert!(described.contains("could not undo"), "{described}");
        let _ = std::fs::remove_dir_all(root);
    }

    /// Everything a fully successful `serve` touches on the remote, in call
    /// order: the readiness probe, the serve command itself, the service
    /// registry lookup that follows it, and the publish that follows that.
    fn full_serve_steps() -> Vec<ScriptedStep> {
        let mut steps = ready_steps();
        steps.push(ScriptedStep::ok("read -r", ""));
        steps.push(ScriptedStep::ok(
            "services list --json --all",
            r#"[{"service_id":"svc-1","engine":"vllm","model_ref":"m","canonical_model_id":"m",
                "host":"127.0.0.1","port":11434,"endpoint_url":"http://127.0.0.1:11434/v1",
                "mode":"managed","status":"ready","supervisor_pid":1,
                "manifest_path":"/a","log_path":"/b","engine_state_path":"/c",
                "created_at_unix_ms":1}]"#,
        ));
        // `publish` checks status once before writing and once after, to confirm
        // the remote actually accepted the forward rather than trusting exit 0.
        // Reporting it published both times keeps this fixture representing a
        // machine with nothing else competing for the port.
        steps.push(ScriptedStep::ok(
            "tailscale serve status --json",
            PUBLISHED_FIXTURE,
        ));
        steps.push(ScriptedStep::ok("tailscale serve --bg", ""));
        steps
    }

    /// The readiness probe steps `bootstrap::ensure_ready_with` needs to find a
    /// machine with an existing CLI, ROCm, and Tailscale already present — the
    /// same fixture `bootstrap`'s own tests use for a ready machine.
    fn ready_steps() -> Vec<ScriptedStep> {
        vec![
            ScriptedStep::ok("rocm --version", "rocm 1.2.3"),
            ScriptedStep::ok("command -v rocminfo", ""),
            ScriptedStep::ok("command -v tailscale", ""),
            ScriptedStep::ok("uname -s", "Linux\nx86_64\n"),
        ]
    }

    #[test]
    fn serve_sends_the_key_over_stdin_when_it_starts_the_model() {
        // Guards the pairing: the command reads stdin, and the caller actually
        // supplies it. Either alone leaves the server without a credential.
        // Driving this through `serve_with_transport` rather than calling
        // `exec_with_stdin` directly is the point: a regression that stops the
        // real call site from passing the key (say, reverting to `None`) has to
        // fail this test, not just a check that never sees production code.
        let (root, paths) = temp_paths("serve-stdin");
        let transport = ScriptedTransport::new(full_serve_steps());

        serve_with_transport(
            &transport,
            &paths,
            "gpu-box.example-tailnet.ts.net",
            &request(),
        )
        .expect("scripted serve");

        let sent_key = transport.calls().iter().find_map(|call| match call {
            TransportCall::Exec {
                command,
                stdin: Some(key),
            } if command.contains("read -r ROCM_SERVE_API_KEY") => Some(key.clone()),
            _ => None,
        });
        let sent_key = sent_key.unwrap_or_else(|| {
            panic!(
                "the model-starting command must receive the key over stdin: {:?}",
                transport.calls()
            )
        });
        assert!(!sent_key.trim().is_empty(), "the key sent was blank");
        // The terminator is as load-bearing as the key. `IFS= read -r` returns
        // non-zero at EOF without one, so an unterminated payload short-circuits
        // the `&&` chain and the model never starts — which is what the
        // container lane caught after the chain was tightened, and what no
        // assertion about the command string could see.
        assert!(
            sent_key.ends_with('\n'),
            "the key must arrive as a complete line or `read` fails and nothing \
             serves; got {sent_key:?}"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    /// A `services list --json` reply holding one record on the session's port,
    /// in the given lifecycle state.
    fn listing_with_status(status: &str) -> String {
        format!(
            r#"[{{"service_id":"svc-1","engine":"vllm","model_ref":"m","canonical_model_id":"m",
               "host":"127.0.0.1","port":11434,"endpoint_url":"http://127.0.0.1:11434/v1",
               "mode":"managed","status":"{status}","supervisor_pid":1,
               "manifest_path":"/a","log_path":"/b","engine_state_path":"/c",
               "created_at_unix_ms":100}}]"#
        )
    }

    /// Whether the transport was ever asked to declare a forward.
    fn issued_a_publish(transport: &ScriptedTransport) -> bool {
        transport.calls().iter().any(|call| {
            matches!(call, TransportCall::Exec { command, .. } if command.contains("serve --bg"))
        })
    }

    #[test]
    fn a_machine_whose_registry_command_failed_is_still_asked_about_publishing() {
        // Reached-but-failed is not never-reached. The endpoint question is
        // still answerable, and reporting "the machine could not be asked" for a
        // machine that just answered sends the user to check the network.
        let transport = ScriptedTransport::new(vec![
            ScriptedStep::fails("services list --json", 127, "rocm: command not found"),
            ScriptedStep::ok("tailscale serve status", PUBLISHED_FIXTURE),
        ]);

        let observed = observe(&transport, &sample_record());
        assert_eq!(observed.server, ServerHealth::Error);
        assert_eq!(
            observed.publish,
            publish::PublishObservation::Known(publish::PublishState::Published),
            "a reachable machine's publishing state is a fact we can read"
        );

        let rendered = render_status(&render_paths(), &[(sample_record(), observed)]);
        assert!(rendered.contains("endpoint published: yes"), "{rendered}");
        assert!(
            !rendered.contains("could not be asked"),
            "a machine that answered must not be reported as unreachable:\n{rendered}"
        );
    }

    #[test]
    fn attach_refuses_to_republish_over_a_model_that_is_not_running() {
        // The refusal is the whole value of `attach`: re-declaring a forward is
        // cheap, so without this guard a dead session is handed an endpoint that
        // answers with connection refused — which reads as "up" to everything
        // that only checks whether the port is published.
        let transport = ScriptedTransport::new(vec![
            ScriptedStep::ok("services list --json", &listing_with_status("failed")),
            ScriptedStep::ok("tailscale serve status", "{}"),
        ]);

        let error = attach_with_transport(&transport, &sample_record())
            .expect_err("a dead model server must not be re-published over");
        let rendered = format!("{error:#}");
        assert!(rendered.contains("nothing behind it"), "{rendered}");
        assert!(
            !issued_a_publish(&transport),
            "a refused attach must not have declared a forward: {:?}",
            transport.calls()
        );
    }

    #[test]
    fn attach_republishes_a_healthy_session_whose_endpoint_went_missing() {
        // The other half: this is the case `status` sends the user here for, so
        // it has to actually issue the publish rather than only decline to fail.
        let transport = ScriptedTransport::new(vec![
            ScriptedStep::ok("services list --json", &listing_with_status("ready")),
            ScriptedStep::ok("tailscale serve status", "{}"),
            ScriptedStep::ok("serve --bg", ""),
        ]);
        // `publish` confirms by reading the state back, so the status step has to
        // answer "absent" first and "ours" afterwards. One scripted reply cannot
        // do both, so the confirmation is driven by a second transport below.
        let error = attach_with_transport(&transport, &sample_record())
            .expect_err("an unconfirmed publish is an error");
        assert!(
            format!("{error:#}").contains("does not report it as active"),
            "{error:#}"
        );
        assert!(
            issued_a_publish(&transport),
            "a healthy session must be re-published: {:?}",
            transport.calls()
        );

        let confirming = ScriptedTransport::new(vec![
            ScriptedStep::ok("services list --json", &listing_with_status("ready")),
            ScriptedStep::ok("tailscale serve status", PUBLISHED_FIXTURE),
            ScriptedStep::ok("serve --bg", ""),
        ]);
        attach_with_transport(&confirming, &sample_record()).expect("a confirmed re-publish");
    }

    /// Put a session and its credential on disk, the way `serve` leaves them.
    fn seeded_session(paths: &AppPaths) -> RemoteSessionRecord {
        let record = sample_record();
        session::store_key(paths, &record.session_id, "the-key").unwrap();
        record.write(paths).unwrap();
        record
    }

    #[test]
    fn a_stop_that_cannot_stop_the_model_keeps_the_session_and_says_why() {
        // Two properties in one flow, because they are the same decision: the
        // record is the only thing on this machine that knows the model's id and
        // where it runs, so dropping it while the model is still up leaves a GPU
        // held by something the user can no longer name. And the error has to
        // carry the remote's own words, or "could not be stopped" reads the same
        // whether the machine refused, the service was gone, or ssh never landed.
        //
        // The endpoint is already absent here — withdrawn out of band, or never
        // re-declared after a reboot — so `withdraw` succeeds early and the model
        // stop is the only thing that can fail. That isolates the branch under
        // test from the withdraw refusal above it.
        let (root, paths) = temp_paths("stop-refused");
        let record = seeded_session(&paths);
        let transport = ScriptedTransport::new(vec![
            ScriptedStep::ok("tailscale serve status", "{}"),
            ScriptedStep::fails("services stop", 3, "engine still shutting down"),
        ]);

        let error = stop_with_transport(&transport, &paths, &record, false)
            .expect_err("an unconfirmed model stop must not drop the session");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("engine still shutting down"),
            "{rendered}"
        );
        assert!(rendered.contains("exit 3"), "{rendered}");
        assert!(rendered.contains("--force"), "{rendered}");

        assert!(
            session::resolve(&paths, &record.session_id).is_ok(),
            "the session must still be listed so the user can retry"
        );
        assert!(
            session::key_path(&paths, &record.session_id).exists(),
            "the credential must outlive a failed teardown: it is the only copy"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_forced_stop_forgets_a_session_the_machine_will_not_confirm() {
        // `--force` is for a machine that is gone for good. It must drop the
        // record even when neither half could be confirmed — that is its purpose
        // — and the withdraw step here is the one that fails, which is the worse
        // of the two to forget.
        let (root, paths) = temp_paths("stop-forced");
        let record = seeded_session(&paths);
        let transport = ScriptedTransport::new(vec![
            ScriptedStep::fails("tailscale serve status", 1, "tailscaled not running"),
            ScriptedStep::fails("services stop", 1, "no such service"),
        ]);

        stop_with_transport(&transport, &paths, &record, true).expect("a forced stop must succeed");

        assert!(
            session::resolve(&paths, &record.session_id).is_err(),
            "a forced stop must forget the session locally"
        );
        assert!(
            !session::key_path(&paths, &record.session_id).exists(),
            "a forgotten session must not leave its credential behind"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_clean_stop_removes_the_session_and_its_credential() {
        let (root, paths) = temp_paths("stop-clean");
        let record = seeded_session(&paths);
        let transport = ScriptedTransport::new(vec![
            ScriptedStep::ok("tailscale serve status", "{}"),
            ScriptedStep::ok("services stop", ""),
        ]);

        stop_with_transport(&transport, &paths, &record, false).expect("a confirmed teardown");

        assert!(
            session::resolve(&paths, &record.session_id).is_err(),
            "a confirmed teardown must drop the session"
        );
        assert!(
            !session::key_path(&paths, &record.session_id).exists(),
            "a dropped session must not leave its credential behind"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_withdrawal_the_remote_will_not_confirm_keeps_the_record() {
        // `withdraw` reads the state back, and the machine still reports our
        // forward. The record is the only thing on this machine that can find
        // that endpoint again, so it has to survive — this is the failure the
        // whole teardown path is shaped around.
        let (root, paths) = temp_paths("stop-unconfirmed");
        let record = seeded_session(&paths);
        let transport = ScriptedTransport::new(vec![
            ScriptedStep::ok("tailscale serve status", PUBLISHED_FIXTURE),
            ScriptedStep::ok("tailscale serve --tcp=8000 off", ""),
            ScriptedStep::ok("services stop", ""),
        ]);

        let error = stop_with_transport(&transport, &paths, &record, false)
            .expect_err("a withdrawal the remote does not confirm is not a teardown");
        assert!(
            format!("{error:#}").contains("still reports port 8000 as published"),
            "{error:#}"
        );
        assert!(
            session::resolve(&paths, &record.session_id).is_ok(),
            "an unconfirmed withdrawal must keep the record: it is the only pointer to a live endpoint"
        );
        assert!(
            session::key_path(&paths, &record.session_id).exists(),
            "and the credential with it — the endpoint may still be answering"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_stop_failure_names_the_machines_answer_or_the_reason_it_never_answered() {
        // The two shapes this module keeps apart everywhere else.
        let answered = describe_stop_failure(&Ok(transport::RemoteOutcome {
            success: false,
            code: Some(2),
            stdout: String::new(),
            stderr: "  no such service  ".to_owned(),
        }))
        .expect("a non-zero exit is a failure");
        assert!(answered.contains("exit 2"), "{answered}");
        assert!(answered.contains("no such service"), "{answered}");

        let silent = describe_stop_failure(&Ok(transport::RemoteOutcome {
            success: false,
            code: None,
            stdout: String::new(),
            stderr: String::new(),
        }))
        .expect("a signalled exit is a failure");
        assert!(silent.contains("signal"), "{silent}");

        let unreachable = describe_stop_failure(&Err(anyhow::anyhow!("could not reach gpu-box")))
            .expect("an unreachable machine is a failure");
        assert!(
            unreachable.contains("could not reach gpu-box"),
            "{unreachable}"
        );

        assert!(
            describe_stop_failure(&Ok(transport::RemoteOutcome {
                success: true,
                code: Some(0),
                stdout: String::new(),
                stderr: String::new(),
            }))
            .is_none(),
            "a successful stop is not a failure"
        );
    }
}
