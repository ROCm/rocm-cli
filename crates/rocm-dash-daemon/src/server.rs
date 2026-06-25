// Copyright Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: MIT

//! Listener + per-client task loop + runner-to-clients broadcast.

// On Windows the listener/handler are #[cfg(unix)] stubs, so these imports are
// only used on unix; silence the resulting unused-import noise there.
#![cfg_attr(windows, allow(unused_imports))]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, anyhow};
use rocm_dash_core::protocol::{Command, Event, PROTOCOL_VERSION};
use tokio::io::{AsyncWriteExt, BufReader};
use tokio::sync::broadcast;
use tracing::{info, warn};

use crate::DAEMON_VERSION;
use crate::bench_ring::BenchRing;
use crate::persist::SessionWriter;
use crate::runner::{self, RunnerOptions};
use crate::snapshot_ring::SnapshotRing;
use crate::transport::{read_line, write_line};

/// Broadcast capacity. A slow client that lags more than this many ticks will
/// receive `RecvError::Lagged` and skip ahead (no crash).
#[cfg(unix)]
const BROADCAST_CAP: usize = 64;

/// Snapshot history kept for late-joining clients (~30s at 1Hz).
#[cfg(unix)]
const RING_CAP: usize = 30;

/// Benchmark-row history kept for late-joining clients (matches TUI `BENCH_CAP`).
#[cfg(unix)]
const BENCH_RING_CAP: usize = 200;

/// Start the daemon on `listen` (e.g. `unix:/run/user/1000/rocmdashd.sock`).
///
/// Access control is enforced by filesystem permissions: the Unix socket is
/// created with mode `0o600` so only the owning user can connect. The parent
/// directory is created if it does not exist.
pub async fn run(listen: &str, opts: RunnerOptions) -> anyhow::Result<()> {
    let (scheme, target) = listen
        .split_once(':')
        .ok_or_else(|| anyhow!("listen must be `unix:/path` or `tcp:host:port`"))?;

    match scheme {
        "unix" => run_unix(PathBuf::from(target), opts).await,
        "tcp" => Err(anyhow!("tcp listener not implemented in scaffold")),
        other => Err(anyhow!("unknown listen scheme: {other}")),
    }
}

#[cfg(unix)]
async fn run_unix(path: PathBuf, opts: RunnerOptions) -> anyhow::Result<()> {
    use tokio::net::UnixListener;

    if let Some(parent) = path.parent() {
        // Skip hardening if parent is an empty path (relative socket target
        // such as `unix:foo.sock`). Operating on `""` would target the CWD.
        if !parent.as_os_str().is_empty() {
            // Create the parent directory with mode 0o700 so other users cannot
            // traverse into it. DirBuilder::mode only applies to directories it
            // creates; pre-existing directories keep their current permissions,
            // so we explicitly set_permissions afterwards to tighten them.
            use std::os::unix::fs::DirBuilderExt;
            use std::os::unix::fs::PermissionsExt;
            std::fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(parent)
                .with_context(|| format!("creating socket directory {}", parent.display()))?;
            std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700)).with_context(
                || {
                    format!(
                        "restricting socket directory {} to mode 0700 — \
                         check that the directory is owned by the current user \
                         (EPERM means it is owned by another user or root)",
                        parent.display()
                    )
                },
            )?;
        }
    }
    if path.exists() {
        std::fs::remove_file(&path)
            .with_context(|| format!("removing stale socket {}", path.display()))?;
    }
    let listener =
        UnixListener::bind(&path).with_context(|| format!("binding {}", path.display()))?;
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("setting permissions on {}", path.display()))?;
    }
    info!(socket = %path.display(), "listening");

    let (snap_tx, _) = broadcast::channel::<Event>(BROADCAST_CAP);
    let ring = Arc::new(Mutex::new(SnapshotRing::new(RING_CAP)));
    let bench_ring = Arc::new(Mutex::new(BenchRing::new(BENCH_RING_CAP)));

    // Optional session writer for --persist-dir / `daemon.persist_dir`.
    let persist = match &opts.persist_dir {
        Some(dir) => match SessionWriter::new(dir) {
            Ok(w) => {
                info!(file = %w.path().display(), "session persistence enabled");
                Some(Arc::new(Mutex::new(w)))
            }
            Err(e) => {
                warn!(dir = %dir.display(), error = %e, "could not open session file; persistence disabled");
                None
            }
        },
        None => None,
    };

    // Runner task — produces snapshots and (optionally) bench rows.
    let runner_tx = snap_tx.clone();
    let runner_ring = ring.clone();
    let runner_bench_ring = bench_ring.clone();
    let runner_persist = persist.clone();
    let runner_opts = opts.clone();
    tokio::spawn(async move {
        runner::run_loop(
            None,
            runner_tx,
            runner_ring,
            runner_bench_ring,
            runner_persist,
            runner_opts,
        )
        .await;
    });

    loop {
        let (stream, _addr) = listener.accept().await?;
        let rx = snap_tx.subscribe();
        let client_ring = ring.clone();
        let client_bench_ring = bench_ring.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_client(stream, rx, client_ring, client_bench_ring).await {
                warn!(error = %e, "client task ended with error");
            }
        });
    }
}

#[cfg(windows)]
async fn run_unix(_path: PathBuf, _opts: RunnerOptions) -> anyhow::Result<()> {
    Err(anyhow!(
        "rocm-dash daemon requires Unix domain sockets; not supported on Windows yet"
    ))
}

#[cfg(unix)]
async fn handle_client(
    stream: tokio::net::UnixStream,
    mut rx: broadcast::Receiver<Event>,
    ring: Arc<Mutex<SnapshotRing>>,
    bench_ring: Arc<Mutex<BenchRing>>,
) -> anyhow::Result<()> {
    let (rd, mut wr) = stream.into_split();
    let mut rd = BufReader::new(rd);

    let welcome = Event::Welcome {
        protocol_version: PROTOCOL_VERSION,
        daemon_version: DAEMON_VERSION.into(),
        host: hostname().unwrap_or_else(|| "unknown".into()),
    };
    write_line(&mut wr, &welcome).await?;

    let mut subscribed = false;

    loop {
        tokio::select! {
            biased;

            // Inbound command from client.
            maybe_cmd = read_line::<_, Command>(&mut rd) => {
                let Some(cmd) = maybe_cmd? else { break };
                match cmd {
                    Command::Hello { client, protocol_version, .. } => {
                        info!(client = %client, protocol_version, "client hello");
                    }
                    Command::Subscribe => {
                        subscribed = true;
                        // Hydrate the late-joining client with whatever history we have.
                        let backlog = ring.lock().map(|r| r.snapshot()).unwrap_or_default();
                        let n = backlog.len();
                        for snap in backlog {
                            if write_line(&mut wr, &Event::Snapshot(snap)).await.is_err() {
                                break;
                            }
                        }
                        let bench_backlog = bench_ring
                            .lock()
                            .map(|r| r.snapshot())
                            .unwrap_or_default();
                        let bn = bench_backlog.len();
                        if !bench_backlog.is_empty() {
                            let _ = write_line(
                                &mut wr,
                                &Event::BenchmarkRowsAppended {
                                    rows: bench_backlog,
                                },
                            )
                            .await;
                        }
                        info!(replayed = n, bench_replayed = bn, "client subscribed");
                    }
                    Command::RequestSnapshot | Command::RescanInstances => {
                        // The runner pushes snapshots regularly; nothing to do here yet.
                    }
                    Command::Pause | Command::Resume => {
                        info!(?cmd, "pause/resume — runner does not honor yet");
                    }
                    Command::Goodbye => {
                        let _ = write_line(&mut wr, &Event::Bye).await;
                        break;
                    }
                }
            }

            // Outbound event from runner broadcast.
            recv = rx.recv() => {
                match recv {
                    Ok(ev) if subscribed => {
                        if write_line(&mut wr, &ev).await.is_err() {
                            break;
                        }
                    }
                    Ok(_) => { /* not subscribed yet — drop */ }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!(skipped = n, "client lagged broadcast");
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        }
    }

    let _ = wr.shutdown().await;
    Ok(())
}

#[cfg(unix)]
fn hostname() -> Option<String> {
    std::env::var("HOSTNAME").ok().or_else(|| {
        std::fs::read_to_string("/proc/sys/kernel/hostname")
            .ok()
            .map(|s| s.trim().to_string())
    })
}
