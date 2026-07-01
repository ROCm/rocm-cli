// Copyright © Advanced Micro Devices, Inc., or its affiliates.
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

/// Ensure the socket's parent directory exists and is private (mode `0o700`).
///
/// Directory permissions are *defense-in-depth*: the socket itself is bound with
/// mode `0o600` (see `run_unix`), so only the owning user can ever connect. We
/// therefore **enforce** `0o700` only on a directory we just created — a
/// directory we own must be securable, and failing to do so is a real error.
///
/// For a directory that already existed and is owned by another user (the classic
/// case is a socket parented directly at `/tmp`, which is root-owned with mode
/// `1777`), `chmod` returns `EPERM`. Aborting there crashed the embedded
/// telemetry daemon for any user whose configured socket lived under `/tmp`.
/// Since the `0o600` socket already restricts access, we log a warning and carry
/// on instead of failing.
#[cfg(unix)]
fn prepare_socket_dir(parent: &std::path::Path) -> anyhow::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    use std::os::unix::fs::PermissionsExt;

    // Skip an empty parent (relative socket target such as `unix:foo.sock`).
    // Operating on `""` would target the CWD.
    if parent.as_os_str().is_empty() {
        return Ok(());
    }

    // Whether we are about to create the directory ourselves. `DirBuilder::mode`
    // only applies to directories it creates; a pre-existing directory keeps its
    // current permissions, which is why we tighten explicitly below.
    let created = !parent.exists();
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(parent)
        .with_context(|| format!("creating socket directory {}", parent.display()))?;

    if let Err(e) = std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700)) {
        if created {
            // We created it, so we own it: inability to secure it is a real error.
            return Err(e).with_context(|| {
                format!(
                    "restricting socket directory {} to mode 0700 — \
                     check that the directory is owned by the current user \
                     (EPERM means it is owned by another user or root)",
                    parent.display()
                )
            });
        }
        // Pre-existing directory we do not own (e.g. /tmp, mode 1777): the
        // 0o600 socket still keeps the endpoint private, so warn and continue.
        warn!(
            dir = %parent.display(),
            error = %e,
            "could not restrict socket directory to mode 0700; continuing because \
             the socket is created with mode 0600 (expected for shared directories \
             such as /tmp)"
        );
    }
    Ok(())
}

#[cfg(unix)]
async fn run_unix(path: PathBuf, opts: RunnerOptions) -> anyhow::Result<()> {
    use tokio::net::UnixListener;

    if let Some(parent) = path.parent() {
        prepare_socket_dir(parent)?;
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

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    use std::path::Path;

    use super::prepare_socket_dir;

    /// A freshly created nested parent must end up at mode 0700.
    #[test]
    fn creates_missing_parent_at_0700() {
        let tmp = tempfile::tempdir().unwrap();
        let parent = tmp.path().join("nested").join("telemetry");
        prepare_socket_dir(&parent).expect("should create and secure a new directory");
        let mode = std::fs::metadata(&parent).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700, "newly created socket dir must be private");
    }

    /// A pre-existing parent we own but with looser perms gets tightened to 0700.
    #[test]
    fn tightens_owned_preexisting_parent() {
        let tmp = tempfile::tempdir().unwrap();
        let parent = tmp.path().join("loose");
        std::fs::create_dir(&parent).unwrap();
        std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o755)).unwrap();
        prepare_socket_dir(&parent).expect("should tighten a directory we own");
        let mode = std::fs::metadata(&parent).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700);
    }

    /// An empty parent (relative socket target like `unix:foo.sock`) is a no-op.
    #[test]
    fn empty_parent_is_noop() {
        prepare_socket_dir(Path::new("")).expect("empty parent must be skipped");
    }

    /// Regression: a socket parented at a pre-existing directory owned by another
    /// user (e.g. `/tmp`, root-owned mode 1777) must NOT abort the daemon — the
    /// chmod EPERM is downgraded to a warning because the 0600 socket already
    /// protects the endpoint. Before the fix this returned an error and crashed
    /// the embedded telemetry daemon.
    #[test]
    fn preexisting_unowned_parent_does_not_abort() {
        let tmpdir = std::env::temp_dir();
        let Ok(meta) = std::fs::metadata(&tmpdir) else {
            return; // no temp dir to probe; nothing to assert
        };
        // Determine our effective uid by stat-ing a file we create ourselves.
        let probe = tmpdir.join(format!("rocmdashd-uidprobe-{}", std::process::id()));
        if std::fs::write(&probe, b"x").is_err() {
            return; // cannot write a probe; skip rather than guess our uid
        }
        let my_uid = std::fs::metadata(&probe).map(|m| m.uid()).ok();
        let _ = std::fs::remove_file(&probe);
        // Only meaningful when the temp dir is owned by someone else (so chmod is
        // guaranteed to EPERM). If we own it or are root, chmod would *succeed*
        // and mutate a shared directory — skip to avoid side effects.
        if my_uid != Some(meta.uid()) {
            prepare_socket_dir(&tmpdir)
                .expect("must not abort when it cannot chmod a pre-existing unowned parent");
        }
    }
}
