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

/// A directory we could not tighten to `0o700` is safe to keep using only if no
/// unprivileged user can tamper with entries inside it. The one such case is a
/// **root-owned, sticky** directory — the `/tmp` signature (mode `1777`): the
/// sticky bit means only the owner of an entry (us) may remove or rename it, so
/// our `0o600` socket cannot be replaced. Any other owner means a local,
/// unprivileged user controls the path (a squatting attack), which is not safe.
#[cfg(unix)]
const fn is_safe_shared_dir(uid: u32, mode: u32) -> bool {
    uid == 0 && (mode & 0o1000 != 0)
}

/// Ensure the socket's parent directory exists and is private (mode `0o700`).
///
/// Directory permissions are *defense-in-depth*: the socket itself is bound with
/// mode `0o600` (see `run_unix`), so only the owning user can ever connect.
/// `set_permissions` fails with `EPERM` only when we do not own the directory,
/// which is exactly what we key the policy on:
///
/// * **We own it** (chmod succeeds) — it is now `0o700`; continue.
/// * **A root-owned sticky directory** such as `/tmp` (chmod fails) — the sticky
///   bit stops other users removing our socket, so warn and continue. Aborting
///   here is what previously crashed the embedded telemetry daemon for any user
///   whose configured socket lived directly under `/tmp`.
/// * **Any other owner** (chmod fails) — another unprivileged user owns this
///   path and could unlink or replace the socket, so refuse rather than bind a
///   telemetry endpoint inside an attacker-controlled directory.
#[cfg(unix)]
fn prepare_socket_dir(parent: &std::path::Path) -> anyhow::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    use std::os::unix::fs::MetadataExt;
    use std::os::unix::fs::PermissionsExt;

    // Skip an empty parent (relative socket target such as `unix:foo.sock`).
    // Operating on `""` would target the CWD.
    if parent.as_os_str().is_empty() {
        return Ok(());
    }

    // `DirBuilder::mode` only applies to directories it creates; a pre-existing
    // directory keeps its current permissions, which is why we tighten
    // explicitly below.
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(parent)
        .with_context(|| format!("creating socket directory {}", parent.display()))?;

    if let Err(e) = std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700)) {
        // chmod only fails because we do not own the directory. Decide from the
        // directory's *actual* owner and mode — not from whether we think we
        // just created it, which is racy — whether it is nonetheless safe.
        let meta = std::fs::metadata(parent).with_context(|| {
            format!(
                "inspecting socket directory {} after it could not be secured",
                parent.display()
            )
        })?;
        if is_safe_shared_dir(meta.uid(), meta.mode()) {
            warn!(
                dir = %parent.display(),
                error = %e,
                "could not restrict socket directory to mode 0700; continuing because \
                 it is a root-owned sticky directory (e.g. /tmp) and the socket is \
                 created with mode 0600"
            );
        } else {
            return Err(e).with_context(|| {
                format!(
                    "restricting socket directory {} to mode 0700 — it is owned by \
                     another user, so a local user could replace the telemetry \
                     socket. Point the socket at a directory you own, for example \
                     under $XDG_RUNTIME_DIR or $HOME",
                    parent.display()
                )
            });
        }
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

    use super::{is_safe_shared_dir, prepare_socket_dir};

    /// The safe-to-continue predicate is the security-critical decision for a
    /// directory we cannot tighten, and the squatting case (a non-root user
    /// owning the parent) cannot be reproduced in a unit test without a second
    /// uid — so pin every owner/mode combination here.
    #[test]
    fn only_root_owned_sticky_dir_is_safe_shared() {
        assert!(is_safe_shared_dir(0, 0o1777), "/tmp signature is safe");
        assert!(is_safe_shared_dir(0, 0o1700), "root-owned + sticky is safe");
        assert!(
            !is_safe_shared_dir(0, 0o0777),
            "root-owned without the sticky bit is not safe: others can unlink our socket"
        );
        assert!(
            !is_safe_shared_dir(1000, 0o1777),
            "a non-root user owning the parent is a squatting risk, even with sticky set"
        );
        assert!(
            !is_safe_shared_dir(1000, 0o0700),
            "another unprivileged owner is never safe"
        );
    }

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

    /// Regression: a socket parented at a root-owned sticky directory (`/tmp`,
    /// mode 1777) must NOT abort the daemon — the chmod EPERM is downgraded to a
    /// warning because the sticky bit + 0600 socket protect the endpoint. Before
    /// the fix this returned an error and crashed the embedded telemetry daemon.
    ///
    /// This exercises the real fix on Linux CI (where `temp_dir()` is `/tmp`,
    /// root-owned, and the test runs unprivileged). It is skipped when we own the
    /// temp dir or it is not a root-owned sticky dir (e.g. a custom `TMPDIR`, some
    /// containers, or macOS) — there chmod would either succeed and mutate a
    /// shared directory, or the safe-shared predicate would legitimately refuse,
    /// neither of which is this regression. `is_safe_shared_dir` is unit-tested
    /// separately for the refuse path.
    #[test]
    fn preexisting_root_sticky_parent_does_not_abort() {
        let tmpdir = std::env::temp_dir();
        let Ok(meta) = std::fs::metadata(&tmpdir) else {
            return; // no temp dir to probe; nothing to assert
        };
        if !is_safe_shared_dir(meta.uid(), meta.mode()) {
            return; // not the /tmp signature; not this regression
        }
        // Determine our effective uid by stat-ing a file we create ourselves.
        let probe = tmpdir.join(format!("rocmdashd-uidprobe-{}", std::process::id()));
        if std::fs::write(&probe, b"x").is_err() {
            return; // cannot write a probe; skip rather than guess our uid
        }
        let my_uid = std::fs::metadata(&probe).map(|m| m.uid()).ok();
        let _ = std::fs::remove_file(&probe);
        // Only meaningful when we do not own the dir, so chmod is guaranteed to
        // EPERM and the warn-and-continue path is actually taken.
        if my_uid != Some(meta.uid()) {
            prepare_socket_dir(&tmpdir)
                .expect("must not abort on a root-owned sticky parent it cannot chmod (e.g. /tmp)");
        }
    }
}
