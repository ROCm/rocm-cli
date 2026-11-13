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

/// Benchmark-row history kept for late-joining clients; the same cap the
/// dashboard holds (see `BENCH_HISTORY_CAP`).
#[cfg(unix)]
const BENCH_RING_CAP: usize = rocm_dash_core::bench_schema::BENCH_HISTORY_CAP;

/// Start the daemon on `listen` (e.g. `unix:/run/user/1000/rocmdashd.sock`).
///
/// Access control is enforced by filesystem permissions: the Unix socket is
/// created with mode `0o600` so only the owning user can connect. The parent
/// directory is created if it does not exist.
///
/// One daemon owns a socket path at a time (see `SocketClaim`): this returns
/// [`SocketInUse`] at once, removing or replacing nothing at the socket path,
/// when another daemon holds the path's `<socket>.lock` or something already
/// answers at the path. A path that exists but is not a socket is refused and
/// left in place. A stale socket nobody answers on is replaced. While serving,
/// the daemon stops with [`SocketReplaced`] if its socket is removed or
/// replaced, so the path can be claimed again, and with an error if its
/// telemetry runner stops. When it returns or is aborted, the runner task and
/// the client tasks it started here are aborted with it; a task the runner
/// itself spawns (its background GPU detection) is not tracked.
pub async fn run(listen: &str, opts: RunnerOptions) -> anyhow::Result<()> {
    run_with_ready(listen, opts, || {}).await
}

/// [`run`], calling `ready` once the daemon has bound its socket and is about
/// to accept clients -- the moment it is serving rather than failing.
pub async fn run_with_ready(
    listen: &str,
    opts: RunnerOptions,
    ready: impl FnOnce() + Send,
) -> anyhow::Result<()> {
    let (scheme, target) = listen
        .split_once(':')
        .ok_or_else(|| anyhow!("listen must be `unix:/path` or `tcp:host:port`"))?;

    match scheme {
        "unix" => run_unix(PathBuf::from(target), opts, ready).await,
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

/// Why a socket directory this daemon could not tighten is refused.
///
/// Lifted out of [`prepare_socket_dir`] so the wording itself is reachable from
/// a test. The branch that emits it needs a directory owned by a *second* uid,
/// which a unit test cannot conjure -- the same limitation already recorded on
/// `only_root_owned_sticky_dir_is_safe_shared` -- so without this split the
/// message has no coverage at all.
#[cfg(unix)]
fn unsecurable_socket_dir_context(parent: &std::path::Path) -> String {
    format!(
        "restricting socket directory {} to mode 0700 - it is owned by \
         another user, so a local user could replace the telemetry \
         socket. Point the socket at a directory you own, for example \
         under $XDG_RUNTIME_DIR or $HOME",
        parent.display()
    )
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
            return Err(e).with_context(|| unsecurable_socket_dir_context(parent));
        }
    }
    Ok(())
}

/// Returned by [`run`] when another telemetry daemon already owns the socket
/// path, so this one did not bind and removed nothing.
///
/// "Owns" means it holds the path's ownership lock (see `SocketClaim`), or
/// -- for a daemon from a build that predates the lock -- that a connection to
/// the socket is answered. A caller that wants to take over once that daemon
/// is gone (the embedded daemon `rocm dash` keeps) retries on this error and
/// treats every other one as a failure; find it with
/// `anyhow::Error::downcast_ref::<SocketInUse>()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SocketInUse {
    /// The socket path that is already served.
    pub path: PathBuf,
    /// Who serves it, which says how soon it may be worth trying again.
    pub holder: SocketHolder,
}

/// Who holds a socket path a daemon found in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SocketHolder {
    /// Another daemon holds the path's ownership lock. It releases the path
    /// when it stops, however it stops.
    LockHolder,
    /// Something answers at the path without holding the lock: a daemon from
    /// a build that predates the lock, or another program. Nothing announces
    /// when it goes away.
    UnlockedListener,
}

impl std::fmt::Display for SocketInUse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.holder {
            SocketHolder::LockHolder => write!(
                f,
                "another telemetry daemon already serves {}",
                self.path.display()
            ),
            SocketHolder::UnlockedListener => write!(
                f,
                "a listener without the ownership lock (an older telemetry daemon or \
                 another program) already serves {}",
                self.path.display()
            ),
        }
    }
}

impl std::error::Error for SocketInUse {}

/// Returned by [`run`] when a serving daemon found its socket path removed or
/// replaced. It stopped and released the path, so a retry can claim it again
/// at once; this is not a failure to report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SocketReplaced {
    /// The socket path that no longer names this daemon's socket.
    pub path: PathBuf,
}

impl std::fmt::Display for SocketReplaced {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} no longer names this daemon's socket (it was removed or replaced); \
             stopping so the path can be claimed again",
            self.path.display()
        )
    }
}

impl std::error::Error for SocketReplaced {}

/// How long the stale-socket probe waits for an existing socket to accept a
/// connection and send its greeting. A listener that is alive but too busy
/// within this is treated as alive: replacing a live daemon is the failure
/// being prevented, and a wrong "in use" only delays a takeover until the next
/// attempt.
#[cfg(unix)]
const STALE_PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(500);

/// How often a serving daemon checks that its socket path still names the
/// socket it bound.
#[cfg(unix)]
const PATH_CHECK_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);

/// What the stale-socket probe found at an existing socket path.
#[cfg(unix)]
#[derive(Debug)]
enum ProbeVerdict {
    /// Something is listening: it answered, was too busy to answer in time
    /// (timed out), or its backlog is full (`WouldBlock` from a non-blocking
    /// connect). A unix-socket connect does not pend on Linux, so the last is
    /// how a busy listener shows up there.
    Answered,
    /// Nobody listens (`ECONNREFUSED`): the socket is left over.
    Stale,
    /// The path went away between the check and the connect.
    Vanished,
    /// Anything else: report it and leave the path alone.
    Failed(std::io::Error),
}

/// Classify a probe's outcome; `None` means it did not finish in time.
#[cfg(unix)]
fn probe_verdict(outcome: Option<std::io::Result<()>>) -> ProbeVerdict {
    use std::io::ErrorKind;

    match outcome {
        None | Some(Ok(())) => ProbeVerdict::Answered,
        Some(Err(e)) => match e.kind() {
            ErrorKind::WouldBlock => ProbeVerdict::Answered,
            ErrorKind::ConnectionRefused => ProbeVerdict::Stale,
            ErrorKind::NotFound => ProbeVerdict::Vanished,
            _ => ProbeVerdict::Failed(e),
        },
    }
}

/// Connect to an existing socket and, if it answers, read its first line
/// before hanging up. A daemon greets every client at once, so leaving before
/// the greeting would have it write to a closed connection and log an error
/// for every probe.
#[cfg(unix)]
async fn probe_socket(path: &std::path::Path) -> Option<std::io::Result<()>> {
    use tokio::io::AsyncBufReadExt;

    tokio::time::timeout(STALE_PROBE_TIMEOUT, async {
        let stream = tokio::net::UnixStream::connect(path).await?;
        let mut greeting = Vec::new();
        // What it says, or whether it says anything, does not matter here.
        let _ = BufReader::new(stream)
            .read_until(b'\n', &mut greeting)
            .await;
        Ok(())
    })
    .await
    .ok()
}

/// The lock file that says which daemon owns `socket`: the socket path with
/// `.lock` appended, in the same (private) directory.
#[cfg(unix)]
fn socket_lock_path(socket: &std::path::Path) -> PathBuf {
    let mut lock = socket.as_os_str().to_owned();
    lock.push(".lock");
    PathBuf::from(lock)
}

/// What the refusal of a path that is not a socket says about it, right after
/// the path. The dashboard's end-to-end check (dash-53 in `dash_steps.rs`)
/// matches this text; it cannot import it.
pub const NOT_A_SOCKET: &str = "exists and is not a socket";

/// Why an existing path that is not a socket is left alone.
#[cfg(unix)]
fn not_a_socket_refusal(path: &std::path::Path) -> String {
    format!(
        "{} {NOT_A_SOCKET}, so the telemetry daemon will not replace it. \
         Move or remove it, or point the dashboard at another socket path",
        path.display()
    )
}

/// One daemon's exclusive hold on a socket path, from before it binds until
/// it stops serving.
///
/// Two dashboards share one socket path, so two daemons can try to bind it.
/// Unlinking whatever is at the path and binding, as this daemon once did,
/// lets the second one silently orphan the first's live listener. A claim
/// makes the path single-owner instead:
///
/// * it holds an exclusive lock on [`socket_lock_path`] for the daemon's whole
///   life -- a second claimant gets [`SocketInUse`] and changes nothing, and
///   the OS releases the lock when the holder exits, however it exits, so a
///   crashed owner never wedges the path;
/// * holding the lock, it removes an existing socket only when a connection to
///   it is refused (nobody listening), so a live daemon from a build without
///   the lock is not replaced either, and it never removes a path that is not
///   a socket;
/// * on drop it unlinks the socket only if the path is still the very socket
///   it bound (same device and inode), then releases the lock.
///
/// The lock file is never deleted: removing it while held would let the next
/// claimant lock a fresh inode while this one still holds the old one.
///
/// This uses `std::fs::File::try_lock` directly rather than `rocm-core`'s
/// `FileLock`: that one cannot create the file with mode `0o600`, and this
/// crate deliberately has no production dependency on `rocm-core` (a new edge
/// would need review in `xtask`'s crate-edge check).
#[cfg(unix)]
#[derive(Debug)]
pub(crate) struct SocketClaim {
    path: PathBuf,
    /// `(device, inode)` of the socket this claim bound, once it has.
    bound: Option<(u64, u64)>,
    /// Held, never read: dropping it (after `Drop::drop`) releases the lock.
    _lock: std::fs::File,
}

#[cfg(unix)]
impl SocketClaim {
    /// Take ownership of `path`, clearing a stale socket left by a dead daemon.
    ///
    /// # Errors
    /// [`SocketInUse`] when another daemon owns the path; another error when
    /// the path is not a socket, or the lock or probe fails.
    pub(crate) async fn claim(path: &std::path::Path) -> anyhow::Result<Self> {
        use std::os::unix::fs::OpenOptionsExt;

        let lock_path = socket_lock_path(path);
        let lock = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(&lock_path)
            .with_context(|| format!("opening socket lock {}", lock_path.display()))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => {
                return Err(SocketInUse {
                    path: path.to_path_buf(),
                    holder: SocketHolder::LockHolder,
                }
                .into());
            }
            Err(std::fs::TryLockError::Error(e)) => {
                return Err(e)
                    .with_context(|| format!("locking socket lock {}", lock_path.display()));
            }
        }
        let claim = Self {
            path: path.to_path_buf(),
            bound: None,
            _lock: lock,
        };
        claim.clear_stale_socket().await?;
        Ok(claim)
    }

    /// With the lock held: leave the path empty, or explain why not.
    async fn clear_stale_socket(&self) -> anyhow::Result<()> {
        use std::os::unix::fs::FileTypeExt;

        let path = &self.path;
        let meta = match std::fs::symlink_metadata(path) {
            Ok(meta) => meta,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => {
                return Err(e).with_context(|| format!("inspecting {}", path.display()));
            }
        };
        if !meta.file_type().is_socket() {
            return Err(anyhow!(not_a_socket_refusal(path)));
        }
        match probe_verdict(probe_socket(path).await) {
            // We hold the lock, so whatever answers is not a daemon of this
            // build: an older one, or another program.
            ProbeVerdict::Answered => Err(SocketInUse {
                path: path.clone(),
                holder: SocketHolder::UnlockedListener,
            }
            .into()),
            ProbeVerdict::Stale => match std::fs::remove_file(path) {
                Ok(()) => Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(e) => {
                    Err(e).with_context(|| format!("removing stale socket {}", path.display()))
                }
            },
            ProbeVerdict::Vanished => Ok(()),
            ProbeVerdict::Failed(e) => {
                Err(e).with_context(|| format!("probing existing socket {}", path.display()))
            }
        }
    }

    /// Bind the claimed path, restricted to mode `0o600`.
    pub(crate) fn bind(&mut self) -> anyhow::Result<tokio::net::UnixListener> {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};

        let path = &self.path;
        let listener = match tokio::net::UnixListener::bind(path) {
            Ok(listener) => listener,
            // Something bound the path between the probe and here without
            // taking the lock (a daemon of an older build): it is in use.
            // Unless what appeared there is not a socket at all.
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
                use std::os::unix::fs::FileTypeExt;

                let is_socket = std::fs::symlink_metadata(path)
                    .map_or(true, |meta| meta.file_type().is_socket());
                if !is_socket {
                    return Err(anyhow!(not_a_socket_refusal(path)));
                }
                return Err(SocketInUse {
                    path: path.clone(),
                    holder: SocketHolder::UnlockedListener,
                }
                .into());
            }
            Err(e) => return Err(e).with_context(|| format!("binding {}", path.display())),
        };
        // Record the identity before the chmod, so drop still removes the
        // socket if the chmod fails. If this stat itself fails, the socket is
        // left behind; nobody answers on it once the listener is dropped, so
        // the next claim replaces it as stale.
        let meta = std::fs::symlink_metadata(path)
            .with_context(|| format!("inspecting bound socket {}", path.display()))?;
        self.bound = Some((meta.dev(), meta.ino()));
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("setting permissions on {}", path.display()))?;
        Ok(listener)
    }
}

/// What a claim's socket path names now.
#[cfg(unix)]
#[derive(Debug)]
enum PathState {
    /// The socket this claim bound.
    Ours,
    /// Nothing, or something else: the socket was removed or replaced.
    Gone,
    /// The path could not be checked (a permission or I/O error); that says
    /// nothing about whose socket it is.
    Unknown(std::io::Error),
}

#[cfg(unix)]
impl SocketClaim {
    /// What the path names now, compared with the socket this claim bound.
    fn path_state(&self) -> PathState {
        use std::os::unix::fs::MetadataExt;

        let Some(bound) = self.bound else {
            return PathState::Gone;
        };
        match std::fs::symlink_metadata(&self.path) {
            Ok(meta) if (meta.dev(), meta.ino()) == bound => PathState::Ours,
            Ok(_) => PathState::Gone,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => PathState::Gone,
            Err(e) => PathState::Unknown(e),
        }
    }
}

#[cfg(unix)]
impl Drop for SocketClaim {
    fn drop(&mut self) {
        if matches!(self.path_state(), PathState::Ours) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// How many path checks in a row may fail to read the path before a serving
/// daemon gives it up as lost.
#[cfg(unix)]
const PATH_CHECK_UNKNOWN_LIMIT: u32 = 5;

/// The unreadable path checks in a row after one more check, or `None` once
/// they reach [`PATH_CHECK_UNKNOWN_LIMIT`]. A readable check starts over.
#[cfg(unix)]
const fn unreadable_streak(streak: u32, readable: bool) -> Option<u32> {
    if readable {
        return Some(0);
    }
    if streak + 1 >= PATH_CHECK_UNKNOWN_LIMIT {
        None
    } else {
        Some(streak + 1)
    }
}

/// The parts of a daemon its telemetry runner is started with.
#[cfg(unix)]
struct RunnerParts {
    tx: broadcast::Sender<Event>,
    ring: Arc<Mutex<SnapshotRing>>,
    bench_ring: Arc<Mutex<BenchRing>>,
    persist: Option<Arc<Mutex<SessionWriter>>>,
    opts: RunnerOptions,
}

/// The production runner: [`runner::run_loop`], which runs until aborted.
#[cfg(unix)]
fn start_run_loop(parts: RunnerParts) -> futures::future::BoxFuture<'static, ()> {
    Box::pin(runner::run_loop(
        None,
        parts.tx,
        parts.ring,
        parts.bench_ring,
        parts.persist,
        parts.opts,
    ))
}

/// Starts a daemon's telemetry runner.
#[cfg(unix)]
type StartRunner = Box<dyn FnOnce(RunnerParts) -> futures::future::BoxFuture<'static, ()> + Send>;

/// What can be swapped in a serving daemon for a test.
#[cfg(unix)]
struct ServeSeams {
    start_runner: StartRunner,
    /// Set to the number of client tasks still held after every event the
    /// accept loop handles.
    client_tasks: Option<Arc<std::sync::atomic::AtomicUsize>>,
    /// How often the socket path is checked.
    path_check_interval: std::time::Duration,
}

#[cfg(unix)]
impl ServeSeams {
    fn production() -> Self {
        Self {
            start_runner: Box::new(start_run_loop),
            client_tasks: None,
            path_check_interval: PATH_CHECK_INTERVAL,
        }
    }
}

/// Why a daemon stopped because its runner did.
#[cfg(unix)]
fn runner_stopped_error(outcome: Result<(), tokio::task::JoinError>) -> anyhow::Error {
    match outcome {
        Ok(()) => anyhow!("the telemetry runner stopped; restarting the daemon"),
        Err(e) if e.is_panic() => anyhow!("the telemetry runner panicked: {e}"),
        Err(e) => anyhow!("the telemetry runner was cancelled: {e}"),
    }
}

#[cfg(unix)]
async fn run_unix(
    path: PathBuf,
    opts: RunnerOptions,
    ready: impl FnOnce() + Send,
) -> anyhow::Result<()> {
    serve_unix(path, opts, ready, ServeSeams::production()).await
}

#[cfg(unix)]
async fn serve_unix(
    path: PathBuf,
    opts: RunnerOptions,
    ready: impl FnOnce() + Send,
    seams: ServeSeams,
) -> anyhow::Result<()> {
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
    // The lock file lives in the socket's directory, so that directory has to
    // exist before the claim. It is tightened (or refused) only after the claim
    // succeeds: a dashboard that is not the owner retries several times a
    // second, and must not repeat `prepare_socket_dir`'s chmod -- or its
    // warning for a shared sticky directory -- on every attempt.
    if let Some(parent) = parent
        && !parent.is_dir()
    {
        prepare_socket_dir(parent)?;
    }
    let mut claim = SocketClaim::claim(&path).await?;
    if let Some(parent) = parent {
        prepare_socket_dir(parent)?;
    }
    // `claim` is dropped when this function returns or the task running it
    // is aborted -- that is, when this daemon stops serving -- and removes the
    // socket only if it is still the one bound here.
    let listener = claim.bind()?;
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

    // The runner and every client task live in these sets, which abort what
    // they hold when dropped -- when this function returns or the task running
    // it is aborted. Abort takes effect at each task's next await point.
    let mut runner_task = tokio::task::JoinSet::new();
    let mut clients = tokio::task::JoinSet::new();

    // Runner task — produces snapshots and (optionally) bench rows.
    runner_task.spawn((seams.start_runner)(RunnerParts {
        tx: snap_tx.clone(),
        ring: ring.clone(),
        bench_ring: bench_ring.clone(),
        persist: persist.clone(),
        opts: opts.clone(),
    }));
    ready();

    let mut path_check = tokio::time::interval(seams.path_check_interval);
    path_check.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut unknown_checks = 0;
    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let (stream, _addr) = accepted?;
                let rx = snap_tx.subscribe();
                let client_ring = ring.clone();
                let client_bench_ring = bench_ring.clone();
                clients.spawn(async move {
                    if let Err(e) = handle_client(stream, rx, client_ring, client_bench_ring).await {
                        warn!(error = %e, "client task ended with error");
                    }
                });
            }
            // A daemon whose runner stopped would greet clients and never
            // send them anything; stop, so the caller can start it again.
            Some(outcome) = runner_task.join_next() => {
                return Err(runner_stopped_error(outcome));
            }
            // Reap finished client tasks so the set does not grow.
            Some(outcome) = clients.join_next() => {
                if let Err(e) = outcome
                    && e.is_panic()
                {
                    warn!(error = %e, "client task panicked");
                }
            }
            _ = path_check.tick() => {
                let unreadable = match claim.path_state() {
                    PathState::Ours => None,
                    // Serving an inode nobody can reach would hold the lock,
                    // and so the path, forever.
                    PathState::Gone => return Err(SocketReplaced { path }.into()),
                    PathState::Unknown(e) => {
                        warn!(socket = %path.display(), error = %e, "could not check the socket path");
                        Some(e)
                    }
                };
                let readable = unreadable.is_none();
                match (unreadable_streak(unknown_checks, readable), unreadable) {
                    (Some(streak), _) => unknown_checks = streak,
                    (None, e) => {
                        return Err(anyhow!(
                            "checking socket {} failed {PATH_CHECK_UNKNOWN_LIMIT} times in a \
                             row: {}",
                            path.display(),
                            e.map_or_else(String::new, |e| e.to_string())
                        ));
                    }
                }
            }
        }
        if let Some(count) = &seams.client_tasks {
            count.store(clients.len(), std::sync::atomic::Ordering::SeqCst);
        }
    }
}

#[cfg(windows)]
// Mirrors the signature of the `cfg(unix)` arm, which callers `.await`;
// dropping `async` here would not compile on Windows.
#[allow(clippy::unused_async)]
async fn run_unix(
    _path: PathBuf,
    _opts: RunnerOptions,
    _ready: impl FnOnce() + Send,
) -> anyhow::Result<()> {
    Err(anyhow!(
        "rocm-dash daemon requires Unix domain sockets; not supported on Windows yet"
    ))
}

/// Logged when a client sends `Pause` or `Resume`, which the daemon accepts
/// but the runner does not act on yet.
///
/// ROCMAI-543: this is not TTY output, but it is not exempt either. It is
/// routed by `apps/rocm/src/logging.rs` (`rocm_dash_daemon=info`) into the
/// daily-rotated client log file, a plain UTF-8 writer that -- unlike the
/// spinner ellipses this sweep deliberately leaves alone -- is never gated on
/// `is_terminal()`. A Windows reader assuming the OEM code page would render a
/// UTF-8 em dash here as mojibake, which is exactly the reported bug.
///
/// Lifted out of [`handle_client`] so the wording is reachable from a test:
/// the branch that emits it sits inside a `select!` loop that a unit test
/// cannot drive without a live socket peer and a runner. This mirrors the
/// split already used for [`unsecurable_socket_dir_context`].
#[cfg(unix)]
const PAUSE_RESUME_UNSUPPORTED: &str = "pause/resume - runner does not honor yet";

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
                        info!(?cmd, "{PAUSE_RESUME_UNSUPPORTED}");
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

    use super::{
        PAUSE_RESUME_UNSUPPORTED, is_safe_shared_dir, prepare_socket_dir,
        unsecurable_socket_dir_context,
    };

    /// ROCMAI-543: the pause/resume notice is written to the daily-rotated
    /// client log file, which is a plain UTF-8 writer and is never gated on
    /// `is_terminal()`, so the TTY-only exemption this sweep grants the spinner
    /// does not apply to it. Pin both the ASCII property and the exact wording.
    #[test]
    fn the_pause_resume_notice_is_ascii() {
        assert!(
            PAUSE_RESUME_UNSUPPORTED.is_ascii(),
            "the pause/resume log notice must be ASCII-only (ROCMAI-543): \
             {PAUSE_RESUME_UNSUPPORTED}"
        );
        assert_eq!(
            PAUSE_RESUME_UNSUPPORTED, "pause/resume - runner does not honor yet",
            "the pause/resume log notice changed; keep it ASCII (ROCMAI-543)"
        );
    }

    /// ROCMAI-543: the refusal is an error the CLI prints on a redirectable
    /// stream, so it must be plain ASCII -- a Windows console on the OEM code
    /// page renders a UTF-8 em dash as mojibake like `ΓÇö`. It also has to keep
    /// naming the remediation, which is the only thing in it a user can act on.
    #[test]
    fn the_unsecurable_socket_dir_refusal_is_ascii_and_names_the_remedy() {
        let message = unsecurable_socket_dir_context(Path::new("/shared/telemetry"));
        assert!(
            message.is_ascii(),
            "the socket-directory refusal must be ASCII-only (ROCMAI-543): {message}"
        );
        assert!(
            message.contains("/shared/telemetry"),
            "the refusal must name the directory it is about: {message}"
        );
        assert!(
            message.contains("$XDG_RUNTIME_DIR") && message.contains("$HOME"),
            "the refusal must keep naming somewhere the user can point the socket instead: \
             {message}"
        );
    }

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
            eprintln!("skip: no temp dir to probe; nothing to assert");
            return;
        };
        if !is_safe_shared_dir(meta.uid(), meta.mode()) {
            eprintln!("skip: temp dir is not the root-owned-sticky /tmp signature");
            return;
        }
        // Determine our effective uid by stat-ing a file we create ourselves.
        let probe = tmpdir.join(format!("rocmdashd-uidprobe-{}", std::process::id()));
        if std::fs::write(&probe, b"x").is_err() {
            eprintln!("skip: cannot write a probe file; cannot determine our uid");
            return;
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

    // ── Socket ownership (ROCMAI-456) ─────────────────────────────────
    //
    // Two dashboards share one socket path. These pin what a second daemon
    // does at that path in each state it can find it in: nothing there, a
    // stale socket, a live owner, a live daemon without the lock, and a path
    // that is not a socket at all.

    use std::time::Duration;

    use rocm_dash_core::protocol::Event;
    use tokio::io::BufReader;

    use super::{
        ProbeVerdict, ServeSeams, SocketClaim, SocketHolder, SocketInUse, SocketReplaced,
        not_a_socket_refusal, probe_verdict, serve_unix, socket_lock_path,
    };
    use crate::runner::RunnerOptions;
    use crate::transport::read_line;

    /// Long enough for a loaded CI host; every wait below polls and returns
    /// as soon as its condition holds.
    const DEADLINE: Duration = Duration::from_secs(10);

    fn start_daemon(path: &Path) -> tokio::task::JoinHandle<anyhow::Result<()>> {
        let listen = format!("unix:{}", path.display());
        tokio::spawn(async move { super::run(&listen, RunnerOptions::default()).await })
    }

    /// Run a daemon that is expected to give up, and fail the test if it is
    /// still serving at the deadline instead.
    async fn run_expecting_refusal(path: &Path) -> anyhow::Error {
        let listen = format!("unix:{}", path.display());
        match tokio::time::timeout(DEADLINE, super::run(&listen, RunnerOptions::default())).await {
            Ok(Ok(())) => panic!("the daemon at {} stopped without an error", path.display()),
            Ok(Err(e)) => e,
            Err(elapsed) => panic!(
                "the daemon at {} was still serving after {DEADLINE:?} ({elapsed}): it took \
                 the path instead of refusing it",
                path.display()
            ),
        }
    }

    /// Connect to `path` and read the greeting a daemon sends every client.
    async fn welcome_from(path: &Path) -> std::io::Result<tokio::net::UnixStream> {
        let mut stream = tokio::net::UnixStream::connect(path).await?;
        let mut reader = BufReader::new(&mut stream);
        let first = tokio::time::timeout(DEADLINE, read_line::<_, Event>(&mut reader))
            .await
            .map_err(|_| std::io::Error::other("no greeting before the deadline"))??;
        match first {
            Some(Event::Welcome { .. }) => Ok(stream),
            other => Err(std::io::Error::other(format!(
                "expected a Welcome, got {other:?}"
            ))),
        }
    }

    async fn wait_until_serving(path: &Path) -> tokio::net::UnixStream {
        let deadline = tokio::time::Instant::now() + DEADLINE;
        loop {
            match welcome_from(path).await {
                Ok(stream) => return stream,
                Err(e) => assert!(
                    tokio::time::Instant::now() < deadline,
                    "no daemon greeted a client at {} within {DEADLINE:?}: {e}",
                    path.display()
                ),
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    fn inode(path: &Path) -> u64 {
        std::fs::symlink_metadata(path)
            .unwrap_or_else(|e| panic!("{} should exist: {e}", path.display()))
            .ino()
    }

    /// A socket file with nobody listening, as a killed daemon leaves behind.
    fn stale_socket(path: &Path) {
        drop(std::os::unix::net::UnixListener::bind(path).expect("bind the soon-stale socket"));
        assert!(
            std::os::unix::net::UnixStream::connect(path).is_err(),
            "the fixture must be stale: nothing may answer"
        );
    }

    /// Keep the inode at `path` allocated by giving it a second name, so that
    /// once `path` is unlinked a socket bound there next cannot be handed the
    /// same inode number. Without this an `inode(..) != before` check fails
    /// whenever the filesystem reuses the number it just freed, as ext4 often
    /// does.
    fn pin_inode(path: &Path) {
        let mut pinned = path.as_os_str().to_owned();
        pinned.push(".pinned");
        std::fs::hard_link(path, std::path::PathBuf::from(pinned))
            .expect("pin the inode with a hard link");
    }

    /// Claim `path` once a dropped claim's lock is really gone. Another test in
    /// this process can be starting a subprocess at the moment a claim is
    /// dropped; until that child execs it shares the lock file's descriptor,
    /// and so the lock, for a few milliseconds.
    async fn claim_once_released(path: &Path) -> SocketClaim {
        let deadline = tokio::time::Instant::now() + DEADLINE;
        loop {
            match SocketClaim::claim(path).await {
                Ok(claim) => return claim,
                Err(e) => assert!(
                    in_use_holder(&e) == Some(SocketHolder::LockHolder)
                        && tokio::time::Instant::now() < deadline,
                    "the path's lock was not released: {e:#}"
                ),
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    fn in_use_path(e: &anyhow::Error) -> Option<&Path> {
        e.downcast_ref::<SocketInUse>()
            .map(|in_use| in_use.path.as_path())
    }

    fn in_use_holder(e: &anyhow::Error) -> Option<SocketHolder> {
        e.downcast_ref::<SocketInUse>().map(|in_use| in_use.holder)
    }

    /// The reported bug's second half: a second daemon used to unlink the live
    /// socket and bind its own, leaving the first daemon -- and every client
    /// on it -- unreachable at the path.
    #[tokio::test]
    async fn a_second_daemon_refuses_a_live_socket_instead_of_replacing_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let first = start_daemon(&path);
        let _attached = wait_until_serving(&path).await;
        pin_inode(&path);
        let before = inode(&path);

        let refusal = run_expecting_refusal(&path).await;

        assert_eq!(
            in_use_path(&refusal),
            Some(path.as_path()),
            "a second daemon must report the path as in use, got: {refusal:#}"
        );
        assert_eq!(in_use_holder(&refusal), Some(SocketHolder::LockHolder));
        assert_eq!(inode(&path), before, "the live socket must not be replaced");
        welcome_from(&path)
            .await
            .expect("the first daemon must still greet new clients at the path");
        assert!(
            !first.is_finished(),
            "the first daemon must still be serving"
        );
        first.abort();
        let _ = first.await;
    }

    /// A live daemon from a build without the lock is left serving. This one
    /// accepts the probe's connection only when the test accepts it and never
    /// greets, so the probe gives up after its 500 ms limit -- the path a busy
    /// or silent listener takes -- and still counts it as in use.
    #[tokio::test]
    async fn an_answering_socket_without_a_lock_holder_is_left_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let older = tokio::net::UnixListener::bind(&path).unwrap();
        pin_inode(&path);
        let before = inode(&path);

        let refusal = run_expecting_refusal(&path).await;

        assert_eq!(in_use_path(&refusal), Some(path.as_path()), "{refusal:#}");
        assert_eq!(
            in_use_holder(&refusal),
            Some(SocketHolder::UnlockedListener),
            "{refusal:#}"
        );
        assert_eq!(
            inode(&path),
            before,
            "the answering socket must not be replaced"
        );
        let _client = tokio::net::UnixStream::connect(&path)
            .await
            .expect("the older daemon must still be reachable");
        older
            .accept()
            .await
            .expect("the older daemon still accepts");
    }

    /// Whoever holds the lock owns the path, even while its socket does not
    /// answer (it may be between binding and accepting).
    #[tokio::test]
    async fn a_lock_holder_keeps_the_path_even_when_its_socket_looks_stale() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        stale_socket(&path);
        pin_inode(&path);
        let before = inode(&path);
        let held = std::fs::File::create(socket_lock_path(&path)).unwrap();
        held.try_lock().expect("the test takes the lock first");

        let refusal = run_expecting_refusal(&path).await;

        assert_eq!(in_use_path(&refusal), Some(path.as_path()), "{refusal:#}");
        assert_eq!(inode(&path), before, "the lock holder's socket must stay");
        drop(held);
    }

    /// Boundary: what a crashed or killed daemon leaves behind is replaced.
    #[tokio::test]
    async fn a_stale_socket_from_a_dead_daemon_is_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        stale_socket(&path);
        pin_inode(&path);
        let stale = inode(&path);

        let daemon = start_daemon(&path);
        let _attached = wait_until_serving(&path).await;

        assert_ne!(
            inode(&path),
            stale,
            "the stale socket must have been replaced"
        );
        let lock_mode = std::fs::metadata(socket_lock_path(&path))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(
            lock_mode, 0o600,
            "the lock file must be private to its owner"
        );
        daemon.abort();
        let _ = daemon.await;
    }

    /// A path that holds something other than a socket is somebody's file:
    /// the daemon refuses, says why, and deletes nothing.
    #[tokio::test]
    async fn a_path_that_is_not_a_socket_is_never_removed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        std::fs::write(&path, "keep me").unwrap();

        let refusal = run_expecting_refusal(&path).await;

        assert!(
            in_use_path(&refusal).is_none(),
            "not an ownership conflict: {refusal:#}"
        );
        assert!(
            format!("{refusal:#}").contains(&not_a_socket_refusal(&path)),
            "the error must name the path and say it is left alone: {refusal:#}"
        );
        assert!(
            !not_a_socket_refusal(&path).contains("  "),
            "the refusal must read as one sentence, without a run of spaces: {refusal:#}"
        );
        assert_eq!(
            std::fs::read_to_string(&path).expect("the file must still exist"),
            "keep me",
            "the file the error says is left alone must be untouched"
        );
    }

    /// A daemon removes its socket when it stops, but never a socket someone
    /// else has since put at the path.
    #[tokio::test]
    async fn dropping_a_claim_removes_only_its_own_socket() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");

        let mut replaced = SocketClaim::claim(&path).await.unwrap();
        let listener = replaced.bind().unwrap();
        pin_inode(&path);
        std::fs::remove_file(&path).unwrap();
        let _other = std::os::unix::net::UnixListener::bind(&path).unwrap();
        let other = inode(&path);
        drop(listener);
        drop(replaced);
        assert_eq!(
            inode(&path),
            other,
            "another socket at the path must survive"
        );

        std::fs::remove_file(&path).unwrap();
        let mut own = claim_once_released(&path).await;
        let listener = own.bind().unwrap();
        assert!(path.exists());
        drop(listener);
        drop(own);
        assert!(!path.exists(), "a claim must remove the socket it bound");
    }

    /// Survivors of a closed dashboard race for the path: exactly one binds,
    /// the rest learn it is in use.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn exactly_one_of_several_concurrent_daemons_binds() {
        const CONTENDERS: usize = 4;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        stale_socket(&path);

        let daemons: Vec<_> = (0..CONTENDERS).map(|_| start_daemon(&path)).collect();
        let deadline = tokio::time::Instant::now() + DEADLINE;
        while daemons.iter().filter(|d| d.is_finished()).count() < CONTENDERS - 1 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "only {} of {CONTENDERS} contenders gave up within {DEADLINE:?}; more than \
                 one is serving",
                daemons.iter().filter(|d| d.is_finished()).count()
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        let (finished, running): (Vec<_>, Vec<_>) = daemons
            .into_iter()
            .partition(tokio::task::JoinHandle::is_finished);
        assert_eq!(running.len(), 1, "exactly one contender must be serving");
        for daemon in finished {
            let refusal = daemon
                .await
                .unwrap()
                .expect_err("a loser cannot end cleanly");
            assert_eq!(in_use_path(&refusal), Some(path.as_path()), "{refusal:#}");
        }
        let _attached = wait_until_serving(&path).await;
        for daemon in running {
            daemon.abort();
            let _ = daemon.await;
        }
    }

    /// Every way the probe can end, including the two that only a busy
    /// listener produces: a timeout, and a full backlog (`WouldBlock`, which
    /// is what a non-blocking unix connect returns on Linux instead of
    /// pending). Neither may be read as stale.
    #[test]
    fn a_busy_or_slow_listener_counts_as_answering_not_stale() {
        use std::io::{Error, ErrorKind};

        let verdict = |outcome| format!("{:?}", probe_verdict(outcome));
        assert_eq!(verdict(None), "Answered", "a probe that timed out");
        assert_eq!(verdict(Some(Ok(()))), "Answered");
        assert_eq!(
            verdict(Some(Err(Error::from(ErrorKind::WouldBlock)))),
            "Answered",
            "a full backlog"
        );
        assert_eq!(
            verdict(Some(Err(Error::from(ErrorKind::ConnectionRefused)))),
            "Stale"
        );
        assert_eq!(
            verdict(Some(Err(Error::from(ErrorKind::NotFound)))),
            "Vanished"
        );
        assert!(matches!(
            probe_verdict(Some(Err(Error::from(ErrorKind::PermissionDenied)))),
            ProbeVerdict::Failed(_)
        ));
    }

    /// The probe of an answering daemon stays for its greeting, so the daemon
    /// is never left writing to a closed connection -- which it logs as a
    /// client error, once per probe, several times a second.
    #[tokio::test]
    async fn the_probe_waits_for_the_greeting_before_hanging_up() {
        use tokio::io::AsyncWriteExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let older = tokio::net::UnixListener::bind(&path).unwrap();
        let greeter = tokio::spawn(async move {
            let (mut stream, _) = older.accept().await.unwrap();
            // A daemon greets at once; the delay makes a probe that hung up
            // straight after connecting show as a failed write.
            tokio::time::sleep(Duration::from_millis(100)).await;
            stream.write_all(b"{\"greeting\":true}\n").await
        });

        let refusal = run_expecting_refusal(&path).await;

        assert_eq!(
            in_use_holder(&refusal),
            Some(SocketHolder::UnlockedListener),
            "{refusal:#}"
        );
        greeter
            .await
            .unwrap()
            .expect("the greeting must reach a probe that is still connected");
    }

    /// Losing the race to bind against a daemon without the lock is the same
    /// "in use" as finding it there, not a failure to report.
    #[tokio::test]
    async fn a_bind_lost_to_a_listener_without_the_lock_is_reported_in_use() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let mut claim = SocketClaim::claim(&path).await.unwrap();
        let _older = std::os::unix::net::UnixListener::bind(&path).unwrap();
        let theirs = inode(&path);

        let refusal = claim.bind().expect_err("the path is taken");

        assert_eq!(in_use_path(&refusal), Some(path.as_path()), "{refusal:#}");
        assert_eq!(
            in_use_holder(&refusal),
            Some(SocketHolder::UnlockedListener)
        );
        drop(claim);
        assert_eq!(inode(&path), theirs, "their socket must stay");
    }

    /// A daemon stopped mid-session closes its clients' connections, so they
    /// see it go and reconnect, instead of holding a connection to a daemon
    /// that no longer exists.
    #[tokio::test]
    async fn a_stopped_daemon_closes_its_clients_connections() {
        use tokio::io::AsyncReadExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let daemon = start_daemon(&path);
        let mut attached = wait_until_serving(&path).await;

        daemon.abort();
        let _ = daemon.await;

        let mut rest = Vec::new();
        let read = tokio::time::timeout(DEADLINE, attached.read_to_end(&mut rest))
            .await
            .expect("the client connection must be closed when the daemon stops");
        assert!(read.is_ok(), "the connection must end cleanly: {read:?}");
    }

    /// A daemon whose socket path is taken from under it would otherwise hold
    /// the lock -- and so the path -- while serving an inode nobody can reach.
    #[tokio::test]
    async fn a_daemon_whose_socket_is_replaced_gives_up_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let daemon = start_daemon(&path);
        let _attached = wait_until_serving(&path).await;

        std::fs::remove_file(&path).unwrap();
        let _replacement = std::os::unix::net::UnixListener::bind(&path).unwrap();
        let replacement = inode(&path);

        let outcome = tokio::time::timeout(DEADLINE, daemon)
            .await
            .expect("the daemon must stop once its socket is replaced")
            .unwrap();
        let error = outcome.expect_err("stopping for this is an error");
        assert_eq!(
            error.downcast_ref::<SocketReplaced>(),
            Some(&SocketReplaced { path: path.clone() }),
            "{error:#}"
        );
        assert_eq!(inode(&path), replacement, "the replacement must stay");
        // Waits out the subprocess window `claim_once_released` describes.
        let lock = std::fs::File::open(socket_lock_path(&path)).unwrap();
        let deadline = tokio::time::Instant::now() + DEADLINE;
        while lock.try_lock().is_err() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the daemon must have released the path's lock"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// The runner a stopped daemon started stops with it: nothing more is
    /// written to that daemon's session recording afterwards.
    #[tokio::test]
    async fn a_stopped_daemons_runner_stops_with_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let recordings = dir.path().join("telemetry");
        let listen = format!("unix:{}", path.display());
        let opts = RunnerOptions {
            persist_dir: Some(recordings.clone()),
            ..RunnerOptions::default()
        };
        let daemon = tokio::spawn(async move { super::run(&listen, opts).await });
        let recorded = || -> u64 {
            std::fs::read_dir(&recordings).map_or(0, |entries| {
                entries
                    .filter_map(Result::ok)
                    .filter_map(|e| e.metadata().ok())
                    .map(|m| m.len())
                    .sum()
            })
        };
        let deadline = tokio::time::Instant::now() + DEADLINE;
        while recorded() == 0 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the daemon recorded nothing within {DEADLINE:?}"
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        daemon.abort();
        let _ = daemon.await;
        let at_stop = recorded();
        // The runner records a snapshot every second; three would land in
        // this window if it were still running.
        tokio::time::sleep(Duration::from_secs(3)).await;

        assert_eq!(
            recorded(),
            at_stop,
            "the recording grew after the daemon stopped: its runner is still running"
        );
    }

    /// Serve `path` with `seams`, in a task the test can abort.
    fn start_serving(
        path: &Path,
        seams: ServeSeams,
    ) -> tokio::task::JoinHandle<anyhow::Result<()>> {
        let path = path.to_path_buf();
        tokio::spawn(serve_unix(path, RunnerOptions::default(), || {}, seams))
    }

    /// A runner that never sends anything and never stops.
    fn idle_runner() -> super::StartRunner {
        Box::new(|_parts| Box::pin(std::future::pending::<()>()))
    }

    /// A daemon whose runner stops would greet clients and never send them
    /// anything again; it stops instead, so its keeper starts it over.
    #[tokio::test]
    async fn a_daemon_stops_when_its_runner_does() {
        let dir = tempfile::tempdir().unwrap();
        let returned: super::StartRunner = Box::new(|_| Box::pin(async {}));
        let panicked: super::StartRunner = Box::new(|_| Box::pin(async { panic!("runner bug") }));
        for (name, start_runner) in [("returned", returned), ("panicked", panicked)] {
            let path = dir.path().join(format!("{name}.sock"));
            let seams = ServeSeams {
                start_runner,
                client_tasks: None,
                path_check_interval: super::PATH_CHECK_INTERVAL,
            };
            let outcome = tokio::time::timeout(DEADLINE, start_serving(&path, seams))
                .await
                .unwrap_or_else(|_| panic!("a daemon whose runner {name} kept serving"))
                .unwrap();
            let error = format!("{:#}", outcome.expect_err("a stopped runner is an error"));
            assert!(error.contains("telemetry runner"), "{name}: {error}");
            if name == "panicked" {
                assert!(error.contains("panicked"), "{error}");
            }
            assert!(!path.exists(), "{name}: the daemon must remove its socket");
        }
    }

    /// A stopped daemon closes its clients' connections itself, not only
    /// because its broadcast channel went away: here the channel outlives it.
    #[tokio::test]
    async fn a_stopped_daemon_closes_its_clients_even_while_its_channel_lives() {
        use tokio::io::AsyncReadExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let kept = std::sync::Arc::new(std::sync::Mutex::new(None));
        let keep = std::sync::Arc::clone(&kept);
        let seams = ServeSeams {
            start_runner: Box::new(move |parts| {
                *keep.lock().unwrap() = Some(parts.tx);
                Box::pin(std::future::pending::<()>())
            }),
            client_tasks: None,
            path_check_interval: super::PATH_CHECK_INTERVAL,
        };
        let daemon = start_serving(&path, seams);
        let mut attached = wait_until_serving(&path).await;

        daemon.abort();
        let _ = daemon.await;
        assert!(
            kept.lock().unwrap().is_some(),
            "the channel must still be alive"
        );

        let mut rest = Vec::new();
        tokio::time::timeout(DEADLINE, attached.read_to_end(&mut rest))
            .await
            .expect("the client connection must be closed when the daemon stops")
            .expect("the connection must end cleanly");
    }

    /// Finished client tasks are released, so a long-running daemon does not
    /// hold one for every client it ever had.
    #[tokio::test]
    async fn finished_client_tasks_are_released() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let count = std::sync::Arc::new(AtomicUsize::new(0));
        let seams = ServeSeams {
            start_runner: idle_runner(),
            client_tasks: Some(std::sync::Arc::clone(&count)),
            path_check_interval: super::PATH_CHECK_INTERVAL,
        };
        let daemon = start_serving(&path, seams);
        let held = |n: usize| {
            let count = std::sync::Arc::clone(&count);
            async move {
                let deadline = tokio::time::Instant::now() + DEADLINE;
                while count.load(Ordering::SeqCst) != n {
                    assert!(
                        tokio::time::Instant::now() < deadline,
                        "the daemon still holds {} client tasks, expected {n}",
                        count.load(Ordering::SeqCst)
                    );
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
            }
        };

        let client = wait_until_serving(&path).await;
        held(1).await;
        drop(client);
        held(0).await;

        daemon.abort();
        let _ = daemon.await;
    }

    /// Gives a directory its mode back however a test ends, so the temporary
    /// directory can be removed.
    struct Restore<'a>(&'a Path);

    impl Drop for Restore<'_> {
        fn drop(&mut self) {
            let _ = std::fs::set_permissions(self.0, std::fs::Permissions::from_mode(0o700));
        }
    }

    /// A path that cannot be checked for a moment says nothing about whose
    /// socket it is: the daemon keeps serving through it.
    #[tokio::test]
    async fn a_daemon_keeps_serving_while_its_path_briefly_cannot_be_checked() {
        let dir = tempfile::tempdir().unwrap();
        let socket_dir = dir.path().join("private");
        std::fs::create_dir(&socket_dir).unwrap();
        let path = socket_dir.join("rocmdashd.sock");
        let daemon = start_serving(
            &path,
            ServeSeams {
                start_runner: idle_runner(),
                client_tasks: None,
                path_check_interval: super::PATH_CHECK_INTERVAL,
            },
        );
        let _attached = wait_until_serving(&path).await;

        let restore = Restore(&socket_dir);
        std::fs::set_permissions(&socket_dir, std::fs::Permissions::from_mode(0o000)).unwrap();
        if std::fs::symlink_metadata(&path).is_ok() {
            drop(restore);
            eprintln!("skip: this user can read a mode-000 directory (root?)");
            daemon.abort();
            let _ = daemon.await;
            return;
        }
        // Long enough for at least one path check to fail to read the path.
        tokio::time::sleep(super::PATH_CHECK_INTERVAL + Duration::from_millis(500)).await;
        drop(restore);

        assert!(
            !daemon.is_finished(),
            "a daemon must not give its path up over one unreadable check"
        );
        welcome_from(&path)
            .await
            .expect("the daemon must still serve its path");
        daemon.abort();
        let _ = daemon.await;
    }

    /// A path that stays unreadable is given up after five checks in a row,
    /// so the daemon stops and its keeper can claim the path again.
    #[tokio::test]
    async fn a_daemon_gives_its_path_up_when_it_stays_unreadable() {
        let dir = tempfile::tempdir().unwrap();
        let socket_dir = dir.path().join("private");
        std::fs::create_dir(&socket_dir).unwrap();
        let path = socket_dir.join("rocmdashd.sock");
        let daemon = start_serving(
            &path,
            ServeSeams {
                start_runner: idle_runner(),
                client_tasks: None,
                path_check_interval: Duration::from_millis(20),
            },
        );
        let _attached = wait_until_serving(&path).await;

        let restore = Restore(&socket_dir);
        std::fs::set_permissions(&socket_dir, std::fs::Permissions::from_mode(0o000)).unwrap();
        if std::fs::symlink_metadata(&path).is_ok() {
            drop(restore);
            eprintln!("skip: this user can read a mode-000 directory (root?)");
            daemon.abort();
            let _ = daemon.await;
            return;
        }
        // Well under five checks at the production interval (10 s), so this
        // fails if the test interval is not the one the loop uses.
        let outcome = tokio::time::timeout(Duration::from_secs(3), daemon)
            .await
            .expect("a daemon whose path stays unreadable must stop")
            .unwrap();
        drop(restore);

        let error = format!("{:#}", outcome.expect_err("giving the path up is an error"));
        assert!(
            error.contains(&format!(
                "failed {} times in a row",
                super::PATH_CHECK_UNKNOWN_LIMIT
            )),
            "{error}"
        );
    }

    /// Something other than a socket appearing at the path between the probe
    /// and the bind is refused as not a socket, not taken for a daemon.
    #[tokio::test]
    async fn a_file_that_appears_before_the_bind_is_refused_as_not_a_socket() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let mut claim = SocketClaim::claim(&path).await.unwrap();
        std::fs::write(&path, "keep me").unwrap();

        let refusal = claim.bind().expect_err("the path is taken");

        assert!(in_use_path(&refusal).is_none(), "{refusal:#}");
        assert!(
            format!("{refusal:#}").contains(&not_a_socket_refusal(&path)),
            "{refusal:#}"
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "keep me");
    }

    /// Unreadable path checks give the path up only five in a row; a check
    /// that reads the path starts the count over.
    #[test]
    fn only_five_unreadable_path_checks_in_a_row_give_the_path_up() {
        use super::unreadable_streak;

        let mut streak = 0;
        for _ in 0..4 {
            streak = unreadable_streak(streak, false).expect("fewer than five");
        }
        assert_eq!(unreadable_streak(streak, false), None, "the fifth gives up");
        assert_eq!(
            unreadable_streak(streak, true),
            Some(0),
            "a good check resets"
        );

        let mut streak = 0;
        for readable in [false, false, false, false, true, false, false, false, false] {
            streak = unreadable_streak(streak, readable).expect("never five in a row");
        }
    }
}
