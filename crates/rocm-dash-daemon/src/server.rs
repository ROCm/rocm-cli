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
/// created with mode `0o600` so only the owning user can connect. The socket's
/// directory is checked before anything is created in it, judged by what it
/// resolves to when it is a symlink (a bare file name, created in the current
/// directory, is the exception: that directory is not checked, see
/// `socket_parent`):
///
/// * a missing one is created with mode `0o700`;
/// * something that is not a directory is always refused, first;
/// * a sticky directory (such as `/tmp`) is never chmodded, whoever owns it,
///   and is used when root or this user owns it;
/// * any other directory owned by another user is refused;
/// * one this user owns is restricted to `0o700` when group or others have
///   any access to it, and otherwise (`0o700`, `0o500`) left as it is.
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

/// What to do with an existing socket directory, decided from its owner and
/// mode alone (see [`socket_dir_verdict`]).
#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SocketDirVerdict {
    /// Use it as it is.
    Use,
    /// A sticky directory others may write in, such as `/tmp`: use it as it
    /// is, and say so once the path is claimed.
    UseShared,
    /// Ours, but group or others have some access to it: restrict it to
    /// `0o700` first.
    Tighten,
    /// Another user controls it: refuse it.
    Refuse,
}

/// The socket-directory policy, as a pure function of the directory's owner
/// and mode and this process's effective uid.
///
/// * A **sticky** directory is never chmodded, whoever owns it -- root with
///   its socket directly in `/tmp` included. The sticky bit means only an
///   entry's owner may remove or rename it, so our `0o600` socket and lock
///   cannot be replaced there. It is usable when root or this user owns it;
///   any other owner could remove our entries, so it is refused.
/// * A directory **we own** is restricted to `0o700` whenever group or others
///   have any access to it (`mode & 0o077 != 0`), so a `0o755` or `0o740` one
///   is tightened too; one they cannot reach at all, `0o700` or `0o500`, is
///   left as it is, so no chmod repeats once it is private.
/// * Any **other owner** means another local user controls the path (a
///   squatting attack), so it is refused.
#[cfg(unix)]
const fn socket_dir_verdict(dir_uid: u32, mode: u32, our_uid: u32) -> SocketDirVerdict {
    if mode & 0o1000 != 0 {
        return if dir_uid != 0 && dir_uid != our_uid {
            SocketDirVerdict::Refuse
        } else if mode & 0o022 != 0 {
            SocketDirVerdict::UseShared
        } else {
            SocketDirVerdict::Use
        };
    }
    if dir_uid != our_uid {
        SocketDirVerdict::Refuse
    } else if mode & 0o077 != 0 {
        SocketDirVerdict::Tighten
    } else {
        SocketDirVerdict::Use
    }
}

/// Why a socket directory another user controls is refused. Its own function
/// so a test can check the wording directly as well as through
/// [`check_socket_dir`], whose refusal path a test reaches by running the
/// check as a different uid (see `ServeSeams::our_uid`).
#[cfg(unix)]
fn unsecurable_socket_dir_context(parent: &std::path::Path) -> String {
    format!(
        "refusing socket directory {} - it is owned by another user, so a \
         local user could replace the telemetry socket. Point the socket at a \
         directory you own, for example under $XDG_RUNTIME_DIR or $HOME",
        parent.display()
    )
}

/// This process's effective user id: who owns what it creates.
#[cfg(unix)]
#[allow(unsafe_code)] // libc FFI
fn effective_uid() -> u32 {
    // SAFETY: `geteuid` takes no arguments, cannot fail, and is always safe
    // to call.
    unsafe { libc::geteuid() }
}

/// Make sure the socket's parent directory may be used, before anything is
/// created, opened, probed or removed in it, and say whether it is a shared
/// sticky directory (for the warning [`warn_shared_socket_dir`] gives once
/// the path is claimed).
///
/// A missing parent is created at mode `0o700` (it is then ours). An existing
/// one is judged by what it resolves to, following symlinks -- the directory
/// the socket and its lock would really be created in. Something that is not
/// a directory is always refused, first; otherwise [`socket_dir_verdict`]
/// decides. The only chmod is the one it asks for, on a non-sticky directory
/// of ours that group or others have any access to, so the claim never works
/// in it while it is open to them, and logs that it did; a directory already
/// acceptable is left untouched, with no chmod and no log line, since a
/// dashboard that does not own the daemon runs this several times a second.
#[cfg(unix)]
fn check_socket_dir(parent: &std::path::Path, our_uid: u32) -> anyhow::Result<bool> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};

    let inspecting = || format!("inspecting socket directory {}", parent.display());
    let meta = match std::fs::metadata(parent) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            std::fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(parent)
                .with_context(|| format!("creating socket directory {}", parent.display()))?;
            // `create` also succeeds when someone else made the directory
            // between the stat above and here, so judge what is there now
            // like any existing directory.
            std::fs::metadata(parent).with_context(inspecting)?
        }
        Err(e) => {
            return Err(e).with_context(inspecting);
        }
    };
    if !meta.is_dir() {
        return Err(anyhow!(
            "socket directory {} is not a directory; point the socket at a directory you own",
            parent.display()
        ));
    }
    match socket_dir_verdict(meta.uid(), meta.mode(), our_uid) {
        SocketDirVerdict::Use => Ok(false),
        SocketDirVerdict::UseShared => Ok(true),
        SocketDirVerdict::Tighten => {
            std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700)).with_context(
                || {
                    format!(
                        "restricting socket directory {} to mode 0700",
                        parent.display()
                    )
                },
            )?;
            // Once per directory: it is private from now on, so later checks
            // leave it alone. `info`, not `warn`: the policy did what it is
            // for, and nothing failed; the dashboard's default log filter
            // (`DEFAULT_FILTER` in `apps/rocm/src/logging.rs`) keeps this
            // crate's info lines.
            info!(
                dir = %parent.display(),
                from = format_args!("{:04o}", meta.mode() & 0o7777),
                "restricted socket directory to mode 0700"
            );
            Ok(false)
        }
        SocketDirVerdict::Refuse => Err(anyhow!(unsecurable_socket_dir_context(parent))),
    }
}

/// The warning for a socket directory [`check_socket_dir`] found shared and
/// sticky, given once the path is claimed rather than on every attempt.
#[cfg(unix)]
fn warn_shared_socket_dir(parent: &std::path::Path) {
    warn!(
        dir = %parent.display(),
        "socket directory is a shared sticky directory (e.g. /tmp); leaving its mode as \
         it is and continuing, because only an entry's owner may remove it there and \
         the socket is created with mode 0600"
    );
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
/// replaced.
///
/// It stopped and released the path, so a retry can claim it again (the
/// dashboard's keeper retries after one poll); this is not a failure to
/// report.
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

/// The most a probe reads while waiting for a greeting. A daemon's greeting is
/// one short line; a listener that sends more without ending a line has
/// answered all the same, and the probe stops reading there.
#[cfg(unix)]
const PROBE_GREETING_LIMIT: u64 = 4096;

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

/// Connect to an existing socket and, if it answers, read its first line (at
/// most [`PROBE_GREETING_LIMIT`] bytes of it) before hanging up. A daemon
/// greets every client at once, so leaving before the greeting would have it
/// write to a closed connection and log an error for every probe.
#[cfg(unix)]
async fn probe_socket(path: &std::path::Path) -> Option<std::io::Result<()>> {
    use tokio::io::{AsyncBufReadExt, AsyncReadExt};

    tokio::time::timeout(STALE_PROBE_TIMEOUT, async {
        let stream = tokio::net::UnixStream::connect(path).await?;
        let mut greeting = Vec::new();
        // What it says, or whether it says anything, does not matter here.
        let _ = BufReader::new(stream.take(PROBE_GREETING_LIMIT))
            .read_until(b'\n', &mut greeting)
            .await;
        Ok(())
    })
    .await
    .ok()
}

/// Whether something answers at `path` right now.
///
/// A connection to it is accepted (or the listener is too busy to accept in
/// time). The same probe a claim uses to tell a live socket from a stale one,
/// and as careful with the daemon it reaches: it waits for the greeting before
/// hanging up. `false` when nothing is there, nobody listens, or the path is
/// not a socket.
#[cfg(unix)]
pub async fn socket_answers(path: &std::path::Path) -> bool {
    matches!(
        probe_verdict(probe_socket(path).await),
        ProbeVerdict::Answered
    )
}

/// [`socket_answers`] for a caller that cannot await.
///
/// That is a process about to exit on a signal, from a thread that may be a
/// Tokio worker. The same probe, on blocking I/O, runs on a thread of its
/// own, and this waits for its verdict at most [`STALE_PROBE_TIMEOUT`]; a
/// probe still running then counts as answering, as a timed-out one does in
/// [`socket_answers`]. A probe stuck past that is left to finish, or to end
/// with the process.
#[cfg(unix)]
pub fn socket_answers_blocking(path: &std::path::Path) -> bool {
    use std::io::{BufRead, Read};

    let path = path.to_path_buf();
    let (verdict, outcome) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let probe = || -> std::io::Result<()> {
            let stream = std::os::unix::net::UnixStream::connect(&path)?;
            stream.set_read_timeout(Some(STALE_PROBE_TIMEOUT))?;
            let mut greeting = Vec::new();
            // What it says, or whether it says anything, does not matter here.
            let _ = std::io::BufReader::new(stream.take(PROBE_GREETING_LIMIT))
                .read_until(b'\n', &mut greeting);
            Ok(())
        };
        let _ = verdict.send(probe());
    });
    matches!(
        probe_verdict(outcome.recv_timeout(STALE_PROBE_TIMEOUT).ok()),
        ProbeVerdict::Answered
    )
}

/// The lock file that says which daemon owns `socket`: the socket path with
/// `.lock` appended, in the same directory.
#[cfg(unix)]
fn socket_lock_path(socket: &std::path::Path) -> PathBuf {
    let mut lock = socket.as_os_str().to_owned();
    lock.push(".lock");
    PathBuf::from(lock)
}

/// What the refusal of a path that is not a socket says about it, right after
/// the path. The dashboard's end-to-end check (dash-57 in `dash_steps.rs`)
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
            // A symlink planted at the lock path is refused, not followed to
            // create or lock a file somewhere else.
            .custom_flags(libc::O_NOFOLLOW)
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
        use std::os::unix::fs::PermissionsExt;

        self.bind_with(|path| {
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        })
    }

    /// [`Self::bind`] with the chmod to `0o600` swapped, so a test can make
    /// it fail.
    fn bind_with(
        &mut self,
        restrict: impl FnOnce(&std::path::Path) -> std::io::Result<()>,
    ) -> anyhow::Result<tokio::net::UnixListener> {
        use std::os::unix::fs::MetadataExt;

        let path = &self.path;
        let listener = match tokio::net::UnixListener::bind(path) {
            Ok(listener) => listener,
            // Something bound the path between the probe and here without
            // taking the lock (a daemon of an older build): it is in use.
            // Unless what appeared there is not a socket at all.
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
                return Err(bind_in_use_refusal(path, std::fs::symlink_metadata(path)));
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
        restrict(path).with_context(|| format!("setting permissions on {}", path.display()))?;
        Ok(listener)
    }
}

/// Why a bind that found the path taken (`AddrInUse`) is refused, given what
/// the path names now: something that is not a socket is refused as such;
/// a socket, or a path that vanished or cannot be inspected since, is
/// reported in use by something without the lock (a daemon of an older
/// build), which the caller retries later. For a path that vanished, the
/// error's "already serves" overstates what is there, but the response it
/// gets -- the keeper's slower retry -- is the right one: whatever held the
/// path a moment ago may be back.
#[cfg(unix)]
fn bind_in_use_refusal(
    path: &std::path::Path,
    now: std::io::Result<std::fs::Metadata>,
) -> anyhow::Error {
    use std::os::unix::fs::FileTypeExt;

    if !now.map_or(true, |meta| meta.file_type().is_socket()) {
        return anyhow!(not_a_socket_refusal(path));
    }
    SocketInUse {
        path: path.to_path_buf(),
        holder: SocketHolder::UnlockedListener,
    }
    .into()
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

/// The unreadable path checks in a row after one more `check`: a check that
/// read the path (`Ok`) starts the count over at zero, and one that could not
/// adds one. Once they reach [`PATH_CHECK_UNKNOWN_LIMIT`] it returns that
/// last check's error instead.
#[cfg(unix)]
fn unreadable_streak<E>(streak: u32, check: Result<(), E>) -> Result<u32, E> {
    match check {
        Ok(()) => Ok(0),
        Err(e) if streak + 1 >= PATH_CHECK_UNKNOWN_LIMIT => Err(e),
        Err(_) => Ok(streak + 1),
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
    /// The user the socket directory must belong to (see
    /// [`check_socket_dir`]).
    our_uid: u32,
}

#[cfg(unix)]
impl ServeSeams {
    fn production() -> Self {
        Self {
            start_runner: Box::new(start_run_loop),
            client_tasks: None,
            path_check_interval: PATH_CHECK_INTERVAL,
            our_uid: effective_uid(),
        }
    }
}

/// Why a daemon stopped because its runner did.
#[cfg(unix)]
fn runner_stopped_error(outcome: Result<(), tokio::task::JoinError>) -> anyhow::Error {
    match outcome {
        Ok(()) => anyhow!("the telemetry runner stopped"),
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

/// The directory a socket path is created in, or `None` for a bare file name
/// such as `unix:rocmdashd.sock`, which is created in the current directory.
/// That directory is deliberately left unchecked: the check may chmod a
/// directory of ours to `0o700`, which must not happen to whatever directory
/// the daemon was started from, and its parent, the empty path, is one the
/// check refuses, so the daemon would never serve.
#[cfg(unix)]
fn socket_parent(path: &std::path::Path) -> Option<&std::path::Path> {
    path.parent().filter(|p| !p.as_os_str().is_empty())
}

#[cfg(unix)]
async fn serve_unix(
    path: PathBuf,
    opts: RunnerOptions,
    ready: impl FnOnce() + Send,
    seams: ServeSeams,
) -> anyhow::Result<()> {
    let parent = socket_parent(&path);
    // The lock file lives in the socket's directory, so that directory has to
    // exist, be one another user does not control, and be private (no group
    // or other access, see `socket_dir_verdict`) if it is ours and not sticky,
    // before the claim creates, opens or probes anything in it. The check
    // changes nothing in a directory that is already acceptable: a dashboard
    // that is not the owner retries several times a second. The warning for a
    // shared sticky directory waits until the path is claimed, so it is given
    // once per daemon rather than per attempt.
    let shared = match parent {
        Some(parent) => check_socket_dir(parent, seams.our_uid)?,
        None => false,
    };
    let mut claim = SocketClaim::claim(&path).await?;
    if let Some(parent) = parent
        && shared
    {
        warn_shared_socket_dir(parent);
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
                // `path_state` stats the socket path (`symlink_metadata`) on
                // this serving task, blocking it for as long as the stat
                // takes. Accepted on purpose: by default
                // (`dashboard_socket_path` in rocm-core's `config.rs`) the
                // socket lives in `$XDG_RUNTIME_DIR`, else in
                // `$HOME/.rocm/data/telemetry`, else in a per-user directory
                // under the temp dir, where a stat usually returns at once.
                // A path on a filesystem that hangs would stall this task with
                // it: one configured there, or the `$HOME` fallback when
                // `XDG_RUNTIME_DIR` is unset and the home directory is
                // network-mounted. That case is tracked in ROCMAI-1152.
                let check = match claim.path_state() {
                    PathState::Ours => Ok(()),
                    // Serving an inode nobody can reach would hold the lock,
                    // and so the path, forever.
                    PathState::Gone => return Err(SocketReplaced { path }.into()),
                    PathState::Unknown(e) => {
                        warn!(socket = %path.display(), error = %e, "could not check the socket path");
                        Err(e)
                    }
                };
                match unreadable_streak(unknown_checks, check) {
                    Ok(streak) => unknown_checks = streak,
                    Err(e) => {
                        return Err(anyhow!(
                            "checking socket {} failed {PATH_CHECK_UNKNOWN_LIMIT} times in a \
                             row: {e}",
                            path.display(),
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
        PAUSE_RESUME_UNSUPPORTED, SocketDirVerdict, check_socket_dir, effective_uid,
        socket_dir_verdict, unsecurable_socket_dir_context,
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

    /// The socket-directory policy for every owner/mode combination that
    /// matters. Directories owned by root or by a second user cannot be made
    /// in a unit test, so the decision is pinned here as a pure function; the
    /// serving-level tests below reach the same decision by running the check
    /// as a different uid (`ServeSeams::our_uid`).
    #[test]
    fn the_socket_directory_policy() {
        use SocketDirVerdict::{Refuse, Tighten, Use, UseShared};

        let cases = [
            // (owner, mode, our uid, verdict, why)
            (0, 0o1777, 1000, UseShared, "/tmp, as a user"),
            (0, 0o1777, 0, UseShared, "/tmp, as root: never chmodded"),
            (
                1000,
                0o1777,
                1000,
                UseShared,
                "our own sticky shared directory",
            ),
            (0, 0o1700, 1000, Use, "root-owned, sticky, private"),
            (
                0,
                0o1770,
                1000,
                UseShared,
                "root-owned, sticky, group may write",
            ),
            (
                0,
                0o1707,
                1000,
                UseShared,
                "root-owned, sticky, others may write",
            ),
            (
                1000,
                0o1770,
                1000,
                UseShared,
                "ours, sticky, group may write",
            ),
            (
                1000,
                0o1707,
                1000,
                UseShared,
                "ours, sticky, others may write",
            ),
            (
                1001,
                0o1777,
                1000,
                Refuse,
                "another user's sticky directory",
            ),
            (1000, 0o777, 1000, Tighten, "ours, anyone may write"),
            (1000, 0o775, 1000, Tighten, "ours, group may write"),
            (1000, 0o755, 1000, Tighten, "ours, others may read"),
            (1000, 0o740, 1000, Tighten, "ours, group may read"),
            (1000, 0o701, 1000, Tighten, "ours, others may search"),
            (1000, 0o500, 1000, Use, "ours, read-only"),
            (1000, 0o700, 1000, Use, "ours, private"),
            (0, 0o755, 0, Tighten, "root's own directory, as root"),
            (0, 0o700, 0, Use, "root's own private directory, as root"),
            (1001, 0o700, 1000, Refuse, "another user's directory"),
            (
                1001,
                0o775,
                1000,
                Refuse,
                "another user's writable directory",
            ),
            (
                0,
                0o777,
                1000,
                Refuse,
                "root-owned, not sticky, anyone may write",
            ),
            (0, 0o755, 1000, Refuse, "root-owned, not sticky"),
        ];
        for (owner, mode, ours, verdict, why) in cases {
            assert_eq!(
                socket_dir_verdict(owner, mode, ours),
                verdict,
                "{why}: owner {owner}, mode {mode:o}, we are {ours}"
            );
        }
    }

    fn mode_bits(path: &Path) -> u32 {
        std::fs::metadata(path).unwrap().permissions().mode() & 0o7777
    }

    /// A directory of ours that others may read but not write, the common
    /// `0o755`, is tightened to 0700 too, and the change is logged with the
    /// directory and its old mode; a second check, of the now private
    /// directory, logs nothing.
    #[test]
    fn tightens_owned_preexisting_parent() {
        let (captured, _guard) = log_at(tracing::Level::INFO);
        let log = || String::from_utf8_lossy(&captured.0.lock().unwrap()).into_owned();
        let tmp = tempfile::tempdir().unwrap();
        let parent = tmp.path().join("loose");
        std::fs::create_dir(&parent).unwrap();
        std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o755)).unwrap();

        check_socket_dir(&parent, effective_uid()).expect("should tighten a directory we own");

        assert_eq!(mode_bits(&parent), 0o700);
        let line = log();
        assert!(
            line.contains("restricted socket directory to mode 0700")
                && line.contains(&format!("dir={}", parent.display()))
                && line.contains("from=0755"),
            "{line}"
        );
        check_socket_dir(&parent, effective_uid()).expect("checking it again is fine");
        assert_eq!(log(), line, "nothing more is logged once it is private");
    }

    /// A missing nested parent is created at mode 0700, and so is every
    /// directory the recursive create makes on the way to it.
    #[test]
    fn creates_missing_parent_at_0700() {
        let (captured, _guard) = log_at(tracing::Level::INFO);
        let tmp = tempfile::tempdir().unwrap();
        let nested = tmp.path().join("nested");
        let parent = nested.join("telemetry");
        check_socket_dir(&parent, effective_uid()).expect("should create a new directory");
        assert_eq!(
            mode_bits(&parent),
            0o700,
            "newly created socket dir must be private"
        );
        // Created private, not made so afterwards: the directory the
        // recursive create made on the way, which the check never chmods, is
        // private too, and no restriction was logged.
        assert_eq!(mode_bits(&nested), 0o700, "an intermediate directory");
        let log = String::from_utf8_lossy(&captured.0.lock().unwrap()).into_owned();
        assert!(
            !log.contains("restricted socket directory"),
            "a directory created private needs no chmod: {log}"
        );
    }

    /// A directory of ours that group or others have any access to is
    /// restricted to 0700; one they cannot reach at all is left exactly as it
    /// is.
    #[test]
    fn only_a_socket_directory_others_can_reach_is_tightened() {
        let tmp = tempfile::tempdir().unwrap();
        for (mode, after) in [
            (0o777, 0o700),
            (0o775, 0o700),
            (0o750, 0o700),
            (0o701, 0o700),
            (0o700, 0o700),
            (0o500, 0o500),
        ] {
            let parent = tmp.path().join(format!("dir-{mode:o}"));
            std::fs::create_dir(&parent).unwrap();
            std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(mode)).unwrap();
            check_socket_dir(&parent, effective_uid()).expect("our own directory is usable");
            assert_eq!(mode_bits(&parent), after, "a {mode:o} directory of ours");
            std::fs::set_permissions(&parent, std::fs::Permissions::from_mode(0o700)).unwrap();
        }
    }

    /// The real `/tmp`, when it is the root-owned sticky kind and not ours: it
    /// is used as it is (and reported as shared, for the warning), never
    /// chmodded. Skipped where the temp dir is something else (a custom
    /// `TMPDIR`, some containers) or is our own.
    #[test]
    fn preexisting_root_sticky_parent_is_used_as_it_is() {
        let tmpdir = std::env::temp_dir();
        let Ok(meta) = std::fs::metadata(&tmpdir) else {
            eprintln!("skip: no temp dir to probe; nothing to assert");
            return;
        };
        if meta.uid() != 0 || meta.mode() & 0o1000 == 0 || effective_uid() == 0 {
            eprintln!("skip: temp dir is not a root-owned sticky directory of someone else's");
            return;
        }
        let before = mode_bits(&tmpdir);
        let shared = check_socket_dir(&tmpdir, effective_uid())
            .expect("must not refuse a root-owned sticky parent (e.g. /tmp)");
        assert!(shared, "/tmp is reported as shared, for the warning");
        assert_eq!(mode_bits(&tmpdir), before, "/tmp is never chmodded");
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
    ///
    /// Dropping the listener does not always close the socket at once: a
    /// test running in parallel may have started a child process (the
    /// runner's `amd-smi` detection, through `std`/`tokio` `Command`) while
    /// the listener was open, and the child holds a copy of its descriptor
    /// until it execs, when `O_CLOEXEC` closes it. Until then the socket still
    /// accepts. So wait, within [`DEADLINE`], until a connection is refused;
    /// no later child can inherit a descriptor this process already closed.
    fn stale_socket(path: &Path) {
        drop(std::os::unix::net::UnixListener::bind(path).expect("bind the soon-stale socket"));
        let deadline = std::time::Instant::now() + DEADLINE;
        while std::os::unix::net::UnixStream::connect(path).is_ok() {
            assert!(
                std::time::Instant::now() < deadline,
                "the fixture must be stale: something still answers after {DEADLINE:?}"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
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

    /// The blocking probe a process uses on its way out gives the same
    /// verdicts as the async one: a listener that greets, or one that never
    /// does, answers; a stale socket and a missing path do not.
    #[test]
    fn the_blocking_probe_tells_an_answering_socket_from_a_stale_one() {
        use std::io::Write;

        use super::socket_answers_blocking;

        let dir = tempfile::tempdir().unwrap();
        let greeting = dir.path().join("greeting.sock");
        let listener = std::os::unix::net::UnixListener::bind(&greeting).unwrap();
        std::thread::spawn(move || {
            if let Ok((mut client, _)) = listener.accept() {
                let _ = client.write_all(b"{}\n");
            }
        });
        assert!(socket_answers_blocking(&greeting), "a daemon that greets");

        let silent = dir.path().join("silent.sock");
        let _silent = std::os::unix::net::UnixListener::bind(&silent).unwrap();
        assert!(
            socket_answers_blocking(&silent),
            "a listener that never greets"
        );

        let stale = dir.path().join("stale.sock");
        stale_socket(&stale);
        assert!(!socket_answers_blocking(&stale), "a stale socket");
        assert!(
            !socket_answers_blocking(&dir.path().join("missing.sock")),
            "a missing path"
        );
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

    /// A listener that streams bytes and never ends a line still answers,
    /// and the probe stops reading after a bounded greeting rather than
    /// buffering whatever arrives until its timeout.
    #[tokio::test]
    async fn the_probe_reads_a_bounded_greeting_from_a_listener_that_never_ends_a_line() {
        use tokio::io::AsyncWriteExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let listener = tokio::net::UnixListener::bind(&path).unwrap();
        let streamer = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let chunk = [b'x'; 1024];
            // Until the probe hangs up.
            while stream.write_all(&chunk).await.is_ok() {}
        });

        let outcome = super::probe_socket(&path).await;

        assert!(
            matches!(outcome, Some(Ok(()))),
            "the probe must finish reading before its timeout, got {outcome:?}"
        );
        assert!(matches!(probe_verdict(outcome), ProbeVerdict::Answered));
        tokio::time::timeout(Duration::from_secs(5), streamer)
            .await
            .expect("the listener must see the probe hang up")
            .unwrap();
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
                our_uid: super::effective_uid(),
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
            our_uid: super::effective_uid(),
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
            our_uid: super::effective_uid(),
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
                our_uid: super::effective_uid(),
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
                our_uid: super::effective_uid(),
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
        // Well under five checks at the production interval (2 s each, so
        // about 10 s), so this fails if the test interval is not the one the
        // loop uses.
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

    /// A bind that finds the path taken reports what the path names now: a
    /// file as not a socket, and a socket -- or a path that vanished since,
    /// which is not a reason to call it something else -- as in use by
    /// something without the lock.
    #[test]
    fn a_bind_that_finds_the_path_taken_reports_what_is_there_now() {
        use super::bind_in_use_refusal;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");

        let vanished = bind_in_use_refusal(
            &path,
            Err(std::io::Error::from(std::io::ErrorKind::NotFound)),
        );
        assert_eq!(
            in_use_holder(&vanished),
            Some(SocketHolder::UnlockedListener),
            "{vanished:#}"
        );

        let _listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
        let socket = bind_in_use_refusal(&path, std::fs::symlink_metadata(&path));
        assert_eq!(
            in_use_holder(&socket),
            Some(SocketHolder::UnlockedListener),
            "{socket:#}"
        );

        let file = dir.path().join("file.sock");
        std::fs::write(&file, "keep me").unwrap();
        let refusal = bind_in_use_refusal(&file, std::fs::symlink_metadata(&file));
        assert!(in_use_path(&refusal).is_none(), "{refusal:#}");
        assert!(
            format!("{refusal:#}").contains(&not_a_socket_refusal(&file)),
            "{refusal:#}"
        );
    }

    /// A claim whose chmod of the bound socket fails still removes that socket
    /// when it is dropped: the socket's identity is recorded before the chmod.
    #[tokio::test]
    async fn a_claim_whose_chmod_fails_still_removes_its_socket() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let mut claim = SocketClaim::claim(&path).await.unwrap();

        let refusal = claim
            .bind_with(|_| Err(std::io::Error::other("chmod refused")))
            .expect_err("a failed chmod fails the bind");

        assert!(
            format!("{refusal:#}").contains("setting permissions on"),
            "{refusal:#}"
        );
        assert!(
            std::fs::symlink_metadata(&path).is_ok(),
            "the socket was bound"
        );
        drop(claim);
        assert!(
            std::fs::symlink_metadata(&path).is_err(),
            "dropping the claim removes the socket it bound"
        );
    }

    /// A socket the probe cannot connect to for a reason other than nobody
    /// listening -- here its mode forbids connecting -- is reported as a
    /// failed probe: not replaced as stale, and not taken as answering.
    /// Skipped as root, whom the mode does not stop.
    #[tokio::test]
    async fn a_socket_the_probe_cannot_reach_is_reported_not_replaced() {
        if effective_uid() == 0 {
            eprintln!("skip: root may connect whatever the socket's mode");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        drop(std::os::unix::net::UnixListener::bind(&path).unwrap());
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();
        let theirs = inode(&path);

        let Err(refusal) = SocketClaim::claim(&path).await else {
            panic!("a socket that cannot be probed must be refused");
        };

        assert!(
            format!("{refusal:#}").contains("probing existing socket"),
            "{refusal:#}"
        );
        assert!(in_use_path(&refusal).is_none(), "not in use: {refusal:#}");
        assert_eq!(inode(&path), theirs, "the socket is left in place");
    }

    /// `socket_parent` gives a bare socket file name, such as
    /// `unix:rocmdashd.sock`, no directory to check -- its parent is the
    /// empty path, which the check refuses -- and every other path its own
    /// directory. This drives the helper only: `serve_unix`'s use of it for a
    /// bare name is not exercised, since that would change the process-wide
    /// current directory under tests running in parallel.
    #[test]
    fn a_bare_socket_name_has_no_directory_to_check() {
        use super::socket_parent;

        assert_eq!(socket_parent(Path::new("rocmdashd.sock")), None);
        assert!(
            check_socket_dir(Path::new(""), effective_uid()).is_err(),
            "the empty path is no directory to check"
        );
        assert_eq!(
            socket_parent(Path::new("/run/user/1000/rocmdashd.sock")),
            Some(Path::new("/run/user/1000"))
        );
        assert_eq!(
            socket_parent(Path::new("/rocmdashd.sock")),
            Some(Path::new("/"))
        );
    }

    /// Unreadable path checks give the path up only five in a row, with the
    /// fifth check's error; a check that reads the path starts the count over.
    #[test]
    fn only_five_unreadable_path_checks_in_a_row_give_the_path_up() {
        use super::unreadable_streak;

        let mut streak = 0;
        for expected in 1..=4 {
            streak = unreadable_streak(streak, Err("unreadable")).expect("fewer than five");
            assert_eq!(streak, expected);
        }
        assert_eq!(
            unreadable_streak(streak, Err("fifth")),
            Err("fifth"),
            "the fifth gives up with its error"
        );
        assert_eq!(
            unreadable_streak::<&str>(streak, Ok(())),
            Ok(0),
            "a good check resets"
        );

        let mut streak = 0;
        for readable in [false, false, false, false, true, false, false, false, false] {
            let check = if readable { Ok(()) } else { Err("unreadable") };
            streak = unreadable_streak(streak, check).expect("never five in a row");
        }
        assert_eq!(streak, 4);
    }

    // ── The socket directory (ROCMAI-456) ─────────────────────────────

    /// A daemon with a runner that never stops, running as user `our_uid`.
    fn idle_seams(our_uid: u32) -> ServeSeams {
        ServeSeams {
            start_runner: idle_runner(),
            client_tasks: None,
            path_check_interval: super::PATH_CHECK_INTERVAL,
            our_uid,
        }
    }

    fn mode_of(path: &Path) -> u32 {
        std::fs::symlink_metadata(path)
            .unwrap_or_else(|e| panic!("{} should exist: {e}", path.display()))
            .permissions()
            .mode()
            & 0o777
    }

    /// A socket directory another user controls is refused before the daemon
    /// creates, opens, probes or removes anything in it, and the refusal is a
    /// failure, not "another daemon serves this path" -- which a dashboard
    /// would keep waiting on without a word.
    #[tokio::test]
    async fn a_socket_directory_another_user_owns_is_refused_before_anything_is_touched() {
        let dir = tempfile::tempdir().unwrap();
        let shared = dir.path().join("shared");
        std::fs::create_dir(&shared).unwrap();
        std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o755)).unwrap();
        let path = shared.join("rocmdashd.sock");
        stale_socket(&path);
        pin_inode(&path);
        let stale = inode(&path);
        // The directory is this user's; a daemon running as someone else
        // sees it as another user's.
        let someone_else = super::effective_uid().wrapping_add(1);

        let refusal =
            tokio::time::timeout(DEADLINE, start_serving(&path, idle_seams(someone_else)))
                .await
                .expect("a refused directory must not be served")
                .unwrap()
                .expect_err("another user's directory must be refused");

        assert!(in_use_path(&refusal).is_none(), "not in use: {refusal:#}");
        assert!(
            format!("{refusal:#}").contains(&unsecurable_socket_dir_context(&shared)),
            "the refusal must name the directory and why: {refusal:#}"
        );
        assert!(
            !socket_lock_path(&path).exists(),
            "no lock file may be created in a refused directory"
        );
        assert_eq!(inode(&path), stale, "nothing in it may be removed");
        assert_eq!(
            mode_of(&shared),
            0o755,
            "a refused directory is not changed"
        );
    }

    /// The lock is never taken through a symlink planted at its path: that
    /// would create or lock a file wherever the link points.
    #[tokio::test]
    async fn a_symlink_at_the_lock_path_is_not_followed() {
        let dir = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let target = elsewhere.path().join("planted");
        std::os::unix::fs::symlink(&target, socket_lock_path(&path)).unwrap();

        let refusal = SocketClaim::claim(&path)
            .await
            .expect_err("a symlinked lock must be refused");

        assert!(in_use_path(&refusal).is_none(), "{refusal:#}");
        assert!(
            !target.exists(),
            "the link's target must not be created: {refusal:#}"
        );
        assert!(
            std::fs::symlink_metadata(socket_lock_path(&path))
                .unwrap()
                .file_type()
                .is_symlink(),
            "the planted link is left as it was"
        );
    }

    /// A missing socket directory is created private, nested levels too, and
    /// the socket in it is private to its owner.
    #[tokio::test]
    async fn serving_creates_a_missing_socket_directory_privately() {
        let dir = tempfile::tempdir().unwrap();
        let socket_dir = dir.path().join("nested").join("telemetry");
        let path = socket_dir.join("rocmdashd.sock");
        let daemon = start_serving(&path, idle_seams(super::effective_uid()));
        let _attached = wait_until_serving(&path).await;

        assert_eq!(mode_of(&socket_dir), 0o700);
        assert_eq!(mode_of(&path), 0o600, "only the owner may connect");
        daemon.abort();
        let _ = daemon.await;
    }

    /// An existing socket directory this user owns that group or others have
    /// any access to is restricted to `0o700` when the daemon serves it, and
    /// the socket to `0o600`.
    #[tokio::test]
    async fn serving_restricts_a_socket_directory_it_owns() {
        let dir = tempfile::tempdir().unwrap();
        let socket_dir = dir.path().join("loose");
        std::fs::create_dir(&socket_dir).unwrap();
        std::fs::set_permissions(&socket_dir, std::fs::Permissions::from_mode(0o775)).unwrap();
        let path = socket_dir.join("rocmdashd.sock");
        let daemon = start_serving(&path, idle_seams(super::effective_uid()));
        let _attached = wait_until_serving(&path).await;

        assert_eq!(mode_of(&socket_dir), 0o700);
        assert_eq!(mode_of(&path), 0o600, "only the owner may connect");
        daemon.abort();
        let _ = daemon.await;
    }

    /// A parent that cannot be inspected (here: a path through a regular
    /// file) is reported as such, before anything is attempted in it.
    #[tokio::test]
    async fn a_socket_directory_that_cannot_be_inspected_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a-file");
        std::fs::write(&file, "keep me").unwrap();
        let parent = file.join("telemetry");
        let path = parent.join("rocmdashd.sock");

        let refusal = tokio::time::timeout(
            DEADLINE,
            start_serving(&path, idle_seams(super::effective_uid())),
        )
        .await
        .expect("an unusable directory must not be served")
        .unwrap()
        .expect_err("a directory that cannot be inspected must be refused");

        assert!(
            format!("{refusal:#}")
                .contains(&format!("inspecting socket directory {}", parent.display())),
            "{refusal:#}"
        );
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "keep me");
    }

    /// Something that is not a directory where the socket's directory goes is
    /// refused as such -- before whose it is matters -- and left alone.
    #[tokio::test]
    async fn a_socket_directory_that_is_not_a_directory_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let parent = dir.path().join("telemetry");
        std::fs::write(&parent, "keep me").unwrap();
        let path = parent.join("rocmdashd.sock");
        let someone_else = super::effective_uid().wrapping_add(1);

        let refusal =
            tokio::time::timeout(DEADLINE, start_serving(&path, idle_seams(someone_else)))
                .await
                .expect("an unusable directory must not be served")
                .unwrap()
                .expect_err("a file where the directory goes must be refused");

        assert!(
            format!("{refusal:#}").contains(&format!(
                "socket directory {} is not a directory",
                parent.display()
            )),
            "{refusal:#}"
        );
        assert!(in_use_path(&refusal).is_none(), "{refusal:#}");
        assert_eq!(std::fs::read_to_string(&parent).unwrap(), "keep me");
        assert_eq!(
            std::fs::read_dir(dir.path()).unwrap().count(),
            1,
            "nothing may be created beside it"
        );
    }

    /// A socket directory reached through a symlink is judged by the directory
    /// the link resolves to -- the one the socket and its lock would really be
    /// created in: a link to a writable directory of ours is served, and that
    /// directory (not the link) is the one made private.
    #[tokio::test]
    async fn a_symlinked_socket_directory_is_judged_by_its_target() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real");
        std::fs::create_dir(&real).unwrap();
        std::fs::set_permissions(&real, std::fs::Permissions::from_mode(0o777)).unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let path = link.join("rocmdashd.sock");

        let daemon = start_serving(&path, idle_seams(super::effective_uid()));
        let _attached = wait_until_serving(&path).await;

        assert!(real.join("rocmdashd.sock").exists());
        assert_eq!(mode_of(&real), 0o700, "the target is made private");
        assert!(
            std::fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink(),
            "the link stays a link"
        );
        daemon.abort();
        let _ = daemon.await;
    }

    /// A link to a directory another user owns is refused as theirs -- its
    /// target decides, not the link.
    #[tokio::test]
    async fn a_symlink_to_another_users_socket_directory_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let real = dir.path().join("real");
        std::fs::create_dir(&real).unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let path = link.join("rocmdashd.sock");
        let someone_else = super::effective_uid().wrapping_add(1);

        let refusal =
            tokio::time::timeout(DEADLINE, start_serving(&path, idle_seams(someone_else)))
                .await
                .expect("a refused directory must not be served")
                .unwrap()
                .expect_err("another user's directory must be refused");

        assert!(
            format!("{refusal:#}").contains(&unsecurable_socket_dir_context(&link)),
            "refused as another user's, not as something else: {refusal:#}"
        );
        assert!(!real.join("rocmdashd.sock.lock").exists());
    }

    /// A socket directory of ours that group or others have any access to is
    /// made private before the claim; checking an acceptable directory again
    /// changes nothing -- a private mode set between the two checks survives
    /// the second.
    #[test]
    fn a_loose_socket_directory_of_ours_is_made_private_before_the_claim() {
        let dir = tempfile::tempdir().unwrap();
        let socket_dir = dir.path().join("loose");
        std::fs::create_dir(&socket_dir).unwrap();
        std::fs::set_permissions(&socket_dir, std::fs::Permissions::from_mode(0o777)).unwrap();

        super::check_socket_dir(&socket_dir, super::effective_uid())
            .expect("our own directory is usable");

        assert_eq!(mode_of(&socket_dir), 0o700);
        assert_eq!(
            std::fs::read_dir(&socket_dir).unwrap().count(),
            0,
            "the check creates nothing"
        );
        std::fs::set_permissions(&socket_dir, std::fs::Permissions::from_mode(0o500)).unwrap();
        super::check_socket_dir(&socket_dir, super::effective_uid())
            .expect("checking an acceptable directory again is fine");
        assert_eq!(mode_of(&socket_dir), 0o500, "no chmod the second time");
    }

    /// At the call site: a directory of ours that group or others have any
    /// access to is already private by the time the claim works in it. Here
    /// the claim then fails (a symlink sits at the lock path), and the
    /// directory is 0700 anyway.
    #[tokio::test]
    async fn a_loose_socket_directory_is_tightened_before_the_claim_runs() {
        let dir = tempfile::tempdir().unwrap();
        let socket_dir = dir.path().join("loose");
        std::fs::create_dir(&socket_dir).unwrap();
        std::fs::set_permissions(&socket_dir, std::fs::Permissions::from_mode(0o777)).unwrap();
        let path = socket_dir.join("rocmdashd.sock");
        std::os::unix::fs::symlink(dir.path().join("elsewhere"), socket_lock_path(&path)).unwrap();

        let refusal = tokio::time::timeout(
            DEADLINE,
            start_serving(&path, idle_seams(super::effective_uid())),
        )
        .await
        .expect("the claim must fail, not serve")
        .unwrap()
        .expect_err("a symlinked lock must be refused");

        assert!(
            format!("{refusal:#}").contains("opening socket lock"),
            "it is the claim that failed: {refusal:#}"
        );
        assert_eq!(mode_of(&socket_dir), 0o700);
    }

    /// The decision is the directory owner's: a writable directory of another
    /// user is refused, never chmodded.
    #[tokio::test]
    async fn another_users_writable_socket_directory_is_refused_not_chmodded() {
        let dir = tempfile::tempdir().unwrap();
        let shared = dir.path().join("theirs");
        std::fs::create_dir(&shared).unwrap();
        std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o775)).unwrap();
        let path = shared.join("rocmdashd.sock");
        let someone_else = super::effective_uid().wrapping_add(1);

        let refusal =
            tokio::time::timeout(DEADLINE, start_serving(&path, idle_seams(someone_else)))
                .await
                .expect("a refused directory must not be served")
                .unwrap()
                .expect_err("another user's directory must be refused");

        assert!(
            format!("{refusal:#}").contains(&unsecurable_socket_dir_context(&shared)),
            "{refusal:#}"
        );
        assert_eq!(mode_of(&shared), 0o775, "it is not chmodded");
        assert!(!socket_lock_path(&path).exists());
    }

    /// A sticky directory is never chmodded, ours included: it is shared on
    /// purpose, and only an entry's owner may remove the entry, so our `0o600`
    /// socket and lock cannot be replaced in it.
    #[tokio::test]
    async fn a_sticky_socket_directory_is_served_as_it_is() {
        let dir = tempfile::tempdir().unwrap();
        let shared = dir.path().join("shared");
        std::fs::create_dir(&shared).unwrap();
        std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o1777)).unwrap();
        let path = shared.join("rocmdashd.sock");

        let daemon = start_serving(&path, idle_seams(super::effective_uid()));
        let _attached = wait_until_serving(&path).await;

        assert_eq!(
            std::fs::metadata(&shared).unwrap().permissions().mode() & 0o7777,
            0o1777,
            "a sticky directory keeps its mode"
        );
        daemon.abort();
        let _ = daemon.await;
    }

    /// `check_socket_dir` says a sticky directory others may write in is
    /// shared, which is what the warning after the claim keys on, and that a
    /// private one is not.
    #[test]
    fn a_sticky_socket_directory_is_reported_as_shared() {
        let dir = tempfile::tempdir().unwrap();
        let shared = dir.path().join("shared");
        let private = dir.path().join("private");
        for (path, mode) in [(&shared, 0o1777), (&private, 0o700)] {
            std::fs::create_dir(path).unwrap();
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
        }

        assert!(check_socket_dir(&shared, effective_uid()).unwrap());
        assert!(!check_socket_dir(&private, effective_uid()).unwrap());
    }

    /// A writer the tests' own `tracing` subscriber logs into.
    #[derive(Clone, Default)]
    struct Captured(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    /// Install a subscriber at `level` for this test's thread and return what
    /// it writes. The code under test must run on this thread (a
    /// `#[tokio::test]` runtime is single-threaded).
    ///
    /// It first makes sure two more dispatchers are registered for the rest
    /// of the process. tracing-core caches each callsite's interest when the
    /// callsite is first hit, and while at most one dispatcher is registered
    /// (`callsite::dispatchers::Rebuilder::JustOne` in tracing-core) it asks
    /// only the default subscriber *of the thread that hit it*. A test running
    /// in parallel with no subscriber would then cache the callsite as never
    /// enabled, and this subscriber would miss it. With more than one
    /// registered, tracing-core asks every live dispatcher, under the lock a
    /// new dispatcher takes to rebuild the cache, so the interest cached is
    /// the same on every thread. Mirrors `log_at` in the `rocm` app's
    /// `dash_daemon.rs` tests.
    fn log_at(level: tracing::Level) -> (Captured, tracing::subscriber::DefaultGuard) {
        static KEEP_INTEREST_SHARED: std::sync::OnceLock<[tracing::Dispatch; 2]> =
            std::sync::OnceLock::new();
        KEEP_INTEREST_SHARED.get_or_init(|| {
            std::array::from_fn(|_| {
                tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default())
            })
        });
        let captured = Captured::default();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(level)
            .with_ansi(false)
            .with_writer(captured.clone())
            .finish();
        (captured, tracing::subscriber::set_default(subscriber))
    }

    impl std::io::Write for Captured {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Captured {
        type Writer = Self;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// The shared-sticky-directory warning is given once, by the daemon that
    /// claims the path -- not by one that finds the path owned and retries.
    /// The daemons run on this test's thread (`#[tokio::test]` is
    /// single-threaded), where the subscriber is installed.
    #[tokio::test]
    async fn the_shared_directory_warning_comes_once_from_the_daemon_that_claims_the_path() {
        let (captured, _guard) = log_at(tracing::Level::WARN);
        let warnings = || {
            String::from_utf8_lossy(&captured.0.lock().unwrap())
                .matches("shared sticky directory")
                .count()
        };
        let dir = tempfile::tempdir().unwrap();
        let shared = dir.path().join("shared");
        std::fs::create_dir(&shared).unwrap();
        std::fs::set_permissions(&shared, std::fs::Permissions::from_mode(0o1777)).unwrap();
        let path = shared.join("rocmdashd.sock");

        let daemon = start_serving(&path, idle_seams(super::effective_uid()));
        let _attached = wait_until_serving(&path).await;
        assert_eq!(warnings(), 1, "the daemon that claimed the path warns once");

        let refusal = tokio::time::timeout(
            DEADLINE,
            start_serving(&path, idle_seams(super::effective_uid())),
        )
        .await
        .expect("a second daemon must give up")
        .unwrap()
        .expect_err("the path is owned");
        assert_eq!(in_use_holder(&refusal), Some(SocketHolder::LockHolder));
        assert_eq!(
            warnings(),
            1,
            "a daemon that found the path owned does not warn"
        );

        daemon.abort();
        let _ = daemon.await;
    }

    /// A socket directory that had to be created is judged afterwards like
    /// any existing one: `create` also succeeds when someone else made it
    /// first, so what is there is checked again. Here the daemon runs as a
    /// different uid (`ServeSeams::our_uid`), so the directory it just made
    /// counts as another user's and is refused. (The race itself -- another
    /// user creating it between the stat and the mkdir -- is not driven.)
    #[tokio::test]
    async fn a_socket_directory_just_created_is_judged_like_an_existing_one() {
        let dir = tempfile::tempdir().unwrap();
        let socket_dir = dir.path().join("made");
        let path = socket_dir.join("rocmdashd.sock");
        let someone_else = super::effective_uid().wrapping_add(1);

        let refusal =
            tokio::time::timeout(DEADLINE, start_serving(&path, idle_seams(someone_else)))
                .await
                .expect("a refused directory must not be served")
                .unwrap()
                .expect_err("a directory that is not ours must be refused");

        assert!(
            format!("{refusal:#}").contains(&unsecurable_socket_dir_context(&socket_dir)),
            "{refusal:#}"
        );
        assert!(socket_dir.is_dir(), "it was created");
        assert!(!socket_lock_path(&path).exists());
    }
}
