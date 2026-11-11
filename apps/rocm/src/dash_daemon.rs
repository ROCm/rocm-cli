// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The embedded telemetry daemon behind a live `rocm dash`.
//!
//! Every live dashboard of a user points at the same unix socket, and the
//! daemon runs inside a dashboard process rather than on its own. So each live
//! dashboard runs a *keeper*: a task that tries to serve the socket, and while
//! another dashboard's daemon owns it, keeps trying a few times a second. The
//! daemon's own socket claim (`rocm_dash_daemon::server`) lets exactly one of
//! them bind and refuses the rest with `SocketInUse`, which is what the keeper
//! retries on. When the owning dashboard closes -- or is killed -- one of the
//! dashboards still open takes over within a poll, and every client's
//! reconnect loop finds the socket served again.
//!
//! Unix only: there is no daemon socket on Windows, where the dashboard
//! collects in process.

#[cfg(unix)]
use std::time::Duration;

use rocm_core::{AppPaths, RocmCliConfig};
use tokio::task::JoinHandle;

/// How often a keeper whose socket another dashboard owns tries to take over.
#[cfg(unix)]
const TAKEOVER_POLL: Duration = Duration::from_millis(250);

/// First wait after the daemon failed for a reason other than another owner
/// (doubling up to [`FAILURE_BACKOFF_MAX`]), so a path the daemon refuses --
/// say a regular file -- is retried and logged without spinning.
#[cfg(unix)]
const FAILURE_BACKOFF_START: Duration = Duration::from_secs(1);
#[cfg(unix)]
const FAILURE_BACKOFF_MAX: Duration = Duration::from_secs(30);

/// How long the dashboard waits for the socket to answer before it draws
/// anyway; the TUI client keeps reconnecting on its own after that.
#[cfg(unix)]
const READY_WAIT: Duration = Duration::from_secs(5);

/// Start this dashboard's keeper for a local `unix:` socket and wait (at most
/// [`READY_WAIT`]) until the socket answers, whoever serves it.
///
/// Returns `None`, starting nothing, for a non-unix `connect` target and on
/// Windows. Stop the returned task with [`stop_keeper`].
// One function with `#[cfg]` blocks inside, not a cfg-split pair: the unix
// body awaits, the Windows body does not. Callers `.await` this either way,
// so dropping `async` off unix would not compile.
#[allow(clippy::unused_async)]
pub(crate) async fn spawn_keeper(
    connect: &str,
    config: &RocmCliConfig,
    paths: &AppPaths,
) -> Option<JoinHandle<()>> {
    #[cfg(unix)]
    {
        // Only a LOCAL unix-socket daemon is ours to keep.
        let target = connect.strip_prefix("unix:")?;
        let opts = crate::dash::runner_options(config, paths, false);
        let handle = tokio::spawn(keep_daemon(connect.to_string(), opts, TAKEOVER_POLL));
        // Poll until something accepts connections; a fixed sleep is
        // race-prone on slow or loaded systems.
        let deadline = tokio::time::Instant::now() + READY_WAIT;
        while tokio::net::UnixStream::connect(target).await.is_err()
            && tokio::time::Instant::now() < deadline
        {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        Some(handle)
    }
    #[cfg(windows)]
    {
        let _ = (connect, config, paths);
        None
    }
}

/// Stop a keeper and wait for it, so the daemon it may own has removed its
/// socket (only if it still is that daemon's socket) and released the path
/// before the runtime is dropped. A keeper that never owned the socket
/// touches nothing on the way out.
pub(crate) async fn stop_keeper(handle: JoinHandle<()>) {
    handle.abort();
    let _ = handle.await;
}

/// Serve `listen` whenever no other daemon does, until aborted.
///
/// Logs through `tracing` only: the dashboard owns the terminal while this
/// runs, so a stderr write would land inside the TUI's screen.
#[cfg(unix)]
async fn keep_daemon(
    listen: String,
    opts: rocm_dash_daemon::runner::RunnerOptions,
    poll: Duration,
) {
    use rocm_dash_daemon::server::SocketInUse;

    let mut backoff = FAILURE_BACKOFF_START;
    loop {
        match rocm_dash_daemon::server::run(&listen, opts.clone()).await {
            Ok(()) => break,
            Err(e) if e.downcast_ref::<SocketInUse>().is_some() => {
                backoff = FAILURE_BACKOFF_START;
                tokio::time::sleep(poll).await;
            }
            Err(e) => {
                tracing::warn!(
                    socket = %listen,
                    error = %format!("{e:#}"),
                    retry_in = ?backoff,
                    "embedded telemetry daemon could not serve its socket; retrying"
                );
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(FAILURE_BACKOFF_MAX);
            }
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::MetadataExt;
    use std::path::Path;
    use std::time::Duration;

    use rocm_dash_core::protocol::Event;
    use rocm_dash_daemon::runner::RunnerOptions;
    use rocm_dash_daemon::transport::read_line;
    use tokio::io::BufReader;
    use tokio::task::JoinHandle;

    use super::{keep_daemon, stop_keeper};

    /// Fast enough that a takeover is quick in a test, slow enough not to spin.
    const POLL: Duration = Duration::from_millis(50);
    const DEADLINE: Duration = Duration::from_secs(10);

    fn keeper(path: &Path) -> JoinHandle<()> {
        let listen = format!("unix:{}", path.display());
        tokio::spawn(keep_daemon(listen, RunnerOptions::default(), POLL))
    }

    /// Connect and read the greeting every daemon sends a new client.
    async fn greeted(path: &Path) -> Result<tokio::net::UnixStream, String> {
        let mut stream = tokio::net::UnixStream::connect(path)
            .await
            .map_err(|e| e.to_string())?;
        let mut reader = BufReader::new(&mut stream);
        match tokio::time::timeout(DEADLINE, read_line::<_, Event>(&mut reader)).await {
            Ok(Ok(Some(Event::Welcome { .. }))) => Ok(stream),
            other => Err(format!("expected a Welcome, got {other:?}")),
        }
    }

    async fn wait_until_served(path: &Path, what: &str) -> tokio::net::UnixStream {
        let deadline = tokio::time::Instant::now() + DEADLINE;
        loop {
            match greeted(path).await {
                Ok(stream) => return stream,
                Err(e) => assert!(
                    tokio::time::Instant::now() < deadline,
                    "{what}: nothing served {} within {DEADLINE:?}: {e}",
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

    /// Keep the inode at `path` allocated through a second name, so a socket
    /// bound at `path` after it is unlinked cannot be handed the same inode
    /// number (ext4 often reuses the one it just freed) and pass for the old
    /// socket in an `inode(..)` comparison.
    fn pin_inode(path: &Path) {
        let mut pinned = path.as_os_str().to_owned();
        pinned.push(".pinned");
        std::fs::hard_link(path, std::path::PathBuf::from(pinned))
            .expect("pin the inode with a hard link");
    }

    /// The reported bug: with two dashboards open, closing the one whose
    /// process hosted the daemon left the other disconnected for good.
    #[tokio::test]
    async fn a_surviving_session_takes_over_when_the_owner_leaves() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let first = keeper(&path);
        let _attached = wait_until_served(&path, "the first dashboard's daemon").await;
        pin_inode(&path);
        let first_socket = inode(&path);
        let second = keeper(&path);
        // Let the second keeper find the path owned and start polling.
        tokio::time::sleep(POLL * 4).await;
        assert!(
            !second.is_finished(),
            "a keeper that finds the socket owned must keep waiting, not give up"
        );
        assert_eq!(
            inode(&path),
            first_socket,
            "the second keeper must not take a live socket"
        );

        stop_keeper(first).await;

        let _reattached = wait_until_served(&path, "the takeover after the owner left").await;
        assert_ne!(
            inode(&path),
            first_socket,
            "the socket must now be the surviving dashboard's own"
        );
        stop_keeper(second).await;
        assert!(
            !path.exists(),
            "the last owner removes its socket when it stops"
        );
    }

    /// Closing a dashboard that never owned the daemon must not disturb the
    /// one that does.
    #[tokio::test]
    async fn a_session_that_never_owned_the_daemon_leaves_it_running() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let owner = keeper(&path);
        let _attached = wait_until_served(&path, "the owner's daemon").await;
        pin_inode(&path);
        let socket = inode(&path);
        let bystander = keeper(&path);
        // Let the bystander try, and be refused, at least once.
        tokio::time::sleep(POLL * 4).await;

        stop_keeper(bystander).await;

        assert_eq!(
            inode(&path),
            socket,
            "the owner's socket must be left in place"
        );
        greeted(&path)
            .await
            .expect("the owner's daemon must still greet new clients");
        stop_keeper(owner).await;
    }
}
