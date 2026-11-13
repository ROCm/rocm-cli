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

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
#[cfg(unix)]
use std::time::Duration;

use rocm_core::{AppPaths, RocmCliConfig};
use tokio::task::JoinHandle;

/// How often a keeper whose socket another dashboard owns tries to take over.
#[cfg(unix)]
const TAKEOVER_POLL: Duration = Duration::from_millis(250);

/// How often a keeper retries while something without the ownership lock (a
/// daemon of an older build, or another program) answers at the socket.
/// Nothing announces when that goes away, and each try connects to it, so
/// this is slower than [`TAKEOVER_POLL`].
#[cfg(unix)]
const UNLOCKED_LISTENER_POLL: Duration = Duration::from_secs(5);

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

/// The failure a keeper's daemon was stuck in: the socket it could not serve,
/// why, how many times it had tried, and whether the warning it logged for it
/// was recorded anywhere.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KeeperFailure {
    pub(crate) socket: PathBuf,
    pub(crate) error: String,
    /// Failed runs in a row up to this one.
    pub(crate) failed_attempts: u64,
    /// Whether the active `tracing` subscriber accepted the keeper's warning.
    /// The only subscriber `rocm` installs writes the client log file, so
    /// `false` means the log has nothing about this (logging did not start,
    /// or `RUST_LOG` filters it out).
    pub(crate) logged: bool,
}

/// A running keeper, and the failure its daemon is currently stuck in, if any.
#[derive(Debug)]
// Only ever built on unix (see `spawn_keeper`).
#[cfg_attr(windows, allow(dead_code))]
pub(crate) struct Keeper {
    socket: PathBuf,
    handle: JoinHandle<()>,
    failure: Arc<Mutex<Option<KeeperFailure>>>,
}

/// Start this dashboard's keeper for a local `unix:` socket and wait (at most
/// [`READY_WAIT`]) until the socket answers, whoever serves it.
///
/// Returns `None`, starting nothing, for a non-unix `connect` target and on
/// Windows. Stop the returned keeper with [`stop_keeper`].
// One function with `#[cfg]` blocks inside, not a cfg-split pair: the unix
// body awaits, the Windows body does not. Callers `.await` this either way,
// so dropping `async` off unix would not compile.
#[allow(clippy::unused_async)]
pub(crate) async fn spawn_keeper(
    connect: &str,
    config: &RocmCliConfig,
    paths: &AppPaths,
) -> Option<Keeper> {
    #[cfg(unix)]
    {
        // Only a LOCAL unix-socket daemon is ours to keep.
        let target = connect.strip_prefix("unix:")?;
        let opts = crate::dash::runner_options(config, paths, false);
        let keeper = start(Path::new(target), opts, TAKEOVER_POLL);
        // Poll until something accepts connections; a fixed sleep is
        // race-prone on slow or loaded systems. Stop waiting as soon as the
        // daemon has failed: nothing will answer, and the TUI client keeps
        // reconnecting on its own.
        let deadline = tokio::time::Instant::now() + READY_WAIT;
        while tokio::net::UnixStream::connect(target).await.is_err()
            && tokio::time::Instant::now() < deadline
            && keeper.failure.lock().is_ok_and(|slot| slot.is_none())
        {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        Some(keeper)
    }
    #[cfg(windows)]
    {
        let _ = (connect, config, paths);
        None
    }
}

#[cfg(unix)]
fn start(socket: &Path, opts: rocm_dash_daemon::runner::RunnerOptions, poll: Duration) -> Keeper {
    let failure = Arc::new(Mutex::new(None));
    let handle = tokio::spawn(keep_daemon(
        socket.to_path_buf(),
        opts,
        poll,
        Arc::clone(&failure),
    ));
    Keeper {
        socket: socket.to_path_buf(),
        handle,
        failure,
    }
}

/// Stop a keeper and wait for it, so the daemon it may own has removed its
/// socket (only if it still is that daemon's socket) and released the path
/// before the runtime is dropped. A keeper that never owned the socket
/// touches nothing on the way out.
///
/// Returns the failure the keeper's daemon was stuck in when it was stopped,
/// for the caller to report once the terminal is its own again. A keeper that
/// had panicked is logged and reported the same way.
pub(crate) async fn stop_keeper(keeper: Keeper) -> Option<KeeperFailure> {
    let Keeper {
        socket,
        handle,
        failure,
    } = keeper;
    handle.abort();
    let panicked = match handle.await {
        Err(e) if e.is_panic() => {
            tracing::error!(error = %e, "the embedded telemetry daemon's keeper panicked");
            Some(e.to_string())
        }
        _ => None,
    };
    let recorded = failure.lock().map_or(None, |slot| slot.clone());
    match (recorded, panicked) {
        (Some(failure), _) => Some(failure),
        (None, Some(error)) => Some(KeeperFailure {
            socket,
            error: format!("its keeper stopped unexpectedly: {error}"),
            failed_attempts: 0,
            logged: tracing::enabled!(tracing::Level::ERROR),
        }),
        (None, None) => None,
    }
}

/// How the report opens when the error names the socket path itself. dash-53
/// (`dash_steps.rs`), which cannot import it, matches its tail from
/// "telemetry daemon".
pub(crate) const REPORT_COULD_NOT_START: &str =
    "rocm: the dashboard's telemetry daemon could not start:";

/// The report's line pointing at the client log. dash-53 matches it without
/// its leading `rocm: `.
pub(crate) const REPORT_LOG_LINE: &str = "rocm: each failed attempt is logged in the files under";

/// The warning the keeper logs for each failed run. dash-53 looks for the
/// fragment "could not serve its socket" from it in the client log.
#[cfg(unix)]
pub(crate) const KEEPER_WARNING: &str =
    "embedded telemetry daemon could not serve its socket; retrying";

/// The lines `rocm dash` prints after the dashboard closes when its daemon
/// was stuck in `failure`; built from the recorded failure itself, so it says
/// what actually went wrong, and names the log only when the warning reached
/// it.
pub(crate) fn failure_report(failure: &KeeperFailure, log_dir: &Path) -> String {
    let socket = failure.socket.display().to_string();
    // Most errors already name the path; say it once.
    let what = if failure.error.contains(&socket) {
        format!("{REPORT_COULD_NOT_START} {}", failure.error)
    } else {
        format!(
            "rocm: the dashboard's telemetry daemon could not serve {socket}: {}",
            failure.error
        )
    };
    let tries = match failure.failed_attempts {
        0 | 1 => String::new(),
        n => format!(" ({n} failed attempts)"),
    };
    if failure.logged {
        format!("{what}{tries}\n{REPORT_LOG_LINE} {}", log_dir.display())
    } else {
        format!("{what}{tries}")
    }
}

/// What a keeper does after one run of the daemon ended.
#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NextStep {
    /// Another daemon serves the socket: try again after this long.
    OwnedElsewhere(Duration),
    /// This daemon's socket was removed or replaced while it served: claim
    /// the path again after this long. Not a failure.
    Reclaim(Duration),
    /// The daemon failed: record it, log it, and try again after this long.
    Failed(Duration),
}

/// A keeper's run of failures: how long to wait after the next one, and how
/// many failed in a row. Both start over whenever the daemon serves, another
/// daemon is found serving, or the socket was taken from a serving daemon.
#[cfg(unix)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Retry {
    backoff: Duration,
    failed_in_a_row: u64,
}

#[cfg(unix)]
impl Retry {
    const fn new() -> Self {
        Self {
            backoff: FAILURE_BACKOFF_START,
            failed_in_a_row: 0,
        }
    }
}

/// Decide the wait after one run of the daemon, advancing `retry`: on a
/// failure the backoff doubles (capped) and the count goes up; on anything
/// else both start over. A run that `served` before it ended ended the run of
/// failures before it, so a failure after it is counted from one again.
#[cfg(unix)]
fn next_step(
    outcome: &anyhow::Result<()>,
    served: bool,
    retry: &mut Retry,
    poll: Duration,
) -> NextStep {
    use rocm_dash_daemon::server::{SocketHolder, SocketInUse, SocketReplaced};

    if served {
        *retry = Retry::new();
    }

    if outcome
        .as_ref()
        .err()
        .is_some_and(|e| e.downcast_ref::<SocketReplaced>().is_some())
    {
        *retry = Retry::new();
        return NextStep::Reclaim(poll);
    }
    let in_use = outcome
        .as_ref()
        .err()
        .and_then(|e| e.downcast_ref::<SocketInUse>());
    if let Some(in_use) = in_use {
        *retry = Retry::new();
        return NextStep::OwnedElsewhere(match in_use.holder {
            SocketHolder::LockHolder => poll,
            SocketHolder::UnlockedListener => UNLOCKED_LISTENER_POLL.max(poll),
        });
    }
    let wait = retry.backoff;
    retry.backoff = (retry.backoff * 2).min(FAILURE_BACKOFF_MAX);
    retry.failed_in_a_row += 1;
    NextStep::Failed(wait)
}

/// Serve `socket` whenever no other daemon does, until aborted, keeping
/// `failure` set to the failure the daemon is stuck in: set when a run fails,
/// cleared only once a run is serving or another daemon is found serving.
///
/// Logs through `tracing` only: the dashboard owns the terminal while this
/// runs, so a stderr write would land inside the TUI's screen.
#[cfg(unix)]
async fn keep_daemon(
    socket: PathBuf,
    opts: rocm_dash_daemon::runner::RunnerOptions,
    poll: Duration,
    failure: Arc<Mutex<Option<KeeperFailure>>>,
) {
    fn set(slot: &Mutex<Option<KeeperFailure>>, value: Option<KeeperFailure>) {
        if let Ok(mut slot) = slot.lock() {
            *slot = value;
        }
    }
    let listen = format!("unix:{}", socket.display());
    let mut retry = Retry::new();
    loop {
        let serving = Arc::clone(&failure);
        let served = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ready = Arc::clone(&served);
        let outcome = rocm_dash_daemon::server::run_with_ready(&listen, opts.clone(), move || {
            ready.store(true, std::sync::atomic::Ordering::SeqCst);
            set(&serving, None);
        })
        .await;
        let served = served.load(std::sync::atomic::Ordering::SeqCst);
        match next_step(&outcome, served, &mut retry, poll) {
            NextStep::OwnedElsewhere(wait) => {
                set(&failure, None);
                tokio::time::sleep(wait).await;
            }
            NextStep::Reclaim(wait) => tokio::time::sleep(wait).await,
            NextStep::Failed(wait) => {
                // `run` serves until it fails, so returning at all is one.
                let error = outcome.err().map_or_else(
                    || "the daemon stopped without reporting an error".to_string(),
                    |e| format!("{e:#}"),
                );
                let logged = tracing::enabled!(tracing::Level::WARN);
                tracing::warn!(
                    socket = %socket.display(),
                    error = %error,
                    failed_in_a_row = retry.failed_in_a_row,
                    retry_in = ?wait,
                    "{KEEPER_WARNING}"
                );
                set(
                    &failure,
                    Some(KeeperFailure {
                        socket: socket.clone(),
                        error,
                        failed_attempts: retry.failed_in_a_row,
                        logged,
                    }),
                );
                tokio::time::sleep(wait).await;
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

    use super::{
        FAILURE_BACKOFF_MAX, KEEPER_WARNING, Keeper, NextStep, REPORT_COULD_NOT_START,
        REPORT_LOG_LINE, Retry, UNLOCKED_LISTENER_POLL, failure_report, next_step, spawn_keeper,
        start, stop_keeper,
    };
    use rocm_dash_daemon::server::NOT_A_SOCKET;

    /// Fast enough that a takeover is quick in a test, slow enough not to spin.
    const POLL: Duration = Duration::from_millis(50);
    const DEADLINE: Duration = Duration::from_secs(10);

    fn keeper(path: &Path) -> Keeper {
        start(path, RunnerOptions::default(), POLL)
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
            !second.handle.is_finished(),
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

        let reported = stop_keeper(bystander).await;

        assert_eq!(reported, None, "finding the socket served is not a failure");
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

    /// The waits a keeper takes: a failure backs off from 1 s, doubling to a
    /// 30 s cap, so a path the daemon refuses is retried without spinning;
    /// finding another daemon serving resets that and polls instead -- more
    /// slowly when what serves holds no lock, since every poll connects to it.
    #[test]
    fn a_failing_daemon_backs_off_and_finding_an_owner_resets_it() {
        use rocm_dash_daemon::server::{SocketHolder, SocketInUse};

        let poll = Duration::from_millis(250);
        let failed = || -> anyhow::Result<()> { Err(anyhow::anyhow!("not a socket")) };
        let in_use = |holder| -> anyhow::Result<()> {
            Err(SocketInUse {
                path: "/run/x.sock".into(),
                holder,
            }
            .into())
        };
        let mut retry = Retry::new();
        let waits: Vec<_> = (0..7)
            .map(|_| next_step(&failed(), false, &mut retry, poll))
            .collect();
        let secs = |s| NextStep::Failed(Duration::from_secs(s));
        assert_eq!(
            waits,
            [
                secs(1),
                secs(2),
                secs(4),
                secs(8),
                secs(16),
                secs(30),
                secs(30)
            ]
        );
        assert_eq!(
            next_step(&Ok(()), false, &mut retry, poll),
            NextStep::Failed(FAILURE_BACKOFF_MAX),
            "a daemon that stops without an error has still stopped"
        );

        assert_eq!(
            next_step(&in_use(SocketHolder::LockHolder), false, &mut retry, poll),
            NextStep::OwnedElsewhere(poll)
        );
        assert_eq!(
            next_step(&failed(), false, &mut retry, poll),
            secs(1),
            "finding an owner starts the failure backoff over"
        );
        assert_eq!(
            next_step(
                &in_use(SocketHolder::UnlockedListener),
                false,
                &mut retry,
                poll
            ),
            NextStep::OwnedElsewhere(UNLOCKED_LISTENER_POLL)
        );
        next_step(&failed(), false, &mut retry, poll);
        let replaced: anyhow::Result<()> = Err(rocm_dash_daemon::server::SocketReplaced {
            path: "/run/x.sock".into(),
        }
        .into());
        assert_eq!(
            next_step(&replaced, false, &mut retry, poll),
            NextStep::Reclaim(poll),
            "a socket taken from a serving daemon is reclaimed at once, not reported"
        );
        assert_eq!(next_step(&failed(), false, &mut retry, poll), secs(1));

        // Only failures in a row are counted: polls that find an owner, however
        // many, are not attempts.
        let mut retry = Retry::new();
        next_step(&failed(), false, &mut retry, poll);
        next_step(&failed(), false, &mut retry, poll);
        for _ in 0..5 {
            next_step(&in_use(SocketHolder::LockHolder), false, &mut retry, poll);
        }
        next_step(&failed(), false, &mut retry, poll);
        assert_eq!(retry.failed_in_a_row, 1);
        next_step(&failed(), false, &mut retry, poll);
        assert_eq!(retry.failed_in_a_row, 2);

        // A run that served before it stopped starts the backoff and the
        // count over, however long the failures before it went on.
        for _ in 0..8 {
            next_step(&failed(), false, &mut retry, poll);
        }
        assert_eq!(
            next_step(&failed(), true, &mut retry, poll),
            secs(1),
            "a daemon that served and then stopped is retried after 1 s, not 30 s"
        );
        assert_eq!(retry.failed_in_a_row, 1);
    }

    /// Only a local `unix:` socket gets a keeper.
    #[tokio::test]
    async fn a_non_unix_target_starts_no_keeper() {
        let dir = tempfile::tempdir().unwrap();
        let paths = rocm_core::AppPaths {
            config_dir: dir.path().join("config"),
            data_dir: dir.path().join("data"),
            cache_dir: dir.path().join("cache"),
        };
        let keeper = spawn_keeper(
            "tcp:127.0.0.1:9",
            &rocm_core::RocmCliConfig::default(),
            &paths,
        )
        .await;
        assert!(keeper.is_none(), "a tcp target must start nothing");
    }

    /// A writer the tests' own `tracing` subscriber logs into.
    #[derive(Clone, Default)]
    struct Captured(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

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

    impl Captured {
        fn text(&self) -> String {
            String::from_utf8_lossy(&self.0.lock().unwrap()).into_owned()
        }
    }

    /// Install a subscriber at `level` for this test's thread (the keeper runs
    /// on it: `#[tokio::test]` is single-threaded) and return what it writes.
    fn log_at(level: tracing::Level) -> (Captured, tracing::subscriber::DefaultGuard) {
        let captured = Captured::default();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(level)
            .with_ansi(false)
            .with_writer(captured.clone())
            .finish();
        (captured, tracing::subscriber::set_default(subscriber))
    }

    async fn wait_for_failure(keeper: &Keeper, attempts: u64) {
        let deadline = tokio::time::Instant::now() + DEADLINE;
        while keeper
            .failure
            .lock()
            .unwrap()
            .as_ref()
            .is_none_or(|f| f.failed_attempts < attempts)
        {
            assert!(
                tokio::time::Instant::now() < deadline,
                "no failure from attempt {attempts} was recorded within {DEADLINE:?}"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// A daemon that cannot serve its path keeps being retried while the
    /// dashboard is open, and what stopped it is reported once the dashboard
    /// closes -- naming the path once, which is left exactly as it was, and
    /// naming the log, which holds the warning.
    #[tokio::test]
    async fn a_daemon_refused_its_path_is_retried_and_reported_after_the_dashboard_closes() {
        let (log, _guard) = log_at(tracing::Level::WARN);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        std::fs::write(&path, "keep me").unwrap();
        let keeper = keeper(&path);

        // A second attempt, after the first backoff: it kept retrying.
        wait_for_failure(&keeper, 2).await;
        assert!(
            !keeper.handle.is_finished(),
            "the keeper must keep retrying"
        );
        let failure = stop_keeper(keeper)
            .await
            .expect("the failure must be reported when the dashboard closes");
        let report = failure_report(&failure, Path::new("/home/u/.rocm/logs"));

        let shown = path.display().to_string();
        assert_eq!(
            report.matches(&shown).count(),
            1,
            "the report must name the path once: {report}"
        );
        assert!(
            report.starts_with(&format!("{REPORT_COULD_NOT_START} {shown} {NOT_A_SOCKET}")),
            "{report}"
        );
        assert!(report.contains("failed attempts)"), "{report}");
        assert!(
            report.contains(&format!("{REPORT_LOG_LINE} /home/u/.rocm/logs")),
            "the warning was logged, so the report must say where: {report}"
        );
        let logged = log.text();
        assert!(
            logged.contains(KEEPER_WARNING) && logged.contains(&shown),
            "the log the report points at must hold the warning: {logged}"
        );
        assert_eq!(
            std::fs::read_to_string(&path).expect("the file the report names must still exist"),
            "keep me",
            "the report says the path was refused, so it must be untouched"
        );
    }

    /// When the warning is not recorded -- no log, or a filter that drops it --
    /// the report does not send the user to a log that has nothing.
    #[tokio::test]
    async fn a_report_names_the_log_only_when_the_warning_reached_it() {
        let (log, _guard) = log_at(tracing::Level::ERROR);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        std::fs::write(&path, "keep me").unwrap();
        let keeper = keeper(&path);

        wait_for_failure(&keeper, 1).await;
        let failure = stop_keeper(keeper).await.expect("still failing");
        let report = failure_report(&failure, Path::new("/home/u/.rocm/logs"));

        assert!(!failure.logged);
        assert!(!report.contains("logged"), "{report}");
        assert!(report.contains(NOT_A_SOCKET), "{report}");
        assert!(log.text().is_empty(), "nothing reached the log");
    }

    /// A failure stops counting once another daemon is found serving the
    /// socket: that dashboard is connected, and nothing is wrong to report.
    #[tokio::test]
    async fn a_failure_ends_when_another_daemon_serves_the_socket() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        std::fs::write(&path, "in the way").unwrap();
        let keeper = keeper(&path);
        wait_for_failure(&keeper, 1).await;

        // Something else starts serving the path.
        std::fs::remove_file(&path).unwrap();
        let _other = std::os::unix::net::UnixListener::bind(&path).unwrap();
        let deadline = tokio::time::Instant::now() + DEADLINE;
        while keeper.failure.lock().unwrap().is_some() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the failure was not cleared once another daemon served the socket"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        assert_eq!(stop_keeper(keeper).await, None);
    }

    /// A daemon that recovers is not reported when the dashboard closes.
    #[tokio::test]
    async fn a_daemon_that_recovered_is_not_reported() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        std::fs::write(&path, "in the way").unwrap();
        let keeper = keeper(&path);
        wait_for_failure(&keeper, 1).await;

        std::fs::remove_file(&path).unwrap();
        let _attached = wait_until_served(&path, "the daemon after the path was cleared").await;

        assert_eq!(
            stop_keeper(keeper).await,
            None,
            "it was serving when stopped"
        );
    }

    /// The dashboard does not wait out the ready wait for a daemon that has
    /// already failed.
    #[tokio::test]
    async fn a_failed_daemon_does_not_hold_up_the_dashboard() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        std::fs::write(&path, "in the way").unwrap();
        let paths = rocm_core::AppPaths {
            config_dir: dir.path().join("config"),
            data_dir: dir.path().join("data"),
            cache_dir: dir.path().join("cache"),
        };

        let started = tokio::time::Instant::now();
        let keeper = spawn_keeper(
            &format!("unix:{}", path.display()),
            &rocm_core::RocmCliConfig::default(),
            &paths,
        )
        .await
        .expect("a unix target gets a keeper");

        assert!(
            started.elapsed() < super::READY_WAIT / 2,
            "the dashboard waited {:?} for a daemon that had failed",
            started.elapsed()
        );
        assert!(stop_keeper(keeper).await.is_some());
    }

    /// A keeper that died of a panic is reported too, not dropped silently.
    #[tokio::test]
    async fn a_keeper_that_panicked_is_reported() {
        let keeper = Keeper {
            socket: "/run/user/1000/rocmdashd.sock".into(),
            handle: tokio::spawn(async { panic!("keeper bug") }),
            failure: std::sync::Arc::default(),
        };
        while !keeper.handle.is_finished() {
            tokio::task::yield_now().await;
        }
        let failure = stop_keeper(keeper)
            .await
            .expect("a panicked keeper must be reported");
        assert_eq!(failure.socket, Path::new("/run/user/1000/rocmdashd.sock"));
        assert!(
            failure.error.contains("stopped unexpectedly") && failure.error.contains("keeper bug"),
            "{failure:?}"
        );
    }
}
