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
use std::sync::atomic::{AtomicBool, Ordering};
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

/// How long a run has to serve before its stopping starts the backoff and the
/// count of failed runs over.
///
/// Serving at all is not enough: a daemon serves the moment it has bound, so
/// a runner that fails a second later would otherwise be restarted every
/// second forever, dropping its clients each time, and never counted as
/// failing twice. The first snapshot is not enough either -- a runner that
/// sends one and then fails is the same loop. A run that lasted as long as
/// the longest backoff wait restarts at most about as often as the backoff
/// would allow anyway, so that is the bar.
#[cfg(unix)]
const SERVED_LONG_ENOUGH: Duration = FAILURE_BACKOFF_MAX;

/// How long the dashboard waits for the socket to answer before it draws
/// anyway; the TUI client keeps reconnecting on its own after that. The wait
/// can run over by up to one [`READY_POLL`] and one probe: a probe of a
/// listener that does not greet lasts until its own timeout
/// (`STALE_PROBE_TIMEOUT` in rocm-dash-daemon's `server.rs`, 500 ms), and the
/// limit is checked between probes.
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
    /// Whether this keeper's daemon had served the socket before it failed
    /// (in the failed run or an earlier one): it stopped serving, rather
    /// than never started.
    pub(crate) served: bool,
    /// Whether everything the report says is in the log reached it: the
    /// keeper's warning for each failed run, and for a keeper that panicked
    /// also its error (so a panic after failures counts as logged only when
    /// both reached it). The only subscriber `rocm` installs writes the client
    /// log file, so `false` means the log is missing some of it (logging did
    /// not start, or `RUST_LOG` filters it out).
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
    /// Set once this keeper's daemon has served the socket.
    served: Arc<AtomicBool>,
}

/// Start this dashboard's keeper for a local `unix:` socket and wait (about
/// [`READY_WAIT`] at most; see there for how far it can run over) until the
/// socket answers, whoever serves it.
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
        // Probe with `socket_answers`, which reads the daemon's greeting
        // before hanging up: a bare connect that hung up first would have the
        // daemon log a client error for every dashboard opened.
        // `the_probe_waits_for_the_greeting_before_hanging_up`
        // (rocm-dash-daemon's server.rs) proves the probe reads the greeting.
        wait_until_ready(
            || rocm_dash_daemon::server::socket_answers(Path::new(target)),
            &keeper.failure,
        )
        .await;
        Some(keeper)
    }
    #[cfg(windows)]
    {
        let _ = (connect, config, paths);
        None
    }
}

/// How often [`wait_until_ready`] probes the socket.
#[cfg(unix)]
const READY_POLL: Duration = Duration::from_millis(20);

/// Wait until `answers` says something serves the socket, polling every
/// [`READY_POLL`] rather than sleeping a fixed time, which is race-prone on
/// slow or loaded systems. Stop at once if a failure is recorded in `failure`
/// (nothing will answer, and the TUI client keeps reconnecting on its own),
/// and once [`READY_WAIT`] has passed (see there for how far a slow probe can
/// carry it over), so the dashboard draws anyway.
#[cfg(unix)]
async fn wait_until_ready<P, F>(mut answers: P, failure: &Mutex<Option<KeeperFailure>>)
where
    P: FnMut() -> F,
    F: std::future::Future<Output = bool>,
{
    let deadline = tokio::time::Instant::now() + READY_WAIT;
    while !answers().await
        && tokio::time::Instant::now() < deadline
        && failure.lock().is_ok_and(|slot| slot.is_none())
    {
        tokio::time::sleep(READY_POLL).await;
    }
}

#[cfg(unix)]
fn start(socket: &Path, opts: rocm_dash_daemon::runner::RunnerOptions, poll: Duration) -> Keeper {
    let listen = format!("unix:{}", socket.display());
    start_with(socket, poll, move |ready| {
        let listen = listen.clone();
        let opts = opts.clone();
        async move { rocm_dash_daemon::server::run_with_ready(&listen, opts, ready).await }
    })
}

/// Called by one run of the daemon once it serves.
#[cfg(unix)]
type Ready = Box<dyn FnOnce() + Send>;

/// [`start`] with the daemon swapped: `serve` runs it once, calling the
/// [`Ready`] it is given once it serves, until it stops.
#[cfg(unix)]
fn start_with<S, F>(socket: &Path, poll: Duration, serve: S) -> Keeper
where
    S: FnMut(Ready) -> F + Send + 'static,
    F: std::future::Future<Output = anyhow::Result<()>> + Send + 'static,
{
    let failure = Arc::new(Mutex::new(None));
    let served = Arc::new(AtomicBool::new(false));
    let handle = spawn_logging_panic(keep_daemon(
        socket.to_path_buf(),
        poll,
        Arc::clone(&failure),
        Arc::clone(&served),
        serve,
    ));
    Keeper {
        socket: socket.to_path_buf(),
        handle,
        failure,
        served,
    }
}

/// The error a keeper logs when it panics.
#[cfg(unix)]
const KEEPER_PANICKED: &str = "the embedded telemetry daemon's keeper panicked";

/// Spawn a keeper's task, logging a panic in it at error level, with its
/// message, from inside the task ([`LogPanic`]), and then letting the panic go
/// on, so the task's `JoinHandle` still reports it to [`stop_keeper`]. Logged
/// there rather than in [`stop_keeper`] because a signal or a typed Ctrl-C
/// ends the process inside the TUI and [`stop_keeper`] never runs then.
#[cfg(unix)]
fn spawn_logging_panic(
    task: impl std::future::Future<Output = ()> + Send + 'static,
) -> JoinHandle<()> {
    tokio::spawn(LogPanic(Box::pin(task)))
}

/// A future that logs a panic in the one it wraps; see [`spawn_logging_panic`].
#[cfg(unix)]
struct LogPanic(std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>);

#[cfg(unix)]
impl std::future::Future for LogPanic {
    type Output = ();

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<()> {
        let inner = &mut self.0;
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| inner.as_mut().poll(cx))) {
            Ok(poll) => poll,
            Err(payload) => {
                let message = panic_message(payload.as_ref());
                tracing::error!(error = %message, "{KEEPER_PANICKED}");
                std::panic::resume_unwind(payload)
            }
        }
    }
}

/// Stop a keeper and wait for it, so the daemon it may own has removed its
/// socket (only if it still is that daemon's socket) and released the path
/// before the runtime is dropped. A keeper that never owned the socket
/// touches nothing on the way out.
///
/// Returns the failure the keeper's daemon was stuck in when it was stopped,
/// for the caller to report once the terminal is its own again -- unless
/// another daemon serves the socket by then: a failure is recorded before the
/// keeper waits to retry, and while it waited another dashboard's daemon may
/// have taken the socket and this dashboard worked on it until it closed. A
/// keeper that had panicked is reported, ahead of any failure its daemon was
/// in before the panic; the panic itself was logged by the task as it
/// happened ([`spawn_logging_panic`]).
pub(crate) async fn stop_keeper(keeper: Keeper) -> Option<KeeperFailure> {
    let Keeper {
        socket,
        handle,
        failure,
        served,
    } = keeper;
    handle.abort();
    let panicked = match handle.await {
        Err(e) if e.is_panic() => Some(panic_message(e.into_panic().as_ref())),
        _ => None,
    };
    let recorded = failure.lock().map_or(None, |slot| slot.clone());
    // This keeper's own daemon has stopped by now, so whatever answers at the
    // socket is another one.
    let recorded = match recorded {
        Some(_) if another_daemon_serves(&socket).await => None,
        recorded => recorded,
    };
    // A panic comes first, and is always reported: it is why nothing retries
    // any more, whatever failure the daemon was in before it. (A failure
    // recorded before it was dropped above if another daemon serves now.)
    match (recorded, panicked) {
        (recorded, Some(error)) => Some(KeeperFailure {
            socket,
            error: match &recorded {
                Some(before) => format!(
                    "the dashboard's supervision of it stopped unexpectedly: {error} \
                     (before that: {})",
                    before.error
                ),
                None => format!("the dashboard's supervision of it stopped unexpectedly: {error}"),
            },
            failed_attempts: recorded.as_ref().map_or(0, |before| before.failed_attempts),
            served: served.load(Ordering::SeqCst),
            logged: tracing::enabled!(tracing::Level::ERROR)
                && recorded.as_ref().is_none_or(|before| before.logged),
        }),
        (Some(failure), None) => Some(failure),
        (None, None) => None,
    }
}

/// What a panic said: its `&str` or `String` payload, or just "panicked".
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| (*message).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "panicked".to_string())
}

/// Whether a daemon answers at `socket`.
#[cfg(unix)]
async fn another_daemon_serves(socket: &Path) -> bool {
    rocm_dash_daemon::server::socket_answers(socket).await
}

/// No daemon socket on Windows (and no keeper there to ask).
#[cfg(windows)]
// Mirrors the `cfg(unix)` arm, which callers `.await`.
#[allow(clippy::unused_async)]
async fn another_daemon_serves(socket: &Path) -> bool {
    let _ = socket;
    false
}

/// Whether a daemon answers at `socket`, for a caller that cannot await (see
/// [`report_on_forced_exit`]).
#[cfg(unix)]
fn another_daemon_serves_blocking(socket: &Path) -> bool {
    rocm_dash_daemon::server::socket_answers_blocking(socket)
}

/// No daemon socket on Windows (and no keeper there to ask).
#[cfg(windows)]
const fn another_daemon_serves_blocking(socket: &Path) -> bool {
    let _ = socket;
    false
}

/// Have the failure report printed if the process is ended by a signal or a
/// typed Ctrl-C while `keeper` runs, until the returned guard is dropped.
///
/// Those exits end in `std::process::exit` inside the TUI
/// (`spawn_termination_watcher` and `exit_on_ctrl_c` in rocm-dash-tui's
/// `app/terminal.rs`), so [`stop_keeper`] never runs on them; the TUI runs
/// this hook instead, once the terminal is restored. It prints the report for
/// the failure recorded at that moment, through [`forced_exit_report`], and
/// nothing when none is. Unlike [`stop_keeper`] it cannot stop the keeper and
/// wait for it, so a keeper that panicked is not reported on these exits; the
/// panic is still logged, by the keeper's own task as it happens
/// ([`spawn_logging_panic`]).
pub(crate) fn report_on_forced_exit(
    keeper: &Keeper,
    log_dir: PathBuf,
) -> rocm_dash_tui::app::PreExitHookGuard {
    let failure = Arc::clone(&keeper.failure);
    rocm_dash_tui::app::set_pre_exit_hook(move || {
        if let Some(report) = forced_exit_report(&failure, &log_dir, another_daemon_serves_blocking)
        {
            // Not `eprintln!`, which panics if stderr is gone: the process is
            // on its way out, and a report it cannot write is simply lost.
            use std::io::Write as _;
            let _ = writeln!(std::io::stderr(), "{report}");
        }
    })
}

/// The report [`report_on_forced_exit`] prints: [`failure_report`] for the
/// failure recorded in `failure`, unless none is or a daemon answers at its
/// socket by now -- the same rule [`stop_keeper`] applies, here with
/// `another_daemon_serves` as the probe. This keeper's daemon may still be
/// running on this path, so a daemon that answers may be its own, recovered;
/// there is nothing to report either way.
fn forced_exit_report(
    failure: &Mutex<Option<KeeperFailure>>,
    log_dir: &Path,
    another_daemon_serves: impl FnOnce(&Path) -> bool,
) -> Option<String> {
    let recorded = failure.lock().ok()?.clone()?;
    if another_daemon_serves(&recorded.socket) {
        return None;
    }
    Some(failure_report(&recorded, log_dir))
}

/// How the report opens for a daemon that never served the socket. dash-57
/// and dash-60 (`dash_steps.rs`), which cannot import it, match its tail from
/// "telemetry daemon".
pub(crate) const REPORT_COULD_NOT_START: &str =
    "rocm: the dashboard's telemetry daemon could not start:";

/// How the report opens for a daemon that served the socket and then
/// stopped (and has not served it again since). dash-59 (`dash_steps.rs`)
/// matches its tail from "telemetry daemon".
pub(crate) const REPORT_STOPPED_SERVING: &str =
    "rocm: the dashboard's telemetry daemon stopped serving:";

/// The report's line pointing at the client log. dash-57 and dash-60 match it
/// without its leading `rocm: `.
pub(crate) const REPORT_LOG_LINE: &str = "rocm: each failed attempt is logged in the files under";

/// The warning the keeper logs for each failed run. dash-57, dash-59 and
/// dash-60 look for the fragment "could not serve its socket" from it in the
/// client log.
#[cfg(unix)]
pub(crate) const KEEPER_WARNING: &str =
    "embedded telemetry daemon could not serve its socket; retrying";

/// The lines `rocm dash` prints after the dashboard closes when its daemon
/// was stuck in `failure`; built from the recorded failure itself, so it says
/// what actually went wrong -- whether the daemon never started or stopped
/// after serving -- and names the log only when the warning reached it.
pub(crate) fn failure_report(failure: &KeeperFailure, log_dir: &Path) -> String {
    let socket = failure.socket.display().to_string();
    let opening = if failure.served {
        REPORT_STOPPED_SERVING
    } else {
        REPORT_COULD_NOT_START
    };
    // Most errors already name the path; say it once.
    let what = if failure.error.contains(&socket) {
        format!("{opening} {}", failure.error)
    } else {
        format!("{opening} {socket}: {}", failure.error)
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
/// many failed in a row. Both start over once a run has served for
/// [`SERVED_LONG_ENOUGH`], another daemon is found serving, or the socket was
/// taken from a serving daemon.
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
/// else both start over. A run that served for `served_for` of at least
/// [`SERVED_LONG_ENOUGH`] ended the run of failures before it, so a failure
/// after it is counted from one again; a shorter one did not.
#[cfg(unix)]
fn next_step(
    outcome: &anyhow::Result<()>,
    served_for: Option<Duration>,
    retry: &mut Retry,
    poll: Duration,
) -> NextStep {
    use rocm_dash_daemon::server::{SocketHolder, SocketInUse, SocketReplaced};

    if served_for.is_some_and(|served| served >= SERVED_LONG_ENOUGH) {
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

/// Serve `socket` whenever no other daemon does, until aborted, running the
/// daemon through `serve` and keeping `failure` set to the failure the daemon
/// is stuck in: set when a run fails, cleared once a run is serving or
/// another daemon is found serving. `served` is set once a run serves.
///
/// Logs through `tracing` only: the dashboard owns the terminal while this
/// runs, so a stderr write would land inside the TUI's screen.
#[cfg(unix)]
async fn keep_daemon<S, F>(
    socket: PathBuf,
    poll: Duration,
    failure: Arc<Mutex<Option<KeeperFailure>>>,
    served: Arc<AtomicBool>,
    mut serve: S,
) where
    S: FnMut(Ready) -> F,
    F: std::future::Future<Output = anyhow::Result<()>>,
{
    fn set(slot: &Mutex<Option<KeeperFailure>>, value: Option<KeeperFailure>) {
        if let Ok(mut slot) = slot.lock() {
            *slot = value;
        }
    }
    let mut retry = Retry::new();
    loop {
        let serving_since = Arc::new(Mutex::new(None));
        let ready: Ready = {
            let serving_since = Arc::clone(&serving_since);
            let failure = Arc::clone(&failure);
            let served = Arc::clone(&served);
            Box::new(move || {
                if let Ok(mut since) = serving_since.lock() {
                    *since = Some(tokio::time::Instant::now());
                }
                served.store(true, Ordering::SeqCst);
                set(&failure, None);
            })
        };
        let outcome = serve(ready).await;
        let served_for = serving_since
            .lock()
            .ok()
            .and_then(|since| *since)
            .map(|since: tokio::time::Instant| since.elapsed());
        match next_step(&outcome, served_for, &mut retry, poll) {
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
                        served: served.load(Ordering::SeqCst),
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
        FAILURE_BACKOFF_MAX, KEEPER_WARNING, Keeper, KeeperFailure, NextStep,
        REPORT_COULD_NOT_START, REPORT_LOG_LINE, REPORT_STOPPED_SERVING, Ready, Retry,
        SERVED_LONG_ENOUGH, TAKEOVER_POLL, UNLOCKED_LISTENER_POLL, failure_report, next_step,
        spawn_keeper, start, start_with, stop_keeper,
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
            .map(|_| next_step(&failed(), None, &mut retry, poll))
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
            next_step(&Ok(()), None, &mut retry, poll),
            NextStep::Failed(FAILURE_BACKOFF_MAX),
            "a daemon that stops without an error has still stopped"
        );

        assert_eq!(
            next_step(&in_use(SocketHolder::LockHolder), None, &mut retry, poll),
            NextStep::OwnedElsewhere(poll)
        );
        assert_eq!(
            next_step(&failed(), None, &mut retry, poll),
            secs(1),
            "finding an owner starts the failure backoff over"
        );
        assert_eq!(
            next_step(
                &in_use(SocketHolder::UnlockedListener),
                None,
                &mut retry,
                poll
            ),
            NextStep::OwnedElsewhere(UNLOCKED_LISTENER_POLL)
        );
        next_step(&failed(), None, &mut retry, poll);
        let replaced: anyhow::Result<()> = Err(rocm_dash_daemon::server::SocketReplaced {
            path: "/run/x.sock".into(),
        }
        .into());
        assert_eq!(
            next_step(&replaced, None, &mut retry, poll),
            NextStep::Reclaim(poll),
            "a socket taken from a serving daemon is reclaimed after one poll, not reported"
        );
        assert_eq!(next_step(&failed(), None, &mut retry, poll), secs(1));

        // Only failures in a row are counted: polls that find an owner, however
        // many, are not attempts.
        let mut retry = Retry::new();
        next_step(&failed(), None, &mut retry, poll);
        next_step(&failed(), None, &mut retry, poll);
        for _ in 0..5 {
            next_step(&in_use(SocketHolder::LockHolder), None, &mut retry, poll);
        }
        next_step(&failed(), None, &mut retry, poll);
        assert_eq!(retry.failed_in_a_row, 1);
        next_step(&failed(), None, &mut retry, poll);
        assert_eq!(retry.failed_in_a_row, 2);

        // A run that served long enough before it stopped starts the backoff
        // and the count over, however long the failures before it went on.
        for _ in 0..8 {
            next_step(&failed(), None, &mut retry, poll);
        }
        assert_eq!(
            next_step(&failed(), Some(SERVED_LONG_ENOUGH), &mut retry, poll),
            secs(1),
            "a daemon that served for a while and then stopped is retried after 1 s, not 30 s"
        );
        assert_eq!(retry.failed_in_a_row, 1);

        // A run that served only briefly did not: it is one more failure.
        next_step(&failed(), None, &mut retry, poll);
        assert_eq!(
            next_step(
                &failed(),
                SERVED_LONG_ENOUGH.checked_sub(Duration::from_millis(1)),
                &mut retry,
                poll
            ),
            secs(4)
        );
        assert_eq!(retry.failed_in_a_row, 3);
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
    /// the same on every thread. Mirrors `log_at` in rocm-dash-daemon's
    /// `server.rs` tests.
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
        assert!(!failure.served, "it never served, so it could not start");
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

    /// Waits for the socket to answer, with a probe that answers after
    /// `answers_after` probes (never, for `None`) and a failure slot holding
    /// `failure`; returns how long it waited and how many probes it made.
    async fn ready_wait(
        answers_after: Option<usize>,
        failure: Option<KeeperFailure>,
    ) -> (Duration, usize) {
        let probes = std::sync::atomic::AtomicUsize::new(0);
        let slot = std::sync::Mutex::new(failure);
        let started = tokio::time::Instant::now();
        tokio::time::timeout(
            super::READY_WAIT * 2,
            super::wait_until_ready(
                || {
                    let made = probes.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                    let answers = answers_after.is_some_and(|after| made > after);
                    async move { answers }
                },
                &slot,
            ),
        )
        .await
        .expect("the ready wait must end by itself");
        (
            started.elapsed(),
            probes.load(std::sync::atomic::Ordering::SeqCst),
        )
    }

    /// A socket that answers ends the ready wait at the first probe that
    /// finds it answering, not after the full wait.
    #[tokio::test(start_paused = true)]
    async fn the_ready_wait_ends_once_the_socket_answers() {
        assert_eq!(ready_wait(Some(0), None).await, (Duration::ZERO, 1));
        assert_eq!(
            ready_wait(Some(2), None).await,
            (super::READY_POLL * 2, 3),
            "it polls until the socket answers"
        );
    }

    /// A socket that never answers, with no failure recorded, ends the ready
    /// wait after [`READY_WAIT`], so the dashboard draws anyway.
    #[tokio::test(start_paused = true)]
    async fn the_ready_wait_gives_up_after_its_limit() {
        let (waited, _) = ready_wait(None, None).await;
        assert!(
            waited >= super::READY_WAIT && waited < super::READY_WAIT + super::READY_POLL,
            "waited {waited:?}"
        );
    }

    /// A failure already recorded ends the ready wait at once: nothing will
    /// answer.
    #[tokio::test(start_paused = true)]
    async fn the_ready_wait_ends_at_once_when_the_daemon_has_failed() {
        let failure = KeeperFailure {
            socket: std::path::PathBuf::from("/run/user/1000/rocmdashd.sock"),
            error: "refused".into(),
            failed_attempts: 1,
            served: false,
            logged: true,
        };
        assert_eq!(ready_wait(None, Some(failure)).await, (Duration::ZERO, 1));
    }

    /// A daemon that starts and serves ends the dashboard's wait as soon as
    /// its socket answers -- the real probe, not the time limit -- and a
    /// daemon that served without failing leaves nothing to report.
    #[tokio::test]
    async fn the_dashboard_waits_only_until_its_daemon_answers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
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
        let waited = started.elapsed();

        assert!(
            waited < super::READY_WAIT / 2,
            "the dashboard waited {waited:?} for a daemon that serves"
        );
        assert!(
            rocm_dash_daemon::server::socket_answers(&path).await,
            "the daemon serves by the time the wait ends"
        );
        assert_eq!(stop_keeper(keeper).await, None);
    }

    /// A keeper that died of a panic is reported too, not dropped silently.
    #[tokio::test]
    async fn a_keeper_that_panicked_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let keeper = Keeper {
            socket: path.clone(),
            handle: super::spawn_logging_panic(async { panic!("keeper bug") }),
            failure: std::sync::Arc::default(),
            served: std::sync::Arc::default(),
        };
        while !keeper.handle.is_finished() {
            tokio::task::yield_now().await;
        }
        let failure = stop_keeper(keeper)
            .await
            .expect("a panicked keeper must be reported");
        assert_eq!(failure.socket, path);
        assert_eq!(
            failure.error,
            "the dashboard's supervision of it stopped unexpectedly: keeper bug"
        );
    }

    /// A keeper that panics logs it from its own task, with the message, so it
    /// is logged even when the process ends without [`stop_keeper`] (a signal
    /// or a typed Ctrl-C); stopping it afterwards does not log it twice.
    #[tokio::test]
    async fn a_keeper_panic_is_logged_once_without_waiting_for_stop_keeper() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let (log, _guard) = log_at(tracing::Level::ERROR);

        let keeper = panicked_keeper(&path, None).await;

        let logged = || log.text().matches(super::KEEPER_PANICKED).count();
        assert_eq!(logged(), 1, "{}", log.text());
        assert!(log.text().contains("keeper bug"), "{}", log.text());
        assert!(stop_keeper(keeper).await.is_some());
        assert_eq!(logged(), 1, "stop_keeper does not log it again");
    }

    /// A keeper whose task panicked, after its daemon served (`served`) or not.
    async fn keeper_that_panics(
        path: &Path,
        served: bool,
        panics: impl std::future::Future<Output = ()> + Send + 'static,
    ) -> Keeper {
        let keeper = Keeper {
            socket: path.to_path_buf(),
            handle: super::spawn_logging_panic(panics),
            failure: std::sync::Arc::default(),
            served: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(served)),
        };
        while !keeper.handle.is_finished() {
            tokio::task::yield_now().await;
        }
        keeper
    }

    /// A keeper that panicked after its daemon had served is reported as
    /// having stopped serving.
    #[tokio::test]
    async fn a_keeper_that_panicked_after_serving_is_reported_as_stopped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let (_log, _guard) = log_at(tracing::Level::WARN);
        let keeper = keeper_that_panics(&path, true, async { panic!("keeper bug") }).await;

        let failure = stop_keeper(keeper).await.expect("reported");

        assert!(failure.served, "{failure:?}");
        assert_eq!(
            failure_report(&failure, Path::new(LOG_DIR)),
            format!(
                "{REPORT_STOPPED_SERVING} {}: the dashboard's supervision of it stopped \
                 unexpectedly: keeper bug\n{REPORT_LOG_LINE} {LOG_DIR}",
                path.display()
            )
        );
    }

    /// The panic's own message is reported whether it was formatted (a
    /// `String`) or not (a `&str`); a payload that is neither says only that
    /// it panicked.
    #[tokio::test]
    async fn a_keeper_panic_is_reported_with_its_message() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let attempt = 7;
        let formatted =
            keeper_that_panics(&path, false, async move { panic!("keeper bug {attempt}") }).await;
        let opaque =
            keeper_that_panics(&path, false, async { std::panic::panic_any(42_u32) }).await;

        let formatted = stop_keeper(formatted).await.expect("reported");
        let opaque = stop_keeper(opaque).await.expect("reported");

        assert_eq!(
            formatted.error,
            "the dashboard's supervision of it stopped unexpectedly: keeper bug 7"
        );
        assert_eq!(
            opaque.error,
            "the dashboard's supervision of it stopped unexpectedly: panicked"
        );
    }

    /// A keeper whose task panicked, built with `recorded` as the failure its
    /// daemon was in before the panic.
    async fn panicked_keeper(path: &Path, recorded: Option<KeeperFailure>) -> Keeper {
        let keeper = Keeper {
            socket: path.to_path_buf(),
            handle: super::spawn_logging_panic(async { panic!("keeper bug") }),
            failure: std::sync::Arc::new(std::sync::Mutex::new(recorded)),
            served: std::sync::Arc::default(),
        };
        while !keeper.handle.is_finished() {
            tokio::task::yield_now().await;
        }
        keeper
    }

    fn refused(path: &Path, logged: bool) -> KeeperFailure {
        KeeperFailure {
            socket: path.to_path_buf(),
            error: "refused".into(),
            failed_attempts: 3,
            served: false,
            logged,
        }
    }

    /// A keeper that panicked is reported as such even when a failure of its
    /// daemon was still recorded -- the panic is why nothing retries any more
    /// -- keeping that failure's error and count, and saying the log holds it
    /// all only when both the panic and the earlier warnings reached it.
    #[tokio::test]
    async fn a_keeper_panic_is_reported_over_a_recorded_failure() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let (_log, _guard) = log_at(tracing::Level::WARN);

        let keeper = panicked_keeper(&path, Some(refused(&path, true))).await;
        let failure = stop_keeper(keeper).await;

        assert_eq!(
            failure,
            Some(KeeperFailure {
                socket: path.clone(),
                error: "the dashboard's supervision of it stopped unexpectedly: keeper bug \
                        (before that: refused)"
                    .into(),
                failed_attempts: 3,
                served: false,
                logged: true,
            })
        );
        assert_eq!(
            failure_report(&failure.unwrap(), Path::new(LOG_DIR)),
            format!(
                "{REPORT_COULD_NOT_START} {}: the dashboard's supervision of it stopped \
                 unexpectedly: keeper bug (before that: refused) (3 failed attempts)\n\
                 {REPORT_LOG_LINE} {LOG_DIR}",
                path.display()
            )
        );
    }

    /// Under a filter that drops warnings, the earlier failures are not in the
    /// log even though the panic is, so the report does not point at the log.
    #[tokio::test]
    async fn a_keeper_panic_after_unlogged_failures_is_not_reported_as_logged() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let (log, _guard) = log_at(tracing::Level::ERROR);

        let keeper = panicked_keeper(&path, Some(refused(&path, false))).await;
        let failure = stop_keeper(keeper).await.expect("reported");

        assert!(!failure.logged, "{failure:?}");
        assert!(
            !failure_report(&failure, Path::new(LOG_DIR)).contains(REPORT_LOG_LINE),
            "{failure:?}"
        );
        assert!(
            log.text().contains("keeper bug"),
            "the panic itself is logged"
        );
    }

    /// A panic is reported even when another daemon serves the socket by the
    /// time the dashboard closes; only the failure recorded before it is
    /// dropped then, since the dashboard was connected to that daemon.
    #[tokio::test]
    async fn a_keeper_panic_is_reported_even_when_another_daemon_serves() {
        use tokio::io::AsyncWriteExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let other = tokio::net::UnixListener::bind(&path).unwrap();
        let _greeter = tokio::spawn(async move {
            while let Ok((mut client, _)) = other.accept().await {
                let _ = client.write_all(b"{}\n").await;
            }
        });

        let keeper = panicked_keeper(&path, Some(refused(&path, true))).await;
        let failure = stop_keeper(keeper).await;

        assert_eq!(
            failure,
            Some(KeeperFailure {
                socket: path.clone(),
                error: "the dashboard's supervision of it stopped unexpectedly: keeper bug".into(),
                failed_attempts: 0,
                served: false,
                logged: tracing::enabled!(tracing::Level::ERROR),
            })
        );
    }

    // ── The keeper's timing, with the daemon scripted ─────────────────
    //
    // These run on tokio's paused clock: each sleep the keeper takes ends at
    // once, and the clock moves by exactly that sleep, so the waits between
    // runs are read off the times the runs started.

    /// One scripted run of the daemon.
    #[derive(Clone, Copy)]
    enum Run {
        /// Fail at once, without serving (say, a path it refuses).
        Refused,
        /// Serve for this long, then stop with this error.
        ServeThenStop(Duration, &'static str),
        /// Serve, then find the socket taken from it at once.
        Replaced,
        /// Find the socket in use by this holder, without serving.
        InUse(rocm_dash_daemon::server::SocketHolder),
    }

    type RunFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send>>;

    /// A daemon that runs `script` one entry per run and then hangs without
    /// serving, so the last failure stays recorded; and the times its runs
    /// started.
    fn scripted(
        path: &Path,
        script: Vec<Run>,
    ) -> (
        impl FnMut(Ready) -> RunFuture + Send + 'static,
        std::sync::Arc<std::sync::Mutex<Vec<tokio::time::Instant>>>,
    ) {
        let started = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let record = std::sync::Arc::clone(&started);
        let path = path.to_path_buf();
        let mut script = script.into_iter();
        let serve = move |ready: Ready| {
            record.lock().unwrap().push(tokio::time::Instant::now());
            let run = script.next();
            let path = path.clone();
            Box::pin(async move {
                match run {
                    None => std::future::pending().await,
                    Some(Run::Refused) => Err(anyhow::anyhow!("refused")),
                    Some(Run::ServeThenStop(serving, error)) => {
                        ready();
                        tokio::time::sleep(serving).await;
                        Err(anyhow::anyhow!(error))
                    }
                    Some(Run::Replaced) => {
                        ready();
                        Err(rocm_dash_daemon::server::SocketReplaced { path }.into())
                    }
                    Some(Run::InUse(holder)) => {
                        Err(rocm_dash_daemon::server::SocketInUse { path, holder }.into())
                    }
                }
            }) as RunFuture
        };
        (serve, started)
    }

    /// The time between one run's start and the next's.
    fn gaps(started: &std::sync::Mutex<Vec<tokio::time::Instant>>) -> Vec<Duration> {
        started
            .lock()
            .unwrap()
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .collect()
    }

    fn recorded(keeper: &Keeper) -> Option<KeeperFailure> {
        keeper.failure.lock().unwrap().clone()
    }

    const LOG_DIR: &str = "/home/u/.rocm/logs";

    /// Failed runs in a row are spaced by the backoff -- 1 s, doubling -- not
    /// retried back to back, and the report counts every one of them.
    #[tokio::test(start_paused = true)]
    async fn failed_runs_are_spaced_by_the_growing_backoff() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let (serve, started) = scripted(&path, vec![Run::Refused; 5]);
        let keeper = start_with(&path, POLL, serve);

        tokio::time::sleep(Duration::from_mins(1)).await;

        let secs = Duration::from_secs;
        assert_eq!(
            gaps(&started),
            [secs(1), secs(2), secs(4), secs(8), secs(16)]
        );
        let failure = recorded(&keeper).expect("the last failure is recorded");
        assert_eq!((failure.failed_attempts, failure.served), (5, false));
        let report = failure_report(
            &stop_keeper(keeper).await.expect("reported"),
            Path::new(LOG_DIR),
        );
        assert_eq!(
            report,
            format!(
                "{REPORT_COULD_NOT_START} {}: refused (5 failed attempts)",
                path.display()
            )
        );
    }

    /// A daemon whose runner keeps stopping soon after it starts serving is
    /// backed off like any other failure, not restarted every second with its
    /// clients dropped each time, and the report counts the failures.
    #[tokio::test(start_paused = true)]
    async fn a_daemon_that_keeps_stopping_soon_after_serving_backs_off() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let briefly = Run::ServeThenStop(Duration::from_secs(1), "the telemetry runner panicked");
        let (serve, started) = scripted(&path, vec![briefly; 4]);
        let keeper = start_with(&path, POLL, serve);

        tokio::time::sleep(Duration::from_mins(1)).await;

        // Each gap is the second it served plus the wait after it.
        let secs = Duration::from_secs;
        assert_eq!(gaps(&started), [secs(2), secs(3), secs(5), secs(9)]);
        let failure = recorded(&keeper).expect("the last failure is recorded");
        assert_eq!((failure.failed_attempts, failure.served), (4, true));
        let report = failure_report(
            &stop_keeper(keeper).await.expect("reported"),
            Path::new(LOG_DIR),
        );
        assert_eq!(
            report,
            format!(
                "{REPORT_STOPPED_SERVING} {}: the telemetry runner panicked (4 failed attempts)",
                path.display()
            )
        );
    }

    /// A run that served for [`SERVED_LONG_ENOUGH`] before it stopped starts
    /// the backoff and the count over.
    #[tokio::test(start_paused = true)]
    async fn a_run_that_served_long_enough_starts_the_backoff_over() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let (serve, started) = scripted(
            &path,
            vec![
                Run::Refused,
                Run::Refused,
                Run::ServeThenStop(SERVED_LONG_ENOUGH, "lost the path"),
                Run::Refused,
            ],
        );
        let keeper = start_with(&path, POLL, serve);

        tokio::time::sleep(Duration::from_mins(2)).await;

        let secs = Duration::from_secs;
        assert_eq!(
            gaps(&started),
            [secs(1), secs(2), SERVED_LONG_ENOUGH + secs(1), secs(2)]
        );
        let failure = recorded(&keeper).expect("the last failure is recorded");
        assert_eq!((failure.failed_attempts, failure.served), (2, true));
        stop_keeper(keeper).await;
    }

    /// A daemon that served and then lost its path is reported as having
    /// stopped serving, not as having failed to start.
    #[tokio::test(start_paused = true)]
    async fn a_daemon_that_served_and_then_stopped_is_reported_as_stopped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let hours = Duration::from_hours(3);
        let (serve, _) = scripted(&path, vec![Run::ServeThenStop(hours, "lost the path")]);
        let keeper = start_with(&path, POLL, serve);

        tokio::time::sleep(hours + Duration::from_secs(10)).await;

        let failure = recorded(&keeper).expect("the failure is recorded");
        assert_eq!((failure.failed_attempts, failure.served), (1, true));
        let report = failure_report(
            &stop_keeper(keeper).await.expect("reported"),
            Path::new(LOG_DIR),
        );
        assert_eq!(
            report,
            format!("{REPORT_STOPPED_SERVING} {}: lost the path", path.display())
        );
    }

    /// A socket taken from a serving daemon is claimed again one poll later,
    /// and is neither logged nor reported as a failure.
    #[tokio::test(start_paused = true)]
    async fn a_replaced_socket_is_claimed_again_after_one_poll_without_a_failure() {
        let (log, _guard) = log_at(tracing::Level::WARN);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let (serve, started) = scripted(&path, vec![Run::Replaced]);
        let keeper = start_with(&path, POLL, serve);

        tokio::time::sleep(Duration::from_secs(10)).await;

        assert_eq!(gaps(&started), [POLL]);
        assert_eq!(recorded(&keeper), None);
        assert!(!log.text().contains(KEEPER_WARNING), "{}", log.text());
        assert_eq!(stop_keeper(keeper).await, None);
    }

    /// A socket another daemon holds the lock on is tried again after one
    /// takeover poll, and one something without the lock answers at only
    /// after the slower poll; neither is recorded as a failure.
    #[tokio::test(start_paused = true)]
    async fn a_socket_in_use_is_tried_again_after_its_holders_poll() {
        use rocm_dash_daemon::server::SocketHolder::{LockHolder, UnlockedListener};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let (serve, started) = scripted(
            &path,
            vec![Run::InUse(LockHolder), Run::InUse(UnlockedListener)],
        );
        let keeper = start_with(&path, TAKEOVER_POLL, serve);

        tokio::time::sleep(Duration::from_secs(30)).await;

        assert_eq!(
            gaps(&started),
            [Duration::from_millis(250), Duration::from_secs(5)]
        );
        assert_eq!(recorded(&keeper), None);
        assert_eq!(stop_keeper(keeper).await, None);
    }

    /// A failure recorded before the keeper waits to retry is not reported if,
    /// by the time the dashboard closes, another daemon serves the socket: the
    /// dashboard was connected to that one.
    #[tokio::test]
    async fn a_failure_is_not_reported_once_another_daemon_serves_the_socket() {
        use tokio::io::AsyncWriteExt;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let (serve, _) = scripted(&path, vec![Run::Refused]);
        let keeper = start_with(&path, POLL, serve);
        wait_for_failure(&keeper, 1).await;

        let other = tokio::net::UnixListener::bind(&path).unwrap();
        let _greeter = tokio::spawn(async move {
            while let Ok((mut client, _)) = other.accept().await {
                let _ = client.write_all(b"{}\n").await;
            }
        });
        assert!(
            recorded(&keeper).is_some(),
            "the failure is still recorded when the dashboard closes"
        );

        assert_eq!(stop_keeper(keeper).await, None);
    }

    /// On a forced exit the report is the one for the failure recorded then,
    /// and there is none when nothing is recorded or a daemon answers at the
    /// socket by then.
    #[test]
    fn the_forced_exit_report_is_the_recorded_failures_unless_a_daemon_answers() {
        let failure = KeeperFailure {
            socket: "/run/x.sock".into(),
            error: "/run/x.sock exists and is not a socket".into(),
            failed_attempts: 3,
            served: false,
            logged: true,
        };
        let log = Path::new(LOG_DIR);
        let recorded = std::sync::Mutex::new(Some(failure.clone()));
        let probed = std::cell::Cell::new(None);
        let nobody = |socket: &Path| {
            probed.set(Some(socket.to_path_buf()));
            false
        };

        assert_eq!(
            super::forced_exit_report(&recorded, log, nobody),
            Some(failure_report(&failure, log))
        );
        assert_eq!(probed.take().as_deref(), Some(Path::new("/run/x.sock")));
        assert_eq!(
            super::forced_exit_report(&recorded, log, |_| true),
            None,
            "a daemon answers by then"
        );
        assert_eq!(
            super::forced_exit_report(&std::sync::Mutex::new(None), log, |_| false),
            None,
            "nothing recorded"
        );
    }

    /// The report counts failed attempts only from two on: "(1 failed
    /// attempts)" says nothing the report does not already say.
    #[test]
    fn the_report_counts_failed_attempts_from_two() {
        let failure = |failed_attempts| KeeperFailure {
            socket: "/run/x.sock".into(),
            error: "/run/x.sock exists and is not a socket".into(),
            failed_attempts,
            served: false,
            logged: false,
        };
        let log = Path::new(LOG_DIR);
        let once = format!("{REPORT_COULD_NOT_START} /run/x.sock exists and is not a socket");
        assert_eq!(failure_report(&failure(1), log), once);
        assert_eq!(
            failure_report(&failure(2), log),
            format!("{once} (2 failed attempts)")
        );
    }

    /// End to end: a serving daemon whose socket is removed gives the path up,
    /// and the keeper serves it again without recording or logging a failure.
    #[tokio::test]
    async fn a_removed_socket_is_served_again_without_a_failure() {
        let (log, _guard) = log_at(tracing::Level::WARN);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rocmdashd.sock");
        let keeper = keeper(&path);
        let _attached = wait_until_served(&path, "the daemon").await;
        pin_inode(&path);
        let first = inode(&path);

        std::fs::remove_file(&path).unwrap();
        let _reattached = wait_until_served(&path, "the daemon after its socket was removed").await;

        assert_ne!(inode(&path), first, "a new socket serves the path");
        assert_eq!(recorded(&keeper), None);
        assert!(!log.text().contains(KEEPER_WARNING), "{}", log.text());
        assert_eq!(stop_keeper(keeper).await, None);
    }
}
