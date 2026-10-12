// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Bounded command execution that captures output without deadlocking on it.

use std::io::{ErrorKind, Read};
use std::process::{Command, ExitStatus, Output, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use crate::examine::truncate_to_chars;

/// How often the wait loop polls for the child's exit.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// How much a reader accumulates before publishing it to the shared buffer.
const READ_CHUNK: usize = 16 * 1024;

/// How long the readers get to finish once the child is no longer running.
///
/// Reaching this means something other than the child still holds the pipe's
/// write end — a grandchild the child left behind — so there is no EOF coming
/// and waiting longer would make `timeout` meaningless. A child cannot exit
/// before writing everything, and the readers run throughout, so a child that
/// genuinely finished has at most a buffer or two left to collect here.
const DRAIN_GRACE: Duration = Duration::from_millis(200);

/// How much of a timed-out child's own output the timeout error quotes.
///
/// Bounded here rather than at any one caller because the quote is assembled
/// here, and this is the only place that knows those characters are raw child
/// output rather than something this crate wrote. All four callers inherit it.
///
/// The bound matters because the message does not stop at a terminal. `rocmd`
/// renders the whole error chain into the bridge snapshot's `note`, and the
/// GPU-metrics watcher copies that note into the automation-event and audit
/// JSONL files once a minute — files whose append helpers rewrite the whole
/// file each time. Unbounded, a child that prints a megabyte and then stalls
/// past its timeout grows both logs by that much per minute, and every
/// subsequent append costs more than the last.
///
/// Sized to still hold a real stderr backtrace: the point is to stop a
/// payload-sized quote, not to shorten diagnostics.
const MAX_QUOTED_OUTPUT_CHARS: usize = 2000;

/// One captured stream: the bytes so far, whether its reader saw EOF, and the
/// read error that stopped it, if one did.
///
/// # Lifetime of the reader thread
///
/// The thread outlives the call that started it whenever the drain grace runs
/// out — that is the point of the grace — and it then lives until the pipe's
/// last write end closes, which is the abandoned grandchild's business, not
/// ours. What it must *not* do is keep growing: `take()` marks the capture
/// abandoned, and from that moment the reader drains to EOF and discards, so a
/// buffer nobody will read again cannot grow without bound in a daemon that
/// probes every 60 seconds.
///
/// Discarding rather than closing the read end is deliberate. Dropping the pipe
/// would send the still-writing grandchild `EPIPE`/`SIGPIPE`, which for the
/// background helpers a sandboxed installer legitimately leaves behind would
/// turn a leak we can bound into a behaviour change we cannot predict.
#[derive(Clone)]
struct Capture {
    buffer: Arc<Mutex<Vec<u8>>>,
    finished: Arc<AtomicBool>,
    failure: Arc<Mutex<Option<std::io::Error>>>,
    /// Set by `take()`: nothing will read this capture's bytes again.
    ///
    /// Read and written under `buffer`'s mutex, so "emptied and marked" is one
    /// step as far as the reader is concerned. Checking it outside the lock
    /// would leave the window where a reader that has just passed the check
    /// appends to the buffer `take()` is in the middle of emptying.
    abandoned: Arc<AtomicBool>,
}

impl Capture {
    /// Start draining `pipe` on its own thread.
    ///
    /// Publishing in chunks as they arrive, rather than returning the whole
    /// buffer at EOF, is what lets the caller take partial output from a reader
    /// that is still blocked.
    fn draining<R: Read + Send + 'static>(pipe: Option<R>) -> Self {
        let capture = Self {
            buffer: Arc::new(Mutex::new(Vec::new())),
            finished: Arc::new(AtomicBool::new(false)),
            failure: Arc::new(Mutex::new(None)),
            abandoned: Arc::new(AtomicBool::new(false)),
        };
        let Some(mut pipe) = pipe else {
            capture.finished.store(true, Ordering::Release);
            return capture;
        };
        let sink = capture.clone();
        thread::spawn(move || {
            let mut chunk = [0_u8; READ_CHUNK];
            loop {
                match pipe.read(&mut chunk) {
                    Ok(0) => break,
                    Ok(read) => {
                        // Keep draining once abandoned, but stop accumulating:
                        // see the type's doc. The check shares the buffer's
                        // lock with `take()`, so after `take()` returns this
                        // buffer cannot grow by another byte.
                        let mut buffer = sink.lock();
                        if !sink.abandoned.load(Ordering::Acquire) {
                            buffer.extend_from_slice(&chunk[..read]);
                        }
                    }
                    // `read_to_end` — what `wait_with_output` used before this
                    // function existed — retries an interrupted read and
                    // returns every other error. Treating either as EOF would
                    // silently shorten the output instead, and the callers
                    // parse these bytes as JSON, where that surfaces as a
                    // parse error pointing at the wrong thing.
                    // Round again rather than break: not the end of anything.
                    Err(error) if error.kind() == ErrorKind::Interrupted => {}
                    Err(error) => {
                        // Recorded only while someone is still listening: after
                        // `take()` the failure slot is as unread as the buffer.
                        if !sink.abandoned.load(Ordering::Acquire) {
                            *sink.failure() = Some(error);
                        }
                        break;
                    }
                }
            }
            sink.finished.store(true, Ordering::Release);
        });
        capture
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<u8>> {
        self.buffer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn failure(&self) -> std::sync::MutexGuard<'_, Option<std::io::Error>> {
        self.failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }

    /// Take what has been captured, leaving a still-running reader unblocked.
    ///
    /// The bytes come back alongside any read error rather than instead of it,
    /// so a caller already reporting something else — a timeout — can still
    /// quote what it managed to collect.
    ///
    /// This is also the point of no return for the reader: whatever it collects
    /// after this is discarded, because there is no second caller to hand it to.
    fn take(&self) -> (Vec<u8>, Option<std::io::Error>) {
        let bytes = {
            // Marked while the buffer is held, so the reader cannot append
            // between the mark and the emptying and strand bytes in a buffer
            // that is never read again.
            let mut buffer = self.lock();
            self.abandoned.store(true, Ordering::Release);
            std::mem::take(&mut *buffer)
        };
        (bytes, self.failure().take())
    }
}

/// Run `command` to completion with both output streams captured, giving up
/// after `timeout`.
///
/// A thread per pipe drains stdout and stderr, and both are started *before*
/// the wait loop. That ordering is the whole point. Waiting for the child to
/// exit and only then calling `wait_with_output` — the shape this replaced —
/// deadlocks: the pipes are bounded by the OS (commonly ~64KiB, less on
/// Windows), so a child that writes more than that blocks in `write`, which
/// means it never exits, which means the drain is never reached. The only way
/// out was the timeout, so it measured how much the child printed rather than
/// how long it took, and killed successful commands mid-flight.
///
/// Both pipes need their own reader for the same reason: draining stdout to
/// completion and only then reading stderr stalls on a child that fills stderr
/// first.
///
/// The call returns within roughly `timeout` either way: once the child is no
/// longer running the readers get `DRAIN_GRACE` to finish and are then abandoned
/// rather than joined. Joining unconditionally would hand the schedule to a
/// grandchild the child left behind, which holds the pipe's write end open so no
/// EOF ever arrives — that is how an abandoned `sleep` comes to decide how long
/// a 2-second probe takes.
///
/// The stdio configuration is this function's own: it sets null stdin and piped
/// stdout/stderr on `command`, overriding whatever the caller set, because
/// capturing both streams is the whole contract. Callers need not — and should
/// not — set them.
///
/// `label` names the subject in error messages (e.g. `"process"`,
/// `"sandbox process"`).
pub fn run_with_timeout(mut command: Command, timeout: Duration, label: &str) -> Result<Output> {
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to spawn {label}"))?;

    // Taking the pipes here is what obliges us to assemble `Output` by hand
    // below: `wait_with_output` reads the handles it owns, and these are gone.
    let stdout = Capture::draining(child.stdout.take());
    let stderr = Capture::draining(child.stderr.take());

    let deadline = Instant::now() + timeout;
    let mut timed_out = false;
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .with_context(|| format!("failed to poll {label}"))?
        {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            timed_out = true;
            break child
                .wait()
                .with_context(|| format!("failed to reap timed-out {label}"))?;
        }
        thread::sleep(POLL_INTERVAL);
    };

    let drained_by = Instant::now() + DRAIN_GRACE;
    while !(stdout.is_finished() && stderr.is_finished()) && Instant::now() < drained_by {
        thread::sleep(POLL_INTERVAL.min(DRAIN_GRACE / 4));
    }
    assemble_output(
        status,
        timed_out,
        stdout.take(),
        stderr.take(),
        label,
        timeout,
    )
}

/// Turn what the wait loop and the two readers produced into the call's result.
///
/// Split out from [`run_with_timeout`] because it is the only way to test it.
/// Every branch here is plain logic over values already in hand — which error
/// wins, which stream gets quoted — but reaching it through a real child means
/// making a pipe fail on demand, which needs a kernel having a bad day. Given
/// the two `(bytes, error)` pairs directly, each branch is an ordinary unit
/// test; see the cases at the bottom of this file.
fn assemble_output(
    status: ExitStatus,
    timed_out: bool,
    (stdout, stdout_failure): (Vec<u8>, Option<std::io::Error>),
    (stderr, stderr_failure): (Vec<u8>, Option<std::io::Error>),
    label: &str,
    timeout: Duration,
) -> Result<Output> {
    if timed_out {
        let quoted_stderr = String::from_utf8_lossy(&stderr).trim().to_owned();
        let quoted_stdout = String::from_utf8_lossy(&stdout).trim().to_owned();
        // Truncated, not dropped: what the child managed to say before it
        // stalled is usually the whole explanation, and the timeout is named
        // first either way, so a reader who only sees the head still learns
        // which command gave up and why.
        //
        // stderr first because that is where a child explains itself; stdout
        // only as a fallback, since a child whose whole output is data has
        // nothing else to offer.
        bail!(
            "{label} exceeded {}s timeout: {}",
            timeout.as_secs(),
            truncate_to_chars(
                if !quoted_stderr.is_empty() {
                    quoted_stderr
                } else if !quoted_stdout.is_empty() {
                    quoted_stdout
                } else {
                    "no output".to_owned()
                },
                MAX_QUOTED_OUTPUT_CHARS,
            )
        );
    }

    // Only once the timeout has had its say: a killed child whose pipe then
    // fails to read is a timeout, and reporting it as a read error would name
    // the symptom instead of the cause.
    if let Some(error) = stdout_failure.or(stderr_failure) {
        return Err(anyhow::Error::new(error))
            .with_context(|| format!("failed to read the output of {label}"));
    }

    Ok(Output {
        status,
        stdout,
        stderr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shells every grandchild case below is run against.
    ///
    /// Whether a shell forks or execs decides whether a grandchild exists at
    /// all, so it decides whether these tests test anything: dash forks `sleep`
    /// in `sh -c "sleep 30"`, bash execs it into the shell's own process and no
    /// grandchild is left holding the pipe. A test written against one is dead
    /// weight on a host running the other, and which one `/bin/sh` is varies
    /// across the distributions this ships on. So the scripts below are written
    /// to fork under either, and each case runs under every shell present to
    /// keep that true.
    #[cfg(unix)]
    fn grandchild_shells() -> Vec<&'static str> {
        let shells: Vec<&'static str> = ["/bin/sh", "/bin/bash", "/bin/dash"]
            .into_iter()
            .filter(|shell| std::path::Path::new(shell).exists())
            .collect();
        assert!(
            !shells.is_empty(),
            "no shell to run the grandchild cases in"
        );
        shells
    }

    #[test]
    fn a_command_that_succeeds_reports_its_output() {
        let mut command = Command::new(if cfg!(windows) { "cmd" } else { "sh" });
        if cfg!(windows) {
            // No space around `&`: cmd would carry a trailing one into the
            // echoed text, which `trim` hides on stdout but is easier not to
            // write than to explain.
            command.args(["/C", "echo hello&echo oops 1>&2"]);
        } else {
            command.args(["-c", "echo hello; echo oops >&2"]);
        }

        let output = run_with_timeout(command, Duration::from_secs(10), "process").expect("ran");

        assert!(output.status.success());
        assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "hello");
        // Asserted on Windows too: `1>&2` is a cmd redirection like any other,
        // and skipping it there left the second pipe — the one whose separate
        // reader this function exists to provide — unchecked on that platform.
        assert_eq!(String::from_utf8_lossy(&output.stderr).trim(), "oops");
    }

    #[cfg(unix)]
    #[test]
    fn a_genuinely_slow_command_times_out_on_time() {
        for shell in grandchild_shells() {
            // `sleep 30 & wait` rather than plain `sleep 30`: a lone final
            // command is exec-ed into the shell itself by bash, which would
            // leave nothing behind to hold the pipe and quietly turn this into
            // a test of the easy case. Backgrounding forces a real grandchild
            // under every shell.
            //
            // Killing the shell at the deadline does not kill that grandchild,
            // so it keeps the inherited write end of both pipes open and no EOF
            // arrives. An implementation that joins its readers therefore waits
            // out the full 30 seconds and the timeout bounds nothing — which
            // for the 2-second amd-smi probe is the difference between a snappy
            // telemetry read and a stalled one.
            let mut command = Command::new(shell);
            command.args(["-c", "sleep 30 & wait"]);

            let started = Instant::now();
            let error = run_with_timeout(command, Duration::from_secs(1), "process")
                .expect_err("a command that outlives its timeout must fail");
            let elapsed = started.elapsed();

            assert!(
                error.to_string().contains("exceeded 1s timeout"),
                "the timeout must be reported as one, under {shell}: {error}"
            );
            assert!(
                elapsed < Duration::from_secs(5),
                "the call must return at its deadline rather than waiting for an abandoned \
                 grandchild to exit, but under {shell} it took {elapsed:?}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_grandchild_holding_the_pipe_does_not_hold_a_successful_call() {
        for shell in grandchild_shells() {
            // The timeout case above covers a child that is killed. This is the
            // same hazard on the path that has no timeout to blame: the child
            // exits 0 straight away, having left a background `sleep` holding
            // the write end of both pipes, so there is no EOF for another five
            // seconds. Joining the readers — or giving them an unbounded grace
            // — hands the schedule to that `sleep` and a command that finished
            // instantly takes 5s. `DRAIN_GRACE` is what bounds it.
            let mut command = Command::new(shell);
            command.args(["-c", "echo ready; sleep 5 &"]);

            let started = Instant::now();
            let output = run_with_timeout(command, Duration::from_secs(20), "process")
                .expect("a child that exits cleanly must not be reported as a timeout");
            let elapsed = started.elapsed();

            assert!(output.status.success(), "under {shell}: {output:?}");
            assert_eq!(
                String::from_utf8_lossy(&output.stdout).trim(),
                "ready",
                "what the child wrote before exiting must still arrive, under {shell}"
            );
            // Generous against a loaded machine, and still a third of the
            // grandchild's lifetime: the regression this guards takes the full
            // five seconds, not a few hundred milliseconds more than the grace.
            assert!(
                elapsed < Duration::from_secs(3),
                "the call must not wait for the grandchild, but under {shell} it took {elapsed:?}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn output_arriving_after_the_child_exits_is_still_collected() {
        for shell in grandchild_shells() {
            // What `DRAIN_GRACE` is for, and the only way to ask for it on
            // demand. A child that writes and exits normally cannot pin it: its
            // reader is already running and drains the pipe microseconds later,
            // long before the wait loop's next poll even notices the exit, so
            // taking the buffer the instant the child is seen to have exited
            // still collects everything. Here a grandchild writes once the
            // child is gone instead.
            //
            // The 0.1s sits between the two bounds that decide the outcome:
            // past `POLL_INTERVAL` (50ms), the longest the wait loop can take
            // to notice the exit, so taking the output there misses
            // `late-canary` outright; and well inside `DRAIN_GRACE` (200ms), so
            // draining collects it. Both margins are ~2x, and the grandchild
            // exits immediately after writing, so a working implementation sees
            // EOF and returns rather than sitting out the whole grace.
            let mut command = Command::new(shell);
            command.args(["-c", "echo early-canary; (sleep 0.1; echo late-canary) &"]);

            let started = Instant::now();
            let output = run_with_timeout(command, Duration::from_secs(20), "process")
                .expect("a child that exits cleanly must not be reported as a timeout");
            let elapsed = started.elapsed();
            let stdout = String::from_utf8_lossy(&output.stdout);

            assert!(
                stdout.contains("early-canary"),
                "under {shell}, the child's own output is missing: {stdout:?}"
            );
            assert!(
                stdout.contains("late-canary"),
                "under {shell}, output still in flight when the child exited was dropped \
                 instead of drained: {stdout:?}"
            );
            assert!(
                elapsed < Duration::from_secs(3),
                "draining must end at EOF rather than running long, but under {shell} the call \
                 took {elapsed:?}"
            );
        }
    }

    #[test]
    fn an_interrupted_read_is_retried_and_a_failed_one_is_reported() {
        // Neither branch is reachable through a real child here — EINTR needs a
        // signal to land mid-read, and a pipe read failing outright needs the
        // kernel to be having a bad day — so the reader is fed a scripted
        // stream directly. The property is the one `read_to_end` gave before
        // this function replaced it: an interrupted read is not the end of the
        // stream, and a failed one is not a successful short read.
        struct ScriptedPipe(std::collections::VecDeque<std::io::Result<&'static [u8]>>);

        impl Read for ScriptedPipe {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                match self.0.pop_front() {
                    Some(Ok(bytes)) => {
                        out[..bytes.len()].copy_from_slice(bytes);
                        Ok(bytes.len())
                    }
                    Some(Err(error)) => Err(error),
                    None => Ok(0),
                }
            }
        }

        fn drain(script: Vec<std::io::Result<&'static [u8]>>) -> (Vec<u8>, Option<std::io::Error>) {
            let capture = Capture::draining(Some(ScriptedPipe(script.into())));
            let deadline = Instant::now() + Duration::from_secs(10);
            while !capture.is_finished() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(5));
            }
            assert!(capture.is_finished(), "the reader never finished");
            capture.take()
        }

        let (bytes, failure) = drain(vec![
            Ok(b"before"),
            Err(std::io::Error::from(ErrorKind::Interrupted)),
            Ok(b"after"),
            Ok(b""),
        ]);
        assert_eq!(
            String::from_utf8_lossy(&bytes),
            "beforeafter",
            "an interrupted read must be retried, not taken for the end of the stream"
        );
        assert!(failure.is_none(), "{failure:?}");

        let (bytes, failure) = drain(vec![
            Ok(b"partial"),
            Err(std::io::Error::from(ErrorKind::BrokenPipe)),
            Ok(b"unreachable"),
        ]);
        assert_eq!(String::from_utf8_lossy(&bytes), "partial");
        assert_eq!(
            failure.map(|error| error.kind()),
            Some(ErrorKind::BrokenPipe),
            "a read that failed must be reported rather than passed off as EOF"
        );
    }

    /// A pipe that never ends until the test says so.
    ///
    /// Stoppable rather than genuinely endless so the reader thread finishes
    /// with the test instead of spinning for the life of the test binary.
    struct EndlessPipe {
        reads: Arc<std::sync::atomic::AtomicUsize>,
        stop: Arc<AtomicBool>,
    }

    impl Read for EndlessPipe {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            if self.stop.load(Ordering::Acquire) {
                return Ok(0);
            }
            // Paced so the test can interleave with the reader rather than
            // racing a tight loop, and so a regression accumulates a size the
            // assertion can see without the test allocating gigabytes.
            thread::sleep(Duration::from_millis(1));
            let filled = out.len().min(4096);
            out[..filled].fill(b'x');
            self.reads.fetch_add(1, Ordering::Release);
            Ok(filled)
        }
    }

    #[test]
    fn an_abandoned_reader_stops_filling_a_buffer_nobody_will_read() {
        // The shape this guards: the child exited but a grandchild still holds
        // the write end, so `DRAIN_GRACE` expires, `take()` hands the caller
        // what there is and the reader is left running. In `rocmd` — a daemon
        // that probes amd-smi every 60s and runs MCP captures besides — a
        // reader that kept appending would grow a `Vec` that nothing will ever
        // read, for as long as the grandchild lives. There is no way to ask
        // `run_with_timeout` for that state on demand, so the capture is driven
        // directly, which is also where the invariant lives.
        let reads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let capture = Capture::draining(Some(EndlessPipe {
            reads: Arc::clone(&reads),
            stop: Arc::clone(&stop),
        }));

        let wait_for_reads = |target: usize| {
            let deadline = Instant::now() + Duration::from_secs(10);
            while reads.load(Ordering::Acquire) < target && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(5));
            }
            assert!(
                reads.load(Ordering::Acquire) >= target,
                "the reader stalled before {target} reads"
            );
        };

        wait_for_reads(4);
        let (bytes, failure) = capture.take();
        // Establishes that the reader really was accumulating up to this point,
        // so the emptiness asserted below is the flag working rather than a
        // reader that never ran.
        assert!(
            !bytes.is_empty(),
            "the reader must collect while someone is still listening"
        );
        assert!(failure.is_none(), "{failure:?}");

        // Well past what fits in anything the implementation might buffer
        // internally: 50 more 4KiB reads is ~200KB, which a regression leaves
        // sitting in the buffer and a correct implementation discards.
        wait_for_reads(reads.load(Ordering::Acquire) + 50);
        let leaked = capture.lock().len();
        assert_eq!(
            leaked, 0,
            "after `take()` nothing will read this buffer again, so a reader that keeps \
             appending grows it without bound for the life of the daemon; {leaked} bytes \
             accumulated"
        );

        stop.store(true, Ordering::Release);
        let deadline = Instant::now() + Duration::from_secs(10);
        while !capture.is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert!(capture.is_finished(), "the reader never finished");
    }

    /// A successful `ExitStatus` to feed `assemble_output`.
    ///
    /// `ExitStatus` has no portable public constructor; the per-platform
    /// `ExitStatusExt` is the supported way to make one without spawning a
    /// process purely to obtain it.
    fn success_status() -> ExitStatus {
        #[cfg(unix)]
        use std::os::unix::process::ExitStatusExt as _;
        #[cfg(windows)]
        use std::os::windows::process::ExitStatusExt as _;
        ExitStatus::from_raw(0)
    }

    fn read_failure() -> Option<std::io::Error> {
        Some(std::io::Error::from(ErrorKind::BrokenPipe))
    }

    #[test]
    fn a_read_failure_without_a_timeout_is_reported_rather_than_returned_as_output() {
        // The silent short read this function exists to prevent: handing back
        // `Ok` with a truncated buffer sends the caller off to parse half a
        // JSON document and blame the document.
        let error = assemble_output(
            success_status(),
            false,
            (b"partial".to_vec(), read_failure()),
            (Vec::new(), None),
            "process",
            Duration::from_secs(2),
        )
        .expect_err("a stream that failed to read must not come back as success");

        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("failed to read the output of process"),
            "{rendered:?}"
        );
    }

    #[test]
    fn a_failure_on_either_stream_is_reported() {
        // stdout is checked first, so a stderr-only failure is a separate
        // branch: `stdout_failure.or(stderr_failure)` reduced to
        // `stdout_failure` would still pass the case above.
        let error = assemble_output(
            success_status(),
            false,
            (Vec::new(), None),
            (b"partial".to_vec(), read_failure()),
            "process",
            Duration::from_secs(2),
        )
        .expect_err("a stderr stream that failed to read must not come back as success");

        assert!(
            format!("{error:#}").contains("failed to read the output of process"),
            "{error:#}"
        );
    }

    #[test]
    fn a_timeout_outranks_a_read_failure() {
        // Killing the child is what broke the pipe, so a read error here is the
        // symptom. Reporting it would name the consequence and hide the cause,
        // and the caller would go looking for a kernel problem.
        let error = assemble_output(
            success_status(),
            true,
            (Vec::new(), read_failure()),
            (b"the real explanation".to_vec(), read_failure()),
            "process",
            Duration::from_secs(2),
        )
        .expect_err("a timed-out call must fail");

        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("exceeded 2s timeout"),
            "the timeout is the cause and must be what is reported: {rendered:?}"
        );
        assert!(
            rendered.contains("the real explanation"),
            "the child's own account must survive: {rendered:?}"
        );
        assert!(
            !rendered.contains("failed to read the output of"),
            "the read error is downstream of the kill and must not displace it: {rendered:?}"
        );
    }

    #[test]
    fn a_timeout_quotes_stderr_in_preference_to_stdout() {
        let error = assemble_output(
            success_status(),
            true,
            (b"stdout-data".to_vec(), None),
            (b"stderr-explanation".to_vec(), None),
            "process",
            Duration::from_secs(2),
        )
        .expect_err("a timed-out call must fail");

        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("stderr-explanation"),
            "stderr is where a child explains itself: {rendered:?}"
        );
        assert!(
            !rendered.contains("stdout-data"),
            "quoting both would push the explanation behind a payload: {rendered:?}"
        );
    }

    #[test]
    fn a_timeout_falls_back_to_stdout_and_then_to_saying_there_was_none() {
        // Every other timeout fixture writes to stderr, so without these two
        // the fallback chain could be reordered or deleted unnoticed — and a
        // child whose whole output is stdout is exactly the amd-smi case.
        let error = assemble_output(
            success_status(),
            true,
            (b"stdout-only".to_vec(), None),
            (Vec::new(), None),
            "process",
            Duration::from_secs(2),
        )
        .expect_err("a timed-out call must fail");
        assert!(
            format!("{error:#}").contains("stdout-only"),
            "a child that only wrote stdout must still be quoted: {error:#}"
        );

        let error = assemble_output(
            success_status(),
            true,
            (Vec::new(), None),
            (Vec::new(), None),
            "process",
            Duration::from_secs(2),
        )
        .expect_err("a timed-out call must fail");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("no output"),
            "a silent child must say so rather than trailing a bare colon: {rendered:?}"
        );
        assert!(rendered.contains("exceeded 2s timeout"), "{rendered:?}");
    }

    #[test]
    fn a_clean_run_returns_both_streams() {
        let output = assemble_output(
            success_status(),
            false,
            (b"out".to_vec(), None),
            (b"err".to_vec(), None),
            "process",
            Duration::from_secs(2),
        )
        .expect("a clean run must succeed");

        assert!(output.status.success());
        assert_eq!(output.stdout, b"out");
        assert_eq!(output.stderr, b"err");
    }

    #[cfg(unix)]
    #[test]
    fn a_child_that_floods_both_pipes_does_not_deadlock() {
        // The defect this function exists to prevent. Both pipes are bounded by
        // the OS (commonly ~64KiB), so a child writing past that blocks in
        // `write` and never exits — and a version of this function that drains
        // only after the child exits never reads a byte, so the child is killed
        // at the timeout and a successful command is reported as too slow.
        //
        // Flooding *both* streams is deliberate: a single-pipe child still
        // passes against an implementation that drains stdout fully and only
        // then starts on stderr, which is its own deadlock.
        //
        // `sh` rather than a real subject program (amd-smi, an SDK install),
        // because the property is about this function's pipe handling and
        // nothing else, and a shell makes it a unit test rather than one that
        // needs a multi-GPU host.
        let mut command = Command::new("sh");
        command.args([
            "-c",
            "yes deadlock-canary | head -c 2000000; yes deadlock-canary | head -c 2000000 >&2",
        ]);

        // Short, because a regression fails by exhausting this timeout: ten
        // seconds is far past the ~0.1s a working drain takes, and far short of
        // stalling the suite.
        let output = run_with_timeout(command, Duration::from_secs(10), "process")
            .expect("a child that outruns the pipe buffer must not be killed as a timeout");

        assert!(output.status.success(), "{output:?}");
        // Without these the test would keep passing if the child ever stopped
        // producing more than one buffer, which is to say it would stop testing
        // the hazard.
        assert!(
            output.stdout.len() > 64 * 1024,
            "stdout must exceed one pipe buffer: {} bytes",
            output.stdout.len()
        );
        assert!(
            output.stderr.len() > 64 * 1024,
            "stderr must exceed one pipe buffer: {} bytes",
            output.stderr.len()
        );
    }

    /// How large a quoted-output assertion lets the whole error message get.
    ///
    /// Comfortably above `MAX_QUOTED_OUTPUT_CHARS` plus the label and marker,
    /// and two orders of magnitude below the ~200KB the fixtures below feed in,
    /// so the assertion answers "is it bounded at all" rather than pinning the
    /// exact constant — which is a tuning decision, not a contract.
    const QUOTE_ASSERTION_CEILING: usize = 4000;

    /// The flooding fixtures below write this many bytes, past one pipe buffer
    /// and ~100x the quote bound.
    const FLOOD_BYTES: usize = 200_000;

    const FLOOD_CANARY: &str = "flood-canary";

    /// The assertions shared by both platforms' timeout-quote cases.
    ///
    /// Checked together because they trade off against each other: a bound that
    /// kept nothing would satisfy the size check while making the message
    /// useless, and quoting everything keeps it useful while costing the logs
    /// the thing this bound exists to protect.
    fn assert_timeout_quote_is_bounded_and_useful(message: &str, timeout_secs: u64) {
        assert!(
            message.contains(&format!("exceeded {timeout_secs}s timeout")),
            "the timeout must still be named: {message:?}"
        );
        assert!(
            message.contains(FLOOD_CANARY),
            "the child's own output must still be quoted, not dropped: {message:?}"
        );
        assert!(
            message.contains("[truncated]"),
            "a shortened quote must say so, or a reader takes the head for the whole: {message:?}"
        );
        let length = message.chars().count();
        assert!(
            length <= QUOTE_ASSERTION_CEILING,
            "the quote must be bounded: the message is {length} characters, from a child that \
             printed {FLOOD_BYTES} bytes"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_timed_out_child_has_its_output_quoted_within_a_bound() {
        for shell in grandchild_shells() {
            // Both halves are load-bearing. The flood is what makes the quote
            // large; the backgrounded `sleep` is what makes this a timeout,
            // which is the only path that quotes the child's output at all.
            //
            // Unbounded, this message travels: `rocmd` renders the error chain
            // into the bridge snapshot's `note`, and the GPU-metrics watcher
            // writes that note into two JSONL logs every 60s, rewriting each
            // file whole. A real amd-smi on a multi-GPU host prints well past
            // this fixture's size.
            let mut command = Command::new(shell);
            command.args([
                "-c",
                &format!("yes {FLOOD_CANARY} | head -c {FLOOD_BYTES} >&2; sleep 30 & wait"),
            ]);

            let error = run_with_timeout(command, Duration::from_secs(1), "process")
                .expect_err("a command that outlives its timeout must fail");

            assert_timeout_quote_is_bounded_and_useful(&format!("{error:#}"), 1);
        }
    }

    /// A scratch directory holding a Windows fixture's script and payload.
    ///
    /// Per-process unique so concurrent test binaries cannot collide on it, and
    /// removed on drop so a failing assertion does not leave the payload behind.
    #[cfg(windows)]
    struct WindowsFixture {
        dir: std::path::PathBuf,
    }

    #[cfg(windows)]
    impl WindowsFixture {
        /// Write `script` as a batch file next to a `FLOOD_BYTES` payload.
        ///
        /// A batch file rather than `cmd /C "<command line>"`: cmd's rules for
        /// stripping the quotes Rust adds around an argument containing spaces
        /// and redirections are subtle enough that the fixture would become the
        /// thing under test. A bare path has none of that, and `%~dp0` inside
        /// the script resolves the payload without the test passing a path in.
        fn new(name: &str, script: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("rocm-cli-process-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("failed to create the fixture directory");

            let line = format!("{FLOOD_CANARY}\r\n");
            let mut payload = String::with_capacity(FLOOD_BYTES + line.len());
            while payload.len() < FLOOD_BYTES {
                payload.push_str(&line);
            }
            std::fs::write(dir.join("payload.txt"), payload).expect("failed to write the payload");
            std::fs::write(dir.join("fixture.bat"), script).expect("failed to write the script");

            Self { dir }
        }

        fn command(&self) -> Command {
            let mut command = Command::new("cmd");
            command.args(["/C"]).arg(self.dir.join("fixture.bat"));
            command
        }
    }

    #[cfg(windows)]
    impl Drop for WindowsFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    #[cfg(windows)]
    #[test]
    fn a_child_that_floods_both_pipes_does_not_deadlock() {
        // The Windows half of the unix case above, and the same property: both
        // pipes are bounded by the OS here too — more tightly than on Linux —
        // so an implementation that drains only after the child exits, or
        // drains stdout to completion before starting on stderr, never lets
        // this child finish. `type` twice is the shell-free equivalent of the
        // unix `yes | head`; cmd has no `yes`.
        let fixture = WindowsFixture::new(
            "flood",
            "@echo off\r\ntype \"%~dp0payload.txt\"\r\ntype \"%~dp0payload.txt\" 1>&2\r\n",
        );

        let output = run_with_timeout(fixture.command(), Duration::from_secs(30), "process")
            .expect("a child that outruns the pipe buffer must not be killed as a timeout");

        assert!(output.status.success(), "{output:?}");
        assert!(
            output.stdout.len() > 64 * 1024,
            "stdout must exceed one pipe buffer: {} bytes",
            output.stdout.len()
        );
        assert!(
            output.stderr.len() > 64 * 1024,
            "stderr must exceed one pipe buffer: {} bytes",
            output.stderr.len()
        );
    }

    #[cfg(windows)]
    #[test]
    fn a_timed_out_child_has_its_output_quoted_within_a_bound() {
        // Covers on Windows what three separate unix cases cover there, because
        // one fixture happens to exhibit all of it: `ping -n 30` runs for ~29
        // seconds, so the call times out; `ping` is a grandchild of the `cmd`
        // this function kills at the deadline, and killing a process on Windows
        // does not kill its children, so it keeps the inherited write ends open
        // and no EOF arrives — an implementation that joins its readers waits
        // the full 29 seconds; and the preceding flood makes the resulting
        // quote large enough for the bound to matter.
        let fixture = WindowsFixture::new(
            "timeout",
            "@echo off\r\ntype \"%~dp0payload.txt\" 1>&2\r\nping -n 30 127.0.0.1\r\n",
        );

        let started = Instant::now();
        let error = run_with_timeout(fixture.command(), Duration::from_secs(2), "process")
            .expect_err("a command that outlives its timeout must fail");
        let elapsed = started.elapsed();

        assert_timeout_quote_is_bounded_and_useful(&format!("{error:#}"), 2);
        assert!(
            elapsed < Duration::from_secs(10),
            "the call must return at its deadline rather than waiting for an abandoned \
             grandchild to exit, but it took {elapsed:?}"
        );
    }
}
