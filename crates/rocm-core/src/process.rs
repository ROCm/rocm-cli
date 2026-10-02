// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Bounded command execution that captures output without deadlocking on it.

use std::io::Read;
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

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

/// One captured stream: the bytes so far, and whether its reader saw EOF.
#[derive(Clone)]
struct Capture {
    buffer: Arc<Mutex<Vec<u8>>>,
    finished: Arc<AtomicBool>,
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
                    Ok(0) | Err(_) => break,
                    Ok(read) => sink.lock().extend_from_slice(&chunk[..read]),
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

    fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }

    /// Take what has been captured, leaving a still-running reader unblocked.
    fn take(&self) -> Vec<u8> {
        std::mem::take(&mut *self.lock())
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
    let stdout = stdout.take();
    let stderr = stderr.take();

    if timed_out {
        let stderr = String::from_utf8_lossy(&stderr).trim().to_owned();
        let stdout = String::from_utf8_lossy(&stdout).trim().to_owned();
        bail!(
            "{label} exceeded {}s timeout: {}",
            timeout.as_secs(),
            if !stderr.is_empty() {
                stderr
            } else if !stdout.is_empty() {
                stdout
            } else {
                "no output".to_owned()
            }
        );
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

    #[test]
    fn a_command_that_succeeds_reports_its_output() {
        let mut command = Command::new(if cfg!(windows) { "cmd" } else { "sh" });
        if cfg!(windows) {
            command.args(["/C", "echo hello"]);
        } else {
            command.args(["-c", "echo hello; echo oops >&2"]);
        }

        let output = run_with_timeout(command, Duration::from_secs(10), "process").expect("ran");

        assert!(output.status.success());
        assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "hello");
        if !cfg!(windows) {
            assert_eq!(String::from_utf8_lossy(&output.stderr).trim(), "oops");
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_genuinely_slow_command_times_out_on_time() {
        // `sh -c "sleep 30"` is not one process: dash forks `sleep` rather than
        // exec-ing it, so killing the shell at the deadline leaves `sleep`
        // holding the inherited write end of both pipes. There is therefore no
        // EOF, and an implementation that joins its readers before returning
        // waits out the full 30 seconds — the timeout stops bounding anything,
        // which for the 2-second amd-smi probe is the difference between a
        // snappy telemetry read and a stalled one.
        let mut command = Command::new("sh");
        command.args(["-c", "sleep 30"]);

        let started = Instant::now();
        let error = run_with_timeout(command, Duration::from_secs(1), "process")
            .expect_err("a command that outlives its timeout must fail");
        let elapsed = started.elapsed();

        assert!(
            error.to_string().contains("exceeded 1s timeout"),
            "the timeout must be reported as one: {error}"
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "the call must return at its deadline rather than waiting for an abandoned \
             grandchild to exit, but it took {elapsed:?}"
        );
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
}
