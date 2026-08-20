// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Retry loop for sending an idempotent terminal input until its effect appears.

use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, Instant};

/// Borrowing future returned by the wait operation used by [`send_until`].
pub type WaitFuture<'a> = Pin<Box<dyn Future<Output = Result<(), String>> + Send + 'a>>;

/// Deadline and per-attempt wait used by [`send_until`].
#[derive(Clone, Copy, Debug)]
pub struct RetryTiming {
    pub timeout: Duration,
    pub resend_interval: Duration,
}

/// Result of checking whether a failed wait may be retried.
#[derive(Debug, PartialEq, Eq)]
pub enum TerminalState {
    /// The session is still live, so another send is allowed.
    Running,
    /// The session stopped, and the failed wait already carries the best error.
    Stopped,
    /// A terminal error landed after the wait and supersedes its retryable error.
    Failed(String),
}

/// Send `bytes`, wait for `marker`, and retry until success, terminal state, or
/// the deadline. A terminal state returns the failed wait's exact error.
pub async fn send_until<S, SendInput, Wait, Terminal>(
    state: &mut S,
    bytes: &str,
    marker: &str,
    timing: RetryTiming,
    mut send: SendInput,
    mut wait: Wait,
    mut terminal: Terminal,
) -> Result<(), String>
where
    S: Send,
    SendInput: FnMut(&mut S, &str) -> Result<(), String> + Send,
    Wait: for<'a> FnMut(&'a mut S, &'a str, Duration) -> WaitFuture<'a> + Send,
    Terminal: FnMut(&mut S, &str) -> TerminalState + Send,
{
    let RetryTiming {
        timeout,
        resend_interval,
    } = timing;
    let deadline = Instant::now() + timeout;
    loop {
        send(state, bytes)?;
        let remaining = deadline.saturating_duration_since(Instant::now());
        let attempt = resend_interval.min(remaining);
        let last_error = match wait(state, marker, attempt).await {
            Ok(()) => return Ok(()),
            Err(error) => error,
        };
        match terminal(state, marker) {
            TerminalState::Running => {}
            TerminalState::Stopped => return Err(last_error),
            TerminalState::Failed(error) => return Err(error),
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "timed out after {timeout:?} waiting for {marker:?} while repeating {bytes:?}; last attempt: {last_error}"
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{RetryTiming, TerminalState, send_until};
    use crate::reader_failure::{ReaderFailure, ReaderFailureObservation};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    #[derive(Default)]
    struct TestState {
        failure: Arc<ReaderFailure>,
        reader_finished: bool,
        sends: usize,
        waits: usize,
    }

    #[tokio::test(flavor = "current_thread")]
    async fn captured_reader_panic_stops_retry_before_reader_thread_finishes() {
        const PANIC_ERROR: &str = "captured reader panic error";

        let mut state = TestState::default();
        state.failure.publish("captured reader panic".to_string());
        assert!(!state.reader_finished, "reader must still be finishing");

        let error = send_until(
            &mut state,
            "4",
            "marker",
            RetryTiming {
                timeout: Duration::from_millis(30),
                resend_interval: Duration::from_millis(5),
            },
            |state, _bytes| {
                state.sends += 1;
                Ok(())
            },
            |state, _marker, attempt| {
                Box::pin(async move {
                    state.waits += 1;
                    if state.failure.take_message().is_some() {
                        Err(PANIC_ERROR.to_string())
                    } else {
                        tokio::time::sleep(attempt).await;
                        Err("marker timeout".to_string())
                    }
                })
            },
            |state, _marker| match state.failure.observe() {
                ReaderFailureObservation::Message(error) => TerminalState::Failed(error),
                ReaderFailureObservation::FailedWithoutMessage => TerminalState::Stopped,
                ReaderFailureObservation::Running if state.reader_finished => {
                    TerminalState::Stopped
                }
                ReaderFailureObservation::Running => TerminalState::Running,
            },
        )
        .await
        .expect_err("reader panic must stop the retry loop");

        assert_eq!(error, PANIC_ERROR);
        assert_eq!(state.sends, 1, "terminal failure must not resend input");
        assert_eq!(state.waits, 1, "terminal failure must not wait again");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn terminal_error_published_as_wait_times_out_wins_over_marker_timeout() {
        const PANIC_ERROR: &str = "captured reader panic error";
        const MARKER_TIMEOUT: &str = "marker timeout";

        let failure = Arc::new(ReaderFailure::default());
        let reader_failure = Arc::clone(&failure);
        let (at_boundary_tx, at_boundary_rx) = mpsc::channel();
        let (published_tx, published_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let reader = std::thread::spawn(move || {
            at_boundary_rx
                .recv()
                .expect("wait should reach its final terminal check");
            reader_failure.publish(PANIC_ERROR.to_string());
            published_tx.send(()).expect("test waiter should remain");
            release_rx.recv().expect("test should release reader");
        });

        let mut state = TestState {
            failure: Arc::clone(&failure),
            ..TestState::default()
        };
        let error = send_until(
            &mut state,
            "4",
            "marker",
            RetryTiming {
                timeout: Duration::from_millis(30),
                resend_interval: Duration::from_millis(5),
            },
            |state, _bytes| {
                state.sends += 1;
                Ok(())
            },
            move |state, _marker, _attempt| {
                state.waits += 1;
                at_boundary_tx
                    .send(())
                    .expect("reader should wait for the boundary");
                published_rx
                    .recv()
                    .expect("reader should publish before wait returns");
                Box::pin(async { Err(MARKER_TIMEOUT.to_string()) })
            },
            |state, _marker| match state.failure.observe() {
                ReaderFailureObservation::Message(error) => TerminalState::Failed(error),
                ReaderFailureObservation::FailedWithoutMessage => TerminalState::Stopped,
                ReaderFailureObservation::Running if state.reader_finished => {
                    TerminalState::Stopped
                }
                ReaderFailureObservation::Running => TerminalState::Running,
            },
        )
        .await
        .expect_err("terminal reader error must stop the retry loop");

        assert!(!reader.is_finished(), "reader must still be finishing");
        release_tx.send(()).expect("reader should remain blocked");
        reader.join().expect("test reader should exit cleanly");

        assert_eq!(error, PANIC_ERROR);
        assert_eq!(state.sends, 1, "terminal failure must not resend input");
        assert_eq!(state.waits, 1, "terminal failure must not wait again");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn terminal_error_published_between_former_take_and_check_is_preserved() {
        const PANIC_ERROR: &str = "captured reader panic error";
        const MARKER_TIMEOUT: &str = "marker timeout";

        let failure = Arc::new(ReaderFailure::default());
        let reader_failure = Arc::clone(&failure);
        let (after_take_tx, after_take_rx) = mpsc::channel();
        let (published_tx, published_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let reader = std::thread::spawn(move || {
            after_take_rx
                .recv()
                .expect("terminal check should pass the former take point");
            reader_failure.publish(PANIC_ERROR.to_string());
            published_tx.send(()).expect("test waiter should remain");
            release_rx.recv().expect("test should release reader");
        });

        let mut state = TestState {
            failure: Arc::clone(&failure),
            ..TestState::default()
        };
        let error = send_until(
            &mut state,
            "4",
            "marker",
            RetryTiming {
                timeout: Duration::from_millis(30),
                resend_interval: Duration::from_millis(5),
            },
            |state, _bytes| {
                state.sends += 1;
                Ok(())
            },
            |state, _marker, _attempt| {
                state.waits += 1;
                Box::pin(async { Err(MARKER_TIMEOUT.to_string()) })
            },
            move |state, _marker| {
                assert_eq!(
                    state.failure.take_message(),
                    None,
                    "failure must publish after the former take point"
                );
                after_take_tx
                    .send(())
                    .expect("reader should wait for the boundary");
                published_rx
                    .recv()
                    .expect("reader should publish before atomic observation");
                match state.failure.observe() {
                    ReaderFailureObservation::Message(error) => TerminalState::Failed(error),
                    ReaderFailureObservation::FailedWithoutMessage => TerminalState::Stopped,
                    ReaderFailureObservation::Running => TerminalState::Running,
                }
            },
        )
        .await
        .expect_err("terminal reader error must stop the retry loop");

        assert!(!reader.is_finished(), "reader must still be finishing");
        release_tx.send(()).expect("reader should remain blocked");
        reader.join().expect("test reader should exit cleanly");

        assert_eq!(error, PANIC_ERROR);
        assert_eq!(state.sends, 1, "terminal failure must not resend input");
        assert_eq!(state.waits, 1, "terminal failure must not wait again");
    }
}
