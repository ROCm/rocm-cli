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
mod diagnostic_wording {
    //! The TUI driver's wait diagnostics are described by doc comments in
    //! `tests/e2e/tui_driver.rs` — which templates exist, and what shape of
    //! label each expects. Nothing ran those descriptions, so three review
    //! rounds in a row found them describing code they no longer matched, each
    //! time only because a person read both.
    //!
    //! These read the driver source and assert the templates the docs name are
    //! the templates that exist. Reword a message and this fails, naming the
    //! doc that has gone stale; the doc is then wrong for as long as it takes
    //! to run the tests, rather than until someone notices.
    //!
    //! Source text rather than rendered output on purpose: rendering one needs
    //! a live pty, a child process and a reader thread, which is what the
    //! scenarios themselves are for. What rots here is the wording, and the
    //! wording is in the file.

    fn driver_source() -> String {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/e2e/tui_driver.rs");
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
    }

    /// A format placeholder as it appears in source: `arg("describe")` is
    /// `{describe}`. Built rather than written so no literal in this file
    /// contains an uninterpolated `{…}`.
    fn arg(name: &str) -> String {
        format!("{{{name}}}")
    }

    /// The four templates `wait_for_screen_where`'s doc enumerates, which its
    /// contract requires a caller's `describe` clause to read correctly after.
    #[test]
    fn the_clause_templates_are_the_ones_the_contract_names() {
        let source = driver_source();
        // Each brace-delimited placeholder is spelled through `arg()` rather
        // than written into the literal: these are fragments of OTHER format
        // strings, and a literal containing `{describe}` reads to clippy as a
        // formatting argument nobody interpolated
        // (`literal_string_with_formatting_args`).
        for template in [
            format!("panicked while waiting until {}", arg("describe")),
            format!(
                "timed out after {} waiting until {}",
                arg("timeout:?"),
                arg("describe")
            ),
            format!("before {}.", arg("describe")),
            format!("draining the final frame{}", arg("context")),
            format!(", waiting until {}", arg("wanted")),
        ] {
            assert!(
                source.contains(&template),
                "tui_driver.rs no longer contains {template:?}, so the clause contract \
                 documented on `wait_for_screen_where` describes a template that is gone \
                 — update that doc comment with this change"
            );
        }
    }

    /// `terminal_state_after_wait` quotes its own bare marker, which its doc
    /// states is deliberate and distinct from the clause convention above.
    #[test]
    fn the_bare_marker_templates_still_quote_the_marker_themselves() {
        let source = driver_source();
        // Spelled through `arg()`, for the reason given in the sibling test.
        for template in [
            format!("panicked while waiting for {}", arg("marker:?")),
            format!("before {} appeared.", arg("marker:?")),
        ] {
            assert!(
                source.contains(&template),
                "tui_driver.rs no longer contains {template:?}, so the noun convention \
                 documented on `terminal_state_after_wait` describes a template that is \
                 gone — update that doc comment with this change"
            );
        }
    }

    /// The contract says the label is interpolated verbatim. A `{describe:?}`
    /// anywhere would quote it a second time on top of whatever the caller
    /// already put in, which is what the doc promises does not happen.
    #[test]
    fn the_clause_label_is_never_debug_formatted() {
        let source = driver_source();
        assert!(
            !source.contains(&arg("describe:?")),
            "a diagnostic Debug-formats `describe`, but `wait_for_screen_where` \
             documents it as interpolated verbatim — callers that quote their own \
             text would now be double-quoted"
        );
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
