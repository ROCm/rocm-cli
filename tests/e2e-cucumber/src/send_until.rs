// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Retry loop for sending an idempotent terminal input until its effect appears.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use crate::polled_deadline::PolledDeadline;
pub use crate::polled_deadline::RetryTiming;

/// Borrowing future returned by the wait operation used by [`send_until`].
pub type WaitFuture<'a> = Pin<Box<dyn Future<Output = Result<(), String>> + Send + 'a>>;

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
///
/// The deadline is a [`PolledDeadline`], as in
/// [`quit_and_wait`](crate::quit_and_wait::quit_and_wait): if the first key was
/// lost, the resend is the only way the marker can appear, so a stall counted
/// against the timeout would give up before resending. Each wait must re-check
/// at least every `timing.poll_interval`.
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
    let mut deadline = PolledDeadline::start(timing);
    loop {
        send(state, bytes)?;
        let attempt = deadline.budget(timing.resend_interval);
        let last_error = match deadline.run(wait(state, marker, attempt)).await {
            Ok(()) => return Ok(()),
            Err(error) => error,
        };
        match terminal(state, marker) {
            TerminalState::Running => {}
            TerminalState::Stopped => return Err(last_error),
            TerminalState::Failed(error) => return Err(error),
        }
        if let Some(expiry) = deadline.expired() {
            return Err(format!(
                "timed out after {expiry} waiting for {marker:?} while repeating {bytes:?}; last attempt: {last_error}"
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{RetryTiming, TerminalState, send_until};
    use crate::polled_deadline::test_support::{TICK, credited, stalls_once, ticking_wait};
    use crate::reader_failure::{ReaderFailure, ReaderFailureObservation};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;
    use tokio::time::Instant;

    #[derive(Default)]
    struct TestState {
        failure: Arc<ReaderFailure>,
        reader_finished: bool,
        sends: usize,
        waits: usize,
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
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
                poll_interval: Duration::from_millis(5),
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

    #[tokio::test(flavor = "current_thread", start_paused = true)]
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
                poll_interval: Duration::from_millis(5),
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

    #[tokio::test(flavor = "current_thread", start_paused = true)]
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
                poll_interval: Duration::from_millis(5),
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

    /// A screen that shows its marker once a key that was not lost has
    /// arrived. The first `lost` keys sent are swallowed.
    struct LossyScreen {
        lost: usize,
        sends: usize,
    }

    /// Repeat a key into `screen`, stalling each wait by whatever `stall`
    /// returns, as when another scenario's synchronous step holds cucumber's
    /// task.
    async fn send_into(
        screen: &mut LossyScreen,
        timing: RetryTiming,
        mut stall: impl FnMut() -> Duration + Send,
    ) -> Result<(), String> {
        let run = send_until(
            screen,
            "4",
            "marker",
            timing,
            |screen, _bytes| {
                screen.sends += 1;
                Ok(())
            },
            |screen, _marker, attempt| {
                let stall = stall();
                Box::pin(async move {
                    let shown = screen.sends > screen.lost;
                    if ticking_wait(attempt, stall, || shown).await {
                        Ok(())
                    } else {
                        Err("marker timeout".to_string())
                    }
                })
            },
            |_, _| TerminalState::Running,
        );
        tokio::time::timeout(Duration::from_secs(5), run)
            .await
            .expect("send_until must end on its own")
    }

    fn timing(timeout_ms: u64) -> RetryTiming {
        timing_with_resend(timeout_ms, 50)
    }

    fn timing_with_resend(timeout_ms: u64, resend_ms: u64) -> RetryTiming {
        RetryTiming {
            timeout: Duration::from_millis(timeout_ms),
            resend_interval: Duration::from_millis(resend_ms),
            poll_interval: TICK,
        }
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_lost_key_is_resent_after_a_stall_longer_than_the_timeout() {
        // The first key is lost, and the first wait comes back only after the
        // whole timeout. Counting that stall gives up before the resend that
        // is the only way the marker can appear.
        let mut screen = LossyScreen { lost: 1, sends: 0 };

        let result = send_into(
            &mut screen,
            timing(300),
            stalls_once(Duration::from_millis(400)),
        )
        .await;

        assert_eq!(result, Ok(()));
        assert_eq!(screen.sends, 2);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_lost_key_is_resent_after_a_stall_inside_a_long_budget() {
        // The resend interval outlasts the 300 ms timeout, so the first wait's
        // budget is the whole timeout, and a 250 ms stall lands inside it: the
        // wait comes back on time, never past its budget, with the first key
        // lost. The stall must still be credited, or the timeout has passed
        // and the loop gives up before the resend.
        let mut screen = LossyScreen { lost: 1, sends: 0 };

        let result = send_into(
            &mut screen,
            timing_with_resend(300, 400),
            stalls_once(Duration::from_millis(250)),
        )
        .await;

        assert_eq!(result, Ok(()));
        assert_eq!(screen.sends, 2);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_wait_longer_than_the_time_left_is_cut_to_the_deadline() {
        // The resend interval is 50 times the timeout. Without the cut, the
        // first wait runs its whole 5 s interval past the 100 ms deadline.
        let mut screen = LossyScreen {
            lost: usize::MAX,
            sends: 0,
        };
        let started = Instant::now();

        let error = send_into(&mut screen, timing_with_resend(100, 5000), || {
            Duration::ZERO
        })
        .await
        .expect_err("a marker that never appears must time out");

        let elapsed = started.elapsed();
        assert!(error.starts_with("timed out after 100ms "), "{error}");
        assert!(
            elapsed <= Duration::from_millis(100) + TICK,
            "the wait must stop at the 100ms deadline, took {elapsed:?}"
        );
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_marker_that_never_appears_times_out_and_names_the_stall() {
        let mut screen = LossyScreen {
            lost: usize::MAX,
            sends: 0,
        };

        let error = send_into(
            &mut screen,
            timing(100),
            stalls_once(Duration::from_millis(150)),
        )
        .await
        .expect_err("a marker that never appears must time out");

        let stalled = credited(Duration::from_millis(150));
        assert!(
            error.starts_with(&format!(
                "timed out after 100ms of polling, not counting {stalled:?} the harness left this step unpolled"
            )),
            "{error}"
        );
        assert!(error.ends_with("last attempt: marker timeout"), "{error}");
    }
}
