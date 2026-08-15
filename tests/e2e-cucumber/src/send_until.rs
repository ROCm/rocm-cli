// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Retry loop for sending an idempotent terminal input until its effect appears.

use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, Instant};

/// Borrowing future returned by the wait operation used by [`send_until`].
pub type WaitFuture<'a> = Pin<Box<dyn Future<Output = Result<(), String>> + 'a>>;

/// Deadline and per-attempt wait used by [`send_until`].
#[derive(Clone, Copy, Debug)]
pub struct RetryTiming {
    pub timeout: Duration,
    pub resend_interval: Duration,
}

/// Send `bytes`, wait for `marker`, and retry until success, terminal state, or
/// the deadline. A terminal state returns the failed wait's exact error.
pub async fn send_until<S, Send, Wait, Stopped>(
    state: &mut S,
    bytes: &str,
    marker: &str,
    timing: RetryTiming,
    mut send: Send,
    mut wait: Wait,
    stopped: Stopped,
) -> Result<(), String>
where
    Send: FnMut(&mut S, &str) -> Result<(), String>,
    Wait: for<'a> FnMut(&'a mut S, &'a str, Duration) -> WaitFuture<'a>,
    Stopped: Fn(&S) -> bool,
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
        if stopped(state) {
            return Err(last_error);
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
    use super::{RetryTiming, send_until};
    use crate::reader_failure::ReaderFailure;
    use std::time::Duration;

    #[derive(Default)]
    struct TestState {
        failure: ReaderFailure,
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
            |state| state.failure.has_failed() || state.reader_finished,
        )
        .await
        .expect_err("reader panic must stop the retry loop");

        assert_eq!(error, PANIC_ERROR);
        assert_eq!(state.sends, 1, "terminal failure must not resend input");
        assert_eq!(state.waits, 1, "terminal failure must not wait again");
    }
}
