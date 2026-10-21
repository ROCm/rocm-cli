// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Retry loop for quitting a TUI that may open a quit-confirm prompt first.

use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, Instant};

use crate::send_until::RetryTiming;

/// Key that approves the dashboard's quit-confirm prompt.
pub const QUIT_CONFIRM_KEY: &str = "y";

/// What one bounded wait for the child to exit observed.
#[derive(Debug, PartialEq, Eq)]
pub enum ExitPoll {
    /// The child exited, or the session hit a terminal error; the loop stops
    /// and returns this result.
    Finished(Result<(), String>),
    /// The child is still running.
    Running,
}

/// Borrowing future returned by the exit wait used by [`quit_and_wait`].
pub type ExitFuture<'a> = Pin<Box<dyn Future<Output = ExitPoll> + Send + 'a>>;

/// Why [`quit_and_wait`] gave up.
#[derive(Debug, PartialEq, Eq)]
pub enum QuitError {
    /// The child exited unsuccessfully, or the session failed.
    Failed(String),
    /// The child was still running at the deadline.
    TimedOut,
}

/// Send `gesture` until the child exits, answering the quit-confirm prompt
/// with [`QUIT_CONFIRM_KEY`] whenever `confirming` reports it on screen.
///
/// The gesture is resent because a bare send can be consumed by a transient
/// state and never reach the quit handler. Resending is not idempotent once
/// the prompt can open, though: the prompt reads the gesture as Cancel. If the
/// TUI takes longer than the resend interval to act on a key, every decision is
/// made with an earlier `gesture` still in flight, which flips the prompt after
/// the screen was read — at a fixed cadence that can repeat until the deadline.
/// So the wait doubles after each attempt: once it outlasts the TUI's latency,
/// nothing is in flight when the screen is read, and the key sent matches it.
pub async fn quit_and_wait<S, Confirming, SendInput, Wait>(
    state: &mut S,
    gesture: &str,
    timing: RetryTiming,
    mut confirming: Confirming,
    mut send: SendInput,
    mut wait: Wait,
) -> Result<(), QuitError>
where
    S: Send,
    Confirming: FnMut(&S) -> bool + Send,
    SendInput: FnMut(&mut S, &str) -> Result<(), String> + Send,
    Wait: for<'a> FnMut(&'a mut S, Duration) -> ExitFuture<'a> + Send,
{
    let RetryTiming {
        timeout,
        resend_interval,
    } = timing;
    let deadline = Instant::now() + timeout;
    let mut interval = resend_interval;
    loop {
        let key = if confirming(state) {
            QUIT_CONFIRM_KEY
        } else {
            gesture
        };
        send(state, key).map_err(QuitError::Failed)?;
        let remaining = deadline.saturating_duration_since(Instant::now());
        match wait(state, interval.min(remaining)).await {
            ExitPoll::Finished(result) => return result.map_err(QuitError::Failed),
            ExitPoll::Running => {}
        }
        if Instant::now() >= deadline {
            return Err(QuitError::TimedOut);
        }
        interval = interval.saturating_mul(2);
    }
}

#[cfg(test)]
mod tests {
    use super::{ExitPoll, QUIT_CONFIRM_KEY, QuitError, quit_and_wait};
    use crate::send_until::RetryTiming;
    use std::time::{Duration, Instant};

    /// A dashboard whose prompt and exit only reflect a key `latency` after it
    /// was sent, as on a loaded host: `q` toggles the prompt (it opens it, and
    /// the open prompt reads `q` as Cancel), `y` on the open prompt quits, and
    /// `y` with no prompt is ignored.
    struct SlowDashboard {
        latency: Duration,
        sent: Vec<(Instant, String)>,
    }

    impl SlowDashboard {
        fn new(latency: Duration) -> Self {
            Self {
                latency,
                sent: Vec::new(),
            }
        }

        /// `(prompt open, exited)` after every key whose latency has elapsed.
        fn observed(&self) -> (bool, bool) {
            let now = Instant::now();
            let mut prompt = false;
            for (at, key) in &self.sent {
                if *at + self.latency > now {
                    break;
                }
                match key.as_str() {
                    "q" => prompt = !prompt,
                    QUIT_CONFIRM_KEY if prompt => return (true, true),
                    _ => {}
                }
            }
            (prompt, false)
        }
    }

    async fn quit(dashboard: &mut SlowDashboard, timing: RetryTiming) -> Result<(), QuitError> {
        quit_and_wait(
            dashboard,
            "q",
            timing,
            |d| d.observed().0,
            |d, key| {
                d.sent.push((Instant::now(), key.to_string()));
                Ok(())
            },
            |d, budget| {
                Box::pin(async move {
                    let deadline = Instant::now() + budget;
                    loop {
                        if d.observed().1 {
                            return ExitPoll::Finished(Ok(()));
                        }
                        if Instant::now() >= deadline {
                            return ExitPoll::Running;
                        }
                        tokio::time::sleep(Duration::from_millis(2)).await;
                    }
                })
            },
        )
        .await
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_prompt_that_renders_within_the_resend_interval_is_confirmed() {
        let mut dashboard = SlowDashboard::new(Duration::from_millis(10));
        let timing = RetryTiming {
            timeout: Duration::from_secs(3),
            resend_interval: Duration::from_millis(50),
        };

        quit(&mut dashboard, timing)
            .await
            .expect("a responsive prompt must be confirmed");

        let keys: Vec<&str> = dashboard.sent.iter().map(|(_, k)| k.as_str()).collect();
        assert_eq!(keys, ["q", QUIT_CONFIRM_KEY]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_prompt_slower_than_the_resend_interval_is_still_confirmed() {
        // Each key takes between one and two resend intervals to show, so a
        // fixed cadence always has exactly one `q` in flight when it decides:
        // that `q` flips the prompt after the decision, the decided key lands on
        // the other state, and the two stay out of phase until the deadline.
        let mut dashboard = SlowDashboard::new(Duration::from_millis(80));
        let timing = RetryTiming {
            timeout: Duration::from_secs(3),
            resend_interval: Duration::from_millis(50),
        };

        quit(&mut dashboard, timing).await.unwrap_or_else(|e| {
            let keys: Vec<&str> = dashboard.sent.iter().map(|(_, k)| k.as_str()).collect();
            panic!("slow prompt must still be confirmed, got {e:?} after {keys:?}")
        });
    }

    #[tokio::test(flavor = "current_thread")]
    async fn an_unsuccessful_exit_is_returned_without_resending() {
        let mut sends = 0;
        let error = quit_and_wait(
            &mut sends,
            "q",
            RetryTiming {
                timeout: Duration::from_secs(1),
                resend_interval: Duration::from_millis(10),
            },
            |_| false,
            |sends, _| {
                *sends += 1;
                Ok(())
            },
            |_, _| Box::pin(async { ExitPoll::Finished(Err("exit code 1".to_string())) }),
        )
        .await
        .expect_err("an unsuccessful exit must fail");

        assert_eq!(error, QuitError::Failed("exit code 1".to_string()));
        assert_eq!(sends, 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_failed_send_is_returned_without_waiting() {
        let mut waits = 0;
        let error = quit_and_wait(
            &mut waits,
            "q",
            RetryTiming {
                timeout: Duration::from_secs(1),
                resend_interval: Duration::from_millis(10),
            },
            |_| false,
            |_, _| Err("pty closed".to_string()),
            |waits, _| {
                *waits += 1;
                Box::pin(async { ExitPoll::Running })
            },
        )
        .await
        .expect_err("a failed send must fail");

        assert_eq!(error, QuitError::Failed("pty closed".to_string()));
        assert_eq!(
            waits, 0,
            "nothing was sent, so there is nothing to wait for"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_wait_longer_than_the_time_left_is_cut_to_the_deadline() {
        // Without the clamp a wait runs its whole interval past the deadline,
        // and with doubling that can be as long as the timeout itself.
        let timeout = Duration::from_millis(100);
        let started = Instant::now();
        let error = quit_and_wait(
            &mut (),
            "q",
            RetryTiming {
                timeout,
                resend_interval: Duration::from_secs(5),
            },
            |()| false,
            |(), _| Ok(()),
            |(), budget| {
                Box::pin(async move {
                    tokio::time::sleep(budget).await;
                    ExitPoll::Running
                })
            },
        )
        .await
        .expect_err("a child that never exits must time out");

        assert_eq!(error, QuitError::TimedOut);
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "the wait must stop at the {timeout:?} deadline, took {:?}",
            started.elapsed()
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_child_that_never_exits_times_out() {
        let error = quit_and_wait(
            &mut (),
            "q",
            RetryTiming {
                timeout: Duration::from_millis(60),
                resend_interval: Duration::from_millis(10),
            },
            |()| false,
            |(), _| Ok(()),
            |(), budget| {
                Box::pin(async move {
                    tokio::time::sleep(budget).await;
                    ExitPoll::Running
                })
            },
        )
        .await
        .expect_err("a child that never exits must time out");

        assert_eq!(error, QuitError::TimedOut);
    }
}
