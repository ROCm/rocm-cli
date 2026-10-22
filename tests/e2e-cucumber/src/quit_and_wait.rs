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
    /// The child was still running at the deadline. `stalled` is how long the
    /// loop itself went unpolled while it waited, which was added to the
    /// deadline (see [`quit_and_wait`]).
    TimedOut { stalled: Duration },
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
///
/// The timeout counts only the time this loop was actually being polled. The
/// E2E harness runs its concurrent scenarios as futures that cucumber polls
/// from one task, and many steps run `rocm` synchronously, so while one of
/// them runs, no other scenario is polled at all. A wait that returns long
/// after its budget was not waiting on the TUI for that extra time: the TUI
/// had already opened the prompt, but this loop was not running to read it
/// and answer it. If that time counted, a stall longer than the timeout would
/// end the quit before the answer was ever sent, even though the TUI was ready
/// to exit. So any time a wait runs past its budget is added to the deadline.
/// A TUI that never exits still times out, after `timeout` of polled time.
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
    let mut deadline = Instant::now() + timeout;
    let mut stalled = Duration::ZERO;
    let mut interval = resend_interval;
    loop {
        let key = if confirming(state) {
            QUIT_CONFIRM_KEY
        } else {
            gesture
        };
        send(state, key).map_err(QuitError::Failed)?;
        let started = Instant::now();
        let budget = interval.min(deadline.saturating_duration_since(started));
        match wait(state, budget).await {
            ExitPoll::Finished(result) => return result.map_err(QuitError::Failed),
            ExitPoll::Running => {}
        }
        let overrun = started.elapsed().saturating_sub(budget);
        stalled += overrun;
        deadline += overrun;
        if Instant::now() >= deadline {
            return Err(QuitError::TimedOut { stalled });
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

        assert!(matches!(error, QuitError::TimedOut { .. }), "{error:?}");
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

        assert!(matches!(error, QuitError::TimedOut { .. }), "{error:?}");
    }

    /// A wait that returns `stall` late the first time it is called, as when
    /// another scenario's synchronous step holds cucumber's task. Every later
    /// wait runs on time.
    fn stalls_once(stall: Duration) -> impl FnMut(Duration) -> Duration {
        let mut first = true;
        move |budget| {
            if std::mem::take(&mut first) {
                budget + stall
            } else {
                budget
            }
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_stall_longer_than_the_timeout_still_answers_the_prompt() {
        // The prompt is on screen within 10 ms of `q`, but the first wait comes
        // back only after the whole timeout has passed. Counting that stall
        // gives up without ever sending `y` to a TUI that was ready to quit.
        let mut dashboard = SlowDashboard::new(Duration::from_millis(10));
        let timing = RetryTiming {
            timeout: Duration::from_millis(300),
            resend_interval: Duration::from_millis(50),
        };
        let mut stall = stalls_once(Duration::from_millis(400));

        let result = quit_and_wait(
            &mut dashboard,
            "q",
            timing,
            |d| d.observed().0,
            |d, key| {
                d.sent.push((Instant::now(), key.to_string()));
                Ok(())
            },
            |d, budget| {
                let held = stall(budget);
                Box::pin(async move {
                    let deadline = Instant::now() + held;
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
        .await;

        let keys: Vec<&str> = dashboard.sent.iter().map(|(_, k)| k.as_str()).collect();
        assert_eq!(result, Ok(()), "keys sent: {keys:?}");
        assert_eq!(keys, ["q", QUIT_CONFIRM_KEY]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_stalled_child_that_never_exits_still_times_out() {
        let stall = Duration::from_millis(200);
        let mut late = stalls_once(stall);
        let started = Instant::now();
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
                let held = late(budget);
                Box::pin(async move {
                    tokio::time::sleep(held).await;
                    ExitPoll::Running
                })
            },
        )
        .await
        .expect_err("a child that never exits must time out");

        let QuitError::TimedOut { stalled } = error else {
            panic!("expected a timeout, got {error:?}");
        };
        assert!(stalled >= stall, "reported stall {stalled:?} < {stall:?}");
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "the stall must extend the deadline once, not indefinitely: took {:?}",
            started.elapsed()
        );
    }
}
