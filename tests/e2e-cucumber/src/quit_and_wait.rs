// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Retry loop for quitting a TUI that may open a quit-confirm prompt first.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use crate::polled_deadline::{Expiry, PolledDeadline, RetryTiming};

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
    /// The child was still running when the deadline expired.
    TimedOut(Expiry),
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
/// The deadline is a [`PolledDeadline`]: time the harness leaves a wait
/// unpolled is added to it, up to a wall-clock cap. This loop has to act on
/// what it reads, so a stall counted against the timeout would end the quit
/// before the prompt the TUI had already opened was ever answered. Each wait
/// must re-check at least every `timing.poll_interval`.
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
    let mut deadline = PolledDeadline::start(timing);
    let mut interval = timing.resend_interval;
    loop {
        let key = if confirming(state) {
            QUIT_CONFIRM_KEY
        } else {
            gesture
        };
        send(state, key).map_err(QuitError::Failed)?;
        let budget = deadline.budget(interval);
        match deadline.run(wait(state, budget)).await {
            ExitPoll::Finished(result) => return result.map_err(QuitError::Failed),
            ExitPoll::Running => {}
        }
        if let Some(expiry) = deadline.expired() {
            return Err(QuitError::TimedOut(expiry));
        }
        interval = interval.saturating_mul(2);
    }
}

#[cfg(test)]
mod tests {
    use super::{ExitPoll, QUIT_CONFIRM_KEY, QuitError, quit_and_wait};
    use crate::polled_deadline::test_support::{TICK, credited, stalls_once, ticking_wait};
    use crate::polled_deadline::{Expiry, RetryTiming};
    use std::time::Duration;
    use tokio::time::Instant;

    /// Outer bound on every test here, so a deadline that keeps moving fails
    /// the test instead of hanging it. The tests run on tokio's paused clock,
    /// so reaching it takes no real time.
    const HANG: Duration = Duration::from_secs(5);

    fn timing(timeout_ms: u64, resend_ms: u64) -> RetryTiming {
        RetryTiming {
            timeout: Duration::from_millis(timeout_ms),
            resend_interval: Duration::from_millis(resend_ms),
            poll_interval: TICK,
        }
    }

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

        fn keys(&self) -> Vec<&str> {
            self.sent.iter().map(|(_, k)| k.as_str()).collect()
        }
    }

    /// Quit `dashboard`, stalling each wait by whatever `stall` returns.
    async fn quit(
        dashboard: &mut SlowDashboard,
        timing: RetryTiming,
        mut stall: impl FnMut() -> Duration + Send,
    ) -> Result<(), QuitError> {
        let run = quit_and_wait(
            dashboard,
            "q",
            timing,
            |d| d.observed().0,
            |d, key| {
                d.sent.push((Instant::now(), key.to_string()));
                Ok(())
            },
            |d, budget| {
                let stall = stall();
                Box::pin(async move {
                    if ticking_wait(budget, stall, || d.observed().1).await {
                        ExitPoll::Finished(Ok(()))
                    } else {
                        ExitPoll::Running
                    }
                })
            },
        );
        tokio::time::timeout(HANG, run)
            .await
            .expect("quit_and_wait must end on its own")
    }

    /// Run `quit_and_wait` against a child that never exits, stalling the
    /// first wait by `stall`; returns the error and the wall time it took.
    async fn never_exits(timing: RetryTiming, stall: Duration) -> (QuitError, Duration) {
        let mut stall = stalls_once(stall);
        let mut child = ();
        let started = Instant::now();
        let run = quit_and_wait(
            &mut child,
            "q",
            timing,
            |()| false,
            |(), _| Ok(()),
            |(), budget| {
                let stall = stall();
                Box::pin(async move {
                    ticking_wait(budget, stall, || false).await;
                    ExitPoll::Running
                })
            },
        );
        let error = tokio::time::timeout(HANG, run)
            .await
            .expect("quit_and_wait must end on its own")
            .expect_err("a child that never exits must time out");
        (error, started.elapsed())
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_prompt_that_renders_within_the_resend_interval_is_confirmed() {
        let mut dashboard = SlowDashboard::new(Duration::from_millis(10));

        quit(&mut dashboard, timing(3000, 50), || Duration::ZERO)
            .await
            .expect("a responsive prompt must be confirmed");

        assert_eq!(dashboard.keys(), ["q", QUIT_CONFIRM_KEY]);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_prompt_slower_than_the_resend_interval_is_still_confirmed() {
        // Each key takes between one and two resend intervals to show, so a
        // fixed cadence always has exactly one `q` in flight when it decides:
        // that `q` flips the prompt after the decision, the decided key lands on
        // the other state, and the two stay out of phase until the deadline.
        let mut dashboard = SlowDashboard::new(Duration::from_millis(80));

        let result = quit(&mut dashboard, timing(3000, 50), || Duration::ZERO).await;

        assert_eq!(result, Ok(()), "keys sent: {:?}", dashboard.keys());
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn an_unsuccessful_exit_is_returned_without_resending() {
        let mut sends = 0;
        let error = quit_and_wait(
            &mut sends,
            "q",
            timing(1000, 10),
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

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_failed_send_is_returned_without_waiting() {
        let mut waits = 0;
        let error = quit_and_wait(
            &mut waits,
            "q",
            timing(1000, 10),
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

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_wait_longer_than_the_time_left_is_cut_to_the_deadline() {
        // Without the clamp a wait runs its whole interval past the deadline,
        // and with doubling that can be as long as the timeout itself.
        let (error, elapsed) = never_exits(timing(100, 5000), Duration::ZERO).await;

        assert!(matches!(error, QuitError::TimedOut(_)), "{error:?}");
        assert!(
            elapsed <= Duration::from_millis(100) + TICK,
            "the wait must stop at the 100ms deadline, took {elapsed:?}"
        );
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_wait_is_cut_to_the_wall_clock_cap_when_stalls_push_the_deadline_past_it() {
        // A 380 ms stall in the first wait moves the 100 ms deadline to 425 ms,
        // past the 400 ms cap. The doubled 10 s interval must then be cut to
        // the cap, not to the deadline, so the loop ends at 400 ms on the cap.
        let (error, elapsed) = never_exits(timing(100, 5000), Duration::from_millis(380)).await;

        assert!(
            matches!(error, QuitError::TimedOut(Expiry::WallClock { .. })),
            "{error:?}"
        );
        assert_eq!(elapsed, Duration::from_millis(400));
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_child_that_never_exits_times_out() {
        let (error, _) = never_exits(timing(60, 10), Duration::ZERO).await;

        assert!(
            matches!(error, QuitError::TimedOut(Expiry::Polled { .. })),
            "{error:?}"
        );
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_stall_longer_than_the_timeout_still_answers_the_prompt() {
        // The prompt is on screen within 10 ms of `q`, but the first wait comes
        // back only after the whole timeout has passed. Counting that stall
        // gives up without ever sending `y` to a TUI that was ready to quit.
        let mut dashboard = SlowDashboard::new(Duration::from_millis(10));

        let result = quit(
            &mut dashboard,
            timing(300, 50),
            stalls_once(Duration::from_millis(400)),
        )
        .await;

        assert_eq!(result, Ok(()), "keys sent: {:?}", dashboard.keys());
        assert_eq!(dashboard.keys(), ["q", QUIT_CONFIRM_KEY]);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_stall_inside_a_long_budget_still_answers_the_prompt() {
        // The first wait's budget is the whole 300 ms timeout, and a 250 ms
        // stall lands inside it: the wait comes back on time, never past its
        // budget, with the prompt open since 10 ms after `q`. The stall must
        // still be credited, or the timeout has passed and the loop gives up
        // without answering.
        let mut dashboard = SlowDashboard::new(Duration::from_millis(10));

        let result = quit(
            &mut dashboard,
            timing(300, 400),
            stalls_once(Duration::from_millis(250)),
        )
        .await;

        assert_eq!(result, Ok(()), "keys sent: {:?}", dashboard.keys());
        assert_eq!(dashboard.keys(), ["q", QUIT_CONFIRM_KEY]);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_stalled_child_that_never_exits_moves_the_deadline_by_the_stall_once() {
        // The stall moves the deadline by its credited part, once, so the loop
        // ends within one tick after `timeout + credited(stall)`. Missing the
        // credit ends it near `timeout`; crediting the stall twice, or counting
        // every poll as a stall, ends it later. All fall outside the window.
        let timeout = Duration::from_millis(200);
        let stall = Duration::from_millis(150);

        let (error, elapsed) = never_exits(timing(200, 10), stall).await;

        let QuitError::TimedOut(Expiry::Polled { stalled, .. }) = error else {
            panic!("expected a polled-time timeout, got {error:?}");
        };
        assert_eq!(
            stalled,
            credited(stall),
            "reported stall for a {stall:?} stall"
        );
        let moved = timeout + credited(stall);
        assert!(
            elapsed >= moved,
            "the stall must move the deadline to {moved:?}: ended after {elapsed:?}"
        );
        assert!(
            elapsed <= moved + TICK,
            "the stall must move the deadline once, to {moved:?}: ended after {elapsed:?}"
        );
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_stall_past_the_wall_clock_cap_fails_the_step() {
        // 4x a 100 ms timeout is a 400 ms cap. A 500 ms stall overruns it, and
        // moves the deadline past it, so only the cap can end the loop.
        let stall = Duration::from_millis(500);
        let (error, elapsed) = never_exits(timing(100, 10), stall).await;

        assert!(
            matches!(
                error,
                QuitError::TimedOut(Expiry::WallClock { cap, .. }) if cap == Duration::from_millis(400)
            ),
            "{error:?}"
        );
        assert!(elapsed <= stall + TICK, "took {elapsed:?}");
    }
}
