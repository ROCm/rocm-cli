// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Deadline for a retry loop that counts only the time the loop was polled.
//!
//! The E2E harness runs its concurrent scenarios as futures that cucumber
//! polls from one task, and many steps run `rocm` synchronously. While one of
//! those runs, no other scenario is polled at all. A retry loop that must act
//! on what it reads (answer a prompt, resend a lost key) and measures its
//! timeout in wall-clock time can then run out of time while it is not
//! running, and give up without acting on a screen that was ready long ago.
//!
//! [`PolledDeadline`] measures the gaps between polls of each wait instead. A
//! wait is expected to be polled at least every `poll_interval` (the cadence
//! of its own re-check loop), and its timer may fire up to [`TIMER_SLACK`]
//! late; whatever a gap runs past both is a stall, and is added to the
//! deadline. So the timeout is the time the loop's waits were actually
//! polled, whether a stall lands inside a wait's budget or runs past it. A
//! wall-clock cap of [`WALL_CLOCK_CAP_FACTOR`] times the timeout bounds how far
//! stalls can push the deadline, so a pathological stall still fails the step
//! rather than the whole job.
//!
//! Time is read from [`tokio::time::Instant`]: the host clock in the harness,
//! and a paused, virtual one in the unit tests below, so their timing
//! assertions do not depend on how precisely the host's timers fire.

use std::fmt;
use std::future::{Future, poll_fn};
use std::pin::Pin;
use std::time::Duration;

use tokio::time::Instant;

/// How many times the timeout a loop may run in wall-clock time, however much
/// of it was stalled.
pub const WALL_CLOCK_CAP_FACTOR: u32 = 4;

/// How late a wait's timer may fire and still count as polling, not a stall.
///
/// A sleep on Windows ends on a tick of its default 15.6 ms timer, and a
/// loaded host can miss that tick by another one or two; this is about three
/// ticks. Real harness stalls (another scenario running `rocm` synchronously)
/// last hundreds of milliseconds or more, so they still register.
pub const TIMER_SLACK: Duration = Duration::from_millis(50);

/// Deadline and per-attempt wait shared by the retry loops.
#[derive(Clone, Copy, Debug)]
pub struct RetryTiming {
    /// Polled time the loop may spend before it gives up.
    pub timeout: Duration,
    /// Wait for one attempt to take effect before acting again.
    pub resend_interval: Duration,
    /// Cadence at which a wait re-checks its condition. A gap between two
    /// polls of the wait that runs past this plus [`TIMER_SLACK`] is a stall.
    pub poll_interval: Duration,
}

/// Why a [`PolledDeadline`] expired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expiry {
    /// The loop was polled for its whole timeout. `stalled` is the time it
    /// went unpolled on top of that, which did not count.
    Polled {
        timeout: Duration,
        stalled: Duration,
    },
    /// The wall-clock cap ran out first, because the loop went unpolled for
    /// `stalled` of it.
    WallClock {
        timeout: Duration,
        cap: Duration,
        stalled: Duration,
    },
}

impl fmt::Display for Expiry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Polled { timeout, stalled } if stalled.is_zero() => write!(f, "{timeout:?}"),
            Self::Polled { timeout, stalled } => write!(
                f,
                "{timeout:?} of polling, not counting {stalled:?} the harness left this step unpolled"
            ),
            Self::WallClock {
                timeout,
                cap,
                stalled,
            } => write!(
                f,
                "the {cap:?} wall-clock cap ({WALL_CLOCK_CAP_FACTOR}x the {timeout:?} timeout); \
                 the harness left this step unpolled for {stalled:?} of it"
            ),
        }
    }
}

/// A deadline that moves out by every stall measured while a wait runs.
#[derive(Debug)]
pub struct PolledDeadline {
    timeout: Duration,
    poll_interval: Duration,
    deadline: Instant,
    cap: Duration,
    wall_deadline: Instant,
    stalled: Duration,
}

impl PolledDeadline {
    /// Start the clock now.
    #[must_use]
    pub fn start(timing: RetryTiming) -> Self {
        let now = Instant::now();
        let cap = timing.timeout.saturating_mul(WALL_CLOCK_CAP_FACTOR);
        Self {
            timeout: timing.timeout,
            poll_interval: timing.poll_interval,
            deadline: now + timing.timeout,
            cap,
            wall_deadline: now + cap,
            stalled: Duration::ZERO,
        }
    }

    /// `interval` cut to the time left before either deadline, so a wait never
    /// runs past the point where the loop would give up.
    #[must_use]
    pub fn budget(&self, interval: Duration) -> Duration {
        let now = Instant::now();
        interval
            .min(self.deadline.saturating_duration_since(now))
            .min(self.wall_deadline.saturating_duration_since(now))
    }

    /// Await `wait`, adding every stall between its polls to the deadline.
    pub async fn run<F: Future>(&mut self, wait: F) -> F::Output {
        let mut wait = Box::pin(wait);
        let mut last = Instant::now();
        poll_fn(|cx| {
            let stall = stall_in(last.elapsed(), self.poll_interval);
            self.stalled += stall;
            self.deadline += stall;
            let output = Pin::as_mut(&mut wait).poll(cx);
            last = Instant::now();
            output
        })
        .await
    }

    /// Whether the loop must give up now, and why.
    #[must_use]
    pub fn expired(&self) -> Option<Expiry> {
        let now = Instant::now();
        if now >= self.deadline {
            Some(Expiry::Polled {
                timeout: self.timeout,
                stalled: self.stalled,
            })
        } else if now >= self.wall_deadline {
            Some(Expiry::WallClock {
                timeout: self.timeout,
                cap: self.cap,
                stalled: self.stalled,
            })
        } else {
            None
        }
    }
}

/// How much of a `gap` between two polls of a wait was a stall: whatever it
/// ran past the `poll_interval` the wait meant to sleep plus the
/// [`TIMER_SLACK`] its timer may fire late by. Neither is the harness stalling,
/// so a wait whose timer is coarse (as on Windows) is never reported as one.
fn stall_in(gap: Duration, poll_interval: Duration) -> Duration {
    gap.saturating_sub(poll_interval + TIMER_SLACK)
}

/// Test doubles shared by the retry loops' unit tests. They run on tokio's
/// paused clock (`start_paused = true`), so a sleep advances time by exactly
/// what it asks for and the tests' timing assertions are exact on every host.
#[cfg(test)]
pub(crate) mod test_support {
    use std::time::Duration;
    use tokio::time::Instant;

    /// Poll cadence of [`ticking_wait`], and the `poll_interval` the tests
    /// give the loops.
    pub const TICK: Duration = Duration::from_millis(5);

    /// What a `stall`, as [`ticking_wait`] takes it, adds to the deadline:
    /// all of it except the tick and timer slack of ordinary polling.
    pub fn credited(stall: Duration) -> Duration {
        stall.saturating_sub(TICK + super::TIMER_SLACK)
    }

    /// A wait that re-checks `done` every [`TICK`] for up to `budget`, after
    /// first going unpolled for `stall` (as when another scenario's
    /// synchronous step holds cucumber's task). Returns whether `done` held.
    ///
    /// A wait that gives up without having slept at all sleeps one [`TICK`]
    /// first. Then a loop that keeps handing out empty budgets, because its
    /// deadline is broken, still advances the paused clock, and the test's
    /// outer `tokio::time::timeout` fails it instead of spinning forever.
    pub async fn ticking_wait(
        budget: Duration,
        stall: Duration,
        mut done: impl FnMut() -> bool,
    ) -> bool {
        let deadline = Instant::now() + budget;
        let mut slept = !stall.is_zero();
        if slept {
            tokio::time::sleep(stall).await;
        }
        loop {
            if done() {
                return true;
            }
            if Instant::now() >= deadline {
                if !slept {
                    tokio::time::sleep(TICK).await;
                }
                return false;
            }
            tokio::time::sleep(TICK).await;
            slept = true;
        }
    }

    /// Stall the first wait by `stall`, and no later one.
    pub fn stalls_once(stall: Duration) -> impl FnMut() -> Duration {
        let mut first = true;
        move || {
            if std::mem::take(&mut first) {
                stall
            } else {
                Duration::ZERO
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{TICK, credited, ticking_wait};
    use super::{
        Expiry, PolledDeadline, RetryTiming, TIMER_SLACK, WALL_CLOCK_CAP_FACTOR, stall_in,
    };
    use std::time::Duration;
    use tokio::time::Instant;

    fn timing(timeout_ms: u64) -> RetryTiming {
        RetryTiming {
            timeout: Duration::from_millis(timeout_ms),
            resend_interval: Duration::from_millis(10),
            poll_interval: TICK,
        }
    }

    #[test]
    fn polling_slack_is_not_a_stall() {
        // The harness polls every 20 ms. A wait sleeps that long, and its timer
        // can fire late: on Windows a 20 ms sleep ends on a 15.6 ms tick, after
        // two or three of them (31 or 47 ms), and `wait_for_exit_code_within`
        // notices its budget only after the sleep that runs one tick past it.
        // All of that is polling, not the harness stalling.
        let tick = Duration::from_millis(20);
        for gap in [0, 20, 23, 31, 40, 47, 70] {
            assert_eq!(
                stall_in(Duration::from_millis(gap), tick),
                Duration::ZERO,
                "a {gap} ms gap"
            );
        }
        assert_eq!(
            stall_in(Duration::from_millis(71), tick),
            Duration::from_millis(1)
        );
        assert_eq!(
            stall_in(Duration::from_secs(102), tick),
            Duration::from_secs(102).saturating_sub(tick + TIMER_SLACK)
        );
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_stall_inside_the_budget_moves_the_deadline() {
        // The stall ends well inside the wait's 200 ms budget, so the wait itself
        // comes back on time; the stall must still be credited, once.
        let mut deadline = PolledDeadline::start(timing(10_000));
        let before = deadline.deadline;
        let stall = Duration::from_millis(150);
        deadline
            .run(ticking_wait(Duration::from_millis(200), stall, || false))
            .await;
        assert_eq!(deadline.stalled, credited(stall));
        assert_eq!(deadline.deadline - before, credited(stall));
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_coarse_timer_is_not_a_stall() {
        // A wait polling every 20 ms on a host whose timer only fires on a
        // 15.625 ms tick, and late by a tick: every sleep lasts three ticks.
        let poll_interval = Duration::from_millis(20);
        let coarse = Duration::from_micros(3 * 15_625);
        let mut deadline = PolledDeadline::start(RetryTiming {
            timeout: Duration::from_secs(10),
            resend_interval: Duration::from_secs(1),
            poll_interval,
        });
        let before = deadline.deadline;
        let started = Instant::now();
        deadline
            .run(async {
                while started.elapsed() < Duration::from_secs(1) {
                    tokio::time::sleep(coarse).await;
                }
            })
            .await;
        assert_eq!(deadline.stalled, Duration::ZERO);
        assert_eq!(deadline.deadline, before);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn the_wall_clock_cap_ends_a_loop_that_keeps_stalling() {
        // Every wait stalls for 1 s against a 100 ms timeout, so each one moves
        // the deadline past the time it took, and only the 400 ms cap ends it.
        // The outer timeout fails the test, instead of hanging it, if the cap
        // never fires.
        let mut deadline = PolledDeadline::start(timing(100));
        let cap = Duration::from_millis(100) * WALL_CLOCK_CAP_FACTOR;
        let run = async {
            while deadline.expired().is_none() {
                let budget = deadline.budget(Duration::from_millis(10));
                deadline
                    .run(ticking_wait(budget, Duration::from_secs(1), || false))
                    .await;
            }
        };
        tokio::time::timeout(Duration::from_secs(5), run)
            .await
            .expect("the wall-clock cap must end the loop");
        let Some(Expiry::WallClock { cap: reported, .. }) = deadline.expired() else {
            panic!("expected the wall-clock cap, got {:?}", deadline.expired());
        };
        assert_eq!(reported, cap);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn a_budget_is_cut_to_the_wall_clock_cap_once_stalls_pass_it() {
        // A 380 ms stall against a 100 ms timeout credits 325 ms, so the
        // deadline is 425 ms, past the 400 ms cap. A wait that asks for 10 s
        // must get only the 20 ms left before the cap, and end there; cut to
        // the deadline alone, it would get 45 ms and end at 425 ms.
        let mut deadline = PolledDeadline::start(timing(100));
        let started = Instant::now();
        let cap = Duration::from_millis(100) * WALL_CLOCK_CAP_FACTOR;
        let stall = Duration::from_millis(380);
        deadline
            .run(ticking_wait(Duration::ZERO, stall, || false))
            .await;
        assert!(
            deadline.deadline > started + cap,
            "the stall must push the deadline past the cap"
        );
        assert_eq!(deadline.expired(), None);

        let budget = deadline.budget(Duration::from_secs(10));
        assert_eq!(
            budget,
            Duration::from_millis(20),
            "the 20 ms left before the cap"
        );
        deadline
            .run(ticking_wait(budget, Duration::ZERO, || false))
            .await;

        assert_eq!(started.elapsed(), cap);
        assert!(
            matches!(deadline.expired(), Some(Expiry::WallClock { .. })),
            "{:?}",
            deadline.expired()
        );
    }

    #[test]
    fn the_report_names_only_time_the_step_went_unpolled() {
        let timeout = Duration::from_secs(30);
        assert_eq!(
            Expiry::Polled {
                timeout,
                stalled: Duration::ZERO
            }
            .to_string(),
            "30s"
        );
        assert_eq!(
            Expiry::Polled {
                timeout,
                stalled: Duration::from_secs(5)
            }
            .to_string(),
            "30s of polling, not counting 5s the harness left this step unpolled"
        );
        assert_eq!(
            Expiry::WallClock {
                timeout,
                cap: Duration::from_mins(2),
                stalled: Duration::from_secs(100)
            }
            .to_string(),
            "the 120s wall-clock cap (4x the 30s timeout); the harness left this step unpolled for 100s of it"
        );
    }
}
