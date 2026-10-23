// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Monotonic correction for cucumber event timestamps.
//!
//! Cucumber stamps every event with `SystemTime::now()` (`event::Event::new`)
//! and its writers then subtract those stamps, treating the wall clock as if it
//! were monotonic. It is not. Both writers this suite uses turn a negative
//! difference into a panic rather than a degraded duration:
//!
//! | writer | site | granularity |
//! |---|---|---|
//! | `writer::Json`  | `json.rs:285`  | per step |
//! | `writer::JUnit` | `junit.rs:432` | per scenario |
//!
//! A panic raised by a *writer* is invisible: cucumber swaps the panic hook for
//! an empty one while a run is in flight (`runner/basic.rs:929`, restored at
//! `:1070`) and a writer panic unwinds past that restore, so the process dies
//! with only an exit code and leaves 0-byte reports behind.
//!
//! That is what kills the Strix Halo WSL2 lane: the guest's wall clock steps
//! backward (34 s observed) because a loaded WSL2 guest drifts and Hyper-V time
//! sync corrects it with a step rather than a slew.
//!
//! [`MonotonicClock`] rebuilds a non-decreasing timeline from those stamps so
//! the writers never see a pair they would panic on. Its logic is kept free of
//! cucumber types so it gets real `#[test]` coverage under `cargo test -p
//! e2e-cucumber --lib` — the cucumber binary's custom harness never runs plain
//! `#[test]` functions placed inside it.

use std::time::{Duration, SystemTime};

use cucumber::{Event, Writer, event, parser, writer};

/// Backward movement at or above this is treated as the clock being reset;
/// anything smaller is treated as delivery jitter.
///
/// Events are created by concurrently running scenarios and delivered over a
/// channel, so two stamps can arrive marginally out of order on a perfectly
/// healthy clock. That jitter is sub-millisecond, while a guest clock
/// resynchronising moves by seconds, so a one-second threshold separates them
/// with room to spare. Panic-safety does not depend on this value — the
/// non-decreasing clamp below covers every case — only whether the run loses
/// one pair's duration or the rest of the run's.
const CLOCK_STEP_THRESHOLD: Duration = Duration::from_secs(1);

/// Rewrites a sequence of wall-clock readings into a non-decreasing one.
#[derive(Clone, Debug, Default)]
pub struct MonotonicClock {
    /// Accumulated compensation for every backward step seen so far.
    offset: Duration,
    /// Previous reading, as supplied.
    last_raw: Option<SystemTime>,
    /// Previous reading, as returned.
    last_corrected: Option<SystemTime>,
}

impl MonotonicClock {
    /// A clock that has seen no readings yet.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            offset: Duration::ZERO,
            last_raw: None,
            last_corrected: None,
        }
    }

    /// Maps one raw reading onto the corrected timeline.
    ///
    /// Readings must be supplied in the order they were taken; the correction
    /// is defined against the previous call, not against real time.
    ///
    /// A backward move of at least [`CLOCK_STEP_THRESHOLD`] is absorbed into a
    /// running offset, which keeps every *later* interval accurate instead of
    /// flattening the remainder of the run. The pair that straddles the step
    /// still collapses to roughly zero, which is unavoidable: the step erased
    /// the only evidence of how much time actually passed.
    pub fn correct(&mut self, at: SystemTime) -> SystemTime {
        if let Some(last_raw) = self.last_raw
            && let Ok(moved_back) = last_raw.duration_since(at)
            && moved_back >= CLOCK_STEP_THRESHOLD
        {
            self.offset += moved_back;
        }
        self.last_raw = Some(at);

        let mut corrected = at + self.offset;
        if let Some(last_corrected) = self.last_corrected
            && corrected < last_corrected
        {
            corrected = last_corrected;
        }
        self.last_corrected = Some(corrected);
        corrected
    }
}

/// Applies [`MonotonicClock`] to every event on its way to an inner writer.
///
/// Belongs *outside* `.normalized()`, so it sees events in arrival order. After
/// normalisation a lower timestamp is ordinary rather than anomalous —
/// `Normalize` re-emits events grouped by scenario, and the suite runs up to 64
/// scenarios at once — so correcting there would fire on healthy runs and
/// destroy the durations it was added to protect.
#[derive(Clone, Debug)]
pub struct MonotonicClockWriter<Wr> {
    writer: Wr,
    clock: MonotonicClock,
}

impl<Wr> MonotonicClockWriter<Wr> {
    /// Wraps `writer`, correcting each event's timestamp before forwarding it.
    pub const fn new(writer: Wr) -> Self {
        Self {
            writer,
            clock: MonotonicClock::new(),
        }
    }
}

impl<World, Wr: Writer<World>> Writer<World> for MonotonicClockWriter<Wr> {
    type Cli = Wr::Cli;

    // `Writer` bounds neither `World` nor `Self::Cli` by `Send`, so no
    // implementation of it can produce a `Send` future.
    #[allow(clippy::future_not_send)]
    async fn handle_event(
        &mut self,
        event: parser::Result<Event<event::Cucumber<World>>>,
        cli: &Self::Cli,
    ) {
        let event = event.map(|mut event| {
            event.at = self.clock.correct(event.at);
            event
        });
        self.writer.handle_event(event, cli).await;
    }
}

#[warn(clippy::missing_trait_methods)]
impl<World, Wr: writer::Stats<World>> writer::Stats<World> for MonotonicClockWriter<Wr> {
    fn passed_steps(&self) -> usize {
        self.writer.passed_steps()
    }

    fn skipped_steps(&self) -> usize {
        self.writer.skipped_steps()
    }

    fn failed_steps(&self) -> usize {
        self.writer.failed_steps()
    }

    fn retried_steps(&self) -> usize {
        self.writer.retried_steps()
    }

    fn parsing_errors(&self) -> usize {
        self.writer.parsing_errors()
    }

    fn hook_errors(&self) -> usize {
        self.writer.hook_errors()
    }

    fn execution_has_failed(&self) -> bool {
        self.writer.execution_has_failed()
    }
}

#[warn(clippy::missing_trait_methods)]
impl<Wr: writer::Normalized> writer::Normalized for MonotonicClockWriter<Wr> {}

#[cfg(test)]
mod tests {
    use super::{CLOCK_STEP_THRESHOLD, MonotonicClock};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    fn at(secs: u64, nanos: u32) -> SystemTime {
        UNIX_EPOCH + Duration::new(secs, nanos)
    }

    /// The exact readings that killed the Strix Halo WSL2 lane, taken from the
    /// panic captured in `harness-diagnostics.log` on run 36393752400: a
    /// scenario whose Finished stamp is 34 s *earlier* than its own Started
    /// stamp. `duration_since` on that pair is the call cucumber turns into a
    /// panic at `json.rs:285` / `junit.rs:432`, so asserting it succeeds is
    /// exactly the regression.
    #[test]
    fn observed_wsl2_backward_step_no_longer_inverts_a_pair() {
        let started_raw = at(1_790_582_185, 178_234_361);
        let ended_raw = at(1_790_582_151, 969_468_442);
        // Precondition: the raw readings really are inverted, so this test
        // fails against an unwrapped clock rather than passing vacuously.
        assert!(ended_raw.duration_since(started_raw).is_err());

        let mut clock = MonotonicClock::new();
        let started = clock.correct(started_raw);
        let ended = clock.correct(ended_raw);

        assert!(
            ended.duration_since(started).is_ok(),
            "corrected pair must not be inverted"
        );
    }

    /// The point of accumulating an offset rather than merely clamping: only
    /// the pair straddling the step loses its duration. Everything measured
    /// afterwards must still be accurate, which a clamp-only correction would
    /// flatten for the remainder of the run.
    #[test]
    fn intervals_after_a_step_keep_their_real_length() {
        let mut clock = MonotonicClock::new();
        clock.correct(at(1_000, 0));
        // Clock resets 34 s backward, then two seconds genuinely elapse.
        let after_step = clock.correct(at(966, 0));
        let two_seconds_later = clock.correct(at(968, 0));

        assert_eq!(
            two_seconds_later.duration_since(after_step).unwrap(),
            Duration::from_secs(2)
        );
    }

    /// Two successive steps must both be absorbed, not just the first.
    #[test]
    fn successive_steps_each_add_to_the_offset() {
        let mut clock = MonotonicClock::new();
        clock.correct(at(1_000, 0));
        let first = clock.correct(at(970, 0));
        let second = clock.correct(at(950, 0));
        let later = clock.correct(at(951, 0));

        assert!(second >= first, "second step must not go backward");
        assert_eq!(
            later.duration_since(second).unwrap(),
            Duration::from_secs(1)
        );
    }

    /// On a healthy clock the correction must be invisible: every reading is
    /// returned unchanged, so ordinary runs report the durations they always
    /// did.
    #[test]
    fn a_healthy_clock_is_passed_through_untouched() {
        let readings = [
            at(500, 0),
            at(500, 1),
            at(500, 250_000_000),
            at(501, 0),
            at(700, 999_999_999),
        ];

        let mut clock = MonotonicClock::new();
        let corrected: Vec<_> = readings.iter().map(|&r| clock.correct(r)).collect();

        assert_eq!(corrected, readings);
    }

    /// Sub-threshold inversion is delivery jitter between concurrent
    /// scenarios, not a clock reset. Clamp it flat — adopting it as an offset
    /// would shift the rest of the run by a spurious amount.
    #[test]
    fn jitter_below_the_threshold_is_clamped_not_offset() {
        let jitter = Duration::from_millis(5);
        assert!(jitter < CLOCK_STEP_THRESHOLD);

        let mut clock = MonotonicClock::new();
        let first = clock.correct(at(100, 0));
        let out_of_order = clock.correct(at(100, 0) - jitter);
        // The clamp holds the line rather than moving it.
        assert_eq!(out_of_order, first);

        // And no offset was absorbed, so later readings are still themselves.
        assert_eq!(clock.correct(at(102, 0)), at(102, 0));
    }

    /// Whatever the readings do, the output must never decrease — that
    /// property alone is what guarantees the writers cannot panic.
    #[test]
    fn output_never_decreases_for_an_erratic_clock() {
        let readings = [
            at(1_000, 0),
            at(1_001, 0),
            at(940, 500_000_000),
            at(940, 400_000_000),
            at(945, 0),
            at(900, 0),
            at(901, 0),
        ];

        let mut clock = MonotonicClock::new();
        let mut previous = None;
        for raw in readings {
            let corrected = clock.correct(raw);
            if let Some(previous) = previous {
                assert!(
                    corrected >= previous,
                    "corrected timeline went backward at {raw:?}"
                );
            }
            previous = Some(corrected);
        }
    }
}
