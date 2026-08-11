// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Pure observation-policy helpers for generation-throughput sampling.
//!
//! All timing is caller-injected; no wall-clock access inside this module.
//! See the EAI-7960 contract for timing constants and freshness semantics.

// ── Imports (used by the impl below) ────────────────────────────────────────
use std::collections::VecDeque;
use std::time::Duration;

use chrono::{DateTime, Utc};

use crate::metrics::{ObservationFreshness, ObservationMetadata};

// ── Timing constants (from the frozen contract) ──────────────────────────────

const MIN_RATE_WINDOW: Duration = Duration::from_secs(10);
const MAX_RATE_WINDOW: Duration = Duration::from_secs(30);
const MIN_VALIDITY: Duration = Duration::from_secs(6);
const MAX_VALIDITY: Duration = Duration::from_secs(30);

/// Minimum number of counter samples retained in the rolling window.
const MIN_CAPACITY: usize = 3;

/// Maximum number of counter samples retained in the rolling window.
///
/// For supported real cadences (instance_tick ≥ 1 s) the derived capacity is
/// at most `ceil(30 / 1) + 2 = 32`, well inside this ceiling — `MAX_CAPACITY`
/// is non-binding for all production deployments.
///
/// It only binds for sub-161 ms ticks. At such rates the buffer holds 64
/// samples spanning less than `rate_window`, which shortens the effective
/// averaging span but retains a valid bounded rate and preserves the
/// memory-safety guarantee.
const MAX_CAPACITY: usize = 64;

// ── Public helpers ───────────────────────────────────────────────────────────

/// Rate-computation window: `clamp(5 × instance_tick, 10 s, 30 s)`.
///
/// Zero or extremely large ticks are handled without panic or overflow via
/// [`Duration::saturating_mul`].
pub fn rate_window(instance_tick: Duration) -> Duration {
    instance_tick
        .saturating_mul(5)
        .clamp(MIN_RATE_WINDOW, MAX_RATE_WINDOW)
}

/// Validity window for a held observation: `clamp(3 × instance_tick, 6 s, 30 s)`.
///
/// Zero or extremely large ticks are handled without panic or overflow via
/// [`Duration::saturating_mul`].
pub fn observation_validity(instance_tick: Duration) -> Duration {
    instance_tick
        .saturating_mul(3)
        .clamp(MIN_VALIDITY, MAX_VALIDITY)
}

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Convert a [`std::time::Duration`] to a [`chrono::Duration`], saturating
/// on overflow (only possible for durations > ~292 years).
fn to_chrono(d: Duration) -> chrono::Duration {
    // `unwrap_or_else` — lazy so the fallback is never evaluated on the happy
    // path. i64::MAX * 1e9 overflows; use i64::MAX / 1e9 ≈ 292 years instead.
    chrono::Duration::from_std(d)
        .unwrap_or_else(|_| chrono::Duration::seconds(i64::MAX / 1_000_000_000))
}

/// Derive counter-sample buffer capacity from the rate window and cadence.
///
/// Capacity is `ceil(rate_window / instance_tick) + 2`, clamped to
/// `[MIN_CAPACITY, MAX_CAPACITY]`. A zero or sub-millisecond tick uses
/// `MAX_CAPACITY` as a safe fallback.
fn derive_capacity(rw: Duration, instance_tick: Duration) -> usize {
    if instance_tick.is_zero() {
        return MAX_CAPACITY;
    }
    let ratio = rw.as_secs_f64() / instance_tick.as_secs_f64();
    if !ratio.is_finite() {
        return MAX_CAPACITY;
    }
    (ratio.ceil() as usize + 2).clamp(MIN_CAPACITY, MAX_CAPACITY)
}

// ── GenerationObservationTracker ─────────────────────────────────────────────

/// Tracks `gen_tps` (generation tokens/s) observations for a single engine
/// instance with allocation-conscious, time-bounded rolling counter samples.
///
/// # Invariants
///
/// * All time is injected by the caller; no wall-clock access inside this type.
/// * Samples in the VecDeque are strictly monotonically increasing in timestamp.
/// * Counter resets (new total < last total) immediately clear the window and
///   value so recovery is clean.
/// * Expiry during [`Self::snapshot`] clears the window so subsequent observe
///   calls re-baseline rather than reusing stale anchors.
/// * [`Self::observe_counter`] and [`Self::observe_direct`] never erase each
///   other's windows; both update the same `current_value` / `last_observed_at`
///   slot on success.
///
/// # Caller responsibilities
///
/// * Call [`Self::invalidate`] on identity change, container removal, or any
///   transition to a confirmed non-serving terminal state.
/// * Supply monotonically increasing timestamps to `observe_counter`; out-of-
///   order or duplicate timestamps are silently ignored.
/// * Pass `now` to [`Self::snapshot`] at or after the latest `observed_at`;
///   a `now` earlier than `observed_at` yields a negative signed duration that
///   never exceeds the positive validity window, causing a stale hold rather
///   than correct expiry.
pub struct GenerationObservationTracker {
    /// Derived from constructor cadence; exposed via accessor.
    rate_window: Duration,
    /// Derived from constructor cadence; exposed via accessor.
    validity: Duration,
    /// Maximum number of cumulative-counter samples retained.
    capacity: usize,
    /// Rolling window of `(timestamp, cumulative_count)` samples.
    ///
    /// Invariant: strictly increasing timestamps; len ≤ capacity; all
    /// timestamps are within `rate_window` of the newest entry.
    samples: VecDeque<(DateTime<Utc>, f64)>,
    /// Most recently computed or directly observed rate (tokens/s), or `None`
    /// when no valid observation has been made (or after expiry/invalidation).
    current_value: Option<f64>,
    /// Timestamp of the observation that produced `current_value`. `None` iff
    /// `current_value` is `None`. Set only on successful rate computation or
    /// valid direct-rate injection; never set on missing/transient data.
    last_observed_at: Option<DateTime<Utc>>,
}

impl GenerationObservationTracker {
    /// Construct a tracker for the given polling cadence.
    ///
    /// Derives `rate_window` and `validity` from `instance_tick` according to
    /// the frozen EAI-7960 contract and sizes the internal buffer accordingly.
    pub fn new(instance_tick: Duration) -> Self {
        let rw = rate_window(instance_tick);
        let validity = observation_validity(instance_tick);
        let capacity = derive_capacity(rw, instance_tick);
        Self {
            rate_window: rw,
            validity,
            capacity,
            samples: VecDeque::with_capacity(capacity.min(16)),
            current_value: None,
            last_observed_at: None,
        }
    }

    /// The rolling rate-computation window.
    pub const fn rate_window(&self) -> Duration {
        self.rate_window
    }

    /// The observation validity window (how long a value stays Held).
    pub const fn validity(&self) -> Duration {
        self.validity
    }

    /// Record a new cumulative counter sample (`total` tokens generated so far).
    ///
    /// **Non-finite or negative values** are treated as missing and ignored;
    /// neither the counter window nor the current observation is modified.
    ///
    /// **Counter reset** (new total < most-recent total): the counter window
    /// and current observation are cleared, the new sample becomes the sole
    /// baseline, and no rate is emitted until a second sample arrives.
    ///
    /// **Out-of-order or duplicate timestamps** are silently discarded.
    ///
    /// **Unchanged counter** over a window with ≥ 2 samples produces
    /// `Some(0.0)` — zero is a real value, not "unavailable".
    pub fn observe_counter(&mut self, total: f64, observed_at: DateTime<Utc>) {
        // Non-finite or negative total is missing, not zero.
        if !total.is_finite() || total < 0.0 {
            return;
        }

        if let Some(&(back_ts, back_count)) = self.samples.back() {
            // Out-of-order or duplicate timestamp.
            if observed_at <= back_ts {
                return;
            }
            // Counter reset: lower total than most-recent sample.
            if total < back_count {
                self.samples.clear();
                self.current_value = None;
                self.last_observed_at = None;
                // New baseline; no rate until second sample.
                self.samples.push_back((observed_at, total));
                return;
            }
        }

        // Add sample (enforce capacity before pushing to avoid over-allocation).
        if self.samples.len() >= self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back((observed_at, total));

        // Discard samples outside the rate window (older than newest − rate_window).
        let window_chrono = to_chrono(self.rate_window);
        while self.samples.len() > 1 {
            let front_ts = self.samples.front().unwrap().0;
            if observed_at.signed_duration_since(front_ts) > window_chrono {
                self.samples.pop_front();
            } else {
                break;
            }
        }

        // Compute rate when ≥ 2 samples exist in the window.
        if self.samples.len() >= 2 {
            let &(oldest_ts, oldest_count) = self.samples.front().unwrap();
            let newest_count = total; // self.samples.back() is (observed_at, total)
            let dt_ms = observed_at
                .signed_duration_since(oldest_ts)
                .num_milliseconds();
            if dt_ms > 0 {
                let dt_s = dt_ms as f64 / 1000.0;
                let rate = (newest_count - oldest_count) / dt_s;
                // rate is always ≥ 0 here because:
                //   - newest_count ≥ oldest_count (reset guard above)
                //   - dt_s > 0
                if rate.is_finite() && rate >= 0.0 {
                    self.current_value = Some(rate);
                    self.last_observed_at = Some(observed_at);
                }
            }
            // dt_ms == 0: identical timestamps should not reach here (guard
            // above returns on observed_at <= back_ts), but be safe.
        }
        // Single sample: baseline established; no rate yet.
    }

    /// Record an engine-reported direct rate (tokens/s).
    ///
    /// Valid means: finite and non-negative. Invalid or missing values
    /// (`f64::NAN`, `f64::INFINITY`, negative) are silently ignored;
    /// the existing observation and counter window are preserved.
    ///
    /// Does not touch the counter-sample window.
    pub fn observe_direct(&mut self, rate: f64, observed_at: DateTime<Utc>) {
        if !rate.is_finite() || rate < 0.0 {
            return;
        }
        self.current_value = Some(rate);
        self.last_observed_at = Some(observed_at);
    }

    /// Return the current rate and observation metadata for the given `now`.
    ///
    /// | Condition | Returns |
    /// |---|---|
    /// | No valid observation | `(None, None)` |
    /// | Expired (`now − last_observed_at > validity`) | `(None, None)` + clears window |
    /// | Observed at this snapshot timestamp | `(Some(rate), Some(Fresh))` |
    /// | Valid but from a prior scrape | `(Some(rate), Some(Held))` |
    ///
    /// **Expiry clears the counter window** so subsequent `observe_counter`
    /// calls re-baseline rather than computing a rate from stale anchors.
    ///
    /// **Freshness** is `Fresh` when `observed_at == now` (value was computed
    /// in this scrape cycle) and `Held` when `observed_at < now`.
    pub fn snapshot(&mut self, now: DateTime<Utc>) -> (Option<f64>, Option<ObservationMetadata>) {
        let (Some(value), Some(observed_at)) = (self.current_value, self.last_observed_at) else {
            return (None, None);
        };

        // Expiry check.
        let validity_chrono = to_chrono(self.validity);
        if now.signed_duration_since(observed_at) > validity_chrono {
            // Clear everything so the next observe_counter call re-baselines.
            self.samples.clear();
            self.current_value = None;
            self.last_observed_at = None;
            return (None, None);
        }

        // Fresh iff the value was produced at this exact snapshot timestamp;
        // any earlier observed_at means the datum came from a prior scrape.
        let freshness = if observed_at == now {
            ObservationFreshness::Fresh
        } else {
            ObservationFreshness::Held
        };

        (
            Some(value),
            Some(ObservationMetadata {
                observed_at,
                freshness,
            }),
        )
    }

    /// Immediately clear all state (value, counter window, freshness tracking).
    ///
    /// Call on identity change, container removal, or any confirmed
    /// non-serving terminal state. The next `observe_counter` call after
    /// `invalidate` re-establishes the baseline from scratch.
    pub fn invalidate(&mut self) {
        self.samples.clear();
        self.current_value = None;
        self.last_observed_at = None;
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    /// A fixed UTC epoch for deterministic test timestamps.
    fn t(secs_offset: i64) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0)
            .unwrap()
            .checked_add_signed(chrono::Duration::seconds(secs_offset))
            .unwrap()
    }

    // ── cadence-clamp tests ──────────────────────────────────────────────────

    #[test]
    fn cadence_rate_window_clamps_to_minimum() {
        // tick = 1 s  →  5×1 = 5 s < 10 s  →  rate_window = 10 s
        let rw = rate_window(Duration::from_secs(1));
        assert_eq!(
            rw,
            Duration::from_secs(10),
            "below minimum must clamp to 10 s"
        );
    }

    #[test]
    fn cadence_observation_validity_clamps_to_minimum() {
        // tick = 1 s  →  3×1 = 3 s < 6 s  →  validity = 6 s
        let v = observation_validity(Duration::from_secs(1));
        assert_eq!(v, Duration::from_secs(6), "below minimum must clamp to 6 s");
    }

    #[test]
    fn cadence_rate_window_clamps_to_maximum() {
        // tick = 20 s  →  5×20 = 100 s > 30 s  →  rate_window = 30 s
        let rw = rate_window(Duration::from_secs(20));
        assert_eq!(
            rw,
            Duration::from_secs(30),
            "above maximum must clamp to 30 s"
        );
    }

    #[test]
    fn cadence_observation_validity_clamps_to_maximum() {
        // tick = 20 s  →  3×20 = 60 s > 30 s  →  validity = 30 s
        let v = observation_validity(Duration::from_secs(20));
        assert_eq!(
            v,
            Duration::from_secs(30),
            "above maximum must clamp to 30 s"
        );
    }

    #[test]
    fn cadence_rate_window_mid_range() {
        // tick = 4 s  →  5×4 = 20 s  →  rate_window = 20 s
        let rw = rate_window(Duration::from_secs(4));
        assert_eq!(rw, Duration::from_secs(20));
    }

    #[test]
    fn cadence_observation_validity_mid_range() {
        // tick = 4 s  →  3×4 = 12 s  →  validity = 12 s
        let v = observation_validity(Duration::from_secs(4));
        assert_eq!(v, Duration::from_secs(12));
    }

    #[test]
    fn cadence_exact_minimum_boundary() {
        // tick = 2 s  →  5×2 = 10 s  →  rate_window exactly at minimum = 10 s
        let rw = rate_window(Duration::from_secs(2));
        assert_eq!(
            rw,
            Duration::from_secs(10),
            "exact min boundary must not be clamped"
        );

        // tick = 2 s  →  3×2 = 6 s  →  validity exactly at minimum = 6 s
        let v = observation_validity(Duration::from_secs(2));
        assert_eq!(
            v,
            Duration::from_secs(6),
            "exact min boundary must not be clamped"
        );
    }

    #[test]
    fn cadence_exact_maximum_boundary() {
        // tick = 6 s  →  5×6 = 30 s  →  rate_window exactly at maximum = 30 s
        let rw = rate_window(Duration::from_secs(6));
        assert_eq!(
            rw,
            Duration::from_secs(30),
            "exact max boundary must not be clamped"
        );

        // tick = 10 s  →  3×10 = 30 s  →  validity exactly at maximum = 30 s
        let v = observation_validity(Duration::from_secs(10));
        assert_eq!(
            v,
            Duration::from_secs(30),
            "exact max boundary must not be clamped"
        );
    }

    #[test]
    fn cadence_zero_tick_no_panic() {
        // Zero tick must clamp to the respective minimums, not panic.
        let rw = rate_window(Duration::ZERO);
        assert_eq!(rw, MIN_RATE_WINDOW, "zero tick rate_window must be minimum");
        let v = observation_validity(Duration::ZERO);
        assert_eq!(v, MIN_VALIDITY, "zero tick validity must be minimum");
    }

    #[test]
    fn cadence_extreme_tick_no_overflow() {
        // u64::MAX seconds would overflow a plain `*5`; saturating_mul must handle it.
        let huge = Duration::from_secs(u64::MAX);
        let rw = rate_window(huge);
        assert_eq!(rw, MAX_RATE_WINDOW, "saturated tick must clamp to 30 s");
        let v = observation_validity(huge);
        assert_eq!(v, MAX_VALIDITY, "saturated tick must clamp to 30 s");
    }

    // ── tracker construction ─────────────────────────────────────────────────

    #[test]
    fn tracker_new_exposes_derived_windows() {
        let tick = Duration::from_secs(4);
        let tracker = GenerationObservationTracker::new(tick);
        assert_eq!(tracker.rate_window(), Duration::from_secs(20));
        assert_eq!(tracker.validity(), Duration::from_secs(12));
    }

    // ── single-sample baseline ───────────────────────────────────────────────

    #[test]
    fn tracker_first_sample_no_rate() {
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_counter(100.0, t(0));
        // First sample establishes baseline; no rate yet.
        let (val, meta) = tracker.snapshot(t(0));
        assert!(val.is_none(), "first sample must not produce a rate");
        assert!(meta.is_none(), "no metadata without a rate");
    }

    // ── rolling counter rate ─────────────────────────────────────────────────

    #[test]
    fn tracker_two_samples_positive_rate() {
        // 200 tokens in 10 s = 20 tok/s.
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_counter(0.0, t(0));
        tracker.observe_counter(200.0, t(10));
        let (val, meta) = tracker.snapshot(t(10));
        let rate = val.expect("two samples must produce a rate");
        assert!(
            (rate - 20.0).abs() < 1e-9,
            "rate must be 20 tok/s, got {rate}"
        );
        let obs = meta.expect("metadata must be present");
        assert_eq!(obs.freshness, ObservationFreshness::Fresh);
    }

    #[test]
    fn tracker_unchanged_counter_yields_zero_rate() {
        // Same count across two samples = 0 tok/s, not "unavailable".
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_counter(500.0, t(0));
        tracker.observe_counter(500.0, t(10));
        let (val, meta) = tracker.snapshot(t(10));
        assert_eq!(val, Some(0.0), "unchanged counter must yield 0.0, not None");
        assert!(meta.is_some());
    }

    #[test]
    fn tracker_zero_counter_value_is_valid() {
        // A counter that starts and stays at 0 is real, not missing.
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_counter(0.0, t(0));
        tracker.observe_counter(0.0, t(5));
        let (val, _) = tracker.snapshot(t(5));
        assert_eq!(val, Some(0.0), "zero counter must yield 0.0, not None");
    }

    // ── counter reset ────────────────────────────────────────────────────────

    #[test]
    fn tracker_counter_reset_clears_and_rebaselines() {
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        // Establish a valid rate first.
        tracker.observe_counter(0.0, t(0));
        tracker.observe_counter(100.0, t(5));
        let (val, _) = tracker.snapshot(t(5));
        assert!(val.is_some(), "pre-reset rate must exist");

        // Simulate counter reset (engine restarted: count drops).
        tracker.observe_counter(10.0, t(6)); // 10 < 100 → reset
        let (val2, _) = tracker.snapshot(t(6));
        assert!(
            val2.is_none(),
            "immediately after reset, rate must be unavailable (re-baselining)"
        );

        // Second sample after reset → rate resumes.
        tracker.observe_counter(30.0, t(8)); // 20 tokens in 2 s = 10 tok/s
        let (val3, meta3) = tracker.snapshot(t(8));
        let rate = val3.expect("rate must resume after reset + second sample");
        assert!(
            (rate - 10.0).abs() < 1e-9,
            "rate after reset must be 10 tok/s, got {rate}"
        );
        assert_eq!(meta3.unwrap().freshness, ObservationFreshness::Fresh);
    }

    // ── non-finite / invalid counter values ──────────────────────────────────

    #[test]
    fn tracker_nonfinite_counter_nan_ignored() {
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_counter(0.0, t(0));
        tracker.observe_counter(100.0, t(5));
        let (before, _) = tracker.snapshot(t(5));
        assert!(before.is_some());

        // NaN is not finite → ignored; observation not refreshed.
        tracker.observe_counter(f64::NAN, t(6));
        let (after, meta) = tracker.snapshot(t(6));
        // Value should still be held from t(5).
        assert!(after.is_some(), "NaN observe must not erase existing value");
        assert_eq!(meta.unwrap().freshness, ObservationFreshness::Held);
    }

    #[test]
    fn tracker_nonfinite_counter_inf_ignored() {
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_counter(0.0, t(0));
        tracker.observe_counter(50.0, t(5));
        tracker.observe_counter(f64::INFINITY, t(6)); // ignored
        let (val, _) = tracker.snapshot(t(6));
        // Rate from (0,t0)→(50,t5) = 10 tok/s; inf must not alter it.
        assert!(val.is_some());
    }

    #[test]
    fn tracker_negative_counter_ignored() {
        // Negative totals are physically impossible for a counter; treat as missing.
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_counter(0.0, t(0));
        tracker.observe_counter(100.0, t(5));
        tracker.observe_counter(-1.0, t(6)); // ignored
        let (val, _) = tracker.snapshot(t(6));
        assert!(
            val.is_some(),
            "negative counter must be ignored, not treated as reset"
        );
    }

    // ── out-of-order / duplicate timestamps ─────────────────────────────────

    #[test]
    fn tracker_out_of_order_timestamp_ignored() {
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_counter(0.0, t(10));
        tracker.observe_counter(100.0, t(20));
        // An older timestamp must be silently dropped.
        tracker.observe_counter(200.0, t(15)); // t(15) < t(20) → ignored
        let (val, _) = tracker.snapshot(t(20));
        // Rate should still be from (0,t10)→(100,t20) = 10 tok/s.
        let rate = val.expect("rate must exist");
        assert!(
            (rate - 10.0).abs() < 1e-9,
            "out-of-order sample must not corrupt rate, got {rate}"
        );
    }

    #[test]
    fn tracker_duplicate_timestamp_ignored() {
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_counter(0.0, t(0));
        tracker.observe_counter(100.0, t(10));
        tracker.observe_counter(150.0, t(10)); // same ts → ignored
        let (val, _) = tracker.snapshot(t(10));
        let rate = val.expect("rate must exist");
        // Rate should still be (0→100)/10 = 10, not (0→150)/10 = 15.
        assert!(
            (rate - 10.0).abs() < 1e-9,
            "duplicate timestamp must not update rate, got {rate}"
        );
    }

    // ── direct-rate injection ────────────────────────────────────────────────

    #[test]
    fn tracker_observe_direct_sets_value_immediately() {
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_direct(42.5, t(0));
        let (val, meta) = tracker.snapshot(t(0));
        assert_eq!(val, Some(42.5));
        assert_eq!(meta.unwrap().freshness, ObservationFreshness::Fresh);
    }

    #[test]
    fn tracker_direct_zero_rate_is_valid() {
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_direct(0.0, t(0));
        let (val, _) = tracker.snapshot(t(0));
        assert_eq!(val, Some(0.0), "zero direct rate must be valid");
    }

    #[test]
    fn tracker_direct_nan_does_not_refresh() {
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_direct(10.0, t(0));
        tracker.observe_direct(f64::NAN, t(5)); // ignored
        let (val, meta) = tracker.snapshot(t(5));
        assert_eq!(val, Some(10.0), "NaN direct must be ignored");
        assert_eq!(meta.unwrap().freshness, ObservationFreshness::Held);
    }

    #[test]
    fn tracker_direct_infinity_does_not_refresh() {
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_direct(5.0, t(0));
        tracker.observe_direct(f64::INFINITY, t(5));
        let (val, _) = tracker.snapshot(t(5));
        assert_eq!(val, Some(5.0));
    }

    #[test]
    fn tracker_direct_negative_does_not_refresh() {
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_direct(7.0, t(0));
        tracker.observe_direct(-1.0, t(5));
        let (val, _) = tracker.snapshot(t(5));
        assert_eq!(val, Some(7.0), "negative direct rate must be ignored");
    }

    // ── transient hold: missing data preserves valid observation ─────────────

    #[test]
    fn tracker_transient_missing_preserves_held_observation() {
        // After a valid rate, a missing counter cycle must hold the value.
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_counter(0.0, t(0));
        tracker.observe_counter(100.0, t(5));
        let (_, m1) = tracker.snapshot(t(5));
        assert_eq!(m1.unwrap().freshness, ObservationFreshness::Fresh);

        // Missing data cycle: NaN counter, no direct rate.
        tracker.observe_counter(f64::NAN, t(7)); // ignored

        // Snapshot before validity expires: value must still be Held.
        let (val2, m2) = tracker.snapshot(t(7));
        assert!(
            val2.is_some(),
            "transient missing must not erase held observation"
        );
        assert_eq!(
            m2.unwrap().freshness,
            ObservationFreshness::Held,
            "must be Held, not Fresh, when no new observation"
        );
    }

    #[test]
    fn tracker_counter_window_preserved_through_missing_data() {
        // Counter window must survive a missing-data cycle for recovery.
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_counter(0.0, t(0));
        tracker.observe_counter(100.0, t(5));

        // Simulate missing cycle.
        tracker.observe_counter(f64::NAN, t(7));

        // New valid counter: rate should use the existing window for computation.
        tracker.observe_counter(110.0, t(8)); // 10 tokens from t(0) to t(8) = 110/8 ≠ 10/3
        let (val, _) = tracker.snapshot(t(8));
        // Window has (t0,0), (t5,100), (t8,110) → oldest is t0, newest is t8
        // rate = (110-0)/8 = 13.75 tok/s
        let rate = val.expect("rate must recover after missing cycle");
        assert!(
            rate > 0.0,
            "rate after missing cycle must be positive, got {rate}"
        );
    }

    // ── fresh vs held ────────────────────────────────────────────────────────

    #[test]
    fn tracker_snapshot_fresh_on_new_observation_held_on_repeat() {
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_counter(0.0, t(0));
        tracker.observe_counter(100.0, t(5));

        // First snapshot: Fresh (observed since last snapshot = never).
        let (_, m1) = tracker.snapshot(t(5));
        assert_eq!(m1.unwrap().freshness, ObservationFreshness::Fresh);

        // Second snapshot without new observe: Held.
        let (_, m2) = tracker.snapshot(t(6));
        assert_eq!(
            m2.unwrap().freshness,
            ObservationFreshness::Held,
            "repeated snapshot without new observe must be Held"
        );

        // New observation: Fresh again.
        tracker.observe_counter(150.0, t(7));
        let (_, m3) = tracker.snapshot(t(7));
        assert_eq!(
            m3.unwrap().freshness,
            ObservationFreshness::Fresh,
            "observation after Held snapshot must be Fresh again"
        );
    }

    // ── expiry ───────────────────────────────────────────────────────────────

    #[test]
    fn tracker_snapshot_expiry_returns_none_and_clears() {
        // validity = clamp(3×2, 6, 30) = 6 s.
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_counter(0.0, t(0));
        tracker.observe_counter(100.0, t(5));
        let (before, _) = tracker.snapshot(t(5));
        assert!(before.is_some(), "pre-expiry snapshot must have a value");

        // Advance past validity (6 s from t(5) = t(11)).
        let (expired, meta) = tracker.snapshot(t(12));
        assert!(expired.is_none(), "post-expiry snapshot must return None");
        assert!(meta.is_none(), "no metadata after expiry");
    }

    #[test]
    fn tracker_expiry_at_exact_boundary() {
        // Snapshot exactly at observed_at + validity must return None (strict >).
        let validity = observation_validity(Duration::from_secs(2)); // 6 s
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_counter(0.0, t(0));
        tracker.observe_counter(100.0, t(5));
        tracker.snapshot(t(5)); // mark as seen

        // At exactly t(5) + 6 s = t(11): still within validity (not strictly >).
        let boundary_secs =
            5 + i64::try_from(validity.as_secs()).expect("validity ≤ 30 s; fits i64");
        let (at_boundary, _) = tracker.snapshot(t(boundary_secs));
        // The boundary behavior depends on whether > or >= is used; document actual:
        // Our contract uses strict `>` so at-boundary is still Held.
        assert!(
            at_boundary.is_some(),
            "at exact validity boundary must still be Held (strictly >, not >=)"
        );
    }

    #[test]
    fn tracker_expiry_clears_counter_window_for_recovery() {
        // After expiry, counter window must be cleared so the next observe_counter
        // starts a fresh baseline rather than computing a rate from stale anchors.
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_counter(0.0, t(0));
        tracker.observe_counter(1000.0, t(5));
        tracker.snapshot(t(5));

        // Expire.
        tracker.snapshot(t(12));

        // First new sample after expiry → baseline only.
        tracker.observe_counter(50.0, t(13));
        let (after_first, _) = tracker.snapshot(t(13));
        assert!(
            after_first.is_none(),
            "first sample after expiry must be baseline, not a rate"
        );

        // Second sample → rate computed fresh (from baseline at t13).
        tracker.observe_counter(70.0, t(15)); // 20 tok in 2 s = 10 tok/s
        let (after_second, _) = tracker.snapshot(t(15));
        let rate = after_second.expect("second sample after re-baseline must yield rate");
        assert!(
            (rate - 10.0).abs() < 1e-9,
            "re-baselined rate must be 10 tok/s, got {rate}"
        );
    }

    // ── explicit invalidation ────────────────────────────────────────────────

    #[test]
    fn tracker_invalidate_clears_all_state() {
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_counter(0.0, t(0));
        tracker.observe_counter(100.0, t(5));
        tracker.snapshot(t(5));

        tracker.invalidate();

        let (val, meta) = tracker.snapshot(t(6));
        assert!(val.is_none(), "invalidate must clear value");
        assert!(meta.is_none(), "invalidate must clear metadata");
    }

    #[test]
    fn tracker_invalidate_forces_rebaseline_on_next_observe() {
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_counter(0.0, t(0));
        tracker.observe_counter(100.0, t(5));
        tracker.invalidate();

        // First sample after invalidate: baseline only, no rate.
        tracker.observe_counter(200.0, t(10));
        let (val, _) = tracker.snapshot(t(10));
        assert!(
            val.is_none(),
            "first observe after invalidate must be baseline"
        );

        // Second sample: rate resumes.
        tracker.observe_counter(250.0, t(15)); // 50 tok in 5 s = 10 tok/s
        let (val2, _) = tracker.snapshot(t(15));
        let rate = val2.expect("must have rate after re-baseline");
        assert!(
            (rate - 10.0).abs() < 1e-9,
            "rate must be 10 tok/s, got {rate}"
        );
    }

    // ── storage bound ────────────────────────────────────────────────────────

    #[test]
    fn tracker_storage_cannot_grow_without_bound() {
        // Feed 1000 samples at 1 s intervals; the VecDeque must stay ≤ capacity.
        let tick = Duration::from_secs(2);
        let mut tracker = GenerationObservationTracker::new(tick);
        let cap = tracker.capacity;

        for i in 0..1000_i64 {
            tracker.observe_counter(i as f64 * 10.0, t(i));
        }
        assert!(
            tracker.samples.len() <= cap,
            "sample buffer must not exceed capacity ({cap}), got {}",
            tracker.samples.len()
        );
    }

    #[test]
    fn tracker_capacity_is_bounded_by_constants() {
        // All cadences must produce a capacity in [MIN_CAPACITY, MAX_CAPACITY].
        for tick_secs in [0u64, 1, 2, 5, 10, 20, 100, u64::MAX / 2] {
            let tick = Duration::from_secs(tick_secs);
            let rw = rate_window(tick);
            let cap = derive_capacity(rw, tick);
            assert!(
                cap >= MIN_CAPACITY,
                "capacity {cap} below MIN_CAPACITY for tick={tick_secs} s"
            );
            assert!(
                cap <= MAX_CAPACITY,
                "capacity {cap} above MAX_CAPACITY for tick={tick_secs} s"
            );
        }
    }

    // ── direct rate parity ───────────────────────────────────────────────────

    #[test]
    fn tracker_direct_and_counter_rates_use_same_slot() {
        // observe_direct and observe_counter both update the same value/metadata.
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        tracker.observe_direct(99.0, t(0));
        let (v1, _) = tracker.snapshot(t(0));
        assert_eq!(v1, Some(99.0));

        // Counter replaces the direct rate on next successful compute.
        tracker.observe_counter(0.0, t(1));
        tracker.observe_counter(50.0, t(6)); // 50/5 = 10 tok/s
        let (v2, _) = tracker.snapshot(t(6));
        assert!(
            (v2.unwrap() - 10.0).abs() < 1e-9,
            "counter rate must supersede prior direct rate"
        );
    }

    // ── counter→direct→counter interleaving (M3) ─────────────────────────────

    #[test]
    fn tracker_direct_does_not_clear_counter_window() {
        // observe_direct must not touch the counter VecDeque; a subsequent
        // observe_counter call must still compute its rate from the retained
        // window anchors as if the direct injection never happened.
        let mut tracker = GenerationObservationTracker::new(Duration::from_secs(2));
        // Establish counter window: (t0, 0) and (t5, 100).
        tracker.observe_counter(0.0, t(0));
        tracker.observe_counter(100.0, t(5)); // rate = 20 tok/s
        // Direct injection overrides current_value but must not clear samples.
        tracker.observe_direct(50.0, t(6));
        // Resume counter. If the window was cleared this would be a lone
        // baseline (no rate). Retained: window is (t0,0),(t5,100),(t8,130).
        tracker.observe_counter(130.0, t(8));
        let (val, _) = tracker.snapshot(t(8));
        let rate = val.expect("counter must resume from retained window after direct injection");
        // rate = (130 - 0) / (8 - 0) = 16.25 tok/s.
        assert!(
            (rate - 16.25).abs() < 1e-9,
            "rate must be (130-0)/(8-0) = 16.25 tok/s, got {rate}"
        );
    }

    // ── supported-cadence capacity bound (M4) ────────────────────────────────

    #[test]
    fn tracker_capacity_within_max_for_supported_cadences() {
        // For all real deployment cadences (instance_tick >= 1 s) the derived
        // capacity is at most ceil(30/1)+2 = 32, well below MAX_CAPACITY = 64.
        for tick_secs in [1u64, 2, 4, 5, 6, 10, 20] {
            let tick = Duration::from_secs(tick_secs);
            let cap = derive_capacity(rate_window(tick), tick);
            assert!(
                cap < MAX_CAPACITY,
                "tick={tick_secs}s cap={cap} must be below MAX_CAPACITY={MAX_CAPACITY}"
            );
        }
        // 1 s tick: rate_window = 10 s → ceil(10/1)+2 = 12.
        let cap_1s = derive_capacity(rate_window(Duration::from_secs(1)), Duration::from_secs(1));
        assert_eq!(cap_1s, 12, "1 s tick must yield capacity exactly 12");
    }
}
