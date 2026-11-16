// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! What scenario teardown does about a process a scenario started and recorded.
//!
//! A scenario that records a process a stop is supposed to take down (a parent
//! and the child it started, say) can fail with one of them left running, and a
//! later stop cannot find a child whose parent is gone. Teardown therefore
//! keeps the child's own identity and ends it itself. Whether to do so is
//! decided here, apart from the system reads, so a table test covers each case
//! on every host: only a process that is still the one recorded is ended, never
//! one that has since exited or whose PID another process has taken.

/// What the host says about a recorded process now.
///
/// It is read against the identity the scenario captured for the process. The
/// harness maps the production identity check onto these (`rocm-core` is a
/// dev-dependency of this crate, so the library does not name it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordedState {
    /// Running, and the same process (or recorded without a token, which cannot
    /// be refuted).
    Same,
    /// No longer running.
    Gone,
    /// Running, but provably a different process that took the PID.
    Reused,
    /// Running, but its identity cannot be confirmed.
    Unconfirmed,
}

/// What teardown does about one recorded process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Leftover {
    /// End it: it is still the process the scenario recorded.
    Terminate,
    /// Leave it: it is gone, or is not (provably) the recorded process.
    Leave,
}

/// The teardown action for a recorded process whose identity reads as `state`.
///
/// A process that cannot be confirmed as the recorded one is left alone: ending
/// a stranger that happens to hold a recycled PID is worse than leaving a
/// stray five-minute process to time out.
#[must_use]
pub const fn leftover_action(state: RecordedState) -> Leftover {
    match state {
        RecordedState::Same => Leftover::Terminate,
        RecordedState::Gone | RecordedState::Reused | RecordedState::Unconfirmed => Leftover::Leave,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_process_that_is_still_the_recorded_one_is_ended() {
        for (state, expected) in [
            // Alive and the same process (or recorded without a token, which
            // cannot be refuted): the stray child a failed stop left behind.
            (RecordedState::Same, Leftover::Terminate),
            // The stop took it down already.
            (RecordedState::Gone, Leftover::Leave),
            // Its PID has been reused: a stranger.
            (RecordedState::Reused, Leftover::Leave),
            // Its token cannot be read back, so it cannot be confirmed.
            (RecordedState::Unconfirmed, Leftover::Leave),
        ] {
            assert_eq!(leftover_action(state), expected, "{state:?}");
        }
    }
}
