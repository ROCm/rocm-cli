// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Failure state shared between a background terminal reader and its waiters.

use std::sync::Mutex;

#[derive(Debug, Default)]
struct State {
    failed: bool,
    message: Option<String>,
}

/// Atomic observation of the reader's failure state.
#[derive(Debug, PartialEq, Eq)]
pub enum ReaderFailureObservation {
    /// The reader has not published a failure.
    Running,
    /// A failure was published, but its diagnostic was already consumed.
    FailedWithoutMessage,
    /// A newly observed failure diagnostic.
    Message(String),
}

/// Stores persistent reader-failure state and a diagnostic for one-time reporting.
#[derive(Debug, Default)]
pub struct ReaderFailure {
    state: Mutex<State>,
}

impl ReaderFailure {
    /// Publish a reader failure for a waiter to report.
    pub fn publish(&self, message: String) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.failed = true;
        state.message = Some(message);
    }

    /// Take the failure diagnostic, if it has not already been reported.
    pub fn take_message(&self) -> Option<String> {
        match self.observe() {
            ReaderFailureObservation::Message(message) => Some(message),
            ReaderFailureObservation::Running | ReaderFailureObservation::FailedWithoutMessage => {
                None
            }
        }
    }

    /// Observe failure state and consume its diagnostic under one lock.
    pub fn observe(&self) -> ReaderFailureObservation {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(message) = state.message.take() {
            ReaderFailureObservation::Message(message)
        } else if state.failed {
            ReaderFailureObservation::FailedWithoutMessage
        } else {
            ReaderFailureObservation::Running
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ReaderFailure, ReaderFailureObservation};
    use std::sync::{Arc, mpsc};

    #[test]
    fn failure_remains_set_after_message_is_consumed_while_reader_finishes() {
        let failure = Arc::new(ReaderFailure::default());
        let reader_failure = Arc::clone(&failure);
        let (published_tx, published_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let reader = std::thread::spawn(move || {
            reader_failure.publish("captured reader panic".to_string());
            published_tx.send(()).expect("test waiter should remain");
            release_rx.recv().expect("test should release reader");
        });

        published_rx.recv().expect("reader should publish failure");
        assert!(!reader.is_finished(), "reader must still be finishing");
        assert_eq!(
            failure.take_message().as_deref(),
            Some("captured reader panic")
        );
        let observation = failure.observe();

        release_tx.send(()).expect("reader should remain blocked");
        reader.join().expect("test reader should exit cleanly");

        assert_eq!(
            observation,
            ReaderFailureObservation::FailedWithoutMessage,
            "consuming the diagnostic must preserve terminal failure state"
        );
    }
}
