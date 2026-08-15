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
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .message
            .take()
    }

    /// Whether the reader has failed and can no longer update the screen.
    pub fn has_failed(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .failed
    }
}

#[cfg(test)]
mod tests {
    use super::ReaderFailure;
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
        let remains_failed = failure.has_failed();

        release_tx.send(()).expect("reader should remain blocked");
        reader.join().expect("test reader should exit cleanly");

        assert!(
            remains_failed,
            "consuming the diagnostic must not clear terminal failure state"
        );
    }
}
