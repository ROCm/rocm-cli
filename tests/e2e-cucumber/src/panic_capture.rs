// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Panic-message extraction and mutex-poison recovery.
//!
//! Shared by black-box drivers (currently `tests/e2e/tui_driver.rs`) that run
//! background OS threads they can't afford to let fail silently. This lives in
//! the library target (unlike `tui_driver.rs`, which is deliberately part of
//! the `harness = false` cucumber test binary and has no `#[test]`s of its own)
//! specifically so its pure logic gets real `#[test]` coverage under `cargo
//! test -p e2e-cucumber --lib` — the cucumber binary's custom harness never
//! executes plain `#[test]` functions placed inside it.

/// Best-effort extraction of a human-readable message from a `catch_unwind`
/// payload (which is typically a `&str` or `String`, but isn't guaranteed to be).
pub fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "thread panicked with a non-string payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::panic_message;
    use std::sync::{Arc, Mutex};

    /// A lock poisoned by a panicking holder must still be recoverable rather
    /// than treated as unusable: this is the same pattern `tui_driver`'s
    /// `screen_snapshot`/`use_detail_size` rely on to keep reading/writing a
    /// shared parser after some unrelated panic, instead of falling back to a
    /// silently blank screen.
    #[test]
    fn poisoned_mutex_is_recoverable() {
        let mutex = Arc::new(Mutex::new(vec![1, 2, 3]));
        let poisoned = Arc::clone(&mutex);
        let result = std::panic::catch_unwind(move || {
            let mut guard = poisoned.lock().unwrap();
            guard.push(4);
            panic!("simulated panic while holding the lock");
        });
        assert!(result.is_err());
        assert!(mutex.is_poisoned());

        // The same recovery `tui_driver` uses: `unwrap_or_else` with
        // `PoisonError::into_inner` instead of propagating or defaulting.
        let guard = mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // The data written before the panic is still intact and readable.
        assert_eq!(*guard, vec![1, 2, 3, 4]);
    }

    /// `panic_message` must handle both common `catch_unwind` payload shapes
    /// (`&str` from `panic!("literal")`, `String` from `panic!("{}", x)`) plus
    /// the fallback for anything else, since a poorly-typed panic payload
    /// shouldn't itself cause a second failure while building a diagnostic.
    #[test]
    fn panic_message_handles_str_and_string_and_other_payloads() {
        let str_payload: Box<dyn std::any::Any + Send> = Box::new("literal panic");
        assert_eq!(panic_message(&str_payload), "literal panic");

        let string_payload: Box<dyn std::any::Any + Send> =
            Box::new(format!("formatted {}", "panic"));
        assert_eq!(panic_message(&string_payload), "formatted panic");

        let other_payload: Box<dyn std::any::Any + Send> = Box::new(42_i32);
        assert_eq!(
            panic_message(&other_payload),
            "thread panicked with a non-string payload"
        );
    }

    /// Exercises the same catch/record sequence `tui_driver::spawn_reader` uses
    /// around `vt100::Parser::process`, without needing a real PTY or parser:
    /// a panic while holding a lock is caught, the lock is dropped (not held
    /// across the panic boundary), and the message is captured for a caller to
    /// report later — rather than the OS thread just dying silently.
    #[test]
    fn captured_panic_is_recorded_not_lost() {
        let state: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        let captured: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        let mut guard = state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            guard.extend_from_slice(b"partial frame");
            panic!("simulated processing panic");
        }));
        drop(guard);
        let payload = result.expect_err("closure above always panics");
        let message = panic_message(&payload);
        *captured
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(message);

        let message = captured
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        assert_eq!(message.as_deref(), Some("simulated processing panic"));
        // The state itself is still usable afterwards (poison was recovered).
        assert_eq!(
            *state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            b"partial frame"
        );
    }
}
