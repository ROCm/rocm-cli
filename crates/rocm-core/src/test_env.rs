// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Restoring a process environment variable for the duration of a test.
//!
//! A test that replaces a variable, calls into the code under test, and then
//! puts the old value back in straight-line code leaks that variable if
//! anything in between panics. The `*_TEST_LOCK` mutexes these tests take are
//! acquired with [`std::sync::PoisonError::into_inner`], deliberately, so a
//! poisoned lock does not cascade into every later test — which means the leak
//! is not contained either: the next test to take the lock reads the value the
//! panicking one planted, usually pointing at a temporary directory that has
//! since been deleted.
//!
//! [`RestoredEnvVar`] moves the restore into `Drop`, where unwinding runs it.

use std::ffi::OsString;
use std::path::Path;

/// Sets an environment variable and puts the previous value back on drop.
///
/// This RESTORES; it does not SERIALIZE. Two tests holding one of these for the
/// same key still race each other, so the caller must still take the key's
/// `*_TEST_LOCK` first.
///
/// The env-mutation contract guard in `xtask` enforces that, but only because
/// `RestoredEnvVar::set(` is named in its `MUTATIONS` list. The guard matches
/// call TEXT, so wrapping a mutation in a method hides it by default — a
/// restoring helper added later is outside the guard until it is listed too.
pub(crate) struct RestoredEnvVar {
    key: &'static str,
    previous: Option<OsString>,
}

impl RestoredEnvVar {
    /// Sets `key` to `value`, remembering what was there before.
    #[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
    pub(crate) fn set(key: &'static str, value: &Path) -> Self {
        let previous = std::env::var_os(key);
        // SAFETY: every caller holds the process-wide test lock for this key,
        // and the value is restored below before that lock is released.
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }
}

impl Drop for RestoredEnvVar {
    #[allow(unsafe_code)] // std::env::set_var/remove_var are unsafe in edition 2024
    fn drop(&mut self) {
        // SAFETY: as above -- still inside the caller's lock scope, since the
        // guard is declared after the lock and so drops before it.
        unsafe {
            match self.previous.as_ref() {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RestoredEnvVar;
    use std::path::Path;

    /// Serializes the env mutation below, as the contract guard requires.
    static TEST_ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Used by nothing else in the tree, so no other test reads it mid-flight.
    const KEY: &str = "ROCM_CORE_TEST_ENV_RESTORE_PROBE";

    /// The restore is this type's whole purpose, and it is what the two callers
    /// rely on to keep a panicking test from leaking a value into the next one.
    ///
    /// Both prior states are needed, because they catch different mistakes:
    ///
    /// * *previously unset* must come back **unset**, not merely empty — which a
    ///   `var().unwrap_or_default()`-style capture would get wrong — and it also
    ///   catches capturing the previous value *after* the mutation, which would
    ///   "restore" the new value and leave the key set;
    /// * *previously set* must come back to that exact value, which a `Drop`
    ///   that does nothing would leave as the planted one.
    #[test]
    #[allow(unsafe_code)] // std::env::set_var/remove_var are unsafe in edition 2024
    fn dropping_the_guard_restores_the_previous_state() {
        let _guard = TEST_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        // SAFETY: serialized by the lock above, and nothing else reads this key.
        unsafe { std::env::remove_var(KEY) };
        drop(RestoredEnvVar::set(KEY, Path::new("planted")));
        assert_eq!(
            std::env::var_os(KEY),
            None,
            "a key that was unset must be unset again, not empty or still planted"
        );

        // SAFETY: as above.
        unsafe { std::env::set_var(KEY, "before") };
        let restore = RestoredEnvVar::set(KEY, Path::new("planted"));
        assert_eq!(
            std::env::var_os(KEY).as_deref(),
            Some(std::ffi::OsStr::new("planted")),
            "the guard must actually set the value it was given"
        );
        drop(restore);
        assert_eq!(
            std::env::var_os(KEY).as_deref(),
            Some(std::ffi::OsStr::new("before")),
            "a key that was set must get its previous value back"
        );

        // SAFETY: as above.
        unsafe { std::env::remove_var(KEY) };
    }
}
