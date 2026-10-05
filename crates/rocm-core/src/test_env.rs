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
    use std::ffi::OsString;
    use std::path::Path;

    /// Serializes the env mutation below, as the contract guard requires.
    static TEST_ENV_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Used by nothing else in the tree, so no other test reads it mid-flight.
    const KEY: &str = "ROCM_CORE_TEST_ENV_RESTORE_PROBE";

    /// Leaves [`KEY`] unset when a test exits, including by a failed assertion.
    ///
    /// Declare it after the lock so it drops while the lock is still held. A
    /// test about not leaking environment state should not leak the probe when
    /// it fails, which is exactly when someone would be looking at it.
    ///
    /// Holding one mutates the environment from a `Drop` impl the contract
    /// guard cannot see, so `UnsetKeyOnExit` is named in its `MUTATIONS` list:
    /// a test that holds one without the lock is flagged like a direct call.
    struct UnsetKeyOnExit;

    impl Drop for UnsetKeyOnExit {
        #[allow(unsafe_code)] // std::env::remove_var is unsafe in edition 2024
        fn drop(&mut self) {
            // SAFETY: dropped inside the caller's `TEST_ENV_TEST_LOCK` scope,
            // and nothing else reads this key.
            unsafe { std::env::remove_var(KEY) };
        }
    }

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
        let _unset_on_exit = UnsetKeyOnExit;

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
    }

    /// The path the module docs promise: a test that panics while holding the
    /// guard still gets its previous value back. An explicit `drop` runs the
    /// same destructor today, but only an unwind proves the restore survives a
    /// change such as `panic = "abort"`, under which destructors stop running
    /// on panic and this test aborts instead of passing.
    ///
    /// The panic is raised with [`std::panic::resume_unwind`], which unwinds
    /// without calling the panic hook, so a passing run prints no panic message
    /// and no process-wide hook has to be swapped while other tests run.
    ///
    /// The payload carries the value observed just before the panic. Without
    /// it, `catch_unwind` would accept any panic — including one raised by
    /// `set` itself before it planted anything — and a `set` that planted
    /// nothing would leave `"before"` in place for the final assertion to
    /// pass on. An assertion inside the closure would not help: `catch_unwind`
    /// swallows it and the result is still an `Err`.
    #[test]
    #[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
    fn unwinding_past_the_guard_restores_the_previous_state() {
        let _guard = TEST_ENV_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _unset_on_exit = UnsetKeyOnExit;

        // SAFETY: serialized by the lock above, and nothing else reads this key.
        unsafe { std::env::set_var(KEY, "before") };
        let unwound = std::panic::catch_unwind(|| {
            let _restore = RestoredEnvVar::set(KEY, Path::new("planted"));
            std::panic::resume_unwind(Box::new(std::env::var_os(KEY)));
        });

        assert_eq!(
            unwound.unwrap_err().downcast_ref::<Option<OsString>>(),
            Some(&Some(OsString::from("planted"))),
            "the closure must have unwound, with the guard's value in place"
        );
        assert_eq!(
            std::env::var_os(KEY).as_deref(),
            Some(std::ffi::OsStr::new("before")),
            "unwinding past the guard must put the previous value back"
        );
    }
}
