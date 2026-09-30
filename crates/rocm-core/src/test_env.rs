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
/// `*_TEST_LOCK` first — and because this type is deliberately not named like
/// one, the env-mutation contract guard in `xtask` keeps asking it to.
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
