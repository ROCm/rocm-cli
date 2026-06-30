// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, anyhow, bail};
use keyring_core::api::CredentialStoreApi;
use keyring_core::{Entry, Error as KeyringError};

const PROVIDER_KEY_SERVICE: &str = "org.rocm.rocm-cli.provider-key";

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) enum ProviderKeyState {
    Configured,
    Missing,
    Unavailable,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct ProviderKeyStatus {
    pub state: ProviderKeyState,
    pub source: String,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct ProviderApiKey {
    pub value: String,
    pub source: String,
}

pub(crate) trait ProviderKeyStore: Send + Sync {
    fn label(&self) -> &'static str;
    fn get_secret(&self, provider: &str) -> Result<Option<Vec<u8>>>;
    fn set_secret(&self, provider: &str, secret: &[u8]) -> Result<()>;
    fn clear_secret(&self, provider: &str) -> Result<()>;
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct NativeProviderKeyStore;

pub(crate) fn provider_key_status(provider: &str, env_name: &str) -> ProviderKeyStatus {
    let store = NativeProviderKeyStore;
    provider_key_status_with_store(
        &store,
        provider,
        env_name,
        std::env::var(env_name)
            .ok()
            .filter(|value| !value.trim().is_empty()),
    )
}

pub(crate) fn resolve_provider_api_key(provider: &str, env_name: &str) -> Result<ProviderApiKey> {
    let store = NativeProviderKeyStore;
    resolve_provider_api_key_with_store(
        &store,
        provider,
        env_name,
        std::env::var(env_name)
            .ok()
            .filter(|value| !value.trim().is_empty()),
    )
}

pub(crate) fn set_provider_api_key(provider: &str, value: &str) -> Result<ProviderKeyStatus> {
    let store = NativeProviderKeyStore;
    set_provider_api_key_with_store(&store, provider, value)
}

pub(crate) fn set_provider_api_key_with_store(
    store: &dyn ProviderKeyStore,
    provider: &str,
    value: &str,
) -> Result<ProviderKeyStatus> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        bail!("{provider} API key was empty; nothing was saved");
    }
    ensure_cloud_provider(provider)?;
    store
        .set_secret(provider, trimmed.as_bytes())
        .with_context(|| format!("failed to save {provider} API key in secure storage"))?;
    Ok(ProviderKeyStatus {
        state: ProviderKeyState::Configured,
        source: secure_source_label(store.label()),
    })
}

pub(crate) fn clear_provider_api_key(provider: &str) -> Result<ProviderKeyStatus> {
    let store = NativeProviderKeyStore;
    clear_provider_api_key_with_store(&store, provider)
}

pub(crate) fn clear_provider_api_key_with_store(
    store: &dyn ProviderKeyStore,
    provider: &str,
) -> Result<ProviderKeyStatus> {
    ensure_cloud_provider(provider)?;
    store
        .clear_secret(provider)
        .with_context(|| format!("failed to clear {provider} API key from secure storage"))?;
    Ok(ProviderKeyStatus {
        state: ProviderKeyState::Missing,
        source: secure_source_label(store.label()),
    })
}

pub(crate) fn provider_key_status_label(status: &ProviderKeyStatus) -> String {
    match status.state {
        ProviderKeyState::Configured => {
            if let Some(env_name) = status.source.strip_prefix("env:") {
                format!("using {env_name} for this session")
            } else if let Some(label) = status.source.strip_prefix("secure:") {
                format!("saved in {label}")
            } else {
                format!("saved in {}", status.source)
            }
        }
        ProviderKeyState::Missing => {
            if let Some(label) = status.source.strip_prefix("secure:") {
                format!("no key saved in {label}")
            } else {
                format!("no key saved in {}", status.source)
            }
        }
        ProviderKeyState::Unavailable => format!("key storage unavailable: {}", status.source),
    }
}

fn provider_key_status_with_store(
    store: &dyn ProviderKeyStore,
    provider: &str,
    env_name: &str,
    env_value: Option<String>,
) -> ProviderKeyStatus {
    if env_value.is_some() {
        return ProviderKeyStatus {
            state: ProviderKeyState::Configured,
            source: format!("env:{env_name}"),
        };
    }
    match store.get_secret(provider) {
        Ok(Some(secret)) if !secret.is_empty() => ProviderKeyStatus {
            state: ProviderKeyState::Configured,
            source: secure_source_label(store.label()),
        },
        Ok(_) => ProviderKeyStatus {
            state: ProviderKeyState::Missing,
            source: secure_source_label(store.label()),
        },
        Err(error) => ProviderKeyStatus {
            state: ProviderKeyState::Unavailable,
            source: error.to_string(),
        },
    }
}

fn resolve_provider_api_key_with_store(
    store: &dyn ProviderKeyStore,
    provider: &str,
    env_name: &str,
    env_value: Option<String>,
) -> Result<ProviderApiKey> {
    ensure_cloud_provider(provider)?;
    if let Some(value) = env_value {
        return Ok(ProviderApiKey {
            value,
            source: format!("env:{env_name}"),
        });
    }
    match store.get_secret(provider) {
        Ok(Some(secret)) if !secret.is_empty() => {
            let value = String::from_utf8(secret)
                .context("stored provider API key was not valid UTF-8")?
                .trim()
                .to_owned();
            if value.is_empty() {
                bail!("{provider} API key in secure storage is empty");
            }
            Ok(ProviderApiKey {
                value,
                source: secure_source_label(store.label()),
            })
        }
        Ok(_) => bail!(
            "{provider} provider requires a saved API key; run `rocm config set-provider-key {provider}` or set {env_name} for this session"
        ),
        Err(error) => Err(error).with_context(|| {
            format!("secure API-key storage is unavailable for {provider}; no plaintext fallback was used")
        }),
    }
}

fn ensure_cloud_provider(provider: &str) -> Result<()> {
    if matches!(provider, "openai" | "anthropic") {
        Ok(())
    } else {
        bail!("{provider} does not use a cloud provider API key")
    }
}

fn secure_source_label(label: &str) -> String {
    format!("secure:{label}")
}

impl ProviderKeyStore for NativeProviderKeyStore {
    fn label(&self) -> &'static str {
        native_store_label()
    }

    fn get_secret(&self, provider: &str) -> Result<Option<Vec<u8>>> {
        with_native_entry(provider, |entry| match entry.get_secret() {
            Ok(secret) => Ok(Some(secret)),
            Err(KeyringError::NoEntry) => Ok(None),
            Err(error) => Err(keyring_anyhow(error)),
        })
    }

    fn set_secret(&self, provider: &str, secret: &[u8]) -> Result<()> {
        with_native_entry(provider, |entry| {
            entry.set_secret(secret).map_err(keyring_anyhow)
        })
    }

    fn clear_secret(&self, provider: &str) -> Result<()> {
        with_native_entry(provider, |entry| match entry.delete_credential() {
            Ok(()) | Err(KeyringError::NoEntry) => Ok(()),
            Err(error) => Err(keyring_anyhow(error)),
        })
    }
}

/// Build the native credential entry and run `action` against it.
///
/// The native stores drive their own blocking runtime — notably the Linux Secret
/// Service via `zbus::blocking`, which calls `block_on` internally. If a tokio
/// runtime *context* is already entered on the calling thread, that nested
/// `block_on` panics with "Cannot start a runtime from within a runtime". The
/// dash resolves keys off-runtime, but this is the single chokepoint for every
/// store op (get/set/clear) and for `resolve_provider_api_key` /
/// `provider_key_status`, so guard the whole class here: when a runtime is
/// active, run the entry build *and* the action on a fresh OS thread that has no
/// runtime entered.
///
/// `tokio::task::block_in_place` is deliberately NOT used — it keeps the runtime
/// context entered (and panics outright on a current-thread runtime), so the
/// nested `block_on` would still panic. Only a thread with no entered runtime
/// escapes. `Handle::try_current()` also returns `Ok` on tokio's blocking-pool
/// threads (where the context is not actually entered and no panic would occur),
/// so this is a conservative over-trigger: at worst one short-lived thread.
fn with_native_entry<T: Send>(
    provider: &str,
    action: impl FnOnce(&Entry) -> Result<T> + Send,
) -> Result<T> {
    let run = move || {
        let entry = native_entry(provider)?;
        action(&entry)
    };
    if tokio::runtime::Handle::try_current().is_ok() {
        // Must `join()` the handle and convert a worker panic into an `Err`:
        // otherwise `thread::scope` re-propagates the panic when the scope ends,
        // which would defeat the guard by aborting the process anyway.
        std::thread::scope(|scope| {
            scope
                .spawn(run)
                .join()
                .map_err(|_| anyhow!("secure key-store access thread panicked"))?
        })
    } else {
        run()
    }
}

fn native_entry(provider: &str) -> Result<Entry> {
    #[cfg(target_os = "windows")]
    {
        let store = windows_native_keyring_store::Store::new().map_err(keyring_anyhow)?;
        store
            .build(PROVIDER_KEY_SERVICE, provider, None)
            .map_err(keyring_anyhow)
    }

    #[cfg(target_os = "macos")]
    {
        let store = apple_native_keyring_store::keychain::Store::new_with_configuration(
            &std::collections::HashMap::new(),
        )
        .map_err(keyring_anyhow)?;
        store
            .build(PROVIDER_KEY_SERVICE, provider, None)
            .map_err(keyring_anyhow)
    }

    #[cfg(any(target_os = "linux", target_os = "freebsd", target_os = "openbsd"))]
    {
        let store = zbus_secret_service_keyring_store::Store::new_with_configuration(
            &std::collections::HashMap::new(),
        )
        .map_err(keyring_anyhow)?;
        store
            .build(PROVIDER_KEY_SERVICE, provider, None)
            .map_err(keyring_anyhow)
    }

    #[cfg(not(any(
        target_os = "windows",
        target_os = "macos",
        target_os = "linux",
        target_os = "freebsd",
        target_os = "openbsd",
    )))]
    bail!("this platform does not have a supported secure credential store")
}

const fn native_store_label() -> &'static str {
    if cfg!(target_os = "windows") {
        "Windows Credential Manager"
    } else if cfg!(target_os = "macos") {
        "macOS Keychain"
    } else {
        "Secret Service keychain"
    }
}

fn keyring_anyhow(error: KeyringError) -> anyhow::Error {
    anyhow!("{error}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    #[derive(Default)]
    struct MemoryKeyStore {
        secrets: Mutex<BTreeMap<String, Vec<u8>>>,
        fail: Option<&'static str>,
    }

    impl ProviderKeyStore for MemoryKeyStore {
        fn label(&self) -> &'static str {
            "test keychain"
        }

        fn get_secret(&self, provider: &str) -> Result<Option<Vec<u8>>> {
            if let Some(fail) = self.fail {
                bail!("{fail}");
            }
            Ok(self.secrets.lock().unwrap().get(provider).cloned())
        }

        fn set_secret(&self, provider: &str, secret: &[u8]) -> Result<()> {
            self.secrets
                .lock()
                .unwrap()
                .insert(provider.to_owned(), secret.to_vec());
            Ok(())
        }

        fn clear_secret(&self, provider: &str) -> Result<()> {
            self.secrets.lock().unwrap().remove(provider);
            Ok(())
        }
    }

    #[test]
    fn provider_key_status_reports_env_without_touching_store() {
        let store = MemoryKeyStore {
            fail: Some("store should not be read"),
            ..MemoryKeyStore::default()
        };

        let status = provider_key_status_with_store(
            &store,
            "openai",
            "OPENAI_API_KEY",
            Some("sk-test".to_owned()),
        );

        assert_eq!(status.state, ProviderKeyState::Configured);
        assert_eq!(status.source, "env:OPENAI_API_KEY");
        assert_eq!(
            provider_key_status_label(&status),
            "using OPENAI_API_KEY for this session"
        );
    }

    #[test]
    fn provider_key_store_round_trips_without_exposing_value_in_status() -> Result<()> {
        let store = MemoryKeyStore::default();
        store.set_secret("openai", b"sk-secret-sentinel")?;

        let status = provider_key_status_with_store(&store, "openai", "OPENAI_API_KEY", None);
        let resolved =
            resolve_provider_api_key_with_store(&store, "openai", "OPENAI_API_KEY", None)?;

        assert_eq!(status.state, ProviderKeyState::Configured);
        assert_eq!(status.source, "secure:test keychain");
        assert!(!provider_key_status_label(&status).contains("sk-secret"));
        assert_eq!(resolved.value, "sk-secret-sentinel");
        assert_eq!(resolved.source, "secure:test keychain");
        Ok(())
    }

    #[test]
    fn provider_key_resolution_fails_loudly_without_fallback_when_store_unavailable() {
        let store = MemoryKeyStore {
            fail: Some("locked keychain"),
            ..MemoryKeyStore::default()
        };

        let error = resolve_provider_api_key_with_store(&store, "openai", "OPENAI_API_KEY", None)
            .unwrap_err()
            .to_string();

        assert!(error.contains("secure API-key storage is unavailable"));
        assert!(error.contains("no plaintext fallback was used"));
    }

    #[test]
    fn provider_key_resolution_reports_missing_key_next_action() {
        let store = MemoryKeyStore::default();

        let error =
            resolve_provider_api_key_with_store(&store, "anthropic", "ANTHROPIC_API_KEY", None)
                .unwrap_err()
                .to_string();

        assert!(error.contains("requires a saved API key"));
        assert!(error.contains("rocm config set-provider-key anthropic"));
        assert!(error.contains("ANTHROPIC_API_KEY"));
    }

    /// Regression for the "Cannot start a runtime from within a runtime" panic:
    /// the native store reaches a blocking `block_on` (Linux Secret Service via
    /// zbus). Accessing it from inside a live tokio runtime must NOT panic — the
    /// `with_native_entry` guard reroutes the blocking work onto a fresh thread.
    ///
    /// This validates the no-panic / graceful-`Err` contract, not successful
    /// retrieval: CI has no Secret Service, so the store returns `Ok(None)` (no
    /// entry) or an `Err` (unavailable). Either is fine; a panic is not. Pre-fix
    /// this aborted with the nested-runtime panic.
    #[test]
    fn native_store_access_inside_tokio_runtime_does_not_panic() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("build runtime");
        let outcome = rt.block_on(async {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // Read-only: never writes to a real keychain on dev machines.
                NativeProviderKeyStore.get_secret("anthropic")
            }))
        });
        assert!(
            outcome.is_ok(),
            "native key-store access panicked inside a tokio runtime (runtime-in-runtime)"
        );
    }
}
