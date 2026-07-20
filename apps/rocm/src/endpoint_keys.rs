// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Secure storage for per-service inference-endpoint API keys.
//!
//! When `rocm serve` binds a public (non-loopback) interface it protects the
//! endpoint with an API key (see `resolve_endpoint_auth` in `main.rs`). The key
//! is stored in a 0600 file under the services directory, keyed by service id.
//!
//! A file — not the OS keychain — is deliberate: public binding is overwhelmingly
//! a *headless server* action (that is what `0.0.0.0` is for), and headless Linux
//! hosts routinely have no Secret Service / D-Bus session, so a keychain-backed
//! store would make `rocm serve --host 0.0.0.0` fail on its primary platform. An
//! owner-only (0600) file under `~/.rocm` satisfies "not persisted insecurely"
//! while working the same everywhere, and is the same shape (`--api-key-file`)
//! engines like llama-server already use.

use anyhow::{Context, Result};
use std::path::PathBuf;

use rocm_core::AppPaths;

/// Deterministic path of the 0600 endpoint key file for `service_id`.
pub(crate) fn endpoint_key_file_path(paths: &AppPaths, service_id: &str) -> PathBuf {
    paths
        .services_dir()
        .join(format!("{service_id}.endpoint-key"))
}

/// Persist `key` for `service_id`, creating the file 0600. Overwrites any
/// existing value.
pub(crate) fn store_endpoint_api_key(paths: &AppPaths, service_id: &str, key: &str) -> Result<()> {
    let path = endpoint_key_file_path(paths, service_id);
    crate::write_private_file_0600(&path, key.as_bytes())
        .with_context(|| format!("failed to write endpoint API key file {}", path.display()))
}

/// Fetch the stored key for `service_id`, or `None` when no key file exists (a
/// loopback service, or one launched before endpoint auth existed).
pub(crate) fn endpoint_api_key(paths: &AppPaths, service_id: &str) -> Option<String> {
    rocm_engine_protocol::endpoint_api_key_from_file(&endpoint_key_file_path(paths, service_id))
}

/// The key file path if it exists, for handing to the engine child via
/// `ROCM_SERVE_API_KEY_FILE`. `None` for loopback services with no stored key.
pub(crate) fn endpoint_key_file_if_present(paths: &AppPaths, service_id: &str) -> Option<PathBuf> {
    let path = endpoint_key_file_path(paths, service_id);
    path.exists().then_some(path)
}

/// Remove the stored key for `service_id`. Idempotent and best-effort: a missing
/// file is not an error, so this is safe to call unconditionally when any service
/// stops.
pub(crate) fn clear_endpoint_api_key(paths: &AppPaths, service_id: &str) {
    let _ = std::fs::remove_file(endpoint_key_file_path(paths, service_id));
}

#[cfg(test)]
mod tests {
    use super::*;
    use rocm_core::unix_time_millis;

    fn temp_paths(name: &str) -> (PathBuf, AppPaths) {
        let root = std::env::temp_dir().join(format!(
            "rocm-endpoint-key-{name}-{}-{}",
            std::process::id(),
            unix_time_millis()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("data").join("services")).unwrap();
        (
            root.clone(),
            AppPaths {
                config_dir: root.join("config"),
                data_dir: root.join("data"),
                cache_dir: root.join("cache"),
            },
        )
    }

    #[test]
    fn store_then_clear_round_trips_and_removes_the_file() {
        let (root, paths) = temp_paths("roundtrip");
        let service_id = "svc-1";
        assert_eq!(endpoint_api_key(&paths, service_id), None);
        assert_eq!(endpoint_key_file_if_present(&paths, service_id), None);

        store_endpoint_api_key(&paths, service_id, "secret-key").unwrap();
        assert_eq!(
            endpoint_api_key(&paths, service_id),
            Some("secret-key".to_owned())
        );
        assert!(endpoint_key_file_if_present(&paths, service_id).is_some());

        // Teardown must leave no plaintext key on disk. This is the whole point of
        // reusing this managed file for the llama-server fallback rather than a copy.
        clear_endpoint_api_key(&paths, service_id);
        assert_eq!(endpoint_api_key(&paths, service_id), None);
        assert_eq!(endpoint_key_file_if_present(&paths, service_id), None);
        assert!(!endpoint_key_file_path(&paths, service_id).exists());

        // Clearing again is a no-op, so any stop path can call it unconditionally.
        clear_endpoint_api_key(&paths, service_id);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn stored_key_file_is_owner_only_0600() {
        use std::os::unix::fs::PermissionsExt as _;
        let (root, paths) = temp_paths("mode");
        store_endpoint_api_key(&paths, "svc-mode", "secret-key").unwrap();
        let mode = std::fs::metadata(endpoint_key_file_path(&paths, "svc-mode"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600, "endpoint key file must be 0600");
        let _ = std::fs::remove_dir_all(&root);
    }
}
