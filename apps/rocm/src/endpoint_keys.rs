// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Secure storage for per-service inference-endpoint API keys.
//!
//! When `rocm serve` binds a public (non-loopback) interface it protects the
//! endpoint with an API key (see `resolve_endpoint_auth` in `service_launch/endpoint_auth.rs`).
//! The key is stored in an owner-only file under the services directory, keyed
//! by service id: mode 0600 on Unix, and on Windows a protected DACL granting
//! only the user who stored it, so the data directory's own ACL is not
//! inherited. Every store replaces the file with a freshly created owner-only
//! one, so a key file whose permissions were loosened is tightened again rather
//! than rewritten in place (see `write_owner_only_file_atomically` in
//! `rocm_core::atomic_write`).
//!
//! A file — not the OS keychain — is deliberate: public binding is overwhelmingly
//! a *headless server* action (that is what `0.0.0.0` is for), and headless Linux
//! hosts routinely have no Secret Service / D-Bus session, so a keychain-backed
//! store would make `rocm serve --host 0.0.0.0` fail on its primary platform. An
//! owner-only file under `~/.rocm` satisfies "not persisted insecurely"
//! while working the same everywhere, and is the same shape (`--api-key-file`)
//! engines like llama-server already use.

use anyhow::{Context, Result};
use std::path::PathBuf;

use rocm_core::AppPaths;

/// Deterministic path of the owner-only endpoint key file for `service_id`.
///
/// Delegates to the shared helper in `rocm-engine-protocol` so `rocmd` (which
/// needs the same path to re-thread `ROCM_SERVE_API_KEY_FILE` onto engine
/// children it spawns) derives an identical path without depending on this
/// crate-private module.
pub(crate) fn endpoint_key_file_path(paths: &AppPaths, service_id: &str) -> PathBuf {
    rocm_engine_protocol::endpoint_key_file_path(paths, service_id)
}

/// Persist `key` for `service_id` in a file only the current user can read
/// (0600 on Unix, a protected owner-only DACL on Windows). Replaces any
/// existing file, together with whatever permissions it had.
pub(crate) fn store_endpoint_api_key(paths: &AppPaths, service_id: &str, key: &str) -> Result<()> {
    let path = endpoint_key_file_path(paths, service_id);
    rocm_core::atomic_write::write_owner_only_file_atomically(&path, key.as_bytes())
        .with_context(|| format!("failed to write endpoint API key file {}", path.display()))
}

/// Fetch the stored key for `service_id`, or `None` when no key file exists (a
/// loopback service, or one launched before endpoint auth existed).
pub(crate) fn endpoint_api_key(paths: &AppPaths, service_id: &str) -> Option<String> {
    rocm_engine_protocol::endpoint_api_key_from_file(&endpoint_key_file_path(paths, service_id))
}

/// The key file path if it exists, for handing to the engine child via
/// `ROCM_SERVE_API_KEY_FILE`. `None` for loopback services with no stored key.
///
/// Delegates to the shared helper in `rocm-engine-protocol` (see
/// [`endpoint_key_file_path`]).
pub(crate) fn endpoint_key_file_if_present(paths: &AppPaths, service_id: &str) -> Option<PathBuf> {
    rocm_engine_protocol::endpoint_key_file_if_present(paths, service_id)
}

/// Remove the stored key for `service_id`, with any staged copy a store killed
/// before publishing left beside it. Idempotent and best-effort: a missing file
/// is not an error, and a file that cannot be removed is left without being
/// reported, so this is safe to call unconditionally when any service stops.
/// It does not exclude a concurrent store for the same service.
pub(crate) fn clear_endpoint_api_key(paths: &AppPaths, service_id: &str) {
    rocm_engine_protocol::remove_endpoint_key_files(paths, service_id);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_paths(name: &str) -> (PathBuf, AppPaths) {
        let root = std::env::temp_dir().join(format!(
            "rocm-endpoint-key-{name}-{}",
            crate::test_support::unique_suffix()
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

    /// A store killed between staging the key and publishing it leaves a
    /// second copy beside the key file; clearing the key takes that copy too.
    #[test]
    fn clearing_a_key_removes_a_staged_copy_a_killed_store_left() {
        let (root, paths) = temp_paths("staged-leftover");
        store_endpoint_api_key(&paths, "svc-staged", "secret-key").unwrap();
        let staged = rocm_core::atomic_write::stage_owner_only_file_unpublished(
            &endpoint_key_file_path(&paths, "svc-staged"),
            b"secret-key",
        )
        .unwrap();

        clear_endpoint_api_key(&paths, "svc-staged");
        let staged_exists = staged.exists();
        let key_exists = endpoint_key_file_path(&paths, "svc-staged").exists();
        let _ = std::fs::remove_dir_all(&root);
        assert!(!key_exists, "the key file must be removed");
        assert!(
            !staged_exists,
            "the staged copy of the key must be removed too"
        );
    }

    #[test]
    fn temp_paths_gives_each_call_its_own_root_even_for_one_label() {
        crate::test_support::assert_each_call_gets_its_own_root(|| temp_paths("same-label").0);
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

    /// Re-storing a key over a file someone loosened leaves it 0600 again,
    /// with the new key in it: the old mode is not kept (a restart re-stores the
    /// key over the file of the previous run).
    #[cfg(unix)]
    #[test]
    fn restoring_a_key_over_a_loosened_file_makes_it_0600_again() {
        use std::os::unix::fs::PermissionsExt as _;
        let (root, paths) = temp_paths("loosened");
        let path = endpoint_key_file_path(&paths, "svc-loose");
        std::fs::write(&path, "old-key").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();

        store_endpoint_api_key(&paths, "svc-loose", "new-key").unwrap();

        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        let stored = endpoint_api_key(&paths, "svc-loose");
        let _ = std::fs::remove_dir_all(&root);
        assert_eq!(mode, 0o600, "endpoint key file was left at {mode:o}");
        assert_eq!(stored, Some("new-key".to_owned()));
    }

    /// On Windows the stored key has a protected DACL granting only the user
    /// who stored it, whatever the data directory hands down.
    #[cfg(windows)]
    #[test]
    fn stored_key_file_has_an_owner_only_dacl() {
        let (root, paths) = temp_paths("dacl");
        store_endpoint_api_key(&paths, "svc-dacl", "secret-key").unwrap();
        let mismatch = rocm_core::owner_only_file::owner_only_dacl_mismatch(
            &endpoint_key_file_path(&paths, "svc-dacl"),
        );
        let _ = std::fs::remove_dir_all(&root);
        assert_eq!(
            mismatch.unwrap(),
            None,
            "the key file's DACL is not owner-only"
        );
    }
}
