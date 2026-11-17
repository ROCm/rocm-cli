// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Bind-host validation and the endpoint API key checks that gate a public
//! bind: [`validate_bind_host`], [`resolve_endpoint_auth`] and the guards that
//! keep a respawn from reopening a service without its key.

use crate::endpoint_keys;
use anyhow::{Result, bail};
use rocm_core::AppPaths;

pub(crate) fn validate_bind_host(host: &str, allow_public_bind: bool) -> Result<()> {
    if !is_loopback_host(host) && !allow_public_bind {
        bail!(
            "`rocm serve --host {host}` is not loopback; pass `--allow-public-bind` before binding a non-local interface"
        );
    }
    Ok(())
}

/// Inverse of [`rocm_engine_protocol::is_public_bind_host`], which owns the
/// policy so `rocmd` classifies a recorded `host` identically when it respawns
/// the service.
pub(crate) fn is_loopback_host(host: &str) -> bool {
    !rocm_engine_protocol::is_public_bind_host(host)
}

/// Resolve the API key that will guard this endpoint, applying the
/// loopback-vs-public policy for `rocm serve`.
///
/// - **Loopback host** → `None`: local serving stays credential-free (the
///   unchanged default). Any key supplied for a loopback bind is ignored and
///   nothing is persisted — loopback needs no auth.
/// - **Public host** → `Some(key)`: use the user-supplied key when present,
///   otherwise generate a strong random one so a public endpoint can never come
///   up anonymous. An empty/whitespace supplied key is rejected rather than
///   silently treated as "no auth".
/// - **`required`** → treat a loopback bind as public for this purpose.
///
/// That last case exists because "loopback" is a statement about the bind
/// address, not about who can reach the port. Publishing the port onto a
/// tailnet, proxying it, or mapping it out of a container all leave the bind
/// loopback while widening the audience — and the policy above would then hand
/// out an unauthenticated endpoint. Whoever widens the reach is responsible for
/// asking for the credential, so this is an explicit flag rather than a guess.
pub(crate) fn resolve_endpoint_auth(
    host: &str,
    supplied: Option<&str>,
    required: bool,
) -> Result<Option<String>> {
    if is_loopback_host(host) && !required {
        return Ok(None);
    }
    match supplied {
        Some(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                bail!(
                    "`rocm serve --api-key` (or ROCM_SERVE_API_KEY) was empty; a public \
                     endpoint must be protected by a non-empty API key"
                );
            }
            // The key is later interpolated verbatim into raw `Authorization:
            // Bearer {key}\r\n` header lines; reject a control character (e.g. an
            // embedded CR/LF) here so a crafted key cannot inject extra headers.
            if rocm_core::endpoint_api_key_has_forbidden_chars(trimmed) {
                bail!(
                    "`rocm serve --api-key` (or ROCM_SERVE_API_KEY) contained a control \
                     character such as a carriage return or newline; an endpoint API key \
                     must be a single line of printable characters"
                );
            }
            Ok(Some(trimmed.to_owned()))
        }
        None => Ok(Some(rocm_core::generate_endpoint_api_key())),
    }
}

/// When an equivalent managed service is already running, or the launch fails
/// before any record exists, the endpoint key serve() freshly stored for this
/// attempt is unused — drop it so it is not orphaned in storage. The already-running service keeps its own key.
/// Best-effort and idempotent; a loopback attempt (`freshly_stored == None`)
/// is a no-op.
pub(crate) fn drop_orphaned_endpoint_key_on_already_running(
    paths: &AppPaths,
    service_id: &str,
    freshly_stored: Option<&str>,
) {
    if freshly_stored.is_some() {
        endpoint_keys::clear_endpoint_api_key(paths, service_id);
    }
}

/// Reject engine/platform combinations that cannot enforce a public endpoint's
/// API key, so a public bind fails closed instead of coming up unauthenticated.
///
/// The one such case today: Windows managed Lemonade. Its server reads the
/// value-typed `LEMONADE_API_KEY` env var, but the Windows detached-spawn
/// primitive only carries path-valued env overrides, so the key never reaches it.
/// vLLM enforces auth on every platform (`VLLM_API_KEY`), and loopback binds
/// (`public_bind == false`) need no key — both pass. `is_windows` is a parameter
/// so both branches are unit-testable off-Windows.
pub(crate) fn ensure_public_bind_engine_supported(
    engine: &str,
    public_bind: bool,
    is_windows: bool,
) -> Result<()> {
    if public_bind && is_windows && engine == "lemonade" {
        bail!(
            "public binding with the lemonade engine is not supported on Windows: the endpoint \
             API key cannot be enforced there. Use `--engine vllm`, or bind a loopback host \
             (the default 127.0.0.1)."
        );
    }
    Ok(())
}

/// Refuse to (re)spawn a managed service recorded on a public host when its
/// endpoint API key is gone, so a respawn cannot reopen the endpoint anonymously.
///
/// `serve()` applies the loopback-vs-public policy once, via
/// [`resolve_endpoint_auth`], and persists the resulting key. Every later spawn
/// — `rocm services restart`, and `rocmd`'s recovery supervisor — reads that key
/// file back and would otherwise treat "no key file" as "no auth wanted",
/// silently downgrading a protected public endpoint to an open one. The real
/// invariant is a property of the *host*, not of the file: a non-loopback bind
/// must always be authenticated.
///
/// The gap is reachable through ordinary commands, because a stop deletes the
/// key file and `rocm services restart` accepts a stopped service id (its help
/// points at `rocm services list --all`).
///
/// `key_present` is a plain `bool` rather than a path so both branches are
/// unit-testable without touching the filesystem, mirroring `is_windows` in
/// [`ensure_public_bind_engine_supported`].
pub(crate) fn ensure_public_service_has_endpoint_key(
    host: &str,
    key_present: bool,
    requires_api_key: bool,
) -> Result<()> {
    // Two ways a service can need a key. A public bind is the obvious one. The
    // other is a service that asked for auth on a loopback bind, because
    // something outside this process republishes the port — a tailnet publish
    // survives a reboot, let alone a restart, so "loopback" stops meaning
    // "only this machine" and the bind address can no longer be trusted to
    // answer the question on its own.
    if requires_api_key && !key_present {
        bail!(
            "managed service was launched with `--require-api-key` but has no endpoint API key, \
             so restarting it would reopen it without authentication. Something outside this \
             machine may still be publishing its port. The key is dropped when a service stops \
             and cannot be recovered. Launch it again with \
             `rocm serve --require-api-key` (add `--api-key <key>`, or set ROCM_SERVE_API_KEY, \
             to choose the key instead of generating one)."
        );
    }
    if rocm_engine_protocol::is_public_bind_host(host) && !key_present {
        bail!(
            "managed service is bound to the public host `{host}` but has no endpoint API key, \
             so restarting it would reopen it without authentication. The key is dropped when a \
             service stops and cannot be recovered. Launch it again with \
             `rocm serve --host {host} --port <port> --allow-public-bind` (add `--api-key <key>`, or set \
             ROCM_SERVE_API_KEY, to choose the key instead of generating one)."
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::test_paths;
    use std::fs;

    #[test]
    fn serve_bind_validation_requires_public_ack() {
        validate_bind_host("127.0.0.1", false).unwrap();
        validate_bind_host("localhost", false).unwrap();
        validate_bind_host("::1", false).unwrap();
        let error = validate_bind_host("0.0.0.0", false).unwrap_err();
        assert!(
            error.to_string().contains("--allow-public-bind"),
            "{error:#}"
        );
        validate_bind_host("0.0.0.0", true).unwrap();
    }

    #[test]
    fn resolve_endpoint_auth_loopback_stays_credential_free() {
        // Loopback binds never require auth, even if a key is supplied.
        for host in ["127.0.0.1", "localhost", "::1"] {
            assert_eq!(resolve_endpoint_auth(host, None, false).unwrap(), None);
            assert_eq!(
                resolve_endpoint_auth(host, Some("ignored"), false).unwrap(),
                None
            );
        }
    }

    #[test]
    fn resolve_endpoint_auth_loopback_can_be_required_when_something_republishes_it() {
        // "Loopback" describes the bind address, not who can reach the port. A
        // tailnet publish, a proxy, or a container port map all leave the bind
        // loopback while widening the audience, and the default policy would
        // hand out an unauthenticated endpoint. Whoever widens the reach asks
        // for the credential explicitly.
        for host in ["127.0.0.1", "localhost", "::1"] {
            let generated = resolve_endpoint_auth(host, None, true)
                .unwrap()
                .expect("a required key must be generated, not skipped");
            assert!(!generated.trim().is_empty());

            assert_eq!(
                resolve_endpoint_auth(host, Some("supplied-key"), true).unwrap(),
                Some("supplied-key".to_owned()),
                "a supplied key must be honoured rather than ignored as it is by default"
            );
        }

        // The same validation a public bind gets: an empty key is a refusal, not
        // a silent downgrade to no auth.
        assert!(resolve_endpoint_auth("127.0.0.1", Some("  "), true).is_err());
    }

    #[test]
    fn resolve_endpoint_auth_public_uses_supplied_key_trimmed() {
        let key = resolve_endpoint_auth("0.0.0.0", Some("  my-key  "), false)
            .unwrap()
            .expect("public bind must have a key");
        assert_eq!(key, "my-key");
    }

    #[test]
    fn resolve_endpoint_auth_public_generates_key_when_absent() {
        let key = resolve_endpoint_auth("0.0.0.0", None, false)
            .unwrap()
            .expect("public bind must generate a key");
        assert_eq!(key.len(), 48);
        assert!(key.chars().all(|c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn resolve_endpoint_auth_public_rejects_empty_supplied_key() {
        let error = resolve_endpoint_auth("0.0.0.0", Some("   "), false).unwrap_err();
        assert!(error.to_string().contains("non-empty"), "{error:#}");
    }

    #[test]
    fn resolve_endpoint_auth_public_rejects_embedded_crlf() {
        // A supplied key survives `trim()` with embedded CR/LF intact and would
        // otherwise be interpolated into a raw `Authorization: Bearer` header,
        // injecting an extra header line. It must be rejected at input validation.
        for supplied in [
            "good-key\r\nX-Injected: value",
            "good-key\nmore",
            "line\rreturn",
        ] {
            let error = resolve_endpoint_auth("0.0.0.0", Some(supplied), false).unwrap_err();
            assert!(error.to_string().contains("control character"), "{error:#}");
        }
    }

    #[test]
    fn drop_orphaned_endpoint_key_on_already_running_clears_stored_key() {
        let (root, paths) = test_paths("drop-orphaned-key-stored");
        let service_id = "svc-orphaned";
        endpoint_keys::store_endpoint_api_key(&paths, service_id, "secret-key").unwrap();

        drop_orphaned_endpoint_key_on_already_running(&paths, service_id, Some("secret-key"));

        assert_eq!(endpoint_keys::endpoint_api_key(&paths, service_id), None);
        assert!(!endpoint_keys::endpoint_key_file_path(&paths, service_id).exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn drop_orphaned_endpoint_key_on_already_running_is_noop_for_loopback() {
        // A loopback attempt never stores a key (`freshly_stored == None`), so the
        // helper must not panic or error, and no file must appear.
        let (root, paths) = test_paths("drop-orphaned-key-loopback");
        let service_id = "svc-loopback";

        drop_orphaned_endpoint_key_on_already_running(&paths, service_id, None);

        assert_eq!(endpoint_keys::endpoint_api_key(&paths, service_id), None);
        assert!(!endpoint_keys::endpoint_key_file_path(&paths, service_id).exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn public_bind_fails_closed_for_windows_lemonade_only() {
        // Windows + Lemonade + public bind: refuse (cannot enforce the key).
        let error = ensure_public_bind_engine_supported("lemonade", true, true).unwrap_err();
        assert!(error.to_string().contains("lemonade"), "{error:#}");
        // Every other combination is allowed:
        ensure_public_bind_engine_supported("vllm", true, true).unwrap(); // vLLM enforces auth on Windows
        ensure_public_bind_engine_supported("lemonade", true, false).unwrap(); // non-Windows
        ensure_public_bind_engine_supported("lemonade", false, true).unwrap(); // loopback needs no key
    }

    #[test]
    fn a_public_bind_is_refused_with_the_command_that_restores_it() {
        // `resolve_endpoint_auth` mints a key for every non-loopback bind whether
        // or not auth was demanded, so deriving `requires_api_key` from key-file
        // presence marked ordinary public binds as having asked for it. The guard
        // tests that field first, so those services were refused with a message
        // naming a flag they never passed and a relaunch command that drops
        // `--allow-public-bind` — coming back on loopback instead.
        //
        // A plain `--host 0.0.0.0 --allow-public-bind` launch: key file present,
        // `--require-api-key` never passed.
        let error = ensure_public_service_has_endpoint_key("0.0.0.0", false, false)
            .expect_err("a public bind with no key must be refused");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("--allow-public-bind"),
            "the refusal must name the command that restores the public bind: {rendered}"
        );
        assert!(
            !rendered.contains("--require-api-key"),
            "a service that never passed the flag must not be told it did: {rendered}"
        );

        // And the loopback-with-forced-auth case still reports its own reason.
        let error = ensure_public_service_has_endpoint_key("127.0.0.1", false, true)
            .expect_err("a service that demanded auth must not come back without it");
        assert!(
            format!("{error:#}").contains("--require-api-key"),
            "{error:#}"
        );
    }

    #[test]
    fn respawn_fails_closed_for_a_public_service_whose_key_is_gone() {
        // A stop deletes the key file, so a later restart of a public service
        // would otherwise respawn it with no auth at all.
        let error = ensure_public_service_has_endpoint_key("0.0.0.0", false, false).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("0.0.0.0"), "{error:#}");
        assert!(message.contains("without authentication"), "{error:#}");
        // Actionable: name the command that mints a fresh key.
        assert!(message.contains("--allow-public-bind"), "{error:#}");

        // A public service that still has its key restarts normally.
        ensure_public_service_has_endpoint_key("0.0.0.0", true, false).unwrap();
    }

    #[test]
    fn respawn_allows_loopback_services_without_an_endpoint_key() {
        // Loopback stays credential-free, so every accepted spelling must pass
        // the guard with no key present — when nothing asked for auth.
        for host in ["127.0.0.1", "localhost", "::1"] {
            ensure_public_service_has_endpoint_key(host, false, false)
                .unwrap_or_else(|error| panic!("{host} must not require a key: {error:#}"));
        }
    }

    #[test]
    fn respawn_refuses_a_loopback_service_that_was_launched_with_a_key() {
        // The hole this closes: a loopback bind that something else republishes
        // — a tailnet publish, a proxy, a container port map. The publish
        // outlives the process, so a restart after the key was dropped would
        // reopen a reachable endpoint with no authentication, and the bind
        // address gives the guard no way to notice.
        for host in ["127.0.0.1", "localhost", "::1"] {
            let error = ensure_public_service_has_endpoint_key(host, false, true)
                .expect_err("a service launched with a key must not restart without one");
            let message = format!("{error:#}");
            assert!(message.contains("without authentication"), "{message}");
            assert!(message.contains("--require-api-key"), "{message}");
        }

        // With its key still present it restarts normally.
        ensure_public_service_has_endpoint_key("127.0.0.1", true, true).unwrap();
    }
}
