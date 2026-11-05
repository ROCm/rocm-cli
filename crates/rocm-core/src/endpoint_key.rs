// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The bearer key that guards a managed service's endpoint.
//!
//! The endpoint-API-key generator, its header-safety check, and the one guard
//! `rocm` and `rocmd` both apply before bringing a recorded service back up
//! (see [`ensure_public_service_has_endpoint_key`]). Drawn from a
//! CSPRNG — deliberately never derived from [`crate::service_identity`]'s
//! timestamp-based, guessable [`crate::generate_service_id`] — the two meet
//! only at the filesystem layer, where a service's endpoint key is stored in a
//! sidecar named after its [`crate::ServiceId`].

/// Number of random alphanumeric characters in a generated endpoint API key.
/// 48 chars from a 62-symbol alphabet is ~285 bits of entropy — far beyond what
/// a bearer token guarding a network endpoint needs, with no padding characters
/// that would complicate copy/paste into client configs or shell env vars.
const ENDPOINT_API_KEY_LEN: usize = 48;

/// Generate a fresh, cryptographically-random API key for a public endpoint.
///
/// The value is URL-safe alphanumeric (`[A-Za-z0-9]`) so it can be dropped
/// verbatim into an `Authorization: Bearer` header, a client config file, or an
/// environment variable without escaping.
///
/// Drawn from `rand::rng()`, a CSPRNG seeded from the operating system;
/// deliberately *not* derived from `generate_service_id` (a timestamp-based,
/// guessable identifier — unsuitable as a secret).
pub fn generate_endpoint_api_key() -> String {
    use rand::RngExt;
    use rand::distr::Alphanumeric;

    rand::rng()
        .sample_iter(&Alphanumeric)
        .take(ENDPOINT_API_KEY_LEN)
        .map(char::from)
        .collect()
}

/// Return `true` if `key` contains a character that must never appear in an API
/// key used verbatim in an `Authorization: Bearer` header.
///
/// The CLI and engine adapters build those header lines by raw string
/// interpolation (`Authorization: Bearer {key}\r\n`), so an embedded CR or LF in
/// the key would inject additional header lines (HTTP header injection). A
/// control character has no legitimate place in a bearer token, so we reject the
/// whole class rather than only CR/LF. Callers apply this at input validation
/// (rejecting a supplied key) and defensively when reading the key file.
pub fn endpoint_api_key_has_forbidden_chars(key: &str) -> bool {
    key.chars().any(char::is_control)
}

/// Whether `host` names a public (non-loopback) bind, i.e. one that must be
/// protected by an endpoint API key.
///
/// Shared so `rocm` (which decides at `serve` time whether to mint a key) and
/// `rocmd` (which respawns recorded services during recovery) apply the same
/// policy to the same `ManagedServiceRecord::host` value, and so
/// [`ensure_public_service_has_endpoint_key`] can apply it itself.
///
/// Deliberately an exact match against the three loopback spellings the CLI
/// accepts, so anything unrecognized — an IPv6 loopback alias, a differently
/// cased `localhost`, a hostname that happens to resolve locally — is treated
/// as public and therefore *requires* a key. Misclassifying a loopback bind as
/// public costs a key nobody needed; the reverse would open an unauthenticated
/// network listener.
#[must_use]
pub fn is_public_bind_host(host: &str) -> bool {
    !matches!(host, "127.0.0.1" | "localhost" | "::1")
}

/// What is about to bring a recorded service up.
///
/// It only words the refusal in [`ensure_public_service_has_endpoint_key`] for
/// the action; the rules and the commands the refusal names do not depend on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceStart {
    /// A fresh `rocm serve` spawn.
    Launch,
    /// `rocm services restart`.
    Restart,
    /// The daemon's recovery supervisor putting a failed service back up.
    Respawn,
}

impl ServiceStart {
    /// The clause that finishes "so …" in a refusal.
    const fn would_open(self) -> &'static str {
        match self {
            Self::Launch => "launching it would open it",
            Self::Restart => "restarting it would reopen it",
            Self::Respawn => "respawning it would reopen it",
        }
    }
}

/// Refuse to bring a managed service up when it needs an endpoint API key and
/// has none, so a respawn cannot reopen a protected endpoint anonymously.
///
/// `serve()` applies the loopback-vs-public policy once and persists the
/// resulting key. Every later spawn — `rocm services restart`, and `rocmd`'s
/// recovery supervisor — reads that key file back and would otherwise treat
/// "no key file" as "no auth wanted", silently downgrading a protected public
/// endpoint to an open one. A stop deletes the key file and `rocm services
/// restart` accepts a stopped service id, so the gap is reachable through
/// ordinary commands. The real invariant is a property of the *host*, not of
/// the file: a non-loopback bind must always be authenticated.
///
/// Two ways a service can need a key. A public bind is the obvious one. The
/// other is a service that asked for auth on a loopback bind
/// (`requires_api_key`), because something outside this process republishes the
/// port — a tailnet publish survives a reboot, let alone a restart, so the bind
/// address can no longer be trusted to answer the question on its own. The
/// requirement is recorded on the service precisely so a respawn can honour it.
/// The second is checked first, so a service that asked for auth is never told
/// its public bind is the problem.
///
/// `key_present` means a *usable* key: the engine adapters enforce nothing when
/// the file is empty or malformed. It is a plain `bool` rather than a path so
/// every branch is unit-testable without touching the filesystem.
///
/// Both refusals name the command that restores the service — `rocm serve
/// <model> …` — because the key is dropped when a service stops and cannot be
/// recovered, only issued again by a new launch.
pub fn ensure_public_service_has_endpoint_key(
    host: &str,
    key_present: bool,
    requires_api_key: bool,
    start: ServiceStart,
) -> anyhow::Result<()> {
    if key_present {
        return Ok(());
    }
    let would_open = start.would_open();
    if requires_api_key {
        anyhow::bail!(
            "managed service was launched with `--require-api-key` but has no endpoint API key, \
             so {would_open} without authentication. Something outside this machine may still be \
             publishing its port. The key is dropped when a service stops and cannot be \
             recovered. Launch it again with `rocm serve <model> --require-api-key` (add \
             `--api-key <key>`, or set ROCM_SERVE_API_KEY, to choose the key instead of \
             generating one)."
        );
    }
    if is_public_bind_host(host) {
        anyhow::bail!(
            "managed service is bound to the public host `{host}` but has no endpoint API key, \
             so {would_open} without authentication. The key is dropped when a service stops and \
             cannot be recovered. Launch it again with `rocm serve <model> --host {host} \
             --allow-public-bind` (add `--api-key <key>`, or set ROCM_SERVE_API_KEY, to choose \
             the key instead of generating one)."
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_endpoint_api_key_is_random_and_alphanumeric() {
        let key = generate_endpoint_api_key();
        assert_eq!(key.len(), ENDPOINT_API_KEY_LEN);
        // The message deliberately omits the key: it is a credential, and a
        // failing assertion's message ends up in CI logs.
        assert!(
            key.chars().all(|c| c.is_ascii_alphanumeric()),
            "key must be URL-safe alphanumeric"
        );
        // Two draws must differ — a constant key would be catastrophic for auth.
        assert_ne!(generate_endpoint_api_key(), generate_endpoint_api_key());
        // A freshly generated key must itself pass the header-safety predicate.
        assert!(!endpoint_api_key_has_forbidden_chars(
            &generate_endpoint_api_key()
        ));
    }

    const STARTS: [ServiceStart; 3] = [
        ServiceStart::Launch,
        ServiceStart::Restart,
        ServiceStart::Respawn,
    ];

    fn refusal(host: &str, key_present: bool, requires: bool, start: ServiceStart) -> String {
        ensure_public_service_has_endpoint_key(host, key_present, requires, start)
            .expect_err("this state must be refused")
            .to_string()
    }

    #[test]
    fn is_public_bind_host_recognizes_only_the_accepted_loopback_spellings() {
        for host in ["127.0.0.1", "localhost", "::1"] {
            assert!(!is_public_bind_host(host), "{host} must be loopback");
        }
        for host in [
            "0.0.0.0",
            "::",
            "192.168.1.10",
            "example.test",
            // Conservative on purpose: an unrecognized spelling of a loopback
            // address is classified public, so it requires a key rather than
            // opening an unauthenticated listener.
            "LOCALHOST",
            "127.0.0.2",
            "[::1]",
        ] {
            assert!(is_public_bind_host(host), "{host} must be public");
        }
    }

    #[test]
    fn a_service_that_requires_a_key_is_refused_without_one_on_any_host() {
        // The bind address is not the whole story: a loopback service that
        // something republishes is only private until it is respawned keyless.
        for start in STARTS {
            for host in [
                "127.0.0.1",
                "localhost",
                "::1",
                "0.0.0.0",
                "gpu-box.example",
            ] {
                let message = refusal(host, false, true, start);
                assert!(
                    message.contains("launched with `--require-api-key`"),
                    "{host} {start:?}: {message}"
                );
                assert!(
                    message.contains("Launch it again with `rocm serve <model> --require-api-key`"),
                    "{host} {start:?}: {message}"
                );
                // It never blames the bind address, whatever the host is.
                assert!(!message.contains("public host"), "{message}");
                assert!(!message.contains("--allow-public-bind"), "{message}");
            }
        }
    }

    #[test]
    fn a_public_bind_is_refused_without_a_key_even_if_it_never_asked_for_one() {
        for start in STARTS {
            for host in [
                "0.0.0.0",
                "::",
                "192.168.1.10",
                "gpu-box.example",
                "LOCALHOST",
            ] {
                let message = refusal(host, false, false, start);
                assert!(
                    message.contains(&format!("bound to the public host `{host}`")),
                    "{host} {start:?}: {message}"
                );
                // The remediation keeps the host and the public-bind opt-in, so
                // following it does not bring the service back on loopback.
                assert!(
                    message.contains(&format!(
                        "Launch it again with `rocm serve <model> --host {host} --allow-public-bind`"
                    )),
                    "{host} {start:?}: {message}"
                );
                // A service that never passed the flag is not told it did.
                assert!(!message.contains("`--require-api-key`"), "{message}");
            }
        }
    }

    #[test]
    fn a_service_that_needs_no_key_or_has_one_is_not_refused() {
        for start in STARTS {
            for host in ["127.0.0.1", "localhost", "::1"] {
                ensure_public_service_has_endpoint_key(host, false, false, start)
                    .unwrap_or_else(|error| panic!("{host} must not require a key: {error:#}"));
            }
            for (host, requires) in [("0.0.0.0", false), ("127.0.0.1", true), ("0.0.0.0", true)] {
                ensure_public_service_has_endpoint_key(host, true, requires, start)
                    .unwrap_or_else(|error| panic!("{host} has its key: {error:#}"));
            }
        }
    }

    #[test]
    fn a_service_that_asked_for_a_key_on_a_public_host_is_told_about_the_key_requirement() {
        // The first rule wins, so the remediation does not drop what the user
        // asked for.
        for start in STARTS {
            assert_eq!(
                refusal("0.0.0.0", false, true, start),
                refusal("127.0.0.1", false, true, start)
            );
        }
    }

    #[test]
    fn the_start_changes_only_what_the_refusal_says_the_action_would_do() {
        // The same text is shown by `rocm services restart`, a fresh `rocm serve`
        // spawn and the daemon's recovery: only the action's own words differ, so
        // the rules and the commands they name cannot drift apart per caller.
        for (host, requires) in [("127.0.0.1", true), ("0.0.0.0", false)] {
            let restart = refusal(host, false, requires, ServiceStart::Restart);
            for (start, clause) in [
                (ServiceStart::Launch, "launching it would open it"),
                (ServiceStart::Respawn, "respawning it would reopen it"),
            ] {
                assert_eq!(
                    refusal(host, false, requires, start),
                    restart.replace("restarting it would reopen it", clause),
                    "{host} {start:?}"
                );
            }
        }
        assert_eq!(
            refusal("127.0.0.1", false, true, ServiceStart::Restart),
            "managed service was launched with `--require-api-key` but has no endpoint API key, \
             so restarting it would reopen it without authentication. Something outside this \
             machine may still be publishing its port. The key is dropped when a service stops \
             and cannot be recovered. Launch it again with `rocm serve <model> \
             --require-api-key` (add `--api-key <key>`, or set ROCM_SERVE_API_KEY, to choose the \
             key instead of generating one)."
        );
        assert_eq!(
            refusal("0.0.0.0", false, false, ServiceStart::Respawn),
            "managed service is bound to the public host `0.0.0.0` but has no endpoint API key, \
             so respawning it would reopen it without authentication. The key is dropped when a \
             service stops and cannot be recovered. Launch it again with `rocm serve <model> \
             --host 0.0.0.0 --allow-public-bind` (add `--api-key <key>`, or set \
             ROCM_SERVE_API_KEY, to choose the key instead of generating one)."
        );
    }

    #[test]
    fn endpoint_api_key_has_forbidden_chars_flags_control_chars() {
        // Control characters (notably CR/LF) enable header injection when the key
        // is interpolated into a raw `Authorization: Bearer` line.
        for bad in ["key\r\ninject", "key\nother", "tab\there", "nul\0byte"] {
            assert!(
                endpoint_api_key_has_forbidden_chars(bad),
                "should reject {bad:?}"
            );
        }
        // Ordinary printable keys are accepted.
        for good in ["my-key", "AbC123._~-", "sk-proj-abcDEF0123456789"] {
            assert!(
                !endpoint_api_key_has_forbidden_chars(good),
                "should accept {good:?}"
            );
        }
    }
}
