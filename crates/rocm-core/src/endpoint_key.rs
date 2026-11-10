// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! The bearer key that guards a managed service's endpoint.
//!
//! The endpoint-API-key generator, its header-safety check, and the one guard
//! `rocm` and `rocmd` both apply before bringing a recorded service back up
//! (see [`ensure_public_service_has_endpoint_key`]).
//!
//! The key is drawn from a CSPRNG — deliberately never derived from
//! [`crate::service_identity`]'s timestamp-based, guessable
//! [`crate::generate_service_id`] — and the two meet only at the filesystem
//! layer, where a service's endpoint key is stored in a sidecar named after its
//! [`crate::ServiceId`].

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

    /// Whether a service that was running is being brought back. A fresh launch
    /// has nothing that stopped, so it is not told a stop dropped its key, and it
    /// is not told to "launch it again" when launching is what it just did.
    const fn brings_back_a_service(self) -> bool {
        match self {
            Self::Launch => false,
            Self::Restart | Self::Respawn => true,
        }
    }
}

/// The `rocm serve` command that gives a service with this recorded `host`,
/// `port` and requirement a key again. The command keeps everything the
/// service's endpoint depended on, so following it does not bring the service
/// back somewhere else: the port it recorded (always, since a port left out is
/// chosen from `rocm serve`'s own range, and anything that publishes the service
/// — a tailnet or proxy mapping — targets the port it had), the host where it is
/// not the default loopback one, the public-bind opt-in where the host is public,
/// and `--require-api-key` where the service asked for it. The model is the one
/// thing left as a placeholder: the guard does not know it, and it does not
/// decide where the endpoint is.
fn serve_command(host: &str, port: u16, requires_api_key: bool) -> String {
    let mut command = String::from("rocm serve <model>");
    if !crate::is_canonical_loopback_host(host) {
        command.push_str(" --host ");
        command.push_str(host);
    }
    command.push_str(" --port ");
    command.push_str(&port.to_string());
    if is_public_bind_host(host) {
        command.push_str(" --allow-public-bind");
    }
    if requires_api_key {
        command.push_str(" --require-api-key");
    }
    command
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
/// recovered, only issued again by a new launch. The command is built from the
/// recorded host, port and requirement ([`serve_command`]) and keeps what the
/// endpoint depended on: the recorded `--port` always (a launch that leaves it
/// out gets a port from `rocm serve`'s own range, which a mapping that publishes
/// the service does not follow), a host other than the default loopback one
/// carries `--host`, a public host carries `--allow-public-bind`, and a service
/// that asked for a key carries `--require-api-key`. What the start
/// changes is only the wording of the action and of why the key is missing: a
/// service being brought back lost its key when it stopped, and may still be
/// running, in which case a launch would reuse it and discard the new key, so the
/// refusal names the stop (`rocm services stop <id> --yes`, for `service_id`)
/// before the launch; a fresh launch stored a key file that is missing or
/// unusable and has started nothing, and names no stop.
pub fn ensure_public_service_has_endpoint_key(
    host: &str,
    key_present: bool,
    requires_api_key: bool,
    start: ServiceStart,
    service_id: &str,
    port: u16,
) -> anyhow::Result<()> {
    if key_present {
        return Ok(());
    }
    let lead = if requires_api_key {
        if start.brings_back_a_service() {
            "managed service was launched with `--require-api-key` but has no endpoint API key"
                .to_owned()
        } else {
            "this launch asks for `--require-api-key` but has no usable endpoint API key".to_owned()
        }
    } else if is_public_bind_host(host) {
        if start.brings_back_a_service() {
            format!(
                "managed service is bound to the public host `{host}` but has no endpoint API key"
            )
        } else {
            format!("this launch binds the public host `{host}` but has no usable endpoint API key")
        }
    } else {
        return Ok(());
    };
    let would_open = start.would_open();
    let command = serve_command(host, port, requires_api_key);
    let choose_key = "(add `--api-key <key>`, or set ROCM_SERVE_API_KEY, to choose the key \
                      instead of generating one)";
    if start.brings_back_a_service() {
        // Something outside this process may republish the port of a service that
        // asked for a key; that is a reason to refuse that only exists once the
        // service has been running.
        let republished = if requires_api_key {
            "Something outside this machine may still be publishing its port. "
        } else {
            ""
        };
        // A recorded service may still be running: its key file can go missing
        // without a confirmed stop. `rocm serve` reuses a running service for the
        // same engine and model and discards the key it generated for that
        // attempt, so launching again over it would leave the service as keyless
        // as it was and this refusal would come back at the next start. The stop
        // is what clears the condition, and is harmless for one that has stopped.
        let stop = format!(
            "rocm services stop {} --yes",
            crate::posix_shell_quote(service_id)
        );
        anyhow::bail!(
            "{lead}, so {would_open} without authentication. {republished}The key is dropped \
             when a service stops and cannot be recovered. Launching it again while it is \
             still running would reuse the running server and discard the new key, so stop it \
             first with `{stop}`, then launch it again with `{command}` {choose_key}."
        );
    }
    anyhow::bail!(
        "{lead}, so {would_open} without authentication. Nothing was started, and the key file \
         this launch stored is missing or unusable. Run `{command}` again {choose_key}."
    );
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

    const SERVICE_ID: &str = "vllm-qwen-1000";
    /// A port no automatic selection would pick (`rocm serve` chooses from its own
    /// range), as a service published behind a mapping would have.
    const PORT: u16 = 8000;

    fn refusal(host: &str, key_present: bool, requires: bool, start: ServiceStart) -> String {
        ensure_public_service_has_endpoint_key(host, key_present, requires, start, SERVICE_ID, PORT)
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

    /// The command each recorded host and requirement must be told to run: it
    /// keeps what the bind depended on, so following it does not bring the
    /// service back somewhere else. Spelled out here, not derived.
    const ADVISED: [(&str, bool, &str); 8] = [
        (
            "127.0.0.1",
            true,
            "rocm serve <model> --port 8000 --require-api-key",
        ),
        (
            "localhost",
            true,
            "rocm serve <model> --host localhost --port 8000 --require-api-key",
        ),
        (
            "::1",
            true,
            "rocm serve <model> --host ::1 --port 8000 --require-api-key",
        ),
        (
            "0.0.0.0",
            true,
            "rocm serve <model> --host 0.0.0.0 --port 8000 --allow-public-bind --require-api-key",
        ),
        (
            "gpu-box.example",
            true,
            "rocm serve <model> --host gpu-box.example --port 8000 --allow-public-bind \
             --require-api-key",
        ),
        (
            "0.0.0.0",
            false,
            "rocm serve <model> --host 0.0.0.0 --port 8000 --allow-public-bind",
        ),
        (
            "192.168.1.10",
            false,
            "rocm serve <model> --host 192.168.1.10 --port 8000 --allow-public-bind",
        ),
        (
            "LOCALHOST",
            false,
            "rocm serve <model> --host LOCALHOST --port 8000 --allow-public-bind",
        ),
    ];

    /// Whether `message` names `command` as the thing to run, followed by the way to
    /// choose the key: a service being brought back is told to "launch it again
    /// with" it, a fresh launch to "run" it "again".
    fn names_command(message: &str, command: &str, start: ServiceStart) -> bool {
        let tail = if start.brings_back_a_service() {
            "(add `--api-key <key>`"
        } else {
            "again (add `--api-key <key>`"
        };
        message.contains(&format!("`{command}` {tail}"))
    }

    #[test]
    fn the_command_a_refusal_names_keeps_the_host_the_opt_in_and_the_key_flag() {
        for start in STARTS {
            for (host, requires, command) in ADVISED {
                let message = refusal(host, false, requires, start);
                assert!(
                    names_command(&message, command, start),
                    "{host} requires={requires} {start:?}: {message}"
                );
            }
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
                    message.contains("`--require-api-key` but has no"),
                    "{host} {start:?}: {message}"
                );
                // It never blames the bind address, whatever the host is.
                assert!(!message.contains("public host"), "{message}");
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
                    message.contains(&format!("the public host `{host}` but has no")),
                    "{host} {start:?}: {message}"
                );
                // The remediation keeps the host and the public-bind opt-in, so
                // following it does not bring the service back on loopback.
                assert!(
                    names_command(
                        &message,
                        &format!(
                            "rocm serve <model> --host {host} --port 8000 --allow-public-bind"
                        ),
                        start
                    ),
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
                ensure_public_service_has_endpoint_key(host, false, false, start, SERVICE_ID, PORT)
                    .unwrap_or_else(|error| panic!("{host} must not require a key: {error:#}"));
            }
            for (host, requires) in [("0.0.0.0", false), ("127.0.0.1", true), ("0.0.0.0", true)] {
                ensure_public_service_has_endpoint_key(
                    host, true, requires, start, SERVICE_ID, PORT,
                )
                .unwrap_or_else(|error| panic!("{host} has its key: {error:#}"));
            }
        }
    }

    #[test]
    fn a_service_that_asked_for_a_key_on_a_public_host_is_told_about_both() {
        // The first rule wins, so a service that asked for a key is never told its
        // bind is the problem; and its remediation keeps the public host and the
        // opt-in as well as the flag it asked for, since dropping them would bring
        // it back on the default loopback host.
        for start in STARTS {
            let message = refusal("0.0.0.0", false, true, start);
            assert!(!message.contains("public host"), "{message}");
            assert!(
                message.contains(
                    "`rocm serve <model> --host 0.0.0.0 --port 8000 --allow-public-bind \
                     --require-api-key`"
                ),
                "{start:?}: {message}"
            );
        }
    }

    #[test]
    fn a_service_being_brought_back_and_a_fresh_launch_are_worded_from_the_start() {
        // `Restart` and `Respawn` differ from each other only in the clause naming
        // the action; a fresh `Launch` has nothing that stopped, so it says so
        // instead of repeating "the key is dropped when a service stops" and tells
        // the user to run the command again rather than to "launch it again". A
        // service being brought back may still be running, so it is told to stop
        // it first; a launch has no recorded service to stop.
        for (host, requires, _) in ADVISED {
            let restart = refusal(host, false, requires, ServiceStart::Restart);
            assert!(
                restart.contains(
                    "so stop it first with `rocm services stop vllm-qwen-1000 --yes`, then launch \
                     it again with `rocm serve <model>"
                ),
                "{host} requires={requires}: {restart}"
            );
            assert_eq!(
                refusal(host, false, requires, ServiceStart::Respawn),
                restart.replace(
                    "restarting it would reopen it",
                    "respawning it would reopen it"
                ),
                "{host} requires={requires}"
            );
            let launch = refusal(host, false, requires, ServiceStart::Launch);
            for stale in [
                "dropped when a service stops",
                "Launch it again",
                "still be publishing",
                "managed service",
                "rocm services stop",
                "still running",
            ] {
                assert!(
                    !launch.contains(stale),
                    "{host} requires={requires}: a launch must not say {stale:?}: {launch}"
                );
            }
            assert!(launch.contains("Nothing was started"), "{launch}");
            assert!(
                launch.contains("launching it would open it without authentication"),
                "{launch}"
            );
        }
        assert_eq!(
            refusal("127.0.0.1", false, true, ServiceStart::Restart),
            "managed service was launched with `--require-api-key` but has no endpoint API key, \
             so restarting it would reopen it without authentication. Something outside this \
             machine may still be publishing its port. The key is dropped when a service stops \
             and cannot be recovered. Launching it again while it is still running would reuse \
             the running server and discard the new key, so stop it first with `rocm services \
             stop vllm-qwen-1000 --yes`, then launch it again with `rocm serve <model> \
             --port 8000 --require-api-key` (add `--api-key <key>`, or set ROCM_SERVE_API_KEY, to choose the \
             key instead of generating one)."
        );
        assert_eq!(
            refusal("0.0.0.0", false, false, ServiceStart::Respawn),
            "managed service is bound to the public host `0.0.0.0` but has no endpoint API key, \
             so respawning it would reopen it without authentication. The key is dropped when a \
             service stops and cannot be recovered. Launching it again while it is still \
             running would reuse the running server and discard the new key, so stop it first \
             with `rocm services stop vllm-qwen-1000 --yes`, then launch it again with `rocm \
             serve <model> --host 0.0.0.0 --port 8000 --allow-public-bind` (add `--api-key \
             <key>`, or set ROCM_SERVE_API_KEY, to choose the key instead of generating one)."
        );
        assert_eq!(
            refusal("127.0.0.1", false, true, ServiceStart::Launch),
            "this launch asks for `--require-api-key` but has no usable endpoint API key, so \
             launching it would open it without authentication. Nothing was started, and the \
             key file this launch stored is missing or unusable. Run `rocm serve <model> \
             --port 8000 --require-api-key` again (add `--api-key <key>`, or set ROCM_SERVE_API_KEY, to \
             choose the key instead of generating one)."
        );
        assert_eq!(
            refusal("0.0.0.0", false, false, ServiceStart::Launch),
            "this launch binds the public host `0.0.0.0` but has no usable endpoint API key, so \
             launching it would open it without authentication. Nothing was started, and the \
             key file this launch stored is missing or unusable. Run `rocm serve <model> --host \
             0.0.0.0 --port 8000 --allow-public-bind` again (add `--api-key <key>`, or set \
             ROCM_SERVE_API_KEY, to choose the key instead of generating one)."
        );
    }

    #[test]
    fn the_command_a_refusal_names_carries_the_port_the_service_recorded() {
        // Whatever the host, the relaunch has to land on the port the service had:
        // a `rocm serve` that leaves `--port` out picks one from its own range, and
        // a tailnet or proxy mapping that publishes the service keeps targeting the
        // recorded one. So the recorded port is printed as it is, for every start,
        // host and requirement, and not a placeholder for the user to fill in.
        for port in [8000_u16, 11435, 1, 65535] {
            for start in STARTS {
                for (host, requires, _) in ADVISED {
                    let message = ensure_public_service_has_endpoint_key(
                        host, false, requires, start, SERVICE_ID, port,
                    )
                    .unwrap_err()
                    .to_string();
                    assert!(
                        message.contains(&format!(" --port {port} "))
                            || message.contains(&format!(" --port {port}`")),
                        "{host} requires={requires} {start:?} port={port}: {message}"
                    );
                    assert!(!message.contains("<port>"), "{message}");
                    // Only the recorded port appears, so a fixed one cannot hide behind it.
                    assert_eq!(message.matches("--port").count(), 1, "{message}");
                }
            }
        }
    }

    #[test]
    fn the_stop_it_names_quotes_a_service_id_that_is_not_one_word() {
        // The line is meant to be pasted into a shell, so an id with a space or a
        // quote must stay one word, as `rocm`'s own stop hints do.
        let message = ensure_public_service_has_endpoint_key(
            "0.0.0.0",
            false,
            false,
            ServiceStart::Restart,
            "odd id's",
            PORT,
        )
        .unwrap_err()
        .to_string();
        assert!(
            message.contains("`rocm services stop 'odd id'\\''s' --yes`"),
            "{message}"
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
