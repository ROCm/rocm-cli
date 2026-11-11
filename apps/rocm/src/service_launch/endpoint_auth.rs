// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Bind-host validation and the endpoint API key checks that gate a public
//! bind: [`validate_bind_host`], [`resolve_endpoint_auth`] and the guards around
//! them. The guard that keeps a respawn from reopening a service without its key
//! is `rocm_core::ensure_public_service_has_endpoint_key`, shared with `rocmd`.

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

/// Inverse of [`rocm_core::is_public_bind_host`], which owns the policy so
/// `rocmd` classifies a recorded `host` identically when it respawns the
/// service.
pub(crate) fn is_loopback_host(host: &str) -> bool {
    !rocm_core::is_public_bind_host(host)
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

    /// The `rocm serve …` command a refusal tells the user to run: the backtick
    /// span that starts with `rocm serve`, split into words.
    fn advised_serve_command(message: &str) -> Vec<String> {
        let spans: Vec<&str> = message
            .split('`')
            .skip(1)
            .step_by(2)
            .filter(|span| span.starts_with("rocm serve"))
            .collect();
        assert_eq!(
            spans.len(),
            1,
            "the refusal must name exactly one `rocm serve` command: {message}"
        );
        crate::advised_commands::split_words(spans[0])
    }

    /// What the real clap parse makes of an advised `rocm serve` command.
    struct ServeAdvice {
        // Read only by the follower that goes through the spawn, which is unix-only
        // (it needs the spawn interceptor).
        #[cfg(not(windows))]
        model: String,
        host: String,
        port: Option<u16>,
        allow_public_bind: bool,
        require_api_key: bool,
        api_key: Option<String>,
    }

    fn parse_serve_advice(advised: &[String]) -> ServeAdvice {
        use crate::{Cli, Command, cli_command};
        use clap::FromArgMatches as _;

        let matches = cli_command()
            .try_get_matches_from(advised)
            .unwrap_or_else(|error| panic!("{advised:?} does not parse: {error}"));
        let Some(Command::Serve {
            #[cfg(not(windows))]
            model,
            host,
            port,
            allow_public_bind,
            require_api_key,
            api_key,
            ..
        }) = Cli::from_arg_matches(&matches)
            .unwrap_or_else(|error| panic!("{advised:?} does not parse: {error}"))
            .command
        else {
            panic!("{advised:?} is not a `rocm serve` invocation");
        };
        ServeAdvice {
            #[cfg(not(windows))]
            model,
            host,
            port,
            allow_public_bind,
            require_api_key,
            api_key,
        }
    }

    /// The `rocm services stop …` command a refusal tells the user to run: the
    /// backtick span that starts with it, split into words.
    #[cfg(not(windows))]
    fn advised_stop_command(message: &str) -> Vec<String> {
        let spans: Vec<&str> = message
            .split('`')
            .skip(1)
            .step_by(2)
            .filter(|span| span.starts_with("rocm services stop"))
            .collect();
        assert_eq!(
            spans.len(),
            1,
            "the refusal must name exactly one `rocm services stop` command: {message}"
        );
        crate::advised_commands::split_words(spans[0])
    }

    /// Run what `rocm serve` does with `advised` before it spawns anything — the
    /// real clap parse, `validate_bind_host`, `resolve_endpoint_auth`, and the
    /// key file `serve()` stores — and return the guard's answer for the service
    /// that launch would record. Nothing is spawned.
    ///
    /// The key is stored under `service_id`, the id the launch would record, so
    /// this follows a command whose service is not already recorded: a fresh
    /// launch. A recorded service that is still running is followed by
    /// [`follow_serve_through_the_spawn`], where `serve` stores the key under a
    /// new id and the spawn decides what becomes of it.
    fn guard_after_following(
        advised: &[String],
        paths: &AppPaths,
        service_id: &str,
        start: rocm_core::ServiceStart,
        service: (&str, bool, u16),
    ) -> Result<()> {
        let ServeAdvice {
            host,
            port,
            allow_public_bind,
            require_api_key,
            api_key,
            ..
        } = parse_serve_advice(advised);
        // Following the advice brings the service back where it was: on the host
        // and port it recorded, asking for the key again if it had asked for one. A
        // command that cleared the guard by quietly binding somewhere else (the
        // default loopback host, or a port `rocm serve` chooses itself, which a
        // mapping that publishes the service does not follow) would not be advice
        // that restores the service.
        let (recorded_host, recorded_requires_api_key, recorded_port) = service;
        assert_eq!(host, recorded_host, "{advised:?} binds another host");
        assert_eq!(
            port,
            Some(recorded_port),
            "{advised:?} does not bind the port the service had"
        );
        assert_eq!(
            require_api_key, recorded_requires_api_key,
            "{advised:?} does not carry the service's key requirement"
        );
        validate_bind_host(&host, allow_public_bind)?;
        // `serve()` refuses automatic port selection on any host but the
        // canonical loopback one, so advice for a public host without a port
        // would be refused here as it is there.
        crate::serve_port::validate_port_request(
            &host,
            crate::serve_port::PortRequest::from_flag(port),
        )?;
        let key = resolve_endpoint_auth(&host, api_key.as_deref(), require_api_key)?;
        if let Some(key) = key.as_deref() {
            endpoint_keys::store_endpoint_api_key(paths, service_id, key)?;
        }
        // The expression `spawn_managed_engine_child` computes: a *usable* key.
        let key_present = endpoint_keys::endpoint_key_file_if_present(paths, service_id)
            .is_some_and(|path| rocm_engine_protocol::endpoint_api_key_from_file(&path).is_some());
        rocm_core::ensure_public_service_has_endpoint_key(
            &host,
            key_present,
            require_api_key,
            start,
            service_id,
            recorded_port,
        )
    }

    /// What `rocm serve` does with `advised` when it gets as far as the spawn: the
    /// same front half, then the real `spawn_managed_engine_child` under a *new*
    /// service id with the key stored under that id, and, as `serve` does when
    /// the spawn finds an equivalent service already running, the new key
    /// dropped. Returns what the spawn decided, and whether the new key was still
    /// on disk once the spawn had decided.
    #[cfg(not(windows))]
    fn follow_serve_through_the_spawn(
        advised: &[String],
        paths: &AppPaths,
        new_service_id: &str,
        resolve: &rocm_engine_protocol::ResolveModelResponse,
    ) -> (Result<super::super::spawn::ManagedSpawn>, bool) {
        let ServeAdvice {
            model,
            host,
            port,
            allow_public_bind,
            require_api_key,
            api_key,
        } = parse_serve_advice(advised);
        validate_bind_host(&host, allow_public_bind).unwrap();
        let port_request = crate::serve_port::PortRequest::from_flag(port);
        crate::serve_port::validate_port_request(&host, port_request).unwrap();
        let key = resolve_endpoint_auth(&host, api_key.as_deref(), require_api_key).unwrap();
        if let Some(key) = key.as_deref() {
            endpoint_keys::store_endpoint_api_key(paths, new_service_id, key).unwrap();
        }
        let spawned = super::super::spawn::spawn_managed_engine_child(
            paths,
            "vllm",
            new_service_id,
            &model,
            resolve,
            &host,
            port_request,
            &resolve.device_policy,
            &[],
            None,
            None,
            None,
            require_api_key,
            false,
        );
        if matches!(
            spawned,
            Ok(super::super::spawn::ManagedSpawn::AlreadyRunning(_))
        ) {
            drop_orphaned_endpoint_key_on_already_running(paths, new_service_id, key.as_deref());
        }
        let key_kept = endpoint_keys::endpoint_key_file_if_present(paths, new_service_id).is_some();
        (spawned, key_kept)
    }

    /// The command a refusal names is the same for every action that can print
    /// it — `rocm services restart`, a fresh launch, and the daemon's respawn,
    /// whose text is the same shared guard's — so each is followed here, from
    /// the refused state, through the real `rocm serve` front half. A service
    /// that is recorded, and may still be running, is the next test's.
    #[test]
    fn the_command_a_respawn_refusal_names_clears_the_refusal() -> Result<()> {
        use rocm_core::ServiceStart;

        // A port outside `rocm serve`'s own range (a service published behind a
        // mapping), and one inside it.
        for (recorded_port, start) in [8000_u16, 11436].into_iter().flat_map(|port| {
            [
                (port, ServiceStart::Launch),
                (port, ServiceStart::Restart),
                (port, ServiceStart::Respawn),
            ]
        }) {
            for (host, requires_api_key) in [
                ("127.0.0.1", true),
                ("localhost", true),
                ("::1", true),
                ("0.0.0.0", true),
                ("gpu-box.example", true),
                ("0.0.0.0", false),
                ("192.168.1.10", false),
                ("gpu-box.example", false),
                // Not exactly a loopback spelling, so public to the guard and to
                // `rocm serve`: the command has to say `--host` for them, or it
                // brings the service back on the default host with no key.
                (" 127.0.0.1 ", false),
                ("127.0.0.1 ", true),
                ("localhost ", false),
                (" ::1", false),
                ("Localhost", false),
                ("127.0.0.01", false),
            ] {
                let message = rocm_core::ensure_public_service_has_endpoint_key(
                    host,
                    false,
                    requires_api_key,
                    start,
                    "lemonade-qwen-3000",
                    recorded_port,
                )
                .expect_err("a keyless service in this state is refused")
                .to_string();
                let advised = advised_serve_command(&message);

                // `rocm serve` takes a required <MODEL>: the command names it as
                // a placeholder, and parses as printed.
                assert_eq!(advised[2], "<model>", "{advised:?}");
                assert_eq!(
                    crate::advised_commands::rocm_verdict(&advised[1..]),
                    crate::advised_commands::Verdict::Parses,
                    "{advised:?}"
                );

                // With a model and a port in the placeholders' places the command
                // asks for what the refusal needs (`--require-api-key` for the
                // first rule, the public bind for the second), and what it stores
                // satisfies the guard that refused.
                let runnable: Vec<String> = advised
                    .iter()
                    .map(|word| match word.as_str() {
                        "<model>" => "Qwen/Qwen3-0.6B".to_owned(),
                        _ => word.clone(),
                    })
                    .collect();
                assert!(
                    runnable.iter().all(|word| !word.starts_with('<')),
                    "every placeholder in {advised:?} must be filled: {runnable:?}"
                );
                if start != ServiceStart::Launch {
                    // A restart or respawn is of a recorded service, which a launch
                    // under a new id does not replace while it still runs.
                    continue;
                }
                let (root, paths) = test_paths("respawn-advice-clears");
                paths.ensure()?;
                fs::create_dir_all(paths.services_dir())?;
                let service_id = "lemonade-qwen-3000";
                assert!(
                    endpoint_keys::endpoint_key_file_if_present(&paths, service_id).is_none(),
                    "the refused state has no key on disk"
                );
                guard_after_following(
                    &runnable,
                    &paths,
                    service_id,
                    start,
                    (host, requires_api_key, recorded_port),
                )
                .unwrap_or_else(|error| {
                    panic!("{runnable:?} does not clear {host} ({start:?}): {error:#}")
                });
                assert!(
                    endpoint_keys::endpoint_key_file_if_present(&paths, service_id).is_some(),
                    "following the advice leaves the service a key to respawn with"
                );
                let _ = fs::remove_dir_all(root);
            }
        }
        Ok(())
    }

    /// The model a launch of `qwen-canonical` resolves to, as the spawn is given it.
    #[cfg(not(windows))]
    fn resolved_qwen() -> rocm_engine_protocol::ResolveModelResponse {
        rocm_engine_protocol::ResolveModelResponse {
            canonical_model_id: "qwen-canonical".to_owned(),
            served_model_id: None,
            task: "chat".to_owned(),
            source: "hf".to_owned(),
            revision: "main".to_owned(),
            loader: "llama.cpp".to_owned(),
            trust_remote_code: false,
            chat_template_mode: "auto".to_owned(),
            dtype: "auto".to_owned(),
            device_policy: rocm_engine_protocol::DevicePolicy::GpuPreferred,
            estimated_memory: "unknown".to_owned(),
            launch_defaults: serde_json::json!({}),
            engine_recipe: None,
            warnings: Vec::new(),
        }
    }

    /// A recorded service whose key file is gone while the service still runs.
    /// `rocm serve` of the same engine and model reuses it and drops the key it
    /// generated for that attempt, so following only the `rocm serve` command
    /// leaves the service keyless and the next restart refused again. The refusal
    /// therefore names the stop first; this follows both, through the real stop,
    /// the real reuse in the spawn and the key `serve` stores under a new id.
    #[cfg(not(windows))]
    #[test]
    fn a_running_service_is_stopped_before_its_relaunch_so_the_advice_clears_the_refusal()
    -> Result<()> {
        use super::super::spawn::{ManagedSpawn, TEST_SPAWN_HOOK};
        use rocm_core::ServiceStart;
        let resolve = resolved_qwen();
        for start in [ServiceStart::Restart, ServiceStart::Respawn] {
            for (host, requires_api_key) in [
                ("127.0.0.1", true),
                ("localhost", true),
                ("0.0.0.0", true),
                ("0.0.0.0", false),
                ("gpu-box.example", false),
            ] {
                let (root, paths) = test_paths("running-service-advice");
                paths.ensure()?;
                fs::create_dir_all(paths.services_dir())?;
                let service_id = "vllm-qwen-1000";
                let port = std::net::TcpListener::bind(("127.0.0.1", 0))?
                    .local_addr()?
                    .port();
                // The staged state of `service-stop-09`: recorded, running, and
                // with no key file.
                let mut existing = rocm_core::ManagedServiceRecord::new(
                    &paths,
                    service_id,
                    "vllm",
                    "qwen",
                    "qwen-canonical",
                    host,
                    port,
                    "managed",
                    std::process::id(),
                    None,
                    None,
                    None,
                );
                existing.status = "ready".to_owned();
                existing.engine_pid = Some(std::process::id());
                existing.requires_api_key = requires_api_key;
                existing.write()?;
                let message = rocm_core::ensure_public_service_has_endpoint_key(
                    host,
                    false,
                    requires_api_key,
                    start,
                    service_id,
                    port,
                )
                .expect_err("a keyless service is refused")
                .to_string();
                let serve = advised_serve_command(&message);
                assert!(
                    serve
                        .windows(2)
                        .any(|pair| pair == ["--port", &port.to_string()]),
                    "{host} {start:?}: the launch must be on the recorded port {port}: {serve:?}"
                );
                let runnable: Vec<String> = serve
                    .iter()
                    .map(|word| match word.as_str() {
                        "<model>" => "Qwen/Qwen3-0.6B".to_owned(),
                        _ => word.clone(),
                    })
                    .collect();
                let stop = advised_stop_command(&message);
                let case = format!("{host} requires={requires_api_key} {start:?}");

                // The stop is the CLI's own hint, and a command `rocm` accepts.
                assert_eq!(
                    stop.join(" "),
                    crate::service_hints::service_stop_hint(service_id),
                    "{case}"
                );
                assert_eq!(
                    crate::advised_commands::rocm_verdict(&stop[1..]),
                    crate::advised_commands::Verdict::Parses,
                    "{case}"
                );
                // It comes before the launch in the sentence that names both.
                assert!(
                    message.find("rocm services stop").unwrap()
                        < message.find("rocm serve <model>").unwrap(),
                    "{case}: {message}"
                );

                // Only the launch, while the service still runs: `serve` reuses the
                // running service and drops the key it generated, so the recorded
                // service is exactly as keyless as before and is refused again.
                let (spawned, key_kept) =
                    follow_serve_through_the_spawn(&runnable, &paths, "vllm-qwen-2000", &resolve);
                assert!(
                    matches!(spawned, Ok(ManagedSpawn::AlreadyRunning(_))),
                    "{case}: the launch must find the running service"
                );
                assert!(!key_kept, "{case}: the unused key is dropped");
                assert!(
                    endpoint_keys::endpoint_key_file_if_present(&paths, service_id).is_none(),
                    "{case}: the running service did not gain a key"
                );
                assert!(
                    rocm_core::ensure_public_service_has_endpoint_key(
                        host,
                        endpoint_keys::endpoint_key_file_if_present(&paths, service_id).is_some(),
                        requires_api_key,
                        start,
                        service_id,
                        port,
                    )
                    .is_err(),
                    "{case}: launching without stopping first does not clear the refusal"
                );

                // The advised order: the stop, then the launch.
                let report = crate::service_control::stop_internal_managed_service_with(
                    &paths,
                    service_id,
                    |_| Vec::new(),
                )?;
                assert_eq!(report["stopped"], true, "{case}: {report}");
                let recorded = crate::managed_service_status::load_managed_services(&paths)?;
                assert!(
                    recorded
                        .iter()
                        .all(|record| record.service_id != service_id || record.status == "stopped"),
                    "{case}: the stop leaves the service stopped"
                );
                // The launch now finds nothing to reuse, so it goes on to spawn,
                // which is where this stops it: the guard that refused a keyless
                // start admitted this one because the key `serve` stored for it
                // is on disk.
                let reached_spawn = std::rc::Rc::new(std::cell::Cell::new(false));
                let observed = reached_spawn.clone();
                TEST_SPAWN_HOOK.with(|hook| {
                    *hook.borrow_mut() = Some(std::rc::Rc::new(move |_command| {
                        observed.set(true);
                        Err(std::io::Error::other("spawn intercepted by test"))
                    }));
                });
                let (spawned, _) =
                    follow_serve_through_the_spawn(&runnable, &paths, "vllm-qwen-3000", &resolve);
                TEST_SPAWN_HOOK.with(|hook| *hook.borrow_mut() = None);
                let error = spawned
                    .err()
                    .unwrap_or_else(|| panic!("{case}: the intercepted spawn fails"));
                assert!(
                    format!("{error:#}").contains("spawn intercepted by test"),
                    "{case}: {error:#}"
                );
                assert!(
                    reached_spawn.get(),
                    "{case}: after the stop the launch reaches the spawn with its key"
                );
                let _ = fs::remove_dir_all(root);
            }
        }
        Ok(())
    }

    /// The common way to the refusal: a service that has already stopped, whose
    /// key the stop dropped, restarted by id. The refusal still names the stop,
    /// and there is nothing for it to stop, so it must do no harm: succeed, leave
    /// the record stopped with no process and no pending stop request, and leave
    /// the launch that follows it free to go through to the spawn with its key.
    /// This follows the advice as printed through the real stop and the real
    /// spawn, for a recorded service that is not running.
    #[cfg(not(windows))]
    #[test]
    fn the_stop_a_refusal_names_is_harmless_on_a_service_that_has_already_stopped() -> Result<()> {
        use super::super::spawn::TEST_SPAWN_HOOK;
        use rocm_core::ServiceStart;

        let resolve = resolved_qwen();
        for start in [ServiceStart::Restart, ServiceStart::Respawn] {
            for (host, requires_api_key) in [
                ("127.0.0.1", true),
                ("0.0.0.0", true),
                ("0.0.0.0", false),
                ("gpu-box.example", false),
            ] {
                let (root, paths) = test_paths("stopped-service-advice");
                paths.ensure()?;
                fs::create_dir_all(paths.services_dir())?;
                let service_id = "vllm-qwen-1000";
                let port = std::net::TcpListener::bind(("127.0.0.1", 0))?
                    .local_addr()?
                    .port();
                // What a stop leaves: stopped, no process, no key file.
                let mut existing = rocm_core::ManagedServiceRecord::new(
                    &paths,
                    service_id,
                    "vllm",
                    "qwen",
                    "qwen-canonical",
                    host,
                    port,
                    "managed",
                    0,
                    None,
                    None,
                    None,
                );
                existing.status = "stopped".to_owned();
                existing.supervisor_pid = 0;
                existing.requires_api_key = requires_api_key;
                existing.write()?;
                let case = format!("{host} requires={requires_api_key} {start:?}");
                let message = rocm_core::ensure_public_service_has_endpoint_key(
                    host,
                    false,
                    requires_api_key,
                    start,
                    service_id,
                    port,
                )
                .expect_err("a keyless service is refused")
                .to_string();
                let stop = advised_stop_command(&message);
                assert_eq!(
                    stop.join(" "),
                    crate::service_hints::service_stop_hint(service_id),
                    "{case}"
                );

                // The advised stop, on a service with nothing left to stop.
                let report = crate::service_control::stop_internal_managed_service_with(
                    &paths,
                    service_id,
                    |_| Vec::new(),
                )?;
                assert_eq!(report["stopped"], true, "{case}: {report}");
                assert_eq!(report["status"], "stopped", "{case}: {report}");
                // As written to disk: a listing's liveness refresh would tidy a
                // stale stop request away and hide it.
                let (after, _) =
                    crate::services_text::read_managed_service_manifest(&paths, service_id)?;
                assert_eq!(after.status, "stopped", "{case}");
                assert_eq!(after.supervisor_pid, 0, "{case}");
                assert!(after.engine_pid.is_none(), "{case}: {after:?}");
                assert!(
                    after.stop_requested_unix_ms.is_none(),
                    "{case}: a confirmed stop leaves no stop pending: {after:?}"
                );
                assert_eq!(after.requires_api_key, requires_api_key, "{case}");

                // The launch that follows finds nothing running to reuse and
                // reaches the spawn with the key `serve` stored for it.
                let runnable: Vec<String> = advised_serve_command(&message)
                    .iter()
                    .map(|word| match word.as_str() {
                        "<model>" => "Qwen/Qwen3-0.6B".to_owned(),
                        _ => word.clone(),
                    })
                    .collect();
                let reached_spawn = std::rc::Rc::new(std::cell::Cell::new(false));
                let observed = reached_spawn.clone();
                TEST_SPAWN_HOOK.with(|hook| {
                    *hook.borrow_mut() = Some(std::rc::Rc::new(move |_command| {
                        observed.set(true);
                        Err(std::io::Error::other("spawn intercepted by test"))
                    }));
                });
                let (spawned, _) =
                    follow_serve_through_the_spawn(&runnable, &paths, "vllm-qwen-2000", &resolve);
                TEST_SPAWN_HOOK.with(|hook| *hook.borrow_mut() = None);
                let error = spawned
                    .err()
                    .unwrap_or_else(|| panic!("{case}: the intercepted spawn fails"));
                assert!(
                    format!("{error:#}").contains("spawn intercepted by test"),
                    "{case}: {error:#}"
                );
                assert!(
                    reached_spawn.get(),
                    "{case}: after the stop the launch reaches the spawn"
                );
                let _ = fs::remove_dir_all(root);
            }
        }
        Ok(())
    }

    /// The command a refusal names is pasted into a shell, and the host in it is a
    /// string the record holds. Read back through a real `sh` (which is what the
    /// printed quoting has to survive) in a directory with files a glob could
    /// match, every value must come out as the one word the record had, and the
    /// words must parse through the real `rocm serve` parser to the recorded host
    /// and port.
    #[cfg(unix)]
    #[test]
    fn the_command_a_refusal_names_reaches_the_recorded_host_through_a_real_shell() -> Result<()> {
        use rocm_core::ServiceStart;

        let (root, _paths) = test_paths("refusal-command-shell");
        fs::create_dir_all(&root)?;
        for file in ["1", "x", ":"] {
            fs::write(root.join(file), "")?;
        }
        for host in [
            "::1",
            "[::1]",
            "*",
            "gpu box",
            "a;b",
            "$HOME",
            "it's",
            "gpu-box.example",
            " 127.0.0.1 ",
            "localhost ",
            " ::1",
        ] {
            for start in [
                ServiceStart::Launch,
                ServiceStart::Restart,
                ServiceStart::Respawn,
            ] {
                let message = rocm_core::ensure_public_service_has_endpoint_key(
                    host,
                    false,
                    true,
                    start,
                    "lemonade-qwen-3000",
                    8000,
                )
                .expect_err("a keyless service is refused")
                .to_string();
                let printed: Vec<&str> = message
                    .split('`')
                    .skip(1)
                    .step_by(2)
                    .filter(|span| span.starts_with("rocm serve"))
                    .collect();
                assert_eq!(printed.len(), 1, "{message}");
                let script = format!("printf '%s\\n' {}", printed[0].replace("<model>", "qwen"));
                let output = std::process::Command::new("sh")
                    .arg("-c")
                    .arg(&script)
                    .current_dir(&root)
                    .output()?;
                assert!(output.status.success(), "{host:?} {script}: {output:?}");
                let words: Vec<String> = String::from_utf8(output.stdout)?
                    .lines()
                    .map(str::to_owned)
                    .collect();
                // The host is one word, and it is the recorded one.
                let at = words
                    .iter()
                    .position(|word| word == "--host")
                    .unwrap_or_else(|| panic!("{host:?} {start:?}: no --host in {words:?}"));
                assert_eq!(words[at + 1], host, "{start:?}: {words:?} from {script}");
                // And the real parser reads the same host and port out of it.
                let ServeAdvice {
                    host: parsed_host,
                    port,
                    ..
                } = parse_serve_advice(&words);
                assert_eq!(parsed_host, host, "{start:?}: {words:?}");
                assert_eq!(port, Some(8000), "{start:?}: {words:?}");
            }
        }
        let _ = fs::remove_dir_all(root);
        Ok(())
    }
}
