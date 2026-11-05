// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use rocm_core::AppPaths;
use rocm_engine_protocol::{
    EngineMethod, EngineRequestEnvelope, EngineResponseEnvelope, HealthcheckRequest,
    HealthcheckResponse,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::io::Write;
use std::process::{Command as ProcessCommand, Stdio};

pub(crate) fn engine_healthcheck_response(
    paths: &AppPaths,
    engine: &str,
    service_id: &str,
) -> Result<HealthcheckResponse> {
    engine_request::<_, HealthcheckResponse>(
        paths,
        engine,
        service_id,
        EngineMethod::Healthcheck,
        &HealthcheckRequest {
            service_id: service_id.to_owned(),
        },
    )
}

pub(crate) fn healthcheck_response_ready(response: &HealthcheckResponse) -> bool {
    response.status == "ready" && response.model_loaded
}

pub(crate) fn healthcheck_response_recoverable(response: &HealthcheckResponse) -> bool {
    matches!(
        response.status.as_str(),
        "failed" | "unreachable" | "exited"
    )
}

/// Re-thread the endpoint key file (public bind only) onto an engine child's
/// environment, mirroring the initial `rocm serve` spawn. A loopback service
/// with no stored key leaves the command's environment untouched, matching the
/// unauthenticated default.
///
/// Returns whether a key was applied. Callers that spawn a *listener* must
/// check it against the service's host — a public bind with no key would come
/// up anonymous (see [`ensure_public_service_has_endpoint_key`]). Callers that
/// only make a stdio plugin call can ignore it.
///
/// The file must hold a *usable* key, not merely exist: the engine adapters
/// resolve it with `endpoint_api_key_from_file` and enforce nothing when that
/// yields `None`, so treating an empty or malformed file as "protected" would
/// let the guard pass while the endpoint served anonymously.
#[must_use]
pub(crate) fn apply_endpoint_key_env(
    command: &mut ProcessCommand,
    paths: &AppPaths,
    service_id: &str,
) -> bool {
    if let Some(key_file) = rocm_engine_protocol::endpoint_key_file_if_present(paths, service_id)
        .and_then(|path| rocm_engine_protocol::endpoint_api_key_file_if_valid(&path))
    {
        command.env(rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV, key_file);
        return true;
    }
    false
}

/// Refuse to respawn a recorded service on a public host without its endpoint
/// API key, so daemon recovery cannot reopen a protected endpoint anonymously.
///
/// Mirrors the guard of the same name in `rocm`; the shared
/// [`rocm_engine_protocol::is_public_bind_host`] keeps the two classifications
/// identical for a given `ManagedServiceRecord::host`.
pub(crate) fn ensure_public_service_has_endpoint_key(
    host: &str,
    key_present: bool,
    requires_api_key: bool,
) -> Result<()> {
    // The bind address is not the whole story. A service bound to loopback is
    // only private until something republishes the port, and a tailnet publish
    // outlives both the process and this daemon. The requirement is recorded on
    // the service precisely so recovery can honour it without re-deriving it
    // from an address that no longer answers the question.
    if requires_api_key && !key_present {
        bail!(
            "refusing to recover a service that was launched with an endpoint API key but no \
             longer has one: it would come back up without authentication, and something \
             outside this machine may still be publishing its port. Relaunch it with \
             `rocm serve --require-api-key` to issue a new key."
        );
    }
    if rocm_engine_protocol::is_public_bind_host(host) && !key_present {
        bail!(
            "refusing to respawn a service bound to the public host `{host}` without an endpoint \
             API key: it would come back up without authentication. Relaunch it with \
             `rocm serve --host {host} --allow-public-bind` to issue a new key."
        );
    }
    Ok(())
}

pub(crate) fn engine_request<T, R>(
    paths: &AppPaths,
    engine: &str,
    service_id: &str,
    method: EngineMethod,
    request: &T,
) -> Result<R>
where
    T: Serialize,
    R: DeserializeOwned,
{
    let envelope = EngineRequestEnvelope {
        method,
        payload: serde_json::to_value(request).context("failed to serialize engine request")?,
    };
    let engine_binary =
        std::env::current_exe().context("failed to resolve current rocm executable path")?;
    engine_request_via(&engine_binary, paths, engine, service_id, &envelope)
}

/// [`engine_request`] with the executable that serves `__engine-stdio` named by
/// the caller, so a test can point it at a stand-in.
fn engine_request_via<R>(
    engine_binary: &std::path::Path,
    paths: &AppPaths,
    engine: &str,
    service_id: &str,
    envelope: &EngineRequestEnvelope,
) -> Result<R>
where
    R: DeserializeOwned,
{
    let mut command = ProcessCommand::new(engine_binary);
    command
        .arg("__engine-stdio")
        .arg(engine)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // A protected public endpoint enforces its key on every request, including
    // this stdio-transported healthcheck — without the carrier the engine
    // adapter's HTTP probe is unauthenticated and the endpoint looks
    // unreachable, which would misclassify a healthy protected service as
    // recoverable.
    //
    // No host to check here: this is a stdio plugin call, not a listener, so a
    // missing key only weakens this probe rather than opening a port.
    let _ = apply_endpoint_key_env(&mut command, paths, service_id);
    let mut child = command.spawn().with_context(|| {
        format!(
            "failed to spawn engine stdio process {}",
            engine_binary.display()
        )
    })?;

    {
        let stdin = child
            .stdin
            .as_mut()
            .context("engine stdio child did not expose stdin")?;
        serde_json::to_writer(&mut *stdin, envelope).context("failed to write engine request")?;
        stdin.write_all(b"\n")?;
    }

    let output = child
        .wait_with_output()
        .context("failed waiting for engine stdio response")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        if stderr.is_empty() {
            bail!("engine stdio process exited with status {}", output.status);
        }
        bail!(
            "engine stdio process exited with status {}: {}",
            output.status,
            stderr
        );
    }
    let envelope: EngineResponseEnvelope =
        serde_json::from_slice(&output.stdout).context("failed to parse engine response")?;
    if !envelope.ok {
        let detail = envelope.error.map_or_else(
            || "unknown engine error".to_owned(),
            |error| format!("{}: {}", error.code, error.message),
        );
        bail!(detail);
    }
    let data = envelope
        .data
        .context("engine response did not include data")?;
    serde_json::from_value(data).context("failed to deserialize engine response data")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_app_paths;
    use std::fs;

    #[test]
    fn recovery_refuses_a_service_that_lost_a_key_it_was_launched_with() {
        // The daemon keeps its own copy of this guard, and it only knew about
        // public binds. A loopback service that something republishes — a
        // tailnet publish outlives this daemon, let alone the process — would
        // be recovered without authentication, and the rebuilt record would
        // then disarm `rocm services restart` too.
        for host in ["127.0.0.1", "localhost", "::1"] {
            let error = ensure_public_service_has_endpoint_key(host, false, true)
                .expect_err("a service launched with a key must not be recovered without one");
            assert!(
                format!("{error:#}").contains("without authentication"),
                "{error:#}"
            );
        }
        // With the key still present, recovery proceeds.
        ensure_public_service_has_endpoint_key("127.0.0.1", true, true).unwrap();
        // And a service that never had one is untouched.
        ensure_public_service_has_endpoint_key("127.0.0.1", false, false).unwrap();
    }

    #[test]
    fn recovery_respawn_fails_closed_for_a_public_service_without_a_key() {
        // Daemon recovery re-execs `rocmd supervise` for the recorded host. With
        // the key gone the child would listen on that public host anonymously,
        // so the spawn must be refused instead.
        let error = ensure_public_service_has_endpoint_key("0.0.0.0", false, false).unwrap_err();
        assert!(
            error.to_string().contains("without authentication"),
            "{error:#}"
        );

        ensure_public_service_has_endpoint_key("0.0.0.0", true, false).unwrap();
        for host in ["127.0.0.1", "localhost", "::1"] {
            ensure_public_service_has_endpoint_key(host, false, false)
                .unwrap_or_else(|error| panic!("{host} must not require a key: {error:#}"));
        }
    }

    #[test]
    fn apply_endpoint_key_env_sets_var_only_when_key_file_present() {
        let (_root, paths) = temp_app_paths("apply-endpoint-key-env");
        let service_id = "svc-endpoint-key-env";

        // Loopback service: no key file has ever been written, so the child's
        // environment must be left untouched.
        let mut command = ProcessCommand::new("true");
        assert!(!apply_endpoint_key_env(&mut command, &paths, service_id));
        assert!(
            command
                .get_envs()
                .all(|(key, _)| key != rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV),
            "loopback service must not receive the endpoint key env var"
        );

        // Public service: once a key file exists at the deterministic path,
        // it must be threaded onto the child so the engine's HTTP probe
        // authenticates.
        let key_path = rocm_engine_protocol::endpoint_key_file_path(&paths, service_id);
        fs::create_dir_all(paths.services_dir()).unwrap();
        fs::write(&key_path, "secret-key").unwrap();
        let mut command = ProcessCommand::new("true");
        assert!(apply_endpoint_key_env(&mut command, &paths, service_id));
        let env_value = command
            .get_envs()
            .find_map(|(key, value)| {
                (key == rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV)
                    .then_some(value)
                    .flatten()
            })
            .expect("endpoint key env var must be set once the key file exists");
        assert_eq!(env_value, key_path.as_os_str());

        // An empty (or otherwise unusable) key file is not a key: the engine
        // adapters would enforce nothing, so reporting `true` here would let the
        // respawn guard pass and still open an anonymous public listener.
        fs::write(&key_path, "   \n").unwrap();
        let mut command = ProcessCommand::new("true");
        assert!(!apply_endpoint_key_env(&mut command, &paths, service_id));
        assert!(
            command
                .get_envs()
                .all(|(key, _)| key != rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV),
            "an unusable key file must not be threaded onto the child"
        );
    }

    #[test]
    fn healthcheck_readiness_requires_ready_loaded_model() {
        let ready = HealthcheckResponse {
            status: "ready".to_owned(),
            model_loaded: true,
            device: "cuda".to_owned(),
            uptime_sec: 1,
            queue_depth: 0,
            last_error: None,
            tokens_per_sec: None,
        };
        assert!(healthcheck_response_ready(&ready));

        let mut loading = ready.clone();
        loading.status = "loading_model".to_owned();
        assert!(!healthcheck_response_ready(&loading));

        let mut unloaded = ready;
        unloaded.model_loaded = false;
        assert!(!healthcheck_response_ready(&unloaded));
    }

    #[test]
    fn healthcheck_recoverability_tracks_failed_endpoint_state() {
        let mut response = HealthcheckResponse {
            status: "ready".to_owned(),
            model_loaded: true,
            device: "cuda".to_owned(),
            uptime_sec: 1,
            queue_depth: 0,
            last_error: None,
            tokens_per_sec: None,
        };
        assert!(!healthcheck_response_recoverable(&response));

        response.status = "unreachable".to_owned();
        assert!(healthcheck_response_recoverable(&response));

        response.status = "failed".to_owned();
        assert!(healthcheck_response_recoverable(&response));

        response.status = "loading_model".to_owned();
        assert!(!healthcheck_response_recoverable(&response));

        // The status the engines report for a model that is listed but has not
        // yet served an inference request. Restarting it would kill a model
        // mid-load and start the wait over.
        response.status = "loading".to_owned();
        assert!(!healthcheck_response_recoverable(&response));
    }

    #[test]
    fn healthcheck_readiness_withheld_while_the_model_only_lists() {
        // What an engine reports once `/v1/models` answers but inference has not:
        // not ready, so `rocm serve` keeps waiting instead of handing the caller
        // an endpoint that will hang on its first request.
        let listing_only = HealthcheckResponse {
            status: "loading".to_owned(),
            model_loaded: false,
            device: "unknown".to_owned(),
            uptime_sec: 1,
            queue_depth: 0,
            last_error: None,
            tokens_per_sec: None,
        };
        assert!(!healthcheck_response_ready(&listing_only));
    }

    // ---- The stdio client, driven against a stand-in engine child. ----
    //
    // These pin what `rocmd` puts on the child's command line, environment and
    // stdin and how it reads the answer back. `rocm` keeps its own client with a
    // different contract; its tests are in `apps/rocm/src/engine_ipc.rs`.
    #[cfg(unix)]
    mod stdio_child {
        use super::*;
        use rocm_engine_protocol::fake_engine::FakeEngine;
        use serde_json::{Value, json};
        use std::os::unix::process::ExitStatusExt as _;

        const SERVICE_ID: &str = "svc-stdio-child";
        const STOPPED: &str = r#"{"ok":true,"data":{"stopped":true}}"#;

        fn healthcheck_envelope() -> EngineRequestEnvelope {
            EngineRequestEnvelope {
                method: EngineMethod::Healthcheck,
                payload: json!({"service_id": SERVICE_ID}),
            }
        }

        fn request(engine: &FakeEngine, paths: &AppPaths) -> Result<Value> {
            engine.run(|| {
                engine_request_via::<Value>(
                    engine.path(),
                    paths,
                    "vllm",
                    SERVICE_ID,
                    &healthcheck_envelope(),
                )
            })
        }

        fn ambient(key: &str) -> Option<String> {
            std::env::var(key).ok()
        }

        #[test]
        fn the_child_is_asked_to_serve_the_engine_over_stdio_with_one_line_of_request() {
            let (_root, paths) = temp_app_paths("stdio-framing");
            let engine = FakeEngine::answering("rocmd-framing", STOPPED, 0);
            let answer = request(&engine, &paths).unwrap();

            assert_eq!(answer, json!({"stopped": true}));
            assert_eq!(engine.argv(), ["__engine-stdio", "vllm"]);
            assert_eq!(
                engine.stdin(),
                "{\"method\":\"healthcheck\",\"payload\":{\"service_id\":\"svc-stdio-child\"}}\n"
            );
        }

        #[test]
        fn the_endpoint_key_file_reaches_the_child_only_when_it_holds_a_usable_key() {
            let (_root, paths) = temp_app_paths("stdio-endpoint-key");
            let key_env = rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV;

            let keyless = FakeEngine::answering("rocmd-keyless", STOPPED, 0);
            request(&keyless, &paths).unwrap();
            assert_eq!(keyless.env(key_env), ambient(key_env));

            let key_path = rocm_engine_protocol::endpoint_key_file_path(&paths, SERVICE_ID);
            fs::create_dir_all(paths.services_dir()).unwrap();
            fs::write(&key_path, "   \n").unwrap();
            let unusable = FakeEngine::answering("rocmd-unusable-key", STOPPED, 0);
            request(&unusable, &paths).unwrap();
            assert_eq!(unusable.env(key_env), ambient(key_env));

            fs::write(&key_path, "secret-key").unwrap();
            let keyed = FakeEngine::answering("rocmd-keyed", STOPPED, 0);
            request(&keyed, &paths).unwrap();
            assert_eq!(keyed.env(key_env).as_deref(), key_path.to_str());
        }

        #[test]
        fn the_child_is_not_given_the_daemons_app_paths() {
            // It relies on the environment it inherits to find the same folders
            // the daemon runs on; nothing here sets `ROCM_CLI_*_DIR` for it.
            let (_root, paths) = temp_app_paths("stdio-no-app-paths");
            let engine = FakeEngine::answering("rocmd-no-app-paths", STOPPED, 0);
            request(&engine, &paths).unwrap();
            for key in [
                "ROCM_CLI_CONFIG_DIR",
                "ROCM_CLI_DATA_DIR",
                "ROCM_CLI_CACHE_DIR",
            ] {
                assert_eq!(engine.env(key), ambient(key), "{key}");
            }
        }

        #[test]
        fn any_nonzero_exit_is_an_error_even_when_the_child_answered() {
            let (_root, paths) = temp_app_paths("stdio-nonzero");
            let crash = FakeEngine::new("rocmd-crash", true, "echo boom >&2\nexit 3");
            assert_eq!(
                request(&crash, &paths).unwrap_err().to_string(),
                "engine stdio process exited with status exit status: 3: boom"
            );

            let silent = FakeEngine::new("rocmd-silent-crash", true, "exit 4");
            assert_eq!(
                request(&silent, &paths).unwrap_err().to_string(),
                "engine stdio process exited with status exit status: 4"
            );

            // `rocm` decodes an answer that came with a failing exit; this client
            // discards it for the status.
            let answered = FakeEngine::answering("rocmd-answers-then-fails", STOPPED, 2);
            assert_eq!(
                request(&answered, &paths).unwrap_err().to_string(),
                "engine stdio process exited with status exit status: 2"
            );
        }

        #[test]
        fn an_unusable_answer_is_reported_in_this_clients_own_words() {
            let (_root, paths) = temp_app_paths("stdio-answers");

            let garbage =
                FakeEngine::new("rocmd-garbage", true, "printf 'not json'\necho oops >&2");
            assert_eq!(
                request(&garbage, &paths).unwrap_err().to_string(),
                "failed to parse engine response"
            );

            let failed = FakeEngine::answering(
                "rocmd-failed",
                r#"{"ok":false,"error":{"code":"stop_failed","message":"it broke"}}"#,
                0,
            );
            assert_eq!(
                request(&failed, &paths).unwrap_err().to_string(),
                "stop_failed: it broke"
            );

            let unexplained = FakeEngine::answering("rocmd-unexplained", r#"{"ok":false}"#, 0);
            assert_eq!(
                request(&unexplained, &paths).unwrap_err().to_string(),
                "unknown engine error"
            );

            let empty = FakeEngine::answering("rocmd-empty", r#"{"ok":true}"#, 0);
            assert_eq!(
                request(&empty, &paths).unwrap_err().to_string(),
                "engine response did not include data"
            );

            let wrong = FakeEngine::answering("rocmd-wrong", r#"{"ok":true,"data":[]}"#, 0);
            let error = engine_request_via::<rocm_engine_protocol::StopResponse>(
                wrong.path(),
                &paths,
                "vllm",
                SERVICE_ID,
                &healthcheck_envelope(),
            )
            .unwrap_err();
            assert_eq!(
                error.to_string(),
                "failed to deserialize engine response data"
            );
        }

        /// Writing a request to a child that has already exited, in this client:
        /// where SIGPIPE is ignored (Rust's startup default, which the test
        /// harness keeps) the write fails as an ordinary error, and where it is
        /// at its default action (`rocm` resets it to that in `main`, and `rocm
        /// daemon run` runs this code in that process) the write kills the
        /// process. This client has no guard around the write, so the second half
        /// pins that gap. Re-executes this test in a fresh process for each case,
        /// since changing the disposition process-wide would race sibling tests.
        #[test]
        #[allow(unsafe_code)] // libc::signal, to restore the default action in the child
        fn a_request_to_a_child_that_already_exited_depends_on_the_sigpipe_disposition() {
            if let Some(arm) = std::env::var_os("ROCMD_TEST_ENGINE_SIGPIPE_ARM") {
                if arm == "default-action" {
                    // SAFETY: installing a signal disposition is always safe.
                    unsafe {
                        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
                    }
                }
                let (_root, paths) = temp_app_paths("stdio-sigpipe");
                let engine = FakeEngine::new("rocmd-sigpipe", false, "exit 3");
                // Larger than a pipe buffer, so the write reaches the closed read end.
                let big = "x".repeat(1 << 20);
                let envelope = EngineRequestEnvelope {
                    method: EngineMethod::Healthcheck,
                    payload: json!({ "blob": big }),
                };
                let result = engine.run(|| {
                    engine_request_via::<Value>(
                        engine.path(),
                        &paths,
                        "vllm",
                        SERVICE_ID,
                        &envelope,
                    )
                });
                let error = result.expect_err("the engine never answered");
                assert!(
                    format!("{error:#}").contains("failed to write engine request"),
                    "{error:#}"
                );
                return;
            }

            let run_arm = |arm: &str| {
                std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "engine_ipc::tests::stdio_child::a_request_to_a_child_that_already_exited_depends_on_the_sigpipe_disposition",
                        "--exact",
                        "--nocapture",
                    ])
                    .env("ROCMD_TEST_ENGINE_SIGPIPE_ARM", arm)
                    .status()
                    .unwrap()
            };

            let ignored = run_arm("ignored");
            assert!(ignored.success(), "{ignored:?}");

            let default_action = run_arm("default-action");
            assert_eq!(
                default_action.signal(),
                Some(libc::SIGPIPE),
                "sanity check: this client has no SIGPIPE guard, so under the default action the \
                 write kills the process ({default_action:?})"
            );
        }
    }
}
