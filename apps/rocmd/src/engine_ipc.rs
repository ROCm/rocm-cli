// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result};
use rocm_core::AppPaths;
use rocm_engine_protocol::{
    EngineMethod, EngineStdioOptions, HealthcheckRequest, HealthcheckResponse,
    decode_engine_response, engine_request_envelope, run_engine_stdio,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

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
/// up anonymous (see [`rocm_core::ensure_public_service_has_endpoint_key`]). Callers that
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
    if let Some(key_file) = usable_endpoint_key_file(paths, service_id) {
        command.env(rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV, key_file);
        return true;
    }
    false
}

/// The service's endpoint key file, when it exists and holds a usable key.
fn usable_endpoint_key_file(paths: &AppPaths, service_id: &str) -> Option<PathBuf> {
    rocm_engine_protocol::endpoint_key_file_if_present(paths, service_id)
        .and_then(|path| rocm_engine_protocol::endpoint_api_key_file_if_valid(&path))
}

/// Ask an engine a question about a managed service, through the stdio client
/// `rocmd` shares with `rocm`.
///
/// The engine is reached by running the `rocm` executable as
/// `__engine-stdio <engine>`, the verb `rocm` serves for its built-in engines
/// ([`rocm_executable`] says which executable that is).
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
    let envelope = engine_request_envelope(method, request)?;
    let engine_binary = rocm_executable()?;
    engine_request_via(&engine_binary, paths, engine, service_id, &envelope)
}

/// The `rocm` executable that serves `__engine-stdio`.
///
/// `rocmd` runs in two shapes. Embedded, as `rocm daemon`, this process
/// is `rocm` and its own executable is the answer. Run as the standalone
/// `rocmd` binary shipped beside `rocm`, its own executable has no such verb
/// (its parser answers "unrecognized subcommand"), so every engine request
/// failed; there the answer is the `rocm` next to it.
fn rocm_executable() -> Result<PathBuf> {
    let current =
        std::env::current_exe().context("failed to resolve current rocm executable path")?;
    rocm_executable_from(current, || rocm_core::sibling_binary_path("rocm"))
}

/// [`rocm_executable`] given this process's executable and the way to find a
/// sibling `rocm`, so the choice can be tested without either.
fn rocm_executable_from(
    current: PathBuf,
    sibling_rocm: impl FnOnce() -> Result<PathBuf>,
) -> Result<PathBuf> {
    // Exactly `rocmd` (or `rocmd.exe`). A renamed or test binary is not known to
    // lack the verb, and has no `rocm` beside it to prefer.
    if current.file_stem().is_some_and(|stem| stem == "rocmd") {
        return sibling_rocm().context(
            "the standalone rocmd needs the rocm executable beside it to reach an engine",
        );
    }
    Ok(current)
}

/// [`engine_request`] with the executable that serves `__engine-stdio` named by
/// the caller, so a test can point it at a stand-in.
fn engine_request_via<R>(
    engine_binary: &Path,
    paths: &AppPaths,
    engine: &str,
    service_id: &str,
    envelope: &rocm_engine_protocol::EngineRequestEnvelope,
) -> Result<R>
where
    R: DeserializeOwned,
{
    // A protected public endpoint enforces its key on every request, including
    // this stdio-transported healthcheck — without the carrier the engine
    // adapter's HTTP probe is unauthenticated and the endpoint looks
    // unreachable, which would misclassify a healthy protected service as
    // recoverable.
    //
    // No host to check here: this is a stdio plugin call, not a listener, so a
    // missing key only weakens this probe rather than opening a port.
    let key_file = usable_endpoint_key_file(paths, service_id);
    let options = stdio_options(paths, key_file.as_deref());
    decode_engine_response(run_engine_stdio(
        engine_binary,
        ["__engine-stdio", engine],
        &options,
        envelope,
    )?)
}

/// What `rocmd` asks of the shared stdio client; see [`engine_request`].
///
/// The child runs on the daemon's own app paths and carries the service's
/// endpoint key file when there is a usable one. Its stderr is kept for the
/// error, not forwarded: nobody watches a daemon's terminal, and a healthcheck
/// asks for no progress.
const fn stdio_options<'a>(
    paths: &'a AppPaths,
    endpoint_key_file: Option<&'a Path>,
) -> EngineStdioOptions<'a> {
    EngineStdioOptions {
        app_paths: Some(paths),
        env_root: None,
        endpoint_key_file,
        request_progress: false,
        forward_stderr: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_app_paths;
    use std::fs;

    #[test]
    fn recovery_refuses_a_service_that_lost_a_key_it_was_launched_with() {
        // The daemon once kept its own copy of this guard, and it only knew
        // about public binds. A loopback service that something republishes — a
        // tailnet publish outlives this daemon, let alone the process — would
        // be recovered without authentication, and the rebuilt record would
        // then disarm `rocm services restart` too.
        for host in ["127.0.0.1", "localhost", "::1"] {
            let error = rocm_core::ensure_public_service_has_endpoint_key(
                host,
                false,
                true,
                rocm_core::ServiceStart::Respawn,
            )
            .expect_err("a service launched with a key must not be recovered without one");
            let message = format!("{error:#}");
            assert!(
                message.contains("respawning it would reopen it without authentication"),
                "{message}"
            );
        }
        // With the key still present, recovery proceeds.
        rocm_core::ensure_public_service_has_endpoint_key(
            "127.0.0.1",
            true,
            true,
            rocm_core::ServiceStart::Respawn,
        )
        .unwrap();
        // And a service that never had one is untouched.
        rocm_core::ensure_public_service_has_endpoint_key(
            "127.0.0.1",
            false,
            false,
            rocm_core::ServiceStart::Respawn,
        )
        .unwrap();
    }

    #[test]
    fn recovery_respawn_fails_closed_for_a_public_service_without_a_key() {
        // Daemon recovery re-execs `rocmd supervise` for the recorded host. With
        // the key gone the child would listen on that public host anonymously,
        // so the spawn must be refused instead.
        let error = rocm_core::ensure_public_service_has_endpoint_key(
            "0.0.0.0",
            false,
            false,
            rocm_core::ServiceStart::Respawn,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("respawning it would reopen it without authentication"),
            "{error:#}"
        );

        rocm_core::ensure_public_service_has_endpoint_key(
            "0.0.0.0",
            true,
            false,
            rocm_core::ServiceStart::Respawn,
        )
        .unwrap();
        for host in ["127.0.0.1", "localhost", "::1"] {
            rocm_core::ensure_public_service_has_endpoint_key(
                host,
                false,
                false,
                rocm_core::ServiceStart::Respawn,
            )
            .unwrap_or_else(|error| panic!("{host} must not require a key: {error:#}"));
        }
    }

    #[test]
    fn the_key_file_rocm_serve_stores_clears_the_recovery_refusal() {
        // The condition both refusals report is a missing *usable* key. Relaunching
        // with the command they name makes `rocm serve` write the key file at the
        // deterministic path, and the expression recovery uses to decide
        // `key_present` must then read it as present. The parse of the named
        // command itself is checked in `rocm`, which owns the clap tree, against
        // the very text this guard produces
        // (`the_command_a_respawn_refusal_names_clears_the_refusal`).
        let (_root, paths) = temp_app_paths("recovery-refusal-cleared");
        let service_id = "svc-recovery-refusal-cleared";
        let key_present = || {
            rocm_engine_protocol::endpoint_key_file_if_present(&paths, service_id)
                .and_then(|path| rocm_engine_protocol::endpoint_api_key_file_if_valid(&path))
                .is_some()
        };
        for (host, requires_api_key) in [("127.0.0.1", true), ("0.0.0.0", false)] {
            let _ = fs::remove_file(rocm_engine_protocol::endpoint_key_file_path(
                &paths, service_id,
            ));
            assert!(!key_present());
            rocm_core::ensure_public_service_has_endpoint_key(
                host,
                key_present(),
                requires_api_key,
                rocm_core::ServiceStart::Respawn,
            )
            .expect_err("no key on disk: refused");

            fs::create_dir_all(paths.services_dir()).unwrap();
            fs::write(
                rocm_engine_protocol::endpoint_key_file_path(&paths, service_id),
                "freshly-issued-key\n",
            )
            .unwrap();
            assert!(key_present());
            rocm_core::ensure_public_service_has_endpoint_key(
                host,
                key_present(),
                requires_api_key,
                rocm_core::ServiceStart::Respawn,
            )
            .unwrap_or_else(|error| panic!("{host}: a stored key must clear it: {error:#}"));
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

    // ---- What `rocmd` asks of the shared stdio client. ----
    //
    // The conversation itself (framing, SIGPIPE guard, exit status and answer
    // handling) is the client's, tested in `rocm-engine-protocol`. These tests
    // run `rocmd`'s entry point against a stand-in engine child and pin what
    // `rocmd` chooses: the verb, the environment and the stderr handling. The
    // other caller's choices are pinned in `apps/rocm/src/engine_ipc.rs`.

    #[test]
    fn the_standalone_rocmd_reaches_engines_through_the_rocm_beside_it() {
        let beside = || Ok(PathBuf::from("/opt/rocm-cli/bin/rocm"));

        // The standalone binary has no `__engine-stdio` verb, so it must not
        // re-execute itself.
        for standalone in ["/opt/rocm-cli/bin/rocmd", "C:/rocm-cli/bin/rocmd.exe"] {
            assert_eq!(
                rocm_executable_from(PathBuf::from(standalone), beside).unwrap(),
                Path::new("/opt/rocm-cli/bin/rocm"),
                "{standalone}"
            );
        }

        // Embedded as `rocm daemon`, this process is `rocm`; a test binary
        // or a renamed copy has no `rocm` beside it to prefer.
        for own in [
            "/opt/rocm-cli/bin/rocm",
            "C:/rocm-cli/bin/rocm.exe",
            "/work/target/debug/deps/rocmd-0f3a9c1d",
            "/opt/rocm-cli/bin/rocm-0.3",
        ] {
            let chosen = rocm_executable_from(PathBuf::from(own), || {
                panic!("{own} has no use for a sibling")
            })
            .unwrap();
            assert_eq!(chosen, Path::new(own));
        }

        // A standalone `rocmd` with no `rocm` beside it fails with a message that
        // says why, rather than running something that cannot serve the verb.
        let error = rocm_executable_from(PathBuf::from("/opt/rocm-cli/bin/rocmd"), || {
            anyhow::bail!("unable to locate sibling binary rocm")
        })
        .unwrap_err();
        assert_eq!(
            format!("{error:#}"),
            "the standalone rocmd needs the rocm executable beside it to reach an engine: \
             unable to locate sibling binary rocm"
        );
    }

    #[test]
    fn rocmd_runs_the_child_on_its_own_paths_keeps_stderr_and_asks_for_no_progress() {
        let (_root, paths) = temp_app_paths("stdio-options");
        let key_file = Path::new("/keys/svc.endpoint-key");

        let options = stdio_options(&paths, Some(key_file));
        assert_eq!(
            options.app_paths.map(|p| &p.data_dir),
            Some(&paths.data_dir)
        );
        assert_eq!(options.endpoint_key_file, Some(key_file));
        // A healthcheck has no install to report progress on, and nobody watches
        // a daemon's terminal.
        assert!(!options.request_progress);
        assert!(!options.forward_stderr);
        assert_eq!(options.env_root, None);

        assert_eq!(stdio_options(&paths, None).endpoint_key_file, None);
    }

    #[cfg(unix)]
    mod stdio_child {
        use super::*;
        use rocm_engine_protocol::EngineRequestEnvelope;
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
            engine_request_via::<Value>(
                engine.path(),
                paths,
                "vllm",
                SERVICE_ID,
                &healthcheck_envelope(),
            )
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
        fn the_child_runs_on_the_daemons_app_paths() {
            let (_root, paths) = temp_app_paths("stdio-app-paths");
            let engine = FakeEngine::answering("rocmd-app-paths", STOPPED, 0);
            request(&engine, &paths).unwrap();
            assert_eq!(
                engine.env("ROCM_CLI_CONFIG_DIR").as_deref(),
                paths.config_dir.to_str()
            );
            assert_eq!(
                engine.env("ROCM_CLI_DATA_DIR").as_deref(),
                paths.data_dir.to_str()
            );
            assert_eq!(
                engine.env("ROCM_CLI_CACHE_DIR").as_deref(),
                paths.cache_dir.to_str()
            );
        }

        #[test]
        fn a_nonzero_exit_that_still_answered_is_decoded_not_discarded() {
            let (_root, paths) = temp_app_paths("stdio-nonzero");
            let answered = FakeEngine::answering("rocmd-answers-then-fails", STOPPED, 2);
            assert_eq!(
                request(&answered, &paths).unwrap(),
                json!({"stopped": true})
            );

            let failed = FakeEngine::answering(
                "rocmd-error-then-fails",
                r#"{"ok":false,"error":{"code":"stop_failed","message":"it broke"}}"#,
                1,
            );
            assert_eq!(
                request(&failed, &paths).unwrap_err().to_string(),
                "stop_failed: it broke"
            );
        }

        #[test]
        fn a_crash_with_no_answer_reports_the_status_and_stderr() {
            let (_root, paths) = temp_app_paths("stdio-crash");
            let crash = FakeEngine::new("rocmd-crash", true, "echo boom >&2\nexit 3");
            assert_eq!(
                request(&crash, &paths).unwrap_err().to_string(),
                "engine stdio process exited with status exit status: 3: boom"
            );
        }

        #[test]
        fn an_unparseable_answer_carries_the_childs_stderr() {
            let (_root, paths) = temp_app_paths("stdio-garbage");
            let garbage =
                FakeEngine::new("rocmd-garbage", true, "printf 'not json'\necho oops >&2");
            assert_eq!(
                request(&garbage, &paths).unwrap_err().to_string(),
                "failed to parse engine response envelope; stderr: oops"
            );
        }

        /// A request to a child that has already exited is an ordinary error
        /// wherever SIGPIPE stands: ignored (Rust's startup default, which the
        /// test harness keeps) or at its default action, which is how `rocm`
        /// runs and so how `rocm daemon` runs this code. Re-executes this
        /// test in a fresh process for each case, since changing the disposition
        /// process-wide would race sibling tests.
        #[test]
        #[allow(unsafe_code)] // libc::signal, to restore the default action in the child
        fn a_request_to_a_child_that_already_exited_is_an_error_whatever_the_sigpipe_disposition() {
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
                let result = engine_request_via::<Value>(
                    engine.path(),
                    &paths,
                    "vllm",
                    SERVICE_ID,
                    &envelope,
                );
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
                        "engine_ipc::tests::stdio_child::a_request_to_a_child_that_already_exited_is_an_error_whatever_the_sigpipe_disposition",
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
                None,
                "SIGPIPE killed the process instead of surfacing as an error: {default_action:?}"
            );
            assert!(default_action.success(), "{default_action:?}");
        }
    }
}
