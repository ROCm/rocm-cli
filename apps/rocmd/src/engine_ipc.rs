// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use crate::daemon_host::DaemonHost;
use anyhow::Result;
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
    engine_healthcheck_response_on(&DaemonHost::detect()?, paths, engine, service_id)
}

/// [`engine_healthcheck_response`] on a given host. The supervisor polls
/// through the host it started its engine on, so a test can supply both.
pub(crate) fn engine_healthcheck_response_on(
    daemon_host: &DaemonHost,
    paths: &AppPaths,
    engine: &str,
    service_id: &str,
) -> Result<HealthcheckResponse> {
    engine_request_on::<_, HealthcheckResponse>(
        daemon_host,
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

/// Re-thread the service's endpoint key file onto an engine child's
/// environment, mirroring the initial `rocm serve` spawn. A service with no
/// usable stored key leaves the command's environment untouched, which is
/// right only for a service that does not need one: a loopback bind that did
/// not ask for `--require-api-key`, the unauthenticated default.
///
/// Returns whether a key was applied. Callers that spawn a *listener* must
/// check it against the service's record — a service that needs a key (a public
/// bind, or one launched with `--require-api-key`) and has none would come up
/// anonymous (see [`rocm_core::ensure_public_service_has_endpoint_key`], which
/// refuses it). Callers that only make a stdio plugin call can ignore it.
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

/// Whether the service has a *usable* endpoint key on disk: a file that exists
/// and holds a key the engine adapters will enforce. This is the one answer
/// every recovery path gives the respawn guard as its `key_present`, so the
/// supervisor, the recovery watcher and the spawn-site check cannot disagree
/// about what "has a key" means.
pub(crate) fn has_usable_endpoint_key(paths: &AppPaths, service_id: &str) -> bool {
    usable_endpoint_key_file(paths, service_id).is_some()
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
/// ([`DaemonHost::rocm_binary`] says which executable that is).
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
    engine_request_on(
        &DaemonHost::detect()?,
        paths,
        engine,
        service_id,
        method,
        request,
    )
}

/// [`engine_request`] on a given host, so a test can supply the binary that
/// answers. The stdio command is `rocm`'s alone, so the daemon runs it on the
/// `rocm` binary, not on whatever it is itself.
fn engine_request_on<T, R>(
    daemon_host: &DaemonHost,
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
    let engine_binary = daemon_host.rocm_binary()?;
    engine_request_via(&engine_binary, paths, engine, service_id, &envelope)
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
        note_failing_exit_after_answer: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_app_paths;
    use std::fs;

    #[test]
    fn a_key_file_is_a_key_only_when_the_engine_adapters_would_enforce_it() {
        // The respawn guard's `key_present` is this answer. The engine adapters
        // enforce nothing for an empty, blank or control-character key file, so
        // counting any of them as "has a key" would let the guard pass while the
        // endpoint served anonymously.
        let (_root, paths) = temp_app_paths("has-usable-endpoint-key");
        let service_id = "svc-usable-key";
        let key_path = rocm_engine_protocol::endpoint_key_file_path(&paths, service_id);
        fs::create_dir_all(paths.services_dir()).unwrap();

        assert!(!has_usable_endpoint_key(&paths, service_id), "no file");
        for (contents, usable) in [
            ("", false),
            ("  \n", false),
            ("bad\r\nkey", false),
            ("a-usable-key", true),
            ("a-usable-key\n", true),
        ] {
            fs::write(&key_path, contents).unwrap();
            assert_eq!(
                has_usable_endpoint_key(&paths, service_id),
                usable,
                "{contents:?}"
            );
            // The key an engine child is given is the same file, so what recovery
            // reports as present is exactly what it threads onto the child.
            let mut command = ProcessCommand::new("true");
            assert_eq!(
                apply_endpoint_key_env(&mut command, &paths, service_id),
                usable,
                "{contents:?}"
            );
        }
        // Another service's key is not this one's.
        assert!(!has_usable_endpoint_key(&paths, "svc-other"));
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
        // A daemon's stderr is its log, so a failing exit after an answer leaves a note there.
        assert!(options.note_failing_exit_after_answer);
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
                "engine stdio process exited with code 3: boom"
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

    /// The stdio command is `rocm`'s alone: a daemon run as `rocmd` asks the
    /// engine through the `rocm` beside it, which `rocmd` answered with an
    /// unrecognised-subcommand error, so its readiness checks and its engine
    /// stops could never succeed.
    ///
    /// The stand-in `rocm` records its arguments, reads the request line and
    /// answers with an `ok` envelope.
    #[cfg(target_os = "linux")]
    #[test]
    fn engine_request_runs_the_stdio_command_on_the_rocm_beside_rocmd() -> Result<()> {
        use crate::test_support::{lock_script_tests, script_tools, write_executable_script};

        let _serial = lock_script_tests();
        let tools = script_tools();
        let (root, paths) = temp_app_paths("engine-request-on-rocm");
        let seen = root.join("rocm-args.txt");
        write_executable_script(
            &root.join("rocm"),
            &tools,
            &format!(
                "printf '%s\\n' \"$@\" > '{}'\nread -r _request\n{}\n",
                seen.display(),
                r#"printf '%s\n' '{"ok":true,"data":{"status":"ready"}}'"#
            ),
        )?;
        let daemon_host = DaemonHost::new(root.join("rocmd"), false);

        // The first exec of a new file can lose a race with an unrelated fork
        // (ETXTBSY); the spawn then fails at once, and one more try is enough.
        let mut outcome = Err(anyhow::anyhow!("not attempted"));
        for _ in 0..10 {
            outcome = engine_request_on::<_, serde_json::Value>(
                &daemon_host,
                &paths,
                "vllm",
                "svc-engine-request",
                EngineMethod::Healthcheck,
                &serde_json::json!({ "service_id": "svc-engine-request" }),
            );
            match &outcome {
                Err(error) if format!("{error:#}").contains("Text file busy") => {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
                _ => break,
            }
        }
        let args = fs::read_to_string(&seen).unwrap_or_default();
        fs::remove_dir_all(&root).ok();

        assert_eq!(
            args.lines().collect::<Vec<_>>(),
            ["__engine-stdio", "vllm"],
            "the engine must be asked through `rocm __engine-stdio <engine>`"
        );
        assert_eq!(outcome?, serde_json::json!({ "status": "ready" }));
        Ok(())
    }

    /// With no `rocm` beside `rocmd` the request fails, and says what it looked
    /// for, instead of running a command `rocmd` does not have.
    #[test]
    fn engine_request_without_a_rocm_beside_rocmd_names_it() {
        let (root, paths) = temp_app_paths("engine-request-no-rocm");
        let error = engine_request_on::<_, serde_json::Value>(
            &DaemonHost::new(root.join("rocmd"), false),
            &paths,
            "vllm",
            "svc-engine-request",
            EngineMethod::Healthcheck,
            &serde_json::json!({ "service_id": "svc-engine-request" }),
        )
        .expect_err("no rocm to ask");
        fs::remove_dir_all(root).ok();

        let text = format!("{error:#}");
        assert!(text.contains("no engine commands"), "{text}");
        assert!(text.contains("unable to locate sibling binary"), "{text}");
    }
}
