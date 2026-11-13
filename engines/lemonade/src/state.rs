// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, format_http_base_url};
use rocm_engine_protocol::{ENGINE_RECIPE_CONTRACT_VERSION, EngineRecipeHint};
use serde_json::{Value, json};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::device_policy::device_policy_name;
use crate::install::{LemonadeInstallManifest, LemonadeRuntime};
use crate::{DEFAULT_HOST, DEFAULT_MODEL, ENGINE_NAME, ServeHttpRequest, current_unix_millis};

/// Maximum bytes to read from the end of a log file when extracting tail lines.
/// Prevents reading entire gigabyte-sized logs on startup timeout.
pub(crate) const MAX_TAIL_READ: u64 = 4 * 1024 * 1024; // 4MB

#[derive(Debug, Clone)]
pub(crate) struct ServiceFiles {
    pub(crate) state_path: PathBuf,
    pub(crate) log_path: PathBuf,
}

/// Whether a real inference request has succeeded against this service.
///
/// Latch and backoff bookkeeping lives in `rocm-core` so both engines share one
/// implementation — what counts as *listed* differs per engine, what counts as
/// *serving* does not.
pub(crate) fn inference_verified(
    state_path: &Path,
    state: Option<&Value>,
    endpoint_url: &str,
    model_ref: &str,
) -> bool {
    let Some((host, port)) = parse_http_endpoint(endpoint_url) else {
        return false;
    };
    rocm_core::engine_state_inference_verified(
        state_path,
        state,
        &format_http_base_url(&host, port),
        model_ref,
        rocm_engine_protocol::resolve_endpoint_api_key().as_deref(),
    )
}

pub(crate) fn write_running_state(
    request: &ServeHttpRequest,
    runtime: &LemonadeRuntime,
    pid: u32,
    server_pid: Option<u32>,
    status: &str,
) -> Result<()> {
    write_state(
        &request.state_path,
        &json!({
            "service_id": request.service_id,
            "engine": ENGINE_NAME,
            "status": status,
            "pid": pid,
            "server_pid": server_pid,
            "model_ref": request.model_ref,
            "canonical_model_id": request.model_ref,
            "host": request.host,
            "port": request.port,
            "endpoint_url": endpoint_url(&request.host, request.port),
            "device_policy": device_policy_name(&request.device_policy),
            "gpu_indices": request.gpu_indices,
            "runtime_id": request.runtime_id.as_deref().unwrap_or(runtime.manifest.env_id.as_str()),
            "env_id": request.env_id.as_deref().unwrap_or(runtime.manifest.env_id.as_str()),
            "runtime_kind": "lemonade_embeddable",
            "runtime_executable": runtime.manifest.lemond,
            "log_path": request.log_path.as_ref().map(|path| path.display().to_string()),
            "engine_recipe": request.engine_recipe,
            "started_at_unix_ms": current_unix_millis(),
            // Kernel start-times captured while each PID is alive. Paired with the
            // PIDs, they identify these exact processes across PID recycling so a
            // later stop never signals a reused PID.
            "start_ticks": rocm_core::process_start_ticks(pid),
            "server_start_ticks": server_pid.and_then(rocm_core::process_start_ticks)
        }),
    )
}

pub(crate) fn service_files(service_id: &str) -> Result<ServiceFiles> {
    let paths = AppPaths::discover()?;
    Ok(ServiceFiles {
        state_path: paths
            .engine_state_dir(ENGINE_NAME)
            .join(format!("{service_id}.json")),
        log_path: paths
            .engine_logs_dir(ENGINE_NAME)
            .join(format!("{service_id}.log")),
    })
}

pub(crate) fn endpoint_url(host: &str, port: u16) -> String {
    format!("{}/v1", format_http_base_url(host, port))
}

pub(crate) fn endpoint_url_from_state(state: &Value) -> Option<String> {
    value_string(state, "endpoint_url").or_else(|| {
        let host = value_string(state, "host")?;
        let port = value_u32(state, "port")?;
        let port = u16::try_from(port).ok()?;
        Some(endpoint_url(&host, port))
    })
}

pub(crate) fn read_service_state(path: &Path) -> Result<Value> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))
}

fn write_state(path: &Path, value: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

pub(crate) fn merge_json_state(path: &Path, patch: &Value) -> Result<()> {
    let mut value = read_service_state(path).unwrap_or_else(|_| json!({}));
    if !value.is_object() {
        value = json!({});
    }
    let object = value.as_object_mut().expect("object checked above");
    if let Some(patch) = patch.as_object() {
        for (key, value) in patch {
            object.insert(key.clone(), value.clone());
        }
    }
    write_state(path, &value)
}

pub(crate) fn mark_json_status(path: &Path, status: &str) -> Result<()> {
    merge_json_state(
        path,
        &json!({
            "engine": ENGINE_NAME,
            "status": status,
            "stopped_at_unix_ms": current_unix_millis(),
        }),
    )
}

pub(crate) fn value_string(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(ToOwned::to_owned)
}

fn value_u32(value: &Value, key: &str) -> Option<u32> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

/// Reconstruct the identity (PID + kernel start-time) of the process a stop
/// should terminate. The server PID is preferred, falling back to the launcher
/// PID, each paired with its recorded start-time. `start_ticks` is absent in
/// pre-existing state files, in which case verification degrades to best-effort.
pub(crate) fn identity_from_state(state: &Value) -> Option<rocm_core::ProcessIdentity> {
    if let Some(server_pid) = value_u32(state, "server_pid") {
        let start_ticks = state.get("server_start_ticks").and_then(Value::as_u64);
        return Some(rocm_core::ProcessIdentity::new(server_pid, start_ticks));
    }
    let pid = value_u32(state, "pid")?;
    let start_ticks = state.get("start_ticks").and_then(Value::as_u64);
    Some(rocm_core::ProcessIdentity::new(pid, start_ticks))
}

/// Stop a process this engine spawned and still holds as an unreaped child, and
/// report whether it is confirmed gone.
///
/// On Unix it sends `SIGTERM` to the process tree and waits up to
/// [`STOP_GRACE`](crate::STOP_GRACE) for every process in it to exit. Under
/// `force` it then escalates to `SIGKILL` and waits up to the grace period again,
/// so the call blocks for up to about twice [`STOP_GRACE`](crate::STOP_GRACE) and
/// the caller's following `child.wait()` returns instead of hanging on a process
/// that ignores `SIGTERM`. With `force == false` a process that outlives the
/// grace period is left running. On Windows there is no graceful signal and no
/// tree walk: a forced stop terminates `pid` itself and waits for it, and an
/// unforced one sends nothing.
///
/// `false` means the stop is not confirmed, and the reason is written to
/// stderr. That includes a process whose identity could not be confirmed
/// ([`rocm_core::TerminationOutcome::Unverified`]): it is not signalled at all,
/// so a caller that then waits on it blocks until it exits by itself.
///
/// The process identity is captured here, at call time, not when the process was
/// spawned. That is safe only while the caller holds an unreaped child for `pid`
/// (a `std::process::Child` not yet waited on): until it is reaped, the PID
/// cannot be handed to another process. A Windows `LemondChild::Pid` holds no
/// process handle, so that guarantee does not cover it: if that `lemond` has
/// already exited, its PID may name another process by the time of this call.
/// Do not call this with a PID read back from a state file; stop recorded
/// processes through `rocm_core::terminate_verified` with their recorded
/// identity instead.
pub(crate) fn terminate_pid(pid: u32, force: bool) -> bool {
    terminate_pid_within(pid, force, crate::STOP_GRACE)
}

/// [`terminate_pid`] with the grace period chosen by the caller, so a test can
/// drive the escalation without waiting the production grace period.
fn terminate_pid_within(pid: u32, force: bool, grace: Duration) -> bool {
    // `Tree`, unlike the `Single` scope the recorded-state stop uses
    // (`crate::STOP_SCOPE`): the children stopped here are often `lemond`, the
    // Lemonade router, which runs each loaded model in a `llama-server` child
    // process. A forced kill of `lemond` alone would orphan that child, still
    // holding the GPU. A child with no children of its own (a direct
    // `llama-server`) is stopped exactly as `Single` would stop it.
    let outcome = rocm_core::terminate_verified(
        &rocm_core::ProcessIdentity::capture(pid),
        rocm_core::KillScope::Tree,
        grace,
        force,
    );
    if let Some(reason) = outcome.unconfirmed_reason() {
        eprintln!("Warning: process {pid} was not confirmed stopped: {reason}.");
    }
    outcome.stopped()
}

pub(crate) fn free_local_port() -> Result<u16> {
    let listener = TcpListener::bind((DEFAULT_HOST, 0)).context("failed to reserve local port")?;
    Ok(listener.local_addr()?.port())
}

pub(crate) fn parse_http_endpoint(endpoint_url: &str) -> Option<(String, u16)> {
    let without_scheme = endpoint_url.trim().strip_prefix("http://")?;
    let authority = without_scheme.split('/').next()?.trim();
    if let Some(rest) = authority.strip_prefix('[') {
        let end = rest.find(']')?;
        let host = rest[..end].to_owned();
        let port = rest[end + 1..].strip_prefix(':')?.parse().ok()?;
        return Some((host, port));
    }
    let (host, port) = authority.rsplit_once(':')?;
    Some((host.to_owned(), port.parse().ok()?))
}

pub(crate) fn accepted_engine_recipe(
    engine_recipe: Option<EngineRecipeHint>,
) -> Result<Option<EngineRecipeHint>> {
    if let Some(hint) = &engine_recipe {
        if hint.engine != ENGINE_NAME {
            bail!(
                "engine_recipe target `{}` does not match adapter `{}`",
                hint.engine,
                ENGINE_NAME
            );
        }
        if hint.contract_version != ENGINE_RECIPE_CONTRACT_VERSION {
            bail!(
                "engine_recipe contract `{}` is unsupported; expected `{}`",
                hint.contract_version,
                ENGINE_RECIPE_CONTRACT_VERSION
            );
        }
    }
    Ok(engine_recipe)
}

pub(crate) fn parse_engine_recipe_json(value: Option<String>) -> Result<Option<EngineRecipeHint>> {
    value
        .map(|text| {
            serde_json::from_str::<EngineRecipeHint>(&text)
                .context("failed to parse engine recipe JSON")
        })
        .transpose()
        .and_then(accepted_engine_recipe)
}

pub(crate) fn resolve_lemonade_model_ref(model_ref: &str) -> String {
    let trimmed = model_ref.trim();
    let lower = trimmed.to_ascii_lowercase();
    // Only exact, recognized shorthand aliases map to the bundled assistant GGUF.
    // A syntactically valid `owner/repo` Hugging Face checkpoint must never be
    // silently rewritten here — e.g. `Qwen/Qwen2.5-1.5B-Instruct` is a real
    // checkpoint id, not a shorthand, and is passed through unchanged so the
    // serve path can honour (or explicitly reject) it rather than substituting
    // an unrelated model behind the user's back.
    if trimmed.is_empty()
        || matches!(
            lower.as_str(),
            "qwen"
                | "assistant"
                | "default"
                | "small"
                | "lemonade-qwen"
                | "qwen-gguf"
                | "qwen3-4b"
                | "qwen3-4b-instruct"
                | "qwen3-4b-instruct-2507-gguf"
        )
    {
        DEFAULT_MODEL.to_owned()
    } else if matches!(
        lower.as_str(),
        "tiny" | "qwen-smoke" | "lemonade-tiny" | "qwen3-0.6b-gguf"
    ) {
        "Qwen3-0.6B-GGUF".to_owned()
    } else {
        trimmed.to_owned()
    }
}

pub(crate) fn manifest_lock_hash(manifest: &LemonadeInstallManifest) -> String {
    let mut hasher = DefaultHasher::new();
    manifest.env_id.hash(&mut hasher);
    manifest.version.hash(&mut hasher);
    manifest.runtime_dir.hash(&mut hasher);
    manifest.lemond.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::scratch_dir;
    use std::io::{Read, Write};
    use std::time::Duration;

    #[test]
    fn identity_from_state_prefers_server_pid_and_its_start_ticks() {
        let state = json!({
            "pid": 100,
            "start_ticks": 111_u64,
            "server_pid": 200,
            "server_start_ticks": 222_u64,
        });
        let identity = identity_from_state(&state).expect("identity");
        assert_eq!(identity.pid, 200);
        assert_eq!(identity.start_ticks, Some(222));
    }

    #[test]
    fn identity_from_state_falls_back_to_launcher_pid() {
        let state = json!({ "pid": 100, "start_ticks": 111_u64 });
        let identity = identity_from_state(&state).expect("identity");
        assert_eq!(identity.pid, 100);
        assert_eq!(identity.start_ticks, Some(111));
    }

    #[test]
    fn identity_from_legacy_state_has_no_start_ticks() {
        // Pre-existing state files carry only PIDs; verification must degrade.
        let state = json!({ "server_pid": 200 });
        let identity = identity_from_state(&state).expect("identity");
        assert_eq!(identity.pid, 200);
        assert_eq!(identity.start_ticks, None);
    }

    #[test]
    fn identity_from_state_without_any_pid_is_none() {
        assert!(identity_from_state(&json!({ "status": "running" })).is_none());
    }

    #[test]
    fn qwen_alias_resolves_to_validated_assistant_gguf_model() {
        assert_eq!(resolve_lemonade_model_ref("qwen"), DEFAULT_MODEL);
        assert_eq!(resolve_lemonade_model_ref("qwen-smoke"), "Qwen3-0.6B-GGUF");
    }

    #[test]
    fn canonical_hugging_face_name_passes_through_unchanged() {
        assert_eq!(
            resolve_lemonade_model_ref("LiquidAI/LFM2.5-230M-GGUF:Q4_0"),
            "LiquidAI/LFM2.5-230M-GGUF:Q4_0"
        );
        // Regression (EAI-7370): a fully-qualified checkpoint id that merely
        // contains a known-alias substring must not be swapped for the bundled
        // default. `Qwen/Qwen2.5-1.5B-Instruct` was silently served as
        // `Qwen3-4B-Instruct-2507-GGUF`; it must now pass through verbatim.
        assert_eq!(
            resolve_lemonade_model_ref("Qwen/Qwen2.5-1.5B-Instruct"),
            "Qwen/Qwen2.5-1.5B-Instruct"
        );
    }

    /// Answer `count` chat requests on a loopback port with the given status,
    /// recording how many arrived.
    fn spawn_chat_endpoint(
        status_line: &'static str,
        count: usize,
    ) -> (u16, std::thread::JoinHandle<usize>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let port = listener.local_addr().expect("local addr").port();
        let handle = std::thread::spawn(move || {
            let mut served = 0;
            for _ in 0..count {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut buffer = [0_u8; 1024];
                let _ = stream.read(&mut buffer);
                let body = r#"{"choices":[{"message":{"content":"ok"}}]}"#;
                let _ = write!(
                    stream,
                    "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                served += 1;
            }
            served
        });
        (port, handle)
    }

    #[test]
    fn inference_verification_latches_into_the_state_file() {
        // First check probes and records the verdict; the second reads the latch
        // and leaves the model alone.
        let (port, server) = spawn_chat_endpoint("HTTP/1.1 200 OK", 1);
        let dir = scratch_dir("inference-latch");
        let state_path = dir.join("state.json");
        fs::write(&state_path, json!({"status": "ready"}).to_string()).expect("seed state");
        let endpoint = format_http_base_url("127.0.0.1", port);

        let state = read_service_state(&state_path).ok();
        assert!(inference_verified(
            &state_path,
            state.as_ref(),
            &endpoint,
            DEFAULT_MODEL
        ));

        let state = read_service_state(&state_path).expect("state readable");
        assert!(
            state
                .get(rocm_core::INFERENCE_VERIFIED_STATE_KEY)
                .and_then(Value::as_u64)
                .is_some(),
            "a passing probe is latched so later healthchecks skip it"
        );
        assert!(inference_verified(
            &state_path,
            Some(&state),
            &endpoint,
            DEFAULT_MODEL
        ));

        assert_eq!(
            server.join().expect("server thread"),
            1,
            "the latched check must not send a second inference request"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn inference_verification_withheld_while_the_endpoint_cannot_serve() {
        // The reported failure: the model is listed but inference still 5xxs.
        let (port, server) = spawn_chat_endpoint("HTTP/1.1 503 Service Unavailable", 1);
        let dir = scratch_dir("inference-unverified");
        let state_path = dir.join("state.json");
        fs::write(&state_path, json!({"status": "ready"}).to_string()).expect("seed state");
        let endpoint = format_http_base_url("127.0.0.1", port);

        let state = read_service_state(&state_path).ok();
        assert!(!inference_verified(
            &state_path,
            state.as_ref(),
            &endpoint,
            DEFAULT_MODEL
        ));

        let state = read_service_state(&state_path).expect("state readable");
        assert!(
            state.get(rocm_core::INFERENCE_VERIFIED_STATE_KEY).is_none(),
            "nothing is latched until inference actually answers"
        );

        server.join().expect("server thread");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn endpoint_parser_supports_ipv6_loopback() {
        assert_eq!(
            parse_http_endpoint("http://[::1]:11435/v1"),
            Some(("::1".to_owned(), 11435))
        );
    }

    // ---- `terminate_pid`: bounded termination of a spawned child. ----

    /// A grace period short enough to keep these tests fast, and long enough
    /// for a cooperative `sh` to act on `SIGTERM`.
    #[cfg(target_os = "linux")]
    const TEST_GRACE: Duration = Duration::from_millis(500);

    /// Spawn `sh -c script`, which must print one line once it is ready (for a
    /// trapping script: after the trap is installed, so `SIGTERM` can never
    /// arrive first), and return the child with that line.
    #[cfg(target_os = "linux")]
    fn spawn_ready(script: &str) -> (std::process::Child, String) {
        use std::io::BufRead;
        let mut child = std::process::Command::new("sh")
            .args(["-c", script])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            // Not inherited: a child left running by a failing test must not hold
            // open the stderr pipe a parent test reads to its end.
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn test child");
        let stdout = child.stdout.take().expect("piped stdout");
        let mut line = String::new();
        std::io::BufReader::new(stdout)
            .read_line(&mut line)
            .expect("read readiness line");
        (child, line.trim().to_owned())
    }

    /// `SIGKILL` `identity` if it is still that process, so a failing test does
    /// not leave a stray process behind.
    #[cfg(target_os = "linux")]
    fn kill_if_still_running(identity: &rocm_core::ProcessIdentity) {
        if rocm_core::identity_state(identity) == rocm_core::IdentityState::Matches {
            let _ = std::process::Command::new("kill")
                .args(["-KILL", &identity.pid.to_string()])
                .status();
        }
    }

    /// `SIGKILL`s each process, if it is still that process, when dropped.
    #[cfg(target_os = "linux")]
    struct KillOnDrop(Vec<rocm_core::ProcessIdentity>);

    #[cfg(target_os = "linux")]
    impl Drop for KillOnDrop {
        fn drop(&mut self) {
            self.0.iter().for_each(kill_if_still_running);
        }
    }

    #[cfg(target_os = "linux")]
    const TRAP_TERM: &str = "trap '' TERM; echo ready; while :; do sleep 1; done";

    #[cfg(target_os = "linux")]
    #[test]
    fn forced_terminate_pid_kills_a_child_that_ignores_sigterm() {
        use std::os::unix::process::ExitStatusExt;
        let (mut child, _) = spawn_ready(TRAP_TERM);
        let identity = rocm_core::ProcessIdentity::capture(child.id());

        let stopped = terminate_pid_within(child.id(), true, TEST_GRACE);

        let status = child.try_wait().expect("try_wait");
        if status.is_none() {
            kill_if_still_running(&identity);
            let _ = child.wait();
        }
        assert!(stopped, "a forced stop must report the child stopped");
        let status = status.expect("the child must have exited after a forced stop");
        assert_eq!(
            status.signal(),
            Some(libc::SIGKILL),
            "a child that ignores SIGTERM can only have been stopped by SIGKILL: {status}"
        );
    }

    /// The pattern every caller uses, `terminate_pid` then `child.wait()`, must
    /// return within about twice the grace period for a child that ignores
    /// `SIGTERM`. Before the fix this hung forever.
    #[cfg(target_os = "linux")]
    #[test]
    fn terminate_then_wait_returns_for_a_child_that_ignores_sigterm() {
        let (mut child, _) = spawn_ready(TRAP_TERM);
        let identity = rocm_core::ProcessIdentity::capture(child.id());
        let (done, finished) = std::sync::mpsc::channel();
        let teardown = std::thread::spawn(move || {
            let stopped = terminate_pid_within(child.id(), true, TEST_GRACE);
            let status = child.wait();
            let _ = done.send((stopped, status.is_ok()));
        });

        // The margin is wide: the bound only turns a hang into a failure, and a
        // loaded host can be slow to run the stop.
        let bound = TEST_GRACE * 2 + Duration::from_secs(10);
        let result = finished.recv_timeout(bound);
        if result.is_err() {
            // Unblock the teardown thread's `wait` so the test fails rather than hangs.
            kill_if_still_running(&identity);
        }
        let _ = teardown.join();
        let (stopped, waited) =
            result.unwrap_or_else(|_| panic!("terminate + wait did not return within {bound:?}"));
        assert!(stopped);
        assert!(waited);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn unforced_terminate_pid_leaves_a_child_that_ignores_sigterm_running() {
        let (mut child, _) = spawn_ready(TRAP_TERM);
        let identity = rocm_core::ProcessIdentity::capture(child.id());

        let stopped = terminate_pid_within(child.id(), false, TEST_GRACE);
        let still_running = child.try_wait().expect("try_wait").is_none();

        kill_if_still_running(&identity);
        let _ = child.wait();
        assert!(
            !stopped,
            "an unforced stop must not claim a surviving child stopped"
        );
        assert!(still_running, "an unforced stop must not kill the child");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn terminate_pid_stops_a_cooperative_child_with_sigterm() {
        use std::os::unix::process::ExitStatusExt;
        let (mut child, _) = spawn_ready("echo ready; while :; do sleep 1; done");
        let identity = rocm_core::ProcessIdentity::capture(child.id());

        let stopped = terminate_pid_within(child.id(), true, Duration::from_secs(5));

        kill_if_still_running(&identity);
        let status = child.wait().expect("wait");
        assert!(stopped);
        // Exited on SIGTERM, not SIGKILL: the stop did not escalate. Asserted on
        // the signal rather than on timing, which a loaded host makes unreliable.
        assert_eq!(status.signal(), Some(libc::SIGTERM), "{status}");
    }

    /// The warning `terminate_pid` writes is checked on the real stderr of a
    /// process running it, together with the state it reports: a child that
    /// outlives an unforced stop is named, and is still running; a child a
    /// forced stop killed is not named. The test re-runs itself in a child
    /// process (selected by an environment variable), because a warning written
    /// to this process's stderr cannot be read back here.
    #[cfg(target_os = "linux")]
    #[test]
    fn terminate_pid_reports_an_unconfirmed_stop_on_stderr_and_not_a_confirmed_one() {
        const ARM: &str = "ROCM_TEST_LEMONADE_STOP_WARNING_ARM";
        if std::env::var_os(ARM).is_some() {
            let (mut forced, _) = spawn_ready(TRAP_TERM);
            let (mut unforced, _) = spawn_ready(TRAP_TERM);
            let forced_identity = rocm_core::ProcessIdentity::capture(forced.id());
            let unforced_identity = rocm_core::ProcessIdentity::capture(unforced.id());
            // Kills both if this arm panics before its own cleanup below.
            let _cleanup = KillOnDrop(vec![forced_identity, unforced_identity]);
            let forced_stopped = terminate_pid_within(forced.id(), true, TEST_GRACE);
            let unforced_stopped = terminate_pid_within(unforced.id(), false, TEST_GRACE);
            let running = unforced.try_wait().expect("try_wait").is_none();
            // On a line of its own: the harness has already printed the test's
            // name without a line break.
            println!("\nforced={} stopped={forced_stopped}", forced.id());
            println!(
                "unforced={} stopped={unforced_stopped} running={running}",
                unforced.id()
            );
            kill_if_still_running(&forced_identity);
            kill_if_still_running(&unforced_identity);
            let _ = unforced.wait();
            let _ = forced.wait();
            return;
        }

        let output = std::process::Command::new(
            std::env::current_exe().expect("current test executable"),
        )
        .args([
            "state::tests::terminate_pid_reports_an_unconfirmed_stop_on_stderr_and_not_a_confirmed_one",
            "--exact",
            "--nocapture",
        ])
        .env(ARM, "1")
        .stdin(std::process::Stdio::null())
        .output()
        .expect("re-run the test arm in a child process");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{stdout}\n{stderr}");
        let pid_after = |prefix: &str| -> u32 {
            stdout
                .lines()
                .find_map(|line| line.strip_prefix(prefix))
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|pid| pid.parse().ok())
                .unwrap_or_else(|| panic!("no {prefix} line in: {stdout}"))
        };
        let forced = pid_after("forced=");
        let unforced = pid_after("unforced=");

        assert!(
            stdout.contains(&format!("forced={forced} stopped=true")),
            "{stdout}"
        );
        assert!(
            stdout.contains(&format!("unforced={unforced} stopped=false running=true")),
            "{stdout}"
        );
        assert!(
            stderr.contains(&format!(
                "Warning: process {unforced} was not confirmed stopped: it, or a process under it, \
                 was still running when the stop ended."
            )),
            "{stderr}"
        );
        assert!(
            !stderr.contains(&format!("process {forced} was not confirmed stopped")),
            "a confirmed stop must not warn: {stderr}"
        );
    }

    /// `lemond` runs models in `llama-server` children; a forced stop must take
    /// those down too, not orphan them.
    #[cfg(target_os = "linux")]
    #[test]
    fn forced_terminate_pid_kills_a_grandchild_that_ignores_sigterm() {
        // Both shells ignore SIGTERM (`SIG_IGN` survives `exec`); the inner one
        // prints its PID once it is running, after the outer trap is installed.
        let (mut child, grandchild_pid) =
            spawn_ready(r#"trap '' TERM; sh -c "echo \$\$; exec sleep 300" & wait"#);
        let identity = rocm_core::ProcessIdentity::capture(child.id());
        let grandchild =
            rocm_core::ProcessIdentity::capture(grandchild_pid.parse().expect("grandchild pid"));
        assert_eq!(
            rocm_core::identity_state(&grandchild),
            rocm_core::IdentityState::Matches,
            "grandchild should be running before the stop"
        );

        let stopped = terminate_pid_within(child.id(), true, TEST_GRACE);
        let grandchild_state = rocm_core::identity_state(&grandchild);

        kill_if_still_running(&grandchild);
        kill_if_still_running(&identity);
        let _ = child.wait();
        assert!(stopped);
        assert!(
            matches!(
                grandchild_state,
                rocm_core::IdentityState::Gone | rocm_core::IdentityState::Recycled
            ),
            "a forced stop must kill the child's whole tree, got {grandchild_state:?}"
        );
    }

    /// `terminate_pid` itself, as every caller runs it: the production grace
    /// period passes before the escalation, the child is then killed, and the
    /// caller's `wait` returns within about twice that period. Takes about
    /// [`STOP_GRACE`](crate::STOP_GRACE).
    #[cfg(target_os = "linux")]
    #[test]
    fn terminate_pid_escalates_after_the_stop_grace_and_the_callers_wait_returns() {
        use std::os::unix::process::ExitStatusExt;
        let (mut child, _) = spawn_ready(TRAP_TERM);
        let identity = rocm_core::ProcessIdentity::capture(child.id());
        let (done, finished) = std::sync::mpsc::channel();
        let teardown = std::thread::spawn(move || {
            let started = std::time::Instant::now();
            let stopped = terminate_pid(child.id(), true);
            let status = child.wait();
            let _ = done.send((stopped, status.ok(), started.elapsed()));
        });

        let bound = crate::STOP_GRACE * 2 + Duration::from_secs(5);
        let result = finished.recv_timeout(bound);
        if result.is_err() {
            // Unblock the teardown thread's `wait` so the test fails rather than hangs.
            kill_if_still_running(&identity);
        }
        let _ = teardown.join();
        let (stopped, status, elapsed) = result
            .unwrap_or_else(|_| panic!("terminate_pid + wait did not return within {bound:?}"));
        assert!(stopped, "a forced stop must report the child stopped");
        assert!(
            elapsed >= crate::STOP_GRACE,
            "the child must get the whole stop grace before SIGKILL: {elapsed:?}"
        );
        let status = status.expect("wait");
        assert_eq!(status.signal(), Some(libc::SIGKILL), "{status}");
    }

    #[cfg(windows)]
    #[test]
    fn forced_terminate_pid_stops_a_long_running_child_and_wait_returns() {
        let mut child = std::process::Command::new(rocm_core::windows_system_tool("ping.exe"))
            .args(["-n", "300", "127.0.0.1"])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .spawn()
            .expect("spawn test child");
        let pid = child.id();
        let (done, finished) = std::sync::mpsc::channel();
        let teardown = std::thread::spawn(move || {
            let stopped = terminate_pid(pid, true);
            let _ = done.send(stopped);
            child
        });

        let bound = crate::STOP_GRACE * 2 + Duration::from_secs(5);
        let result = finished.recv_timeout(bound);
        if result.is_err() {
            // Unblock the teardown thread so the test fails rather than hangs.
            let _ = rocm_core::terminate_process(pid);
        }
        let mut child = teardown.join().expect("teardown thread");
        let exited = child.try_wait().expect("try_wait").is_some();
        if !exited {
            let _ = child.kill();
        }
        let _ = child.wait();
        assert_eq!(
            result,
            Ok(true),
            "terminate_pid must stop the child within {bound:?}"
        );
        assert!(exited, "the child must have exited after terminate_pid");
    }
}
