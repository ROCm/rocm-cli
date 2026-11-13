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
use std::process::Child;

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
    server: Option<rocm_core::ProcessIdentity>,
    status: &str,
) -> Result<()> {
    write_state(
        &request.state_path,
        &json!({
            "service_id": request.service_id,
            "engine": ENGINE_NAME,
            "status": status,
            "pid": pid,
            "server_pid": server.map(|server| server.pid),
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
            // Start-time tokens that, paired with the PIDs, identify these exact
            // processes across PID recycling so a later stop never signals a
            // reused PID: this wrapper's own, read now while it runs, and the
            // server's, captured once when it was spawned.
            "start_ticks": rocm_core::process_start_ticks(pid),
            "server_start_ticks": server.and_then(|server| server.start_ticks)
        }),
    )
}

/// The keys that record a server as ready: the server's identity and backend,
/// as an object the caller adds its own keys to before merging it. The
/// identity is the one captured when the server was spawned, so its start
/// token is written as captured, not read again from a pid that may have
/// changed hands since.
pub(crate) fn ready_state_patch(
    server: rocm_core::ProcessIdentity,
    backend: &str,
) -> serde_json::Map<String, Value> {
    let mut patch = serde_json::Map::new();
    patch.insert("status".to_owned(), json!("ready"));
    patch.insert("server_pid".to_owned(), json!(server.pid));
    patch.insert("server_start_ticks".to_owned(), json!(server.start_ticks));
    patch.insert("backend_state".to_owned(), json!("ready"));
    patch.insert("backend_requested".to_owned(), json!(backend));
    patch
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

/// Replace the service's state file whole. Shares the state directory's lock
/// with every merge (`rocm_core::write_json_state_file`), so a full rewrite
/// never lands inside another writer's merge, and publishes atomically, so a
/// lock-free reader such as `rocm services list` never sees half a document.
fn write_state(path: &Path, value: &Value) -> Result<()> {
    rocm_core::write_json_state_file(path, value)
}

/// Merge `patch` into the service's state file under the lock the healthcheck's
/// inference latch and `rocmd`'s endpoint publish also take
/// (`rocm_core::merge_json_state_file`), so no merge drops a key another merge
/// wrote — `server_pid` above all, which a later stop needs. A full rewrite
/// ([`write_state`]) is ordered by the same lock but replaces every key.
///
/// A state file that is empty, torn or not a JSON object is an error and left
/// as it was.
pub(crate) fn merge_json_state(path: &Path, patch: &Value) -> Result<()> {
    rocm_core::merge_json_state_file(path, patch)
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

pub(crate) fn terminate_pid(pid: u32, _force: bool) -> bool {
    rocm_core::terminate_process(pid).is_ok()
}

/// A server process the wrapper started, which [`record_server_or_stop_it`]
/// reaps after it has stopped it.
pub(crate) trait StartedServer {
    /// Collect the exit of a server that is known to be gone. Called only once
    /// a stop has confirmed that, so for a child of this process it returns at
    /// once.
    fn reap(&mut self) -> Result<()>;

    /// Whether the server is a child of this process that has not been reaped
    /// yet, so its pid cannot be given to another process before it is.
    fn is_unreaped_child(&self) -> bool;
}

impl StartedServer for Child {
    fn reap(&mut self) -> Result<()> {
        self.wait().map(|_| ()).map_err(Into::into)
    }

    fn is_unreaped_child(&self) -> bool {
        true
    }
}

/// Run `record`, the state write that names a server the wrapper has
/// started; when it fails, stop the server ([`stop_unrecorded_server`]) before
/// returning the error.
///
/// The wrapper waits on its server only after recording it, so returning the
/// error alone would leave a server running that no state file names, which
/// `rocm services stop` could then not find.
///
/// What follows the stop depends on its outcome, and the error says which:
/// - stopped by this call (`graceful`, `forced`) or found already exited
///   (`already_gone`): an existing state file is marked `failed`, best effort
///   (the write that just failed may fail again, and a state file that was
///   never written is not created just to say so), then the server is reaped;
/// - its pid now names another process (`identity_mismatch`): the server
///   exited earlier, so the state file is marked `failed` the same way, and
///   nothing is reaped or signalled, since the pid is no longer the server's;
/// - anything else: the stop could not confirm that the server and the
///   processes under it are gone, so nothing is marked or reaped and the
///   error says so.
pub(crate) fn record_server_or_stop_it(
    server: &mut impl StartedServer,
    identity: rocm_core::ProcessIdentity,
    state_path: &Path,
    record: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let stop = UnrecordedServerStop::for_server(server);
    record_server_or_stop_it_with(server, identity, state_path, record, |identity| {
        stop_unrecorded_server(identity, stop)
    })
}

/// How [`record_server_or_stop_it`] stops a server it could not record:
/// whether the server is this process's unreaped child, which decides whether
/// it may be stopped without a start token, and how long each step waits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UnrecordedServerStop {
    owned: bool,
    grace: std::time::Duration,
}

impl UnrecordedServerStop {
    fn for_server(server: &impl StartedServer) -> Self {
        Self {
            owned: server.is_unreaped_child(),
            grace: crate::STOP_GRACE,
        }
    }
}

/// Stop a server the wrapper started but could not record, together with the
/// processes under it, through the routine `rocm`'s failed managed-launch
/// cleanup also uses (`rocm_core::terminate_recorded_pid_within`).
///
/// The stop is forced and bounded: `SIGTERM`, then `SIGKILL` once `stop.grace`
/// has passed, then up to `stop.grace` again for the exit, so a server that ignores
/// `SIGTERM` (mid model-load, say) cannot hold the wrapper forever. On Linux
/// the processes under the server are read from `/proc` when the stop begins
/// and signalled with it; a server that is already gone when the stop begins
/// is not signalled, and neither is anything it left running. On Windows,
/// which has no `SIGTERM`, the routine holds the pid, takes the processes
/// under the server down with `taskkill /T`, then kills the server, and
/// reports the stop unconfirmed when that tree kill did not complete.
///
/// Unlike Lemonade's engine `Stop`, whose `STOP_SCOPE` signals only the recorded
/// identity, this stop reaches the whole tree: nothing records the processes
/// under a server whose own record failed, so no later stop could find them.
///
/// An identity with no start token is refused (`unverified`) on the hosts that
/// record one, as the managed-launch cleanup refuses it, unless the server is
/// `stop.owned`, an unreaped child of this process. Without a token nothing shows
/// that the pid still names the server: it may have exited before its token
/// could be read and the pid passed to another process, and a forced tree kill
/// of that process could not be undone. An unreaped child's pid cannot pass to
/// another process, so it is stopped by pid alone.
fn stop_unrecorded_server(
    identity: rocm_core::ProcessIdentity,
    stop: UnrecordedServerStop,
) -> rocm_core::TerminationOutcome {
    if identity.start_ticks.is_none() && !stop.owned && cfg!(any(target_os = "linux", windows)) {
        return rocm_core::TerminationOutcome::Unverified;
    }
    rocm_core::terminate_recorded_pid_within(&identity, stop.grace)
}

/// [`record_server_or_stop_it`] with the stop supplied by the caller, so a
/// test can drive every outcome a stop can report.
fn record_server_or_stop_it_with(
    server: &mut impl StartedServer,
    identity: rocm_core::ProcessIdentity,
    state_path: &Path,
    record: impl FnOnce() -> Result<()>,
    stop: impl FnOnce(rocm_core::ProcessIdentity) -> rocm_core::TerminationOutcome,
) -> Result<()> {
    use rocm_core::TerminationOutcome as Outcome;

    let Err(error) = record() else {
        return Ok(());
    };
    let pid = identity.pid;
    let stopped = stop(identity);
    let (gone, reap, what) = match stopped {
        Outcome::Graceful | Outcome::Forced => (true, true, "so stopped it".to_owned()),
        Outcome::AlreadyGone => (true, true, "and it had already exited".to_owned()),
        Outcome::IdentityMismatch => (
            true,
            false,
            "and it had already exited; its pid now names another process, which was left \
             alone"
                .to_owned(),
        ),
        Outcome::TimedOut
        | Outcome::Unverified
        | Outcome::DescendantsUnconfirmed
        | Outcome::ExitedBeforeTreeKill => (
            false,
            false,
            format!(
                "and could not confirm that it and the processes under it stopped ({}), so \
                 they may still be running",
                stopped.as_str()
            ),
        ),
    };
    // Only into a file that exists: merging would otherwise create one holding
    // nothing but the status.
    if gone && state_path.exists() {
        let _ = mark_json_status(state_path, "failed");
    }
    let what = match reap.then(|| server.reap()) {
        Some(Err(reap_error)) => format!("{what}, but reaping it failed: {reap_error:#}"),
        Some(Ok(())) | None => what,
    };
    Err(error.context(format!(
        "could not record the server (pid {pid}) in its state file, {what}"
    )))
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

    /// A wrapper merge into a torn state file — the half-written document the
    /// old non-atomic writers could leave — fails and keeps the file as it
    /// was, rather than replacing it with the patch and losing `server_pid`.
    #[test]
    fn a_merge_into_a_torn_state_file_fails_and_keeps_it() {
        let dir = scratch_dir("torn-state");
        let path = dir.join("svc.json");
        let torn = br#"{"status": "running", "server_pid": 4242, "server_st"#;
        fs::write(&path, torn).unwrap();

        let merge = merge_json_state(&path, &json!({ "status": "ready" }));
        let mark = mark_json_status(&path, "stopped");

        let after = fs::read(&path).unwrap();
        fs::remove_dir_all(&dir).ok();
        for result in [merge, mark] {
            let error = format!(
                "{:#}",
                result.expect_err("a torn state file must be refused")
            );
            assert!(error.contains("is not valid JSON"), "{error}");
            assert!(
                error.contains("left it untouched and recorded nothing"),
                "{error}"
            );
        }
        assert_eq!(after, torn, "the torn file must be left as it was");
    }

    /// The race behind the lost `server_pid`: the wrapper merging into its own
    /// state file while a healthcheck latches its inference probe into the same
    /// file. Both must keep the server's identity, so a later stop still
    /// targets the server rather than the wrapper.
    #[test]
    fn wrapper_and_healthcheck_merges_keep_the_server_identity() {
        // Few rounds: the race itself is carried by rocm-core's cross-process
        // merge test; this one shows the wrapper's own writers take part in it.
        const ROUNDS: u64 = 20;
        let dir = scratch_dir("merge-race");
        let path = dir.join("svc.json");
        write_state(
            &path,
            &json!({
                "status": "running",
                "pid": 100,
                "start_ticks": 111_u64,
                "server_pid": 4242,
                "server_start_ticks": 99_u64,
            }),
        )
        .unwrap();

        let (wrapper, healthcheck) = std::thread::scope(|scope| {
            let wrapper = scope.spawn(|| -> Result<()> {
                for round in 0..ROUNDS {
                    merge_json_state(&path, &json!({ "backend_state": "ready", "round": round }))?;
                    mark_json_status(&path, "ready")?;
                }
                Ok(())
            });
            let healthcheck = scope.spawn(|| -> Result<()> {
                for round in 0..ROUNDS {
                    rocm_core::merge_json_state_file(
                        &path,
                        &json!({ rocm_core::INFERENCE_PROBE_ATTEMPTED_STATE_KEY: round }),
                    )?;
                }
                Ok(())
            });
            (
                wrapper.join().expect("wrapper thread"),
                healthcheck.join().expect("healthcheck thread"),
            )
        });

        let state = read_service_state(&path);
        fs::remove_dir_all(&dir).ok();
        wrapper.expect("every wrapper merge succeeds");
        healthcheck.expect("every healthcheck merge succeeds");
        let state = state.expect("the state file is still a whole document");
        let identity = identity_from_state(&state).expect("an identity to stop");
        assert_eq!(identity.pid, 4242, "a stop must target the server: {state}");
        assert_eq!(identity.start_ticks, Some(99), "{state}");
        assert_eq!(state["round"], ROUNDS - 1, "{state}");
        assert_eq!(
            state[rocm_core::INFERENCE_PROBE_ATTEMPTED_STATE_KEY],
            ROUNDS - 1,
            "{state}"
        );
    }

    /// Runs `write` on another thread while this one holds the state
    /// directory's lock, and returns whether `write` finished before the lock
    /// was released (it must not, if it takes the lock) and its result.
    fn finished_while_the_state_lock_was_held(
        path: &Path,
        write: impl FnOnce() -> Result<()> + Send,
    ) -> (bool, Result<()>) {
        use std::sync::atomic::{AtomicBool, Ordering};

        let lock = rocm_core::FileLock::acquire(rocm_core::json_state_lock_path(path))
            .expect("hold the state directory's lock");
        let started = AtomicBool::new(false);
        let finished = AtomicBool::new(false);
        std::thread::scope(|scope| {
            let writer = scope.spawn(|| {
                started.store(true, Ordering::SeqCst);
                let result = write();
                finished.store(true, Ordering::SeqCst);
                result
            });
            // The window starts once the writer runs, so a starved host cannot
            // pass the check by not scheduling the writer at all.
            while !started.load(Ordering::SeqCst) {
                std::thread::yield_now();
            }
            std::thread::sleep(Duration::from_millis(300));
            let finished_early = finished.load(Ordering::SeqCst);
            drop(lock);
            (finished_early, writer.join().expect("writer thread"))
        })
    }

    /// A full rewrite of the state file waits for the lock every merge takes,
    /// so it cannot land inside another writer's read-modify-write.
    #[test]
    fn a_state_rewrite_waits_for_the_state_directory_lock() {
        let dir = scratch_dir("rewrite-lock");
        let path = dir.join("svc.json");

        let (early, result) = finished_while_the_state_lock_was_held(&path, || {
            write_state(&path, &json!({ "status": "starting" }))
        });

        let state = read_service_state(&path);
        fs::remove_dir_all(&dir).ok();
        result.expect("the rewrite succeeds once the lock is free");
        assert!(!early, "the rewrite must wait for the lock");
        assert_eq!(
            state.expect("the rewrite was published"),
            json!({ "status": "starting" })
        );
    }

    #[test]
    fn a_state_merge_waits_for_the_state_directory_lock() {
        let dir = scratch_dir("merge-lock");
        let path = dir.join("svc.json");
        fs::write(&path, br#"{"server_pid": 4242}"#).unwrap();

        let (early, result) =
            finished_while_the_state_lock_was_held(&path, || mark_json_status(&path, "ready"));

        let state = read_service_state(&path);
        fs::remove_dir_all(&dir).ok();
        result.expect("the merge succeeds once the lock is free");
        assert!(!early, "the merge must wait for the lock");
        let state = state.expect("the merge was published");
        assert_eq!(state["server_pid"], 4242, "{state}");
        assert_eq!(state["status"], "ready", "{state}");
    }

    /// A [`StartedServer`] whose reap only counts calls or fails, for driving
    /// the stop outcomes a real process cannot be made to produce on demand.
    struct FakeServer {
        reaps: usize,
        reap_error: Option<&'static str>,
    }

    impl StartedServer for FakeServer {
        fn reap(&mut self) -> Result<()> {
            self.reaps += 1;
            match self.reap_error {
                Some(error) => Err(anyhow::anyhow!(error)),
                None => Ok(()),
            }
        }

        fn is_unreaped_child(&self) -> bool {
            false
        }
    }

    /// A state file the failing record left alone, holding a running server.
    fn running_state_file(tag: &str) -> (PathBuf, PathBuf) {
        let dir = scratch_dir(tag);
        let path = dir.join("svc.json");
        fs::write(&path, br#"{"status": "running", "pid": 100}"#).unwrap();
        (dir, path)
    }

    /// Every outcome a stop can report, and what the helper does after it: the
    /// error's wording, whether the server is reaped, and whether an existing
    /// state file is marked `failed`. Each message is asserted with the state
    /// it describes.
    #[test]
    fn each_stop_outcome_is_reported_with_the_state_it_leaves() {
        use rocm_core::TerminationOutcome as Outcome;

        let unconfirmed = |label: &str| {
            format!(
                "and could not confirm that it and the processes under it stopped ({label}), so \
                 they may still be running"
            )
        };
        let cases = [
            (Outcome::Graceful, "so stopped it".to_owned(), 1, "failed"),
            (Outcome::Forced, "so stopped it".to_owned(), 1, "failed"),
            (
                Outcome::AlreadyGone,
                "and it had already exited".to_owned(),
                1,
                "failed",
            ),
            (
                Outcome::IdentityMismatch,
                "and it had already exited; its pid now names another process, which was left \
                 alone"
                    .to_owned(),
                0,
                "failed",
            ),
            (Outcome::TimedOut, unconfirmed("timed_out"), 0, "running"),
            (Outcome::Unverified, unconfirmed("unverified"), 0, "running"),
            (
                Outcome::DescendantsUnconfirmed,
                unconfirmed("descendants_unconfirmed"),
                0,
                "running",
            ),
            (
                Outcome::ExitedBeforeTreeKill,
                unconfirmed("exited_before_tree_kill"),
                0,
                "running",
            ),
        ];
        for (outcome, what, reaps, status) in cases {
            let (dir, path) = running_state_file("stop-outcome");
            let mut server = FakeServer {
                reaps: 0,
                reap_error: None,
            };

            let result = record_server_or_stop_it_with(
                &mut server,
                rocm_core::ProcessIdentity::new(4242, Some(7)),
                &path,
                || Err(anyhow::anyhow!("state file is torn")),
                |_| outcome,
            );

            let state = read_service_state(&path).expect("the state file");
            fs::remove_dir_all(&dir).ok();
            let error = format!("{:#}", result.expect_err("the record failed"));
            assert_eq!(
                error,
                format!(
                    "could not record the server (pid 4242) in its state file, {what}: state \
                     file is torn"
                ),
                "{outcome:?}"
            );
            assert_eq!(server.reaps, reaps, "{outcome:?}");
            assert_eq!(state["status"], status, "{outcome:?}: {state}");
        }
    }

    /// On a host that records start tokens, a server that is not this
    /// process's unreaped child and has no token is refused and nothing is
    /// signalled: the server still runs afterwards.
    #[cfg(any(target_os = "linux", windows))]
    #[test]
    fn an_unrecorded_server_without_a_start_token_is_not_signalled_unless_owned() {
        let mut server = spawn_stand_in_server();
        let pid = server.id();

        let outcome = stop_unrecorded_server(
            rocm_core::ProcessIdentity::new(pid, None),
            UnrecordedServerStop {
                owned: false,
                grace: Duration::ZERO,
            },
        );

        let running = server
            .try_wait()
            .expect("query the stand-in server")
            .is_none();
        let _ = server.kill();
        let _ = server.wait();
        assert_eq!(outcome, rocm_core::TerminationOutcome::Unverified);
        assert!(running, "a refused stop must not signal the server");
    }

    /// A child of this process that has not been reaped keeps its pid, so it
    /// is stopped and reaped without a start token, through the production
    /// entry point and the `Child` it is handed.
    #[test]
    fn an_owned_server_without_a_start_token_is_stopped() {
        let (dir, path) = running_state_file("owned-no-token");
        let mut server = spawn_stand_in_server();
        let pid = server.id();

        let result = record_server_or_stop_it(
            &mut server,
            rocm_core::ProcessIdentity::new(pid, None),
            &path,
            || Err(anyhow::anyhow!("state file is torn")),
        );

        let reaped = reaped_by_the_helper(pid);
        let exited = exit_status_left_by_the_helper(&mut server);
        let state = read_service_state(&path).expect("the state file");
        fs::remove_dir_all(&dir).ok();
        let error = format!("{:#}", result.expect_err("the record failed"));
        assert_eq!(
            error,
            format!(
                "could not record the server (pid {pid}) in its state file, so stopped it: \
                 state file is torn"
            )
        );
        assert!(reaped, "the helper must reap the server it stopped");
        let status = exited.expect("the server must have been stopped and reaped");
        assert!(!status.success(), "stopped, not left to finish: {status}");
        assert_eq!(state["status"], "failed", "{state}");
    }

    /// The stop of an unrecorded server waits the engine's stop grace, and
    /// takes the server's own word for whether it is an unreaped child.
    #[test]
    fn an_unrecorded_server_is_stopped_with_the_stop_grace_and_its_own_ownership() {
        let fake = FakeServer {
            reaps: 0,
            reap_error: None,
        };
        let mut child = spawn_stand_in_server();

        let for_fake = UnrecordedServerStop::for_server(&fake);
        let for_child = UnrecordedServerStop::for_server(&child);

        let _ = child.kill();
        let _ = child.wait();
        assert!(!crate::STOP_GRACE.is_zero());
        assert_eq!(
            for_fake,
            UnrecordedServerStop {
                owned: false,
                grace: crate::STOP_GRACE,
            }
        );
        assert_eq!(
            for_child,
            UnrecordedServerStop {
                owned: true,
                grace: crate::STOP_GRACE,
            }
        );
    }

    /// The ready patch records the server's start token as captured at spawn,
    /// not read again from its pid.
    #[test]
    fn the_ready_patch_records_the_server_token_captured_at_spawn() {
        // This process's own pid, whose real token differs from the one handed
        // in; no real process has a start token of 1.
        let server = rocm_core::ProcessIdentity::new(std::process::id(), Some(1));

        let patch = ready_state_patch(server, "rocm");

        assert_eq!(
            Value::Object(patch),
            json!({
                "status": "ready",
                "server_pid": std::process::id(),
                "server_start_ticks": 1,
                "backend_state": "ready",
                "backend_requested": "rocm",
            })
        );
    }

    /// The server's start token is written as handed in, the one captured at
    /// spawn, rather than read again from the pid, which may have changed
    /// hands since.
    #[test]
    fn the_running_state_records_the_server_token_captured_at_spawn() {
        let dir = scratch_dir("running-state-token");
        let request = ServeHttpRequest {
            service_id: "svc".to_owned(),
            model_ref: DEFAULT_MODEL.to_owned(),
            host: DEFAULT_HOST.to_owned(),
            port: 11435,
            device_policy: rocm_engine_protocol::DevicePolicy::GpuRequired,
            gpu_indices: Vec::new(),
            runtime_id: None,
            env_id: None,
            state_path: dir.join("svc.json"),
            log_path: None,
            engine_recipe: None,
        };
        let runtime = LemonadeRuntime {
            manifest: LemonadeInstallManifest {
                env_id: "lemonade-embeddable-test".to_owned(),
                version: "test".to_owned(),
                runtime_dir: PathBuf::from("rt"),
                lemond: PathBuf::from("rt").join("lemond"),
                lemonade: PathBuf::from("rt").join("lemonade"),
                backend_recipe: "llamacpp".to_owned(),
                backend_name: "rocm".to_owned(),
                installed_at_unix_ms: 0,
            },
        };
        // This process's own pid, whose real token differs from the one handed
        // in; no real process has a start token of 1.
        let server = rocm_core::ProcessIdentity::new(std::process::id(), Some(1));

        let written = write_running_state(&request, &runtime, 100, Some(server), "running");

        let state = read_service_state(&request.state_path);
        fs::remove_dir_all(&dir).ok();
        written.expect("the running state is written");
        let state = state.expect("the state file");
        assert_eq!(state["server_pid"], std::process::id(), "{state}");
        assert_eq!(state["server_start_ticks"], 1, "{state}");
    }

    /// A server confirmed stopped is reaped and the service marked failed even
    /// when reaping it reports an error, and the error says which part failed.
    #[test]
    fn a_stopped_server_whose_reap_fails_says_so() {
        let (dir, path) = running_state_file("reap-fails");
        let mut server = FakeServer {
            reaps: 0,
            reap_error: Some("no child process"),
        };

        let result = record_server_or_stop_it_with(
            &mut server,
            rocm_core::ProcessIdentity::new(4242, Some(7)),
            &path,
            || Err(anyhow::anyhow!("state file is torn")),
            |_| rocm_core::TerminationOutcome::Forced,
        );

        let state = read_service_state(&path).expect("the state file");
        fs::remove_dir_all(&dir).ok();
        let error = format!("{:#}", result.expect_err("the record failed"));
        assert_eq!(
            error,
            "could not record the server (pid 4242) in its state file, so stopped it, but \
             reaping it failed: no child process: state file is torn"
        );
        assert_eq!(server.reaps, 1);
        assert_eq!(state["status"], "failed", "{state}");
    }

    #[test]
    fn a_recorded_server_is_neither_stopped_nor_reaped() {
        let (dir, path) = running_state_file("recorded");
        let mut server = FakeServer {
            reaps: 0,
            reap_error: None,
        };
        let mut stops = 0;

        let result = record_server_or_stop_it_with(
            &mut server,
            rocm_core::ProcessIdentity::new(4242, Some(7)),
            &path,
            || Ok(()),
            |_| {
                stops += 1;
                rocm_core::TerminationOutcome::Graceful
            },
        );

        let state = read_service_state(&path).expect("the state file");
        fs::remove_dir_all(&dir).ok();
        result.expect("the record succeeded");
        assert_eq!((stops, server.reaps), (0, 0));
        assert_eq!(state["status"], "running", "{state}");
    }

    const SLEEPER_CHILD_ENV: &str = "ROCM_LEMONADE_TEST_SLEEPER_CHILD";

    /// This test binary, run again as a stand-in server that stays up for a
    /// minute unless something stops it.
    fn spawn_stand_in_server() -> Child {
        std::process::Command::new(std::env::current_exe().expect("current test executable"))
            .args([
                "state::tests::a_server_whose_state_cannot_be_recorded_is_stopped",
                "--exact",
                "--quiet",
            ])
            .env(SLEEPER_CHILD_ENV, "1")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn the stand-in server")
    }

    /// Whether the helper reaped the server with pid `pid`: on Linux an exited
    /// child that nobody reaped stays in `/proc` as a zombie, so its entry is
    /// gone only once it was reaped. Read before the test touches the child
    /// itself, since a test-side `try_wait` would reap it too. Elsewhere there
    /// is nothing to read, so it answers `true`.
    fn reaped_by_the_helper(pid: u32) -> bool {
        if cfg!(target_os = "linux") {
            !Path::new(&format!("/proc/{pid}")).exists()
        } else {
            true
        }
    }

    /// Kills and reaps `server` if the helper left it running, and returns
    /// what the helper's reap had recorded: `None` means it was still running.
    fn exit_status_left_by_the_helper(server: &mut Child) -> Option<std::process::ExitStatus> {
        // Already reaped by the helper, so this returns the recorded status at
        // once; `None` would mean the server is still running.
        let exited = server.try_wait().expect("query the stand-in server");
        if exited.is_none() {
            let _ = server.kill();
            let _ = server.wait();
        }
        exited
    }

    #[test]
    fn a_server_whose_state_cannot_be_recorded_is_stopped() {
        if std::env::var_os(SLEEPER_CHILD_ENV).is_some() {
            std::thread::sleep(Duration::from_mins(1));
            return;
        }
        let (dir, path) = running_state_file("record-fails");
        let mut server = spawn_stand_in_server();
        let pid = server.id();

        let result = record_server_or_stop_it(
            &mut server,
            rocm_core::ProcessIdentity::capture(pid),
            &path,
            || Err(anyhow::anyhow!("state file is torn")),
        );

        let reaped = reaped_by_the_helper(pid);
        let exited = exit_status_left_by_the_helper(&mut server);
        let state = read_service_state(&path).expect("the state file");
        fs::remove_dir_all(&dir).ok();
        let error = format!("{:#}", result.expect_err("the record failed"));
        assert_eq!(
            error,
            format!(
                "could not record the server (pid {pid}) in its state file, so stopped it: \
                 state file is torn"
            )
        );
        assert!(reaped, "the helper must reap the server it stopped");
        let status = exited.expect("the server must have been stopped and reaped");
        assert!(
            !status.success(),
            "the server was stopped, not left to finish its minute: {status}"
        );
        assert_eq!(state["status"], "failed", "{state}");
    }

    /// The pids of `pid`'s direct children, from `/proc`.
    #[cfg(target_os = "linux")]
    fn child_pids(pid: u32) -> Vec<u32> {
        fs::read_to_string(format!("/proc/{pid}/task/{pid}/children"))
            .unwrap_or_default()
            .split_whitespace()
            .filter_map(|child| child.parse().ok())
            .collect()
    }

    /// Whether `pid` has exited: gone from `/proc`, or a zombie awaiting its
    /// reaper. Waits up to five seconds for that, since an orphan is reaped
    /// by whichever process adopts it.
    #[cfg(target_os = "linux")]
    fn exits_soon(pid: u32) -> bool {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        loop {
            let exited = fs::read_to_string(format!("/proc/{pid}/stat")).map_or(true, |stat| {
                stat.rsplit_once(')')
                    .and_then(|(_, rest)| rest.split_whitespace().next())
                    == Some("Z")
            });
            if exited || std::time::Instant::now() >= deadline {
                return exited;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// The production stop of an unrecorded server ends a server that ignores
    /// `SIGTERM`, as a server busy loading a model may, and the process it
    /// started under it, which nothing else records: by the `SIGKILL` its
    /// bounded stop escalates to, rather than holding the wrapper in a wait.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_server_that_ignores_sigterm_is_killed_with_its_child_when_its_state_cannot_be_recorded() {
        use std::os::unix::process::ExitStatusExt;

        let (dir, path) = running_state_file("ignores-sigterm");
        // A shell that ignores SIGTERM, starts a child that inherits that,
        // and keeps the disposition across its own `exec` into `sleep`.
        let mut server = std::process::Command::new("sh")
            .args(["-c", "trap '' TERM; sleep 600 & exec sleep 600"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawn the stand-in server");
        let pid = server.id();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let mut children = child_pids(pid);
        while children.is_empty() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
            children = child_pids(pid);
        }
        // Let the `exec` happen, so SIGTERM meets the ignoring `sleep`.
        std::thread::sleep(Duration::from_millis(500));

        // On another thread, so a helper that blocks on the server fails this
        // test after a deadline instead of hanging it. The deadline is far
        // shorter than the stand-in's sleep, so only a stop can beat it.
        let (done, finished) = std::sync::mpsc::channel();
        let (result, returned_in_time) = std::thread::scope(|scope| {
            let helper = scope.spawn(|| {
                let result = record_server_or_stop_it_with(
                    &mut server,
                    rocm_core::ProcessIdentity::capture(pid),
                    &path,
                    || Err(anyhow::anyhow!("state file is torn")),
                    |identity| {
                        stop_unrecorded_server(
                            identity,
                            UnrecordedServerStop {
                                owned: true,
                                grace: Duration::from_millis(500),
                            },
                        )
                    },
                );
                let _ = done.send(());
                result
            });
            let returned_in_time = finished.recv_timeout(Duration::from_secs(30)).is_ok();
            if !returned_in_time {
                // Unblock the helper so the test can report instead of hang.
                let _ = std::process::Command::new("kill")
                    .args(["-KILL", &pid.to_string()])
                    .status();
            }
            (helper.join().expect("helper thread"), returned_in_time)
        });
        let children_exited: Vec<(u32, bool)> = children
            .iter()
            .map(|&child| (child, exits_soon(child)))
            .collect();
        for &(child, exited) in &children_exited {
            if !exited {
                let _ = std::process::Command::new("kill")
                    .args(["-KILL", &child.to_string()])
                    .status();
            }
        }

        let reaped = reaped_by_the_helper(pid);
        let exited = exit_status_left_by_the_helper(&mut server);
        fs::remove_dir_all(&dir).ok();
        assert!(
            returned_in_time,
            "the stop must not wait on the server unbounded"
        );
        let error = format!("{:#}", result.expect_err("the record failed"));
        assert_eq!(
            error,
            format!(
                "could not record the server (pid {pid}) in its state file, so stopped it: \
                 state file is torn"
            )
        );
        assert!(reaped, "the helper must reap the server it stopped");
        let status = exited.expect("the server must have been killed and reaped");
        assert_eq!(
            status.signal(),
            Some(libc::SIGKILL),
            "SIGTERM was ignored, so only SIGKILL stops it: {status}"
        );
        assert_eq!(children.len(), 1, "the stand-in started one child");
        assert!(
            children_exited.iter().all(|&(_, exited)| exited),
            "the process under the server must be stopped too: {children_exited:?}"
        );
    }

    /// A confirmed stop marks an existing state file `failed` (see
    /// `a_server_whose_state_cannot_be_recorded_is_stopped`) but does not
    /// create one that the failed record never wrote.
    #[test]
    fn a_stopped_server_with_no_state_file_creates_none() {
        let dir = scratch_dir("no-state-file");
        let path = dir.join("svc.json");
        let mut server = FakeServer {
            reaps: 0,
            reap_error: None,
        };

        let result = record_server_or_stop_it_with(
            &mut server,
            rocm_core::ProcessIdentity::new(4242, Some(7)),
            &path,
            || Err(anyhow::anyhow!("could not create the state file")),
            |_| rocm_core::TerminationOutcome::Graceful,
        );

        let created = path.exists();
        fs::remove_dir_all(&dir).ok();
        let error = format!("{:#}", result.expect_err("the record failed"));
        assert_eq!(
            error,
            "could not record the server (pid 4242) in its state file, so stopped it: \
             could not create the state file"
        );
        assert_eq!(server.reaps, 1);
        assert!(!created, "no state file existed, so none may be created");
    }

    #[test]
    fn a_server_whose_state_is_recorded_is_left_running() {
        let (dir, path) = running_state_file("record-ok");
        let mut server = spawn_stand_in_server();
        let pid = server.id();

        let result = record_server_or_stop_it(
            &mut server,
            rocm_core::ProcessIdentity::capture(pid),
            &path,
            || Ok(()),
        );

        let running = server
            .try_wait()
            .expect("query the stand-in server")
            .is_none();
        let _ = server.kill();
        let _ = server.wait();
        fs::remove_dir_all(&dir).ok();
        result.expect("the record succeeded");
        assert!(running, "a recorded server is the wrapper's to wait on");
    }
}
