// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Engine plugin binary resolution and the engine IPC request/response
//! envelope: locating an engine's executable (packaged plugin vs. the
//! platform-packaged fallback), talking to it over JSON-over-stdio, and the
//! built-in (in-process) fast path for `lemonade`/`vllm`/`aim` that skips
//! spawning a child process entirely.
//!
//! The stdio conversation itself is not implemented here. The request framing,
//! the child's environment, stderr handling, the SIGPIPE guard around the write
//! and the reading of the answer are the client in
//! [`rocm_engine_protocol`]'s `stdio_client` module, which `rocmd` uses too.
//! This module chooses what to run (the built-in handler, or the resolved
//! engine binary with `stdio`) and what `rocm` asks of the client for it
//! ([`engine_stdio_child_request`]).
//!
//! `dispatch()` calls directly into this module at two points —
//! `Command::EngineStdio { engine } => run_builtin_engine_stdio(&engine)` and
//! the `__engine-serve-http` arm's `run_builtin_engine_serve_http(...)` call —
//! both re-imported via `use crate::engine_ipc::{run_builtin_engine_serve_http,
//! run_builtin_engine_stdio};`.
//!
//! This module owns: resolving an engine's binary path
//! (`resolve_engine_binary_path`, `resolve_engine_binary_path_with_paths`,
//! `find_engine_plugin_binary`), the request entry points
//! (`engine_request`, `engine_request_with_env_root`,
//! `engine_stdio_child_request`), the `ScopedEnvVar`/
//! `with_scoped_builtin_engine_env` env-scoping helpers and the
//! `BUILTIN_ENGINE_ENV_LOCK` mutex they serialize on, `builtin_engine_request`,
//! `run_builtin_engine_stdio`, and the built-in HTTP-serving path
//! (`builtin_engine_available`, `run_builtin_engine_serve_http`,
//! `builtin_engine_serve_http_args`).
//!
//! `main.rs` calls into it for `resolve_engine_binary_path`,
//! `resolve_engine_binary_path_with_paths`, `engine_request`,
//! `builtin_engine_available`, and `builtin_engine_serve_http_args`, plus the
//! two `dispatch()` call sites above; `engines_cmd.rs` and `serve_cmd.rs` call
//! into it for `engine_request`; `engines_cmd.rs` and
//! `runtime_install_finalize.rs` call into it for
//! `engine_request_with_env_root`; `service_launch.rs` calls into it for
//! `builtin_engine_serve_http_args`. `find_engine_plugin_binary`,
//! `with_scoped_builtin_engine_env`, `builtin_engine_request`, and
//! `ScopedEnvVar` have no callers outside this module and stay private. The
//! one helper this module calls back into at the crate root, `device_policy_name`,
//! stays in `main.rs` and is reached via `use crate::device_policy_name;`.
//! `parse_device_policy` is in `gpu_policy.rs`, `engine_recipe_json_arg` is in
//! `engine_runtime_select.rs`, and `optional_arg` is in `rocm_engine_protocol`.

use std::ffi::OsString;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use rocm_core::{AppPaths, ENGINE_ENVS_ROOT_ENV, engine_binary_path, engine_plugin_dirs};
use rocm_engine_protocol::{
    DevicePolicy, EngineMethod, EnginePluginDescriptor, EngineRecipeHint, EngineRequestEnvelope,
    EngineResponseEnvelope, EngineStdioOptions, HealthcheckRequest, HealthcheckResponse,
    decode_engine_response, engine_request_envelope, optional_arg, run_engine_stdio,
};
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::device_policy_name;
use crate::engine_runtime_select::engine_recipe_json_arg;
use crate::gpu_policy::parse_device_policy;

static BUILTIN_ENGINE_ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

pub(crate) fn resolve_engine_binary_path(engine: &str) -> Result<PathBuf> {
    let paths = AppPaths::discover()?;
    resolve_engine_binary_path_with_paths(engine, &paths)
}

pub(crate) fn resolve_engine_binary_path_with_paths(
    engine: &str,
    paths: &AppPaths,
) -> Result<PathBuf> {
    if let Some(path) = find_engine_plugin_binary(engine, engine_plugin_dirs(paths))? {
        return Ok(path);
    }
    engine_binary_path(engine)
}

fn find_engine_plugin_binary<I, P>(engine: &str, plugin_dirs: I) -> Result<Option<PathBuf>>
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path>,
{
    Ok(rocm_engine_protocol::discover_engine_plugins(plugin_dirs)
        .context("failed to discover engine plugin binaries")?
        .into_iter()
        .find(|plugin: &EnginePluginDescriptor| plugin.id == engine)
        .map(|plugin| plugin.executable_path))
}

pub(crate) fn engine_request<T, R>(
    paths: Option<&AppPaths>,
    engine: &str,
    method: EngineMethod,
    request: &T,
) -> Result<R>
where
    T: Serialize,
    R: DeserializeOwned,
{
    engine_request_with_env_root(paths, engine, method, request, None)
}

pub(crate) fn engine_request_with_env_root<T, R>(
    paths: Option<&AppPaths>,
    engine: &str,
    method: EngineMethod,
    request: &T,
    env_root: Option<&Path>,
) -> Result<R>
where
    T: Serialize,
    R: DeserializeOwned,
{
    engine_request_with_scoped_env(paths, engine, method, request, env_root, None)
}

/// [`engine_request_with_env_root`], additionally handing a **built-in** engine
/// the 0600 endpoint key file for one service via `ROCM_SERVE_API_KEY_FILE`.
///
/// An engine that probes its own server's HTTP endpoint while answering a
/// request reads that variable to authenticate
/// (`rocm_engine_protocol::resolve_endpoint_api_key`, reached from
/// `engines/lemonade`'s and `engines/vllm`'s `/v1/models` probes). Without it,
/// a public or `--require-api-key` service answers 401 and the engine reports
/// the service as not listed - indistinguishable, from the CLI's side, from a
/// server that is simply down. It is threaded per call rather than per process
/// because the key is per service id.
///
/// **External plugin children are not given it.** The built-in engines run
/// in-process, so the variable is set and restored around one call
/// ([`with_scoped_builtin_engine_env`]); a plugin is a third-party binary this
/// process spawns, and a variable put in its environment is readable from
/// `/proc/<pid>/environ` and inherited by every grandchild it starts. That is
/// more exposure than an informational `device:` row is worth — the only caller
/// that passes a key file is [`fetch_service_device`]. The cost is that a
/// plugin-backed service with endpoint auth reports no device; it is a row the
/// summary already declares optional.
///
/// This is a reduction in incidental exposure, not a security boundary: a
/// plugin child is handed `ROCM_CLI_DATA_DIR` by [`apply_app_path_env`] and the
/// key file's location under it is a documented, deterministic convention
/// (`rocm_engine_protocol::endpoint_key_file_path`), so a plugin that wants the
/// key can still derive the path. What changes is that it must go looking.
fn engine_request_with_scoped_env<T, R>(
    paths: Option<&AppPaths>,
    engine: &str,
    method: EngineMethod,
    request: &T,
    env_root: Option<&Path>,
    endpoint_key_file: Option<&Path>,
) -> Result<R>
where
    T: Serialize,
    R: DeserializeOwned,
{
    let stream_progress = matches!(&method, EngineMethod::Install);
    let envelope = engine_request_envelope(method, request)?;
    if let Some(envelope) =
        with_scoped_builtin_engine_env(paths, env_root, endpoint_key_file, || {
            builtin_engine_request(engine, &envelope)
        })
    {
        return decode_engine_response(envelope);
    }

    let engine_binary = resolve_engine_binary_path(engine).with_context(|| {
        format!(
            "unable to locate engine binary for {engine}; build the workspace or install the engine package"
        )
    })?;
    engine_stdio_child_request(&engine_binary, paths, &envelope, env_root, stream_progress)
}

/// The out-of-process half of [`engine_request_with_env_root`]: run
/// `engine_binary stdio` through the client `rocm` shares with `rocmd`, send
/// `envelope` on its stdin and decode its answer.
///
/// What `rocm` asks of that client: the child runs on this process's app paths
/// when it has them and on the engine environments root when the request names
/// one, an install asks for progress on stderr, and the child's stderr is
/// written to this process's as it arrives, which is how a terminal user
/// watches an install. No endpoint key is handed over, because none of the
/// requests `rocm` makes is about a running service's endpoint. Split out so a
/// test can point it at a stand-in binary.
fn engine_stdio_child_request<R>(
    engine_binary: &Path,
    paths: Option<&AppPaths>,
    envelope: &EngineRequestEnvelope,
    env_root: Option<&Path>,
    stream_progress: bool,
) -> Result<R>
where
    R: DeserializeOwned,
{
    let options = stdio_options(paths, env_root, stream_progress);
    decode_engine_response(run_engine_stdio(
        engine_binary,
        ["stdio"],
        &options,
        envelope,
    )?)
}

/// What `rocm` asks of the shared stdio client; see
/// [`engine_stdio_child_request`].
const fn stdio_options<'a>(
    paths: Option<&'a AppPaths>,
    env_root: Option<&'a Path>,
    stream_progress: bool,
) -> EngineStdioOptions<'a> {
    EngineStdioOptions {
        app_paths: paths,
        env_root,
        endpoint_key_file: None,
        request_progress: stream_progress,
        forward_stderr: true,
        note_failing_exit_after_answer: false,
    }
}

struct ScopedEnvVar {
    key: &'static str,
    previous: Option<OsString>,
}

impl ScopedEnvVar {
    #[allow(unsafe_code)] // std::env::set_var is unsafe in edition 2024
    fn set_path(key: &'static str, value: &Path) -> Self {
        let previous = std::env::var_os(key);
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }
}

impl Drop for ScopedEnvVar {
    #[allow(unsafe_code)] // std::env::set_var/remove_var are unsafe in edition 2024
    fn drop(&mut self) {
        unsafe {
            match self.previous.as_ref() {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

fn with_scoped_builtin_engine_env<R>(
    paths: Option<&AppPaths>,
    env_root: Option<&Path>,
    endpoint_key_file: Option<&Path>,
    f: impl FnOnce() -> R,
) -> R {
    if paths.is_none() && env_root.is_none() && endpoint_key_file.is_none() {
        return f();
    }
    let lock = BUILTIN_ENGINE_ENV_LOCK.get_or_init(|| Mutex::new(()));
    let _guard = lock.lock().expect("builtin engine env lock poisoned");
    let mut vars = Vec::new();
    if let Some(paths) = paths {
        for (key, value) in paths.env_vars() {
            vars.push(ScopedEnvVar::set_path(key, value));
        }
    }
    if let Some(env_root) = env_root {
        vars.push(ScopedEnvVar::set_path(ENGINE_ENVS_ROOT_ENV, env_root));
    }
    if let Some(endpoint_key_file) = endpoint_key_file {
        vars.push(ScopedEnvVar::set_path(
            rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV,
            endpoint_key_file,
        ));
    }
    let result = f();
    drop(vars);
    result
}

fn builtin_engine_request(
    engine: &str,
    envelope: &EngineRequestEnvelope,
) -> Option<EngineResponseEnvelope> {
    match engine {
        "lemonade" => Some(rocm_engine_lemonade::builtin_handle_envelope(
            envelope.clone(),
        )),
        "vllm" => Some(rocm_engine_vllm::builtin_handle_envelope(envelope.clone())),
        "aim" => Some(rocm_engine_aim::builtin_handle_envelope(envelope.clone())),
        _ => None,
    }
}

pub(crate) fn run_builtin_engine_stdio(engine: &str) -> Result<()> {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .context("failed to read engine stdio request")?;
    let envelope: EngineRequestEnvelope =
        serde_json::from_str(&input).context("failed to parse engine stdio request")?;
    let response = builtin_engine_request(engine, &envelope)
        .with_context(|| format!("engine `{engine}` is not built into this rocm binary"))?;
    print!("{}", serde_json::to_string(&response)?);
    Ok(())
}

/// Best-effort query of the backend/device a running managed service actually
/// selected (e.g. `"vulkan gpu 0"` / `"rocm gpu 0"`), via the same
/// `Healthcheck` RPC every engine plugin already answers
/// (`engines/lemonade` reads it back from its own `backend_requested` state;
/// `engines/vllm` and `engines/aim` report the ROCm device they pinned). Read
/// from the engine, never guessed from the launch-time GPU selection, so it
/// reflects what the server is actually running on. `None` on any failure -
/// this is informational text, not worth failing `serve`/`services list` over.
///
/// How long one call can take, since `rocm services list` makes one per ready
/// service: [`engine_request`] itself imposes no deadline, so the bound comes
/// from the engine. The three built-in engines answer in-process and every
/// endpoint probe they make is timeout-bounded (`engines/vllm`'s
/// `HEALTHCHECK_TIMEOUT_MS`, `engines/lemonade`'s
/// `rocm_core::INFERENCE_PROBE_TIMEOUT`), so a stalled server costs seconds,
/// not forever. An *external* plugin is a spawned child that
/// `engine_request` waits on unbounded, exactly as every other engine call
/// site in the CLI does. [`fetch_service_devices`] is what keeps a list of
/// services from multiplying that cost by the number of services.
pub(crate) fn fetch_service_device(
    paths: &AppPaths,
    engine: &str,
    service_id: &str,
) -> Option<String> {
    // Hand the engine this service's 0600 key file. Both built-in engines
    // authenticate their own `/v1/models` probe from
    // `ROCM_SERVE_API_KEY_FILE`, so without it a public or `--require-api-key`
    // service answers 401, the engine reports it as not listed, and the device
    // comes back `"unknown"` - the row would vanish for exactly the services
    // that are protected. `None` for a loopback service, which has no key file
    // and needs none.
    let endpoint_key_file = crate::endpoint_keys::endpoint_key_file_if_present(paths, service_id);
    let response: HealthcheckResponse = engine_request_with_scoped_env(
        Some(paths),
        engine,
        EngineMethod::Healthcheck,
        &HealthcheckRequest {
            service_id: service_id.to_owned(),
        },
        None,
        endpoint_key_file.as_deref(),
    )
    .ok()?;
    device_worth_reporting(response.device)
}

/// The engine's reported device, or `None` when it said nothing usable.
///
/// `"unknown"` is the engines' own device value when no state is on disk for
/// this service id yet (see `engines/vllm/src/state.rs` and
/// `engines/lemonade/src/lib.rs`), and it is not more informative than showing
/// nothing — a `device: unknown` row claims the CLI asked and learned
/// something, which is the opposite of what happened.
///
/// A named function rather than an inline `then_some` because the only caller
/// is an RPC to a live engine: in default CI that call never succeeds, so the
/// filter is unreachable there and deleting it would be invisible until a
/// freshly-launched Lemonade service printed `device: unknown` on a GPU lane.
fn device_worth_reporting(device: String) -> Option<String> {
    (!device.trim().is_empty() && device != "unknown").then_some(device)
}

/// How long `rocm serve`'s deployment summary may keep asking the engine which
/// device it selected, and how long it waits between asks.
///
/// A managed launch reports `ready` as soon as its server answers, but
/// `engines/lemonade` reports `device: "unknown"` until its own
/// `inference_verified` probe has come back — so asking once, immediately,
/// loses the row on exactly the Vulkan case ROCMAI-550 is about. One retry
/// window closes the common case: the default smoke test has just put a real
/// completion through the server, so the engine's own probe is a cache hit.
///
/// **It cannot close every case, and is not meant to.** A server that reported
/// `ready` while weights are still loading can stay unverified for minutes, and
/// waiting that out would hold the terminal hostage for an informational row
/// that `rocm services list` will show as soon as it exists. So the row is
/// *optional by contract* — see [`crate::serve_summary::DeploymentSummary`]'s
/// `device` field, `docs/manual-testing.md` and README.md, which all say it can
/// be absent.
///
/// The budget is a start gate, like [`SERVICE_DEVICE_PROBE_BUDGET`]: it is
/// tested before each attempt, never inside one, so the worst-case addition to
/// `rocm serve` is this budget plus one in-flight probe — about 14s with
/// Lemonade's 8s `rocm_core::INFERENCE_PROBE_TIMEOUT`, and only on a launch
/// that was going to have no row at all.
const SERVE_DEVICE_PROBE_BUDGET: Duration = Duration::from_secs(6);
/// Pause between the attempts [`SERVE_DEVICE_PROBE_BUDGET`] allows. Short
/// enough that the common case (the engine is a second away from verifying)
/// resolves quickly, long enough not to spin on an engine that answers
/// instantly.
const SERVE_DEVICE_PROBE_INTERVAL: Duration = Duration::from_secs(2);

/// [`fetch_service_device`], retried briefly, for `rocm serve`'s one-shot
/// deployment summary.
///
/// `None` is a legitimate answer and must stay cheap to get: see
/// [`SERVE_DEVICE_PROBE_BUDGET`] for why the wait is bounded and why the
/// summary prints no `device:` row rather than blocking on one.
pub(crate) fn fetch_service_device_briefly(
    paths: &AppPaths,
    engine: &str,
    service_id: &str,
) -> Option<String> {
    let started = Instant::now();
    retry_service_device(
        || started.elapsed(),
        || fetch_service_device(paths, engine, service_id),
        std::thread::sleep,
    )
}

/// The retry loop of [`fetch_service_device_briefly`], with the clock, the
/// probe and the wait injected.
///
/// Split out for the same reason as [`collect_service_devices`]: driving this
/// against the live clock would mean a test that really sleeps out
/// [`SERVE_DEVICE_PROBE_BUDGET`] against a real engine. What has to be
/// guaranteed is not just that a late answer is reported but that the loop
/// *stops* — an unbounded retry here would hang every `rocm serve` whose engine
/// never verifies.
fn retry_service_device<C, P, S>(mut elapsed: C, mut probe: P, mut wait: S) -> Option<String>
where
    C: FnMut() -> Duration,
    P: FnMut() -> Option<String>,
    S: FnMut(Duration),
{
    loop {
        if let Some(device) = probe() {
            return Some(device);
        }
        if elapsed() >= SERVE_DEVICE_PROBE_BUDGET {
            return None;
        }
        wait(SERVE_DEVICE_PROBE_INTERVAL);
    }
}

/// How long [`fetch_service_devices`] may spend on live `Healthcheck` RPCs in
/// total, across however many services were listed.
///
/// `rocm services list` was pure local I/O before the `device:` row; the row is
/// worth a short wait but not an unbounded one, and the cost has to stay flat
/// in the number of services rather than growing with it. Sized to let a
/// couple of healthy engines answer (`engines/lemonade`'s endpoint probe is
/// the slowest at `rocm_core::INFERENCE_PROBE_TIMEOUT`, 8s worst case, and
/// only on a miss) while still bounding the whole render.
///
/// **Worst-case wall time for `rocm services list` is this budget plus one
/// in-flight probe, i.e. ~18s** with the built-in engines, not 10s. The budget
/// is a *start* gate: it is tested before each probe, never inside one, so a
/// probe begun at 9.99s still runs to its own timeout. The arithmetic is 10s
/// of elapsed budget plus Lemonade's 8s `INFERENCE_PROBE_TIMEOUT`. That bound
/// applies equally to the chat `services` tool, which renders through the same
/// [`crate::services_text::render_services_text`].
///
/// # The decision, recorded (review on #678)
///
/// This is accepted rather than merely observed. `rocm services list` and the
/// chat `services` tool can both block ~18s on a host whose engines are slow to
/// answer, and for the duration the built-in probe holds the process-wide
/// engine env lock [`with_scoped_builtin_engine_env`] takes, so a concurrent
/// caller waits behind it.
///
/// Two cheaper options were considered and not taken. Moving the probe off the
/// render path means the row stops reflecting the host at the moment it is
/// printed, which is the whole reason ROCMAI-550 added it — the `device:` row
/// exists to answer "what is this actually running on *now*". Probing once and
/// printing nothing for the rest makes the output depend on service ordering,
/// which reads as a bug to anyone who lists twice and sees different rows.
///
/// Both remain reasonable calls and this one is reversible: it is a constant and
/// a start gate, not a structural choice. If a maintainer prefers a 10s hard
/// cap, the change is a per-probe deadline, which needs a cancellable probe
/// (see below) rather than a smaller number here.
///
/// Why not a per-probe deadline that would hold the total at 10s: the probe is
/// `engine_request`, which for a built-in engine is a synchronous in-process
/// call and for an external plugin is a `wait()` on a spawned child. Neither
/// can be cancelled from here without either a thread per probe (ruled out by
/// the process-wide env/mutex scoping `fetch_service_devices` documents) or a
/// kill path for a child mid-RPC. Both are larger changes than the row is
/// worth; the start gate is what keeps the cost flat in the number of
/// services, which is the growth that actually matters.
pub(crate) const SERVICE_DEVICE_PROBE_BUDGET: Duration = Duration::from_secs(10);

/// Devices for several services at once, keyed by service id, under one shared
/// time budget.
///
/// The probes run sequentially on purpose. They cannot usefully be run in
/// parallel: [`engine_request`] routes the built-in engines through
/// [`with_scoped_builtin_engine_env`], which takes a process-wide mutex *and*
/// mutates process-wide `ROCM_CLI_*` env vars for the duration, so concurrent
/// callers would serialize on that lock anyway and racing them would be
/// unsound. What this does instead is cap the aggregate: once
/// [`SERVICE_DEVICE_PROBE_BUDGET`] is spent, the remaining services are left
/// without a `device:` row rather than each adding its own wait. The row is
/// informational, so dropping it is strictly better than making
/// `rocm services list --all` hang behind one unresponsive server.
///
/// The budget is checked between probes, not inside one: a single external
/// plugin that never exits still blocks, because `engine_request` waits on its
/// child unbounded. That is the pre-existing behavior of every engine call
/// site in the CLI and is not made worse here. See
/// [`SERVICE_DEVICE_PROBE_BUDGET`] for what that makes the worst-case wall
/// time of a `rocm services list`.
pub(crate) fn fetch_service_devices<'a, I>(
    paths: &AppPaths,
    services: I,
) -> std::collections::HashMap<String, String>
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    let started = Instant::now();
    collect_service_devices(
        services,
        || started.elapsed(),
        |service_id, engine| fetch_service_device(paths, engine, service_id),
    )
}

/// The budget loop of [`fetch_service_devices`], with the clock and the probe
/// injected.
///
/// Split out so the budget-exhausted exit has a test. In the real function the
/// clock is `Instant::elapsed` and the probe is a live engine RPC, so covering
/// that branch would otherwise mean making a test wait out
/// [`SERVICE_DEVICE_PROBE_BUDGET`] against real engines — which is both slow
/// and exactly the timing-dependent shape that flakes. What the branch has to
/// guarantee is not just that later services lack a `device:` row but that
/// they were never *probed*, since the whole point is to stop paying; only an
/// injected probe can observe that.
fn collect_service_devices<'a, I, C, P>(
    services: I,
    mut elapsed: C,
    mut probe: P,
) -> std::collections::HashMap<String, String>
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
    C: FnMut() -> Duration,
    P: FnMut(&str, &str) -> Option<String>,
{
    let mut devices = std::collections::HashMap::new();
    for (service_id, engine) in services {
        if elapsed() >= SERVICE_DEVICE_PROBE_BUDGET {
            break;
        }
        if let Some(device) = probe(service_id, engine) {
            devices.insert(service_id.to_owned(), device);
        }
    }
    devices
}

pub(crate) fn builtin_engine_available(engine: &str) -> bool {
    matches!(engine, "lemonade" | "vllm" | "aim")
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_builtin_engine_serve_http(
    engine: &str,
    service_id: String,
    model_ref: String,
    host: String,
    port: u16,
    device_policy: &str,
    gpu_indices: Vec<u32>,
    runtime_id: Option<String>,
    env_id: Option<String>,
    state_path: PathBuf,
    log_path: Option<PathBuf>,
    engine_recipe: Option<EngineRecipeHint>,
) -> Result<()> {
    let parsed_policy = parse_device_policy(Some(device_policy))?;
    // Every engine is fronted the same way, so the Anthropic endpoint is started
    // here once rather than inside each adapter. It runs on a thread of this
    // process, which is the service's supervisor: it ends when the service does.
    rocmd::anthropic_proxy::spawn_alongside_engine(rocmd::anthropic_proxy::AnthropicProxyConfig {
        host: host.clone(),
        engine_port: port,
        served_model: model_ref.clone(),
        state_path: state_path.clone(),
        probe_api_key: rocm_engine_protocol::resolve_endpoint_api_key(),
    });
    match engine {
        "lemonade" => rocm_engine_lemonade::builtin_serve_http(
            service_id,
            model_ref,
            host,
            port,
            parsed_policy,
            gpu_indices,
            runtime_id,
            env_id,
            state_path,
            log_path,
            engine_recipe,
        ),
        "vllm" => rocm_engine_vllm::builtin_serve_http(
            service_id,
            model_ref,
            host,
            port,
            parsed_policy,
            gpu_indices,
            runtime_id,
            env_id,
            state_path,
            log_path,
            engine_recipe,
        ),
        "aim" => rocm_engine_aim::builtin_serve_http(
            service_id,
            model_ref,
            host,
            port,
            parsed_policy,
            gpu_indices,
            runtime_id,
            env_id,
            state_path,
            log_path,
            engine_recipe,
        ),
        other => bail!("engine `{other}` is not built into this rocm binary"),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn builtin_engine_serve_http_args(
    engine: &str,
    service_id: &str,
    canonical_model_id: &str,
    host: &str,
    port: u16,
    device_policy: &DevicePolicy,
    gpu_indices: &[u32],
    runtime_id: Option<&str>,
    env_id: Option<&str>,
    engine_recipe: Option<&EngineRecipeHint>,
    state_path: &Path,
    log_path: Option<&Path>,
) -> Result<Vec<String>> {
    let mut args = vec![
        "__engine-serve-http".to_owned(),
        engine.to_owned(),
        service_id.to_owned(),
        canonical_model_id.to_owned(),
        "--host".to_owned(),
        host.to_owned(),
        "--port".to_owned(),
        port.to_string(),
        "--device-policy".to_owned(),
        device_policy_name(device_policy).to_owned(),
    ];
    if let Some(csv) = rocm_engine_protocol::gpu_indices_to_csv(gpu_indices) {
        args.extend(["--gpu".to_owned(), csv]);
    }
    if env_id.is_none() {
        args.extend(optional_arg("--runtime-id", runtime_id));
    }
    args.extend(optional_arg("--env-id", env_id));
    args.extend(engine_recipe_json_arg(engine_recipe)?);
    args.extend(["--state-path".to_owned(), state_path.display().to_string()]);
    if let Some(log_path) = log_path {
        args.extend(["--log-path".to_owned(), log_path.display().to_string()]);
    }
    Ok(args)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::tests::test_paths;

    /// The `unknown` filter, which nothing else can reach: its caller is an RPC
    /// to a live engine, so in default CI it never runs and deleting it stays
    /// green.
    #[test]
    fn a_device_the_engine_does_not_know_is_not_reported_as_one() {
        for nothing in ["unknown", "", "   "] {
            assert_eq!(
                device_worth_reporting(nothing.to_owned()),
                None,
                "`{nothing}` is the engine saying it has no answer yet, and a \
                 `device:` row would claim otherwise"
            );
        }
        for device in ["rocm gpu 0", "vulkan (AMD Radeon Graphics)"] {
            assert_eq!(
                device_worth_reporting(device.to_owned()),
                Some(device.to_owned())
            );
        }
    }

    /// The endpoint key file reaches a built-in engine through the process
    /// environment, and is gone again afterwards.
    ///
    /// Both halves matter and neither was covered. Without the variable a
    /// `--require-api-key` service's own `/v1/models` probe answers 401, the
    /// engine reports the service as unlisted, and the `device:` row silently
    /// vanishes for exactly the protected services. Without the restore, one
    /// service's key would leak into the next service's probe in the same
    /// `rocm services list` — `with_scoped_builtin_engine_env` mutates
    /// process-wide state, which is why the probes are serialized in the first
    /// place.
    #[test]
    fn the_endpoint_key_file_is_scoped_to_one_builtin_engine_call() {
        let key = rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV;
        // SAFETY-adjacent: the scoped setter takes the same process-wide lock
        // this test observes through, so no other engine call can interleave.
        let before = std::env::var_os(key);
        let seen = with_scoped_builtin_engine_env(
            None,
            None,
            Some(Path::new("/tmp/rocm-e2e-endpoint-key")),
            || std::env::var_os(key),
        );
        assert_eq!(
            seen.as_deref(),
            Some(std::ffi::OsStr::new("/tmp/rocm-e2e-endpoint-key")),
            "the engine must be able to read the key file path it was given"
        );
        assert_eq!(
            std::env::var_os(key),
            before,
            "the key must not outlive the call it was scoped to"
        );

        // And nothing is set when no key file was supplied: a loopback service
        // has none, and inventing one would point the engine at a file that
        // does not exist.
        assert_eq!(
            with_scoped_builtin_engine_env(None, None, None, || std::env::var_os(key)),
            before
        );
    }

    /// Every service inside the budget is probed and reported.
    #[test]
    fn service_device_probes_run_while_the_budget_holds() {
        let mut probed = Vec::new();
        let devices = collect_service_devices(
            [("svc-a", "lemonade"), ("svc-b", "vllm")],
            || Duration::ZERO,
            |service_id, engine| {
                probed.push(service_id.to_owned());
                Some(format!("{engine}-device"))
            },
        );

        assert_eq!(probed, ["svc-a", "svc-b"]);
        assert_eq!(
            devices.get("svc-a").map(String::as_str),
            Some("lemonade-device")
        );
        assert_eq!(
            devices.get("svc-b").map(String::as_str),
            Some("vllm-device")
        );
    }

    /// Once [`SERVICE_DEVICE_PROBE_BUDGET`] is spent the remaining services are
    /// *not probed at all*, and what was already learned is still rendered.
    ///
    /// This is the branch that keeps `rocm services list` — and the chat
    /// `services` tool, which renders through the same
    /// `render_services_text` — from scaling its wall time with the number of
    /// services. Dropping the later rows is only half of it; the point is to
    /// stop paying for them, so the probe counter below is the real assertion.
    /// Asserting it against the live clock and live engines would mean a test
    /// that waits out ten real seconds, which is why the loop takes both
    /// injected.
    #[test]
    fn service_device_probes_stop_once_the_budget_is_exhausted() {
        let mut probed = Vec::new();
        // Inside the budget for the first service, over it from then on.
        let mut tick = 0;
        let devices = collect_service_devices(
            [
                ("svc-fast", "lemonade"),
                ("svc-hung", "vllm"),
                ("svc-never-reached", "aim"),
            ],
            || {
                tick += 1;
                if tick <= 1 {
                    Duration::ZERO
                } else {
                    SERVICE_DEVICE_PROBE_BUDGET
                }
            },
            |service_id, _engine| {
                probed.push(service_id.to_owned());
                Some("rocm gpu 0".to_owned())
            },
        );

        assert_eq!(
            probed,
            ["svc-fast"],
            "the budget must stop further probing, not merely hide its results"
        );
        assert_eq!(
            devices.get("svc-fast").map(String::as_str),
            Some("rocm gpu 0"),
            "what was learned before the budget ran out is still reported"
        );
        assert!(
            !devices.contains_key("svc-hung") && !devices.contains_key("svc-never-reached"),
            "an unprobed service must get no `device:` row: {devices:?}"
        );
    }

    /// The serve summary asks again when the first answer is `None`: the
    /// engine can be seconds from verifying its own inference probe, which is
    /// the `device: "unknown"` window `engines/lemonade` has right after a
    /// launch reports ready.
    #[test]
    fn the_serve_summary_probe_waits_for_a_device_that_is_not_ready_yet() {
        let mut attempts = 0;
        let mut waited = Duration::ZERO;
        let device = retry_service_device(
            || Duration::ZERO,
            || {
                attempts += 1;
                (attempts > 2).then(|| "vulkan gpu 0".to_owned())
            },
            |interval| waited += interval,
        );

        assert_eq!(device.as_deref(), Some("vulkan gpu 0"));
        assert_eq!(attempts, 3);
        assert_eq!(
            waited,
            SERVE_DEVICE_PROBE_INTERVAL * 2,
            "the retries must pace themselves rather than spin"
        );
    }

    /// And it gives up. A lemonade service whose weights are still loading can
    /// stay `unknown` for minutes; the summary prints no `device:` row rather
    /// than holding the terminal, which is why the row is documented as
    /// optional.
    #[test]
    fn the_serve_summary_probe_gives_up_within_its_budget() {
        let mut attempts = 0;
        // Inside the budget for the first attempt, past it from then on.
        let mut tick = 0;
        let device = retry_service_device(
            || {
                tick += 1;
                if tick <= 1 {
                    Duration::ZERO
                } else {
                    SERVE_DEVICE_PROBE_BUDGET
                }
            },
            || {
                attempts += 1;
                None
            },
            |_| {},
        );

        assert_eq!(device, None);
        assert_eq!(
            attempts, 2,
            "the budget must end the retries, not merely discard their results"
        );
    }

    /// A service whose engine cannot answer costs its own probe and nothing
    /// more: it is skipped, and the services after it are still probed.
    #[test]
    fn a_service_whose_probe_fails_does_not_suppress_the_rest() {
        let devices = collect_service_devices(
            [("svc-quiet", "lemonade"), ("svc-ok", "vllm")],
            || Duration::ZERO,
            |service_id, _engine| (service_id == "svc-ok").then(|| "rocm gpu 1".to_owned()),
        );

        assert!(!devices.contains_key("svc-quiet"), "{devices:?}");
        assert_eq!(
            devices.get("svc-ok").map(String::as_str),
            Some("rocm gpu 1")
        );
    }

    #[test]
    fn engine_plugin_discovery_finds_runtime_binary() -> Result<()> {
        let (root, paths) = test_paths("engine-plugin");
        let plugin_dir = paths.primary_engine_plugin_dir();
        fs::create_dir_all(&plugin_dir)?;
        let plugin_path = plugin_dir.join(
            rocm_engine_protocol::platform_engine_plugin_binary_name("vllm"),
        );
        fs::write(&plugin_path, "plugin")?;

        let discovered = find_engine_plugin_binary("vllm", engine_plugin_dirs(&paths))?;
        let _ = fs::remove_dir_all(root);

        assert_eq!(discovered, Some(plugin_path));
        Ok(())
    }

    #[test]
    fn engine_plugin_discovery_prefers_primary_plugin_dir() -> Result<()> {
        let (root, paths) = test_paths("engine-plugin-precedence");
        let primary_dir = paths.primary_engine_plugin_dir();
        let compatibility_dir = paths.data_dir.join("engines");
        fs::create_dir_all(&primary_dir)?;
        fs::create_dir_all(&compatibility_dir)?;
        let name = rocm_engine_protocol::platform_engine_plugin_binary_name("vllm");
        let primary_path = primary_dir.join(&name);
        fs::write(&primary_path, "primary")?;
        fs::write(compatibility_dir.join(&name), "compatibility")?;

        let discovered = find_engine_plugin_binary("vllm", engine_plugin_dirs(&paths))?;
        let _ = fs::remove_dir_all(root);

        assert_eq!(discovered, Some(primary_path));
        Ok(())
    }

    // ---- What `rocm` asks of the shared stdio client. ----
    //
    // The conversation itself (framing, SIGPIPE guard, exit status and answer
    // handling) is the client's, tested in `rocm-engine-protocol`. These tests
    // run `rocm`'s entry point against a stand-in engine child and pin what
    // `rocm` chooses: the verb, the environment and the stderr handling. The
    // other caller's choices are pinned in `apps/rocmd/src/engine_ipc.rs`.

    #[test]
    fn rocm_asks_for_forwarded_stderr_and_no_endpoint_key() {
        let (root, paths) = crate::tests::test_paths("engine-ipc-options");
        let env_root = root.join("envs");

        let options = stdio_options(Some(&paths), Some(&env_root), true);
        assert_eq!(
            options.app_paths.map(|p| &p.data_dir),
            Some(&paths.data_dir)
        );
        assert_eq!(options.env_root, Some(env_root.as_path()));
        assert!(options.request_progress);
        // The terminal user watches an install through the child's stderr.
        assert!(options.forward_stderr);
        assert!(!options.note_failing_exit_after_answer);
        // No request `rocm` makes is about a running service's endpoint.
        assert_eq!(options.endpoint_key_file, None);

        let bare = stdio_options(None, None, false);
        assert!(bare.app_paths.is_none() && bare.env_root.is_none());
        assert!(!bare.request_progress);
        assert!(bare.forward_stderr);
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    mod stdio_child {
        use super::*;
        use rocm_engine_protocol::fake_engine::FakeEngine;
        use serde_json::{Value, json};
        use std::os::unix::process::ExitStatusExt as _;

        const STOPPED: &str = r#"{"ok":true,"data":{"stopped":true}}"#;

        fn envelope(method: EngineMethod, payload: serde_json::Value) -> EngineRequestEnvelope {
            EngineRequestEnvelope { method, payload }
        }

        fn stop_request() -> EngineRequestEnvelope {
            envelope(EngineMethod::Stop, json!({"service_id": "svc"}))
        }

        fn request(
            engine: &FakeEngine,
            paths: Option<&AppPaths>,
            env_root: Option<&Path>,
            stream_progress: bool,
        ) -> Result<Value> {
            engine_stdio_child_request::<Value>(
                engine.path(),
                paths,
                &stop_request(),
                env_root,
                stream_progress,
            )
        }

        /// Run `f` holding the lock that serialises every test that sets
        /// `ROCM_CLI_*` process-wide (`with_scoped_builtin_engine_env`), so that
        /// what a child inherits when `f` starts it and what [`ambient`] reads
        /// inside `f` come from the same instant.
        fn under_env_lock<R>(f: impl FnOnce() -> R) -> R {
            let lock = BUILTIN_ENGINE_ENV_LOCK.get_or_init(|| Mutex::new(()));
            let _guard = lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            f()
        }

        fn ambient(key: &str) -> Option<String> {
            std::env::var(key).ok()
        }

        #[test]
        fn the_child_gets_the_stdio_verb_and_one_line_of_request() {
            let engine = FakeEngine::answering("rocm-framing", STOPPED, 0);
            let answer = request(&engine, None, None, false).unwrap();

            assert_eq!(answer, json!({"stopped": true}));
            assert_eq!(engine.argv(), ["stdio"]);
            assert_eq!(
                engine.stdin(),
                "{\"method\":\"stop\",\"payload\":{\"service_id\":\"svc\"}}\n"
            );
        }

        #[test]
        fn the_child_runs_on_the_callers_app_paths_when_given_them() {
            let (root, paths) = crate::tests::test_paths("engine-ipc-app-paths");
            let engine = FakeEngine::answering("rocm-app-paths", STOPPED, 0);
            request(&engine, Some(&paths), None, false).unwrap();
            let _ = fs::remove_dir_all(root);

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
        fn the_child_inherits_the_callers_environment_when_no_app_paths_are_given() {
            const KEYS: [&str; 3] = [
                "ROCM_CLI_CONFIG_DIR",
                "ROCM_CLI_DATA_DIR",
                "ROCM_CLI_CACHE_DIR",
            ];
            let engine = FakeEngine::answering("rocm-no-app-paths", STOPPED, 0);
            let expected = under_env_lock(|| {
                request(&engine, None, None, false).unwrap();
                KEYS.map(ambient)
            });

            for (key, expected) in KEYS.into_iter().zip(expected) {
                assert_eq!(engine.env(key), expected, "{key}");
            }
        }

        /// The comparison of a child's inherited environment with the ambient one
        /// holds while another thread scopes `ROCM_CLI_*` process-wide, which
        /// other tests of this module do: the spawn and the read happen under the
        /// same lock. Without the lock the child can start while the variables
        /// are scoped and the read see them restored (or the other way round).
        #[test]
        fn the_inherited_environment_is_compared_at_the_instant_the_child_starts() {
            use std::sync::Arc;
            use std::sync::atomic::{AtomicBool, Ordering};

            let (root, paths) = crate::tests::test_paths("engine-ipc-env-race");
            let done = Arc::new(AtomicBool::new(false));
            let flipper = {
                let done = Arc::clone(&done);
                std::thread::spawn(move || {
                    while !done.load(Ordering::Relaxed) {
                        with_scoped_builtin_engine_env(Some(&paths), None, None, || {
                            std::thread::sleep(std::time::Duration::from_millis(2));
                        });
                        // Outside the lock, so the test's own turn is not starved.
                        std::thread::sleep(std::time::Duration::from_millis(2));
                    }
                })
            };

            for round in 0..25 {
                let engine = FakeEngine::answering("rocm-env-race", STOPPED, 0);
                let expected = under_env_lock(|| {
                    request(&engine, None, None, false).unwrap();
                    ambient("ROCM_CLI_DATA_DIR")
                });
                assert_eq!(engine.env("ROCM_CLI_DATA_DIR"), expected, "round {round}");
            }

            done.store(true, Ordering::Relaxed);
            flipper.join().unwrap();
            let _ = fs::remove_dir_all(root);
        }

        #[test]
        fn an_env_root_reaches_the_child_as_the_engine_envs_root() {
            let engine = FakeEngine::answering("rocm-env-root", STOPPED, 0);
            let env_root = Path::new("/an/engine/envs/root");
            request(&engine, None, Some(env_root), false).unwrap();
            assert_eq!(
                engine.env("ROCM_CLI_ENGINE_ENVS_ROOT").as_deref(),
                Some("/an/engine/envs/root")
            );

            let without = FakeEngine::answering("rocm-no-env-root", STOPPED, 0);
            let expected = under_env_lock(|| {
                request(&without, None, None, false).unwrap();
                ambient("ROCM_CLI_ENGINE_ENVS_ROOT")
            });
            assert_eq!(without.env("ROCM_CLI_ENGINE_ENVS_ROOT"), expected);
        }

        #[test]
        fn progress_streaming_is_asked_for_only_when_the_caller_wants_it() {
            let streaming = FakeEngine::answering("rocm-progress-on", STOPPED, 0);
            request(&streaming, None, None, true).unwrap();
            assert_eq!(
                streaming.env("ROCM_ENGINE_PROGRESS_STDERR").as_deref(),
                Some("1")
            );

            let quiet = FakeEngine::answering("rocm-progress-off", STOPPED, 0);
            let expected = under_env_lock(|| {
                request(&quiet, None, None, false).unwrap();
                ambient("ROCM_ENGINE_PROGRESS_STDERR")
            });
            assert_eq!(quiet.env("ROCM_ENGINE_PROGRESS_STDERR"), expected);
        }

        #[test]
        fn the_endpoint_key_is_not_passed_to_the_child() {
            // `rocm`'s requests are not about a running service's endpoint (detect,
            // install, resolve-model, stop), so this client has no service to find
            // a key for. `rocmd`'s healthcheck is the call that threads one.
            let engine = FakeEngine::answering("rocm-no-key", STOPPED, 0);
            let key = rocm_engine_protocol::ENDPOINT_API_KEY_FILE_ENV;
            let expected = under_env_lock(|| {
                request(&engine, None, None, false).unwrap();
                ambient(key)
            });
            assert_eq!(engine.env(key), expected);
        }

        #[test]
        fn a_nonzero_exit_that_still_answered_is_decoded_not_discarded() {
            // The engine's own error envelope is the useful diagnostic, so an
            // answer on stdout wins over the exit status.
            let engine = FakeEngine::answering(
                "rocm-answers-then-fails",
                r#"{"ok":false,"error":{"code":"stop_failed","message":"it broke"}}"#,
                1,
            );
            let error = request(&engine, None, None, false).unwrap_err();
            assert_eq!(error.to_string(), "stop_failed: it broke");

            let engine = FakeEngine::answering("rocm-answers-then-exits-2", STOPPED, 2);
            assert_eq!(
                request(&engine, None, None, false).unwrap(),
                json!({"stopped": true})
            );
        }

        /// The write of a request to an engine that has already exited must come
        /// back as an error even where SIGPIPE is at its default action, which is
        /// how `main` runs `rocm`. Re-executes this test in a fresh process that
        /// installs `SIG_DFL` first (the test harness leaves SIGPIPE ignored, and
        /// changing it process-wide would race sibling tests).
        #[test]
        fn a_request_to_an_engine_that_already_exited_is_an_error_under_sig_dfl() {
            if std::env::var_os("ROCM_TEST_ENGINE_SIGPIPE_ARM").is_some() {
                crate::reset_sigpipe();
                let engine = FakeEngine::new("rocm-sigpipe", false, "exit 3");
                // Larger than a pipe buffer, so the write reaches the closed read end.
                let big = "x".repeat(1 << 20);
                let result = engine_stdio_child_request::<Value>(
                    engine.path(),
                    None,
                    &envelope(EngineMethod::Stop, json!({ "blob": big })),
                    None,
                    false,
                );
                let error = result.expect_err("the engine never answered");
                assert!(
                    format!("{error:#}").contains("failed to write engine request"),
                    "{error:#}"
                );
                return;
            }

            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "engine_ipc::tests::stdio_child::a_request_to_an_engine_that_already_exited_is_an_error_under_sig_dfl",
                    "--exact",
                    "--nocapture",
                ])
                .env("ROCM_TEST_ENGINE_SIGPIPE_ARM", "1")
                .status()
                .unwrap();
            assert_eq!(
                status.signal(),
                None,
                "SIGPIPE killed the process instead of surfacing as an error: {status:?}"
            );
            assert!(status.success(), "{status:?}");
        }
    }
}
