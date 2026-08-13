// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use std::sync::LazyLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use cucumber::{given, then, when};

use crate::E2eWorld;
use e2e_cucumber::mock_server::MockServer;
use e2e_cucumber::serve_log::{
    ServeAttempt, archive_service_log, serve_attempt_report, service_log_tail,
};

/// How long to wait for a freshly served model's endpoint to become ready.
///
/// On real GPU hardware the first serve of a model downloads its weights and
/// loads them onto the device before `/v1/models` responds, which can far exceed
/// a minute for a multi-billion-parameter model on a cold cache (the built-in
/// catalog now resolves `qwen2.5` to a 4B GGUF). Default high; override with
/// `E2E_SERVE_TIMEOUT_SECS` for slower hardware or a warm-cache local run.
fn serve_timeout_secs() -> u64 {
    std::env::var("E2E_SERVE_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(600)
}

/// Serve-readiness timeout for a scenario: a per-scenario override set by the
/// `before` hook takes precedence over the global `E2E_SERVE_TIMEOUT_SECS` /
/// default. The override is a `@serve-timeout:<secs>` tag on an expected-pass
/// scenario (lengthen a genuinely slow serve, e.g. a large model), else an
/// `expectations.toml` xfail `serve_timeout_secs` (shorten a known-bug serve so
/// it fails fast).
fn serve_timeout_for(world: &E2eWorld) -> u64 {
    world
        .serve_timeout_override
        .unwrap_or_else(serve_timeout_secs)
}

/// Wait for `<endpoint>/models` to return 200. When `expect_model` is given,
/// wait until that model id actually appears in the listing — not merely any
/// 200. This defends against a leaked serve from a prior scenario still
/// answering on the shared port (11435): scenarios run in isolated data dirs, so
/// scenario A's `rocm` has no record of scenario B's managed service and can't
/// stop it; a plain 200 check would then proceed against the WRONG model. Wait
/// for the expected model so the readiness signal reflects this scenario's serve.
async fn model_is_ready(models_url: &str, expect_model: Option<&str>, timeout_secs: u64) -> bool {
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    while Instant::now() < deadline {
        if let Ok(resp) = reqwest::get(models_url).await
            && resp.status().is_success()
        {
            match expect_model {
                None => return true,
                Some(model) => {
                    if let Ok(body) = resp.text().await
                        && body.contains(model)
                    {
                        return true;
                    }
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
    false
}

/// The OpenAI-compatible base URL every GPU serve scenario serves on (see
/// [`SERVE_PORT`]), and the model listing readiness is judged by.
static SERVE_BASE_URL: LazyLock<String> =
    LazyLock::new(|| format!("http://127.0.0.1:{SERVE_PORT}/v1"));
static MODELS_URL: LazyLock<String> =
    LazyLock::new(|| format!("http://127.0.0.1:{SERVE_PORT}/v1/models"));

/// The one-line verdict on a `--managed` serve that launched cleanly and then
/// never produced a usable endpoint.
///
/// Says "exited 0" out loud because that is the whole difficulty of this failure:
/// nothing in the exit status, and nothing the CLI printed, distinguishes it from
/// a healthy serve — the evidence sections below it are the only account there is.
fn stall_headline(invocation: &str, ready_substr: &str, timeout_secs: u64) -> String {
    let models_url = MODELS_URL.as_str();
    format!(
        "{invocation} exited 0, but {models_url} never served `{ready_substr}` within \
         {timeout_secs}s — the launch reported success and the engine then failed on its own"
    )
}

/// Everything a serve attempt left behind, gathered and rendered for a panic.
///
/// The ORDER here is the load-bearing part, and it is why this is one function
/// rather than three calls at each site: the log must be read and copied out
/// while the service still owns it, because stopping the service both changes
/// what the log's last lines say and is the step most likely to fail outright.
///
/// Mirrors what [`setup_gpu_model`] has always collected, so a stalled serve
/// reports the same evidence wherever it happens.
fn serve_failure_evidence(
    world: &E2eWorld,
    headline: &str,
    device_state: &str,
    stdout: &str,
    stderr: &str,
) -> String {
    let log_tail = service_log_tail(stdout);
    // Copy the whole log into the results directory before the scenario's
    // TempDir takes it: the tail below is enough to triage most stalls, but not
    // one whose cause is in the engine's startup banner (see `archive_service_log`).
    let archived_log = archive_service_log(
        stdout,
        &crate::results_path(),
        world.current_scenario.as_deref().unwrap_or("scenario"),
    );
    let stop_status = stop_scenario_services(world);
    serve_attempt_report(&ServeAttempt {
        headline,
        device_state,
        stdout,
        stderr,
        log_tail: &log_tail,
        archived_log: &archived_log,
        stop_status: &stop_status,
    })
}

/// Launch a managed serve and wait until the endpoint really serves it, failing
/// with the full evidence bundle if it does not.
///
/// Replaces the `run_rocm_ok` + readiness-poll pair the serve preconditions used
/// to open with. That pair cannot describe the failure this exists for: with
/// `--managed`, `rocm serve` returns once the supervisor is launched, so an
/// engine that dies afterwards exits **0** and the poll times out with nothing
/// attached — the CLI's output, the engine's log and the device state were all
/// dropped before the panic, which is why #260 could not be root-caused from a
/// CI artifact. A non-zero exit is reported the same way rather than through
/// `run_rocm_ok`, since the engine log explains those launches too.
async fn serve_and_wait(world: &mut E2eWorld, args: &[&str], model: &str, ready_substr: &str) {
    let timeout_secs = serve_timeout_for(world);
    let device_state = ensure_serve_port_free().await;
    let (stdout, stderr, rc) = crate::run_rocm(world, args);
    if rc == 0 && model_is_ready(&MODELS_URL, Some(ready_substr), timeout_secs).await {
        world.endpoint = Some(SERVE_BASE_URL.to_string());
        world.model_name = Some(model.to_string());
        return;
    }
    let invocation = format!("`rocm {}`", args.join(" "));
    let headline = if rc == 0 {
        stall_headline(&invocation, ready_substr, timeout_secs)
    } else {
        format!("{invocation} failed (rc={rc})")
    };
    panic!(
        "{}",
        serve_failure_evidence(world, &headline, &device_state, &stdout, &stderr)
    );
}

/// The shared port every GPU serve scenario uses. Because scenarios run in
/// isolated data dirs on one serial GPU box, a serve from a prior scenario can
/// still hold this port (and GPU memory) when the next starts — its managed
/// service lives in a different isolated dir, so this scenario's `rocm` can't
/// stop it. Left unchecked, servers accumulate and oversubscribe the GPU until
/// the job times out.
const SERVE_PORT: u16 = 11435;

/// The port the CLI's built-in local assistant (lemonade Qwen3-4B) listens on.
/// The CLI auto-starts this assistant independently of any scenario; on Instinct
/// it falls back to a Vulkan llama-server that pins a GPU core (EAI-7052),
/// starving the vLLM serves the scenarios actually test until they exceed the
/// job timeout. No scenario needs the built-in assistant, so we free this port
/// too before serving.
const ASSISTANT_PORT: u16 = 8001;

/// Best-effort: ensure the shared serve port is free before starting a new
/// serve, so a leaked server from a prior scenario can't linger on the GPU.
/// Polls until nothing answers on the port (bounded), killing any listener.
///
/// Returns a one-line description of the device state the next serve starts on
/// (see [`wait_for_free_vram`]). Callers that only need the reset can ignore it;
/// a serve that then fails to become ready reports it, because "the previous
/// engine had not released the GPU yet" is otherwise invisible in the log.
async fn ensure_serve_port_free() -> String {
    // Always kill any listener on the shared port — NOT just one that already
    // answers /v1/models. A prior scenario's vLLM that is still LOADING holds the
    // port and GPU memory without yet serving /v1/models; if we only checked HTTP
    // readiness we'd start a second server, overcommit GPU memory (each claims
    // vLLM's default fraction of the device), and the collision crashes a server
    // → the next chat POST fails with
    // "error sending request". Killing by port (fuser/lsof) catches the starting
    // server too. Best-effort; then wait for the socket to actually close.
    // Also kill the CLI's auto-started lemonade assistant — it hogs a GPU core on
    // Vulkan (EAI-7052) and starves the vLLM serve under test; no scenario needs it.
    kill_listeners_on_port(SERVE_PORT);
    kill_listeners_on_port(ASSISTANT_PORT);
    let deadline = Instant::now() + Duration::from_mins(1);
    loop {
        // TcpStream connect succeeds only while something holds the port.
        let free = tokio::net::TcpStream::connect(("127.0.0.1", SERVE_PORT))
            .await
            .is_err();
        if free || Instant::now() >= deadline {
            break;
        }
        kill_listeners_on_port(SERVE_PORT);
        kill_listeners_on_port(ASSISTANT_PORT);
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
    // The port closing does NOT mean the prior serve's VRAM is back: a killed
    // vLLM releases ~tens of GiB of device memory only as the process fully
    // exits, which lags the socket close. The next `rocm serve` reads free VRAM
    // at startup and demands vLLM's `gpu-memory-utilization` fraction of the
    // TOTAL (rocm-cli passes no value, so vLLM's own default applies) — after a
    // large model (e.g. 27B ~54 GiB) the residue can drop free memory below that
    // request, so the next serve dies with "Free memory ... less than desired GPU
    // memory utilization" (engine core init failed). Wait for the device to
    // actually drain before returning.
    wait_for_free_vram().await
}

/// Upper bound on the free-VRAM floor (MiB). Sized so the largest single
/// scenario model can allocate its engine's memory share without tripping its
/// startup check: Qwen3.6-27B plus vLLM's default fraction-of-total KV
/// reservation needs the MI300X mostly clear. Only a data-center GPU has this
/// much, so on smaller cards (e.g. Strix Halo's smaller unified-memory pool) the
/// floor is capped to 90% of the device total (see [`required_free_vram_mib`]) —
/// otherwise the check could never pass. Note the cap no longer sits above what
/// vLLM asks for: rocm-cli used to pin utilization below it, and now defers to
/// vLLM's own (higher) default, so the wait has little headroom left and can
/// time out on a device that is merely holding display memory. The wait is
/// best-effort and the serve proceeds regardless, so this costs time rather
/// than correctness; sizing a deliberate margin is follow-up work.
const MAX_FREE_VRAM_FLOOR_MIB: u64 = 150_000;

/// The free-VRAM floor to wait for on this host: the model-sized ceiling, but
/// never more than 90% of the device's total VRAM. A hardcoded 150 GiB floor is
/// unreachable on a small card (e.g. Strix Halo's ~48 GiB unified VRAM), so
/// `wait_for_free_vram` would burn its full deadline on every serve scenario;
/// scaling to the device keeps the drain-check meaningful everywhere.
fn required_free_vram_mib(total_mib: u64) -> u64 {
    MAX_FREE_VRAM_FLOOR_MIB.min(total_mib / 100 * 90)
}

/// How long to wait for a stopped engine to hand its device memory back.
const VRAM_DRAIN_DEADLINE: Duration = Duration::from_mins(2);

/// Best-effort: wait until the GPU reports enough free VRAM (see
/// [`required_free_vram_mib`]), so a just-killed serve's memory is actually
/// reclaimed before the next serve starts. Queries `amd-smi` then `rocm-smi`;
/// if neither is present (mock/local, no ROCm), returns immediately so non-GPU
/// runs are unaffected.
///
/// The wait is bounded and best-effort: on timeout the serve still starts,
/// because a stale reading must not turn a slow drain into a hard failure. The
/// returned line records which of the two happened — an undrained device is the
/// single most likely reason the serve that follows never becomes ready, and
/// without it the failure looks identical to a genuinely broken serve.
async fn wait_for_free_vram() -> String {
    // No GPU tooling → nothing to wait on (mock/local). Probe once up front.
    let Some(total) = total_vram_mib() else {
        return "device state: no GPU tooling (mock/local run)".to_owned();
    };
    let floor = required_free_vram_mib(total);
    let deadline = Instant::now() + VRAM_DRAIN_DEADLINE;
    loop {
        let free = free_vram_mib();
        if let Some(free) = free
            && free >= floor
        {
            return format!(
                "device state: drained ({free} MiB free of {total} MiB, floor {floor} MiB)"
            );
        }
        if Instant::now() >= deadline {
            let free = free.map_or_else(|| "unreadable".to_owned(), |mib| format!("{mib} MiB"));
            return format!(
                "device state: NOT drained after {}s ({free} free of {total} MiB, floor \
                 {floor} MiB) — a previous engine is still holding the GPU",
                VRAM_DRAIN_DEADLINE.as_secs()
            );
        }
        tokio::time::sleep(Duration::from_secs(3)).await;
    }
}

/// Free device VRAM in MiB for GPU 0, via `amd-smi` (then `rocm-smi`). `None`
/// when no such tool exists or its output can't be parsed.
fn free_vram_mib() -> Option<u64> {
    vram_mib().map(|(_total, free)| free)
}

/// Total device VRAM in MiB for GPU 0. `None` when no GPU tool is available —
/// used as the "is there a GPU to wait on at all?" probe in mock/local runs.
fn total_vram_mib() -> Option<u64> {
    vram_mib().map(|(total, _free)| total)
}

/// `(total, free)` device VRAM in MiB for GPU 0, via `amd-smi` (then `rocm-smi`).
/// `None` when no such tool exists or its output can't be parsed.
fn vram_mib() -> Option<(u64, u64)> {
    use std::process::Command;
    // amd-smi: lines like "        TOTAL_VRAM: 196592 MB" / "FREE_VRAM: 196309 MB".
    if let Ok(out) = Command::new("amd-smi").args(["metric", "-m"]).output()
        && out.status.success()
    {
        let text = String::from_utf8_lossy(&out.stdout);
        let field = |name: &str| {
            text.lines()
                .find_map(|l| l.trim().strip_prefix(name))
                .and_then(|v| v.split_whitespace().next())
                .and_then(|n| n.parse::<u64>().ok())
        };
        if let (Some(total), Some(free)) = (field("TOTAL_VRAM:"), field("FREE_VRAM:")) {
            return Some((total, free));
        }
    }
    // rocm-smi fallback: `--showmeminfo vram --csv` → vram total/used per card.
    if let Ok(out) = Command::new("rocm-smi")
        .args(["--showmeminfo", "vram", "--csv"])
        .output()
        && out.status.success()
    {
        let text = String::from_utf8_lossy(&out.stdout);
        // CSV columns include "VRAM Total Memory (B)" and "VRAM Total Used Memory (B)".
        // Parse the first data row's total and used to derive free.
        let mut lines = text.lines();
        let header = lines.next()?;
        let cols: Vec<&str> = header.split(',').collect();
        let total_idx = cols.iter().position(|c| c.contains("Total Memory"))?;
        let used_idx = cols.iter().position(|c| c.contains("Total Used Memory"))?;
        let row = lines.next()?;
        let vals: Vec<&str> = row.split(',').collect();
        let total: u64 = vals.get(total_idx)?.trim().parse().ok()?;
        let used: u64 = vals.get(used_idx)?.trim().parse().ok()?;
        let mib = 1024 * 1024;
        return Some((total / mib, total.saturating_sub(used) / mib));
    }
    None
}

/// Kill whatever process is listening on `port`. Best-effort and
/// platform-specific; failures are ignored (the caller only needs the port
/// eventually free, verified by polling).
fn kill_listeners_on_port(port: u16) {
    use std::process::Command;
    #[cfg(unix)]
    {
        // `fuser -k <port>/tcp` kills listeners; fall back to lsof→kill if absent.
        let _ = Command::new("bash")
            .arg("-c")
            .arg(format!(
                "fuser -k {port}/tcp 2>/dev/null || \
                 (for p in $(lsof -t -iTCP:{port} -sTCP:LISTEN 2>/dev/null); do kill -9 \"$p\"; done)"
            ))
            .status();
    }
    #[cfg(windows)]
    {
        let _ = Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                &format!(
                    "Get-NetTCPConnection -LocalPort {port} -State Listen -EA SilentlyContinue | \
                     ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -EA SilentlyContinue }}"
                ),
            ])
            .status();
    }
}

// ── Given ──────────────────────────────────────────────────────────

#[given("a model is being served on the default port")]
async fn setup_mock_default_port(world: &mut E2eWorld) {
    let mock = MockServer::start("TestModel/E2E-1B").await;
    world.endpoint = Some(mock.base_url());
    world.model_name = Some("TestModel/E2E-1B".to_string());
    world.mock = Some(mock);
}

#[given("a model is being served on a non-default port")]
async fn setup_mock_custom_port(world: &mut E2eWorld) {
    let mock = MockServer::start("TestModel/E2E-1B").await;
    world.endpoint = Some(mock.base_url());
    world.model_name = Some("TestModel/E2E-1B".to_string());
    world.mock = Some(mock);
}

/// The (model, engine, ready-substring) this host should serve for an
/// engine-agnostic "serve a real model" precondition.
///
/// The behaviour these preconditions set up — serve a model, then chat/infer —
/// is not vLLM-specific, so the concrete model+engine follows the host's
/// effective serve engine: a safetensors model on vLLM (Instinct), a GGUF model
/// on lemonade (Strix Halo / native Windows). This mirrors the two dedicated
/// single-engine steps (`setup_gpu_model`'s old vLLM body and
/// `setup_lemonade_model`), but lets a scenario tagged only `@requires-gpu` run
/// on whichever engine the platform actually uses.
///
/// MODEL-SIZE POLICY: GPU serve scenarios use small models that still satisfy
/// their assertions — real GPU serves are the E2E wall-clock long pole on serial
/// hardware, so weight-load time is pure overhead here. Current paths use vLLM
/// `Qwen3.5-0.8B` and lemonade `Qwen3-0.6B-GGUF`. Large-model behaviour is
/// exercised exactly once, in the `@nightly` `serve-large-model-inference`
/// scenario (Qwen3.6-27B) — never on the per-PR path. Do not raise a serve
/// scenario's model unless a smaller one genuinely cannot prove the assertion.
fn host_serve_target() -> (&'static str, &'static str, &'static str) {
    if e2e_cucumber::capability::host_capability().effective_serve_engine == "lemonade" {
        // GGUF via lemonade's llama.cpp backend; endpoint reports e.g.
        // Qwen3-0.6B-Q4_0.gguf, so "Qwen3-0.6B" is the distinctive substring.
        ("Qwen3-0.6B-GGUF", "lemonade", "Qwen3-0.6B")
    } else {
        // Safetensors via vLLM; "Qwen3.5-0.8B" is the distinctive substring.
        ("Qwen/Qwen3.5-0.8B", "vllm", "Qwen3.5-0.8B")
    }
}

/// The model the default-engine (no `--engine`) serve step should request.
///
/// This is deliberately NOT `host_serve_target` — that step passes no `--engine`,
/// so the CLI resolves the engine from the model's own `preferred_engines`, and
/// the model choice therefore determines WHICH engine runs. The default-engine
/// xfail matrix depends on that resolution: on an Instinct host the request must
/// resolve to a GGUF recipe on lemonade (EAI-7052 — see expectations.toml), NOT
/// to vLLM. So this stays lemonade-preferred on every host; the smaller model
/// used by vLLM applies only to the explicit-engine path. The model hangs at load
/// on Instinct (EAI-7052, xfail) so its size is irrelevant to
/// wall-clock here — behaviour preservation is what matters.
fn default_engine_serve_target() -> &'static str {
    if e2e_cucumber::capability::host_capability().effective_serve_engine == "lemonade" {
        "Qwen3-0.6B-GGUF"
    } else {
        "Qwen/Qwen2.5-1.5B-Instruct"
    }
}

/// Stop every managed service this scenario launched, tree-killing the engine
/// processes that hold the GPU. Used between serve attempts; the World's `Drop`
/// runs the same teardown at scenario end.
///
/// Returns the stop's own account of itself (see [`crate::stop_managed_services`]),
/// which the retry quotes: a stop that found no record or failed means the next
/// attempt shares the device with this one, and that must be visible in the
/// failure rather than inferred from a serve that looks broken.
fn stop_scenario_services(world: &E2eWorld) -> String {
    world.isolated_root.as_ref().map_or_else(
        || "not attempted: scenario has no isolated root".to_owned(),
        |root| crate::stop_managed_services(root.path()),
    )
}

/// How many stalled serves one RUN may relaunch, in total.
///
/// A relaunch costs a full cold start: the readiness budget
/// (`E2E_SERVE_TIMEOUT_SECS`, 300s on the GPU lanes) behind the port-free and
/// VRAM-drain waits ahead of it — roughly eight minutes. Those lanes cap the
/// whole job at 35 minutes, and a job killed by that cap writes no
/// `platform.json`: the entire run is lost, including the service-log
/// diagnostics this step collects. Budgeting relaunches per RUN is what bounds
/// the added wall clock — a per-scenario cap still multiplies by however many
/// scenarios stall, which is exactly the double-stall case that would blow the
/// job. One rescue matches the observed failure (a single scenario stalling in a
/// run); a run that stalls twice is not one relaunch away from healthy, and is
/// worth more as a report than as an unfinished job.
fn relaunch_budget() -> &'static AtomicUsize {
    static BUDGET: LazyLock<AtomicUsize> = LazyLock::new(|| {
        AtomicUsize::new(
            std::env::var("E2E_SERVE_RELAUNCH_BUDGET")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(1),
        )
    });
    &BUDGET
}

/// Take one relaunch from the run's budget, reporting whether one was left.
fn claim_relaunch() -> bool {
    relaunch_budget()
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
            left.checked_sub(1)
        })
        .is_ok()
}

#[given("a model is being served on GPU")]
async fn setup_gpu_model(world: &mut E2eWorld) {
    // Serve by the canonical HuggingFace ID (not the `qwen2.5` alias) with an
    // explicit engine matching this host. This step is a *precondition* for
    // scenarios that test inference/chat behavior, so it must not fail for
    // reasons unrelated to what those scenarios assert. Serving the alias would
    // trip EAI-7219 (alias resolution) and sink every downstream scenario for a
    // bug they aren't testing; a fixed engine would false-fail on a host that
    // can't run it. Alias resolution and engine selection have their own
    // dedicated scenarios in model_serving.feature.
    // Free the shared serve port first so a prior scenario's leaked server can't
    // linger on the GPU and oversubscribe it (which otherwise piles up serves
    // until the job times out).
    let (model, engine, ready_substr) = host_serve_target();
    let timeout_secs = serve_timeout_for(world);
    let mut diagnostics = Vec::new();
    // A managed vLLM launch can return `status: starting` without publishing the
    // model within this scenario's readiness budget. Qwen3.5 has both timed out
    // and served successfully in the same MI300X run, so preserve that coverage
    // and allow one clean relaunch rather than replacing the model fixture.
    //
    // Only a scenario that is expected to PASS gets that relaunch. Where the
    // matrix already declares a known bug, the run's failure is the expected
    // outcome and the deliberately shortened `serve_timeout_secs` says to fail
    // fast — a second cold start there buys no signal and spends minutes of
    // serial GPU time (plus another engine load) that the scenarios which do
    // carry a result have to wait behind. The relaunch is additionally drawn
    // from a budget shared by the whole run (see `relaunch_budget`), so several
    // stalling scenarios cannot together push the job past its timeout.
    //
    // A failing launch (rc != 0) goes down the same path as a stall rather than
    // asserting out: the case worth retrying most is vLLM's own free-memory
    // check rejecting a device the previous attempt has not finished releasing,
    // and that one exits non-zero. It also gains the service-log tail below,
    // which an assert would have skipped.
    let attempts = if world.expect_xfail { 1 } else { 2 };
    let mut made = 0;
    for attempt in 1..=attempts {
        made = attempt;
        let device_state = ensure_serve_port_free().await;
        let (stdout, stderr, rc) =
            crate::run_rocm(world, &["serve", model, "--engine", engine, "--managed"]);
        if rc == 0 && model_is_ready(&MODELS_URL, Some(ready_substr), timeout_secs).await {
            world.endpoint = Some(SERVE_BASE_URL.to_string());
            world.model_name = Some(model.to_string());
            return;
        }
        // `serve_failure_evidence` reads and copies out the log BEFORE it stops
        // this attempt's stalled service, which matters twice over here. The tail
        // then reflects what the engine wrote on its own rather than anything the
        // stop provokes; and the stop itself is load-bearing for the RETRY. A vLLM
        // still in engine init has not bound the serve port yet, so the port kill
        // in `ensure_serve_port_free` cannot see it — it would survive into the
        // next attempt, hold its fraction-of-device memory reservation, and guarantee
        // the relaunch dies on vLLM's free-memory check. Going through `rocm
        // services stop` is what actually clears it: that path signals the whole
        // process tree (the EngineCore worker pins the allocation, not the parent)
        // and escalates past the grace period. Without this the retry is not a
        // retry — it is a second serve competing with the first, which is why its
        // outcome is quoted: a stop that found no record (the launch had not yet
        // written one) or failed means the next attempt did NOT get a clean
        // device, and the failure says so instead of looking like a broken serve.
        diagnostics.push(serve_failure_evidence(
            world,
            &format!("attempt {attempt} of {attempts} (rc={rc})"),
            &device_state,
            &stdout,
            &stderr,
        ));
        if attempt == attempts {
            break;
        }
        if !claim_relaunch() {
            diagnostics.push(
                "relaunch not attempted: this run's serve-relaunch budget is spent \
                 (see E2E_SERVE_RELAUNCH_BUDGET). A second cold start here risks the \
                 job timeout, which would kill the run before it writes any report."
                    .to_owned(),
            );
            break;
        }
    }
    let models_url = MODELS_URL.as_str();
    panic!(
        "endpoint {models_url} did not serve model {ready_substr} after {made} attempt(s) of {timeout_secs}s each:\n{}",
        diagnostics.join("\n\n")
    );
}

#[given("a GGUF model is being served on lemonade")]
async fn setup_lemonade_model(world: &mut E2eWorld) {
    // Lemonade serves GGUF models via its bundled llama.cpp backend, so use a
    // GGUF model (not the safetensors Qwen2.5) served explicitly on lemonade —
    // the parallel of setup_gpu_model's vLLM path, giving both engines their own
    // serve+inference coverage. Qwen3-0.6B-GGUF is the smallest lemonade recipe.
    let model = "Qwen3-0.6B-GGUF";
    // Wait for this lemonade model specifically (see setup_gpu_model): guards
    // against a leaked serve on the shared port. "Qwen3-0.6B" is the distinctive
    // substring (the endpoint reports it as e.g. Qwen3-0.6B-Q4_0.gguf).
    serve_and_wait(
        world,
        &["serve", model, "--engine", "lemonade", "--managed"],
        model,
        "Qwen3-0.6B",
    )
    .await;
}

#[given("a canonical Hugging Face GGUF checkpoint is being served on lemonade")]
async fn setup_lemonade_hf_checkpoint_model(world: &mut E2eWorld) {
    // Forces the `owner/repo:variant` direct-serve path (`serve_hf_checkpoint`,
    // EAI-8026) instead of the short-recipe-name router that `setup_lemonade_model`
    // exercises. Same underlying checkpoint (unsloth/Qwen3-0.6B-GGUF, Q4_0), so the
    // GGUF is already warm in the HF cache when both scenarios run in one job.
    //
    // This is the path #260 reports: on Strix Halo Windows the launch exits 0 and
    // the endpoint never answers. `serve_and_wait` is what makes that outcome
    // explain itself — the plain readiness poll this used to call reported the
    // timeout and discarded everything that could say why.
    let model = "unsloth/Qwen3-0.6B-GGUF:Q4_0";
    serve_and_wait(
        world,
        &["serve", model, "--engine", "lemonade", "--managed"],
        model,
        "Qwen3-0.6B",
    )
    .await;
}

#[given("a large model is being served on GPU")]
async fn setup_large_gpu_model(world: &mut E2eWorld) {
    // Large-model nightly coverage follows the host's serving path. MI300X uses
    // the dense BF16 model through vLLM; Strix Halo uses the hardware-verified
    // UD-Q4_K_XL GGUF checkpoint through Lemonade. The scenario's @serve-timeout
    // tag widens both the readiness poll and heavy-model inference timeout. For
    // vLLM it also raises ROCM_CLI_VLLM_READY_TIMEOUT_SECS (see isolate_cmd),
    // preventing the CLI's default readiness cap from terminating the server
    // mid-load.
    let (model, engine, ready_substr) =
        if e2e_cucumber::capability::host_capability().effective_serve_engine == "lemonade" {
            // ready_substr is the base name WITHOUT the quant. lemonade serves
            // this explicit `owner/repo:variant` checkpoint under its verbatim
            // ref, so the `/v1/models` id keeps the `UD-Q4_K_XL` tag — but matching
            // only the quant-free base keeps the readiness check robust to any
            // future id-normalization on the serve path (e.g. the quant rewriting
            // lemonade applies to shorthand refs like `Qwen3-0.6B-GGUF`).
            (
                "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL",
                "lemonade",
                "Qwen3.6-35B-A3B",
            )
        } else {
            ("Qwen/Qwen3.6-27B", "vllm", "Qwen3.6-27B")
        };
    serve_and_wait(
        world,
        &["serve", model, "--engine", engine, "--managed"],
        model,
        ready_substr,
    )
    .await;
}

#[given("a model is served in the background")]
async fn setup_background_model(world: &mut E2eWorld) {
    // This precondition backs behavioural chat scenarios (tool-defs accepted,
    // end-to-end reply) whose When/Then talk to the served endpoint over HTTP and
    // never assert on real generation — only that a served model answers. So it
    // doesn't need a real GPU serve: on a host WITH an AMD GPU we still exercise
    // the real `rocm serve --managed` path (extra real coverage where hardware
    // exists), but on a no-GPU host we back it with the in-process MockServer +
    // a planted managed-service record. This lets the scenarios drop
    // `@requires-gpu` and run on the GitHub-hosted mock lane every PR, with no
    // coverage loss. Real inference stays covered by the `@requires-gpu`
    // serve-*-inference scenarios.
    if e2e_cucumber::capability::host_capability().has_amd_gpu {
        setup_gpu_model(world).await;
    } else {
        let mock = MockServer::start("TestModel/E2E-1B").await;
        world.endpoint = Some(mock.base_url());
        world.model_name = Some("TestModel/E2E-1B".to_string());
        world.mock = Some(mock);
        world.register_mock_service();
    }
}

#[given("the served model has been detected")]
async fn setup_model_detected(world: &mut E2eWorld) {
    let (stdout, _, _) = crate::run_rocm(world, &["services", "list"]);
    let model = world.model_name.as_deref().unwrap_or("");
    assert!(
        stdout.contains(model),
        "model {model} not found in services:\n{stdout}"
    );
}

// ── When ───────────────────────────────────────────────────────────

#[when("the user serves a model using its short name")]
async fn user_serves_short_name(world: &mut E2eWorld) {
    let (stdout, _, _) = crate::run_rocm(world, &["serve", "qwen2.5", "--engine", "vllm"]);
    world.cli_output = Some(stdout);
}

#[when("the user serves the same short name with different engines")]
async fn user_serves_multiple_engines(world: &mut E2eWorld) {
    let mut outputs = Vec::new();
    for engine in ["lemonade", "vllm"] {
        let (stdout, _, _) = crate::run_rocm(world, &["serve", "qwen2.5", "--engine", engine]);
        outputs.push(stdout);
    }
    world.cli_outputs = Some(outputs);
}

#[when("the user lists running services")]
async fn user_lists_services(world: &mut E2eWorld) {
    let (stdout, _, _) = crate::run_rocm(world, &["services", "list"]);
    world.cli_output = Some(stdout);
}

#[when("the user serves a model without specifying an engine")]
async fn user_serves_default_engine(world: &mut E2eWorld) {
    // This scenario tests automatic *engine selection* — a platform-agnostic
    // behaviour. Omit `--engine` so the CLI picks the engine itself (the behaviour
    // under test), and serve by canonical ID (not the `qwen2.5` alias, which would
    // also depend on EAI-7219).
    //
    // Selection is RECIPE-driven, not platform-driven: `rocm serve <model>` with
    // no `--engine` resolves the request to the recipe's preferred model+engine,
    // which may differ from what was requested (e.g. Qwen2.5-1.5B → a GGUF recipe
    // on lemonade). So the readiness wait must key on the model the CLI ACTUALLY
    // resolved (parsed from the serve plan), not the requested one — hardcoding the
    // requested model made this time out on hosts where the recipe resolved to a
    // different model, for reasons unrelated to engine selection.
    //
    // Uses `default_engine_serve_target` (a lemonade-preferred model), NOT
    // `host_serve_target`: the model here drives engine resolution, and the xfail
    // matrix requires this to resolve to lemonade on Instinct (EAI-7052). See that
    // fn's doc comment.
    let model = default_engine_serve_target();
    let device_state = ensure_serve_port_free().await;
    let (stdout, stderr, rc) = crate::run_rocm(world, &["serve", model, "--managed"]);
    // The model the CLI resolved (what actually gets served) can differ from the
    // requested id, so downstream reachability/readiness checks must look for the
    // resolved model on the shared port; fall back to the requested id.
    let served = resolved_model(&stdout).unwrap_or(model).to_string();
    let ready_substr = ready_substr_for(&served).to_string();
    world.endpoint = Some(SERVE_BASE_URL.to_string());
    world.model_name = Some(served);
    // Both streams and the rc are carried on the World so whichever Then step
    // fires can explain itself: the scenarios behind this step assert on the
    // engine the plan named and on reachability, and each of those reports the
    // serve output when it fails.
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
    // A non-zero rc is deliberately NOT failed here. This is a `When`: the CLI
    // still printed the plan line the scenario is about, so failing on the exit
    // code would pre-empt the Then step that names the actual disagreement with a
    // blunter message. The other outcome — an exit-0 serve whose endpoint never
    // answers — has no Then step to catch it, so this step reports it, with the
    // same evidence a GPU serve collects rather than the bare timeout line that
    // left #260 undiagnosable.
    //
    // Wait for THE RESOLVED model specifically (not just any 200) — the shared port
    // 11435 may still answer from a prior scenario's leaked serve, and a
    // model-agnostic wait would then proceed against the wrong server.
    let timeout_secs = serve_timeout_for(world);
    if rc == 0 && !model_is_ready(&MODELS_URL, Some(&ready_substr), timeout_secs).await {
        let headline = stall_headline(
            &format!("`rocm serve {model} --managed`"),
            &ready_substr,
            timeout_secs,
        );
        panic!(
            "{}",
            serve_failure_evidence(
                world,
                &headline,
                &device_state,
                world.cli_output.as_deref().unwrap_or_default(),
                world.cli_stderr.as_deref().unwrap_or_default(),
            )
        );
    }
}

#[when("the user serves a vLLM-capable model without specifying an engine")]
async fn user_serves_vllm_capable_default(world: &mut E2eWorld) {
    // Use a vLLM-capable (safetensors) model so the GPU-family default can apply;
    // a GGUF-only model would legitimately fall through to lemonade regardless of
    // platform. Qwen2.5-0.5B is the smallest vLLM-preferred catalog entry. Omit
    // `--engine` so the CLI's own default selection is what's exercised.
    ensure_serve_port_free().await;
    let (stdout, stderr, rc) =
        crate::run_rocm(world, &["serve", "Qwen/Qwen2.5-0.5B-Instruct", "--managed"]);
    world.cli_output = Some(stdout);
    // See the note in user_serves_default_engine: the rc is asserted later, so
    // stderr must be carried along to explain a non-zero one.
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[given("Lemonade preparation cannot complete")]
async fn lemonade_preparation_cannot_complete(world: &mut E2eWorld) {
    world.command_env.push((
        "ROCM_E2E_LEMONADE_BACKEND_INSTALL_FAILURE",
        "repeated".into(),
    ));
}

#[when("the user serves a model with Lemonade")]
async fn user_serves_with_failing_lemonade_preparation(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm_with_scenario_env(
        world,
        &[
            "serve",
            "Qwen3-0.6B-GGUF",
            "--engine",
            "lemonade",
            "--managed",
        ],
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[when("the user sends a chat completion request")]
async fn user_sends_completion(world: &mut E2eWorld) {
    crate::send_chat(world).await;
}

/// Serve under the GPU-required default (no `--device`), pinning the engine to the
/// host's effective serve engine so the serve reaches GPU enforcement rather than
/// tripping on engine selection.
#[when("the user serves a model under the GPU-required default")]
async fn user_serves_gpu_required(world: &mut E2eWorld) {
    let (model, engine, _) = host_serve_target();
    let (stdout, stderr, rc) = crate::run_rocm(world, &["serve", model, "--engine", engine]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

/// Serve with every GPU hidden from the CLI and engine via an empty
/// `HIP_VISIBLE_DEVICES`, so a GPU host presents as having no usable device.
#[when("the user serves a model with every GPU masked from view")]
async fn user_serves_with_masked_gpus(world: &mut E2eWorld) {
    let (model, engine, _) = host_serve_target();
    let (stdout, stderr, rc) = crate::run_rocm_with_env(
        world,
        &["serve", model, "--engine", engine],
        &[("HIP_VISIBLE_DEVICES", "")],
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

/// Serve pinned to a GPU ordinal far beyond any real device count, so the request
/// names a device that does not exist on the host.
#[when("the user serves a model pinned to a GPU index that does not exist")]
async fn user_serves_absent_gpu_index(world: &mut E2eWorld) {
    let (model, engine, _) = host_serve_target();
    let (stdout, stderr, rc) =
        crate::run_rocm(world, &["serve", model, "--engine", engine, "--gpu", "99"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

/// Serve bound to a public (non-loopback) interface without the public-bind
/// opt-in. The bind-host validation is the first thing `serve` does — before any
/// engine, model, or GPU work — so this refusal is observable with no GPU (mock
/// lane). The model name is arbitrary; the refusal happens before it is resolved.
#[when("the user serves a model bound to a public interface without allowing public binding")]
async fn user_serves_public_bind_no_optin(world: &mut E2eWorld) {
    let (stdout, stderr, rc) =
        crate::run_rocm(world, &["serve", "some-model", "--host", "0.0.0.0"]);
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

/// Serve naming both a runtime and an environment. These are mutually exclusive
/// selectors, rejected during argument parsing before any engine or GPU work — so
/// the refusal is observable with no GPU (mock lane).
#[when("the user serves a model selecting both a runtime and an environment")]
async fn user_serves_runtime_and_env(world: &mut E2eWorld) {
    let (stdout, stderr, rc) = crate::run_rocm(
        world,
        &[
            "serve",
            "some-model",
            "--runtime-id",
            "therock-release:gfx942",
            "--env-id",
            "some-env",
        ],
    );
    world.cli_output = Some(stdout);
    world.cli_stderr = Some(stderr);
    world.cli_rc = Some(rc);
}

#[then("the user is told to allow public binding first")]
async fn assert_public_bind_message(world: &mut E2eWorld) {
    let output = serve_output(world);
    assert!(
        output.contains("--allow-public-bind"),
        "expected guidance to pass --allow-public-bind, got:\n{output}"
    );
}

#[then("the user is told the two selectors cannot be combined")]
async fn assert_selector_conflict_message(world: &mut E2eWorld) {
    let output = serve_output(world);
    // clap's conflict error names both flags; assert both appear so a reworded
    // message that still identifies the conflict keeps passing.
    assert!(
        output.contains("--runtime-id") && output.contains("--env-id"),
        "expected a conflict naming --runtime-id and --env-id, got:\n{output}"
    );
}

#[when("the CLI reports the service as ready")]
async fn when_cli_reports_ready(world: &mut E2eWorld) {
    // Read readiness from the CLI's own view (`services list`), not a direct
    // endpoint poll — this is the signal a user/automation waits on before
    // sending traffic (EAI-7333 concerns exactly this signal being trustworthy).
    let (stdout, _, _) = crate::run_rocm(world, &["services", "list"]);
    assert!(
        stdout.contains("ready"),
        "CLI does not report any service ready:\n{stdout}"
    );
    world.cli_output = Some(stdout);
}

// ── Then ───────────────────────────────────────────────────────────

#[then("serving stops after one automatic retry")]
async fn assert_lemonade_preparation_retry_is_bounded(world: &mut E2eWorld) {
    let output = serve_output(world);
    assert_ne!(
        world.cli_rc,
        Some(0),
        "serve unexpectedly succeeded:\n{output}"
    );
    assert_eq!(
        output.matches("retrying once").count(),
        1,
        "expected exactly one retry announcement:\n{output}"
    );
}

#[then("the user is told how to reinstall Lemonade and retry serving")]
async fn assert_lemonade_recovery_guidance(world: &mut E2eWorld) {
    let output = serve_output(world);
    assert!(
        output.contains("rocm engines install lemonade --reinstall"),
        "expected a forced-reinstall recovery command:\n{output}"
    );
    assert!(
        output.contains("retry `rocm serve`"),
        "expected guidance to retry serving:\n{output}"
    );
}

#[then("an inference request succeeds immediately")]
async fn assert_inference_succeeds_now(world: &mut E2eWorld) {
    // No extra wait: the CLI already reported ready, so inference must work now.
    // If this fails, readiness was a false positive (the gap tracked by EAI-7333).
    crate::send_chat(world).await;
    let resp = world.chat_response.as_ref().expect("no chat response");
    let content = resp["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(
        !content.is_empty(),
        "service reported ready but inference returned no content: {resp}"
    );
}

/// The CLI's combined stdout+stderr from the last recorded serve attempt.
fn serve_output(world: &E2eWorld) -> String {
    format!(
        "{}\n{}",
        world.cli_output.as_deref().unwrap_or(""),
        world.cli_stderr.as_deref().unwrap_or("")
    )
}

#[then("serving is refused before any engine starts")]
async fn assert_serve_refused(world: &mut E2eWorld) {
    // A non-zero exit is the observable "refused". The GPU-required pre-flight
    // rejects the launch before any engine is prepared or spawned, so this fails
    // fast rather than after a download or a late engine crash.
    let rc = world.cli_rc.expect("no serve rc recorded");
    assert!(
        rc != 0,
        "expected serving to be refused, but it exited 0:\n{}",
        serve_output(world)
    );
}

#[then("the user is told no AMD GPU was detected")]
async fn assert_no_gpu_message(world: &mut E2eWorld) {
    let output = serve_output(world);
    assert!(
        output.to_lowercase().contains("no usable amd gpu"),
        "expected a no-AMD-GPU message, got:\n{output}"
    );
}

#[then("the user is told that GPU index is unavailable")]
async fn assert_absent_index_message(world: &mut E2eWorld) {
    let output = serve_output(world).to_lowercase();
    // The named index must appear alongside an unavailability reason — whether the
    // CLI rejects it against the detected count ("out of range") or the engine's
    // probe rejects it ("not available") on a host where amd-smi can't count.
    assert!(
        output.contains("99")
            && (output.contains("out of range") || output.contains("not available")),
        "expected the absent GPU index to be reported unavailable, got:\n{}",
        serve_output(world)
    );
}

#[then("the output shows the full model name")]
async fn assert_full_model_name(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no CLI output");
    let resolved = resolved_model(output)
        .unwrap_or_else(|| panic!("no 'resolved model' in output:\n{output}"));
    assert!(
        resolved.contains('/'),
        "expected a fully qualified model name (org/model), got '{resolved}'\n\nfull output:\n{output}"
    );
}

#[then("all engines expand to the same full model name")]
async fn assert_consistent_expansion(world: &mut E2eWorld) {
    let outputs = world.cli_outputs.as_ref().expect("no CLI outputs");
    assert!(
        outputs.len() >= 2,
        "need at least 2 serve outputs to compare"
    );
    let mut resolved: Vec<String> = Vec::new();
    for (i, output) in outputs.iter().enumerate() {
        // A missing `resolved model:` line means the engine never got far enough
        // to expand the name (e.g. it errored before serving). Fail loudly rather
        // than defaulting to an empty string — otherwise two engines that both
        // fail would produce equal ("") values and pass this check vacuously.
        let Some(model) = resolved_model(output).map(str::to_string) else {
            panic!("engine #{i} produced no 'resolved model' line:\n{output}");
        };
        resolved.push(model);
    }
    let first = &resolved[0];
    for (i, r) in resolved.iter().enumerate().skip(1) {
        assert_eq!(
            r, first,
            "inconsistent model name expansion across engines: {resolved:?} (index {i})"
        );
    }
}

#[then("the service appears with the correct model name and connection details")]
async fn assert_service_in_list(world: &mut E2eWorld) {
    let (stdout, _, _) = crate::run_rocm(world, &["services", "list"]);
    let model = world.model_name.as_deref().unwrap_or("");
    assert!(
        stdout.to_lowercase().contains(&model.to_lowercase()),
        "model name not in services list:\n{stdout}"
    );
    assert!(
        stdout.contains("127.0.0.1"),
        "endpoint not in services list:\n{stdout}"
    );
}

#[then("the connection details match the actual server port")]
async fn assert_endpoint_port(world: &mut E2eWorld) {
    let mock = world.mock.as_ref().expect("no mock server running");
    let port = mock.port();
    let (stdout, _, _) = crate::run_rocm(world, &["services", "list"]);
    assert!(
        stdout.contains(&port.to_string()),
        "port {port} not found in services list:\n{stdout}"
    );
}

/// Extract the engine name from a serve plan's `engine: <name>` line.
fn selected_engine(output: &str) -> &str {
    let Some(engine) = output
        .lines()
        .find_map(|l| l.trim().strip_prefix("engine:"))
        .map(str::trim)
    else {
        panic!("no 'engine:' line in serve output:\n{output}");
    };
    engine
}

/// The model a serve plan actually resolved to (`resolved model: <id>`), which
/// can differ from the requested id — `rocm serve <model>` with no `--engine`
/// picks the recipe's preferred model+engine. `None` if the plan has no such line
/// (e.g. the serve errored before resolving).
fn resolved_model(output: &str) -> Option<&str> {
    output
        .lines()
        .find_map(|l| l.trim().strip_prefix("resolved model:"))
        .map(str::trim)
}

/// A distinctive substring of a model id that appears in the served endpoint's
/// `/v1/models` response, for [`model_is_ready`]'s containment check. Strips the
/// `org/` prefix and the `-GGUF` catalog marker so a resolved catalog id
/// (`Qwen3-4B-Instruct-2507-GGUF`) matches the concrete artifact the endpoint
/// reports (`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`) — both share the base
/// `Qwen3-4B-Instruct-2507`.
fn ready_substr_for(model_id: &str) -> &str {
    let base = model_id.rsplit('/').next().unwrap_or(model_id);
    base.strip_suffix("-GGUF")
        .or_else(|| base.strip_suffix("-gguf"))
        .unwrap_or(base)
}

#[then("an engine is selected automatically")]
async fn assert_engine_auto_selected(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no serve output");
    // Parse the actual engine the CLI chose from the `engine: <name>` plan line,
    // not just the presence of the word — and assert it is one of the supported
    // serving engines. Since #79 the only serving backends are lemonade and
    // vllm; auto-selection landing on a removed engine (pytorch/atom/sglang/
    // llama-cpp) is a regression this must catch.
    let engine = selected_engine(output);
    assert!(
        matches!(engine, "lemonade" | "vllm"),
        "auto-selected an unsupported engine '{engine}' (expected lemonade or vllm):\n{output}"
    );
}

#[then("vLLM is selected as the default engine")]
async fn assert_vllm_default(world: &mut E2eWorld) {
    let output = world.cli_output.as_ref().expect("no serve output");
    // The serve must have actually launched, not just printed a correct plan: a
    // non-zero rc after a good plan-print (e.g. the engine fails to start) would
    // otherwise go undetected since this scenario only inspected the plan line.
    let rc = world.cli_rc.expect("no serve rc recorded");
    assert!(
        rc == 0,
        "{}",
        e2e_cucumber::cli_failure_report(
            &["serve", "<default engine>", "--managed"],
            rc,
            output,
            world.cli_stderr.as_deref().unwrap_or(""),
        )
    );
    let engine = selected_engine(output);
    assert_eq!(
        engine, "vllm",
        "expected vLLM as the default engine on an Instinct GPU, got '{engine}':\n{output}"
    );
}

#[then("the model is reachable")]
async fn assert_model_reachable(world: &mut E2eWorld) {
    let endpoint = world.endpoint.as_ref().expect("no endpoint configured");
    let expected = world.model_name.as_deref().expect("no model name set");
    let url = format!("{endpoint}/models");
    let resp: serde_json::Value = reqwest::get(&url)
        .await
        .unwrap_or_else(|e| panic!("GET {url} failed: {e}"))
        .json()
        .await
        .unwrap_or_else(|e| panic!("GET {url} returned non-JSON: {e}"));
    let ids: Vec<&str> = resp["data"]
        .as_array()
        .map(|d| d.iter().filter_map(|m| m["id"].as_str()).collect())
        .unwrap_or_default();
    // Assert THIS scenario's model is the one listed — not merely "some model" —
    // so a leaked prior serve still answering on the shared port can't satisfy it.
    assert!(
        ids.iter().any(|id| model_ids_match(id, expected)),
        "endpoint {url} does not list the served model '{expected}'; got {ids:?}"
    );
}

#[then("the model responds to inference requests")]
async fn assert_endpoint_responds(world: &mut E2eWorld) {
    crate::send_chat(world).await;
    let resp = world.chat_response.as_ref().expect("no chat response");
    let content = resp["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "empty reply in chat response: {resp}");
    // Verify the reply came from THIS scenario's model, so a leaked prior serve
    // on the shared port can't answer in its place.
    let expected = world.model_name.as_deref().expect("no model name set");
    let resp_model = resp["model"].as_str().unwrap_or("");
    assert!(
        model_ids_match(resp_model, expected),
        "inference reply model '{resp_model}' does not identify the served '{expected}'"
    );
}

#[then("the response contains a model reply")]
async fn assert_response_has_reply(world: &mut E2eWorld) {
    let resp = world.chat_response.as_ref().expect("no chat response");
    let choices = resp["choices"].as_array();
    assert!(
        choices.is_some_and(|c| !c.is_empty()),
        "no choices in chat response: {resp}"
    );
    let content = resp["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "empty reply in chat response: {resp}");
}

#[then("the response identifies the correct model")]
async fn assert_response_model_correct(world: &mut E2eWorld) {
    let resp = world.chat_response.as_ref().expect("no chat response");
    let resp_model = resp["model"].as_str().unwrap_or("");
    let expected = world.model_name.as_deref().unwrap_or("");
    assert!(
        model_ids_match(resp_model, expected),
        "response model '{resp_model}' does not identify '{expected}'"
    );
}

/// Whether a chat response's `model` field identifies the model we served.
///
/// vLLM echoes the exact id we passed (`Qwen/Qwen3.5-0.8B`), so a plain
/// containment holds. Lemonade instead reports the concrete GGUF artifact it
/// loaded — e.g. serving `Qwen3-0.6B-GGUF` yields `Qwen3-0.6B-Q4_0.gguf` — so an
/// exact/containment check on the catalog name fails even though it IS the right
/// model. Compare on a normalized base (lowercased, `.gguf` and the `-gguf`
/// catalog suffix and quantization tokens like `-q4_0` stripped) and accept a
/// match in either direction.
fn model_ids_match(resp_model: &str, expected: &str) -> bool {
    e2e_cucumber::model_id::model_ids_match(resp_model, expected)
}
