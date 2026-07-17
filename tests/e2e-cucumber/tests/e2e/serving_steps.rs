// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use std::time::{Duration, Instant};

use cucumber::{given, then, when};

use crate::E2eWorld;
use e2e_cucumber::mock_server::MockServer;

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
async fn wait_for_model(models_url: &str, expect_model: Option<&str>, timeout_secs: u64) {
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    while Instant::now() < deadline {
        if let Ok(resp) = reqwest::get(models_url).await
            && resp.status().is_success()
        {
            match expect_model {
                None => return,
                Some(model) => {
                    if let Ok(body) = resp.text().await
                        && body.contains(model)
                    {
                        return;
                    }
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
    match expect_model {
        Some(m) => panic!("endpoint {models_url} did not serve model {m} within {timeout_secs}s"),
        None => panic!("endpoint {models_url} not ready after {timeout_secs}s"),
    }
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
async fn ensure_serve_port_free() {
    // Always kill any listener on the shared port — NOT just one that already
    // answers /v1/models. A prior scenario's vLLM that is still LOADING holds the
    // port and GPU memory without yet serving /v1/models; if we only checked HTTP
    // readiness we'd start a second server, overcommit GPU memory (each asks for
    // 0.80), and the collision crashes a server → the next chat POST fails with
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
    // at startup and demands `gpu-memory-utilization` of the TOTAL — after a large
    // model (e.g. 27B ~54 GiB) the residue can drop free memory below that
    // request, so the next serve dies with "Free memory ... less than desired GPU
    // memory utilization" (engine core init failed). Wait for the device to
    // actually drain before returning.
    wait_for_free_vram().await;
}

/// Upper bound on the free-VRAM floor (MiB). Sized so the largest single
/// scenario model can allocate its `gpu-memory-utilization` share without
/// tripping vLLM's startup memory check: the suite's biggest model
/// (Qwen3.6-27B, ~54 GiB) plus vLLM's ~0.8-of-total KV reservation needs the
/// card mostly clear. Only a data-center GPU (MI300X, ~192 GiB) has this much,
/// so on smaller cards the floor is capped to a fraction of the device total
/// (see [`required_free_vram_mib`]) — otherwise the check could never pass.
const MAX_FREE_VRAM_FLOOR_MIB: u64 = 150_000;

/// The free-VRAM floor to wait for on this host: the model-sized ceiling, but
/// never more than 90% of the device's total VRAM. A hardcoded 150 GiB floor is
/// unreachable on a small card (e.g. Strix Halo's ~48 GiB unified VRAM), so
/// `wait_for_free_vram` would burn its full deadline on every serve scenario;
/// scaling to the device keeps the drain-check meaningful everywhere.
fn required_free_vram_mib(total_mib: u64) -> u64 {
    MAX_FREE_VRAM_FLOOR_MIB.min(total_mib / 100 * 90)
}

/// Best-effort: wait until the GPU reports enough free VRAM (see
/// [`required_free_vram_mib`]), so a just-killed serve's memory is actually
/// reclaimed before the next serve starts. Queries `amd-smi` then `rocm-smi`;
/// if neither is present (mock/local, no ROCm), returns immediately so non-GPU
/// runs are unaffected.
async fn wait_for_free_vram() {
    // No GPU tooling → nothing to wait on (mock/local). Probe once up front.
    let Some(total) = total_vram_mib() else {
        return;
    };
    let floor = required_free_vram_mib(total);
    let deadline = Instant::now() + Duration::from_mins(2);
    loop {
        match free_vram_mib() {
            Some(free) if free >= floor => return,
            _ if Instant::now() >= deadline => return,
            _ => tokio::time::sleep(Duration::from_secs(3)).await,
        }
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
fn host_serve_target() -> (&'static str, &'static str, &'static str) {
    if e2e_cucumber::capability::host_capability().effective_serve_engine == "lemonade" {
        // GGUF via lemonade's llama.cpp backend; endpoint reports e.g.
        // Qwen3-0.6B-Q4_0.gguf, so "Qwen3-0.6B" is the distinctive substring.
        ("Qwen3-0.6B-GGUF", "lemonade", "Qwen3-0.6B")
    } else {
        // Safetensors via vLLM; "Qwen2.5-0.5B" is the distinctive substring.
        // Smallest vLLM-preferred catalog entry — see user_serves_vllm_capable_default.
        ("Qwen/Qwen2.5-0.5B-Instruct", "vllm", "Qwen2.5-0.5B")
    }
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
    ensure_serve_port_free().await;
    let (stdout, _, rc) =
        crate::run_rocm(world, &["serve", model, "--engine", engine, "--managed"]);
    assert!(rc == 0, "rocm serve failed:\n{stdout}");
    world.endpoint = Some("http://127.0.0.1:11435/v1".to_string());
    world.model_name = Some(model.to_string());
    // Wait for THIS model specifically: the shared port 11435 may still be
    // answering from a prior scenario's leaked serve (isolated data dirs mean
    // this scenario can't stop it), so a plain readiness check could pass against
    // the wrong model.
    wait_for_model(
        "http://127.0.0.1:11435/v1/models",
        Some(ready_substr),
        serve_timeout_for(world),
    )
    .await;
}

#[given("a GGUF model is being served on lemonade")]
async fn setup_lemonade_model(world: &mut E2eWorld) {
    // Lemonade serves GGUF models via its bundled llama.cpp backend, so use a
    // GGUF model (not the safetensors Qwen2.5) served explicitly on lemonade —
    // the parallel of setup_gpu_model's vLLM path, giving both engines their own
    // serve+inference coverage. Qwen3-0.6B-GGUF is the smallest lemonade recipe.
    let model = "Qwen3-0.6B-GGUF";
    ensure_serve_port_free().await;
    let (stdout, _, rc) = crate::run_rocm(
        world,
        &["serve", model, "--engine", "lemonade", "--managed"],
    );
    assert!(rc == 0, "rocm serve failed:\n{stdout}");
    world.endpoint = Some("http://127.0.0.1:11435/v1".to_string());
    world.model_name = Some(model.to_string());
    // Wait for this lemonade model specifically (see setup_gpu_model): guards
    // against a leaked serve on the shared port. "Qwen3-0.6B" is the distinctive
    // substring (the endpoint reports it as e.g. Qwen3-0.6B-Q4_0.gguf).
    wait_for_model(
        "http://127.0.0.1:11435/v1/models",
        Some("Qwen3-0.6B"),
        serve_timeout_for(world),
    )
    .await;
}

#[given("a large model is being served on GPU")]
async fn setup_large_gpu_model(world: &mut E2eWorld) {
    // Large-model coverage (dogfooding W9): a big vLLM model, served explicitly
    // on vLLM (the scenario is @requires-engine:vllm, so this only runs on an
    // Instinct host). Qwen/Qwen3.6-27B is the catalog's Instinct recommendation
    // (BF16, ~54 GiB, fits 1x MI300X). The scenario's @serve-timeout tag both
    // widens the harness poll AND raises the CLI's own vLLM readiness cap via
    // ROCM_CLI_VLLM_READY_TIMEOUT_SECS (see isolate_cmd) — the 5-min default
    // would SIGTERM the server mid-load. Weights are pre-seeded in the shared HF
    // cache so this is load-only, not a 54 GiB download.
    let model = "Qwen/Qwen3.6-27B";
    ensure_serve_port_free().await;
    let (stdout, _, rc) =
        crate::run_rocm(world, &["serve", model, "--engine", "vllm", "--managed"]);
    assert!(rc == 0, "rocm serve failed:\n{stdout}");
    world.endpoint = Some("http://127.0.0.1:11435/v1".to_string());
    world.model_name = Some(model.to_string());
    wait_for_model(
        "http://127.0.0.1:11435/v1/models",
        Some("Qwen3.6-27B"),
        serve_timeout_for(world),
    )
    .await;
}

#[given("a model is served in the background")]
async fn setup_background_model(world: &mut E2eWorld) {
    setup_gpu_model(world).await;
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
    let model = host_serve_target().0;
    ensure_serve_port_free().await;
    let (stdout, _, rc) = crate::run_rocm(world, &["serve", model, "--managed"]);
    // The model the CLI resolved (what actually gets served) can differ from the
    // requested id, so downstream reachability/readiness checks must look for the
    // resolved model on the shared port; fall back to the requested id.
    let served = resolved_model(&stdout).unwrap_or(model).to_string();
    let ready_substr = ready_substr_for(&served).to_string();
    world.cli_output = Some(stdout);
    world.cli_rc = Some(rc);
    world.endpoint = Some("http://127.0.0.1:11435/v1".to_string());
    world.model_name = Some(served);
    if rc == 0 {
        // Wait for THE RESOLVED model specifically (not just any 200) — the shared
        // port 11435 may still answer from a prior scenario's leaked serve, and a
        // model-agnostic wait would then proceed against the wrong server.
        wait_for_model(
            "http://127.0.0.1:11435/v1/models",
            Some(&ready_substr),
            serve_timeout_for(world),
        )
        .await;
    }
}

#[when("the user serves a vLLM-capable model without specifying an engine")]
async fn user_serves_vllm_capable_default(world: &mut E2eWorld) {
    // Use a vLLM-capable (safetensors) model so the GPU-family default can apply;
    // a GGUF-only model would legitimately fall through to lemonade regardless of
    // platform. Qwen2.5-0.5B is the smallest vLLM-preferred catalog entry. Omit
    // `--engine` so the CLI's own default selection is what's exercised.
    ensure_serve_port_free().await;
    let (stdout, _, rc) =
        crate::run_rocm(world, &["serve", "Qwen/Qwen2.5-0.5B-Instruct", "--managed"]);
    world.cli_output = Some(stdout);
    world.cli_rc = Some(rc);
}

#[when("the user sends a chat completion request")]
async fn user_sends_completion(world: &mut E2eWorld) {
    crate::send_chat(world).await;
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
/// `/v1/models` response, for `wait_for_model`'s containment check. Strips the
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
    assert!(rc == 0, "rocm serve failed (rc={rc}):\n{output}");
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
/// vLLM echoes the exact id we passed (`Qwen/Qwen2.5-0.5B-Instruct`), so a plain
/// containment holds. Lemonade instead reports the concrete GGUF artifact it
/// loaded — e.g. serving `Qwen3-0.6B-GGUF` yields `Qwen3-0.6B-Q4_0.gguf` — so an
/// exact/containment check on the catalog name fails even though it IS the right
/// model. Compare on a normalized base (lowercased, `.gguf` and the `-gguf`
/// catalog suffix and quantization tokens like `-q4_0` stripped) and accept a
/// match in either direction.
fn model_ids_match(resp_model: &str, expected: &str) -> bool {
    let a = normalize_model_id(resp_model);
    let b = normalize_model_id(expected);
    if a.is_empty() || b.is_empty() {
        return false;
    }
    a.contains(&b) || b.contains(&a)
}

fn normalize_model_id(id: &str) -> String {
    let mut s = id.to_ascii_lowercase();
    if let Some(stripped) = s.strip_suffix(".gguf") {
        s = stripped.to_owned();
    }
    // Drop the catalog `-gguf` marker and common quantization suffixes so the
    // catalog name and the concrete artifact reduce to the same base.
    s = s.replace("-gguf", "");
    for q in [
        "-q4_0", "-q4_k_m", "-q4_k_s", "-q5_k_m", "-q8_0", "-f16", "-fp16",
    ] {
        s = s.replace(q, "");
    }
    s.trim_matches(['-', '_', '.']).to_owned()
}
