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

async fn wait_for_endpoint(url: &str, timeout_secs: u64) {
    wait_for_model(url, None, timeout_secs).await;
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

#[given("a model is being served on GPU")]
async fn setup_gpu_model(world: &mut E2eWorld) {
    // Serve by the canonical HuggingFace ID (not the `qwen2.5` alias) with an
    // explicit engine. This step is a *precondition* for scenarios that test
    // inference/chat behavior, so it must not fail for reasons unrelated to what
    // those scenarios assert. Serving the alias would trip EAI-7219 (vLLM does
    // not resolve aliases) and sink every downstream scenario for a bug they
    // aren't testing. Alias resolution has its own dedicated scenarios in
    // model_serving.feature.
    let (stdout, _, rc) = crate::run_rocm(
        world,
        &[
            "serve",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "--engine",
            "vllm",
            "--managed",
        ],
    );
    assert!(rc == 0, "rocm serve failed:\n{stdout}");
    world.endpoint = Some("http://127.0.0.1:11435/v1".to_string());
    world.model_name = Some("Qwen/Qwen2.5-1.5B-Instruct".to_string());
    // Wait for THIS model specifically: the shared port 11435 may still be
    // answering from a prior scenario's leaked serve (isolated data dirs mean
    // this scenario can't stop it), so a plain readiness check could pass against
    // the wrong model. "Qwen2.5-1.5B" is the distinctive substring of the id.
    wait_for_model(
        "http://127.0.0.1:11435/v1/models",
        Some("Qwen2.5-1.5B"),
        serve_timeout_secs(),
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
        serve_timeout_secs(),
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
    // This scenario tests automatic *engine selection*, not alias resolution, so
    // serve by the canonical ID — using the `qwen2.5` alias would make the
    // scenario also depend on EAI-7219 being fixed. Omit `--engine` so the CLI
    // still picks the engine itself, which is the behavior under test.
    let (stdout, _, rc) =
        crate::run_rocm(world, &["serve", "Qwen/Qwen2.5-1.5B-Instruct", "--managed"]);
    world.cli_output = Some(stdout);
    world.cli_rc = Some(rc);
    world.endpoint = Some("http://127.0.0.1:11435/v1".to_string());
    world.model_name = Some("Qwen/Qwen2.5-1.5B-Instruct".to_string());
    if rc == 0 {
        wait_for_endpoint("http://127.0.0.1:11435/v1/models", serve_timeout_secs()).await;
    }
}

#[when("the user serves a vLLM-capable model without specifying an engine")]
async fn user_serves_vllm_capable_default(world: &mut E2eWorld) {
    // Use a vLLM-capable (safetensors) model so the GPU-family default can apply;
    // a GGUF-only model would legitimately fall through to lemonade regardless of
    // platform. Qwen2.5-0.5B is the smallest vLLM-preferred catalog entry. Omit
    // `--engine` so the CLI's own default selection is what's exercised.
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
    let resolved = output
        .lines()
        .find(|l| l.contains("resolved model:"))
        .and_then(|l| l.split(':').nth(1))
        .map_or_else(
            || panic!("no 'resolved model' in output:\n{output}"),
            str::trim,
        );
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
        let model = output
            .lines()
            .find(|l| l.contains("resolved model:"))
            .and_then(|l| l.split(':').nth(1))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| panic!("engine #{i} produced no 'resolved model' line:\n{output}"));
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
    output
        .lines()
        .find_map(|l| l.trim().strip_prefix("engine:"))
        .map(str::trim)
        .unwrap_or_else(|| panic!("no 'engine:' line in serve output:\n{output}"))
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
    let engine = selected_engine(output);
    assert_eq!(
        engine, "vllm",
        "expected vLLM as the default engine on an Instinct GPU, got '{engine}':\n{output}"
    );
}

#[then("the model is reachable")]
async fn assert_model_reachable(world: &mut E2eWorld) {
    let endpoint = world.endpoint.as_ref().expect("no endpoint configured");
    let url = format!("{endpoint}/models");
    let resp: serde_json::Value = reqwest::get(&url)
        .await
        .unwrap_or_else(|e| panic!("GET {url} failed: {e}"))
        .json()
        .await
        .unwrap_or_else(|e| panic!("GET {url} returned non-JSON: {e}"));
    let data = resp["data"].as_array();
    assert!(
        data.is_some_and(|d| !d.is_empty()),
        "no models at {url}: {resp}"
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
        resp_model.contains(expected),
        "response model '{resp_model}' does not match '{expected}'"
    );
}
