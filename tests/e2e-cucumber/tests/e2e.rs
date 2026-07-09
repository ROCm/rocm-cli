// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

// Cucumber step functions share one uniform signature — `async fn(world: &mut
// E2eWorld, ...)` — so the `#[given/when/then]` macros can register them the
// same way. Many steps neither `.await` nor mutate the world; allow both rather
// than splitting the step API into sync/async and mut/non-mut variants.
#![allow(clippy::unused_async, clippy::needless_pass_by_ref_mut)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use cucumber::{World as _, WriterExt as _};
use e2e_cucumber::mock_server::MockServer;

mod e2e {
    pub mod chat_steps;
    pub mod examine_steps;
    pub mod runtime_steps;
    pub mod serving_steps;
}

// ── World ──────────────────────────────────────────────────────────

#[derive(Debug, cucumber::World)]
pub struct E2eWorld {
    pub mock: Option<MockServer>,
    pub endpoint: Option<String>,
    pub model_name: Option<String>,
    pub discovered_model: Option<String>,
    pub chat_response: Option<serde_json::Value>,
    pub cli_output: Option<String>,
    pub cli_outputs: Option<Vec<String>>,
    pub cli_stderr: Option<String>,
    pub cli_rc: Option<i32>,
    pub isolated_root: Option<PathBuf>,
}

/// Per-World counter so each scenario gets its own isolated root. Keying only on
/// PID made all concurrent scenarios share one config/data/cache tree, and the
/// first scenario's `Drop` would `remove_dir_all` it out from under the others.
static WORLD_SEQ: AtomicU64 = AtomicU64::new(0);

impl Default for E2eWorld {
    fn default() -> Self {
        let seq = WORLD_SEQ.fetch_add(1, Ordering::Relaxed);
        // CodeQL flags `temp_dir()` (which reads TMPDIR) as a tainted path source.
        // This is test-only code: the base is the OS temp dir and every joined
        // segment is a program-controlled constant (pid, an atomic counter, and
        // fixed names), so there is no attacker-controlled traversal.
        let tmp = std::env::temp_dir().join(format!("rocm-e2e-{}-{}", std::process::id(), seq));
        // codeql[rust/path-injection]
        std::fs::create_dir_all(tmp.join("config")).ok();
        // codeql[rust/path-injection]
        std::fs::create_dir_all(tmp.join("data")).ok();
        // codeql[rust/path-injection]
        std::fs::create_dir_all(tmp.join("cache")).ok();

        Self {
            mock: None,
            endpoint: None,
            model_name: None,
            discovered_model: None,
            chat_response: None,
            cli_output: None,
            cli_outputs: None,
            cli_stderr: None,
            cli_rc: None,
            isolated_root: Some(tmp),
        }
    }
}

impl E2eWorld {
    pub fn isolate_cmd(&self, cmd: &mut std::process::Command) {
        if let Some(root) = &self.isolated_root {
            cmd.env("ROCM_CLI_CONFIG_DIR", root.join("config"));
            cmd.env("ROCM_CLI_DATA_DIR", root.join("data"));
            cmd.env("ROCM_CLI_CACHE_DIR", root.join("cache"));
        }
    }

    /// Register the running mock server with the CLI by writing a managed-service
    /// record into the isolated services directory (`<data>/services/`), exactly
    /// as `rocm serve --managed` would. This lets `rocm services list` and the
    /// `local` chat provider discover the mock — so scenarios exercise the real
    /// binary instead of asserting against the test's own helper. Black-box: the
    /// record is plain JSON matching the CLI's on-disk schema, not a typed import
    /// from the rocm-cli crates.
    pub fn register_mock_service(&self) {
        let root = self.isolated_root.as_ref().expect("no isolated root");
        let mock = self.mock.as_ref().expect("no mock server running");
        let model = self.model_name.as_deref().expect("no model name set");
        let port = mock.port();
        let services = root.join("data").join("services");
        std::fs::create_dir_all(&services).expect("failed to create services dir");

        // The CLI only extracts host:port from `endpoint_url` and appends
        // `/v1/models` for its readiness probe, which the mock serves.
        let record = serde_json::json!({
            "service_id": "e2e-mock",
            "engine": "vllm",
            "model_ref": model,
            "canonical_model_id": model,
            "host": "127.0.0.1",
            "port": port,
            "endpoint_url": format!("http://127.0.0.1:{port}/v1"),
            "mode": "managed",
            "status": "ready",
            "supervisor_pid": 0,
            "engine_pid": null,
            "runtime_id": null,
            "env_id": null,
            "device_policy": null,
            "gpu_indices": [],
            "engine_recipe_json": null,
            "restart_count": 0,
            "last_restart_unix_ms": null,
            "manifest_path": services.join("e2e-mock.json"),
            "log_path": services.join("e2e-mock.log"),
            "engine_state_path": services.join("e2e-mock.state.json"),
            "created_at_unix_ms": 1_700_000_000_000_u64,
        });
        std::fs::write(
            services.join("e2e-mock.json"),
            serde_json::to_vec_pretty(&record).expect("failed to serialize record"),
        )
        .expect("failed to write service record");
    }
}

impl Drop for E2eWorld {
    fn drop(&mut self) {
        if let Some(mock) = self.mock.take() {
            mock.stop();
        }
        if let Some(root) = self.isolated_root.take() {
            std::fs::remove_dir_all(root).ok();
        }
    }
}

// ── Shared helpers ─────────────────────────────────────────────────

pub fn rocm_binary() -> String {
    std::env::var("ROCM_CLI_BINARY").unwrap_or_else(|_| "rocm".to_string())
}

pub async fn send_chat(world: &mut E2eWorld) {
    let endpoint = world.endpoint.as_ref().expect("no endpoint configured");

    let models_url = format!("{endpoint}/models");
    let resp: serde_json::Value = reqwest::get(&models_url)
        .await
        .unwrap_or_else(|e| panic!("GET {models_url} failed: {e}"))
        .json()
        .await
        .unwrap_or_else(|e| panic!("GET {models_url} returned non-JSON: {e}"));
    let model = resp["data"][0]["id"]
        .as_str()
        .unwrap_or_else(|| panic!("no model id in response: {resp}"))
        .to_string();
    world.discovered_model = Some(model.clone());

    let chat_url = format!("{endpoint}/chat/completions");
    let client = reqwest::Client::new();
    let chat_resp: serde_json::Value = client
        .post(&chat_url)
        .json(&serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}]
        }))
        .send()
        .await
        .unwrap_or_else(|e| panic!("POST {chat_url} failed: {e}"))
        .json()
        .await
        .unwrap_or_else(|e| panic!("POST {chat_url} returned non-JSON: {e}"));
    world.chat_response = Some(chat_resp);
}

// ── Runner ─────────────────────────────────────────────────────────

fn results_dir() -> PathBuf {
    let dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/results"));
    std::fs::create_dir_all(&dir).expect("failed to create results directory");
    dir
}

#[tokio::main]
async fn main() {
    use cucumber::writer::{self, Stats as _};

    let dir = results_dir();
    let json_file =
        std::fs::File::create(dir.join("report.json")).expect("failed to create report.json");
    let junit_file =
        std::fs::File::create(dir.join("junit.xml")).expect("failed to create junit.xml");

    // `.run()` records failures into the writers but never sets a non-zero exit
    // code — only the returned writer knows. Capture it (summarized, so it tracks
    // failed/parsing/hook counts) and exit non-zero below if anything failed, so
    // CI actually gates on the result.
    // `summarized()` must wrap the stdout writer (only `Basic` accepts the
    // summary's arbitrary string writes); the file writers are teed in with
    // `discard_stats_writes()` so the `Tee` bound (both sides implement `Stats`)
    // is satisfied — `Tee`'s counts then come from the summarized side.
    let summary = E2eWorld::cucumber()
        .with_writer(
            writer::Basic::raw(std::io::stdout(), writer::Coloring::Auto, 1)
                .summarized()
                .tee(writer::Json::new(json_file).discard_stats_writes())
                .tee(writer::JUnit::new(junit_file, 0).discard_stats_writes())
                .normalized(),
        )
        .run(concat!(env!("CARGO_MANIFEST_DIR"), "/features/"))
        .await;

    // Generate the HTML report before exiting so the artifact still uploads on
    // failure.
    e2e_cucumber::report::generate(&dir.join("report.json"), &dir.join("report.html"))
        .expect("failed to generate HTML report");

    eprintln!("Report: {}/report.html", dir.display());

    if summary.execution_has_failed() {
        eprintln!(
            "E2E run failed: {} failed step(s), {} parsing error(s), {} hook error(s)",
            summary.failed_steps(),
            summary.parsing_errors(),
            summary.hook_errors(),
        );
        std::process::exit(1);
    }
}
